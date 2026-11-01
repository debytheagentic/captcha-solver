"""Tests for the Turnstile hybrid offload router (`turnstile/offload.py`).

Router contract: exactly one local seat; when it is taken and the secondary pool is
healthy, solve via the offload HTTP API; on any pool failure degrade to the local
queue without raising. The pool is stubbed with stdlib `http.server` (no new deps).
"""
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from turnstile import offload
from turnstile import solve as tsolve
import server as server_mod
from server import app

client = TestClient(app)

# A dead port (nothing listens on 1) -> instant connection refused, no DNS.
DEAD_URL = "http://127.0.0.1:1/solve"


class _Req:
    """Minimal stand-in for server.SolveRequest (only the offload payload fields)."""
    def __init__(self, **kw):
        self.sitekey = "0xSITEKEY"
        self.url = "https://target.example.com/app"
        self.action = None
        self.cdata = None
        self.real_page = False
        self.proxy = None
        self.timeout_s = 60
        self.__dict__.update(kw)


class _StubPool:
    """Stdlib HTTP stand-in for the sctg-api-pool (:8878) — counts offload calls."""

    def __init__(self, solve_status=200, solve_body=None, health_status=200, solve_abort=False):
        self.solve_status = solve_status
        self.health_status = health_status
        self.solve_abort = solve_abort      # health OK, but /solve drops the connection
        self.solve_body = {"type": "turnstile", "token": "0.pool-token", "solved": True} \
            if solve_body is None else solve_body
        self.solve_calls = []
        self.health_calls = 0
        pool = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep pytest output clean
                pass

            def _reply(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                pool.health_calls += 1
                self._reply(pool.health_status, {"status": "ok"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                pool.solve_calls.append(json.loads(self.rfile.read(n) or b"{}"))
                if pool.solve_abort:
                    self.close_connection = True   # no response -> httpx RemoteProtocolError
                    return
                self._reply(pool.solve_status, pool.solve_body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/solve"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def _router_state():
    """Reset config + module-level circuit breaker so tests can't leak into each other."""
    before = (offload._OFFLOAD_ENABLED, offload._OFFLOAD_URL,
              offload._OFFLOAD_COOLDOWN, offload._secondary_down_until,
              offload._health_cache)
    offload._OFFLOAD_ENABLED = False
    offload._OFFLOAD_URL = ""
    offload._OFFLOAD_COOLDOWN = 30.0
    offload._secondary_down_until = 0.0
    offload._health_cache = (0.0, False)
    yield
    (offload._OFFLOAD_ENABLED, offload._OFFLOAD_URL, offload._OFFLOAD_COOLDOWN,
     offload._secondary_down_until, offload._health_cache) = before


def _busy_lock():
    """A local seat already taken by an in-flight solve (mirrors `_solve_lock.locked()`)."""
    lock = asyncio.Lock()
    asyncio.run(lock.acquire())
    return lock


# ── 1. Offload disabled -> local ───────────────────────────────────────

def test_disabled_offload_solves_locally_even_when_seat_is_busy():
    lock = _busy_lock()          # built outside asyncio.run: one lock, two loops

    async def scenario():
        calls = []

        async def local():
            calls.append(1)
            return {"token": "local-token", "method": "route"}

        r = await offload.solve_turnstile_maybe_offload(_Req(), local, "route", lock=lock)
        return r, calls

    r, calls = asyncio.run(scenario())
    assert r["token"] == "local-token"
    assert calls == [1]


# ── 2. Seat free -> local, pool never consulted ────────────────────────

def test_free_seat_solves_locally_even_when_pool_is_healthy():
    pool = _StubPool()
    try:
        offload._OFFLOAD_ENABLED = True
        offload._OFFLOAD_URL = pool.url
        ran = []

        async def local():
            ran.append(1)
            return {"token": "local-token", "method": "route"}

        r = asyncio.run(offload.solve_turnstile_maybe_offload(_Req(), local, "route", lock=asyncio.Lock()))
        assert r["token"] == "local-token"
        assert ran == [1]
        assert pool.solve_calls == []
        assert pool.health_calls == 0
    finally:
        pool.close()


# ── 3. Seat busy + pool healthy -> offload ─────────────────────────────

def test_busy_seat_offloads_to_healthy_pool():
    pool = _StubPool()
    try:
        offload._OFFLOAD_ENABLED = True
        offload._OFFLOAD_URL = pool.url
        ran = []

        async def local():
            ran.append(1)
            return {"token": "local-token", "method": "route"}

        r = asyncio.run(offload.solve_turnstile_maybe_offload(
            _Req(action="login", cdata="ctx"), local, "verify", lock=_busy_lock()))

        assert r["token"] == "0.pool-token"
        assert r["method"] == "offload"
        assert r["verify_success"] is True          # keeps server._is_solved honest
        assert ran == []                            # local seat untouched
        body = pool.solve_calls[0]
        assert body["type"] == "turnstile" and body["mode"] == "verify"
        assert body["sitekey"] == "0xSITEKEY"
        assert body["action"] == "login" and body["cdata"] == "ctx"
        assert body["timeout_s"] == 60
        assert "url" not in body or body["url"]     # present, non-null
        assert "proxy" not in body                  # None dropped, not sent as null
        assert not {"solved", "elapsed", "method"} & body.keys()
    finally:
        pool.close()

# ── 3b. Un-offloadable requests keep the local seat ────────────────────

@pytest.mark.parametrize("req_kw,reason", [
    ({"real_page": True}, "real_page"),
    ({"proxy": "http://192.168.1.100:8080"}, "192.168.x LAN proxy"),
    ({"proxy": "http://127.0.0.1:3128"}, "loopback proxy"),
    ({"proxy": "socks5://localhost:1080"}, "scheme-less dialect, hostname localhost"),
    ({"proxy": "http://user:pass@10.0.0.5:8888"}, "credentials stripped before host parse"),
    ({"proxy": "http://169.254.1.1:80"}, "link-local proxy"),
    ({"proxy": "http://[fd00::1]:80"}, "private IPv6 proxy"),
    ({"proxy": "http://cloakbox.lan:8080"}, "local-TLD hostname proxy"),
    ({"proxy": "http://proxy.internal:8080"}, "internal-TLD hostname proxy"),
    ({"real_page": True, "proxy": "http://8.8.8.8:80"}, "real_page wins over a public proxy"),
    ({"proxy": "http://192.168.1.100:80", "verify_url": "https://x.example.com"}, "LAN proxy"),
])
def test_unreproducible_requests_never_offload(req_kw, reason):
    """SCTG runs no jsd/main.js telemetry and strips private proxies -> its token fails
    Cloudflare validation. Such requests must wait for the local seat, not be offloaded."""
    pool = _StubPool()
    try:
        offload._OFFLOAD_ENABLED = True
        offload._OFFLOAD_URL = pool.url
        ran = []

        async def local():
            ran.append(1)
            return {"token": "local-token", "method": "route"}

        r = asyncio.run(offload.solve_turnstile_maybe_offload(
            _Req(**req_kw), local, "real_page", lock=_busy_lock()))

        assert r["token"] == "local-token", reason
        assert ran == [1], reason                 # seat served it despite being busy
        assert pool.solve_calls == [], reason     # pool never POSTed
    finally:
        pool.close()


@pytest.mark.parametrize("proxy", [
    "http://8.8.8.8:3128",                         # public IPv4
    "http://[2606:4700::1111]:3128",               # public IPv6
    "http://proxy.example.com:3128",               # DNS name the pool can reach too
])
def test_public_proxy_still_offloads(proxy):
    """The veto is scoped to LAN hosts: a public proxy remains offloadable."""
    pool = _StubPool()
    try:
        offload._OFFLOAD_ENABLED = True
        offload._OFFLOAD_URL = pool.url

        async def local():                         # pragma: no cover - must not run
            raise AssertionError("local seat must not serve an offloadable request")

        r = asyncio.run(offload.solve_turnstile_maybe_offload(
            _Req(proxy=proxy), local, "route", lock=_busy_lock()))

        assert r["token"] == "0.pool-token"
        assert pool.solve_calls[0]["proxy"] == proxy
    finally:
        pool.close()

# ── 4. Pool failure -> circuit breaker + local fallback (no raise) ─────

@pytest.mark.parametrize("mode", ["500", "abort", "unreachable"])
def test_pool_failure_marks_down_and_falls_back_to_local(mode):
    # "abort" = health answers 200 but /solve drops the connection (network error on the
    # POST path); "unreachable" = nothing listens, so the health probe already vetoes.
    pool = None
    if mode == "500":
        pool = _StubPool(solve_status=500)
    elif mode == "abort":
        pool = _StubPool(solve_abort=True)
    try:
        offload._OFFLOAD_ENABLED = True
        offload._OFFLOAD_URL = pool.url if pool else DEAD_URL
        ran = []

        async def local():
            ran.append(1)
            return {"token": "local-token", "method": "route"}

        lock = _busy_lock()
        r = asyncio.run(offload.solve_turnstile_maybe_offload(_Req(), local, "route", lock=lock))
        assert r["token"] == "local-token"           # degraded, not failed
        assert ran == [1]

        # A second busy request must not re-hit the pool: breaker open, or (unreachable)
        # the health probe vetoes before any POST. Either way it serves locally.
        asyncio.run(offload.solve_turnstile_maybe_offload(_Req(), local, "route", lock=lock))
        assert ran == [1, 1]
        if mode == "unreachable":
            # Health probe vetoed before any POST, so the breaker never opened: the
            # seat still served the request, and the 5s health cache bounds re-probing.
            assert offload.offload_available() is True
        else:
            assert len(pool.solve_calls) == 1
            assert offload.offload_available() is False          # breaker open
            assert offload._secondary_down_until > time.monotonic()
    finally:
        if pool:
            pool.close()


def test_offload_disabled_config_is_never_available():
    offload._OFFLOAD_ENABLED = False
    offload._OFFLOAD_URL = "http://127.0.0.1:8878/solve"
    assert offload.offload_available() is False
    offload._OFFLOAD_ENABLED = True
    offload._OFFLOAD_URL = ""
    assert offload.offload_available() is False
    offload._OFFLOAD_URL = "http://127.0.0.1:8878/solve"
    assert offload.offload_available() is True


# ── 5. Token normalisation across pool dialects ────────────────────────

@pytest.mark.parametrize("payload,expected", [
    ({"request": "0.req-token"}, "0.req-token"),                          # SCTG/2captcha res.php
    ({"solution": {"token": "0.sol-token"}}, "0.sol-token"),              # nested solution
    ({"data": {"token": "0.data-token"}}, "0.data-token"),                # nested data
    ({"gRecaptchaResponse": "0.grr-token"}, "0.grr-token"),               # capmonster style
    ({"cf_clearance": "cf-cookie"}, "cf-cookie"),                         # cleared cookie
    ({"token": "0.direct-token"}, "0.direct-token"),                      # already local shape
    ({"status": 0, "request": "ERROR_CAPTCHA_UNSOLVABLE"}, None),         # failure: NO token key
    ({"error": "ERROR_ZERO_BALANCE"}, None),                              # failure: NO token key
    ({"status": 0, "request": "WRONG_USER_KEY"}, None),                   # invalid key: NO token
    ({"data": {"token": "ERROR_CAPTCHA_UNSOLVABLE"}}, None),              # nested failure too
])
def test_normalise_across_pool_dialects(payload, expected):
    r = offload._normalise(dict(payload), time.monotonic())
    assert r["method"] == "offload"
    assert isinstance(r["elapsed"], float)
    if expected is None:
        assert not r.get("token")            # truthiness check is the fallback trigger
    else:
        assert r["token"] == expected
        assert r["verify_success"] is True


# ── 6. Server wiring: /solve never 500s on pool failure ────────────────

def test_solve_endpoint_offloads_while_local_seat_is_busy(monkeypatch):
    pool = _StubPool()
    try:
        offload._OFFLOAD_ENABLED = True
        offload._OFFLOAD_URL = pool.url
        monkeypatch.setattr(server_mod, "_ALLOW_PRIVATE", True)
        monkeypatch.setattr(tsolve, "_solve_lock", _busy_lock())   # seat taken
        local_solve = []
        monkeypatch.setattr(tsolve, "solve_turnstile",
                            lambda *a, **k: local_solve.append(1))

        resp = client.post("/solve", json={"type": "turnstile", "sitekey": "0xSITEKEY",
                                           "url": "http://127.0.0.1/app", "timeout_s": 30})
        assert resp.status_code == 200
        body = resp.json()
        assert body["token"] == "0.pool-token"
        assert body["method"] == "offload"
        assert body["solved"] is True
        assert local_solve == []
    finally:
        pool.close()

def test_solve_endpoint_never_offloads_real_page_requests(monkeypatch):
    """End-to-end: a real SolveRequest with real_page=True must take the local seat
    even while it's busy, never the pool (SCTG has no jsd/main.js telemetry)."""
    pool = _StubPool()
    try:
        offload._OFFLOAD_ENABLED = True
        offload._OFFLOAD_URL = pool.url
        monkeypatch.setattr(server_mod, "_ALLOW_PRIVATE", True)
        monkeypatch.setattr(tsolve, "_solve_lock", _busy_lock())
        calls = []

        async def fake_realpage(*a, **k):
            calls.append(k)
            return {"token": "local-token", "verify_success": True, "method": "real-page"}

        monkeypatch.setattr(tsolve, "solve_turnstile_realpage", fake_realpage)

        resp = client.post("/solve", json={"type": "turnstile", "real_page": True,
                                           "sitekey": "0xSITEKEY",
                                           "url": "http://127.0.0.1/app", "timeout_s": 30})
        assert resp.status_code == 200
        assert resp.json()["token"] == "local-token"
        assert len(calls) == 1                      # local realpage solver ran
        assert pool.solve_calls == []               # pool never POSTed
    finally:
        pool.close()


def test_solve_endpoint_falls_back_without_500_when_pool_is_down(monkeypatch):
    pool = _StubPool(solve_abort=True)          # healthy probe, broken solve POST
    try:
        offload._OFFLOAD_ENABLED = True
        offload._OFFLOAD_URL = pool.url
        monkeypatch.setattr(server_mod, "_ALLOW_PRIVATE", True)
        monkeypatch.setattr(tsolve, "_solve_lock", _busy_lock())

        async def fake_solve_turnstile(*a, **k):
            return {"token": "local-token", "method": "route"}

        monkeypatch.setattr(tsolve, "solve_turnstile", fake_solve_turnstile)

        resp = client.post("/solve", json={"type": "turnstile", "sitekey": "0xSITEKEY",
                                           "url": "http://127.0.0.1/app", "timeout_s": 30})
        assert resp.status_code == 200              # degraded, never 500
        body = resp.json()
        assert body["token"] == "local-token"
        assert body["solved"] is True
        assert offload.offload_available() is False
    finally:
        pool.close()


def test_solve_endpoint_falls_back_without_500_when_pool_is_unreachable(monkeypatch):
    offload._OFFLOAD_ENABLED = True
    offload._OFFLOAD_URL = DEAD_URL
    monkeypatch.setattr(server_mod, "_ALLOW_PRIVATE", True)
    monkeypatch.setattr(tsolve, "_solve_lock", _busy_lock())

    async def fake_solve_turnstile(*a, **k):
        return {"token": "local-token", "method": "route"}

    monkeypatch.setattr(tsolve, "solve_turnstile", fake_solve_turnstile)

    resp = client.post("/solve", json={"type": "turnstile", "sitekey": "0xSITEKEY",
                                       "url": "http://127.0.0.1/app", "timeout_s": 30})
    assert resp.status_code == 200
    assert resp.json()["token"] == "local-token"


# ── 7. Burst invariant: exactly one local seat, the rest offloaded ─────

def test_concurrent_burst_keeps_one_local_seat_and_offloads_the_rest():
    """The point of the router: a burst never starts a second Chromium, and never queues."""
    pool = _StubPool()
    try:
        offload._OFFLOAD_ENABLED = True
        offload._OFFLOAD_URL = pool.url

        async def scenario():
            seat = asyncio.Lock()          # stands in for turnstile.solve._solve_lock
            holders = 0
            max_holders = 0

            async def local():
                nonlocal holders, max_holders
                async with seat:
                    holders += 1
                    max_holders = max(max_holders, holders)
                    await asyncio.sleep(0.2)          # simulated browser solve
                    return {"token": "local-token", "method": "route"}

            first = asyncio.create_task(
                offload.solve_turnstile_maybe_offload(_Req(), local, "route", lock=seat))
            await asyncio.wait_for(_seat_taken(seat), timeout=1)   # seat now genuinely busy
            second = asyncio.create_task(
                offload.solve_turnstile_maybe_offload(_Req(), local, "route", lock=seat))
            return await asyncio.gather(first, second), max_holders

        (r1, r2), max_holders = asyncio.run(scenario())
        assert max_holders == 1                                     # single-seat preserved
        # One request won the seat ("route" = the local solver's own tag), the other offloaded.
        assert sorted((r1["method"], r2["method"])) == ["offload", "route"]
        assert sorted((r1["token"], r2["token"])) == ["0.pool-token", "local-token"]
        assert len(pool.solve_calls) == 1
    finally:
        pool.close()


async def _seat_taken(seat):
    while not seat.locked():
        await asyncio.sleep(0)


# ── 8. /solve carries the Turnstile replay fingerprint ─────────────────

def test_solve_endpoint_returns_turnstile_fingerprint(monkeypatch):
    """cf_clearance is a plain string on this path (cloudflare returns a dict record):
    the response model must accept both, and the extra fingerprint keys must survive."""
    offload._OFFLOAD_ENABLED = False          # free seat -> local lane, no pool probe
    monkeypatch.setattr(server_mod, "_ALLOW_PRIVATE", True)
    monkeypatch.setattr(tsolve, "_solve_lock", asyncio.Lock())

    async def fake_solve_turnstile(*a, **k):
        return {"token": "0.local-token", "expires_in": 300, "elapsed": 1.2, "method": "route",
                "user_agent": "Mozilla/5.0 Chrome/146.0.0.0 Safari/537.36",
                "cf_clearance": "cf-clearance-value",
                "cookies": [{"name": "cf_clearance", "value": "cf-clearance-value"}],
                "browser": {"major": 146}}

    monkeypatch.setattr(tsolve, "solve_turnstile", fake_solve_turnstile)

    resp = client.post("/solve", json={"type": "turnstile", "sitekey": "0xSITEKEY",
                                       "url": "http://127.0.0.1/app", "timeout_s": 30})

    assert resp.status_code == 200
    body = resp.json()
    assert body["token"] == "0.local-token"
    assert body["cf_clearance"] == "cf-clearance-value"
    assert body["user_agent"] == "Mozilla/5.0 Chrome/146.0.0.0 Safari/537.36"
    assert body["cookies"] == [{"name": "cf_clearance", "value": "cf-clearance-value"}]
    assert body["browser"] == {"major": 146}


# ── 9. /solve forwards the realpage `managed` flag (separate query, default legacy) ──

@pytest.mark.parametrize("sent,expected", [(None, False), (True, True), (False, False)])
def test_solve_endpoint_forwards_managed_flag(monkeypatch, sent, expected):
    """The operator's contract: existing consumers omit `managed` and keep the legacy
    click-first path; only an explicit `managed: true` switches to token-first."""
    offload._OFFLOAD_ENABLED = False          # free seat -> local lane, no pool probe
    monkeypatch.setattr(server_mod, "_ALLOW_PRIVATE", True)
    monkeypatch.setattr(tsolve, "_solve_lock", asyncio.Lock())
    calls = []

    async def fake_realpage(*a, **k):
        calls.append(k)
        return {"token": "0.tok", "verify_success": True, "method": "real-page"}

    monkeypatch.setattr(tsolve, "solve_turnstile_realpage", fake_realpage)

    body = {"type": "turnstile", "real_page": True, "sitekey": "0xSITEKEY",
            "url": "http://127.0.0.1/app", "timeout_s": 30}
    if sent is not None:
        body["managed"] = sent

    resp = client.post("/solve", json=body)

    assert resp.status_code == 200
    assert resp.json()["token"] == "0.tok"
    assert len(calls) == 1
    assert calls[0]["managed"] is expected, f"managed={sent!r} reached the solver as {calls[0]['managed']!r}"

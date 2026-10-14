"""Offload concurrent Turnstile solves to a secondary solver HTTP API.

The local solver is single-seat: every turnstile entrypoint takes a module-level
`asyncio.Lock` (`turnstile.solve._solve_lock`) so only ONE browser solve runs at a
time on the low-power host hub. That is correct for CPU protection, but it means a
burst of concurrent Turnstile requests serialises.

This module keeps the single-seat invariant for the LOCAL browser while removing the
queue latency: a non-blocking try-acquire decides, per request, whether to solve
locally or offload to a secondary solver pool (sctg-api-pool, default
``http://127.0.0.1:8878/solve``). Exactly one in-flight request wins the local seat;
everyone else is offloaded.
"""
import asyncio
import ipaddress
import logging
import os
import time
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

log = logging.getLogger(__name__)

_OFFLOAD_URL = os.getenv("TURNSTILE_OFFLOAD_URL", "").strip()
_OFFLOAD_TIMEOUT = float(os.getenv("TURNSTILE_OFFLOAD_TIMEOUT", "60"))
_OFFLOAD_HEALTH_TIMEOUT = float(os.getenv("TURNSTILE_OFFLOAD_HEALTH_TIMEOUT", "2.0"))
_OFFLOAD_COOLDOWN = float(os.getenv("TURNSTILE_OFFLOAD_COOLDOWN", "30"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_OFFLOAD_ENABLED = _env_bool("TURNSTILE_OFFLOAD_ENABLED", False)

# Module-level so one uvicorn process shares a single circuit breaker across requests.
_secondary_down_until: float = 0.0
_health_cache: tuple[float, bool] = (0.0, False)
_HEALTH_TTL = 5.0


def offload_available() -> bool:
    if not (_OFFLOAD_ENABLED and _OFFLOAD_URL):
        return False
    return time.monotonic() >= _secondary_down_until


def _mark_secondary_down(reason: str) -> None:
    global _secondary_down_until
    _secondary_down_until = time.monotonic() + _OFFLOAD_COOLDOWN
    log.warning("Turnstile offload pool marked DOWN for %.0fs: %s",
                _OFFLOAD_COOLDOWN, reason)


async def secondary_healthy() -> bool:
    global _health_cache
    if not (_OFFLOAD_ENABLED and _OFFLOAD_URL):
        return False
    cached_at, cached = _health_cache
    if time.monotonic() - cached_at < _HEALTH_TTL:
        return cached
    base = _OFFLOAD_URL.rsplit("/solve", 1)[0].rstrip("/")
    url = f"{base}/health"
    ok = False
    try:
        async with httpx.AsyncClient(timeout=_OFFLOAD_HEALTH_TIMEOUT) as c:
            r = await c.get(url)
            ok = r.status_code < 500
    except Exception as e:
        log.debug("Offload health probe failed: %s", e)
        ok = False
    _health_cache = (time.monotonic(), ok)
    return ok


_LOCAL_SUFFIXES = (".local", ".lan", ".internal", ".home.arpa")


def _proxy_is_local(proxy: Optional[str]) -> bool:
    """True when `proxy` targets a private/loopback/link-local host.

    The pool strips private-LAN proxies, so only the local browser can reproduce such
    a request. Hostnames are matched on local TLDs rather than resolved: `getaddrinfo`
    would block the event loop, and a misjudged public host merely degrades to the
    local queue instead of minting an invalid token.
    """
    if not proxy:
        return False                   # no proxy -> nothing the pool must reproduce
    host = (urlsplit(proxy if "//" in proxy else f"//{proxy}").hostname or "")
    if not host:
        return True                    # unparseable -> assume LAN-side
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
        return True
    try:
        # is_global (not is_private): also flags CGNAT 100.64/10 and the reserved ranges.
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False                   # a DNS name -> the pool can reach it too


async def should_offload(local_lock: asyncio.Lock, req) -> bool:
    """NON-BLOCKING decision: never `await lock.acquire()` to decide.

    `real_page` and private-LAN-proxied requests are NEVER offloaded: SCTG runs no
    `jsd/main.js` telemetry and strips private proxies, so its token fails Cloudflare
    server-side validation. Those requests wait for the local CloakBrowser seat.
    """
    if req.real_page or _proxy_is_local(req.proxy):
        return False                   # pool can't reproduce this request -> local seat only
    if not offload_available():        # enabled + not circuit-broken
        return False
    if not local_lock.locked():
        return False                   # seat is free -> take it locally
    return await secondary_healthy()   # seat busy -> offload only if the pool answers


_BOOKKEEPING = {"solved", "elapsed", "method"}


def _is_error_value(v: Any) -> bool:
    """SCTG/2captcha report failures in the SAME field as success (`request: ERROR_*`,
    `WRONG_USER_KEY`, …), so those strings must never become a token."""
    return isinstance(v, str) and v.upper().startswith(("ERROR", "WRONG"))


def _normalise(payload: dict, t0: float) -> dict:
    """Unwrap a secondary-pool dialect into the local `_is_solved` contract."""
    r: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {"result": payload}
    for _k in ("solution", "data"):
        _v = r.get(_k)
        if isinstance(_v, dict):
            _t = _v.get("token") or _v.get("gRecaptchaResponse")
            if _t and not _is_error_value(_t):
                r["token"] = _t
                break
    tok = (r.get("token") or r.get("cf_clearance")
           or r.get("request") or r.get("gRecaptchaResponse"))
    if isinstance(tok, dict):
        tok = tok.get("token") or tok.get("gRecaptchaResponse")
    if _is_error_value(tok):
        tok = ""       # an ERROR_*/WRONG_* string is a failure, never a token (`r["request"]` keeps it)
    token = str(tok or "")
    if token:
        r["token"] = token
        r.setdefault("verify_success", True)   # so _is_solved accepts it
    r["method"] = "offload"                    # distinguish the lane in /logs
    r["elapsed"] = round(time.monotonic() - t0, 1)
    return r


def _offload_payload(req, kind: str) -> dict:
    """Mirror the local mode downstream; drop nulls and local bookkeeping keys."""
    payload = {
        "type": "turnstile",
        "sitekey": req.sitekey,
        "url": req.url,
        "action": req.action,
        "cdata": req.cdata,
        "real_page": bool(req.real_page),
        "proxy": req.proxy,
        "timeout_s": req.timeout_s,
        "mode": kind,
    }
    return {k: v for k, v in payload.items() if v is not None and k not in _BOOKKEEPING}


async def solve_via_pool(req_payload: dict, t0: Optional[float] = None) -> dict:
    """POST an already-filtered payload; NEVER raises — a pool failure is a normalised dict."""
    t0 = time.monotonic() if t0 is None else t0
    if not offload_available():
        return {"token": "", "error": "offload disabled", "method": "offload",
                "solved": False, "elapsed": round(time.monotonic() - t0, 1)}
    try:
        async with httpx.AsyncClient(timeout=_OFFLOAD_TIMEOUT) as c:
            resp = await c.post(_OFFLOAD_URL, json=req_payload)
        if resp.status_code >= 500:
            _mark_secondary_down(f"HTTP {resp.status_code}")
            return {"token": "", "error": f"offload HTTP {resp.status_code}",
                    "method": "offload", "solved": False,
                    "elapsed": round(time.monotonic() - t0, 1)}
        try:
            payload = resp.json()
        except Exception:
            payload = {"result": resp.text}
        if resp.status_code >= 400:
            payload.setdefault("error", f"offload HTTP {resp.status_code}")
        return _normalise(payload, t0)
    except Exception as e:
        _mark_secondary_down(f"{type(e).__name__}: {e}")
        log.warning("Turnstile offload failed: %s", e)
        return {"token": "", "error": f"offload failed: {e}", "method": "offload",
                "solved": False, "elapsed": round(time.monotonic() - t0, 1)}


async def solve_turnstile_maybe_offload(req, local_call, kind: str, lock=None) -> dict:
    """Route one Turnstile solve: local seat if free, else the healthy pool, else the local queue."""
    if lock is None:
        from turnstile.solve import _solve_lock as lock

    if await should_offload(lock, req):
        log.info("[ROUTER] Local CloakBrowser is busy (locked). Offloading Turnstile "
                 "request to secondary pool (%s)...", _OFFLOAD_URL)
        r = await solve_via_pool(_offload_payload(req, kind))
        if r.get("token"):
            log.info("[ROUTER] Secondary pool (%s) solved Turnstile in %.1fs (token len: %d)",
                     _OFFLOAD_URL, r.get("elapsed", 0), len(r.get("token", "")))
            return r
        log.info("[ROUTER] Secondary pool produced no token (%s); falling back to local queue",
                 r.get("error"))
    return await local_call()          # queued local solve = graceful degradation
"""Unit tests for the Clerk Protect attestation harvester.

No browser launch: the CloakBrowser launcher is replaced with fakes, so these pin
the harvest contract (what the in-page result must contain, what the bundle
carries) and the /solve + /health wiring for type=clerk_protect.
"""
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clerk_protect.solve import _CLERK_PROTECT_JS, solve_clerk_protect
from server import app

NODE = shutil.which("node")

client = TestClient(app)

_PAGE_URL = "https://example.com/login"
_UA_146 = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
_PROTECT_PARAMS = {
    "__clerk_protect_token": "protect-token-value",
    "__clerk_protect_status": "ok",
    "__clerk_protect_cid": "cid-abc123",
}
_COOKIES = [{"name": "__client", "value": "client_1", "domain": ".example.com", "path": "/"}]
# What the in-page harvester returns once ClerkJS mints the attestation.
_JS_RESULT = {
    "params": _PROTECT_PARAMS,
    "client_id": "client_abc",
    "publishable_key": "pk_test_123",
    "fapi_url": "https://clerk.example.com",
    "user_agent": _UA_146,
}


# ── The in-page harvester itself (real JS, executed by node) ─────────

def _js_valid(snippet: str) -> bool:
    """Return True when `snippet` parses as a syntactically valid JS expression."""
    p = subprocess.run([NODE, "--check"], input=f"const _$fn = ({snippet});\n",
                       capture_output=True, text=True)
    return p.returncode == 0


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_clerk_protect_js_is_valid_javascript():
    assert _js_valid(_CLERK_PROTECT_JS)


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_clerk_protect_js_polls_until_clerkjs_mints_the_token():
    """Run the real snippet against mocked page globals: absent ClerkJS -> null, a
    not-yet-ready Clerk -> polled until ready, and the harvested shape is exactly what
    the Python side unpacks (params + trimmed FAPI url + publishable key + UA)."""
    harness = f"""
const harvest = ({_CLERK_PROTECT_JS});

function page(clerk) {{
  globalThis.window = {{ Clerk: clerk }};
  Object.defineProperty(globalThis, 'navigator',
                        {{ value: {{ userAgent: 'UA-NODE' }}, configurable: true }});
}}

function clerk({{ readyAfter = 0, protectParams = {{ __clerk_protect_token: 'tk_1' }} }} = {{}}) {{
  let polls = 0;
  return {{
    loaded: false,
    isReady: () => ++polls > readyAfter,
    client: {{ id: 'client_abc' }},
    publishableKey: 'pk_1',
    frontendApi: 'https://fapi.example.com/',
    getFapiClient: () => ({{ apiUrl: 'https://fapi.example.com/',
                            getProtectParams: async () => protectParams }}),
  }};
}}

function clerkHookOnly(token = 'tok-hook') {{
  const state = {{ probes: 0 }};
  let hook = null;
  return {{
    state,                                 // exposed so asserts can read the probe count
    loaded: true,
    isReady: () => true,
    client: {{ id: 'client_hook', signIn: {{ create: async () => {{
      state.probes++;
      // ClerkJS merges the Protect params into the body, then fires before-request hooks.
      hook({{ method: 'POST', path: '/client/sign_ins',
              body: 'identifier=x%40example.com&__clerk_protect_token=' + token
                    + '&__clerk_protect_status=ok&__clerk_protect_cid=1-cid-rid' }});
      return {{}};
    }} }} }},
    publishableKey: 'pk_hook',
    frontendApi: 'https://fapi.example.com/',
    getFapiClient: () => ({{ apiUrl: 'https://fapi.example.com/',
                            onBeforeRequest: (cb) => {{ hook = cb; }} }}),
  }};
}}

function assert(name, cond, detail) {{
  if (!cond) throw new Error(name + ': ' + detail);
}}

(async () => {{
  page(undefined);
  assert('no-clerk-returns-null', await harvest({{ timeoutMs: 50 }}) === null, 'expected null');

  page(clerk({{ readyAfter: 1 }}));                       // ready only on the 2nd poll
  const got = await harvest({{ timeoutMs: 3000 }});
  assert('late-ready-is-polled-for',
         got && got.params.__clerk_protect_token === 'tk_1', JSON.stringify(got));
  assert('client-id', got.client_id === 'client_abc', JSON.stringify(got));
  assert('publishable-key', got.publishable_key === 'pk_1', JSON.stringify(got));
  assert('fapi-url-strips-trailing-slash',
         got.fapi_url === 'https://fapi.example.com', got.fapi_url);
  assert('user-agent', got.user_agent === 'UA-NODE', String(got.user_agent));

  page(clerk({{ readyAfter: 0, protectParams: {{}} }}));  // params without a token yet
  assert('tokenless-params-return-null', await harvest({{ timeoutMs: 400 }}) === null,
         'expected null');

  const hooked = clerkHookOnly();                        // ClerkJS 6.x: no getProtectParams
  page(hooked);
  const gotHook = await harvest({{ timeoutMs: 3000 }});
  assert('hook-captures-token',
         gotHook && gotHook.params.__clerk_protect_token === 'tok-hook', JSON.stringify(gotHook));
  assert('hook-status-and-cid', gotHook.params.__clerk_protect_status === 'ok'
         && gotHook.params.__clerk_protect_cid === '1-cid-rid', JSON.stringify(gotHook.params));
  assert('hook-publishable-key', gotHook.publishable_key === 'pk_hook',
         String(gotHook.publishable_key));
  assert('probe-is-throttled', hooked.state.probes === 1, 'probes=' + hooked.state.probes);
}})().then(() => process.exit(0), (e) => {{ console.error(e.message); process.exit(1); }});
"""
    p = subprocess.run([NODE, "-e", harness], capture_output=True, text=True)
    assert p.returncode == 0, f"node harness failed: {p.stderr}"


def _install_fake(monkeypatch, *, evaluate_result=_JS_RESULT, existing_context=False):
    """Install a fake CloakBrowser answering the harvest calls.

    `existing_context=True` models a persistent profile that already holds a
    context (`browser.contexts` non-empty). Returns the call log; `events` records
    the order of goto/pre-action/harvest/close.
    """
    calls = {"goto": [], "closed": 0, "new_context": 0, "events": []}

    class _Page:
        async def goto(self, url, **k):
            calls["events"].append("goto")
            calls["goto"].append((url, k.get("wait_until")))

        async def evaluate(self, js, arg=None):
            calls["events"].append("harvest")
            calls["evaluate_arg"] = arg
            if isinstance(evaluate_result, Exception):
                raise evaluate_result
            return evaluate_result

        async def close(self):
            calls["events"].append("close")
            calls["closed"] += 1

    class _Context:
        async def new_page(self):
            return _Page()

        async def cookies(self):
            return list(_COOKIES)

    ctx = _Context()

    class _Browser:
        @property
        def contexts(self):
            return [ctx] if existing_context else []

        async def new_context(self, **kw):
            calls["new_context"] += 1
            calls["viewport"] = kw.get("viewport")
            return ctx

    class _Launcher:
        async def __aenter__(self):
            return _Browser()

        async def __aexit__(self, *a):
            return False

    async def _launch(**kw):
        calls["kwargs"] = kw
        return _Launcher()

    monkeypatch.setattr("clerk_protect.solve.cloakbrowser.launch_async", _launch)
    return calls


def test_solve_clerk_protect_returns_attestation_bundle(monkeypatch):
    """The bundle must carry the token, the raw params, and the replay material."""
    calls = _install_fake(monkeypatch)

    r = asyncio.run(solve_clerk_protect(_PAGE_URL, proxy="http://127.0.0.1:8080"))

    assert r["token"] == "protect-token-value"
    assert r["protect_params"] == _PROTECT_PARAMS
    assert r["client_id"] == "client_abc"
    assert r["publishable_key"] == "pk_test_123"
    assert r["fapi_url"] == "https://clerk.example.com"
    assert r["cookies"] == _COOKIES
    assert r["user_agent"] == _UA_146
    assert r["method"] == "clerk_protect_harvest"
    assert isinstance(r["elapsed"], float)
    assert calls["goto"] == [(_PAGE_URL, "domcontentloaded")]
    assert calls["kwargs"]["proxy"] == "http://127.0.0.1:8080"
    assert calls["closed"] == 1, "browser page leaked"
    assert calls["viewport"] == {"width": 1920, "height": 1080}


@pytest.mark.parametrize("timeout_s,expected_ms", [(45, 20000), (10, 10000)])
def test_solve_clerk_protect_caps_the_poll_at_the_caller_budget(monkeypatch, timeout_s, expected_ms):
    """The in-page poll never outlives the caller's budget, and never exceeds 20s."""
    calls = _install_fake(monkeypatch)

    asyncio.run(solve_clerk_protect(_PAGE_URL, timeout_s=timeout_s))

    assert calls["evaluate_arg"] == {"timeoutMs": expected_ms}


def test_solve_clerk_protect_runs_pre_actions_before_harvesting(monkeypatch):
    """ClerkJS often only loads after opening the form — pre_actions must land first."""
    calls = _install_fake(monkeypatch)

    async def _pre_actions(page, actions):
        calls["events"].append("pre_actions")

    monkeypatch.setattr("clerk_protect.solve.run_pre_actions", _pre_actions)

    actions = [{"type": "click", "selector": "text=Sign in"}]
    asyncio.run(solve_clerk_protect(_PAGE_URL, pre_actions=actions))

    assert calls["events"] == ["goto", "pre_actions", "harvest", "close"]


def test_solve_clerk_protect_reuses_an_existing_browser_context(monkeypatch):
    """A persistent CloakBrowser profile already holds a context — don't fork a new one."""
    calls = _install_fake(monkeypatch, existing_context=True)

    r = asyncio.run(solve_clerk_protect(_PAGE_URL))

    assert calls["new_context"] == 0
    assert r["cookies"] == _COOKIES


@pytest.mark.parametrize("evaluate_result", [
    None,                                        # ClerkJS never loaded / poll deadline hit
    {"params": {}},                              # params landed without the token
    {"params": {"__clerk_protect_status": "ok"}},
])
def test_solve_clerk_protect_raises_without_a_token(monkeypatch, evaluate_result):
    calls = _install_fake(monkeypatch, evaluate_result=evaluate_result)

    with pytest.raises(RuntimeError) as err:
        asyncio.run(solve_clerk_protect(_PAGE_URL))

    assert "Clerk Protect failed to generate attestation token" in str(err.value)
    assert calls["closed"] == 1, "browser page leaked on the error path"


# ── /solve + /health wiring ──────────────────────────────────────────

def test_solve_endpoint_returns_the_clerk_protect_bundle(monkeypatch):
    seen = {}

    async def _fake_solve(*, url, proxy, timeout_s, pre_actions):
        seen.update({"url": url, "proxy": proxy, "timeout_s": timeout_s,
                     "pre_actions": pre_actions})
        return {"token": "protect-token-value", "protect_params": _PROTECT_PARAMS,
                "publishable_key": "pk_test_123", "fapi_url": "https://clerk.example.com",
                "client_id": "client_abc", "cookies": _COOKIES, "user_agent": _UA_146,
                "method": "clerk_protect_harvest", "elapsed": 3.2}

    monkeypatch.setattr("clerk_protect.solve.solve_clerk_protect", _fake_solve)

    resp = client.post("/solve", json={"type": "clerk_protect", "url": _PAGE_URL,
                                      "timeout_s": 30})

    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "clerk_protect"
    assert body["solved"] is True
    assert body["token"] == "protect-token-value"
    assert body["protect_params"] == _PROTECT_PARAMS
    assert body["user_agent"] == _UA_146
    assert body["cookies"] == _COOKIES
    # No sitekey on this lane; the caller's URL is what gets driven.
    assert seen == {"url": _PAGE_URL, "proxy": None, "timeout_s": 30, "pre_actions": None}


def test_solve_endpoint_clerk_protect_returns_structured_failure(monkeypatch):
    """A failing solve (nav/proxy/timeout) must be a 200 {solved: false, error}, not a 500."""
    async def _boom(*, url, proxy, timeout_s, pre_actions):
        raise RuntimeError("net::ERR_INVALID_AUTH_CREDENTIALS")

    monkeypatch.setattr("clerk_protect.solve.solve_clerk_protect", _boom)

    resp = client.post("/solve", json={"type": "clerk_protect", "url": _PAGE_URL})

    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "clerk_protect"
    assert body["solved"] is False
    assert "ERR_INVALID_AUTH_CREDENTIALS" in body["error"]


def test_solve_endpoint_clerk_protect_requires_a_url():
    resp = client.post("/solve", json={"type": "clerk_protect"})

    assert resp.status_code == 400
    assert "url is required" in resp.json()["detail"]


def test_health_advertises_clerk_protect():
    resp = client.get("/health")

    assert resp.status_code == 200
    assert "clerk_protect" in resp.json()["supported_types"]

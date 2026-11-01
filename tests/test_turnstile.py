"""Unit tests for the Turnstile JS snippets and string formatting.

No browser launch and no image rendering: JS snippets are syntax-checked with
node and the token-harvest snippet is executed against mocked globals; the
route-intercept widget-div builder is tested directly.
"""
import asyncio
import inspect
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turnstile.solve import (_GET_TOKEN_JS, _HAS_WIDGET_JS, _WIDGET_INJECT_JS, _browser_major,
                             _turnstile_div, solve_and_verify, solve_turnstile,
                             solve_turnstile_realpage)


NODE = shutil.which("node")

_INPUT_SEL = 'input[name="cf-turnstile-response"], [name="cf-turnstile-response"]'
_TEXTAREA_SEL = 'textarea[name="cf-turnstile-response"]'


def _js_valid(snippet: str) -> bool:
    """Return True when `snippet` parses as a syntactically valid JS expression."""
    script = f"const _$fn = ({snippet});\n"
    p = subprocess.run([NODE, "--check"], input=script, capture_output=True, text=True)
    return p.returncode == 0


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_widget_inject_js_is_valid_javascript():
    assert _js_valid(_WIDGET_INJECT_JS)


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_get_token_js_is_valid_javascript():
    assert _js_valid(_GET_TOKEN_JS)


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_get_token_js_harvest_vectors():
    """Execute the token-harvest snippet against mocks and check every vector."""
    harness = f"""
const extract = ({_GET_TOKEN_JS});

function check(name, setup, expected) {{
  delete globalThis.turnstile;
  delete globalThis.window;
  delete globalThis.document;
  setup();
  const got = extract();
  if (got !== expected) {{
    throw new Error(name + ': expected ' + JSON.stringify(expected) + ', got ' + JSON.stringify(got));
  }}
}}

const INP = {_INPUT_SEL!r};
const TXT = {_TEXTAREA_SEL!r};

check('vector1-widget-id', () => {{
  globalThis.turnstile = {{ getResponse: (id) => id !== undefined ? 'RESP_' + id : 'FALLBACK' }};
  globalThis.window = {{ __turnstileWidgetId: 5 }};
}}, 'RESP_5');

check('vector1-fallback', () => {{
  globalThis.turnstile = {{ getResponse: () => 'FALLBACK' }};
  globalThis.window = {{}};
}}, 'FALLBACK');

check('vector2-custom-prop', () => {{
  globalThis.window = {{ turnstileToken: 'CUSTOM' }};
}}, 'CUSTOM');

check('vector3-input', () => {{
  globalThis.window = {{}};
  globalThis.document = {{ querySelectorAll: (sel) => sel === INP ? [{{ value: '' }}, {{ value: 'INPUT_TOKEN' }}] : [] }};
}}, 'INPUT_TOKEN');

check('vector4-textarea', () => {{
  globalThis.window = {{}};
  globalThis.document = {{ querySelectorAll: (sel) => sel === TXT ? [{{ value: 'TEXTA_TOKEN' }}] : [] }};
}}, 'TEXTA_TOKEN');

check('none-found', () => {{
  globalThis.window = {{}};
  globalThis.document = {{ querySelectorAll: () => [] }};
}}, '');
"""
    p = subprocess.run([NODE, "-e", harness], capture_output=True, text=True)
    assert p.returncode == 0, f"node harness failed: {p.stderr}"


def test_turnstile_div_minimal():
    assert _turnstile_div("SITEKEY") == (
        '<div class="cf-turnstile" data-sitekey="SITEKEY"></div>'
    )


def test_turnstile_div_with_action_and_cdata():
    assert _turnstile_div("SITEKEY", "login", "ctx") == (
        '<div class="cf-turnstile" data-sitekey="SITEKEY" '
        'data-action="login" data-cdata="ctx"></div>'
    )


def test_turnstile_div_omits_empty_optional_attrs():
    assert _turnstile_div("SITEKEY", "", "") == (
        '<div class="cf-turnstile" data-sitekey="SITEKEY"></div>'
    )


def test_js_snippets_do_not_interpolate_sitekey():
    # Zero secret exposure: evaluate receives arguments object ({ sitekey, action, cdata }), never baked into source.
    assert "const renderOpts = { sitekey };" in _WIDGET_INJECT_JS
    assert "{sitekey" not in _WIDGET_INJECT_JS
    assert "{sitekey" not in _GET_TOKEN_JS


def test_solve_turnstile_realpage_signature_accepts_action_and_cdata():
    sig = inspect.signature(solve_turnstile_realpage)
    params = sig.parameters
    assert "action" in params
    assert "cdata" in params
    assert params["action"].default is None
    assert params["cdata"].default is None


@pytest.mark.parametrize("exc", [
    TimeoutError("page.goto: Timeout 45000ms exceeded"),
    RuntimeError("net::ERR_CONNECTION_CLOSED"),
    RuntimeError("net::ERR_TIMED_OUT"),
    RuntimeError("net::ERR_EMPTY_RESPONSE"),
])
def test_realpage_nav_failure_fails_fast_without_harvesting(monkeypatch, exc):
    """A dead/blank page has no widget: the 140s harvest loop must never run, so the
    single seat is released immediately instead of burning the whole timeout_s."""
    calls = {"goto": 0, "evaluate": 0, "closed": 0}

    class _Page:
        async def goto(self, *a, **k):
            calls["goto"] += 1
            raise exc

        async def evaluate(self, *a, **k):
            calls["evaluate"] += 1
            return ""

        async def close(self):
            calls["closed"] += 1

    class _Browser:
        async def new_page(self):
            return _Page()

    class _Launcher:
        async def __aenter__(self):
            return _Browser()

        async def __aexit__(self, *a):
            return False

    async def _launch(**_kw):
        return _Launcher()

    monkeypatch.setattr("turnstile.solve.cloakbrowser.launch_async", _launch)

    r = asyncio.run(solve_turnstile_realpage("https://dead.example.com/", "0xSITEKEY", timeout_s=140))

    assert calls["goto"] == 1
    assert calls["evaluate"] == 0, "harvested a token off a page that never loaded"
    assert calls["closed"] == 1, "browser page leaked"
    assert r == {"token": "", "verify_success": False,
                 "error": f"nav_failed: {exc}",
                 "method": "real-page", "elapsed": r["elapsed"]}
    assert r["elapsed"] < 5


# ── Replay fingerprint on the route-intercept solve ──────────────────

_UA_146 = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
_UA_FIREFOX = "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0"
_CF_COOKIE = {"name": "cf_clearance", "value": "cf-clearance-value",
              "domain": ".dahl.global", "path": "/"}


def _fake_browser(monkeypatch, *, ua=_UA_146, cookies=(_CF_COOKIE,), token="0.tok",
                  prime_fails=False):
    """Install a fake CloakBrowser answering the route-intercept calls; returns the call log."""
    calls = {"goto": [], "closed": 0}

    class _Page:
        def __init__(self):
            self.context = self          # page.context.cookies() -> the jar

        async def route(self, *a, **k):
            pass

        async def goto(self, url, **k):
            calls["goto"].append(k.get("wait_until"))
            if prime_fails and len(calls["goto"]) == 1:
                raise RuntimeError("net::ERR_TIMED_OUT")

        async def input_value(self, sel):
            return token

        async def evaluate(self, js, *a):
            # Only the fingerprint reads the UA; fetch_from_page wants {status, body}.
            return ua if "navigator.userAgent" in js else {"status": 200, "body": "{}"}

        async def cookies(self):
            return list(cookies)

        async def close(self):
            calls["closed"] += 1

    class _Launcher:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def new_page(self):
            return _Page()

    async def _launch(**_kw):
        return _Launcher()

    monkeypatch.setattr("turnstile.solve.cloakbrowser.launch_async", _launch)
    return calls


def test_route_solve_returns_replay_fingerprint(monkeypatch):
    """A curl_cffi replay needs the UA + cf_clearance + jar that produced the token."""
    calls = _fake_browser(monkeypatch)

    r = asyncio.run(solve_turnstile("0xSITEKEY", "https://dahl.example.com/signup",
                                    action="signup"))

    assert r["token"] == "0.tok"
    assert r["user_agent"] == _UA_146
    assert r["cf_clearance"] == "cf-clearance-value"
    assert r["cookies"] == [_CF_COOKIE]
    assert r["browser"] == {"major": 146}
    # Phase 1 = real page (networkidle, primes cf_clearance), phase 2 = intercepted shell.
    assert calls["goto"] == ["networkidle", "domcontentloaded"]
    assert calls["closed"] == 1


def test_route_solve_survives_prime_nav_failure(monkeypatch):
    """An unreachable target still yields the token — just without a cf_clearance."""
    calls = _fake_browser(monkeypatch, cookies=(), prime_fails=True)

    r = asyncio.run(solve_turnstile("0xSITEKEY", "https://dead.example.com/"))

    assert r["token"] == "0.tok"
    assert r["cf_clearance"] is None
    assert r["cookies"] == []
    assert calls["goto"] == ["networkidle", "domcontentloaded"]


def test_route_solve_major_is_none_without_a_chrome_token(monkeypatch):
    _fake_browser(monkeypatch, ua=_UA_FIREFOX, cookies=())

    r = asyncio.run(solve_turnstile("0xSITEKEY", "https://example.com/"))

    assert r["browser"] == {"major": None}   # never invent a version the UA doesn't state
    assert r["user_agent"] == _UA_FIREFOX


@pytest.mark.parametrize("ua,expected", [
    (_UA_146, 146),
    ("Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/142.0.7444.60 Safari/537.36", 142),
    (_UA_FIREFOX, None),
    ("", None),
    (None, None),
])
def test_browser_major_parses_chrome_version_or_falls_back(ua, expected):
    assert _browser_major(ua) == expected


def test_solve_and_verify_returns_fingerprint(monkeypatch):
    _fake_browser(monkeypatch)

    r = asyncio.run(solve_and_verify("0xSITEKEY", "https://dahl.example.com/api/verify"))

    assert r["verify_status"] == 200
    assert r["user_agent"] == _UA_146
    assert r["cf_clearance"] == "cf-clearance-value"
    assert r["cookies"] == [_CF_COOKIE]
    assert r["browser"] == {"major": 146}


# ── Real-page: conditional widget injection on the resident CloakBrowser ─────

def _realpage_fakes(monkeypatch, *, has_widget, token="0.tok"):
    """Install a fake resident CloakBrowser for solve_turnstile_realpage; returns the call log."""
    calls = {"launch": 0, "goto": 0, "inject": 0, "evaluate": [], "click": 0,
             "page_closed": 0}

    class _Page:
        def __init__(self):
            self.context = self        # page.context.cookies() -> the jar

        async def goto(self, *a, **k):
            calls["goto"] += 1

        async def evaluate(self, js, *a):
            calls["evaluate"].append(js)
            if js is _HAS_WIDGET_JS:
                return has_widget
            if js is _GET_TOKEN_JS:
                return token
            return _UA_146 if "navigator.userAgent" in js else {"status": 200, "body": "{}"}

        async def cookies(self):
            return [_CF_COOKIE]

        async def close(self):
            calls["page_closed"] += 1

    class _Launcher:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def new_page(self):
            return _Page()

    async def _launch(**_kw):
        calls["launch"] += 1
        return _Launcher()

    async def _no_click(page, attempts=25):
        calls["click"] += 1
        return True

    async def _inject(page, sitekey, action=None, cdata=None):
        calls["inject"] += 1

    monkeypatch.setattr("turnstile.solve.cloakbrowser.launch_async", _launch)
    monkeypatch.setattr("turnstile.solve._click_turnstile_checkbox", _no_click)
    monkeypatch.setattr("turnstile.solve._inject_turnstile_widget", _inject)
    return calls


def test_realpage_skips_injection_when_page_has_its_own_widget(monkeypatch):
    """Dahl renders the widget itself: a second synthetic one conflicts with the native challenge."""
    calls = _realpage_fakes(monkeypatch, has_widget=True)

    r = asyncio.run(solve_turnstile_realpage("https://dahl.example.com/account", "0xSITEKEY"))

    assert calls["evaluate"][0] is _HAS_WIDGET_JS
    assert calls["inject"] == 0
    assert r["token"] == "0.tok"


def test_realpage_injects_widget_on_a_bare_page(monkeypatch):
    calls = _realpage_fakes(monkeypatch, has_widget=False)

    asyncio.run(solve_turnstile_realpage("https://bare.example.com/", "0xSITEKEY"))

    assert calls["inject"] == 1


def test_realpage_launches_resident_cloakbrowser(monkeypatch):
    """No CDP attach path: every real_page solve launches the resident CloakBrowser once."""
    calls = _realpage_fakes(monkeypatch, has_widget=True)

    r = asyncio.run(solve_turnstile_realpage("https://dahl.example.com/account", "0xSITEKEY"))

    assert calls["launch"] == 1
    assert r["method"] == "real-page" and r["token"] == "0.tok"


# ── Real-page: `managed` gates token-first vs legacy click-first ──────────────

def test_realpage_managed_false_preserves_legacy_checkbox_click(monkeypatch):
    """Default (managed=False) is the legacy contract: 25-attempt click before harvesting."""
    calls = _realpage_fakes(monkeypatch, has_widget=True, token="0.tok")

    r = asyncio.run(solve_turnstile_realpage("https://dahl.example.com/account", "0xSITEKEY"))

    assert calls["click"] == 1, "legacy mode must click the checkbox upfront"
    assert r["token"] == "0.tok" and r["verify_success"] is True


def test_realpage_managed_true_skips_upfront_checkbox_click(monkeypatch):
    """Managed / interaction-only widgets never render a checkbox: token-first, no upfront click."""
    calls = _realpage_fakes(monkeypatch, has_widget=True, token="0.tok")

    r = asyncio.run(solve_turnstile_realpage("https://dahl.example.com/account", "0xSITEKEY",
                                             managed=True))

    assert r["token"] == "0.tok" and r["verify_success"] is True
    assert calls["click"] == 0, "managed mode clicked the checkbox before waiting for the token"
    assert calls["evaluate"].count(_GET_TOKEN_JS) == 1, "polled again after the token had arrived"
    assert r["elapsed"] < 1, "slept between polls despite a token on the first attempt"


def test_realpage_managed_true_falls_back_to_one_checkbox_click(monkeypatch):
    """Managed mode still gets exactly one click attempt when no token ever lands."""
    calls = _realpage_fakes(monkeypatch, has_widget=True, token="")

    r = asyncio.run(solve_turnstile_realpage("https://dahl.example.com/account", "0xSITEKEY",
                                             timeout_s=4, managed=True))

    assert r["token"] == "" and r["verify_success"] is False
    assert calls["click"] == 1, f"expected one fallback click, got {calls['click']}"
    assert calls["evaluate"].count(_GET_TOKEN_JS) >= 3, "clicked before polling for a token"

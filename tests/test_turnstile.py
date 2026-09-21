"""Unit tests for the Turnstile JS snippets and string formatting.

No browser launch and no image rendering: JS snippets are syntax-checked with
node and the token-harvest snippet is executed against mocked globals; the
route-intercept widget-div builder is tested directly.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turnstile.solve import _GET_TOKEN_JS, _WIDGET_INJECT_JS, _turnstile_div

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
    # Zero secret exposure: sitekey is the evaluate() arg `k`, never baked into source.
    assert "sitekey: k" in _WIDGET_INJECT_JS
    assert "{sitekey" not in _WIDGET_INJECT_JS
    assert "{sitekey" not in _GET_TOKEN_JS
"""Harvest Clerk Protect attestation params from a live Clerk-fronted page.

Clerk's bot protection mints a short-lived attestation triple on the client:
`__clerk_protect_token` (+ `__clerk_protect_status` / `__clerk_protect_cid`).
This solver drives a real CloakBrowser page until ClerkJS is loaded and its
Protect attestation lands, then returns the params plus everything needed to
replay the sign-up / sign-in requests over plain HTTP (cookies, User-Agent,
publishable key, FAPI url, client_id).

Framework-agnostic: the caller supplies the URL (any page that loads ClerkJS —
a sign-in page, a Next.js route, a marketing page with the Clerk component), so
no sitekey or app-specific knowledge is required.
"""
import asyncio
import logging
import time

import cloakbrowser

from common.browser import browser_kwargs, run_pre_actions

log = logging.getLogger(__name__)
_solve_lock = asyncio.Lock()

# Polls for ClerkJS and extracts Clerk Protect attestation params.
#
# Supports two extraction routes:
# 1. Direct FAPI: `fapi.getProtectParams()` when exposed by the ClerkJS version.
# 2. Universal ClerkJS 6.x / FAPI onBeforeRequest hook: ClerkJS 6 attaches Protect
#    params via an internal private method `#eH()` right before mutation requests
#    (/client/sign_ins, /client/sign_ups). Registering an `onBeforeRequest` hook on
#    the FAPI client and triggering a benign dummy probe request captures the exact
#    minted `__clerk_protect_token`, `__clerk_protect_status`, and `__clerk_protect_cid`
#    live from Clerk's internal Protect instance without side effects.
#
# Returns null when the token never lands within `timeoutMs`.
_CLERK_PROTECT_JS = r"""async ({ timeoutMs }) => {
    const start = Date.now();
    let hookedParams = null;
    let hookInstalled = false;
    let lastPath1 = 0;
    let lastProbe = 0;
    function parseParams(body) {
        if (!body) return null;
        if (typeof body === 'object' && body.__clerk_protect_token) return body;
        if (typeof body === 'string') {
            let flat = body;
            if (flat.charAt(0) === '{') {
                try { flat = new URLSearchParams(JSON.parse(flat)).toString(); } catch (e) { return null; }
            }
            try {
                const sp = new URLSearchParams(flat);
                const tok = sp.get('__clerk_protect_token');
                if (tok) {
                    return {
                        __clerk_protect_token: tok,
                        __clerk_protect_status: sp.get('__clerk_protect_status') || 'ok',
                        __clerk_protect_cid: sp.get('__clerk_protect_cid') || ''
                    };
                }
            } catch (e) {}
        }
        return null;
    }

    while (Date.now() - start < timeoutMs) {
        const clerk = window.Clerk;
        if (clerk && (clerk.loaded || (typeof clerk.isReady === 'function' && clerk.isReady()))) {
            try {
                const fapi = typeof clerk.getFapiClient === 'function' ? clerk.getFapiClient() : null;

                // Path 1: Direct FAPI helper (spec contract)
                if (fapi && typeof fapi.getProtectParams === 'function'
                    && Date.now() - lastPath1 > 2000) {
                    lastPath1 = Date.now();
                    const params = await fapi.getProtectParams();
                    if (params && params.__clerk_protect_token) {
                        return {
                            params,
                            client_id: clerk.client ? clerk.client.id : "",
                            publishable_key: clerk.publishableKey || window.__clerk_publishable_key || "",
                            fapi_url: (fapi.apiUrl || clerk.frontendApi || "").replace(/\/$/, ""),
                            user_agent: navigator.userAgent
                        };
                    }
                }

                // Path 2: Universal hook on FAPI client requests (ClerkJS 6.x)
                if (fapi && typeof fapi.onBeforeRequest === 'function' && !hookInstalled) {
                    try {
                        fapi.onBeforeRequest((req) => {
                            const p = parseParams(req && req.body);
                            if (p && p.__clerk_protect_token) hookedParams = p;
                        });
                        hookInstalled = true;
                    } catch (e) {}
                }

                if (hookedParams && hookedParams.__clerk_protect_token) {
                    return {
                        params: hookedParams,
                        client_id: clerk.client ? clerk.client.id : "",
                        publishable_key: clerk.publishableKey || window.__clerk_publishable_key || "",
                        fapi_url: (fapi && fapi.apiUrl || clerk.frontendApi || "").replace(/\/$/, ""),
                        user_agent: navigator.userAgent
                    };
                }

                // Fire a benign probe via clerk.client.signIn to force Protect attestation
                if (hookInstalled && !hookedParams && Date.now() - lastProbe > 5000
                    && clerk.client && clerk.client.signIn
                    && typeof clerk.client.signIn.create === 'function') {
                    lastProbe = Date.now();
                    clerk.client.signIn.create({
                        identifier: 'probe-' + Date.now() + '@clerk-probe.invalid'
                    }).catch(() => {});
                }
            } catch (e) {}
        }
        await new Promise(r => setTimeout(r, 250));
    }
    return null;
}"""


def _kwargs(proxy: str = None) -> dict:
    return browser_kwargs("CLERK_PROTECT", proxy=proxy)


async def solve_clerk_protect(url: str, proxy: str = None, timeout_s: int = 45,
                              pre_actions: list = None) -> dict:
    """Navigate a Clerk-Protect fronted page and harvest the attestation bundle.

    Returns {token, protect_params, publishable_key, fapi_url, client_id, cookies,
    user_agent, method, elapsed}. The attestation triple is minted per page session
    and is bound to the cookies + User-Agent that produced it, so replay from the
    same proxy IP with this exact jar/UA, and immediately — it is short-lived and
    single-use. Pass `pre_actions` to open the sign-up/sign-in form first when the
    target only loads ClerkJS after a click.
    """
    t0 = time.monotonic()
    # Cap the in-page poll at 20s (or the caller's budget, if smaller) — the caller's
    # `timeout_s` also bounds the navigation and the /solve deadline.
    poll_ms = max(1000, min(timeout_s * 1000, 20000))
    async with _solve_lock:
        async with await cloakbrowser.launch_async(**_kwargs(proxy)) as browser:
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
                if pre_actions:
                    await run_pre_actions(page, pre_actions)
                result = await page.evaluate(_CLERK_PROTECT_JS, {"timeoutMs": poll_ms})
                if not result or not (result.get("params") or {}).get("__clerk_protect_token"):
                    raise RuntimeError(
                        "Clerk Protect failed to generate attestation token within timeout")
                params = result["params"]
                return {
                    "token": params.get("__clerk_protect_token"),
                    "protect_params": params,
                    "publishable_key": result.get("publishable_key", ""),
                    "fapi_url": result.get("fapi_url", ""),
                    "client_id": result.get("client_id", ""),
                    "cookies": await context.cookies(),
                    "user_agent": result.get("user_agent", ""),
                    "method": "clerk_protect_harvest",
                    "elapsed": round(time.monotonic() - t0, 1),
                }
            finally:
                await page.close()

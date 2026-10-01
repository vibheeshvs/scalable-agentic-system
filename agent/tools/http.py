"""One generic executor for every HTTP tool in the catalog.

Responsibilities that should never be left to the LLM:
  - auth (OAuth token caching / refresh), base URLs, headers
  - idempotency keys on the writes whose provider honours one, so a retry can't double-charge
  - retries with backoff for 429/5xx/timeouts, honouring Retry-After; a write with no idempotency
    guarantee is never replayed after a 5xx or timeout, it is reported as "outcome unknown"
  - pagination, so "total sales last month" doesn't silently stop at page 1
  - turning provider error payloads into short, actionable messages for the repair loop
"""

from __future__ import annotations

import base64
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import quote

import httpx

from ..registry.models import ToolSpec

RETRYABLE = {408, 425, 429, 500, 502, 503, 504}
UNKNOWN_OUTCOME = (" - not retried: this endpoint has no idempotency guarantee, so the action may or may not have "
                   "gone through. Check its current state before trying again.")


@dataclass
class CallResult:
    ok: bool
    status: int | None
    data: Any = None
    error: str | None = None
    retryable: bool = False
    attempts: int = 1
    latency_ms: int = 0
    request: dict[str, Any] = field(default_factory=dict)


class AuthProvider:
    def headers(self, client: httpx.Client, force_refresh: bool = False) -> dict[str, str]:
        return {}


class BearerTokenAuth(AuthProvider):
    def __init__(self, token: str):
        self.token = token

    def headers(self, client, force_refresh=False):
        return {"Authorization": f"Bearer {self.token}"}


class PayPalAuth(AuthProvider):
    """OAuth2 client-credentials with an in-memory token cache (thread-safe)."""

    def __init__(self, client_id: str, secret: str, base_url: str):
        self.client_id, self.secret, self.base_url = client_id, secret, base_url
        self._token: str | None = None
        self._expires = 0.0
        self._lock = threading.Lock()

    def headers(self, client, force_refresh=False):
        with self._lock:
            if force_refresh or not self._token or time.time() > self._expires - 60:
                basic = base64.b64encode(f"{self.client_id}:{self.secret}".encode()).decode()
                r = client.post(f"{self.base_url}/v1/oauth2/token", data={"grant_type": "client_credentials"},
                                headers={"Authorization": f"Basic {basic}"})
                r.raise_for_status()
                body = r.json()
                self._token, self._expires = body["access_token"], time.time() + int(body.get("expires_in", 3600))
        return {"Authorization": f"Bearer {self._token}"}


class HttpExecutor:
    def __init__(self, auth: dict[str, AuthProvider] | None = None, transport: httpx.BaseTransport | None = None,
                 timeout: float = 20.0, max_retries: int = 3, max_pages: int = 10,
                 sleep: Callable[[float], None] = time.sleep):
        self.auth = auth or {}
        self.client = httpx.Client(transport=transport, timeout=timeout)
        self.max_retries = max_retries
        self.max_pages = max_pages
        self.sleep = sleep

    # ---------------------------------------------------------------------------------
    def call(self, tool: ToolSpec, args: dict[str, Any], idempotency_key: str | None = None) -> CallResult:
        args = args or {}
        path_args = args.get("path") or {}
        query = {k: v for k, v in (args.get("query") or {}).items() if v is not None}
        body = args.get("body")

        path = tool.path
        for name, value in path_args.items():
            path = path.replace("{" + name + "}", quote(str(value), safe=""))
        if "{" in path:
            missing = [seg for seg in path.split("/") if seg.startswith("{")]
            return CallResult(False, None, error=f"missing path parameter(s): {', '.join(missing)}")

        url = tool.base_url.rstrip("/") + path
        headers = {"Accept": "application/json"}
        # A key only protects a write if the provider honours it on this endpoint (PayPal does on orders,
        # captures, refunds, payouts..., but not on invoicing). Where it doesn't, the write is sent once
        # and a 5xx / timeout is reported as "outcome unknown" instead of being replayed.
        keyed = bool(idempotency_key and tool.idempotency_header and tool.method != "GET")
        if keyed:
            headers[tool.idempotency_header] = idempotency_key
        req = {"method": tool.method, "url": url, "query": query, "body": body}

        result = self._send(tool, url, query, body, headers, idempotent=tool.method in ("GET", "PUT", "DELETE") or keyed)
        result.request = req
        if result.ok and tool.method == "GET":
            result = self._paginate(tool, url, query, headers, result)
        return result

    def _send(self, tool, url, query, body, headers, idempotent: bool) -> CallResult:
        auth = self.auth.get(tool.service, AuthProvider())
        attempts, refreshed, need_refresh, t0 = 0, False, False, time.perf_counter()
        while True:
            attempts += 1
            try:
                h = {**headers, **auth.headers(self.client, force_refresh=need_refresh)}
                need_refresh = False
                kwargs: dict[str, Any] = {"params": query, "headers": h}
                if body is not None and tool.method != "GET":
                    kwargs["json" if tool.body_encoding != "form" else "data"] = body
                r = self.client.request(tool.method, url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if idempotent and attempts <= self.max_retries:
                    self.sleep(self._backoff(attempts))
                    continue
                return CallResult(False, None, error=f"network error: {e}" + ("" if idempotent else UNKNOWN_OUTCOME),
                                  retryable=True, attempts=attempts, latency_ms=_ms(t0))

            if r.status_code == 401 and not refreshed:  # token expired or revoked -> refresh once
                refreshed = need_refresh = True
                continue
            # 429 means the request was turned away before it ran, so that one is always safe to retry
            if r.status_code in RETRYABLE and (idempotent or r.status_code == 429) and attempts <= self.max_retries:
                self.sleep(self._retry_after(r) or self._backoff(attempts))
                continue
            data = _json(r)
            if r.is_success:
                return CallResult(True, r.status_code, data, attempts=attempts, latency_ms=_ms(t0))
            unknown = UNKNOWN_OUTCOME if r.status_code >= 500 and not idempotent else ""
            return CallResult(False, r.status_code, data, error=describe_error(r.status_code, data) + unknown,
                              retryable=r.status_code in RETRYABLE, attempts=attempts, latency_ms=_ms(t0))

    def _paginate(self, tool: ToolSpec, url, query, headers, first: CallResult) -> CallResult:
        """PayPal-style page/total_pages pagination: fetch the rest and concatenate the item list."""
        data = first.data
        qprops = tool.parameters.get("properties", {}).get("query", {}).get("properties", {})
        if not isinstance(data, dict) or "page" not in qprops or int(data.get("total_pages") or 1) <= 1:
            return first
        list_key = max((k for k, v in data.items() if isinstance(v, list)), key=lambda k: len(data[k]), default=None)
        if not list_key:
            return first
        total = min(int(data["total_pages"]), self.max_pages)
        start = int(query.get("page", 1))
        fetched = 1
        for page in range(start + 1, total + 1):
            nxt = self._send(tool, url, {**query, "page": page}, None, headers, idempotent=True)
            if not nxt.ok:
                data["_pagination_warning"] = f"stopped at page {page - 1} of {data['total_pages']}: {nxt.error}"
                break
            data[list_key].extend(nxt.data.get(list_key, []))
            first.attempts += nxt.attempts
            fetched += 1
        if int(data["total_pages"]) > self.max_pages and "_pagination_warning" not in data:
            data["_pagination_warning"] = f"only the first {self.max_pages} of {data['total_pages']} pages fetched"
        data["_pages_fetched"] = fetched
        return first

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(8.0, 0.5 * 2 ** (attempt - 1)) + random.uniform(0, 0.25)

    @staticmethod
    def _retry_after(r: httpx.Response) -> float | None:
        try:
            return min(30.0, float(r.headers.get("Retry-After", "")))
        except ValueError:
            return None


def describe_error(status: int, data: Any) -> str:
    """PayPal errors look like {name, message, details:[{field, issue, description}]} - keep what's useful."""
    if isinstance(data, dict):
        parts = [str(data.get("name") or data.get("error") or ""), str(data.get("message") or data.get("error_description") or "")]
        for d in (data.get("details") or [])[:5]:
            if isinstance(d, dict):
                parts.append(f"{d.get('field', '')}: {d.get('issue', '')} {d.get('description', '')}".strip())
        msg = " | ".join(p for p in parts if p.strip())
    else:
        msg = str(data)[:300]
    hint = {400: "bad request - fix the arguments", 401: "not authenticated", 403: "not permitted for this account",
            404: "not found - the id is probably wrong", 409: "conflict with current state",
            422: "valid request but not allowed in the resource's current state", 429: "rate limited"}.get(status, "")
    return f"HTTP {status}{' (' + hint + ')' if hint else ''}: {msg}"[:800]


def _json(r: httpx.Response) -> Any:
    if not r.content:
        return {"status": r.status_code}
    try:
        return r.json()
    except ValueError:
        return {"text": r.text[:2000]}


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def build_executor(settings, catalog) -> HttpExecutor:
    """Wire auth per service. Mock mode swaps the network for the in-process fake PayPal."""
    from .mock_paypal import MockPayPal

    auth: dict[str, AuthProvider] = {}
    transport = None
    paypal = catalog.services.get("paypal")
    if paypal and settings.paypal_mode == "sandbox":
        auth["paypal"] = PayPalAuth(os.environ["PAYPAL_CLIENT_ID"], os.environ["PAYPAL_CLIENT_SECRET"], paypal.base_url)
    elif paypal:
        transport = MockPayPal().transport()
        auth["paypal"] = PayPalAuth("mock-id", "mock-secret", paypal.base_url)
    for svc in catalog.services:
        token = os.getenv(f"{svc.upper()}_TOKEN")
        if token and svc not in auth:
            auth[svc] = BearerTokenAuth(token)
    return HttpExecutor(auth, transport=transport, timeout=settings.http_timeout, max_retries=settings.http_retries)

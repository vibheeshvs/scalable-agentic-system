"""In-process fake of the PayPal endpoints used in the demo, plumbed in via httpx.MockTransport.

It exists so the whole agent (planning, confirmation, idempotency, pagination, error repair)
can be exercised without sandbox credentials. Response shapes follow the real API closely
enough for the agent to behave the same way; one simplification is that dispute list items
carry the buyer, which the real list endpoint doesn't (you'd need a follow-up GET per dispute).
Anything not modelled explicitly gets a generic 200/201 so every catalog tool is callable.
"""

from __future__ import annotations

import json
import random
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx


def _err(status: int, name: str, message: str, details: list[dict] | None = None) -> httpx.Response:
    return httpx.Response(status, json={"name": name, "message": message, "details": details or [],
                                        "debug_id": uuid.uuid4().hex[:12]})


class MockPayPal:
    def __init__(self, seed: int = 7, now: datetime | None = None, fail_plan: dict[str, list[int]] | None = None):
        self.now = now or datetime.now(timezone.utc)
        self.rng = random.Random(seed)
        self.invoices: dict[str, dict[str, Any]] = {}
        self.requests: list[httpx.Request] = []
        self.idempotency: dict[str, httpx.Response] = {}
        self.fail_plan = fail_plan or {}  # path-regex -> list of status codes to return first
        self.transactions = self._seed_transactions()
        self.disputes = self._seed_disputes()

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # ------------------------------------------------------------------ seed data
    def _seed_transactions(self) -> list[dict[str, Any]]:
        out = []
        start = self.now - timedelta(days=95)
        payers = ["ana@example.com", "bob@example.com", "user_123@example.com", "li.wei@example.com", "sam@example.com"]
        for i in range(140):
            ts = start + timedelta(hours=self.rng.randint(0, 95 * 24))
            refund = self.rng.random() < 0.08
            currency = "EUR" if self.rng.random() < 0.15 else "USD"
            value = round(self.rng.uniform(15, 400), 2) * (-1 if refund else 1)
            out.append({
                "transaction_info": {
                    "transaction_id": f"{self.rng.randint(10**16, 10**17 - 1):X}"[:17],
                    "transaction_event_code": "T1107" if refund else self.rng.choice(["T0006", "T0006", "T0003"]),
                    "transaction_initiation_date": ts.strftime("%Y-%m-%dT%H:%M:%S+0000"),
                    "transaction_amount": {"currency_code": currency, "value": f"{value:.2f}"},
                    "fee_amount": {"currency_code": currency, "value": f"{-abs(value) * 0.0349:.2f}"},
                    "transaction_status": "S" if self.rng.random() > 0.05 else "P",
                },
                "payer_info": {"email_address": self.rng.choice(payers)},
            })
        return sorted(out, key=lambda t: t["transaction_info"]["transaction_initiation_date"])

    def _seed_disputes(self) -> list[dict[str, Any]]:
        def d(i, buyer, status, state, amount, reason, days):
            return {"dispute_id": f"PP-D-{27800 + i}", "reason": reason, "status": status, "dispute_state": state,
                    "dispute_amount": {"currency_code": "USD", "value": amount},
                    "create_time": (self.now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "buyer": {"name": buyer, "email": f"{buyer}@example.com"}}
        return [
            d(3, "user_123", "WAITING_FOR_SELLER_RESPONSE", "REQUIRED_ACTION", "89.00", "MERCHANDISE_OR_SERVICE_NOT_RECEIVED", 3),
            d(4, "maria_g", "UNDER_REVIEW", "UNDER_PAYPAL_REVIEW", "240.00", "UNAUTHORISED", 12),
            d(5, "user_123", "RESOLVED", "RESOLVED", "35.50", "MERCHANDISE_OR_SERVICE_NOT_AS_DESCRIBED", 60),
        ]

    # ------------------------------------------------------------------ dispatcher
    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method
        for pattern, codes in self.fail_plan.items():
            if re.search(pattern, path) and codes:
                code = codes.pop(0)
                headers = {"Retry-After": "0"} if code == 429 else {}
                return httpx.Response(code, headers=headers, json={"name": "SERVICE_UNAVAILABLE" if code >= 500 else "RATE_LIMIT_REACHED", "message": "injected failure"})
        if path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "A21AA-mock-" + uuid.uuid4().hex[:8], "token_type": "Bearer", "expires_in": 32400})
        if not request.headers.get("Authorization", "").startswith("Bearer "):
            return _err(401, "AUTHENTICATION_FAILURE", "Authentication failed due to missing or invalid credentials.")

        key = request.headers.get("PayPal-Request-Id")
        if key and method == "POST" and key in self.idempotency:
            return self.idempotency[key]  # same key -> same result, no second side effect

        body = json.loads(request.content) if request.content else {}
        q = dict(request.url.params)
        resp = self._route(method, path, body, q)
        if key and method == "POST" and resp.is_success:
            self.idempotency[key] = resp
        return resp

    def _route(self, method: str, path: str, body: dict, q: dict) -> httpx.Response:
        m = re.fullmatch(r"/v2/invoicing/invoices/([^/]+)(?:/(send|remind|cancel))?", path)
        if path == "/v2/invoicing/invoices" and method == "POST":
            return self.create_invoice(body)
        if path == "/v2/invoicing/invoices" and method == "GET":
            return self._page(list(self.invoices.values()), "items", q, default_size=20)
        if path == "/v2/invoicing/search-invoices" and method == "POST":
            return self.search_invoices(body, q)
        if m:
            return self.invoice_action(method, m.group(1), m.group(2), body)
        if path == "/v1/reporting/transactions" and method == "GET":
            return self.transactions_search(q)
        if path == "/v1/reporting/balances" and method == "GET":
            return httpx.Response(200, json={"balances": [
                {"currency": "USD", "primary": True, "total_balance": {"currency_code": "USD", "value": "4821.37"},
                 "available_balance": {"currency_code": "USD", "value": "4521.37"}, "withheld_balance": {"currency_code": "USD", "value": "300.00"}},
                {"currency": "EUR", "total_balance": {"currency_code": "EUR", "value": "612.10"},
                 "available_balance": {"currency_code": "EUR", "value": "612.10"}}],
                "account_id": "MOCKACCT123", "as_of_time": self.now.isoformat()})
        if path == "/v1/customer/disputes" and method == "GET":
            items = self.disputes
            if q.get("dispute_state"):
                states = q["dispute_state"].split(",")
                items = [d for d in items if d["dispute_state"] in states]
            return httpx.Response(200, json={"items": items})
        m = re.fullmatch(r"/v1/customer/disputes/([^/]+)", path)
        if m and method == "GET":
            d = next((x for x in self.disputes if x["dispute_id"] == m.group(1)), None)
            return httpx.Response(200, json=d) if d else _err(404, "RESOURCE_NOT_FOUND", "The specified resource does not exist.")
        return self._generic(method, path, body)

    # ------------------------------------------------------------------ invoices
    def create_invoice(self, body: dict) -> httpx.Response:
        detail = body.get("detail") or {}
        if not detail.get("currency_code"):
            return _err(400, "INVALID_REQUEST", "Request is not well-formed, syntactically incorrect, or violates schema.",
                        [{"field": "/detail/currency_code", "location": "body", "issue": "MISSING_REQUIRED_PARAMETER",
                          "description": "A required field is missing."}])
        total = 0.0
        for it in body.get("items") or []:
            total += float(it.get("quantity", 1)) * float((it.get("unit_amount") or {}).get("value", 0))
        if not body.get("items") and body.get("amount"):
            total = float((body["amount"] or {}).get("value", 0))
        inv_id = "INV2-" + "-".join(uuid.uuid4().hex[i:i + 4].upper() for i in range(0, 20, 4))
        inv = {"id": inv_id, "status": "DRAFT", "detail": {**detail, "invoice_number": str(1000 + len(self.invoices))},
               "primary_recipients": body.get("primary_recipients", []), "items": body.get("items", []),
               "amount": {"currency_code": detail["currency_code"], "value": f"{total:.2f}"},
               "links": [{"href": f"https://api-m.sandbox.paypal.com/v2/invoicing/invoices/{inv_id}", "rel": "self"}]}
        self.invoices[inv_id] = inv
        return httpx.Response(201, json=inv)

    def invoice_action(self, method: str, inv_id: str, action: str | None, body: dict) -> httpx.Response:
        inv = self.invoices.get(inv_id)
        if not inv:
            return _err(404, "RESOURCE_NOT_FOUND", "The specified resource does not exist.",
                        [{"issue": "INVALID_RESOURCE_ID", "description": f"Invoice {inv_id} not found."}])
        if action is None and method == "GET":
            return httpx.Response(200, json=inv)
        if action is None and method == "DELETE":
            if inv["status"] != "DRAFT":
                return _err(422, "UNPROCESSABLE_ENTITY", "Only draft invoices can be deleted.")
            del self.invoices[inv_id]
            return httpx.Response(204)
        if action == "send":
            if inv["status"] != "DRAFT":
                return _err(422, "UNPROCESSABLE_ENTITY", "The requested action could not be performed.",
                            [{"issue": "INVALID_INVOICE_STATUS", "description": f"Invoice is {inv['status']}; only DRAFT invoices can be sent."}])
            inv["status"] = "SENT"
            return httpx.Response(200, json={"links": [{"href": f"https://www.sandbox.paypal.com/invoice/p/#{inv_id}", "rel": "payer-view", "method": "GET"}]})
        if action == "remind":
            return httpx.Response(204) if inv["status"] in ("SENT", "UNPAID", "PARTIALLY_PAID") else _err(422, "UNPROCESSABLE_ENTITY", "Reminders need a sent invoice.")
        if action == "cancel":
            inv["status"] = "CANCELLED"
            return httpx.Response(204)
        return self._generic(method, f"/v2/invoicing/invoices/{inv_id}/{action}", body)

    def search_invoices(self, body: dict, q: dict) -> httpx.Response:
        items = list(self.invoices.values())
        if body.get("recipient_email"):
            items = [i for i in items if any(r.get("billing_info", {}).get("email_address") == body["recipient_email"]
                                             for r in i["primary_recipients"])]
        if body.get("status"):
            items = [i for i in items if i["status"] in body["status"]]
        return self._page(items, "items", q, default_size=20)

    # ------------------------------------------------------------------ reporting
    def transactions_search(self, q: dict) -> httpx.Response:
        if not q.get("start_date") or not q.get("end_date"):
            return _err(400, "INVALID_REQUEST", "Request is not well-formed, syntactically incorrect, or violates schema.",
                        [{"field": "start_date" if not q.get("start_date") else "end_date", "issue": "MISSING_REQUIRED_PARAMETER"}])
        try:
            start, end = _parse_dt(q["start_date"]), _parse_dt(q["end_date"])
        except ValueError:
            return _err(400, "INVALID_REQUEST", "Dates must be in RFC 3339 format, e.g. 2026-08-01T00:00:00-0000",
                        [{"field": "start_date", "issue": "INVALID_PARAMETER_SYNTAX"}])
        if end - start > timedelta(days=31, hours=1) or end < start:
            return _err(400, "INVALID_REQUEST", "The date range must be 31 days or less.",
                        [{"field": "end_date", "issue": "INVALID_DATE_RANGE", "description": "Maximum supported range is 31 days."}])
        items = [t for t in self.transactions if start <= _parse_dt(t["transaction_info"]["transaction_initiation_date"]) <= end]
        if q.get("transaction_status"):
            items = [t for t in items if t["transaction_info"]["transaction_status"] == q["transaction_status"]]
        resp = self._page(items, "transaction_details", q, default_size=20)
        payload = json.loads(resp.content)
        payload.update({"account_number": "MOCKACCT123", "start_date": q["start_date"], "end_date": q["end_date"]})
        return httpx.Response(200, json=payload)

    # ------------------------------------------------------------------ helpers
    def _page(self, items: list, key: str, q: dict, default_size: int) -> httpx.Response:
        size = int(q.get("page_size", default_size))
        page = int(q.get("page", 1))
        total_pages = max(1, -(-len(items) // size))
        chunk = items[(page - 1) * size: page * size]
        return httpx.Response(200, json={key: chunk, "page": page, "total_items": len(items), "total_pages": total_pages,
                                         "links": [{"href": "...", "rel": "next"}] if page < total_pages else []})

    def _generic(self, method: str, path: str, body: dict) -> httpx.Response:
        if method == "GET":
            return httpx.Response(200, json={"items": [], "mock": True, "note": f"generic mock response for {path}"})
        if method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(201 if method == "POST" else 200,
                              json={"id": "MOCK-" + uuid.uuid4().hex[:10].upper(), "status": "COMPLETED" if method == "POST" else "UPDATED",
                                    "mock": True, "path": path, "echo": body})


def _parse_dt(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

from agent.tools.http import CallResult, HttpExecutor, PayPalAuth
from agent.tools.mock_paypal import MockPayPal

from conftest import NOW


def _executor(mock, catalog):
    return HttpExecutor({"paypal": PayPalAuth("id", "secret", catalog.services["paypal"].base_url)},
                        transport=mock.transport(), sleep=lambda s: None)


def test_retries_transient_errors(paypal_catalog):
    mock = MockPayPal(now=NOW, fail_plan={r"/v1/reporting/balances": [503, 429]})
    res = _executor(mock, paypal_catalog).call(paypal_catalog.get("paypal.balances.get"), {})
    assert res.ok and res.attempts == 3


def test_does_not_blindly_retry_non_idempotent_writes(paypal_catalog):
    mock = MockPayPal(now=NOW, fail_plan={r"/v1/payments/payouts": [503]})
    tool = paypal_catalog.get("paypal.payouts.post")
    res = _executor(mock, paypal_catalog).call(tool, {"body": {"sender_batch_header": {}, "items": []}}, idempotency_key=None)
    assert not res.ok and res.status == 503 and res.attempts == 1


def test_idempotency_key_prevents_duplicate_side_effects(paypal_catalog):
    mock = MockPayPal(now=NOW)
    ex = _executor(mock, paypal_catalog)
    tool = paypal_catalog.get("paypal.orders.create")   # PayPal documents PayPal-Request-Id for this endpoint
    body = {"body": {"intent": "CAPTURE", "purchase_units": [{"amount": {"currency_code": "USD", "value": "50.00"}}]}}
    a = ex.call(tool, body, idempotency_key="key-1")
    b = ex.call(tool, body, idempotency_key="key-1")   # e.g. a retry after a crash / resume
    assert a.ok and b.ok and a.data["id"] == b.data["id"]
    assert ex.call(tool, body, idempotency_key="key-2").data["id"] != a.data["id"]
    assert mock.requests[-1].headers["PayPal-Request-Id"] == "key-2"


def test_keyed_write_is_retried_but_unkeyed_write_is_not(paypal_catalog):
    # refunds accept an idempotency key -> a 503 is safe to retry with the same key
    mock = MockPayPal(now=NOW, fail_plan={r"/refund$": [503]})
    res = _executor(mock, paypal_catalog).call(paypal_catalog.get("paypal.captures.refund"),
                                               {"path": {"capture_id": "CAP-1"}, "body": {}}, idempotency_key="k")
    assert res.ok and res.attempts == 2

    # invoicing has no idempotency header in PayPal's spec: a key can't protect it, so no header and no replay
    create = paypal_catalog.get("paypal.invoices.create")
    assert create.idempotency_header is None
    mock = MockPayPal(now=NOW, fail_plan={r"/v2/invoicing/invoices$": [503]})
    res = _executor(mock, paypal_catalog).call(create, {"body": {"detail": {"currency_code": "USD"}}}, idempotency_key="k")
    assert not res.ok and res.attempts == 1 and "may or may not have gone through" in res.error
    assert "PayPal-Request-Id" not in mock.requests[-1].headers and not mock.invoices

    # a 429 was rejected before it ran, so even an unkeyed write can be retried
    mock = MockPayPal(now=NOW, fail_plan={r"/v2/invoicing/invoices$": [429]})
    res = _executor(mock, paypal_catalog).call(create, {"body": {"detail": {"currency_code": "USD"}}}, idempotency_key="k")
    assert res.ok and res.attempts == 2 and len(mock.invoices) == 1


def test_partial_pagination_is_flagged_not_hidden(paypal_catalog):
    ex = _executor(MockPayPal(now=NOW), paypal_catalog)
    real_send = ex._send

    def later_pages_fail(tool, url, query, *args, **kwargs):   # page 1 succeeds, every follow-up page errors
        if "page" in query:
            return CallResult(False, 500, error="HTTP 500: injected")
        return real_send(tool, url, query, *args, **kwargs)

    ex._send = later_pages_fail
    res = ex.call(paypal_catalog.get("paypal.search.get"),
                  {"query": {"start_date": "2026-08-01T00:00:00-0000", "end_date": "2026-08-31T23:59:59-0000"}})
    assert res.ok and res.data["total_pages"] > 1
    assert res.data["_pages_fetched"] == 1 and "stopped at page 1" in res.data["_pagination_warning"]


def test_refreshes_token_on_401(paypal_catalog):
    mock = MockPayPal(now=NOW, fail_plan={r"/v1/reporting/balances": [401]})
    res = _executor(mock, paypal_catalog).call(paypal_catalog.get("paypal.balances.get"), {})
    assert res.ok
    assert sum(r.url.path == "/v1/oauth2/token" for r in mock.requests) == 2


def test_error_messages_are_actionable(paypal_catalog):
    res = _executor(MockPayPal(now=NOW), paypal_catalog).call(
        paypal_catalog.get("paypal.invoices.get"), {"path": {"invoice_id": "INV2-NOPE"}})
    assert not res.ok and res.status == 404
    assert "HTTP 404" in res.error and "RESOURCE_NOT_FOUND" in res.error


def test_path_params_are_url_encoded(paypal_catalog):
    mock = MockPayPal(now=NOW)
    _executor(mock, paypal_catalog).call(paypal_catalog.get("paypal.invoices.get"), {"path": {"invoice_id": "a/b c"}})
    assert mock.requests[-1].url.raw_path.endswith(b"/invoices/a%2Fb%20c")

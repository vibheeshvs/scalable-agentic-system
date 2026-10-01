import json
from pathlib import Path

from agent.registry import Risk, load_postman
from agent.registry.risk import classify

FIXTURES = Path(__file__).parent / "fixtures"


def test_paypal_catalog_has_all_operations(paypal_catalog):
    # 115 operations in PayPal's 13 published OpenAPI specs, minus 3 disabled in data/policy/tool_policy.json
    assert len(paypal_catalog) == 112
    assert "paypal.server.callback" not in paypal_catalog.tools
    assert paypal_catalog.get("paypal.disputes.provide-supporting-info").risk == Risk.HIGH
    names = [t.name for t in paypal_catalog.tools.values()]
    assert len(set(names)) == len(names) and all(len(n) <= 64 for n in names)
    assert all(t.base_url == "https://api-m.sandbox.paypal.com" for t in paypal_catalog.tools.values())


def test_risk_levels(paypal_catalog):
    risk = {tid: t.risk for tid, t in paypal_catalog.tools.items()}
    assert risk["paypal.invoices.list"] == Risk.READ
    assert risk["paypal.invoices.search-invoices"] == Risk.READ   # POST, but a search
    assert risk["paypal.invoices.create"] == Risk.WRITE           # draft only
    for high in ("paypal.invoices.send", "paypal.captures.refund", "paypal.payouts.post", "paypal.invoices.delete",
                 "paypal.disputes.accept-claim", "paypal.subscriptions.cancel", "paypal.orders.capture"):
        assert risk[high] == Risk.HIGH, high


def test_schemas_are_compacted_but_keep_what_matters(paypal_catalog):
    create = paypal_catalog.get("paypal.invoices.create")
    body = create.parameters["properties"]["body"]
    assert "detail" in body["required"]
    item = body["properties"]["items"]["items"]["properties"]
    assert set(item["unit_amount"]["properties"]) >= {"currency_code", "value"}
    assert "email_address" in body["properties"]["primary_recipients"]["items"]["properties"]["billing_info"]["properties"]
    assert len(json.dumps(create.to_llm_tool())) < 16_000
    send = paypal_catalog.get("paypal.invoices.send")
    assert send.parameters["properties"]["path"]["required"] == ["invoice_id"]


def test_postman_loader_infers_params():
    tools, svc = load_postman(FIXTURES / "mini.postman_collection.json", service="paypal", base_url="https://api-m.sandbox.paypal.com")
    by = {t.summary: t for t in tools}
    assert svc.title == "Mini PayPal (fixture)" and len(tools) == 4

    create = by["Create draft invoice"]
    assert create.method == "POST" and create.path == "/v2/invoicing/invoices" and create.group == "invoices"
    body = create.parameters["properties"]["body"]
    assert body["properties"]["items"]["items"]["properties"]["unit_amount"]["properties"]["value"]["type"] == "number"

    send = by["Send invoice"]
    assert send.path == "/v2/invoicing/invoices/{invoice_id}/send"
    assert send.parameters["properties"]["path"]["required"] == ["invoice_id"]
    assert send.risk == Risk.HIGH

    assert by["Show invoice details"].path == "/v2/invoicing/invoices/{invoice_id}"
    disputes = by["List disputes"]
    assert set(disputes.parameters["properties"]["query"]["properties"]) == {"dispute_state", "page_size"}  # disabled one dropped
    assert disputes.risk == Risk.READ


def test_postman_loader_survives_real_world_collections():
    def req(name, method, path, **extra):
        return {"name": name, "request": {"method": method, "url": {"raw": "{{base_url}}/" + "/".join(path), "path": path}, **extra}}

    tools, _ = load_postman({"info": {"name": "Messy"}, "item": [
        {"name": "Authorization", "item": [req("Generate access_token", "POST", ["v1", "oauth2", "token"])]},
        {"name": "Invoices", "item": [
            req("Invoice", "GET", ["v2", "invoicing", "invoices", ":id"]),       # same name, three verbs
            req("Invoice", "DELETE", ["v2", "invoicing", "invoices", ":id"]),
            req("Invoice", "DELETE", ["v2", "invoicing", "invoices", ":id"]),    # and a saved duplicate
            {"name": "Empty placeholder", "request": {"method": "GET"}},
            {"name": "List", "request": {"method": "GET", "url": {"raw": "{{base_url}}/v2/invoicing/invoices",
                                                                  "path": "v2/invoicing/invoices"}}},
        ]}]}, service="paypal")
    assert [t.path for t in tools] == ["/v2/invoicing/invoices/{id}"] * 3 + ["/v2/invoicing/invoices"]  # no token tool
    ids, names = [t.id for t in tools], [t.name for t in tools]
    assert len(set(ids)) == 4 and len(set(names)) == 4     # duplicate function names would be rejected by the LLM API
    assert [t.risk for t in tools] == [Risk.READ, Risk.HIGH, Risk.HIGH, Risk.READ]


def test_idempotency_support_is_read_from_the_spec(paypal_catalog):
    # PayPal only documents PayPal-Request-Id on some writes; the executor relies on this to decide what is safe to retry
    header = {tid: t.idempotency_header for tid, t in paypal_catalog.tools.items()}
    for tid in ("paypal.orders.create", "paypal.orders.capture", "paypal.captures.refund", "paypal.payouts.post"):
        assert header[tid] == "PayPal-Request-Id", tid
    for tid in ("paypal.invoices.create", "paypal.invoices.send", "paypal.disputes.send-message", "paypal.invoices.list"):
        assert header[tid] is None, tid
    assert sum(h is not None for h in header.values()) == 15


def test_classify_generic_apis():
    assert classify("POST", "/v1/charges", "Create a charge", "PostCharges") == Risk.HIGH
    assert classify("POST", "/v1/customers", "Create a customer", "PostCustomers") == Risk.WRITE
    assert classify("GET", "/v1/refunds/{id}", "Retrieve a refund") == Risk.READ
    assert classify("DELETE", "/v1/products/{id}", "Delete a product") == Risk.HIGH

"""End-to-end graph tests: real graph, real retrieval, real executor + mock PayPal; only the LLM is scripted."""

import re

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.llm import LLMError
from agent.prompts import PlanOut, Route
from agent.tools.mock_paypal import _parse_dt

CFG = {"configurable": {"thread_id": "t1"}}
INVOICE_BODY = {"body": {
    "detail": {"currency_code": "USD"},
    "primary_recipients": [{"billing_info": {"email_address": "vibheesh@example.com"}}],
    "items": [{"name": "Consulting", "quantity": "1", "unit_amount": {"currency_code": "USD", "value": "50.00"}}]}}


def _send_created_invoice(messages, tools):
    invoice_id = re.search(r"INV2-[A-Z0-9-]+", messages[-1].content).group(0)  # taken from step 1's result
    return ("paypal__invoices_send", {"path": {"invoice_id": invoice_id}, "body": {"send_to_invoicer": True}})


def invoice_script(responder="Invoice sent."):
    return {
        "router": [Route(intent="action", request="Send an invoice for $50 to vibheesh@example.com", services=["paypal"])],
        "planner": [PlanOut(steps=["Create a draft invoice for $50 USD billed to vibheesh@example.com",
                                   "Send the invoice created in step 1 to the recipient"])],
        "selector": [("paypal__invoices_create", INVOICE_BODY), _send_created_invoice],
        "responder": [responder],
    }


def ask(graph, text, cfg=CFG):
    return graph.invoke({"messages": [HumanMessage(text)]}, cfg)


def test_send_invoice_asks_for_confirmation_then_sends(make_agent, mock_paypal):
    graph, ctx, llm = make_agent(invoice_script())
    out = ask(graph, "Send an invoice for $50 to vibheesh@example.com")

    # creating the draft is a low-risk write -> no confirmation; sending it is high-risk -> pause
    pending = out["__interrupt__"][0].value
    assert pending["tool"] == "paypal.invoices.send" and pending["risk"] == "high"
    assert list(mock_paypal.invoices.values())[0]["status"] == "DRAFT"

    out = graph.invoke(Command(resume={"action": "approve"}), CFG)
    assert out["outcome"] == "succeeded"
    assert out["messages"][-1].content == "Invoice sent."
    assert [inv["status"] for inv in mock_paypal.invoices.values()] == ["SENT"]

    # the LLM never saw more than top_k retrieved tools + 4 built-ins, and the right one was among them
    selector_calls = [extra for name, _, extra in llm.calls if name == "selector"]
    assert all(len(tools) <= ctx.settings.top_k + 4 for tools in selector_calls)
    assert "paypal__invoices_create" in selector_calls[0] and "paypal__invoices_send" in selector_calls[1]

    # each write went out exactly once. PayPal's invoicing API has no idempotency header (unlike orders,
    # refunds, payouts), so none is sent here and these calls are never auto-retried (see test_executor.py)
    writes = [r for r in mock_paypal.requests if r.method == "POST" and "oauth2" not in r.url.path]
    assert [r.url.path.rsplit("/", 1)[-1] for r in writes] == ["invoices", "send"]
    assert not any(r.headers.get("PayPal-Request-Id") for r in writes)
    assert all(s["idem_key"] for s in out["plan"])  # the key is still decided and checkpointed before each call


def test_declined_confirmation_does_not_send(make_agent, mock_paypal):
    graph, _, _ = make_agent(invoice_script(responder="OK, I left the invoice as a draft."))
    ask(graph, "Send an invoice for $50 to vibheesh@example.com")
    out = graph.invoke(Command(resume={"action": "reject", "reason": "wrong amount"}), CFG)
    assert out["outcome"] == "cancelled"
    assert [s["status"] for s in out["plan"]] == ["done", "cancelled"]
    assert [inv["status"] for inv in mock_paypal.invoices.values()] == ["DRAFT"]
    assert not any(r.url.path.endswith("/send") for r in mock_paypal.requests)


def test_sales_volume_paginates_and_computes_deterministically(make_agent, mock_paypal):
    start, end = "2026-08-01T00:00:00-0000", "2026-08-31T23:59:59-0000"
    graph, _, llm = make_agent({
        "router": [Route(intent="action", request="What was my total sales volume last month?")],
        "planner": [PlanOut(steps=[f"List transactions between {start} and {end}",
                                   {"goal": "Sum completed incoming transaction amounts from step 1 per currency",
                                    "kind": "compute"}])],
        "selector": [
            ("paypal__search_get", {"query": {"start_date": start, "end_date": end}}),
            ("analyze_data", {"step": 1, "items_path": "transaction_details", "op": "sum",
                              "value_path": "transaction_info.transaction_amount.value",
                              "where": [{"path": "transaction_info.transaction_status", "value": "S"},
                                        {"path": "transaction_info.transaction_amount.value", "op": "gt", "value": "0"}],
                              "group_by": "transaction_info.transaction_amount.currency_code"}),
        ],
        "responder": ["Sales volume computed."],
    })
    out = ask(graph, "What was my total sales volume last month?")
    assert out["outcome"] == "succeeded" and "__interrupt__" not in out  # read-only, no confirmation

    txs = out["results"]["1"]["transaction_details"]
    in_aug = [t for t in mock_paypal.transactions
              if _parse_dt(start) <= _parse_dt(t["transaction_info"]["transaction_initiation_date"]) <= _parse_dt(end)]
    assert len(txs) == len(in_aug) > 20            # more than one page (page size 20) -> pagination worked

    expected = {}
    for t in in_aug:
        info = t["transaction_info"]
        v = float(info["transaction_amount"]["value"])
        if info["transaction_status"] == "S" and v > 0:
            cur = info["transaction_amount"]["currency_code"]
            expected[cur] = round(expected.get(cur, 0) + v, 2)
    assert out["results"]["2"]["result"] == expected

    # A computation step offers only analyze_data (and the way out). In the first live run the model answered this
    # step with rag_search ("how is sales volume defined?"), the step counted as done, and no total was computed.
    offered = [tools for name, _, tools in llm.calls if name == "selector"]
    assert len(offered[0]) == 12 and offered[1] == ["analyze_data", "ask_user"]


def test_validation_error_is_repaired(make_agent):
    graph, _, llm = make_agent({
        "router": [Route(intent="action", request="show transactions from August")],
        "planner": [PlanOut(steps=["List transactions between 2026-08-01 and 2026-08-31"])],
        "selector": [
            ("paypal__search_get", {"query": {"transaction_status": "S"}}),   # forgot the required dates
            ("paypal__search_get", {"query": {"start_date": "2026-08-01T00:00:00-0000", "end_date": "2026-08-31T23:59:59-0000"}}),
        ],
        "responder": ["Here they are."],
    })
    out = ask(graph, "show transactions from August")
    assert out["outcome"] == "succeeded"
    retry_prompt = [m for name, m, _ in llm.calls if name == "selector"][1][-1].content
    assert "previous attempt" in retry_prompt and "start_date" in retry_prompt


def test_api_error_is_fed_back_and_repaired(make_agent):
    graph, _, llm = make_agent({
        "router": [Route(intent="action", request="transactions for the last quarter")],
        "planner": [PlanOut(steps=["List transactions between 2026-06-01 and 2026-08-31"])],
        "selector": [
            ("paypal__search_get", {"query": {"start_date": "2026-06-01T00:00:00-0000", "end_date": "2026-08-31T23:59:59-0000"}}),
            ("paypal__search_get", {"query": {"start_date": "2026-08-01T00:00:00-0000", "end_date": "2026-08-31T23:59:59-0000"}}),
        ],
        "responder": ["PayPal only allows 31 days per query, so here is August."],
    })
    out = ask(graph, "transactions for the last quarter")
    assert out["outcome"] == "succeeded"
    retry_prompt = [m for name, m, _ in llm.calls if name == "selector"][1][-1].content
    assert "INVALID_DATE_RANGE" in retry_prompt or "31 days" in retry_prompt


def test_invented_ids_are_blocked_and_agent_asks_instead(make_agent, mock_paypal):
    graph, _, _ = make_agent({
        "router": [Route(intent="action", request="send my latest invoice")],
        "planner": [PlanOut(steps=["Send the latest draft invoice"])],
        "selector": [
            ("paypal__invoices_send", {"path": {"invoice_id": "INV2-MADE-UP1-2345"}}),   # hallucinated id
            ("ask_user", {"question": "Which invoice should I send? I can list your drafts if that helps."}),
        ],
    })
    out = ask(graph, "send my latest invoice")
    assert out["outcome"] == "needs_input"
    assert out["messages"][-1].content.startswith("Which invoice")
    assert not any("/send" in r.url.path for r in mock_paypal.requests)


def test_knowledge_question_uses_rag_tool_with_citations(make_agent):
    graph, _, llm = make_agent({
        "router": [Route(intent="knowledge", request="When do we add a late fee to an invoice?")],
        "rag_generate": ["After 30 days overdue, 1.5% - only if the contract has the clause "
                         "[invoicing-playbook.md > Invoicing playbook / Late fees]."],
    })
    out = ask(graph, "When do we add a late fee?")
    answer = out["messages"][-1].content
    assert "1.5%" in answer and "Sources:" in answer and "invoicing-playbook.md" in answer
    rag_prompt = next(m for name, m, _ in llm.calls if name == "rag_generate")[-1].content
    assert "Late fees" in rag_prompt  # the right chunk was retrieved into the prompt


def test_system_search_capabilities_and_activity(make_agent):
    script = invoice_script()
    script["router"].append(Route(intent="system", request="What tools are available for managing invoices?"))
    script["router"].append(Route(intent="system", request="What's the status of my last request?"))
    script["responder"] += ["Here are the invoice tools.", "Your last request succeeded."]
    graph, _, _ = make_agent(script)

    ask(graph, "Send an invoice for $50 to vibheesh@example.com")
    graph.invoke(Command(resume=True), CFG)

    caps = ask(graph, "What tools are available for managing invoices?")["results"]["1"]
    assert caps["scope"] == "capabilities"
    assert any("invoices.send" in t for t in caps["matching_tools"])
    assert any("invoices.create" in t for t in caps["matching_tools"])

    activity = ask(graph, "What's the status of my last request?")["results"]["1"]
    assert activity["scope"] == "activity"
    last = activity["recent_requests"][0]  # most recent *other* request (the capabilities question)
    assert last["request"].startswith("What tools")
    invoice_run = activity["recent_requests"][1]
    assert invoice_run["status"] == "succeeded"
    assert invoice_run["api_calls"] == ["paypal.invoices.create -> ok (HTTP 201)", "paypal.invoices.send -> ok (HTTP 200)"]


def test_chat_short_circuits(make_agent):
    graph, _, llm = make_agent({"router": [Route(intent="chat", request="hi", reply="Hi! What can I do for you?")]})
    out = ask(graph, "hi")
    assert out["messages"][-1].content == "Hi! What can I do for you?"
    assert [name for name, _, _ in llm.calls] == ["router"]  # one cheap call, nothing else


def test_disabled_service_tools_are_rejected(make_agent):
    graph, _, _ = make_agent({
        "router": [Route(intent="action", request="list disputes")],
        "planner": [PlanOut(steps=["List open disputes"])],
        "selector": [("paypal__disputes_list", {})] * 3,
        "responder": ["I can't reach PayPal for this account."],
    }, enabled_services=["stripe"])
    out = ask(graph, "list disputes")
    assert out["outcome"] == "failed"
    assert "not connected" in out["plan"][0]["error"]


def _provider_down(messages, extra):
    raise LLMError("429 RESOURCE_EXHAUSTED: quota exceeded")


def test_model_failure_ends_the_turn_cleanly(make_agent, mock_paypal):
    """A provider error (quota, overload) must not crash the turn or leave the thread half-written."""
    script = invoice_script()
    script["router"].insert(0, _provider_down)
    graph, ctx, _ = make_agent(script)

    out = ask(graph, "Send an invoice for $50 to vibheesh@example.com")
    assert out["outcome"] == "failed" and "model call failed" in out["messages"][-1].content
    assert not mock_paypal.invoices                                   # nothing was done
    assert ctx.runlog.recent_runs("t1")[0]["status"] == "failed"      # not stuck on "running"

    out = ask(graph, "Send an invoice for $50 to vibheesh@example.com")   # the same thread still works afterwards
    assert out["__interrupt__"][0].value["tool"] == "paypal.invoices.send"


def test_model_failure_mid_plan_reports_what_already_happened(make_agent, mock_paypal):
    script = invoice_script()
    script["selector"][1] = _provider_down      # step 1 created the draft, then the model went away
    script["responder"] = [_provider_down]
    graph, _, _ = make_agent(script)

    out = ask(graph, "Send an invoice for $50 to vibheesh@example.com")
    assert out["outcome"] == "partial" and [s["status"] for s in out["plan"]] == ["done", "failed"]
    reply = out["messages"][-1].content           # written without the model, straight from the step statuses
    assert "[done] Create a draft invoice" in reply and "[failed] Send the invoice" in reply
    assert [inv["status"] for inv in mock_paypal.invoices.values()] == ["DRAFT"]


def test_open_dispute_lookup_for_a_buyer(make_agent):
    graph, _, llm = make_agent({
        "router": [Route(intent="action", request="Is there a dispute open from user_123?")],
        "planner": [PlanOut(steps=["List disputes that are still open (not resolved)",
                                   {"goal": "From step 1, list the dispute ids whose buyer is user_123", "kind": "compute"}])],
        "selector": [
            ("paypal__disputes_list", {"query": {"dispute_state": "REQUIRED_ACTION,UNDER_PAYPAL_REVIEW"}}),
            ("analyze_data", {"step": 1, "items_path": "items", "op": "list", "value_path": "dispute_id",
                              "where": [{"path": "buyer.name", "value": "user_123"}]}),
        ],
        "responder": ["Yes - PP-D-27803 ($89.00, item not received) is waiting for your response."],
    })
    out = ask(graph, "Is there a dispute open from user_123?")
    assert out["outcome"] == "succeeded"
    assert out["results"]["2"]["result"] == ["PP-D-27803"]   # the resolved one from user_123 is excluded
    assert "paypal__disputes_list" in [extra for name, _, extra in llm.calls if name == "selector"][0]


def test_live_check_harness_with_scripted_model(make_agent):
    """Mechanics of scripts/live_check.py (the real thing needs an API key)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import live_check

    graph, _, _ = make_agent(invoice_script(responder="Invoice INV2 sent to vibheesh@example.com."))
    r = live_check.run_one(graph, live_check.SCENARIOS[0], thread="lc")
    assert r["ok"], r["problems"]
    assert r["approvals"] == ["paypal.invoices.send"]
    assert "PASS" in live_check.report([r], "scripted")

    graph, _, _ = make_agent({"router": [Route(intent="chat", request="hi", reply="Hello")]})
    r = live_check.run_one(graph, live_check.SCENARIOS[5], thread="lc2")
    assert not r["ok"] and "builtin.rag_search did not run" in r["problems"]

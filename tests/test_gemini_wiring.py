"""The real Gemini adapter (langchain-google-genai + our ChatLLM), with only the network call faked.

The scripted-LLM tests prove our own logic; this one proves the provider plumbing: tools go to
Gemini as JSON Schema with function-calling mode ANY, optional fields stay optional, and
Gemini's replies (function calls, JSON, text) come back in the shapes the graph expects.
"""

import json
import re
from datetime import date

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

pytest.importorskip("langchain_google_genai")
from google.genai import types  # noqa: E402

from agent.config import Settings  # noqa: E402
from agent.graph.build import build_agent  # noqa: E402
from agent.llm import ChatLLM  # noqa: E402
from agent.runlog import RunLog  # noqa: E402


class FakeGemini:
    """Stands in for client.models.generate_content and answers like Gemini would."""

    def __init__(self):
        self.requests = []
        self.selector_calls = 0

    def __call__(self, **kw):
        self.requests.append(kw)
        cfg = kw["config"]
        prompt = " ".join(p.text or "" for c in kw["contents"] for p in (c.parts or []))
        if cfg.tools:  # selector: a function call
            self.selector_calls += 1
            if self.selector_calls == 1:
                part = types.Part(function_call=types.FunctionCall(name="paypal__invoices_create", args={"body": {
                    "detail": {"currency_code": "USD"},
                    "primary_recipients": [{"billing_info": {"email_address": "vibheesh@example.com"}}],
                    "items": [{"name": "Consulting", "quantity": "1",
                               "unit_amount": {"currency_code": "USD", "value": "50.00"}}]}}))
            else:
                invoice_id = re.search(r"INV2-[A-Z0-9-]+", prompt).group(0)
                part = types.Part(function_call=types.FunctionCall(
                    name="paypal__invoices_send", args={"path": {"invoice_id": invoice_id}}))
        elif cfg.response_json_schema or cfg.response_schema:  # structured output
            schema = json.dumps(cfg.response_json_schema or {})
            if '"intent"' in schema:
                payload = {"intent": "action", "request": "Send an invoice for $50 to vibheesh@example.com",
                           "services": ["paypal"]}
            else:
                payload = {"steps": ["Create a draft invoice for $50 USD billed to vibheesh@example.com",
                                     "Send the invoice created in step 1"]}
            part = types.Part(text=json.dumps(payload))
        else:  # responder: plain text
            part = types.Part(text="Invoice sent to vibheesh@example.com.")
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[part]), finish_reason="STOP")],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100, candidates_token_count=20, total_token_count=120))


def _required_fields_present(schema):
    """Every name in a `required` list still has its property, and no object was left without its fields."""
    if not isinstance(schema, dict):
        return True
    props = schema.get("properties")
    if "properties" in schema and not props:
        return False
    if any(r not in (props or {}) for r in schema.get("required") or []):
        return False
    return all(_required_fields_present(c) for c in [*(props or {}).values(), schema.get("items")])


def test_tool_schemas_are_fitted_to_what_gemini_accepts(paypal_catalog):
    """Gemini refuses a forced function call when the schemas are too big (measured live: one function over
    ~270 nodes, or a request over ~450 nodes + enum values). About a quarter of the eval queries retrieve a
    top-8 that is over one of those limits."""
    from agent.llm import (GEMINI_MAX_SCHEMA_NODES, GEMINI_REQUEST_BUDGET, fit_alone, fit_request, schema_cost,
                           schema_nodes)
    from agent.tools.builtins import BUILTINS

    # the candidate list of the invoice step that failed in the first live run, in retrieval order
    ranked = ["paypal.invoices.create", "paypal.invoices.delete", "paypal.invoices.send", "paypal.templates.create",
              "paypal.invoices.update", "paypal.invoices.cancel"]
    builtins = [b.parameters for b in BUILTINS.values()]
    schemas = [paypal_catalog.get(t).parameters for t in ranked] + builtins
    assert sum(map(schema_cost, schemas)) > GEMINI_REQUEST_BUDGET
    before = json.dumps(schemas)

    fitted = fit_request(schemas)
    assert json.dumps(schemas) == before                                   # the catalog itself is never changed
    assert sum(map(schema_cost, fitted)) <= GEMINI_REQUEST_BUDGET
    assert fitted[0] == schemas[0]                                         # the top-ranked tool keeps everything
    assert fitted[4] != schemas[4]                                         # room is made lower down the ranking
    assert fitted[-4:] == builtins                                         # built-ins are never shortened
    assert all(_required_fields_present(s) for s in fitted)
    unit_amount = fitted[0]["properties"]["body"]["properties"]["items"]["items"]["properties"]["unit_amount"]
    assert set(unit_amount["properties"]) == {"currency_code", "value"}    # the field the first live run lost

    # a single schema over the per-function limit: optional detail goes, the required skeleton stays
    full = paypal_catalog.get("paypal.orders.create").parameters
    solo = fit_alone(full)
    assert schema_nodes(full) > GEMINI_MAX_SCHEMA_NODES >= schema_nodes(solo) and _required_fields_present(solo)
    body = solo["properties"]["body"]
    assert body["required"] == ["intent", "purchase_units"]
    assert set(body["properties"]["purchase_units"]["items"]["properties"]["amount"]["properties"]) >= {"currency_code", "value"}


def test_a_shortened_tool_is_asked_again_with_its_full_schema(monkeypatch, paypal_catalog):
    """If the model picks a tool whose schema was shortened to fit the request, it gets a second call with that
    tool alone (plus ask_user), so it can fill in the fields it could not see the first time."""
    from agent.llm import fit_alone
    from agent.tools.builtins import BUILTINS

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    llm = ChatLLM("google_genai:gemini-3.8-flash")
    seen = []

    def fake(**kw):
        decls = kw["config"].tools[0].function_declarations
        seen.append({d.name: d.parameters_json_schema for d in decls})
        part = types.Part(function_call=types.FunctionCall(name="paypal__invoices_update", args={"path": {"invoice_id": "INV2-1"}}))
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model", parts=[part]), finish_reason="STOP")])

    llm.model.client.models.generate_content = fake
    ranked = ["paypal.invoices.create", "paypal.invoices.delete", "paypal.invoices.send", "paypal.templates.create",
              "paypal.invoices.update", "paypal.invoices.cancel"]
    tools = [paypal_catalog.get(t).to_llm_tool() for t in ranked] + [b.to_llm_tool() for b in BUILTINS.values()]
    ai = llm.call_tools([HumanMessage("change the note on invoice INV2-1")], tools, name="selector")

    assert ai.tool_calls[0]["name"] == "paypal__invoices_update" and len(seen) == 2
    full = paypal_catalog.get("paypal.invoices.update").parameters
    assert seen[0]["paypal__invoices_update"] != full                      # first call: shortened to make room
    assert list(seen[1]) == ["paypal__invoices_update", "ask_user"]        # second call: just that tool + the way out
    assert seen[1]["paypal__invoices_update"] == fit_alone(full) == full


def _text_reply(text):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason="STOP")])


def test_overloaded_model_falls_back_then_backs_off(monkeypatch):
    """Seen in every live run on the free tier: 503 'high demand' in bursts, and a daily quota per model."""
    from agent.llm import LLMError

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    llm = ChatLLM("google_genai:gemini-3.8-flash", fallback_model="google_genai:gemini-3.6-flash")
    waits, calls = [], []
    llm.sleep = waits.append
    ask = lambda: llm.text([HumanMessage("hi")], name="responder")  # noqa: E731

    def overloaded(**kw):
        calls.append("main")
        raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")

    def answers(**kw):
        calls.append("fallback")
        return _text_reply("fine")

    llm.model.client.models.generate_content = overloaded
    llm.fallback.client.models.generate_content = answers
    assert ask() == "fine" and calls == ["main", "fallback"] and waits == []   # the second model answers at once

    attempts = []

    def recovers_on_third(**kw):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("503 UNAVAILABLE")
        return _text_reply("recovered")

    llm.model.client.models.generate_content = llm.fallback.client.models.generate_content = recovers_on_third
    assert ask() == "recovered" and waits == [5.0]                             # both busy: wait, then go round again

    def out_of_quota(**kw):
        raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, retry in 30s")

    llm.model.client.models.generate_content = llm.fallback.client.models.generate_content = out_of_quota
    with pytest.raises(LLMError, match="RESOURCE_EXHAUSTED"):
        ask()
    assert waits == [5.0]                                                      # a spent daily quota is not waited on


def test_invoice_flow_through_the_real_gemini_adapter(monkeypatch, paypal_catalog, executor, mock_paypal, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    llm = ChatLLM("google_genai:gemini-3.8-flash")
    fake = FakeGemini()
    llm.model.client.models.generate_content = fake

    graph, ctx = build_agent(Settings(data_dir=tmp_path), llm=llm, executor=executor, embedder=None,
                             checkpointer=InMemorySaver(), runlog=RunLog(":memory:"), catalog=paypal_catalog)
    ctx.today = lambda: date(2026, 9, 30)
    cfg = {"configurable": {"thread_id": "g1"}}

    out = graph.invoke({"messages": [HumanMessage("Send an invoice for $50 to vibheesh@example.com")]}, cfg)
    assert out["__interrupt__"][0].value["tool"] == "paypal.invoices.send"
    out = graph.invoke(Command(resume={"action": "approve"}), cfg)
    assert out["outcome"] == "succeeded"
    assert out["messages"][-1].content == "Invoice sent to vibheesh@example.com."
    assert [inv["status"] for inv in mock_paypal.invoices.values()] == ["SENT"]

    # what actually went to Gemini for the tool-choosing calls
    tool_reqs = [r for r in fake.requests if r["config"].tools]
    assert len(tool_reqs) == 2
    for r in tool_reqs:
        decls = r["config"].tools[0].function_declarations
        assert len(decls) <= ctx.settings.top_k + 4
        assert all(d.parameters_json_schema is not None for d in decls)        # raw JSON Schema, not converted
        assert r["config"].tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.ANY
    create = next(d for d in tool_reqs[0]["config"].tools[0].function_declarations if d.name == "paypal__invoices_create")
    body = create.parameters_json_schema["properties"]["body"]
    assert body["required"] == ["detail"]           # optional fields stay optional
    assert tool_reqs[0]["config"].temperature is None  # Gemini 3: default temperature

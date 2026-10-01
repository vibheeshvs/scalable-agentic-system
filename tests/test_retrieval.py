"""Retrieval sanity checks (lexical only, so they run offline and fast)."""

import pytest

from agent.registry.models import Catalog, ServiceInfo
from agent.retrieval.index import ToolIndex


@pytest.fixture(scope="module")
def index(paypal_catalog):
    return ToolIndex(paypal_catalog, embedder=None)


@pytest.mark.parametrize("goal, expected", [
    # planner-style step goals for the three scenarios in the brief
    ("Create a draft invoice for $50 USD billed to vibheesh@example.com", "paypal.invoices.create"),
    ("Send the invoice created in step 1 to the recipient", "paypal.invoices.send"),
    ("List transactions between 2026-08-01 and 2026-08-31", "paypal.search.get"),
    ("List open disputes and find the ones raised by user_123", "paypal.disputes.list"),
    ("Refund the captured payment 2GG279541U471931P", "paypal.captures.refund"),
    ("Show the current account balances", "paypal.balances.get"),
])
def test_step_goals_retrieve_the_right_tool(index, goal, expected):
    top = [h.tool.id for h in index.search(goal, k=5)]
    assert expected in top, top


def test_service_scope_filters_other_services(paypal_catalog):
    other = Catalog.model_validate(paypal_catalog.model_dump())
    other.services["acme"] = ServiceInfo(name="acme")
    for t in list(other.tools.values())[:20]:  # clone some tools into a fake second service
        clone = t.model_copy(update={"id": "acme." + t.id.split(".", 1)[1], "name": "acme__" + t.name.split("__", 1)[1], "service": "acme"})
        other.tools[clone.id] = clone
    idx = ToolIndex(other, embedder=None)
    hits = idx.search("create a draft invoice", k=10, services=["paypal"])
    assert hits and all(h.tool.service == "paypal" for h in hits)
    assert idx.search("create a draft invoice", k=10, services=[]) == []
    assert idx.detect_services("create an invoice in acme please") == ["acme"]
    assert idx.detect_services("What's my balance in PayPal?") == ["paypal"]   # punctuation must not hide the name


def test_selection_eval_harness_runs_with_a_fake_model(paypal_catalog, index):
    """Mechanics only: the real numbers need an API key (scripts/eval_selection.py)."""
    import sys
    from pathlib import Path

    from langchain_core.messages import AIMessage

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_selection

    class FirstToolModel:  # same interface as agent.llm.ChatLLM
        def call_tools(self, messages, tools, *, name):
            return AIMessage(content="", tool_calls=[{"name": tools[0]["function"]["name"], "args": {}, "id": "1"}])

    queries = [{"query": "Send invoice", "gold": ["paypal.invoices.send"]}]
    stats = eval_selection.run(FirstToolModel(), queries, paypal_catalog, index, k=8)
    assert stats["retrieved top-8"][0]["ok"] is True
    assert "| retrieved top-8 | 1.00 |" in eval_selection.summarize(stats)

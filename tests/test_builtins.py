from agent.tools.builtins import analyze
from agent.tools.preview import list_fields, preview

DATA = {"transaction_details": [
    {"info": {"amount": {"value": "10.00", "currency": "USD"}, "status": "S"}},
    {"info": {"amount": {"value": "25.50", "currency": "USD"}, "status": "S"}},
    {"info": {"amount": {"value": "-5.00", "currency": "USD"}, "status": "S"}},
    {"info": {"amount": {"value": "40.00", "currency": "EUR"}, "status": "P"}},
    {"info": {"amount": {"value": "7.00", "currency": "EUR"}, "status": "S"}},
], "links": [{"href": "x"}]}


def test_sum_grouped_with_filters():
    out = analyze(DATA, "transaction_details", "sum", "info.amount.value",
                  where={"info.status": "S", "info.amount.value": {"op": "gt", "value": 0}}, group_by="info.amount.currency")
    assert out["result"] == {"USD": 35.5, "EUR": 7.0} and out["matched_items"] == 3


def test_amounts_in_different_currencies_are_never_added_together():
    # asked for one plain sum over USD and EUR records: the tool splits it per currency instead of adding them up
    out = analyze(DATA, "transaction_details", "sum", "info.amount.value")
    assert out["result"] == {"USD": 30.5, "EUR": 47.0} and "per currency" in out["note"]
    # one currency only -> an ordinary single number, no note
    usd = analyze(DATA, "transaction_details", "sum", "info.amount.value", where={"info.amount.currency": "USD"})
    assert usd["result"] == 30.5 and "note" not in usd
    assert usd["filters"] == [{"path": "info.amount.currency", "op": "eq", "value": "USD"}]
    # grouping by something that still mixes currencies inside a group is refused, with a way to fix it
    mixed = analyze(DATA, "transaction_details", "sum", "info.amount.value", group_by="info.status")
    assert "mix currencies" in mixed["error"] and "info.amount.currency" in mixed["error"]
    assert analyze(DATA, "transaction_details", "count")["result"] == 5          # counting is unaffected


def test_filters_as_a_list_of_conditions():
    """The form the tool schema declares (every field named, values as text), which any provider can fill in."""
    out = analyze(DATA, "transaction_details", "sum", "info.amount.value", group_by="info.amount.currency",
                  where=[{"path": "info.status", "value": "S"}, {"path": "info.amount.value", "op": "gt", "value": "0"}])
    assert out["result"] == {"USD": 35.5, "EUR": 7.0} and out["matched_items"] == 3
    assert analyze(DATA, "transaction_details", "count", where=[{"path": "info.status", "op": "ne", "value": "S"}])["result"] == 1
    assert analyze(DATA, "transaction_details", "count", where=[{"path": "info.amount.currency", "op": "in", "value": "EUR, GBP"}])["result"] == 2
    assert analyze(DATA, "transaction_details", "count", where=[{"path": "info.amount.value", "op": "prefix", "value": "-"}])["result"] == 1

    from jsonschema import Draft202012Validator
    from agent.tools.builtins import ANALYZE_DATA
    where = ANALYZE_DATA.parameters["properties"]["where"]
    assert where["type"] == "array" and set(where["items"]["properties"]) == {"path", "op", "value"}   # nothing free-form
    assert not list(Draft202012Validator(ANALYZE_DATA.parameters).iter_errors(
        {"step": 1, "items_path": "x", "op": "sum", "where": [{"path": "a.b", "op": "gt", "value": "0"}]}))


def test_count_and_bad_path():
    assert analyze(DATA, "transaction_details", "count")["result"] == 5
    assert "error" in analyze(DATA, "nope", "count")


def test_preview_drops_links_and_truncates_lists():
    text = preview({"items": list(range(100)), "links": ["a"]})
    assert "links" not in text and "95 more items" in text
    assert list_fields(DATA) == ["transaction_details (5 items)"]


def test_knowledge_answers_reach_later_steps_whole(make_agent):
    """A definition looked up in step 1 has to arrive intact at the step that applies it."""
    from langchain_core.messages import HumanMessage

    from agent.prompts import PlanOut, Route

    definition = ("Sales volume is the gross amount of completed incoming payments, before fees. " * 3
                  + "Two conditions: transaction_status is S, and the transaction_amount value is greater than 0.")
    graph, _, llm = make_agent({
        "router": [Route(intent="action", request="What was my total sales volume last month?")],
        "planner": [PlanOut(steps=[{"goal": "Look up the definition of sales volume", "kind": "knowledge"},
                                   {"goal": "Count the records from step 1", "kind": "compute"}])],
        "selector": [("rag_search", {"question": "How is sales volume defined?"}),
                     ("ask_user", {"question": "Which period?"})],
        "rag_generate": [definition],
    })
    graph.invoke({"messages": [HumanMessage("What was my total sales volume last month?")]}, {"configurable": {"thread_id": "kb"}})
    shown_to_step_2 = [m for name, m, _ in llm.calls if name == "selector"][1][-1].content
    assert len(definition) > 200 and "value is greater than 0" in shown_to_step_2


def test_json_string_arguments_are_coerced_to_objects():
    from agent.graph.nodes import coerce_json_strings
    schema = {"type": "object", "properties": {"where": {"type": "object"}, "body": {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "object"}}}}, "step": {"type": "integer"}}}
    args = {"where": '{"buyer.name": "user_123"}', "body": {"items": '[{"name": "x"}]'}, "step": 1}
    assert coerce_json_strings(schema, args) == {"where": {"buyer.name": "user_123"}, "body": {"items": [{"name": "x"}]}, "step": 1}
    assert coerce_json_strings(schema, {"where": "not json"}) == {"where": "not json"}

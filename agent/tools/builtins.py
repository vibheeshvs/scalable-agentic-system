"""Built-in tools that are always offered to the LLM, next to the retrieved API tools.

  rag_search     - RAG over the knowledge base (docs, policies, guides)
  system_search  - the agent searching itself: its tool catalog and its own run log
  analyze_data   - deterministic sum/count/avg/group-by over a previous step's full result
  ask_user       - the sanctioned way to get missing information instead of inventing it
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..registry.models import ToolSpec

RAG_SEARCH = ToolSpec(
    id="builtin.rag_search", name="rag_search", service="builtin", group="knowledge", kind="builtin",
    summary="Answer a question from the company knowledge base (product docs, policies, how-to guides) with citations. "
            "Use for 'how do I', 'what is our policy', definitions, and anything that isn't live account data.",
    parameters={"type": "object", "properties": {"question": {"type": "string", "description": "the question, self-contained"}},
                "required": ["question"]},
)
SYSTEM_SEARCH = ToolSpec(
    id="builtin.system_search", name="system_search", service="builtin", group="system", kind="builtin",
    summary="Search this assistant itself: which tools/APIs it can use for a topic (scope=capabilities), or the "
            "history/status of this user's previous requests and API calls (scope=activity).",
    parameters={"type": "object", "properties": {
        "query": {"type": "string"},
        "scope": {"type": "string", "enum": ["capabilities", "activity", "auto"], "default": "auto"}},
        "required": ["query"]},
)
ANALYZE_DATA = ToolSpec(
    id="builtin.analyze_data", name="analyze_data", service="builtin", group="compute", kind="builtin",
    summary="Compute over the FULL result of an earlier step (not just the preview you saw): sum, count, avg, min, max "
            "or list values, with optional filters and group_by. Use this instead of doing arithmetic yourself.",
    parameters={"type": "object", "properties": {
        "step": {"type": "integer", "description": "step number whose result to analyse"},
        "items_path": {"type": "string", "description": "dotted path to the list of records, e.g. 'transaction_details'"},
        "value_path": {"type": "string", "description": "dotted path inside each record, e.g. 'transaction_info.transaction_amount.value'"},
        "op": {"type": "string", "enum": ["sum", "count", "avg", "min", "max", "list"]},
        # A list of explicit conditions, not a free-form {path: value} object: with a forced function call some
        # providers (Gemini) can only send {} for an object that declares no fields, and the filter silently vanishes.
        "where": {"type": "array", "description": "keep only the records that meet ALL of these conditions", "items": {
            "type": "object", "properties": {
                "path": {"type": "string", "description": "dotted path inside each record, e.g. 'transaction_info.transaction_status'"},
                "op": {"type": "string", "enum": ["eq", "ne", "gt", "lt", "in", "prefix"], "description": "comparison, default eq"},
                "value": {"type": "string", "description": "what to compare with; numbers as text ('0'), several values for 'in' separated by commas"}},
            "required": ["path", "value"]}},
        "group_by": {"type": "string", "description": "dotted path to group by, e.g. currency code"}},
        "required": ["step", "items_path", "op"]},
)
ASK_USER = ToolSpec(
    id="builtin.ask_user", name="ask_user", service="builtin", group="dialog", kind="builtin",
    summary="Ask the user for information that is required but missing or ambiguous (e.g. recipient email, which "
            "invoice, currency). Use this rather than guessing.",
    parameters={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
)
BUILTINS = {t.name: t for t in (RAG_SEARCH, SYSTEM_SEARCH, ANALYZE_DATA, ASK_USER)}  # all Risk.READ


# ----------------------------------------------------------------------------------- impls
def get_path(obj: Any, path: str | None) -> Any:
    if not path:
        return obj
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, list) and part.isdigit():
            obj = obj[int(part)] if int(part) < len(obj) else None
        else:
            return None
    return obj


def _conditions(where: Any) -> list[dict[str, Any]]:
    """Normalise filters to [{path, op, value}]. The older {path: value} / {path: {op, value}} form still works."""
    if isinstance(where, dict):
        return [{"path": p, **(c if isinstance(c, dict) and "op" in c else {"op": "eq", "value": c})} for p, c in where.items()]
    return [c for c in where or [] if isinstance(c, dict) and c.get("path")]


def _match(item: Any, where: Any) -> bool:
    for cond in _conditions(where):
        v, op, target = get_path(item, cond["path"]), cond.get("op") or "eq", cond.get("value")
        if op in ("gt", "lt"):
            try:
                if not (float(v) > float(target) if op == "gt" else float(v) < float(target)):
                    return False
            except (TypeError, ValueError):
                return False
        elif op == "in":
            allowed = target if isinstance(target, list) else str(target or "").split(",")
            if str(v) not in [str(t).strip() for t in allowed]:
                return False
        elif op == "prefix":
            if not str(v or "").startswith(str(target)):
                return False
        elif (str(v).lower() == str(target).lower()) != (op != "ne"):  # eq / ne
            return False
    return True


def analyze(data: Any, items_path: str, op: str, value_path: str | None = None,
            where: list[dict[str, Any]] | dict[str, Any] | None = None, group_by: str | None = None) -> dict[str, Any]:
    items = get_path(data, items_path)
    if not isinstance(items, list):
        return {"error": f"'{items_path}' is not a list in that result"}
    items = [i for i in items if _match(i, where)]
    note = None
    if value_path and op in ("sum", "avg", "min", "max"):
        # Money is never added across currencies, whatever the model asked for: 100 USD + 100 EUR is not 200 of
        # anything. (A live run did exactly that and reported the result as a USD total.)
        money = value_path.rsplit(".", 1)[0] + "." if "." in value_path else ""
        currency_path = next((money + k for k in ("currency_code", "currency")
                              if any(get_path(i, money + k) for i in items)), None)
        mixed: dict[str, set[str]] = defaultdict(set)
        for it in items if currency_path else []:
            mixed[str(get_path(it, group_by)) if group_by else "all"].add(str(get_path(it, currency_path)))
        if any(len(c) > 1 for c in mixed.values()):
            if group_by:
                return {"error": f"some '{group_by}' groups mix currencies; group_by '{currency_path}' instead, or "
                                 f"filter to one currency with where"}
            group_by, note = currency_path, f"the amounts are in more than one currency, so the {op} is given per currency"
    groups: dict[str, list[Any]] = defaultdict(list)
    for it in items:
        key = str(get_path(it, group_by)) if group_by else "all"
        groups[key].append(get_path(it, value_path) if value_path else it)

    def reduce(vals: list[Any]) -> Any:
        if op == "count":
            return len(vals)
        if op == "list":
            return vals[:50]
        nums = []
        for v in vals:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                pass
        if not nums:
            return None
        return round({"sum": sum, "min": min, "max": max}.get(op, lambda x: sum(x) / len(x))(nums), 2)

    result = {k: reduce(v) for k, v in groups.items()}
    out = {"op": op, "matched_items": len(items), "result": result if group_by else result.get("all", reduce([]))}
    if where:
        out["filters"] = _conditions(where)  # so the reply can say exactly what the number covers
    if note:
        out["note"] = note
    return out


def system_search(query: str, scope: str, *, tool_index, runlog, thread_id: str | None, current_run: str | None,
                  services: list[str] | None = None) -> dict[str, Any]:
    q = query.lower()
    if scope == "auto":
        activity_words = ("status", "last request", "history", "did i", "did you", "previous", "earlier", "log", "what happened")
        scope = "activity" if any(w in q for w in activity_words) else "capabilities"
    if scope == "activity":
        runs = runlog.recent_runs(thread_id, limit=5, exclude_run=current_run)
        return {"scope": "activity", "recent_requests": [
            {"request": r["request"], "status": r["status"], "started_at": r["started_at"], "summary": r["summary"],
             "api_calls": [f"{c['tool_id']} -> {c['status']}" + (f" (HTTP {c['http_status']})" if c["http_status"] else "")
                           for c in r["tool_calls"]]} for r in runs]}
    hits = tool_index.search(query, k=12, services=services)
    return {"scope": "capabilities", "total_tools_in_catalog": len(tool_index.tools),
            "matching_tools": [h.tool.card() for h in hits],
            "builtin_tools": [t.card() for t in BUILTINS.values()]}

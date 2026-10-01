"""Graph nodes.

    router -> planner -> [select -> validate -> guard -> execute -> advance]* -> respond
                  \\-> (knowledge / system questions skip the planner with a preset tool)

The LLM only ever sees: the conversation, a service/group overview (planner), and the ~8 tools
retrieved for the current step plus 4 built-ins (selector). Everything else - validation,
confirmation, auth, retries, idempotency, pagination, logging - is plain code.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from jsonschema import Draft202012Validator
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt

from ..config import Settings
from ..llm import LLM, LLMError, message_text
from ..prompts import PLANNER, RESPONDER, ROUTER, SELECTOR, SELECTOR_CONTEXT, PlanOut, Route
from ..registry.models import Catalog, ToolSpec
from ..retrieval.index import ToolIndex
from ..runlog import RunLog
from ..tools.builtins import BUILTINS, analyze, system_search
from ..tools.http import HttpExecutor
from ..tools.preview import list_fields, preview
from ..tools.rag import KnowledgeBase, rag_answer
from .state import AgentState, Step

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
FIXABLE_HTTP = {400, 404, 409, 422}
STEP_TOOL = {"compute": "analyze_data", "knowledge": "rag_search", "system": "system_search"}  # step kind -> its built-in


@dataclass
class Context:
    settings: Settings
    catalog: Catalog
    index: ToolIndex
    kb: KnowledgeBase
    llm: LLM
    executor: HttpExecutor
    runlog: RunLog
    today: Callable[[], date] = date.today
    by_name: dict[str, ToolSpec] = field(default_factory=dict)

    def __post_init__(self):
        self.by_name = {t.name: t for t in self.catalog.tools.values()} | BUILTINS

    @property
    def enabled(self) -> list[str]:
        return [s for s in (self.settings.enabled_services or list(self.catalog.services)) if s in self.catalog.services]


class Nodes:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    # ============================================================== router
    def router(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = self.ctx
        thread = (config.get("configurable") or {}).get("thread_id", "default")
        history = self._history(state)
        last = next((str(m.content) for m in reversed(history) if isinstance(m, HumanMessage)), "")
        run_id = uuid.uuid4().hex[:12]
        ctx.runlog.start_run(run_id, thread, last)

        try:
            route = ctx.llm.structured(Route, [SystemMessage(ROUTER.format(services=", ".join(ctx.enabled))), *history], name="router")
        except LLMError as e:  # quota, overload...: end the turn with a reply so the thread and the run log stay consistent
            return {"run_id": run_id, "thread_id": thread, "request": last, "plan": [], "cursor": 0, "results": {},
                    **self._model_failed(run_id, e)}
        request = route.request or last
        mentioned = [s.lower() for s in route.services] + ctx.index.detect_services(request)
        services = [s for s in ctx.enabled if s in mentioned] or ctx.enabled
        ctx.runlog.update_run(run_id, intent=route.intent)

        update: dict[str, Any] = {"run_id": run_id, "thread_id": thread, "intent": route.intent, "request": request, "services": services,
                                  "plan": [], "cursor": 0, "results": {}, "outcome": None}
        if route.intent == "chat":
            reply = route.reply or "Hi! I can work with your connected accounts - invoices, payments, disputes, reports. What do you need?"
            ctx.runlog.update_run(run_id, status="succeeded", summary=reply[:300])
            return {**update, "messages": [AIMessage(reply)], "outcome": "succeeded"}
        if route.intent == "knowledge":  # fast path: the RAG tool, no planning needed
            update["plan"] = [_preset(1, request, "rag_search", {"question": request})]
        elif route.intent == "system":
            update["plan"] = [_preset(1, request, "system_search", {"query": request, "scope": "auto"})]
        return update

    @staticmethod
    def after_router(state: AgentState) -> str:
        if state.get("outcome"):
            return END
        return "planner" if state["intent"] == "action" else "select"

    # ============================================================== planner
    def planner(self, state: AgentState) -> dict[str, Any]:
        ctx = self.ctx
        overview = ctx.catalog.subset(state["services"]).overview()
        try:
            out = ctx.llm.structured(PlanOut, [
                SystemMessage(PLANNER.format(today=ctx.today().isoformat(), overview=overview)),
                *self._history(state), HumanMessage(f"Plan this request: {state['request']}")], name="planner")
        except LLMError as e:
            return {"plan": [], **self._model_failed(state["run_id"], e)}
        if not out.steps:
            q = out.clarification or "Could you tell me a bit more about what you'd like me to do?"
            ctx.runlog.update_run(state["run_id"], status="needs_input", summary=q)
            return {"messages": [AIMessage(q)], "outcome": "needs_input", "plan": []}
        plan = [Step(n=i + 1, goal=s.goal, kind=s.kind, status="pending", attempts=0)
                for i, s in enumerate(out.steps[: ctx.settings.max_steps])]
        return {"plan": plan, "cursor": 0}

    @staticmethod
    def after_planner(state: AgentState) -> str:
        return END if not state.get("plan") else "select"

    # ============================================================== select (retrieve + choose)
    def select(self, state: AgentState) -> dict[str, Any]:
        ctx = self.ctx
        plan, i = _copy(state["plan"]), state["cursor"]
        step = plan[i]
        if step.get("preset"):
            return {}
        own = STEP_TOOL.get(step.get("kind") or "api")
        if own:  # a computation or lookup step: only its own built-in is on offer, so it can't be "done" by something else
            hits = []
            tools = [BUILTINS[own].to_llm_tool(), BUILTINS["ask_user"].to_llm_tool()]
        else:
            hits = ctx.index.search(step["goal"], k=ctx.settings.top_k, services=state.get("services"))
            tools = [h.tool.to_llm_tool() for h in hits] + [b.to_llm_tool() for b in BUILTINS.values()]
        error = (f"\n\nYour previous attempt ({step.get('tool')} with {json.dumps(step.get('args'))[:600]}) failed: "
                 f"{step['error']}\nFix the arguments or pick a different tool.") if step.get("error") else ""
        context = SELECTOR_CONTEXT.format(request=state["request"], plan=_plan_text(plan), n=step["n"],
                                          results=_results_text(plan, i), goal=step["goal"], error=error)
        try:
            ai = ctx.llm.call_tools([SystemMessage(SELECTOR.format(today=ctx.today().isoformat())),
                                     *self._history(state, 6), HumanMessage(context)], tools, name="selector")
        except LLMError as e:  # no tool was chosen, so nothing ran: fail this step and report what did happen
            step.update(status="failed", error=f"the model call failed ({e})", candidates=[h.tool.id for h in hits])
            return {"plan": plan}
        if ai.tool_calls:
            name, args = ai.tool_calls[0]["name"], ai.tool_calls[0].get("args") or {}
        else:  # model answered in text instead of calling a tool -> treat as a question for the user
            name, args = "ask_user", {"question": message_text(ai.content) or "Could you clarify what you need?"}
        step.update(tool=name, args=args, candidates=[h.tool.id for h in hits])
        return {"plan": plan}

    # ============================================================== validate
    def validate(self, state: AgentState) -> dict[str, Any]:
        ctx = self.ctx
        plan, i = _copy(state["plan"]), state["cursor"]
        step = plan[i]
        if step["status"] == "failed":  # select could not get an answer from the model
            return {}
        spec = ctx.by_name.get(step.get("tool") or "")
        args = step.get("args") or {}
        errors: list[str] = []
        if spec is None:
            errors.append(f"'{step.get('tool')}' is not an available tool")
        elif spec.kind == "http" and spec.service not in ctx.enabled:
            errors.append(f"service '{spec.service}' is not connected for this user")
        else:
            args = coerce_json_strings(spec.parameters, args)  # some models send objects as JSON text
            step["args"] = args
            for e in sorted(Draft202012Validator(spec.parameters).iter_errors(args), key=lambda e: list(e.path))[:4]:
                errors.append(f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message[:200]}")
            if spec.kind == "http":
                errors += self._ungrounded(args, state)
        if errors:
            step["attempts"] = step.get("attempts", 0) + 1
            step["error"] = "; ".join(errors)
            if step["attempts"] > ctx.settings.max_repairs:
                step["status"] = "failed"
            ctx.runlog.log_call(state["run_id"], self._thread(state), step["n"], spec.id if spec else str(step.get("tool")),
                                spec.risk.value if spec else "?", args, "rejected_by_validation", error=step["error"])
        else:
            step["error"] = None
            if spec.kind == "http" and spec.method != "GET" and not step.get("idem_key"):
                step["idem_key"] = str(uuid.uuid4())  # decided (and checkpointed) BEFORE the call
            step["tool_id"] = spec.id
        return {"plan": plan}

    def after_validate(self, state: AgentState) -> str:
        step = state["plan"][state["cursor"]]
        if step["status"] == "failed":
            return "advance"
        if step.get("error"):
            return "select"
        return "ask" if step.get("tool") == "ask_user" else "guard"

    def _ungrounded(self, args: dict[str, Any], state: AgentState) -> list[str]:
        """Cheap hallucination check: ids in the URL and any email must come from the user or earlier results."""
        corpus = " ".join(str(m.content) for m in state.get("messages", []))
        corpus += " " + state.get("request", "") + " " + json.dumps(state.get("results", {}), default=str)
        corpus = corpus.lower()
        errs = []
        for name, val in (args.get("path") or {}).items():
            if str(val).lower() not in corpus:
                errs.append(f"path.{name}='{val}' does not appear in the conversation or earlier results - don't invent "
                            f"ids; look it up with a list/search tool first or ask the user")
        for email in sorted(set(EMAIL_RE.findall(json.dumps(args)))):
            if email.lower() not in corpus:
                errs.append(f"email '{email}' was never given by the user")
        return errs

    # ============================================================== guard (human in the loop)
    def guard(self, state: AgentState) -> dict[str, Any]:
        ctx = self.ctx
        plan, i = _copy(state["plan"]), state["cursor"]
        step = plan[i]
        spec = ctx.by_name[step["tool"]]
        if spec.kind != "http" or spec.risk.value not in ctx.settings.confirm_risk:
            return {}
        args = step.get("args") or {}
        path = spec.path
        for k, v in (args.get("path") or {}).items():
            path = path.replace("{" + k + "}", str(v))
        # interrupt() pauses the graph and checkpoints it; the CLI/API resumes with Command(resume=...)
        decision = interrupt({"type": "confirmation", "step": step["n"], "goal": step["goal"], "tool": spec.id,
                              "summary": spec.summary, "risk": spec.risk.value, "method": spec.method, "path": path,
                              "query": args.get("query"), "body": args.get("body")})
        if _approved(decision):
            return {}
        reason = decision.get("reason") if isinstance(decision, dict) else None
        step.update(status="cancelled", error="declined by user" + (f": {reason}" if reason else ""))
        for later in plan[i + 1:]:
            later["status"] = "skipped"
        ctx.runlog.log_call(state["run_id"], self._thread(state), step["n"], spec.id, spec.risk.value, args, "declined_by_user")
        return {"plan": plan}

    @staticmethod
    def after_guard(state: AgentState) -> str:
        return "respond" if state["plan"][state["cursor"]]["status"] == "cancelled" else "execute"

    # ============================================================== execute
    def execute(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ctx = self.ctx
        plan, i = _copy(state["plan"]), state["cursor"]
        step = plan[i]
        spec = ctx.by_name[step["tool"]]
        args = step.get("args") or {}
        results = dict(state.get("results") or {})

        if spec.kind == "builtin":
            ok, data, err, http_status, attempts, latency = *self._run_builtin(spec.name, args, state), None, 1, 0
        else:
            res = ctx.executor.call(spec, args, step.get("idem_key"))
            ok, data, err, http_status, attempts, latency = res.ok, res.data, res.error, res.status, res.attempts, res.latency_ms
        ctx.runlog.log_call(state["run_id"], self._thread(state), step["n"], spec.id, spec.risk.value, args,
                            "ok" if ok else "error", http_status, attempts, latency, err)

        if ok:
            results[str(step["n"])] = data
            lists = list_fields(data)
            # API payloads are previewed with long strings cut; a knowledge answer is the payload, so it stays whole
            # (cut at 200 characters, a definition lost its second condition before the next step could apply it)
            text = preview(data, max_str=2500) if spec.name == "rag_search" else preview(data)
            step["preview"] = text + (f"\n(lists you can analyze_data: {', '.join(lists)})" if lists else "")
            warning = data.get("_pagination_warning") if isinstance(data, dict) else None
            if warning:  # never let a partial list pass for the whole thing: any total built on it is incomplete
                step["preview"] += f"\nWARNING, INCOMPLETE DATA: {warning}"
            step.update(status="done", error=None)
        else:
            step["attempts"] = step.get("attempts", 0) + 1
            step["error"] = err
            step["idem_key"] = None  # a corrected request is a new request
            fixable = spec.kind == "builtin" or http_status in FIXABLE_HTTP
            step["status"] = "pending" if fixable and step["attempts"] <= ctx.settings.max_repairs and not step.get("preset") else "failed"
        return {"plan": plan, "results": results}

    @staticmethod
    def after_execute(state: AgentState) -> str:
        return "select" if state["plan"][state["cursor"]]["status"] == "pending" else "advance"

    def _run_builtin(self, name: str, args: dict[str, Any], state: AgentState) -> tuple[bool, Any, str | None]:
        ctx = self.ctx
        try:
            if name == "rag_search":
                return True, rag_answer(ctx.kb, ctx.llm, args["question"]), None
            if name == "system_search":
                return True, system_search(args["query"], args.get("scope", "auto"), tool_index=ctx.index,
                                           runlog=ctx.runlog, thread_id=self._thread(state),
                                           current_run=state.get("run_id"), services=state.get("services")), None
            if name == "analyze_data":
                src = (state.get("results") or {}).get(str(args["step"]))
                if src is None:
                    return False, None, f"step {args['step']} has no stored result"
                out = analyze(src, args["items_path"], args["op"], args.get("value_path"), args.get("where"), args.get("group_by"))
                return ("error" not in out), out, out.get("error")
        except Exception as e:  # a bug in a built-in shouldn't take the whole run down
            return False, None, f"{name} failed: {type(e).__name__}: {e}"
        return False, None, f"unknown built-in {name}"

    # ============================================================== advance / ask / respond
    def advance(self, state: AgentState) -> dict[str, Any]:
        plan, i = _copy(state["plan"]), state["cursor"]
        if plan[i]["status"] == "failed":
            for later in plan[i + 1:]:
                later["status"] = "skipped"
            return {"plan": plan}
        return {"cursor": i + 1} if i + 1 < len(plan) else {}

    @staticmethod
    def after_advance(state: AgentState) -> str:
        return "select" if state["plan"][state["cursor"]]["status"] == "pending" else "respond"

    def ask(self, state: AgentState) -> dict[str, Any]:
        plan, i = _copy(state["plan"]), state["cursor"]
        q = (plan[i].get("args") or {}).get("question") or "Could you give me a bit more detail?"
        plan[i]["status"] = "needs_input"
        for later in plan[i + 1:]:
            later["status"] = "skipped"
        self.ctx.runlog.update_run(state["run_id"], status="needs_input", summary=q)
        return {"messages": [AIMessage(q)], "plan": plan, "outcome": "needs_input"}

    def respond(self, state: AgentState) -> dict[str, Any]:
        ctx = self.ctx
        plan = state["plan"]
        statuses = {s["status"] for s in plan}
        if statuses <= {"done"}:
            outcome = "succeeded"
        elif "cancelled" in statuses:
            outcome = "cancelled"
        else:
            outcome = "partial" if "done" in statuses else "failed"

        first = plan[0] if plan else {}
        if len(plan) == 1 and first.get("tool") == "rag_search" and first["status"] == "done":
            r = state["results"]["1"]  # the RAG tool already produced a grounded, cited answer
            text = r["answer"] + (f"\n\nSources: {'; '.join(r['sources'])}" if r["sources"] else "")
        else:
            try:
                text = ctx.llm.text([SystemMessage(RESPONDER), HumanMessage(
                    f"User request: {state['request']}\n\nWhat happened:\n{_plan_text(plan, with_results=True)}")], name="responder")
            except LLMError as e:  # the work is done (or not) regardless: report the step statuses without the model
                text = (f"I couldn't write a summary because the model call failed ({e}). "
                        f"Here is what happened:\n{_plan_text(plan, with_errors=True)}")
        ctx.runlog.update_run(state["run_id"], status=outcome, summary=text[:500])
        return {"messages": [AIMessage(text)], "outcome": outcome}

    def _model_failed(self, run_id: str, e: LLMError) -> dict[str, Any]:
        text = f"I couldn't work on that because the model call failed ({e}). Nothing was changed. Please try again."
        self.ctx.runlog.update_run(run_id, status="failed", summary=text[:500])
        return {"messages": [AIMessage(text)], "outcome": "failed"}

    # ============================================================== helpers
    @staticmethod
    def _history(state: AgentState, n: int = 10) -> list[AnyMessage]:
        return [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))][-n:]

    @staticmethod
    def _thread(state: AgentState) -> str:
        return state.get("thread_id") or "default"


def coerce_json_strings(schema: dict[str, Any], value: Any) -> Any:
    """Where the schema wants an object/array but the model sent a JSON string, parse it."""
    if not isinstance(schema, dict):
        return value
    want = schema.get("type")
    if isinstance(value, str) and want in ("object", "array"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict if want == "object" else list):
                value = parsed
        except ValueError:
            return value
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        return {k: coerce_json_strings(props.get(k, {}), v) for k, v in value.items()}
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [coerce_json_strings(schema["items"], v) for v in value]
    return value


def _preset(n: int, goal: str, tool: str, args: dict[str, Any]) -> Step:
    return Step(n=n, goal=goal, status="pending", preset=True, tool=tool, args=args, attempts=0)


def _copy(plan: list[Step]) -> list[Step]:
    return [Step(**s) for s in plan]


def _approved(decision: Any) -> bool:
    if isinstance(decision, dict):
        return bool(decision.get("approved")) or str(decision.get("action", "")).lower() in ("approve", "yes")
    return decision is True or str(decision).strip().lower() in ("y", "yes", "approve", "ok")


def _plan_text(plan: list[Step], with_results: bool = False, with_errors: bool = False) -> str:
    lines = []
    for s in plan:
        tool = f" -> {s.get('tool_id') or s.get('tool')}" if s.get("tool") else ""
        lines.append(f"{s['n']}. [{s['status']}] {s['goal']}{tool}")
        if with_results and s.get("preview"):
            lines.append(f"   result: {s['preview'][:2500]}")
        if (with_results or with_errors) and s.get("error"):
            lines.append(f"   error: {s['error']}")
    return "\n".join(lines)


def _results_text(plan: list[Step], upto: int) -> str:
    done = [f"Step {s['n']} ({s.get('tool_id') or s.get('tool')}): {s['preview']}" for s in plan[:upto] if s.get("preview")]
    return "\n".join(done) or "(none yet)"

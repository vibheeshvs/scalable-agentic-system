"""A small local web UI for the agent. Standard library only, bound to 127.0.0.1.

    python ui/server.py                 # then open http://127.0.0.1:8765
    python ui/server.py --port 9000

Two modes, switchable in the page:
  live  - the real model from .env (needs an API key and quota)
  demo  - a scripted stand-in model that only knows the example requests. No API calls; everything else
          (retrieval, validation, approval, executor, mock PayPal, run log) is the real code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent.cli import _load_dotenv, quiet_logs  # noqa: E402
from agent.config import Settings  # noqa: E402
from agent.graph.build import build_agent  # noqa: E402
from agent.llm import GEMINI_MAX_SCHEMA_NODES, GEMINI_REQUEST_BUDGET, fit_request, schema_cost, schema_nodes  # noqa: E402
from agent.prompts import PlanOut, Route  # noqa: E402
from agent.runlog import RunLog  # noqa: E402
from agent.tools.builtins import BUILTINS  # noqa: E402

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
EXAMPLES = [
    "Send an invoice for $50 to vibheesh@example.com",
    "What was my total sales volume last month?",
    "Is there a dispute open from user_123?",
    "What tools are available for managing invoices?",
    "What's the status of my last request?",
    "When do we add a late fee to an invoice?",
    "Send an invoice for $50 to",
]


class DemoLLM:
    """Scripted stand-in for the model, so the UI can be explored without an API key or quota.
    It recognises the example requests by keyword and answers the way a model is expected to."""

    def __init__(self, today=date.today):
        self.today = today

    # ---- router / planner
    def structured(self, schema, messages, *, name):
        text = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        text = text.removeprefix("Plan this request: ")
        low = text.lower()
        if schema is Route:
            if re.fullmatch(r"\W*(hi|hello|hey|thanks|thank you)\W*", low):
                return Route(intent="chat", request=text, reply="Hello. This is the offline demo: try one of the example requests.")
            if any(w in low for w in ("what tools", "which tools", "can you do", "status of my", "last request", "history")):
                return Route(intent="system", request=text)
            if any(w in low for w in ("late fee", "policy", "how do we", "when do we", "what is our")):
                return Route(intent="knowledge", request=text)
            return Route(intent="action", request=text, services=["paypal"])
        if "invoice" in low and ("send" in low or "create" in low or "bill" in low):
            if not EMAIL.search(text):
                return PlanOut(clarification="Who should I send the invoice to? I need the recipient's email address.")
            return PlanOut(steps=[{"goal": f"Create a draft invoice for {text.split(' for ', 1)[-1]}", "kind": "api"},
                                  {"goal": "Send the invoice created in step 1", "kind": "api"}])
        if "sales" in low or "volume" in low or "revenue" in low:
            end = self.today().replace(day=1) - timedelta(days=1)
            return PlanOut(steps=[
                {"goal": f"List transactions between {end.replace(day=1)}T00:00:00Z and {end}T23:59:59Z", "kind": "api"},
                {"goal": "Look up how sales volume is defined", "kind": "knowledge"},
                {"goal": "Total the transactions from step 1 using the definition from step 2", "kind": "compute"}])
        if "dispute" in low:
            who = re.search(r"\buser_\w+", text)
            return PlanOut(steps=[{"goal": "List disputes that are still open", "kind": "api"},
                                  {"goal": f"From step 1, list the dispute ids whose buyer is {who.group(0) if who else 'the buyer'}", "kind": "compute"}])
        if "balance" in low:
            return PlanOut(steps=[{"goal": "Show the current account balances", "kind": "api"}])
        return PlanOut(clarification="The offline demo only knows the example requests. Switch to Live to use the real model.")

    # ---- selector
    def call_tools(self, messages, tools, *, name):
        ctx = str(messages[-1].content)
        goal = ctx.rsplit("Current step", 1)[-1].split(":", 1)[-1].strip().split("\n")[0]
        request = ctx.split("User request:", 1)[-1].split("\n")[0].strip()
        low = goal.lower()
        if low.startswith("create a draft invoice"):
            amount = re.search(r"\$?\s?(\d+(?:\.\d+)?)", request.replace(",", ""))
            call = ("paypal__invoices_create", {"body": {
                "detail": {"currency_code": "USD"},
                "primary_recipients": [{"billing_info": {"email_address": EMAIL.search(request).group(0)}}],
                "items": [{"name": "Services", "quantity": "1",
                           "unit_amount": {"currency_code": "USD", "value": f"{float(amount.group(1)) if amount else 0:.2f}"}}]}})
        elif low.startswith("send the invoice"):
            call = ("paypal__invoices_send", {"path": {"invoice_id": re.search(r"INV2-[A-Z0-9-]+", ctx).group(0)}})
        elif low.startswith("list transactions"):
            start, end = re.findall(r"\d{4}-\d{2}-\d{2}T[\d:]+Z", goal)
            call = ("paypal__search_get", {"query": {"start_date": start, "end_date": end}})
        elif "sales volume is defined" in low:
            call = ("rag_search", {"question": "How is sales volume defined and which transactions count toward it?"})
        elif low.startswith("total the transactions"):
            call = ("analyze_data", {"step": 1, "items_path": "transaction_details", "op": "sum",
                                     "value_path": "transaction_info.transaction_amount.value",
                                     "where": [{"path": "transaction_info.transaction_status", "value": "S"},
                                               {"path": "transaction_info.transaction_amount.value", "op": "gt", "value": "0"}],
                                     "group_by": "transaction_info.transaction_amount.currency_code"})
        elif low.startswith("list disputes"):
            call = ("paypal__disputes_list", {"query": {"dispute_state": "REQUIRED_ACTION,UNDER_PAYPAL_REVIEW"}})
        elif "whose buyer is" in low:
            call = ("analyze_data", {"step": 1, "items_path": "items", "op": "list", "value_path": "dispute_id",
                                     "where": [{"path": "buyer.name", "value": goal.rsplit(" ", 1)[-1]}]})
        elif "balance" in low:
            call = ("paypal__balances_get", {})
        else:
            call = ("ask_user", {"question": "The offline demo doesn't know how to do that step."})
        return AIMessage(content="", tool_calls=[{"name": call[0], "args": call[1], "id": "demo"}])

    # ---- responder / rag_generate
    def text(self, messages, *, name):
        body = str(messages[-1].content)
        if name == "rag_generate":
            cite, excerpt = re.search(r"Excerpts:\n\[(.+?)\]\n(.+?)(?:\n\n\[|\Z)", body, re.S).groups()
            return f"{' '.join(excerpt.split())} [{cite}]"
        happened = body.split("What happened:\n", 1)[-1]
        steps = [line.strip() for line in happened.splitlines() if re.match(r"\d+\. \[", line)]
        errors = [line.strip() for line in happened.splitlines() if line.strip().startswith("error:")]
        if any("[done]" not in s for s in steps):  # something failed, was declined or skipped: say exactly that
            return "Not everything was done:\n" + "\n".join(steps + errors)
        ran = " ".join(steps)
        if "builtin.system_search" in ran and "matching_tools" in happened:
            tools = re.findall(r'"((?:\w+)\.[\w.-]+) \((?:GET|POST|PUT|PATCH|DELETE) ', happened)
            total = re.search(r"(\d+) total\)", happened)
            return f"{total.group(1) if total else len(tools)} matching tools, for example: " + ", ".join(tools[:6]) + "."
        if "builtin.system_search" in ran:
            runs = re.findall(r'"request": "(.+?)", "status": "(\w+)"', happened)
            return "Recent requests:\n" + "\n".join(f"- {r} ({s})" for r, s in runs) if runs else "No earlier requests in this conversation."
        if "paypal.invoices.send" in ran:
            inv = re.search(r'"id": "(INV2-[A-Z0-9-]+)"', happened)
            amount = re.search(r'"amount": \{"currency_code": "(\w+)", "value": "([\d.]+)"', happened)
            return (f"Invoice {inv.group(1)} for {amount.group(2)} {amount.group(1)} was created and sent to "
                    f"{EMAIL.search(happened).group(0)}.")
        if "paypal.balances.get" in ran:
            funds = re.findall(r'"available_balance": \{"currency_code": "(\w+)", "value": "([\d.]+)"', happened)
            return "Available balance: " + ", ".join(f"{v} {c}" for c, v in funds) + "."
        totals = re.findall(r'"matched_items": (\d+), "result": (\{[^{}]*\}|\[[^\[\]]*\])', happened)
        if totals:
            count, result = totals[-1]
            if result.startswith("["):
                ids = re.findall(r'"([^"]+)"', result)
                return f"Open dispute from that buyer: {', '.join(ids)}." if ids else "No open dispute from that buyer."
            amounts = ", ".join(f"{v} {k}" for k, v in re.findall(r'"(\w+)": ([\d.]+)', result))
            return f"Sales volume (completed incoming payments, per currency): {amounts}, from {count} transactions."
        return "\n".join(steps)


class App:
    def __init__(self):
        _load_dotenv()
        quiet_logs()
        self.lock = threading.Lock()  # one turn at a time
        self.agents, self.errors = {}, {}
        for mode in ("live", "demo"):
            try:
                settings = Settings()
                if mode == "demo":
                    settings.paypal_mode = "mock"  # the scripted model never touches a real account
                self.agents[mode] = build_agent(settings, llm=DemoLLM() if mode == "demo" else None,
                                                checkpointer=InMemorySaver(), runlog=RunLog(":memory:"))
            except Exception as e:  # usually: no API key for the live model
                self.errors[mode] = f"{type(e).__name__}: {' '.join(str(e).split())[:220]}"

    # ------------------------------------------------------------------ views
    def status(self) -> dict:
        graph, ctx = self.agents["demo"]
        s = ctx.settings
        emb = ctx.index.index.embedder
        risk: dict[str, int] = {}
        for t in ctx.catalog.tools.values():
            risk[t.risk.value] = risk.get(t.risk.value, 0) + 1
        return {"modes": {m: m in self.agents for m in ("live", "demo")}, "errors": self.errors,
                "model": s.llm_model, "router_model": s.router_model, "fallback_model": s.fallback_model,
                "paypal": s.paypal_mode, "tools": len(ctx.catalog), "top_k": s.top_k,
                "services": {n: {"title": v.title, "groups": len(v.groups), "tools": sum(v.groups.values())}
                             for n, v in ctx.catalog.services.items()},
                "embeddings": emb.name if emb else "off (BM25 only)", "risk": risk, "examples": EXAMPLES,
                "confirm_risk": s.confirm_risk,
                "limits": {"nodes": GEMINI_MAX_SCHEMA_NODES, "request": GEMINI_REQUEST_BUDGET}}

    @staticmethod
    def view(out: dict) -> dict:
        pending = out.get("__interrupt__")
        reply = None if pending else next((str(m.content) for m in reversed(out.get("messages", [])) if isinstance(m, AIMessage)), None)
        steps = [{"n": s["n"], "goal": s["goal"], "kind": s.get("kind") or "api", "status": s["status"],
                  "tool": s.get("tool_id") or s.get("tool"), "args": s.get("args"), "candidates": s.get("candidates") or [],
                  "preview": (s.get("preview") or "")[:1800], "error": s.get("error"), "attempts": s.get("attempts", 0),
                  "preset": bool(s.get("preset"))} for s in out.get("plan") or []]
        return {"reply": reply, "interrupt": pending[0].value if pending else None, "plan": steps, "cursor": out.get("cursor"),
                "outcome": out.get("outcome"), "intent": out.get("intent"), "request": out.get("request"),
                "services": out.get("services")}

    def cfg(self, mode: str, thread: str) -> dict:
        return {"configurable": {"thread_id": f"{mode}-{thread}"}, "tags": ["ui"], "run_name": "agent_turn"}

    def turn(self, mode: str, thread: str, payload) -> dict:
        if mode not in self.agents:
            return {"error": self.errors.get(mode, f"mode {mode} is not available")}
        with self.lock:
            try:
                return self.view(self.agents[mode][0].invoke(payload, self.cfg(mode, thread)))
            except Exception as e:
                return {"error": f"{type(e).__name__}: {' '.join(str(e).split())[:400]}"}

    def progress(self, mode: str, thread: str) -> dict:
        """Plan as checkpointed so far; polled by the page while a turn is running."""
        if mode not in self.agents:
            return {"plan": []}
        try:
            values = self.agents[mode][0].get_state(self.cfg(mode, thread)).values
            return {k: v for k, v in self.view({**values, "__interrupt__": None}).items() if k in ("plan", "intent", "request", "cursor")}
        except Exception:
            return {"plan": []}

    def search(self, q: str, k: int, mode: str, service: str | None) -> dict:
        ctx = self.agents["demo"][1]
        hits = ctx.index.search(q, k=k, services=[service] if service else None, mode=mode) if q.strip() else []
        full = [h.tool.parameters for h in hits] + [b.parameters for b in BUILTINS.values()]
        sent = fit_request(full)
        rows = [{"id": h.tool.id, "method": h.tool.method, "path": h.tool.path, "summary": h.tool.summary,
                 "risk": h.tool.risk.value, "service": h.tool.service, "score": round(h.score, 4),
                 "lexical": round(h.lexical, 2), "dense": None if h.dense is None else round(h.dense, 3),
                 "nodes": schema_nodes(h.tool.parameters), "sent_nodes": schema_nodes(s), "idempotent": bool(h.tool.idempotency_header),
                 "tokens": len(json.dumps(h.tool.to_llm_tool())) // 4}
                for h, s in zip(hits, sent)]
        return {"hits": rows, "catalog": len(ctx.index.tools), "builtins": list(BUILTINS),
                "cost": sum(map(schema_cost, full)), "sent_cost": sum(map(schema_cost, sent)),
                "tokens": sum(r["tokens"] for r in rows),
                "all_tokens": sum(len(json.dumps(t.to_llm_tool())) for t in ctx.catalog.tools.values()) // 4}

    def catalog(self) -> dict:
        ctx = self.agents["demo"][1]
        groups: dict[str, list] = {}
        for t in ctx.catalog.tools.values():
            groups.setdefault(f"{t.service} / {t.group}", []).append(
                {"id": t.id, "method": t.method, "path": t.path, "summary": t.summary, "risk": t.risk.value,
                 "idempotent": bool(t.idempotency_header)})
        return {"groups": [{"name": n, "tools": ts} for n, ts in sorted(groups.items())]}

    def activity(self, mode: str) -> dict:
        if mode not in self.agents:
            return {"runs": []}
        return {"runs": self.agents[mode][1].runlog.recent_runs(None, limit=25)}


class Handler(BaseHTTPRequestHandler):
    app: App

    def log_message(self, *args):  # keep the terminal quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200):
        self._send(json.dumps(data, default=str).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in ("/", "/index.html"):
            return self._send((Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/status":
            return self._json(self.app.status())
        if url.path == "/api/tools":
            k = max(1, min(25, int(q.get("k", "8") or 8)))
            return self._json(self.app.search(q.get("q", ""), k, q.get("mode", "hybrid"), q.get("service") or None))
        if url.path == "/api/catalog":
            return self._json(self.app.catalog())
        if url.path == "/api/activity":
            return self._json(self.app.activity(q.get("mode", "demo")))
        if url.path == "/api/progress":
            return self._json(self.app.progress(q.get("mode", "demo"), q.get("thread", "default")))
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", "0")), 100_000)) or b"{}")
        except ValueError:
            return self._json({"error": "invalid JSON"}, 400)
        mode, thread = body.get("mode", "demo"), str(body.get("thread", "default"))[:40]
        if self.path == "/api/chat":
            text = str(body.get("message", "")).strip()
            if not text:
                return self._json({"error": "empty message"}, 400)
            return self._json(self.app.turn(mode, thread, {"messages": [HumanMessage(text)]}))
        if self.path == "/api/resume":
            decision = {"action": "approve"} if body.get("action") == "approve" else \
                {"action": "reject", "reason": (str(body.get("reason") or "").strip() or None)}
            return self._json(self.app.turn(mode, thread, Command(resume=decision)))
        self._json({"error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.app = App()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    modes = ", ".join(Handler.app.agents) + ("" if "live" in Handler.app.agents else "  (live unavailable: " + Handler.app.errors.get("live", "") + ")")
    print(f"Agent console: http://127.0.0.1:{args.port}   modes: {modes}   Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""Chat with the agent in a terminal.

    python -m agent.cli                         # mock PayPal unless PAYPAL_CLIENT_ID is set
    python -m agent.cli -m "What was my total sales volume last month?"

Commands: /new (fresh conversation), /tools <text> (show what retrieval returns), /log (recent runs), /quit
"""

from __future__ import annotations

import argparse
import json
import os
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from .config import Settings
from .graph.build import build_agent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def quiet_logs() -> None:
    """Hide per-request INFO/WARNING chatter from httpx and the Google SDK."""
    import logging

    for name in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(name).setLevel(logging.ERROR)


def confirm(payload: dict) -> dict:
    print("\n" + "-" * 70)
    print(f"  Confirmation needed ({payload['risk']} risk): {payload['summary']}")
    print(f"  Step {payload['step']}: {payload['goal']}")
    print(f"  {payload['method']} {payload['path']}")
    if payload.get("query"):
        print(f"  query: {json.dumps(payload['query'])}")
    if payload.get("body"):
        print("  body:  " + json.dumps(payload["body"], indent=2).replace("\n", "\n         "))
    print("-" * 70)
    ans = input("  Approve? [y/N] ").strip().lower()
    if ans in ("y", "yes"):
        return {"action": "approve"}
    return {"action": "reject", "reason": input("  Reason (optional): ").strip() or None}


def run_turn(graph, text: str, thread_id: str) -> None:
    cfg = {"configurable": {"thread_id": thread_id}, "metadata": {"thread_id": thread_id, "surface": "cli"},
           "tags": ["cli"], "run_name": "agent_turn"}
    out = graph.invoke({"messages": [HumanMessage(text)]}, cfg)
    while out.get("__interrupt__"):
        out = graph.invoke(Command(resume=confirm(out["__interrupt__"][0].value)), cfg)
    msg = next((m for m in reversed(out.get("messages", [])) if isinstance(m, AIMessage)), None)
    print(f"\nassistant> {msg.content if msg else '(no reply)'}")
    plan = out.get("plan") or []
    if plan:
        trail = ", ".join(f"{s['n']}:{s.get('tool_id') or s.get('tool') or '-'}[{s['status']}]" for s in plan)
        print(f"           ({out.get('outcome')}; {trail})")


def main() -> None:
    quiet_logs()
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", help="send one message and exit")
    ap.add_argument("--thread", default=None, help="resume a conversation thread id")
    args = ap.parse_args()

    settings = Settings()
    graph, ctx = build_agent(settings)
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    emb = ctx.index.index.embedder
    print(f"catalog: {len(ctx.catalog)} tools / {len(ctx.catalog.services)} service(s) | PayPal: {settings.paypal_mode} | "
          f"model: {settings.llm_model} | embeddings: {emb.name if emb else 'off'} | LangSmith: {'on' if tracing else 'off'}")
    thread = args.thread or uuid.uuid4().hex[:8]

    if args.message:
        run_turn(graph, args.message, thread)
        return
    print(f"thread {thread}. Type /quit to exit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            thread = uuid.uuid4().hex[:8]
            print(f"new thread {thread}")
            continue
        if text.startswith("/tools"):
            for h in ctx.index.search(text[6:].strip() or "invoice", k=8):
                print(f"  {h.score:.4f}  {h.tool.card()}")
            continue
        if text == "/log":
            for r in ctx.runlog.recent_runs(thread, limit=5):
                print(f"  {r['started_at']} [{r['status']}] {r['request']}")
                for c in r["tool_calls"]:
                    print(f"      step {c['step']}: {c['tool_id']} -> {c['status']} {c['http_status'] or ''}")
            continue
        try:
            run_turn(graph, text, thread)
        except Exception as e:  # keep the REPL alive; the trace has the details
            print(f"\n[error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

"""Live check with a REAL model (your .env key) against the built-in mock PayPal.

    python scripts/live_check.py              # all scenarios, ~5 s pause between them
    python scripts/live_check.py --only 1 3   # just some of them

It runs the task's example requests end to end, auto-approves confirmations (safe: PayPal is
always the mock here, whatever your .env says), checks each result, and writes
live_check_report.md. Nothing real is ever sent.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402


@dataclass
class Scenario:
    message: str
    expect_tools: list[str] = field(default_factory=list)   # tool ids that must have run successfully
    expect_outcome: str | None = "succeeded"
    expect_text: list[str] = field(default_factory=list)    # words the reply should contain
    expect_approval: bool = False
    verify: Callable[[dict], str | None] | None = None      # extra check on the final state; returns a problem or None
    thread: str = "live-check"                              # same thread = shared history (needed for #5)


def _sales_total_is_right(out: dict) -> str | None:
    """Recompute the answer independently from the transactions the agent fetched: completed, incoming, per
    currency (the definition in data/knowledge/month-end-reporting.md). Running analyze_data is not enough -
    one live run summed euros and dollars, refunds and pending payments into a single 'USD' figure."""
    from agent.tools.builtins import analyze

    by_tool = {s.get("tool_id"): str(s["n"]) for s in out.get("plan") or [] if s.get("status") == "done"}
    fetched, computed = (out["results"].get(by_tool.get(t, "")) for t in ("paypal.search.get", "builtin.analyze_data"))
    if not fetched or not computed:
        return "no computed total to check"
    expected = analyze(fetched, "transaction_details", "sum", "transaction_info.transaction_amount.value",
                       where={"transaction_info.transaction_status": "S",
                              "transaction_info.transaction_amount.value": {"op": "gt", "value": 0}},
                       group_by="transaction_info.transaction_amount.currency_code")["result"]
    return None if computed.get("result") == expected else f"total is {computed.get('result')}, should be {expected}"


SCENARIOS = [
    Scenario("Send an invoice for $50 to vibheesh@example.com", ["paypal.invoices.create", "paypal.invoices.send"],
             expect_approval=True),
    Scenario("What was my total sales volume last month?", ["paypal.search.get", "builtin.analyze_data"],
             expect_text=["USD"], verify=_sales_total_is_right),
    Scenario("Is there a dispute open from user_123?", ["paypal.disputes.list"], expect_text=["PP-D-27803"]),
    Scenario("What tools are available for managing invoices?", ["builtin.system_search"]),
    Scenario("What's the status of my last request?", ["builtin.system_search"]),
    Scenario("When do we add a late fee to an invoice?", ["builtin.rag_search"], expect_text=["1.5%"]),
    Scenario("Send an invoice for $50 to", [], expect_outcome="needs_input", thread="live-check-fresh"),
]


def run_one(graph, sc: Scenario, thread: str) -> dict:
    cfg = {"configurable": {"thread_id": thread}, "tags": ["live_check"]}
    t0 = time.perf_counter()
    approvals = []
    try:
        out = graph.invoke({"messages": [HumanMessage(sc.message)]}, cfg)
        while out.get("__interrupt__"):
            approvals.append(out["__interrupt__"][0].value.get("tool"))
            out = graph.invoke(Command(resume={"action": "approve"}), cfg)
        error = None
    except Exception as e:  # report it, keep going with the next scenario
        out, error = {}, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    plan = out.get("plan") or []
    done = [s.get("tool_id") for s in plan if s.get("status") == "done"]
    reply = next((m.content for m in reversed(out.get("messages", [])) if isinstance(m, AIMessage)), "")
    reply = reply if isinstance(reply, str) else str(reply)
    problems = []
    if error:
        problems.append("crashed")
    if sc.expect_outcome and out.get("outcome") != sc.expect_outcome:
        problems.append(f"outcome {out.get('outcome')} (expected {sc.expect_outcome})")
    problems += [f"{t} did not run" for t in sc.expect_tools if t not in done]
    problems += [f"reply lacks '{w}'" for w in sc.expect_text if w.lower() not in reply.lower()]
    if sc.expect_approval and not approvals:
        problems.append("no approval was requested")
    if sc.verify and not problems:
        problems += [p for p in [sc.verify(out)] if p]
    return {"message": sc.message, "ok": not problems, "problems": problems, "outcome": out.get("outcome"),
            "steps": [f"{s['n']}. {s.get('tool_id') or s.get('tool') or '-'} [{s['status']}]"
                      + (f" error: {s['error'][:200]}" if s.get("error") else "") for s in plan],
            "approvals": approvals, "reply": reply, "error": error, "seconds": round(time.perf_counter() - t0, 1)}


def report(results: list[dict], model: str) -> str:
    passed = sum(r["ok"] for r in results)
    lines = [f"# Live check: {passed}/{len(results)} passed", "", f"Model: `{model}`, PayPal: mock", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {'PASS' if r['ok'] else 'CHECK'}: {r['message']}")
        lines.append(f"- outcome: {r['outcome']} ({r['seconds']} s)")
        if r["approvals"]:
            lines.append(f"- approval asked for: {', '.join(r['approvals'])}")
        lines += [f"- step {s}" for s in r["steps"]]
        if r["problems"]:
            lines.append(f"- problems: {'; '.join(r['problems'])}")
        if r["error"]:
            lines += ["```", r["error"].strip(), "```"]
        lines += ["", "> " + (r["reply"] or "(no reply)").replace("\n", "\n> "), ""]
    return "\n".join(lines)


def main() -> None:
    from agent.cli import _load_dotenv, quiet_logs
    from agent.config import Settings
    from agent.graph.build import build_agent

    _load_dotenv()
    quiet_logs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", help="scenario numbers to run (1-based)")
    ap.add_argument("--pause", type=float, default=5.0, help="seconds between scenarios (free-tier rate limits)")
    args = ap.parse_args()

    settings = Settings(data_dir=Path(tempfile.mkdtemp(prefix="live_check_")))
    settings.paypal_mode = "mock"  # never touch real PayPal from this script
    graph, ctx = build_agent(settings)
    chosen = [(i, s) for i, s in enumerate(SCENARIOS, 1) if not args.only or i in args.only]
    print(f"model: {settings.llm_model} | {len(chosen)} scenario(s) | PayPal: mock\n")

    results = []
    for n, (i, sc) in enumerate(chosen):
        if n:
            time.sleep(args.pause)
        r = run_one(graph, sc, thread=sc.thread)
        results.append(r)
        print(f"{i}. {'PASS ' if r['ok'] else 'CHECK'} {sc.message}  ({r['seconds']} s)"
              + (f"\n      {'; '.join(r['problems'])}" if r["problems"] else ""))

    text = report(results, settings.llm_model)
    out = ROOT / "live_check_report.md"
    out.write_text(text, encoding="utf-8")
    print(f"\n{sum(r['ok'] for r in results)}/{len(results)} passed. Full report: {out.name}")


if __name__ == "__main__":
    main()

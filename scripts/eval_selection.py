"""End-to-end tool *selection* accuracy with a real LLM: all tools bound vs. retrieved top-k.

    python scripts/eval_selection.py --limit 20 --pause 4     # uses LLM_MODEL from .env (Gemini by default)

This is the experiment behind the whole design. Condition A gives the model every PayPal tool
(112); condition B gives it only the top-k retrieved tools + the 4 built-ins. Both see the same
user message. Reports accuracy, prompt tokens and latency. Needs an API key; not run in CI.
Condition A sends ~72k tokens per question and each query is two calls, so this needs a paid key
(Gemini's free tier allows 20 requests a day). Use --pause to stay under per-minute limits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from agent.registry.build import build_paypal  # noqa: E402
from agent.retrieval.dense import get_embedder  # noqa: E402
from agent.retrieval.index import ToolIndex  # noqa: E402
from agent.tools.builtins import BUILTINS  # noqa: E402

SYSTEM = "Call exactly one tool that performs the user's request against their PayPal account. Fill arguments you know; leave out the rest."


def run(llm, queries, catalog, index, k: int, pause: float = 0.0):
    """llm: anything with call_tools(messages, tools, name=...) -> AIMessage (agent.llm.ChatLLM)."""
    all_tools = [t.to_llm_tool() for t in catalog.tools.values()]
    name_to_id = {t.name: t.id for t in catalog.tools.values()}
    stats = {"all tools": [], f"retrieved top-{k}": []}
    for q in queries:
        retrieved = [h.tool.to_llm_tool() for h in index.search(q["query"], k=k)] + [b.to_llm_tool() for b in BUILTINS.values()]
        for cond, tools in (("all tools", all_tools), (f"retrieved top-{k}", retrieved)):
            t0 = time.perf_counter()
            try:
                msg = llm.call_tools([SystemMessage(SYSTEM), HumanMessage(q["query"])], tools, name="eval_selection")
                picked = name_to_id.get(msg.tool_calls[0]["name"], msg.tool_calls[0]["name"]) if msg.tool_calls else None
                tokens = (getattr(msg, "usage_metadata", None) or {}).get("input_tokens")
                err = None
            except Exception as e:  # e.g. provider limits on number of tools / context length
                picked, tokens, err = None, None, f"{type(e).__name__}: {str(e)[:120]}"
            stats[cond].append({"ok": picked in q["gold"], "tokens": tokens, "s": time.perf_counter() - t0, "err": err,
                                "query": q["query"], "picked": picked})
            if pause:
                time.sleep(pause)  # stay under free-tier requests/tokens per minute
    return stats


def summarize(stats) -> str:
    lines = ["| condition | accuracy | avg input tokens | avg latency (s) | errors |", "|---|---|---|---|---|"]
    for cond, rows in stats.items():
        n = len(rows)
        toks = [r["tokens"] for r in rows if r["tokens"]]
        lines.append(f"| {cond} | {sum(r['ok'] for r in rows) / n:.2f} | {int(sum(toks) / len(toks)) if toks else '-'} | "
                     f"{sum(r['s'] for r in rows) / n:.2f} | {sum(bool(r['err']) for r in rows)} |")
    return "\n".join(lines)


def main() -> None:
    from agent.cli import _load_dotenv, quiet_logs
    from agent.config import Settings
    from agent.llm import ChatLLM

    _load_dotenv()
    quiet_logs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries (keep small on a free tier)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait between calls (free-tier rate limits)")
    args = ap.parse_args()
    queries = [json.loads(l) for l in (ROOT / "data/eval/paypal_tool_queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    queries = queries[: args.limit] if args.limit else queries
    catalog = build_paypal()
    index = ToolIndex(catalog, get_embedder())
    model = Settings().llm_model
    stats = run(ChatLLM(model), queries, catalog, index, args.k, args.pause)
    table = summarize(stats)
    print(table)
    out = ROOT / "data/eval/selection_results.md"
    out.write_text(f"# Tool selection eval ({model}, {len(queries)} queries)\n\n{table}\n\n"
                   + "\n".join(f"- [{c}] {r['query']} -> {r['picked']}{' ERROR ' + r['err'] if r['err'] else ''}"
                               for c, rows in stats.items() for r in rows if not r["ok"]) + "\n", encoding="utf-8")
    print(f"details -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

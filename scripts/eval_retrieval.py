"""Tool-retrieval eval: does the right tool make it into the top-k as the catalog grows?

    python scripts/eval_retrieval.py                  # lexical only (offline)
    EMBEDDINGS=openai python scripts/eval_retrieval.py  # + dense and hybrid

This is the number that matters in this design. The LLM only ever sees the top-k tools, so
end-to-end tool-selection accuracy is capped by recall@k here. If recall@k is high, the
LLM's job is choosing between ~8 tools, which it does well regardless of catalog size.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.registry.build import build_all, build_paypal  # noqa: E402
from agent.retrieval.dense import get_embedder  # noqa: E402
from agent.retrieval.index import ToolIndex  # noqa: E402

KS = (1, 3, 5, 8, 10)
QUERIES = [json.loads(l) for l in (ROOT / "data/eval/paypal_tool_queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(index: ToolIndex, mode: str, services=None):
    hits = {k: 0 for k in KS}
    mrr, misses, t0 = 0.0, [], time.perf_counter()
    for q in QUERIES:
        res = index.search(q["query"], k=max(KS), services=services, mode=mode)
        ids = [h.tool.id for h in res]
        rank = next((i + 1 for i, t in enumerate(ids) if t in q["gold"]), None)
        for k in KS:
            hits[k] += bool(rank and rank <= k)
        mrr += 1 / rank if rank else 0
        if not rank or rank > 5:
            misses.append((q["query"], q["gold"][0], ids[:3], rank))
    n = len(QUERIES)
    ms = (time.perf_counter() - t0) / n * 1000
    return {f"R@{k}": hits[k] / n for k in KS} | {"MRR": mrr / n, "ms/query": ms}, misses


def ctx_tokens(tools) -> int:
    return sum(len(json.dumps(t.to_llm_tool())) for t in tools) // 4  # ~4 chars/token


def main() -> None:
    embedder = get_embedder()
    modes = ["lexical"] + (["dense", "hybrid"] if embedder else [])
    print(f"embedder: {embedder.name if embedder else 'none (lexical only)'}; {len(QUERIES)} queries\n")

    paypal = build_paypal()
    everything = build_all()
    catalogs = {
        "PayPal only": paypal,
        "PayPal + Slack": everything.subset(["paypal", "slack"]),
        "PayPal + Slack + Twilio": everything.subset(["paypal", "slack", "twilio"]),
        "PayPal + Stripe": everything.subset(["paypal", "stripe"]),
        "All 4 services": everything,
    }

    rows, all_misses = [], {}
    for name, cat in catalogs.items():
        idx = ToolIndex(cat, embedder)
        scopes = [("unscoped", None)]
        if len(cat.services) > 1:
            scopes.append(("scoped to paypal", ["paypal"]))
        for scope_name, scope in scopes:
            for mode in modes:
                m, misses = evaluate(idx, mode, scope)
                rows.append((name, len(cat), scope_name, mode, m))
                all_misses[(name, scope_name, mode)] = misses

    header = "| catalog | #tools | scope | retrieval | R@1 | R@3 | R@5 | R@8 | R@10 | MRR | ms/query |"
    lines = [header, "|" + "---|" * 11]
    for name, n, scope, mode, m in rows:
        lines.append(f"| {name} | {n} | {scope} | {mode} | " + " | ".join(
            f"{m[f'R@{k}']:.2f}" for k in KS) + f" | {m['MRR']:.2f} | {m['ms/query']:.1f} |")
    table = "\n".join(lines)

    all_paypal = list(paypal.tools.values())
    ctx = (f"Prompt cost of tool schemas (compacted, ~4 chars/token):\n"
           f"- all {len(all_paypal)} PayPal tools bound at once: ~{ctx_tokens(all_paypal):,} tokens\n"
           f"- all {len(everything)} tools bound at once: ~{ctx_tokens(everything.tools.values()):,} tokens\n")
    idx = ToolIndex(paypal, embedder)
    per_q = [ctx_tokens([h.tool for h in idx.search(q['query'], k=8)]) for q in QUERIES]
    ctx += f"- top-8 retrieved per step: ~{sum(per_q) // len(per_q):,} tokens on average (max {max(per_q):,})\n"

    miss_lines = []
    for key in [("PayPal only", "unscoped", modes[-1]), ("All 4 services", "unscoped", modes[-1])]:
        miss_lines.append(f"\nMisses outside top-5 for {key}:")
        for q, gold, top, rank in all_misses.get(key, []):
            miss_lines.append(f"- \"{q}\" -> expected {gold} (rank {rank}); got {top}")

    out = f"{table}\n\n{ctx}" + "\n".join(miss_lines) + "\n"
    print(out)
    (ROOT / "data/eval/retrieval_results.md").write_text(
        f"# Tool retrieval eval\n\nEmbedder: {embedder.name if embedder else 'none'}; {len(QUERIES)} hand-written queries "
        f"(data/eval/paypal_tool_queries.jsonl), gold = PayPal tools.\n\n{out}", encoding="utf-8")


if __name__ == "__main__":
    main()

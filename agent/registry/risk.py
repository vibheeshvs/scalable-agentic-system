"""Heuristic risk classification. Cheap, explainable, and overridable per tool.

The point is not to be perfect; it's to make sure nothing that moves money, emails a
customer or deletes data runs without a human saying yes. False positives just cost a
confirmation click, false negatives cost real money, so the rules lean towards HIGH.
"""

from __future__ import annotations

import re

from .models import Risk

HIGH_VERBS = {
    "send", "resend", "remind", "pay", "payment", "payments", "payout", "payouts", "refund", "refunds",
    "capture", "cancel", "void", "delete", "remove", "transfer", "charge", "authorize",
    "reauthorize", "accept", "deny", "escalate", "appeal", "adjudicate", "settle", "suspend",
    "activate", "deactivate", "revise", "close", "approve", "finalize", "terminate", "revoke",
    "reverse", "disable", "archive", "kick", "invite", "revert", "offer", "pricing", "evidence", "acknowledge",
}
READ_VERBS = {"search", "list", "find", "lookup", "verify", "validate", "calculate", "preview", "retrieve", "get", "show"}


def words(*texts: str) -> set[str]:
    out: set[str] = set()
    for t in texts:
        t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t or "")
        out.update(w.lower() for w in re.split(r"[^A-Za-z]+", t) if w)
    return out


def classify(method: str, path: str, summary: str = "", op_id: str = "") -> Risk:
    method = method.upper()
    if method in ("GET", "HEAD", "OPTIONS"):
        return Risk.READ
    if method == "DELETE":
        return Risk.HIGH
    # look at the *action* part: last path segment + operation id + summary
    last_seg = path.rstrip("/").split("/")[-1] if path else ""
    w = words(last_seg, op_id, summary)
    if w & HIGH_VERBS:
        return Risk.HIGH
    first = next(iter(words(summary.split(" ")[0] if summary else "")), "")
    if method == "POST" and (first in READ_VERBS or "search" in w):
        return Risk.READ
    return Risk.WRITE

"""Shrink API responses before they go back into the prompt.

A single PayPal transaction search can return hundreds of records with HATEOAS links on every
object. The LLM gets a preview (first few items, no links, long strings cut) plus the shape
of the data; the full payload stays in state for analyze_data to crunch deterministically.
"""

from __future__ import annotations

import json
from typing import Any

DROP_KEYS = {"links", "_links"}


def shrink(data: Any, max_items: int = 5, max_str: int = 200, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return "..."
    if isinstance(data, dict):
        return {k: shrink(v, max_items, max_str, depth + 1, max_depth) for k, v in data.items() if k not in DROP_KEYS}
    if isinstance(data, list):
        out = [shrink(v, max_items, max_str, depth + 1, max_depth) for v in data[:max_items]]
        if len(data) > max_items:
            out.append(f"... ({len(data) - max_items} more items, {len(data)} total)")
        return out
    if isinstance(data, str) and len(data) > max_str:
        return data[:max_str] + "..."
    return data


def preview(data: Any, max_chars: int = 3000, max_str: int = 200) -> str:
    items = 5
    while True:
        text = json.dumps(shrink(data, max_items=items, max_str=max_str), default=str)
        if len(text) <= max_chars or items == 1:
            return text if len(text) <= max_chars else text[:max_chars] + "...(truncated)"
        items = max(1, items // 2)


def list_fields(data: Any, prefix: str = "") -> list[str]:
    """Top-level lists and their sizes, so the LLM knows what analyze_data can work on."""
    out = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k in DROP_KEYS:
                continue
            if isinstance(v, list) and v and isinstance(v[0], dict):
                out.append(f"{prefix}{k} ({len(v)} items)")
            elif isinstance(v, dict) and not prefix:
                out.extend(list_fields(v, prefix=f"{k}."))
    return out

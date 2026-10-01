"""JSON-schema helpers: $ref resolution + compaction.

Real API specs have enormous request schemas (PayPal's invoice body is ~6 levels deep with
hundreds of fields). Dumping that into the prompt burns tokens and *increases* hallucinated
parameters, so every schema is compacted before it's shown to the LLM:
  - $refs resolved, allOf merged, readOnly fields dropped
  - depth capped, long descriptions/enums trimmed, huge objects capped (required fields first)
The full spec is still used by the executor; only the LLM's view is compacted.
"""

from __future__ import annotations

import html
import re
from typing import Any

MAX_DEPTH = 4       # deeper than this and the LLM gets "nested object, fields omitted"
MAX_PROPS = 25
MAX_PROPS_NESTED = 8  # objects at depth >= 3 keep required fields + the first few others
MAX_ENUM = 12
MAX_DESC = 160
MAX_DESC_NESTED = 70

_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str | None, limit: int | None = None) -> str:
    if not text:
        return ""
    t = html.unescape(_TAG_RE.sub(" ", text))
    t = re.sub(r"`", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if limit and len(t) > limit:
        t = t[: limit - 3].rsplit(" ", 1)[0] + "..."
    return t


class RefResolver:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec

    def resolve(self, ref: str) -> dict[str, Any]:
        if not ref.startswith("#/"):
            return {}
        node: Any = self.spec
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                return {}
            node = node[part]
        return node if isinstance(node, dict) else {}

    def deref(self, obj: Any) -> Any:
        """Resolve a top-level $ref only (used for parameters / requestBodies)."""
        seen = set()
        while isinstance(obj, dict) and "$ref" in obj and obj["$ref"] not in seen:
            seen.add(obj["$ref"])
            obj = self.resolve(obj["$ref"])
        return obj


def compact(schema: Any, resolver: RefResolver, depth: int = 0, stack: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return a small, LLM-friendly JSON schema."""
    if not isinstance(schema, dict):
        return {}
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in stack:  # recursive type, stop
            return {"type": "object", "description": "recursive structure (omitted)"}
        target = resolver.resolve(ref)
        merged = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
        return compact(merged, resolver, depth, stack + (ref,))

    if "allOf" in schema:
        props: dict[str, Any] = {}
        required: list[str] = []
        scalar: dict[str, Any] | None = None
        desc = schema.get("description") or schema.get("title")
        for part in schema["allOf"]:
            sub = compact(part, resolver, depth, stack)
            props.update(sub.get("properties", {}))
            required += sub.get("required", [])
            desc = desc or sub.get("description")
            if sub.get("type") not in (None, "object") and not sub.get("properties"):
                scalar = sub  # allOf used to decorate a primitive (e.g. string + description)
        if scalar is not None and not props:
            merged = dict(scalar)
        else:
            merged = {"type": "object"}
            if props:
                merged["properties"] = props
            if required:
                merged["required"] = sorted(set(required), key=required.index)
        if desc:
            merged["description"] = desc
        return compact_leaf(merged)

    for key in ("oneOf", "anyOf"):
        if key in schema:
            options = [compact(o, resolver, depth, stack) for o in schema[key][:3]]
            options = [o for o in options if o]
            if len(options) == 1:
                return options[0]
            out: dict[str, Any] = {"anyOf": options}
            if schema.get("description"):
                out["description"] = clean_text(schema["description"], MAX_DESC)
            return out

    out = {}
    typ = schema.get("type")
    if isinstance(typ, list):  # OpenAPI 3.1 style ["string","null"]
        typ = next((t for t in typ if t != "null"), "string")
    if typ is None and "properties" in schema:
        typ = "object"
    if typ:
        out["type"] = typ
    desc = schema.get("description") or schema.get("title")
    if desc and depth < 3:  # field names are self-explanatory deep down; descriptions there cost a lot
        out["description"] = clean_text(desc, MAX_DESC if depth < 2 else MAX_DESC_NESTED)
    # patterns / length limits are mostly noise ("^.*$", maxLength 2147483647); the API
    # validates them anyway and returns a clear 400 that the repair loop can act on.
    for k in ("format", "minimum", "maximum", "default"):
        if k in schema and not isinstance(schema[k], (dict, list)):
            out[k] = schema[k]
    if "enum" in schema:
        out["enum"] = schema["enum"][:MAX_ENUM]

    if typ == "object" or "properties" in schema:
        if depth >= MAX_DEPTH:
            out["description"] = (out.get("description", "") + " (nested object, fields omitted)").strip()
            return out
        props = schema.get("properties") or {}
        required = [r for r in schema.get("required", []) if r in props]
        # required first, then the rest, dropping read-only (server generated) fields
        ordered = required + [p for p in props if p not in required]
        new_props = {}
        for name in ordered:
            p = props[name]
            if isinstance(p, dict) and "$ref" in p:
                target = resolver.resolve(p["$ref"])
                if target.get("readOnly"):
                    continue
            if isinstance(p, dict) and p.get("readOnly"):
                continue
            if len(new_props) >= (MAX_PROPS if depth < 3 else MAX_PROPS_NESTED) and name not in required:
                break
            new_props[name] = compact(p, resolver, depth + 1, stack)
        if new_props:
            out["properties"] = new_props
        req = [r for r in required if r in new_props]
        if req:
            out["required"] = req
    elif typ == "array":
        items = schema.get("items")
        if items:
            # the array and its items are one level for the reader, so don't charge depth twice
            out["items"] = compact(items, resolver, depth, stack)
        if not out.get("items"):
            out["items"] = {"type": "string"}  # OpenAI rejects arrays without an items schema
    return out


def compact_leaf(s: dict[str, Any]) -> dict[str, Any]:
    if "description" in s:
        s["description"] = clean_text(s["description"], MAX_DESC)
    return s


def infer_schema(example: Any, depth: int = 0) -> dict[str, Any]:
    """Infer a JSON schema from an example value (used for Postman bodies, which have no schema)."""
    if isinstance(example, bool):
        return {"type": "boolean"}
    if isinstance(example, int):
        return {"type": "integer"}
    if isinstance(example, float):
        return {"type": "number"}
    if isinstance(example, str):
        return {"type": "string", "description": f"e.g. {example[:60]}"} if example else {"type": "string"}
    if isinstance(example, list):
        return {"type": "array", "items": infer_schema(example[0], depth + 1) if example else {"type": "string"}}
    if isinstance(example, dict):
        if depth >= MAX_DEPTH:
            return {"type": "object"}
        return {"type": "object", "properties": {k: infer_schema(v, depth + 1) for k, v in list(example.items())[:MAX_PROPS]}}
    return {}


def schema_size(schema: dict[str, Any]) -> int:
    import json

    return len(json.dumps(schema))

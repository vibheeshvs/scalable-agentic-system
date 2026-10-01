"""Postman collection (v2.1) -> ToolSpec loader.

Postman collections don't carry parameter schemas, only example requests, so the loader
infers them: `:param` / `{{param}}` path segments become required path params, query
params keep their example values as hints, and a raw JSON body is turned into a schema by
type inference. When an OpenAPI spec for the same API exists I prefer it (better schemas),
but this lets you drop in any exported collection, which is how most internal APIs live.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import ServiceInfo, ToolSpec
from .openapi import llm_name, slug
from .risk import classify
from .schema import clean_text, infer_schema

VAR_RE = re.compile(r"^\{\{(.+)\}\}$")
# Token requests sit at the top of most collections. Auth is the executor's job, never an agent tool.
AUTH_PATH_RE = re.compile(r"/oauth2?/token$|/identity/(openidconnect|oauth2)/", re.I)


def load_postman(source: str | Path | dict[str, Any], service: str, base_url: str = "") -> tuple[list[ToolSpec], ServiceInfo]:
    col = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    info = col.get("info", {})
    svc = ServiceInfo(name=service, title=info.get("name", service),
                      description=clean_text(_desc(info.get("description")), 300), base_url=base_url)
    tools: list[ToolSpec] = []
    seen: set[str] = set()
    for folder_path, item in _walk(col.get("item", []), []):
        req = item.get("request")
        if not isinstance(req, dict):
            continue
        if not req.get("url"):  # empty placeholder request
            continue
        method = req.get("method", "GET").upper()
        path, path_params, query = _parse_url(req.get("url"))
        if AUTH_PATH_RE.search(path):
            continue
        group = slug(folder_path[-1]).replace("_", "-") if folder_path else _first_seg(path)
        op = f"{group}.{slug(item.get('name', method + path)).replace('_', '-')}"
        # collections reuse names a lot ("Invoice" as GET / PUT / DELETE, or two saved variants of one call)
        tool_id, n = f"{service}.{op}", 2
        if tool_id in seen:
            tool_id += f".{method.lower()}"
        while tool_id in seen:
            tool_id = f"{service}.{op}.{method.lower()}-{n}"
            n += 1
        seen.add(tool_id)
        op = tool_id.split(".", 1)[1]  # the LLM function name must be unique too

        props: dict[str, Any] = {}
        required: list[str] = []
        if path_params:
            props["path"] = {"type": "object", "properties": {p: {"type": "string"} for p in path_params},
                             "required": path_params}
            required.append("path")
        if query:
            props["query"] = {"type": "object", "properties": {
                q["key"]: {"type": "string", **({"description": f"e.g. {q['value']}"} if q.get("value") else {})}
                for q in query if q.get("key")}}
        body = req.get("body") or {}
        encoding = "none"
        if body.get("mode") == "raw" and body.get("raw", "").strip():
            try:
                example = json.loads(_strip_vars(body["raw"]))
                props["body"] = infer_schema(example)
                encoding = "json"
            except json.JSONDecodeError:
                props["body"] = {"type": "object", "description": "raw body"}
                encoding = "json"
        elif body.get("mode") in ("urlencoded", "formdata"):  # file parts can't be filled in by an LLM, so text fields only
            props["body"] = {"type": "object", "properties": {
                f["key"]: {"type": "string"} for f in body.get(body["mode"]) or []
                if f.get("key") and not f.get("disabled") and f.get("type") != "file"}}
            encoding = "form"

        summary = clean_text(item.get("name"), 150)
        params: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            params["required"] = required
        tools.append(ToolSpec(
            id=tool_id, name=llm_name(service, op), service=service, group=group, method=method,
            path=path, base_url=base_url, summary=summary,
            description=clean_text(_desc(req.get("description") or item.get("description")), 700),
            parameters=params, body_encoding=encoding,
            risk=classify(method, path, summary, op), source=f"postman:{svc.title}",
        ))
    return tools, svc


def _walk(items: list[dict[str, Any]], prefix: list[str]):
    for it in items:
        if "item" in it:  # folder
            yield from _walk(it["item"], prefix + [it.get("name", "")])
        else:
            yield prefix, it


def _desc(d: Any) -> str:
    if isinstance(d, dict):
        return d.get("content", "")
    return d or ""


def _strip_vars(raw: str) -> str:
    # "{{invoice_id}}" inside JSON strings is fine; bare {{var}} (non-string) would break json.loads
    return re.sub(r'(?<!")\{\{([^}]+)\}\}(?!")', r'"\1"', raw)


def _parse_url(url: Any) -> tuple[str, list[str], list[dict[str, Any]]]:
    if isinstance(url, str):
        raw = url
        segs = re.sub(r"^\{\{[^}]+\}\}", "", raw.split("?")[0]).strip("/").split("/")
        query = []
        if "?" in raw:
            for kv in raw.split("?", 1)[1].split("&"):
                k, _, v = kv.partition("=")
                query.append({"key": k, "value": v})
    else:
        segs = url.get("path") or []
        if isinstance(segs, str):  # the v2.1 schema allows the path as one string
            segs = segs.strip("/").split("/")
        elif not segs and url.get("raw"):
            return _parse_url(url["raw"])
        segs = [s if isinstance(s, str) else str(s.get("value", "")) for s in segs]
        query = [q for q in url.get("query") or [] if not q.get("disabled")]
    out, params = [], []
    for s in segs:
        if not s:
            continue
        m = VAR_RE.match(s)
        if s.startswith(":"):
            params.append(s[1:])
            out.append("{" + s[1:] + "}")
        elif m:
            params.append(m.group(1))
            out.append("{" + m.group(1) + "}")
        else:
            out.append(s)
    return "/" + "/".join(out), params, query


def _first_seg(path: str) -> str:
    segs = [s for s in path.strip("/").split("/") if s and not s.startswith("{") and not re.match(r"^v\d+$", s)]
    return segs[0] if segs else "root"

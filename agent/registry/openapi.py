"""OpenAPI 3.x / Swagger 2.0 -> ToolSpec loader."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .models import ServiceInfo, ToolSpec
from .risk import classify
from .schema import RefResolver, clean_text, compact

METHODS = ("get", "post", "put", "patch", "delete")
SKIP_PARAMS = {"token", "authorization", "paypal-request-id", "paypal-auth-assertion", "paypal-partner-attribution-id",
               "prefer", "content-type", "accept", "paypal-client-metadata-id", "paypal-mock-response"}
IDEMPOTENCY_HEADERS = {"paypal-request-id", "idempotency-key"}


def slug(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def llm_name(service: str, op: str) -> str:
    name = f"{service}__{slug(op)}"
    if len(name) > 64:
        h = hashlib.md5(name.encode()).hexdigest()[:6]
        name = name[:57] + "_" + h
    return name


def _group_for(path: str, op: dict[str, Any], version_re=re.compile(r"^v\d+(\.\d+)?$")) -> str:
    tags = op.get("tags") or []
    if tags:
        g = re.sub(r"^api_?\d+_?", "", slug(tags[0]))  # Twilio: Api20100401Message -> message
        return g.replace("_", "-") or "root"
    segs = [s for s in path.strip("/").split("/") if s and not s.startswith("{") and not version_re.match(s)]
    # Twilio-style /2010-04-01/Accounts/{AccountSid}/Messages.json -> "messages"
    segs = [s for s in segs if not re.match(r"^\d{4}-\d{2}-\d{2}$", s)]
    if segs and segs[0].lower() == "accounts" and len(segs) > 1:
        segs = segs[1:]
    if not segs:
        return "root"
    first = segs[0].split(".")[0] if "." in segs[0] and segs[0].endswith(".json") else segs[0]
    # Slack-style /chat.postMessage -> "chat"
    first = first.split(".")[0]
    return slug(first).replace("_", "-") or "root"


def load_openapi(source: str | Path | dict[str, Any], service: str, base_url: str | None = None,
                 title: str | None = None, idempotency_header: str | None = None) -> tuple[list[ToolSpec], ServiceInfo]:
    """`idempotency_header` is a service-wide default for providers that accept one on every write but
    document it outside the spec (Stripe). A header declared on the operation itself always wins."""
    spec = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    resolver = RefResolver(spec)
    swagger2 = str(spec.get("swagger", "")).startswith("2")

    if base_url is None:
        if swagger2:
            scheme = (spec.get("schemes") or ["https"])[0]
            base_url = f"{scheme}://{spec.get('host', '')}{spec.get('basePath', '')}".rstrip("/")
        else:
            base_url = (spec.get("servers") or [{"url": ""}])[0]["url"].rstrip("/")

    info = spec.get("info", {})
    svc = ServiceInfo(
        name=service,
        title=title or info.get("title", service),
        description=clean_text(info.get("description"), 300),
        base_url=base_url,
    )

    tools: list[ToolSpec] = []
    seen: set[str] = set()
    for path, path_item in (spec.get("paths") or {}).items():
        path_item = resolver.deref(path_item)
        shared_params = path_item.get("parameters", [])
        for method in METHODS:
            op = path_item.get(method)
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId") or f"{method}_{path}"
            op_slug = slug(op_id).replace("_", "-") if "." not in op_id else op_id.lower()
            tool_id = f"{service}.{op_slug}"
            if tool_id in seen:
                tool_id = f"{tool_id}.{method}"
            seen.add(tool_id)

            params, encoding = _build_params(op, shared_params, resolver, swagger2)
            summary = clean_text(op.get("summary"), 150) or clean_text(op.get("description"), 120) or op_id
            tools.append(ToolSpec(
                id=tool_id,
                name=llm_name(service, tool_id.split(".", 1)[1]),
                service=service,
                group=_group_for(path, op),
                method=method.upper(),
                path=path,
                base_url=base_url,
                summary=summary,
                description=clean_text(op.get("description"), 700),
                parameters=params,
                body_encoding=encoding,
                risk=classify(method, path, summary, op_id),
                idempotency_header=None if method == "get" else (
                    _declared_idempotency_header(op, shared_params, resolver)
                    or (idempotency_header if method == "post" else None)),
                source=f"openapi:{info.get('title', service)}",
            ))
    return tools, svc


def _declared_idempotency_header(op: dict[str, Any], shared: list[Any], resolver: RefResolver) -> str | None:
    for p in list(shared) + list(op.get("parameters", [])):
        p = resolver.deref(p)
        if isinstance(p, dict) and p.get("in") == "header" and str(p.get("name", "")).lower() in IDEMPOTENCY_HEADERS:
            return p["name"]
    return None


def _build_params(op: dict[str, Any], shared: list[Any], resolver: RefResolver, swagger2: bool):
    by_loc: dict[str, dict[str, Any]] = {"path": {}, "query": {}, "body": {}}
    required: dict[str, list[str]] = {"path": [], "query": [], "body": []}
    encoding = "none"
    body_schema: dict[str, Any] | None = None
    body_required = False

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for p in list(shared) + list(op.get("parameters", [])):
        p = resolver.deref(p)
        if isinstance(p, dict) and "name" in p:
            merged[(p.get("in", ""), p["name"])] = p

    for (loc, name), p in merged.items():
        if name.lower() in SKIP_PARAMS or loc in ("header", "cookie"):
            continue
        if swagger2 and loc == "body":
            body_schema = compact(p.get("schema", {}), resolver)
            body_required = bool(p.get("required"))
            encoding = "json"
            continue
        if swagger2 and loc == "formData":
            loc = "body"
            encoding = "form"
        schema = p.get("schema") or {k: v for k, v in p.items() if k in ("type", "enum", "format", "items", "default")}
        s = compact(schema, resolver)
        desc = clean_text(p.get("description"), 160)
        if desc:
            s["description"] = desc
        target = "body" if loc == "body" else loc
        if target not in by_loc:
            continue
        by_loc[target][name] = s
        if p.get("required") or loc == "path":
            required[target].append(name)

    rb = resolver.deref(op.get("requestBody")) if op.get("requestBody") else None
    if rb:
        content = rb.get("content", {})
        for ctype in ("application/json", "application/x-www-form-urlencoded", "multipart/form-data"):
            if ctype in content:
                body_schema = compact(content[ctype].get("schema", {}), resolver)
                encoding = "json" if ctype == "application/json" else "form"
                break
        body_required = bool(rb.get("required"))

    props: dict[str, Any] = {}
    top_required: list[str] = []
    for loc in ("path", "query"):
        if by_loc[loc]:
            sub = {"type": "object", "properties": by_loc[loc]}
            if required[loc]:
                sub["required"] = required[loc]
                top_required.append(loc)
            props[loc] = sub
    if body_schema is not None and (body_schema.get("properties") or body_schema.get("type") not in (None, "object")):
        if by_loc["body"]:  # swagger2 formData merged with body schema (rare)
            body_schema.setdefault("properties", {}).update(by_loc["body"])
        props["body"] = body_schema
        if body_required:
            top_required.append("body")
    elif by_loc["body"]:
        sub = {"type": "object", "properties": by_loc["body"]}
        if required["body"]:
            sub["required"] = required["body"]
            top_required.append("body")
        props["body"] = sub
    elif body_schema is not None:
        props["body"] = {"type": "object", "description": "request body (free-form JSON object)"}
    if encoding == "none" and "body" in props:
        encoding = "json"

    params: dict[str, Any] = {"type": "object", "properties": props}
    if top_required:
        params["required"] = top_required
    return params, encoding

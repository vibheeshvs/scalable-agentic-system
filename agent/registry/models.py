"""Data model for the tool catalog.

A tool is *data*, not code. Every API operation (from OpenAPI, Postman, MCP, whatever)
is normalised into a ToolSpec, and a single generic executor knows how to call any of
them. That is what lets the catalog grow to thousands of entries without anyone writing
a Python function per endpoint.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Risk(str, Enum):
    READ = "read"    # no side effects (GET, search)
    WRITE = "write"  # creates/updates something reversible (draft invoice, product)
    HIGH = "high"    # moves money, notifies a customer, or deletes (send, refund, payout, cancel)


class ToolSpec(BaseModel):
    id: str                                   # "paypal.invoices.send" - stable, namespaced
    name: str                                 # LLM-safe function name, <= 64 chars
    service: str                              # "paypal"
    group: str                                # "invoices" (tag / folder / first path segment)
    kind: Literal["http", "builtin"] = "http"
    method: str = "GET"
    path: str = ""
    base_url: str = ""
    summary: str = ""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    body_encoding: Literal["json", "form", "none"] = "json"
    risk: Risk = Risk.READ
    # header the provider documents for replaying this write safely (PayPal-Request-Id, Idempotency-Key);
    # None = the provider gives no such guarantee here, so the executor must not retry it blindly
    idempotency_header: str | None = None
    examples: list[str] = Field(default_factory=list)   # synthetic user queries (doc2query), optional
    source: str = ""

    # ---- text used for retrieval -------------------------------------------------------
    def search_text(self) -> str:
        """Field-weighted text for lexical indexing (weights done by repetition, BM25-style)."""
        path_words = re.sub(r"[{}/_\-.]", " ", self.path)
        op_words = re.sub(r"[._\-]", " ", self.id.split(".", 1)[-1])
        params = " ".join(self._param_names())
        parts = [
            self.summary, self.summary, self.summary,
            self.group, self.group,
            op_words, op_words,
            path_words,
            self.description[:600],
            params,
            " ".join(self.examples), " ".join(self.examples),
        ]
        return " ".join(p for p in parts if p)

    def embed_text(self) -> str:
        """Shorter natural-language text for dense embeddings."""
        ex = f" Example requests: {'; '.join(self.examples[:5])}" if self.examples else ""
        return f"{self.service} {self.group}: {self.summary}. {self.description[:400]}{ex}"

    def _param_names(self) -> list[str]:
        names: list[str] = []
        for loc in ("path", "query", "body"):
            sub = self.parameters.get("properties", {}).get(loc, {})
            names.extend(sub.get("properties", {}).keys())
        return names

    # ---- what the LLM sees ---------------------------------------------------------------
    def to_llm_tool(self) -> dict[str, Any]:
        """OpenAI-style function schema. LangChain converts it for Anthropic etc."""
        desc = self.summary
        if self.description and self.description.lower() != self.summary.lower():
            desc = f"{self.summary}. {self.description[:300]}"
        if self.kind == "http":
            desc = f"[{self.service} | {self.method} {self.path} | risk={self.risk.value}] {desc}"
        return {
            "type": "function",
            "function": {"name": self.name, "description": desc[:1000], "parameters": self.parameters},
        }

    def card(self) -> str:
        """One-line description used by the system-search tool and the planner."""
        if self.kind == "builtin":
            return f"{self.name}: {self.summary}"
        return f"{self.id} ({self.method} {self.path}) - {self.summary} [risk: {self.risk.value}]"


class ServiceInfo(BaseModel):
    name: str
    title: str = ""
    description: str = ""
    base_url: str = ""
    groups: dict[str, int] = Field(default_factory=dict)  # group -> number of tools


class Catalog(BaseModel):
    """All tools plus a small service/group summary that is cheap enough to show the planner."""

    tools: dict[str, ToolSpec] = Field(default_factory=dict)
    services: dict[str, ServiceInfo] = Field(default_factory=dict)

    def add(self, tools: list[ToolSpec], service: ServiceInfo | None = None) -> None:
        if service:
            self.services.setdefault(service.name, service)
        for t in tools:
            if t.id in self.tools:
                raise ValueError(f"duplicate tool id {t.id}")
            self.tools[t.id] = t
            info = self.services.setdefault(t.service, ServiceInfo(name=t.service))
            info.groups[t.group] = info.groups.get(t.group, 0) + 1

    def by_name(self, name: str) -> ToolSpec | None:
        for t in self.tools.values():
            if t.name == name:
                return t
        return None

    def get(self, tool_id: str) -> ToolSpec | None:
        return self.tools.get(tool_id)

    def __len__(self) -> int:
        return len(self.tools)

    def subset(self, services: list[str]) -> "Catalog":
        c = Catalog()
        for s in services:
            if s in self.services:
                c.services[s] = self.services[s].model_copy(deep=True)
        c.tools = {k: v for k, v in self.tools.items() if v.service in services}
        return c

    def overview(self, max_groups: int = 40) -> str:
        """Service -> groups map. Grows with #groups, not #tools, so it stays small."""
        lines = []
        for s in self.services.values():
            groups = sorted(s.groups.items(), key=lambda kv: -kv[1])
            shown = ", ".join(f"{g} ({n})" for g, n in groups[:max_groups])
            more = f", +{len(groups) - max_groups} more" if len(groups) > max_groups else ""
            title = f" - {s.title}" if s.title else ""
            lines.append(f"- {s.name}{title}: {shown}{more}")
        return "\n".join(lines)

    # ---- persistence -------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=1), encoding="utf-8", newline="\n")  # same bytes on every OS

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

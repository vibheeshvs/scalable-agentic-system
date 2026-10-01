"""Thin LLM wrapper so graph nodes don't care which provider (or a scripted fake) is behind it.

Every call is named ("router", "planner", "selector", ...) which shows up as the run name in
LangSmith, and lets the test double return scripted answers per node.

Provider notes
- Gemini (default, free tier): tool schemas are sent to Gemini as plain JSON Schema
  (`parameters_json_schema`). LangChain's generic Gemini converter marks every nested field
  without a default as *required*, which would force the model to invent values for every
  optional query parameter, so we bypass it. Gemini 3 models are also left at their default
  temperature, as Google recommends. Gemini also refuses a forced function call when the schemas
  are too large, so oversized ones are trimmed on the way out (see `fit_request`).
- OpenAI / Anthropic / others go through LangChain's normal `bind_tools`.
"""

from __future__ import annotations

import copy
import os
import re
import time
from typing import Any, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))  # extra rounds after a temporary failure (overload, per-minute limit)


class LLMError(RuntimeError):
    """The model provider failed (quota, overload, refused request). Nodes turn this into a failed step or turn."""


class LLM(Protocol):
    def structured(self, schema: type[T], messages: list[BaseMessage], *, name: str) -> T: ...
    def call_tools(self, messages: list[BaseMessage], tools: list[dict[str, Any]], *, name: str) -> AIMessage: ...
    def text(self, messages: list[BaseMessage], *, name: str) -> str: ...


def make_chat_model(model: str | BaseChatModel) -> BaseChatModel:
    if not isinstance(model, str):
        return model
    from langchain.chat_models import init_chat_model

    # No SDK retries: they come within a second or two of each other, which doesn't outlast an overload, and
    # every attempt counts against the free tier's daily quota. ChatLLM does its own, slower, retrying.
    if model.startswith(("google_genai:", "google_vertexai:")):
        return init_chat_model(model, max_retries=0)  # Gemini 3: keep the default temperature
    return init_chat_model(model, temperature=0, max_retries=0)


_OVERLOADED = re.compile(r"\b(500|502|503|504|529)\b|UNAVAILABLE|overloaded|timed out|timeout", re.I)
_RATE_LIMITED = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|rate.?limit", re.I)


def retry_wait(error: str, attempt: int) -> float | None:
    """Seconds to wait before trying a failed model call again, or None when waiting can't help."""
    if "PerDay" in error:  # the daily quota is gone; a pause of seconds changes nothing
        return None
    if _RATE_LIMITED.search(error):  # per-minute limit: the provider usually says how long
        hinted = re.search(r"retry in ([\d.]+)s", error)
        return min(60.0, float(hinted.group(1)) + 1) if hinted else 20.0
    if _OVERLOADED.search(error):
        return (5.0, 15.0, 30.0)[min(attempt, 2)]
    return None


def message_text(content: Any) -> str:
    """AIMessage.content can be a string or a list of parts (Gemini, Anthropic)."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content or [])


def is_gemini(model: BaseChatModel) -> bool:
    return type(model).__name__ == "ChatGoogleGenerativeAI"


# With a forced function call (mode ANY) Gemini answers a bare "400 INVALID_ARGUMENT" when the tool schemas
# are too big. The limit isn't documented, so these two numbers are measured against the live API:
#   - one function: paypal.orders.confirm (271 schema nodes) is accepted, orders.create cut to 297 is rejected,
#     with or without its enums and descriptions;
#   - whole request: nodes + enum values over all functions. Accepted up to 443, rejected from 468.
# About a quarter of the eval queries retrieve a top-8 that breaks one of the two (anything near the order,
# invoice or payment schemas), so both are enforced here, on the way out, and the catalog stays complete.
GEMINI_MAX_SCHEMA_NODES = 270
GEMINI_REQUEST_BUDGET = 440
SMALL_SCHEMA = 30  # built-ins and simple GETs: never shortened, there is nothing to gain


def _children(schema: dict[str, Any]) -> list[Any]:
    return [*(schema.get("properties") or {}).values(), schema.get("items"), *(schema.get("anyOf") or [])]


def schema_nodes(schema: Any) -> int:
    return 1 + sum(schema_nodes(c) for c in _children(schema)) if isinstance(schema, dict) else 0


def schema_cost(schema: Any) -> int:
    """Nodes + enum values: what the request-level limit appears to count."""
    if not isinstance(schema, dict):
        return 0
    return 1 + len(schema.get("enum") or []) + sum(schema_cost(c) for c in _children(schema))


def _optional_fields(schema: Any, depth: int = 0, out: list | None = None) -> list[tuple[int, dict, str]]:
    """(depth, the properties dict it lives in, name) for every field its parent does not require."""
    out = [] if out is None else out
    if isinstance(schema, dict):
        required = set(schema.get("required") or [])
        for name, sub in (schema.get("properties") or {}).items():
            if name not in required:
                out.append((depth + 1, schema["properties"], name))
            _optional_fields(sub, depth + 1, out)
        for sub in (schema.get("items"), *(schema.get("anyOf") or [])):
            _optional_fields(sub, depth, out)  # an array and its items are one level for the reader
    return out


def _hollow(schema: Any) -> bool:
    """True if an object anywhere inside lost all of its fields. Under a forced call the model can only
    answer {} for such an object, so the optional field that contains it is better left out altogether."""
    if not isinstance(schema, dict):
        return False
    if "properties" in schema and not schema["properties"]:
        return True
    return any(_hollow(child) for child in _children(schema))


def fit_schema(schema: dict[str, Any], max_nodes: int = GEMINI_MAX_SCHEMA_NODES,
               max_cost: int = GEMINI_REQUEST_BUDGET) -> dict[str, Any]:
    """Shorten a schema by dropping whole optional fields, deepest first, until it fits.

    Required fields are never touched and no object is left without its fields, so everything that remains can
    still be filled in properly. (Cutting by depth instead leaves empty objects, and with a forced function call
    the model can only send {} for those: the first live run sent `unit_amount: {}` three times in a row.)
    """
    nodes, cost = schema_nodes(schema), schema_cost(schema)
    if nodes <= max_nodes and cost <= max_cost:
        return schema
    out = copy.deepcopy(schema)
    fields = _optional_fields(out)
    order = sorted(range(len(fields)), key=lambda i: (-fields[i][0], -i))  # deepest first, later fields before earlier
    for i in order:
        depth, props, name = fields[i]
        if depth < 2 or (nodes <= max_nodes and cost <= max_cost):
            break  # depth 1 is path / query / body themselves
        sub = props.pop(name, None)
        nodes, cost = nodes - schema_nodes(sub), cost - schema_cost(sub)
    for i in order:  # an optional field left holding an emptied object goes too (deepest first, so the smallest cut)
        depth, props, name = fields[i]
        if depth >= 2 and _hollow(props.get(name)):
            del props[name]
    return out


def fit_alone(schema: dict[str, Any]) -> dict[str, Any]:
    """The most of one schema Gemini takes in a request of its own (with room left for ask_user)."""
    return fit_schema(schema, max_cost=GEMINI_REQUEST_BUDGET - SMALL_SCHEMA)


def fit_request(schemas: list[dict[str, Any]], budget: int = GEMINI_REQUEST_BUDGET) -> list[dict[str, Any]]:
    """Make a whole tool list acceptable. Each schema is first fitted on its own; if the request is still too
    big, the lowest-ranked tools are shortened first (the list is in retrieval order), so the likeliest
    candidates keep everything."""
    fitted = [fit_alone(s) for s in schemas]
    total = sum(map(schema_cost, fitted))
    for i in range(len(fitted) - 1, -1, -1):
        cost = schema_cost(fitted[i])
        if total <= budget:
            break
        if cost > SMALL_SCHEMA:
            fitted[i] = fit_schema(fitted[i], max_cost=max(SMALL_SCHEMA, cost - (total - budget)))
            total -= cost - schema_cost(fitted[i])
    return fitted


def gemini_tool(tools: list[dict[str, Any]], schemas: list[dict[str, Any]] | None = None):
    """OpenAI-style tool dicts -> one google.genai Tool whose declarations carry raw JSON Schema."""
    from google.genai import types

    schemas = schemas if schemas is not None else fit_request([t["function"]["parameters"] for t in tools])
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(name=t["function"]["name"], description=t["function"].get("description", ""),
                                  parameters_json_schema=s)
        for t, s in zip(tools, schemas)])


class ChatLLM:
    def __init__(self, model: str | BaseChatModel, fast_model: str | BaseChatModel | None = None,
                 fallback_model: str | BaseChatModel | None = None):
        self.model = make_chat_model(model)
        self.fast = make_chat_model(fast_model) if fast_model and fast_model != model else self.model
        self.fallback = make_chat_model(fallback_model) if fallback_model else None
        self.sleep = time.sleep

    def _pick(self, name: str) -> BaseChatModel:
        return self.fast if name == "router" else self.model

    def structured(self, schema, messages, *, name):
        return self._guard(name, self._structured, schema, messages, name)

    def call_tools(self, messages, tools, *, name):
        return self._guard(name, self._call_tools, messages, tools, name)

    def text(self, messages, *, name):
        return self._guard(name, self._text, messages, name)

    def _guard(self, name, fn, *args):
        """Run one call. If the model fails, try the fallback model straight away (an overload or a quota is per
        model); if that fails too and the cause is temporary, wait and go round again. Provider SDKs raise their
        own exception types, so whatever is left at the end becomes one LLMError for the graph to handle."""
        models = [self._pick(name)] + ([self.fallback] if self.fallback is not None else [])
        for attempt in range(LLM_MAX_RETRIES + 1):
            for model in models:
                try:
                    return fn(model, *args)
                except Exception as e:
                    error = e
            wait = retry_wait(str(error), attempt)
            if wait is None or attempt == LLM_MAX_RETRIES:
                raise LLMError(f"{name}: {' '.join(str(error).split())[:240]}") from error
            self.sleep(wait)

    @staticmethod
    def _structured(model, schema, messages, name):
        return model.with_structured_output(schema).invoke(messages, config={"run_name": name, "tags": [name]})

    def _call_tools(self, model, messages, tools, name):
        # tool_choice="any" forces a tool call; "ask_user" is always offered, so the model
        # has a legitimate way out instead of guessing a parameter.
        config = {"run_name": name, "tags": [name]}
        if is_gemini(model):
            return self._call_gemini(model, messages, tools, config)
        if type(model).__name__ in ("ChatOpenAI", "AzureChatOpenAI"):
            return model.bind_tools(tools, tool_choice="any", parallel_tool_calls=False).invoke(messages, config=config)
        return model.bind_tools(tools, tool_choice="any").invoke(messages, config=config)

    @staticmethod
    def _call_gemini(model, messages, tools, config):
        def ask(subset, schemas):
            return model.invoke(messages, config=config, tools=[gemini_tool(subset, schemas)], tool_choice="any")

        full = [t["function"]["parameters"] for t in tools]
        sent = fit_request(full)
        try:
            ai = ask(tools, sent)
        except Exception as e:  # the limits are measured, not documented: if it is still refused, retry once at half size
            if "INVALID_ARGUMENT" not in str(e):
                raise
            sent = fit_request(full, GEMINI_REQUEST_BUDGET // 2)
            ai = ask(tools, sent)
        # If the tool it picked was one of those shortened to make room, ask again with that tool alone, so every
        # field it supports is on offer. ask_user stays available: the model must never be cornered into guessing.
        picked = ai.tool_calls[0]["name"] if ai.tool_calls else None
        i = next((n for n, t in enumerate(tools) if t["function"]["name"] == picked), None)
        if i is not None and sent[i] != fit_alone(full[i]):
            pair = [tools[i]] + [t for t in tools if t["function"]["name"] == "ask_user"]
            ai = ask(pair, [fit_alone(t["function"]["parameters"]) for t in pair])
        return ai

    @staticmethod
    def _text(model, messages, name):
        return message_text(model.invoke(messages, config={"run_name": name, "tags": [name]}).content)


class ScriptedLLM:
    """Test double: returns pre-baked outputs per call name, in order."""

    def __init__(self, script: dict[str, list[Any]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, list[BaseMessage], Any]] = []

    def _next(self, name: str, messages, extra=None):
        self.calls.append((name, messages, extra))
        if not self.script.get(name):
            raise AssertionError(f"ScriptedLLM: no scripted response left for '{name}'")
        out = self.script[name].pop(0)
        return out(messages, extra) if callable(out) else out

    def structured(self, schema, messages, *, name):
        out = self._next(name, messages)
        return out if isinstance(out, schema) else schema.model_validate(out)

    def call_tools(self, messages, tools, *, name):
        out = self._next(name, messages, [t["function"]["name"] for t in tools])
        if isinstance(out, AIMessage):
            return out
        tool, args = out  # (tool_name, args) shorthand
        return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": f"call_{len(self.calls)}"}])

    def text(self, messages, *, name):
        return str(self._next(name, messages))

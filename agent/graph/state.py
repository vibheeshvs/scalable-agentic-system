"""Graph state. Everything the agent knows about the current request lives here and is checkpointed
after every node, so a crash, a deploy or a human-confirmation pause never loses work."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Step(TypedDict, total=False):
    n: int                    # 1-based step number (referenced as "step 1" by prompts and analyze_data)
    goal: str                 # natural-language goal from the planner; also the retrieval query
    kind: str                 # api | compute | knowledge | system (from the planner): decides which tools are offered
    status: str               # pending | done | failed | cancelled | skipped | needs_input
    preset: bool              # tool fixed up front (router fast path for rag/system search)
    tool: str | None          # chosen tool (LLM function name)
    tool_id: str | None       # catalog id of the chosen tool, e.g. paypal.invoices.send
    args: dict[str, Any] | None
    candidates: list[str]     # tool ids retrieved for this step (for tracing / debugging)
    preview: str | None       # compact view of the result that later prompts see
    error: str | None         # last validation / API error, fed back to the selector
    attempts: int             # repair attempts used
    idem_key: str | None      # idempotency key, fixed before the call so retries/resumes reuse it


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # the user-visible conversation only
    run_id: str
    thread_id: str
    intent: str
    request: str              # standalone rewrite of the current request
    services: list[str]       # services in scope for retrieval
    plan: list[Step]
    cursor: int               # index of the step being worked on
    results: dict[str, Any]   # full results by step number (big payloads stay out of prompts)
    outcome: str | None       # succeeded | partial | failed | cancelled | needs_input

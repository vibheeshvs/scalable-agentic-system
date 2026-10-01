"""Wire the graph together."""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..llm import LLM, ChatLLM
from ..registry.build import build_paypal
from ..registry.models import Catalog
from ..retrieval.dense import get_embedder
from ..retrieval.index import ToolIndex
from ..runlog import RunLog
from ..tools.http import HttpExecutor, build_executor
from ..tools.rag import KnowledgeBase
from .nodes import Context, Nodes
from .state import AgentState


def build_graph(ctx: Context, checkpointer: Any = None):
    n = Nodes(ctx)
    g = StateGraph(AgentState)
    for name in ("router", "planner", "select", "validate", "guard", "execute", "advance", "ask", "respond"):
        g.add_node(name, getattr(n, name))
    g.add_edge(START, "router")
    g.add_conditional_edges("router", n.after_router, ["planner", "select", END])
    g.add_conditional_edges("planner", n.after_planner, ["select", END])
    g.add_edge("select", "validate")
    g.add_conditional_edges("validate", n.after_validate, ["select", "guard", "ask", "advance"])
    g.add_conditional_edges("guard", n.after_guard, ["execute", "respond"])
    g.add_conditional_edges("execute", n.after_execute, ["select", "advance"])
    g.add_conditional_edges("advance", n.after_advance, ["select", "respond"])
    g.add_edge("ask", END)
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


def build_agent(settings: Settings | None = None, *, llm: LLM | None = None, executor: HttpExecutor | None = None,
                embedder: Any = "auto", checkpointer: Any = None, runlog: RunLog | None = None,
                catalog: Catalog | None = None):
    """Returns (compiled_graph, context). Every dependency can be swapped (tests pass fakes)."""
    settings = settings or Settings()
    if catalog is None:
        catalog = Catalog.load(settings.catalog_path) if settings.catalog_path.exists() else build_paypal()
    emb = get_embedder() if embedder == "auto" else embedder
    ctx = Context(
        settings=settings,
        catalog=catalog,
        index=ToolIndex(catalog, emb),
        kb=KnowledgeBase(settings.knowledge_dir, emb),
        llm=llm or ChatLLM(settings.llm_model, settings.router_model, settings.fallback_model),
        executor=executor or build_executor(settings, catalog),
        runlog=runlog or RunLog(settings.data_dir / "runlog.sqlite"),
    )
    if checkpointer is None:
        from langgraph.checkpoint.sqlite import SqliteSaver

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        checkpointer = SqliteSaver(sqlite3.connect(settings.data_dir / "checkpoints.sqlite", check_same_thread=False))
    return build_graph(ctx, checkpointer), ctx

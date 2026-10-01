import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent.config import Settings  # noqa: E402
from agent.graph.build import build_agent  # noqa: E402
from agent.llm import ScriptedLLM  # noqa: E402
from agent.registry.build import build_paypal  # noqa: E402
from agent.registry.models import Catalog  # noqa: E402
from agent.runlog import RunLog  # noqa: E402
from agent.tools.http import HttpExecutor, PayPalAuth  # noqa: E402
from agent.tools.mock_paypal import MockPayPal  # noqa: E402


@pytest.fixture(scope="session")
def paypal_catalog() -> Catalog:
    path = ROOT / "data/catalog/paypal.json"
    return Catalog.load(path) if path.exists() else build_paypal()


NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_paypal():
    return MockPayPal(seed=7, now=NOW)


@pytest.fixture
def executor(mock_paypal, paypal_catalog):
    base = paypal_catalog.services["paypal"].base_url
    return HttpExecutor({"paypal": PayPalAuth("id", "secret", base)}, transport=mock_paypal.transport(),
                        sleep=lambda s: None)


@pytest.fixture
def make_agent(paypal_catalog, executor, tmp_path):
    """Build the real graph with a scripted LLM, the mock PayPal and in-memory persistence."""
    def _make(script: dict, **settings_overrides):
        settings = Settings(data_dir=tmp_path, **settings_overrides)
        llm = ScriptedLLM(script)
        graph, ctx = build_agent(settings, llm=llm, executor=executor, embedder=None, checkpointer=InMemorySaver(),
                                 runlog=RunLog(":memory:"), catalog=paypal_catalog)
        ctx.today = lambda: date(2026, 9, 30)
        return graph, ctx, llm
    return _make

"""Runtime settings, all overridable through environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "google_genai:gemini-3.8-flash"


def _env_list(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass
class Settings:
    # Default: Gemini on Google AI Studio's free tier, so anyone can run it without paying.
    # Any init_chat_model string works ("openai:...", "anthropic:..."); the router can use a
    # lighter model than the planner/selector (e.g. ROUTER_MODEL=google_genai:gemini-3.5-flash-lite).
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", DEFAULT_MODEL))
    router_model: str = field(default_factory=lambda: os.getenv("ROUTER_MODEL", os.getenv("LLM_MODEL", DEFAULT_MODEL)))
    # tried straight away when the main (or router) model is overloaded or out of quota; empty = no fallback
    fallback_model: str | None = field(default_factory=lambda: os.getenv("LLM_FALLBACK_MODEL") or None)

    catalog_path: Path = field(default_factory=lambda: Path(os.getenv("CATALOG_PATH", ROOT / "data/catalog/paypal.json")))
    knowledge_dir: Path = field(default_factory=lambda: Path(os.getenv("KNOWLEDGE_DIR", ROOT / "data/knowledge")))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("AGENT_DATA_DIR", ROOT / ".data")))  # checkpoints + run log

    # services this user/tenant has connected; empty = everything in the catalog
    enabled_services: list[str] = field(default_factory=lambda: _env_list("ENABLED_SERVICES"))

    top_k: int = field(default_factory=lambda: int(os.getenv("TOOL_TOP_K", "8")))
    max_steps: int = field(default_factory=lambda: int(os.getenv("MAX_PLAN_STEPS", "6")))
    max_repairs: int = field(default_factory=lambda: int(os.getenv("MAX_REPAIRS", "2")))
    confirm_risk: list[str] = field(default_factory=lambda: _env_list("CONFIRM_RISK", "high"))

    # "mock" runs against an in-process fake PayPal; "sandbox" uses PAYPAL_CLIENT_ID/SECRET
    paypal_mode: str = field(default_factory=lambda: os.getenv("PAYPAL_MODE", "sandbox" if os.getenv("PAYPAL_CLIENT_ID") else "mock"))
    http_timeout: float = field(default_factory=lambda: float(os.getenv("HTTP_TIMEOUT", "20")))
    http_retries: int = field(default_factory=lambda: int(os.getenv("HTTP_RETRIES", "3")))

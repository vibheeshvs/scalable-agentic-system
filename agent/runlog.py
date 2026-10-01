"""Append-only audit log of runs and tool calls (SQLite).

This is separate from the LangGraph checkpointer on purpose: checkpoints are the agent's
working memory (resumable, overwritten as the graph moves), while this is the business
record - who asked for what, which API was called with which arguments, what happened.
The system-search tool reads it to answer "what's the status of my last request?".
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, thread_id TEXT, started_at TEXT, finished_at TEXT,
    request TEXT, intent TEXT, status TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, thread_id TEXT, step INTEGER, tool_id TEXT,
    risk TEXT, args TEXT, status TEXT, http_status INTEGER, attempts INTEGER, latency_ms INTEGER,
    error TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS runs_thread ON runs(thread_id, started_at);
CREATE INDEX IF NOT EXISTS calls_run ON tool_calls(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLog:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def start_run(self, run_id: str, thread_id: str, request: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?,?,?,?)",
                              (run_id, thread_id, _now(), None, request, None, "running", None))

    def update_run(self, run_id: str, **fields: Any) -> None:
        if fields.get("status") in ("succeeded", "failed", "cancelled", "needs_input", "partial"):
            fields.setdefault("finished_at", _now())
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.lock, self.conn:
            self.conn.execute(f"UPDATE runs SET {cols} WHERE run_id=?", (*fields.values(), run_id))

    def log_call(self, run_id: str, thread_id: str, step: int, tool_id: str, risk: str, args: Any, status: str,
                 http_status: int | None = None, attempts: int = 1, latency_ms: int = 0, error: str | None = None) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO tool_calls (run_id, thread_id, step, tool_id, risk, args, status, http_status, attempts,"
                " latency_ms, error, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, thread_id, step, tool_id, risk, json.dumps(args, default=str)[:4000], status, http_status,
                 attempts, latency_ms, error, _now()))

    def recent_runs(self, thread_id: str | None, limit: int = 5, exclude_run: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM runs WHERE (? IS NULL OR thread_id = ?) AND run_id != ? ORDER BY started_at DESC, rowid DESC LIMIT ?"
        with self.lock:
            runs = [dict(r) for r in self.conn.execute(sql, (thread_id, thread_id, exclude_run or "", limit))]
            for r in runs:
                r["tool_calls"] = [dict(c) for c in self.conn.execute(
                    "SELECT step, tool_id, risk, status, http_status, attempts, error FROM tool_calls WHERE run_id=? ORDER BY id",
                    (r["run_id"],))]
        return runs

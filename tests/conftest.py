"""全量测试的本地持久化副作用隔离。"""

import pytest

from app.config import Settings
from app.observability import metrics
from app.tools import runtime as tool_runtime


@pytest.fixture(autouse=True)
def isolate_production_storage_paths(monkeypatch, tmp_path):
    """把审计、指标、会话、trace 和 checkpoint 写入 pytest 临时目录。"""
    monkeypatch.setattr(
        Settings,
        "TOOL_AUDIT_PATH",
        tmp_path / "tool_audit.jsonl",
    )
    monkeypatch.setattr(
        Settings,
        "SQLITE_DB_PATH",
        tmp_path / "app.sqlite3",
    )
    monkeypatch.setattr(
        Settings,
        "LANGGRAPH_CHECKPOINT_DB_PATH",
        tmp_path / "langgraph_checkpoints.sqlite3",
    )
    monkeypatch.setattr(
        metrics,
        "METRICS_PATH",
        tmp_path / "metrics.jsonl",
    )
    monkeypatch.setattr(
        metrics,
        "BAD_CASES_PATH",
        tmp_path / "bad_cases.jsonl",
    )
    tool_runtime.reset_tool_runtime()
    yield
    tool_runtime.reset_tool_runtime()

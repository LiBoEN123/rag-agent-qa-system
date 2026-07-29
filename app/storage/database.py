"""SQLite 数据库基础模块。

负责创建连接、初始化数据表，并提供统一的时间字段生成方法。
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from app.config import get_settings


def now_iso() -> str:
    """返回适合 SQLite 文本字段存储的本地时间。"""
    return datetime.now().isoformat(timespec="milliseconds")


def get_db_path() -> Path:
    """返回当前配置的 SQLite 数据库路径。"""
    return get_settings().sqlite_db_path


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """创建 SQLite 连接，并确保数据库所在目录存在。"""
    path = Path(db_path) if db_path is not None else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    """初始化会话历史和 Agent trace 所需的数据表。"""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                ON messages(session_id, id);

                CREATE TABLE IF NOT EXISTS agent_runs (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    rewritten_question TEXT,
                    final_answer TEXT,
                    status TEXT,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    total_cost_ms INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS agent_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    step_id INTEGER,
                    action TEXT,
                    tool_name TEXT,
                    tool_args_json TEXT,
                    observation_json TEXT,
                    status TEXT,
                    cost_ms INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_steps_trace_id_id
                ON agent_steps(trace_id, id);
                """
            )
    finally:
        conn.close()


__all__ = ["get_connection", "get_db_path", "init_db", "now_iso"]

"""Agent 追踪持久化模块。

将 Agent 运行摘要和计划步骤写入 SQLite，并提供按 trace_id 查询的接口。
"""

import json
from pathlib import Path

from app.storage.database import get_connection, init_db, now_iso


_MAX_FINAL_ANSWER_CHARS = 10_000


def _safe_int(value) -> int:
    """把可能缺失或非法的数值安全转换为整数。"""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def save_agent_run(
    *,
    trace_id: str,
    session_id: str | None,
    question: str,
    rewritten_question: str = "",
    final_answer: str = "",
    status: str = "",
    reason: str = "",
    total_cost_ms: int = 0,
    db_path: Path | None = None,
) -> None:
    """新增或替换一条 Agent v2 运行记录。"""
    normalized_trace_id = (trace_id or "").strip()
    if not normalized_trace_id:
        return

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_runs(
                    trace_id,
                    session_id,
                    question,
                    rewritten_question,
                    final_answer,
                    status,
                    reason,
                    created_at,
                    total_cost_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_trace_id,
                    session_id,
                    str(question or ""),
                    str(rewritten_question or ""),
                    str(final_answer or "")[:_MAX_FINAL_ANSWER_CHARS],
                    str(status or ""),
                    str(reason or ""),
                    now_iso(),
                    _safe_int(total_cost_ms),
                ),
            )
    finally:
        conn.close()


def save_agent_steps(
    *,
    trace_id: str,
    tool_trace: list[dict],
    db_path: Path | None = None,
) -> None:
    """替换指定 trace 的全部步骤记录。"""
    normalized_trace_id = (trace_id or "").strip()
    if not normalized_trace_id or not tool_trace:
        return

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "DELETE FROM agent_steps WHERE trace_id = ?",
                (normalized_trace_id,),
            )
            for raw_item in tool_trace:
                item = raw_item if isinstance(raw_item, dict) else {}
                conn.execute(
                    """
                    INSERT INTO agent_steps(
                        trace_id,
                        step_id,
                        action,
                        tool_name,
                        tool_args_json,
                        observation_json,
                        status,
                        cost_ms,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_trace_id,
                        item.get("step_id"),
                        str(item.get("action") or item.get("step") or ""),
                        str(item.get("tool_name") or ""),
                        json.dumps(
                            item.get("tool_args", {}),
                            ensure_ascii=False,
                            default=str,
                        ),
                        json.dumps(item, ensure_ascii=False, default=str),
                        str(item.get("status") or item.get("tool_status") or ""),
                        _safe_int(item.get("cost_ms")),
                        now_iso(),
                    ),
                )
    finally:
        conn.close()


def save_agent_trace(
    *,
    trace_id: str,
    session_id: str | None,
    question: str,
    rewritten_question: str = "",
    final_answer: str = "",
    status: str = "",
    reason: str = "",
    tool_trace: list[dict] | None = None,
    total_cost_ms: int = 0,
    db_path: Path | None = None,
) -> None:
    """保存 Agent v2 运行摘要及其步骤。"""
    save_agent_run(
        trace_id=trace_id,
        session_id=session_id,
        question=question,
        rewritten_question=rewritten_question,
        final_answer=final_answer,
        status=status,
        reason=reason,
        total_cost_ms=total_cost_ms,
        db_path=db_path,
    )
    save_agent_steps(
        trace_id=trace_id,
        tool_trace=tool_trace or [],
        db_path=db_path,
    )


def get_agent_run(
    trace_id: str,
    db_path: Path | None = None,
) -> dict | None:
    """读取一条 Agent 运行记录。"""
    normalized_trace_id = (trace_id or "").strip()
    if not normalized_trace_id:
        return None

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM agent_runs WHERE trace_id = ?",
            (normalized_trace_id,),
        ).fetchone()
    finally:
        conn.close()

    return dict(row) if row is not None else None


def get_agent_steps(
    trace_id: str,
    db_path: Path | None = None,
) -> list[dict]:
    """按写入顺序读取指定 trace 的步骤。"""
    normalized_trace_id = (trace_id or "").strip()
    if not normalized_trace_id:
        return []

    init_db(db_path)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM agent_steps
            WHERE trace_id = ?
            ORDER BY id
            """,
            (normalized_trace_id,),
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


__all__ = [
    "get_agent_run",
    "get_agent_steps",
    "save_agent_run",
    "save_agent_steps",
    "save_agent_trace",
]

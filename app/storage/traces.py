"""使用共享 SQLAlchemy Session 保存和读取 Agent 运行摘要与步骤。"""

import json
from pathlib import Path

from sqlalchemy import delete, insert, select

from app.storage.database import (
    agent_runs, agent_steps, get_session_factory, init_db, now_iso,
)


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
    """新增或替换一条 Agent 运行记录。"""
    normalized_trace_id = (trace_id or "").strip()
    if not normalized_trace_id:
        return

    init_db(db_path)
    with get_session_factory(db_path).begin() as session:
        # 保留 SQLite REPLACE 的删除再插入语义（包括 rowid）。
        session.execute(insert(agent_runs).prefix_with("OR REPLACE").values(
            trace_id=normalized_trace_id, session_id=session_id,
            question=str(question or ""), rewritten_question=str(rewritten_question or ""),
            final_answer=str(final_answer or "")[:_MAX_FINAL_ANSWER_CHARS],
            status=str(status or ""), reason=str(reason or ""),
            created_at=now_iso(), total_cost_ms=_safe_int(total_cost_ms),
        ))


def save_agent_steps(
    *,
    trace_id: str,
    tool_trace: list[dict],
    db_path: Path | None = None,
) -> None:
    """替换指定 trace 的全部步骤记录；空列表保持原有 no-op。"""
    normalized_trace_id = (trace_id or "").strip()
    if not normalized_trace_id or not tool_trace:
        return

    init_db(db_path)
    with get_session_factory(db_path).begin() as session:
        session.execute(delete(agent_steps).where(agent_steps.c.trace_id == normalized_trace_id))
        for raw_item in tool_trace:
            item = raw_item if isinstance(raw_item, dict) else {}
            session.execute(insert(agent_steps).values(
                trace_id=normalized_trace_id, step_id=item.get("step_id"),
                action=str(item.get("action") or item.get("step") or ""),
                tool_name=str(item.get("tool_name") or ""),
                tool_args_json=json.dumps(item.get("tool_args", {}), ensure_ascii=False, default=str),
                observation_json=json.dumps(item, ensure_ascii=False, default=str),
                status=str(item.get("status") or item.get("tool_status") or ""),
                cost_ms=_safe_int(item.get("cost_ms")), created_at=now_iso(),
            ))


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
    """保存 Agent 运行摘要及其步骤。"""
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
    with get_session_factory(db_path)() as session:
        row = session.execute(
            select(agent_runs).where(agent_runs.c.trace_id == normalized_trace_id)
        ).mappings().first()
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
    with get_session_factory(db_path)() as session:
        rows = session.execute(
            select(agent_steps).where(agent_steps.c.trace_id == normalized_trace_id)
            .order_by(agent_steps.c.id.asc())
        ).mappings().all()
    return [dict(row) for row in rows]


__all__ = ["get_agent_run", "get_agent_steps", "save_agent_run",
           "save_agent_steps", "save_agent_trace"]

"""SQL 查询结果的确定性转换、单元格截断和总输出限制。"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Iterable, Sequence
from uuid import UUID

from app.tools.errors import ToolErrorCode
from app.tools.sql_readonly.errors import SqlReadonlyError


_TRUNCATION_MARKER = "...[truncated]"
_COLUMN_NAME_MAX_CHARS = 256


def _truncate_text(value: str, max_chars: int) -> str:
    """把文本限制在固定长度，并保留显式截断标记。"""
    if len(value) <= max_chars:
        return value
    keep = max_chars - len(_TRUNCATION_MARKER)
    return value[:keep] + _TRUNCATION_MARKER


def serialize_sql_value(value: object, *, max_chars: int) -> object:
    """把常见数据库值转换为有限、JSON 安全的稳定标量。"""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        return _truncate_text(str(value), max_chars)
    if isinstance(value, (datetime, date, time)):
        return _truncate_text(value.isoformat(), max_chars)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        digest = hashlib.sha256(value).hexdigest()[:16]
        return _truncate_text(
            f"<binary:{len(value)} bytes sha256:{digest}>",
            max_chars,
        )
    if isinstance(value, str):
        return _truncate_text(value, max_chars)
    try:
        rendered = str(value)
    except Exception:
        rendered = f"<{type(value).__name__}>"
    return _truncate_text(rendered, max_chars)


def normalize_column_names(columns: Iterable[object]) -> list[str]:
    """生成有限、非空且唯一的列名，同时保持原始列顺序。"""
    normalized: list[str] = []
    used: set[str] = set()
    for index, value in enumerate(columns, start=1):
        raw_base = str(value).strip() or f"column_{index}"
        base = _truncate_text(raw_base, _COLUMN_NAME_MAX_CHARS)
        occurrence = 1
        candidate = base
        while candidate in used:
            occurrence += 1
            suffix = f"_{occurrence}"
            candidate = _truncate_text(
                raw_base,
                _COLUMN_NAME_MAX_CHARS - len(suffix),
            ) + suffix
        used.add(candidate)
        normalized.append(candidate)
    return normalized


def serialize_query_output(
    *,
    columns: Sequence[object],
    rows: Sequence[Sequence[object]],
    requested_max_rows: int,
    max_cell_chars: int,
    max_output_chars: int,
    dialect: str,
    execution_ms: int,
) -> dict[str, object]:
    """按完整行构建不超过总字符预算的查询输出。"""
    normalized_columns = normalize_column_names(columns)
    row_limit_truncated = len(rows) > requested_max_rows
    serialized_rows: list[list[object]] = []

    # 封装已序列化的查询行、列信息、执行耗时、截断状态和不可信标记。
    def payload(*, truncated: bool) -> dict[str, object]:
        return {
            "columns": normalized_columns,
            "rows": serialized_rows,
            "row_count": len(serialized_rows),
            "truncated": truncated,
            "dialect": dialect,
            "execution_ms": max(0, execution_ms),
            "content_is_untrusted": True,
        }

    empty_payload = payload(truncated=row_limit_truncated or bool(rows))
    if len(_json_text(empty_payload)) > max_output_chars:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_RESULT_TOO_WIDE,
            "结果元数据超过输出字符上限",
        )

    output_truncated = row_limit_truncated
    for raw_row in rows[:requested_max_rows]:
        serialized_row = [
            serialize_sql_value(value, max_chars=max_cell_chars)
            for value in raw_row
        ]
        serialized_rows.append(serialized_row)
        candidate = payload(truncated=output_truncated)
        if len(_json_text(candidate)) > max_output_chars:
            serialized_rows.pop()
            output_truncated = True
            break
    return payload(truncated=output_truncated)


def ensure_sql_output_size(value: object, *, max_output_chars: int) -> None:
    """拒绝无法按完整结构安全返回的非行式 SQL 工具输出。"""
    if len(_json_text(value)) > max_output_chars:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_RESULT_TOO_WIDE,
            "结构化结果超过输出字符上限",
        )


def _json_text(value: object) -> str:
    """使用与 ToolMessage 接近的紧凑 JSON 计算字符预算。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


__all__ = [
    "normalize_column_names",
    "ensure_sql_output_size",
    "serialize_query_output",
    "serialize_sql_value",
]

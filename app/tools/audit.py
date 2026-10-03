"""结构化工具审计事件、输出摘要和可注入写入器。"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from app.config import AppSettings, get_settings
from app.common.numeric import safe_float
from app.tools.contracts import ToolAuditOutputMode, ToolSpec
from app.common.redaction import (
    DEFAULT_SENSITIVE_KEYS,
    redact_sensitive_data,
)


class ToolAuditEventType(str, Enum):
    """一次工具调用生命周期内允许记录的审计事件。"""

    REQUESTED = "requested"
    DENIED = "denied"
    STARTED = "started"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class ToolAuditEvent:
    """不包含原始敏感载荷的结构化审计事件。"""

    timestamp: str
    event_type: ToolAuditEventType

    tool_call_id: str
    trace_id: str
    session_id: str

    principal_id: str
    principal_type: str
    source: str

    tool_name: str
    tool_version: str
    risk_level: str

    policy_decision: str
    args_fingerprint: str

    attempt_count: int
    elapsed_ms: int

    success: bool
    error_code: str
    error_message: str

    sanitized_input: Any
    sanitized_output_summary: Any
    server_alias: str = ""

    def to_dict(self) -> dict[str, Any]:
        """返回可直接写入 JSON 的事件字典。"""
        value = asdict(self)
        value["event_type"] = self.event_type.value
        if not value["server_alias"]:
            value.pop("server_alias")
        return value


class ToolAuditSink(Protocol):
    """审计事件写入器边界。"""

    def write(self, event: ToolAuditEvent) -> None:
        """持久化一条审计事件。"""
        ...


class DisabledToolAuditSink:
    """配置关闭审计时报告不可用，保留只读调用的 fail-open 行为。"""

    def write(self, event: ToolAuditEvent) -> None:
        """显式报告审计不可用，由调用方记录写入失败。"""
        del event
        raise RuntimeError("tool audit is disabled")


class JsonlToolAuditSink:
    """按需打开 UTF-8 文件、每行写入一个 JSON 审计事件。"""

    # 保存审计文件路径，拒绝目录路径并初始化线程写入锁。
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise ValueError("工具审计路径必须指向文件，不能是目录")
        self._lock = threading.Lock()

    def write(self, event: ToolAuditEvent) -> None:
        """线程安全地追加单条事件；不会在模块导入时打开文件。"""
        record = event.to_dict()
        serialized = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(serialized + "\n")


def get_default_tool_audit_sink(
    settings: AppSettings | None = None,
) -> ToolAuditSink:
    """根据当前配置惰性创建默认审计写入器。"""
    resolved = settings or get_settings()
    if not resolved.tool_audit_enabled:
        return DisabledToolAuditSink()
    return JsonlToolAuditSink(resolved.tool_audit_path)


def utc_timestamp() -> str:
    """返回带 UTC 时区的 ISO 8601 时间。"""
    return datetime.now(UTC).isoformat()


def audit_redaction_limits(
    settings: AppSettings | None = None,
) -> dict[str, int]:
    """返回当前审计脱敏和体量限制。"""
    resolved = settings or get_settings()
    return {
        "max_depth": resolved.tool_audit_max_depth,
        "max_items": resolved.tool_audit_max_items,
        "max_string_chars": resolved.tool_audit_max_string_chars,
        "max_serialized_chars": resolved.tool_audit_max_serialized_chars,
    }


def sanitize_tool_input(
    spec: ToolSpec,
    value: Any,
    *,
    settings: AppSettings | None = None,
) -> Any:
    """按全局和工具契约敏感字段脱敏工具输入。"""
    return redact_sensitive_data(
        value,
        sensitive_keys=(
            *DEFAULT_SENSITIVE_KEYS,
            *spec.sensitive_arg_fields,
        ),
        **audit_redaction_limits(settings),
    )


def retrieval_evidence_metadata(output: dict[str, Any]) -> list[dict[str, Any]]:
    """只读取已序列化 evidence 的前十项 metadata，供审计与公开 trace 派生。"""
    evidence = output.get("evidence")
    if not isinstance(evidence, list):
        return []
    return [
        item["metadata"] if isinstance(item.get("metadata"), dict) else {}
        for item in evidence[:10] if isinstance(item, dict)
    ]


def summarize_tool_output(
    spec: ToolSpec,
    value: Any,
    *,
    settings: AppSettings | None = None,
) -> Any:
    """按契约模式生成不会默认包含大型业务正文的输出摘要。"""
    if spec.audit_output_mode is ToolAuditOutputMode.NONE:
        return {
            "mode": "none",
            "type": type(value).__name__,
        }

    if not isinstance(value, dict):
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "count": len(value)}
        return {"type": type(value).__name__}

    if spec.name == "search_knowledge_base":
        metadata = retrieval_evidence_metadata(value)
        summary = {
            "count": value.get("count", 0),
            "top_chunk_ids": [item.get("chunk_id") for item in metadata],
            "top_scores": [round(safe_float(item.get("hybrid_score")), 4) for item in metadata],
        }
        return redact_sensitive_data(
            summary,

            **audit_redaction_limits(settings),
        )
    if spec.name == "web_fetch":
        summary = {
            "status_code": value.get("status_code", 0),
            "content_type": value.get("content_type", ""),
            "content_length": value.get("content_length", 0),
            "truncated": bool(value.get("truncated", False)),
            "content_is_untrusted": bool(
                value.get("content_is_untrusted", False)
            ),
        }
        return redact_sensitive_data(
            summary,

            **audit_redaction_limits(settings),
        )
    if spec.name == "web_pdf_fetch":
        summary = {
            "status_code": value.get("status_code", 0),
            "content_type": value.get("content_type", ""),
            "total_pages": value.get("total_pages", 0),
            "start_page": value.get("start_page", 0),
            "pages_read": value.get("pages_read", 0),
            "content_length": value.get("content_length", 0),
            "truncated": bool(value.get("truncated", False)),
            "content_is_untrusted": bool(
                value.get("content_is_untrusted", False)
            ),
        }
        return redact_sensitive_data(
            summary,

            **audit_redaction_limits(settings),
        )

    summary: dict[str, Any] = {
        "type": "dict",
        "fields": list(value.keys())[
            : (settings or get_settings()).tool_audit_max_items
        ],
    }
    for key in ("count", "conflict_detected", "exists"):
        if key in value and isinstance(value[key], (bool, int, float, str)):
            summary[key] = value[key]
    return redact_sensitive_data(
        summary,

        **audit_redaction_limits(settings),
    )


__all__ = [
    "JsonlToolAuditSink",
    "DisabledToolAuditSink",
    "ToolAuditEvent",
    "ToolAuditEventType",
    "ToolAuditSink",
    "audit_redaction_limits",
    "get_default_tool_audit_sink",
    "sanitize_tool_input",
    "summarize_tool_output",
    "utc_timestamp",
]

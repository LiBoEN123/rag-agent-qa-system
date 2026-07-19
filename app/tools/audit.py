"""结构化工具审计事件、输出摘要和可注入写入器。"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings, validate_tool_governance_settings
from app.tools.contracts import ToolAuditOutputMode, ToolSpec
from app.tools.redaction import (
    DEFAULT_SENSITIVE_KEYS,
    redact_sensitive_data,
)


class ToolAuditEventType(str, Enum):
    """一次工具调用生命周期内允许记录的审计事件。"""

    REQUESTED = "requested"
    DENIED = "denied"
    APPROVAL_REQUIRED = "approval_required"
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
    request_id: str
    session_id: str

    principal_id: str
    principal_type: str
    source: str
    tenant_id: str

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

    def to_dict(self) -> dict[str, Any]:
        """返回可直接写入 JSON 的事件字典。"""
        value = asdict(self)
        value["event_type"] = self.event_type.value
        return value


class ToolAuditSink(Protocol):
    """审计事件写入器边界。"""

    def write(self, event: ToolAuditEvent) -> None:
        """持久化一条审计事件。"""
        ...


class NullToolAuditSink:
    """显式关闭审计时使用的无副作用写入器。"""

    def write(self, event: ToolAuditEvent) -> None:
        """接收事件但不持久化。"""
        del event


class DisabledToolAuditSink:
    """配置关闭审计时保持只读 fail-open、写工具 fail-closed。"""

    def write(self, event: ToolAuditEvent) -> None:
        """显式报告审计不可用，由 Runtime 按风险级别处理。"""
        del event
        raise RuntimeError("tool audit is disabled")


class JsonlToolAuditSink:
    """按需打开 UTF-8 文件、每行写入一个 JSON 审计事件。"""

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


def get_default_tool_audit_sink() -> ToolAuditSink:
    """根据当前配置惰性创建默认审计写入器。"""
    validate_tool_governance_settings(Settings)
    if not Settings.TOOL_AUDIT_ENABLED:
        return DisabledToolAuditSink()
    return JsonlToolAuditSink(Settings.TOOL_AUDIT_PATH)


def utc_timestamp() -> str:
    """返回带 UTC 时区的 ISO 8601 时间。"""
    return datetime.now(UTC).isoformat()


def audit_redaction_limits() -> dict[str, int]:
    """返回当前审计脱敏和体量限制。"""
    return {
        "max_depth": Settings.TOOL_AUDIT_MAX_DEPTH,
        "max_items": Settings.TOOL_AUDIT_MAX_ITEMS,
        "max_string_chars": Settings.TOOL_AUDIT_MAX_STRING_CHARS,
        "max_serialized_chars": Settings.TOOL_AUDIT_MAX_SERIALIZED_CHARS,
    }


def sanitize_tool_input(spec: ToolSpec, value: Any) -> Any:
    """按全局和工具契约敏感字段脱敏工具输入。"""
    return redact_sensitive_data(
        value,
        sensitive_keys=(
            *DEFAULT_SENSITIVE_KEYS,
            *spec.sensitive_arg_fields,
        ),
        **audit_redaction_limits(),
    )


def _limited_sequence(value: Any, *, limit: int = 10) -> list[Any]:
    """返回只含标量的有限序列摘要。"""
    if not isinstance(value, (list, tuple)):
        return []
    return [
        item if item is None or isinstance(item, (str, int, float, bool))
        else f"<{type(item).__name__}>"
        for item in value[:limit]
    ]


def summarize_tool_output(spec: ToolSpec, value: Any) -> Any:
    """按契约模式生成不会默认包含大型业务正文的输出摘要。"""
    if spec.audit_output_mode is ToolAuditOutputMode.NONE:
        return {
            "mode": "none",
            "type": type(value).__name__,
        }

    if spec.audit_output_mode is ToolAuditOutputMode.SANITIZED:
        return redact_sensitive_data(
            value,
            sensitive_keys=(
                *DEFAULT_SENSITIVE_KEYS,
                *spec.sensitive_output_fields,
            ),
            **audit_redaction_limits(),
        )

    if not isinstance(value, dict):
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "count": len(value)}
        return {"type": type(value).__name__}

    if spec.name == "search_knowledge_base":
        summary = {
            "count": value.get("count", 0),
            "top_chunk_ids": _limited_sequence(
                value.get("top_chunk_ids"),
            ),
            "top_scores": _limited_sequence(value.get("top_scores")),
        }
        return redact_sensitive_data(
            summary,
            sensitive_keys=spec.sensitive_output_fields,
            **audit_redaction_limits(),
        )
    if spec.name == "compare_documents":
        evidence_a = value.get("doc_a_evidence")
        evidence_b = value.get("doc_b_evidence")
        summary = {
            "doc_a": value.get("doc_a", ""),
            "doc_b": value.get("doc_b", ""),
            "doc_a_evidence_count": (
                len(evidence_a) if isinstance(evidence_a, list) else 0
            ),
            "doc_b_evidence_count": (
                len(evidence_b) if isinstance(evidence_b, list) else 0
            ),
            "conflict_detected": bool(value.get("conflict_detected", False)),
        }
        return redact_sensitive_data(
            summary,
            sensitive_keys=spec.sensitive_output_fields,
            **audit_redaction_limits(),
        )
    if spec.name == "list_documents":
        documents = value.get("documents")
        summary = {
            "count": value.get(
                "count",
                len(documents) if isinstance(documents, list) else 0,
            )
        }
        return redact_sensitive_data(
            summary,
            sensitive_keys=spec.sensitive_output_fields,
            **audit_redaction_limits(),
        )

    summary: dict[str, Any] = {
        "type": "dict",
        "fields": list(value.keys())[: Settings.TOOL_AUDIT_MAX_ITEMS],
    }
    for key in ("count", "conflict_detected", "exists"):
        if key in value and isinstance(value[key], (bool, int, float, str)):
            summary[key] = value[key]
    return redact_sensitive_data(
        summary,
        sensitive_keys=spec.sensitive_output_fields,
        **audit_redaction_limits(),
    )


__all__ = [
    "JsonlToolAuditSink",
    "DisabledToolAuditSink",
    "NullToolAuditSink",
    "ToolAuditEvent",
    "ToolAuditEventType",
    "ToolAuditSink",
    "audit_redaction_limits",
    "get_default_tool_audit_sink",
    "sanitize_tool_input",
    "summarize_tool_output",
    "utc_timestamp",
]

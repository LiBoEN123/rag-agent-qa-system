"""工具调用期间使用的不可变请求上下文。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from app.config import AppSettings, get_settings
from app.contracts.chat import ToolMode

_PERMISSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$"
)


class PrincipalType(str, Enum):
    """发起工具调用的主体类别。"""

    ANONYMOUS = "anonymous"
    USER = "user"
    SERVICE = "service"
    SYSTEM = "system"


def _normalize_sequence(
    value: object,
    *,
    field_name: str,
    pattern: re.Pattern[str],
    format_hint: str,
) -> tuple[str, ...]:
    """规范化字符串序列，去重并保持首次出现顺序。"""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} 必须是字符串序列，不能是单个字符串")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} 中的每一项都必须是字符串")
        cleaned = item.strip()
        if not cleaned:
            raise ValueError(f"{field_name} 不允许包含空值")
        if not pattern.fullmatch(cleaned):
            raise ValueError(f"{field_name} 中的值必须匹配 {format_hint}")
        if cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return tuple(normalized)


def normalize_permissions(value: object) -> tuple[str, ...]:
    """规范化精确权限序列，本阶段不支持通配权限。"""
    return _normalize_sequence(
        value,
        field_name="permissions",
        pattern=_PERMISSION_PATTERN,
        format_hint="resource:action",
    )


def has_permission(
    granted_permissions: Sequence[str],
    required_permission: str,
) -> bool:
    """执行区分大小写的精确权限匹配。"""
    return required_permission in granted_permissions


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """承载不暴露到工具参数 Schema 的请求级治理元数据。"""

    trace_id: str = ""
    session_id: str = ""

    principal_id: str = ""
    principal_type: PrincipalType = PrincipalType.SERVICE
    permissions: tuple[str, ...] = ()

    source: str = ""

    # 保留请求问题上下文；工具的检索参数通过本轮 StructuredTool args 传入。
    effective_question: str = ""
    tool_mode: ToolMode = ToolMode.FULL

    def __post_init__(self) -> None:
        """清理请求字段并冻结集合型数据。"""
        for field_name in (
            "trace_id",
            "session_id",
            "principal_id",
            "source",
            "effective_question",
        ):
            raw_value = getattr(self, field_name)
            if not isinstance(raw_value, str):
                raise ValueError(f"{field_name} 必须是字符串")
            object.__setattr__(self, field_name, raw_value.strip())

        raw_principal_type = self.principal_type
        if isinstance(raw_principal_type, str):
            raw_principal_type = raw_principal_type.strip()
        try:
            principal_type = PrincipalType(raw_principal_type)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "principal_type 必须是 PrincipalType 或合法字符串"
            ) from exc
        object.__setattr__(self, "principal_type", principal_type)
        object.__setattr__(self, "tool_mode", ToolMode(self.tool_mode))
        object.__setattr__(
            self,
            "permissions",
            normalize_permissions(self.permissions),
        )


def build_default_tool_context(
    *,
    trace_id: str = "",
    session_id: str = "",
    source: str = "direct_dispatch",
    effective_question: str = "",
    tool_mode: ToolMode = ToolMode.FULL,
    settings: AppSettings | None = None,
) -> ToolInvocationContext:
    """构造无认证架构下权限最小化的服务级只读主体。"""
    resolved = settings or get_settings()
    return ToolInvocationContext(
        trace_id=trace_id,
        session_id=session_id,
        principal_id=resolved.tool_default_principal_id,
        principal_type=PrincipalType.SERVICE,
        permissions=resolved.tool_default_permissions,
        source=source,
        effective_question=effective_question,
        tool_mode=tool_mode,
    )


__all__ = [
    "PrincipalType",
    "ToolInvocationContext",
    "build_default_tool_context",
    "has_permission",
    "normalize_permissions",
]

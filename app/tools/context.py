"""工具调用期间使用的不可变请求上下文。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


_ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PERMISSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$"
)
_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "authorization",
        "password",
        "access_token",
        "refresh_token",
        "api_key",
        "cookie",
        "private_key",
    }
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


def normalize_roles(value: object) -> tuple[str, ...]:
    """规范化角色序列；角色本身不会自动推导权限。"""
    return _normalize_sequence(
        value,
        field_name="roles",
        pattern=_ROLE_PATTERN,
        format_hint="^[a-z][a-z0-9_]*$",
    )


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


def _normalized_key(value: object) -> str:
    """规整用于敏感键检测的键名。"""
    return re.sub(r"[-\s]+", "_", str(value).strip().lower())


def _reject_sensitive_metadata(value: object, *, path: str = "metadata") -> None:
    """递归拒绝 metadata 中明显用于承载凭证的键。"""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_key(key)
            if normalized_key in _SENSITIVE_METADATA_KEYS:
                raise ValueError(
                    f"metadata 不允许包含敏感键: {path}.{key}"
                )
            _reject_sensitive_metadata(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive_metadata(item, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """承载不暴露到工具参数 Schema 的请求级治理元数据。"""

    trace_id: str = ""
    request_id: str = ""
    session_id: str = ""

    principal_id: str = ""
    principal_type: PrincipalType = PrincipalType.SERVICE

    roles: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()

    tenant_id: str = ""
    source: str = ""

    idempotency_key: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """清理请求字段并冻结集合型数据。"""
        for field_name in (
            "trace_id",
            "request_id",
            "session_id",
            "principal_id",
            "tenant_id",
            "source",
            "idempotency_key",
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
        object.__setattr__(self, "roles", normalize_roles(self.roles))
        object.__setattr__(
            self,
            "permissions",
            normalize_permissions(self.permissions),
        )

        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata 必须是 Mapping")
        metadata_copy = dict(self.metadata)
        _reject_sensitive_metadata(metadata_copy)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_copy),
        )


def build_default_tool_context(
    *,
    trace_id: str = "",
    session_id: str = "",
    source: str = "direct_dispatch",
) -> ToolInvocationContext:
    """构造无认证架构下权限最小化的服务级只读主体。"""
    from app.config import Settings

    permissions = Settings.TOOL_DEFAULT_PERMISSIONS
    if isinstance(permissions, str):
        permissions = tuple(
            item.strip() for item in permissions.split(",") if item.strip()
        )
    return ToolInvocationContext(
        trace_id=trace_id,
        request_id=trace_id,
        session_id=session_id,
        principal_id=Settings.TOOL_DEFAULT_PRINCIPAL_ID,
        principal_type=PrincipalType.SERVICE,
        roles=("reader",),
        permissions=tuple(permissions),
        source=source,
    )


__all__ = [
    "PrincipalType",
    "ToolInvocationContext",
    "build_default_tool_context",
    "has_permission",
    "normalize_permissions",
    "normalize_roles",
]

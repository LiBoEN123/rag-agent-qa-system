"""工具契约定义。

集中定义工具元数据、风险等级、创建期校验和安全序列化能力。
"""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from app.tools.context import ToolInvocationContext


ToolHandler = Callable[
    [dict[str, Any], ToolInvocationContext],
    dict[str, Any],
]

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PERMISSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$"
)
_SEMVER_IDENTIFIER = r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER_PATTERN = re.compile(
    rf"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    rf"(?:-{_SEMVER_IDENTIFIER}(?:\.{_SEMVER_IDENTIFIER})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class ToolRiskLevel(str, Enum):
    """工具可能产生的副作用风险等级。"""

    READ_ONLY = "read_only"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


class ToolAuditOutputMode(str, Enum):
    """工具输出允许进入审计事件的安全模式。"""

    NONE = "none"
    SUMMARY = "summary"
    SANITIZED = "sanitized"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """描述工具的完整、可校验契约。"""

    name: str
    version: str
    description: str

    args_schema: dict[str, Any]
    output_schema: dict[str, Any]
    handler: ToolHandler
    args_model: type[BaseModel]

    output_model: type[BaseModel] | None = None

    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY
    required_permissions: tuple[str, ...] = ()

    timeout: float = 10.0
    max_retries: int = 0

    idempotent: bool = True
    requires_approval: bool = False

    sensitive_arg_fields: tuple[str, ...] = ()
    sensitive_output_fields: tuple[str, ...] = ()
    audit_output_mode: ToolAuditOutputMode = ToolAuditOutputMode.SUMMARY

    def __post_init__(self) -> None:
        """规范化契约字段，并在注册前拒绝非法配置。"""
        raw_name = self.name
        if not isinstance(raw_name, str):
            self._raise_field_error("name", "必须是字符串", raw_name)
        normalized_name = raw_name.strip()
        if not _NAME_PATTERN.fullmatch(normalized_name):
            self._raise_field_error(
                "name",
                "必须匹配 ^[a-z][a-z0-9_]*$",
                raw_name,
            )
        object.__setattr__(self, "name", normalized_name)

        if not isinstance(self.description, str):
            self._raise_field_error("description", "必须是字符串")
        normalized_description = self.description.strip()
        if not normalized_description:
            self._raise_field_error("description", "去除首尾空格后不能为空")
        object.__setattr__(self, "description", normalized_description)

        if not isinstance(self.version, str):
            self._raise_field_error("version", "必须是字符串")
        normalized_version = self.version.strip()
        if not _SEMVER_PATTERN.fullmatch(normalized_version):
            self._raise_field_error("version", "必须是合法的基础 SemVer")
        object.__setattr__(self, "version", normalized_version)

        if not isinstance(self.args_schema, dict):
            self._raise_field_error("args_schema", "必须是 dict")
        if not isinstance(self.output_schema, dict):
            self._raise_field_error("output_schema", "必须是 dict")
        if not callable(self.handler):
            self._raise_field_error("handler", "必须可调用")
        if not self._is_base_model_subclass(self.args_model):
            self._raise_field_error("args_model", "必须是 BaseModel 子类")
        if (
            self.output_model is not None
            and not self._is_base_model_subclass(self.output_model)
        ):
            self._raise_field_error(
                "output_model",
                "非空时必须是 BaseModel 子类",
            )

        try:
            normalized_risk_level = ToolRiskLevel(self.risk_level)
        except (TypeError, ValueError) as exc:
            self._raise_field_error(
                "risk_level",
                "必须是 ToolRiskLevel 或合法风险等级字符串",
                cause=exc,
            )
        object.__setattr__(self, "risk_level", normalized_risk_level)

        normalized_permissions = self._normalize_permissions(
            self.required_permissions
        )
        object.__setattr__(
            self,
            "required_permissions",
            normalized_permissions,
        )

        try:
            normalized_timeout = float(self.timeout)
        except (TypeError, ValueError, OverflowError) as exc:
            self._raise_field_error(
                "timeout",
                "必须可转换为有限正浮点数",
                cause=exc,
            )
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
            self._raise_field_error("timeout", "必须是有限正数")
        object.__setattr__(self, "timeout", normalized_timeout)

        if isinstance(self.max_retries, bool) or not isinstance(
            self.max_retries, int
        ):
            self._raise_field_error(
                "max_retries",
                "必须是非负整数，且不能是 bool",
            )
        if self.max_retries < 0:
            self._raise_field_error("max_retries", "必须大于或等于 0")

        if not isinstance(self.idempotent, bool):
            self._raise_field_error("idempotent", "必须是 bool")
        if not isinstance(self.requires_approval, bool):
            self._raise_field_error("requires_approval", "必须是 bool")

        object.__setattr__(
            self,
            "sensitive_arg_fields",
            self._normalize_sensitive_fields(
                "sensitive_arg_fields",
                self.sensitive_arg_fields,
            ),
        )
        object.__setattr__(
            self,
            "sensitive_output_fields",
            self._normalize_sensitive_fields(
                "sensitive_output_fields",
                self.sensitive_output_fields,
            ),
        )
        try:
            audit_output_mode = ToolAuditOutputMode(self.audit_output_mode)
        except (TypeError, ValueError) as exc:
            self._raise_field_error(
                "audit_output_mode",
                "必须是 NONE、SUMMARY 或 SANITIZED",
                cause=exc,
            )
        object.__setattr__(self, "audit_output_mode", audit_output_mode)

        if not self.idempotent and self.max_retries > 0:
            self._raise_field_error(
                "max_retries",
                "非幂等工具不能声明自动重试",
            )
        if (
            self.risk_level
            in {ToolRiskLevel.EXTERNAL_WRITE, ToolRiskLevel.DESTRUCTIVE}
            and not self.requires_approval
        ):
            self._raise_field_error(
                "requires_approval",
                f"风险等级 {self.risk_level.value!r} 必须要求人工审批",
            )

    def to_contract_dict(self) -> dict[str, Any]:
        """返回不含运行时对象且可 JSON 序列化的完整契约。"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "args_schema": self.args_schema,
            "output_schema": self.output_schema,
            "args_model": self.args_model.__name__,
            "output_model": (
                self.output_model.__name__
                if self.output_model is not None
                else None
            ),
            "risk_level": self.risk_level.value,
            "required_permissions": list(self.required_permissions),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "idempotent": self.idempotent,
            "requires_approval": self.requires_approval,
            "sensitive_arg_fields": list(self.sensitive_arg_fields),
            "sensitive_output_fields": list(self.sensitive_output_fields),
            "audit_output_mode": self.audit_output_mode.value,
        }

    @staticmethod
    def _is_base_model_subclass(value: object) -> bool:
        """判断值是否为 Pydantic BaseModel 子类。"""
        return isinstance(value, type) and issubclass(value, BaseModel)

    def _normalize_permissions(self, value: object) -> tuple[str, ...]:
        """规范化权限序列，去重并保持首次出现顺序。"""
        if isinstance(value, str) or not isinstance(value, Sequence):
            self._raise_field_error(
                "required_permissions",
                "必须是字符串序列，不能是单个字符串",
            )

        normalized_permissions: list[str] = []
        seen: set[str] = set()
        for permission in value:
            if not isinstance(permission, str):
                self._raise_field_error(
                    "required_permissions",
                    "每个权限都必须是字符串",
                )
            normalized_permission = permission.strip()
            if not _PERMISSION_PATTERN.fullmatch(normalized_permission):
                self._raise_field_error(
                    "required_permissions",
                    "权限必须匹配 resource:action 格式",
                )
            if normalized_permission not in seen:
                seen.add(normalized_permission)
                normalized_permissions.append(normalized_permission)
        return tuple(normalized_permissions)

    def _normalize_sensitive_fields(
        self,
        field_name: str,
        value: object,
    ) -> tuple[str, ...]:
        """规范化额外敏感字段名，禁止单个字符串和空字段。"""
        if isinstance(value, str) or not isinstance(value, Sequence):
            self._raise_field_error(
                field_name,
                "必须是字符串序列，不能是单个字符串",
            )
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                self._raise_field_error(field_name, "每个字段名都必须是字符串")
            cleaned = item.strip()
            if not cleaned:
                self._raise_field_error(field_name, "不允许包含空字段名")
            if cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
        return tuple(normalized)

    def _raise_field_error(
        self,
        field_name: str,
        reason: str,
        tool_name: object | None = None,
        *,
        cause: Exception | None = None,
    ) -> None:
        """抛出包含工具名、字段名和具体原因的契约错误。"""
        display_name = self.name if tool_name is None else tool_name
        error = ValueError(
            f"tool {display_name!r} 字段 {field_name!r} 非法: {reason}"
        )
        if cause is None:
            raise error
        raise error from cause


__all__ = [
    "ToolAuditOutputMode",
    "ToolHandler",
    "ToolRiskLevel",
    "ToolSpec",
]

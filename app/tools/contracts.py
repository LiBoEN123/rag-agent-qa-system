"""由 Pydantic 校验的工具声明与项目治理契约。"""

import json
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from enum import Enum
from typing import Annotated, Any, Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
    BeforeValidator,
    TypeAdapter,
    field_validator,
    model_validator,
)
from app.tools.context import ToolInvocationContext

ToolHandler = Callable[[dict[str, Any], ToolInvocationContext], dict[str, Any]]
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PERMISSION_PATTERN = re.compile(r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$")
_SEMVER_IDENTIFIER = r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER_PATTERN = re.compile(
    rf"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    rf"(?:-{_SEMVER_IDENTIFIER}(?:\.{_SEMVER_IDENTIFIER})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class ToolRiskLevel(str, Enum):
    READ_ONLY = "read_only"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


class ToolAuditOutputMode(str, Enum):
    NONE = "none"
    SUMMARY = "summary"
    SANITIZED = "sanitized"


TrimmedString = Annotated[
    StrictStr, StringConstraints(strip_whitespace=True, min_length=1)
]
Permission = Annotated[TrimmedString, StringConstraints(pattern=_PERMISSION_PATTERN)]
_strict_string = BeforeValidator(TypeAdapter(StrictStr).validate_python)


def _legacy_timeout(value: Any) -> float:
    """保留 float() 支持的下划线字符串等输入；范围由 Pydantic 校验。"""
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timeout 必须可转换为浮点数") from exc


class ToolSpec(BaseModel):
    """本地工具的冻结声明；通用字段交给 StructuredTool，治理字段由项目执行。"""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    name: Annotated[TrimmedString, StringConstraints(pattern=_NAME_PATTERN)]
    version: Annotated[TrimmedString, StringConstraints(pattern=_SEMVER_PATTERN)]
    description: TrimmedString
    handler: ToolHandler
    args_model: type[BaseModel]
    output_schema: Annotated[dict[Any, Any], Field(strict=True)] | None = None
    output_model: type[BaseModel] | None = None
    risk_level: Annotated[ToolRiskLevel, _strict_string] = ToolRiskLevel.READ_ONLY
    required_permissions: Sequence[Permission] = ()
    timeout: Annotated[float, BeforeValidator(_legacy_timeout)] = Field(
        default=10.0, gt=0, allow_inf_nan=False
    )
    max_retries: StrictInt = Field(default=0, ge=0)
    idempotent: StrictBool = True
    requires_approval: StrictBool = False
    sensitive_arg_fields: Sequence[TrimmedString] = ()
    sensitive_output_fields: Sequence[TrimmedString] = ()
    audit_output_mode: Annotated[ToolAuditOutputMode, _strict_string] = (
        ToolAuditOutputMode.SUMMARY
    )

    @field_validator(
        "required_permissions", "sensitive_arg_fields", "sensitive_output_fields"
    )
    @classmethod
    def normalize_sequences(cls, value: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value))

    @field_validator("output_schema")
    @classmethod
    def copy_schema(cls, value: dict[Any, Any] | None) -> dict[Any, Any] | None:
        if value is not None:
            try:
                json.dumps(value, allow_nan=False, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("必须可稳定序列化为 JSON") from exc
            return deepcopy(value)
        return value

    @model_validator(mode="after")
    def validate_governance(self) -> Self:
        if self.output_model is None and self.output_schema is None:
            raise ValueError("缺少 output_model 时必须显式声明 output_schema")
        if self.output_model is not None and self.output_schema is not None:
            raise ValueError("output_schema 不能与 output_model 同时声明")
        if not self.idempotent and self.max_retries > 0:
            raise ValueError("非幂等工具不能声明自动重试")
        if (
            self.risk_level in {ToolRiskLevel.EXTERNAL_WRITE, ToolRiskLevel.DESTRUCTIVE}
            and not self.requires_approval
        ):
            raise ValueError("必须要求人工审批")
        return self

    @property
    def args_schema(self) -> dict[str, Any]:
        """从唯一权威 args_model 派生参数 JSON Schema。"""
        return self.args_model.model_json_schema()

    def resolved_output_schema(self) -> dict[str, Any]:
        """优先从 output_model 派生，否则返回显式输出契约副本。"""
        if self.output_model is not None:
            return self.output_model.model_json_schema()
        assert self.output_schema is not None
        return deepcopy(self.output_schema)

    def to_contract_dict(self) -> dict[str, Any]:
        """返回不含运行时对象且可 JSON 序列化的完整契约。"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "args_schema": self.args_schema,
            "output_schema": self.resolved_output_schema(),
            "args_model": self.args_model.__name__,
            "output_model": (
                self.output_model.__name__ if self.output_model is not None else None
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


__all__ = [
    "ToolAuditOutputMode",
    "ToolHandler",
    "ToolRiskLevel",
    "ToolSpec",
]

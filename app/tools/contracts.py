"""由 Pydantic 校验的工具声明与项目治理契约。"""

import re
from collections.abc import Callable, Sequence
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


# 定义工具只读、内部写入、外部写入和破坏性操作的风险级别。
class ToolRiskLevel(str, Enum):
    READ_ONLY = "read_only"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


# 定义审计输出不记录、仅摘要和脱敏内容三种记录方式。
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
    """本地工具的冻结构建声明；框架导出契约，项目执行治理及输出校验。"""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    name: Annotated[TrimmedString, StringConstraints(pattern=_NAME_PATTERN)]
    version: Annotated[TrimmedString, StringConstraints(pattern=_SEMVER_PATTERN)]
    description: TrimmedString
    handler: ToolHandler
    args_model: type[BaseModel]
    output_model: type[BaseModel]
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

    # 将字符串序列去重并转为元组，保留首次出现的顺序。
    @field_validator(
        "required_permissions", "sensitive_arg_fields", "sensitive_output_fields"
    )
    @classmethod
    def normalize_sequences(cls, value: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value))

    # 校验幂等重试限制及高风险工具的人工审批要求。
    @model_validator(mode="after")
    def validate_governance(self) -> Self:
        if not self.idempotent and self.max_retries > 0:
            raise ValueError("非幂等工具不能声明自动重试")
        if (
            self.risk_level in {ToolRiskLevel.EXTERNAL_WRITE, ToolRiskLevel.DESTRUCTIVE}
            and not self.requires_approval
        ):
            raise ValueError("必须要求人工审批")
        return self


__all__ = [
    "ToolAuditOutputMode",
    "ToolHandler",
    "ToolRiskLevel",
    "ToolSpec",
]

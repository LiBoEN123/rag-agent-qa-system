"""由 Pydantic 校验的工具声明与项目治理契约。"""

import re
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Annotated, Any
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    StringConstraints,
    BeforeValidator,
    TypeAdapter,
    field_validator,
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


# MCP 来源仍需区分只读与需要审批的外部写入。
class ToolRiskLevel(str, Enum):
    READ_ONLY = "read_only"
    EXTERNAL_WRITE = "external_write"


# 本地只读工具仅记录有限摘要或不记录输出。
class ToolAuditOutputMode(str, Enum):
    NONE = "none"
    SUMMARY = "summary"


TrimmedString = Annotated[
    StrictStr, StringConstraints(strip_whitespace=True, min_length=1)
]
Permission = Annotated[TrimmedString, StringConstraints(pattern=_PERMISSION_PATTERN)]
_strict_string = BeforeValidator(TypeAdapter(StrictStr).validate_python)


class ToolSpec(BaseModel):
    """本地工具的冻结构建声明；框架导出契约，项目执行治理及输出校验。"""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    name: Annotated[TrimmedString, StringConstraints(pattern=_NAME_PATTERN)]
    version: Annotated[TrimmedString, StringConstraints(pattern=_SEMVER_PATTERN)]
    description: TrimmedString
    handler: ToolHandler
    args_model: type[BaseModel]
    output_model: type[BaseModel]
    required_permissions: Sequence[Permission] = ()
    timeout: float = Field(
        default=10.0, gt=0, allow_inf_nan=False
    )
    max_retries: StrictInt = Field(default=0, ge=0)
    sensitive_arg_fields: Sequence[TrimmedString] = ()
    audit_output_mode: Annotated[ToolAuditOutputMode, _strict_string] = (
        ToolAuditOutputMode.SUMMARY
    )

    # 将字符串序列去重并转为元组，保留首次出现的顺序。
    @field_validator(
        "required_permissions", "sensitive_arg_fields"
    )
    @classmethod
    def normalize_sequences(cls, value: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value))


__all__ = [
    "ToolAuditOutputMode",
    "ToolHandler",
    "ToolRiskLevel",
    "ToolSpec",
]

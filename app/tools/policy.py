"""工具治理策略与稳定参数指纹。"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from app.contracts.chat import RAG_TOOL_NAMES, ToolMode
from app.tools.context import (
    PrincipalType,
    ToolInvocationContext,
    has_permission,
)
from app.tools.contracts import ToolSpec
from app.tools.errors import ToolErrorCode


class ToolPolicyDecisionType(str, Enum):
    """只读工具策略门的允许或拒绝决策。"""

    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class ToolPolicyDecision:
    """一次工具调用的结构化策略决策。"""

    decision: ToolPolicyDecisionType
    reason: str
    missing_permissions: tuple[str, ...] = ()
    args_fingerprint: str = ""
    error_code: ToolErrorCode | None = None


def _canonicalize_json_value(value: Any) -> Any:
    """把常见参数类型转换为无内存地址的稳定 JSON 值。"""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("工具参数不能包含 NaN 或 Infinity")
        return value
    if isinstance(value, Decimal):
        return {"$decimal": format(value, "f")}
    if isinstance(value, (datetime, date, time)):
        return {"$datetime": value.isoformat()}
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, bytes):
        return {
            "$bytes_base64": base64.b64encode(value).decode("ascii")
        }
    if isinstance(value, Enum):
        return _canonicalize_json_value(value.value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("工具参数字典的键必须是字符串")
            normalized[key] = _canonicalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, Sequence):
        return [_canonicalize_json_value(item) for item in value]
    raise TypeError(
        f"工具参数包含不支持稳定序列化的类型: {type(value).__name__}"
    )


def build_tool_args_fingerprint(
    *,
    tool_name: str,
    tool_version: str,
    args: dict[str, Any],
) -> str:
    """生成绑定工具身份、版本和规范化参数的 SHA-256 指纹。"""
    payload = {
        "tool_name": tool_name,
        "tool_version": tool_version,
        "args": _canonicalize_json_value(args),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


class ToolPolicyEngine:
    """评估当前只读 Local Tool 的请求范围、主体和权限。"""

    def evaluate(
        self,
        *,
        spec: ToolSpec,
        args: dict[str, Any],
        context: ToolInvocationContext,
        args_fingerprint: str,
    ) -> ToolPolicyDecision:
        """返回确定性策略结果，不调用 handler 或修改上下文。"""
        del args  # 参数内容仅通过已经绑定的指纹进入本阶段决策结果。

        if context.tool_mode is ToolMode.RAG_ONLY and spec.name not in RAG_TOOL_NAMES:
            return self._deny(
                reason="当前请求仅允许使用知识库检索与文档发现工具",
                code=ToolErrorCode.POLICY_DENIED,

                args_fingerprint=args_fingerprint,
            )

        if not isinstance(context.principal_type, PrincipalType):
            return self._deny(
                reason="调用主体类型无效",
                code=ToolErrorCode.INVALID_INVOCATION_CONTEXT,

                args_fingerprint=args_fingerprint,
            )

        missing_permissions = tuple(
            permission
            for permission in spec.required_permissions
            if not has_permission(context.permissions, permission)
        )
        if missing_permissions:
            return self._deny(
                reason="调用主体缺少工具要求的权限",
                code=ToolErrorCode.PERMISSION_DENIED,

                args_fingerprint=args_fingerprint,
                missing_permissions=missing_permissions,
            )

        return ToolPolicyDecision(
            decision=ToolPolicyDecisionType.ALLOW,
            reason="工具调用满足当前治理策略",

            args_fingerprint=args_fingerprint,
        )

    @staticmethod
    def _deny(
        *,
        reason: str,
        code: ToolErrorCode,
        args_fingerprint: str,
        missing_permissions: tuple[str, ...] = (),
    ) -> ToolPolicyDecision:
        """构造策略拒绝结果。"""
        return ToolPolicyDecision(
            decision=ToolPolicyDecisionType.DENY,
            reason=reason,
            missing_permissions=missing_permissions,

            args_fingerprint=args_fingerprint,
            error_code=code,
        )


__all__ = [
    "ToolPolicyDecision",
    "ToolPolicyDecisionType",
    "ToolPolicyEngine",
    "build_tool_args_fingerprint",
]

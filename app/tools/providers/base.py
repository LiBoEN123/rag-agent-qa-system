"""工具 Provider 抽象与可见性规则。"""

from collections.abc import Sequence
from typing import Protocol

from app.tools.context import ToolInvocationContext, has_permission
from app.tools.contracts import ToolSpec


class ToolProvider(Protocol):
    """按请求上下文提供当前 Agent 可见的注册工具定义。"""

    def list_specs(
        self,
        *,
        context: ToolInvocationContext,
    ) -> Sequence[ToolSpec]:
        """返回当前请求允许暴露给 Agent Model 的稳定有序 ToolSpec。"""
        ...


def is_tool_visible(
    spec: ToolSpec,
    *,
    context: ToolInvocationContext,
) -> bool:
    """仅按工具声明的必需权限判断当前请求是否可见。"""
    return all(
        has_permission(context.permissions, permission)
        for permission in spec.required_permissions
    )


__all__ = ["ToolProvider", "is_tool_visible"]

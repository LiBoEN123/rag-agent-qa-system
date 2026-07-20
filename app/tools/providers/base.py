"""工具 Provider 抽象与多 Provider 收集辅助函数。"""

from collections.abc import Sequence
from typing import Protocol

from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec


class ToolProvider(Protocol):
    """按请求上下文提供当前 Agent 可见的注册工具定义。"""

    def list_specs(
        self,
        *,
        context: ToolInvocationContext,
    ) -> Sequence[ToolSpec]:
        """返回稳定有序的 ToolSpec，不执行任何工具。"""
        ...


def collect_tool_specs(
    providers: Sequence[ToolProvider],
    *,
    context: ToolInvocationContext,
) -> list[ToolSpec]:
    """按 Provider 顺序收集工具，并拒绝跨 Provider 的重名定义。"""
    collected: list[ToolSpec] = []
    seen_names: set[str] = set()
    for provider in providers:
        for spec in provider.list_specs(context=context):
            if not isinstance(spec, ToolSpec):
                raise TypeError("ToolProvider 只能返回 ToolSpec")
            if spec.name in seen_names:
                raise ValueError(f"tool {spec.name!r} 在多个 Provider 中重复")
            seen_names.add(spec.name)
            collected.append(spec)
    return collected


__all__ = ["ToolProvider", "collect_tool_specs"]

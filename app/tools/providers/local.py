"""本地 ToolRegistry Provider。"""

from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.providers.base import is_tool_visible
from app.tools.registry import ToolRegistry, get_tool_registry


class LocalToolProvider:
    """按注册顺序暴露当前请求有权看到的本地工具定义。"""

    def __init__(self, registry: ToolRegistry):
        """绑定一个注册表；Provider 本身不缓存任何请求上下文。"""
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry 必须是 ToolRegistry")
        self._registry = registry

    def __eq__(self, other: object) -> bool:
        """同一 Registry 的无状态 Provider 共享 Runtime 编译缓存。"""
        return (
            isinstance(other, LocalToolProvider)
            and self._registry is other._registry
        )

    def __hash__(self) -> int:
        """缓存身份由实际 Registry 对象决定。"""
        return hash(self._registry)

    @property
    def registry(self) -> ToolRegistry:
        """返回工具定义所属注册表，供适配层统一分发。"""
        return self._registry

    def list_specs(
        self,
        *,
        context: ToolInvocationContext,
    ) -> list[ToolSpec]:
        """返回权限可见的 ToolSpec，不执行 handler、Policy 或 Runtime。"""
        specs: list[ToolSpec] = []
        for name in self._registry.names():
            spec = self._registry.get(name)
            if spec is None:
                raise RuntimeError(
                    f"tool registry 中不存在已列出的工具: {name}"
                )
            if is_tool_visible(spec, context=context):
                specs.append(spec)
        return specs


def get_default_tool_provider() -> LocalToolProvider:
    """构造默认本地 Provider，不持有请求级 Context。"""
    return LocalToolProvider(get_tool_registry())


__all__ = ["LocalToolProvider", "get_default_tool_provider"]

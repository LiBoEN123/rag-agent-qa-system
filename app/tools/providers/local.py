"""本地 ToolRegistry Provider。"""

from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.registry import ToolRegistry, tool_registry


class LocalToolProvider:
    """按注册顺序暴露指定 ToolRegistry 中的工具定义。

    当前版本保留 Runtime 作为唯一权限判定点，因此不在模型可见性阶段过滤
    权限。这样模型误选或伪造工具调用时，仍会得到结构化治理拒绝结果。
    """

    def __init__(self, registry: ToolRegistry):
        """绑定一个注册表；Provider 本身不缓存任何请求上下文。"""
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry 必须是 ToolRegistry")
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        """返回工具定义所属注册表，供适配层统一分发。"""
        return self._registry

    def list_specs(
        self,
        *,
        context: ToolInvocationContext,
    ) -> list[ToolSpec]:
        """返回当前注册表中的 ToolSpec，不执行 handler 或 Runtime。"""
        del context
        specs: list[ToolSpec] = []
        for name in self._registry.names():
            spec = self._registry.get(name)
            if spec is None:
                raise RuntimeError(
                    f"tool registry 中不存在已列出的工具: {name}"
                )
            specs.append(spec)
        return specs


def get_default_tool_provider() -> LocalToolProvider:
    """构造默认本地 Provider，不持有请求级 Context。"""
    return LocalToolProvider(tool_registry)


__all__ = ["LocalToolProvider", "get_default_tool_provider"]

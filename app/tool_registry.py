from dataclasses import dataclass
from typing import Callable

import app.tools as tools


@dataclass
class ToolSpec:
    """描述一个可注册工具的基础信息。"""
    name: str
    description: str
    args_schema: dict
    output_schema: dict
    handler: Callable[[dict], dict]
    timeout: int = 10
    safe_level: str = "safe"


class ToolRegistry:
    """统一管理工具注册、查询和分发。"""

    def __init__(self):
        """初始化空工具注册表。"""
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """注册一个工具定义。"""
        if not spec.name:
            raise ValueError("tool name 不能为空")

        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        """按名称获取工具定义。"""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """返回已注册工具名称列表。"""
        return list(self._tools.keys())

    def router_specs(self) -> list[dict]:
        """返回 Router 可读取的工具描述。"""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "args_schema": spec.args_schema,
            }
            for spec in self._tools.values()
        ]

    def dispatch(self, name: str, args: dict | None = None) -> dict:
        """按名称分发工具调用。"""
        tool_args = args or {}
        spec = self.get(name)

        if spec is None:
            return tools.tool_error(
                tool_name=name,
                tool_input=tool_args,
                error=f"unknown tool: {name}"
            )

        try:
            return spec.handler(tool_args)
        except Exception as e:
            return tools.tool_error(
                tool_name=name,
                tool_input=tool_args,
                error=str(e)
            )


def _handle_simple_calculator(args: dict) -> dict:
    """适配计算器工具的参数格式。"""
    return tools.simple_calculator(args.get("expression", ""))


def _handle_list_documents(args: dict) -> dict:
    """适配文档列表工具的参数格式。"""
    return tools.list_documents()


def _handle_get_chunk_text(args: dict) -> dict:
    """适配 chunk 原文工具的参数格式。"""
    return tools.get_chunk_text(args.get("chunk_id"))


def register_default_tools(registry: ToolRegistry) -> None:
    """注册项目内置工具。"""
    registry.register(
        ToolSpec(
            name="simple_calculator",
            description="执行简单数字四则运算",
            args_schema={"expression": "string, 必填，例如 '1 + 2 * 3'"},
            output_schema={"expression": "string", "result": "number"},
            handler=_handle_simple_calculator,
        )
    )
    registry.register(
        ToolSpec(
            name="list_documents",
            description="列出知识库文档及数量",
            args_schema={},
            output_schema={"count": "integer", "documents": "list"},
            handler=_handle_list_documents,
        )
    )
    registry.register(
        ToolSpec(
            name="get_chunk_text",
            description="查看指定 chunk_id 的完整原文",
            args_schema={"chunk_id": "integer|string, 必填"},
            output_schema={"chunk_id": "integer|string", "chunk": "dict"},
            handler=_handle_get_chunk_text,
        )
    )


tool_registry = ToolRegistry()
register_default_tools(tool_registry)

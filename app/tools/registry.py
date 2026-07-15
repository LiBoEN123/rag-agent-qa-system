"""工具注册表模块。

定义工具描述、参数校验和分发协议，并注册系统内置工具集合。
"""

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ValidationError

from app.tools.calculator import simple_calculator
from app.tools.compare_documents import compare_documents_tool
from app.tools.context import ToolInvocationContext
from app.tools.knowledge_base import (
    get_chunk_text,
    list_documents,
    search_knowledge_base,
)
from app.tools.result import tool_error
from app.tools.schemas import (
    CompareDocumentsArgs,
    GetChunkTextArgs,
    ListDocumentsArgs,
    SearchKnowledgeBaseArgs,
    SimpleCalculatorArgs,
)


@dataclass
class ToolSpec:
    """描述一个可注册工具的基础信息。"""
    name: str
    description: str
    args_schema: dict
    output_schema: dict
    handler: Callable[[dict, ToolInvocationContext], dict]
    timeout: int = 10
    safe_level: str = "safe"
    args_model: type[BaseModel] | None = None


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

    def dispatch(
        self,
        name: str,
        args: dict | None = None,
        *,
        context: ToolInvocationContext | None = None,
    ) -> dict:
        """按名称分发工具调用。"""
        tool_args = args or {}
        invocation_context = context or ToolInvocationContext()
        spec = self.get(name)

        if spec is None:
            return tool_error(
                tool_name=name,
                tool_input=tool_args,
                error=f"unknown tool: {name}"
            )

        if spec.args_model is not None:
            try:
                validated_args = spec.args_model.model_validate(tool_args).model_dump()
            except ValidationError as e:
                return tool_error(
                    tool_name=name,
                    tool_input=tool_args,
                    error=f"invalid tool args: {e}"
                )
        else:
            validated_args = tool_args

        try:
            return spec.handler(validated_args, invocation_context)
        except Exception as e:
            return tool_error(
                tool_name=name,
                tool_input=validated_args,
                error=str(e)
            )


def _handle_simple_calculator(
    args: dict,
    context: ToolInvocationContext,
) -> dict:
    """适配计算器工具的参数格式。"""
    return simple_calculator(args.get("expression", ""))


def _handle_list_documents(
    args: dict,
    context: ToolInvocationContext,
) -> dict:
    """适配文档列表工具的参数格式。"""
    return list_documents()


def _handle_get_chunk_text(
    args: dict,
    context: ToolInvocationContext,
) -> dict:
    """适配 chunk 原文工具的参数格式。"""
    return get_chunk_text(args.get("chunk_id"))


def _handle_search_knowledge_base(
    args: dict,
    context: ToolInvocationContext,
) -> dict:
    """适配知识库检索工具的参数格式。"""
    return search_knowledge_base(args)


def _handle_compare_documents(
    args: dict,
    context: ToolInvocationContext,
) -> dict:
    """适配文档对比工具的参数格式。"""
    return compare_documents_tool(args, trace_id=context.trace_id)


def register_default_tools(registry: ToolRegistry) -> None:
    """注册项目内置工具。"""
    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            description="检索知识库并返回相关证据 chunk",
            args_schema={
                "query": "string, 必填，用户问题或检索问题",
                "top_k": "integer, 可选，返回候选 chunk 数量",
            },
            output_schema={
                "count": "integer, 命中的 chunk 数量",
                "chunks": "list, 候选证据 chunk 列表",
                "top_chunk_ids": "list, 命中的 chunk_id 列表",
            },
            handler=_handle_search_knowledge_base,
            args_model=SearchKnowledgeBaseArgs,
        )
    )
    registry.register(
        ToolSpec(
            name="compare_documents",
            description="对比两个知识库文档中关于同一问题的相关规定差异",
            args_schema={
                "query": "string, 必填，对比主题或问题",
                "doc_a": "string, 必填，第一个文档名或文档名关键词",
                "doc_b": "string, 必填，第二个文档名或文档名关键词",
                "top_k": "integer, 可选，每个文档返回的证据数量，默认 3",
            },
            output_schema={
                "query": "string, 对比主题",
                "doc_a": "string, 第一个文档",
                "doc_b": "string, 第二个文档",
                "doc_a_evidence": "list, 第一个文档的证据片段",
                "doc_b_evidence": "list, 第二个文档的证据片段",
                "comparison": "string, 差异说明",
                "conflict_detected": "boolean, 是否可能存在差异或冲突",
                "summary": "string, 简要摘要",
            },
            handler=_handle_compare_documents,
            args_model=CompareDocumentsArgs,
        )
    )
    registry.register(
        ToolSpec(
            name="simple_calculator",
            description="执行简单数字四则运算",
            args_schema={"expression": "string, 必填，例如 '1 + 2 * 3'"},
            output_schema={"expression": "string", "result": "number"},
            handler=_handle_simple_calculator,
            args_model=SimpleCalculatorArgs,
        )
    )
    registry.register(
        ToolSpec(
            name="list_documents",
            description="列出知识库文档及数量",
            args_schema={},
            output_schema={"count": "integer", "documents": "list"},
            handler=_handle_list_documents,
            args_model=ListDocumentsArgs,
        )
    )
    registry.register(
        ToolSpec(
            name="get_chunk_text",
            description="查看指定 chunk_id 的完整原文",
            args_schema={"chunk_id": "integer|string, 必填"},
            output_schema={"chunk_id": "integer|string", "chunk": "dict"},
            handler=_handle_get_chunk_text,
            args_model=GetChunkTextArgs,
        )
    )


tool_registry = ToolRegistry()
register_default_tools(tool_registry)

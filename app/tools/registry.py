"""工具注册表模块。

管理工具查询和分发，并注册系统内置工具集合。
"""

from app.tools.calculator import simple_calculator
from app.tools.compare_documents import compare_documents_tool
from app.tools.contracts import (
    ToolAuditOutputMode,
    ToolRiskLevel,
    ToolSpec,
)
from app.tools.context import ToolInvocationContext
from app.tools.knowledge_base import (
    get_chunk_text,
    list_documents,
    search_knowledge_base,
)
from app.tools.schemas import (
    CompareDocumentsArgs,
    GetChunkTextArgs,
    ListDocumentsArgs,
    SearchKnowledgeBaseArgs,
    SimpleCalculatorArgs,
)


class ToolRegistry:
    """统一管理工具注册、查询和分发。"""

    def __init__(self):
        """初始化空工具注册表。"""
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """注册一个工具定义。"""
        if not isinstance(spec, ToolSpec):
            raise TypeError("spec 必须是 ToolSpec")

        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} 已注册，禁止静默覆盖")

        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        """按名称获取工具定义。"""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """返回已注册工具名称列表。"""
        return list(self._tools.keys())

    def router_specs(self) -> list[dict]:
        """返回 Router 可读取的工具描述。"""
        return [spec.to_router_spec() for spec in self._tools.values()]

    def contract_specs(self) -> list[dict]:
        """返回完整且可序列化的工具契约。"""
        return [spec.to_contract_dict() for spec in self._tools.values()]

    def dispatch(
        self,
        name: str,
        args: dict | None = None,
        *,
        context: ToolInvocationContext | None = None,
    ) -> dict:
        """按名称通过统一 ToolRuntime 分发工具调用。"""
        from app.tools.runtime import execute_registered_tool

        return execute_registered_tool(
            registry=self,
            tool_name=name,
            tool_args=args,
            context=context,
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
            version="1.0.0",
            description=(
                "用于回答需要依据当前本地知识库文档的问题；传入检索问题和"
                "候选数量，返回相关证据 chunk。仅检索本地知识库，不用于互联网搜索。"
            ),
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
            risk_level=ToolRiskLevel.READ_ONLY,
            required_permissions=("knowledge_base:read",),
            timeout=10.0,
            max_retries=0,
            idempotent=True,
            requires_approval=False,
            audit_output_mode=ToolAuditOutputMode.SUMMARY,
        )
    )
    registry.register(
        ToolSpec(
            name="compare_documents",
            version="1.0.0",
            description=(
                "用于比较两个已知知识库文档对同一主题的规定；传入主题、两个"
                "文档名和候选数量，返回双方证据与差异摘要。不用于未知文档的泛检索。"
            ),
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
            risk_level=ToolRiskLevel.READ_ONLY,
            required_permissions=("knowledge_base:read",),
            timeout=20.0,
            max_retries=0,
            idempotent=True,
            requires_approval=False,
            audit_output_mode=ToolAuditOutputMode.SUMMARY,
        )
    )
    registry.register(
        ToolSpec(
            name="simple_calculator",
            version="1.0.0",
            description=(
                "用于计算明确的数字四则运算表达式并返回数值结果；参数 expression"
                " 是待计算表达式。不用于解释知识库中的数字含义或执行代码。"
            ),
            args_schema={"expression": "string, 必填，例如 '1 + 2 * 3'"},
            output_schema={"expression": "string", "result": "number"},
            handler=_handle_simple_calculator,
            args_model=SimpleCalculatorArgs,
            risk_level=ToolRiskLevel.READ_ONLY,
            required_permissions=(),
            timeout=3.0,
            max_retries=0,
            idempotent=True,
            requires_approval=False,
            audit_output_mode=ToolAuditOutputMode.SANITIZED,
        )
    )
    registry.register(
        ToolSpec(
            name="list_documents",
            version="1.0.0",
            description=(
                "用于查看当前本地知识库包含的文档名称和数量，无需参数；"
                "返回文档列表，不返回文档全文或互联网内容。"
            ),
            args_schema={},
            output_schema={"count": "integer", "documents": "list"},
            handler=_handle_list_documents,
            args_model=ListDocumentsArgs,
            risk_level=ToolRiskLevel.READ_ONLY,
            required_permissions=("knowledge_base:read",),
            timeout=5.0,
            max_retries=0,
            idempotent=True,
            requires_approval=False,
            audit_output_mode=ToolAuditOutputMode.SUMMARY,
        )
    )
    registry.register(
        ToolSpec(
            name="get_chunk_text",
            version="1.0.0",
            description=(
                "用于在已知 chunk_id 时读取该知识库片段的完整原文；参数 chunk_id"
                " 是片段标识。若不知道 chunk_id，应先检索知识库。"
            ),
            args_schema={"chunk_id": "integer|string, 必填"},
            output_schema={"chunk_id": "integer|string", "chunk": "dict"},
            handler=_handle_get_chunk_text,
            args_model=GetChunkTextArgs,
            risk_level=ToolRiskLevel.READ_ONLY,
            required_permissions=("knowledge_base:read",),
            timeout=5.0,
            max_retries=0,
            idempotent=True,
            requires_approval=False,
            audit_output_mode=ToolAuditOutputMode.NONE,
        )
    )


tool_registry = ToolRegistry()
register_default_tools(tool_registry)


__all__ = [
    "ToolRegistry",
    "ToolAuditOutputMode",
    "ToolRiskLevel",
    "ToolSpec",
    "register_default_tools",
    "tool_registry",
]

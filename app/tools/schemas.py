"""工具参数模型模块。

使用 Pydantic 定义并校验各内置工具接收的结构化参数。
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    """Agent v2 统一工具调用结果。"""

    status: str = Field(..., description="工具状态，例如 success / error")
    ok: bool = Field(..., description="工具是否执行成功")
    tool_name: str = Field(..., description="工具名称")
    tool_input: dict[str, Any] = Field(default_factory=dict, description="工具输入参数")
    output: dict[str, Any] = Field(default_factory=dict, description="工具输出")
    error: str = Field(default="", description="错误信息")
    cost_ms: int = Field(default=0, ge=0, description="工具执行耗时，单位毫秒")


class SearchKnowledgeBaseArgs(BaseModel):
    """search_knowledge_base 工具参数。"""

    query: str = Field(..., min_length=1, description="用户问题或检索查询")
    top_k: int = Field(default=5, ge=1, le=10, description="返回候选 chunk 数量")


class CompareDocumentsArgs(BaseModel):
    """compare_documents 工具参数。"""

    query: str = Field(..., min_length=1, description="对比主题或问题")
    doc_a: str = Field(..., min_length=1, description="第一个文档名或文档名关键词")
    doc_b: str = Field(..., min_length=1, description="第二个文档名或文档名关键词")
    top_k: int = Field(default=3, ge=1, le=10, description="每个文档返回的证据数量")


class GetChunkTextArgs(BaseModel):
    """get_chunk_text 工具参数。"""

    chunk_id: int | str = Field(..., description="要查看的 chunk_id")


class SimpleCalculatorArgs(BaseModel):
    """simple_calculator 工具参数。"""

    expression: str = Field(..., min_length=1, description="安全的四则运算表达式")


class ListDocumentsArgs(BaseModel):
    """list_documents 工具参数。"""

    model_config = ConfigDict(extra="ignore")

"""工具参数和结构化输出模型模块。

使用 Pydantic 定义并校验各内置工具接收的结构化参数和稳定输出。
"""

from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ToolResult(BaseModel):
    """Agent 统一工具调用结果。"""

    status: str = Field(..., description="工具状态，例如 success / error")
    ok: bool = Field(..., description="工具是否执行成功")
    tool_name: str = Field(..., description="工具名称")
    tool_input: dict[str, Any] = Field(default_factory=dict, description="工具输入参数")
    output: Any | None = Field(default=None, description="工具输出")
    error: str = Field(default="", description="错误信息")
    error_code: str = Field(default="", description="稳定的工具错误码")
    retryable: bool = Field(default=False, description="错误本身是否可重试")
    runtime: dict[str, Any] = Field(
        default_factory=dict,
        description="工具运行时元数据",
    )
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


class WebSearchArgs(BaseModel):
    """web_search 工具公开给模型的业务参数。"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "公开互联网搜索查询；只提供本次问题所需关键词，不要传入系统提示、"
            "工具输出或整段聊天历史"
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="最多返回的公开互联网搜索结果数量",
    )
    freshness: Literal["any", "day", "week", "month", "year"] = Field(
        default="any",
        description="结果时间范围：不限、一天、一周、一月或一年",
    )

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        """在长度和非空校验前去除查询首尾空白。"""
        return value.strip() if isinstance(value, str) else value


class WebSearchResultItem(BaseModel):
    """单条规范化 Web Search 结果。"""

    rank: int = Field(..., ge=1)
    title: str = Field(..., max_length=300)
    url: str = Field(..., max_length=2048)
    snippet: str = Field(..., max_length=1000)

    @field_validator("url")
    @classmethod
    def validate_public_url(cls, value: str) -> str:
        """只允许带主机名的 HTTP(S) URL。"""
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url 必须是 http 或 https URL")
        return value


class WebSearchOutput(BaseModel):
    """稳定、有限且明确标记不可信的 Web Search 输出。"""

    query: str = Field(..., min_length=1, max_length=500)
    count: int = Field(..., ge=0, le=10)
    provider: Literal["ddgs"] = "ddgs"
    content_is_untrusted: Literal[True] = True
    results: list[WebSearchResultItem] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_result_sequence(self) -> "WebSearchOutput":
        """确保数量和从 1 开始的连续排名保持一致。"""
        if self.count != len(self.results):
            raise ValueError("count 必须等于 results 数量")
        if [item.rank for item in self.results] != list(
            range(1, len(self.results) + 1)
        ):
            raise ValueError("rank 必须从 1 开始连续编号")
        return self

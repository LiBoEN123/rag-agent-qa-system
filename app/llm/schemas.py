"""项目中各 LLM 角色的结构化输出协议。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AnswerOutput(BaseModel):
    """RAG 最终回答模型输出。"""

    model_config = ConfigDict(strict=True)

    answer: str = Field(description="严格依据检索证据生成的回答正文")
    references: list[str] = Field(
        default_factory=list,
        description="回答引用的 reference_id 列表",
    )


class QueryRewriteOutput(BaseModel):
    """多轮查询改写模型输出。"""

    model_config = ConfigDict(strict=True, str_strip_whitespace=True)

    need_rewrite: bool
    rewritten_question: str = Field(min_length=1)
    reason: str = ""


class QueryDecomposeItem(BaseModel):
    """问题拆分协议中的单个子问题。"""

    model_config = ConfigDict(strict=True, str_strip_whitespace=True)

    id: int | str = 0
    sub_question: str = Field(min_length=1)
    search_query: str = Field(min_length=1)
    must_answer: bool = True


class QueryDecomposeOutput(BaseModel):
    """问题拆分模型输出，字段与现有检索链路的真实协议一致。"""

    model_config = ConfigDict(strict=True, str_strip_whitespace=True)

    need_decompose: bool = False
    sub_queries: list[QueryDecomposeItem] = Field(default_factory=list)
    reason: str = ""

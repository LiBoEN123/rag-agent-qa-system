"""项目中各 LLM 角色的结构化输出协议。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnswerOutput(BaseModel):
    """RAG 最终回答模型输出。"""

    answer: str
    references: list[str] = Field(default_factory=list)


class RouterDecisionOutput(BaseModel):
    """Agent Router 决策输出，非关键字段均提供安全默认值。"""

    decision: str = ""
    action: str
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    clarification_question: str = ""
    confidence: float = 0.0


class QueryRewriteOutput(BaseModel):
    """多轮查询改写模型输出。"""

    need_rewrite: bool = False
    rewritten_question: str
    reason: str = ""


class QueryDecomposeItem(BaseModel):
    """问题拆分协议中的单个子问题。"""

    id: int | str = 0
    sub_question: str
    search_query: str
    must_answer: bool = True


class QueryDecomposeOutput(BaseModel):
    """问题拆分模型输出，字段与现有检索链路的真实协议一致。"""

    need_decompose: bool = False
    sub_queries: list[QueryDecomposeItem] = Field(default_factory=list)
    reason: str = ""

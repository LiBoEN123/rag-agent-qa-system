"""项目中各 LLM 角色的结构化输出协议。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QueryRewriteOutput(BaseModel):
    """多轮查询改写模型输出。"""

    model_config = ConfigDict(strict=True, str_strip_whitespace=True)

    need_rewrite: bool
    rewritten_question: str = Field(min_length=1)
    reason: str = ""

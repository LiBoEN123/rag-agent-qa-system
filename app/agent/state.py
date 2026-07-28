"""Agent observation 与 API 兼容运行状态模型。"""

from typing import Any, Literal

from pydantic import BaseModel, Field


AgentStatus = Literal[
    "running",
    "ok",
    "clarify",
    "direct_answer",
    "tool_error",
    "model_error",
    "no_evidence",
    "error",
]


class AgentObservation(BaseModel):
    """Agent 中某一步执行后的观察结果。"""

    step_id: int = Field(..., ge=1, description="对应的计划步骤编号")
    tool_name: str = Field(default="", description="本次观察对应的工具名")
    success: bool = Field(default=False, description="该步骤是否执行成功")
    summary: str = Field(default="", description="工具执行结果摘要")
    output: dict[str, Any] = Field(default_factory=dict, description="工具原始输出或规整后的输出")
    error: str = Field(default="", description="失败原因")
    cost_ms: int = Field(default=0, ge=0, description="该步骤耗时，单位毫秒")


class AgentState(BaseModel):
    """Agent 单次请求的完整运行状态。"""

    trace_id: str = Field(default="", description="请求追踪 ID")
    session_id: str | None = Field(default=None, description="多轮会话 ID")

    original_question: str = Field(..., description="用户原始问题")
    rewritten_question: str = Field(default="", description="多轮改写后的问题")

    top_k: int = Field(default=5, ge=1, le=10, description="单次召回数量")
    decompose_top_k: int = Field(default=10, ge=1, le=20, description="多子问题召回合并后的候选上限")

    observations: list[AgentObservation] = Field(default_factory=list, description="每一步工具执行后的观察结果")

    final_answer: str = Field(default="", description="最终回答")
    references: list[dict[str, Any]] = Field(default_factory=list, description="最终引用")
    status: AgentStatus = Field(default="running", description="Agent 当前状态")
    reason: str = Field(default="", description="状态原因或失败原因")

    router_llm_used: bool = Field(default=False, description="是否调用 Router LLM")
    decompose_llm_used: bool = Field(default=False, description="是否调用问题拆分 LLM")
    answer_llm_used: bool = Field(default=False, description="是否调用最终回答 LLM")
    llm_call_count: int = Field(default=0, ge=0, description="本次请求累计 LLM 调用次数")

    tool_trace: list[dict[str, Any]] = Field(default_factory=list, description="兼容现有响应结构的工具执行 trace")


def get_effective_question(state: AgentState) -> str:
    """返回当前 Agent 真正用于规划、检索和回答的问题。"""
    return (
        (state.rewritten_question or "").strip()
        or (state.original_question or "").strip()
    )


__all__ = [
    "AgentObservation",
    "AgentState",
    "AgentStatus",
    "get_effective_question",
]

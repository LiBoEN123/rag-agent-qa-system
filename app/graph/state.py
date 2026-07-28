"""LangGraph 父图状态定义。"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agent.router import RouteDecision
from app.agent.state import AgentObservation, AgentStatus


class AgentGraphState(TypedDict, total=False):
    """一次 Agent 请求唯一的完整 LangGraph 运行状态。"""

    original_question: str
    effective_question: str
    top_k: int
    decompose_top_k: int
    decision: RouteDecision | None
    route_cost_ms: int
    trace_id: str
    session_id: str | None
    messages: Annotated[list[BaseMessage], add_messages]
    governance_events: Annotated[list[dict[str, Any]], operator.add]
    framework_events_applied: int
    iteration_count: int
    tool_call_count: int
    tool_error_count: int
    last_tool_call_fingerprint: str
    consecutive_same_tool_call_count: int
    last_tool_error_fingerprint: str
    consecutive_same_tool_error_count: int
    agent_elapsed_ms: int
    stop_reason: str
    pending_rag_result: dict[str, Any]
    observations: list[AgentObservation]
    final_answer: str
    references: list[dict[str, Any]]
    status: AgentStatus
    reason: str
    decompose_llm_used: bool
    answer_llm_used: bool
    llm_call_count: int
    tool_trace: list[dict[str, Any]]
    result: dict[str, Any]

__all__ = ["AgentGraphState"]

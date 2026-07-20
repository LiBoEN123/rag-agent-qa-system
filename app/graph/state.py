"""LangGraph 主图状态定义。"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agent.state import AgentState


class AgentGraphState(TypedDict, total=False):
    """在 LangGraph 节点间传递可变业务状态。"""

    original_question: str
    effective_question: str
    top_k: int
    decompose_top_k: int
    decision: dict[str, Any] | None
    trace_id: str
    session_id: str | None
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int
    max_iterations: int
    tool_call_count: int
    max_tool_calls: int
    tool_error_count: int
    stop_reason: str
    fallback_to_legacy: bool
    agent_state: AgentState
    result: dict[str, Any]

__all__ = ["AgentGraphState"]

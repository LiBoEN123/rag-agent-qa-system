"""LangGraph 父图状态定义。"""

import operator
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
    agent_state: AgentState
    result: dict[str, Any]

__all__ = ["AgentGraphState"]

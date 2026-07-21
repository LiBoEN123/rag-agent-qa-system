"""LangGraph 主图的条件边函数。"""

from typing import Literal

from langchain_core.messages import AIMessage

from app.agent.guardrails import (
    TERMINAL_AGENT_STOP_REASONS,
    is_agent_elapsed_budget_exhausted,
)
from app.agent.state import AgentState
from app.graph.state import AgentGraphState


def route_condition(state: AgentGraphState) -> Literal["route", "initialize"]:
    """仅在调用方未提供路由决策时进入 Router 节点。"""
    return "route" if state.get("decision") is None else "initialize"


def after_execute_condition(
    state: AgentGraphState,
) -> Literal["rag_subgraph", "compose"]:
    """成功知识库 observation 进入 RAG 子图，其余状态继续旧 Compose。"""
    agent_state = state.get("agent_state")
    if not isinstance(agent_state, AgentState):
        return "compose"
    if agent_state.status in {
        "clarify",
        "direct_answer",
        "tool_error",
        "model_error",
        "error",
    }:
        return "compose"

    for observation in agent_state.observations:
        if observation.tool_name != "search_knowledge_base":
            continue
        if not observation.success or not isinstance(observation.output, dict):
            continue
        if isinstance(observation.output.get("chunks"), list):
            return "rag_subgraph"
    return "compose"


def after_initialize_condition(
    state: AgentGraphState,
) -> Literal["agent_model", "legacy_plan"]:
    """知识库初始路由保留专用 RAG 链，其余请求进入模型循环。"""
    decision = state.get("decision")
    if isinstance(decision, dict) and decision.get("action") in {
        "rag_fast_path",
        "search_knowledge_base",
        "clarify",
        "direct_answer",
    }:
        return "legacy_plan"
    return "agent_model"


def _last_ai_message(state: AgentGraphState) -> AIMessage | None:
    """返回消息轨迹中最后一条 AIMessage。"""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage):
            return message
    return None


def after_agent_model_condition(
    state: AgentGraphState,
) -> Literal["tools", "compose", "limit", "legacy_plan"]:
    """根据最新模型输出和 Agent 预算确定下一节点，不执行副作用。"""
    if state.get("stop_reason") == "agent_model_call_failed":
        return "compose"
    if state.get("fallback_to_legacy", False):
        return "legacy_plan"

    message = _last_ai_message(state)
    tool_calls = message.tool_calls if message is not None else []
    if not tool_calls:
        return "compose"

    iteration_count = max(0, int(state.get("iteration_count", 0) or 0))
    max_iterations = max(1, int(state.get("max_iterations", 1) or 1))
    tool_call_count = max(0, int(state.get("tool_call_count", 0) or 0))
    max_tool_calls = max(1, int(state.get("max_tool_calls", 1) or 1))
    agent_elapsed_ms = max(0, int(state.get("agent_elapsed_ms", 0) or 0))
    max_agent_elapsed_ms = max(
        1,
        int(state.get("max_agent_elapsed_ms", 1) or 1),
    )
    if iteration_count >= max_iterations:
        return "limit"
    if tool_call_count + len(tool_calls) > max_tool_calls:
        return "limit"
    if is_agent_elapsed_budget_exhausted(
        elapsed_ms=agent_elapsed_ms,
        limit_ms=max_agent_elapsed_ms,
    ):
        return "limit"
    return "tools"


def after_tools_condition(
    state: AgentGraphState,
) -> Literal["agent_model", "rag_subgraph", "compose"]:
    """治理错误停止，成功检索进入 RAG，其余工具观察返回模型。"""
    agent_state = state.get("agent_state")
    if not isinstance(agent_state, AgentState):
        return "compose"
    if str(state.get("stop_reason") or "") in TERMINAL_AGENT_STOP_REASONS:
        return "compose"

    for observation in reversed(agent_state.observations):
        if observation.tool_name != "search_knowledge_base":
            continue
        if observation.success and isinstance(observation.output, dict):
            if isinstance(observation.output.get("chunks"), list):
                return "rag_subgraph"
        break
    return "agent_model"


__all__ = [
    "after_agent_model_condition",
    "after_execute_condition",
    "after_initialize_condition",
    "after_tools_condition",
    "route_condition",
]

"""最小 LangGraph Agent 主图节点。

各节点只编排现有 Agent 业务函数，不重新实现路由、规划、执行或答案合成。
"""

from typing import Any

from app.agent import route_agent_action
from app.agent.answer import apply_rag_result_to_state, compose_final_answer
from app.agent.executor import execute_plan
from app.agent.planner import build_plan_from_decision
from app.agent.runtime import (
    append_llm_usage_trace,
    build_decide_trace,
    build_plan_trace,
    build_v2_result,
    maybe_save_agent_trace,
    normalize_llm_call_count,
)
from app.agent.state import AgentState
from app.config import Settings
from app.graph.state import AgentGraphState
from app.rag.graph import run_rag_subgraph


def _safe_decision(decision: object) -> dict[str, Any]:
    """把 Graph 中的路由决策规整为现有业务函数可读取的字典。"""
    return decision if isinstance(decision, dict) else {}


def _require_agent_state(state: AgentGraphState) -> AgentState:
    """读取已经初始化的 AgentState，缺失时抛出明确编排错误。"""
    agent_state = state.get("agent_state")
    if not isinstance(agent_state, AgentState):
        raise RuntimeError("LangGraph agent_state is not initialized")
    return agent_state


def route_node(state: AgentGraphState) -> dict[str, Any]:
    """调用现有 Router，并把决策写回 Graph State。"""
    decision = route_agent_action(
        state.get("question", ""),
        state.get("top_k", Settings.TOP_K),
        trace_id=state.get("trace_id", ""),
    )
    return {"decision": decision}


def initialize_agent_state_node(state: AgentGraphState) -> dict[str, AgentState]:
    """按现有业务协议创建 AgentState。"""
    decision = _safe_decision(state.get("decision"))
    router_llm_used = bool(
        decision.get(
            "router_llm_used",
            decision.get("router_source") in {"llm", "rule_based_fallback"},
        )
    )
    agent_state = AgentState(
        trace_id=state.get("trace_id", ""),
        session_id=state.get("session_id"),
        original_question=state.get("question", ""),
        rewritten_question="",
        top_k=state.get("top_k", Settings.TOP_K),
        decompose_top_k=state.get(
            "decompose_top_k",
            Settings.DECOMPOSE_TOP_K,
        ),
        max_steps=Settings.AGENT_MAX_STEPS,
        router_llm_used=router_llm_used,
    )
    agent_state.tool_trace.append(build_decide_trace(decision, router_llm_used))
    return {"agent_state": agent_state}


def plan_node(state: AgentGraphState) -> dict[str, AgentState]:
    """调用现有 Planner，并追加共享的 plan trace。"""
    agent_state = _require_agent_state(state)
    agent_state.plan = build_plan_from_decision(
        decision=_safe_decision(state.get("decision")),
        question=state.get("question", ""),
        top_k=state.get("top_k", Settings.TOP_K),
    )
    agent_state.tool_trace.append(build_plan_trace(agent_state))
    return {"agent_state": agent_state}


def execute_node(state: AgentGraphState) -> dict[str, AgentState]:
    """调用现有 Executor，并显式返回其更新后的 AgentState。"""
    agent_state = execute_plan(_require_agent_state(state))
    return {"agent_state": agent_state}


def rag_subgraph_node(state: AgentGraphState) -> dict[str, AgentState]:
    """复用成功知识库观察结果执行 RAG 子图，并写回现有 AgentState。"""
    agent_state = _require_agent_state(state)
    observation = next(
        (
            item
            for item in agent_state.observations
            if item.tool_name == "search_knowledge_base" and item.success
        ),
        None,
    )
    if observation is None or not isinstance(observation.output, dict):
        raise RuntimeError(
            "LangGraph RAG path requires a successful knowledge base observation"
        )

    retrieved_chunks = observation.output.get("chunks")
    if not isinstance(retrieved_chunks, list):
        raise RuntimeError(
            "Knowledge base observation chunks must be a list"
        )

    question = agent_state.rewritten_question or agent_state.original_question
    rag_result = run_rag_subgraph(
        question=question,
        top_k=agent_state.top_k,
        decompose_top_k=agent_state.decompose_top_k,
        retrieved_chunks=retrieved_chunks,
        trace_id=agent_state.trace_id,
    )
    return {
        "agent_state": apply_rag_result_to_state(agent_state, rag_result)
    }


def compose_node(state: AgentGraphState) -> dict[str, AgentState]:
    """调用现有 Answer Composer，并显式返回最终业务状态。"""
    agent_state = compose_final_answer(_require_agent_state(state))
    return {"agent_state": agent_state}


def finalize_node(state: AgentGraphState) -> dict[str, Any]:
    """复用共享的统计、结果映射和 trace 持久化逻辑。"""
    agent_state = _require_agent_state(state)
    decision = _safe_decision(state.get("decision"))

    normalize_llm_call_count(agent_state)
    append_llm_usage_trace(agent_state)
    result = build_v2_result(agent_state, decision)
    maybe_save_agent_trace(
        trace_id=agent_state.trace_id,
        session_id=agent_state.session_id,
        question=agent_state.original_question,
        rewritten_question=agent_state.rewritten_question,
        final_answer=result.get("answer", ""),
        status=result.get("status", ""),
        reason=result.get("reason", ""),
        tool_trace=result.get("tool_trace", []),
    )
    return {"agent_state": agent_state, "result": result}


__all__ = [
    "compose_node",
    "execute_node",
    "finalize_node",
    "initialize_agent_state_node",
    "plan_node",
    "rag_subgraph_node",
    "route_node",
]

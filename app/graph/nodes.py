"""LangGraph Agent 主图节点。

迭代节点只负责模型消息和请求级 StructuredTool 编排；工具治理仍由
ToolRegistry 与 ToolRuntime 完整负责。
"""

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.agent import route_agent_action
from app.agent.answer import apply_rag_result_to_state, compose_final_answer
from app.agent.executor import (
    build_observation,
    build_tool_trace_step,
    execute_plan,
    summarize_tool_result,
)
from app.agent.model import get_agent_chat_model
from app.agent.planner import build_plan_from_decision
from app.agent.runtime import (
    append_llm_usage_trace,
    build_decide_trace,
    build_plan_trace,
    build_v2_result,
    maybe_save_agent_trace,
    normalize_llm_call_count,
)
from app.agent.state import AgentPlanStep, AgentState, get_effective_question
from app.config import Settings
from app.graph.state import AgentGraphState
from app.rag.graph import run_rag_subgraph
from app.tools.errors import ToolErrorCode
from app.tools.langchain_tools import (
    get_langchain_tool_map,
    get_langchain_tools,
)
from app.tools.redaction import redact_sensitive_data
from app.tools.result import tool_error
from app.tools.context import ToolInvocationContext


def _safe_decision(decision: object) -> dict[str, Any]:
    """把 Graph 中的路由决策规整为现有业务函数可读取的字典。"""
    return decision if isinstance(decision, dict) else {}


def _require_agent_state(state: AgentGraphState) -> AgentState:
    """读取已经初始化的 AgentState，缺失时抛出明确编排错误。"""
    agent_state = state.get("agent_state")
    if not isinstance(agent_state, AgentState):
        raise RuntimeError("LangGraph agent_state is not initialized")
    return agent_state


def _resolve_state_questions(state: AgentGraphState) -> tuple[str, str]:
    """读取并清理 Graph 中的原始问题与执行问题。"""
    return (
        (state.get("original_question") or "").strip(),
        (state.get("effective_question") or "").strip(),
    )


def route_node(state: AgentGraphState) -> dict[str, Any]:
    """调用现有 Router，并把决策写回 Graph State。"""
    _, effective_question = _resolve_state_questions(state)
    decision = route_agent_action(
        effective_question,
        state.get("top_k", Settings.TOP_K),
        trace_id=state.get("trace_id", ""),
    )
    return {"decision": decision}


def initialize_agent_state_node(state: AgentGraphState) -> dict[str, Any]:
    """按现有业务协议创建 AgentState。"""
    original_question, effective_question = _resolve_state_questions(state)
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
        original_question=original_question,
        rewritten_question=(
            effective_question
            if effective_question != original_question
            else ""
        ),
        top_k=state.get("top_k", Settings.TOP_K),
        decompose_top_k=state.get(
            "decompose_top_k",
            Settings.DECOMPOSE_TOP_K,
        ),
        max_steps=Settings.AGENT_MAX_STEPS,
        router_llm_used=router_llm_used,
    )
    agent_state.tool_trace.append(build_decide_trace(decision, router_llm_used))
    return {
        "original_question": original_question,
        "effective_question": effective_question,
        "messages": [HumanMessage(content=effective_question)],
        "iteration_count": 0,
        "max_iterations": Settings.AGENT_MAX_ITERATIONS,
        "tool_call_count": 0,
        "max_tool_calls": Settings.AGENT_MAX_TOOL_CALLS,
        "tool_error_count": 0,
        "stop_reason": "",
        "fallback_to_legacy": False,
        "agent_state": agent_state,
    }


def _last_ai_message(state: AgentGraphState) -> AIMessage | None:
    """读取当前消息轨迹中最后一条 AIMessage。"""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _normalize_ai_content(content: object) -> str:
    """把不同供应商的文本块安全规整为最终回答字符串。"""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content or "").strip()

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def _agent_model_failure_updates(
    state: AgentGraphState,
    agent_state: AgentState,
    *,
    iteration_index: int,
    reason: str,
    error_type: str = "",
) -> dict[str, Any]:
    """首轮模型失败降级旧计划，工具后的失败则受控结束。"""
    trace: dict[str, Any] = {
        "step": "agent_iteration",
        "iteration_index": iteration_index,
        "status": "fallback",
        "reason": reason,
    }
    if error_type:
        trace["error_type"] = error_type
    agent_state.tool_trace.append(trace)

    if not agent_state.observations and not state.get("iteration_count", 0):
        return {
            "agent_state": agent_state,
            "fallback_to_legacy": True,
        }
    agent_state.status = "model_error"
    agent_state.reason = "Agent model failed after tool execution"
    agent_state.final_answer = "工具已执行，但模型暂时无法继续生成可靠回答。"
    return {
        "agent_state": agent_state,
        "fallback_to_legacy": False,
        "stop_reason": "agent_model_call_failed",
    }


def agent_model_node(
    state: AgentGraphState,
    runtime: Runtime[ToolInvocationContext],
) -> dict[str, Any]:
    """绑定请求级工具调用模型，并把一轮 AIMessage 追加到轨迹。"""
    agent_state = _require_agent_state(state)
    iteration_index = max(0, int(state.get("iteration_count", 0) or 0)) + 1
    try:
        model = get_agent_chat_model()
    except Exception as exc:
        return _agent_model_failure_updates(
            state,
            agent_state,
            iteration_index=iteration_index,
            reason="agent_model_initialization_failed",
            error_type=type(exc).__name__,
        )
    if model is None:
        return _agent_model_failure_updates(
            state,
            agent_state,
            iteration_index=iteration_index,
            reason="agent_model_unavailable",
        )

    try:
        tools = get_langchain_tools(context=runtime.context)
        bound_model = model.bind_tools(tools)
        response = bound_model.invoke(list(state.get("messages") or []))
        if not isinstance(response, AIMessage):
            raise TypeError("Agent model must return an AIMessage")
    except Exception as exc:
        return _agent_model_failure_updates(
            state,
            agent_state,
            iteration_index=iteration_index,
            reason="agent_model_call_failed",
            error_type=type(exc).__name__,
        )

    agent_state.answer_llm_used = True
    agent_state.llm_call_count += 1
    has_tool_calls = bool(response.tool_calls)
    iteration_trace = {
        "step": "agent_iteration",
        "iteration_index": iteration_index,
        "status": "success",
        "model_has_tool_calls": has_tool_calls,
        "requested_tool_call_count": len(response.tool_calls),
        "tool_call_count": int(state.get("tool_call_count", 0) or 0),
        "max_iterations": int(state.get("max_iterations", 0) or 0),
        "max_tool_calls": int(state.get("max_tool_calls", 0) or 0),
        "stop_reason": "" if has_tool_calls else "completed",
    }
    agent_state.tool_trace.append(iteration_trace)
    updates: dict[str, Any] = {
        "messages": [response],
        "iteration_count": iteration_index,
        "fallback_to_legacy": False,
        "agent_state": agent_state,
    }
    if not has_tool_calls:
        answer = _normalize_ai_content(response.content)
        agent_state.final_answer = answer or "模型未生成可用的最终回答。"
        agent_state.references = []
        agent_state.status = "ok" if answer else "model_error"
        agent_state.reason = (
            "Agent model produced final answer"
            if answer
            else "Agent model returned empty final answer"
        )
        updates["stop_reason"] = "completed"
    return updates


def _invoke_tool_call(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_map: dict[str, BaseTool],
) -> dict[str, Any]:
    """调用请求级 StructuredTool；适配层异常转换为统一工具结果。"""
    tool = tool_map.get(tool_name)
    if tool is None:
        return tool_error(
            tool_name,
            tool_args,
            f"unknown tool: {tool_name}",
            error_code=ToolErrorCode.UNKNOWN_TOOL.value,
        )
    try:
        result = tool.invoke(tool_args)
    except ValidationError:
        return tool_error(
            tool_name,
            tool_args,
            "StructuredTool rejected invalid arguments",
            error_code=ToolErrorCode.INVALID_ARGUMENTS.value,
        )
    except Exception as exc:
        return tool_error(
            tool_name,
            tool_args,
            f"StructuredTool invocation failed ({type(exc).__name__})",
            error_code=ToolErrorCode.EXECUTION_ERROR.value,
        )
    if not isinstance(result, dict):
        return tool_error(
            tool_name,
            tool_args,
            "LangChain tool returned a non-dict result",
            error_code=ToolErrorCode.INVALID_RESULT.value,
        )
    return result


def _tool_message_content(tool_name: str, result: dict[str, Any]) -> str:
    """保留统一结果关键字段，递归脱敏后序列化为 ToolMessage。"""
    error_code = str(result.get("error_code") or "")
    safe_policy_error_codes = {
        ToolErrorCode.PERMISSION_DENIED.value,
        ToolErrorCode.POLICY_DENIED.value,
        ToolErrorCode.AUDIT_WRITE_FAILED.value,
        ToolErrorCode.INVALID_INVOCATION_CONTEXT.value,
        ToolErrorCode.APPROVAL_REQUIRED.value,
        ToolErrorCode.MISSING_IDEMPOTENCY_KEY.value,
    }
    if error_code and error_code not in safe_policy_error_codes:
        safe_error: object = f"工具执行失败（{error_code}）"
    else:
        safe_error = result.get("error") or ""
    payload = {
        "ok": bool(result.get("ok", result.get("status") == "success")),
        "status": str(result.get("status") or ""),
        "tool_name": str(result.get("tool_name") or tool_name),
        "output": redact_sensitive_data(
            result.get("output"),
            max_serialized_chars=3000,
        ),
        "summary": redact_sensitive_data(
            result.get("summary")
            or summarize_tool_result(tool_name, result),
            max_serialized_chars=1000,
        ),
        "error": redact_sensitive_data(
            safe_error,
            max_serialized_chars=1000,
        ),
        "error_code": error_code,
        "retryable": bool(result.get("retryable", False)),
        "runtime": redact_sensitive_data(
            result.get("runtime") or {},
            max_serialized_chars=1500,
        ),
        "governance": redact_sensitive_data(
            result.get("governance") or {},
            max_serialized_chars=1500,
        ),
        "tool_call_id": str(result.get("tool_call_id") or ""),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def governed_tools_node(
    state: AgentGraphState,
    runtime: Runtime[ToolInvocationContext],
) -> dict[str, Any]:
    """执行最新一轮全部 tool_calls，并生成一一关联的 ToolMessage。

    官方 ToolNode 在构图时固定工具对象，无法安全持有本项目请求级 Context；
    因此这里动态创建 StructuredTool，但不复制 ToolRuntime 的任何治理能力。
    """
    agent_state = _require_agent_state(state)
    message = _last_ai_message(state)
    if message is None or not message.tool_calls:
        return {"agent_state": agent_state, "messages": []}

    tool_map = get_langchain_tool_map(context=runtime.context)
    tool_messages: list[ToolMessage] = []
    tool_error_count = max(0, int(state.get("tool_error_count", 0) or 0))
    previous_call_count = max(0, int(state.get("tool_call_count", 0) or 0))
    fatal_error_codes = {
        ToolErrorCode.PERMISSION_DENIED.value,
        ToolErrorCode.POLICY_DENIED.value,
        ToolErrorCode.AUDIT_WRITE_FAILED.value,
        ToolErrorCode.INVALID_INVOCATION_CONTEXT.value,
    }
    fatal_error = False

    for offset, call in enumerate(message.tool_calls, start=1):
        tool_name = str(call.get("name") or "").strip()
        raw_args = call.get("args")
        tool_args = raw_args if isinstance(raw_args, dict) else {}
        model_tool_call_id = str(call.get("id") or f"tool-call-{offset}")
        start = time.perf_counter()
        result = _invoke_tool_call(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_map=tool_map,
        )
        cost_ms = max(0, int((time.perf_counter() - start) * 1000))
        step = AgentPlanStep(
            step_id=len(agent_state.observations) + 1,
            action="tool_call",
            tool_name=tool_name,
            tool_args=tool_args,
            reason="Agent model requested tool call",
        )
        observation = build_observation(step, result, cost_ms)
        step.status = "success" if observation.success else "failed"
        agent_state.observations.append(observation)
        agent_state.tool_trace.append(
            build_tool_trace_step(step, result, observation)
        )
        if not observation.success:
            tool_error_count += 1
            if str(result.get("error_code") or "") in fatal_error_codes:
                fatal_error = True
        tool_messages.append(
            ToolMessage(
                content=_tool_message_content(tool_name, result),
                tool_call_id=model_tool_call_id,
                name=tool_name or None,
                status="success" if observation.success else "error",
            )
        )

    updates: dict[str, Any] = {
        "messages": tool_messages,
        "tool_call_count": previous_call_count + len(message.tool_calls),
        "tool_error_count": tool_error_count,
        "agent_state": agent_state,
    }
    if fatal_error:
        agent_state.status = "tool_error"
        agent_state.reason = "Tool governance denied further execution"
        agent_state.final_answer = "工具调用被安全策略拒绝，无法继续执行本次请求。"
        updates["stop_reason"] = "fatal_tool_error"
    return updates


def limit_handler_node(state: AgentGraphState) -> dict[str, Any]:
    """把执行预算耗尽规整为可正常 finalize 的受控模型错误。"""
    agent_state = _require_agent_state(state)
    iteration_count = int(state.get("iteration_count", 0) or 0)
    tool_call_count = int(state.get("tool_call_count", 0) or 0)
    max_iterations = int(state.get("max_iterations", 0) or 0)
    max_tool_calls = int(state.get("max_tool_calls", 0) or 0)
    reason = (
        "max_iterations_reached"
        if iteration_count >= max_iterations
        else "max_tool_calls_reached"
    )
    agent_state.status = "model_error"
    agent_state.reason = reason
    agent_state.final_answer = "本次请求达到 Agent 执行上限，已安全停止。请缩小问题范围后重试。"
    agent_state.tool_trace.append(
        {
            "step": "agent_limit",
            "status": "stopped",
            "stop_reason": reason,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
            "tool_call_count": tool_call_count,
            "max_tool_calls": max_tool_calls,
        }
    )
    return {"agent_state": agent_state, "stop_reason": reason}


def plan_node(state: AgentGraphState) -> dict[str, AgentState]:
    """调用现有 Planner，并追加共享的 plan trace。"""
    agent_state = _require_agent_state(state)
    agent_state.plan = build_plan_from_decision(
        decision=_safe_decision(state.get("decision")),
        question=get_effective_question(agent_state),
        top_k=state.get("top_k", Settings.TOP_K),
    )
    agent_state.tool_trace.append(build_plan_trace(agent_state))
    return {"agent_state": agent_state}


def execute_node(
    state: AgentGraphState,
    runtime: Runtime[ToolInvocationContext],
) -> dict[str, AgentState]:
    """调用现有 Executor，并显式返回其更新后的 AgentState。"""
    agent_state = execute_plan(
        _require_agent_state(state),
        context=runtime.context,
    )
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

    rag_result = run_rag_subgraph(
        question=get_effective_question(agent_state),
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
    "agent_model_node",
    "compose_node",
    "execute_node",
    "finalize_node",
    "governed_tools_node",
    "initialize_agent_state_node",
    "limit_handler_node",
    "plan_node",
    "rag_subgraph_node",
    "route_node",
]

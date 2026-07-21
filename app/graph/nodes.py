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
from app.agent.guardrails import (
    AgentGuardrailCode,
    AgentStopReason,
    AgentToolFailureKind,
    advance_consecutive_fingerprint,
    build_agent_tool_call_fingerprint,
    build_agent_tool_error_fingerprint,
    classify_tool_failure,
    is_agent_elapsed_budget_exhausted,
    stop_reason_for_tool_failure,
)
from app.agent.executor import (
    build_observation,
    build_tool_trace_step,
    execute_plan,
    summarize_tool_result,
)
from app.agent.model import get_agent_chat_model, get_agent_tool_provider
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
from app.tools.audit import sanitize_tool_input
from app.tools.contracts import ToolSpec
from app.tools.redaction import redact_sensitive_data
from app.tools.result import tool_error
from app.tools.context import ToolInvocationContext


_ERROR_SOURCE_TOOL_RUNTIME = "tool_runtime"
_ERROR_SOURCE_STRUCTURED_VALIDATION = "structured_tool_validation"
_ERROR_SOURCE_STRUCTURED_ADAPTER = "structured_tool_adapter"
_ERROR_SOURCE_AGENT_GUARDRAIL = "agent_guardrail"


def _agent_perf_counter() -> float:
    """返回 Agent 节点使用的单调时钟；测试可注入确定性时钟。"""
    return time.perf_counter()


def _elapsed_ms(started_at: float) -> int:
    """把单调时钟差规整为非负整数毫秒。"""
    return max(0, int((_agent_perf_counter() - started_at) * 1000))


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
        "max_tool_errors": Settings.AGENT_MAX_TOOL_ERRORS,
        "last_tool_call_fingerprint": "",
        "consecutive_same_tool_call_count": 0,
        "max_consecutive_same_tool_calls": (
            Settings.AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS
        ),
        "last_tool_error_fingerprint": "",
        "consecutive_same_tool_error_count": 0,
        "max_consecutive_same_tool_errors": (
            Settings.AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS
        ),
        "agent_elapsed_ms": 0,
        "max_agent_elapsed_ms": Settings.AGENT_MAX_TOTAL_COST_MS,
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
    model_cost_ms: int = 0,
) -> dict[str, Any]:
    """首轮模型失败降级旧计划，工具后的失败则受控结束。"""
    trace: dict[str, Any] = {
        "step": "agent_iteration",
        "iteration_index": iteration_index,
        "status": "fallback",
        "reason": reason,
        "model_cost_ms": model_cost_ms,
        "agent_elapsed_ms": (
            max(0, int(state.get("agent_elapsed_ms", 0) or 0))
            + model_cost_ms
        ),
    }
    if error_type:
        trace["error_type"] = error_type
    agent_state.tool_trace.append(trace)

    decision = _safe_decision(state.get("decision"))
    if (
        not agent_state.observations
        and not state.get("iteration_count", 0)
        and decision.get("action") != "agent"
    ):
        return {
            "agent_state": agent_state,
            "fallback_to_legacy": True,
            "agent_elapsed_ms": trace["agent_elapsed_ms"],
        }
    agent_state.status = "model_error"
    agent_state.reason = "Agent model is unavailable for dynamic tool selection"
    agent_state.final_answer = "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
    return {
        "agent_state": agent_state,
        "fallback_to_legacy": False,
        "stop_reason": "agent_model_call_failed",
        "agent_elapsed_ms": trace["agent_elapsed_ms"],
    }


def _budget_snapshot(
    state: AgentGraphState,
    *,
    tool_call_count: int | None = None,
    tool_error_count: int | None = None,
    agent_elapsed_ms: int | None = None,
) -> dict[str, int]:
    """构造只包含数字的 Agent 预算快照。"""
    return {
        "iterations_used": max(0, int(state.get("iteration_count", 0) or 0)),
        "iterations_limit": max(1, int(state.get("max_iterations", 1) or 1)),
        "tool_calls_used": max(
            0,
            int(
                state.get("tool_call_count", 0)
                if tool_call_count is None
                else tool_call_count
            ),
        ),
        "tool_calls_limit": max(1, int(state.get("max_tool_calls", 1) or 1)),
        "tool_errors_used": max(
            0,
            int(
                state.get("tool_error_count", 0)
                if tool_error_count is None
                else tool_error_count
            ),
        ),
        "tool_errors_limit": max(1, int(state.get("max_tool_errors", 1) or 1)),
        "elapsed_ms": max(
            0,
            int(
                state.get("agent_elapsed_ms", 0)
                if agent_elapsed_ms is None
                else agent_elapsed_ms
            ),
        ),
        "elapsed_limit_ms": max(
            1,
            int(state.get("max_agent_elapsed_ms", 1) or 1),
        ),
    }


def _apply_stop_to_agent_state(
    agent_state: AgentState,
    stop_reason: str,
) -> None:
    """按兼容 API 状态映射设置预算或终止错误。"""
    tool_error_reasons = {
        AgentStopReason.FATAL_TOOL_ERROR.value,
        AgentStopReason.APPROVAL_REQUIRED.value,
        AgentStopReason.MISSING_IDEMPOTENCY_KEY.value,
        AgentStopReason.RUNTIME_UNAVAILABLE.value,
    }
    if stop_reason in tool_error_reasons:
        agent_state.status = "tool_error"
        agent_state.reason = stop_reason
        agent_state.final_answer = "工具调用被治理策略或运行时终止，无法继续本次请求。"
        return
    agent_state.status = "model_error"
    agent_state.reason = stop_reason
    agent_state.final_answer = "本次请求达到 Agent 执行上限，已安全停止。请缩小问题范围后重试。"


def agent_model_node(
    state: AgentGraphState,
    runtime: Runtime[ToolInvocationContext],
) -> dict[str, Any]:
    """绑定请求级工具调用模型，并把一轮 AIMessage 追加到轨迹。"""
    agent_state = _require_agent_state(state)
    iteration_index = max(0, int(state.get("iteration_count", 0) or 0)) + 1
    previous_elapsed_ms = max(0, int(state.get("agent_elapsed_ms", 0) or 0))
    max_elapsed_ms = max(1, int(state.get("max_agent_elapsed_ms", 1) or 1))
    if is_agent_elapsed_budget_exhausted(
        elapsed_ms=previous_elapsed_ms,
        limit_ms=max_elapsed_ms,
    ):
        reason = AgentStopReason.MAX_TOTAL_COST.value
        _apply_stop_to_agent_state(agent_state, reason)
        return {"agent_state": agent_state, "stop_reason": reason}

    model_started_at = _agent_perf_counter()
    try:
        model = get_agent_chat_model()
    except Exception as exc:
        return _agent_model_failure_updates(
            state,
            agent_state,
            iteration_index=iteration_index,
            reason="agent_model_initialization_failed",
            error_type=type(exc).__name__,
            model_cost_ms=_elapsed_ms(model_started_at),
        )
    if model is None:
        return _agent_model_failure_updates(
            state,
            agent_state,
            iteration_index=iteration_index,
            reason="agent_model_unavailable",
            model_cost_ms=_elapsed_ms(model_started_at),
        )

    try:
        provider = get_agent_tool_provider()
        tools = get_langchain_tools(
            context=runtime.context,
            provider=provider,
        )
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
            model_cost_ms=_elapsed_ms(model_started_at),
        )

    model_cost_ms = _elapsed_ms(model_started_at)
    agent_elapsed_ms = previous_elapsed_ms + model_cost_ms
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
        "model_cost_ms": model_cost_ms,
        "agent_elapsed_ms": agent_elapsed_ms,
        "max_agent_elapsed_ms": max_elapsed_ms,
        "stop_reason": "" if has_tool_calls else "completed",
        "available_tool_names": [tool.name for tool in tools[:50]],
        "budget_snapshot": {
            "iterations_used": iteration_index,
            "iterations_limit": max(
                1,
                int(state.get("max_iterations", 1) or 1),
            ),
            "tool_calls_used": max(
                0,
                int(state.get("tool_call_count", 0) or 0),
            ),
            "tool_calls_limit": max(
                1,
                int(state.get("max_tool_calls", 1) or 1),
            ),
            "tool_errors_used": max(
                0,
                int(state.get("tool_error_count", 0) or 0),
            ),
            "tool_errors_limit": max(
                1,
                int(state.get("max_tool_errors", 1) or 1),
            ),
            "elapsed_ms": agent_elapsed_ms,
            "elapsed_limit_ms": max_elapsed_ms,
        },
    }
    agent_state.tool_trace.append(iteration_trace)
    updates: dict[str, Any] = {
        "messages": [response],
        "iteration_count": iteration_index,
        "fallback_to_legacy": False,
        "agent_elapsed_ms": agent_elapsed_ms,
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
) -> tuple[dict[str, Any], str, bool]:
    """调用请求级 StructuredTool；适配层异常转换为统一工具结果。"""
    tool = tool_map.get(tool_name)
    if tool is None:
        return (
            tool_error(
                tool_name,
                tool_args,
                f"unknown tool: {tool_name}",
                error_code=ToolErrorCode.UNKNOWN_TOOL.value,
            ),
            _ERROR_SOURCE_STRUCTURED_ADAPTER,
            False,
        )
    try:
        result = tool.invoke(tool_args)
    except ValidationError as exc:
        return (
            tool_error(
                tool_name,
                tool_args,
                _safe_validation_error(exc),
                error_code=ToolErrorCode.INVALID_ARGUMENTS.value,
            ),
            _ERROR_SOURCE_STRUCTURED_VALIDATION,
            False,
        )
    except Exception as exc:
        return (
            tool_error(
                tool_name,
                tool_args,
                f"StructuredTool invocation failed ({type(exc).__name__})",
                error_code=ToolErrorCode.EXECUTION_ERROR.value,
            ),
            _ERROR_SOURCE_STRUCTURED_ADAPTER,
            False,
        )
    if not isinstance(result, dict):
        return (
            tool_error(
                tool_name,
                tool_args,
                "LangChain tool returned a non-dict result",
                error_code=ToolErrorCode.INVALID_RESULT.value,
            ),
            _ERROR_SOURCE_STRUCTURED_ADAPTER,
            False,
        )
    runtime_metadata = result.get("runtime")
    runtime_metadata = runtime_metadata if isinstance(runtime_metadata, dict) else {}
    handler_executed = max(
        0,
        int(runtime_metadata.get("attempt_count", 0) or 0),
    ) > 0
    return result, _ERROR_SOURCE_TOOL_RUNTIME, handler_executed


def _safe_validation_error(exc: ValidationError) -> str:
    """保留字段位置与错误类型，但不序列化原始 input 或 traceback。"""
    details: list[str] = []
    for item in exc.errors(include_input=False, include_url=False)[:10]:
        location = ".".join(str(part)[:80] for part in item.get("loc", ()))
        error_type = str(item.get("type") or "validation_error")[:80]
        details.append(f"{location or '<root>'}: {error_type}")
    detail = "; ".join(details)[:1000]
    return f"StructuredTool 参数校验失败: {detail or '参数不合法'}"


def _tool_suggestion(error_code: str) -> str:
    """按稳定错误码返回确定性恢复建议，不额外调用模型。"""
    suggestions = {
        ToolErrorCode.INVALID_ARGUMENTS.value: (
            "请根据工具参数 Schema 修正必填字段、字段名称和字段类型。"
        ),
        ToolErrorCode.UNKNOWN_TOOL.value: (
            "请从当前绑定的工具列表中重新选择，不要自行生成工具名称。"
        ),
        ToolErrorCode.TIMEOUT.value: (
            "不要无条件重复相同调用，可选择替代工具或结束任务。"
        ),
        ToolErrorCode.DEPENDENCY_UNAVAILABLE.value: (
            "不要无条件重复相同调用，可选择替代工具或结束任务。"
        ),
        ToolErrorCode.RATE_LIMITED.value: (
            "不要无条件重复相同调用，可选择替代工具或结束任务。"
        ),
        ToolErrorCode.EXECUTION_ERROR.value: (
            "可修改参数或选择其他工具，不要重复完全相同调用。"
        ),
        ToolErrorCode.APPROVAL_REQUIRED.value: (
            "该调用需要人工批准，本次流程不能继续执行。"
        ),
    }
    return suggestions.get(str(error_code or ""), "")


def _normalize_tool_call(
    call: object,
    *,
    offset: int,
) -> tuple[str, object, dict[str, Any], str]:
    """规整模型工具调用，同时保留非法原始参数供稳定指纹使用。"""
    normalized = call if isinstance(call, dict) else {}
    tool_name = str(normalized.get("name") or "").strip()
    raw_args = normalized.get("args")
    tool_args = raw_args if isinstance(raw_args, dict) else {}
    model_tool_call_id = str(
        normalized.get("id") or f"tool-call-{offset}"
    )
    return tool_name, raw_args, tool_args, model_tool_call_id


def _enrich_tool_result(
    result: dict[str, Any],
    *,
    model_tool_call_id: str,
    call_fingerprint: str,
    failure_kind: str,
    error_source: str,
    handler_executed: bool,
    stop_reason: str,
) -> dict[str, Any]:
    """补齐 Agent ToolMessage 字段，并保留 Runtime 内部调用 ID。"""
    enriched = dict(result)
    runtime_tool_call_id = str(enriched.get("tool_call_id") or "")
    if runtime_tool_call_id and runtime_tool_call_id != model_tool_call_id:
        enriched["runtime_tool_call_id"] = runtime_tool_call_id
    enriched.update(
        {
            "tool_call_id": model_tool_call_id,
            "failure_kind": failure_kind,
            "error_source": error_source,
            "handler_executed": bool(handler_executed),
            "suggestion": _tool_suggestion(
                str(enriched.get("error_code") or "")
            ),
            "call_fingerprint": call_fingerprint,
            "stop_reason": stop_reason,
        }
    )
    return enriched


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
    error_source = str(result.get("error_source") or "")
    if error_source == _ERROR_SOURCE_STRUCTURED_VALIDATION:
        safe_error = result.get("error") or ""
    elif error_code and error_code not in safe_policy_error_codes:
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
        "failure_kind": str(result.get("failure_kind") or ""),
        "error_source": str(result.get("error_source") or ""),
        "handler_executed": bool(result.get("handler_executed", False)),
        "suggestion": str(result.get("suggestion") or ""),
        "call_fingerprint": str(result.get("call_fingerprint") or ""),
        "stop_reason": str(result.get("stop_reason") or ""),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _guardrail_result(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    model_tool_call_id: str,
    call_fingerprint: str,
    stop_reason: str,
    error_code: str,
) -> dict[str, Any]:
    """构造从未进入 StructuredTool/Runtime 的标准 Agent 错误结果。"""
    result = tool_error(
        tool_name,
        tool_args,
        f"Agent guardrail stopped tool execution ({stop_reason})",
        error_code=error_code,
    )
    return _enrich_tool_result(
        result,
        model_tool_call_id=model_tool_call_id,
        call_fingerprint=call_fingerprint,
        failure_kind="",
        error_source=_ERROR_SOURCE_AGENT_GUARDRAIL,
        handler_executed=False,
        stop_reason=stop_reason,
    )


def _build_agent_tool_trace(
    *,
    state: AgentGraphState,
    step: AgentPlanStep,
    result: dict[str, Any],
    observation,
    failure_kind: str,
    error_source: str,
    handler_executed: bool,
    will_return_to_agent_model: bool,
    stop_reason: str,
    consecutive_call_count: int,
    consecutive_error_count: int,
    tool_call_count: int,
    tool_error_count: int,
    agent_elapsed_ms: int,
    tool_spec: ToolSpec | None,
) -> dict[str, Any]:
    """在现有工具 trace 上增加 Agent 恢复与预算信息。"""
    trace = build_tool_trace_step(
        step,
        result,
        observation,
        tool_spec=tool_spec,
    )
    trace.update(
        {
            "failure_kind": failure_kind,
            "error_source": error_source,
            "handler_executed": bool(handler_executed),
            "will_return_to_agent_model": bool(will_return_to_agent_model),
            "stop_reason": stop_reason,
            "call_fingerprint": str(result.get("call_fingerprint") or ""),
            "consecutive_same_tool_call_count": consecutive_call_count,
            "consecutive_same_tool_error_count": consecutive_error_count,
            "tool_error_count": tool_error_count,
            "budget_snapshot": _budget_snapshot(
                state,
                tool_call_count=tool_call_count,
                tool_error_count=tool_error_count,
                agent_elapsed_ms=agent_elapsed_ms,
            ),
        }
    )
    return trace


def _build_guardrail_trace(
    *,
    state: AgentGraphState,
    tool_name: str,
    tool_args: dict[str, Any],
    result: dict[str, Any],
    consecutive_call_count: int,
    consecutive_error_count: int,
    tool_call_count: int,
    tool_error_count: int,
    agent_elapsed_ms: int,
    tool_spec: ToolSpec | None = None,
) -> dict[str, Any]:
    """为未执行调用构造不含原始敏感参数的 trace。"""
    safe_args = (
        sanitize_tool_input(tool_spec, tool_args)
        if tool_spec is not None
        else redact_sensitive_data(
            tool_args,
            sensitive_keys=tuple(tool_args),
            max_serialized_chars=1000,
        )
    )
    return {
        "step": "tool_call",
        "action": "tool_call",
        "tool_name": tool_name,
        "tool_args": safe_args,
        "sanitized_tool_args": safe_args,
        "tool_status": "error",
        "success": False,
        "error": "Agent guardrail stopped tool execution",
        "error_code": str(result.get("error_code") or ""),
        "tool_call_id": str(result.get("tool_call_id") or ""),
        "cost_ms": 0,
        "attempt_count": 0,
        "max_attempts": 0,
        "timed_out": False,
        "risk_level": "",
        "policy_decision": "",
        "args_fingerprint": "",
        "call_fingerprint": str(result.get("call_fingerprint") or ""),
        "failure_kind": "",
        "error_source": _ERROR_SOURCE_AGENT_GUARDRAIL,
        "handler_executed": False,
        "will_return_to_agent_model": False,
        "stop_reason": str(result.get("stop_reason") or ""),
        "consecutive_same_tool_call_count": consecutive_call_count,
        "consecutive_same_tool_error_count": consecutive_error_count,
        "tool_error_count": tool_error_count,
        "budget_snapshot": _budget_snapshot(
            state,
            tool_call_count=tool_call_count,
            tool_error_count=tool_error_count,
            agent_elapsed_ms=agent_elapsed_ms,
        ),
    }


def _failure_stop_reason(
    *,
    error_code: str,
    failure_kind: AgentToolFailureKind,
    tool_error_count: int,
    max_tool_errors: int,
    consecutive_error_count: int,
    max_consecutive_errors: int,
    agent_elapsed_ms: int,
    max_agent_elapsed_ms: int,
) -> str:
    """按治理、总错误、重复错误、耗时的固定优先级选择停止原因。"""
    immediate = stop_reason_for_tool_failure(
        error_code=error_code,
        failure_kind=failure_kind,
    )
    if immediate is not None:
        return immediate.value
    if tool_error_count >= max_tool_errors:
        return AgentStopReason.MAX_TOOL_ERRORS.value
    if consecutive_error_count >= max_consecutive_errors:
        return AgentStopReason.REPEATED_TOOL_ERROR.value
    if is_agent_elapsed_budget_exhausted(
        elapsed_ms=agent_elapsed_ms,
        limit_ms=max_agent_elapsed_ms,
    ):
        return AgentStopReason.MAX_TOTAL_COST.value
    return ""


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

    provider = get_agent_tool_provider()
    tool_map = get_langchain_tool_map(
        context=runtime.context,
        provider=provider,
    )
    provider_registry = getattr(provider, "registry", None)
    tool_messages: list[ToolMessage] = []
    tool_error_count = max(0, int(state.get("tool_error_count", 0) or 0))
    max_tool_errors = max(1, int(state.get("max_tool_errors", 1) or 1))
    previous_call_count = max(0, int(state.get("tool_call_count", 0) or 0))
    max_tool_calls = max(1, int(state.get("max_tool_calls", 1) or 1))
    executed_call_count = 0
    last_call_fingerprint = str(state.get("last_tool_call_fingerprint") or "")
    consecutive_call_count = max(
        0,
        int(state.get("consecutive_same_tool_call_count", 0) or 0),
    )
    max_consecutive_calls = max(
        1,
        int(state.get("max_consecutive_same_tool_calls", 1) or 1),
    )
    last_error_fingerprint = str(state.get("last_tool_error_fingerprint") or "")
    consecutive_error_count = max(
        0,
        int(state.get("consecutive_same_tool_error_count", 0) or 0),
    )
    max_consecutive_errors = max(
        1,
        int(state.get("max_consecutive_same_tool_errors", 1) or 1),
    )
    agent_elapsed_ms = max(0, int(state.get("agent_elapsed_ms", 0) or 0))
    max_agent_elapsed_ms = max(
        1,
        int(state.get("max_agent_elapsed_ms", 1) or 1),
    )
    stop_reason = str(state.get("stop_reason") or "")
    batch_trace_indexes: list[int] = []

    for offset, call in enumerate(message.tool_calls, start=1):
        tool_name, raw_args, tool_args, model_tool_call_id = _normalize_tool_call(
            call,
            offset=offset,
        )
        call_fingerprint = build_agent_tool_call_fingerprint(
            tool_name=tool_name,
            tool_args=raw_args,
        )
        tool_spec = (
            provider_registry.get(tool_name)
            if provider_registry is not None
            and callable(getattr(provider_registry, "get", None))
            else None
        )
        if not isinstance(tool_spec, ToolSpec):
            tool_spec = None
        if stop_reason:
            result = _guardrail_result(
                tool_name=tool_name,
                tool_args=tool_args,
                model_tool_call_id=model_tool_call_id,
                call_fingerprint=call_fingerprint,
                stop_reason=stop_reason,
                error_code=AgentGuardrailCode.AGENT_BATCH_ABORTED.value,
            )
            tool_messages.append(
                ToolMessage(
                    content=_tool_message_content(tool_name, result),
                    tool_call_id=model_tool_call_id,
                    name=tool_name or None,
                    status="error",
                )
            )
            agent_state.tool_trace.append(
                _build_guardrail_trace(
                    state=state,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result,
                    consecutive_call_count=consecutive_call_count,
                    consecutive_error_count=consecutive_error_count,
                    tool_call_count=previous_call_count + executed_call_count,
                    tool_error_count=tool_error_count,
                    agent_elapsed_ms=agent_elapsed_ms,
                    tool_spec=tool_spec,
                )
            )
            batch_trace_indexes.append(len(agent_state.tool_trace) - 1)
            continue

        last_call_fingerprint, consecutive_call_count = (
            advance_consecutive_fingerprint(
                previous_fingerprint=last_call_fingerprint,
                previous_count=consecutive_call_count,
                current_fingerprint=call_fingerprint,
            )
        )
        if previous_call_count + executed_call_count >= max_tool_calls:
            stop_reason = AgentStopReason.MAX_TOOL_CALLS.value
        elif is_agent_elapsed_budget_exhausted(
            elapsed_ms=agent_elapsed_ms,
            limit_ms=max_agent_elapsed_ms,
        ):
            stop_reason = AgentStopReason.MAX_TOTAL_COST.value
        elif consecutive_call_count > max_consecutive_calls:
            stop_reason = AgentStopReason.REPEATED_TOOL_CALL.value

        if stop_reason:
            _apply_stop_to_agent_state(agent_state, stop_reason)
            result = _guardrail_result(
                tool_name=tool_name,
                tool_args=tool_args,
                model_tool_call_id=model_tool_call_id,
                call_fingerprint=call_fingerprint,
                stop_reason=stop_reason,
                error_code=stop_reason,
            )
            tool_messages.append(
                ToolMessage(
                    content=_tool_message_content(tool_name, result),
                    tool_call_id=model_tool_call_id,
                    name=tool_name or None,
                    status="error",
                )
            )
            agent_state.tool_trace.append(
                _build_guardrail_trace(
                    state=state,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result,
                    consecutive_call_count=consecutive_call_count,
                    consecutive_error_count=consecutive_error_count,
                    tool_call_count=previous_call_count + executed_call_count,
                    tool_error_count=tool_error_count,
                    agent_elapsed_ms=agent_elapsed_ms,
                    tool_spec=tool_spec,
                )
            )
            batch_trace_indexes.append(len(agent_state.tool_trace) - 1)
            continue

        started_at = _agent_perf_counter()
        result, error_source, handler_executed = _invoke_tool_call(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_map=tool_map,
        )
        cost_ms = _elapsed_ms(started_at)
        agent_elapsed_ms += cost_ms
        executed_call_count += 1
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
        failure_kind = ""
        if not observation.success:
            tool_error_count += 1
            error_code = str(result.get("error_code") or "")
            classified_failure = classify_tool_failure(
                error_code=error_code,
                retryable=bool(result.get("retryable", False)),
            )
            failure_kind = classified_failure.value
            error_fingerprint = build_agent_tool_error_fingerprint(
                call_fingerprint=call_fingerprint,
                error_code=error_code,
            )
            last_error_fingerprint, consecutive_error_count = (
                advance_consecutive_fingerprint(
                    previous_fingerprint=last_error_fingerprint,
                    previous_count=consecutive_error_count,
                    current_fingerprint=error_fingerprint,
                )
            )
            stop_reason = _failure_stop_reason(
                error_code=error_code,
                failure_kind=classified_failure,
                tool_error_count=tool_error_count,
                max_tool_errors=max_tool_errors,
                consecutive_error_count=consecutive_error_count,
                max_consecutive_errors=max_consecutive_errors,
                agent_elapsed_ms=agent_elapsed_ms,
                max_agent_elapsed_ms=max_agent_elapsed_ms,
            )
        else:
            last_error_fingerprint = ""
            consecutive_error_count = 0
            if is_agent_elapsed_budget_exhausted(
                elapsed_ms=agent_elapsed_ms,
                limit_ms=max_agent_elapsed_ms,
            ):
                stop_reason = AgentStopReason.MAX_TOTAL_COST.value

        if stop_reason:
            _apply_stop_to_agent_state(agent_state, stop_reason)
        will_return_to_agent_model = (
            not stop_reason
            and not (observation.success and tool_name == "search_knowledge_base")
        )
        result = _enrich_tool_result(
            result,
            model_tool_call_id=model_tool_call_id,
            call_fingerprint=call_fingerprint,
            failure_kind=failure_kind,
            error_source=error_source,
            handler_executed=handler_executed,
            stop_reason=stop_reason,
        )
        tool_messages.append(
            ToolMessage(
                content=_tool_message_content(tool_name, result),
                tool_call_id=model_tool_call_id,
                name=tool_name or None,
                status="success" if observation.success else "error",
            )
        )
        agent_state.tool_trace.append(
            _build_agent_tool_trace(
                state=state,
                step=step,
                result=result,
                observation=observation,
                failure_kind=failure_kind,
                error_source=error_source,
                handler_executed=handler_executed,
                will_return_to_agent_model=will_return_to_agent_model,
                stop_reason=stop_reason,
                consecutive_call_count=consecutive_call_count,
                consecutive_error_count=consecutive_error_count,
                tool_call_count=previous_call_count + executed_call_count,
                tool_error_count=tool_error_count,
                agent_elapsed_ms=agent_elapsed_ms,
                tool_spec=tool_spec,
            )
        )
        batch_trace_indexes.append(len(agent_state.tool_trace) - 1)

    if stop_reason:
        for trace_index in batch_trace_indexes:
            agent_state.tool_trace[trace_index]["will_return_to_agent_model"] = False
    updates: dict[str, Any] = {
        "messages": tool_messages,
        "tool_call_count": previous_call_count + executed_call_count,
        "tool_error_count": tool_error_count,
        "last_tool_call_fingerprint": last_call_fingerprint,
        "consecutive_same_tool_call_count": consecutive_call_count,
        "last_tool_error_fingerprint": last_error_fingerprint,
        "consecutive_same_tool_error_count": consecutive_error_count,
        "agent_elapsed_ms": agent_elapsed_ms,
        "agent_state": agent_state,
    }
    if stop_reason:
        updates["stop_reason"] = stop_reason
    return updates


def limit_handler_node(state: AgentGraphState) -> dict[str, Any]:
    """规整预算错误，并为尚未执行的模型调用补齐 ToolMessage。"""
    agent_state = _require_agent_state(state)
    iteration_count = int(state.get("iteration_count", 0) or 0)
    tool_call_count = int(state.get("tool_call_count", 0) or 0)
    max_iterations = int(state.get("max_iterations", 0) or 0)
    max_tool_calls = int(state.get("max_tool_calls", 0) or 0)
    agent_elapsed_ms = max(0, int(state.get("agent_elapsed_ms", 0) or 0))
    max_agent_elapsed_ms = max(
        1,
        int(state.get("max_agent_elapsed_ms", 1) or 1),
    )
    if iteration_count >= max_iterations:
        reason = AgentStopReason.MAX_ITERATIONS.value
    elif tool_call_count >= max_tool_calls or (
        (_last_ai_message(state) is not None)
        and tool_call_count
        + len(_last_ai_message(state).tool_calls)
        > max_tool_calls
    ):
        reason = AgentStopReason.MAX_TOOL_CALLS.value
    elif is_agent_elapsed_budget_exhausted(
        elapsed_ms=agent_elapsed_ms,
        limit_ms=max_agent_elapsed_ms,
    ):
        reason = AgentStopReason.MAX_TOTAL_COST.value
    else:
        reason = AgentStopReason.MAX_TOOL_CALLS.value
    _apply_stop_to_agent_state(agent_state, reason)
    tool_messages: list[ToolMessage] = []
    message = _last_ai_message(state)
    if message is not None:
        for offset, call in enumerate(message.tool_calls, start=1):
            tool_name, raw_args, tool_args, model_tool_call_id = (
                _normalize_tool_call(call, offset=offset)
            )
            call_fingerprint = build_agent_tool_call_fingerprint(
                tool_name=tool_name,
                tool_args=raw_args,
            )
            result = _guardrail_result(
                tool_name=tool_name,
                tool_args=tool_args,
                model_tool_call_id=model_tool_call_id,
                call_fingerprint=call_fingerprint,
                stop_reason=reason,
                error_code=reason,
            )
            tool_messages.append(
                ToolMessage(
                    content=_tool_message_content(tool_name, result),
                    tool_call_id=model_tool_call_id,
                    name=tool_name or None,
                    status="error",
                )
            )
            agent_state.tool_trace.append(
                _build_guardrail_trace(
                    state=state,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result,
                    consecutive_call_count=max(
                        0,
                        int(
                            state.get("consecutive_same_tool_call_count", 0)
                            or 0
                        ),
                    ),
                    consecutive_error_count=max(
                        0,
                        int(
                            state.get("consecutive_same_tool_error_count", 0)
                            or 0
                        ),
                    ),
                    tool_call_count=tool_call_count,
                    tool_error_count=max(
                        0,
                        int(state.get("tool_error_count", 0) or 0),
                    ),
                    agent_elapsed_ms=agent_elapsed_ms,
                )
            )
    agent_state.tool_trace.append(
        {
            "step": "agent_limit",
            "status": "stopped",
            "stop_reason": reason,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
            "tool_call_count": tool_call_count,
            "max_tool_calls": max_tool_calls,
            "agent_elapsed_ms": agent_elapsed_ms,
            "max_agent_elapsed_ms": max_agent_elapsed_ms,
            "budget_snapshot": _budget_snapshot(state),
        }
    )
    return {
        "agent_state": agent_state,
        "messages": tool_messages,
        "stop_reason": reason,
    }


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

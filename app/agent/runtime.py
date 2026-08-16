"""唯一 Agent Runtime：确定性路由、``create_agent`` 执行与结果归一化。"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from pydantic import TypeAdapter, ValidationError

from app.agent.guardrails import (
    AgentStopReason,
    build_agent_tool_error_fingerprint,
    classify_tool_failure,
    stop_reason_for_tool_failure,
)
from app.agent.answer import (
    build_compose_answer_updates,
    build_rag_answer_updates,
)
from app.agent.results import (
    build_agent_result,
    build_internal_error_result,
    validate_agent_result_payload,
)
from app.agent.router import RouteDecision, route_agent_action
from app.agent.state import AgentObservation
from app.agent.tool_events import build_tool_observation, build_tool_trace
from app.common.logging import get_logger
from app.config import AppSettings, get_settings
from app.storage.traces import save_agent_trace
from app.tools.context import ToolInvocationContext, build_default_tool_context


logger = get_logger("chat_service")

_INTEGER_ADAPTER = TypeAdapter(int)


def sum_governance_event_costs(events: list[dict[str, Any]]) -> int:
    """汇总模型和工具治理事件记录的非负耗时。"""
    return sum(max(0, int(item.get("cost_ms", 0) or 0)) for item in events)


def count_consecutive_events(
    events: list[dict[str, Any]],
    *,
    key: str,
    value: str,
) -> int:
    """统计已计数工具事件尾部连续出现的指定字段值。"""
    count = 0
    for event in reversed(events):
        if event.get("kind") != "tool" or not event.get("counted_call"):
            continue
        if str(event.get(key) or "") != value:
            break
        count += 1
    return count


def build_tool_governance_event(
    *,
    tool_call: dict[str, Any],
    result: dict[str, Any],
    call_fingerprint: str,
    prior_events: list[dict[str, Any]],
    cost_ms: int,
    known_tool: bool,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """根据标准工具结果计算错误分类、预算和最小治理事件。"""
    resolved_settings = settings or get_settings()
    tool_args = tool_call.get("args") or {}
    ok = result.get("ok") is True
    error_code = str(result.get("error_code") or "")
    failure_kind = ""
    stop_reason = ""
    error_fingerprint = ""

    if not ok:
        failure = classify_tool_failure(
            error_code=error_code, retryable=result.get("retryable", False)
        )
        failure_kind = failure.value
        error_fingerprint = build_agent_tool_error_fingerprint(
            call_fingerprint=call_fingerprint, error_code=error_code
        )
        immediate_stop = stop_reason_for_tool_failure(
            error_code=error_code, failure_kind=failure
        )
        error_count = (
            sum(
                event.get("kind") == "tool"
                and event.get("counted_call")
                and not event.get("ok")
                for event in prior_events
            )
            + 1
        )
        repeated_errors = (
            count_consecutive_events(
                prior_events, key="error_fingerprint", value=error_fingerprint
            )
            + 1
        )
        if immediate_stop:
            stop_reason = immediate_stop.value
        elif error_count >= resolved_settings.agent_max_tool_errors:
            stop_reason = AgentStopReason.MAX_TOOL_ERRORS.value
        elif (
            repeated_errors
            >= resolved_settings.agent_max_consecutive_same_tool_errors
        ):
            stop_reason = AgentStopReason.REPEATED_TOOL_ERROR.value

    elapsed_ms = sum_governance_event_costs(prior_events) + max(0, cost_ms)
    if (
        not stop_reason
        and elapsed_ms >= resolved_settings.agent_max_total_cost_ms
    ):
        stop_reason = AgentStopReason.MAX_TOTAL_COST.value

    runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
    error_source = "tool_runtime"
    if not runtime:
        error_source = (
            "structured_tool_validation" if known_tool else "structured_tool_adapter"
        )

    return {
        "kind": "tool",
        "name": str(tool_call.get("name") or ""),
        "args": tool_args,
        "result": result,
        "ok": ok,
        "counted_call": True,
        "call_fingerprint": call_fingerprint,
        "error_fingerprint": error_fingerprint,
        "error_code": error_code,
        "failure_kind": failure_kind,
        "error_source": error_source,
        "handler_executed": int(runtime.get("attempt_count", 0) or 0) > 0,
        "stop_reason": stop_reason,
        "cost_ms": max(0, int(cost_ms)),
        "tool_call_id": str(tool_call.get("id") or ""),
    }


def get_route_decision_for_trace(decision: dict) -> str:
    """根据路由 action 生成 trace 使用的 route_decision。"""
    action = decision.get("action", "")

    if action in {"clarify", "direct_answer"}:
        return action

    if action:
        return action

    return "unknown"


def build_decide_trace(
    decision: dict,
    *,
    route_cost_ms: int = 0,
) -> dict[str, Any]:
    """将最小内部路由协议扩展为稳定的外部兼容 trace。"""
    return {
        "step": "decide",
        "route_decision": get_route_decision_for_trace(decision),
        "decision_type": decision.get("decision", ""),
        "action": decision.get("action", ""),
        "tool_name": "",
        "tool_args": {},
        "sanitized_tool_args": {},
        "reason": decision.get("reason", ""),
        "router_source": decision.get("router_source", ""),
        "router_confidence": decision.get("router_confidence"),
        "router_fallback_reason": "",
        "router_llm_used": False,
        "route_cost_ms": max(0, int(route_cost_ms or 0)),
    }


def resolve_llm_call_count(
    *,
    decompose_llm_used: bool,
    answer_llm_used: bool,
    current_count: int,
) -> int:
    """解析 Agent Model 与 RAG 回答侧已经累计的精确调用数。"""
    return max(
        int(bool(decompose_llm_used)) + int(bool(answer_llm_used)),
        max(0, int(current_count or 0)),
    )


def upsert_llm_usage_trace(
    tool_trace: list[dict[str, Any]],
    *,
    decompose_llm_used: bool,
    answer_llm_used: bool,
    llm_call_count: int,
) -> list[dict[str, Any]]:
    """返回包含唯一最新 LLM usage 项的新工具 trace。"""
    usage_trace = {
        "step": "llm_usage",
        "router_llm_used": False,
        "decompose_llm_used": bool(decompose_llm_used),
        "answer_llm_used": bool(answer_llm_used),
        "llm_call_count": max(0, int(llm_call_count or 0)),
    }
    updated: list[dict[str, Any]] = []
    usage_written = False
    for step in tool_trace:
        if not isinstance(step, dict):
            continue
        if step.get("step") == "llm_usage":
            if not usage_written:
                updated.append(usage_trace)
                usage_written = True
            continue
        updated.append(dict(step))
    if not usage_written:
        updated.append(usage_trace)
    return updated


def validate_runtime_counts(
    *,
    top_k: Any,
    decompose_top_k: Any,
) -> tuple[int, int]:
    """保留公开 Runtime 数字输入原有的边界约束。"""

    def validate(name: str, value: Any, lower: int, upper: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            normalized = _INTEGER_ADAPTER.validate_python(value)
        except ValidationError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if normalized < lower or normalized > upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
        return normalized

    return (
        validate("top_k", top_k, 1, 10),
        validate("decompose_top_k", decompose_top_k, 1, 20),
    )


def _default_bad_case(trace_id: str, question: str) -> dict:
    """构造 Agent Runtime 使用的 bad case 默认结构。"""
    return {
        "trace_id": trace_id,
        "question": question,
        "answer": "",
        "references": [],
        "route": "",
        "error_type": "",
        "note": "",
    }


def _write_internal_bad_case(
    write_bad_cases,
    bad_case: dict | None,
    trace_id: str,
    question: str,
    exc: Exception,
    *,
    runtime_route: str = "agent_runtime",
    runtime_label: str = "Agent Runtime",
) -> None:
    """Runtime 顶层异常时写入 bad case，写入失败不继续抛错。"""
    if not callable(write_bad_cases):
        return

    record = dict(bad_case or _default_bad_case(trace_id, question))
    record["answer"] = f"系统内部执行 {runtime_label} 时发生错误，无法生成可靠回答。"
    record["references"] = []
    record["route"] = runtime_route
    record["error_type"] = "internal_error"
    record["note"] = str(exc)

    try:
        write_bad_cases(record)
    except Exception:
        return


def maybe_save_agent_trace(
    *,
    trace_id: str,
    session_id: str | None,
    question: str,
    rewritten_question: str = "",
    final_answer: str = "",
    status: str = "",
    reason: str = "",
    tool_trace: list[dict] | None = None,
    total_cost_ms: int = 0,
) -> None:
    """按配置保存 Agent trace，持久化失败不影响主流程。"""
    if not get_settings().enable_agent_trace_store:
        return

    try:
        save_agent_trace(
            trace_id=trace_id,
            session_id=session_id,
            question=question,
            rewritten_question=rewritten_question,
            final_answer=final_answer,
            status=status,
            reason=reason,
            tool_trace=tool_trace or [],
            total_cost_ms=total_cost_ms,
        )
    except Exception as exc:
        logger.warning("保存 Agent trace 失败，忽略持久化错误: %s", exc)


def _model_trace(event: Mapping[str, Any], index: int) -> dict[str, Any]:
    """把一次模型治理事件转换为稳定迭代 trace。"""
    return {
        "step": "agent_iteration",
        "iteration_index": index,
        "status": "success" if event.get("ok") else "fallback",
        "model_has_tool_calls": bool(event.get("has_tool_calls")),
        "requested_tool_call_count": int(
            event.get("requested_tool_call_count", 0) or 0
        ),
        "model_cost_ms": max(0, int(event.get("cost_ms", 0) or 0)),
        "available_tool_names": list(event.get("available_tool_names", []))[:50],
        "error_type": str(event.get("error_type") or ""),
    }


def _guardrail_trace(event: Mapping[str, Any]) -> dict[str, Any]:
    """为未执行的治理拦截生成兼容 trace。"""
    raw_result = event.get("result")
    result = raw_result if isinstance(raw_result, dict) else {}
    return {
        "step": "tool_call",
        "action": "tool_call",
        "tool_name": str(event.get("name") or ""),
        "tool_status": "error",
        "success": False,
        "error": str(
            result.get("error") or "Agent governance stopped tool execution"
        ),
        "error_code": str(event.get("error_code") or ""),
        "tool_call_id": str(event.get("tool_call_id") or ""),
        "cost_ms": 0,
        "handler_executed": False,
        "will_return_to_agent_model": False,
        "stop_reason": str(event.get("stop_reason") or ""),
        "call_fingerprint": str(event.get("call_fingerprint") or ""),
    }


def _normalize_tool_event(
    event: Mapping[str, Any],
    *,
    step_id: int,
) -> tuple[AgentObservation, dict[str, Any]]:
    """把一次已执行工具事件归一化为 observation 与公开 trace。"""
    raw_result = event.get("result")
    result = raw_result if isinstance(raw_result, dict) else {}
    tool_name = str(event.get("name") or "")
    raw_args = event.get("args")
    tool_args = raw_args if isinstance(raw_args, dict) else {}
    observation = build_tool_observation(
        step_id=step_id,
        tool_name=tool_name,
        result=result,
        cost_ms=max(0, int(event.get("cost_ms", 0) or 0)),
    )
    trace = build_tool_trace(
        step_id=step_id,
        tool_name=tool_name,
        tool_args=tool_args,
        result=result,
        observation=observation,
    )
    trace.update(
        {
            "failure_kind": str(event.get("failure_kind") or ""),
            "handler_executed": bool(event.get("handler_executed")),
            "will_return_to_agent_model": not bool(event.get("stop_reason")),
            "stop_reason": str(event.get("stop_reason") or ""),
            "call_fingerprint": str(event.get("call_fingerprint") or ""),
        }
    )
    return observation, trace


def normalize_governance_events(
    events: list[dict[str, Any]],
) -> tuple[list[AgentObservation], list[dict[str, Any]]]:
    """在 Runtime 出口一次性归一化 append-only 治理事件。"""
    observations: list[AgentObservation] = []
    tool_trace: list[dict[str, Any]] = []
    model_index = 0
    for event in events:
        if event.get("kind") == "model":
            model_index += 1
            tool_trace.append(_model_trace(event, model_index))
            continue
        if event.get("counted_call"):
            observation, trace = _normalize_tool_event(
                event,
                step_id=len(observations) + 1,
            )
            observations.append(observation)
        else:
            trace = _guardrail_trace(event)
        tool_trace.append(trace)
    return observations, tool_trace


def framework_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """从治理事件计算兼容计数和连续指纹。"""
    calls = [
        item
        for item in events
        if item.get("kind") == "tool"
        and item.get("error_code") != "agent_batch_aborted"
    ]
    tools = [item for item in calls if item.get("counted_call")]
    errors = [item for item in tools if not item.get("ok")]

    def tail(key: str) -> tuple[str, int]:
        value = str(calls[-1].get(key) or "") if calls else ""
        count = 0
        for item in reversed(calls):
            if not value or str(item.get(key) or "") != value:
                break
            count += 1
        return value, count

    last_call, call_count = tail("call_fingerprint")
    last_error = str(errors[-1].get("error_fingerprint") or "") if errors else ""
    error_count = 0
    for item in reversed(tools):
        if item.get("ok") or str(item.get("error_fingerprint") or "") != last_error:
            break
        error_count += 1
    return {
        "iteration_count": sum(item.get("kind") == "model" for item in events),
        "tool_call_count": len(tools),
        "tool_error_count": len(errors),
        "last_tool_call_fingerprint": last_call,
        "consecutive_same_tool_call_count": call_count,
        "last_tool_error_fingerprint": last_error,
        "consecutive_same_tool_error_count": error_count,
    }


def _normalize_ai_content(content: object) -> str:
    """把模型支持的文本内容形式归一化为字符串。"""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content or "").strip()
    parts = [
        item if isinstance(item, str) else item.get("text", "")
        for item in content
        if isinstance(item, (str, dict))
    ]
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def _last_final_ai(messages: object) -> AIMessage | None:
    """查找最近一条不再请求工具的最终模型消息。"""
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message
    return None


def _build_stop_updates(reason: str) -> dict[str, str]:
    """把框架停止原因映射为 API 兼容状态。"""
    tool_errors = {
        AgentStopReason.FATAL_TOOL_ERROR.value,
        AgentStopReason.APPROVAL_REQUIRED.value,
        AgentStopReason.MISSING_IDEMPOTENCY_KEY.value,
        AgentStopReason.RUNTIME_UNAVAILABLE.value,
    }
    if reason in tool_errors:
        status = "tool_error"
        final_answer = "工具调用被治理策略或运行时终止，无法继续本次请求。"
    else:
        status = "model_error"
        final_answer = "本次请求达到 Agent 执行上限，已安全停止。请缩小问题范围后重试。"
    if reason == "agent_model_call_failed":
        final_answer = "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
    return {
        "status": status,
        "final_answer": final_answer,
        "reason": reason,
    }


@lru_cache(maxsize=4)
def _get_cached_framework_agent(settings: AppSettings, provider: Any):
    """缓存不依赖 MCP discovery 的本地 ``create_agent`` 结构。"""
    from app.agent.framework_agent import build_framework_agent
    from app.agent.framework_middleware import ProjectToolGovernanceMiddleware
    from app.agent.framework_tools import build_framework_tools

    governance = ProjectToolGovernanceMiddleware(settings=settings)
    context = build_default_tool_context(
        source="agent_runtime",
        settings=settings,
    )
    return build_framework_agent(
        settings,
        tools=build_framework_tools(
            provider,
            context=context,
            governance=governance,
        ),
        governance=governance,
    )


def get_compiled_agent_runtime(
    settings: AppSettings | None = None,
    *,
    tool_context: ToolInvocationContext | None = None,
):
    """返回与当前请求可见工具集合一致的 CompiledStateGraph。"""
    resolved_settings = settings or get_settings()
    from app.agent.framework_tools import get_agent_tool_provider

    provider = get_agent_tool_provider()
    if tool_context is None:
        return _get_cached_framework_agent(resolved_settings, provider)
    if (
        tool_context.principal_id
        == resolved_settings.tool_default_principal_id
        and tool_context.permissions
        == resolved_settings.tool_default_permissions
    ):
        from app.tools.mcp.provider import list_mcp_tools_for_context

        if not list_mcp_tools_for_context(tool_context):
            return _get_cached_framework_agent(resolved_settings, provider)
    from app.agent.framework_agent import build_framework_agent
    from app.agent.framework_tools import resolve_agent_tools
    from app.agent.framework_middleware import ProjectToolGovernanceMiddleware

    governance = ProjectToolGovernanceMiddleware(settings=resolved_settings)
    return build_framework_agent(
        resolved_settings,
        tools=resolve_agent_tools(tool_context, governance=governance),
        governance=governance,
    )


def reset_agent_runtime() -> None:
    """失效唯一 Agent Runtime 缓存，供配置切换和关闭流程使用。"""
    _get_cached_framework_agent.cache_clear()


@dataclass(frozen=True)
class AgentRuntimeInvocation:
    """一次直接 ``create_agent`` 调用的已解析输入。"""

    agent: Any | None
    agent_input: dict[str, Any]
    agent_config: dict[str, Any]
    tool_context: ToolInvocationContext
    decision: RouteDecision
    route_cost_ms: int
    original_question: str
    effective_question: str
    top_k: int
    decompose_top_k: int
    trace_id: str
    session_id: str | None


def prepare_agent_runtime_invocation(
    *,
    top_k: int,
    decompose_top_k: int,
    trace_id: str,
    session_id: str | None,
    original_question: str,
    effective_question: str,
    tool_context: ToolInvocationContext | None = None,
    settings: AppSettings | None = None,
) -> AgentRuntimeInvocation:
    """用普通 Python 完成参数校验、Router 与 Agent 调用准备。"""
    resolved_settings = settings or get_settings()
    resolved_top_k, resolved_decompose_top_k = validate_runtime_counts(
        top_k=top_k,
        decompose_top_k=decompose_top_k,
    )
    resolved_original = (original_question or "").strip()
    resolved_effective = (effective_question or "").strip()
    route_started = time.perf_counter()
    decision = route_agent_action(resolved_effective)
    route_cost_ms = max(0, int((time.perf_counter() - route_started) * 1000))
    if tool_context is None:
        resolved_context = build_default_tool_context(
            trace_id=trace_id,
            session_id=session_id or "",
            source="chat_api",
            effective_question=resolved_effective,
            top_k=resolved_top_k,
            decompose_top_k=resolved_decompose_top_k,
            settings=resolved_settings,
        )
    else:
        resolved_context = replace(
            tool_context,
            effective_question=resolved_effective,
            top_k=resolved_top_k,
            decompose_top_k=resolved_decompose_top_k,
        )
    action = decision.get("action")
    agent = (
        get_compiled_agent_runtime(
            resolved_settings,
            tool_context=resolved_context,
        )
        if action == "agent"
        else None
    )
    return AgentRuntimeInvocation(
        agent=agent,
        agent_input={"messages": [HumanMessage(content=resolved_effective)]},
        agent_config={
            "recursion_limit": max(
                25,
                resolved_settings.agent_max_iterations * 8 + 20,
            ),
            "max_concurrency": 1,
        },
        tool_context=resolved_context,
        decision=decision,
        route_cost_ms=route_cost_ms,
        original_question=resolved_original,
        effective_question=resolved_effective,
        top_k=resolved_top_k,
        decompose_top_k=resolved_decompose_top_k,
        trace_id=trace_id,
        session_id=session_id,
    )


def _base_runtime_state(invocation: AgentRuntimeInvocation) -> dict[str, Any]:
    """构造只服务于最终结果归一化的普通 Python 状态。"""
    return {
        "original_question": invocation.original_question,
        "effective_question": invocation.effective_question,
        "decision": dict(invocation.decision),
        "route_cost_ms": invocation.route_cost_ms,
        "observations": [],
        "final_answer": "",
        "references": [],
        "status": "running",
        "reason": "",
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
        "tool_trace": [
            build_decide_trace(
                dict(invocation.decision),
                route_cost_ms=invocation.route_cost_ms,
            )
        ],
        "stop_reason": "",
        **framework_metrics([]),
    }


def _finalize_runtime_result(
    invocation: AgentRuntimeInvocation,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """生成唯一 LLM 统计、Result 合同和可选持久化快照。"""
    llm_call_count = resolve_llm_call_count(
        decompose_llm_used=bool(state.get("decompose_llm_used")),
        answer_llm_used=bool(state.get("answer_llm_used")),
        current_count=int(state.get("llm_call_count", 0) or 0),
    )
    tool_trace = upsert_llm_usage_trace(
        list(state.get("tool_trace", [])),
        decompose_llm_used=bool(state.get("decompose_llm_used")),
        answer_llm_used=bool(state.get("answer_llm_used")),
        llm_call_count=llm_call_count,
    )
    snapshot = {
        **state,
        "llm_call_count": llm_call_count,
        "tool_trace": tool_trace,
    }
    result = build_agent_result(snapshot).model_dump(mode="json")
    maybe_save_agent_trace(
        trace_id=invocation.trace_id,
        session_id=invocation.session_id,
        question=invocation.original_question,
        rewritten_question=(
            invocation.effective_question
            if invocation.effective_question != invocation.original_question
            else ""
        ),
        final_answer=result.get("answer", ""),
        status=result.get("status", ""),
        reason=result.get("reason", ""),
        tool_trace=result.get("tool_trace", []),
    )
    return validate_agent_result_payload(result)


def finish_short_circuit(
    invocation: AgentRuntimeInvocation,
) -> dict[str, Any]:
    """完成 clarify/direct_answer 或未知 Router action 的确定性结果。"""
    state = _base_runtime_state(invocation)
    action = str(invocation.decision.get("action") or "").strip()
    if action == "clarify":
        clarification = (
            str(invocation.decision.get("clarification_question") or "").strip()
            or str(invocation.decision.get("reason") or "").strip()
            or "问题需要进一步澄清"
        )
        state.update(
            {
                "status": "clarify",
                "reason": clarification,
                "final_answer": clarification,
            }
        )
        state["tool_trace"].append(
            {
                "step": "clarify",
                "step_id": 1,
                "status": "success",
                "reason": clarification,
            }
        )
    elif action == "direct_answer":
        state.update(
            {
                "status": "direct_answer",
                "reason": str(
                    invocation.decision.get("reason") or "该问题可以直接回答"
                ),
            }
        )
        state["tool_trace"].append(
            {
                "step": "direct_answer",
                "step_id": 1,
                "status": "success",
                "reason": state["reason"],
            }
        )
    else:
        state.update(
            {
                "status": "error",
                "reason": f"Unsupported route action: {action[:100] or '<missing>'}",
            }
        )
    state.update(build_compose_answer_updates(state))
    return _finalize_runtime_result(invocation, state)


def finish_framework_agent(
    invocation: AgentRuntimeInvocation,
    framework_output: Mapping[str, Any],
) -> dict[str, Any]:
    """把 ``create_agent`` 最终状态一次性归一化为 AgentRunResult。"""
    state = _base_runtime_state(invocation)
    raw_events = framework_output.get("governance_events")
    events = (
        [dict(item) for item in raw_events if isinstance(item, Mapping)]
        if isinstance(raw_events, list)
        else []
    )
    observations, event_trace = normalize_governance_events(events)
    successful_models = sum(
        item.get("kind") == "model" and item.get("ok") for item in events
    )
    state.update(
        {
            **framework_metrics(events),
            "observations": observations,
            "tool_trace": [*state["tool_trace"], *event_trace],
            "answer_llm_used": successful_models > 0,
            "llm_call_count": successful_models,
            "stop_reason": str(framework_output.get("stop_reason") or ""),
        }
    )

    raw_knowledge_result = framework_output.get("knowledge_base_result")
    knowledge_result = (
        raw_knowledge_result
        if isinstance(raw_knowledge_result, Mapping)
        else {}
    )
    knowledge_output = knowledge_result.get("output")
    rag_result = (
        knowledge_output.get("rag_result")
        if isinstance(knowledge_output, Mapping)
        else None
    )
    if isinstance(rag_result, Mapping):
        state.update(build_rag_answer_updates(state, rag_result))
        return _finalize_runtime_result(invocation, state)

    stop_reason = str(state.get("stop_reason") or "")
    if stop_reason and stop_reason != AgentStopReason.COMPLETED.value:
        state.update(_build_stop_updates(stop_reason))
    else:
        message = _last_final_ai(framework_output.get("messages"))
        answer = _normalize_ai_content(message.content if message else "")
        state.update(
            {
                "final_answer": answer or "模型未生成可用的最终回答。",
                "references": [],
                "status": "ok" if answer else "model_error",
                "reason": (
                    "Agent model produced final answer"
                    if answer
                    else "Agent model returned empty final answer"
                ),
                "stop_reason": AgentStopReason.COMPLETED.value,
            }
        )
    state.update(build_compose_answer_updates(state))
    return _finalize_runtime_result(invocation, state)


def build_agent_runtime_error_result(
    *,
    write_bad_cases: Callable[[dict[str, Any]], None] | None,
    bad_case: dict[str, Any] | None,
    trace_id: str,
    session_id: str | None,
    original_question: str,
    effective_question: str,
    exc: Exception,
) -> dict[str, Any]:
    """把同步与流式 Runtime 异常转换成同一脱敏内部错误结果。"""
    _write_internal_bad_case(
        write_bad_cases=write_bad_cases,
        bad_case=bad_case,
        trace_id=trace_id,
        question=original_question,
        exc=exc,
        runtime_route="agent_runtime",
        runtime_label="Agent Runtime",
    )
    result = build_internal_error_result(
        exc,
        runtime_label="Agent Runtime",
        trace_step="agent_runtime_error",
    )
    maybe_save_agent_trace(
        trace_id=trace_id,
        session_id=session_id,
        question=original_question,
        rewritten_question=(
            effective_question
            if effective_question != original_question
            else ""
        ),
        final_answer=result.get("answer", ""),
        status=result.get("status", ""),
        reason=result.get("reason", ""),
        tool_trace=result.get("tool_trace", []),
    )
    return result


def run_agent(
    write_bad_cases: Callable[[dict[str, Any]], None] | None,
    top_k: int,
    decompose_top_k: int,
    trace_id: str = "",
    session_id: str | None = None,
    bad_case: dict[str, Any] | None = None,
    *,
    original_question: str,
    effective_question: str,
    tool_context: ToolInvocationContext | None = None,
) -> dict[str, Any]:
    """路由后直接执行 ``create_agent``，不经过 Parent StateGraph。"""
    resolved_original = (original_question or "").strip()
    resolved_effective = (effective_question or "").strip()
    try:
        invocation = prepare_agent_runtime_invocation(
            top_k=top_k,
            decompose_top_k=decompose_top_k,
            trace_id=trace_id,
            session_id=session_id,
            original_question=resolved_original,
            effective_question=resolved_effective,
            tool_context=tool_context,
        )
        logger.info(
            "[trace_id=%s] [agent] runtime_start action=%s session_id_present=%s",
            trace_id,
            invocation.decision.get("action", ""),
            bool((session_id or "").strip()),
        )
        if invocation.agent is None:
            result = finish_short_circuit(invocation)
        else:
            output = invocation.agent.invoke(
                invocation.agent_input,
                config=invocation.agent_config,
                context=invocation.tool_context,
            )
            if not isinstance(output, Mapping):
                raise RuntimeError("create_agent did not produce a valid state")
            result = finish_framework_agent(invocation, output)
        logger.info(
            "[trace_id=%s] [agent] runtime_end status=%s",
            trace_id,
            result.get("status", ""),
        )
        return result
    except Exception as exc:
        logger.exception("[trace_id=%s] [agent] runtime_failed", trace_id)
        return build_agent_runtime_error_result(
            write_bad_cases=write_bad_cases,
            bad_case=bad_case,
            trace_id=trace_id,
            session_id=session_id,
            original_question=resolved_original,
            effective_question=resolved_effective,
            exc=exc,
        )


__all__ = [
    "AgentRuntimeInvocation",
    "build_agent_runtime_error_result",
    "build_decide_trace",
    "build_tool_governance_event",
    "count_consecutive_events",
    "finish_framework_agent",
    "finish_short_circuit",
    "framework_metrics",
    "get_compiled_agent_runtime",
    "get_route_decision_for_trace",
    "maybe_save_agent_trace",
    "normalize_governance_events",
    "prepare_agent_runtime_invocation",
    "reset_agent_runtime",
    "resolve_llm_call_count",
    "run_agent",
    "sum_governance_event_costs",
    "upsert_llm_usage_trace",
    "validate_runtime_counts",
]

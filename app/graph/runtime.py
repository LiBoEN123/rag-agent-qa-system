"""LangGraph Agent Runtime 的共享调用准备与同步执行门面。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agent.results import (
    build_internal_error_result,
    validate_agent_result_payload,
)
from app.agent.runtime import (
    _write_internal_bad_case,
    maybe_save_agent_trace,
    validate_runtime_counts,
)
from app.common.logging import get_logger
from app.config import AppSettings, get_settings
from app.graph.builder import (
    get_async_compiled_agent_graph,
    get_compiled_agent_graph,
)
from app.graph.checkpointing import (
    build_checkpoint_thread_id,
    get_checkpointer_mode,
)
from app.tools.context import (
    ToolInvocationContext,
    build_default_tool_context,
)


logger = get_logger("chat_service")


@dataclass(frozen=True)
class AgentRuntimeInvocation:
    """同步与流式 Runtime 共同使用的一次图调用配置。"""

    graph: Any
    graph_input: dict[str, Any]
    graph_config: dict[str, Any]
    tool_context: ToolInvocationContext
    checkpointer_mode: str
    checkpoint_thread_id: str


def _prepare_agent_runtime_fields(
    *,
    top_k: int,
    decompose_top_k: int,
    trace_id: str,
    session_id: str | None,
    original_question: str,
    effective_question: str,
    tool_context: ToolInvocationContext | None = None,
    settings: AppSettings,
) -> dict[str, Any]:
    """构造同步与流式调用共享的输入、Context 和运行配置。"""
    resolved_top_k, resolved_decompose_top_k = validate_runtime_counts(
        top_k=top_k,
        decompose_top_k=decompose_top_k,
    )
    resolved_tool_context = tool_context or build_default_tool_context(
        trace_id=trace_id,
        session_id=session_id or "",
        source="chat_api",
        settings=settings,
    )
    graph_input: dict[str, Any] = {
        "original_question": (original_question or "").strip(),
        "effective_question": (effective_question or "").strip(),
        "top_k": resolved_top_k,
        "decompose_top_k": resolved_decompose_top_k,
        "trace_id": trace_id,
        "session_id": session_id,
    }
    mode = get_checkpointer_mode(settings)
    checkpoint_thread_id = (
        build_checkpoint_thread_id(
            trace_id=trace_id,
            session_id=session_id,
        )
        if mode != "disabled"
        else ""
    )
    graph_config: dict[str, Any] = {
        "recursion_limit": max(
            25,
            settings.agent_max_iterations * 8 + 20,
        ),
        "max_concurrency": 1,
    }
    if checkpoint_thread_id:
        graph_config["configurable"] = {"thread_id": checkpoint_thread_id}
    return {
        "graph_input": graph_input,
        "graph_config": graph_config,
        "tool_context": resolved_tool_context,
        "checkpointer_mode": mode,
        "checkpoint_thread_id": checkpoint_thread_id,
    }


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
    """为同步 ``graph.invoke`` 构造一次完整调用。"""
    resolved_settings = settings or get_settings()
    fields = _prepare_agent_runtime_fields(
        top_k=top_k,
        decompose_top_k=decompose_top_k,
        trace_id=trace_id,
        session_id=session_id,
        original_question=original_question,
        effective_question=effective_question,
        tool_context=tool_context,
        settings=resolved_settings,
    )
    return AgentRuntimeInvocation(
        graph=get_compiled_agent_graph(resolved_settings),
        **fields,
    )


async def prepare_async_agent_runtime_invocation(
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
    """为流式 ``graph.astream`` 构造使用异步 Saver 的完整调用。"""
    resolved_settings = settings or get_settings()
    fields = _prepare_agent_runtime_fields(
        top_k=top_k,
        decompose_top_k=decompose_top_k,
        trace_id=trace_id,
        session_id=session_id,
        original_question=original_question,
        effective_question=effective_question,
        tool_context=tool_context,
        settings=resolved_settings,
    )
    return AgentRuntimeInvocation(
        graph=await get_async_compiled_agent_graph(resolved_settings),
        **fields,
    )


def log_agent_runtime_start(
    invocation: AgentRuntimeInvocation,
    *,
    trace_id: str,
    session_id: str | None,
) -> None:
    """记录同步与流式入口一致的图调用起点。"""
    logger.info(
        "[trace_id=%s] [agent] langgraph_graph_start "
        "session_id_present=%s checkpointer_mode=%s checkpoint_thread_id=%s",
        trace_id,
        bool((session_id or "").strip()),
        invocation.checkpointer_mode,
        invocation.checkpoint_thread_id or "-",
    )


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
    """把同步与流式图异常转换成同一脱敏内部错误结果。"""
    _write_internal_bad_case(
        write_bad_cases=write_bad_cases,
        bad_case=bad_case,
        trace_id=trace_id,
        question=original_question,
        exc=exc,
        runtime_route="agent_langgraph",
        runtime_label="LangGraph Agent",
    )
    result = build_internal_error_result(
        exc,
        runtime_label="LangGraph Agent",
        trace_step="agent_langgraph_error",
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


def run_agent_langgraph(
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
    """同步执行 LangGraph 主图，并返回 API 主流程使用的结果字典。"""
    resolved_original_question = (original_question or "").strip()
    resolved_effective_question = (effective_question or "").strip()

    try:
        invocation = prepare_agent_runtime_invocation(
            top_k=top_k,
            decompose_top_k=decompose_top_k,
            trace_id=trace_id,
            session_id=session_id,
            original_question=resolved_original_question,
            effective_question=resolved_effective_question,
            tool_context=tool_context,
        )
        log_agent_runtime_start(
            invocation,
            trace_id=trace_id,
            session_id=session_id,
        )
        output = invocation.graph.invoke(
            invocation.graph_input,
            config=invocation.graph_config,
            context=invocation.tool_context,
        )
        result = output.get("result") if isinstance(output, dict) else None
        if not isinstance(result, dict):
            raise RuntimeError("LangGraph did not produce a valid result")
        result = validate_agent_result_payload(result)

        logger.info(
            "[trace_id=%s] [agent] langgraph_graph_end status=%s",
            trace_id,
            result.get("status", ""),
        )
        return result
    except Exception as exc:
        logger.exception(
            "[trace_id=%s] [agent] langgraph_graph_failed",
            trace_id,
        )
        return build_agent_runtime_error_result(
            write_bad_cases=write_bad_cases,
            bad_case=bad_case,
            trace_id=trace_id,
            session_id=session_id,
            original_question=resolved_original_question,
            effective_question=resolved_effective_question,
            exc=exc,
        )


__all__ = [
    "AgentRuntimeInvocation",
    "build_agent_runtime_error_result",
    "log_agent_runtime_start",
    "prepare_async_agent_runtime_invocation",
    "prepare_agent_runtime_invocation",
    "run_agent_langgraph",
]

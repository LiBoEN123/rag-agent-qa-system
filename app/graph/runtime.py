"""LangGraph Agent Runtime 同步执行门面。"""

from collections.abc import Callable
from typing import Any

from app.agent.runtime import (
    _write_internal_bad_case,
    build_internal_error_result,
    maybe_save_agent_trace,
)
from app.common.logging import get_logger
from app.graph.builder import get_compiled_agent_graph
from app.graph.checkpointing import (
    build_checkpoint_thread_id,
    get_checkpointer_mode,
)


logger = get_logger("chat_service")


def run_agent_langgraph(
    write_bad_cases: Callable[[dict[str, Any]], None] | None,
    top_k: int,
    decompose_top_k: int,
    decision: dict[str, Any] | None = None,
    trace_id: str = "",
    session_id: str | None = None,
    bad_case: dict[str, Any] | None = None,
    *,
    original_question: str,
    effective_question: str,
    principal_id: str = "chat_api",
    principal_type: str = "service",
    roles: tuple[str, ...] = ("reader",),
    permissions: tuple[str, ...] = ("knowledge_base:read",),
    tenant_id: str = "",
    request_source: str = "chat_api",
    idempotency_key: str = "",
) -> dict[str, Any]:
    """同步执行 LangGraph 主图，并返回 API 主流程使用的结果字典。"""
    resolved_original_question = (original_question or "").strip()
    resolved_effective_question = (effective_question or "").strip()
    decision_for_error = decision if isinstance(decision, dict) else {}
    graph_input: dict[str, Any] = {
        "original_question": resolved_original_question,
        "effective_question": resolved_effective_question,
        "top_k": top_k,
        "decompose_top_k": decompose_top_k,
        "decision": decision,
        "trace_id": trace_id,
        "session_id": session_id,
        "principal_id": principal_id,
        "principal_type": principal_type,
        "roles": list(roles),
        "permissions": list(permissions),
        "tenant_id": tenant_id,
        "request_source": request_source,
        "idempotency_key": idempotency_key,
    }

    try:
        mode = get_checkpointer_mode()
        checkpointer_enabled = mode != "disabled"
        checkpoint_thread_id = (
            build_checkpoint_thread_id(
                trace_id=trace_id,
                session_id=session_id,
            )
            if checkpointer_enabled
            else ""
        )
        logger.info(
            "[trace_id=%s] [agent] langgraph_graph_start "
            "session_id_present=%s checkpointer_mode=%s checkpoint_thread_id=%s",
            trace_id,
            bool((session_id or "").strip()),
            mode if checkpointer_enabled else "disabled",
            checkpoint_thread_id or "-",
        )
        graph = get_compiled_agent_graph()
        if checkpointer_enabled:
            output = graph.invoke(
                graph_input,
                config={
                    "configurable": {
                        "thread_id": checkpoint_thread_id,
                    }
                },
            )
        else:
            output = graph.invoke(graph_input)
        result = output.get("result") if isinstance(output, dict) else None
        if not isinstance(result, dict):
            raise RuntimeError("LangGraph did not produce a valid result")

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
        _write_internal_bad_case(
            write_bad_cases=write_bad_cases,
            bad_case=bad_case,
            trace_id=trace_id,
            question=resolved_original_question,
            exc=exc,
            runtime_route="agent_langgraph",
            runtime_label="LangGraph Agent",
        )
        result = build_internal_error_result(
            exc,
            decision_for_error,
            runtime_label="LangGraph Agent",
            trace_step="agent_langgraph_error",
        )
        maybe_save_agent_trace(
            trace_id=trace_id,
            session_id=session_id,
            question=resolved_original_question,
            rewritten_question=(
                resolved_effective_question
                if resolved_effective_question != resolved_original_question
                else ""
            ),
            final_answer=result.get("answer", ""),
            status=result.get("status", ""),
            reason=result.get("reason", ""),
            tool_trace=result.get("tool_trace", []),
        )
        return result


__all__ = ["run_agent_langgraph"]

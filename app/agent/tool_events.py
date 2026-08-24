"""把框架治理事件投影为 observation、tool trace 与指标。"""

from collections.abc import Mapping
from typing import Any

from app.agent.state import AgentObservation
from app.tools.audit import sanitize_tool_input, summarize_tool_output
from app.tools.redaction import redact_sensitive_data
from app.tools.registry import get_tool_registry


def summarize_tool_result(tool_name: str, tool_result: dict[str, Any]) -> str:
    """为标准工具结果生成简短摘要。"""
    success = bool(tool_result.get("ok", tool_result.get("status") == "success"))
    if not success:
        return "工具执行失败"

    summary = tool_result.get("summary") or tool_result.get("result_summary")
    if summary:
        return str(summary)

    summaries = {
        "compare_documents": "文档对比工具执行完成",
        "simple_calculator": "计算工具执行完成",
        "list_documents": "文档列表工具执行完成",
        "get_chunk_text": "chunk 原文工具执行完成",
    }
    if tool_name == "search_knowledge_base":
        count = (tool_result.get("output") or {}).get("count", 0)
        return f"检索到 {count} 个候选 chunk"
    return summaries.get(tool_name, f"{tool_name} 工具执行完成")


def build_tool_observation(
    *,
    step_id: int,
    tool_name: str,
    result: dict[str, Any],
    cost_ms: int,
) -> AgentObservation:
    """将一次框架工具结果规整为 AgentObservation。"""
    if step_id < 1:
        raise ValueError("observation step_id must be at least 1")
    if cost_ms < 0:
        raise ValueError("observation cost_ms must be non-negative")
    success = bool(result.get("ok", result.get("status") == "success"))
    output = result.get("output") or {}
    if not isinstance(output, dict):
        output = {}
    error_code = str(result.get("error_code") or "")
    raw_error = str(result.get("error") or "")
    safe_policy_codes = {
        "permission_denied",
        "sql_permission_denied",
        "approval_required",
        "missing_idempotency_key",
        "invalid_invocation_context",
        "policy_denied",
        "audit_write_failed",
    }
    if error_code in safe_policy_codes:
        error = raw_error
    elif error_code:
        error = f"工具执行失败（{error_code}）"
    else:
        error = raw_error
    if not success and not error:
        error = "Tool execution failed"

    return {
        "step_id": step_id,
        "tool_name": tool_name,
        "success": success,
        "summary": summarize_tool_result(tool_name, result),
        "output": output,
        "error": error,
        "cost_ms": cost_ms,
    }


def build_tool_trace(
    *,
    step_id: int,
    tool_name: str,
    tool_args: dict[str, Any],
    result: dict[str, Any],
    observation: AgentObservation,
) -> dict[str, Any]:
    """构造兼容现有响应结构的工具调用 trace。"""
    spec = get_tool_registry().get(tool_name)
    sanitized_args = (
        sanitize_tool_input(spec, tool_args)
        if spec is not None
        else redact_sensitive_data(tool_args)
    )
    governance = result.get("governance")
    governance = governance if isinstance(governance, dict) else {}
    runtime = result.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    trace: dict[str, Any] = {
        "step": "tool_call",
        "step_id": step_id,
        "action": "tool_call",
        "tool_name": tool_name,
        "tool_args": sanitized_args,
        "sanitized_tool_args": sanitized_args,
        "tool_status": result.get("status", ""),
        "success": observation["success"],
        "summary": redact_sensitive_data(observation["summary"]),
        "error": redact_sensitive_data(observation["error"]),
        "cost_ms": observation["cost_ms"],
        "error_code": result.get("error_code", ""),
        "tool_call_id": result.get("tool_call_id", ""),
        "policy_decision": governance.get("decision", ""),
        "risk_level": governance.get("risk_level", ""),
        "approval_required": governance.get("approval_required", False),
        "args_fingerprint": governance.get("args_fingerprint", ""),
        "attempt_count": runtime.get("attempt_count", 0),
        "max_attempts": runtime.get("max_attempts", 0),
        "timed_out": runtime.get("timed_out", False),
    }

    if tool_name == "search_knowledge_base":
        output = result.get("output") or {}
        trace.update(
            {
                "final_chunk_count": output.get("count", 0),
                "top_chunk_ids": list(output.get("top_chunk_ids", []))[:10]
                if isinstance(output.get("top_chunk_ids"), (list, tuple))
                else [],
                "top_scores": list(output.get("top_scores", []))[:10]
                if isinstance(output.get("top_scores"), (list, tuple))
                else [],
            }
        )
        hits = []
        for chunk in output.get("chunks", [])[:10]:
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            hits.append(
                {
                    "chunk_id": chunk.get("chunk_id") or metadata.get("chunk_id"),
                    "rerank": chunk.get("rerank_score"),
                    "score": chunk.get("hybrid_score"),
                    "bm25": chunk.get("bm25_score"),
                    "vector": chunk.get("vector_score"),
                    "file_name": chunk.get("file_name") or metadata.get("file_name"),
                }
            )
        trace["hits"] = hits
        trace["top_chunk_ids"] = redact_sensitive_data(trace["top_chunk_ids"])
        trace["top_scores"] = redact_sensitive_data(trace["top_scores"])
        trace["hits"] = redact_sensitive_data(trace["hits"])
    elif spec is not None:
        trace["output_summary"] = summarize_tool_output(
            spec, observation["output"]
        )
    else:
        trace["output_summary"] = redact_sensitive_data(observation["output"])

    return trace


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
    """把 append-only framework 事件一次性投影为稳定公开数据。"""
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


__all__ = [
    "build_tool_observation",
    "build_tool_trace",
    "framework_metrics",
    "normalize_governance_events",
    "summarize_tool_result",
]

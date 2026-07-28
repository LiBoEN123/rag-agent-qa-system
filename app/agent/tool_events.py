"""把框架工具事件转换为兼容的 observation 和 tool trace。"""

from typing import Any

from app.agent.state import AgentObservation
from app.tools.audit import sanitize_tool_input, summarize_tool_output
from app.tools.redaction import redact_sensitive_data
from app.tools.registry import tool_registry


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
    success = bool(result.get("ok", result.get("status") == "success"))
    output = result.get("output") or {}
    error_code = str(result.get("error_code") or "")
    raw_error = str(result.get("error") or "")
    safe_policy_codes = {
        "permission_denied",
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

    return AgentObservation(
        step_id=step_id,
        tool_name=tool_name,
        success=success,
        summary=summarize_tool_result(tool_name, result),
        output=output,
        error=error,
        cost_ms=cost_ms,
    )


def build_tool_trace(
    *,
    step_id: int,
    tool_name: str,
    tool_args: dict[str, Any],
    result: dict[str, Any],
    observation: AgentObservation,
) -> dict[str, Any]:
    """构造兼容现有响应结构的工具调用 trace。"""
    spec = tool_registry.get(tool_name)
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
        "success": observation.success,
        "summary": redact_sensitive_data(observation.summary),
        "error": redact_sensitive_data(observation.error),
        "cost_ms": observation.cost_ms,
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
        trace["output_summary"] = summarize_tool_output(spec, observation.output)
    else:
        trace["output_summary"] = redact_sensitive_data(observation.output)

    return trace


__all__ = [
    "build_tool_observation",
    "build_tool_trace",
    "summarize_tool_result",
]

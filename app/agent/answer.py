"""Agent 答案合成与 RAG 结果的纯更新映射。"""

from collections.abc import Mapping
from typing import Any

from app.agent.results import map_rag_status_to_agent_status

ERROR_STATUSES = {"tool_error", "error", "model_error"}
INTERNAL_ERROR_TEXT = "Agent 内部执行异常，请稍后重试"

DIRECT_ANSWER_TEXT = (
    "我是一个 RAG 知识库问答助手。"
    "我可以基于已导入的文档回答问题并提供引用，"
    "也可以列出知识库文档、查看指定 chunk、对比文档内容，以及进行简单四则运算。"
)

def _copy_trace(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """复制当前 trace，避免答案 helper 修改调用方列表。"""
    return [
        dict(item)
        for item in state.get("tool_trace", [])
        if isinstance(item, dict)
    ]


def build_rag_answer_updates(
    state: Mapping[str, Any],
    rag_result: Mapping[str, Any],
) -> dict[str, Any]:
    """将 RAG 子图结果映射为平铺 AgentGraphState 更新。"""
    status = map_rag_status_to_agent_status(rag_result.get("status", ""))
    final_answer = str(rag_result.get("answer", "") or "")
    raw_references = rag_result.get("references", [])
    references = (
        [dict(item) for item in raw_references if isinstance(item, dict)]
        if isinstance(raw_references, list)
        else []
    )
    reason = str(rag_result.get("reason", "") or "")
    decompose_llm_used = bool(
        rag_result.get("decompose_llm_used", False)
    )
    answer_llm_used = bool(rag_result.get("answer_llm_used", False))
    rag_llm_call_count = max(
        0, int(rag_result.get("llm_call_count", 0) or 0)
    )
    llm_call_count = max(0, int(state.get("llm_call_count", 0) or 0))
    llm_call_count += rag_llm_call_count

    timings = rag_result.get("timings")
    tool_trace = _copy_trace(state)
    tool_trace.append(
        {
            "step": "compose_final_answer",
            "mode": "rag",
            "status": status,
            "answer_len": len(final_answer),
            "reference_count": len(references),
            "reason": reason,
            "retrieval_confidence": rag_result.get(
                "retrieval_confidence",
                0.0,
            ),
            "answer_confidence": rag_result.get("answer_confidence", 0.0),
            "timings": dict(timings) if isinstance(timings, dict) else {},
            "rag_llm_call_count": rag_llm_call_count,
        }
    )
    return {
        "status": status,
        "final_answer": final_answer,
        "references": references,
        "reason": reason,
        "decompose_llm_used": decompose_llm_used,
        "answer_llm_used": answer_llm_used,
        "llm_call_count": llm_call_count,
        "tool_trace": tool_trace,
    }


def build_compose_answer_updates(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """为确定性短路、框架回答或错误状态构造平铺更新。"""
    status = str(state.get("status") or "running")
    final_answer = str(state.get("final_answer") or "")
    reason = str(state.get("reason") or "")
    tool_trace = _copy_trace(state)

    if status == "clarify":
        tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "clarify",
                "status": status,
                "answer_len": len(final_answer),
                "reason": reason,
            }
        )
    elif status == "direct_answer":
        final_answer = final_answer or DIRECT_ANSWER_TEXT
        tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "direct",
                "status": status,
                "answer_len": len(final_answer),
                "reason": reason,
            }
        )
    elif status in ERROR_STATUSES:
        if status == "error":
            final_answer = INTERNAL_ERROR_TEXT
        elif status == "model_error" and not final_answer:
            final_answer = "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
        elif not final_answer:
            final_answer = (
                f"工具执行失败，无法生成可靠回答。原因：{reason}"
                if reason
                else "工具执行失败，无法生成可靠回答。"
            )
        tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "error",
                "status": status,
                "answer_len": len(final_answer),
                "reason": reason,
            }
        )
    elif status == "ok" and final_answer:
        tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "agent_loop",
                "status": status,
                "answer_len": len(final_answer),
                "reason": reason,
            }
        )
    else:
        previous_status = status
        status = "error"
        reason = f"Unexpected Agent state before compose: {previous_status}"
        final_answer = INTERNAL_ERROR_TEXT
        tool_trace.append(
            {
                "step": "compose_final_answer",
                "mode": "error",
                "status": status,
                "reason": reason,
            }
        )
    return {
        "status": status,
        "final_answer": final_answer,
        "reason": reason,
        "tool_trace": tool_trace,
    }


__all__ = [
    "build_compose_answer_updates",
    "build_rag_answer_updates",
]

"""把聊天应用服务结果适配为稳定的 HTTP 响应 Schema。"""

from app.api.schemas import ChatData, ChatResponse
from app.observability.metrics import extract_llm_usage
from app.rag.retrieval import safe_float
from app.services.chat_models import ChatServiceOutcome


def build_chat_response_from_outcome(
    outcome: ChatServiceOutcome,
    *,
    trace_id: str,
) -> ChatResponse:
    """沿唯一构造路径把应用服务结果转换为 ChatResponse。"""
    result = outcome.result
    retrieval_confidence = max(
        0.0,
        min(1.0, safe_float(result.get("retrieval_confidence", 0.0))),
    )
    answer_confidence = max(
        0.0,
        min(1.0, safe_float(result.get("answer_confidence", 0.0))),
    )
    tool_trace = list(result.get("tool_trace") or [])
    llm_usage = extract_llm_usage(
        tool_trace,
        router_llm_used=result.get("router_llm_used"),
        decompose_llm_used=result.get("decompose_llm_used"),
        answer_llm_used=result.get("answer_llm_used"),
        llm_call_count=result.get("llm_call_count"),
    )
    return ChatResponse(
        code=outcome.code,
        message=outcome.message,
        trace_id=trace_id,
        data=ChatData(
            question=outcome.question,
            answer=result.get("answer", ""),
            references=list(result.get("references") or []),
            retrieval_confidence=retrieval_confidence,
            answer_confidence=answer_confidence,
            need_clarification=result.get("need_clarification", False),
            status=outcome.response_status,
            clarification_question=result.get(
                "clarification_question",
                "",
            ),
            reason=result.get(
                "reason",
                "正常回答" if outcome.code == 0 else "",
            ),
            action=result.get("action", ""),
            tool_result_summary=result.get("tool_result_summary", ""),
            tool_trace=tool_trace,
            **llm_usage,
        ),
    )


__all__ = ["build_chat_response_from_outcome"]

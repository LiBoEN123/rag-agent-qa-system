"""Chat response construction and internal-status mapping."""

from dataclasses import dataclass
from typing import Any, Callable


CHAT_RESULT_RESPONSE_MAP = {
    "MODEL_ERROR": {
        "code": "MODEL_ERROR",
        "message": "模型调用失败",
        "status": "error",
        "default_reason": "模型调用失败",
        "references": [],
        "need_clarification": False,
        "clarification_question": "",
        "log_status": "error",
    },
    "NEED_CLARIFICATION": {
        "code": "NEED_CLARIFICATION",
        "message": "问题信息不足，需要补充后再回答",
        "status": "clarify",
        "default_reason": "问题缺少关键限定信息",
        "answer": "",
        "references": [],
        "need_clarification": True,
        "log_status": "need_clarification",
    },
    "NO_RELEVANT_INFO": {
        "code": "NO_RELEVANT_INFO",
        "message": "我没有在文档中找到相关信息",
        "status": "no_evidence",
        "default_reason": "模型判断当前上下文不足以支撑回答",
        "clarification_question": "",
        "log_status": "no_evidence",
    },
    "TOOL_ERROR": {
        "code": "TOOL_ERROR",
        "message": "工具调用失败",
        "status": "error",
        "default_reason": "工具调用失败",
        "references": [],
        "log_status": "error",
    },
    "OUT_OF_SCOPE": {
        "code": "OUT_OF_SCOPE",
        "message": "问题超出当前知识库范围",
        "status": "out_of_scope",
        "default_reason": "问题超出知识库范围",
        "references": [],
        "need_clarification": False,
        "clarification_question": "",
        "log_status": "out_of_scope",
    },
    "MODEL_BAD_OUTPUT": {
        "code": "MODEL_BAD_OUTPUT",
        "message": "模型输出格式异常，请稍候重试",
        "status": "error",
        "default_reason": "模型返回结构不符合预期",
        "references": [],
        "clarification_question": "",
        "log_status": "error",
    },
    "INTERNAL_ERROR": {
        "code": "INTERNAL_ERROR",
        "message": "服务器内部错误",
        "status": "error",
        "default_reason": "Agent 内部执行异常",
        "references": [],
        "need_clarification": False,
        "clarification_question": "",
        "log_status": "error",
    },
}


@dataclass(frozen=True)
class ChatResponseBuilder:
    response_type: type
    data_type: type
    safe_float: Callable
    extract_llm_usage: Callable

    def build(
        self,
        code: int | str,
        message: str,
        trace_id: str,
        question: str,
        answer: str = "",
        references: list | None = None,
        retrieval_confidence: float = 0.0,
        answer_confidence: float = 0.0,
        need_clarification: bool = False,
        status: str = "ok",
        clarification_question: str = "",
        reason: str = "",
        action: str = "",
        tool_result_summary: str = "",
        tool_trace: list | None = None,
        router_llm_used: bool | None = None,
        decompose_llm_used: bool | None = None,
        answer_llm_used: bool | None = None,
        llm_call_count: int | None = None,
    ) -> Any:
        retrieval_confidence = max(
            0.0, min(1.0, self.safe_float(retrieval_confidence))
        )
        answer_confidence = max(0.0, min(1.0, self.safe_float(answer_confidence)))
        tool_trace = tool_trace or []
        llm_usage = self.extract_llm_usage(
            tool_trace,
            router_llm_used=router_llm_used,
            decompose_llm_used=decompose_llm_used,
            answer_llm_used=answer_llm_used,
            llm_call_count=llm_call_count,
        )
        return self.response_type(
            code=code,
            message=message,
            trace_id=trace_id,
            data=self.data_type(
                question=question,
                answer=answer,
                references=references or [],
                retrieval_confidence=retrieval_confidence,
                answer_confidence=answer_confidence,
                need_clarification=need_clarification,
                status=status,
                clarification_question=clarification_question,
                reason=reason,
                action=action,
                tool_result_summary=tool_result_summary,
                tool_trace=tool_trace,
                **llm_usage,
            ),
        )

    def build_from_result(
        self,
        result: dict,
        trace_id: str,
        question: str,
        response_config: dict,
    ) -> Any:
        return self.build(
            code=response_config["code"],
            message=response_config["message"],
            trace_id=trace_id,
            question=question,
            answer=response_config.get("answer", result.get("answer", "")),
            references=response_config.get("references", result.get("references", [])),
            retrieval_confidence=result.get("retrieval_confidence", 0.0),
            answer_confidence=result.get("answer_confidence", 0.0),
            need_clarification=response_config.get(
                "need_clarification", result.get("need_clarification", False)
            ),
            status=response_config["status"],
            clarification_question=response_config.get(
                "clarification_question",
                result.get("clarification_question", ""),
            ),
            reason=result.get("reason", response_config["default_reason"]),
            action=result.get("action", ""),
            tool_result_summary=result.get("tool_result_summary", ""),
            tool_trace=result.get("tool_trace", []),
            router_llm_used=result.get("router_llm_used"),
            decompose_llm_used=result.get("decompose_llm_used"),
            answer_llm_used=result.get("answer_llm_used"),
            llm_call_count=result.get("llm_call_count"),
        )

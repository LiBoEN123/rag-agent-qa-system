"""聊天响应构造模块。

把 Agent 内部状态和结果转换为稳定、统一的 HTTP 响应数据结构。
"""

from dataclasses import dataclass
from typing import Any, Callable

from app.services.chat_service import CHAT_RESULT_RESPONSE_MAP

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
        """根据标准字段构造统一的聊天响应对象。"""
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
        """根据 Agent 结果和状态配置构造聊天响应。"""
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

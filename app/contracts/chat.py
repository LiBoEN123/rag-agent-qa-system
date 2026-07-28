"""聊天结果状态在 HTTP、API 与 metrics 之间的唯一转换策略。"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, TypeAlias


class AgentResultStatus(StrEnum):
    """Agent Runtime 对外提供给聊天应用服务的稳定结果状态。"""

    OK = "OK"
    NEED_CLARIFICATION = "NEED_CLARIFICATION"
    NO_RELEVANT_INFO = "NO_RELEVANT_INFO"
    TOOL_ERROR = "TOOL_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


PublicStatus: TypeAlias = Literal[
    "ok",
    "no_evidence",
    "empty_index",
    "error",
    "clarify",
    "out_of_scope",
]

MetricStatus: TypeAlias = Literal[
    "success",
    "clarify",
    "no_relevant_info",
    "tool_error",
    "model_error",
    "internal_error",
]


@dataclass(frozen=True, slots=True)
class ChatStatusPolicy:
    """一个 Result status 对应的完整聊天响应与指标策略。"""

    result_status: AgentResultStatus
    http_status: int
    code: int | str
    message: str
    public_status: PublicStatus
    metric_status: MetricStatus
    default_reason: str
    log_status: str
    expose_answer: bool = True
    expose_references: bool = True
    force_need_clarification: bool | None = None
    clear_clarification_question: bool = False


CHAT_STATUS_POLICIES: Mapping[AgentResultStatus, ChatStatusPolicy] = (
    MappingProxyType(
        {
            AgentResultStatus.OK: ChatStatusPolicy(
                result_status=AgentResultStatus.OK,
                http_status=200,
                code=0,
                message="success",
                public_status="ok",
                metric_status="success",
                default_reason="正常回答",
                log_status="ok",
            ),
            AgentResultStatus.NEED_CLARIFICATION: ChatStatusPolicy(
                result_status=AgentResultStatus.NEED_CLARIFICATION,
                http_status=200,
                code="NEED_CLARIFICATION",
                message="问题信息不足，需要补充后再回答",
                public_status="clarify",
                metric_status="clarify",
                default_reason="问题缺少关键限定信息",
                log_status="need_clarification",
                expose_answer=False,
                expose_references=False,
                force_need_clarification=True,
            ),
            AgentResultStatus.NO_RELEVANT_INFO: ChatStatusPolicy(
                result_status=AgentResultStatus.NO_RELEVANT_INFO,
                http_status=200,
                code="NO_RELEVANT_INFO",
                message="我没有在文档中找到相关信息",
                public_status="no_evidence",
                metric_status="no_relevant_info",
                default_reason="模型判断当前上下文不足以支撑回答",
                log_status="no_evidence",
                clear_clarification_question=True,
            ),
            AgentResultStatus.TOOL_ERROR: ChatStatusPolicy(
                result_status=AgentResultStatus.TOOL_ERROR,
                http_status=200,
                code="TOOL_ERROR",
                message="工具调用失败",
                public_status="error",
                metric_status="tool_error",
                default_reason="工具调用失败",
                log_status="error",
                expose_references=False,
                force_need_clarification=False,
                clear_clarification_question=True,
            ),
            AgentResultStatus.MODEL_ERROR: ChatStatusPolicy(
                result_status=AgentResultStatus.MODEL_ERROR,
                http_status=200,
                code="MODEL_ERROR",
                message="模型调用失败",
                public_status="error",
                metric_status="model_error",
                default_reason="模型调用失败",
                log_status="error",
                expose_references=False,
                force_need_clarification=False,
                clear_clarification_question=True,
            ),
            AgentResultStatus.INTERNAL_ERROR: ChatStatusPolicy(
                result_status=AgentResultStatus.INTERNAL_ERROR,
                http_status=500,
                code="INTERNAL_ERROR",
                message="服务器内部错误",
                public_status="error",
                metric_status="internal_error",
                default_reason="Agent 内部执行异常",
                log_status="error",
                expose_references=False,
                force_need_clarification=False,
                clear_clarification_question=True,
            ),
        }
    )
)


def get_chat_status_policy(
    status: AgentResultStatus | str,
) -> ChatStatusPolicy:
    """解析唯一状态策略；未知值一律按内部错误 fail closed。"""
    try:
        normalized = AgentResultStatus(str(status or "").strip())
    except ValueError:
        normalized = AgentResultStatus.INTERNAL_ERROR
    return CHAT_STATUS_POLICIES[normalized]


__all__ = [
    "AgentResultStatus",
    "CHAT_STATUS_POLICIES",
    "ChatStatusPolicy",
    "MetricStatus",
    "PublicStatus",
    "get_chat_status_policy",
]

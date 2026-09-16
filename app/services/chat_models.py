"""聊天应用服务使用的框架无关数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.contracts.chat import PublicStatus, ToolMode


@dataclass(frozen=True)
class ChatServiceRequest:
    """一次聊天业务调用所需的框架无关输入。"""

    question: str
    top_k: int | None
    decompose_top_k: int | None
    session_id: str
    trace_id: str
    request_enter_time: str = ""
    request_enter_perf: float | None = None
    tool_mode: ToolMode = ToolMode.FULL


@dataclass(frozen=True)
class ChatServiceOutcome:
    """聊天业务结果及其 HTTP 适配元数据。"""

    http_status: int
    code: int | str
    message: str
    question: str
    result: dict[str, Any]
    response_status: PublicStatus = "ok"


@dataclass
class ChatRequestContext:
    """单次聊天调用在应用服务内部共享的处理上下文。"""

    request: ChatServiceRequest
    process_start_perf: float
    process_start_time: str
    request_enter_perf: float
    original_question: str
    session_id: str
    effective_question: str
    metrics: dict[str, Any]
    bad_case: dict[str, Any]
    final_top_k: int = 0
    decompose_top_k: int = 0
    rewrite_trace: dict[str, Any] = field(default_factory=dict)
    session_should_store: bool = False
    session_assistant_content: str = ""
    finalized: bool = False

"""把 ChatService 白名单事件编码为稳定的 SSE 信封。"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi.sse import ServerSentEvent

from app.api.responses import build_chat_response_from_outcome
from app.observability.metrics import now_iso
from app.services.chat_models import ChatServiceOutcome
from app.services.chat_stream_models import ChatStreamEvent


_PROGRESS_STAGES = frozenset(
    {"query_rewrite", "routing", "agent", "rag", "finalizing"}
)
_PROGRESS_STATUSES = frozenset(
    {"started", "completed", "skipped", "failed"}
)
_TOOL_PHASES = frozenset({"started", "completed"})
_DONE_REASONS = frozenset({"completed", "error"})


def _non_negative_int(value: object) -> int:
    """把公开耗时字段规整为非负整数。"""
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _project_public_data(
    event: ChatStreamEvent,
    *,
    trace_id: str,
) -> dict[str, Any]:
    """逐事件重建公开字段，未知内部字段一律不进入 SSE。"""
    data = event.data
    if event.event == "start":
        return {"session_id_present": bool(data.get("session_id_present"))}
    if event.event == "progress":
        stage = str(data.get("stage") or "")
        status = str(data.get("status") or "")
        if stage not in _PROGRESS_STAGES or status not in _PROGRESS_STATUSES:
            raise ValueError("invalid progress stream event")
        return {"stage": stage, "status": status}
    if event.event == "tool":
        tool_name = str(data.get("tool_name") or "")
        phase = str(data.get("phase") or "")
        status = str(data.get("status") or "")
        if not tool_name or phase not in _TOOL_PHASES:
            raise ValueError("invalid tool stream event")
        projected: dict[str, Any] = {
            "tool_name": tool_name,
            "phase": phase,
            "status": status,
        }
        if phase == "completed":
            projected["cost_ms"] = _non_negative_int(data.get("cost_ms"))
        return projected
    if event.event == "answer_delta":
        text = data.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("invalid answer delta stream event")
        return {"text": text}
    if event.event == "final":
        outcome = data.get("outcome")
        if not isinstance(outcome, ChatServiceOutcome):
            raise TypeError("final stream event requires ChatServiceOutcome")
        response = build_chat_response_from_outcome(
            outcome,
            trace_id=trace_id,
        )
        return {
            "http_status": outcome.http_status,
            "response": response.model_dump(mode="json"),
            "total_cost_ms": _non_negative_int(data.get("total_cost_ms")),
        }
    if event.event == "error":
        return {
            "code": "STREAM_ERROR",
            "message": "流式处理失败",
        }
    if event.event == "done":
        reason = str(data.get("reason") or "")
        if reason not in _DONE_REASONS:
            raise ValueError("invalid done stream event")
        return {"reason": reason}
    raise ValueError(f"unsupported stream event: {event.event}")


class ChatStreamSSEEncoder:
    """为单个请求分配 sequence，并构造 FastAPI SSE 事件。"""

    def __init__(
        self,
        *,
        trace_id: str,
        timestamp_provider: Callable[[], str] = now_iso,
    ) -> None:
        self._trace_id = trace_id
        self._timestamp_provider = timestamp_provider
        self._sequence = 0

    def build(self, event: ChatStreamEvent) -> ServerSentEvent:
        """把服务事件投影为统一信封及 ``event``/``id`` 字段。"""
        public_data = _project_public_data(event, trace_id=self._trace_id)
        self._sequence += 1
        envelope = {
            "schema_version": 1,
            "trace_id": self._trace_id,
            "sequence": self._sequence,
            "timestamp": self._timestamp_provider(),
            "data": public_data,
        }
        return ServerSentEvent(
            event=event.event,
            id=str(self._sequence),
            data=envelope,
        )

    def encode(self, event: ChatStreamEvent) -> bytes:
        """编码为 SSE wire bytes，供可在建流前返回 JSON 的路由使用。"""
        return encode_server_sent_event(self.build(event))


def encode_server_sent_event(event: ServerSentEvent) -> bytes:
    """编码已校验的 FastAPI ServerSentEvent，不接受 raw data。"""
    if event.raw_data is not None:
        raise ValueError("raw SSE data is not allowed")
    payload = json.dumps(
        event.data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    lines = []
    if event.id is not None:
        lines.append(f"id: {event.id}")
    if event.event is not None:
        lines.append(f"event: {event.event}")
    lines.append(f"data: {payload}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


__all__ = [
    "ChatStreamSSEEncoder",
    "encode_server_sent_event",
]

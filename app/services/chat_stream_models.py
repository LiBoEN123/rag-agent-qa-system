"""聊天流式编排使用的框架无关事件模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CHAT_STREAM_EVENT_NAMES = frozenset(
    {
        "start",
        "progress",
        "tool",
        "answer_delta",
        "final",
        "error",
        "done",
    }
)


@dataclass(frozen=True)
class ChatStreamEvent:
    """ChatService 向 HTTP adapter 发送的单个白名单事件。"""

    event: str
    data: dict[str, Any]

    # 在事件创建后校验事件名称和字典形式的数据载荷。
    def __post_init__(self) -> None:
        if self.event not in CHAT_STREAM_EVENT_NAMES:
            raise ValueError(f"unsupported chat stream event: {self.event}")
        if not isinstance(self.data, dict):
            raise TypeError("chat stream event data must be a dict")


__all__ = ["CHAT_STREAM_EVENT_NAMES", "ChatStreamEvent"]

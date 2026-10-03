"""聊天 HTTP Router。"""

from contextlib import aclosing

from fastapi import APIRouter, Request, Response
from fastapi.sse import EventSourceResponse

from app.api.schemas import ChatRequest
from app.api.streaming import ChatStreamSSEEncoder
from app.observability import create_trace_id
from app.services.chat_stream_models import ChatStreamEvent
from app.services.chat_models import ChatServiceRequest
from app.services.chat_service import ChatService


def _build_service_request(
    req: ChatRequest,
    request: Request,
    *,
    trace_id: str,
) -> ChatServiceRequest:
    """把聊天 HTTP 上下文适配为服务请求。"""
    return ChatServiceRequest(
        question=req.question,
        session_id=req.session_id or "",
        tool_mode=req.tool_mode,
        trace_id=trace_id,
        request_enter_time=getattr(
            request.state,
            "request_enter_time",
            "",
        ),
        request_enter_perf=getattr(
            request.state,
            "request_enter_perf",
            None,
        ),
    )


def create_chat_router(chat_service: ChatService) -> APIRouter:
    """创建只负责 HTTP 输入输出适配的聊天路由。"""
    router = APIRouter()

    @router.post(
        "/chat/stream",
        response_model=None,
        response_class=Response,
        responses={
            200: {
                "description": "聊天 SSE 事件流",
                "content": {"text/event-stream": {}},
            },
        },
    )
    async def ask_stream(
        req: ChatRequest,
        request: Request,
    ):
        """所有通过 Schema 校验的问题统一进入服务事件流。"""
        trace_id = (
            getattr(request.state, "trace_id", "")
            or create_trace_id()
        )
        service_request = _build_service_request(
            req,
            request,
            trace_id=trace_id,
        )
        encoder = ChatStreamSSEEncoder(trace_id=trace_id)
        service_stream = chat_service.stream(service_request)

        # 编码服务事件，处理客户端断开和流异常，并确保关闭底层生成器。
        async def encoded_events():
            async with aclosing(service_stream):
                try:
                    iterator = service_stream.__aiter__()
                    first_event = True
                    while True:
                        if not first_event and await request.is_disconnected():
                            break
                        try:
                            event = await anext(iterator)
                        except StopAsyncIteration:
                            break
                        if await request.is_disconnected():
                            break
                        yield encoder.encode(event)
                        first_event = False
                except Exception:
                    if not await request.is_disconnected():
                        yield encoder.encode(
                            ChatStreamEvent(
                                event="error",
                                data={
                                    "code": "STREAM_ERROR",
                                    "message": "流式处理失败",
                                },
                            )
                        )
                        yield encoder.encode(
                            ChatStreamEvent(
                                event="done",
                                data={"reason": "error"},
                            )
                        )

        return EventSourceResponse(
            encoded_events(),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router


__all__ = ["create_chat_router"]

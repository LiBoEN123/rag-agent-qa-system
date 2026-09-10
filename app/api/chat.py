"""聊天 HTTP Router。"""

import asyncio

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse

from app.api.responses import build_chat_response_from_outcome
from app.api.schemas import ChatRequest, ChatResponse
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
    """把同步和流式入口的相同 HTTP 上下文适配为服务请求。"""
    return ChatServiceRequest(
        question=req.question,
        top_k=req.top_k,
        decompose_top_k=req.decompose_top_k,
        session_id=req.session_id or "",
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


async def _close_stream(stream: object) -> None:
    """关闭 Service async generator，并让取消继续传到 Agent Runtime。"""
    close = getattr(stream, "aclose", None)
    if callable(close):
        await close()


def create_chat_router(chat_service: ChatService) -> APIRouter:
    """创建只负责 HTTP 输入输出适配的聊天路由。"""
    router = APIRouter()

    @router.post("/chat", response_model=ChatResponse)
    def ask_api(
        req: ChatRequest,
        request: Request,
        response: Response,
    ) -> ChatResponse:
        """读取 HTTP 上下文、调用应用服务并设置响应状态。"""
        trace_id = (
            getattr(request.state, "trace_id", "")
            or create_trace_id()
        )
        outcome = chat_service.handle(
            _build_service_request(
                req,
                request,
                trace_id=trace_id,
            )
        )
        response.status_code = outcome.http_status
        return build_chat_response_from_outcome(
            outcome,
            trace_id=trace_id,
        )

    @router.post(
        "/chat/stream",
        response_model=None,
        response_class=Response,
        responses={
            200: {
                "description": "聊天 SSE 事件流",
                "content": {"text/event-stream": {}},
            },
            400: {
                "description": "空问题",
                "model": ChatResponse,
            },
        },
    )
    async def ask_stream(
        req: ChatRequest,
        request: Request,
    ):
        """在建流前处理空问题，其余事件只做 HTTP/SSE 适配。"""
        trace_id = (
            getattr(request.state, "trace_id", "")
            or create_trace_id()
        )
        service_request = _build_service_request(
            req,
            request,
            trace_id=trace_id,
        )
        if not (req.question or "").strip():
            outcome = await asyncio.to_thread(
                chat_service.handle,
                service_request,
            )
            response = build_chat_response_from_outcome(
                outcome,
                trace_id=trace_id,
            )
            return JSONResponse(
                status_code=outcome.http_status,
                content=response.model_dump(mode="json"),
            )

        encoder = ChatStreamSSEEncoder(trace_id=trace_id)
        service_stream = chat_service.stream(service_request)

        # 编码服务事件，处理客户端断开和流异常，并确保关闭底层生成器。
        async def encoded_events():
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
            except asyncio.CancelledError:
                raise
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
            finally:
                await _close_stream(service_stream)

        return EventSourceResponse(
            encoded_events(),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router


__all__ = ["create_chat_router"]

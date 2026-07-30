"""聊天 HTTP Router。"""

from fastapi import APIRouter, Request, Response

from app.api.responses import build_chat_response_from_outcome
from app.api.schemas import ChatRequest, ChatResponse
from app.observability import create_trace_id
from app.services.chat_models import ChatServiceRequest
from app.services.chat_service import ChatService


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
            ChatServiceRequest(
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
        )
        response.status_code = outcome.http_status
        return build_chat_response_from_outcome(
            outcome,
            trace_id=trace_id,
        )

    return router


__all__ = ["create_chat_router"]

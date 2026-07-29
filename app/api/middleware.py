"""FastAPI 请求追踪与端到端计时中间件。"""

import time

from fastapi import Request

from app.common.logging import get_logger
from app.observability import create_trace_id
from app.observability.metrics import now_iso


logger = get_logger("chat_service")


async def trace_and_timing_middleware(request: Request, call_next):
    """为聊天请求注入追踪编号，并记录端到端耗时响应头。"""
    if request.url.path != "/chat":
        return await call_next(request)

    trace_id = request.headers.get("X-Trace-Id") or create_trace_id()
    request_enter_perf = time.perf_counter()
    request.state.trace_id = trace_id
    request.state.request_enter_perf = request_enter_perf
    request.state.request_enter_time = now_iso()

    logger.info(
        "[trace_id=%s] [http] request_enter path=%s method=%s request_enter_time=%s",
        trace_id,
        request.url.path,
        request.method,
        request.state.request_enter_time,
    )
    try:
        response = await call_next(request)
    except Exception as exc:
        total_cost_ms = int((time.perf_counter() - request_enter_perf) * 1000)
        logger.info(
            "[trace_id=%s] [http] request_error total_cost_ms=%s error=%r",
            trace_id,
            total_cost_ms,
            exc,
        )
        raise

    total_cost_ms = int((time.perf_counter() - request_enter_perf) * 1000)
    logger.info(
        "[trace_id=%s] [http] response_end status_code=%s total_cost_ms=%s",
        trace_id,
        response.status_code,
        total_cost_ms,
    )
    response.headers["X-Trace-Id"] = trace_id
    response.headers["X-Total-Cost-Ms"] = str(total_cost_ms)
    return response

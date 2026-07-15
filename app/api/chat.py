"""聊天 API 控制器。

组装 FastAPI 应用，并把 HTTP 请求与响应适配到独立聊天应用服务。
"""

from fastapi import FastAPI, Response, Request
from contextlib import asynccontextmanager
from app.rag import (
    preview_docs,
    preview_chunks,
    search_relevant_chunks,
    rerank,
    get_rag_status,
    needs_rebuild,
    load_index_cache,
    index_cache_exists,
    rebuild_and_persist_index,
)
from app.config import Settings
from app.agent import route_agent_action
from app.observability import create_trace_id
from app.rag.retrieval import safe_float, should_refuse
from app.observability.metrics import (
    extract_llm_usage,
    read_recent_metrics,
)
from app.observability.middleware import trace_and_timing_middleware
from app.api.diagnostics import (
    DiagnosticsDependencies,
    create_diagnostics_router,
)
from app.api.responses import ChatResponseBuilder
from app.api.lifecycle import (
    StartupDependencies,
    close_agent_graph_runtime,
    initialize_agent_graph_runtime,
    initialize_rag,
)
from app.api.schemas import (
    ChatData,
    ChatRequest,
    ChatResponse,
    SearchData,
    SearchResponse,
    SearchResult,
)
from app.services.chat_service import (
    ChatServiceOutcome,
    ChatServiceRequest,
    create_default_chat_service,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时初始化 RAG/Graph，关闭时释放可选 Checkpointer。"""
    try:
        start_on_event()
        yield
    finally:
        close_agent_graph_runtime()


app = FastAPI(title='Mini Rag API', version='1.0.0', lifespan=lifespan)

app.middleware("http")(trace_and_timing_middleware)

_chat_response_builder = ChatResponseBuilder(
    response_type=ChatResponse,
    data_type=ChatData,
    safe_float=safe_float,
    extract_llm_usage=extract_llm_usage,
)
build_chat_response = _chat_response_builder.build
build_chat_response_from_result = _chat_response_builder.build_from_result
chat_service = create_default_chat_service()


def build_chat_response_from_outcome(
    outcome: ChatServiceOutcome,
    trace_id: str,
) -> ChatResponse:
    """使用现有 Response Builder 把应用服务结果适配为 API Schema。"""
    if outcome.response_config is not None:
        return build_chat_response_from_result(
            result=outcome.result,
            trace_id=trace_id,
            question=outcome.question,
            response_config=outcome.response_config,
        )

    result = outcome.result
    return build_chat_response(
        code=outcome.code,
        message=outcome.message,
        trace_id=trace_id,
        question=outcome.question,
        answer=result.get("answer", ""),
        references=result.get("references", []),
        retrieval_confidence=result.get("retrieval_confidence", 0.0),
        answer_confidence=result.get("answer_confidence", 0.0),
        need_clarification=result.get("need_clarification", False),
        status=outcome.response_status,
        clarification_question=result.get("clarification_question", ""),
        reason=result.get(
            "reason",
            "正常回答" if outcome.code == 0 else "",
        ),
        action=result.get("action", ""),
        tool_result_summary=result.get("tool_result_summary", ""),
        tool_trace=result.get("tool_trace", []),
        router_llm_used=result.get("router_llm_used"),
        decompose_llm_used=result.get("decompose_llm_used"),
        answer_llm_used=result.get("answer_llm_used"),
        llm_call_count=result.get("llm_call_count"),
    )


# =========================
# FastAPI 路由
# =========================

def start_on_event():
    """启动时校验配置、初始化 RAG，并按需预编译 Agent Graph。"""
    initialize_rag(
        StartupDependencies(
            settings=Settings,
            needs_rebuild=needs_rebuild,
            index_cache_exists=index_cache_exists,
            load_index_cache=load_index_cache,
            rebuild_and_persist_index=rebuild_and_persist_index,
        )
    )
    initialize_agent_graph_runtime(Settings)

@app.get("/health")
def health():
    """返回服务健康检查结果。"""
    return {
  "code": 0,
  "message": "success",
  "data": {
    "status": "ok"
  }
}

@app.post("/chat", response_model=ChatResponse)
def ask_api(req: ChatRequest, request: Request, response: Response):
    """读取 HTTP 上下文，调用 ChatService，并构造现有响应 Schema。"""
    trace_id = getattr(request.state, "trace_id", "") or create_trace_id()
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
    return build_chat_response_from_outcome(outcome, trace_id)

_diagnostics_router = create_diagnostics_router(
    DiagnosticsDependencies(
        settings=Settings,
        search_result=SearchResult,
        search_data=SearchData,
        search_response=SearchResponse,
        preview_docs=lambda: preview_docs(),
        preview_chunks=lambda *args, **kwargs: preview_chunks(*args, **kwargs),
        get_rag_status=lambda: get_rag_status(),
        search_relevant_chunks=lambda *args, **kwargs: search_relevant_chunks(
            *args, **kwargs
        ),
        rerank=lambda *args, **kwargs: rerank(*args, **kwargs),
        should_refuse=lambda *args, **kwargs: should_refuse(*args, **kwargs),
        safe_float=lambda value: safe_float(value),
        route_agent_action=lambda *args, **kwargs: route_agent_action(
            *args, **kwargs
        ),
        read_recent_metrics=lambda limit: read_recent_metrics(limit),
    )
)
app.include_router(_diagnostics_router)

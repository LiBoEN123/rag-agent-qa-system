"""FastAPI 应用组合根。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import create_chat_router
from app.api.diagnostics import (
    DiagnosticsDependencies,
    create_diagnostics_router,
)
from app.api.health import create_health_router
from app.api.http_logging import configure_uvicorn_access_log
from app.api.lifecycle import (
    StartupDependencies,
    aclose_agent_runtime,
    close_mcp_runtime,
    close_sql_engine_runtime,
    close_tool_runtime,
    initialize_agent_runtime,
    initialize_mcp_runtime,
    initialize_rag,
)
from app.api.middleware import trace_and_timing_middleware
from app.api.schemas import SearchData, SearchResponse, SearchResult
from app.config import get_settings, validate_startup_environment
from app.agent.router import route_agent_action
from app.observability.metrics import read_recent_metrics
from app.rag import (
    get_rag_status,
    index_cache_exists,
    load_index_cache,
    needs_rebuild,
    preview_chunks,
    preview_docs,
    rebuild_and_persist_index,
    rerank,
    search_relevant_chunks,
)
from app.rag.policy import should_refuse
from app.rag.scoring import safe_float
from app.services.chat_service import (
    ChatService,
    create_default_chat_service,
)


def get_default_diagnostics_dependencies() -> DiagnosticsDependencies:
    """组装诊断 Router 的默认依赖，不执行检索或持久化调用。"""
    return DiagnosticsDependencies(
        settings_provider=get_settings,
        search_result=SearchResult,
        search_data=SearchData,
        search_response=SearchResponse,
        preview_docs=preview_docs,
        preview_chunks=preview_chunks,
        get_rag_status=get_rag_status,
        search_relevant_chunks=search_relevant_chunks,
        rerank=rerank,
        should_refuse=should_refuse,
        safe_float=safe_float,
        route_agent_action=route_agent_action,
        read_recent_metrics=read_recent_metrics,
    )


def start_application_resources() -> None:
    """按现有顺序校验配置并初始化 RAG 与 Agent Runtime。"""
    settings = get_settings()
    validate_startup_environment(settings)
    initialize_rag(
        StartupDependencies(
            settings=settings,
            needs_rebuild=needs_rebuild,
            index_cache_exists=index_cache_exists,
            load_index_cache=load_index_cache,
            rebuild_and_persist_index=rebuild_and_persist_index,
        )
    )
    initialize_agent_runtime(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时初始化资源，关闭时按既有顺序释放 Runtime。"""
    try:
        configure_uvicorn_access_log()
        start_application_resources()
        await initialize_mcp_runtime(get_settings())
        yield
    finally:
        await close_mcp_runtime()
        close_tool_runtime()
        close_sql_engine_runtime()
        await aclose_agent_runtime()


def create_app(
    *,
    chat_service: ChatService | None = None,
    diagnostics_dependencies: DiagnosticsDependencies | None = None,
) -> FastAPI:
    """创建一套独立的 FastAPI 应用、Service 和 Router。"""
    resolved_chat_service = chat_service or create_default_chat_service()
    resolved_diagnostics = (
        diagnostics_dependencies
        or get_default_diagnostics_dependencies()
    )
    application = FastAPI(
        title="Mini Rag API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.middleware("http")(trace_and_timing_middleware)
    application.include_router(create_chat_router(resolved_chat_service))
    application.include_router(create_health_router())
    application.include_router(
        create_diagnostics_router(resolved_diagnostics)
    )
    return application


__all__ = [
    "create_app",
    "get_default_diagnostics_dependencies",
    "lifespan",
    "start_application_resources",
]

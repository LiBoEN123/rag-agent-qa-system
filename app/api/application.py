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
from app.api.ingestion import (
    IngestionAPIError, IngestionOperationsDependencies,
    create_ingestion_router, handle_ingestion_api_error,
)
from app.api.lifecycle import (
    StartupDependencies,
    aclose_agent_runtime,
    close_mcp_runtime,
    close_sql_engine_runtime,
    close_tool_executor,
    initialize_agent_runtime,
    initialize_mcp_runtime,
    initialize_rag,
)
from app.api.middleware import trace_and_timing_middleware
from app.config import AppSettings, get_settings, validate_startup_environment
from app.ingestion import IngestionPrincipalType, IngestionRequester, IngestionService
from app.observability.metrics import read_recent_metrics
from app.rag.pipeline import (
    get_rag_status,
    preview_chunks,
    preview_docs,
    rebuild_and_persist_index,
    run_diagnostic_search,
)
from app.rag.store import needs_rebuild
from app.services.chat_service import (
    ChatService,
    create_default_chat_service,
)
from app.sources import ExternalKnowledgeSourceService
from app.storage.database import init_db
from app.storage.ingestion import SQLiteCandidateRepository, SQLiteIndexRunRepository


def get_default_diagnostics_dependencies() -> DiagnosticsDependencies:
    """组装诊断 Router 的默认依赖，不执行检索或持久化调用。"""
    return DiagnosticsDependencies(
        settings_provider=get_settings,
        preview_docs=preview_docs,
        preview_chunks=preview_chunks,
        get_rag_status=get_rag_status,
        run_diagnostic_search=run_diagnostic_search,
        read_recent_metrics=read_recent_metrics,
    )


def get_default_ingestion_dependencies(
    settings: AppSettings,
) -> IngestionOperationsDependencies:
    """用同一个 application SQLite 和 requester 组装持久 operations。"""
    if settings.ingestion_admin_api_key is None:
        raise ValueError(
            "INGESTION_ADMIN_API_ENABLED=true 时必须配置 INGESTION_ADMIN_API_KEY"
        )
    requester = IngestionRequester(
        principal_id="ingestion-admin-api",
        principal_type=IngestionPrincipalType.SERVICE,
        source="api",
    )
    candidates = SQLiteCandidateRepository(settings.sqlite_db_path)
    ingestion_service = IngestionService(candidates)
    return IngestionOperationsDependencies(
        ingestion_service=ingestion_service,
        source_service=ExternalKnowledgeSourceService(
            requester=requester,
            ingestion_service=ingestion_service,
            settings=settings,
        ),
        index_runs=SQLiteIndexRunRepository(settings.sqlite_db_path),
        requester=requester,
        admin_api_key=settings.ingestion_admin_api_key,
    )


def _register_ingestion_operations(
    application: FastAPI,
    settings: AppSettings,
    dependencies: IngestionOperationsDependencies | None = None,
) -> None:
    if not settings.ingestion_admin_api_enabled or getattr(
        application.state, "ingestion_router_registered", False
    ):
        return
    resolved = dependencies or get_default_ingestion_dependencies(settings)
    application.include_router(create_ingestion_router(resolved))
    application.state.ingestion_router_registered = True


def start_application_resources(settings: AppSettings | None = None) -> None:
    """按现有顺序校验配置并初始化 RAG 与 Agent Runtime。"""
    resolved = settings or get_settings()
    validate_startup_environment(resolved)
    init_db(resolved.sqlite_db_path)
    initialize_rag(
        StartupDependencies(
            settings=resolved,
            needs_rebuild=needs_rebuild,
            rebuild_and_persist_index=rebuild_and_persist_index,
        )
    )
    initialize_agent_runtime(resolved)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时初始化资源，关闭时按既有顺序释放 Runtime。"""
    try:
        configure_uvicorn_access_log()
        settings = (
            getattr(getattr(_app, "state", None), "settings_override", None)
            or get_settings()
        )
        start_application_resources(settings)
        await initialize_mcp_runtime(settings)
        if isinstance(_app, FastAPI):
            _register_ingestion_operations(_app, settings)
        yield
    finally:
        await close_mcp_runtime()
        close_tool_executor()
        close_sql_engine_runtime()
        await aclose_agent_runtime()


def create_app(
    *,
    chat_service: ChatService | None = None,
    diagnostics_dependencies: DiagnosticsDependencies | None = None,
    settings: AppSettings | None = None,
    ingestion_dependencies: IngestionOperationsDependencies | None = None,
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
    application.add_exception_handler(IngestionAPIError, handle_ingestion_api_error)
    application.state.settings_override = settings
    application.state.ingestion_router_registered = False
    application.include_router(create_chat_router(resolved_chat_service))
    application.include_router(create_health_router())
    application.include_router(
        create_diagnostics_router(resolved_diagnostics)
    )
    if settings is not None:
        _register_ingestion_operations(
            application,
            settings,
            ingestion_dependencies,
        )
    return application


__all__ = [
    "create_app",
    "get_default_diagnostics_dependencies",
    "get_default_ingestion_dependencies",
    "lifespan",
    "start_application_resources",
]

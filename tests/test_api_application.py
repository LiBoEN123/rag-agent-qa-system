"""FastAPI 组合根、路由和生命周期隔离测试。"""

import asyncio
from dataclasses import replace

from fastapi import Response

import app.api.application as api_application
from app.agent.results import (
    build_empty_question_result,
    build_internal_error_result,
    create_agent_result,
)
from app.api.diagnostics import create_diagnostics_router
from app.api.schemas import ChatRequest
from app.contracts.chat import AgentResultStatus
from app.services.chat_models import ChatServiceOutcome


class RecordingService:
    def __init__(self, outcome: ChatServiceOutcome):
        self.outcome = outcome
        self.requests = []

    def handle(self, request):
        self.requests.append(request)
        return self.outcome


def success_outcome(answer: str = "测试回答") -> ChatServiceOutcome:
    return ChatServiceOutcome(
        http_status=200,
        code=0,
        message="success",
        question="测试问题",
        result=create_agent_result(
            status=AgentResultStatus.OK,
            answer=answer,
            reason="正常回答",
        ),
    )


def test_create_app_returns_independent_instances_and_services(monkeypatch):
    services = [
        RecordingService(success_outcome("first")),
        RecordingService(success_outcome("second")),
    ]
    monkeypatch.setattr(
        api_application,
        "create_default_chat_service",
        lambda: services.pop(0),
    )

    first_app = api_application.create_app()
    second_app = api_application.create_app()
    first_endpoint = next(
        route.endpoint for route in first_app.routes if route.path == "/chat"
    )
    second_endpoint = next(
        route.endpoint for route in second_app.routes if route.path == "/chat"
    )

    class RequestState:
        trace_id = "trace-isolation"

    class Request:
        state = RequestState()

    assert first_endpoint(
        ChatRequest(question="first"),
        Request(),
        Response(),
    ).data.answer == "first"
    assert second_endpoint(
        ChatRequest(question="second"),
        Request(),
        Response(),
    ).data.answer == "second"


def test_application_registers_middleware_and_all_routes():
    application = api_application.create_app(
        chat_service=RecordingService(success_outcome()),
    )
    routes = {
        (method, route.path)
        for route in application.routes
        for method in getattr(route, "methods", set())
        if method not in {"HEAD", "OPTIONS"}
    }

    assert {
        ("GET", "/health"),
        ("POST", "/chat"),
        ("GET", "/docs-preview"),
        ("GET", "/chunks-preview"),
        ("POST", "/search"),
        ("POST", "/kb_summary"),
        ("POST", "/decide"),
        ("GET", "/metrics-summary"),
    } <= routes
    assert any(
        item.cls.__name__ == "BaseHTTPMiddleware"
        for item in application.user_middleware
    )


def test_create_app_does_not_run_lifespan_at_import_or_construction(monkeypatch):
    calls = []
    monkeypatch.setattr(
        api_application,
        "start_application_resources",
        lambda: calls.append("start"),
    )

    application = api_application.create_app(
        chat_service=RecordingService(success_outcome()),
    )

    assert application
    assert calls == []


def test_lifespan_awaits_mcp_startup_and_closes_it_before_other_runtimes(
    monkeypatch,
):
    calls = []

    async def initialize_mcp(settings):
        assert settings is api_application.get_settings()
        calls.append("mcp_start")

    async def close_mcp():
        calls.append("mcp_close")

    async def close_agent():
        calls.append("agent_close")

    monkeypatch.setattr(api_application, "configure_uvicorn_access_log", lambda: None)
    monkeypatch.setattr(
        api_application,
        "start_application_resources",
        lambda: calls.append("application_start"),
    )
    monkeypatch.setattr(api_application, "initialize_mcp_runtime", initialize_mcp)
    monkeypatch.setattr(api_application, "close_mcp_runtime", close_mcp)
    monkeypatch.setattr(
        api_application,
        "close_tool_runtime",
        lambda: calls.append("tool_close"),
    )
    monkeypatch.setattr(
        api_application,
        "close_sql_engine_runtime",
        lambda: calls.append("sql_close"),
    )
    monkeypatch.setattr(
        api_application,
        "aclose_agent_runtime",
        close_agent,
    )

    async def run_lifespan():
        async with api_application.lifespan(object()):
            calls.append("serve")

    asyncio.run(run_lifespan())

    assert calls == [
        "application_start",
        "mcp_start",
        "serve",
        "mcp_close",
        "tool_close",
        "sql_close",
        "agent_close",
    ]


def test_main_exposes_factory_created_app():
    from app.main import app

    assert app.title == "Mini Rag API"
    assert any(route.path == "/chat" for route in app.routes)


def test_application_factory_chat_success_and_health():
    service = RecordingService(success_outcome())
    application = api_application.create_app(chat_service=service)
    chat_endpoint = next(
        route.endpoint
        for route in application.routes
        if route.path == "/chat"
    )
    health_endpoint = next(
        route.endpoint
        for route in application.routes
        if route.path == "/health"
    )

    class RequestState:
        trace_id = "trace-http"
        request_enter_time = "2026-07-29T10:00:00.000"
        request_enter_perf = 10.0

    class Request:
        state = RequestState()

    http_response = Response()
    response = chat_endpoint(
        ChatRequest(question="测试问题"),
        Request(),
        http_response,
    )
    health = health_endpoint()

    assert http_response.status_code == 200
    assert response.trace_id == "trace-http"
    assert response.data.answer == "测试回答"
    assert health == {
        "code": 0,
        "message": "success",
        "data": {"status": "ok"},
    }


def test_application_factory_clarify_empty_and_runtime_error():
    outcomes = [
        ChatServiceOutcome(
            http_status=200,
            code="NEED_CLARIFICATION",
            message="问题信息不足，需要补充后再回答",
            question="测试问题",
            result=create_agent_result(
                status=AgentResultStatus.NEED_CLARIFICATION,
                need_clarification=True,
                clarification_question="请补充对象",
            ),
            response_status="clarify",
        ),
        ChatServiceOutcome(
            http_status=400,
            code="EMPTY_QUESTION",
            message="问题不能为空，请输入你想查询的内容",
            question="",
            result=build_empty_question_result(),
            response_status="clarify",
        ),
        ChatServiceOutcome(
            http_status=500,
            code="MODEL_ERROR",
            message="处理失败，请稍后重试",
            question="测试问题",
            result=build_internal_error_result(
                RuntimeError("runtime unavailable"),
                runtime_label="ChatService Agent invocation",
                reason="工具调用或模型调用异常",
            ),
            response_status="error",
        ),
    ]

    for outcome, expected_status in zip(
        outcomes,
        (200, 400, 500),
        strict=True,
    ):
        service = RecordingService(outcome)
        application = api_application.create_app(chat_service=service)
        endpoint = next(
            route.endpoint
            for route in application.routes
            if route.path == "/chat"
        )

        class Request:
            class state:
                trace_id = "trace-status"

        response = Response()
        endpoint(
            ChatRequest(question=outcome.question),
            Request(),
            response,
        )
        assert response.status_code == expected_status

    clarify = outcomes[0]
    assert clarify.result["need_clarification"] is True
    assert clarify.result["references"] == []


def test_metrics_summary_only_has_current_chat_metric_statuses():
    dependencies = replace(
        api_application.get_default_diagnostics_dependencies(),
        read_recent_metrics=lambda limit: [
            {"status": "success", "latency_ms": 10, "llm_call_count": 1},
            {"status": "model_bad_output", "latency_ms": 20},
        ],
    )
    router = create_diagnostics_router(dependencies)
    endpoint = next(
        route.endpoint
        for route in router.routes
        if route.path == "/metrics-summary"
    )

    data = endpoint(limit=20)["data"]

    assert data["status_count"] == {
        "success": 1,
        "model_bad_output": 1,
    }
    assert {
        "success_count",
        "no_relevant_info_count",
        "tool_error_count",
        "model_error_count",
        "clarify_count",
        "internal_error_count",
    } <= set(data)
    for retired_status in (
        "model_bad_output",
        "out_of_scope",
        "empty_index",
    ):
        assert f"{retired_status}_count" not in data

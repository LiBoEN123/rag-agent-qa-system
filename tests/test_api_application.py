"""FastAPI 组合根、路由和生命周期隔离测试。"""

import asyncio
import json
from dataclasses import fields, replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from fastapi import Response

import app.api.application as api_application
from app.agent.results import (
    create_agent_result,
)
from app.api.diagnostics import create_diagnostics_router
from app.api.schemas import ChatRequest, SearchRequest, SearchResponse
from app.contracts.chat import AgentResultStatus
from app.rag import pipeline
from app.services.chat_models import ChatServiceOutcome
from app.services.chat_stream_models import ChatStreamEvent


class RecordingService:
    def __init__(self, outcome: ChatServiceOutcome):
        self.outcome = outcome
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        yield ChatStreamEvent("start", {"session_id_present": bool(request.session_id)})
        yield ChatStreamEvent("final", {"outcome": self.outcome, "total_cost_ms": 1})
        yield ChatStreamEvent("done", {"reason": "completed"})


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


def build_diagnostics_dependencies(**overrides):
    """构造不访问真实 RAG 状态或 metrics 的诊断 Router 依赖。"""
    replacements = {
        "settings_provider": lambda: SimpleNamespace(top_k=2),
        "preview_docs": lambda: [],
        "preview_chunks": lambda max_len, limit, file_name: [],
        "get_rag_status": lambda: {
            "ready": True,
            "message": "ready",
            "doc_count": 0,
            "chunk_count": 0,
        },
        "run_diagnostic_search": lambda question, top_k: {
            "status": "no_evidence",
            "message": "success",
            "reason": "未检索到结果",
            "chunks": [],
        },
        "read_recent_metrics": lambda limit: [],
    }
    replacements.update(overrides)
    return replace(
        api_application.get_default_diagnostics_dependencies(),
        **replacements,
    )


def get_diagnostics_endpoint(router, path):
    return next(route.endpoint for route in router.routes if route.path == path)


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
        route.endpoint for route in first_app.routes if route.path == "/chat/stream"
    )
    second_endpoint = next(
        route.endpoint for route in second_app.routes if route.path == "/chat/stream"
    )

    class RequestState:
        trace_id = "trace-isolation"

    class Request:
        state = RequestState()

        async def is_disconnected(self):
            return False

    async def answer(endpoint, question):
        response = await endpoint(ChatRequest(question=question), Request())
        chunks = [chunk async for chunk in response.body_iterator]
        for block in b"".join(chunks).decode().split("\n\n"):
            if "event: final\n" in block:
                data = next(
                    line[6:] for line in block.splitlines() if line.startswith("data: ")
                )
                return json.loads(data)["data"]["response"]["data"]["answer"]
        raise AssertionError("missing final")

    assert asyncio.run(answer(first_endpoint, "first")) == "first"
    assert asyncio.run(answer(second_endpoint, "second")) == "second"


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
        ("POST", "/chat/stream"),
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


def test_diagnostics_dependencies_only_expose_replaceable_runtime_boundaries():
    dependencies = build_diagnostics_dependencies()

    assert [field.name for field in fields(dependencies)] == [
        "settings_provider",
        "preview_docs",
        "preview_chunks",
        "get_rag_status",
        "run_diagnostic_search",
        "read_recent_metrics",
    ]

    def replacement(limit):
        return [{"status": "success"}]

    assert replace(
        dependencies,
        read_recent_metrics=replacement,
    ).read_recent_metrics is replacement
    assert (
        api_application.get_default_diagnostics_dependencies().run_diagnostic_search
        is pipeline.run_diagnostic_search
    )


def test_search_response_model_and_empty_question_behavior_are_unchanged():
    diagnostic = Mock()
    settings_provider = Mock()
    router = create_diagnostics_router(build_diagnostics_dependencies(
        run_diagnostic_search=diagnostic,
        settings_provider=settings_provider,
    ))
    route = next(route for route in router.routes if route.path == "/search")
    response = Response()

    result = route.endpoint(SearchRequest(question="  "), response)

    assert route.response_model is SearchResponse
    diagnostic.assert_not_called()
    settings_provider.assert_not_called()
    assert response.status_code == 400
    assert result.model_dump() == {
        "code": "EMPTY_QUESTION",
        "message": "问题不能为空",
        "data": {
            "question": "",
            "status": "error",
            "reason": "用户未输入问题",
            "results": [],
        },
    }


def test_search_empty_index_behavior_is_unchanged():
    dependencies = build_diagnostics_dependencies(
        run_diagnostic_search=lambda question, top_k: {
            "status": "empty_index",
            "message": "知识库索引为空",
            "reason": "知识库未准备完成",
            "chunks": [],
        },
    )
    endpoint = get_diagnostics_endpoint(
        create_diagnostics_router(dependencies),
        "/search",
    )
    response = Response()

    result = endpoint(SearchRequest(question="测试问题"), response)

    assert response.status_code == 503
    assert result.model_dump() == {
        "code": "EMPTY_INDEX",
        "message": "知识库索引为空",
        "data": {
            "question": "测试问题",
            "status": "empty_index",
            "reason": "知识库未准备完成",
            "results": [],
        },
    }


def test_search_no_evidence_result_mapping_is_unchanged():
    endpoint = get_diagnostics_endpoint(
        create_diagnostics_router(build_diagnostics_dependencies()),
        "/search",
    )
    response = Response()

    result = endpoint(SearchRequest(question="测试问题"), response)

    assert response.status_code == 200
    assert result.model_dump() == {
        "code": 0,
        "message": "success",
        "data": {
            "question": "测试问题",
            "status": "no_evidence",
            "reason": "未检索到结果",
            "results": [],
        },
    }


def test_search_success_shape_and_top_k_resolution_are_unchanged():
    calls = []
    chunks = [
        {
            "source_path": "docs/example.md",
            "file_name": "example.md",
            "chunk_id": "chunk-1",
            "position": 3,
            "snippet": "示例内容",
            "hybrid_score": 0.8,
            "bm25_score": 0.7,
            "vector_score": 0.9,
            "rerank_score": 0.95,
        },
        {
            "source_path": "docs/second.md",
            "file_name": "second.md",
            "chunk_id": "chunk-2",
            "snippet": "第二段内容",
            "hybrid_score": 0.6,
            "bm25_score": 0.5,
            "vector_score": 0.7,
        },
    ]

    def search(question, top_k):
        calls.append((question, top_k))
        return {
            "status": "ok",
            "message": "success",
            "reason": "检索成功",
            "chunks": chunks,
        }

    dependencies = build_diagnostics_dependencies(
        run_diagnostic_search=search,
    )
    endpoint = get_diagnostics_endpoint(
        create_diagnostics_router(dependencies),
        "/search",
    )
    response = Response()

    result = endpoint(SearchRequest(question="  测试问题  "), response)

    assert calls == [("测试问题", 2)]
    assert response.status_code == 200
    assert result.model_dump() == {
        "code": 0,
        "message": "success",
        "data": {
            "question": "测试问题",
            "status": "ok",
            "reason": "检索成功",
            "results": [
                {
                    "reference_id": "Reference 1",
                    "source_path": "docs/example.md",
                    "file_name": "example.md",
                    "chunk_id": "chunk-1",
                    "position": 3,
                    "snippet": "示例内容",
                    "score": 0.8,
                    "bm25_score": 0.7,
                    "vector_score": 0.9,
                    "rerank_score": 0.95,
                },
                {
                    "reference_id": "Reference 2",
                    "source_path": "docs/second.md",
                    "file_name": "second.md",
                    "chunk_id": "chunk-2",
                    "position": None,
                    "snippet": "第二段内容",
                    "score": 0.6,
                    "bm25_score": 0.5,
                    "vector_score": 0.7,
                    "rerank_score": 0.0,
                },
            ],
        },
    }
    endpoint(SearchRequest(question="测试问题", top_k=1), Response())
    assert calls == [("测试问题", 2), ("测试问题", 1)]
    assert all("reference_id" not in chunk for chunk in chunks)


def test_search_internal_error_behavior_is_unchanged():
    diagnostic = Mock(side_effect=RuntimeError("diagnostic failure"))
    endpoint = get_diagnostics_endpoint(
        create_diagnostics_router(build_diagnostics_dependencies(
            run_diagnostic_search=diagnostic,
        )),
        "/search",
    )
    response = Response()

    result = endpoint(SearchRequest(question="测试问题"), response)

    assert response.status_code == 500
    assert result.model_dump() == {
        "code": "INTERNAL_ERROR",
        "message": "服务器内部错误",
        "data": {
            "question": "测试问题",
            "status": "error",
            "reason": "search 接口处理异常",
            "results": [],
        },
    }


@pytest.fixture
def diagnostic_search_steps(monkeypatch):
    steps = Mock()
    steps.get_rag_status.return_value = {"ready": True, "message": "ready"}
    steps.search_relevant_chunks.return_value = []
    steps.rerank.return_value = []
    steps.should_refuse.return_value = False
    for name in ("get_rag_status", "search_relevant_chunks", "rerank", "should_refuse"):
        monkeypatch.setattr(pipeline, name, getattr(steps, name))
    return steps


def test_diagnostic_search_not_ready_skips_retrieval(diagnostic_search_steps):
    steps = diagnostic_search_steps
    steps.get_rag_status.return_value = {"ready": False, "message": "索引尚未就绪"}

    assert pipeline.run_diagnostic_search("测试问题", 2) == {
        "status": "empty_index",
        "message": "索引尚未就绪",
        "reason": "知识库未准备完成",
        "chunks": [],
    }
    assert steps.mock_calls == [call.get_rag_status()]


def test_diagnostic_search_empty_retrieval_skips_rerank(diagnostic_search_steps):
    assert pipeline.run_diagnostic_search("测试问题", 2) == {
        "status": "no_evidence",
        "message": "success",
        "reason": "未检索到结果",
        "chunks": [],
    }
    assert diagnostic_search_steps.mock_calls == [
        call.get_rag_status(),
        call.search_relevant_chunks("测试问题", 2),
    ]


def test_diagnostic_search_empty_rerank_skips_evidence_gate(diagnostic_search_steps):
    steps = diagnostic_search_steps
    retrieved = [{"vector_score": 0.9}]
    steps.search_relevant_chunks.return_value = retrieved

    assert pipeline.run_diagnostic_search("测试问题", 2) == {
        "status": "no_evidence",
        "message": "success",
        "reason": "重排后没有可用结果",
        "chunks": [],
    }
    assert steps.mock_calls == [
        call.get_rag_status(),
        call.search_relevant_chunks("测试问题", 2),
        call.rerank(retrieved, "测试问题"),
    ]


def test_diagnostic_search_refusal_preserves_safe_vector_reason(diagnostic_search_steps):
    steps = diagnostic_search_steps
    reranked = [
        {"vector_score": None},
        {"vector_score": "nan"},
        {"vector_score": "0.123456"},
        {"vector_score": 0.9},
    ]
    steps.search_relevant_chunks.return_value = reranked
    steps.rerank.return_value = reranked
    steps.should_refuse.return_value = True

    assert pipeline.run_diagnostic_search("测试问题", 3) == {
        "status": "no_evidence",
        "message": "success",
        "reason": "证据分数低于阈值：max_vector=0.1235",
        "chunks": [],
    }
    steps.should_refuse.assert_called_once_with(reranked[:3])


def test_diagnostic_search_success_preserves_order_and_chunks(diagnostic_search_steps):
    steps = diagnostic_search_steps
    retrieved = [{"chunk_id": "first"}, {"chunk_id": "second"}, {"chunk_id": "third"}]
    reranked = [retrieved[2], retrieved[0], retrieved[1]]
    original = [dict(chunk) for chunk in retrieved]
    steps.search_relevant_chunks.return_value = retrieved
    steps.rerank.return_value = reranked

    result = pipeline.run_diagnostic_search("测试问题", 2)

    assert result == {
        "status": "ok",
        "message": "success",
        "reason": "检索成功",
        "chunks": reranked[:2],
    }
    assert result["chunks"][0] is reranked[0]
    assert result["chunks"][1] is reranked[1]
    assert retrieved == original
    assert all("reference_id" not in chunk for chunk in reranked)
    assert steps.mock_calls == [
        call.get_rag_status(),
        call.search_relevant_chunks("测试问题", 2),
        call.rerank(retrieved, "测试问题"),
        call.should_refuse(reranked[:2]),
    ]


def test_decide_behavior_uses_the_deterministic_router_unchanged():
    endpoint = get_diagnostics_endpoint(
        create_diagnostics_router(build_diagnostics_dependencies()),
        "/decide",
    )
    response = Response()

    result = endpoint(ChatRequest(question="你能做什么？"), response)

    assert result == {
        "code": 0,
        "message": "success",
        "data": {
            "question": "你能做什么？",
            "decision": {
                "decision": "direct_answer",
                "action": "direct_answer",
                "reason": "用户询问系统能力，可以直接回答",
                "router_source": "rule_based",
                "router_confidence": 1.0,
            },
        },
    }


def test_default_diagnostics_reads_current_pipeline_state(monkeypatch):
    endpoint = get_diagnostics_endpoint(
        create_diagnostics_router(api_application.get_default_diagnostics_dependencies()),
        "/kb_summary",
    )
    documents = [{"file_name": "current.md"}]
    monkeypatch.setattr(pipeline, "docs_registry", documents)
    monkeypatch.setattr(pipeline, "chunks", [{"text": "current"}])
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "RAG_INIT_MESSAGE", "OK")
    monkeypatch.setattr(pipeline, "INDEX_SOURCE", "chroma")

    assert endpoint()["data"] == {
        "ready": True,
        "message": "OK",
        "doc_count": 1,
        "chunk_count": 1,
        "index_source": "chroma",
        "file_names": ["current.md"],
        "documents": documents,
    }


def test_metrics_summary_excludes_legacy_usage_from_historical_records():
    endpoint = get_diagnostics_endpoint(
        create_diagnostics_router(build_diagnostics_dependencies(
            read_recent_metrics=lambda limit: [{
                "status": "success", "latency_ms": 12, "llm_call_count": 2,
                "router_llm_used": False, "decompose_llm_used": True,
                "answer_llm_used": True, "decompose_cost_ms": 7,
            }],
        )),
        "/metrics-summary",
    )
    data = endpoint(limit=20)["data"]
    assert "decompose_llm_used_count" not in data
    assert "decompose_cost_ms" not in data
    assert data["answer_llm_used_count"] == 1
    assert data["router_llm_used_count"] == 0
    assert data["avg_llm_call_count"] == 2
    assert data["avg_latency_ms"] == 12


def test_start_application_resources_wires_incremental_sync(monkeypatch):
    settings = api_application.get_settings()
    calls = []
    monkeypatch.setattr(api_application, "validate_startup_environment", lambda _: None)
    monkeypatch.setattr(api_application, "init_db", lambda _: None)
    monkeypatch.setattr(api_application, "initialize_agent_runtime", lambda _: None)
    monkeypatch.setattr(api_application, "needs_rebuild", lambda _: (False, "unchanged"))

    def sync(**kwargs):
        calls.append(kwargs)
        return True, "ok"

    monkeypatch.setattr(api_application, "rebuild_and_persist_index", sync)
    api_application.start_application_resources(settings)

    assert calls == [{
        "data_dir": str(settings.rag_file_path),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "force_rebuild": settings.auto_rebuild_on_startup,
    }]


def test_create_app_does_not_run_lifespan_at_import_or_construction(monkeypatch):
    calls = []
    monkeypatch.setattr(
        api_application,
        "start_application_resources",
        lambda *_args: calls.append("start"),
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
        lambda *_args: calls.append("application_start"),
    )
    monkeypatch.setattr(api_application, "initialize_mcp_runtime", initialize_mcp)
    monkeypatch.setattr(api_application, "close_mcp_runtime", close_mcp)
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
        "agent_close",
    ]


def test_main_exposes_factory_created_app():
    from app.main import app

    assert app.title == "Mini Rag API"
    assert any(route.path == "/chat/stream" for route in app.routes)


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

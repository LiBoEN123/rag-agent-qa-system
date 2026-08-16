"""聊天 API、health 与应用服务适配的兼容回归。"""

from dataclasses import replace

import pytest
from fastapi import Response

from app.agent.results import build_internal_error_result
from app.api.chat import create_chat_router
from app.api.health import create_health_router
from app.api.responses import build_chat_response_from_outcome
from app.api.schemas import ChatRequest
from app.services.chat_models import ChatServiceOutcome
from app.services.chat_service import (
    ChatService,
    get_default_chat_service_dependencies,
)
from app.storage.session import clear_session, get_history


@pytest.fixture(autouse=True)
def use_temporary_sqlite_database(settings_env, tmp_path):
    """让每个 API 测试使用独立的临时 SQLite 数据库。"""
    settings_env(sqlite_db_path=tmp_path / "api.sqlite")


class DummyState:
    pass


class DummyRequest:
    def __init__(self):
        self.state = DummyState()


def build_chat_service(**overrides) -> ChatService:
    """构造不写生产 metrics 的显式 ChatService。"""
    dependencies = replace(
        get_default_chat_service_dependencies(),
        log_and_write_metrics=lambda record: None,
        **overrides,
    )
    return ChatService(dependencies)


def ask_api(
    service: ChatService,
    request: ChatRequest,
    *,
    http_request: DummyRequest | None = None,
    response: Response | None = None,
):
    """调用真实 Chat Router 端点，同时保持依赖由 Router 闭包持有。"""
    endpoint = create_chat_router(service).routes[0].endpoint
    return endpoint(
        request,
        http_request or DummyRequest(),
        response or Response(),
    )


def test_health():
    """验证 /health 接口返回成功状态且不依赖外部资源。"""
    response = create_health_router().routes[0].endpoint()

    assert response == {
        "code": 0,
        "message": "success",
        "data": {"status": "ok"},
    }


def test_internal_error_response_adapter_does_not_expose_runtime_exception():
    sensitive_message = (
        "/home/test-user/project/checkpoints.sqlite: "
        "model request failed with api_key=sk-test-secret"
    )
    result = build_internal_error_result(
        RuntimeError(sensitive_message),
        runtime_label="LangGraph Agent",
        trace_step="agent_langgraph_error",
    )
    outcome = ChatServiceOutcome(
        http_status=200,
        code="INTERNAL_ERROR",
        message="服务器内部错误",
        question="测试异常",
        result=result,
        response_status="error",
    )

    response = build_chat_response_from_outcome(
        outcome,
        trace_id="trace-error",
    )
    body = response.model_dump()
    response_text = response.model_dump_json()

    assert "Agent 内部执行异常，请稍后重试" in response_text
    assert "/home/test-user/project" not in response_text
    assert "checkpoints.sqlite" not in response_text
    assert "sk-test-secret" not in response_text
    assert "RuntimeError" not in response_text
    assert "model request failed" not in response_text
    assert body["data"]["status"] == "error"
    assert body["data"]["tool_trace"] == [
        {
            "step": "agent_langgraph_error",
            "status": "error",
            "error": "internal_error",
        }
    ]


def test_chat_langgraph_internal_error_result_returns_http_500():
    sensitive_message = (
        "/home/test-user/project/checkpoints.sqlite: "
        "model request failed with api_key=sk-test-secret"
    )
    runtime_calls = []

    def fake_run_agent(**kwargs):
        runtime_calls.append(kwargs)
        return build_internal_error_result(
            RuntimeError(sensitive_message),
            runtime_label="LangGraph Agent",
            trace_step="agent_langgraph_error",
        )

    service = build_chat_service(
        rewrite_query_with_history=lambda question, history, trace_id="": {
            "original_question": question,
            "rewritten_question": question,
            "history_used": False,
            "rewritten": False,
            "need_check": False,
            "need_rewrite": False,
            "reason": "no rewrite",
            "rewrite_llm_used": False,
            "error": "",
        },
        run_agent=fake_run_agent,
    )
    http_response = Response()
    response_body = ask_api(
        service,
        ChatRequest(question="触发 LangGraph 内部异常"),
        response=http_response,
    )
    payload = response_body.model_dump()
    response_text = response_body.model_dump_json()

    assert len(runtime_calls) == 1
    assert "decision" not in runtime_calls[0]
    assert http_response.status_code == 500
    assert payload["code"] == "INTERNAL_ERROR"
    assert payload["data"]["status"] == "error"
    assert payload["data"]["reason"] == "Agent 内部执行异常，请稍后重试"
    assert {
        "step": "agent_langgraph_error",
        "status": "error",
        "error": "internal_error",
    } in payload["data"]["tool_trace"]
    assert "/home/test-user/project" not in response_text
    assert "checkpoints.sqlite" not in response_text
    assert "sk-test-secret" not in response_text
    assert "RuntimeError" not in response_text
    assert "model request failed" not in response_text


def test_chat_uses_rewritten_question_for_follow_up():
    """验证 /chat 第二轮追问会用改写后的问题进入 LangGraph。"""
    session_id = "api_follow_up_session"
    clear_session(session_id)
    rewrite_calls = []

    def fake_rewrite(question, history, trace_id=""):
        rewrite_calls.append({"question": question, "history": list(history)})
        if question == "那休学创业呢？":
            return {
                "original_question": question,
                "rewritten_question": (
                    "普通本科四年制专业中，休学创业学生的最长学习年限是多少？"
                ),
                "history_used": True,
                "rewritten": True,
                "reason": "测试改写",
                "rewrite_llm_used": False,
                "error": "",
            }
        return {
            "original_question": question,
            "rewritten_question": question,
            "history_used": False,
            "rewritten": False,
            "reason": "no history",
            "rewrite_llm_used": False,
            "error": "",
        }

    agent_questions = []

    def fake_run_agent(**kwargs):
        question = kwargs["effective_question"]
        agent_questions.append(question)
        return {
            "status": "OK",
            "answer": f"回答：{question}",
            "references": [],
            "retrieval_confidence": 0.7,
            "answer_confidence": 0.9,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "测试通过",
            "action": "agent",
            "tool_result_summary": "测试",
            "tool_trace": [
                {
                    "step": "llm_usage",
                    "router_llm_used": False,
                    "decompose_llm_used": False,
                    "answer_llm_used": False,
                    "llm_call_count": 0,
                }
            ],
        }

    service = build_chat_service(
        rewrite_query_with_history=fake_rewrite,
        run_agent=fake_run_agent,
    )
    ask_api(
        service,
        ChatRequest(
            session_id=session_id,
            question="普通本科四年制专业最长学习年限是多少？",
            top_k=5,
        ),
    )
    second_response = ask_api(
        service,
        ChatRequest(
            session_id=session_id,
            question="那休学创业呢？",
            top_k=5,
        ),
    )

    body = second_response.model_dump()
    rewritten_question = (
        "普通本科四年制专业中，休学创业学生的最长学习年限是多少？"
    )
    assert rewrite_calls[1]["question"] == "那休学创业呢？"
    assert rewrite_calls[1]["history"][0] == {
        "role": "user",
        "content": "普通本科四年制专业最长学习年限是多少？",
    }
    assert agent_questions[-1] == rewritten_question
    assert body["data"]["question"] == "那休学创业呢？"
    assert body["data"]["tool_trace"][0]["step"] == "query_rewrite"
    assert body["data"]["tool_trace"][0]["original_question"] == "那休学创业呢？"
    assert (
        body["data"]["tool_trace"][0]["rewritten_question"]
        == rewritten_question
    )
    assert body["data"]["tool_trace"][0]["rewritten"] is True
    clear_session(session_id)


def test_chat_uses_agent_runtime_once_and_keeps_rewrite_api_protocol():
    """唯一 Runtime 只调用一次，且保留改写和 API 协议。"""
    calls = {"rewrite": 0, "agent": 0}
    runtime_kwargs: dict = {}

    def fake_rewrite(question, history, trace_id=""):
        calls["rewrite"] += 1
        return {
            "original_question": question,
            "rewritten_question": "改写后的问题",
            "history_used": True,
            "rewritten": True,
            "need_check": True,
            "need_rewrite": True,
            "reason": "测试改写",
            "rewrite_llm_used": False,
            "error": "",
        }

    def fake_agent_runner(**kwargs):
        calls["agent"] += 1
        runtime_kwargs.update(kwargs)
        return {
            "status": "OK",
            "answer": "agent answer",
            "references": [],
            "retrieval_confidence": 0.0,
            "answer_confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "ok",
            "action": "direct_answer",
            "tool_result_summary": "未调用工具",
            "tool_trace": [
                {
                    "step": "llm_usage",
                    "router_llm_used": False,
                    "decompose_llm_used": False,
                    "answer_llm_used": False,
                    "llm_call_count": 0,
                }
            ],
            "router_llm_used": False,
            "decompose_llm_used": False,
            "answer_llm_used": False,
            "llm_call_count": 0,
            "timings": {},
        }

    service = build_chat_service(
        rewrite_query_with_history=fake_rewrite,
        run_agent=fake_agent_runner,
    )
    response = ask_api(
        service,
        ChatRequest(
            session_id="runtime-switch-session",
            question="原始问题",
            top_k=4,
            decompose_top_k=8,
        ),
    )
    body = response.model_dump()

    assert calls == {"rewrite": 1, "agent": 1}
    assert "question" not in runtime_kwargs
    assert runtime_kwargs["original_question"] == "原始问题"
    assert runtime_kwargs["effective_question"] == "改写后的问题"
    assert runtime_kwargs["top_k"] == 4
    assert runtime_kwargs["decompose_top_k"] == 8
    assert "decision" not in runtime_kwargs
    assert runtime_kwargs["session_id"] == "runtime-switch-session"
    assert body["code"] == 0
    assert body["data"]["answer"] == "agent answer"
    assert body["data"]["question"] == "原始问题"
    assert body["data"]["tool_trace"][0]["step"] == "query_rewrite"
    assert get_history("runtime-switch-session") == [
        {"role": "user", "content": "原始问题"},
        {"role": "assistant", "content": "agent answer"},
    ]

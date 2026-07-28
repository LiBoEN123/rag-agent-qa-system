from dataclasses import replace

import pytest

import app.api.chat as chat_api
from app.agent.runtime import build_internal_error_result
from app.services.chat_service import (
    CHAT_RESULT_RESPONSE_MAP,
    ChatService,
    ChatServiceOutcome,
    get_default_chat_service_dependencies,
)
from app.storage.session import clear_session, get_history


@pytest.fixture(autouse=True)
def use_temporary_sqlite_database(monkeypatch, tmp_path):
    """让每个 API 测试使用独立的临时 SQLite 数据库。"""
    monkeypatch.setattr(
        chat_api.Settings,
        "SQLITE_DB_PATH",
        tmp_path / "api.sqlite",
        raising=False,
    )


def patch_app(monkeypatch):
    """隔离索引、路由和指标依赖，让 API 测试只关注响应结构。"""
    monkeypatch.setattr(chat_api.Settings, "validate", lambda : None, raising=False)
    monkeypatch.setattr(chat_api.Settings, "RAG_FILE_PATH", "data", raising=False)
    monkeypatch.setattr(chat_api.Settings, "CHUNK_SIZE", 800, raising=False)
    monkeypatch.setattr(chat_api.Settings, "CHUNK_OVERLAP", 120, raising=False)
    monkeypatch.setattr(chat_api.Settings, "AUTO_REBUILD_ON_STARTUP", False, raising=False)
    monkeypatch.setattr(chat_api, "needs_rebuild", lambda data_dir: (False, "test"), raising=False)
    monkeypatch.setattr(chat_api, "index_cache_exists", lambda : True, raising=False)
    monkeypatch.setattr(chat_api, "load_index_cache", lambda : (True, "test"), raising=False)


def install_chat_service(monkeypatch, **overrides):
    """用显式注入依赖安装一次隔离外部副作用的 ChatService。"""
    dependencies = replace(
        get_default_chat_service_dependencies(),
        log_and_write_metrics=lambda record: None,
        **overrides,
    )
    monkeypatch.setattr(chat_api, "chat_service", ChatService(dependencies))


class DummyState:
    pass


class DummyRequest:
    def __init__(self):
        self.state = DummyState()

def test_health(monkeypatch):
    """验证 /health 接口返回成功状态。"""
    patch_app(monkeypatch)
    resp = chat_api.health()

    assert resp["code"] == 0
    assert resp["data"]["status"] == "ok"


def test_internal_error_response_builder_does_not_expose_runtime_exception():
    sensitive_message = (
        "/home/test-user/project/checkpoints.sqlite: "
        "model request failed with api_key=sk-test-secret"
    )
    result = build_internal_error_result(
        RuntimeError(sensitive_message),
        {"action": "direct_answer"},
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
        response_config=CHAT_RESULT_RESPONSE_MAP["INTERNAL_ERROR"],
    )

    response = chat_api.build_chat_response_from_outcome(outcome, "trace-error")
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


def test_chat_langgraph_internal_error_result_returns_http_500(monkeypatch):
    sensitive_message = (
        "/home/test-user/project/checkpoints.sqlite: "
        "model request failed with api_key=sk-test-secret"
    )
    decision = {
        "decision": "direct_answer",
        "action": "direct_answer",
        "tool_name": "",
        "tool_args": {},
        "reason": "直接回答",
        "router_source": "rule_based",
        "router_llm_used": False,
    }
    runtime_calls = []

    def fake_run_agent_langgraph(**kwargs):
        runtime_calls.append(kwargs)
        return build_internal_error_result(
            RuntimeError(sensitive_message),
            kwargs["decision"],
            runtime_label="LangGraph Agent",
            trace_step="agent_langgraph_error",
        )

    patch_app(monkeypatch)
    install_chat_service(
        monkeypatch,
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
        route_agent_action=lambda question, top_k, trace_id="": decision,
        run_agent_langgraph=fake_run_agent_langgraph,
    )

    http_response = chat_api.Response()
    response_body = chat_api.ask_api(
        chat_api.ChatRequest(question="触发 LangGraph 内部异常"),
        DummyRequest(),
        http_response,
    )
    payload = response_body.model_dump()
    response_text = response_body.model_dump_json()

    assert len(runtime_calls) == 1
    assert runtime_calls[0]["decision"] is decision
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


def test_chat_uses_rewritten_question_for_follow_up(monkeypatch):
    """验证 /chat 第二轮追问会用改写后的问题进入 Router 和 Agent。"""
    session_id = "api_follow_up_session"
    clear_session(session_id)

    rewrite_calls = []

    def fake_rewrite(question, history, trace_id=""):
        rewrite_calls.append({"question": question, "history": list(history)})
        if question == "那休学创业呢？":
            return {
                "original_question": question,
                "rewritten_question": "普通本科四年制专业中，休学创业学生的最长学习年限是多少？",
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

    routed_questions = []

    def fake_route(question, top_k, trace_id=""):
        routed_questions.append(question)
        return {
            "decision": "use_agent",
            "action": "agent",
            "tool_name": "",
            "tool_args": {},
            "reason": "交给 Agent",
            "router_source": "rule_based",
            "router_llm_used": False,
        }

    agent_questions = []

    def fake_run_agent_langgraph(**kwargs):
        """模拟 LangGraph 并记录收到的改写后问题。"""
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

    patch_app(monkeypatch)
    install_chat_service(
        monkeypatch,
        rewrite_query_with_history=fake_rewrite,
        route_agent_action=fake_route,
        run_agent_langgraph=fake_run_agent_langgraph,
    )

    chat_api.ask_api(
        chat_api.ChatRequest(
            session_id=session_id,
            question="普通本科四年制专业最长学习年限是多少？",
            top_k=5,
        ),
        DummyRequest(),
        chat_api.Response(),
    )
    second_resp = chat_api.ask_api(
        chat_api.ChatRequest(
            session_id=session_id,
            question="那休学创业呢？",
            top_k=5,
        ),
        DummyRequest(),
        chat_api.Response(),
    )

    body = second_resp.model_dump()
    rewritten_question = "普通本科四年制专业中，休学创业学生的最长学习年限是多少？"

    assert rewrite_calls[1]["question"] == "那休学创业呢？"
    assert rewrite_calls[1]["history"][0]["role"] == "user"
    assert rewrite_calls[1]["history"][0]["content"] == "普通本科四年制专业最长学习年限是多少？"
    assert routed_questions[-1] == rewritten_question
    assert agent_questions[-1] == rewritten_question
    assert body["data"]["question"] == "那休学创业呢？"
    assert body["data"]["tool_trace"][0]["step"] == "query_rewrite"
    assert body["data"]["tool_trace"][0]["original_question"] == "那休学创业呢？"
    assert body["data"]["tool_trace"][0]["rewritten_question"] == rewritten_question
    assert body["data"]["tool_trace"][0]["rewritten"] is True

    clear_session(session_id)


def test_chat_uses_langgraph_once_and_keeps_rewrite_route_api_protocol(monkeypatch):
    """唯一 Runtime 只调用一次，且保留改写、路由和 API 协议。"""
    patch_app(monkeypatch)

    calls = {
        "rewrite": 0,
        "route": 0,
        "langgraph": 0,
    }
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

    decision = {
        "decision": "direct_answer",
        "action": "direct_answer",
        "tool_name": "",
        "tool_args": {},
        "reason": "测试直接回答",
        "router_source": "rule_based",
        "router_llm_used": False,
    }

    def fake_route(question, top_k, trace_id=""):
        calls["route"] += 1
        assert question == "改写后的问题"
        return decision

    def fake_langgraph_runner(**kwargs):
        calls["langgraph"] += 1
        runtime_kwargs.update(kwargs)
        return {
            "status": "OK",
            "answer": "langgraph answer",
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
        }

    install_chat_service(
        monkeypatch,
        rewrite_query_with_history=fake_rewrite,
        route_agent_action=fake_route,
        run_agent_langgraph=fake_langgraph_runner,
    )

    response = chat_api.ask_api(
        chat_api.ChatRequest(
            session_id="runtime-switch-session",
            question="原始问题",
            top_k=4,
            decompose_top_k=8,
        ),
        DummyRequest(),
        chat_api.Response(),
    )
    body = response.model_dump()

    assert calls == {
        "rewrite": 1,
        "route": 1,
        "langgraph": 1,
    }
    assert "question" not in runtime_kwargs
    assert runtime_kwargs["original_question"] == "原始问题"
    assert runtime_kwargs["effective_question"] == "改写后的问题"
    assert runtime_kwargs["top_k"] == 4
    assert runtime_kwargs["decompose_top_k"] == 8
    assert runtime_kwargs["decision"] is decision
    assert runtime_kwargs["session_id"] == "runtime-switch-session"
    assert body["code"] == 0
    assert body["data"]["answer"] == "langgraph answer"
    assert body["data"]["question"] == "原始问题"
    assert body["data"]["tool_trace"][0]["step"] == "query_rewrite"
    assert get_history("runtime-switch-session") == [
        {"role": "user", "content": "原始问题"},
        {"role": "assistant", "content": "langgraph answer"},
    ]

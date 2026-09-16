"""聊天流式多轮会话与 health 的隔离回归。"""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agent.streaming import AgentRuntimeStreamEvent
from app.api.chat import create_chat_router
from app.api.health import create_health_router
from app.api.schemas import ChatRequest
from app.services.chat_service import ChatService, get_default_chat_service_dependencies
from app.storage.memory import clear_session_sqlite, get_history_sqlite


@pytest.fixture(autouse=True)
def use_temporary_sqlite_database(settings_env, tmp_path):
    settings_env(sqlite_db_path=tmp_path / "api.sqlite")


def build_chat_service(**overrides):
    return ChatService(
        replace(
            get_default_chat_service_dependencies(),
            log_and_write_metrics=lambda record: None,
            **overrides,
        )
    )


def ask_stream(service, request):
    class HttpRequest:
        state = SimpleNamespace()

        async def is_disconnected(self):
            return False

    async def consume():
        response = (
            await create_chat_router(service).routes[0].endpoint(request, HttpRequest())
        )
        chunks = [chunk async for chunk in response.body_iterator]
        for block in b"".join(chunks).decode().split("\n\n"):
            if "event: final\n" in block:
                data = next(
                    line[6:] for line in block.splitlines() if line.startswith("data: ")
                )
                return json.loads(data)["data"]["response"]
        raise AssertionError("missing final")

    return asyncio.run(consume())


def test_health():
    assert create_health_router().routes[0].endpoint() == {
        "code": 0,
        "message": "success",
        "data": {"status": "ok"},
    }


def test_chat_uses_rewritten_question_for_follow_up():
    """验证流式第二轮追问通过真实 SQLite 历史改写后进入 Runtime。"""
    session_id = "api_follow_up_session"
    clear_session_sqlite(session_id)
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

    async def fake_stream_agent(**kwargs):
        question = kwargs["effective_question"]
        agent_questions.append(question)
        yield AgentRuntimeStreamEvent(
            "result",
            {
                "result": {
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
                    "router_llm_used": False,
                    "decompose_llm_used": False,
                    "answer_llm_used": False,
                    "llm_call_count": 0,
                    "timings": {},
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
            },
        )

    service = build_chat_service(
        rewrite_query_with_history=fake_rewrite,
        stream_agent=fake_stream_agent,
    )
    ask_stream(
        service,
        ChatRequest(
            session_id=session_id,
            question="普通本科四年制专业最长学习年限是多少？",
            top_k=5,
        ),
    )
    second_response = ask_stream(
        service,
        ChatRequest(
            session_id=session_id,
            question="那休学创业呢？",
            top_k=5,
        ),
    )

    body = second_response
    rewritten_question = "普通本科四年制专业中，休学创业学生的最长学习年限是多少？"
    assert len(agent_questions) == len(rewrite_calls) == 2
    assert [message["role"] for message in get_history_sqlite(session_id)] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert body["code"] == 0
    assert body["data"]["status"] == "ok"
    assert rewrite_calls[1]["question"] == "那休学创业呢？"
    assert rewrite_calls[1]["history"][0] == {
        "role": "user",
        "content": "普通本科四年制专业最长学习年限是多少？",
    }
    assert agent_questions[-1] == rewritten_question
    assert body["data"]["question"] == "那休学创业呢？"
    assert body["data"]["tool_trace"][0]["step"] == "query_rewrite"
    assert body["data"]["tool_trace"][0]["original_question"] == "那休学创业呢？"
    assert body["data"]["tool_trace"][0]["rewritten_question"] == rewritten_question
    assert body["data"]["tool_trace"][0]["rewritten"] is True
    clear_session_sqlite(session_id)

import app.main as main
from app.session_store import clear_session


def patch_app(monkeypatch):
    """隔离索引、路由和指标依赖，让 API 测试只关注响应结构。"""
    monkeypatch.setattr(main.Settings, "validate", lambda : None, raising=False)
    monkeypatch.setattr(main.Settings, "RAG_FILE_PATH", "data", raising=False)
    monkeypatch.setattr(main.Settings, "CHUNK_SIZE", 800, raising=False)
    monkeypatch.setattr(main.Settings, "CHUNK_OVERLAP", 120, raising=False)
    monkeypatch.setattr(main.Settings, "AUTO_REBUILD_ON_STARTUP", False, raising=False)
    monkeypatch.setattr(main, "needs_rebuild", lambda data_dir: (False, "test"), raising=False)
    monkeypatch.setattr(main, "index_cache_exists", lambda : True, raising=False)
    monkeypatch.setattr(main, "load_index_cache", lambda : (True, "test"), raising=False)
    monkeypatch.setattr(main, "log_and_write_metrics", lambda record: None, raising=False)


class DummyState:
    pass


class DummyRequest:
    def __init__(self):
        self.state = DummyState()

def test_health(monkeypatch):
    """验证 /health 接口返回成功状态。"""
    patch_app(monkeypatch)
    resp = main.health()

    assert resp["code"] == 0
    assert resp["data"]["status"] == "ok"

def test_chat_basic(monkeypatch):
    """验证 /chat 在 mock Agent 结果下返回标准成功响应。"""
    monkeypatch.setattr(
        main,
        "get_rag_status",
        lambda :{
            "ready": True,
            "message":"ok",
            "doc_count": 1,
            "chunk_count": 1
        }
    )

    monkeypatch.setattr(
        main,
        "route_agent_action",
        lambda question, top_k, trace_id="": {
            "decision": "use_tool",
            "action": "search_knowledge_base",
            "tool_name": "search_knowledge_base",
            "reason": "测试路由",
        }
    )

    monkeypatch.setattr(
        main,
        "run_agent_once",
        lambda *args, **kwargs: {
            "status": "OK",
            "answer": "测试回答",
            "references": [],
            "retrieval_confidence": 0.7,
            "answer_confidence": 0.9,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "测试通过",
            "action": "search_knowledge_base",
            "tool_result_summary": "测试",
            "tool_trace": [
                {
                    "step": "llm_usage",
                    "router_llm_used": False,
                    "decompose_llm_used": False,
                    "answer_llm_used": True,
                    "llm_call_count": 1,
                }
            ],
        }
    )

    patch_app(monkeypatch)
    resp = main.ask_api(
        main.ChatRequest(
            question="DevHub 支持哪些 macOS 版本？",
            top_k=5
        ),
        DummyRequest(),
        main.Response()
    )

    body = resp.model_dump()
    assert body["code"] == 0
    assert body["message"] == "success"
    assert body["data"]["answer"] == "测试回答"
    assert body["data"]["references"] == []
    assert body["data"]["retrieval_confidence"] == 0.7
    assert body["data"]["answer_confidence"] == 0.9
    assert body["data"]["router_llm_used"] is False
    assert body["data"]["decompose_llm_used"] is False
    assert body["data"]["answer_llm_used"] is True
    assert body["data"]["llm_call_count"] == 1


def test_chat_uses_rewritten_question_for_follow_up(monkeypatch):
    """验证 /chat 第二轮追问会用改写后的问题进入 Router 和 Agent。"""
    session_id = "api_follow_up_session"
    clear_session(session_id)

    monkeypatch.setattr(
        main,
        "get_rag_status",
        lambda: {
            "ready": True,
            "message": "ok",
            "doc_count": 1,
            "chunk_count": 1,
        },
    )

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
            "decision": "use_knowledge_base",
            "action": "search_knowledge_base",
            "tool_name": "search_knowledge_base",
            "tool_args": {"query": question, "top_k": top_k},
            "reason": "测试路由",
            "router_source": "rule_based",
            "router_llm_used": False,
        }

    agent_questions = []

    def fake_run_agent_once(write_bad_cases, question, *args, **kwargs):
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
            "action": "search_knowledge_base",
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

    monkeypatch.setattr(main, "rewrite_query_with_history", fake_rewrite)
    monkeypatch.setattr(main, "route_agent_action", fake_route)
    monkeypatch.setattr(main, "run_agent_once", fake_run_agent_once)

    patch_app(monkeypatch)

    main.ask_api(
        main.ChatRequest(
            session_id=session_id,
            question="普通本科四年制专业最长学习年限是多少？",
            top_k=5,
        ),
        DummyRequest(),
        main.Response(),
    )
    second_resp = main.ask_api(
        main.ChatRequest(
            session_id=session_id,
            question="那休学创业呢？",
            top_k=5,
        ),
        DummyRequest(),
        main.Response(),
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










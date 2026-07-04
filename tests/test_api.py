import app.main as main


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












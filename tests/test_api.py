from fastapi.testclient import TestClient
import app.main as main

def make_client(monkeypatch):
    monkeypatch.setattr(main.Settings, "validate", lambda : None, raising=False)
    monkeypatch.setattr(main.Settings, "RAG_FILE_PATH", "data", raising=False)
    monkeypatch.setattr(main.Settings, "CHUNK_SIZE", 800, raising=False)
    monkeypatch.setattr(main.Settings, "CHUNK_OVERLAP", 120, raising=False)
    monkeypatch.setattr(main, "init_rag", lambda *args, **kwargs:None)

    return TestClient(main.app)

def test_health(monkeypatch):
    with make_client(monkeypatch) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["code"] == 0
    assert resp.json()["data"]["status"] == "ok"

def test_chat_basic(monkeypatch):
    fake_chunks = [
        {
        "reference_id": "Reference 1",
        "source_file": "CHANGELOG.zh.md",
        "chunk_id": "CHANGELOG.zh_chunk_5",
        "text": "[标题路径] [v2.0.0](https://github.com/jaywcjlove/DevHub/releases/tag/v2.0.0)\n\n1. feat: 添加对 macOS 26 的支持\n2. fix: 修复一些已知问题",
        "score": 0.9693357944488525,
        "bm25_score": 11.312523933418605,
        "vector_score": 0.5419507026672363
      }
    ]

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

    monkeypatch.setattr(main, "search_relevant_chunks", lambda question, top_k:fake_chunks)

    monkeypatch.setattr(
        main,
        "run_agent_once",
        lambda *args, **kwargs: {
            "status": "OK",
            "answer": "测试回答",
            "references": [],
            "confidence": 0.9,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "测试通过",
            "action": "search_knowledge_base",
            "tool_result_summary": "测试",
            "tool_trace": []
        }
    )

    with make_client(monkeypatch) as client:
        resp = client.post(
            "/chat",
            json={
                "question": "DevHub 支持哪些 macOS 版本？",
                "top_k": 5
                }
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["message"] == "success"
    assert body["data"]["answer"] == "DevHub 最低要求 macOS 14.0，并且 v2.0.0 版本添加了对 macOS 26 的支持。"
    assert len(body["data"]["references"]) == 1
    assert body["data"]["references"][0]["reference_id"] == "Reference 1"



















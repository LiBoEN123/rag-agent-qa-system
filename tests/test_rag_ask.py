import app.rag as rag


def make_chunk(**overrides):
    chunk = {
        "reference_id": "Reference 1",
        "file_name": "test.md",
        "source_path": "/tmp/test.md",
        "chunk_id": 1,
        "position": 1,
        "text": "测试内容",
        "snippet": "测试内容",
        "bm25_score": 0.0,
        "vector_score": 0.1,
        "hybrid_score": 0.1,
        "rerank_score": 0.1,
    }
    chunk.update(overrides)
    return chunk


def test_ask_skips_decompose_and_answer_llm_when_evidence_is_weak(monkeypatch):
    """验证普通短问题不调用拆分模型，证据不足时也不调用最终回答模型。"""
    calls = {"decompose": 0, "answer": 0}

    def fake_decompose(question):
        calls["decompose"] += 1
        raise AssertionError("普通短问题不应调用拆分模型")

    def fake_call_llm(question, context, trace_id=""):
        calls["answer"] += 1
        raise AssertionError("证据不足时不应调用最终回答模型")

    monkeypatch.setattr(rag.Settings, "ENABLE_QUERY_DECOMPOSE", True, raising=False)
    monkeypatch.setattr(rag.Settings, "DECOMPOSE_MODEL_NAME", "fake", raising=False)
    monkeypatch.setattr(rag.Settings, "DECOMPOSE_API_URL", "http://example.test", raising=False)
    monkeypatch.setattr(rag.Settings, "VECTOR_THRESHOLD", 0.35, raising=False)
    monkeypatch.setattr(rag, "decompose_query_with_llm", fake_decompose)
    monkeypatch.setattr(rag, "call_llm", fake_call_llm)
    monkeypatch.setattr(rag, "rerank", lambda candidates, question: candidates)

    result = rag.ask(
        "最长学习年限是几年？",
        top_k=1,
        retrieved_chunks=[make_chunk()],
        trace_id="test",
    )

    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["decompose_llm_used"] is False
    assert result["answer_llm_used"] is False
    assert result["llm_call_count"] == 0
    assert result["evidence_gate"]["refused"] is True
    assert result["evidence_gate"]["max_vector_score"] == 0.1
    assert calls == {"decompose": 0, "answer": 0}


def test_ask_calls_decompose_llm_for_compare_question(monkeypatch):
    """验证对比、新旧等复杂问题会触发拆分模型。"""
    calls = {"decompose": 0, "answer": 0}

    def fake_decompose(question):
        calls["decompose"] += 1
        return {
            "need_decompose": False,
            "reason": "测试中由拆分模型判定为单查询",
            "sub_queries": [],
        }

    def fake_call_llm(question, context, trace_id=""):
        calls["answer"] += 1
        raise AssertionError("本测试使用弱证据，不应调用最终回答模型")

    monkeypatch.setattr(rag.Settings, "ENABLE_QUERY_DECOMPOSE", True, raising=False)
    monkeypatch.setattr(rag.Settings, "DECOMPOSE_MODEL_NAME", "fake", raising=False)
    monkeypatch.setattr(rag.Settings, "DECOMPOSE_API_URL", "http://example.test", raising=False)
    monkeypatch.setattr(rag.Settings, "VECTOR_THRESHOLD", 0.35, raising=False)
    monkeypatch.setattr(rag, "decompose_query_with_llm", fake_decompose)
    monkeypatch.setattr(rag, "call_llm", fake_call_llm)
    monkeypatch.setattr(rag, "rerank", lambda candidates, question: candidates)

    result = rag.ask(
        "请对比新旧政策的变化。",
        top_k=1,
        retrieved_chunks=[make_chunk()],
        trace_id="test",
    )

    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["decompose_llm_used"] is True
    assert result["answer_llm_used"] is False
    assert result["llm_call_count"] == 1
    assert calls == {"decompose": 1, "answer": 0}

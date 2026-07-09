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


def test_ask_returns_no_relevant_info_when_citation_verification_fails(monkeypatch):
    """验证引用验证失败时，RAG 会拒答并返回验证结果。"""
    final_chunk = make_chunk(
        text="普通本科四年制专业学生学习年限调整为3～6年。",
        snippet="普通本科四年制专业学生学习年限调整为3～6年。",
        vector_score=0.9,
        hybrid_score=0.9,
        rerank_score=0.9,
        bm25_score=5.0,
    )
    verification = {
        "supported": False,
        "reason": "答案与引用片段关键词重合度过低",
        "overlap_ratio": 0.08,
        "matched_keywords": [],
        "expected_keyword_count": 12,
        "evidence_keyword_count": 5,
    }

    monkeypatch.setattr(rag.Settings, "ENABLE_QUERY_DECOMPOSE", False, raising=False)
    monkeypatch.setattr(rag, "rerank", lambda candidates, question: candidates)
    monkeypatch.setattr(rag, "should_refuse", lambda final_chunks: False)
    monkeypatch.setattr(rag, "call_llm", lambda question, context, trace_id="": "{}")
    monkeypatch.setattr(
        rag,
        "parse_llm_output",
        lambda llm_output: {
            "ok": True,
            "answer": "普通本科四年制最长学习年限为 8 年。",
            "references": ["Reference 1"],
        },
    )
    monkeypatch.setattr(
        rag,
        "format_reference",
        lambda final_chunks, references: [
            {
                "reference_id": "Reference 1",
                "snippet": "学生应当遵守课堂纪律，按时参加教学活动。",
            }
        ],
    )
    monkeypatch.setattr(
        rag,
        "verify_answer_citations",
        lambda question, answer, references, **kwargs: verification,
    )

    result = rag.ask(
        "最长学习年限是多少？",
        top_k=1,
        retrieved_chunks=[final_chunk],
        trace_id="test",
    )

    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["answer_confidence"] == 0.0
    assert "citation_verification" in result
    assert result["citation_verification"]["supported"] is False

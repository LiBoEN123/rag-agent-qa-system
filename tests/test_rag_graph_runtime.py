"""LangGraph RAG 子图运行时与业务分支测试。"""

import json

import pytest

from app.llm.errors import ModelCallError
from app.rag.graph import builder as rag_graph_builder
from app.rag.graph import nodes as rag_graph_nodes
from app.rag.graph import runtime as rag_graph_runtime
from app.rag.graph.runtime import RAGGraphError, run_rag_subgraph


def make_chunk(index: int = 1, **overrides) -> dict:
    """构造包含 RAG 子图各阶段必要字段的受控候选 chunk。"""
    chunk = {
        "reference_id": f"Reference {index}",
        "file_name": "policy.md",
        "source_path": "/tmp/policy.md",
        "chunk_id": f"chunk-{index}",
        "position": index,
        "text": "测试政策允许用户申请退款。",
        "snippet": "测试政策允许用户申请退款。",
        "bm25_score": 2.0,
        "vector_score": 0.9,
        "hybrid_score": 0.9,
        "rerank_score": 0.9,
    }
    chunk.update(overrides)
    return chunk


@pytest.fixture(autouse=True)
def isolate_rag_graph(monkeypatch):
    """关闭真实拆分模型并隔离已编译 Graph。"""
    rag_graph_builder.get_compiled_rag_graph.cache_clear()
    monkeypatch.setattr(
        rag_graph_nodes.Settings,
        "ENABLE_QUERY_DECOMPOSE",
        False,
    )
    yield
    rag_graph_builder.get_compiled_rag_graph.cache_clear()


def configure_success_path(monkeypatch) -> None:
    """为成功路径注入完全本地、确定性的模型输出。"""
    monkeypatch.setattr(rag_graph_nodes, "rerank", lambda candidates, question: candidates)
    monkeypatch.setattr(rag_graph_nodes, "should_refuse", lambda chunks: False)
    monkeypatch.setattr(
        rag_graph_nodes,
        "call_llm",
        lambda question, context, trace_id="": json.dumps(
            {
                "answer": "测试政策允许用户申请退款。",
                "references": ["Reference 1"],
            },
            ensure_ascii=False,
        ),
    )


def test_single_query_reuses_retrieved_chunks_without_retrieve(monkeypatch):
    """已有工具召回时只重排一次并按 top_k 截断。"""
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    calls = {"retrieve": 0, "rerank": 0}

    def fail_retrieve(*args, **kwargs):
        calls["retrieve"] += 1
        raise AssertionError("已有 retrieved_chunks 时不应再次基础召回")

    def fake_rerank(candidates, question):
        calls["rerank"] += 1
        assert candidates is chunks
        return [candidates[1], candidates[0], candidates[2]]

    configure_success_path(monkeypatch)
    monkeypatch.setattr(rag_graph_nodes, "retrieve", fail_retrieve)
    monkeypatch.setattr(rag_graph_nodes, "rerank", fake_rerank)

    result = run_rag_subgraph(
        question="测试政策如何申请退款？",
        top_k=2,
        retrieved_chunks=chunks,
        trace_id="trace-single",
    )

    assert result["status"] == "OK"
    assert result["final_chunks"] == [chunks[1], chunks[0]]
    assert calls == {"retrieve": 0, "rerank": 1}
    assert set(result["timings"]) == {
        "decompose_cost_ms",
        "decompose_llm_used",
        "decompose_rule_reason",
        "decompose_skipped_reason",
        "rerank_cost_ms",
        "retrieve_with_sub_queries_cost_ms",
        "context_build_cost_ms",
        "llm_cost_ms",
        "parse_cost_ms",
    }


def test_single_query_without_candidates_retrieves_once(monkeypatch):
    """独立调用子图且未提供候选时执行一次基础召回和一次重排。"""
    calls = {"retrieve": 0, "rerank": 0}
    chunks = [make_chunk()]

    def fake_retrieve(question, top_k):
        calls["retrieve"] += 1
        return chunks

    def fake_rerank(candidates, question):
        calls["rerank"] += 1
        return candidates

    configure_success_path(monkeypatch)
    monkeypatch.setattr(rag_graph_nodes, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_graph_nodes, "rerank", fake_rerank)

    result = run_rag_subgraph(
        question="测试政策如何申请退款？",
        retrieved_chunks=None,
    )

    assert result["status"] == "OK"
    assert result["final_chunks"] == chunks
    assert calls == {"retrieve": 1, "rerank": 1}


def test_rule_selected_single_query_skips_decompose_and_answer_on_weak_evidence(
    monkeypatch,
):
    """规则判定单查询时不调用拆分模型，弱证据也不调用回答模型。"""
    calls = {"decompose": 0, "answer": 0}
    monkeypatch.setattr(rag_graph_nodes.Settings, "ENABLE_QUERY_DECOMPOSE", True)
    monkeypatch.setattr(rag_graph_nodes.Settings, "DECOMPOSE_MODEL_NAME", "fake")
    monkeypatch.setattr(
        rag_graph_nodes.Settings,
        "DECOMPOSE_API_URL",
        "http://example.test",
    )
    monkeypatch.setattr(
        rag_graph_nodes,
        "should_decompose_query",
        lambda question: (False, "single intent"),
    )

    def fail_decompose(question, trace_id=""):
        calls["decompose"] += 1
        raise AssertionError("规则判定单查询后不应调用拆分模型")

    def fail_answer(*args, **kwargs):
        calls["answer"] += 1
        raise AssertionError("证据不足时不应调用回答模型")

    monkeypatch.setattr(rag_graph_nodes, "decompose_query_with_llm", fail_decompose)
    monkeypatch.setattr(rag_graph_nodes, "rerank", lambda candidates, question: candidates)
    monkeypatch.setattr(rag_graph_nodes, "should_refuse", lambda chunks: True)
    monkeypatch.setattr(rag_graph_nodes, "call_llm", fail_answer)

    result = run_rag_subgraph(
        question="最长学习年限是几年？",
        top_k=1,
        retrieved_chunks=[make_chunk(vector_score=0.1, rerank_score=0.1)],
        trace_id="test-single-rule",
    )

    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["decompose_llm_used"] is False
    assert result["answer_llm_used"] is False
    assert result["llm_call_count"] == 0
    assert result["timings"]["decompose_rule_reason"] == "single intent"
    assert calls == {"decompose": 0, "answer": 0}


def test_decompose_model_can_keep_single_query(monkeypatch):
    """复杂问题调用拆分模型后仍可判定为无需拆分的单查询。"""
    calls = {"decompose": 0, "answer": 0}
    received = {}
    monkeypatch.setattr(rag_graph_nodes.Settings, "ENABLE_QUERY_DECOMPOSE", True)
    monkeypatch.setattr(rag_graph_nodes.Settings, "DECOMPOSE_MODEL_NAME", "fake")
    monkeypatch.setattr(
        rag_graph_nodes.Settings,
        "DECOMPOSE_API_URL",
        "http://example.test",
    )
    monkeypatch.setattr(
        rag_graph_nodes,
        "should_decompose_query",
        lambda question: (True, "comparison intent"),
    )

    def fake_decompose(question, trace_id=""):
        calls["decompose"] += 1
        received["question"] = question
        received["trace_id"] = trace_id
        return {
            "need_decompose": False,
            "reason": "拆分模型判定为单查询",
            "sub_queries": [],
        }

    def fail_answer(*args, **kwargs):
        calls["answer"] += 1
        raise AssertionError("证据不足时不应调用回答模型")

    monkeypatch.setattr(rag_graph_nodes, "decompose_query_with_llm", fake_decompose)
    monkeypatch.setattr(rag_graph_nodes, "rerank", lambda candidates, question: candidates)
    monkeypatch.setattr(rag_graph_nodes, "should_refuse", lambda chunks: True)
    monkeypatch.setattr(rag_graph_nodes, "call_llm", fail_answer)

    result = run_rag_subgraph(
        question="请对比新旧政策的变化。",
        top_k=1,
        retrieved_chunks=[make_chunk(vector_score=0.1, rerank_score=0.1)],
        trace_id="test-single-after-decompose",
    )

    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["decompose_llm_used"] is True
    assert result["answer_llm_used"] is False
    assert result["llm_call_count"] == 1
    assert calls == {"decompose": 1, "answer": 0}
    assert received == {
        "question": "请对比新旧政策的变化。",
        "trace_id": "test-single-after-decompose",
    }


def test_multi_query_uses_current_index_facade_only(monkeypatch):
    """多查询使用 decompose_top_k，且不进入单查询 retrieve/rerank。"""
    calls = {"decompose": 0, "facade": 0, "retrieve": 0, "rerank": 0}
    final_chunks = [make_chunk()]
    monkeypatch.setattr(rag_graph_nodes.Settings, "ENABLE_QUERY_DECOMPOSE", True)
    monkeypatch.setattr(rag_graph_nodes.Settings, "DECOMPOSE_MODEL_NAME", "fake")
    monkeypatch.setattr(rag_graph_nodes.Settings, "DECOMPOSE_API_URL", "http://fake")
    monkeypatch.setattr(
        rag_graph_nodes,
        "should_decompose_query",
        lambda question: (True, "multiple intents"),
    )

    def fake_decompose(question, trace_id=""):
        calls["decompose"] += 1
        return {
            "need_decompose": True,
            "reason": "multiple intents",
            "sub_queries": [
                {
                    "id": 1,
                    "sub_question": "问题一",
                    "search_query": "查询一",
                    "must_answer": True,
                }
            ],
        }

    def fake_facade(question, sub_queries, final_top_k):
        calls["facade"] += 1
        assert final_top_k == 7
        assert sub_queries[0]["sub_question"] == "问题一"
        return final_chunks

    def fail_retrieve(*args, **kwargs):
        calls["retrieve"] += 1
        raise AssertionError("多查询不应执行单查询基础召回")

    def fail_rerank(*args, **kwargs):
        calls["rerank"] += 1
        raise AssertionError("多查询不应执行单查询重排分支")

    configure_success_path(monkeypatch)
    monkeypatch.setattr(rag_graph_nodes, "decompose_query_with_llm", fake_decompose)
    monkeypatch.setattr(
        rag_graph_nodes,
        "retrieve_with_sub_queries_from_current_index",
        fake_facade,
    )
    monkeypatch.setattr(rag_graph_nodes, "retrieve", fail_retrieve)
    monkeypatch.setattr(rag_graph_nodes, "rerank", fail_rerank)

    result = run_rag_subgraph(
        question="请分别说明测试政策入口和退款条件？",
        decompose_top_k=7,
        retrieved_chunks=[make_chunk(2)],
    )

    assert result["status"] == "OK"
    assert result["final_chunks"] == final_chunks
    assert result["decompose_llm_used"] is True
    assert result["llm_call_count"] == 2
    assert calls == {"decompose": 1, "facade": 1, "retrieve": 0, "rerank": 0}


def test_ambiguous_question_short_circuits_retrieval_and_model(monkeypatch):
    """问题歧义时两个 LLM 标志均为 False，且不执行后续节点。"""
    monkeypatch.setattr(
        rag_graph_nodes,
        "retrieve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("歧义路径不应检索")
        ),
    )
    monkeypatch.setattr(
        rag_graph_nodes,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("歧义路径不应调用回答模型")
        ),
    )

    result = run_rag_subgraph(question="这个？")

    assert result["status"] == "NEED_CLARIFICATION"
    assert result["need_clarification"] is True
    assert result["decompose_llm_used"] is False
    assert result["answer_llm_used"] is False
    assert result["llm_call_count"] == 0


def test_evidence_refusal_does_not_call_answer_model(monkeypatch):
    """证据门控拒答保留 evidence_gate，且不调用回答模型。"""
    monkeypatch.setattr(rag_graph_nodes, "rerank", lambda candidates, question: candidates)
    monkeypatch.setattr(rag_graph_nodes, "should_refuse", lambda chunks: True)
    monkeypatch.setattr(
        rag_graph_nodes,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("证据不足时不应调用回答模型")
        ),
    )

    result = run_rag_subgraph(
        question="测试政策如何申请退款？",
        retrieved_chunks=[make_chunk(vector_score=0.1, rerank_score=0.1)],
    )

    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["answer_llm_used"] is False
    assert result["evidence_gate"]["refused"] is True


def test_model_error_keeps_answer_llm_usage(monkeypatch):
    """回答模型真正调用后失败返回 MODEL_ERROR，并保留用户可读消息。"""
    configure_success_path(monkeypatch)
    monkeypatch.setattr(
        rag_graph_nodes,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ModelCallError("模型暂时不可用", detail="internal")
        ),
    )

    result = run_rag_subgraph(
        question="测试政策如何申请退款？",
        retrieved_chunks=[make_chunk()],
    )

    assert result["status"] == "MODEL_ERROR"
    assert result["answer"] == "模型暂时不可用"
    assert result["answer_llm_used"] is True
    assert result["llm_call_count"] == 1


def test_bad_model_output_stops_before_citation_binding(monkeypatch):
    """模型坏输出返回 MODEL_BAD_OUTPUT 和空引用，不继续绑定。"""
    configure_success_path(monkeypatch)
    monkeypatch.setattr(rag_graph_nodes, "call_llm", lambda *args, **kwargs: "not-json")
    monkeypatch.setattr(
        rag_graph_nodes,
        "format_reference",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("解析失败后不应绑定引用")
        ),
    )

    result = run_rag_subgraph(
        question="测试政策如何申请退款？",
        retrieved_chunks=[make_chunk()],
    )

    assert result["status"] == "MODEL_BAD_OUTPUT"
    assert result["references"] == []
    assert result["answer_llm_used"] is True


def test_citation_verification_failure_returns_no_evidence(monkeypatch):
    """引用验证失败返回无证据状态并保留验证详情。"""
    configure_success_path(monkeypatch)
    verification = {
        "supported": False,
        "reason": "答案与引用片段关键词重合度过低",
        "overlap_ratio": 0.0,
        "matched_keywords": [],
        "expected_keyword_count": 3,
        "evidence_keyword_count": 2,
    }
    monkeypatch.setattr(
        rag_graph_nodes,
        "verify_answer_citations",
        lambda **kwargs: verification,
    )

    result = run_rag_subgraph(
        question="测试政策如何申请退款？",
        retrieved_chunks=[make_chunk()],
    )

    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["citation_verification"] == verification
    assert result["answer_confidence"] == 0.0


@pytest.mark.parametrize(
    ("references", "answer"),
    [
        ([], "测试政策允许用户申请退款。"),
        (["Reference 1"], "我无法根据提供的文档确定答案"),
    ],
)
def test_missing_reference_or_stop_word_returns_no_evidence(
    monkeypatch,
    references,
    answer,
):
    """没有有效引用或答案包含 STOP_WORD 时继续返回无证据状态。"""
    configure_success_path(monkeypatch)
    monkeypatch.setattr(
        rag_graph_nodes,
        "call_llm",
        lambda *args, **kwargs: json.dumps(
            {"answer": answer, "references": references},
            ensure_ascii=False,
        ),
    )
    monkeypatch.setattr(
        rag_graph_nodes,
        "verify_answer_citations",
        lambda **kwargs: {"supported": True, "reason": ""},
    )

    result = run_rag_subgraph(
        question="测试政策如何申请退款？",
        retrieved_chunks=[make_chunk()],
    )

    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["reason"] == "模型给出的答案没有足够的引用支撑"


def test_success_result_contains_compatible_fields(monkeypatch):
    """成功结果保留引用、置信度、timings 和准确 LLM 计数。"""
    configure_success_path(monkeypatch)

    result = run_rag_subgraph(
        question="测试政策如何申请退款？",
        retrieved_chunks=[make_chunk()],
    )

    assert result["status"] == "OK"
    assert result["answer"] == "测试政策允许用户申请退款。"
    assert result["references"][0]["reference_id"] == "Reference 1"
    assert result["references"][0]["chunk_id"] == "chunk-1"
    assert result["retrieval_confidence"] > 0
    assert result["answer_confidence"] > 0
    assert result["decompose_llm_used"] is False
    assert result["answer_llm_used"] is True
    assert result["llm_call_count"] == 1
    assert result["citation_verification"]["supported"] is True


@pytest.mark.parametrize("output", [None, {}, {"result": None}, {"result": {}}])
def test_runtime_rejects_malformed_graph_output(monkeypatch, output):
    """Runtime 不返回内部 Graph State，并拒绝缺失统一字段的结果。"""
    class MalformedGraph:
        def invoke(self, graph_input):
            return output

    monkeypatch.setattr(
        rag_graph_runtime,
        "get_compiled_rag_graph",
        lambda: MalformedGraph(),
    )

    with pytest.raises(RAGGraphError):
        run_rag_subgraph(question="测试政策如何申请退款？")

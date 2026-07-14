"""旧 ``rag.ask`` 与 LangGraph RAG 子图等价性测试。"""

import json

import pytest

import app.rag as rag
from app.llm.errors import ModelCallError
from app.rag import pipeline as rag_pipeline
from app.rag.graph import builder as rag_graph_builder
from app.rag.graph import nodes as rag_graph_nodes
from app.rag.graph.runtime import run_rag_subgraph


EQUIVALENT_FIELDS = [
    "status",
    "answer",
    "references",
    "final_chunks",
    "retrieval_confidence",
    "answer_confidence",
    "need_clarification",
    "clarification_question",
    "reason",
    "decompose_llm_used",
    "answer_llm_used",
    "llm_call_count",
]


def make_chunk(**overrides) -> dict:
    """构造新旧路径共同使用的确定性证据。"""
    chunk = {
        "reference_id": "Reference 1",
        "file_name": "policy.md",
        "source_path": "/tmp/policy.md",
        "chunk_id": "chunk-1",
        "position": 1,
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
def isolate_graph(monkeypatch):
    """隔离 Graph 缓存，并让新旧路径共享关闭拆分的默认配置。"""
    rag_graph_builder.get_compiled_rag_graph.cache_clear()
    monkeypatch.setattr(rag.Settings, "ENABLE_QUERY_DECOMPOSE", False)
    yield
    rag_graph_builder.get_compiled_rag_graph.cache_clear()


def patch_both(monkeypatch, name: str, value) -> None:
    """把同一受控依赖注入旧 pipeline 和新子图节点。"""
    monkeypatch.setattr(rag, name, value, raising=False)
    if hasattr(rag_pipeline, name):
        monkeypatch.setattr(rag_pipeline, name, value)
    monkeypatch.setattr(rag_graph_nodes, name, value)


def assert_equivalent(legacy_result: dict, graph_result: dict) -> None:
    """比较稳定业务字段与 timings 阶段集合。"""
    assert {key: legacy_result[key] for key in EQUIVALENT_FIELDS} == {
        key: graph_result[key] for key in EQUIVALENT_FIELDS
    }
    assert set(legacy_result["timings"]) == set(graph_result["timings"])


@pytest.mark.parametrize(
    ("scenario", "question"),
    [
        ("clarify", "这个？"),
        ("refuse", "测试政策如何申请退款？"),
        ("model_error", "测试政策如何申请退款？"),
        ("bad_output", "测试政策如何申请退款？"),
        ("citation_failure", "测试政策如何申请退款？"),
        ("success", "测试政策如何申请退款？"),
    ],
)
def test_legacy_and_graph_business_results_are_equivalent(
    monkeypatch,
    scenario,
    question,
):
    """主要业务终止路径在相同 mock 输入下保持返回等价。"""
    chunks = [make_chunk()]
    patch_both(monkeypatch, "rerank", lambda candidates, query: candidates)
    patch_both(monkeypatch, "should_refuse", lambda final_chunks: scenario == "refuse")

    if scenario == "model_error":
        def fake_call_llm(*args, **kwargs):
            raise ModelCallError("模型暂时不可用", detail="internal")
    elif scenario == "bad_output":
        def fake_call_llm(*args, **kwargs):
            return "not-json"
    else:
        def fake_call_llm(*args, **kwargs):
            return json.dumps(
                {
                    "answer": "测试政策允许用户申请退款。",
                    "references": ["Reference 1"],
                },
                ensure_ascii=False,
            )
    patch_both(monkeypatch, "call_llm", fake_call_llm)

    if scenario == "citation_failure":
        verification = {
            "supported": False,
            "reason": "答案与引用片段关键词重合度过低",
            "overlap_ratio": 0.0,
            "matched_keywords": [],
            "expected_keyword_count": 3,
            "evidence_keyword_count": 2,
        }
        patch_both(
            monkeypatch,
            "verify_answer_citations",
            lambda **kwargs: verification,
        )

    legacy_result = rag.ask(
        question=question,
        top_k=1,
        retrieved_chunks=chunks,
        trace_id=f"legacy-{scenario}",
    )
    graph_result = run_rag_subgraph(
        question=question,
        top_k=1,
        retrieved_chunks=chunks,
        trace_id=f"graph-{scenario}",
    )

    assert_equivalent(legacy_result, graph_result)


def test_legacy_and_graph_multi_query_results_are_equivalent(monkeypatch):
    """多查询路径复用同一现有结果结构，并使用相同 decompose_top_k。"""
    final_chunks = [make_chunk()]
    decompose_result = {
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
    monkeypatch.setattr(rag.Settings, "ENABLE_QUERY_DECOMPOSE", True)
    monkeypatch.setattr(rag.Settings, "DECOMPOSE_MODEL_NAME", "fake")
    monkeypatch.setattr(rag.Settings, "DECOMPOSE_API_URL", "http://fake")
    monkeypatch.setattr(
        rag_pipeline,
        "should_decompose_query",
        lambda question: (True, "multiple intents"),
    )
    monkeypatch.setattr(
        rag_graph_nodes,
        "should_decompose_query",
        lambda question: (True, "multiple intents"),
    )
    patch_both(
        monkeypatch,
        "decompose_query_with_llm",
        lambda question: decompose_result,
    )
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_with_sub_queries",
        lambda **kwargs: final_chunks,
    )
    monkeypatch.setattr(
        rag_graph_nodes,
        "retrieve_with_sub_queries_from_current_index",
        lambda question, sub_queries, final_top_k: final_chunks,
    )
    patch_both(monkeypatch, "should_refuse", lambda chunks: False)
    patch_both(
        monkeypatch,
        "call_llm",
        lambda *args, **kwargs: json.dumps(
            {
                "answer": "测试政策允许用户申请退款。",
                "references": ["Reference 1"],
            },
            ensure_ascii=False,
        ),
    )

    legacy_result = rag.ask(
        question="请分别说明测试政策入口和退款条件？",
        decompose_top_k=7,
        retrieved_chunks=[make_chunk(chunk_id="initial")],
    )
    graph_result = run_rag_subgraph(
        question="请分别说明测试政策入口和退款条件？",
        decompose_top_k=7,
        retrieved_chunks=[make_chunk(chunk_id="initial")],
    )

    assert_equivalent(legacy_result, graph_result)
    assert graph_result["decompose_llm_used"] is True
    assert graph_result["llm_call_count"] == 2

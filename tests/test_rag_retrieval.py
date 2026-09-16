import numpy as np
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import ValidationError

from app.llm.schemas import QueryDecomposeOutput
from app.rag import decomposition
from app.rag.decomposition import normalize_decompose_result, should_decompose_query
from app.rag import retrieval
from app.rag.policy import should_refuse
from app.ingestion import IngestionRequester, IngestionService, build_knowledge_document
from app.rag import indexing, store
from tests.support.structured_document import admission_structure, imut_admission_structure, registration_structure


def test_table_semantics_reach_real_chroma_bm25_retrieval(tmp_path, monkeypatch):
    document = build_knowledge_document(
        source_type="local_file", source_uri=str(tmp_path / "table.pdf"), title="企业登记办理要求",
        content="", structured_content=registration_structure(), metadata={"file_type": "pdf"},
    )
    candidate = IngestionService().propose(document=document, requester=IngestionRequester(
        principal_id="table-smoke", principal_type="system", source="pytest",
    ))
    [table_chunk] = indexing.candidate_to_langchain_documents(candidate, chunk_size=600, chunk_overlap=0)
    for number in range(12):
        (tmp_path / f"distractor-{number}.txt").write_text(f"校园图书馆开放时间：周一至周五。分馆编号{number}。")
    documents = [table_chunk, *indexing.load_local_langchain_documents(str(tmp_path))]
    embeddings = DeterministicFakeEmbedding(size=32)
    indexing.sync_documents(documents, embeddings)
    bundle = store.load_index_cache(model=embeddings, rerank_model=object())
    assert bundle["ok"], bundle["message"]

    def no_fallback(*args, **kwargs):
        pytest.fail("table smoke must query real Chroma, not fall back to local vectors")

    monkeypatch.setattr(retrieval, "_retrieve_with_local_embeddings", no_fallback)
    query = "企业注册需要哪些材料"
    hits = retrieval.retrieve(
        query, bundle["chunks"], bundle["bm25"], bundle["doc_embeddings"], embeddings, top_k=3,
    )
    hit = next(item for item in hits if item["metadata"].get("element_type") == "table")
    assert hit["text"] == table_chunk.page_content
    assert "事项：企业注册；条件：年满18岁；材料：身份证、申请表。" in hit["text"]
    assert hit["bm25_score"] > 0
    assert hit["bm25_score"] == max(item["bm25_score"] for item in hits)
    assert hit["metadata"]["source_id"] == document.source_id
    assert hit["position"] == 1


@pytest.mark.parametrize("structure_factory, physics_row_offset", [
    (admission_structure, 0), (imut_admission_structure, 1),
])
@pytest.mark.parametrize("query, section, expected_row, evidence", [
    ("内蒙古本科批物理类安全工程最低分是多少？", "内蒙古本科批物理类", 7,
     ("专业名称：安全工程", "最低分：501")),
    ("内蒙古本科批物理类材料成型及控制工程民族班最低分是多少？", "内蒙古本科批物理类", 9,
     ("专业名称：材料成型及控制工程", "最低分：480", "备注：民族班")),
    ("内蒙古本科提前B段艺术类环境设计平均分是多少？", "内蒙古本科提前B段艺术类", 4,
     ("专业名称：环境设计", "平均分：477.87")),
])
def test_admission_section_semantics_reach_real_chroma_bm25(
    tmp_path, monkeypatch, query, section, expected_row, evidence, structure_factory, physics_row_offset,
):
    if expected_row >= 7:
        expected_row += physics_row_offset  # 实抓表保留了艺术类末尾的总计行。
    document = build_knowledge_document(
        source_type="local_file", source_uri=str(tmp_path / "admission.pdf"), title="录取分数统计",
        content="", structured_content=structure_factory(), metadata={"file_type": "pdf"},
    )
    candidate = IngestionService().propose(document=document, requester=IngestionRequester(
        principal_id="section-smoke", principal_type="system", source="pytest",
    ))
    table_chunks = indexing.candidate_to_langchain_documents(candidate, chunk_size=160, chunk_overlap=0)
    for number in range(12):
        (tmp_path / f"distractor-{number}.txt").write_text(f"校园图书馆开放时间：周一至周五。分馆编号{number}。")
    embeddings = DeterministicFakeEmbedding(size=32)
    indexing.sync_documents([*table_chunks, *indexing.load_local_langchain_documents(str(tmp_path))], embeddings)
    bundle = store.load_index_cache(model=embeddings, rerank_model=object())
    assert bundle["ok"], bundle["message"]

    def forbidden(*args, **kwargs):
        pytest.fail("section smoke must use real Chroma and existing BM25, without fallback or reranking")

    monkeypatch.setattr(retrieval, "_retrieve_with_local_embeddings", forbidden)
    monkeypatch.setattr(retrieval, "rerank", forbidden)
    hits = retrieval.retrieve(
        query, bundle["chunks"], bundle["bm25"], bundle["doc_embeddings"], embeddings, top_k=3,
    )
    hit = next(item for item in hits if item["metadata"].get("row_start") == expected_row)
    assert hit["metadata"]["section_title"] == section
    assert hit["metadata"]["row_end"] == expected_row
    assert f"分区：{section}" in hit["text"]
    assert "上下文：内蒙古工业大学2026年分省录取分数统计" in hit["text"]
    assert all(value in hit["text"] for value in evidence)
    assert hit["text"] == next(chunk.page_content for chunk in table_chunks
                               if chunk.metadata["row_start"] == expected_row)
    assert hit["bm25_score"] > 0
    assert hit["bm25_score"] == max(item["bm25_score"] for item in hits)
    assert hit["metadata"]["source_id"] == document.source_id


@pytest.mark.parametrize(
    "query_error",
    [
        retrieval.ChromaUnavailableError("ChromaDB unavailable"),
        RuntimeError("generic ChromaDB query failure"),
    ],
    ids=["chroma-unavailable", "generic-query-error"],
)
def test_retrieve_falls_back_to_local_embeddings_after_chroma_error(
    monkeypatch,
    query_error,
):
    chunks = [
        {"text": "first", "metadata": {"chunk_id": 1}},
        {"text": "second", "metadata": {"chunk_id": 2}},
    ]
    doc_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class FakeBm25:
        def get_scores(self, _tokens):
            return np.array([0.0, 2.0], dtype=np.float32)

    class FakeModel:
        def __init__(self):
            self.encoded_texts = []

        def encode(self, texts, **_kwargs):
            self.encoded_texts.append(texts)
            return np.array([[1.0, 0.0]], dtype=np.float32)

    def fail_chroma_query(*_args, **_kwargs):
        raise query_error

    model = FakeModel()
    monkeypatch.setattr(retrieval, "query_chroma", fail_chroma_query)

    results = retrieval.retrieve(
        question="question",
        chunks=chunks,
        bm25=FakeBm25(),
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=2,
    )

    assert results
    assert {result["chunk_id"] for result in results} == {1, 2}
    assert model.encoded_texts == [["question"], ["question"]]


def test_retrieval_does_not_reexport_policy_or_scoring_symbols():
    """检索模块不再代理 policy 或 scoring 的公共符号。"""
    for symbol in [
        "build_clarification_question",
        "is_ambiguous_question",
        "should_refuse",
        "safe_float",
    ]:
        assert symbol not in retrieval.__all__
        assert not hasattr(retrieval, symbol)


def test_should_refuse_empty_results():
    """验证没有召回结果时拒答。"""
    assert should_refuse([], vector_threshold=0.35) is True


def test_should_refuse_accepts_high_vector_score():
    """验证向量分数达标时直接放行。"""
    chunks = [
        {
            "vector_score": 0.42,
            "bm25_score": 0.0,
            "hybrid_score": 0.0,
            "rerank_score": 0.0,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is False


def test_should_refuse_rejects_weak_evidence():
    """验证多路分数都弱时拒答。"""
    chunks = [
        {
            "vector_score": 0.12,
            "bm25_score": 0.0,
            "hybrid_score": 0.1,
            "rerank_score": 0.1,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is True


def test_should_refuse_accepts_strong_rerank_with_near_vector_score():
    """验证向量略低但重排和混合分数较强时放行。"""
    chunks = [
        {
            "vector_score": 0.28,
            "bm25_score": 0.0,
            "hybrid_score": 0.38,
            "rerank_score": 0.5,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is False


def test_should_refuse_rejects_near_vector_with_weak_keyword_signal():
    """验证向量略低时不会被很弱的关键词命中误放行。"""
    chunks = [
        {
            "vector_score": 0.28,
            "bm25_score": 0.1,
            "hybrid_score": 0.05,
            "rerank_score": 0.1,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is True


def test_should_refuse_accepts_strong_keyword_and_rerank_signal():
    """验证关键词命中和重排分数都强时不被向量阈值误拒。"""
    chunks = [
        {
            "vector_score": 0.0,
            "bm25_score": 8.2,
            "hybrid_score": 0.24,
            "rerank_score": 0.49,
        }
    ]

    assert should_refuse(chunks, vector_threshold=0.35) is False


def test_should_decompose_query_skips_short_single_intent_question():
    """验证普通短问题不触发拆分模型。"""
    should_decompose, reason = should_decompose_query("最长学习年限是几年？")

    assert should_decompose is False
    assert "single-intent" in reason


def test_should_decompose_query_triggers_for_multi_intent_keywords():
    """验证对比、新旧、分别、以及等关键词会触发拆分。"""
    examples = [
        "新旧政策对比有哪些变化？",
        "请分别说明缓考入口以及申请时间。",
        "毕业条件和材料分别是什么？",
    ]

    for question in examples:
        should_decompose, reason = should_decompose_query(question)
        assert should_decompose is True, (question, reason)


def test_normalize_decompose_result_preserves_business_fields():
    """经过 Schema 校验后继续保留拆分字段和业务规整。"""
    result = normalize_decompose_result(
        QueryDecomposeOutput.model_validate(
            {
                "need_decompose": True,
                "reason": "多证据",
                "sub_queries": [
                    {
                        "id": 1,
                        "sub_question": "入口是什么？",
                        "search_query": "缓考 免考 线上填报 入口",
                        "must_answer": True,
                    }
                ],
            }
        )
    )

    assert result["need_decompose"] is True
    assert result["reason"] == "多证据"
    assert result["sub_queries"] == [
        {
            "id": 1,
            "sub_question": "入口是什么？",
            "search_query": "缓考 免考 线上填报 入口",
            "must_answer": True,
        }
    ]


def test_normalize_decompose_result_keeps_no_decomposition_decision():
    """模型判定无需拆分时继续返回单查询结果。"""
    result = normalize_decompose_result(
        QueryDecomposeOutput(
            need_decompose=False,
            reason="单一问题",
            sub_queries=[],
        )
    )

    assert result == {
        "need_decompose": False,
        "reason": "单一问题",
        "sub_queries": [],
    }


def test_normalize_decompose_result_compacts_text_and_limits_to_five():
    """业务 normalization 保留单行压缩和最多五条语义。"""
    result = normalize_decompose_result(
        QueryDecomposeOutput(
            need_decompose=True,
            reason="多方面",
            sub_queries=[
                {
                    "id": index,
                    "sub_question": f"问题 {index}\n详情",
                    "search_query": f"查询 {index}\n关键词",
                    "must_answer": True,
                }
                for index in range(1, 7)
            ],
        )
    )

    assert result["need_decompose"] is True
    assert len(result["sub_queries"]) == 5
    assert result["sub_queries"][0]["sub_question"] == "问题 1 详情"
    assert result["sub_queries"][0]["search_query"] == "查询 1 关键词"
    assert result["sub_queries"][0]["must_answer"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"need_decompose": True, "sub_queries": "bad"},
        {
            "need_decompose": True,
            "sub_queries": [{"sub_question": "问题", "must_answer": True}],
        },
        {
            "need_decompose": True,
            "sub_queries": [
                {
                    "sub_question": "问题",
                    "search_query": "查询",
                    "must_answer": "true",
                }
            ],
        },
    ],
)
def test_query_decompose_schema_rejects_malformed_sub_queries(payload):
    """拆分 Schema 拒绝错误列表、缺字段和字符串 bool。"""
    with pytest.raises(ValidationError):
        QueryDecomposeOutput.model_validate(payload)


def test_decompose_query_with_llm_uses_native_structured_result(monkeypatch):
    """拆分模型通过对应工厂取得原生结构化结果并保留消息顺序。"""

    class FakeDecomposeModel:
        def __init__(self):
            self.calls = []

        def with_structured_output(self, schema):
            assert schema is QueryDecomposeOutput
            return self

        def invoke(self, messages):
            self.calls.append(messages)
            return QueryDecomposeOutput(
                need_decompose=False,
                reason="单一问题",
                sub_queries=[],
            )

    model = FakeDecomposeModel()
    monkeypatch.setattr(decomposition, "get_decompose_model", lambda: model)

    result = decomposition.decompose_query_with_llm("缓考和免考入口是什么？")

    assert result["need_decompose"] is False
    assert result["reason"] == "单一问题"
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[0][1], HumanMessage)
    assert model.calls[0][1].content == "缓考和免考入口是什么？"

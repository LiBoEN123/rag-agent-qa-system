import numpy as np
import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.rag import retrieval
from app.rag.policy import should_refuse
from app.ingestion import IngestionRequester, IngestionService, build_knowledge_document
from app.indexing import langchain as indexing
from app.rag import runtime as store
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
    bundle = store._load_index_bundle(model=embeddings, rerank_model=object())
    assert bundle["ok"], bundle["message"]

    _forbid_chroma_reads(monkeypatch)
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
    bundle = store._load_index_bundle(model=embeddings, rerank_model=object())
    assert bundle["ok"], bundle["message"]

    def forbidden(*args, **kwargs):
        pytest.fail("base retrieval must not run reranking")

    _forbid_chroma_reads(monkeypatch)
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


def _forbid_chroma_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("retrieval must not query Chroma or reload its embeddings")

    monkeypatch.setattr(Chroma, "get", forbidden)
    monkeypatch.setattr(Chroma, "similarity_search_by_vector_with_relevance_scores", forbidden)
    monkeypatch.setattr(Chroma, "similarity_search_by_vector", forbidden)


class FixedBm25:
    def __init__(self, scores):
        self.scores = np.asarray(scores)
        self.calls = []

    def get_scores(self, tokens):
        self.calls.append(tokens)
        return self.scores


class FixedEmbedding:
    def __init__(self, vector=(1.0, 0.0)):
        self.vector = vector
        self.queries = []

    def embed_query(self, question):
        self.queries.append(question)
        return self.vector

    def embed_documents(self, texts):
        pytest.fail("retrieval must reuse document embeddings")


def _chunks(count):
    return [
        {"text": f"body {index}\n detail", "metadata": {
            "chunk_id": index, "source_id": "source", "position": index,
            "source_path": "/kb/guide.txt", "file_name": "guide.txt",
        }}
        for index in range(count)
    ]


def _documents_with_cosines(cosines):
    cosines = np.asarray(cosines, dtype=np.float32)
    return np.column_stack((cosines, np.sqrt(1 - cosines ** 2)))


@pytest.mark.parametrize("historical_vectors", [False, True])
def test_full_corpus_scores_and_legacy_fields_without_chroma(monkeypatch, historical_vectors):
    _forbid_chroma_reads(monkeypatch)
    chunks = _chunks(8)
    bm25_scores = np.array([0, 0, 6, 4, 2, 7, 8, 0], dtype=np.float64)
    cosines = np.array([1, 0.6, 0, -0.6, -1, 0.8, -0.8, 0.5])
    documents = _documents_with_cosines(cosines)
    if historical_vectors:
        documents *= np.arange(1, 9, dtype=np.float32)[:, None]
    documents.setflags(write=False)
    original = documents.copy()
    bm25 = FixedBm25(bm25_scores)
    model = FixedEmbedding((3.0, 0.0) if historical_vectors else (1.0, 0.0))

    results = retrieval.retrieve("  question  ", chunks, bm25, documents, model, top_k=2)

    expected_hybrid = 0.4 * bm25_scores / 8 + 0.6 * (cosines + 1) / 2
    assert len(results) == len(chunks)
    assert bm25.calls == [retrieval.tokenize("question")]
    assert model.queries == ["question"]
    assert [item["chunk_id"] for item in results] == [5, 0, 2, 1, 6, 7, 3, 4]
    for rank, item in enumerate(results, 1):
        index = item["chunk_id"]
        assert set(item) == {
            "reference_id", "file_name", "source_path", "chunk_id", "position", "text",
            "snippet", "bm25_score", "vector_score", "hybrid_score", "metadata",
        }
        assert item["bm25_score"] == bm25_scores[index]
        assert item["vector_score"] == pytest.approx(cosines[index], abs=1e-6)
        assert item["hybrid_score"] == pytest.approx(expected_hybrid[index], abs=1e-6)
        assert item["reference_id"] == f"Reference {rank}"
        assert item["metadata"] == chunks[index]["metadata"]
        assert item["position"] == index
        assert item["text"] == chunks[index]["text"]
        assert item["snippet"] == f"body {index} detail"
        assert item["file_name"] == "guide.txt"
        assert item["source_path"] == "/kb/guide.txt"
    np.testing.assert_array_equal(documents, original)


def test_bm25_candidate_outside_old_vector_top_k_has_real_cosine(monkeypatch):
    _forbid_chroma_reads(monkeypatch)
    # top_k=1 时旧 Chroma 只取 20 个向量候选；第 25 个 chunk 的关键词分数最高。
    cosines = [0.9] * 24 + [0.2, 1.0, -1.0]
    bm25 = FixedBm25([0] * 24 + [100, 0, 0])
    results = retrieval.retrieve(
        "question", _chunks(27), bm25, _documents_with_cosines(cosines), FixedEmbedding(), top_k=1,
    )

    assert len(results) == 5
    assert results[0]["chunk_id"] == 24
    assert results[0]["bm25_score"] == 100
    assert results[0]["vector_score"] == pytest.approx(0.2)
    assert results[0]["hybrid_score"] == pytest.approx(0.76)
    assert [item["chunk_id"] for item in results] == [24, 25, 0, 1, 2]


def test_minmax_uses_full_corpus_extrema_before_candidate_cutoff():
    # 最小值所在的第 6 项不进入 candidate_k=5，但必须参与两路归一化。
    results = retrieval.retrieve(
        "question", _chunks(6), FixedBm25([10, 8, 6, 4, 2, 0]),
        _documents_with_cosines([1, 0.8, 0.6, 0.4, 0.2, -1]), FixedEmbedding(), top_k=1,
    )

    assert [item["chunk_id"] for item in results] == [0, 1, 2, 3, 4]
    assert [item["hybrid_score"] for item in results] == pytest.approx([1, 0.86, 0.72, 0.58, 0.44])


@pytest.mark.parametrize("count, top_k", [(1, 1), (3, 1), (7, 1), (12, 2), (3, 10)])
def test_equal_scores_are_stable_and_candidates_are_bounded(count, top_k):
    chunks = _chunks(count)
    docs = np.tile([1.0, 0.0], (count, 1))
    bm25 = FixedBm25([2.0] * count)
    model = FixedEmbedding()
    first = retrieval.retrieve("same", chunks, bm25, docs, model, top_k)
    second = retrieval.retrieve("same", chunks, bm25, docs, model, top_k)

    assert first == second
    assert [item["chunk_id"] for item in first] == list(range(min(count, top_k * 5)))
    assert all(item["hybrid_score"] == 0 for item in first)
    assert all(item["vector_score"] == 1 for item in first)


@pytest.mark.parametrize("question", [None, "", " \n\t "])
def test_empty_query_does_not_score_or_encode(question):
    assert retrieval.retrieve(question, _chunks(1), None, None, None) == []


def test_empty_corpus_does_not_require_models_or_matrix():
    assert retrieval.retrieve("question", [], None, None, None) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_top_k_returns_no_candidates(top_k):
    assert retrieval.retrieve("question", _chunks(1), None, None, None, top_k) == []


@pytest.mark.parametrize("documents, message", [
    (None, "缺少必需的文档 embeddings"),
    ([], "非空二维矩阵"),
    ([1, 0], "非空二维矩阵"),
    ([[], []], "非空二维矩阵"),
    ([[1, 0]], "数量与 chunk 数量不一致"),
    ([[1, 0], [0, 0]], "零向量"),
    ([[1, 0], [np.nan, 1]], "非有限"),
    ([[1, 0], [np.inf, 1]], "非有限"),
    ([[1, 0], [-np.inf, 1]], "非有限"),
    ([[1, 0], ["invalid", 1]], "could not convert"),
    ([[1, 0], [1]], "inhomogeneous"),
])
def test_invalid_document_vectors_fail_explicitly(documents, message):
    with pytest.raises(ValueError, match=message):
        retrieval.retrieve("question", _chunks(2), FixedBm25([0, 1]), documents, FixedEmbedding())


@pytest.mark.parametrize("query, message", [
    (None, "一维向量"),
    ([], "维度"),
    ([1, 0, 0], "维度"),
    ([[1, 0]], "一维向量"),
    ([0, 0], "零向量"),
    ([np.nan, 1], "非有限"),
    ([np.inf, 1], "非有限"),
    ([-np.inf, 1], "非有限"),
])
def test_invalid_query_vectors_fail_explicitly(query, message):
    with pytest.raises(ValueError, match=message):
        retrieval.retrieve("question", _chunks(2), FixedBm25([0, 1]), np.eye(2), FixedEmbedding(query))


@pytest.mark.parametrize("scores, message", [
    ([0], "数量与 chunk 数量一致"),
    ([[0, 1]], "一维数组"),
    ([0, np.nan], "有限数值"),
    ([0, np.inf], "有限数值"),
    ([0, -np.inf], "有限数值"),
])
def test_invalid_bm25_scores_fail_explicitly(scores, message):
    model = FixedEmbedding()
    with pytest.raises(ValueError, match=message):
        retrieval.retrieve("question", _chunks(2), FixedBm25(scores), np.eye(2), model)
    assert model.queries == []


@pytest.mark.parametrize("missing", ["bm25", "model"])
def test_missing_required_index_component_fails_explicitly(missing):
    with pytest.raises(ValueError, match="缺少必需的 BM25 索引或 embedding 模型"):
        retrieval.retrieve(
            "question", _chunks(2), None if missing == "bm25" else FixedBm25([0, 1]),
            np.eye(2), None if missing == "model" else FixedEmbedding(),
        )


@pytest.mark.parametrize("values, expected", [
    ([], []), ([3, 3, 3], [0, 0, 0]), ([-2, 0, 2], [0, 0.5, 1]),
    ([-3e38, 0, 3e38], [0, 0.5, 1]),
])
def test_minmax_handles_empty_equal_and_extreme_scores(values, expected):
    np.testing.assert_allclose(retrieval.minmax_score(np.array(values)), expected)


def test_real_retrieval_and_rerank_preserve_tool_top_k_and_evidence(monkeypatch, tool_registry):
    from tests.support.local_tools import invoke_local_tool
    from app.tools.knowledge_base import chunk_to_evidence_document

    chunks = _chunks(7)
    monkeypatch.setattr(store, "chunks", chunks)
    monkeypatch.setattr(store, "bm25", FixedBm25([7, 6, 5, 4, 3, 2, 1]))
    monkeypatch.setattr(store, "doc_embeddings", np.tile([1.0, 0.0], (7, 1)))
    monkeypatch.setattr(store, "model", FixedEmbedding())
    monkeypatch.setattr(store, "RAG_READY", True)
    calls = []

    class FixedCrossEncoder:
        def predict(self, pairs):
            calls.append(pairs)
            # 让融合排名第 5 的候选胜出，证明先重排再截取 top_k=1。
            return [0.1, 0.2, 0.3, 0.4, 0.9]

    monkeypatch.setattr(store, "rerank_model", FixedCrossEncoder())
    _forbid_chroma_reads(monkeypatch)
    result = invoke_local_tool(
        registry=tool_registry, tool_name="search_knowledge_base",
        tool_args={"query": "question", "top_k": 1},
    )

    assert calls == [[("question", chunk["text"]) for chunk in chunks[:5]]]
    assert result["ok"] is True and result["status"] == "success"
    assert result["tool_input"] == {"query": "question", "top_k": 1}
    output = result["output"]
    assert set(output) == {"query", "count", "evidence"}
    assert output["query"] == "question"
    assert output["count"] == len(output["evidence"]) == 1
    evidence = output["evidence"][0]
    assert evidence["page_content"] == chunks[4]["text"]
    assert evidence["metadata"]["chunk_id"] == 4
    assert evidence["metadata"]["rerank_score"] == 0.9
    assert evidence["metadata"]["vector_score"] == 1.0
    expected = chunk_to_evidence_document(chunks[4]).model_dump(mode="json")
    assert evidence["id"] == expected["id"]


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

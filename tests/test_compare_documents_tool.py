"""双文档检索、共享状态隔离及真实 Agent structured grounding。"""

import json
from copy import deepcopy
from unittest.mock import Mock, call

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

import app.tools.compare_documents as compare_documents_module
from app.agent import framework_agent, framework_tools
from app.agent.final_response import FinalAgentResponse
from app.config import get_settings
from app.llm.gateway import ModelGateway
from app.rag import pipeline
from app.tools.audit import summarize_tool_output
from app.tools.catalog import get_local_tool_specs
from app.tools.context import build_default_tool_context
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.schemas import CompareDocumentsArgs, CompareDocumentsOutput
from tests.support.agent_runtime import FakeToolCallingModel, build_agent_harness
from tests.support.local_tools import invoke_local_tool, local_tools


compare_documents_tool = compare_documents_module.compare_documents_tool


def _validated_compare_args(values: dict | None = None, **overrides) -> dict:
    """为 handler 业务单测构造已经过 Runtime 同款 Schema 校验的参数。"""
    payload = dict(values or {})
    payload.update(overrides)
    return CompareDocumentsArgs.model_validate(payload).model_dump(mode="json")


@pytest.mark.parametrize(
    "tool_args",
    [
        {"query": "", "doc_a": "old.md", "doc_b": "new.md"},
        {"query": "   ", "doc_a": "old.md", "doc_b": "new.md"},
        {"query": "对比主题", "doc_a": "", "doc_b": "new.md"},
        {"query": "对比主题", "doc_a": "   ", "doc_b": "new.md"},
        {"query": "对比主题", "doc_a": "old.md", "doc_b": ""},
        {"query": "对比主题", "doc_a": "old.md", "doc_b": "   "},
        {
            "query": "对比主题",
            "doc_a": "old.md",
            "doc_b": "new.md",
            "top_k": 0,
        },
        {
            "query": "对比主题",
            "doc_a": "old.md",
            "doc_b": "new.md",
            "top_k": 11,
        },
        {
            "query": "对比主题",
            "doc_a": "old.md",
            "doc_b": "new.md",
            "top_k": "bad",
        },
    ],
)
def test_compare_documents_rejects_invalid_args_at_runtime(tool_args):
    """基础字段和范围错误由 StructuredTool/Pydantic 边界拒绝。"""
    with pytest.raises(ValidationError):
        invoke_local_tool(
            registry=get_local_tool_specs(get_settings()),
            tool_name="compare_documents",
            tool_args=tool_args,
        )


def test_compare_documents_same_docs_returns_error():
    """验证两个文档名相同时返回工具错误。"""
    result = compare_documents_tool(
        _validated_compare_args(
            query="\u6700\u957f\u5b66\u4e60\u5e74\u9650",
            doc_a="old.md",
            doc_b="old",
        )
    )

    assert result["ok"] is False
    assert "doc_a" in result["error"]
    assert "doc_b" in result["error"]


@pytest.fixture
def compare_retrieval(monkeypatch):
    chunks = [
        {
            "text": text,
            "metadata": {
                "file_name": f"{name}.pdf", "source_id": f"source-{name}",
                "source_uri": f"file:///data/{name}.pdf", "chunk_id": f"{name}{i}",
                "position": i, "page_number": i, "source_type": "local_file",
            },
        }
        for name, texts in (
            ("学生手册", ["普通本科最长修读八年。", "休学期间计入修读年限。", "特殊情况须经批准。"]),
            ("学籍规定", ["普通本科最长修读六年。", "延期须经学院批准。", "适用于本年度入学学生。"]),
        )
        for i, text in enumerate(texts, 1)
    ]
    scores = dict(zip((chunk["text"] for chunk in chunks), [0.9, 0.8, 0.1, 0.9, 0.8, 0.1]))
    chunks = [chunks[i] for i in (2, 0, 1, 5, 3, 4)]
    encoder = Mock()
    encoder.predict.side_effect = lambda pairs: [scores[text] for _, text in pairs]
    monkeypatch.setattr(pipeline, "chunks", chunks)
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "rerank_model", encoder)
    forbidden = Mock(side_effect=AssertionError("compare must not construct a nested LLM"))
    monkeypatch.setattr(ModelGateway, "__init__", forbidden)
    monkeypatch.setattr("app.llm.factory._create_model", forbidden)
    yield chunks, encoder
    forbidden.assert_not_called()


def _compare_args(**overrides):
    # 查询与中文原文无词面重合，仍必须返回两份文档各自的 Top-K。
    return _validated_compare_args(
        {"query": "enrollment duration", "doc_a": "《学生手册》",
         "doc_b": "/data/学籍规定.pdf", "top_k": 2}, **overrides,
    )


def test_compare_documents_returns_ranked_documents_without_mutation(compare_retrieval):
    chunks, encoder = compare_retrieval
    original = deepcopy(chunks)
    args = _compare_args()
    result = compare_documents_tool(args)
    assert result["ok"] is True
    output = result["output"]
    assert set(output) == {
        "query", "doc_a", "doc_b", "doc_a_count", "doc_b_count", "count", "evidence",
    }
    assert (output["doc_a_count"], output["doc_b_count"], output["count"]) == (2, 2, 4)
    assert [doc.metadata["chunk_id"] for doc in output["evidence"]] == [
        "学生手册1", "学生手册2", "学籍规定1", "学籍规定2",
    ]
    for index, document in enumerate(output["evidence"]):
        side = "doc_a" if index < 2 else "doc_b"
        assert isinstance(document, Document)
        assert document.metadata["comparison_side"] == side
        assert document.metadata["comparison_document"] == args[side]
        assert document.metadata["rerank_score"] == [0.9, 0.8][index % 2]
        source = chunks[[1, 2, 4, 5][index]]
        assert document.page_content == source["text"]
        for key in ("file_name", "source_uri", "chunk_id", "position", "page_number"):
            assert document.metadata[key] == source["metadata"][key]
    assert encoder.predict.call_args_list == [
        call([(args["query"], chunk["text"]) for chunk in chunks[:3]]),
        call([(args["query"], chunk["text"]) for chunk in chunks[3:]]),
    ]
    assert pipeline.chunks is chunks and chunks == original
    assert CompareDocumentsOutput.model_validate(output).count == 4


def test_compare_isolates_candidate_and_document_metadata(monkeypatch, compare_retrieval):
    chunks, _ = compare_retrieval
    original = deepcopy(chunks)
    rerank = pipeline.rerank
    convert = compare_documents_module.chunk_to_evidence_document
    converted = []

    def rerank_with_metadata(candidates, query):
        for candidate in candidates:
            candidate["metadata"]["rerank_marker"] = "local"
        return rerank(candidates, query)

    def capture_document(chunk):
        document = convert(chunk)
        converted.append(document)
        return document

    monkeypatch.setattr(pipeline, "rerank", rerank_with_metadata)
    monkeypatch.setattr(compare_documents_module, "chunk_to_evidence_document", capture_document)
    result = compare_documents_tool(_compare_args())
    assert result["ok"] is True
    assert chunks == original
    for view, document in zip(result["output"]["evidence"], converted):
        assert view is not document and view.metadata is not document.metadata
        assert view.id == document.id
        assert "comparison_side" not in document.metadata
        assert "comparison_document" not in document.metadata


@pytest.mark.parametrize("source_id", ["stable-source", None])
def test_search_get_chunk_and_compare_share_identity(
    monkeypatch, tool_registry, compare_retrieval, source_id,
):
    chunks, _ = compare_retrieval
    chunk = chunks[1]
    chunk["metadata"]["chunk_id"] = 382
    if source_id:
        chunk["metadata"]["source_id"] = source_id
    else:
        chunk["metadata"].pop("source_id")
    original = deepcopy(chunks)
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda query, top_k: [deepcopy(chunk)])
    search = invoke_local_tool(tool_registry, "search_knowledge_base", {"query": "修读年限", "top_k": 1})
    lookup = invoke_local_tool(tool_registry, "get_chunk_text", {"chunk_id": 382})
    compare = invoke_local_tool(tool_registry, "compare_documents", _compare_args())
    assert search["ok"] is lookup["ok"] is compare["ok"] is True
    documents = [result["output"]["evidence"][0] for result in (search, lookup, compare)]
    assert len({doc["id"] for doc in documents}) == 1
    assert all(doc["metadata"]["chunk_id"] == 382 for doc in documents)
    assert all(doc["page_content"] == chunk["text"] for doc in documents)
    assert "comparison_side" not in documents[0]["metadata"]
    assert "comparison_side" not in documents[1]["metadata"]
    assert chunks == original


@pytest.mark.parametrize("side", ["doc_a", "doc_b"])
def test_missing_document_returns_tool_error(compare_retrieval, side):
    _, encoder = compare_retrieval
    result = compare_documents_tool(_compare_args(**{side: "missing.pdf"}))
    assert result["ok"] is False
    assert result["error"] == f"{side} 未找到匹配文档 chunk"
    encoder.predict.assert_not_called()


def test_unready_kb_raises_standard_dependency_error(monkeypatch, compare_retrieval):
    _, encoder = compare_retrieval
    monkeypatch.setattr(pipeline, "RAG_READY", False)
    filter_chunks = Mock(side_effect=AssertionError("must not filter an unready index"))
    monkeypatch.setattr(compare_documents_module, "_filter_chunks_by_document", filter_chunks)
    with pytest.raises(ToolExecutionError) as exc:
        compare_documents_tool(_compare_args())
    assert exc.value.code == ToolErrorCode.DEPENDENCY_UNAVAILABLE
    filter_chunks.assert_not_called()
    encoder.predict.assert_not_called()


def test_compare_reloads_current_chunks(monkeypatch, compare_retrieval):
    chunks, _ = compare_retrieval
    assert compare_documents_tool(_compare_args())["ok"] is True
    monkeypatch.setattr(pipeline, "chunks", chunks[3:])
    result = compare_documents_tool(_compare_args())
    assert result["ok"] is False and "doc_a" in result["error"]


def test_document_matching_preserves_exact_precedence_and_fuzzy_fallback():
    chunks = [
        {"file_name": "学生手册补充.pdf"},
        {"metadata": {"source_path": "/data/学生手册.pdf"}},
        {"metadata": {"source": r"C:\data\学籍规定.pdf"}},
    ]
    match = compare_documents_module._filter_chunks_by_document
    assert match(chunks, "《学生手册》") == [chunks[1]]
    assert match(chunks, "手册") == chunks[:2]
    assert match(chunks, "学籍规定.docx") == [chunks[2]]


@pytest.mark.parametrize("update", [
    {"count": 3}, {"doc_a_count": 1}, {"doc_b_count": -1},
    {"comparison": "答案"}, {"conflict_detected": True}, {"summary": "结论"},
    {"doc_a_evidence": []}, {"doc_b_evidence": []},
])
def test_compare_output_rejects_inconsistent_counts_and_legacy_fields(compare_retrieval, update):
    output = compare_documents_tool(_compare_args())["output"]
    with pytest.raises(ValidationError):
        CompareDocumentsOutput.model_validate({**output, **update})


def test_compare_audit_limits_metadata_and_excludes_body(tool_registry, compare_retrieval):
    output = CompareDocumentsOutput.model_validate(
        compare_documents_tool(_compare_args())["output"],
    ).model_dump(mode="json")
    output.update(evidence=output["evidence"] * 5, doc_a_count=10, doc_b_count=10, count=20)
    spec = next(spec for spec in tool_registry if spec.name == "compare_documents")
    summary = summarize_tool_output(spec, output)
    assert summary["count"] == 20 and summary["doc_a_count"] == summary["doc_b_count"] == 10
    assert len(summary["top_chunk_ids"]) == len(summary["top_rerank_scores"]) == 10
    assert "page_content" not in str(summary) and "evidence" not in summary
    assert all(doc["page_content"] not in str(summary) for doc in output["evidence"])


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["grounded", "missing_ids", "unknown_id", "no_evidence", "missing_doc", "unready"])
async def test_compare_evidence_in_real_agent_loop(
    monkeypatch, settings_factory, compare_retrieval, outcome,
):
    chunks, encoder = compare_retrieval
    settings = settings_factory(agent_max_tool_calls=1)
    spec = next(spec for spec in get_local_tool_specs(settings) if spec.name == "compare_documents")
    audit = []
    sink = Mock()
    sink.write.side_effect = audit.append
    monkeypatch.setattr(framework_tools, "get_default_tool_audit_sink", lambda settings: sink)
    tools = local_tools([spec], settings)
    monkeypatch.setattr(framework_tools, "get_local_tools", lambda settings: tools)
    args = _compare_args(doc_b="missing.pdf") if outcome == "missing_doc" else _compare_args()
    if outcome == "unready":
        monkeypatch.setattr(pipeline, "RAG_READY", False)

    def answer(model, messages):
        message = messages[-1]
        assert isinstance(message, ToolMessage) and message.name == "compare_documents"
        payload = json.loads(message.content)
        if outcome in {"missing_doc", "unready"}:
            assert message.status == "error"
            assert payload["error_code"] == ("dependency_unavailable" if outcome == "unready" else "execution_error")
            return model.final_response(outcome="no_evidence", answer="当前无法取得两份文档。")
        assert message.status == "success" and payload["ok"] is True
        output = payload["output"]
        documents = [Document.model_validate(doc) for doc in output["evidence"]]
        assert output["doc_a_count"] == output["doc_b_count"] == 2 and output["count"] == 4
        assert [doc.metadata["comparison_side"] for doc in documents] == ["doc_a"] * 2 + ["doc_b"] * 2
        assert not {"comparison", "conflict_detected", "summary"} & output.keys()
        if outcome == "no_evidence":
            return model.final_response(outcome="no_evidence", answer="现有证据不足以确定是否适用。")
        ids = [documents[0].id, documents[3].id]
        if outcome == "missing_ids":
            ids = []
        elif outcome == "unknown_id":
            ids = ["knowledge_base:fake"]
        return model.final_response(
            outcome="answer", answer="学生手册规定最长八年；学籍规定要求延期经学院批准。", evidence_ids=ids,
        )

    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{
            "name": "compare_documents", "args": args, "id": "compare-1", "type": "tool_call",
        }]), answer,
    ])
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    context = build_default_tool_context(trace_id="trace-compare", settings=settings)
    output = [state async for state in build_agent_harness(settings).astream({
        "original_question": "比较两份文档的修读年限和延期要求",
        "effective_question": "比较两份文档的修读年限和延期要求",
    }, context=context)][-1]
    result = output["result"]
    assert len(model.received_messages) == result["llm_call_count"] == 2
    assert isinstance(output["structured_response"], FinalAgentResponse)
    assert {event.trace_id for event in audit} == {"trace-compare"}
    assert {event.tool_name for event in audit} == {"compare_documents"}
    assert all(chunk["text"] not in str(audit) for chunk in chunks)
    assert all(chunk["text"] not in str(result["tool_trace"]) for chunk in chunks)
    if outcome == "grounded":
        assert result["status"] == "OK"
        assert result["answer"] == output["structured_response"].answer
        assert [ref["reference_id"] for ref in result["references"]] == output["structured_response"].evidence_ids
        assert [ref["chunk_id"] for ref in result["references"]] == ["学生手册1", "学籍规定2"]
        assert result["tool_result_summary"] == "双文档证据检索完成"
        assert audit[-1].sanitized_output_summary == {
            "doc_a": args["doc_a"], "doc_b": args["doc_b"],
            "doc_a_count": 2, "doc_b_count": 2, "count": 4,
            "top_chunk_ids": ["学生手册1", "学生手册2", "学籍规定1", "学籍规定2"],
            "top_rerank_scores": [0.9, 0.8, 0.9, 0.8],
        }
    elif outcome in {"missing_ids", "unknown_id"}:
        assert result["status"] == "MODEL_ERROR" and result["references"] == []
        assert output["structured_response"].answer not in str(result)
    else:
        assert result["status"] == "NO_RELEVANT_INFO" and result["references"] == []
    if outcome in {"missing_doc", "unready"}:
        encoder.predict.assert_not_called()
    assert result["answer_confidence"] == result["retrieval_confidence"] == 0.0

import app.rag as rag
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.tools import compare_documents as compare_documents_module
from app.tools import dispatch as tool_dispatch


compare_documents_tool = compare_documents_module.compare_documents_tool


def test_compare_documents_query_empty_returns_error():
    """验证 query 为空时返回工具错误。"""
    result = compare_documents_tool({
        "query": "",
        "doc_a": "old.md",
        "doc_b": "new.md",
    })

    assert result["ok"] is False
    assert result["status"] == "error"
    assert "query" in result["error"]


def test_compare_documents_doc_a_empty_returns_error():
    """验证 doc_a 为空时返回工具错误。"""
    result = compare_documents_tool({
        "query": "\u6700\u957f\u5b66\u4e60\u5e74\u9650",
        "doc_a": "",
        "doc_b": "new.md",
    })

    assert result["ok"] is False
    assert "doc_a" in result["error"]


def test_compare_documents_doc_b_empty_returns_error():
    """验证 doc_b 为空时返回工具错误。"""
    result = compare_documents_tool({
        "query": "\u6700\u957f\u5b66\u4e60\u5e74\u9650",
        "doc_a": "old.md",
        "doc_b": "",
    })

    assert result["ok"] is False
    assert "doc_b" in result["error"]


def test_compare_documents_same_docs_returns_error():
    """验证两个文档名相同时返回工具错误。"""
    result = compare_documents_tool({
        "query": "\u6700\u957f\u5b66\u4e60\u5e74\u9650",
        "doc_a": "old.md",
        "doc_b": "old",
    })

    assert result["ok"] is False
    assert "doc_a" in result["error"]
    assert "doc_b" in result["error"]


def test_compare_documents_invalid_top_k_returns_tool_error():
    """验证 top_k 非整数时返回工具错误且不抛异常。"""
    result = compare_documents_tool({
        "query": "\u6700\u957f\u5b66\u4e60\u5e74\u9650",
        "doc_a": "old.md",
        "doc_b": "new.md",
        "top_k": "bad",
    })

    assert result["ok"] is False
    assert result["status"] == "error"
    assert "top_k" in result["error"]


def test_compare_documents_success(monkeypatch):
    """验证 compare_documents 能基于假 chunks 返回双文档证据和对比结果。"""
    fake_chunks = [
        {
            "chunk_id": 1,
            "text": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u5b66\u751f\u5b66\u4e60\u5e74\u9650\u4e3a 3 \u81f3 8 \u5e74\u3002",
            "metadata": {"file_name": "old.md", "chunk_id": 1},
        },
        {
            "chunk_id": 2,
            "text": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u5b66\u751f\u5b66\u4e60\u5e74\u9650\u8c03\u6574\u4e3a 3 \u81f3 6 \u5e74\u3002",
            "metadata": {"file_name": "new.md", "chunk_id": 2},
        },
        {
            "chunk_id": 3,
            "text": "\u4e0e\u67e5\u8be2\u65e0\u5173\u7684\u7247\u6bb5\u3002",
            "metadata": {"file_name": "new.md", "chunk_id": 3},
        },
    ]
    monkeypatch.setattr(rag, "chunks", fake_chunks, raising=False)
    monkeypatch.setattr(
        compare_documents_module,
        "_call_compare_documents_llm",
        lambda **kwargs: {
            "comparison": "旧文档规定为 3 至 8 年，新文档调整为 3 至 6 年。",
            "conflict_detected": True,
            "summary": "最长学习年限由 8 年缩短为 6 年。",
        },
    )

    result = compare_documents_tool({
        "query": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u5b66\u4e60\u5e74\u9650",
        "doc_a": "old.md",
        "doc_b": "new.md",
        "top_k": 2,
    })

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["tool_name"] == "compare_documents"

    output = result["output"]
    assert output["doc_a_evidence"]
    assert output["doc_b_evidence"]
    assert "comparison" in output
    assert "conflict_detected" in output
    assert output["conflict_detected"] is True
    assert output["comparison"] == "旧文档规定为 3 至 8 年，新文档调整为 3 至 6 年。"
    assert output["summary"] == "最长学习年限由 8 年缩短为 6 年。"
    assert output["doc_a_evidence"][0]["chunk_id"] == 1
    assert output["doc_b_evidence"][0]["chunk_id"] == 2


def test_compare_documents_same_numeric_terms_not_conflict(monkeypatch):
    """验证同一规定值的不同措辞不会仅因文本不完全一致而判定冲突。"""
    fake_chunks = [
        {
            "chunk_id": 1,
            "text": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u5b66\u751f\u6700\u957f\u5b66\u4e60\u5e74\u9650\u4e3a 3 \u81f3 6 \u5e74\u3002",
            "metadata": {"file_name": "old.md", "chunk_id": 1},
        },
        {
            "chunk_id": 2,
            "text": "\u6839\u636e\u901a\u77e5\uff0c\u56db\u5e74\u5236\u666e\u901a\u672c\u79d1\u6700\u957f\u5b66\u4e60\u5e74\u9650\u4e3a 3\u81f36 \u5e74\u3002",
            "metadata": {"file_name": "new.md", "chunk_id": 2},
        },
    ]
    monkeypatch.setattr(rag, "chunks", fake_chunks, raising=False)
    monkeypatch.setattr(
        compare_documents_module,
        "_call_compare_documents_llm",
        lambda **kwargs: {
            "comparison": "两份文档均规定普通本科四年制最长学习年限为 3 至 6 年。",
            "conflict_detected": False,
            "summary": "两份文档关于最长学习年限的规定一致。",
        },
    )

    result = compare_documents_tool({
        "query": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u6700\u957f\u5b66\u4e60\u5e74\u9650",
        "doc_a": "old.md",
        "doc_b": "new.md",
    })

    assert result["ok"] is True
    assert result["output"]["conflict_detected"] is False
    assert result["output"]["summary"] == "\u4e24\u4efd\u6587\u6863\u5173\u4e8e\u6700\u957f\u5b66\u4e60\u5e74\u9650\u7684\u89c4\u5b9a\u4e00\u81f4\u3002"


def test_compare_documents_returns_error_when_llm_fails(monkeypatch):
    """验证 LLM 调用失败时返回工具错误，不再用规则兜底生成差异。"""
    fake_chunks = [
        {
            "chunk_id": 1,
            "text": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u6700\u957f\u5b66\u4e60\u5e74\u9650\u4e3a 3 \u81f3 8 \u5e74\u3002",
            "metadata": {"file_name": "old.md", "chunk_id": 1},
        },
        {
            "chunk_id": 2,
            "text": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u6700\u957f\u5b66\u4e60\u5e74\u9650\u4e3a 3 \u81f3 6 \u5e74\u3002",
            "metadata": {"file_name": "new.md", "chunk_id": 2},
        },
    ]
    monkeypatch.setattr(rag, "chunks", fake_chunks, raising=False)

    def fake_llm(**kwargs):
        """模拟 LLM 调用失败。"""
        raise RuntimeError("llm down")

    monkeypatch.setattr(
        compare_documents_module,
        "_call_compare_documents_llm",
        fake_llm,
    )

    result = compare_documents_tool({
        "query": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u6700\u957f\u5b66\u4e60\u5e74\u9650",
        "doc_a": "old.md",
        "doc_b": "new.md",
    })

    assert result["ok"] is False
    assert result["status"] == "error"
    assert "LLM \u5bf9\u6bd4\u751f\u6210\u5931\u8d25" in result["error"]


def test_compare_llm_uses_compare_factory_and_parses_business_fields(monkeypatch):
    """文档对比通过独立工厂调用，并保留 JSON 字段解析。"""

    class FakeCompareModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(
                content=(
                    '{"comparison":"存在差异",'
                    '"conflict_detected":true,'
                    '"summary":"摘要"}'
                )
            )

    model = FakeCompareModel()
    monkeypatch.setattr(
        compare_documents_module,
        "get_compare_model",
        lambda: model,
    )

    result = compare_documents_module._call_compare_documents_llm(
        query="对比主题",
        doc_a="a.md",
        doc_b="b.md",
        doc_a_evidence=[{"chunk_id": 1, "snippet": "证据 A"}],
        doc_b_evidence=[{"chunk_id": 2, "snippet": "证据 B"}],
    )

    assert result == {
        "comparison": "存在差异",
        "conflict_detected": True,
        "summary": "摘要",
    }
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[0][1], HumanMessage)
    assert "证据 A" in model.calls[0][1].content
    assert "证据 B" in model.calls[0][1].content


def test_dispatch_compare_documents_propagates_trace_id(monkeypatch, tmp_path):
    """请求 trace_id 经分发器和注册表传到文档对比模型网关。"""
    fake_chunks = [
        {
            "chunk_id": 1,
            "text": "普通本科四年制最长学习年限为 3 至 8 年。",
            "metadata": {"file_name": "old.md", "chunk_id": 1},
        },
        {
            "chunk_id": 2,
            "text": "普通本科四年制最长学习年限为 3 至 6 年。",
            "metadata": {"file_name": "new.md", "chunk_id": 2},
        },
    ]
    received = {}

    def fake_invoke_text(self, messages, *, trace_id=""):
        received["trace_id"] = trace_id
        return (
            '{"comparison":"存在差异",'
            '"conflict_detected":true,"summary":"摘要"}'
        )

    monkeypatch.setattr(rag, "chunks", fake_chunks, raising=False)
    monkeypatch.setattr(
        compare_documents_module.ModelGateway,
        "invoke_text",
        fake_invoke_text,
    )
    result = tool_dispatch.dispatch_tool(
        "compare_documents",
        {
            "query": "普通本科四年制最长学习年限",
            "doc_a": "old.md",
            "doc_b": "new.md",
        },
        trace_id="trace-tool",
    )

    assert result["ok"] is True
    assert received["trace_id"] == "trace-tool"

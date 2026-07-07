from app.tool_registry import tool_registry
from app.tools import dispatch_tool


def test_tool_registry_names_include_builtin_tools():
    """验证内置工具已注册到统一注册中心。"""
    names = tool_registry.names()

    assert "search_knowledge_base" in names
    assert "simple_calculator" in names
    assert "list_documents" in names
    assert "get_chunk_text" in names
    assert "compare_documents" in names


def test_search_knowledge_base_registered():
    """验证知识库检索已作为标准工具注册。"""
    assert "search_knowledge_base" in tool_registry.names()


def test_router_specs_include_search_knowledge_base():
    """验证 Router 工具描述包含知识库检索工具。"""
    specs = tool_registry.router_specs()
    names = {item["name"] for item in specs}

    assert "search_knowledge_base" in names


def test_tool_registry_router_specs_shape():
    """验证 Router 可读取注册工具描述。"""
    specs = tool_registry.router_specs()

    assert isinstance(specs, list)
    assert specs

    for spec in specs:
        assert "name" in spec
        assert "description" in spec
        assert "args_schema" in spec


def test_tool_registry_dispatch_simple_calculator():
    """验证注册中心可以调用计算器工具。"""
    result = tool_registry.dispatch(
        "simple_calculator",
        {"expression": "1 + 2 * 3"}
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"]["result"] == 7


def test_tool_registry_dispatch_search_knowledge_base(monkeypatch):
    """验证注册中心可以调用知识库检索工具。"""
    import app.rag as rag

    def fake_search_relevant_chunks(query, top_k=5):
        """模拟知识库检索返回单个候选 chunk。"""
        assert query == "普通本科四年制专业最长学习年限是多少？"
        assert top_k == 5
        return [
            {
                "chunk_id": 1,
                "text": "测试 chunk",
                "hybrid_score": 0.91,
            }
        ]

    monkeypatch.setattr(
        rag,
        "search_relevant_chunks",
        fake_search_relevant_chunks,
        raising=False
    )

    result = tool_registry.dispatch(
        "search_knowledge_base",
        {
            "query": "普通本科四年制专业最长学习年限是多少？",
            "top_k": 5,
        }
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["tool_name"] == "search_knowledge_base"
    assert result["tool_input"] == {
        "query": "普通本科四年制专业最长学习年限是多少？",
        "top_k": 5,
    }
    assert result["output"]["count"] == 1
    assert result["output"]["chunks"][0]["chunk_id"] == 1
    assert result["output"]["top_chunk_ids"] == [1]
    assert result["output"]["top_scores"] == [0.91]


def test_tool_registry_dispatch_unknown_tool():
    """验证未知工具返回兼容错误结构。"""
    result = tool_registry.dispatch("unknown_tool", {})

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["tool_name"] == "unknown_tool"
    assert result["tool_input"] == {}
    assert result["output"] is None
    assert result["error"] == "unknown tool: unknown_tool"


def test_dispatch_tool_keeps_compatible_entrypoint(monkeypatch, tmp_path):
    """验证旧的 dispatch_tool 入口仍可调用注册工具。"""
    import app.tools as tools

    monkeypatch.setattr(
        tools,
        "TOOL_CALLS_PATH",
        tmp_path / "tool_calls.jsonl",
        raising=False
    )

    result = dispatch_tool(
        "simple_calculator",
        {"expression": "2 + 3"},
        trace_id="test_trace"
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"]["result"] == 5


def test_compare_documents_registered():
    """验证 compare_documents 已注册到工具注册表。"""
    assert "compare_documents" in tool_registry.names()


def test_router_specs_include_compare_documents():
    """验证 Router 工具描述中包含 compare_documents。"""
    names = {item["name"] for item in tool_registry.router_specs()}
    assert "compare_documents" in names

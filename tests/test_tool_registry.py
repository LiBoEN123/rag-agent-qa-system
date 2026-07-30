import json

from app.tools.registry import ToolRegistry, ToolSpec
from app.tools.context import ToolInvocationContext
from app.tools.schemas import SearchKnowledgeBaseArgs
from app.tools.dispatch import dispatch_tool


def test_tool_registry_names_include_builtin_tools(tool_registry):
    """验证内置工具已注册到统一注册中心。"""
    names = tool_registry.names()

    assert "search_knowledge_base" in names
    assert "simple_calculator" in names
    assert "list_documents" in names
    assert "get_chunk_text" in names
    assert "compare_documents" in names
    assert "web_search" in names
    assert "web_fetch" in names
    assert "web_pdf_fetch" in names


def test_builtin_tools_publish_v2_contract_metadata(tool_registry):
    """验证八个注册生产工具发布完整且符合要求的 V2 契约。"""
    specs = tool_registry.contract_specs()
    specs_by_name = {spec["name"]: spec for spec in specs}
    expected_permissions = {
        "search_knowledge_base": ["knowledge_base:read"],
        "compare_documents": ["knowledge_base:read"],
        "simple_calculator": [],
        "list_documents": ["knowledge_base:read"],
        "get_chunk_text": ["knowledge_base:read"],
        "web_search": ["web:search"],
        "web_fetch": ["web:fetch"],
        "web_pdf_fetch": ["web:fetch"],
    }
    expected_timeouts = {
        "search_knowledge_base": 10.0,
        "compare_documents": 20.0,
        "simple_calculator": 3.0,
        "list_documents": 5.0,
        "get_chunk_text": 5.0,
        "web_search": 10.0,
        "web_fetch": 30.0,
        "web_pdf_fetch": 45.0,
    }

    assert len(specs) == len(tool_registry.names())
    assert set(specs_by_name) == set(tool_registry.names())
    assert [spec["name"] for spec in specs] == tool_registry.names()
    json.dumps(specs)

    for name, spec in specs_by_name.items():
        assert spec["version"] == "1.0.0"
        assert spec["risk_level"] == "read_only"
        assert spec["required_permissions"] == expected_permissions[name]
        assert spec["timeout"] == expected_timeouts[name]
        assert spec["max_retries"] == (
            1 if name in {"web_search", "web_fetch", "web_pdf_fetch"} else 0
        )
        assert spec["idempotent"] is True
        assert spec["requires_approval"] is False
        assert spec["output_model"] == (
            {
                "web_search": "WebSearchOutput",
                "web_fetch": "WebFetchOutput",
                "web_pdf_fetch": "WebPdfFetchOutput",
            }.get(name)
        )
        assert "handler" not in spec

    web = specs_by_name["web_search"]
    assert web["sensitive_arg_fields"] == ["query"]
    assert web["audit_output_mode"] == "summary"
    fetch = specs_by_name["web_fetch"]
    assert fetch["sensitive_arg_fields"] == ["url"]
    assert fetch["audit_output_mode"] == "summary"
    pdf = specs_by_name["web_pdf_fetch"]
    assert pdf["sensitive_arg_fields"] == ["url"]
    assert pdf["audit_output_mode"] == "summary"


def test_tool_registry_dispatch_simple_calculator(tool_registry):
    """验证注册中心可以调用计算器工具。"""
    result = tool_registry.dispatch(
        "simple_calculator",
        {"expression": "1 + 2 * 3"}
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"]["result"] == 7


def test_tool_registry_dispatch_search_knowledge_base(
    monkeypatch,
    tool_registry,
):
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


def test_tool_registry_dispatch_unknown_tool(tool_registry):
    """验证未知工具返回兼容错误结构。"""
    result = tool_registry.dispatch("unknown_tool", {})

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["tool_name"] == "unknown_tool"
    assert result["tool_input"] == {}
    assert result["output"] is None
    assert result["error"] == "unknown tool: unknown_tool"
    assert result["error_code"] == "unknown_tool"
    assert result["retryable"] is False
    assert result["runtime"]["attempt_count"] == 0


def test_dispatch_validates_args_before_handler():
    """验证 dispatch 在调用 handler 前执行参数校验。"""
    registry = ToolRegistry()
    seen = {}

    def fake_handler(args, context):
        seen["args"] = args
        seen["context"] = context
        return {
            "status": "success",
            "ok": True,
            "tool_name": "search_knowledge_base",
            "tool_input": args,
            "output": {"count": 0},
            "error": "",
        }

    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            version="1.0.0",
            description="检索知识库",
            args_schema={"query": "string", "top_k": "integer"},
            output_schema={},
            handler=fake_handler,
            args_model=SearchKnowledgeBaseArgs,
        )
    )

    context = ToolInvocationContext(trace_id="trace-registry")
    result = registry.dispatch(
        "search_knowledge_base",
        {"query": "测试", "top_k": 5},
        context=context,
    )

    assert result["status"] == "success"
    assert seen["args"] == {"query": "测试", "top_k": 5}
    assert seen["context"] is context


def test_dispatch_returns_tool_error_when_args_invalid():
    """验证参数非法时返回工具错误且不调用 handler。"""
    registry = ToolRegistry()
    called = {"count": 0}

    def fake_handler(args, context):
        called["count"] += 1
        return {"status": "success", "ok": True, "tool_name": "search_knowledge_base"}

    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            version="1.0.0",
            description="检索知识库",
            args_schema={"query": "string", "top_k": "integer"},
            output_schema={},
            handler=fake_handler,
            args_model=SearchKnowledgeBaseArgs,
        )
    )

    result = registry.dispatch(
        "search_knowledge_base",
        {"query": "测试", "top_k": 999},
    )

    assert called["count"] == 0
    assert result["status"] != "success"
    assert result["ok"] is False
    assert "invalid tool args" in result["error"]
    assert result["error_code"] == "invalid_arguments"
    assert result["runtime"]["attempt_count"] == 0


def test_dispatch_tool_keeps_compatible_entrypoint():
    """验证旧的 dispatch_tool 入口仍可调用注册工具。"""
    result = dispatch_tool(
        "simple_calculator",
        {"expression": "2 + 3"},
        trace_id="test_trace"
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"]["result"] == 5

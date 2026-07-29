"""Runtime inspection CLI 的公开架构契约。"""

import json
from importlib import import_module

from pydantic import BaseModel

from app.agent import framework_agent
from app.graph import builder as graph_builder
from app.tools.context import build_default_tool_context
from app.tools.contracts import ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry
from scripts import inspect_agent_runtime as inspection


def _knowledge_only(settings_factory, *permissions):
    """固定文档声明的最小默认权限，隔离本地 .env。"""
    return settings_factory(
        tool_default_permissions=(
            "knowledge_base:read",
            *permissions,
        )
    )


def test_runtime_report_preserves_graph_contract(
    monkeypatch,
    settings_factory,
):
    """报告来自真实编译图，并保留父图与嵌入子图关键边界。"""
    settings = _knowledge_only(settings_factory)
    report = inspection.inspect_runtime(settings=settings)
    parent_nodes = set(report["parent_graph"]["nodes"])
    parent_edges = {
        (edge["source"], edge["target"], edge["conditional"])
        for edge in report["parent_graph"]["edges"]
    }
    framework_nodes = set(report["framework_agent"]["nodes"])
    state_fields = set(report["state_schema"]["fields"])

    assert inspection.EXPECTED_PARENT_NODES <= parent_nodes
    assert parent_edges == inspection.EXPECTED_PARENT_EDGES
    assert ("__start__", "initialize", False) not in parent_edges
    assert ("__start__", "initialize", True) not in parent_edges
    assert inspection.FORBIDDEN_PARENT_NODES.isdisjoint(parent_nodes)
    assert {"model", "tools"} <= framework_nodes
    assert any(
        name.endswith(":model") for name in report["parent_graph"]["xray_nodes"]
    )
    assert any(
        name.endswith(":tools") for name in report["parent_graph"]["xray_nodes"]
    )
    assert any("ModelCallLimitMiddleware" in name for name in framework_nodes)
    assert any("ToolCallLimitMiddleware" in name for name in framework_nodes)
    assert inspection.BUSINESS_STATE_FIELDS <= state_fields
    assert "agent_state" not in state_fields
    assert {
        "single_runtime_state",
        "no_nested_agent_state",
        "state_has_business_fields",
        "agent_result_contract",
        "status_policy_complete",
        "single_status_policy",
        "result_is_checkpoint_safe",
        "api_response_builder_boundary",
        "pydantic_settings_model",
        "lazy_settings",
        "immutable_settings",
        "no_static_settings_class",
        "no_manual_env_parsers",
        "no_import_time_dotenv",
        "import_linter_configured",
        "application_layer_contract",
        "top_level_acyclic_contract",
        "http_framework_boundary",
        "settings_framework_boundary",
        "zero_import_ignores",
    } <= {item["name"] for item in report["checks"]}
    assert report["module_boundaries"] == {
        "import_linter_configured": True,
        "application_layer_contract": True,
        "top_level_acyclic_contract": True,
        "http_framework_boundary": True,
        "settings_framework_boundary": True,
        "zero_import_ignores": True,
    }
    assert report["agent_result_contract"]["model"] == "AgentRunResult"
    assert report["agent_result_contract"]["extra"] == "forbid"
    assert {
        item["result_status"] for item in report["chat_status_policy"]
    } == {
        "OK",
        "NEED_CLARIFICATION",
        "NO_RELEVANT_INFO",
        "TOOL_ERROR",
        "MODEL_ERROR",
        "INTERNAL_ERROR",
    }


def test_tool_contracts_use_model_facing_json_schema(
    monkeypatch,
    settings_factory,
):
    """默认工具输出公开模型 Schema 和治理元数据，不暴露 Runtime 字段。"""
    settings = _knowledge_only(settings_factory)
    report = inspection.inspect_runtime(settings=settings)
    tools = report["registered_tools"]
    context = build_default_tool_context(
        source="agent_runtime_inspection",
        settings=settings,
    )

    assert {tool["name"] for tool in tools} == {
        "search_knowledge_base",
        "web_search",
        "web_fetch",
        "web_pdf_fetch",
        "compare_documents",
        "simple_calculator",
        "list_documents",
        "get_chunk_text",
    }
    assert all(tool["args_schema"]["type"] == "object" for tool in tools)
    assert all("runtime" not in tool["args_schema"].get("properties", {}) for tool in tools)
    assert all(tool["description"] and tool["version"] for tool in tools)
    assert all(tool["risk_level"] == "read_only" for tool in tools)
    assert report["default_visible_tools"] == [
        "search_knowledge_base",
        "compare_documents",
        "simple_calculator",
        "list_documents",
        "get_chunk_text",
    ]
    web = next(tool for tool in tools if tool["name"] == "web_search")
    assert web["required_permissions"] == ["web:search"]
    assert set(web["args_schema"]["properties"]) == {
        "query",
        "max_results",
        "freshness",
    }
    fetch = next(tool for tool in tools if tool["name"] == "web_fetch")
    assert fetch["required_permissions"] == ["web:fetch"]
    assert fetch["output_model"] == "WebFetchOutput"
    assert set(fetch["args_schema"]["properties"]) == {"url", "max_chars"}
    pdf = next(tool for tool in tools if tool["name"] == "web_pdf_fetch")
    assert pdf["required_permissions"] == ["web:fetch"]
    assert pdf["output_model"] == "WebPdfFetchOutput"
    assert set(pdf["args_schema"]["properties"]) == {
        "url",
        "start_page",
        "max_pages",
        "max_chars",
    }
    assert "web_pdf_fetch" not in report["default_visible_tools"]
    assert report["tool_visibility"] == {
        "permission_count": len(context.permissions),
        "registered_tool_count": len(tools),
        "visible_tool_count": len(report["default_visible_tools"]),
    }


def test_json_cli_is_parseable_stable_and_safe(capsys):
    """JSON 输出可稳定解析，且不包含运行时对象表示。"""
    assert inspection.main(["--format", "json"]) == 0
    first = capsys.readouterr().out
    assert inspection.main(["--format", "json"]) == 0
    second = capsys.readouterr().out

    payload = json.loads(first)
    assert first == second
    assert payload["versions"]["langgraph"]
    assert all(item["passed"] for item in payload["checks"])
    assert "object at 0x" not in first
    assert '"handler"' not in first


def test_mermaid_cli_uses_xray_compiled_graph(capsys):
    """Mermaid 文本由真实 xray 图生成并包含父图与框架子图。"""
    assert inspection.main(["--format", "mermaid"]) == 0
    output = capsys.readouterr().out

    assert "graph TD;" in output
    assert "subgraph framework_agent" in output
    assert "route(route)" in output
    assert "rag_subgraph(rag_subgraph)" in output
    assert "ModelCallLimitMiddleware" in output


def test_check_cli_passes_without_extra_output(capsys):
    """当前架构满足检查时返回稳定消息和零退出码。"""
    assert inspection.main(["--check"]) == 0
    captured = capsys.readouterr()

    assert captured.out == "Agent runtime architecture checks passed.\n"
    assert captured.err == ""


def test_inspection_reports_web_search_when_default_permission_is_enabled(
    settings_factory,
):
    """默认授予搜索权限时，内省报告包含 Web 搜索工具。"""
    settings = _knowledge_only(settings_factory, "web:search")
    report = inspection.inspect_runtime(
        settings=settings,
    )

    assert "web_search" in report["default_visible_tools"]
    assert all(item["passed"] for item in report["checks"])


def test_inspection_reports_web_fetch_for_explicit_permission(
    settings_factory,
):
    """未显式授予抓取权限时，内省报告隐藏 Web 抓取工具。"""
    settings = _knowledge_only(settings_factory)
    report = inspection.inspect_runtime(settings=settings)

    assert "web_fetch" not in report["default_visible_tools"]
    assert "web_pdf_fetch" not in report["default_visible_tools"]
    assert all(item["passed"] for item in report["checks"])


def test_inspection_reports_both_fetch_tools_when_permission_is_enabled(
    settings_factory,
):
    """授予抓取权限时，内省报告同时包含网页与 PDF 抓取工具。"""
    settings = _knowledge_only(settings_factory, "web:fetch")
    report = inspection.inspect_runtime(settings=settings)

    assert "web_fetch" in report["default_visible_tools"]
    assert "web_pdf_fetch" in report["default_visible_tools"]
    assert all(item["passed"] for item in report["checks"])


def test_inspection_does_not_initialize_model_checkpointer_or_handler(
    monkeypatch,
    settings_factory,
):
    """内省只读取契约，不初始化模型、Checkpointer 或工具 handler。"""

    class ProbeArgs(BaseModel):
        value: int

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("inspection must not execute runtime dependencies")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="inspection_probe",
            version="1.0.0",
            description="仅用于验证 Runtime inspection 不执行 handler",
            args_schema={"value": "integer"},
            output_schema={"value": "integer"},
            handler=forbidden,
            args_model=ProbeArgs,
        )
    )
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", forbidden)
    monkeypatch.setattr(graph_builder, "get_checkpointer", forbidden)
    monkeypatch.setattr(
        import_module("app.tools.web_download"),
        "Manager",
        forbidden,
    )
    monkeypatch.setattr(
        import_module("app.tools.web_pdf_fetch"),
        "PdfReader",
        forbidden,
    )

    report = inspection.inspect_runtime(
        LocalToolProvider(registry),
        settings=_knowledge_only(settings_factory),
    )

    assert [tool["name"] for tool in report["registered_tools"]] == [
        "inspection_probe"
    ]
    assert report["default_visible_tools"] == ["inspection_probe"]
    assert all(item["passed"] for item in report["checks"])

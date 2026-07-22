"""Runtime inspection CLI 的公开架构契约。"""

import json

from pydantic import BaseModel

from app.agent import framework_agent
from app.config import Settings
from app.graph import builder as graph_builder
from app.tools.context import build_default_tool_context
from app.tools.contracts import ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry
from scripts import inspect_agent_runtime as inspection


def test_runtime_report_preserves_graph_contract():
    """报告来自真实编译图，并保留父图与嵌入子图关键边界。"""
    report = inspection.inspect_runtime()
    parent_nodes = set(report["parent_graph"]["nodes"])
    framework_nodes = set(report["framework_agent"]["nodes"])

    assert inspection.EXPECTED_PARENT_NODES <= parent_nodes
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


def test_tool_contracts_use_model_facing_json_schema():
    """默认工具输出公开模型 Schema 和治理元数据，不暴露 Runtime 字段。"""
    report = inspection.inspect_runtime()
    tools = report["registered_tools"]
    context = build_default_tool_context(source="agent_runtime_inspection")

    assert {tool["name"] for tool in tools} == {
        "search_knowledge_base",
        "web_search",
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
    monkeypatch,
):
    monkeypatch.setattr(
        Settings,
        "TOOL_DEFAULT_PERMISSIONS",
        ("knowledge_base:read", "web:search"),
    )
    report = inspection.inspect_runtime()

    assert "web_search" in report["default_visible_tools"]
    assert all(item["passed"] for item in report["checks"])


def test_inspection_does_not_initialize_model_checkpointer_or_handler(monkeypatch):
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

    report = inspection.inspect_runtime(LocalToolProvider(registry))

    assert [tool["name"] for tool in report["registered_tools"]] == [
        "inspection_probe"
    ]
    assert report["default_visible_tools"] == ["inspection_probe"]
    assert all(item["passed"] for item in report["checks"])

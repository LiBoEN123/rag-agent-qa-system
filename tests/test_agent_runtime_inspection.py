"""Runtime inspection CLI 的公开架构契约。"""

from pydantic import RootModel

import json
from importlib import import_module

from pydantic import BaseModel

from app.tools.catalog import get_local_tool_specs
from app.agent import framework_agent
from app.agent.framework_tools import get_local_tools
from app.tools.context import build_default_tool_context
from app.tools.contracts import ToolSpec
from tests.support.local_tools import local_tools
from scripts import inspect_agent_runtime as inspection


def _knowledge_only(settings_factory, *permissions):
    """固定文档声明的最小默认权限，隔离本地 .env。"""
    return settings_factory(
        tool_default_permissions=(
            "knowledge_base:read",
            *permissions,
        )
    )


def test_runtime_report_preserves_direct_create_agent_contract(
    monkeypatch,
    settings_factory,
):
    """报告来自真实 create_agent，并拒绝 Parent Graph 节点。"""
    settings = _knowledge_only(settings_factory)
    report = inspection.inspect_runtime(settings=settings)
    agent_nodes = set(report["agent_runtime"]["nodes"])
    state_fields = set(report["state_schema"]["fields"])

    assert inspection.EXPECTED_AGENT_NODES <= agent_nodes
    assert inspection.FORBIDDEN_PARENT_NODES.isdisjoint(agent_nodes)
    assert "knowledge_base_result" in state_fields
    assert "framework_events_applied" not in state_fields
    assert {
        "create_agent_is_only_agent_loop",
        "no_parent_agent_graph",
        "router_is_plain_python",
        "runtime_only_prepares_agent_invocation",
        "middleware_does_not_depend_on_runtime",
        "native_tool_middleware_hooks",
        "accepted_model_governance_boundary",
        "no_tool_governance_artifact_protocol",
        "framework_tools_does_not_simulate_middleware",
        "mcp_tool_has_no_framework_rewrapper",
        "result_builder_owns_agent_result",
        "chat_service_consumes_agent_result",
        "streaming_uses_create_agent_directly",
        "rag_graph_is_tool_owned",
        "local_tool_execution_entry_is_unique",
        "local_handler_is_adapter_owned",
        "local_visibility_is_model_middleware",
        "mcp_governance_is_independent",
        "no_parent_event_projection_cursor",
        "agent_result_contract",
        "status_policy_complete",
        "current_architecture_documented",
        "history_archived",
        "import_linter_configured",
        "application_layer_contract",
        "top_level_acyclic_contract",
        "middleware_runtime_boundary",
        "http_framework_boundary",
        "settings_framework_boundary",
        "zero_import_ignores",
        "mcp_disabled_by_default",
        "tool_runtime_governance_preserved",
    } <= {item["name"] for item in report["checks"]}
    assert report["module_boundaries"] == {
        "import_linter_configured": True,
        "application_layer_contract": True,
        "top_level_acyclic_contract": True,
        "middleware_runtime_boundary": True,
        "http_framework_boundary": True,
        "settings_framework_boundary": True,
        "zero_import_ignores": True,
    }
    assert report["agent_result_contract"]["model"] == "AgentRunResult"
    assert report["agent_result_contract"]["extra"] == "forbid"
    assert report["model_governance"] == {
        "decision": "KEEP_CURRENT_MODEL_MARKER",
        "gate_result": "PHASE6B_SKIPPED",
        "adapter_chain": ["_DeferredAgentModel", "_GovernedBoundModel"],
        "marker": "_project_model_governance",
        "state_hooks": ["aafter_model"],
        "model_wrap_hooks_overridden": False,
        "status": "accepted_model_governance_boundary",
    }
    assert report["mcp_client"] == {
        "enabled_by_default": False,
        "config_version": 1,
        "transport": "streamable_http",
        "dependency": "0.3.2",
    }
    assert report["local_tool_execution"] == {
        "catalog": "get_local_tools",
        "visibility": "ProjectToolVisibilityMiddleware",
        "adapter": "ProjectStructuredTool",
        "input_validation": "LangChain StructuredTool/Pydantic",
        "retry": "LangChain ToolRetryMiddleware",
        "timeout": "shared bounded executor helper",
        "terminal": "ToolSpec.handler",
    }
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


def test_all_local_framework_tool_schemas_match_registry_contract(settings_factory):
    """全部七个 Local Tool 保持名称、描述、字段、required 与类型不变。"""
    settings = _knowledge_only(settings_factory, "web:search", "web:fetch")
    registry = get_local_tool_specs(settings)
    tools = get_local_tools(settings)

    assert len(tools) == 7
    for tool in tools:
        spec = next((item for item in registry if item.name == tool.name), None)
        assert spec is not None
        expected = spec.args_model.model_json_schema()
        actual = tool.tool_call_schema.model_json_schema()
        assert tool.name == spec.name
        assert tool.description == spec.description
        assert actual.get("properties") == expected.get("properties")
        assert actual.get("required", []) == expected.get("required", [])


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
    """Mermaid 文本直接来自 create_agent xray 图。"""
    assert inspection.main(["--format", "mermaid"]) == 0
    output = capsys.readouterr().out

    assert "graph TD;" in output
    assert "model" in output
    assert "tools" in output
    assert "rag_subgraph(rag_subgraph)" not in output
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


def test_inspection_does_not_initialize_model_or_handler(
    monkeypatch,
    settings_factory,
):
    """内省只读取契约，不初始化模型或工具 handler。"""

    class ProbeArgs(BaseModel):
        value: int

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("inspection must not execute runtime dependencies")

    registry = []
    registry.append(
        ToolSpec(
            name="inspection_probe",
            version="1.0.0",
            description="仅用于验证 Runtime inspection 不执行 handler",
            output_model=RootModel[dict],
            handler=forbidden,
            args_model=ProbeArgs,
        )
    )
    monkeypatch.setattr(framework_agent, "get_answer_model", forbidden)
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
        local_tools(registry),
        settings=_knowledge_only(settings_factory),
    )

    assert [tool["name"] for tool in report["registered_tools"]] == [
        "inspection_probe"
    ]
    assert report["default_visible_tools"] == ["inspection_probe"]
    assert all(item["passed"] for item in report["checks"])

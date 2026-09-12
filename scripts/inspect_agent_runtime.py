#!/usr/bin/env python3
"""只读检查当前无 Parent Graph 的单一 ``create_agent`` Runtime。"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.framework_agent import (  # noqa: E402
    FrameworkAgentState,
    _DeferredAgentModel,
    _GovernedBoundModel,
    build_framework_agent,
)
from app.agent.framework_middleware import (  # noqa: E402
    _MODEL_GOVERNANCE_KEY,
    ProjectToolGovernanceMiddleware,
)
from app.agent.framework_tools import get_local_tools, local_tool_contract  # noqa: E402
from app.agent.results import AgentRunResult  # noqa: E402
from app.agent.streaming import stream_agent  # noqa: E402
from app.config import AppSettings  # noqa: E402
from app.contracts.chat import CHAT_STATUS_POLICIES  # noqa: E402
from app.tools.context import build_default_tool_context  # noqa: E402
from app.agent.tool_visibility import ProjectToolVisibilityMiddleware  # noqa: E402
from langchain.agents.middleware import ModelRequest  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402
from langchain_core.tools import BaseTool  # noqa: E402


EXPECTED_AGENT_NODES = {"__start__", "model", "tools", "__end__"}
FORBIDDEN_PARENT_NODES = {
    "route",
    "initialize",
    "framework_agent",
    "adapt_framework",
    "rag_subgraph",
    "compose",
    "finalize",
}
FORBIDDEN_PARENT_SYMBOLS = {
    "build_agent_graph",
    "get_compiled_agent_graph",
    "run_agent_langgraph",
    "route_node",
    "initialize_node",
    "adapt_framework_agent_node",
    "rag_subgraph_node",
    "finalize_node",
    "AgentGraphState",
    "project_framework_events",
    "framework_events_applied",
    "Command.PARENT",
}


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _graph_edges(graph) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "source": edge.source,
                "target": edge.target,
                "conditional": bool(edge.conditional),
            }
            for edge in graph.edges
        ),
        key=lambda item: (
            item["source"],
            item["target"],
            item["conditional"],
        ),
    )


def _tool_report(
    tools: tuple[BaseTool, ...], *, settings: AppSettings,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    context = build_default_tool_context(source="agent_runtime_inspection", settings=settings)
    registered = [local_tool_contract(tool) for tool in tools]
    request = ModelRequest(model=_DeferredAgentModel(), tools=list(tools), messages=[], runtime=Runtime(context=context))
    visible = [tool.name for tool in ProjectToolVisibilityMiddleware.visible_request(request).tools]
    return registered, visible, {
        "permission_count": len(context.permissions),
        "registered_tool_count": len(registered),
        "visible_tool_count": len(visible),
    }


def _module_boundaries() -> dict[str, bool]:
    text = (PROJECT_ROOT / ".importlinter").read_text(encoding="utf-8")
    return {
        "import_linter_configured": "[importlinter]" in text,
        "application_layer_contract": (
            "[importlinter:contract:application-layers]" in text
            and "app.graph" not in text
        ),
        "top_level_acyclic_contract": (
            "[importlinter:contract:top-level-acyclic]" in text
        ),
        "middleware_runtime_boundary": (
            "[importlinter:contract:framework-middleware-does-not-depend-on-runtime]"
            in text
        ),
        "http_framework_boundary": (
            "[importlinter:contract:http-framework-only-in-api]" in text
        ),
        "settings_framework_boundary": (
            "[importlinter:contract:pydantic-settings-only-in-config]" in text
        ),
        "zero_import_ignores": "ignore_imports" not in text,
    }


def _source_has_forbidden_parent_symbols() -> list[str]:
    production_files = [
        path
        for path in (PROJECT_ROOT / "app").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in production_files
    )
    return sorted(symbol for symbol in FORBIDDEN_PARENT_SYMBOLS if symbol in combined)


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def inspect_runtime(
    tools: tuple[BaseTool, ...] | None = None,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """构造只含普通 JSON 值的当前 Runtime 架构报告。"""
    resolved_settings = settings or AppSettings(_env_file=None)
    resolved_tools = tools if tools is not None else get_local_tools(resolved_settings)
    governance = ProjectToolGovernanceMiddleware(settings=resolved_settings)
    compiled_agent = build_framework_agent(
        resolved_settings,
        tools=resolved_tools,
        governance=governance,
    )
    graph = compiled_agent.get_graph()
    xray_graph = compiled_agent.get_graph(xray=True)
    nodes = sorted(graph.nodes)
    xray_nodes = sorted(xray_graph.nodes)
    registered, visible, visibility = _tool_report(
        resolved_tools,
        settings=resolved_settings,
    )
    module_boundaries = _module_boundaries()
    forbidden_symbols = _source_has_forbidden_parent_symbols()
    graph_dir = PROJECT_ROOT / "app" / "graph"
    graph_python_files = sorted(path.name for path in graph_dir.glob("*.py"))
    runtime_source = (
        PROJECT_ROOT / "app" / "agent" / "runtime.py"
    ).read_text(encoding="utf-8")
    framework_agent_source = (
        PROJECT_ROOT / "app" / "agent" / "framework_agent.py"
    ).read_text(encoding="utf-8")
    middleware_source = (
        PROJECT_ROOT / "app" / "agent" / "framework_middleware.py"
    ).read_text(encoding="utf-8")
    results_source = (
        PROJECT_ROOT / "app" / "agent" / "results.py"
    ).read_text(encoding="utf-8")
    chat_service_source = (
        PROJECT_ROOT / "app" / "services" / "chat_service.py"
    ).read_text(encoding="utf-8")
    streaming_source = inspect.getsource(stream_agent)
    knowledge_source = (
        PROJECT_ROOT / "app" / "tools" / "knowledge_base.py"
    ).read_text(encoding="utf-8")
    framework_tools_source = (
        PROJECT_ROOT / "app" / "agent" / "framework_tools.py"
    ).read_text(encoding="utf-8")
    agent_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "app" / "agent").glob("*.py"))
    )
    catalog_source = (
        PROJECT_ROOT / "app" / "tools" / "catalog.py"
    ).read_text(encoding="utf-8")
    tool_runtime_source = (
        PROJECT_ROOT / "app" / "tools" / "runtime.py"
    ).read_text(encoding="utf-8")
    tool_execution_source = (
        PROJECT_ROOT / "app" / "tools" / "execution.py"
    ).read_text(encoding="utf-8")
    visibility_source = (
        PROJECT_ROOT / "app" / "agent" / "tool_visibility.py"
    ).read_text(encoding="utf-8")
    mcp_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "app" / "tools" / "mcp").glob("*.py"))
    )
    current_architecture = (
        PROJECT_ROOT / "docs" / "architecture" / "current-system.md"
    ).read_text(encoding="utf-8")
    framework_runtime_adr = (
        PROJECT_ROOT / "docs" / "adr" / "0006-framework-native-agent-runtime.md"
    ).read_text(encoding="utf-8")
    result_extra = AgentRunResult.model_config.get("extra")
    result_fields = list(AgentRunResult.model_fields)
    state_fields = list(FrameworkAgentState.__annotations__)
    status_policy = [
        {
            "result_status": status.value,
            "http_status": policy.http_status,
            "code": policy.code,
            "public_status": policy.public_status,
            "metric_status": policy.metric_status,
        }
        for status, policy in CHAT_STATUS_POLICIES.items()
    ]
    checks = [
        _check(
            "create_agent_is_only_agent_loop",
            EXPECTED_AGENT_NODES <= set(nodes)
            and FORBIDDEN_PARENT_NODES.isdisjoint(nodes),
            "create_agent owns model/tools loop",
        ),
        _check(
            "no_parent_agent_graph",
            not graph_python_files and not forbidden_symbols,
            f"graph_files={graph_python_files}, forbidden={forbidden_symbols}",
        ),
        _check(
            "router_is_plain_python",
            "route_agent_action(" in runtime_source,
            "Agent Runtime calls deterministic router before create_agent",
        ),
        _check(
            "runtime_only_prepares_agent_invocation",
            "def prepare_agent_runtime_invocation(" in runtime_source
            and ".invoke(" not in runtime_source
            and ".astream(" not in runtime_source,
            "runtime prepares input; streaming owns Agent execution",
        ),
        _check(
            "middleware_does_not_depend_on_runtime",
            "app.agent.runtime" not in middleware_source,
            "framework middleware imports guardrails instead of Runtime",
        ),
        _check(
            "native_tool_middleware_hooks",
            "def wrap_tool_call(" not in middleware_source
            and "def awrap_tool_call(" in middleware_source,
            "project governance overrides LangChain native tool hooks",
        ),
        _check(
            "accepted_model_governance_boundary",
            _DeferredAgentModel.__name__ in framework_agent_source
            and _GovernedBoundModel.__name__ in framework_agent_source
            and _MODEL_GOVERNANCE_KEY == "_project_model_governance"
            and "_MODEL_GOVERNANCE_KEY" in framework_agent_source
            and "after_model" not in ProjectToolGovernanceMiddleware.__dict__
            and "aafter_model" in ProjectToolGovernanceMiddleware.__dict__
            and "invoke" not in _GovernedBoundModel.__dict__
            and "wrap_model_call" not in ProjectToolGovernanceMiddleware.__dict__
            and "awrap_model_call"
            not in ProjectToolGovernanceMiddleware.__dict__
            and "KEEP_CURRENT_MODEL_MARKER" in framework_runtime_adr
            and "PHASE6B_SKIPPED" in framework_runtime_adr
            and "KEEP_CURRENT_MODEL_MARKER" in current_architecture,
            "Phase 6B model marker is an accepted architecture boundary",
        ),
        _check(
            "no_tool_governance_artifact_protocol",
            "_TOOL_GOVERNANCE_KEY" not in agent_source
            and "_project_tool_governance" not in agent_source,
            "tool governance updates state through Command instead of artifacts",
        ),
        _check(
            "framework_tools_does_not_simulate_middleware",
            "ToolCallRequest(" not in framework_tools_source
            and "govern_tool_call" not in framework_tools_source
            and "agovern_tool_call" not in framework_tools_source,
            "framework tools do not construct or invoke middleware requests",
        ),
        _check(
            "mcp_tool_has_no_framework_rewrapper",
            "_build_governed_external_tool" not in framework_tools_source,
            "governed MCP BaseTool enters the shared native middleware directly",
        ),
        _check(
            "result_builder_owns_agent_result",
            "def build_framework_agent_result(" in results_source
            and "def build_framework_agent_result(" not in runtime_source,
            "results owns framework output to AgentRunResult normalization",
        ),
        _check(
            "chat_service_consumes_agent_result",
            "validate_agent_result_payload" not in chat_service_source
            and "AgentRunResult.model_validate" not in chat_service_source,
            "ChatService applies HTTP policy without rebuilding AgentResult",
        ),
        _check(
            "streaming_uses_create_agent_directly",
            "invocation.agent.astream(" in streaming_source,
            "stream Runtime directly consumes create_agent.astream",
        ),
        _check(
            "rag_graph_is_tool_owned",
            "run_rag_subgraph(" in knowledge_source,
            "knowledge base tool invokes independent RAG graph",
        ),
        _check(
            "local_tool_execution_entry_is_unique",
            "class ProjectStructuredTool(StructuredTool)" in framework_tools_source
            and "execute_project_tool(" in framework_tools_source
            and "execute_registered_tool" not in framework_tools_source
            and "registry.dispatch(" not in framework_tools_source
            and "def dispatch(" not in catalog_source
            and not (PROJECT_ROOT / "app" / "tools" / "dispatch.py").exists(),
            "ProjectStructuredTool directly enters the thin project adapter",
        ),
        _check(
            "local_handler_is_adapter_owned",
            tool_execution_source.count("spec.handler") == 1
            and "spec.handler" not in tool_runtime_source
            and "spec.handler" not in framework_tools_source
            and "spec.handler" not in visibility_source,
            "only the thin project execution adapter submits ToolSpec.handler",
        ),
        _check(
            "local_visibility_is_model_middleware",
            "request.override(" in visibility_source
            and "def wrap_model_call(" not in visibility_source
            and "def awrap_model_call(" in visibility_source
            and "isinstance(tool, ProjectStructuredTool)" in visibility_source
            and not (PROJECT_ROOT / "app" / "tools" / "providers" / "local.py").exists(),
            "LangChain async model middleware filters local tools",
        ),
        _check(
            "mcp_governance_is_independent",
            "class McpToolCallInterceptor" in mcp_source
            and "def build_governed_mcp_tool(" in mcp_source
            and "from app.tools.runtime" not in mcp_source,
            "MCP keeps its adapter interceptor instead of local ToolRuntime",
        ),
        _check(
            "no_parent_event_projection_cursor",
            "framework_events_applied" not in state_fields,
            "governance events normalize once at Runtime exit",
        ),
        _check(
            "agent_result_contract",
            result_extra == "forbid" and bool(result_fields),
            "AgentRunResult is complete and forbids extras",
        ),
        _check(
            "status_policy_complete",
            len(status_policy) == 6,
            "six runtime result statuses have one policy",
        ),
        _check(
            "current_architecture_documented",
            "系统不存在 Parent Agent Graph" in current_architecture
            and "→ Agent Runtime\n→ create_agent" in current_architecture,
            "current architecture documents direct Agent Runtime",
        ),
        _check(
            "history_archived",
            (PROJECT_ROOT / "docs" / "history" / "agent-evolution-2026.md").is_file(),
            "historical parent graph decisions remain archived",
        ),
        *(
            _check(name, passed, name.replace("_", " "))
            for name, passed in module_boundaries.items()
        ),
        _check(
            "sql_tools_disabled_by_default",
            resolved_settings.sql_tool_enabled is False,
            "SQL tools remain opt-in",
        ),
        _check(
            "mcp_disabled_by_default",
            resolved_settings.mcp_tools_enabled is False,
            "MCP tools remain opt-in",
        ),
        _check(
            "tool_runtime_governance_preserved",
            all(item["risk_level"] == "read_only" for item in registered),
            "registered default tools retain read-only governance metadata",
        ),
    ]
    return {
        "versions": {
            "langchain": _version("langchain"),
            "langgraph": _version("langgraph"),
            "langchain_mcp_adapters": _version("langchain-mcp-adapters"),
        },
        "agent_runtime": {
            "implementation": "langchain.create_agent",
            "nodes": nodes,
            "edges": _graph_edges(graph),
            "xray_nodes": xray_nodes,
            "mermaid": xray_graph.draw_mermaid(),
        },
        "state_schema": {
            "model": "FrameworkAgentState",
            "fields": state_fields,
        },
        "model_governance": {
            "decision": "KEEP_CURRENT_MODEL_MARKER",
            "gate_result": "PHASE6B_SKIPPED",
            "adapter_chain": [
                _DeferredAgentModel.__name__,
                _GovernedBoundModel.__name__,
            ],
            "marker": _MODEL_GOVERNANCE_KEY,
            "state_hooks": ["aafter_model"],
            "model_wrap_hooks_overridden": False,
            "status": "accepted_model_governance_boundary",
        },
        "agent_result_contract": {
            "model": "AgentRunResult",
            "fields": result_fields,
            "extra": result_extra,
        },
        "chat_status_policy": status_policy,
        "registered_tools": registered,
        "default_visible_tools": visible,
        "tool_visibility": visibility,
        "local_tool_execution": {
            "catalog": "get_local_tools",
            "visibility": "ProjectToolVisibilityMiddleware",
            "adapter": "ProjectStructuredTool",
            "input_validation": "LangChain StructuredTool/Pydantic",
            "retry": "LangChain ToolRetryMiddleware",
            "timeout": "shared bounded executor helper",
            "terminal": "ToolSpec.handler",
        },
        "module_boundaries": module_boundaries,
        "mcp_client": {
            "enabled_by_default": False,
            "config_version": 1,
            "transport": "streamable_http",
            "dependency": _version("langchain-mcp-adapters"),
        },
        "checks": checks,
    }


def _text_report(report: dict[str, Any]) -> str:
    lines = [
        "Agent Runtime: langchain.create_agent",
        f"Nodes: {', '.join(report['agent_runtime']['nodes'])}",
        f"Registered tools: {', '.join(item['name'] for item in report['registered_tools'])}",
        f"Default visible tools: {', '.join(report['default_visible_tools'])}",
        "Checks:",
    ]
    lines.extend(
        f"- {'PASS' if item['passed'] else 'FAIL'} {item['name']}: {item['detail']}"
        for item in report["checks"]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("text", "json", "mermaid"),
        default="text",
    )
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inspect_runtime()
    failed = [item for item in report["checks"] if not item["passed"]]
    if args.check:
        if failed:
            for item in failed:
                print(f"FAIL {item['name']}: {item['detail']}", file=sys.stderr)
            return 1
        print("Agent runtime architecture checks passed.")
        return 0
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.format == "mermaid":
        print(report["agent_runtime"]["mermaid"])
    else:
        print(_text_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

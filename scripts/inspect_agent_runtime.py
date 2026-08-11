"""Agent 运行时与架构合同的离线检查程序。

用途：检查 LangGraph 结构、工具可见性、数据合同、依赖版本、模块边界和文件规模预算。
输入：当前源码、Settings 默认合同、工具注册信息和 ``.importlinter`` 配置。
输出：文本、JSON 或 Mermaid 视图；``--check`` 通过退出码表达架构门禁结果。
边界：不会调用模型、执行工具、访问网络或写持久化数据；Mermaid 仅构建而不运行图。
"""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import re
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, get_origin

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_LINTER_CONFIG = PROJECT_ROOT / ".importlinter"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

from app.agent.framework_agent import build_framework_agent  # noqa: E402
from app.agent.framework_tools import build_framework_tools  # noqa: E402
from app.agent.results import AgentRunResult  # noqa: E402
from app.contracts.chat import (  # noqa: E402
    CHAT_STATUS_POLICIES,
    AgentResultStatus,
)
from app.config import AppSettings  # noqa: E402
from app.graph.builder import build_agent_graph  # noqa: E402
from app.graph.state import AgentGraphState  # noqa: E402
from app.services.chat_stream_models import (  # noqa: E402
    CHAT_STREAM_EVENT_NAMES,
)
from app.tools.context import (  # noqa: E402
    ToolInvocationContext,
    build_default_tool_context,
)
from app.tools.contracts import ToolSpec  # noqa: E402
from app.tools.providers import (  # noqa: E402
    LocalToolProvider,
    ToolProvider,
)
from app.tools.providers.base import is_tool_visible  # noqa: E402
from app.tools.registry import ToolRegistry  # noqa: E402
from app.tools.registry import register_default_tools  # noqa: E402


PACKAGES = (
    "langchain",
    "langchain-core",
    "langchain-mcp-adapters",
    "langchain-openai",
    "langgraph",
    "pydantic",
    "pydantic-settings",
    "pypdf",
    "requests-hardened",
    "SQLAlchemy",
    "sqlglot",
)
SQL_TOOL_NAMES = {
    "list_sql_tables",
    "describe_sql_table",
    "query_sql_readonly",
}
EXPECTED_PARENT_NODES = {
    "route", "initialize", "framework_agent", "adapt_framework",
    "rag_subgraph", "compose", "finalize",
}
EXPECTED_PARENT_EDGES = {
    ("__start__", "route", False),
    ("route", "initialize", False),
    ("initialize", "framework_agent", True),
    ("initialize", "compose", True),
    ("framework_agent", "adapt_framework", False),
    ("adapt_framework", "rag_subgraph", True),
    ("adapt_framework", "compose", True),
    ("rag_subgraph", "finalize", False),
    ("compose", "finalize", False),
    ("finalize", "__end__", False),
}
FORBIDDEN_PARENT_NODES = {
    "plan",
    "execute",
    "agent_model",
    "governed_tools",
    "limit_handler",
}
SOURCE_FILE_LIMITS = {
    "app/graph/nodes.py": 350,
    "app/agent/framework_agent.py": 300,
    "app/agent/framework_middleware.py": 350,
    "app/agent/framework_tools.py": 300,
    "app/agent/tool_events.py": 250,
    "app/agent/guardrails.py": 150,
}
BUSINESS_STATE_FIELDS = {
    "observations",
    "final_answer",
    "references",
    "status",
    "reason",
    "decompose_llm_used",
    "answer_llm_used",
    "llm_call_count",
    "tool_trace",
}
NESTED_STATE_FIELD = "agent_" + "state"
AGENT_RESULT_FIELDS = {
    "status",
    "answer",
    "references",
    "retrieval_confidence",
    "answer_confidence",
    "need_clarification",
    "clarification_question",
    "reason",
    "action",
    "tool_result_summary",
    "tool_trace",
    "router_llm_used",
    "decompose_llm_used",
    "answer_llm_used",
    "llm_call_count",
    "timings",
}
IMPORT_LINTER_CONTRACT_SECTIONS = {
    "application_layer_contract": (
        "importlinter:contract:application-layers",
        "layers",
    ),
    "top_level_acyclic_contract": (
        "importlinter:contract:top-level-acyclic",
        "acyclic_siblings",
    ),
    "http_framework_boundary": (
        "importlinter:contract:http-framework-only-in-api",
        "forbidden",
    ),
    "settings_framework_boundary": (
        "importlinter:contract:pydantic-settings-only-in-config",
        "forbidden",
    ),
}


def _line_count(relative_path: str) -> int:
    """忽略末尾空行与换行符，统计逻辑行数。"""
    lines = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return len(lines)


def _graph_data(drawable_graph) -> dict[str, Any]:
    """序列化 LangGraph 可绘制图所公开的稳定字段。"""
    nodes = sorted(name for name in drawable_graph.nodes if not name.startswith("__"))
    edges = sorted(
        (
            {
                "source": edge.source,
                "target": edge.target,
                "conditional": bool(edge.conditional),
            }
            for edge in drawable_graph.edges
        ),
        key=lambda item: (item["source"], item["target"], item["conditional"]),
    )
    return {"nodes": nodes, "edges": edges}


def _tool_data(
    provider: ToolProvider,
    settings: AppSettings,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[str],
    dict[str, int],
    list[str],
]:
    """读取已注册工具契约和模型可见模式，但不执行分发。"""
    context = build_default_tool_context(
        source="agent_runtime_inspection",
        settings=settings,
    )
    registry = getattr(provider, "registry", None)
    if not isinstance(registry, ToolRegistry):
        return [], [], ["provider does not expose a ToolRegistry"], {}, []
    visible_specs = list(provider.list_specs(context=context))
    visible_names = [spec.name for spec in visible_specs if isinstance(spec, ToolSpec)]
    registered_specs = [registry.get(name) for name in registry.names()]
    all_permissions = tuple(
        sorted(
            {
                permission
                for spec in registered_specs
                if isinstance(spec, ToolSpec)
                for permission in spec.required_permissions
            }
        )
    )
    inspection_context = ToolInvocationContext(
        principal_id="agent_runtime_inspection",
        permissions=all_permissions,
        source="agent_runtime_inspection",
    )
    all_tools_by_name = {
        tool.name: tool
        for tool in build_framework_tools(
            LocalToolProvider(registry),
            context=inspection_context,
        )
    }
    errors: list[str] = []
    visibility_errors: list[str] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    registered_names = set(registry.names())
    if not set(visible_names) <= registered_names:
        visibility_errors.append("default visible tools are not a registry subset")
    for spec in visible_specs:
        if not isinstance(spec, ToolSpec):
            visibility_errors.append(
                f"provider returned non-ToolSpec: {type(spec).__name__}"
            )
            continue
        if not is_tool_visible(spec, context=context):
            visibility_errors.append(
                f"visible tool lacks a required permission: {spec.name}"
            )
    for spec in registered_specs:
        if not isinstance(spec, ToolSpec):
            errors.append(f"registry returned non-ToolSpec: {type(spec).__name__}")
            continue
        if spec.name in seen:
            errors.append(f"duplicate tool name: {spec.name}")
        seen.add(spec.name)
        if not spec.description or not spec.version or spec.args_model is None:
            errors.append(f"incomplete ToolSpec contract: {spec.name}")
        if registry.get(spec.name) is not spec:
            errors.append(f"registry cannot resolve ToolSpec: {spec.name}")
        tool = all_tools_by_name.get(spec.name)
        if tool is None:
            errors.append(f"StructuredTool is missing: {spec.name}")
            continue
        try:
            schema = tool.tool_call_schema.model_json_schema()
        except Exception as exc:
            errors.append(f"invalid args schema for {spec.name}: {exc}")
            schema = {}
        description = " ".join(spec.description.split())
        rows.append(
            {
                "name": spec.name,
                "version": spec.version,
                "description": description if len(description) <= 240 else description[:237] + "...",
                "args_schema": schema,
                "risk_level": spec.risk_level.value,
                "required_permissions": list(spec.required_permissions),
                "requires_approval": spec.requires_approval,
                "output_model": (
                    spec.output_model.__name__
                    if spec.output_model is not None
                    else None
                ),
            }
        )
    visibility = {
        "permission_count": len(context.permissions),
        "registered_tool_count": len(rows),
        "visible_tool_count": len(visible_names),
    }
    return (
        rows,
        visible_names,
        errors,
        visibility,
        visibility_errors,
    )


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    """把一项架构断言规范化为统一检查结果。"""
    return {"name": name, "passed": passed, "detail": detail}


def _inspect_import_linter_config() -> dict[str, bool]:
    """读取公开合同配置；实际 import graph 仍由 lint-imports 验证。"""
    parser = configparser.ConfigParser()
    configured = IMPORT_LINTER_CONFIG.is_file()
    if configured:
        parser.read(IMPORT_LINTER_CONFIG, encoding="utf-8")

    contract_status = {
        report_name: (
            parser.has_section(section)
            and parser.get(section, "type", fallback="") == contract_type
        )
        for report_name, (
            section,
            contract_type,
        ) in IMPORT_LINTER_CONTRACT_SECTIONS.items()
    }
    zero_import_ignores = all(
        option != "ignore_imports"
        for section in parser.sections()
        for option in parser.options(section)
    )
    return {
        "import_linter_configured": (
            configured
            and parser.has_section("importlinter")
            and parser.get(
                "importlinter",
                "root_package",
                fallback="",
            )
            == "app"
        ),
        **contract_status,
        "zero_import_ignores": zero_import_ignores,
    }


def inspect_runtime(
    provider: ToolProvider | None = None,
    *,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """返回当前运行时的确定性、可安全编码为 JSON 的快照。"""
    resolved_settings = settings or AppSettings(_env_file=None)
    versions: dict[str, str] = {}
    missing_packages: list[str] = []
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = ""
            missing_packages.append(package)

    parent = build_agent_graph(resolved_settings).compile()
    parent_graph = parent.get_graph()
    parent_xray = parent.get_graph(xray=True)
    framework_graph = build_framework_agent(resolved_settings).get_graph()
    parent_data = _graph_data(parent_graph)
    framework_data = _graph_data(framework_graph)
    xray_nodes = sorted(parent_xray.nodes)
    enforce_default_registry = provider is None
    if provider is None:
        registry = ToolRegistry()
        register_default_tools(registry, settings=resolved_settings)
        resolved_provider = LocalToolProvider(registry)
    else:
        resolved_provider = provider
    (
        registered_tools,
        default_visible_tools,
        tool_errors,
        tool_visibility,
        visibility_errors,
    ) = _tool_data(
        resolved_provider,
        resolved_settings,
    )
    source_files = {
        path: {"lines": _line_count(path), "limit": limit}
        for path, limit in SOURCE_FILE_LIMITS.items()
    }

    parent_nodes = set(parent_data["nodes"])
    parent_edges = {
        (edge["source"], edge["target"], edge["conditional"])
        for edge in parent_data["edges"]
    }
    framework_nodes = set(framework_data["nodes"])
    missing_parent = sorted(EXPECTED_PARENT_NODES - parent_nodes)
    missing_parent_edges = sorted(EXPECTED_PARENT_EDGES - parent_edges)
    unexpected_parent_edges = sorted(parent_edges - EXPECTED_PARENT_EDGES)
    forbidden_parent = sorted(FORBIDDEN_PARENT_NODES & parent_nodes)
    missing_framework = sorted({"model", "tools"} - framework_nodes)
    missing_limits = [
        fragment
        for fragment in ("ModelCallLimitMiddleware", "ToolCallLimitMiddleware")
        if not any(fragment in node for node in framework_nodes)
    ]
    xray_has_agent = any(node.endswith(":model") for node in xray_nodes) and any(
        node.endswith(":tools") for node in xray_nodes
    )
    over_budget = [
        f"{path}: {item['lines']} > {item['limit']}"
        for path, item in source_files.items()
        if item["lines"] > item["limit"]
    ]
    state_fields = set(AgentGraphState.__annotations__)
    missing_business_fields = sorted(BUSINESS_STATE_FIELDS - state_fields)
    nested_state_fields = sorted(
        {NESTED_STATE_FIELD, "rewritten_question"} & state_fields
    )
    result_fields = set(AgentRunResult.model_fields)
    result_contract_ok = (
        issubclass(AgentRunResult, BaseModel)
        and result_fields == AGENT_RESULT_FIELDS
        and AgentRunResult.model_config.get("extra") == "forbid"
    )
    policy_statuses = set(CHAT_STATUS_POLICIES)
    policy_values = {
        policy.result_status
        for policy in CHAT_STATUS_POLICIES.values()
    }
    status_policy_complete = (
        policy_statuses == set(AgentResultStatus)
        and policy_values == set(AgentResultStatus)
    )
    service_source = (
        PROJECT_ROOT / "app/services/chat_service.py"
    ).read_text(encoding="utf-8")
    telemetry_source = (
        PROJECT_ROOT / "app/services/chat_telemetry.py"
    ).read_text(encoding="utf-8")
    chat_router_source = (
        PROJECT_ROOT / "app/api/chat.py"
    ).read_text(encoding="utf-8")
    graph_runtime_source = (
        PROJECT_ROOT / "app/graph/runtime.py"
    ).read_text(encoding="utf-8")
    graph_streaming_source = (
        PROJECT_ROOT / "app/graph/streaming.py"
    ).read_text(encoding="utf-8")
    application_source = (
        PROJECT_ROOT / "app/api/application.py"
    ).read_text(encoding="utf-8")
    main_source = (
        PROJECT_ROOT / "app/main.py"
    ).read_text(encoding="utf-8")
    metrics_source = (
        PROJECT_ROOT / "app/observability/metrics.py"
    ).read_text(encoding="utf-8")
    response_source = (
        PROJECT_ROOT / "app/api/responses.py"
    ).read_text(encoding="utf-8")
    config_source = (
        PROJECT_ROOT / "app/config.py"
    ).read_text(encoding="utf-8")
    framework_agent_source = (
        PROJECT_ROOT / "app/agent/framework_agent.py"
    ).read_text(encoding="utf-8")
    framework_tools_source = (
        PROJECT_ROOT / "app/agent/framework_tools.py"
    ).read_text(encoding="utf-8")
    mcp_bridge_source = (
        PROJECT_ROOT / "app/tools/mcp/bridge.py"
    ).read_text(encoding="utf-8")
    mcp_client_source = (
        PROJECT_ROOT / "app/tools/mcp/client.py"
    ).read_text(encoding="utf-8")
    mcp_config_source = (
        PROJECT_ROOT / "app/tools/mcp/config.py"
    ).read_text(encoding="utf-8")
    mcp_discovery_source = (
        PROJECT_ROOT / "app/tools/mcp/discovery.py"
    ).read_text(encoding="utf-8")
    mcp_provider_source = (
        PROJECT_ROOT / "app/tools/mcp/provider.py"
    ).read_text(encoding="utf-8")
    mcp_serialization_source = (
        PROJECT_ROOT / "app/tools/mcp/serialization.py"
    ).read_text(encoding="utf-8")
    immutable_settings = AppSettings.model_config.get("frozen") is True
    no_static_settings_class = re.search(
        r"^class\s+Settings\b",
        config_source,
        flags=re.MULTILINE,
    ) is None
    legacy_parser_names = tuple(
        f"env_{suffix}" for suffix in ("bool", "int", "float", "csv")
    )
    no_manual_env_parsers = not any(
        f"def {name}(" in config_source
        for name in legacy_parser_names
    )
    dotenv_loader_name = "_".join(("load", "dotenv"))
    no_import_time_dotenv = dotenv_loader_name not in config_source
    lazy_settings = not re.search(
        r"^(?:settings\s*=\s*get_settings\(\)|def\s+\w+\([^)]*get_settings\(\))",
        config_source,
        flags=re.MULTILINE,
    )
    pydantic_settings_model = issubclass(AppSettings, BaseSettings)
    single_status_policy = (
        status_policy_complete
        and "get_chat_status_policy" in service_source
        and "get_chat_status_policy" not in metrics_source
        and "get_chat_status_policy" not in response_source
    )
    result_annotation = AgentGraphState.__annotations__.get("result")
    result_is_checkpoint_safe = get_origin(result_annotation) is dict
    response_builder_name = "ChatResponse" + "Builder"
    api_response_adapter_ok = (
        "def build_chat_response_from_outcome(" in response_source
        and response_builder_name not in response_source
        and not any(
            fragment in response_source
            for fragment in (
                "app.agent.runtime",
                "app.graph",
            )
        )
    )
    application_factory_present = (
        "def create_app(" in application_source
        and "from app.api.application import create_app" in main_source
        and "app = create_app()" in main_source
    )
    chat_router_isolated = (
        "def create_chat_router(" in chat_router_source
        and not any(
            fragment in chat_router_source
            for fragment in (
                "FastAPI(",
                "lifespan",
                "create_diagnostics_router",
                "initialize_rag",
                "ToolRuntime",
            )
        )
    )
    chat_service_orchestration_boundary = (
        "class ChatService:" in service_source
        and "def handle(" in service_source
        and "app.services.chat_models" in service_source
        and "app.services.chat_telemetry" in service_source
        and "_TIMING_METRIC_KEYS" not in service_source
        and "rag_ready_check" + "_cost_ms" not in service_source
    )
    chat_telemetry_extracted = all(
        fragment in telemetry_source
        for fragment in (
            "def build_initial_chat_metrics(",
            "def build_initial_bad_case(",
            "def format_tool_call_chain(",
            "def apply_result_metrics(",
            "def finalize_chat_telemetry(",
        )
    )
    chat_router_tree = ast.parse(chat_router_source)
    graph_streaming_tree = ast.parse(graph_streaming_source)
    route_paths = {
        decorator.args[0].value
        for node in ast.walk(chat_router_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "post"
        and decorator.args
        and isinstance(decorator.args[0], ast.Constant)
        and isinstance(decorator.args[0].value, str)
    }
    chat_service_calls = {
        node.func.attr
        for node in ast.walk(chat_router_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "chat_service"
    }
    graph_stream_functions = {
        node.name
        for node in ast.walk(graph_streaming_tree)
        if isinstance(node, ast.AsyncFunctionDef)
    }
    graph_stream_calls = {
        node.func.attr
        for node in ast.walk(graph_streaming_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    chat_stream_route_present = "/chat/stream" in route_paths
    chat_stream_uses_same_service = {
        "handle",
        "stream",
    } <= chat_service_calls
    graph_streaming_runtime_present = (
        "stream_agent_langgraph" in graph_stream_functions
        and "astream" in graph_stream_calls
        and "prepare_async_agent_runtime_invocation"
        in graph_streaming_source
        and "prepare_async_agent_runtime_invocation" in graph_runtime_source
        and "prepare_agent_runtime_invocation" in graph_runtime_source
        and "_prepare_agent_runtime_fields" in graph_runtime_source
    )
    stream_event_contract_present = CHAT_STREAM_EVENT_NAMES == {
        "start",
        "progress",
        "tool",
        "answer_delta",
        "final",
        "error",
        "done",
    }
    sync_chat_route_preserved = (
        "/chat" in route_paths
        and "handle" in chat_service_calls
        and "build_chat_response_from_outcome" in chat_router_source
    )
    current_architecture_documented = (
        PROJECT_ROOT / "docs/architecture/current-system.md"
    ).is_file()
    history_archived = (
        not (
            PROJECT_ROOT
            / "docs"
            / ("agent-evolution-" + "baseline.md")
        ).exists()
        and (
            PROJECT_ROOT / "docs/history/agent-evolution-2026.md"
        ).is_file()
    )
    web_spec = next(
        (
            item
            for item in registered_tools
            if item.get("name") == "web_search"
        ),
        None,
    )
    web_schema_fields = set(
        (web_spec or {}).get("args_schema", {}).get("properties", {})
    )
    web_contract_ok = bool(
        web_spec and web_spec.get("required_permissions") == ["web:search"]
    )
    default_has_web_permission = (
        "web:search"
        in build_default_tool_context(
            source="agent_runtime_inspection",
            settings=resolved_settings,
        ).permissions
    )
    web_default_visibility_ok = (
        ("web_search" in default_visible_tools) == default_has_web_permission
    )
    web_authorized_visibility_ok = True
    fetch_spec = next(
        (
            item
            for item in registered_tools
            if item.get("name") == "web_fetch"
        ),
        None,
    )
    fetch_schema_fields = set(
        (fetch_spec or {}).get("args_schema", {}).get("properties", {})
    )
    fetch_contract_ok = bool(
        fetch_spec
        and fetch_spec.get("risk_level") == "read_only"
        and fetch_spec.get("required_permissions") == ["web:fetch"]
        and fetch_spec.get("output_model") == "WebFetchOutput"
    )
    forbidden_fetch_fields = {
        "headers",
        "cookies",
        "proxy",
        "timeout",
        "method",
        "body",
        "runtime",
    }
    fetch_schema_ok = fetch_schema_fields == {"url", "max_chars"} and not (
        forbidden_fetch_fields & fetch_schema_fields
    )
    default_has_fetch_permission = (
        "web:fetch"
        in build_default_tool_context(
            source="agent_runtime_inspection",
            settings=resolved_settings,
        ).permissions
    )
    fetch_default_visibility_ok = (
        ("web_fetch" in default_visible_tools) == default_has_fetch_permission
    )
    fetch_authorized_visibility_ok = True
    pdf_spec = next(
        (
            item
            for item in registered_tools
            if item.get("name") == "web_pdf_fetch"
        ),
        None,
    )
    pdf_schema_fields = set(
        (pdf_spec or {}).get("args_schema", {}).get("properties", {})
    )
    pdf_contract_ok = bool(
        pdf_spec
        and pdf_spec.get("risk_level") == "read_only"
        and pdf_spec.get("required_permissions") == ["web:fetch"]
        and pdf_spec.get("output_model") == "WebPdfFetchOutput"
    )
    forbidden_pdf_fields = {
        "proxy",
        "headers",
        "cookies",
        "timeout",
        "password",
        "strict",
        "runtime",
    }
    pdf_schema_ok = pdf_schema_fields == {
        "url",
        "start_page",
        "max_pages",
        "max_chars",
    } and not (forbidden_pdf_fields & pdf_schema_fields)
    pdf_default_visibility_ok = (
        ("web_pdf_fetch" in default_visible_tools)
        == default_has_fetch_permission
    )
    pdf_authorized_visibility_ok = True
    module_boundaries = _inspect_import_linter_config()
    expected_tool_count = 11 if resolved_settings.sql_tool_enabled else 8
    registered_tool_count_ok = len(registered_tools) == expected_tool_count
    mcp_disabled_by_default = (
        AppSettings.model_fields["mcp_tools_enabled"].default is False
        and AppSettings.model_fields["mcp_servers_config_path"].default is None
    )
    mcp_uses_langchain_adapter = (
        "langchain-mcp-adapters" not in missing_packages
        and "from langchain_mcp_adapters.client import MultiServerMCPClient"
        in mcp_client_source
        and "MultiServerMCPClient(" in mcp_client_source
    )
    mcp_http_only_v1 = (
        'version: Literal[1]' in mcp_config_source
        and 'transport: Literal["http"]' in mcp_config_source
        and '"transport": "http"' in mcp_config_source
    )
    mcp_tools_require_server_permission = (
        'return f"mcp:{self.server_alias}"' in mcp_provider_source
        and "has_permission(context.permissions, entry.required_permission)"
        in mcp_provider_source
        and 'f"mcp:{request.server_name}"' in mcp_bridge_source
    )
    mcp_tools_have_local_allowlist = (
        "allowed_tools: tuple[str, ...]" in mcp_config_source
        and "for remote_name in server.allowed_tools:" in mcp_discovery_source
        and "by_exposed_name.get(exposed_name)" in mcp_discovery_source
    )
    mcp_tool_names_are_prefixed = (
        "tool_name_prefix=True" in mcp_client_source
        and 'exposed_name = f"{alias}_{remote_name}"' in mcp_discovery_source
    )
    mcp_outputs_are_untrusted = (
        '"content_is_untrusted": True' in mcp_serialization_source
        and "redact_sensitive_data(" in mcp_serialization_source
        and "mcp_max_structured_chars" in mcp_serialization_source
    )
    mcp_local_registry_path_preserved = (
        "tools = build_framework_tools(provider, context=context)"
        in framework_tools_source
        and "registry.dispatch(spec.name, tool_args, context=context)"
        in framework_tools_source
    )
    mcp_governance_path_preserved = (
        "RegistryToolsMiddleware()," in framework_agent_source
        and "ProjectToolGovernanceMiddleware(settings=resolved),"
        in framework_agent_source
        and "build_agent_framework_tools(" in framework_tools_source
        and "parse_tool_result(" in framework_tools_source
    )

    disabled_sql_settings = resolved_settings.model_copy(
        update={"sql_tool_enabled": False}
    )
    disabled_sql_registry = ToolRegistry()
    register_default_tools(
        disabled_sql_registry,
        settings=disabled_sql_settings,
    )
    enabled_sql_settings = resolved_settings.model_copy(
        update={
            "sql_tool_enabled": True,
            "sql_database_url": SecretStr(
                "sqlite:////tmp/agent-runtime-inspection-business.sqlite3"
            ),
            "sql_allowed_tables": ("inspection_table",),
        }
    )
    enabled_sql_registry = ToolRegistry()
    register_default_tools(enabled_sql_registry, settings=enabled_sql_settings)
    enabled_sql_specs = [
        enabled_sql_registry.get(name) for name in sorted(SQL_TOOL_NAMES)
    ]
    sql_tools_disabled_by_default = (
        AppSettings.model_fields["sql_tool_enabled"].default is False
        and not (SQL_TOOL_NAMES & set(disabled_sql_registry.names()))
    )
    sql_tools_require_database_read = all(
        isinstance(spec, ToolSpec)
        and spec.required_permissions == ("database:read",)
        for spec in enabled_sql_specs
    )
    sql_tools_are_read_only = all(
        isinstance(spec, ToolSpec)
        and spec.risk_level.value == "read_only"
        and spec.max_retries == 0
        and spec.idempotent
        for spec in enabled_sql_specs
    )
    sql_tools_have_pydantic_contracts = all(
        isinstance(spec, ToolSpec)
        and isinstance(spec.args_model, type)
        and issubclass(spec.args_model, BaseModel)
        and isinstance(spec.output_model, type)
        and issubclass(spec.output_model, BaseModel)
        for spec in enabled_sql_specs
    )
    sql_tools_do_not_require_approval = all(
        isinstance(spec, ToolSpec) and not spec.requires_approval
        for spec in enabled_sql_specs
    )
    sql_context = ToolInvocationContext(
        principal_id="agent_runtime_inspection",
        permissions=("database:read",),
        source="agent_runtime_inspection",
    )
    sql_framework_names = {
        tool.name
        for tool in build_framework_tools(
            LocalToolProvider(enabled_sql_registry),
            context=sql_context,
        )
    }
    sql_tools_use_registry_runtime = SQL_TOOL_NAMES <= sql_framework_names
    sql_tools_share_chat_paths = not any(
        name in source
        for name in SQL_TOOL_NAMES
        for source in (
            chat_router_source,
            service_source,
            graph_streaming_source,
        )
    )
    if enforce_default_registry:
        default_context = build_default_tool_context(
            source="agent_runtime_inspection",
            settings=resolved_settings,
        )
        authorized_context = ToolInvocationContext(
            principal_id=default_context.principal_id,
            principal_type=default_context.principal_type,
            permissions=tuple((*default_context.permissions, "web:search")),
            source="agent_runtime_inspection",
        )
        authorized_names = {
            spec.name
            for spec in resolved_provider.list_specs(context=authorized_context)
        }
        web_authorized_visibility_ok = "web_search" in authorized_names
        fetch_context = ToolInvocationContext(
            principal_id=default_context.principal_id,
            principal_type=default_context.principal_type,
            permissions=("knowledge_base:read", "web:fetch"),
            source="agent_runtime_inspection",
        )
        fetch_authorized_names = {
            spec.name
            for spec in resolved_provider.list_specs(context=fetch_context)
        }
        fetch_authorized_visibility_ok = {
            "web_fetch",
            "web_pdf_fetch",
        } <= fetch_authorized_names
        pdf_authorized_visibility_ok = (
            "web_pdf_fetch" in fetch_authorized_names
        )
    checks = [
        _check("dependency_versions", not missing_packages, ", ".join(missing_packages)),
        _check(
            "pydantic_settings_model",
            pydantic_settings_model,
            "AppSettings must inherit pydantic-settings BaseSettings",
        ),
        _check(
            "lazy_settings",
            bool(lazy_settings),
            "config import must not instantiate AppSettings",
        ),
        _check(
            "immutable_settings",
            immutable_settings,
            "AppSettings must be frozen",
        ),
        _check(
            "no_static_settings_class",
            no_static_settings_class,
            "legacy static Settings class must be absent",
        ),
        _check(
            "no_manual_env_parsers",
            no_manual_env_parsers,
            "manual env parser helpers must be absent",
        ),
        _check(
            "no_import_time_dotenv",
            no_import_time_dotenv,
            "dotenv loader must be absent",
        ),
        *[
            _check(
                name,
                passed,
                (
                    ".importlinter must declare the expected public contract "
                    "without ignore_imports"
                ),
            )
            for name, passed in module_boundaries.items()
        ],
        _check(
            "single_runtime_state",
            not nested_state_fields and not missing_business_fields,
            (
                f"nested={nested_state_fields}; "
                f"missing={missing_business_fields}"
            ),
        ),
        _check(
            "no_nested_" + NESTED_STATE_FIELD,
            NESTED_STATE_FIELD not in state_fields,
            "AgentGraphState must not contain a nested runtime state",
        ),
        _check(
            "state_has_business_fields",
            not missing_business_fields,
            ", ".join(missing_business_fields),
        ),
        _check(
            "agent_result_contract",
            result_contract_ok,
            (
                f"missing={sorted(AGENT_RESULT_FIELDS - result_fields)}; "
                f"extra={sorted(result_fields - AGENT_RESULT_FIELDS)}"
            ),
        ),
        _check(
            "status_policy_complete",
            status_policy_complete,
            (
                f"keys={sorted(str(item) for item in policy_statuses)}; "
                f"values={sorted(str(item) for item in policy_values)}"
            ),
        ),
        _check(
            "single_status_policy",
            single_status_policy,
            "ChatService must be the only high-level policy consumer",
        ),
        _check(
            "result_is_checkpoint_safe",
            result_is_checkpoint_safe,
            "AgentGraphState.result must remain a plain dict annotation",
        ),
        _check(
            "api_response_adapter_boundary",
            api_response_adapter_ok,
            "API response adapter must have one concrete outcome path",
        ),
        _check(
            "application_factory_present",
            application_factory_present,
            "app.main must expose an app created by app.api.application",
        ),
        _check(
            "chat_router_isolated",
            chat_router_isolated,
            "chat router must not assemble FastAPI or runtime resources",
        ),
        _check(
            "chat_service_orchestration_boundary",
            chat_service_orchestration_boundary,
            "ChatService must delegate models and telemetry responsibilities",
        ),
        _check(
            "chat_telemetry_extracted",
            chat_telemetry_extracted,
            "chat telemetry helpers must live in the services telemetry module",
        ),
        _check(
            "chat_stream_route_present",
            chat_stream_route_present,
            "chat router must register POST /chat/stream",
        ),
        _check(
            "chat_stream_uses_same_service",
            chat_stream_uses_same_service,
            "sync and streaming routes must close over the same ChatService",
        ),
        _check(
            "graph_streaming_runtime_present",
            graph_streaming_runtime_present,
            "streaming Runtime must use graph.astream, async Graph prep, "
            "and shared invocation fields",
        ),
        _check(
            "stream_event_contract_present",
            stream_event_contract_present,
            "stream event contract must contain the seven public event names",
        ),
        _check(
            "sync_chat_route_preserved",
            sync_chat_route_preserved,
            "POST /chat and its response adapter must remain present",
        ),
        _check(
            "current_architecture_documented",
            current_architecture_documented,
            "current-system architecture document must exist",
        ),
        _check(
            "history_archived",
            history_archived,
            "agent evolution history must be archived under docs/history",
        ),
        _check("parent_required_nodes", not missing_parent, ", ".join(missing_parent)),
        _check("parent_forbidden_nodes", not forbidden_parent, ", ".join(forbidden_parent)),
        _check(
            "parent_graph_edges",
            not missing_parent_edges and not unexpected_parent_edges,
            (
                f"missing={missing_parent_edges}; "
                f"unexpected={unexpected_parent_edges}"
            ),
        ),
        _check("framework_core_nodes", not missing_framework, ", ".join(missing_framework)),
        _check("framework_call_limits", not missing_limits, ", ".join(missing_limits)),
        _check("parent_xray_subgraph", xray_has_agent, "framework model/tools visible"),
        _check(
            "registered_tool_contracts",
            not tool_errors and bool(registered_tools),
            "; ".join(tool_errors),
        ),
        _check(
            "registered_tool_count",
            not enforce_default_registry or registered_tool_count_ok,
            "registry count must be eight by default and eleven when SQL is enabled",
        ),
        _check(
            "default_tool_visibility",
            not visibility_errors,
            "; ".join(visibility_errors),
        ),
        _check(
            "web_search_contract",
            not enforce_default_registry or web_contract_ok,
            "web_search must require web:search",
        ),
        _check(
            "web_search_model_schema",
            not enforce_default_registry
            or not ({"proxy", "backend", "timeout", "runtime"} & web_schema_fields),
            "web_search model schema exposes runtime configuration",
        ),
        _check(
            "web_search_default_visibility",
            not enforce_default_registry or web_default_visibility_ok,
            "web_search visibility does not match default permissions",
        ),
        _check(
            "web_search_authorized_visibility",
            not enforce_default_registry or web_authorized_visibility_ok,
            "web_search is hidden despite web:search permission",
        ),
        _check(
            "web_fetch_contract",
            not enforce_default_registry or fetch_contract_ok,
            "web_fetch must be read-only, require web:fetch, and validate output",
        ),
        _check(
            "web_fetch_model_schema",
            not enforce_default_registry or fetch_schema_ok,
            "web_fetch model schema must expose only url and max_chars",
        ),
        _check(
            "web_fetch_default_visibility",
            not enforce_default_registry or fetch_default_visibility_ok,
            "web_fetch visibility does not match default permissions",
        ),
        _check(
            "web_fetch_authorized_visibility",
            not enforce_default_registry or fetch_authorized_visibility_ok,
            "web_fetch or web_pdf_fetch is hidden despite web:fetch permission",
        ),
        _check(
            "web_pdf_fetch_contract",
            not enforce_default_registry or pdf_contract_ok,
            "web_pdf_fetch must be read-only, require web:fetch, and validate output",
        ),
        _check(
            "web_pdf_fetch_model_schema",
            not enforce_default_registry or pdf_schema_ok,
            "web_pdf_fetch model schema exposes runtime configuration",
        ),
        _check(
            "web_pdf_fetch_default_visibility",
            not enforce_default_registry or pdf_default_visibility_ok,
            "web_pdf_fetch visibility does not match default permissions",
        ),
        _check(
            "web_pdf_fetch_authorized_visibility",
            not enforce_default_registry or pdf_authorized_visibility_ok,
            "web_pdf_fetch is hidden despite web:fetch permission",
        ),
        _check(
            "sql_tools_disabled_by_default",
            sql_tools_disabled_by_default,
            "SQL tools must be absent from the default disabled registry",
        ),
        _check(
            "sql_tools_require_database_read",
            sql_tools_require_database_read,
            "all SQL tools must require exactly database:read",
        ),
        _check(
            "sql_tools_are_read_only",
            sql_tools_are_read_only,
            "SQL tools must be read-only, idempotent, and non-retrying",
        ),
        _check(
            "sql_tools_use_registry_runtime",
            sql_tools_use_registry_runtime,
            "enabled SQL tools must be projected from ToolRegistry",
        ),
        _check(
            "sql_tools_have_pydantic_contracts",
            sql_tools_have_pydantic_contracts,
            "SQL tools must declare Pydantic input and output models",
        ),
        _check(
            "sql_tools_do_not_require_approval",
            sql_tools_do_not_require_approval,
            "read-only SQL tools must not require approval",
        ),
        _check(
            "sql_tools_share_chat_paths",
            sql_tools_share_chat_paths,
            "chat and streaming modules must not contain SQL-specific branches",
        ),
        _check(
            "mcp_disabled_by_default",
            mcp_disabled_by_default,
            "MCP tools and config path must be disabled by default",
        ),
        _check(
            "mcp_uses_langchain_adapter",
            mcp_uses_langchain_adapter,
            "MCP Client must use the official LangChain adapter",
        ),
        _check(
            "mcp_http_only_v1",
            mcp_http_only_v1,
            "MCP config v1 must expose only Streamable HTTP transport",
        ),
        _check(
            "mcp_tools_require_server_permission",
            mcp_tools_require_server_permission,
            "every MCP tool must require its mcp:<server_alias> permission",
        ),
        _check(
            "mcp_tools_have_local_allowlist",
            mcp_tools_have_local_allowlist,
            "MCP discovery must intersect with a local exact allowlist",
        ),
        _check(
            "mcp_tool_names_are_prefixed",
            mcp_tool_names_are_prefixed,
            "MCP tools must use server-prefixed exposed names",
        ),
        _check(
            "mcp_outputs_are_untrusted",
            mcp_outputs_are_untrusted,
            "MCP output must be bounded, redacted, and marked untrusted",
        ),
        _check(
            "mcp_local_registry_path_preserved",
            mcp_local_registry_path_preserved,
            "local tools must still use ToolRegistry dispatch",
        ),
        _check(
            "mcp_governance_path_preserved",
            mcp_governance_path_preserved,
            "MCP tools must share existing agent governance and result parsing",
        ),
        _check("source_file_budgets", not over_budget, "; ".join(over_budget)),
    ]
    return {
        "versions": versions,
        "settings_contract": {
            "model": AppSettings.__name__,
            "base": BaseSettings.__name__,
            "frozen": immutable_settings,
            "env_file_disabled_for_inspection": True,
        },
        "module_boundaries": module_boundaries,
        "state_schema": {"fields": sorted(state_fields)},
        "agent_result_contract": {
            "model": AgentRunResult.__name__,
            "fields": sorted(result_fields),
            "extra": AgentRunResult.model_config.get("extra"),
        },
        "chat_status_policy": [
            {
                "result_status": status.value,
                "http_status": CHAT_STATUS_POLICIES[status].http_status,
                "code": CHAT_STATUS_POLICIES[status].code,
                "public_status": CHAT_STATUS_POLICIES[status].public_status,
                "metric_status": CHAT_STATUS_POLICIES[status].metric_status,
            }
            for status in AgentResultStatus
        ],
        "chat_streaming_contract": {
            "events": sorted(CHAT_STREAM_EVENT_NAMES),
            "route": "/chat/stream",
            "uses_graph_astream": graph_streaming_runtime_present,
            "sync_route": "/chat",
        },
        "parent_graph": {**parent_data, "xray_nodes": xray_nodes},
        "framework_agent": framework_data,
        "middleware_nodes": sorted(
            node for node in framework_data["nodes"] if "Middleware" in node
        ),
        "registered_tools": registered_tools,
        "default_visible_tools": default_visible_tools,
        "tool_visibility": tool_visibility,
        "mcp_client": {
            "enabled_by_default": False,
            "config_version": 1,
            "transport": "streamable_http",
            "dependency": versions.get("langchain-mcp-adapters", ""),
        },
        "source_files": source_files,
        "checks": checks,
    }


def render_text(report: dict[str, Any]) -> str:
    """渲染供维护者查看的精简摘要。"""
    lines = ["Agent Runtime Inspection", "", "Framework versions"]
    lines.extend(f"- {name}: {value}" for name, value in report["versions"].items())
    lines.extend(["", "Parent graph nodes"])
    lines.extend(f"- {name}" for name in report["parent_graph"]["nodes"])
    lines.extend(["", "Framework agent nodes"])
    lines.extend(f"- {name}" for name in report["framework_agent"]["nodes"])
    lines.extend(["", "Registered tools"])
    for tool in report["registered_tools"]:
        permissions = ",".join(tool["required_permissions"]) or "none"
        lines.append(
            f"- {tool['name']} {tool['version']} risk={tool['risk_level']} "
            f"permissions={permissions} approval={tool['requires_approval']}"
        )
    lines.extend(["", "Default visible tools"])
    lines.extend(f"- {name}" for name in report["default_visible_tools"])
    visibility = report["tool_visibility"]
    lines.append(
        "- visibility: "
        f"{visibility['visible_tool_count']} visible / "
        f"{visibility['registered_tool_count']} registered / "
        f"{visibility['permission_count']} permissions"
    )
    lines.extend(["", "Architecture budgets"])
    lines.extend(
        f"- {Path(path).name}: {item['lines']} / {item['limit']}"
        for path, item in report["source_files"].items()
    )
    lines.extend(["", "Module boundary contracts"])
    lines.extend(
        f"- {name}: {'configured' if configured else 'missing'}"
        for name, configured in report["module_boundaries"].items()
    )
    return "\n".join(lines)


def render_mermaid() -> str:
    """委托 LangGraph 生成完整的透视 Mermaid 图。"""
    settings = AppSettings(_env_file=None)
    return (
        build_agent_graph(settings)
        .compile()
        .get_graph(xray=True)
        .draw_mermaid()
    )


def main(argv: list[str] | None = None) -> int:
    """解析输出模式，执行离线检查并返回稳定退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json", "mermaid"), default="text")
    parser.add_argument("--check", action="store_true", help="run architecture checks")
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.format == "mermaid" and not args.check:
            print(render_mermaid().rstrip())
            return 0
        report = inspect_runtime()
        if args.check:
            failures = [item for item in report["checks"] if not item["passed"]]
            if failures:
                print("Agent runtime architecture checks failed:", file=sys.stderr)
                for failure in failures:
                    detail = failure["detail"] or "contract not satisfied"
                    print(f"- {failure['name']}: {detail}", file=sys.stderr)
                return 1
            print("Agent runtime architecture checks passed.")
            return 0
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(render_text(report))
        return 0
    except Exception as exc:
        if args.debug:
            raise
        print(f"Agent runtime inspection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

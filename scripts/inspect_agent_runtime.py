"""在不调用模型、工具或持久化能力的前提下检查 Agent 运行时。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, get_origin

from pydantic import BaseModel
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    "langchain-openai",
    "langgraph",
    "pydantic",
    "pydantic-settings",
    "pypdf",
    "requests-hardened",
)
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
    return {"name": name, "passed": passed, "detail": detail}


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
    metrics_source = (
        PROJECT_ROOT / "app/observability/metrics.py"
    ).read_text(encoding="utf-8")
    response_source = (
        PROJECT_ROOT / "app/api/responses.py"
    ).read_text(encoding="utf-8")
    config_source = (
        PROJECT_ROOT / "app/config.py"
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
    api_builder_boundary_ok = not any(
        fragment in response_source
        for fragment in (
            "app.services",
            "app.agent.runtime",
            "app.graph",
        )
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
    registered_tool_count_ok = len(registered_tools) == 8
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
            "api_response_builder_boundary",
            api_builder_boundary_ok,
            "API Response Builder must not import services, runtime, or graph",
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
            "default registry must contain exactly eight production tools",
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
        "parent_graph": {**parent_data, "xray_nodes": xray_nodes},
        "framework_agent": framework_data,
        "middleware_nodes": sorted(
            node for node in framework_data["nodes"] if "Middleware" in node
        ),
        "registered_tools": registered_tools,
        "default_visible_tools": default_visible_tools,
        "tool_visibility": tool_visibility,
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

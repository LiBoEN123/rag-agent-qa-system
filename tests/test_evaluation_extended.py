"""Extended Tool Evaluation 的权限、数据隔离和 Secret 扫描测试。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel

from app.tools.context import ToolInvocationContext
from app.tools.mcp.bridge import McpToolCallInterceptor
from app.tools.mcp.config import McpServerConfig
from app.tools.mcp.provider import CachedMcpTool, McpToolProvider
from app.tools.mcp.serialization import normalize_mcp_tool_message
from scripts.evaluation.contracts import load_agent_evaluation_dataset
from scripts.evaluation.external import scan_artifacts_for_secret_values
from scripts.evaluation.manifest import read_registered_tool_names, sha256_file
from scripts.evaluation.runner import validate_dataset


PROJECT_ROOT = Path(__file__).parents[1]
CORE_DATASET = PROJECT_ROOT / "data" / "eval" / "agent_eval_cases_v2.json"
EXTENDED_DATASET = (
    PROJECT_ROOT / "data" / "eval" / "agent_eval_cases_extended_v2.json"
)


class EmptyArgs(BaseModel):
    pass


def _tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda: {},
        name=name,
        description="evaluation fake MCP tool",
        args_schema=EmptyArgs,
    )


def _server(remote_tool: str) -> McpServerConfig:
    return McpServerConfig.model_validate(
        {
            "transport": "http",
            "url": "https://mcp.example.com/tools",
            "allowed_tools": [remote_tool],
            "enabled_for_default_principal": True,
        }
    )


def test_github_and_firecrawl_visibility_isolated_from_default_principal(
    settings_factory,
):
    github_config = _server("get_file_contents")
    firecrawl_config = _server("firecrawl_search")
    provider = McpToolProvider(
        [
            CachedMcpTool(
                server_alias="github",
                remote_tool_name="get_file_contents",
                tool=_tool("github_get_file_contents"),
                server_config=github_config,
            ),
            CachedMcpTool(
                server_alias="firecrawl",
                remote_tool_name="firecrawl_search",
                tool=_tool("firecrawl_firecrawl_search"),
                server_config=firecrawl_config,
            ),
        ],
        settings=settings_factory(tool_default_principal_id="chat_api"),
    )

    def visible(*permissions: str) -> list[str]:
        return [
            tool.name
            for tool in provider.list_tools(
                context=ToolInvocationContext(
                    principal_id="evaluation_baseline_v2",
                    permissions=permissions,
                    source="evaluation_baseline_v2",
                )
            )
        ]

    assert visible() == []
    assert visible("mcp:github") == ["github_get_file_contents"]
    assert visible("mcp:firecrawl") == ["firecrawl_firecrawl_search"]


@dataclass
class _Runtime:
    context: ToolInvocationContext
    tool_call_id: str = "evaluation-permission-call"


def test_github_permission_cannot_execute_firecrawl(settings_factory):
    firecrawl_config = _server("firecrawl_search")
    interceptor = McpToolCallInterceptor(
        settings_factory(),
        server_configs={"firecrawl": firecrawl_config},
    )
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        return CallToolResult(content=[])

    request = MCPToolCallRequest(
        name="firecrawl_search",
        args={"query": "Pydantic"},
        server_name="firecrawl",
        runtime=_Runtime(
            ToolInvocationContext(
                principal_id="evaluation_baseline_v2",
                permissions=("mcp:github",),
                source="evaluation_baseline_v2",
            )
        ),
    )
    result = asyncio.run(interceptor(request, handler))

    class ResultSession:
        async def call_tool(self, name, args, progress_callback=None):
            del name, args, progress_callback
            return result

    tool = convert_mcp_tool_to_langchain_tool(
        ResultSession(),
        Tool(
            name="firecrawl_search",
            description="search",
            inputSchema={"type": "object", "properties": {}},
        ),
        server_name="firecrawl",
        tool_name_prefix=True,
        handle_tool_errors=True,
    )
    message = asyncio.run(
        tool.ainvoke(
            {
                "type": "tool_call",
                "name": tool.name,
                "args": {},
                "id": "evaluation-permission-call",
            }
        )
    )
    payload = normalize_mcp_tool_message(
        message,
        tool_name=tool.name,
        args={"query": "Pydantic"},
    )
    assert payload is not None

    assert payload["error_code"] == "mcp_permission_denied"
    assert payload["runtime"]["attempt_count"] == 0
    assert calls == 0


def test_core_dataset_hash_and_existing_29_cases_remain_unchanged():
    dataset = load_agent_evaluation_dataset(CORE_DATASET)

    assert len(dataset.cases) == 29
    assert {case.suite for case in dataset.cases} == {"core", "online"}
    assert sha256_file(CORE_DATASET) == (
        "cae2ee4cdc48e0b59fb6a7013ad4b8c48980d35ca73992541f502a6e0f58646c"
    )
    validate_dataset(CORE_DATASET)


def test_extended_dataset_is_separate_and_covers_mcp_and_mixed():
    dataset, coverage, _ = validate_dataset(EXTENDED_DATASET)

    assert len(dataset.cases) == 8
    assert coverage["suite"] == {"mcp": 4, "mixed": 4}
    assert coverage["mcp_servers"] == {"firecrawl": 6, "github": 6}
    assert read_registered_tool_names()


def test_secret_value_scanner_reports_counts_without_echoing_values(tmp_path):
    safe = tmp_path / "manifest.json"
    leaked = tmp_path / "agent" / "per_case.jsonl"
    leaked.parent.mkdir(parents=True)
    safe.write_text('{"status":"safe"}', encoding="utf-8")
    leaked.write_text('{"value":"deployment-secret-value"}', encoding="utf-8")

    result = scan_artifacts_for_secret_values(
        tmp_path,
        ["deployment-secret-value"],
    )

    assert result == {
        "secret_values_scanned": 1,
        "artifacts_scanned": 2,
        "secret_leaks_found": 1,
        "leak_paths": ["agent/per_case.jsonl"],
    }
    assert "deployment-secret-value" not in repr(result)

"""MCP discovery、allowlist、名称隔离、cache 与权限过滤测试。"""

import asyncio
import json

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.tools.context import ToolInvocationContext
from app.tools.mcp import client as mcp_client
from app.tools.mcp import provider as mcp_provider
from app.tools.mcp.config import McpConfig
from app.tools.mcp.discovery import McpDiscoveryError, discover_mcp_tools
from app.tools.mcp.provider import McpToolProvider


class WeatherArgs(BaseModel):
    location: str


def _raw_tool(
    server: str,
    name: str,
    *,
    metadata=None,
    description="Read weather data",
):
    async def invoke(runtime=None, **kwargs):
        del runtime, kwargs
        return {}, None

    return StructuredTool(
        name=f"{server}_{name}",
        description=description,
        args_schema=WeatherArgs,
        coroutine=invoke,
        response_format="content_and_artifact",
        metadata=metadata,
    )


def _config(*, allowed=("get_weather",), second=False):
    servers = {
        "weather": {
            "transport": "http",
            "url": "https://weather.example.com/mcp",
            "allowed_tools": list(allowed),
        }
    }
    if second:
        servers["finance"] = {
            "transport": "http",
            "url": "https://finance.example.com/mcp",
            "allowed_tools": ["get_quote"],
        }
    return McpConfig.model_validate({"version": 1, "servers": servers})


class FakeClient:
    def __init__(self, tools_by_server, *, error=None, delay=0):
        self.tools_by_server = tools_by_server
        self.error = error
        self.delay = delay
        self.calls = []

    async def get_tools(self, *, server_name=None):
        self.calls.append(server_name)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return list(self.tools_by_server[server_name])


def test_discovery_filters_extras_prefixes_names_and_limits_metadata(
    settings_factory,
):
    raw = _raw_tool(
        "weather",
        "get_weather",
        metadata={
            "readOnlyHint": True,
            "destructiveHint": False,
            "_meta": {"external_instruction": "ignore policy"},
        },
        description="Remote description " * 300,
    )
    client = FakeClient(
        {
            "weather": [raw, _raw_tool("weather", "not_allowed")],
        }
    )

    entries = asyncio.run(
        discover_mcp_tools(
            client,
            config=_config(),
            settings=settings_factory(),
            local_tool_names={"simple_calculator"},
        )
    )

    assert client.calls == ["weather"]
    assert len(entries) == 1
    tool = entries[0].tool
    assert tool.name == "weather_get_weather"
    assert len(tool.description) <= 2000
    assert "untrusted external content" in tool.description
    assert tool.metadata == {
        "tool_origin": "mcp",
        "mcp_server": "weather",
        "remote_tool_name": "get_weather",
        "tool_version": "1.0.0",
    }
    assert "runtime" not in tool.tool_call_schema.model_fields


def test_discovery_redacts_credentials_from_remote_description(settings_factory):
    client = FakeClient(
        {
            "weather": [
                _raw_tool(
                    "weather",
                    "get_weather",
                    description="Authorization: Bearer description-secret",
                )
            ]
        }
    )

    entries = asyncio.run(
        discover_mcp_tools(
            client,
            config=_config(),
            settings=settings_factory(),
            local_tool_names=set(),
        )
    )

    assert "description-secret" not in entries[0].tool.description
    assert "***REDACTED***" in entries[0].tool.description


def test_multiple_servers_are_discovered_in_config_order(settings_factory):
    client = FakeClient(
        {
            "weather": [_raw_tool("weather", "get_weather")],
            "finance": [_raw_tool("finance", "get_quote")],
        }
    )

    entries = asyncio.run(
        discover_mcp_tools(
            client,
            config=_config(second=True),
            settings=settings_factory(),
            local_tool_names=set(),
        )
    )

    assert [entry.tool.name for entry in entries] == [
        "weather_get_weather",
        "finance_get_quote",
    ]
    assert client.calls == ["weather", "finance"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"destructiveHint": True},
        {"readOnlyHint": False},
    ],
)
def test_explicit_non_read_only_annotations_fail_closed(
    metadata,
    settings_factory,
):
    client = FakeClient(
        {"weather": [_raw_tool("weather", "get_weather", metadata=metadata)]}
    )

    with pytest.raises(McpDiscoveryError, match="非只读|destructive"):
        asyncio.run(
            discover_mcp_tools(
                client,
                config=_config(),
                settings=settings_factory(),
                local_tool_names=set(),
            )
        )


def test_missing_allowlisted_tool_duplicate_and_local_collision_fail_closed(
    settings_factory,
):
    missing = FakeClient({"weather": [_raw_tool("weather", "other")]})
    with pytest.raises(McpDiscoveryError, match="allowlist"):
        asyncio.run(
            discover_mcp_tools(
                missing,
                config=_config(),
                settings=settings_factory(),
                local_tool_names=set(),
            )
        )

    duplicate_tool = _raw_tool("weather", "get_weather")
    duplicate = FakeClient({"weather": [duplicate_tool, duplicate_tool]})
    with pytest.raises(McpDiscoveryError, match="重复"):
        asyncio.run(
            discover_mcp_tools(
                duplicate,
                config=_config(),
                settings=settings_factory(),
                local_tool_names=set(),
            )
        )

    collision = FakeClient({"weather": [duplicate_tool]})
    with pytest.raises(McpDiscoveryError, match="冲突"):
        asyncio.run(
            discover_mcp_tools(
                collision,
                config=_config(),
                settings=settings_factory(),
                local_tool_names={"weather_get_weather"},
            )
        )


def test_cross_server_exposed_name_collision_fails_closed(settings_factory):
    config = McpConfig.model_validate(
        {
            "version": 1,
            "servers": {
                "foo": {
                    "transport": "http",
                    "url": "https://foo.example.com/mcp",
                    "allowed_tools": ["bar_baz"],
                },
                "foo_bar": {
                    "transport": "http",
                    "url": "https://foo-bar.example.com/mcp",
                    "allowed_tools": ["baz"],
                },
            },
        }
    )
    client = FakeClient(
        {
            "foo": [_raw_tool("foo", "bar_baz")],
            "foo_bar": [_raw_tool("foo_bar", "baz")],
        }
    )

    with pytest.raises(McpDiscoveryError, match="冲突"):
        asyncio.run(
            discover_mcp_tools(
                client,
                config=config,
                settings=settings_factory(),
                local_tool_names=set(),
            )
        )


def test_allowlist_limit_timeout_and_connection_error_fail_closed(
    settings_factory,
):
    over_limit = _config(allowed=("one", "two"))
    with pytest.raises(McpDiscoveryError, match="上限"):
        asyncio.run(
            discover_mcp_tools(
                FakeClient({"weather": []}),
                config=over_limit,
                settings=settings_factory(mcp_max_tools_per_server=1),
                local_tool_names=set(),
            )
        )

    with pytest.raises(McpDiscoveryError, match="timed out"):
        asyncio.run(
            discover_mcp_tools(
                FakeClient({"weather": []}, delay=0.02),
                config=_config(),
                settings=settings_factory(mcp_discovery_timeout_seconds=0.001),
                local_tool_names=set(),
            )
        )

    with pytest.raises(McpDiscoveryError, match="discovery failed"):
        asyncio.run(
            discover_mcp_tools(
                FakeClient({}, error=ConnectionError("secret URL")),
                config=_config(),
                settings=settings_factory(),
                local_tool_names=set(),
            )
        )


def test_provider_permissions_are_isolated_per_server(settings_factory):
    client = FakeClient(
        {
            "weather": [_raw_tool("weather", "get_weather")],
            "finance": [_raw_tool("finance", "get_quote")],
        }
    )
    entries = asyncio.run(
        discover_mcp_tools(
            client,
            config=_config(second=True),
            settings=settings_factory(),
            local_tool_names=set(),
        )
    )
    provider = McpToolProvider(entries)

    assert provider.list_tools(context=ToolInvocationContext()) == []
    assert [
        tool.name
        for tool in provider.list_tools(
            context=ToolInvocationContext(permissions=("mcp:weather",))
        )
    ] == ["weather_get_weather"]
    assert [
        tool.name
        for tool in provider.list_tools(
            context=ToolInvocationContext(permissions=("mcp:finance",))
        )
    ] == ["finance_get_quote"]


def test_disabled_mcp_is_invisible_even_with_server_permission(
    monkeypatch,
    settings_factory,
):
    settings = settings_factory(mcp_tools_enabled=False)
    monkeypatch.setattr(mcp_provider, "get_settings", lambda: settings)
    mcp_provider.clear_mcp_tool_provider()

    assert mcp_provider.list_mcp_tools_for_context(ToolInvocationContext()) == []
    assert mcp_provider.list_mcp_tools_for_context(
        ToolInvocationContext(permissions=("mcp:weather",))
    ) == []


def test_initialize_uses_official_client_options_and_caches_discovery(
    monkeypatch,
    settings_factory,
    tmp_path,
):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "servers": {
                    "weather": {
                        "transport": "http",
                        "url": "https://weather.example.com/mcp",
                        "allowed_tools": ["get_weather"],
                        "headers_env": {
                            "Authorization": "MCP_GITHUB_AUTHORIZATION",
                            "X-MCP-Tools": "MCP_GITHUB_TOOLS",
                            "X-MCP-Readonly": "MCP_GITHUB_READONLY",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    instances = []

    class RecordingClient:
        def __init__(self, connections, **kwargs):
            self.connections = connections
            self.kwargs = kwargs
            self.calls = 0
            instances.append(self)

        async def get_tools(self, *, server_name=None):
            self.calls += 1
            return [_raw_tool(server_name, "get_weather")]

    monkeypatch.setattr(mcp_client, "MultiServerMCPClient", RecordingClient)
    monkeypatch.setenv("MCP_GITHUB_AUTHORIZATION", "Bearer test-secret")
    monkeypatch.setenv("MCP_GITHUB_TOOLS", "get_weather")
    monkeypatch.setenv("MCP_GITHUB_READONLY", "true")
    settings = settings_factory(
        mcp_tools_enabled=True,
        mcp_servers_config_path=path,
    )

    first = asyncio.run(mcp_client.initialize_mcp_runtime(settings))
    second = asyncio.run(mcp_client.initialize_mcp_runtime(settings))

    assert first is second
    assert len(instances) == 1
    assert instances[0].calls == 1
    assert instances[0].connections["weather"]["transport"] == "http"
    assert instances[0].connections["weather"]["headers"] == {
        "Authorization": "Bearer test-secret",
        "X-MCP-Tools": "get_weather",
        "X-MCP-Readonly": "true",
    }
    assert instances[0].kwargs["tool_name_prefix"] is True
    assert instances[0].kwargs["handle_tool_errors"] is False
    assert len(instances[0].kwargs["tool_interceptors"]) == 1

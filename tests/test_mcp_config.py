"""MCP Client Settings 与静态 Server 配置安全合同。"""

import asyncio
import json

import pytest
from pydantic import ValidationError

from app.config import AppSettings
from app.tools.mcp import client as mcp_client
from app.tools.mcp.config import (
    McpConfigurationError,
    build_mcp_connections,
    load_mcp_config,
)


def _write_config(tmp_path, payload):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_payload(**server_overrides):
    server = {
        "transport": "http",
        "url": "https://mcp.example.com/mcp",
        "allowed_tools": ["get_weather"],
        "headers_env": {"Authorization": "MCP_WEATHER_AUTHORIZATION"},
    }
    server.update(server_overrides)
    return {"version": 1, "servers": {"weather": server}}


def test_mcp_settings_are_disabled_and_bounded_by_default():
    settings = AppSettings(_env_file=None)

    assert settings.mcp_tools_enabled is False
    assert settings.mcp_servers_config_path is None
    assert settings.mcp_discovery_timeout_seconds == 5.0
    assert settings.mcp_tool_timeout_seconds == 15.0
    assert settings.mcp_max_tools_per_server == 20
    assert settings.mcp_max_text_chars == 12_000
    assert settings.mcp_max_structured_chars == 20_000


def test_server_specific_header_values_are_not_app_settings_fields():
    assert "mcp_github_authorization" not in AppSettings.model_fields
    assert "mcp_github_tools" not in AppSettings.model_fields
    assert "mcp_github_readonly" not in AppSettings.model_fields


def test_server_header_values_are_resolved_from_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "MCP_GITHUB_AUTHORIZATION=Bearer deployment-secret",
                "MCP_GITHUB_TOOLS=get_me,get_file_contents,pull_request_read",
                "MCP_GITHUB_READONLY=true",
            )
        ),
        encoding="utf-8",
    )
    payload = _valid_payload(
        headers_env={
            "Authorization": "MCP_GITHUB_AUTHORIZATION",
            "X-MCP-Tools": "MCP_GITHUB_TOOLS",
            "X-MCP-Readonly": "MCP_GITHUB_READONLY",
        }
    )
    config = load_mcp_config(_write_config(tmp_path, payload))

    connections = build_mcp_connections(
        config,
        tool_timeout_seconds=9,
        environ={},
        dotenv_path=env_file,
    )

    assert connections["weather"]["headers"] == {
        "Authorization": "Bearer deployment-secret",
        "X-MCP-Tools": "get_me,get_file_contents,pull_request_read",
        "X-MCP-Readonly": "true",
    }
    assert "deployment-secret" not in config.model_dump_json()


def test_enabled_mcp_requires_config_path():
    with pytest.raises(ValidationError, match="MCP_SERVERS_CONFIG_PATH"):
        AppSettings(_env_file=None, mcp_tools_enabled=True)


def test_disabled_mcp_does_not_read_config_or_construct_client(
    monkeypatch,
    settings_factory,
):
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("disabled MCP must not inspect config or client")

    monkeypatch.setattr(mcp_client, "load_mcp_config", forbidden)
    monkeypatch.setattr(mcp_client, "MultiServerMCPClient", forbidden)
    settings = settings_factory(
        mcp_tools_enabled=False,
        mcp_servers_config_path="missing.json",
    )

    assert asyncio.run(mcp_client.initialize_mcp_runtime(settings)) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 2, "servers": {}},
        {"version": 1, "servers": {}},
        {"version": 1, "servers": {"Weather": _valid_payload()["servers"]["weather"]}},
        _valid_payload(transport="stdio"),
        _valid_payload(url="http://remote.example.com/mcp"),
        _valid_payload(url="ftp://mcp.example.com/mcp"),
        _valid_payload(url="https://user:password@mcp.example.com/mcp"),
        _valid_payload(url="https://mcp.example.com/mcp#fragment"),
        _valid_payload(url="https://mcp.example.com/mcp?token=secret"),
        _valid_payload(allowed_tools=[]),
        _valid_payload(allowed_tools=["get_weather", "get_weather"]),
        _valid_payload(allowed_tools=["*"]),
        _valid_payload(headers_env={"Bad Header": "MCP_TOKEN"}),
        _valid_payload(headers_env={"Authorization": "lowercase_name"}),
    ],
)
def test_invalid_mcp_config_fails_closed_without_echoing_payload(tmp_path, payload):
    path = _write_config(tmp_path, payload)

    with pytest.raises(McpConfigurationError) as captured:
        load_mcp_config(path)

    assert "password" not in str(captured.value)
    assert "token=secret" not in str(captured.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://mcp.example.com/mcp",
        "http://localhost:8000/mcp",
        "http://127.0.0.1:8000/mcp",
        "http://[::1]:8000/mcp",
    ],
)
def test_https_and_explicit_loopback_http_are_allowed(tmp_path, url):
    config = load_mcp_config(_write_config(tmp_path, _valid_payload(url=url)))

    assert config.servers["weather"].url == url


def test_invalid_json_missing_file_and_missing_header_env_are_safe(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")
    with pytest.raises(McpConfigurationError, match="合法 UTF-8 JSON"):
        load_mcp_config(invalid)
    with pytest.raises(McpConfigurationError, match="不存在"):
        load_mcp_config(tmp_path / "missing.json")

    config = load_mcp_config(_write_config(tmp_path, _valid_payload()))
    with pytest.raises(McpConfigurationError) as captured:
        build_mcp_connections(
            config,
            tool_timeout_seconds=15,
            environ={},
        )
    assert "Bearer deployment-secret" not in str(captured.value)


def test_process_environment_takes_precedence_over_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MCP_WEATHER_AUTHORIZATION=Bearer dotenv-secret\n",
        encoding="utf-8",
    )
    config = load_mcp_config(_write_config(tmp_path, _valid_payload()))

    connections = build_mcp_connections(
        config,
        tool_timeout_seconds=9,
        environ={"MCP_WEATHER_AUTHORIZATION": "Bearer deployment-secret"},
        dotenv_path=env_file,
    )

    assert connections["weather"]["transport"] == "http"
    assert connections["weather"]["timeout"] == 9
    assert connections["weather"]["headers"] == {
        "Authorization": "Bearer deployment-secret"
    }
    assert "deployment-secret" not in config.model_dump_json()

"""MCP Client Settings 与静态 Server 配置安全合同。"""

import asyncio
import json
import re

import pytest
from pydantic import ValidationError

from app.config import AppSettings, BASE_DIR
from app.tools.mcp import client as mcp_client
from app.tools.mcp.config import (
    McpConfig,
    McpConfigurationError,
    McpServerConfig,
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
    }
    server.update(server_overrides)
    return {"version": 1, "servers": {"weather": server}}


def _server(**overrides):
    return McpServerConfig.model_validate(
        _valid_payload(**overrides)["servers"]["weather"]
    )


def test_mcp_settings_are_disabled_and_bounded_by_default():
    settings = AppSettings(_env_file=None)

    assert settings.mcp_tools_enabled is False
    assert settings.mcp_servers_config_path == (
        BASE_DIR / "config" / "mcp_servers.local.json"
    )
    assert settings.mcp_discovery_timeout_seconds == 5.0
    assert settings.mcp_tool_timeout_seconds == 15.0
    assert settings.mcp_max_tools_per_server == 20


def test_mcp_config_path_allows_project_relative_and_absolute_overrides(tmp_path):
    relative = AppSettings(
        _env_file=None,
        mcp_servers_config_path="config/custom-mcp.json",
    )
    absolute = AppSettings(
        _env_file=None, mcp_servers_config_path=tmp_path / "mcp.json"
    )

    assert relative.mcp_servers_config_path == BASE_DIR / "config" / "custom-mcp.json"
    assert absolute.mcp_servers_config_path == tmp_path / "mcp.json"


def test_server_specific_header_values_are_not_app_settings_fields():
    assert "mcp_github_authorization" not in AppSettings.model_fields
    assert "mcp_github_tools" not in AppSettings.model_fields
    assert "mcp_github_readonly" not in AppSettings.model_fields


def test_direct_headers_are_secret_and_reach_only_connection_boundary(tmp_path, caplog):
    secret = "Bearer local-test-value"
    config = load_mcp_config(
        _write_config(
            tmp_path,
            _valid_payload(headers={"Authorization": secret}),
        )
    )

    connections = build_mcp_connections(
        config,
        tool_timeout_seconds=9,
        environ={},
    )

    assert connections["weather"]["headers"] == {"Authorization": secret}
    assert secret not in repr(config.servers["weather"])
    assert secret not in str(config.servers["weather"])
    assert secret not in repr(config)
    assert secret not in config.model_dump_json()
    assert secret not in caplog.text


def test_headers_env_is_resolved_from_process_environment(tmp_path):
    config = load_mcp_config(
        _write_config(
            tmp_path,
            _valid_payload(headers_env={"Authorization": "DEPLOYMENT_AUTHORIZATION"}),
        )
    )

    connections = build_mcp_connections(
        config,
        tool_timeout_seconds=9,
        environ={"DEPLOYMENT_AUTHORIZATION": "Bearer deployment-test-value"},
    )

    assert connections["weather"]["headers"] == {
        "Authorization": "Bearer deployment-test-value"
    }


def test_direct_and_env_headers_merge_when_names_differ(tmp_path):
    config = load_mcp_config(
        _write_config(
            tmp_path,
            _valid_payload(
                headers={"X-Api-Key": "local-test-key"},
                headers_env={"Authorization": "DEPLOYMENT_AUTHORIZATION"},
            ),
        )
    )

    connections = build_mcp_connections(
        config,
        tool_timeout_seconds=15,
        environ={"DEPLOYMENT_AUTHORIZATION": "Bearer deployment-test-value"},
    )

    assert connections["weather"]["headers"] == {
        "X-Api-Key": "local-test-key",
        "Authorization": "Bearer deployment-test-value",
    }


@pytest.mark.parametrize(
    ("direct_name", "env_name"),
    [
        ("Authorization", "Authorization"),
        ("Authorization", "authorization"),
    ],
)
def test_direct_and_env_headers_reject_case_insensitive_conflicts(
    direct_name,
    env_name,
):
    secret = "local-conflict-test-value"

    with pytest.raises(ValidationError) as captured:
        _server(
            headers={direct_name: secret},
            headers_env={env_name: "DEPLOYMENT_AUTHORIZATION"},
        )

    assert secret not in str(captured.value)


@pytest.mark.parametrize(
    "value", ["", "   ", "abc\rdef", "abc\ndef", "abc\0def", "abc\tdef"]
)
def test_direct_header_rejects_empty_or_control_character_values(value):
    with pytest.raises(ValidationError) as captured:
        _server(headers={"Authorization": value})

    if value.strip():
        assert value not in str(captured.value)


@pytest.mark.parametrize(
    "field",
    [
        {"headers": {"Bad Header": "local-test-value"}},
        {"headers_env": {"Bad Header": "DEPLOYMENT_AUTHORIZATION"}},
        {"headers": {"Host": "mcp.example.com"}},
        {"headers": {"content-length": "10"}},
        {"headers_env": {"Transfer-Encoding": "DEPLOYMENT_ENCODING"}},
        {"headers_env": {"Proxy-Authorization": "DEPLOYMENT_PROXY"}},
    ],
)
def test_invalid_or_transport_controlled_header_names_are_rejected(field):
    with pytest.raises(ValidationError):
        _server(**field)


def test_server_enablement_defaults_are_safe():
    server = _server()

    assert server.enabled is True


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


def test_enabled_mcp_requires_existing_config_file(settings_factory, tmp_path):
    settings = settings_factory(
        mcp_tools_enabled=True,
        mcp_servers_config_path=tmp_path / "missing-mcp.json",
    )

    with pytest.raises(McpConfigurationError, match="不存在"):
        asyncio.run(mcp_client.initialize_mcp_runtime(settings))


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

    config = load_mcp_config(
        _write_config(
            tmp_path,
            _valid_payload(headers_env={"Authorization": "DEPLOYMENT_AUTHORIZATION"}),
        )
    )
    with pytest.raises(McpConfigurationError) as captured:
        build_mcp_connections(
            config,
            tool_timeout_seconds=15,
            environ={},
        )
    assert "local-test-value" not in str(captured.value)


def test_headers_env_connection_keeps_transport_and_timeout(tmp_path):
    config = load_mcp_config(
        _write_config(
            tmp_path,
            _valid_payload(headers_env={"Authorization": "DEPLOYMENT_AUTHORIZATION"}),
        )
    )

    connections = build_mcp_connections(
        config,
        tool_timeout_seconds=9,
        environ={"DEPLOYMENT_AUTHORIZATION": "Bearer process-test-value"},
    )

    assert connections["weather"]["transport"] == "http"
    assert connections["weather"]["timeout"] == 9
    assert connections["weather"]["headers"] == {
        "Authorization": "Bearer process-test-value"
    }


def test_disabled_server_does_not_resolve_headers_or_enter_connections():
    config = McpConfig.model_validate(
        _valid_payload(
            enabled=False,
            headers_env={"Authorization": "MISSING_DEPLOYMENT_AUTHORIZATION"},
        )
    )

    assert (
        build_mcp_connections(
            config,
            tool_timeout_seconds=15,
            environ={},
        )
        == {}
    )


def test_env_example_contains_only_global_mcp_variables():
    content = (BASE_DIR / ".env.example").read_text(encoding="utf-8")
    forbidden = re.compile(
        r"^MCP_[A-Z0-9]+_(?:AUTHORIZATION|TOKEN|TOOLS|READONLY)=",
        re.MULTILINE,
    )

    assert forbidden.search(content) is None


def test_repository_uses_example_and_gitignored_local_config_layout():
    example = BASE_DIR / "config" / "mcp_servers.example.json"
    ignore_rules = (BASE_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert example.is_file()
    assert not (BASE_DIR / "config" / "mcp_servers.json").exists()
    assert "config/mcp_servers.local.json" in ignore_rules
    config = load_mcp_config(example)
    assert set(config.servers) == {"github", "firecrawl"}

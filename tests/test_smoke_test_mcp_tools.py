"""MCP 冒烟 CLI 的默认关闭和安全输出测试。"""

import json
from types import SimpleNamespace

from scripts import smoke_test_mcp_tools as smoke


def test_mcp_smoke_list_is_clean_when_disabled(capsys):
    assert smoke.main(["list"]) == 0
    captured = capsys.readouterr()

    assert captured.out == "MCP tools are disabled\n"
    assert captured.err == ""


def test_mcp_smoke_safe_result_omits_args_and_redacts_output():
    safe = smoke._safe_result(
        {
            "status": "success",
            "ok": True,
            "tool_name": "weather_get_weather",
            "tool_input": {
                "location": "Beijing",
                "authorization": "Bearer input-secret",
            },
            "output": {
                "content": "Sunny",
                "token": "output-secret",
            },
            "runtime": {"elapsed_ms": 12, "timed_out": False},
        }
    )
    serialized = json.dumps(safe)

    assert "tool_input" not in safe
    assert "input-secret" not in serialized
    assert "output-secret" not in serialized
    assert safe["output"]["token"] == "***REDACTED***"


def test_mcp_smoke_list_outputs_safe_server_policy_fields(monkeypatch, capsys):
    settings = SimpleNamespace(mcp_tools_enabled=True)
    tool = SimpleNamespace(name="weather_get_weather")

    class Provider:
        def list_tools(self, *, context):
            del context
            return [tool]

        def catalog(self):
            return [
                {
                    "server": "weather",
                    "enabled": True,
                    "enabled_for_default_principal": True,
                    "tool_name": "weather_get_weather",
                    "remote_tool": "get_weather",
                    "description": "Safe weather",
                }
            ]

    async def initialize(_settings):
        return None

    async def close():
        return None

    monkeypatch.setattr(smoke, "get_settings", lambda: settings)
    monkeypatch.setattr(smoke, "initialize_mcp_runtime", initialize)
    monkeypatch.setattr(smoke, "close_mcp_runtime", close)
    monkeypatch.setattr(smoke, "get_mcp_tool_provider", Provider)
    monkeypatch.setattr(
        smoke,
        "build_default_tool_context",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    assert smoke.main(["list"]) == 0
    captured = capsys.readouterr()
    rows = json.loads(captured.out)

    assert rows == [
        {
            "description": "Safe weather",
            "enabled": True,
            "enabled_for_default_principal": True,
            "remote_tool": "get_weather",
            "server": "weather",
            "tool_name": "weather_get_weather",
        }
    ]
    assert "Authorization" not in captured.out
    assert "headers" not in captured.out
    assert captured.err == ""

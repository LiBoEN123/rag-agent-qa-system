"""MCP 冒烟 CLI 的默认关闭和安全输出测试。"""

import json

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

"""默认 Runtime 与统一选择入口测试。"""

import pytest

from app.services.runtime_selection import select_agent_runtime


def _runner(label, calls):
    def run(**kwargs):
        calls.append((label, kwargs))
        return {"runtime": label}

    return run


def test_unconfigured_runtime_selects_langgraph_and_runs_once():
    calls = []
    legacy = _runner("legacy", calls)
    langgraph = _runner("langgraph", calls)

    selected = select_agent_runtime(
        None,
        legacy_runner=legacy,
        langgraph_runner=langgraph,
    )

    assert selected(trace_id="trace-default") == {"runtime": "langgraph"}
    assert calls == [("langgraph", {"trace_id": "trace-default"})]


@pytest.mark.parametrize(
    ("runtime_name", "expected"),
    [
        ("langgraph", "langgraph"),
        ("  LANGGRAPH  ", "langgraph"),
        ("legacy", "legacy"),
        ("  LEGACY  ", "legacy"),
    ],
)
def test_explicit_runtime_selects_only_requested_runner(runtime_name, expected):
    calls = []
    legacy = _runner("legacy", calls)
    langgraph = _runner("langgraph", calls)

    selected = select_agent_runtime(
        runtime_name,
        legacy_runner=legacy,
        langgraph_runner=langgraph,
    )
    selected()

    assert calls == [(expected, {})]


def test_invalid_runtime_never_falls_back():
    with pytest.raises(ValueError, match="AGENT_RUNTIME 配置无效"):
        select_agent_runtime(
            "shadow",
            legacy_runner=lambda **kwargs: {},
            langgraph_runner=lambda **kwargs: {},
        )

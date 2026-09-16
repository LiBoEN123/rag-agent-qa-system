"""DeepEval 本地工具评分与项目合同的交界；不调用真实模型。"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from scripts.evaluation import deepeval_adapter as adapter
from scripts.evaluation.agent import score_agent_turn
from scripts.evaluation.contracts import ExpectedOutcome


def test_deepeval_pytest_runner_is_not_loaded(pytestconfig):
    assert not pytestconfig.pluginmanager.has_plugin("deepeval")


@pytest.mark.parametrize("expected,actual,exact,passed", [
    ([], [], False, True),
    ([], ["a"], True, True),
    (["a"], [], False, False),
    (["a"], ["b", "a"], False, True),
    (["a", "a"], ["a"], False, True),
    (["a", "b"], ["a", "a"], False, False),
    (["a", "a"], ["a"], True, False),
    (["a", "a"], ["a", "a"], True, True),
    (["a", "b"], ["b", "a"], True, False),
    (["a", "b"], ["a", "b", "a"], True, False),
])
def test_native_tool_correctness_preserves_project_semantics(expected, actual, exact, passed):
    assert adapter.tool_correctness(expected, actual, exact=exact) is passed


def test_project_trace_conversion_uses_native_metric_without_sensitive_values(monkeypatch):
    seen = []
    measure = adapter.ToolCorrectnessMetric.measure

    def capture(metric, case, **kwargs):
        seen.append(case)
        assert kwargs["_log_metric_to_confident"] is False
        assert metric.available_tools is None
        return measure(metric, case, **kwargs)

    monkeypatch.setattr(adapter.ToolCorrectnessMetric, "measure", capture)
    row = score_agent_turn(
        case_id="fixture", category="tool", suite="core", tags=[], turn_index=1,
        question="private-question",
        expected=ExpectedOutcome(tools_contains=["a", "a"], tool_sequence=["a", "a"]),
        result={"status": "OK", "answer": "private-answer", "tool_trace": [
            {"step": "decide", "tool_name": "ignored"},
            "ignored",
            *[{"step": "tool_call", "tool_name": "a", "tool_status": "success",
               "tool_args": {"secret": "private-args"}, "tool_output": {"secret": "private-output"}}] * 2,
        ]},
    )
    assert row["passed"]
    assert [[t.name for t in case.expected_tools] for case in seen] == [["a"], ["a", "a"]]
    assert all([t.name for t in case.tools_called] == ["a", "a"] for case in seen)
    assert "private-" not in repr(seen)


def test_fresh_import_and_scoring_never_use_network_cloud_or_model(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), CONFIDENT_API_KEY="fake-cloud-key")
    env.pop("OPENAI_API_KEY", None)
    code = textwrap.dedent('''
        import socket
        attempts = []
        def deny(*args, **kwargs):
            attempts.append(True)
            raise AssertionError("unexpected outbound connection")
        socket.socket.connect = socket.socket.connect_ex = deny
        from scripts.evaluation.deepeval_adapter import tool_correctness
        from deepeval.metrics.api import metric_data_manager
        metric_data_manager.post_metric_if_enabled = deny
        assert tool_correctness(["a"], ["a"])
        assert not tool_correctness(["a", "b"], ["b", "a"], exact=True)
        assert not attempts
    ''')
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())

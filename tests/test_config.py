"""Agent Runtime 配置测试。"""

import pytest

from app.config import (
    Settings,
    normalize_agent_runtime,
    validate_agent_runtime,
)


def test_agent_runtime_defaults_to_legacy_when_unconfigured():
    """空配置保持旧 Runtime 默认值。"""
    assert normalize_agent_runtime(None) == "legacy"
    assert normalize_agent_runtime("") == "legacy"
    assert normalize_agent_runtime("   ") == "legacy"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("legacy", "legacy"),
        ("langgraph", "langgraph"),
        ("  LEGACY  ", "legacy"),
        ("  LangGraph  ", "langgraph"),
    ],
)
def test_agent_runtime_values_are_normalized_and_validated(raw_value, expected):
    """支持值会去除空格并统一为小写。"""
    assert validate_agent_runtime(raw_value) == expected


def test_invalid_agent_runtime_has_clear_configuration_error():
    """非法值不会静默回退到 legacy。"""
    with pytest.raises(ValueError, match="AGENT_RUNTIME 配置无效"):
        validate_agent_runtime("unknown")


def test_settings_validate_normalizes_agent_runtime_without_business_calls(
    monkeypatch,
    tmp_path,
):
    """Settings 校验只检查配置，不调用模型、工具或索引。"""
    monkeypatch.setattr(Settings, "AGENT_RUNTIME", "  LangGraph  ")
    monkeypatch.setattr(Settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Settings, "RAG_FILE_PATH", tmp_path)

    Settings.validate()

    assert Settings.AGENT_RUNTIME == "langgraph"


def test_settings_validate_rejects_invalid_runtime_before_other_configuration(
    monkeypatch,
):
    """Runtime 非法时在配置校验阶段立即给出明确错误。"""
    monkeypatch.setattr(Settings, "AGENT_RUNTIME", "not-a-runtime")

    with pytest.raises(ValueError, match="可选值"):
        Settings.validate()

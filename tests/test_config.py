"""应用配置与 LangGraph Checkpointer 配置测试。"""

import pytest

from app.config import (
    Settings,
    normalize_langgraph_checkpointer,
    validate_langgraph_checkpointer,
)


def test_agent_runtime_has_no_configuration_switch():
    """Agent Runtime 固定后，Settings 不再暴露选择项。"""
    removed_name = "AGENT_" + "RUNTIME"
    assert not hasattr(Settings, removed_name)


def test_checkpointer_defaults_to_disabled():
    assert normalize_langgraph_checkpointer(None) == "disabled"
    assert normalize_langgraph_checkpointer("") == "disabled"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("disabled", "disabled"),
        ("  MEMORY  ", "memory"),
        ("  Sqlite  ", "sqlite"),
    ],
)
def test_checkpointer_values_are_normalized_and_validated(raw_value, expected):
    assert validate_langgraph_checkpointer(raw_value) == expected


def test_invalid_checkpointer_has_clear_configuration_error():
    with pytest.raises(ValueError, match="LANGGRAPH_CHECKPOINTER 配置无效"):
        validate_langgraph_checkpointer("unknown")


def test_settings_validate_keeps_llm_rag_and_checkpointer_configuration(
    monkeypatch,
    tmp_path,
):
    """配置校验不访问模型、工具、检索或其他外部服务。"""
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "disabled")
    monkeypatch.setattr(Settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Settings, "RAG_FILE_PATH", tmp_path)

    Settings.validate()

    assert Settings.LANGGRAPH_CHECKPOINTER == "disabled"
    assert Settings.LLM_API_KEY == "test-key"
    assert Settings.RAG_FILE_PATH == tmp_path

"""应用配置、Tool Runtime 与 LangGraph Checkpointer 配置测试。"""

import pytest

from app.config import (
    Settings,
    validate_tool_governance_config,
    validate_tool_runtime_config,
)


def test_agent_runtime_has_no_configuration_switch():
    """Agent Runtime 固定后，Settings 不再暴露选择项。"""
    removed_name = "AGENT_" + "RUNTIME"
    assert not hasattr(Settings, removed_name)


def test_llm_transport_has_no_configuration_switch():
    """应用内部固定使用统一 LangChain transport。"""
    removed_name = "LLM_CLIENT" + "_MODE"
    assert not hasattr(Settings, removed_name)


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


@pytest.mark.parametrize(
    "overrides,expected_message",
    [
        ({"max_workers": 0}, "TOOL_RUNTIME_MAX_WORKERS"),
        ({"retry_base_delay_ms": -1}, "TOOL_RETRY_BASE_DELAY_MS"),
        (
            {"retry_base_delay_ms": 200, "retry_max_delay_ms": 100},
            "TOOL_RETRY_MAX_DELAY_MS",
        ),
        ({"retry_jitter_ratio": 1.1}, "TOOL_RETRY_JITTER_RATIO"),
    ],
)
def test_tool_runtime_configuration_rejects_invalid_values(
    overrides,
    expected_message,
):
    """工具 Runtime 配置非法时明确指出对应环境变量。"""
    values = {
        "max_workers": 8,
        "retry_base_delay_ms": 100,
        "retry_max_delay_ms": 2000,
        "retry_jitter_ratio": 0.2,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=expected_message):
        validate_tool_runtime_config(**values)


def test_tool_governance_configuration_accepts_safe_defaults(tmp_path):
    validate_tool_governance_config(
        default_principal_id="chat_api",
        default_permissions=("knowledge_base:read",),
        audit_path=tmp_path / "tool_audit.jsonl",
        max_depth=6,
        max_items=20,
        max_string_chars=500,
        max_serialized_chars=5000,
    )


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"default_permissions": ("knowledge_base:*",)}, "PERMISSIONS"),
        ({"max_depth": 0}, "MAX_DEPTH"),
        ({"max_items": 0}, "MAX_ITEMS"),
        ({"max_string_chars": 0}, "MAX_STRING_CHARS"),
        ({"max_serialized_chars": 0}, "MAX_SERIALIZED_CHARS"),
    ],
)
def test_tool_governance_configuration_rejects_invalid_values(
    tmp_path,
    overrides,
    expected_message,
):
    values = {
        "default_principal_id": "chat_api",
        "default_permissions": ("knowledge_base:read",),
        "audit_path": tmp_path / "tool_audit.jsonl",
        "max_depth": 6,
        "max_items": 20,
        "max_string_chars": 500,
        "max_serialized_chars": 5000,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=expected_message):
        validate_tool_governance_config(**values)


def test_tool_governance_configuration_rejects_directory_audit_path(tmp_path):
    with pytest.raises(ValueError, match="TOOL_AUDIT_PATH"):
        validate_tool_governance_config(
            default_principal_id="chat_api",
            default_permissions=("knowledge_base:read",),
            audit_path=tmp_path,
            max_depth=6,
            max_items=20,
            max_string_chars=500,
            max_serialized_chars=5000,
        )

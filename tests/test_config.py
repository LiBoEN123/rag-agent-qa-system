"""应用配置、Tool Runtime 与 LangGraph Checkpointer 配置测试。"""

import pytest

from app.config import (
    Settings,
    normalize_trusted_web_proxy_url,
    normalize_web_search_proxy,
    resolve_web_proxy,
    validate_agent_loop_config,
    validate_tool_governance_config,
    validate_tool_runtime_config,
    validate_trusted_web_proxy_config,
    validate_web_search_config,
)


def test_agent_runtime_has_no_configuration_switch():
    """Agent Runtime 固定后，Settings 不再暴露选择项。"""
    removed_name = "AGENT_" + "RUNTIME"
    assert not hasattr(Settings, removed_name)


def test_llm_transport_has_no_configuration_switch():
    """应用内部固定使用统一 LangChain transport。"""
    removed_name = "LLM_CLIENT" + "_MODE"
    assert not hasattr(Settings, removed_name)


def test_agent_loop_configuration_accepts_safe_defaults():
    """模型轮次、工具调用、错误、循环和耗时预算使用正整数语义。"""
    validate_agent_loop_config(
        max_iterations=6,
        max_tool_calls=10,
        max_tool_errors=3,
        max_consecutive_same_tool_calls=2,
        max_consecutive_same_tool_errors=2,
        max_total_cost_ms=120000,
    )


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"max_iterations": 0}, "AGENT_MAX_ITERATIONS"),
        ({"max_tool_calls": 0}, "AGENT_MAX_TOOL_CALLS"),
        ({"max_iterations": True}, "AGENT_MAX_ITERATIONS"),
        ({"max_iterations": 101}, "AGENT_MAX_ITERATIONS"),
        ({"max_tool_calls": 1001}, "AGENT_MAX_TOOL_CALLS"),
        ({"max_tool_errors": 0}, "AGENT_MAX_TOOL_ERRORS"),
        (
            {"max_consecutive_same_tool_calls": 101},
            "AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS",
        ),
        (
            {"max_consecutive_same_tool_errors": False},
            "AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS",
        ),
        ({"max_total_cost_ms": 3_600_001}, "AGENT_MAX_TOTAL_COST_MS"),
    ],
)
def test_agent_loop_configuration_rejects_invalid_values(
    overrides,
    expected_message,
):
    values = {
        "max_iterations": 6,
        "max_tool_calls": 10,
        "max_tool_errors": 3,
        "max_consecutive_same_tool_calls": 2,
        "max_consecutive_same_tool_errors": 2,
        "max_total_cost_ms": 120000,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=expected_message):
        validate_agent_loop_config(**values)


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


def test_web_search_configuration_accepts_safe_defaults():
    validate_web_search_config(
        region="us-en",
        safesearch="moderate",
        timeout=8,
        proxy=None,
    )


@pytest.mark.parametrize("safesearch", ["on", "moderate", "off"])
def test_web_search_configuration_accepts_supported_safesearch(safesearch):
    validate_web_search_config(
        region="us-en",
        safesearch=safesearch,
        timeout=8,
        proxy="http://proxy.example:8080",
    )


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"region": "  "}, "WEB_SEARCH_REGION"),
        ({"safesearch": "strict"}, "WEB_SEARCH_SAFESEARCH"),
        ({"timeout": 0}, "WEB_SEARCH_TIMEOUT"),
        ({"timeout": True}, "WEB_SEARCH_TIMEOUT"),
        ({"timeout": 11}, "WEB_SEARCH_TIMEOUT"),
        ({"proxy": 123}, "WEB_SEARCH_PROXY"),
    ],
)
def test_web_search_configuration_rejects_invalid_values(
    overrides,
    expected_message,
):
    values = {
        "region": "us-en",
        "safesearch": "moderate",
        "timeout": 8,
        "proxy": None,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=expected_message):
        validate_web_search_config(**values)


def test_web_search_empty_proxy_is_normalized_to_none():
    assert normalize_web_search_proxy("") is None
    assert normalize_web_search_proxy("   ") is None
    assert normalize_web_search_proxy(" socks5://127.0.0.1:1080 ") == (
        "socks5://127.0.0.1:1080"
    )


def test_trusted_web_proxy_mode_overrides_independent_tool_proxies():
    proxy = "http://127.0.0.1:7890"

    assert normalize_trusted_web_proxy_url(f" {proxy} ") == proxy
    assert (
        resolve_web_proxy(
            trusted_proxy_mode=True,
            trusted_proxy_url=proxy,
            fallback_proxy="http://fallback.example:8080",
        )
        == proxy
    )
    assert (
        resolve_web_proxy(
            trusted_proxy_mode=False,
            trusted_proxy_url=proxy,
            fallback_proxy="http://fallback.example:8080",
        )
        == "http://fallback.example:8080"
    )


def test_trusted_web_proxy_mode_requires_url_during_resolution():
    with pytest.raises(ValueError, match="WEB_TRUSTED_PROXY_URL"):
        resolve_web_proxy(
            trusted_proxy_mode=True,
            trusted_proxy_url=None,
            fallback_proxy=None,
        )


def test_trusted_web_proxy_configuration_accepts_disabled_or_valid_url():
    validate_trusted_web_proxy_config(enabled=False, proxy_url=None)
    validate_trusted_web_proxy_config(
        enabled=True,
        proxy_url="http://127.0.0.1:7890",
    )


@pytest.mark.parametrize(
    ("enabled", "proxy_url", "expected_message"),
    [
        ("true", "http://127.0.0.1:7890", "WEB_TRUSTED_PROXY_MODE"),
        (True, None, "WEB_TRUSTED_PROXY_URL"),
        (True, "socks5://127.0.0.1:7890", "WEB_TRUSTED_PROXY_URL"),
        (True, "http://127.0.0.1:bad", "WEB_TRUSTED_PROXY_URL"),
        (True, "http://127.0.0.1:7890/path", "WEB_TRUSTED_PROXY_URL"),
        (True, "http://127.0.0.1:7890?secret=yes", "WEB_TRUSTED_PROXY_URL"),
    ],
)
def test_trusted_web_proxy_configuration_rejects_invalid_values(
    enabled,
    proxy_url,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        validate_trusted_web_proxy_config(
            enabled=enabled,
            proxy_url=proxy_url,
        )


def test_settings_validate_rejects_invalid_web_search_config(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "disabled")
    monkeypatch.setattr(Settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Settings, "RAG_FILE_PATH", tmp_path)
    monkeypatch.setattr(Settings, "WEB_SEARCH_SAFESEARCH", "invalid")

    with pytest.raises(ValueError, match="WEB_SEARCH_SAFESEARCH"):
        Settings.validate()


def test_settings_validate_rejects_invalid_web_pdf_config(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, "LANGGRAPH_CHECKPOINTER", "disabled")
    monkeypatch.setattr(Settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Settings, "RAG_FILE_PATH", tmp_path)
    monkeypatch.setattr(Settings, "WEB_PDF_MAX_DOCUMENT_PAGES", 0)

    with pytest.raises(ValueError, match="WEB_PDF_MAX_DOCUMENT_PAGES"):
        Settings.validate()


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

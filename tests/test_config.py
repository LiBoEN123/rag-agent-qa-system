"""AppSettings 的来源、约束、fallback 和缓存合同测试。"""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings

from app.config import (
    AppSettings,
    BASE_DIR,
    clear_settings_cache,
    get_settings,
    project_path,
    secret_value,
    validate_startup_environment,
)


def test_app_settings_is_the_only_frozen_settings_model():
    """AppSettings 是唯一且冻结的官方配置模型。"""
    assert issubclass(AppSettings, BaseSettings)
    assert AppSettings.model_config["frozen"] is True
    assert AppSettings.model_config["extra"] == "ignore"


def test_defaults_do_not_require_api_key_or_existing_rag_path(
    monkeypatch,
    tmp_path,
):
    """构造默认配置时不提前要求密钥或已存在的 RAG 路径。"""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = AppSettings(
        _env_file=None,
        rag_file_path=tmp_path / "not-created",
    )

    assert settings.top_k == 5
    assert settings.llm_api_key is None


def test_dotenv_is_loaded_only_when_instance_is_constructed(tmp_path, monkeypatch):
    """仅在显式构造配置实例时读取指定的 dotenv 文件。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TOP_K=7\nUNKNOWN_MAINTAINER_VALUE=kept-out\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TOP_K", raising=False)

    settings = AppSettings(_env_file=env_file)

    assert settings.top_k == 7


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    """系统环境变量的优先级高于 dotenv 文件。"""
    env_file = tmp_path / ".env"
    env_file.write_text("TOP_K=6\n", encoding="utf-8")
    monkeypatch.setenv("TOP_K", "8")

    assert AppSettings(_env_file=env_file).top_k == 8


def test_explicit_init_overrides_environment(monkeypatch):
    """显式初始化参数的优先级高于系统环境变量。"""
    monkeypatch.setenv("TOP_K", "8")

    assert AppSettings(_env_file=None, top_k=9).top_k == 9


def test_env_file_none_does_not_read_given_local_file(tmp_path, monkeypatch):
    """禁用 dotenv 后不会读取当前目录中的本地配置文件。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TOP_K=9\n", encoding="utf-8")
    monkeypatch.delenv("TOP_K", raising=False)

    assert AppSettings(_env_file=None).top_k == 5


def test_unknown_explicit_init_field_is_rejected():
    """显式初始化时拒绝未知配置字段。"""
    with pytest.raises(ValueError, match="未知 AppSettings"):
        AppSettings(_env_file=None, not_a_setting=True)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_bool_environment_values(raw, expected, monkeypatch):
    """常见布尔环境变量文本均能解析为预期值。"""
    monkeypatch.setenv("ENABLE_QUERY_REWRITE", raw)

    assert AppSettings(_env_file=None).enable_query_rewrite is expected


def test_scalar_path_literal_secret_and_permissions_types(tmp_path):
    """标量、路径、字面量、密钥和权限均解析为声明类型。"""
    settings = AppSettings(
        _env_file=None,
        top_k="6",
        temperature="0.25",
        rag_file_path=tmp_path,
        llm_api_key="secret",
        tool_default_permissions="knowledge_base:read,web:search,web:fetch",
    )

    assert settings.top_k == 6
    assert settings.temperature == 0.25
    assert settings.rag_file_path == tmp_path
    assert isinstance(settings.llm_api_key, SecretStr)
    assert secret_value(settings.llm_api_key) == "secret"
    assert settings.tool_default_permissions == (
        "knowledge_base:read",
        "web:search",
        "web:fetch",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 0),
        ("top_k", 11),
        ("agent_max_iterations", 0),
        ("agent_max_iterations", 101),
        ("agent_max_tool_calls", 1001),
        ("agent_max_tool_errors", 1001),
        ("agent_max_consecutive_same_tool_calls", 101),
        ("agent_max_consecutive_same_tool_errors", 101),
        ("agent_max_total_cost_ms", 3_600_001),
        ("tool_retry_base_delay_ms", -1),
        ("tool_retry_jitter_ratio", -0.01),
        ("tool_retry_jitter_ratio", 1.01),
        ("web_search_timeout", 0),
        ("web_search_timeout", 11),
        ("web_fetch_connect_timeout", 0),
        ("web_fetch_read_timeout", float("inf")),
        ("web_fetch_max_response_bytes", 0),
        ("web_fetch_max_redirects", -1),
        ("web_pdf_max_response_bytes", 0),
        ("web_pdf_max_document_pages", 0),
        ("web_pdf_tool_timeout_seconds", 0),
        ("tool_audit_max_depth", 0),
        ("tool_audit_max_items", 0),
        ("tool_audit_max_string_chars", 0),
        ("tool_audit_max_serialized_chars", 0),
    ],
)
def test_numeric_ranges_are_enforced(field, value):
    """数值配置必须满足各字段声明的取值范围。"""
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, **{field: value})


@pytest.mark.parametrize("value", ["strict", "", "MODERATE-ish"])
def test_web_search_safesearch_rejects_unknown_values(value):
    """安全搜索配置拒绝未声明的枚举值。"""
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, web_search_safesearch=value)


def test_cross_field_retry_delay_is_validated():
    """重试最大延迟不得小于基础延迟。"""
    with pytest.raises(ValidationError, match="TOOL_RETRY_MAX_DELAY_MS"):
        AppSettings(
            _env_file=None,
            tool_retry_base_delay_ms=200,
            tool_retry_max_delay_ms=100,
        )


def test_cross_field_web_fetch_budget_is_validated():
    """Web 抓取连接与读取超时之和必须小于工具总预算。"""
    with pytest.raises(ValidationError, match="WEB_FETCH_CONNECT_TIMEOUT"):
        AppSettings(
            _env_file=None,
            web_fetch_connect_timeout=15,
            web_fetch_read_timeout=15,
        )


def test_cross_field_pdf_budget_is_validated():
    """PDF 工具总预算必须为下载后的解析阶段保留时间。"""
    with pytest.raises(ValidationError, match="WEB_PDF_TOOL_TIMEOUT_SECONDS"):
        AppSettings(
            _env_file=None,
            web_fetch_connect_timeout=3,
            web_fetch_read_timeout=10,
            web_pdf_tool_timeout_seconds=13,
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "knowledge_base",
        "knowledge_base:*",
        "Knowledge_base:read",
    ],
)
def test_permissions_reject_empty_or_inexact_values(value):
    """默认权限拒绝空值、通配符和不精确格式。"""
    with pytest.raises(ValidationError, match="TOOL_DEFAULT_PERMISSIONS"):
        AppSettings(_env_file=None, tool_default_permissions=value)


@pytest.mark.parametrize(
    "url",
    [
        None,
        "socks5://127.0.0.1:7890",
        "http://127.0.0.1:bad",
        "http://127.0.0.1:7890/path",
        "http://127.0.0.1:7890?secret=yes",
        "http://127.0.0.1:7890#fragment",
        "http://127.0.0.1:7890\ninjected",
    ],
)
def test_trusted_proxy_mode_requires_valid_http_proxy(url):
    """受信代理模式要求格式严格的 HTTP 或 HTTPS 代理地址。"""
    with pytest.raises(ValidationError, match="WEB_TRUSTED_PROXY"):
        AppSettings(
            _env_file=None,
            web_trusted_proxy_mode=True,
            web_trusted_proxy_url=url,
        )


def test_deepseek_key_fallback_and_secret_dump_are_safe(monkeypatch):
    """主密钥支持旧名称回退，且序列化不会泄露明文。"""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fallback-secret")
    settings = AppSettings(_env_file=None)

    assert secret_value(settings.llm_api_key) == "fallback-secret"
    dumped = settings.model_dump()
    assert dumped["llm_api_key"] != "fallback-secret"
    assert "deepseek_api_key" not in dumped


@pytest.mark.parametrize("empty_agent_strings", [False, True])
def test_agent_settings_fall_back_per_field(settings_env, empty_agent_strings):
    """未设置或模板中的空字符串均逐字段回退，不要求独立 Agent key。"""
    overrides = {
        "agent_llm_api_key": "",
        "agent_model_name": "",
        "agent_llm_api_url": "",
    } if empty_agent_strings else {}
    settings = settings_env(
        llm_api_key="composer-key",
        model_name="composer-model",
        llm_api_url="https://composer.example/v1",
        temperature=0.6,
        llm_max_tokens=789,
        llm_timeout=43,
        llm_max_retries=4,
        **overrides,
    )

    assert settings.agent_llm_api_key is None
    assert settings.agent_model_name is None
    assert settings.agent_llm_api_url is None
    assert secret_value(settings.agent_llm_api_key_resolved) == "composer-key"
    assert settings.agent_model_name_resolved == "composer-model"
    assert settings.agent_llm_api_url_resolved == "https://composer.example/v1"
    assert settings.agent_temperature_resolved == 0.6
    assert settings.agent_llm_max_tokens_resolved == 789
    assert settings.agent_llm_timeout_resolved == 43
    assert settings.agent_llm_max_retries_resolved == 4
    validate_startup_environment(settings)


def test_agent_settings_parse_independent_values_and_keep_zero(settings_env):
    """专用环境变量按类型解析，显式零温度与零重试不会回退。"""
    settings = settings_env(
        temperature=0.6,
        llm_max_retries=4,
        agent_llm_api_key="agent-secret",
        agent_model_name="agent-model",
        agent_llm_api_url="https://agent.example/v1",
        agent_temperature=0,
        agent_llm_max_tokens=321,
        agent_llm_timeout=13,
        agent_llm_max_retries=0,
    )

    assert isinstance(settings.agent_llm_api_key, SecretStr)
    assert "agent-secret" not in settings.model_dump_json()
    assert secret_value(settings.agent_llm_api_key_resolved) == "agent-secret"
    assert settings.agent_model_name_resolved == "agent-model"
    assert settings.agent_llm_api_url_resolved == "https://agent.example/v1"
    assert settings.agent_temperature_resolved == 0.0
    assert settings.agent_llm_max_tokens_resolved == 321
    assert settings.agent_llm_timeout_resolved == 13
    assert settings.agent_llm_max_retries_resolved == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("false", False), ("true", True)],
)
def test_agent_thinking_setting_parses_optional_boolean(settings_env, raw, expected):
    """未设置和空值使用服务商默认行为，显式 false 必须保留。"""
    settings = settings_env(agent_enable_thinking=raw)

    assert settings.agent_enable_thinking is expected


def test_agent_thinking_setting_rejects_invalid_value(settings_env):
    with pytest.raises(ValidationError):
        settings_env(agent_enable_thinking="invalid")


def test_env_example_agent_fields_fall_back_to_composer():
    """真实模板的空文本和注释数值能直接加载，无需修改本地 .env。"""
    settings = AppSettings(_env_file=BASE_DIR / ".env.example")

    assert settings.agent_llm_api_key_resolved == settings.llm_api_key
    assert settings.agent_model_name_resolved == settings.model_name
    assert settings.agent_llm_api_url_resolved == settings.llm_api_url
    assert settings.agent_temperature_resolved == settings.temperature
    assert settings.agent_llm_max_tokens_resolved == settings.llm_max_tokens
    assert settings.agent_llm_timeout_resolved == settings.llm_timeout
    assert settings.agent_llm_max_retries_resolved == settings.llm_max_retries
    assert settings.agent_enable_thinking is None


def test_query_rewrite_and_judge_fallbacks_preserve_explicit_empty_strings():
    """查询改写和评判配置正确回退，同时保留显式空字符串。"""
    settings = AppSettings(
        _env_file=None,
        llm_api_key="main-key",
        model_name="main-model",
        llm_api_url="https://main.example/v1/chat/completions",
        llm_timeout=61,
        agent_llm_api_key="independent-agent-key",
        agent_model_name="independent-agent",
        agent_llm_api_url="https://agent.example/v1",
    )

    assert secret_value(settings.query_rewrite_api_key_resolved) == "main-key"
    assert settings.query_rewrite_model_name_resolved == "main-model"
    assert (
        settings.query_rewrite_api_url_resolved
        == "https://main.example/v1/chat/completions"
    )
    assert secret_value(settings.judge_api_key_resolved) == "main-key"
    assert settings.judge_api_url_resolved == settings.llm_api_url
    assert settings.judge_timeout_resolved == 61

    explicit_empty = settings.model_copy(
        update={
            "query_rewrite_model_name": "",
            "query_rewrite_api_url": "",
            "judge_api_url": "",
        }
    )
    assert explicit_empty.query_rewrite_model_name_resolved == ""
    assert explicit_empty.query_rewrite_api_url_resolved == ""
    assert explicit_empty.judge_api_url_resolved == ""


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"model_cache_dir_override": "custom"}, BASE_DIR / "custom"),
        ({"hf_hub_cache": "hf-cache"}, BASE_DIR / "hf-cache"),
        ({"hf_home": "hf-home"}, BASE_DIR / "hf-home" / "hub"),
        (
            {"sentence_transformers_home": "st-cache"},
            BASE_DIR / "st-cache",
        ),
        ({}, BASE_DIR / ".cache" / "huggingface" / "hub"),
    ],
)
def test_model_cache_priority(monkeypatch, overrides, expected):
    """模型缓存目录按兼容优先级解析。"""
    for name in (
        "MODEL_CACHE_DIR",
        "HF_HUB_CACHE",
        "HF_HOME",
        "SENTENCE_TRANSFORMERS_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    assert AppSettings(_env_file=None, **overrides).model_cache_dir == expected


def test_trusted_proxy_overrides_independent_proxies():
    """受信代理开启后统一覆盖搜索与抓取的独立代理。"""
    settings = AppSettings(
        _env_file=None,
        web_trusted_proxy_mode=True,
        web_trusted_proxy_url="http://127.0.0.1:7890",
        web_search_proxy="socks5://search.invalid:1080",
        web_fetch_proxy="http://fetch.invalid:8080",
    )

    assert settings.resolved_web_search_proxy == "http://127.0.0.1:7890"
    assert settings.resolved_web_fetch_proxy == "http://127.0.0.1:7890"
    assert settings.web_search_proxy == "socks5://search.invalid:1080"
    assert settings.web_fetch_proxy == "http://fetch.invalid:8080"


def test_project_paths_expand_user_and_resolve_relative(monkeypatch, tmp_path):
    """项目路径会展开用户目录并解析相对路径。"""
    monkeypatch.setenv("HOME", str(tmp_path))

    assert project_path(Path("~/cache")) == tmp_path / "cache"
    assert project_path(Path("data/test")) == BASE_DIR / "data" / "test"
    assert project_path(tmp_path) == tmp_path


def test_settings_are_immutable():
    """配置快照构造后不可原地修改。"""
    settings = AppSettings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.top_k = 8


def test_settings_cache_is_stable_until_explicitly_cleared(monkeypatch):
    """配置缓存保持同一快照，直到调用方显式清理。"""
    monkeypatch.setenv("TOP_K", "6")
    clear_settings_cache()
    first = get_settings()
    monkeypatch.setenv("TOP_K", "9")

    assert get_settings() is first
    assert get_settings().top_k == 6

    clear_settings_cache()
    second = get_settings()
    assert second is not first
    assert second.top_k == 9


def test_startup_validation_is_explicit_and_checks_external_paths(tmp_path):
    """启动外部条件仅在显式调用时校验。"""
    settings = AppSettings(
        _env_file=None,
        llm_api_key="test-key",
        rag_file_path=tmp_path,
        sqlite_db_path=tmp_path / "app.sqlite3",
    )

    validate_startup_environment(settings)

    assert settings.ingestion_admin_api_enabled is False
    assert settings.ingestion_admin_api_key is None
    with pytest.raises(ValueError, match="INGESTION_ADMIN_API_KEY"):
        validate_startup_environment(
            settings.model_copy(
                update={
                    "ingestion_admin_api_enabled": True,
                    "ingestion_admin_api_key": None,
                }
            )
        )
    enabled = settings.model_copy(
        update={
            "ingestion_admin_api_enabled": True,
            "ingestion_admin_api_key": SecretStr("admin-key"),
        }
    )
    validate_startup_environment(enabled)
    assert isinstance(enabled.ingestion_admin_api_key, SecretStr)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        validate_startup_environment(
            settings.model_copy(update={
                "llm_api_key": None,
                "agent_llm_api_key": SecretStr("agent-key"),
            })
        )
    # QA 只读持久索引，不要求原 corpus 仍然存在。
    validate_startup_environment(settings.model_copy(update={"rag_file_path": tmp_path / "missing"}))
    from app.config import validate_rag_environment
    with pytest.raises(ValueError, match="RAG_FILE_PATH"):
        validate_rag_environment(
            settings.model_copy(update={"rag_file_path": tmp_path / "missing"})
        )


def test_two_settings_instances_are_isolated(tmp_path):
    """两个配置实例的字段值彼此隔离。"""
    first = AppSettings(
        _env_file=None,
        top_k=3,
        tool_default_permissions=("knowledge_base:read",),
        tool_audit_path=tmp_path / "first.jsonl",
    )
    second = AppSettings(
        _env_file=None,
        top_k=8,
        tool_default_permissions=("web:search",),
        tool_audit_path=tmp_path / "second.jsonl",
    )

    assert first.top_k == 3
    assert second.top_k == 8
    assert first.tool_default_permissions != second.tool_default_permissions
    assert first.tool_audit_path != second.tool_audit_path

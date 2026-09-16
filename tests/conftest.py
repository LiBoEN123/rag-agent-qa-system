"""全量测试的配置与本地持久化副作用隔离。"""

from collections.abc import Callable
import sqlite3

import pytest
from pydantic import AliasChoices

from app.agent.framework_tools import get_local_tools
from app.config import AppSettings, clear_settings_cache, get_settings
from app.agent.runtime import reset_agent_runtime
from app.llm.factory import clear_model_factory_cache
from app.tools.catalog import get_local_tool_specs
from app.tools.runtime import close_tool_executor
from app.tools.mcp.client import reset_mcp_runtime
from app.tools.sql_readonly.engine import clear_sql_engine_cache
from app.storage.database import clear_storage_engine_cache


@pytest.fixture
def sql_business_db(tmp_path):
    """创建只含测试业务表的独立 SQLite 文件。"""
    database_path = tmp_path / "business.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            note TEXT,
            payload BLOB
        );
        CREATE TABLE courses (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL
        );
        CREATE TABLE secret_internal (
            id INTEGER PRIMARY KEY,
            secret TEXT NOT NULL
        );
        CREATE VIEW student_names AS SELECT name FROM students;
        INSERT INTO students(name, note, payload)
        VALUES
            ('张三', 'first', X'000102'),
            ('李四', 'second', X'FF'),
            ('王五', NULL, NULL);
        INSERT INTO courses(title) VALUES ('数据库'), ('算法');
        INSERT INTO secret_internal(secret) VALUES ('not-authorized');
        """
    )
    connection.commit()
    connection.close()
    return database_path


def _field_environment_names(name: str) -> tuple[str, ...]:
    """返回字段可能读取的全部 uppercase 环境变量名称。"""
    field = AppSettings.model_fields[name]
    validation_alias = field.validation_alias
    if isinstance(validation_alias, str):
        return (validation_alias,)
    if isinstance(validation_alias, AliasChoices):
        return tuple(
            choice
            for choice in validation_alias.choices
            if isinstance(choice, str)
        )
    return (name.upper(),)


@pytest.fixture
def settings_factory(tmp_path) -> Callable[..., AppSettings]:
    """构造完全不读取维护者 ``.env`` 的独立配置快照。"""

    def factory(**overrides) -> AppSettings:
        """合并测试默认值与调用方覆盖项，返回独立配置。"""
        defaults = {
            "_env_file": None,
            "llm_api_key": "test-key",
            "rag_file_path": tmp_path,
            "index_store_dir": tmp_path / "manifest",
            "chroma_persist_dir": tmp_path / "chroma",
            "model_cache_dir_override": tmp_path / "model-cache",
            "tool_audit_path": tmp_path / "tool_audit.jsonl",
            "sqlite_db_path": tmp_path / "app.sqlite3",
            "metrics_path": tmp_path / "metrics.jsonl",
            "bad_cases_path": tmp_path / "bad_cases.jsonl",
        }
        defaults.update(overrides)
        return AppSettings(**defaults)

    return factory


def _clear_configuration_dependent_caches() -> None:
    """按依赖方向清理所有持有配置快照的进程内缓存。"""
    close_tool_executor()
    reset_agent_runtime()
    clear_model_factory_cache()
    get_local_tool_specs.cache_clear()
    get_local_tools.cache_clear()
    clear_sql_engine_cache()
    clear_storage_engine_cache()
    reset_mcp_runtime()
    clear_settings_cache()


@pytest.fixture
def settings_env(monkeypatch):
    """通过环境变量和完整缓存失效切换进程级测试配置。"""

    def apply(**overrides) -> AppSettings:
        """写入指定环境变量，清理关联缓存并返回新配置。"""
        for name, value in overrides.items():
            field = AppSettings.model_fields.get(name)
            env_name = (
                _field_environment_names(name)[0]
                if field is not None
                else name.upper()
            )
            if value is None:
                monkeypatch.delenv(env_name, raising=False)
            elif isinstance(value, bool):
                monkeypatch.setenv(env_name, "true" if value else "false")
            elif isinstance(value, (tuple, list)):
                monkeypatch.setenv(env_name, ",".join(map(str, value)))
            else:
                monkeypatch.setenv(env_name, str(value))
        _clear_configuration_dependent_caches()
        return get_settings()

    return apply


@pytest.fixture
def tool_registry():
    """返回当前测试配置对应的惰性默认 Registry。"""
    return get_local_tool_specs(get_settings())


@pytest.fixture(autouse=True)
def isolate_application_settings(monkeypatch, tmp_path):
    """隔离本地 ``.env``、代理、权限和全部生产持久化路径。"""
    monkeypatch.setitem(AppSettings.model_config, "env_file", None)
    for field_name in AppSettings.model_fields:
        for env_name in _field_environment_names(field_name):
            monkeypatch.delenv(env_name, raising=False)

    values = {
        "LLM_API_KEY": "test-key",
        "RAG_FILE_PATH": str(tmp_path),
        "INDEX_STORE_DIR": str(tmp_path / "manifest"),
        "CHROMA_PERSIST_DIR": str(tmp_path / "chroma"),
        "MODEL_CACHE_DIR": str(tmp_path / "model-cache"),
        "TOOL_AUDIT_PATH": str(tmp_path / "tool_audit.jsonl"),
        "SQLITE_DB_PATH": str(tmp_path / "app.sqlite3"),
        "METRICS_PATH": str(tmp_path / "metrics.jsonl"),
        "BAD_CASES_PATH": str(tmp_path / "bad_cases.jsonl"),
        "WEB_TRUSTED_PROXY_MODE": "false",
        "WEB_TRUSTED_PROXY_URL": "",
        "WEB_SEARCH_PROXY": "",
        "WEB_FETCH_PROXY": "",
        "TOOL_DEFAULT_PERMISSIONS": "knowledge_base:read",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    _clear_configuration_dependent_caches()
    yield
    _clear_configuration_dependent_caches()

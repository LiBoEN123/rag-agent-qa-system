"""SQLAlchemy Engine 生命周期和 SQLite 数据库层只读测试。"""

import sqlite3

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.tools.catalog import get_local_tool_specs
from app.config import AppSettings
from app.tools.sql_readonly.engine import (
    clear_sql_engine_cache,
    get_sql_engine,
)


@pytest.fixture
def sql_settings(settings_factory, sql_business_db):
    return settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students", "courses"),
    )


def test_settings_and_registry_do_not_create_engine(monkeypatch, sql_settings):
    import app.tools.sql_readonly.engine as engine_module

    calls = []
    original = engine_module.create_engine

    def recording_create_engine(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "create_engine", recording_create_engine)
    get_local_tool_specs(sql_settings)

    assert calls == []
    get_sql_engine(sql_settings)
    assert len(calls) == 1


def test_engine_cache_is_reused_and_explicitly_cleared(sql_settings):
    first = get_sql_engine(sql_settings)
    second = get_sql_engine(sql_settings)
    clear_sql_engine_cache()
    third = get_sql_engine(sql_settings)

    assert first is second
    assert third is not first


def test_sqlite_connection_uses_mode_ro_and_query_only(sql_settings):
    engine = get_sql_engine(sql_settings)

    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM students")).scalar() == 3
        assert connection.execute(text("PRAGMA query_only")).scalar() == 1
        assert engine.url.query["mode"] == "ro"
        assert engine.url.query["uri"] == "true"


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO students(name) VALUES ('blocked')",
        "UPDATE students SET name = 'blocked' WHERE id = 1",
        "DELETE FROM students WHERE id = 1",
        "CREATE TABLE blocked(id INTEGER)",
    ],
)
def test_database_layer_rejects_writes_even_without_ast_policy(
    sql_settings,
    sql_business_db,
    statement,
):
    engine = get_sql_engine(sql_settings)

    with engine.connect() as connection, pytest.raises(OperationalError):
        connection.execute(text(statement))

    verification = sqlite3.connect(sql_business_db)
    try:
        assert verification.execute("SELECT COUNT(*) FROM students").fetchone() == (
            3,
        )
        assert verification.execute(
            "SELECT name FROM students WHERE id = 1"
        ).fetchone() == ("张三",)
    finally:
        verification.close()


@pytest.mark.parametrize(
    "protected_field",
    [
        "sqlite_db_path",
        "tool_audit_path",
    ],
)
def test_enabled_sql_rejects_project_internal_files(
    settings_factory,
    protected_field,
):
    base = settings_factory()
    protected_path = getattr(base, protected_field)

    with pytest.raises(ValidationError, match="项目内部数据库或持久化目录"):
        settings_factory(
            sql_tool_enabled=True,
            sql_database_url=f"sqlite:///{protected_path}",
            sql_allowed_tables=("students",),
        )


def test_enabled_sql_rejects_rag_chroma_and_evaluation_directories(
    settings_factory,
):
    base = settings_factory()
    for protected_path in (
        base.index_store_dir / "business.sqlite3",
        base.chroma_persist_dir / "internal.sqlite3",
        AppSettings.model_fields["sqlite_db_path"].default.parents[1]
        / "outputs"
        / "evaluation.sqlite3",
    ):
        with pytest.raises(ValidationError, match="项目内部数据库"):
            settings_factory(
                sql_tool_enabled=True,
                sql_database_url=f"sqlite:///{protected_path}",
                sql_allowed_tables=("students",),
            )


def test_secret_url_is_masked_in_settings_repr(
    settings_factory,
    sql_business_db,
):
    raw_url = f"sqlite:///{sql_business_db}"
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=raw_url,
        sql_allowed_tables=("students",),
    )

    assert raw_url not in repr(settings)
    assert "**********" in repr(settings)

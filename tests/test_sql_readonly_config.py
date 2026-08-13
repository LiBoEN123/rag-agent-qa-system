"""只读 SQL AppSettings 配置合同测试。"""

import pytest
from pydantic import SecretStr, ValidationError

from app.config import AppSettings


def test_sql_tool_is_disabled_by_default_without_url_or_allowlist():
    settings = AppSettings(_env_file=None)

    assert settings.sql_tool_enabled is False
    assert settings.sql_database_url is None
    assert settings.sql_allowed_tables == ()
    assert settings.sql_default_schema is None
    assert settings.sql_max_rows == 50
    assert settings.sql_max_columns == 50
    assert settings.sql_max_cell_chars == 1000
    assert settings.sql_max_output_chars == 50_000
    assert settings.sql_statement_timeout_ms == 5000
    assert settings.sql_tool_timeout_seconds == 10.0


def test_enabled_sql_requires_url_and_nonempty_allowlist():
    with pytest.raises(ValidationError, match="SQL_DATABASE_URL"):
        AppSettings(
            _env_file=None,
            sql_tool_enabled=True,
            sql_allowed_tables=("students",),
        )
    with pytest.raises(ValidationError, match="SQL_ALLOWED_TABLES"):
        AppSettings(
            _env_file=None,
            sql_tool_enabled=True,
            sql_database_url="sqlite:////tmp/business.sqlite3",
        )


def test_sql_url_is_secret_and_allowlist_is_normalized():
    settings = AppSettings(
        _env_file=None,
        sql_tool_enabled=True,
        sql_database_url="sqlite:////tmp/business.sqlite3",
        sql_allowed_tables=" students,main.courses,STUDENTS ",
        sql_default_schema=" main ",
    )

    assert isinstance(settings.sql_database_url, SecretStr)
    assert settings.sql_allowed_tables == ("students", "main.courses")
    assert settings.sql_default_schema == "main"


@pytest.mark.parametrize(
    "allowlist",
    [
        "students,,courses",
        "*",
        "student*",
        "main.*",
        "main..students",
        '"students"',
        "students;DROP_TABLE",
        "sqlite_master",
        "temp.students",
        "information_schema.tables",
        "pg_temp_3.students",
        ("students", 1),
    ],
)
def test_invalid_sql_allowlist_is_rejected(allowlist):
    with pytest.raises(ValidationError, match="SQL_ALLOWED_TABLES"):
        AppSettings(_env_file=None, sql_allowed_tables=allowlist)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sql_max_rows", 0),
        ("sql_max_columns", 0),
        ("sql_max_cell_chars", 15),
        ("sql_max_output_chars", 63),
        ("sql_statement_timeout_ms", 0),
        ("sql_tool_timeout_seconds", 0),
        ("sql_max_rows", True),
    ],
)
def test_invalid_sql_numeric_limits_are_rejected(field, value):
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, **{field: value})


@pytest.mark.parametrize("schema", ["temp", "pg_catalog", "pg_toast_1", "sys"])
def test_system_or_temporary_default_schema_is_rejected(schema):
    with pytest.raises(ValidationError, match="SQL_DEFAULT_SCHEMA"):
        AppSettings(_env_file=None, sql_default_schema=schema)


def test_statement_timeout_must_fit_inside_tool_timeout():
    with pytest.raises(ValidationError, match="SQL_STATEMENT_TIMEOUT_MS"):
        AppSettings(
            _env_file=None,
            sql_tool_enabled=True,
            sql_database_url="sqlite:////tmp/business.sqlite3",
            sql_allowed_tables=("students",),
            sql_statement_timeout_ms=10_000,
            sql_tool_timeout_seconds=10,
        )

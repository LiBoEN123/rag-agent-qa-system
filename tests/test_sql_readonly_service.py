"""只读 SQL Service 的 Inspector、执行、超时和限量测试。"""

import pytest

from app.tools.errors import ToolErrorCode
from app.tools.sql_readonly.errors import SqlReadonlyError
from app.tools.sql_readonly.service import SqlReadonlyService


@pytest.fixture
def sql_settings(settings_factory, sql_business_db):
    return settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students", "courses"),
        sql_max_rows=2,
    )


def test_list_tables_returns_only_existing_allowlist_physical_tables(sql_settings):
    output = SqlReadonlyService(sql_settings).list_tables()

    assert output == {
        "dialect": "sqlite",
        "schema": "",
        "count": 2,
        "tables": ["students", "courses"],
        "content_is_untrusted": True,
    }


def test_list_tables_does_not_return_allowlisted_view(settings_factory, sql_business_db):
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students", "student_names"),
    )

    output = SqlReadonlyService(settings).list_tables()

    assert output["tables"] == ["students"]


def test_list_tables_rejects_arbitrary_schema(sql_settings):
    with pytest.raises(SqlReadonlyError) as captured:
        SqlReadonlyService(sql_settings).list_tables(schema="other")

    assert captured.value.code is ToolErrorCode.SQL_TABLE_NOT_ALLOWED


def test_default_schema_applies_to_unqualified_and_explicit_allowlist_items(
    settings_factory,
    sql_business_db,
):
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students", "main.courses"),
        sql_default_schema="main",
    )
    service = SqlReadonlyService(settings)

    listed = service.list_tables()
    described = service.describe_table(table_name="courses")
    qualified_description = service.describe_table(table_name="main.students")
    queried = service.query(sql="SELECT id FROM main.students ORDER BY id LIMIT 1")

    assert listed["schema"] == "main"
    assert listed["tables"] == ["students", "courses"]
    assert described["table_name"] == "main.courses"
    assert qualified_description["table_name"] == "students"
    assert queried["rows"] == [[1]]


def test_describe_returns_only_limited_portable_column_contract(sql_settings):
    output = SqlReadonlyService(sql_settings).describe_table(
        table_name="students"
    )

    assert output["table_name"] == "students"
    assert output["column_count"] == 4
    assert output["content_is_untrusted"] is True
    assert output["columns"][0] == {
        "name": "id",
        "type": "INTEGER",
        "nullable": True,
        "primary_key": True,
    }
    assert set(output["columns"][0]) == {
        "name",
        "type",
        "nullable",
        "primary_key",
    }


@pytest.mark.parametrize("table_name", ["secret_internal", "student_names"])
def test_describe_rejects_unauthorized_table_or_view(
    settings_factory,
    sql_business_db,
    table_name,
):
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students", "student_names"),
    )

    with pytest.raises(SqlReadonlyError) as captured:
        SqlReadonlyService(settings).describe_table(table_name=table_name)

    assert captured.value.code is ToolErrorCode.SQL_TABLE_NOT_ALLOWED


def test_query_returns_columns_rows_and_detects_truncation(sql_settings):
    output = SqlReadonlyService(sql_settings).query(
        sql="SELECT id, name FROM students ORDER BY id"
    )

    assert output["columns"] == ["id", "name"]
    assert output["rows"] == [[1, "张三"], [2, "李四"]]
    assert output["row_count"] == 2
    assert output["truncated"] is True
    assert output["dialect"] == "sqlite"
    assert output["execution_ms"] >= 0
    assert output["content_is_untrusted"] is True


def test_query_preserves_small_limit_and_compresses_large_limit(sql_settings):
    service = SqlReadonlyService(sql_settings)

    small = service.query(sql="SELECT id FROM students ORDER BY id LIMIT 1")
    large = service.query(sql="SELECT id FROM students ORDER BY id LIMIT 999")

    assert small["rows"] == [[1]]
    assert small["truncated"] is False
    assert large["rows"] == [[1], [2]]
    assert large["truncated"] is True


def test_query_handles_empty_null_and_binary_values(sql_settings):
    service = SqlReadonlyService(sql_settings)

    empty = service.query(sql="SELECT id FROM students WHERE id < 0")
    values = service.query(
        sql="SELECT note, payload FROM students WHERE id = 3"
    )
    binary = service.query(sql="SELECT payload FROM students WHERE id = 1")

    assert empty["rows"] == []
    assert empty["truncated"] is False
    assert values["rows"] == [[None, None]]
    assert binary["rows"][0][0].startswith("<binary:3 bytes sha256:")


def test_query_rejects_too_many_columns(settings_factory, sql_business_db):
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students",),
        sql_max_columns=1,
    )

    with pytest.raises(SqlReadonlyError) as captured:
        SqlReadonlyService(settings).query(sql="SELECT id, name FROM students")

    assert captured.value.code is ToolErrorCode.SQL_RESULT_TOO_WIDE


def test_query_truncates_long_cells_with_explicit_marker(
    settings_factory,
    sql_business_db,
):
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students",),
        sql_max_cell_chars=20,
    )

    output = SqlReadonlyService(settings).query(
        sql="SELECT printf('%050d', id) AS long_text FROM students WHERE id = 1"
    )

    assert len(output["rows"][0][0]) == 20
    assert output["rows"][0][0].endswith("...[truncated]")


def test_total_output_limit_stops_only_at_complete_row(
    settings_factory,
    sql_business_db,
):
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students",),
        sql_max_rows=3,
        sql_max_output_chars=230,
    )

    output = SqlReadonlyService(settings).query(
        sql="SELECT printf('%080d', id) AS text FROM students ORDER BY id"
    )

    assert output["truncated"] is True
    assert output["row_count"] == len(output["rows"])
    assert all(len(row) == 1 for row in output["rows"])


def test_query_rejects_requested_rows_above_deployment_limit(sql_settings):
    with pytest.raises(SqlReadonlyError) as captured:
        SqlReadonlyService(sql_settings).query(
            sql="SELECT * FROM students",
            max_rows=3,
        )

    assert captured.value.code is ToolErrorCode.SQL_INVALID_STATEMENT


def test_query_maps_missing_database_to_safe_connection_error(
    settings_factory,
    tmp_path,
):
    missing = tmp_path / "missing.sqlite3"
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{missing}",
        sql_allowed_tables=("students",),
    )

    with pytest.raises(SqlReadonlyError) as captured:
        SqlReadonlyService(settings).query(sql="SELECT * FROM students")

    assert captured.value.code is ToolErrorCode.SQL_CONNECTION_ERROR
    assert str(missing) not in str(captured.value)


def test_sqlite_recursive_query_is_interrupted_by_statement_timeout(
    settings_factory,
    sql_business_db,
):
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students",),
        sql_statement_timeout_ms=1,
        sql_tool_timeout_seconds=2,
    )
    sql = """
        WITH RECURSIVE counter(value) AS (
            SELECT 1
            UNION ALL
            SELECT value + 1 FROM counter WHERE value < 100000000
        )
        SELECT SUM(value) FROM counter
    """

    with pytest.raises(SqlReadonlyError) as captured:
        SqlReadonlyService(settings).query(sql=sql)

    assert captured.value.code is ToolErrorCode.SQL_TIMEOUT

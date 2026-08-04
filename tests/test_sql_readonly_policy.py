"""sqlglot 只读 SQL 策略测试。"""

import pytest

from app.tools.errors import ToolErrorCode
from app.tools.sql_readonly.errors import SqlReadonlyError
from app.tools.sql_readonly.policy import validate_and_normalize_query


@pytest.fixture
def sql_settings(settings_factory, sql_business_db):
    return settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students", "courses"),
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM students",
        "SELECT s.name FROM students AS s JOIN courses AS c ON s.id = c.id",
        "SELECT name, COUNT(*) FROM students GROUP BY name HAVING COUNT(*) > 0",
        "SELECT * FROM students ORDER BY id",
        "SELECT * FROM (SELECT * FROM students) AS selected_students",
        "WITH active AS (SELECT * FROM students) SELECT * FROM active",
        "SELECT id FROM students UNION SELECT id FROM courses",
        "SELECT COUNT(*) FROM students",
        "SELECT 1 AS one",
        'SELECT * FROM "students"',
    ],
)
def test_allows_readonly_query_shapes(sql_settings, sql):
    validated = validate_and_normalize_query(
        sql,
        sql_settings,
        dialect_name="sqlite",
        max_rows=5,
    )

    assert validated.sql
    assert validated.fetch_limit == 6


@pytest.mark.parametrize(
    ("sql", "error_code"),
    [
        ("SELECT FROM", ToolErrorCode.SQL_INVALID_STATEMENT),
        ("SELECT * FROM students; SELECT * FROM courses", ToolErrorCode.SQL_MULTIPLE_STATEMENTS),
        ("INSERT INTO students(id, name) VALUES (10, 'x')", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("UPDATE students SET name = 'x'", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("DELETE FROM students", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("CREATE TABLE extra(id INTEGER)", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("DROP TABLE students", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("ALTER TABLE students ADD COLUMN extra TEXT", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("TRUNCATE TABLE students", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("PRAGMA query_only", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("ATTACH DATABASE '/tmp/x' AS other", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("DETACH DATABASE other", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("VACUUM", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("BEGIN", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("COMMIT", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("ROLLBACK", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        ("SELECT * FROM students FOR UPDATE", ToolErrorCode.SQL_WRITE_FORBIDDEN),
        (
            "WITH changed AS (DELETE FROM students RETURNING *) SELECT * FROM changed",
            ToolErrorCode.SQL_WRITE_FORBIDDEN,
        ),
        ("SELECT * FROM secret_internal", ToolErrorCode.SQL_TABLE_NOT_ALLOWED),
        ("SELECT * FROM sqlite_master", ToolErrorCode.SQL_TABLE_NOT_ALLOWED),
        ("SELECT * FROM main.students", ToolErrorCode.SQL_TABLE_NOT_ALLOWED),
        ('SELECT * FROM "secret_internal"', ToolErrorCode.SQL_TABLE_NOT_ALLOWED),
        ("SELECT sleep(1)", ToolErrorCode.SQL_DANGEROUS_FUNCTION),
        ("SELECT load_extension('/tmp/x')", ToolErrorCode.SQL_DANGEROUS_FUNCTION),
        ("SELECT pg_read_file('/tmp/x')", ToolErrorCode.SQL_DANGEROUS_FUNCTION),
        ("SELECT LOAD_FILE('/tmp/x')", ToolErrorCode.SQL_DANGEROUS_FUNCTION),
        ("SELECT * FROM read_csv('/tmp/x')", ToolErrorCode.SQL_DANGEROUS_FUNCTION),
    ],
)
def test_rejects_unsafe_or_unauthorized_statements(
    sql_settings,
    sql,
    error_code,
):
    with pytest.raises(SqlReadonlyError) as captured:
        validate_and_normalize_query(
            sql,
            sql_settings,
            dialect_name="sqlite",
            max_rows=5,
        )

    assert captured.value.code is error_code


def test_cte_name_is_not_treated_as_physical_table(sql_settings):
    validated = validate_and_normalize_query(
        "WITH active AS (SELECT * FROM students) SELECT * FROM active",
        sql_settings,
        dialect_name="sqlite",
        max_rows=5,
    )

    assert [table.display_name for table in validated.physical_tables] == [
        "students"
    ]


def test_default_schema_maps_qualified_query_to_unqualified_allowlist(
    settings_factory,
    sql_business_db,
):
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students",),
        sql_default_schema="main",
    )

    validated = validate_and_normalize_query(
        "SELECT * FROM main.students",
        settings,
        dialect_name="sqlite",
        max_rows=5,
    )

    assert [table.display_name for table in validated.physical_tables] == [
        "students"
    ]


def test_dangerous_function_name_inside_string_is_not_a_false_positive(
    sql_settings,
):
    validated = validate_and_normalize_query(
        "SELECT * FROM students WHERE note = 'sleep(1)'",
        sql_settings,
        dialect_name="sqlite",
        max_rows=5,
    )

    assert "sleep(1)" in validated.sql


def test_limit_is_added_preserved_or_compressed_without_string_concatenation(
    sql_settings,
):
    added = validate_and_normalize_query(
        "SELECT * FROM students ORDER BY id",
        sql_settings,
        dialect_name="sqlite",
        max_rows=5,
    )
    preserved = validate_and_normalize_query(
        "SELECT * FROM students ORDER BY id LIMIT 2",
        sql_settings,
        dialect_name="sqlite",
        max_rows=5,
    )
    compressed = validate_and_normalize_query(
        "SELECT * FROM students ORDER BY id LIMIT 999",
        sql_settings,
        dialect_name="sqlite",
        max_rows=5,
    )

    assert added.sql.endswith("LIMIT 6")
    assert preserved.sql.endswith("LIMIT 2")
    assert compressed.sql.endswith("LIMIT 6")

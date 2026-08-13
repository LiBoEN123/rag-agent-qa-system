"""SQL 结果标量转换和结构体积限制测试。"""

from datetime import date, datetime, time, timezone
from decimal import Decimal
from uuid import UUID

from app.tools.sql_readonly.serialization import (
    normalize_column_names,
    serialize_query_output,
    serialize_sql_value,
)


def test_supported_database_values_have_deterministic_json_scalars():
    values = [
        None,
        True,
        7,
        2.5,
        "文本",
        Decimal("12.3400"),
        date(2026, 8, 4),
        datetime(2026, 8, 4, 12, 30, tzinfo=timezone.utc),
        time(12, 30, 15),
        UUID("12345678-1234-5678-1234-567812345678"),
        b"\x00\x01",
    ]

    serialized = [serialize_sql_value(value, max_chars=100) for value in values]

    assert serialized[:5] == [None, True, 7, 2.5, "文本"]
    assert serialized[5:10] == [
        "12.3400",
        "2026-08-04",
        "2026-08-04T12:30:00+00:00",
        "12:30:15",
        "12345678-1234-5678-1234-567812345678",
    ]
    assert serialized[10].startswith("<binary:2 bytes sha256:")


def test_unknown_object_uses_finite_safe_string():
    class CustomValue:
        def __str__(self) -> str:
            return "x" * 100

    serialized = serialize_sql_value(CustomValue(), max_chars=20)

    assert serialized == "xxxxxx...[truncated]"


def test_column_names_are_nonempty_unique_and_finite():
    columns = normalize_column_names(
        ["id", "id", "id_2", "", "x" * 500, "x" * 500]
    )

    assert columns[:4] == ["id", "id_2", "id_2_2", "column_4"]
    assert len(columns[4]) == 256
    assert len(columns[5]) == 256
    assert columns[4].endswith("...[truncated]")
    assert columns[5].endswith("...[truncated]_2")
    assert len(set(columns)) == len(columns)


def test_output_limit_never_emits_partial_rows():
    output = serialize_query_output(
        columns=["value"],
        rows=[["a" * 80], ["b" * 80], ["c" * 80]],
        requested_max_rows=3,
        max_cell_chars=100,
        max_output_chars=260,
        dialect="sqlite",
        execution_ms=1,
    )

    assert output["truncated"] is True
    assert output["row_count"] == len(output["rows"])
    assert all(row in (["a" * 80], ["b" * 80]) for row in output["rows"])

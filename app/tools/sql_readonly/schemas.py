"""只读 SQL 工具的 Pydantic 输入与输出合同。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SQL_TEXT_MAX_CHARS = 20_000
SqlCell = str | int | float | bool | None


class ListSqlTablesArgs(BaseModel):
    """列出 allowlist 内真实表的参数。"""

    model_config = ConfigDict(
        extra="forbid",
        serialize_by_alias=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    schema_name: str | None = Field(
        default=None,
        alias="schema",
        max_length=128,
    )

    @field_validator("schema_name", mode="before")
    @classmethod
    def normalize_schema(cls, value: object) -> object:
        """去除可选 Schema 两端空白。"""
        if isinstance(value, str):
            return value.strip() or None
        return value


class DescribeSqlTableArgs(BaseModel):
    """查看单个 allowlist 表有限 Schema 的参数。"""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(..., min_length=1, max_length=257)

    @field_validator("table_name", mode="before")
    @classmethod
    def normalize_table_name(cls, value: object) -> object:
        """去除表名两端空白。"""
        return value.strip() if isinstance(value, str) else value


class QuerySqlReadonlyArgs(BaseModel):
    """执行单条受治理只读查询的参数。"""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(..., min_length=1, max_length=SQL_TEXT_MAX_CHARS)
    max_rows: int | None = Field(default=None, ge=1, le=1000)

    @field_validator("sql", mode="before")
    @classmethod
    def normalize_sql(cls, value: object) -> object:
        """去除 SQL 两端空白，同时保留语句内部内容。"""
        return value.strip() if isinstance(value, str) else value


class SqlColumnDescription(BaseModel):
    """单列的有限、可移植 Schema 描述。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    nullable: bool
    primary_key: bool


class ListSqlTablesOutput(BaseModel):
    """list_sql_tables 稳定输出。"""

    model_config = ConfigDict(
        extra="forbid",
        serialize_by_alias=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    dialect: str
    schema_name: str = Field(alias="schema")
    count: int = Field(..., ge=0)
    tables: list[str]
    content_is_untrusted: Literal[True] = True

    @model_validator(mode="after")
    def validate_count(self) -> "ListSqlTablesOutput":
        """确保表数量与列表一致。"""
        if self.count != len(self.tables):
            raise ValueError("count 必须等于 tables 数量")
        return self


class DescribeSqlTableOutput(BaseModel):
    """describe_sql_table 稳定输出。"""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    columns: list[SqlColumnDescription]
    column_count: int = Field(..., ge=0)
    content_is_untrusted: Literal[True] = True

    @model_validator(mode="after")
    def validate_column_count(self) -> "DescribeSqlTableOutput":
        """确保列数量与列表一致。"""
        if self.column_count != len(self.columns):
            raise ValueError("column_count 必须等于 columns 数量")
        return self


class QuerySqlReadonlyOutput(BaseModel):
    """query_sql_readonly 稳定、有限的二维结果。"""

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    rows: list[list[SqlCell]]
    row_count: int = Field(..., ge=0)
    truncated: bool
    dialect: str
    execution_ms: int = Field(..., ge=0)
    content_is_untrusted: Literal[True] = True

    @model_validator(mode="after")
    def validate_rectangular_rows(self) -> "QuerySqlReadonlyOutput":
        """确保二维行宽和公开计数与 columns 一致。"""
        if self.row_count != len(self.rows):
            raise ValueError("row_count 必须等于 rows 数量")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("每一行的单元格数量必须等于 columns 数量")
        return self


__all__ = [
    "DescribeSqlTableArgs",
    "DescribeSqlTableOutput",
    "ListSqlTablesArgs",
    "ListSqlTablesOutput",
    "QuerySqlReadonlyArgs",
    "QuerySqlReadonlyOutput",
    "SQL_TEXT_MAX_CHARS",
    "SqlColumnDescription",
]

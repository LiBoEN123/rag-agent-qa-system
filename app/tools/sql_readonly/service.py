"""只读 SQL 工具服务：Inspector、事务边界、执行与结果转换。"""

from __future__ import annotations

import time
from collections.abc import Mapping

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.config import AppSettings, validate_sql_allowed_table
from app.tools.errors import ToolErrorCode
from app.tools.sql_readonly.engine import get_sql_engine, statement_timeout
from app.tools.sql_readonly.errors import SqlReadonlyError
from app.tools.sql_readonly.policy import (
    AllowedTable,
    normalized_table_key,
    parse_allowed_tables,
    resolve_allowed_table,
    validate_and_normalize_query,
)
from app.tools.sql_readonly.serialization import (
    ensure_sql_output_size,
    serialize_query_output,
)


class SqlReadonlyService:
    """持有不可变配置快照，按调用惰性使用缓存 Engine。"""

    # 保存只读 SQL 服务使用的应用配置。
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def list_tables(self, *, schema: str | None = None) -> dict[str, object]:
        """只列出请求 Schema 内 allowlist 中确实存在的普通表。"""
        engine = get_sql_engine(self._settings)
        dialect_name = engine.dialect.name
        selected_schema = self._resolve_schema(schema, dialect_name=dialect_name)
        connected = False
        timeout_state = None
        try:
            with engine.connect() as connection:
                connected = True
                with statement_timeout(
                    connection,
                    self._settings.sql_statement_timeout_ms,
                ) as timeout_state:
                    inspector = inspect(connection)
                    actual_names = inspector.get_table_names(
                        schema=selected_schema or None
                    )
        except SQLAlchemyError as exc:
            self._raise_database_error(
                exc,
                timed_out=bool(timeout_state and timeout_state.timed_out),
                connection_established=connected,
            )

        actual_by_key = {
            normalized_table_key(
                selected_schema,
                name,
                dialect_name=dialect_name,
            )[1]: name
            for name in actual_names
            if not name.casefold().startswith("sqlite_")
        }
        tables: list[str] = []
        normalized_default = self._normalized_schema(
            self._settings.sql_default_schema,
            dialect_name=dialect_name,
        )
        normalized_selected = self._normalized_schema(
            selected_schema,
            dialect_name=dialect_name,
        )
        for item in parse_allowed_tables(
            self._settings,
            dialect_name=dialect_name,
        ):
            effective_schema = item.schema or normalized_default
            if effective_schema != normalized_selected:
                continue
            actual = actual_by_key.get(item.name)
            if actual is not None:
                tables.append(actual)
        output = {
            "dialect": dialect_name,
            "schema": selected_schema or "",
            "count": len(tables),
            "tables": tables,
            "content_is_untrusted": True,
        }
        ensure_sql_output_size(
            output,
            max_output_chars=self._settings.sql_max_output_chars,
        )
        return output

    def describe_table(self, *, table_name: str) -> dict[str, object]:
        """返回 allowlist 普通表的有限列信息，不暴露默认值或建表 SQL。"""
        engine = get_sql_engine(self._settings)
        dialect_name = engine.dialect.name
        allowed = resolve_allowed_table(
            table_name,
            self._settings,
            dialect_name=dialect_name,
        )
        schema, _name = self._display_parts(allowed)
        schema = schema or self._settings.sql_default_schema
        connected = False
        timeout_state = None
        try:
            with engine.connect() as connection:
                connected = True
                with statement_timeout(
                    connection,
                    self._settings.sql_statement_timeout_ms,
                ) as timeout_state:
                    inspector = inspect(connection)
                    actual_name = self._existing_table_name(
                        inspector.get_table_names(schema=schema),
                        allowed,
                        dialect_name=dialect_name,
                    )
                    if actual_name is None:
                        raise SqlReadonlyError(
                            ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
                            "请求的表不存在或不是允许访问的物理表",
                        )
                    raw_columns = inspector.get_columns(
                        actual_name,
                        schema=schema,
                    )
                    primary_key = inspector.get_pk_constraint(
                        actual_name,
                        schema=schema,
                    ) or {}
        except SqlReadonlyError:
            raise
        except SQLAlchemyError as exc:
            self._raise_database_error(
                exc,
                timed_out=bool(timeout_state and timeout_state.timed_out),
                connection_established=connected,
            )

        if len(raw_columns) > self._settings.sql_max_columns:
            raise SqlReadonlyError(
                ToolErrorCode.SQL_RESULT_TOO_WIDE,
                "表的列数超过允许返回的上限",
            )
        constrained = {
            normalized_table_key(
                None,
                str(value),
                dialect_name=dialect_name,
            )[1]
            for value in (primary_key.get("constrained_columns") or [])
        }
        columns = [
            self._column_description(
                column,
                primary_key_names=constrained,
                dialect_name=dialect_name,
            )
            for column in raw_columns
        ]
        output = {
            "table_name": allowed.display_name,
            "columns": columns,
            "column_count": len(columns),
            "content_is_untrusted": True,
        }
        ensure_sql_output_size(
            output,
            max_output_chars=self._settings.sql_max_output_chars,
        )
        return output

    def query(self, *, sql: str, max_rows: int | None = None) -> dict[str, object]:
        """执行经过 sqlglot 校验和 LIMIT 规范化的单条只读查询。"""
        requested_max_rows = max_rows or self._settings.sql_max_rows
        if requested_max_rows > self._settings.sql_max_rows:
            raise SqlReadonlyError(
                ToolErrorCode.SQL_INVALID_STATEMENT,
                "max_rows 超过部署配置允许的上限",
            )
        engine = get_sql_engine(self._settings)
        dialect_name = engine.dialect.name
        validated = validate_and_normalize_query(
            sql,
            self._settings,
            dialect_name=dialect_name,
            max_rows=requested_max_rows,
        )

        started_at = time.perf_counter()
        connected = False
        timeout_state = None
        try:
            with engine.connect() as connection:
                connected = True
                with statement_timeout(
                    connection,
                    self._settings.sql_statement_timeout_ms,
                ) as timeout_state:
                    inspector = inspect(connection)
                    self._verify_physical_tables(
                        inspector,
                        validated.physical_tables,
                        dialect_name=dialect_name,
                    )
                    result = connection.execute(text(validated.sql))
                    columns = list(result.keys())
                    if len(columns) > self._settings.sql_max_columns:
                        raise SqlReadonlyError(
                            ToolErrorCode.SQL_RESULT_TOO_WIDE,
                            "查询返回列数超过上限",
                        )
                    rows = result.fetchmany(validated.fetch_limit)
        except SqlReadonlyError:
            raise
        except SQLAlchemyError as exc:
            self._raise_database_error(
                exc,
                timed_out=bool(timeout_state and timeout_state.timed_out),
                connection_established=connected,
            )

        execution_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        return serialize_query_output(
            columns=columns,
            rows=rows,
            requested_max_rows=requested_max_rows,
            max_cell_chars=self._settings.sql_max_cell_chars,
            max_output_chars=self._settings.sql_max_output_chars,
            dialect=dialect_name,
            execution_ms=execution_ms,
        )

    def _resolve_schema(
        self,
        schema: str | None,
        *,
        dialect_name: str,
    ) -> str | None:
        """只允许默认 Schema 或 allowlist 明示的 Schema。"""
        requested = schema or self._settings.sql_default_schema
        if requested is None:
            return None
        try:
            safe_schema = validate_sql_allowed_table(requested)
        except ValueError as exc:
            raise SqlReadonlyError(
                ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
                "请求的 Schema 不在允许访问范围内",
            ) from exc
        if "." in safe_schema:
            raise SqlReadonlyError(
                ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
                "请求的 Schema 不在允许访问范围内",
            )
        normalized = self._normalized_schema(
            safe_schema,
            dialect_name=dialect_name,
        )
        allowed_schemas = {
            item.schema
            for item in parse_allowed_tables(
                self._settings,
                dialect_name=dialect_name,
            )
            if item.schema is not None
        }
        default_schema = self._normalized_schema(
            self._settings.sql_default_schema,
            dialect_name=dialect_name,
        )
        if normalized != default_schema and normalized not in allowed_schemas:
            raise SqlReadonlyError(
                ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
                "请求的 Schema 不在允许访问范围内",
            )
        return safe_schema

    # 按数据库方言规范化非空的数据库模式名称。
    @staticmethod
    def _normalized_schema(
        schema: str | None,
        *,
        dialect_name: str,
    ) -> str | None:
        if schema is None:
            return None
        return normalized_table_key(
            schema,
            "placeholder",
            dialect_name=dialect_name,
        )[0]

    # 将允许访问表的显示名称拆分为可选模式名和表名。
    @staticmethod
    def _display_parts(table: AllowedTable) -> tuple[str | None, str]:
        parts = table.display_name.split(".")
        return (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])

    # 按方言规则匹配实际表名与白名单表名，并跳过 SQLite 内部表。
    @staticmethod
    def _existing_table_name(
        actual_names: list[str],
        allowed: AllowedTable,
        *,
        dialect_name: str,
    ) -> str | None:
        for name in actual_names:
            if name.casefold().startswith("sqlite_"):
                continue
            if normalized_table_key(
                None,
                name,
                dialect_name=dialect_name,
            )[1] == allowed.name:
                return name
        return None

    def _verify_physical_tables(
        self,
        inspector,
        tables: tuple[AllowedTable, ...],
        *,
        dialect_name: str,
    ) -> None:
        """拒绝不存在的表、View、系统表和临时表。"""
        by_schema: dict[str | None, list[AllowedTable]] = {}
        for table in tables:
            schema, _name = self._display_parts(table)
            schema = schema or self._settings.sql_default_schema
            by_schema.setdefault(schema, []).append(table)
        for schema, schema_tables in by_schema.items():
            actual_names = inspector.get_table_names(schema=schema)
            for table in schema_tables:
                if self._existing_table_name(
                    actual_names,
                    table,
                    dialect_name=dialect_name,
                ) is None:
                    raise SqlReadonlyError(
                        ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
                        "查询引用了不存在、临时或非物理表",
                    )

    # 整理列名、类型和可空信息，并按方言规范化名称后判断是否为主键。
    @staticmethod
    def _column_description(
        column: Mapping[str, object],
        *,
        primary_key_names: set[str],
        dialect_name: str,
    ) -> dict[str, object]:
        name = str(column.get("name") or "")
        type_name = str(column.get("type") or "UNKNOWN")[:256]
        normalized_name = normalized_table_key(
            None,
            name,
            dialect_name=dialect_name,
        )[1]
        return {
            "name": name,
            "type": type_name,
            "nullable": bool(column.get("nullable", True)),
            "primary_key": normalized_name in primary_key_names,
        }

    # 根据超时和连接状态，将数据库异常转换为固定消息的只读 SQL 错误。
    @staticmethod
    def _raise_database_error(
        exc: SQLAlchemyError,
        *,
        timed_out: bool,
        connection_established: bool = True,
    ) -> None:
        if timed_out:
            raise SqlReadonlyError(
                ToolErrorCode.SQL_TIMEOUT,
                "只读 SQL 查询超过数据库语句超时",
            ) from exc
        if not connection_established:
            raise SqlReadonlyError(
                ToolErrorCode.SQL_CONNECTION_ERROR,
                "无法连接只读数据库",
            ) from exc
        raise SqlReadonlyError(
            ToolErrorCode.SQL_EXECUTION_ERROR,
            "只读 SQL 查询执行失败",
        ) from exc


__all__ = ["SqlReadonlyService"]

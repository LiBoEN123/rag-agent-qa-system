"""基于 sqlglot AST 的单语句只读 SQL 策略。"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import traverse_scope

from app.config import AppSettings, validate_sql_allowed_table
from app.tools.errors import ToolErrorCode
from app.tools.sql_readonly.errors import SqlReadonlyError


DANGEROUS_FUNCTIONS = frozenset(
    {
        "benchmark",
        "dblink",
        "dblink_connect",
        "dblink_connect_u",
        "dblink_disconnect",
        "dblink_exec",
        "dblink_get_notify",
        "dblink_open",
        "fts3_tokenizer",
        "http_get",
        "load_extension",
        "load_file",
        "lo_export",
        "lo_import",
        "mysql_scan",
        "nextval",
        "pg_read_binary_file",
        "pg_read_file",
        "pg_ls_dir",
        "pg_ls_logdir",
        "pg_ls_waldir",
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_stat_file",
        "postgres_scan",
        "read_csv",
        "read_csv_auto",
        "read_json",
        "read_parquet",
        "readfile",
        "setval",
        "sleep",
        "sqlite_scan",
        "sys_eval",
        "sys_exec",
        "writefile",
        "xp_cmdshell",
    }
)

_FORBIDDEN_NODE_NAMES = (
    "Alter",
    "Analyze",
    "Attach",
    "Cache",
    "Command",
    "Commit",
    "Copy",
    "Create",
    "Declare",
    "Delete",
    "Detach",
    "Drop",
    "Execute",
    "Fetch",
    "Grant",
    "Insert",
    "Into",
    "Kill",
    "LoadData",
    "Lock",
    "Merge",
    "Pragma",
    "Revoke",
    "Rollback",
    "Set",
    "Transaction",
    "TruncateTable",
    "Uncache",
    "Update",
    "Use",
)
_FORBIDDEN_NODE_TYPES = tuple(
    node_type
    for node_type in (
        exp.DDL,
        exp.DML,
        *(getattr(exp, name, None) for name in _FORBIDDEN_NODE_NAMES),
    )
    if isinstance(node_type, type)
)


@dataclass(frozen=True, slots=True)
class AllowedTable:
    """规范化后的单个 allowlist 表。"""

    schema: str | None
    name: str
    display_name: str


@dataclass(frozen=True, slots=True)
class ValidatedQuery:
    """通过 AST 策略后可交给 SQLAlchemy 的查询。"""

    sql: str
    physical_tables: tuple[AllowedTable, ...]
    fetch_limit: int


def sqlglot_dialect(dialect_name: str) -> str:
    """把 SQLAlchemy Dialect 名称映射为 sqlglot 正式 Dialect 名称。"""
    aliases = {
        "mariadb": "mysql",
        "postgresql": "postgres",
    }
    return aliases.get(dialect_name, dialect_name)


def _normalize_identifier(
    name: str,
    *,
    dialect_name: str,
    quoted: bool = False,
) -> str:
    """按第一版支持的 Dialect 标识符规则生成比较键。"""
    if dialect_name in {"sqlite", "mysql", "mariadb"}:
        return name.casefold()
    if dialect_name in {"postgres", "postgresql"} and not quoted:
        return name.casefold()
    return name


def parse_allowed_tables(
    settings: AppSettings,
    *,
    dialect_name: str,
) -> tuple[AllowedTable, ...]:
    """把 Settings 中已校验的 allowlist 转成 Dialect 比较形式。"""
    parsed: list[AllowedTable] = []
    seen: set[tuple[str | None, str]] = set()
    for value in settings.sql_allowed_tables:
        parts = value.split(".")
        schema = parts[0] if len(parts) == 2 else None
        name = parts[-1]
        item = AllowedTable(
            schema=(
                _normalize_identifier(schema, dialect_name=dialect_name)
                if schema is not None
                else None
            ),
            name=_normalize_identifier(name, dialect_name=dialect_name),
            display_name=value,
        )
        key = (item.schema, item.name)
        if key not in seen:
            seen.add(key)
            parsed.append(item)
    return tuple(parsed)


def normalized_table_key(
    schema: str | None,
    name: str,
    *,
    dialect_name: str,
) -> tuple[str | None, str]:
    """为 Inspector 返回的物理表名生成与 allowlist 相同的比较键。"""
    return (
        (
            _normalize_identifier(schema, dialect_name=dialect_name)
            if schema is not None
            else None
        ),
        _normalize_identifier(name, dialect_name=dialect_name),
    )


def resolve_allowed_table(
    table_name: str,
    settings: AppSettings,
    *,
    dialect_name: str,
) -> AllowedTable:
    """把模型请求的表名解析为 allowlist 中的唯一项。"""
    try:
        safe_name = validate_sql_allowed_table(table_name)
    except ValueError as exc:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
            "请求的表不在允许访问范围内",
        ) from exc
    parts = safe_name.split(".")
    requested_schema = parts[0] if len(parts) == 2 else None
    requested_name = parts[-1]
    requested_key = (
        (
            _normalize_identifier(
                requested_schema,
                dialect_name=dialect_name,
            )
            if requested_schema is not None
            else None
        ),
        _normalize_identifier(requested_name, dialect_name=dialect_name),
    )
    allowed = parse_allowed_tables(settings, dialect_name=dialect_name)
    for item in allowed:
        if (item.schema, item.name) == requested_key:
            return item
    if requested_key[0] is not None and settings.sql_default_schema:
        normalized_default = _normalize_identifier(
            settings.sql_default_schema,
            dialect_name=dialect_name,
        )
        if requested_key[0] == normalized_default:
            for item in allowed:
                if item.schema is None and item.name == requested_key[1]:
                    return item
    if requested_key[0] is None and settings.sql_default_schema:
        default_key = (
            _normalize_identifier(
                settings.sql_default_schema,
                dialect_name=dialect_name,
            ),
            requested_key[1],
        )
        for item in allowed:
            if (item.schema, item.name) == default_key:
                return item
    raise SqlReadonlyError(
        ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
        "请求的表不在允许访问范围内",
    )


def _function_name(function: exp.Func) -> str:
    """从函数 AST 节点提取大小写无关名称，不检查普通字符串。"""
    if isinstance(function, exp.Anonymous):
        return function.name.casefold()
    return function.sql_name().casefold()


def _physical_tables(expression: exp.Query) -> tuple[exp.Table, ...]:
    """使用 sqlglot Scope 排除 CTE 和子查询别名，只返回物理源。"""
    tables: list[exp.Table] = []
    seen: set[int] = set()
    for scope in traverse_scope(expression):
        for _alias, (_node, source) in scope.selected_sources.items():
            if isinstance(source, exp.Table) and id(source) not in seen:
                seen.add(id(source))
                tables.append(source)
    return tuple(tables)


def _table_key(
    table: exp.Table,
    *,
    dialect_name: str,
) -> tuple[str | None, str]:
    """读取普通物理表标识符；拒绝 catalog 与表值/外部函数。"""
    identifier = table.this
    if not isinstance(identifier, exp.Identifier) or table.catalog:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
            "查询包含不允许的表来源",
        )
    schema_identifier = table.args.get("db")
    if schema_identifier is not None and not isinstance(
        schema_identifier, exp.Identifier
    ):
        raise SqlReadonlyError(
            ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
            "查询包含不允许的 Schema 来源",
        )
    schema = (
        _normalize_identifier(
            schema_identifier.name,
            dialect_name=dialect_name,
            quoted=bool(schema_identifier.args.get("quoted")),
        )
        if schema_identifier is not None
        else None
    )
    name = _normalize_identifier(
        identifier.name,
        dialect_name=dialect_name,
        quoted=bool(identifier.args.get("quoted")),
    )
    return schema, name


def validate_and_normalize_query(
    sql: str,
    settings: AppSettings,
    *,
    dialect_name: str,
    max_rows: int,
) -> ValidatedQuery:
    """执行单语句、只读 AST、函数、表 allowlist 与 LIMIT 检查。"""
    dialect = sqlglot_dialect(dialect_name)
    try:
        parsed = [
            statement
            for statement in sqlglot.parse(sql, read=dialect)
            if statement is not None and not isinstance(statement, exp.Semicolon)
        ]
    except (ParseError, ValueError) as exc:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_INVALID_STATEMENT,
            "SQL 语法无法解析",
        ) from exc
    if len(parsed) != 1:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_MULTIPLE_STATEMENTS,
            "只允许执行单条只读查询",
        )
    expression = parsed[0]
    if not isinstance(expression, exp.Query) or expression.find(exp.Select) is None:
        raise SqlReadonlyError(
            ToolErrorCode.SQL_WRITE_FORBIDDEN,
            "只允许执行 SELECT 或 WITH 查询",
        )
    if any(isinstance(node, _FORBIDDEN_NODE_TYPES) for node in expression.walk()):
        raise SqlReadonlyError(
            ToolErrorCode.SQL_WRITE_FORBIDDEN,
            "查询包含写入、管理、事务或锁定操作",
        )
    for function in expression.find_all(exp.Func):
        if _function_name(function) in DANGEROUS_FUNCTIONS:
            raise SqlReadonlyError(
                ToolErrorCode.SQL_DANGEROUS_FUNCTION,
                "查询包含不允许的数据库函数",
            )

    allowed = {
        (item.schema, item.name): item
        for item in parse_allowed_tables(settings, dialect_name=dialect_name)
    }
    physical: list[AllowedTable] = []
    seen_tables: set[tuple[str | None, str]] = set()
    for table in _physical_tables(expression):
        key = _table_key(table, dialect_name=dialect_name)
        item = allowed.get(key)
        if item is None and key[0] is not None and settings.sql_default_schema:
            normalized_default = _normalize_identifier(
                settings.sql_default_schema,
                dialect_name=dialect_name,
            )
            if key[0] == normalized_default:
                item = allowed.get((None, key[1]))
        if item is None and key[0] is None and settings.sql_default_schema:
            default_key = (
                _normalize_identifier(
                    settings.sql_default_schema,
                    dialect_name=dialect_name,
                ),
                key[1],
            )
            item = allowed.get(default_key)
        if item is None:
            raise SqlReadonlyError(
                ToolErrorCode.SQL_TABLE_NOT_ALLOWED,
                "查询引用了未授权表",
            )
        item_key = (item.schema, item.name)
        if item_key not in seen_tables:
            seen_tables.add(item_key)
            physical.append(item)

    fetch_limit = max_rows + 1
    limit = expression.args.get("limit")
    limit_value: int | None = None
    if isinstance(limit, exp.Limit) and isinstance(limit.expression, exp.Literal):
        try:
            limit_value = int(limit.expression.this)
        except (TypeError, ValueError):
            limit_value = None
    if limit_value is None or limit_value < 0 or limit_value > fetch_limit:
        expression = expression.limit(fetch_limit, copy=True)
    normalized_sql = expression.sql(dialect=dialect)
    return ValidatedQuery(
        sql=normalized_sql,
        physical_tables=tuple(physical),
        fetch_limit=fetch_limit,
    )


__all__ = [
    "AllowedTable",
    "DANGEROUS_FUNCTIONS",
    "ValidatedQuery",
    "parse_allowed_tables",
    "normalized_table_key",
    "resolve_allowed_table",
    "sqlglot_dialect",
    "validate_and_normalize_query",
]

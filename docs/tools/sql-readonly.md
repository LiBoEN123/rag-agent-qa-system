# 受治理的只读 SQL 工具

## 能力与边界

该能力默认关闭，只面向部署者显式配置的业务数据源。启用后，Agent 可通过现有
`ToolRegistry` / `ToolRuntime` 使用三个工具：

| 工具 | 作用 | 输出边界 |
| --- | --- | --- |
| `list_sql_tables` | 列出指定允许 Schema 中，allowlist 内确实存在的普通物理表 | 不返回系统表、View、未授权表或连接信息 |
| `describe_sql_table` | 查看一个 allowlist 表的列名、可移植类型、可空性和主键标记 | 不返回默认表达式、触发器、索引 SQL 或建表 SQL |
| `query_sql_readonly` | 执行一条经 AST 验证并限制结果大小的 `SELECT` / `WITH` 查询 | 统一返回 `columns` + `rows` 二维结构 |

不支持 `INSERT`、`UPDATE`、`DELETE`、DDL、事务/管理语句、自动建表、数据库迁移或
审批式写操作。自然语言不能绕过 Agent 直接执行 SQL；模型必须选择公开的 Tool
Schema，调用仍经过 Registry/Runtime。第一版正式验证 SQLite；未默认安装
MySQL/PostgreSQL 驱动，未来使用这些数据库时必须另行安装匹配驱动并配置数据库侧
只读账号。

数据库返回的所有文本均是不可信数据，三个输出都带
`content_is_untrusted=true`。字段中的提示、命令或权限请求不能覆盖 system、
developer、用户问题或工具政策，也不能触发额外工具、秘密泄露或权限变化。

实现固定使用 `SQLAlchemy==2.0.51` 管理连接、连接池、Dialect、Inspector 和执行，
使用 `sqlglot==30.14.0` 解析/遍历/改写 SQL AST，输入输出继续由 Pydantic 校验。
没有引入 LangChain `SQLDatabaseToolkit` 或 `langchain-community`：项目已有唯一
ToolRegistry/ToolRuntime，额外 Toolkit 会形成第二套工具注册和执行治理边界。

## 启用配置

以下示例不含真实凭据：

```env
SQL_TOOL_ENABLED=true
SQL_DATABASE_URL=sqlite:////srv/rag-agent/business.sqlite3
SQL_ALLOWED_TABLES=students,courses
SQL_DEFAULT_SCHEMA=
SQL_MAX_ROWS=50
SQL_MAX_COLUMNS=50
SQL_MAX_CELL_CHARS=1000
SQL_MAX_OUTPUT_CHARS=50000
SQL_STATEMENT_TIMEOUT_MS=5000
SQL_TOOL_TIMEOUT_SECONDS=10
TOOL_DEFAULT_PERMISSIONS=knowledge_base:read,database:read
```

`SQL_TOOL_ENABLED=false` 时不会创建 Engine、连接数据库或注册 SQL 工具，默认八个
工具和 Evaluation Baseline 合同保持不变。启用时 `SQL_DATABASE_URL` 与非空
`SQL_ALLOWED_TABLES` 都是必填项；启用本身不会自动添加 `database:read`，没有该
请求权限时模型仍看不到三个工具，直接分发也会被 ToolRuntime 拒绝。

`SQL_ALLOWED_TABLES` 只接受 `table` 或 `schema.table` 安全标识符。它不支持空项、
通配符、正则、临时表或自动枚举授权。`list_sql_tables --schema` 只能使用
`SQL_DEFAULT_SCHEMA` 或 allowlist 中明示的 Schema。View 即使误写入 allowlist 也
不会被列出、描述或查询。

示例数据库不能是 `./data/app.sqlite` 或 Tool Audit 数据库，也不能位于项目 RAG
索引清单目录、Chroma
持久化目录或 `outputs/` 评测目录。配置构造会解析、展开并 `resolve()`
SQLite 文件路径后拒绝这些内部目标；没有绕过开关。完整数据库 URL 是
`SecretStr`，不得进入日志、
ToolResult、Tool Audit、metrics、bad case、Agent trace 或 SSE。

## 安全边界

执行采用多层边界，而不是把文本前缀当作 SQL 安全判断：

1. SQLAlchemy 2.x 负责 URL/Dialect、连接池、Inspector、连接与参数执行；Engine 在
   第一次工具调用时惰性创建，使用小型池、`pool_pre_ping`、关闭 SQL echo 和参数
   隐藏，并在应用 shutdown 时统一 `dispose()`。
2. sqlglot 按当前 Dialect 解析单条语句、遍历 AST 与 Scope；只允许查询 AST，拒绝
   写入、DDL、管理、事务、锁、隐藏在 CTE/子查询中的写操作和多语句。
3. 每个物理表引用都必须命中 allowlist；CTE 名和子查询别名不会被误判为物理表，
   schema qualification、quoted identifier、View 和表函数不能绕过检查。
4. 小型显式 denylist 拒绝 `sleep`、`benchmark`、`load_file`、`readfile`、
   `pg_read_file`、`dblink`、`xp_cmdshell` 等时间、文件或外部调用函数。
5. sqlglot 在 AST 上增加或压缩外层 `LIMIT`，内部最多获取 `max_rows + 1` 行判断
   `truncated`，不会用字符串拼接改写查询。
6. SQLite URL 被重建为 `mode=ro` URI，每个连接同时执行 `PRAGMA query_only=ON`；
   progress handler 对 execute/fetch 全阶段实施语句超时。ToolRuntime 另有总体超时。
7. 返回列数、行数、单元格字符和整体 JSON 字符分别受限。总预算不足时只移除完整
   行，绝不输出半行；过宽结果直接拒绝。

AST 检查不能替代数据库权限。SQLite 由文件级 `mode=ro` + `query_only` 提供数据库
层防线；其他数据库即使未来安装驱动，也必须使用由数据库管理员授予的只读账号。

## 执行链与同步/流式一致性

```text
LangChain create_agent
→ 请求级 governed StructuredTool
→ execute_registered_tool
→ ToolRuntime.execute
→ SQL service
→ sqlglot / SQLAlchemy
→ ToolResult / ToolMessage
→ Agent
```

`/chat`、`/chat/stream`、`ChatService` 和 stream projector 中没有 SQL 特判，因此两种
入口自动共享动态可见性、Pydantic 参数校验、权限、超时、审计、错误治理和 Tool
Trace。SSE `tool` 事件仍只公开 `tool_name`、`phase`、`status` 和完成时的
`cost_ms`；SQL 文本、行数据和 Schema 不进入 SSE tool 事件。

## 手动冒烟

确认应用配置使用只读业务数据库，且默认主体已显式拥有 `database:read` 后运行：

```bash
python scripts/smoke_test_sql_readonly.py list
python scripts/smoke_test_sql_readonly.py describe --table students
python scripts/smoke_test_sql_readonly.py query \
  --sql "SELECT id, name FROM students ORDER BY id LIMIT 5"
```

脚本刻意经 `ToolRegistry/ToolRuntime` 分发，不直接调用 Service，不显示 URL/密码，且
查询行只打印有限预览。它可能连接真实配置的数据源，只供人工运行，不属于 CI。

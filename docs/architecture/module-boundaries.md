# 模块依赖边界

本页描述由仓库根目录 `.importlinter` 执行的生产模块依赖合同。人工评审负责判断
职责是否合理，Import Linter 2.13 负责对完整 `app` import graph 做确定性验证。

## 总体方向

```text
app.api
  → app.services
  → app.graph
  → app.agent
  → app.tools
  → app.rag
  → app.llm
  → app.storage / app.observability
  → app.contracts / app.config / app.common
```

箭头表示允许依赖的方向。高层可以直接依赖任意更低层；低层不能直接或间接依赖
更高层。`app.storage : app.observability` 暂时属于同一层，允许保留必要的相互
依赖。`app.contracts | app.config | app.common` 是三个彼此独立的基础模块。

`app.main` 是只导出 ASGI app 的进程入口，允许依赖 API；当前并不直接导入
FastAPI。`app.rebuild_kb` 是维护命令入口，负责组装 config 与 RAG。`app.prompts`
是无应用层 import 的共享常量模块。这些单文件入口不用于规避 package 合同，仍受
顶层无环检查覆盖。

## Package 职责

| Package | 主要职责 | 允许依赖 | 禁止依赖 |
| --- | --- | --- | --- |
| `app.api` | FastAPI app、HTTP Schema/middleware、Uvicorn 适配、lifespan、路由和依赖组装 | 所有下层 | 无低层限制；它是组合根 |
| `app.services` | Chat 应用用例，编排 Graph、Result、Session、Metrics、Query Rewrite | graph 及所有更低层 | api、FastAPI、Starlette |
| `app.graph` | LangGraph 父图、状态、节点、Runtime、Checkpointer 组装 | agent、tools、rag 及更低层 | api、services |
| `app.agent` | Agent 业务、框架适配、治理事件和结果转换 | tools、rag、llm 及基础设施/基础层 | api、services、graph |
| `app.tools` | ToolSpec、Provider、Registry、Runtime 和具体工具 | rag、llm、storage 及基础层 | api、services、graph、agent |
| `app.rag` | 索引、检索、重排、生成、引用和 RAG 子图 | llm、storage、observability 及基础层 | api、services、graph、agent、tools |
| `app.llm` | 模型传输、工厂、错误和模型 Schema | contracts、config、common | rag、tools、agent、graph、services、api |
| `app.storage` | SQLite、session、trace 持久化 | observability 及基础层 | 业务层 |
| `app.observability` | 框架无关 metrics、trace ID 和日志数据处理 | storage 及基础层 | 业务层、FastAPI、Starlette |
| `app.contracts` | 跨层稳定值对象和状态策略 | 无 | config、common 及所有应用/基础设施层 |
| `app.config` | 唯一 AppSettings、配置 helper、校验和惰性缓存 | 无内部 package | contracts、common 及所有上层 |
| `app.common` | 框架无关纯 helper 与通用日志构造 | 无 | config、contracts 及所有上层 |

## API 组合根

API 负责把 HTTP 请求转换为框架无关的 `ChatServiceRequest`，组装 RAG、Graph、
Tools、Diagnostics 和 Response Builder，再把 `ChatServiceOutcome` 映射回 HTTP。
因此 `app.api` 可以跨层直接组装依赖，不要求通过无价值包装器逐层转发。

限制只沿反方向生效：ChatService、Graph、Agent、Tools、RAG、LLM、Storage、
Observability 和基础层都不能导入 API。ChatService 继续接收框架无关 Request 并
返回框架无关 Outcome。

## HTTP 框架边界

FastAPI 与 Starlette 只允许出现在 `app.api`；`app.main` 也被允许作为 ASGI 入口，
但当前只 re-export `app.api.chat.app`。

- `app/api/middleware.py` 拥有 `/chat` trace ID、`request_enter_time`、
  `request_enter_perf` 和 `X-Trace-Id`/`X-Total-Cost-Ms`。
- `app/api/http_logging.py` 拥有 Uvicorn access-log URL 解码与控制字符转义。
- lifespan 显式、幂等调用 `configure_uvicorn_access_log()`。
- `app.common.logging` 只提供 `get_logger()`；导入它不会修改全局 Uvicorn logger。
- `app.observability` 只保留 metrics、trace ID 和框架无关数据处理。

HTTP 合同使用 Import Linter 的 external package 支持，禁止所有低层到
`fastapi`/`starlette` 的直接或间接路径。

## Settings 框架边界

`pydantic_settings` 只能由 `app.config` 直接导入。其他模块消费冻结的
`AppSettings`、惰性 `get_settings()` 或明确 config helper，不创建第二个
`BaseSettings`。

Forbidden contract 默认也检查间接链，但
`业务模块 → app.config → pydantic_settings` 正是合法封装。因此该合同使用
Import Linter 2.13 的 `allow_indirect_imports = True`，只禁止 source package
直接导入外部包；这不是 ignore，也不会隐藏经过其他应用 package 的反向依赖。

## 合同清单

`.importlinter` 声明五项合同：

1. `application-layers`：固化总体单向层级；
2. `top-level-acyclic`：禁止 `app` 顶层 package 形成循环；
3. `foundation-does-not-depend-on-application`：为基础层反向依赖提供明确错误；
4. `http-framework-only-in-api`：限制 FastAPI/Starlette；
5. `pydantic-settings-only-in-config`：限制 pydantic-settings 的直接导入。

合同不包含 `ignore_imports`。函数内 import 和包级 re-export 都会进入 Import
Linter 的图，不是规避手段；也不得用 `importlib`、Service Locator 或动态 Registry
隐藏依赖。

## 执行方式

```bash
lint-imports
lint-imports --verbose
python scripts/inspect_agent_runtime.py --check
```

`lint-imports` 是依赖图判定者。Runtime inspection 只解析 `.importlinter` 的公开
配置，确认文件、关键合同和零 ignore；它不重新实现 import graph。CI 在 Ruff
之后、Runtime inspection 之前运行 Import Linter，任何破坏合同的 import 都会使
任务失败。

## 新增模块如何选择位置

1. 先按责任选择最低的正确层，而不是按现有调用方位置选择。
2. HTTP Request/Response、middleware、路由、lifespan 或 Uvicorn 适配放入
   `app.api`。
3. 跨层且框架无关的稳定类型放入 `app.contracts`；纯 helper 放入
   `app.common`；配置解析和语义留在 `app.config`。
4. Graph 编排放入 `app.graph`，Agent 业务放入 `app.agent`，工具执行放入
   `app.tools`，索引/检索/生成放入 `app.rag`。
5. 更新 `.importlinter` 时把新 package 加到职责匹配的层，不通过调整顺序把偶然
   耦合合法化，也不添加 ignore。
6. 运行 `lint-imports --verbose` 查看完整 importer/imported 链，再运行 focused
   tests、Runtime inspection 和全量 Pytest。

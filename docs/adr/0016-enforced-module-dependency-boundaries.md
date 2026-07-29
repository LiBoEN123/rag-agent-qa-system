# ADR 0016：以 Import Linter 强制模块依赖边界

- 状态：Accepted
- 日期：2026-07-29

## 背景

项目经过多轮 Runtime 收敛后已经形成 API、Service、Graph、Agent、Tools、RAG、
LLM、基础设施和基础模块的职责方向，但这些规则主要依赖文档与 Code Review。
普通 review 容易漏掉函数内 import、包级 re-export 和间接依赖链，也无法持续保证
后续改动仍遵守同一架构。

修改前扫描还发现两处职责错位：

- `app.observability.middleware` 是 FastAPI HTTP middleware；
- `app.common.logging` 包含 Uvicorn access filter，并在 import 时修改全局 logger。

第一处是实际 import graph 违规，第二处虽然不导入 `uvicorn`，仍把 HTTP server
适配放进了基础模块。

## 决策

采用 `import-linter==2.13` 的原生 contracts，并在根目录维护 `.importlinter`：

- layers 合同固化应用 package 的单向依赖；
- acyclic siblings 合同禁止 `app` 顶层 package 循环；
- forbidden 合同明确保护 foundation、HTTP 外部包与 Settings 外部包；
- `include_external_packages = True` 使 FastAPI、Starlette 和
  pydantic-settings 进入官方 import graph；
- 不添加 `ignore_imports`，也不编写第二套 AST/import graph 分析器。

API 是最高层组合根，可以跨层组装 ChatService、Graph、Tools、RAG、Diagnostics
和 Response Builder。强制 API 只能依赖 Service 会产生无业务价值的包装器；真正
需要阻止的是低层反向依赖 API。

HTTP middleware 移到 `app/api/middleware.py`。Uvicorn URL 解码 filter 与配置函数
移到 `app/api/http_logging.py`，由 FastAPI lifespan 显式调用；配置保持幂等，
`app.common.logging` 不再产生 import-time Uvicorn 副作用。

`app.contracts`、`app.config`、`app.common` 是彼此独立的基础层：

- contracts 保存跨层稳定、框架无关的类型和值策略；
- config 独占 AppSettings、配置校验和 pydantic-settings；
- common 保存不依赖框架或业务层的纯 helper。

Tools 可以依赖 RAG，因为知识库工具是受治理的 RAG 能力入口，仍通过
`ToolRegistry.dispatch → ToolRuntime.execute → ToolSpec.handler`。RAG 不允许依赖
Tools，否则索引/检索算法会反向绑定工具协议并形成错误所有权。

pydantic-settings 合同允许合法的
`其他模块 → app.config → pydantic_settings` 间接链，因此使用 Import Linter 原生
`allow_indirect_imports = True`，只禁止其他模块直接导入该外部包。这个精确选项不
是 ignore；HTTP 和应用层合同仍检查间接链。

## 为什么不使用 ignore

ignore 会把当前违规固化成不可见债务，使合同只在名义上存在。发现违规时应移动
HTTP 适配、提取中立 contract/pure helper 或修正调用方向。函数内 import、
`importlib`、Service Locator、动态 Registry 和临时 re-export 同样不能作为逃生口。

## ChatService 范围

本次只固定模块方向并移动明确错位的 HTTP 责任。ChatService 已不依赖 FastAPI 或
Starlette，并使用框架无关 Request/Outcome；完整拆分 ChatService 或 API
Controller 会扩大主题并改变后续设计空间，因此留给独立 PR。

## CI

`.github/workflows/tests.yml` 的固定顺序为：

```text
Install dependencies
→ Compile Python sources
→ Ruff
→ Import Linter
→ Agent Runtime architecture inspection
→ Pytest
```

`lint-imports` 以正常非零退出码阻止 CI，不使用 `|| true`，不动态生成合同，也不只
检查本次改动文件。Runtime inspection 只确认配置声明，不复制图分析。

## 后续 package

新增 package 时先在模块边界文档中确定责任和允许方向，再加入 layers 或基础合同。
不能为适配当前偶然 import 而重排层级。新增中立类型优先进入 contracts，纯 helper
进入 common，HTTP/server 适配进入 API；任何合同变化都必须同时通过
`lint-imports --verbose` 和全量测试。

## 结果与影响

- 模块方向从评审约定变成可执行、CI 强制的架构合同；
- API 仍可作为组合根，不影响现有 `/chat` Schema 或状态语义；
- HTTP trace headers、URL 解码和控制字符转义保持；
- common/observability 不再承载 HTTP server 适配；
- 新依赖只增加基础 Import Linter 与 Grimp，不安装 UI extra，不修改现有核心包；
- ChatService、Agent Result/Policy、Graph、RAG、Tool Runtime、SQLite 和观测数据契约
  不变。

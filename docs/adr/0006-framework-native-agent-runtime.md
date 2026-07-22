# ADR 0006：框架原生 Agent Runtime

- 状态：Accepted
- 日期：2026-07-21
- 决策者：Project Maintainer

## 背景

PR1–PR3 在 `app/graph/nodes.py` 中逐步实现模型—工具循环、动态工具绑定、StructuredTool 调用、ToolMessage 批处理、错误分类、指纹、预算、trace 和 RAG 交接。该文件增长到 1276 行，同时承担父图业务节点、Agent Runtime 和工具治理，框架已能提供的能力也由项目重复维护，修改任一错误路径都容易破坏消息协议或条件边。

## 决策

采用与既有锁定依赖兼容的 `langchain==1.3.14`，用 `langchain.agents.create_agent` 构建标准 Agent 子图并嵌入 LangGraph 父图。没有采用 ToolNode-only fallback，因为 dry-run 证明该精确版本可直接复用 `langchain-core==1.4.9`、`langgraph==1.2.9`、`langchain-openai==1.3.5` 和 `pydantic==2.12.5`，无需升级现有核心依赖。

框架与项目职责如下：

| 能力 | 实现位置 |
| --- | --- |
| 模型—工具循环 | `create_agent` 内部 LangGraph model/tools 节点 |
| 动态工具绑定与解析 | `RegistryToolsMiddleware` 的模型和工具 hook |
| 参数校验、基础 ToolMessage | LangChain StructuredTool 与框架工具节点 |
| 模型、工具调用上限 | `ModelCallLimitMiddleware`、`ToolCallLimitMiddleware` |
| 图递归最终兜底 | 父图 invoke 的 `recursion_limit` |
| 业务错误、循环和耗时治理 | `ProjectToolGovernanceMiddleware` |
| 父图状态/API trace 适配 | `framework_agent.py` 与 `nodes.py` 薄适配节点 |
| 持久化 | 父图 LangGraph Checkpointer，包含子图消息状态 |

## 动态工具与治理边界

每次请求的模型 hook 从 `ToolProvider` 构造当前 Context 有权看到的动态 StructuredTool；执行 hook 使用同一 Provider 语义按名称解析实际工具。Provider 只检查 `required_permissions`，不会把审批、幂等键、主体、来源或风险策略前移。适配工具通过 LangChain `ToolRuntime[ToolInvocationContext]` 获得 trace、session、principal、permissions、source 和 idempotency key。该 runtime 参数是注入字段，不出现在模型业务参数 Schema 中。

所有生产调用只进入：

```text
Agent Model
→ create_agent framework tool node
→ request StructuredTool
→ ToolRegistry.dispatch
→ project ToolRuntime.execute
→ ToolSpec.handler
```

框架适配层不直接调用 handler。未绑定或隐藏名称由框架按 `unknown_tool` 处理，不反查 Registry 证明其存在；已解析调用仍进入项目 ToolRuntime，由它继续独占最终权限、策略、审批、审计、超时、有限重试、输出校验和幂等治理。本次不增加 ModelRetryMiddleware、ToolRetryMiddleware 或 RetryPolicy，因此不存在双重重试。

## 保留的自定义 Middleware

框架无法理解项目 `ToolErrorCode` 的业务严重性，也不能自行决定错误累计、连续相同调用/错误、总耗时预算或 terminal error 同批 fail-closed。因此仅保留 `ProjectToolGovernanceMiddleware` 处理这些规则。它包裹框架调用、读取标准 ToolMessage 并输出最小治理事件，不负责工具执行、Pydantic 校验、并行调度或 Runtime 重试。

调用指纹方面，注册工具复用 `app.tools.policy.build_tool_args_fingerprint()`；未知工具参数仅可能来自模型 JSON，因此使用小型 JSON-compatible fallback。`guardrails.py` 不再复制 Decimal、日期、UUID、bytes、Enum、Pydantic 等通用 canonicalization。

框架工具节点对同批调用使用并发 Send。为保持 PR3 “第一个 terminal error 后同批后续调用不执行”的确定性，当前 Agent 子图和父图均配置 `max_concurrency=1`，Middleware 只协调当前批次终止状态；它没有重建工具执行循环。后续若框架提供原生批次取消协议，可删除这段协调代码。

## RAG handoff

`search_knowledge_base` 仍是动态 Agent 工具，并继续经过 Registry/Runtime。单工具成功时 StructuredTool 返回公开的 `Command(graph=Command.PARENT, update=...)`，同时写入请求 AIMessage、匹配 `tool_call_id` 的 ToolMessage 和 `pending_rag_result`。由于嵌入子图节点还存在固定父图后继边，直接在该 Command 中指定父图 `goto` 会与后继更新并发冲突；因此 Command 只向父图更新状态，固定后继 `adapt_framework` 再通过条件边进入 `rag_subgraph`。同批多工具的成功结果也采用同一 pending-state 交接。

这保证模型不能读取原始 chunks 后自由生成无引用答案，后续仍由现有 RAG 子图执行拆分、检索/重排、证据门、回答、引用绑定和拒答。

## 删除与维护入口

删除父图中的 `agent_model`、`governed_tools`、`limit_handler` 节点，以及手工工具批次循环、StructuredTool ValidationError 捕获、基础 ToolMessage JSON 批处理、预算条件边和调用/错误指纹计算。`nodes.py` 仅保留父图 route、initialize、framework adapter、legacy plan/execute、RAG、compose 和 finalize。

今后的维护入口为：

- `framework_agent.py`：组装 `create_agent`、内置调用限制和父图兼容事件转换。
- `framework_middleware.py`：项目特有错误与预算治理。
- `framework_tools.py`：动态 ToolSpec、ToolRuntime Context、Registry dispatch 与 RAG Command 适配。

## 后果

标准循环、校验和基础消息协议随 LangChain/LangGraph 维护，项目代码只表达业务治理语义。父图 Checkpointer 继续保存完整消息轨迹，`recursion_limit` 提供框架级最终安全兜底。代价是新增一个精确顶层 LangChain 依赖，并需要在升级框架时回归 Middleware hook、ToolRuntime 注入、Command.PARENT 和调用限制的契约测试。

本决策不修改 API Schema、SQLite Schema、工具名称或业务参数 Schema，也不改变 RAG 检索、重排、证据门、引用和拒答算法。Planner/Executor 仅作为 `rag_fast_path` 和旧 decision 兼容路径保留。

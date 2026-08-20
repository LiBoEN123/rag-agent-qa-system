# 当前系统架构

本文是 `rag-agent-qa-system` 当前实现的主要架构入口。历史演进和已经删除的设计见
[2026 年 Agent 演进归档](../history/agent-evolution-2026.md)，可执行依赖规则见
[模块依赖边界](module-boundaries.md)。

## 总体结论

系统不存在 Parent Agent Graph。确定性 Router 位于 `create_agent` 之外，由 Agent
Runtime 以普通 Python 调用；正常 agent 请求直接进入 LangChain `create_agent` 返回的
CompiledStateGraph。项目仍用
LangGraph 实现独立 RAG Graph，但它只由本地知识库工具调用，不是 Agent 外层控制图。

```text
FastAPI application
→ Chat Router
→ ChatService
→ Query Rewrite
→ Agent Runtime
→ create_agent
→ LocalToolProvider / execute_registered_tool / ToolRuntime
  或 governed MCP adapter tools
→ AgentRunResult
→ ChatStatusPolicy
→ ChatServiceOutcome
→ ChatResponse（同步 JSON 或 SSE final）
```

本地知识库分支为：

```text
create_agent
→ search_knowledge_base StructuredTool
→ execute_registered_tool
→ ToolRuntime.execute
→ search_knowledge_base handler
→ 独立 RAG Graph
→ 结构化工具结果 + 配对 ToolMessage
→ Agent Result builder 归一化
```

## HTTP 与应用服务

`app.main` 只调用 `app.api.application.create_app()`。组合根创建 FastAPI、middleware、
Chat Router、health Router 和 diagnostics Router；真实 RAG/Agent/MCP 资源只由 lifespan
初始化。`app.api.chat` 只把 HTTP 请求转换为 `ChatServiceRequest`，调用已注入的
`ChatService`，再通过 `app.api.responses` 的单一函数构造响应。

`ChatService` 是同步与流式应用流程协调器：取得请求级 Settings 快照、读取 session
历史、执行 Query Rewrite、调用 `run_agent()` 或 `stream_agent()`、应用唯一状态策略、
安排 session 保存，并委托 `chat_telemetry` 完成 metrics、bad case、流式事件记账和
调用链收尾。两条入口共享请求准备和 Result 收尾，差异只保留在同步/异步 Runtime
执行方式和流式中间事件；`ChatServiceDependencies` 只表达 Settings、模型/Runtime、
Session、持久化和时钟等外部边界，不注入稳定的 trace 构造/汇总纯函数。Service 模型
不包含 FastAPI Request/Response，也不维护第二套 Agent State。

```text
handle  → prepare → run_agent    → finish Result → finalize
stream  → prepare → stream_agent → finish Result → finalize
```

其中 `chat_service.py` 决定何时发送 `start/progress/tool/answer_delta/final/done`，
`chat_telemetry.py` 决定事件如何更新 event count、首事件/首 delta、答案字符数、工具状态、
最终答案一致性和 `client_disconnected`。客户端取消仍关闭下游 Agent stream，且不保存
未完成的 assistant turn。

`POST /chat` 保持同步 JSON 行为；`POST /chat/stream` 由同一个 Router 闭包和同一个
Service 实例提供 SSE。流式执行直接调用
`create_agent.astream(updates/messages/values, subgraphs=False, version=v2)`，安全投影只
输出允许的进度、工具状态和回答增量；`final` 仍调用唯一
`build_chat_response_from_outcome()`。完整 HTTP/事件/安全合同见
[聊天流式 API](../api/chat-streaming.md)。

## Router 与唯一 Agent Loop

`app.agent.runtime.prepare_agent_runtime_invocation()` 以普通 Python 完成数字边界校验、
确定性 `route_agent_action()`、请求级 `ToolInvocationContext` 和框架 config 构造。
Router 只决定 `agent`、`direct_answer` 或 `clarify`，不选择具体工具：

- `direct_answer`、`clarify` 在普通 Python 中直接归一化结果，不构建或调用 Agent；
- `agent` 直接调用缓存的 `create_agent` CompiledStateGraph；
- 未知 action 继续 fail closed 为内部错误结果。

`create_agent` 是唯一 Agent Loop，其图只包含框架自己的 `model ↔ tools` 循环。动态
工具由 Runtime 按当前请求权限解析，并在构造本次 `create_agent` 时绑定；
`ProjectToolGovernanceMiddleware` 与受治理 StructuredTool 保留模型/工具调用计数、错误分类、重复调用检测、
累计耗时、同批 fail-closed、停止原因和 ToolMessage 配对。Agent 内部职责固定为：

- `runtime.py`：参数校验、Router、请求 Context、Agent 构建/缓存和同步 invoke；
- `streaming.py`：`astream` 消费与安全 SSE 事件投影；
- `guardrails.py`：治理规则、预算、错误分类和指纹计算；
- `tool_events.py`：append-only framework event 到 observations、tool trace 和 metrics；
- `results.py`：短路或 Framework output 到唯一 `AgentRunResult`，以及同一结果 trace 快照；
- `framework_middleware.py`：只在 LangChain 生命周期 Hook 中协调这些规则和 ToolMessage。

`framework_middleware.py` 不依赖 `runtime.py`；`.importlinter` 和 Runtime Inspection 同时
固化这条边界。系统不再维护事件游标或父图适配节点。

同步入口直接使用 `create_agent.invoke()`，流式入口直接使用
`create_agent.astream()`。两者共享输入准备和最终结果归一化，因此公开
`AgentRunResult`、status、references、LLM 计数与 tool trace 语义一致。

## 工具系统与治理边界

生产 Registry 默认固定注册八个 `ToolSpec`：

1. `search_knowledge_base`
2. `compare_documents`
3. `simple_calculator`
4. `list_documents`
5. `get_chunk_text`
6. `web_search`
7. `web_fetch`
8. `web_pdf_fetch`

只有 `SQL_TOOL_ENABLED=true` 时才额外注册 `list_sql_tables`、
`describe_sql_table` 和 `query_sql_readonly`。模型可见性由 Provider 根据请求权限过滤；
本地工具的执行期治理始终经过：

```text
LocalToolProvider.list_specs
→ request-level StructuredTool
→ execute_registered_tool()
→ ToolRuntime.execute()
→ ToolSpec.handler
```

`ToolRegistry` 只拥有 ToolSpec 注册、重名拒绝、按名查找、稳定顺序列举和契约导出；
`execute_registered_tool()` 是生产本地工具的唯一公共执行入口，按 Registry 取得进程内
共享 Runtime 后调用 `ToolRuntime.execute()`。Registry 不再提供 dispatch 方法，项目也
不再保留额外的 `dispatch_tool` 兼容门面。

本地 `ToolSpec` 的参数契约只由 Pydantic `args_model` 定义：同一个模型同时用于
`ToolRuntime` 参数校验、LangChain 模型可见 Schema 和 `contract_specs()` 导出，不再
手写第二份 `args_schema`。输出契约按真实能力非对称处理：存在 `output_model` 时由该
模型负责 Runtime 校验并派生导出 Schema；没有输出模型时才要求显式 `output_schema`
作为 handler payload 的展示契约。两者不能同时声明。这里的输出仅指 handler 的
`output` payload，不包含 `ToolResult` 外层状态、治理、attempt、耗时或审计元数据。

```text
StructuredTool
→ execute_registered_tool()
→ ToolRuntime.execute()
→ ToolSpec.handler
```

`ToolRuntime` 独占最终权限、策略、审批、幂等、审计、超时、有限重试、脱敏和输出
校验。Provider 可见性与 Framework Middleware 都不能绕过或复制这条边界。

可选 MCP Client 工具源默认关闭。显式启用后，FastAPI 启动期使用官方
`langchain-mcp-adapters.MultiServerMCPClient` discovery 并缓存受治理的只读工具。
MCP adapter 工具与本地工具进入同一个 `create_agent`，继续受请求权限、调用预算、
错误分类、重复调用和安全输出边界约束。`/chat`、`/chat/stream`、ChatService 与 SSE
projector 没有 MCP 专属分支。MCP 不进入本地 ToolRuntime；其执行链保持为
`McpToolProvider → governed MCP StructuredTool → adapter coroutine →
McpToolCallInterceptor → remote MCP server`，继续独立承担 Server Access Policy、审批、
超时、审计、序列化和 fail-closed。完整配置见
[受治理的 MCP Client 工具源](../tools/mcp-client.md)。

## 知识库工具与独立 RAG Graph

`search_knowledge_base` 是普通 StructuredTool。它先经过唯一执行入口和 ToolRuntime 的
统一治理，再由 handler 检索候选 chunks，并调用 `app.rag.graph.run_rag_subgraph()`。
传入 RAG 的问题是 Query Rewrite 后的 `effective_question`，`top_k`、
`decompose_top_k` 和 `trace_id` 来自请求级 Tool Context。

RAG Graph 保持原有问题歧义判断、Query Decompose、BM25 + Embedding 混合召回、
CrossEncoder 重排、Evidence Gate、Context Build、回答生成、JSON 解析、引用绑定和
验证。工具成功结果包含 RAG 的答案、引用、状态、置信度、timings 与 trace，并形成与
原 tool call ID 配对的普通 ToolMessage。Middleware 看到完整知识库结果后结束下一轮
Agent Model，Result builder 直接采用 RAG 的权威结果；不存在 `Command.PARENT`、
`pending_rag_result` 或父图条件边。

Web Search、Web Fetch、Web PDF Fetch、SQL 和 MCP 都是普通 ToolMessage 路径；它们不
进入本地 RAG、不写 ChromaDB/BM25，也不生成本地 RAG references。

## 状态与结果

运行中只有两类必要状态：

- `FrameworkAgentState`：`create_agent` 内部的 messages、append-only
  `governance_events`、累计耗时、停止原因和可选知识库结构化结果；
- 普通 Python Result 快照：Router decision、observations、references、tool trace、
  LLM 计数和最终状态，仅由 `results.py` 用于结果归一化与 trace 持久化。

最终结果固定经过：

```text
Router short-circuit 或 create_agent final state
→ app.agent.tool_events 事件投影（agent 路径）
→ app.agent.results 答案、状态与 LLM usage 归一化
→ AgentRunResult 校验
→ JSON-compatible dict
→ ChatService 附加受控 Query Rewrite trace
→ ChatStatusPolicy
→ HTTP status / code / public status / metrics status
```

`AgentRunResult` 只在 `results.py` 的构造边界通过 Pydantic 校验一次；ChatService 不再
把同一结果做 dict/model 往返或重复业务构造。未知 Runtime/Result status 按
`INTERNAL_ERROR` fail closed。项目没有 Parent Graph State，也不保存另一份嵌套业务状态。

## 配置、生命周期与持久化

`AppSettings` 是唯一完整配置模型，由官方 `pydantic-settings` 实现并冻结。
`get_settings()` 惰性返回缓存快照；模块导入不实例化 Settings，不创建模型、索引、
Registry 或数据库。FastAPI lifespan 依次初始化 RAG、直接 Agent Runtime 和可选 MCP，
关闭时依次清理 MCP、Tool Runtime、SQL Engine 与 Agent Runtime 缓存。

Phase 1 没有保留 LangGraph Checkpointer：当前产品没有 Interrupt/Resume、Replay、
Time Travel 或跨请求图恢复语义，为外层 Agent Graph 维护 checkpoint 只会保存一份无
消费者的中间状态。应用级 session 历史仍由 SQLite 保存，并继续供 Query Rewrite 使用；
Agent trace、RAG index、metrics、bad case 和 tool audit 的既有存储所有权不变。

## 模块边界

生产依赖方向由 `.importlinter` 的六项合同执行：

```text
api → services → agent → tools → rag → llm
    → storage / observability → contracts / config / common
```

`app.api` 是唯一 HTTP 组合层；services 不依赖 API，FastAPI/Starlette 不进入低层，
`pydantic-settings` 只由 `app.config` 直接导入。生产代码中没有 `app.graph` 包；独立
RAG Graph 属于 `app.rag`。额外 forbidden contract 明确禁止
`app.agent.framework_middleware → app.agent.runtime`。详细规则见
[模块依赖边界](module-boundaries.md)。

## 测试与质量门

维护当前架构时执行：

```bash
ruff check .
lint-imports
python -m compileall app scripts tests
python scripts/inspect_agent_runtime.py --check
pytest -q
python -m pip check
git diff --check
```

Runtime Inspection 检查直接 `create_agent` 节点、Parent Graph 文件/符号不存在、默认
ToolSpec 与可选 SQL/MCP 合同、本地工具唯一执行入口和 Runtime handler 所有权、MCP
独立治理、知识库 Tool 到 RAG Graph 的边界、同步/流式直接调用、唯一 Result/Status
Policy、应用组合根和当前文档。Import Linter 负责完整生产 import graph。自动测试用
fake 隔离模型、网络、RAG 索引和生产持久化副作用。

## 已知产品边界

- RAG JSON 生成节点不提供 token streaming；完整结果由 SSE `final` 返回。
- 无 Interrupt/Resume、跨请求人工审批恢复、Replay 或 Time Travel。
- MCP 第一版仅为默认关闭的 Streamable HTTP read-only Client。
- 无 SQL 写操作/自动迁移和多 Agent 编排。
- 无 OCR；扫描件 PDF 不能提取文本时不会自动识别图片内容。

这些是产品能力边界，不是待处理的 Parent Graph 清理清单。

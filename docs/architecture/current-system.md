# 当前系统架构

本文是 `rag-agent-qa-system` 当前实现的主要架构入口。历史演进和已经删除的设计见
[2026 年 Agent 演进归档](../history/agent-evolution-2026.md)，可执行依赖规则见
[模块依赖边界](module-boundaries.md)。

## 请求链

```text
FastAPI application
→ Chat Router
→ ChatService
→ Query Rewrite
→ LangGraph Parent Graph
→ LangChain create_agent
→ Local ToolRegistry / ToolRuntime 或 governed MCP adapter tools
→ RAG Subgraph（仅本地知识库检索成功时）
→ AgentRunResult
→ ChatStatusPolicy
→ ChatResponse（同步 JSON 或 SSE final）
```

`app.main` 只调用 `app.api.application.create_app()`。组合根创建 FastAPI、
middleware、Chat Router、health Router 和 diagnostics Router；真实 RAG/Graph
资源只由 lifespan 初始化。`app.api.chat` 只把 HTTP 请求转换为
`ChatServiceRequest`，调用已注入的 `ChatService`，再通过
`app.api.responses` 的单一函数构造响应。

`ChatService` 是同步与流式应用流程协调器：取得请求级 Settings 快照、读取 session
历史、执行 Query Rewrite、调用 `run_agent_langgraph()` 或
`stream_agent_langgraph()`、应用唯一状态策略、安排 session 保存，并委托
`chat_telemetry` 完成 metrics、bad case 和调用链收尾。Service 模型不包含 FastAPI
Request/Response 或第二套 Agent State。

`POST /chat` 保持同步 JSON 行为；`POST /chat/stream` 由同一个 Router 闭包和同一个
Service 实例提供 SSE。流式链为
`ChatService.stream → graph.astream(updates/messages, subgraphs=True, v2) → 安全投影 → SSE`。
`final` 仍调用唯一 `build_chat_response_from_outcome()`，详细 HTTP/事件/安全合同见
[聊天流式 API](../api/chat-streaming.md)。

## LangGraph 父图

```text
START
→ route
→ initialize
→ framework_agent / compose
→ adapt_framework
→ rag_subgraph / compose
→ finalize
→ END
```

`route_node` 是确定性 Router 的唯一生产所有者，只产生 `agent`、
`direct_answer` 或 `clarify`。普通请求进入 LangChain `create_agent` 子图；本地
知识库工具成功时通过 `Command.PARENT` 返回父图并进入原 RAG 子图，其他工具结果
作为 `ToolMessage` 回到模型或进入 compose。

## 状态与结果

父图只有一个平铺的 `AgentGraphState`。框架消息与 append-only 治理事件使用
reducer；observations、references 和 tool trace 由节点以新列表返回，节点不原地
修改输入状态。`FrameworkAgentState` 只属于嵌入的 `create_agent` 子图。

最终结果固定经过：

```text
AgentGraphState.status
→ app.agent.results 状态映射
→ AgentRunResult 校验
→ JSON-compatible dict
→ ChatStatusPolicy
→ HTTP status / code / public status / metrics status
```

Graph State 和 Checkpoint 保存普通字典，不保存 Pydantic 实例。未知 Result status
按 `INTERNAL_ERROR` fail closed。

## 工具系统

生产 Registry 默认固定注册八个 `ToolSpec`：

1. `search_knowledge_base`
2. `compare_documents`
3. `simple_calculator`
4. `list_documents`
5. `get_chunk_text`
6. `web_search`
7. `web_fetch`
8. `web_pdf_fetch`

只有 `SQL_TOOL_ENABLED=true` 时才额外注册：

9. `list_sql_tables`
10. `describe_sql_table`
11. `query_sql_readonly`

三个 SQL 工具全部要求请求 Context 显式拥有 `database:read`，启用配置不会自动授权。
它们仍走同一个 Provider、StructuredTool、Registry 和 Runtime，只在 handler 之后进入
独立的 `app/tools/sql_readonly/`：sqlglot 负责单语句只读 AST、物理表 allowlist、
危险函数和 LIMIT，SQLAlchemy 负责 Engine、Inspector、连接和执行。`/chat`、
`/chat/stream`、ChatService 与 SSE projector 没有 SQL 分支；完整安全合同见
[只读 SQL 工具](../tools/sql-readonly.md)。

模型可见性由 Provider 根据请求权限过滤。本地工具的执行期治理保持原路径：

```text
StructuredTool
→ ToolRegistry.dispatch()
→ ToolRuntime.execute()
→ ToolSpec.handler
```

`ToolRuntime` 独占最终权限、策略、审批、幂等、审计、超时、有限重试、脱敏和输出
校验。Provider 可见性与 Framework Middleware 都不能绕过或复制这条边界。

可选 MCP Client 工具源默认关闭。显式启用后，FastAPI 启动期使用官方
`langchain-mcp-adapters.MultiServerMCPClient` 从部署者静态配置的 Streamable HTTP
Server discovery 工具，只缓存本地精确 allowlist 中的 read-only tool，并加
`<server>_` 前缀。请求还必须拥有 `mcp:<server>` 权限。MCP adapter 已经提供
`StructuredTool`，因此不会反向转换成 `ToolSpec`；项目 bridge 在调用前完成权限、
超时、审计和有限不可信输出归一化，随后仍交给同一个
`ProjectToolGovernanceMiddleware` 计算调用/错误/重复/耗时预算：

```text
Local ToolRegistry → ToolRuntime → StructuredTool ─┐
                                                   ├→ create_agent
MultiServerMCPClient → governed StructuredTool ────┘
```

MCP Tool 不设置 `pending_rag_result`，`/chat`、`/chat/stream`、ChatService 和 SSE
projector 没有 MCP 分支。完整配置与安全边界见
[受治理的 MCP Client 工具源](../tools/mcp-client.md)。

## RAG

本地 RAG 保持原有文档加载、BM25 + Embedding 混合召回、CrossEncoder 重排、
证据门控、问题拆分、回答生成、引用绑定和引用验证。`app.rag.__init__` 是有真实
运行时语义的动态门面：索引重建会重新绑定 `chunks`、`texts`、
`docs_registry`、`bm25`、模型和 ready 状态，因此这些属性必须动态读取；
`run_rag_subgraph()` 保持延迟导入。

Web Search、Web Fetch 和 Web PDF Fetch 是普通工具路径，不写本地知识库，也不
生成本地 RAG references。

## 配置与生命周期

`AppSettings` 是唯一完整配置模型，由官方 `pydantic-settings` 实现并冻结。
`get_settings()` 惰性返回缓存快照；模块导入不实例化 Settings，不创建模型、
索引、Registry、数据库或 Checkpointer。

FastAPI lifespan 依次执行：

```text
configure_uvicorn_access_log()
→ validate_startup_environment()
→ initialize_rag()
→ initialize_agent_graph_runtime()
→ initialize_mcp_runtime()（仅显式启用时读取配置并 discovery）
→ yield
→ close_mcp_runtime()
→ close_tool_runtime()
→ close_sql_engine_runtime()
→ close_agent_graph_runtime()
```

## 模块边界

生产依赖方向由 `.importlinter` 的五项合同执行：

```text
api → services → graph → agent → tools → rag → llm
    → storage / observability → contracts / config / common
```

`app.api` 是唯一 HTTP 组合层；services 不依赖 API，FastAPI/Starlette 不进入低层，
`pydantic-settings` 只由 `app.config` 直接导入。详细规则和新增模块放置方法见
[模块依赖边界](module-boundaries.md)。

## 持久化边界

| 数据 | 位置与语义 |
| --- | --- |
| Session | SQLite user/assistant 会话消息，用于多轮 Query Rewrite |
| Agent trace | SQLite `agent_runs` / `agent_steps`，按配置选择性记录运行摘要与步骤 |
| LangGraph Checkpointer | disabled、memory 或本地 SQLite 的图状态快照；同步/流式 Graph 分别使用同步/异步 Saver，与 session 表职责不同 |
| RAG index | ChromaDB 向量与 metadata、BM25 内存索引及 manifest 构建指纹 |
| Metrics / bad case / tool audit | 独立 JSONL 可观测记录，不作为业务状态源 |
| 可选业务 SQL 数据源 | 部署者显式配置的外部只读数据源；禁止指向 Session、Checkpoint、Chroma 或 Evaluation 持久化 |

这些存储的 Schema 和所有权彼此独立；ChatService 拆分没有改变 SQLite、Chroma 或
Checkpointer 协议。

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

Runtime Inspection 检查父图、Framework Agent、默认八个 ToolSpec、可选 SQL/MCP 合同、单一 State/Result/
Status Policy、同步/流式聊天路由、共享 Service、Graph streaming Runtime、Settings
边界、应用组合根和文档归档；Import Linter 负责完整 import graph。自动测试使用 fake
隔离模型、网络、RAG 索引和生产持久化副作用。

产品质量阶段使用
[Evaluation Baseline v2](../evaluation/baseline-v2.md) 组合现有 Agent 与 RAG
evaluator。正式 CLI 位于 `scripts/` 根目录，可导入实现集中在
`scripts/evaluation/`；Runner 通过 `ChatService` 的现有依赖边界注入内存 session
和 no-op telemetry。它不属于生产请求链，也没有改变本页描述的 API、Graph、
Agent Result、工具、RAG、配置或持久化所有权。HTTP 性能测试、真实 Online Tool
冒烟、模型 Tool Calling 协议、架构检查和日志汇总各有独立入口。CI 只执行 v2 数据
合同校验，不运行真实模型、索引、FastAPI 服务或在线工具。

## 已知产品边界

- RAG JSON 生成节点不提供 token streaming；完整结果由 SSE `final` 返回。
- 无 Interrupt/Resume 和跨请求人工审批恢复。
- MCP 第一版仅为默认关闭的 Streamable HTTP read-only Client；无 MCP Server、stdio、
  OAuth、Resources、Prompts 或写工具。
- 无 SQL 写操作/自动迁移和多 Agent 编排。
- 无生产级分布式 Checkpointer。
- 无 OCR；扫描件 PDF 不能提取文本时不会自动识别图片内容。

这些是产品能力边界，不是待处理的架构清理清单。

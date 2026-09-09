# Agent Runtime 维护指南

当前唯一 Agent Loop 是 LangChain `create_agent` 返回的 CompiledStateGraph。项目没有
Parent Agent Graph；LangGraph 只继续承载框架内部 model/tools 循环和独立 RAG Graph。
`scripts/inspect_agent_runtime.py` 是结构与 Tool Schema 的维护入口。

## 十分钟阅读路径

| 顺序 | 文件 | 只需要理解 |
| --- | --- | --- |
| 1 | `app/api/application.py`、`app/api/chat.py`、`app/api/streaming.py` | FastAPI 如何组装，以及同步 JSON/SSE 如何进入 ChatService |
| 2 | `app/services/chat_service.py`、`chat_models.py`、`chat_stream_models.py`、`chat_telemetry.py` | 同步/流式应用流程和 telemetry 分工 |
| 3 | `app/contracts/chat.py` | Result status 如何唯一决定 HTTP、API 和 metrics 状态 |
| 4 | `app/agent/runtime.py` | 普通 Python Router、请求准备、Agent 缓存和同步 invoke |
| 5 | `app/agent/streaming.py` | 直接 astream、消息缓冲和安全事件投影 |
| 6 | `app/agent/framework_agent.py` | `create_agent`、最小内部 State 和 Middleware 顺序 |
| 7 | `app/agent/framework_middleware.py` | LangChain 生命周期 Hook、同批 fail-closed 与 RAG 终止协调 |
| 8 | `app/agent/framework_tools.py` | ToolSpec 到动态 StructuredTool 的薄适配和 MCP 工具合并 |
| 9 | `app/tools/runtime.py` | 单次本地工具调用的权限、审批、审计、超时、重试和输出校验 |
| 10 | `app/tools/knowledge_base.py`、`app/rag/graph/` | 知识库 Tool 如何调用独立 RAG Graph |
| 11 | `app/agent/guardrails.py`、`tool_events.py`、`results.py`、`answer.py` | 治理计算、事件投影、RAG 映射与 Result 合同 |
| 12 | `app/tools/mcp/` | 可选 MCP discovery、权限、同步/异步 bridge 和安全输出 |

## 主调用链

```text
FastAPI application
→ Chat Router
→ ChatService
→ Session History / Query Rewrite
→ app.agent.runtime.prepare_agent_runtime_invocation
→ 普通 Python route_agent_action
  ├─ direct_answer / clarify → 普通 Python 结果归一化
  └─ agent → create_agent.invoke
             → ProjectToolGovernanceMiddleware.wrap_tool_call
             → 请求级 StructuredTool / governed MCP BaseTool
             → ToolMessage + Command State Update
             → tool_events 投影 → results 结果归一化
→ AgentRunResult
→ ChatStatusPolicy
→ ChatResponse
```

流式入口只替换执行方式：

```text
POST /chat/stream
→ ChatService.stream
→ app.agent.streaming.stream_agent
→ create_agent.astream(updates/messages/values, subgraphs=False, version=v2)
→ project_agent_stream_part 安全投影
→ ChatStreamSSEEncoder
→ SSE final / done
```

同步与流式入口共享参数校验、Router、ToolInvocationContext、recursion limit 和最终
结果归一化。流式 projector 不把内部 State、ToolMessage、工具参数/原始结果、
reasoning 或 RAG JSON 交给 API；最终结果仍经过唯一 `ChatStatusPolicy` 和
`build_chat_response_from_outcome()`。

## Router 与 Agent Loop

Router 只决定 `agent`、`direct_answer` 或 `clarify`，不决定具体工具。
`prepare_agent_runtime_invocation()` 是生产代码中唯一 Router 调用点；普通问题由
`create_agent` 根据请求可见的动态 Tool Schema 选择工具。直答与澄清不会初始化或
调用 Agent Model。

`build_framework_agent()` 只构造一次 `create_agent`：

- Runtime 为模型绑定和实际 ToolNode 执行构造同一请求级 governed 工具集合；
- `ProjectToolGovernanceMiddleware` 通过原生 `wrap_tool_call/awrap_tool_call` 记录
  append-only 治理事件并实施项目停止语义；
- `ModelCallLimitMiddleware` 与 `ToolCallLimitMiddleware` 实施框架调用上限；
- `max_concurrency=1` 让同批 terminal fail-closed 行为确定。

项目 Middleware 必须成对提供同步和异步 hook。同步入口调用 `wrap_*`，流式入口
调用 `awrap_*`；只实现一套会导致另一条传输路径绕开治理或直接失败。

## Model Governance：已接受边界

模型事件没有迁移到 `wrap_model_call/awrap_model_call`。Phase 6B 在当前
`langchain==1.3.14` 上验证了同步可行性，但最小异步 Model Hook 会使
`ainvoke/astream` 在模型已返回后无法可靠结束，因此正式结论为
`KEEP_CURRENT_MODEL_MARKER` / `PHASE6B_SKIPPED`。

当前固定链为 `_DeferredAgentModel → _GovernedBoundModel → AIMessage private marker →
after_model/aafter_model → FrameworkAgentState`。Inspection 应把它报告为
`accepted_model_governance_boundary`，不能把 private marker 当作残留协议或 TODO。
未来只有升级依赖并重新通过 sync、async、streaming 三条隔离门禁后才重新评估；不要用
ContextVar、thread-local、pending event dict、全局缓存或额外 Model Graph Node 替代它。

## 工具与 RAG 边界

普通本地工具固定经过：

```text
StructuredTool
→ execute_project_tool
→ ToolSpec.handler
→ 标准工具结果字典
→ 配对 ToolMessage
→ Command(update={messages, governance_events, ...})
```

模型可见性不是授权替代。`ProjectToolVisibilityMiddleware` 只过滤模型可见 Schema；
`execute_project_tool` 是唯一公共本地执行入口。最终权限、策略、审批、幂等、审计、
超时、有限重试、脱敏和输出模型仍由该执行边界独占。

`search_knowledge_base` 也遵守这条链。其 handler 先取得候选 chunks，再使用 Context 中
的 `effective_question`、`top_k`、`decompose_top_k` 和 `trace_id` 调用独立
`run_rag_subgraph()`。成功工具结果包含 `rag_result`；Middleware 通过同一个原生
Command 写入配对 ToolMessage、治理事件和内部 `knowledge_base_result`，在下一轮模型前
结束 Agent，Result builder 将 RAG 状态映射为权威 AgentRunResult。项目不再通过
`ToolMessage.artifact` 回传 Tool Governance；不存在 `Command.PARENT`、父图或事件投影
游标。

`web_search`、`web_fetch`、`web_pdf_fetch`、只读 SQL 与 MCP 是普通 ToolMessage
路径，不进入本地 RAG，也不生成本地 references。MCP adapter `StructuredTool` 不
反向转换为 ToolSpec，但继续受统一的请求权限、治理事件、调用预算与安全输出约束。

## 状态与结果

`FrameworkAgentState` 只保存框架真正需要的字段：

- `messages`；
- append-only `governance_events`；
- `agent_elapsed_ms` 和 `stop_reason`；
- 可选 `knowledge_base_result`。

`tool_events.py` 在调用结束后一次性把治理事件转换成 observations、tool trace、
迭代/工具/错误计数和连续指纹；没有 `framework_events_applied` 之类重放游标。
Router decision 和问题语义保留在 Runtime Invocation；`results.py` 从框架输出直接
构造 `AgentRunResult`，不再维护包含 references、业务 status 和 answer 的临时结果状态。

固定结果链为：

```text
Router short-circuit 或 create_agent final state
→ app.agent.tool_events 投影（agent 路径）
→ app.agent.results 确定性归一化（RAG status 直接映射最终 Result status）
→ AgentRunResult
→ model_dump(mode="json")
→ ChatService 附加受控 Query Rewrite trace
→ ChatStatusPolicy
→ ChatServiceOutcome / API / metrics
```

RAG 的 `OK`、`NEED_CLARIFICATION`、`NO_RELEVANT_INFO`、`MODEL_ERROR` 和
`MODEL_BAD_OUTPUT` 先由 RAG-to-Agent mapper 收敛。未知状态与未知异常一律 fail
closed；Pydantic 模型只在 Result builder 做一次边界校验，ChatService 收到普通字典且
不再重复构造或校验 Agent Result。

## 配置与生命周期

`AppSettings` 是唯一完整应用配置模型。模块导入不实例化配置，也不创建模型、
Registry、Runtime、数据库或 RAG 索引。FastAPI lifespan 使用同一个配置快照初始化
RAG、直接 Agent Runtime 与可选 MCP，关闭时清理 MCP、Tool Runtime、SQL Engine 和
Agent Runtime 缓存。

| 缓存 | 创建时机 | 清理入口 |
| --- | --- | --- |
| Settings | 首次 `get_settings()` | `clear_settings_cache()` |
| 角色模型 | 首次对应 Model Factory getter | `clear_model_factory_cache()` |
| Local ToolSpec catalog | 首次 `get_local_tool_specs(settings)` | `get_local_tool_specs.cache_clear()` |
| Local StructuredTool catalog | 首次 `get_local_tools(settings)` | `get_local_tools.cache_clear()` |
| Tool Runtime | 首次正式执行本地工具 | `reset_tool_runtime()` |
| SQL Engine | 第一次实际 SQL 工具调用 | `clear_sql_engine_cache()` |
| MCP Client / Tool cache | 显式启用时由 lifespan discovery | `close_mcp_runtime()` |
| `create_agent` Runtime | lifespan 预编译或首次普通请求 | `reset_agent_runtime()` |

当前没有 Interrupt/Resume、Replay、Time Travel 或跨请求图恢复语义，因此没有 Agent
Checkpointer 配置与数据库。SQLite session 历史仍是 Query Rewrite 的应用级存储，
Agent trace 仍可按 `ENABLE_AGENT_TRACE_STORE` 单独保存。

## 文件职责

| 文件 | 责任 |
| --- | --- |
| `app/agent/runtime.py` | 参数校验、Router、请求 Context、Agent 缓存和同步 invoke |
| `app/agent/streaming.py` | 直接 astream、安全 projector、取消传播和最终状态捕获 |
| `app/agent/framework_agent.py` | `create_agent` 与最小 Framework State |
| `app/agent/framework_middleware.py` | LangChain 模型/工具生命周期 Hook 和 RAG 终止协调 |
| `app/agent/framework_tools.py` | ToolSpec 到缓存 StructuredTool 的薄适配及 MCP 合并 |
| `app/agent/tool_visibility.py` | 模型调用前按 Runtime Context 过滤本地可见工具 |
| `app/agent/guardrails.py` | 工具治理规则、预算、错误分类、停止原因和指纹 |
| `app/agent/tool_events.py` | Framework event 到 observation、trace 和 metrics 的投影 |
| `app/agent/results.py` | Framework output 到唯一 AgentRunResult、校验和 trace 快照 |
| `app/tools/context.py` | 请求主体、权限、trace 和 RAG 请求参数 |
| `app/tools/catalog.py` | 不可变 ToolSpec catalog、重名检查、配置缓存与名称导出 |
| `app/tools/runtime.py` | 执行期权限、策略、审计、超时、重试、脱敏和输出校验 |
| `app/tools/knowledge_base.py` | 知识库检索和独立 RAG Graph 调用 |
| `app/rag/graph/` | 专用 RAG 状态、节点、条件和运行入口 |
| `app/services/chat_service.py` | Query Rewrite、Runtime、状态策略、session 与 telemetry 编排 |

## 常见问题定位

| 现象 | 首先检查 |
| --- | --- |
| Router 执行两次 | 是否绕开 `run_agent()` / `stream_agent()` 又手动调用 Router |
| 模型看不到本地工具 | Visibility Middleware、ToolSpec `required_permissions`、Runtime Context |
| Catalog 中存在但 Agent 不可见 | `request.override(tools=...)` 的权限过滤结果 |
| 工具可见但执行被拒绝 | ToolRuntime 权限、审批、幂等和策略 |
| 工具错误后未恢复 | error code 分类、stop reason、同批 fail-closed |
| 重复调用未停止 | governance event 指纹与连续计数 |
| 知识库未返回引用 | 工具结果是否含合法 `rag_result`，RAG 状态和引用验证 |
| 流式泄漏参数或正文 | `project_agent_stream_part()` 白名单与 tainted buffer |
| 同步/流式结果不一致 | 两入口的最终 `build_framework_agent_result()` 输入 |
| MCP 工具不存在 | 启动期 discovery、Server allowlist、默认主体/请求权限 |
| 配置变更不生效 | `.env` 非热更新；清缓存或重启进程 |

## 新增普通工具

1. 定义严格 Pydantic `args_model`；输出可严格建模时定义 `output_model`，否则只声明
   显式 `output_schema` fallback，然后创建 ToolSpec。不要再手写 `args_schema`，也不要
   同时维护 `output_model` 与重复 `output_schema`。
2. 在 `get_local_tool_specs()` 的普通列表中追加 ToolSpec，设置权限、风险、超时、重试、幂等、审批和审计模式。
3. 让 handler 返回标准 `tool_success()` / `tool_error()`。
4. 为 catalog、执行期 Policy、Middleware 可见性和框架调用补测试。
5. 运行 Runtime inspection、Import Linter 和全量测试。

通常不需要修改 Router、`create_agent` 或 Middleware。只有知识库完整 RAG 工具使用
`knowledge_base_result` 终止下一轮模型；新增普通工具不能复制这个专用语义。

## 不应该做什么

- 不重新创建包裹 `create_agent` 的 Parent StateGraph。
- 不在 Router 中枚举工具意图或加入 URL/PDF/SQL/MCP 关键词分支。
- 不让 StructuredTool 绕过 `execute_project_tool` 直接调用本地 handler。
- 不在 ChatService 或 SSE projector 中加入具体工具业务分支。
- 不复制 LangChain 已提供的模型—工具循环、参数校验或 ToolMessage 协议。
- 不把 ToolMessage 正文、工具参数、网页/PDF/chunk 内容或 reasoning 投影到 SSE。
- 不把应用级 session 历史误当成 Agent checkpoint。

## 检查与升级

```bash
ruff check .
lint-imports
python -m compileall app scripts tests
python scripts/inspect_agent_runtime.py --check
pytest -q
python -m pip check
git diff --check
```

Inspection 直接读取 `create_agent.get_graph()` / `get_graph(xray=True)` 和模型可见的
Pydantic JSON Schema；不会初始化真实模型、执行 handler、访问网络或读取/重建 RAG
索引。它同时检查 Parent Graph 文件/符号不存在、同步/流式入口直接调用
`invoke/astream`、知识库 Tool 调用独立 RAG Graph、默认工具和可选 SQL/MCP 合同、
Result/Status Policy、应用组合根和当前架构文档。实际模块方向由官方 `lint-imports`
全量判定。

Inspection 还锁定 `ProjectToolGovernanceMiddleware` 的原生同步/异步 Tool Hook、
`framework_tools.py` 不手工构造 `ToolCallRequest`、项目私有 Tool artifact marker 不存在，
以及 MCP 工具不经过 framework 二次包装。

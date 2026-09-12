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
→ POST /chat/stream
→ ChatService.stream
→ Query Rewrite
→ Agent Runtime
→ create_agent
→ ProjectToolVisibilityMiddleware
→ cached Local StructuredTool / execute_project_tool
  或 governed MCP adapter tools
→ AgentRunResult
→ ChatStatusPolicy
→ ChatServiceOutcome
→ ChatResponse（SSE final.data.response）
→ done
```

本地知识库分支为：

```text
create_agent
→ search_knowledge_base StructuredTool
→ execute_project_tool
→ search_knowledge_base handler
→ 独立 RAG Graph
→ 结构化工具结果 + 配对 ToolMessage
→ Agent Result builder 归一化
```

## Architecture Freeze（2026-08-21）

架构状态：`frozen`。这表示本轮 Phase 1～6B 的结构整理已经结束；后续功能应落入以下
现有边界，而不是继续以“瘦身”为理由改写 Runtime。

### Stable Boundaries

- FastAPI/API 只负责组合、协议转换、HTTP/SSE 与生命周期。
- ChatService 拥有 Query Rewrite、session、Agent 调用、状态策略和 telemetry 编排。
- JSON 形态的业务模型输出由 `app.llm.schemas` 定义，并统一经过 `ModelGateway` 的原生
  Structured Output 或受控文本 fallback；业务模块不自行解析模型 JSON。
- `runtime.py` 以普通 Python 准备请求和 Router；`stream_agent` 唯一执行 `create_agent.astream`。
- LangChain `create_agent` 是唯一 Agent Loop；项目不再包裹 Parent Agent Graph。
- Project Governance Middleware 拥有模型事件消费、Tool Hook、预算和 fail-closed 语义。
- Local catalog 以不可变 tuple 缓存全部本地 StructuredTool；Model Middleware 只按请求
  Context 过滤模型可见集合。
- `execute_project_tool` 是唯一本地工具执行治理边界。
- MCP Bridge / Interceptor 独立治理远端 MCP；MCP 不进入本地 ToolRuntime。
- 独立 RAG Graph 只由知识库工具调用，是唯一专用业务工作流 Graph。
- Framework State 只保存框架运行所需状态；`AgentRunResult` 是 Runtime 输出边界。
- `ChatStatusPolicy` 唯一决定业务目标 HTTP 状态、公开状态与 metrics 状态，结果通过 SSE final 公开。

### Explicitly Accepted Design

Phase 6B 的正式结论是 `KEEP_CURRENT_MODEL_MARKER` / `PHASE6B_SKIPPED`。当前
LangChain 1.3.14 的原生异步 Model Middleware 门禁无法可靠通过，因此保留
`_DeferredAgentModel → _GovernedBoundModel → AIMessage private marker →
aafter_model → FrameworkAgentState`。这是已接受的内部模型治理边界，不是
待迁移技术债；只有依赖升级后重新通过异步模型与流式执行门禁，才可复议。

ToolNode 同批调用共享输入 State，因此 `_batch_events` 与 `max_concurrency=1` 继续承担
terminal 后续调用的确定性 fail-closed。这也是明确保留的并发安全语义。

### No Longer Exists

- Parent Agent Graph、`Command.PARENT`、`pending_rag_result` 和父图事件游标；
- `ToolRegistry.dispatch` 与 `dispatch_tool` 转发层；
- `ToolRegistry`、`ToolProvider`、`LocalToolProvider` 和构建期请求 Context；
- 项目私有 Tool governance artifact bus；
- ToolSpec 手写 `args_schema` 与 Pydantic `args_model` 的重复契约。

### Do Not Refactor Without Evidence

今后只有可复现测试失败、性能 profiling、Evaluation 数据、明确功能需求或真实维护痛点
可以触发架构修改。文件尺寸、类数量或“看起来还能更简单”都不是修改这些边界的依据。

## HTTP 与应用服务

`app.main` 只调用 `app.api.application.create_app()`。组合根创建 FastAPI、middleware、
Chat Router、health Router 和 diagnostics Router；真实 RAG/Agent/MCP 资源只由 lifespan
初始化。`app.api.chat` 只把 HTTP 请求转换为 `ChatServiceRequest`，调用已注入的
`ChatService`，再通过 `app.api.responses` 的单一函数构造响应。

`ChatService` 编排唯一流式聊天流程：取得请求级 Settings 快照、读取 session
历史、执行 Query Rewrite、调用 `stream_agent()`、应用唯一状态策略、
安排 session 保存，并委托 `chat_telemetry` 完成 metrics、bad case、流式事件记账和
调用链收尾。`stream()` 复用现有请求准备和 Result 收尾；`ChatServiceDependencies` 只表达 Settings、模型/Runtime、
Session、持久化和时钟等外部边界，不注入稳定的 trace 构造/汇总纯函数。Service 模型
不包含 FastAPI Request/Response，也不维护第二套 Agent State。

```text
stream  → prepare → stream_agent → finish Result → finalize
```

其中 `chat_service.py` 决定何时发送 `start/progress/tool/answer_delta/final/done`，
`chat_telemetry.py` 决定事件如何更新 event count、首事件/首 delta、答案字符数、工具状态、
最终答案一致性和 `client_disconnected`。客户端取消仍关闭下游 Agent stream，且不保存
未完成的 assistant turn。

`POST /chat/stream` 是唯一聊天 HTTP 入口，包括空问题也经过 `ChatService.stream()`，
以 `final.data.http_status` 表达业务状态。Agent 执行统一经过
`ChatService.stream → stream_agent`，直接调用
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
`ProjectToolGovernanceMiddleware.awrap_tool_call` 直接包裹框架 ToolNode
调用，保留模型/工具调用计数、错误分类、重复调用检测、累计耗时、同批 fail-closed、
停止原因和 ToolMessage 配对。工具治理结果通过原生 `Command(update=...)` 写入
`FrameworkAgentState`，不再借用 `ToolMessage.artifact` 传递项目私有事件。Agent 内部
职责固定为：

- `runtime.py`：参数校验、Router、请求 Context、Agent 构建/缓存；
- `streaming.py`：唯一 Agent executor，消费 `astream` 并安全投影 SSE 事件；
- `guardrails.py`：治理规则、预算、错误分类和指纹计算；
- `tool_events.py`：append-only governance event 一次性生成公开 tool trace；
- `results.py`：短路或 Framework output 到唯一 `AgentRunResult`，以及同一结果 trace 快照；
- `framework_middleware.py`：只在 LangChain 生命周期 Hook 中协调这些规则和 ToolMessage。

`framework_middleware.py` 不依赖 `runtime.py`；`.importlinter` 和 Runtime Inspection 同时
固化这条边界。系统不再维护事件游标或父图适配节点。

唯一执行链为 `ChatService.stream → stream_agent → create_agent.astream`。
模型适配器只提供 `ainvoke`，项目 Middleware 只覆盖异步 hooks；最终 `values` 状态
由 `build_framework_agent_result` 归一化，保留公开 `AgentRunResult`、status、references、
LLM 计数与 tool trace 语义。本地同步工具由框架调度，MCP 由异步 ToolNode 直接 await coroutine。

## 工具系统与治理边界

生产 catalog 默认固定包含八个 `ToolSpec`：

1. `search_knowledge_base`
2. `compare_documents`
3. `simple_calculator`
4. `list_documents`
5. `get_chunk_text`
6. `web_search`
7. `web_fetch`
8. `web_pdf_fetch`

只有 `SQL_TOOL_ENABLED=true` 时才额外注册 `list_sql_tables`、
`describe_sql_table` 和 `query_sql_readonly`。模型可见性由 LangChain Model Middleware
根据请求权限过滤；
本地工具的执行期治理始终经过：

```text
cached tuple[ProjectStructuredTool, ...]
→ create_agent tools
→ ProjectToolVisibilityMiddleware
→ request.override(tools=visible_tools)

create_agent ToolNode
→ ProjectToolGovernanceMiddleware.awrap_tool_call
→ cached StructuredTool
→ execute_project_tool()
→ ToolSpec.handler
→ paired ToolMessage
→ Command(update={governance_events, stop_reason, ...})
```

`app.tools.catalog` 用普通 tuple 保持稳定顺序，在构造时一次拒绝重名，并按完整
`AppSettings` 缓存 ToolSpec。`app.agent.framework_tools` 缓存不绑定请求 Context 的
StructuredTool；本地权限差异复用同一个 compiled Agent。

本地 `ToolSpec` 的参数契约只由 Pydantic `args_model` 定义：同一个模型同时用于
LangChain 模型可见 Schema、执行输入和 manifest contract 导出，不再
手写第二份 `args_schema`。输出契约按真实能力非对称处理：存在 `output_model` 时由该
模型负责 Runtime 校验并派生导出 Schema；没有输出模型时才要求显式 `output_schema`
作为 handler payload 的展示契约。两者不能同时声明。这里的输出仅指 handler 的
`output` payload，不包含 Runtime result envelope 的外层状态、治理、attempt、耗时或审计元数据。

```text
StructuredTool
→ execute_project_tool()
→ ToolSpec.handler
```

`execute_project_tool` 独占最终权限、策略、审批、幂等、审计、超时、有限重试、脱敏和
输出校验。模型可见性 Middleware 不能绕过或复制这条边界。

可选 MCP Client 工具源默认关闭。显式启用后，FastAPI 启动期使用官方
`langchain-mcp-adapters.MultiServerMCPClient` discovery 并缓存受治理的只读工具。
MCP adapter 工具与本地工具进入同一个 `create_agent`，继续受请求权限、调用预算、
错误分类、重复调用和安全输出边界约束。`/chat/stream`、ChatService 与 SSE
projector 没有 MCP 专属分支。MCP BaseTool 不再由 framework tools 二次包装，而是直接
经过同一个原生 Tool Middleware。MCP 不进入本地 ToolRuntime；其执行链保持为
`ToolNode → ProjectToolGovernanceMiddleware → McpToolProvider 提供的 governed MCP
StructuredTool → adapter coroutine →
McpToolCallInterceptor → remote MCP server`，继续独立承担 Server Access Policy、审批、
超时、审计、序列化和 fail-closed。完整配置见
[受治理的 MCP Client 工具源](../tools/mcp-client.md)。

## 知识库工具与独立 RAG Graph

`search_knowledge_base` 是普通 StructuredTool。它先经过唯一执行入口和 ToolRuntime 的
统一治理，再由 handler 检索候选 chunks，并调用 `app.rag.graph.run_rag_subgraph()`。
传入 RAG 的问题是 Query Rewrite 后的 `effective_question`，`top_k`、
`decompose_top_k` 和 `trace_id` 来自请求级 Tool Context。

RAG Graph 保持原有问题歧义判断、Query Decompose、BM25 + Embedding 混合召回、
CrossEncoder 重排、Evidence Gate、Context Build、回答生成、结构化结果规范化、引用
绑定和验证。模型调用失败与结构化输出校验失败仍分别映射为 `MODEL_ERROR` 和
`MODEL_BAD_OUTPUT`。工具成功结果包含 RAG 的答案、引用、状态、置信度、timings 与 trace，并形成与
原 tool call ID 配对的普通 ToolMessage。Middleware 在同一个原生 Command 中把该消息、
治理事件和 `knowledge_base_result` 写入 State，下一轮 `abefore_model` 直接读取 State
并结束 Agent Model，Result builder 采用 RAG 的权威结果；不存在 `Command.PARENT`、
`pending_rag_result` 或父图条件边。

Web Search、Web Fetch、Web PDF Fetch、SQL 和 MCP 都是普通 ToolMessage 路径；它们不
进入本地 RAG、不写 ChromaDB/BM25，也不生成本地 RAG references。

## 状态与结果

运行中只有两类必要状态：

- `FrameworkAgentState`：`create_agent` 内部的 messages、append-only
  `governance_events`、累计耗时、停止原因和可选知识库结构化结果；
- `AgentRunResult`：`results.py` 直接构造的最终结果，包含 references、公开 tool trace、
  LLM usage 和最终状态；`tool_result_summary` 仅从同一份公开 tool trace 生成。

治理事件保留原始执行事实；`tool_events.py` 直接投影为模型、counted tool call 和
guardrail trace。只有 counted call 分配连续 `step_id`，模型与 guardrail 不占工具编号。
Result summary 按 `step == "tool_call"` 且含 `step_id` 选取同一计数范围，未计数的
拦截不进入摘要；已计数但 handler 未执行的失败仍保留失败摘要。

最终结果固定经过：

```text
Router short-circuit 或 create_agent final state
→ app.agent.tool_events 直接生成 public tool trace（agent 路径）
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
独立治理、原生 Tool Middleware Hook、项目 artifact 协议不存在、知识库 Tool 到 RAG
Graph 的边界、唯一流式执行入口、唯一 Result/Status
Policy、应用组合根和当前文档。Import Linter 负责完整生产 import graph。自动测试用
fake 隔离模型、网络、RAG 索引和生产持久化副作用。

## 已知产品边界

- RAG JSON 生成节点不提供 token streaming；完整结果由 SSE `final` 返回。
- 无 Interrupt/Resume、跨请求人工审批恢复、Replay 或 Time Travel。
- MCP 第一版仅为默认关闭的 Streamable HTTP read-only Client。
- 无 SQL 写操作/自动迁移和多 Agent 编排。
- 无 OCR；扫描件 PDF 不能提取文本时不会自动识别图片内容。

这些是产品能力边界，不是待处理的 Parent Graph 清理清单。

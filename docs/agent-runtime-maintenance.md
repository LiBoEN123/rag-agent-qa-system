# Agent Runtime 维护指南

本文面向需要定位或维护 Agent Runtime、但不需要逐行阅读全部实现的开发者。当前唯一主实现是 LangGraph 父图嵌入 LangChain `create_agent` 子图；`inspect_agent_runtime.py` 是图结构和工具 Schema 的维护入口，Evaluation Baseline 的可导入实现统一位于 `scripts/evaluation/`。

## 十分钟阅读路径

按以下顺序阅读，每个文件只关注表中内容：

| 顺序 | 文件 | 只需要理解 |
| --- | --- | --- |
| 1 | `app/api/application.py`、`app/api/chat.py`、`app/api/streaming.py` | FastAPI 如何组装，以及同步 JSON/SSE 如何进入 ChatService |
| 2 | `app/services/chat_service.py`、`chat_models.py`、`chat_stream_models.py`、`chat_telemetry.py` | 同步/流式流程、事件模型和 telemetry 如何分工 |
| 3 | `app/contracts/chat.py` | Result status 如何唯一决定 HTTP、API 和 metrics 状态 |
| 4 | `app/graph/builder.py` | 父图固定入口、节点、边和条件出口 |
| 5 | `app/graph/state.py` | 唯一完整请求状态及 reducer |
| 6 | `app/agent/results.py` | AgentRunResult、内部/RAG 状态 mapper 和结果构造 |
| 7 | `app/graph/conditions.py` | 框架 Agent、RAG 与 compose 如何分流 |
| 8 | `app/graph/nodes.py` | `route_node` 如何独占 Router 调用/计时，以及节点如何返回显式更新 |
| 9 | `app/agent/framework_agent.py` | `create_agent`、调用限制 Middleware 和事件纯投影如何组装 |
| 10 | `app/agent/tool_events.py` | 标准工具结果和治理事件如何转换为 observation/trace |
| 11 | `app/agent/answer.py` | RAG 与 compose 如何生成无副作用状态更新 |
| 12 | `app/agent/framework_middleware.py` | 项目错误分类、预算、循环防护和同批 fail-closed |
| 13 | `app/agent/framework_tools.py` | 动态 StructuredTool、Registry dispatch 和 RAG handoff |
| 14 | `app/tools/runtime.py` | 单次工具调用的策略、审计、超时、重试和输出校验 |

## 主调用链

```text
FastAPI application
→ Chat Router
→ ChatService
→ Session History / Query Rewrite
→ run_agent_langgraph
→ LangGraph Parent Graph: route → initialize
→ LangChain create_agent
→ RegistryToolsMiddleware / ProjectToolGovernanceMiddleware
→ StructuredTool
→ ToolRegistry.dispatch
→ ToolRuntime.execute
→ ToolSpec.handler
→ ToolMessage / Command.PARENT
→ RAG Subgraph 或 Compose
→ Finalize
```

流式入口复用同一条业务链，只把执行门面和 HTTP 适配替换为：

```text
POST /chat/stream
→ ChatService.stream
→ stream_agent_langgraph
→ graph.astream(updates/messages, subgraphs=True, version=v2)
→ project_graph_stream_part 安全投影
→ ChatStreamSSEEncoder
→ SSE final / done
```

`app.graph.runtime` 是同步 `invoke` 与异步 `astream` 共享的参数校验、graph input、
ToolInvocationContext、Checkpointer thread ID、recursion limit 和 max concurrency
所有者；同步与流式入口分别取得匹配的同步/异步 compiled Graph。流式 projector 不把 State update、
ToolMessage、工具参数/原始结果、reasoning 或 RAG JSON 交给 API；最终结果仍经过唯一
`ChatStatusPolicy` 和 `build_chat_response_from_outcome()`。完整接口合同见
[聊天流式 API](api/chat-streaming.md)。

`RegistryToolsMiddleware` 与 `ProjectToolGovernanceMiddleware` 的 model/tool wrap hook
必须成对提供同步和异步实现。同步入口调用 `wrap_*`，流式入口调用 `awrap_*`；两套入口
复用同一纯治理逻辑。只实现同步 hook 会让 `graph.astream()` 在首次模型或工具调用前失败。

ChatService 不调用 Router，也不生成或注入 decision。`route_node` 是生产代码中确定性 Router 的唯一所有者；正常 `/chat`、评测脚本和其他 `run_agent_langgraph()` 调用都会先经过它一次。Router 只决定 `agent`、`direct_answer` 或 `clarify`，具体工具始终由 `create_agent` 根据请求可见 Schema 选择。

`search_knowledge_base` 与普通工具一样由 Agent Model 根据动态 Schema 选择，并经过相同的 Registry/Runtime 治理链。成功结果通过 `Command.PARENT` 和 `pending_rag_result` 回到父图，再进入原 RAG 子图。不要把它移回 Router 或绕过 Runtime。

`web_search` 也是动态普通工具，但成功结果只形成 ToolMessage 并回到 Agent Model，不写 `pending_rag_result`。它通过 DDGS 查询公开互联网；本地受控证据仍必须使用 `search_knowledge_base`。

`web_fetch` 与 `web_search` 一样走普通 ToolMessage 路径：Agent 可先搜索，再从搜索结果选择一个公开 URL 抓取正文。HTML 正文通过 Trafilatura 提取；它不进入 RAG、不持久化网页，也不生成本地 references。

`web_pdf_fetch` 同样是普通 ToolMessage 路径，负责已知公开 PDF 的有限页范围。HTML 与 PDF 共用 `app/tools/web_download.py` 中的 requests-hardened 下载入口；PDF 使用已有 pypdf 提取带 `[Page N]` 边界的文本。它不进入本地 RAG、不持久化 PDF，也不生成本地 references。模型误用 `web_fetch` 读取 PDF 时，可根据内容类型错误 ToolMessage 在同一 `create_agent` 循环中改用 `web_pdf_fetch`，不需要 Router 或父图特判。

`list_sql_tables`、`describe_sql_table` 和 `query_sql_readonly` 是默认关闭的普通
ToolMessage 路径，只在 SQL 配置启用时注册，并统一要求 `database:read`。Registry
handler 只调用 `app/tools/sql_readonly/` Service；sqlglot AST、SQLAlchemy Engine/
Inspector、SQLite 数据库层只读和输出截断不进入 Router、父图、ChatService 或 SSE
projector。数据库内容始终是不可信数据。详见[只读 SQL 工具](tools/sql-readonly.md)。

## 配置快照与生命周期

`app.config.AppSettings` 是唯一完整应用配置模型。它基于官方
`pydantic-settings`，字段使用 lower snake case、实例被冻结；已有 uppercase
环境变量名称保持不变。模块导入不实例化配置，生产入口在请求或 lifespan 边界调用
惰性缓存的 `get_settings()`。来源优先级是显式初始化参数、环境变量、项目根
`.env`、字段默认值。

配置解析不创建目录、模型、Registry、Runtime、数据库或 Checkpointer。FastAPI
lifespan 取得一个快照后，先调用 `validate_startup_environment(settings)` 检查
API key、知识库目录和持久化父目录，再用同一个实例初始化 RAG 与 Graph。测试和
inspection 使用 `AppSettings(_env_file=None, ...)`，不会读取维护者 `.env`。

以下缓存都可能持有配置或由配置派生的资源：

| 缓存 | 创建时机 | 受控清理入口 |
| --- | --- | --- |
| Settings | 首次 `get_settings()` | `clear_settings_cache()` |
| 角色模型 | 首次对应 Model Factory getter | `clear_model_factory_cache()` |
| 默认 Tool Registry | 首次 `get_tool_registry()` | `clear_tool_registry_cache()` |
| Tool Runtime | 首次 dispatch / `get_tool_runtime()` | `reset_tool_runtime()` |
| SQL Engine | 第一次实际 SQL 工具调用 | `clear_sql_engine_cache()`；lifespan shutdown 自动 dispose |
| Graph / Checkpointer | 同步 Graph 在 lifespan 预编译，流式 Graph 首次请求创建 | 同步上下文使用 `reset_agent_graph_runtime()`，应用异步关闭使用 `areset_agent_graph_runtime()` |

`.env` 不是动态热更新接口。若测试要切换进程级配置，应先关闭 Runtime 和
Graph/Checkpointer，再依次失效 Model、Registry、Settings 缓存；生产环境优先重启
进程。不要恢复模块级 `settings = get_settings()`、全局 `tool_registry` 或可变
Settings 类属性。

## 可执行模块边界

`.importlinter` 是生产包依赖图的权威合同，完整规则和新增模块放置指南见
[模块依赖边界](architecture/module-boundaries.md)。主方向是：

```text
api → services → graph → agent → tools → rag → llm
    → storage/observability → contracts/config/common
```

API 是组合根，可以跨层组装依赖；所有低层都不能反向依赖 API。FastAPI 与
Starlette 只能由 `app.api` 使用，HTTP middleware 位于
`app/api/middleware.py`。`app/api/http_logging.py` 独占 Uvicorn access-log
适配，并由 lifespan 显式配置；`app.common.logging` 只创建框架无关 logger。

`pydantic_settings` 只能由 `app.config` 直接导入。其他模块依赖
`AppSettings`、`get_settings()` 或明确 config helper，因此 Settings 外部包合同使用
Import Linter 原生的 `allow_indirect_imports = True`：允许经过 `app.config` 的合法
间接链，但仍禁止其他模块直接绕过配置门面。合同没有 `ignore_imports`。

## 单一运行状态

`AgentGraphState` 是父图唯一的完整请求状态。节点不再取出嵌套业务对象后原地修改；initialize、框架事件投影、RAG 映射、compose 和 finalize 都返回明确的 LangGraph 字段 updates。

| 分组 | 字段 |
| --- | --- |
| 请求输入 | `original_question`、`effective_question`、`top_k`、`decompose_top_k`、`trace_id`、`session_id` |
| 路由 | `decision`、`route_cost_ms` |
| 框架治理 | `messages`、`governance_events`、`framework_events_applied`、迭代/工具/错误计数与指纹、`agent_elapsed_ms`、`stop_reason`、`pending_rag_result` |
| 业务结果 | `observations`、`final_answer`、`references`、`status`、`reason`、`decompose_llm_used`、`answer_llm_used`、`llm_call_count`、`tool_trace` |
| 最终输出 | `result` |

只有框架需要合并的 `messages` 和 append-only `governance_events` 使用 reducer。`observations`、`references` 和 `tool_trace` 由节点返回完整新列表，避免 checkpoint resume 或节点重放时重复追加；`llm_usage` 在 finalize 中执行 upsert。改写问题需要持久化到 trace 时，由 `effective_question != original_question` 派生，不在状态中重复保存。

## 单一结果契约与状态策略

父图内部状态、Runtime 结果和应用层响应是不同职责，固定转换链如下：

```text
AgentGraphState.status
→ map_agent_status_to_result_status
→ AgentRunResult
→ model_dump(mode="json")
→ ChatStatusPolicy
→ ChatServiceOutcome / API data.status / metrics status
```

RAG 子图协议不变；`map_rag_status_to_agent_status()` 先把 `OK`、
`NEED_CLARIFICATION`、`NO_RELEVANT_INFO`、`MODEL_ERROR` 和
`MODEL_BAD_OUTPUT` 收敛到内部 Agent 状态。`MODEL_BAD_OUTPUT` 在这里是 RAG 的真实
输入状态，但不是 Chat Result status。

`AgentRunResult` 是唯一完整 Runtime 结果模型，拒绝额外字段、非法状态、越界
confidence、负 LLM 计数以及非字典 references/tool trace。正常 finalize 和所有内部
错误结果都经过该模型。Pydantic 实例只用于边界校验，写入
`AgentGraphState.result`、Checkpoint 和返回 ChatService 的值均为普通字典。

`CHAT_STATUS_POLICIES` 是 HTTP status、code/message、公开小写状态、metrics
状态和错误字段暴露策略的唯一来源。未知 Result status 解析为
`INTERNAL_ERROR`，不会按成功处理。`EMPTY_QUESTION` 是服务级 400 兼容分支；
ChatService 调用 Runtime 本身抛出的未知异常继续保留公开
`code=MODEL_ERROR` 的历史兼容行为，但其完整 result 和 metrics 都按内部错误构造。

## 当前图

以下 Mermaid 由 `python scripts/inspect_agent_runtime.py --format mermaid` 的 `compiled_graph.get_graph(xray=True).draw_mermaid()` 输出生成。升级框架或修改图后应重新运行命令并替换本段，不要手工维护第二套图。

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	route(route)
	initialize(initialize)
	adapt_framework(adapt_framework)
	rag_subgraph(rag_subgraph)
	compose(compose)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> route;
	adapt_framework -.-> compose;
	adapt_framework -.-> rag_subgraph;
	compose --> finalize;
	framework_agent\3aProjectToolGovernanceMiddleware\2eafter_agent --> adapt_framework;
	initialize -.-> compose;
	initialize -.-> framework_agent\3a__start__;
	rag_subgraph --> finalize;
	route --> initialize;
	finalize --> __end__;
	subgraph framework_agent
	framework_agent\3a__start__(<p>__start__</p>)
	framework_agent\3amodel(model)
	framework_agent\3atools(tools)
	framework_agent\3aProjectToolGovernanceMiddleware\2ebefore_model(ProjectToolGovernanceMiddleware.before_model)
	framework_agent\3aProjectToolGovernanceMiddleware\2eafter_agent(ProjectToolGovernanceMiddleware.after_agent)
	framework_agent\3aModelCallLimitMiddleware\2ebefore_model(ModelCallLimitMiddleware.before_model)
	framework_agent\3aModelCallLimitMiddleware\2eafter_model(ModelCallLimitMiddleware.after_model)
	framework_agent\3aToolCallLimitMiddleware\2eafter_model(ToolCallLimitMiddleware.after_model)
	framework_agent\3aModelCallLimitMiddleware\2eafter_model -.-> framework_agent\3aProjectToolGovernanceMiddleware\2eafter_agent;
	framework_agent\3aModelCallLimitMiddleware\2eafter_model -.-> framework_agent\3aProjectToolGovernanceMiddleware\2ebefore_model;
	framework_agent\3aModelCallLimitMiddleware\2eafter_model -.-> framework_agent\3atools;
	framework_agent\3aModelCallLimitMiddleware\2ebefore_model -.-> framework_agent\3aProjectToolGovernanceMiddleware\2eafter_agent;
	framework_agent\3aModelCallLimitMiddleware\2ebefore_model -.-> framework_agent\3amodel;
	framework_agent\3aProjectToolGovernanceMiddleware\2ebefore_model -.-> framework_agent\3aModelCallLimitMiddleware\2ebefore_model;
	framework_agent\3aProjectToolGovernanceMiddleware\2ebefore_model -.-> framework_agent\3aProjectToolGovernanceMiddleware\2eafter_agent;
	framework_agent\3aToolCallLimitMiddleware\2eafter_model -.-> framework_agent\3aModelCallLimitMiddleware\2eafter_model;
	framework_agent\3aToolCallLimitMiddleware\2eafter_model -.-> framework_agent\3aProjectToolGovernanceMiddleware\2eafter_agent;
	framework_agent\3a__start__ --> framework_agent\3aProjectToolGovernanceMiddleware\2ebefore_model;
	framework_agent\3amodel --> framework_agent\3aToolCallLimitMiddleware\2eafter_model;
	framework_agent\3atools -.-> framework_agent\3aProjectToolGovernanceMiddleware\2ebefore_model;
	end
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

`RegistryToolsMiddleware` 使用 wrap hook 动态覆盖模型和工具请求，不会作为独立图节点出现；它是否仍正确提供工具由检查脚本的默认工具与 Schema 契约覆盖。

## 文件职责

| 文件 | 唯一主要职责 |
| --- | --- |
| `app/config.py` | 声明冻结 AppSettings、纯校验、computed fallback 和惰性缓存 |
| `app/graph/builder.py` | 声明父图节点和边 |
| `app/graph/state.py` | 声明唯一完整 `AgentGraphState` 及 reducer |
| `app/graph/conditions.py` | 声明父图无副作用分支条件 |
| `app/graph/nodes.py` | 独占 Router 调用/计时；组合各节点的显式字段更新 |
| `app/graph/runtime.py` | 共享调用准备与同步 `graph.invoke` 门面 |
| `app/graph/streaming.py` | 异步 `graph.astream`、消息缓冲和安全事件投影 |
| `app/contracts/chat.py` | 声明唯一 Result-to-HTTP/API/metrics 状态策略 |
| `app/agent/router.py` | 只产生 agent/direct_answer/clarify 的确定性决策 |
| `app/agent/results.py` | 校验完整 AgentRunResult；集中内部和 RAG 状态转换 |
| `app/agent/framework_agent.py` | 组装 `create_agent`；纯投影框架事件 |
| `app/agent/tool_events.py` | 将工具结果转换为 observation 和兼容 trace |
| `app/agent/answer.py` | 生成 RAG/compose 的 pure updates |
| `app/agent/framework_middleware.py` | 执行项目特有 Agent 治理 |
| `app/agent/framework_tools.py` | 适配动态工具；处理 RAG handoff |
| `app/tools/providers/local.py` | 按请求 `required_permissions` 过滤模型可见工具 |
| `app/tools/registry.py` | 注册 ToolSpec；统一分发工具 |
| `app/tools/runtime.py` | 治理并执行单次工具调用 |
| `app/tools/web_search.py` | 调用 DDGS；映射参数、规范化结果和转换公开异常 |
| `app/tools/web_download.py` | 为 HTML/PDF 提供共用 hardened 有限下载适配 |
| `app/tools/web_fetch.py` | 处理 HTML/XHTML/纯文本并调用 Trafilatura |
| `app/tools/web_pdf_fetch.py` | 校验 PDF magic；通过 pypdf 提取页范围和有限元数据 |
| `app/tools/sql_readonly/engine.py` | 惰性 SQLAlchemy Engine、SQLite 只读连接、语句超时和 dispose |
| `app/tools/sql_readonly/policy.py` | sqlglot 单语句只读 AST、物理表 allowlist、危险函数和 LIMIT |
| `app/tools/sql_readonly/service.py` | Inspector、查询事务边界、安全错误和结果转换 |
| `app/tools/sql_readonly/serialization.py` | 列名/值规范化与单元格、行、总输出限制 |
| `app/agent/runtime.py` | 构造兼容 decide/LLM trace；保存 trace |
| `app/services/chat_models.py` | 声明框架无关 Chat 请求、结果和内部上下文 |
| `app/services/chat_stream_models.py` | 声明七种框架无关 Chat stream 事件 |
| `app/services/chat_service.py` | 编排同步/流式 Query Rewrite、Runtime、状态策略和 session |
| `app/services/chat_telemetry.py` | 初始化/合并/finalize metrics，安全写 bad case，格式化工具调用链 |
| `app/api/application.py` | 创建 FastAPI；组装 lifespan、middleware 和全部 Router |
| `app/api/chat.py` | 把 `/chat` 与 `/chat/stream` HTTP 上下文适配为同一 Service |
| `app/api/streaming.py` | 严格公开字段投影、sequence/timestamp 与 SSE wire 编码 |
| `app/api/health.py` | 提供无外部依赖的进程存活检查 |
| `app/api/lifecycle.py` | 校验并初始化 RAG/Graph；关闭 Graph/Tool Runtime |
| `app/api/middleware.py` | 维护两个聊天入口的 trace；只为同步 `/chat` 写总耗时响应头 |
| `app/api/http_logging.py` | 显式、幂等配置 Uvicorn access-log URL 展示 |
| `app/api/responses.py` | 沿单一具体函数规范化 confidence/LLM usage 并构造 ChatResponse |
| `app/common/logging.py` | 构造框架无关的应用 logger，不修改 Uvicorn 全局状态 |
| `app/observability/metrics.py` | 从 decide/tool/RAG trace 汇总稳定 metrics，不映射状态 |
| `app/rag/graph/` | 编排检索、证据、生成和引用流程 |

## 框架与项目边界

框架负责 `model ↔ tools` 循环、Pydantic 参数校验、基础 ToolMessage、`ModelCallLimitMiddleware` 和 `ToolCallLimitMiddleware`。项目 Middleware 只负责框架无法判断的 ToolErrorCode 语义、重复调用/错误、累计错误、总耗时和 terminal 同批 fail-closed。

工具执行的唯一受治理路径是：

```text
StructuredTool → ToolRegistry.dispatch → ToolRuntime.execute → ToolSpec.handler
```

Provider 的模型可见性层只复用 `has_permission()` 检查全部 `ToolSpec.required_permissions`，不运行完整 Policy。审批、幂等键、主体、来源和风险仍是执行期治理，因此权限满足的审批工具和写工具仍可见。项目 `ToolRuntime` 独占最终权限、策略、审批、审计、幂等、超时、有限重试和输出校验；可见性过滤和框架 Middleware 都不能替代或复制这些能力。

## 常见问题定位

| 问题 | 首先查看 |
| --- | --- |
| 状态字段在哪里定义 | `app/graph/state.py` |
| 内部状态为什么变成某个 Result status | `app/agent/results.py` |
| 某 Result status 的 HTTP/code/message 是什么 | `app/contracts/chat.py` |
| metrics 状态为什么这样记录 | `ChatStatusPolicy.metric_status` |
| API 字段如何构造 | `app/api/responses.py` |
| SSE sequence/信封为什么这样构造 | `app/api/streaming.py` |
| 为什么某个 token 没有流出 | `app/graph/streaming.py` 的消息类型/block/namespace/metadata 白名单 |
| 客户端断开为何不保存部分回答 | `app/api/chat.py`、`ChatService.stream()` 的 generator close/finalize |
| RAG 状态如何进入 Agent | `app/agent/results.py::map_rag_status_to_agent_status` |
| 工具事件如何进入状态 | `app/agent/framework_agent.py`、`app/agent/tool_events.py` |
| RAG 结果如何写回 | `app/agent/answer.py`、`rag_subgraph_node` |
| 最终结果如何构建 | `app/agent/results.py`、`finalize_node` |
| Checkpoint 保存什么 | `AgentGraphState` channels |
| Graph 为什么走这个分支 | `app/graph/builder.py`、`app/graph/conditions.py` |
| 请求为什么进入 direct/clarify/agent | `app/agent/router.py`、`app/graph/nodes.py::route_node` |
| Router 为什么执行两次 | 不应发生；检查调用方是否绕开 `run_agent_langgraph()` 或重复执行父图 |
| Router 耗时 | `decide` trace 的 `route_cost_ms`、`collect_trace_metrics()` |
| 模型为什么选择工具 | `app/agent/framework_tools.py`、ToolSpec description/Pydantic Schema 和 Agent Model |
| 工具为什么失败 | `app/tools/runtime.py`、Policy 和对应 ToolSpec handler |
| 模型为什么看不到某个工具 | `app/tools/providers/local.py`、`ToolSpec.required_permissions` |
| Web Search 为什么模型看不到 | `TOOL_DEFAULT_PERMISSIONS`、`app/tools/providers/local.py` |
| SQL 工具为什么模型看不到 | `SQL_TOOL_ENABLED`、`SQL_ALLOWED_TABLES`、`database:read` 与 Provider 可见性 |
| Web Search 为什么执行失败 | `app/tools/web_search.py`、`app/tools/runtime.py` |
| 搜索结果为什么为空 | DDGS region、`backend="auto"`、网络条件 |
| Web Search 为什么没有进入 RAG | 它是普通 Agent 工具；只有本地知识库检索进入 RAG |
| 如何更换搜索后端 | 保持 `app/tools/web_search.py` 的开源后端适配边界和 ToolSpec 契约 |
| 模型为什么看不到 `web_fetch` / `web_pdf_fetch` | `app/tools/providers/local.py`、`web:fetch` 权限 |
| 为什么 URL 被拒绝 | requests-hardened 配置和 `app/tools/web_download.py` 的 URL 业务规则 |
| 为什么没有正文 | Trafilatura 提取结果和页面实际可读正文 |
| 为什么页面过大 | `WEB_FETCH_MAX_RESPONSE_BYTES` |
| 为什么 `web_fetch` 拒绝 PDF | HTML 与 PDF 使用独立输出协议；PDF 应调用 `web_pdf_fetch` |
| 为什么 PDF 无文字 | 选定范围可能是扫描件/图片型 PDF；PR8 不支持 OCR |
| 为什么 PDF 被拒绝 | Content-Type/magic、加密、损坏、`WEB_PDF_MAX_RESPONSE_BYTES` 或 `WEB_PDF_MAX_DOCUMENT_PAGES` |
| 为什么 Web Fetch 没有 RAG references | `web_fetch` 是普通工具路径，不进入本地 RAG |
| 为什么 Web PDF 没有 RAG references | `web_pdf_fetch` 是普通工具路径，不写入本地知识库 |
| 工具可见但为什么执行被拒绝 | `app/tools/policy.py`、`app/tools/runtime.py` |
| 工具 trace 为什么这样生成 | `app/agent/tool_events.py` |
| ToolMessage 为什么异常 | `app/agent/framework_middleware.py`、`app/agent/framework_tools.py` |
| 模型失败为什么没有回退 | `app/agent/framework_middleware.py`、`adapt_framework_agent_node` |
| 知识库为什么进入 RAG | `Command.PARENT`、`rag_subgraph_node` |
| 知识库为什么没有引用 | `app/rag/graph/`、`app/rag/citations.py` |
| API 为什么返回某状态 | `app/contracts/chat.py`、`app/services/chat_service.py` |
| 图为什么创建 Checkpoint 文件 | `app/graph/checkpointing.py`、`app/graph/runtime.py` |
| 核心文件为何触发 CI | `scripts/inspect_agent_runtime.py` 的规模预算 |

## 新增普通工具

1. 定义 Pydantic Args Model。
2. 定义接收已校验参数和 `ToolInvocationContext` 的 handler。
3. 创建合法 `ToolSpec`，填写版本、描述、Schema 与治理元数据。
4. 注册到相应 `ToolRegistry`。
5. 增加工具行为、权限和错误结果测试。

正常情况下不需要修改 Router、Parent Graph、`create_agent` 或 Middleware。新增权限要求后应分别测试无权 Context 不可见、有权 Context 可见，以及直接 dispatch 的 Runtime 最终拒绝。模型可见 Schema 来自 StructuredTool/Pydantic，检查脚本不会维护第二份参数 Schema。

## 不应该做什么

- 不在 `nodes.py` 再实现工具循环。
- 不在 Router 硬编码普通工具意图。
- 不绕过 Registry 或 ToolRuntime 调用 handler。
- 不在 Provider 调用完整 Policy Engine 或复制权限字符串比较。
- 不重新手写框架已经生成的基础 ToolMessage。
- 不实现第二套参数 Schema。
- 不为框架已有能力创建自定义 Runtime。
- 不同时维护两套 Agent Loop。
- 不在父图状态中嵌套另一套完整业务状态。
- 不把 `AgentRunResult` 实例写入 Graph State 或 Checkpoint。
- 不在 Runtime、ChatService、API 或 metrics 中另建状态映射表。
- 不让 API response adapter 导入 Agent Runtime 或 Graph。
- 不让 API 以外的生产包导入 FastAPI、Starlette 或承载 Uvicorn 适配。
- 不让 `app.config` 以外的生产模块直接导入 `pydantic_settings`。
- 不用函数内 import、`importlib` 或 `ignore_imports` 隐藏依赖方向。
- 不在节点或 helper 中原地修改输入的 observations、references 或 tool_trace。
- 不从 `builder.py` 源码、AST 或正则反推图结构。

## 检查与升级

```bash
python scripts/inspect_agent_runtime.py
python scripts/inspect_agent_runtime.py --format json
python scripts/inspect_agent_runtime.py --format mermaid
python scripts/inspect_agent_runtime.py --check
lint-imports
lint-imports --verbose
```

脚本只使用普通 `build_agent_graph().compile()`，不会调用带 Checkpointer 的缓存入口；也不会初始化模型、DDGS、requests-hardened Manager、Trafilatura、PdfReader、执行 handler、访问网络或读取/重建 RAG 索引。报告验证 `AgentGraphState` 没有嵌套状态且包含全部业务字段，`AgentRunResult` 字段完整并拒绝 extra，所有 Result status 恰好有一项 policy，Graph result 仍是普通字典，API response adapter 没有 Runtime/Graph 反向依赖，并检查 application factory、同步/流式 Chat Router、共享 Service、`graph.astream` Runtime、事件合同、ChatService/telemetry 分工、当前架构文档和历史归档。报告还分别列出 Registry 的八个 `registered_tools` 和默认 Context 的 `default_visible_tools`，验证默认可见集合、权限及 Web 工具契约。它只读取 `.importlinter` 的公开配置结构，报告 `import_linter_configured`、四项关键合同和 `zero_import_ignores`；不分析或复制 import graph。版本来自 `importlib.metadata`，图来自 `get_graph()`/`get_graph(xray=True)`，Mermaid 来自 `draw_mermaid()`，工具参数来自模型可见的 Pydantic JSON Schema。实际模块方向必须由官方 `lint-imports` 全量检查。

升级 LangChain、LangGraph、Pydantic 或调整 package 时，先运行 `lint-imports`、契约测试和 `--check`。重点确认父图节点未改变、xray 仍能展开 `framework_agent`、model/tools 和调用限制节点仍存在、`ToolRuntime` 注入字段未暴露给模型，并重新生成本页 Mermaid。

真实 Online Tool 连通性统一使用 `python scripts/smoke_test_online_tools.py`
的 `search`、`fetch`、`pdf` 子命令，三者都通过
`ToolRegistry.dispatch → ToolRuntime.execute`。模型供应商 Tool Calling 协议仍由
`smoke_test_agent_tool_calling.py` 单独检查；两类冒烟不得合并，也不进入 Pytest。

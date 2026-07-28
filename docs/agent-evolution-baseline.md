# Agent Evolution Baseline

## 1. 基线信息

本基线冻结 PR0 开始时的代码、测试、质量检查和架构事实，不改变业务行为。

| 项目 | 基线值 |
| --- | --- |
| 检查日期 | 2026-07-19 |
| Git 分支 | `fix/langgraph-runtime-hardening` |
| Git commit | `1f1180197d7aee165e99832df07755a8776fa0b6` |
| Python | 3.12.3 |
| Python 环境 | 仓库 `.venv` |

## 2. 当前系统定位

当前项目是一个面向中文制度类文档的 Agentic RAG 系统，已经具备 LangGraph 编排、Router、Planner、Executor、RAG 子图、工具注册、统一 `ToolRuntime`、权限审计、可选 LangGraph Checkpointer、SQLite 会话/轨迹存储和离线评测能力。当前 Agent 主要仍是固定计划驱动的工作流，尚未形成模型—工具—观察—重新决策的迭代式 Agent Loop。

模型通过 LangChain `ChatOpenAI` 访问 OpenAI-compatible Chat Completions 接口。API 主协议、RAG 检索与引用算法和工具业务实现彼此保持明确边界。

## 3. 当前 Agent 主流程

外层请求链路为：

```text
FastAPI /chat
→ ChatService
→ SQLite Session History / Query Rewrite
→ Router
→ LangGraph Agent Graph
→ RAG Subgraph / 普通工具
→ Finalize
→ Session / Metrics
→ API Response
```

LangGraph 主图为：

```text
START
→ route（仅未传入 decision 时）
→ initialize
→ plan
→ execute
→ 成功的知识库检索：rag_subgraph → finalize → END
→ 其他结果：compose → finalize → END
```

`ChatService` 当前会把已经生成的 Router decision 传给主图，因此正常 `/chat` 请求会跳过图内 `route` 节点，但仍只执行一次 Router。Planner 按具体 action 生成固定计划；工具 action 通常映射为 `tool_call → final_answer`。Executor 顺序执行计划，工具失败时立即中止，工具成功后也不会回到模型重新决策。成功的 `search_knowledge_base` observation 进入 RAG 子图；其他工具、澄清、直接回答和工具错误进入 Answer Composer。

当前主图没有 `messages` 工作轨迹、模型 `bind_tools()`、LangGraph `ToolNode`、工具调用后的循环边、Streaming 或 Interrupt。

## 4. 当前内置工具

以下内容由当前 `ToolRegistry` 的契约核实。权限列为“无”时不要求权限；所有工具当前均不要求审批、均声明幂等且不启用自动重试。

| 名称 | 用途 | 风险等级 | 所需权限 | 审批 | 超时 | 幂等 |
| --- | --- | --- | --- | --- | ---: | --- |
| `search_knowledge_base` | 检索知识库并返回相关证据 chunk | `READ_ONLY` (`read_only`) | `knowledge_base:read` | 否 | 10 秒 | 是 |
| `compare_documents` | 对比两份知识库文档关于同一问题的证据和差异 | `READ_ONLY` (`read_only`) | `knowledge_base:read` | 否 | 20 秒 | 是 |
| `simple_calculator` | 执行简单数字四则运算 | `READ_ONLY` (`read_only`) | 无 | 否 | 3 秒 | 是 |
| `list_documents` | 列出知识库文档及数量 | `READ_ONLY` (`read_only`) | `knowledge_base:read` | 否 | 5 秒 | 是 |
| `get_chunk_text` | 查看指定 chunk_id 的完整原文 | `READ_ONLY` (`read_only`) | `knowledge_base:read` | 否 | 5 秒 | 是 |

五个工具版本均为 `1.0.0`，`max_retries` 均为 0。`get_chunk_text` 的审计输出模式为 `none`，不会把 chunk 正文写入工具审计摘要。

## 5. 当前工具治理能力

- `ToolSpec` 统一契约和注册期一致性校验。
- Pydantic 输入参数校验及可选输出模型校验。
- 工具版本、四级风险、精确权限、审批要求、幂等性、超时和有限重试配置。
- 非幂等工具禁止自动重试；运行时支持指数退避和抖动。
- `ToolRegistry` 统一注册、查询与分发，并拒绝同名工具静默覆盖。
- `ToolRuntime` 在线程池提交 handler 前执行策略检查。
- 参数指纹、结构化审计生命周期、可注入审计 sink 和统一错误码。
- 敏感字段递归脱敏、循环引用防护与审计体量限制。
- 按工具选择不记录正文、有限摘要或脱敏输出。
- 受限共享线程池、每工具超时、运行尝试/耗时元数据和关闭生命周期。
- 请求级不可变 `ToolInvocationContext`，包含 trace、session、主体、权限、来源和幂等键。
- 当前写工具策略会检查非匿名主体、主体标识、调用来源、精确权限和幂等键；外部写入与破坏性操作返回审批要求。执行前 `requested` 审计失败时写工具 fail-closed。

当前主调用链为：

```text
Planner / Executor
→ LangChain StructuredTool
→ dispatch_tool
→ ToolRegistry.dispatch
→ ToolRuntime.execute
→ ToolSpec.handler
```

审批策略门已存在，但尚无人工批准后恢复 handler 执行的流程。

## 6. 当前 RAG 能力

- 读取本地 `.txt`、`.md`、`.pdf` 和 `.docx` 文档，清洗并按结构切分 chunk。
- 以 BM25 和 SentenceTransformer Embedding 做混合召回。
- 使用 CrossEncoder reranker 进行二阶段重排。
- 使用 ChromaDB 持久化向量、chunk 和 metadata，并通过 manifest 判断索引状态。
- 支持可选 Query Decompose；复杂问题可拆分后多路召回并合并证据。
- RAG 子图依次完成问题歧义判断、拆分、检索/重排、Evidence Gate、上下文构造、模型生成、JSON 解析、引用绑定与验证。
- 通过检索分数、引用绑定和启发式置信度执行证据不足拒答。
- 通过 SQLite 会话历史和 Query Rewrite 支持短追问与指代改写。
- 提供预测生成、规则/检索评测、可选 LLM Judge 和 Agent 路由评测入口。

LangGraph Checkpointer 支持 `disabled`、`memory` 和 `sqlite` 三种模式，默认是 `disabled`。SQLite 模式保存图节点快照，但不替代应用级 SQLite 会话历史，也尚未实现自动恢复、Replay、Time Travel、Interrupt 或管理 API。

## 7. 当前测试基线

在仓库 `.venv` 中执行：

```text
pytest -q
415 passed, 1 warning in 12.31s
```

| Pytest 指标 | 数量/结果 |
| --- | ---: |
| 收集并通过 | 415 |
| 失败 | 0 |
| 跳过 | 0 |
| 警告 | 1 |
| 命令状态 | 成功（exit code 0） |

唯一警告来自 `.venv` 中 `jieba/_compat.py` 使用已弃用的 `pkg_resources` API，不是当前项目测试失败。

静态检查结果：

```text
ruff check .
All checks passed!
```

Ruff 成功（exit code 0）。

README 还记录了 `compileall`、RAG 预测/评测、Agent 评测、性能测试和本地模型接口检查入口。本次未运行真实离线评测与性能任务，因为它们可能依赖本地索引、模型/API 配置，可能调用外部模型并产生输出或费用；对应 mock 与回归测试已包含在上述 Pytest 基线中。

## 8. 当前依赖版本

以下版本来自 `requirements.lock`：

| 依赖 | 版本 |
| --- | --- |
| `fastapi` | 0.135.1 |
| `pydantic` | 2.12.5 |
| `langchain-core` | 1.4.9 |
| `langchain-openai` | 1.3.5 |
| `langgraph` | 1.2.9 |
| `langgraph-checkpoint-sqlite` | 3.1.0 |
| `chromadb` | 1.3.7 |
| `sentence-transformers` | 5.2.3 |
| `pytest` | 9.0.2 |
| `ruff` | 0.15.21 |

## 9. 已知限制

- 主图没有工具调用后的模型循环。
- Planner 主要是 Router action 到固定计划的映射。
- Router 和 Planner 对具体工具及参数处理存在硬编码。
- 工具失败通常会中止计划，模型无法根据失败观察修正参数或改选工具。
- 当前工具以知识库读取和简单计算为主，全部为只读工具。
- 暂无外部网络搜索工具。
- 暂无只读数据库工具。
- 暂无 MCP Provider。
- 暂无 LangGraph interrupt 审批恢复流程；现有审批能力只会在 handler 前返回策略结果。
- 暂无通用 Agent 工具轨迹评测；现有 `agent_eval.py` 主要覆盖 Router、动作、工具结果字段和端到端结果。
- SQLite Checkpointer 适合单进程本地开发，尚无生产级分布式 Checkpointer、自动恢复或数据清理能力。
- 工具超时依赖线程 Future 的等待超时，不能强制终止已经开始运行的底层 Python handler。

## 10. PR0 不包含的内容

- Agent Loop 实现。
- Router 重构。
- Planner 重构。
- Executor 错误恢复。
- 新工具。
- MCP。
- SQL。
- Web Search。
- 人工审批。
- 多 Agent。
- 沙箱代码执行。

PR0 不修改生产代码、API Schema、Agent 执行流程、工具调用逻辑或依赖版本。

## 11. 下一阶段入口

下一阶段为：

```text
PR1：feat/iterative-agent-loop
```

目标调用链为：

```text
模型选择工具
→ ToolRuntime 执行
→ ToolMessage 返回模型
→ 模型继续选择工具或生成最终答案
```

PR1 最适合从 `app/graph/state.py` 开始，先定义 messages、循环计数和预算状态；随后修改 `app/graph/nodes.py` 与 `app/graph/builder.py`，增加模型节点、受治理工具节点和循环条件。`app/tools/langchain_tools.py` 应继续只做适配，所有调用仍必须进入现有 Registry/Runtime 治理链。

## 12. PR1 实现增量（2026-07-19）

本文件第 1–11 节继续保留 PR0 的历史基线；PR1 在该基线上完成了以下增量：

- 主图状态新增 LangChain `messages` reducer、模型轮次、工具调用数、工具错误数和停止原因。
- 主路径新增 `agent_model → governed_tools → agent_model` 循环，支持无工具直接回答、顺序多工具、同轮多个 `tool_calls` 和普通错误后的参数修正。
- 每个工具结果以关联模型 `tool_call_id` 的标准 `ToolMessage` 返回，并保留 Runtime、governance、错误码、摘要及 Runtime 工具调用 ID。
- `governed_tools` 动态创建请求级 StructuredTool，继续经过统一 `dispatch_tool → ToolRuntime → handler` 链；没有把业务 handler 暴露给模型。
- 模型轮次和工具调用总数分别由 `AGENT_MAX_ITERATIONS`（默认 6）与 `AGENT_MAX_TOOL_CALLS`（默认 10）限制，耗尽时受控进入 Finalize。
- 初始知识库路由以及循环中成功的 `search_knowledge_base` 都继续进入专用 RAG 子图，引用绑定、证据门控和拒答算法未改动。
- Planner/Executor、Router 的工具硬编码和 `AGENT_MAX_STEPS` 保留为知识库/澄清兼容路径及首轮模型不可用时的降级能力。
- SQLite Checkpointer 已验证可以序列化完整的 `HumanMessage → AIMessage → ToolMessage → AIMessage` 轨迹。

PR1 没有实现工具错误次数上限、总耗时预算、连续相同错误检测、Streaming、Interrupt、人工审批恢复、多 Agent 或新工具；这些仍属于后续演进范围。

## 13. PR2 实现增量（2026-07-20）

本文件第 1–11 节是 PR0 历史基线，第 12 节记录 PR1；PR2 在此基础上完成 Registry-driven 工具发现：

- 新增 `ToolProvider` Protocol、`LocalToolProvider`、默认 Provider 工厂和多 Provider ToolSpec 合并/重名检测函数。
- Agent Model 和 `governed_tools` 节点通过可替换的 `get_agent_tool_provider()` 获得同一请求可用工具；测试可以注入独立 Registry，不修改生产全局 Registry。
- LangChain Adapter 从 Provider 的 ToolSpec 自动生成 StructuredTool，并校验定义属于分发 Registry。执行链改为 `StructuredTool → ToolRegistry.dispatch → ToolRuntime → handler`，治理能力没有复制或绕过。
- Router 只输出 `agent`、`rag_fast_path`、`direct_answer`、`clarify`，不再接收所有工具 Schema，也不再解析或校验普通工具参数。正常自然语言请求默认进入 `agent`，避免 Router 在看不到动态工具时按关键词或口吻提前误分流。
- Planner 不参与正常 `agent` 路径；旧具体工具 decision 由一个通用 Registry 兼容分支处理。Executor 与 Graph 无需为普通新工具增加分支。
- 显式/旧协议 `rag_fast_path` 继续使用现有知识库检索工具、Planner/Executor 兼容入口和 RAG 子图；正常 Agent 调用知识库检索后也进入同一子图，检索、重排、证据门控、拒答及引用算法未修改。
- Provider 当前不按权限过滤模型可见性；Runtime 仍对每次调用执行最终权限校验，缺权 handler 不会运行。
- 新增的 `echo_text` 与 `join_texts` 只存在于测试 Registry。它们覆盖独立参数 Schema、Context 贯穿、未知工具恢复和请求隔离，生产 Registry 仍只有原五个工具。

PR2 开始前基线为 `434 passed, 1 warning` 且 Ruff 通过。PR2 完成后的回归基线为 `444 passed, 1 warning`；唯一警告仍来自 `jieba` 对 `pkg_resources` 的弃用提示。没有调用真实外部模型，也没有修改 API Schema、SQLite 表结构、依赖版本或 RAG 算法。

当前仍未实现多个生产 Provider、MCP、Web、SQL、外部工具加载、写工具审批恢复、多 Agent、工具错误次数上限或总耗时预算。旧 Planner/Executor 和 RAG 专用工具名引用作为兼容边界保留。

## 14. PR3 实现增量（2026-07-20）

本文件第 1–11 节继续保留 PR0 历史快照，第 12–13 节记录 PR1/PR2；PR3 增加 Agent 错误分类、预算和循环防护：

- 新增独立 `app/agent/guardrails.py`，定义稳定失败分类、Stop Reason、Agent 专属批次中止码、稳定调用/错误 SHA-256 指纹、连续计数和耗时预算纯函数。未知失败码采用保守终止默认值。
- `AgentGraphState` 新增工具错误上限、连续相同调用/错误状态和 Agent 模型/工具累计整数毫秒；所有字段可由 LangGraph Checkpointer 序列化，不保存 Provider、ToolSpec、模型、异常、traceback 或时钟对象。
- 新增 `AGENT_MAX_TOOL_ERRORS=3`、`AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS=2`、`AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS=2`、`AGENT_MAX_TOTAL_COST_MS=120000`，统一由现有 `validate_agent_loop_config()` 校验。
- 可恢复错误和 Runtime 已重试完的瞬时错误可在预算内返回模型；治理终止、缺少幂等键、待审批和 Runtime 不可用不会返回模型绕过。连续第三次相同调用在 StructuredTool 前拦截；连续第二次相同调用/参数/错误码失败后保存消息并停止。
- 任一终止错误后，同一 AIMessage 的后续工具 fail-closed，不进入 StructuredTool、Runtime 或 handler；后续调用仍按原模型 ID 生成 `agent_batch_aborted` ToolMessage。轮次、调用和耗时 limit 也为所有未执行调用补齐标准 ToolMessage。
- ToolMessage 和 tool trace 增加失败分类、稳定错误来源、handler 是否执行、确定性建议、调用指纹、Stop Reason、连续计数、错误累计数和纯数字预算快照；StructuredTool ValidationError 不包含原始 input 或 traceback。
- `search_knowledge_base` 仍由 Agent Model 统一绑定。严格测试真实经过 `SearchKnowledgeBaseArgs → LocalToolProvider → StructuredTool → ToolRegistry.dispatch → ToolRuntime → handler → RAG 子图`；成功零结果不会增加工具错误数。
- ChatService 主路径测试从 `ChatService.handle()` 进入真实 `run_agent_langgraph` 与编译 Graph，真实经过 Provider、StructuredTool、Registry 和 ToolRuntime，只替换 Agent 模型以及 session/metrics 等外部副作用。
- SQLite Checkpointer 已覆盖普通成功、参数修正、重复调用停止和工具预算停止，所有 AIMessage tool call 均有对应 ToolMessage。
- 新增手动 `scripts/smoke_test_agent_tool_calling.py` 验证真实供应商 Tool Calling；它不进入 Pytest/CI，本次未运行。

PR3 开始前本地实际基线为 `452 passed, 1 warning`（高于远端文档记录的 444）；完成后为 `503 passed, 1 warning`，Ruff 与 `git diff --check` 均通过。唯一警告仍来自 `jieba` 对 `pkg_resources` 的弃用提示。本次自动测试未调用真实外部模型或网络。

PR3 未修改 API Schema、SQLite 表结构、依赖版本或 RAG 检索/重排/引用/拒答算法。正式人工审批、LangGraph interrupt/resume、持久化幂等去重、强制取消正在运行的调用、Streaming、多 Agent、MCP、Web、SQL 和代码沙箱仍未实现。

## 15. PR3R 框架原生 Runtime 重构（2026-07-21）

本节只记录 PR3R 增量；第 1–11 节 PR0 历史快照以及 PR1–PR3 增量保持原样。PR3R 是 behavior-preserving refactor，不增加业务工具或新 API：

- 父图原 `agent_model ↔ governed_tools` 自定义循环迁移为 LangChain `create_agent` 子图。框架内部 model/tools 节点负责循环、Pydantic 参数校验、同轮调用以及基础 ToolMessage 与 `tool_call_id` 对应。
- `RegistryToolsMiddleware` 在模型和工具 hook 中从同一请求 Provider 解析动态工具。StructuredTool 的 `ToolRuntime` 注入字段不会暴露给模型 Schema；适配函数从框架 runtime context 构造原请求语义，并只调用 `ToolRegistry.dispatch()`。
- `ModelCallLimitMiddleware` 和 `ToolCallLimitMiddleware` 接管 `AGENT_MAX_ITERATIONS` 与 `AGENT_MAX_TOOL_CALLS`；父图不再维护重复调用预算。`run_agent_langgraph()` 同时设置 `recursion_limit`，只作为图级最终兜底。
- `ProjectToolGovernanceMiddleware` 只保留项目错误码分类、累计错误、重复调用/错误、总耗时和 terminal 同批 fail-closed。Runtime 超时、重试、权限、审批、审计和幂等策略仍只由项目 `ToolRuntime` 执行。
- `search_knowledge_base` 仍由 Agent Model 从动态工具中选择。成功的单工具调用返回 `Command.PARENT`，更新配对 ToolMessage 和 `pending_rag_result`；父图薄适配节点再进入原 RAG 子图。多工具同批结果也通过相同 pending state 条件交接，引用、证据门和拒答算法未改。
- Checkpointer 继续编译在父图，保存嵌入子图产生的完整消息轨迹。父图只保留 route、initialize、framework adapter、legacy plan/execute、RAG、compose 和 finalize 节点。
- 新增精确依赖 `langchain==1.3.14`。依赖解析确认它兼容并复用已锁定的 `langchain-core==1.4.9`、`langgraph==1.2.9`、`langchain-openai==1.3.5` 与 `pydantic==2.12.5`，没有升级其他核心依赖。
- 旧完整工具循环、手工 StructuredTool ValidationError 捕获、基础 ToolMessage 批处理、父图调用/错误指纹和第二套通用 canonicalization 已删除。当前维护入口是 `framework_agent.py`、`framework_middleware.py` 和 `framework_tools.py`；`app/graph/nodes.py` 只维护父图业务适配。

PR3R 未修改 API Schema、SQLite Schema、生产工具名称/业务参数 Schema、RAG 算法或 Registry/Runtime 治理边界，也未运行真实外部模型。

## 16. PR4 Runtime 稳定化与维护契约（2026-07-22）

本节只记录 PR4 增量。PR4 没有修改 Agent 行为、业务工具、父图节点/条件边、`create_agent` Middleware 栈、工具执行链、RAG handoff、API Schema 或 SQLite Schema。

- 新增 `scripts/inspect_agent_runtime.py`，使用普通 `build_agent_graph().compile()`，不初始化 Checkpointer、模型、工具 Runtime 或 RAG 索引。
- 父图和框架子图结构来自 `get_graph()` 与 `get_graph(xray=True)`；Mermaid 来自 `draw_mermaid()`，没有解析源码、AST 或调用外部渲染服务。
- 默认工具参数来自 StructuredTool 模型可见的 Pydantic JSON Schema；版本来自 `importlib.metadata`，没有手工复制 Schema 或锁死脚本中的依赖版本。
- `--check` 覆盖父图预期/禁用节点、框架 model/tools 与调用限制节点、xray 子图、默认 Registry 工具契约、关键依赖可见性和五个核心文件规模预算。
- 新增 `docs/agent-runtime-maintenance.md`，记录十分钟阅读路径、真实 xray Mermaid、职责边界、故障定位、新增工具步骤和禁止事项。
- 现有 GitHub Actions 质量 Job 增加无网络架构检查；没有创建重复 Workflow。
- 新增 6 个公开行为契约测试，覆盖真实图、xray、Mermaid、JSON、`--check`，并证明检查过程不会初始化模型、Checkpointer 或执行 handler。

PR4 开始前本地 HEAD 为 `ffc64fb`，与任务给出的远端查验值一致；基线为 `501 passed, 1 warning`。完成后为 `507 passed, 1 warning`，Ruff、compileall、四种检查脚本模式与 `git diff --check` 全部通过；唯一警告仍来自 `jieba` 对 `pkg_resources` 的弃用提示。生产目录 `app/` 净变化为 0 行；新增脚本 239 行，没有新增第三方依赖或生产模块。

## 17. PR5 权限感知工具可见性（2026-07-22）

本节只记录 PR5 增量。PR5 沿用 LangChain Middleware 的请求级工具覆盖，不新增 Agent 节点、工具执行器、权限引擎或依赖：

- `is_tool_visible()` 复用 `has_permission()`，仅当 Context 精确满足全部 `required_permissions` 时返回真；空要求始终可见，多权限采用 AND 语义。
- `LocalToolProvider` 按 Registry 注册顺序过滤 ToolSpec，不缓存 Context 或结果，也不执行 Policy、Runtime、handler 或审计。
- `RegistryToolsMiddleware` 原有 `request.override(tools=...)` 与 `request.override(tool=...)` 自动获得过滤结果；Agent Graph、`create_agent` 循环和 Middleware 栈未修改。
- 审批、幂等键、主体、来源和风险等级不参与可见性；权限满足的工具仍由 `ToolRuntime → ToolPolicyEngine` 做执行期治理。
- 模型伪造隐藏名称时 `request.tool is None`，沿用通用 `unknown_tool` ToolMessage，不泄漏 description、Schema 或缺失权限；直接 Registry dispatch 仍由 Runtime 返回 `permission_denied`。
- Runtime inspection 改用 `build_default_tool_context()`，报告可见工具的 `required_permissions`、默认权限数量和可见工具数量，并检查每个输出工具都满足默认 Context。

PR5 修改前本地 HEAD 为 `5d2754f`，基线为 `507 passed, 1 warning`；Ruff、compileall、Runtime architecture check 和 `git diff --check` 均通过。PR5 未修改 Agent Graph、Middleware 栈、API/SQLite Schema、RAG 算法、生产工具名称或参数 Schema。

PR5 完成后为 `515 passed, 1 warning`；重点测试、Ruff、compileall、三种 Runtime inspection 模式、架构检查与 `git diff --check` 全部通过。唯一警告仍来自 `jieba` 对 `pkg_resources` 的弃用提示；自动测试未调用真实外部模型或网络。

## 18. PR6 开源 Web Search 生产工具（2026-07-22）

本节只记录 PR6 增量；PR0–PR5 历史内容保持原样。修改前实际分支为 `fix/langgraph-runtime-hardening`，HEAD 为 `50468f9611e325c4b4ffb2461f625c92b3f4ce47`，工作区干净。修改前本地基线为 `515 passed, 1 warning`，Ruff、`python -m compileall app`、Runtime architecture check 与 `git diff --check` 全部通过。

- 新增 MIT 许可证 `ddgs==9.14.4` 基础依赖；要求 Python `>=3.10`，当前 Python 3.12 可用。pip dry-run 只新增 DDGS 及传输依赖，复用 `httpx==0.28.1`，没有升级或降级 LangChain、LangGraph、Pydantic、FastAPI、requests 或 ChromaDB。未安装 extras、`langchain-community` 或 MCP。
- 生产 Registry 新增 `web_search@1.0.0`：`READ_ONLY`、`required_permissions=("web:search",)`、幂等、无需审批、总体超时 10 秒、最多一次 Runtime 重试、`SUMMARY` 审计、query 敏感字段、`WebSearchOutput` 输出校验。
- 模型参数只有 `query`、`max_results` 和 `freshness`；region、safesearch、DDGS 内部 timeout 和 proxy 来自统一 Settings。默认权限仍为 `knowledge_base:read`，因此默认可见工具仍是原五个；显式加入 `web:search` 后才暴露第六个工具。
- DDGS `title/href/body` 被规范化为有限的 `rank/title/url/snippet`，过滤非 HTTP(S) URL、按 URL 去重并保序。空列表是成功零结果，所有成功输出带 `provider="ddgs"` 和 `content_is_untrusted=true`，不抓取网页正文。
- 公开 `RatelimitException` 映射 `rate_limited/True`，`TimeoutException` 映射 `timeout/True`，`DDGSException` 保守映射 `dependency_unavailable/True`，调用参数错误映射 `invalid_arguments/False`，不可规范化顶层结构映射 `invalid_result/False`。handler 不重试；ToolRuntime 继续负责有限重试、退避、超时、审计和 attempt 元数据。
- Runtime inspection 分离 `registered_tools` 与 `default_visible_tools`，验证注册契约、模型 Schema、默认权限子集、Web Search 的隐藏/授权可见性，且不会实例化 DDGS 或执行 handler。
- Fake-DDGS 自动测试覆盖参数、freshness 映射、规范化、零结果、异常、安全审计、Runtime 权限/重试/timeout/output_model、连续请求可见性、隐藏名称伪造、真实 Graph 与 ChatService 主链路。自动测试不访问真实搜索引擎；真实网络冒烟脚本已提供但未自动运行。
- Agent Graph、Middleware 栈、Router、Planner、Executor、API Schema、SQLite Schema、RAG 算法、Checkpointer 和原五个工具业务契约均未修改。Web Search 是普通 ToolMessage 路径，不设置 `pending_rag_result`，不进入本地 RAG，也不产生本地 references。

PR6 完成后为 `565 passed, 1 warning`；Ruff、`python -m compileall app`、三种 Runtime inspection 模式、架构检查、`python -m pip check` 与 `git diff --check` 全部通过。唯一警告仍来自 `jieba` 对 `pkg_resources` 的弃用提示。自动测试全部使用 Fake DDGS，没有访问真实搜索引擎；按任务约束未运行真实网络冒烟。

## 19. PR7 安全 Web Fetch 网页正文工具（2026-07-22）

本节只记录 PR7 增量；PR0–PR6 历史内容保持原样。修改前实际分支为 `fix/langgraph-runtime-hardening`，HEAD 为 `582cab9e80e0c0d01be3b7f7a5497a114d2ffd08`，工作区干净。修改前本地 `pytest -q` 实际为 `4 failed, 561 passed, 1 warning`：本地未跟踪 `.env` 已显式授予 `web:search`，而四个测试隐式假设默认只有知识库权限；Ruff、`python -m compileall app`、Runtime architecture check、`python -m pip check` 和 `git diff --check` 均通过。相关测试已改为显式构造权限上下文，不修改或覆盖维护者的 `.env`。

- 新增 BSD-3-Clause 的 `requests-hardened==1.3.0` 与 Apache-2.0 的 `trafilatura==2.1.0`，两者均要求 Python `>=3.10` 并声明支持 Python 3.12；本地验证版本为 Python 3.12.3。pip dry-run 没有升级或降级 LangChain、LangChain Core/OpenAI、LangGraph、Pydantic、FastAPI、ChromaDB、sentence-transformers、DDGS、requests 或 httpx。
- 实际解析复用 `requests==2.32.5`、`certifi==2026.6.17`、`charset-normalizer==3.4.5`、`idna==3.18`、`urllib3==2.7.0`、`lxml==6.1.1`、`python-dateutil==2.9.0.post0` 和 `regex==2026.5.9`；新增 Trafilatura 基础依赖为 `babel==2.18.0`、`courlan==1.4.0`、`dateparser==1.4.1`、`htmldate==1.10.0`、`jusText==3.0.2`、`lxml-html-clean==0.4.5`、`pytz==2026.2`、`tld==0.13.2`、`tzlocal==5.4.4`。未安装 dev、docs、eval 或 all extras。
- Registry 新增第七个生产工具 `web_fetch@1.0.0`：`WebFetchArgs`、`WebFetchOutput`、`READ_ONLY`、`required_permissions=("web:fetch",)`、总体超时 30 秒、最多一次 Runtime 重试、URL 敏感参数、`SUMMARY` 审计、幂等且无需审批。30 秒外层预算大于 3 秒连接与 10 秒读取预算，为安全重定向、streaming 和 Trafilatura 提取保留余量。
- 模型 Schema 只有 `url` 和 `max_chars`。下载固定经过 `requests-hardened.Manager(Config(...))`，启用 IP filter、禁止 loopback、保留安全自动重定向；Session 禁用环境代理与 `.netrc`，只发 GET、受信 User-Agent 和有限 Accept。响应先检查 Content-Length，再按解压后的实际字节 streaming 累计，默认上限 2,000,000 bytes，并始终关闭 Response/Session。
- 应用层只处理 HTTP(S)、无 credentials/控制字符、最长 2048 字符、80/443 端口和 fragment 移除；private、loopback、link-local、metadata、DNS 与逐跳 redirect 安全由 requests-hardened 处理，不自行实现 IP 枚举、DNS 固定或 redirect 验证。测试覆盖 `127.0.0.1`、`localhost`、`::1`、`169.254.169.254`、`10.0.0.1`、`192.168.1.1`、public URL 和 public-to-private redirect，且不会进入真实 socket transport。
- HTML/XHTML 只把安全下载后的文本交给 Trafilatura 公共 `extract(..., output_format="json", with_metadata=True)`；纯文本直接解码和有限规范化。只接受 HTML、XHTML 和 text/plain，PDF、JSON、二进制与缺失 Content-Type 均拒绝。输出由 Pydantic 校验正文长度、2xx 状态、HTTP(S) URL，并固定 `extractor="trafilatura"`、`content_is_untrusted=true`。
- URL、元数据和正文均是不可信数据；ToolSpec 明确禁止执行网页命令、把页面当作系统/开发者/工具指令或因页面要求调用其他工具。审计只保留状态、MIME、长度、截断和不可信标记，不保存完整 URL、正文或代理。
- 默认权限保持 `knowledge_base:read`，所以最小默认上下文仍只看到原五个工具。`web:search` 与 `web:fetch` 独立控制搜索和单页正文读取；两者都有时支持 `web_search → ToolMessage → web_fetch → ToolMessage → final answer`，隐藏名称伪造仍返回 `unknown_tool`，直接无权 dispatch 仍返回 `permission_denied`。
- 自动测试使用 Fake DDGS、Fake hardened Manager/Session/Response 和 requests-hardened 的公开 adapter 边界；固定 HTML fixture 调用真实 Trafilatura。全量结果为 `639 passed, 1 warning`，Ruff 通过；唯一警告仍来自 `jieba` 对 `pkg_resources` 的弃用提示。自动测试没有访问真实模型、搜索引擎、网页、代理、RAG 索引、生产 Checkpoint 或生产审计文件；按任务约束未运行真实网络冒烟。
- Agent Graph、Middleware 栈、Router、Planner、Executor、API Schema、SQLite Schema、RAG 算法、Checkpointer 和现有六个工具契约均未修改。`web_fetch` 是普通 ToolMessage 路径，不设置 `pending_rag_result`，不写 ChromaDB/BM25，不产生本地 RAG references。

## 20. PR8 安全 Web PDF 读取工具（2026-07-23）

本节只记录 PR8 增量；PR0–PR7 历史内容保持原样。修改前实际分支为 `fix/langgraph-runtime-hardening`，HEAD 为 `b677f477f7e1f06a6de99b05c11ee2b05bcf75b9`，工作区干净；该 HEAD 已包含 Agent 请求结束时的脱敏 `tool_call_chain` INFO 日志。修改前本地基线为 `640 passed, 1 warning`，Ruff、`python -m compileall app`、`python -m pip check` 和 `git diff --check` 通过。原 `python scripts/inspect_agent_runtime.py --check` 唯一失败项为 `web_fetch_default_visibility`：本地未跟踪 `.env` 已显式授予 `web:fetch`，但旧检查硬编码要求该工具始终隐藏；PR8 将检查改为验证可见性与当前权限一致，同时继续用显式最小权限测试证明默认代码配置下 Web 工具不可见。

- 未增加或升级依赖。安全下载继续使用锁定的 `requests-hardened==1.3.0`，PDF 解析复用锁定的 `pypdf==6.14.2`；Agent Loop、参数/输出 Schema 和执行治理继续分别复用 LangChain `create_agent`、Pydantic、ToolRegistry 与 ToolRuntime。
- 新增第八个生产 ToolSpec `web_pdf_fetch@1.0.0`：`WebPdfFetchArgs`、`WebPdfFetchOutput`、`READ_ONLY`、`required_permissions=("web:fetch",)`、默认总体超时 45 秒、最多一次 Runtime 重试、URL 敏感参数、`SUMMARY` 审计、幂等且无需审批。模型参数只有 `url`、`start_page`、`max_pages` 和 `max_chars`。
- 原 `web_fetch.py` 的 URL 业务校验、requests-hardened Manager、只读 GET 参数、HTTP 状态映射、Content-Type 白名单、streaming 字节限制及 Response/Session 关闭移动到薄共享适配 `web_download.py`。HTML/XHTML/纯文本仍由 Trafilatura 处理并继续拒绝 PDF；`web_pdf_fetch` 调用同一下载入口，没有 raw requests 回退或第二套 SSRF、代理、重定向、重试、权限和审计实现。
- PDF 允许 `application/pdf`、`application/x-pdf`，并只在内容具有 `%PDF-` 签名时兼容 `application/octet-stream`。默认 PDF 下载上限 10,000,000 bytes、文档总页数上限 1,000；连接/读取/重定向/User-Agent/代理继续复用 Web Fetch 配置。45 秒 ToolSpec 总体上限大于默认 3+10 秒内部网络预算，为解析保留时间。
- pypdf 只通过 `PdfReader(BytesIO(body), strict=False)`、`reader.is_encrypted`、`reader.metadata`、`reader.pages` 和 `page.extract_text()` 读取 PDF。输出保留实际 `[Page N]` 标记、跳过空白页、有限返回 title/author/subject，并在组合页文本后截断。加密、损坏、超过字节/页数上限和扫描件/无文本均为稳定非重试错误；不接收密码，不提取图片、附件、表单、JavaScript 或外部链接，也不调用 OCR。
- URL、PDF bytes、正文和元数据原值不会进入 SUMMARY 审计；审计只保留 HTTP 状态、MIME、total/start page、pages read、content length、truncated 和不可信标记。输出固定 `content_is_untrusted=true`；ToolSpec 禁止把 PDF 内容当作 system/developer/tool 指令或因其中要求调用其他工具。
- 默认代码配置仍为 `TOOL_DEFAULT_PERMISSIONS=knowledge_base:read`，最小上下文只看到原五个本地工具；`web:search` 只暴露 `web_search`，`web:fetch` 同时暴露 `web_fetch` 与 `web_pdf_fetch`。隐藏名称伪造返回 `unknown_tool`，直接无权 dispatch 返回 `permission_denied`，并且 HTTP Manager 与 PdfReader 都不会构造。
- 真实 `build_agent_graph().compile().invoke(...)` 测试覆盖 `web_search → web_pdf_fetch → final answer`，以及 `web_fetch` 误读 PDF 收到 unsupported content type ToolMessage 后改用 `web_pdf_fetch` 的恢复链。ToolMessage call ID、Registry/ToolRuntime 调用、页码正文、tool trace 顺序、脱敏 `tool_call_chain`、空本地 references 和不进入 RAG 均已验证。
- 自动测试使用确定性两页 PDF fixture、Fake DDGS 和 Fake hardened Manager/Session/Response；加密、空白、超页数和损坏 PDF 在测试中通过 pypdf 生成。完成后全量结果为 `698 passed, 1 warning`；唯一警告仍来自 jieba 对 `pkg_resources` 的弃用提示。自动测试没有访问真实模型、搜索引擎、PDF URL、代理、RAG 索引、生产 Checkpoint 或生产审计文件；按任务约束未运行真实网络 PDF 冒烟。
- Agent Graph、Middleware 栈、Router、Planner、Executor、FastAPI API Schema、SQLite Schema、Checkpointer、本地 RAG 算法和原七个工具业务契约均未修改。`web_pdf_fetch` 只走普通 ToolMessage，不设置 `pending_rag_result`，不写 ChromaDB/BM25/manifest/知识库目录，不产生本地 RAG references，也不永久保存 PDF。

## 21. PR9 删除失效 LLM Router 兼容层

本节只记录 PR9 增量；PR0–PR8 历史内容保持原样。开始时实际分支为 `fix/langgraph-runtime-hardening`，HEAD 为 `59d67591882559d702487f0ce739aab954f53c4e`，与任务记录一致，工作区干净。修改前本地基线为 `709 passed, 1 warning`；Ruff、`python -m compileall app`、`python scripts/inspect_agent_runtime.py --check`、`python -m pip check` 和 `git diff --check` 全部通过。

修改前先用仓库级 `rg`、明确导入路径和实际 `/chat` 调用链逐类核对引用：

| 删除类别 | 修改前生产引用 | 测试/文档引用 | 结论 |
| --- | --- | --- | --- |
| `app/agent/llm_router.py`、`call_router_llm()`、`parse_router_json_output()` | 仅由 `router.py` 中无人调用的 `route_agent_action_by_llm()` 引用，并由 `app.agent.__init__` re-export；当前 `route_agent_action()` 不到达该链 | `tests/test_agent_router.py` 直接测试旧模型调用/JSON 解析；README 记录兼容层 | 删除模块、旧调用测试和文档描述 |
| `route_agent_action_by_llm()`、`validate_route_decision()`、Router action/tool 常量及 normalizer | 只在 `router.py` 内组成旧 Router 输出兼容解析，或由 `app.agent.__init__` re-export；生产入口无调用方 | 旧 Router 测试直接导入和断言 | 删除函数、常量、旧解析测试和公共导出 |
| `get_router_model()` 与 Router 专属 Settings | 只服务已删除的 `llm_router.py`；`app.llm.__init__` 仅做 re-export | Factory 参数化测试、README 和 `.env.example` | 删除模型工厂入口、缓存清理项、全部 Router 专属环境配置和导出；保留 Answer、Query Rewrite、Decompose、Compare、Judge 与 Agent Chat Model |
| `ROUTER_SYSTEM_PROMPT`、`ROUTER_USER_PROMPT` | 只由已删除的 `llm_router.py` 导入 | 旧 Router 调用测试间接覆盖 | 删除 Prompt，不改其他 RAG/Agent Prompt |
| `extract_chunk_id()`、数学表达式辅助、文档对比规则辅助及 `COMPARE_DOCUMENT_KEYWORDS` | 只有定义和 `app.agent.__init__` re-export；生产代码无消费者 | 无现行行为消费者 | 删除；`rules.py` 只保留确定性入口实际使用的 `normalize_question()` |
| `app.agent` 业务函数聚合导出 | `api/chat.py`、`chat_service.py`、`graph/nodes.py` 从包级导入 Router，`answer.py` 从包级导入 formatter | 部分测试从包级导入 Router | 生产和集成测试改为明确模块导入；`app.agent.__all__` 收缩为空，不影响显式子模块导入 |

保留的 `route_agent_action()` 仍只执行确定性入口规则：空问题返回 `clarify`，少量系统能力问题返回 `direct_answer`，少量明确指代不明问题返回 `clarify`，其他普通问题稳定返回 `action="agent"`。现有 `decision`、`action`、`tool_name`、`tool_args`、`reason`、`clarification_question`（适用场景）、`router_source`、`router_confidence`、`router_llm_used` 和 `router_fallback_reason` 结构未做统一重构；`router_llm_used` 继续输出 `False`。显式注入/旧协议使用的 `rag_fast_path` 消费路径也继续保留在 ChatService、Graph 和 Planner/Executor 兼容层，本 PR 只删除已无入口的 Router LLM 生产者。

普通工具选择继续由 LangChain `create_agent` 根据 `RegistryToolsMiddleware` 注入的当前可见 StructuredTool/Pydantic Schema 完成，执行边界仍为 `StructuredTool → ToolRegistry.dispatch() → ToolRuntime.execute() → ToolSpec.handler`。新增 Registry Tool 的回归测试证明 Router 输出不随注册工具集合变化，也不需要修改 Router。

新增/整理的测试覆盖空问题、`direct_answer`、`clarify`、普通问题进入 `agent`、不调用 Router 模型、不选择具体工具、新增 Registry Tool 不修改 Router、包级旧 API 不可导出、`llm_router` 模块不存在，以及 ChatService/Graph 继续绑定同一明确模块 Router。只保护旧模型调用、Markdown JSON 解析、旧具体工具 action 与 `ENABLE_LLM_ROUTER` 分支的测试已删除。完成后全量结果为 `721 passed, 1 warning`；唯一警告仍来自 jieba 对 `pkg_resources` 的弃用提示。

父图节点和边、Framework Agent 节点、Middleware 栈、Planner、Executor、`legacy_plan`、`fallback_to_legacy`、AgentState 计划字段、RAG 子图、ToolRegistry、ToolRuntime、Provider、权限/审计、Checkpointer、FastAPI API Schema、HTTP 状态码、SQLite Schema、session、metrics、tool trace 与 `tool_call_chain` 日志均未修改。八个生产工具的名称、参数/输出 Schema、权限、风险等级、timeout、重试、审计与可见性保持不变。自动测试使用隔离 fake，没有调用真实模型、搜索引擎、网页/PDF、代理，没有重建 RAG 索引、创建生产 Checkpoint 或写生产审计文件。

## 22. PR10 单一框架 Agent 执行链

本节只记录 PR10 增量；PR0–PR9 历史内容保持原样。开始时实际分支为 `fix/langgraph-runtime-hardening`，HEAD 为 `ee3fe56c450580fdb8c3630c112eeb10e5050d1e`，工作区干净。修改前本地基线为 `721 passed, 1 warning`；Ruff、`python -m compileall app`、`python scripts/inspect_agent_runtime.py --check`、`python -m pip check` 和 `git diff --check` 全部通过。

修改前引用盘点确认：`planner.py` 和 `executor.py` 由父图 `plan`/`execute` 节点执行；`framework_agent.py` 只为 observation/trace 转换依赖 Executor 并构造虚假 `AgentPlanStep`；`langchain_tools.py` 的生产消费者只剩旧 Executor，此外只有公共 re-export、手工脚本和旧实现测试；`fallback_to_legacy` 贯穿框架状态、模型异常 Middleware、父图适配和条件边；`rag_fast_path` 与具体工具 action 仍由 Planner/ChatService 前置 RAG 检查消费；`AGENT_MAX_STEPS` 只约束旧 Executor；`router_specs()`/`to_router_spec()` 只剩定义和测试。

- 删除 `app/agent/planner.py`、`app/agent/executor.py`、`app/tools/langchain_tools.py` 和无生产消费者的 `app/agent/formatting.py`。删除旧 StructuredTool 公共导出、Router tool-spec 表面、旧实现测试和手工脚本对闭包适配的依赖；手工 Tool Calling 脚本改用 `build_framework_tools()`。
- 新增 `app/agent/tool_events.py`，只负责把标准工具结果和框架治理事件转换为 `AgentObservation` 与兼容 `tool_call` trace。它不接受计划步骤，不执行工具，不构造 ToolMessage，不调用模型，也不复制 ToolRuntime。
- 删除 `AgentPlanStep`、`StepStatus`、`AgentState.plan`、`AgentState.max_steps` 和 `AGENT_MAX_STEPS`。保留 observation、稳定步号、框架治理计数、`pending_rag_result`、API 状态和结果映射。
- 父图节点收缩为 `route`、`initialize`、`framework_agent`、`adapt_framework`、`rag_subgraph`、`compose`、`finalize`。普通 `agent` 进入 `create_agent`；`direct_answer` 和 `clarify` 在 initialize 写入确定性状态并直接 compose；其他旧 action 返回 `INTERNAL_ERROR`，不调用模型或工具。
- 模型异常只记录失败事件与 `stop_reason=agent_model_call_failed`，经适配和 compose 返回 `MODEL_ERROR`；不再计算工具使用情况，不再写 fallback 状态，也不进入第二套执行链。
- `search_knowledge_base` 仍经唯一的框架 StructuredTool、Registry 和项目 ToolRuntime。成功结果继续用 `Command.PARENT` 进入原 RAG 子图；知识库不可用则形成标准错误 ToolMessage，Agent Model 可以读取并安全结束。ChatService 删除旧 action 专属 RAG 前置检查和依赖，兼容 metrics 字段 `rag_ready_check_cost_ms` 保留为 `0`。
- 删除内部 `plan`、`executor` 和旧计划 `final_answer` trace。继续保留 `query_rewrite`、`decide`、`agent_iteration`、`tool_call`、`compose_final_answer`、`llm_usage`，工具 trace 的治理、重试、命中和框架扩展字段不变，`tool_call_chain` 仍只聚合真实 `tool_call`。
- 自动测试使用真实 `build_agent_graph().compile().invoke(...)`、`create_agent`、RegistryToolsMiddleware、ToolRegistry 和 ToolRuntime，只替换 Agent Model、DDGS、HTTP transport 与 RAG 外部副作用。覆盖无工具回答、计算器、多工具 Web/PDF、知识库 RAG handoff、知识库不可用错误 ToolMessage、模型失败、短路、旧 action fail-closed、memory/sqlite Checkpointer、metrics、session 和终端工具链日志。

完成后全量结果为 `653 passed, 1 warning`；Ruff、compileall、Runtime architecture inspection、pip check 和 diff check 通过。唯一警告仍来自 jieba 对 `pkg_resources` 的弃用提示。自动测试没有调用真实模型、搜索引擎、网页/PDF、代理，没有重建 RAG 索引、创建生产 Checkpoint 或写生产审计文件。

FastAPI `/chat` 请求/响应 Schema、HTTP 业务状态码、SQLite session/Checkpointer Schema、thread_id、RAG 检索/重排/证据/引用算法、八个 ToolSpec 名称与参数/输出 Schema、Provider 权限可见性、ToolRuntime 权限/审批/超时/重试/审计、metrics 字段和 session 行为均未修改。Router 双重所有权、`AgentGraphState`/`AgentState` 统一、Result/status 映射统一、Settings 迁移、模块导入边界和历史文档归档仍留给后续独立整理。

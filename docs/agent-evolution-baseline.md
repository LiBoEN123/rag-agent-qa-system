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

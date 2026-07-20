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

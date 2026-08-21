# ADR 0013：单一平铺 Agent Runtime State

- 状态：Accepted
- 日期：2026-07-28
- 决策者：Project Maintainer

> 2026-08-21 Superseded in implementation：消除重复业务状态的目标继续有效，但完整
> `AgentGraphState`、Parent Graph Checkpointer 和事件重放游标已删除。当前仅保留最小
> `FrameworkAgentState` 与由 `results.py` 一次性归一化的普通 Python Result 快照。

## 背景

PR11 后，LangGraph 父图保存请求输入、Router、框架消息与治理字段、RAG handoff 和最终 result，同时还嵌套一个可变 Pydantic 业务状态。问题语义、检索参数、trace/session 和业务结果因此分布在两层；节点既返回 LangGraph update，又原地修改嵌套对象。该模式让节点输入输出、checkpoint payload、重放语义和测试构造都不清晰。

## 决策

1. `AgentGraphState` 是一次请求唯一的完整运行状态。请求输入、Router、框架治理、业务结果和最终输出都作为其平铺 channel 声明。
2. 删除项目自有的完整 Pydantic `AgentState`、父图 `agent_state` 字段、`_require_agent_state()` 和 `get_effective_question()`。LangChain 框架自己的状态类型继续用于 `create_agent` 子图。
3. 不保存 `rewritten_question`。运行时始终使用 `effective_question`；trace 持久化需要改写值时，通过它与 `original_question` 是否不同来派生。
4. `AgentObservation` 使用 `TypedDict`，使 checkpoint 中的 observation 成为普通字典。工具事件 builder 继续验证 `step_id >= 1` 和非负 `cost_ms`；Runtime 门面继续验证 `top_k` 为 1–10、`decompose_top_k` 为 1–20。
5. initialize、框架事件投影、RAG 结果映射、compose 和 finalize 都返回显式字段 updates，不修改输入状态或输入列表。
6. `messages` 保留 LangGraph `add_messages` reducer，`governance_events` 保留 append-only reducer。`observations`、`references` 和 `tool_trace` 返回完整新列表，不使用追加 reducer，避免 resume 或节点重放产生重复。
7. 框架投影只消费 `framework_events_applied` 之后的事件，稳定生成模型迭代、工具 observation/trace 和治理拦截 trace；无新事件时重复调用结果不变。
8. finalize 先解析 LLM 调用数，再 upsert 唯一 `llm_usage`，以同一份 trace 快照构建 result 和持久化 trace。
9. 本决策不统一内部/外部 Result/status 映射，不删除 `build_v2_result()`，也不改变 FastAPI Response Schema。

## Checkpointer 边界

SQLite Checkpointer 的表结构、配置模式、数据库路径和 thread ID 算法不变；改变的是新运行写入的 State channels。memory 与 SQLite Saver 都直接序列化平铺字段、LangChain Message 和普通 observation/trace 容器。

当前 thread ID 包含一次运行的 trace 语义，同一 session 的不同请求使用独立 thread。项目尚未实现跨版本 interrupt/resume，因此不承诺恢复 PR12 之前尚未完成、payload 仍含旧嵌套对象的内部 checkpoint；不需要清空或迁移用户数据库，已完成运行也不作为聊天历史恢复来源。

## 后果

节点的输入输出边界和 checkpoint payload 与 LangGraph channel 对齐，测试不再构造两层状态，事件重放不会因对象 mutation 或 reducer 追加而重复 observation/trace。Graph 节点和边、`create_agent`、Middleware 顺序、RAG `Command.PARENT` handoff、八个 ToolSpec、ToolRuntime、API、metrics、session、审计和 `tool_call_chain` 均保持原契约。

后续仍需独立处理 Result/status 映射统一、Settings 迁移、模块依赖边界、ChatService 进一步拆分和历史文档归档。

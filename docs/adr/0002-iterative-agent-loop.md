# ADR 0002：迭代式 Agent Loop

- 状态：Accepted
- 日期：2026-07-19
- 决策者：Project Maintainer

> 2026-08-21 Superseded in implementation：单 Agent 迭代式模型—工具循环的目标保留，
> 但下文手写父图节点和兼容路径已由 ADR 0006、0011 及 Phase 1 删除。当前唯一循环是
> LangChain `create_agent`；确定性 Router 位于 Agent Runtime 的普通 Python 入口。

## 背景

项目当前唯一 Agent Runtime 是 LangGraph。FastAPI `/chat` 经 `ChatService` 完成会话历史读取、Query Rewrite 和 Router 决策后调用主图；主图复用既有 Router、Planner、Executor、RAG 子图和 Answer Composer。该架构已经统一运行入口并保留 RAG 行为，但 Agent 决策仍以一次路由和预先生成的固定计划为中心。

当前主图的准确流程为：

```text
START
→ route（仅调用方未预先传入 decision 时执行）
→ initialize
→ plan
→ execute
→ 成功的 search_knowledge_base observation：rag_subgraph → finalize → END
→ 其他工具、澄清、直接回答或工具错误：compose → finalize → END
```

Planner 将 Router 的具体 action 映射成固定步骤。当前五种工具 action 都主要生成：

```text
tool_call
→ final_answer
```

`clarify` 和 `direct_answer` 则生成单个短路步骤。Executor 按计划顺序执行并受 `max_steps` 限制；工具执行完成后不会返回模型进行下一轮决策。当前主图未使用 `messages` 轨迹、模型 `bind_tools()`、LangGraph `ToolNode` 或循环边。

## 当前问题

- Agent 更接近固定工作流，而不是可以根据观察继续决策的迭代式 Agent。
- Router 同时承担入口分类、具体工具选择和部分参数构造，职责过多。
- Planner 对 `search_knowledge_base`、`compare_documents`、`simple_calculator`、`list_documents` 和 `get_chunk_text` 等工具名存在硬编码。
- 工具调用结果只进入 observation 和最终答案合成，不能驱动模型重新规划。
- 一个工具参数错误或执行失败通常会将状态设为 `tool_error` 并立即中止整个计划。
- 固定的一次工具调用难以自然完成需要多个观察、参数修正或条件分支的任务。

## 决策

后续将现有 LangGraph 主图演进为单 Agent 的迭代式模型—工具循环：

```text
START
→ route / initialize
→ agent_model
→ 如果存在 tool_calls：
     governed_tools
     → agent_model
     → 循环
→ 如果不存在 tool_calls：
     compose
     → finalize
     → END
```

`messages` 将作为 Agent 工作轨迹保存，至少包含：

- `HumanMessage`：用户输入或经过明确标记的有效问题。
- `AIMessage`：模型决策、工具调用请求和最终响应草稿。
- `ToolMessage`：与 `tool_call_id` 关联的工具观察或结构化失败。

模型可以根据工具观察结果决定继续调用工具、修正参数、切换路径或结束回答。正常工具选择交给支持 Tool Calling 的模型；Router 后续只负责粗粒度入口判断和必要的安全/降级路由。当前 Planner 保留为规则降级路径，不再作为主要规划核心。

循环必须同时设置并强制执行：

- 最大迭代次数。
- 最大工具调用次数。
- 最大工具错误次数。
- 总耗时预算。

任一预算耗尽时，主图必须进入可解释、可审计的终止路径，不能无限循环。

## 采用的方案

- 在现有 `AgentGraphState` 中引入可归约的 `messages` 工作轨迹，并为循环计数、错误计数和时间预算增加显式状态。
- 新建 `agent_model` 节点，由可配置的 LangChain Chat Model 绑定注册工具的 Schema。
- 新建 `governed_tools` 节点，将每个模型工具调用转换为项目现有的受治理执行链，并将结果转换为对应 `ToolMessage`。
- 用条件边检查最近 `AIMessage` 是否包含 `tool_calls`：有则进入工具节点并回到模型，无则进入 `compose` 与 `finalize`。
- 保留 RAG 子图。RAG 可以继续作为粗粒度专用路径，也可以在验证治理和消息协议后包装为高级工具能力；不得重写现有检索、重排、引用和拒答算法。
- 保留当前规则 Router 和 Planner 作为模型不可用、输出无效或功能开关降级时的确定性路径。
- 保持现有 API Schema、状态语义、trace_id、会话历史和离线评测入口兼容。

## 暂不采用的方案

- 不在第一阶段引入多 Agent、角色协作或跨 Agent 状态同步。
- 不由 Router 继续承担所有具体工具选择，否则无法形成观察驱动的模型循环。
- 不以递归 Python 调用在 LangGraph 外部实现循环，避免预算、Checkpoint 和轨迹脱离主图状态。
- 不让模型或 `ToolNode` 直接调用业务 handler，工具执行必须遵守 ADR 0003 的治理边界。
- 不删除 Planner；它仍是明确的规则降级路径。

## 影响

- Agent 将能完成多工具、多观察和参数纠正类任务，但模型调用次数、延迟和状态体积会增加。
- `messages` 和工具结果可能包含敏感业务信息，Checkpoint、日志和轨迹存储需要延续现有脱敏与数据保护要求。
- 循环预算、工具错误语义、ToolMessage 对齐和模型降级路径需要新增单元测试与离线轨迹评测。
- Router、Planner 和 Executor 的主要职责会调整，但现有 RAG 子图、工具业务实现和 API 协议保持稳定。

## PR1 实现状态

PR1 已完成本 ADR 的最小迭代循环：

- `AgentGraphState.messages` 使用 LangGraph `add_messages` reducer，保存 Human、AI 和 Tool 消息。
- `agent_model` 使用现有 OpenAI-compatible 回答模型工厂和请求级 StructuredTool 执行 `bind_tools()`，测试可通过 `get_agent_chat_model()` 注入确定性模型。
- `governed_tools` 支持同一 AIMessage 的全部 `tool_calls`，每次结果都生成带正确模型调用 ID 和工具名的标准 `ToolMessage`。
- 普通工具错误会返回模型继续决策；权限拒绝、策略拒绝、审计写入失败和非法调用上下文会受控停止。
- `AGENT_MAX_ITERATIONS` 与 `AGENT_MAX_TOOL_CALLS` 分别限制模型轮次和工具执行总数，预算耗尽会设置明确 `stop_reason` 并正常进入 Composer/Finalize。
- 最终 AIMessage 的文本写回 `AgentState.final_answer`，仍通过现有 Composer、`build_v2_result` 和 `/chat` Response Builder 输出。

当前没有直接采用 `ToolNode`。LangGraph 1.2.9 的 ToolNode 在构图时接收固定工具集合，而项目 StructuredTool 捕获每次请求独立的 `ToolInvocationContext`；静态复用会造成 trace、session、权限和来源跨请求混用。当前薄节点只做动态工具适配，实际执行仍严格遵循：

```text
Agent Model
→ 请求级 LangChain StructuredTool
→ ToolRegistry.dispatch
→ ToolRuntime
→ ToolSpec.handler
```

RAG 继续使用专用子图：初始 Router 选择知识库检索时走保留的 Planner/Executor 兼容入口；模型循环中选择 `search_knowledge_base` 后也会转入同一 RAG 子图。该选择确保问题拆分、混合检索、重排、证据门控、拒答和引用验证不回退。

PR2 已按 ADR 0004 完成 Registry-driven 工具发现：Agent Model 通过可注入 `ToolProvider` 动态绑定 ToolSpec；Router 只输出粗粒度入口 action；Planner 使用一个通用 Registry 分支兼容旧具体工具 decision，不再按普通工具逐项规划。`agent` 路径的模型不可用会受控停止，不让 Planner 猜测具体工具；Planner/Executor 继续负责 RAG Fast Path、澄清、直接回答和旧入口兼容。

尚未完成的部分包括：工具错误次数上限、总耗时预算、连续相同错误检测和更完整的恢复策略。Planner/Executor 暂不删除，RAG 专用路径和旧协议兼容层仍保留。

## 后续行动

1. PR1 首先扩展 `app/graph/state.py`，定义 messages、预算和循环计数状态。
2. 在 `app/graph/nodes.py` 与 `app/graph/builder.py` 中增加 `agent_model`、`governed_tools` 和条件循环边。
3. 为无工具调用、单次工具调用、多次工具调用、参数错误恢复和各类预算终止增加测试。
4. 明确 Checkpointer 中 messages 的保留、脱敏和清理策略。
5. 保持 Planner/Executor 规则路径可单独回归，待新循环稳定后再评估职责收缩。

本 ADR 在 PR0 记录演进方向，并由上述 PR1 实现状态说明当前已落地范围。

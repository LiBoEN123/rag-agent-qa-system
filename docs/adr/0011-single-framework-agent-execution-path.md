# ADR 0011：单一框架 Agent 执行链

- 状态：Accepted
- 日期：2026-07-28
- 决策者：Project Maintainer

## 背景

父图在迁移到 LangChain `create_agent` 后仍同时保留 `plan → execute` 兼容链。正常 `agent` 请求由框架管理模型—工具循环，但显式旧 action、短路回答和部分模型失败仍可能进入 Planner/Executor。旧 Executor 还依赖第二套闭包式 StructuredTool 适配，框架事件转换则为了复用其中的 observation/trace 逻辑构造虚假的 `AgentPlanStep`。

这使同一请求存在两套工具选择与执行协议，也让已经不再由 Router 生产的内部 action、计划状态和配置继续影响父图、测试与维护文档。

## 决策

LangChain `create_agent` 成为普通工具和知识库工具唯一的模型—工具执行循环。父图只保留：

```text
START
→ route / initialize
→ initialize
→ framework_agent / compose
→ adapt_framework
→ rag_subgraph / compose
→ finalize
→ END
```

具体决策如下：

1. 删除 Planner、Executor、父图 `plan`/`execute` 节点、`legacy_plan` 条件结果及其内部 trace。
2. 普通工具选择、多轮工具调用、参数修正和 ToolMessage 配对继续由 `create_agent`、框架 ToolNode、`ModelCallLimitMiddleware` 与 `ToolCallLimitMiddleware` 完成。
3. 模型调用失败由 `ProjectToolGovernanceMiddleware` 写入模型失败事件和 `stop_reason=agent_model_call_failed`，经父图适配为 `MODEL_ERROR`；不再回退另一套执行架构。
4. `direct_answer` 和 `clarify` 在 initialize 阶段确定状态，再直接进入 compose；两者不解析或调用 Agent Model。
5. 删除 `rag_fast_path`、具体工具 action、`decision=use_tool` 和未知 action 自动回退知识库的消费协议。不支持的 action fail-closed 为稳定内部错误。
6. 删除 `app/tools/langchain_tools.py` 及其公共导出。唯一 Agent 工具适配是 `framework_tools.py` 中由 LangChain `ToolRuntime` 注入请求 Context 的 StructuredTool。
7. `StructuredTool → ToolRegistry.dispatch() → ToolRuntime.execute() → ToolSpec.handler` 的治理边界不变。Provider 只控制模型可见性，项目 ToolRuntime 仍独占最终权限、审批、幂等、超时、重试、审计和输出校验。
8. `search_knowledge_base` 成功后继续通过 `Command.PARENT` 携带配对消息和 `pending_rag_result`，由父图进入原 RAG 子图。
9. observation 与 `tool_call` trace 转换移动到 `tool_events.py`。该模块不接受计划步骤、不执行工具、不构造 ToolMessage，也不调用模型。
10. FastAPI Schema、HTTP 状态码、SQLite session/Checkpointer Schema、thread_id、RAG references、ToolSpec 名称与 Schema、权限/审计、metrics 和 `tool_call_chain` 保持稳定。

## 后果

父图不再有第二个模型—工具执行循环，新增普通工具也不需要 Planner、Executor、Router 或父图分支。旧具体工具 decision 会立即暴露为内部协议错误，不再被静默解释或自动检索知识库。知识库不可用由 `search_knowledge_base` 的标准工具错误和 ToolMessage 反馈给 Agent Model，ChatService 不在工具选择前猜测是否会使用 RAG；兼容 metrics 字段 `rag_ready_check_cost_ms` 保留并默认为 `0`。

允许有意消失的内部 trace 是 `plan`、`executor` 和旧计划中的 `final_answer` step。`query_rewrite`、`decide`、`agent_iteration`、`tool_call`、`compose_final_answer` 与 `llm_usage` 继续保留，`tool_call` 字段和终端 `tool_call_chain` 聚合格式不变。

本决策不统一 `AgentGraphState` 与嵌套 `AgentState`，不重写 Result/status 映射，不迁移 Settings，也不改变 Router 在 ChatService 与父图直接入口之间的双重所有权。这些属于后续独立整理。

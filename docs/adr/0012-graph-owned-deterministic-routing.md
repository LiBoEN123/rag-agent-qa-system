# ADR 0012：Graph-owned 确定性 Router

- 状态：Accepted
- 日期：2026-07-28
- 决策者：Project Maintainer

## 背景

PR10 后存在两个 Router 入口：正常 `/chat` 先由 ChatService 调用 Router，再把 decision 注入父图；直接调用 `run_agent_langgraph()` 时，父图只有在 decision 缺失时才执行 `route`。因此 Router 的所有权取决于调用方，评测、脚本和新入口可能绕过或重复路由。ChatService 还单独保存 action/decision、测量 route 耗时并推导 Router LLM 指标。

Router LLM 和具体工具 Router 已经删除。现行确定性 Router 只需要区分 `agent`、`direct_answer` 和 `clarify`；具体工具由 LangChain `create_agent` 根据当前请求可见的 StructuredTool Schema 选择。

## 决策

1. LangGraph 父图的 `route_node` 是确定性 Router 的唯一生产所有者。父图入口固定为：

   ```text
   START → route → initialize
   ```

2. ChatService 只负责 session history、Query Rewrite、Runtime 调用、metrics/session 收尾和 HTTP outcome，不调用 Router，也不保存或注入 decision。
3. `run_agent_langgraph()` 不接受外部 decision。正常 `/chat`、评测脚本和其他 Runtime 调用方都自动经过同一个父图 Router。
4. 即使直接调用编译图时传入旧 decision，`route_node` 也会覆盖它；调用方不能用 `direct_answer`、`clarify` 或具体工具 action 绕过当前规则。
5. 保留确定性 Router，是为了让稳定的系统能力直答和明确指代不明澄清不调用 Agent Model，同时让其他问题统一进入框架 Agent。
6. Router 不读取 Registry、Tool Schema、知识库 ready 状态或搜索/计算意图，也不返回 `tool_name`、`tool_args`。普通工具选择完全属于 `create_agent`。
7. `route_node` 使用单调时钟测量 `route_cost_ms`。initialize 将其写入 `decide` trace，`collect_trace_metrics()` 再把该值交给 ChatService metrics；ChatService 不维护第二个 Router 计时器。
8. 内部 RouteDecision 删除 `router_llm_used` 和 `router_fallback_reason`。外部结果、`llm_usage` trace、`decide` trace、API 和 metrics 继续显式输出 `router_llm_used=false`，保证已有消费者兼容。
9. `decide` trace 继续保留 `route_decision`、`decision_type`、`action`、空 `tool_name`、空 `tool_args`、空 `sanitized_tool_args`、`reason`、`router_source`、`router_confidence`、空 `router_fallback_reason` 和 `route_cost_ms`。
10. `/decide` 诊断端点保留，但复用 `route_node` 并只返回其 decision；它不成为第二个 Router 所有者，也不执行后续 Agent、工具或 RAG。

## 后果

Router 对每次正常 Graph 调用执行一次，Query Rewrite 后的 effective question 是唯一输入。直接 Runtime/评测调用与 `/chat` 不再有不同入口语义，Graph architecture inspection 会拒绝 `START → initialize` 和任何缺失的固定父图边。

Router 耗时来源从 ChatService 转为 decide trace，但 `logs/metrics.jsonl`、metrics 日志和 `/metrics-summary` 的兼容字段保持。Router 不计入 `llm_call_count`；该计数只包含 Agent Model、Query Decompose、RAG Answer Model，以及 ChatService 汇总时追加的 Query Rewrite Model。

FastAPI `/chat` Schema、HTTP 状态码、SQLite session/Checkpointer Schema、checkpoint thread_id、RAG references、八个 ToolSpec、Provider 权限可见性、ToolRuntime 权限/审批/超时/重试/审计、bad case、session 和 `tool_call_chain` 均不改变。

本决策不统一 `AgentGraphState` 与嵌套 `AgentState`，不重写 Result/status 映射，不迁移 Settings，也不整理全部模块依赖和历史文档；这些仍属于后续独立工作。

# ADR 0005：Agent 错误恢复、预算与循环防护

- 状态：Accepted
- 日期：2026-07-20
- 决策者：Project Maintainer

## 背景

PR1 建立了 `agent_model ↔ governed_tools` 循环，PR2 将模型工具发现改为 Provider/Registry 驱动。ToolRuntime 已负责参数校验、权限、审批判断、审计、超时和契约内有限重试，但 Agent 层此前没有累计工具错误上限、重复调用/错误检测或模型与工具累计耗时预算；终止错误还可能把同批后续调用继续执行。

## ToolRuntime 与 Agent 恢复的职责边界

ToolRuntime 是单次工具执行和治理的唯一入口：

```text
Agent Model
→ StructuredTool
→ ToolRegistry.dispatch
→ ToolRuntime.execute
→ ToolSpec.handler
```

Runtime 根据 ToolSpec 实施参数校验、治理门、超时、重试、输出校验和审计。Agent 不直接调用 handler，也不复制 Runtime 自动重试。Agent 只在 Runtime 返回最终结构化结果后决定：把失败作为 ToolMessage 返回模型、因治理错误立即停止，或因累计预算/循环防护停止。

## 错误分类

| Agent 分类 | ToolErrorCode | Agent 行为 |
| --- | --- | --- |
| `recoverable` | `unknown_tool`、`invalid_arguments`、`execution_error`、`invalid_result`、`output_validation_error` | 返回模型，可在预算内修正参数、换工具或结束 |
| `transient_exhausted` | `timeout`、`dependency_unavailable`、`rate_limited` | Runtime 重试已结束；可换方案，但不得无条件重复同一调用 |
| `terminal_governance` | `permission_denied`、`policy_denied`、`audit_write_failed`、`invalid_invocation_context`、`missing_idempotency_key` | 当前请求停止 |
| `approval_required` | `approval_required` | handler 不执行，当前请求停止；本 PR 不恢复审批 |
| `terminal_runtime` | `runtime_unavailable` | Runtime 整体不可用，当前请求停止 |

未知或空的失败码不会抛出未处理异常，采用 `terminal_runtime` 的保守默认分类，并映射为 `fatal_tool_error`。

## 终止治理错误

`permission_denied`、`policy_denied`、`audit_write_failed` 和 `invalid_invocation_context` 使用 `fatal_tool_error`；幂等键、审批和 Runtime 不可用分别使用稳定原因 `missing_idempotency_key`、`approval_required`、`runtime_unavailable`。这些错误不会再次返回模型尝试改参数绕过。

## 错误预算

`AGENT_MAX_TOOL_ERRORS=3` 表示累计第三次工具失败保存 Observation、ToolMessage 和 trace 后停止，不再进入下一轮模型。Guardrail 自己拦截的未执行调用不计作 Runtime 工具失败。

## 重复调用检测

调用指纹绑定规范化工具名和语义参数，使用稳定 JSON 与 SHA-256；字典 key 顺序不影响结果，trace 只保存哈希。连续相同调用计数跨模型轮次和同批调用延续，不同工具或参数会重置计数。默认最多执行两个连续完全相同调用，第三个在 StructuredTool/Runtime 前拦截。

## 重复错误检测

错误指纹绑定调用指纹与稳定错误码，不使用异常文本、路径、UUID 或 traceback。相同工具、相同参数和相同错误码连续第二次失败时，第二条失败 ToolMessage 仍会保存，然后以 `repeated_tool_error` 停止。成功或不同错误指纹会中断连续序列。

## 累计耗时预算

`AGENT_MAX_TOTAL_COST_MS=120000` 累计 Agent Model 节点（Provider、StructuredTool 构造、`bind_tools`、`model.invoke`）和每次工具调用的耗时。它是协作式预算：不能中断正在执行的单次模型或工具调用，但达到上限后不再开始下一模型或工具步骤。State 和 trace 只保存整数毫秒与数字预算快照，不保存时钟或 Runtime 对象。

## 同批 fail-closed

任一终止治理错误、审批要求或 Runtime 不可用出现后，同一 AIMessage 中后续调用都不会进入 StructuredTool、Runtime 或 handler。每个后续调用仍生成保留原模型 `tool_call_id` 的 Guardrail ToolMessage，`error_code=agent_batch_aborted`、`error_source=agent_guardrail`、`handler_executed=false`。预算和重复调用拦截同样不会留下悬空 tool call。

## ToolMessage 与 Trace

ToolMessage 保留原有结果、Runtime 和 governance 字段，并增加稳定的 `failure_kind`、`error_source`、`handler_executed`、`suggestion`、`call_fingerprint` 和 `stop_reason`。Trace 增加相同恢复字段、连续计数、累计错误数和纯数字 `budget_snapshot`；不会重复写入完整消息、chunk、原始敏感参数或异常堆栈。

## 测试分层

- 纯函数测试覆盖全部 ToolErrorCode、未知码、稳定指纹、连续计数、预算和 Stop Reason。
- 独立 Registry/LocalToolProvider 集成测试真实经过 StructuredTool、Registry、ToolRuntime 和 handler。
- Graph 测试使用 Fake Model，但调用真实 `build_agent_graph().compile().invoke()`。
- 知识库严格测试只替换 RAG 子图，真实经过 `SearchKnowledgeBaseArgs` 与完整工具链；成功零结果不计为错误。
- ChatService 测试调用真实 `ChatService.handle()`、真实 `run_agent_langgraph` 和编译 Graph，只替换模型及存储/metrics 等外部副作用。
- SQLite Checkpointer 覆盖成功、参数修正、重复调用停止和预算停止轨迹。
- 手动真实模型脚本验证供应商 Tool Calling 协议，不进入 Pytest/CI。

## 暂未实现

本 PR 不实现正式人工审批、LangGraph interrupt/resume、持久化幂等去重、强制取消正在运行的模型/handler、Streaming、多 Agent、MCP、Web、SQL 或代码沙箱。`approval_required` 当前只能受控停止，不能批准后恢复。

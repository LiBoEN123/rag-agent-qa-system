# ADR 0014：单一 Agent Result 契约与聊天状态策略

- 状态：Accepted
- 日期：2026-07-28
- 决策者：Project Maintainer

> 2026-08-21 Current implementation note：`AgentRunResult` 与唯一
> `ChatStatusPolicy` 决策仍完整生效；下文对 `AgentGraphState`/Checkpointer 的描述是
> 历史背景。当前 `results.py` 独占 Framework State 到 Result 的构造边界，ChatService
> 不重新构造或验证该 Result。

## 背景

PR12 已把一次请求的内部运行状态统一为平铺 `AgentGraphState`，但最终结果仍由
多个边界重复解释：

- Agent Runtime 把内部小写状态映射为大写结果状态，并手写正常/内部错误字典；
- ChatService 的可变配置表再次决定 HTTP、code/message、公开状态和字段覆盖；
- API Response Builder 反向导入 ChatService 配置，并维护两条构造路径；
- metrics 使用独立字典再次映射同一结果状态。

这种分散所有权允许新增状态时只修改其中一层，也让无生产者的旧状态继续表现为
受支持协议。

## 决策

1. `app/agent/results.py` 中的 `AgentRunResult` 是唯一完整 Runtime 结果模型。
   它使用项目已有 Pydantic，`extra="forbid"`，校验状态、confidence、LLM
   计数、references、tool trace 和 timings。
2. 正常 finalize、Graph Runtime 返回校验以及内部错误结果全部经过
   `AgentRunResult`。校验异常只向用户暴露稳定脱敏错误；日志不记录 Pydantic
   原始 input。
3. Pydantic 结果实例不写入 `AgentGraphState`。finalize 立即使用
   `model_dump(mode="json")`，因此 Checkpointer、Runtime 门面和 ChatService
   继续只接收普通 JSON-compatible 字典。
4. 内部 `AgentStatus → AgentResultStatus` 和
   `RAG status → AgentStatus` 的转换集中在 `app/agent/results.py`。RAG 子图
   自身协议不改。
5. `app/contracts/chat.py` 定义 `AgentResultStatus`、完整 `PublicStatus`、
   `MetricStatus`、不可变 `ChatStatusPolicy` 和唯一
   `CHAT_STATUS_POLICIES`。该中立模块只依赖 Python 标准库，不依赖 FastAPI、
   Pydantic Schema、ChatService、Agent Runtime、LangGraph、RAG 或 ToolRuntime。
6. 每个 `AgentResultStatus` 恰好对应一个 policy。HTTP status、code/message、
   `data.status`、metrics status、默认 reason、日志状态和响应字段覆盖都来自该
   policy；未知状态 fail closed 到 `INTERNAL_ERROR`。
7. ChatService 应用 policy 后才把 API-ready 结果交给控制器。API Response
   Builder 只规范化 confidence、提取 LLM usage 并实例化 Pydantic API Schema，
   不导入 ChatService、Agent Runtime 或 Graph。
8. `PublicStatus` 保留 `ok`、`no_evidence`、`empty_index`、`error`、
   `clarify` 和 `out_of_scope` 六个 OpenAPI 字面量。后两者中的部分值仍服务
   Search API 或外部兼容，不代表 Chat Runtime 必须生产对应 Result status。

## 状态边界

| 层 | 示例 | 所有者 |
| --- | --- | --- |
| 内部 Agent 状态 | `ok`、`clarify`、`model_error` | `app/agent/state.py` |
| Agent Result 状态 | `OK`、`NEED_CLARIFICATION`、`MODEL_ERROR` | `AgentResultStatus` |
| HTTP/API 状态 | HTTP 200/500、`ok`/`clarify`/`error` | `ChatStatusPolicy` |
| metrics 状态 | `success`、`clarify`、`model_error` | `ChatStatusPolicy.metric_status` |

`MODEL_BAD_OUTPUT` 仍由 RAG 子图在模型结构解析失败时真实产生，但进入 Agent 时
统一映射为内部 `model_error`，所以它不是 Chat Result status。
`OUT_OF_SCOPE` 在当前 Chat Runtime 没有生产者；它只存在于旧响应/metrics 映射、
评测兼容或完整 PublicStatus 字面量中。因此删除 Chat Result 分支，同时保留 Search
API 的 `out_of_scope` OpenAPI 字面量和与 Result 状态无关的历史日志读取能力。

## 服务级特殊结果

空问题不属于 Agent Runtime 执行结果，继续返回 HTTP 400、
`code=EMPTY_QUESTION` 和 `data.status=clarify`，但 result 通过同一模型 helper
构造成完整字段。

ChatService 调用 Runtime 本身抛出的未知异常继续保留现有公开 HTTP 500、
`code=MODEL_ERROR` 和 `message=处理失败，请稍后重试`，避免本 PR 引入额外外部
兼容变化。该分支的完整 result status 与 metrics status 均为内部错误。这是服务级
兼容例外，不加入 `AgentResultStatus` 或 `CHAT_STATUS_POLICIES`。

## 外部兼容

六个真实 Agent Result status 的主要行为保持：

| Result status | HTTP | code | `data.status` | metrics |
| --- | ---: | --- | --- | --- |
| `OK` | 200 | `0` | `ok` | `success` |
| `NEED_CLARIFICATION` | 200 | `NEED_CLARIFICATION` | `clarify` | `clarify` |
| `NO_RELEVANT_INFO` | 200 | `NO_RELEVANT_INFO` | `no_evidence` | `no_relevant_info` |
| `TOOL_ERROR` | 200 | `TOOL_ERROR` | `error` | `tool_error` |
| `MODEL_ERROR` | 200 | `MODEL_ERROR` | `error` | `model_error` |
| `INTERNAL_ERROR` | 500 | `INTERNAL_ERROR` | `error` | `internal_error` |

澄清继续清空 answer/references 并保留 clarification question；错误状态继续清空
不可信 references。`/chat` 请求/响应字段、OpenAPI PublicStatus、SQLite
session/Checkpointer Schema、thread ID、RAG references、八个 ToolSpec、Provider
可见性、ToolRuntime 治理、metrics 字段、bad case、session 和
`tool_call_chain` 不变。

## 后果

删除 `map_agent_status_to_public_status()`、`map_rag_status()`、
`build_v2_result()`、`CHAT_RESULT_RESPONSE_MAP`、可变 `response_map` 注入、
`response_config`、`normalize_metrics_status()` 的独立字典、
`ChatService._response_result()` 和 API 的第二条 result builder。新增状态必须先
加入 `AgentResultStatus` 和唯一 policy，完整性测试与 Runtime inspection 会拒绝
缺失或多余分支。

本决策不迁移 Settings，不引入 pydantic-settings 或 Import Linter，不重设计 API
Schema，不拆分完整 ChatService，不修改 Tool Trace/RAG/ToolSpec/权限/审计/
SQLite/Checkpointer，也不处理历史文档归档。这些事项继续独立演进。

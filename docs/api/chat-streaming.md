# `/chat/stream` SSE 接口

`POST /chat/stream` 是唯一聊天 HTTP 入口。Vue 通过 `ChatService.stream()` 调用
`stream_agent()`，继续使用 Query Rewrite、直接 `create_agent`、Agent/Tool Runtime、
`AgentRunResult`、`ChatStatusPolicy`、Session 和 telemetry。

## HTTP 语义

`ChatRequest` 只接受 `question`、`session_id`、`tool_mode`（默认 `full`）：

```bash
curl --no-buffer -X POST "http://127.0.0.1:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "X-Trace-Id: demo-stream-001" \
  -d '{
    "question": "缓考应该如何办理？",
    "session_id": "demo_session_001"
  }'
```

- 旧 `top_k`、`decompose_top_k` 及其他未知字段由 `extra="forbid"` 拒绝，返回 HTTP 422，不会静默忽略。
- `rag_only` 仅允许 search_knowledge_base、search_document、get_chunk_context 正文工具及 find_documents 目录工具；仍需 `knowledge_base:read` 权限，Web 被排除；MCP 仅用于知识入库，不属于 Agent 工具集合。
- 每次检索的 query/top_k 由 Agent 通过工具 schema 选择；`/search` 仍接受诊断 top_k。
- Schema 校验失败：FastAPI 在建流前返回 `HTTP 422 application/json`。
- 通过 Schema 校验的请求（包括空问题）：返回 `HTTP 200 text/event-stream`。
- 问题去除空白后为空：发送 `start → final → done(completed)`；
  `final.data.http_status=400`，`final.data.response.code=EMPTY_QUESTION`。
- 流建立后的澄清、无证据、工具错误、模型错误和内部错误都是业务结果，通过
  `final.data.http_status`、`final.data.response.code` 和
  `final.data.response.data.status` 表达，不转换成 `error` 事件。

响应头始终提供 `X-Trace-Id`。响应头建立时业务尚未完成，不提供
`X-Total-Cost-Ms`，真实总耗时位于
`final.data.total_cost_ms`。

生产响应已移除 `decompose_llm_used`；保留 `answer_llm_used`、`llm_call_count` 和固定 false 的 `router_llm_used`。Chat metrics 保留 route/agent 总耗时及工具、模型 trace，不再发布 legacy RAG 的补零 timing 字段。

## 事件信封与顺序

每个事件的 SSE `event` 是事件类型，`id` 等于递增 sequence，JSON `data` 是统一
信封：

```json
{
  "schema_version": 1,
  "trace_id": "demo-stream-001",
  "sequence": 1,
  "timestamp": "2026-08-02T12:00:00.000",
  "data": {}
}
```

正常流的边界顺序是 `start → progress/tool/answer_delta... → final → done`：

| 事件 | 公开字段 | 语义 |
| --- | --- | --- |
| `start` | `session_id_present` | 第一条业务事件；只说明是否存在 session ID |
| `progress` | `stage`、`status` | 高层阶段；stage 为 `query_rewrite/routing/agent/rag/finalizing` |
| `tool` | `tool_name`、`phase`、`status`、完成时的 `cost_ms` | 安全工具开始/完成状态 |
| `answer_delta` | `text` | Evidence Gate 通过后，正文模型实际生成的公开文本 chunk；非 Document 回答保留单次交付 |
| `final` | `http_status`、完整 `response`、`total_cost_ms` | 唯一权威业务结果 |
| `error` | 固定 `STREAM_ERROR` 和通用消息 | 仅表示 SSE 协议自身无法继续 |
| `done` | `reason=completed/error` | 正常可发送结束事件 |

客户端必须以 `final.data.response` 作为最终结果，不能只依赖拼接
`answer_delta`。澄清、治理失败等请求可能没有 `answer_delta`，但仍会实时收到进度和工具
事件。成功的 Document 回答保证 `"".join(answer_delta_parts) == final.data.response.data.answer`，
包括正文首尾的空格、换行；收到部分文本不代表业务已成功。

## 答案生成

基于 `search_knowledge_base`、`search_document`、`get_chunk_context`、`web_fetch`、
`web_pdf_fetch` 返回的有效 Document 的回答，在 `rag_only` / `full` 模式均使用
**model chunk streaming**：

```text
create_agent.astream(updates/messages/values, version="v2")
→ 完整 structured_response（outcome、evidence_ids）
→ Evidence Gate
→ get_answer_model().astream(问题 + 已验证、已选中的 Document)
→ AIMessageChunk.text → answer_delta...
→ AgentRunResult → final → done
```

沿用唯一 `FinalAgentResponse` schema：使用 Document 时 `answer` 留空，由结构化响应
先决定依据什么回答；兼容模型仍填写正文的情况，但该正文不作为流式答案发送。
正文模型只接收有效问题、选中证据的 ID/正文/来源和回答约束，不接收 Agent State、
工具历史、原始工具 JSON 或未选中的证据。它不绑定工具，只根据已锁定证据生成回答。
chunk 边界由模型/供应商决定，不保证每个 token 单独对应事件，也不会把完整答案切片重放。

无 Document 的目录操作结果，以及 `no_evidence`，保留结构化结果
完成后的单次文本交付；不宣称这些场景已实现模型正文流式。`clarify`、错误和直接路由
回复无需正文模型流。full 模式若混合使用多类工具且选择了 Document，正文仅依据选中的 Document。

## 安全投影

`updates` 只投影工具名、状态和耗时；`values` 只在 Runtime 内捕获最终状态，不向 SSE
序列化。`messages` 中的 `AIMessage`/`AIMessageChunk` 仅用于业务工具开始事件，
普通模型文本和 ToolStrategy 合成调用均不直接发送。首次正文 `answer_delta` 之前，
Agent 已完成且没有终止治理错误，结构化 outcome 和 evidence_ids 已确定。Runtime 与
Result Builder 复用同一 Evidence Gate：只从本次成功正文工具的 `ToolMessage` 恢复并
验证 Document，拒绝未知 ID、重复 ID、缺少应声明的引用、无效 Document 和无效结构化响应。
校验失败不会启动正文模型，也不会产生答案 delta。

只有正文模型的 `AIMessageChunk.text` 进入增量事件，框架负责提取 text block，
reasoning、工具调用和 metadata 不做文本投影。正文流失败、空输出或超时会形成
安全的错误 `final`；已发出的部分文本不升级为成功答案，客户端以错误 final 覆盖临时显示。

增量事件不会公开 reasoning/thinking、tool call JSON、工具参数、`ToolMessage` 正文、
原始工具结果、RAG JSON、网页/PDF/chunk 正文、系统 Prompt、provider/usage
metadata、内部 State、认证头、Cookie、API Key、数据库凭据、本地敏感路径或异常
堆栈。`final` 复用现有 `ChatResponse` 映射和公开字段合同，不另建第二套响应。

## Session、telemetry 与断开

正常完成和业务错误都只保存一轮 user/最终 assistant 消息，并只写一条最终
metrics；不会保存 delta、progress、tool event 或中间 State。流式 metrics 在原字段
之外记录事件数、delta 数/字符数、首事件/首 delta 耗时、客户端断开和 delta 与最终
答案一致性；不会为每个 token 单独写 metrics。

每次正文生成尝试追加一条现有 model governance event，成功、失败和总超时均计入
`llm_call_count`；Query Rewrite 仍由 ChatService 单独合并一次。Agent 的原生模型调用
限制继续生效，正文调用也必须有 `agent_max_iterations` 剩余名额；工具上限不受正文调用影响。
`agent_max_total_cost_ms` 沿用 Runtime 同一个 `asyncio.timeout`，覆盖 Agent 循环、证据
校验及正文流，不在第二阶段重新计时。模型工厂的 provider timeout/retry 设置继续有效，
不增加额外重试层；流中途失败不会自动重试或重放已输出文本。
Runtime 内部的 `llm_usage` 通知仅更新 Service telemetry，不进入公开 SSE；用户在
正文生成中途取消时，已记录的 Agent 调用与正文生成尝试仍保留在 metrics 中。

客户端断开后，HTTP adapter 停止读取并关闭 Service/Agent async generator，取消会
尽可能向下传播。不再发送 `final`/`done`，不保存部分 assistant 文本，不把断开转换成
`MODEL_ERROR`，也不写伪 bad case；metrics 记录 `client_disconnected=true`。已经进入
第三方同步 handler 的调用以及 provider 请求采用 best-effort cancel，不保证立刻终止；
Runtime 同时关闭正在读取的模型异步流，断开后不会继续向客户端输出。

## Vue Demo

`frontend/src/views/ChatView.vue` 只使用本接口，POST SSE 由
`@microsoft/fetch-event-source` 处理。页面展示 `progress` 和 `tool`，即时显示
`answer_delta`，收到 `final.data.response` 后用完整 `ChatResponse` 覆盖本轮结果。
没有 delta 的请求继续显示进度并等待 `final`。中断、切换页面或取消生成不会
自动重发 POST；用户可显式重新发送。
流收到 final 后若未正常 done，保留 final 并显示中断错误。新建对话会中止旧请求并清空消息。

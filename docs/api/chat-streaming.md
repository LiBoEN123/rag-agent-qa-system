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
- `rag_only` 仅允许 search_knowledge_base、search_document、get_chunk_context 正文工具及 find_documents 目录工具；仍需 `knowledge_base:read` 权限，Web/MCP 被排除。
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
| `answer_delta` | `text` | 通过 structured-response 与 evidence-ID 校验的完整答案文本 |
| `final` | `http_status`、完整 `response`、`total_cost_ms` | 唯一权威业务结果 |
| `error` | 固定 `STREAM_ERROR` 和通用消息 | 仅表示 SSE 协议自身无法继续 |
| `done` | `reason=completed/error` | 正常可发送结束事件 |

客户端必须以 `final.data.response` 作为最终结果，不能只依赖拼接
`answer_delta`。部分 RAG 请求可能没有 `answer_delta`，但仍会实时收到进度和工具
事件，完整答案始终包含在 `final` 中。

## 安全投影

`updates` 只投影工具名、状态和耗时；`values` 只在 Runtime 内捕获最终状态，不向 SSE
序列化。`messages` 中的 `AIMessage`/`AIMessageChunk` 仅用于业务工具开始事件，
普通模型文本和 ToolStrategy 合成调用均不直接发送。最终 `structured_response` 通过
Result Builder 的 evidence-ID 校验后，answer/no_evidence 在 result 前发送一次完整
`answer_delta`，与 `final.data.response.answer` 一致。当前不提供逐 token 答案流；
治理失败、未知引用或缺少结构化响应不会提前泄漏候选答案。

增量事件不会公开 reasoning/thinking、tool call JSON、工具参数、`ToolMessage` 正文、
原始工具结果、RAG JSON、网页/PDF/chunk 正文、系统 Prompt、provider/usage
metadata、内部 State、认证头、Cookie、API Key、数据库凭据、本地敏感路径或异常
堆栈。`final` 复用现有 `ChatResponse` 映射和公开字段合同，不另建第二套响应。

## Session、telemetry 与断开

正常完成和业务错误都只保存一轮 user/最终 assistant 消息，并只写一条最终
metrics；不会保存 delta、progress、tool event 或中间 State。流式 metrics 在原字段
之外记录事件数、delta 数/字符数、首事件/首 delta 耗时、客户端断开和 delta 与最终
答案一致性；不会为每个 token 单独写 metrics。

客户端断开后，HTTP adapter 停止读取并关闭 Service/Agent async generator，取消会
尽可能向下传播。不再发送 `final`/`done`，不保存部分 assistant 文本，不把断开转换成
`MODEL_ERROR`，也不写伪 bad case；metrics 记录 `client_disconnected=true`。已经进入
第三方同步 handler 的调用不保证立刻终止，但断开后不会继续向客户端输出。

## 手动冒烟

先启动服务：

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

再运行只打印安全摘要的客户端：

```bash
python scripts/smoke_test_chat_stream.py \
  --base-url http://127.0.0.1:8000 \
  --question "缓考应该如何办理？"
```

该脚本检查 content type、sequence、SSE id、trace ID、`final` 和 `done`；不会打印
完整答案、工具原始结果、网页/PDF/chunk 正文或凭据。它是人工真实服务检查，不在
Pytest/CI 中运行。

## Vue Demo

`frontend/src/views/ChatView.vue` 只使用本接口，POST SSE 由
`@microsoft/fetch-event-source` 处理。页面展示 `progress` 和 `tool`，即时显示
`answer_delta`，收到 `final.data.response` 后用完整 `ChatResponse` 覆盖本轮结果。
没有 delta 的 RAG 请求继续显示进度并等待 `final`。中断、切换页面或取消生成不会
自动重发 POST；用户可显式重新发送。
流收到 final 后若未正常 done，保留 final 并显示中断错误。新建对话会中止旧请求并清空消息。

## 性能测量

`scripts/benchmark_chat_api.py` 使用当前环境的 `httpx-sse` 消费本接口。
`latency_ms` 从发起 POST 计时到收到合法 `done`，业务码、状态、答案长度和引用数
取自 `final.data.response`。`http_status` 记录实际 HTTP 传输状态；空问题为 200，
业务失败仍按 `EMPTY_QUESTION` 统计。缺失 final/done、非法终止顺序或流错误均记为失败，
不会自动重试请求。

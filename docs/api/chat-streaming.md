# `/chat/stream` SSE 接口

`POST /chat/stream` 是现有 `POST /chat` 的流式补充，不替代同步接口。两者复用同一个
`ChatService`、Query Rewrite、直接 `create_agent`、Agent/Tool/RAG Runtime、
`AgentRunResult`、`ChatStatusPolicy`、Session 和 telemetry；区别只在传输方式。

## HTTP 语义

请求体仍是现有 `ChatRequest`：

```bash
curl --no-buffer -X POST "http://127.0.0.1:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "X-Trace-Id: demo-stream-001" \
  -d '{
    "question": "缓考应该如何办理？",
    "top_k": 5,
    "decompose_top_k": 10,
    "session_id": "demo_session_001"
  }'
```

- Schema 校验失败：FastAPI 在建流前返回 `HTTP 422 application/json`。
- 问题去除空白后为空：在建流前返回 `HTTP 400 application/json` 和标准
  `ChatResponse`。
- 合法非空请求：返回 `HTTP 200 text/event-stream`。
- 流建立后的澄清、无证据、工具错误、模型错误和内部错误都是业务结果，通过
  `final.data.http_status`、`final.data.response.code` 和
  `final.data.response.data.status` 表达，不转换成 `error` 事件。

响应头始终提供 `X-Trace-Id`。同步 `/chat` 仍提供 `X-Total-Cost-Ms`；流式响应头
建立时业务尚未完成，因此不提供不准确的总耗时，真实值位于
`final.data.total_cost_ms`。

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
| `answer_delta` | `text` | 已确认属于直接回答的普通文本增量 |
| `final` | `http_status`、完整 `response`、`total_cost_ms` | 唯一权威业务结果 |
| `error` | 固定 `STREAM_ERROR` 和通用消息 | 仅表示 SSE 协议自身无法继续 |
| `done` | `reason=completed/error` | 正常可发送结束事件 |

客户端必须以 `final.data.response` 作为最终结果，不能只依赖拼接
`answer_delta`。部分 RAG 请求可能没有 `answer_delta`，但仍会实时收到进度和工具
事件，完整答案始终包含在 `final` 中。

## 安全投影

`updates` 只投影工具名、状态和耗时；`values` 只在 Runtime 内捕获最终状态，不向 SSE
序列化。`messages` 只接受 `create_agent` 模型节点中的
`AIMessage`/`AIMessageChunk`
普通字符串或显式 text block。一次模型调用的 chunk 会先缓冲，确认没有 tool call
且不属于 RAG 命名空间后才作为原始增量释放。

增量事件不会公开 reasoning/thinking、tool call JSON、工具参数、`ToolMessage` 正文、
原始 `ToolResult`、RAG JSON、网页/PDF/chunk 正文、系统 Prompt、provider/usage
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

## Streamlit Demo

`demo/streamlit_app.py` 的聊天面板默认使用本接口。页面把 `progress` 和 `tool` 显示为
请求状态，把安全 `answer_delta` 写入临时占位框；只有收到并校验 `final` 后，才把完整
`ChatResponse` 保存到前端消息历史。没有 delta 的 RAG 请求会保持进度提示并等待
`final`。侧边栏保留显式同步模式，但流已经建立或收到事件后不会自动调用 `/chat`
重试，避免重复工具调用、Session 写入和 metrics。

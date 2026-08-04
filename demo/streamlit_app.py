from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120
ApiResult = tuple[dict[str, Any] | None, int | None, str | None]
StreamEventHandler = Callable[[str, dict[str, Any]], None]

CHAT_STREAM_EVENTS = frozenset(
    {"start", "progress", "tool", "answer_delta", "final", "error", "done"}
)
PROGRESS_STAGE_LABELS = {
    "query_rewrite": "检查会话上下文",
    "routing": "分析问题处理路径",
    "agent": "执行 Agent",
    "rag": "整理知识库证据",
    "finalizing": "确认最终结果",
}


class ChatStreamProtocolError(ValueError):
    """表示后端聊天流不符合当前公开 SSE 合同。"""


def new_session_id() -> str:
    """生成前端会话 ID，用于让后端识别同一轮多轮对话。"""
    return f"streamlit_{uuid.uuid4().hex}"


def init_chat_state() -> None:
    """初始化聊天页所需的 Streamlit 会话状态。"""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = new_session_id()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def reset_chat_state() -> None:
    """开始一段新对话，清空前端消息并换用新的后端 session_id。"""
    st.session_state["session_id"] = new_session_id()
    st.session_state["messages"] = []


def normalize_backend_url(raw_url: str) -> str:
    """规范化后端地址，去掉首尾空白和末尾斜杠，空值时使用默认地址。"""
    url = (raw_url or DEFAULT_BACKEND_URL).strip().rstrip("/")
    return url or DEFAULT_BACKEND_URL


def request_json(
    backend_url: str,
    endpoint: str,
    method: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> ApiResult:
    """请求后端 JSON 接口，并统一返回响应数据、状态码和错误信息。"""
    url = f"{normalize_backend_url(backend_url)}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            params=params,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        return (
            None,
            None,
            f"无法连接后端：{url}。请先启动 FastAPI 服务：uvicorn app.main:app --reload",
        )
    except requests.exceptions.Timeout:
        return (
            None,
            None,
            f"请求超时：{url}。后端可能仍在加载模型或处理请求，请稍后重试。",
        )
    except requests.exceptions.RequestException as exc:
        return None, None, f"请求失败：{url}，错误信息：{exc}"

    try:
        data = response.json()
    except ValueError:
        body_preview = response.text[:500] if response.text else "<empty response>"
        return (
            None,
            response.status_code,
            f"后端返回了非 JSON 响应，HTTP {response.status_code}：{body_preview}",
        )

    return data, response.status_code, None


def post_json(
    backend_url: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> ApiResult:
    """向指定后端接口发送 JSON POST 请求。"""
    return request_json(
        backend_url,
        endpoint,
        method="POST",
        payload=payload or {},
        timeout=timeout,
    )


def get_json(
    backend_url: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> ApiResult:
    """向指定后端接口发送 JSON GET 请求。"""
    return request_json(
        backend_url,
        endpoint,
        method="GET",
        params=params,
        timeout=timeout,
    )


def iter_sse_events(
    lines: Iterable[str | bytes],
) -> Iterator[dict[str, str]]:
    """把 HTTP 行流解析为 SSE event/id/data 字段，忽略心跳注释。"""
    fields: dict[str, str] = {}
    data_lines: list[str] = []
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ChatStreamProtocolError("SSE 响应不是 UTF-8") from exc
        else:
            line = raw_line
        line = line.rstrip("\r\n")

        if not line:
            if fields or data_lines:
                fields["data"] = "\n".join(data_lines)
                yield fields
                fields = {}
                data_lines = []
            continue
        if line.startswith(":"):
            continue

        name, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if name == "data":
            data_lines.append(value)
        elif name in {"event", "id"}:
            fields[name] = value

    if fields or data_lines:
        fields["data"] = "\n".join(data_lines)
        yield fields


def consume_chat_stream_events(
    lines: Iterable[str | bytes],
    *,
    on_event: StreamEventHandler | None = None,
) -> ApiResult:
    """校验聊天 SSE 合同、通知 UI，并返回 final 中的权威响应。"""
    handler = (
        on_event
        if on_event is not None
        else (lambda event, envelope: None)
    )
    expected_sequence = 1
    trace_id = ""
    event_count = 0
    final_response: dict[str, Any] | None = None
    final_http_status: int | None = None
    final_seen = False
    error_seen = False
    done_seen = False

    for fields in iter_sse_events(lines):
        if done_seen:
            raise ChatStreamProtocolError("done 后仍收到业务事件")

        event = fields.get("event", "")
        if event not in CHAT_STREAM_EVENTS:
            raise ChatStreamProtocolError("收到未知 SSE 事件")
        if event_count == 0 and event != "start":
            raise ChatStreamProtocolError("聊天流第一条事件不是 start")
        if event_count > 0 and event == "start":
            raise ChatStreamProtocolError("聊天流重复发送 start")
        if (final_seen or error_seen) and event != "done":
            raise ChatStreamProtocolError("final 或 error 后只能发送 done")

        try:
            envelope = json.loads(fields.get("data", ""))
        except json.JSONDecodeError as exc:
            raise ChatStreamProtocolError("SSE data 不是有效 JSON") from exc
        if not isinstance(envelope, dict):
            raise ChatStreamProtocolError("SSE 信封不是 JSON object")
        if envelope.get("schema_version") != 1:
            raise ChatStreamProtocolError("SSE schema_version 不受支持")
        timestamp = envelope.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ChatStreamProtocolError("SSE timestamp 缺失")

        sequence = envelope.get("sequence")
        if isinstance(sequence, bool) or sequence != expected_sequence:
            raise ChatStreamProtocolError("SSE sequence 不连续")
        if fields.get("id") != str(sequence):
            raise ChatStreamProtocolError("SSE id 与 sequence 不一致")

        current_trace_id = envelope.get("trace_id")
        if not isinstance(current_trace_id, str) or not current_trace_id:
            raise ChatStreamProtocolError("SSE trace_id 缺失")
        if trace_id and current_trace_id != trace_id:
            raise ChatStreamProtocolError("SSE trace_id 在请求中发生变化")
        trace_id = current_trace_id

        data = envelope.get("data")
        if not isinstance(data, dict):
            raise ChatStreamProtocolError("SSE data 字段不是 object")

        if event == "final":
            response_json = data.get("response")
            http_status = data.get("http_status")
            if not isinstance(response_json, dict):
                raise ChatStreamProtocolError("final 缺少标准 ChatResponse")
            if isinstance(http_status, bool) or not isinstance(http_status, int):
                raise ChatStreamProtocolError("final http_status 非法")
            final_response = response_json
            final_http_status = http_status
            final_seen = True
        elif event == "error":
            error_seen = True
        elif event == "done":
            reason = data.get("reason")
            if reason == "completed" and not final_seen:
                raise ChatStreamProtocolError("completed 流缺少 final")
            if reason == "error" and not error_seen:
                raise ChatStreamProtocolError("error 流缺少 error 事件")
            if reason not in {"completed", "error"}:
                raise ChatStreamProtocolError("done reason 非法")
            done_seen = True

        handler(event, envelope)
        event_count += 1
        expected_sequence += 1

    if event_count == 0:
        raise ChatStreamProtocolError("后端没有返回 SSE 事件")
    if not done_seen:
        raise ChatStreamProtocolError("聊天流未发送 done")
    if final_response is not None:
        return final_response, final_http_status, None
    if error_seen:
        return None, 200, "流式处理失败"
    raise ChatStreamProtocolError("聊天流没有权威 final 结果")


def request_chat_stream(
    backend_url: str,
    payload: dict[str, Any],
    *,
    on_event: StreamEventHandler | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> ApiResult:
    """调用聊天 SSE 接口；建流前 JSON 错误仍按普通响应返回。"""
    url = f"{normalize_backend_url(backend_url)}/chat/stream"
    try:
        with requests.post(
            url,
            json=payload,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                try:
                    data = response.json()
                except ValueError:
                    return (
                        None,
                        response.status_code,
                        f"后端返回 HTTP {response.status_code}，但没有标准 JSON 错误响应。",
                    )
                if not isinstance(data, dict):
                    return (
                        None,
                        response.status_code,
                        "后端错误响应不是 JSON object。",
                    )
                return data, response.status_code, None

            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("text/event-stream"):
                return None, 200, "后端没有返回 text/event-stream。"
            response.encoding = "utf-8"
            try:
                return consume_chat_stream_events(
                    response.iter_lines(chunk_size=1, decode_unicode=True),
                    on_event=on_event,
                )
            except ChatStreamProtocolError as exc:
                return None, response.status_code, f"流式响应协议错误：{exc}"
    except requests.exceptions.ConnectionError:
        return (
            None,
            None,
            f"无法连接后端：{url}。请先启动 FastAPI 服务：uvicorn app.main:app --reload",
        )
    except requests.exceptions.Timeout:
        return None, None, f"流式请求超时：{url}。请稍后重试。"
    except requests.exceptions.RequestException as exc:
        return None, None, f"流式请求失败：{url}，错误信息：{exc}"


@st.cache_data(ttl=5, show_spinner=False)
def request_health(backend_url: str) -> ApiResult:
    """短时缓存健康检查，避免每次 Streamlit 重跑都访问后端。"""
    return get_json(backend_url, "/health", timeout=5)


@st.cache_data(ttl=30, show_spinner=False)
def request_kb_summary(backend_url: str) -> ApiResult:
    """短时缓存知识库摘要，避免聊天交互重复执行摘要查询。"""
    return post_json(backend_url, "/kb_summary", payload={}, timeout=20)


def remember_api_result(
    state_key: str,
    backend_url: str,
    result: ApiResult,
) -> None:
    """把按需请求结果保存在当前页面会话中。"""
    st.session_state[state_key] = {
        "backend_url": normalize_backend_url(backend_url),
        "result": result,
    }


def recall_api_result(state_key: str, backend_url: str) -> ApiResult | None:
    """读取当前后端地址对应的按需请求结果。"""
    stored = st.session_state.get(state_key)
    if not stored:
        return None
    if stored.get("backend_url") != normalize_backend_url(backend_url):
        return None
    return stored.get("result")


def render_http_warning(endpoint: str, status_code: int | None) -> None:
    """当接口返回 4xx/5xx 状态码时，在页面上提示 HTTP 警告。"""
    if status_code is not None and status_code >= 400:
        st.warning(
            f"{endpoint} 返回 HTTP {status_code}。下方仍会展示后端返回的结构化响应。"
        )


def pick(data: dict[str, Any], key: str, default: Any = "") -> Any:
    """从字典中安全读取字段，缺失或值为 None 时返回默认值。"""
    value = data.get(key, default)
    return default if value is None else value


def table_rows(items: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    """按指定列名整理列表数据，生成适合 st.dataframe 展示的行数据。"""
    return [{column: item.get(column, "") for column in columns} for item in items]


def render_chat_response_details(
    response_json: dict[str, Any],
    status_code: int | None,
    *,
    endpoint: str = "/chat",
) -> None:
    """渲染聊天接口的指标、引用、工具链路和完整 JSON。"""
    with st.expander("本轮详情", expanded=False):
        render_http_warning(endpoint, status_code)

        data = response_json.get("data") or {}
        trace_id = pick(response_json, "trace_id", "")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("status", str(pick(data, "status", "")))
        col2.metric("retrieval", str(pick(data, "retrieval_confidence", "")))
        col3.metric("answer", str(pick(data, "answer_confidence", "")))
        col4.metric("action", str(pick(data, "action", "")))
        col5.metric("trace_id", str(trace_id))

        st.write("reason:", pick(data, "reason", ""))

        references = data.get("references") or []
        st.subheader("references")
        if not references:
            st.info("本次响应没有 references。")
        else:
            columns = [
                "reference_id",
                "file_name",
                "chunk_id",
                "score",
                "rerank_score",
                "source_path",
            ]
            st.dataframe(
                table_rows(references, columns),
                width="stretch",
                hide_index=True,
            )

            for idx, reference in enumerate(references, start=1):
                reference_id = pick(reference, "reference_id", f"Reference {idx}")
                file_name = pick(reference, "file_name", "")
                st.markdown(f"**{reference_id} | {file_name}**")
                st.write(pick(reference, "snippet", ""))

        tool_trace = data.get("tool_trace") or []
        st.subheader("tool_trace")
        if not tool_trace:
            st.info("本次响应没有 tool_trace。")
        else:
            columns = [
                "step",
                "action",
                "tool_name",
                "tool_status",
                "final_status",
                "router_source",
                "reason",
            ]
            st.dataframe(
                table_rows(tool_trace, columns),
                width="stretch",
                hide_index=True,
            )
            st.json(tool_trace, expanded=False)

        st.subheader("完整响应 JSON")
        st.json(response_json, expanded=False)


def render_health_status(backend_url: str) -> None:
    """在侧边栏展示后端健康状态。"""
    st.sidebar.subheader("服务状态")

    response_json, status_code, error = request_health(backend_url)
    if error:
        st.sidebar.error("后端不可用")
        st.sidebar.caption(error)
        return

    data = (response_json or {}).get("data") or {}
    if status_code is not None and status_code < 400 and data.get("status") == "ok":
        st.sidebar.success("后端在线")
    else:
        st.sidebar.warning(f"健康检查异常（HTTP {status_code}）")


def render_kb_summary(backend_url: str) -> None:
    """请求并渲染缓存后的知识库状态信息。"""
    st.sidebar.subheader("知识库状态")

    response_json, status_code, error = request_kb_summary(backend_url)

    if error:
        st.sidebar.error(error)
        return

    if status_code is not None and status_code >= 400:
        st.sidebar.warning(f"/kb_summary 返回 HTTP {status_code}")

    data = (response_json or {}).get("data") or {}

    st.sidebar.write("ready:", data.get("ready", ""))
    st.sidebar.write("doc_count:", data.get("doc_count", ""))
    st.sidebar.write("chunk_count:", data.get("chunk_count", ""))
    st.sidebar.write("index_source:", data.get("index_source", ""))

    documents = data.get("documents") or []
    file_names = data.get("file_names") or [
        doc.get("file_name", "") for doc in documents
    ]

    with st.sidebar.expander(f"文档列表 ({len(file_names)})", expanded=True):
        if not file_names:
            st.info("暂无文档。")
        else:
            for file_name in file_names:
                st.write(file_name)

    with st.sidebar.expander("kb_summary 完整 JSON", expanded=False):
        st.json(response_json)


def response_from_result(
    endpoint: str, result: ApiResult | None
) -> dict[str, Any] | None:
    """处理按需请求的公共错误和 HTTP 状态展示。"""
    if result is None:
        return None

    response_json, status_code, error = result
    if error:
        st.error(error)
        return None

    render_http_warning(endpoint, status_code)
    return response_json or {}


def render_raw_response(endpoint: str, response_json: dict[str, Any]) -> None:
    """折叠展示诊断接口的完整 JSON。"""
    with st.expander(f"{endpoint} 完整 JSON", expanded=False):
        st.json(response_json)


def render_search_result(result: ApiResult | None) -> None:
    """渲染 /search 的检索状态、分数和片段。"""
    response_json = response_from_result("/search", result)
    if response_json is None:
        return

    data = response_json.get("data") or {}
    results = data.get("results") or []
    col1, col2 = st.columns(2)
    col1.metric("status", str(pick(data, "status", "")))
    col2.metric("结果数", len(results))
    st.write("reason:", pick(data, "reason", ""))

    if not results:
        st.info("本次检索没有返回候选片段。")
    else:
        columns = [
            "reference_id",
            "file_name",
            "chunk_id",
            "score",
            "bm25_score",
            "vector_score",
            "rerank_score",
        ]
        st.dataframe(
            table_rows(results, columns),
            width="stretch",
            hide_index=True,
        )
        for idx, item in enumerate(results, start=1):
            reference_id = pick(item, "reference_id", f"Reference {idx}")
            file_name = pick(item, "file_name", "")
            with st.expander(f"{reference_id} | {file_name}", expanded=False):
                st.write(pick(item, "snippet", ""))
                st.caption(f"source_path: {pick(item, 'source_path', '')}")

    render_raw_response("/search", response_json)


def render_search_panel(backend_url: str, default_top_k: int) -> None:
    """提供独立检索调试表单。"""
    st.subheader("检索调试")
    st.caption("直接查看召回、重排和证据门控结果，不生成最终回答。")

    with st.form("search_form"):
        question = st.text_area(
            "检索问题",
            placeholder="请输入要检索的问题...",
            key="search_question",
        )
        search_top_k = st.number_input(
            "召回条数",
            min_value=1,
            max_value=10,
            value=int(default_top_k),
            step=1,
            key="search_top_k",
        )
        submitted = st.form_submit_button("执行检索", type="primary")

    if submitted:
        if not question.strip():
            st.warning("请输入检索问题。")
        else:
            with st.spinner("正在检索并重排..."):
                result = post_json(
                    backend_url,
                    "/search",
                    payload={
                        "question": question.strip(),
                        "top_k": int(search_top_k),
                    },
                )
            remember_api_result("search_result", backend_url, result)

    render_search_result(recall_api_result("search_result", backend_url))


def render_docs_result(result: ApiResult | None) -> None:
    """渲染 /docs-preview 的文档注册信息。"""
    response_json = response_from_result("/docs-preview", result)
    if response_json is None:
        return

    documents = response_json.get("data") or []
    st.metric("文档数", response_json.get("count", len(documents)))
    if not documents:
        st.info("知识库中没有可展示的文档。")
    else:
        columns = ["doc_id", "file_name", "file_type", "chunk_count", "source_path"]
        st.dataframe(
            table_rows(documents, columns),
            width="stretch",
            hide_index=True,
        )

    render_raw_response("/docs-preview", response_json)


def render_chunks_result(result: ApiResult | None) -> None:
    """渲染 /chunks-preview 的切片预览。"""
    response_json = response_from_result("/chunks-preview", result)
    if response_json is None:
        return

    chunks = response_json.get("data") or []
    st.metric("切片数", response_json.get("count", len(chunks)))
    if not chunks:
        st.info("没有符合筛选条件的切片。")
    else:
        columns = [
            "file_name",
            "chunk_id",
            "position",
            "text_length",
            "text_preview",
            "source_path",
        ]
        st.dataframe(
            table_rows(chunks, columns),
            width="stretch",
            hide_index=True,
        )

    render_raw_response("/chunks-preview", response_json)


def render_knowledge_panel(backend_url: str) -> None:
    """提供文档注册表和知识库切片的按需浏览。"""
    docs_tab, chunks_tab = st.tabs(["文档", "切片"])

    with docs_tab:
        st.subheader("文档列表")
        st.caption("读取当前知识库的文档注册信息。")
        if st.button("加载文档", key="load_docs", type="primary"):
            with st.spinner("正在加载文档信息..."):
                result = get_json(backend_url, "/docs-preview", timeout=20)
            remember_api_result("docs_result", backend_url, result)
        render_docs_result(recall_api_result("docs_result", backend_url))

    with chunks_tab:
        st.subheader("切片预览")
        st.caption("可按文件名筛选，并限制返回数量和预览文字长度。")
        with st.form("chunks_form"):
            file_name = st.text_input(
                "文件名（可选，需完整匹配）",
                key="chunks_file_name",
            )
            col1, col2 = st.columns(2)
            max_len = col1.number_input(
                "预览长度",
                min_value=1,
                max_value=1000,
                value=300,
                step=50,
                key="chunks_max_len",
            )
            limit = col2.number_input(
                "返回数量",
                min_value=1,
                max_value=1000,
                value=70,
                step=10,
                key="chunks_limit",
            )
            submitted = st.form_submit_button("加载切片", type="primary")

        if submitted:
            params: dict[str, Any] = {
                "max_len": int(max_len),
                "limit": int(limit),
            }
            if file_name.strip():
                params["file_name"] = file_name.strip()
            with st.spinner("正在加载切片预览..."):
                result = get_json(
                    backend_url,
                    "/chunks-preview",
                    params=params,
                    timeout=20,
                )
            remember_api_result("chunks_result", backend_url, result)

        render_chunks_result(recall_api_result("chunks_result", backend_url))


def render_decide_result(result: ApiResult | None) -> None:
    """渲染 /decide 返回的 Agent 路由决策。"""
    response_json = response_from_result("/decide", result)
    if response_json is None:
        return

    data = response_json.get("data") or {}
    decision = data.get("decision")
    st.write("question:", pick(data, "question", ""))
    st.subheader("路由决策")
    if isinstance(decision, dict):
        columns = st.columns(3)
        columns[0].metric("decision", str(pick(decision, "decision", "")))
        columns[1].metric("action", str(pick(decision, "action", "")))
        columns[2].metric("source", str(pick(decision, "router_source", "")))
        st.write("reason:", pick(decision, "reason", ""))
        st.json(decision, expanded=False)
    else:
        st.write(decision or pick(data, "reason", ""))

    render_raw_response("/decide", response_json)


def render_metrics_result(result: ApiResult | None) -> None:
    """渲染 /metrics-summary 的近期请求统计。"""
    response_json = response_from_result("/metrics-summary", result)
    if response_json is None:
        return

    data = response_json.get("data") or {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("请求数", pick(data, "total", 0))
    col2.metric("平均耗时 (ms)", pick(data, "avg_latency_ms", 0))
    col3.metric("平均 LLM 调用", pick(data, "avg_llm_call_count", 0))
    col4.metric("成功数", pick(data, "success_count", 0))

    status_count = data.get("status_count") or {}
    if status_count:
        st.subheader("状态分布")
        rows = [
            {"status": status, "count": count} for status, count in status_count.items()
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("LLM 使用情况")
    llm_columns = st.columns(3)
    llm_columns[0].metric(
        "router",
        pick(data, "router_llm_used_count", 0),
    )
    llm_columns[1].metric(
        "decompose",
        pick(data, "decompose_llm_used_count", 0),
    )
    llm_columns[2].metric(
        "answer",
        pick(data, "answer_llm_used_count", 0),
    )

    render_raw_response("/metrics-summary", response_json)


def render_diagnostics_panel(backend_url: str, default_top_k: int) -> None:
    """提供 Agent 路由决策和运行指标诊断。"""
    decide_tab, metrics_tab = st.tabs(["路由决策", "运行指标"])

    with decide_tab:
        st.subheader("Agent 路由决策")
        st.caption("查看问题会被路由到知识库、工具、澄清还是直接回答。")
        with st.form("decide_form"):
            question = st.text_area(
                "待决策问题",
                placeholder="请输入要分析的问题...",
                key="decide_question",
            )
            decide_top_k = st.number_input(
                "路由 top_k",
                min_value=1,
                max_value=10,
                value=int(default_top_k),
                step=1,
                key="decide_top_k",
            )
            submitted = st.form_submit_button("分析路由", type="primary")

        if submitted:
            if not question.strip():
                st.warning("请输入待决策问题。")
            else:
                with st.spinner("正在分析路由..."):
                    result = post_json(
                        backend_url,
                        "/decide",
                        payload={
                            "question": question.strip(),
                            "top_k": int(decide_top_k),
                        },
                    )
                remember_api_result("decide_result", backend_url, result)

        render_decide_result(recall_api_result("decide_result", backend_url))

    with metrics_tab:
        st.subheader("近期运行指标")
        st.caption("汇总最近请求的状态、耗时和 LLM 调用情况。")
        with st.form("metrics_form"):
            limit = st.number_input(
                "统计最近请求数",
                min_value=1,
                max_value=200,
                value=20,
                step=10,
                key="metrics_limit",
            )
            submitted = st.form_submit_button("加载指标", type="primary")

        if submitted:
            with st.spinner("正在读取指标..."):
                result = get_json(
                    backend_url,
                    "/metrics-summary",
                    params={"limit": int(limit)},
                    timeout=20,
                )
            remember_api_result("metrics_result", backend_url, result)

        render_metrics_result(recall_api_result("metrics_result", backend_url))


def extract_assistant_content(
    response_json: dict[str, Any] | None,
    error: str | None,
) -> str:
    """从 /chat 响应中提取适合显示在聊天气泡里的助手文本。"""
    if error:
        return f"请求失败：{error}"

    if not response_json:
        return "后端没有返回可展示的数据。"

    data = response_json.get("data") or {}
    candidates = [
        pick(data, "answer", ""),
        pick(data, "clarification_question", ""),
        pick(response_json, "message", ""),
        pick(data, "reason", ""),
    ]

    for candidate in candidates:
        content = str(candidate).strip()
        if content:
            return content

    return "后端没有返回 answer。"


def build_chat_payload(
    question: str,
    top_k: int,
    decompose_top_k: int,
) -> dict[str, Any]:
    """构造同步和流式聊天共用的公开请求体。"""
    return {
        "question": question,
        "top_k": int(top_k),
        "decompose_top_k": int(decompose_top_k),
        "session_id": st.session_state["session_id"],
    }


def request_chat_response(
    backend_url: str,
    question: str,
    top_k: int,
    decompose_top_k: int,
) -> ApiResult:
    """调用 /chat 接口，并携带当前 Streamlit 会话中的 session_id。"""
    return post_json(
        backend_url,
        "/chat",
        payload=build_chat_payload(question, top_k, decompose_top_k),
    )


def request_chat_stream_response(
    backend_url: str,
    question: str,
    top_k: int,
    decompose_top_k: int,
    *,
    on_event: StreamEventHandler | None = None,
) -> ApiResult:
    """调用 /chat/stream，并返回 final 内的标准 ChatResponse。"""
    return request_chat_stream(
        backend_url,
        build_chat_payload(question, top_k, decompose_top_k),
        on_event=on_event,
    )


def render_message_detail(message: dict[str, Any]) -> None:
    """在助手消息下方折叠展示本轮响应的引用、trace 和原始 JSON。"""
    if message.get("role") != "assistant":
        return

    response_json = message.get("response_json")
    if not response_json:
        return

    render_chat_response_details(
        response_json,
        message.get("status_code"),
        endpoint=str(message.get("endpoint") or "/chat"),
    )


def render_chat_message(message: dict[str, Any]) -> None:
    """渲染单条聊天消息。"""
    role = message.get("role", "assistant")
    content = message.get("content", "")

    with st.chat_message(role):
        st.write(content)
        render_message_detail(message)


def render_chat_history() -> None:
    """渲染 st.session_state 中保存的聊天历史。"""
    messages = st.session_state.get("messages", [])
    if not messages:
        st.info(
            "开始一段对话吧。当前页面会保留 session_id，让后端使用同一段上下文回答。"
        )
        return

    for message in messages:
        render_chat_message(message)


def queue_chat_prompt() -> None:
    """在输入框提交回调中暂存问题，供下一次页面执行优先处理。"""
    prompt = st.session_state.get("chat_prompt")
    if prompt is not None:
        st.session_state["pending_chat_prompt"] = str(prompt)


def handle_chat_prompt(
    backend_url: str,
    prompt: str,
    top_k: int,
    decompose_top_k: int,
    *,
    streaming: bool = True,
) -> None:
    """处理一次聊天输入，更新前端消息并展示助手响应。"""
    question = prompt.strip()
    if not question:
        st.warning("请输入 question 后再发起请求。")
        return

    user_message = {"role": "user", "content": question}
    st.session_state["messages"].append(user_message)
    render_chat_message(user_message)

    with st.chat_message("assistant"):
        endpoint = "/chat/stream" if streaming else "/chat"
        if streaming:
            status_panel = st.status("正在建立聊天流...", expanded=True)
            answer_placeholder = st.empty()
            streamed_answer_parts: list[str] = []

            def render_stream_event(
                event: str,
                envelope: dict[str, Any],
            ) -> None:
                """只使用 SSE 公开合同字段增量更新当前助手气泡。"""
                data = envelope.get("data")
                data = data if isinstance(data, dict) else {}
                if event == "start":
                    status_panel.update(label="聊天请求已开始", state="running")
                elif event == "progress":
                    stage = str(data.get("stage") or "")
                    stage_label = PROGRESS_STAGE_LABELS.get(stage, "处理请求")
                    progress_status = str(data.get("status") or "")
                    if progress_status == "started":
                        status_panel.update(
                            label=f"正在{stage_label}...",
                            state="running",
                        )
                    elif progress_status == "failed":
                        status_panel.write(f"{stage_label}：失败")
                    else:
                        status_panel.write(
                            f"{stage_label}：{progress_status or '已更新'}"
                        )
                elif event == "tool":
                    tool_name = str(data.get("tool_name") or "未知工具")
                    phase = str(data.get("phase") or "")
                    tool_status = str(data.get("status") or "")
                    if phase == "started":
                        status_panel.update(
                            label=f"正在调用工具 {tool_name}...",
                            state="running",
                        )
                    else:
                        cost_ms = data.get("cost_ms", "-")
                        status_panel.write(
                            f"工具 {tool_name}：{tool_status or 'completed'} "
                            f"({cost_ms} ms)"
                        )
                elif event == "answer_delta":
                    text = data.get("text")
                    if isinstance(text, str) and text:
                        streamed_answer_parts.append(text)
                        answer_placeholder.markdown(
                            "".join(streamed_answer_parts)
                        )
                elif event == "final":
                    status_panel.update(
                        label="已收到最终结果",
                        state="running",
                    )
                elif event == "error":
                    status_panel.update(
                        label="聊天流处理失败",
                        state="error",
                    )

            response_json, status_code, error = request_chat_stream_response(
                backend_url,
                question,
                top_k,
                decompose_top_k,
                on_event=render_stream_event,
            )
            content = extract_assistant_content(response_json, error)
            answer_placeholder.markdown(content)
            if error:
                status_panel.update(
                    label="本轮请求未正常完成",
                    state="error",
                    expanded=True,
                )
            else:
                public_status = str(
                    ((response_json or {}).get("data") or {}).get("status")
                    or "completed"
                )
                status_panel.update(
                    label=f"处理完成（{public_status}）",
                    state="complete",
                    expanded=False,
                )
        else:
            with st.spinner("正在生成回答..."):
                response_json, status_code, error = request_chat_response(
                    backend_url,
                    question,
                    top_k,
                    decompose_top_k,
                )
            content = extract_assistant_content(response_json, error)
            st.write(content)

        assistant_message = {
            "role": "assistant",
            "content": content,
            "response_json": response_json,
            "status_code": status_code,
            "error": error,
            "endpoint": endpoint,
        }
        render_message_detail(assistant_message)
        st.session_state["messages"].append(assistant_message)


def main() -> None:
    """组装 Streamlit RAG 演示与诊断页面。"""
    st.set_page_config(page_title="RAG Demo", layout="wide")
    init_chat_state()

    st.title("FastAPI RAG Demo")

    backend_url = normalize_backend_url(
        st.sidebar.text_input("后端地址", value=DEFAULT_BACKEND_URL)
    )

    button_col1, button_col2 = st.sidebar.columns(2)
    if button_col1.button("新对话", width="stretch"):
        reset_chat_state()
    if button_col2.button("刷新状态", width="stretch"):
        request_health.clear()
        request_kb_summary.clear()

    st.sidebar.caption(f"session_id: `{st.session_state['session_id']}`")

    st.sidebar.subheader("请求参数")
    chat_mode = st.sidebar.radio(
        "聊天模式",
        options=("流式", "同步"),
        index=0,
        key="chat_mode",
    )
    st.sidebar.caption(
        "流式模式展示安全进度和答案增量；最终结果始终以 final 为准。"
    )
    top_k = st.sidebar.number_input("top_k", min_value=1, max_value=10, value=5, step=1)
    decompose_top_k = st.sidebar.number_input(
        "decompose_top_k",
        min_value=1,
        max_value=10,
        value=10,
        step=1,
    )

    render_health_status(backend_url)
    render_kb_summary(backend_url)

    chat_tab, search_tab, knowledge_tab, diagnostics_tab = st.tabs(
        ["聊天", "检索调试", "知识库", "诊断"]
    )

    with chat_tab:
        render_chat_history()

        pending_prompt = st.session_state.pop("pending_chat_prompt", None)
        if pending_prompt is not None:
            handle_chat_prompt(
                backend_url,
                pending_prompt,
                top_k,
                decompose_top_k,
                streaming=chat_mode == "流式",
            )
            if pending_prompt.strip():
                st.rerun()

        st.chat_input(
            "请输入要询问知识库的问题...",
            key="chat_prompt",
            on_submit=queue_chat_prompt,
        )

    with search_tab:
        render_search_panel(backend_url, int(top_k))

    with knowledge_tab:
        render_knowledge_panel(backend_url)

    with diagnostics_tab:
        render_diagnostics_panel(backend_url, int(top_k))


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120


def normalize_backend_url(raw_url: str) -> str:
    """规范化后端地址，去掉首尾空白和末尾斜杠，空值时使用默认地址。"""
    url = (raw_url or DEFAULT_BACKEND_URL).strip().rstrip("/")
    return url or DEFAULT_BACKEND_URL


def post_json(
    backend_url: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """向指定后端接口发送 JSON POST 请求，并统一返回响应数据、状态码和错误信息。"""
    url = f"{normalize_backend_url(backend_url)}{endpoint}"

    try:
        response = requests.post(url, json=payload or {}, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return (
            None,
            None,
            f"无法连接后端：{url}。请先启动 FastAPI 服务：uvicorn app.main:app --reload",
        )
    except requests.exceptions.Timeout:
        return None, None, f"请求超时：{url}。后端可能仍在加载模型或处理请求，请稍后重试。"
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


def render_http_warning(endpoint: str, status_code: int | None) -> None:
    """当接口返回 4xx/5xx 状态码时，在页面上提示 HTTP 警告。"""
    if status_code is not None and status_code >= 400:
        st.warning(f"{endpoint} 返回 HTTP {status_code}。下方仍会展示后端返回的结构化响应。")


def pick(data: dict[str, Any], key: str, default: Any = "") -> Any:
    """从字典中安全读取字段，缺失或值为 None 时返回默认值。"""
    value = data.get(key, default)
    return default if value is None else value


def table_rows(items: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    """按指定列名整理列表数据，生成适合 st.dataframe 展示的行数据。"""
    return [{column: item.get(column, "") for column in columns} for item in items]


def render_references(references: list[dict[str, Any]]) -> None:
    """渲染回答引用的知识库片段，包含来源文件、分块编号和片段内容。"""
    st.subheader("references")

    if not references:
        st.info("本次响应没有 references。")
        return

    for idx, reference in enumerate(references, start=1):
        reference_id = pick(reference, "reference_id", f"Reference {idx}")
        file_name = pick(reference, "file_name", "")
        chunk_id = pick(reference, "chunk_id", "")
        label = f"{reference_id} | {file_name} | chunk_id={chunk_id}"

        with st.expander(label, expanded=False):
            st.write("reference_id:", reference_id)
            st.write("file_name:", file_name)
            st.write("chunk_id:", chunk_id)
            st.write("score:", pick(reference, "score", ""))
            st.write("rerank_score:", pick(reference, "rerank_score", ""))
            st.write("source_path:", pick(reference, "source_path", ""))
            st.markdown("**snippet**")
            st.write(pick(reference, "snippet", ""))


def render_tool_trace(tool_trace: list[dict[str, Any]]) -> None:
    """渲染工具调用链路，展示每一步的动作、状态和完整 JSON。"""
    st.subheader("tool_trace")

    if not tool_trace:
        st.info("本次响应没有 tool_trace。")
        return

    columns = [
        "step",
        "action",
        "tool_name",
        "tool_status",
        "final_status",
        "router_source",
        "reason",
    ]
    st.dataframe(table_rows(tool_trace, columns), use_container_width=True, hide_index=True)

    with st.expander("tool_trace 完整 JSON", expanded=False):
        st.json(tool_trace)


def render_chat_result(response_json: dict[str, Any], status_code: int | None) -> None:
    """渲染 /chat 接口的问答结果，包括答案、指标、引用和工具链路。"""
    render_http_warning("/chat", status_code)

    data = response_json.get("data") or {}
    trace_id = pick(response_json, "trace_id", "")

    st.subheader("answer")
    answer = pick(data, "answer", "")
    if answer:
        st.write(answer)
    else:
        st.info("后端响应中没有 answer。")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("status", str(pick(data, "status", "")))
    col2.metric("confidence", str(pick(data, "confidence", "")))
    col3.metric("action", str(pick(data, "action", "")))
    col4.metric("trace_id", str(trace_id))

    st.write("reason:", pick(data, "reason", ""))

    render_references(data.get("references") or [])
    render_tool_trace(data.get("tool_trace") or [])

    with st.expander("完整响应 JSON", expanded=False):
        st.json(response_json)


def render_search_result(response_json: dict[str, Any], status_code: int | None) -> None:
    """渲染 /search 接口的检索结果，包括匹配列表、分数和文本片段。"""
    render_http_warning("/search", status_code)

    data = response_json.get("data") or {}
    status = pick(data, "status", "")
    reason = pick(data, "reason", "")
    st.write("status:", status)
    st.write("reason:", reason)

    results = data.get("results") or []
    if not results:
        st.info("本次检索没有返回结果。")
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
        st.dataframe(table_rows(results, columns), use_container_width=True, hide_index=True)

        st.subheader("snippet")
        for idx, item in enumerate(results, start=1):
            reference_id = pick(item, "reference_id", f"Reference {idx}")
            file_name = pick(item, "file_name", "")
            with st.expander(f"{reference_id} | {file_name}", expanded=False):
                st.write(pick(item, "snippet", ""))
                st.caption(str(pick(item, "source_path", "")))

    with st.expander("完整响应 JSON", expanded=False):
        st.json(response_json)


def render_decide_result(response_json: dict[str, Any], status_code: int | None) -> None:
    """渲染 /decide 接口的路由判断结果，直接展示完整 JSON。"""
    render_http_warning("/decide", status_code)
    st.json(response_json)


def render_kb_summary(backend_url: str) -> None:
    """请求并渲染知识库状态信息，展示在 Streamlit 侧边栏中。"""
    st.sidebar.subheader("知识库状态")

    response_json, status_code, error = post_json(
        backend_url,
        "/kb_summary",
        payload={},
        timeout=20,
    )

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
    file_names = data.get("file_names") or [doc.get("file_name", "") for doc in documents]

    with st.sidebar.expander(f"文档列表 ({len(file_names)})", expanded=True):
        if not file_names:
            st.info("暂无文档。")
        else:
            for file_name in file_names:
                st.write(file_name)

    with st.sidebar.expander("kb_summary 完整 JSON", expanded=False):
        st.json(response_json)


def run_action(
    endpoint: str,
    backend_url: str,
    question: str,
    top_k: int,
    decompose_top_k: int,
) -> None:
    """校验用户问题，调用指定接口，并将最新请求结果保存到会话状态。"""
    question = question.strip()
    if not question:
        st.warning("请输入 question 后再发起请求。")
        return

    payload = {
        "question": question,
        "top_k": int(top_k),
        "decompose_top_k": int(decompose_top_k),
    }

    with st.spinner(f"正在请求 {endpoint} ..."):
        response_json, status_code, error = post_json(backend_url, endpoint, payload=payload)

    st.session_state["last_result"] = {
        "endpoint": endpoint,
        "response_json": response_json,
        "status_code": status_code,
        "error": error,
    }


def render_last_result() -> None:
    """读取会话中的最近一次请求结果，并根据接口类型选择对应渲染函数。"""
    result = st.session_state.get("last_result")
    if not result:
        return

    endpoint = result["endpoint"]
    response_json = result["response_json"]
    status_code = result["status_code"]
    error = result["error"]

    st.divider()
    st.header("结果")

    if error:
        st.error(error)
        return

    if not response_json:
        st.error("后端没有返回可展示的数据。")
        return

    if endpoint == "/chat":
        render_chat_result(response_json, status_code)
    elif endpoint == "/search":
        render_search_result(response_json, status_code)
    elif endpoint == "/decide":
        render_decide_result(response_json, status_code)


def main() -> None:
    """组装 Streamlit Demo 页面，绑定侧边栏、输入区、操作按钮和结果展示。"""
    st.set_page_config(page_title="RAG Demo", layout="wide")

    st.title("FastAPI RAG Demo")

    backend_url = normalize_backend_url(
        st.sidebar.text_input("后端地址", value=DEFAULT_BACKEND_URL)
    )
    render_kb_summary(backend_url)

    question = st.text_area(
        "question",
        height=120,
        placeholder="请输入要询问知识库的问题...",
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.number_input("top_k", min_value=1, max_value=10, value=5, step=1)
    with col2:
        decompose_top_k = st.number_input(
            "decompose_top_k",
            min_value=1,
            max_value=10,
            value=10,
            step=1,
        )

    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        if st.button("开始问答", use_container_width=True):
            run_action("/chat", backend_url, question, top_k, decompose_top_k)
    with btn2:
        if st.button("只看检索", use_container_width=True):
            run_action("/search", backend_url, question, top_k, decompose_top_k)
    with btn3:
        if st.button("只看路由", use_container_width=True):
            run_action("/decide", backend_url, question, top_k, decompose_top_k)

    render_last_result()


if __name__ == "__main__":
    main()

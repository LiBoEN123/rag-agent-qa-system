from __future__ import annotations

import uuid
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120


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


def render_chat_response_details(
    response_json: dict[str, Any],
    status_code: int | None,
) -> None:
    """渲染 /chat 接口的指标、引用、工具链路和完整 JSON。"""
    with st.expander("本轮详情", expanded=False):
        render_http_warning("/chat", status_code)

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
                use_container_width=True,
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
                use_container_width=True,
                hide_index=True,
            )
            st.json(tool_trace, expanded=False)

        st.subheader("完整响应 JSON")
        st.json(response_json, expanded=False)


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


def request_chat_response(
    backend_url: str,
    question: str,
    top_k: int,
    decompose_top_k: int,
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """调用 /chat 接口，并携带当前 Streamlit 会话中的 session_id。"""
    payload = {
        "question": question,
        "top_k": int(top_k),
        "decompose_top_k": int(decompose_top_k),
        "session_id": st.session_state["session_id"],
    }

    return post_json(backend_url, "/chat", payload=payload)


def render_message_detail(message: dict[str, Any]) -> None:
    """在助手消息下方折叠展示本轮响应的引用、trace 和原始 JSON。"""
    if message.get("role") != "assistant":
        return

    response_json = message.get("response_json")
    if not response_json:
        return

    render_chat_response_details(response_json, message.get("status_code"))


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
        st.info("开始一段对话吧。当前页面会保留 session_id，让后端使用同一段上下文回答。")
        return

    for message in messages:
        render_chat_message(message)


def handle_chat_prompt(
    backend_url: str,
    prompt: str,
    top_k: int,
    decompose_top_k: int,
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
        }
        render_message_detail(assistant_message)
        st.session_state["messages"].append(assistant_message)


def main() -> None:
    """组装 Streamlit 聊天 Demo 页面，绑定侧边栏、聊天历史和输入框。"""
    st.set_page_config(page_title="RAG Demo", layout="wide")
    init_chat_state()

    st.title("FastAPI RAG Chat")

    backend_url = normalize_backend_url(
        st.sidebar.text_input("后端地址", value=DEFAULT_BACKEND_URL)
    )

    if st.sidebar.button("新对话", use_container_width=True):
        reset_chat_state()

    st.sidebar.caption(f"session_id: `{st.session_state['session_id']}`")

    st.sidebar.subheader("请求参数")
    top_k = st.sidebar.number_input("top_k", min_value=1, max_value=10, value=5, step=1)
    decompose_top_k = st.sidebar.number_input(
        "decompose_top_k",
        min_value=1,
        max_value=10,
        value=10,
        step=1,
    )

    render_kb_summary(backend_url)
    render_chat_history()

    prompt = st.chat_input("请输入要询问知识库的问题...")
    if prompt is not None:
        handle_chat_prompt(backend_url, prompt, top_k, decompose_top_k)


if __name__ == "__main__":
    main()

"""聊天 SSE 接口的手动冒烟客户端。

用途：向已启动的 ``/chat/stream`` 发起一次 POST，并检查事件顺序与 final。
输入：服务地址、问题、可选 session ID、召回参数和超时。
输出：事件类型、sequence 与不含正文/凭据/工具原始结果的安全摘要。
边界：只供人工运行；测试和 CI 不会调用真实服务、模型、RAG 或网络。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from typing import Any

import httpx


PUBLIC_EVENTS = frozenset(
    {"start", "progress", "tool", "answer_delta", "final", "error", "done"}
)


def parse_args() -> argparse.Namespace:
    """解析聊天流手动冒烟参数。"""
    parser = argparse.ArgumentParser(
        description="检查 POST /chat/stream 的 SSE 事件合同",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI 服务根地址",
    )
    parser.add_argument("--question", required=True, help="测试问题")
    parser.add_argument("--session-id", default="", help="可选会话 ID")
    parser.add_argument("--top-k", type=int, default=None, help="可选召回条数")
    parser.add_argument(
        "--decompose-top-k",
        type=int,
        default=None,
        help="可选问题拆分召回条数",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="请求超时秒数",
    )
    return parser.parse_args()


def iter_sse_events(lines: Iterator[str]) -> Iterator[dict[str, str]]:
    """把 HTTP 行流解析为最小 SSE event/id/data 字典。"""
    fields: dict[str, str] = {}
    for line in lines:
        if not line:
            if fields:
                yield fields
                fields = {}
            continue
        if line.startswith(":") or ":" not in line:
            continue
        name, value = line.split(":", 1)
        if name in {"event", "id", "data"}:
            fields[name] = value.lstrip()
    if fields:
        yield fields


def safe_event_summary(event: str, envelope: dict[str, Any]) -> str:
    """只展示合同字段统计，不打印答案、页面、PDF 或 chunk 正文。"""
    data = envelope.get("data")
    data = data if isinstance(data, dict) else {}
    if event == "start":
        return f"session_id_present={bool(data.get('session_id_present'))}"
    if event == "progress":
        return f"stage={data.get('stage', '')} status={data.get('status', '')}"
    if event == "tool":
        return (
            f"tool_name={data.get('tool_name', '')} "
            f"phase={data.get('phase', '')} status={data.get('status', '')} "
            f"cost_ms={data.get('cost_ms', '-')}"
        )
    if event == "answer_delta":
        return f"text_chars={len(str(data.get('text') or ''))}"
    if event == "final":
        response = data.get("response")
        response = response if isinstance(response, dict) else {}
        response_data = response.get("data")
        response_data = response_data if isinstance(response_data, dict) else {}
        references = response_data.get("references")
        reference_count = len(references) if isinstance(references, list) else 0
        return (
            f"http_status={data.get('http_status', '')} "
            f"code={response.get('code', '')} "
            f"status={response_data.get('status', '')} "
            f"answer_chars={len(str(response_data.get('answer') or ''))} "
            f"reference_count={reference_count} "
            f"total_cost_ms={data.get('total_cost_ms', '')}"
        )
    if event == "error":
        return f"code={data.get('code', '')}"
    if event == "done":
        return f"reason={data.get('reason', '')}"
    return "unsupported_event"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """只发送 ChatRequest 的公开字段。"""
    payload: dict[str, Any] = {"question": args.question}
    if args.session_id:
        payload["session_id"] = args.session_id
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.decompose_top_k is not None:
        payload["decompose_top_k"] = args.decompose_top_k
    return payload


def main() -> int:
    """执行请求，验证 sequence、trace、final 和 done 合同。"""
    args = parse_args()
    url = f"{args.base_url.rstrip('/')}/chat/stream"
    expected_sequence = 1
    trace_id = ""
    final_seen = False
    done_seen = False

    with httpx.stream(
        "POST",
        url,
        json=build_payload(args),
        timeout=args.timeout,
        headers={"Accept": "text/event-stream"},
    ) as response:
        if response.status_code != 200:
            print(
                f"HTTP {response.status_code} "
                f"content_type={response.headers.get('content-type', '')}"
            )
            return 1
        if not response.headers.get("content-type", "").startswith(
            "text/event-stream"
        ):
            raise RuntimeError("response is not text/event-stream")

        for fields in iter_sse_events(response.iter_lines()):
            event = fields.get("event", "")
            if event not in PUBLIC_EVENTS:
                raise RuntimeError(f"unsupported SSE event: {event!r}")
            try:
                envelope = json.loads(fields.get("data", ""))
            except json.JSONDecodeError as exc:
                raise RuntimeError("SSE data is not valid JSON") from exc
            if not isinstance(envelope, dict):
                raise RuntimeError("SSE envelope must be an object")
            if envelope.get("schema_version") != 1:
                raise RuntimeError("unexpected SSE schema_version")
            if not envelope.get("timestamp"):
                raise RuntimeError("SSE timestamp is missing")
            sequence = envelope.get("sequence")
            if sequence != expected_sequence:
                raise RuntimeError(
                    f"unexpected sequence: expected {expected_sequence}, got {sequence}"
                )
            if fields.get("id") != str(sequence):
                raise RuntimeError("SSE id does not match envelope sequence")
            current_trace = str(envelope.get("trace_id") or "")
            if not current_trace or (trace_id and current_trace != trace_id):
                raise RuntimeError("SSE trace_id is missing or changed")
            trace_id = current_trace
            print(
                f"event={event} sequence={sequence} "
                f"{safe_event_summary(event, envelope)}"
            )
            expected_sequence += 1
            final_seen = final_seen or event == "final"
            if event == "done" and not final_seen:
                raise RuntimeError("done arrived before final")
            done_seen = done_seen or event == "done"

    if not final_seen:
        raise RuntimeError("stream ended without final")
    if not done_seen:
        raise RuntimeError("stream ended without done")
    print(f"stream_ok events={expected_sequence - 1} trace_id_present=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline tests for the unified chat API benchmark CLI."""

import argparse
import asyncio
import json

import httpx
import pytest

from scripts import benchmark_chat_api as benchmark


def _case(case_id: str = "case-1") -> dict[str, str]:
    return {"id": case_id, "question": "测试问题"}


def _sse(*events):
    return "".join(
        f"event: {name}\r\ndata: {json.dumps({'data': data})}\r\n\r\n"
        for name, data in events
    ).encode()


def _final(code=0, answer="回答", http_status=200):
    return (
        "final",
        {
            "http_status": http_status,
            "response": {
                "code": code,
                "data": {
                    "status": "ok" if code == 0 else "error",
                    "answer": answer,
                    "references": [{"chunk_id": 1}],
                },
            },
        },
    )


def _response(*events):
    return httpx.Response(
        200, content=_sse(*events), headers={"Content-Type": "text/event-stream"}
    )


START = ("start", {"session_id_present": False})
DONE = ("done", {"reason": "completed"})
ERROR = ("error", {"code": "STREAM_ERROR", "message": "流式处理失败"})
ERROR_DONE = ("done", {"reason": "error"})


def test_parse_concurrency_supports_serial_multiple_and_deduplication():
    assert benchmark.parse_concurrency("1") == [1]
    assert benchmark.parse_concurrency("1,2,3,5,2") == [1, 2, 3, 5]
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.parse_concurrency("1,0")


def test_percentile_and_empty_summary_are_deterministic():
    assert benchmark.percentile([10, 20, 30, 40], 50) == 25
    assert benchmark.percentile([10, 20, 30, 40], 95) == 38.5
    summary = benchmark.summarize_rows([], 0)
    assert summary["request_count"] == 0
    assert summary["success_rate"] == 0
    assert summary["throughput_rps"] == 0
    assert summary["p99_latency_ms"] == 0


def test_send_request_records_final_and_stops_timing_at_done(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])

    class DelayedDone(httpx.AsyncByteStream):
        async def __aiter__(self):
            body = b": keep-alive\r\n\r\n" + _sse(
                START, ("answer_delta", {"text": "临时文本"}), _final()
            )
            # Exercise transport fragmentation and UTF-8 decoding in the library.
            for offset in range(0, len(body), 7):
                yield body[offset : offset + 7]
            clock[0] += 0.25
            yield _sse(DONE)
            raise AssertionError("benchmark must stop reading at done")

        async def aclose(self):
            clock[0] += 10

    async def scenario():
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["X-Trace-Id"].startswith("benchmark-")
            assert request.url.path == "/chat/stream"
            assert request.headers["Accept"] == "text/event-stream"
            return httpx.Response(
                200, stream=DelayedDone(), headers={"Content-Type": "text/event-stream"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await benchmark.send_request(
                client,
                chat_url="http://test/chat/stream",
                case=_case(),
                concurrency=1,
                round_index=2,
                top_k=5,
            )

    row = asyncio.run(scenario())

    assert row["success"] is True
    assert row["business_code"] == 0
    assert row["status"] == "ok"
    assert row["latency_ms"] == 250
    assert row["answer_length"] == 2
    assert row["reference_count"] == 1
    assert set(row) == set(benchmark.REQUEST_FIELDS)


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (httpx.Response(503, json={"code": "DOWN"}), "http_error"),
        (_response(START, _final("ERROR"), DONE), "business_error"),
        (
            _response(
                START, _final("EMPTY_QUESTION", answer="", http_status=400), DONE
            ),
            "business_error",
        ),
        (_response(START, _final(answer=""), DONE), "empty_answer"),
        (httpx.Response(200, content=b"not-sse"), "invalid_stream"),
        (
            httpx.Response(
                200,
                content=b"event: final\ndata: not-json\n\n",
                headers={"Content-Type": "text/event-stream"},
            ),
            "invalid_stream",
        ),
        (_response(START, _final()), "invalid_stream"),
        (_response(START, DONE), "invalid_stream"),
        (_response(START, _final(), ERROR_DONE), "invalid_stream"),
        (_response(START, ("final", {"response": {}}), DONE), "invalid_stream"),
        (_response(START, ERROR, ERROR_DONE), "stream_error"),
        (_response(START, _final(), ERROR, ERROR_DONE), "stream_error"),
    ],
)
def test_send_request_normalizes_http_and_payload_errors(response, error_type):
    calls = []

    def handler(request):
        calls.append(request)
        return response

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await benchmark.send_request(
                client,
                chat_url="http://test/chat/stream",
                case=_case(),
                concurrency=2,
                round_index=1,
                top_k=5,
            )

    row = asyncio.run(scenario())
    assert row["success"] is False
    assert row["error_type"] == error_type
    assert len(calls) == 1


def test_send_request_normalizes_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await benchmark.send_request(
                client,
                chat_url="http://test/chat/stream",
                case=_case(),
                concurrency=1,
                round_index=1,
                top_k=5,
            )

    row = asyncio.run(scenario())
    assert row["error_type"] == "timeout"
    assert row["http_status"] is None


def test_concurrency_one_and_multiple_levels_are_bounded(monkeypatch):
    active = 0
    observed = 0

    async def fake_send_request(client, **kwargs):
        nonlocal active, observed
        del client
        active += 1
        observed = max(observed, active)
        await asyncio.sleep(0)
        active -= 1
        return {
            field: {
                "concurrency": kwargs["concurrency"],
                "round_index": kwargs["round_index"],
                "case_id": kwargs["case"]["id"],
                "question": kwargs["case"]["question"],
                "success": True,
            }.get(field, "")
            for field in benchmark.REQUEST_FIELDS
        }

    monkeypatch.setattr(benchmark, "send_request", fake_send_request)

    async def run(level):
        return await benchmark.run_concurrency_level(
            object(),
            chat_url="http://test/chat/stream",
            cases=[_case("a"), _case("b")],
            concurrency=level,
            repeats=2,
            top_k=5,
        )

    serial_rows, _ = asyncio.run(run(1))
    assert len(serial_rows) == 4
    assert observed == 1

    observed = 0
    parallel_rows, _ = asyncio.run(run(3))
    assert len(parallel_rows) == 4
    assert observed == 3


def test_summary_and_output_files_have_one_stable_shape(tmp_path):
    rows = [
        {
            "trace_id": "t1",
            "concurrency": 1,
            "round_index": 1,
            "case_id": "a",
            "question": "q",
            "latency_ms": 100.0,
            "http_status": 200,
            "business_code": 0,
            "status": "OK",
            "success": True,
            "answer_length": 10,
            "reference_count": 2,
            "error_type": "",
            "error_message": "",
            "started_at": "start",
            "finished_at": "finish",
        },
        {
            "trace_id": "t2",
            "concurrency": 1,
            "round_index": 1,
            "case_id": "b",
            "question": "q",
            "latency_ms": 300.0,
            "http_status": 500,
            "business_code": "ERROR",
            "status": "INTERNAL_ERROR",
            "success": False,
            "answer_length": 0,
            "reference_count": 0,
            "error_type": "http_error",
            "error_message": "HTTP 500",
            "started_at": "start",
            "finished_at": "finish",
        },
    ]
    summary = benchmark.summarize_rows(rows, 2)
    summary["by_concurrency"] = {"1": dict(summary)}

    benchmark.write_outputs(tmp_path, rows, summary)

    assert summary["success_rate"] == 0.5
    assert summary["throughput_rps"] == 1
    assert summary["mean_latency_ms"] == 200
    assert summary["http_status_distribution"] == {"200": 1, "500": 1}
    assert {path.name for path in tmp_path.iterdir()} == {
        "requests.jsonl",
        "summary.json",
        "summary.md",
    }
    first_line = (tmp_path / "requests.jsonl").read_text().splitlines()[0]
    written = json.loads(first_line)
    assert set(written) == set(benchmark.REQUEST_FIELDS)


def test_load_cases_accepts_project_schema_and_rejects_duplicates(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps({"cases": [{"id": "a", "question": "问题"}]}),
        encoding="utf-8",
    )
    assert benchmark.load_cases(valid) == [{"id": "a", "question": "问题"}]

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            {"cases": [{"id": "a", "question": "q1"}, {"id": "a", "question": "q2"}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        benchmark.load_cases(duplicate)

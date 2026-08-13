"""Offline tests for the unified chat API benchmark CLI."""

import argparse
import asyncio
import json

import httpx
import pytest

from scripts import benchmark_chat_api as benchmark


def _case(case_id: str = "case-1") -> dict[str, str]:
    return {"id": case_id, "question": "测试问题"}


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


def test_send_request_records_success_without_real_network():
    async def scenario():
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["X-Trace-Id"].startswith("benchmark-")
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "status": "OK",
                        "answer": "回答",
                        "references": [{"chunk_id": 1}],
                    },
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await benchmark.send_request(
                client,
                chat_url="http://test/chat",
                case=_case(),
                concurrency=1,
                round_index=2,
                top_k=5,
            )

    row = asyncio.run(scenario())

    assert row["success"] is True
    assert row["business_code"] == 0
    assert row["status"] == "OK"
    assert row["answer_length"] == 2
    assert row["reference_count"] == 1
    assert set(row) == set(benchmark.REQUEST_FIELDS)


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (httpx.Response(503, json={"code": "DOWN"}), "http_error"),
        (httpx.Response(200, json={"code": "ERROR"}), "business_error"),
        (httpx.Response(200, content=b"not-json"), "invalid_json"),
    ],
)
def test_send_request_normalizes_http_and_payload_errors(response, error_type):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: response)
        ) as client:
            return await benchmark.send_request(
                client,
                chat_url="http://test/chat",
                case=_case(),
                concurrency=2,
                round_index=1,
                top_k=5,
            )

    row = asyncio.run(scenario())
    assert row["success"] is False
    assert row["error_type"] == error_type


def test_send_request_normalizes_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await benchmark.send_request(
                client,
                chat_url="http://test/chat",
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
            chat_url="http://test/chat",
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

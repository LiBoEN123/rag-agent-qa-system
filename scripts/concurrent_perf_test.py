import asyncio
import json
import math
import time
from datetime import datetime
from pathlib import Path

import httpx


CHAT_URL = "http://127.0.0.1:8000/chat"

OUTPUT_DIR = Path("outputs/perf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "concurrent_Ollama_perf_results.jsonl"

# 建议保留 1 作为基线，然后再测 2、3、5 并发
CONCURRENCY_LEVELS = [1, 2, 3, 5]

# 根据你的系统实际情况调大一点
TIMEOUT_SECONDS = 120

# 尽量使用你知识库里能稳定回答的问题
QUESTIONS = [
    "每个 Vue 应用都是通过什么函数创建应用实例的？",
    "Vue 模板语法中如何绑定属性？",
    "组件的 props 有什么作用？",
    "Vue 中 v-if 和 v-show 有什么区别？",
    "Vue 应用实例可以挂载到哪里？",
]


def now_str() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def percentile(values, p: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


async def send_one_request(
    client: httpx.AsyncClient,
    concurrency: int,
    request_index: int,
    question: str,
):
    trace_id = f"perf_c{concurrency}_r{request_index}_{int(time.time() * 1000)}"

    payload = {
        "question": question,
        "top_k": 5,
    }

    client_start_wall = now_str()
    client_start = time.perf_counter()

    try:
        response = await client.post(
            CHAT_URL,
            json=payload,
            headers={"X-Trace-Id": trace_id},
        )

        client_cost = time.perf_counter() - client_start
        client_end_wall = now_str()

        try:
            data = response.json()
        except Exception:
            data = {}

        answer = ""
        references = []

        if isinstance(data, dict):
            data_part = data.get("data") or {}
            if isinstance(data_part, dict):
                answer = data_part.get("answer") or ""
                references = data_part.get("references") or []

        success = (
            response.status_code == 200
            and isinstance(data, dict)
            and str(data.get("code")) in ["0", "success"]
            and len(answer) > 0
        )

        return {
            "trace_id": trace_id,
            "concurrency": concurrency,
            "request_index": request_index,
            "question": question,
            "client_start": client_start_wall,
            "client_end": client_end_wall,
            "latency": round(client_cost, 3),
            "http_status": response.status_code,
            "success": success,
            "answer_len": len(answer),
            "references_count": len(references),
            "error": "",
        }

    except Exception as e:
        client_cost = time.perf_counter() - client_start
        client_end_wall = now_str()

        return {
            "trace_id": trace_id,
            "concurrency": concurrency,
            "request_index": request_index,
            "question": question,
            "client_start": client_start_wall,
            "client_end": client_end_wall,
            "latency": round(client_cost, 3),
            "http_status": None,
            "success": False,
            "answer_len": 0,
            "references_count": 0,
            "error": repr(e),
        }


async def run_one_concurrency_level(concurrency: int):
    print(f"\n[perf] 开始测试并发数: {concurrency}")

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(
        timeout=TIMEOUT_SECONDS,
        limits=limits,
    ) as client:
        tasks = []

        for i in range(concurrency):
            question = QUESTIONS[i % len(QUESTIONS)]
            tasks.append(
                send_one_request(
                    client=client,
                    concurrency=concurrency,
                    request_index=i + 1,
                    question=question,
                )
            )

        batch_start = time.perf_counter()

        # 关键：这里才是真正并发
        results = await asyncio.gather(*tasks)

        batch_cost = time.perf_counter() - batch_start

    latencies = [item["latency"] for item in results]
    success_count = sum(1 for item in results if item["success"])

    print(f"[perf] 并发数: {concurrency}")
    print(f"[perf] 批次总耗时: {batch_cost:.3f}s")
    print(f"[perf] 成功数: {success_count}/{concurrency}")
    print(f"[perf] 平均延迟: {sum(latencies) / len(latencies):.3f}s")
    print(f"[perf] P50 延迟: {percentile(latencies, 50):.3f}s")
    print(f"[perf] P95 延迟: {percentile(latencies, 95):.3f}s")
    print(f"[perf] 最大延迟: {max(latencies):.3f}s")

    for item in results:
        print(
            f"  请求 {item['request_index']} | "
            f"trace_id={item['trace_id']} | "
            f"latency={item['latency']}s | "
            f"success={item['success']} | "
            f"answer_len={item['answer_len']} | "
            f"refs={item['references_count']} | "
            f"error={item['error']}"
        )

    with RESULT_FILE.open("a", encoding="utf-8") as f:
        for item in results:
            item["batch_total_cost"] = round(batch_cost, 3)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return results


async def warm_up():
    print("[perf] warm-up 请求开始，不计入最终结果")

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        result = await send_one_request(
            client=client,
            concurrency=1,
            request_index=0,
            question=QUESTIONS[0],
        )

    print(
        f"[warm-up] latency={result['latency']}s, "
        f"success={result['success']}, "
        f"answer_len={result['answer_len']}"
    )


async def main():
    if RESULT_FILE.exists():
        RESULT_FILE.unlink()

    print("[perf] 开始并发压测 /chat")
    print(f"[perf] 请求地址: {CHAT_URL}")
    print(f"[perf] 结果文件: {RESULT_FILE}")

    await warm_up()

    for concurrency in CONCURRENCY_LEVELS:
        await run_one_concurrency_level(concurrency)

    print("\n[perf] 测试完成")
    print(f"[perf] 原始结果已保存到: {RESULT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
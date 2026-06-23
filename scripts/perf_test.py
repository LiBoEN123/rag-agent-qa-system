import csv
import time
import statistics
from pathlib import Path
from datetime import datetime

import requests


# =========================
# 1. 基础配置
# =========================

BASE_URL = "http://127.0.0.1:8000"
CHAT_URL = f"{BASE_URL}/chat"

OUTPUT_DIR = Path("outputs/perf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "perf_result.csv"

# 每个 case 重复请求次数
REPEAT_TIMES = 5

# 请求超时时间，单位：秒
TIMEOUT_SECONDS = 120


# =========================
# 2. 测试用例
# =========================
# 这里故意准备三类：
# 1）短问题短回答
# 2）长问题短回答
# 3）短问题长回答

TEST_CASES = [
    {
        "case_name": "short_question_short_answer",
        "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
        "top_k": 5,
    },
    {
        "case_name": "long_question_short_answer",
        "question": (
            "我正在学习 Vue 官方文档中关于应用创建的部分。"
            "请你只根据当前知识库内容回答：每个 Vue 应用通常是通过哪个函数"
            "来创建应用实例的？请只回答函数名，并用一句话简单说明。"
        ),
        "top_k": 5,
    },
    {
        "case_name": "short_question_long_answer",
        "question": "快速上手中的 CDN 使用方式和应用挂载文档中的 DOM 内根组件模板有什么关系？",
        "top_k": 5,
    },
]


# =========================
# 3. 单次请求函数
# =========================

def run_one_request(case: dict, round_index: int) -> dict:
    """
    发送一次 /chat 请求，并记录耗时、答案长度、引用数量等信息。
    """

    question = case["question"]

    payload = {
        "question": question,
        "top_k": case.get("top_k", 5),
    }

    start_time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = time.perf_counter()

    status_code = None
    success = False
    error = ""
    answer = ""
    reference_count = 0

    try:
        response = requests.post(
            CHAT_URL,
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )

        status_code = response.status_code
        result = response.json()

        # 兼容你现在常见的返回结构：
        # {
        #   "code": 0,
        #   "message": "success",
        #   "data": {
        #       "answer": "...",
        #       "references": [...]
        #   }
        # }

        data = result.get("data") or {}

        answer = data.get("answer") or ""
        references = data.get("references") or []
        reference_count = len(references)

        # 判断是否成功：
        # HTTP 200 并且没有明显异常，就先认为成功
        success = response.status_code == 200

        # 如果你的业务 code 不是 0，也记录一下
        if result.get("code") not in (0, "0", None):
            error = f"业务 code 非 0: {result.get('code')}, message={result.get('message')}"

    except Exception as e:
        error = str(e)

    end = time.perf_counter()
    end_time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    latency_seconds = round(end - start, 3)

    return {
        "case_name": case["case_name"],
        "round_index": round_index,
        "question": question,
        "question_len": len(question),
        "start_time": start_time_text,
        "end_time": end_time_text,
        "latency_seconds": latency_seconds,
        "status_code": status_code,
        "success": success,
        "answer_len": len(answer),
        "reference_count": reference_count,
        "error": error,
    }


# =========================
# 4. 保存 CSV
# =========================

def save_results(rows: list[dict]):
    """将压测明细写入 CSV 文件。"""
    if not rows:
        print("[perf] 没有结果可保存")
        return

    fieldnames = [
        "case_name",
        "round_index",
        "question",
        "question_len",
        "start_time",
        "end_time",
        "latency_seconds",
        "status_code",
        "success",
        "answer_len",
        "reference_count",
        "error",
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[perf] 结果已保存到：{OUTPUT_FILE}")


# =========================
# 5. 打印汇总结果
# =========================

def print_summary(rows: list[dict]):
    """按 case 打印压测成功次数和延迟统计。"""
    print("\n========== 压测汇总 ==========")

    case_names = sorted(set(row["case_name"] for row in rows))

    for case_name in case_names:
        case_rows = [row for row in rows if row["case_name"] == case_name]
        success_rows = [row for row in case_rows if row["success"]]

        if not success_rows:
            print(f"\ncase_name: {case_name}")
            print("全部请求失败")
            continue

        latencies = [row["latency_seconds"] for row in success_rows]
        answer_lens = [row["answer_len"] for row in success_rows]

        avg_latency = round(statistics.mean(latencies), 3)
        min_latency = round(min(latencies), 3)
        max_latency = round(max(latencies), 3)
        avg_answer_len = round(statistics.mean(answer_lens), 1)

        print(f"\ncase_name: {case_name}")
        print(f"请求次数: {len(case_rows)}")
        print(f"成功次数: {len(success_rows)}")
        print(f"平均耗时: {avg_latency} 秒")
        print(f"最小耗时: {min_latency} 秒")
        print(f"最大耗时: {max_latency} 秒")
        print(f"平均答案长度: {avg_answer_len}")


# =========================
# 6. 主流程
# =========================

def main():
    """执行同步 /chat 压测并保存结果。"""
    print("[perf] 开始压测 /chat")
    print(f"[perf] 请求地址: {CHAT_URL}")
    print(f"[perf] 每个 case 重复次数: {REPEAT_TIMES}")

    all_rows = []

    # 可选：先热身一次，不计入结果
    # 这样可以减少第一次模型冷启动对统计的影响
    print("\n[perf] warm-up 请求开始，不计入最终结果")
    warmup_case = TEST_CASES[0]
    warmup_result = run_one_request(warmup_case, round_index=0)
    print(
        f"[warm-up] latency={warmup_result['latency_seconds']}s, "
        f"success={warmup_result['success']}, "
        f"answer_len={warmup_result['answer_len']}"
    )

    print("\n[perf] 正式测试开始")

    for case in TEST_CASES:
        print(f"\n[perf] 当前 case: {case['case_name']}")

        for i in range(1, REPEAT_TIMES + 1):
            row = run_one_request(case, round_index=i)
            all_rows.append(row)

            print(
                f"  第 {i} 次 | "
                f"耗时: {row['latency_seconds']}s | "
                f"成功: {row['success']} | "
                f"答案长度: {row['answer_len']} | "
                f"引用数: {row['reference_count']}"
            )

    save_results(all_rows)
    print_summary(all_rows)


if __name__ == "__main__":
    main()

























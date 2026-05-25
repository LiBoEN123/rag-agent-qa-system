import json
import csv
import time
import re
import math
from pathlib import Path

from app.rag import init_rag, search_relevant_chunks, get_rag_status, ask
from app.config import Settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 注意：
# 你需要把题集文件放到 data/eval/ 下面，并命名为：
# generated_20_questions_key_points_compact.json
QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "generated_20_questions_key_points_compact.json"


# =========================
# 基础读取与 RAG 调用
# =========================

def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_rag_system(question, top_k=None):
    final_top_k = top_k if top_k is not None else Settings.TOP_K

    rag_status = get_rag_status()
    if not rag_status["ready"]:
        raise RuntimeError(rag_status["message"])

    retrieved_chunks = search_relevant_chunks(question, final_top_k)

    # 保留 retrieved_chunk_ids，虽然现在不作为最终评分依据，
    # 但后续排查检索问题时很有用。
    retrieved_chunk_ids = [
        c.get("chunk_id", "")
        for c in retrieved_chunks
    ]

    result = ask(
        question,
        top_k=final_top_k,
        retrieved_chunks=retrieved_chunks
    )

    refs = result.get("references", [])

    retrieved_doc_ids = [
        c.get("file_name", "")
        for c in retrieved_chunks
    ]

    return {
        "status": result.get("status", ""),
        "answer": result.get("answer", ""),
        "retrieved_count": len(retrieved_chunks),
        "retrieved_doc_ids": retrieved_doc_ids,
        "top3_doc_ids": retrieved_doc_ids[:3],

        # 这两个字段现在不参与最终答案评分，
        # 但保留，方便你判断答案错误是不是因为引用绑定问题。
        "reference_chunk_ids": [
            r.get("chunk_id", "")
            for r in refs
        ],
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }


def find_first_hit_rank(retrieved_doc_ids, gold_doc_ids):
    if not gold_doc_ids:
        return None

    gold_set = set(gold_doc_ids)

    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_set:
            return i

    return None


def infer_expected_behavior(item):
    value = str(item.get("expected_behavior", "")).strip().lower()

    if value in {"answer", "abstain"}:
        return value

    # 兜底逻辑：
    # 如果没有写 expected_behavior，则按照 category 推断。
    if item.get("category") == "not_in_doc":
        return "abstain"

    return "answer"


# =========================
# 关键词命中评测
# =========================

KEY_POINT_PASS_RATE = 0.6
KEY_POINT_UNIT_PASS_RATE = 0.6


def normalize_for_match(text):
    """
    用于关键词匹配的简单归一化。

    主要处理：
    1. 大小写统一
    2. 空格、反引号、引号去掉
    3. 少量常见同义写法统一

    注意：
    这里不要做得太复杂。
    这是一个轻量关键词评测，不是语义评测模型。
    """
    text = str(text or "").lower()

    replacements = {
        "javascript": "js",
        "dom 元素": "dom元素",
        "css 选择器": "css选择器",
        "xss 漏洞": "xss风险",
        "单文件组件": "sfc",
        "根组件的实例": "根组件实例",
        "组件实例": "根组件实例",
        "scriptsetup": "script setup",
        "<scriptsetup>": "script setup",
        "<script setup>": "script setup",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", "", text)

    for ch in ["`", "“", "”", "\"", "'", "‘", "’"]:
        text = text.replace(ch, "")

    return text


def build_match_units(key_point):
    """
    把一个 gold_key_point 拆成更小的匹配单元。

    例如：
    - "简介createApp+mount" 会拆出 createapp 和 mount
    - "$emit/defineEmits" 会拆出 $emit 和 defineemits
    - "返回根组件实例" 会拆成若干中文二字片段

    这样可以避免关键词写得略微压缩时完全匹配失败。
    """
    key_point = normalize_for_match(key_point)

    rough_parts = re.split(r"[+/|｜、,，;；()（）]+", key_point)

    units = []

    for part in rough_parts:
        if not part:
            continue

        tokens = re.findall(
            r"\$?[a-z0-9_.@#-]+|[\u4e00-\u9fff]+",
            part
        )

        for token in tokens:
            if not token:
                continue

            # 中文短语拆成 2 字滑动窗口。
            # 例如“根组件实例” -> “根组”“组件”“件实”“实例”
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                if len(token) <= 2:
                    units.append(token)
                else:
                    for i in range(len(token) - 1):
                        units.append(token[i:i + 2])
            else:
                units.append(token)

    # 过滤太泛的词，避免“应用”“文档”这类词导致误判。
    stop_units = {
        "文档",
        "应用",
        "简介",
        "共同",
        "基础",
        "说明",
        "当前",
        "可以",
        "没有",
        "使用",
        "需要",
        "支持",
    }

    final_units = []

    for u in units:
        if not u:
            continue

        if u in stop_units:
            continue

        if len(u) == 1 and u not in {"@"}:
            continue

        if u not in final_units:
            final_units.append(u)

    return final_units


def is_key_point_hit(pred_answer, key_point):
    """
    判断一个 gold_key_point 是否被预测答案命中。

    两层判断：
    1. 完整关键词直接出现在答案中，直接算命中。
    2. 如果完整关键词没有出现，则拆成小单元，命中一定比例也算命中。
    """
    answer_text = normalize_for_match(pred_answer)
    key_text = normalize_for_match(key_point)

    if not key_text:
        return False

    # 第一层：完整关键词直接命中。
    if key_text in answer_text:
        return True

    # 第二层：拆分匹配。
    units = build_match_units(key_text)

    if not units:
        return False

    hit_units = [
        u for u in units
        if u in answer_text
    ]

    required_hits = max(
        1,
        math.ceil(len(units) * KEY_POINT_UNIT_PASS_RATE)
    )

    return len(hit_units) >= required_hits


def evaluate_key_points(pred_answer, gold_key_points):
    """
    对一条预测答案做关键词命中评测。

    返回字段：
    - hit_key_points：命中的关键点
    - miss_key_points：未命中的关键点
    - key_point_hit_count：命中数量
    - key_point_total：关键点总数
    - key_point_recall：关键点命中率
    - required_key_points：至少需要命中几个关键点才算通过
    - answer_key_point_passed：本题答案关键词是否达标
    """
    gold_key_points = gold_key_points or []

    hit_key_points = []
    miss_key_points = []

    for kp in gold_key_points:
        if is_key_point_hit(pred_answer, kp):
            hit_key_points.append(kp)
        else:
            miss_key_points.append(kp)

    total = len(gold_key_points)
    hit_count = len(hit_key_points)

    recall = round(hit_count / total, 4) if total else 0.0

    required_key_points = (
        max(1, math.ceil(total * KEY_POINT_PASS_RATE))
        if total
        else 0
    )

    passed = total > 0 and hit_count >= required_key_points

    return {
        "gold_key_points": gold_key_points,
        "hit_key_points": hit_key_points,
        "miss_key_points": miss_key_points,
        "key_point_hit_count": hit_count,
        "key_point_total": total,
        "key_point_recall": recall,
        "required_key_points": required_key_points,
        "answer_key_point_passed": passed,
    }


# =========================
# 错误类型分类
# =========================

def classify_simple_error(row):
    """
    最小分类逻辑。

    对应回答题：
    1. 程序异常 -> exception
    2. 正确文档没进入 top-3 -> miss_retrieval
    3. 正确文档进入 top-3，但系统拒答 -> false_abstain
    4. 答案关键词命中不足 -> missing_key_points
    5. 其他 -> success

    对应拒答题：
    1. 系统拒答 -> success
    2. 系统回答 -> should_abstain_but_answered
    """
    expected_behavior = row["expected_behavior"]
    pred_status = row["pred_status"]
    hit_at_3 = row["hit_at_3"]

    if pred_status == "EXCEPTION":
        return "exception", "程序异常"

    if expected_behavior == "abstain":
        if pred_status == "NO_RELEVANT_INFO":
            return "success", "拒答正确"

        return (
            "should_abstain_but_answered",
            "本题应拒答，但系统给了具体回答"
        )

    # expected_behavior == "answer"

    if not hit_at_3:
        return (
            "miss_retrieval",
            "正确文档没有进入 top-3，属于漏召回"
        )

    if pred_status == "NO_RELEVANT_INFO":
        return (
            "false_abstain",
            "正确文档已进入 top-3，但系统仍然拒答"
        )

    if not row.get("answer_key_point_passed", False):
        return (
            "missing_key_points",
            f"答案关键点命中不足，"
            f"命中率={row.get('key_point_recall', 0.0)}，"
            f"未命中={row.get('miss_key_points', [])}"
        )

    return "success", "检索和回答基本正常"


# =========================
# 单题运行
# =========================

def run_one_question(item, top_k=5):
    start = time.time()

    expected_behavior = infer_expected_behavior(item)
    gold_doc_ids = item.get("gold_doc_ids", [])
    gold_key_points = item.get("gold_key_points", [])

    try:
        resp = call_rag_system(
            item["question"],
            top_k=top_k
        )

        gold_rank = find_first_hit_rank(
            resp.get("retrieved_doc_ids", []),
            gold_doc_ids
        )

        hit_at_1 = gold_rank == 1
        hit_at_3 = gold_rank is not None and gold_rank <= 3
        hit_at_5 = gold_rank is not None and gold_rank <= 5

        is_correct_abstain = (
            expected_behavior == "abstain"
            and resp.get("status", "") == "NO_RELEVANT_INFO"
        )

        key_point_eval = evaluate_key_points(
            pred_answer=resp.get("answer", ""),
            gold_key_points=gold_key_points
        )

        result = {
            "qid": item["qid"],
            "question": item["question"],
            "category": item.get("category", ""),
            "gold_answer": item.get("gold_answer", ""),
            "gold_doc_ids": gold_doc_ids,
            "gold_key_points": gold_key_points,
            "expected_behavior": expected_behavior,

            "pred_answer": resp.get("answer", ""),
            "pred_status": resp.get("status", ""),

            "retrieved_count": resp.get("retrieved_count", 0),
            "retrieved_doc_ids": resp.get("retrieved_doc_ids", []),
            "top3_doc_ids": resp.get("top3_doc_ids", []),
            "retrieved_chunk_ids": resp.get("retrieved_chunk_ids", []),
            "reference_chunk_ids": resp.get("reference_chunk_ids", []),

            "gold_rank": gold_rank,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "is_correct_abstain": is_correct_abstain,

            "hit_key_points": key_point_eval["hit_key_points"],
            "miss_key_points": key_point_eval["miss_key_points"],
            "key_point_hit_count": key_point_eval["key_point_hit_count"],
            "key_point_total": key_point_eval["key_point_total"],
            "key_point_recall": key_point_eval["key_point_recall"],
            "required_key_points": key_point_eval["required_key_points"],
            "answer_key_point_passed": key_point_eval["answer_key_point_passed"],

            "latency_sec": round(time.time() - start, 3),
        }

        error_type, failure_reason = classify_simple_error(result)

        result["error_type"] = error_type
        result["failure_reason"] = failure_reason

        return result

    except Exception as e:
        return {
            "qid": item["qid"],
            "question": item["question"],
            "category": item.get("category", ""),
            "gold_answer": item.get("gold_answer", ""),
            "gold_doc_ids": gold_doc_ids,
            "gold_key_points": gold_key_points,
            "expected_behavior": expected_behavior,

            "pred_answer": "",
            "pred_status": "EXCEPTION",

            "retrieved_count": 0,
            "retrieved_doc_ids": [],
            "top3_doc_ids": [],
            "retrieved_chunk_ids": [],
            "reference_chunk_ids": [],

            "gold_rank": None,
            "hit_at_1": False,
            "hit_at_3": False,
            "hit_at_5": False,
            "is_correct_abstain": False,

            "hit_key_points": [],
            "miss_key_points": gold_key_points,
            "key_point_hit_count": 0,
            "key_point_total": len(gold_key_points),
            "key_point_recall": 0.0,
            "required_key_points": 0,
            "answer_key_point_passed": False,

            "latency_sec": round(time.time() - start, 3),
            "error_type": "exception",
            "failure_reason": f"程序异常: {e}",
        }


# =========================
# 保存结果
# =========================

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def save_csv(results, path):
    rows = []

    for r in results:
        rows.append({
            "qid": r["qid"],
            "question": r["question"],
            "category": r["category"],
            "expected_behavior": r["expected_behavior"],

            "gold_answer": r["gold_answer"],
            "gold_doc_ids": "|".join(map(str, r.get("gold_doc_ids", []))),
            "gold_key_points": "|".join(map(str, r.get("gold_key_points", []))),

            "pred_status": r["pred_status"],
            "pred_answer": r["pred_answer"],

            "hit_key_points": "|".join(map(str, r.get("hit_key_points", []))),
            "miss_key_points": "|".join(map(str, r.get("miss_key_points", []))),
            "key_point_hit_count": r.get("key_point_hit_count", 0),
            "key_point_total": r.get("key_point_total", 0),
            "key_point_recall": r.get("key_point_recall", 0.0),
            "required_key_points": r.get("required_key_points", 0),
            "answer_key_point_passed": r.get("answer_key_point_passed", False),

            "gold_rank": r["gold_rank"],
            "hit_at_1": r["hit_at_1"],
            "hit_at_3": r["hit_at_3"],
            "hit_at_5": r["hit_at_5"],
            "is_correct_abstain": r["is_correct_abstain"],

            "retrieved_count": r["retrieved_count"],
            "retrieved_doc_ids": "|".join(map(str, r.get("retrieved_doc_ids", []))),
            "top3_doc_ids": "|".join(map(str, r.get("top3_doc_ids", []))),
            "retrieved_chunk_ids": "|".join(map(str, r.get("retrieved_chunk_ids", []))),
            "reference_chunk_ids": "|".join(map(str, r.get("reference_chunk_ids", []))),

            "error_type": r["error_type"],
            "failure_reason": r["failure_reason"],
            "latency_sec": r["latency_sec"],
        })

    if not rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


# =========================
# 汇总指标
# =========================

def build_summary(results):
    total = len(results)

    status_count = {}
    category_count = {}
    error_type_count = {}

    answer_questions = [
        r for r in results
        if r["expected_behavior"] == "answer"
    ]

    abstain_questions = [
        r for r in results
        if r["expected_behavior"] == "abstain"
    ]

    for r in results:
        s = r.get("pred_status", "UNKNOWN")
        status_count[s] = status_count.get(s, 0) + 1

        c = r.get("category", "UNKNOWN")
        category_count[c] = category_count.get(c, 0) + 1

        e = r.get("error_type", "UNKNOWN")
        error_type_count[e] = error_type_count.get(e, 0) + 1

    def safe_rate(n, d):
        return round(n / d, 4) if d else 0.0

    hit_at_1 = safe_rate(
        sum(r["hit_at_1"] for r in answer_questions),
        len(answer_questions)
    )

    hit_at_3 = safe_rate(
        sum(r["hit_at_3"] for r in answer_questions),
        len(answer_questions)
    )

    hit_at_5 = safe_rate(
        sum(r["hit_at_5"] for r in answer_questions),
        len(answer_questions)
    )

    abstain_accuracy = safe_rate(
        sum(r["is_correct_abstain"] for r in abstain_questions),
        len(abstain_questions)
    )

    keyword_answer_accuracy = safe_rate(
        sum(
            r.get("answer_key_point_passed", False)
            for r in answer_questions
        ),
        len(answer_questions)
    )

    avg_key_point_recall = (
        round(
            sum(
                r.get("key_point_recall", 0.0)
                for r in answer_questions
            ) / len(answer_questions),
            4
        )
        if answer_questions
        else 0.0
    )

    return {
        "total_questions": total,
        "answer_questions": len(answer_questions),
        "abstain_questions": len(abstain_questions),

        "status_count": status_count,
        "category_count": category_count,
        "error_type_count": error_type_count,

        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,

        "abstain_accuracy": abstain_accuracy,

        # 新增：关键词答案评测指标
        "keyword_answer_accuracy": keyword_answer_accuracy,
        "avg_key_point_recall": avg_key_point_recall,
    }


# =========================
# 失败案例 Markdown
# =========================

def build_bad_cases_md(results, path, top_n=3):
    """
    只挑最差的几个案例。

    优先级：
    1. exception
    2. should_abstain_but_answered
    3. miss_retrieval
    4. false_abstain
    5. missing_key_points
    """
    priority = {
        "exception": 100,
        "should_abstain_but_answered": 90,
        "miss_retrieval": 80,
        "false_abstain": 70,
        "missing_key_points": 60,
        "success": 0,
    }

    bad_cases = [
        r for r in results
        if r["error_type"] != "success"
    ]

    bad_cases.sort(
        key=lambda x: priority.get(x["error_type"], 10),
        reverse=True
    )

    bad_cases = bad_cases[:top_n]

    lines = ["# Bad Cases\n"]

    if not bad_cases:
        lines.append("没有失败案例。")
    else:
        for i, r in enumerate(bad_cases, start=1):
            lines.append(f"## Case {i}: {r['qid']}")
            lines.append(f"- 问题: {r['question']}")
            lines.append(f"- 类别: {r['category']}")
            lines.append(f"- 期望行为: {r['expected_behavior']}")
            lines.append(
                f"- 期望来源: "
                f"{', '.join(r['gold_doc_ids']) if r['gold_doc_ids'] else '无'}"
            )
            lines.append(
                f"- 实际 top-3: "
                f"{', '.join(r['top3_doc_ids']) if r['top3_doc_ids'] else '无'}"
            )
            lines.append(f"- gold_rank: {r['gold_rank']}")
            lines.append(f"- pred_status: {r['pred_status']}")
            lines.append(f"- 最终回答: {r['pred_answer'] or '空'}")

            lines.append(
                f"- 标准关键点: "
                f"{', '.join(r.get('gold_key_points', [])) if r.get('gold_key_points') else '无'}"
            )
            lines.append(
                f"- 已命中关键点: "
                f"{', '.join(r.get('hit_key_points', [])) if r.get('hit_key_points') else '无'}"
            )
            lines.append(
                f"- 未命中关键点: "
                f"{', '.join(r.get('miss_key_points', [])) if r.get('miss_key_points') else '无'}"
            )
            lines.append(f"- 关键点命中率: {r.get('key_point_recall', 0.0)}")

            lines.append(f"- 错误类型: {r['error_type']}")
            lines.append(f"- 失败原因: {r['failure_reason']}")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 主程序入口
# =========================

def main():
    run_name = "baseline_20q_topk5_key_points_rerank_eval"
    top_k = 7

    Settings.RAG_FILE_PATH = PROJECT_ROOT / "data" / "kb"
    Settings.validate()

    init_rag(
        str(Settings.RAG_FILE_PATH),
        chunk_size=Settings.CHUNK_SIZE,
        chunk_overlap=Settings.CHUNK_OVERLAP
    )

    questions = load_questions(QUESTIONS_PATH)

    results = [
        run_one_question(q, top_k=top_k)
        for q in questions
    ]

    out_dir = OUTPUTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        results,
        out_dir / "results.json"
    )

    save_csv(
        results,
        out_dir / "results.csv"
    )

    save_json(
        build_summary(results),
        out_dir / "summary.json"
    )

    build_bad_cases_md(
        results,
        out_dir / "bad_cases.md",
        top_n=3
    )

    print(f"[eval] 评测完成，共 {len(results)} 道题")
    print(f"[eval] 输出目录：{out_dir}")


if __name__ == "__main__":
    main()
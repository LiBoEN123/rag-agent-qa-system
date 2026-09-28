"""RAG 评测使用的通用解析、归一化与数据加载工具。

用途：读取当前题集与 Prediction v3，统一召回标识、拒答表达和关键点文本。
输入：评测 JSON、预测 CSV/JSON/JSONL，以及 Schema v3 预测记录。
输出：规范化文本、题目/预测索引和轻量 ``ChunkIndex`` 数据视图。
边界：本模块不计算最终分数、不调用模型或网络，也不改变原始评测文件。
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]


PREDICTION_SCHEMA_VERSION = 3

REFUSAL_PATTERNS = [
    # 强拒答
    "无法确认", "不能确认", "无法确定", "无法回答", "不能回答", "不知道",
    # 文档范围限制
    "无法根据提供的文档确定", "无法根据当前文档确定", "无法根据提供的资料确定",
    "根据提供的文档无法", "根据当前文档无法", "根据现有文档无法", "从给定文档无法",
    "参考资料中未提供", "当前资料中未提供", "提供的资料中未提供",
    "文档中未说明", "文档未说明", "文档没有说明", "文档没有提供",
    "资料中未说明", "资料未说明", "没有相关信息", "无相关信息",
    "未提到", "未提及", "没有提到", "没有提及", "不在文档中",
]

STANDALONE_REFUSAL_PATTERNS = [
    "无法确认", "不能确认", "无法确定", "无法回答", "不能回答", "不知道",
    "无法根据提供的文档确定", "无法根据当前文档确定", "无法根据提供的资料确定",
    "根据提供的文档无法", "根据当前文档无法", "根据现有文档无法", "从给定文档无法",
    "没有相关信息", "无相关信息",
]

CANONICAL_REPLACEMENTS = [
    ("无需", "不需要"),
    ("不用", "不需要"),
    ("不必", "不需要"),
    ("未提及", "未说明"),
    ("未提到", "未说明"),
    ("没有提及", "未说明"),
    ("没有提到", "未说明"),
    ("参考资料", "文档"),
    ("当前资料", "文档"),
    ("提供的资料", "文档"),
]


@dataclass(frozen=True)
class ChunkIndex:
    """评测所需的当前 chunk 索引视图。"""

    chunk_doc_map: Dict[int, str]
    chunk_text_map: Dict[int, str]
    source: str


def normalize_text(s: Any) -> str:
    """标准化中英文文本，便于宽松匹配。"""
    if s is None:
        return ""
    s = str(s).lower()
    for old, new in CANONICAL_REPLACEMENTS:
        s = s.replace(old.lower(), new.lower())
    s = re.sub(r"\s+", "", s)
    # 去掉常见标点，同时保留 Vue 等技术词中有区分度的代码式字符。
    s = re.sub(r"[，。！？、；：,.!?;:（）()【】\[\]{}<>《》“”\"'`·\-_—|/\\]", "", s)
    return s


def is_subsequence(short: str, long: str) -> bool:
    """判断短文本是否按顺序出现在长文本中，允许中间插入字符。"""
    if not short or len(short) < 4:
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def has_any(text: str, aliases: Iterable[str]) -> bool:
    """判断文本是否命中任一同义表达或宽松子序列。"""
    norm = normalize_text(text)
    for a in aliases:
        na = normalize_text(a)
        if not na:
            continue
        if na in norm:
            return True
        # 对中文短语做宽松匹配，避免“可以多个应用实例”与“可以创建多个 Vue 应用实例”这种同义表述漏判
        if len(na) >= 5 and is_subsequence(na, norm):
            return True
    return False


def is_refusal(answer: str) -> bool:
    """判断答案是否属于拒答或无法回答。"""
    return has_any(answer, REFUSAL_PATTERNS)


def is_standalone_refusal(answer: str) -> bool:
    """高精度识别整段拒答，避免把长答案中的局部限制误判为拒答。"""
    normalized = normalize_text(answer)
    if not normalized:
        return False
    for pattern in STANDALONE_REFUSAL_PATTERNS:
        normalized_pattern = normalize_text(pattern)
        if (
            normalized_pattern in normalized
            and len(normalized) <= max(48, len(normalized_pattern) * 3)
        ):
            return True
    return False


def parse_chunk_ids_field(pred: Dict[str, Any], field: str) -> List[int]:
    """读取 v3 的整数列表或 CSV 逗号分隔 ID，保留首次观察顺序。"""
    value = pred.get(field, [])
    if isinstance(value, str):
        value = value.split(",") if value else []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list or comma-separated chunk IDs")
    if any(isinstance(item, bool) or not str(item).isascii() or not str(item).isdigit() for item in value):
        raise ValueError(f"{field} must contain integer chunk IDs")
    return list(dict.fromkeys(int(item) for item in value))


def validate_prediction(pred: Dict[str, Any]) -> None:
    """正式评分只接受 Schema v3，缺字段不能伪装成空召回。"""
    if str(pred.get("prediction_schema_version")) != str(PREDICTION_SCHEMA_VERSION):
        raise ValueError("RAG prediction requires Schema v3 records")
    required = {"answer", "retrieved_chunk_ids", "reference_chunk_ids"}
    if missing := required - pred.keys():
        raise ValueError(f"RAG prediction missing fields: {', '.join(sorted(missing))}")
    for field in ("retrieved_chunk_ids", "reference_chunk_ids"):
        parse_chunk_ids_field(pred, field)


def load_eval(path: str | Path) -> List[Dict[str, Any]]:
    """读取并校验当前 RAG 题集，不加载索引或模型。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list) or not data:
        raise ValueError("评测文件必须包含非空题目列表")
    qids = set()
    for item in data:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("qid", "question")
        ):
            raise ValueError("评测题必须包含非空 qid 和 question")
        if item["qid"] in qids:
            raise ValueError(f"评测题 qid 重复：{item['qid']}")
        qids.add(item["qid"])
    return data


def load_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取 CSV、JSONL 或 JSON 预测结果并按 qid 建索引。"""
    suffix = path.suffix.lower()
    rows: List[Dict[str, Any]] = []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "items" in obj:
            rows = obj["items"]
        elif isinstance(obj, list):
            rows = obj
        else:
            raise ValueError("Unsupported JSON prediction format")
    else:
        raise ValueError("Prediction file must be .csv, .json, or .jsonl")

    out = {}
    for row in rows:
        validate_prediction(row)
        qid = str(row.get("qid", "")).strip()
        if not qid or qid in out:
            raise ValueError("Prediction qid must be nonempty and unique")
        out[qid] = row
    return out


def warn(message: str) -> None:
    """向命令行输出非致命评测提示。"""
    print(f"[warn] {message}", file=sys.stderr)


def point_text(point: Any) -> str:
    """从 gold_key_points / forbidden_points 的字符串或字典格式中取原始点文本。"""
    if isinstance(point, dict):
        return str(point.get("point", "")).strip()
    return str(point).strip()


def point_texts(points: Iterable[Any]) -> List[str]:
    """提取关键点文本列表并去掉空项。"""
    return [text for text in (point_text(p) for p in points or []) if text]


def json_dumps(value: Any) -> str:
    """统一 JSON 序列化，便于 CSV 和 JSONL 中保留中文。"""
    return json.dumps(value, ensure_ascii=False)


def mean(values: List[float]) -> float:
    """安全计算均值，空列表返回 0。"""
    return sum(values) / len(values) if values else 0.0

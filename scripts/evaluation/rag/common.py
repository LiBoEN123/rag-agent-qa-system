"""RAG 评测使用的通用解析、归一化与数据加载工具。

用途：统一答案字段、召回标识、拒答表达、关键点文本和多种历史文件格式。
输入：评测 JSON、预测 CSV/JSON/JSONL，以及可能含旧字段形态的单条记录。
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


ANSWER_FIELDS = ["answer", "generated_answer", "actual_answer", "response", "model_answer"]

CHUNK_FIELDS = ["retrieved_chunk_ids", "retrieved_chunks", "chunk_ids", "top_chunk_ids"]

DOC_FIELDS = ["retrieved_doc_ids", "retrieved_docs", "doc_ids", "top_doc_ids"]

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


def parse_listish(value: Any) -> List[Any]:
    """将 JSON 列表、分隔字符串或字典列表统一解析为列表。"""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            if isinstance(x, dict):
                # 兼容常见的检索结果字典形态。
                for key in ("chunk_id",):
                    if key in x:
                        out.append(x[key])
                        break
                else:
                    metadata = x.get("metadata") or {}
                    found = False
                    if isinstance(metadata, dict):
                        for key in ("chunk_id", "id", "doc_id", "file_name"):
                            if key in metadata:
                                out.append(metadata[key])
                                found = True
                                break
                    if not found:
                        for key in ("id", "doc_id", "file_name"):
                            if key in x:
                                out.append(x[key])
                                break
            else:
                out.append(x)
        return out
    if isinstance(value, (int, float)):
        return [int(value)]
    s = str(value).strip()
    if not s:
        return []
    # 优先按 JSON 数组解析字符串。
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parse_listish(parsed)
        if isinstance(parsed, dict):
            return parse_listish([parsed])
    except Exception:
        pass
    # JSON 解析失败后，再按常见分隔符提取标识。
    parts = re.split(r"[,\|;，；\s]+", s)
    return [p for p in parts if p != ""]


def parse_chunk_ids(pred: Dict[str, Any]) -> List[int]:
    """从预测记录中提取检索到的 chunk_id 列表。"""
    value = None
    for k in CHUNK_FIELDS:
        if k in pred:
            value = pred[k]
            break
    raw = parse_listish(value)
    ids = []
    for x in raw:
        if isinstance(x, int):
            ids.append(x)
            continue
        sx = str(x).strip()
        if re.fullmatch(r"\d+", sx):
            ids.append(int(sx))
            continue
        m = re.search(r"\b(?:chunk_id|chunk-id|cid|id)\D+(\d+)\b", sx, flags=re.I)
        if not m:
            m = re.search(r"\bchunk[_-]?(\d+)(?![A-Za-z0-9])", sx, flags=re.I)
        if m:
            ids.append(int(m.group(1)))
    return ids


def parse_doc_ids(pred: Dict[str, Any]) -> List[str]:
    """从预测记录中提取检索到的文档标识列表。"""
    value = None
    for k in DOC_FIELDS:
        if k in pred:
            value = pred[k]
            break
    return [str(x) for x in parse_listish(value)]


def get_answer(pred: Dict[str, Any]) -> str:
    """从预测记录的常见答案字段中取出答案文本。"""
    for k in ANSWER_FIELDS:
        if k in pred:
            return "" if pred[k] is None else str(pred[k])
    return ""


def load_eval(path: Path) -> List[Dict[str, Any]]:
    """读取评测集文件并兼容 list 或带 items 的对象。"""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "items" in obj:
        return obj["items"]
    if isinstance(obj, list):
        return obj
    raise ValueError("eval file must be a list or a JSON object with an 'items' field")


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
        elif isinstance(obj, dict):
            # 兼容 {"q001": {...}} 这种以 qid 为键的对象。
            rows = []
            for qid, val in obj.items():
                if isinstance(val, dict):
                    val = {"qid": qid, **val}
                else:
                    val = {"qid": qid, "answer": val}
                rows.append(val)
        elif isinstance(obj, list):
            rows = obj
        else:
            raise ValueError("Unsupported JSON prediction format")
    else:
        raise ValueError("Prediction file must be .csv, .json, or .jsonl")

    out = {}
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        if qid:
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

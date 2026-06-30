# 启动命令：python scripts/evaluate_rag_results_semantic_fixed.py --eval-file data/eval/rag_eval_questions_v2_semantic_fixed.json --pred-file outputs/predictions.csv --out-dir outputs/rag_baseline1 --k 5

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

NEGATION_CUES = [
    "未提供", "没有提供", "未说明", "没有说明", "未提到", "没有提到", "未提及", "没有提及",
    "无法确认", "不能确认", "无法确定", "不能确定", "不包含", "不在文档", "文档中没有",
    "参考资料中没有", "参考资料中未", "当前资料中没有", "当前资料中未",
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
    # remove common punctuation while keeping code-ish chars useful for Vue terms
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


def has_all_terms(text: str, terms: Iterable[str]) -> bool:
    """一个语义关键点可由若干词共同表达，例如：导入 + 模板 + 直接 + 使用。"""
    norm = normalize_text(text)
    for term in terms:
        nt = normalize_text(term)
        if not nt:
            continue
        if nt not in norm and not (len(nt) >= 5 and is_subsequence(nt, norm)):
            return False
    return True


def match_point(answer: str, point: Dict[str, Any]) -> bool:
    """
    判断一个 gold_key_point 是否被答案覆盖。

    支持三层匹配：
    1. aliases：原有的短语/同义词包含匹配；
    2. match_any：若干“词组共同出现”规则，适合自然表达、代码示例、换说法；
       例：["导入", "模板", "直接", "使用"]；
    3. regex：少量需要正则的情况。
    """
    aliases = point.get("aliases") or [point.get("point", "")]
    if has_any(answer, aliases):
        return True

    for group in point.get("match_any", []) or []:
        if has_all_terms(answer, group):
            return True

    regex_patterns = point.get("regex", []) or []
    if isinstance(regex_patterns, str):
        regex_patterns = [regex_patterns]

    for pattern in regex_patterns:
        if re.search(pattern, answer, flags=re.I):
            return True

    return False


def is_refusal(answer: str) -> bool:
    """判断答案是否属于拒答或无法回答。"""
    return has_any(answer, REFUSAL_PATTERNS)


def parse_listish(value: Any) -> List[Any]:
    """将 JSON 列表、分隔字符串或字典列表统一解析为列表。"""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            if isinstance(x, dict):
                # common retrieval result shapes
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
    # try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parse_listish(parsed)
        if isinstance(parsed, dict):
            return parse_listish([parsed])
    except Exception:
        pass
    # extract ids from common separators
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
            # allow {"q001": {...}}
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


def coerce_int(value: Any) -> int | None:
    """宽松转换整数，失败时返回 None。"""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        m = re.search(r"\d+", str(value))
        return int(m.group(0)) if m else None


def normalize_doc_id(value: Any) -> str:
    """统一文档标识，兼容 file_name、doc_id 和 source_path。"""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return Path(text).name if "/" in text or "\\" in text else text


def doc_id_matches(candidate: str, gold_docs: Iterable[str]) -> bool:
    """判断当前 chunk 文档名是否可视为命中 gold_doc_ids。"""
    candidate_norm = normalize_doc_id(candidate)
    if not candidate_norm:
        return False
    for gold_doc in gold_docs:
        gold_norm = normalize_doc_id(gold_doc)
        if not gold_norm:
            continue
        if candidate_norm == gold_norm:
            return True
        if candidate_norm.endswith(gold_norm) or gold_norm.endswith(candidate_norm):
            return True
    return False


def doc_label_from_metadata(metadata: Dict[str, Any]) -> str:
    """从 chunk metadata 中提取最稳定的文档标签。"""
    return str(
        metadata.get("file_name")
        or normalize_doc_id(metadata.get("source_path"))
        or metadata.get("doc_id")
        or ""
    )


def chunks_to_index(chunks: List[Dict[str, Any]], source: str) -> ChunkIndex:
    """将 ChromaDB 或 legacy chunks.json 统一成评测索引。"""
    chunk_doc_map: Dict[int, str] = {}
    chunk_text_map: Dict[int, str] = {}

    for idx, item in enumerate(chunks):
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        chunk_id = coerce_int(metadata.get("chunk_id"))
        if chunk_id is None:
            chunk_id = coerce_int(item.get("chunk_id"))
        if chunk_id is None:
            # 评测预测使用 chunk_id，对缺少 id 的记录无法可靠对齐。
            continue

        text = item.get("text")
        if text is None:
            text = item.get("document")
        if text is None:
            text = item.get("content")

        chunk_doc_map[chunk_id] = doc_label_from_metadata(metadata) or f"chunk_{idx}"
        chunk_text_map[chunk_id] = "" if text is None else str(text)

    return ChunkIndex(chunk_doc_map=chunk_doc_map, chunk_text_map=chunk_text_map, source=source)


def load_chunks_file_index(chunks_file: Path | None) -> ChunkIndex:
    """从旧 chunks.json 读取 chunk 文本和文档映射，作为兼容回退。"""
    if not chunks_file:
        return ChunkIndex({}, {}, "none")
    if not chunks_file.exists():
        warn(f"chunks file not found: {chunks_file}; will try ChromaDB")
        return ChunkIndex({}, {}, "none")

    data = json.loads(chunks_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items") or data.get("chunks") or []
    if not isinstance(data, list):
        raise ValueError("chunks file must be a JSON list or an object with 'items'/'chunks'")

    return chunks_to_index(data, source=f"chunks-file:{chunks_file}")


def load_chroma_index(chroma_persist_dir: Path | None, chroma_collection_name: str | None) -> ChunkIndex:
    """从当前 ChromaDB collection 读取 chunk 文本和 metadata。"""
    try:
        from app.config import Settings

        if chroma_persist_dir is not None:
            Settings.CHROMA_PERSIST_DIR = (
                chroma_persist_dir
                if chroma_persist_dir.is_absolute()
                else PROJECT_ROOT / chroma_persist_dir
            )
        if chroma_collection_name:
            Settings.CHROMA_COLLECTION_NAME = chroma_collection_name

        from app.chroma_store import load_chunks_from_chroma

        chunks = load_chunks_from_chroma()
        return chunks_to_index(chunks, source="chroma")
    except Exception as exc:
        warn(f"failed to load chunks from ChromaDB: {exc}")
        return ChunkIndex({}, {}, "none")


def load_chunk_index(
    chunks_file: Path | None,
    index_source: str,
    chroma_persist_dir: Path | None,
    chroma_collection_name: str | None,
) -> ChunkIndex:
    """按参数加载评测索引；默认优先使用 ChromaDB，旧 chunks.json 仅作回退。"""
    if index_source in {"auto", "chroma"}:
        index = load_chroma_index(chroma_persist_dir, chroma_collection_name)
        if index.chunk_text_map or index_source == "chroma":
            return index

    if index_source in {"auto", "chunks-file"}:
        try:
            index = load_chunks_file_index(chunks_file)
            if index.chunk_text_map or index_source == "chunks-file":
                return index
        except Exception as exc:
            warn(f"failed to load chunks file: {exc}")

    return ChunkIndex({}, {}, "none")


def evidence_texts(question: Dict[str, Any]) -> List[str]:
    """提取 gold_evidence 中用于检索评估的证据片段。"""
    out = []
    for item in question.get("gold_evidence", []) or []:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = item
        text = "" if text is None else str(text).strip()
        if text:
            out.append(text)
    return out


def evidence_matches_text(evidence: str, text: str) -> bool:
    """判断当前 chunk 是否包含 gold evidence 片段。"""
    ne = normalize_text(evidence)
    nt = normalize_text(text)
    if not ne or not nt:
        return False
    if ne in nt:
        return True
    # 知识库重切片后，证据片段可能被标题、空白或轻微格式差异打断。
    return len(ne) >= 12 and is_subsequence(ne, nt)


def find_gold_chunks_for_evidence(
    evidence: str,
    question: Dict[str, Any],
    chunk_text_map: Dict[int, str],
    chunk_doc_map: Dict[int, str],
) -> List[int]:
    """在当前索引中定位能承载某条 gold evidence 的 chunk_id。"""
    gold_docs = [str(x) for x in question.get("gold_doc_ids", [])]

    def scan(restrict_doc: bool) -> List[int]:
        matches = []
        for chunk_id, text in chunk_text_map.items():
            if restrict_doc and gold_docs and not doc_id_matches(chunk_doc_map.get(chunk_id, ""), gold_docs):
                continue
            if evidence_matches_text(evidence, text):
                matches.append(chunk_id)
        return sorted(matches)

    # 先在原 gold_doc_ids 中找，防止新增汇编类文档造成重复证据；找不到再全库兜底。
    matches = scan(restrict_doc=True)
    if not matches and gold_docs:
        matches = scan(restrict_doc=False)
    return matches


def resolve_gold_chunk_ids(
    question: Dict[str, Any],
    chunk_text_map: Dict[int, str],
    chunk_doc_map: Dict[int, str],
) -> List[int]:
    """根据当前 ChromaDB chunk 文本重定位旧评测集中的 gold evidence。"""
    resolved = []
    if chunk_text_map:
        for evidence in evidence_texts(question):
            resolved.extend(find_gold_chunks_for_evidence(evidence, question, chunk_text_map, chunk_doc_map))

    if resolved:
        return sorted(set(resolved))

    return [int(x) for x in question.get("gold_chunk_ids", [])]


def score_key_points(answer: str, question: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    """计算答案对标准关键点的加权召回率。"""
    points = question.get("gold_key_points", [])
    if not points:
        return 1.0, [], []
    total = 0.0
    hit = 0.0
    missing = []
    matched = []
    for p in points:
        weight = float(p.get("weight", 1.0))
        total += weight
        if match_point(answer, p):
            hit += weight
            matched.append(p.get("point", ""))
        else:
            missing.append(p.get("point", ""))
    return (hit / total if total else 1.0), missing, matched


def has_negation_near(answer: str, start: int, end: int, window: int = 36) -> bool:
    """判断命中片段附近是否存在否定语境，避免把拒答当幻觉。"""
    left = max(0, start - window)
    right = min(len(answer), end + window)
    ctx = answer[left:right]
    return any(cue in ctx for cue in NEGATION_CUES)


def regex_forbidden_hit(answer: str, point: str) -> bool:
    """用精确正则识别常见禁止信息，降低误报。"""
    # 只有真的给出百分比数值，才算“百分比数值”幻觉；“没有提供百分比”不算。
    if "百分比" in point:
        return bool(re.search(r"\d+(?:\.\d+)?\s*%|百分之[一二三四五六七八九十百千万零点\d]+", answer))

    # 只有真的给出日期，才算发布日期幻觉；“没有发布日期”不算。
    if "发布日期" in point:
        return bool(re.search(r"\d{4}[-年/]\d{1,2}(?:[-月/]\d{1,2}日?)?", answer))

    # 只有给出具体 star 数，才算；“无法确认 star 数量”不算。
    if "star" in point.lower() or "星标" in point:
        return bool(re.search(r"\d+(?:\.\d+)?\s*(?:k|m|万)?\s*(?:star|stars|星标)", answer, flags=re.I))

    # 只有给出具体源码路径/行号，才算；“没有行号”不算。
    if "源码" in point or "行号" in point:
        return bool(re.search(r"packages/|runtime-core|apiCreateApp|第?\d+\s*行", answer, flags=re.I))

    return False


def alias_forbidden_hit(answer: str, alias: str) -> bool:
    """结合否定语境和英文边界判断 forbidden alias 是否命中。"""
    if not alias:
        return False
    # 单字符 forbidden 太容易误报，比如 k、行，直接跳过。
    if len(alias.strip()) <= 1:
        return False

    # 英文内部实现词用单词边界，避免 Set 命中 setter。
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", alias):
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])"
        for m in re.finditer(pattern, answer, flags=re.I):
            if not has_negation_near(answer, m.start(), m.end()):
                return True
        return False

    # 中文/混合短语：找到原文位置后看附近是否是否定语境。
    for m in re.finditer(re.escape(alias), answer, flags=re.I):
        if not has_negation_near(answer, m.start(), m.end()):
            return True
    return False


def is_regex_only_forbidden(point: str) -> bool:
    """这些 forbidden 类型只能按精确正则判定，不能再用 alias 模糊匹配，否则会误报。"""
    p = point.lower()
    return (
        "百分比" in p
        or "发布日期" in p
        or "star" in p
        or "星标" in p
        or "源码" in p
        or "行号" in p
    )


def score_forbidden(answer: str, question: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """检查答案是否命中不应出现的禁止信息。"""
    hits = []
    for p in question.get("forbidden_points", []):
        if isinstance(p, dict):
            point = str(p.get("point", ""))
            aliases = p.get("aliases") or [point]
        else:
            point = str(p)
            aliases = [point]

        # 数值/日期/star/源码等类型必须用精确规则，不能因为答案里出现“star/发布日期”等词就判幻觉。
        if regex_forbidden_hit(answer, point):
            hits.append(point)
            continue
        if is_regex_only_forbidden(point):
            continue

        if any(alias_forbidden_hit(answer, str(a)) for a in aliases):
            hits.append(point)
    return bool(hits), hits


def score_retrieval(
    pred: Dict[str, Any],
    question: Dict[str, Any],
    chunk_doc_map: Dict[int, str],
    chunk_text_map: Dict[int, str],
    k: int,
) -> Dict[str, Any]:
    """基于预测召回结果计算 top-k 检索命中和召回指标。"""
    retrieved = parse_chunk_ids(pred)
    retrieved_at_k = retrieved[:k] if k and k > 0 else retrieved
    retrieved_set_at_k = set(retrieved_at_k)

    evidence_items = evidence_texts(question)
    matched_evidence = []
    missing_evidence = []
    matched_gold_chunks = []
    match_mode = "chunk_id"

    if evidence_items and chunk_text_map:
        match_mode = "gold_evidence"
        for evidence in evidence_items:
            matched_for_evidence = [
                chunk_id
                for chunk_id in retrieved_at_k
                if evidence_matches_text(evidence, chunk_text_map.get(chunk_id, ""))
            ]
            if matched_for_evidence:
                matched_evidence.append(evidence)
                matched_gold_chunks.extend(matched_for_evidence)
            else:
                missing_evidence.append(evidence)

        matched_gold_chunks = sorted(set(matched_gold_chunks))
        gold_chunks = resolve_gold_chunk_ids(question, chunk_text_map, chunk_doc_map)
        gold_set = set(gold_chunks)
        missing_gold_chunks = sorted(gold_set - set(matched_gold_chunks)) if gold_set else []
        chunk_hit = 1.0 if matched_evidence else 0.0
        gold_chunk_recall = len(matched_evidence) / len(evidence_items)
        all_gold_hit = 1.0 if len(matched_evidence) == len(evidence_items) else 0.0
    else:
        gold_chunks = [int(x) for x in question.get("gold_chunk_ids", [])]
        gold_set = set(gold_chunks)
        matched_gold_chunks = sorted(gold_set & retrieved_set_at_k)
        missing_gold_chunks = sorted(gold_set - retrieved_set_at_k)

        if gold_set:
            chunk_hit = 1.0 if matched_gold_chunks else 0.0
            gold_chunk_recall = len(matched_gold_chunks) / len(gold_set)
            all_gold_hit = 1.0 if gold_set.issubset(retrieved_set_at_k) else 0.0
        else:
            # 如果评测题没有 gold evidence / gold_chunk_ids，就不惩罚检索。
            chunk_hit = 1.0
            gold_chunk_recall = 1.0
            all_gold_hit = 1.0

    retrieved_docs = parse_doc_ids(pred)
    if not retrieved_docs and chunk_doc_map:
        retrieved_docs = [chunk_doc_map.get(cid, "") for cid in retrieved_at_k if cid in chunk_doc_map]
    gold_docs = set(str(x) for x in question.get("gold_doc_ids", []))
    doc_hit = 0.0
    if gold_docs:
        doc_hit = 1.0 if any(doc_id_matches(doc, gold_docs) for doc in retrieved_docs) else 0.0
    if not gold_docs:
        doc_hit = 1.0

    return {
        "retrieved_chunk_ids": retrieved,
        "retrieved_chunk_ids_at_k": retrieved_at_k,
        "retrieved_doc_ids": retrieved_docs,
        "chunk_hit_at_k": chunk_hit,

        # 旧字段保留，兼容已有输出；它和 gold_chunk_recall_at_k 含义一致。
        "chunk_recall_at_k": gold_chunk_recall,

        # 新增两个更有解释力的指标。
        "gold_chunk_recall_at_k": gold_chunk_recall,
        "all_gold_hit_at_k": all_gold_hit,

        "doc_hit_at_k": doc_hit,
        "hit_chunk_ids": matched_gold_chunks,
        "matched_gold_chunk_ids": matched_gold_chunks,
        "missing_gold_chunk_ids": missing_gold_chunks,
        "resolved_gold_chunk_ids": gold_chunks,
        "retrieval_match_mode": match_mode,
        "gold_evidence_count": len(evidence_items),
        "matched_gold_evidence_count": len(matched_evidence),
        "missing_gold_evidence": missing_evidence,
    }


def behavior_score(answer: str, expected: str, key_recall: float, forbidden_hit: bool) -> Tuple[float, str]:
    """按期望行为、关键点覆盖和禁用信息给答案行为打分。"""
    refused = is_refusal(answer)
    if not answer.strip():
        return 0.0, "empty_answer"

    if expected == "abstain":
        if refused and not forbidden_hit:
            return 1.0, "correct_abstain"
        if refused and forbidden_hit:
            return 0.5, "abstain_with_forbidden_detail"
        return 0.0, "should_abstain_but_answered"

    if expected == "partial_answer":
        # A good partial answer should answer the known part and explicitly refuse/limit the unknown part.
        if refused and key_recall >= 0.45 and not forbidden_hit:
            return 1.0, "correct_partial"
        if refused and key_recall >= 0.25:
            return 0.7, "partial_but_incomplete"
        if not refused and key_recall >= 0.60:
            return 0.45, "answered_known_part_but_no_refusal"
        if refused:
            return 0.35, "refused_too_much"
        return 0.0, "poor_partial"

    # expected == answer
    # 注意：trap 题里“文档没有说明 X / 不能确认 X”经常是正确答案的一部分，
    # 不能把所有 refusal pattern 都视为拒答。关键点覆盖足够高时，视为“有依据的限制性回答”。
    if refused:
        if key_recall >= 0.70 and not forbidden_hit:
            return 1.0, "answered_with_supported_limitation"
        if key_recall >= 0.45 and not forbidden_hit:
            return 0.6, "partly_answered_with_limitation"
        return 0.0, "should_answer_but_refused"
    if forbidden_hit:
        return 0.5, "answered_with_forbidden_detail"
    return 1.0, "answered"


def final_score(expected: str, behavior: float, key_recall: float, retrieval_hit: float, forbidden_hit: bool) -> float:
    """综合行为、关键点、检索和禁用命中计算最终分数。"""
    clean = 0.0 if forbidden_hit else 1.0
    if expected == "abstain":
        return 0.70 * behavior + 0.20 * key_recall + 0.10 * clean
    if expected == "partial_answer":
        return 0.25 * behavior + 0.40 * key_recall + 0.20 * retrieval_hit + 0.15 * clean
    return 0.20 * behavior + 0.50 * key_recall + 0.20 * retrieval_hit + 0.10 * clean


def evaluate(eval_items: List[Dict[str, Any]], preds: Dict[str, Dict[str, Any]], chunk_index: ChunkIndex, k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """对所有评测题逐条判分并生成汇总。"""
    rows = []
    for q in eval_items:
        qid = str(q["qid"])
        pred = preds.get(qid, {})
        answer = get_answer(pred)
        key_recall, missing, matched = score_key_points(answer, q)
        forbidden_hit, forbidden_items = score_forbidden(answer, q)
        retrieval = score_retrieval(pred, q, chunk_index.chunk_doc_map, chunk_index.chunk_text_map, k)
        behavior, behavior_label = behavior_score(answer, q.get("expected_behavior", "answer"), key_recall, forbidden_hit)
        # final_score 使用 gold_chunk_recall_at_k，而不是 loose 的 chunk_hit_at_k。
        # 否则只命中一个证据 chunk 的 cross-doc 问题会被高估。
        fscore = final_score(q.get("expected_behavior", "answer"), behavior, key_recall, retrieval["gold_chunk_recall_at_k"], forbidden_hit)

        rows.append({
            "qid": qid,
            "category": q.get("category", ""),
            "skill_type": q.get("skill_type", ""),
            "difficulty": q.get("difficulty", ""),
            "expected_behavior": q.get("expected_behavior", ""),
            "has_prediction": bool(pred),
            "behavior_score": round(behavior, 4),
            "behavior_label": behavior_label,
            "key_point_recall": round(key_recall, 4),
            "retrieval_chunk_hit_at_k": round(retrieval["chunk_hit_at_k"], 4),
            # 旧字段保留：等价于 gold_chunk_recall_at_k。
            "retrieval_chunk_recall_at_k": round(retrieval["chunk_recall_at_k"], 4),
            # 新增：标准证据 chunk 召回比例。
            "retrieval_gold_chunk_recall_at_k": round(retrieval["gold_chunk_recall_at_k"], 4),
            # 新增：是否召回全部标准证据 chunk。
            "retrieval_all_gold_hit_at_k": round(retrieval["all_gold_hit_at_k"], 4),
            "retrieval_doc_hit_at_k": round(retrieval["doc_hit_at_k"], 4),
            "retrieval_match_mode": retrieval["retrieval_match_mode"],
            "forbidden_hit": forbidden_hit,
            "final_score": round(fscore, 4),
            "missing_key_points": " | ".join(missing),
            "matched_key_points": " | ".join(matched),
            "forbidden_points_hit": " | ".join(forbidden_items),
            "gold_chunk_ids": json.dumps(q.get("gold_chunk_ids", []), ensure_ascii=False),
            "resolved_gold_chunk_ids": json.dumps(retrieval["resolved_gold_chunk_ids"], ensure_ascii=False),
            "hit_chunk_ids": json.dumps(retrieval["hit_chunk_ids"], ensure_ascii=False),
            "matched_gold_chunk_ids": json.dumps(retrieval["matched_gold_chunk_ids"], ensure_ascii=False),
            "missing_gold_chunk_ids": json.dumps(retrieval["missing_gold_chunk_ids"], ensure_ascii=False),
            "gold_evidence_count": retrieval["gold_evidence_count"],
            "matched_gold_evidence_count": retrieval["matched_gold_evidence_count"],
            "missing_gold_evidence": " | ".join(retrieval["missing_gold_evidence"]),
            "retrieved_chunk_ids": json.dumps(retrieval["retrieved_chunk_ids"], ensure_ascii=False),
            "question": q.get("question", ""),
            "gold_answer": q.get("gold_answer", ""),
            "answer": answer,
        })
    summary = summarize(rows)
    return rows, summary


def mean(values: List[float]) -> float:
    """安全计算均值，空列表返回 0。"""
    return sum(values) / len(values) if values else 0.0


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按整体、类别、难度和期望行为汇总评测指标。"""
    groups = {
        "overall": rows,
    }
    for key in ("category", "difficulty", "expected_behavior"):
        by = defaultdict(list)
        for r in rows:
            by[r[key]].append(r)
        for name, vals in by.items():
            groups[f"{key}:{name}"] = vals

    out = {}
    for name, vals in groups.items():
        out[name] = {
            "num_questions": len(vals),
            "prediction_coverage": round(mean([1.0 if r["has_prediction"] else 0.0 for r in vals]), 4),
            "avg_final_score": round(mean([float(r["final_score"]) for r in vals]), 4),
            "avg_behavior_score": round(mean([float(r["behavior_score"]) for r in vals]), 4),
            "avg_key_point_recall": round(mean([float(r["key_point_recall"]) for r in vals]), 4),
            "chunk_hit_at_k": round(mean([float(r["retrieval_chunk_hit_at_k"]) for r in vals]), 4),
            "gold_chunk_recall_at_k": round(mean([float(r["retrieval_gold_chunk_recall_at_k"]) for r in vals]), 4),
            "all_gold_hit_at_k": round(mean([float(r["retrieval_all_gold_hit_at_k"]) for r in vals]), 4),
            "doc_hit_at_k": round(mean([float(r["retrieval_doc_hit_at_k"]) for r in vals]), 4),
            "forbidden_rate": round(mean([1.0 if r["forbidden_hit"] else 0.0 for r in vals]), 4),
        }
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """将逐题评测结果写入 CSV。"""
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    """将汇总指标写入 CSV。"""
    rows = []
    for group, metrics in summary.items():
        rows.append({"group": group, **metrics})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    """解析参数、运行语义评测并输出报告文件。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", required=True, type=Path)
    ap.add_argument("--pred-file", required=True, type=Path)
    ap.add_argument(
        "--chunks-file",
        type=Path,
        default=None,
        help="Legacy chunks.json fallback. Auto mode prefers ChromaDB because manifest no longer stores chunks.",
    )
    ap.add_argument(
        "--index-source",
        choices=["auto", "chroma", "chunks-file", "none"],
        default="auto",
        help="Where to load current chunk text/metadata for retrieval scoring. Default: auto (ChromaDB first).",
    )
    ap.add_argument(
        "--chroma-persist-dir",
        type=Path,
        default=None,
        help="Override Settings.CHROMA_PERSIST_DIR for evaluation.",
    )
    ap.add_argument(
        "--chroma-collection-name",
        default=None,
        help="Override Settings.CHROMA_COLLECTION_NAME for evaluation.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_report"))
    ap.add_argument("--k", type=int, default=5, help="Use top-k retrieved chunks for retrieval hit metrics")
    args = ap.parse_args()

    eval_items = load_eval(args.eval_file)
    preds = load_predictions(args.pred_file)
    chunk_index = load_chunk_index(
        chunks_file=args.chunks_file,
        index_source=args.index_source,
        chroma_persist_dir=args.chroma_persist_dir,
        chroma_collection_name=args.chroma_collection_name,
    )
    if args.index_source != "none" and not chunk_index.chunk_text_map:
        warn("no chunk index loaded; doc-hit and evidence-based retrieval metrics may fall back or be unavailable")

    rows, summary = evaluate(eval_items, preds, chunk_index, args.k)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.out_dir / "per_question_report.csv", rows)
    write_summary_csv(args.out_dir / "summary_report.csv", summary)
    (args.out_dir / "summary_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = summary["overall"]
    print("=== RAG Evaluation Summary ===")
    print(f"Chunk index source: {chunk_index.source} ({len(chunk_index.chunk_text_map)} chunks)")
    print(f"Questions: {overall['num_questions']}")
    print(f"Prediction coverage: {overall['prediction_coverage']:.2%}")
    print(f"Average final score: {overall['avg_final_score']:.2%}")
    print(f"Average behavior score: {overall['avg_behavior_score']:.2%}")
    print(f"Average key-point recall: {overall['avg_key_point_recall']:.2%}")
    print(f"Chunk hit@{args.k}: {overall['chunk_hit_at_k']:.2%}")
    print(f"Gold chunk recall@{args.k}: {overall['gold_chunk_recall_at_k']:.2%}")
    print(f"All gold hit@{args.k}: {overall['all_gold_hit_at_k']:.2%}")
    print(f"Doc hit@{args.k}: {overall['doc_hit_at_k']:.2%}")
    print(f"Forbidden/hallucination rate: {overall['forbidden_rate']:.2%}")
    print(f"\nReports written to: {args.out_dir}")


if __name__ == "__main__":
    main()

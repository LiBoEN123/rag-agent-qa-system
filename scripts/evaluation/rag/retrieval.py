"""RAG 评测的索引加载、证据对齐与检索指标计算。

用途：把当前 ChromaDB 或兼容 chunks 文件映射成统一索引，并核对 top-k 召回证据。
输入：索引位置、标准文档/chunk/evidence、预测召回标识和 top-k。
输出：``ChunkIndex``、证据对应的当前 chunk_id，以及 Gold Chunk Recall。
边界：只读取现有索引并计算评测指标，不修改生产检索算法、向量库或评测标准答案。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from scripts.evaluation.rag.common import (
    PROJECT_ROOT,
    ChunkIndex,
    is_subsequence,
    normalize_text,
    parse_chunk_ids_field,
    warn,
)


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
        from app.config import get_settings

        settings = get_settings()
        updates = {}
        if chroma_persist_dir is not None:
            updates["chroma_persist_dir"] = (
                chroma_persist_dir
                if chroma_persist_dir.is_absolute()
                else PROJECT_ROOT / chroma_persist_dir
            )
        if chroma_collection_name:
            updates["chroma_collection_name"] = chroma_collection_name
        settings = settings.model_copy(update=updates)

        from app.rag.chroma import load_index_from_chroma

        chunks, _doc_embeddings = load_index_from_chroma(settings)
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
        """扫描匹配证据的 chunk，可选择先限制在标准文档范围内。"""
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


def score_evidence_chunks(
    chunk_ids: List[int],
    scored_gold_chunk_ids: List[int],
) -> Dict[str, Any]:
    """按已对齐的 Gold chunk IDs 计算严格 Recall 与命中/缺失明细。"""
    gold_set = set(scored_gold_chunk_ids)
    chunk_set = set(chunk_ids)
    return {
        "gold_chunk_recall": len(gold_set & chunk_set) / len(gold_set) if gold_set else 1.0,
        "hit_chunk_ids": sorted(gold_set & chunk_set),
        "missing_gold_chunk_ids": sorted(gold_set - chunk_set),
    }


def score_retrieval(
    pred: Dict[str, Any],
    question: Dict[str, Any],
    chunk_doc_map: Dict[int, str],
    chunk_text_map: Dict[int, str],
    k: int,
) -> Dict[str, Any]:
    """计算有序去重观察的 Recall@K；不修改 Agent 工具的 top_k。"""
    retrieved = parse_chunk_ids_field(pred, "retrieved_chunk_ids")
    retrieved_at_k = retrieved[:k] if k and k > 0 else retrieved
    gold = resolve_gold_chunk_ids(question, chunk_text_map, chunk_doc_map)
    scored = score_evidence_chunks(retrieved_at_k, gold)
    return {
        "retrieved_chunk_ids": retrieved,
        "retrieved_chunk_ids_at_k": retrieved_at_k,
        "gold_chunk_recall_at_k": scored["gold_chunk_recall"],
        "scored_gold_chunk_ids": gold,
        "hit_chunk_ids": scored["hit_chunk_ids"],
        "missing_gold_chunk_ids": scored["missing_gold_chunk_ids"],
    }


def collect_retrieved_evidence(
    retrieval: Dict[str, Any],
    chunk_index: ChunkIndex,
    max_items: int = 5,
    max_chars_per_item: int = 800,
) -> List[str]:
    """收集 top-k 检索片段，作为 judge 的可选背景信息。"""
    evidence = []
    for chunk_id in retrieval.get("retrieved_chunk_ids_at_k", [])[:max_items]:
        text = chunk_index.chunk_text_map.get(chunk_id, "")
        if not text:
            continue
        doc = chunk_index.chunk_doc_map.get(chunk_id, "")
        snippet = text.strip()
        if len(snippet) > max_chars_per_item:
            snippet = snippet[:max_chars_per_item] + "..."
        evidence.append(f"chunk_id={chunk_id}; doc={doc}; text={snippet}")
    return evidence

import json
import re
from typing import Any, Dict, List

import jieba
import numpy as np
import requests

from app.config import Settings
from app.chroma_store import ChromaUnavailableError, query_chroma
from app.prompts import DECOMPOSE_PROMPT
from app.untils import get_logger

logger = get_logger('chat_service')


AMBIGUOUS_WORDS = [
    "这个", "那个", "它", "这条", "这项", "这份",
    "上面", "前面", "后面", "这里", "那里"
]

VECTOR_RELAX_RATIO = 0.70
RERANK_STRONG_THRESHOLD = 0.45
RERANK_VERY_STRONG_THRESHOLD = 0.55
HYBRID_USABLE_THRESHOLD = 0.20
HYBRID_STRONG_THRESHOLD = 0.35


# =========================
# 通用文本与分数工具
# =========================
def tokenize(text: str):
    """使用 jieba 对文本分词，供 BM25 检索使用。"""
    return list(jieba.cut(text))


def minmax_score(x: np.ndarray):
    """将一组分数归一化到 0 到 1 区间。"""
    x = x.astype(np.float32)
    if np.allclose(x.min(), x.max()):
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def clamp01(value: float) -> float:
    """将数值限制在 0 到 1 区间。"""
    value = float(value)
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def safe_float(value, default: float = 0.0) -> float:
    """将分数字段转换为有限浮点数，脏数据统一按默认值处理。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(score):
        return default
    return score


def make_snippet(text: str, max_len: int = 300) -> str:
    """去除文本空白并截取指定长度的预览片段。"""
    text = "".join((text or "").split())
    return text[:max_len]


# =========================
# 重排
# =========================
def rerank(rerank_model, candidates: list[dict], question: str=""):

    """使用交叉编码器对候选 chunk 按相关性重排。"""
    if not candidates:
        return []

    pairs = []
    for item in candidates:
        pairs.append((question, item["text"]))

    scores = rerank_model.predict(pairs)

    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)

    candidates = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return candidates


# =========================
# 问题拆分
# =========================
def default_decompose_result(reason: str = "query decomposition skipped") -> dict:
    """问题拆分关闭或失败时使用的统一兜底结构。"""
    return {
        "need_decompose": False,
        "reason": reason,
        "sub_queries": []
    }


def should_decompose_query(question: str) -> tuple[bool, str]:
    """用轻量规则判断是否值得调用问题拆分模型。"""
    q = (question or "").strip()
    if not q:
        return False, "empty query"

    normalized = q.replace("？", "?")
    question_mark_count = normalized.count("?")
    if question_mark_count >= 2:
        return True, "multiple question marks"

    multi_intent_keywords = [
        "分别",
        "同时",
        "以及",
        "对比",
        "区别",
        "新旧",
        "变化",
        "调整",
        "修订",
        "流程和条件",
        "条件和材料",
        "入口和时间",
    ]
    for keyword in multi_intent_keywords:
        if keyword in q:
            return True, f"multi-intent keyword: {keyword}"

    years = re.findall(r"(?:19|20)\d{2}", q)
    if len(set(years)) >= 2:
        return True, "compares multiple years"

    version_or_policy_pattern = (
        r"(新版|旧版|新政策|旧政策|新规|旧规|新办法|旧办法|"
        r"现行|原规定|以前|之前|修订前|修订后|调整前|调整后)"
        r".{0,20}"
        r"(对比|区别|变化|调整|修订|相比|不同|差异)"
        r"|"
        r"(对比|区别|变化|调整|修订|相比|不同|差异)"
        r".{0,20}"
        r"(新版|旧版|新政策|旧政策|新规|旧规|新办法|旧办法|"
        r"现行|原规定|以前|之前|修订前|修订后|调整前|调整后)"
    )
    if re.search(version_or_policy_pattern, q):
        return True, "compares policy versions"

    intent_words = (
        "怎么|如何|什么|哪些|是否|能否|有没有|多少|哪里|入口|时间|"
        "条件|材料|流程|要求|办法"
    )
    multi_clause_pattern = (
        rf"({intent_words}).{{0,30}}(，|,|；|;|、|和|及|并且|同时).{{0,30}}"
        rf"({intent_words})"
    )
    if re.search(multi_clause_pattern, q):
        return True, "multiple explicit intents"

    if len(q) <= 30:
        return False, "short single-intent query"

    return False, "single-intent query"


def parse_decompose_bool(value) -> bool:
    """将拆分模型返回的布尔样值转换为 bool。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def compact_text(value) -> str:
    """将模型字段规整为单行文本。"""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_decompose_result(result: dict) -> dict:
    """校验并规整问题拆分模型的返回值，避免下游分支处理脏数据。"""
    if not isinstance(result, dict):
        raise ValueError(f"拆分模型返回值必须是 dict，实际为 {type(result)}")

    need_decompose = parse_decompose_bool(result.get("need_decompose", False))
    reason = compact_text(result.get("reason", ""))

    if not need_decompose:
        return default_decompose_result(reason or "decompose model decided single query")

    raw_sub_queries = result.get("sub_queries", [])
    if not isinstance(raw_sub_queries, list):
        raise ValueError("拆分模型返回的 sub_queries 必须是 list")

    sub_queries = []
    for item in raw_sub_queries[:5]:
        if not isinstance(item, dict):
            continue

        sub_question = compact_text(item.get("sub_question", ""))
        search_query = compact_text(item.get("search_query", ""))

        if not sub_question or not search_query:
            continue

        sub_queries.append({
            "id": item.get("id", len(sub_queries) + 1),
            "sub_question": sub_question,
            "search_query": search_query,
            "must_answer": parse_decompose_bool(item.get("must_answer", True))
        })

    if not sub_queries:
        return default_decompose_result("decompose model returned no valid sub_queries")

    return {
        "need_decompose": True,
        "reason": reason or "decompose model decided multi-query retrieval",
        "sub_queries": sub_queries
    }


def compact_error_preview(text: str, max_len: int = 500) -> str:
    """将模型原始输出压成单行预览，避免异常日志被换行打断。"""
    preview = re.sub(r"\s+", " ", (text or "")).strip()
    if len(preview) > max_len:
        return preview[:max_len] + "..."
    return preview


def strip_json_code_fence(content: str) -> str:
    """去掉模型可能返回的 Markdown JSON 代码块。"""
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def extract_first_json_object(content: str) -> str:
    """从模型输出中截取第一个完整 JSON 对象字符串。"""
    start = content.find("{")
    if start < 0:
        raise ValueError("拆分模型未返回 JSON 对象")

    in_string = False
    escaped = False
    depth = 0

    for index in range(start, len(content)):
        char = content[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start:index + 1]

    raise ValueError("拆分模型返回的 JSON 对象不完整，可能被 max_tokens 截断")


def parse_decompose_llm_json(content: str) -> dict:
    """解析拆分模型输出，兼容代码块和 JSON 前后夹杂文本。"""
    content = strip_json_code_fence(content)

    try:
        result = json.loads(content, strict=False)
    except json.JSONDecodeError:
        json_object = extract_first_json_object(content)
        try:
            result = json.loads(json_object, strict=False)
        except json.JSONDecodeError as e:
            preview = compact_error_preview(content)
            raise ValueError(f"LLM 返回的不是合法 JSON: {preview}") from e

    return normalize_decompose_result(result)


def decompose_query_with_llm(question: str):
    """调用拆分模型，把复杂问题转换为可单独检索的子问题。"""
    model_name = Settings.DECOMPOSE_MODEL_NAME
    api_url = Settings.DECOMPOSE_API_URL

    pay_load = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": DECOMPOSE_PROMPT},
            {"role": "user", "content": question}
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": Settings.DECOMPOSE_MAX_TOKENS
    }
    headers = {
        "Content-Type": "application/json"
    }
    if Settings.DECOMPOSE_API_KEY:
        headers["Authorization"] = f"Bearer {Settings.DECOMPOSE_API_KEY}"

    response = requests.post(
        api_url,
        headers=headers,
        json=pay_load,
        timeout=Settings.DECOMPOSE_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    return parse_decompose_llm_json(content)


# =========================
# 检索
# =========================
def _make_retrieve_result(
    chunk: dict,
    rank: int,
    bm25_score: float,
    vector_score: float,
    hybrid_score: float,
):
    """将 chunk 和分数组装成统一检索结果结构。"""
    metadata = chunk.get("metadata") or {}
    text = chunk.get("text", "") or ""

    return {
        "reference_id": f"Reference {rank}",
        "file_name": metadata.get("file_name", "unknown"),
        "source_path": metadata.get("source_path", "unknown"),
        "chunk_id": metadata.get("chunk_id", f"chunk_{rank}"),
        "position": metadata.get("position"),
        "text": text,
        "snippet": make_snippet(text),
        "bm25_score": float(bm25_score),
        "vector_score": float(vector_score),
        "hybrid_score": float(hybrid_score),
        "metadata": metadata,
    }


def _retrieve_with_local_embeddings(
    question: str,
    chunks: list[dict],
    bm25,
    doc_embeddings,
    model,
    top_k: int = 5
):
    """使用本地 numpy embedding 矩阵执行旧版混合召回。"""
    if doc_embeddings is None:
        return []

    tokenized_question = tokenize(question)
    bm25_score = bm25.get_scores(tokenized_question)

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    vector_score = doc_embeddings @ question_embedding

    bm25_norm = minmax_score(bm25_score)
    vector_norm = minmax_score(vector_score)
    hybrid_score = 0.4 * bm25_norm + 0.6 * vector_norm

    candidate_k = max(top_k * 5, top_k)
    top_indices = np.argsort(hybrid_score)[::-1][:candidate_k]

    raw_results = []
    for rank, idx in enumerate(top_indices, start=1):
        raw_results.append(
            _make_retrieve_result(
                chunk=chunks[idx],
                rank=rank,
                bm25_score=float(bm25_score[idx]),
                vector_score=float(vector_score[idx]),
                hybrid_score=float(hybrid_score[idx]),
            )
        )

    return raw_results


def _retrieve_with_chroma(
    question: str,
    chunks: list[dict],
    bm25,
    model,
    top_k: int = 5
):
    """使用 ChromaDB 做向量召回，并与 BM25 候选合并。"""
    candidate_k = max(top_k * 5, top_k)

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    chroma_hits = query_chroma(question_embedding, n_results=candidate_k)
    if not chroma_hits:
        return []

    tokenized_question = tokenize(question)
    bm25_scores = bm25.get_scores(tokenized_question)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

    chunk_by_id = {}
    for idx, chunk in enumerate(chunks):
        chunk_id = str((chunk.get("metadata") or {}).get("chunk_id", ""))
        if chunk_id:
            chunk_by_id[chunk_id] = (idx, chunk)

    candidates = {}

    for hit in chroma_hits:
        metadata = hit.get("metadata") or {}
        chunk_id = str(metadata.get("chunk_id", ""))
        idx, chunk = chunk_by_id.get(chunk_id, (None, None))

        if chunk is None:
            chunk = {
                "text": hit.get("text", "") or "",
                "metadata": metadata,
            }

        candidate_key = chunk_id or str(hit.get("record_id", ""))
        candidates[candidate_key] = {
            "chunk": chunk,
            "bm25_score": float(bm25_scores[idx]) if idx is not None else 0.0,
            "vector_score": float(hit.get("vector_score", 0.0)),
        }

    for idx in bm25_top_indices:
        idx = int(idx)
        chunk = chunks[idx]
        metadata = chunk.get("metadata") or {}
        chunk_id = str(metadata.get("chunk_id", idx))

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "chunk": chunk,
                "bm25_score": float(bm25_scores[idx]),
                "vector_score": 0.0,
            }

    if not candidates:
        return []

    candidate_items = list(candidates.values())
    bm25_values = np.array([item["bm25_score"] for item in candidate_items], dtype=np.float32)
    vector_values = np.array([item["vector_score"] for item in candidate_items], dtype=np.float32)

    bm25_norm = minmax_score(bm25_values)
    vector_norm = minmax_score(vector_values)

    for item, bm25_part, vector_part in zip(candidate_items, bm25_norm, vector_norm):
        item["hybrid_score"] = float(0.4 * bm25_part + 0.6 * vector_part)

    candidate_items.sort(key=lambda item: item["hybrid_score"], reverse=True)

    raw_results = []
    for rank, item in enumerate(candidate_items[:candidate_k], start=1):
        raw_results.append(
            _make_retrieve_result(
                chunk=item["chunk"],
                rank=rank,
                bm25_score=item["bm25_score"],
                vector_score=item["vector_score"],
                hybrid_score=item["hybrid_score"],
            )
        )

    return raw_results


def retrieve(
    question: str,
    chunks: list[dict],
    bm25,
    doc_embeddings,
    model,
    top_k: int = 5
):
    """综合 BM25 和向量相似度召回候选 chunk，优先使用 ChromaDB 向量库。"""
    if bm25 is None or model is None:
        return []

    question = (question or "").strip()
    if not question:
        return []

    try:
        chroma_results = _retrieve_with_chroma(
            question=question,
            chunks=chunks,
            bm25=bm25,
            model=model,
            top_k=top_k,
        )
        if chroma_results:
            return chroma_results
    except ChromaUnavailableError as e:
        logger.info(f"[retrieve] ChromaDB unavailable, fallback to local embeddings: {e}")
    except Exception as e:
        logger.warning(f"[retrieve] ChromaDB query failed, fallback to local embeddings: {e}")

    return _retrieve_with_local_embeddings(
        question=question,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=top_k,
    )

def retrieve_with_sub_queries(
    question,
    sub_queries,
    chunks,
    bm25,
    doc_embeddings,
    model,
    rerank_model,
    per_query_top_k=10,
    per_query_keep=2,
    final_top_k=10
):
    """多子问题召回时，保护每个子问题的高质量命中，再做全局补足。"""
    protected_chunks = {}
    all_candidates = {}

    # 原问题整体召回一次，保留全局语义。
    global_candidates = retrieve(
        question=question,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=5
    )

    for item in global_candidates:
        all_candidates[item["chunk_id"]] = item

    # 每个子查询单独召回、单独 rerank。
    for sub in sub_queries:
        sub_question = sub["sub_question"]
        search_query = sub["search_query"]

        candidates = retrieve(
            question=search_query,
            chunks=chunks,
            bm25=bm25,
            doc_embeddings=doc_embeddings,
            model=model,
            top_k=per_query_top_k
        )

        for item in candidates:
            all_candidates[item["chunk_id"]] = item

        reranked = rerank(
            candidates=candidates,
            question=sub_question,
            rerank_model=rerank_model
        )

        # 每个子问题至少保留 topN，避免某个子问题被全局排序完全挤掉。
        for item in reranked[:per_query_keep]:
            item["matched_sub_question"] = sub_question
            item["protected"] = True
            protected_chunks[item["chunk_id"]] = item

    merged_candidates = list(all_candidates.values())

    global_reranked = rerank(
        candidates=merged_candidates,
        question=question,
        rerank_model=rerank_model
    )

    final = []
    used_ids = set()

    for item in protected_chunks.values():
        final.append(item)
        used_ids.add(item["chunk_id"])

    for item in global_reranked:
        if item["chunk_id"] not in used_ids:
            item["protected"] = False
            final.append(item)
            used_ids.add(item["chunk_id"])

        if len(final) >= final_top_k:
            break

    for rank, item in enumerate(final, start=1):
        item["reference_id"] = f"Reference {rank}"

    return final


def search_relevant_chunks(
    question: str,
    chunks: list[dict],
    bm25,
    doc_embeddings,
    model,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """对基础 retrieve 做薄封装，返回相关 chunk 列表。"""
    return retrieve(question, chunks, bm25, doc_embeddings, model, top_k)


# =========================
# 结果判定与澄清
# =========================
def should_refuse(retrieved_chunks, vector_threshold: float = None):
    """根据召回证据强度判断是否应该拒答。

    向量分数是主信号；当向量分数略低于阈值，但 rerank、BM25 或
    hybrid 分数显示出明确证据时，允许进入生成阶段，避免误拒关键词强命中。
    """
    vector_threshold = safe_float(
        vector_threshold
        if vector_threshold is not None
        else Settings.VECTOR_THRESHOLD
    )
    vector_threshold = max(0.0, vector_threshold)

    if not retrieved_chunks:
        return True

    candidates = [
        item
        for item in retrieved_chunks
        if isinstance(item, dict)
    ]
    if not candidates:
        return True

    max_vector = max(
        safe_float(item.get("vector_score"))
        for item in candidates
    )
    if max_vector >= vector_threshold:
        return False

    relaxed_vector_threshold = vector_threshold * VECTOR_RELAX_RATIO

    for item in candidates[:3]:
        vector_score = safe_float(item.get("vector_score"))
        rerank_score = safe_float(item.get("rerank_score"))
        hybrid_score = safe_float(item.get("hybrid_score"))
        bm25_score = safe_float(item.get("bm25_score"))

        has_near_vector_match = vector_score >= relaxed_vector_threshold
        has_strong_rerank = rerank_score >= RERANK_STRONG_THRESHOLD
        has_very_strong_rerank = rerank_score >= RERANK_VERY_STRONG_THRESHOLD
        has_lexical_hit = bm25_score > 0
        has_usable_hybrid = hybrid_score >= HYBRID_USABLE_THRESHOLD
        has_strong_hybrid = hybrid_score >= HYBRID_STRONG_THRESHOLD

        if has_near_vector_match and (
            has_strong_rerank
            or has_strong_hybrid
            or (has_lexical_hit and has_usable_hybrid)
        ):
            return False

        if has_very_strong_rerank and has_strong_hybrid:
            return False

        if has_strong_rerank and has_lexical_hit and has_usable_hybrid:
            return False

        if item.get("protected") and (has_near_vector_match or has_strong_rerank):
            return False

    return True


def is_ambiguous_question(question: str) -> bool:
    """判断问题是否过短或存在明显指代不明。"""
    q = (question or "").strip()
    if not q:
        return True
    if len(q) <= 3:
        return True
    if any(word in q for word in AMBIGUOUS_WORDS) and len(q) < 18:
        return True
    return False


def build_clarification_question(question: str) -> str:
    """根据问题关键词生成澄清追问。"""
    q = (question or "").strip()

    if "退款" in q:
        return "你想问的是退款条件、退款时间窗口，还是退款申请方式？"
    if "报销" in q:
        return "你指的是哪一类费用？比如交通费、住宿费、餐费，还是办公用品？"
    if "流程" in q:
        return "你想查询哪一类流程？比如退款流程、报销流程、请假流程，还是采购流程？"

    return "你的问题还不够具体，请补充对象、场景或范围后再问。"

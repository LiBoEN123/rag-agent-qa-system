from typing import Any, Dict, List
import jieba
import numpy as np

from app.config import Settings


AMBIGUOUS_WORDS = [
    "这个", "那个", "它", "这条", "这项", "这份",
    "上面", "前面", "后面", "这里", "那里"
]


def tokenize(text: str):
    return list(jieba.cut(text))


def minmax_score(x: np.ndarray):
    x = x.astype(np.float32)
    if np.allclose(x.min(), x.max()):
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def normalize_text(text: str):
    return "".join((text or "").split()).lower()


def dedupe_results(items: list[dict], prefix_len: int = 80) -> list[dict]:
    seen = set() # seen 是一个集合，用来记录已经见过的文本特征。
    result = []

    for item in items:
        key = normalize_text(item.get("text", "")[:prefix_len])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)

    return result


def clamp01(value: float) -> float:
    value = float(value)
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def make_snippet(text: str, max_len: int = 300) -> str:
    text = "".join((text or "").split())
    return text[:max_len]

def rerank(rerank_model, candidates: list[dict], question: str=""):

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



def retrieve(
    question: str,
    chunks: list[dict],
    bm25,
    doc_embeddings,
    model,
    rerank_model,
    top_k: int = 5
):
    if bm25 is None or doc_embeddings is None or model is None:
        return []

    question = (question or "").strip()
    if not question:
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
        metadata = chunks[idx]["metadata"]
        raw_results.append({
            "reference_id": f"Reference {rank}",
            "file_name": metadata.get("file_name", "unknown"),
            "source_path": metadata.get("source_path", "unknown"),
            "chunk_id": metadata.get("chunk_id", f"chunk_{idx}"),
            "position": metadata.get("position"),
            "text": chunks[idx]["text"],
            "snippet": make_snippet(chunks[idx]["text"]),
            "bm25_score": float(bm25_score[idx]),
            "vector_score": float(vector_score[idx]),
            "hybrid_score": float(hybrid_score[idx]),
            "metadata": metadata,
        })

    deduped_results = dedupe_results(raw_results, prefix_len=80)
    rerank_chunks = rerank(rerank_model, deduped_results, question)
    final_results = rerank_chunks[:top_k]

    for rank, item in enumerate(final_results, start=1):
        item["reference_id"] = f"Reference {rank}"

    return final_results


def search_relevant_chunks(
    question: str,
    chunks: list[dict],
    bm25,
    doc_embeddings,
    model,
    rerank_model,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    return retrieve(question, chunks, bm25, doc_embeddings, model, rerank_model, top_k)


def should_refuse(
    retrieved_chunks,
    vector_threshold: float = None,
):
    vector_threshold = (
        vector_threshold
        if vector_threshold is not None
        else Settings.VECTOR_THRESHOLD
    )

    if not retrieved_chunks:
        return True

    top1_vector = retrieved_chunks[0].get("vector_score", 0.0)

    if top1_vector < vector_threshold:
        return True

    return False


def is_ambiguous_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return True
    if len(q) <= 3:
        return True
    if any(word in q for word in AMBIGUOUS_WORDS) and len(q) < 18:
        return True
    return False


def build_clarification_question(question: str) -> str:
    q = (question or "").strip()

    if "退款" in q:
        return "你想问的是退款条件、退款时间窗口，还是退款申请方式？"
    if "报销" in q:
        return "你指的是哪一类费用？比如交通费、住宿费、餐费，还是办公用品？"
    if "流程" in q:
        return "你想查询哪一类流程？比如退款流程、报销流程、请假流程，还是采购流程？"

    return "你的问题还不够具体，请补充对象、场景或范围后再问。"
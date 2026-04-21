import json
import re
from pathlib import Path
from typing import Dict, List, Any

import jieba
import numpy as np
import requests
from charset_normalizer import from_bytes
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import Settings
from app.prompts import SYSTEM_PROMPT_A, USER_PROMPT, NO_ANSWER_FALLBACK


# =========================
# 全局状态 / 常量
# =========================
chunks: list[dict] = []
texts: list[str] = []
docs_registry: list[dict] = []

bm25 = None
model = None
doc_embeddings = None

SUPPORTED_SUFFIXES = {".txt", ".md"}

MANIFEST_PATH = Path("data/manifest/index_manifest.json")

RAG_READY = False
RAG_INIT_MESSAGE = "RAG 尚未初始化"
NO_DOCUMENTS_MESSAGE = "当前知识库为空，请先在 data 目录中放入文档后再提问"

AMBIGUOUS_WORDS = [
    "这个", "那个", "它", "这条", "这项", "这份",
    "上面", "前面", "后面", "这里", "那里"
]

STOPWORD = "我无法根据提供的文档确定答案"


# =========================
# 最底层工具函数
# =========================
def read_txt_auto(file_path: str) -> str:
    raw = Path(file_path).read_bytes()
    result = from_bytes(raw).best()
    if result is None:
        raise ValueError(f"无法识别编码: {file_path}")
    return str(result)


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
    seen = set()
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


def make_snippet(text: str, max_len: int = 100) -> str:
    text = "".join((text or "").split())
    return text[:max_len]


# =========================
# 准备阶段：加载
# =========================
def collect_supported_files(data_dir: str):
    root = Path(data_dir)

    if not root.exists():
        return []

    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return files


def load_files(data_dir: str):
    files = collect_supported_files(data_dir)
    if not files:
        return [], []

    raw_docs = []
    docs_registry_local = []

    for idx, file_path in enumerate(files, start=1):
        file_type = file_path.suffix.lower().lstrip(".")
        text = read_txt_auto(str(file_path))

        raw_docs.append({
            "doc_id": f"doc_{idx}",
            "file_name": file_path.name,
            "source_path": str(file_path.resolve()),
            "file_type": file_type,
            "text": text,
        })

        docs_registry_local.append({
            "doc_id": f"doc_{idx}",
            "file_name": file_path.name,
            "source_path": str(file_path.resolve()),
            "file_type": file_type,
            "chunk_count": 0,  # 切分后回填
        })

    return raw_docs, docs_registry_local


# =========================
# 准备阶段：清洗
# =========================
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_markdown_text(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return text.strip()


def clean_documents(raw_docs: list[dict]):
    cleaned_docs = []

    for doc in raw_docs:
        text = doc["text"]

        if doc["file_type"] == "md":
            cleaned_text = clean_markdown_text(text)
        else:
            cleaned_text = clean_text(text)

        cleaned_docs.append({
            **doc,
            "text": cleaned_text
        })

    return cleaned_docs


# =========================
# 准备阶段：切分
# =========================
def split_txt_doc(doc: dict, chunk_size: int = Settings.CHUNK_SIZE, chunk_overlap: int = Settings.CHUNK_OVERLAP):
    text = doc["text"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ""]
    )

    split_docs = splitter.create_documents(
        [text],
        metadatas=[{"file_type": "txt"}]
    )

    result = []
    for idx, item in enumerate(split_docs, start=1):
        metadata = dict(item.metadata)
        metadata.update({
            "file_name": doc["file_name"],
            "source_path": doc["source_path"],
            "file_type": doc["file_type"],
            "doc_id": doc["doc_id"],
            "chunk_id": None,
            "position": idx,
        })
        result.append({
            "text": item.page_content,
            "metadata": metadata
        })

    return result


def split_md_doc(doc: dict, chunk_size: int = Settings.CHUNK_SIZE, chunk_overlap: int = Settings.CHUNK_OVERLAP):
    text = doc["text"]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )
    md_docs = md_splitter.split_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ""]
    )
    split_docs = splitter.split_documents(md_docs)

    result = []
    for idx, item in enumerate(split_docs, start=1):
        title_path = " > ".join(
            [item.metadata[k] for k in ["h1", "h2", "h3"] if k in item.metadata]
        )

        text_with_title = (
            f"[标题路径] {title_path}\n\n{item.page_content}"
            if title_path else item.page_content
        )

        result.append({
            "text": text_with_title,
            "metadata": {
                "source_path": doc["source_path"],
                "file_type": doc["file_type"],
                "file_name": doc["file_name"],
                "doc_id": doc["doc_id"],
                "chunk_id": None,
                "position": idx,
                "title_path": title_path,
                **item.metadata
            }
        })

    return result


def split_doc(doc: dict, chunk_size: int = Settings.CHUNK_SIZE, chunk_overlap: int = Settings.CHUNK_OVERLAP):
    if doc["file_type"] == "txt":
        return split_txt_doc(doc, chunk_size, chunk_overlap)
    if doc["file_type"] == "md":
        return split_md_doc(doc, chunk_size, chunk_overlap)
    raise ValueError(f"不支持的文件类型: {doc['file_type']}")


def split_chunks(cleaned_docs: list[dict], chunk_size: int = Settings.CHUNK_SIZE, chunk_overlap: int = Settings.CHUNK_OVERLAP):
    all_chunks = []
    chunk_count_map = {}

    for doc in cleaned_docs:
        file_chunks = split_doc(doc, chunk_size, chunk_overlap)
        all_chunks.extend(file_chunks)
        chunk_count_map[doc["source_path"]] = len(file_chunks)

    for global_idx, chunk in enumerate(all_chunks, start=1):
        chunk["metadata"]["chunk_id"] = global_idx

    return all_chunks, chunk_count_map


# =========================
# 准备阶段：建索引
# =========================
def build_index(all_chunks: list[dict]):
    global model

    if not all_chunks:
        return {
            "texts": [],
            "bm25": None,
            "doc_embeddings": None,
        }

    local_texts = [chunk["text"] for chunk in all_chunks]
    tokenized_chunks = [tokenize(text) for text in local_texts]

    if model is None:
        model = SentenceTransformer(Settings.EMBEDDING_MODEL_NAME)

    local_doc_embeddings = model.encode(
        local_texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    local_bm25 = BM25Okapi(tokenized_chunks)

    return {
        "texts": local_texts,
        "bm25": local_bm25,
        "doc_embeddings": local_doc_embeddings,
    }


def rebuild_index(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    global chunks, texts, docs_registry, bm25, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE

    # 1. 清空状态
    chunks = []
    texts = []
    docs_registry = []
    bm25 = None
    doc_embeddings = None
    RAG_READY = False
    RAG_INIT_MESSAGE = "RAG 尚未初始化"

    # 2. 加载文件
    raw_docs, docs_registry_local = load_files(data_dir)
    if not raw_docs:
        docs_registry = []
        RAG_INIT_MESSAGE = NO_DOCUMENTS_MESSAGE
        return

    # 3. 清洗文档
    cleaned_docs = clean_documents(raw_docs)

    # 4. 切分 chunk
    all_chunks, chunk_count_map = split_chunks(
        cleaned_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 5. 回填每个文档的 chunk_count
    for doc in docs_registry_local:
        doc["chunk_count"] = chunk_count_map.get(doc["source_path"], 0)

    # 6. 建索引
    index_bundle = build_index(all_chunks)

    # 7. 写回全局状态
    chunks = all_chunks
    texts = index_bundle["texts"]
    bm25 = index_bundle["bm25"]
    doc_embeddings = index_bundle["doc_embeddings"]
    docs_registry = docs_registry_local

    # 8. 更新状态
    if not chunks:
        RAG_INIT_MESSAGE = "文档已加载，但切分后没有可用 chunk"
        return

    RAG_READY = True
    RAG_INIT_MESSAGE = "OK"


def init_rag(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    # 兼容旧调用
    rebuild_index(data_dir, chunk_size, chunk_overlap)


# =========================
# 状态函数
# =========================
def get_rag_status():
    return {
        "ready": RAG_READY,
        "message": RAG_INIT_MESSAGE,
        "doc_count": len(docs_registry),
        "chunk_count": len(chunks)
    }


# =========================
# 查询阶段：检索
# =========================
def retrieve(question: str, top_k: int = 3):
    if bm25 is None or doc_embeddings is None:
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

    candidate_k = max(top_k * 3, top_k)
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
    final_results = deduped_results[:top_k]

    for rank, item in enumerate(final_results, start=1):
        item["reference_id"] = f"Reference {rank}"

    return final_results


def search_relevant_chunks(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    return retrieve(question, top_k)


# =========================
# 查询阶段：判断
# =========================
def should_refuse(retrieved_chunks, vector_threshold: float = None, margin_threshold: float = None):
    vector_threshold = (
        vector_threshold if vector_threshold is not None else Settings.VECTOR_THRESHOLD
    )
    margin_threshold = (
        margin_threshold if margin_threshold is not None else Settings.MARGIN_THRESHOLD
    )

    if not retrieved_chunks:
        return True

    top1_vector = retrieved_chunks[0].get("vector_score", 0.0)
    top1_hybrid = retrieved_chunks[0].get("hybrid_score", 0.0)
    top2_hybrid = retrieved_chunks[1].get("hybrid_score", 0.0) if len(retrieved_chunks) > 1 else 0.0
    margin = top1_hybrid - top2_hybrid

    if top1_vector < vector_threshold:
        return True

    if margin < margin_threshold:
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


def estimate_confidence(retrieved_chunks: List[dict], final_references: List[dict], status: str) -> float:
    if status != "OK" or not retrieved_chunks:
        return 0.0

    top1_vector = clamp01(retrieved_chunks[0].get("vector_score", 0.0))

    scope_gap = 0.0
    if len(retrieved_chunks) > 1:
        scope_gap = clamp01(
            retrieved_chunks[0].get("vector_score", 0.0) -
            retrieved_chunks[1].get("vector_score", 0.0)
        )

    ref_ratio = clamp01(len(final_references) / max(1, min(len(retrieved_chunks), 5)))

    confidence = 0.65 * top1_vector + 0.20 * scope_gap + 0.15 * ref_ratio
    return round(clamp01(confidence), 3)


# =========================
# 查询阶段：上下文 + 模型
# =========================
def build_context(retrieved_chunks, max_chars: int = 2500):
    parts = []
    total = 0

    for item in retrieved_chunks:
        snippet = item["text"][:300]
        block = (
            f"[{item['reference_id']}]\n"
            f"file_name:{item['file_name']}\n"
            f"score:{item['hybrid_score']}\n"
            f"chunk_id:{item['chunk_id']}\n"
            f"snippet:{snippet}\n"
        )

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts)


def call_llm(question: str, context: str):
    api_key = Settings.LLM_API_KEY
    model_name = Settings.MODEL_NAME
    api_url = Settings.LLM_API_URL

    if not api_key:
        raise ValueError("请先设置模型 API Key")

    payload = {
        "model": model_name,
        "temperature": Settings.TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_A},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    context=context,
                    question=question
                )
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content


def parse_llm_output(llm_output: str):
    llm_output = llm_output.strip()

    if llm_output.startswith("```"):
        llm_output = llm_output.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(llm_output)
        answer = data.get("answer", NO_ANSWER_FALLBACK)
        references = data.get("references", [])
        if not isinstance(references, list):
            references = []

        return {
            "ok": True,
            "answer": answer,
            "references": references
        }
    except Exception:
        return {
            "ok": False,
            "answer": NO_ANSWER_FALLBACK,
            "references": []
        }


def format_reference(retrieved_chunks, selected_reference: list):
    ref_map = {item["reference_id"]: item for item in retrieved_chunks}

    final_refs = []
    for ref_id in selected_reference:
        if ref_id in ref_map:
            item = ref_map[ref_id]
            final_refs.append({
                "reference_id": item["reference_id"],
                "file_name": item["file_name"],
                "source_path": item["source_path"],
                "chunk_id": item["chunk_id"],
                "position": item.get("position"),
                "snippet": make_snippet(item["text"]),
                "score": round(float(item.get("hybrid_score", 0.0)), 4)
            })

    return final_refs


# =========================
# 查询阶段：总控
# =========================
def ask(question: str, top_k: int = 3, retrieved_chunks: List[Dict[str, Any]] | None = None):
    if retrieved_chunks is None:
        retrieved_chunks = retrieve(question, top_k)

    if is_ambiguous_question(question):
        return {
            "status": "NEED_CLARIFICATION",
            "answer": "",
            "references": [],
            "confidence": 0.0,
            "need_clarification": True,
            "clarification_question": build_clarification_question(question),
            "reason": "问题存在指代不明或缺少关键限定信息"
        }

    if should_refuse(retrieved_chunks):
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": NO_ANSWER_FALLBACK,
            "references": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "检索结果为空或 top1 分数低于阈值"
        }

    context = build_context(retrieved_chunks)
    llm_out = call_llm(question, context)
    parsed = parse_llm_output(llm_out)

    if not parsed["ok"]:
        return {
            "status": "MODEL_BAD_OUTPUT",
            "answer": NO_ANSWER_FALLBACK,
            "references": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "模型未按约定 JSON 格式返回"
        }

    final_references = format_reference(retrieved_chunks, parsed["references"])

    if not final_references or STOPWORD in parsed["answer"]:
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": parsed["answer"],
            "references": final_references,
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "模型给出的答案没有足够的引用支撑"
        }

    confidence = estimate_confidence(retrieved_chunks, final_references, status="OK")

    return {
        "status": "OK",
        "answer": parsed["answer"],
        "references": final_references,
        "confidence": confidence,
        "need_clarification": False,
        "clarification_question": "",
        "reason": "检索命中有效证据，且答案成功绑定引用"
    }


# =========================
# 调试 / 预览
# =========================
def preview_docs():
    return docs_registry


def preview_chunks(limit: int = 70, file_name: str | None = None):
    result = []

    for item in chunks:
        if file_name and item["metadata"].get("file_name") != file_name:
            continue

        result.append({
            "file_name": item["metadata"].get("file_name"),
            "chunk_id": item["metadata"].get("chunk_id"),
            "source_path": item["metadata"].get("source_path"),
            "position": item["metadata"].get("position"),
            "text_preview": make_snippet(item.get("text", ""), max_len=200),
            "text_length": len(item.get("text", ""))
        })

        if len(result) >= limit:
            break

    return result


def get_chunk_full_text(chunk_id: int | str):
    for item in chunks:
        if str(item["metadata"].get("chunk_id")) == str(chunk_id):
            return {
                "found": True,
                "file_name": item["metadata"].get("file_name"),
                "chunk_id": item["metadata"].get("chunk_id"),
                "source_path": item["metadata"].get("source_path"),
                "position": item["metadata"].get("position"),
                "text": item.get("text")
            }

    return {
        "found": False,
        "file_name": "",
        "chunk_id": chunk_id,
        "source_path": "",
        "position": None,
        "text": ""
    }

# =========================
# 调试 / manifest
# =========================

def build_file_snapshot(data_dir:str):
    files = collect_supported_files(data_dir)
    snapshot = []

    for file_path in files:
        stat = file_path.stat()
        snapshot.append({
            "path":str(file_path.resolve()),
            "size":stat.st_size,
            "time":stat.st_mtime
        })

    snapshot.sort(key=lambda x: x["path"])
    return snapshot

def build_current_manifest(data_dir:str):
    return {
        "pipeline_version":1,
        "embedding_model":Settings.EMBEDDING_MODEL_NAME,
        "build_config":{
            "chunk_size":Settings.CHUNK_SIZE,
            "chunk_overlap":Settings.CHUNK_OVERLAP,
        },
        "files":build_file_snapshot(data_dir),
    }

def load_manifest():
    if not MANIFEST_PATH.exists():
        return None

    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_manifest(manifest:dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def needs_rebuild(data_dir: str) -> tuple[bool, str]:
    old_manifest = load_manifest()
    current_manifest = build_current_manifest(data_dir)

    if old_manifest is None:
        return True, "没有找到历史 manifest，需要首次构建"

    if old_manifest.get("pipeline_version") != current_manifest.get("pipeline_version"):
        return True, "pipeline_version 变化，需要重建"

    if old_manifest.get("embedding_model") != current_manifest.get("embedding_model"):
        return True, "embedding 模型变化，需要重建"

    if old_manifest.get("build_config") != current_manifest.get("build_config"):
        return True, "chunk 参数变化，需要重建"

    if old_manifest.get("files") != current_manifest.get("files"):
        return True, "data 目录文件变化，需要重建"

    return False, "目录和参数都没有变化，无需重建"


















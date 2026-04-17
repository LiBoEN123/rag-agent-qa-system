import json
from pathlib import Path
from charset_normalizer import from_bytes
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import requests
from app.config import Settings
from app.prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, USER_PROMPT, NO_ANSWER_FALLBACK
from typing import Dict, List, Any

# 全局变量
chunks = []
texts = []
docs_registry = []
bm25 = None
model = None
doc_embeddings = None
SUPPORTED_SUFFIXES = {".txt",".md"}
RAG_READY = False
RAG_INIT_MESSAGE = "RAG 尚未初始化"
NO_DOCUMENTS_MESSAGE = "当前知识库为空，请先在 data 目录中放入文档后再提问"
AMBIGUOUS_WORDS = [
    "这个", "那个", "它", "这条", "这项", "这份",
    "上面", "前面", "后面", "这里", "那里"
]
STOPWORD = "我无法根据提供的文档确定答案"


def read_txt_auto(file_path:str) -> str:
    raw = Path(file_path).read_bytes()
    result = from_bytes(raw).best()  # 得到：1.检测到的编码 2.解码后的文本 3.置信度或质量信息
    if result is None:
        raise ValueError(f'无法识别编码:{file_path}')
    return str(result) # 表示result中的文本字符串

def chunk_txt(file_path:str, chunk_size:int=600, chunk_overlap:int=100):
    text = read_txt_auto(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ""]
    )

    docs = splitter.create_documents(
        [text],
        metadatas=[{'file_type':'txt'}]
    )

    file_name = Path(file_path).name
    source_path = str(Path(file_path).resolve())

    result = []

    for idx, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata)
        metadata.update({
            "file_name":file_name,
            "source_path":source_path,
            "chunk_id":None,
            "position":idx
        })
        result.append({
            'text':doc.page_content,
            'metadata':metadata
        })

    return result

def chunk_md(file_path:str, chunk_size:int=600, chunk_overlap:int=100):
    text=read_txt_auto(file_path)

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ('#', 'h1'),
            ('##', 'h2'),
            ('###', 'h3')
        ]
    )
    docs = md_splitter.split_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ""]
    )
    docs = splitter.split_documents(docs)

    file_name = Path(file_path).name
    source_path = str(Path(file_path).resolve())

    result = []
    for idx, doc in enumerate(docs, start=1):
        title_path = " > ".join(
            [doc.metadata[k] for k in ['h1', 'h2', 'h3'] if k in doc.metadata]
        )

        text_with_title = (
            f'[标题路径] {title_path}\n\n{doc.page_content}'
            if title_path else doc.page_content
        )

        result.append({
            "text":text_with_title,
            "metadata":{
                'source_path':source_path,
                'file_type':"md",
                "file_name": file_name,
                "chunk_id":None,
                "position":idx,
                "title_path":title_path,
                **doc.metadata
            }
        })

    return result

def chunk_file(file_path:str, chunk_size:int=600, chunk_overlap:int=100):
    ex = Path(file_path).suffix.lower()
    if ex == '.txt':
        return chunk_txt(file_path, chunk_size, chunk_overlap)
    if ex == '.md':
        return chunk_md(file_path, chunk_size, chunk_overlap)
    else:
        raise ValueError('只支持.txt和.md')

def tokenize(text:str):
    return list(jieba.cut(text)) # 把一段中文文本切分成词，并返回一个列表

def collect_supported_files(data_dir:str):
    root = Path(data_dir)
    if not root.exists():
        return []
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_SUFFIXES:
            return []
        return [root]
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    return files

def load_documents(data_dir:str, chunk_size:int=Settings.CHUNK_SIZE, chunk_overlap:int=Settings.CHUNK_OVERLAP):
    files = collect_supported_files(data_dir)
    if not files:
        return [], []

    all_chunks = []
    docs_registry = []

    for idx, file_path in enumerate(files, start=1):
        file_chunks = chunk_file(str(file_path), chunk_size, chunk_overlap)
        all_chunks.extend(file_chunks)
        docs_registry.append({
            "doc_id":f"doc_{idx}",
            "file_name":file_path.name,
            "source_path":str(Path(file_path).resolve()),
            "file_type":file_path.suffix.lower().lstrip("."),
            "chunk_count":len(file_chunks)
        })

    for global_idx, chunk in enumerate(all_chunks, start=1):
        chunk["metadata"]["chunk_id"] = global_idx

    return all_chunks, docs_registry

def init_rag(data_dir:str, chunk_size:int=Settings.CHUNK_SIZE, chunk_overlap:int=Settings.CHUNK_OVERLAP):
    global chunks, texts, docs_registry, bm25, model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE

    # 每次初始化先清空状态
    chunks = []
    texts = []
    docs_registry = []
    bm25 = None
    doc_embeddings = None
    RAG_READY = False
    RAG_INIT_MESSAGE = "RAG 尚未初始化"

    chunks, docs_registry = load_documents(data_dir, chunk_size, chunk_overlap)

    texts = [chunk['text'] for chunk in chunks]

    # 关键：没有文档时，直接退出，不再继续建 embedding 和 bm25
    if not texts:
        RAG_INIT_MESSAGE = NO_DOCUMENTS_MESSAGE
        return

    tokenized_chunks = [tokenize(c) for c in texts]


    if model is None:
        model = SentenceTransformer(Settings.EMBEDDING_MODEL_NAME)

    doc_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    bm25 = BM25Okapi(tokenized_chunks)

    RAG_READY = True
    RAG_INIT_MESSAGE = "OK"

def get_rag_status():
    return {
        "ready": RAG_READY,
        "message": RAG_INIT_MESSAGE,
        "doc_count": len(docs_registry),
        "chunk_count": len(chunks)
    }

def minmax_score(x:np.ndarray):
    x = x.astype(np.float32)
    if np.allclose(x.min(), x.max()):
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

def retrieve(question:str, top_k:int = 3):
    if bm25 is None or doc_embeddings is None:
        return []
    question = (question or "").strip()
    if not question:
        return []

    tokenized_question = tokenize(question)
    bm25_score = bm25.get_scores(tokenized_question)

    question_embeddings = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    vector_score = doc_embeddings @ question_embeddings
    bm25_norm = minmax_score(bm25_score)
    vector_norm = minmax_score(vector_score)
    hybrid_score = 0.3 * bm25_norm + 0.7 * vector_norm

    top_indices = np.argsort(hybrid_score)[::-1][:top_k]
    result = []
    for rank, idx in enumerate(top_indices, 1):
        metadata = chunks[idx]['metadata']
        result.append({
            'reference_id':f'Reference {rank}',
            'file_name':metadata.get("file_name", "unknown"),
            'source_path':metadata.get("source_path", "unknown"),
            'chunk_id':metadata.get("chunk_id", f"chunk_{idx}"),
            'position':metadata.get("position", ""),
            'text':chunks[idx]['text'],
            'bm25_score':float(bm25_score[idx]),
            'vector_score':float(vector_score[idx]),
            'hybrid_score':float(hybrid_score[idx]),
            'metadata':metadata
        })
    return result

def search_relevant_chunks(question:str, top_k:int=3) -> List[Dict[str, Any]]:
    return retrieve(question, top_k)

def should_refuse(retrieved_chunks, vector_threshold=0.35, margin_threshold=0.02):
    vector_threshold = vector_threshold if vector_threshold is not None else Settings.VECTOR_THRESHOLD
    margin_threshold = margin_threshold if margin_threshold is not None else Settings.MARGIN_THRESHOLD

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

def clamp01(value:float) -> float:
    value = float(value)
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value

def is_ambiguous_question(question:str) -> bool:
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

def estimate_confidence(retrieved_chunks:List[dict], final_references:List[dict], status:str) -> float:
    if status != "OK" or not retrieved_chunks:
        return 0.0

    top1_vector = clamp01(retrieved_chunks[0].get("vector_score", 0.0))

    scope_gap = 0.0
    if len(retrieved_chunks) > 1:
        scope_gap = clamp01(retrieved_chunks[0].get("vector_score", 0.0) -
                            retrieved_chunks[1].get("vector_score", 0.0))

    ref_ratio = clamp01(len(final_references) / max(1, min(len(retrieved_chunks), 5)))

    confidence = 0.65 * top1_vector + 0.20 * scope_gap + 0.15 * ref_ratio

    return round(clamp01(confidence), 3) # 最终限制置信度0到1，且保留三位小数

def estimate_need_clarification(question:str, final_references:List[dict], status:str) -> bool:
    if is_ambiguous_question(question):
        return True
    if status == "OK" and not final_references:
        return True
    return False

def build_context(retrieved_chunks):
    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[{item['reference_id']}]\n"
            f"file_name:{item['file_name']}\n"
            f"source_path:{item['source_path']}\n"
            f"chunk_id:{item['chunk_id']}\n"
            f"context:{item['text']}\n"
        )
    return '\n'.join(context_parts)

def call_llm(question:str, context: str):
    api_key = Settings.LLM_API_KEY
    model_name = Settings.MODEL_NAME  # 默认模型改为 DeepSeek 的聊天模型
    api_url = Settings.LLM_API_URL  # 默认 URL 改为 DeepSeek 的端点

    if not api_key:
        raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")

    pay_load = {
        'model':model_name,
        'temperature':Settings.TEMPERATURE,
        'messages':[
            {"role": "system", "content": SYSTEM_PROMPT_A},
            {"role": 'user', "content": USER_PROMPT.format(
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

    resp = requests.post(api_url, headers=headers, json=pay_load, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content

def parse_llm_output(llm_output:str):
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
            "ok":True,
            "answer":answer,
            "references":references
        }
    except Exception:
        return {
            "ok":False,
            "answer":NO_ANSWER_FALLBACK,
            "references":[]
        }

def format_reference(retrieved_chunks, selected_reference:list):
    ref_map = {item['reference_id']:item for item in retrieved_chunks}

    final_refs = []
    for ref_id in selected_reference:
        if ref_id in ref_map:
            item = ref_map[ref_id]
            final_refs.append({
                'reference_id':item['reference_id'],
                'file_name':item['file_name'],
                'source_path':item['source_path'],
                'chunk_id':item['chunk_id'],
                'text':item['text']
            })
    return final_refs

def ask(question:str, top_k:int=3, retrieved_chunks:List[Dict[str, Any]] | None=None):

    if retrieved_chunks is None:
        retrieved_chunks = retrieve(question, top_k)

    ambiguous = is_ambiguous_question(question)

    # 1. 问题模糊：直接走澄清，不进入正常回答
    if ambiguous:
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
            'status':"NO_RELEVANT_INFO",
            'answer':NO_ANSWER_FALLBACK,
            'references':[],
            'confidence':0.0,
            'need_clarification':False,
            "clarification_question": "",
            "reason": "检索结果为空或 top1 分数低于阈值"
        }

    context = build_context(retrieved_chunks)
    llm_out = call_llm(question, context)
    parase = parse_llm_output(llm_out)



    if not parase["ok"]:
        return {
            "status": "MODEL_BAD_OUTPUT",
            "answer": NO_ANSWER_FALLBACK,
            "references": [],
            'confidence': 0.0,
            'need_clarification': False,
            "clarification_question": "",
            "reason": "模型未按约定 JSON 格式返回"
        }

    final_references = format_reference(retrieved_chunks, parase['references'])

    if not final_references or STOPWORD in parase['answer']:
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": parase['answer'],
            "references": final_references,
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "模型给出的答案没有足够的引用支撑"
        }

    confidence = estimate_confidence(retrieved_chunks, final_references, status="OK")

    return {
        'status':"OK",
        'answer':parase['answer'],
        'references':final_references,
        'confidence': confidence,
        'need_clarification': False,
        "clarification_question": "",
        "reason": "检索命中有效证据，且答案成功绑定引用"
    }

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
            "text_preview": item.get("text", "")[:200],
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





















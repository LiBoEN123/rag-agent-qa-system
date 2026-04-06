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
from app.prompts import SYSTEM_PROMPT, USER_PROMPT, NO_ANSWER_FALLBACK
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
        metadatas=[{'source':file_path, 'file_type':'txt'}]
    )

    source_file = Path(file_path).name
    result = []

    for idx, doc in enumerate(docs, start=1):
        metadata = dict(doc.metadata)
        metadata.update({
            "source_file":source_file,
            "chunk_id":f"{Path(file_path).stem}_chunk_{idx}",
            "chunk_index":idx
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

    source_file = Path(file_path).name
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
                'source':file_path,
                'file_type':"md",
                "source_file": source_file,
                "chunk_id": f"{Path(file_path).stem}_chunk_{idx}",
                "chunk_index": idx,
                "title_path":title_path,
                **doc.metadata
            }
        })

    return result

def chunk_file(file_path:str, chunk_size:int=800, chunk_overlap:int=120):
    ex = Path(file_path).suffix.lower()
    if ex == '.txt':
        return chunk_txt(file_path, chunk_size, chunk_overlap)
    if ex == '.md':
        return chunk_md(file_path, chunk_size, chunk_overlap)
    else:
        raise ValueError('只支持.txt和.md')

def tokenize(text:str):
    return list(jieba.cut(text)) # 把一段中文文本切分成词，并返回一个列表

def collecte_supported_files(data_dir:str):
    root = Path(data_dir)
    if not root.exists():
        return []
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_SUFFIXES:
            return []
        return [root]
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    return files

def load_documents(data_dir:str, chunk_size:int=800, chunk_overlap:int=120):
    files = collecte_supported_files(data_dir)
    if not files:
        return [], []

    all_chunks = []
    docs_registry = []

    for idx, file_path in enumerate(files, start=1):
        file_chunks = chunk_file(str(file_path), chunk_size, chunk_overlap)
        all_chunks.extend(file_chunks)
        docs_registry.append({
            "doc_id":f"doc_{idx}",
            "source_file":file_path.name,
            "source_path":str(file_path),
            "file_type":file_path.suffix.lower().lstrip("."),
            "chunk_count":len(file_chunks)
        })
    return all_chunks, docs_registry

def init_rag(file_path:str, chunk_size:int=800, chunk_overlap:int=120):
    global chunks, texts, docs_registry, bm25, model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE

    # 每次初始化先清空状态
    chunks = []
    texts = []
    docs_registry = []
    bm25 = None
    doc_embeddings = None
    RAG_READY = False
    RAG_INIT_MESSAGE = "RAG 尚未初始化"

    chunks, docs_registry = load_documents(file_path, chunk_size, chunk_overlap)

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
    hybrid_score = 0.4 * bm25_norm + 0.6 * vector_norm

    top_indices = np.argsort(hybrid_score)[::-1][:top_k]
    result = []
    for rank, idx in enumerate(top_indices, 1):
        metadata = chunks[idx]['metadata']
        result.append({
            'reference_id':f'Reference {rank}',
            'source_file':metadata.get("source_file", "unknown"),
            'chunk_id':metadata.get("chunk_id", f"chunk_{idx}"),
            'text':chunks[idx]['text'],
            'bm25_score':float(bm25_score[idx]),
            'vector_score':float(vector_score[idx]),
            'hybrid_score':float(hybrid_score[idx]),
            'metadata':metadata
        })
    return result

def search_relevant_chunks(question:str, top_k:int=3) -> List[Dict[str, Any]]:
    return retrieve(question, top_k)

def should_refuse(retrieved_chunks, hybrid_threshold:float=None, vector_threshold=None):
    hybrid_threshold = hybrid_threshold if hybrid_threshold is not None else Settings.HYBRID_THRESHOLD
    vector_threshold = vector_threshold if vector_threshold is not None else Settings.VECTOR_THRESHOLD
    if not retrieved_chunks:
        return True
    top1_score = retrieved_chunks[0]
    top1_hybrid_score = top1_score['hybrid_score']
    top1_vector_score = top1_score['vector_score']
    if top1_hybrid_score< hybrid_threshold:
        return True
    if top1_vector_score < vector_threshold:
        return True
    return False

def build_context(retrieved_chunks):
    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[{item['reference_id']}]\n"
            f"source_file:{item['source_file']}\n"
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
            {"role": "system", "content": SYSTEM_PROMPT},
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
                'source_file':item['source_file'],
                'chunk_id':item['chunk_id'],
                'text':item['text']
            })
    return final_refs

def ask(question:str, top_k:int=3, retrieved_chunks:List[Dict[str, Any]] | None=None):

    if retrieved_chunks is None:
        retrieved_chunks = retrieve(question, top_k)

    if should_refuse(retrieved_chunks):
        return {
            'status':"NO_RELEVANT_INFO",
            'answer':NO_ANSWER_FALLBACK,
            'references':[]
        }

    context = build_context(retrieved_chunks)
    llm_out = call_llm(question, context)
    parase = parse_llm_output(llm_out)

    if not parase["ok"]:
        return {
            "status": "MODEL_BAD_OUTPUT",
            "answer": NO_ANSWER_FALLBACK,
            "references": []
        }

    final_references = format_reference(retrieved_chunks, parase['references'])

    return {
        'status':"OK",
        'answer':parase['answer'],
        'references':final_references
    }

def preview_docs():
    return docs_registry

def preview_chunks(limit:int=20, source_file:str | None=None):
    result = []
    for item in chunks:
        metadata = item["metadata"]
        if source_file and metadata.get("source_file") != source_file:
            continue

        result.append({
            "chunk_id":metadata.get("chunk_id"),
            "source_file":metadata.get("source_file"),
            "file_type":metadata.get("file_type"),
            "title_path":metadata.get("title_path"),
            "text_preview":item["text"][:200],
            "text_length":len(item["text"])
        })

        if len(result) >= limit:
            break
    return result






















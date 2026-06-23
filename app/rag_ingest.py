import re
from pathlib import Path

from charset_normalizer import from_bytes
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from app.config import Settings


SUPPORTED_SUFFIXES = {".txt", ".md"}


def read_txt_auto(file_path: str) -> str:
    """自动识别文本编码并读取文件内容。"""
    raw = Path(file_path).read_bytes()
    result = from_bytes(raw).best()
    if result is None:
        raise ValueError(f"无法识别编码: {file_path}")
    return str(result)


def collect_supported_files(data_dir: str):
    """收集 data 目录中支持的 txt 和 md 文件。"""
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
    """读取支持的文档并生成原始文档列表和文档注册表。"""
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
            "chunk_count": 0,
        })

    return raw_docs, docs_registry_local


def clean_text(text: str) -> str:
    """统一换行和空白，得到更稳定的纯文本。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_markdown_text(text: str) -> str:
    """在通用清洗基础上移除 Markdown 注释。"""
    text = clean_text(text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return text.strip()


def clean_documents(raw_docs: list[dict]):
    """按文件类型清洗原始文档内容。"""
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


def split_txt_doc(
    doc: dict,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    """将 txt 文档切分成带元数据的 chunk。"""
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


def split_md_doc(
    doc: dict,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    """按 Markdown 标题层级和长度切分文档。"""
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


def split_doc(
    doc: dict,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    """按文件类型选择合适的切分逻辑。"""
    if doc["file_type"] == "txt":
        return split_txt_doc(doc, chunk_size, chunk_overlap)
    if doc["file_type"] == "md":
        return split_md_doc(doc, chunk_size, chunk_overlap)
    raise ValueError(f"不支持的文件类型: {doc['file_type']}")


def split_chunks(
    cleaned_docs: list[dict],
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    """批量切分文档并分配全局 chunk_id。"""
    all_chunks = []
    chunk_count_map = {}

    for doc in cleaned_docs:
        file_chunks = split_doc(doc, chunk_size, chunk_overlap)
        all_chunks.extend(file_chunks)
        chunk_count_map[doc["source_path"]] = len(file_chunks)

    for global_idx, chunk in enumerate(all_chunks, start=1):
        chunk["metadata"]["chunk_id"] = global_idx

    return all_chunks, chunk_count_map
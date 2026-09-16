"""Local Knowledge 文件读取；PDF/DOCX 共用结构 parser。"""

import re
from pathlib import Path

from charset_normalizer import from_bytes

from app.ingestion.document_blocks import canonical_document_text
from app.rag.parsing import parse_document


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


def read_txt_auto(file_path: str) -> str:
    """自动识别文本编码并读取文件内容。"""
    raw = Path(file_path).read_bytes()
    result = from_bytes(raw).best()
    if result is None:
        raise ValueError(f"无法识别编码: {file_path}")
    return str(result)


def read_pdf_text(file_path: str) -> str:
    """从共享结构解析结果取得兼容的正文视图。"""
    content = canonical_document_text(parse_document(file_path))
    if not content:
        raise ValueError(f"PDF 未提取到可索引文本，可能是扫描件或图片型 PDF: {file_path}")
    return content


def read_docx_text(file_path: str) -> str:
    """按原始 reading order 返回 DOCX 的兼容正文视图。"""
    content = canonical_document_text(parse_document(file_path))
    if not content:
        raise ValueError(f"DOCX 未提取到可索引文本: {file_path}")
    return content


def read_file_text(file_path: str) -> str:
    """按文件扩展名选择对应的文档解析器。"""
    suffix = Path(file_path).suffix.lower()

    if suffix in {".txt", ".md"}:
        return read_txt_auto(file_path)
    if suffix == ".pdf":
        return read_pdf_text(file_path)
    if suffix == ".docx":
        return read_docx_text(file_path)

    raise ValueError(f"不支持的文件类型: {suffix}")


def collect_supported_files(data_dir: str):
    """收集 data 目录中支持的知识库文件。"""
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

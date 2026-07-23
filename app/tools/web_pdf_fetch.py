"""安全下载公开 PDF，并使用 pypdf 提取有限页范围文本。"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.config import Settings
from app.tools.context import ToolInvocationContext
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.result import tool_success
from app.tools.web_download import (
    download_public_web_resource,
    normalize_public_web_url,
)


_ALLOWED_CONTENT_TYPES = frozenset(
    {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    }
)
_PDF_MAGIC = b"%PDF-"
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v]+")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_METADATA_LIMITS = {
    "title": 500,
    "author": 300,
    "subject": 500,
}


def _invalid_pdf(message: str) -> ToolExecutionError:
    """构造不会泄漏 PDF 内部结构或原始内容的稳定错误。"""
    return ToolExecutionError(
        message,
        code=ToolErrorCode.INVALID_RESULT,
        retryable=False,
    )


def _normalize_page_text(value: object) -> str:
    """统一页面换行和连续空白，不摘要、改写或翻译正文。"""
    if not isinstance(value, str):
        return ""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _HORIZONTAL_WHITESPACE.sub(" ", normalized)
    normalized = _SPACE_AROUND_NEWLINE.sub("\n", normalized)
    return _MULTIPLE_BLANK_LINES.sub("\n\n", normalized).strip()


def _metadata_text(value: object, *, max_chars: int) -> str:
    """把 PDF 常见元数据规范化为有限单行字符串。"""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:max_chars]


def _read_metadata(reader: PdfReader) -> tuple[str, str, str]:
    """仅读取公开 metadata 映射中的 title、author 和 subject。"""
    try:
        metadata = reader.metadata
    except PyPdfError as exc:
        raise _invalid_pdf("PDF metadata could not be parsed") from exc
    except Exception as exc:
        raise _invalid_pdf("PDF metadata could not be parsed") from exc
    if metadata is None:
        return "", "", ""
    return (
        _metadata_text(metadata.get("/Title"), max_chars=_METADATA_LIMITS["title"]),
        _metadata_text(
            metadata.get("/Author"),
            max_chars=_METADATA_LIMITS["author"],
        ),
        _metadata_text(
            metadata.get("/Subject"),
            max_chars=_METADATA_LIMITS["subject"],
        ),
    )


def _reader_from_bytes(body: bytes) -> PdfReader:
    """通过 pypdf 公开 PdfReader API 构造只读解析器。"""
    if not body.startswith(_PDF_MAGIC):
        raise _invalid_pdf("Web PDF response does not contain a valid PDF signature")
    try:
        return PdfReader(BytesIO(body), strict=False)
    except PyPdfError as exc:
        raise _invalid_pdf("PDF document is malformed or could not be parsed") from exc
    except Exception as exc:
        raise _invalid_pdf("PDF document is malformed or could not be parsed") from exc


def _extract_pages(
    reader: PdfReader,
    *,
    start_page: int,
    max_pages: int,
    max_chars: int,
) -> tuple[int, list[int], str, bool]:
    """提取有限连续页，并在组合带页码正文后执行字符截断。"""
    try:
        if reader.is_encrypted:
            raise _invalid_pdf("Encrypted PDF documents are not supported")
        total_pages = len(reader.pages)
    except ToolExecutionError:
        raise
    except PyPdfError as exc:
        raise _invalid_pdf("PDF page tree could not be parsed") from exc
    except Exception as exc:
        raise _invalid_pdf("PDF page tree could not be parsed") from exc

    if total_pages < 1:
        raise _invalid_pdf("PDF document does not contain any pages")
    if total_pages > Settings.WEB_PDF_MAX_DOCUMENT_PAGES:
        raise _invalid_pdf("PDF document exceeded the configured page limit")
    if start_page > total_pages:
        raise ToolExecutionError(
            "Requested PDF start_page exceeds the document page count",
            code=ToolErrorCode.INVALID_ARGUMENTS,
            retryable=False,
        )

    start_index = start_page - 1
    end_index = min(start_index + max_pages, total_pages)
    sections: list[tuple[int, str]] = []
    try:
        for page_index in range(start_index, end_index):
            text = _normalize_page_text(reader.pages[page_index].extract_text())
            if text:
                page_number = page_index + 1
                sections.append((page_number, f"[Page {page_number}]\n{text}"))
    except PyPdfError as exc:
        raise _invalid_pdf("PDF page text could not be extracted") from exc
    except Exception as exc:
        raise _invalid_pdf("PDF page text could not be extracted") from exc

    if not sections:
        raise _invalid_pdf(
            "PDF 未包含可提取文本，可能是扫描件或图片型 PDF；PR8 不支持 OCR。"
        )

    full_content = "\n\n".join(section for _, section in sections)
    truncated = len(full_content) > max_chars
    content = full_content[:max_chars]
    returned_page_numbers: list[int] = []
    section_offset = 0
    for page_number, section in sections:
        text_offset = section_offset + len(f"[Page {page_number}]\n")
        if text_offset < len(content):
            returned_page_numbers.append(page_number)
        section_offset += len(section) + 2
    return total_pages, returned_page_numbers, content, truncated


def web_pdf_fetch(
    args: dict[str, Any],
    context: ToolInvocationContext,
) -> dict[str, Any]:
    """下载公开 PDF，提取指定页范围并返回统一成功结果。"""
    del context
    resource = download_public_web_resource(
        normalize_public_web_url(args.get("url")),
        allowed_content_types=_ALLOWED_CONTENT_TYPES,
        max_response_bytes=Settings.WEB_PDF_MAX_RESPONSE_BYTES,
    )
    reader = _reader_from_bytes(resource.body)
    total_pages, page_numbers, content, truncated = _extract_pages(
        reader,
        start_page=args["start_page"],
        max_pages=args["max_pages"],
        max_chars=args["max_chars"],
    )
    title, author, subject = _read_metadata(reader)
    output = {
        "requested_url": resource.requested_url,
        "final_url": resource.final_url,
        "status_code": resource.status_code,
        "content_type": resource.content_type,
        "title": title,
        "author": author,
        "subject": subject,
        "total_pages": total_pages,
        "start_page": args["start_page"],
        "pages_read": len(page_numbers),
        "page_numbers": page_numbers,
        "content": content,
        "content_length": len(content),
        "truncated": truncated,
        "extractor": "pypdf",
        "content_is_untrusted": True,
    }
    return tool_success("web_pdf_fetch", args, output)


__all__ = ["web_pdf_fetch"]

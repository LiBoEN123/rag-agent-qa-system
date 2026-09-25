"""受限临时文件与共享结构 parser 对接；只生成完整知识文档，不执行索引。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO
from zipfile import ZipFile

from app.ingestion import KnowledgeDocument, KnowledgeSourceType, build_knowledge_document
from app.ingestion.document_blocks import ParsedDocument, canonical_document_text
from app.ingestion.files import read_txt_auto
from app.ingestion.parsing import parse_document


_MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_READ_BLOCK_BYTES = 64 * 1024


class UploadSourceError(ValueError):
    """仅携带固定错误码和安全消息，不透传 loader 异常或临时路径。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _load_content(path: Path, max_bytes: int) -> tuple[str, ParsedDocument | None]:
    if path.suffix == ".pdf":
        with path.open("rb") as stream:
            if not stream.read(5).startswith(b"%PDF-"):
                raise ValueError("invalid PDF header")
    if path.suffix == ".docx":
        # DOCX 是压缩包；在 loader 解压正文前限制展开大小，不提取包内路径。
        with ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > max_bytes:
                raise UploadSourceError("UPLOAD_TOO_LARGE", "DOCX 展开大小超过上传限制")
    if path.suffix in {".pdf", ".docx"}:
        structure = parse_document(path)
        return canonical_document_text(structure), structure
    content = read_txt_auto(str(path))
    if re.search(r"[\x00-\x08\x0e-\x1f\x7f]", content):
        raise ValueError("invalid text content")
    return content, None


def load_upload_document(
    file: BinaryIO, *, filename: str | None, max_bytes: int,
) -> KnowledgeDocument:
    """分块读取单文件，以原始字节摘要固定 URI，交由 factory 生成领域 identity。"""
    if (
        not filename
        or len(filename) > 255
        or any(char in filename for char in "/\\:")
        or re.search(r"[\x00-\x1f\x7f]", filename)
    ):
        raise UploadSourceError("UPLOAD_INVALID_FILENAME", "文件名无效，请使用不含路径的文件名")
    suffix = Path(filename).suffix.lower()
    if suffix not in _MIME_TYPES:
        raise UploadSourceError("UPLOAD_UNSUPPORTED_TYPE", "仅支持 PDF、DOCX、TXT、MD 文件")

    digest = hashlib.sha256()
    size = 0
    with TemporaryDirectory(prefix="knowledge-upload-") as directory:
        path = Path(directory) / f"document{suffix}"
        with path.open("wb") as target:
            while block := file.read(min(_READ_BLOCK_BYTES, max_bytes - size + 1)):
                size += len(block)
                if size > max_bytes:
                    raise UploadSourceError("UPLOAD_TOO_LARGE", "文件超过上传大小限制")
                digest.update(block)
                target.write(block)
        if size == 0:
            raise UploadSourceError("UPLOAD_EMPTY_FILE", "不能上传空文件")
        try:
            content, structure = _load_content(path, max_bytes)
        except UploadSourceError:
            raise
        except Exception:
            raise UploadSourceError(
                "UPLOAD_PARSE_FAILED", "无法解析文件，请检查文件格式或是否损坏、加密",
            ) from None
        if not content.strip():
            raise UploadSourceError(
                "UPLOAD_NO_TEXT", "文件未提取到文本；扫描或图片型 PDF 暂不支持 OCR",
            )

    file_sha256 = digest.hexdigest()
    metadata = {
        "file_name": filename,
        "file_type": suffix.lstrip("."),
        "file_size": size,
        "file_sha256": file_sha256,
    }
    if structure is not None and structure.page_count is not None:
        metadata["page_count"] = structure.page_count
    try:
        return build_knowledge_document(
            source_type=KnowledgeSourceType.UPLOAD,
            source_uri=f"upload://local/{file_sha256}",
            title=filename,
            content=content,
            structured_content=structure,
            mime_type=_MIME_TYPES[suffix],
            metadata=metadata,
            content_is_untrusted=True,
        )
    except Exception:
        raise UploadSourceError(
            "UPLOAD_PARSE_FAILED", "无法解析文件，请检查文件格式或是否损坏、加密",
        ) from None

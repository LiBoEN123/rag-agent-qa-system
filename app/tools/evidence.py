"""工具正文使用原生 Document；这里只定义 identity 与公共 metadata 约定。"""

import hashlib
import json
from typing import Any

from langchain_core.documents import Document


def build_evidence_id(source_type: str, *identity: object) -> str:
    """使用带来源命名空间的确定性摘要，避免 locator 拼接歧义。"""
    payload = json.dumps([source_type, *identity], ensure_ascii=False, sort_keys=True)
    return f"{source_type}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def build_evidence_document(
    *,
    page_content: str,
    source_type: str,
    source_uri: str,
    identity: tuple[object, ...],
    metadata: dict[str, Any],
) -> Document:
    """保留来源 metadata，并将正文标记为数据而非指令。"""
    document = Document(
        id=build_evidence_id(source_type, *identity),
        page_content=page_content,
        metadata={
            **metadata,
            "source_type": source_type,
            "source_uri": source_uri,
            "content_is_untrusted": True,
        },
    )
    validate_evidence_documents([document])
    return document


def validate_evidence_documents(documents: list[Document]) -> list[Document]:
    """补充 Document 本身不约束的项目必填字段，不创建第二套模型。"""
    for document in documents:
        for key, value in (
            ("id", document.id),
            ("page_content", document.page_content),
            ("source_type", document.metadata.get("source_type")),
            ("source_uri", document.metadata.get("source_uri")),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evidence {key} 必须是非空字符串")
        if document.metadata.get("content_is_untrusted") is not True:
            raise ValueError("evidence content_is_untrusted 必须为 True")
    return documents

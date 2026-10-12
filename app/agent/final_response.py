"""最终回答静态合同及 ToolMessage Document 到公开引用的薄投影。"""

import json
from collections.abc import Mapping
from typing import Literal, Self

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.chat import RAG_EVIDENCE_TOOL_NAMES
from app.rag.retrieval import make_snippet
from app.common.numeric import safe_float
from app.tools.evidence import validate_evidence_documents
from app.prompts import ANSWER_COMPOSER_SYSTEM_PROMPT


class FinalAgentResponse(BaseModel):
    """声明回答决策；Document 答案正文由证据校验后的模型流生成。"""

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, revalidate_instances="always",
    )

    outcome: Literal["answer", "no_evidence", "clarify"]
    answer: str = Field(default="", description="有 evidence_ids 时留空；无 Document 的操作结果可填写回答。")
    evidence_ids: list[str] = Field(default_factory=list)
    clarification_question: str = ""

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """仅校验静态规则；本次 Run 的证据存在性由 Result Builder 检查。"""
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        if self.outcome == "answer":
            if (not self.answer and not self.evidence_ids) or self.clarification_question:
                raise ValueError("answer requires evidence_ids or text and no clarification_question")
        elif self.outcome == "no_evidence":
            if self.evidence_ids or self.clarification_question:
                raise ValueError("no_evidence must not cite evidence or ask clarification")
        elif self.evidence_ids or not self.clarification_question:
            raise ValueError("clarify requires a question and no evidence_ids")
        return self


_EVIDENCE_TOOLS = RAG_EVIDENCE_TOOL_NAMES | frozenset({"web_fetch", "web_pdf_fetch"})


def collect_evidence_from_messages(messages: object) -> dict[str, Document]:
    """仅恢复本次成功正文工具的 Document；不把搜索摘要当证据。"""
    evidence: dict[str, Document] = {}
    if not isinstance(messages, (list, tuple)):
        return evidence
    for message in messages:
        if (
            not isinstance(message, ToolMessage)
            or message.name not in _EVIDENCE_TOOLS
            or message.status != "success"
        ):
            continue
        result = message.artifact
        if not isinstance(result, dict):
            raise ValueError("Invalid evidence tool result")
        if result.get("ok") is not True or result.get("status") != "success":
            continue
        if result.get("tool_name") != message.name:
            raise ValueError("Invalid evidence tool name")
        output = result.get("output")
        if not isinstance(output, dict) or not isinstance(output.get("evidence"), list):
            raise ValueError("Invalid evidence tool output")
        documents = validate_evidence_documents([
            Document.model_validate(value) for value in output["evidence"]
        ])
        for document in documents:
            evidence.setdefault(document.id, document)
    return evidence


class FinalResponseError(ValueError):
    """仅携带安全原因和引用 ID 的 grounding 错误。"""

    def __init__(self, reason: str, **diagnostics: list[str]) -> None:
        super().__init__(reason)
        self.diagnostics = diagnostics


def validate_final_response(
    framework_output: Mapping[str, object],
) -> tuple[FinalAgentResponse, list[Document]]:
    """共用唯一 Evidence Gate；只接受框架对象和本次成功 ToolMessage。"""
    final = framework_output.get("structured_response")
    if not isinstance(final, FinalAgentResponse):
        raise FinalResponseError("Agent structured response is missing or invalid")
    final = FinalAgentResponse.model_validate(final)
    evidence = collect_evidence_from_messages(framework_output.get("messages"))
    if any(key not in evidence for key in final.evidence_ids):
        raise FinalResponseError(
            "Agent structured response referenced unavailable evidence",
            requested_evidence_ids=list(final.evidence_ids),
            available_evidence_ids=list(evidence),
            unavailable_evidence_ids=[key for key in final.evidence_ids if key not in evidence],
        )
    if final.outcome == "answer" and evidence and not final.evidence_ids:
        raise FinalResponseError("Agent structured answer did not cite available evidence")
    return final, [evidence[key] for key in final.evidence_ids]


def build_answer_messages(
    question: str, evidence: list[Document],
) -> list[SystemMessage | HumanMessage]:
    """只把问题和已锁定 Document 的公开内容交给正文生成模型。"""
    return [
        SystemMessage(content=ANSWER_COMPOSER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "question": question,
            "evidence": [{
                "id": document.id,
                "page_content": document.page_content,
                "source": document.metadata["source_uri"],
            } for document in evidence],
        }, ensure_ascii=False)),
    ]


def document_to_reference(document: Document) -> dict:
    """保持现有 ChatReferences 字段，以 Document.id 作为稳定引用身份。"""
    metadata = document.metadata
    is_kb = metadata["source_type"] == "knowledge_base"
    source_uri = metadata["source_uri"]
    return {
        "reference_id": document.id,
        "file_name": str(metadata.get("file_name") or "") if is_kb else (
            metadata.get("title") or source_uri
        ),
        "source_path": (metadata.get("source_path") or source_uri) if is_kb else source_uri,
        "chunk_id": (
            metadata["chunk_id"] if is_kb and metadata.get("chunk_id") is not None
            else document.id
        ),
        "position": metadata.get("position") if is_kb else metadata.get("start_page"),
        "snippet": make_snippet(document.page_content),
        "score": safe_float(metadata.get("hybrid_score")) if is_kb else 0.0,
        "rerank_score": safe_float(metadata.get("rerank_score")) if is_kb else 0.0,
    }

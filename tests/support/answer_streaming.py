"""可控的原生 ChatModel 流；工具与 create_agent 仍走真实框架。"""

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import StructuredTool
from pydantic import Field

from app.agent import framework_agent, framework_tools, streaming
from app.agent.final_response import FinalAgentResponse
from app.tools.evidence import build_evidence_document
from app.tools.result import tool_content, tool_success
from tests.support.agent_runtime import FakeToolCallingModel


class ControlledAnswerModel(BaseChatModel):
    release: asyncio.Event = Field(default_factory=asyncio.Event)
    waiting: asyncio.Event = Field(default_factory=asyncio.Event)
    completed: asyncio.Event = Field(default_factory=asyncio.Event)
    closed: asyncio.Event = Field(default_factory=asyncio.Event)
    received: list = Field(default_factory=list)
    fail: bool = False
    empty: bool = False
    first_text: str = " 已确认"
    last_text: str = "的回答。\n"

    @property
    def _llm_type(self):
        return "controlled-answer"

    def _generate(self, *args, **kwargs):
        raise AssertionError("Composer must use native astream")

    async def _astream(self, messages, **kwargs):
        self.received.append(messages)
        try:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=[
                    {"type": "reasoning", "text": "PRIVATE REASONING"},
                    *([] if self.empty else [{"type": "text", "text": self.first_text}]),
                ],
                additional_kwargs={"private": "PRIVATE PROVIDER METADATA"},
            ))
            if self.empty:
                return
            self.waiting.set()
            await self.release.wait()
            if self.fail:
                raise RuntimeError("PRIVATE PROVIDER FAILURE")
            yield ChatGenerationChunk(message=AIMessageChunk(content=self.last_text))
            self.completed.set()
        finally:
            self.closed.set()


def install_evidence_agent(monkeypatch, composer, *, tool_name="search_knowledge_base"):
    """用真实 ToolNode 产生两个 Document，只选择其中一个供正文模型使用。"""
    source_type = {"web_fetch": "web_page", "web_pdf_fetch": "web_pdf"}.get(tool_name, "knowledge_base")
    documents = [build_evidence_document(
        page_content=body, source_type=source_type, source_uri="https://example.com/policy",
        identity=(index,), metadata={"private": "PRIVATE DOCUMENT METADATA"},
    ) for index, body in enumerate(("已验证的事实。", "UNSELECTED DOCUMENT BODY"))]
    payload = tool_success(tool_name, {}, {"evidence": documents, "count": 2})

    async def retrieve():
        result = {**payload, "output": {
            **payload["output"], "evidence": [document.model_dump() for document in documents],
        }}
        return tool_content(result), result

    tool = StructuredTool.from_function(
        coroutine=retrieve, name=tool_name, description="Read evidence",
        response_format="content_and_artifact",
    )
    monkeypatch.setattr(framework_tools, "get_local_tools", lambda settings: (tool,))
    model = FakeToolCallingModel([
        AIMessage(content="PRIVATE AGENT TEXT", tool_calls=[{
            "name": tool_name, "args": {}, "id": "evidence-read",
        }]),
        FinalAgentResponse(outcome="answer", evidence_ids=[documents[0].id]),
    ])
    monkeypatch.setattr(framework_agent, "get_agent_model", lambda: model)
    monkeypatch.setattr(streaming, "get_answer_model", lambda: composer)
    return model, documents, payload

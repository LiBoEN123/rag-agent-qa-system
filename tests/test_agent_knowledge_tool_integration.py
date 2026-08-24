"""知识库工具从 Agent Model 到 RAG 子图的严格真实链路测试。"""

import json

from langchain_core.messages import AIMessage, ToolMessage

from app.agent import framework_agent, framework_tools
import app.tools.runtime as tool_runtime
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry
from app.tools.result import tool_success
from app.tools.schemas import SearchKnowledgeBaseArgs
from tests.support.agent_runtime import build_agent_harness


class KnowledgeModel:
    def __init__(self):
        self.bound_tools = []
        self.received_messages: list[list] = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.received_messages.append(list(messages))
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "奖学金申请条件", "top_k": 5},
                    "id": "knowledge-call",
                    "type": "tool_call",
                }
            ],
        )


def _invoke_knowledge_graph(
    monkeypatch,
    *,
    output: dict,
    rag_result: dict,
):
    registry = ToolRegistry()
    seen_contexts: list[ToolInvocationContext] = []
    seen_args: list[dict] = []
    runtime_calls: list[str] = []
    rag_calls: list[dict] = []

    def handler(args, context):
        seen_args.append(dict(args))
        seen_contexts.append(context)
        resolved_rag = fake_rag_subgraph(
            question=context.effective_question,
            top_k=context.top_k,
            decompose_top_k=context.decompose_top_k,
            retrieved_chunks=output["chunks"],
            trace_id=context.trace_id,
        )
        return tool_success(
            "search_knowledge_base",
            args,
            {**output, "rag_result": resolved_rag},
        )

    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            version="1.0.0",
            description="检索本地测试知识库并返回受控证据 chunk",
            output_schema={"count": "integer", "chunks": "list"},
            handler=handler,
            args_model=SearchKnowledgeBaseArgs,
            required_permissions=("knowledge_base:read",),
        )
    )
    provider = LocalToolProvider(registry)
    model = KnowledgeModel()
    original_execute = tool_runtime.ToolRuntime.execute

    def recording_execute(self, tool_name, tool_args=None, *, context=None):
        runtime_calls.append(tool_name)
        return original_execute(self, tool_name, tool_args, context=context)

    def fake_rag_subgraph(**kwargs):
        rag_calls.append(kwargs)
        return dict(rag_result)

    monkeypatch.setattr(tool_runtime.ToolRuntime, "execute", recording_execute)
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_default_tool_provider",
        lambda: provider,
    )
    context = ToolInvocationContext(
        trace_id="trace-knowledge-strict",
        session_id="session-knowledge-strict",
        principal_id="knowledge-test",
        permissions=("knowledge_base:read",),
        source="knowledge_tool_integration_test",
    )

    graph_output = build_agent_harness().invoke(
        {
            "original_question": "奖学金申请条件是什么？",
            "effective_question": "奖学金申请条件是什么？",
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": {"action": "agent", "router_llm_used": False},
            "trace_id": context.trace_id,
            "session_id": context.session_id,
        },
        context=context,
    )
    return {
        "output": graph_output,
        "model": model,
        "context": context,
        "seen_contexts": seen_contexts,
        "seen_args": seen_args,
        "runtime_calls": runtime_calls,
        "rag_calls": rag_calls,
    }


def test_search_knowledge_base_uses_real_structured_tool_registry_and_runtime(
    monkeypatch,
):
    chunk = {
        "chunk_id": "scholarship-1",
        "text": "申请人需要提交成绩单。",
        "file_name": "scholarship.md",
    }
    result = _invoke_knowledge_graph(
        monkeypatch,
        output={"count": 1, "chunks": [chunk]},
        rag_result={
            "status": "OK",
            "answer": "申请人需要提交成绩单。",
            "references": [{"reference_id": "Reference 1"}],
            "reason": "证据充分",
            "decompose_llm_used": False,
            "answer_llm_used": True,
            "llm_call_count": 1,
            "timings": {},
            "retrieval_confidence": 0.9,
            "answer_confidence": 0.9,
        },
    )

    bound_tool = result["model"].bound_tools[0]
    assert [tool.name for tool in result["model"].bound_tools] == [
        "search_knowledge_base"
    ]
    assert issubclass(bound_tool.args_schema, SearchKnowledgeBaseArgs)
    assert "runtime" not in bound_tool.tool_call_schema.model_fields
    assert result["runtime_calls"] == ["search_knowledge_base"]
    assert result["seen_args"] == [
        {"query": "奖学金申请条件", "top_k": 5}
    ]
    assert len(result["seen_contexts"]) == 1
    seen_context = result["seen_contexts"][0]
    assert seen_context.trace_id == result["context"].trace_id
    assert seen_context.session_id == result["context"].session_id
    assert seen_context.permissions == result["context"].permissions
    assert seen_context.effective_question == "奖学金申请条件是什么？"
    assert seen_context.top_k == 5
    assert seen_context.decompose_top_k == 10
    assert result["rag_calls"][0]["retrieved_chunks"] == [chunk]
    assert result["output"]["result"]["references"] == [
        {"reference_id": "Reference 1"}
    ]


def test_successful_zero_result_search_is_not_a_tool_error(monkeypatch):
    result = _invoke_knowledge_graph(
        monkeypatch,
        output={"count": 0, "chunks": []},
        rag_result={
            "status": "NO_RELEVANT_INFO",
            "answer": "我没有在文档中找到相关信息。",
            "references": [],
            "reason": "无可用证据",
            "decompose_llm_used": False,
            "answer_llm_used": False,
            "llm_call_count": 0,
            "timings": {},
            "retrieval_confidence": 0.0,
            "answer_confidence": 0.0,
        },
    )

    assert result["runtime_calls"] == ["search_knowledge_base"]
    assert result["output"]["tool_error_count"] == 0
    assert result["output"]["observations"][0]["success"] is True
    assert result["rag_calls"][0]["retrieved_chunks"] == []
    assert result["output"]["result"]["status"] == "NO_RELEVANT_INFO"


def test_unavailable_knowledge_base_returns_tool_message_and_agent_recovers(
    monkeypatch,
):
    """知识库不可用由工具链返回标准错误，模型读取后仍可安全结束。"""

    class RecoveringModel:
        def __init__(self):
            self.calls = 0
            self.received_messages: list[list] = []

        def bind_tools(self, tools):
            assert [tool.name for tool in tools] == ["search_knowledge_base"]
            return self

        def invoke(self, messages):
            self.received_messages.append(list(messages))
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_knowledge_base",
                            "args": {"query": "测试不可用知识库", "top_k": 5},
                            "id": "knowledge-unavailable",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="知识库暂时不可用，请稍后重试。")

    registry = ToolRegistry()

    def unavailable_handler(args, context):
        del args, context
        raise ToolExecutionError(
            "knowledge base unavailable",
            code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            retryable=False,
        )

    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            version="1.0.0",
            description="检索本地测试知识库",
            output_schema={"count": "integer", "chunks": "list"},
            handler=unavailable_handler,
            args_model=SearchKnowledgeBaseArgs,
            required_permissions=("knowledge_base:read",),
        )
    )
    model = RecoveringModel()
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_default_tool_provider",
        lambda: LocalToolProvider(registry),
    )
    context = ToolInvocationContext(
        trace_id="trace-knowledge-unavailable",
        permissions=("knowledge_base:read",),
        source="knowledge_tool_integration_test",
    )

    output = build_agent_harness().invoke(
        {
            "original_question": "知识库问题",
            "effective_question": "知识库问题",
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": {"action": "agent", "router_llm_used": False},
            "trace_id": context.trace_id,
        },
        context=context,
    )

    tool_message = model.received_messages[1][-1]
    assert isinstance(tool_message, ToolMessage)
    payload = json.loads(tool_message.content)
    assert payload["ok"] is False
    assert payload["error_code"] == "dependency_unavailable"
    assert output["result"]["status"] == "OK"
    assert output["result"]["answer"] == "知识库暂时不可用，请稍后重试。"
    assert "knowledge_base_result" not in output

import pytest
from pydantic import BaseModel

from app.agent import executor as agent_executor
from app.agent.router import (
    route_agent_action,
    route_agent_action_rule_based,
)
from app.graph import nodes as graph_nodes
from app.graph import runtime as graph_runtime
from app.llm import factory as model_factory
from app.llm.gateway import ModelGateway
from app.tools.contracts import ToolSpec
from app.tools.registry import tool_registry


def test_empty_question_returns_clarify_without_router_model(monkeypatch):
    """空问题由确定性入口要求澄清，不产生 Router 模型调用。"""

    def fail_model_call(*args, **kwargs):
        raise AssertionError("确定性 Router 不应调用模型")

    monkeypatch.setattr(ModelGateway, "invoke_text", fail_model_call)
    monkeypatch.setattr(model_factory, "_create_model", fail_model_call)

    decision = route_agent_action(" \n ", 5, trace_id="router-test")

    assert decision["decision"] == "clarify"
    assert decision["action"] == "clarify"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}
    assert decision["clarification_question"]
    assert decision["router_source"] == "rule_based_guardrail"
    assert decision["router_confidence"] == 1.0
    assert decision["router_llm_used"] is False
    assert decision["router_fallback_reason"] == ""


@pytest.mark.parametrize(
    "question",
    ["你是谁？", "你能做什么？", "你可以做什么？", "你有什么功能？"],
)
def test_capability_questions_return_direct_answer(question):
    """少量稳定的系统能力问题继续走 direct_answer。"""
    decision = route_agent_action(question, 5)

    assert decision["decision"] == "direct_answer"
    assert decision["action"] == "direct_answer"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}
    assert decision["router_source"] == "rule_based"
    assert decision["router_llm_used"] is False


@pytest.mark.parametrize(
    "question",
    ["这个怎么弄", "这个怎么配置", "这个是什么意思", "这个怎么解决"],
)
def test_ambiguous_references_return_clarify(question):
    """少量明确指代不明的问题继续走 clarify。"""
    decision = route_agent_action(question, 5)

    assert decision["decision"] == "clarify"
    assert decision["action"] == "clarify"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}
    assert "具体指" in decision["clarification_question"]
    assert decision["router_source"] == "rule_based"
    assert decision["router_llm_used"] is False


@pytest.mark.parametrize(
    "question",
    [
        "知识库有哪些文档？",
        "请对比《学生手册》和《调整通知》的差异。",
        "12 * 8",
        "查看切块 5 的内容",
        "搜索一下最新公开资料",
        "读取 https://example.com/page",
        "调用刚注册的新能力处理这段输入",
    ],
)
def test_normal_questions_enter_agent_without_concrete_tool_choice(question):
    """普通问题统一进入 Agent，Router 不选择工具或构造参数。"""
    decision = route_agent_action(question, 5)

    assert decision["decision"] == "use_agent"
    assert decision["action"] == "agent"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}
    assert decision["router_source"] == "rule_based"
    assert decision["router_confidence"] == 1.0
    assert decision["router_llm_used"] is False
    assert decision["router_fallback_reason"] == ""


def test_registered_tool_does_not_require_router_change(monkeypatch):
    """新增 Registry Tool 后，确定性 Router 仍返回同一通用 Agent 入口。"""

    class NewToolArgs(BaseModel):
        value: str = ""

    monkeypatch.setattr(tool_registry, "_tools", dict(tool_registry._tools))
    before = route_agent_action("调用刚注册的新能力处理这段输入", 5)
    tool_registry.register(
        ToolSpec(
            name="pr9_test_tool",
            version="1.0.0",
            description="仅用于验证新增工具不要求修改 Router。",
            args_schema={"value": "string"},
            output_schema={"value": "string"},
            args_model=NewToolArgs,
            handler=lambda args, context: {"value": args.get("value", "")},
        )
    )
    after = route_agent_action("调用刚注册的新能力处理这段输入", 5)

    assert "pr9_test_tool" in tool_registry.names()
    assert after == before
    assert after["action"] == "agent"
    assert after["tool_name"] == ""


def test_rule_route_keeps_current_decision_shape():
    """规则函数继续返回供 Graph/API 兼容层消费的现有字段。"""
    decision = route_agent_action_rule_based("普通问题", 3)

    assert decision == {
        "decision": "use_agent",
        "action": "agent",
        "tool_name": "",
        "tool_args": {},
        "reason": "交给 Agent 根据当前请求可见的动态工具定义决策",
    }


def test_langgraph_uses_search_knowledge_base_tool_once(monkeypatch):
    """验证 LangGraph 知识库路径通过 LangChain Tool 调用标准检索工具。"""
    called = {"count": 0}
    fake_chunk = {
        "chunk_id": 1,
        "text": "测试 chunk",
        "snippet": "测试 chunk",
        "hybrid_score": 0.9,
        "bm25_score": 1.0,
        "vector_score": 0.8,
        "rerank_score": 0.7,
        "file_name": "test.md",
        "metadata": {
            "chunk_id": 1,
            "file_name": "test.md",
        },
    }

    def fake_langchain_call(step, *, context):
        """模拟 search_knowledge_base 的 StructuredTool 调用结果。"""
        called["count"] += 1
        assert step.tool_name == "search_knowledge_base"
        assert step.tool_args == {"query": "测试问题", "top_k": 5}
        assert context.trace_id == "test_trace"
        return {
            "status": "success",
            "ok": True,
            "tool_name": step.tool_name,
            "tool_input": step.tool_args,
            "output": {
                "count": 1,
                "chunks": [fake_chunk],
                "top_chunk_ids": [1],
                "top_scores": [0.9],
            },
            "error": "",
        }

    def fake_ask(*args, **kwargs):
        """模拟 RAG 生成结果并校验收到的检索 chunks。"""
        assert kwargs.get("retrieved_chunks") == [fake_chunk]
        return {
            "status": "OK",
            "answer": "测试答案",
            "references": [],
            "final_chunks": [fake_chunk],
            "retrieval_confidence": 0.9,
            "answer_confidence": 0.8,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "测试",
            "tool_trace": [],
            "decompose_llm_used": False,
            "answer_llm_used": True,
            "llm_call_count": 1,
        }

    monkeypatch.setattr(
        agent_executor,
        "_call_langchain_tool",
        fake_langchain_call,
    )
    monkeypatch.setattr(graph_nodes, "run_rag_subgraph", fake_ask)

    result = graph_runtime.run_agent_langgraph(
        write_bad_cases=lambda record: None,
        original_question="测试问题",
        effective_question="测试问题",
        top_k=5,
        decompose_top_k=10,
        decision={
            "decision": "use_knowledge_base",
            "action": "search_knowledge_base",
            "tool_name": "search_knowledge_base",
            "tool_args": {"query": "测试问题", "top_k": 5},
            "reason": "测试知识库检索",
            "router_source": "rule_based",
            "router_llm_used": False,
        },
        trace_id="test_trace",
    )

    call_tool_steps = [
        step
        for step in result["tool_trace"]
        if step.get("step") == "tool_call"
        and step.get("tool_name") == "search_knowledge_base"
    ]

    assert called["count"] == 1
    assert result["status"] == "OK"
    assert call_tool_steps
    assert call_tool_steps[0]["tool_args"] == {"query": "测试问题", "top_k": 5}
    assert call_tool_steps[0]["tool_status"] == "success"
    assert call_tool_steps[0]["success"] is True
    assert call_tool_steps[0]["final_chunk_count"] == 1
    assert call_tool_steps[0]["top_chunk_ids"] == [1]
    assert call_tool_steps[0]["top_scores"] == [0.9]
    assert call_tool_steps[0]["hits"][0]["file_name"] == "test.md"

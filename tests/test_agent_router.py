import pytest

from app import agent
from app.agent import answer as agent_answer
from app.agent import executor as agent_executor
from app.agent import runtime as agent_runtime
from app.tools.registry import tool_registry


def test_agent_allowed_tools_from_registry():
    """验证 Agent Router 的工具白名单来自注册中心。"""
    assert agent.ROUTER_ALLOWED_TOOLS == set(tool_registry.names())


def test_agent_router_tool_specs_from_registry():
    """验证 Agent Router 的工具说明来自注册中心。"""
    specs = agent.ROUTER_TOOL_SPECS
    names = {item["name"] for item in specs}

    assert "search_knowledge_base" in names
    assert "simple_calculator" in names
    assert "list_documents" in names
    assert "get_chunk_text" in names
    assert "compare_documents" in names

    for item in specs:
        assert "name" in item
        assert "description" in item
        assert "args_schema" in item


def test_search_knowledge_base_is_now_standard_tool():
    """验证知识库检索已经成为 Router 可调用的标准工具。"""
    assert "search_knowledge_base" in agent.ROUTER_ALLOWED_TOOLS
    assert "search_knowledge_base" in agent.ROUTER_ALLOWED_ACTIONS


def test_compare_documents_is_allowed_tool():
    """验证 compare_documents 已进入 Agent Router 白名单。"""
    assert "compare_documents" in agent.ROUTER_ALLOWED_TOOLS
    assert "compare_documents" in agent.ROUTER_ALLOWED_ACTIONS


def test_validate_llm_compare_documents_route_keeps_safe_args():
    """验证 LLM 路由选择 compare_documents 时会校验并规整参数。"""
    decision = agent.validate_route_decision(
        {
            "decision": "use_tool",
            "action": "compare_documents",
            "tool_name": "compare_documents",
            "tool_args": {
                "query": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u6700\u957f\u5b66\u4e60\u5e74\u9650",
                "doc_a": "old.md",
                "doc_b": "new.md",
                "top_k": "20",
            },
            "reason": "compare two docs",
            "confidence": 0.7,
        },
        "compare docs",
        3,
    )

    assert decision["decision"] == "use_tool"
    assert decision["action"] == "compare_documents"
    assert decision["tool_name"] == "compare_documents"
    assert decision["tool_args"] == {
        "query": "\u666e\u901a\u672c\u79d1\u56db\u5e74\u5236\u6700\u957f\u5b66\u4e60\u5e74\u9650",
        "doc_a": "old.md",
        "doc_b": "new.md",
        "top_k": 10,
    }


def test_rule_route_detects_explicit_compare_documents_question():
    """验证显式双标题对比问题会命中 compare_documents 规则路由。"""
    decision = agent.route_agent_action_rule_based(
        "\u8bf7\u5bf9\u6bd4\u300a\u5185\u8499\u53e4\u5de5\u4e1a\u5927\u5b66\u5b66\u751f\u624b\u518c\u300b\u548c\u300a\u8c03\u6574\u901a\u77e5\u300b\u4e2d\u5173\u4e8e\u6700\u957f\u5b66\u4e60\u5e74\u9650\u7684\u5dee\u5f02\u3002",
        3,
    )

    assert decision["action"] == "compare_documents"
    assert decision["tool_name"] == "compare_documents"
    assert decision["tool_args"]["doc_a"] == "\u5185\u8499\u53e4\u5de5\u4e1a\u5927\u5b66\u5b66\u751f\u624b\u518c"
    assert decision["tool_args"]["doc_b"] == "\u8c03\u6574\u901a\u77e5"


def test_rule_route_does_not_guess_compare_documents_from_vague_names():
    """验证缺少两个显式文档标题时，规则路由不会硬编码猜测对比文件。"""
    decision = agent.route_agent_action_rule_based(
        "\u5b66\u751f\u624b\u518c\u548c\u901a\u77e5\u4e2d\u6700\u957f\u5b66\u4e60\u5e74\u9650\u6709\u4ec0\u4e48\u5dee\u5f02\uff1f",
        3,
    )

    assert decision["action"] != "compare_documents"


def test_validate_llm_rag_route_normalizes_safe_args():
    """验证知识库路由会忽略不安全的模型参数并使用原问题。"""
    decision = agent.validate_route_decision(
        {
            "decision": "use_knowledge_base",
            "action": "search_knowledge_base",
            "tool_name": "some_hallucinated_tool",
            "tool_args": {"query": "bad rewrite", "top_k": 999},
            "reason": "需要查知识库",
            "confidence": 0.9,
        },
        "Vue 应用如何创建？",
        3,
    )

    assert decision["decision"] == "use_knowledge_base"
    assert decision["action"] == "search_knowledge_base"
    assert decision["tool_name"] == "search_knowledge_base"
    assert decision["tool_args"] == {"query": "Vue 应用如何创建?", "top_k": 3}
    assert decision["router_source"] == "llm"
    assert decision["router_confidence"] == 0.9


def test_validate_llm_tool_route_checks_calculator_expression():
    """验证计算器路由会规范化并校验表达式参数。"""
    decision = agent.validate_route_decision(
        {
            "decision": "use_tool",
            "action": "simple_calculator",
            "tool_name": "simple_calculator",
            "tool_args": {"expression": "１＋２*3"},
            "reason": "需要计算",
        },
        "帮我算 1 + 2 * 3",
        5,
    )

    assert decision["decision"] == "use_tool"
    assert decision["tool_name"] == "simple_calculator"
    assert decision["tool_args"] == {"expression": "1+2*3"}


def test_validate_llm_route_rejects_unknown_tool():
    """验证未知工具路由会被拒绝。"""
    with pytest.raises(ValueError):
        agent.validate_route_decision(
            {
                "decision": "use_tool",
                "action": "delete_documents",
                "tool_name": "delete_documents",
                "tool_args": {},
            },
            "删除所有文档",
            5,
        )


def test_tool_route_uses_rules_before_llm_when_router_enabled(monkeypatch):
    """验证工具类问题命中高置信规则时不会调用 LLM Router。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", True, raising=False)
    called = {"count": 0}

    def fake_llm(question, top_k, trace_id=""):
        """模拟不应被调用的 LLM Router。"""
        called["count"] += 1
        raise AssertionError("工具类问题不应调用 LLM Router")

    monkeypatch.setattr(agent, "route_agent_action_by_llm", fake_llm)

    decision = agent.route_agent_action("知识库有哪些文档？", 5)

    assert decision["action"] == "list_documents"
    assert decision["tool_args"] == {}
    assert decision["router_source"] == "rule_based"
    assert decision["router_llm_used"] is False
    assert called["count"] == 0


def test_default_kb_route_can_use_llm_when_enabled(monkeypatch):
    """验证默认知识库问题在启用 Router 时仍可走 LLM 路由。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", True, raising=False)
    monkeypatch.setattr(
        agent,
        "route_agent_action_by_llm",
        lambda question, top_k, trace_id="": {
            "decision": "use_knowledge_base",
            "action": "search_knowledge_base",
            "tool_args": {"query": "bad rewrite"},
            "reason": "需要查知识库",
            "confidence": 0.8,
        },
    )

    decision = agent.route_agent_action("Vue 应用如何创建？", 5)

    assert decision["action"] == "search_knowledge_base"
    assert decision["router_source"] == "llm"
    assert decision["router_llm_used"] is True


def test_route_falls_back_to_rules_when_llm_is_invalid(monkeypatch):
    """验证 LLM 路由非法时会回退到规则路由。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", True, raising=False)
    monkeypatch.setattr(
        agent,
        "route_agent_action_by_llm",
        lambda question, top_k, trace_id="": {"action": "unknown_tool"},
    )

    decision = agent.route_agent_action("Vue 应用如何创建？", 5)

    assert decision["action"] == "search_knowledge_base"
    assert decision["router_source"] == "rule_based_fallback"
    assert decision["router_llm_used"] is True
    assert "未知路由 action" in decision["router_fallback_reason"]


def test_route_uses_rules_when_router_disabled(monkeypatch):
    """验证关闭 LLM 路由时直接使用规则路由。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", False, raising=False)

    decision = agent.route_agent_action("知识库有多少文档？", 5)

    assert decision["action"] == "list_documents"
    assert decision["router_source"] == "rule_based"
    assert decision["router_llm_used"] is False


def test_run_agent_v2_uses_search_knowledge_base_tool(monkeypatch):
    """验证 Agent v2 知识库路径通过 dispatch_tool 调用标准检索工具。"""
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

    def fake_dispatch_tool(tool_name, tool_args=None, trace_id=""):
        """模拟 search_knowledge_base 工具分发结果。"""
        called["count"] += 1
        assert tool_name == "search_knowledge_base"
        assert tool_args == {"query": "测试问题", "top_k": 5}
        return {
            "status": "success",
            "ok": True,
            "tool_name": tool_name,
            "tool_input": tool_args,
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

    monkeypatch.setattr(agent_executor, "dispatch_tool", fake_dispatch_tool)
    monkeypatch.setattr(agent_answer, "_call_ask", fake_ask)

    result = agent_runtime.run_agent_v2(
        write_bad_cases=lambda record: None,
        question="测试问题",
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

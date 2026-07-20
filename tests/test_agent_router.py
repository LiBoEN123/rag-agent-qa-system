import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app import agent
from app.agent import executor as agent_executor
from app.agent import llm_router
from app.agent import router as agent_router
from app.graph import nodes as graph_nodes
from app.graph import runtime as graph_runtime
from app.tools.registry import tool_registry


def test_router_llm_uses_only_coarse_grained_actions(monkeypatch):
    """Router Prompt 只暴露入口动作，不再暴露具体工具或 Schema。"""

    class FakeRouterModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(
                content='{"action":"rag_fast_path","reason":"需要检索"}'
            )

    model = FakeRouterModel()
    monkeypatch.setattr(llm_router, "get_router_model", lambda: model)

    output = llm_router.call_router_llm("测试问题", 4, trace_id="router-trace")
    parsed = llm_router.parse_router_json_output(output)

    assert parsed["action"] == "rag_fast_path"
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[0][1], HumanMessage)
    user_prompt = model.calls[0][1].content
    assert "agent" in user_prompt
    assert "rag_fast_path" in user_prompt
    assert "direct_answer" in user_prompt
    assert "search_knowledge_base" not in user_prompt
    assert "compare_documents" not in user_prompt


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


def test_validate_legacy_tool_route_normalizes_to_agent():
    """旧版具体工具输出只作为兼容输入并规整为通用 Agent。"""
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

    assert decision["decision"] == "use_agent"
    assert decision["action"] == "agent"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}


def test_rule_route_sends_document_comparison_to_agent():
    """文档对比进入 Agent，Router 不提取具体工具参数。"""
    decision = agent.route_agent_action_rule_based(
        "\u8bf7\u5bf9\u6bd4\u300a\u5185\u8499\u53e4\u5de5\u4e1a\u5927\u5b66\u5b66\u751f\u624b\u518c\u300b\u548c\u300a\u8c03\u6574\u901a\u77e5\u300b\u4e2d\u5173\u4e8e\u6700\u957f\u5b66\u4e60\u5e74\u9650\u7684\u5dee\u5f02\u3002",
        3,
    )

    assert decision["action"] == "agent"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}


def test_rule_route_does_not_guess_compare_documents_from_vague_names():
    """验证缺少两个显式文档标题时，规则路由不会硬编码猜测对比文件。"""
    decision = agent.route_agent_action_rule_based(
        "\u5b66\u751f\u624b\u518c\u548c\u901a\u77e5\u4e2d\u6700\u957f\u5b66\u4e60\u5e74\u9650\u6709\u4ec0\u4e48\u5dee\u5f02\uff1f",
        3,
    )

    assert decision["action"] == "agent"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}


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
    assert decision["action"] == "rag_fast_path"
    assert decision["tool_name"] == "search_knowledge_base"
    assert decision["tool_args"] == {"query": "Vue 应用如何创建?", "top_k": 3}
    assert decision["router_source"] == "llm"
    assert decision["router_confidence"] == 0.9


def test_validate_llm_tool_route_leaves_args_to_agent_schema():
    """Router 不再为计算工具规范化或校验参数。"""
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

    assert decision["decision"] == "use_agent"
    assert decision["action"] == "agent"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}


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


def test_normal_request_reaches_agent_without_llm_router(monkeypatch):
    """普通请求不在动态工具发现前被静态或 LLM Router 截走。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", True, raising=False)
    called = {"count": 0}

    def fake_llm(question, top_k, trace_id=""):
        """模拟不应被调用的 LLM Router。"""
        called["count"] += 1
        raise AssertionError("普通请求应直接进入动态 Agent")

    monkeypatch.setattr(agent_router, "route_agent_action_by_llm", fake_llm)

    decision = agent.route_agent_action("知识库有哪些文档？", 5)

    assert decision["action"] == "agent"
    assert decision["tool_args"] == {}
    assert decision["router_source"] == "rule_based"
    assert decision["router_llm_used"] is False
    assert called["count"] == 0


@pytest.mark.parametrize(
    "question",
    [
        "查看切块5的内容",
        "查看分块5的内容",
        "把编号为5的知识片段原文发给我",
        "我想读一下第五段保存的完整文字",
        "麻烦展示 ID 是 5 的那条记录",
        "What is stored under item number 5?",
        "调用刚注册的能力处理这段输入",
    ],
)
def test_request_wording_does_not_bypass_dynamic_agent(monkeypatch, question):
    """Router 不靠工具关键词猜意图，换口吻和新工具请求都进入 Agent。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", True, raising=False)

    def fail_if_called(question, top_k, trace_id=""):
        raise AssertionError("自然语言工具意图不应在 Agent 前被静态分类")

    monkeypatch.setattr(
        agent_router,
        "route_agent_action_by_llm",
        fail_if_called,
    )

    decision = agent.route_agent_action(question, 5)

    assert decision["decision"] == "use_agent"
    assert decision["action"] == "agent"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}
    assert decision["router_source"] == "rule_based"
    assert decision["router_llm_used"] is False


def test_llm_router_cannot_intercept_normal_request_as_rag(monkeypatch):
    """即使配置启用 LLM Router，正常入口也不能抢在动态 Agent 前选 RAG。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", True, raising=False)
    called = {"count": 0}

    def fake_llm(question, top_k, trace_id=""):
        called["count"] += 1
        return {
            "decision": "use_knowledge_base",
            "action": "rag_fast_path",
            "reason": "错误地当成普通事实问答",
            "confidence": 0.99,
        }

    monkeypatch.setattr(
        agent_router,
        "route_agent_action_by_llm",
        fake_llm,
    )

    decision = agent.route_agent_action("Vue 应用如何创建？", 5)

    assert decision["action"] == "agent"
    assert decision["router_source"] == "rule_based"
    assert decision["router_llm_used"] is False
    assert called["count"] == 0


def test_normal_route_does_not_depend_on_llm_router_availability(monkeypatch):
    """动态 Agent 入口不因 Router 模型异常而改变或增加一次模型调用。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", True, raising=False)
    called = {"count": 0}

    def unavailable_llm(question, top_k, trace_id=""):
        called["count"] += 1
        raise RuntimeError("router unavailable")

    monkeypatch.setattr(
        agent_router,
        "route_agent_action_by_llm",
        unavailable_llm,
    )

    decision = agent.route_agent_action("Vue 应用如何创建？", 5)

    assert decision["action"] == "agent"
    assert decision["router_source"] == "rule_based"
    assert decision["router_llm_used"] is False
    assert called["count"] == 0


def test_route_uses_rules_when_router_disabled(monkeypatch):
    """验证关闭 LLM 路由时直接使用规则路由。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", False, raising=False)

    decision = agent.route_agent_action("知识库有多少文档？", 5)

    assert decision["action"] == "agent"
    assert decision["router_source"] == "rule_based"
    assert decision["router_llm_used"] is False


def test_plain_arithmetic_expression_routes_to_agent_without_extracting_args():
    """纯算式属于工具任务，但具体参数仍由 Agent Model 与 Schema 处理。"""
    decision = agent.route_agent_action_rule_based("12 * 8", 5)

    assert decision["decision"] == "use_agent"
    assert decision["action"] == "agent"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {}


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

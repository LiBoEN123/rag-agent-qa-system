from app.agent.planner import build_plan_from_decision
from app.agent.state import AgentPlanStep


def test_rag_decision_builds_search_knowledge_base_and_final_answer_plan():
    """验证 RAG 路由决策会生成检索和最终回答两步计划。"""
    decision = {
        "decision": "use_knowledge_base",
        "action": "search_knowledge_base",
        "tool_name": "search_knowledge_base",
        "tool_args": {
            "query": "普通本科四年制最长学习年限是多少？",
            "top_k": 5,
        },
        "reason": "需要查知识库",
    }

    plan = build_plan_from_decision(decision, "普通本科四年制最长学习年限是多少？", 5)

    assert all(isinstance(step, AgentPlanStep) for step in plan)
    assert len(plan) == 2
    assert plan[0].step_id == 1
    assert plan[0].action == "tool_call"
    assert plan[0].tool_name == "search_knowledge_base"
    assert plan[0].tool_args["query"] == "普通本科四年制最长学习年限是多少？"
    assert plan[0].tool_args["top_k"] == 5
    assert plan[0].status == "pending"
    assert plan[1].step_id == 2
    assert plan[1].action == "final_answer"
    assert plan[1].tool_name == ""


def test_search_knowledge_base_uses_question_and_top_k_when_tool_args_missing():
    """验证知识库检索缺少 tool_args 时使用问题和 top_k 兜底。"""
    decision = {
        "decision": "use_knowledge_base",
        "action": "search_knowledge_base",
        "reason": "需要查知识库",
    }

    plan = build_plan_from_decision(decision, "测试问题", 3)

    assert plan[0].tool_name == "search_knowledge_base"
    assert plan[0].tool_args == {"query": "测试问题", "top_k": 3}


def test_compare_documents_builds_tool_call_and_final_answer_plan():
    """验证文档对比决策会生成对比工具和最终回答两步计划。"""
    decision = {
        "decision": "use_tool",
        "action": "compare_documents",
        "tool_name": "compare_documents",
        "tool_args": {
            "query": "对比最长学习年限",
            "doc_a": "学生手册",
            "doc_b": "调整通知",
            "top_k": 3,
        },
        "reason": "用户要求对比两份文档",
    }

    plan = build_plan_from_decision(decision, "请对比两份文档", 5)

    assert len(plan) == 2
    assert plan[0].action == "tool_call"
    assert plan[0].tool_name == "compare_documents"
    assert plan[0].tool_args["doc_a"] == "学生手册"
    assert plan[0].tool_args["doc_b"] == "调整通知"
    assert plan[1].action == "final_answer"


def test_simple_calculator_builds_tool_call_and_final_answer_plan():
    """验证计算器决策会生成计算工具和最终回答两步计划。"""
    decision = {
        "decision": "use_tool",
        "action": "simple_calculator",
        "tool_name": "simple_calculator",
        "tool_args": {"expression": "1+2*3"},
        "reason": "用户要求计算",
    }

    plan = build_plan_from_decision(decision, "帮我算 1+2*3", 5)

    assert len(plan) == 2
    assert plan[0].tool_name == "simple_calculator"
    assert plan[0].tool_args == {"expression": "1+2*3"}
    assert plan[1].action == "final_answer"


def test_list_documents_builds_tool_call_and_final_answer_plan():
    """验证文档列表决策会生成列表工具和最终回答两步计划。"""
    decision = {
        "decision": "use_tool",
        "action": "list_documents",
        "tool_name": "list_documents",
        "tool_args": {},
        "reason": "用户询问文档列表",
    }

    plan = build_plan_from_decision(decision, "知识库有哪些文档？", 5)

    assert len(plan) == 2
    assert plan[0].tool_name == "list_documents"
    assert plan[0].tool_args == {}
    assert plan[1].action == "final_answer"


def test_get_chunk_text_builds_tool_call_and_final_answer_plan():
    """验证 chunk 原文决策会生成查看原文工具和最终回答两步计划。"""
    decision = {
        "decision": "use_tool",
        "action": "get_chunk_text",
        "tool_name": "get_chunk_text",
        "tool_args": {"chunk_id": 7},
        "reason": "用户要求查看指定 chunk",
    }

    plan = build_plan_from_decision(decision, "查看 chunk 7", 5)

    assert len(plan) == 2
    assert plan[0].tool_name == "get_chunk_text"
    assert plan[0].tool_args == {"chunk_id": 7}
    assert plan[1].action == "final_answer"


def test_clarify_only_builds_clarify_step():
    """验证澄清决策只生成 clarify 步骤。"""
    decision = {
        "decision": "clarify",
        "action": "clarify",
        "clarification_question": "你说的这个具体指哪份文档？",
        "reason": "问题存在指代不明",
    }

    plan = build_plan_from_decision(decision, "这个怎么弄？", 5)

    assert len(plan) == 1
    assert plan[0].step_id == 1
    assert plan[0].action == "clarify"
    assert plan[0].tool_name == ""
    assert plan[0].tool_args == {}
    assert "具体指哪份文档" in plan[0].reason


def test_direct_answer_only_builds_direct_answer_step():
    """验证直接回答决策只生成 direct_answer 步骤。"""
    decision = {
        "decision": "direct_answer",
        "action": "direct_answer",
        "reason": "用户询问系统能力，可以直接回答",
    }

    plan = build_plan_from_decision(decision, "你能做什么？", 5)

    assert len(plan) == 1
    assert plan[0].action == "direct_answer"
    assert plan[0].tool_name == ""
    assert plan[0].tool_args == {}


def test_unknown_action_falls_back_to_search_knowledge_base_plan():
    """验证未知 action 会稳定回退到知识库检索计划。"""
    decision = {
        "decision": "use_tool",
        "action": "unknown_tool",
        "tool_name": "unknown_tool",
        "tool_args": {},
        "reason": "异常工具",
    }

    plan = build_plan_from_decision(decision, "测试问题", 4)

    assert len(plan) == 2
    assert plan[0].action == "tool_call"
    assert plan[0].tool_name == "search_knowledge_base"
    assert plan[0].tool_args == {"query": "测试问题", "top_k": 4}
    assert plan[1].action == "final_answer"


def test_non_dict_decision_falls_back_to_search_knowledge_base_plan():
    """验证非 dict decision 会稳定回退到知识库检索计划。"""
    plan = build_plan_from_decision(None, "测试问题", 5)

    assert plan[0].tool_name == "search_knowledge_base"
    assert plan[0].tool_args == {"query": "测试问题", "top_k": 5}

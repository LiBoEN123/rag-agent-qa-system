from app.agent.answer import (
    DIRECT_ANSWER_TEXT,
    INTERNAL_ERROR_TEXT,
    apply_rag_result_to_state,
    compose_final_answer,
    map_rag_status,
)
from app.agent.state import AgentObservation, AgentState


def test_apply_rag_result_to_state_maps_graph_result():
    """RAG 子图结果完整写回答案、引用、计数和 trace。"""
    state = AgentState(
        trace_id="trace_001",
        original_question="原始问题",
        rewritten_question="改写问题",
        top_k=5,
        decompose_top_k=10,
    )
    result = apply_rag_result_to_state(
        state,
        {
            "status": "OK",
            "answer": "这是 RAG 答案",
            "references": [{"reference_id": "Reference 1"}],
            "reason": "检索命中有效证据",
            "decompose_llm_used": False,
            "answer_llm_used": True,
            "llm_call_count": 1,
            "retrieval_confidence": 0.9,
            "answer_confidence": 0.8,
            "timings": {"llm_cost_ms": 3},
        },
    )

    assert result.status == "ok"
    assert result.final_answer == "这是 RAG 答案"
    assert result.references == [{"reference_id": "Reference 1"}]
    assert result.answer_llm_used is True
    assert result.llm_call_count == 1
    assert result.tool_trace[-1]["mode"] == "rag"
    assert result.tool_trace[-1]["retrieval_confidence"] == 0.9
    assert result.tool_trace[-1]["answer_confidence"] == 0.8
    assert result.tool_trace[-1]["timings"] == {"llm_cost_ms": 3}


def test_apply_rag_result_to_state_maps_no_relevant_info():
    state = AgentState(original_question="测试问题")

    result = apply_rag_result_to_state(
        state,
        {
            "status": "NO_RELEVANT_INFO",
            "answer": "我无法根据提供的文档确定答案",
            "references": [],
            "reason": "检索结果为空或证据分数低于阈值",
            "llm_call_count": 0,
        },
    )

    assert result.status == "no_evidence"
    assert result.final_answer == "我无法根据提供的文档确定答案"


def test_compose_final_answer_keeps_clarify_state_and_adds_trace():
    state = AgentState(
        original_question="这个怎么弄？",
        status="clarify",
        final_answer="你说的这个具体指哪份文档？",
        reason="问题需要澄清",
    )

    result = compose_final_answer(state)

    assert result.status == "clarify"
    assert result.final_answer == "你说的这个具体指哪份文档？"
    assert result.tool_trace[-1]["mode"] == "clarify"


def test_compose_final_answer_generates_direct_answer_instead_of_using_reason():
    state = AgentState(
        original_question="你能做什么？",
        status="direct_answer",
        reason="用户询问系统能力，属于 direct_answer 范畴",
    )

    result = compose_final_answer(state)

    assert result.final_answer == DIRECT_ANSWER_TEXT
    assert result.final_answer != result.reason
    assert result.tool_trace[-1]["mode"] == "direct"


def test_compose_final_answer_keeps_framework_agent_answer():
    state = AgentState(
        original_question="帮我计算",
        status="ok",
        final_answer="计算结果是 7。",
        reason="Agent model produced final answer",
        observations=[
            AgentObservation(
                step_id=1,
                tool_name="simple_calculator",
                success=True,
                output={"result": 7},
            )
        ],
    )

    result = compose_final_answer(state)

    assert result.final_answer == "计算结果是 7。"
    assert result.tool_trace[-1]["mode"] == "agent_loop"


def test_compose_final_answer_keeps_tool_error_semantics():
    state = AgentState(
        original_question="测试问题",
        status="tool_error",
        reason="mock tool failed",
    )

    result = compose_final_answer(state)

    assert result.status == "tool_error"
    assert "mock tool failed" in result.final_answer
    assert result.tool_trace[-1]["mode"] == "error"


def test_compose_final_answer_uses_stable_model_error_text():
    state = AgentState(
        original_question="测试问题",
        status="model_error",
        reason="agent_model_call_failed",
    )

    result = compose_final_answer(state)

    assert result.status == "model_error"
    assert result.final_answer == "Agent 模型暂时不可用，无法安全选择或继续调用工具。"


def test_unexpected_state_fails_closed_without_formatting_observation():
    state = AgentState(
        original_question="测试问题",
        observations=[
            AgentObservation(
                step_id=1,
                tool_name="simple_calculator",
                success=True,
                output={"result": 7},
            )
        ],
    )

    result = compose_final_answer(state)

    assert result.status == "error"
    assert result.reason == "Unexpected Agent state before compose: running"
    assert result.final_answer == INTERNAL_ERROR_TEXT
    assert result.tool_trace[-1]["mode"] == "error"


def test_map_rag_status():
    assert map_rag_status("OK") == "ok"
    assert map_rag_status("NEED_CLARIFICATION") == "clarify"
    assert map_rag_status("NO_RELEVANT_INFO") == "no_evidence"
    assert map_rag_status("MODEL_ERROR") == "model_error"
    assert map_rag_status("MODEL_BAD_OUTPUT") == "model_error"
    assert map_rag_status("unexpected") == "error"

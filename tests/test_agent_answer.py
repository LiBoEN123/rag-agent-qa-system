"""Agent 答案 pure-update helper 测试。"""

from copy import deepcopy

from app.agent.answer import (
    DIRECT_ANSWER_TEXT,
    INTERNAL_ERROR_TEXT,
    build_compose_answer_updates,
    build_rag_answer_updates,
)


def _state(**updates):
    state = {
        "original_question": "测试问题",
        "effective_question": "测试问题",
        "observations": [],
        "final_answer": "",
        "references": [],
        "status": "running",
        "reason": "",
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
        "tool_trace": [],
    }
    state.update(updates)
    return state


def test_build_rag_answer_updates_maps_graph_result_without_mutation():
    """RAG 结果完整写回答案、引用、计数和 trace，输入保持不变。"""
    state = _state(tool_trace=[{"step": "decide"}])
    rag_result = {
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
    }
    original_state = deepcopy(state)
    original_rag_result = deepcopy(rag_result)

    updates = build_rag_answer_updates(state, rag_result)

    assert state == original_state
    assert rag_result == original_rag_result
    assert updates["status"] == "ok"
    assert updates["final_answer"] == "这是 RAG 答案"
    assert updates["references"] == [{"reference_id": "Reference 1"}]
    assert updates["answer_llm_used"] is True
    assert updates["llm_call_count"] == 1
    assert updates["tool_trace"] is not state["tool_trace"]
    assert updates["tool_trace"][-1]["mode"] == "rag"
    assert updates["tool_trace"][-1]["retrieval_confidence"] == 0.9
    assert updates["tool_trace"][-1]["answer_confidence"] == 0.8
    assert updates["tool_trace"][-1]["timings"] == {"llm_cost_ms": 3}


def test_build_rag_answer_updates_maps_no_relevant_info():
    updates = build_rag_answer_updates(
        _state(),
        {
            "status": "NO_RELEVANT_INFO",
            "answer": "我无法根据提供的文档确定答案",
            "references": [],
            "reason": "检索结果为空或证据分数低于阈值",
            "llm_call_count": 0,
        },
    )

    assert updates["status"] == "no_evidence"
    assert updates["final_answer"] == "我无法根据提供的文档确定答案"


def test_compose_keeps_clarify_state_and_does_not_mutate_trace():
    state = _state(
        status="clarify",
        final_answer="你说的这个具体指哪份文档？",
        reason="问题需要澄清",
        tool_trace=[{"step": "clarify"}],
    )
    original = deepcopy(state)

    updates = build_compose_answer_updates(state)

    assert state == original
    assert updates["status"] == "clarify"
    assert updates["final_answer"] == "你说的这个具体指哪份文档？"
    assert updates["tool_trace"] is not state["tool_trace"]
    assert updates["tool_trace"][-1]["mode"] == "clarify"


def test_compose_generates_direct_answer_instead_of_using_reason():
    state = _state(
        status="direct_answer",
        reason="用户询问系统能力，属于 direct_answer 范畴",
    )

    updates = build_compose_answer_updates(state)

    assert updates["final_answer"] == DIRECT_ANSWER_TEXT
    assert updates["final_answer"] != updates["reason"]
    assert updates["tool_trace"][-1]["mode"] == "direct"


def test_compose_keeps_framework_agent_answer():
    updates = build_compose_answer_updates(
        _state(
            status="ok",
            final_answer="计算结果是 7。",
            reason="Agent model produced final answer",
        )
    )

    assert updates["final_answer"] == "计算结果是 7。"
    assert updates["tool_trace"][-1]["mode"] == "agent_loop"


def test_compose_keeps_tool_error_semantics():
    updates = build_compose_answer_updates(
        _state(status="tool_error", reason="mock tool failed")
    )

    assert updates["status"] == "tool_error"
    assert "mock tool failed" in updates["final_answer"]
    assert updates["tool_trace"][-1]["mode"] == "error"


def test_compose_uses_stable_model_error_text():
    updates = build_compose_answer_updates(
        _state(status="model_error", reason="agent_model_call_failed")
    )

    assert updates["status"] == "model_error"
    assert (
        updates["final_answer"]
        == "Agent 模型暂时不可用，无法安全选择或继续调用工具。"
    )


def test_unexpected_state_fails_closed():
    updates = build_compose_answer_updates(_state())

    assert updates["status"] == "error"
    assert updates["reason"] == "Unexpected Agent state before compose: running"
    assert updates["final_answer"] == INTERNAL_ERROR_TEXT
    assert updates["tool_trace"][-1]["mode"] == "error"

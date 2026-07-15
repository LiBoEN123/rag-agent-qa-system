import pytest

from app.agent.answer import (
    DIRECT_ANSWER_TEXT,
    apply_rag_result_to_state,
    compose_final_answer,
    map_rag_status,
)
from app.agent.state import AgentObservation, AgentState


def test_apply_rag_result_to_state_maps_graph_result():
    """验证 RAG 子图结果会完整写回答案、引用、计数和 trace。"""
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
    assert result.tool_trace[-1]["step"] == "compose_final_answer"
    assert result.tool_trace[-1]["mode"] == "rag"
    assert result.tool_trace[-1]["retrieval_confidence"] == 0.9
    assert result.tool_trace[-1]["answer_confidence"] == 0.8
    assert result.tool_trace[-1]["timings"] == {"llm_cost_ms": 3}


def test_apply_rag_result_to_state_maps_no_relevant_info():
    """验证 RAG 子图的无证据状态映射为 Agent no_evidence。"""
    state = AgentState(
        original_question="测试问题",
    )
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


def test_compose_rejects_successful_knowledge_base_observation():
    """成功知识库 observation 必须由 LangGraph RAG 子图处理。"""
    state = AgentState(
        original_question="测试问题",
        observations=[
            AgentObservation(
                step_id=1,
                tool_name="search_knowledge_base",
                success=True,
                output={"chunks": []},
            )
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="Successful knowledge-base observations must be handled",
    ):
        compose_final_answer(state)


def test_compose_tool_answer_uses_format_tool_answer(monkeypatch):
    """验证普通工具观察结果会调用 format_tool_answer 生成最终回答。"""
    called = {}

    def fake_format_tool_answer(tool_name, output):
        """模拟普通工具结果格式化，并记录调用参数。"""
        called["tool_name"] = tool_name
        called["output"] = output
        return "计算结果是：7"

    monkeypatch.setattr("app.agent.answer._format_tool_answer", fake_format_tool_answer)

    state = AgentState(
        original_question="帮我算 1+2*3",
        observations=[
            AgentObservation(
                step_id=1,
                tool_name="simple_calculator",
                success=True,
                output={"expression": "1+2*3", "result": 7},
            )
        ],
    )

    result = compose_final_answer(state)

    assert called["tool_name"] == "simple_calculator"
    assert called["output"] == {"expression": "1+2*3", "result": 7}
    assert result.status == "ok"
    assert result.final_answer == "计算结果是：7"
    assert result.references == []
    assert result.tool_trace[-1]["mode"] == "tool"


def test_compose_tool_answer_uses_last_successful_observation(monkeypatch):
    """验证多个普通工具观察结果中会使用最后一个成功结果。"""
    def fake_format_tool_answer(tool_name, output):
        """模拟普通工具结果格式化，便于断言使用的观察结果。"""
        return f"{tool_name}: {output.get('value')}"

    monkeypatch.setattr("app.agent.answer._format_tool_answer", fake_format_tool_answer)

    state = AgentState(
        original_question="测试",
        observations=[
            AgentObservation(
                step_id=1,
                tool_name="list_documents",
                success=True,
                output={"value": "first"},
            ),
            AgentObservation(
                step_id=2,
                tool_name="simple_calculator",
                success=True,
                output={"value": "second"},
            ),
        ],
    )

    result = compose_final_answer(state)

    assert result.final_answer == "simple_calculator: second"


def test_compose_tool_answer_handles_format_exception(monkeypatch):
    """验证工具答案格式化异常会转为 error 状态。"""
    def fake_format_tool_answer(tool_name, output):
        """模拟工具答案格式化失败。"""
        raise RuntimeError("format failed")

    monkeypatch.setattr("app.agent.answer._format_tool_answer", fake_format_tool_answer)

    state = AgentState(
        original_question="测试",
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
    assert "format failed" in result.reason
    assert result.final_answer == "工具结果格式化失败，无法生成可靠回答。"


def test_compose_final_answer_keeps_clarify_state():
    """验证 clarify 状态的已有最终回答不会被覆盖。"""
    state = AgentState(
        original_question="这个怎么弄？",
        status="clarify",
        final_answer="你说的这个具体指哪份文档？",
        reason="问题需要澄清",
    )

    result = compose_final_answer(state)

    assert result.status == "clarify"
    assert result.final_answer == "你说的这个具体指哪份文档？"


def test_compose_final_answer_keeps_direct_answer_state():
    """验证 direct_answer 状态的已有最终回答不会被覆盖。"""
    state = AgentState(
        original_question="你能做什么？",
        status="direct_answer",
        final_answer="我是一个知识库问答助手。",
        reason="可以直接回答",
    )

    result = compose_final_answer(state)

    assert result.status == "direct_answer"
    assert result.final_answer == "我是一个知识库问答助手。"


def test_compose_final_answer_generates_direct_answer_instead_of_using_reason():
    """验证 direct_answer 会生成用户回答，不会把路由理由暴露为答案。"""
    state = AgentState(
        original_question="你能做什么？",
        status="direct_answer",
        reason="用户询问系统能力，属于 direct_answer 范畴",
    )

    result = compose_final_answer(state)

    assert result.final_answer == DIRECT_ANSWER_TEXT
    assert result.final_answer != result.reason
    assert result.tool_trace[-1]["mode"] == "direct"


def test_compose_final_answer_keeps_tool_error():
    """验证 tool_error 状态不会被 Composer 伪造成正常答案。"""
    state = AgentState(
        original_question="测试问题",
        status="tool_error",
        reason="mock tool failed",
    )

    result = compose_final_answer(state)

    assert result.status == "tool_error"
    assert "mock tool failed" in result.final_answer


def test_compose_final_answer_without_observations_returns_error():
    """验证没有任何观察结果时 Composer 会返回 error。"""
    state = AgentState(original_question="测试问题")

    result = compose_final_answer(state)

    assert result.status == "error"
    assert result.reason == "No observations available for final answer"
    assert result.final_answer == "当前没有可用于生成回答的工具结果。"
    assert result.tool_trace[-1]["step"] == "compose_final_answer"


def test_compose_final_answer_without_successful_observation_returns_tool_error():
    """验证没有成功观察结果时 Composer 会返回 tool_error。"""
    state = AgentState(
        original_question="测试问题",
        observations=[
            AgentObservation(
                step_id=1,
                tool_name="simple_calculator",
                success=False,
                error="failed",
                output={},
            )
        ],
    )

    result = compose_final_answer(state)

    assert result.status == "tool_error"
    assert result.final_answer == "工具没有返回可用结果，无法生成可靠回答。"


def test_map_rag_status():
    """验证 RAG 状态到 Agent 状态的映射规则。"""
    assert map_rag_status("OK") == "ok"
    assert map_rag_status("NEED_CLARIFICATION") == "clarify"
    assert map_rag_status("NO_RELEVANT_INFO") == "no_evidence"
    assert map_rag_status("MODEL_ERROR") == "model_error"
    assert map_rag_status("MODEL_BAD_OUTPUT") == "model_error"
    assert map_rag_status("unexpected") == "error"

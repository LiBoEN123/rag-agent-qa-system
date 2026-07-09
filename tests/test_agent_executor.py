from app.agent.executor import execute_plan
from app.agent.state import AgentPlanStep, AgentState


def test_execute_single_tool_success(monkeypatch):
    def fake_dispatch_tool(tool_name, tool_args=None, trace_id=""):
        assert tool_name == "search_knowledge_base"
        assert tool_args == {"query": "测试问题", "top_k": 5}
        assert trace_id == "trace_001"
        return {
            "status": "success",
            "ok": True,
            "tool_name": tool_name,
            "tool_input": tool_args,
            "output": {
                "count": 1,
                "chunks": [
                    {
                        "chunk_id": 1,
                        "text": "测试 chunk",
                        "hybrid_score": 0.9,
                        "bm25_score": 1.0,
                        "vector_score": 0.8,
                        "rerank_score": 0.7,
                        "file_name": "test.md",
                    }
                ],
                "top_chunk_ids": [1],
                "top_scores": [0.9],
            },
            "error": "",
        }

    monkeypatch.setattr("app.agent.executor.dispatch_tool", fake_dispatch_tool)

    state = AgentState(
        trace_id="trace_001",
        original_question="测试问题",
        plan=[
            AgentPlanStep(
                step_id=1,
                action="tool_call",
                tool_name="search_knowledge_base",
                tool_args={"query": "测试问题", "top_k": 5},
            ),
            AgentPlanStep(
                step_id=2,
                action="final_answer",
                reason="根据检索证据生成最终回答",
            ),
        ],
    )

    result = execute_plan(state)

    assert result.status == "ok"
    assert result.reason == "Agent plan executed"
    assert result.plan[0].status == "success"
    assert result.plan[1].status == "skipped"
    assert len(result.observations) == 1
    assert result.observations[0].success is True
    assert result.observations[0].output["count"] == 1
    assert len(result.tool_trace) >= 2
    assert result.tool_trace[0]["step"] == "tool_call"
    assert result.tool_trace[0]["tool_name"] == "search_knowledge_base"
    assert result.tool_trace[0]["success"] is True
    assert result.tool_trace[0]["final_chunk_count"] == 1
    assert result.tool_trace[0]["top_chunk_ids"] == [1]


def test_execute_tool_failure_stops_plan(monkeypatch):
    def fake_dispatch_tool(tool_name, tool_args=None, trace_id=""):
        return {
            "status": "error",
            "ok": False,
            "tool_name": tool_name,
            "tool_input": tool_args or {},
            "output": {},
            "error": "mock tool failed",
        }

    monkeypatch.setattr("app.agent.executor.dispatch_tool", fake_dispatch_tool)

    state = AgentState(
        original_question="测试问题",
        plan=[
            AgentPlanStep(
                step_id=1,
                action="tool_call",
                tool_name="search_knowledge_base",
                tool_args={"query": "测试问题", "top_k": 5},
            ),
            AgentPlanStep(
                step_id=2,
                action="final_answer",
            ),
        ],
    )

    result = execute_plan(state)

    assert result.status == "tool_error"
    assert result.reason == "mock tool failed"
    assert result.plan[0].status == "failed"
    assert result.plan[1].status == "pending"
    assert len(result.observations) == 1
    assert result.observations[0].success is False
    assert result.tool_trace[0]["success"] is False
    assert "mock tool failed" in result.tool_trace[0]["error"]


def test_execute_tool_exception_becomes_tool_error(monkeypatch):
    def fake_dispatch_tool(tool_name, tool_args=None, trace_id=""):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.agent.executor.dispatch_tool", fake_dispatch_tool)

    state = AgentState(
        original_question="测试问题",
        plan=[
            AgentPlanStep(
                step_id=1,
                action="tool_call",
                tool_name="search_knowledge_base",
                tool_args={"query": "测试问题", "top_k": 5},
            )
        ],
    )

    result = execute_plan(state)

    assert result.status == "tool_error"
    assert "boom" in result.reason
    assert result.plan[0].status == "failed"
    assert result.observations[0].success is False


def test_execute_clarify_step_stops():
    state = AgentState(
        original_question="这个怎么弄？",
        plan=[
            AgentPlanStep(
                step_id=1,
                action="clarify",
                reason="你说的这个具体指哪份文档？",
            ),
            AgentPlanStep(
                step_id=2,
                action="tool_call",
                tool_name="search_knowledge_base",
                tool_args={"query": "这个怎么弄？", "top_k": 5},
            ),
        ],
    )

    result = execute_plan(state)

    assert result.status == "clarify"
    assert result.final_answer == "你说的这个具体指哪份文档？"
    assert result.plan[0].status == "success"
    assert result.plan[1].status == "pending"
    assert result.tool_trace[0]["step"] == "clarify"


def test_execute_direct_answer_step_stops():
    state = AgentState(
        original_question="你能做什么？",
        plan=[
            AgentPlanStep(
                step_id=1,
                action="direct_answer",
                reason="我是一个知识库问答助手。",
            )
        ],
    )

    result = execute_plan(state)

    assert result.status == "direct_answer"
    assert result.final_answer == "我是一个知识库问答助手。"
    assert result.plan[0].status == "success"
    assert result.tool_trace[0]["step"] == "direct_answer"


def test_execute_final_answer_step_is_skipped():
    state = AgentState(
        original_question="测试问题",
        plan=[
            AgentPlanStep(
                step_id=1,
                action="final_answer",
                reason="后续生成最终答案",
            )
        ],
    )

    result = execute_plan(state)

    assert result.status == "ok"
    assert result.final_answer == ""
    assert result.plan[0].status == "skipped"
    assert result.tool_trace[0]["step"] == "final_answer"
    assert result.tool_trace[0]["status"] == "skipped"


def test_execute_empty_plan_returns_error():
    state = AgentState(original_question="测试问题")

    result = execute_plan(state)

    assert result.status == "error"
    assert result.reason == "Agent plan is empty"
    assert result.tool_trace[0]["step"] == "executor"


def test_execute_unknown_action_returns_error():
    state = AgentState(
        original_question="测试问题",
        plan=[
            AgentPlanStep(
                step_id=1,
                action="unknown_action",
            )
        ],
    )

    result = execute_plan(state)

    assert result.status == "error"
    assert "Unknown agent step action" in result.reason
    assert result.plan[0].status == "failed"
    assert result.tool_trace[0]["step"] == "unknown_action"


def test_execute_respects_max_steps(monkeypatch):
    def fake_dispatch_tool(tool_name, tool_args=None, trace_id=""):
        return {
            "status": "success",
            "ok": True,
            "tool_name": tool_name,
            "tool_input": tool_args or {},
            "output": {"count": 0},
            "error": "",
        }

    monkeypatch.setattr("app.agent.executor.dispatch_tool", fake_dispatch_tool)

    state = AgentState(
        original_question="测试问题",
        max_steps=1,
        plan=[
            AgentPlanStep(
                step_id=1,
                action="tool_call",
                tool_name="list_documents",
                tool_args={},
            ),
            AgentPlanStep(
                step_id=2,
                action="final_answer",
            ),
        ],
    )

    result = execute_plan(state)

    assert result.plan[0].status == "success"
    assert result.plan[1].status == "pending"
    assert result.status == "error"
    assert result.reason == "Agent exceeded max_steps"


def test_execute_tool_trace_handles_missing_fields(monkeypatch):
    def fake_dispatch_tool(tool_name, tool_args=None, trace_id=""):
        return {
            "status": "success",
            "ok": True,
            "tool_name": tool_name,
            "tool_input": tool_args or {},
            "output": {},
            "error": "",
        }

    monkeypatch.setattr("app.agent.executor.dispatch_tool", fake_dispatch_tool)

    state = AgentState(
        original_question="测试问题",
        plan=[
            AgentPlanStep(
                step_id=1,
                action="tool_call",
                tool_name="search_knowledge_base",
                tool_args={"query": "测试问题", "top_k": 5},
            )
        ],
    )

    result = execute_plan(state)

    assert result.status == "ok"
    assert result.tool_trace[0]["success"] is True
    assert result.tool_trace[0]["final_chunk_count"] == 0
    assert result.tool_trace[0]["top_chunk_ids"] == []

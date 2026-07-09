from app.agent.runtime import map_agent_status_to_public_status, run_agent_v2
from app.agent.state import AgentPlanStep, AgentState


def test_run_agent_v2_happy_path(monkeypatch):
    """验证 Runtime 按 route -> plan -> execute -> compose 的顺序串联。"""
    calls = []

    fake_decision = {
        "decision": "use_tool",
        "action": "simple_calculator",
        "tool_name": "simple_calculator",
        "tool_args": {"expression": "1+2*3"},
        "reason": "需要计算",
        "router_source": "rule_based",
        "router_llm_used": False,
    }

    def fake_route(question, top_k, trace_id=""):
        """模拟 Router 返回计算器决策。"""
        calls.append("route")
        return fake_decision

    def fake_build_plan(decision, question, top_k):
        """模拟 Planner 生成工具调用和最终回答步骤。"""
        calls.append("plan")
        return [
            AgentPlanStep(
                step_id=1,
                action="tool_call",
                tool_name="simple_calculator",
                tool_args={"expression": "1+2*3"},
            ),
            AgentPlanStep(
                step_id=2,
                action="final_answer",
            ),
        ]

    def fake_execute(state):
        """模拟 Executor 执行计划并校验状态对象。"""
        calls.append("execute")
        assert isinstance(state, AgentState)
        assert state.session_id == "session_001"
        state.status = "ok"
        state.reason = "Agent plan executed"
        return state

    def fake_compose(state):
        """模拟 Composer 写入最终答案。"""
        calls.append("compose")
        state.status = "ok"
        state.final_answer = "计算结果是：7"
        state.reason = "Tool result formatted as final answer"
        state.answer_llm_used = False
        return state

    monkeypatch.setattr("app.agent.runtime._route_agent_action", fake_route)
    monkeypatch.setattr("app.agent.runtime.build_plan_from_decision", fake_build_plan)
    monkeypatch.setattr("app.agent.runtime.execute_plan", fake_execute)
    monkeypatch.setattr("app.agent.runtime.compose_final_answer", fake_compose)

    result = run_agent_v2(
        write_bad_cases=lambda record: None,
        question="帮我算 1+2*3",
        top_k=5,
        decompose_top_k=10,
        trace_id="trace_001",
        session_id="session_001",
    )

    assert calls == ["route", "plan", "execute", "compose"]
    assert result["status"] == "OK"
    assert result["answer"] == "计算结果是：7"
    assert result["references"] == []
    assert result["action"] == "simple_calculator"
    assert result["router_llm_used"] is False
    assert result["answer_llm_used"] is False
    assert any(step.get("step") == "decide" for step in result["tool_trace"])
    assert any(step.get("step") == "plan" for step in result["tool_trace"])
    assert any(step.get("step") == "llm_usage" for step in result["tool_trace"])


def test_run_agent_v2_uses_provided_decision(monkeypatch):
    """验证外部传入 decision 时不会重复调用 Router。"""
    called = {"route": 0}

    def fake_route(question, top_k, trace_id=""):
        """如果被调用则说明 Runtime 没有复用外部 decision。"""
        called["route"] += 1
        raise AssertionError("route should not be called")

    def fake_build_plan(decision, question, top_k):
        """模拟 Planner 生成最小最终回答计划。"""
        return [AgentPlanStep(step_id=1, action="final_answer")]

    def fake_execute(state):
        """模拟 Executor 正常结束。"""
        state.status = "ok"
        return state

    def fake_compose(state):
        """模拟 Composer 生成简单答案。"""
        state.status = "ok"
        state.final_answer = "ok"
        return state

    monkeypatch.setattr("app.agent.runtime._route_agent_action", fake_route)
    monkeypatch.setattr("app.agent.runtime.build_plan_from_decision", fake_build_plan)
    monkeypatch.setattr("app.agent.runtime.execute_plan", fake_execute)
    monkeypatch.setattr("app.agent.runtime.compose_final_answer", fake_compose)

    decision = {
        "decision": "direct_answer",
        "action": "direct_answer",
        "reason": "可以直接回答",
        "router_llm_used": False,
    }

    result = run_agent_v2(
        write_bad_cases=lambda record: None,
        question="你能做什么？",
        top_k=5,
        decompose_top_k=10,
        decision=decision,
        trace_id="trace_002",
    )

    assert called["route"] == 0
    assert result["answer"] == "ok"
    assert result["action"] == "direct_answer"


def test_run_agent_v2_maps_clarify_status(monkeypatch):
    """验证 clarify 内部状态会映射为旧业务状态。"""
    decision = {
        "decision": "clarify",
        "action": "clarify",
        "clarification_question": "请补充具体文档。",
        "reason": "问题需要澄清",
        "router_llm_used": False,
    }

    monkeypatch.setattr(
        "app.agent.runtime._route_agent_action",
        lambda question, top_k, trace_id="": decision,
    )
    monkeypatch.setattr(
        "app.agent.runtime.build_plan_from_decision",
        lambda decision, question, top_k: [
            AgentPlanStep(step_id=1, action="clarify", reason="请补充具体文档。")
        ],
    )

    def fake_execute(state):
        """模拟 Executor 进入澄清状态。"""
        state.status = "clarify"
        state.final_answer = "请补充具体文档。"
        state.reason = "请补充具体文档。"
        return state

    monkeypatch.setattr("app.agent.runtime.execute_plan", fake_execute)
    monkeypatch.setattr("app.agent.runtime.compose_final_answer", lambda state: state)

    result = run_agent_v2(
        write_bad_cases=lambda record: None,
        question="这个怎么弄？",
        top_k=5,
        decompose_top_k=10,
        trace_id="trace_003",
    )

    assert result["status"] == "NEED_CLARIFICATION"
    assert result["need_clarification"] is True
    assert result["clarification_question"] == "请补充具体文档。"


def test_run_agent_v2_maps_no_evidence_status(monkeypatch):
    """验证 no_evidence 内部状态会映射为旧无证据状态。"""
    decision = {
        "decision": "use_knowledge_base",
        "action": "search_knowledge_base",
        "tool_name": "search_knowledge_base",
        "tool_args": {"query": "测试", "top_k": 5},
        "router_llm_used": False,
    }

    monkeypatch.setattr("app.agent.runtime._route_agent_action", lambda question, top_k, trace_id="": decision)
    monkeypatch.setattr("app.agent.runtime.build_plan_from_decision", lambda decision, question, top_k: [])
    monkeypatch.setattr("app.agent.runtime.execute_plan", lambda state: state)

    def fake_compose(state):
        """模拟 Composer 判断证据不足。"""
        state.status = "no_evidence"
        state.final_answer = "我无法根据提供的文档确定答案"
        state.reason = "证据不足"
        return state

    monkeypatch.setattr("app.agent.runtime.compose_final_answer", fake_compose)

    result = run_agent_v2(
        write_bad_cases=lambda record: None,
        question="测试",
        top_k=5,
        decompose_top_k=10,
    )

    assert result["status"] == "NO_RELEVANT_INFO"
    assert result["answer"] == "我无法根据提供的文档确定答案"


def test_run_agent_v2_counts_router_llm(monkeypatch):
    """验证 Router LLM 计数会补入最终 llm_call_count。"""
    decision = {
        "decision": "use_knowledge_base",
        "action": "search_knowledge_base",
        "tool_name": "search_knowledge_base",
        "tool_args": {"query": "测试", "top_k": 5},
        "router_source": "llm",
        "router_llm_used": True,
    }

    monkeypatch.setattr("app.agent.runtime._route_agent_action", lambda question, top_k, trace_id="": decision)
    monkeypatch.setattr("app.agent.runtime.build_plan_from_decision", lambda decision, question, top_k: [])
    monkeypatch.setattr("app.agent.runtime.execute_plan", lambda state: state)

    def fake_compose(state):
        """模拟 Composer 已经统计一次答案 LLM 调用。"""
        state.status = "ok"
        state.final_answer = "答案"
        state.answer_llm_used = True
        state.llm_call_count = 1
        return state

    monkeypatch.setattr("app.agent.runtime.compose_final_answer", fake_compose)

    result = run_agent_v2(
        write_bad_cases=lambda record: None,
        question="测试",
        top_k=5,
        decompose_top_k=10,
    )

    assert result["router_llm_used"] is True
    assert result["answer_llm_used"] is True
    assert result["llm_call_count"] >= 2
    assert result["tool_trace"][-1]["step"] == "llm_usage"
    assert result["tool_trace"][-1]["llm_call_count"] >= 2


def test_run_agent_v2_handles_internal_error(monkeypatch):
    """验证 Runtime 内部异常会返回兼容错误结构并写 bad case。"""
    bad_cases = []

    def fake_route(question, top_k, trace_id=""):
        """模拟 Router 抛出 Runtime 顶层异常。"""
        raise RuntimeError("router exploded")

    monkeypatch.setattr("app.agent.runtime._route_agent_action", fake_route)

    result = run_agent_v2(
        write_bad_cases=lambda record: bad_cases.append(record),
        question="测试",
        top_k=5,
        decompose_top_k=10,
        trace_id="trace_error",
    )

    assert result["status"] == "INTERNAL_ERROR"
    assert "Agent v2" in result["answer"]
    assert "router exploded" in result["reason"]
    assert result["references"] == []
    assert result["tool_trace"][0]["step"] == "agent_v2_error"
    assert len(bad_cases) == 1
    assert bad_cases[0]["trace_id"] == "trace_error"
    assert bad_cases[0]["error_type"] == "internal_error"


def test_map_agent_status_to_public_status():
    """验证 v2 内部状态到 API 业务状态的映射。"""
    assert map_agent_status_to_public_status("ok") == "OK"
    assert map_agent_status_to_public_status("clarify") == "NEED_CLARIFICATION"
    assert map_agent_status_to_public_status("direct_answer") == "OK"
    assert map_agent_status_to_public_status("tool_error") == "TOOL_ERROR"
    assert map_agent_status_to_public_status("model_error") == "MODEL_ERROR"
    assert map_agent_status_to_public_status("no_evidence") == "NO_RELEVANT_INFO"
    assert map_agent_status_to_public_status("error") == "INTERNAL_ERROR"


def test_run_agent_v2_saves_trace_when_enabled(monkeypatch):
    """验证 trace 开关开启时 Runtime 保存最终结果和执行步骤。"""
    from app.agent import runtime as agent_runtime

    saved = {}
    decision = {
        "decision": "direct_answer",
        "action": "direct_answer",
        "reason": "可以直接回答",
        "router_llm_used": False,
    }

    monkeypatch.setattr(
        agent_runtime.Settings,
        "ENABLE_AGENT_TRACE_STORE",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        agent_runtime,
        "_route_agent_action",
        lambda question, top_k, trace_id="": decision,
    )
    monkeypatch.setattr(
        agent_runtime,
        "build_plan_from_decision",
        lambda decision, question, top_k: [
            AgentPlanStep(
                step_id=1,
                action="direct_answer",
                reason="可以直接回答",
            )
        ],
    )

    def fake_execute(state):
        """模拟直接回答步骤执行完成。"""
        state.status = "direct_answer"
        state.final_answer = "可以直接回答"
        state.reason = "可以直接回答"
        return state

    monkeypatch.setattr(agent_runtime, "execute_plan", fake_execute)
    monkeypatch.setattr(agent_runtime, "compose_final_answer", lambda state: state)
    monkeypatch.setattr(
        agent_runtime,
        "save_agent_trace",
        lambda **kwargs: saved.update(kwargs),
    )

    result = agent_runtime.run_agent_v2(
        write_bad_cases=lambda record: None,
        question="你能做什么？",
        top_k=5,
        decompose_top_k=10,
        trace_id="trace_save",
        session_id="session_save",
    )

    assert result["status"] == "OK"
    assert saved["trace_id"] == "trace_save"
    assert saved["session_id"] == "session_save"
    assert saved["question"] == "你能做什么？"
    assert saved["final_answer"] == "可以直接回答"
    assert saved["tool_trace"]

"""Current metrics-schema contracts for summarize_logs."""

from scripts import summarize_logs


def test_summary_uses_current_status_and_llm_fields():
    rows = [
        {
            "status": "success",
            "action": "agent",
            "latency_ms": 100,
            "reference_count": 2,
            "tool_called": True,
            "tool_success": True,
            "tool_name": "search_knowledge_base",
            "router_llm_used": False,
            "decompose_llm_used": True,
            "answer_llm_used": True,
            "llm_call_count": 2,
        },
        {
            "status": "clarify",
            "action": "agent",
            "latency_ms": 50,
            "reference_count": 0,
            "tool_called": False,
            "tool_success": False,
            "tool_name": "",
            "router_llm_used": False,
            "decompose_llm_used": False,
            "answer_llm_used": False,
            "llm_call_count": 0,
        },
    ]

    summary = summarize_logs.build_summary(rows, [], [], [], top_n=5)

    assert summary["success_count"] == 1
    assert summary["refuse_count"] == 1
    assert summary["avg_reference_count"] == 1
    assert summary["router_llm_used_count"] == 0
    assert summary["decompose_llm_used_count"] == 1
    assert summary["answer_llm_used_count"] == 1
    assert summary["total_llm_call_count"] == 2
    assert summary["error_type_counter"] == {"clarification": 1}


def test_error_fallback_does_not_infer_removed_planner_semantics():
    row = {
        "status": "unknown",
        "action": "refuse",
        "reason": "旧 Planner 文案：检索结果为空",
        "question_len": 10,
    }

    assert summarize_logs.infer_error_type_from_metric(row) == "unknown"

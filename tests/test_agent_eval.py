from scripts import agent_eval
from scripts.agent_eval import calc_metrics, check_tool_output_keys, evaluate_one_case


def test_check_tool_output_keys_reads_tool_result_output():
    """验证工具输出字段检查可以读取 result.tool_result.output。"""
    case = {
        "expected_tool_output_keys": [
            "doc_a_evidence",
            "doc_b_evidence",
            "comparison",
        ]
    }
    result = {
        "tool_result": {
            "output": {
                "doc_a_evidence": [],
                "doc_b_evidence": [],
                "comparison": "compared",
            }
        }
    }

    check = check_tool_output_keys(case, result)

    assert check["tool_output_keys_pass"] is True
    assert check["tool_output_keys_missing"] == []


def test_check_tool_output_keys_reports_missing_from_trace_output():
    """验证工具输出字段检查能报告 trace 输出里的缺失字段。"""
    case = {"expected_tool_output_keys": ["comparison", "conflict_detected"]}
    result = {
        "tool_trace": [
            {
                "step": "call_tool",
                "tool_name": "compare_documents",
                "tool_output": {"comparison": "compared"},
            }
        ]
    }

    check = check_tool_output_keys(case, result)

    assert check["tool_output_keys_pass"] is False
    assert check["tool_output_keys_missing"] == ["conflict_detected"]


def test_refuse_metrics_use_final_status_not_refuse_action():
    """验证拒答指标基于最终状态统计，而不依赖 Router action。"""
    metrics = calc_metrics([
        {
            "passed": True,
            "category": "refuse",
            "expected_decision": "",
            "actual_decision": "use_knowledge_base",
            "actual_action": "search_knowledge_base",
            "expected_tool": "",
            "actual_tool": "",
            "actual_status": "NO_RELEVANT_INFO",
            "keyword_pass": True,
            "source_pass": True,
        }
    ])

    assert metrics["refuse_success_count"] == 1
    assert metrics["refuse_success_rate"] == 1.0


def test_evaluate_one_case_checks_tool_output_keys(monkeypatch):
    """验证单条 Agent Eval 会把工具输出字段检查纳入通过判定。"""
    def fake_run_agent_once(**kwargs):
        """模拟 compare_documents 的 Agent 返回结果。"""
        return {
            "status": "OK",
            "action": "compare_documents",
            "answer": "两个文档关于最长学习年限存在差异。",
            "reason": "Agent 选择工具：compare_documents",
            "tool_result_summary": "compare_documents 执行成功",
            "references": [],
            "tool_result": {
                "output": {
                    "doc_a_evidence": [],
                    "doc_b_evidence": [],
                    "comparison": "存在差异",
                    "conflict_detected": True,
                }
            },
            "tool_trace": [
                {
                    "step": "decide",
                    "decision_type": "use_tool",
                    "action": "compare_documents",
                    "tool_name": "compare_documents",
                }
            ],
        }

    monkeypatch.setattr(agent_eval, "run_agent_once", fake_run_agent_once)

    row = evaluate_one_case(
        {
            "id": "compare_documents_test",
            "category": "compare_documents",
            "question": "请对比两个文档的最长学习年限差异",
            "expected_decision": "use_tool",
            "expected_action": "compare_documents",
            "expected_tool": "compare_documents",
            "expected_status": "OK",
            "expected_keywords": ["最长学习年限", "差异"],
            "expected_tool_output_keys": [
                "doc_a_evidence",
                "doc_b_evidence",
                "comparison",
                "conflict_detected",
            ],
        },
        top_k=3,
    )

    assert row["passed"] is True
    assert row["error_type"] == "ok"
    assert row["tool_output_keys_pass"] is True

import json
from app.agent import route_agent_action


TEST_CASES = [
    {
        "question": "你能做什么?",
        "expected_decision": "direct_answer",
        "expected_tool_name": ""
    },
    {
        "question": "知识库里有哪些文档？",
        "expected_decision": "use_tool",
        "expected_tool_name": "list_documents"
    },
    {
        "question": "现在知识库里有多少个文档？",
        "expected_decision": "use_tool",
        "expected_tool_name": "list_documents"
    },
    {
        "question": "1 + 2 * 3",
        "expected_decision": "use_tool",
        "expected_tool_name": "simple_calculator"
    },
    {
        "question": "请计算 10 / 2",
        "expected_decision": "use_tool",
        "expected_tool_name": "simple_calculator"
    },
    {
        "question": "查看 chunk_id 为 1 的内容",
        "expected_decision": "use_tool",
        "expected_tool_name": "get_chunk_text"
    },
    {
        "question": "每个 Vue 应用是通过什么函数创建应用实例的？",
        "expected_decision": "direct_answer",
        "expected_tool_name": ""
    },
    {
        "question": "今天天气怎么样？",
        "expected_decision": "refuse",
        "expected_tool_name": ""
    },
    {
        "question": "给我讲个笑话",
        "expected_decision": "refuse",
        "expected_tool_name": ""
    },
    {
        "question": "这个怎么弄",
        "expected_decision": "clarify",
        "expected_tool_name": ""
    },
]


def run_decision_eval():
    rows = []

    for item in TEST_CASES:
        question = item["question"]
        expected_decision = item["expected_decision"]
        expected_tool_name = item["expected_tool_name"]

        actual =route_agent_action(question, top_k=5)

        actual_decision = actual.get("decision")
        actual_tool_name = actual.get("tool_name", "")

        passed = (
            actual_decision == expected_decision
            and actual_tool_name == expected_tool_name
        )

        rows.append({
            "question": question,
            "expected_decision": expected_decision,
            "actual_decision": actual_decision,
            "expected_tool_name": expected_tool_name,
            "actual_tool_name": actual_tool_name,
            "passed": passed,
            "reason": actual.get("reason", "")
        })

    for row in rows:
        print(json.dumps(row, ensure_ascii=False, indent=2))

    total = len(rows)
    passed_count = sum(1 for row in rows if row["passed"])

    print(f"\nDecision Eval: {passed_count}/{total} passed")


if __name__ == "__main__":
    run_decision_eval()
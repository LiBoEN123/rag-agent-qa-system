"""Agent 确定性规则模块。

负责规则入口仍需使用的问题规范化。
"""


SYSTEM_CAPABILITY_QUESTIONS = frozenset(
    {
        "你是谁",
        "你能做什么",
        "你可以做什么",
        "你有什么功能",
    }
)


def normalize_question(question: str) -> str:
    """清理问题首尾空白，并统一常见中文标点。"""
    return (
        (question or "")
        .strip()
        .replace("？", "?")
        .replace("：", ":")
        .replace("，", ",")
        .replace("。", "")
    )


def is_system_capability_question(question: str) -> bool:
    """判断问题是否属于可由系统直接回答的能力说明。"""
    normalized_question = normalize_question(question)
    if normalized_question.endswith("?"):
        normalized_question = normalized_question[:-1]
    return normalized_question in SYSTEM_CAPABILITY_QUESTIONS


__all__ = [
    "SYSTEM_CAPABILITY_QUESTIONS",
    "is_system_capability_question",
    "normalize_question",
]

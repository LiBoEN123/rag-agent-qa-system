"""Agent 确定性规则模块。

负责规则入口仍需使用的问题规范化。
"""


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

"""Agent 确定性规则模块。

负责问题规范化、数学表达式识别、chunk 编号提取和双文档对比参数解析。
"""

import ast
import re
import unicodedata


COMPARE_DOCUMENT_KEYWORDS = ("对比", "比较", "区别", "差异", "不同", "冲突", "一致")


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


def extract_chunk_id(question: str) -> int | None:
    """从用户问题中提取显式给出的 chunk 编号。"""
    patterns = [
        r"chunk[_\s-]?id\s*[:=]?\s*(\d+)",
        r"chunk\s*(\d+)",
        r"第\s*(\d+)\s*个\s*chunk",
        r"查看\s*(\d+)\s*号\s*chunk",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def normalize_math_text(text: str) -> str:
    """把全角字符和中文运算词规整为标准数学表达式。"""
    text = unicodedata.normalize("NFKC", text or "")
    replacements = [
        (r"加上|加", "+"),
        (r"减去|减掉|减", "-"),
        (r"乘以|乘上|乘", "*"),
        (r"除以|除", "/"),
        (r"×|x|X", "*"),
        (r"÷", "/"),
        (r"负\s*(?=\d|\()", "-"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def is_safe_math_expression(expression: str) -> bool:
    """检查表达式是否只包含允许的数字和四则运算节点。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Div,
        ast.Mult,
        ast.UAdd,
        ast.USub,
    )
    has_binary_operator = False
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False
        if isinstance(node, ast.BinOp):
            has_binary_operator = True
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return False
    return has_binary_operator


def extract_simple_math_expression(question: str) -> str | None:
    """从自然语言问题中提取可安全计算的数学表达式。"""
    candidates = re.findall(r"[0-9+\-*/().\s]+", normalize_math_text(question))
    for expression in sorted(
        (candidate.strip() for candidate in candidates if candidate.strip()),
        key=len,
        reverse=True,
    ):
        if is_safe_math_expression(expression):
            return expression
    return None


def build_compare_documents_rule_args(question: str, top_k: int) -> dict | None:
    """从带书名号的双文档对比问题中构造工具参数。"""
    normalized = normalize_question(question)
    if not any(keyword in normalized for keyword in COMPARE_DOCUMENT_KEYWORDS):
        return None
    titles = [
        title.strip()
        for title in re.findall(r"\u300a([^\u300b]+)\u300b", normalized)
        if title.strip()
    ]
    if len(titles) < 2:
        return None
    return {
        "query": normalized,
        "doc_a": titles[0],
        "doc_b": titles[1],
        "top_k": max(1, min(int(top_k or 3), 10)),
    }

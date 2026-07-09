"""Safe arithmetic tool."""

import ast
import operator as op

from app.tools.result import tool_error, tool_success


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
    ast.UAdd: lambda value: value,
}


def safe_eval_expr(expression: str) -> int | float:
    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("只支持简单数字四则运算")

    return evaluate(ast.parse(expression, mode="eval"))


def simple_calculator(expression: str) -> dict:
    tool_name = "simple_calculator"
    tool_input = {"expression": expression}
    try:
        if not expression or not str(expression).strip():
            return tool_error(tool_name, tool_input, "expression 不能为空")
        return tool_success(
            tool_name,
            tool_input,
            {"expression": expression, "result": safe_eval_expr(expression)},
        )
    except Exception as exc:
        return tool_error(tool_name, tool_input, str(exc))

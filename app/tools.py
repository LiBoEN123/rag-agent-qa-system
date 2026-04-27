import json
import time
from app.rag import preview_docs, get_chunk_full_text
import ast
import operator as op

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
    ast.UAdd: lambda x: x,
}

def safe_eval_expr(expression:str) -> int | float:

    def _eval(node):

        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            left = _eval(node.left)
            right = _eval(node.right)
            return _ALLOWED_OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            value = _eval(node.operand)
            return _ALLOWED_OPERATORS[type(node.op)](value)

        raise ValueError("只支持简单数字四则运算")

    tree = ast.parse(expression, mode='eval')
    return _eval(tree)

def simple_calculator(expression:str) -> dict:
    try:
        result = safe_eval_expr(expression)
        return {
            "ok": True,
            "result": result,
            "expression": expression,
            "error": ""
        }

    except Exception as e:
        return {
            "ok": False,
            "result": None,
            "expression": expression,
            "error": str(e)
        }

def list_documents():
    docs = preview_docs()
    return {
        "count": len(docs),
        "documents": docs
    }

def get_chunk_text(chunk_id: int | str):
    return get_chunk_full_text(chunk_id)

def append_tool_log(record: dict):
    with open("tool_calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def dispatch_tool(tool_name: str, tool_args: dict):
    start = time.time()
    try:
        if tool_name == "list_documents":
            result = list_documents()
        elif tool_name == "simple_calculator":
            result = simple_calculator(tool_args.get("expression", ""))
        elif tool_name == "get_chunk_text":
            result = get_chunk_text(tool_args.get("chunk_id"))
        else:
            result = {"error": f"unknown tool: {tool_name}"}

        success = result.get("ok", True) if isinstance(result, dict) else True
        if isinstance(result, dict) and result.get("error"):
            success = False

        append_tool_log({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": result,
            "elapsed_ms": int((time.time() - start) * 1000),
            "success": success
        })
        return result

    except Exception as e:
        err = {"error": str(e)}
        append_tool_log({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": err,
            "elapsed_ms": int((time.time() - start) * 1000),
            "success": False
        })
        return err













































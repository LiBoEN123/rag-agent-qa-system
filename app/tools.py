import ast
import json
import operator as op
import time
from pathlib import Path
from typing import Any

from app.rag import get_chunk_full_text, preview_docs


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
    ast.UAdd: lambda x: x,
}

TOOL_CALLS_PATH = Path("logs") / "tool_calls.jsonl"


# =========================
# 工具返回格式
# =========================
def tool_success(tool_name: str, tool_input: dict, output: Any) -> dict:
    """生成统一的工具成功返回结构。"""
    return {
        "status": "success",
        "ok": True,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "output": output,
        "error": ""
    }


def tool_error(tool_name: str, tool_input: dict, error: str) -> dict:
    """生成统一的工具失败返回结构。"""
    return {
        "status": "error",
        "ok": False,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "output": None,
        "error": error
    }


# =========================
# 工具实现
# =========================
def safe_eval_expr(expression: str) -> int | float:
    """只允许数字、括号和四则运算，避免执行任意代码。"""

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

    tree = ast.parse(expression, mode="eval")
    return _eval(tree)


def simple_calculator(expression: str) -> dict:
    tool_name = "simple_calculator"
    tool_input = {"expression": expression}

    try:
        if not expression or not str(expression).strip():
            return tool_error(tool_name, tool_input, "expression 不能为空")

        result = safe_eval_expr(expression)
        return tool_success(
            tool_name=tool_name,
            tool_input=tool_input,
            output={
                "expression": expression,
                "result": result
            }
        )

    except Exception as e:
        return tool_error(
            tool_name=tool_name,
            tool_input=tool_input,
            error=str(e)
        )


def list_documents() -> dict:
    tool_name = "list_documents"
    tool_input = {}

    try:
        docs = preview_docs()

        return tool_success(
            tool_name=tool_name,
            tool_input=tool_input,
            output={
                "count": len(docs),
                "documents": docs
            }
        )

    except Exception as e:
        return tool_error(
            tool_name=tool_name,
            tool_input=tool_input,
            error=str(e)
        )


def get_chunk_text(chunk_id: int | str) -> dict:
    tool_name = "get_chunk_text"
    tool_input = {"chunk_id": chunk_id}

    try:
        if not chunk_id or str(chunk_id).strip() == "":
            return tool_error(tool_name, tool_input, "chunk_id 不能为空")

        chunk = get_chunk_full_text(chunk_id)

        if not chunk.get("found"):
            return tool_error(
                tool_name,
                tool_input,
                f"未找到 chunk_id={chunk_id} 的 chunk，可能是知识库未初始化或 chunk_id 不存在"
            )

        return tool_success(
            tool_name=tool_name,
            tool_input=tool_input,
            output={
                "chunk_id": chunk_id,
                "chunk": chunk
            }
        )

    except Exception as e:
        return tool_error(
            tool_name=tool_name,
            tool_input=tool_input,
            error=str(e)
        )


# =========================
# 工具调度与日志
# =========================
def append_tool_log(record: dict) -> None:
    """每次工具调用写一行 JSON 到 logs/tool_calls.jsonl。"""
    TOOL_CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with TOOL_CALLS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def dispatch_tool(tool_name: str, tool_args: dict | None=None, trace_id: str = ""):
    """统一分发工具调用，并把 trace_id 写入工具调用日志。"""
    tool_args = tool_args or {}
    start = time.time()

    try:
        if tool_name == "list_documents":
            result = list_documents()
        elif tool_name == "simple_calculator":
            result = simple_calculator(tool_args.get("expression", ""))
        elif tool_name == "get_chunk_text":
            result = get_chunk_text(tool_args.get("chunk_id"))
        else:
            result = tool_error(
                tool_name=tool_name,
                tool_input=tool_args,
                error=f"unknown tool: {tool_name}"
            )

        append_tool_log({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trace_id": trace_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": result,
            "elapsed_ms": int((time.time() - start) * 1000),
            "success": result.get("ok", False)
        })
        return result

    except Exception as e:
        result = tool_error(
            tool_name=tool_name,
            tool_input=tool_args,
            error=str(e)
        )

        append_tool_log({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trace_id": trace_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": result,
            "elapsed_ms": int((time.time() - start) * 1000),
            "success": False
        })

        return result


# =========================
# 本地手动验证
# =========================
def run_manual_examples():
    examples = [
        {
            "tool_name": "simple_calculator",
            "tool_args": {"expression": "1 + 2 * 3"}
        },
        {
            "tool_name": "simple_calculator",
            "tool_args": {"expression": "10 / 0"}
        },
        {
            "tool_name": "list_documents",
            "tool_args": {}
        },
        {
            "tool_name": "get_chunk_text",
            "tool_args": {"chunk_id": 1}
        },
        {
            "tool_name": "unknown_tool",
            "tool_args": {}
        }
    ]

    for item in examples:
        result = dispatch_tool(item["tool_name"], item["tool_args"])
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_manual_examples()

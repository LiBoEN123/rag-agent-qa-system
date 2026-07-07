import ast
import json
import operator as op
import time
from pathlib import Path
from typing import Any


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

        """递归计算表达式 AST 节点，只允许白名单运算。"""
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
    """执行安全的简单四则运算并返回统一工具结果。"""
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
    """读取知识库文档列表并包装成统一工具结果。"""
    tool_name = "list_documents"
    tool_input = {}

    try:
        from app.rag import preview_docs

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
    """根据 chunk_id 获取完整 chunk 文本并包装成统一工具结果。"""
    tool_name = "get_chunk_text"
    tool_input = {"chunk_id": chunk_id}

    try:
        from app.rag import get_chunk_full_text

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


def _get_chunk_value(chunk: dict, key: str, default=None):
    """从 chunk 顶层或 metadata 中读取字段。"""
    if not isinstance(chunk, dict):
        return default

    value = chunk.get(key)
    if value is not None:
        return value

    metadata = chunk.get("metadata") or {}
    if isinstance(metadata, dict):
        return metadata.get(key, default)

    return default


def _round_chunk_score(chunk: dict, key: str) -> float:
    """安全读取 chunk 分数字段。"""
    try:
        return round(float(_get_chunk_value(chunk, key, 0.0) or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


def search_knowledge_base(args: dict) -> dict:
    """检索知识库候选 chunk，并保持 chunk 原始结构供后续 RAG 生成使用。"""
    tool_name = "search_knowledge_base"

    try:
        from app.config import Settings
        from app.rag import search_relevant_chunks

        args = args or {}
        query = str(args.get("query") or "").strip()
        raw_top_k = args.get("top_k", Settings.TOP_K)

        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            return tool_error(
                tool_name=tool_name,
                tool_input={"query": query, "top_k": raw_top_k},
                error="top_k 必须是整数"
            )

        top_k = max(1, min(top_k, 20))
        tool_input = {"query": query, "top_k": top_k}

        if not query:
            return tool_error(tool_name, tool_input, "query 不能为空")

        chunks = search_relevant_chunks(query, top_k=top_k)

        return tool_success(
            tool_name=tool_name,
            tool_input=tool_input,
            output={
                "count": len(chunks),
                "chunks": chunks,
                "top_chunk_ids": [
                    _get_chunk_value(chunk, "chunk_id")
                    for chunk in chunks
                ],
                "top_scores": [
                    _round_chunk_score(chunk, "hybrid_score")
                    for chunk in chunks
                ],
            }
        )

    except Exception as e:
        return tool_error(
            tool_name=tool_name,
            tool_input=args if isinstance(args, dict) else {},
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
        from app.tool_registry import tool_registry

        result = tool_registry.dispatch(tool_name, tool_args)

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

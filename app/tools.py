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


# =========================
# 文档对比工具
# =========================
def _normalize_doc_label(value: Any) -> str:
    """标准化文档名，便于文件名精确或模糊匹配。"""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = text.replace("\\", "/").split("/")[-1]
    text = text.strip(" 《》<>\"'“”‘’")
    return re.sub(r"\s+", "", text)


def _strip_known_suffix(value: str) -> str:
    """去掉常见文档扩展名，兼容用户只输入标题的情况。"""
    for suffix in [".md", ".txt", ".pdf", ".docx", ".doc"]:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _get_compare_chunk_file_name(chunk: dict) -> str:
    """从 chunk 顶层或 metadata 中提取用于文档匹配的文件名。"""
    metadata = chunk.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return str(
        chunk.get("file_name")
        or metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("source_path")
        or ""
    )


def _get_compare_chunk_id(chunk: dict):
    """从 chunk 顶层或 metadata 中提取 chunk_id。"""
    metadata = chunk.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return chunk.get("chunk_id") or metadata.get("chunk_id")


def _get_compare_chunk_text(chunk: dict) -> str:
    """从不同 chunk 字段中提取可用于证据打分的文本。"""
    return str(chunk.get("text") or chunk.get("snippet") or chunk.get("content") or "")


def _doc_name_matches(file_name: str, doc_name: str, *, exact: bool) -> bool:
    """判断 chunk 文件名是否匹配用户指定的文档名或关键词。"""
    file_label = _normalize_doc_label(file_name)
    doc_label = _normalize_doc_label(doc_name)
    if not file_label or not doc_label:
        return False

    file_no_suffix = _strip_known_suffix(file_label)
    doc_no_suffix = _strip_known_suffix(doc_label)
    if exact:
        return (
            doc_label in {file_label, file_no_suffix}
            or doc_no_suffix in {file_label, file_no_suffix}
        )

    return (
        doc_label in file_label
        or file_label in doc_label
        or doc_no_suffix in file_no_suffix
        or file_no_suffix in doc_no_suffix
    )


def _filter_chunks_by_document(chunks: list[dict], doc_name: str) -> list[dict]:
    """按文档名筛选 chunk，优先精确匹配，失败后再做包含匹配。"""
    candidates = [chunk for chunk in chunks if isinstance(chunk, dict)]
    exact_matches = [
        chunk
        for chunk in candidates
        if _doc_name_matches(_get_compare_chunk_file_name(chunk), doc_name, exact=True)
    ]
    if exact_matches:
        return exact_matches

    return [
        chunk
        for chunk in candidates
        if _doc_name_matches(_get_compare_chunk_file_name(chunk), doc_name, exact=False)
    ]


def _score_chunk_for_query(query: str, chunk_text: str) -> float:
    """基于问题关键词和 chunk 关键词重合度计算轻量相关性分数。"""
    from app.citation_verifier import extract_reference_keywords

    query_keywords = extract_reference_keywords(query)
    chunk_keywords = extract_reference_keywords(chunk_text)
    return len(query_keywords & chunk_keywords) / max(len(query_keywords), 1)


def _select_compare_evidence(
    chunks: list[dict],
    doc_name: str,
    query: str,
    top_k: int,
) -> tuple[list[dict], int]:
    """在单个文档内选择与对比主题最相关的 evidence 片段。"""
    doc_chunks = _filter_chunks_by_document(chunks, doc_name)
    scored = []
    for index, chunk in enumerate(doc_chunks):
        text = _get_compare_chunk_text(chunk)
        if not text.strip():
            continue
        score = _score_chunk_for_query(query, text)
        if score > 0:
            scored.append((score, index, chunk, text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    evidence = [
        {
            "chunk_id": _get_compare_chunk_id(chunk),
            "file_name": _get_compare_chunk_file_name(chunk),
            "score": round(score, 4),
            "snippet": text[:500],
        }
        for score, _, chunk, text in scored[:top_k]
    ]
    return evidence, len(doc_chunks)


def _format_compare_evidence_for_llm(label: str, evidence: list[dict]) -> str:
    """把 evidence 列表整理成适合 LLM 阅读的紧凑文本。"""
    blocks = []
    for index, item in enumerate(evidence, start=1):
        blocks.append(
            "\n".join([
                f"{label}_evidence_{index}:",
                f"chunk_id: {item.get('chunk_id')}",
                f"file_name: {item.get('file_name')}",
                f"score: {item.get('score')}",
                f"snippet: {item.get('snippet')}",
            ])
        )
    return "\n\n".join(blocks)


def _parse_compare_llm_output(llm_output: str) -> dict:
    """解析 compare_documents 的 LLM JSON 输出。"""
    text = (llm_output or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {"comparison": text}
        data = json.loads(text[start:end + 1])

    if not isinstance(data, dict):
        return {"comparison": str(data)}
    return data


def _coerce_optional_bool(value):
    """把 LLM 返回的布尔值或常见布尔字符串规整为 bool。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "是", "存在", "有差异"}:
            return True
        if normalized in {"false", "no", "0", "否", "不存在", "无差异"}:
            return False
    return None


def _call_compare_documents_llm(
    query: str,
    doc_a: str,
    doc_b: str,
    doc_a_evidence: list[dict],
    doc_b_evidence: list[dict],
) -> dict:
    """调用 LLM 根据两份文档 evidence 生成差异总结。"""
    import requests

    from app.config import Settings
    from app.rag_generation import build_llm_header, get_http_error_message

    system_prompt = (
        "你是制度类文档差异对比助手。"
        "你只能依据用户提供的 evidence 对比两份文档中同一问题的规定，"
        "不要引入外部知识，不要判断哪份文件更新或优先级更高。"
        "不要仅因为措辞不同就判定存在差异。"
        "必须返回严格 JSON，不要输出 Markdown。"
    )
    user_prompt = f"""
对比主题：
{query}

doc_a：
{doc_a}

doc_b：
{doc_b}

doc_a evidence：
{_format_compare_evidence_for_llm("doc_a", doc_a_evidence)}

doc_b evidence：
{_format_compare_evidence_for_llm("doc_b", doc_b_evidence)}

请返回严格 JSON，字段如下：
{{
  "comparison": "说明两份文档关于该主题的相同点和差异点；如果证据不足也要说明不足",
  "conflict_detected": true 或 false,
  "summary": "一句话摘要"
}}
""".strip()

    payload = {
        "model": Settings.MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": min(Settings.LLM_MAX_TOKENS, 1024),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_error = ""
    total_attempts = Settings.LLM_MAX_RETRIES + 1
    for attempt in range(total_attempts):
        try:
            response = requests.post(
                Settings.LLM_API_URL,
                headers=build_llm_header(),
                json=payload,
                timeout=Settings.LLM_TIMEOUT,
            )
            if response.status_code != 200:
                last_error = get_http_error_message(response.status_code)
                if response.status_code >= 500 and attempt < total_attempts - 1:
                    time.sleep(Settings.LLM_RETRY_SLEEP)
                    continue
                raise RuntimeError(f"{last_error}: {response.text[:300]}")

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return _parse_compare_llm_output(content)
        except Exception as e:
            last_error = str(e)
            if attempt < total_attempts - 1:
                time.sleep(Settings.LLM_RETRY_SLEEP)
                continue
            raise RuntimeError(last_error)

    raise RuntimeError(last_error or "compare_documents LLM 调用失败")


def compare_documents_tool(args: dict) -> dict:
    """对比两个知识库文档中关于同一问题的相关证据差异。"""
    tool_name = "compare_documents"
    args = args or {}
    try:
        query = str(args.get("query") or "").strip()
        doc_a = str(args.get("doc_a") or "").strip()
        doc_b = str(args.get("doc_b") or "").strip()
        raw_top_k = args.get("top_k", 3)
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            return tool_error(
                tool_name,
                {"query": query, "doc_a": doc_a, "doc_b": doc_b, "top_k": raw_top_k},
                "top_k 必须是整数"
            )

        top_k = max(1, min(top_k, 10))
        tool_input = {"query": query, "doc_a": doc_a, "doc_b": doc_b, "top_k": top_k}
        if not query:
            return tool_error(tool_name, tool_input, "query 不能为空")
        if not doc_a:
            return tool_error(tool_name, tool_input, "doc_a 不能为空")
        if not doc_b:
            return tool_error(tool_name, tool_input, "doc_b 不能为空")
        if (
            _strip_known_suffix(_normalize_doc_label(doc_a))
            == _strip_known_suffix(_normalize_doc_label(doc_b))
        ):
            return tool_error(tool_name, tool_input, "doc_a 和 doc_b 不能相同")

        from app import rag

        current_chunks = getattr(rag, "chunks", []) or []
        if not isinstance(current_chunks, list):
            return tool_error(tool_name, tool_input, "当前知识库 chunks 不可用")

        doc_a_evidence, doc_a_chunk_count = _select_compare_evidence(
            current_chunks,
            doc_a,
            query,
            top_k,
        )
        doc_b_evidence, doc_b_chunk_count = _select_compare_evidence(
            current_chunks,
            doc_b,
            query,
            top_k,
        )
        if doc_a_chunk_count == 0:
            return tool_error(tool_name, tool_input, "doc_a 未找到匹配文档 chunk")
        if doc_b_chunk_count == 0:
            return tool_error(tool_name, tool_input, "doc_b 未找到匹配文档 chunk")
        if not doc_a_evidence:
            return tool_error(tool_name, tool_input, "doc_a 未找到相关证据 chunk")
        if not doc_b_evidence:
            return tool_error(tool_name, tool_input, "doc_b 未找到相关证据 chunk")

        try:
            llm_result = _call_compare_documents_llm(
                query=query,
                doc_a=doc_a,
                doc_b=doc_b,
                doc_a_evidence=doc_a_evidence,
                doc_b_evidence=doc_b_evidence,
            )
            comparison = str(llm_result.get("comparison") or "").strip()
            summary = str(llm_result.get("summary") or "").strip()
            llm_conflict = _coerce_optional_bool(llm_result.get("conflict_detected"))
        except Exception as e:
            return tool_error(
                tool_name,
                tool_input,
                f"LLM 对比生成失败：{str(e)[:200]}"
            )

        if not comparison:
            return tool_error(tool_name, tool_input, "LLM 返回缺少 comparison")
        if llm_conflict is None:
            return tool_error(tool_name, tool_input, "LLM 返回缺少合法 conflict_detected")
        if not summary:
            return tool_error(tool_name, tool_input, "LLM 返回缺少 summary")

        conflict_detected = llm_conflict

        return tool_success(
            tool_name=tool_name,
            tool_input=tool_input,
            output={
                "query": query,
                "doc_a": doc_a,
                "doc_b": doc_b,
                "doc_a_evidence": doc_a_evidence,
                "doc_b_evidence": doc_b_evidence,
                "comparison": comparison,
                "conflict_detected": conflict_detected,
                "summary": summary,
            }
        )
    except Exception as e:
        return tool_error(
            tool_name=tool_name,
            tool_input=args if isinstance(args, dict) else {},
            error=str(e)
        )

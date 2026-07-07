import ast
import json
import re
import time
import unicodedata

import requests

from app.citation_verifier import verify_answer_citations
from app.config import Settings
from app.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from app.rag import ask
from app.tool_registry import tool_registry
from app.tools import dispatch_tool
from app.untils import get_logger


logger = get_logger('chat_service')

ROUTER_ALLOWED_TOOLS = set(tool_registry.names())
ROUTER_TOOL_SPECS = tool_registry.router_specs()

ROUTER_ALLOWED_ACTIONS = ROUTER_ALLOWED_TOOLS | {
    "clarify",
    "direct_answer",
}

HIGH_CONFIDENCE_RULE_ACTIONS = {
    "simple_calculator",
    "list_documents",
    "get_chunk_text",
    "compare_documents",
    "direct_answer",
    "clarify",
}

ROUTER_ACTION_TO_DECISION = {
    "search_knowledge_base": "use_knowledge_base",
    "simple_calculator": "use_tool",
    "list_documents": "use_tool",
    "get_chunk_text": "use_tool",
    "compare_documents": "use_tool",
    "clarify": "clarify",
    "direct_answer": "direct_answer",
}

DEFAULT_CLARIFICATION_QUESTION = "请补充你想查询的具体对象、文档、功能或报错信息。"


# =========================
# 路由通用工具
# =========================
def get_route_decision(decision:dict) -> str:
    """根据决策中的 action 返回主流程使用的路由标识。"""
    action = decision.get("action", "")

    if action == "search_knowledge_base":
        return "rag"

    if action in ROUTER_ALLOWED_TOOLS:
        return action

    if action in ["clarify", "direct_answer"]:
        return action

    return "unknown"


def normalize_question(question: str) -> str:
    """清理用户问题的首尾空白，并把常见中文标点统一成英文标点。"""
    q = (question or "").strip()
    q = q.replace("？", "?")
    q = q.replace("：", ":")
    q = q.replace("，", ",")
    q = q.replace("。", "")
    return q


# Agent决策层：判断走哪个动作


# =========================
# 工具路由识别
# =========================
def extract_chunk_id(q: str) -> int | None:
    """从用户问题中识别显式提到的 chunk_id。"""
    patterns = [
        r"chunk[_\s-]?id\s*[:=]?\s*(\d+)",
        r"chunk\s*(\d+)",
        r"第\s*(\d+)\s*个\s*chunk",
        r"查看\s*(\d+)\s*号\s*chunk",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def normalize_math_text(text: str) -> str:
    # 全角字符转半角，例如：１＋２ -> 1+2
    """将全角数字、中文运算词和千分位格式标准化为可解析的算式。"""
    text = unicodedata.normalize("NFKC", text or "")

    # 中文运算词转成符号，例如：加 -> +
    replacements = [
        (r"加上|加", "+"),
        (r"减去|减掉|减", "-"),
        (r"乘以|乘上|乘", "*"),
        (r"除以|除", "/"),
        (r"×|x|X", "*"),
        (r"÷", "/"),
        (r"负\s*(?=\d|\()", "-"),
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # 处理千分位：1,000 + 2,000 -> 1000 + 2000
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    return text

def is_safe_math_expression(expr: str) -> bool:

    """使用 AST 检查表达式是否只包含数字和允许的四则运算。"""
    try:
        tree = ast.parse(expr, mode="eval")
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
        ast.USub
    )

    has_binary_operator = False

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False

        if isinstance(node, ast.BinOp):
            has_binary_operator = True

        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                return False

    return has_binary_operator


def extract_simple_math_expression(question:str) -> str | None:
    """从自然语言问题中提取可安全计算的简单数学表达式。"""
    text = normalize_math_text(question)

    # 从整句话中提取由数字、运算符、小数点、括号、空格组成的片段
    candidates = re.findall(r"[0-9+\-*/().\s]+", text)

    # 优先尝试更长的候选表达式
    candidates = sorted(
        (c.strip() for c in candidates if c.strip()),
        key=len,
        reverse=True
    )

    for expr in candidates:
        if is_safe_math_expression(expr):
            return expr

    return None


# =========================
# 工具结果格式化
# =========================
COMPARE_DOCUMENT_KEYWORDS = ("对比", "比较", "区别", "差异", "不同", "冲突", "一致")


def build_compare_documents_rule_args(question: str, top_k: int) -> dict | None:
    """从显式双文档对比问题中构造 compare_documents 参数。"""
    q = normalize_question(question)
    if not any(keyword in q for keyword in COMPARE_DOCUMENT_KEYWORDS):
        return None

    titles = [
        title.strip()
        for title in re.findall(r"\u300a([^\u300b]+)\u300b", q) # \u300a和\u300b代表中文书名号《》
        if title.strip()
    ]
    if len(titles) >= 2:
        return {
            "query": q,
            "doc_a": titles[0],
            "doc_b": titles[1],
            "top_k": max(1, min(int(top_k or 3), 10)),
        }

    return None


def format_tool_answer(tool_name: str, output: dict) -> str:
    """将工具输出转换成可以直接返回给用户的自然语言答案。"""
    if tool_name == "simple_calculator":
        return f"计算结果是：{output.get('result')}"
    if tool_name == "list_documents":
        docs = output.get("documents", [])
        count = output.get("count", 0)

        doc_names = []
        for d in docs[:10]:
            doc_names.append(
                d.get("source_file")
                or d.get("file_name")
                or d.get("title")
                or d.get("doc_name")
                or d.get("doc_id")
                or "未命名文档"
            )

        answer = f"当前系统中共有 {count} 个文档。"
        if doc_names:
            answer += " 文档包括：" + "、".join(doc_names)

        return answer

    if tool_name == "get_chunk_text":
        chunk_id = output.get("chunk_id")
        chunk = output.get("chunk", {})

        text = (
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("chunk_text")
            or str(chunk)
        )

        return f"chunk_{chunk_id} 内容如下：\n{text}"

    if tool_name == "search_knowledge_base":
        return f"知识库检索完成，共找到 {output.get('count', 0)} 个候选 chunk。"

    if tool_name == "compare_documents":
        return output.get("comparison") or output.get("summary") or str(output)

    return str(output)


def summarize_tool_result(tool_name: str, tool_result: dict) -> str:
    """为工具调用结果生成简短摘要，便于响应和日志展示。"""
    if tool_result["status"] != "success":
        return f"{tool_name} 工具执行失败：{tool_result.get('error', '')}"

    output = tool_result.get("output", {})

    if tool_name == "simple_calculator":
        return f"simple_calculator 执行成功，结果为 {output.get('result')}"

    if tool_name == "list_documents":
        return f"list_documents 执行成功，共找到 {output.get('count', 0)} 个文档"

    if tool_name == "get_chunk_text":
        return f"get_chunk_text 执行成功，返回 chunk_id={output.get('chunk_id')} 的内容"

    if tool_name == "search_knowledge_base":
        return f"search_knowledge_base 执行成功，召回 {output.get('count', 0)} 个 chunk"

    if tool_name == "compare_documents":
        return (
            "compare_documents 执行成功，"
            f"doc_a_evidence={len(output.get('doc_a_evidence', []))}，"
            f"doc_b_evidence={len(output.get('doc_b_evidence', []))}，"
            f"conflict_detected={output.get('conflict_detected')}"
        )

    return f"{tool_name} 执行成功"


# =========================
# 引用质量检查
# =========================
def check_reference_reliability(
        question: str,
        answer: str,
        references: list[dict],
        min_overlap_ratio: float = 0.2
) -> tuple[bool, str]:

    """检查答案关键词与引用片段是否有足够重合，判断引用是否可信。"""
    if not question:
        return False, ""

    verify_result = verify_answer_citations(
        question=question,
        answer=answer,
        references=references,
        min_overlap_ratio=min_overlap_ratio,
    )

    if verify_result["supported"]:
        return False, ""

    if verify_result.get("expected_keyword_count") == 0 and references:
        return False, ""

    reason = verify_result.get("reason", "")
    if verify_result.get("expected_keyword_count", 0) > 0:
        reason = (
            f"{reason}，"
            f"overlap_ratio={verify_result.get('overlap_ratio', 0.0):.2f}，"
            f"matched_keywords={verify_result.get('matched_keywords', [])[:10]}"
        )

    return True, reason

# =========================
# Agent 路由决策
# =========================

def add_router_metadata(
        decision: dict,
        source: str,
        confidence: float | None = None,
        fallback_reason: str = "",
        llm_used: bool | None = None
) -> dict:
    """给路由决策补充来源、置信度和回退原因等元数据。"""
    decision = dict(decision or {})
    decision["router_source"] = source

    if confidence is not None:
        decision["router_confidence"] = confidence

    if llm_used is None:
        llm_used = source in {"llm", "rule_based_fallback"}
    decision["router_llm_used"] = bool(llm_used)

    if fallback_reason:
        decision["router_fallback_reason"] = fallback_reason[:300]

    decision.setdefault("router_fallback_reason", "")

    return decision


def build_router_headers() -> dict:
    """构造调用路由模型时使用的 HTTP 请求头。"""
    headers = {"Content-Type": "application/json"}

    if Settings.ROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {Settings.ROUTER_API_KEY}"

    return headers


def parse_router_json_output(llm_output: str) -> dict:
    """从路由模型输出中解析出 JSON 决策对象。"""
    output = (llm_output or "").strip()

    if output.startswith("```"):
        output = output.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        start = output.find("{")
        end = output.rfind("}")

        if start < 0 or end <= start:
            raise ValueError("路由模型未返回 JSON 对象")

        data = json.loads(output[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("路由模型返回值不是 JSON 对象")

    return data


def call_router_llm(question: str, top_k: int, trace_id: str = "") -> str:
    """调用路由模型生成 Agent 动作决策，并处理重试和错误。"""
    payload = {
        "model": Settings.ROUTER_MODEL_NAME,
        "temperature": Settings.ROUTER_TEMPERATURE,
        "max_tokens": Settings.ROUTER_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ROUTER_USER_PROMPT.format(
                    question=question,
                    top_k=top_k,
                    available_actions=json.dumps(
                        sorted(ROUTER_ALLOWED_ACTIONS),
                        ensure_ascii=False
                    ),
                    tool_specs=json.dumps(
                        ROUTER_TOOL_SPECS,
                        ensure_ascii=False,
                        indent=2
                    )
                )
            }
        ]
    }

    headers = build_router_headers()
    total_attempts = Settings.ROUTER_MAX_RETRIES + 1
    last_error = ""

    for attempt in range(total_attempts):
        try:
            logger.info(
                f"[trace_id={trace_id}] [router_llm] attempt={attempt + 1}/{total_attempts} "
                f"url={Settings.ROUTER_API_URL} model={Settings.ROUTER_MODEL_NAME} "
                f"timeout={Settings.ROUTER_TIMEOUT}"
            )

            response = requests.post(
                Settings.ROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=Settings.ROUTER_TIMEOUT
            )

            if response.status_code != 200:
                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
                raise RuntimeError(last_error)

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            last_error = str(e)

            if attempt < Settings.ROUTER_MAX_RETRIES:
                logger.warning(
                    f"[trace_id={trace_id}] [router_llm] 调用失败，准备重试 "
                    f"attempt={attempt + 1}/{total_attempts} error={last_error}"
                )
                time.sleep(Settings.LLM_RETRY_SLEEP)
                continue

    raise RuntimeError(f"路由模型调用失败：{last_error}")


def route_agent_action_by_llm(question: str, top_k: int, trace_id: str = "") -> dict:
    """使用 LLM 路由并解析模型返回的决策。"""
    llm_output = call_router_llm(question, top_k, trace_id=trace_id)
    return parse_router_json_output(llm_output)


def normalize_router_confidence(value) -> float:
    """将路由置信度安全转换并限制在 0 到 1 之间。"""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, confidence))


def normalize_router_reason(raw_decision: dict) -> str:
    """提取并截断路由模型给出的决策理由。"""
    reason = str(raw_decision.get("reason") or "LLM 路由决策").strip()
    return reason[:200]


def normalize_chunk_id_value(chunk_id):
    """将工具参数里的 chunk_id 规整为整数或可查询字符串。"""
    if chunk_id is None:
        return None

    if isinstance(chunk_id, int):
        return chunk_id

    chunk_id_text = str(chunk_id).strip()
    if not chunk_id_text:
        return None

    return int(chunk_id_text) if chunk_id_text.isdigit() else chunk_id_text


def validate_route_decision(raw_decision: dict, question: str, top_k: int) -> dict:
    """校验并规整 LLM 路由输出，保证下游只收到合法动作和参数。"""
    if not isinstance(raw_decision, dict):
        raise ValueError("路由模型返回值不是 dict")

    q = normalize_question(question)
    if not q:
        return add_router_metadata(
            route_agent_action_rule_based(question, top_k),
            source="rule_based_guardrail"
        )

    raw_action = str(raw_decision.get("action") or "").strip()
    raw_decision_type = str(raw_decision.get("decision") or "").strip()

    if not raw_action:
        if raw_decision_type == "use_knowledge_base":
            raw_action = "search_knowledge_base"
        elif raw_decision_type in {"clarify", "direct_answer"}:
            raw_action = raw_decision_type
        elif raw_decision_type == "use_tool":
            raw_action = str(raw_decision.get("tool_name") or "").strip()

    if raw_action not in ROUTER_ALLOWED_ACTIONS:
        raise ValueError(f"未知路由 action: {raw_action}")

    decision_type = ROUTER_ACTION_TO_DECISION[raw_action]
    reason = normalize_router_reason(raw_decision)
    confidence = normalize_router_confidence(raw_decision.get("confidence"))
    tool_args = raw_decision.get("tool_args")
    if not isinstance(tool_args, dict):
        tool_args = {}

    if raw_action == "search_knowledge_base":
        return add_router_metadata(
            {
                "decision": decision_type,
                "action": "search_knowledge_base",
                "tool_name": "search_knowledge_base",
                "tool_args": {
                    "query": q,
                    "top_k": top_k,
                },
                "reason": reason,
                "clarification_question": "",
            },
            source="llm",
            confidence=confidence
        )

    if raw_action == "clarify":
        clarification_question = str(
            raw_decision.get("clarification_question")
            or DEFAULT_CLARIFICATION_QUESTION
        ).strip()

        return add_router_metadata(
            {
                "decision": decision_type,
                "action": "clarify",
                "tool_name": "",
                "tool_args": {},
                "reason": reason,
                "clarification_question": clarification_question,
            },
            source="llm",
            confidence=confidence
        )

    if raw_action == "direct_answer":
        return add_router_metadata(
            {
                "decision": decision_type,
                "action": "direct_answer",
                "tool_name": "",
                "tool_args": {},
                "reason": reason,
                "clarification_question": "",
            },
            source="llm",
            confidence=confidence
        )

    tool_name = str(raw_decision.get("tool_name") or raw_action).strip()
    if tool_name != raw_action or tool_name not in ROUTER_ALLOWED_TOOLS:
        raise ValueError(f"非法工具选择: action={raw_action}, tool_name={tool_name}")

    if raw_action == "list_documents":
        checked_args = {}

    elif raw_action == "simple_calculator":
        expression = str(tool_args.get("expression") or "").strip()
        expression = normalize_math_text(expression)

        if not expression or not is_safe_math_expression(expression):
            expression = extract_simple_math_expression(question) or ""

        if not expression:
            raise ValueError("simple_calculator 缺少合法 expression")

        checked_args = {"expression": expression}

    elif raw_action == "get_chunk_text":
        chunk_id = normalize_chunk_id_value(tool_args.get("chunk_id"))

        if chunk_id is None:
            chunk_id = extract_chunk_id(q)

        if chunk_id is None:
            raise ValueError("get_chunk_text 缺少 chunk_id")

        checked_args = {"chunk_id": chunk_id}

    elif raw_action == "compare_documents":
        query_arg = str(tool_args.get("query") or q).strip()
        doc_a = str(tool_args.get("doc_a") or "").strip()
        doc_b = str(tool_args.get("doc_b") or "").strip()

        if not doc_a or not doc_b:
            raise ValueError("compare_documents 缺少 doc_a 或 doc_b")

        try:
            compare_top_k = int(tool_args.get("top_k", top_k))
        except (TypeError, ValueError):
            compare_top_k = top_k

        checked_args = {
            "query": query_arg,
            "doc_a": doc_a,
            "doc_b": doc_b,
            "top_k": max(1, min(compare_top_k, 10)),
        }

    else:
        raise ValueError(f"未处理的工具 action: {raw_action}")

    return add_router_metadata(
        {
            "decision": decision_type,
            "action": raw_action,
            "tool_name": tool_name,
            "tool_args": checked_args,
            "reason": reason,
            "clarification_question": "",
        },
        source="llm",
        confidence=confidence
    )


def route_agent_action_rule_based(question: str, top_k:int) -> dict:
    """旧版规则路由：作为 LLM 路由不可用或输出非法时的 fallback。"""
    q = normalize_question(question)

    # 1. 空问题：需要澄清
    if not q:
        return {
            "decision": "clarify",
            "action": "clarify",
            "tool_name": "",
            "tool_args": {},
            "reason": "用户未输入问题",
            "clarification_question": "请直接输入你想查询的具体问题，例如：每个 Vue 应用是通过什么函数创建实例的？"
        }

    # 2. 用户问系统能力：可以直接回答，不需要查知识库
    if q in ["你是谁?", "你能做什么?", "你可以做什么?", "你有什么功能?"]:
        return {
            "decision": "direct_answer",
            "action": "direct_answer",
            "tool_name": "",
            "tool_args": {},
            "reason": "用户询问系统能力，可以直接回答"
        }

    # 3. 用户明确要求工具型文档操作：对比文档或查看 chunk
    compare_args = build_compare_documents_rule_args(question, top_k)
    if compare_args is not None:
        return {
            "decision": "use_tool",
            "action": "compare_documents",
            "tool_name": "compare_documents",
            "tool_args": compare_args,
            "reason": "用户明确要求对比两份文档中的规定差异"
        }

    chunk_id = extract_chunk_id(q)
    if chunk_id is not None:
        return {
            "decision": "use_tool",
            "action": "get_chunk_text",
            "tool_name": "get_chunk_text",
            "tool_args": {"chunk_id": chunk_id},
            "reason": "用户要求查看指定 chunk 内容"
            }

    # 4. 用户问知识库文档数量或列表：调用 list_documents
    if "文档" in q and any(x in q for x in ["几个", "多少", "数量", "列表", "有哪些"]):
        return {
            "decision": "use_tool",
            "action": "list_documents",
            "tool_name": "list_documents",
            "tool_args": {},
            "reason": "用户询问知识库文档列表或文档数量"
        }

    # 5. 用户输入简单数学表达式：调用 simple_calculator
    expression = extract_simple_math_expression(question)
    if expression is not None:
        return {
            "decision": "use_tool",
            "action": "simple_calculator",
            "tool_name": "simple_calculator",
            "tool_args": {
                "expression": expression
            },
            "reason": "用户问题是简单四则运算"
        }

    # 6. 指代太模糊：澄清
    if q in ["这个怎么弄", "这个怎么配置", "这个是什么意思", "这个怎么解决"]:
        return {
            "decision": "clarify",
            "action": "clarify",
            "tool_name": "",
            "tool_args": {},
            "reason": "问题存在明显指代不明",
            "clarification_question": "你说的“这个”具体指哪个功能、哪份文档或哪段报错？请补充一下对象。"
        }

    # 7. 默认：当成知识库问题，走 RAG
    return {
        "decision": "use_knowledge_base",
        "action": "search_knowledge_base",
        "tool_name": "search_knowledge_base",
        "tool_args": {
            "query": q,
            "top_k": top_k
        },
        "reason": "默认作为知识库问答问题处理"
    }


def route_agent_action(question: str, top_k:int, trace_id: str = "") -> dict:
    """根据问题内容决定走澄清、工具调用还是知识库 RAG。"""
    q = normalize_question(question)
    rule_decision = route_agent_action_rule_based(question, top_k)

    if not q:
        return add_router_metadata(
            rule_decision,
            source="rule_based_guardrail",
            confidence=1.0,
            llm_used=False
        )

    if rule_decision.get("action") in HIGH_CONFIDENCE_RULE_ACTIONS:
        return add_router_metadata(
            rule_decision,
            source="rule_based",
            confidence=1.0,
            llm_used=False
        )

    if not Settings.ENABLE_LLM_ROUTER:
        return add_router_metadata(
            rule_decision,
            source="rule_based",
            confidence=0.6,
            llm_used=False
        )

    try:
        raw_decision = route_agent_action_by_llm(question, top_k, trace_id=trace_id)
        return validate_route_decision(raw_decision, question, top_k)

    except Exception as e:
        logger.warning(
            f"[trace_id={trace_id}] [router_llm] fallback_to_rule_based "
            f"error={str(e)[:300]}"
        )
        return add_router_metadata(
            rule_decision,
            source="rule_based_fallback",
            confidence=0.4,
            fallback_reason=str(e),
            llm_used=True
        )


def attach_llm_usage(
    result: dict,
    router_llm_used: bool,
    decompose_llm_used: bool | None = None,
    answer_llm_used: bool | None = None,
) -> dict:
    """把本次请求各阶段 LLM 调用情况写回结果和 tool_trace。"""
    result = dict(result or {})

    if decompose_llm_used is None:
        decompose_llm_used = bool(result.get("decompose_llm_used", False))

    if answer_llm_used is None:
        answer_llm_used = bool(result.get("answer_llm_used", False))

    llm_call_count = (
        int(bool(router_llm_used))
        + int(bool(decompose_llm_used))
        + int(bool(answer_llm_used))
    )

    result["router_llm_used"] = bool(router_llm_used)
    result["decompose_llm_used"] = bool(decompose_llm_used)
    result["answer_llm_used"] = bool(answer_llm_used)
    result["llm_call_count"] = llm_call_count

    tool_trace = list(result.get("tool_trace") or [])
    if not any(step.get("step") == "llm_usage" for step in tool_trace):
        tool_trace.append({
            "step": "llm_usage",
            "router_llm_used": bool(router_llm_used),
            "decompose_llm_used": bool(decompose_llm_used),
            "answer_llm_used": bool(answer_llm_used),
            "llm_call_count": llm_call_count,
        })
    result["tool_trace"] = tool_trace

    return result


def build_empty_evidence_gate_step(reason: str) -> dict:
    """构造无召回结果时的证据门控 trace。"""
    return {
        "step": "evidence_gate",
        "refused": True,
        "max_vector_score": 0.0,
        "max_rerank_score": 0.0,
        "max_hybrid_score": 0.0,
        "reason": reason,
    }


def get_chunk_trace_value(chunk: dict, key: str, default=None):
    """从 chunk 顶层或 metadata 中取 trace 展示字段。"""
    if not isinstance(chunk, dict):
        return default

    value = chunk.get(key)
    if value is not None:
        return value

    metadata = chunk.get("metadata") or {}
    if isinstance(metadata, dict):
        return metadata.get(key, default)

    return default


def round_chunk_score(chunk: dict, key: str) -> float:
    """安全读取并四舍五入 chunk 分数字段。"""
    try:
        return round(float(get_chunk_trace_value(chunk, key, 0) or 0), 4)
    except (TypeError, ValueError):
        return 0.0


def build_chunk_hits(chunks: list[dict]) -> list[dict]:
    """构造工具 trace 中可快速查看的 chunk 命中摘要。"""
    return [
        {
            "chunk_id": get_chunk_trace_value(chunk, "chunk_id"),
            "rerank": round_chunk_score(chunk, "rerank_score"),
            "score": round_chunk_score(chunk, "hybrid_score"),
            "bm25": round_chunk_score(chunk, "bm25_score"),
            "vector": round_chunk_score(chunk, "vector_score"),
            "file_name": get_chunk_trace_value(chunk, "file_name"),
        }
        for chunk in (chunks or [])
        if isinstance(chunk, dict)
    ]


def build_agent_result(
    status: str,
    answer: str = "",
    references: list | None = None,
    retrieval_confidence: float = 0.0,
    answer_confidence: float = 0.0,
    need_clarification: bool = False,
    clarification_question: str = "",
    reason: str = "",
    action: str = "",
    tool_result_summary: str = "",
    tool_trace: list | None = None,
    **extra,
) -> dict:
    """构造 Agent 返回结构，避免各分支重复填同一批字段。"""
    result = {
        "status": status,
        "answer": answer,
        "references": references or [],
        "retrieval_confidence": retrieval_confidence,
        "answer_confidence": answer_confidence,
        "need_clarification": need_clarification,
        "clarification_question": clarification_question,
        "reason": reason,
        "action": action or "",
        "tool_result_summary": tool_result_summary,
        "tool_trace": tool_trace or [],
    }
    result.update(extra)
    return result


# =========================
# Agent 执行主流程
# =========================
def run_agent_once(
        write_bad_cases,
        question: str, top_k: int, decompose_top_k: int,
        decision: dict | None = None, trace_id : str = "",
        bad_case: dict | None = None) -> dict:
    """
    执行一次最小 Agent 流程。

    这版是和当前 main.py 适配的最小改动版：
    1. 不重复校验 top_k，因为接口层已经做了。
    2. 不引入 trace 大字典，继续使用 tool_trace。
    3. 保持原来的大写业务状态，方便 main.py 分流。
    4. 补上工具异常、工具失败、RAG 空结果时的可读返回。
    """

    if bad_case is None:
        bad_case = {
            "trace_id": trace_id,
            "question": question,
            "answer": "",
            "references": [],
            "route": "",
            "error_type": "",
            "note": ""
        }

    # =========================
    # 1. 先做路由决策
    # =========================
    if decision is None:
        decision = route_agent_action(question, top_k, trace_id=trace_id)

    decision_type = decision.get("decision")
    route_decision = get_route_decision(decision)
    action = decision.get("action")
    tool_name = decision.get("tool_name", "")
    tool_args = decision.get("tool_args", {})
    router_llm_used = bool(
        decision.get(
            "router_llm_used",
            decision.get("router_source") in {"llm", "rule_based_fallback"}
        )
    )

    tool_trace = [
        {
            "step": "decide",
            "route_decision": route_decision,
            "decision_type": decision_type,
            "action": action,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reason": decision.get("reason", ""),
            "router_source": decision.get("router_source", ""),
            "router_confidence": decision.get("router_confidence"),
            "router_fallback_reason": decision.get("router_fallback_reason", ""),
            "router_llm_used": router_llm_used
        }
    ]

    def finalize_result(result: dict) -> dict:
        """统一给结果附加本次流程的 LLM 使用统计。"""
        return attach_llm_usage(result, router_llm_used=router_llm_used)

    def agent_result(**kwargs) -> dict:
        """构造带默认 action 和 tool_trace 的 Agent 返回结构。"""
        kwargs.setdefault("action", action)
        kwargs.setdefault("tool_trace", tool_trace)
        return build_agent_result(**kwargs)

    def finalize_tool_error(
        answer: str,
        reason: str,
        summary: str,
        bad_case_note: str,
        trace_error: str | None = None,
    ) -> dict:
        """记录工具失败轨迹和 bad case，并返回统一错误结果。"""
        tool_trace.append({
            "step": "call_tool",
            "tool_name": tool_name,
            "tool_status": "error",
            "final_status": "tool_error",
            "error": trace_error or reason
        })

        bad_case["answer"] = answer
        bad_case["route"] = route_decision
        bad_case["error_type"] = "tool_error"
        bad_case["note"] = bad_case_note
        write_bad_cases(bad_case)

        return finalize_result(agent_result(
            status="TOOL_ERROR",
            answer=answer,
            reason=reason,
            tool_result_summary=summary,
        ))

    logger.info(
        f"[trace_id={trace_id}] [agent] route_decision={route_decision} "
        f"decision={decision_type} action={action} "
        f"tool_name={tool_name} router_source={decision.get('router_source', '')} "
        f"reason={decision.get('reason')} question={(question or '')[:50]}"
    )

    # =========================
    # 2. 需要澄清
    # =========================
    if decision_type == "clarify":
        tool_trace.append({
            "step": "final_answer",
            "tool_status": "skipped",
            "final_status": "clarify",
            "answer_preview": decision.get("clarification_question", "")[:80]
        })

        return finalize_result(agent_result(
            status="NEED_CLARIFICATION",
            need_clarification=True,
            clarification_question=decision.get("clarification_question", ""),
            reason=decision.get("reason", "问题需要澄清"),
            tool_result_summary="未调用工具，问题需要澄清",
        ))

    # =========================
    # 3. 直接回答系统能力问题
    # =========================
    if action == "direct_answer":
        answer = (
            "我是一个基于当前知识库的问答助手。"
            "我可以根据已加载文档回答问题，也可以列出知识库文档，"
            "并处理简单四则运算。对于知识库没有覆盖的问题，我会说明无法根据文档确定。"
        )

        tool_trace.append({
            "step": "final_answer",
            "tool_status": "skipped",
            "final_status": "ok",
            "answer_preview": answer[:80]
        })

        return finalize_result(agent_result(
            status="OK",
            answer=answer,
            answer_confidence=1.0,
            reason=decision.get("reason", "直接回答系统能力问题"),
            tool_result_summary="未调用工具，直接回答",
        ))

    # =========================
    # 4. 调用工具
    # =========================
    if decision_type == "use_tool" and action != "search_knowledge_base":
        tool_trace.append({
            "step": "call_tool",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_status": "running"
        })

        try:
            logger.info(
                f"[trace_id={trace_id}] [tool] tool_name={tool_name} "
                f"tool_status=running tool_error="
            )
            tool_result = dispatch_tool(tool_name, tool_args, trace_id=trace_id)

        except Exception as e:
            error_msg = str(e)

            logger.warning(
                f"[trace_id={trace_id}] [tool] tool_name={tool_name} "
                f"tool_status=error tool_error={error_msg}"
            )

            return finalize_tool_error(
                answer=f"工具 {tool_name} 执行异常，无法完成本次请求。",
                reason=error_msg,
                summary=f"{tool_name} 工具执行异常：{error_msg}",
                bad_case_note=f"工具 {tool_name} 执行错误：{error_msg}",
            )

        if not isinstance(tool_result, dict):
            logger.info(f"[trace_id={trace_id}] [agent] 工具返回异常，tool_status:error tool_error:{type(tool_result)}")
            return finalize_tool_error(
                answer=f"工具 {tool_name} 返回格式异常，无法完成本次请求。",
                reason="工具返回值不是 dict",
                summary=f"{tool_name} 工具返回格式异常",
                bad_case_note=f"工具 {tool_name} 返回格式异常，返回值不是 dict",
                trace_error=f"工具返回值类型异常：{type(tool_result)}",
            )

        if tool_result.get("status") != "success":
            error_msg = tool_result.get("error", f"{tool_name} 工具执行失败")

            logger.warning(
                f"[trace_id={trace_id}] [tool] tool_name={tool_name} "
                f"tool_status=error tool_error={error_msg}"
            )

            return finalize_tool_error(
                answer=f"工具 {tool_name} 执行失败，无法完成本次请求。",
                reason=error_msg,
                summary=summarize_tool_result(tool_name, tool_result),
                bad_case_note=f"工具 {tool_name} 执行失败：{error_msg}",
            )

        logger.info(
            f"[trace_id={trace_id}] [tool] tool_name={tool_name} "
            f"tool_status=success tool_error="
        )
        tool_output = tool_result.get("output", {})
        tool_summary = summarize_tool_result(tool_name, tool_result)
        answer = format_tool_answer(tool_name, tool_output)

        if tool_trace and tool_trace[-1].get("step") == "call_tool":
            tool_trace[-1].update({
                "tool_status": "success",
                "success": True,
                "result_summary": tool_summary,
                "tool_output": tool_output,
            })

        tool_trace.append({
            "step": "final_answer",
            "tool_name": tool_name,
            "tool_status": "success",
            "final_status": "ok",
            "answer_preview": answer[:80]
        })

        return finalize_result(agent_result(
            status="OK",
            answer=answer,
            answer_confidence=1.0,
            reason=f"Agent 选择工具：{tool_name}",
            tool_result_summary=tool_summary,
            tool_result=tool_result,
        ))

    # =========================
    # 5. 走知识库 RAG
    # =========================
    if action == "search_knowledge_base":
        search_tool_name = "search_knowledge_base"
        search_tool_args = {
            "query": question,
            "top_k": top_k
        }

        logger.info(f"[trace_id={trace_id}] [retrieve] 开始检索 top_k={top_k}")
        retrieve_start = time.perf_counter()

        try:
            tool_result = dispatch_tool(
                search_tool_name,
                search_tool_args,
                trace_id=trace_id
            )
        except Exception as e:
            tool_result = {
                "status": "error",
                "ok": False,
                "tool_name": search_tool_name,
                "tool_input": search_tool_args,
                "output": None,
                "error": str(e),
            }

        retrieve_cost = int((time.perf_counter() - retrieve_start) * 1000)

        logger.info(f"[trace_id={trace_id}] [retrieve] retrieve_cost_ms={retrieve_cost}")

        if not isinstance(tool_result, dict):
            error_msg = f"工具返回值类型异常：{type(tool_result)}"

            tool_trace.append({
                "step": "call_tool",
                "tool_name": search_tool_name,
                "tool_args": search_tool_args,
                "tool_status": "error",
                "success": False,
                "final_chunk_count": 0,
                "top_chunk_ids": [],
                "top_scores": [],
                "result_summary": "知识库检索工具返回格式异常",
                "hits": [],
                "error": error_msg,
            })

            bad_case["answer"] = f"工具 {search_tool_name} 返回格式异常，无法完成本次请求。"
            bad_case["route"] = route_decision
            bad_case["error_type"] = "tool_error"
            bad_case["note"] = error_msg
            write_bad_cases(bad_case)

            return finalize_result(agent_result(
                status="TOOL_ERROR",
                answer=f"工具 {search_tool_name} 返回格式异常，无法完成本次请求。",
                reason=error_msg,
                tool_result_summary="search_knowledge_base 工具返回格式异常",
                timings={
                    "retrieve_cost_ms": retrieve_cost,
                    "rag_answer_cost_ms": 0,
                },
            ))

        if tool_result.get("status") != "success":
            error_msg = tool_result.get("error", "search_knowledge_base 工具执行失败")

            logger.warning(
                f"[trace_id={trace_id}] [tool] tool_name={search_tool_name} "
                f"tool_status=error tool_error={error_msg}"
            )

            tool_trace.append({
                "step": "call_tool",
                "tool_name": search_tool_name,
                "tool_args": search_tool_args,
                "tool_status": "error",
                "success": False,
                "final_chunk_count": 0,
                "top_chunk_ids": [],
                "top_scores": [],
                "result_summary": f"知识库检索工具执行失败：{error_msg}",
                "hits": [],
                "error": error_msg,
            })

            bad_case["answer"] = f"工具 {search_tool_name} 执行失败，无法完成本次请求。"
            bad_case["route"] = route_decision
            bad_case["error_type"] = "tool_error"
            bad_case["note"] = f"工具 {search_tool_name} 执行失败：{error_msg}"
            write_bad_cases(bad_case)

            return finalize_result(agent_result(
                status="TOOL_ERROR",
                answer=f"工具 {search_tool_name} 执行失败，无法完成本次请求。",
                reason=error_msg,
                tool_result_summary=summarize_tool_result(search_tool_name, tool_result),
                timings={
                    "retrieve_cost_ms": retrieve_cost,
                    "rag_answer_cost_ms": 0,
                },
            ))

        tool_output = tool_result.get("output") or {}
        if not isinstance(tool_output, dict):
            tool_output = {}

        retrieved_chunks = tool_output.get("chunks", [])
        if not isinstance(retrieved_chunks, list):
            retrieved_chunks = []

        retrieved_count = len(retrieved_chunks)
        retrieved_chunk_ids = (
            tool_output.get("top_chunk_ids")
            or [
                get_chunk_trace_value(chunk, "chunk_id")
                for chunk in retrieved_chunks
            ]
        )
        retrieved_scores = (
            tool_output.get("top_scores")
            or [
                round_chunk_score(chunk, "hybrid_score")
                for chunk in retrieved_chunks
            ]
        )

        rag_answer_start = time.perf_counter()

        if not retrieved_chunks:

            logger.warning(
                f"[trace_id={trace_id}] [retrieve] retrieved_count=0 "
                f"top_chunk_ids=[] top_scores=[]"
            )

            tool_trace.append({
                "step": "call_tool",
                "tool_name": search_tool_name,
                "tool_args": search_tool_args,
                "tool_status": "success",
                "final_status": "no_evidence",
                "success": True,
                "final_chunk_count": 0,
                "top_chunk_ids": [],
                "top_scores": [],
                "result_summary": "没有检索到候选 chunk",
                "hits": [],
            })

            tool_trace.append(build_empty_evidence_gate_step("知识库检索结果为空"))

            return finalize_result({
                "status": "NO_RELEVANT_INFO",
                "answer": "没有检索到足够相关的知识库内容，无法根据当前文档回答。",
                "references": [],
                "retrieval_confidence": 0.0,
                "answer_confidence": 0.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": "知识库检索结果为空",
                "action": action,
                "tool_result_summary": "知识库检索完成，但没有召回候选 chunk",
                "timings": {
                    "retrieve_cost_ms": retrieve_cost,
                    "rag_answer_cost_ms": 0,
                },
                "tool_trace": tool_trace
            })

        logger.info(
            f"[trace_id={trace_id}] [retrieve] retrieved_count={retrieved_count} "
            f"top_chunk_ids={retrieved_chunk_ids} top_scores={retrieved_scores}"
        )

        rag_result = ask(
            question,
            top_k=top_k,
            decompose_top_k=decompose_top_k,
            retrieved_chunks=retrieved_chunks,
            trace_id=trace_id
        )

        final_chunks = rag_result.get("final_chunks", [])
        final_chunk_count = len(final_chunks)
        top_chunk_ids = [
            get_chunk_trace_value(chunk, "chunk_id")
            for chunk in final_chunks
        ]
        top_scores = [
            round_chunk_score(chunk, "hybrid_score")
            for chunk in final_chunks
        ]
        hit_brief = build_chunk_hits(final_chunks)

        tool_trace.append({
            "step": "call_tool",
            "tool_name": search_tool_name,
            "tool_args": search_tool_args,
            "tool_status": "success",
            "success": True,
            "final_chunk_count": final_chunk_count,
            "top_chunk_ids": top_chunk_ids,
            "top_scores": top_scores,
            "result_summary": f"检索到 {len(final_chunks)} 个候选 chunk",
            "hits": hit_brief
        })

        evidence_gate = rag_result.get("evidence_gate")
        if evidence_gate:
            tool_trace.append(evidence_gate)

        if evidence_gate and evidence_gate.get("refused"):
            tool_trace.append({
                "step": "format_context",
                "tool_status": "skipped",
                "summary": "证据不足，跳过上下文构造和最终回答 LLM"
            })
        else:
            tool_trace.append({
                "step": "format_context",
                "summary": f"将 {len(final_chunks)} 个 chunk 交给 RAG 回答逻辑"
            })

        citation_verification = rag_result.get("citation_verification")
        if isinstance(citation_verification, dict):
            tool_trace.append({
                "step": "citation_verifier",
                "supported": bool(citation_verification.get("supported", False)),
                "overlap_ratio": citation_verification.get("overlap_ratio", 0.0),
                "matched_keywords": citation_verification.get("matched_keywords", []),
                "reason": citation_verification.get("reason", ""),
            })

        rag_answer_cost = int((time.perf_counter() - rag_answer_start) * 1000)

        logger.info(f"[trace_id={trace_id}] [rag] rag_answer_cost_ms={rag_answer_cost}")

        timings = {
            "retrieve_cost_ms": retrieve_cost,
            "rag_answer_cost_ms": rag_answer_cost,
        }
        timings.update(rag_result.get("timings") or {})
        rag_result["timings"] = timings

        rag_status = rag_result.get("status", "OK")

        if rag_status == "NO_RELEVANT_INFO":

            final_status = "no_evidence"

            answer_text = rag_result.get("answer", "")
            references = rag_result.get("references", []) or []

            is_bad_reference, reference_note = check_reference_reliability(
                question,
                answer_text,
                references
            )

            if is_bad_reference:
                bad_case["answer"] = answer_text
                bad_case["references"] = references
                bad_case["route"] = route_decision
                bad_case["error_type"] = "reference_error"
                bad_case["note"] = reference_note
                write_bad_cases(bad_case)

                logger.warning(
                    f"[trace_id={trace_id}] [bad_case] "
                    f"error_type=reference_error note={reference_note}"
                )

        elif rag_status == "NEED_CLARIFICATION":

            final_status = "clarify"

        elif rag_status == "MODEL_BAD_OUTPUT":

            final_status = "error"

            bad_case["answer"] = rag_result.get("answer", "")
            bad_case["references"] = rag_result.get("references", [])
            bad_case["route"] = route_decision
            bad_case["error_type"] = "parse_error"
            bad_case["note"] = "模型未按约定 JSON 格式返回"
            write_bad_cases(bad_case)

        else:
            final_status = "ok"

            is_bad_reference, reference_note = check_reference_reliability(
                question,
                rag_result.get("answer", ""),
                rag_result.get("references", [])
            )

            if is_bad_reference:
                bad_case["answer"] = rag_result.get("answer", "")
                bad_case["references"] = rag_result.get("references", [])
                bad_case["route"] = route_decision
                bad_case["error_type"] = "reference_error"
                bad_case["note"] = reference_note
                write_bad_cases(bad_case)

                logger.warning(
                    f"[trace_id={trace_id}] [bad_case] "
                    f"error_type=reference_error note={reference_note}"
                )

        tool_trace.append({
            "step": "final_answer",
            "tool_status": "skipped",
            "final_status": final_status,
            "answer_preview": rag_result.get("answer", "")[:80]
        })

        rag_result["action"] = action
        rag_result["tool_result_summary"] = f"知识库检索完成，召回 {len(final_chunks)} 个 chunk"
        rag_result["tool_trace"] = tool_trace

        return finalize_result(rag_result)

    # =========================
    # 6. 防御分支：理论上不应该走到这里
    # =========================
    tool_trace.append({
        "step": "final_answer",
        "tool_status": "skipped",
        "final_status": "error",
        "error": f"未知 Agent action: {action}"
    })

    bad_case["answer"] = "Agent 决策结果异常，无法完成本次请求。"
    bad_case["references"] = []
    bad_case["route"] = route_decision
    bad_case["error_type"] = "agent_decision_error"
    bad_case["note"] = f"未知 Agent action: {action}"
    write_bad_cases(bad_case)

    return finalize_result(agent_result(
        status="MODEL_BAD_OUTPUT",
        answer="Agent 决策结果异常，无法完成本次请求。",
        reason=f"未知 Agent action: {action}",
        action=action or "",
        tool_result_summary="未知 Agent action",
    ))

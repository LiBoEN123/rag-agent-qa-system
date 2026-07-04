import ast
import json
import re
import time
import unicodedata

import requests

from app.config import Settings
from app.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from app.rag import ask, search_relevant_chunks
from app.tools import dispatch_tool
from app.untils import get_logger


logger = get_logger('chat_service')

ROUTER_ALLOWED_ACTIONS = {
    "search_knowledge_base",
    "simple_calculator",
    "list_documents",
    "get_chunk_text",
    "clarify",
    "direct_answer",
}

ROUTER_ALLOWED_TOOLS = {
    "simple_calculator",
    "list_documents",
    "get_chunk_text",
}

ROUTER_ACTION_TO_DECISION = {
    "search_knowledge_base": "use_knowledge_base",
    "simple_calculator": "use_tool",
    "list_documents": "use_tool",
    "get_chunk_text": "use_tool",
    "clarify": "clarify",
    "direct_answer": "direct_answer",
}

ROUTER_TOOL_SPECS = [
    {
        "name": "simple_calculator",
        "description": "执行简单数字四则运算",
        "args_schema": {"expression": "string, 必填，例如 '1 + 2 * 3'"},
    },
    {
        "name": "list_documents",
        "description": "列出知识库文档及数量",
        "args_schema": {},
    },
    {
        "name": "get_chunk_text",
        "description": "查看指定 chunk_id 的完整原文",
        "args_schema": {"chunk_id": "integer|string, 必填"},
    },
]

DEFAULT_CLARIFICATION_QUESTION = "请补充你想查询的具体对象、文档、功能或报错信息。"


# =========================
# 路由通用工具
# =========================
def get_route_decision(decision:dict) -> str:
    """根据决策中的 action 返回主流程使用的路由标识。"""
    action = decision.get("action", "")

    if action == "search_knowledge_base":
        return "rag"

    if action in ["simple_calculator", "list_documents", "get_chunk_text"]:
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

    return f"{tool_name} 执行成功"


# =========================
# 引用质量检查
# =========================
REFERENCE_STOPWORDS = {
    "这个", "那个", "什么", "怎么", "如何", "为什么", "是否", "可以", "需要",
    "用户", "问题", "回答", "根据", "文档", "内容", "相关", "说明",
    "的是", "一个", "以及", "进行", "通过", "如果", "时候", "没有",
    "the", "and", "for", "with", "from", "this", "that", "what", "how"
}


def extract_reference_keywords(text: str) -> set[str]:
    """从问题、答案或引用文本中提取用于粗略匹配的关键词集合。"""
    text = (text or "").lower()

    keywords = set()

    # 英文、数字、函数名，例如 createapp、mount、v-model、computed。
    english_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-]{1,}|\d+(?:\.\d+)?", text)
    for token in english_tokens:
        token = (token or "").strip().lower()
        if token and token not in REFERENCE_STOPWORDS:
            keywords.add(token)

    # 中文连续片段用二字滑窗做粗粒度匹配，足够用于 bad case 兜底判断。
    chinese_blocks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for block in chinese_blocks:
        for i in range(len(block) - 1):
            word = block[i: i+2]
            if word and word not in REFERENCE_STOPWORDS:
                keywords.add(word)

    return keywords


def check_reference_reliability(
        question: str,
        answer: str,
        references: list[dict],
        min_overlap_ratio: float = 0.2
) -> tuple[bool, str]:

    """检查答案关键词与引用片段是否有足够重合，判断引用是否可信。"""
    if not question:
        return False, ""

    if not references:
        return True, "答案非空，但reference为空，无法验证答案依据"

    evidence_text = "\n".join(
        str(ref.get("snippet") or ref.get("text") or ref.get("content") or "")
        for ref in references
    )

    if not evidence_text.strip():
        return True, "references 存在，但 snippet/text 为空，无法验证答案依据"

    expected_keywords = extract_reference_keywords(question + "\n" + answer)
    evidence_keywords = extract_reference_keywords(evidence_text)

    if not expected_keywords:
        return False, ""

    overlap_keywords = expected_keywords & evidence_keywords
    overlap_ratio = len(overlap_keywords) / max(len(expected_keywords), 1)

    if overlap_ratio < min_overlap_ratio:
        return True, (
            f"答案与引用片段关键词重合度过低，"
            f"overlap_ratio={overlap_ratio:.2f}，"
            f"matched_keywords={sorted(list(overlap_keywords))[:10]}"
        )

    return False, ""


def should_record_reference_bad_case(
        question: str,
        answer: str,
        references: list[dict],
        reason: str
) -> tuple[bool, str]:

    """判断引用不足的回答是否应该记录为 bad case。"""
    answer = (answer or "").strip()
    reason = (reason or "").strip()

    if not question:
        return False, ""

    if "我无法根据提供的文档确定答案" in answer:
        return False, ""

    if (
        "检索结果为空" in reason
        or "top1 分数低于阈值" in reason
        or "证据分数低于阈值" in reason
    ):
        return False, ""

    if "模型给出的答案没有足够的引用支撑" in reason:
        return check_reference_reliability(
            question,
            answer,
            references
        )

    return False, ""


# =========================
# Agent 路由决策
# =========================

def add_router_metadata(
        decision: dict,
        source: str,
        confidence: float | None = None,
        fallback_reason: str = ""
) -> dict:
    """给路由决策补充来源、置信度和回退原因等元数据。"""
    decision = dict(decision or {})
    decision["router_source"] = source

    if confidence is not None:
        decision["router_confidence"] = confidence

    if fallback_reason:
        decision["router_fallback_reason"] = fallback_reason[:300]

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
                "tool_name": "",
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

    # 3. 用户问知识库某文档内容：调用 get_chunk_text
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
        "tool_name": "",
        "tool_args": {
            "query": q,
            "top_k": top_k
        },
        "reason": "默认作为知识库问答问题处理"
    }


def route_agent_action(question: str, top_k:int, trace_id: str = "") -> dict:
    """根据问题内容决定走澄清、工具调用还是知识库 RAG。"""
    q = normalize_question(question)

    if not q:
        return add_router_metadata(
            route_agent_action_rule_based(question, top_k),
            source="rule_based_guardrail"
        )

    if not Settings.ENABLE_LLM_ROUTER:
        return add_router_metadata(
            route_agent_action_rule_based(question, top_k),
            source="rule_based"
        )

    try:
        raw_decision = route_agent_action_by_llm(question, top_k, trace_id=trace_id)
        return validate_route_decision(raw_decision, question, top_k)

    except Exception as e:
        fallback = route_agent_action_rule_based(question, top_k)
        logger.warning(
            f"[trace_id={trace_id}] [router_llm] fallback_to_rule_based "
            f"error={str(e)[:300]}"
        )
        return add_router_metadata(
            fallback,
            source="rule_based_fallback",
            fallback_reason=str(e)
        )

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
            "router_fallback_reason": decision.get("router_fallback_reason", "")
        }
    ]

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

        return {
            "status": "NEED_CLARIFICATION",
            "answer": "",
            "references": [],
            "retrieval_confidence": 0.0,
            "answer_confidence": 0.0,
            "need_clarification": True,
            "clarification_question": decision.get("clarification_question", ""),
            "reason": decision.get("reason", "问题需要澄清"),
            "action": action,
            "tool_result_summary": "未调用工具，问题需要澄清",
            "tool_trace": tool_trace
        }

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

        return {
            "status": "OK",
            "answer": answer,
            "references": [],
            "retrieval_confidence": 0.0,
            "answer_confidence": 1.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": decision.get("reason", "直接回答系统能力问题"),
            "action": action,
            "tool_result_summary": "未调用工具，直接回答",
            "tool_trace": tool_trace
        }

    # =========================
    # 4. 调用工具
    # =========================
    if decision_type == "use_tool":
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

            tool_trace.append({
                "step": "call_tool",
                "tool_name": tool_name,
                "tool_status": "error",
                "final_status": "tool_error",
                "error": error_msg
            })

            bad_case["answer"] = f"工具 {tool_name} 执行异常，无法完成本次请求。"
            bad_case["route"] = route_decision
            bad_case["error_type"] = "tool_error"
            bad_case["note"] = f"工具 {tool_name} 执行错误：{error_msg}"
            write_bad_cases(bad_case)

            return {
                "status": "TOOL_ERROR",
                "answer": f"工具 {tool_name} 执行异常，无法完成本次请求。",
                "references": [],
                "retrieval_confidence": 0.0,
                "answer_confidence": 0.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": error_msg,
                "action": action,
                "tool_result_summary": f"{tool_name} 工具执行异常：{error_msg}",
                "tool_trace": tool_trace
            }

        if not isinstance(tool_result, dict):
            logger.info(f"[trace_id={trace_id}] [agent] 工具返回异常，tool_status:error tool_error:{type(tool_result)}")
            tool_trace.append({
                "step": "call_tool",
                "tool_name": tool_name,
                "tool_status": "error",
                "final_status": "tool_error",
                "error": f"工具返回值类型异常：{type(tool_result)}"
            })

            bad_case["answer"] = f"工具 {tool_name} 返回格式异常，无法完成本次请求。"
            bad_case["route"] =route_decision
            bad_case["error_type"] = "tool_error"
            bad_case["note"] = f"工具 {tool_name} 返回格式异常，返回值不是 dict"
            write_bad_cases(bad_case)

            return {
                "status": "TOOL_ERROR",
                "answer": f"工具 {tool_name} 返回格式异常，无法完成本次请求。",
                "references": [],
                "retrieval_confidence": 0.0,
                "answer_confidence": 0.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": "工具返回值不是 dict",
                "action": action,
                "tool_result_summary": f"{tool_name} 工具返回格式异常",
                "tool_trace": tool_trace
            }

        if tool_result.get("status") != "success":
            error_msg = tool_result.get("error", f"{tool_name} 工具执行失败")

            logger.warning(
                f"[trace_id={trace_id}] [tool] tool_name={tool_name} "
                f"tool_status=error tool_error={error_msg}"
            )

            tool_trace.append({
                "step": "call_tool",
                "tool_name": tool_name,
                "tool_status": "error",
                "final_status": "tool_error",
                "error": error_msg
            })

            bad_case["answer"] = f"工具 {tool_name} 执行失败，无法完成本次请求。"
            bad_case["route"] = route_decision
            bad_case["error_type"] = "tool_error"
            bad_case["note"] = f"工具 {tool_name} 执行失败：{error_msg}"
            write_bad_cases(bad_case)

            return {
                "status": "TOOL_ERROR",
                "answer": f"工具 {tool_name} 执行失败，无法完成本次请求。",
                "references": [],
                "retrieval_confidence": 0.0,
                "answer_confidence": 0.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": error_msg,
                "action": action,
                "tool_result_summary": summarize_tool_result(tool_name, tool_result),
                "tool_trace": tool_trace
            }

        logger.info(
            f"[trace_id={trace_id}] [tool] tool_name={tool_name} "
            f"tool_status=success tool_error="
        )
        answer = format_tool_answer(tool_name, tool_result.get("output", {}))

        tool_trace.append({
            "step": "final_answer",
            "tool_name": tool_name,
            "tool_status": "success",
            "final_status": "ok",
            "answer_preview": answer[:80]
        })

        return {
            "status": "OK",
            "answer": answer,
            "references": [],
            "retrieval_confidence": 0.0,
            "answer_confidence": 1.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": f"Agent 选择工具：{tool_name}",
            "action": action,
            "tool_result_summary": summarize_tool_result(tool_name, tool_result),
            "tool_trace": tool_trace
        }

    # =========================
    # 5. 走知识库 RAG
    # =========================
    if action == "search_knowledge_base":
        logger.info(f"[trace_id={trace_id}] [retrieve] 开始检索 top_k={top_k}")
        retrieve_start = time.perf_counter()

        retrieved_chunks = search_relevant_chunks(question, top_k)

        retrieve_cost = int((time.perf_counter() - retrieve_start) * 1000)

        logger.info(f"[trace_id={trace_id}] [retrieve] retrieve_cost_ms={retrieve_cost}")

        rag_answer_start = time.perf_counter()

        if not retrieved_chunks:

            logger.warning(
                f"[trace_id={trace_id}] [retrieve] retrieved_count=0 "
                f"top_chunk_ids=[] top_scores=[]"
            )

            tool_trace.append({
                "step": "call_tool",
                "tool_name": "search_knowledge_base",
                "tool_args": {
                    "query": question,
                    "top_k": top_k
                },
                "tool_status": "skipped",
                "final_status": "no_evidence",
                "success": False,
                "result_summary": "没有检索到候选 chunk"
            })

            return {
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
            }

        retrieved_count = len(retrieved_chunks)
        retrieved_chunk_ids = [chunk.get("chunk_id") for chunk in retrieved_chunks]

        logger.info(
            f"[trace_id={trace_id}] [retrieve] retrieved_count={retrieved_count} "
            f"top_chunk_ids={retrieved_chunk_ids}"
        )

        rag_result = ask(question, top_k=top_k, retrieved_chunks=retrieved_chunks,
                         decompose_top_k=decompose_top_k,trace_id=trace_id)

        final_chunks = rag_result.get("final_chunks", [])
        final_chunk_count = len(final_chunks)
        top_chunk_ids = [chunk.get("chunk_id") for chunk in final_chunks]
        top_scores = [
            round(chunk.get("hybrid_score", 0), 4)
            for chunk in final_chunks]

        hit_brief = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "rerank": round(chunk.get("rerank_score", 0), 4),
                "score": round(chunk.get("hybrid_score", 0), 4),
                "bm25": round(chunk.get("bm25_score", 0), 4),
                "vector": round(chunk.get("vector_score", 0), 4),
                "file_name": chunk.get("file_name"),
            }
            for chunk in final_chunks
        ]

        tool_trace.append({
            "step": "call_tool",
            "tool_name": "search_knowledge_base",
            "tool_args": {
                "query": question,
                "top_k": top_k
            },
            "tool_status": "success",
            "success": True,
            "final_chunk_count": final_chunk_count,
            "top_chunk_ids": top_chunk_ids,
            "top_scores": top_scores,
            "result_summary": f"检索到 {len(final_chunks)} 个候选 chunk",
            "hits": hit_brief
        })

        tool_trace.append({
            "step": "format_context",
            "summary": f"将 {len(final_chunks)} 个 chunk 交给 RAG 回答逻辑"
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
            reason_text = rag_result.get("reason", "")

            is_bad_reference, reference_note = should_record_reference_bad_case(
                question=question,
                answer=answer_text,
                references=references,
                reason=reason_text
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

        return rag_result

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

    return {
        "status": "MODEL_BAD_OUTPUT",
        "answer": "Agent 决策结果异常，无法完成本次请求。",
        "references": [],
        "retrieval_confidence": 0.0,
        "answer_confidence": 0.0,
        "need_clarification": False,
        "clarification_question": "",
        "reason": f"未知 Agent action: {action}",
        "action": action or "",
        "tool_result_summary": "未知 Agent action",
        "tool_trace": tool_trace
    }

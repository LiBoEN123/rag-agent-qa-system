from app.untils import get_logger
from app.tools import dispatch_tool
from app.rag import ask, search_relevant_chunks
import re


logger = get_logger('chat_service')

OUT_OF_SCOPE_KEYWORDS = [
    "天气", "气温", "下雨",
    "写诗", "作诗", "故事", "笑话",
    "股票", "基金", "彩票",
    "电影", "明星", "八卦",
    "数学题", "微积分", "量子力学",
    "python教程", "java教程", "怎么写代码"
]

def get_route_decision(decision:dict) -> str:
    action = decision.get("action", "")

    if action == "search_knowledge_base":
        return "rag"

    if action in ["simple_calculator", "list_documents", "get_chunk_text"]:
        return action

    if action in ["clarify", "refuse", "direct_answer"]:
        return action

    return "unknown"

def normalize_question(question: str) -> str:
    q = (question or "").strip()
    q = q.replace("？", "?")
    q = q.replace("：", ":")
    q = q.replace("，", ",")
    q = q.replace("。", "")
    return q


# Agent决策层：判断走哪个动作
def route_agent_action(question: str, top_k:int) -> dict:
    q =normalize_question(question)

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

    # 6. 明显超出知识库范围：拒答
    if is_out_of_scope(q):
        return {
            "decision": "refuse",
            "action": "refuse",
            "tool_name": "",
            "tool_args": {},
            "reason": "问题主题超出当前知识库范围"
        }

    # 7. 指代太模糊：澄清
    if q in ["这个怎么弄", "这个怎么配置", "这个是什么意思", "这个怎么解决"]:
        return {
            "decision": "clarify",
            "action": "clarify",
            "tool_name": "",
            "tool_args": {},
            "reason": "问题存在明显指代不明",
            "clarification_question": "你说的“这个”具体指哪个功能、哪份文档或哪段报错？请补充一下对象。"
        }

    # 8. 默认：当成知识库问题，走 RAG
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

def run_agent_once(question: str, top_k: int, decision: dict | None = None, trace_id : str = "") -> dict:
    """
    执行一次最小 Agent 流程。

    这版是和当前 main.py 适配的最小改动版：
    1. 不重复校验 top_k，因为接口层已经做了。
    2. 不引入 trace 大字典，继续使用 tool_trace。
    3. 保持原来的大写业务状态，方便 main.py 分流。
    4. 补上工具异常、工具失败、RAG 空结果时的可读返回。
    """

    # =========================
    # 1. 先做路由决策
    # =========================
    if decision is None:
        decision = route_agent_action(question, top_k)

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
            "reason": decision.get("reason", "")
        }
    ]

    logger.info(
        f"[trace_id={trace_id}] [agent] route_decision={route_decision} "
        f"decision={decision_type} action={action} "
        f"tool_name={tool_name} reason={decision.get('reason')} "
        f"question={(question or '')[:50]}"
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
            "confidence": 0.0,
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
            "confidence": 1.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": decision.get("reason", "直接回答系统能力问题"),
            "action": action,
            "tool_result_summary": "未调用工具，直接回答",
            "tool_trace": tool_trace
        }

    # =========================
    # 4. 拒答：超出知识库范围
    # =========================
    if decision_type == "refuse":
        answer = "这个问题不属于当前知识库覆盖范围，我目前只能根据已收录资料回答相关问题。"

        tool_trace.append({
            "step": "final_answer",
            "tool_status": "skipped",
            "final_status": "out_of_scope",
            "answer_preview": answer[:80]
        })

        return {
            "status": "OUT_OF_SCOPE",
            "answer": answer,
            "references": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": decision.get("reason", "问题超出知识库范围"),
            "action": action,
            "tool_result_summary": "未调用工具，直接拒答",
            "tool_trace": tool_trace
        }

    # =========================
    # 5. 调用工具
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
            tool_result = dispatch_tool(tool_name, tool_args)

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

            return {
                "status": "TOOL_ERROR",
                "answer": f"工具 {tool_name} 执行异常，无法完成本次请求。",
                "references": [],
                "confidence": 0.0,
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

            return {
                "status": "TOOL_ERROR",
                "answer": f"工具 {tool_name} 返回格式异常，无法完成本次请求。",
                "references": [],
                "confidence": 0.0,
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

            return {
                "status": "TOOL_ERROR",
                "answer": f"工具 {tool_name} 执行失败，无法完成本次请求。",
                "references": [],
                "confidence": 0.0,
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
            "confidence": 1.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": f"Agent 选择工具：{tool_name}",
            "action": action,
            "tool_result_summary": summarize_tool_result(tool_name, tool_result),
            "tool_trace": tool_trace
        }

    # =========================
    # 6. 走知识库 RAG
    # =========================
    if action == "search_knowledge_base":
        logger.info(f"[trace_id={trace_id}] [retrieve] 开始检索 top_k={top_k}")

        retrieved_chunks = search_relevant_chunks(question, top_k)

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
                "confidence": 0.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": "知识库检索结果为空",
                "action": action,
                "tool_result_summary": "知识库检索完成，但没有召回候选 chunk",
                "tool_trace": tool_trace
            }

        hit_brief = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "score": round(chunk.get("hybrid_score", 0), 4),
                "bm25": round(chunk.get("bm25_score", 0), 4),
                "vector": round(chunk.get("vector_score", 0), 4),
                "file_name": chunk.get("file_name"),
            }
            for chunk in retrieved_chunks
        ]

        retrieved_count = len(retrieved_chunks)
        top_chunk_ids = [chunk.get("chunk_id") for chunk in retrieved_chunks]
        top_scores = [
            round(chunk.get("hybrid_score", 0), 4)
            for chunk in retrieved_chunks]

        logger.info(
            f"[trace_id={trace_id}] [retrieve] retrieved_count={retrieved_count} "
            f"top_chunk_ids={top_chunk_ids} top_scores={top_scores} hits={hit_brief}"
        )

        tool_trace.append({
            "step": "call_tool",
            "tool_name": "search_knowledge_base",
            "tool_args": {
                "query": question,
                "top_k": top_k
            },
            "tool_status": "success",
            "success": True,
            "retrieved_count": retrieved_count,
            "top_chunk_ids": top_chunk_ids,
            "top_scores": top_scores,
            "result_summary": f"检索到 {len(retrieved_chunks)} 个候选 chunk",
            "hits": hit_brief
        })

        tool_trace.append({
            "step": "format_context",
            "summary": f"将 {len(retrieved_chunks)} 个 chunk 交给 RAG 回答逻辑"
        })

        rag_result = ask(question, top_k=top_k, retrieved_chunks=retrieved_chunks, trace_id=trace_id)

        rag_status = rag_result.get("status", "OK")

        if rag_status == "NO_RELEVANT_INFO":
            final_status = "no_evidence"
        elif rag_status == "NEED_CLARIFICATION":
            final_status = "clarify"
        elif rag_status == "MODEL_BAD_OUTPUT":
            final_status = "error"
        else:
            final_status = "ok"

        tool_trace.append({
            "step": "final_answer",
            "tool_status": "skipped",
            "final_status": final_status,
            "answer_preview": rag_result.get("answer", "")[:80]
        })

        rag_result["action"] = action
        rag_result["tool_result_summary"] = f"知识库检索完成，召回 {len(retrieved_chunks)} 个 chunk"
        rag_result["tool_trace"] = tool_trace

        return rag_result

    # =========================
    # 7. 防御分支：理论上不应该走到这里
    # =========================
    tool_trace.append({
        "step": "final_answer",
        "tool_status": "skipped",
        "final_status": "error",
        "error": f"未知 Agent action: {action}"
    })

    return {
        "status": "MODEL_BAD_OUTPUT",
        "answer": "Agent 决策结果异常，无法完成本次请求。",
        "references": [],
        "confidence": 0.0,
        "need_clarification": False,
        "clarification_question": "",
        "reason": f"未知 Agent action: {action}",
        "action": action or "",
        "tool_result_summary": "未知 Agent action",
        "tool_trace": tool_trace
    }

def format_tool_answer(tool_name: str, output: dict) -> str:
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

def extract_chunk_id(q: str) -> int | None:
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


def is_out_of_scope(question:str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    return any(keyword in q for keyword in OUT_OF_SCOPE_KEYWORDS)

# 简单计算判断函数
def extract_simple_math_expression(question:str) -> str |None:
    q = (question or "").strip()

    # 去掉一些常见中文表达
    remove_words = [
    "请帮我算一下",
    "帮我算一下",
    "请计算",
    "计算",
    "结果是多少",
    "等于多少",
    "是多少",
    "？",
    "?",
    "，",
    ",",
    "。"
]

    for word in remove_words:
        q = q.replace(word, "")

    q = q.strip()

    # 只保留数字，小数点，四则运算符和括号
    allowed_chars = set("0123456789.+-*/() ")
    if q and all(ch in allowed_chars for ch in q):
        if any(op in q for op in ["+", "-", "*", "/"]):
            return q

    return None
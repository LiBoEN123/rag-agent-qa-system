from app.untils import get_logger
from app.tools import dispatch_tool
from app.rag import ask, search_relevant_chunks


logger = get_logger('chat_service')

OUT_OF_SCOPE_KEYWORDS = [
    "天气", "气温", "下雨",
    "写诗", "作诗", "故事", "笑话",
    "股票", "基金", "彩票",
    "电影", "明星", "八卦",
    "数学题", "微积分", "量子力学",
    "python教程", "java教程", "怎么写代码"
]

# Agent决策层：判断走哪个动作
def route_agent_action(question: str, top_k:int) -> dict:
    q = (question or "").strip()

    # 1. 空问题：需要澄清
    if not q:
        return {
            "action": "clarify",
            "tool_args": {},
            "reason": "用户未输入问题",
            "clarification_question": "请直接输入你想查询的具体问题，例如：每个 Vue 应用是通过什么函数创建实例的？"
        }

    # 2. 用户问系统能力：可以直接回答，不需要查知识库
    if q in ["你是谁?", "你能做什么?", "你可以做什么?", "你有什么功能?"]:
        return {
            "action": "direct_answer",
            "tool_args": {},
            "reason": "用户询问系统能力，可以直接回答"
        }

    # 3. 用户问知识库文档数量或列表：调用 list_documents
    if "文档" in q and any(x in q for x in ["几个", "多少", "数量", "列表", "有哪些"]):
        return {
            "action": "list_documents",
            "tool_args": {},
            "reason": "用户询问知识库文档列表或文档数量"
        }

    # 4. 用户输入简单数学表达式：调用 simple_calculator
    expression = extract_simple_math_expression(question)
    if expression is not None:
        return {
            "action": "simple_calculator",
            "tool_args": {
                "expression": expression
            },
            "reason": "用户问题是简单四则运算"
        }

    # 5. 明显超出知识库范围：拒答
    if is_out_of_scope(q):
        return {
            "action": "refuse",
            "tool_args": {},
            "reason": "问题主题超出当前知识库范围"
        }

    # 6. 指代太模糊：澄清
    if q in ["这个怎么弄", "这个怎么配置", "这个是什么意思", "这个怎么解决"]:
        return {
            "action": "clarify",
            "tool_args": {},
            "reason": "问题存在明显指代不明",
            "clarification_question": "你说的“这个”具体指哪个功能、哪份文档或哪段报错？请补充一下对象。"
        }

    # 7. 默认：当成知识库问题，走 RAG
    return {
        "action": "search_knowledge_base",
        "tool_args": {
            "query": q,
            "top_k": top_k
        },
        "reason": "默认作为知识库问答问题处理"
    }

def run_agent_once(question: str, top_k: int, decision: dict | None=None) -> dict:
    decision = decision or route_agent_action(question, top_k)
    action = decision['action']

    logger.info(
        f"[agent] action={action} reason={decision.get('reason')} question={question[:50]}"
    )

    # 1. 需要澄清
    if action == "clarify":
        return {
            "status": "NEED_CLARIFICATION",
            "answer": "",
            "references": [],
            "confidence": 0.0,
            "need_clarification": True,
            "clarification_question": decision.get("clarification_question", ""),
            "reason": decision.get("reason", "问题需要澄清")
        }

    # 2. 直接回答
    if action == "direct_answer":
        return {
            "status": "OK",
            "answer": (
                "我是一个基于当前知识库的问答助手。"
                "我可以根据已加载文档回答问题，也可以列出知识库文档，"
                "并处理简单四则运算。对于知识库没有覆盖的问题，我会说明无法根据文档确定。"
            ),
            "references": [],
            "confidence": 1.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": decision.get("reason", "直接回答系统能力问题")
        }

    # 3. 拒答
    if action == "refuse":
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": "这个问题不属于当前知识库覆盖范围，我目前只能根据已收录资料回答相关问题。",
            "references": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": decision.get("reason", "问题超出知识库范围")
        }

    # 4. 调用 list_documents 工具
    if action == "list_documents":
        tool_result = dispatch_tool(action, decision.get("tool_args", {}))

        if tool_result.get("status") != "success":
            return {
                "status": "MODEL_BAD_OUTPUT",
                "answer": "",
                "references": [],
                "confidence": 0.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": tool_result.get("error", "list_documents 工具执行失败")
            }

        output = tool_result.get("output", {})
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

        return {
            "status": "OK",
            "answer": answer,
            "references": [],
            "confidence": 1.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "命中文档列表工具"
        }
    # 5. 调用 simple_calculator 工具
    if action == "simple_calculator":
        tool_result = dispatch_tool("simple_calculator", decision["tool_args"])

        if tool_result.get("status") != "success":
            return {
                "status": "MODEL_BAD_OUTPUT",
                "answer": "",
                "references": [],
                "confidence": 0.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": tool_result.get("error", "计算工具执行失败")
            }

        output = tool_result.get("output", {})
        answer = output.get("result")

        return {
            "status": "OK",
            "answer": f"计算结果是：{answer}",
            "references": [],
            "confidence": 1.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "Agent 选择 simple_calculator 工具"
        }

    # 6. 调用 search_knowledge_base，也就是原来的 RAG
    if action == "search_knowledge_base":
        logger.info(f"[retrieve] 开始检索 top_k={top_k}")
        retrieved_chunks = search_relevant_chunks(question, top_k)

        hit_brief = [
            {
                "chunk_id": chunk["chunk_id"],
                "score": round(chunk["hybrid_score"], 4),
                "bm25": round(chunk["bm25_score"], 4),
                "vector": round(chunk["vector_score"], 4),
                "file_name": chunk["file_name"],
            }
            for chunk in retrieved_chunks
        ]

        logger.info(
            f"[retrieve] 检索完成 chunks_len={len(retrieved_chunks)} hits={hit_brief}"
        )

        logger.info("[answer] 开始调用 RAG 问答逻辑")
        return ask(question, top_k=top_k, retrieved_chunks=retrieved_chunks)

    # 7. 防御分支：理论上不应该走到这里
    return {
        "status": "MODEL_BAD_OUTPUT",
        "answer": "",
        "references": [],
        "confidence": 0.0,
        "need_clarification": False,
        "clarification_question": "",
        "reason": f"未知 Agent action: {action}"
    }

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
        "是多少",
        "等于多少",
        "计算",
        "帮我算一下",
        "请帮我算一下",
        "请计算",
        "结果是多少",
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
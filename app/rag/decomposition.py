"""复杂问题拆分模块。

通过规则判断拆分必要性，调用拆分模型，并规范化模型返回的子查询结构。
"""

import re

from app.llm.factory import get_decompose_model
from app.llm.gateway import ModelGateway
from app.llm.schemas import QueryDecomposeOutput
from app.prompts import DECOMPOSE_PROMPT


def default_decompose_result(reason: str = "query decomposition skipped") -> dict:
    """构造无需问题拆分时使用的统一默认结果。"""
    return {"need_decompose": False, "reason": reason, "sub_queries": []}


def should_decompose_query(question: str) -> tuple[bool, str]:
    """用轻量规则判断问题是否值得调用拆分模型。"""
    query = (question or "").strip()
    if not query:
        return False, "empty query"
    if query.replace("？", "?").count("?") >= 2:
        return True, "multiple question marks"
    keywords = [
        "分别", "同时", "以及", "对比", "区别", "新旧", "变化", "调整",
        "修订", "流程和条件", "条件和材料", "入口和时间",
    ]
    for keyword in keywords:
        if keyword in query:
            return True, f"multi-intent keyword: {keyword}"
    if len(set(re.findall(r"(?:19|20)\d{2}", query))) >= 2:
        return True, "compares multiple years"
    version_pattern = (
        r"(新版|旧版|新政策|旧政策|新规|旧规|新办法|旧办法|现行|原规定|"
        r"以前|之前|修订前|修订后|调整前|调整后).{0,20}"
        r"(对比|区别|变化|调整|修订|相比|不同|差异)|"
        r"(对比|区别|变化|调整|修订|相比|不同|差异).{0,20}"
        r"(新版|旧版|新政策|旧政策|新规|旧规|新办法|旧办法|现行|原规定|"
        r"以前|之前|修订前|修订后|调整前|调整后)"
    )
    if re.search(version_pattern, query):
        return True, "compares policy versions"
    intents = (
        "怎么|如何|什么|哪些|是否|能否|有没有|多少|哪里|入口|时间|"
        "条件|材料|流程|要求|办法"
    )
    if re.search(
        rf"({intents}).{{0,30}}(，|,|；|;|、|和|及|并且|同时).{{0,30}}({intents})",
        query,
    ):
        return True, "multiple explicit intents"
    if len(query) <= 30:
        return False, "short single-intent query"
    return False, "single-intent query"


def compact_text(value: str) -> str:
    """把已经通过 Schema 校验的文本压缩为单行。"""
    return re.sub(r"\s+", " ", value).strip()


def normalize_decompose_result(result: QueryDecomposeOutput) -> dict:
    """应用最多五条和单行文本等拆分业务规则。"""
    reason = compact_text(result.reason)
    if not result.need_decompose:
        return default_decompose_result(reason or "decompose model decided single query")
    sub_queries = []
    for item in result.sub_queries[:5]:
        sub_question = compact_text(item.sub_question)
        search_query = compact_text(item.search_query)
        item_id = (
            item.id
            if "id" in item.model_fields_set
            else len(sub_queries) + 1
        )
        sub_queries.append(
            {
                "id": item_id,
                "sub_question": sub_question,
                "search_query": search_query,
                "must_answer": item.must_answer,
            }
        )
    if not sub_queries:
        return default_decompose_result("decompose model returned no valid sub_queries")
    return {
        "need_decompose": True,
        "reason": reason or "decompose model decided multi-query retrieval",
        "sub_queries": sub_queries,
    }


def decompose_query_with_llm(question: str, trace_id: str = ""):
    """调用问题拆分模型并返回规范化的子查询结果。"""
    gateway = ModelGateway(
        role="decompose",
        langchain_model=get_decompose_model(),
    )
    result = gateway.invoke_structured(
        [
            {"role": "system", "content": DECOMPOSE_PROMPT},
            {"role": "user", "content": question},
        ],
        QueryDecomposeOutput,
        trace_id=trace_id,
    )
    return normalize_decompose_result(result)

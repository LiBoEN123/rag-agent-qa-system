import json
from typing import List

import requests

from app.config import Settings
from app.prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, USER_PROMPT, NO_ANSWER_FALLBACK
from app.rag_retrieval import clamp01, make_snippet
from app.untils import get_logger

logger = get_logger('chat_service')


def build_context(retrieved_chunks, max_chars: int = Settings.MAX_CONTEXT_CHARS):
    parts = []
    total = 0

    for item in retrieved_chunks:
        snippet = item["text"][:300]
        block = (
            f"[{item['reference_id']}]\n"
            f"file_name:{item['file_name']}\n"
            f"score:{item['hybrid_score']}\n"
            f"chunk_id:{item['chunk_id']}\n"
            f"snippet:{snippet}\n"
        )

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts)


def call_llm(question: str, context: str, trace_id : str = ""):
    api_key = Settings.LLM_API_KEY
    model_name = Settings.MODEL_NAME
    api_url = Settings.LLM_API_URL

    if not api_key:
        raise ValueError("请先设置模型 API Key")

    payload = {
        "model": model_name,
        "temperature": Settings.TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_A},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    context=context,
                    question=question
                )
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()

    data = resp.json()

    logger.info(
        f"[trace_id={trace_id}] [llm] response_model={data.get('model')}, "
        f"response_id={data.get('id', '')}"
    )

    content = data["choices"][0]["message"]["content"]
    return content


def parse_llm_output(llm_output: str):
    llm_output = llm_output.strip()

    if llm_output.startswith("```"):
        llm_output = llm_output.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(llm_output)
        answer = data.get("answer", NO_ANSWER_FALLBACK)
        references = data.get("references", [])
        if not isinstance(references, list):
            references = []

        return {
            "ok": True,
            "answer": answer,
            "references": references
        }
    except Exception:
        return {
            "ok": False,
            "answer": NO_ANSWER_FALLBACK,
            "references": []
        }


def format_reference(retrieved_chunks, selected_reference: list):
    ref_map = {item["reference_id"]: item for item in retrieved_chunks}

    final_refs = []
    for ref_id in selected_reference:
        if ref_id in ref_map:
            item = ref_map[ref_id]
            final_refs.append({
                "reference_id": item["reference_id"],
                "file_name": item["file_name"],
                "source_path": item["source_path"],
                "chunk_id": item["chunk_id"],
                "position": item.get("position"),
                "snippet": make_snippet(item["text"]),
                "score": round(float(item.get("hybrid_score", 0.0)), 4)
            })

    return final_refs


def estimate_confidence(
    retrieved_chunks: List[dict],
    final_references: List[dict],
    status: str
) -> float:
    if status != "OK" or not retrieved_chunks:
        return 0.0

    top1_vector = clamp01(retrieved_chunks[0].get("vector_score", 0.0))

    scope_gap = 0.0
    if len(retrieved_chunks) > 1:
        scope_gap = clamp01(
            retrieved_chunks[0].get("vector_score", 0.0) -
            retrieved_chunks[1].get("vector_score", 0.0)
        )

    ref_ratio = clamp01(len(final_references) / max(1, min(len(retrieved_chunks), 5)))

    confidence = 0.65 * top1_vector + 0.20 * scope_gap + 0.15 * ref_ratio
    return round(clamp01(confidence), 3)
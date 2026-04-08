from app.config import Settings
from app.prompts import EXTRACT_USER_PROMPT, EXTRACT_SYSTEM_PROMPT
import requests
import json

def call_llm(text: str):
    api_key = Settings.LLM_API_KEY
    model_name = Settings.MODEL_NAME  # 默认模型改为 DeepSeek 的聊天模型
    api_url = Settings.LLM_API_URL  # 默认 URL 改为 DeepSeek 的端点

    if not api_key:
        raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")

    pay_load = {
        'model':model_name,
        'temperature':Settings.TEMPERATURE,
        'messages':[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": 'user', "content": EXTRACT_USER_PROMPT.format(
                text=text
                )
             }
        ]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    resp = requests.post(api_url, headers=headers, json=pay_load, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content

def parse_llm_output(llm_output:str):
    llm_output = llm_output.strip()

    if llm_output.startswith("```"):
        llm_output = llm_output.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(llm_output)
        is_refundable = data.get("is_refundable", None)
        refund_window_days = data.get("refund_window_days", None)
        exceptions = data.get("exceptions", [])
        evidence = data.get("evidence", [])
        return {
            "ok":True,
            "is_refundable":is_refundable,
            "refund_window_days":refund_window_days,
            "exceptions":exceptions,
            "evidence":evidence
        }
    except Exception:
        return {
            "ok":False,
            "is_refundable":None,
            "refund_window_days":None,
            "exceptions":[],
            "evidence":[]
        }

def normalize_extract_result(data: dict):
    if not isinstance(data, dict):
        return {
            "is_refundable": None,
            "refund_window_days": None,
            "exceptions": [],
            "evidence": []
        }

    is_refundable = data.get("is_refundable", None)
    if not isinstance(is_refundable, (bool, type(None))):
        is_refundable = None

    refund_window_days = data.get("refund_window_days", None)
    if not isinstance(refund_window_days, (int, type(None))):
        refund_window_days = None

    exceptions = data.get("exceptions", [])
    if exceptions is None:
        exceptions = []
    elif isinstance(exceptions, str):
        exceptions = [exceptions]
    elif not isinstance(exceptions, list):
        exceptions = []

    evidence = data.get("evidence", [])
    if evidence is None:
        evidence = []
    elif isinstance(evidence, str):
        evidence = [evidence]
    elif not isinstance(evidence, list):
        evidence = []

    return {
        "is_refundable": is_refundable,
        "refund_window_days": refund_window_days,
        "exceptions": exceptions,
        "evidence": evidence
    }

def extract_refund_policy(text:str):
    llm_out = call_llm(text)
    parase = parse_llm_output(llm_out)

    if not parase['ok']:
        return {
            "status":"MODEL_BAD_OUTPUT",
            "is_refundable":None,
            "refund_window_days":None,
            "exceptions":[],
            "evidence":[]
        }

    norm_data = normalize_extract_result(parase)

    is_refundable = norm_data.get("is_refundable", None)
    refund_window_days = norm_data.get("refund_window_days", None)
    exceptions = norm_data.get("exceptions", [])
    evidence = norm_data.get("evidence", [])

    return {
        "status": "OK",
        "is_refundable": norm_data.get("is_refundable", None),
        "refund_window_days": norm_data.get("refund_window_days", None),
        "exceptions": norm_data.get("exceptions", []),
        "evidence": norm_data.get("evidence", [])
    }
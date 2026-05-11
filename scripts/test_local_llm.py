import requests


def chat_once(question: str) -> str:
    url = "http://127.0.0.1:11434/v1/chat/completions"

    payload = {
        "model": "qwen2:7b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个 RAG / Agent 项目助手。"
                    "这里的 RAG 指 Retrieval-Augmented Generation，中文叫检索增强生成。"
                    "请只用一句中文回答。"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.2,
        "stream": False
    }

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": "Bearer ollama"
    }

    # 关键：trust_env=False 可以避免 Python requests 误走系统代理
    session = requests.Session()
    session.trust_env = False

    response = session.post(
        url,
        json=payload,
        headers=headers,
        timeout=120
    )

    print("status_code:", response.status_code)
    print("response_text:", response.text[:1000])

    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    question = "用一句话解释什么是 RAG。"
    answer = chat_once(question)

    print("问题：", question)
    print("回答：", answer)

























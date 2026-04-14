import json
import time
from app.rag import preview_docs, get_chunk_full_text

def list_documents():
    docs = preview_docs()
    return {
        "count": len(docs),
        "documents": docs
    }

def get_chunk_text(chunk_id: int | str):
    return get_chunk_full_text(chunk_id)

def append_tool_log(record: dict):
    with open("tool_calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def dispatch_tool(tool_name: str, tool_args: dict):
    start = time.time()
    try:
        if tool_name == "list_documents":
            result = list_documents()
        elif tool_name == "get_chunk_text":
            result = get_chunk_text(tool_args.get("chunk_id"))
        else:
            result = {"error": f"unknown tool: {tool_name}"}

        append_tool_log({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": result,
            "elapsed_ms": int((time.time() - start) * 1000),
            "success": "error" not in result
        })
        return result

    except Exception as e:
        err = {"error": str(e)}
        append_tool_log({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": err,
            "elapsed_ms": int((time.time() - start) * 1000),
            "success": False
        })
        return err













































from app.rag.retrieval import make_snippet


def preview_docs(docs_registry):
    """原样返回文档注册信息，用于调试接口展示。"""
    return docs_registry


def preview_chunks(chunks, max_len:int, limit: int = 70, file_name: str | None = None):
    """生成 chunk 预览数据，可按文件名过滤并限制长度。"""
    result = []

    for item in chunks:
        if file_name and item["metadata"].get("file_name") != file_name:
            continue

        result.append({
            "file_name": item["metadata"].get("file_name"),
            "chunk_id": item["metadata"].get("chunk_id"),
            "source_path": item["metadata"].get("source_path"),
            "position": item["metadata"].get("position"),
            "text_preview": make_snippet(item.get("text", ""), max_len=max_len),
            "text_length": len(item.get("text", ""))
        })

        if len(result) >= limit:
            break

    return result


def get_chunk_full_text(chunks, chunk_id: int | str):
    """在 chunk 列表中按 chunk_id 查找完整文本和元信息。"""
    for item in chunks:
        if str(item["metadata"].get("chunk_id")) == str(chunk_id):
            return {
                "found": True,
                "file_name": item["metadata"].get("file_name"),
                "chunk_id": item["metadata"].get("chunk_id"),
                "source_path": item["metadata"].get("source_path"),
                "position": item["metadata"].get("position"),
                "text": item.get("text")
            }

    return {
        "found": False,
        "file_name": "",
        "chunk_id": chunk_id,
        "source_path": "",
        "position": None,
        "text": ""
    }

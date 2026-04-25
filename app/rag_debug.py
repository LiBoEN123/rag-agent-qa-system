from app.rag_retrieval import make_snippet


def preview_docs(docs_registry):
    return docs_registry


def preview_chunks(chunks, max_len:int, limit: int = 70, file_name: str | None = None):
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
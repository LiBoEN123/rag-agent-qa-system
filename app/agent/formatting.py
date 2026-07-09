"""Natural-language formatting for non-RAG tool results."""


def format_tool_answer(tool_name: str, output: dict) -> str:
    if tool_name == "simple_calculator":
        return f"计算结果是：{output.get('result')}"
    if tool_name == "list_documents":
        documents = output.get("documents", [])
        names = [
            document.get("source_file")
            or document.get("file_name")
            or document.get("title")
            or document.get("doc_name")
            or document.get("doc_id")
            or "未命名文档"
            for document in documents[:10]
        ]
        answer = f"当前系统中共有 {output.get('count', 0)} 个文档。"
        return answer + (" 文档包括：" + "、".join(names) if names else "")
    if tool_name == "get_chunk_text":
        chunk = output.get("chunk", {})
        text = (
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("chunk_text")
            or str(chunk)
        )
        return f"chunk_{output.get('chunk_id')} 内容如下：\n{text}"
    if tool_name == "search_knowledge_base":
        return f"知识库检索完成，共找到 {output.get('count', 0)} 个候选 chunk。"
    if tool_name == "compare_documents":
        return output.get("comparison") or output.get("summary") or str(output)
    return str(output)

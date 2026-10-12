from app.ingestion import files as rag_ingest


def test_collect_supported_files_includes_pdf_and_docx(tmp_path):
    for file_name in ["notes.txt", "guide.md", "manual.pdf", "policy.docx", "image.png"]:
        (tmp_path / file_name).write_text("content", encoding="utf-8")

    files = rag_ingest.collect_supported_files(str(tmp_path))
    file_names = [file_path.name for file_path in files]

    assert file_names == ["guide.md", "manual.pdf", "notes.txt", "policy.docx"]




def test_cleaners_preserve_local_whitespace_and_markdown_comment_semantics():
    assert rag_ingest.clean_text(
        "  alpha\t beta\r\n\r\n\r\n gamma  "
    ) == "alpha beta\n\n gamma"
    assert rag_ingest.clean_markdown_text(
        "# 标题\r\n<!-- hidden -->\r\n正文\t 内容"
    ) == "# 标题\n\n正文 内容"





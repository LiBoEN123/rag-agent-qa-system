from app.rag import ingest as rag_ingest
from app.ingestion.document_blocks import ParsedDocument, TextBlock
from tests.support.structured_document import build_table_docx


def test_collect_supported_files_includes_pdf_and_docx(tmp_path):
    for file_name in ["notes.txt", "guide.md", "manual.pdf", "policy.docx", "image.png"]:
        (tmp_path / file_name).write_text("content", encoding="utf-8")

    files = rag_ingest.collect_supported_files(str(tmp_path))
    file_names = [file_path.name for file_path in files]

    assert file_names == ["guide.md", "manual.pdf", "notes.txt", "policy.docx"]


def test_read_file_text_dispatches_supported_readers(monkeypatch):
    monkeypatch.setattr(rag_ingest, "read_txt_auto", lambda path: f"text:{path}")
    monkeypatch.setattr(rag_ingest, "read_pdf_text", lambda path: f"pdf:{path}")
    monkeypatch.setattr(rag_ingest, "read_docx_text", lambda path: f"docx:{path}")

    assert rag_ingest.read_file_text("notes.txt") == "text:notes.txt"
    assert rag_ingest.read_file_text("guide.md") == "text:guide.md"
    assert rag_ingest.read_file_text("manual.pdf") == "pdf:manual.pdf"
    assert rag_ingest.read_file_text("policy.docx") == "docx:policy.docx"


def test_cleaners_preserve_local_whitespace_and_markdown_comment_semantics():
    assert rag_ingest.clean_text(
        "  alpha\t beta\r\n\r\n\r\n gamma  "
    ) == "alpha beta\n\n gamma"
    assert rag_ingest.clean_markdown_text(
        "# 标题\r\n<!-- hidden -->\r\n正文\t 内容"
    ) == "# 标题\n\n正文 内容"


def test_read_pdf_text_preserves_page_markers(monkeypatch):
    parsed = ParsedDocument(blocks=(
        TextBlock(block_id="b0", order=0, text="first page", page_number=1),
        TextBlock(block_id="b1", order=1, text="third page", page_number=3),
    ), page_count=3)
    monkeypatch.setattr(rag_ingest, "parse_document", lambda _path: parsed)

    assert rag_ingest.read_pdf_text("manual.pdf") == (
        "[第 1 页]\n\nfirst page\n\n[第 3 页]\n\nthird page"
    )


def test_read_docx_text_preserves_paragraph_and_table_extraction(tmp_path):
    path = tmp_path / "policy.docx"
    path.write_bytes(build_table_docx())
    assert rag_ingest.read_docx_text(str(path)) == (
        "段落 A\n\n[表格]\n事项 | 条件 | 材料\n注册 | 年满18岁 | 身份证\n[/表格]\n\n段落 B"
    )

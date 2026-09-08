from types import SimpleNamespace

from app.rag import ingest as rag_ingest


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
    pages = [
        SimpleNamespace(extract_text=lambda: " first\t page "),
        SimpleNamespace(extract_text=lambda: ""),
        SimpleNamespace(extract_text=lambda: " third page "),
    ]
    monkeypatch.setattr("pypdf.PdfReader", lambda _path: SimpleNamespace(pages=pages))

    assert rag_ingest.read_pdf_text("manual.pdf") == (
        "[第 1 页]\nfirst page\n\n[第 3 页]\nthird page"
    )


def test_read_docx_text_preserves_paragraph_and_table_extraction(monkeypatch):
    document = SimpleNamespace(
        paragraphs=[SimpleNamespace(text=" first\t paragraph ")],
        tables=[
            SimpleNamespace(
                rows=[
                    SimpleNamespace(
                        cells=[
                            SimpleNamespace(text=" cell A "),
                            SimpleNamespace(text="cell\tB"),
                        ]
                    )
                ]
            )
        ],
    )
    monkeypatch.setattr("docx.Document", lambda _path: document)

    assert rag_ingest.read_docx_text("policy.docx") == (
        "first paragraph\n\ncell A | cell B"
    )

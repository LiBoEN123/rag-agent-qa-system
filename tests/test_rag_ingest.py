from app import rag_ingest


def test_collect_supported_files_includes_pdf_and_docx(tmp_path):
    for file_name in ["notes.txt", "guide.md", "manual.pdf", "policy.docx", "image.png"]:
        (tmp_path / file_name).write_text("content", encoding="utf-8")

    files = rag_ingest.collect_supported_files(str(tmp_path))
    file_names = [file_path.name for file_path in files]

    assert file_names == ["guide.md", "manual.pdf", "notes.txt", "policy.docx"]


def test_load_files_dispatches_pdf_and_docx_readers(tmp_path, monkeypatch):
    pdf_path = tmp_path / "manual.pdf"
    docx_path = tmp_path / "policy.docx"
    pdf_path.write_bytes(b"%PDF fake")
    docx_path.write_bytes(b"fake docx")

    monkeypatch.setattr(rag_ingest, "read_pdf_text", lambda file_path: f"pdf:{file_path}")
    monkeypatch.setattr(rag_ingest, "read_docx_text", lambda file_path: f"docx:{file_path}")

    raw_docs, docs_registry = rag_ingest.load_files(str(tmp_path))

    assert [doc["file_name"] for doc in raw_docs] == ["manual.pdf", "policy.docx"]
    assert [doc["file_type"] for doc in raw_docs] == ["pdf", "docx"]
    assert raw_docs[0]["text"].startswith("pdf:")
    assert raw_docs[1]["text"].startswith("docx:")
    assert [doc["file_type"] for doc in docs_registry] == ["pdf", "docx"]


def test_split_doc_treats_pdf_and_docx_as_plain_text():
    base_doc = {
        "doc_id": "doc_1",
        "file_name": "manual.pdf",
        "source_path": "/tmp/manual.pdf",
        "file_type": "pdf",
        "text": "第一段内容。\n\n第二段内容。",
    }

    pdf_chunks = rag_ingest.split_doc(base_doc, chunk_size=20, chunk_overlap=0)
    docx_chunks = rag_ingest.split_doc(
        {**base_doc, "file_name": "policy.docx", "file_type": "docx"},
        chunk_size=20,
        chunk_overlap=0,
    )

    assert pdf_chunks
    assert docx_chunks
    assert {chunk["metadata"]["file_type"] for chunk in pdf_chunks} == {"pdf"}
    assert {chunk["metadata"]["file_type"] for chunk in docx_chunks} == {"docx"}

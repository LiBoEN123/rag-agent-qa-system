"""上传 source 的读取限额、临时文件清理和真实 loader 失败边界。"""

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from pypdf import PdfWriter

from app.sources import upload
from tests.support.structured_document import build_table_docx


@pytest.mark.parametrize("filename", ["../bad.txt", "C:\\Users\\bad.txt", "/tmp/bad.txt", "bad\x00.txt"])
def test_adapter_rejects_paths_and_control_characters(filename):
    with pytest.raises(upload.UploadSourceError) as error:
        upload.load_upload_document(BytesIO(b"text"), filename=filename, max_bytes=4)
    assert error.value.code == "UPLOAD_INVALID_FILENAME"


@pytest.mark.parametrize("failure", [False, True])
def test_temporary_path_is_controlled_and_cleaned(monkeypatch, failure):
    paths = []

    def parse(path, _max_bytes):
        paths.append(path)
        assert path.name == "document.txt"
        assert path.read_bytes() == b"text"
        if failure:
            raise RuntimeError(f"private detail: {path}")
        return "text", None

    monkeypatch.setattr(upload, "_load_content", parse)
    if failure:
        with pytest.raises(upload.UploadSourceError, match="无法解析") as error:
            upload.load_upload_document(BytesIO(b"text"), filename="safe.txt", max_bytes=4)
        assert "private detail" not in str(error.value)
    else:
        document = upload.load_upload_document(
            BytesIO(b"text"), filename="safe.txt", max_bytes=4,
        )
        assert str(paths[0]) not in document.model_dump_json()
    assert not paths[0].parent.exists()


def test_size_check_reads_at_most_limit_plus_one_and_cleans_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    class BoundedReader(BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 64 * 1024
            return super().read(size)

    stream = BoundedReader(b"x" * 100_000)
    with pytest.raises(upload.UploadSourceError) as error:
        upload.load_upload_document(stream, filename="large.txt", max_bytes=70_000)
    assert error.value.code == "UPLOAD_TOO_LARGE"
    assert stream.tell() == 70_001
    assert list(tmp_path.iterdir()) == []


def test_pdf_without_text_and_encrypted_pdf_fail_closed():
    for encrypted, code in ((False, "UPLOAD_NO_TEXT"), (True, "UPLOAD_PARSE_FAILED")):
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        if encrypted:
            writer.encrypt("test-password")
        stream = BytesIO()
        writer.write(stream)
        stream.seek(0)
        with pytest.raises(upload.UploadSourceError) as error:
            upload.load_upload_document(stream, filename="scan.pdf", max_bytes=10_000)
        assert error.value.code == code


def test_docx_expansion_limit_rejects_compressed_large_payload():
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 20_000)
    assert stream.tell() < 1000
    stream.seek(0)
    with pytest.raises(upload.UploadSourceError) as error:
        upload.load_upload_document(stream, filename="large.docx", max_bytes=1000)
    assert error.value.code == "UPLOAD_TOO_LARGE"


def test_text_encoding_and_redaction_are_reused():
    content = "上传的中文资料。\n第二行正文。\n" * 10
    document = upload.load_upload_document(
        BytesIO(content.encode("gb18030")), filename="中文.TXT", max_bytes=4096,
    )
    assert document.content == content.rstrip()
    assert document.metadata["file_type"] == "txt"
    redacted = upload.load_upload_document(
        BytesIO(b"Authorization: Bearer body-secret"), filename="secret.txt", max_bytes=4096,
    )
    assert redacted.content_redacted
    assert "body-secret" not in redacted.model_dump_json()


@pytest.mark.parametrize("suffix", ["docx", "pdf"])
def test_structured_upload_uses_shared_parser_and_cleans_temp_path(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    payload = build_table_docx(merged=True) if suffix == "docx" else (
        Path(__file__).parent / "fixtures" / "structured-table.pdf"
    ).read_bytes()
    paths = []
    real_parser = upload.parse_document

    def parse(path):
        paths.append(path)
        return real_parser(path)

    monkeypatch.setattr(upload, "parse_document", parse)
    document = upload.load_upload_document(
        BytesIO(payload), filename=f"Table.{suffix.upper()}", max_bytes=2_000_000,
    )
    assert document.structured_content is not None
    assert any(block.kind == "table" for block in document.structured_content.blocks)
    assert document.structure_hash
    assert len(paths) == 1
    assert not paths[0].exists()
    assert list(tmp_path.iterdir()) == []
    assert str(paths[0]) not in document.model_dump_json()
    if suffix == "pdf":
        assert document.metadata["page_count"] == 1


@pytest.mark.parametrize("payload,filename,code", [
    (b"not a pdf", "bad.pdf", "UPLOAD_PARSE_FAILED"),
    (b"%PDF-1.4 broken", "bad.pdf", "UPLOAD_PARSE_FAILED"),
    (b"bad zip", "bad.docx", "UPLOAD_PARSE_FAILED"),
    (b"", "empty.docx", "UPLOAD_EMPTY_FILE"),
    (b"file", "image.png", "UPLOAD_UNSUPPORTED_TYPE"),
])
def test_malformed_uploads_still_fail_closed_and_clean(tmp_path, monkeypatch, payload, filename, code):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with pytest.raises(upload.UploadSourceError) as error:
        upload.load_upload_document(BytesIO(payload), filename=filename, max_bytes=1000)
    assert error.value.code == code
    assert str(tmp_path) not in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_structured_limit_failure_uses_safe_message_and_cleans(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr("app.ingestion.document_blocks.STRUCTURE_MAX_SERIALIZED_BYTES", 100)
    with pytest.raises(upload.UploadSourceError) as error:
        upload.load_upload_document(BytesIO(build_table_docx()), filename="table.docx", max_bytes=2_000_000)
    assert error.value.code == "UPLOAD_PARSE_FAILED"
    assert "段落" not in str(error.value)
    assert list(tmp_path.iterdir()) == []

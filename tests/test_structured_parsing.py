"""Docling 公共 API 与项目结构 adapter 的 focused acceptance。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling_core.types.doc import (
    BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, TableCell, TableData,
)

from app.ingestion.document_blocks import TableBlock, TextBlock, canonical_document_text
from app.ingestion import parsing
from tests.support.structured_document import (
    ADMISSION_HTML, REGISTRATION_HTML, admission_structure, build_table_docx,
)


PDF_FIXTURE = Path(__file__).parent / "fixtures" / "structured-table.pdf"


def test_real_html_admission_sections_preserve_full_width_and_repeated_headers():
    parsed = parsing.parse_html_document(ADMISSION_HTML, name="section-admission")
    [table] = parsed.blocks
    assert isinstance(table, TableBlock)
    assert (table.row_count, table.column_count) == (13, 5)
    for row in (0, 1, 5):
        [cell] = [cell for cell in table.cells if cell.row_index == row]
        assert (cell.column_index, cell.row_span, cell.column_span, cell.is_header) == (0, 1, 5, False)
    for row in (2, 6):
        cells = [cell for cell in table.cells if cell.row_index == row]
        assert [cell.column_index for cell in cells] == list(range(5))
        assert all(cell.is_header and cell.row_span == cell.column_span == 1 for cell in cells)
    # 包含每个 cell 的原始坐标、span、header 与文本；chunking fixture 不臆造 parser 输出。
    assert parsed == admission_structure()


@pytest.mark.parametrize("merged", [False, True])
def test_real_docx_reading_order_cells_and_spans(tmp_path, merged):
    path = tmp_path / "table.docx"
    path.write_bytes(build_table_docx(merged=merged))
    parsed = parsing.parse_document(path)
    assert [type(block) for block in parsed.blocks] == [TextBlock, TableBlock, TextBlock]
    before, table, after = parsed.blocks
    assert (before.text, after.text) == ("段落 A", "段落 B")
    assert [block.order for block in parsed.blocks] == [0, 1, 2]
    assert len({block.block_id for block in parsed.blocks}) == 3
    assert parsed.page_count is None
    assert all(block.page_number is None for block in parsed.blocks)
    cells = {(cell.row_index, cell.column_index): cell for cell in table.cells}
    assert cells[1, 0].text == "注册"
    assert cells[1, 1].text == "年满18岁"
    assert cells[1, 2].text == ("身份证\n证明" if merged else "身份证")
    assert cells[0, 0].is_header
    assert cells[0, 0].column_span == (2 if merged else 1)
    assert cells[1, 2].row_span == (2 if merged else 1)
    assert table.row_count == (3 if merged else 2)
    assert table.column_count == 3
    assert parsing.parse_document(path) == parsed
    content = canonical_document_text(parsed)
    assert content.index("段落 A") < content.index("[表格]") < content.index("段落 B")
    assert "注册 | 年满18岁 | 身份证" in content


def test_adapter_preserves_official_caption_provenance_and_headers():
    doc = DoclingDocument(name="fixture")
    prov = ProvenanceItem(page_no=2, bbox=BoundingBox(l=0, t=0, r=100, b=100), charspan=(0, 7))
    doc.add_text(label=DocItemLabel.TEXT, text="before", prov=prov)
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Table caption", prov=prov)
    doc.add_table(data=TableData(num_rows=1, num_cols=2, table_cells=[
        TableCell(text="merged", start_row_offset_idx=0, end_row_offset_idx=1,
                  start_col_offset_idx=0, end_col_offset_idx=2, col_span=2, row_header=True),
    ]), caption=caption, prov=prov)
    doc.add_text(label=DocItemLabel.TEXT, text="after", prov=prov)
    parsed = parsing.adapt_docling_document(doc, page_count=2)
    assert [block.kind for block in parsed.blocks] == ["text", "table", "text"]
    assert all(block.page_number == 2 for block in parsed.blocks)
    assert parsed.blocks[1].caption == "Table caption"
    assert parsed.blocks[1].cells[0].column_span == 2
    assert parsed.blocks[1].cells[0].is_header
    assert canonical_document_text(parsed).count("Table caption") == 1


def test_converter_enables_tables_and_disables_ocr_remote_and_enrichment():
    converter = parsing._document_converter()
    assert set(converter.allowed_formats) == {InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML}
    options = converter.format_to_options[InputFormat.PDF].pipeline_options
    assert options.do_table_structure
    assert not any((options.do_ocr, options.enable_remote_services, options.allow_external_plugins,
                    options.do_picture_classification, options.do_picture_description,
                    options.do_code_enrichment, options.do_formula_enrichment))
    html_options = converter.format_to_options[InputFormat.HTML].backend_options
    assert not any((html_options.enable_remote_fetch, html_options.enable_local_fetch,
                    html_options.fetch_images, html_options.render_page, html_options.infer_furniture))
    assert html_options.source_uri is None


def test_partial_conversion_is_rejected(monkeypatch):
    converter = SimpleNamespace(convert=lambda *args, **kwargs: SimpleNamespace(
        status=ConversionStatus.PARTIAL_SUCCESS,
    ))
    monkeypatch.setattr(parsing, "_document_converter", lambda: converter)
    with pytest.raises(ValueError, match="未完整成功"):
        parsing.parse_document(PDF_FIXTURE)


def test_real_pdf_table_and_page_provenance():
    parsed = parsing.parse_document(PDF_FIXTURE)
    assert parsed.page_count == 1
    tables = [block for block in parsed.blocks if isinstance(block, TableBlock)]
    assert len(tables) == 1
    table = tables[0]
    assert (table.row_count, table.column_count, table.page_number) == (3, 3, 1)
    cells = {(cell.row_index, cell.column_index): cell for cell in table.cells}
    assert cells[1, 0].text == "Registration"
    assert cells[1, 1].text == "Age 18+"
    assert cells[1, 2].text == "Identity card"
    content = canonical_document_text(parsed)
    assert content.index("Paragraph A") < content.index("[表格]") < content.index("Paragraph B")


def test_real_html_reading_order_and_official_caption_limit():
    result = parsing._document_converter().convert_string(REGISTRATION_HTML, format=InputFormat.HTML)
    assert result.status is ConversionStatus.SUCCESS
    # 2.127.0 HTML backend 未关联这个 <caption>；如实记录，不自研补齐。
    assert result.document.tables[0].captions == []
    parsed = parsing.parse_html_document(REGISTRATION_HTML, name="registration")
    assert parsed == parsing.adapt_docling_document(result.document)
    assert [type(block) for block in parsed.blocks] == [TextBlock, TableBlock, TextBlock]
    before, table, after = parsed.blocks
    assert (before.text, after.text) == ("办理说明", "其他说明")
    assert (table.row_count, table.column_count, table.caption) == (2, 3, None)
    assert parsed.page_count is None
    assert all(block.page_number is None for block in parsed.blocks)
    assert parsed == parsing.parse_html_document(REGISTRATION_HTML)


def test_real_html_spans_and_headers():
    parsed = parsing.parse_html_document('''<html><body><table>
      <tr><th rowspan="2">事项</th><th colspan="2">申请要求</th></tr>
      <tr><th>条件</th><th>材料</th></tr>
      <tr><td>企业注册</td><td>年满18岁</td><td>身份证、申请表</td></tr>
    </table></body></html>''')
    [table] = parsed.blocks
    cells = {(cell.row_index, cell.column_index): cell for cell in table.cells}
    assert (table.row_count, table.column_count) == (3, 3)
    assert cells[0, 0].row_span == 2
    assert cells[0, 1].column_span == 2
    assert all(cell.is_header == (cell.row_index < 2) for cell in table.cells)
    assert cells[2, 2].text == "身份证、申请表"


def test_html_keeps_cleaned_intro_before_first_heading():
    parsed = parsing.parse_html_document(
        "<html><body><p>办理说明</p><h2>办理要求</h2><p>其他说明</p></body></html>"
    )
    assert [block.text for block in parsed.blocks] == ["办理说明", "办理要求", "其他说明"]


@pytest.mark.parametrize("status", [ConversionStatus.PARTIAL_SUCCESS, ConversionStatus.FAILURE])
def test_html_incomplete_conversion_is_rejected(monkeypatch, status):
    def convert_string(html, *, format, name):
        assert (html, format, name) == (REGISTRATION_HTML, InputFormat.HTML, "fixture")
        return SimpleNamespace(status=status)

    monkeypatch.setattr(parsing, "_document_converter", lambda: SimpleNamespace(convert_string=convert_string))
    with pytest.raises(ValueError, match="未完整成功"):
        parsing.parse_html_document(REGISTRATION_HTML, name="fixture")


def test_html_does_not_fetch_resources_render_or_open_local_content(monkeypatch, tmp_path):
    from docling.backend.html_backend import HTMLDocumentBackend

    def forbidden(*args, **kwargs):
        pytest.fail("HTML must not fetch, render or read local resource content")

    # connect 拦住公网；这两个 backend 入口覆盖本地/远程图片和浏览器渲染。
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr(HTMLDocumentBackend, "_create_image_ref", forbidden)
    monkeypatch.setattr(HTMLDocumentBackend, "_render_with_browser", forbidden)
    local = tmp_path / "secret.txt"
    local.write_text("LOCAL_RESOURCE_MUST_NOT_BE_READ")
    html = REGISTRATION_HTML.replace("</body>", f'''
      <img src="https://example.com/remote.png"><img src="{local.as_uri()}">
      <img src="relative.png"><link rel="stylesheet" href="https://example.com/a.css">
      <iframe src="{local.as_uri()}"></iframe><script src="https://example.com/a.js"></script>
    </body>''')
    assert parsing.parse_html_document(html) == parsing.parse_html_document(REGISTRATION_HTML)

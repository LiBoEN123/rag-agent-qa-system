"""Docling 公共 API 与项目结构 adapter 的 focused acceptance。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling_core.types.doc import (
    BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, TableCell, TableData,
)

from app.ingestion.document_blocks import TableBlock, TextBlock, canonical_document_text
from app.rag import parsing
from tests.support.structured_document import build_table_docx


PDF_FIXTURE = Path(__file__).parent / "fixtures" / "structured-table.pdf"


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
    assert set(converter.allowed_formats) == {InputFormat.PDF, InputFormat.DOCX}
    options = converter.format_to_options[InputFormat.PDF].pipeline_options
    assert options.do_table_structure
    assert not any((options.do_ocr, options.enable_remote_services, options.allow_external_plugins,
                    options.do_picture_classification, options.do_picture_description,
                    options.do_code_enrichment, options.do_formula_enrichment))


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

"""PDF/DOCX/HTML 共用的 Docling infrastructure；只映射公开结构，不识别表格。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

from app.ingestion.document_blocks import ParsedDocument, TableBlock, TableCell, TextBlock


@lru_cache(maxsize=1)
def _document_converter():
    # 惰性创建使 TXT/MD 不初始化模型；Docling 自己复用已初始化的 pipeline。
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.backend_options import HTMLBackendOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, HTMLFormatOption, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=False, do_table_structure=True,
        enable_remote_services=False, allow_external_plugins=False,
        do_picture_classification=False, do_picture_description=False,
        do_code_enrichment=False, do_formula_enrichment=False,
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.HTML: HTMLFormatOption(backend_options=HTMLBackendOptions(
                enable_remote_fetch=False, enable_local_fetch=False,
                fetch_images=False, render_page=False,
                # Firecrawl 已清理正文，不能把首个 heading 前的正文推断为页眉。
                infer_furniture=False,
            )),
        },
    )


def adapt_docling_document(document: DoclingDocument, *, page_count: int | None = None) -> ParsedDocument:
    """保留官方 reading order、cell offsets/spans、header、caption 和起始页。"""
    from docling_core.types.doc import TableItem, TextItem

    blocks: list[TextBlock | TableBlock] = []
    caption_refs = {ref.cref for table in document.tables for ref in table.captions}
    table_level: int | None = None
    for item, level in document.iterate_items():
        if table_level is not None and level <= table_level:
            table_level = None
        if not isinstance(item, (TextItem, TableItem)) or item.self_ref in caption_refs:
            continue
        # rich cell 的文字已在 cells 中，不能再作为独立段落重复输出。
        if table_level is not None and isinstance(item, TextItem):
            continue
        order = len(blocks)
        common = {
            "block_id": f"b{order}", "order": order,
            "page_number": item.prov[0].page_no if item.prov else None,
        }
        if isinstance(item, TableItem):
            blocks.append(TableBlock(
                **common, caption=item.caption_text(document) or None,
                row_count=item.data.num_rows, column_count=item.data.num_cols,
                cells=tuple(TableCell(
                    text=cell.text,
                    row_index=cell.start_row_offset_idx,
                    column_index=cell.start_col_offset_idx,
                    row_span=cell.row_span, column_span=cell.col_span,
                    is_header=cell.column_header or cell.row_header,
                ) for cell in item.data.table_cells),
            ))
            table_level = level
        else:
            blocks.append(TextBlock(**common, text=item.text))
    return ParsedDocument(blocks=tuple(blocks), page_count=page_count)


def parse_document(file_path: str | Path) -> ParsedDocument:
    """只解析本地 PDF/DOCX；损坏或部分转换失败时不接收残缺文档。"""
    from docling.datamodel.base_models import ConversionStatus

    path = Path(file_path)
    if path.suffix.lower() not in {".pdf", ".docx"} or not path.is_file():
        raise ValueError("structured parser 只接受本地 PDF/DOCX 文件")
    result = _document_converter().convert(path, raises_on_error=True)
    if result.status is not ConversionStatus.SUCCESS:
        raise ValueError("structured document conversion 未完整成功")
    return adapt_docling_document(
        result.document,
        page_count=len(result.document.pages) if path.suffix.lower() == ".pdf" else None,
    )


def parse_html_document(html: str, *, name: str | None = None) -> ParsedDocument:
    """只消费已抓取的 cleaned HTML 字符串；不传 URL、不落临时文件。"""
    from docling.datamodel.base_models import ConversionStatus, InputFormat

    result = _document_converter().convert_string(html, format=InputFormat.HTML, name=name)
    if result.status is not ConversionStatus.SUCCESS:
        raise ValueError("structured HTML conversion 未完整成功")
    return adapt_docling_document(result.document, page_count=None)

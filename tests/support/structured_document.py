"""小型真实 DOCX/HTML fixture，避免提交不必要的二进制或网页快照。"""

from io import BytesIO

from docx import Document

from app.ingestion.document_blocks import ParsedDocument, TableBlock, TableCell


REGISTRATION_HTML = """<html><body>
<p>办理说明</p>
<table><caption>企业登记办理要求</caption>
<thead><tr><th>事项</th><th>条件</th><th>材料</th></tr></thead>
<tbody><tr><td>企业注册</td><td>年满18岁</td><td>身份证、申请表</td></tr></tbody>
</table><p>其他说明</p>
</body></html>"""


def registration_structure() -> ParsedDocument:
    """已持久化合同形态的中文表格，供 chunking 与真实索引/检索测试共用。"""
    values = [
        ("事项", "条件", "材料"),
        ("企业注册", "年满18岁", "身份证、申请表"),
        ("企业变更", "企业已注册", "营业执照、变更表"),
    ]
    return ParsedDocument(page_count=2, blocks=(TableBlock(
        block_id="b0", order=0, page_number=2, caption="企业登记办理要求",
        row_count=3, column_count=3,
        cells=tuple(
            TableCell(text=text, row_index=row, column_index=column, is_header=row == 0)
            for row, values_in_row in enumerate(values)
            for column, text in enumerate(values_in_row)
        ),
    ),))


def build_table_docx(*, merged: bool = False) -> bytes:
    document = Document()
    document.add_paragraph("段落 A")
    table = document.add_table(rows=3 if merged else 2, cols=3)
    table.style = "Table Grid"
    values = [("事项", "条件", "材料"), ("注册", "年满18岁", "身份证")]
    if merged:
        values.append(("续期", "成人", "证明"))
    for row, texts in zip(table.rows, values):
        for cell, text in zip(row.cells, texts):
            cell.text = text
    if merged:
        table.cell(0, 0).merge(table.cell(0, 1))
        table.cell(1, 2).merge(table.cell(2, 2))
    document.add_paragraph("段落 B")
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()

"""小型真实 DOCX fixture，避免提交不必要的 Office binary。"""

from io import BytesIO

from docx import Document


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

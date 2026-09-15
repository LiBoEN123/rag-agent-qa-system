"""把已校验的 ParsedDocument 按 reading order 转为检索 chunks，不访问源文件。"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.document_blocks import ParsedDocument, TableBlock, TableCell, TextBlock


def _table_view(table: TableBlock) -> tuple[list[str], list[tuple[int, str]]]:
    """临时展开 spans；仅前导、非空且全部标记为 header 的行构成列标题带。

    合同不区分 row/column header，混合数据行中的 header cell 仍作为该行的值保留。
    缺少列标题时使用“列N”，不猜测第一行是表头；原 cells 不变。
    """
    grid: list[list[TableCell | None]] = [
        [None] * table.column_count for _ in range(table.row_count)
    ]
    for cell in table.cells:
        for row in range(cell.row_index, cell.row_index + cell.row_span):
            for column in range(cell.column_index, cell.column_index + cell.column_span):
                grid[row][column] = cell

    header_end = 0
    for row in grid:
        populated = [cell for cell in row if cell is not None and cell.text.strip()]
        if any(not cell.is_header for cell in populated):
            break
        header_end += 1

    labels = []
    for column in range(table.column_count):
        # 锚点去重防止 rowspan 重复同一标题，colspan 则贡献给覆盖的每一列。
        headers = {
            (cell.row_index, cell.column_index): cell.text
            for row in grid[:header_end]
            if (cell := row[column]) is not None and cell.text.strip()
        }
        labels.append(" > ".join(headers.values()) or f"列{column + 1}")

    rows = []
    for row_index in range(header_end, table.row_count):
        fields = []
        seen = set()
        for cell in grid[row_index]:
            if cell is None or not cell.text.strip():
                continue
            anchor = (cell.row_index, cell.column_index)
            if anchor in seen:
                continue
            seen.add(anchor)
            label = " / ".join(dict.fromkeys(
                labels[cell.column_index:cell.column_index + cell.column_span]
            ))
            fields.append(f"{label}：{cell.text}")
        if fields:
            rows.append((row_index, "；".join(fields) + "。"))
    return labels, rows


def _chunk_table_block(table: TableBlock, chunk_size: int) -> list[Document]:
    """按完整连续 body rows 分组，字符预算为 soft target，不做 row overlap。

    row_start/row_end 沿用原 TableCell.row_index：0-based、inclusive，排除列标题带。
    全空行不生成证据；只有标题/表头、没有数据行的表仍保留，省略 row 范围。
    """
    if not (table.caption or "").strip() and not any(cell.text.strip() for cell in table.cells):
        return []
    labels, rows = _table_view(table)
    prefix = "\n".join([
        "[表格]",
        *([f"标题：{table.caption}"] if table.caption else []),
        *(["表头：" + "；".join(labels)] if labels else []),
    ])
    suffix = "\n[/表格]"
    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    size = len(prefix) + len(suffix)
    for row in rows:
        row_size = 1 + len(row[1])
        if current and size + row_size > chunk_size:
            groups.append(current)
            current = []
            size = len(prefix) + len(suffix)
        current.append(row)
        size += row_size
    if current or not groups:
        groups.append(current)

    chunks = []
    for chunk_index, group in enumerate(groups, start=1):
        metadata = {
            "element_type": "table",
            "table_id": table.block_id,
            "block_order": table.order,
            "table_chunk_index": chunk_index,
            "table_chunk_count": len(groups),
        }
        if group:
            metadata.update(row_start=group[0][0], row_end=group[-1][0])
        if table.page_number is not None:
            metadata["page_number"] = table.page_number
        chunks.append(Document(
            page_content=prefix + "".join("\n" + text for _, text in group) + suffix,
            metadata=metadata,
        ))
    return chunks


def chunk_structured_document(
    document: ParsedDocument,
    *,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[Document]:
    """缓冲同页连续 TextBlock；表格、页变化及文档结束时 flush。

    Text metadata 的 block_start/end 是该连续 segment 的原始 order 范围（含端点），
    拆出的文本 chunks 共用此范围。全局 position 与 source provenance 由调用方统一追加。
    """
    chunks: list[Document] = []
    pending: list[TextBlock] = []

    def flush_text() -> None:
        if not pending:
            return
        metadata = {
            "element_type": "text",
            "block_start": pending[0].order,
            "block_end": pending[-1].order,
        }
        if pending[0].page_number is not None:
            metadata["page_number"] = pending[0].page_number
        chunks.extend(splitter.create_documents(
            ["\n\n".join(block.text for block in pending)], metadatas=[metadata],
        ))
        pending.clear()

    for block in document.blocks:
        if isinstance(block, TextBlock):
            if pending and pending[-1].page_number != block.page_number:
                flush_text()
            pending.append(block)
        else:
            flush_text()
            chunks.extend(_chunk_table_block(block, chunk_size))
    flush_text()
    return chunks

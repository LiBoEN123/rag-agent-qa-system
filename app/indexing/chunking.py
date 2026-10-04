"""把已校验的 ParsedDocument 按 reading order 转为检索 chunks，不访问源文件。"""

from collections import Counter
from collections.abc import Iterator
from unicodedata import normalize

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.document_blocks import ParsedDocument, TableBlock, TableCell, TextBlock


def _unique_row_cells(row: list[TableCell | None]) -> list[TableCell]:
    return list({
        (cell.row_index, cell.column_index): cell
        for cell in row if cell is not None and cell.text.strip()
    }.values())


def _is_full_width_row(cells: list[TableCell], column_count: int) -> bool:
    return (len(cells) == 1 and cells[0].column_index == 0
            and cells[0].column_span == column_count)


def _row_signature(
    row: list[TableCell | None], row_index: int,
) -> tuple[tuple[int, int, str], ...] | None:
    """按列序记录 unique anchors（含空值），排除 rowspan 及继承的 grid 投影。"""
    cells = {
        (cell.row_index, cell.column_index): cell
        for cell in row if cell is not None
    }.values()
    if any(cell.row_index != row_index or cell.row_span != 1 for cell in cells):
        return None
    signature = tuple(
        (cell.column_index, cell.column_span, " ".join(normalize("NFKC", cell.text).split()))
        for cell in cells
    )
    return signature if sum(bool(text) for _, _, text in signature) >= 2 else None


def _build_header_labels(grid: list[list[TableCell | None]], start: int, end: int) -> list[str]:
    labels = []
    for column in range(len(grid[start])):
        # 锚点去重防止 rowspan 重复同一标题，colspan 则贡献给覆盖的每一列。
        headers = {
            (cell.row_index, cell.column_index): cell.text
            for row in grid[start:end]
            if (cell := row[column]) is not None and cell.text.strip()
        }
        labels.append(" > ".join(headers.values()) or f"列{column + 1}")

    return labels


def _render_data_row(cells: list[TableCell], labels: list[str]) -> str:
    fields = []
    for cell in cells:
        label = " / ".join(dict.fromkeys(
            labels[cell.column_index:cell.column_index + cell.column_span]
        ))
        fields.append(f"{label}：{cell.text}")
    return "；".join(fields) + "。"


def _table_view(table: TableBlock) -> Iterator[
    tuple[str, dict[str, str | int], list[tuple[int | None, str]]]
]:
    """扫描结构状态；section/header 变化时交出旧行组，保留原 spans 投影。

    Docling header 标记优先；缺失时仅接纳有 section/data 证据的 exact 重复签名。
    混合行的 row header 仍作为值；单列表没有横向合并信号，仍走普通数据路径。
    """
    grid: list[list[TableCell | None]] = [
        [None] * table.column_count for _ in range(table.row_count)
    ]
    for cell in table.cells:
        for row in range(cell.row_index, cell.row_index + cell.row_span):
            for column in range(cell.column_index, cell.column_index + cell.column_span):
                grid[row][column] = cell
    unique = [_unique_row_cells(row) for row in grid]
    headers = [bool(cells) and all(cell.is_header for cell in cells) for cells in unique]
    contextual = [
        table.column_count > 1 and not headers[row] and _is_full_width_row(cells, table.column_count)
        for row, cells in enumerate(unique)
    ]
    signatures = [_row_signature(cells, row) for row, cells in enumerate(grid)]
    repeated = {signature for signature, count in Counter(signatures).items()
                if signature is not None and count >= 2}
    # 至少一个 occurrence 必须紧接独立全宽行，并紧邻正常 data。
    # 重复候选本身不能充当后续 data 证据，避免纯重复行带互相证明。
    reliable = {
        signature for row, signature in enumerate(signatures)
        if signature in repeated and 0 < row < table.row_count - 1
        and contextual[row - 1] and unique[row - 1][0].row_index == row - 1
        and not headers[row + 1] and signatures[row + 1] is not None
        and signatures[row + 1] not in repeated
    }
    headers = [header or signature in reliable for header, signature in zip(headers, signatures)]
    # 从后向前记录连续 header band 终点，lookahead 不反复扫描长表头。
    header_ends = list(range(table.row_count + 1))
    for row in reversed(range(table.row_count)):
        if headers[row]:
            header_ends[row] = header_ends[row + 1]
    sections = {
        row for row in range(table.row_count - 1)
        if contextual[row] and headers[row + 1]
        and unique[row][0].row_index == row
        and (end := header_ends[row + 1]) < table.row_count
        and unique[end] and not contextual[end]
    }
    # 只有紧连首个可靠 section 的前导全宽序列才传播为 global context。
    start = 0
    while start < table.row_count and contextual[start] and start not in sections:
        start += 1
    if start not in sections:
        start = 0
    global_context = [unique[row][0].text for row in range(start) if unique[row][0].row_index == row]
    section_title = None
    section_index = 0
    labels = [f"列{column + 1}" for column in range(table.column_count)]
    rows: list[tuple[int | None, str]] = []

    def view():
        prefix = "\n".join([
            "[表格]",
            *([f"标题：{table.caption}"] if table.caption else []),
            *[f"上下文：{text}" for text in global_context],
            *([f"分区：{section_title}"] if section_title else []),
            *(["表头：" + "；".join(labels)] if labels else []),
        ])
        metadata = {"section_index": section_index, "section_title": section_title} if section_title else {}
        return prefix, metadata, rows

    row = start
    while row < table.row_count:
        if row in sections:
            if rows:
                yield view()
                rows = []
            section_title = unique[row][0].text
            section_index += 1
        elif headers[row]:
            if rows:
                yield view()
                rows = []
            labels = _build_header_labels(grid, row, header_ends[row])
            row = header_ends[row]
            continue
        elif contextual[row]:
            # 不可靠全宽行只出现一次，不改变 section，也不计入 data row 范围。
            if unique[row][0].row_index == row:
                rows.append((None, f"说明：{unique[row][0].text}"))
        elif unique[row]:
            rows.append((row, _render_data_row(unique[row], labels)))
        row += 1
    yield view()


def _chunk_table_block(table: TableBlock, chunk_size: int) -> list[Document]:
    """按完整连续 body rows 分组，字符预算为 soft target，不做 row overlap。

    row_start/row_end 沿用原 TableCell.row_index：0-based、inclusive，仅取 data rows 的端点。
    全空行不生成证据；只有标题/表头、没有数据行的表仍保留，省略 row 范围。
    """
    if not (table.caption or "").strip() and not any(cell.text.strip() for cell in table.cells):
        return []
    suffix = "\n[/表格]"
    chunks = []
    for prefix, section_metadata, rows in _table_view(table):
        groups: list[list[tuple[int | None, str]]] = []
        current: list[tuple[int | None, str]] = []
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

        for group in groups:
            metadata = {
                "element_type": "table", "table_id": table.block_id,
                "block_order": table.order, **section_metadata,
            }
            data_rows = [row for row, _ in group if row is not None]
            if data_rows:
                metadata.update(row_start=data_rows[0], row_end=data_rows[-1])
            if table.page_number is not None:
                metadata["page_number"] = table.page_number
            chunks.append(Document(
                page_content=prefix + "".join("\n" + text for _, text in group) + suffix,
                metadata=metadata,
            ))
    for chunk_index, chunk in enumerate(chunks, start=1):
        chunk.metadata.update(table_chunk_index=chunk_index, table_chunk_count=len(chunks))
    return chunks


def chunk_structured_document(
    document: ParsedDocument,
    *,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
    document_title: str | None = None,
) -> list[Document]:
    """按标题、表格和页边界切分；Docling level 原样维护，title 位于 level 0。

    Text metadata 的 block_start/end 是该连续 segment 的原始 order 范围（含端点），
    拆出的文本 chunks 共用此范围。标题作为上下文在切分后加入每个 text/table chunk。
    全局 position 与 source provenance 由调用方统一追加。
    """
    chunks: list[Document] = []
    pending: list[TextBlock] = []
    headings: list[tuple[int, str]] = []
    pending_heading: TextBlock | None = None

    def append_with_context(section_chunks: list[Document]) -> None:
        titles: list[str] = []
        for title in [document_title or "", *(text for _, text in headings)]:
            title = title.strip()
            if title and (not titles or titles[-1] != title):
                titles.append(title)
        title_path = " > ".join(titles)
        for chunk in section_chunks:
            if title_path:
                chunk.metadata["title_path"] = title_path
                chunk.page_content = f"[标题路径] {title_path}" + (
                    f"\n\n{chunk.page_content}" if chunk.page_content else ""
                )
        chunks.extend(section_chunks)

    def flush_text() -> None:
        nonlocal pending_heading
        if pending and not any(block.text.strip() for block in pending):
            pending.clear()
        blocks = pending or ([pending_heading] if pending_heading is not None else [])
        if not blocks:
            return
        metadata = {
            "element_type": "text",
            "block_start": blocks[0].order,
            "block_end": blocks[-1].order,
        }
        if blocks[0].page_number is not None:
            metadata["page_number"] = blocks[0].page_number
        section_chunks = splitter.create_documents(
            ["\n\n".join(block.text for block in pending)], metadatas=[metadata],
        ) if pending else [Document(page_content="", metadata=metadata)]
        append_with_context(section_chunks)
        pending.clear()
        pending_heading = None

    for block in document.blocks:
        if isinstance(block, TextBlock):
            if block.text_role != "text":
                level = block.heading_level if block.heading_level is not None else 0
                # 连续祖先标题由子章节路径承载；空的同级/末尾章节仍保留标题证据。
                if pending_heading is not None and level > headings[-1][0]:
                    pending_heading = None
                flush_text()
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, block.text))
                pending_heading = block if block.text.strip() else None
                continue
            if pending and pending[-1].page_number != block.page_number:
                flush_text()
            if block.text.strip():
                pending_heading = None
            pending.append(block)
        else:
            table_chunks = _chunk_table_block(block, chunk_size)
            if table_chunks:
                pending_heading = None
            flush_text()
            append_with_context(table_chunks)
    flush_text()
    return chunks

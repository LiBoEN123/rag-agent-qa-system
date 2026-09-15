"""结构优先的语义表格投影与完整行分组，不调用 parser 或模型。"""

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion import IngestionRequester, IngestionService, build_knowledge_document
from app.ingestion.document_blocks import ParsedDocument, TableBlock, TableCell, TextBlock
from app.rag import indexing
from app.rag.structured_chunking import chunk_structured_document
from tests.support.structured_document import registration_structure


def _candidate(structure, *, file_type="pdf", source_type="local_file"):
    source_uri = {
        "local_file": "/fixture/table", "upload": "upload://local/table",
        "firecrawl": "https://example.com/table", "github": "github://owner/repo/table",
    }[source_type]
    document = build_knowledge_document(
        source_type=source_type, source_uri=source_uri, title="table",
        content="ignored in favor of structure", structured_content=structure,
        metadata={"file_type": file_type, "file_name": "table", "source_path": "fixture/table"},
    )
    requester = IngestionRequester(principal_id="chunk-test", principal_type="system", source="pytest")
    service = IngestionService()
    candidate = service.propose(document=document, requester=requester)
    if candidate.status.value == "pending":
        candidate = service.approve(candidate.candidate_id, decided_by=requester)
    return candidate


def _chunks(table, *, chunk_size=600):
    return indexing.candidate_to_langchain_documents(
        _candidate(ParsedDocument(blocks=(table,))), chunk_size=chunk_size, chunk_overlap=0,
    )


@pytest.mark.parametrize("file_type", ["pdf", "docx", "html", "md", "txt"])
@pytest.mark.parametrize("source_type", ["local_file", "upload", "firecrawl", "github"])
def test_structure_precedes_file_type_and_has_exact_semantics_and_scalar_provenance(
    file_type, source_type, monkeypatch,
):
    candidate = _candidate(registration_structure(), file_type=file_type, source_type=source_type)

    def forbidden(*args, **kwargs):
        pytest.fail("a TableBlock must never enter the text splitter or parser")

    monkeypatch.setattr(RecursiveCharacterTextSplitter, "create_documents", forbidden)
    monkeypatch.setattr(indexing, "parse_document", forbidden)
    [chunk] = indexing.candidate_to_langchain_documents(candidate, chunk_size=600, chunk_overlap=0)
    assert chunk.page_content == (
        "[表格]\n标题：企业登记办理要求\n表头：事项；条件；材料\n"
        "事项：企业注册；条件：年满18岁；材料：身份证、申请表。\n"
        "事项：企业变更；条件：企业已注册；材料：营业执照、变更表。\n[/表格]"
    )
    assert chunk.metadata == {
        "source_id": candidate.document.source_id,
        "document_uid": candidate.document.document_uid,
        "source_type": source_type, "source_uri": candidate.document.source_uri,
        "file_name": "table", "file_type": file_type, "source_path": "fixture/table",
        "position": 1, "element_type": "table", "table_id": "b0", "block_order": 0,
        "table_chunk_index": 1, "table_chunk_count": 1, "row_start": 1, "row_end": 2,
        "page_number": 2,
    }
    assert all(type(value) in (str, int, float, bool) for value in chunk.metadata.values())


def test_ordered_text_buffer_uses_existing_splitter_overlap_and_page_boundaries(monkeypatch):
    table = registration_structure().blocks[0]
    structure = ParsedDocument(blocks=(
        TextBlock(block_id="b0", order=0, page_number=1, text="前置短段。"),
        TextBlock(block_id="b1", order=1, page_number=1, text="连续短段。"),
        table.model_copy(update={"block_id": "b2", "order": 2}),
        TextBlock(block_id="b3", order=3, page_number=2, text="正文长段。" * 30),
        TextBlock(block_id="b4", order=4, page_number=2, text="同页尾段。"),
        TextBlock(block_id="b5", order=5, page_number=3, text="下一页。"),
        table.model_copy(update={"block_id": "b6", "order": 6, "page_number": 3}),
    ))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=70, chunk_overlap=10, separators=indexing._TEXT_SEPARATORS,
    )
    expected_long_text = splitter.split_text("正文长段。" * 30 + "\n\n同页尾段。")
    calls = []
    create = RecursiveCharacterTextSplitter.create_documents

    def capture(self, texts, **kwargs):
        calls.extend(texts)
        return create(self, texts, **kwargs)

    monkeypatch.setattr(RecursiveCharacterTextSplitter, "create_documents", capture)
    chunks = indexing.candidate_to_langchain_documents(_candidate(structure), chunk_size=70, chunk_overlap=10)
    assert calls == ["前置短段。\n\n连续短段。", "正文长段。" * 30 + "\n\n同页尾段。", "下一页。"]
    assert chunks[0].page_content == calls[0]
    assert chunks[0].metadata["block_start"] == 0
    assert chunks[0].metadata["block_end"] == 1
    assert chunks[0].metadata["page_number"] == 1
    text_chunks = [chunk for chunk in chunks if chunk.metadata.get("block_start") == 3]
    assert [chunk.page_content for chunk in text_chunks] == expected_long_text
    assert all(chunk.metadata["block_end"] == 4 and chunk.metadata["page_number"] == 2 for chunk in text_chunks)
    assert [chunk.metadata["position"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert [chunk.metadata.get("block_order", chunk.metadata.get("block_start")) for chunk in chunks] == (
        [0, 2, 2] + [3] * len(text_chunks) + [5, 6, 6]
    )


def test_large_table_groups_contiguous_rows_without_loss_or_body_overlap():
    table = registration_structure().blocks[0]
    body = tuple(
        TableCell(text=text, row_index=row, column_index=column)
        for row in range(1, 21)
        for column, text in enumerate((f"事项{row:02}", "年满18岁", f"材料{row:02}"))
    )
    table = table.model_copy(update={"row_count": 21, "cells": table.cells[:3] + body})
    chunks = _chunks(table, chunk_size=145)
    assert 1 < len(chunks) < 20
    covered = []
    for index, chunk in enumerate(chunks, start=1):
        assert len(chunk.page_content) <= 145
        assert chunk.page_content.startswith("[表格]\n标题：企业登记办理要求\n表头：事项；条件；材料\n")
        assert chunk.page_content.endswith("\n[/表格]")
        assert chunk.metadata["table_chunk_index"] == index
        assert chunk.metadata["table_chunk_count"] == len(chunks)
        rows = range(chunk.metadata["row_start"], chunk.metadata["row_end"] + 1)
        covered.extend(rows)
        assert chunk.page_content.splitlines()[3:-1] == [
            f"事项：事项{row:02}；条件：年满18岁；材料：材料{row:02}。" for row in rows
        ]
    assert covered == list(range(1, 21))  # 原行索引含端点，header row 0 不属于 body。
    assert _chunks(table, chunk_size=145) == chunks


def test_multilevel_headers_use_colspan_and_do_not_duplicate_rowspan_header():
    table = TableBlock(block_id="b0", order=0, row_count=3, column_count=3, cells=(
        TableCell(text="事项", row_index=0, column_index=0, row_span=2, is_header=True),
        TableCell(text="申请要求", row_index=0, column_index=1, column_span=2, is_header=True),
        TableCell(text="条件", row_index=1, column_index=1, is_header=True),
        TableCell(text="材料", row_index=1, column_index=2, is_header=True),
        TableCell(text="企业注册", row_index=2, column_index=0),
        TableCell(text="年满18岁", row_index=2, column_index=1),
        TableCell(text="身份证", row_index=2, column_index=2),
    ))
    [chunk] = _chunks(table)
    assert "事项：企业注册；申请要求 > 条件：年满18岁；申请要求 > 材料：身份证。" in chunk.page_content
    assert "事项 > 事项" not in chunk.page_content
    assert (chunk.metadata["row_start"], chunk.metadata["row_end"]) == (2, 2)
    shuffled = table.model_copy(update={"cells": tuple(reversed(table.cells))})
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)
    assert chunk_structured_document(ParsedDocument(blocks=(table,)), splitter=splitter, chunk_size=600) == (
        chunk_structured_document(ParsedDocument(blocks=(shuffled,)), splitter=splitter, chunk_size=600)
    )


@pytest.mark.parametrize("row_header", [False, True])
def test_body_rowspan_retains_owner_across_chunk_boundary_without_mutation(row_header):
    table = TableBlock(block_id="b0", order=0, row_count=3, column_count=2, cells=(
        TableCell(text="事项", row_index=0, column_index=0, is_header=True),
        TableCell(text="材料", row_index=0, column_index=1, is_header=True),
        TableCell(text="企业注册", row_index=1, column_index=0, row_span=2, is_header=row_header),
        TableCell(text="身份证", row_index=1, column_index=1),
        TableCell(text="申请表", row_index=2, column_index=1),
    ))
    before = table.model_dump_json()
    chunks = _chunks(table, chunk_size=1)
    assert len(chunks) == 2
    for index, material in enumerate(("身份证", "申请表")):
        assert f"事项：企业注册；材料：{material}。" in chunks[index].page_content
        assert chunks[index].metadata["row_start"] == chunks[index].metadata["row_end"] == index + 1
    assert table.model_dump_json() == before


def test_oversized_row_is_alone_and_whole_with_other_rows_preserved():
    table = registration_structure().blocks[0]
    huge_value = "完整超长材料\n" * 100
    cells = tuple(cell.model_copy(update={"text": huge_value}) if (
        cell.row_index == 1 and cell.column_index == 2
    ) else cell for cell in table.cells)
    chunks = _chunks(table.model_copy(update={"cells": cells}), chunk_size=100)
    assert len(chunks) == 2
    # Domain 规范化行尾空白；chunker 保留其后的完整证据。
    assert f"材料：{huge_value.rstrip()}。" in chunks[0].page_content
    assert len(chunks[0].page_content) > 100
    assert chunks[0].metadata["row_start"] == chunks[0].metadata["row_end"] == 1
    assert chunks[1].metadata["row_start"] == chunks[1].metadata["row_end"] == 2
    assert "材料：营业执照、变更表。" in chunks[1].page_content


def test_headerless_sparse_and_body_colspan_use_positional_labels_without_guessing():
    table = TableBlock(block_id="b0", order=0, row_count=3, column_count=3, cells=(
        TableCell(text="企业注册", row_index=0, column_index=0),
        TableCell(text="身份证", row_index=0, column_index=1, column_span=2),
        TableCell(text="申请表", row_index=2, column_index=2),
    ))
    [chunk] = _chunks(table)
    assert "列1：企业注册；列2 / 列3：身份证。" in chunk.page_content
    assert "列3：申请表。" in chunk.page_content
    assert (chunk.metadata["row_start"], chunk.metadata["row_end"]) == (0, 2)
    assert "page_number" not in chunk.metadata


@pytest.mark.parametrize("kind", ["empty", "caption_only", "header_only"])
def test_empty_and_header_only_tables_do_not_invent_body_ranges(kind):
    table = TableBlock(block_id="b0", order=0, row_count=0, column_count=0, cells=())
    if kind == "caption_only":
        table = table.model_copy(update={"caption": "空表说明"})
    if kind == "header_only":
        table = table.model_copy(update={"row_count": 1, "column_count": 1, "cells": (
            TableCell(text="材料", row_index=0, column_index=0, is_header=True),
        )})
    chunks = chunk_structured_document(
        ParsedDocument(blocks=(table,)),
        splitter=RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0), chunk_size=600,
    )
    if kind == "empty":
        assert chunks == []
    else:
        [chunk] = chunks
        assert ("空表说明" if kind == "caption_only" else "表头：材料") in chunk.page_content
        assert "row_start" not in chunk.metadata and "row_end" not in chunk.metadata

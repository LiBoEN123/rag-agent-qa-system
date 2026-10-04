"""结构优先的语义表格投影与完整行分组，不调用 parser 或模型。"""

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion import IngestionRequester, IngestionService, build_knowledge_document
from app.ingestion.document_blocks import ParsedDocument, TableBlock, TableCell, TextBlock
from app.indexing import langchain as indexing
from app.indexing.chunking import chunk_structured_document
from tests.support.structured_document import (
    ADMISSION_ROWS, admission_structure, imut_admission_structure, registration_structure,
)


def _candidate(structure, *, file_type="pdf", source_type="local_file", title="table"):
    source_uri = {
        "local_file": "/fixture/table", "upload": "upload://local/table",
        "firecrawl": "https://example.com/table", "github": "github://owner/repo/table",
    }[source_type]
    document = build_knowledge_document(
        source_type=source_type, source_uri=source_uri, title=title,
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
    # 单独验证既有 table projection；完整 candidate 的标题与 provenance 在上层用例覆盖。
    candidate = _candidate(ParsedDocument(blocks=(table,)))
    return chunk_structured_document(
        candidate.document.structured_content, chunk_size=chunk_size,
        splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0),
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
        "[标题路径] table\n\n[表格]\n标题：企业登记办理要求\n表头：事项；条件；材料\n"
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
        "page_number": 2, "title_path": "table",
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
    assert chunks[0].page_content == "[标题路径] table\n\n" + calls[0]
    assert chunks[0].metadata["block_start"] == 0
    assert chunks[0].metadata["block_end"] == 1
    assert chunks[0].metadata["page_number"] == 1
    text_chunks = [chunk for chunk in chunks if chunk.metadata.get("block_start") == 3]
    assert [chunk.page_content for chunk in text_chunks] == ["[标题路径] table\n\n" + text for text in expected_long_text]
    assert all(chunk.metadata["block_end"] == 4 and chunk.metadata["page_number"] == 2 for chunk in text_chunks)
    assert [chunk.metadata["position"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert [chunk.metadata.get("block_order", chunk.metadata.get("block_start")) for chunk in chunks] == (
        [0, 2, 2] + [3] * len(text_chunks) + [5, 6, 6]
    )


def test_heading_boundaries_stack_reset_and_oversized_section_keep_context():
    entries = [
        ("A", "title", None), ("B", "section_header", 1), ("C", "section_header", 2),
        ("正文一。" * 30, "text", None), ("D", "section_header", 1), ("正文二。", "text", None),
        ("E", "title", None), ("正文三。", "text", None),
        ("F", "section_header", 7), ("正文四。", "text", None),
    ]
    structure = ParsedDocument(blocks=tuple(TextBlock(
        block_id=f"b{i}", order=i, text=text, text_role=role, heading_level=level,
    ) for i, (text, role, level) in enumerate(entries)))
    candidate = _candidate(structure, title="A", source_type="firecrawl", file_type="html")
    chunks = indexing.candidate_to_langchain_documents(candidate, chunk_size=40, chunk_overlap=10)
    first = [chunk for chunk in chunks if chunk.metadata["block_start"] == 3]
    expected = RecursiveCharacterTextSplitter(
        chunk_size=40, chunk_overlap=10, separators=indexing._TEXT_SEPARATORS,
    ).split_text(entries[3][0])
    assert len(first) > 1
    assert [chunk.page_content for chunk in first] == ["[标题路径] A > B > C\n\n" + text for text in expected]
    assert {chunk.metadata["title_path"] for chunk in first} == {"A > B > C"}
    assert [(chunk.metadata["title_path"], chunk.page_content) for chunk in chunks[len(first):]] == [
        ("A > D", "[标题路径] A > D\n\n正文二。"),
        ("A > E", "[标题路径] A > E\n\n正文三。"),
        ("A > E > F", "[标题路径] A > E > F\n\n正文四。"),
    ]
    assert [chunk.metadata["position"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(type(value) in (str, int, float, bool) for chunk in chunks for value in chunk.metadata.values())


def test_section_levels_pop_siblings_and_roots_without_document_title():
    entries = [("A", 1), ("B", 2), ("C", 3), ("正文1", None),
               ("D", 2), ("正文2", None), ("E", 1), ("正文3", None)]
    structure = ParsedDocument(blocks=tuple(TextBlock(
        block_id=f"b{i}", order=i, text=text,
        text_role="section_header" if level else "text", heading_level=level,
    ) for i, (text, level) in enumerate(entries)))
    chunks = chunk_structured_document(
        structure, splitter=RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0), chunk_size=600,
    )
    assert [chunk.metadata["title_path"] for chunk in chunks] == ["A > B > C", "A > D", "E"]
    assert [chunk.page_content.split("\n\n")[-1] for chunk in chunks] == ["正文1", "正文2", "正文3"]


def test_tables_inherit_heading_across_pages_without_changing_table_metadata():
    table = admission_structure().blocks[0].model_copy(update={"block_id": "b2", "order": 2})
    structure = ParsedDocument(page_count=2, blocks=(
        TextBlock(block_id="b0", order=0, text="管理办法", text_role="title", page_number=1),
        TextBlock(block_id="b1", order=1, text="申请材料", text_role="section_header", heading_level=1, page_number=1),
        table,
        TextBlock(block_id="b3", order=3, text="材料正文 A", page_number=1),
        TextBlock(block_id="b4", order=4, text="材料正文 B", page_number=2),
        TextBlock(block_id="b5", order=5, text="申请条件", text_role="section_header", heading_level=1, page_number=2),
        TextBlock(block_id="b6", order=6, text="条件正文 C", page_number=2),
    ))
    splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=0)
    baseline = chunk_structured_document(
        ParsedDocument(blocks=(table.model_copy(update={"block_id": "b0", "order": 0}),)),
        splitter=splitter, chunk_size=180,
    )
    chunks = indexing.candidate_to_langchain_documents(
        _candidate(structure, title="管理办法"), chunk_size=180, chunk_overlap=0,
    )
    tables = [chunk for chunk in chunks if chunk.metadata["element_type"] == "table"]
    assert len(tables) == len(baseline) > 1
    assert {chunk.metadata["section_index"] for chunk in tables} == {1, 2}
    for actual, expected in zip(tables, baseline):
        assert actual.metadata["title_path"] == "管理办法 > 申请材料"
        assert actual.page_content == "[标题路径] 管理办法 > 申请材料\n\n" + expected.page_content
        assert actual.metadata["table_id"] == "b2"
        assert actual.metadata["block_order"] == 2
        for key in ("table_chunk_index", "table_chunk_count", "row_start", "row_end", "section_index", "section_title"):
            assert actual.metadata[key] == expected.metadata[key]
    text_chunks = chunks[len(tables):]
    assert [chunk.metadata["title_path"] for chunk in text_chunks] == [
        "管理办法 > 申请材料", "管理办法 > 申请材料", "管理办法 > 申请条件",
    ]
    assert [chunk.metadata["page_number"] for chunk in text_chunks] == [1, 2, 2]
    assert [chunk.page_content.split("\n\n")[-1] for chunk in text_chunks] == ["材料正文 A", "材料正文 B", "条件正文 C"]


def test_heading_only_sections_retain_evidence_without_repeating_ancestors():
    structure = ParsedDocument(blocks=(
        TextBlock(block_id="b0", order=0, text="管理办法", text_role="title"),
        TextBlock(block_id="b1", order=1, text="待公布", text_role="section_header", heading_level=1),
        TextBlock(block_id="b2", order=2, text="补充说明", text_role="section_header", heading_level=1),
        TextBlock(block_id="b3", order=3, text="\n  "),
    ))
    chunks = indexing.candidate_to_langchain_documents(_candidate(structure, title="管理办法"))
    assert [chunk.page_content for chunk in chunks] == [
        "[标题路径] 管理办法 > 待公布", "[标题路径] 管理办法 > 补充说明",
    ]


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


@pytest.mark.parametrize("chunk_size", [160, 10000])
def test_admission_sections_propagate_context_isolate_rows_and_preserve_duplicates(chunk_size):
    table = admission_structure().blocks[0]
    original = table.model_dump_json()
    chunks = _chunks(table, chunk_size=chunk_size)
    covered = []
    for index, chunk in enumerate(chunks, 1):
        section = chunk.metadata["section_index"]
        title = ADMISSION_ROWS[1 if section == 1 else 5][0]
        assert chunk.page_content.startswith(
            f"[表格]\n上下文：{ADMISSION_ROWS[0][0]}\n分区：{title}\n"
            "表头：专业名称；最高分；最低分；平均分；备注\n"
        )
        assert chunk.metadata["section_title"] == title
        assert ADMISSION_ROWS[5 if section == 1 else 1][0] not in chunk.page_content
        rows = list(range(chunk.metadata["row_start"], chunk.metadata["row_end"] + 1))
        assert set(rows) <= (set(range(3, 5)) if section == 1 else set(range(7, 13)))
        covered.extend(rows)
        expected = ["；".join(f"{label}：{value}" for label, value in zip(ADMISSION_ROWS[2], ADMISSION_ROWS[row])) + "。"
                    for row in rows]
        assert chunk.page_content.splitlines()[4:-1] == expected
        assert chunk.metadata["table_chunk_index"] == index
        assert chunk.metadata["table_chunk_count"] == len(chunks)
        assert all(type(value) in (str, int, float, bool) for value in chunk.metadata.values())
    assert covered == [3, 4, 7, 8, 9, 10, 11, 12]
    text = "\n".join(chunk.page_content for chunk in chunks)
    for label in ADMISSION_ROWS[2]:
        assert f"{label}：{label}" not in text
    assert text.count("专业名称：材料成型及控制工程；") == 2
    assert text.count("专业名称：储能科学与工程；") == 3
    assert "最高分：549；最低分：509；平均分：517.98；备注：物理类。" in text
    assert "最高分：503；最低分：480；平均分：487.75；备注：民族班。" in text
    physics = [chunk for chunk in chunks if chunk.metadata["section_index"] == 2]
    if chunk_size == 160:
        assert len(physics) > 1
    else:
        assert len(chunks) == 2
    assert table.model_dump_json() == original


def test_section_detection_is_structural_and_cell_order_independent():
    table = admission_structure().blocks[0]
    cells = tuple(cell.model_copy(update={"text": {0: "Overview", 1: "Group A", 5: "Group B"}.get(
        cell.row_index, cell.text,
    )}) for cell in reversed(table.cells))
    chunks = _chunks(table.model_copy(update={"cells": cells}), chunk_size=10000)
    assert [chunk.metadata["section_title"] for chunk in chunks] == ["Group A", "Group B"]
    assert all("上下文：Overview" in chunk.page_content for chunk in chunks)


@pytest.mark.parametrize("chunk_size", [1, 10000])
def test_ambiguous_full_width_note_is_kept_once_without_changing_section(chunk_size):
    table = admission_structure().blocks[0]
    # 数据中间的全宽说明后面直接跟 data，不能成为 section 或 global context。
    cells = tuple(cell for cell in table.cells if cell.row_index != 8) + (
        TableCell(text="unclassified note", row_index=8, column_index=0, column_span=5),
    )
    chunks = _chunks(table.model_copy(update={"cells": cells}), chunk_size=chunk_size)
    assert sum(chunk.page_content.count("说明：unclassified note") for chunk in chunks) == 1
    physics = [chunk for chunk in chunks if chunk.metadata["section_index"] == 2]
    assert all(chunk.metadata["section_title"] == ADMISSION_ROWS[5][0] for chunk in physics)
    if chunk_size == 1:
        [note] = [chunk for chunk in chunks if "unclassified note" in chunk.page_content]
        assert "row_start" not in note.metadata and "row_end" not in note.metadata
        assert all(chunk.metadata.get("row_start") != 8 for chunk in chunks)


def test_full_width_without_header_followed_by_data_stays_note():
    table = TableBlock(block_id="b0", order=0, row_count=3, column_count=2, cells=(
        TableCell(text="uncertain", row_index=0, column_index=0, column_span=2),
        TableCell(text="value", row_index=1, column_index=0),
        TableCell(text="trailing note", row_index=2, column_index=0, column_span=2),
    ))
    [chunk] = _chunks(table)
    assert "说明：uncertain\n列1：value。\n说明：trailing note" in chunk.page_content
    assert (chunk.metadata["row_start"], chunk.metadata["row_end"]) == (1, 1)
    assert "section_title" not in chunk.metadata and "section_index" not in chunk.metadata
    assert "上下文：" not in chunk.page_content and "分区：" not in chunk.page_content


def test_full_width_and_header_without_data_do_not_invent_section():
    table = TableBlock(block_id="b0", order=0, row_count=2, column_count=2, cells=(
        TableCell(text="uncertain", row_index=0, column_index=0, column_span=2),
        TableCell(text="one", row_index=1, column_index=0, is_header=True),
        TableCell(text="two", row_index=1, column_index=1, is_header=True),
    ))
    chunks = _chunks(table)
    assert any("说明：uncertain" in chunk.page_content for chunk in chunks)
    assert all("section_title" not in chunk.metadata and "row_start" not in chunk.metadata for chunk in chunks)


def test_full_width_rowspan_note_does_not_turn_inherited_cell_into_section():
    table = TableBlock(block_id="b0", order=0, row_count=4, column_count=2, cells=(
        TableCell(text="uncertain", row_index=0, column_index=0, column_span=2, row_span=2),
        TableCell(text="one", row_index=2, column_index=0, is_header=True),
        TableCell(text="two", row_index=2, column_index=1, is_header=True),
        TableCell(text="value", row_index=3, column_index=0),
    ))
    chunks = _chunks(table)
    assert sum(chunk.page_content.count("uncertain") for chunk in chunks) == 1
    assert all("section_title" not in chunk.metadata for chunk in chunks)
    assert [(chunk.metadata.get("row_start"), chunk.metadata.get("row_end")) for chunk in chunks] == [
        (None, None), (3, 3),
    ]


def test_repeated_multilevel_header_refreshes_labels_and_keeps_body_spans():
    cells = []
    for start, title, parent in ((0, "Group A", "要求"), (5, "Group B", "新要求")):
        cells.extend((
            TableCell(text=title, row_index=start, column_index=0, column_span=3),
            TableCell(text="事项", row_index=start + 1, column_index=0, row_span=2, is_header=True),
            TableCell(text=parent, row_index=start + 1, column_index=1, column_span=2, is_header=True),
            TableCell(text="条件", row_index=start + 2, column_index=1, is_header=True),
            TableCell(text="材料", row_index=start + 2, column_index=2, is_header=True),
            TableCell(text="办理", row_index=start + 3, column_index=0, row_span=2),
            TableCell(text="完整值", row_index=start + 3, column_index=1, column_span=2),
            TableCell(text="次行", row_index=start + 4, column_index=1),
        ))
    chunks = _chunks(TableBlock(block_id="b0", order=0, row_count=10, column_count=3, cells=tuple(cells)), chunk_size=1)
    assert [chunk.metadata["row_start"] for chunk in chunks] == [3, 4, 8, 9]
    for first, second, parent in ((chunks[0], chunks[1], "要求"), (chunks[2], chunks[3], "新要求")):
        assert f"事项：办理；{parent} > 条件 / {parent} > 材料：完整值。" in first.page_content
        assert f"事项：办理；{parent} > 条件：次行。" in second.page_content
        assert first.metadata["section_title"] == second.metadata["section_title"]


def test_repeated_header_without_new_section_flushes_and_keeps_current_section():
    table = admission_structure().blocks[0]
    # 去掉第二个分区标签，但保留 repeated header，仍属于第一个 section。
    table = table.model_copy(update={"cells": tuple(cell for cell in table.cells if cell.row_index != 5)})
    chunks = _chunks(table, chunk_size=10000)
    assert len(chunks) == 2
    assert all(chunk.metadata["section_title"] == ADMISSION_ROWS[1][0] for chunk in chunks)
    assert [(chunk.metadata["row_start"], chunk.metadata["row_end"]) for chunk in chunks] == [(3, 4), (7, 12)]


def test_full_width_header_and_single_column_table_preserve_existing_projection():
    table = TableBlock(block_id="b0", order=0, row_count=3, column_count=2, cells=(
        TableCell(text="要求", row_index=0, column_index=0, column_span=2, is_header=True),
        TableCell(text="条件", row_index=1, column_index=0, is_header=True),
        TableCell(text="材料", row_index=1, column_index=1, is_header=True),
        TableCell(text="成人", row_index=2, column_index=0),
        TableCell(text="证件", row_index=2, column_index=1),
    ))
    [chunk] = _chunks(table)
    assert "要求 > 条件：成人；要求 > 材料：证件。" in chunk.page_content
    assert "section_title" not in chunk.metadata
    [single] = _chunks(TableBlock(block_id="b0", order=0, row_count=2, column_count=1, cells=(
        TableCell(text="材料", row_index=0, column_index=0, is_header=True),
        TableCell(text="证件", row_index=1, column_index=0),
    )))
    assert single.page_content == "[表格]\n表头：材料\n材料：证件。\n[/表格]"
    assert (single.metadata["row_start"], single.metadata["row_end"]) == (1, 1)


@pytest.mark.parametrize("chunk_size", [160, 10000])
def test_real_header_signature_fallback_reaches_candidate_chunks_without_mutating_structure(chunk_size):
    candidate = _candidate(imut_admission_structure(), source_type="firecrawl", file_type="html")
    original = candidate.model_dump_json()
    assert all(not cell.is_header for cell in candidate.document.structured_content.blocks[0].cells)
    chunks = indexing.candidate_to_langchain_documents(candidate, chunk_size=chunk_size, chunk_overlap=0)
    assert {chunk.metadata["section_title"] for chunk in chunks} == {
        "内蒙古本科提前B段艺术类", "内蒙古本科批物理类",
    }
    for chunk in chunks:
        assert "上下文：内蒙古工业大学2026年分省录取分数统计" in chunk.page_content
        assert "表头：专业名称；最高分；最低分；平均分；备注" in chunk.page_content
        assert "专业名称：专业名称" not in chunk.page_content
        assert chunk.metadata["source_type"] == "firecrawl"
    art = next(chunk for chunk in chunks if "专业名称：服装与服饰设计" in chunk.page_content)
    assert art.metadata["section_title"] == "内蒙古本科提前B段艺术类"
    assert art.metadata["section_index"] == 1
    physics = next(chunk for chunk in chunks if "专业名称：安全工程" in chunk.page_content)
    assert physics.metadata["section_title"] == "内蒙古本科批物理类"
    assert physics.metadata["section_index"] == 2
    assert "分区：内蒙古本科批物理类" in physics.page_content
    assert "专业名称：安全工程；最高分：527；最低分：501；平均分：508.32；备注：物理类。" in physics.page_content
    text = "\n".join(chunk.page_content for chunk in chunks)
    assert text.count("专业名称：材料成型及控制工程；") == 2
    assert "最高分：549；最低分：509；平均分：517.98；备注：物理类。" in text
    assert "最高分：503；最低分：480；平均分：487.75；备注：民族班。" in text
    assert [row for chunk in chunks for row in range(chunk.metadata["row_start"], chunk.metadata["row_end"] + 1)] == [
        3, 4, 5, 8, 9, 10,
    ]
    assert candidate.model_dump_json() == original


def _neutral_repeated_table():
    cells = []
    for start, section, value in ((0, "Group A", "alpha"), (3, "Group B", "beta")):
        cells.extend((
            TableCell(text=section, row_index=start, column_index=0, column_span=3),
            TableCell(text="Item", row_index=start + 1, column_index=0, column_span=2),
            TableCell(text="Value", row_index=start + 1, column_index=2),
            TableCell(text=value, row_index=start + 2, column_index=0, column_span=2),
            TableCell(text="10", row_index=start + 2, column_index=2),
        ))
    return TableBlock(block_id="b0", order=0, row_count=6, column_count=3, cells=tuple(cells))


def test_fallback_is_generic_normalized_exact_and_uses_colspan_anchors_in_column_order():
    table = _neutral_repeated_table()
    cells = tuple(cell.model_copy(update={"text": "  Ｉｔｅｍ\n name  " if cell.row_index == 1 else "Item name"})
                  if cell.column_index == 0 and cell.row_index in (1, 4) else cell
                  for cell in reversed(table.cells))
    structure = ParsedDocument(blocks=(table.model_copy(update={"cells": cells}),))
    original = structure.model_dump_json()
    chunks = chunk_structured_document(
        structure, splitter=RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0), chunk_size=600,
    )
    assert [chunk.metadata["section_title"] for chunk in chunks] == ["Group A", "Group B"]
    assert [chunk.metadata["row_start"] for chunk in chunks] == [2, 5]
    assert "Item name：beta；Value：10。" in chunks[1].page_content
    assert structure.model_dump_json() == original


@pytest.mark.parametrize("missing_evidence", [
    "context", "second_occurrence", "data", "two_nonempty_cells", "same_columns", "same_text", "multi_column",
])
def test_repeated_rows_without_structural_evidence_remain_data(missing_evidence):
    table = _neutral_repeated_table()
    cells = list(table.cells)
    if missing_evidence == "context":
        cells = [cell for cell in cells if cell.row_index not in (0, 3)]
    elif missing_evidence == "second_occurrence":
        cells = [cell for cell in cells if cell.row_index != 4]
    elif missing_evidence == "data":
        cells = [cell for cell in cells if cell.row_index not in (2, 5)]
    elif missing_evidence == "two_nonempty_cells":
        cells = [cell.model_copy(update={"text": ""}) if cell.row_index in (1, 4) and cell.column_index == 2
                 else cell for cell in cells]
    elif missing_evidence == "same_columns":
        cells = [cell.model_copy(update={"column_span": 1}) if cell.row_index == 4 and cell.column_index == 0
                 else cell.model_copy(update={"column_index": 1, "column_span": 2}) if cell.row_index == 4
                 else cell for cell in cells]
    elif missing_evidence == "same_text":
        cells = [cell.model_copy(update={"text": "Items"}) if cell.row_index == 4 and cell.column_index == 0
                 else cell for cell in cells]
    else:
        cells = [cell for cell in cells if cell.row_index not in (1, 4)] + [
            TableCell(text="Item", row_index=row, column_index=0, column_span=3) for row in (1, 4)
        ]
    chunks = _chunks(table.model_copy(update={"cells": tuple(cells)}), chunk_size=1)
    assert all("section_title" not in chunk.metadata for chunk in chunks)
    text = "\n".join(chunk.page_content for chunk in chunks)
    assert "表头：列1；列2；列3" in text
    assert "Item" in text
    if missing_evidence == "context":
        # 两行完全相同的普通数据必须保留两次。
        assert text.count("列1 / 列2：Item；列3：Value。") == 2


@pytest.mark.parametrize("inherited_context", [False, True])
def test_fallback_rejects_rowspan_inherited_headers_or_context(inherited_context):
    if inherited_context:
        table = _neutral_repeated_table()
        positions = {0: 0, 1: 2, 2: 3, 3: 4, 4: 6, 5: 7}
        cells = tuple(cell.model_copy(update={"row_index": positions[cell.row_index], "row_span": 2})
                      if cell.row_index in (0, 3) else
                      cell.model_copy(update={"row_index": positions[cell.row_index]}) for cell in table.cells)
        table = table.model_copy(update={"cells": cells, "row_count": 8})
    else:
        table = TableBlock(block_id="b0", order=0, row_count=4, column_count=2, cells=(
            TableCell(text="Group", row_index=0, column_index=0, column_span=2),
            TableCell(text="Item", row_index=1, column_index=0, row_span=2),
            TableCell(text="Value", row_index=1, column_index=1, row_span=2),
            TableCell(text="alpha", row_index=3, column_index=0),
            TableCell(text="10", row_index=3, column_index=1),
        ))
    assert all("section_title" not in chunk.metadata for chunk in _chunks(table))


def test_fallback_requires_repetition_within_same_table_and_real_data_after_header():
    table = _neutral_repeated_table()
    # 每张表只有一个表头；不能跨 TableBlock 累计签名。
    first = table.model_copy(update={"row_count": 3, "cells": table.cells[:5]})
    structure = ParsedDocument(blocks=(first, first.model_copy(update={"block_id": "b1", "order": 1})))
    chunks = indexing.candidate_to_langchain_documents(_candidate(structure))
    assert all("section_title" not in chunk.metadata for chunk in chunks)
    # 没有 data 的连续重复行不能互相当成后续 data 的证据。
    cells = table.cells[:3] + tuple(cell.model_copy(update={"row_index": 2}) for cell in table.cells[1:3])
    assert all("section_title" not in chunk.metadata for chunk in _chunks(
        first.model_copy(update={"cells": cells}),
    ))


def test_fallback_header_refresh_without_new_context_keeps_first_section():
    table = imut_admission_structure().blocks[0]
    table = table.model_copy(update={"cells": tuple(cell for cell in table.cells if cell.row_index != 6)})
    chunks = _chunks(table, chunk_size=10000)
    assert len(chunks) == 2
    assert all(chunk.metadata["section_title"] == "内蒙古本科提前B段艺术类" for chunk in chunks)
    assert [(chunk.metadata["row_start"], chunk.metadata["row_end"]) for chunk in chunks] == [(3, 5), (8, 10)]
    assert all("专业名称：专业名称" not in chunk.page_content for chunk in chunks)

"""与 parser 无关的文档结构合同及确定性正文视图。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator


STRUCTURE_MAX_BLOCKS = 10_000
STRUCTURE_MAX_CELLS = 50_000
STRUCTURE_MAX_GRID_SLOTS = 100_000
STRUCTURE_MAX_TEXT_CHARS = 1_000_000
STRUCTURE_MAX_SERIALIZED_BYTES = 8_000_000


class _StructureValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class TextBlock(_StructureValue):
    kind: Literal["text"] = "text"
    block_id: StrictStr = Field(pattern=r"^b[0-9]+$", max_length=32)
    text: StrictStr = Field(max_length=STRUCTURE_MAX_TEXT_CHARS, repr=False)
    order: StrictInt = Field(ge=0)
    page_number: StrictInt | None = Field(default=None, ge=1)
    text_role: Literal["text", "title", "section_header"] = "text"
    heading_level: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_heading(self) -> TextBlock:
        if (self.text_role == "section_header") != (self.heading_level is not None):
            raise ValueError("仅 section_header 必须提供 heading_level")
        return self


class TableCell(_StructureValue):
    text: StrictStr = Field(max_length=STRUCTURE_MAX_TEXT_CHARS, repr=False)
    row_index: StrictInt = Field(ge=0)
    column_index: StrictInt = Field(ge=0)
    row_span: StrictInt = Field(default=1, ge=1)
    column_span: StrictInt = Field(default=1, ge=1)
    is_header: StrictBool = False


class TableBlock(_StructureValue):
    kind: Literal["table"] = "table"
    block_id: StrictStr = Field(pattern=r"^b[0-9]+$", max_length=32)
    caption: StrictStr | None = Field(default=None, max_length=STRUCTURE_MAX_TEXT_CHARS, repr=False)
    order: StrictInt = Field(ge=0)
    page_number: StrictInt | None = Field(default=None, ge=1)
    cells: tuple[TableCell, ...] = Field(max_length=STRUCTURE_MAX_CELLS, repr=False)
    row_count: StrictInt = Field(ge=0, le=STRUCTURE_MAX_GRID_SLOTS)
    column_count: StrictInt = Field(ge=0, le=STRUCTURE_MAX_GRID_SLOTS)

    @model_validator(mode="after")
    def _validate_grid(self) -> TableBlock:
        if (self.row_count == 0) != (self.column_count == 0):
            raise ValueError("空表格的行数与列数必须同时为零")
        if self.row_count * self.column_count > STRUCTURE_MAX_GRID_SLOTS:
            raise ValueError("table grid 超过结构大小限制")
        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            if (cell.row_index + cell.row_span > self.row_count
                    or cell.column_index + cell.column_span > self.column_count):
                raise ValueError("table cell/span 超出表格边界")
            for row in range(cell.row_index, cell.row_index + cell.row_span):
                for column in range(cell.column_index, cell.column_index + cell.column_span):
                    if (row, column) in occupied:
                        raise ValueError("table cells 不能重叠")
                    occupied.add((row, column))
        return self


class ParsedDocument(_StructureValue):
    structure_version: Literal[1] = 1
    blocks: tuple[Annotated[TextBlock | TableBlock, Field(discriminator="kind")], ...] = Field(
        max_length=STRUCTURE_MAX_BLOCKS, repr=False,
    )
    page_count: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_reading_order(self) -> ParsedDocument:
        if any(block.order != index for index, block in enumerate(self.blocks)):
            raise ValueError("blocks 必须按连续 reading order 保存")
        if len({block.block_id for block in self.blocks}) != len(self.blocks):
            raise ValueError("block_id 必须唯一")
        if sum(len(block.cells) for block in self.blocks if isinstance(block, TableBlock)) > STRUCTURE_MAX_CELLS:
            raise ValueError("document cells 超过结构大小限制")
        if sum(
            block.row_count * block.column_count for block in self.blocks
            if isinstance(block, TableBlock)
        ) > STRUCTURE_MAX_GRID_SLOTS:
            raise ValueError("document grid 超过结构大小限制")
        if self.page_count is not None and any(
            block.page_number is not None and block.page_number > self.page_count
            for block in self.blocks
        ):
            raise ValueError("block page_number 超出文档页数")
        if len(self.model_dump_json().encode("utf-8")) > STRUCTURE_MAX_SERIALIZED_BYTES:
            raise ValueError("structured_content JSON 超过结构大小限制")
        return self


def canonical_document_text(document: ParsedDocument) -> str:
    """按 reading order 输出正文；合并区域只在锚点输出文字，结构仍以 cells 为准。"""
    parts: list[str] = []
    page_number = None
    for block in document.blocks:
        if isinstance(block, TextBlock):
            text = block.text
        else:
            if not (block.caption or "").strip() and not any(cell.text.strip() for cell in block.cells):
                continue
            rows = [[""] * block.column_count for _ in range(block.row_count)]
            for cell in block.cells:
                rows[cell.row_index][cell.column_index] = (
                    cell.text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")
                )
            text = "\n".join([
                "[表格]", *([block.caption] if block.caption else []),
                *(" | ".join(row) for row in rows), "[/表格]",
            ])
        if not text.strip():
            continue
        if block.page_number is not None and block.page_number != page_number:
            page_number = block.page_number
            parts.append(f"[第 {page_number} 页]")
        parts.append(text)
    return "\n\n".join(parts)

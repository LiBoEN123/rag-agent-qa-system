"""生产 RAG Context 选择与原字符串合同回归测试。"""

from types import SimpleNamespace

from app.rag import generation


def _chunk(index: int, text: str = "证据") -> dict:
    return {
        "reference_id": f"Reference {index}",
        "file_name": f"doc-{index}.md",
        "hybrid_score": 0.5 + index / 10,
        "chunk_id": index,
        "text": text,
    }


def _block(chunk: dict) -> str:
    return (
        f"[{chunk['reference_id']}]\n"
        f"file_name:{chunk['file_name']}\n"
        f"score:{chunk['hybrid_score']}\n"
        f"chunk_id:{chunk['chunk_id']}\n"
        f"snippet:{chunk['text']}\n"
    )


def test_context_selection_empty_input():
    context, selected = generation.build_context_with_selection([], 10)

    assert context == ""
    assert selected == []


def test_context_selection_keeps_all_chunks_and_exact_legacy_string():
    chunks = [_chunk(1, "完整证据一"), _chunk(2, "完整证据二")]
    expected = "\n".join(_block(chunk) for chunk in chunks)

    context, selected = generation.build_context_with_selection(
        chunks,
        10_000,
    )

    assert context == expected
    assert selected == chunks
    assert generation.build_context(chunks, 10_000) == expected


def test_context_selection_includes_exact_budget_boundary():
    chunks = [_chunk(1), _chunk(2)]
    max_chars = sum(len(_block(chunk)) for chunk in chunks)

    context, selected = generation.build_context_with_selection(
        chunks,
        max_chars,
    )

    assert context == "\n".join(_block(chunk) for chunk in chunks)
    assert selected == chunks


def test_context_selection_stops_wholly_when_one_character_over_budget():
    chunks = [
        _chunk(1, "证据一"),
        _chunk(2, "证据二"),
        _chunk(3, "证据三"),
    ]
    max_chars = len(_block(chunks[0])) + len(_block(chunks[1])) - 1

    context, selected = generation.build_context_with_selection(
        chunks,
        max_chars,
    )

    assert context == _block(chunks[0])
    assert selected == chunks[:1]
    assert _block(chunks[1]) not in context
    assert chunks[2]["text"] not in context


def test_context_selection_returns_empty_when_first_block_is_over_budget():
    chunk = _chunk(1, "不能被部分截断的完整文本")

    context, selected = generation.build_context_with_selection(
        [chunk],
        len(_block(chunk)) - 1,
    )

    assert context == ""
    assert selected == []
    assert chunk["text"] not in context


def test_context_selection_uses_settings_default(monkeypatch):
    chunks = [_chunk(1), _chunk(2)]
    monkeypatch.setattr(
        generation,
        "get_settings",
        lambda: SimpleNamespace(max_context_chars=len(_block(chunks[0]))),
    )

    context, selected = generation.build_context_with_selection(chunks)

    assert context == _block(chunks[0])
    assert selected == chunks[:1]

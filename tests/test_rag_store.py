from app import rag_store


def test_index_cache_exists_only_depends_on_chroma(monkeypatch):
    monkeypatch.setattr(rag_store, "chroma_collection_has_chunks", lambda: True)

    assert rag_store.index_cache_exists() is True


def test_build_docs_registry_from_chunks_rebuilds_registry_from_metadata():
    chunks = [
        {
            "text": "first",
            "metadata": {
                "doc_id": "doc_1",
                "file_name": "guide.md",
                "source_path": "D:/kb/guide.md",
                "file_type": "md",
            },
        },
        {
            "text": "second",
            "metadata": {
                "doc_id": "doc_1",
                "file_name": "guide.md",
                "source_path": "D:/kb/guide.md",
                "file_type": "md",
            },
        },
        {
            "text": "third",
            "metadata": {
                "doc_id": "doc_2",
                "file_name": "notes.txt",
                "source_path": "D:/kb/notes.txt",
                "file_type": "txt",
            },
        },
    ]

    assert rag_store.build_docs_registry_from_chunks(chunks) == [
        {
            "doc_id": "doc_1",
            "file_name": "guide.md",
            "source_path": "D:/kb/guide.md",
            "file_type": "md",
            "chunk_count": 2,
        },
        {
            "doc_id": "doc_2",
            "file_name": "notes.txt",
            "source_path": "D:/kb/notes.txt",
            "file_type": "txt",
            "chunk_count": 1,
        },
    ]


def test_load_index_cache_uses_chroma_metadata_for_docs_registry(monkeypatch):
    chunks = [
        {
            "text": "hello",
            "metadata": {
                "doc_id": "doc_1",
                "file_name": "guide.md",
                "source_path": "D:/kb/guide.md",
                "file_type": "md",
            },
        }
    ]

    monkeypatch.setattr(rag_store, "index_cache_exists", lambda: True)
    monkeypatch.setattr(rag_store, "load_chunks_from_chroma", lambda: chunks)
    monkeypatch.setattr(rag_store, "_build_bm25", lambda texts: "bm25")
    monkeypatch.setattr(
        rag_store,
        "_load_models",
        lambda model=None, rerank_model=None: ("model", "rerank_model"),
    )

    result = rag_store.load_index_cache()

    assert result["ok"] is True
    assert result["docs_registry"] == [
        {
            "doc_id": "doc_1",
            "file_name": "guide.md",
            "source_path": "D:/kb/guide.md",
            "file_type": "md",
            "chunk_count": 1,
        }
    ]
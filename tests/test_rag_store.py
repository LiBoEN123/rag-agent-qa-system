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


def test_load_models_uses_configured_model_cache_dir(monkeypatch, tmp_path):
    cache_dir = tmp_path / "models"
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, device, cache_folder):
            calls["embedding"] = {
                "model_name": model_name,
                "device": device,
                "cache_folder": cache_folder,
            }

    class FakeCrossEncoder:
        def __init__(self, model_name, device, cache_folder):
            calls["rerank"] = {
                "model_name": model_name,
                "device": device,
                "cache_folder": cache_folder,
            }

    monkeypatch.setattr(rag_store.Settings, "MODEL_CACHE_DIR", cache_dir)
    monkeypatch.setattr(rag_store, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(rag_store, "CrossEncoder", FakeCrossEncoder)

    rag_store._load_models()

    assert cache_dir.is_dir()
    assert calls["embedding"]["cache_folder"] == str(cache_dir)
    assert calls["rerank"]["cache_folder"] == str(cache_dir)
    assert calls["embedding"]["model_name"] == rag_store.Settings.EMBEDDING_MODEL_NAME
    assert calls["rerank"]["model_name"] == rag_store.Settings.RERANK_MODEL_NAME

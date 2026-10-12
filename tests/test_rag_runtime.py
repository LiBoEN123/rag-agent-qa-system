import numpy as np
import pytest

from app.rag import runtime as rag_store
from app.storage import vector




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
    doc_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(rag_store, "chroma_collection_has_chunks", lambda: True)
    monkeypatch.setattr(
        rag_store,
        "load_index_from_chroma",
        lambda: (chunks, doc_embeddings),
    )
    monkeypatch.setattr(rag_store, "_build_bm25", lambda texts: "bm25")
    monkeypatch.setattr(
        rag_store,
        "_load_models",
        lambda model=None, rerank_model=None: ("model", "rerank_model"),
    )

    result = rag_store._load_index_bundle()

    assert result["ok"] is True
    assert result["doc_embeddings"] is doc_embeddings
    assert result["docs_registry"] == [
        {
            "doc_id": "doc_1",
            "file_name": "guide.md",
            "source_path": "D:/kb/guide.md",
            "file_type": "md",
            "chunk_count": 1,
        }
    ]


def test_load_index_cache_restores_embeddings_without_reencoding_documents(monkeypatch):
    chunks = [
        {"text": "first", "metadata": {"chunk_id": 1}},
        {"text": "second", "metadata": {"chunk_id": 2}},
    ]
    doc_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class ModelThatMustNotEncode:
        def encode(self, *_args, **_kwargs):
            pytest.fail("cache load 不应重新 encode 文档")

    model = ModelThatMustNotEncode()
    rerank_model = object()
    monkeypatch.setattr(rag_store, "chroma_collection_has_chunks", lambda: True)
    monkeypatch.setattr(
        rag_store,
        "load_index_from_chroma",
        lambda: (chunks, doc_embeddings),
    )

    result = rag_store._load_index_bundle(model=model, rerank_model=rerank_model)

    assert result["ok"] is True
    assert result["chunks"] == chunks
    assert result["doc_embeddings"] is doc_embeddings
    assert result["doc_embeddings"].shape == (len(chunks), 2)
    assert result["bm25"] is not None
    assert result["model"] is model
    assert result["rerank_model"] is rerank_model


@pytest.mark.parametrize(
    ("doc_embeddings", "error_message"),
    [
        (None, "ChromaDB 索引缓存中缺少 embeddings"),
        (
            np.array([[1.0, 0.0]], dtype=np.float32),
            "ChromaDB 索引缓存中的 embedding 数量与 chunk 数量不一致",
        ),
    ],
)
def test_load_index_cache_rejects_corrupt_embeddings(
    monkeypatch,
    doc_embeddings,
    error_message,
):
    chunks = [
        {"text": "first", "metadata": {"chunk_id": 1}},
        {"text": "second", "metadata": {"chunk_id": 2}},
    ]
    monkeypatch.setattr(rag_store, "chroma_collection_has_chunks", lambda: True)
    monkeypatch.setattr(
        rag_store,
        "load_index_from_chroma",
        lambda: (chunks, doc_embeddings),
    )

    result = rag_store._load_index_bundle(model=object(), rerank_model=object())

    assert result["ok"] is False
    assert error_message in result["message"]
    assert result["chunks"] == []
    assert result["doc_embeddings"] is None


def test_load_models_uses_configured_model_cache_dir(
    monkeypatch,
    tmp_path,
    settings_env,
):
    """加载嵌入与重排模型时使用配置指定的缓存目录。"""
    cache_dir = tmp_path / "models"
    settings = settings_env(model_cache_dir_override=cache_dir)
    calls = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, model, cache_folder, model_kwargs, encode_kwargs):
            calls["embedding"] = {
                "model_name": model,
                "device": model_kwargs["device"],
                "cache_folder": cache_folder,
                "local_files_only": model_kwargs["local_files_only"],
                "normalize_embeddings": encode_kwargs["normalize_embeddings"],
            }

    class FakeCrossEncoder:
        def __init__(self, model_name, device, cache_folder, **kwargs):
            calls["rerank"] = {
                "model_name": model_name,
                "device": device,
                "cache_folder": cache_folder,
            }

    monkeypatch.setattr(
        vector,
        "HuggingFaceEmbeddings",
        FakeHuggingFaceEmbeddings,
    )
    monkeypatch.setattr(rag_store, "CrossEncoder", FakeCrossEncoder)

    rag_store._load_models()

    assert cache_dir.is_dir()
    assert calls["embedding"]["cache_folder"] == str(cache_dir)
    assert calls["rerank"]["cache_folder"] == str(cache_dir)
    assert calls["embedding"]["model_name"] == settings.embedding_model_name
    assert calls["embedding"]["local_files_only"] is True
    assert calls["embedding"]["normalize_embeddings"] is True
    assert calls["rerank"]["model_name"] == settings.rerank_model_name

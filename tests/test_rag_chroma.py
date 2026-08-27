import numpy as np
import pytest

from app.rag import chroma as rag_chroma


class FakeCollection:
    def __init__(self, data):
        self.data = data
        self.includes = []

    def get(self, include):
        self.includes.append(include)
        return self.data


def test_load_index_from_chroma_keeps_embeddings_aligned_when_sorting(monkeypatch):
    collection = FakeCollection(
        {
            "documents": ["third", "first", "second"],
            "metadatas": [
                {"chunk_id": 3},
                {"chunk_id": 1},
                {"chunk_id": 2},
            ],
            "embeddings": [
                [3.0, 30.0],
                [1.0, 10.0],
                [2.0, 20.0],
            ],
        }
    )
    monkeypatch.setattr(
        rag_chroma,
        "get_chroma_collection",
        lambda create=False, settings=None: collection,
    )

    chunks, doc_embeddings = rag_chroma.load_index_from_chroma()

    assert collection.includes == [["documents", "metadatas", "embeddings"]]
    assert [chunk["metadata"]["chunk_id"] for chunk in chunks] == [1, 2, 3]
    np.testing.assert_array_equal(
        doc_embeddings,
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32),
    )
    assert doc_embeddings.dtype == np.float32


@pytest.mark.parametrize(
    ("embeddings", "error_message"),
    [
        (None, "缺少 embeddings"),
        ([[1.0, 10.0]], "embedding 数量与 document 数量不一致"),
        ([1.0, 10.0], "embeddings 必须是非空二维矩阵"),
    ],
)
def test_load_index_from_chroma_rejects_corrupt_embeddings(
    monkeypatch,
    embeddings,
    error_message,
):
    collection = FakeCollection(
        {
            "documents": ["first", "second"],
            "metadatas": [{"chunk_id": 1}, {"chunk_id": 2}],
            "embeddings": embeddings,
        }
    )
    monkeypatch.setattr(
        rag_chroma,
        "get_chroma_collection",
        lambda create=False, settings=None: collection,
    )

    with pytest.raises(ValueError, match=error_message):
        rag_chroma.load_index_from_chroma()

from __future__ import annotations

import numpy as np

from rag.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder


class FakeModel:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        return np.array([[float(len(t)), 0.0] for t in texts])


def test_embed_documents_returns_list_of_lists(monkeypatch):
    monkeypatch.setattr(
        "rag.embedders.sentence_transformers_embedder.SentenceTransformer", FakeModel
    )
    embedder = SentenceTransformersEmbedder(model_name="fake-model")

    vectors = embedder.embed_documents(["ab", "abcd"])

    assert vectors == [[2.0, 0.0], [4.0, 0.0]]


def test_embed_query_returns_single_vector(monkeypatch):
    monkeypatch.setattr(
        "rag.embedders.sentence_transformers_embedder.SentenceTransformer", FakeModel
    )
    embedder = SentenceTransformersEmbedder(model_name="fake-model")

    vector = embedder.embed_query("abc")

    assert vector == [3.0, 0.0]

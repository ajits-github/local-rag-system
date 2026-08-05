from __future__ import annotations

import numpy as np

from rag.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder


class FakeModel:
    """Stand-in for `SentenceTransformer` that encodes text length, not semantics."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept and ignore any `SentenceTransformer` constructor args."""

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        """Return a 2-D vector per text: [len(text), 0.0]."""
        return np.array([[float(len(t)), 0.0] for t in texts])


def test_embed_documents_returns_list_of_lists(monkeypatch):
    """embed_documents returns one plain-list vector per input text."""
    monkeypatch.setattr(
        "rag.embedders.sentence_transformers_embedder.SentenceTransformer", FakeModel
    )
    embedder = SentenceTransformersEmbedder(model_name="fake-model")

    vectors = embedder.embed_documents(["ab", "abcd"])

    assert vectors == [[2.0, 0.0], [4.0, 0.0]]


def test_embed_query_returns_single_vector(monkeypatch):
    """embed_query returns a single vector, not a batch of one."""
    monkeypatch.setattr(
        "rag.embedders.sentence_transformers_embedder.SentenceTransformer", FakeModel
    )
    embedder = SentenceTransformersEmbedder(model_name="fake-model")

    vector = embedder.embed_query("abc")

    assert vector == [3.0, 0.0]

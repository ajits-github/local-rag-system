"""Default `Embedder`: a local `sentence-transformers` model."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from rag.embedders.base import Embedder


class SentenceTransformersEmbedder(Embedder):
    """Embeds text locally on CPU/GPU via a `sentence-transformers` model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        """Load the model once and cache batching/normalization settings.

        Parameters
        ----------
        model_name : str, optional
            Hugging Face model id or local path, by default
            ``"sentence-transformers/all-MiniLM-L6-v2"``.
        device : str, optional
            Torch device to run inference on, by default ``"cpu"``.
        batch_size : int, optional
            Batch size used by `embed_documents`, by default 32.
        normalize : bool, optional
            Whether to L2-normalize output embeddings, by default True.
        """
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts for storage.

        Parameters
        ----------
        texts : list[str]
            Chunk texts to embed.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text, same order.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval.

        Parameters
        ----------
        text : str
            Query text to embed.

        Returns
        -------
        list[float]
            The query's embedding vector.
        """
        return self.embed_documents([text])[0]

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chunkers.base import Chunker


class RecursiveCharacterChunker(Chunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)

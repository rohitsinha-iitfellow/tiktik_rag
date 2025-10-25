"""Embedding utilities for persisting chunks in vector stores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence

from .metadata import ContentChunk


class EmbeddingModel(Protocol):
    """Minimal protocol implemented by embedding providers."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return an embedding for each provided text."""


class VectorStore(Protocol):
    """Protocol for vector database clients."""

    def add(
        self,
        *,
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
        documents: Sequence[str],
    ) -> None:
        """Persist embeddings with associated metadata."""


@dataclass
class StoredChunk:
    """Container capturing what has been written to the vector store."""

    text: str
    metadata: dict
    embedding: Sequence[float]


class ChunkEmbeddingPipeline:
    """Coordinate chunk embedding and persistence into a vector store."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.batch_size = batch_size

    def _batched(self, chunks: Sequence[ContentChunk]) -> Iterable[Sequence[ContentChunk]]:
        for start in range(0, len(chunks), self.batch_size):
            yield chunks[start : start + self.batch_size]

    def ingest(self, chunks: Sequence[ContentChunk]) -> List[StoredChunk]:
        """Embed the supplied chunks and persist them to the vector store."""

        filtered = [chunk for chunk in chunks if chunk.text.strip()]
        stored: List[StoredChunk] = []

        for batch in self._batched(filtered):
            texts = [chunk.text for chunk in batch]
            embeddings = list(self.embedding_model.embed(texts))
            if len(embeddings) != len(batch):
                raise ValueError("embedding_model returned a mismatched number of embeddings")

            metadatas = [chunk.metadata.as_dict() for chunk in batch]
            self.vector_store.add(embeddings=embeddings, metadatas=metadatas, documents=texts)

            for text, metadata, embedding in zip(texts, metadatas, embeddings):
                stored.append(StoredChunk(text=text, metadata=metadata, embedding=embedding))

        return stored


__all__ = [
    "ChunkEmbeddingPipeline",
    "EmbeddingModel",
    "VectorStore",
    "StoredChunk",
]

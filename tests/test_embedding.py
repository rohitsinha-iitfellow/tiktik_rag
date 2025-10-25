from typing import List, Sequence

import pytest

from tiktik_rag.embedding import ChunkEmbeddingPipeline, StoredChunk
from tiktik_rag.metadata import ContentChunk, Metadata


class DummyEmbeddingModel:
    def __init__(self) -> None:
        self.calls: List[Sequence[str]] = []

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]


class DummyVectorStore:
    def __init__(self) -> None:
        self.records: List[StoredChunk] = []

    def add(self, *, embeddings, metadatas, documents) -> None:  # type: ignore[override]
        for text, metadata, embedding in zip(documents, metadatas, embeddings):
            self.records.append(StoredChunk(text=text, metadata=metadata, embedding=embedding))


def test_pipeline_embeds_and_persists_chunks():
    model = DummyEmbeddingModel()
    store = DummyVectorStore()
    pipeline = ChunkEmbeddingPipeline(model, store, batch_size=1)

    chunks = [
        ContentChunk(text="First chunk", metadata=Metadata(source="doc", page=1)),
        ContentChunk(
            text="Second chunk",
            metadata=Metadata(source="doc", timestamp_start=0.0, timestamp_end=1.0, asset_url="https://media.mp4"),
        ),
    ]

    stored = pipeline.ingest(chunks)

    assert len(model.calls) == 2
    assert len(store.records) == 2
    assert stored == store.records
    assert stored[1].metadata["asset_url"] == "https://media.mp4"


def test_pipeline_validates_embedding_length():
    class BrokenEmbeddingModel(DummyEmbeddingModel):
        def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            super().embed(texts)
            return []  # missing embedding

    pipeline = ChunkEmbeddingPipeline(BrokenEmbeddingModel(), DummyVectorStore())
    chunk = ContentChunk(text="Hello", metadata=Metadata(source="doc"))

    with pytest.raises(ValueError):
        pipeline.ingest([chunk])

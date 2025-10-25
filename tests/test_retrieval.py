from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from tiktik_rag.metadata import ContentChunk, Metadata
from tiktik_rag.retrieval import ChunkRetriever, RetrievedChunk


@dataclass
class FakeSearcher:
    results: Sequence[RetrievedChunk]

    def search(self, query: str, *, top_k: int) -> Sequence[RetrievedChunk]:  # type: ignore[override]
        return list(self.results)[:top_k]


@dataclass
class FakeCrossEncoder:
    scores: Sequence[float]

    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]:  # type: ignore[override]
        assert len(texts) <= len(self.scores)
        return self.scores[: len(texts)]


@pytest.fixture
def base_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=ContentChunk(
                text="General introduction",
                metadata=Metadata(source="doc.pdf", page=1),
            ),
            score=0.4,
        ),
        RetrievedChunk(
            chunk=ContentChunk(
                text="Figure caption",
                metadata=Metadata(source="doc.pdf", page=2, figure="3", asset_url="https://example.com/fig3.png"),
            ),
            score=0.3,
        ),
        RetrievedChunk(
            chunk=ContentChunk(
                text="Transcript snippet",
                metadata=Metadata(source="talk.mp3", timestamp_start=5, timestamp_end=15),
            ),
            score=0.2,
        ),
    ]


def test_visual_query_prefers_captions(base_chunks: list[RetrievedChunk]) -> None:
    searcher = FakeSearcher(results=list(reversed(base_chunks)))
    retriever = ChunkRetriever(searcher)

    results = retriever.retrieve("What does figure 3 show?", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.metadata.figure == "3"
    assert results[0].chunk.metadata.asset_url == "https://example.com/fig3.png"
    assert results[1].chunk.metadata.page == 1


def test_cross_encoder_reranks_for_technical_query(base_chunks: list[RetrievedChunk]) -> None:
    # Set the searcher order opposite to the cross encoder preference.
    searcher = FakeSearcher(results=base_chunks)
    cross_encoder = FakeCrossEncoder(scores=[0.1, 0.9, 0.8])
    retriever = ChunkRetriever(searcher, cross_encoder=cross_encoder)

    results = retriever.retrieve("Explain the RSA algorithm", top_k=3)

    assert [chunk.score for chunk in results] == pytest.approx([0.9, 0.8, 0.1])
    assert results[0].chunk.text == "Figure caption"
    assert results[1].chunk.text == "Transcript snippet"
    assert results[2].chunk.text == "General introduction"

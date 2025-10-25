"""Multi-modal retrieval utilities for TikTik RAG."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Protocol, Sequence

from .metadata import ContentChunk
from .modalities import determine_modality


class SimilaritySearcher(Protocol):
    """Protocol implemented by vector stores capable of similarity search."""

    def search(self, query: str, *, top_k: int) -> Sequence["RetrievedChunk"]:
        """Return the most similar chunks for the supplied query."""


class CrossEncoder(Protocol):
    """Protocol implemented by re-ranking models such as cross encoders."""

    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        """Return a relevance score for each ``query``/``text`` pair."""


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a chunk emitted by the retrieval pipeline."""

    chunk: ContentChunk
    score: float

    @property
    def modality(self) -> str:
        return determine_modality(self.chunk.metadata)


def _query_mentions_visuals(query: str) -> bool:
    visual_terms = {
        "figure",
        "image",
        "diagram",
        "visual",
        "illustration",
        "photo",
        "picture",
        "chart",
        "graph",
        "screenshot",
    }
    lowered = query.lower()
    return any(term in lowered for term in visual_terms)


def _query_seems_technical(query: str) -> bool:
    lowered = query.lower()
    technical_terms = {
        "algorithm",
        "specification",
        "protocol",
        "architecture",
        "calculation",
        "derivation",
        "complexity",
        "implementation",
        "analysis",
    }
    if any(term in lowered for term in technical_terms):
        return True
    # Numeric expressions often indicate technical content.
    return any(char.isdigit() for char in query)


class ChunkRetriever:
    """Coordinate modality-aware retrieval with optional re-ranking."""

    def __init__(
        self,
        searcher: SimilaritySearcher,
        *,
        cross_encoder: CrossEncoder | None = None,
        oversample_factor: int = 3,
    ) -> None:
        if oversample_factor <= 0:
            raise ValueError("oversample_factor must be greater than zero")
        self.searcher = searcher
        self.cross_encoder = cross_encoder
        self.oversample_factor = oversample_factor

    def retrieve(self, query: str, *, top_k: int = 5) -> List[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        search_k = max(top_k, top_k * self.oversample_factor)
        candidates = list(self.searcher.search(query, top_k=search_k))
        if not candidates:
            return []

        visual_focus = _query_mentions_visuals(query)
        selected: List[RetrievedChunk]

        if visual_focus:
            caption_candidates = [chunk for chunk in candidates if chunk.modality == "caption"]
            if caption_candidates:
                selected = sorted(caption_candidates, key=lambda chunk: chunk.score, reverse=True)
            else:
                selected = sorted(candidates, key=lambda chunk: chunk.score, reverse=True)
        else:
            selected = sorted(candidates, key=lambda chunk: chunk.score, reverse=True)

        if len(selected) > top_k:
            selected = selected[:top_k]
        elif len(selected) < top_k:
            missing = top_k - len(selected)
            fallback = [
                chunk
                for chunk in sorted(candidates, key=lambda chunk: chunk.score, reverse=True)
                if chunk not in selected
            ]
            selected.extend(fallback[:missing])

        if self.cross_encoder and _query_seems_technical(query):
            rerank_scores = list(self.cross_encoder.score(query, [chunk.chunk.text for chunk in selected]))
            if len(rerank_scores) != len(selected):
                raise ValueError("cross_encoder returned a mismatched number of scores")
            selected = [replace(chunk, score=score) for chunk, score in zip(selected, rerank_scores)]
            selected.sort(key=lambda chunk: chunk.score, reverse=True)

        return selected


__all__ = [
    "ChunkRetriever",
    "RetrievedChunk",
    "SimilaritySearcher",
    "CrossEncoder",
]

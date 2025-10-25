"""Utilities for composing grounded responses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from .metadata import Metadata
from .modalities import determine_modality
from .retrieval import RetrievedChunk


@dataclass(frozen=True)
class ResponsePayload:
    """Structured response returned to downstream callers."""

    answer: str
    citations: List[str]
    assets: List[str]


class ResponseComposer:
    """Compose answers enriched with detailed citations and assets."""

    def __init__(self) -> None:
        pass

    def compose(self, answer: str, chunks: Sequence[RetrievedChunk]) -> ResponsePayload:
        citations: List[str] = []
        assets: List[str] = []
        seen: Set[Tuple[object, ...]] = set()

        for retrieved in chunks:
            metadata = retrieved.chunk.metadata
            key = (
                metadata.source,
                metadata.page,
                metadata.figure,
                metadata.timestamp_start,
                metadata.timestamp_end,
            )
            if key not in seen:
                seen.add(key)
                citations.append(_format_citation(metadata))

            modality = determine_modality(metadata)
            if modality == "caption" and metadata.asset_url:
                if metadata.asset_url not in assets:
                    assets.append(metadata.asset_url)

        return ResponsePayload(answer=answer, citations=citations, assets=assets)


def _format_citation(metadata: Metadata) -> str:
    parts: List[str] = [metadata.source]
    if metadata.page is not None:
        parts.append(f"p. {metadata.page}")
    if metadata.figure:
        parts.append(f"fig. {metadata.figure}")
    timestamp = _format_timestamp_range(metadata.timestamp_start, metadata.timestamp_end)
    if timestamp:
        parts.append(timestamp)
    return " ".join(parts)


def _format_timestamp_range(start: float | None, end: float | None) -> str | None:
    if start is None and end is None:
        return None

    if start is not None and end is not None:
        return f"{_format_timestamp(start)}-{_format_timestamp(end)}"

    if start is not None:
        return _format_timestamp(start)

    return _format_timestamp(end) if end is not None else None


def _format_timestamp(value: float) -> str:
    total_seconds = max(value, 0)
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


__all__ = ["ResponseComposer", "ResponsePayload"]

"""Utilities for normalising text into retrieval-friendly chunks."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Mapping, Sequence, Tuple

from .metadata import Caption, ContentChunk, Metadata


def _normalise_step(chunk_size: int, chunk_overlap: int) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to zero")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return chunk_size - chunk_overlap


def chunk_content_chunks(
    chunks: Sequence[ContentChunk],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[ContentChunk]:
    """Split content chunks into overlapping windows.

    Parameters
    ----------
    chunks:
        Original chunks sourced from PDFs, captions, or transcripts.
    chunk_size:
        Maximum number of characters in the resulting chunks.
    chunk_overlap:
        Number of characters shared between adjacent chunks. Must be smaller
        than ``chunk_size``.
    """

    step = _normalise_step(chunk_size, chunk_overlap)
    windowed: List[ContentChunk] = []

    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue

        if len(text) <= chunk_size:
            windowed.append(chunk)
            continue

        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            piece = text[start:end].strip()
            if piece:
                metadata = replace(chunk.metadata)
                windowed.append(ContentChunk(text=piece, metadata=metadata))
            start += step

    return windowed


def captions_to_chunks(
    captions: Iterable[Caption],
    *,
    asset_lookup: Mapping[Tuple[str, int, str], str] | None = None,
) -> List[ContentChunk]:
    """Convert caption objects into retrievable content chunks."""

    chunks: List[ContentChunk] = []
    for caption in captions:
        key_tuple = caption.key.as_tuple()
        asset_url = asset_lookup.get(key_tuple) if asset_lookup else None
        metadata = Metadata(
            source=caption.key.doc_id,
            page=caption.key.page,
            figure=caption.key.figure_id,
            asset_url=asset_url,
        )
        chunks.append(ContentChunk(text=caption.text.strip(), metadata=metadata))
    return chunks


__all__ = ["chunk_content_chunks", "captions_to_chunks"]

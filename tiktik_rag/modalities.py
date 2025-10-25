"""Helpers for reasoning about content modalities."""
from __future__ import annotations

from .metadata import Metadata


def determine_modality(metadata: Metadata) -> str:
    """Infer a modality label for a chunk based on its metadata."""

    if metadata.figure is not None or metadata.asset_url is not None:
        return "caption"
    if metadata.timestamp_start is not None or metadata.timestamp_end is not None:
        return "audio"
    return "text"


__all__ = ["determine_modality"]

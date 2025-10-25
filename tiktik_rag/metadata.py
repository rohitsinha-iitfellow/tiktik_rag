"""Shared data structures for TikTik RAG ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class CaptionKey:
    """Uniquely identifies an image caption within a PDF document."""

    doc_id: str
    page: int
    figure_id: str

    def as_tuple(self) -> Tuple[str, int, str]:
        """Return a tuple representation suitable for dictionary keys."""
        return (self.doc_id, self.page, self.figure_id)


@dataclass(frozen=True)
class Caption:
    """Represents a caption supplied for an embedded media item."""

    key: CaptionKey
    text: str


@dataclass
class Metadata:
    """Normalised metadata payload for downstream components."""

    source: str
    page: Optional[int] = None
    figure: Optional[str] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    asset_url: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[Any]]:
        """Convert the metadata into a JSON serialisable dictionary."""
        payload: Dict[str, Optional[Any]] = {
            "source": self.source,
            "page": self.page,
            "figure": self.figure,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "asset_url": self.asset_url,
        }
        return {key: value for key, value in payload.items() if value is not None or key == "source"}


@dataclass
class ContentChunk:
    """Represents a unit of text along with associated metadata."""

    text: str
    metadata: Metadata

    def as_record(self) -> Dict[str, object]:
        """Return a serialisable representation of the chunk."""
        return {"text": self.text, "metadata": self.metadata.as_dict()}

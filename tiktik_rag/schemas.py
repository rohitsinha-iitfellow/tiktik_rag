"""Pydantic models describing the public FastAPI contract.

These models are used to generate the OpenAPI schema exposed by FastAPI so
that the interactive Swagger UI can be used to exercise the service from a
browser.  The definitions mirror the JSON payloads accepted and returned by
``create_app`` in :mod:`tiktik_rag.service`.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from pydantic import BaseModel, Field, field_validator


class MetadataPayload(BaseModel):
    """Metadata describing the origin of an ingested chunk."""

    source: str = Field(..., description="Identifier for the source document")
    page: Optional[int] = Field(
        default=None, description="Page number the chunk originated from"
    )
    figure: Optional[str] = Field(
        default=None, description="Figure identifier referenced by the chunk"
    )
    timestamp_start: Optional[float] = Field(
        default=None,
        description="Start timestamp in seconds for audio/video derived chunks",
    )
    timestamp_end: Optional[float] = Field(
        default=None,
        description="End timestamp in seconds for audio/video derived chunks",
    )
    asset_url: Optional[str] = Field(
        default=None, description="URL of any related external asset"
    )

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("source must be a non-empty string")
        return trimmed


class ChunkPayload(BaseModel):
    """Payload describing an ingestible chunk."""

    text: str = Field(..., description="The textual contents of the chunk")
    metadata: MetadataPayload

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("text must be a non-empty string")
        return trimmed


class IngestRequest(BaseModel):
    """Request body for ``POST /ingest``."""

    chunks: List[ChunkPayload]
    replace_existing: bool = Field(
        default=False,
        description="Whether to replace existing chunks for matching sources",
    )


class IngestResponse(BaseModel):
    """Response payload for ``POST /ingest``."""

    ingested: int
    metrics: Dict[str, int]


class ReindexDocumentPayload(BaseModel):
    """Document payload used for batch re-indexing."""

    source: str
    chunks: List[ChunkPayload]

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("source must be a non-empty string")
        return trimmed


class ReindexRequest(BaseModel):
    """Request body for ``POST /reindex``."""

    documents: List[ReindexDocumentPayload]


class ReindexResponse(BaseModel):
    """Response payload for ``POST /reindex``."""

    results: Dict[str, int]
    metrics: Dict[str, int]


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    query: str
    top_k: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of citations to retrieve",
    )

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("query must be a non-empty string")
        return trimmed


class QueryResponse(BaseModel):
    """Response payload for ``POST /query``."""

    answer: str
    citations: List[str]
    assets: Dict[str, str]


class PDFIngestRequest(BaseModel):
    """Request body for ``POST /ingest/pdf``."""

    pdf_path: str
    doc_id: Optional[str] = Field(
        default=None, description="Identifier to assign to the ingested PDF"
    )
    captions: Optional[Dict[int, Dict[str, str]]] = Field(
        default=None,
        description="Optional mapping of page -> figure -> caption text",
    )
    assets: Optional[Dict[int, Dict[str, str]]] = Field(
        default=None,
        description="Optional mapping of page -> figure -> asset URL",
    )
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    replace_existing: bool = Field(default=False)

    @field_validator("pdf_path")
    @classmethod
    def _validate_pdf_path(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("pdf_path must be a non-empty string")
        return trimmed

    @field_validator("doc_id")
    @classmethod
    def _validate_doc_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("doc_id must be a non-empty string if provided")
        return trimmed


class PDFIngestResponse(BaseModel):
    """Response payload for ``POST /ingest/pdf``."""

    doc_id: str
    ingested: int
    captions: Dict[int, Dict[str, str]]
    metrics: Dict[str, int]


class MediaIngestRequest(BaseModel):
    """Request body for ``POST /ingest/media``."""

    media_path: str
    file_id: Optional[str] = Field(
        default=None, description="Identifier to assign to the ingested media"
    )
    model_name: str = Field(default="base")
    word_timestamps: bool = Field(default=False)
    replace_existing: bool = Field(default=False)
    chunk_size: Optional[int] = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    transcribe_kwargs: Optional[Mapping[str, object]] = Field(default=None)
    load_kwargs: Optional[Mapping[str, object]] = Field(default=None)

    @field_validator("media_path")
    @classmethod
    def _validate_media_path(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("media_path must be a non-empty string")
        return trimmed

    @field_validator("file_id")
    @classmethod
    def _validate_file_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("file_id must be a non-empty string if provided")
        return trimmed


class MediaIngestResponse(BaseModel):
    """Response payload for ``POST /ingest/media``."""

    file_id: str
    ingested: int
    transcript: str
    segments: List[Dict[str, object]]
    metrics: Dict[str, int]


__all__ = [
    "ChunkPayload",
    "IngestRequest",
    "IngestResponse",
    "MediaIngestRequest",
    "MediaIngestResponse",
    "MetadataPayload",
    "PDFIngestRequest",
    "PDFIngestResponse",
    "QueryRequest",
    "QueryResponse",
    "ReindexDocumentPayload",
    "ReindexRequest",
    "ReindexResponse",
]


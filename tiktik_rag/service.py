"""FastAPI service exposing TikTik RAG ingestion and query flows."""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, TYPE_CHECKING

try:  # pragma: no cover - optional dependency in constrained environments
    from fastapi import FastAPI, HTTPException
except ModuleNotFoundError:  # pragma: no cover - fallback used in tests
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:  # type: ignore[override]
        def __init__(self, title: str | None = None) -> None:
            self.title = title
            self._routes: Dict[tuple[str, str], callable] = {}

        def post(self, path: str, response_model=None):
            def decorator(func):
                self._routes[("POST", path)] = func
                return func

            return decorator

        def get(self, path: str, response_model=None):
            def decorator(func):
                self._routes[("GET", path)] = func
                return func

            return decorator

        @property
        def routes(self) -> Dict[tuple[str, str], callable]:
            return self._routes

from .chunking import captions_to_chunks, chunk_content_chunks
from .embedding import ChunkEmbeddingPipeline, EmbeddingModel, StoredChunk
from .media_transcriber import TranscriptionResult, WhisperTranscriber
from .metadata import Caption, ContentChunk, Metadata
from .pdf_loader import PDFLoader
from .response import ResponseComposer, ResponsePayload
from .retrieval import ChunkRetriever, CrossEncoder, RetrievedChunk, SimilaritySearcher

try:  # pragma: no cover - optional dependency in constrained environments
    from pydantic import BaseModel, ValidationError
except ModuleNotFoundError:  # pragma: no cover - fallback when Pydantic unavailable
    BaseModel = object  # type: ignore[assignment]
    ValidationError = Exception  # type: ignore[assignment]
    HAS_PYDANTIC = False
else:
    HAS_PYDANTIC = True

if TYPE_CHECKING:
    from .schemas import (
        ChunkPayload,
        IngestRequest,
        IngestResponse,
        MediaIngestRequest,
        MediaIngestResponse,
        PDFIngestRequest,
        PDFIngestResponse,
        QueryRequest,
        QueryResponse,
        ReindexRequest,
        ReindexResponse,
    )

if HAS_PYDANTIC:
    from .schemas import (
        ChunkPayload,
        IngestRequest,
        IngestResponse,
        MediaIngestRequest,
        MediaIngestResponse,
        PDFIngestRequest,
        PDFIngestResponse,
        QueryRequest,
        QueryResponse,
        ReindexRequest,
        ReindexResponse,
    )

    ModelT = TypeVar("ModelT", bound=BaseModel)

    def _parse_model(
        payload: BaseModel | Mapping[str, object],
        model_type: Type[ModelT],
        *,
        context: str,
    ) -> ModelT:
        if isinstance(payload, model_type):
            return payload
        if isinstance(payload, BaseModel):
            data = payload.model_dump()
        elif isinstance(payload, Mapping):
            data = payload
        else:  # pragma: no cover - defensive programming
            raise HTTPException(status_code=400, detail=f"{context} payload must be an object")

        try:
            return model_type.model_validate(data)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc


    def _chunk_payload_to_content(chunk_payload: "ChunkPayload") -> ContentChunk:
        metadata_payload = chunk_payload.metadata
        metadata = Metadata(
            source=metadata_payload.source,
            page=metadata_payload.page,
            figure=metadata_payload.figure,
            timestamp_start=metadata_payload.timestamp_start,
            timestamp_end=metadata_payload.timestamp_end,
            asset_url=metadata_payload.asset_url,
        )
        return ContentChunk(text=chunk_payload.text, metadata=metadata)


logger = logging.getLogger("tiktik_rag.service")


def _ensure_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class RAGServiceConfig:
    """Configuration controlling batching and retrieval behaviour."""

    default_top_k: int = 5
    embedding_batch_size: int = 32
    reindex_batch_size: int = 256

    def __post_init__(self) -> None:
        _ensure_positive("default_top_k", self.default_top_k)
        _ensure_positive("embedding_batch_size", self.embedding_batch_size)
        _ensure_positive("reindex_batch_size", self.reindex_batch_size)


@dataclass
class ServiceMetrics:
    """Simple analytics payload for monitoring service activity."""

    ingested_documents: int = 0
    ingested_chunks: int = 0
    queries: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "ingested_documents": self.ingested_documents,
            "ingested_chunks": self.ingested_chunks,
            "queries": self.queries,
        }


@dataclass(frozen=True)
class DocumentRecord:
    """Metadata describing an uploaded source document."""

    doc_id: str
    kind: str
    source_path: str
    captions: Dict[int, Dict[str, str]]


class DocumentRegistry:
    """Assign and track unique identifiers for uploaded documents."""

    def __init__(self) -> None:
        self._records: Dict[str, DocumentRecord] = {}

    def _generate_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    def register(
        self,
        *,
        kind: str,
        source_path: Path | str,
        requested_id: Optional[str] = None,
        captions: Optional[Mapping[int, Mapping[str, str]]] = None,
        replace_existing: bool = False,
    ) -> DocumentRecord:
        doc_id = (requested_id or self._generate_id(kind)).strip()
        if not doc_id:
            raise ValueError("document id must be a non-empty string")
        if not replace_existing and doc_id in self._records:
            raise ValueError(f"document with id '{doc_id}' is already registered")

        normalised_captions: Dict[int, Dict[str, str]] = {}
        for page, figure_map in (captions or {}).items():
            normalised_captions[int(page)] = {str(figure): str(text) for figure, text in figure_map.items()}

        record = DocumentRecord(
            doc_id=doc_id,
            kind=kind,
            source_path=str(source_path),
            captions=normalised_captions,
        )
        self._records[doc_id] = record
        return record

    def get(self, doc_id: str) -> DocumentRecord:
        return self._records[doc_id]


@dataclass(frozen=True)
class DocumentIngestionResult:
    """Outcome information after ingesting a document."""

    doc_id: str
    ingested_chunks: int
    caption_index: Dict[int, Dict[str, str]]


@dataclass(frozen=True)
class MediaIngestionResult:
    """Outcome information after ingesting an audio or video asset."""

    file_id: str
    ingested_chunks: int
    transcription: TranscriptionResult


class InMemoryVectorStore:
    """In-memory persistence backing the default service implementation."""

    def __init__(self) -> None:
        self.records: List[StoredChunk] = []

    def add(self, *, embeddings, metadatas, documents) -> None:  # type: ignore[override]
        for text, metadata, embedding in zip(documents, metadatas, embeddings):
            self.records.append(StoredChunk(text=text, metadata=dict(metadata), embedding=list(embedding)))

    def clear_source(self, source: str) -> None:
        self.records = [record for record in self.records if record.metadata.get("source") != source]


def _normalise_caption_mapping(
    doc_id: str,
    caption_payload: Mapping[int, Mapping[str, str]] | None,
) -> List[Caption]:
    if not caption_payload:
        return []
    return PDFLoader.build_captions_from_dict(doc_id, caption_payload)


def _build_asset_lookup(
    doc_id: str, asset_payload: Mapping[int, Mapping[str, str]] | None
) -> Dict[Tuple[str, int, str], str]:
    lookup: Dict[Tuple[str, int, str], str] = {}
    if not asset_payload:
        return lookup
    for page_key, figure_map in asset_payload.items():
        page = int(page_key)
        for figure_id, asset_url in figure_map.items():
            lookup[(doc_id, page, str(figure_id))] = str(asset_url)
    return lookup


def _dict_to_metadata(payload: Mapping[str, object]) -> Metadata:
    return Metadata(
        source=str(payload["source"]),
        page=int(payload["page"]) if payload.get("page") is not None else None,
        figure=str(payload["figure"]) if payload.get("figure") is not None else None,
        timestamp_start=float(payload["timestamp_start"]) if payload.get("timestamp_start") is not None else None,
        timestamp_end=float(payload["timestamp_end"]) if payload.get("timestamp_end") is not None else None,
        asset_url=str(payload["asset_url"]) if payload.get("asset_url") is not None else None,
    )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0

    numerator = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(l * l for l in left))
    right_norm = math.sqrt(sum(r * r for r in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class EmbeddingSimilaritySearcher(SimilaritySearcher):
    """Similarity search built on top of :class:`InMemoryVectorStore`."""

    def __init__(self, embedding_model: EmbeddingModel, store: InMemoryVectorStore) -> None:
        self.embedding_model = embedding_model
        self.store = store

    def search(self, query: str, *, top_k: int) -> Sequence[RetrievedChunk]:  # type: ignore[override]
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not self.store.records:
            return []

        query_embedding = list(self.embedding_model.embed([query])[0])
        ranked: List[RetrievedChunk] = []
        for record in self.store.records:
            metadata = _dict_to_metadata(record.metadata)
            chunk = ContentChunk(text=record.text, metadata=metadata)
            score = _cosine_similarity(query_embedding, record.embedding)
            ranked.append(RetrievedChunk(chunk=chunk, score=score))

        ranked.sort(key=lambda candidate: candidate.score, reverse=True)
        return ranked[:top_k]


class SimpleAnswerSynthesiser:
    """Fallback answer generator concatenating retrieved context."""

    def __call__(self, chunks: Sequence[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant information was retrieved."
        texts = [chunk.chunk.text.strip() for chunk in chunks if chunk.chunk.text.strip()]
        return "\n\n".join(texts)


class RAGService:
    """Coordinate ingestion, batch re-indexing, and querying."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        config: RAGServiceConfig | None = None,
        cross_encoder: CrossEncoder | None = None,
        response_composer: ResponseComposer | None = None,
    ) -> None:
        self.config = config or RAGServiceConfig()
        self.metrics = ServiceMetrics()
        self.vector_store = InMemoryVectorStore()
        self.embedding_model = embedding_model
        self.document_registry = DocumentRegistry()
        self.pipeline = ChunkEmbeddingPipeline(
            embedding_model,
            self.vector_store,
            batch_size=self.config.embedding_batch_size,
        )
        self.searcher = EmbeddingSimilaritySearcher(embedding_model, self.vector_store)
        self.retriever = ChunkRetriever(self.searcher, cross_encoder=cross_encoder)
        self.response_composer = response_composer or ResponseComposer()
        self.answer_synthesiser = SimpleAnswerSynthesiser()

    def ingest(self, chunks: Sequence[ContentChunk], *, replace_existing: bool = False) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            logger.info("Received ingestion request with no chunks; skipping")
            return 0

        sources = {chunk.metadata.source for chunk in chunk_list}
        if replace_existing:
            for source in sources:
                logger.info("Clearing existing chunks for source=%s", source)
                self.vector_store.clear_source(source)

        stored = self.pipeline.ingest(chunk_list)
        self.metrics.ingested_documents += len(sources)
        self.metrics.ingested_chunks += len(stored)
        logger.info(
            "Ingested %d chunks across %d documents", len(stored), len(sources)
        )
        return len(stored)

    def ingest_pdf_document(
        self,
        pdf_path: Path | str,
        doc_id: str | None = None,
        *,
        captions: Mapping[int, Mapping[str, str]] | None = None,
        assets: Mapping[int, Mapping[str, str]] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        replace_existing: bool = False,
    ) -> DocumentIngestionResult:
        registration = self.document_registry.register(
            kind="pdf",
            source_path=pdf_path,
            requested_id=doc_id,
            captions=captions,
            replace_existing=replace_existing,
        )
        resolved_doc_id = registration.doc_id
        caption_objects = _normalise_caption_mapping(resolved_doc_id, captions)
        loader = PDFLoader(pdf_path, resolved_doc_id, caption_objects)
        page_chunks, caption_map = loader.load()
        text_chunks = chunk_content_chunks(
            page_chunks, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        asset_lookup = _build_asset_lookup(resolved_doc_id, assets)
        caption_chunks = captions_to_chunks(caption_map.values(), asset_lookup=asset_lookup)
        all_chunks = list(text_chunks) + list(caption_chunks)
        ingested = self.ingest(all_chunks, replace_existing=replace_existing)
        return DocumentIngestionResult(
            doc_id=resolved_doc_id,
            ingested_chunks=ingested,
            caption_index=dict(registration.captions),
        )

    def ingest_media_transcript(
        self,
        media_path: Path | str,
        file_id: str,
        *,
        model_name: str = "base",
        word_timestamps: bool = False,
        replace_existing: bool = False,
        chunk_size: int | None = 1000,
        chunk_overlap: int = 200,
        transcribe_kwargs: Mapping[str, object] | None = None,
        load_kwargs: Mapping[str, object] | None = None,
    ) -> MediaIngestionResult:
        registration = self.document_registry.register(
            kind="media",
            source_path=media_path,
            requested_id=file_id,
            replace_existing=replace_existing,
        )
        resolved_file_id = registration.doc_id
        transcriber = WhisperTranscriber(model_name=model_name, **dict(load_kwargs or {}))
        result = transcriber.transcribe(
            media_path,
            resolved_file_id,
            word_timestamps=word_timestamps,
            **dict(transcribe_kwargs or {}),
        )

        ingest_chunks: Sequence[ContentChunk]
        if chunk_size is not None:
            ingest_chunks = chunk_content_chunks(
                result.chunks,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            ingest_chunks = result.chunks

        ingested = self.ingest(ingest_chunks, replace_existing=replace_existing)
        if ingest_chunks is not result.chunks:
            result = TranscriptionResult(text=result.text, chunks=list(ingest_chunks), raw=result.raw)
        return MediaIngestionResult(
            file_id=resolved_file_id,
            ingested_chunks=ingested,
            transcription=result,
        )

    def batch_reindex(self, documents: Mapping[str, Sequence[ContentChunk]]) -> Dict[str, int]:
        results: Dict[str, int] = {}
        for source, doc_chunks in documents.items():
            logger.info(
                "Starting batch re-index for source=%s with %d chunks", source, len(doc_chunks)
            )
            self.vector_store.clear_source(source)
            total = 0
            chunk_seq = list(doc_chunks)
            for start in range(0, len(chunk_seq), self.config.reindex_batch_size):
                batch = chunk_seq[start : start + self.config.reindex_batch_size]
                stored = self.pipeline.ingest(batch)
                total += len(stored)
                logger.info(
                    "Re-indexed %d chunks for source=%s (batch %d-%d)",
                    len(stored),
                    source,
                    start,
                    start + len(batch) - 1,
                )

            self.metrics.ingested_documents += 1
            self.metrics.ingested_chunks += total
            results[source] = total
        return results

    def query(self, query: str, *, top_k: int | None = None) -> ResponsePayload:
        requested_top_k = top_k or self.config.default_top_k
        retrieved = self.retriever.retrieve(query, top_k=requested_top_k)
        answer = self.answer_synthesiser(retrieved)
        payload = self.response_composer.compose(answer, retrieved)
        self.metrics.queries += 1
        logger.info(
            "Executed query with top_k=%d and returned %d citations",
            requested_top_k,
            len(payload.citations),
        )
        return payload

    def metrics_snapshot(self) -> Dict[str, int]:
        return self.metrics.as_dict()


def _payload_to_chunk(payload: Mapping[str, object]) -> ContentChunk:
    if "text" not in payload:
        raise HTTPException(status_code=400, detail="chunk payload missing 'text'")
    text = str(payload["text"]).strip()
    if not text:
        raise HTTPException(status_code=400, detail="chunk text cannot be empty")

    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, Mapping):
        raise HTTPException(status_code=400, detail="chunk metadata must be an object")

    if "source" not in metadata_payload:
        raise HTTPException(status_code=400, detail="metadata must include 'source'")

    metadata = Metadata(
        source=str(metadata_payload["source"]),
        page=int(metadata_payload["page"]) if metadata_payload.get("page") is not None else None,
        figure=str(metadata_payload["figure"]) if metadata_payload.get("figure") is not None else None,
        timestamp_start=float(metadata_payload["timestamp_start"]) if metadata_payload.get("timestamp_start") is not None else None,
        timestamp_end=float(metadata_payload["timestamp_end"]) if metadata_payload.get("timestamp_end") is not None else None,
        asset_url=str(metadata_payload["asset_url"]) if metadata_payload.get("asset_url") is not None else None,
    )
    return ContentChunk(text=text, metadata=metadata)


def create_app(service: RAGService | None = None) -> FastAPI:
    """Create a FastAPI application exposing the service endpoints."""

    if service is None:
        raise ValueError("A RAGService instance must be provided to create the app.")

    app = FastAPI(title="TikTik RAG Service")

    if HAS_PYDANTIC:

        @app.post("/ingest", response_model=IngestResponse)
        def ingest(request: Dict[str, object] | IngestRequest) -> Dict[str, object]:
            payload = _parse_model(request, IngestRequest, context="ingest")
            chunks = [_chunk_payload_to_content(chunk) for chunk in payload.chunks]
            ingested = service.ingest(chunks, replace_existing=payload.replace_existing)
            response = IngestResponse(ingested=ingested, metrics=service.metrics_snapshot())
            return response.model_dump()

        @app.post("/reindex", response_model=ReindexResponse)
        def reindex(request: Dict[str, object] | ReindexRequest) -> Dict[str, object]:
            payload = _parse_model(request, ReindexRequest, context="reindex")
            mapping: Dict[str, Sequence[ContentChunk]] = {}
            for document in payload.documents:
                mapping[document.source] = [
                    _chunk_payload_to_content(chunk) for chunk in document.chunks
                ]
            results = service.batch_reindex(mapping)
            response = ReindexResponse(results=results, metrics=service.metrics_snapshot())
            return response.model_dump()

        @app.post("/query", response_model=QueryResponse)
        def query(request: Dict[str, object] | QueryRequest) -> Dict[str, object]:
            payload = _parse_model(request, QueryRequest, context="query")
            result = service.query(payload.query, top_k=payload.top_k)
            response = QueryResponse(
                answer=result.answer,
                citations=result.citations,
                assets=result.assets,
            )
            return response.model_dump()

        @app.post("/ingest/pdf", response_model=PDFIngestResponse)
        def ingest_pdf(request: Dict[str, object] | PDFIngestRequest) -> Dict[str, object]:
            payload = _parse_model(request, PDFIngestRequest, context="ingest/pdf")

            try:
                result = service.ingest_pdf_document(
                    payload.pdf_path,
                    payload.doc_id,
                    captions=payload.captions,
                    assets=payload.assets,
                    chunk_size=payload.chunk_size,
                    chunk_overlap=payload.chunk_overlap,
                    replace_existing=payload.replace_existing,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:  # pragma: no cover - dependency errors
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            response = PDFIngestResponse(
                doc_id=result.doc_id,
                ingested=result.ingested_chunks,
                captions=result.caption_index,
                metrics=service.metrics_snapshot(),
            )
            return response.model_dump()

        @app.post("/ingest/media", response_model=MediaIngestResponse)
        def ingest_media(request: Dict[str, object] | MediaIngestRequest) -> Dict[str, object]:
            payload = _parse_model(request, MediaIngestRequest, context="ingest/media")

            try:
                result = service.ingest_media_transcript(
                    payload.media_path,
                    payload.file_id,
                    model_name=payload.model_name,
                    word_timestamps=payload.word_timestamps,
                    replace_existing=payload.replace_existing,
                    chunk_size=payload.chunk_size,
                    chunk_overlap=payload.chunk_overlap,
                    transcribe_kwargs=dict(payload.transcribe_kwargs or {}),
                    load_kwargs=dict(payload.load_kwargs or {}),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:  # pragma: no cover - dependency errors
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            response = MediaIngestResponse(
                file_id=result.file_id,
                ingested=result.ingested_chunks,
                transcript=result.transcription.text,
                segments=[chunk.as_record() for chunk in result.transcription.chunks],
                metrics=service.metrics_snapshot(),
            )
            return response.model_dump()

    else:

        def _parse_nested_mapping(
            name: str, payload: object
        ) -> Mapping[int, Mapping[str, str]] | None:
            if payload is None:
                return None
            if not isinstance(payload, Mapping):
                raise HTTPException(
                    status_code=400, detail=f"{name} must be an object keyed by page"
                )
            normalised: Dict[int, Dict[str, str]] = {}
            for page_key, figure_map in payload.items():
                try:
                    page = int(page_key)
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400, detail=f"{name} keys must be integers"
                    ) from None
                if not isinstance(figure_map, Mapping):
                    raise HTTPException(
                        status_code=400,
                        detail=f"{name}[{page}] must be an object of figure -> value",
                    )
                normalised[page] = {
                    str(figure): str(value) for figure, value in figure_map.items()
                }
            return normalised

        @app.post("/ingest")
        def ingest(request: Dict[str, object]) -> Dict[str, object]:
            raw_chunks = request.get("chunks")
            if not isinstance(raw_chunks, list):
                raise HTTPException(status_code=400, detail="'chunks' must be a list")
            replace_existing = bool(request.get("replace_existing", False))
            chunks = [_payload_to_chunk(chunk) for chunk in raw_chunks]
            ingested = service.ingest(chunks, replace_existing=replace_existing)
            return {"ingested": ingested, "metrics": service.metrics_snapshot()}

        @app.post("/reindex")
        def reindex(request: Dict[str, object]) -> Dict[str, object]:
            documents = request.get("documents")
            if not isinstance(documents, list) or not documents:
                raise HTTPException(
                    status_code=400, detail="No documents supplied for re-index"
                )
            mapping: Dict[str, Sequence[ContentChunk]] = {}
            for document in documents:
                if not isinstance(document, Mapping):
                    raise HTTPException(
                        status_code=400, detail="document entries must be objects"
                    )
                source = document.get("source")
                if not source:
                    raise HTTPException(
                        status_code=400, detail="document missing 'source'"
                    )
                raw_chunks = document.get("chunks")
                if not isinstance(raw_chunks, list):
                    raise HTTPException(
                        status_code=400, detail="document chunks must be a list"
                    )
                chunk_objects = [_payload_to_chunk(chunk) for chunk in raw_chunks]
                mapping[str(source)] = chunk_objects
            results = service.batch_reindex(mapping)
            return {"results": results, "metrics": service.metrics_snapshot()}

        @app.post("/query")
        def query(request: Dict[str, object]) -> Dict[str, object]:
            query_text = request.get("query")
            if not isinstance(query_text, str) or not query_text.strip():
                raise HTTPException(
                    status_code=400, detail="query must be a non-empty string"
                )
            top_k = request.get("top_k")
            if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
                raise HTTPException(
                    status_code=400, detail="top_k must be a positive integer"
                )
            payload = service.query(
                query_text, top_k=top_k if isinstance(top_k, int) else None
            )
            return {
                "answer": payload.answer,
                "citations": payload.citations,
                "assets": payload.assets,
            }

        @app.post("/ingest/pdf")
        def ingest_pdf(request: Dict[str, object]) -> Dict[str, object]:
            pdf_path = request.get("pdf_path")
            doc_id_raw = request.get("doc_id")
            if not isinstance(pdf_path, str) or not pdf_path.strip():
                raise HTTPException(
                    status_code=400, detail="pdf_path must be a non-empty string"
                )
            if doc_id_raw is None:
                doc_id = None
            elif isinstance(doc_id_raw, str) and doc_id_raw.strip():
                doc_id = doc_id_raw
            else:
                raise HTTPException(
                    status_code=400,
                    detail="doc_id must be a non-empty string if provided",
                )
            captions = _parse_nested_mapping("captions", request.get("captions"))
            assets = _parse_nested_mapping("assets", request.get("assets"))
            chunk_size = int(request.get("chunk_size", 1000) or 1000)
            chunk_overlap = int(request.get("chunk_overlap", 200) or 0)
            replace_existing = bool(request.get("replace_existing", False))

            try:
                result = service.ingest_pdf_document(
                    pdf_path,
                    doc_id,
                    captions=captions,
                    assets=assets,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    replace_existing=replace_existing,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:  # pragma: no cover - dependency errors
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            return {
                "doc_id": result.doc_id,
                "ingested": result.ingested_chunks,
                "captions": result.caption_index,
                "metrics": service.metrics_snapshot(),
            }

        @app.post("/ingest/media")
        def ingest_media(request: Dict[str, object]) -> Dict[str, object]:
            media_path = request.get("media_path")
            file_id_raw = request.get("file_id")
            if not isinstance(media_path, str) or not media_path.strip():
                raise HTTPException(
                    status_code=400, detail="media_path must be a non-empty string"
                )
            if file_id_raw is None:
                file_id = None
            elif isinstance(file_id_raw, str) and file_id_raw.strip():
                file_id = file_id_raw
            else:
                raise HTTPException(
                    status_code=400,
                    detail="file_id must be a non-empty string if provided",
                )

            model_name = str(request.get("model_name", "base"))
            word_timestamps = bool(request.get("word_timestamps", False))
            replace_existing = bool(request.get("replace_existing", False))
            chunk_size_raw = request.get("chunk_size", 1000)
            chunk_size = int(chunk_size_raw) if chunk_size_raw is not None else None
            chunk_overlap = int(request.get("chunk_overlap", 200) or 0)

            transcribe_kwargs = request.get("transcribe_kwargs") or {}
            load_kwargs = request.get("load_kwargs") or {}
            if not isinstance(transcribe_kwargs, Mapping):
                raise HTTPException(
                    status_code=400,
                    detail="transcribe_kwargs must be an object if provided",
                )
            if not isinstance(load_kwargs, Mapping):
                raise HTTPException(
                    status_code=400,
                    detail="load_kwargs must be an object if provided",
                )

            try:
                result = service.ingest_media_transcript(
                    media_path,
                    file_id,
                    model_name=model_name,
                    word_timestamps=word_timestamps,
                    replace_existing=replace_existing,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    transcribe_kwargs=transcribe_kwargs,
                    load_kwargs=load_kwargs,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:  # pragma: no cover - dependency errors
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            return {
                "file_id": result.file_id,
                "ingested": result.ingested_chunks,
                "transcript": result.transcription.text,
                "segments": [chunk.as_record() for chunk in result.transcription.chunks],
                "metrics": service.metrics_snapshot(),
            }

    @app.get("/metrics")
    def metrics() -> Dict[str, int]:
        return service.metrics_snapshot()

    return app


__all__ = [
    "DocumentIngestionResult",
    "MediaIngestionResult",
    "RAGService",
    "RAGServiceConfig",
    "create_app",
    "ServiceMetrics",
]


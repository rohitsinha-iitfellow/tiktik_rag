"""FastAPI service exposing TikTik RAG ingestion and query flows."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

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

from .embedding import ChunkEmbeddingPipeline, EmbeddingModel, StoredChunk
from .metadata import ContentChunk, Metadata
from .response import ResponseComposer, ResponsePayload
from .retrieval import ChunkRetriever, CrossEncoder, RetrievedChunk, SimilaritySearcher


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


class InMemoryVectorStore:
    """In-memory persistence backing the default service implementation."""

    def __init__(self) -> None:
        self.records: List[StoredChunk] = []

    def add(self, *, embeddings, metadatas, documents) -> None:  # type: ignore[override]
        for text, metadata, embedding in zip(documents, metadatas, embeddings):
            self.records.append(StoredChunk(text=text, metadata=dict(metadata), embedding=list(embedding)))

    def clear_source(self, source: str) -> None:
        self.records = [record for record in self.records if record.metadata.get("source") != source]


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
            raise HTTPException(status_code=400, detail="No documents supplied for re-index")
        mapping: Dict[str, Sequence[ContentChunk]] = {}
        for document in documents:
            if not isinstance(document, Mapping):
                raise HTTPException(status_code=400, detail="document entries must be objects")
            source = document.get("source")
            if not source:
                raise HTTPException(status_code=400, detail="document missing 'source'")
            raw_chunks = document.get("chunks")
            if not isinstance(raw_chunks, list):
                raise HTTPException(status_code=400, detail="document chunks must be a list")
            chunk_objects = [_payload_to_chunk(chunk) for chunk in raw_chunks]
            mapping[str(source)] = chunk_objects
        results = service.batch_reindex(mapping)
        return {"results": results, "metrics": service.metrics_snapshot()}

    @app.post("/query")
    def query(request: Dict[str, object]) -> Dict[str, object]:
        query_text = request.get("query")
        if not isinstance(query_text, str) or not query_text.strip():
            raise HTTPException(status_code=400, detail="query must be a non-empty string")
        top_k = request.get("top_k")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise HTTPException(status_code=400, detail="top_k must be a positive integer")
        payload = service.query(query_text, top_k=top_k if isinstance(top_k, int) else None)
        return {"answer": payload.answer, "citations": payload.citations, "assets": payload.assets}

    @app.get("/metrics")
    def metrics() -> Dict[str, int]:
        return service.metrics_snapshot()

    return app


__all__ = [
    "RAGService",
    "RAGServiceConfig",
    "create_app",
    "ServiceMetrics",
]


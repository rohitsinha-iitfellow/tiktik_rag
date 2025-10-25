from __future__ import annotations

from typing import Dict, List, Sequence

from tiktik_rag.embedding import StoredChunk
from tiktik_rag.media_transcriber import TranscriptionResult
from tiktik_rag.metadata import ContentChunk, Metadata
from tiktik_rag.service import (
    DocumentIngestionResult,
    MediaIngestionResult,
    RAGService,
    RAGServiceConfig,
    create_app,
)


class SimpleResponse:
    def __init__(self, status_code: int, data: Dict[str, object]) -> None:
        self.status_code = status_code
        self._data = data

    def json(self) -> Dict[str, object]:
        return self._data


class SimpleTestClient:
    def __init__(self, app) -> None:
        self.app = app

    def _resolve(self, method: str, path: str):
        routes = getattr(self.app, "routes", {})
        if isinstance(routes, dict):
            return routes.get((method, path))
        raise RuntimeError("Unsupported app routes structure for SimpleTestClient")

    def post(self, path: str, json: Dict[str, object] | None = None) -> SimpleResponse:
        handler = self._resolve("POST", path)
        assert handler is not None, f"No handler registered for POST {path}"
        try:
            data = handler(json or {})
            return SimpleResponse(200, data)
        except Exception as exc:  # pragma: no cover - error path
            status_code = getattr(exc, "status_code", 500)
            detail = getattr(exc, "detail", str(exc))
            return SimpleResponse(status_code, {"detail": detail})

    def get(self, path: str) -> SimpleResponse:
        handler = self._resolve("GET", path)
        assert handler is not None, f"No handler registered for GET {path}"
        try:
            data = handler()
            return SimpleResponse(200, data)
        except Exception as exc:  # pragma: no cover - error path
            status_code = getattr(exc, "status_code", 500)
            detail = getattr(exc, "detail", str(exc))
            return SimpleResponse(status_code, {"detail": detail})


class KeywordEmbeddingModel:
    """Toy embedding model counting keyword occurrences."""

    def __init__(self) -> None:
        self.calls: List[Sequence[str]] = []
        self.vocabulary = ("alpha", "beta", "gamma")

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(texts)
        vectors: List[List[float]] = []
        for text in texts:
            lowered = text.lower()
            vector = [float(lowered.count(token)) for token in self.vocabulary]
            if not any(vector):
                vector[-1] = float(len(text)) or 1.0
            vectors.append(vector)
        return vectors


def _make_chunk(text: str, **metadata) -> Dict[str, object]:
    payload = {
        "text": text,
        "metadata": {
            "source": metadata.get("source", "doc"),
            "page": metadata.get("page"),
            "figure": metadata.get("figure"),
            "timestamp_start": metadata.get("timestamp_start"),
            "timestamp_end": metadata.get("timestamp_end"),
            "asset_url": metadata.get("asset_url"),
        },
    }
    return payload


def test_service_ingest_and_query_returns_citations():
    model = KeywordEmbeddingModel()
    service = RAGService(model, config=RAGServiceConfig(default_top_k=2))
    app = create_app(service)
    client = SimpleTestClient(app)

    response = client.post(
        "/ingest",
        json={
            "replace_existing": True,
            "chunks": [
                _make_chunk(
                    "Alpha particles referenced", source="doc-alpha", page=3, figure="A1", timestamp_start=0.0, timestamp_end=5.0
                ),
                _make_chunk("Beta coverage for comparison", source="doc-beta", page=1),
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["ingested"] == 2

    query_response = client.post("/query", json={"query": "alpha insights", "top_k": 2})
    assert query_response.status_code == 200
    payload = query_response.json()
    assert "doc-alpha" in payload["citations"][0]
    assert "p. 3" in payload["citations"][0]
    assert "fig. A1" in payload["citations"][0]
    assert "00:00-00:05" in payload["citations"][0]

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["ingested_chunks"] == 2
    assert metrics["queries"] == 1


def test_batch_reindex_honours_configuration_batches():
    model = KeywordEmbeddingModel()
    config = RAGServiceConfig(reindex_batch_size=2, embedding_batch_size=4)
    service = RAGService(model, config=config)

    class SpyPipeline:
        def __init__(self) -> None:
            self.calls: List[List[str]] = []

        def ingest(self, chunks: Sequence[ContentChunk]) -> List[StoredChunk]:
            self.calls.append([chunk.text for chunk in chunks])
            embeddings = model.embed([chunk.text for chunk in chunks])
            metadatas = [chunk.metadata.as_dict() for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            service.vector_store.add(embeddings=embeddings, metadatas=metadatas, documents=documents)
            return [StoredChunk(text=text, metadata=metadata, embedding=embedding) for text, metadata, embedding in zip(documents, metadatas, embeddings)]

    spy_pipeline = SpyPipeline()
    service.pipeline = spy_pipeline  # type: ignore[assignment]

    chunks = [
        ContentChunk(text=f"Alpha chunk {index}", metadata=Metadata(source="doc", page=index))
        for index in range(1, 5)
    ]

    results = service.batch_reindex({"doc": chunks})

    assert results == {"doc": 4}
    assert spy_pipeline.calls == [
        ["Alpha chunk 1", "Alpha chunk 2"],
        ["Alpha chunk 3", "Alpha chunk 4"],
    ]
    assert service.metrics.ingested_documents == 1
    assert service.metrics.ingested_chunks == 4


def test_ingest_pdf_endpoint_invokes_service():
    model = KeywordEmbeddingModel()
    service = RAGService(model)
    app = create_app(service)
    client = SimpleTestClient(app)

    captured = {}

    def fake_ingest_pdf_document(pdf_path, doc_id, **kwargs):
        captured["pdf_path"] = pdf_path
        captured["doc_id"] = doc_id
        captured["kwargs"] = kwargs
        return DocumentIngestionResult(doc_id="doc-123", ingested_chunks=7, caption_index={1: {"A": "Caption text"}})

    service.ingest_pdf_document = fake_ingest_pdf_document  # type: ignore[assignment]

    response = client.post(
        "/ingest/pdf",
        json={
            "pdf_path": "/tmp/doc.pdf",
            "doc_id": "doc-123",
            "captions": {"1": {"A": "Caption text"}},
            "assets": {1: {"A": "http://example/fig"}},
            "chunk_size": 500,
            "chunk_overlap": 50,
            "replace_existing": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_id"] == "doc-123"
    assert payload["ingested"] == 7
    assert payload["captions"] == {1: {"A": "Caption text"}}
    assert captured["pdf_path"] == "/tmp/doc.pdf"
    assert captured["doc_id"] == "doc-123"
    assert captured["kwargs"]["captions"] == {1: {"A": "Caption text"}}
    assert captured["kwargs"]["assets"] == {1: {"A": "http://example/fig"}}
    assert captured["kwargs"]["chunk_size"] == 500
    assert captured["kwargs"]["chunk_overlap"] == 50
    assert captured["kwargs"]["replace_existing"] is True


def test_ingest_pdf_endpoint_generates_identifier_when_missing():
    model = KeywordEmbeddingModel()
    service = RAGService(model)
    app = create_app(service)
    client = SimpleTestClient(app)

    captured: Dict[str, object] = {}

    def fake_ingest_pdf_document(pdf_path, doc_id, **kwargs):
        captured["doc_id"] = doc_id
        return DocumentIngestionResult(doc_id="auto-generated", ingested_chunks=1, caption_index={})

    service.ingest_pdf_document = fake_ingest_pdf_document  # type: ignore[assignment]

    response = client.post("/ingest/pdf", json={"pdf_path": "/tmp/doc.pdf"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_id"] == "auto-generated"
    assert payload["ingested"] == 1
    assert payload["captions"] == {}
    assert captured["doc_id"] is None


def test_ingest_media_endpoint_invokes_service():
    model = KeywordEmbeddingModel()
    service = RAGService(model)
    app = create_app(service)
    client = SimpleTestClient(app)

    segments = [
        ContentChunk(text="hello", metadata=Metadata(source="file", timestamp_start=0.0, timestamp_end=1.0))
    ]
    transcript_result = TranscriptionResult(text="hello", chunks=segments, raw={})

    captured = {}

    def fake_ingest_media_transcript(media_path, file_id, **kwargs):
        captured["media_path"] = media_path
        captured["file_id"] = file_id
        captured["kwargs"] = kwargs
        return MediaIngestionResult(
            file_id="audio-1",
            ingested_chunks=3,
            transcription=transcript_result,
        )

    service.ingest_media_transcript = fake_ingest_media_transcript  # type: ignore[assignment]

    response = client.post(
        "/ingest/media",
        json={
            "media_path": "/tmp/audio.mp3",
            "file_id": "audio-1",
            "model_name": "base",
            "word_timestamps": True,
            "replace_existing": True,
            "chunk_size": None,
            "chunk_overlap": 0,
            "transcribe_kwargs": {"temperature": 0.2},
            "load_kwargs": {"device": "cpu"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_id"] == "audio-1"
    assert payload["ingested"] == 3
    assert payload["transcript"] == "hello"
    assert len(payload["segments"]) == 1
    assert payload["segments"][0]["metadata"]["timestamp_end"] == 1.0
    assert captured["media_path"] == "/tmp/audio.mp3"
    assert captured["file_id"] == "audio-1"
    assert captured["kwargs"]["model_name"] == "base"
    assert captured["kwargs"]["word_timestamps"] is True
    assert captured["kwargs"]["replace_existing"] is True
    assert captured["kwargs"]["chunk_size"] is None
    assert captured["kwargs"]["transcribe_kwargs"] == {"temperature": 0.2}
    assert captured["kwargs"]["load_kwargs"] == {"device": "cpu"}


def test_ingest_media_endpoint_generates_identifier_when_missing():
    model = KeywordEmbeddingModel()
    service = RAGService(model)
    app = create_app(service)
    client = SimpleTestClient(app)

    captured: Dict[str, object] = {}

    def fake_ingest_media_transcript(media_path, file_id, **kwargs):
        captured["file_id"] = file_id
        return MediaIngestionResult(
            file_id="auto-file",
            ingested_chunks=2,
            transcription=TranscriptionResult(text="hi", chunks=[], raw={}),
        )

    service.ingest_media_transcript = fake_ingest_media_transcript  # type: ignore[assignment]

    response = client.post(
        "/ingest/media",
        json={
            "media_path": "/tmp/video.mp4",
            "model_name": "base",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_id"] == "auto-file"
    assert payload["ingested"] == 2
    assert payload["transcript"] == "hi"
    assert payload["segments"] == []
    assert captured["file_id"] is None


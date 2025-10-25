# TikTik RAG ingestion utilities

This repository provides ingestion helpers that prepare rich metadata for retrieval augmented generation pipelines.

## Features

- **PDF ingestion** using `PDFLoader`, extracting page-level text and mapping supplied image captions to composite keys of `{doc_id, page, figure_id}`.
- **Audio and video ingestion** via `WhisperTranscriber`, producing segment- or word-level transcripts that carry timestamp metadata.
- **Open-weight embeddings** powered by Hugging Face's `sentence-transformers` for completely self-hosted retrieval pipelines.
- **FastAPI service** exposing ingestion endpoints for PDFs, captions, images, audio, and video assets, returning citation-ready metadata.

## Development

Install optional dependencies when you plan to parse PDFs, run Whisper locally, or use the bundled Hugging Face embedding model:

```bash
pip install -e .[ingestion,embeddings]
```

## Running the API service

Launch the FastAPI app configured with the default open-weights embedding model (`sentence-transformers/all-MiniLM-L6-v2`) using Uvicorn:

```bash
uvicorn tiktik_rag.api:app --host 0.0.0.0 --port 8000
```

Set `TIKTIK_RAG_EMBEDDING_MODEL` to override the Hugging Face model name. Once running you can:

- **Ingest PDFs and captions** by POSTing to `/ingest/pdf` with the file path, document identifier, optional caption text, and optional asset URLs to return alongside answers.
- **Ingest audio or video** by POSTing to `/ingest/media`, which will run Whisper transcription (with optional word-level timestamps) before chunking and indexing the transcript.
- **Query** the knowledge base via `/query` to retrieve answers enriched with document/page references, figure identifiers, asset URLs, or timestamp ranges.

Run the test suite with:

```bash
pytest
```

## API Endpoints

All endpoints expect and return JSON payloads. Unless otherwise noted, responses include a `metrics` object with running totals of documents, chunks, and queries processed by the service.

### `POST /ingest`

Bulk-ingest pre-chunked content. Use this when you have already prepared `text` + `metadata` pairs.

**Request body**

```json
{
  "chunks": [
    {
      "text": "Chunk body text",
      "metadata": {
        "source": "dataset-name",
        "page": 3,
        "figure": "2A",
        "timestamp_start": 12.5,
        "timestamp_end": 24.0,
        "asset_url": "https://example.com/fig-2A.png"
      }
    }
  ],
  "replace_existing": false
}
```

**Response body**

```json
{
  "ingested": 1,
  "metrics": {"ingested_documents": 0, "ingested_chunks": 1, "queries": 0}
}
```

### `POST /reindex`

Rebuild stored embeddings for one or more sources. Provide the full set of chunks per `source`.

**Request body**

```json
{
  "documents": [
    {
      "source": "dataset-name",
      "chunks": [
        {"text": "Chunk body text", "metadata": {"source": "dataset-name"}}
      ]
    }
  ]
}
```

**Response body**

```json
{
  "results": {"dataset-name": 1},
  "metrics": {"ingested_documents": 0, "ingested_chunks": 1, "queries": 0}
}
```

### `POST /query`

Retrieve an answer and contextual citations.

**Request body**

```json
{
  "query": "What does the paper say about evaluation?",
  "top_k": 5
}
```

**Response body**

```json
{
  "answer": "Summary generated from retrieved chunks...",
  "citations": [
    {
      "text": "Supporting chunk text",
      "metadata": {
        "source": "dataset-name",
        "page": 3,
        "figure": "2A",
        "timestamp_start": 12.5,
        "timestamp_end": 24.0,
        "asset_url": "https://example.com/fig-2A.png"
      }
    }
  ],
  "assets": [
    {"source": "dataset-name", "page": 3, "figure": "2A", "asset_url": "https://example.com/fig-2A.png"}
  ]
}
```

### `POST /ingest/pdf`

Parse a PDF on disk, chunk its content, and index it. Optionally attach caption or asset metadata keyed by page and figure.

**Request body**

```json
{
  "pdf_path": "/path/to/paper.pdf",
  "doc_id": "paper-2024",
  "captions": {"1": {"A": "Figure caption"}},
  "assets": {"1": {"A": "https://example.com/figure-a.png"}},
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "replace_existing": false
}
```

**Response body**

```json
{
  "doc_id": "paper-2024",
  "ingested": 42,
  "captions": {"1": {"A": "Figure caption"}},
  "metrics": {"ingested_documents": 1, "ingested_chunks": 42, "queries": 0}
}
```

### `POST /ingest/media`

Transcribe an audio or video file with Whisper, chunk the transcript, and index it.

**Request body**

```json
{
  "media_path": "/path/to/podcast.mp3",
  "file_id": "podcast-episode",
  "model_name": "base",
  "word_timestamps": false,
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "replace_existing": false,
  "transcribe_kwargs": {"language": "en"},
  "load_kwargs": {}
}
```

**Response body**

```json
{
  "file_id": "podcast-episode",
  "ingested": 64,
  "transcript": "Full transcript text...",
  "segments": [
    {"text": "Segment text", "metadata": {"source": "podcast-episode", "timestamp_start": 0.0}}
  ],
  "metrics": {"ingested_documents": 1, "ingested_chunks": 64, "queries": 0}
}
```

### `GET /metrics`

Return cumulative service metrics.

**Response body**

```json
{
  "ingested_documents": 2,
  "ingested_chunks": 107,
  "queries": 5
}
```

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

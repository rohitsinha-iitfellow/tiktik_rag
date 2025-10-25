# TikTik RAG ingestion utilities

This repository provides ingestion helpers that prepare rich metadata for retrieval augmented generation pipelines.

## Features

- **PDF ingestion** using `PDFLoader`, extracting page-level text and mapping supplied image captions to composite keys of `{doc_id, page, figure_id}`.
- **Audio and video ingestion** via `WhisperTranscriber`, producing segment- or word-level transcripts that carry timestamp metadata.
- **Shared data structures** for normalised metadata (`source`, `page`, `figure`, `timestamp_start`, `timestamp_end`) and text chunks.

## Development

Install optional dependencies when you plan to parse PDFs or run Whisper locally:

```bash
pip install -e .[ingestion]
```

Run the test suite with:

```bash
pytest
```

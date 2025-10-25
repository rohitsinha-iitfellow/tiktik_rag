"""Convenience module exposing a FastAPI app backed by Hugging Face embeddings."""
from __future__ import annotations

import os

from .hf_embedding import HuggingFaceEmbeddingModel
from .service import RAGService, create_app

DEFAULT_MODEL_NAME = os.getenv("TIKTIK_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def build_default_service() -> RAGService:
    """Construct a :class:`RAGService` configured with Hugging Face embeddings."""

    embedding_model = HuggingFaceEmbeddingModel(DEFAULT_MODEL_NAME)
    return RAGService(embedding_model)


service = build_default_service()
app = create_app(service)

__all__ = ["app", "service", "build_default_service", "DEFAULT_MODEL_NAME"]

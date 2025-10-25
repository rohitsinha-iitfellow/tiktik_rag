"""Hugging Face powered embedding model for TikTik RAG."""
from __future__ import annotations

import importlib
from typing import Sequence


class HuggingFaceEmbeddingModel:
    """Embedding model backed by Hugging Face sentence transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
        **model_kwargs,
    ) -> None:
        try:
            module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is required for HuggingFaceEmbeddingModel. "
                "Install it via `pip install sentence-transformers`."
            ) from exc

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        SentenceTransformer = getattr(module, "SentenceTransformer")
        self._model = SentenceTransformer(model_name, device=device, **model_kwargs)

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        encoded = self._model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self.batch_size or len(texts),
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return encoded.tolist()


__all__ = ["HuggingFaceEmbeddingModel"]

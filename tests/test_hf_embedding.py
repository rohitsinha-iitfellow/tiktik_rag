from __future__ import annotations

import sys
import types

import pytest


def test_hf_embedding_model_requires_dependency(monkeypatch):
    import importlib
    import tiktik_rag.hf_embedding as hf_embedding

    real_import = importlib.import_module

    def fake_import(name):  # pragma: no cover - helper for clarity
        if name == "sentence_transformers":
            raise ModuleNotFoundError
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        hf_embedding.HuggingFaceEmbeddingModel("dummy")
    assert "sentence-transformers" in str(excinfo.value)


def test_hf_embedding_model_encodes_using_sentence_transformers(monkeypatch):
    dummy_module = types.SimpleNamespace()

    class DummyArray:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    class DummyModel:
        def __init__(self, model_name: str, device: str | None = None, **kwargs):
            self.model_name = model_name
            self.device = device
            self.kwargs = kwargs

        def encode(self, texts, batch_size=None, normalize_embeddings=True, convert_to_numpy=True):
            assert batch_size == 2
            assert normalize_embeddings is True
            assert convert_to_numpy is True
            return DummyArray([[float(len(text))] for text in texts])

    dummy_module.SentenceTransformer = DummyModel
    import importlib
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_module)
    import tiktik_rag.hf_embedding as hf_embedding
    importlib.reload(hf_embedding)

    model = hf_embedding.HuggingFaceEmbeddingModel(batch_size=2)
    vectors = model.embed(["alpha", "beta"])
    assert vectors == [[5.0], [4.0]]


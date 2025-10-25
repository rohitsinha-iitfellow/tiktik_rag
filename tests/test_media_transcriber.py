import types

import pytest

import tiktik_rag.media_transcriber as media_transcriber


class DummyModel:
    def __init__(self, result):
        self._result = result
        self.transcribe_calls = []

    def transcribe(self, path: str, **kwargs):
        self.transcribe_calls.append((path, kwargs))
        return self._result


@pytest.fixture(autouse=True)
def patch_whisper(monkeypatch):
    dummy_namespace = types.SimpleNamespace()

    def load_model(_name: str, **_kwargs):
        return DummyModel(result={})

    dummy_namespace.load_model = load_model
    monkeypatch.setattr(media_transcriber, "whisper", dummy_namespace)
    yield


def test_transcriber_segment_level(monkeypatch):
    result = {
        "text": " Hello world ",
        "segments": [
            {"text": " Hello ", "start": 0.0, "end": 1.0},
            {"text": " world ", "start": 1.0, "end": 2.0},
        ],
    }

    model = DummyModel(result=result)
    monkeypatch.setattr(media_transcriber, "whisper", types.SimpleNamespace(load_model=lambda *_, **__: model))

    transcriber = media_transcriber.WhisperTranscriber()
    output = transcriber.transcribe("file.wav", file_id="file-1")

    assert output.text == "Hello world"
    assert [chunk.text for chunk in output.chunks] == ["Hello", "world"]
    assert [chunk.metadata.timestamp_start for chunk in output.chunks] == [0.0, 1.0]
    assert [chunk.metadata.timestamp_end for chunk in output.chunks] == [1.0, 2.0]
    assert all(chunk.metadata.source == "file-1" for chunk in output.chunks)


def test_transcriber_word_level(monkeypatch):
    result = {
        "text": " Testing one two ",
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "words": [
                    {"word": " Testing", "start": 0.0, "end": 0.5},
                    {"word": " one", "start": 0.5, "end": 1.0},
                    {"word": " two", "start": 1.0, "end": 1.5},
                ],
            }
        ],
    }
    model = DummyModel(result=result)
    monkeypatch.setattr(media_transcriber, "whisper", types.SimpleNamespace(load_model=lambda *_, **__: model))

    transcriber = media_transcriber.WhisperTranscriber()
    output = transcriber.transcribe("file.wav", file_id="file-2", word_timestamps=True)

    assert [chunk.text for chunk in output.chunks] == ["Testing", "one", "two"]
    assert [chunk.metadata.timestamp_start for chunk in output.chunks] == [0.0, 0.5, 1.0]
    assert [chunk.metadata.timestamp_end for chunk in output.chunks] == [0.5, 1.0, 1.5]

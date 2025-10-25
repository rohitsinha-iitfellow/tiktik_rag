"""Audio and video ingestion via Whisper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .metadata import ContentChunk, Metadata

try:  # pragma: no cover - optional dependency
    import whisper  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    whisper = None


@dataclass
class TranscriptionResult:
    """Container for the Whisper transcription output."""

    text: str
    chunks: List[ContentChunk]
    raw: Dict[str, Any]


class WhisperTranscriber:
    """Generate transcripts with time-aligned metadata using Whisper."""

    def __init__(self, model_name: str = "base", **load_kwargs: Any) -> None:
        if whisper is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai-whisper is required for transcription. Install it via `pip install openai-whisper`."
            )
        self.model = whisper.load_model(model_name, **load_kwargs)

    def transcribe(
        self,
        media_path: Path | str,
        file_id: str,
        word_timestamps: bool = False,
        **transcribe_kwargs: Any,
    ) -> TranscriptionResult:
        """Run Whisper transcription and return structured chunks."""

        path = Path(media_path)
        if word_timestamps:
            transcribe_kwargs.setdefault("word_timestamps", True)
        result: Dict[str, Any] = self.model.transcribe(str(path), **transcribe_kwargs)
        segments: Iterable[Dict[str, Any]] = result.get("segments", [])

        chunks: List[ContentChunk] = []
        if word_timestamps:
            for segment in segments:
                words: Iterable[Dict[str, Any]] = segment.get("words", [])
                for word in words:
                    chunks.append(
                        ContentChunk(
                            text=str(word.get("word", "")).strip(),
                            metadata=Metadata(
                                source=file_id,
                                timestamp_start=word.get("start"),
                                timestamp_end=word.get("end"),
                            ),
                        )
                    )
        else:
            for segment in segments:
                chunks.append(
                    ContentChunk(
                        text=str(segment.get("text", "")).strip(),
                        metadata=Metadata(
                            source=file_id,
                            timestamp_start=segment.get("start"),
                            timestamp_end=segment.get("end"),
                        ),
                    )
                )

        text = str(result.get("text", "")).strip()
        return TranscriptionResult(text=text, chunks=chunks, raw=result)

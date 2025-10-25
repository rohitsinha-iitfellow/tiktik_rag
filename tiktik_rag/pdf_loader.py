"""Utilities for extracting structured data from PDF documents."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from .metadata import Caption, CaptionKey, ContentChunk, Metadata

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None


class PDFLoader:
    """Extract text and caption metadata from PDF documents."""

    def __init__(
        self,
        pdf_path: Path | str,
        doc_id: str,
        captions: Iterable[Caption] | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.doc_id = doc_id
        self._captions = list(captions or [])

    @staticmethod
    def build_captions_from_dict(
        doc_id: str,
        caption_payload: Mapping[int, Mapping[str, str]],
    ) -> List[Caption]:
        """Normalise nested caption dictionaries into :class:`Caption` objects.

        Parameters
        ----------
        doc_id:
            Identifier associated with the source PDF document.
        caption_payload:
            Mapping of page numbers to dictionaries of ``figure_id`` -> ``caption``.
        """

        captions: List[Caption] = []
        for page, figure_map in caption_payload.items():
            for figure_id, text in figure_map.items():
                key = CaptionKey(doc_id=doc_id, page=page, figure_id=str(figure_id))
                captions.append(Caption(key=key, text=text))
        return captions

    def _ensure_reader(self) -> PdfReader:
        if PdfReader is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pypdf is required to read PDF files. Install it via `pip install pypdf`."
            )
        return PdfReader(str(self.pdf_path))

    def load(self) -> Tuple[List[ContentChunk], Dict[Tuple[str, int, str], Caption]]:
        """Read the PDF document and return text chunks and caption mappings."""

        reader = self._ensure_reader()
        chunks: List[ContentChunk] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            metadata = Metadata(source=self.doc_id, page=index)
            chunks.append(ContentChunk(text=text, metadata=metadata))

        caption_map: Dict[Tuple[str, int, str], Caption] = {
            caption.key.as_tuple(): caption for caption in self._captions
        }
        return chunks, caption_map

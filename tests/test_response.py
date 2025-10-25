from __future__ import annotations

from tiktik_rag.metadata import ContentChunk, Metadata
from tiktik_rag.response import ResponseComposer
from tiktik_rag.retrieval import RetrievedChunk


def test_response_composer_formats_citations_and_assets() -> None:
    chunks = [
        RetrievedChunk(
            chunk=ContentChunk(
                text="Figure caption",
                metadata=Metadata(
                    source="design.pdf",
                    page=4,
                    figure="2",
                    asset_url="https://example.com/fig2.png",
                ),
            ),
            score=0.9,
        ),
        RetrievedChunk(
            chunk=ContentChunk(
                text="Audio transcript",
                metadata=Metadata(
                    source="talk.mp3",
                    timestamp_start=5,
                    timestamp_end=17,
                ),
            ),
            score=0.8,
        ),
    ]

    composer = ResponseComposer()
    payload = composer.compose("Answer", chunks)

    assert payload.answer == "Answer"
    assert "design.pdf p. 4 fig. 2" in payload.citations[0]
    assert any("00:05-00:17" in citation for citation in payload.citations)
    assert payload.assets == ["https://example.com/fig2.png"]

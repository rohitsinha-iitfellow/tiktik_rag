import types

import pytest

from tiktik_rag.metadata import Caption, CaptionKey
from tiktik_rag.pdf_loader import PDFLoader


class DummyPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class DummyReader:
    def __init__(self, _path: str) -> None:
        self.pages = [DummyPage("First page"), DummyPage("Second page")]


@pytest.fixture(autouse=True)
def patch_pypdf(monkeypatch):
    module = types.SimpleNamespace()
    module.PdfReader = DummyReader
    monkeypatch.setattr("tiktik_rag.pdf_loader.PdfReader", module.PdfReader)


def test_pdf_loader_extracts_text_and_captions():
    captions = [
        Caption(key=CaptionKey(doc_id="doc", page=1, figure_id="fig1"), text="Figure 1"),
        Caption(key=CaptionKey(doc_id="doc", page=2, figure_id="fig2"), text="Figure 2"),
    ]
    loader = PDFLoader(pdf_path="/tmp/file.pdf", doc_id="doc", captions=captions)

    chunks, caption_map = loader.load()

    assert [chunk.text for chunk in chunks] == ["First page", "Second page"]
    assert [chunk.metadata.page for chunk in chunks] == [1, 2]
    assert caption_map[("doc", 1, "fig1")].text == "Figure 1"
    assert caption_map[("doc", 2, "fig2")].text == "Figure 2"


def test_build_captions_from_dict():
    captions = PDFLoader.build_captions_from_dict(
        doc_id="doc",
        caption_payload={1: {"fig1": "Caption"}},
    )
    assert captions[0].key.as_tuple() == ("doc", 1, "fig1")
    assert captions[0].text == "Caption"

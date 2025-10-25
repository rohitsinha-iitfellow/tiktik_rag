from tiktik_rag.chunking import captions_to_chunks, chunk_content_chunks
from tiktik_rag.metadata import Caption, CaptionKey, ContentChunk, Metadata


def test_chunk_content_chunks_splits_long_text_with_overlap():
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
    metadata = Metadata(source="doc-1", page=1)
    chunk = ContentChunk(text=long_text, metadata=metadata)

    result = chunk_content_chunks([chunk], chunk_size=120, chunk_overlap=20)

    assert len(result) > 1
    assert all(len(piece.text) <= 120 for piece in result)
    assert all(piece.metadata.page == 1 for piece in result)
    assert result[0].text.startswith("Lorem ipsum")
    assert all(piece.text in long_text for piece in result)


def test_chunk_content_chunks_skips_empty_text():
    metadata = Metadata(source="doc-2")
    chunk = ContentChunk(text="   ", metadata=metadata)

    assert chunk_content_chunks([chunk]) == []


def test_captions_to_chunks_sets_metadata_and_asset_url():
    key = CaptionKey(doc_id="doc-3", page=5, figure_id="fig-7")
    caption = Caption(key=key, text="A detailed description")
    asset_lookup = {key.as_tuple(): "s3://assets/fig-7.png"}

    result = captions_to_chunks([caption], asset_lookup=asset_lookup)

    assert len(result) == 1
    chunk = result[0]
    assert chunk.text == "A detailed description"
    assert chunk.metadata.figure == "fig-7"
    assert chunk.metadata.asset_url == "s3://assets/fig-7.png"

from tiktik_rag.metadata import Caption, CaptionKey, ContentChunk, Metadata


def test_metadata_as_dict_filters_none():
    metadata = Metadata(source="doc-123", page=2, figure=None, timestamp_start=None, timestamp_end=3.5)
    assert metadata.as_dict() == {
        "source": "doc-123",
        "page": 2,
        "timestamp_end": 3.5,
    }


def test_metadata_includes_asset_url_when_present():
    metadata = Metadata(source="doc-789", asset_url="s3://bucket/image.png")
    assert metadata.as_dict() == {
        "source": "doc-789",
        "asset_url": "s3://bucket/image.png",
    }


def test_content_chunk_as_record():
    metadata = Metadata(source="doc-456", page=1)
    chunk = ContentChunk(text="Hello", metadata=metadata)
    assert chunk.as_record() == {
        "text": "Hello",
        "metadata": {"source": "doc-456", "page": 1},
    }


def test_caption_key_tuple():
    key = CaptionKey(doc_id="doc", page=1, figure_id="fig-1")
    caption = Caption(key=key, text="A caption")
    assert key.as_tuple() == ("doc", 1, "fig-1")
    assert caption.key is key

from src.ingestion import build_chunks, load_documents


def test_no_chunk_is_only_a_source_id_line():
    chunks = build_chunks(load_documents())
    assert chunks, "expected at least one chunk"
    assert not any("Source ID:" in chunk.text for chunk in chunks)


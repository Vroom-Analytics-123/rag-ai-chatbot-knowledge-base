"""Tests for rag.chunker — overlapping word-window chunking with metadata."""

from rag.chunker import chunk_text

LONG_TEXT = " ".join(f"word{i}" for i in range(200))


def test_chunks_carry_doc_and_chunk_metadata():
    chunks = chunk_text("hours", "Hours & Holidays", LONG_TEXT, chunk_size=60, overlap=15)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c["doc"] == "hours"
        assert c["title"] == "Hours & Holidays"
        assert c["chunk"] == i
        assert c["text"].strip() != ""


def test_overlap_is_exact():
    # chunk i+1 must begin with the last `overlap` words of chunk i
    overlap = 15
    chunks = chunk_text("d", "T", LONG_TEXT, chunk_size=60, overlap=overlap)
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_words = prev["text"].split()
        next_words = nxt["text"].split()
        assert next_words[:overlap] == prev_words[-overlap:], (
            f"chunk {nxt['chunk']} does not overlap chunk {prev['chunk']}"
        )


def test_chunk_size_respected():
    chunks = chunk_text("d", "T", LONG_TEXT, chunk_size=60, overlap=15)
    for c in chunks:
        assert len(c["text"].split()) <= 60


def test_short_text_yields_single_chunk():
    chunks = chunk_text("d", "T", "just a few words", chunk_size=60, overlap=15)
    assert len(chunks) == 1
    assert chunks[0]["chunk"] == 0
    assert "just a few words" in chunks[0]["text"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("d", "T", "", chunk_size=60, overlap=15) == []
    assert chunk_text("d", "T", "   \n  ", chunk_size=60, overlap=15) == []


def test_overlap_capped_when_text_short():
    # overlap larger than the text must not crash or duplicate infinitely
    chunks = chunk_text("d", "T", "one two three", chunk_size=60, overlap=50)
    assert len(chunks) == 1

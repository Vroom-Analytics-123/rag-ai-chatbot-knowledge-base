"""Tests for rag.retriever — TF-IDF-style scoring over chunks."""

from rag.chunker import chunk_text
from rag.kb import load_kb
from rag.retriever import build_index, score_query


def make_chunks():
    kb = load_kb()
    chunks = []
    for doc in kb:
        chunks.extend(
            chunk_text(doc["id"], doc["title"], doc["text"], chunk_size=60, overlap=15)
        )
    return chunks


def test_kb_loads_expected_docs_and_chunks():
    chunks = make_chunks()
    doc_ids = {c["doc"] for c in chunks}
    assert len(doc_ids) >= 3
    assert 8 <= len(chunks) <= 14, f"expected 8-14 chunks, got {len(chunks)}"


def test_retrieval_ranks_right_chunk_first():
    chunks = make_chunks()
    index = build_index(chunks)
    results = score_query(index, "What is the no-show fee?", top_k=3)
    assert len(results) > 0
    assert results[0]["chunk"]["doc"] == "cancellation", (
        f"expected cancellation doc first, got {results[0]['chunk']['doc']}"
    )


def test_retrieval_finds_hours_answer():
    chunks = make_chunks()
    index = build_index(chunks)
    results = score_query(index, "What are your Saturday hours?", top_k=3)
    assert results[0]["chunk"]["doc"] == "hours"


def test_retrieval_finds_insurance_answer():
    chunks = make_chunks()
    index = build_index(chunks)
    results = score_query(index, "Do you accept Delta Dental insurance?", top_k=3)
    assert results[0]["chunk"]["doc"] == "insurance"


def test_scores_sorted_descending_and_bounded():
    chunks = make_chunks()
    index = build_index(chunks)
    results = score_query(index, "booking appointment phone", top_k=3)
    assert len(results) <= 3
    scores = [r["score"] for r in results]
    assert all(s >= 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_empty_kb_scores_cleanly():
    index = build_index([])
    assert score_query(index, "anything at all", top_k=3) == []


def test_gibberish_query_scores_low():
    chunks = make_chunks()
    index = build_index(chunks)
    real = score_query(index, "What is the no-show fee?", top_k=1)[0]["score"]
    junk = score_query(index, "xylophone quantum trombone", top_k=1)
    junk_score = junk[0]["score"] if junk else 0.0
    assert junk_score < real, "gibberish should score well below a real question"

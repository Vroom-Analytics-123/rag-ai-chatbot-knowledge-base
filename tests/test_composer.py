"""Tests for rag.composer — extractive answers with citations, never guessing."""

import re

from rag.chunker import chunk_text
from rag.composer import answer
from rag.kb import load_kb
from rag.retriever import build_index, score_query

CITATION = re.compile(r"\[[a-z0-9_-]+:\d+\]")

# Distinctive facts that live ONLY in the sample KB. A refusal must never
# invent or leak any of them.
KB_FACTS = [
    "Bright Smile",
    "$50",
    "Delta Dental",
    "014-2288",
    "24 hours",
    "15 minutes",
    "9:00 AM",
]


def pipeline(question, top_k=3, min_score=None):
    chunks = []
    for doc in load_kb():
        chunks.extend(
            chunk_text(doc["id"], doc["title"], doc["text"], chunk_size=60, overlap=15)
        )
    index = build_index(chunks)
    retrieved = score_query(index, question, top_k=top_k)
    kwargs = {} if min_score is None else {"min_score": min_score}
    return answer(question, retrieved, **kwargs)


def test_answer_cites_every_claim():
    result = pipeline("What is the no-show fee?")
    assert result["refused"] is False
    text = result["answer"]
    assert CITATION.search(text), "answer must contain citation markers"
    # every bullet line carries its own citation
    bullets = [ln for ln in text.splitlines() if ln.strip().startswith(("-", "*", "\u2022"))]
    assert bullets, "answer should present claims as cited bullets"
    for b in bullets:
        assert CITATION.search(b), f"uncited claim: {b}"


def test_answer_comes_from_retrieved_passages():
    result = pipeline("What is the no-show fee?")
    assert "$50" in result["answer"]  # the fact is genuinely in the KB passages
    assert result["citations"], "citations list must not be empty"
    for cite in result["citations"]:
        assert re.fullmatch(r"[a-z0-9_-]+:\d+", cite), f"bad citation format: {cite}"


def test_out_of_kb_question_refuses():
    result = pipeline("What is the capital of France?")
    assert result["refused"] is True
    for fact in KB_FACTS:
        assert fact not in result["answer"], (
            f"refusal leaked a KB fact: {fact!r}"
        )


def test_unrelated_business_question_refuses():
    result = pipeline("Do you do oil changes on Hondas?")
    assert result["refused"] is True
    for fact in KB_FACTS:
        assert fact not in result["answer"]


def test_low_score_retrieval_refuses_instead_of_guessing():
    chunks = []
    for doc in load_kb():
        chunks.extend(
            chunk_text(doc["id"], doc["title"], doc["text"], chunk_size=60, overlap=15)
        )
    index = build_index(chunks)
    retrieved = score_query(index, "What is the no-show fee?", top_k=3)
    # force the threshold above the best real score: must refuse, not guess
    best = max(r["score"] for r in retrieved)
    result = answer("What is the no-show fee?", retrieved, min_score=best + 1.0)
    assert result["refused"] is True
    # a forced refusal must not smuggle the real answer in anyway
    assert "$50" not in result["answer"]
    for fact in KB_FACTS:
        assert fact not in result["answer"]


def test_empty_retrieval_is_clean_refusal():
    result = answer("anything?", [])
    assert result["refused"] is True
    assert isinstance(result["answer"], str) and result["answer"].strip()


def test_refusal_points_to_humans():
    result = pipeline("What is the capital of France?")
    lowered = result["answer"].lower()
    assert "front-desk" in lowered or "front desk" in lowered or "team" in lowered, (
        "refusal should route the caller to a human"
    )

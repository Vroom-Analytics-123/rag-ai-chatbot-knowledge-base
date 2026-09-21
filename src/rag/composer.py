"""Extractive composer: answers ONLY from retrieved passages, with citations.

The hard rule lives here: if no retrieved passage scores at or above
`min_score`, the composer refuses — it never invents an answer from outside
the knowledge base. Answers are assembled from verbatim sentences out of the
passages, each cited [doc:chunk], so every claim traces back to a source.

This is extractive on purpose. A generative rewriter could paraphrase a fact
into a falsehood; quoting cannot.
"""

import re

from rag.retriever import tokenize

REFUSAL = (
    "I couldn't find that in the front-desk manual, so I can't answer it from "
    "our documents. Please ask the front-desk team directly and they'll help you."
)


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def answer(question, retrieved, min_score=0.095, max_bullets=2):
    """Build a cited answer from scored passages, or refuse.

    `retrieved` is a list of {'chunk', 'score'} as returned by
    retriever.score_query. Returns {'answer', 'refused', 'citations'}.
    """
    usable = [r for r in retrieved if r["score"] >= min_score]
    if not usable:
        return {"answer": REFUSAL, "refused": True, "citations": []}

    qterms = set(tokenize(question))
    candidates = []  # (hits, sentence, chunk)
    seen = set()
    for r in usable:
        chunk = r["chunk"]
        for sent in _split_sentences(chunk["text"]):
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            hits = len(qterms & set(tokenize(sent)))
            if hits > 0:
                candidates.append((hits, sent, chunk))

    # Sentences matched the passages but none matched the question's terms:
    # that's a weak retrieval wearing a costume. Refuse.
    candidates.sort(key=lambda c: c[0], reverse=True)

    # Prefer whole sentences over chunk-boundary fragments (which start
    # mid-sentence, lowercase), and prefer sentences matching 2+ query terms
    # when the question has them. Fall back gracefully rather than refusing
    # a question the KB genuinely covers.
    whole = [c for c in candidates if c[1][:1].isupper()] or candidates
    need = 2 if len(qterms) >= 2 else 1
    strong = [c for c in whole if c[0] >= need]
    picked = (strong or whole)[:max_bullets]
    if not picked:
        return {"answer": REFUSAL, "refused": True, "citations": []}

    lines = ["Here's what the front-desk manual says:"]
    citations = []
    for _, sent, chunk in picked:
        cite = f"{chunk['doc']}:{chunk['chunk']}"
        lines.append(f"\u2022 {sent} [{cite}]")
        citations.append(cite)
    return {"answer": "\n".join(lines), "refused": False, "citations": citations}

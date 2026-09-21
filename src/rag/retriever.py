"""TF-IDF-style retriever over chunks.

Deliberately simple and dependency-free: tokenize, drop stopwords, score each
chunk by sum over query terms of (term frequency x inverse document frequency).
Swap this module for an embedding retriever in production; the composer only
cares that it gets (chunk, score) pairs back.
"""

import math
import re

STOPWORDS = frozenset(
    """
    the a an is are was were be been to of and or for in on at with by it its
    this that these those as from you your we our us i me my do does did what
    when where which who whom how why can could if then than so such no not
    have has had will would should may might must there their here his her
    him she he all any each other own same just about into over after before
    """.split()
)


def _stem(token):
    """Tiny stemmer: strips -ing/-ed and plural -s so 'booked' matches 'book'."""
    t = token
    if len(t) > 4:
        for suffix in ("ing", "ed"):
            if t.endswith(suffix):
                t = t[: -len(suffix)]
                break
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [
        _stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1
    ]


def build_index(chunks):
    """Precompute per-chunk tokens and document frequencies. Empty KB -> empty index."""
    df = {}
    chunk_tokens = []
    for chunk in chunks:
        toks = tokenize(chunk["text"])
        chunk_tokens.append(toks)
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    return {"chunks": chunks, "tokens": chunk_tokens, "df": df, "n": len(chunks)}


def _idf(df, n, term):
    # +1 smoothing: a term in every chunk still carries a little weight,
    # a term in one chunk carries a lot.
    return math.log(1 + n / df.get(term, n))


def score_query(index, query, top_k=3):
    """Return up to top_k [{'chunk', 'score'}] sorted by score, descending."""
    if index["n"] == 0:
        return []
    qterms = set(tokenize(query))
    if not qterms:
        return []

    scored = []
    for chunk, toks in zip(index["chunks"], index["tokens"]):
        if not toks:
            continue
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        length = len(toks)
        score = sum(
            (counts.get(t, 0) / length) * _idf(index["df"], index["n"], t)
            for t in qterms
        )
        if score > 0:
            scored.append({"chunk": chunk, "score": score})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]

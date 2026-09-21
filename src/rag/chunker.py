"""Overlapping word-window chunker.

Splits a document into fixed-size word windows that overlap by `overlap`
words, so a fact sitting on a boundary is never cut in half. Each chunk
carries its document id, title, and index so answers can cite [doc:chunk].
"""


def chunk_text(doc_id, title, text, chunk_size=60, overlap=15):
    words = text.split()
    if not words:
        return []
    # Overlap can never swallow the window or loop forever on short texts.
    overlap = max(0, min(overlap, chunk_size - 1, len(words) - 1))
    step = chunk_size - overlap

    chunks = []
    i, idx = 0, 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(
            {
                "doc": doc_id,
                "title": title,
                "chunk": idx,
                "text": " ".join(chunk_words),
            }
        )
        if i + chunk_size >= len(words):
            break
        i += step
        idx += 1
    return chunks

"""Sample knowledge base: a fictional small business's front-desk SOP manual.

Plain markdown files in this directory — one file per document. Adding a new
doc is adding a new file; the first line (# Title) becomes the document title.
"""

import os

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb")


def load_kb():
    """Return [{'id', 'title', 'text'}] for every .md file in kb/, sorted by name."""
    docs = []
    for fname in sorted(os.listdir(KB_DIR)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(KB_DIR, fname), encoding="utf-8") as f:
            raw = f.read().strip()
        lines = raw.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else fname[:-3]
        text = "\n".join(lines[1:]).strip()
        docs.append({"id": fname[:-3], "title": title, "text": text})
    return docs

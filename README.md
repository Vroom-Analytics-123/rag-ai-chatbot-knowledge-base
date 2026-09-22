# RAG AI chatbot

Built by [Vroom Analytics](https://vroomanalytics.com) — the working pattern behind our [AI chatbot / knowledge-base service](https://vroomanalytics.com/faq/).

## What this is

Your team answers the same ten questions every day — hours, cancellations,
insurance, booking rules — and the answers already exist in a manual nobody
opens. This repo is a small, readable chatbot that answers those questions
straight from your documents (here over WhatsApp), cites every claim, and
says "I don't know, let me get a person" instead of guessing when the
manual doesn't cover it.

## 5-minute setup

Every step below was run verbatim on a fresh copy of this repo (Sep 19,
2026) — copy, paste, and it works.

1. **Create the environment and install the one dependency (pytest):**
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Run the test suite — 35 tests, all green:**
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```
   Expected: `35 passed`.

3. **Text the bot.** This simulates a customer texting the business on
   WhatsApp — message in, RAG answer, message out. No WhatsApp account, no
   network; the transport is in-memory:
   ```bash
   PYTHONPATH=src .venv/bin/python - <<'EOF'
   from rag.whatsapp import WhatsAppBot, SimulatedTransport

   bot = WhatsAppBot()          # builds the index over the sample KB once
   t = SimulatedTransport()     # in-memory WhatsApp stand-in, no network

   t.receive("+15550142", "What are your Saturday hours?")
   t.receive("+15550142", "Do you do oil changes?")
   t.pump(bot)

   for m in t.outbox:
       print("->", m["to"])
       print(m["body"])
       print()
   EOF
   ```
   Real output from the run above:
   ```
   -> +15550142
   Here's what the front-desk manual says:
   • Saturday hours are 9:00 AM to 1:00 PM. [hours:0]

   -> +15550142
   I couldn't find that in the front-desk manual, so I can't answer it from our documents. Please ask the front-desk team directly and they'll help you.

   Want a person to help instead? Reply HUMAN and the front-desk team will pick this up.
   ```
   The first question is answered with a citation. The second — "oil
   changes" at a dental clinic — is refused with a human handoff. That
   refusal is the whole product: the bot cannot answer from outside the
   manual because it has no other source of facts.

## Running the tests

```bash
.venv/bin/python -m pytest tests/ -q
```

35 tests, all green, no network calls — everything is fixture-driven. The
refusal paths are covered hardest: out-of-KB questions, weak retrieval
scores, empty KBs, blank/overlong WhatsApp messages, and refusals are
asserted to contain zero KB facts. TDD red→green; the WhatsApp tests were
written before the module existed.

## File tour

- `src/rag/kb/` — the sample knowledge base: a fictional "Bright Smile
  Dental" front-desk SOP manual (hours, booking, cancellations, insurance
  FAQs). Your documents replace these files; adding a doc is adding a
  markdown file.
- `src/rag/kb.py` — loads the markdown files into `{id, title, text}` docs.
- `src/rag/chunker.py` — splits docs into overlapping word windows
  (default 60 words, 15 overlap) with doc/chunk metadata for citations.
- `src/rag/retriever.py` — TF-IDF-style scoring, dependency-free on purpose.
  Swap this module for embeddings in production and nothing else changes.
- `src/rag/composer.py` — builds answers from retrieved passages only,
  quoting verbatim sentences with one `[doc:chunk]` citation per claim.
  **Hard rule, enforced in code:** no passage above the relevance threshold
  → refusal routed to a human. It cannot hallucinate.
- `src/rag/whatsapp.py` — the WhatsApp channel: message-in → RAG answer →
  message-out. `WhatsAppBot` answers from the manual; unanswerable questions
  get the refusal **plus a human handoff** ("Reply HUMAN…"). `SimulatedTransport`
  is an in-memory stand-in for the WhatsApp Business API — swap it for the
  real client in production; the bot never notices.
- `tests/` — 35 tests (20 core pipeline + 15 WhatsApp channel).

## What not to automate / what not to RAG

- **Don't RAG what should be a database lookup.** "Is slot X free?" is a
  query against the booking system, not a question for a document. RAG over
  a stale export of availability is how you double-book someone.
- **The KB needs a human owner or the answers rot.** If nobody updates the
  manual when the cancellation fee changes, the bot will confidently cite
  the old fee — with a citation, which is worse. Every deployment needs a
  named owner and a review cadence.
- **Don't let it answer outside the manual.** The refusal path isn't a
  fallback, it's the feature. The moment the system improvises beyond its
  sources, you have a chatbot with a library card it never uses.

## How this maps to the Vroom offer

This is the working pattern behind Vroom's **AI chatbot / knowledge-base
service** — the same architecture that powers the RAG knowledge demo: your
SOPs and manuals become a chatbot that answers customers (here on WhatsApp,
the channel your customers already use) from your documents only, with
every claim cited and every gap routed to a person. A real deployment swaps
in your documents, your channels, and embedding-based retrieval.

## The Vroom page

See it running and read how the offer works:
**[RAG knowledge demo — ask the manual](https://vroomanalytics.com/faq/)**
(that page also links the companion article on the repeated-question tax).

## Monthly peace of mind

Setup is just day one. The **$99/mo care plan** keeps this running —
monitoring, fixes, and monthly optimization, so you never think about it
again. Details on the [RAG knowledge demo page](https://vroomanalytics.com/faq/).

## License

MIT — see [LICENSE](LICENSE). Use the pattern, keep the refusal rule.

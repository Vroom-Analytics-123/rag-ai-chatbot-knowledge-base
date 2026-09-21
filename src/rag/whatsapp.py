"""WhatsApp channel adapter: message-in -> RAG answer -> message-out.

A customer texts the business on WhatsApp; the bot answers from the
front-desk manual using the same chunker -> retriever -> composer pipeline
as the core library. The no-hallucination rule survives the channel: an
unanswerable question gets a polite refusal PLUS a human handoff — the bot
never invents an answer to fill a chat bubble.

The transport is simulated (two in-memory lists, no WhatsApp Business API,
no network) so the whole message-in/message-out flow runs in tests and
demos. Swap SimulatedTransport for the real API client in production; the
bot never notices.
"""

from rag.chunker import chunk_text
from rag.composer import REFUSAL, answer
from rag.kb import load_kb
from rag.retriever import build_index, score_query

# WhatsApp's real inbound cap is ~65k characters; a customer question should
# never be anywhere near that. Past this, we ask for a shorter question
# instead of feeding a wall of text to retrieval.
MAX_INBOUND_CHARS = 1000

EMPTY_MESSAGE_REPLY = (
    "I didn't get any text in that message. Ask me a question about the "
    "clinic — hours, bookings, cancellations, insurance — and I'll look it "
    "up in the front-desk manual."
)

TOO_LONG_MESSAGE_REPLY = (
    "That message is a bit too long for me to read carefully "
    f"(I take up to {MAX_INBOUND_CHARS} characters). Could you send it as "
    "a shorter question?"
)

# The human handoff. The refusal alone leaves the customer stuck; this line
# gives them a one-word path to a person. It appears ONLY on refusals.
HANDOFF_LINE = (
    "Want a person to help instead? Reply HUMAN and the front-desk team "
    "will pick this up."
)


def format_for_whatsapp(result):
    """Render a composer result as a short, plain-text WhatsApp message.

    `result` is the {'answer', 'refused', 'citations'} dict from
    rag.composer.answer. Refusals keep the composer's wording and gain the
    human-handoff line; answered messages pass through with their [doc:chunk]
    citations intact (plain text, no markdown — chat bubbles are not docs).
    """
    if result["refused"]:
        return f"{REFUSAL}\n\n{HANDOFF_LINE}"
    return result["answer"]


class WhatsAppBot:
    """Answers inbound WhatsApp messages from the front-desk manual.

    The pipeline is built once in __init__; every message is answered
    independently, so a refusal never poisons the next answer.
    """

    def __init__(self, min_score=0.095, top_k=3):
        """Build the retrieval index over the sample KB.

        `min_score` / `top_k` mirror the composer's: tune per KB, there is
        no universal number.
        """
        chunks = []
        for doc in load_kb():
            chunks.extend(chunk_text(doc["id"], doc["title"], doc["text"]))
        self._index = build_index(chunks)
        self._min_score = min_score
        self._top_k = top_k

    def reply_to(self, body):
        """Return the outbound text for one inbound message body.

        Empty/whitespace/None bodies get a friendly prompt; overlong bodies
        get a polite rejection; everything else goes through RAG. Never
        raises on bad input — a chat channel must not crash on a blank text.
        """
        text = (body or "").strip()
        if not text:
            return EMPTY_MESSAGE_REPLY
        if len(text) > MAX_INBOUND_CHARS:
            return TOO_LONG_MESSAGE_REPLY
        retrieved = score_query(self._index, text, top_k=self._top_k)
        result = answer(text, retrieved, min_score=self._min_score)
        return format_for_whatsapp(result)

    def handle_message(self, inbound):
        """Message-in -> message-out.

        `inbound` is {'from': phone_number, 'body': text}; returns
        {'to': phone_number, 'body': reply_text} ready for the transport.
        """
        sender = inbound.get("from", "")
        return {"to": sender, "body": self.reply_to(inbound.get("body"))}


class SimulatedTransport:
    """In-memory stand-in for the WhatsApp Business API. No network.

    `inbox` collects messages the customer "sends"; `outbox` collects what
    the business "sends back". `pump` runs the full loop for tests/demos.
    """

    def __init__(self):
        """Start with empty inbox and outbox."""
        self.inbox = []
        self.outbox = []

    def receive(self, from_number, body):
        """Simulate a customer sending a WhatsApp message to the business."""
        message = {"from": from_number, "body": body}
        self.inbox.append(message)
        return message

    def send(self, to_number, body):
        """Simulate the business sending a WhatsApp message. Records it."""
        message = {"to": to_number, "body": body}
        self.outbox.append(message)
        return message

    def pump(self, bot):
        """Deliver every queued inbound message through `bot`, one reply each.

        Drains the inbox in order and returns the outbox.
        """
        while self.inbox:
            inbound = self.inbox.pop(0)
            outbound = bot.handle_message(inbound)
            self.send(outbound["to"], outbound["body"])
        return self.outbox

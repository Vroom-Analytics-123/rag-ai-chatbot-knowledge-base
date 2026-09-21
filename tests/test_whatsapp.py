"""Tests for rag.whatsapp — the WhatsApp channel over the RAG pipeline.

RED-FIRST: these tests were written before src/rag/whatsapp.py existed.
They must fail on a fresh checkout (no module), then go green.

The channel must preserve the product's core guarantee: an unanswerable
question gets a polite refusal PLUS a human handoff — never a hallucinated
answer, not even in a chat bubble.
"""

import re

import pytest

from rag.whatsapp import (
    EMPTY_MESSAGE_REPLY,
    MAX_INBOUND_CHARS,
    TOO_LONG_MESSAGE_REPLY,
    SimulatedTransport,
    WhatsAppBot,
    format_for_whatsapp,
)

CITATION = re.compile(r"\[[a-z0-9_-]+:\d+\]")

# Distinctive facts that live ONLY in the sample KB. A refusal must never
# invent or leak any of them — not in the pipeline, not in the chat bubble.
KB_FACTS = [
    "Bright Smile",
    "$50",
    "Delta Dental",
    "014-2288",
    "24 hours",
    "15 minutes",
    "9:00 AM",
]

SENDER = "+15550142"


@pytest.fixture()
def bot():
    return WhatsAppBot()


def inbound(body, sender=SENDER):
    return {"from": sender, "body": body}


# --- happy path -----------------------------------------------------------


def test_answerable_question_gets_cited_answer(bot):
    out = bot.handle_message(inbound("What is the no-show fee?"))
    assert out["to"] == SENDER
    assert "$50" in out["body"], "the genuine KB fact must appear"
    assert CITATION.search(out["body"]), "chat answer must still cite its source"
    assert "HUMAN" not in out["body"], "handoff must only appear on refusals"


def test_second_answerable_question(bot):
    out = bot.handle_message(inbound("What are your Saturday hours?"))
    assert out["to"] == SENDER
    assert CITATION.search(out["body"])


def test_format_for_whatsapp_keeps_answer_short_and_plain(bot):
    result_body = bot.reply_to("Do you take Delta Dental?")
    assert CITATION.search(result_body)
    # No markdown that renders badly in WhatsApp, and comfortably short.
    assert "**" not in result_body
    assert len(result_body) <= 1500


# --- refusal path: polite + human handoff, never a guess -------------------


def test_unanswerable_question_refuses_with_handoff(bot):
    out = bot.handle_message(inbound("Do you do oil changes?"))
    body = out["body"]
    assert out["to"] == SENDER
    assert not CITATION.search(body), "a refusal must not wear citations"
    assert "HUMAN" in body, "refusal must offer the human handoff"
    for fact in KB_FACTS:
        assert fact not in body, f"refusal leaked a KB fact: {fact}"


def test_out_of_domain_question_refuses_with_handoff(bot):
    out = bot.handle_message(inbound("What is the capital of France?"))
    assert "HUMAN" in out["body"]
    assert not CITATION.search(out["body"])


def test_format_for_whatsapp_refusal_shape():
    refused = {"answer": "not in the manual", "refused": True, "citations": []}
    text = format_for_whatsapp(refused)
    assert "HUMAN" in text
    assert not CITATION.search(text)


# --- edge cases ------------------------------------------------------------


def test_empty_message_gets_friendly_prompt(bot):
    assert bot.reply_to("") == EMPTY_MESSAGE_REPLY
    assert "HUMAN" not in EMPTY_MESSAGE_REPLY


def test_whitespace_only_message_treated_as_empty(bot):
    assert bot.reply_to("   \n  ") == EMPTY_MESSAGE_REPLY


def test_none_body_treated_as_empty(bot):
    assert bot.reply_to(None) == EMPTY_MESSAGE_REPLY


def test_overlong_message_gets_polite_rejection(bot):
    long_body = "What is the no-show fee? " + "please " * 500
    assert len(long_body) > MAX_INBOUND_CHARS
    out = bot.handle_message(inbound(long_body))
    assert out["body"] == TOO_LONG_MESSAGE_REPLY
    assert not CITATION.search(out["body"]), "overlong message must not reach RAG"
    assert "$50" not in out["body"]


def test_message_exactly_at_limit_is_still_answered(bot):
    filler = "What is the no-show fee? " + "please " * 500
    body = filler[:MAX_INBOUND_CHARS]
    assert len(body) == MAX_INBOUND_CHARS
    out = bot.handle_message(inbound(body))
    assert "$50" in out["body"], "a max-length question must still be answered"
    assert CITATION.search(out["body"])


def test_handle_message_missing_body_key(bot):
    out = bot.handle_message({"from": SENDER})
    assert out["to"] == SENDER
    assert out["body"] == EMPTY_MESSAGE_REPLY


# --- transport: message-in -> message-out, no network ----------------------


def test_transport_pump_answers_messages_in_order(bot):
    t = SimulatedTransport()
    t.receive("+15550142", "What is the no-show fee?")
    t.receive("+15550143", "Do you do oil changes?")
    t.receive("+15550142", "")
    t.pump(bot)

    assert len(t.outbox) == 3
    assert t.outbox[0]["to"] == "+15550142"
    assert "$50" in t.outbox[0]["body"]
    assert t.outbox[1]["to"] == "+15550143"
    assert "HUMAN" in t.outbox[1]["body"]
    assert t.outbox[2]["to"] == "+15550142"
    assert t.outbox[2]["body"] == EMPTY_MESSAGE_REPLY
    assert t.inbox == [], "pump must drain the inbox"


def test_transport_is_in_memory_only():
    t = SimulatedTransport()
    # No endpoint, no credentials, no socket: the transport is two lists.
    assert not any(
        attr in vars(t) for attr in ("url", "endpoint", "api_key", "token", "session")
    )
    msg = t.receive(SENDER, "hello")
    assert msg in t.inbox
    sent = t.send(SENDER, "hi back")
    assert sent in t.outbox


def test_bot_is_stateless_across_messages(bot):
    # A refusal must not poison the next answer, and vice versa.
    bot.handle_message(inbound("Do you do oil changes?"))
    out = bot.handle_message(inbound("What is the no-show fee?"))
    assert "$50" in out["body"]
    out2 = bot.handle_message(inbound("What is the capital of France?"))
    assert "HUMAN" in out2["body"]
    for fact in KB_FACTS:
        assert fact not in out2["body"]

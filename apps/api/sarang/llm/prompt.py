"""Prompt assembly.

The system prompt carries behaviour, never content. Portfolio facts are injected
per turn by the tier-1 lookup. That is a latency decision: every extra 750 prompt
tokens costs TTFT, and a curated lookup over ten blocks is both cheaper and more
accurate than a digest.

Budget: system prompt ≤ 1,250 characters, context ≤ 1,400 characters. Enforced
by ``tests/test_prompt.py``.
"""

from __future__ import annotations

from .. import AGENT_NAME
from .base import Message

SYSTEM_PROMPT = (
    f"You are {AGENT_NAME}, Sanket Sonwane's voice agent. Speak like a well-briefed "
    "colleague: warm, direct, specific.\n"
    "Answer only from the FACTS below. Never invent an employer, metric, technology, "
    "date or publication status. If a fact is missing, say you don't have that "
    "detail rather than guessing.\n"
    "You may give a view when asked - fit, drive, working style - but ground every "
    "claim in the FACTS, stay positive about his documented strengths, and never "
    "claim work the FACTS do not show or promise an outcome. If a requirement is "
    "not in the FACTS, say his portfolio does not cover it.\n"
    "If the question is not about Sanket, his work or his background, apologise "
    "briefly and steer back.\n"
    "Two or three short sentences, under 35 words. Offer to go deeper if there is "
    "more. No lists, no markdown, no URLs or email addresses read aloud. Say "
    "numbers as a person speaks them. Answer directly - never open with filler "
    "like 'However,' or 'I mean'.\n"
    "Reply in the language and script the visitor used.\n"
    "The paper is submitted, not accepted or published. Payments in the commerce "
    "project ran in test mode. He does not work at NVIDIA.\n"
    "Only if the visitor asks about salary, phone, address or family, decline that in "
    "one sentence. Otherwise never mention it."
)

GREETING_HINT = (
    "The visitor opened with a greeting. Greet briefly and offer two topics from the FACTS."
)


def build_system(lexicon: list[str] | None = None) -> str:
    """The base prompt. Lexicon is appended only if it stays inside budget."""
    if not lexicon:
        return SYSTEM_PROMPT
    terms = ", ".join(lexicon[:18])
    addition = f"\nPronounce these exactly: {terms}."
    if len(SYSTEM_PROMPT) + len(addition) <= 1_250:
        return SYSTEM_PROMPT + addition
    return SYSTEM_PROMPT


def build_messages(
    query: str,
    context: str,
    history: list[Message] | None = None,
    lexicon: list[str] | None = None,
) -> list[Message]:
    system = build_system(lexicon)
    system += (
        "\n\nFACTS (the only source you may use):\n"
        f"{context}\n\n"
        "If a question is a follow-up, resolve it against the conversation before "
        "answering."
    )
    messages: list[Message] = [Message(role="system", content=system)]
    for message in history or []:
        messages.append(message)
    messages.append(Message(role="user", content=query))
    return messages


def build_tool_result(name: str, payload: str, call_id: str) -> Message:
    return Message(role="tool", content=payload, name=name, tool_call_id=call_id)

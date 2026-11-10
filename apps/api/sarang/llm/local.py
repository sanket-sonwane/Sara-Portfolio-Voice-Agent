"""Deterministic grounded answerer.

Two jobs:

1. **Offline dev.** The whole pipeline — VAD, turn engine, segmenter, TTS, egress,
   diagnostics — runs with zero API keys, so the media path is testable before a
   single provider account exists.
2. **Runtime fallback.** If the LLM call fails or times out, the turn is answered
   from the same fact blocks the model would have seen. Degraded, never silent.

It is extractive by design. It cannot invent a metric because it only quotes
fields that exist in ``fact.yaml``.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

from ..kb.loader import FactBlock
from ..kb.lookup import LookupResult
from ..kb.retrieval import Chunk
from .base import Done, LLMEvent, TextDelta

GREETING_RE = re.compile(r"^\s*(hi|hey|hello|yo|good (morning|afternoon|evening))\b", re.I)

REFUSALS = {
    "salary": "I don't discuss compensation on this call. Happy to talk about his work.",
    "compensation": "I don't discuss compensation on this call. Happy to talk about his work.",
    "phone": "I won't share a phone number. The public contact is the email on the portfolio.",
    "patent": "There are no patents documented in this portfolio, so I can't claim any.",
    "revenue": "The project ran in test mode, so there is no revenue figure to report.",
    "google": "There is no Google experience in this portfolio.",
    "microsoft": "There is no Microsoft experience in this portfolio.",
    "cluster": "There is no production cluster documented, so I can't describe one.",
    "years of experience": (
        "The portfolio doesn't reduce the work to a years-of-experience number. "
        "It documents three internships, six projects and a submitted paper instead."
    ),
    "how many years": (
        "The portfolio doesn't give a years-of-experience figure. "
        "It documents the internships and projects themselves."
    ),
    "12th": "The portfolio documents the B.E. and the diploma, not 12th or HSC marks, so I can't quote them.",
    "hsc": "The portfolio documents the B.E. and the diploma, not 12th or HSC marks, so I can't quote them.",
    "ssc": "The portfolio documents the B.E. and the diploma, not 10th or SSC marks, so I can't quote them.",
}

NOT_DOCUMENTED = "I don't have that detail, and I'd rather say so than guess."

OFF_TOPIC_FALLBACK = (
    "Sorry, I'm Sanket's voice agent, so I can only speak to him and his work. "
    "Happy to talk about his projects, experience, or how he'd fit a role."
)

BOUNDARY_RESPONSES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(salary|compensation|ctc|pay|package|stipend)\b"),
        "I don't discuss compensation on this call. Happy to talk about his work.",
    ),
    (
        re.compile(r"\b(phone|mobile|whatsapp)\b|\bcontact number\b"),
        "I won't share a phone number. The public email is on the portfolio page.",
    ),
    (
        re.compile(r"\b(address|dob|birth|age|family|married|religion|caste)\b"),
        "I keep personal details private. Happy to talk about his work and background.",
    ),
    (
        re.compile(r"\b(weakness|weaknesses|weak|gap|gaps|flaw|shortcoming)\b"),
        "I'd rather not speculate about weaknesses. I can walk you through what he has "
        "built and let the work speak.",
    ),
    (
        re.compile(r"\b(google|microsoft)\b"),
        "There is no Google or Microsoft experience in this record, so I won't imply one.",
    ),
    (
        re.compile(r"\b(patent|patents)\b"),
        "There are no patents documented here, so I can't claim any.",
    ),
)

PROFILE_FIELDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vision", re.compile(r"\b(vision|future|five years|long term|where does he see)\b")),
    ("mission", re.compile(r"\b(mission|working on|right now|currently|focus)\b")),
    (
        "community",
        re.compile(r"\b(community|volunteer|meetup|mentor|give back|developer tech society)\b"),
    ),
    (
        "interests",
        re.compile(r"\b(hobb|interest|free time|outside work|chess|travel|for fun)\b"),
    ),
    ("values", re.compile(r"\b(values|working style|how does he work|principles?)\b")),
    ("motivation", re.compile(r"\b(motivat|driven|passion|what drives)\b")),
)


def _first_sentence(text: str) -> str:
    clean = " ".join(text.split())
    return clean.split(". ")[0].rstrip(".")


def _numbers(block: FactBlock) -> str:
    metrics = block.data.get("metrics") or []
    parts = []
    for metric in metrics:
        label = str(metric.get("label", "")).strip()
        value = str(metric.get("value", "")).strip()
        if label and value:
            parts.append(f"{label}: {value}")
    return ". ".join(parts)


def _project(block: FactBlock) -> str:
    data = block.data
    sentence = f"{data.get('name', 'That project')} is {data.get('tagline', 'a project in the portfolio')}."
    summary = data.get("summary", "")
    if summary:
        sentence += " " + _first_sentence(summary) + "."
    numbers = _numbers(block)
    if numbers:
        sentence += " " + numbers + "."
    return sentence


def _experience(block: FactBlock) -> str:
    data = block.data
    sentence = (
        f"At {data.get('org', 'that organisation')} he worked as {data.get('role', 'an intern')} "
        f"from {data.get('start', '')} to {data.get('end', '')}."
    )
    summary = data.get("summary", "")
    if summary:
        sentence += " " + _first_sentence(summary) + "."
    numbers = _numbers(block)
    if numbers:
        sentence += " " + numbers + "."
    return sentence


def _education(block: FactBlock) -> str:
    data = block.data
    return (
        f"{data.get('qualification', '')} at {data.get('institution', '')}, "
        f"{data.get('score_label', 'score')} {data.get('score', '')}."
    )


def _research(block: FactBlock) -> str:
    data = block.data
    sentence = (
        f"The paper is {data.get('title', '')}, submitted to {data.get('venue', '')}. "
        f"Its status is {data.get('status', 'submitted')}, not accepted or published."
    )
    numbers = _numbers(block)
    if numbers:
        sentence += " " + numbers + "."
    return sentence


def _skills(block: FactBlock) -> str:
    entries = block.data.get("entries") or []
    names = [entry.get("name", "") for entry in entries if entry.get("name")]
    if not names:
        return NOT_DOCUMENTED
    return f"In that area: {', '.join(names[:10])}."


def _contact(block: FactBlock) -> str:
    data = block.data
    email = str(data.get("email") or "").strip()
    links = data.get("links") or {}
    parts = [f"The public email is {email}."] if email else []
    if links.get("github"):
        parts.append("GitHub and LinkedIn are linked from the portfolio.")
    elif links.get("portfolio"):
        parts.append(f"The portfolio is {links['portfolio']}.")
    return " ".join(parts) or NOT_DOCUMENTED


def _owner(block: FactBlock) -> str:
    data = block.data
    name = str(data.get("name") or "Sanket").strip()
    headline = str(data.get("headline") or "").strip()
    location = str(data.get("location") or "").strip()
    if headline and location:
        parts = [f"{name}, {headline} based in {location}."]
    elif headline:
        parts = [f"{name} — {headline}."]
    else:
        parts = [f"{name} works on applied AI systems."]
    tagline = str(data.get("tagline") or "").strip()
    if tagline:
        parts.append(tagline.rstrip(".") + ".")
    focus = str(data.get("focus") or "").strip()
    if focus:
        parts.append(f"His focus is {focus.rstrip('.')}.")
    availability = str(data.get("availability") or "").strip()
    if availability:
        parts.append(availability.rstrip(".") + ".")
    # The public email is given only when contact is asked for: reading an address
    # aloud unprompted is noise, and the TTS has to spell it out.
    return " ".join(parts)


def _profile(block: FactBlock, lowered: str) -> str:
    data = block.data
    for field, pattern in PROFILE_FIELDS:
        if not pattern.search(lowered):
            continue
        value = data.get(field)
        if isinstance(value, list):
            parts = [str(item).rstrip(".") + "." for item in value if str(item).strip()]
            if parts:
                return " ".join(parts)
        elif value:
            return str(value).strip()
    parts = [
        str(data.get(key) or "").strip()
        for key in ("motivation", "vision")
        if str(data.get(key) or "").strip()
    ]
    return " ".join(parts) or NOT_DOCUMENTED


def _role_fit(block: FactBlock) -> str:
    data = block.data
    headline = str(data.get("headline") or "").strip()
    proof = str(data.get("proof") or "").strip()
    strengths = [
        str(item.get("claim", "")).strip()
        for item in data.get("strengths") or []
        if str(item.get("claim", "")).strip()
    ]
    parts = [headline] if headline else []
    if strengths:
        parts.append("Strongest documented areas: " + ", ".join(strengths[:3]) + ".")
    if proof:
        parts.append(proof)
    parts.append("Happy to go deeper on any of them.")
    return " ".join(part for part in parts if part) or NOT_DOCUMENTED


def _portfolio_overview(blocks: list[FactBlock]) -> str:
    entries: list[str] = []
    for block in blocks:
        if block.kind != "project":
            continue
        name = str(block.data.get("name") or "").strip()
        tagline = str(block.data.get("tagline") or "").strip()
        if not name:
            continue
        entries.append(f"{name} ({tagline})" if tagline else name)
        if len(entries) == 4:
            break
    if not entries:
        return NOT_DOCUMENTED
    if len(entries) == 1:
        return f"The portfolio includes {entries[0]}."
    return (
        "The portfolio includes "
        + ", ".join(entries[:-1])
        + f" and {entries[-1]}. Ask about any of them."
    )


def answer_from(query: str, lookup: LookupResult, chunks: list[Chunk] | None = None) -> str:
    lowered = query.lower()

    if GREETING_RE.search(lowered):
        return "Hi. Ask me about Sanket's work, background, or how he'd fit a role."

    for keyword, refusal in REFUSALS.items():
        if keyword in lowered:
            return refusal

    if lookup.reason in {"off topic", "task request"}:
        return (lookup.top.text if lookup.top is not None else "") or OFF_TOPIC_FALLBACK

    if lookup.blocked:
        for pattern, response in BOUNDARY_RESPONSES:
            if pattern.search(lowered):
                return response
        return "That's outside what I can cover here, so I'll stay on his work."

    if not lookup.confident or not lookup.blocks:
        if chunks:
            top = chunks[0]
            return f"From the case study: {top.text.split('. ')[0].rstrip('.')}."
        return NOT_DOCUMENTED

    if lookup.reason == "generic project overview":
        return _portfolio_overview(lookup.blocks)

    block = lookup.blocks[0]
    if block.kind in {"role_fit", "capabilities"}:
        fit = next((item for item in lookup.blocks if item.kind == "role_fit"), block)
        return _role_fit(fit)
    if block.kind == "profile":
        return _profile(block, lowered)
    if block.kind == "off_topic":
        return block.text or OFF_TOPIC_FALLBACK
    if block.kind == "project":
        return _project(block)
    if block.kind == "experience":
        return _experience(block)
    if block.kind == "education":
        return _education(block)
    if block.kind == "research":
        return _research(block)
    if block.kind == "skills":
        return _skills(block)
    if block.kind == "achievements":
        items = block.data.get("items") or []
        if items:
            first = items[0]
            return f"{first.get('title', '')}. {first.get('summary', '')}"
        return NOT_DOCUMENTED
    if block.kind == "leadership":
        items = block.data.get("items") or []
        if items:
            return " ".join(f"{item.get('role', '')}: {item.get('summary', '')}" for item in items[:2])
        return NOT_DOCUMENTED
    if block.kind == "pending":
        return "That project exists but isn't documented in the portfolio yet, so I won't guess at details."
    if block.kind == "boundaries":
        return "That's outside what this portfolio covers."
    if block.kind == "owner":
        if re.search(r"\b(email|mail|contact|reach|linkedin|github|hire)\b", lowered):
            return _contact(block)
        return _owner(block)
    return NOT_DOCUMENTED


async def stream_local_answer(
    query: str,
    lookup: LookupResult,
    chunks: list[Chunk] | None = None,
    *,
    delay: float = 0.012,
) -> AsyncIterator[LLMEvent]:
    """Emit the answer word by word so downstream stages see a real stream."""
    text = answer_from(query, lookup, chunks)
    for index, word in enumerate(text.split()):
        prefix = "" if index == 0 else " "
        yield TextDelta(prefix + word)
        await asyncio.sleep(delay)
    yield Done(usage={"provider": "local"}, finish_reason="stop")

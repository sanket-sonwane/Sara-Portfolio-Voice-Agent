"""Render fact blocks into the smallest prompt that can answer the question.

Budget rule: tier-1 context stays under ~700 characters and the whole system
prompt under ~1,200 characters (roughly 300 tokens). Prompt length is a measured
cost, not a style choice.

Two invariants matter more than the exact number:

* **Nothing disappears silently.** A block that does not fit is named in an
  ``[also documented, not shown: ...]`` line, because a model that is told "these
  are the only facts" will report a dropped block as a fact that does not exist.
* **No block is cut mid-word.** The first block is always shown, trimmed at a
  sentence boundary if it alone exceeds the budget.
"""

from __future__ import annotations

import re

from .loader import FactBlock

MAX_CONTEXT_CHARS = 1_400
# Room kept back for the "also documented" marker so the whole context still
# respects MAX_CONTEXT_CHARS when more than one block was selected.
INDEX_RESERVE = 180
INDEX_MAX_TITLES = 8

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def render_block(block: FactBlock) -> str:
    text = " ".join(block.text.split())
    # Project ids are slugs; a model that sees `[project:agentic-commerce-gateway]`
    # will sometimes repeat the slug aloud. Project blocks carry a display title.
    label = block.title if block.kind == "project" else block.id
    return f"[{label}] {text}"


def _trim_to_sentence(text: str, limit: int) -> str:
    """Longest prefix of ``text`` within ``limit`` that ends on a sentence."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = 0
    for match in _SENTENCE_END.finditer(window):
        cut = match.start()
    if cut >= max(160, limit // 2):
        return window[:cut].rstrip()
    space = window.rfind(" ")
    return (window[:space] if space > 0 else window).rstrip()


def _index_line(omitted: list[str]) -> str:
    titles = omitted[:INDEX_MAX_TITLES]
    suffix = f"; and {len(omitted) - len(titles)} more" if len(omitted) > len(titles) else ""
    return "[also documented, not shown: " + "; ".join(titles) + suffix + "]"


def render_context(blocks: list[FactBlock], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not blocks:
        return "(no matching portfolio evidence)"

    # Reserve room for the completeness marker only when there is more than one
    # block, so a single-block answer keeps the whole budget.
    limit = max_chars - (INDEX_RESERVE if len(blocks) > 1 else 0)

    parts: list[str] = []
    used = 0
    omitted: list[str] = []
    for block in blocks:
        rendered = render_block(block)
        remaining = limit - used
        if len(rendered) <= remaining:
            parts.append(rendered)
            used += len(rendered) + 1
            continue
        if not parts:
            # The primary evidence is never dropped: trim it instead.
            trimmed = _trim_to_sentence(rendered, max(remaining, 400))
            parts.append(trimmed)
            used += len(trimmed) + 1
            if len(trimmed) < len(rendered):
                omitted.append(block.title)
            continue
        omitted.append(block.title)

    if omitted:
        parts.append(_index_line(omitted))
    return "\n".join(parts)


def render_overview(blocks: list[FactBlock], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """One line per project for "what has he built" questions.

    A full block per project does not fit, and dropping five of six makes the
    model deny they exist. A compact index is the honest shape for an overview.
    """
    focus = ""
    lines: list[str] = []
    for block in blocks:
        if block.kind == "project":
            name = str(block.data.get("name") or block.title).strip()
            tagline = str(block.data.get("tagline") or "").strip()
            lines.append(f"{name} — {tagline}" if tagline else name)
        elif block.kind in {"profile", "owner"} and not focus:
            text = " ".join(block.text.split())
            cut = text.find(". ")
            focus = text[: cut + 1] if cut > 0 else text[:220]
    if not lines:
        return render_context(blocks, max_chars=max_chars)
    body = (focus + " " if focus else "") + (
        "Projects documented in the portfolio: " + "; ".join(lines) + "."
    )
    if len(body) > max_chars:
        body = _trim_to_sentence(body, max_chars)
    return body


def render_evidence(blocks: list[FactBlock]) -> list[dict]:
    """What the UI shows as sources for the answer."""
    out = []
    for block in blocks:
        out.append(
            {
                "id": block.id,
                "kind": block.kind,
                "title": block.title,
                "snippet": " ".join(block.text.split())[:220],
            }
        )
    return out

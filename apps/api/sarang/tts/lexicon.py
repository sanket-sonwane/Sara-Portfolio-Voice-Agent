"""TTS-only pronunciation lexicon.

Some names in this portfolio are not pronounceable as written by an English
voice: ``ARAI`` is an acronym, ``sched_ext`` is an identifier, and
``VisionMitra``/``ClassNode`` are camel-case product names. The fix belongs at
the synthesis boundary and nowhere else.

Rules:

* **Display text is never rewritten.** The transcript, the evidence card and
  the history record what the model actually said. Only the string handed to
  the TTS provider passes through ``speakable``.
* **Matching is by whole token.** ``Visions`` must not match ``VisionMitra``
  and ordinary prose must not be altered by a substring rule.
* **Every rule is explicit.** This is a curated list, not transliteration; a
  guessed pronunciation is worse than the provider's own guess.
"""

from __future__ import annotations

import re
from functools import lru_cache

# (pattern, spoken form). Acronyms are spelled out; identifiers are split into
# words the provider can pronounce.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bARAI\b"), "A R A I"),
    (re.compile(r"\bMEC\b"), "M E C"),
    (re.compile(r"\bRAG\b"), "R A G"),
    (re.compile(r"\bLLM\b"), "L L M"),
    (re.compile(r"\bASR\b"), "A S R"),
    (re.compile(r"\bTTS\b"), "T T S"),
    (re.compile(r"\bVisionMitra\b", re.IGNORECASE), "Vision Mitra"),
    (re.compile(r"\bClassNode\b", re.IGNORECASE), "Class Node"),
    (re.compile(r"\bsched_ext\b", re.IGNORECASE), "sched ex"),
    (re.compile(r"\beBPF\b", re.IGNORECASE), "e B P F"),
)


@lru_cache(maxsize=512)
def speakable(text: str) -> str:
    """Return ``text`` as it should be pronounced, never as it is displayed."""
    if not text:
        return text
    spoken = text
    for pattern, replacement in _RULES:
        spoken = pattern.sub(replacement, spoken)
    return spoken


__all__ = ["speakable"]

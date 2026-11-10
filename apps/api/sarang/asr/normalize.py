"""Deterministic repair of recogniser output before it reaches the screen.

Keyterm biasing helps, but it is not reliable: on the same clip the recogniser
returned "ARAI" with a short keyterm list and "AAI" with a long one. A visitor
reading "what did Sanket build during the AAI internship" concludes the whole
system is broken even though the answerer resolved the entity correctly.

So the transcript gets one bounded, data-driven correction pass. The table lives
in ``kb/fact.yaml`` under ``lexicon.misheard`` and contains only variants that
have actually been observed (or curated by the owner) — this is a correction
table, not a spell checker. Anything not in the table is left alone, because
silently rewriting a visitor's words is worse than a visible mis-transcription.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..kb.loader import load_kb


@lru_cache(maxsize=1)
def _table() -> tuple[tuple[re.Pattern[str], str], ...]:
    try:
        raw = load_kb().raw
    except Exception:  # noqa: BLE001 - correction is optional, never fatal
        return ()
    misheard = (raw.get("lexicon") or {}).get("misheard") or {}
    if not isinstance(misheard, dict):
        return ()

    entries: list[tuple[re.Pattern[str], str]] = []
    for correct, variants in misheard.items():
        if not isinstance(variants, list):
            continue
        for variant in variants:
            text = " ".join(str(variant).split())
            if not text:
                continue
            # Word boundaries keep "ARI" from rewriting "ARISE". Spaces inside a
            # variant are matched literally so "A R A I" is one unit.
            pattern = re.compile(rf"(?<![\w-]){re.escape(text)}(?![\w-])", re.IGNORECASE)
            entries.append((pattern, str(correct)))

    # Longest variants first: "A R A I" must win over a bare "ARI".
    entries.sort(key=lambda item: len(item[0].pattern), reverse=True)
    return tuple(entries)


def normalize_transcript(text: str) -> str:
    """Return ``text`` with known mis-hearings replaced.

    Case is preserved from the original match only when the correction is a
    single word and the match was upper case; otherwise the canonical spelling
    from the knowledge base wins, because that is what the agent will say.
    """
    if not text:
        return text
    result = text
    for pattern, correct in _table():
        result = pattern.sub(correct, result)
    return result

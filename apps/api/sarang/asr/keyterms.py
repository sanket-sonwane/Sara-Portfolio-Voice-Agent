"""Recognition biasing for Sarvam's realtime recogniser.

``saaras:v4`` accepts two features that stop proper nouns from being mangled at
the source instead of patching them afterwards:

* ``keyterms`` — terms the recogniser should prefer.
* ``prompt`` — a short domain hint applied by the non-streaming finalizer.

Both are derived from the same knowledge base the agent answers from, so the
recogniser and the answerer agree on spelling.

**The list is short on purpose, and that is measured, not stylistic.** On a
fixture clip of "what did Sanket build during the ARAI internship", repeated
runs against the live endpoint gave:

============================  ===================
keyterms sent                 transcript
============================  ===================
none                          "ARII" (3/3)
1 ("ARAI")                    "ARAI" (2/2)
2 acronyms                    "ARAI" (4/4)
3 acronyms + projects         "ARAI" (2/2)
7 acronyms                    "ARII" (2/2)
20 terms (all acronyms)       "AAI" (4/4)
============================  ===================

So keyterms help only while the acronym set stays small: past a handful, the
candidates compete with each other and the decoder lands on a *different*
plausible acronym. Project names are free — they are not confusable with each
other — so only the acronyms are rationed.

Two further failure modes were measured and are now excluded:

* **Spelled-out letter sequences.** A keyterm like ``"A E U E"`` teaches the
  decoder to spell, and "ARAI" came back as "A E U E".
* **Separators.** The provider rejects the whole session
  (``invalid_request``, fatal) if any single term contains a comma or semicolon,
  which used to happen because of ``"AISSMS Polytechnic, Pune"``.
"""

from __future__ import annotations

from functools import lru_cache

from ..kb.loader import load_kb

MAX_KEYTERMS = 12
MAX_TERM_CHARS = 64

# Acronyms are the scarce budget (see the module docstring): the decoder can
# guess a project name from context, but not a four-letter initialism. These
# three are the ones a visitor actually says, ordered by how often they appear
# in the knowledge base. Everything else in the lexicon still reaches the
# recogniser through ``DOMAIN_PROMPT``, which does not compete for slots.
PRIORITY_ACRONYMS = ("ARAI", "AEUE", "ZEISS INSPECT")

DOMAIN_PROMPT = (
    "Technical interview about Sanket Sonwane, an AI engineer in Pune, India. "
    "Projects: Agentic Commerce Gateway, Municipal Asset Intelligence Copilot, "
    "TattvaQuest, QA-Forge, Class-Node, VisionMitra. Institutions: ARAI, ZEISS, "
    "COEP, SPPU, AISSMS, PVPIT. Terms: RAG, QLoRA, sched_ext, Nemotron, AEUE."
)

_SEPARATORS = (",", ";", "\u2014", "\u2013")


def _clean(term: str) -> str:
    """Normalise one keyterm to something the provider will accept.

    Anything after a separator is dropped: ``"AISSMS Polytechnic, Pune"`` becomes
    ``"AISSMS Polytechnic"`` and ``"ARAI — Automotive Research Association of
    India"`` becomes ``"ARAI"``. Both are the part a speaker actually says.
    """
    raw = str(term)
    for separator in _SEPARATORS:
        raw = raw.split(separator)[0]
    return " ".join(raw.split())[:MAX_TERM_CHARS]


def _usable(term: str) -> bool:
    if not term or len(term) < 2:
        return False
    tokens = term.split()
    if len(tokens) > 5:
        return False
    # "A E U E" is the poison: it biases the decoder toward spelling.
    if len(tokens) > 1 and all(len(token) <= 2 for token in tokens):
        return False
    return True


@lru_cache(maxsize=1)
def build_keyterms() -> tuple[str, ...]:
    """Return the keyterms to send: rationed acronyms, then project names."""
    try:
        raw = load_kb().raw
    except Exception:  # noqa: BLE001 - biasing is optional, never fatal
        return ()

    ordered: list[str] = list(PRIORITY_ACRONYMS)

    owner = raw.get("owner") or {}
    ordered.append(str(owner.get("name", "")))

    for project in raw.get("projects") or []:
        ordered.extend(
            [str(project.get("name", "")), str(project.get("short_name", ""))]
        )

    ordered.append("Sara")

    seen: set[str] = set()
    terms: list[str] = []
    for candidate in ordered:
        cleaned = _clean(candidate)
        key = cleaned.lower()
        if key in seen or not _usable(cleaned):
            continue
        seen.add(key)
        terms.append(cleaned)
        if len(terms) >= MAX_KEYTERMS:
            break
    return tuple(terms)


def domain_prompt() -> str:
    return DOMAIN_PROMPT

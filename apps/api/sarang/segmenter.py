"""Incremental speech segmentation.

The first segment is the whole latency story: TTS cannot start until it exists.
So the segmenter is biased toward releasing early, and only after the first
segment does it care about natural phrasing.

Rules
-----
* Release at sentence punctuation as soon as the segment is long enough.
* Hold the first segment for at most ``first_hold_ms`` once it has a few words.
* Never exceed ``max_words`` / ``max_chars``; cut at the last space.
* Do not split after a known abbreviation, an initial, a decimal or a version.
* Text is normalised for speech (no markdown, no URLs, symbols spoken).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[.!?]+(?=\s|$)")
_CLAUSE_RE = re.compile(r"[,;:](?=\s)")
_WORD_RE = re.compile(r"\S+")

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "approx", "dept", "est", "inc", "ltd", "co", "fig", "no", "vol", "al", "ca",
}

REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("e.g.", "for example"),
    ("i.e.", "that is"),
    ("vs.", "versus"),
    ("&", " and "),
    ("→", " to "),
    ("->", " to "),
    ("—", ", "),
    ("–", ", "),
    ("₹", " rupees "),
    ("$", " dollars "),
    ("%", " percent"),
    ("#", " "),
    ("*", ""),
    ("`", ""),
    ("_", " "),
    ("  ", " "),
)


def normalize_for_speech(text: str) -> str:
    cleaned = _URL_RE.sub("the link", text)
    cleaned = cleaned.replace("**", "").replace("__", "")
    for source, target in REPLACEMENTS:
        cleaned = cleaned.replace(source, target)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _is_safe_boundary(text: str, punctuation_index: int) -> bool:
    """True when the punctuation really ends a clause rather than an abbreviation."""
    before = text[:punctuation_index].rstrip()
    if not before:
        return False
    token = before.rsplit(" ", 1)[-1].lower().strip("([{")
    if token in ABBREVIATIONS:
        return False
    # "v2.5", "3.14", "1.2.3"
    if token and token[-1].isdigit() and text[punctuation_index] == ".":
        return False
    # Single initial, as in "J. Smith".
    if len(token) == 1 and token.isalpha():
        return False
    return True


@dataclass
class TextSegmenter:
    min_words_first: int = 3
    max_words_first: int = 14
    max_words: int = 24
    min_chars: int = 14
    max_chars: int = 150
    # Release the first speakable chunk quickly so TTS can start while the
    # model is still generating; later segments prefer natural phrasing.
    first_hold_ms: int = 120

    _buffer: str = ""
    _first_released: bool = False
    _started_at: float = 0.0
    _last_append: float = 0.0
    _emitted: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ feed

    def push(self, delta: str) -> list[str]:
        if not delta:
            return []
        if not self._started_at:
            self._started_at = time.perf_counter()
        self._buffer += delta
        self._last_append = time.perf_counter()
        return self._extract(final=False)

    def due(self, now: float | None = None) -> list[str]:
        """Force the first segment out once it has waited too long."""
        if self._first_released or not self._buffer.strip():
            return []
        moment = now if now is not None else time.perf_counter()
        words = _words(self._buffer)
        waited_ms = (moment - self._started_at) * 1000.0
        if (len(words) >= self.min_words_first and waited_ms >= self.first_hold_ms) or len(
            self._buffer
        ) >= self.max_chars:
            return self._extract(final=False, force_first=True)
        return []

    def flush(self) -> list[str]:
        return self._extract(final=True)

    @property
    def has_pending(self) -> bool:
        return bool(self._buffer.strip())

    @property
    def emitted(self) -> list[str]:
        return list(self._emitted)

    # --------------------------------------------------------------- internal

    def _extract(self, *, final: bool, force_first: bool = False) -> list[str]:
        out: list[str] = []
        while True:
            candidate = self._pick_boundary(final=final, force_first=force_first and not out)
            if candidate is None:
                break
            segment, remainder = candidate
            segment = normalize_for_speech(segment)
            self._buffer = remainder
            if len(segment) >= 2 and any(char.isalnum() for char in segment):
                self._emitted.append(segment)
                self._first_released = True
                out.append(segment)
        return out

    def _pick_boundary(
        self, *, final: bool, force_first: bool
    ) -> tuple[str, str] | None:
        text = self._buffer
        stripped = text.strip()
        if not stripped:
            return None
        words = _words(stripped)
        first = not self._first_released

        # 1. Sentence punctuation, from the end so we take the longest safe cut.
        for match in reversed(list(_SENTENCE_RE.finditer(stripped))):
            index = match.start()
            if not _is_safe_boundary(stripped, index):
                continue
            head = stripped[: match.end()].strip()
            if len(_words(head)) < (self.min_words_first if first else 2):
                continue
            if len(head) > self.max_chars:
                continue
            tail = stripped[match.end():].strip()
            if not tail and not final and not force_first:
                # A sentence that ends exactly at the buffer edge is almost
                # certainly still being written; wait unless told to release.
                continue
            return head, tail

        # 2. Clause punctuation for a long-enough clause.
        if len(words) >= (self.max_words_first if first else 10):
            for match in reversed(list(_CLAUSE_RE.finditer(stripped))):
                index = match.start()
                if not _is_safe_boundary(stripped, index):
                    continue
                head = stripped[: index + 1].strip()
                if len(_words(head)) >= (self.min_words_first if first else 6):
                    if len(head) <= self.max_chars:
                        return head, stripped[index + 1:].strip()

        # 3. Hard limits: always cut, because speech cannot wait for punctuation.
        if len(stripped) > self.max_chars:
            head = stripped[: self.max_chars]
            cut = head.rfind(" ")
            cut = cut if cut >= self.min_chars else self.max_chars
            return stripped[:cut].strip(), stripped[cut:].strip()

        limit_words = self.max_words_first if first else self.max_words
        if len(words) > limit_words:
            return self._cut_words(words, limit_words)

        # 4. Final flush, or a forced first segment, takes whatever is left.
        if (final or force_first) and stripped:
            return stripped, ""
        return None

    def _cut_words(self, words: list[str], limit: int) -> tuple[str, str]:
        head = " ".join(words[:limit])
        if len(head) > self.max_chars:
            cut = head.rfind(" ", 0, self.max_chars)
            cut = cut if cut >= self.min_chars else self.max_chars
            head = head[:cut].strip()
            tail = self._buffer[cut:].strip()
            return head, tail
        tail = " ".join(words[limit:])
        return head, tail


def split_script(text: str, max_chars: int = 160) -> list[str]:
    """Non-streaming convenience used by probes and tests."""
    segmenter = TextSegmenter(max_chars=max_chars)
    out = segmenter.push(text)
    out.extend(segmenter.flush())
    return out

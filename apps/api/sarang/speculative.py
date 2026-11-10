"""Speculative preparation from partial transcripts.

Only the safe half of speculation is implemented. A stable partial hypothesis is
used to warm the tier-1 lookup and, on a depth question, the tier-2 retrieval. The
LLM is *not* invoked speculatively: the PRD forbids speaking speculative content,
and an LLM call is the expensive, hard-to-cancel part.

On the final transcript the speculation is adopted only when the hypotheses agree
well enough. Otherwise it is discarded and counted as wasted work, which is the
number the PRD asks to be measured.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .kb.lookup import LookupResult, lookup, tokenize
from .kb.retrieval import Chunk, search_docs


@dataclass
class Speculation:
    text: str
    tokens: set[str]
    result: LookupResult
    chunks: list[Chunk] = field(default_factory=list)
    created_at: float = field(default_factory=time.perf_counter)
    used: bool = False

    def matches(self, final_text: str, threshold: float = 0.6) -> bool:
        final_tokens = set(tokenize(final_text))
        if not final_tokens or not self.tokens:
            return False
        overlap = len(final_tokens & self.tokens) / len(final_tokens)
        return overlap >= threshold


class Speculator:
    def __init__(self, *, enabled: bool = True, min_words: int = 4, stable_hits: int = 2) -> None:
        self.enabled = enabled
        self.min_words = min_words
        self.stable_hits = stable_hits
        self._last_text = ""
        self._hits = 0
        self._candidate: Speculation | None = None
        self._lock = asyncio.Lock()
        self.prepared = 0
        self.adopted = 0
        self.wasted = 0

    def reset(self) -> None:
        self._last_text = ""
        self._hits = 0
        self._candidate = None

    async def note(self, partial: str) -> Speculation | None:
        if not self.enabled:
            return None
        text = " ".join(partial.split())
        if len(text.split()) < self.min_words:
            self._last_text = text
            self._hits = 0
            return None
        if text == self._last_text:
            self._hits += 1
        else:
            self._last_text = text
            self._hits = 1
            self._candidate = None
        if self._hits < self.stable_hits:
            return None
        if self._candidate is not None and self._candidate.text == text:
            return self._candidate

        async with self._lock:
            if self._candidate is not None and self._candidate.text == text:
                return self._candidate
            started = time.perf_counter()
            result = await asyncio.to_thread(lookup, text)
            chunks: list[Chunk] = []
            if result.needs_depth:
                try:
                    chunks = await asyncio.wait_for(
                        asyncio.to_thread(search_docs, text, 3), timeout=0.5
                    )
                except Exception:
                    chunks = []
            self._candidate = Speculation(text=text, tokens=set(tokenize(text)), result=result)
            self._candidate.chunks = chunks
            self.prepared += 1
            _ = (time.perf_counter() - started) * 1000
            return self._candidate

    def consume(self, final_text: str) -> Speculation | None:
        candidate = self._candidate
        self._candidate = None
        self.reset()
        if candidate is None:
            return None
        if candidate.matches(final_text):
            candidate.used = True
            self.adopted += 1
            return candidate
        self.wasted += 1
        return None

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "prepared": self.prepared,
            "adopted": self.adopted,
            "wasted": self.wasted,
        }

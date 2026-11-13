"""Speculative preparation from partial transcripts.

Two halves, both safe:

* A stable partial hypothesis warms the tier-1 lookup and, on a depth
  question, the tier-2 retrieval - the answer's evidence is ready before the
  visitor stops talking.
* When a language model is configured, an *arm* starts streaming an answer
  for that hypothesis while the visitor is still talking (PRD §17.6 allows
  bounded speculation; unbounded speculation is forbidden). The stream is
  never spoken before the final transcript confirms it: it sits in a queue
  and is replayed through the normal consume path only when every gate
  passes.

Gates before an arm is replayed, cheapest first:

1. token overlap - 0.6 of the final transcript's topical tokens were in the
   hypothesis (``Speculation.matches``).
2. evidence - the final transcript routes to exactly the blocks the arm's
   answer was grounded on. The same question with a boundary tacked on
   (``... and also his salary``) clears the token gate and must still be
   refused here.
3. prompt equality - everything *except* the question text (system prompt
   with its context, conversation history) must be identical to what the
   turn will send. The question itself may differ: the final transcript can
   add words the partial had not heard yet, which gates 1 and 2 bound.

Wasted work is counted. ``armed`` / ``arm_adopted`` / ``arm_wasted`` are the
numbers the PRD asks to be measured; they surface in ``/api/metrics``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .kb.lookup import LookupResult, lookup, tokenize
from .kb.retrieval import Chunk, search_docs
from .llm.base import Message


@dataclass
class Speculation:
    text: str
    tokens: set[str]
    result: LookupResult
    chunks: list[Chunk] = field(default_factory=list)
    created_at: float = field(default_factory=time.perf_counter)
    used: bool = False
    # The LLM arm: the queue a speculative stream fills, the task that fills
    # it, and the exact messages it was started with (for the prompt-equality
    # gate). All three are None until an arm actually starts.
    arm_queue: asyncio.Queue[Any] | None = None
    arm_task: asyncio.Task[Any] | None = None
    arm_messages: list[Message] | None = None
    arm_error: BaseException | None = None
    # One arm is either adopted or wasted, never both.
    arm_counted: bool = False

    def matches(self, final_text: str, threshold: float = 0.6) -> bool:
        final_tokens = set(tokenize(final_text))
        if not final_tokens or not self.tokens:
            return False
        overlap = len(final_tokens & self.tokens) / len(final_tokens)
        return overlap >= threshold


class Speculator:
    def __init__(
        self,
        *,
        enabled: bool = True,
        min_words: int = 4,
        stable_hits: int = 2,
        min_age_ms: int = 180,
        armer: Callable[[Speculation], Awaitable[bool]] | None = None,
        max_arms: int = 12,
        max_arms_per_turn: int = 3,
    ) -> None:
        self.enabled = enabled
        self.min_words = min_words
        self.stable_hits = stable_hits
        # A hypothesis that has been on screen for less than this is still
        # moving; arming it spends a provider stream on a guess the recogniser
        # is about to rewrite.
        self.min_age_ms = min_age_ms
        self.armer = armer
        # Cost ceiling. Every arm is a full provider stream, and a session
        # that arms and never adopts is paying for nothing; the cap bounds
        # that waste per session instead of trusting good behaviour. The
        # per-turn cap is the tighter one: a single utterance should not burn
        # a dozen streams as its partial churns.
        self.max_arms = max_arms
        self.max_arms_per_turn = max_arms_per_turn
        self._last_text = ""
        self._hits = 0
        self._first_seen_at = 0.0
        self._arms_this_turn = 0
        self._candidate: Speculation | None = None
        self._lock = asyncio.Lock()
        self.prepared = 0
        self.adopted = 0
        self.wasted = 0
        self.armed = 0
        self.arm_adopted = 0
        self.arm_wasted = 0

    def reset(self) -> None:
        """Drop the current hypothesis and stop paying for its arm."""
        self._last_text = ""
        self._hits = 0
        self._first_seen_at = 0.0
        self._arms_this_turn = 0
        candidate, self._candidate = self._candidate, None
        if candidate is not None:
            self.discard(candidate)

    async def note(self, partial: str) -> Speculation | None:
        if not self.enabled:
            return None
        text = " ".join(partial.split())
        if len(text.split()) < self.min_words:
            self._last_text = text
            self._hits = 0
            self._first_seen_at = 0.0
            return None
        if text == self._last_text:
            self._hits += 1
        else:
            self._last_text = text
            self._hits = 1
            self._first_seen_at = time.perf_counter()
            # The hypothesis moved on: whatever was armed for the old text
            # can no longer be adopted for this one, so stop paying for it.
            previous, self._candidate = self._candidate, None
            if previous is not None:
                self.discard(previous)
        # Stability is two independent conditions: the same text must be seen
        # ``stable_hits`` times *and* must have been on screen for a moment.
        # With the old ``stable_hits=1`` the first eligible partial armed
        # immediately, so every recogniser revision bought a full stream.
        if self._hits < self.stable_hits:
            return None
        if self.min_age_ms and (
            time.perf_counter() - self._first_seen_at < self.min_age_ms / 1000.0
        ):
            return None
        if self._candidate is not None and self._candidate.text == text:
            return self._candidate

        async with self._lock:
            if self._candidate is not None and self._candidate.text == text:
                return self._candidate
            result = await asyncio.to_thread(lookup, text)
            chunks: list[Chunk] = []
            if result.needs_depth:
                try:
                    chunks = await asyncio.wait_for(
                        asyncio.to_thread(search_docs, text, 3), timeout=0.5
                    )
                except Exception:
                    chunks = []
            spec = Speculation(text=text, tokens=set(tokenize(text)), result=result)
            spec.chunks = chunks
            self._candidate = spec
            self.prepared += 1
            if (
                self.armer is not None
                and self.armed < self.max_arms
                and self._arms_this_turn < self.max_arms_per_turn
                and result.confident
            ):
                self._arms_this_turn += 1
                spec.arm_task = asyncio.create_task(self._run_armer(spec))
            return spec

    async def _run_armer(self, spec: Speculation) -> None:
        """Invoke the armer; a speculative failure must never break a note()."""
        armer = self.armer
        if armer is None:
            return
        try:
            started = await armer(spec)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            spec.arm_error = error
            return
        if started:
            self.armed += 1

    def consume(self, final_text: str) -> Speculation | None:
        candidate = self._candidate
        self._candidate = None
        self.reset()
        if candidate is None:
            return None
        if candidate.matches(final_text) and self._evidence_matches(
            candidate, final_text
        ):
            candidate.used = True
            self.adopted += 1
            return candidate
        self.discard(candidate)
        self.wasted += 1
        return None

    @staticmethod
    def _evidence_matches(candidate: Speculation, final_text: str) -> bool:
        """The final transcript must route to the exact blocks the arm used."""
        try:
            fresh = lookup(final_text)
        except Exception:  # noqa: BLE001 - a failed check is a failed gate
            return False
        return {block.id for block in candidate.result.blocks} == {
            block.id for block in fresh.blocks
        }

    def discard(self, spec: Speculation) -> None:
        """Drop a speculation whose arm (if any) will never be replayed."""
        task = spec.arm_task
        if task is not None and not task.done():
            task.cancel()
        if spec.arm_queue is not None and not spec.arm_counted:
            spec.arm_counted = True
            self.arm_wasted += 1

    def mark_adopted(self, spec: Speculation) -> None:
        if spec.arm_queue is not None and not spec.arm_counted:
            spec.arm_counted = True
            self.arm_adopted += 1

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "prepared": self.prepared,
            "adopted": self.adopted,
            "wasted": self.wasted,
            "armed": self.armed,
            "arm_adopted": self.arm_adopted,
            "arm_wasted": self.arm_wasted,
        }

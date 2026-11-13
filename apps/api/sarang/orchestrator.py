"""The turn engine.

One object owns the whole response path: lookup, optional tool call, LLM stream,
segmentation, TTS and egress. The transport layer only feeds it text or audio and
listens to the bus.

The invariants this file is responsible for:

* Every turn has a generation. Work from an older generation is dropped at the
  LLM token, tool result, TTS chunk and egress boundaries — not just at the UI.
* The first speech segment leaves for TTS while the model is still writing.
* A slow path may speak a pre-rendered acknowledgement; a fast path never does.
* A provider failure degrades to the deterministic local answerer or the
  synthetic voice. It never leaves the visitor in silence.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from . import AGENT_NAME
from .config import REPO_ROOT, Settings
from .kb.lookup import LookupResult, lookup, strip_discourse
from .kb.render import render_context, render_evidence, render_overview
from .kb.retrieval import Chunk, search_docs
from .llm import build_llm
from .llm.base import (
    TOOL_SCHEMA,
    LLMError,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
)
from .llm.local import stream_local_answer
from .llm.prompt import build_messages
from .metrics import MetricsStore, TurnMarks, now
from .schemas import EventType, SessionState
from .segmenter import TextSegmenter, split_script
from .session import Session, SessionRegistry
from .speculative import Speculation, Speculator
from .tools import ToolRunner
from .tts import TTS, build_tts
from .tts.filler import FillerBank

log = logging.getLogger(__name__)

_END = object()


@dataclass
class _StreamError:
    error: BaseException


def filler_voice_tag(tts: TTS) -> str:
    """Identity of the audio a TTS arm produces; used to key cached fillers.

    A cached clip is only valid for the exact voice that rendered it, so the tag
    names everything that changes the audio: provider, model, voice and language.
    Without it, switching provider leaves the old voice saying "one moment" ahead
    of an answer spoken in the new one.
    """
    primary = getattr(tts, "primary", tts)
    voice = getattr(primary, "voice_id", "") or getattr(primary, "voice", "")
    return "-".join(
        str(part)
        for part in (
            tts.name,
            getattr(primary, "model", ""),
            voice,
            getattr(primary, "language", ""),
        )
        if part
    )


class OrderedSpeech:
    """Synthesise segments concurrently, play them strictly in order.

    Segment N+1 starts synthesising while segment N is still being heard, but its
    audio is not released until N has finished. That is the difference between
    "streaming" and "overlapping".
    """

    def __init__(
        self,
        tts: TTS,
        egress: Any,
        generation: int,
        turn_id: str,
        marks: TurnMarks,
    ) -> None:
        self.tts = tts
        self.egress = egress
        self.generation = generation
        self.turn_id = turn_id
        self.marks = marks
        self._inbox: asyncio.Queue[int | None] = asyncio.Queue()
        self._queues: dict[int, asyncio.Queue[np.ndarray | None]] = {}
        self._tasks: dict[int, asyncio.Task] = {}
        self._contexts: list[str] = []
        self._emitter: asyncio.Task | None = None
        self._counter = 0
        self._submitted = False
        self._first_span = False
        self._first_chunk = False
        self.finished = False
        self._cancelled = False
        self.errors = 0

    def start(self) -> None:
        if self._emitter is None:
            self._emitter = asyncio.create_task(self._emit_loop())

    def submit(self, text: str) -> None:
        if not text.strip() or self.finished:
            return
        index = self._counter
        self._counter += 1
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._queues[index] = queue
        context_id = f"{self.turn_id}-{index}"
        self._contexts.append(context_id)
        if not self._submitted:
            self._submitted = True
            # The first segment reaching TTS is the end of segmentation. It used
            # to be closed after the whole model stream finished, which reported
            # "the rest of the generation" under the name of the stage that
            # decides when speech can start.
            self.marks.exit_open("segment")
            self.marks.enter("tts_ttfa")
        self._tasks[index] = asyncio.create_task(self._synthesize(index, text, context_id, queue))
        self._inbox.put_nowait(index)

    async def _synthesize(
        self,
        index: int,
        text: str,
        context_id: str,
        queue: asyncio.Queue[np.ndarray | None],
    ) -> None:
        try:
            async for span in self.tts.synthesize(text, context_id=context_id, final=True):
                if span.pcm.size:
                    if not self._first_span:
                        self._first_span = True
                        self.marks.exit("tts_ttfa")
                        self.marks.mark_tts_first_span()
                    await queue.put(span.pcm)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - one bad segment must not kill the turn
            self.errors += 1
            log.warning("tts segment %s failed: %s", index, error)
        finally:
            await queue.put(None)

    async def _emit_loop(self) -> None:
        while True:
            index = await self._inbox.get()
            if index is None:
                return
            queue = self._queues.pop(index, None)
            if queue is None:
                continue
            while True:
                pcm = await queue.get()
                if pcm is None:
                    break
                if not self._first_chunk:
                    self._first_chunk = True
                    self.marks.enter("egress")
                self.egress.put(pcm, self.generation, self.turn_id)

    async def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self._inbox.put_nowait(None)
        if self._emitter is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._emitter
        if self._tasks:
            done = await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            _ = done
            self._tasks.clear()

    async def cancel(self) -> None:
        if getattr(self, "_cancelled", False):
            return
        self._cancelled = True
        self.finished = True
        for task in self._tasks.values():
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            self._tasks.clear()
        if self._emitter is not None:
            self._emitter.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._emitter
        await self._cancel_provider()

    async def _cancel_provider(self) -> None:
        client = getattr(self.tts, "primary", self.tts)
        cancel = getattr(client, "cancel", None)
        if cancel is None:
            return
        # Concurrently: one segment per context, and a provider that answers
        # slowly must not make the next of N cancellations queue behind it.
        await asyncio.gather(
            *(cancel(context_id) for context_id in self._contexts),
            return_exceptions=True,
        )


@dataclass
class TurnRuntime:
    session: Session
    turn_id: str
    generation: int
    marks: TurnMarks
    speech: OrderedSpeech
    segmenter: TextSegmenter
    answer: str = ""
    segments: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    first_token: bool = False
    filler_used: bool = False
    llm_started: float = 0.0
    spoken_words: int = 0
    truncated: bool = False


class Orchestrator:
    # A follow-up ("what were the numbers?", "how does that approval work?")
    # carries little topical vocabulary of its own. When the previous turn's
    # evidence still answers it, that evidence must travel with the question.
    FOLLOWUP_RE = re.compile(
        r"\b(it|that|this|those|these|the numbers|more about|tell me more|"
        r"what about|and the|same|there)\b",
        re.I,
    )

    def __init__(self, settings: Settings, registry: SessionRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry
        self.llm = build_llm(settings)
        self.tts = build_tts(settings)
        self.tools = ToolRunner()
        self.metrics = MetricsStore()
        cache_root = Path(settings.cache_dir) if settings.cache_dir else REPO_ROOT / "cache"
        self.filler = FillerBank(
            self.tts,
            cache_root / "fillers",
            enabled=settings.filler_enabled,
            voice_tag=filler_voice_tag(self.tts),
        )
        self.filler_after_s = max(0.35, settings.speculative_ms / 1000.0 or 0.6)
        self.ready = False
        self.started_at = time.time()
        self.detector = "energy"
        self.detector_silence_ms = 800
        self._active_speech: set[OrderedSpeech] = set()
        # Interrupt cleanup runs in the background: barge-in must not wait for
        # provider cancels. The set exists only so shutdown can drain them.
        self._cleanup_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ setup

    async def startup(self) -> None:
        from .kb.loader import load_kb
        from .kb.lookup import get_index
        from .kb.retrieval import LocalRetriever, PgRetriever, set_retriever

        kb = load_kb()
        get_index(kb)
        local = LocalRetriever()
        if self.settings.database_url:
            pg = PgRetriever(self.settings.database_url)
            if await pg.connect():
                with contextlib.suppress(Exception):
                    await pg.ingest(local.chunks)
                set_retriever(pg)
                self.settings.retrieval_backend = "pgvector"
            else:
                set_retriever(local)
                self.settings.retrieval_backend = "bm25-local (database unreachable)"
        else:
            set_retriever(local)
            self.settings.retrieval_backend = "bm25-local"
        self.ready = True
        self.filler.warm_background()
        # Load the detector once, at startup, so a missing model or a bad hash
        # is a startup warning rather than a silent per-connection fallback, and
        # so the effective window is on the record before the first question.
        from .audio.vad import VadConfig, build_vad

        probe = build_vad(
            VadConfig(
                silence_ms=self.settings.endpoint_silence_ms,
                min_speech_ms=self.settings.endpoint_min_speech_ms,
                hangover_ms=self.settings.endpoint_hangover_ms,
                silero_threshold=self.settings.vad_silero_threshold,
            ),
            mode=self.settings.vad_mode,
            model_path=self.settings.vad_silero_model or None,
        )
        self.detector = probe.detector
        self.detector_silence_ms = probe.config.silence_ms
        log.info(
            "sarang ready profile=%s asr=%s llm=%s tts=%s vad=%s/%dms blocks=%d",
            self.settings.profile.name,
            self.settings.asr_provider,
            self.settings.llm_provider,
            self.settings.tts_provider,
            self.detector,
            self.detector_silence_ms,
            len(kb.blocks),
        )

    async def shutdown(self) -> None:
        self.ready = False
        if self.registry is not None:
            await self.registry.close_all()
        for task in list(self._cleanup_tasks):
            task.cancel()
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            self._cleanup_tasks.clear()
        for speech in list(self._active_speech):
            await speech.cancel()
        await self.tts.close()
        if self.llm is not None:
            with contextlib.suppress(Exception):
                await self.llm.close()

    def create_session(self, client: str = "web") -> Session:
        if self.registry is None:
            from .bus import Bus

            self.registry = SessionRegistry(self.settings, Bus())
        session = self.registry.create(client)
        session.speculation = Speculator(
            enabled=self.settings.speculation_enabled,
            # Binds this session's arm to this engine; ``speculative_llm``
            # decides whether an arm exists at all.
            armer=partial(self._spec_arm, session)
            if self.settings.speculative_llm
            else None,
        )
        session.egress.on_first_audio = lambda generation: self._on_first_audio(
            session, generation
        )
        return session

    def _watch_playback(self, session: Session, generation: int) -> None:
        """Flip back to LISTENING when the egress has no audio left.

        Synthesis finishing is not playback finishing: the egress queue can
        still hold the rest of the answer. Transitioning early made both
        barge-in paths stop listening while the visitor was still hearing
        speech. This runs as its own task so the turn does not hold
        ``turn_lock`` for the duration of the answer.
        """
        task = asyncio.create_task(self._await_playback(session, generation))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _await_playback(self, session: Session, generation: int) -> None:
        try:
            await session.egress.wait_drained(generation, timeout=30.0)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - lifecycle must not raise into a task
            log.debug("playback drain wait failed", exc_info=True)
        if session.closed or session.generation != generation:
            return
        if session.state in {
            SessionState.SPEAKING,
            SessionState.THINKING,
            SessionState.TOOL_WORKING,
        }:
            session.set_state(SessionState.LISTENING)

    def _on_first_audio(self, session: Session, generation: int) -> None:
        marks = session.marks.get(generation)
        if marks is None or marks.first_audio is not None:
            return
        marks.mark_first_audio()
        if "egress" in marks.stage_enter:
            marks.exit("egress")
        # Audio is now actually leaving the process, which is the moment the
        # visitor hears something. The UI state should follow that, not the
        # earlier "we think we have an answer" state.
        if session.generation == generation and session.state not in {
            SessionState.ENDED,
            SessionState.ERROR,
        }:
            session.set_state(SessionState.SPEAKING)

    # ------------------------------------------------------------- life cycle

    async def greet(self, session: Session) -> None:
        if session.welcome_sent or session.closed:
            return
        session.welcome_sent = True
        text = (
            f"Hi, I'm {AGENT_NAME}. I can talk you through Sanket's work, background, "
            "and how he'd fit a role. What would you like to know?"
        )
        await self.speak_static(session, text, kind="greeting")

    async def speak_static(self, session: Session, text: str, *, kind: str = "notice") -> None:
        if session.closed or not text.strip():
            return
        async with session.turn_lock:
            turn_id, generation = session.begin_turn()
            session.active_turn = turn_id
            marks = TurnMarks(turn_id)
            marks.source = kind
            marks.mark_end_of_speech()
            session.marks[generation] = marks
            session.egress.set_generation(generation)
            session.set_state(SessionState.SPEAKING, "Greeting")
            speech = OrderedSpeech(
                self.tts, session.egress, generation, turn_id, marks
            )
            self._active_speech.add(speech)
            speech.start()
            try:
                for segment in split_script(text):
                    speech.submit(segment)
                await speech.finish()
                session.publish(
                    EventType.ASSISTANT_FINAL,
                    {"text": text},
                    turn_id=turn_id,
                    generation=generation,
                )
            finally:
                self._active_speech.discard(speech)
                if session.generation == generation and not session.closed:
                    self._watch_playback(session, generation)

    def _interrupt_now(self, session: Session, reason: str) -> int:
        """The part of an interruption that must not wait for anybody.

        Generation bump, egress gate, turn cancellation and the EPOCH event are
        all synchronous. The browser can flush on EPOCH immediately, and every
        frame from the old generation is rejected from this instant. Only
        provider cleanup (TTS ``cancel`` calls) is asynchronous, and it used to
        hold EPOCH back - which made barge-in feel a full provider round-trip
        slower than it was.
        """
        session.interruptions += 1
        session.generation += 1
        generation = session.generation
        session.egress.interrupt(generation)
        active = session.marks.get(generation - 1)
        if active is not None:
            active.cancelled = True
        session.cancel_active_turn()
        speculation = session.speculation
        if speculation is not None:
            # A barge-in rewrites the conversation: an arm warmed for the
            # interrupted turn must never be adopted by the next one.
            speculation.reset()
        session.publish(
            EventType.EPOCH,
            {"generation": generation, "reason": reason},
            generation=generation,
        )
        session.set_state(SessionState.INTERRUPTED, reason)
        return generation

    async def _cancel_active_speech(self) -> None:
        """Cancel every active speech object concurrently, with a bound.

        Sequential cancellation meant an interruption cost the sum of every
        provider's cancel latency. The gather is bounded because a wedged
        provider must not keep the old generation's tasks alive either.
        """
        speeches = list(self._active_speech)
        if not speeches:
            return
        with contextlib.suppress(TimeoutError, Exception):
            await asyncio.wait_for(
                asyncio.gather(
                    *(speech.cancel() for speech in speeches),
                    return_exceptions=True,
                ),
                timeout=1.0,
            )

    async def interrupt(self, session: Session, reason: str = "barge-in") -> None:
        if session.closed:
            return
        generation = self._interrupt_now(session, reason)
        await self._cancel_active_speech()
        asyncio.create_task(self._resume_listening(session, generation))

    def interrupt_soon(self, session: Session, reason: str = "barge-in") -> None:
        """Interrupt from the audio ingest path without awaiting cleanup.

        ``_begin_utterance`` runs inside ``push``: every millisecond spent here
        is microphone audio that is not yet in the recogniser.
        """
        if session.closed:
            return
        generation = self._interrupt_now(session, reason)
        cleanup = asyncio.create_task(self._finish_interrupt(session, generation))
        self._cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._cleanup_tasks.discard)

    async def _finish_interrupt(self, session: Session, generation: int) -> None:
        await self._cancel_active_speech()
        await self._resume_listening(session, generation)

    async def _resume_listening(self, session: Session, generation: int) -> None:
        await asyncio.sleep(0.12)
        if session.closed or session.generation != generation:
            return
        if session.state == SessionState.INTERRUPTED:
            session.set_state(SessionState.LISTENING)

    # ------------------------------------------------------------------ turns

    async def run_turn(
        self,
        session: Session,
        text: str,
        *,
        marks: TurnMarks | None = None,
        source: str = "text",
        turn_id: str | None = None,
    ) -> None:
        text = " ".join(text.split())
        if not text or session.closed:
            return

        async with session.turn_lock:
            if session.closed:
                return
            started_turn, generation = session.begin_turn()
            turn_id = turn_id or started_turn
            session.active_turn = turn_id
            turn_marks = marks or TurnMarks(turn_id)
            turn_marks.turn_id = turn_id
            turn_marks.source = source
            if turn_marks.end_of_speech is None:
                turn_marks.mark_end_of_speech()
            session.marks[generation] = turn_marks
            session.egress.set_generation(generation)

            speech = OrderedSpeech(self.tts, session.egress, generation, turn_id, turn_marks)
            self._active_speech.add(speech)
            runtime = TurnRuntime(
                session=session,
                turn_id=turn_id,
                generation=generation,
                marks=turn_marks,
                speech=speech,
                segmenter=TextSegmenter(),
            )
            speech.start()
            try:
                await self._run_turn(runtime, text, source=source)
            except asyncio.CancelledError:
                turn_marks.cancelled = True
                if runtime.answer.strip():
                    # Preserve what was actually said so a follow-up can refer
                    # to it, while marking that the sentence was cut off.
                    session.remember("assistant", runtime.answer.strip() + " …")
                self._record(runtime)
                # Provider cleanup runs in the background: awaiting it here let
                # a slow TTS cancel keep turn_lock held, delaying the next turn.
                cleanup = asyncio.create_task(speech.cancel())
                self._cleanup_tasks.add(cleanup)
                cleanup.add_done_callback(self._cleanup_tasks.discard)
                raise
            except Exception as error:  # noqa: BLE001
                log.exception("turn failed: %s", error)
                turn_marks.cancelled = True
                await speech.cancel()
                self._record(runtime)
                if session.generation == generation and not session.closed:
                    session.error("That turn failed. Ask again when you're ready.", recoverable=True)
            finally:
                self._active_speech.discard(speech)
                if session.turn_task is asyncio.current_task():
                    session.turn_task = None

    def _record(self, runtime: TurnRuntime) -> None:
        payload = self.metrics.record(runtime.marks)
        runtime.session.publish(
            EventType.METRICS_TURN,
            payload,
            turn_id=runtime.turn_id,
            generation=runtime.generation,
        )

    async def _run_turn(self, runtime: TurnRuntime, text: str, *, source: str) -> None:
        session = runtime.session
        generation = runtime.generation

        session.set_state(SessionState.THINKING, "Understanding the question")
        if source == "text":
            session.publish(
                EventType.TRANSCRIPT_FINAL,
                {"text": text, "source": "text"},
                turn_id=runtime.turn_id,
                generation=generation,
            )

        # The card has shown what the visitor said (voice published the raw
        # transcript before the turn started); the router, the retriever and
        # the model get the question without the discourse wrapped around it.
        text = strip_discourse(text)

        # ---------------------------------------------------------- tier-1 lookup
        runtime.marks.enter("lookup")
        speculator = session.speculation
        speculation = speculator.consume(text) if speculator is not None else None
        previous = session.last_lookup
        result = speculation.result if speculation else lookup(text)
        if (
            previous is not None
            and previous is not result
            and previous.blocks
            and self.FOLLOWUP_RE.search(text)
        ):
            # "It"/"that"/"the numbers" refers to the previous subject. Lead with
            # the previous evidence, keep any specific current evidence, drop
            # generic skills blocks that would drown the subject.
            current = [
                block
                for block in result.blocks
                if block not in previous.blocks
                and not (block.kind == "skills" and previous.top
                         and previous.top.kind in {"project", "experience", "research", "education"})
            ]
            merged: list = []
            for block in [*previous.blocks, *current]:
                if block not in merged:
                    merged.append(block)
            result.blocks = merged[:4]
            result.confident = True
        runtime.marks.exit("lookup")
        session.last_lookup = result
        session.publish(
            EventType.EVIDENCE,
            {
                "blocks": render_evidence(result.blocks),
                "lookup": result.to_payload(),
                "speculative": bool(speculation),
            },
            turn_id=runtime.turn_id,
            generation=generation,
        )
        session.remember("user", text)

        # ------------------------------------------------------------- tier-2
        chunks: list[Chunk] = list(speculation.chunks) if speculation else []
        if result.needs_depth and not chunks:
            session.set_state(SessionState.TOOL_WORKING, "Searching the case studies")
            chunks = await self._retrieve(runtime, text)
            session.publish(
                EventType.EVIDENCE,
                {"chunks": [chunk.to_payload() for chunk in chunks]},
                turn_id=runtime.turn_id,
                generation=generation,
            )

        if result.reason == "generic project overview":
            context = render_overview(result.blocks)
        else:
            context = render_context(result.blocks)
        if chunks:
            evidence = "\n".join(
                f"[{chunk.doc_id}] {chunk.title}: {chunk.text}" for chunk in chunks
            )[:1_300]
            context = f"{context}\n\nCASE STUDY EVIDENCE:\n{evidence}"

        runtime.marks.enter("context")
        messages = build_messages(
            text,
            context,
            history=session.history[:-1],
            lexicon=None,
        )
        runtime.marks.exit("context")

        # ------------------------------------------------------- speculative arm
        # Replay the stream that started while the visitor was still talking,
        # but only if it was answering exactly this turn's question. The
        # evidence gate already ran in consume(); here the rest of the prompt
        # must be identical - system prompt with its context and the whole
        # conversation. Only the question text may differ, bounded by the
        # token and evidence gates, because the final transcript can add words
        # the partial had not heard yet.
        arm: Speculation | None = None
        if speculation is not None and speculator is not None:
            if (
                speculation.arm_queue is not None
                and speculation.arm_error is None
                and speculation.arm_messages is not None
                and not result.needs_depth
                and messages[:-1] == speculation.arm_messages[:-1]
            ):
                arm = speculation
            else:
                speculator.discard(speculation)

        # ---------------------------------------------------------------- LLM
        runtime.llm_started = now()
        runtime.marks.enter("llm_ttft")
        runtime.marks.enter("llm_total")
        tool_used = False
        try:
            if arm is not None:
                try:
                    answer, tool_calls = await self._replay(runtime, arm)
                    assert speculator is not None
                    speculator.mark_adopted(arm)
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001
                    if runtime.first_token or runtime.segments:
                        # Mid-answer: degrade exactly like a live stream
                        # failure, through the outer handler below.
                        raise
                    # Nothing was spoken yet: the arm failed early, so pay
                    # for a fresh stream rather than the local answerer.
                    log.warning("speculative arm failed early: %s", error)
                    speculator.discard(arm)
                    answer, tool_calls = await self._stream_once(
                        runtime, messages, allow_tools=True
                    )
            else:
                answer, tool_calls = await self._stream_once(
                    runtime, messages, allow_tools=True
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            log.warning("llm %s failed, using local answerer: %s", self.settings.llm_provider, error)
            session.notice("Language model unavailable; answering from the fact base.", "warning")
            answer, tool_calls = await self._stream_fallback(runtime, text, result, chunks)

        if tool_calls and not tool_used:
            tool_used = True
            call = tool_calls[0]
            session.set_state(SessionState.TOOL_WORKING, "Retrieving the case study")
            tool_result = await self.tools.execute(call.name, call.id, call.arguments)
            session.publish(
                EventType.EVIDENCE,
                {"tool": tool_result.to_payload()},
                turn_id=runtime.turn_id,
                generation=generation,
            )
            if session.generation != generation:
                return
            continuation = [
                *messages,
                Message(role="assistant", content=answer or "Let me check the case study."),
                Message(
                    role="user",
                    content=(
                        f"Tool result for the original question: {tool_result.content}\n\n"
                        "Answer the original question now using this evidence and the FACTS. "
                        "Do not mention the tool or the search."
                    ),
                ),
            ]
            session.set_state(SessionState.THINKING, "Reasoning over the evidence")
            more, _ = await self._stream_once(runtime, continuation, allow_tools=False)
            if more.strip():
                answer = (answer + " " + more).strip()

        runtime.marks.exit("llm_total")

        if not answer.strip():
            answer, _ = await self._stream_fallback(runtime, text, result, chunks)

        await runtime.speech.finish()

        if session.generation != generation:
            return

        final_text = answer.strip()
        if runtime.truncated:
            # Publish what was actually said, so the transcript cannot claim a
            # sentence the visitor never heard.
            final_text = " ".join(runtime.segments).strip() or final_text

        if final_text:
            session.remember("assistant", final_text)
            session.publish(
                EventType.ASSISTANT_FINAL,
                {"text": final_text},
                turn_id=runtime.turn_id,
                generation=generation,
            )
        self._record(runtime)
        if session.generation == generation and not session.closed:
            # Not LISTENING yet: the egress may still be playing the answer.
            self._watch_playback(session, generation)

    # ------------------------------------------------------------- streaming

    async def _spec_arm(self, session: Session, spec: Speculation) -> bool:
        """Stream an answer for a stable partial while the visitor talks.

        The result is never spoken early: it fills ``spec.arm_queue`` and is
        replayed through the normal consume path only when the turn's gates
        pass. Returns whether a stream was actually started - declining (no
        provider, unconfident or out-of-scope evidence) costs nothing.

        The messages are built exactly as ``_run_turn`` builds them, except
        that the question is the stable partial, since the final transcript
        does not exist yet. Adoption later requires every other message to be
        byte-identical, so a divergence here simply means no adoption.
        """
        if self.llm is None or not self.settings.speculative_llm or session.closed:
            return False
        result = spec.result
        if (
            not result.confident
            or result.needs_depth
            or result.blocked
            or not result.blocks
            or result.reason in {"off topic", "task request"}
        ):
            return False
        if result.reason == "generic project overview":
            context = render_overview(result.blocks)
        else:
            context = render_context(result.blocks)
        # The turn appends the current question with history[:-1]; at arm time
        # the question is not in history yet, so the whole history goes in.
        messages = build_messages(
            spec.text,
            context,
            history=list(session.history),
            lexicon=None,
        )
        spec.arm_messages = messages
        queue: asyncio.Queue[Any] = asyncio.Queue()
        spec.arm_queue = queue
        stream = self.llm.stream(
            messages,
            tools=None,
            max_tokens=self.settings.max_answer_tokens,
            temperature=0.25,
        )

        async def pump() -> None:
            try:
                async for event in stream:
                    await queue.put(event)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                spec.arm_error = error
                queue.put_nowait(_StreamError(error))
            finally:
                queue.put_nowait(_END)

        spec.arm_task = asyncio.create_task(pump())
        return True

    async def _stream_once(
        self,
        runtime: TurnRuntime,
        messages: list[Message],
        *,
        allow_tools: bool,
    ) -> tuple[str, list[ToolCall]]:
        if self.llm is None:
            raise LLMError("no language model configured")

        stream = self.llm.stream(
            messages,
            tools=TOOL_SCHEMA if allow_tools else None,
            max_tokens=self.settings.max_answer_tokens,
            temperature=0.25,
        )
        return await self._consume(runtime, stream, allow_tools=allow_tools)

    async def _stream_fallback(
        self,
        runtime: TurnRuntime,
        text: str,
        result: LookupResult,
        chunks: list[Chunk],
    ) -> tuple[str, list[ToolCall]]:
        stream = stream_local_answer(text, result, chunks)
        return await self._consume(runtime, stream, allow_tools=False)

    async def _consume(
        self,
        runtime: TurnRuntime,
        stream: AsyncIterator[LLMEvent],
        *,
        allow_tools: bool,
    ) -> tuple[str, list[ToolCall]]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        pump = asyncio.create_task(self._pump(stream, queue))
        return await self._drain(runtime, queue, pump, allow_tools=allow_tools)

    async def _replay(
        self,
        runtime: TurnRuntime,
        speculation: Speculation,
    ) -> tuple[str, list[ToolCall]]:
        """Drain an arm that streamed while the visitor was still talking.

        No provider request is made here - that is where the saved latency
        comes from. The queue may still be filling; ``_drain`` reads it at
        the same pace either way.
        """
        queue = speculation.arm_queue
        if queue is None:
            raise LLMError("speculative arm has no queue")
        pump = speculation.arm_task
        return await self._drain(runtime, queue, pump, allow_tools=False)

    async def _drain(
        self,
        runtime: TurnRuntime,
        queue: asyncio.Queue[Any],
        pump: asyncio.Task[Any] | None,
        *,
        allow_tools: bool,
    ) -> tuple[str, list[ToolCall]]:
        session = runtime.session
        generation = runtime.generation
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        filler_used = runtime.filler_used
        started = now()

        try:
            while True:
                if session.generation != generation or session.closed:
                    raise asyncio.CancelledError()
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.05)
                except TimeoutError:
                    for segment in runtime.segmenter.due():
                        self._submit(runtime, segment)
                    if runtime.truncated:
                        break
                    if (
                        not runtime.first_token
                        and not filler_used
                        and self.filler.enabled
                        and (now() - started) >= self.filler_after_s
                    ):
                        if await self._play_cached_filler(runtime):
                            filler_used = True
                    continue

                if event is _END:
                    break
                if isinstance(event, _StreamError):
                    raise event.error
                if isinstance(event, TextDelta):
                    if not runtime.first_token:
                        runtime.first_token = True
                        runtime.marks.exit("llm_ttft")
                        runtime.marks.enter("segment")
                    delta = event.text
                    text_parts.append(delta)
                    runtime.answer += delta
                    session.publish(
                        EventType.ASSISTANT_DELTA,
                        {"text": delta},
                        turn_id=runtime.turn_id,
                        generation=generation,
                    )
                    for segment in runtime.segmenter.push(delta):
                        self._submit(runtime, segment)
                    for segment in runtime.segmenter.due():
                        self._submit(runtime, segment)
                    if runtime.truncated:
                        break
                elif isinstance(event, ToolCall) and allow_tools:
                    tool_calls.append(event)
                # Done carries usage; the metrics we publish are wall-clock and
                # do not depend on provider token accounting.
        finally:
            if pump is not None and not pump.done():
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pump

        for segment in runtime.segmenter.flush():
            if runtime.truncated:
                break
            self._submit(runtime, segment)
        return "".join(text_parts), tool_calls

    async def _pump(
        self, stream: AsyncIterator[LLMEvent], queue: asyncio.Queue[Any]
    ) -> None:
        try:
            async for event in stream:
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            await queue.put(_StreamError(error))
        finally:
            await queue.put(_END)

    def _submit(self, runtime: TurnRuntime, segment: str) -> None:
        if not segment.strip():
            return
        words = len(segment.split())
        limit = self.settings.max_spoken_words
        if runtime.segments and runtime.spoken_words + words > limit:
            # Hard stop, at a sentence boundary. The prompt asks for 35 words; this
            # is what happens when the model ignores it. Never cut mid-sentence.
            runtime.truncated = True
            return
        runtime.spoken_words += words
        runtime.segments.append(segment)
        runtime.speech.submit(segment)

    async def _retrieve(self, runtime: TurnRuntime, text: str) -> list[Chunk]:
        """Search with a short grace period, then acknowledge only if still slow."""
        task = asyncio.create_task(asyncio.to_thread(search_docs, text, 3))
        try:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            if not done:
                await self._play_cached_filler(runtime)
            chunks = await asyncio.wait_for(task, timeout=0.65)
            return list(chunks)
        except Exception:
            task.cancel()
            return []

    async def _play_cached_filler(self, runtime: TurnRuntime) -> bool:
        if runtime.filler_used or not self.filler.enabled:
            return False
        pcm = await self.filler.get(timeout=0.02)
        if pcm is None or not pcm.size:
            return False
        runtime.session.egress.put(pcm, runtime.generation, runtime.turn_id)
        runtime.marks.filler_used = True
        runtime.filler_used = True
        return True


__all__ = ["Orchestrator", "OrderedSpeech", "TurnRuntime"]

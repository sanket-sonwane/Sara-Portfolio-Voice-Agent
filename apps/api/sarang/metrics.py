"""Latency instrumentation.

Every stage records a monotonic timestamp at entry and exit. For a voice turn
the origin is the **last speech frame** the detector heard, because that is the
start of what the visitor waits through. For a text turn it is the request.
``timeline()`` puts every stage on that one axis, which is the view a report
should read: endpoint, ASR, model, TTS and egress as disjoint segments rather
than numbers measured on different clocks.

Marks are kept per turn and aggregated in a ring buffer, so the diagnostics
overlay can show p50/p90/p95 without a metrics backend.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field


def now() -> float:
    return time.perf_counter()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


@dataclass
class TurnMarks:
    """Stage durations for one turn, in milliseconds from the turn origin."""

    turn_id: str
    origin: float = field(default_factory=now)
    stage_enter: dict[str, float] = field(default_factory=dict)
    stage_exit: dict[str, float] = field(default_factory=dict)
    first_audio: float | None = None
    end_of_speech: float | None = None
    cancelled: bool = False
    filler_used: bool = False
    source: str = ""
    # Three different events that used to share one name:
    #   tts_first_span       - first synthesized PCM received from the provider
    #   first_audio          - first frame the egress selected for transmission
    #   transport_first_frame- first frame actually written to the transport
    tts_first_span: float | None = None
    answer_first_audio: float | None = None
    transport_first_frame: float | None = None

    def enter(self, stage: str) -> None:
        self.stage_enter[stage] = now()

    def exit(self, stage: str) -> float:
        start = self.stage_enter.get(stage, self.origin)
        self.stage_exit[stage] = now()
        return (self.stage_exit[stage] - start) * 1000.0

    def exit_open(self, stage: str) -> float | None:
        """Close a stage only if it is open.

        A stage that was never entered has no start, and ``exit`` would silently
        measure it from the turn origin, which reports a plausible-looking but
        meaningless number. Callers that may or may not have opened a stage use
        this instead.
        """
        if stage not in self.stage_enter or stage in self.stage_exit:
            return None
        return self.exit(stage)

    def mark_first_audio(self) -> float:
        if self.first_audio is None:
            self.first_audio = now()
        return (self.first_audio - self.origin) * 1000.0

    def mark_end_of_speech(self, at: float | None = None) -> None:
        """Stamp the last speech frame, not the moment the window expired.

        The turn origin for a voice turn is the *last frame the detector heard
        as speech*. Everything the visitor waits through - the endpoint window,
        ASR finalisation, retrieval, the model, TTS, egress - is measured from
        there, so ``first_audio_ms`` is the honest last-speech-to-audible number
        and ``endpoint`` is not double-counted.
        """
        self.end_of_speech = at if at is not None else now()

    def mark_endpoint(self, duration_ms: float) -> None:
        """Record the silence window observed before the turn was declared over.

        The window starts at the last speech frame (the turn origin) and ends
        when the detector declared the turn over, so it is written forward from
        the origin rather than back-dated from it. Keeping both on the same
        axis is what stops an endpoint delay from being reported as ASR latency.
        """
        if duration_ms <= 0:
            return
        self.stage_enter["endpoint"] = self.origin
        self.stage_exit["endpoint"] = self.origin + duration_ms / 1000.0

    def offsets(self) -> dict[str, float]:
        return {
            stage: round((stamp - self.origin) * 1000.0, 1)
            for stage, stamp in self.stage_exit.items()
        }

    def timeline(self) -> dict[str, float]:
        """Every stage exit on one axis: milliseconds since the last speech frame.

        This is the view a latency report should read. ``offsets()`` is relative
        to the turn origin, which for a text turn is the request; for a voice
        turn the origin *is* the last speech frame, so the two coincide.
        """
        if self.end_of_speech is None:
            return self.offsets()
        return {
            stage: round((stamp - self.end_of_speech) * 1000.0, 1)
            for stage, stamp in self.stage_exit.items()
        }

    def durations(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for stage, end in self.stage_exit.items():
            start = self.stage_enter.get(stage, self.origin)
            out[stage] = round((end - start) * 1000.0, 1)
        return out

    def to_payload(self) -> dict:
        payload: dict[str, object] = {
            "turn_id": self.turn_id,
            "durations_ms": self.durations(),
            "offsets_ms": self.offsets(),
            "timeline_ms": self.timeline(),
            "cancelled": self.cancelled,
            "filler_used": self.filler_used,
            "source": self.source,
        }
        if self.tts_first_span is not None:
            payload["tts_first_span_ms"] = round(
                (self.tts_first_span - self.origin) * 1000.0, 1
            )
            if self.end_of_speech is not None:
                payload["eos_to_tts_first_span_ms"] = round(
                    (self.tts_first_span - self.end_of_speech) * 1000.0, 1
                )
        if self.first_audio is not None:
            payload["egress_first_frame_ms"] = round(
                (self.first_audio - self.origin) * 1000.0, 1
            )
            if self.end_of_speech is not None:
                payload["eos_to_egress_first_frame_ms"] = round(
                    (self.first_audio - self.end_of_speech) * 1000.0, 1
                )
            # Deprecated names, kept for one release so stored results and the
            # existing diagnostics overlay stay readable. They mean "first
            # frame the egress selected", not "the visitor heard it".
            payload["first_audio_ms"] = payload["egress_first_frame_ms"]
            if self.end_of_speech is not None:
                payload["eos_to_first_audio_ms"] = payload["eos_to_egress_first_frame_ms"]
        if self.transport_first_frame is not None:
            payload["transport_first_frame_ms"] = round(
                (self.transport_first_frame - self.origin) * 1000.0, 1
            )
            if self.end_of_speech is not None:
                payload["eos_to_transport_first_frame_ms"] = round(
                    (self.transport_first_frame - self.end_of_speech) * 1000.0, 1
                )
        if self.answer_first_audio is not None:
            payload["answer_first_audio_ms"] = round(
                (self.answer_first_audio - self.origin) * 1000.0, 1
            )
            if self.end_of_speech is not None:
                payload["eos_to_answer_audio_ms"] = round(
                    (self.answer_first_audio - self.end_of_speech) * 1000.0, 1
                )
        return payload

    def mark_tts_first_span(self) -> None:
        if self.tts_first_span is None:
            self.tts_first_span = now()
        # ``answer_first_audio`` used to be stamped here by the emitter; keep
        # the alias pointing at the same event, named for what it is.
        self.mark_answer_first_audio()

    def mark_answer_first_audio(self) -> None:
        if self.answer_first_audio is None:
            self.answer_first_audio = self.tts_first_span or now()

    def mark_transport_first_frame(self) -> None:
        if self.transport_first_frame is None:
            self.transport_first_frame = now()


# `tts_ttfa` and `egress` split one provider-side wait, and the split is not
# meaningful to the product. `tts_ttfa` ends when the first synthesized span
# arrives; `egress` then runs until a span carrying a phoneme arrives, because
# the first span is dead air and is discarded. Read `egress` alone and it looks
# like framework overhead, which it is not: the gateway's handoff is under 10 ms
# (see `test_first_frame_of_a_generation_is_not_paced`). What the visitor
# experiences is the sum, "TTS request -> first audible sample".
STAGE_ORDER = (
    "endpoint",
    "asr_finalize",
    "asr_final",
    "lookup",
    "context",
    "llm_ttft",
    "llm_total",
    "segment",
    "tts_ttfa",
    "tts_total",
    "egress",
    "first_audio",
    "egress_first_frame",
    "transport_first_frame",
    "tts_first_span",
    "answer_first_audio",
)


class MetricsStore:
    """Rolling per-stage distributions for the diagnostics panel."""

    def __init__(self, capacity: int = 200) -> None:
        self.capacity = capacity
        self.turns: deque[dict] = deque(maxlen=capacity)
        self.samples: dict[str, deque[float]] = {stage: deque(maxlen=capacity) for stage in STAGE_ORDER}

    def record(self, marks: TurnMarks) -> dict:
        payload = marks.to_payload()
        for stage, value in payload.get("durations_ms", {}).items():  # type: ignore[union-attr]
            if stage in self.samples:
                self.samples[stage].append(float(value))
        first_audio = payload.get("first_audio_ms")
        if isinstance(first_audio, (int, float)) and first_audio > 0:
            self.samples["first_audio"].append(float(first_audio))
        for key in ("egress_first_frame_ms", "transport_first_frame_ms", "tts_first_span_ms"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and value > 0:
                stage = key.removesuffix("_ms")
                self.samples[stage].append(float(value))
        answer_first_audio = payload.get("answer_first_audio_ms")
        if isinstance(answer_first_audio, (int, float)) and answer_first_audio > 0:
            self.samples["answer_first_audio"].append(float(answer_first_audio))
        self.turns.append(payload)
        return payload

    def summary(self) -> dict:
        stages = {}
        for stage, values in self.samples.items():
            data = list(values)
            if not data:
                continue
            stages[stage] = {
                "n": len(data),
                "p50": round(percentile(data, 50), 1),
                "p90": round(percentile(data, 90), 1),
                "p95": round(percentile(data, 95), 1),
                "mean": round(statistics.fmean(data), 1),
                "max": round(max(data), 1),
            }
        return {"stages": stages, "turns": list(self.turns)[-20:]}

    def reset(self) -> None:
        self.turns.clear()
        for values in self.samples.values():
            values.clear()

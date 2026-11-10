"""End-to-end voice-turn latency: PCM in -> ASR -> LLM -> TTS -> PCM out.

This is the number the product promises, measured through the real gateway over
the same HTTP/WebSocket surface the browser uses. It starts the gateway in this
process (or targets ``--url``), opens a session, waits for the greeting, streams
a 16 kHz question into ``/ws/audio`` at real-time pace, and timestamps the first
non-silent frame that comes back.

Three totals are reported, because they answer different questions:

* **client total** — last speech frame sent -> first non-silent frame received.
  This is what the visitor experiences, and it includes the endpointing silence.
* **first audible** (``eos_to_first_audio_ms``) — VAD end-of-speech -> first
  audio. May be a cached acknowledgement on a slow turn.
* **first answer audio** (``eos_to_answer_audio_ms``) — VAD end-of-speech ->
  first real answer audio. Excludes fillers.

The component breakdown comes from ``metrics.turn`` and is measured on the
server: ASR finalisation, lookup, context assembly, LLM TTFT, first-segment
segmentation, TTS time-to-first-audio and egress.

Usage::

    uv run python -m bench.probe.e2e_turn --runs 5
    uv run python -m bench.probe.e2e_turn --url http://127.0.0.1:8000 --runs 5
    uv run python -m bench.probe.e2e_turn --audio bench/audio/question-16k.wav
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from bench.probe.common import (
    AUDIO_DIR,
    REPO_ROOT,
    Result,
    chunks,
    format_table,
    load_env,
    load_wav_16k,
    now,
    summarise,
)

DEFAULT_AUDIO = AUDIO_DIR / "question-16k.wav"
SPEECH_RATE = 16_000
SEND_CHUNK_MS = 100
TRAILING_SILENCE_MS = 1_500


# --------------------------------------------------------------------------- server


async def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def start_server() -> tuple[Any, Any, str]:
    """Start the gateway in this loop and return (server, task, base_url)."""
    sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
    import uvicorn
    from sarang.app import create_app

    port = await _free_port()
    config = uvicorn.Config(
        create_app(),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    deadline = now() + 30.0
    while not server.started:
        if task.done():
            raise RuntimeError("gateway exited during startup")
        if now() > deadline:
            raise RuntimeError("gateway did not start within 30 s")
        await asyncio.sleep(0.05)
    return server, task, f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------- ws


class EventStream:
    """Collects ``/ws/events`` envelopes and exposes small wait helpers."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.states: list[tuple[float, str]] = []
        self.transcript_final: dict[str, Any] | None = None
        self.metrics_turn: dict[str, Any] | None = None
        self.assistant_final: dict[str, Any] | None = None
        self._changed = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def run(self, ws) -> None:
        async for raw in ws:
            try:
                event = json.loads(raw)
            except (TypeError, ValueError):
                continue
            self.events.append(event)
            kind = event.get("type")
            if kind == "state":
                self.states.append((now(), str(event.get("payload", {}).get("state", ""))))
            elif kind == "transcript.final":
                self.transcript_final = event
            elif kind == "metrics.turn":
                self.metrics_turn = event
            elif kind == "assistant.final":
                self.assistant_final = event
            self._changed.set()
            self._changed = asyncio.Event()

    async def wait_state(self, state: str, timeout: float = 15.0) -> None:
        deadline = now() + timeout
        while now() < deadline:
            if any(value == state for _, value in self.states):
                return
            wait = self._changed.wait()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wait, timeout=min(0.2, deadline - now()))
        raise TimeoutError(f"state {state!r} not seen within {timeout}s")

    async def wait_metrics(self, timeout: float = 30.0) -> dict[str, Any]:
        deadline = now() + timeout
        while now() < deadline:
            if self.metrics_turn is not None:
                return self.metrics_turn
            wait = self._changed.wait()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wait, timeout=min(0.2, deadline - now()))
        raise TimeoutError(f"metrics.turn not seen within {timeout}s")

    async def wait_transcript(self, timeout: float = 15.0) -> dict[str, Any]:
        deadline = now() + timeout
        while now() < deadline:
            if self.transcript_final is not None:
                return self.transcript_final
            wait = self._changed.wait()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wait, timeout=min(0.2, deadline - now()))
        raise TimeoutError(f"transcript.final not seen within {timeout}s")


class AudioStream:
    """Receives paced 20 ms frames and timestamps the first non-silent one."""

    def __init__(self) -> None:
        self.frames: list[tuple[float, bool]] = []
        self._task: asyncio.Task | None = None

    async def run(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, str):
                continue
            samples = np.frombuffer(raw, dtype="<i2")
            self.frames.append((now(), bool(samples.size and np.any(samples))))

    def last_voice(self) -> float | None:
        for stamp, voiced in reversed(self.frames):
            if voiced:
                return stamp
        return None

    def first_voice_after(self, stamp: float) -> float | None:
        for at, voiced in self.frames:
            if at > stamp and voiced:
                return at
        return None


# -------------------------------------------------------------------------- turn


async def run_one(base_url: str, samples: np.ndarray) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        response = await client.post("/api/session", json={"client": "bench"})
        response.raise_for_status()
        session = response.json()
        session_id = session["session_id"]
        token = session["token"]

    query = f"session_id={session_id}&token={token}"
    events = EventStream()
    audio = AudioStream()

    from websockets.asyncio.client import connect as ws_connect

    async with ws_connect(f"{base_url.replace('http', 'ws')}/ws/events?{query}") as events_ws:
        events._task = asyncio.create_task(events.run(events_ws))
        async with ws_connect(
            f"{base_url.replace('http', 'ws')}/ws/audio?{query}&sr={SPEECH_RATE}"
        ) as audio_ws:
            audio._task = asyncio.create_task(audio.run(audio_ws))
            try:
                # Wait for the greeting to finish so its tail is not mistaken
                # for the answer. The state machine goes speaking -> listening.
                await events.wait_state("speaking", timeout=20.0)
                await events.wait_state("listening", timeout=20.0)
                quiet_deadline = now() + 3.0
                while now() < quiet_deadline:
                    last = audio.last_voice()
                    if last is not None and now() - last > 0.35:
                        break
                    await asyncio.sleep(0.05)

                for piece in chunks(samples, SEND_CHUNK_MS, SPEECH_RATE):
                    await audio_ws.send(piece)
                    await asyncio.sleep(SEND_CHUNK_MS / 1000.0)
                last_speech = now()

                silence = np.zeros(
                    int(SPEECH_RATE * TRAILING_SILENCE_MS / 1000.0), dtype=np.int16
                )
                for piece in chunks(silence, SEND_CHUNK_MS, SPEECH_RATE):
                    await audio_ws.send(piece)
                    await asyncio.sleep(SEND_CHUNK_MS / 1000.0)

                transcript = await events.wait_transcript(timeout=20.0)
                metrics = await events.wait_metrics(timeout=40.0)
                payload = metrics.get("payload", {})

                first_audio = audio.first_voice_after(last_speech)
                client_total = (first_audio - last_speech) * 1000.0 if first_audio else None
                return {
                    "session_id": session_id,
                    "transcript": transcript.get("payload", {}).get("text", ""),
                    "answer": (events.assistant_final or {}).get("payload", {}).get("text", ""),
                    "client_total_ms": client_total,
                    "metrics": payload,
                }
            finally:
                for task in (audio._task, events._task):
                    if task is not None:
                        task.cancel()
                with contextlib.suppress(Exception):
                    await audio_ws.send(json.dumps({"type": "end"}))
        with contextlib.suppress(Exception):
            await events_ws.send(json.dumps({"type": "end"}))


# ------------------------------------------------------------------------- report


def _stage_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stages = [
        "asr_final",
        "lookup",
        "context",
        "llm_ttft",
        "segment",
        "tts_ttfa",
        "egress",
        "llm_total",
    ]
    rows = []
    for stage in stages:
        values = [
            float(run["metrics"]["durations_ms"][stage])
            for run in runs
            if stage in run["metrics"].get("durations_ms", {})
        ]
        if not values:
            continue
        summary = summarise(values)
        rows.append(
            {
                "component": stage,
                "n": summary["n"],
                "p50_ms": summary["p50"],
                "p95_ms": summary["p95"],
                "max_ms": summary["max"],
            }
        )
    return rows


def _totals(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def collect(key: str, getter) -> list[float]:
        out = []
        for run in runs:
            value = getter(run)
            if value is not None:
                out.append(float(value))
        return out

    specs = [
        (
            "client: last speech -> first audible",
            lambda r: r["client_total_ms"],
        ),
        (
            "server: EOS -> first audible",
            lambda r: r["metrics"].get("eos_to_first_audio_ms"),
        ),
        (
            "server: EOS -> first answer audio",
            lambda r: r["metrics"].get("eos_to_answer_audio_ms"),
        ),
    ]
    rows = []
    for label, getter in specs:
        values = collect(label, getter)
        if not values:
            continue
        summary = summarise(values)
        rows.append(
            {
                "total": label,
                "n": summary["n"],
                "p50_ms": summary["p50"],
                "p95_ms": summary["p95"],
                "max_ms": summary["max"],
            }
        )
    return rows


async def run(args: argparse.Namespace) -> int:
    load_env()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(
            f"question fixture not found at {audio_path}; generate a 16 kHz WAV first"
        )
    samples = load_wav_16k(audio_path)
    if samples.size == 0:
        raise SystemExit("question fixture is empty")

    server = None
    server_task = None
    base_url = args.url.rstrip("/") if args.url else ""
    result = Result(probe="e2e-turn", target=base_url or "in-process", label=args.label)
    try:
        if not base_url:
            server, server_task, base_url = await start_server()
            result.target = base_url
        result.note(f"audio fixture: {audio_path} ({len(samples) / 16.0:.0f} ms)")
        result.note(f"trailing silence: {TRAILING_SILENCE_MS} ms")
        result.note("client total is measured from the last speech frame sent")

        runs: list[dict[str, Any]] = []
        for index in range(args.runs):
            run_result = await run_one(base_url, samples)
            runs.append(run_result)
            metrics = run_result["metrics"]
            print(
                f"run {index + 1}/{args.runs}: "
                f"client={run_result['client_total_ms']:.0f} ms "
                f"eos_first={metrics.get('eos_to_first_audio_ms')} ms "
                f"eos_answer={metrics.get('eos_to_answer_audio_ms')} ms "
                f"transcript={run_result['transcript']!r}"
            )

        total_rows = _totals(runs)
        stage_rows = _stage_rows(runs)
        print()
        print(format_table(total_rows, ["total", "n", "p50_ms", "p95_ms", "max_ms"]))
        print()
        print(format_table(stage_rows, ["component", "n", "p50_ms", "p95_ms", "max_ms"]))
        print()
        print("sample transcript:", runs[0]["transcript"])
        print("sample answer:", runs[0]["answer"][:240])

        result.metrics["totals"] = total_rows
        result.metrics["components"] = stage_rows
        for run_result in runs:
            for stage, value in run_result["metrics"].get("durations_ms", {}).items():
                result.add(f"component.{stage}", float(value))
            for key in ("client_total_ms",):
                if run_result.get(key) is not None:
                    result.add(key, float(run_result[key]))
            for key in ("eos_to_first_audio_ms", "eos_to_answer_audio_ms"):
                if run_result["metrics"].get(key) is not None:
                    result.add(key, float(run_result["metrics"][key]))
        result.metrics["transcripts"] = [run_result["transcript"] for run_result in runs]
        result.metrics["answers"] = [run_result["answer"] for run_result in runs]
        path = result.write()
        print(f"\nJSON: {path}")
        return 0
    finally:
        if server is not None:
            server.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(server_task, timeout=15.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--url", default="", help="target an already-running gateway")
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO))
    parser.add_argument("--label", default="local-india")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

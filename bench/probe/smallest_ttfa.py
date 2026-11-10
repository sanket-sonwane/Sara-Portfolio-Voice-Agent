"""Measure Smallest AI Lightning streaming TTS: TTFA, RTF and cancel behaviour.

This is the head-to-head twin of ``cartesia_ttfa.py``: it reuses that probe's
passage, segmentation and metric names, so the two JSON files can be compared
row for row without a normalisation step.

Phase 0 questions this probe answers:

* What is first-audio latency for the first text segment?
* Does incremental submission across segments hold TTFA steady?
* What is the real-time factor (generation wall time / audio duration)?
* How long after a barge-in does audio keep arriving?

Two things are specific to Lightning and worth measuring separately:

* ``--mode separate`` sends each segment under its own ``context_id``. Audio
  starts immediately, but every segment is its own concurrency slot and prosody
  resets at each boundary.
* ``--mode continuation`` sends every segment under one ``context_id`` with
  ``continue: true``. That is one concurrency slot and prosody carries across
  segments, but the provider buffers a fragment until it reaches a sentence
  boundary or ``max_buffer_delay_ms`` elapses, which is latency the caller
  hears. The default is ``separate`` because it is the lower bound.

Usage::

    uv run python -m bench.probe.smallest_ttfa --list-voices
    uv run python -m bench.probe.smallest_ttfa --runs 5 --save-audio
    uv run python -m bench.probe.smallest_ttfa --mode continuation --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import httpx
import numpy as np

from bench.probe.cartesia_ttfa import LONG_TEXT, PASSAGE, segments
from bench.probe.common import (
    AUDIO_DIR,
    Result,
    b64_pcm16,
    duration_ms,
    format_table,
    load_env,
    now,
    optional_env,
    require_env,
    write_wav,
)

try:  # websockets >= 13
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover
    from websockets import connect as ws_connect

WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"
VOICES_URL = "https://api.smallest.ai/waves/v1/lightning-v3.1-pro/get_voices"
DEFAULT_MODEL = "lightning_v3.1_pro"
# "Rhea" is first in Smallest's recommended Indian female ordering.
DEFAULT_VOICE = "rhea"


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _request(
    *,
    text: str,
    context_id: str,
    voice: str,
    model: str,
    language: str,
    sample_rate: int,
    continue_: bool,
    request_id: str = "",
    max_buffer_delay_ms: int | None = None,
    complete_backoff_ms: int = 0,
) -> str:
    payload: dict[str, Any] = {
        "text": text,
        "voice_id": voice,
        "model": model,
        "language": language,
        "sample_rate": sample_rate,
        "context_id": context_id,
        "continue": continue_,
        # The provider holds `complete` for 4 s by default. That is a billing /
        # teardown delay, not synthesis, and it would poison any wall-clock
        # measurement, so every request in this probe asks for none.
        "complete_backoff_ms": complete_backoff_ms,
    }
    if request_id:
        payload["request_id"] = request_id
    if max_buffer_delay_ms is not None:
        payload["max_buffer_delay_ms"] = max_buffer_delay_ms
    return json.dumps(payload)


def _raise_for_error(payload: dict[str, Any]) -> None:
    if payload.get("status") != "error":
        return
    detail = payload.get("message") or payload.get("error") or payload
    raise RuntimeError(f"Smallest error: {detail}")


def _audio_of(payload: dict[str, Any]) -> str:
    data = payload.get("data") or {}
    return data.get("audio") or "" if isinstance(data, dict) else ""


async def list_voices(api_key: str) -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(VOICES_URL, headers=_headers(api_key))
        response.raise_for_status()
        payload = response.json()
    voices = payload.get("voices", payload if isinstance(payload, list) else [])
    rows = []
    for voice in voices:
        tags = voice.get("tags") or {}
        rows.append(
            {
                "id": voice.get("voiceId", "?"),
                "name": voice.get("displayName", "?"),
                "accent": tags.get("accent", "?"),
                "gender": tags.get("gender", "?"),
                "age": tags.get("age", "?"),
            }
        )
    rows.sort(key=lambda row: (row["accent"], row["gender"], row["id"]))
    print(format_table(rows, ["id", "name", "accent", "gender", "age"]))
    print(f"\n{len(rows)} voices visible to this account")
    return 0


async def stream_passage(
    *,
    api_key: str,
    voice: str,
    model: str,
    language: str,
    sample_rate: int,
    mode: str,
    max_buffer_delay_ms: int,
    text: str = PASSAGE,
) -> dict[str, Any]:
    """Measure one segment at a time on one warm socket.

    Smallest does not echo ``context_id`` on audio frames; it echoes our
    ``request_id`` as ``external_request_id``. One in-flight request per socket
    therefore keeps attribution exact and matches the adapter, which submits a
    segment as soon as the model produces it and never overlaps them on the same
    connection.

    ``rtf`` is per segment (generation wall time / that segment's audio
    duration), which is the number that answers "can synthesis keep ahead of
    playback". It is stricter than a whole-passage RTF.
    """
    context_id = f"probe-{int(time.time() * 1000)}"
    pieces = segments(text)
    ttfas: list[float] = []
    walls: list[float] = []
    rtfs: list[float] = []
    all_audio: list[Any] = []

    async with ws_connect(
        f"{WS_URL}?timeout=120",
        additional_headers=_headers(api_key),
        max_size=None,
        open_timeout=20,
    ) as socket:
        for index, piece in enumerate(pieces):
            is_last = index == len(pieces) - 1
            this_context = context_id if mode == "continuation" else f"{context_id}-{index}"
            continue_ = (not is_last) if mode == "continuation" else False
            request_id = f"seg-{index}"

            await socket.send(
                _request(
                    text=piece,
                    context_id=this_context,
                    voice=voice,
                    model=model,
                    language=language,
                    sample_rate=sample_rate,
                    continue_=continue_,
                    request_id=request_id,
                    max_buffer_delay_ms=max_buffer_delay_ms
                    if mode == "continuation"
                    else None,
                )
            )
            sent_at = now()
            first_at = None
            segment_audio: list[Any] = []
            while True:
                raw = await asyncio.wait_for(socket.recv(), timeout=20)
                payload = json.loads(raw)
                _raise_for_error(payload)
                data = _audio_of(payload)
                if data:
                    if first_at is None:
                        first_at = now()
                    segment_audio.append(b64_pcm16(data))
                if payload.get("status") == "complete":
                    break
            finished = now()

            if first_at is not None:
                ttfas.append((first_at - sent_at) * 1000)
            walls.append((finished - sent_at) * 1000)

            segment_samples = (
                np.concatenate(segment_audio) if segment_audio else np.zeros(0, dtype="<i2")
            )
            segment_ms = duration_ms(segment_samples, sample_rate)
            if segment_ms:
                rtfs.append((finished - sent_at) * 1000 / segment_ms)
            all_audio.extend(segment_audio)

    samples = np.concatenate(all_audio) if all_audio else np.zeros(0, dtype="<i2")
    audio_ms = duration_ms(samples, sample_rate)
    return {
        "segments": len(pieces),
        "ttfa_first_ms": ttfas[0] if ttfas else float("nan"),
        "ttfa_all_ms": ttfas,
        "audio_ms": audio_ms,
        "wall_ms": float(sum(walls)),
        "rtf": (sum(rtfs) / len(rtfs)) if rtfs else float("nan"),
        "chunks": len(all_audio),
        "samples": samples,
    }


async def cancel_probe(
    *,
    api_key: str,
    voice: str,
    model: str,
    language: str,
    sample_rate: int,
    cancel_mode: str,
    delay_ms: int = 250,
) -> dict[str, Any]:
    """Measure cancel-to-silence. ``context_close`` and ``socket`` differ a lot."""
    context_id = f"cancel-{int(time.time() * 1000)}"
    late_chunks = 0
    last_audio_at = 0.0
    cancel_at = 0.0
    first_at = 0.0
    sent_at = 0.0

    async with ws_connect(
        f"{WS_URL}?timeout=120",
        additional_headers=_headers(api_key),
        max_size=None,
        open_timeout=20,
    ) as socket:
        await socket.send(
            _request(
                text=LONG_TEXT,
                context_id=context_id,
                voice=voice,
                model=model,
                language=language,
                sample_rate=sample_rate,
                continue_=False,
            )
        )
        sent_at = now()
        while True:
            payload = json.loads(await asyncio.wait_for(socket.recv(), timeout=20))
            _raise_for_error(payload)
            if _audio_of(payload):
                break
        first_at = now()
        await asyncio.sleep(max(0.0, delay_ms / 1000.0 - (now() - first_at)))

        cancel_at = now()
        if cancel_mode == "socket":
            await socket.close()
        else:
            await socket.send(json.dumps({"context_id": context_id, "context_close": True}))

        last_audio_at = cancel_at
        while True:
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=0.6)
            except (TimeoutError, Exception):
                break
            payload = json.loads(raw)
            if payload.get("status") == "error":
                break
            if _audio_of(payload):
                late_chunks += 1
                last_audio_at = now()
            if payload.get("status") == "complete":
                break

    return {
        "cancel_to_silence_ms": (last_audio_at - cancel_at) * 1000,
        "late_chunks": late_chunks,
        "time_to_first_chunk_ms": (first_at - sent_at) * 1000,
    }


async def run(args: argparse.Namespace) -> int:
    api_key = require_env(
        "SMALLEST_API_KEY", "Create a key at https://app.smallest.ai/dashboard/api-keys"
    )
    model = args.model or optional_env("SMALLEST_MODEL", DEFAULT_MODEL)
    voice = args.voice or optional_env("SMALLEST_VOICE", DEFAULT_VOICE)
    language = args.language or optional_env("SMALLEST_LANGUAGE", "en")

    result = Result(probe="smallest-ttfa", target=WS_URL, label=args.label)
    result.metrics.update(
        {
            "model": model,
            "voice": voice,
            "language": language,
            "mode": args.mode,
            "sample_rate": args.sample_rate,
            "runs": args.runs,
        }
    )
    result.note(f"voice={voice} model={model} language={language} mode={args.mode}")

    print(
        f"model={model} voice={voice} language={language} "
        f"mode={args.mode} sample_rate={args.sample_rate}"
    )

    saved_samples = None
    for run_index in range(args.runs):
        run_result = await stream_passage(
            api_key=api_key,
            voice=voice,
            model=model,
            language=language,
            sample_rate=args.sample_rate,
            mode=args.mode,
            max_buffer_delay_ms=args.max_buffer_delay_ms,
        )
        result.add("ttfa_first", run_result["ttfa_first_ms"])
        for value in run_result["ttfa_all_ms"][1:]:
            result.add("ttfa_subsequent", value)
        result.add("rtf", run_result["rtf"])
        result.add("wall", run_result["wall_ms"])
        result.metrics["audio_ms"] = round(run_result["audio_ms"], 1)
        result.metrics["chunks"] = run_result["chunks"]
        result.metrics["segments"] = run_result["segments"]
        saved_samples = run_result["samples"]
        print(
            f"run {run_index + 1}/{args.runs}: first TTFA "
            f"{run_result['ttfa_first_ms']:.1f} ms, RTF {run_result['rtf']:.3f}, "
            f"{run_result['segments']} segments"
        )

    for cancel_mode in (args.cancel_mode, "socket"):
        cancels = [
            await cancel_probe(
                api_key=api_key,
                voice=voice,
                model=model,
                language=language,
                sample_rate=args.sample_rate,
                cancel_mode=cancel_mode,
            )
            for _ in range(max(1, min(3, args.runs)))
        ]
        for cancel in cancels:
            result.add(f"cancel_to_silence_{cancel_mode}", cancel["cancel_to_silence_ms"])
            result.add(f"late_chunks_{cancel_mode}", cancel["late_chunks"])
        result.metrics[f"time_to_first_chunk_ms_{cancel_mode}"] = round(
            cancels[0]["time_to_first_chunk_ms"], 1
        )

    if args.save_audio and saved_samples is not None:
        path = write_wav(
            AUDIO_DIR / f"smallest-{voice}-{language}-{args.mode}.wav",
            saved_samples,
            args.sample_rate,
        )
        result.note(f"wrote {path.relative_to(path.parents[2])}")

    path = result.write()
    rows = [{"metric": key, **value} for key, value in result.distributions().items()]
    print()
    print(format_table(rows, ["metric", "n", "mean", "p50", "p95", "max"]))
    print(f"\nJSON: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--voice", default="", help=f"voice id (default: {DEFAULT_VOICE})")
    parser.add_argument(
        "--model",
        default="",
        help=(
            "lightning_v3.1_pro (default), lightning_v3.1_pro_10_26, lightning_v3.1"
        ),
    )
    parser.add_argument(
        "--language",
        default="",
        help="en (UK/US accents) or hi (Indian accented English + Hindi)",
    )
    parser.add_argument(
        "--mode",
        default="separate",
        choices=("separate", "continuation"),
        help="one context per segment (lowest latency) or one shared context",
    )
    parser.add_argument(
        "--max-buffer-delay-ms",
        type=int,
        default=200,
        dest="max_buffer_delay_ms",
        help="continuation only: how long a fragment may wait for a sentence boundary",
    )
    parser.add_argument("--sample-rate", type=int, default=24000, choices=(8000, 16000, 24000, 44100))
    parser.add_argument(
        "--cancel-mode",
        default="context_close",
        choices=("context_close", "socket"),
        help="barge-in mechanism to measure first; socket is always measured too",
    )
    parser.add_argument("--save-audio", action="store_true", help="Write a WAV into bench/audio/")
    parser.add_argument("--label", default="local")
    parser.add_argument("--list-voices", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    if args.list_voices:
        return asyncio.run(list_voices(require_env("SMALLEST_API_KEY")))
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure Cartesia Sonic streaming TTS: voice selection, TTFA, RTF and cancellation.

Phase 0 questions this probe answers:

* What is first-audio latency for the first text segment?
* Does incremental submission across segments hold TTFA steady?
* What is the real-time factor (generation wall time / audio duration)?
* How long after ``cancel`` does audio keep arriving? (the barge-in floor)

Usage::

    uv run python -m bench.probe.cartesia_ttfa --list-voices
    uv run python -m bench.probe.cartesia_ttfa --runs 5 --save-audio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from typing import Any

import httpx

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

WS_URL = "wss://api.cartesia.ai/tts/websocket"
VOICES_URL = "https://api.cartesia.ai/voices"
API_VERSION = "2024-06-10"

PASSAGE = (
    "Sanket built an agentic commerce gateway that lets unknown AI buyers transact with a "
    "merchant. The system keeps money authority deterministic and requires a signed user "
    "approval for every purchase. Payments run in Razorpay test mode, and each order is "
    "finalised idempotently. Every transaction leaves a hash chained audit trail that a "
    "third party can verify."
)

LONG_TEXT = (
    "The portfolio describes a streaming speech pipeline that turns microphone audio into "
    "recognised text, retrieves supporting evidence, generates a short spoken answer and "
    "synthesizes it incrementally while the visitor is still listening. "
) * 3


def segments(text: str, max_chars: int = 100) -> list[str]:
    """Split into TTS-sized segments on sentence boundaries, then on spaces."""
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        remaining = sentence.strip()
        while len(remaining) > max_chars:
            cut = remaining.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            out.append(remaining)
    return out


async def list_voices(api_key: str) -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            VOICES_URL,
            params={"limit": 100},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Cartesia-Version": API_VERSION,
            },
        )
        response.raise_for_status()
        payload = response.json()
    voices = payload if isinstance(payload, list) else payload.get("data", [])
    rows = [
        {
            "id": voice.get("id", "?"),
            "name": voice.get("name", "?"),
            "language": voice.get("language", "?"),
            "gender": voice.get("gender", "?"),
        }
        for voice in voices
    ]
    print(format_table(rows, ["id", "name", "language", "gender"]))
    print(f"\n{len(rows)} voices visible to this account")
    return 0


async def resolve_voice(api_key: str) -> str:
    pinned = optional_env("CARTESIA_VOICE_ID")
    if pinned:
        return pinned
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            VOICES_URL,
            params={"limit": 1},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Cartesia-Version": API_VERSION,
            },
        )
        response.raise_for_status()
        payload = response.json()
    voices = payload if isinstance(payload, list) else payload.get("data", [])
    if not voices:
        raise SystemExit(
            "no Cartesia voices visible. Set CARTESIA_VOICE_ID in .env "
            "(run --list-voices to see available ids)."
        )
    return voices[0]["id"]


def _message(
    *,
    text: str,
    context_id: str,
    voice: str,
    model: str,
    sample_rate: int,
    continue_: bool,
) -> str:
    return json.dumps(
        {
            "model_id": model,
            "transcript": text,
            "voice": {"mode": "id", "id": voice},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
            "language": "en",
            "context_id": context_id,
            "continue": continue_,
            "add_timestamps": False,
        }
    )


def _raise_for_error(payload: dict[str, Any]) -> None:
    status = payload.get("status_code")
    if payload.get("type") == "error" or (isinstance(status, int) and status >= 400):
        detail = payload.get("error") or payload.get("message") or payload
        raise RuntimeError(f"Cartesia error: {detail}")


async def _drain(
    socket: Any, *, idle_timeout: float = 5.0
) -> tuple[list[Any], list[float]]:
    """Read until the context reports done. Returns audio arrays and chunk timestamps."""
    audio: list[Any] = []
    stamps: list[float] = []
    while True:
        try:
            raw = await asyncio.wait_for(socket.recv(), timeout=idle_timeout)
        except TimeoutError:
            break
        except Exception:
            # Provider may close the socket after a cancel; treat as end of audio.
            break
        payload = json.loads(raw)
        _raise_for_error(payload)
        data = payload.get("data")
        if data:
            audio.append(b64_pcm16(data))
            stamps.append(now())
        if payload.get("done") is True or payload.get("type") == "done":
            break
    return audio, stamps


async def stream_passage(
    *,
    api_key: str,
    voice: str,
    model: str,
    sample_rate: int,
    text: str = PASSAGE,
) -> dict[str, Any]:
    """Submit segments incrementally; measure TTFA per segment and overall RTF."""
    context_id = f"probe-{int(time.time() * 1000)}"
    url = f"{WS_URL}?api_key={api_key}&cartesia_version={API_VERSION}"
    headers = {"Cartesia-Version": API_VERSION}

    pieces = segments(text)
    ttfas: list[float] = []
    all_audio: list[Any] = []
    started = None

    async with ws_connect(url, additional_headers=headers, max_size=None, open_timeout=20) as socket:
        for index, piece in enumerate(pieces):
            is_last = index == len(pieces) - 1
            await socket.send(
                _message(
                    text=piece,
                    context_id=context_id,
                    voice=voice,
                    model=model,
                    sample_rate=sample_rate,
                    continue_=not is_last,
                )
            )
            sent_at = now()
            started = started or sent_at
            # Wait for the first audio chunk of this segment.
            while True:
                raw = await asyncio.wait_for(socket.recv(), timeout=15)
                payload = json.loads(raw)
                _raise_for_error(payload)
                data = payload.get("data")
                if data:
                    ttfas.append((now() - sent_at) * 1000)
                    all_audio.append(b64_pcm16(data))
                    break

        rest, stamps = await _drain(socket)
        all_audio.extend(rest)
        finished = stamps[-1] if stamps else now()

    import numpy as np

    samples = np.concatenate(all_audio) if all_audio else np.zeros(0, dtype="<i2")
    audio_ms = duration_ms(samples, sample_rate)
    wall_ms = (finished - started) * 1000 if started else float("nan")
    return {
        "segments": len(pieces),
        "ttfa_first_ms": ttfas[0] if ttfas else float("nan"),
        "ttfa_all_ms": ttfas,
        "audio_ms": audio_ms,
        "wall_ms": wall_ms,
        "rtf": wall_ms / audio_ms if audio_ms else float("nan"),
        "chunks": len(all_audio),
        "samples": samples,
    }


async def cancel_probe(
    *, api_key: str, voice: str, model: str, sample_rate: int, delay_ms: int = 250
) -> dict[str, Any]:
    """Measure cancel-to-silence: how much audio still arrives after cancelling."""
    context_id = f"cancel-{int(time.time() * 1000)}"
    url = f"{WS_URL}?api_key={api_key}&cartesia_version={API_VERSION}"
    headers = {"Cartesia-Version": API_VERSION}

    async with ws_connect(url, additional_headers=headers, max_size=None, open_timeout=20) as socket:
        await socket.send(
            _message(
                text=LONG_TEXT,
                context_id=context_id,
                voice=voice,
                model=model,
                sample_rate=sample_rate,
                continue_=False,
            )
        )
        sent_at = now()
        # Wait for the first chunk, then let it play for a moment.
        while True:
            payload = json.loads(await asyncio.wait_for(socket.recv(), timeout=15))
            _raise_for_error(payload)
            if payload.get("data"):
                break
        first_at = now()
        await asyncio.sleep(max(0.0, delay_ms / 1000.0 - (now() - first_at)))

        cancel_at = now()
        await socket.send(json.dumps({"context_id": context_id, "cancel": True}))

        late_chunks = 0
        last_audio_at = cancel_at
        while True:
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=0.6)
            except TimeoutError:
                break
            except Exception:
                break
            payload = json.loads(raw)
            if payload.get("type") == "error":
                break
            if payload.get("data"):
                late_chunks += 1
                last_audio_at = now()
            if payload.get("done") is True or payload.get("type") == "done":
                break

    return {
        "cancel_to_silence_ms": (last_audio_at - cancel_at) * 1000,
        "late_chunks": late_chunks,
        "time_to_first_chunk_ms": (first_at - sent_at) * 1000,
    }


async def run(args: argparse.Namespace) -> int:
    api_key = require_env(
        "CARTESIA_API_KEY", "Create a key at https://play.cartesia.ai/console"
    )
    model = optional_env("CARTESIA_MODEL", "sonic-latest")
    voice = args.voice or await resolve_voice(api_key)

    result = Result(probe="cartesia-ttfa", target=WS_URL, label=args.label)
    result.metrics.update(
        {"model": model, "voice": voice, "sample_rate": args.sample_rate, "runs": args.runs}
    )
    result.note(f"voice={voice} model={model}")

    print(f"model={model} voice={voice} sample_rate={args.sample_rate}")

    saved_samples = None
    for run_index in range(args.runs):
        run_result = await stream_passage(
            api_key=api_key,
            voice=voice,
            model=model,
            sample_rate=args.sample_rate,
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

    cancels = [await cancel_probe(
        api_key=api_key, voice=voice, model=model, sample_rate=args.sample_rate
    ) for _ in range(max(1, min(3, args.runs)))]
    for cancel in cancels:
        result.add("cancel_to_silence", cancel["cancel_to_silence_ms"])
        result.add("late_chunks_after_cancel", cancel["late_chunks"])
    result.metrics["time_to_first_chunk_ms"] = round(cancels[0]["time_to_first_chunk_ms"], 1)

    if args.save_audio and saved_samples is not None:
        path = write_wav(AUDIO_DIR / "tts-sample.wav", saved_samples, args.sample_rate)
        result.note(f"wrote {path.relative_to(path.parents[2])}")

    path = result.write()
    rows = [
        {"metric": key, **value} for key, value in result.distributions().items()
    ]
    print()
    print(format_table(rows, ["metric", "n", "mean", "p50", "p95", "max"]))
    print(f"\nJSON: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--voice", default="", help="Cartesia voice id (default: env or first)")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--save-audio", action="store_true", help="Write bench/audio/tts-sample.wav")
    parser.add_argument("--label", default="local")
    parser.add_argument("--list-voices", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    if args.list_voices:
        return asyncio.run(list_voices(require_env("CARTESIA_API_KEY")))
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

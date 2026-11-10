"""Measure the Sarvam arms of the India-first profile: Saarika ASR and Bulbul TTS.

Both endpoints are in-region, which is the whole reason for ADR 0002. This probe
answers two Phase 0 questions:

* TTS: how long from POST to the first playable audio, and how much audio comes
  back per call? Bulbul is request/response, not a socket, so "first audio" is the
  response body, not a stream chunk. The turn engine hides that by synthesising
  one short segment at a time.
* ASR: how long after the socket opens does the first partial transcript arrive,
  and how long after the final audio chunk does the final transcript arrive?

Usage::

    uv run python -m bench.probe.sarvam_probe --tts --runs 5 --save-audio
    uv run python -m bench.probe.sarvam_probe --asr --audio bench/audio/probe.wav
    uv run python -m bench.probe.sarvam_probe --tts --asr --audio bench/audio/probe.wav
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from bench.probe.common import (
    AUDIO_DIR,
    Result,
    chunks,
    format_table,
    load_env,
    load_wav_16k,
    now,
    optional_env,
    require_env,
    write_wav,
)

PHRASE = "Sarang can walk you through Sanket's projects, research and experience."
TTS_RATE = 22_050

try:  # websockets >= 13
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover
    from websockets import connect as ws_connect  # type: ignore


def decode_audio(encoded: str) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(encoded)
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        if width != 2:
            raise ValueError(f"expected 16-bit audio, got {width * 8}-bit")
        samples = np.frombuffer(frames, dtype="<i2")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype("<i2")
        return samples, rate
    return np.frombuffer(raw, dtype="<i2"), TTS_RATE


async def measure_tts(
    *,
    api_key: str,
    base_url: str,
    model: str,
    voice: str,
    runs: int,
    save_audio: bool,
) -> tuple[dict[str, Any], Path | None]:
    url = f"{base_url.rstrip('/')}/text-to-speech"
    payload = {
        "text": PHRASE,
        "target_language_code": "en-IN",
        "speaker": voice,
        "model": model,
        "speech_sample_rate": TTS_RATE,
        "enable_preprocessing": True,
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.0,
    }
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    latencies: list[float] = []
    bytes_out: list[float] = []
    audio_path: Path | None = None

    async with httpx.AsyncClient(timeout=30) as client:
        for index in range(runs + 1):
            started = now()
            response = await client.post(url, headers=headers, json=payload)
            elapsed = (now() - started) * 1000
            response.raise_for_status()
            data = response.json()
            audios = data.get("audios") or ([data["audio"]] if data.get("audio") else [])
            if not audios:
                raise RuntimeError("no audio in response")
            samples = np.concatenate([decode_audio(item)[0] for item in audios])
            if index == 0:
                continue  # warm-up
            latencies.append(elapsed)
            bytes_out.append(float(samples.size))
            if save_audio and audio_path is None:
                AUDIO_DIR.mkdir(parents=True, exist_ok=True)
                audio_path = write_wav(AUDIO_DIR / "sarvam_probe.wav", samples, TTS_RATE)

    ordered = sorted(latencies)
    return (
        {
            "runs": len(latencies),
            "ttfa_p50_ms": round(ordered[len(ordered) // 2], 1),
            "ttfa_max_ms": round(ordered[-1], 1),
            "samples_returned": int(bytes_out[0]) if bytes_out else 0,
            "audio_ms": round(bytes_out[0] / TTS_RATE * 1000, 1) if bytes_out else 0,
        },
        audio_path,
    )


async def measure_asr(
    *,
    api_key: str,
    base_url: str,
    model: str,
    audio_path: Path,
    runs: int,
) -> dict[str, Any]:
    host = base_url.split("://", 1)[-1]
    scheme = "wss" if base_url.startswith("https") else "ws"
    url = (
        f"{scheme}://{host}/speech-to-text/ws"
        f"?language-code=en-IN&model={model}&sample_rate=16000"
    )
    samples = load_wav_16k(audio_path)
    first_partial: list[float] = []
    final_latency: list[float] = []
    finals: list[str] = []

    for index in range(runs + 1):
        opened = now()
        async with ws_connect(
            url, additional_headers={"api-subscription-key": api_key}, max_size=None
        ) as socket:
            connected = now()
            first: float | None = None
            final: float | None = None
            final_text = ""
            for piece in chunks(samples, 100, 16_000):
                await socket.send(piece)
            await socket.send(json.dumps({"type": "flush"}))

            while True:
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=8)
                except TimeoutError:
                    break
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                body = payload.get("data") if isinstance(payload.get("data"), dict) else payload
                text = body.get("transcript") or body.get("text") or body.get("partial") or ""
                if text and first is None:
                    first = now()
                is_final = bool(
                    body.get("is_final")
                    or body.get("final")
                    or str(payload.get("type", "")).lower() in {"final", "transcript.final"}
                )
                if text and is_final:
                    final = now()
                    final_text = str(text)
                    break
            if index == 0:
                continue
            if first is not None:
                first_partial.append((first - connected) * 1000)
            if final is not None:
                final_latency.append((final - opened) * 1000)
            finals.append(final_text[:120])

    ordered = sorted(first_partial)
    return {
        "runs": len(finals),
        "connect_ms": round((connected - opened) * 1000, 1),
        "first_partial_p50_ms": round(ordered[len(ordered) // 2], 1) if ordered else None,
        "final_p50_ms": (
            round(sorted(final_latency)[len(final_latency) // 2], 1) if final_latency else None
        ),
        "sample_final": finals[0] if finals else "",
    }


async def run(args: argparse.Namespace) -> int:
    api_key = require_env("SARVAM_API_KEY", "Create a key at https://dashboard.sarvam.ai")
    base_url = optional_env("SARVAM_BASE_URL", "https://api.sarvam.ai")
    result = Result(probe="sarvam", target=base_url, label=args.label)
    rows: list[dict[str, Any]] = []

    if args.tts:
        metrics, audio_path = await measure_tts(
            api_key=api_key,
            base_url=base_url,
            model=args.tts_model,
            voice=args.voice,
            runs=args.runs,
            save_audio=args.save_audio,
        )
        result.metrics["tts"] = metrics
        rows.append(
            {
                "arm": "tts",
                "p50_ms": metrics["ttfa_p50_ms"],
                "max_ms": metrics["ttfa_max_ms"],
                "detail": f"{metrics['audio_ms']} ms audio",
            }
        )
        if audio_path:
            result.note(f"saved TTS fixture: {audio_path}")

    if args.asr:
        audio = args.audio or (AUDIO_DIR / "sarvam_probe.wav")
        if not Path(audio).exists():
            result.note(f"asr skipped: fixture not found at {audio}; run --tts --save-audio first")
        else:
            metrics = await measure_asr(
                api_key=api_key,
                base_url=base_url,
                model=args.asr_model,
                audio_path=Path(audio),
                runs=args.runs,
            )
            result.metrics["asr"] = metrics
            rows.append(
                {
                    "arm": "asr",
                    "p50_ms": metrics["first_partial_p50_ms"] or "n/a",
                    "max_ms": metrics["final_p50_ms"] or "n/a",
                    "detail": metrics["sample_final"],
                }
            )

    print()
    print(format_table(rows, ["arm", "p50_ms", "max_ms", "detail"]))
    path = result.write()
    print(f"\nJSON: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tts", action="store_true", help="Measure Bulbul TTS")
    parser.add_argument("--asr", action="store_true", help="Measure Saarika ASR")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tts-model", default=optional_env("SARVAM_TTS_MODEL", "bulbul:v4-flash"))
    parser.add_argument("--asr-model", default=optional_env("SARVAM_ASR_MODEL", "saarika:v2.5"))
    parser.add_argument("--voice", default=optional_env("SARVAM_VOICE", "anushka"))
    parser.add_argument("--audio", default="", help="16 kHz WAV fixture for ASR")
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--label", default="local")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    if not args.tts and not args.asr:
        args.tts = True
        args.asr = True
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

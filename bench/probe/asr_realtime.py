"""Compare the live realtime recogniser with the REST finalizer on one WAV.

This is the diagnostic that separates "the provider is inaccurate" from "our
pipeline mangled the audio": both arms get the same 16 kHz file, but only the
realtime arm goes through the socket, the keyterm biasing and the chunking the
gateway actually uses.

Usage::

    .venv\\Scripts\\python.exe -m bench.probe.asr_realtime --audio bench/audio/question-16k.wav
    .venv\\Scripts\\python.exe -m bench.probe.asr_realtime --audio clip.wav --reference "what did you build at arai"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import httpx
from sarang.asr.keyterms import build_keyterms, domain_prompt
from sarang.asr.sarvam import SarvamASR

from bench.probe.common import (
    AUDIO_DIR,
    edit_distance,
    format_table,
    load_env,
    optional_env,
    require_env,
)


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path} must be mono 16-bit PCM")
        if handle.getframerate() != 16_000:
            raise SystemExit(f"{path} is {handle.getframerate()} Hz; resample to 16 kHz first")
        return handle.readframes(handle.getnframes())


async def run_realtime(
    pcm: bytes,
    *,
    api_key: str,
    base_url: str,
    model: str,
    language: str,
    stream_type: str,
    chunk_ms: int,
    keyterms: list[str],
    prompt: str,
) -> dict:
    asr = SarvamASR(
        api_key=api_key,
        base_url=base_url,
        model=model,
        language=language,
        stream_type=stream_type,
        chunk_ms=chunk_ms,
        keyterms=keyterms,
        prompt=prompt,
    )
    started = time.perf_counter()
    await asr.connect()
    connected = time.perf_counter() - started
    config = dict(asr.server_config)

    chunk_bytes = max(320, 16_000 * chunk_ms // 1000 * 2)
    partials: list[str] = []
    first_partial: float | None = None
    final_text = ""
    final_ms: float | None = None
    pump_done: float | None = None

    async def pump() -> None:
        nonlocal pump_done
        await asr.begin_utterance()
        for offset in range(0, len(pcm), chunk_bytes):
            await asr.push(pcm[offset : offset + chunk_bytes])
            await asyncio.sleep(chunk_ms / 1000.0)
        await asr.finalize()
        # When the last audio left us, so a final can be scored as
        # "speech_end -> transcript.final" instead of clip-length noise.
        pump_done = (time.perf_counter() - started) * 1000.0

    task = asyncio.create_task(pump())
    try:
        async for event in asr.events():
            now = (time.perf_counter() - started) * 1000.0
            if event.final:
                final_text = event.text
                final_ms = now
                break
            if first_partial is None:
                first_partial = now
            partials.append(event.text)
    finally:
        await asyncio.gather(task, return_exceptions=True)
        await asr.close()

    return {
        "arm": "realtime",
        "text": final_text,
        "connect_ms": round(connected * 1000.0, 1),
        "first_partial_ms": round(first_partial, 1) if first_partial else None,
        "final_ms": round(final_ms, 1) if final_ms else None,
        "pump_ms": round(pump_done, 1) if pump_done is not None else None,
        "partials": len(partials),
        "config": config,
        "error": asr.last_error,
    }


def run_rest(
    path: Path, *, api_key: str, base_url: str, model: str, language: str
) -> dict:
    started = time.perf_counter()
    with path.open("rb") as handle:
        response = httpx.post(
            f"{base_url.rstrip('/')}/speech-to-text",
            headers={"api-subscription-key": api_key},
            data={
                "model": model,
                "mode": "transcribe",
                "language_code": language,
            },
            files={"file": (path.name, handle, "audio/wav")},
            timeout=60.0,
        )
    elapsed = (time.perf_counter() - started) * 1000.0
    if response.status_code >= 400:
        return {"arm": "rest", "text": "", "error": response.text[:300], "final_ms": None}
    body = response.json()
    return {
        "arm": "rest",
        "text": body.get("transcript", ""),
        "final_ms": round(elapsed, 1),
        "error": "",
    }


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default=str(AUDIO_DIR / "question-16k.wav"))
    parser.add_argument("--reference", default="")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--stream-type", default=optional_env("SARVAM_ASR_STREAM_TYPE", "balanced"))
    parser.add_argument("--chunk-ms", type=int, default=int(optional_env("SARVAM_ASR_CHUNK_MS", "100")))
    parser.add_argument("--model", default=optional_env("SARVAM_ASR_MODEL", "saaras:v4"))
    parser.add_argument("--language", default=optional_env("SARVAM_ASR_LANGUAGE", "en-IN"))
    parser.add_argument("--no-bias", action="store_true", help="Disable keyterms/prompt")
    parser.add_argument(
        "--no-prompt", action="store_true", help="Send keyterms but not the domain prompt"
    )
    parser.add_argument(
        "--keyterms",
        default="",
        help="Semicolon-separated override instead of the KB-derived list",
    )
    args = parser.parse_args()

    api_key = require_env("SARVAM_API_KEY", "Create a key at https://dashboard.sarvam.ai")
    base_url = optional_env("SARVAM_BASE_URL", "https://api.sarvam.ai")
    path = Path(args.audio)
    pcm = read_pcm(path)

    if args.keyterms:
        keyterms = [term.strip() for term in args.keyterms.split(";") if term.strip()]
    else:
        keyterms = [] if args.no_bias else list(build_keyterms())
    prompt = "" if (args.no_bias or args.no_prompt) else domain_prompt()

    print(f"audio={path} ({len(pcm) / 32 / 1000:.2f}s) model={args.model} "
          f"stream={args.stream_type} chunk={args.chunk_ms}ms "
          f"keyterms={len(keyterms)} prompt={'on' if prompt else 'off'}")

    rest = run_rest(path, api_key=api_key, base_url=base_url, model=args.model, language=args.language)
    rows = [rest]
    for _ in range(max(1, args.runs)):
        rows.append(
            asyncio.run(
                run_realtime(
                    pcm,
                    api_key=api_key,
                    base_url=base_url,
                    model=args.model,
                    language=args.language,
                    stream_type=args.stream_type,
                    chunk_ms=args.chunk_ms,
                    keyterms=keyterms,
                    prompt=prompt,
                )
            )
        )

    print()
    print(format_table(
        [
            {
                "arm": row["arm"],
                "first_partial_ms": row.get("first_partial_ms") or "n/a",
                "final_ms": row.get("final_ms") or "n/a",
                "text": row["text"][:90],
            }
            for row in rows
        ],
        ["arm", "first_partial_ms", "final_ms", "text"],
    ))

    if args.reference:
        reference = args.reference.split()
        for row in rows:
            hypothesis = row["text"].split()
            distance = edit_distance(reference, hypothesis)
            print(f"{row['arm']}: WER {distance / max(1, len(reference)):.1%}")

    for row in rows:
        if row.get("config"):
            print(f"\n{row['arm']} session config: {json.dumps(row['config'], sort_keys=True)}")
        if row.get("error"):
            print(f"\n{row['arm']} error: {row['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

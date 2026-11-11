"""What is the recogniser actually doing between end-of-speech and its final?

`asr_final` is the time from `speech_end` to `transcript.final`, and it is the
second-largest term the gateway owns. Part of it is the adapter's own chunking:
audio is aggregated to ``chunk_ms`` before it goes on the wire, so up to one
chunk is still unsent when the turn ends and has to be flushed first.

This probe measures that tail for several chunk sizes on a single socket, at
real-time pace, so the difference is the chunking and not the provider. It is
the only latency lever left that touches neither turn-taking nor accuracy.

Usage::

    uv run python -m bench.probe.asr_final_tail
    uv run python -m bench.probe.asr_final_tail --chunks 20 50 100 --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from sarang.asr.keyterms import build_keyterms, domain_prompt  # noqa: E402
from sarang.asr.sarvam import SarvamASR  # noqa: E402

from bench.probe.common import AUDIO_DIR, load_env, optional_env  # noqa: E402

FRAME_MS = 20
FRAME_SAMPLES = 16_000 * FRAME_MS // 1000


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path} must be mono 16-bit PCM")
        if handle.getframerate() != 16_000:
            raise SystemExit(f"{path} is {handle.getframerate()} Hz; resample to 16 kHz first")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


async def one_run(
    args: argparse.Namespace, api_key: str, samples: np.ndarray, chunk_ms: int
) -> dict:
    asr = SarvamASR(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        language=args.language,
        stream_type=args.stream_type,
        chunk_ms=chunk_ms,
        keyterms=build_keyterms(),
        prompt=domain_prompt(),
    )
    await asr.connect()

    final: dict[str, object] = {"text": "", "at": None}

    async def listen() -> None:
        async for event in asr.events():
            if event.final:
                final["text"] = event.text
                final["at"] = time.perf_counter()
                return

    listener = asyncio.create_task(listen())
    await asr.begin_utterance()

    for offset in range(0, samples.size, FRAME_SAMPLES):
        frame = samples[offset : offset + FRAME_SAMPLES]
        if frame.size == 0:
            break
        await asr.push(frame.tobytes())
        await asyncio.sleep(FRAME_MS / 1000.0)

    # The tail: whatever the adapter still holds, then the provider's final.
    pending = len(asr.pending_bytes) if hasattr(asr, "pending_bytes") else None
    closed_at = time.perf_counter()
    await asr.finalize()
    try:
        await asyncio.wait_for(listener, timeout=args.timeout)
    except TimeoutError:
        listener.cancel()
    final_ms = (final["at"] - closed_at) * 1000.0 if final["at"] else None
    await asr.close()

    return {
        "chunk_ms": chunk_ms,
        "final_ms": final_ms,
        "text": str(final["text"]),
        "pending_bytes": pending,
        "error": asr.last_error,
    }


async def main(args: argparse.Namespace) -> int:
    api_key = optional_env("SARVAM_API_KEY", "")
    if not api_key:
        raise SystemExit("SARVAM_API_KEY is not set")
    samples = read_wav(Path(args.audio))
    print(f"{samples.size / 16.0:.0f} ms of audio, {args.runs} run(s) per chunk size\n")

    rows: list[dict] = []
    for chunk_ms in args.chunks:
        for index in range(args.runs):
            row = await one_run(args, api_key, samples, chunk_ms)
            rows.append(row)
            value = row["final_ms"]
            print(
                f"chunk {chunk_ms:>3} ms  run {index + 1}: "
                f"speech_end -> final {value if value is None else round(value)} ms"
            )
            if row["error"]:
                print(f"  error: {row['error']}")

    print()
    print("chunk_ms   n   p50_ms   min_ms   max_ms")
    for chunk_ms in args.chunks:
        values = [
            row["final_ms"]
            for row in rows
            if row["chunk_ms"] == chunk_ms and row["final_ms"] is not None
        ]
        if not values:
            print(f"{chunk_ms:>8}   -        -        -        -")
            continue
        print(
            f"{chunk_ms:>8} {len(values):>3} "
            f"{statistics.median(values):>8.0f} {min(values):>8.0f} {max(values):>8.0f}"
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", default=str(AUDIO_DIR / "question-16k.wav"))
    parser.add_argument("--chunks", type=int, nargs="*", default=[20, 50, 100])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--model", default="saaras:v4")
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--stream-type", default="fast")
    parser.add_argument("--base-url", default="https://api.sarvam.ai")
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    raise SystemExit(asyncio.run(main(parse_args())))

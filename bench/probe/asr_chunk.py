"""Does a smaller ASR batch buy partial latency without costing accuracy?

Sarvam's own guidance is 100-500 ms per ``audio_input`` message, and the gateway
batches 20 ms browser frames into 100 ms blocks. Smaller blocks should surface a
partial transcript sooner - the provider cannot score audio it has not received -
but they may also destabilise recognition, because each block is a shorter
context for the decoder.

This probe replays one real question at several batch sizes, in real time (the
provider's partial cadence depends on wall time, so bursting the clip would
measure the wrong thing), and reports for each:

* ``first_partial_ms`` - first push -> first non-empty partial.
* ``full_partial_ms``  - first push -> first partial within 10% WER of the
  spoken text. This is the number that decides when speculation can start.
* ``final_ms``         - last speech frame -> final transcript.
* ``wer``              - final transcript error rate.

Usage::

    uv run python -m bench.probe.asr_chunk
    uv run python -m bench.probe.asr_chunk --chunks 20 40 60 100 --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from sarang.asr.keyterms import build_keyterms, domain_prompt  # noqa: E402
from sarang.asr.sarvam import SarvamASR  # noqa: E402

from bench.probe.common import (  # noqa: E402
    AUDIO_DIR,
    Result,
    load_env,
    optional_env,
    read_wav,
    word_error_rate,
)

FRAME_MS = 20
FRAME_SAMPLES = 16_000 * FRAME_MS // 1000
SPOKEN = "What did Sanket build during the ARAI internship?"


def silence(ms: int) -> np.ndarray:
    return np.zeros(16_000 * ms // 1000, dtype=np.int16)


async def measure(
    samples: np.ndarray,
    *,
    api_key: str,
    base_url: str,
    model: str,
    language: str,
    stream_type: str,
    chunk_ms: int,
    trailing_ms: int,
) -> dict:
    asr = SarvamASR(
        api_key=api_key,
        base_url=base_url,
        model=model,
        language=language,
        stream_type=stream_type,
        chunk_ms=chunk_ms,
        keyterms=build_keyterms(),
        prompt=domain_prompt(),
    )
    await asr.connect()

    partials: list[tuple[float, str]] = []
    final: dict[str, object] = {"text": "", "at": None}

    async def listen() -> None:
        async for event in asr.events():
            at = time.perf_counter()
            if event.final:
                if not final["text"]:
                    final["text"] = event.text
                    final["at"] = at
                continue
            if event.text:
                partials.append((at, event.text))

    listener = asyncio.create_task(listen())
    await asr.begin_utterance()

    started = time.perf_counter()
    last_speech = started
    for offset in range(0, samples.size, FRAME_SAMPLES):
        frame = samples[offset : offset + FRAME_SAMPLES]
        await asr.push(frame.tobytes())
        last_speech = time.perf_counter()
        await asyncio.sleep(FRAME_MS / 1000.0)
    for _ in range(trailing_ms // FRAME_MS):
        await asr.push(silence(FRAME_MS).tobytes())
        await asyncio.sleep(FRAME_MS / 1000.0)

    await asr.finalize()
    try:
        await asyncio.wait_for(listener, timeout=8.0)
    except TimeoutError:
        listener.cancel()
    await asr.close()

    final_text = str(final["text"])
    final_at = final["at"]
    first_partial = partials[0] if partials else None
    full = next(
        (
            (stamp, text)
            for stamp, text in partials
            if word_error_rate(SPOKEN, text) <= 0.1
        ),
        None,
    )
    return {
        "chunk_ms": chunk_ms,
        "first_partial_ms": round((first_partial[0] - started) * 1000.0, 1)
        if first_partial
        else None,
        "first_partial": first_partial[1] if first_partial else "",
        "full_partial_ms": round((full[0] - started) * 1000.0, 1) if full else None,
        "final_ms": round((final_at - last_speech) * 1000.0, 1)
        if isinstance(final_at, float)
        else None,
        "final": final_text,
        "wer": round(word_error_rate(SPOKEN, final_text), 3),
        "partials": len(partials),
        "error": asr.last_error,
    }


async def main(args: argparse.Namespace) -> int:
    api_key = optional_env("SARVAM_API_KEY", "")
    if not api_key:
        raise SystemExit("SARVAM_API_KEY is required (set it in .env)")
    base_url = optional_env("SARVAM_BASE_URL", "https://api.sarvam.ai")
    samples, rate = read_wav(Path(args.audio))
    if rate != 16_000:
        raise SystemExit(f"{args.audio} is {rate} Hz; the gateway streams 16 kHz")
    if samples.dtype != np.int16:
        samples = samples.astype(np.int16)

    result = Result(probe="asr-chunk", target=f"{args.model}/{args.stream_type}", label=args.label)
    result.metrics["spoken"] = SPOKEN
    result.metrics["audio_ms"] = round(samples.size / 16.0, 1)

    rows = []
    for chunk_ms in args.chunks:
        runs = []
        for _ in range(args.runs):
            runs.append(
                await measure(
                    samples,
                    api_key=api_key,
                    base_url=base_url,
                    model=args.model,
                    language=args.language,
                    stream_type=args.stream_type,
                    chunk_ms=chunk_ms,
                    trailing_ms=args.trailing_ms,
                )
            )
        # Report the median run: a single slow provider response should not
        # decide a configuration.
        def median(key: str, runs: list[dict] = runs) -> float | None:
            values = sorted(
                run[key] for run in runs if isinstance(run[key], (int, float))
            )
            if not values:
                return None
            return values[len(values) // 2]

        best = min(runs, key=lambda run: run["wer"])
        row = {
            "chunk_ms": chunk_ms,
            "first_partial_ms": median("first_partial_ms"),
            "full_partial_ms": median("full_partial_ms"),
            "final_ms": median("final_ms"),
            "wer": round(median("wer") or 0.0, 3),
            "final": best["final"],
            "error": best["error"],
        }
        rows.append(row)
        for key in ("first_partial_ms", "full_partial_ms", "final_ms"):
            value = row[key]
            if isinstance(value, (int, float)):
                result.add(f"{key}@{chunk_ms}", value)
        result.metrics[f"chunk_{chunk_ms}"] = row
        print(
            f"{chunk_ms:>4} ms  first={row['first_partial_ms']} ms  "
            f"full={row['full_partial_ms']} ms  final={row['final_ms']} ms  "
            f"wer={row['wer']}  {row['final']!r}"
        )

    print()
    for row in rows:
        print(
            f"  {row['chunk_ms']:>4} ms  wer={row['wer']:<5} "
            f"first={row['first_partial_ms']}  full={row['full_partial_ms']}"
        )
    path = result.write()
    print(f"\nJSON: {path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", default=str(AUDIO_DIR / "question-16k.wav"))
    parser.add_argument("--chunks", nargs="+", type=int, default=[20, 40, 60, 100])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--trailing-ms", type=int, default=1200)
    parser.add_argument("--model", default="saaras:v4")
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--stream-type", default="fast")
    parser.add_argument("--label", default="local-india")
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    raise SystemExit(asyncio.run(main(parse_args())))

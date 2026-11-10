"""How much dead air does the TTS put in front of the first phoneme?

The egress now drops inaudible leading samples on the first chunk of a
generation (``sarang.audio.shaper.trim_leading_silence``). That is only safe if
the leading samples really are silence and not the attack of a word, so this
probe measures it against the live provider and records the numbers.

A first span with a peak of 0-6 out of 32 768 is padding. A first span with a
peak near the later spans is speech that must be kept.

Usage::

    uv run python -m bench.probe.tts_leading_silence --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from sarang.tts.smallest import SmallestTTS  # noqa: E402

from bench.probe.common import (  # noqa: E402
    AUDIO_DIR,
    Result,
    format_table,
    load_env,
    optional_env,
    write_wav,
)

TEXT = "During his ARAI internship, Sanket built and deployed a production Python pipeline."
SPANS = 4


def leading_silence_ms(samples: np.ndarray, threshold: int) -> float:
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if not loud.size:
        return samples.size / 48.0
    return float(loud[0]) / 48.0


async def run(args: argparse.Namespace) -> int:
    api_key = optional_env("SMALLEST_API_KEY", "")
    if not api_key:
        raise SystemExit("SMALLEST_API_KEY is not set")

    result = Result(probe="tts-leading-silence", target="smallest", label=args.label)
    result.note(f"threshold={args.threshold} spans={SPANS}")
    tts = SmallestTTS(api_key=api_key, voice_id=args.voice or "rhea")
    await tts.prepare(TEXT)
    try:
        first_span_samples = None
        for index in range(args.runs):
            spans: list[np.ndarray] = []
            async for span in tts.synthesize(TEXT, context_id=f"lead-{index}", final=True):
                spans.append(span.pcm)
                if len(spans) == SPANS:
                    break
            leads = [leading_silence_ms(span, args.threshold) for span in spans]
            peaks = [int(np.abs(span).max()) for span in spans]
            result.add("leading_ms_first_span", leads[0])
            result.add("peak_first_span", peaks[0])
            for later in leads[1:]:
                result.add("leading_ms_later_span", later)
            for later in peaks[1:]:
                result.add("peak_later_span", later)
            result.metrics["first_span_ms"] = round(spans[0].size / 48.0, 1)
            first_span_samples = spans[0]
            print(
                f"run {index + 1}: first_span_ms={spans[0].size / 48:.1f} "
                f"leading_ms={[round(value, 1) for value in leads]} peak={peaks}"
            )
    finally:
        await tts.close()

    if args.save_audio and first_span_samples is not None:
        path = write_wav(AUDIO_DIR / "smallest-first-span.wav", first_span_samples, 48_000)
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
    parser.add_argument("--voice", default="", help="Smallest voice id (default: rhea)")
    parser.add_argument("--threshold", type=int, default=96)
    parser.add_argument("--save-audio", action="store_true", help="write the first span as a WAV")
    parser.add_argument("--label", default="local")
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    raise SystemExit(asyncio.run(run(parse_args())))

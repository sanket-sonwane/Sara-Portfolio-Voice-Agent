"""How long are the silences *inside* a normally-spoken question?

The endpointing window can only be shortened to a value that is longer than the
pauses a speaker leaves mid-sentence. This measures them with the same VAD the
gateway uses, so the numbers are directly comparable to
``ENDPOINT_SILENCE_MS``.

Usage::

    uv run python -m bench.probe.pause_profile
    uv run python -m bench.probe.pause_profile --audio bench/audio/question-16k.wav
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from sarang.audio.vad import Vad, VadConfig  # noqa: E402

from bench.probe.common import AUDIO_DIR  # noqa: E402

FRAME_MS = 20
FRAME_SAMPLES = 16_000 * FRAME_MS // 1000


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path} must be mono 16-bit PCM")
        if handle.getframerate() != 16_000:
            raise SystemExit(f"{path} is {handle.getframerate()} Hz; resample to 16 kHz first")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def profile(samples: np.ndarray) -> tuple[list[float], float, float]:
    """Return (internal silence runs, leading silence, trailing silence) in ms."""
    vad = Vad(VadConfig(silence_ms=800))
    runs: list[float] = []
    run = 0.0
    spoken = False
    trailing = 0.0
    for offset in range(0, samples.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frame = samples[offset : offset + FRAME_SAMPLES]
        update = vad.push(frame, 16_000)
        if update.silence_ms < FRAME_MS:
            if spoken and run:
                runs.append(run)
            run = 0.0
            spoken = True
        else:
            run += FRAME_MS
            if spoken:
                trailing = run
    return runs, trailing


def main(args: argparse.Namespace) -> int:
    path = Path(args.audio)
    samples = read_wav(path)
    runs, trailing = profile(samples)
    internal = [value for value in runs if value >= 120]

    print(f"{path.name}: {samples.size / 16.0:.0f} ms of audio")
    print(f"  internal silences >= 120 ms: {[round(value) for value in internal]}")
    if internal:
        print(f"  longest internal pause: {max(internal):.0f} ms")
    print(f"  trailing silence: {trailing:.0f} ms")
    print()
    for window in (300, 400, 450, 500, 600, 700, 800):
        cuts = [value for value in internal if value >= window]
        verdict = "would cut" if cuts else "safe"
        print(f"  ENDPOINT_SILENCE_MS={window:<4} {verdict:<10} ({len(cuts)} internal pause(s) reach it)")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", default=str(AUDIO_DIR / "question-16k.wav"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))

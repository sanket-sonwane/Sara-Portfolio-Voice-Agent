"""Would an earlier endpoint have closed the turn on a complete question?

Adaptive endpointing needs evidence that arrives before the 800 ms window
expires. The only evidence available that early is the live partial transcript,
and the recorded dumps show partials churning mid-utterance, so the rule cannot
simply look for a full stop: it has to know whether the text is *stable* and
whether it *ends* a sentence.

This probe measures exactly that, on two variants of the same utterance:

* ``whole`` — the question as spoken.
* ``split`` — the same audio cut at its quietest point and rejoined with a
  pause, which is the failure case ADR 0003 was written for: a visitor who
  stops mid-sentence to think.

For each checkpoint (silence elapsed since the last speech frame) it prints the
partial the recogniser had produced, how long that text had been unchanged, and
whether it ended a sentence. A rule is only safe if the two variants look
different at the cutoff it would use.

Usage::

    uv run python -m bench.probe.endpoint_evidence
    uv run python -m bench.probe.endpoint_evidence --pause-ms 600 --audio bench/audio/question-16k.wav
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from sarang.asr.keyterms import build_keyterms, domain_prompt  # noqa: E402
from sarang.asr.sarvam import SarvamASR  # noqa: E402
from sarang.audio.vad import Vad, VadConfig  # noqa: E402

from bench.probe.common import AUDIO_DIR, load_env, optional_env  # noqa: E402

FRAME_MS = 20
FRAME_BYTES = 16_000 * FRAME_MS // 1000 * 2
CHECKPOINTS = (250, 350, 450, 550, 650, 750)
TERMINAL = re.compile(r"[.!?][\"')\]]?$")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path} must be mono 16-bit PCM")
        if handle.getframerate() != 16_000:
            raise SystemExit(f"{path} is {handle.getframerate()} Hz; resample to 16 kHz first")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def quietest_split(samples: np.ndarray, window_ms: int = 100) -> int:
    """Sample offset of the quietest window in the middle 60% of the clip."""
    window = 16_000 * window_ms // 1000
    low = int(samples.size * 0.2)
    high = int(samples.size * 0.8) - window
    if high <= low:
        return samples.size // 2
    energies = [
        float(np.sqrt(np.mean(samples[at : at + window].astype(np.float32) ** 2)))
        for at in range(low, high, window // 2)
    ]
    return low + int(np.argmin(energies)) * (window // 2)


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
    early_final_ms: int = 0,
) -> dict:
    """Stream one utterance and sample the partial at each silence checkpoint.

    With ``early_final_ms`` the recogniser is closed that far into the silence
    and reopened if speech resumes, which is the ASR-flush idea: the final would
    already be in hand when the 800 ms endpoint expires. The thing that decides
    whether it is safe is what the *final* says for the truncated audio.
    """
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
    vad = Vad(VadConfig(silence_ms=800))

    partials: list[tuple[float, str]] = []
    final: dict[str, object] = {"text": "", "at": None}
    segment_finals: list[str] = []
    segment_open = True

    async def listen() -> None:
        nonlocal segment_open
        async for event in asr.events():
            at = time.perf_counter()
            if event.final:
                if segment_open:
                    segment_finals.append(event.text)
                    segment_open = False
                if not final["text"]:
                    final["text"] = event.text
                    final["at"] = at
                continue
            if event.text:
                partials.append((at, event.text))

    listener = asyncio.create_task(listen())
    await asr.begin_utterance()

    silence_frame = np.zeros(FRAME_BYTES // 2, dtype=np.int16)
    frames = [
        samples[offset : offset + FRAME_BYTES // 2]
        for offset in range(0, samples.size, FRAME_BYTES // 2)
    ]
    frames.extend([silence_frame] * (trailing_ms // FRAME_MS))

    checkpoints: dict[int, dict] = {}
    last_speech = time.perf_counter()
    closed = False
    for frame in frames:
        if frame.size == 0:
            continue
        update = vad.push(frame, 16_000)
        now = time.perf_counter()
        # `silence_ms` resets to zero on a frame above threshold, which is the
        # only per-frame speech signal the VAD exposes (`speech` is the sticky
        # "utterance is active" flag).
        above = update.silence_ms < FRAME_MS
        if above:
            last_speech = now
            if closed:
                # Speech resumed after an early final: reopen the recogniser.
                # The audio either side of the pause is silence, so the join is
                # not mid-word.
                await asr.begin_utterance()
                segment_open = True
                closed = False
        await asr.push(frame.tobytes())
        await asyncio.sleep(FRAME_MS / 1000.0)

        elapsed = (now - last_speech) * 1000.0
        if early_final_ms and segment_open and update.silence_ms >= early_final_ms:
            await asr.finalize()
            closed = True
        for point in CHECKPOINTS:
            if point not in checkpoints and elapsed >= point:
                checkpoints[point] = {"at": now, "elapsed": elapsed}

    await asr.finalize()
    try:
        await asyncio.wait_for(listener, timeout=8.0)
    except TimeoutError:
        listener.cancel()
    await asr.close()

    for _point, row in checkpoints.items():
        at = row["at"]
        seen = [(stamp, text) for stamp, text in partials if stamp <= at]
        if not seen:
            row.update(text="", stable_ms=0.0)
            continue
        text = seen[-1][1]
        changed = next(
            (stamp for stamp, earlier in reversed(seen) if earlier != text), None
        )
        row.update(
            text=text,
            stable_ms=(at - changed) * 1000.0 if changed is not None else float("inf"),
        )

    return {
        "checkpoints": checkpoints,
        "partials": len(partials),
        "final": str(final["text"]),
        "segments": segment_finals,
        "error": asr.last_error,
    }


def describe(row: dict) -> str:
    text = str(row.get("text") or "")
    stable = row.get("stable_ms", 0.0)
    stable_label = ">1 window" if stable == float("inf") else f"{stable:.0f} ms"
    terminal = "yes" if TERMINAL.search(text) else "no"
    return (
        f"{text[:52]:<52} stable {stable_label:>10}  ends-sentence {terminal:<3} "
        f"words {len(text.split()):>2}"
    )


async def main(args: argparse.Namespace) -> int:
    api_key = optional_env("SARVAM_API_KEY", "")
    if not api_key:
        raise SystemExit("SARVAM_API_KEY is not set")
    base_url = optional_env("SARVAM_BASE_URL", "https://api.sarvam.ai")

    whole = read_wav(Path(args.audio))
    cut = quietest_split(whole)
    split = np.concatenate([whole[:cut], silence(args.pause_ms), whole[cut:]])

    variants = {"whole": whole, "split": split}
    if args.only:
        variants = {name: variants[name] for name in args.only}

    for name, samples in variants.items():
        print(f"\n=== {name} ({samples.size / 16.0:.0f} ms of audio) ===")
        result = await measure(
            samples,
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            language=args.language,
            stream_type=args.stream_type,
            chunk_ms=args.chunk_ms,
            trailing_ms=args.trailing_ms,
            early_final_ms=args.early_final_ms,
        )
        print(f"partials={result['partials']}  error={result['error'] or '-'}")
        print(f"final: {result['final']}")
        if args.early_final_ms:
            segments = result["segments"]
            print(f"segments ({len(segments)}):")
            for number, text in enumerate(segments, start=1):
                print(f"  [{number}] {text}")
            print(f"joined: {' '.join(segments)}")
        for point in CHECKPOINTS:
            row = result["checkpoints"].get(point)
            if row is None:
                continue
            print(f"  {point:>4} ms  {describe(row)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", default=str(AUDIO_DIR / "question-16k.wav"))
    parser.add_argument("--pause-ms", type=int, default=600)
    parser.add_argument("--trailing-ms", type=int, default=1100)
    parser.add_argument("--model", default="saaras:v4")
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--stream-type", default="fast")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument(
        "--early-final-ms",
        type=int,
        default=0,
        help="close the recogniser this far into the silence, reopening on resume (0 = off)",
    )
    parser.add_argument("--only", nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    raise SystemExit(asyncio.run(main(parse_args())))

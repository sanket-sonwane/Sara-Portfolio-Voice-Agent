"""Can a second recogniser have the final ready before the endpoint expires?

The speculative recogniser is only worth building if a parallel socket, fed the
same audio in real time, produces its final early enough to be used when the
800 ms endpointing window runs out. That means:

* its final must arrive comfortably inside the window, and
* it must say the same thing as the live socket's final, or using it would
  change the answer.

This probe feeds two independent recogniser sockets the same audio, closes the
speculative one at a configurable silence offset, and reports both. If the
speculative final is late, empty, or different, the design is not viable and the
measurement says so before any product code is written.

Usage::

    uv run python -m bench.probe.asr_speculation
    uv run python -m bench.probe.asr_speculation --spec-ms 500 --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
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
FRAME_SAMPLES = 16_000 * FRAME_MS // 1000
EOS_MS = 800


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path} must be mono 16-bit PCM")
        if handle.getframerate() != 16_000:
            raise SystemExit(f"{path} is {handle.getframerate()} Hz; resample to 16 kHz first")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def build_asr(args: argparse.Namespace, api_key: str) -> SarvamASR:
    return SarvamASR(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        language=args.language,
        stream_type=args.stream_type,
        chunk_ms=args.chunk_ms,
        keyterms=build_keyterms(),
        prompt=domain_prompt(),
    )


class Socket:
    """One recogniser plus the final it eventually reports."""

    def __init__(self, asr: SarvamASR, name: str) -> None:
        self.asr = asr
        self.name = name
        self.final = ""
        self.final_at: float | None = None
        self.partials = 0
        self.ready = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for event in self.asr.events():
                if event.final:
                    if not self.final:
                        self.final = event.text
                        self.final_at = time.perf_counter()
                        self.ready.set()
                    continue
                if event.text:
                    self.partials += 1
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            print(f"  {self.name} reader failed: {error}")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await self.asr.close()


async def one_run(args: argparse.Namespace, api_key: str, samples: np.ndarray) -> dict:
    live = Socket(build_asr(args, api_key), "live")
    spec = Socket(build_asr(args, api_key), "spec")
    await live.asr.connect()
    await spec.asr.connect()
    live.start()
    spec.start()

    await live.asr.begin_utterance()
    await spec.asr.begin_utterance()

    vad = Vad(VadConfig(silence_ms=EOS_MS))
    spec_closed_at: float | None = None
    eos_at: float | None = None
    started = time.perf_counter()

    frames = [
        samples[offset : offset + FRAME_SAMPLES]
        for offset in range(0, samples.size, FRAME_SAMPLES)
    ]
    # Trailing silence so the endpointing window can actually elapse.
    frames.extend([np.zeros(FRAME_SAMPLES, dtype=np.int16)] * ((EOS_MS + 400) // FRAME_MS))

    for frame in frames:
        update = vad.push(frame, 16_000)
        await live.asr.push(frame.tobytes())
        if spec_closed_at is None:
            await spec.asr.push(frame.tobytes())
        await asyncio.sleep(FRAME_MS / 1000.0)

        if spec_closed_at is None and update.silence_ms >= args.spec_ms:
            spec_closed_at = time.perf_counter()
            await spec.asr.finalize()
        if eos_at is None and update.silence_ms >= EOS_MS:
            eos_at = time.perf_counter()
            await live.asr.finalize()
            break

    if eos_at is None:
        eos_at = time.perf_counter()
        await live.asr.finalize()

    # How long each socket took from its own close to its own final.
    with_spec = asyncio.create_task(spec.ready.wait())
    with_live = asyncio.create_task(live.ready.wait())
    try:
        await asyncio.wait_for(asyncio.gather(with_spec, with_live), timeout=args.timeout)
    except TimeoutError:
        pass

    spec_ready_ms = (
        (spec.final_at - spec_closed_at) * 1000.0
        if spec.final_at and spec_closed_at
        else None
    )
    live_ready_ms = (live.final_at - eos_at) * 1000.0 if live.final_at else None
    # The number that matters: was the speculative final already in hand when
    # the endpointing window expired?
    margin_ms = (eos_at - spec.final_at) * 1000.0 if spec.final_at else None

    await live.stop()
    await spec.stop()

    return {
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "spec_ms": args.spec_ms,
        "spec_finalize_ms": spec_ready_ms,
        "live_finalize_ms": live_ready_ms,
        "margin_ms": margin_ms,
        "spec_text": spec.final,
        "live_text": live.final,
        "same": spec.final.strip() == live.final.strip(),
        "spec_partials": spec.partials,
        "live_partials": live.partials,
        "spec_error": spec.asr.last_error,
        "live_error": live.asr.last_error,
    }


async def main(args: argparse.Namespace) -> int:
    api_key = optional_env("SARVAM_API_KEY", "")
    if not api_key:
        raise SystemExit("SARVAM_API_KEY is not set")
    samples = read_wav(Path(args.audio))

    rows: list[dict] = []
    for index in range(args.runs):
        row = await one_run(args, api_key, samples)
        rows.append(row)
        print(
            f"run {index + 1}: spec_finalize={row['spec_finalize_ms'] and round(row['spec_finalize_ms'])} ms "
            f"live_finalize={row['live_finalize_ms'] and round(row['live_finalize_ms'])} ms "
            f"margin_before_eos={row['margin_ms'] and round(row['margin_ms'])} ms "
            f"same={row['same']}"
        )
        print(f"  spec: {row['spec_text']}")
        print(f"  live: {row['live_text']}")
        if row["spec_error"] or row["live_error"]:
            print(f"  errors: spec={row['spec_error']} live={row['live_error']}")

    usable = [row for row in rows if row["margin_ms"] is not None and row["margin_ms"] > 0]
    agree = [row for row in usable if row["same"]]
    print()
    print(f"speculative final ready before end-of-speech: {len(usable)}/{len(rows)}")
    print(f"and identical to the live final:              {len(agree)}/{len(rows)}")
    return 0 if len(agree) == len(rows) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", default=str(AUDIO_DIR / "question-16k.wav"))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--spec-ms", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--model", default="saaras:v4")
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--stream-type", default="fast")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--base-url", default="https://api.sarvam.ai")
    return parser.parse_args()


if __name__ == "__main__":
    load_env()
    raise SystemExit(asyncio.run(main(parse_args())))

"""Does the energy VAD actually detect speech correctly?

Everything so far has measured *how long* the endpointing window is. Nothing has
measured whether the detector underneath it is right, which is the question that
decides whether the window can ever come down: a window that is generous because
the detector is unreliable cannot be shortened by tuning, only by replacing.

The detector is judged against ground truth it cannot see. Every scenario here is
synthesised from known pieces, so the true start and end of speech are known
exactly:

* ``level`` sweep   - one utterance scaled to a range of loudness, over silence.
                      Answers "how quiet can the speaker be before we go deaf?"
* ``snr`` sweep     - the same utterance over white noise at a range of levels.
                      Answers "does a noisy room produce false starts or misses?"
* ``hum``           - the same utterance under a steady 50 Hz mains hum, which is
                      what the zero-crossing guard exists for.
* ``pause``         - the reference question, whose internal pause is known to be
                      460 ms from pause_profile.py. Answers "where does the
                      detector place the end of speech?"

Metrics are per scenario: whether speech was detected at all, how late it started
relative to the true start, and how far the detected end sat from the true end.
The last one is the number that bounds ``ENDPOINT_SILENCE_MS``: if the detector
already places the end late, the silence window starts late too.

Usage::

    uv run python -m bench.probe.vad_accuracy
    uv run python -m bench.probe.vad_accuracy --audio bench/audio/question-16k.wav
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from sarang.audio.vad import Vad, VadConfig, dbfs, zero_crossing_rate  # noqa: E402

from bench.probe.common import AUDIO_DIR  # noqa: E402

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path} must be mono 16-bit PCM")
        if handle.getframerate() != SAMPLE_RATE:
            raise SystemExit(f"{path} is {handle.getframerate()} Hz; resample to 16 kHz first")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def scale_to_dbfs(samples: np.ndarray, target_db: float) -> np.ndarray:
    """Scale so the *RMS* of the whole clip sits at ``target_db`` dBFS."""
    current = dbfs(samples)
    gain = 10.0 ** ((target_db - current) / 20.0)
    scaled = samples.astype(np.float32) * gain
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def silence(ms: int) -> np.ndarray:
    return np.zeros(SAMPLE_RATE * ms // 1000, dtype=np.int16)


def white_noise(count: int, level_db: float, seed: int) -> np.ndarray:
    """``count`` samples of noise with RMS at ``level_db`` dBFS."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(count).astype(np.float32)
    noise /= max(float(np.sqrt(np.mean(noise**2))), 1e-9)
    target = 10.0 ** (level_db / 20.0)
    return np.clip(noise * target * 32768.0, -32768, 32767).astype(np.int16)


def hum(count: int, level_db: float, freq_hz: float = 50.0) -> np.ndarray:
    """``count`` samples of a steady tone at ``level_db`` dBFS RMS."""
    t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    tone = np.sin(2.0 * np.pi * freq_hz * t) * np.sqrt(2.0)  # unit RMS
    target = 10.0 ** (level_db / 20.0)
    return np.clip(tone * target * 32768.0, -32768, 32767).astype(np.int16)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate([part for part in parts if part.size])


def frames(samples: np.ndarray):
    for offset in range(0, samples.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        yield samples[offset : offset + FRAME_SAMPLES]


def run_vad(samples: np.ndarray, config: VadConfig) -> dict:
    """Feed a stream through the VAD and record what it decided, and when."""
    vad = Vad(config)
    started_at: float | None = None
    ended_at: float | None = None
    starts: list[float] = []
    speech_ms = 0.0
    for index, frame in enumerate(frames(samples)):
        at = index * FRAME_MS
        update = vad.push(frame, SAMPLE_RATE)
        if update.speech_started:
            starts.append(float(at))
            if started_at is None:
                started_at = float(at)
        if update.speech:
            speech_ms += FRAME_MS
        if update.speech_ended and ended_at is None:
            ended_at = float(at)
    return {
        "starts": starts,
        "start": started_at,
        "end": ended_at,
        "speech_ms": speech_ms,
    }


def level_profile(samples: np.ndarray, floor_db: float) -> dict:
    """Where does speech actually sit relative to the detector's floor?"""
    values = [dbfs(frame) for frame in frames(samples)]
    speech = [value for value in values if value > -60.0]
    if not speech:
        return {}
    ordered = sorted(speech)
    return {
        "frames": len(speech),
        "p05": ordered[int(len(ordered) * 0.05)],
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[int(len(ordered) * 0.95)],
        "below_floor": sum(1 for value in speech if value < floor_db),
    }


def level_series(samples: np.ndarray) -> list[float]:
    return [dbfs(frame) for frame in frames(samples)]


def sliding_std(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values) - window + 1):
        chunk = values[i : i + window]
        out.append(float(np.std(chunk)))
    return out


def discriminator_report(speech: np.ndarray) -> None:
    """What actually separates speech from a steady hum or steady noise?

    Two candidate signals, both cheap enough to compute per frame:

    * ``zcr``      - a 50 Hz hum crosses zero 100 times a second (rate 0.00625)
                     and white noise crosses ~8000 times (0.5); speech sits
                     between. This is what the existing guard uses, badly.
    * ``modulation`` - the standard deviation of frame level over a 300 ms
                     window. Speech is amplitude-modulated by syllables
                     (10-20 dB swings); a hum or a fan is steady (<2 dB).
    """
    cases = {
        "speech": speech,
        "hum -35 only": hum(SAMPLE_RATE * 8, -35.0),
        "hum -30 only": hum(SAMPLE_RATE * 8, -30.0),
        "noise -35 only": white_noise(SAMPLE_RATE * 8, -35.0, seed=11),
        "noise -25 only": white_noise(SAMPLE_RATE * 8, -25.0, seed=11),
    }
    print(f"{'signal':<16} {'zcr p05':>8} {'zcr p50':>8} {'zcr p95':>8} "
          f"{'mod p05':>8} {'mod p50':>8} {'mod p95':>8}")
    print("-" * 64)
    for name, stream in cases.items():
        zcrs = sorted(z for z in (zero_crossing_rate(f) for f in frames(stream)))
        mods = sorted(sliding_std(level_series(stream), 15))
        if not zcrs or not mods:
            continue
        def pct(values: list[float], q: float) -> float:
            return values[min(int(len(values) * q), len(values) - 1)]
        print(f"{name:<16} {pct(zcrs, 0.05):>8.3f} {pct(zcrs, 0.50):>8.3f} "
              f"{pct(zcrs, 0.95):>8.3f} {pct(mods, 0.05):>7.1f} "
              f"{pct(mods, 0.50):>7.1f} {pct(mods, 0.95):>7.1f}")
    print()


def main(args: argparse.Namespace) -> int:
    speech = read_wav(Path(args.audio))
    # Trim the clip's own leading and trailing silence so the scenario controls
    # the boundaries rather than inheriting the fixture's.
    active = np.where(np.abs(speech.astype(np.int32)) > 512)[0]
    if active.size:
        speech = speech[active[0] : active[-1] + 1]
    speech = scale_to_dbfs(speech, -25.0)

    config = VadConfig(
        silence_ms=args.silence_ms,
        min_speech_ms=args.min_speech_ms,
        hangover_ms=args.hangover_ms,
        threshold_dbfs=args.threshold_dbfs,
        adaptive_margin_db=args.margin_db,
    )

    print(f"speech source: {args.audio} ({speech.size / SAMPLE_RATE * 1000:.0f} ms)")
    print(f"vad: silence={config.silence_ms} min_speech={config.min_speech_ms} "
          f"hangover={config.hangover_ms} threshold={config.threshold_dbfs} dBFS "
          f"margin={config.adaptive_margin_db} dB")
    print()
    print("Where real speech sits:")
    profile = level_profile(speech, config.threshold_dbfs)
    if profile:
        print(f"  frames above -60 dBFS : {profile['frames']}")
        print(f"  RMS p05 / p50 / p95   : {profile['p05']:.1f} / {profile['p50']:.1f} / "
              f"{profile['p95']:.1f} dBFS")
        print(f"  frames below floor    : {profile['below_floor']} "
              f"({100.0 * profile['below_floor'] / max(profile['frames'], 1):.1f}%)")
        print(f"  -> the detector's absolute floor is {config.threshold_dbfs:.0f} dBFS; "
              f"anything quieter")
        print("     can never be heard, however silent the room is.")
    print()

    if not args.skip_discriminators:
        print("Discriminators (can a frame tell speech from steady sound?):")
        discriminator_report(speech)
        print()

    lead = silence(args.lead_ms)
    tail = silence(args.tail_ms)
    lead_ms = lead.size / SAMPLE_RATE * 1000.0
    speech_ms = speech.size / SAMPLE_RATE * 1000.0
    # Ground truth is in *stream* coordinates: (true speech start, true speech
    # end), or (None, None) when the stream must not be detected at all.
    scenarios: list[tuple[str, np.ndarray, float | None, float | None]] = []

    # --- loudness sweep: same words, quieter each time --------------------
    for target in args.levels:
        quiet = scale_to_dbfs(speech, target)
        quiet_ms = quiet.size / SAMPLE_RATE * 1000.0
        scenarios.append((
            f"level {target:+.0f} dBFS",
            concat(lead, quiet, tail),
            lead_ms,
            lead_ms + quiet_ms,
        ))

    # --- signal to noise sweep -------------------------------------------
    for snr in args.snr:
        noise_level = -25.0 - snr
        noisy = speech.astype(np.int32) + white_noise(speech.size, noise_level, seed=7)
        noisy = np.clip(noisy, -32768, 32767).astype(np.int16)
        scenarios.append((
            f"snr  {snr:+.0f} dB",
            concat(lead, noisy, tail),
            lead_ms,
            lead_ms + speech_ms,
        ))

    # --- mains hum, the case the zero-crossing guard exists for -----------
    for level in (-50.0, -40.0, -30.0):
        hummed = speech.astype(np.int32) + hum(speech.size, level)
        hummed = np.clip(hummed, -32768, 32767).astype(np.int16)
        scenarios.append((
            f"hum  {level:+.0f} dBFS",
            concat(lead, hummed, tail),
            lead_ms,
            lead_ms + speech_ms,
        ))

    # --- hum with no speech: the zero-crossing guard on its own ----------
    # The speech+hum rows above only prove speech survives a hum. This proves
    # the hum itself never opens a turn, which is the guard's whole job, and
    # it is the case most sensitive to lowering the threshold (the guard's
    # rejection band is ``threshold + 6``).
    for level in (-50.0, -45.0, -40.0, -35.0, -30.0):
        scenarios.append((
            f"hum only {level:+.0f}",
            hum(SAMPLE_RATE * 8, level),
            None,
            None,
        ))

    # --- no speech at all: false-start rate -------------------------------
    # Longer and at several levels, because false starts accumulate with
    # duration: a 2 s window can look clean while a 10 s one does not.
    for level in (-55.0, -50.0, -45.0, -35.0, -25.0):
        scenarios.append((
            f"noise only {level:+.0f}",
            white_noise(SAMPLE_RATE * 8, level, seed=11),
            None,
            None,
        ))

    # --- impulsive noise: keyboard clacks and door slams ------------------
    # Steady noise is the easy case. The hard one is short transients, which
    # a level-based detector sees as speech for a few frames.
    impulses = white_noise(SAMPLE_RATE * 8, -70.0, seed=5).astype(np.int32)
    for at in (700, 1900, 3400, 5100, 6800):
        start = SAMPLE_RATE * at // 1000
        burst = white_noise(SAMPLE_RATE * 30 // 1000, -28.0, seed=at)
        impulses[start : start + burst.size] += burst.astype(np.int32)
    scenarios.append((
        "impulse noise only",
        np.clip(impulses, -32768, 32767).astype(np.int16),
        None,
        None,
    ))

    # --- quiet speech in a noisy room: the realistic worst case -----------
    quiet = scale_to_dbfs(speech, -42.0)
    room = quiet.astype(np.int32) + white_noise(quiet.size, -50.0, seed=13)
    scenarios.append((
        "quiet speech + room",
        concat(lead, np.clip(room, -32768, 32767).astype(np.int16), tail),
        lead_ms,
        lead_ms + quiet.size / SAMPLE_RATE * 1000.0,
    ))

    # --- speech from frame 0: no silence for the modulation window --------
    scenarios.append((
        "speech from frame 0",
        concat(speech, tail),
        0.0,
        speech_ms,
    ))

    # --- continuous room noise: the noise does not stop when speech does --
    # In the fixtures above the background stops with the speech, which makes
    # endpointing easy. In a real room it keeps going, so the detector has to
    # endpoint *through* it - the case that hangs if steady sound counts as
    # speech.
    for level in (-45.0, -40.0):
        pre = white_noise(lead.size, level, seed=21)
        body = speech.astype(np.int32) + white_noise(speech.size, level, seed=22)
        post = white_noise(tail.size, level, seed=23)
        stream = concat(pre, np.clip(body, -32768, 32767).astype(np.int16), post)
        scenarios.append((
            f"room {level:+.0f} continuous",
            stream,
            pre.size / SAMPLE_RATE * 1000.0,
            (pre.size + speech.size) / SAMPLE_RATE * 1000.0,
        ))

    print(f"{'scenario':<22} {'detected':>8} {'start delay':>12} {'end vs true':>12} {'speech':>8}")
    print("-" * 66)
    failures = 0
    for name, stream, true_start, true_end in scenarios:
        result = run_vad(stream, config)
        detected = result["start"] is not None
        if true_start is None:
            # Nothing should have been detected at all.
            verdict = "no" if not detected else "FALSE START"
            if detected:
                failures += 1
            print(f"{name:<22} {verdict:>8} {'-':>12} {'-':>12} "
                  f"{result['speech_ms']:>7.0f}m")
            continue
        if not detected:
            failures += 1
            print(f"{name:<22} {'MISSED':>8} {'-':>12} {'-':>12} {'-':>8}")
            continue
        start_delay = (result["start"] or 0.0) - true_start
        end_error = ((result["end"] or 0.0) - true_end) if result["end"] else float("nan")
        if start_delay > 200:
            failures += 1
        print(f"{name:<22} {'yes':>8} {start_delay:>10.0f}m {end_error:>10.0f}m "
              f"{result['speech_ms']:>7.0f}m")

    print()
    print(f"problem scenarios: {failures}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", default=str(AUDIO_DIR / "question-16k.wav"))
    parser.add_argument("--silence-ms", type=int, default=800)
    parser.add_argument("--min-speech-ms", type=int, default=120)
    parser.add_argument("--hangover-ms", type=int, default=250)
    parser.add_argument("--threshold-dbfs", type=float, default=-55.0)
    parser.add_argument("--margin-db", type=float, default=9.0)
    parser.add_argument("--skip-discriminators", action="store_true")
    parser.add_argument("--lead-ms", type=int, default=600)
    parser.add_argument("--tail-ms", type=int, default=1200)
    parser.add_argument("--levels", type=float, nargs="*",
                        default=[-20.0, -30.0, -38.0, -44.0, -50.0])
    parser.add_argument("--snr", type=float, nargs="*", default=[30.0, 20.0, 10.0, 5.0])
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))

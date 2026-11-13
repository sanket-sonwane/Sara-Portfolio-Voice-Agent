"""Shared helpers for the Sarang Phase 0 probes.

Run probes from the repository root:

    uv run python -m bench.probe.groq_ttft --help

Every probe emits JSON into ``bench/results/`` so numbers stay comparable across
runs, machines and regions. All durations are measured with ``time.perf_counter``
(monotonic) and reported in milliseconds, per the measurement contract in the
PRD (section 23.3).
"""

from __future__ import annotations

import base64
import json
import math
import os
import platform
import re
import socket
import statistics
import time
import wave
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
AUDIO_DIR = REPO_ROOT / "bench" / "audio"


# --------------------------------------------------------------------------- env


def load_env() -> None:
    """Load ``.env`` from the repository root, without overriding real env vars."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:  # pragma: no cover - dependency is declared
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def require_env(name: str, hint: str = "") -> str:
    load_env()
    value = os.environ.get(name, "").strip()
    if not value:
        message = f"missing environment variable {name}. Copy .env.example to .env and fill it in."
        if hint:
            message += f"\n  {hint}"
        raise SystemExit(message)
    return value


def optional_env(name: str, default: str = "") -> str:
    load_env()
    return os.environ.get(name, default).strip()


# ----------------------------------------------------------------------- clocks


def now() -> float:
    """Monotonic seconds. Never use wall-clock deltas for durations."""
    return time.perf_counter()


def elapsed_ms(t0: float) -> float:
    return (now() - t0) * 1000.0


def utc_stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def machine_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
    }


# ------------------------------------------------------------------------ stats


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` in 0..100."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (q / 100.0)
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def summarise(values: Sequence[float]) -> dict[str, float]:
    """Distribution summary in milliseconds."""
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p90": round(percentile(values, 90), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def format_table(rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> str:
    """Minimal aligned text table (no third-party table dependency)."""
    rows = list(rows)
    if not rows:
        return "(no rows)"
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


# ------------------------------------------------------------------------ text


def words(text: str) -> list[str]:
    """Lowercased alphanumeric tokens; punctuation never scores as a word."""
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i]
        for j, hyp in enumerate(hypothesis, start=1):
            cost = 0 if ref == hyp else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = words(reference)
    return edit_distance(ref, words(hypothesis)) / max(1, len(ref))


# ----------------------------------------------------------------------- result


@dataclass
class Result:
    """One probe run: scalar metrics plus raw per-trial samples."""

    probe: str
    target: str = ""
    label: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    samples: dict[str, list[float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=utc_stamp)

    def add(self, key: str, value: float) -> None:
        self.samples.setdefault(key, []).append(float(value))

    def note(self, message: str) -> None:
        self.notes.append(message)

    def distributions(self) -> dict[str, dict[str, float]]:
        return {key: summarise(values) for key, values in self.samples.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "target": self.target,
            "label": self.label or "local",
            "started_at": self.started_at,
            "host": machine_info(),
            "metrics": self.metrics,
            "distributions_ms": self.distributions(),
            "samples_ms": {key: [round(v, 3) for v in values] for key, values in self.samples.items()},
            "notes": self.notes,
        }

    def write(self, stem: str | None = None) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = self.label or "local"
        path = RESULTS_DIR / f"{stem or self.probe}-{label}-{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def report(self, path: Path, columns: Sequence[str] | None = None) -> None:
        print()
        if columns:
            rows = [
                {"metric": key, **value} for key, value in self.distributions().items()
            ]
            print(format_table(rows, columns))
        else:
            print(format_table([{"metric": key, "value": value} for key, value in self.metrics.items()], ["metric", "value"]))
        print(f"\nJSON: {path}")


# ------------------------------------------------------------------------ audio


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM16 WAV as mono int16 samples plus sample rate."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"audio file not found: {path}")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    data = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype("<i2")
    return data, rate


def write_wav(path: str | Path, samples: np.ndarray, rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Streaming-quality resample of int16 PCM."""
    if src_rate == dst_rate:
        return samples.astype("<i2")
    import soxr

    as_float = samples.astype(np.float32) / 32768.0
    converted = soxr.resample(as_float, src_rate, dst_rate, quality="HQ")
    return np.clip(converted * 32768.0, -32768, 32767).astype("<i2")


def load_wav_16k(path: str | Path) -> np.ndarray:
    samples, rate = read_wav(path)
    return resample(samples, rate, 16000)


def pcm_bytes(samples: np.ndarray) -> bytes:
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def chunks(samples: np.ndarray, chunk_ms: int, rate: int) -> Iterator[bytes]:
    step = max(1, int(rate * chunk_ms / 1000))
    for start in range(0, len(samples), step):
        yield pcm_bytes(samples[start : start + step])


def duration_ms(samples: np.ndarray, rate: int) -> float:
    return len(samples) / float(rate) * 1000.0


def b64_pcm16(payload: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(payload), dtype="<i2")


# ------------------------------------------------------------------ formatting


def ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"

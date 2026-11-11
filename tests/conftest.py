"""Shared fixtures.

Everything here runs with zero API keys. The local profile exercises the real
turn engine with the deterministic answerer and the synthetic voice, which is
the same code path a production deployment falls back to.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest
from sarang.bus import Bus
from sarang.config import resolve_settings
from sarang.orchestrator import Orchestrator
from sarang.schemas import Envelope
from sarang.session import Session, SessionRegistry


def local_settings(**overrides: str):
    env = {
        "SESSION_SECRET": "test-secret",
        "SARANG_PROFILE": "local",
        "SARANG_ENV": "test",
        "ENDPOINT_SILENCE_MS": "300",
        **overrides,
    }
    return resolve_settings(env)


@pytest.fixture
def settings():
    return local_settings()


@pytest.fixture
async def orchestrator(settings) -> AsyncIterator[Orchestrator]:
    bus = Bus()
    registry = SessionRegistry(settings, bus)
    engine = Orchestrator(settings, registry)
    await engine.startup()
    try:
        yield engine
    finally:
        await engine.shutdown()


@pytest.fixture
def session(orchestrator: Orchestrator) -> Session:
    return orchestrator.create_session("test")


async def drain(bus: Bus, key: str, limit: int = 500) -> list[Envelope]:
    """Read every event currently queued, without waiting for more."""
    out: list[Envelope] = []
    queue = bus._subscribers[key]  # tests may look at the queue directly
    for _ in range(limit):
        try:
            out.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return out


def tone(ms: int = 20, rate: int = 16_000, amplitude: int = 9_000) -> np.ndarray:
    count = rate * ms // 1000
    t = np.arange(count, dtype=np.float32) / rate
    signal = np.sin(2 * np.pi * 220.0 * t) * amplitude
    return signal.astype(np.int16)


def vocal(ms: int = 20, rate: int = 16_000, amplitude: int = 12_000,
          offset_ms: float = 0.0) -> np.ndarray:
    """A speech stand-in: a tone with a syllabic tremolo.

    The VAD rejects steady sound by its lack of level modulation over a 300 ms
    window, so test "speech" has to modulate the way speech does - a plain
    ``tone()`` is exactly what the detector must ignore. ``offset_ms`` keeps the
    modulation continuous across consecutive frames; pass ``i * 20`` when
    pushing frame ``i``.
    """
    count = rate * ms // 1000
    start = int(rate * offset_ms // 1000)
    t = np.arange(start, start + count, dtype=np.float32) / rate
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 12.0 * t)
    signal = np.sin(2 * np.pi * 220.0 * t) * amplitude * envelope
    return signal.astype(np.int16)


def silence(ms: int = 20, rate: int = 16_000) -> np.ndarray:
    return np.zeros(rate * ms // 1000, dtype=np.int16)

"""Voice activity detection and endpointing.

Energy-based with an adaptive noise floor, plus a zero-crossing guard so a fan or
a hum does not register as speech. Deliberately dependency-free: a neural VAD adds
20-40 ms and a model file, and the endpointing floor (silence we must observe) is
the dominant cost anyway.

The timings are the product. ``silence_ms`` is how long we wait after speech stops
before deciding the turn is over, and it is the single biggest term in the latency
budget. 700 ms is conversational; 500 ms starts cutting people off mid-thought.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class VadConfig:
    threshold_dbfs: float = -42.0
    adaptive_margin_db: float = 9.0
    min_speech_ms: int = 180
    silence_ms: int = 700
    partial_after_ms: int = 320
    hangover_ms: int = 140


@dataclass
class VadUpdate:
    speech: bool = False
    speech_started: bool = False
    speech_ended: bool = False
    partial_due: bool = False
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    level_dbfs: float = -90.0

    def to_payload(self) -> dict:
        return {
            "speech": self.speech,
            "speech_ms": round(self.speech_ms, 1),
            "silence_ms": round(self.silence_ms, 1),
            "level_dbfs": round(self.level_dbfs, 1),
        }


def dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -90.0
    float_samples = samples.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(float_samples * float_samples) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-6))


def zero_crossing_rate(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 0.0
    signs = np.signbit(samples.astype(np.int32))
    return float(np.mean(signs[1:] != signs[:-1]))


class Vad:
    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        self.noise_floor = -55.0
        self.active = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._hangover_ms = 0.0
        self._partial_sent = False

    def reset(self) -> None:
        self.active = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._hangover_ms = 0.0
        self._partial_sent = False

    def _is_speech(self, level: float, zcr: float) -> bool:
        threshold = max(self.config.threshold_dbfs, self.noise_floor + self.config.adaptive_margin_db)
        # A pure tone or a very steady hum leaks energy with almost no crossings.
        if zcr < 0.005 and level < threshold + 6:
            return False
        return level > threshold

    def push(self, samples: np.ndarray, sample_rate: int = 16_000) -> VadUpdate:
        step_ms = len(samples) / sample_rate * 1000.0
        level = dbfs(samples)
        zcr = zero_crossing_rate(samples)
        speech = self._is_speech(level, zcr)

        update = VadUpdate(level_dbfs=level)
        if not self.active:
            # Track the floor only while idle, so speech never raises it.
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * level
            self.noise_floor = max(-75.0, min(-30.0, self.noise_floor))
            if speech:
                self._speech_ms += step_ms
                if self._speech_ms >= self.config.min_speech_ms:
                    self.active = True
                    self._silence_ms = 0.0
                    self._hangover_ms = 0.0
                    self._partial_sent = False
                    update.speech_started = True
            else:
                self._speech_ms = max(0.0, self._speech_ms - step_ms * 1.5)
        else:
            if speech:
                self._silence_ms = 0.0
                self._hangover_ms = 0.0
                self._speech_ms += step_ms
            else:
                self._hangover_ms += step_ms
                if self._hangover_ms >= self.config.hangover_ms:
                    self._silence_ms += step_ms

            if not self._partial_sent and self._speech_ms >= self.config.partial_after_ms:
                self._partial_sent = True
                update.partial_due = True

            if self._silence_ms >= self.config.silence_ms:
                update.speech_ended = True
                self.active = False
                self._speech_ms = 0.0
                self._silence_ms = 0.0
                self._hangover_ms = 0.0
                self._partial_sent = False

        update.speech = self.active
        update.speech_ms = self._speech_ms
        update.silence_ms = self._silence_ms
        return update

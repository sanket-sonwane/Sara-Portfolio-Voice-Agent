"""Voice activity detection and endpointing.

Energy-based with an adaptive noise floor, plus two guards that keep a steady
sound from being mistaken for speech. Deliberately dependency-free: a neural VAD
adds 20-40 ms and a model file, and the endpointing floor (silence we must
observe) is the dominant cost anyway.

Two properties are measured, not assumed (``bench/probe/vad_accuracy.py``):

* **The absolute floor is −55 dBFS.** Real speech sits at RMS p05/p50/p95 =
  −47/−27/−20 dBFS, so a −42 dBFS floor went deaf to 11% of speech frames and
  missed whole quiet utterances. The floor only binds in a *quiet* room - in a
  noisy one ``noise_floor + adaptive_margin_db`` is higher - so lowering it is
  free there; what it buys is hearing a quiet speaker at all.
* **Steady is not speech.** A 50 Hz hum or a fan is level-correct and
  zero-crosses like neither speech nor noise; what separates them is
  *modulation*: speech swings ≥3 dB across a 300 ms window (syllables), steady
  sound <0.5 dB. Without this gate, ambient sound at or above the floor opened a
  turn that - because every subsequent frame was also "speech" - never
  endpointed at all: the loudest possible failure of turn detection. The gate
  also keeps the low floor safe: room hiss at −50 dBFS is level-speech and
  steady-not-speech.

The timings are the product. ``silence_ms`` is how long we wait after speech stops
before deciding the turn is over, and it is the single biggest term in the latency
budget. 700 ms is conversational; 500 ms starts cutting people off mid-thought.

``silence_ms`` is measured **from the last frame above threshold**, not from the
end of the hangover. Counting them in series made ``ENDPOINT_SILENCE_MS=800``
behave like 1 050 ms once the 250 ms hangover was added, which was 250 ms of pure
latency nobody had budgeted. The hangover survives only as a floor: it stops a
configuration with a very short ``silence_ms`` from clipping soft word endings.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class VadConfig:
    threshold_dbfs: float = -55.0
    adaptive_margin_db: float = 9.0
    min_speech_ms: int = 180
    silence_ms: int = 700
    partial_after_ms: int = 320
    hangover_ms: int = 140
    steady_window_ms: int = 300
    steady_max_db: float = 1.5


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
        self._clock_ms = 0.0
        self._levels: deque[float] = deque()
        self._level_times: deque[float] = deque()

    def reset(self) -> None:
        self.active = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._hangover_ms = 0.0
        self._partial_sent = False
        self._levels.clear()
        self._level_times.clear()

    def _modulation(self) -> float:
        """Level spread over the recent window, in dB.

        0.0 for steady sound, several dB for speech. Two frames are enough to
        judge: a lone frame can never prove it is steady, but the window fills
        within ``steady_window_ms`` and the silence-to-speech transition on any
        real onset spikes the spread wide open.
        """
        if len(self._levels) < 2:
            return 0.0
        return float(np.std(list(self._levels)))

    def _is_speech(self, level: float, zcr: float, modulation: float) -> bool:
        threshold = max(
            self.config.threshold_dbfs,
            self.noise_floor + self.config.adaptive_margin_db,
        )
        if level <= threshold:
            return False
        # A pure tone or a very steady hum leaks energy with almost no crossings.
        # Only reject when barely above threshold; clearly loud signals should
        # still be considered.
        if zcr < 0.005 and level < threshold + 6:
            return False
        # Steady sound is never speech: a hum, a fan or a noise floor sits at a
        # constant level where syllables swing. Measured separation is wide
        # (speech p05 = 3.1 dB over 300 ms, noise p95 = 0.4 dB), and without it
        # ambient noise at or above the floor opens a turn that never ends.
        if modulation < self.config.steady_max_db:
            return False
        return True

    def push(self, samples: np.ndarray, sample_rate: int = 16_000) -> VadUpdate:
        step_ms = len(samples) / sample_rate * 1000.0
        level = dbfs(samples)
        zcr = zero_crossing_rate(samples)

        self._clock_ms += step_ms
        self._levels.append(level)
        self._level_times.append(self._clock_ms)
        cutoff = self._clock_ms - self.config.steady_window_ms
        while self._level_times and self._level_times[0] <= cutoff:
            self._level_times.popleft()
            self._levels.popleft()
        modulation = self._modulation()

        speech = self._is_speech(level, zcr, modulation)

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
                # Both clocks run from the last frame above threshold. Running
                # them in series made the effective endpoint
                # ``hangover_ms + silence_ms``, which is not what any operator
                # reading ENDPOINT_SILENCE_MS expects.
                self._hangover_ms += step_ms
                self._silence_ms += step_ms

            if not self._partial_sent and self._speech_ms >= self.config.partial_after_ms:
                self._partial_sent = True
                update.partial_due = True

            if self._silence_ms >= max(
                self.config.silence_ms, self.config.hangover_ms
            ):
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

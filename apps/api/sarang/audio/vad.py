"""Voice activity detection and endpointing.

Two detectors share one contract (``reset`` / ``push`` / ``VadUpdate``):

* ``SileroVad`` - the default. A small ONNX model scores each 32 ms window on
  the signal itself, so it is level-invariant and needs no modulation gate.
  It is what makes a 500 ms silence window safe (``bench/probe/vad_accuracy.py``,
  ADR 0003 section 9).
* ``Vad`` - the energy fallback (``VAD_MODE=energy``), kept dependency-free for
  the bench reference arm and for the case where the model cannot load.

The energy detector below is documented in its own terms because its
limitations are the reason the neural path exists.

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

import hashlib
import logging
import math
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

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
    # Silero probability above which a 32 ms window counts as speech. 0.5 is
    # the model's own default; it is a probability, not a level.
    silero_threshold: float = 0.5


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
    detector = "energy"

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


# --------------------------------------------------------------------- neural
#
# The energy detector above is kept as the fallback (``VAD_MODE=energy``) and as
# the reference arm in ``bench/probe/vad_accuracy.py``. It has two properties
# that no amount of tuning fixes: its threshold is an absolute level learned
# from the room (a fan can lift it above the speaker), and it needs a
# *modulation* gate to reject steady sound - the same gate that marks a
# sustained vowel as silence mid-word.
#
# Silero VAD scores 32 ms windows on the signal itself, so it is level-
# invariant and needs no such gate. ADR 0003 section 9 names it as the next
# quality step and predicts exactly the outcome we need: a silence window near
# 500 ms without clipping.

SILERO_WINDOW = 512           # samples at 16 kHz; the model's fixed window
SILERO_CONTEXT = 64           # samples of the previous window, prepended (v5)
SILERO_WINDOW_MS = 32.0
SILERO_MODEL_SHA256 = (
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
)
SILERO_MODEL_PATH = Path(__file__).with_name("models") / "silero_vad.onnx"

log = logging.getLogger(__name__)

# One ONNX session per model file, shared by every connection. The detector
# state (``_state``, ``_context``) is per instance; the graph and its weights are
# immutable, and ``InferenceSession.run`` is thread-safe. Without this cache each
# WebSocket connection paid the model load again.
_SESSIONS: dict[Path, object] = {}


def _load_silero_session(model_path: Path):
    import onnxruntime as ort

    cached = _SESSIONS.get(model_path)
    if cached is not None:
        return cached
    if not model_path.is_file():
        raise FileNotFoundError(f"silero VAD model not found: {model_path}")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != SILERO_MODEL_SHA256:
        raise RuntimeError(
            f"silero VAD model hash mismatch: {digest} != {SILERO_MODEL_SHA256}"
        )
    options = ort.SessionOptions()
    # One thread: the call is ~1 ms on a 32 ms window and running it inline on
    # the event loop is cheaper than a thread hop.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    _SESSIONS[model_path] = session
    return session


class SileroVad:
    """Neural voice activity detection with the same contract as :class:`Vad`.

    ``predict`` accepts a 512-sample int16 window and returns the probability
    that it contains speech. Tests inject one so they never need the model.
    """

    detector = "silero"

    def __init__(
        self,
        config: VadConfig | None = None,
        *,
        model_path: Path | str | None = None,
        predict=None,
    ) -> None:
        self.config = config or VadConfig()
        self.threshold = self.config.silero_threshold
        self.active = False
        self._remainder = np.zeros(0, dtype=np.int16)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(SILERO_CONTEXT, dtype=np.float32)
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._hangover_ms = 0.0
        self._partial_sent = False
        self._probability = 0.0
        self._level = -90.0
        if predict is None:
            session = _load_silero_session(Path(model_path or SILERO_MODEL_PATH))
            self._session = session

            def predict(window: np.ndarray) -> float:  # noqa: F811
                audio = window.astype(np.float32) / 32768.0
                # v5 is trained on 512 samples *plus* the last 64 samples of
                # the previous window; without that context the model scores
                # real speech at 0.0.
                model_input = np.concatenate((self._context, audio)).reshape(1, -1)
                self._context = audio[-SILERO_CONTEXT:]
                output, self._state = session.run(
                    None,
                    {
                        "input": model_input,
                        "state": self._state,
                        "sr": np.array(16_000, dtype=np.int64),
                    },
                )
                return float(output[0, 0])

            self._predict = predict
        else:
            self._session = None
            self._predict = predict

    def reset(self) -> None:
        self.active = False
        self._remainder = np.zeros(0, dtype=np.int16)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(SILERO_CONTEXT, dtype=np.float32)
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._hangover_ms = 0.0
        self._partial_sent = False
        self._probability = 0.0

    def push(self, samples: np.ndarray, sample_rate: int = 16_000) -> VadUpdate:
        if sample_rate != 16_000:
            raise ValueError("silero VAD expects 16 kHz")
        update = VadUpdate()
        self._level = dbfs(samples)
        if samples.size:
            self._remainder = np.concatenate((self._remainder, samples))
        while self._remainder.size >= SILERO_WINDOW:
            window = self._remainder[:SILERO_WINDOW]
            self._remainder = self._remainder[SILERO_WINDOW:]
            self._probability = self._predict(window)
            self._step(update, self._probability)

        update.speech = self.active
        update.speech_ms = self._speech_ms
        update.silence_ms = self._silence_ms
        update.level_dbfs = self._level
        return update

    def _step(self, update: VadUpdate, probability: float) -> None:
        positive = probability >= self.threshold
        if not self.active:
            if positive:
                self._speech_ms += SILERO_WINDOW_MS
                if self._speech_ms >= self.config.min_speech_ms:
                    self.active = True
                    self._silence_ms = 0.0
                    self._hangover_ms = 0.0
                    self._partial_sent = False
                    update.speech_started = True
            else:
                # A gap before the commit restarts the run: a door slam must
                # not accumulate into a turn.
                self._speech_ms = 0.0
            return

        if positive:
            self._silence_ms = 0.0
            self._hangover_ms = 0.0
            self._speech_ms += SILERO_WINDOW_MS
        else:
            self._hangover_ms += SILERO_WINDOW_MS
            self._silence_ms += SILERO_WINDOW_MS

        if not self._partial_sent and self._speech_ms >= self.config.partial_after_ms:
            self._partial_sent = True
            update.partial_due = True

        if self._silence_ms >= max(self.config.silence_ms, self.config.hangover_ms):
            update.speech_ended = True
            self.active = False
            self._speech_ms = 0.0
            self._silence_ms = 0.0
            self._hangover_ms = 0.0
            self._partial_sent = False

    @property
    def probability(self) -> float:
        return self._probability


def build_vad(
    config: VadConfig,
    *,
    mode: str = "silero",
    model_path: Path | str | None = None,
):
    """Return the configured detector, degrading loudly to the energy VAD.

    The silence window is only safe below 800 ms with the neural detector: the
    measured 460 ms internal pause in the reference question is invisible to an
    energy VAD but not to Silero. If the model cannot load we therefore do not
    just swap detectors - we also restore the conservative window, so a missing
    ``onnxruntime`` cannot silently truncate every question.
    """
    if mode == "energy":
        return Vad(config)
    try:
        detector = SileroVad(config, model_path=model_path)
    except Exception as error:  # noqa: BLE001 - a missing model must not stop the app
        fallback = config
        if config.silence_ms < 800:
            log.warning(
                "silero VAD unavailable (%s); falling back to the energy VAD and "
                "restoring endpoint silence to 800 ms (was %d ms)",
                error,
                config.silence_ms,
            )
            fallback = replace(config, silence_ms=800)
        else:
            log.warning("silero VAD unavailable (%s); falling back to the energy VAD", error)
        return Vad(fallback)
    log.info(
        "vad: silero (threshold %.2f, silence %d ms)",
        config.silero_threshold,
        config.silence_ms,
    )
    return detector

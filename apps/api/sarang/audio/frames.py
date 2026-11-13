"""Audio primitives: framing, conversion, resampling.

Two hard rules from the media design:

* 48 kHz mono PCM16 everywhere at the edges, 16 kHz mono to ASR.
* Exactly 960 samples per 20 ms frame on the egress track. A frame of the wrong
  size desynchronises the browser's playout; ``AudioStreamTrack`` is the only
  writer and it pads with silence rather than varying size.
"""

from __future__ import annotations

import numpy as np

INT16_MAX = 32767
INT16_MIN = -32768

SAMPLE_RATE_TRACK = 48_000
SAMPLE_RATE_ASR = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE_TRACK * FRAME_MS // 1000  # 960


def to_float(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    return samples.astype(np.float32)


def to_int16(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        return samples
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def downmix(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return samples
    return samples.reshape(-1, channels).mean(axis=1)


class StreamResampler:
    """Stateful resampler for a continuous stream.

    Per-chunk resampling produces clicks at chunk boundaries because the filter
    state is thrown away each time. ``soxr.ResampleStream`` keeps it.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._stream = None
        if src_rate == dst_rate:
            return
        try:
            import soxr

            self._stream = soxr.ResampleStream(
                src_rate, dst_rate, 1, dtype="float32", quality="HQ"
            )
        except Exception:
            self._stream = None

    def process(self, samples: np.ndarray) -> np.ndarray:
        if self.src_rate == self.dst_rate:
            return samples.astype(np.int16)
        as_float = to_float(samples)
        if self._stream is not None:
            converted = self._stream.resample_chunk(as_float, last=False)
        else:  # pragma: no cover - soxr is a declared dependency
            import soxr

            converted = soxr.resample(as_float, self.src_rate, self.dst_rate, quality="HQ")
        return to_int16(converted)


    def flush(self) -> np.ndarray:
        """Return the samples still held in the resampler's filter delay."""
        if self.src_rate == self.dst_rate or self._stream is None:
            return np.zeros(0, dtype=np.int16)
        converted = self._stream.resample_chunk(
            np.zeros(0, dtype=np.float32), last=True
        )
        return to_int16(converted)


def resample_array(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.int16)
    import soxr

    converted = soxr.resample(to_float(samples), src_rate, dst_rate, quality="HQ")
    return to_int16(converted)


def prepend_mono(samples: np.ndarray, target_len: int) -> np.ndarray:
    """Return exactly ``target_len`` samples, padding with silence or truncating."""
    if len(samples) == target_len:
        return samples
    if len(samples) > target_len:
        return samples[:target_len]
    pad = np.zeros(target_len - len(samples), dtype=np.int16)
    return np.concatenate([samples, pad])


def silent_frame(samples: int = FRAME_SAMPLES) -> np.ndarray:
    return np.zeros(samples, dtype=np.int16)


def read_int16(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.int16)


def frame_bytes(samples: np.ndarray) -> bytes:
    return np.clip(samples, INT16_MIN, INT16_MAX).astype("<i2").tobytes()

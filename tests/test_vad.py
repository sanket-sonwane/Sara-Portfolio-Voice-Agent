"""VAD and endpointing.

The speech fixture is ``vocal`` (a tremolo'd tone), never a steady ``tone``:
the detector must reject steady sound, so a plain tone is the hum case, not the
speech case. Utterances are preceded by a few silence frames the way a real
transport is - the modulation window needs something to contrast against, and
every real turn starts after silence.
"""

from __future__ import annotations

import numpy as np
from sarang.audio.vad import Vad, VadConfig, dbfs, zero_crossing_rate

from tests.conftest import silence, tone, vocal


def _speak(vad: Vad, frames: int = 12) -> bool:
    """Push silence then modulated speech; report whether the turn opened."""
    started = False
    for _ in range(6):
        vad.push(silence(20))
    for i in range(frames):
        update = vad.push(vocal(20, offset_ms=i * 20.0))
        started = started or update.speech_started
    return started


def test_dbfs_ranks_tone_above_silence():
    assert dbfs(tone()) > dbfs(silence())
    assert dbfs(silence()) < -80


def test_zero_crossing_rate_separates_noise_from_dc():
    assert zero_crossing_rate(tone()) > 0.01
    assert zero_crossing_rate(silence() + 100) == 0.0


def test_endpointing_detects_start_and_end():
    vad = Vad(VadConfig(min_speech_ms=100, silence_ms=200, hangover_ms=40))
    started = _speak(vad, frames=10)
    assert started
    assert vad.active
    ended = False
    for _ in range(20):
        update = vad.push(silence(20))
        ended = ended or update.speech_ended
    assert ended
    assert not vad.active


def test_short_blip_does_not_start_speech():
    vad = Vad(VadConfig(min_speech_ms=180))
    started = False
    for i in range(3):
        update = vad.push(vocal(20, offset_ms=i * 20.0))
        started = started or update.speech_started
    assert not started
    assert not vad.active


def test_partial_transcript_fires_once_per_utterance():
    vad = Vad(VadConfig(min_speech_ms=60, partial_after_ms=60, silence_ms=400))
    partials = 0
    for _ in range(6):
        vad.push(silence(20))
    for i in range(12):
        update = vad.push(vocal(20, offset_ms=i * 20.0))
        partials += int(update.partial_due)
    assert partials == 1


def test_reset_clears_state():
    vad = Vad(VadConfig(min_speech_ms=60))
    _speak(vad)
    vad.reset()
    assert not vad.active
    assert vad._speech_ms == 0.0
    assert not vad._levels


def _frames_to_end(vad: Vad, limit: int = 100) -> float | None:
    """Push silence until the turn ends and report how much silence it took."""
    elapsed = 0.0
    for _ in range(limit):
        update = vad.push(silence(20))
        elapsed += 20.0
        if update.speech_ended:
            return elapsed
    return None


def test_endpoint_is_measured_from_the_last_speech_frame():
    """The hangover is a floor, not an extra delay on top of the silence window.

    Counting them in series made ENDPOINT_SILENCE_MS=800 behave like 1 050 ms.
    """
    vad = Vad(VadConfig(min_speech_ms=60, silence_ms=200, hangover_ms=240))
    assert _speak(vad, frames=6)
    assert vad.active

    assert _frames_to_end(vad) == 240.0


def test_silence_window_longer_than_hangover_ends_at_the_silence_window():
    vad = Vad(VadConfig(min_speech_ms=60, silence_ms=400, hangover_ms=100))
    assert _speak(vad, frames=6)
    assert vad.active

    assert _frames_to_end(vad) == 400.0


# ------------------------------------------------------------------ detection
# These lock in the findings of bench/probe/vad_accuracy.py: the detector must
# hear quiet speech (the old -42 dBFS floor went deaf to it) and must never
# open a turn on steady sound - a turn opened on a fan or a hum never
# endpoints, because every following frame is also "speech".


def test_detects_speech_below_the_old_floor():
    """Speech at -50 dBFS must open a turn; the old -42 floor missed it."""
    vad = Vad(VadConfig(min_speech_ms=60))
    for _ in range(20):
        vad.push(silence(20))
    started = False
    # amplitude 145 puts a sine at about -50 dBFS RMS.
    for i in range(40):
        update = vad.push(vocal(20, amplitude=145, offset_ms=i * 20.0))
        started = started or update.speech_started
    assert started
    assert vad.active


def test_steady_noise_never_starts_a_turn():
    """Ambient noise at -35 dBFS opens no turn and never hangs the loop.

    Pre-gate this was the worst failure available: the noise floor needed
    780 ms to climb past the level test, ``min_speech_ms`` committed in
    120 ms, and once active every frame was "speech" - so the utterance
    never endpointed at all.
    """
    vad = Vad(VadConfig(min_speech_ms=60))
    rng = np.random.default_rng(7)
    rms = 10.0 ** (-35.0 / 20.0) * 32768.0  # -35 dBFS RMS
    for _ in range(400):  # 8 s of it
        frame = (rng.standard_normal(320) * rms).astype(np.int16)
        update = vad.push(frame)
        assert not update.speech_started
    assert not vad.active


def test_steady_hum_never_starts_a_turn():
    """A 50 Hz mains hum at -30 dBFS is level-speech and steady-not-speech."""
    vad = Vad(VadConfig(min_speech_ms=60))
    t = np.arange(320, dtype=np.float32) / 16_000
    hum = (np.sin(2 * np.pi * 50.0 * t) * 823.0).astype(np.int16)  # -35 dBFS
    for _ in range(400):
        update = vad.push(hum)
        assert not update.speech_started
    assert not vad.active


def test_speech_from_the_first_frame_still_starts():
    """With no lead-in silence the modulation window opens late, but bounded.

    The onset window has nothing to contrast against until the syllabic
    tremolo spans it, so activation waits for the window to fill - a few
    hundred milliseconds, which the pre-roll buffer replays to the recogniser.
    """
    vad = Vad(VadConfig(min_speech_ms=60, silence_ms=400))
    started = False
    for i in range(30):  # 600 ms of speech from frame 0
        update = vad.push(vocal(20, offset_ms=i * 20.0))
        started = started or update.speech_started
    assert started
    assert vad.active

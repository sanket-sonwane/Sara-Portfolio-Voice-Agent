"""VAD and endpointing."""

from __future__ import annotations

from sarang.audio.vad import Vad, VadConfig, dbfs, zero_crossing_rate

from tests.conftest import silence, tone


def test_dbfs_ranks_tone_above_silence():
    assert dbfs(tone()) > dbfs(silence())
    assert dbfs(silence()) < -80


def test_zero_crossing_rate_separates_noise_from_dc():
    assert zero_crossing_rate(tone()) > 0.01
    assert zero_crossing_rate(silence() + 100) == 0.0


def test_endpointing_detects_start_and_end():
    vad = Vad(VadConfig(min_speech_ms=100, silence_ms=200, hangover_ms=40))
    started = False
    ended = False
    for _ in range(10):
        update = vad.push(tone(20))
        started = started or update.speech_started
    assert started
    assert vad.active
    for _ in range(20):
        update = vad.push(silence(20))
        ended = ended or update.speech_ended
    assert ended
    assert not vad.active


def test_short_blip_does_not_start_speech():
    vad = Vad(VadConfig(min_speech_ms=180))
    started = False
    for _ in range(3):
        update = vad.push(tone(20))
        started = started or update.speech_started
    assert not started
    assert not vad.active


def test_partial_transcript_fires_once_per_utterance():
    vad = Vad(VadConfig(min_speech_ms=60, partial_after_ms=60, silence_ms=400))
    partials = 0
    for _ in range(12):
        update = vad.push(tone(20))
        partials += int(update.partial_due)
    assert partials == 1


def test_reset_clears_state():
    vad = Vad(VadConfig(min_speech_ms=60))
    for _ in range(6):
        vad.push(tone(20))
    vad.reset()
    assert not vad.active
    assert vad._speech_ms == 0.0

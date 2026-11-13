"""Turn marks: stage timing that is safe to read."""

from __future__ import annotations

import time

from sarang.metrics import MetricsStore, TurnMarks, percentile


def test_exit_open_closes_a_stage_that_was_entered():
    marks = TurnMarks("trn_1")
    marks.enter("segment")
    time.sleep(0.02)

    duration = marks.exit_open("segment")

    assert duration is not None and duration >= 15.0
    assert set(marks.offsets()) == {"segment"}


def test_exit_open_ignores_a_stage_that_never_started():
    """An unentered stage must not be measured from the turn origin.

    ``exit`` falls back to the origin, which turns "never happened" into a
    number that looks like a real stage cost. That is how a missing mark became
    a plausible-looking latency regression once already.
    """
    marks = TurnMarks("trn_1")

    assert marks.exit_open("segment") is None
    assert marks.offsets() == {}
    assert marks.durations() == {}


def test_exit_open_does_not_overwrite_a_closed_stage():
    marks = TurnMarks("trn_1")
    marks.enter("segment")
    first = marks.exit_open("segment")
    time.sleep(0.01)

    assert marks.exit_open("segment") is None
    assert round(first, 1) == marks.durations()["segment"]


def test_percentile_matches_nearest_rank_expectations():
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_mark_endpoint_opens_at_the_turn_origin():
    """The silence window is the first segment of the turn, not a back-dated one.

    Origin is the last speech frame, so the endpoint stage starts there and
    ends when the detector declared the turn over. Everything after it (ASR,
    model, TTS, egress) then reads as disjoint time on one axis; before this,
    the window was charged to whichever stage happened to be measured next.
    """
    origin = 100.0
    marks = TurnMarks("trn_1", origin=origin)

    marks.mark_endpoint(812.0)

    assert round(marks.durations()["endpoint"], 1) == 812.0
    assert marks.stage_enter["endpoint"] == origin
    assert marks.stage_exit["endpoint"] == origin + 0.812


def test_timeline_places_every_stage_after_the_last_speech_frame():
    marks = TurnMarks("trn_1", origin=100.0)
    marks.mark_end_of_speech(100.0)
    marks.mark_endpoint(500.0)
    marks.enter("asr_final")
    marks.stage_enter["asr_final"] = 100.5
    marks.stage_exit["asr_final"] = 100.8

    timeline = marks.timeline()

    assert timeline["endpoint"] == 500.0
    assert timeline["asr_final"] == 800.0
    # And the payload carries it, so a report never has to recompute clocks.
    assert marks.to_payload()["timeline_ms"]["asr_final"] == 800.0


def test_mark_endpoint_ignores_a_missing_window():
    """No observed window (e.g. a typed turn) must not invent a stage."""
    marks = TurnMarks("trn_1")

    marks.mark_endpoint(0.0)

    assert marks.durations() == {}


def test_first_audio_names_are_split_and_aliased():
    """Synthesis, egress selection and the wire are three different events.

    They used to share ``first_audio``, so a fast provider looked like fast
    playback. The payload names each event, and keeps the deprecated key for
    one release so stored results still read.
    """
    marks = TurnMarks("trn_1", origin=100.0)
    marks.mark_end_of_speech(100.0)
    marks.mark_tts_first_span()
    marks.tts_first_span = 100.2
    marks.answer_first_audio = 100.2
    marks.mark_first_audio()
    marks.first_audio = 100.3
    marks.mark_transport_first_frame()
    marks.transport_first_frame = 100.4

    payload = marks.to_payload()

    assert payload["tts_first_span_ms"] == 200.0
    assert payload["egress_first_frame_ms"] == 300.0
    assert payload["transport_first_frame_ms"] == 400.0
    # ``answer_first_audio`` keeps pointing at the provider span, named for it.
    assert payload["answer_first_audio_ms"] == 200.0
    # Deprecated aliases mean "egress selected the frame", not "it was heard".
    assert payload["first_audio_ms"] == payload["egress_first_frame_ms"]
    assert payload["eos_to_first_audio_ms"] == 300.0


def test_transport_first_frame_is_not_reported_without_an_egress_frame():
    marks = TurnMarks("trn_1", origin=100.0)
    marks.mark_transport_first_frame()
    marks.transport_first_frame = 100.0

    payload = marks.to_payload()

    assert payload["transport_first_frame_ms"] == 0.0
    assert "egress_first_frame_ms" not in payload
    assert "first_audio_ms" not in payload


def test_client_deltas_are_added_to_the_server_axis():
    """The browser cannot share the server clock, so it reports deltas.

    Adding them to ``end_of_speech -> transport_first_frame`` puts the number
    the visitor experienced on the one axis a report should read, without
    pretending the two clocks are the same.
    """
    marks = TurnMarks("trn_1", origin=100.0)
    marks.mark_end_of_speech(100.0)
    marks.mark_transport_first_frame()
    marks.transport_first_frame = 100.9  # 900 ms after the last speech frame
    marks.client_scheduled_ms = 40.0
    marks.client_audible_ms = 85.0

    payload = marks.to_payload()

    assert payload["eos_to_transport_first_frame_ms"] == 900.0
    assert payload["client_scheduled_ms"] == 40.0
    assert payload["eos_to_client_scheduled_ms"] == 940.0
    assert payload["client_audible_ms"] == 85.0
    assert payload["eos_to_client_audible_ms"] == 985.0


def test_client_deltas_without_a_transport_stamp_are_not_summed():
    """No server anchor means no combined number; the delta is still recorded."""
    marks = TurnMarks("trn_1", origin=100.0)
    marks.client_audible_ms = 50.0

    payload = marks.to_payload()

    assert payload["client_audible_ms"] == 50.0
    assert "eos_to_client_audible_ms" not in payload
    assert "eos_to_client_scheduled_ms" not in payload


def test_degraded_is_only_reported_when_the_turn_was_distorted():
    assert "degraded" not in TurnMarks("trn_clean").to_payload()

    marks = TurnMarks("trn_cut")
    marks.degraded = "llm_midstream"
    assert marks.to_payload()["degraded"] == "llm_midstream"


def test_observe_client_keeps_the_late_report_with_its_turn():
    """The audible report arrives after ``metrics.turn``; it must land there."""
    store = MetricsStore(capacity=10)
    marks = TurnMarks("trn_1", origin=100.0)
    marks.mark_end_of_speech(100.0)
    marks.mark_transport_first_frame()
    marks.transport_first_frame = 100.8
    store.record(marks)

    store.observe_client(
        {
            "turn_id": "trn_1",
            "eos_to_client_scheduled_ms": 830.0,
            "eos_to_client_audible_ms": 872.0,
            "audible_chars": 42,
        }
    )

    assert list(store.samples["client_scheduled"]) == [830.0]
    assert list(store.samples["client_audible"]) == [872.0]
    assert store.turns[-1]["audible_chars"] == 42
    assert store.turns[-1]["eos_to_client_audible_ms"] == 872.0


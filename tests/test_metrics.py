"""Turn marks: stage timing that is safe to read."""

from __future__ import annotations

import time

from sarang.metrics import TurnMarks, percentile


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


def test_mark_endpoint_backdates_the_silence_window():
    """Endpointing finishes before the marks object exists; the stage is written back.

    Without this the configured silence window - the single biggest term in
    the budget - never appears in /api/metrics, and every later claim about
    endpointing is inferred from config instead of measured.
    """
    marks = TurnMarks("trn_1")

    marks.mark_endpoint(812.0)

    assert round(marks.durations()["endpoint"], 1) == 812.0
    # The stage closes where the turn begins: origin is speech end.
    assert marks.stage_exit["endpoint"] == marks.origin


def test_mark_endpoint_ignores_a_missing_window():
    """No observed window (e.g. a typed turn) must not invent a stage."""
    marks = TurnMarks("trn_1")

    marks.mark_endpoint(0.0)

    assert marks.durations() == {}

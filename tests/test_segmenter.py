"""Speech segmentation: the first segment is the latency budget."""

from __future__ import annotations

import time

from sarang.segmenter import TextSegmenter, normalize_for_speech, split_script


def test_two_sentences_become_two_segments():
    segments = split_script(
        "Sanket built an agentic commerce gateway. It separates pricing from spend approval."
    )
    assert len(segments) == 2
    assert segments[0].endswith("gateway.")


def test_abbreviation_does_not_split():
    segments = split_script("He worked with Dr. Smith on the ZEISS pipeline for two years.")
    assert len(segments) == 1
    assert "Dr. Smith" in segments[0]


def test_decimal_does_not_split():
    segments = split_script("The score improved by 3.14 points across the whole cohort.")
    assert len(segments) == 1


def test_normalizer_removes_urls_and_markdown():
    text = normalize_for_speech("See **https://example.com** for the & details — now.")
    assert "http" not in text
    assert "**" not in text
    assert " and " in text
    assert ",—" not in text


def test_hard_character_limit_is_respected():
    segmenter = TextSegmenter(max_chars=80)
    text = " ".join(["engineering"] * 40)
    segments = segmenter.push(text)
    segments.extend(segmenter.flush())
    assert len(segments) > 1
    assert all(len(segment) <= 80 for segment in segments)


def test_due_forces_first_segment_after_hold():
    segmenter = TextSegmenter(first_hold_ms=100)
    segmenter.push("Sanket works on agentic")
    assert segmenter.due(now=segmenter._started_at + 0.05) == []
    released = segmenter.due(now=segmenter._started_at + 0.5)
    assert len(released) == 1


def test_flush_emits_remainder():
    segmenter = TextSegmenter()
    segmenter.push("A short trailing clause")
    assert segmenter.flush() == ["A short trailing clause"]
    assert not segmenter.has_pending


def test_streaming_push_is_incremental():
    segmenter = TextSegmenter()
    assert segmenter.push("Sanket is an ") == []
    released = segmenter.push("AI engineer. He builds ")
    assert released == ["Sanket is an AI engineer."]
    # Text that ends exactly at the buffer edge may still be growing, so it is
    # held rather than spoken half-written.
    assert segmenter.push("agentic systems.") == []
    released = segmenter.flush()
    assert released and released[-1].startswith("He builds")


def test_emitted_history_matches_output():
    segmenter = TextSegmenter()
    out = segmenter.push("One sentence here. Two sentence there.")
    out.extend(segmenter.flush())
    assert segmenter.emitted == out


def test_no_empty_or_punctuation_only_segments():
    segmenter = TextSegmenter()
    segments = segmenter.push("... What? ... Over here. ...")
    segments.extend(segmenter.flush())
    assert segments
    assert all(any(char.isalnum() for char in segment) for segment in segments)


def test_wall_clock_due_path_is_cheap():
    segmenter = TextSegmenter(first_hold_ms=0)
    segmenter.push("alpha beta gamma")
    started = time.perf_counter()
    assert segmenter.due()
    assert (time.perf_counter() - started) < 0.05

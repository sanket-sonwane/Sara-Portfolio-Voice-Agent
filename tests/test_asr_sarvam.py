"""Sarvam realtime protocol: connection parameters, chunking, turn framing."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

from sarang.asr.sarvam import SarvamASR


def test_url_carries_the_quality_parameters():
    asr = SarvamASR(
        api_key="key",
        language="en-IN",
        stream_type="balanced",
        keyterms=["ARAI", "TattvaQuest"],
        prompt="technical interview",
    )
    query = parse_qs(urlparse(asr.url).query)

    assert query["stream_type"] == ["balanced"]
    assert query["endpointing"] == ["manual"]
    assert query["encoding"] == ["linear16"]
    assert query["sample_rate"] == ["16000"]
    assert query["mode"] == ["transcribe"]
    assert json.loads(query["keyterms"][0]) == ["ARAI", "TattvaQuest"]
    assert query["prompt"] == ["technical interview"]


def test_keyterms_are_capped_at_the_provider_limit():
    asr = SarvamASR(api_key="key", keyterms=[f"term-{index}" for index in range(80)])
    assert len(asr.keyterms) == 50


async def test_chunks_are_aggregated_and_speech_end_closes_the_turn():
    asr = SarvamASR(api_key="key", chunk_ms=20)
    sent: list[dict] = []

    class Socket:
        async def send(self, payload: str) -> None:
            sent.append(json.loads(payload))

    asr._socket = Socket()
    asr._session_ready.set()

    await asr.begin_utterance()
    assert sent[0] == {"event": "speech_start"}

    # Exactly one 20 ms chunk goes out; the half chunk is buffered.
    await asr.push(b"\x01\x00" * 320)
    await asr.push(b"\x02\x00" * 160)
    assert [message["event"] for message in sent] == ["speech_start", "audio_input"]

    await asr.finalize()
    assert [message["event"] for message in sent] == [
        "speech_start",
        "audio_input",
        "audio_input",
        "speech_end",
    ]
    # ``speech_end`` already finalizes; a trailing ``flush`` could truncate it.
    assert all(message["event"] != "flush" for message in sent)


async def test_audio_is_not_sent_before_speech_start():
    asr = SarvamASR(api_key="key", chunk_ms=20)
    sent: list[dict] = []

    class Socket:
        async def send(self, payload: str) -> None:
            sent.append(json.loads(payload))

    asr._socket = Socket()
    asr._session_ready.set()

    await asr.push(b"\x00\x00" * 320)
    assert sent == []


def test_partial_and_final_carry_the_utterance_index():
    asr = SarvamASR(api_key="key")

    partial = asr._parse(
        {"event": "transcript.partial", "text": "what did", "utterance_idx": 2}
    )
    final = asr._parse(
        {"event": "transcript.final", "text": "what did you build", "utterance_idx": 2}
    )

    assert partial is not None and partial.final is False and partial.utterance == 2
    assert final is not None and final.final is True and final.utterance == 2


def test_errors_are_surfaced_instead_of_swallowed():
    asr = SarvamASR(api_key="key")
    handled = asr._handle_control(
        {
            "event": "error",
            "code": "invalid_config",
            "is_fatal": False,
            "message": "bad stream_type",
        }
    )
    assert handled is True
    assert "invalid_config" in asr.last_error
    assert asr._dead is False


def test_a_resolved_sample_rate_mismatch_is_reported(caplog):
    asr = SarvamASR(api_key="key", sample_rate=16_000)
    asr._server_config = {"sample_rate": 8000, "encoding": "linear16"}
    with caplog.at_level("ERROR"):
        asr._check_server_config()
    assert any("sample_rate" in record.getMessage() for record in caplog.records)


def test_a_hallucinated_non_latin_partial_is_dropped():
    # Observed live: an English session produced a Devanagari partial while the
    # final was correct English. The partial is display-only, so drop it.
    asr = SarvamASR(api_key="key", language="en-IN")
    partial = asr._parse(
        {
            "event": "transcript.partial",
            "text": "संकेत बिल्ड during the ARAI. internship.",
            "utterance_idx": 0,
        }
    )
    assert partial is None

    final = asr._parse(
        {"event": "transcript.final", "text": "what did you build", "utterance_idx": 0}
    )
    assert final is not None and final.final is True


def test_an_english_partial_is_kept():
    asr = SarvamASR(api_key="key", language="en-IN")
    partial = asr._parse(
        {
            "event": "transcript.partial",
            "text": "what did Sanket build during the ARAI",
            "utterance_idx": 0,
        }
    )
    assert partial is not None and partial.text.endswith("ARAI")

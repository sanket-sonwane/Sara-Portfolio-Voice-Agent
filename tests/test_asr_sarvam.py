"""Sarvam realtime protocol: connection parameters, chunking, turn framing."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlparse

import pytest
from sarang.asr.base import ASRError
from sarang.asr.sarvam import SarvamASR


class FakeSocket:
    """Stand-in for the provider socket; records every outbound frame."""

    def __init__(self, *, send_delay: float = 0.0) -> None:
        self.sent: list[dict] = []
        self.send_delay = send_delay
        self.closed = False

    async def send(self, payload: str) -> None:
        if self.send_delay:
            await asyncio.sleep(self.send_delay)
        self.sent.append(json.loads(payload))

    async def close(self) -> None:
        self.closed = True


async def _wire(asr: SarvamASR, socket: FakeSocket) -> None:
    """Attach a fake socket and start the real outbox sender."""
    asr._socket = socket
    asr._session_ready.set()
    asr._sender = asyncio.create_task(asr._sender_loop())


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
    socket = FakeSocket()
    await _wire(asr, socket)

    try:
        assert await asr.begin_utterance() is True
        await asr.settle()
        assert socket.sent[0] == {"event": "speech_start"}

        # Exactly one 20 ms chunk goes out; the half chunk is buffered.
        await asr.push(b"\x01\x00" * 320)
        await asr.push(b"\x02\x00" * 160)
        await asr.settle()
        assert [message["event"] for message in socket.sent] == [
            "speech_start",
            "audio_input",
        ]

        await asr.finalize()
        await asr.settle()
        assert [message["event"] for message in socket.sent] == [
            "speech_start",
            "audio_input",
            "audio_input",
            "speech_end",
        ]
        # ``speech_end`` already finalizes; a trailing ``flush`` could truncate it.
        assert all(message["event"] != "flush" for message in socket.sent)
    finally:
        await asr.close()


async def test_audio_is_not_sent_before_speech_start():
    asr = SarvamASR(api_key="key", chunk_ms=20)
    socket = FakeSocket()
    await _wire(asr, socket)

    try:
        await asr.push(b"\x00\x00" * 320)
        await asr.settle()
        assert socket.sent == []
    finally:
        await asr.close()


async def test_push_never_blocks_on_a_slow_socket():
    """The microphone path must return immediately, whatever the uplink does."""
    asr = SarvamASR(api_key="key", chunk_ms=20, max_queue_ms=4_000)
    socket = FakeSocket(send_delay=0.5)
    await _wire(asr, socket)

    try:
        assert await asr.begin_utterance() is True
        # 20 chunks = 400 ms of audio; the socket takes 0.5 s per send. A
        # blocking implementation would take ten seconds to accept this.
        started = asyncio.get_running_loop().time()
        for _ in range(20):
            await asr.push(b"\x00\x00" * 320)
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < 0.2
        # Audio is still queued, not dropped: the backlog is under budget.
        assert asr.dropped_ms == 0
    finally:
        await asr.close()


async def test_queue_overflow_drops_the_oldest_audio_and_counts_it():
    asr = SarvamASR(api_key="key", chunk_ms=20, max_queue_ms=100)
    socket = FakeSocket(send_delay=10.0)
    await _wire(asr, socket)

    try:
        assert await asr.begin_utterance() is True
        # Hold the sender inside the first send, then push far past the
        # 100 ms budget. The backlog is bounded, so the newest audio wins.
        await asyncio.sleep(0)
        for _ in range(40):
            await asr.push(b"\x00\x00" * 320)

        assert asr._outbox_bytes <= asr._max_outbox_bytes
        assert asr.dropped_ms > 0
    finally:
        await asr.close()


async def test_begin_utterance_is_refused_on_an_unarmed_session():
    asr = SarvamASR(api_key="key")
    # No ``session.begin``: the socket is open but cannot accept a turn, and
    # arming must fail fast instead of waiting for the provider.
    asr._socket = FakeSocket()

    assert await asr.begin_utterance() is False
    assert asr._speech_open is False


async def test_connect_fails_when_the_provider_never_arms(monkeypatch):
    asr = SarvamASR(api_key="key", session_begin_timeout=0.05)

    async def fake_connect(*_args, **_kwargs):
        return FakeSocket()

    monkeypatch.setattr("sarang.asr.sarvam.ws_connect", fake_connect)
    with pytest.raises(ASRError):
        await asr.connect()
    assert asr.last_error == "session.begin timeout"
    assert asr.alive is False


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

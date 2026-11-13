"""Sarvam realtime streaming ASR (``saaras:v3-realtime`` / ``saaras:v4``).

Sarvam is the in-region recogniser for the ``bom-india`` profile: audio never
leaves India. The realtime endpoint streams true partial transcripts and supports
manual turn control, which is what the gateway needs because endpointing is owned
by our own VAD: the browser's silence decision is the turn boundary, not the
provider's.

Protocol (``GET /speech-to-text-realtime/ws``): query parameters carry the model
and audio format; audio is base64 JSON frames; turns are delimited by
``speech_start`` / ``speech_end``; server events are ``session.begin``,
``transcript.partial``, ``transcript.final`` and ``error``.

One socket serves the whole conversation. The old design opened a socket per
utterance, which cost a handshake at exactly the moment the visitor started
talking and forced the caller to guess how much of the first word to buffer.
``speech_start`` / ``speech_end`` delimit turns on a live connection, so the
recogniser is always warm and the utterance loop can replay its pre-roll.

Audio is aggregated to ``chunk_ms`` before it goes on the wire. Sarvam's own
guidance is 100-500 ms chunks; 20 ms browser frames are too small to be a stable
unit for the recogniser.

If the socket cannot be opened the session degrades to ``unavailable`` and the
orchestrator tells the browser rather than failing the turn silently.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections import deque
from collections.abc import AsyncIterator
from urllib.parse import urlencode

from .base import ASRError, ASREvent, ASRSession

try:  # websockets >= 13
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover
    from websockets import connect as ws_connect  # type: ignore

log = logging.getLogger(__name__)


def _mostly_latin(text: str) -> bool:
    """True when the text is plausibly English rather than another script.

    A ratio test is not enough: the observed hallucination was
    ``"संकेत बिल्ड during the ARAI internship"``, which is mostly English by
    character count. Any run of two non-Latin letters means the recogniser
    switched script, and an English session should never show that.
    """
    run = 0
    for char in text:
        # Combining marks (vowel signs, virama) are not ``isalpha``, so test the
        # code point instead: any run of non-ASCII characters means a script
        # switch, even when the base letters are interleaved with diacritics.
        if ord(char) > 127 and not char.isspace():
            run += 1
            if run >= 2:
                return False
        else:
            run = 0
    return True


class SarvamASR(ASRSession):
    name = "sarvam"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.sarvam.ai",
        model: str = "saaras:v4",
        language: str = "en-IN",
        sample_rate: int = 16_000,
        stream_type: str = "balanced",
        mode: str = "transcribe",
        endpointing: str = "manual",
        chunk_ms: int = 100,
        keyterms: list[str] | None = None,
        prompt: str = "",
        open_timeout: float = 6.0,
        session_begin_timeout: float = 2.5,
        send_timeout: float = 2.0,
        max_queue_ms: int = 500,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.stream_type = stream_type
        self.mode = mode
        self.endpointing = endpointing
        self.chunk_ms = max(20, chunk_ms)
        self.keyterms = [term for term in (keyterms or []) if term][:50]
        self.prompt = prompt
        self.open_timeout = open_timeout
        self.session_begin_timeout = session_begin_timeout
        self.send_timeout = send_timeout
        self._socket = None
        self._queue: asyncio.Queue[ASREvent | None] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._closed = False
        self._dead = False
        self._speech_open = False
        self._session_ready = asyncio.Event()
        self._server_config: dict = {}
        self._buffer = bytearray()
        # Outbound audio must never block the microphone. Everything goes
        # through one ordered queue drained by a single sender task; when the
        # uplink cannot keep up, the *oldest* audio is dropped (real time has
        # already moved past it) and the loss is counted.
        self._outbox: deque[tuple[str, dict, int]] = deque()
        self._outbox_bytes = 0
        self._outbox_wake = asyncio.Event()
        self._outbox_idle = asyncio.Event()
        self._outbox_idle.set()
        self._sender: asyncio.Task | None = None
        self._max_outbox_bytes = max(
            3_200, self.sample_rate * 2 * max(100, max_queue_ms) // 1000
        )
        self.dropped_ms = 0.0
        # Keep outbound chunks in a stable range to balance partial latency
        # and recognition stability while preserving explicit small-chunk
        # configurations used by tests/probes.
        clamped_chunk_ms = min(max(self.chunk_ms, 20), 200)
        self._chunk_bytes = max(320, self.sample_rate * clamped_chunk_ms // 1000 * 2)
        self.last_error = ""

    # ------------------------------------------------------------------ config

    @property
    def url(self) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[-1]
        params: dict[str, str] = {
            "language_code": self.language,
            "model": self.model,
            "stream_type": self.stream_type,
            "mode": self.mode,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "endpointing": self.endpointing,
        }
        if self.keyterms:
            params["keyterms"] = json.dumps(self.keyterms)
        if self.prompt:
            params["prompt"] = self.prompt
        return f"{scheme}://{host}/speech-to-text-realtime/ws?{urlencode(params)}"

    @property
    def alive(self) -> bool:
        return self._socket is not None and not self._closed and not self._dead

    @property
    def ready(self) -> bool:
        """True only when the provider confirmed ``session.begin``.

        A socket that is open but unarmed is not usable: sending audio into it
        can silently drop the first word, and waiting for arming inside the
        microphone path is what used to stall ingestion for 2.5 s.
        """
        return self._session_ready.is_set() and self.alive

    @property
    def server_config(self) -> dict:
        return self._server_config

    # --------------------------------------------------------------- lifecycle

    async def connect(self) -> None:
        headers = {"api-subscription-key": self.api_key}
        try:
            self._socket = await ws_connect(
                self.url,
                additional_headers=headers,
                max_size=None,
                open_timeout=self.open_timeout,
            )
        except Exception as error:  # noqa: BLE001 - surface as a typed error
            raise ASRError(f"sarvam asr connect failed: {error}") from error
        self._reader = asyncio.create_task(self._read_loop())
        self._sender = asyncio.create_task(self._sender_loop())
        # ``session.begin`` echoes the resolved sample rate, encoding and turn
        # detection. Waiting for it is the difference between "we asked for 16
        # kHz" and "the server is decoding 16 kHz"; sending audio before the
        # session is armed can silently drop the first word. A connection that
        # never arms is a failure, not a warm socket.
        try:
            await asyncio.wait_for(
                self._session_ready.wait(), timeout=self.session_begin_timeout
            )
        except TimeoutError as error:
            self.last_error = "session.begin timeout"
            log.warning(
                "sarvam asr: no session.begin within %.1fs; closing the socket",
                self.session_begin_timeout,
            )
            await self.close()
            raise ASRError("sarvam asr did not confirm session.begin") from error

    async def _read_loop(self) -> None:
        assert self._socket is not None
        try:
            async for raw in self._socket:
                if self._closed:
                    return
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if self._handle_control(payload):
                    continue
                event = self._parse(payload)
                if event is not None:
                    await self._queue.put(event)
        except Exception as error:  # noqa: BLE001 - socket close is normal
            if not self._closed:
                log.warning("sarvam asr socket ended: %s", error)
        finally:
            self._dead = True
            await self._queue.put(None)

    def _handle_control(self, payload: dict) -> bool:
        event_type = str(payload.get("event", "")).lower()
        if event_type == "session.begin":
            config = payload.get("config")
            self._server_config = config if isinstance(config, dict) else {}
            self._session_ready.set()
            self._check_server_config()
            return True
        if event_type == "error":
            code = payload.get("code", "")
            message = payload.get("message", "")
            fatal = bool(payload.get("is_fatal"))
            self.last_error = f"{code}: {message}".strip(": ")
            log.warning("sarvam asr error code=%s fatal=%s message=%s", code, fatal, message)
            if fatal:
                self._dead = True
            return True
        if event_type in {"session.end", "config.updated", "pong"}:
            return True
        return False

    def _check_server_config(self) -> None:
        config = self._server_config
        if not config:
            return
        resolved_rate = str(config.get("sample_rate", "")).strip()
        if resolved_rate and resolved_rate != str(self.sample_rate):
            log.error(
                "sarvam asr: server resolved sample_rate=%s, we send %s; transcripts will be garbled",
                resolved_rate,
                self.sample_rate,
            )
        resolved_encoding = str(config.get("encoding", "")).strip().lower()
        if resolved_encoding and resolved_encoding != "linear16":
            log.error("sarvam asr: server resolved encoding=%s, expected linear16", resolved_encoding)
        log.info(
            "sarvam asr ready model=%s language=%s stream=%s mode=%s turn_detection=%s keyterms=%d",
            config.get("model", self.model),
            config.get("language_code", self.language),
            config.get("stream_type", self.stream_type),
            config.get("mode", self.mode),
            config.get("turn_detection", self.endpointing),
            len(self.keyterms),
        )

    def _parse(self, payload: dict) -> ASREvent | None:
        event_type = str(payload.get("event", "")).lower()
        if event_type not in {"transcript.partial", "transcript.final"}:
            return None
        text = payload.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            return None
        final = event_type == "transcript.final"
        # Streaming partials hallucinate: on an English session one partial came
        # back in Devanagari script while the final was correct. The final is
        # what the answer is built from, so a stray partial only ever costs
        # credibility on screen — drop it rather than show gibberish.
        if not final and self.language.startswith("en") and not _mostly_latin(text):
            log.debug("sarvam asr: dropped non-latin partial: %r", text[:60])
            return None
        utterance = payload.get("utterance_idx")
        return ASREvent(
            text=text.strip(),
            final=final,
            confidence=0.0,
            language=str(payload.get("language") or self.language),
            utterance=int(utterance) if isinstance(utterance, int) else 0,
        )

    # ------------------------------------------------------------------ input

    def _enqueue(self, payload: dict, *, audio_bytes: int = 0) -> None:
        if self._socket is None or self._closed or self._dead:
            return
        if audio_bytes:
            # Real-time audio is only useful while it is current. If the uplink
            # is behind, drop the oldest queued block rather than growing an
            # unbounded backlog that the recogniser will hear minutes late.
            while (
                self._outbox_bytes + audio_bytes > self._max_outbox_bytes
                and self._drop_oldest_audio()
            ):
                pass
            if self._outbox_bytes + audio_bytes > self._max_outbox_bytes:
                self.dropped_ms += audio_bytes / (self.sample_rate * 2 / 1000)
                return
        self._outbox.append(("audio" if audio_bytes else "control", payload, audio_bytes))
        self._outbox_bytes += audio_bytes
        self._outbox_idle.clear()
        self._outbox_wake.set()

    def _drop_oldest_audio(self) -> bool:
        for index, (kind, _payload, size) in enumerate(self._outbox):
            if kind != "audio":
                continue
            del self._outbox[index]
            self._outbox_bytes = max(0, self._outbox_bytes - size)
            self.dropped_ms += size / (self.sample_rate * 2 / 1000)
            return True
        return False

    async def _sender_loop(self) -> None:
        """Drain the outbox in order, off the microphone path."""
        while not self._closed:
            if not self._outbox:
                # Clear before the re-check: an enqueue that happens while we
                # are between the check and the wait must not be lost.
                self._outbox_wake.clear()
                if not self._outbox:
                    self._outbox_idle.set()
                    try:
                        await self._outbox_wake.wait()
                    except asyncio.CancelledError:
                        raise
                    continue
            self._outbox_idle.clear()
            _kind, payload, size = self._outbox.popleft()
            self._outbox_bytes = max(0, self._outbox_bytes - size)
            if self._socket is None or self._dead:
                continue
            try:
                await asyncio.wait_for(
                    self._socket.send(json.dumps(payload)), timeout=self.send_timeout
                )
            except TimeoutError:
                self.last_error = "asr send timeout"
                log.warning("sarvam asr send timed out after %.1fs", self.send_timeout)
                self._dead = True
                self._outbox.clear()
                self._outbox_bytes = 0
                break
            except Exception as error:  # noqa: BLE001 - the reader surfaces the close
                self.last_error = f"asr send failed: {error}"
                log.warning("sarvam asr send failed: %s", error)
                self._dead = True
                self._outbox.clear()
                self._outbox_bytes = 0
                break
        self._outbox_idle.set()

    async def settle(self) -> None:
        """Wait until the outbox has drained. Tests and shutdown only."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._outbox_idle.wait(), timeout=2.0)

    async def begin_utterance(self) -> bool:
        """Arm a turn on the live socket without waiting for the network."""
        if not self.ready:
            return False
        self._buffer.clear()
        self._speech_open = True
        if self.endpointing == "manual":
            self._enqueue({"event": "speech_start"})
        return True

    async def push(self, pcm: bytes) -> None:
        if not pcm or not self.ready or not self._speech_open:
            return
        self._buffer.extend(pcm)
        while len(self._buffer) >= self._chunk_bytes:
            block = bytes(self._buffer[: self._chunk_bytes])
            del self._buffer[: self._chunk_bytes]
            self._enqueue(
                {
                    "event": "audio_input",
                    "audio": base64.b64encode(block).decode("ascii"),
                },
                audio_bytes=len(block),
            )

    async def events(self) -> AsyncIterator[ASREvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            if event.text:
                yield event

    async def finalize(self) -> None:
        """Flush the tail and close the turn. ``speech_end`` already finalizes.

        Sending ``flush`` as well force-finalized whatever was buffered, which
        could truncate the transcript; the spec makes ``speech_end`` sufficient.
        This only enqueues: the sender preserves audio-before-speech_end order.
        """
        if not self.ready:
            return
        if self._buffer:
            block = bytes(self._buffer)
            self._buffer.clear()
            self._enqueue(
                {
                    "event": "audio_input",
                    "audio": base64.b64encode(block).decode("ascii"),
                },
                audio_bytes=len(block),
            )
        self._speech_open = False
        if self.endpointing == "manual":
            self._enqueue({"event": "speech_end"})

    async def close(self) -> None:
        self._closed = True
        sender, self._sender = self._sender, None
        if sender is not None:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
            self._reader = None
        the_socket, self._socket = self._socket, None
        if the_socket is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    the_socket.send(json.dumps({"event": "end"})), timeout=1.0
                )
            with contextlib.suppress(Exception):
                await the_socket.close()
        with contextlib.suppress(Exception):
            self._queue.put_nowait(None)

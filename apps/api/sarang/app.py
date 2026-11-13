"""FastAPI gateway.

Routes are intentionally few:

* ``GET  /healthz``             readiness and which stages are real
* ``GET  /api/config``          public runtime description
* ``POST /api/session``         short-lived token + ICE configuration
* ``POST /api/offer``           WebRTC offer/answer (optional transport)
* ``POST /api/turn/text``       typed question, same turn engine
* ``POST /api/interrupt``       explicit barge-in from the UI
* ``GET  /api/metrics``         rolling latency distribution
* ``WS   /ws/events``           one bus subscription per browser
* ``WS   /ws/audio``            PCM fallback transport when WebRTC is unavailable

The static widget is mounted at ``/`` when ``apps/web/dist`` exists, so the whole
product can be served from one origin. Provider keys never cross this boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import AGENT_NAME, __version__
from .audio.frames import StreamResampler, frame_bytes
from .auth import issue_token, verify_token
from .bus import Bus
from .config import REPO_ROOT, Settings, get_settings
from .kb.loader import load_kb
from .orchestrator import Orchestrator
from .ratelimit import SlidingWindowLimiter, client_key
from .schemas import (
    ALL_STATES,
    Envelope,
    EventType,
    HealthResponse,
    InterruptRequest,
    InterruptResponse,
    OfferRequest,
    OfferResponse,
    SessionRequest,
    SessionResponse,
    SessionState,
    TextTurnRequest,
    TextTurnResponse,
    new_turn_id,
)
from .session import Session, SessionRegistry
from .voice import UtteranceLoop
from .webrtc import WebRTCSession, webrtc_available

log = logging.getLogger(__name__)

WEB_DIST = REPO_ROOT / "apps" / "web" / "dist"


def _config_payload(settings: Settings) -> dict[str, Any]:
    return {
        **settings.describe(),
        "profile": settings.profile.name,
        "endpoint_silence_ms": settings.endpoint_silence_ms,
        "max_answer_tokens": settings.max_answer_tokens,
        "states": [state.value for state in ALL_STATES],
        "broadcast_sample_rate": 48_000,
        "input_sample_rate": 16_000,
        # PCM frames on /ws/audio are prefixed with (generation, sequence).
        # A client that does not see this must not guess: it reloads rather
        # than play a header as audio.
        "audio_format": "pcm16-v2",
        # Whether WebRTC is actually offered, not merely installed: the library
        # being present while ENABLE_WEBRTC is false used to report "webrtc:
        # true" and made the client attempt an ICE negotiation that cannot
        # succeed on Fly without TURN.
        "webrtc": settings.enable_webrtc and webrtc_available(),
        "speculation_ms": settings.speculative_ms,
        "filler": settings.filler_enabled,
        "version": __version__,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    bus = Bus()
    registry = SessionRegistry(resolved, bus)
    orchestrator = Orchestrator(resolved, registry)
    sweeper: asyncio.Task | None = None
    session_limiter = SlidingWindowLimiter(resolved.session_rate_limit)
    turn_limiter = SlidingWindowLimiter(resolved.turn_rate_limit)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal sweeper
        await orchestrator.startup()
        sweeper = asyncio.create_task(_sweep(registry))
        try:
            yield
        finally:
            if sweeper is not None:
                sweeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sweeper
            await orchestrator.shutdown()

    app = FastAPI(
        title=AGENT_NAME,
        version=__version__,
        description="Real-time voice agent for Sanket Sonwane's portfolio.",
        lifespan=lifespan,
    )
    app.state.orchestrator = orchestrator
    app.state.registry = registry
    app.state.bus = bus
    app.state.settings = resolved

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ helpers

    def authorised(session_id: str, token: str) -> bool:
        return verify_token(token, resolved.session_secret, session_id=session_id)

    def require_session(session_id: str, token: str) -> Session:
        if not authorised(session_id, token):
            raise HTTPException(status_code=401, detail="invalid or expired session token")
        session = registry.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session

    def limit_sessions(request: Request) -> None:
        if len(registry.sessions) >= resolved.max_sessions:
            raise HTTPException(
                status_code=429,
                detail="server is at capacity; try again shortly",
                headers={"Retry-After": "30"},
            )
        ip = client_key(dict(request.headers), request.client.host if request.client else None)
        if not session_limiter.allow(ip):
            raise HTTPException(
                status_code=429,
                detail="too many sessions from this client; try again shortly",
                headers={"Retry-After": str(session_limiter.retry_after(ip))},
            )

    def limit_turn(session: Session) -> None:
        if not turn_limiter.allow(session.id):
            raise HTTPException(
                status_code=429,
                detail="too many turns; slow down a moment",
                headers={"Retry-After": str(turn_limiter.retry_after(session.id))},
            )

    async def start_text_turn(session: Session, text: str) -> str:
        cleaned = " ".join(text.split())
        # A typed question supersedes everything the previous input started,
        # including a voice turn that is still waiting for its ASR final.
        session.input_seq += 1
        input_seq = session.input_seq
        if session.state.value in {
            "speaking",
            "thinking",
            "tool_working",
            "transcribing",
            "user_speaking",
        }:
            await orchestrator.interrupt(session, "new-text-turn")
        else:
            # Even in a quiet state, a pending finalizer must be abandoned:
            # otherwise it can deliver a transcript after this turn began.
            voice = session.voice
            abandon = getattr(voice, "abandon_pending", None)
            if abandon is not None:
                abandon()
        turn_id = new_turn_id()
        task = asyncio.create_task(
            orchestrator.run_turn(
                session,
                cleaned,
                source="text",
                turn_id=turn_id,
                input_seq=input_seq,
            )
        )
        session.turn_task = task
        return turn_id

    # -------------------------------------------------------------------- HTTP

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        kb = load_kb()
        stages = _config_payload(resolved)
        stages["kb_blocks"] = len(kb.blocks)
        stages["ready"] = orchestrator.ready
        stages["vad"] = f"{orchestrator.detector}/{orchestrator.detector_silence_ms}ms"
        return HealthResponse(
            ok=orchestrator.ready,
            version=__version__,
            profile=resolved.profile.name,
            stages=stages,
            uptime_s=round(time.time() - orchestrator.started_at, 1),
        )

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return _config_payload(resolved)

    @app.post("/api/session", response_model=SessionResponse)
    async def create_session(request: SessionRequest, http_request: Request) -> SessionResponse:
        limit_sessions(http_request)
        session = orchestrator.create_session(request.client)
        token = issue_token(session.id, resolved.session_secret, resolved.session_ttl_s)
        return SessionResponse(
            session_id=session.id,
            token=token,
            expires_in=resolved.session_ttl_s,
            ice_servers=resolved.ice_servers,
            force_relay=resolved.force_relay,
            config=_config_payload(resolved),
        )

    @app.post("/api/offer", response_model=OfferResponse)
    async def offer(request: OfferRequest) -> OfferResponse:
        session = require_session(request.session_id, request.token)
        if not resolved.enable_webrtc or not webrtc_available():
            raise HTTPException(
                status_code=503,
                detail="WebRTC is not available here; use /ws/audio",
            )
        if session.webrtc is not None:
            with contextlib.suppress(Exception):
                await session.webrtc.close()
        webrtc = WebRTCSession(orchestrator, session)
        try:
            answer = await webrtc.handle_offer(request.sdp, request.type)
        except Exception as error:  # noqa: BLE001 - a bad offer is a client error
            # A malformed SDP used to escape as an unhandled ValueError (and
            # leak the peer connection). It is a 400, and the half-built
            # session is closed before we answer.
            with contextlib.suppress(Exception):
                await webrtc.close()
            raise HTTPException(status_code=400, detail="invalid WebRTC offer") from error
        session.webrtc = webrtc
        session.ever_connected = True
        return OfferResponse(sdp=answer["sdp"], type=answer["type"])

    @app.post("/api/turn/text", response_model=TextTurnResponse)
    async def text_turn(request: TextTurnRequest) -> TextTurnResponse:
        session = require_session(request.session_id, request.token)
        if not request.text.strip():
            raise HTTPException(status_code=422, detail="empty text")
        limit_turn(session)
        turn_id = await start_text_turn(session, request.text)
        return TextTurnResponse(turn_id=turn_id, accepted=True)

    @app.post("/api/interrupt", response_model=InterruptResponse)
    async def interrupt(request: InterruptRequest) -> InterruptResponse:
        session = require_session(request.session_id, request.token)
        await orchestrator.interrupt(session, request.reason)
        return InterruptResponse(accepted=True, generation=session.generation)

    @app.get("/api/metrics")
    async def metrics() -> dict[str, Any]:
        spec: dict[str, int] = {
            "prepared": 0,
            "adopted": 0,
            "wasted": 0,
            "armed": 0,
            "arm_adopted": 0,
            "arm_wasted": 0,
        }
        for session in registry.sessions.values():
            candidate = getattr(session, "speculation", None)
            if candidate is None:
                continue
            stats = candidate.stats()
            for key in spec:
                spec[key] += int(stats.get(key, 0))
        return {
            "summary": orchestrator.metrics.summary(),
            "profile": resolved.profile.name,
            "providers": resolved.describe(),
            "webrtc": resolved.enable_webrtc and webrtc_available(),
            "filler_cached": orchestrator.filler.cached(),
            "speculation": spec,
            "tools": {
                "ready": orchestrator.tools.ready,
                "calls": orchestrator.tools.calls,
                "timeouts": orchestrator.tools.timeouts,
            },
        }

    # --------------------------------------------------------------- WebSocket

    @app.websocket("/ws/events")
    async def events_socket(websocket: WebSocket) -> None:
        session_id = websocket.query_params.get("session_id", "")
        token = websocket.query_params.get("token", "")
        if not authorised(session_id, token):
            await websocket.close(code=4401)
            return
        session = registry.get(session_id)
        if session is None:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        session.ever_connected = True
        key, queue = bus.subscribe(session.id)
        if session.state == SessionState.IDLE:
            session.set_state(SessionState.READY)

        ready = Envelope(
            type=EventType.SESSION_READY,
            session_id=session.id,
            payload={"config": _config_payload(resolved)},
        )
        with contextlib.suppress(Exception):
            await websocket.send_json(ready.to_wire())

        async def receiver() -> None:
            while True:
                try:
                    message = await websocket.receive_json()
                except (WebSocketDisconnect, RuntimeError, ValueError):
                    return
                kind = str(message.get("type", ""))
                if kind == "text":
                    text = str(message.get("text", ""))
                    if text.strip():
                        if turn_limiter.allow(session.id):
                            await start_text_turn(session, text)
                        else:
                            with contextlib.suppress(Exception):
                                await websocket.send_json(
                                    {
                                        "type": "notice",
                                        "payload": {
                                            "text": "That's a lot of questions at once. Give me a moment."
                                        },
                                    }
                                )
                elif kind == "interrupt":
                    await orchestrator.interrupt(session, str(message.get("reason", "client")))
                elif kind == "audible":
                    # The browser's own account of what it actually played. It
                    # is never inferred from the fact that a frame was sent.
                    orchestrator.note_audible(
                        session,
                        str(message.get("turn_id", "")),
                        chars=message.get("chars"),
                        text=message.get("text"),
                    )
                elif kind == "mute":
                    session.muted = bool(message.get("muted", False))
                    session.set_state(session.state)
                elif kind == "end":
                    await session.close()
                    with contextlib.suppress(Exception):
                        await websocket.close()
                    return
                elif kind == "ping":
                    with contextlib.suppress(Exception):
                        await websocket.send_json({"type": "pong"})

        recv_task = asyncio.create_task(receiver())
        try:
            while not recv_task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    with contextlib.suppress(Exception):
                        await websocket.send_json({"type": "ping"})
                    continue
                # Defence in depth: the bus is already session-scoped, but a
                # mis-stamped envelope must never cross a session boundary.
                if event.session_id != session.id:
                    log.warning(
                        "dropping cross-session event %s for %s (belongs to %s)",
                        event.type,
                        session.id,
                        event.session_id,
                    )
                    continue
                with contextlib.suppress(Exception):
                    await websocket.send_json(event.to_wire())
        except WebSocketDisconnect:
            pass
        finally:
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await recv_task
            bus.unsubscribe(key)

    @app.websocket("/ws/audio")
    async def audio_socket(websocket: WebSocket) -> None:
        session_id = websocket.query_params.get("session_id", "")
        token = websocket.query_params.get("token", "")
        if not authorised(session_id, token):
            await websocket.close(code=4401)
            return
        session = registry.get(session_id)
        if session is None:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        try:
            input_rate = int(websocket.query_params.get("sr", "16000"))
        except ValueError:
            input_rate = 16_000
        if input_rate <= 0:
            input_rate = 16_000
        # One stateful resampler per connection. Resampling each 20 ms message
        # independently discards the filter state at every boundary and puts a
        # click in the recogniser's input fifty times a second.
        resampler = (
            StreamResampler(input_rate, 16_000) if input_rate != 16_000 else None
        )
        if session.voice is not None:
            # Close before installing the replacement. The old order installed
            # the new loop and then closed the old one, so a reconnect could
            # leave `session.voice` pointing at a closed loop.
            previous, session.voice = session.voice, None
            with contextlib.suppress(Exception):
                await previous.close()
        loop = UtteranceLoop(orchestrator, session)
        session.voice = loop
        # Retained so the connection can cancel it. A detached warm-up that
        # finished after close used to install an ASR reader into a dead loop.
        warm_task = asyncio.create_task(loop.warm())
        if not session.welcome_sent:
            asyncio.create_task(orchestrator.greet(session))

        async def sender() -> None:
            while not session.closed:
                frame = await session.egress.next_frame()
                generation = int(frame.generation)
                if generation != session.generation:
                    # Selected before an interrupt. Writing it now would put the
                    # old generation on the wire after EPOCH.
                    continue
                await websocket.send_bytes(
                    struct.pack("<II", generation, int(frame.sequence)) + frame_bytes(frame)
                )
                session.egress.ack_transport(generation, int(frame.sequence))
                marks = session.marks.get(generation)
                if (
                    marks is not None
                    and marks.first_audio is not None
                    and marks.transport_first_frame is None
                ):
                    # The first frame the egress selected has now actually left
                    # the process. It is still not "audible" - that stamp belongs
                    # to the browser - but it is a distinct, honest event.
                    marks.mark_transport_first_frame()

        send_task = asyncio.create_task(sender())
        try:
            while not session.closed:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data:
                    if len(data) % 2:
                        data = data[:-1]
                    samples = np.frombuffer(data, dtype="<i2").astype(np.int16)
                    if resampler is not None and samples.size:
                        samples = resampler.process(samples)
                    if samples.size:
                        await loop.push(samples)
                    continue
                text = message.get("text")
                if text:
                    # Control messages are JSON. Parsing beats substring tests:
                    # a transcript or a stray word must never be read as a
                    # command, and the audible report needs typed fields.
                    try:
                        control = json.loads(text)
                    except (TypeError, ValueError):
                        control = None
                    kind = str(control.get("type", "")) if isinstance(control, dict) else ""
                    if kind == "interrupt":
                        await orchestrator.interrupt(session, "client")
                    elif kind == "end":
                        break
                    elif kind == "audible" and isinstance(control, dict):
                        orchestrator.note_audible(
                            session,
                            str(control.get("turn_id", "")),
                            chars=control.get("chars"),
                            text=control.get("text"),
                            scheduled_ms=control.get("scheduled_ms"),
                            audible_ms=control.get("audible_ms"),
                        )
        except WebSocketDisconnect:
            pass
        finally:
            warm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await warm_task
            send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await send_task
            await loop.close()
            # Only the connection that still owns the slot may clear it. An old
            # socket closing after a reconnect must not erase the new loop.
            if session.voice is loop:
                session.voice = None

    # ------------------------------------------------------------------ static

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
    else:

        @app.get("/")
        async def root() -> dict[str, Any]:
            return {
                "name": AGENT_NAME,
                "version": __version__,
                "profile": resolved.profile.name,
                "web_build": "missing; run `npm install && npm run build` in apps/web",
                "health": "/healthz",
                "docs": "/docs",
            }

    return app


async def _sweep(registry: SessionRegistry) -> None:
    while True:
        await asyncio.sleep(60)
        with contextlib.suppress(Exception):
            await registry.sweep()


app = create_app()

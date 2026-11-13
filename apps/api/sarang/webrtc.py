"""WebRTC transport.

Optional by design: ``aiortc`` and ``av`` are the only heavy dependencies in the
project and they are imported lazily, so the API, the tests and the WebSocket
audio fallback all run without them. ``/healthz`` says whether the WebRTC path is
available in the current environment.

The outbound track is a thin adapter over ``AudioEgress``. It never synthesises,
buffers or re-frames: it asks the shaper for one 20 ms frame and wraps it. That
keeps the fixed-framing and generation-gating guarantees in one place.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from fractions import Fraction
from typing import Any

import numpy as np

from .audio.frames import FRAME_SAMPLES, StreamResampler, downmix
from .audio.shaper import AudioEgress
from .orchestrator import Orchestrator
from .schemas import EventType, SessionState
from .session import Session
from .voice import UtteranceLoop

log = logging.getLogger(__name__)


def webrtc_available() -> bool:
    try:
        import aiortc  # noqa: F401
        import av  # noqa: F401
    except Exception:
        return False
    return True


def _build_egress_track_class():
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError

    class EgressTrack(AudioStreamTrack):
        kind = "audio"

        def __init__(self, egress: AudioEgress, session: Session) -> None:
            super().__init__()
            self.egress = egress
            self.session = session
            self._pts = 0

        async def recv(self):
            import av

            if self.session.closed:
                raise MediaStreamError
            generation = self.session.generation
            samples = await self.egress.next_frame()
            array = samples.reshape(1, FRAME_SAMPLES).astype(np.int16)
            frame = av.AudioFrame.from_ndarray(array, format="s16", layout="mono")
            frame.sample_rate = 48_000
            frame.pts = self._pts
            frame.time_base = Fraction(1, 48_000)
            self._pts += FRAME_SAMPLES
            marks = self.session.marks.get(generation)
            if (
                marks is not None
                and marks.first_audio is not None
                and marks.transport_first_frame is None
            ):
                marks.mark_transport_first_frame()
            return frame

    return EgressTrack


class WebRTCSession:
    def __init__(self, orchestrator: Orchestrator, session: Session) -> None:
        self.orchestrator = orchestrator
        self.session = session
        self.pc = None
        self.track = None
        self.voice: UtteranceLoop | None = None
        self._reader: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    async def handle_offer(self, sdp: str, kind: str = "offer") -> dict[str, str]:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

        EgressTrack = _build_egress_track_class()

        servers = []
        for entry in self.session.settings.ice_servers:
            kwargs: dict[str, Any] = {"urls": entry.get("urls", [])}
            if entry.get("username"):
                kwargs["username"] = entry["username"]
            if entry.get("credential"):
                kwargs["credential"] = entry["credential"]
            servers.append(RTCIceServer(**kwargs))
        config = RTCConfiguration(iceServers=servers)
        if self.session.settings.force_relay:
            config.iceTransportPolicy = "relay"

        pc = RTCPeerConnection(configuration=config)
        self.pc = pc
        self.track = EgressTrack(self.session.egress, self.session)
        pc.addTrack(self.track)

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                task = asyncio.create_task(self._consume(track))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            if state in {"failed", "disconnected"}:
                self.session.notice(
                    "The real-time connection dropped. Reconnecting…", "warning"
                )
                self.session.set_state(SessionState.RECONNECTING)
            elif state == "connected":
                self.session.ever_connected = True
                if self.session.state in {SessionState.CONNECTING, SessionState.RECONNECTING}:
                    self.session.set_state(SessionState.LISTENING)
                if not self.session.welcome_sent:
                    asyncio.create_task(self.orchestrator.greet(self.session))

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=kind))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await self._wait_for_ice(pc, timeout=2.0)
        local = pc.localDescription
        return {"sdp": local.sdp, "type": local.type}

    async def _wait_for_ice(self, pc, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while pc.iceGatheringState != "complete":
            if asyncio.get_running_loop().time() > deadline:
                break
            await asyncio.sleep(0.02)

    async def _consume(self, track) -> None:
        loop = UtteranceLoop(self.orchestrator, self.session)
        self.voice = loop
        asyncio.create_task(loop.warm())
        resampler = StreamResampler(48_000, 16_000)
        try:
            while not self.session.closed:
                frame = await track.recv()
                array = frame.to_ndarray()
                if array.size == 0:
                    continue
                if array.dtype != np.int16:
                    array = np.clip(array, -1.0, 1.0)
                    array = (array * 32767.0).astype(np.int16)
                if array.ndim == 2 and array.shape[0] > 1:
                    mono = downmix(array.T, array.shape[0])
                elif array.ndim == 2:
                    mono = array.reshape(-1)
                else:
                    mono = array
                pcm16k = resampler.process(mono.astype(np.int16))
                if pcm16k.size:
                    await loop.push(pcm16k)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - track end is normal
            log.debug("webrtc input stopped: %s", error)
        finally:
            await loop.close()
            self.voice = None

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.voice is not None:
            await self.voice.close()
            self.voice = None
        if self.track is not None:
            with contextlib.suppress(Exception):
                self.track.stop()
        if self.pc is not None:
            with contextlib.suppress(Exception):
                await self.pc.close()
            self.pc = None
        self.session.publish(EventType.NOTICE, {"message": "Call closed.", "level": "info"})

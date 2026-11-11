"""Conversation session state.

A session owns everything that outlives a single turn: history, generation
counter, the egress shaper, the trailing turn metrics and a reference to the
active task. It is intentionally plain data — the turn engine mutates it under a
single asyncio lock, and the transport layer only reads it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from .audio.shaper import AudioEgress
from .bus import Bus
from .config import Settings
from .kb.lookup import LookupResult
from .llm.base import Message
from .metrics import TurnMarks
from .schemas import EventType, SessionState, new_session_id


@dataclass
class Session:
    settings: Settings
    bus: Bus
    client: str = "web"
    id: str = field(default_factory=new_session_id)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    state: SessionState = SessionState.IDLE
    generation: int = 0
    turn_counter: int = 0
    active_turn: str = ""
    turn_task: asyncio.Task | None = None
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    history: list[Message] = field(default_factory=list)
    marks: dict[int, TurnMarks] = field(default_factory=dict)
    last_lookup: LookupResult | None = None
    speculation: Any = None

    muted: bool = False
    closed: bool = False
    ever_connected: bool = False
    interruptions: int = 0
    welcome_sent: bool = False

    egress: AudioEgress = field(default_factory=AudioEgress)
    webrtc: Any = None
    asr: Any = None
    voice: Any = None

    # ------------------------------------------------------------------ events

    def publish(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
        generation: int | None = None,
    ) -> None:
        from .schemas import Envelope

        self.bus.publish(
            Envelope(
                type=event_type,
                session_id=self.id,
                turn_id=self.active_turn if turn_id is None else turn_id,
                generation_id=self.generation if generation is None else generation,
                payload=payload or {},
            )
        )

    def set_state(self, state: SessionState, detail: str = "") -> None:
        self.state = state
        self.publish(
            EventType.STATE,
            {"state": state.value, "detail": detail, "muted": self.muted},
        )

    def notice(self, message: str, level: str = "info") -> None:
        self.publish(EventType.NOTICE, {"message": message, "level": level})

    def error(self, message: str, recoverable: bool = True) -> None:
        self.publish(EventType.ERROR, {"message": message, "recoverable": recoverable})
        self.set_state(SessionState.ERROR, message)

    # ---------------------------------------------------------------- lifecycle

    def touch(self) -> None:
        self.last_seen = time.time()

    def is_expired(self, ttl_s: int) -> bool:
        return (time.time() - self.last_seen) > ttl_s

    def remember(self, role: str, content: str, *, max_messages: int = 8) -> None:
        if not content.strip():
            return
        self.history.append(Message(role=role, content=content.strip()))
        # Keep a few exchanges, not the whole conversation: prompt tokens are
        # latency. Six messages is enough for pronoun and follow-up resolution.
        while len(self.history) > max_messages:
            self.history.pop(0)

    def begin_turn(self) -> tuple[str, int]:
        self.generation += 1
        self.turn_counter += 1
        self.active_turn = f"trn_{self.turn_counter:04d}"
        return self.active_turn, self.generation

    def cancel_active_turn(self) -> None:
        task = self.turn_task
        self.turn_task = None
        if task is not None and not task.done():
            task.cancel()

    async def close(self) -> None:
        self.closed = True
        self.cancel_active_turn()
        speculation = self.speculation
        if speculation is not None:
            # A closed session stops paying for speculative streams.
            with contextlib.suppress(Exception):
                speculation.reset()
        if self.asr is not None:
            with contextlib.suppress(Exception):
                await self.asr.close()
            self.asr = None
        if self.webrtc is not None:
            with contextlib.suppress(Exception):
                await self.webrtc.close()
            self.webrtc = None
        self.set_state(SessionState.ENDED)
        self.egress.interrupt(self.generation + 1)


class SessionRegistry:
    def __init__(self, settings: Settings, bus: Bus) -> None:
        self.settings = settings
        self.bus = bus
        self.sessions: dict[str, Session] = {}

    def create(self, client: str = "web") -> Session:
        session = Session(settings=self.settings, bus=self.bus, client=client)
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self.sessions.get(session_id)
        if session is None or session.closed:
            return None
        session.touch()
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            await session.close()

    async def sweep(self) -> int:
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if session.is_expired(self.settings.session_ttl_s + 300)
        ]
        for session_id in expired:
            await self.close(session_id)
        return len(expired)

    async def close_all(self) -> None:
        for session_id in list(self.sessions):
            await self.close(session_id)

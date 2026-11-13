"""Wire schemas: the typed contract between browser and gateway.

Internal stages use ``Envelope``; the WebSocket payload is a small JSON object
carrying the same identifiers. Every outbound message is stamped with the
generation it belongs to so the browser can discard anything that arrives after a
barge-in, even if the server is still draining a socket.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SessionState(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    READY = "ready"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    TOOL_WORKING = "tool_working"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    RECONNECTING = "reconnecting"
    MUTED = "muted"
    ERROR = "error"
    ENDED = "ended"


ALL_STATES: tuple[SessionState, ...] = tuple(SessionState)


class EventType(StrEnum):
    SESSION_READY = "session.ready"
    STATE = "state"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    ASSISTANT_DELTA = "assistant.delta"
    ASSISTANT_FINAL = "assistant.final"
    METRICS_TURN = "metrics.turn"
    METRICS_CLIENT = "metrics.client"
    EVIDENCE = "evidence"
    EPOCH = "epoch"
    ERROR = "error"
    NOTICE = "notice"


@dataclass(frozen=True)
class Envelope:
    """One event on the in-process bus."""

    type: EventType
    session_id: str
    turn_id: str = ""
    generation_id: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.perf_counter)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
            "payload": self.payload,
        }


def new_session_id() -> str:
    return f"ses_{uuid.uuid4().hex[:16]}"


def new_turn_id() -> str:
    return f"trn_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------- HTTP schemas


class SessionRequest(BaseModel):
    """Browser asks for a short-lived session token."""

    client: str = Field(default="web", max_length=40)


class SessionResponse(BaseModel):
    session_id: str
    token: str
    expires_in: int
    ice_servers: list[dict[str, Any]]
    force_relay: bool
    config: dict[str, Any]


class OfferRequest(BaseModel):
    session_id: str
    token: str = ""
    sdp: str
    type: str = "offer"


class OfferResponse(BaseModel):
    sdp: str
    type: str = "answer"


class TextTurnRequest(BaseModel):
    session_id: str
    token: str = ""
    text: str = Field(min_length=1, max_length=2000)


class TextTurnResponse(BaseModel):
    turn_id: str
    accepted: bool


class InterruptRequest(BaseModel):
    session_id: str
    token: str = ""
    reason: str = "barge-in"


class InterruptResponse(BaseModel):
    accepted: bool
    generation: int


class HealthResponse(BaseModel):
    ok: bool
    version: str
    profile: str
    stages: dict[str, Any]
    uptime_s: float

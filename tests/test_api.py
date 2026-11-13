"""Gateway contract: routes, auth, WebSocket events, metrics."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sarang.app import create_app
from sarang.webrtc import webrtc_available
from starlette.websockets import WebSocketDisconnect

from tests.conftest import local_settings


@pytest.fixture
def client():
    app = create_app(local_settings())
    with TestClient(app) as test_client:
        yield test_client


def open_session(client: TestClient) -> tuple[str, str]:
    response = client.post("/api/session", json={"client": "pytest"})
    assert response.status_code == 200
    body = response.json()
    return body["session_id"], body["token"]


def test_healthz_reports_ready_local_pipeline(client):
    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["profile"] == "local"
    assert body["stages"]["llm"] == "local"
    assert body["stages"]["tts"] == "synthetic"
    assert body["stages"]["kb_blocks"] > 5


def test_config_exposes_no_credentials(client):
    body = client.get("/api/config").json()
    assert body["llm"] == "local"
    assert "key" not in str(body).lower()
    # The gateway is deliberately more granular than the 12 UI states: it splits
    # user_speaking / transcribing / thinking / muted. Every UI state must still
    # be representable.
    assert {
        "idle",
        "connecting",
        "ready",
        "listening",
        "speaking",
        "tool_working",
        "interrupted",
        "reconnecting",
        "error",
        "ended",
    } <= set(body["states"])
    assert body["broadcast_sample_rate"] == 48_000


def test_session_returns_token_and_ice(client):
    session_id, token = open_session(client)
    assert session_id.startswith("ses_")
    assert token.count(".") == 2


def _drain_to_pong(ws) -> list[str]:
    """Read until the pong, returning the types seen on the way."""
    ws.send_json({"type": "ping"})
    seen: list[str] = []
    for _ in range(20):
        message = ws.receive_json()
        seen.append(str(message.get("type", "")))
        if message.get("type") == "pong":
            break
    return seen


def test_events_do_not_cross_session_boundaries(client):
    from sarang.schemas import Envelope, EventType

    id_a, token_a = open_session(client)
    id_b, token_b = open_session(client)
    with client.websocket_connect(
        f"/ws/events?session_id={id_a}&token={token_a}"
    ) as ws_a, client.websocket_connect(
        f"/ws/events?session_id={id_b}&token={token_b}"
    ) as ws_b:
        assert ws_a.receive_json()["type"] == "session.ready"
        assert ws_b.receive_json()["type"] == "session.ready"
        _drain_to_pong(ws_a)
        _drain_to_pong(ws_b)

        client.app.state.bus.publish(
            Envelope(
                type=EventType.NOTICE,
                session_id=id_a,
                payload={"message": "only for A"},
            )
        )

        notice = ws_a.receive_json()
        assert notice["type"] == "notice"
        assert notice["session_id"] == id_a

        # B asks for a pong. If the notice had leaked, it would be queued ahead
        # of the pong and appear in this list.
        seen = _drain_to_pong(ws_b)
        assert "notice" not in seen


def test_missing_token_is_rejected(client):
    session_id, _ = open_session(client)
    response = client.post(
        "/api/turn/text",
        json={"session_id": session_id, "token": "not-a-token", "text": "Hello"},
    )
    assert response.status_code == 401


def test_wrong_session_is_rejected(client):
    _, token = open_session(client)
    response = client.post(
        "/api/turn/text",
        json={"session_id": "ses_other", "token": token, "text": "Hello"},
    )
    assert response.status_code in {401, 404}


def test_unknown_session_is_rejected(client):
    session_id, token = open_session(client)
    response = client.post(
        "/api/turn/text",
        json={"session_id": session_id + "x", "token": token, "text": "Hello"},
    )
    assert response.status_code in {401, 404}


def test_text_turn_streams_events(client):
    session_id, token = open_session(client)
    with client.websocket_connect(
        f"/ws/events?session_id={session_id}&token={token}"
    ) as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "session.ready"

        response = client.post(
            "/api/turn/text",
            json={
                "session_id": session_id,
                "token": token,
                "text": "What is the Agentic Commerce Gateway?",
            },
        )
        assert response.status_code == 200
        assert response.json()["turn_id"].startswith("trn_")

        kinds: list[str] = []
        for _ in range(300):
            message = websocket.receive_json()
            kinds.append(message["type"])
            if message["type"] == "metrics.turn":
                assert message["payload"]["source"] == "text"
                assert message["payload"]["answer_first_audio_ms"] > 0
                break
        assert "assistant.final" in kinds
        assert "state" in kinds


def test_interrupt_route_bumps_generation(client):
    session_id, token = open_session(client)
    response = client.post(
        "/api/interrupt",
        json={"session_id": session_id, "token": token, "reason": "test"},
    )
    assert response.status_code == 200
    assert response.json() == {"accepted": True, "generation": 1}


def test_offer_route_matches_webrtc_availability(client):
    session_id, token = open_session(client)
    response = client.post(
        "/api/offer",
        json={"session_id": session_id, "token": token, "sdp": "v=0", "type": "offer"},
    )
    if webrtc_available():
        assert response.status_code in {200, 400, 500}
    else:
        assert response.status_code == 503
        assert "/ws/audio" in response.json()["detail"]


def test_metrics_route_summarises_turns(client):
    session_id, token = open_session(client)
    with client.websocket_connect(
        f"/ws/events?session_id={session_id}&token={token}"
    ) as websocket:
        websocket.receive_json()
        client.post(
            "/api/turn/text",
            json={"session_id": session_id, "token": token, "text": "Tell me about TattvaQuest"},
        )
        for _ in range(300):
            if websocket.receive_json()["type"] == "metrics.turn":
                break

    body = client.get("/api/metrics").json()
    assert body["profile"] == "local"
    assert body["summary"]["stages"]["answer_first_audio"]["n"] >= 1
    assert body["tools"]["ready"] is True


def test_websocket_rejects_bad_token(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/events?session_id=ses_nope&token=bad") as websocket:
            websocket.receive_json()

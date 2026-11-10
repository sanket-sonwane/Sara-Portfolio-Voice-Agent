"""Abuse limits: a public demo URL must not be an open door to the provider bill."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sarang.app import create_app
from sarang.ratelimit import SlidingWindowLimiter, client_key

from tests.conftest import local_settings


def test_window_allows_up_to_the_limit_then_blocks():
    limiter = SlidingWindowLimiter(limit=3, window_s=60)
    assert [limiter.allow("ip", now=0.0) for _ in range(3)] == [True, True, True]
    assert limiter.allow("ip", now=1.0) is False
    # A different client is unaffected.
    assert limiter.allow("other", now=1.0) is True


def test_window_reopens_after_the_window():
    limiter = SlidingWindowLimiter(limit=1, window_s=10)
    assert limiter.allow("ip", now=0.0)
    assert not limiter.allow("ip", now=9.9)
    assert limiter.allow("ip", now=10.1)


def test_retry_after_is_bounded_and_positive():
    limiter = SlidingWindowLimiter(limit=1, window_s=30)
    limiter.allow("ip", now=0.0)
    assert 0 < limiter.retry_after("ip", now=1.0) <= 31


def test_client_key_prefers_fly_header():
    assert client_key({"fly-client-ip": "1.2.3.4"}, "9.9.9.9") == "1.2.3.4"
    assert client_key({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, None) == "1.2.3.4"
    assert client_key({}, "9.9.9.9") == "9.9.9.9"


@pytest.fixture
def limited_client():
    app = create_app(local_settings(SESSION_RATE_LIMIT="2", TURN_RATE_LIMIT="2"))
    with TestClient(app) as test_client:
        yield test_client


def test_session_creation_is_rate_limited(limited_client):
    assert limited_client.post("/api/session", json={}).status_code == 200
    assert limited_client.post("/api/session", json={}).status_code == 200
    response = limited_client.post("/api/session", json={})
    assert response.status_code == 429
    assert response.headers["retry-after"]


def test_text_turns_are_rate_limited(limited_client):
    body = limited_client.post("/api/session", json={}).json()
    session_id, token = body["session_id"], body["token"]
    payload = {"session_id": session_id, "token": token, "text": "Hello"}
    assert limited_client.post("/api/turn/text", json=payload).status_code == 200
    assert limited_client.post("/api/turn/text", json=payload).status_code == 200
    assert limited_client.post("/api/turn/text", json=payload).status_code == 429

"""Session tokens: signed, short-lived, bound to one session."""

from __future__ import annotations

import time

from sarang import auth

SECRET = "unit-test-secret"


def test_roundtrip():
    token = auth.issue_token("ses_abc", SECRET, ttl_s=600)
    assert auth.verify_token(token, SECRET, session_id="ses_abc")
    assert auth.token_session(token) == "ses_abc"


def test_token_is_bound_to_its_session():
    token = auth.issue_token("ses_abc", SECRET, ttl_s=600)
    assert not auth.verify_token(token, SECRET, session_id="ses_other")


def test_tampered_signature_is_rejected():
    token = auth.issue_token("ses_abc", SECRET, ttl_s=600)
    # Pick a tail guaranteed to differ: a fixed "00" collided whenever the
    # signature already ended in 00 (1 run in 256), and the test then passed
    # on a perfectly valid token.
    tail = "00" if not token.endswith("00") else "11"
    assert not auth.verify_token(token[:-2] + tail, SECRET)
    assert not auth.verify_token(token, "wrong-secret")


def test_expired_token_is_rejected():
    expires = int(time.time()) - 10
    payload = f"ses_abc.{expires}"
    token = f"{payload}.{auth._sign(SECRET, payload)}"
    assert not auth.verify_token(token, SECRET)


def test_malformed_tokens_are_rejected():
    assert not auth.verify_token("", SECRET)
    assert not auth.verify_token("a.b", SECRET)
    assert not auth.verify_token("a.b.c.d", SECRET)
    assert auth.token_session("nope") is None

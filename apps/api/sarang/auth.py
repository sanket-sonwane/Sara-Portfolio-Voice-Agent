"""Short-lived session tokens.

The browser never sees a provider key. It gets a signed tuple of
``session_id.expiry.signature`` with a one-hour lifetime, which is enough for a
conversation and useless afterwards. The signature is an HMAC over the first two
parts, so a token cannot be edited into a different session.
"""

from __future__ import annotations

import hashlib
import hmac
import time


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def issue_token(session_id: str, secret: str, ttl_s: int = 3600) -> str:
    expires = int(time.time()) + max(60, ttl_s)
    payload = f"{session_id}.{expires}"
    return f"{payload}.{_sign(secret, payload)}"


def verify_token(token: str, secret: str, *, session_id: str | None = None) -> bool:
    parts = token.split(".")
    if len(parts) != 3:
        return False
    sid, expiry_raw, signature = parts
    if session_id is not None and sid != session_id:
        return False
    try:
        expires = int(expiry_raw)
    except ValueError:
        return False
    if expires < int(time.time()):
        return False
    expected = _sign(secret, f"{sid}.{expiry_raw}")
    return hmac.compare_digest(expected, signature)


def token_session(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    return parts[0]

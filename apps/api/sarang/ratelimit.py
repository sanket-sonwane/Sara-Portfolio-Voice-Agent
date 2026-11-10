"""Small in-process rate limiter.

Not a substitute for an edge limiter: it protects one process, which is exactly
the failure that matters here — a public demo URL where a single client can open
sessions until the provider bill or the event loop gives out. Fly's proxy adds
``Fly-Client-IP``; otherwise the peer address is used.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SlidingWindowLimiter:
    limit: int
    window_s: float = 60.0
    max_keys: int = 4_096
    _hits: dict[str, deque[float]] = field(default_factory=dict)

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        moment = time.monotonic() if now is None else now
        cutoff = moment - self.window_s
        bucket = self._hits.get(key)
        if bucket is None:
            if len(self._hits) >= self.max_keys:
                self._prune(cutoff)
            bucket = deque()
            self._hits[key] = bucket
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(moment)
        return True

    def retry_after(self, key: str, now: float | None = None) -> int:
        bucket = self._hits.get(key)
        if not bucket:
            return 0
        moment = time.monotonic() if now is None else now
        return max(1, int(bucket[0] + self.window_s - moment) + 1)

    def _prune(self, cutoff: float) -> None:
        for key in list(self._hits):
            bucket = self._hits[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                self._hits.pop(key, None)


def client_key(headers: dict[str, str], peer: str | None) -> str:
    """Best-effort client identity for the limiter."""
    forwarded = headers.get("fly-client-ip") or headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or peer or "unknown"

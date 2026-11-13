"""In-process pub/sub.

One process, one loop, one bus. Subscribers get their own bounded queue; a slow
consumer is dropped from the message rather than allowed to stall the pipeline.
That is the correct trade for a voice agent: the browser must never apply
backpressure to audio generation.

Routing is session-scoped. An envelope carries the session it belongs to, and a
subscriber registered for a session only ever receives that session's events.
The global queue (``session_id=None``) exists for tests and for anything that
genuinely needs the whole firehose; the browser socket never uses it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from .schemas import Envelope


class Bus:
    def __init__(self, queue_size: int = 256) -> None:
        # Session-scoped queues live apart from the global ones so publishing
        # to a session is a lookup, not a scan with a filter that can be wrong.
        self._by_session: dict[str, dict[str, asyncio.Queue[Envelope]]] = {}
        self._global: dict[str, asyncio.Queue[Envelope]] = {}
        self._queue_size = queue_size
        self._counter = 0

    def subscribe(self, session_id: str | None = None) -> tuple[str, asyncio.Queue[Envelope]]:
        self._counter += 1
        key = f"sub_{self._counter}"
        queue: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=self._queue_size)
        if session_id is None:
            self._global[key] = queue
        else:
            self._by_session.setdefault(session_id, {})[key] = queue
        return key, queue

    def unsubscribe(self, key: str) -> None:
        self._global.pop(key, None)
        for session_id, subscribers in list(self._by_session.items()):
            subscribers.pop(key, None)
            if not subscribers:
                self._by_session.pop(session_id, None)

    @staticmethod
    def _offer(queue: asyncio.Queue[Envelope], event: Envelope) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop the oldest event for this subscriber, then retry once.
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def publish(self, event: Envelope) -> None:
        for queue in list(self._by_session.get(event.session_id, {}).values()):
            self._offer(queue, event)
        for queue in list(self._global.values()):
            self._offer(queue, event)

    async def stream(self, key: str) -> AsyncIterator[Envelope]:
        queue = self._global.get(key)
        if queue is None:
            for subscribers in self._by_session.values():
                queue = subscribers.get(key)
                if queue is not None:
                    break
        if queue is None:
            return
        while True:
            event = await queue.get()
            yield event

    def close(self) -> None:
        self._global.clear()
        self._by_session.clear()

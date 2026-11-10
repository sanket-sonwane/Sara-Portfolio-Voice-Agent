"""In-process pub/sub.

One process, one loop, one bus. Subscribers get their own bounded queue; a slow
consumer is dropped from the message rather than allowed to stall the pipeline.
That is the correct trade for a voice agent: the browser must never apply
backpressure to audio generation.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from .schemas import Envelope


class Bus:
    def __init__(self, queue_size: int = 256) -> None:
        self._subscribers: dict[str, asyncio.Queue[Envelope]] = {}
        self._queue_size = queue_size
        self._counter = 0

    def subscribe(self) -> tuple[str, asyncio.Queue[Envelope]]:
        self._counter += 1
        key = f"sub_{self._counter}"
        queue: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers[key] = queue
        return key, queue

    def unsubscribe(self, key: str) -> None:
        self._subscribers.pop(key, None)

    def publish(self, event: Envelope) -> None:
        for queue in list(self._subscribers.values()):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest event for this subscriber, then retry once.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)

    async def stream(self, key: str) -> AsyncIterator[Envelope]:
        queue = self._subscribers.get(key)
        if queue is None:
            return
        while True:
            event = await queue.get()
            yield event

    def close(self) -> None:
        self._subscribers.clear()

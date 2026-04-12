"""In-process log ring buffer + async fan-out for the dashboard log stream.

Attaches a logging.Handler to the root and 'loki' loggers; keeps the last N
records in memory and pushes new records to any async subscribers
(WebSocket clients).
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Deque, Dict, List, Set


class RingBufferLogHandler(logging.Handler):
    def __init__(self, maxlen: int = 500) -> None:
        super().__init__()
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._subscribers: Set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _record_dict(self, record: logging.LogRecord) -> Dict[str, Any]:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        return {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "file": record.filename,
            "func": record.funcName,
            "line": record.lineno,
            "message": message,
        }

    def emit(self, record: logging.LogRecord) -> None:
        entry = self._record_dict(record)
        self._buffer.append(entry)
        if not self._subscribers or self._loop is None:
            return
        for queue in list(self._subscribers):
            try:
                self._loop.call_soon_threadsafe(queue.put_nowait, entry)
            except Exception:
                pass

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self._buffer)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)


_handler: RingBufferLogHandler | None = None


def install(level: int = logging.INFO, maxlen: int = 500) -> RingBufferLogHandler:
    """Attach the ring-buffer handler to root + 'loki' loggers. Idempotent."""
    global _handler
    if _handler is not None:
        return _handler
    _handler = RingBufferLogHandler(maxlen=maxlen)
    _handler.setLevel(level)
    _handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(_handler)
    logging.getLogger('loki').addHandler(_handler)
    try:
        _handler.bind_loop(asyncio.get_running_loop())
    except RuntimeError:
        pass
    return _handler


def get_handler() -> RingBufferLogHandler | None:
    return _handler

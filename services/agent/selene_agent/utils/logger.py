import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from . import config

import atexit
import json
import logging
import queue as _queue
import requests
from logging.config import dictConfig
from logging.handlers import QueueHandler, QueueListener

LOKI_URL = config.LOKI_URL
  # Update with your Loki URL
# If using Grafana Cloud, the URL format is: https://logs-prod-xxx.grafana.net/loki/api/v1/push
# LOKI_USERNAME = 'your-username'  # Only needed for Grafana Cloud
# LOKI_PASSWORD = 'your-api-key'   # Only needed for Grafana Cloud

# Bound the async Loki queue so a Loki outage can't grow memory without limit;
# records past the cap are dropped (telemetry backpressure) rather than block
# the thread that emitted them.
_LOKI_QUEUE_MAXSIZE = 10000


class LokiHandler(logging.Handler):
    """Ships log records to Loki over HTTP.

    Runs inside a QueueListener background thread (see get_logger), never on the
    asyncio event loop, so its blocking requests.post is safe here. A single
    requests.Session is reused across records, and failures are reported only on
    state transitions rather than once per record.
    """

    def __init__(self, url, username=None, password=None):
        super().__init__()
        self.url = url
        self.headers = {
            'Content-Type': 'application/json'
        }
        # Add authentication for Grafana Cloud
        if username and password:
            import base64
            credentials = base64.b64encode(f'{username}:{password}'.encode()).decode()
            self.headers['Authorization'] = f'Basic {credentials}'
        self._session = requests.Session()
        self._failing = False

    def emit(self, record):
        trace_id = getattr(record, 'trace_id', '')

        labels = {
            'job': 'ai',  # You can customize this
            'level': record.levelname,
            'logger': record.name,
            'filename': record.filename,
            'function': record.funcName
        }

        if trace_id:
            labels['trace_id'] = trace_id

        timestamp_ns = str(int(record.created * 1_000_000_000))

        log_line = record.getMessage()

        payload = {
            'streams': [
                {
                    'stream': labels,
                    'values': [
                        [timestamp_ns, log_line]
                    ]
                }
            ]
        }

        try:
            response = self._session.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                timeout=2
            )
            response.raise_for_status()
            if self._failing:
                self._failing = False
                print(f"Loki logging recovered ({self.url})", file=sys.stderr)
        except Exception as e:
            # Avoid infinite recursion (no logger here) and avoid per-record
            # stderr spam during an outage: report only the failing/recovered
            # transitions, then silently drop until state changes.
            if not self._failing:
                self._failing = True
                print(f"Loki logging unavailable, dropping records: {e}", file=sys.stderr)


class _NonBlockingQueueHandler(QueueHandler):
    """QueueHandler that drops records when the bounded queue is full instead of
    blocking the emitting (event-loop) thread or spilling a stderr traceback."""

    def emit(self, record):
        try:
            self.enqueue(self.prepare(record))
        except _queue.Full:
            pass
        except Exception:
            self.handleError(record)


def get_loki_handler():
    return LokiHandler

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'stream': 'ext://sys.stderr',
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console'],
            'level': config.LOG_LEVEL_OTHERS,
            'propagate': True
        },
        'loki': {  # your logger
            'handlers': ['console'],
            'level': config.LOG_LEVEL_APP,
            'propagate': False
        },
    }
}

_configured = False
_loki_listener = None


def _configure_once():
    """Configure logging exactly once for the process.

    The old code re-ran dictConfig on every get_logger() call, which reset the
    '' and 'loki' logger handler lists and silently detached handlers attached
    out-of-band -- notably the /ws/logs ring buffer (log_stream.install()),
    which broke the live log stream on the first post-startup import. Running
    once also means the Loki QueueListener background thread is started a single
    time rather than re-spawned per call.

    The Loki handler is wired behind a bounded queue + QueueListener so the
    blocking HTTP POST happens on a dedicated background thread; the async event
    loop only ever does a non-blocking put_nowait. When LOKI_URL is empty the
    Loki path is skipped entirely.
    """
    global _configured, _loki_listener
    if _configured:
        return
    dictConfig(LOGGING_CONFIG)
    if LOKI_URL:
        log_queue = _queue.Queue(maxsize=_LOKI_QUEUE_MAXSIZE)
        queue_handler = _NonBlockingQueueHandler(log_queue)
        queue_handler.setLevel(logging.DEBUG)
        logging.getLogger('loki').addHandler(queue_handler)
        _loki_listener = QueueListener(
            log_queue, LokiHandler(LOKI_URL), respect_handler_level=True
        )
        _loki_listener.start()
        atexit.register(_loki_listener.stop)
    _configured = True


def get_logger(name):
    _configure_once()
    return logging.getLogger(name)

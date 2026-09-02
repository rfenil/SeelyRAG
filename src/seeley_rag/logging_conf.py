"""Structured JSON logging.

Nothing in ``src/`` may call ``print()``. A crawl is a long unattended run whose
log is the only record of what happened; JSON lines let you answer "which
articles failed and why" with ``jq`` instead of a regex over prose.

Scripts under ``scripts/`` may print, but only human-facing summaries -- the
machine-readable record still goes through logging.

Call :func:`configure_logging` once, as early as possible in a script.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any

#: Attributes ``logging.LogRecord`` sets itself. Anything else on a record was
#: supplied via ``extra=`` and belongs in the JSON output.
_RESERVED: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON object on one line.

    Anything passed through ``extra=`` is merged into the object, so
    ``log.info("fetched", extra={"url": url, "status": 200})`` produces a row
    that can be filtered on ``url`` without parsing the message text.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialise one record.

        Args:
            record: The record to render.

        Returns:
            A JSON object as a single line, newline-free.
        """
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable format, for interactive runs."""

    def __init__(self) -> None:
        """Configure the terse ``HH:MM:SS LEVEL logger: message`` layout."""
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Install the root log handler. Idempotent -- safe to call more than once.

    Args:
        level: Log level name. Defaults to the configured ``logging.level``.
        fmt: ``"json"`` or ``"console"``. Defaults to the configured
            ``logging.format``.

    Logs go to stderr so a script's stdout stays clean for piped summaries.
    """
    from seeley_rag.settings import get_settings

    settings = get_settings()
    resolved_level = (level or settings.logging.level).upper()
    resolved_fmt = (fmt or settings.logging.format).lower()

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter() if resolved_fmt == "json" else ConsoleFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved_level)

    # httpx logs every request at INFO; at 1 rps that is noise on top of our own
    # structured fetch events.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Args:
        name: Usually ``__name__``.

    Returns:
        The named logger.
    """
    return logging.getLogger(name)

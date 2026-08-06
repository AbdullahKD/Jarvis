"""
Jarvis Logging Configuration
─────────────────────────────
A tiny, dependency-free logging layer for Jarvis.

Why this exists
---------------
Historically every component logged with bare `print()`. That is fine for
one-off event logs (server boot, OAuth connect, routing decisions) but it
breaks down for anything on a timer: the UI polls /live-tick every 20s and
/sidebar every 60s, and each poll printed a line like
    🏏 cricket: cricinfo matches=0 parsed=0
With nothing else logging on a timer, that single line flooded the terminal
and buried every meaningful log.

This module gives us:
  • `get_logger(name)`  — a namespaced logger for any module.
  • `setup_logging()`   — one call at process start to configure the console.

High-frequency / polling logs should use `logger.debug(...)`. They stay
hidden at the default INFO level but can be turned back on by setting
    JARVIS_LOG_LEVEL=DEBUG
Real events (connections, routing, errors) should use info/warning/error and
remain visible.

Existing `print()` calls keep working unchanged — this is additive, so the
migration can be gradual. The immediate win is moving the noisy poll logs
onto a logger so they can be silenced by default.
"""

from __future__ import annotations

import logging
import os
import sys

# Root logger name for everything under Jarvis. Module loggers are created as
# children (e.g. "jarvis.sports") so they all inherit one handler/level.
ROOT_LOGGER_NAME = "jarvis"

_CONFIGURED = False


class _EmojiFormatter(logging.Formatter):
    """Compact, readable console format that matches Jarvis's existing style.

    Keeps the friendly emoji/plain message intact and only prepends a short
    level tag for warnings and above, so INFO logs stay clean while errors
    still stand out.
    """

    _LEVEL_TAG = {
        logging.DEBUG:    "🐛 DEBUG",
        logging.INFO:     "",          # info logs print the message as-is
        logging.WARNING:  "⚠️  WARN",
        logging.ERROR:    "❌ ERROR",
        logging.CRITICAL: "🔥 CRIT",
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        tag = self._LEVEL_TAG.get(record.levelno, record.levelname)
        if record.levelno == logging.INFO:
            return msg
        if record.levelno == logging.DEBUG:
            # Include the short module name so debug noise is traceable.
            short = record.name.replace(ROOT_LOGGER_NAME + ".", "")
            return f"{tag} [{short}] {msg}"
        return f"{tag} {msg}"


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure the root Jarvis logger. Safe to call multiple times.

    Args:
        level: Override the log level. If None, read JARVIS_LOG_LEVEL
               (default "INFO"). Use "DEBUG" to surface polling logs.

    Returns:
        The configured root Jarvis logger.
    """
    global _CONFIGURED

    resolved = (level or os.getenv("JARVIS_LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, resolved, logging.INFO)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(log_level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_EmojiFormatter())
        logger.addHandler(handler)
        # Don't double-log through the (unconfigured) root logger.
        logger.propagate = False
        _CONFIGURED = True
    else:
        for h in logger.handlers:
            h.setLevel(log_level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, e.g. get_logger("sports").

    The first call lazily ensures setup_logging() has run so loggers work
    even if a module is imported before the server explicitly configures
    logging.
    """
    if not _CONFIGURED:
        setup_logging()
    short = name.split(".")[-1]
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{short}")

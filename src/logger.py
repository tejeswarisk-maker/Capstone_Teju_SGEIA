"""
logger.py — Centralised structured logging for SGEIA.

Features:
  - Coloured console output (INFO green, WARNING yellow, ERROR red)
  - Rotating file log at logs/sgeia.log (10 MB per file, 5 backups)
  - A dedicated pipeline_audit log at logs/pipeline_audit.log that records
    every user request and the full data-flow through each agent/pipeline.
  - A simple request_id is generated per query and tagged on every log line
    for end-to-end traceability.

Usage:
    from src.logger import get_logger, log_pipeline_event

    logger = get_logger(__name__)
    logger.info("Starting retrieval", extra={"request_id": req_id})

    log_pipeline_event(
        request_id=req_id,
        stage="Grid Retrieval Agent",
        event="hybrid_search_complete",
        details={"hits": 5, "bm25_top": "INC-012", "chroma_top": "INC-034"},
    )
"""

import json
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Any, Dict, Optional

import colorlog

from src.config import LOGS_DIR, settings

# Ensure the logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Log format strings ─────────────────────────────────────────────────────────
CONSOLE_FORMAT = (
    "%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s"
    " | req=%(request_id)s | %(message)s%(reset)s"
)
FILE_FORMAT = (
    "%(asctime)s [%(levelname)-8s] %(name)s"
    " | req=%(request_id)s | %(message)s"
)
AUDIT_FORMAT = "%(asctime)s | %(message)s"

# Colour mapping for console
LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


class RequestIdFilter(logging.Filter):
    """
    Injects a 'request_id' field into every log record.
    Falls back to 'N/A' when no request_id is attached.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return True


def _build_console_handler(level: int) -> logging.Handler:
    """Create a colour-formatted StreamHandler for the console."""
    handler = colorlog.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        colorlog.ColoredFormatter(CONSOLE_FORMAT, log_colors=LOG_COLORS)
    )
    handler.addFilter(RequestIdFilter())
    return handler


def _build_file_handler(path: Path, level: int) -> logging.Handler:
    """Create a rotating file handler (10 MB × 5 backups)."""
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(FILE_FORMAT))
    handler.addFilter(RequestIdFilter())
    return handler


def _build_audit_handler(path: Path) -> logging.Handler:
    """Create a rotating handler for the structured pipeline audit log."""
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=20 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(AUDIT_FORMAT))
    return handler


# ── Root logger setup ──────────────────────────────────────────────────────────
def _configure_root_logger() -> None:
    """Configure the root logger once at import time."""
    numeric_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g., in tests)
    root.setLevel(numeric_level)
    root.addHandler(_build_console_handler(numeric_level))
    root.addHandler(
        _build_file_handler(LOGS_DIR / "sgeia.log", numeric_level)
    )


_configure_root_logger()

# ── Audit logger ──────────────────────────────────────────────────────────────
_audit_logger = logging.getLogger("sgeia.audit")
_audit_logger.setLevel(logging.DEBUG)
_audit_logger.addHandler(_build_audit_handler(LOGS_DIR / "pipeline_audit.log"))
_audit_logger.propagate = False  # Do not bubble up to root (avoids duplication)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  Modules should call:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def log_pipeline_event(
    request_id: str,
    stage: str,
    event: str,
    details: Optional[Dict[str, Any]] = None,
    level: str = "INFO",
) -> None:
    """
    Write a structured JSON line to the pipeline audit log.

    Every user request flows through multiple pipeline stages.  This function
    records each step so we can trace the full data-flow and pinpoint where
    a failure occurred.

    Args:
        request_id: Unique ID generated at query ingestion time.
        stage:      Pipeline stage name, e.g. "Grid Retrieval Agent".
        event:      Short event key, e.g. "hybrid_search_complete".
        details:    Arbitrary dict with stage-specific metrics or values.
        level:      Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request_id": request_id,
        "stage": stage,
        "event": event,
        "details": details or {},
    }
    log_fn = getattr(_audit_logger, level.lower(), _audit_logger.info)
    log_fn(json.dumps(record, default=str))

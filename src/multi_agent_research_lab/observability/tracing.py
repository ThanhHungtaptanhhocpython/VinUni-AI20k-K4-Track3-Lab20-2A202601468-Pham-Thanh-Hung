import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for tracing execution spans with timing and metadata."""

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "running",
    }
    logger.debug("Started span %s with attributes %s", name, attributes)
    try:
        yield span
        span["status"] = "success"
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        logger.error("Span %s failed with error: %s", name, exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug("Completed span %s in %.3fs", name, span["duration_seconds"])

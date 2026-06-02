"""
Phoenix tracing helpers for Hermes Gateway.
Zero-dependency — uses the stdlib-only hermes_phoenix_tracer.
"""
from __future__ import annotations

import os
import time
import functools
from typing import Optional, Any

_PHOENIX_TRACER: Optional[Any] = None


def _get_tracer():
    """Lazy-init the Phoenix tracer. Called once at module load."""
    global _PHOENIX_TRACER
    if _PHOENIX_TRACER is not None:
        return _PHOENIX_TRACER
    try:
        from hermes_phoenix_tracer import PhoenixTracer
        _PHOENIX_TRACER = PhoenixTracer(
            service_name="hermes-gateway",
            endpoint=os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces"),
        )
    except Exception:
        _PHOENIX_TRACER = False  # sentinel: tried and failed
    return _PHOENIX_TRACER if _PHOENIX_TRACER is not False else None


def trace_span(name: str, **attrs):
    """Context manager for a traced span. No-op if tracer is unavailable."""
    tracer = _get_tracer()
    if tracer is None:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()
    return tracer.span(name, **attrs)


async def trace_async(name: str, coro, **attrs) -> Any:
    """Trace an async operation. Returns the coroutine's result."""
    tracer = _get_tracer()
    if tracer is None:
        return await coro
    span = tracer.start_span(name, **attrs)
    try:
        result = await coro
        span.set_status("ok")
        span.set_attribute("duration_ms", (time.time() * 1000) - (span.start_ns / 1_000_000))
        return result
    except Exception as exc:
        span.set_status("error", str(exc))
        raise
    finally:
        span._finish()

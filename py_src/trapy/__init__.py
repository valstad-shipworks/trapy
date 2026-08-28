"""trapy — structured logging from Python into Rust `tracing`.

Re-exports the level functions and span constructors from the compiled
`_trapy` extension and adds a pure-Python `instrument` decorator that
wraps a function call in a span, plus a `LocalTimer` helper for the
sequential-epoch timing pattern (one `epoch(name)` call closes the
previous span and opens the next, all rooted at the wall-clock instant
the timer was constructed / re-seeded).

Span lifecycle events: every `Span` emits a `cat="timing"` event on
enter and exit at the span's own level, including `elapsed_ms` on the
exit. Use `trapy.add_file_router(...)` and inspect `<dir>/timing.log`,
or use `trapy.add_env_filter(...)` to scope what's recorded. For
aggregate stats install the `tracing-timing` layer with
`trapy.add_timing_layer()` and call `trapy.dump_timing()` to flush
mean/p50/p90/p99/max-per-(span,event) to the same `timing.log`.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from trapy._trapy import (
    CAPSULE_NAME,
    CAPSULE_VERSION,
    Span,
    _capsule,
    add_env_filter,
    add_env_filter_from_env,
    add_file_router,
    add_fmt_layer,
    add_json_file,
    add_opentelemetry_layer,
    add_timing_layer,
    add_tracy_layer,
    debug,
    debug_span,
    dump_timing,
    error,
    error_span,
    flush_opentelemetry,
    flush_tracing,
    flush_tracing_sync,
    get_log_level,
    info,
    info_span,
    init_subscriber,
    init_tracing,
    set_log_level,
    shutdown_opentelemetry,
    trace,
    trace_span,
    warn,
    warn_span,
)

__all__ = [
    "CAPSULE_NAME",
    "CAPSULE_VERSION",
    "LocalTimer",
    "Span",
    "_capsule",
    "add_env_filter",
    "add_env_filter_from_env",
    "add_file_router",
    "add_fmt_layer",
    "add_json_file",
    "add_opentelemetry_layer",
    "add_timing_layer",
    "add_tracy_layer",
    "debug",
    "debug_span",
    "dump_timing",
    "end_span",
    "error",
    "error_span",
    "flush_opentelemetry",
    "flush_tracing",
    "flush_tracing_sync",
    "get_log_level",
    "info",
    "info_span",
    "init_subscriber",
    "init_tracing",
    "instrument",
    "set_log_level",
    "shutdown_opentelemetry",
    "start_span",
    "trace",
    "trace_span",
    "warn",
    "warn_span",
]


F = TypeVar("F", bound=Callable[..., Any])

_SPAN_CTORS = {
    "error": error_span,
    "warn": warn_span,
    "info": info_span,
    "debug": debug_span,
    "trace": trace_span,
}


def instrument(
    name_or_func: str | Callable[..., Any] | None = None,
    *,
    level: str = "info",
    **fields: Any,
) -> Callable[..., Any]:
    """Wrap a function so each call runs inside a span.

    Usage::

        @instrument
        def plan(): ...

        @instrument("plan_path", level="debug", category="planning")
        def plan(path): ...

    Bare `@instrument` uses the function's `__qualname__` as the span name
    and `info` level. The decorated form accepts a span name (positional),
    a `level` keyword, and additional kwargs that are recorded as static
    fields on every entry. Function arguments are not auto-captured — pass
    them through your span body via explicit `trapy.info(...)` calls if you
    need them.
    """

    # Bare `@instrument` (no parens) — `name_or_func` is the function.
    if callable(name_or_func):
        return _wrap(name_or_func, name_or_func.__qualname__, "info", {})

    span_name = name_or_func
    level_key = level.lower()
    if level_key not in _SPAN_CTORS:
        raise ValueError(
            f"instrument(level=...) must be one of {sorted(_SPAN_CTORS)}, got {level!r}"
        )
    captured_fields = dict(fields)

    def decorator(func: F) -> F:
        return _wrap(func, span_name or func.__qualname__, level_key, captured_fields)  # type: ignore[return-value]

    return decorator


def _wrap(
    func: Callable[..., Any],
    span_name: str,
    level: str,
    fields: dict[str, Any],
) -> Callable[..., Any]:
    span_ctor = _SPAN_CTORS[level]

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # `_trapy_caller_skip=1` makes trapy walk past this wrapper's
        # frame so the recorded source location is the caller of the
        # decorated function, not `trapy/__init__.py`.
        with span_ctor(span_name, _trapy_caller_skip=1, **fields):
            return func(*args, **kwargs)

    return wrapper


# Per-thread imperative span stack for `start_span` / `end_span` —
# the with-statement pair (`with trapy.trace_span(...)`) is preferred,
# but long sequential blocks where wrapping the whole body in a `with`
# would force a deep indent can use this instead.
_IMPERATIVE = threading.local()


def _imperative_stack() -> list[Span]:
    s = getattr(_IMPERATIVE, "spans", None)
    if s is None:
        s = []
        _IMPERATIVE.spans = s
    return s


def start_span(name: str, level: str = "trace") -> Span:
    """Push a new span onto this thread's imperative span stack and
    enter it. Pair with :func:`end_span`. Returns the underlying
    :class:`Span` so callers can read `name` / `is_enabled` if needed.
    """
    if level not in _SPAN_CTORS:
        raise ValueError(
            f"start_span(level=...) must be one of {sorted(_SPAN_CTORS)}, got {level!r}"
        )
    # Skip start_span's own frame so the span's source location is
    # the user's call site, not `trapy/__init__.py`.
    span = _SPAN_CTORS[level](name, _trapy_caller_skip=1)
    span.__enter__()
    _imperative_stack().append(span)
    return span


def end_span() -> None:
    """Exit and pop the most recently `start_span`'d span on this
    thread. Raises :class:`RuntimeError` if the stack is empty.
    """
    stack = _imperative_stack()
    if not stack:
        raise RuntimeError("end_span called without a matching start_span")
    span = stack.pop()
    span.__exit__(None, None, None)


class LocalTimer:
    """Sequential-epoch timer.

    One `epoch(name)` call closes the previous epoch (emitting a span
    enter+exit pair with the elapsed time since the last epoch / since
    construction) and opens the next. Useful when you want to label
    consecutive phases of a routine without wrapping each one in its
    own `with` block::

        t = trapy.LocalTimer()
        do_phase_a()
        t.epoch("phase_a")
        do_phase_b()
        t.epoch("phase_b")
        # close the last epoch explicitly if you care about it
        t.close_last("phase_c")

    Internally each epoch opens then immediately closes a `Span` so the
    standard span-enter / span-exit pipeline applies (timing event
    under `cat="spans"`, depth tracking, etc.). Level defaults to
    `"trace"` since these are usually fine-grained probes.
    """

    __slots__ = ("_ctor", "_last", "_level")

    def __init__(self, level: str = "trace") -> None:
        if level not in _SPAN_CTORS:
            raise ValueError(
                f"LocalTimer(level=...) must be one of {sorted(_SPAN_CTORS)}, got {level!r}"
            )
        self._level = level
        self._ctor = _SPAN_CTORS[level]
        self._last = time.perf_counter()

    def reseed_epoch(self) -> None:
        """Discard the in-flight epoch and restart the clock at *now*.
        The next `epoch(name)` measures from this point.
        """
        self._last = time.perf_counter()

    def epoch(self, name: str) -> None:
        """Close the last epoch as `name` and start the next one. The
        elapsed time from the previous epoch (or from construction) is
        recorded by the standard span-exit pipeline.
        """
        # Construct + enter + exit in a single tick — the span exit
        # event carries the elapsed_ms based on the Span's own internal
        # start instant, which we can't override. To preserve the
        # "elapsed since previous epoch" semantics, we manually emit a
        # short-lived span whose start is *now* and whose elapsed is
        # measured against `self._last`. Easiest portable way: use
        # info()/trace() events directly with the same shape.
        now = time.perf_counter()
        elapsed_ms = (now - self._last) * 1000.0
        # Reuse the standard timing-event shape so analyze tooling
        # reads one schema. depth=0 because LocalTimer epochs are flat
        # by design (nesting them inside other trapy spans is fine —
        # those contribute via the global SPAN_DEPTH counter on the
        # Rust side).
        emit = {
            "trace": trace,
            "debug": debug,
            "info": info,
            "warn": warn,
            "error": error,
        }[self._level]
        # Skip past `epoch`'s frame so the events attribute to whoever
        # called `epoch(...)`, not `trapy/__init__.py`.
        emit("", cat="timing", op="enter", name=name, depth=0, _trapy_caller_skip=1)
        emit(
            "",
            cat="timing",
            op="exit",
            name=name,
            depth=0,
            elapsed_ms=elapsed_ms,
            _trapy_caller_skip=1,
        )
        self._last = now

    def close_last(self, name: str) -> None:
        """Alias for `epoch(name)`; reads better at the tail of a
        sequence where there is no next phase to follow it.
        """
        self.epoch(name)

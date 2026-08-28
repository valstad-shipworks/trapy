"""Type stubs for the `_trapy` Rust extension.

The kwargs accept any value that `value.rs::py_to_value` knows how to walk:
primitives (`bool` / `int` / `float` / `str` / `None`), `list` / `tuple`,
`dict[str, ...]`, dataclass instances, anything implementing the `__array__`
protocol (numpy arrays/scalars, torch/jax tensors — converted via `.tolist()`),
and arbitrary nesting of the above. Anything else is stringified via `str(obj)`
rather than raising.

Python's type system can't fully express the recursive union, so we fall back
to `Any` on the value side. The intent is documented via `TrapyValue` below.
"""

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Protocol, TypeAlias, runtime_checkable

@runtime_checkable
class DataClassProto(Protocol):
    """Structural type for `@dataclass`-decorated instances."""

    __dataclass_fields__: dict[str, Any]

# Closest expressible approximation of the recursive value tree. The actual
# runtime accepts deeper nesting than the type checker can verify.
TrapyBase: TypeAlias = bool | int | float | str | None | DataClassProto
TrapyValue: TypeAlias = TrapyBase | Sequence[TrapyValue] | Mapping[str, TrapyValue]

def error(msg: str | None = "", /, **kwargs: TrapyValue) -> None:
    """Emit a `tracing` event at ERROR level with the calling Python frame's file/line."""

def warn(msg: str | None = "", /, **kwargs: TrapyValue) -> None:
    """Emit a `tracing` event at WARN level with the calling Python frame's file/line."""

def info(msg: str | None = "", /, **kwargs: TrapyValue) -> None:
    """Emit a `tracing` event at INFO level with the calling Python frame's file/line."""

def debug(msg: str | None = "", /, **kwargs: TrapyValue) -> None:
    """Emit a `tracing` event at DEBUG level with the calling Python frame's file/line."""

def trace(msg: str | None = "", /, **kwargs: TrapyValue) -> None:
    """Emit a `tracing` event at TRACE level with the calling Python frame's file/line."""

class Span:
    """Frozen handle to a `tracing` span. Use as a context manager.

    Constructed via `error_span` / `warn_span` / `info_span` / `debug_span` /
    `trace_span`. Attribute assignment is rejected — the only mutation the
    object permits is push/pop of an internal entry stack via
    `__enter__` / `__exit__`.
    """

    @property
    def name(self) -> str: ...
    @property
    def is_enabled(self) -> bool: ...
    def __enter__(self) -> Span: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool: ...

def error_span(name: str, /, **kwargs: TrapyValue) -> Span:
    """Create (but do not enter) an ERROR-level span as a child of the current span."""

def warn_span(name: str, /, **kwargs: TrapyValue) -> Span:
    """Create (but do not enter) a WARN-level span as a child of the current span."""

def info_span(name: str, /, **kwargs: TrapyValue) -> Span:
    """Create (but do not enter) an INFO-level span as a child of the current span."""

def debug_span(name: str, /, **kwargs: TrapyValue) -> Span:
    """Create (but do not enter) a DEBUG-level span as a child of the current span."""

def trace_span(name: str, /, **kwargs: TrapyValue) -> Span:
    """Create (but do not enter) a TRACE-level span as a child of the current span."""

# ── tracing backend control ─────────────────────────────────────────────

CAPSULE_NAME: str
CAPSULE_VERSION: int

def init_tracing(dir: str) -> None:
    """One-call convenience for the file-router-only setup.

    Equivalent to ``add_file_router(dir); init_subscriber()``. Use the
    underlying ``add_*`` helpers directly for multi-layer subscribers
    (e.g. file router + EnvFilter + fmt-to-stderr + JSON file).

    One-shot: a second call raises ``RuntimeError``.
    """

def add_file_router(
    dir: str,
    *,
    console_cats: Sequence[str] | None = None,
    console_only_cats: Sequence[str] | None = None,
    split_errors: bool = True,
    split_warnings: bool = False,
) -> None:
    """Spin up the per-category writer thread for ``dir`` and queue a
    ``tracing-catty`` ``CattyLayer``. Composable with the other
    ``add_*`` helpers; finalize with :func:`init_subscriber`. The
    writer thread is one-shot per process — only one
    ``add_file_router`` call succeeds.

    ``console_cats`` mirror to stdout in addition to their log file
    (default ``["progress"]``); ``console_only_cats`` go only to
    stdout with no file (default ``["console"]``). ``split_errors`` /
    ``split_warnings`` mirror those levels to ``errors.log`` /
    ``warnings.log`` on top of their normal ``cat`` routing.
    """

def add_fmt_layer(
    target_stream: str = "stderr",
    with_target: bool = True,
    with_ansi: bool = True,
    *,
    file: str | None = None,
    format: str = "full",
    with_level: bool = True,
    with_file: bool = False,
    with_line_number: bool = False,
    with_thread_ids: bool = False,
    with_thread_names: bool = False,
    span_events: Sequence[str] | None = None,
    json_flatten: bool = False,
    json_current_span: bool = True,
    json_span_list: bool = True,
) -> None:
    """``tracing_subscriber::fmt::Layer`` with the full builder surface.

    ``target_stream`` is ``"stderr"`` (default) or ``"stdout"``; pass
    ``file=`` to write to a path instead (truncated on open).
    ``format`` selects the event formatter: ``"full"`` (default),
    ``"compact"``, ``"pretty"``, or ``"json"``. ``span_events`` is a
    list drawn from ``["new", "enter", "exit", "close", "active",
    "full"]`` controlling which span lifecycle points are synthesized
    as events. The ``json_*`` knobs only apply when ``format="json"``.
    Set ``with_ansi=False`` when piping to a file or non-TTY.
    """

def add_env_filter(directive: str) -> None:
    """``EnvFilter`` layer with the standard ``RUST_LOG`` directive
    syntax, e.g. ``"info,planning=trace,orchestra::pybindings=debug"``.

    Filters apply to *all* downstream layers — events the filter
    rejects never reach the file router, fmt layer, or JSON file.
    """

def add_env_filter_from_env(var: str | None = None) -> None:
    """``EnvFilter`` layer read from an environment variable —
    ``RUST_LOG`` when ``var`` is omitted. Raises ``RuntimeError`` if
    the variable is unset or holds an invalid directive.
    """

def add_json_file(path: str) -> None:
    """JSON-per-line ``fmt`` layer pointed at ``path``. Truncates any
    pre-existing file. Useful for shipping to log aggregators."""

def add_timing_layer(max_value_ns: int = 1_000_000_000, precision: int = 2) -> None:
    """``tracing-timing`` layer recording inter-event timing in HDR
    histograms keyed by ``(span_name, event_message)``. Flush via
    :func:`dump_timing`.
    """

def dump_timing() -> int:
    """Snapshot the timing histograms and emit one ``cat="timing"``
    event per ``(span, event)`` pair carrying mean / p50 / p90 / p99 /
    max in nanoseconds plus the sample count. Returns the number of
    pairs reported.
    """

def add_tracy_layer() -> None:
    """``tracing-tracy`` layer streaming spans and events to a
    connected Tracy profiler. Starts the Tracy client immediately so
    the profiler can attach before the first span fires.
    """

def add_opentelemetry_layer(
    endpoint: str | None = None,
    service_name: str = "trapy",
) -> None:
    """``tracing-opentelemetry`` layer exporting spans over OTLP
    (HTTP + binary protobuf).

    ``endpoint`` defaults to the exporter's standard resolution
    (``OTEL_EXPORTER_OTLP_ENDPOINT``, else ``http://localhost:4318``);
    ``service_name`` becomes the OTel resource ``service.name``. Call
    :func:`shutdown_opentelemetry` before exit to drain the batch
    exporter, or tail spans are lost.
    """

def flush_opentelemetry() -> bool:
    """Block until the OTLP batch exporter has shipped every span
    queued so far. ``False`` if the opentelemetry layer isn't
    installed or the flush failed. Drops the GIL while waiting.
    """

def shutdown_opentelemetry() -> bool:
    """Drain and shut down the OTLP batch exporter. ``False`` if the
    layer isn't installed or shutdown failed (including a second
    call). Drops the GIL while waiting.
    """

def init_subscriber() -> None:
    """Drain queued layers and install the composed subscriber on this
    cdylib's tracing-core static. One-shot; further ``add_*`` calls
    fail.
    """

def flush_tracing() -> None:
    """Async flush — returns immediately; flush runs on the writer thread."""

def flush_tracing_sync() -> bool:
    """Block until every line queued before this call has hit disk.

    Returns ``True`` on writer ack, ``False`` if the backend is not
    initialised. Drops the GIL while waiting.
    """

def set_log_level(level: str) -> None:
    """Set the global filter (``off``/``error``/``warn``/``info``/``debug``/``trace``)."""

def get_log_level() -> str:
    """Read the active level filter as a lowercase string."""

def _capsule() -> Any:
    """Return the C-ABI handshake :class:`PyCapsule` (name ``trapy._capsule``).

    Other PyO3 cdylibs pull this at their own pymodule init time and
    forward their tracing events into trapy's writer thread.
    """

from collections.abc import Callable
from typing import Any, TypeVar, overload

from trapy._trapy import (
    CAPSULE_NAME,
    CAPSULE_VERSION,
    DataClassProto,
    Span,
    TrapyBase,
    TrapyValue,
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
    "DataClassProto",
    "LocalTimer",
    "Span",
    "TrapyBase",
    "TrapyValue",
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

_F = TypeVar("_F", bound=Callable[..., Any])

@overload
def instrument(name_or_func: _F) -> _F: ...
@overload
def instrument(
    name_or_func: str | None = ...,
    *,
    level: str = ...,
    **fields: TrapyValue,
) -> Callable[[_F], _F]: ...
def start_span(name: str, level: str = ...) -> Span: ...
def end_span() -> None: ...

class LocalTimer:
    def __init__(self, level: str = ...) -> None: ...
    def reseed_epoch(self) -> None: ...
    def epoch(self, name: str) -> None: ...
    def close_last(self, name: str) -> None: ...

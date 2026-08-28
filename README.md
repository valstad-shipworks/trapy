# trapy

Python logging facade that emits real Rust [`tracing`](https://docs.rs/tracing) events with structured kwargs.

In a process that mixes Python and Rust (PyO3 cdylibs), trapy gives both sides one tracing pipeline: `trapy.info("msg", cat="planning", x=1)` produces an event indistinguishable from `tracing::info!(cat = "planning", x = 1, "msg")` to any subscriber — including category routing, level filtering, and per-field structured access. Caller file/line/module are pulled from the Python frame and woven into the event metadata (`target = "py:<module>"`), so subscribers report Python locations and can route by Python module.

## Events

```python
import trapy

trapy.info("seam planned", cat="planning", part="bracket", qs=[0.1, 0.2])
trapy.warn("ik fallback", attempts=3)
trapy.error("collision", link="wrist", depth_mm=1.7)
```

Each kwarg becomes a real `tracing` field. Primitives (`bool`/`int`/`float`/`str`/`None`) flow through natively; lists, tuples, dicts, dataclasses, and anything implementing `__array__` (numpy arrays, tensors) flow through `valuable` as structured values. Anything else is stringified rather than raising.

## Spans

```python
with trapy.info_span("plan_part", part="bracket"):
    trapy.debug("inside the span")


@trapy.instrument("plan_path", level="debug", category="planning")
def plan(path): ...


timer = trapy.LocalTimer()
timer.epoch("load")  # closes the previous epoch span, opens the next
timer.epoch("solve")
timer.close_last("write")
```

Every span emits `cat="timing"` events on enter and exit (with `elapsed_ms`), so `<dir>/timing.log` doubles as a wall-clock record. For aggregate statistics install the timing layer (below) and call `trapy.dump_timing()`.

## Subscriber backend

The backend is compiled behind the `subscribers` cargo feature (the wheel enables it). Configure any number of layers, then install once:

```python
trapy.add_file_router("/tmp/trace")  # tracing-catty: events route to <dir>/<cat>.log
trapy.add_env_filter("info,planning=trace")
trapy.add_fmt_layer("stderr", format="compact", span_events=["close"])
trapy.add_json_file("/tmp/trace/events.json")
trapy.add_timing_layer()  # tracing-timing HDR histograms
trapy.add_tracy_layer()  # stream to a Tracy profiler
trapy.add_opentelemetry_layer(service_name="composer")  # OTLP span export
trapy.init_subscriber()  # one-shot; add_* calls fail afterwards
```

Or the one-call convenience for the file-router-only setup: `trapy.init_tracing(dir)`.

Available layers:

| call | layer |
|------|-------|
| `add_file_router(dir, ...)` | [`tracing-catty`](https://github.com/valstad-shipworks/tracing-catty) per-category file router. Defaults: `cat="progress"` mirrors to stdout, `cat="console"` is stdout-only, ERROR events fan out to `errors.log`; override via `console_cats=` / `console_only_cats=` / `split_errors=` / `split_warnings=` |
| `add_fmt_layer(...)` | `tracing_subscriber::fmt` with the full builder surface: `format=` full/compact/pretty/json, stdout/stderr or `file=`, `span_events=`, thread/file/line knobs |
| `add_env_filter(directive)` / `add_env_filter_from_env(var)` | `EnvFilter` with `RUST_LOG` syntax; rejected events never reach downstream layers |
| `add_json_file(path)` | JSON-per-line file, for Loki/Promtail-style shipping |
| `add_timing_layer(...)` + `dump_timing()` | `tracing-timing` inter-event HDR histograms keyed by `(span, event)` |
| `add_tracy_layer()` | `tracing-tracy`; starts the Tracy client eagerly so the profiler can attach before the first span |
| `add_opentelemetry_layer(endpoint, service_name)` | `tracing-opentelemetry` OTLP span export over HTTP + binary protobuf (no embedded tokio runtime) |

Flushing: `flush_tracing()` (async), `flush_tracing_sync()` (blocks until the writer acks — call before process exit), `flush_opentelemetry()` / `shutdown_opentelemetry()` for the OTLP batch exporter.

## Multi-cdylib forwarding

Every PyO3 cdylib statically links its own copy of `tracing-core`, so a subscriber installed in `trapy._trapy.so` is invisible to events emitted from another cdylib. trapy bridges this with a C-ABI capsule (`trapy._capsule`): the host cdylib links trapy as an rlib **without default features** and installs a forwarding subscriber at module init —

```toml
trapy = { version = "0.1.0", default-features = false }
```

```rust
#[pymodule]
fn _my_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = trapy::import_trapy_subscriber(m.py()); // best-effort, never raises
    ...
}
```

Events from the host's `tracing::*!` macros then flow through trapy's full layer chain. The handshake is fail-safe: if trapy isn't importable, the host still loads and its events are dropped. `set_log_level()` / `get_log_level()` gate the FFI boundary coarsely so foreign cdylibs can drop events before paying for formatting; in-process filtering is `EnvFilter`'s job.

## Building

```sh
pip install trapy            # or: maturin develop --release
```

Cargo features:

| feature | meaning |
|---------|---------|
| `subscribers` | the whole backend: layer builders, the file router, the capsule *owner* side. The wheel enables it via `pyproject.toml` |
| `extension-module` | `pyo3/extension-module`, for standalone wheel builds only |

Crates that consume trapy as an rlib (capsule forwarding) must enable **neither** — they stay consumer-only and skip compiling every subscriber crate.

The `valuable` field support rides `tracing`'s unstable API: builds need `--cfg tracing_unstable` (the workspace `.cargo/config.toml` sets it).

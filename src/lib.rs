//! `trapy` — a Python logging facade that emits real `tracing` events.
//!
//! Exposes `error`, `warn`, `info`, `debug`, `trace` Python functions with a
//! `tracing`-style signature: a positional `msg` plus arbitrary `**kwargs`.
//! Each kwarg is emitted as a real `tracing` field with the kwarg name —
//! primitives (bool/int/float/str) flow through `record_bool`/`record_i64`/
//! `record_f64`/`record_str`; lists, dicts, dataclasses, and numpy arrays
//! flow through `record_value` as `valuable::Value`. The result is that
//! `trapy.info("msg", cat="planning", x=1)` produces an event
//! indistinguishable from `tracing::info!(cat = "planning", x = 1, "msg")`
//! to any subscriber — including category-routing, level filtering, and
//! per-field structured access.
//!
//! Caller file/line/module are pulled from `sys._getframe(0)` and woven into
//! the synthesized `tracing::Metadata` (with `target = "py:<module>"`) so
//! subscribers can report Python locations and route by Python module.

pub mod callsite;
pub mod capsule;
#[cfg(feature = "subscribers")]
pub mod init;
pub mod level;
mod span;
mod value;

use pyo3::prelude::*;
#[cfg(feature = "subscribers")]
use pyo3::types::PyCapsule;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
use tracing_core::callsite::Callsite;
use tracing_core::field::Value;
use tracing_core::{Event, Level};
use valuable::Valuable;

use crate::callsite::MESSAGE_FIELD;
use crate::value::{PyValue, py_to_value};

#[cfg(feature = "subscribers")]
pub use capsule::TRAPY_CAPSULE;
pub use capsule::{CAPSULE_NAME, CAPSULE_VERSION, TrapyCapsule, import_trapy_subscriber};
pub use span::Span;

/// Magic kwarg recognised by every trapy emit/span entry point. Lets a
/// pure-Python wrapper (e.g. `LocalTimer.epoch`, `start_span`, the
/// `instrument` decorator's inner closure) ask trapy to skip its own
/// frame and report the *user's* call site instead. Default `0` keeps
/// the natural behaviour for direct callers.
pub const CALLER_SKIP_KWARG: &str = "_trapy_caller_skip";

/// Recover a Python frame's `__file__`, lineno, and `__name__`
/// (module). `skip=0` returns the topmost Python frame — i.e. the one
/// that invoked the surrounding `#[pyfunction]`. Higher values walk
/// further up the stack so wrappers can attribute events to *their*
/// caller. If the stack is exhausted (e.g. interpreter teardown) we
/// fall back to a sentinel location rather than raising.
pub fn caller_location(py: Python<'_>, skip: u32) -> PyResult<(String, u32, String)> {
    let sys = py.import("sys")?;
    let frame = match sys.call_method1("_getframe", (skip,)) {
        Ok(f) => f,
        Err(_) => return Ok(("<unknown>".to_string(), 0, "<unknown>".to_string())),
    };
    let code = frame.getattr("f_code")?;
    let filename: String = code.getattr("co_filename")?.extract()?;
    let lineno: u32 = frame.getattr("f_lineno")?.extract()?;
    let module: String = frame
        .getattr("f_globals")
        .and_then(|g| g.get_item("__name__"))
        .and_then(|n| n.extract())
        .unwrap_or_else(|_| "<unknown>".to_string());
    Ok((filename, lineno, module))
}

/// Parsed kwargs: ordered structured fields, an optional `message=`
/// override, and the `_trapy_caller_skip` depth.
type ParsedKwargs = (Vec<(String, PyValue)>, Option<String>, u32);

/// Pull all kwargs into an ordered `(name, PyValue)` list. Preserves dict
/// iteration order so the resulting fields appear in the same order the
/// caller wrote them — matching `tracing::event!` macro behaviour.
///
/// If the user passed `message=` as a kwarg it is split out and returned
/// separately, so it overrides the positional `msg` argument the way
/// `tracing::event!(message = "x")` shadows the format string.
fn parse_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<ParsedKwargs> {
    let Some(d) = kwargs else {
        return Ok((Vec::new(), None, 0));
    };
    let mut out: Vec<(String, PyValue)> = Vec::with_capacity(d.len());
    let mut message_override: Option<String> = None;
    let mut caller_skip: u32 = 0;
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        if key == MESSAGE_FIELD {
            // `message=` is reserved as the human-readable line. Use the
            // string form and keep it out of the structured fields so a
            // subscriber doesn't see it twice. `str(obj)` matches what
            // Rust's `tracing::event!(message = format!(...))` would do.
            message_override = Some(v.str()?.to_string());
            continue;
        }
        if key == CALLER_SKIP_KWARG {
            // Strip the magic skip kwarg so it never appears as a
            // structured field on the emitted event.
            caller_skip = v.extract::<u32>().unwrap_or(0);
            continue;
        }
        out.push((key, py_to_value(&v)?));
    }
    Ok((out, message_override, caller_skip))
}

fn emit(
    py: Python<'_>,
    level: Level,
    msg_positional: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    // Parse kwargs first so we can strip out the magic
    // `_trapy_caller_skip` (and `message=`) before the field set is
    // sized. Otherwise either would land in the structured fields and
    // a kwarg-name mismatch with the value slice would silently drop
    // fields downstream.
    let (kvs, msg_override, caller_skip) = parse_kwargs(kwargs)?;
    let msg: &str = msg_override.as_deref().unwrap_or(msg_positional);
    let (file, line, module) = caller_location(py, caller_skip)?;

    let kwarg_names: Vec<&str> = kvs.iter().map(|(k, _)| k.as_str()).collect();
    let cs = callsite::get_event(&file, line, level, &module, &kwarg_names);
    let meta = cs.metadata();

    tracing::dispatcher::get_default(|dispatcher| {
        if !dispatcher.enabled(meta) {
            return;
        }
        let fields = meta.fields();

        // For compound kwargs we need a `valuable::Value<'_>` that lives
        // for the duration of the dispatch call so its `&dyn Value`
        // reference stays valid. Build one slot per kwarg here; only
        // List/Map slots are populated.
        let v_views: Vec<Option<valuable::Value<'_>>> = kvs
            .iter()
            .map(|(_, v)| match v {
                PyValue::List(_) | PyValue::Map(_) => Some(v.as_value()),
                _ => None,
            })
            .collect();

        // Build the values slice in field order: [message, kvs...].
        // `value_set_all` requires `values.len() == fields.len()`.
        let mut values: Vec<Option<&dyn Value>> = Vec::with_capacity(kvs.len() + 1);
        values.push(Some(&msg as &dyn Value));
        for (i, (_, v)) in kvs.iter().enumerate() {
            let val: Option<&dyn Value> = match v {
                PyValue::Null => None,
                PyValue::Bool(b) => Some(b as &dyn Value),
                PyValue::Int(n) => Some(n as &dyn Value),
                PyValue::Float(f) => Some(f as &dyn Value),
                PyValue::Str(s) => Some(s as &dyn Value),
                PyValue::List(_) | PyValue::Map(_) => {
                    v_views[i].as_ref().map(|vv| vv as &dyn Value)
                }
            };
            values.push(val);
        }
        debug_assert_eq!(values.len(), fields.len());

        let value_set = fields.value_set_all(&values);
        dispatcher.event(&Event::new(meta, &value_set));
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (msg=None, **kwargs))]
fn error<'py>(
    py: Python<'py>,
    msg: Option<&str>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<()> {
    emit(py, Level::ERROR, msg.unwrap_or(""), kwargs)
}

#[pyfunction]
#[pyo3(signature = (msg=None, **kwargs))]
fn warn<'py>(
    py: Python<'py>,
    msg: Option<&str>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<()> {
    emit(py, Level::WARN, msg.unwrap_or(""), kwargs)
}

#[pyfunction]
#[pyo3(signature = (msg=None, **kwargs))]
fn info<'py>(
    py: Python<'py>,
    msg: Option<&str>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<()> {
    emit(py, Level::INFO, msg.unwrap_or(""), kwargs)
}

#[pyfunction]
#[pyo3(signature = (msg=None, **kwargs))]
fn debug<'py>(
    py: Python<'py>,
    msg: Option<&str>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<()> {
    emit(py, Level::DEBUG, msg.unwrap_or(""), kwargs)
}

#[pyfunction]
#[pyo3(signature = (msg=None, **kwargs))]
fn trace<'py>(
    py: Python<'py>,
    msg: Option<&str>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<()> {
    emit(py, Level::TRACE, msg.unwrap_or(""), kwargs)
}

/// One-call convenience: configure the per-category file router under
/// `dir` and install the subscriber. Equivalent to::
///
///     trapy.add_file_router(dir)
///     trapy.init_subscriber()
///
/// For multi-layer setups (fmt/env-filter/json/tracy/otel) call `add_*`
/// directly before `init_subscriber()`.
#[cfg(feature = "subscribers")]
#[pyfunction]
#[pyo3(signature = (dir))]
fn init_tracing(dir: String) -> PyResult<()> {
    init::init_tracing(&dir).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Spawn the per-category writer thread for `dir` and queue a
/// `tracing-catty` `CattyLayer` on the trapy subscriber. Composable
/// with the other `add_*` helpers; finalize with `init_subscriber()`.
///
/// `console_cats` mirror to stdout in addition to their log file
/// (default `["progress"]`); `console_only_cats` go only to stdout
/// with no file (default `["console"]`). `split_errors` /
/// `split_warnings` mirror those levels to `errors.log` /
/// `warnings.log` on top of their normal `cat` routing.
#[cfg(feature = "subscribers")]
#[pyfunction]
#[pyo3(signature = (dir, *, console_cats=None, console_only_cats=None, split_errors=true, split_warnings=false))]
fn add_file_router(
    dir: String,
    console_cats: Option<Vec<String>>,
    console_only_cats: Option<Vec<String>>,
    split_errors: bool,
    split_warnings: bool,
) -> PyResult<()> {
    let mut config = tracing_catty::CattyConfig::new()
        .split_errors(split_errors)
        .split_warnings(split_warnings);
    for cat in console_cats.unwrap_or_else(|| vec!["progress".to_string()]) {
        config = config.console_cat(cat);
    }
    for cat in console_only_cats.unwrap_or_else(|| vec!["console".to_string()]) {
        config = config.console_only_cat(cat);
    }
    init::add_file_router_with(&dir, config).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// `tracing_subscriber::fmt::Layer` with the full builder surface.
///
/// `target_stream` is `"stderr"` (default) or `"stdout"`; pass `file=`
/// to write to a path instead (truncated on open). `format` selects
/// the event formatter: `"full"` (default), `"compact"`, `"pretty"`,
/// or `"json"`. `span_events` is a list drawn from
/// `["new", "enter", "exit", "close", "active", "full"]` controlling
/// which span lifecycle points are synthesized as events. The `json_*`
/// knobs only apply when `format="json"`.
#[cfg(feature = "subscribers")]
#[pyfunction]
#[pyo3(signature = (
    target_stream="stderr", with_target=true, with_ansi=true, *, file=None, format="full",
    with_level=true, with_file=false, with_line_number=false, with_thread_ids=false,
    with_thread_names=false, span_events=None, json_flatten=false, json_current_span=true,
    json_span_list=true,
))]
#[allow(clippy::too_many_arguments)]
fn add_fmt_layer(
    target_stream: &str,
    with_target: bool,
    with_ansi: bool,
    file: Option<String>,
    format: &str,
    with_level: bool,
    with_file: bool,
    with_line_number: bool,
    with_thread_ids: bool,
    with_thread_names: bool,
    span_events: Option<Vec<String>>,
    json_flatten: bool,
    json_current_span: bool,
    json_span_list: bool,
) -> PyResult<()> {
    let target = match file {
        Some(path) => init::FmtTarget::File(path.into()),
        None => match target_stream.to_ascii_lowercase().as_str() {
            "stderr" => init::FmtTarget::Stderr,
            "stdout" => init::FmtTarget::Stdout,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "target_stream must be 'stderr' or 'stdout', got {:?}",
                    other
                )));
            }
        },
    };
    let format = match format.to_ascii_lowercase().as_str() {
        "full" => init::FmtFormat::Full,
        "compact" => init::FmtFormat::Compact,
        "pretty" => init::FmtFormat::Pretty,
        "json" => init::FmtFormat::Json,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "format must be one of 'full'/'compact'/'pretty'/'json', got {:?}",
                other
            )));
        }
    };
    let span_events = match span_events {
        Some(names) => {
            init::parse_span_events(&names).map_err(pyo3::exceptions::PyValueError::new_err)?
        }
        None => tracing_subscriber::fmt::format::FmtSpan::NONE,
    };
    init::add_fmt_layer(
        target,
        init::FmtOptions {
            format,
            with_target,
            with_ansi,
            with_level,
            with_file,
            with_line_number,
            with_thread_ids,
            with_thread_names,
            span_events,
            json_flatten,
            json_current_span,
            json_span_list,
        },
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// `EnvFilter` layer using the standard `RUST_LOG` directive syntax,
/// e.g. `"info,planning=trace,orchestra::bindings::c_python=debug"`.
#[cfg(feature = "subscribers")]
#[pyfunction]
#[pyo3(signature = (directive))]
fn add_env_filter(directive: &str) -> PyResult<()> {
    init::add_env_filter(directive).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// `EnvFilter` layer read from an environment variable — `RUST_LOG`
/// when `var` is omitted. Raises if the variable is unset or invalid.
#[cfg(feature = "subscribers")]
#[pyfunction]
#[pyo3(signature = (var=None))]
fn add_env_filter_from_env(var: Option<&str>) -> PyResult<()> {
    init::add_env_filter_from_env(var).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// JSON-per-line `fmt` layer pointed at `path`. Truncates any existing
/// file. Useful for shipping to Loki / Promtail / similar.
#[cfg(feature = "subscribers")]
#[pyfunction]
#[pyo3(signature = (path))]
fn add_json_file(path: String) -> PyResult<()> {
    init::add_json_file(&path).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// `tracing-timing` layer recording inter-event timing in HDR
/// histograms keyed by `(span_name, event_message)`. Histograms are
/// flushed to the writer thread on demand via :func:`dump_timing`.
#[cfg(feature = "subscribers")]
#[pyfunction]
#[pyo3(signature = (max_value_ns=1_000_000_000, precision=2))]
fn add_timing_layer(max_value_ns: u64, precision: u8) -> PyResult<()> {
    init::add_timing_layer(max_value_ns, precision)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Snapshot the timing-layer histograms and emit one `cat="timing"`
/// event per `(span, event)` pair carrying mean / p50 / p90 / p99 /
/// max in nanoseconds plus the sample count. Returns the number of
/// pairs reported. `0` if the timing layer wasn't installed or no
/// events have flowed through it yet.
#[cfg(feature = "subscribers")]
#[pyfunction]
fn dump_timing() -> usize {
    init::dump_timing()
}

/// `tracing-tracy` layer streaming spans and events to a connected
/// Tracy profiler. Starts the Tracy client immediately so the profiler
/// can attach before the first span fires.
#[cfg(feature = "subscribers")]
#[pyfunction]
fn add_tracy_layer() -> PyResult<()> {
    init::add_tracy_layer().map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// `tracing-opentelemetry` layer exporting spans over OTLP
/// (HTTP + binary protobuf). `endpoint` defaults to the standard
/// resolution (`OTEL_EXPORTER_OTLP_ENDPOINT`, else
/// `http://localhost:4318`); `service_name` becomes the OTel resource
/// `service.name`. Call :func:`shutdown_opentelemetry` before exit to
/// drain the batch exporter.
#[cfg(feature = "subscribers")]
#[pyfunction]
#[pyo3(signature = (endpoint=None, service_name="trapy"))]
fn add_opentelemetry_layer(endpoint: Option<&str>, service_name: &str) -> PyResult<()> {
    init::add_opentelemetry_layer(endpoint, service_name)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Block until the OTLP batch exporter has shipped every span queued
/// so far. Returns False if the opentelemetry layer isn't installed or
/// the flush failed. Drops the GIL while waiting.
#[cfg(feature = "subscribers")]
#[pyfunction]
fn flush_opentelemetry(py: Python<'_>) -> bool {
    py.detach(init::opentelemetry_flush)
}

/// Drain and shut down the OTLP batch exporter. Returns False if the
/// layer isn't installed or shutdown failed (including a second call).
/// Drops the GIL while waiting.
#[cfg(feature = "subscribers")]
#[pyfunction]
fn shutdown_opentelemetry(py: Python<'_>) -> bool {
    py.detach(init::opentelemetry_shutdown)
}

/// Drain queued layers and install the composed subscriber on this
/// cdylib's tracing-core static. One-shot; further `add_*` calls fail.
#[cfg(feature = "subscribers")]
#[pyfunction]
fn init_subscriber() -> PyResult<()> {
    init::init_subscriber().map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Ask the writer thread to flush every open log file. Returns
/// immediately; the flush itself runs on the worker.
#[cfg(feature = "subscribers")]
#[pyfunction]
fn flush_tracing() {
    tracing_catty::flush();
}

/// Block the current thread until every line queued *up to this call*
/// has been written. Drops Python's GIL while waiting so other threads
/// can keep emitting events.
#[cfg(feature = "subscribers")]
#[pyfunction]
fn flush_tracing_sync(py: Python<'_>) -> bool {
    py.detach(tracing_catty::flush_sync)
}

/// Set the global level filter. Accepts (case-insensitive) `off`,
/// `error`, `warn`, `info`, `debug`, `trace`.
#[pyfunction]
fn set_log_level(level: String) -> PyResult<()> {
    level::set_level(&level).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Read the current level filter back as a lowercase string.
#[pyfunction]
fn get_log_level() -> &'static str {
    level::current_level()
}

/// Returns the C-ABI handshake capsule. Other PyO3 cdylibs grab this at
/// their own pymodule init time and forward their tracing events into
/// trapy's writer thread. See `capsule.rs` for the layout.
#[cfg(feature = "subscribers")]
#[pyfunction]
#[pyo3(name = "_capsule")]
fn capsule_py(py: Python<'_>) -> PyResult<Bound<'_, PyCapsule>> {
    capsule::build_capsule(py)
}

/// Register the full trapy surface (events + spans) on `m`. Convenience
/// wrapper for hosts that want everything; the tracing-control surface
/// (`init_tracing`, `flush_tracing`, …) and the `_capsule` handshake
/// live ONLY on the canonical `trapy` module — host modules must call
/// `import trapy` to reach them — so they're not added here.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(error, m)?)?;
    m.add_function(wrap_pyfunction!(warn, m)?)?;
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_function(wrap_pyfunction!(debug, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    span::register(m)?;
    Ok(())
}

/// The subscriber-control surface: layer builders, init, flush, and
/// the capsule handshake. Only present when the cdylib is built with
/// the `subscribers` backend.
#[cfg(feature = "subscribers")]
fn register_subscriber_controls(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(add_file_router, m)?)?;
    m.add_function(wrap_pyfunction!(add_fmt_layer, m)?)?;
    m.add_function(wrap_pyfunction!(add_env_filter, m)?)?;
    m.add_function(wrap_pyfunction!(add_env_filter_from_env, m)?)?;
    m.add_function(wrap_pyfunction!(add_json_file, m)?)?;
    m.add_function(wrap_pyfunction!(add_timing_layer, m)?)?;
    m.add_function(wrap_pyfunction!(dump_timing, m)?)?;
    m.add_function(wrap_pyfunction!(add_tracy_layer, m)?)?;
    m.add_function(wrap_pyfunction!(add_opentelemetry_layer, m)?)?;
    m.add_function(wrap_pyfunction!(flush_opentelemetry, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown_opentelemetry, m)?)?;
    m.add_function(wrap_pyfunction!(init_subscriber, m)?)?;
    m.add_function(wrap_pyfunction!(flush_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(flush_tracing_sync, m)?)?;
    m.add_function(wrap_pyfunction!(capsule_py, m)?)?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "_trapy")]
fn _trapy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)?;
    // The control surface + capsule handshake are owned by the canonical
    // `trapy` module — they install backend state in this cdylib's own
    // `tracing-core` static, so re-exporting them from orchestra would
    // double-install and the second call would error out.
    #[cfg(feature = "subscribers")]
    register_subscriber_controls(m)?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(get_log_level, m)?)?;
    m.add("CAPSULE_NAME", capsule::CAPSULE_NAME)?;
    m.add("CAPSULE_VERSION", capsule::CAPSULE_VERSION)?;
    Ok(())
}

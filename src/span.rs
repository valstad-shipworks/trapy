//! `tracing` spans exposed to Python.
//!
//! `Span` mirrors `tracing::Span`: the constructor (`info_span`,
//! `warn_span`, …) creates a span as a child of whatever span is currently
//! active on this thread, recording the kwargs as structured fields. The
//! span can then be entered as a context manager:
//!
//! ```python
//! with trapy.info_span("planning", path_id=42) as s:
//!     trapy.info("started")  # event is parented under the span
//! ```
//!
//! Reentrancy: `__enter__` may be called multiple times (each push gets its
//! own guard); `__exit__` pops one guard at a time, matching `with`-block
//! semantics. The instance is frozen — attribute assignment from Python
//! raises, and the guard stack lives behind interior mutability so a single
//! `Span` can be safely re-entered.

use std::cell::{Cell, RefCell};
use std::time::Instant;

use pyo3::exceptions::{PyAttributeError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict};
use tracing_core::Level;
use tracing_core::field::Value;

use crate::CALLER_SKIP_KWARG;
use crate::callsite;
use crate::value::{PyValue, py_to_value};

thread_local! {
    /// Per-thread depth counter for trapy spans. Bumped on `__enter__`,
    /// dropped on `__exit__`. Carried in the `depth=` field of the
    /// "span enter" / "span exit" timing events so downstream tooling
    /// (e.g. `analyze_traces.py`) can rebuild the nesting tree from a
    /// flat `spans.log` without a parent map.
    static SPAN_DEPTH: Cell<u32> = const { Cell::new(0) };
}

#[pyclass(unsendable, name = "Span", module = "trapy")]
pub struct Span {
    inner: tracing::Span,
    name: String,
    /// Level the span was created at. Reused as the level of the
    /// `span enter` / `span exit` timing events so an `info_span` does
    /// info-level timing and a `trace_span` does trace-level timing.
    level: Level,
    /// Source location captured at span construction. The timing
    /// events use this so analyze tooling attributes them to user code,
    /// not to trapy's `span.rs`.
    file: String,
    line: u32,
    module: String,
    /// Stack of active entry guards plus their start instant. Each
    /// `__enter__` pushes one; each `__exit__` pops the most recent
    /// and emits the corresponding `span exit` event with elapsed_ms.
    /// `tracing::span::EnteredSpan` and `Instant` are `!Send`, so we
    /// use `RefCell` + `unsendable` — the GIL keeps access single-
    /// threaded anyway.
    guards: RefCell<Vec<(tracing::span::EnteredSpan, Instant)>>,
}

#[pymethods]
impl Span {
    /// Span name (the human-readable label set at construction).
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// `True` if this span is non-disabled (the active subscriber is
    /// recording it). Mirrors `tracing::Span::is_disabled` inverted.
    #[getter]
    fn is_enabled(&self) -> bool {
        !self.inner.is_disabled()
    }

    /// Frozen: reject all Python-level attribute assignment. Any internal
    /// mutation (the guard stack) goes through `RefCell`, never through
    /// Python attributes.
    fn __setattr__(&self, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyAttributeError::new_err(format!(
            "'Span' is frozen; cannot set attribute {:?}",
            name
        )))
    }

    fn __delattr__(&self, name: &str) -> PyResult<()> {
        Err(PyAttributeError::new_err(format!(
            "'Span' is frozen; cannot delete attribute {:?}",
            name
        )))
    }

    fn __repr__(&self) -> String {
        format!(
            "Span(name={:?}, entered={}, enabled={})",
            self.name,
            self.guards.borrow().len(),
            !self.inner.is_disabled(),
        )
    }

    /// Enter the span. Increments the per-thread span depth, pushes a
    /// guard + start instant, and emits a `span enter` timing event
    /// (`cat="spans"`, `op="enter"`, `name`, `depth`) at the span's
    /// own level. Returns `self` so `with span as s:` binds `s` to the
    /// same `Span` instance.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        let entered = slf.inner.clone().entered();
        let depth = SPAN_DEPTH.with(|d| {
            let cur = d.get();
            d.set(cur + 1);
            cur as i64
        });
        slf.guards.borrow_mut().push((entered, Instant::now()));
        emit_span_event(
            slf.level,
            &slf.file,
            slf.line,
            &slf.module,
            "enter",
            &slf.name,
            depth,
            None,
        );
        slf
    }

    /// Exit the most-recently-entered scope. Computes elapsed time
    /// since the matching `__enter__`, pops the guard, decrements the
    /// per-thread depth, and emits a `span exit` timing event with
    /// `elapsed_ms`. Returns `False` so any in-flight exception keeps
    /// propagating.
    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let popped = self.guards.borrow_mut().pop();
        let Some((guard, start)) = popped else {
            return Err(PyRuntimeError::new_err(
                "Span.__exit__ called without a matching __enter__",
            ));
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        // Drop the guard FIRST so the `span exit` event we emit lands
        // outside the span (matches the spanner-summary semantics where
        // exit events aren't parented under their own span).
        drop(guard);
        let depth = SPAN_DEPTH.with(|d| {
            let cur = d.get();
            let new = cur.saturating_sub(1);
            d.set(new);
            new as i64
        });
        emit_span_event(
            self.level,
            &self.file,
            self.line,
            &self.module,
            "exit",
            &self.name,
            depth,
            Some(elapsed_ms),
        );
        Ok(false)
    }
}

/// Emit a `span enter` / `span exit` event through the trapy event
/// machinery. Same dynamic-callsite trick as the core trapy.info path,
/// but with a fixed field signature so the cache hit rate is high.
#[allow(clippy::too_many_arguments)]
fn emit_span_event(
    level: Level,
    file: &str,
    line: u32,
    module: &str,
    op: &str,
    name: &str,
    depth: i64,
    elapsed_ms: Option<f64>,
) {
    let target = format!("py:{}", module);
    let cs = match elapsed_ms {
        Some(_) => callsite::get_event_full(
            file,
            line,
            level,
            &target,
            module,
            &["cat", "op", "name", "depth", "elapsed_ms"],
        ),
        None => callsite::get_event_full(
            file,
            line,
            level,
            &target,
            module,
            &["cat", "op", "name", "depth"],
        ),
    };
    let meta = cs.meta_static();
    tracing::dispatcher::get_default(|dispatcher| {
        if !dispatcher.enabled(meta) {
            return;
        }
        let fields = meta.fields();
        let cat = "timing";
        // Field order matches the callsite: `[message, cat, op, name, depth, (elapsed_ms)?]`.
        let mut values: Vec<Option<&dyn Value>> = Vec::with_capacity(6);
        values.push(None); // message
        values.push(Some(&cat as &dyn Value));
        values.push(Some(&op as &dyn Value));
        values.push(Some(&name as &dyn Value));
        values.push(Some(&depth as &dyn Value));
        if let Some(ms) = elapsed_ms.as_ref() {
            values.push(Some(ms as &dyn Value));
        }
        debug_assert_eq!(values.len(), fields.len());
        let value_set = fields.value_set_all(&values);
        dispatcher.event(&tracing::Event::new(meta, &value_set));
    });
}

/// Build a tracing span at the given level using a synthesized callsite for
/// the Python caller's `(file, line, level, name, kwarg-names)` signature.
fn make_span(
    py: Python<'_>,
    level: Level,
    name: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Span> {
    let (kvs, caller_skip) = parse_span_kwargs(kwargs)?;
    let (file, line, module) = caller_location(py, caller_skip)?;
    let kwarg_names: Vec<&str> = kvs.iter().map(|(k, _)| k.as_str()).collect();

    let cs = callsite::get_span(&file, line, level, &module, name, &kwarg_names);
    let meta = cs.meta_static();
    let fields = meta.fields();

    // Same trick as the event side: keep `valuable::Value` slots alive for
    // the duration of the value-set construction so the `&dyn Value`
    // borrows remain valid.
    let v_views: Vec<Option<valuable::Value<'_>>> = kvs
        .iter()
        .map(|(_, v)| match v {
            PyValue::List(_) | PyValue::Map(_) => {
                Some(<PyValue as valuable::Valuable>::as_value(v))
            }
            _ => None,
        })
        .collect();

    let mut values: Vec<Option<&dyn Value>> = Vec::with_capacity(kvs.len());
    for (i, (_, v)) in kvs.iter().enumerate() {
        let val: Option<&dyn Value> = match v {
            PyValue::Null => None,
            PyValue::Bool(b) => Some(b as &dyn Value),
            PyValue::Int(n) => Some(n as &dyn Value),
            PyValue::Float(f) => Some(f as &dyn Value),
            PyValue::Str(s) => Some(s as &dyn Value),
            PyValue::List(_) | PyValue::Map(_) => v_views[i].as_ref().map(|vv| vv as &dyn Value),
        };
        values.push(val);
    }
    debug_assert_eq!(values.len(), fields.len());

    let value_set = fields.value_set_all(&values);
    let inner = tracing::Span::new(meta, &value_set);

    Ok(Span {
        inner,
        name: name.to_string(),
        level,
        file,
        line,
        module,
        guards: RefCell::new(Vec::new()),
    })
}

/// Re-uses the shared `crate::caller_location` helper so spans and
/// events read frames the same way.
use crate::caller_location;

/// Span kwargs do not carry a synthetic `message` field — the human label
/// is the span name itself. Strips the magic `_trapy_caller_skip` kwarg
/// so it never leaks into the structured field set, and returns it
/// alongside the kvs so the caller can pass it to `caller_location`.
fn parse_span_kwargs(
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(Vec<(String, PyValue)>, u32)> {
    use pyo3::types::PyDictMethods;
    let Some(d) = kwargs else {
        return Ok((Vec::new(), 0));
    };
    let mut out: Vec<(String, PyValue)> = Vec::with_capacity(d.len());
    let mut caller_skip: u32 = 0;
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        if key == CALLER_SKIP_KWARG {
            caller_skip = v.extract::<u32>().unwrap_or(0);
            continue;
        }
        out.push((key, py_to_value(&v)?));
    }
    Ok((out, caller_skip))
}

#[pyfunction]
#[pyo3(signature = (name, **kwargs))]
fn error_span<'py>(
    py: Python<'py>,
    name: &str,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Span> {
    make_span(py, Level::ERROR, name, kwargs)
}

#[pyfunction]
#[pyo3(signature = (name, **kwargs))]
fn warn_span<'py>(
    py: Python<'py>,
    name: &str,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Span> {
    make_span(py, Level::WARN, name, kwargs)
}

#[pyfunction]
#[pyo3(signature = (name, **kwargs))]
fn info_span<'py>(
    py: Python<'py>,
    name: &str,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Span> {
    make_span(py, Level::INFO, name, kwargs)
}

#[pyfunction]
#[pyo3(signature = (name, **kwargs))]
fn debug_span<'py>(
    py: Python<'py>,
    name: &str,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Span> {
    make_span(py, Level::DEBUG, name, kwargs)
}

#[pyfunction]
#[pyo3(signature = (name, **kwargs))]
fn trace_span<'py>(
    py: Python<'py>,
    name: &str,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Span> {
    make_span(py, Level::TRACE, name, kwargs)
}

/// Register span constructors and the `Span` class on `m`. Mirrors
/// `crate::register` so consumers re-exporting trapy from another cdylib
/// pick up spans in the same call.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Span>()?;
    m.add_function(wrap_pyfunction!(error_span, m)?)?;
    m.add_function(wrap_pyfunction!(warn_span, m)?)?;
    m.add_function(wrap_pyfunction!(info_span, m)?)?;
    m.add_function(wrap_pyfunction!(debug_span, m)?)?;
    m.add_function(wrap_pyfunction!(trace_span, m)?)?;
    Ok(())
}

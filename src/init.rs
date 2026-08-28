//! Layered subscriber init: collect layers via `add_*`, finalize once
//! with [`init_subscriber`]. Only compiled with the `subscribers`
//! feature — capsule-consumer builds skip this module (and every
//! subscriber crate) entirely.
//!
//! `tracing_subscriber::Registry` composes statically — `Layer`s are
//! folded into a deeply-nested `Layered<L1, Layered<L2, …>>` type. To
//! let Python configure layers dynamically (some at startup, some via
//! `add_fmt_layer(...)` from the composer entry point) we collect
//! `Box<dyn Layer<Registry> + Send + Sync>` in a `Mutex<Vec<…>>` and
//! install once on `init_subscriber()`. The vec is consumed; later
//! `add_*` calls return an error.
//!
//! Available layers: the `tracing-catty` per-category file router,
//! `tracing_subscriber::fmt` in all four formats (full / compact /
//! pretty / JSON) against stdout / stderr / a file, `EnvFilter`,
//! `tracing-timing` HDR histograms, `tracing-tracy` profiler spans,
//! and a `tracing-opentelemetry` OTLP span exporter.
//!
//! This module owns *only* trapy's own tracing-core static (the one
//! linked into trapy._trapy.so). Foreign cdylibs install their own
//! `ForwardingSubscriber` via the capsule and pipe events back through
//! `submit_event`, which re-dispatches into trapy's tracing-core so
//! the same layer chain sees them.

use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use tracing::Subscriber;
use tracing_catty::{CattyConfig, CattyLayer};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::registry::{LookupSpan, Registry};
use tracing_timing::{
    Builder as TimingBuilder, Histogram, TimingLayer, group::ByMessage, group::ByName,
};

static LAYERS: Mutex<Vec<BoxedLayer>> = Mutex::new(Vec::new());
static INIT_DONE: AtomicBool = AtomicBool::new(false);

type BoxedLayer = Box<dyn Layer<Registry> + Send + Sync + 'static>;

/// Generic helper: caller passes any `Layer<Registry> + Send + Sync` —
/// gets boxed and queued.
pub fn add_layer<L>(layer: L) -> Result<(), String>
where
    L: Layer<Registry> + Send + Sync + 'static,
{
    add_boxed_layer(Box::new(layer))
}

fn add_boxed_layer(layer: BoxedLayer) -> Result<(), String> {
    if INIT_DONE.load(Ordering::Acquire) {
        return Err(
            "trapy subscriber already initialised; add layers before init_subscriber()".into(),
        );
    }
    LAYERS
        .lock()
        .map_err(|e| format!("trapy LAYERS mutex poisoned: {}", e))?
        .push(layer);
    Ok(())
}

/// trapy's stock routing policy: `cat="progress"` mirrors to stdout
/// (the user-facing milestone stream), `cat="console"` is stdout-only
/// ephemeral chatter, and ERROR events fan out to `errors.log`.
pub fn default_router_config() -> CattyConfig {
    CattyConfig::new()
        .console_cat("progress")
        .console_only_cat("console")
}

/// Spawn the writer thread for `dir` and queue a `tracing-catty`
/// [`CattyLayer`] with trapy's [`default_router_config`]. Subsequent
/// `add_file_router*` calls fail — the writer is one-shot per process.
pub fn add_file_router(dir: impl AsRef<Path>) -> Result<(), String> {
    add_file_router_with(dir, default_router_config())
}

/// Like [`add_file_router`] with an explicit routing policy.
pub fn add_file_router_with(dir: impl AsRef<Path>, config: CattyConfig) -> Result<(), String> {
    let tx =
        tracing_catty::init_writer_global(dir).map_err(|e| format!("file-router writer: {}", e))?;
    add_layer(CattyLayer::with_config(tx, config))
}

/// Where an `fmt` layer's bytes go.
pub enum FmtTarget {
    Stderr,
    Stdout,
    /// Truncates on open; lines are written through a mutex-serialised
    /// `LineWriter` so concurrent emitters never interleave mid-record.
    File(std::path::PathBuf),
}

/// Event formatter shape, mirroring the four `tracing_subscriber::fmt`
/// formats.
pub enum FmtFormat {
    Full,
    Compact,
    Pretty,
    Json,
}

/// Knob set for [`add_fmt_layer`], covering the `fmt::Layer` builder
/// surface. `json_*` fields only apply to [`FmtFormat::Json`].
pub struct FmtOptions {
    pub format: FmtFormat,
    pub with_target: bool,
    pub with_ansi: bool,
    pub with_level: bool,
    pub with_file: bool,
    pub with_line_number: bool,
    pub with_thread_ids: bool,
    pub with_thread_names: bool,
    pub span_events: FmtSpan,
    pub json_flatten: bool,
    pub json_current_span: bool,
    pub json_span_list: bool,
}

impl Default for FmtOptions {
    fn default() -> Self {
        Self {
            format: FmtFormat::Full,
            with_target: true,
            with_ansi: true,
            with_level: true,
            with_file: false,
            with_line_number: false,
            with_thread_ids: false,
            with_thread_names: false,
            span_events: FmtSpan::NONE,
            json_flatten: false,
            json_current_span: true,
            json_span_list: true,
        }
    }
}

/// Parse a list of span-event names (`"new"`, `"enter"`, `"exit"`,
/// `"close"`, `"active"`, `"full"`, `"none"`) into the `FmtSpan`
/// bitflags consumed by `fmt::Layer::with_span_events`.
pub fn parse_span_events(names: &[String]) -> Result<FmtSpan, String> {
    let mut out = FmtSpan::NONE;
    for name in names {
        out |= match name.to_ascii_lowercase().as_str() {
            "none" => FmtSpan::NONE,
            "new" => FmtSpan::NEW,
            "enter" => FmtSpan::ENTER,
            "exit" => FmtSpan::EXIT,
            "close" => FmtSpan::CLOSE,
            "active" => FmtSpan::ACTIVE,
            "full" => FmtSpan::FULL,
            other => {
                return Err(format!(
                    "unknown span event {:?}; expected one of none/new/enter/exit/close/active/full",
                    other
                ));
            }
        };
    }
    Ok(out)
}

/// Queue a `tracing_subscriber::fmt` layer. The writer and every
/// formatter knob are explicit so the Python surface can reach the
/// whole builder; see [`FmtOptions`] for defaults.
pub fn add_fmt_layer(target: FmtTarget, opts: FmtOptions) -> Result<(), String> {
    let writer = match target {
        FmtTarget::Stderr => BoxMakeWriter::new(std::io::stderr),
        FmtTarget::Stdout => BoxMakeWriter::new(std::io::stdout),
        FmtTarget::File(path) => BoxMakeWriter::new(line_writer(&path)?),
    };
    let base = fmt::layer()
        .with_writer(writer)
        .with_target(opts.with_target)
        .with_ansi(opts.with_ansi)
        .with_level(opts.with_level)
        .with_file(opts.with_file)
        .with_line_number(opts.with_line_number)
        .with_thread_ids(opts.with_thread_ids)
        .with_thread_names(opts.with_thread_names)
        .with_span_events(opts.span_events);
    let layer: BoxedLayer = match opts.format {
        FmtFormat::Full => Box::new(base),
        FmtFormat::Compact => Box::new(base.compact()),
        FmtFormat::Pretty => Box::new(base.pretty()),
        FmtFormat::Json => Box::new(
            base.json()
                .flatten_event(opts.json_flatten)
                .with_current_span(opts.json_current_span)
                .with_span_list(opts.json_span_list),
        ),
    };
    add_boxed_layer(layer)
}

/// JSON-per-line `fmt` layer pointed at an opened file. Useful for
/// machine-readable shipping (Loki, Promtail, etc.). The file is
/// truncated on open — pre-existing content is overwritten. Shorthand
/// for [`add_fmt_layer`] with `FmtFormat::Json` and a file target.
pub fn add_json_file(path: impl AsRef<Path>) -> Result<(), String> {
    add_fmt_layer(
        FmtTarget::File(path.as_ref().to_path_buf()),
        FmtOptions {
            format: FmtFormat::Json,
            with_ansi: false,
            ..FmtOptions::default()
        },
    )
}

fn line_writer(path: &Path) -> Result<ArcMutexWriter<std::io::LineWriter<std::fs::File>>, String> {
    use std::fs::OpenOptions;
    use std::io::LineWriter;
    use std::sync::{Arc, Mutex as StdMutex};

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    // `LineWriter` flushes on every `\n`, so the file stays current
    // even though tracing-core's global dispatcher is never dropped
    // on process exit (a `BufWriter` would silently swallow the tail).
    // The `Arc<Mutex<…>>` serialises concurrent emitters so lines
    // never interleave inside one record.
    Ok(ArcMutexWriter(Arc::new(StdMutex::new(LineWriter::new(
        file,
    )))))
}

/// `MakeWriter` adapter around `Arc<Mutex<W>>`. Each `make_writer()`
/// call grabs the lock and hands back a `MutexGuard` that derefs to
/// the underlying writer, so concurrent dispatchers never interleave
/// inside a single line.
struct ArcMutexWriter<W>(std::sync::Arc<std::sync::Mutex<W>>);

impl<W> Clone for ArcMutexWriter<W> {
    fn clone(&self) -> Self {
        Self(std::sync::Arc::clone(&self.0))
    }
}

impl<'a, W: std::io::Write + 'a> tracing_subscriber::fmt::MakeWriter<'a> for ArcMutexWriter<W> {
    type Writer = LockedWriter<'a, W>;
    fn make_writer(&'a self) -> Self::Writer {
        LockedWriter(
            self.0
                .lock()
                .unwrap_or_else(|e| panic!("trapy fmt-file mutex poisoned: {}", e)),
        )
    }
}

struct LockedWriter<'a, W>(std::sync::MutexGuard<'a, W>);

impl<'a, W: std::io::Write> std::io::Write for LockedWriter<'a, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

/// Type alias for the trapy timing layer. Defaults to grouping by span
/// `name` and event `message` so each (span_name, event_message) pair
/// gets its own histogram of inter-event timings.
pub type TrapyTimingLayer = TimingLayer<ByName, ByMessage>;

/// Install a `tracing-timing` layer that records inter-event timing in
/// HDR histograms keyed by `(span_name, event_message)`. Histograms
/// can later be flushed to the writer thread via [`dump_timing`].
///
/// `max_value_ns` and `precision` configure the per-histogram backing
/// — see [`hdrhistogram::Histogram::new_with_max`]. Defaults of
/// `1_000_000_000` (1 s in ns) and `2` (significant digits) cover the
/// typical per-event range without huge memory overhead.
///
/// The layer is retrieved later (inside [`dump_timing`]) via
/// `Dispatch::downcast_ref` so we don't need to keep an extra handle.
pub fn add_timing_layer(max_value_ns: u64, precision: u8) -> Result<(), String> {
    let layer: TrapyTimingLayer = TimingBuilder::default().layer(move || {
        Histogram::new_with_max(max_value_ns, precision)
            .expect("invalid timing histogram parameters")
    });
    add_layer(layer)
}

/// Snapshot every timing histogram and emit one `cat="timing"` event
/// per `(span, event)` pair carrying mean / p50 / p90 / p99 / max in
/// nanoseconds plus the total sample count.
///
/// Returns the number of (span, event) pairs reported. `0` means the
/// timing layer is not installed or no events have flowed through it.
pub fn dump_timing() -> usize {
    let mut reported = 0usize;
    tracing::dispatcher::get_default(|dispatcher| {
        let Some(layer) = dispatcher.downcast_ref::<TrapyTimingLayer>() else {
            return;
        };
        layer.force_synchronize();
        layer.with_histograms(|map| {
            for (span_name, events) in map.iter_mut() {
                for (event_name, hist) in events.iter_mut() {
                    hist.refresh();
                    let count = hist.len();
                    if count == 0 {
                        continue;
                    }
                    let mean = hist.mean();
                    let p50 = hist.value_at_quantile(0.50);
                    let p90 = hist.value_at_quantile(0.90);
                    let p99 = hist.value_at_quantile(0.99);
                    let max = hist.max();
                    let span_str: &str = span_name;
                    let event_str: &str = event_name;
                    tracing::info!(
                        cat = "timing",
                        span = span_str,
                        event = event_str,
                        count = count,
                        mean_ns = mean,
                        p50_ns = p50,
                        p90_ns = p90,
                        p99_ns = p99,
                        max_ns = max,
                        "timing"
                    );
                    reported += 1;
                }
            }
        });
    });
    reported
}

/// The Tracy client handle, held for the process lifetime so the
/// connection stays open after [`add_tracy_layer`] returns.
static TRACY_CLIENT: OnceLock<tracing_tracy::client::Client> = OnceLock::new();

/// Install a `tracing-tracy` layer so spans and events stream to a
/// connected [Tracy](https://github.com/wolfpld/tracy) profiler.
///
/// Starts the Tracy client eagerly (and keeps the handle alive for the
/// process lifetime) — otherwise the layer would only connect once the
/// first span fires.
pub fn add_tracy_layer() -> Result<(), String> {
    let _ = TRACY_CLIENT.set(tracing_tracy::client::Client::start());
    add_layer(tracing_tracy::TracyLayer::default())
}

/// The OTLP tracer provider backing the opentelemetry layer. Held so
/// [`opentelemetry_flush`] / [`opentelemetry_shutdown`] can reach the
/// batch exporter after init.
static OTEL_PROVIDER: OnceLock<opentelemetry_sdk::trace::SdkTracerProvider> = OnceLock::new();

/// Install a `tracing-opentelemetry` layer exporting spans over OTLP.
///
/// `endpoint` defaults to the exporter's standard resolution
/// (`OTEL_EXPORTER_OTLP_ENDPOINT` or `http://localhost:4318`). The
/// transport is HTTP + binary protobuf on a blocking `reqwest` client
/// — deliberately not gRPC/tonic, which would embed a tokio runtime in
/// every process that imports trapy. The batch span processor runs on
/// its own background thread; call [`opentelemetry_shutdown`] before
/// exit to drain it, or tail spans are lost.
pub fn add_opentelemetry_layer(endpoint: Option<&str>, service_name: &str) -> Result<(), String> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::WithExportConfig as _;

    if OTEL_PROVIDER.get().is_some() {
        return Err("opentelemetry layer already configured".into());
    }
    let mut builder = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary);
    if let Some(ep) = endpoint {
        builder = builder.with_endpoint(ep);
    }
    let exporter = builder
        .build()
        .map_err(|e| format!("otlp span exporter: {}", e))?;
    let resource = opentelemetry_sdk::Resource::builder()
        .with_service_name(service_name.to_string())
        .build();
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();
    let tracer = provider.tracer("trapy");
    if OTEL_PROVIDER.set(provider).is_err() {
        return Err("opentelemetry layer already configured".into());
    }
    add_layer(tracing_opentelemetry::layer().with_tracer(tracer))
}

/// Block until the OTLP batch exporter has shipped everything queued
/// so far. `false` if the opentelemetry layer isn't installed or the
/// flush failed.
pub fn opentelemetry_flush() -> bool {
    OTEL_PROVIDER
        .get()
        .map(|p| p.force_flush().is_ok())
        .unwrap_or(false)
}

/// Drain and shut down the OTLP batch exporter. Idempotent at the
/// caller's risk — a second call returns `false`.
pub fn opentelemetry_shutdown() -> bool {
    OTEL_PROVIDER
        .get()
        .map(|p| p.shutdown().is_ok())
        .unwrap_or(false)
}

/// Add an `EnvFilter` directive to the subscriber. Filters are
/// `Layer<S>` themselves so they compose alongside the others — events
/// rejected by the filter never reach the downstream layers.
///
/// `directive` follows the standard `RUST_LOG` syntax (e.g.
/// `info,planning=trace,orchestra::bindings::c_python=debug`).
pub fn add_env_filter(directive: &str) -> Result<(), String> {
    let filter = EnvFilter::from_str(directive)
        .map_err(|e| format!("invalid env-filter directive {:?}: {}", directive, e))?;
    add_layer(filter)
}

/// Add an `EnvFilter` read from an environment variable (`RUST_LOG`
/// when `var` is `None`). Errors if the variable is unset or holds an
/// invalid directive, so a typo'd filter fails loudly instead of
/// silently recording everything.
pub fn add_env_filter_from_env(var: Option<&str>) -> Result<(), String> {
    let builder = EnvFilter::builder();
    let filter = match var {
        Some(v) => builder.with_env_var(v).from_env(),
        None => builder.from_env(),
    }
    .map_err(|e| format!("env-filter from {}: {}", var.unwrap_or("RUST_LOG"), e))?;
    add_layer(filter)
}

/// Drain the queued layers, install a `Registry` with all of them as
/// the global default for *this cdylib's* tracing-core static, and
/// freeze further layer additions.
///
/// One-shot. Returns `Err` if no layers are configured (which would be
/// a no-op subscriber and almost certainly a programming mistake) or
/// if `set_global_default` was already claimed.
pub fn init_subscriber() -> Result<(), String> {
    if INIT_DONE.swap(true, Ordering::AcqRel) {
        return Err("trapy::init_subscriber called twice".into());
    }
    let mut layers = LAYERS
        .lock()
        .map_err(|e| format!("trapy LAYERS mutex poisoned: {}", e))?;
    if layers.is_empty() {
        return Err("no layers configured; call add_file_router/add_fmt_layer/... first".into());
    }
    let drained: Vec<BoxedLayer> = layers.drain(..).collect();
    drop(layers);

    install_layers(drained)
}

fn install_layers(layers: Vec<BoxedLayer>) -> Result<(), String> {
    // `Vec<L: Layer<S>>: Layer<S>` — the impl forwards every callback
    // to each layer in order.
    let subscriber: Box<dyn Subscriber + Send + Sync> = Box::new(Registry::default().with(layers));
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| format!("set_global_default failed: {}", e))
}

/// Convenience: `add_file_router(dir) + init_subscriber()`. Matches the
/// pre-layering API and is what the composer's `init_tracing(dir)` call
/// resolves to.
pub fn init_tracing(dir: impl AsRef<Path>) -> Result<(), String> {
    add_file_router(dir)?;
    init_subscriber()
}

/// True if `init_subscriber` has been called (with whatever set of
/// layers was configured at the time). Used by the capsule's
/// `submit_event` path to decide whether to dispatch or drop.
pub fn is_installed() -> bool {
    INIT_DONE.load(Ordering::Acquire)
}

// `LookupSpan` import is needed for the `with` call to pick the right
// `Layer for Vec<L>` impl — kept here so the mod doesn't collect a
// wandering `use` warning when the type isn't named directly.
#[allow(dead_code)]
fn _phantom_use<S: for<'a> LookupSpan<'a>>() {}

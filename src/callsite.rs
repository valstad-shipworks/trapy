//! Dynamic `tracing` callsite cache keyed by `(file, line, level, kind, name, kwarg names)`.
//!
//! `tracing`'s public API expects `&'static Metadata<'static>` with a static
//! `FieldSet`. To make Python `trapy.info("msg", a=1, b=2)` produce the same
//! shape of event as Rust `tracing::info!(a = 1, b = 2, "msg")` — i.e. one
//! real `tracing` field per kwarg, not a single Valuable blob — we synthesize
//! a callsite per unique combination of source location *and* kwarg names.
//! Each callsite leaks its own `Vec<&'static str>` field-name array so the
//! `FieldSet` can borrow it as `&'static [&'static str]`.
//!
//! The same machinery serves spans (`Kind::SPAN`, name carried in metadata,
//! field set is just the kwarg names with no leading `message`).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use tracing_core::callsite::{Callsite, Identifier};
use tracing_core::field::FieldSet;
use tracing_core::metadata::{Kind, Metadata};
use tracing_core::{Interest, Level};

/// First field of every trapy *event* is the human-readable `message`,
/// matching the convention `tracing::event!` uses for the format string.
/// Spans do not get a synthetic `message` field — the human-readable name
/// lives in `Metadata::name()` instead.
pub const MESSAGE_FIELD: &str = "message";

pub struct DynCallsite {
    meta: OnceLock<Metadata<'static>>,
}

impl DynCallsite {
    /// Returns the leaked, `'static` metadata. Safe to use for
    /// `tracing::Span::new`, which requires `&'static Metadata<'static>`.
    /// Caller must hold a `'static` reference to the callsite (which they do
    /// — `get_event` / `get_span` only ever return `&'static DynCallsite`).
    pub fn meta_static(&'static self) -> &'static Metadata<'static> {
        self.meta
            .get()
            .expect("trapy callsite metadata not initialized")
    }
}

impl Callsite for DynCallsite {
    fn set_interest(&self, _interest: Interest) {
        // Subscribers re-evaluate via `Dispatch::enabled` on every emit, so
        // there's no per-callsite cached interest like `DefaultCallsite`.
    }

    fn metadata(&self) -> &Metadata<'_> {
        self.meta
            .get()
            .expect("trapy callsite metadata not initialized")
    }
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct Key {
    file: String,
    line: u32,
    level: u8,
    /// `true` for `Kind::SPAN`, `false` for `Kind::EVENT`. Span/event
    /// callsites at the same source location must not collide because they
    /// have different field sets and metadata kinds.
    is_span: bool,
    /// Span name (event metadata always uses the fixed `"trapy_event"`).
    name: String,
    /// Kwarg names in caller order. Sorting would lose the "stable order
    /// preserved from the dict iteration" guarantee, which downstream
    /// formatters rely on.
    kwarg_names: Vec<String>,
}

fn level_byte(l: Level) -> u8 {
    match l {
        Level::ERROR => 1,
        Level::WARN => 2,
        Level::INFO => 3,
        Level::DEBUG => 4,
        Level::TRACE => 5,
    }
}

fn intern(s: &str) -> &'static str {
    static INTERN: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let m = INTERN.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = m.lock().unwrap();
    if let Some(p) = g.get(s) {
        return p;
    }
    let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
    g.insert(s.to_string(), leaked);
    leaked
}

fn registry() -> &'static Mutex<HashMap<Key, &'static DynCallsite>> {
    static REGISTRY: OnceLock<Mutex<HashMap<Key, &'static DynCallsite>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

#[allow(clippy::too_many_arguments)]
fn build_callsite(
    name: &'static str,
    target: &'static str,
    level: Level,
    file: &'static str,
    line: u32,
    module: &'static str,
    field_names: &'static [&'static str],
    kind: Kind,
) -> &'static DynCallsite {
    let cs: &'static DynCallsite = Box::leak(Box::new(DynCallsite {
        meta: OnceLock::new(),
    }));
    let identifier = Identifier(cs as &'static dyn Callsite);
    let fieldset = FieldSet::new(field_names, identifier);
    let meta = Metadata::new(
        name,
        target,
        level,
        Some(file),
        Some(line),
        Some(module),
        fieldset,
        kind,
    );
    let _ = cs.meta.set(meta);
    tracing_core::callsite::register(cs);
    cs
}

/// Look up (or build) an *event* callsite. The returned `DynCallsite`
/// has its `Metadata` / `FieldSet` constructed so that the field set is
/// exactly `[message, kwarg_names...]` in caller order, with the given
/// `target` and `module` woven in.
///
/// Generic over Python/Rust callers — the Python emit path supplies a
/// `py:<module>` target (see [`get_event`]); foreign-Rust forwarders
/// pass the actual `tracing` target string directly.
pub fn get_event_full(
    file: &str,
    line: u32,
    level: Level,
    target: &str,
    module: &str,
    kwarg_names: &[&str],
) -> &'static DynCallsite {
    let key = Key {
        file: file.to_string(),
        line,
        level: level_byte(level),
        is_span: false,
        name: String::new(),
        kwarg_names: kwarg_names.iter().map(|s| s.to_string()).collect(),
    };

    {
        let g = registry().lock().unwrap();
        if let Some(cs) = g.get(&key) {
            return cs;
        }
    }

    let mut g = registry().lock().unwrap();
    if let Some(cs) = g.get(&key) {
        return cs;
    }

    let file_static = intern(file);
    let target_static = intern(target);
    let module_static = intern(module);

    let mut names: Vec<&'static str> = Vec::with_capacity(kwarg_names.len() + 1);
    names.push(MESSAGE_FIELD);
    for k in kwarg_names {
        names.push(intern(k));
    }
    let field_names: &'static [&'static str] = Box::leak(names.into_boxed_slice());

    let cs = build_callsite(
        "trapy_event",
        target_static,
        level,
        file_static,
        line,
        module_static,
        field_names,
        Kind::EVENT,
    );
    g.insert(key, cs);
    cs
}

/// Backwards-compatible wrapper: derives `target = "py:<py_module>"`.
/// The Python emit path uses this; foreign-Rust forwarders should call
/// [`get_event_full`] directly so their actual target lands in the
/// metadata.
pub fn get_event(
    file: &str,
    line: u32,
    level: Level,
    py_module: &str,
    kwarg_names: &[&str],
) -> &'static DynCallsite {
    let target = format!("py:{}", py_module);
    get_event_full(file, line, level, &target, py_module, kwarg_names)
}

/// Look up (or build) a *span* callsite. Field set is exactly
/// `[kwarg_names...]` (no synthetic `message` — the human-readable label
/// lives in `Metadata::name`).
pub fn get_span(
    file: &str,
    line: u32,
    level: Level,
    py_module: &str,
    name: &str,
    kwarg_names: &[&str],
) -> &'static DynCallsite {
    let key = Key {
        file: file.to_string(),
        line,
        level: level_byte(level),
        is_span: true,
        name: name.to_string(),
        kwarg_names: kwarg_names.iter().map(|s| s.to_string()).collect(),
    };

    {
        let g = registry().lock().unwrap();
        if let Some(cs) = g.get(&key) {
            return cs;
        }
    }

    let mut g = registry().lock().unwrap();
    if let Some(cs) = g.get(&key) {
        return cs;
    }

    let file_static = intern(file);
    let target_static = intern(&format!("py:{}", py_module));
    let module_static = intern(py_module);
    let name_static = intern(name);

    let mut names: Vec<&'static str> = Vec::with_capacity(kwarg_names.len());
    for k in kwarg_names {
        names.push(intern(k));
    }
    let field_names: &'static [&'static str] = Box::leak(names.into_boxed_slice());

    let cs = build_callsite(
        name_static,
        target_static,
        level,
        file_static,
        line,
        module_static,
        field_names,
        Kind::SPAN,
    );
    g.insert(key, cs);
    cs
}

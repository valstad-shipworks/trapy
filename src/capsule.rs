//! C-ABI handshake so other PyO3 cdylibs can push events into trapy's
//! writer thread without each cdylib growing its own backend.
//!
//! Each loaded cdylib that pulls in `tracing-core` keeps its own copy of
//! the global dispatcher static, so simply linking trapy as an rlib in
//! both trapy._trapy.so and orchestra._orchestra_core.so isn't enough — every
//! cdylib must install its own `Subscriber`. The capsule lets the foreign
//! subscriber forward into trapy's owner-side ringbuffer via raw function
//! pointers that bypass the duplicated statics.
//!
//! Layout is `#[repr(C)]` and frozen by the [`CAPSULE_VERSION`] field; if
//! we ever need to change the table we bump the version and the consumer
//! refuses an older one.

use std::ffi::CStr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "subscribers")]
use pyo3::ffi::PyCapsule_New;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};
use tracing::span::{Attributes, Record};
use tracing::{Event, Id, Metadata, Subscriber};

#[cfg(feature = "subscribers")]
use tracing::Level as TLevel;
#[cfg(feature = "subscribers")]
use tracing_core::callsite::Callsite as _;
#[cfg(feature = "subscribers")]
use tracing_core::field::Value;

#[cfg(feature = "subscribers")]
use crate::callsite::get_event_full;
#[cfg(feature = "subscribers")]
use crate::init as bg_init;
use crate::level::level_byte;
#[cfg(feature = "subscribers")]
use crate::level::{
    LEVEL_DEBUG, LEVEL_ERROR, LEVEL_INFO, LEVEL_OFF, LEVEL_TRACE, LEVEL_WARN, current_level_byte,
    set_level_byte,
};
#[cfg(feature = "subscribers")]
use tracing_catty::{flush as bg_flush, flush_sync as bg_flush_sync, unix_ms_now};

/// Bumped whenever the `TrapyCapsule` layout changes in a not-backwards-
/// compatible way. Consumers must compare `version >= MIN_SUPPORTED`.
pub const CAPSULE_VERSION: u32 = 2;

/// Capsule lookup name. Must match the byte sequence used by
/// `PyCapsule_Import` on the consumer side. The C form has a `'static`
/// lifetime because `PyCapsule_New` doesn't copy the name — it stores
/// the raw pointer and reads from it whenever `PyCapsule_GetName` runs.
pub const CAPSULE_NAME: &str = "trapy._capsule";
pub const CAPSULE_NAME_C: &CStr = c"trapy._capsule";

/// Function-pointer table handed to other cdylibs via `PyCapsule`. The
/// pointers all live in trapy._trapy.so's address space; foreign callers
/// invoke them as plain C calls.
#[repr(C)]
pub struct TrapyCapsule {
    pub version: u32,
    /// Initialize the writer-thread + ringbuffer, creating `dir` if
    /// missing. `0` on success, `1` if already initialized, `-1` on I/O
    /// error. `dir_ptr` must point to `dir_len` UTF-8 bytes.
    pub init: extern "C" fn(dir_ptr: *const u8, dir_len: usize) -> i32,
    /// Re-emit a structured event on trapy._trapy.so's tracing-core,
    /// running it through the full layer chain (EnvFilter, fmt,
    /// FileRouter, JSON, …). All pointers are borrowed for the
    /// duration of the call only.
    pub submit_event: extern "C" fn(
        level: u8,
        target_ptr: *const u8,
        target_len: usize,
        file_ptr: *const u8,
        file_len: usize,
        line_no: u32,
        module_ptr: *const u8,
        module_len: usize,
        fields_ptr: *const ForeignField,
        fields_count: usize,
    ),
    pub flush: extern "C" fn(),
    /// Returns 1 on ack, 0 if backend not initialised or writer exited.
    pub flush_sync: extern "C" fn() -> u8,
    pub set_level_byte: extern "C" fn(u8),
    pub current_level_byte: extern "C" fn() -> u8,
    /// Single-source-of-truth timestamp for cross-cdylib producers, so
    /// every emitter on this process reads the same wall clock.
    pub unix_ms_now: extern "C" fn() -> u64,
}

/// Field-kind tag for [`ForeignField`]. Mirrors the variants of
/// `valuable::Value` we care about; everything else gets stringified
/// on the producer side.
pub const FIELD_KIND_NULL: u8 = 0;
pub const FIELD_KIND_BOOL: u8 = 1;
pub const FIELD_KIND_I64: u8 = 2;
pub const FIELD_KIND_U64: u8 = 3;
pub const FIELD_KIND_F64: u8 = 4;
pub const FIELD_KIND_STR: u8 = 5;

/// Single typed field shipped through `submit_event`. The producer
/// owns all referenced strings — pointers are borrowed for the
/// lifetime of one `submit_event` call.
#[repr(C)]
pub struct ForeignField {
    pub name_ptr: *const u8,
    pub name_len: usize,
    pub kind: u8,
    pub bool_val: u8,
    pub i64_val: i64,
    pub u64_val: u64,
    pub f64_val: f64,
    pub str_ptr: *const u8,
    pub str_len: usize,
}

// ── extern "C" thunks ───────────────────────────────────────────────────

#[cfg(feature = "subscribers")]
extern "C" fn cap_init(dir_ptr: *const u8, dir_len: usize) -> i32 {
    if dir_ptr.is_null() {
        return -1;
    }
    let bytes = unsafe { std::slice::from_raw_parts(dir_ptr, dir_len) };
    let dir = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    // Layered subscriber convenience: file router only. Callers that
    // want fmt/env-filter/json layers configure them via the Python
    // surface (`trapy.add_*` + `trapy.init_subscriber()`) instead of
    // through this capsule entry point.
    match bg_init::init_tracing(dir) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[cfg(feature = "subscribers")]
extern "C" fn cap_flush() {
    bg_flush();
}

#[cfg(feature = "subscribers")]
extern "C" fn cap_flush_sync() -> u8 {
    if bg_flush_sync() { 1 } else { 0 }
}

#[cfg(feature = "subscribers")]
extern "C" fn cap_set_level(b: u8) {
    set_level_byte(b);
}

#[cfg(feature = "subscribers")]
extern "C" fn cap_current_level() -> u8 {
    current_level_byte()
}

#[cfg(feature = "subscribers")]
extern "C" fn cap_unix_ms_now() -> u64 {
    unix_ms_now()
}

#[cfg(feature = "subscribers")]
fn level_from_byte(b: u8) -> Option<TLevel> {
    match b {
        LEVEL_OFF => None,
        LEVEL_ERROR => Some(TLevel::ERROR),
        LEVEL_WARN => Some(TLevel::WARN),
        LEVEL_INFO => Some(TLevel::INFO),
        LEVEL_DEBUG => Some(TLevel::DEBUG),
        LEVEL_TRACE => Some(TLevel::TRACE),
        _ => Some(TLevel::TRACE),
    }
}

#[cfg(feature = "subscribers")]
unsafe fn ptr_to_str<'a>(p: *const u8, len: usize) -> &'a str {
    if p.is_null() || len == 0 {
        return "";
    }
    let bytes = unsafe { std::slice::from_raw_parts(p, len) };
    std::str::from_utf8(bytes).unwrap_or("<invalid utf-8>")
}

#[cfg(feature = "subscribers")]
extern "C" fn cap_submit_event(
    level: u8,
    target_ptr: *const u8,
    target_len: usize,
    file_ptr: *const u8,
    file_len: usize,
    line_no: u32,
    module_ptr: *const u8,
    module_len: usize,
    fields_ptr: *const ForeignField,
    fields_count: usize,
) {
    let Some(lvl) = level_from_byte(level) else {
        return;
    };

    // Coarse early drop: if the capsule-side level filter is below this
    // level, skip the dispatch entirely.
    if level_byte(&lvl) > current_level_byte() {
        return;
    }

    // SAFETY: the caller guarantees the pointers stay valid for the
    // duration of this call, and field counts/lengths describe valid
    // UTF-8 byte ranges.
    let target = unsafe { ptr_to_str(target_ptr, target_len) };
    let file = unsafe { ptr_to_str(file_ptr, file_len) };
    let module = unsafe { ptr_to_str(module_ptr, module_len) };

    let fields_slice: &[ForeignField] = if fields_ptr.is_null() || fields_count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(fields_ptr, fields_count) }
    };

    // Split out `message` (becomes the synthesized callsite's first
    // field, matching the in-process emit path) from the rest.
    let mut message: &str = "";
    let mut other: Vec<(&str, &ForeignField)> = Vec::with_capacity(fields_slice.len());
    for f in fields_slice {
        let name = unsafe { ptr_to_str(f.name_ptr, f.name_len) };
        if name == "message" {
            // For `message`, only string is meaningful — the producer
            // is responsible for stringifying anything else.
            if f.kind == FIELD_KIND_STR {
                message = unsafe { ptr_to_str(f.str_ptr, f.str_len) };
            }
        } else {
            other.push((name, f));
        }
    }

    let kwarg_names: Vec<&str> = other.iter().map(|(n, _)| *n).collect();
    let cs = get_event_full(file, line_no, lvl, target, module, &kwarg_names);
    let meta = cs.metadata();

    tracing::dispatcher::get_default(|dispatcher| {
        if !dispatcher.enabled(meta) {
            return;
        }
        let fields = meta.fields();

        // Field order matches `[message, other...]`. value_set_all
        // requires `values.len() == fields.len()`, so we always emit a
        // slot for `message` even when empty.
        let mut values: Vec<Option<&dyn Value>> = Vec::with_capacity(other.len() + 1);
        values.push(if message.is_empty() {
            None
        } else {
            Some(&message as &dyn Value)
        });

        // Parallel arrays for primitive values so each slot has a
        // stable reference. The dispatcher only borrows for the
        // duration of `dispatcher.event(...)` below.
        let bool_slots: Vec<bool> = other.iter().map(|(_, f)| f.bool_val != 0).collect();
        let i64_slots: Vec<i64> = other.iter().map(|(_, f)| f.i64_val).collect();
        let u64_slots: Vec<u64> = other.iter().map(|(_, f)| f.u64_val).collect();
        let f64_slots: Vec<f64> = other.iter().map(|(_, f)| f.f64_val).collect();
        let str_views: Vec<&str> = other
            .iter()
            .map(|(_, f)| unsafe { ptr_to_str(f.str_ptr, f.str_len) })
            .collect();

        for (i, (_, f)) in other.iter().enumerate() {
            let v: Option<&dyn Value> = match f.kind {
                FIELD_KIND_NULL => None,
                FIELD_KIND_BOOL => Some(&bool_slots[i] as &dyn Value),
                FIELD_KIND_I64 => Some(&i64_slots[i] as &dyn Value),
                FIELD_KIND_U64 => Some(&u64_slots[i] as &dyn Value),
                FIELD_KIND_F64 => Some(&f64_slots[i] as &dyn Value),
                FIELD_KIND_STR => Some(&str_views[i] as &dyn Value),
                _ => None,
            };
            values.push(v);
        }
        debug_assert_eq!(values.len(), fields.len());

        let value_set = fields.value_set_all(&values);
        dispatcher.event(&tracing::Event::new(meta, &value_set));
    });
}

/// Singleton table — the very same address is wrapped in every PyCapsule
/// returned to Python, so foreign cdylibs can cache the pointer once.
/// Owner-side only: the thunks all reach into the `subscribers` backend.
#[cfg(feature = "subscribers")]
pub static TRAPY_CAPSULE: TrapyCapsule = TrapyCapsule {
    version: CAPSULE_VERSION,
    init: cap_init,
    submit_event: cap_submit_event,
    flush: cap_flush,
    flush_sync: cap_flush_sync,
    set_level_byte: cap_set_level,
    current_level_byte: cap_current_level,
    unix_ms_now: cap_unix_ms_now,
};

// ── ForwardingSubscriber + import helper ────────────────────────────────
//
// The host cdylib (e.g. orchestra._orchestra_core) statically links its own
// copy of `tracing-core`, so its `GLOBAL_DISPATCH` static is distinct
// from trapy._trapy.so's. To get the host's `tracing::*!` events into
// trapy's writer thread *and* through every layer (`EnvFilter` /
// `fmt` / JSON / file router) installed on trapy's subscriber, the
// host installs `ForwardingSubscriber` as its own global default. The
// subscriber visits each event locally, packs the fields into a flat
// `Vec<ForeignField>`, then hands them to trapy via the capsule's
// `submit_event` pointer — which re-emits on trapy's tracing-core so
// the layer chain runs.
//
// The capsule pointer that `ForwardingSubscriber` carries lives in
// trapy._trapy.so's address space, so calling its function pointers
// reaches trapy's owner-side state regardless of which cdylib emitted
// the event.

static IMPORTED_CAPSULE: OnceLock<&'static TrapyCapsule> = OnceLock::new();

struct ForwardingSubscriber {
    next_id: AtomicU64,
    cap: &'static TrapyCapsule,
}

/// Producer-side visitor that captures field values as owned typed
/// data, ready to be reshaped into [`ForeignField`] for the FFI call.
/// Mirrors the kinds the in-process `value::PyValue` flow handles.
enum OwnedFieldVal {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    Str(String),
}

struct ForeignVisitor {
    fields: Vec<(String, OwnedFieldVal)>,
}

impl ForeignVisitor {
    fn new() -> Self {
        Self { fields: Vec::new() }
    }
    fn push(&mut self, name: &str, v: OwnedFieldVal) {
        self.fields.push((name.to_string(), v));
    }
}

impl tracing::field::Visit for ForeignVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.push(field.name(), OwnedFieldVal::Str(value.to_string()));
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.push(field.name(), OwnedFieldVal::Str(format!("{:?}", value)));
    }
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.push(field.name(), OwnedFieldVal::I64(value));
    }
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.push(field.name(), OwnedFieldVal::U64(value));
    }
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.push(field.name(), OwnedFieldVal::F64(value));
    }
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.push(field.name(), OwnedFieldVal::Bool(value));
    }
    fn record_error(
        &mut self,
        field: &tracing::field::Field,
        value: &(dyn std::error::Error + 'static),
    ) {
        self.push(field.name(), OwnedFieldVal::Str(value.to_string()));
    }
    fn record_value(&mut self, field: &tracing::field::Field, value: valuable::Value<'_>) {
        self.push(field.name(), OwnedFieldVal::Str(format!("{:?}", value)));
    }
}

impl Subscriber for ForwardingSubscriber {
    fn enabled(&self, meta: &Metadata<'_>) -> bool {
        level_byte(meta.level()) <= (self.cap.current_level_byte)()
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        let raw = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        Id::from_u64(raw)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        if level_byte(event.metadata().level()) > (self.cap.current_level_byte)() {
            return;
        }

        let mut v = ForeignVisitor::new();
        event.record(&mut v);
        let meta = event.metadata();

        // Pack `OwnedFieldVal` → `ForeignField`. Strings have to be
        // referenced from the original String to keep the lifetime
        // alive through the FFI call, so we hold them in a parallel
        // Vec and only build the ForeignField slice after.
        let foreign: Vec<ForeignField> = v
            .fields
            .iter()
            .map(|(name, val)| {
                let mut f = ForeignField {
                    name_ptr: name.as_ptr(),
                    name_len: name.len(),
                    kind: FIELD_KIND_NULL,
                    bool_val: 0,
                    i64_val: 0,
                    u64_val: 0,
                    f64_val: 0.0,
                    str_ptr: std::ptr::null(),
                    str_len: 0,
                };
                match val {
                    OwnedFieldVal::Bool(b) => {
                        f.kind = FIELD_KIND_BOOL;
                        f.bool_val = if *b { 1 } else { 0 };
                    }
                    OwnedFieldVal::I64(i) => {
                        f.kind = FIELD_KIND_I64;
                        f.i64_val = *i;
                    }
                    OwnedFieldVal::U64(u) => {
                        f.kind = FIELD_KIND_U64;
                        f.u64_val = *u;
                    }
                    OwnedFieldVal::F64(x) => {
                        f.kind = FIELD_KIND_F64;
                        f.f64_val = *x;
                    }
                    OwnedFieldVal::Str(s) => {
                        f.kind = FIELD_KIND_STR;
                        f.str_ptr = s.as_ptr();
                        f.str_len = s.len();
                    }
                }
                f
            })
            .collect();

        let target = meta.target();
        let file = meta.file().unwrap_or("");
        let line_no = meta.line().unwrap_or(0);
        let module = meta.module_path().unwrap_or("");

        (self.cap.submit_event)(
            level_byte(meta.level()),
            target.as_ptr(),
            target.len(),
            file.as_ptr(),
            file.len(),
            line_no,
            module.as_ptr(),
            module.len(),
            foreign.as_ptr(),
            foreign.len(),
        );
        // `foreign` and `v.fields` (which owns the strings) live until
        // here — pointers in `foreign` are valid for the duration of
        // the synchronous `submit_event` call above.
        drop(foreign);
        drop(v);
    }

    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

/// One-call handshake for host cdylibs: `import trapy`, fetch the
/// capsule, validate the name + version, and install a forwarding
/// subscriber on the *calling cdylib's* tracing-core static.
///
/// **Fail-safe**: this function never raises. If trapy isn't installed,
/// the capsule is missing/malformed, or `set_global_default` was
/// already claimed, it logs one line to stderr and returns `false`.
/// The host cdylib's tracing-core static then keeps its default null
/// subscriber, which silently drops all events — i.e. orchestra still
/// loads, Rust-side `tracing::*!` calls become no-ops. Returns `true`
/// when the forwarding subscriber is live (installed by this call or a
/// prior one).
///
/// Call from the host's `#[pymodule]` init function — the Python frame
/// must already exist (pyo3 supplies it). The `tracing-core` global
/// default is one-shot per static, so this must run before any other
/// code in the host cdylib calls `set_global_default` on its own.
pub fn import_trapy_subscriber(py: Python<'_>) -> bool {
    if IMPORTED_CAPSULE.get().is_some() {
        return true;
    }
    match try_import_and_install(py) {
        Ok(()) => true,
        Err(e) => {
            // One line to stderr so a missing trapy is debuggable without
            // also crashing the host's module load. Use `eprintln!`
            // rather than `tracing::warn!` — we *are* the tracing setup
            // path, the subscriber doesn't exist yet, and a self-
            // referential dispatch through the null subscriber would
            // just be dropped anyway.
            eprintln!(
                "trapy::import_trapy_subscriber: subscriber NOT installed ({}); tracing events from this cdylib will be dropped",
                e
            );
            // Clear the PyErr that may still be set on the interpreter
            // — pymodule init treats a pending error as failure even if
            // we return Ok.
            if pyo3::PyErr::occurred(py) {
                pyo3::PyErr::fetch(py);
            }
            false
        }
    }
}

fn try_import_and_install(py: Python<'_>) -> PyResult<()> {
    let trapy_mod = py.import("trapy")?;
    let cap_obj = trapy_mod.getattr("_capsule")?.call0()?;
    let cap: Bound<'_, PyCapsule> = cap_obj.cast_into::<PyCapsule>()?;

    // Match the name explicitly so a wrong-name capsule fails with a
    // useful error rather than silently dereferencing the wrong vtable.
    // `pointer_checked` does the same check internally but its error
    // message is generic.
    {
        let actual_holder = cap.name()?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("trapy capsule has no name")
        })?;
        // SAFETY: the capsule keeps the name string alive for at least
        // as long as the capsule itself (we hold `cap` for this scope).
        let actual = unsafe { actual_holder.as_cstr() };
        if actual != CAPSULE_NAME_C {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "expected capsule named {:?}, got {:?}",
                CAPSULE_NAME,
                actual.to_string_lossy().into_owned(),
            )));
        }
    }

    let raw = cap.pointer_checked(Some(CAPSULE_NAME_C))?.as_ptr() as *const TrapyCapsule;
    if raw.is_null() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "trapy capsule pointer is null",
        ));
    }

    // SAFETY: `raw` points into trapy._trapy.so's `TRAPY_CAPSULE`
    // static, which lives for the duration of the process. The struct
    // is `#[repr(C)]` and `Sync` (function pointers + a u32), so a
    // shared `&'static` reference is sound.
    let cap_ref: &'static TrapyCapsule = unsafe { &*raw };
    if cap_ref.version < CAPSULE_VERSION {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "trapy capsule version {} older than required {}",
            cap_ref.version, CAPSULE_VERSION,
        )));
    }

    if IMPORTED_CAPSULE.set(cap_ref).is_err() {
        // Lost a race with another thread doing the same install. The
        // global default may already be set; treat as success.
        return Ok(());
    }

    let sub = ForwardingSubscriber {
        next_id: AtomicU64::new(0),
        cap: cap_ref,
    };
    tracing::subscriber::set_global_default(sub).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "tracing::set_global_default failed: {}",
            e
        ))
    })?;
    Ok(())
}

/// Build a fresh `PyCapsule` wrapping `&TRAPY_CAPSULE`. The capsule has
/// no destructor — the underlying static lives forever — and uses the
/// well-known [`CAPSULE_NAME`] so consumers can validate via
/// `PyCapsule_IsValid`.
#[cfg(feature = "subscribers")]
pub fn build_capsule(py: Python<'_>) -> PyResult<Bound<'_, PyCapsule>> {
    // Use the raw FFI directly: pyo3's `PyCapsule::new` insists on a
    // value the capsule owns, but we want the pointer to point at a
    // process-wide static and have no destructor. The name pointer must
    // also outlive the capsule — `CAPSULE_NAME_C` is a `'static` CStr,
    // so that's free.
    unsafe {
        let raw = PyCapsule_New(
            &TRAPY_CAPSULE as *const TrapyCapsule as *mut std::ffi::c_void,
            CAPSULE_NAME_C.as_ptr(),
            None,
        );
        if raw.is_null() {
            return Err(pyo3::PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, raw).cast_into::<PyCapsule>()?)
    }
}

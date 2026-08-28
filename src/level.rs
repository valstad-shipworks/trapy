//! Coarse process-wide level filter consulted by the cross-FFI capsule
//! path so foreign cdylibs can do an early drop without paying for
//! formatting. The native (in-process) layered `Subscriber` uses
//! `EnvFilter` etc. for its own filtering — this knob only gates the
//! boundary. Always compiled: the capsule *consumer* side needs the
//! byte mapping even when the `subscribers` backend is compiled out.

use std::sync::atomic::{AtomicU8, Ordering};

use tracing::Level;

pub const LEVEL_OFF: u8 = 0;
pub const LEVEL_ERROR: u8 = 1;
pub const LEVEL_WARN: u8 = 2;
pub const LEVEL_INFO: u8 = 3;
pub const LEVEL_DEBUG: u8 = 4;
pub const LEVEL_TRACE: u8 = 5;

static LEVEL_FILTER: AtomicU8 = AtomicU8::new(LEVEL_TRACE);

pub fn level_byte(l: &Level) -> u8 {
    if *l == Level::ERROR {
        LEVEL_ERROR
    } else if *l == Level::WARN {
        LEVEL_WARN
    } else if *l == Level::INFO {
        LEVEL_INFO
    } else if *l == Level::DEBUG {
        LEVEL_DEBUG
    } else {
        LEVEL_TRACE
    }
}

pub fn parse_level(s: &str) -> Option<u8> {
    match s.trim().to_ascii_lowercase().as_str() {
        "off" | "none" => Some(LEVEL_OFF),
        "error" | "err" => Some(LEVEL_ERROR),
        "warn" | "warning" => Some(LEVEL_WARN),
        "info" => Some(LEVEL_INFO),
        "debug" => Some(LEVEL_DEBUG),
        "trace" => Some(LEVEL_TRACE),
        _ => None,
    }
}

pub fn level_name(b: u8) -> &'static str {
    match b {
        LEVEL_OFF => "off",
        LEVEL_ERROR => "error",
        LEVEL_WARN => "warn",
        LEVEL_INFO => "info",
        LEVEL_DEBUG => "debug",
        _ => "trace",
    }
}

pub fn current_level_byte() -> u8 {
    LEVEL_FILTER.load(Ordering::Relaxed)
}

pub fn set_level_byte(b: u8) {
    LEVEL_FILTER.store(b, Ordering::Relaxed);
}

/// Set the coarse capsule-side level filter by name. The previous
/// value is preserved on a parse error.
pub fn set_level(level_str: &str) -> Result<(), String> {
    match parse_level(level_str) {
        Some(b) => {
            set_level_byte(b);
            Ok(())
        }
        None => Err(format!(
            "unknown level {:?}; expected one of off/error/warn/info/debug/trace",
            level_str
        )),
    }
}

pub fn current_level() -> &'static str {
    level_name(current_level_byte())
}

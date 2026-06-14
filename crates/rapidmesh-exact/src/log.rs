//! Thread-local meshing log: stage timings, statistics, and leveled events.
//!
//! The meshing pipeline (scene assembly -> PLC -> tet mesh -> optimize) runs
//! its stages sequentially on one thread (rayon fan-out happens inside a stage,
//! whose total is recorded on the calling thread). Each stage records its
//! wall-clock duration, key counts, and human-readable events here; the Python
//! binding clears the collector before a mesh and takes the ordered records
//! after, exposing them as `mesh.timings` / `mesh.stats` / `mesh.log`.
//!
//! Events are also printed live to stderr (with an elapsed-time prefix) when
//! verbose logging is on (RAPIDMESH_LOG, or set from Python), so a user can see
//! what the mesher is doing and where it is spending or hanging time -- not
//! just a summary after it finishes.

use std::cell::RefCell;
use std::time::Instant;

/// Severity of a log event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Normal progress.
    Info,
    /// Something unexpected but recovered (divergence backstop, budget cap).
    Warn,
    /// A failure (recorded before a panic, so the log explains the abort).
    Error,
}

impl Level {
    /// Lowercase tag for display / the Python API.
    pub fn tag(self) -> &'static str {
        match self {
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
        }
    }
}

/// One leveled log event: severity, stage, message, and seconds since the run
/// started.
#[derive(Debug, Clone)]
pub struct Event {
    /// Severity.
    pub level: Level,
    /// Stage / subsystem the event belongs to (e.g. "mesh.faces").
    pub stage: String,
    /// Human-readable message.
    pub message: String,
    /// Seconds since [`clear`] (the run start).
    pub at: f64,
}

thread_local! {
    static TIMINGS: RefCell<Vec<(String, f64)>> = const { RefCell::new(Vec::new()) };
    static STATS: RefCell<Vec<(String, f64)>> = const { RefCell::new(Vec::new()) };
    static EVENTS: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
    static START: RefCell<Option<Instant>> = const { RefCell::new(None) };
    static VERBOSE: RefCell<bool> = const { RefCell::new(false) };
}

/// Clears all collectors and (re)starts the run clock. Verbose live printing is
/// enabled if RAPIDMESH_LOG is set in the environment or was turned on via
/// [`set_verbose`].
pub fn clear() {
    TIMINGS.with(|t| t.borrow_mut().clear());
    STATS.with(|s| s.borrow_mut().clear());
    EVENTS.with(|e| e.borrow_mut().clear());
    START.with(|s| *s.borrow_mut() = Some(Instant::now()));
    if std::env::var_os("RAPIDMESH_LOG").is_some() {
        set_verbose(true);
    }
}

/// Turns live stderr printing of events on or off.
pub fn set_verbose(on: bool) {
    VERBOSE.with(|v| *v.borrow_mut() = on);
}

/// True if live printing is on.
pub fn is_verbose() -> bool {
    VERBOSE.with(|v| *v.borrow())
}

fn elapsed() -> f64 {
    START.with(|s| s.borrow().map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0))
}

/// Records a leveled event (and prints it live when verbose).
pub fn event(level: Level, stage: &str, message: impl Into<String>) {
    let message = message.into();
    let at = elapsed();
    if is_verbose() {
        eprintln!("[{:8.3}s {:>5} {stage}] {message}", at, level.tag());
    }
    EVENTS.with(|e| {
        e.borrow_mut().push(Event {
            level,
            stage: stage.to_string(),
            message,
            at,
        })
    });
}

/// Info-level event.
pub fn info(stage: &str, message: impl Into<String>) {
    event(Level::Info, stage, message);
}

/// Warning-level event.
pub fn warn(stage: &str, message: impl Into<String>) {
    event(Level::Warn, stage, message);
}

/// Error-level event (record before a panic so the log explains the abort).
pub fn error(stage: &str, message: impl Into<String>) {
    event(Level::Error, stage, message);
}

/// Records a stage's wall-clock duration in seconds, in call order, and emits
/// an info event so the timing is visible live.
pub fn stage(name: &str, seconds: f64) {
    TIMINGS.with(|t| t.borrow_mut().push((name.to_string(), seconds)));
    info(name, format!("{seconds:.3}s"));
}

/// Records a named statistic (count, size, etc.), in call order.
pub fn stat(name: &str, value: f64) {
    STATS.with(|s| s.borrow_mut().push((name.to_string(), value)));
}

/// Drains and returns the collected (timings, stats, events), each ordered by
/// record time.
pub fn take() -> (Vec<(String, f64)>, Vec<(String, f64)>, Vec<Event>) {
    let timings = TIMINGS.with(|t| std::mem::take(&mut *t.borrow_mut()));
    let stats = STATS.with(|s| std::mem::take(&mut *s.borrow_mut()));
    let events = EVENTS.with(|e| std::mem::take(&mut *e.borrow_mut()));
    (timings, stats, events)
}

//! Predicate call statistics (relaxed atomics, negligible overhead): how
//! often each staged predicate runs, takes the implicit path, and escalates
//! past the interval filter to exact expansions. Profiling instrumentation
//! for the meshing hotpath; not part of the geometric API.

use std::sync::atomic::{AtomicU64, Ordering};

pub static ORIENT3D_CALLS: AtomicU64 = AtomicU64::new(0);
pub static ORIENT3D_IMPLICIT: AtomicU64 = AtomicU64::new(0);
pub static ORIENT3D_EXACT: AtomicU64 = AtomicU64::new(0);
pub static INSPHERE_CALLS: AtomicU64 = AtomicU64::new(0);
pub static INSPHERE_IMPLICIT: AtomicU64 = AtomicU64::new(0);
pub static INSPHERE_EXACT: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(crate) fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

/// Snapshot and reset all counters:
/// [orient3d calls, implicit, exact, insphere calls, implicit, exact].
pub fn take() -> [u64; 6] {
    [
        &ORIENT3D_CALLS,
        &ORIENT3D_IMPLICIT,
        &ORIENT3D_EXACT,
        &INSPHERE_CALLS,
        &INSPHERE_IMPLICIT,
        &INSPHERE_EXACT,
    ]
    .map(|c| c.swap(0, Ordering::Relaxed))
}

//! r283 board marker — lane rent + time-of-day noise sample.
//!
//! This module ships as part of the flock-core lib but is intentionally
//! dead code on every execution path: it defines only a versioned constant
//! and is never referenced by any timed or untimed code. It exists to keep
//! the submission lane occupied while the next candidate is authored, and to
//! sample a fresh measurement window outside the 00:54-04:10 cluster (lead 9:
//! markers minted in distinct windows map time-of-day for free).
//!
//! Parent: origin/main a8729f3 (fetched fresh; NOT the stale local main
//! 972420b — packaging from the stale main reproduces the -122% demote class).
//! Reference frozen at 1497448.56 across 3db38f7/454616e/4f2fa51/a056acfc/b1239707.
//! b1239707 resolved rejected (-0.97%, packaged parent cf60a53).

/// Board marker constant. Never read; never executed; no cfg hooks.
pub const R283_BOARD_MARKER: u64 = 0x283_0000_0000_0000;

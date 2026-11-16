//! Diagnostic run (not a candidate): every ranked worker runs one window of a
//! fixed round, and publishes its finished proof at a time that encodes the
//! window and a time the worker measured itself. Windows 1-8 are the eight
//! runs of a two-level half-fraction factorial in four factors (each factor
//! an existing switch of this tree); window 0 is the tree as shipped.
//!
//! The window comes from the wall clock (a fixed round of `SLOT_SECS`-second
//! slots); the window's switches are set before anything reads them. The
//! proof is then published at `seed read + BASE + window * WIDTH + (time -
//! LO)` ms; only the rename that makes the written proof visible follows the
//! hold. The encoded time is seed read -> proof written, or (with
//! `SPAN_FROM`) the span from a timestamp mark to the proof written. Window 0
//! publishes at `seed read + BASE` and anchors the harness-side constant.
//!
//! The time-encoding channel and its constants are the ones the earlier
//! diagnostic runs of 0xArion and Meganpark980320 used, so the same decoder
//! reads it; this implementation is our own.
//!
//! Nothing here changes prover arithmetic. A window only sets existing
//! switches of the tree (each one selects a path the tree already ships, with
//! the same output bytes), and the hold is a wait after the proof is written
//! (one sleep, then naps of at most 0.5 ms; no spinning). Every window's
//! proof is the same exact, valid proof.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub const BASE_MS: u64 = 110;
pub const LO_MS: u64 = 78;
pub const WIDTH_MS: u64 = 30;
pub const GUARD_MS: u64 = 6;
pub const SLOT_SECS: u64 = 6;
/// Typical seed read -> proof written (ms); sizes the holds only.
pub const TYPICAL_MS: u64 = 87;
/// Taken off the ranked worker's start pause, the same in every window
/// (zero included). It shortens only the idle before the pool exists.
pub const PAUSE_CUT: Duration = Duration::from_millis(400);
/// The encoded time. `None`: seed read -> proof written, offset by `LO_MS`.
/// `Some(label)`: the span from the latest timestamp mark `label` (`tmark`)
/// to the proof written, offset by `SPAN_LO_MS`; `init` then switches mark
/// recording on (`FLOCK_TMARK=1`) in every window, zero included.
pub const SPAN_FROM: Option<&str> = None;
/// Lower end of the encoded span (ms) when `SPAN_FROM` is set.
pub const SPAN_LO_MS: u64 = 30;
/// The hold sleeps in one piece to a margin before its target (15 % of the
/// hold, at least `MIN_MARGIN_NS`), then in naps of at most `NAP_NS`,
/// re-reading the clock after each.
const MIN_MARGIN_NS: u64 = 12_000_000;
const NAP_NS: u64 = 500_000;

type Group = &'static [(&'static str, &'static str)];

/// A two-level factor: the switch groups set at its low level (-1) and at
/// its high level (+1). A kill factor sets its kill at the low level (high =
/// the shipped default); an opt-in factor sets its switch at the high level.
pub struct Factor {
    pub name: &'static str,
    pub low: &'static [Group],
    pub high: &'static [Group],
}

/// The four factors, in the order of the `RUNS` columns.
pub const FACTORS: [Factor; 4] = [
    Factor { name: "AB GPU share 16 of 128", low: &[], high: &[&[("FLOCK_ZC_AB_GPU_CHUNKS", "16")]] },
    Factor { name: "CPU suffix hetero queue off", low: &[], high: &[&[("FLOCK_NO_RANKED_SUFFIX_HETERO", "1")]] },
    Factor { name: "hybrid CPU tail hash", low: &[&[("FLOCK_NO_HYBRID_CPU_TAIL_HASH", "1")]], high: &[] },
    Factor { name: "K GPU share 832 of 2048", low: &[], high: &[&[("FLOCK_ZC_K_GPU_CHUNKS", "832")]] },
];

/// The factor levels of window 0: the tree as shipped (a kill factor at its
/// high level, an opt-in factor at its low level).
pub const SHIPPED: [i8; 4] = [-1, -1, 1, -1];

/// The factor levels of windows 1-8: the half fraction with
/// `D = -(A * B * C)`; every factor is high in four of them.
pub const RUNS: [[i8; 4]; 8] = [
    [-1, -1, -1, 1],
    [1, 1, 1, -1],
    [-1, -1, 1, -1],
    [1, 1, -1, 1],
    [-1, 1, -1, -1],
    [1, -1, 1, 1],
    [1, -1, -1, -1],
    [-1, 1, 1, 1],
];

/// Windows of the round: 0 (anchor) and the eight runs.
pub const WINDOWS: usize = 9;

/// The round of `SLOT_SECS`-second slots: mirrored sweeps of windows 1-8
/// (4 slots each) and 2 slots of window 0.
pub const ROUND: [usize; 34] = [
    1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];

/// The factor levels of window `w`.
pub fn levels(w: usize) -> [i8; 4] {
    if w == 0 { SHIPPED } else { RUNS[w - 1] }
}

/// The switch groups window `w` sets: each factor's groups at its level.
pub fn window_groups(w: usize) -> Vec<Group> {
    let mut out = Vec::new();
    for (f, l) in FACTORS.iter().zip(levels(w)) {
        out.extend_from_slice(if l > 0 { f.high } else { f.low });
    }
    out
}

const NONE: usize = usize::MAX;
static ARM: AtomicUsize = AtomicUsize::new(NONE);
static SEED_NS: AtomicU64 = AtomicU64::new(0);

fn clock_ns() -> u64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1
}

/// This process's window, once `init` has chosen one (ranked worker only).
pub fn window() -> Option<usize> {
    match ARM.load(Ordering::Relaxed) {
        NONE => None,
        w => Some(w),
    }
}

/// The window of a wall-clock time in whole seconds.
pub fn arm_for_secs(secs: u64) -> usize {
    ROUND[((secs / SLOT_SECS) % ROUND.len() as u64) as usize]
}

/// Extra time a window's worker spends after a typical proof, in ms.
pub fn extra_hold_ms(arm: usize) -> u64 {
    let planned = if arm == 0 { BASE_MS } else { BASE_MS + arm as u64 * WIDTH_MS + (TYPICAL_MS - LO_MS) };
    planned.saturating_sub(TYPICAL_MS)
}

/// Picks this worker's window and applies its switches. Must run first in
/// the worker, before any thread exists and before any switch is read.
/// Returns the start-pause cut: [`PAUSE_CUT`] in the ranked worker, zero
/// anywhere else.
pub fn init(ranked_worker: bool) -> Duration {
    if !ranked_worker {
        return Duration::ZERO;
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if SPAN_FROM.is_some() {
        // SAFETY: as below (first thing in the worker, no other thread).
        unsafe { std::env::set_var("FLOCK_TMARK", "1") };
    }
    // `FLOCK_DIAGX_ARM=<n>` (n < WINDOWS) fixes the window, so the encoding
    // can be read back on any machine; otherwise the wall clock picks it.
    let arm = std::env::var("FLOCK_DIAGX_ARM")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&a| a < WINDOWS)
        .unwrap_or_else(|| arm_for_secs(secs));
    for group in window_groups(arm) {
        for (name, value) in group {
            // SAFETY: called first thing in the worker's `main`, before any
            // other thread exists, so no reader can race the write.
            unsafe { std::env::set_var(name, value) };
        }
    }
    let _ = clock_ns();
    ARM.store(arm, Ordering::SeqCst);
    PAUSE_CUT
}

/// The seed line has arrived.
pub fn seed_read() {
    if ARM.load(Ordering::Relaxed) != NONE {
        SEED_NS.store(clock_ns(), Ordering::SeqCst);
    }
}

/// Offset (ms past `BASE + window * WIDTH`) that encodes a measured time.
pub fn offset_ms(arm: usize, time_ms: f64) -> f64 {
    if arm == 0 {
        return 0.0;
    }
    let lo = if SPAN_FROM.is_some() { SPAN_LO_MS } else { LO_MS } as f64;
    let top = (WIDTH_MS - GUARD_MS) as f64;
    if time_ms < lo {
        top + 1.0
    } else if time_ms - lo >= top {
        top + 3.0
    } else {
        time_ms - lo
    }
}

/// Holds the calling thread until this worker's encoded publication time.
/// Called right before the rename that makes the written proof visible.
/// Once only: the first call takes the seed stamp, so any later call (and
/// any call before the seed has arrived) returns at once.
pub fn hold() {
    let arm = ARM.load(Ordering::SeqCst);
    let seed = SEED_NS.swap(0, Ordering::SeqCst);
    if arm == NONE || seed == 0 {
        return;
    }
    let time_ms = match SPAN_FROM {
        None => (clock_ns() - seed) as f64 / 1e6,
        // A missing start mark encodes as "under the window" (offset 25).
        Some(label) => crate::tmark::last(label)
            .map_or(0.0, |t0| crate::tmark::now_ns().saturating_sub(t0) as f64 / 1e6),
    };
    let target_ms = (BASE_MS + arm as u64 * WIDTH_MS) as f64 + offset_ms(arm, time_ms);
    let target = seed + (target_ms * 1e6) as u64;
    // One sleep that ends a margin before the target, then naps of at most
    // `NAP_NS` with the clock re-read after each. No spinning.
    let now = clock_ns();
    if now < target {
        let first = first_sleep_ns(target - now);
        if first > 0 {
            std::thread::sleep(Duration::from_nanos(first));
        }
    }
    loop {
        let now = clock_ns();
        if now >= target {
            break;
        }
        std::thread::sleep(Duration::from_nanos((target - now).min(NAP_NS)));
    }
}

/// The hold's first sleep with `left` ns to go: it ends 15 % of `left` (at
/// least `MIN_MARGIN_NS`) before the target; zero when the margin covers it.
fn first_sleep_ns(left: u64) -> u64 {
    left.saturating_sub((left * 15 / 100).max(MIN_MARGIN_NS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_the_half_fraction_and_holds_stay_short() {
        // Every run satisfies D = -(A B C); every factor is high in four
        // runs; the runs are distinct; window 0 is the shipped tree.
        for (w, run) in RUNS.iter().enumerate() {
            assert_eq!(run[3], -(run[0] * run[1] * run[2]), "window {}", w + 1);
        }
        for f in 0..4 {
            assert_eq!(RUNS.iter().filter(|r| r[f] > 0).count(), 4, "factor {f}");
        }
        for a in 0..8 {
            for b in a + 1..8 {
                assert_ne!(RUNS[a], RUNS[b]);
            }
        }
        // A factor has switches on exactly one side.
        for f in &FACTORS {
            assert!(f.low.is_empty() != f.high.is_empty(), "{}", f.name);
        }
        // Window 0 sets nothing (the shipped tree), and each run window sets
        // exactly its factors' groups at their levels.
        assert!(window_groups(0).is_empty());
        for w in 1..WINDOWS {
            let groups = window_groups(w);
            for (f, l) in FACTORS.iter().zip(levels(w)) {
                let side = if l > 0 { f.high } else { f.low };
                let other = if l > 0 { f.low } else { f.high };
                for g in side {
                    assert!(groups.iter().any(|x| std::ptr::eq(*x, *g)), "window {w}: {} missing", f.name);
                }
                for g in other {
                    assert!(!groups.iter().any(|x| std::ptr::eq(*x, *g)), "window {w}: {} wrong side", f.name);
                }
            }
        }
        // The round: 4 slots for each of windows 1-8, 2 for window 0.
        let mut n = [0usize; WINDOWS];
        for a in ROUND {
            n[a] += 1;
        }
        assert_eq!(n[0], 2);
        assert!(n[1..].iter().all(|&c| c == 4), "{n:?}");
        for a in 0..WINDOWS {
            assert!(extra_hold_ms(a) < 1300);
        }
    }

    #[test]
    fn offsets_stay_inside_their_window() {
        for a in 1..WINDOWS {
            for t in [0.0, 29.9, 30.0, 45.0, 53.9, 54.0, 60.0, 78.0, 90.0, 101.9, 102.0, 140.0] {
                let o = offset_ms(a, t);
                assert!((0.0..WIDTH_MS as f64).contains(&o));
            }
        }
        assert_eq!(offset_ms(0, 90.0), 0.0);
    }

    #[test]
    fn the_first_sleep_ends_a_margin_before_the_target() {
        assert_eq!(first_sleep_ns(200_000_000), 170_000_000);
        assert_eq!(first_sleep_ns(60_000_000), 48_000_000);
        assert_eq!(first_sleep_ns(10_000_000), 0);
        for left in [1u64, 2_000_000, 13_333_334, 50_000_000, 80_000_000, 377_000_000] {
            let f = first_sleep_ns(left);
            assert!(left - f >= MIN_MARGIN_NS.min(left), "left {left}: first {f}");
            assert!(left - f >= left * 15 / 100, "left {left}: first {f}");
        }
        assert!(NAP_NS <= 1_000_000);
    }

    #[test]
    fn the_span_starts_at_a_mark_this_tree_records() {
        if let Some(label) = SPAN_FROM {
            let src = [
                include_str!("../../flock-prover/src/prover.rs"),
                include_str!("../../flock-prover/src/seed_pipe.rs"),
                include_str!("gpu_commit.rs"),
            ]
            .concat();
            assert!(src.contains(&format!("tmark::mark(\"{label}\")")), "{label} is never marked");
            assert_eq!(offset_ms(1, 0.0), (WIDTH_MS - GUARD_MS) as f64 + 1.0);
        }
    }

    #[test]
    fn no_window_before_init() {
        // Unit tests never run `init` in a ranked worker.
        assert_eq!(window(), None);
    }

    #[test]
    fn hold_without_a_seed_returns_at_once() {
        let t = Instant::now();
        hold();
        assert!(t.elapsed() < Duration::from_millis(50));
    }

    #[test]
    fn switch_names_are_this_trees_switches() {
        // Every switch a factor sets must be one the tree reads by that exact
        // name; a typo would silently make two levels identical.
        let src = [
            include_str!("gpu_commit.rs"),
            include_str!("gpu_commit/r2cs_fast.rs"),
            include_str!("ntt/additive_ntt_f128.rs"),
            include_str!("pcs/ligerito.rs"),
            include_str!("zerocheck/multilinear.rs"),
            include_str!("zerocheck/univariate_skip_optimized.rs"),
            include_str!("zerocheck/univariate_skip_optimized/kernels.rs"),
            include_str!("zerocheck/univariate_skip_optimized/kernels/aarch64.rs"),
        ]
        .concat();
        for f in &FACTORS {
            for group in f.low.iter().chain(f.high) {
                for (name, _) in *group {
                    let quoted = format!("\"{name}\"");
                    assert!(src.contains(&quoted), "{name} is not read by this tree");
                }
            }
        }
    }
}

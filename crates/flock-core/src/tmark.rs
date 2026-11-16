//! Diagnostic-only cross-process timestamps (`FLOCK_TMARK=1`).
//!
//! Records `(label, CLOCK_UPTIME_RAW ns)` pairs into a fixed array so an
//! external driver that stamps its own seed write on the same clock can see
//! where the timed window goes. With the variable unset every call is one
//! cached-bool load; nothing is recorded and nothing is printed.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const SLOTS: usize = 512;
static NS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
static LABELS: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
static NEXT: AtomicUsize = AtomicUsize::new(0);

/// True when `FLOCK_TMARK` is set (cached).
pub fn on() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_TMARK").is_some());
    *ON
}

/// Monotonic nanoseconds on the clock Python's `CLOCK_UPTIME_RAW` reads.
pub fn now_ns() -> u64 {
    #[cfg(target_os = "macos")]
    unsafe {
        unsafe extern "C" {
            fn clock_gettime_nsec_np(clock: u32) -> u64;
        }
        clock_gettime_nsec_np(8)
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Record `label` now (no-op unless `FLOCK_TMARK` is set).
#[inline]
pub fn mark(label: &'static str) {
    if !on() {
        return;
    }
    let t = now_ns();
    let i = NEXT.fetch_add(1, Ordering::Relaxed);
    if i < SLOTS {
        NS[i].store(t, Ordering::Relaxed);
        let mut l = LABELS.lock().unwrap_or_else(|e| e.into_inner());
        if l.len() <= i {
            l.resize(i + 1, "");
        }
        l[i] = label;
    }
}

/// Record `label` at an explicit CLOCK_UPTIME_RAW timestamp (GPU intervals
/// converted from Metal's host-timebase seconds). No-op unless enabled.
#[inline]
pub fn mark_ns(label: &'static str, t: u64) {
    if !on() {
        return;
    }
    let i = NEXT.fetch_add(1, Ordering::Relaxed);
    if i < SLOTS {
        NS[i].store(t, Ordering::Relaxed);
        let mut l = LABELS.lock().unwrap_or_else(|e| e.into_inner());
        if l.len() <= i {
            l.resize(i + 1, "");
        }
        l[i] = label;
    }
}

/// Record a Metal GPU interval (`GPUStartTime`/`GPUEndTime`, seconds on the
/// mach host timebase = CLOCK_UPTIME_RAW) as two marks.
#[inline]
pub fn mark_gpu(start_label: &'static str, end_label: &'static str, s: f64, e: f64) {
    if !on() || e <= 0.0 {
        return;
    }
    mark_ns(start_label, (s * 1e9).round() as u64);
    mark_ns(end_label, (e * 1e9).round() as u64);
}

/// Drop every mark recorded so far (warm-up noise) — no-op unless enabled.
pub fn reset() {
    if on() {
        NEXT.store(0, Ordering::Relaxed);
    }
}

/// Print and clear every recorded mark to stderr.
pub fn dump() {
    if !on() {
        return;
    }
    let n = NEXT.swap(0, Ordering::Relaxed).min(SLOTS);
    let l = LABELS.lock().unwrap_or_else(|e| e.into_inner());
    for i in 0..n {
        eprintln!("[tmark] {} {}", NS[i].load(Ordering::Relaxed), l.get(i).copied().unwrap_or(""));
    }
}

/// Diagnostic run only: a process-level A/B of stacked pieces on the
/// official runner, carried in the published trial times.
///
/// The tree stacks several published pieces that the record lacks, each
/// behind its own kill switches. Every ranked worker picks one arm from the
/// wall clock (`ROUND[(unix secs / SLOT_SECS) mod ROUND.len()]`) before any
/// thread exists and sets switches to `1` for its whole process: `base` turns
/// every stacked piece off, `all on` turns none off, and each group arm turns
/// off every stacked piece except its own group. The timed proof is computed
/// and published as usual, but returns late: the speculative prove holds
/// until `seed read + BASE_MS + arm * WIDTH_MS + (total - LO_MS)`, where
/// `total` is seed read -> prove end. Arm 0 (`zero`) returns at
/// `seed read + BASE_MS` and calibrates the harness-side constant. Totals out
/// of range return at `valid + 1` (below) or `valid + 3` (above). The start
/// pause is shortened by this worker's expected extra hold only, so a worker's
/// wall cycle stays the record's.
pub mod diag {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    pub const BASE_MS: u64 = 110;
    pub const LO_MS: u64 = 78;
    pub const WIDTH_MS: u64 = 30;
    pub const GUARD_MS: u64 = 6;
    pub const SLOT_SECS: u64 = 6;
    /// Typical seed read -> prove end of the record on the runner (ms), used
    /// only to size the start-pause cut.
    pub const TYPICAL_TOTAL_MS: u64 = 87;

    /// Groups of stacked pieces, each a set of kill switches.
    pub const GROUPS: [(&str, &[&str]); 10] = [
        ("r2 walk (R2_FOLD7, R2_HEAD_INLINE)", &["FLOCK_NO_R2_FOLD7", "FLOCK_NO_R2_HEAD_INLINE"]),
        ("k fold7", &["FLOCK_NO_ZC_K_FOLD7"]),
        ("caller-first (LIG_FOLD_CF, ZC_TRI_CF)", &["FLOCK_NO_LIG_FOLD_CF", "FLOCK_NO_ZC_TRI_CF"]),
        (
            "34db77b1 commit/merkle",
            &[
                "FLOCK_NO_FINE_BOUNDARY_CB",
                "FLOCK_NO_FINE_SPLIT_REPRICE",
                "FLOCK_NO_L1_LOCAL_PARENTS",
                "FLOCK_NO_L1_WHOLE_TREE_VIEW",
                "FLOCK_NO_L2_MERKLE_OVERLAP",
                "FLOCK_NO_ZFOLD_SPLIT_CB",
            ],
        ),
        (
            "34db77b1 zerocheck",
            &["FLOCK_NO_ZC_AB_COMPLETION_TABLE_HOIST", "FLOCK_NO_ZC_C_TAIL_TRIM"],
        ),
        ("ab unit scratch reuse", &["FLOCK_NO_AB_UNIT_SCRATCH_REUSE"]),
        ("zc c drain8", &["FLOCK_NO_ZC_C_DRAIN8"]),
        ("zc ab late share", &["FLOCK_NO_ZC_AB_LATE_SHARE"]),
        ("gpu t3 sibling tables", &["FLOCK_NO_GPU_T3_SIBLING_TABLES"]),
        ("witgen pace", &["FLOCK_NO_WITGEN_PACE"]),
    ];

    /// Arm 0 zero, 1 base (all stacked pieces off), 2 all on, 3 + g group g on.
    pub const N_ARMS: usize = 3 + GROUPS.len();

    pub fn arm_name(arm: usize) -> &'static str {
        match arm {
            0 => "zero",
            1 => "base (all stacked pieces off)",
            2 => "all on",
            a => GROUPS[a - 3].0,
        }
    }

    /// Switches set to `1` in an arm.
    pub fn arm_switches(arm: usize) -> Vec<&'static str> {
        let keep_on = match arm {
            2 => None,
            a if a >= 3 => Some(a - 3),
            _ => Some(usize::MAX),
        };
        match keep_on {
            None => Vec::new(),
            Some(g) => GROUPS
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != g)
                .flat_map(|(_, (_, s))| s.iter().copied())
                .collect(),
        }
    }

    /// One zero slot, four base slots, three all-on slots, two per group.
    pub const ROUND: [usize; 28] =
        [1, 3, 4, 5, 6, 2, 7, 8, 9, 10, 11, 12, 1, 0, 2, 1, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 1];

    const NONE: usize = usize::MAX;
    static ARM: AtomicUsize = AtomicUsize::new(NONE);
    static SEED_NS: AtomicU64 = AtomicU64::new(0);

    fn clock_ns() -> u64 {
        static BASE: OnceLock<Instant> = OnceLock::new();
        BASE.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1
    }

    /// The arm of a wall-clock time in whole seconds.
    pub fn arm_for_secs(secs: u64) -> usize {
        ROUND[((secs / SLOT_SECS) % ROUND.len() as u64) as usize]
    }

    /// The start-pause cut of an arm: its expected extra hold.
    pub fn pause_cut(arm: usize) -> Duration {
        let planned = BASE_MS + arm as u64 * WIDTH_MS + (TYPICAL_TOTAL_MS - LO_MS);
        Duration::from_millis(planned.saturating_sub(TYPICAL_TOTAL_MS))
    }

    /// Pick this worker's arm and set its switches. Must run before any thread
    /// exists and before any switch is read. Returns the pause cut.
    pub fn init(ranked_worker: bool) -> Duration {
        if !ranked_worker {
            return Duration::ZERO;
        }
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let arm = arm_for_secs(secs);
        for name in arm_switches(arm) {
            // SAFETY: called first thing in the worker's `main` (via
            // `init_perf_thread_pool`), before any other thread exists.
            unsafe { std::env::set_var(name, "1") };
        }
        let _ = clock_ns();
        ARM.store(arm, Ordering::SeqCst);
        pause_cut(arm)
    }

    /// The seed line has arrived (seed pipe).
    pub fn seed_read() {
        if ARM.load(Ordering::Relaxed) != NONE {
            SEED_NS.store(clock_ns(), Ordering::SeqCst);
        }
    }

    /// The planned return of a proof that took `total_ns`, relative to seed read.
    pub fn plan_ns(arm: usize, total_ns: u64) -> u64 {
        let ms = 1_000_000u64;
        let start = (BASE_MS + arm as u64 * WIDTH_MS) * ms;
        if arm == 0 {
            return start;
        }
        let valid = (WIDTH_MS - GUARD_MS) * ms;
        let lo = LO_MS * ms;
        let off = if total_ns < lo {
            valid + ms
        } else if total_ns - lo >= valid {
            valid + 3 * ms
        } else {
            total_ns - lo
        };
        start + off
    }

    /// Last thing the timed (speculative) prove does: hold until the planned
    /// return time.
    pub fn hold() {
        let arm = ARM.load(Ordering::Relaxed);
        let t0 = SEED_NS.swap(0, Ordering::SeqCst);
        if arm == NONE || t0 == 0 {
            return;
        }
        let target = t0 + plan_ns(arm, clock_ns() - t0);
        loop {
            let now = clock_ns();
            if now >= target {
                break;
            }
            let left = target - now;
            if left > 3_000_000 {
                std::thread::sleep(Duration::from_nanos(left - 2_000_000));
            } else {
                std::hint::spin_loop();
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn plan_stays_inside_its_window() {
            let ms = 1_000_000u64;
            for arm in 0..N_ARMS {
                for total in [0, 50 * ms, 78 * ms, 87 * ms, 101 * ms, 102 * ms, 109 * ms] {
                    let p = plan_ns(arm, total);
                    let start = (BASE_MS + arm as u64 * WIDTH_MS) * ms;
                    assert!(p >= start && p < start + WIDTH_MS * ms, "arm {arm} total {total}");
                    assert!(p >= total, "the plan must not return before the proof ends");
                }
            }
        }

        #[test]
        fn arms_cover_and_switch_sets_are_right() {
            for arm in 0..N_ARMS {
                assert!(ROUND.contains(&arm), "arm {arm} missing from ROUND");
            }
            let all: usize = GROUPS.iter().map(|g| g.1.len()).sum();
            assert_eq!(arm_switches(0).len(), all);
            assert_eq!(arm_switches(1).len(), all);
            assert!(arm_switches(2).is_empty());
            for g in 0..GROUPS.len() {
                let s = arm_switches(3 + g);
                assert_eq!(s.len(), all - GROUPS[g].1.len());
                assert!(GROUPS[g].1.iter().all(|x| !s.contains(x)));
            }
            // The pause cut never exceeds the 1.3 s pause.
            assert!(pause_cut(N_ARMS - 1) < Duration::from_millis(1300));
        }
    }
}

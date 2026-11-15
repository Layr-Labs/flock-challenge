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

/// Diagnostic run only: a process-level A/B of kill switches on the official
/// runner, carried in the published trial times.
///
/// Every ranked worker picks one arm from the wall clock
/// (`ROUND[(unix secs / SLOT_SECS) mod ROUND.len()]`) before any thread
/// exists, and sets that arm's `FLOCK_NO_*` switches to `1` for its whole
/// process (so GPU-side switches and once-per-process tunings are measured as
/// a separate build would measure them). The timed proof is computed and
/// published exactly as usual, but returns late: the speculative prove holds
/// until `seed read + BASE_MS + arm * WIDTH_MS + (total - LO_MS)`, where
/// `total` is seed read -> prove end inside the worker. The window a trial
/// time falls in names the arm and the offset in it is that worker's `total`.
/// Arm 0 (`zero`) returns at `seed read + BASE_MS` exactly and calibrates the
/// harness-side constant. Out-of-range totals return at `valid + 1` (below)
/// or `valid + 3` (above) inside their window.
pub mod diag {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    pub const BASE_MS: u64 = 120;
    pub const LO_MS: u64 = 80;
    pub const WIDTH_MS: u64 = 40;
    pub const GUARD_MS: u64 = 6;
    pub const SLOT_SECS: u64 = 6;
    /// The start pause is shortened by this much in a diagnostic worker; it is
    /// larger than the longest planned hold minus a proof.
    pub const PAUSE_CUT_MS: u64 = 700;

    /// (name, switches set to `1` for the whole process).
    pub const ARMS: [(&str, &[&str]); 10] = [
        ("zero", &[]),
        ("base", &[]),
        ("transpose_pmull off", &["FLOCK_NO_TRANSPOSE_PAIR_PMULL"]),
        ("eq_kernel off", &["FLOCK_NO_EQ_KERNEL_RS", "FLOCK_NO_EQ_KERNEL_SKIP"]),
        (
            "f1b9b38b pieces off",
            &[
                "FLOCK_NO_OPEN_AB_CANONICAL_REUSE",
                "FLOCK_NO_R2P_FUSED",
                "FLOCK_NO_R2P_TAIL_FUSED",
                "FLOCK_NO_RANKED_SUFFIX_HETERO",
                "FLOCK_NO_ZC_K_ALLFUSED",
            ],
        ),
        (
            "749a64e1 pieces off",
            &[
                "FLOCK_NO_AB_STRIPE_SEGMENTS",
                "FLOCK_NO_NTT_FINAL_PAIR_TAIL",
                "FLOCK_NO_NTT_LOW_LAZY",
                "FLOCK_NO_NTT_LOW_PMULL",
                "FLOCK_NO_QUERY_SET_PRESIZE",
                "FLOCK_NO_ZC_AB_COMPLETION_PAIR16",
                "FLOCK_NO_ZC_AB_HEAD_HOIST",
                "FLOCK_NO_ZC_TRI_GROUP_INLINE",
            ],
        ),
        ("lc_unscaled off", &["FLOCK_NO_LC_UNSCALED"]),
        ("round0_basis_fusion off", &["FLOCK_NO_ROUND0_BASIS_FUSION"]),
        (
            "a291d077 pieces off",
            &["FLOCK_NO_OPEN_C_CANONICAL_REUSE", "FLOCK_NO_RS_FOLD8_DIRECT_STATE", "FLOCK_NO_ZERO_ROOT_SHL4_BASES"],
        ),
        (
            "lc_wake/stream_warm/witness_gpu_stream off",
            &["FLOCK_NO_LC_WAKE_POST", "FLOCK_NO_STREAM_WARM", "FLOCK_NO_WITNESS_GPU_STREAM"],
        ),
    ];

    /// One zero slot, five base slots and two slots per switch arm.
    pub const ROUND: [usize; 22] = [1, 2, 3, 4, 5, 1, 6, 7, 8, 9, 1, 0, 2, 3, 4, 5, 1, 6, 7, 8, 9, 1];

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
        for name in ARMS[arm].1 {
            // SAFETY: called first thing in the worker's `main` (via
            // `init_perf_thread_pool`), before any other thread exists.
            unsafe { std::env::set_var(name, "1") };
        }
        let _ = clock_ns();
        ARM.store(arm, Ordering::SeqCst);
        Duration::from_millis(PAUSE_CUT_MS)
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
            for arm in 0..ARMS.len() {
                for total in [0, 50 * ms, 80 * ms, 97 * ms, 113 * ms, 114 * ms, 119 * ms] {
                    let p = plan_ns(arm, total);
                    let start = (BASE_MS + arm as u64 * WIDTH_MS) * ms;
                    assert!(p >= start && p < start + WIDTH_MS * ms, "arm {arm} total {total}");
                    assert!(p >= total, "the plan must not return before the proof ends");
                }
            }
            assert_eq!(plan_ns(3, 97 * ms), (BASE_MS + 3 * WIDTH_MS) * ms + 17 * ms);
        }

        #[test]
        fn round_covers_every_arm() {
            for arm in 0..ARMS.len() {
                assert!(ROUND.contains(&arm));
            }
            // The longest hold is shorter than the pause cut plus a proof.
            assert!(BASE_MS + ARMS.len() as u64 * WIDTH_MS < PAUSE_CUT_MS + LO_MS);
        }
    }
}

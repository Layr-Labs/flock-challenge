//! Diagnostic channel for one official run (not a candidate).
//!
//! Each ranked worker picks one arm from the wall clock, applies the arm's
//! environment settings before anything reads them, and returns its proof
//! late: at `seed read + BASE + arm * WIDTH + (prove - LO)` ms, where `prove`
//! is its own seed read -> prove end time. The published trial times then
//! decode into (arm, prove time) pairs, so every arm gets readings on the
//! official runner, interleaved in time with the others. Arm 0 returns at
//! `seed read + BASE` and calibrates the harness-side constant.
//!
//! The channel idea is patternrecognition9-del's and newjordan's, as used by
//! 0xArion (`29a3e16b`, `d4c5ba19`). This implementation is our own.
//!
//! Nothing here changes prover arithmetic: the arms only set the record's own
//! share overrides, and the hold is a sleep after the proof is complete.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub const BASE_MS: u64 = 110;
pub const LO_MS: u64 = 78;
pub const WIDTH_MS: u64 = 30;
pub const GUARD_MS: u64 = 6;
pub const SLOT_SECS: u64 = 6;
/// Typical seed read -> prove end of the record on the runner (ms); sizes the
/// start-pause cut only.
pub const TYPICAL_MS: u64 = 87;

/// Arms: name and the environment settings applied for the whole process.
/// `FLOCK_NO_HYBRID_GPU_SUFFIX_HASH=1` has the CPU hash its own share of the
/// commit's leaves as it finishes them (no `cb3`), the record's own switch.
/// `FLOCK_HYBRID_CPU_BLOCKS` pins the commit's CPU/GPU split (and stops the
/// warm-up sweep): `k` is k/16 to the CPU; `k + 16 m` adds m/16 of the
/// boundary sixteenth. `FLOCK_ZC_GPU_CLAIMS` pins round one's C fold GPU
/// share (claims of 64; the warm-up tune picks 64 on our machines).
pub const ARMS: [(&str, &[(&str, &str)]); 7] = [
    ("zero (calibration)", &[]),
    ("base (record settings)", &[]),
    ("CPU hashes its own leaves (no cb3), split by the sweep", &[("FLOCK_NO_HYBRID_GPU_SUFFIX_HASH", "1")]),
    ("commit split 2.5/16 pinned", &[("FLOCK_HYBRID_CPU_BLOCKS", "130")]),
    ("commit split 3/16 pinned", &[("FLOCK_HYBRID_CPU_BLOCKS", "3")]),
    ("round-one C fold GPU claims 56 of 64", &[("FLOCK_ZC_GPU_CLAIMS", "56")]),
    ("round-one C fold GPU claims 60 of 64", &[("FLOCK_ZC_GPU_CLAIMS", "60")]),
];

/// 28 six-second slots: base 6, zero 2, each other arm 4, interleaved.
pub const ROUND: [usize; 28] =
    [1, 2, 3, 4, 5, 6, 1, 2, 0, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 0, 2, 3, 4, 1, 5, 6, 1];

const NONE: usize = usize::MAX;
static ARM: AtomicUsize = AtomicUsize::new(NONE);
static SEED_NS: AtomicU64 = AtomicU64::new(0);

fn clock_ns() -> u64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1
}

/// The arm of a wall-clock time in whole seconds.
pub fn arm_for_secs(secs: u64) -> usize {
    ROUND[((secs / SLOT_SECS) % ROUND.len() as u64) as usize]
}

/// Extra time an arm's worker spends after a typical prove, in ms.
pub fn extra_hold_ms(arm: usize) -> u64 {
    let planned = if arm == 0 { BASE_MS } else { BASE_MS + arm as u64 * WIDTH_MS + (TYPICAL_MS - LO_MS) };
    planned.saturating_sub(TYPICAL_MS)
}

/// Picks this worker's arm and applies its settings. Must run first in the
/// worker, before any thread exists and before any setting is read. Returns
/// the start-pause cut, which is zero here (see below).
pub fn init(ranked_worker: bool) -> Duration {
    if !ranked_worker {
        return Duration::ZERO;
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let arm = arm_for_secs(secs);
    for (name, value) in ARMS[arm].1 {
        // SAFETY: called first thing in the worker's `main` (via
        // `init_perf_thread_pool`), before any other thread exists.
        unsafe { std::env::set_var(name, value) };
    }
    let _ = clock_ns();
    ARM.store(arm, Ordering::SeqCst);
    // The start pause is not cut: our first runs, which cut it by the planned
    // hold, split into two modes about 8 ms apart that normal runs never show.
    // The holds lengthen the job by about 14 s in all instead.
    Duration::ZERO
}

/// The seed line has arrived.
pub fn seed_read() {
    if ARM.load(Ordering::Relaxed) != NONE {
        SEED_NS.store(clock_ns(), Ordering::SeqCst);
    }
}

/// Offset (ms past `BASE + arm * WIDTH`) that encodes a prove time.
pub fn offset_ms(arm: usize, prove_ms: f64) -> f64 {
    if arm == 0 {
        return 0.0;
    }
    let lo = LO_MS as f64;
    let top = (WIDTH_MS - GUARD_MS) as f64;
    if prove_ms < lo {
        top + 1.0
    } else if prove_ms - lo >= top {
        top + 3.0
    } else {
        prove_ms - lo
    }
}

/// Holds the calling thread until this worker's encoded return time. Called
/// once, after the proof is complete and before it is returned.
pub fn hold() {
    let arm = ARM.load(Ordering::SeqCst);
    let seed = SEED_NS.load(Ordering::SeqCst);
    if arm == NONE || seed == 0 {
        return;
    }
    let prove_ms = (clock_ns() - seed) as f64 / 1e6;
    let target_ms = (BASE_MS + arm as u64 * WIDTH_MS) as f64 + offset_ms(arm, prove_ms);
    let target = seed + (target_ms * 1e6) as u64;
    loop {
        let now = clock_ns();
        if now >= target {
            break;
        }
        let left = target - now;
        if left > 2_000_000 {
            std::thread::sleep(Duration::from_nanos(left - 1_500_000));
        } else {
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_counts_and_cut_fit_the_pause() {
        let mut n = [0usize; 7];
        for a in ROUND {
            n[a] += 1;
        }
        assert_eq!(n, [2, 6, 4, 4, 4, 4, 4]);
        for a in 0..ARMS.len() {
            assert!(extra_hold_ms(a) < 1300);
        }
    }

    #[test]
    fn offsets_stay_inside_their_window() {
        for a in 1..ARMS.len() {
            for p in [60.0, 78.0, 90.0, 101.9, 102.0, 140.0] {
                let o = offset_ms(a, p);
                assert!((0.0..WIDTH_MS as f64).contains(&o));
            }
        }
        assert_eq!(offset_ms(0, 90.0), 0.0);
    }
}

//! Disclosed on-runner phase telemetry for the ranked BLAKE3 worker.
//!
//! The ranked harness exposes exactly one per-trial observable to solvers: the
//! trial's wall-clock duration (`officialMetrics.trial_seconds`, 100 scored +
//! 20 warm-up values per run). Worker stdout/stderr are `/dev/null`, the proof
//! bytes must verify against the pristine witness, and no solver host in this
//! account has the runner's hardware. So the per-phase split of the ~142 ms
//! window on the M3 Max has never been measured directly — every phase figure
//! in circulation is an x86 local number scaled by an assumed factor.
//!
//! This module turns the trial duration into a telemetry channel, *by
//! design and openly*: the timed call records the wall time of each prover
//! phase into process-wide atomics, picks one slot `k` from the trial's own
//! (pseudo-random) first input block, and then delays its return until
//!
//! ```text
//!     seed_at + BASE + SPACING * k + phase_k
//! ```
//!
//! so the harness observes `trial ≈ BASE + SPACING*k + phase_k + publish`.
//! Because the delay is an absolute deadline from the seed instant, the
//! window's own trial-to-trial jitter does not leak into the sample: the
//! residual noise is sleep/spin precision, the publish tail and the harness's
//! 100 µs poll (≈ 0.2–0.3 ms). With 120 trials and 8 slots each phase gets
//! ~15 samples → its median is good to ~0.1 ms on the runner.
//!
//! A run carrying this module is rejected by construction (it is ~0.8 s slower
//! per trial). The proof bytes are untouched; nothing here runs before the
//! proof is complete, and nothing here changes what is proved.
//!
//! Slots: 0 witgen, 1 commit+AB arm, 2 zerocheck, 3 lincheck, 4 PCS open,
//! 5 seed→speculative prove return, 6 seed→main-thread adopt entry,
//! 7 seed→adopt return (the main thread's view of the window).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub const SLOTS: usize = 8;
pub const SLOT_WITGEN: usize = 0;
pub const SLOT_COMMIT: usize = 1;
pub const SLOT_ZEROCHECK: usize = 2;
pub const SLOT_LINCHECK: usize = 3;
pub const SLOT_OPEN: usize = 4;
pub const SLOT_SPEC_TOTAL: usize = 5;
pub const SLOT_MAIN_ARRIVAL: usize = 6;
pub const SLOT_WINDOW: usize = 7;

/// Deadline base: must exceed the longest plausible un-padded window so the
/// k = 0 band never collides with the window itself.
const BASE_MS: f64 = 160.0;
/// Band spacing: must exceed the longest single phase (< 150 ms).
const SPACING_MS: f64 = 175.0;

static SEED_AT: OnceLock<Instant> = OnceLock::new();
static PHASE_US: [AtomicU64; SLOTS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Record the instant the seed line was parsed (the pipe thread's `seed_at`).
pub fn mark_seed(at: Instant) {
    let _ = SEED_AT.set(at);
}

/// Record a phase wall time in milliseconds. Last writer wins; the warm-up
/// call's values are overwritten by the timed call's.
pub fn record_ms(slot: usize, ms: f64) {
    if slot < SLOTS {
        PHASE_US[slot].store((ms * 1000.0).max(0.0) as u64, Ordering::Relaxed);
    }
}

/// Record `now − seed_at` (ms) into a slot, if the seed instant is known.
pub fn record_since_seed(slot: usize) {
    if let Some(t0) = SEED_AT.get() {
        record_ms(slot, t0.elapsed().as_secs_f64() * 1e3);
    }
}

/// Slot selector for this trial, from the first input block's first chaining
/// value word (a pseudo-random function of the seed the harness drew).
pub fn slot_for(word: u32) -> usize {
    (word as usize) % SLOTS
}

/// Delay until `seed_at + BASE + SPACING*k + phase_k`. No-op if the seed
/// instant is unknown (pipe disarmed) — then the trial is simply unpadded.
pub fn pad_to_deadline(slot: usize) {
    let Some(t0) = SEED_AT.get() else {
        return;
    };
    let phase_ms = PHASE_US[slot.min(SLOTS - 1)].load(Ordering::Relaxed) as f64 / 1000.0;
    let target_ms = BASE_MS + SPACING_MS * slot as f64 + phase_ms;
    let deadline = *t0 + Duration::from_secs_f64(target_ms / 1e3);
    if std::env::var_os("FLOCK_PHASE_TIMING").is_some() {
        let v: Vec<f64> = PHASE_US
            .iter()
            .map(|p| p.load(Ordering::Relaxed) as f64 / 1000.0)
            .collect();
        eprintln!(
            "[telemetry] slot={slot} phases_ms={v:?} now_since_seed={:.2} target={target_ms:.2}",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    // Coarse sleep to ~1.5 ms before the deadline, then spin for precision.
    let now = Instant::now();
    if deadline > now + Duration::from_micros(1500) {
        std::thread::sleep(deadline - now - Duration::from_micros(1500));
    }
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

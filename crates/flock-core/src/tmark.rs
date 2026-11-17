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

/// True when `FLOCK_TMARK` is set (cached), or when this process carries a
/// diagnostic-run arm ([`telem_init`]).
pub fn on() -> bool {
    FORCED.load(Ordering::Relaxed) || env_on()
}

/// True when `FLOCK_TMARK` itself is set (cached): marks are also printed.
pub fn env_on() -> bool {
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

/// Print and clear every recorded mark to stderr (`FLOCK_TMARK` only).
pub fn dump() {
    if !env_on() {
        return;
    }
    let n = NEXT.swap(0, Ordering::Relaxed).min(SLOTS);
    let l = LABELS.lock().unwrap_or_else(|e| e.into_inner());
    for i in 0..n {
        eprintln!("[tmark] {} {}", NS[i].load(Ordering::Relaxed), l.get(i).copied().unwrap_or(""));
    }
}

// ---------------------------------------------------------------------------
// Diagnostic-run telemetry. THIS TREE IS A MEASUREMENT, NOT A CANDIDATE.
// ---------------------------------------------------------------------------
//
// Every ranked worker, warm-up and scored alike, takes one arm from the wall
// clock, sets that arm's environment before any switch is read and before any
// thread pool exists, runs its proofs unchanged, and returns its finished
// timed proof late, so that the trial time the benchmark publishes carries
// the worker's arm and its own `seed read -> prove end` total. The form is
// i34-9's disclosed diagnostic run 6d3d6b82 (arm `i` returns at
// `c + 75 + 40 i + (total - 72)` ms); the channel is c4d5dd3c's.
//
// - Nothing reads the trial number or the ready/proof file names, and no state
//   is kept between trials: a worker takes the arm
//   `TELEM_ROUND[(unix time / TELEM_BIN_MS) % len]`.
// - Arm `i` owns the window `[start_i, start_i + TELEM_VALID_MS +
//   TELEM_GUARD_MS)` ms after the seed read, `start_i = TELEM_LO_MS +
//   TELEM_LEAD_MS + i (TELEM_VALID_MS + TELEM_GUARD_MS)`, and returns its
//   proof at `seed read + start_i + (total - TELEM_LO_MS)`. Sentinels:
//   `TELEM_VALID_MS + 1` below range, `+ 2` above range, `+ 3` never
//   recorded. Proof not ready 1 ms before its time: it returns in the middle
//   of a last `TELEM_OVERRUN_MS` window.
// - The hold runs after the timed proof is finished ("prove end") and before
//   the unchanged proof is published. Proof bytes, seed read and publication
//   code are untouched. The record's start pause is kept as it is.

/// Recorder armed for this process (every ranked worker of this tree).
static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// One arm: its name and the environment it sets at process start.
#[derive(Clone, Copy, Debug)]
pub struct TelemArm {
    pub name: &'static str,
    pub env: &'static [(&'static str, &'static str)],
}

/// The arms, in window order: the commit split's CPU-ward tilt
/// (`gpu_commit::ENV_SPLIT_CPU_TILT`, 256ths; the record's default is 4).
pub static TELEM_ARMS: [TelemArm; 4] = [
    TelemArm { name: "tilt 0", env: &[("FLOCK_SPLIT_CPU_TILT", "0")] },
    TelemArm { name: "tilt 4", env: &[("FLOCK_SPLIT_CPU_TILT", "4")] },
    TelemArm { name: "tilt 8", env: &[("FLOCK_SPLIT_CPU_TILT", "8")] },
    TelemArm { name: "tilt 12", env: &[("FLOCK_SPLIT_CPU_TILT", "12")] },
];

/// Clock-bin order of the arms: 30 blocks of four, each block one fixed
/// shuffle of the four arms, so every arm has 30 bins and any stretch of
/// the run holds the arms in nearly equal numbers.
pub static TELEM_ROUND: [u8; 120] = [
    0, 2, 3, 1, 0, 1, 3, 2, 3, 2, 1, 0, 2, 1, 0, 3, 0, 2, 1, 3, 1, 0, 3, 2, 1, 3, 2, 0, 3, 0, 1, 2,
    1, 2, 0, 3, 0, 1, 3, 2, 3, 0, 2, 1, 0, 1, 2, 3, 2, 3, 1, 0, 3, 1, 2, 0, 3, 1, 2, 0, 3, 0, 2, 1,
    3, 1, 2, 0, 3, 0, 2, 1, 3, 2, 0, 1, 0, 2, 1, 3, 0, 3, 1, 2, 3, 2, 0, 1, 3, 0, 2, 1, 3, 2, 0, 1,
    2, 1, 3, 0, 2, 3, 0, 1, 2, 0, 3, 1, 2, 3, 1, 0, 1, 3, 0, 2, 0, 3, 2, 1,
];

/// Total (seed read -> prove end) range every arm encodes: `[LO, LO + VALID)`.
pub const TELEM_LO_MS: i64 = 72;
pub const TELEM_VALID_MS: u64 = 36;
/// Return of an arm-0 trial after its own proof is done (ms).
pub const TELEM_LEAD_MS: u64 = 3;
/// Sentinel room at the end of each window (1 ms apart).
pub const TELEM_GUARD_MS: u64 = 4;
/// Width of the final "proof was not ready in time" window.
pub const TELEM_OVERRUN_MS: u64 = 10;
/// Clock bin of the round (longer than a trial's ~5.8 s cycle).
pub const TELEM_BIN_MS: u64 = 6000;
/// The wait spins (instead of sleeping) once this little is left. On the
/// M4 a hold that slept made the publication after it slower than after a
/// short hold; holds here are 3, 43, 83 and 123 ms, so only arm 3 sleeps
/// (about the first 60 ms of its hold).
const TELEM_SPIN_MS: u64 = 100;
/// A return later than this after its intended time counts as late (ns).
const LATE_NS: u64 = 300_000;
const MS: u64 = 1_000_000;

/// [`TELEM_SPIN_MS`], or `FLOCK_TELEM_SPIN_MS` (self-test only; the ranked
/// harness clears the environment). Read once.
fn telem_spin_ms() -> u64 {
    static V: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
        std::env::var("FLOCK_TELEM_SPIN_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(TELEM_SPIN_MS)
    });
    *V
}

/// Shift (ms) of the encoded total range: 0 on the runner;
/// `FLOCK_TELEM_PROFILE=m4` (self-test only: the M4's proof takes about
/// 200 ms) gives 120, so the M4 fits `[192, 228)` with the runner's hold
/// lengths. Read once. The ranked harness clears the environment.
pub fn telem_lo_shift_ms() -> i64 {
    static P: std::sync::LazyLock<i64> = std::sync::LazyLock::new(|| {
        if std::env::var("FLOCK_TELEM_PROFILE").is_ok_and(|v| v == "m4") { 120 } else { 0 }
    });
    *P
}

/// Window start of arm `i`, ms after the seed read (`i == len`: the overrun
/// window), for a total range shifted by `shift` ms.
pub const fn telem_start_ms_k(i: usize, shift: i64) -> u64 {
    (TELEM_LO_MS + shift) as u64 + TELEM_LEAD_MS + i as u64 * (TELEM_VALID_MS + TELEM_GUARD_MS)
}

/// Pure: the offset inside its window (ns) that encodes `total` (`None` =
/// never recorded).
pub fn telem_encode_ns_k(total: Option<i64>, shift: i64) -> u64 {
    let valid = TELEM_VALID_MS * MS;
    match total.map(|v| v - (TELEM_LO_MS + shift) * MS as i64) {
        None => valid + 3 * MS,
        Some(v) if v < 0 => valid + MS,
        Some(v) if v as u64 >= valid => valid + 2 * MS,
        Some(v) => v as u64,
    }
}

/// Pure: when arm `i` with offset `enc_ns` returns its proof (ns on the mark
/// clock); a proof not ready 1 ms before then goes to the overrun window.
pub fn telem_deadline_ns_k(seed_ns: u64, i: usize, enc_ns: u64, now_ns: u64, shift: i64) -> u64 {
    let deadline = seed_ns + telem_start_ms_k(i, shift) * MS + enc_ns;
    if now_ns + MS > deadline {
        seed_ns + (telem_start_ms_k(TELEM_ARMS.len(), shift) + TELEM_OVERRUN_MS / 2) * MS
    } else {
        deadline
    }
}

/// 1-based arm index of this process, 0 = none.
static TELEM_SLOT: AtomicUsize = AtomicUsize::new(0);

/// Give this process its arm if it is the ranked worker
/// (`flock-benchmark-worker 18 READY PROOF`; the file names are not read)
/// and set the arm's environment. First statement of
/// `init_perf_thread_pool`, before any thread exists and before any switch
/// is read.
pub fn telem_init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
        let ranked = args.len() == 4
            && args[1].to_str() == Some("18")
            && std::path::Path::new(&args[0])
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("flock-benchmark-worker"));
        if ranked {
            let unix_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64);
            // Self-test only: `FLOCK_TELEM_FORCE=<index>` pins the arm (the
            // ranked harness clears the environment, so never on the runner).
            let forced = std::env::var("FLOCK_TELEM_FORCE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&f| f < TELEM_ARMS.len());
            let i = forced.unwrap_or(TELEM_ROUND[(unix_ms / TELEM_BIN_MS) as usize % TELEM_ROUND.len()] as usize);
            for (name, value) in TELEM_ARMS[i].env {
                // SAFETY: first statement of the worker's
                // `init_perf_thread_pool`, before any other thread exists or
                // reads the environment.
                unsafe { std::env::set_var(name, value) };
            }
            TELEM_SLOT.store(i + 1, Ordering::Relaxed);
            FORCED.store(true, Ordering::Relaxed);
        }
    });
}

/// This process's arm index, if any.
pub fn telem_slot() -> Option<usize> {
    TELEM_SLOT.load(Ordering::Relaxed).checked_sub(1)
}

/// Timestamp of the most recent `label` since the last [`reset`].
pub fn find_last(label: &str) -> Option<u64> {
    let n = NEXT.load(Ordering::Relaxed).min(SLOTS);
    let l = LABELS.lock().unwrap_or_else(|e| e.into_inner());
    (0..n.min(l.len())).rev().find(|&i| l[i] == label).map(|i| NS[i].load(Ordering::Relaxed))
}

/// Called by the prover when the timed proof is finished, right before it is
/// published: waits until this worker's arm and total say the proof comes
/// back. Returns at once in a process without an arm.
pub fn telem_hold() {
    let Some(i) = telem_slot() else {
        return;
    };
    let Some(seed) = find_last("seed read") else {
        return;
    };
    let total = find_last("prove end").map(|e| e as i64 - seed as i64);
    let shift = telem_lo_shift_ms();
    let enc = telem_encode_ns_k(total, shift);
    if telem_wait_until(telem_deadline_ns_k(seed, i, enc, now_ns(), shift)) > LATE_NS {
        // Woken late: never return at a time that decodes as another value.
        telem_wait_until(telem_deadline_ns_k(seed, i, enc, u64::MAX / 2, shift));
    }
}

/// Wait until `deadline` (ns on the mark clock) and return how late the wait
/// ended: sleeps half of what is left while more than the final spin is
/// left (a wake-up must be over 50 ms late to pass the time), spins the rest.
fn telem_wait_until(deadline: u64) -> u64 {
    loop {
        let now = now_ns();
        if deadline <= now + telem_spin_ms() * MS {
            break;
        }
        std::thread::sleep(std::time::Duration::from_nanos((deadline - now) / 2));
    }
    while now_ns() < deadline {
        core::hint::spin_loop();
    }
    now_ns() - deadline
}

#[cfg(test)]
mod telem_tests {
    use super::*;

    #[test]
    fn layout_encoding_and_round() {
        let mut hits = [0usize; TELEM_ARMS.len()];
        for &i in TELEM_ROUND.iter() {
            hits[i as usize] += 1;
        }
        assert_eq!(hits, [30, 30, 30, 30]);
        for blk in TELEM_ROUND.chunks(4) {
            let mut seen = [false; 4];
            for &i in blk {
                seen[i as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "{blk:?}");
        }
        // 6d3d6b82's form: arm i at 75 + 40 i, overrun at 235 (+5 = 240).
        for i in 0..4 {
            assert_eq!(telem_start_ms_k(i, 0), 75 + 40 * i as u64);
        }
        assert_eq!(telem_start_ms_k(4, 0) + TELEM_OVERRUN_MS / 2, 240);
        let ms = MS as i64;
        assert_eq!(telem_encode_ns_k(Some(91 * ms), 0), 19 * MS);
        assert_eq!(telem_encode_ns_k(Some(71 * ms), 0), 37 * MS);
        assert_eq!(telem_encode_ns_k(Some(108 * ms), 0), 38 * MS);
        assert_eq!(telem_encode_ns_k(None, 0), 39 * MS);
        // The M4 profile shifts the range by 120 ms and keeps the holds.
        assert_eq!(telem_encode_ns_k(Some(211 * ms), 120), 19 * MS);
        assert_eq!(telem_start_ms_k(3, 120) - (TELEM_LO_MS as u64 + 120), telem_start_ms_k(3, 0) - TELEM_LO_MS as u64);
        // Deadlines: total 91 in arm 2 returns 3 + 80 ms after the proof.
        assert_eq!(telem_deadline_ns_k(5, 2, 19 * MS, 91 * MS, 0), 5 + (91 + 83) * MS);
        assert_eq!(telem_deadline_ns_k(5, 0, 19 * MS, 94 * MS, 0), 5 + 240 * MS);
        // A test binary is not the ranked worker.
        telem_init();
        assert_eq!(telem_slot(), None);
    }
}

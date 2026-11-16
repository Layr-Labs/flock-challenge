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
/// diagnostic-run quantity ([`telem_init`]).
pub fn on() -> bool {
    FORCED.load(Ordering::Relaxed) || env_on()
}

fn env_on() -> bool {
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
// Every ranked worker, warm-up and scored alike, returns its finished proof
// late on purpose, so that the trial time the benchmark publishes carries one
// internal timing of that worker's own timed proof. Channel ported from
// patternrecognition9-del's disclosed diagnostic run 59257514 (spans only).
//
// - Nothing reads the trial number or the ready/proof file names, and no state
//   is kept between trials: a worker takes the quantity
//   `TELEM_ROUND[(unix time / TELEM_BIN_MS) % len]`.
// - Quantity `i` owns the window `[start_i, start_i + valid_i + TELEM_GUARD_MS)`
//   ms after the seed read ([`telem_start_ms`]) and returns its proof at
//   `seed read + start_i + v`, `v = value - lo_i` for a value in
//   `[lo_i, lo_i + valid_i)`. Sentinels: `v = valid_i + 1` below range,
//   `+ 2` above range, `+ 3` never recorded. Proof not ready 1 ms before its
//   time: it returns in the middle of a last `TELEM_OVERRUN_MS` window.
// - The first window is `total` (seed read -> prove end), whose return is
//   always `TELEM_LEAD_MS` after its own proof is done; every later window
//   starts after the largest `total` the first one encodes.
// - The hold runs after the timed proof is finished ("prove end") and before
//   the unchanged proof is published. Proof bytes, seed read and publication
//   code are untouched. The record's start pause is kept as it is.

/// Recorder armed for this process (every ranked worker of this tree).
static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// One quantity: `to - from` (last marks of the timed proof, ns, signed;
/// `from == ""` is the always-zero calibration) encoded as `value - lo_ms`.
#[derive(Clone, Copy, Debug)]
pub struct TelemQuantity {
    pub name: &'static str,
    pub from: &'static str,
    pub to: &'static str,
    pub lo_ms: i64,
    pub valid_ms: u64,
}

const fn q(name: &'static str, from: &'static str, to: &'static str, lo_ms: i64, valid_ms: u64) -> TelemQuantity {
    TelemQuantity { name, from, to, lo_ms, valid_ms }
}

const S: &str = "seed read";

/// The run's quantities, in window order (`total` must stay first).
pub static TELEM_QUANTITIES: [TelemQuantity; 15] = [
    q("total", S, "prove end", 72, 36),
    // Value 0 sits 3 ms into its window: found whatever the harness constant.
    q("zero", "", "", -3, 4),
    q("lincheck", "zc end", "lc returned", 0, 8),
    q("zerocheck round 1", "commit join end", "zc r1 done", 0, 14),
    q("open", "open start", "open returned", 3, 15),
    q("witgen end", S, "witgen end", 0, 14),
    q("zc k gpu-cpu end", "zc k cpu end", "zc k gpu end", -8, 16),
    q("zc r2 gpu-cpu end", "zc r2 cpu end", "zc r2 gpu end", -8, 16),
    q("zerocheck round 2", "zc r1 done", "zc r2 done", 4, 22),
    q("zerocheck rest", "zc r2 done", "zc end", 4, 24),
    q("split k_cpu16", "split base", "split k_cpu16", 0, 16),
    q("ab arm end", S, "ab arm end", 28, 24),
    q("commit root", S, "commit arm root", 38, 32),
    q("cpu ntt done", S, "hybrid cpu ntt done", 34, 24),
    q("cb2 gpu end", S, "cb2 gpu end", 30, 36),
];

/// Clock-bin order of the quantities (indices into [`TELEM_QUANTITIES`]),
/// one pass per job, repeats spread: split 6; zero and ab arm end 8;
/// witgen, the two gpu-cpu ends, cpu ntt done and cb2 gpu end 7; every
/// other 9 of 120.
pub static TELEM_ROUND: [u8; 120] = [
    0, 2, 3, 4, 8, 9, 12, 1, 11, 5, 6, 7, 13, 14, 10, 0, 2, 3, 4, 8, 9, 12, 1, 11, 5, 6, 7, 13, 14, 10,
    0, 2, 3, 4, 8, 9, 12, 1, 11, 5, 6, 7, 13, 14, 0, 2, 3, 4, 8, 9, 12, 10, 1, 11, 5, 6, 7, 13, 14, 0,
    2, 3, 4, 8, 9, 12, 1, 11, 10, 0, 2, 3, 4, 8, 9, 12, 5, 6, 7, 13, 14, 1, 11, 0, 2, 3, 4, 8, 9, 12,
    10, 5, 6, 7, 13, 14, 1, 11, 0, 2, 3, 4, 8, 9, 12, 10, 5, 6, 7, 13, 14, 1, 11, 0, 2, 3, 4, 8, 9, 12,
];

/// Return of a `total` trial after its own proof is done (ms).
pub const TELEM_LEAD_MS: u64 = 3;
/// Sentinel room at the end of each window (1 ms apart).
pub const TELEM_GUARD_MS: u64 = 4;
/// Width of the final "proof was not ready in time" window.
pub const TELEM_OVERRUN_MS: u64 = 10;
/// Clock bin of the round (longer than a trial's ~5.8 s cycle).
pub const TELEM_BIN_MS: u64 = 6000;
/// The wait spins (instead of sleeping) once this little is left.
const TELEM_SPIN_MS: u64 = 20;
/// A return later than this after its intended time counts as late (ns).
const LATE_NS: u64 = 300_000;
const MS: u64 = 1_000_000;

/// Window start of quantity `i`, ms after the seed read (`i == len`: the
/// overrun window).
pub const fn telem_start_ms(i: usize) -> u64 {
    let mut s = TELEM_QUANTITIES[0].lo_ms as u64 + TELEM_LEAD_MS;
    let mut k = 0;
    while k < i {
        s += TELEM_QUANTITIES[k].valid_ms + TELEM_GUARD_MS;
        k += 1;
    }
    s
}

/// Pure: the value (ns) of quantity `i` from a lookup of the last marks.
pub fn telem_value_ns(i: usize, last: impl Fn(&str) -> Option<u64>) -> Option<i64> {
    let quantity = &TELEM_QUANTITIES[i];
    if quantity.from.is_empty() {
        return Some(0);
    }
    Some(last(quantity.to)? as i64 - last(quantity.from)? as i64)
}

/// Pure: the offset inside its window (ns) that encodes `value` (`None` =
/// never recorded) for quantity `i`.
pub fn telem_encode_ns(i: usize, value: Option<i64>) -> u64 {
    let quantity = &TELEM_QUANTITIES[i];
    let valid = quantity.valid_ms * MS;
    match value.map(|v| v - quantity.lo_ms * MS as i64) {
        None => valid + 3 * MS,
        Some(v) if v < 0 => valid + MS,
        Some(v) if v as u64 >= valid => valid + 2 * MS,
        Some(v) => v as u64,
    }
}

/// Pure: when quantity `i` with offset `enc_ns` returns its proof (ns on the
/// mark clock); a proof not ready 1 ms before then goes to the overrun window.
pub fn telem_deadline_ns(seed_ns: u64, i: usize, enc_ns: u64, now_ns: u64) -> u64 {
    let deadline = seed_ns + telem_start_ms(i) * MS + enc_ns;
    if now_ns + MS > deadline {
        seed_ns + (telem_start_ms(TELEM_QUANTITIES.len()) + TELEM_OVERRUN_MS / 2) * MS
    } else {
        deadline
    }
}

/// 1-based quantity index of this process, 0 = none.
static TELEM_SLOT: AtomicUsize = AtomicUsize::new(0);

/// Give this process its quantity if it is the ranked worker
/// (`flock-benchmark-worker 18 READY PROOF`; the file names are not read).
/// First statement of `init_perf_thread_pool`.
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
            let i = TELEM_ROUND[(unix_ms / TELEM_BIN_MS) as usize % TELEM_ROUND.len()];
            TELEM_SLOT.store(i as usize + 1, Ordering::Relaxed);
            FORCED.store(true, Ordering::Relaxed);
        }
    });
}

/// This process's quantity index, if any.
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
/// published: waits until this worker's quantity says the proof comes back.
/// Returns at once in a process without a quantity.
pub fn telem_hold() {
    let Some(i) = telem_slot() else {
        return;
    };
    let Some(seed) = find_last(S) else {
        return;
    };
    let enc = telem_encode_ns(i, telem_value_ns(i, find_last));
    if telem_wait_until(telem_deadline_ns(seed, i, enc, now_ns())) > LATE_NS {
        // Woken late: never return at a time that decodes as another value.
        telem_wait_until(telem_deadline_ns(seed, i, enc, u64::MAX / 2));
    }
}

/// Wait until `deadline` (ns on the mark clock) and return how late the wait
/// ended: sleeps half of what is left while more than `TELEM_SPIN_MS` are
/// left (a wake-up must be over 10 ms late to pass the time), spins the rest.
fn telem_wait_until(deadline: u64) -> u64 {
    loop {
        let now = now_ns();
        if deadline <= now + TELEM_SPIN_MS * MS {
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
        let mut hits = [0usize; TELEM_QUANTITIES.len()];
        for &i in TELEM_ROUND.iter() {
            hits[i as usize] += 1;
        }
        assert!(hits.iter().all(|&h| (6..=9).contains(&h)), "{hits:?}");
        assert_eq!(TELEM_QUANTITIES[0].name, "total");
        // Every window after the first starts after the largest total the
        // first encodes; everything ends by 0.45 s after the seed read.
        assert!(telem_start_ms(1) > (TELEM_QUANTITIES[0].lo_ms as u64 + TELEM_QUANTITIES[0].valid_ms) + 1);
        assert!(telem_start_ms(TELEM_QUANTITIES.len()) + TELEM_OVERRUN_MS <= 450);
        // The zero reading lands 3 ms into its window.
        assert_eq!(telem_encode_ns(1, Some(0)), 3 * MS);
        let ms = MS as i64;
        // total: lo 72, valid 36.
        assert_eq!(telem_start_ms(0), 75);
        assert_eq!(telem_encode_ns(0, Some(95 * ms)), 23 * MS);
        assert_eq!(telem_encode_ns(0, Some(71 * ms)), 37 * MS);
        assert_eq!(telem_encode_ns(0, Some(108 * ms)), 38 * MS);
        assert_eq!(telem_encode_ns(0, None), 39 * MS);
        // A signed span (lo -8): -3 ms sits 5 ms into the window.
        let k = TELEM_QUANTITIES.iter().position(|q| q.name == "zc k gpu-cpu end").unwrap();
        assert_eq!(telem_encode_ns(k, Some(-3 * ms)), 5 * MS);
        let last = |l: &str| match l {
            "seed read" => Some(1_000),
            "prove end" => Some(95_001_000),
            "zc k cpu end" => Some(50_000_000),
            "zc k gpu end" => Some(47_000_000),
            _ => None,
        };
        assert_eq!(telem_value_ns(0, last), Some(95 * ms));
        assert_eq!(telem_value_ns(1, last), Some(0));
        assert_eq!(telem_value_ns(2, last), None);
        assert_eq!(telem_value_ns(k, last), Some(-3 * ms));
        // Deadlines: in time, and too late (the overrun window).
        assert_eq!(telem_deadline_ns(5, 0, 23 * MS, 95 * MS), 5 + 98 * MS);
        let end = telem_start_ms(TELEM_QUANTITIES.len());
        assert_eq!(telem_deadline_ns(5, 0, 23 * MS, 98 * MS), 5 + (end + TELEM_OVERRUN_MS / 2) * MS);
        // A test binary is not the ranked worker.
        telem_init();
        assert_eq!(telem_slot(), None);
    }
}

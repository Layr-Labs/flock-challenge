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
        GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

/// Bumped by every [`reset`]: deferred GPU-interval entries taken before the
/// seed read (warm-up) carry an older value and are dropped, not recorded.
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn generation() -> u64 {
    GENERATION.load(Ordering::Relaxed)
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
// internal timing of that worker's own timed proof. The channel is i34-9's
// disclosed diagnostic run c4d5dd3c (after patternrecognition9-del's 59257514
// and newjordan's ead7f97a), as our b2ab82f5 ported it: one quantity per
// trial, a window per quantity.
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
//   code are untouched.
// - Trial period: the record's start pause (`crate::ranked_start_pause`) is
//   shortened by this worker's nominal hold ([`telem_pause_cut`]), chosen with
//   the quantity at process start, so each trial lasts as long as the record's
//   and the job's wall time stays the record's.

/// Recorder armed for this process (every ranked worker of this tree).
static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// One quantity, from the last marks of the timed proof (ns, signed):
/// `to - from`; `to - max(from, from2)` when `from2` is set (the later of two
/// ends); `from == ""` is the always-zero calibration. Encoded as
/// `value - lo_ms`.
#[derive(Clone, Copy, Debug)]
pub struct TelemQuantity {
    pub name: &'static str,
    pub from: &'static str,
    pub from2: &'static str,
    pub to: &'static str,
    pub lo_ms: i64,
    pub valid_ms: u64,
}

const fn q(name: &'static str, from: &'static str, to: &'static str, lo_ms: i64, valid_ms: u64) -> TelemQuantity {
    TelemQuantity { name, from, from2: "", to, lo_ms, valid_ms }
}

const S: &str = "seed read";
const CPU_NTT: &str = "hybrid cpu ntt done";
const CB2_END: &str = "cb2 gpu end";

/// The run's quantities, in window order (`total` must stay first, `zero`
/// second). Runner profile; the M4 self-test profile ([`telem_profile`])
/// doubles every lo and triples every width.
pub static TELEM_QUANTITIES: [TelemQuantity; 11] = [
    q("total", S, "prove end", 74, 32),
    // Value 0 sits 3 ms into its window: found whatever the harness constant.
    q("zero", "", "", -3, 4),
    // Commitment window, signed: the GPU prefix's end (cb2) minus the CPU
    // NTT's end (positive = the GPU finished last).
    q("commit gap", CPU_NTT, CB2_END, -6, 14),
    // Commitment root minus the later of the two ends above (the tail).
    TelemQuantity { name: "commit tail", from: CPU_NTT, from2: CB2_END, to: "commit arm root", lo_ms: -2, valid_ms: 12 },
    // Signed joins: the GPU arm's end minus the CPU side's arrival at the
    // join (positive = the GPU finished last). Round one's AB completion and
    // C fold, round two, the K pass.
    q("zc r1 ab gpu-cpu end", "zc ab cpu end", "zc ab gpu end", -8, 16),
    q("zc r1 c gpu-cpu end", "zc c cpu end", "zc c gpu end", -10, 18),
    q("zc r2 gpu-cpu end", "zc r2 cpu end", "zc r2 gpu end", -6, 10),
    q("zc k gpu-cpu end", "zc k cpu end", "zc k gpu end", -6, 10),
    q("lincheck", "zc end", "lc returned", 0, 8),
    q("materialize", "o:mat start", "o:mat end", 0, 8),
    q("open", "open start", "open returned", 3, 15),
];

/// Clock-bin order of the quantities (indices into [`TELEM_QUANTITIES`]):
/// eleven blocks of eleven, every block one permutation of the eleven
/// quantities, and two bins of the same quantity are at least seven bins
/// apart (cyclically). A job's ~116 bins therefore miss at most one bin of
/// any quantity: every quantity gets ten or eleven workers.
pub static TELEM_ROUND: [u8; 121] = [
    10, 3, 0, 7, 2, 6, 1, 5, 8, 4, 9, //
    10, 6, 7, 1, 3, 5, 2, 9, 0, 4, 8, //
    3, 5, 6, 7, 2, 1, 10, 9, 0, 8, 4, //
    3, 7, 5, 1, 6, 8, 10, 9, 4, 0, 2, //
    7, 8, 6, 9, 5, 10, 1, 4, 2, 3, 0, //
    8, 9, 7, 5, 6, 2, 4, 1, 3, 10, 0, //
    6, 2, 8, 5, 9, 3, 4, 0, 7, 10, 1, //
    6, 9, 8, 3, 7, 4, 5, 0, 1, 2, 10, //
    3, 8, 5, 9, 7, 2, 0, 10, 1, 6, 4, //
    3, 9, 7, 0, 8, 1, 4, 10, 2, 6, 5, //
    3, 0, 7, 10, 2, 4, 9, 1, 8, 6, 5, //
];

/// Return of a `total` trial after its own proof is done (ms).
pub const TELEM_LEAD_MS: u64 = 3;
/// Sentinel room at the end of each window (1 ms apart).
pub const TELEM_GUARD_MS: u64 = 4;
/// Width of the final "proof was not ready in time" window.
pub const TELEM_OVERRUN_MS: u64 = 10;
/// Clock bin of the round (longer than a trial's ~5.8 s cycle).
pub const TELEM_BIN_MS: u64 = 6000;
/// The wait spins (instead of sleeping) once this little is left. On the
/// M4, a hold that slept made the publication after it slower than after a
/// short hold, which biases the later windows high. Every runner-profile
/// hold is shorter than this (at most ~190 ms, see the tests), so no runner
/// hold sleeps.
const TELEM_SPIN_MS: u64 = 250;
/// Nominal `total` (ms) for the start-pause cut only: the cut assumes this
/// prove time and a value in the middle of its window. A different actual
/// value changes that trial's period by the difference, nothing else.
pub const TELEM_NOMINAL_TOTAL_MS: u64 = 89;

/// [`TELEM_SPIN_MS`], or `FLOCK_TELEM_SPIN_MS` (self-test only; the ranked
/// harness clears the environment). Read once.
fn telem_spin_ms() -> u64 {
    static V: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
        std::env::var("FLOCK_TELEM_SPIN_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(TELEM_SPIN_MS)
    });
    *V
}
/// A return later than this after its intended time counts as late (ns).
const LATE_NS: u64 = 300_000;
const MS: u64 = 1_000_000;

/// Profile `(lo factor, width factor)`: `(1, 1)` on the runner;
/// `FLOCK_TELEM_PROFILE=m4` (self-test only: the M4's proof is ~2.3x the
/// runner's) gives `(2, 3)`. Read once. The ranked harness clears the
/// environment, so the runner always uses `(1, 1)`.
pub fn telem_profile() -> (u64, u64) {
    static P: std::sync::LazyLock<(u64, u64)> = std::sync::LazyLock::new(|| {
        if std::env::var("FLOCK_TELEM_PROFILE").is_ok_and(|v| v == "m4") { (2, 3) } else { (1, 1) }
    });
    *P
}

/// Window start of quantity `i`, ms after the seed read (`i == len`: the
/// overrun window), at profile `(kl, kw)`.
pub fn telem_start_ms_k(i: usize, (kl, kw): (u64, u64)) -> u64 {
    let mut s = TELEM_QUANTITIES[0].lo_ms as u64 * kl + TELEM_LEAD_MS * kw;
    for quantity in TELEM_QUANTITIES.iter().take(i) {
        s += (quantity.valid_ms + TELEM_GUARD_MS) * kw;
    }
    s
}

pub fn telem_start_ms(i: usize) -> u64 {
    telem_start_ms_k(i, telem_profile())
}

/// Pure: the value (ns) of quantity `i` from a lookup of the last marks.
pub fn telem_value_ns(i: usize, last: impl Fn(&str) -> Option<u64>) -> Option<i64> {
    let quantity = &TELEM_QUANTITIES[i];
    if quantity.from.is_empty() {
        return Some(0);
    }
    let from = if quantity.from2.is_empty() {
        last(quantity.from)?
    } else {
        last(quantity.from)?.max(last(quantity.from2)?)
    };
    Some(last(quantity.to)? as i64 - from as i64)
}

/// Pure: the offset inside its window (ns) that encodes `value` (`None` =
/// never recorded) for quantity `i` at profile `(kl, kw)`: `value - lo*kl`
/// unscaled (1 ms of value = 1 ms of offset) inside `valid*kw`; sentinels at
/// `valid*kw + {1,2,3}*kw` ms.
pub fn telem_encode_ns_k(i: usize, value: Option<i64>, (kl, kw): (u64, u64)) -> u64 {
    let quantity = &TELEM_QUANTITIES[i];
    let valid = quantity.valid_ms * kw * MS;
    match value.map(|v| v - quantity.lo_ms * (kl * MS) as i64) {
        None => valid + 3 * kw * MS,
        Some(v) if v < 0 => valid + kw * MS,
        Some(v) if v as u64 >= valid => valid + 2 * kw * MS,
        Some(v) => v as u64,
    }
}

pub fn telem_encode_ns(i: usize, value: Option<i64>) -> u64 {
    telem_encode_ns_k(i, value, telem_profile())
}

/// Pure: when quantity `i` with offset `enc_ns` returns its proof (ns on the
/// mark clock); a proof not ready 1 ms before then goes to the overrun window.
pub fn telem_deadline_ns_k(seed_ns: u64, i: usize, enc_ns: u64, now_ns: u64, k: (u64, u64)) -> u64 {
    let deadline = seed_ns + telem_start_ms_k(i, k) * MS + enc_ns;
    if now_ns + MS > deadline {
        seed_ns + (telem_start_ms_k(TELEM_QUANTITIES.len(), k) + TELEM_OVERRUN_MS * k.1 / 2) * MS
    } else {
        deadline
    }
}

/// Pure: the nominal hold (ms) of quantity `i` at profile `(kl, kw)`, i.e.
/// its return time for a value in the middle of its window (`total`: its
/// fixed lead) minus a nominal prove time ([`TELEM_NOMINAL_TOTAL_MS`] * kl).
pub fn telem_nominal_hold_ms_k(i: usize, (kl, kw): (u64, u64)) -> u64 {
    if i == 0 {
        return TELEM_LEAD_MS * kw;
    }
    let ret = telem_start_ms_k(i, (kl, kw)) + TELEM_QUANTITIES[i].valid_ms * kw / 2;
    ret.saturating_sub(TELEM_NOMINAL_TOTAL_MS * kl)
}

/// 1-based quantity index of this process, 0 = none.
static TELEM_SLOT: AtomicUsize = AtomicUsize::new(0);

/// Give this process its quantity if it is the ranked worker
/// (`flock-benchmark-worker 18 READY PROOF`; the file names are not read).
/// First statement of `init_perf_thread_pool`, before the start pause.
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
            // Self-test only: `FLOCK_TELEM_FORCE=<index>` pins the quantity
            // (the ranked harness clears the environment, so never on the runner).
            let forced = std::env::var("FLOCK_TELEM_FORCE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&f| f < TELEM_QUANTITIES.len());
            let i = forced.unwrap_or(TELEM_ROUND[(unix_ms / TELEM_BIN_MS) as usize % TELEM_ROUND.len()] as usize);
            TELEM_SLOT.store(i + 1, Ordering::Relaxed);
            FORCED.store(true, Ordering::Relaxed);
        }
    });
}

/// This process's quantity index, if any.
pub fn telem_slot() -> Option<usize> {
    TELEM_SLOT.load(Ordering::Relaxed).checked_sub(1)
}

/// How much the ranked start pause is shortened in this process: the nominal
/// hold of its quantity ([`telem_nominal_hold_ms_k`]), zero in a process
/// without one. Read by `crate::ranked_start_pause` after [`telem_init`].
pub fn telem_pause_cut() -> std::time::Duration {
    match telem_slot() {
        Some(i) => std::time::Duration::from_millis(telem_nominal_hold_ms_k(i, telem_profile())),
        None => std::time::Duration::ZERO,
    }
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
    let k = telem_profile();
    let enc = telem_encode_ns(i, telem_value_ns(i, find_last));
    if telem_wait_until(telem_deadline_ns_k(seed, i, enc, now_ns(), k)) > LATE_NS {
        // Woken late: never return at a time that decodes as another value.
        telem_wait_until(telem_deadline_ns_k(seed, i, enc, u64::MAX / 2, k));
    }
}

/// Wait until `deadline` (ns on the mark clock) and return how late the wait
/// ended: sleeps half of what is left while more than the final spin is
/// left, spins the rest.
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
        let n = TELEM_QUANTITIES.len();
        let mut hits = [0usize; 11];
        for &i in TELEM_ROUND.iter() {
            hits[i as usize] += 1;
        }
        assert_eq!(hits, [11; 11]);
        // Every block of eleven is a permutation of the quantities.
        for blk in TELEM_ROUND.chunks(n) {
            let mut seen = [false; 11];
            for &i in blk {
                seen[i as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "{blk:?}");
        }
        // Two bins of one quantity are at least seven bins apart (cyclic).
        let len = TELEM_ROUND.len();
        for a in 0..n as u8 {
            let pos: Vec<usize> = (0..len).filter(|&p| TELEM_ROUND[p] == a).collect();
            for (j, &p) in pos.iter().enumerate() {
                let next = pos[(j + 1) % pos.len()];
                assert!((next + len - p) % len >= 7, "quantity {a} at {p} and {next}");
            }
        }
        assert_eq!(TELEM_QUANTITIES[0].name, "total");
        assert_eq!(TELEM_QUANTITIES[1].name, "zero");
        // Windows: total first; every later window starts after the largest
        // total the first encodes (its sentinels included).
        let r = (1, 1);
        assert_eq!(telem_start_ms_k(0, r), 77);
        assert_eq!(telem_start_ms_k(1, r), 77 + 32 + 4);
        assert!(telem_start_ms_k(1, r) > 74 + 32 + 3);
        assert_eq!(telem_start_ms_k(n, r), 268);
        let m4 = (2, 3);
        assert!(telem_start_ms_k(1, m4) > 2 * 74 + 3 * 32 + 3 * 3);
        // The zero reading lands 3 ms into its window, at either profile
        // (lo -3 doubled to -6 on the M4 profile).
        assert_eq!(telem_encode_ns_k(1, Some(0), r), 3 * MS);
        assert_eq!(telem_encode_ns_k(1, Some(0), m4), 6 * MS);
        let ms = MS as i64;
        // total: lo 74, valid 32.
        assert_eq!(telem_encode_ns_k(0, Some(89 * ms), r), 15 * MS);
        assert_eq!(telem_encode_ns_k(0, Some(73 * ms), r), 33 * MS);
        assert_eq!(telem_encode_ns_k(0, Some(106 * ms), r), 34 * MS);
        assert_eq!(telem_encode_ns_k(0, None, r), 35 * MS);
        // A signed span (lo -6): -1 ms sits 5 ms into the window.
        let kk = TELEM_QUANTITIES.iter().position(|q| q.name == "zc k gpu-cpu end").unwrap();
        assert_eq!(telem_encode_ns_k(kk, Some(-ms), r), 5 * MS);
        let gap = TELEM_QUANTITIES.iter().position(|q| q.name == "commit gap").unwrap();
        let tail = TELEM_QUANTITIES.iter().position(|q| q.name == "commit tail").unwrap();
        let last = |l: &str| match l {
            "seed read" => Some(1_000),
            "prove end" => Some(89_001_000),
            "zc k cpu end" => Some(50_000_000),
            "zc k gpu end" => Some(47_000_000),
            "hybrid cpu ntt done" => Some(40_000_000),
            "cb2 gpu end" => Some(41_500_000),
            "commit arm root" => Some(42_250_000),
            _ => None,
        };
        assert_eq!(telem_value_ns(0, last), Some(89 * ms));
        assert_eq!(telem_value_ns(1, last), Some(0));
        assert_eq!(telem_value_ns(kk, last), Some(-3 * ms));
        assert_eq!(telem_value_ns(gap, last), Some(1_500_000));
        // The tail counts from the later of the two ends (here cb2's).
        assert_eq!(telem_value_ns(tail, last), Some(750_000));
        let cpu_later = |l: &str| if l == "hybrid cpu ntt done" { Some(42_000_000) } else { last(l) };
        assert_eq!(telem_value_ns(tail, cpu_later), Some(250_000));
        assert_eq!(telem_value_ns(gap, cpu_later), Some(-500_000));
        let ab = TELEM_QUANTITIES.iter().position(|q| q.name == "zc r1 ab gpu-cpu end").unwrap();
        assert_eq!(telem_value_ns(ab, last), None);
        // Deadlines: in time, and too late (the overrun window).
        assert_eq!(telem_deadline_ns_k(5, 0, 15 * MS, 89 * MS, r), 5 + 92 * MS);
        let end = telem_start_ms_k(n, r);
        assert_eq!(telem_deadline_ns_k(5, 0, 15 * MS, 92 * MS, r), 5 + (end + TELEM_OVERRUN_MS / 2) * MS);
        // A test binary is not the ranked worker.
        telem_init();
        assert_eq!(telem_slot(), None);
        assert_eq!(telem_pause_cut(), std::time::Duration::ZERO);
    }

    /// The runner holds and the start-pause cut: no hold sleeps (every hold
    /// is shorter than the final spin, even for an 80 ms prove that returns
    /// at the far end of the last window), the cut stays well inside the
    /// record's start pause, and over one round the cut matches the mean
    /// hold of mid-window values at the nominal prove time exactly.
    #[test]
    fn holds_spin_and_the_pause_cut_keeps_the_period() {
        let r = (1, 1);
        let n = TELEM_QUANTITIES.len();
        let longest = (1..n)
            .map(|i| telem_start_ms_k(i, r) + TELEM_QUANTITIES[i].valid_ms + 3)
            .max()
            .unwrap()
            .max(telem_start_ms_k(n, r) + TELEM_OVERRUN_MS / 2);
        assert!(longest - 80 < TELEM_SPIN_MS, "longest hold {} ms", longest - 80);
        assert_eq!(telem_nominal_hold_ms_k(0, r), TELEM_LEAD_MS);
        for i in 1..n {
            let cut = telem_nominal_hold_ms_k(i, r);
            assert!(cut > 0 && cut < 200, "quantity {i}: cut {cut} ms");
            let mid = telem_start_ms_k(i, r) + TELEM_QUANTITIES[i].valid_ms / 2;
            assert_eq!(cut, mid - TELEM_NOMINAL_TOTAL_MS);
        }
        // The zero calibration: value 0 at offset 3, so its hold for the
        // nominal prove is 1 ms more than its mid-window cut.
        let z = telem_start_ms_k(1, r) + 3 - TELEM_NOMINAL_TOTAL_MS;
        assert_eq!(z, telem_nominal_hold_ms_k(1, r) + 1);
    }

    /// Prints the window table the decoder mirrors (`cargo test -- --nocapture`).
    #[test]
    fn print_layout() {
        for k in [(1, 1), (2, 3)] {
            for (i, quantity) in TELEM_QUANTITIES.iter().enumerate() {
                println!(
                    "layout {k:?} {:2} {:22} start {:4} lo {:4} valid {:3} cut {:3}",
                    i,
                    quantity.name,
                    telem_start_ms_k(i, k),
                    quantity.lo_ms * k.0 as i64,
                    quantity.valid_ms * k.1,
                    telem_nominal_hold_ms_k(i, k)
                );
            }
            println!("layout {k:?} overrun start {}", telem_start_ms_k(TELEM_QUANTITIES.len(), k));
        }
    }
}

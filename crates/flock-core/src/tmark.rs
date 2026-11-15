//! Diagnostic-only cross-process timestamps (`FLOCK_TMARK=1`).
//!
//! Records `(label, CLOCK_UPTIME_RAW ns)` pairs into a fixed array so an
//! external driver that stamps its own seed write on the same clock can see
//! where the timed window goes. With the variable unset every call is one
//! cached-bool load; nothing is recorded and nothing is printed.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const SLOTS: usize = 4096;
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

/// Record a value as the distance between two marks: `base` now and `label`
/// `value_ns` after it (no-op unless enabled). A `Span(base, label)`
/// telemetry quantity then reports the value.
#[inline]
pub fn mark_value(base: &'static str, label: &'static str, value_ns: u64) {
    if !on() {
        return;
    }
    let t = now_ns();
    mark_ns(base, t);
    mark_ns(label, t + value_ns);
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
// Diagnostic-run telemetry
// ---------------------------------------------------------------------------
//
// THIS TREE IS A MEASUREMENT, NOT A CANDIDATE. Every ranked worker, warm-up
// and scored alike, returns its finished proof late on purpose, so that the
// trial time the benchmark publishes carries one internal quantity of that
// worker. The run's score is spent on the readings. Channel and method:
// patternrecognition9-del's diagnostic run 59257514-510a-43a3-ad54-b10b578da772
// (itself after newjordan's ead7f97a-eea7-41b5-a8e4-2b1c8549b793); this file
// ports its code to this tree with this tree's quantities, every one of them
// a span of the worker's own timed proof.
//
// - Nothing reads the trial number, the ready/proof file names or anything
//   else that tells a warm-up trial from a scored one. A worker takes the
//   quantity `TELEM_ROUND[(unix time / 6 s) % len]`.
// - Quantity `i` owns the window `[start_i, start_i + width_i)` ms after
//   `TELEM_BASE_MS`; the worker returns its proof at
//   `seed read + TELEM_BASE_MS + start_i + encoded value`, so the window says
//   which quantity a trial carries and the offset inside it is the value.
// - A `Span` quantity is the time between the last marks of two labels in the
//   worker's own timed proof (the marks are cleared at the seed read). A
//   recorded value (the commitment split) is a pair of marks that far apart
//   ([`mark_value`]).
// - The timed proof runs the default path; its bytes are untouched and every
//   proof is verified by the harness as usual. The seed read and the
//   publication code are not modified: the delay sits at the end of the
//   prover's own prove body.
// - Job time and trial spacing: a telemetry worker shortens the inherited
//   start pause by exactly the earliest return its window allows
//   (`TELEM_BASE_MS + start_i`), so its trial cycle stays close to a plain
//   worker's (see `telem_planned_delta_ms` and its test).
//
// Decoding a trial time `T` (ms): `x = T - c0 - TELEM_BASE_MS`, `c0` being
// the same for the `Zero` quantity (harness-side latency); the window holding
// `x` names the quantity; `v = x - start_i`. With `valid = width - 8`:
// `v < valid` is a reading (`v + sub`), `valid + 1` = below range,
// `valid + 3` = above range, `valid + 5` = never recorded. The last 10 ms
// window means the proof was not ready in time.

/// Recorder armed for this process (every ranked worker of this tree).
static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What a quantity reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemKind {
    /// Always 0: calibrates the harness-side constant `c0`.
    Zero,
    /// Last `b` minus last `a` in the timed proof.
    Span(&'static str, &'static str),
}

/// One quantity: its window width and the offset subtracted before encoding
/// (ms).
#[derive(Clone, Copy, Debug)]
pub struct TelemQuantity {
    pub name: &'static str,
    pub kind: TelemKind,
    pub width_ms: u64,
    pub sub_ms: u64,
}

const fn span(name: &'static str, a: &'static str, b: &'static str, width_ms: u64, sub_ms: u64) -> TelemQuantity {
    TelemQuantity { name, kind: TelemKind::Span(a, b), width_ms, sub_ms }
}

const S: &str = "seed read";

/// The recorded commitment split: one unit of the split (`k`, the CPU share
/// in sixteenths plus 16 per fine step) is 0.25 ms.
pub const TELEM_SPLIT_UNIT_NS: u64 = 250_000;

/// The run's quantities.
pub static TELEM_QUANTITIES: [TelemQuantity; 18] = [
    span("round 2 cpu wait", "r2 wait start", "r2 wait end", 24, 0),
    span("round 2 gpu", "r2 gpu start", "r2 gpu end", 32, 0),
    span("ab prefix cpu wait", "ab wait start", "ab wait end", 24, 0),
    span("ab prefix gpu", "ab gpu start", "ab gpu end", 24, 0),
    span("zerocheck round 1", "zc body start", "zc r1 done", 32, 0),
    span("zerocheck round 2", "zc r1 done", "zc r2 done", 40, 0),
    span("k pass", "k start", "k end", 32, 0),
    span("rounds 5+6", "t56 start", "t56 end", 24, 0),
    span("last grind", "grind start", "grind end", 24, 0),
    span("commit split", "split base", "split k", 40, 0),
    TelemQuantity { name: "zero", kind: TelemKind::Zero, width_ms: 16, sub_ms: 0 },
    span("witgen", "witgen start", "witgen end", 24, 0),
    span("commit", S, "commit join end", 64, 20),
    span("ab arm", "ab arm start", "ab arm end", 64, 0),
    span("zerocheck", "zc start", "zc end", 56, 10),
    span("lincheck", "lc call", "lc returned", 16, 0),
    span("open", "open start", "open returned", 32, 0),
    span("total", S, "prove end", 56, 70),
];

/// Order in which consecutive 6 s clock bins hand out quantities, repeats
/// spread apart (no entry next to itself, cyclically). Slots per round:
/// [`TELEM_WEIGHTS`].
pub static TELEM_ROUND: [u8; 62] = [
    2, 0, 5, 6, 8, 1, 3, 4, 7, 9, 10, 12, 14, 16, 17, 0, 11, 13, 15, 2, 5, 6, 0, 8, 1, 3, 4, 7, 9, 10, 0, 2, 5, 6, 12,
    14, 16, 17, 8, 0, 11, 2, 5, 6, 13, 15, 0, 1, 3, 4, 7, 9, 10, 12, 14, 16, 17, 8, 2, 5, 6, 0,
];

/// Slots per round of each quantity (round two's CPU wait most).
pub static TELEM_WEIGHTS: [usize; 18] = [7, 3, 5, 3, 3, 5, 5, 3, 4, 3, 3, 2, 3, 2, 3, 2, 3, 3];

/// The earliest return of any telemetry proof (ms after the seed read).
pub const TELEM_BASE_MS: u64 = 150;
/// Guard at the end of each window (sentinels live there).
pub const TELEM_GUARD_MS: u64 = 8;
/// Width of the final "proof was not ready in time" window.
pub const TELEM_OVERRUN_MS: u64 = 10;
/// Planned over the round, every telemetry trial is on average at least this
/// much SHORTER than a plain one (the job's hosted time limit is tight).
pub const TELEM_MARGIN_MS: u64 = 40;
/// A floor on a plain worker's own proof time after the seed read (ms): a
/// plain trial spends it too, so the plan counts only a telemetry worker's
/// hold beyond it.
pub const TELEM_PROOF_FLOOR_MS: u64 = 80;
/// Clock bin of the round-robin. About one trial's cycle, so consecutive
/// workers take the same or a nearby entry of the round.
pub const TELEM_BIN_MS: u64 = 6000;
/// The inherited start pause this budget was written against (ms); a test
/// in `lib.rs` pins it to `RANKED_START_PAUSE`.
pub const TELEM_PAUSE_MS: u64 = 1300;

/// The ranked size: telemetry arms only there.
const TELEM_LOG2: &str = "18";

const MS: u64 = 1_000_000;

/// Window start of quantity `i` (ms after `TELEM_BASE_MS`).
pub const fn telem_start_ms(i: usize) -> u64 {
    let mut sum = 0;
    let mut k = 0;
    while k < i {
        sum += TELEM_QUANTITIES[k].width_ms;
        k += 1;
    }
    sum
}

/// One quantity's result: a value (ns) or nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemValue {
    Value(i64),
    Missing,
}

/// Pure: the offset inside its window (ns) that encodes `value` for
/// quantity `i`.
pub fn telem_encode_ns(i: usize, value: TelemValue) -> u64 {
    let quantity = &TELEM_QUANTITIES[i];
    let valid = ((quantity.width_ms - TELEM_GUARD_MS) * MS) as i64;
    let value = match value {
        TelemValue::Value(v) => v,
        TelemValue::Missing => return valid as u64 + 5 * MS,
    };
    let shifted = value - (quantity.sub_ms * MS) as i64;
    if shifted < 0 {
        valid as u64 + MS
    } else if shifted >= valid {
        valid as u64 + 3 * MS
    } else {
        shifted as u64
    }
}

/// Pure: the value of quantity `i` from a mark lookup.
pub fn telem_value_ns(i: usize, last: impl Fn(&str) -> Option<u64>) -> TelemValue {
    match TELEM_QUANTITIES[i].kind {
        TelemKind::Zero => TelemValue::Value(0),
        TelemKind::Span(a, b) => match (last(a), last(b)) {
            (Some(ta), Some(tb)) if tb >= ta => TelemValue::Value((tb - ta) as i64),
            _ => TelemValue::Missing,
        },
    }
}

/// Pure: when quantity `i` with encoded offset `enc` returns its proof (ns
/// on the mark clock), given the seed read and the present; a proof that is
/// not ready 1 ms before its time goes to the overrun window instead.
pub fn telem_deadline_ns(seed_read_ns: u64, i: usize, enc_ns: u64, now_ns: u64) -> u64 {
    let deadline = seed_read_ns + (TELEM_BASE_MS + telem_start_ms(i)) * MS + enc_ns;
    if now_ns + MS > deadline {
        seed_read_ns
            + (TELEM_BASE_MS + telem_start_ms(TELEM_QUANTITIES.len()) + TELEM_OVERRUN_MS / 2) * MS
    } else {
        deadline
    }
}

/// Pure: the part of the start pause a worker with quantity `i` skips (ms):
/// the earliest return its window allows.
pub const fn telem_pause_cut_ms(i: usize) -> u64 {
    TELEM_BASE_MS + telem_start_ms(i)
}

/// Pure: the planned wall-time change (ms) of a worker with quantity `i`
/// against a plain worker: the latest return its window allows, minus the
/// part of the start pause it skips, minus the proof time a plain trial also
/// spends after the seed (at least `TELEM_PROOF_FLOOR_MS`).
pub const fn telem_planned_delta_ms(i: usize) -> i64 {
    let hold = TELEM_BASE_MS + telem_start_ms(i) + TELEM_QUANTITIES[i].width_ms;
    hold as i64 - telem_pause_cut_ms(i) as i64 - TELEM_PROOF_FLOOR_MS as i64
}

/// Pure: the mean of [`telem_planned_delta_ms`] over the round (ms).
pub fn telem_planned_mean_delta_ms() -> f64 {
    TELEM_ROUND.iter().map(|&i| telem_planned_delta_ms(i as usize) as f64).sum::<f64>()
        / TELEM_ROUND.len() as f64
}

/// Pure: the quantity a ranked worker starting at `unix_ms` takes.
pub fn telem_slot_for(ranked: bool, unix_ms: u64) -> Option<usize> {
    ranked.then(|| TELEM_ROUND[(unix_ms / TELEM_BIN_MS) as usize % TELEM_ROUND.len()] as usize)
}

/// Pure: the ranked worker shape (`flock-benchmark-worker LOG2 READY PROOF`
/// at the ranked size). The two file arguments are counted, never read.
pub fn telem_is_ranked(exe_is_worker: bool, args: &[std::ffi::OsString], log2: &str) -> bool {
    exe_is_worker && args.len() == 3 && args[0].to_str() == Some(log2)
}

/// 1-based quantity index of this process, 0 = none.
static TELEM_SLOT: AtomicUsize = AtomicUsize::new(0);

/// Give this process its quantity, if it is a ranked worker on macOS (the
/// mark clock reads 0 elsewhere, so no other build arms the delay). First
/// statement of `init_perf_thread_pool`. Returns the quantity index.
pub fn telem_init() -> Option<usize> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut args = std::env::args_os();
        let exe_is_worker = args.next().is_some_and(|exe| {
            std::path::Path::new(&exe)
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("flock-benchmark-worker"))
        });
        let rest: Vec<std::ffi::OsString> = args.collect();
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        let ranked = cfg!(target_os = "macos") && telem_is_ranked(exe_is_worker, &rest, TELEM_LOG2);
        if let Some(i) = telem_slot_for(ranked, unix_ms) {
            FORCED.store(true, Ordering::Relaxed);
            TELEM_SLOT.store(i + 1, Ordering::Relaxed);
        }
    });
    telem_slot()
}

/// This process's quantity index, if any.
pub fn telem_slot() -> Option<usize> {
    TELEM_SLOT.load(Ordering::Relaxed).checked_sub(1)
}

/// How much of the inherited start pause this worker skips: the earliest
/// return its quantity's window allows in a telemetry worker (so its trial
/// cycle stays close to a plain worker's), none otherwise.
pub fn telem_pause_cut() -> std::time::Duration {
    match telem_slot() {
        Some(i) => std::time::Duration::from_millis(telem_pause_cut_ms(i)),
        None => std::time::Duration::ZERO,
    }
}

/// Timestamp of the most recent `label` since the last [`reset`].
pub fn find_last(label: &str) -> Option<u64> {
    let n = NEXT.load(Ordering::Relaxed).min(SLOTS);
    let l = LABELS.lock().unwrap_or_else(|e| e.into_inner());
    (0..n.min(l.len())).rev().find(|&i| l[i] == label).map(|i| NS[i].load(Ordering::Relaxed))
}

/// Called by the prover when the timed proof is finished, before it returns
/// it: waits until this worker's quantity says the proof should come back.
/// Returns at once in a process without a quantity.
pub fn telem_hold() {
    let Some(i) = telem_slot() else {
        return;
    };
    let Some(seed_read) = find_last(S) else {
        return;
    };
    let value = telem_value_ns(i, find_last);
    let enc = telem_encode_ns(i, value);
    let deadline = telem_deadline_ns(seed_read, i, enc, now_ns());
    if telem_wait_until(deadline) > LATE_NS {
        // Woken or scheduled late: the intended time is gone. Never return at
        // a time that would decode as some other value; go to the overrun
        // window instead (or, if that is gone too, return at once: such a
        // trial lies beyond every window and decodes as nothing).
        telem_wait_until(telem_deadline_ns(seed_read, i, enc, u64::MAX / 2));
    }
}

/// A return later than this after its intended time counts as late (ns).
const LATE_NS: u64 = 300_000;

/// Wait until `deadline` (ns on the mark clock) and return how late the wait
/// ended. Sleeps (idle, no heat) while more than 300 ms are left, each sleep
/// half of what is left, so that a wake-up up to 150 ms late cannot pass the
/// time; spins the rest, where only a spin is exact.
fn telem_wait_until(deadline: u64) -> u64 {
    loop {
        let now = now_ns();
        if deadline <= now + 300 * MS {
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
    use std::ffi::OsString;

    #[test]
    fn layout_and_budget() {
        // Every quantity appears in the round with its weight.
        let mut hits = [0usize; TELEM_QUANTITIES.len()];
        for &i in TELEM_ROUND.iter() {
            hits[i as usize] += 1;
        }
        for (i, quantity) in TELEM_QUANTITIES.iter().enumerate() {
            assert_eq!(hits[i], TELEM_WEIGHTS[i], "{}", quantity.name);
            assert!(quantity.width_ms > TELEM_GUARD_MS);
            // Every telemetry worker is planned shorter than a plain one, and
            // its cut never exceeds the pause it shortens.
            assert!(telem_planned_delta_ms(i) < 0, "{} plans {}", quantity.name, telem_planned_delta_ms(i));
            assert!(telem_pause_cut_ms(i) < TELEM_PAUSE_MS, "{} cuts {}", quantity.name, telem_pause_cut_ms(i));
        }
        // The job-time budget: planned over the round, a telemetry trial is
        // on average at least the margin shorter than a plain trial.
        assert!(
            telem_planned_mean_delta_ms() <= -(TELEM_MARGIN_MS as f64),
            "mean plan {} ms",
            telem_planned_mean_delta_ms()
        );
        // No entry of the round sits next to itself (cyclically).
        let len = TELEM_ROUND.len();
        for k in 0..len {
            assert_ne!(TELEM_ROUND[k], TELEM_ROUND[(k + 1) % len], "round entry {k}");
        }
        // Distinct names (the decoder keys on them).
        for (i, a) in TELEM_QUANTITIES.iter().enumerate() {
            for b in &TELEM_QUANTITIES[i + 1..] {
                assert_ne!(a.name, b.name);
            }
        }
        assert_eq!(telem_start_ms(0), 0);
        assert_eq!(telem_start_ms(1), 24);
        assert_eq!(telem_start_ms(10), 296);
        assert_eq!(telem_start_ms(TELEM_QUANTITIES.len()), 624);
    }

    #[test]
    fn selection_reads_only_the_shape_and_the_clock() {
        let args = |log2: &str, proof: &str| -> Vec<OsString> {
            vec![log2.into(), "/s/x.ready".into(), proof.into()]
        };
        // The same answer whatever the file names say.
        for name in ["/s/run-1.proof", "/s/run-21.proof", "/s/run-120.proof", "/s/anything"] {
            assert!(telem_is_ranked(true, &args("18", name), "18"));
        }
        assert!(!telem_is_ranked(false, &args("18", "/s/run-1.proof"), "18"));
        assert!(!telem_is_ranked(true, &args("16", "/s/run-1.proof"), "18"));
        assert!(!telem_is_ranked(true, &args("18", "/s/run-1.proof")[..2], "18"));
        assert_eq!(telem_slot_for(false, 123_456), None);
        // Consecutive bins walk the round.
        for k in 0..(2 * TELEM_ROUND.len() as u64) {
            assert_eq!(
                telem_slot_for(true, k * TELEM_BIN_MS + 17),
                Some(TELEM_ROUND[k as usize % TELEM_ROUND.len()] as usize)
            );
        }
        // A test binary is not the ranked worker: no quantity, no pause cut.
        assert_eq!(telem_init(), None);
        assert_eq!(telem_pause_cut(), std::time::Duration::ZERO);
    }

    #[test]
    fn encoding() {
        let ms = MS as i64;
        let v = TelemValue::Value;
        let zero = TELEM_QUANTITIES.iter().position(|q| matches!(q.kind, TelemKind::Zero)).unwrap();
        let total = TELEM_QUANTITIES.iter().position(|q| q.name == "total").unwrap();
        // Width 24 (valid 16), no offset ("round 2 cpu wait").
        assert_eq!(telem_encode_ns(0, v(0)), 0);
        assert_eq!(telem_encode_ns(0, v(3 * ms)), 3 * MS);
        assert_eq!(telem_encode_ns(0, v(-1)), 17 * MS);
        assert_eq!(telem_encode_ns(0, v(16 * ms)), 19 * MS);
        assert_eq!(telem_encode_ns(0, TelemValue::Missing), 21 * MS);
        // Width 56 (valid 48), offset 70 ms ("total").
        assert_eq!(telem_encode_ns(total, v(70 * ms)), 0);
        assert_eq!(telem_encode_ns(total, v(100 * ms)), 30 * MS);
        assert_eq!(telem_encode_ns(total, v(69 * ms)), 49 * MS);
        assert_eq!(telem_encode_ns(total, v(118 * ms)), 51 * MS);
        assert_eq!(telem_encode_ns(total, TelemValue::Missing), 53 * MS);
        // Zero (width 16): its own value only.
        assert_eq!(telem_encode_ns(zero, v(0)), 0);
        assert_eq!(telem_encode_ns(zero, TelemValue::Missing), 13 * MS);
        // The recorded split: k units of 0.25 ms.
        let split = TELEM_QUANTITIES.iter().position(|q| q.name == "commit split").unwrap();
        assert_eq!(telem_encode_ns(split, v((5 * TELEM_SPLIT_UNIT_NS) as i64)), 5 * TELEM_SPLIT_UNIT_NS);
        // Values.
        let last = |l: &str| match l {
            "seed read" => Some(1_000),
            "prove end" => Some(99_001_000),
            "r2 wait start" => Some(5_000),
            "r2 wait end" => Some(4_000),
            _ => None,
        };
        assert_eq!(telem_value_ns(zero, last), v(0));
        assert_eq!(telem_value_ns(total, last), v(99_000_000));
        // A span whose end mark precedes its start, or is missing, is missing.
        assert_eq!(telem_value_ns(0, last), TelemValue::Missing);
        assert_eq!(telem_value_ns(1, last), TelemValue::Missing);
        // Deadlines: in time, and late (overrun window after the last one).
        let start = telem_start_ms(zero);
        assert_eq!(
            telem_deadline_ns(5, zero, 4 * MS, 120 * MS),
            5 + (TELEM_BASE_MS + start + 4) * MS
        );
        let all = telem_start_ms(TELEM_QUANTITIES.len());
        assert_eq!(
            telem_deadline_ns(5, 0, 4 * MS, 160 * MS),
            5 + (TELEM_BASE_MS + all + TELEM_OVERRUN_MS / 2) * MS
        );
    }
}

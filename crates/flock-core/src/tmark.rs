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
// ports its code to this tree with this tree's quantities.
//
// - Nothing reads the trial number, the ready/proof file names or anything
//   else that tells a warm-up trial from a scored one. A worker takes the
//   quantity `TELEM_ROUND[(unix time / 6 s) % len]`.
// - Quantity `i` owns the window `[start_i, start_i + width_i)` ms after
//   `TELEM_BASE_MS`; the worker returns its proof at
//   `seed read + TELEM_BASE_MS + start_i + encoded value`, so the window says
//   which quantity a trial carries and the offset inside it is the value.
// - A `Span` quantity is the time between two marks of the worker's own
//   timed proof (default path). A `Pair` quantity is an in-process A/B run
//   BEFORE readiness, outside every timed interval: one unmeasured proof on
//   the arm (so that the arm's one-time costs, cold code pages and lazy
//   state, are paid outside the measurement, as the warm-up proof paid the
//   default path's), then `TELEM_PAIRS` pairs of extra proofs of the warm-up
//   statement, default path against the arm (one runtime setting changed for
//   that proof only), order alternating. The reading is the median over the
//   pairs of `t(arm) - t(default)` over one part of the proof. Every extra
//   proof is serialized and compared byte for byte with the first one (all
//   of them prove the same statement with the same challenger, so all must
//   be equal); any difference is reported instead of a reading.
// - The timed proof always runs the default path (`arm() == 0`); its bytes
//   are untouched and every proof is verified by the harness as usual. The
//   seed read and the publication code are not modified: the delay sits at
//   the end of the prover's own prove body.
// - Job time: a telemetry worker skips the inherited start pause (whose only
//   purpose is to spread a scored job's trials over more wall time). Planned
//   over the round, the extra proofs and the late returns stay at least
//   `TELEM_MARGIN_MS` per trial below that pause (see
//   `telem_planned_delta_ms` and its test).
//
// Decoding a trial time `T` (ms): `x = T - c0 - TELEM_BASE_MS`, `c0` being
// the same for the `Zero` quantity (harness-side latency); the window holding
// `x` names the quantity; `v = x - start_i`. With `valid = width - 8`:
// `v < valid` is a reading (`Span`: `v + sub`; `Pair`: `v - valid / 2`),
// `valid + 1` = below range, `valid + 3` = above range, `valid + 5` = never
// recorded, `valid + 7` = an extra proof's bytes differed (pairs only). The
// last 10 ms window means the proof was not ready in time.

/// Recorder armed for this process (every ranked worker of this tree).
static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Paired arms. Each changes one runtime setting for the extra, untimed
/// proof it runs in; [`arm`] is 0 (the default path) everywhere else.
///
/// GPU share of the round-one AB completion GPU prefix: 0 (the CPU drains
/// every chunk; the process stays admitted).
pub const ARM_D1_OFF: u8 = 1;
/// GPU share of the round-one AB completion GPU prefix: twice the default.
pub const ARM_D1_X2: u8 = 2;
/// CPU share of the hybrid commitment split: one sixteenth more than the
/// tuned split (the fine part, if any, kept).
pub const ARM_SPLIT_PLUS1: u8 = 3;
/// Hybrid commitment split: the tuner's pick without the fine-split
/// refinement (recorded at tune time, [`telem_set_coarse_split`]).
pub const ARM_NOFINE: u8 = 4;
/// No post-drain linger of the queue workers.
pub const ARM_NO_LINGER: u8 = 5;
/// Changes nothing: the A/A noise floor of the paired readings.
pub const ARM_NULL: u8 = 6;
/// CPU share of the hybrid commitment split: one sixteenth SMALLER than the
/// tuned split (one sixteenth more on the GPU; the fine part, if any, kept).
pub const ARM_SPLIT_MINUS1: u8 = 7;

/// Arm of the proof now running: 0 (the default path) in every timed proof
/// and in every process without a `Pair` quantity.
static ARM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The paired arm in force (0 = default path). One relaxed load.
#[inline]
pub fn arm() -> u8 {
    ARM.load(Ordering::Relaxed)
}

/// Select the paired arm; a no-op in a process without a telemetry quantity.
pub fn set_arm(arm: u8) {
    if telem_slot().is_some() {
        ARM.store(arm, Ordering::Relaxed);
    }
}

/// Split the tuner would have published without the fine refinement
/// (`usize::MAX` = not recorded: the arm then keeps the tuned split).
static COARSE_SPLIT: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Record the coarse split at tune time (untimed warm-up).
pub fn telem_set_coarse_split(k: usize) {
    COARSE_SPLIT.store(k, Ordering::Relaxed);
}

/// Pure: the split the proof now proving uses, given the tuned split
/// `k_split` (`k_cpu16 + 16 * m`), the arm and the recorded coarse split.
pub fn telem_split_for(k_split: usize, arm: u8, coarse: usize) -> usize {
    match arm {
        ARM_SPLIT_PLUS1 if k_split % 16 < 15 => k_split + 1,
        ARM_SPLIT_MINUS1 if k_split % 16 > 0 => k_split - 1,
        ARM_NOFINE if coarse != usize::MAX => coarse,
        _ => k_split,
    }
}

/// [`telem_split_for`] with this process's arm and coarse split.
#[inline]
pub fn telem_split(k_split: usize) -> usize {
    match arm() {
        0 => k_split,
        a => telem_split_for(k_split, a, COARSE_SPLIT.load(Ordering::Relaxed)),
    }
}

/// Which part of a proof a paired reading compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    /// The whole proof.
    Whole,
    /// Commit join to prove end (zerocheck, lincheck, opening).
    Post,
}

/// What a quantity reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemKind {
    /// Always 0: calibrates the harness-side constant `c0`.
    Zero,
    /// Last `b` minus last `a` in the timed proof.
    Span(&'static str, &'static str),
    /// In-process `t(arm) - t(default)` over one part of the proof.
    Pair(u8, Half),
}

/// One quantity: its window width and, for spans, the offset subtracted
/// before encoding (ms).
#[derive(Clone, Copy, Debug)]
pub struct TelemQuantity {
    pub name: &'static str,
    pub kind: TelemKind,
    pub width_ms: u64,
    pub sub_ms: u64,
}

const fn q(name: &'static str, kind: TelemKind, width_ms: u64, sub_ms: u64) -> TelemQuantity {
    TelemQuantity { name, kind, width_ms, sub_ms }
}

const S: &str = "seed read";

/// The run's quantities. Those that run extra proofs come first: their late
/// returns are then the shortest.
pub static TELEM_QUANTITIES: [TelemQuantity; 5] = [
    q("pair split -1", TelemKind::Pair(ARM_SPLIT_MINUS1, Half::Whole), 32, 0),
    q("pair linger off", TelemKind::Pair(ARM_NO_LINGER, Half::Post), 24, 0),
    q("pair null whole", TelemKind::Pair(ARM_NULL, Half::Whole), 24, 0),
    q("pair null post", TelemKind::Pair(ARM_NULL, Half::Post), 24, 0),
    q("zero", TelemKind::Zero, 16, 0),
];

/// Order in which consecutive 6 s clock bins hand out quantities, repeats
/// spread apart (no entry next to itself, cyclically). Slots per round:
/// [`TELEM_WEIGHTS`].
pub static TELEM_ROUND: [u8; 41] = [
    0, 1, 4, 0, 1, 2, 0, 3, 1, 0, 4, 1, 0, 1, 0, 2, 0, 1, 0, 4, 3, 1, 0, 1, 4, 0, 2, 1, 0, 1, 4, 0, 1, 0, 3, 2, 1, 0, 4, 0, 1,
];

/// Slots per round of each quantity (the two arms most).
pub static TELEM_WEIGHTS: [usize; 5] = [15, 13, 4, 3, 6];

/// The earliest return of any telemetry proof (ms after the seed read).
pub const TELEM_BASE_MS: u64 = 150;
/// Guard at the end of each window (sentinels live there).
pub const TELEM_GUARD_MS: u64 = 8;
/// Width of the final "proof was not ready in time" window.
pub const TELEM_OVERRUN_MS: u64 = 10;
/// Pairs of extra proofs per `Pair` worker (the reading is their median).
pub const TELEM_PAIRS: usize = 4;
/// Budgeted time of one extra proof (comfortably above this tree's proof
/// time).
pub const TELEM_PROOF_EST_MS: u64 = 125;
/// Planned over the round, every telemetry trial is on average at least this
/// much SHORTER than a plain one (the job's hosted time limit is tight).
pub const TELEM_MARGIN_MS: u64 = 50;
/// Clock bin of the round-robin. Longer than a trial's cycle, so consecutive
/// workers take the same or the next entry of the round and none is skipped.
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

/// One quantity's result: a value (ns), nothing, or (pairs) an extra proof
/// whose bytes differed from the first extra proof's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemValue {
    Value(i64),
    Missing,
    Differed,
}

/// Pure: the offset inside its window (ns) that encodes `value` for
/// quantity `i`.
pub fn telem_encode_ns(i: usize, value: TelemValue) -> u64 {
    let quantity = &TELEM_QUANTITIES[i];
    let valid = ((quantity.width_ms - TELEM_GUARD_MS) * MS) as i64;
    let value = match value {
        TelemValue::Value(v) => v,
        TelemValue::Missing => return valid as u64 + 5 * MS,
        TelemValue::Differed => return valid as u64 + 7 * MS,
    };
    let shifted = match quantity.kind {
        TelemKind::Pair(..) => value + valid / 2,
        _ => value - (quantity.sub_ms * MS) as i64,
    };
    if shifted < 0 {
        valid as u64 + MS
    } else if shifted >= valid {
        valid as u64 + 3 * MS
    } else {
        shifted as u64
    }
}

/// Pure: the value of quantity `i` from a mark lookup and the stored result
/// of the extra proofs.
pub fn telem_value_ns(i: usize, last: impl Fn(&str) -> Option<u64>, pair: TelemValue) -> TelemValue {
    match TELEM_QUANTITIES[i].kind {
        TelemKind::Zero => TelemValue::Value(0),
        TelemKind::Span(a, b) => match (last(a), last(b)) {
            (Some(ta), Some(tb)) if tb >= ta => TelemValue::Value((tb - ta) as i64),
            _ => TelemValue::Missing,
        },
        TelemKind::Pair(..) => pair,
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

/// Pure: the planned wall-time change (ms) of a worker with quantity `i`
/// against a plain worker: the latest return its window allows, plus the
/// extra proofs of a pair, minus the start pause it skips. (The plain
/// worker's own proof time, which a plain trial also spends after the seed,
/// is not subtracted: that is slack on top of the plan.)
pub const fn telem_planned_delta_ms(i: usize) -> i64 {
    let hold = TELEM_BASE_MS + telem_start_ms(i) + TELEM_QUANTITIES[i].width_ms;
    let extra = match TELEM_QUANTITIES[i].kind {
        TelemKind::Pair(..) => (2 * TELEM_PAIRS as u64 + 1) * TELEM_PROOF_EST_MS,
        _ => 0,
    };
    (hold + extra) as i64 - TELEM_PAUSE_MS as i64
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
/// Clock bin this process drew (order of a pair's two proofs follows it).
static TELEM_BIN: AtomicU64 = AtomicU64::new(0);
/// Median paired difference in ns; `i64::MIN` = not recorded,
/// `i64::MIN + 1` = an extra proof's bytes differed.
static PAIR_DIFF: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
const PAIR_DIFFERED: i64 = i64::MIN + 1;

/// Give this process its quantity, if it is a ranked worker. First statement
/// of `init_perf_thread_pool`. Returns the quantity index.
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
        let ranked = telem_is_ranked(exe_is_worker, &rest, TELEM_LOG2);
        if let Some(i) = telem_slot_for(ranked, unix_ms) {
            FORCED.store(true, Ordering::Relaxed);
            TELEM_BIN.store(unix_ms / TELEM_BIN_MS, Ordering::Relaxed);
            TELEM_SLOT.store(i + 1, Ordering::Relaxed);
        }
    });
    telem_slot()
}

/// This process's quantity index, if any.
pub fn telem_slot() -> Option<usize> {
    TELEM_SLOT.load(Ordering::Relaxed).checked_sub(1)
}

/// How much of the inherited start pause this worker skips: all of it in a
/// telemetry worker, none otherwise.
pub fn telem_pause_cut() -> std::time::Duration {
    match telem_slot() {
        Some(_) => std::time::Duration::from_millis(TELEM_PAUSE_MS),
        None => std::time::Duration::ZERO,
    }
}

/// `(arm, half, pairs, arm_first)` when this process's quantity runs extra
/// proofs.
pub fn telem_pair() -> Option<(u8, Half, usize, bool)> {
    let arm_first = telem_arm_first(TELEM_BIN.load(Ordering::Relaxed));
    match TELEM_QUANTITIES[telem_slot()?].kind {
        TelemKind::Pair(arm, half) => Some((arm, half, TELEM_PAIRS, arm_first)),
        _ => None,
    }
}

/// Pure: whether the paired proofs of the worker that drew clock bin `bin`
/// start with the arm. The successive occurrences of one quantity in the
/// round alternate, and the whole pattern flips every round, so each
/// quantity's workers are split evenly between the two orders (a proof-to-
/// proof trend inside a worker that ABBA does not cancel then cancels across
/// workers). Reads the clock bin only.
pub fn telem_arm_first(bin: u64) -> bool {
    let len = TELEM_ROUND.len() as u64;
    let pos = (bin % len) as usize;
    let slot = TELEM_ROUND[pos];
    let occurrence = TELEM_ROUND[..pos].iter().filter(|&&s| s == slot).count() as u64;
    (occurrence + bin / len) % 2 == 1
}

/// Timestamp of the most recent `label` since the last [`reset`].
pub fn find_last(label: &str) -> Option<u64> {
    let n = NEXT.load(Ordering::Relaxed).min(SLOTS);
    let l = LABELS.lock().unwrap_or_else(|e| e.into_inner());
    (0..n.min(l.len())).rev().find(|&i| l[i] == label).map(|i| NS[i].load(Ordering::Relaxed))
}

/// Pure: one part of a proof that ran from `t0` to `t1`, given a lookup of
/// the proof's last timestamp per label (all ns).
pub fn telem_half_ns(
    half: Half,
    t0: u64,
    t1: u64,
    last: impl Fn(&str) -> Option<u64>,
) -> Option<i64> {
    let at = |label: &str| last(label).filter(|t| (t0..=t1).contains(t));
    Some(match half {
        Half::Whole => t1.checked_sub(t0)? as i64,
        Half::Post => (t1 - at("commit join end")?) as i64,
    })
}

/// The part time of the extra proof that just ran from `t0` to `t1`.
pub fn telem_half_of_last_proof(half: Half, t0: u64, t1: u64) -> Option<i64> {
    telem_half_ns(half, t0, t1, find_last)
}

/// Pure: the median of the paired differences (`None` if there are none);
/// an even count takes the mean of the middle two.
pub fn telem_median(mut diffs: Vec<i64>) -> Option<i64> {
    diffs.sort_unstable();
    let n = diffs.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(diffs[n / 2]),
        _ => Some(((diffs[n / 2 - 1] as i128 + diffs[n / 2] as i128) / 2) as i64),
    }
}

/// Store this worker's extra-proof result: the median of `t(arm) -
/// t(default)` over its pairs (`None` when nothing was measured), unless an
/// extra proof's bytes differed from the first extra proof's.
pub fn telem_pair_record(median_diff_ns: Option<i64>, differed: bool) {
    if differed {
        PAIR_DIFF.store(PAIR_DIFFERED, Ordering::Relaxed);
    } else if let Some(d) = median_diff_ns {
        PAIR_DIFF.store(d.max(PAIR_DIFFERED + 1), Ordering::Relaxed);
    }
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
    let pair = match PAIR_DIFF.load(Ordering::Relaxed) {
        i64::MIN => TelemValue::Missing,
        PAIR_DIFFERED => TelemValue::Differed,
        d => TelemValue::Value(d),
    };
    let value = telem_value_ns(i, find_last, pair);
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
        let zero = TELEM_QUANTITIES.iter().position(|q| matches!(q.kind, TelemKind::Zero)).unwrap();
        for (i, quantity) in TELEM_QUANTITIES.iter().enumerate() {
            let want = TELEM_WEIGHTS[i];
            // Quantities that run extra proofs sit before all others.
            if matches!(quantity.kind, TelemKind::Pair(..)) {
                assert!(i < zero, "{}", quantity.name);
            }
            assert_eq!(hits[i], want, "{}", quantity.name);
            assert!(quantity.width_ms > TELEM_GUARD_MS);
            // No single worker is planned much longer than a plain one.
            assert!(telem_planned_delta_ms(i) <= 200, "{} plans {}", quantity.name, telem_planned_delta_ms(i));
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
        assert_eq!(telem_start_ms(0), 0);
        assert_eq!(telem_start_ms(1), 32);
        assert_eq!(telem_start_ms(zero), 104);
    }

    #[test]
    fn selection_reads_only_the_shape_and_the_clock() {
        // Pair order: within one round every quantity's occurrences alternate
        // and the next round flips them; over two rounds each quantity has
        // as many arm-first workers as default-first ones.
        let len = TELEM_ROUND.len() as u64;
        for q in 0..TELEM_QUANTITIES.len() as u8 {
            let mut seen = Vec::new();
            for bin in 7 * len..9 * len {
                if TELEM_ROUND[(bin % len) as usize] == q {
                    seen.push(telem_arm_first(bin));
                }
            }
            let first = seen.iter().filter(|&&f| f).count();
            assert_eq!(2 * first, seen.len(), "quantity {q}");
            let per_round = seen.len() / 2;
            for (j, &f) in seen[..per_round].iter().enumerate() {
                assert_eq!(f, (j as u64 + 7) % 2 == 1, "quantity {q}");
            }
        }
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
        // A test binary is not the ranked worker: no arm can be selected,
        // and every split stays the tuned one.
        assert_eq!(telem_init(), None);
        set_arm(ARM_NO_LINGER);
        assert_eq!(arm(), 0);
        assert_eq!(telem_pair(), None);
        assert_eq!(telem_pause_cut(), std::time::Duration::ZERO);
        telem_set_coarse_split(3);
        assert_eq!(telem_split(4 + 16 * 5), 4 + 16 * 5);
        telem_set_coarse_split(usize::MAX);
    }

    #[test]
    fn encoding() {
        let ms = MS as i64;
        let v = TelemValue::Value;
        let zero = TELEM_QUANTITIES.iter().position(|q| matches!(q.kind, TelemKind::Zero)).unwrap();
        // Pair (width 24, valid 16): 0 sits mid-window, range is +-8 ms ("pair linger off post").
        assert_eq!(telem_encode_ns(1, v(0)), 8 * MS);
        assert_eq!(telem_encode_ns(1, v(-3 * ms)), 5 * MS);
        assert_eq!(telem_encode_ns(1, v(-9 * ms)), 17 * MS);
        assert_eq!(telem_encode_ns(1, v(8 * ms)), 19 * MS);
        assert_eq!(telem_encode_ns(1, TelemValue::Missing), 21 * MS);
        assert_eq!(telem_encode_ns(1, TelemValue::Differed), 23 * MS);
        // Pair (width 32, valid 24): range is +-12 ms ("pair split -1").
        assert_eq!(telem_encode_ns(0, v(-12 * ms)), 0);
        assert_eq!(telem_encode_ns(0, v(-8 * ms)), 4 * MS);
        assert_eq!(telem_encode_ns(0, v(11 * ms)), 23 * MS);
        // Zero (width 16): its own value only.
        assert_eq!(telem_encode_ns(zero, v(0)), 0);
        assert_eq!(telem_encode_ns(zero, TelemValue::Missing), 13 * MS);
        // Values.
        let last = |l: &str| match l {
            "seed read" => Some(1_000),
            _ => None,
        };
        assert_eq!(telem_value_ns(zero, last, TelemValue::Missing), v(0));
        assert_eq!(telem_value_ns(0, last, v(-7)), v(-7));
        assert_eq!(telem_value_ns(0, last, TelemValue::Differed), TelemValue::Differed);
        assert_eq!(telem_value_ns(1, last, TelemValue::Missing), TelemValue::Missing);
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
        // Halves.
        let join = |j: Option<u64>| move |label: &str| if label == "commit join end" { j } else { None };
        assert_eq!(telem_half_ns(Half::Post, 10, 110, join(Some(60))), Some(50));
        assert_eq!(telem_half_ns(Half::Post, 10, 110, join(Some(5))), None);
        assert_eq!(telem_half_ns(Half::Post, 10, 110, join(None)), None);
        assert_eq!(telem_half_ns(Half::Whole, 10, 110, join(None)), Some(100));
        // Medians.
        assert_eq!(telem_median(vec![]), None);
        assert_eq!(telem_median(vec![5]), Some(5));
        assert_eq!(telem_median(vec![9, -3, 4, 1]), Some(2));
        assert_eq!(telem_median(vec![9, -3, 4, 1, 100]), Some(4));
        // Splits: +1/16 keeps the fine part; NOFINE takes the coarse pick.
        assert_eq!(telem_split_for(4, ARM_SPLIT_PLUS1, usize::MAX), 5);
        assert_eq!(telem_split_for(0, ARM_SPLIT_PLUS1, usize::MAX), 1);
        assert_eq!(telem_split_for(3 + 16 * 7, ARM_SPLIT_PLUS1, usize::MAX), 4 + 16 * 7);
        assert_eq!(telem_split_for(3 + 16 * 7, ARM_NOFINE, 4), 4);
        assert_eq!(telem_split_for(3 + 16 * 7, ARM_NOFINE, usize::MAX), 3 + 16 * 7);
        assert_eq!(telem_split_for(3 + 16 * 7, ARM_NULL, 4), 3 + 16 * 7);
        // -1/16 toward the GPU keeps the fine part; a zero CPU share stays.
        assert_eq!(telem_split_for(4, ARM_SPLIT_MINUS1, usize::MAX), 3);
        assert_eq!(telem_split_for(3 + 16 * 7, ARM_SPLIT_MINUS1, usize::MAX), 2 + 16 * 7);
        assert_eq!(telem_split_for(16 * 7, ARM_SPLIT_MINUS1, usize::MAX), 16 * 7);
    }
}

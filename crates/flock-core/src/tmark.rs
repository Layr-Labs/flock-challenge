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
// worker. The run's score is spent on the readings.
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
//   statement, default path against the arm (one runtime switch flipped),
//   order alternating. The default path is the record's: the two
//   switch-gated pieces this tree adds are off unless an arm turns one on. The reading is the mean
//   of `t(arm) - t(default)` over one part of the proof. The `Extra`
//   quantity reports the mean whole time of such default-path extra proofs.
// - Proof bytes are untouched and every proof is verified by the harness as
//   usual. The seed read and the publication code are not modified: the
//   delay sits at the end of the prover's own prove body.
// - Job time: a telemetry worker skips the inherited start pause (whose only
//   purpose is to spread a scored job's trials over more wall time). Every
//   quantity's planned extra time is at least `TELEM_MARGIN_MS` shorter than
//   that pause, so every trial is planned at least that much shorter than the
//   same trial without telemetry (see `telem_pause_cut` and its test).
//
// Decoding a trial time `T` (ms): `x = T - c0 - TELEM_BASE_MS`, `c0` being
// the same for the `Zero` quantity (harness-side latency); the window holding
// `x` names the quantity; `v = x - start_i`. With `valid = width - 8`:
// `v < valid` is a reading (`Span`/`Extra`: `v + sub`; `Pair`: `v - valid / 2`),
// `valid + 1` = below range, `valid + 3` = above range, `valid + 5` = never
// recorded. The last 10 ms window means the proof was not ready in time.

/// Recorder armed for this process (every ranked worker of this tree).
static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Paired arms. The default path of a telemetry worker is the RECORD's: the
/// e-man07 kernel pieces are NOT in this tree; arms 1..=8 are unused. The
/// two neon pieces this tree carries on top of it are OFF
/// (see [`piece_on`]). Arms 12 and 13 turn one on for the extra untimed
/// proofs; the `NO_` arms turn an inherited mechanism
/// off.
pub const ARM_R2P_FUSED: u8 = 1;
pub const ARM_R2P_TAIL_FUSED: u8 = 2;
pub const ARM_ZC_K_ALLFUSED: u8 = 3;
pub const ARM_TRI_UNROLL2: u8 = 4;
pub const ARM_TRI_FOLD_SEQ: u8 = 5;
pub const ARM_ZC_AB_Z_M1: u8 = 6;
/// ZK needs Z_M1: this arm turns both on.
pub const ARM_ZC_AB_ZK: u8 = 7;
pub const ARM_BUNDLE: u8 = 8;
pub const ARM_NO_LINGER: u8 = 9;
pub const ARM_NO_GPU_WAIT_HELP: u8 = 10;
/// Changes nothing: the A/A noise floor of the paired readings.
pub const ARM_NULL: u8 = 11;
/// Neon pieces, not part of `ARM_BUNDLE`: the NTT prefetch encoding is not in
/// the record (off by default, on in its arm); the transpose PMULL folds are
/// (record 2c81ee2e: on by default, off in their arm).
pub const ARM_NTT_PREFETCH_IMM: u8 = 12;
pub const ARM_TRANSPOSE_PMULL: u8 = 13;

/// Whether the switch-gated piece `piece` runs in the proof now proving.
/// `tree_default` is the piece's own kill switch (on unless its variable is
/// set). A process without a telemetry quantity (tests, benches, any
/// non-ranked run) keeps that default. A telemetry worker runs the record's
/// path, i.e. the piece is off, except in the extra proofs of an arm that
/// turns it on.
#[inline]
pub fn piece_on(piece: u8, tree_default: bool) -> bool {
    if telem_slot().is_none() {
        return tree_default;
    }
    let arm = arm();
    tree_default
        && (arm == piece
            || (arm == ARM_BUNDLE && piece <= ARM_ZC_AB_ZK)
            || (piece == ARM_ZC_AB_Z_M1 && arm == ARM_ZC_AB_ZK))
}

/// Whether a piece the BASE already has (on by default) is switched off in the
/// proof now proving: only in the extra proofs of its own arm or of the
/// bundle arm of a telemetry worker.
#[inline]
pub fn piece_off(piece: u8) -> bool {
    telem_slot().is_some() && {
        let arm = arm();
        arm == piece || arm == ARM_BUNDLE
    }
}

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

/// Which part of a proof a paired reading compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    /// Prove start to the end of the round-one AB precompute arm. (Not to
    /// the commit join: when the GPU arm ends last, the join does not move
    /// with the AB arm and would hide an AB-side piece.)
    Ab,
    /// Commit join to prove end (zerocheck, lincheck, opening).
    Post,
    /// The whole proof.
    Whole,
    /// Prove start to the end of the CPU part of the commitment NTT.
    Ntt,
    /// The opening alone (opening start to opening returned).
    Open,
}

/// What a quantity reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemKind {
    /// Always 0: calibrates the harness-side constant `c0`.
    Zero,
    /// Last `b` minus last `a` in the timed proof.
    Span(&'static str, &'static str),
    /// In-process `t(arm) - t(default)` over one half of the proof.
    Pair(u8, Half),
    /// Mean whole-proof time of the same extra, untimed proofs (default path
    /// only): says how close an in-process proof is to a timed one.
    Extra,
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
const J: &str = "commit join end";

/// The run's quantities. Those that run extra proofs come first: the proofs
/// then fit under the start pause they give back.
pub static TELEM_QUANTITIES: [TelemQuantity; 22] = [
    q("pair linger off", TelemKind::Pair(ARM_NO_LINGER, Half::Post), 24, 0),
    q("pair gpu_wait_help off", TelemKind::Pair(ARM_NO_GPU_WAIT_HELP, Half::Ab), 24, 0),
    q("pair ntt_prefetch_imm on", TelemKind::Pair(ARM_NTT_PREFETCH_IMM, Half::Ntt), 24, 0),
    q("pair transpose_pmull off", TelemKind::Pair(ARM_TRANSPOSE_PMULL, Half::Open), 24, 0),
    q("pair null ab", TelemKind::Pair(ARM_NULL, Half::Ab), 24, 0),
    q("pair null post", TelemKind::Pair(ARM_NULL, Half::Post), 24, 0),
    q("pair null whole", TelemKind::Pair(ARM_NULL, Half::Whole), 24, 0),
    q("extra proof total", TelemKind::Extra, 48, 95),
    q("zero", TelemKind::Zero, 16, 0),
    q("witgen end", TelemKind::Span(S, "witgen end"), 30, 0),
    q("cb2 gpu end", TelemKind::Span(S, "cb2 gpu end"), 52, 25),
    q("ab arm end", TelemKind::Span(S, "ab arm end"), 52, 20),
    q("cpu ntt done", TelemKind::Span(S, "hybrid cpu ntt done"), 52, 20),
    q("fine split wait", TelemKind::Span("fine wait start", "fine wait end"), 24, 0),
    q("split k_cpu16", TelemKind::Span("split base", "split k_cpu16"), 24, 0),
    q("commit root", TelemKind::Span(S, "commit arm root"), 52, 30),
    q("zerocheck round 1", TelemKind::Span(J, "zc r1 done"), 40, 0),
    q("zerocheck round 2", TelemKind::Span("zc r1 done", "zc r2 done"), 40, 0),
    q("zerocheck rest", TelemKind::Span("zc r2 done", "zc end"), 40, 0),
    q("lincheck", TelemKind::Span("zc end", "lc returned"), 30, 0),
    q("open", TelemKind::Span("open start", "open returned"), 40, 0),
    q("total", TelemKind::Span(S, "prove end"), 66, 85),
];

/// Order in which consecutive 6 s clock bins hand out quantities, repeats
/// spread apart. Slots per round: pair ntt_prefetch_imm on 7, pair
/// transpose_pmull off 7, pair null ab 6, pair null post 6, pair null whole 6,
/// pair linger off 4, pair gpu_wait_help off 4, split k_cpu16 2; every other
/// quantity 3.
pub static TELEM_ROUND: [u8; 84] = [
    0, 13, 5, 2, 18, 10, 4, 15, 7, 6, 20, 3, 12, 1, 17, 2, 5, 9, 4, 19, 0, 3, 11, 6, 16, 8, 2, 21,
    14, 13, 5, 18, 10, 1, 3, 4, 15, 7, 6, 20, 2, 12, 0, 17, 5, 9, 3, 4, 19, 11, 2, 6, 16, 8, 1, 21,
    13, 5, 3, 18, 10, 4, 2, 0, 15, 7, 6, 20, 12, 14, 17, 3, 5, 9, 2, 1, 4, 19, 11, 6, 16, 3, 8, 21,
];

/// The earliest return of any telemetry proof (ms after the seed read).
pub const TELEM_BASE_MS: u64 = 150;
/// Guard at the end of each window (sentinels live there).
pub const TELEM_GUARD_MS: u64 = 8;
/// Width of the final "proof was not ready in time" window.
pub const TELEM_OVERRUN_MS: u64 = 10;
/// Pairs of extra proofs per `Pair` worker.
pub const TELEM_PAIRS: usize = 2;
/// Budgeted time of one extra proof (the record proves in about 109 ms).
pub const TELEM_PROOF_EST_MS: u64 = 125;
/// Every telemetry trial is planned at least this much SHORTER than a plain
/// one (the job's hosted time limit is tight).
pub const TELEM_MARGIN_MS: u64 = 309;
/// Clock bin of the round-robin. Longer than a trial's cycle (about 5.2 to
/// 5.6 s in this tree), so consecutive workers take the same or the next
/// entry of the round and none is skipped. (A local gate build, whose
/// log2 = 8 trials cycle in about 2 s, uses 2 s.)
pub const TELEM_BIN_MS: u64 = if TELEM_GATE { 2000 } else { 6000 };
/// The inherited start pause this budget was written against (ms); a test
/// in `lib.rs` pins it to `RANKED_START_PAUSE`.
pub const TELEM_PAUSE_MS: u64 = 1300;

/// Local gate builds only (`FLOCK_TELEM_GATE` set in the build environment;
/// the hosted build has no such variable): telemetry arms at log2 = 8 instead
/// of 18 and every extra proof is verified in-process.
pub const TELEM_GATE: bool = option_env!("FLOCK_TELEM_GATE").is_some();
const TELEM_LOG2: &str = if TELEM_GATE { "8" } else { "18" };

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

/// Pure: the offset inside its window (ns) that encodes `value` (ns; `None`
/// = never recorded) for quantity `i`.
pub fn telem_encode_ns(i: usize, value: Option<i64>) -> u64 {
    let quantity = &TELEM_QUANTITIES[i];
    let valid = ((quantity.width_ms - TELEM_GUARD_MS) * MS) as i64;
    let Some(value) = value else {
        return valid as u64 + 5 * MS;
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

/// Pure: the value (ns) of quantity `i` from a mark lookup and the stored
/// results of the extra proofs.
pub fn telem_value_ns(
    i: usize,
    last: impl Fn(&str) -> Option<u64>,
    pair_diff: Option<i64>,
    extra_total: Option<i64>,
) -> Option<i64> {
    match TELEM_QUANTITIES[i].kind {
        TelemKind::Zero => Some(0),
        TelemKind::Span(a, b) => match (last(a), last(b)) {
            (Some(ta), Some(tb)) if tb >= ta => Some((tb - ta) as i64),
            _ => None,
        },
        TelemKind::Pair(..) => pair_diff,
        TelemKind::Extra => extra_total,
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

/// Pure: the most wall time (ms) quantity `i` plans to add to its worker:
/// the latest return of its window, plus the extra proofs of a pair. (The
/// proof's own ~109 ms, which the return time contains, is not subtracted:
/// that is slack on top of the margin.)
pub const fn telem_planned_extra_ms(i: usize) -> u64 {
    let hold = TELEM_BASE_MS + telem_start_ms(i) + TELEM_QUANTITIES[i].width_ms;
    match TELEM_QUANTITIES[i].kind {
        TelemKind::Pair(..) | TelemKind::Extra => {
            hold + (2 * TELEM_PAIRS as u64 + 1) * TELEM_PROOF_EST_MS
        }
        _ => hold,
    }
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
/// Mean paired difference in ns; `i64::MIN` = not recorded.
static PAIR_DIFF: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
/// Mean whole time of the default-path extra proofs in ns; `i64::MIN` = none.
static EXTRA_TOTAL: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);

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
/// proofs (`Extra` runs the null arm).
pub fn telem_pair() -> Option<(u8, Half, usize, bool)> {
    let arm_first = telem_arm_first(TELEM_BIN.load(Ordering::Relaxed));
    match TELEM_QUANTITIES[telem_slot()?].kind {
        TelemKind::Pair(arm, half) => Some((arm, half, TELEM_PAIRS, arm_first)),
        TelemKind::Extra => Some((ARM_NULL, Half::Post, TELEM_PAIRS, arm_first)),
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
        Half::Ab => (at("ab arm end")? - t0) as i64,
        Half::Post => (t1 - at(J)?) as i64,
        Half::Whole => t1.checked_sub(t0)? as i64,
        Half::Ntt => (at("hybrid cpu ntt done")? - t0) as i64,
        Half::Open => at("open returned")?.checked_sub(at("open start")?)? as i64,
    })
}

/// The part time of the extra proof that just ran from `t0` to `t1`.
pub fn telem_half_of_last_proof(half: Half, t0: u64, t1: u64) -> Option<i64> {
    telem_half_ns(half, t0, t1, find_last)
}

/// Store this worker's extra-proof results: the mean of `t(arm) -
/// t(default)` over its pairs and the mean whole time of its default-path
/// proofs (each `None` when nothing was measured).
pub fn telem_pair_record(mean_diff_ns: Option<i64>, mean_default_total_ns: Option<i64>) {
    if let Some(d) = mean_diff_ns {
        PAIR_DIFF.store(d.max(i64::MIN + 1), Ordering::Relaxed);
    }
    if let Some(t) = mean_default_total_ns {
        EXTRA_TOTAL.store(t.max(i64::MIN + 1), Ordering::Relaxed);
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
    let stored = |v: &std::sync::atomic::AtomicI64| match v.load(Ordering::Relaxed) {
        i64::MIN => None,
        d => Some(d),
    };
    let value = telem_value_ns(i, find_last, stored(&PAIR_DIFF), stored(&EXTRA_TOTAL));
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
        let extra = TELEM_QUANTITIES.iter().position(|q| matches!(q.kind, TelemKind::Extra)).unwrap();
        for (i, quantity) in TELEM_QUANTITIES.iter().enumerate() {
            let want = match quantity.name {
                "pair ntt_prefetch_imm on" => 7,
                "pair transpose_pmull off" => 7,
                "pair null ab" => 6,
                "pair null post" => 6,
                "pair null whole" => 6,
                "pair linger off" => 4,
                "pair gpu_wait_help off" => 4,
                "split k_cpu16" => 2,
                _ => 3,
            };
            // Quantities that run extra proofs sit before all others.
            if matches!(quantity.kind, TelemKind::Pair(..) | TelemKind::Extra) {
                assert!(i <= extra, "{}", quantity.name);
            }
            assert_eq!(hits[i], want, "{}", quantity.name);
            assert!(quantity.width_ms > TELEM_GUARD_MS);
            // The job-time budget: no telemetry trial is planned longer than a
            // plain trial, each is planned at least the margin shorter.
            assert!(
                telem_planned_extra_ms(i) + TELEM_MARGIN_MS <= TELEM_PAUSE_MS,
                "{} plans {} ms",
                quantity.name,
                telem_planned_extra_ms(i)
            );
        }
        assert_eq!(telem_start_ms(0), 0);
        assert_eq!(telem_start_ms(1), 24);
        assert_eq!(telem_start_ms(extra + 1), extra as u64 * 24 + 48);
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
        // A test binary is not the ranked worker.
        assert_eq!(telem_init(), None);
        set_arm(ARM_NO_LINGER);
        assert_eq!(arm(), 0);
        assert_eq!(telem_pair(), None);
        assert_eq!(telem_pause_cut(), std::time::Duration::ZERO);
    }

    #[test]
    fn encoding() {
        let ms = MS as i64;
        let extra = TELEM_QUANTITIES.iter().position(|q| matches!(q.kind, TelemKind::Extra)).unwrap();
        let zero = extra + 1;
        // Pair (width 24, valid 16): 0 sits mid-window, range is +-8 ms.
        assert_eq!(telem_encode_ns(0, Some(0)), 8 * MS);
        assert_eq!(telem_encode_ns(0, Some(-3 * ms)), 5 * MS);
        assert_eq!(telem_encode_ns(0, Some(-9 * ms)), 17 * MS);
        assert_eq!(telem_encode_ns(0, Some(8 * ms)), 19 * MS);
        assert_eq!(telem_encode_ns(0, None), 21 * MS);
        // Extra proof total (width 48, sub 95).
        assert_eq!(telem_encode_ns(extra, Some(110 * ms)), 15 * MS);
        assert_eq!(telem_value_ns(extra, |_| None, None, Some(7)), Some(7));
        // Span with an offset: "total" (width 66, sub 85).
        let total = TELEM_QUANTITIES.len() - 1;
        assert_eq!(telem_encode_ns(total, Some(109 * ms)), 24 * MS);
        assert_eq!(telem_encode_ns(total, Some(80 * ms)), 59 * MS);
        assert_eq!(telem_encode_ns(total, Some(150 * ms)), 61 * MS);
        // Values.
        let last = |l: &str| match l {
            "seed read" => Some(1_000),
            "witgen end" => Some(4_001_000),
            _ => None,
        };
        assert_eq!(telem_value_ns(zero, last, None, None), Some(0));
        assert_eq!(telem_value_ns(zero + 1, last, None, None), Some(4_000_000));
        assert_eq!(telem_value_ns(zero + 2, last, None, None), None);
        assert_eq!(telem_value_ns(0, last, Some(-7), None), Some(-7));
        assert_eq!(telem_value_ns(0, last, None, Some(5)), None);
        // Deadlines: in time, and late (overrun window after the last one).
        let start17 = telem_start_ms(zero + 1);
        assert_eq!(
            telem_deadline_ns(5, zero + 1, 4 * MS, 120 * MS),
            5 + (TELEM_BASE_MS + start17 + 4) * MS
        );
        let all = telem_start_ms(TELEM_QUANTITIES.len());
        assert_eq!(
            telem_deadline_ns(5, 0, 4 * MS, 160 * MS),
            5 + (TELEM_BASE_MS + all + TELEM_OVERRUN_MS / 2) * MS
        );
        // Halves.
        let marks = |ab: Option<u64>, join: Option<u64>| {
            move |label: &str| match label {
                "ab arm end" => ab,
                "commit join end" => join,
                "hybrid cpu ntt done" => Some(45),
                "open start" => Some(90),
                "open returned" => Some(105),
                _ => None,
            }
        };
        assert_eq!(telem_half_ns(Half::Ab, 10, 110, marks(Some(50), Some(60))), Some(40));
        assert_eq!(telem_half_ns(Half::Post, 10, 110, marks(Some(50), Some(60))), Some(50));
        assert_eq!(telem_half_ns(Half::Post, 10, 110, marks(Some(50), Some(5))), None);
        assert_eq!(telem_half_ns(Half::Ab, 10, 110, marks(None, Some(60))), None);
        assert_eq!(telem_half_ns(Half::Ab, 10, 110, marks(Some(111), Some(60))), None);
        assert_eq!(telem_half_ns(Half::Whole, 10, 110, |_| None), Some(100));
        assert_eq!(telem_half_ns(Half::Ntt, 10, 110, marks(None, None)), Some(35));
        assert_eq!(telem_half_ns(Half::Open, 10, 110, marks(None, None)), Some(15));
        assert_eq!(telem_half_ns(Half::Open, 10, 100, marks(None, None)), None);
        assert_eq!(telem_half_ns(Half::Ntt, 10, 110, |_| None), None);
        // No quantity in a test binary: a piece keeps the tree's own default.
        assert!(piece_on(ARM_R2P_FUSED, true));
        assert!(!piece_on(ARM_R2P_FUSED, false));
    }
}

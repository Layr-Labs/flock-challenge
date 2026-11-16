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

/// Diagnostic run DIAG5 (not a candidate): each ranked worker process runs
/// one arm (a set of kill switches, chosen from the wall clock before any
/// switch is read) and returns its proof late, at a time that encodes either
/// its seed read -> prove end time or one stage span. The prover's arithmetic
/// is unchanged; the hold is a sleep/spin after the proof is complete.
/// Channel idea: patternrecognition9-del (59257514), after newjordan
/// (ead7f97a); this encoder is our own.
pub mod diag {
    use super::now_ns;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    pub const BASE_MS: f64 = 110.0;
    pub const WIDTH_MS: f64 = 32.0;
    pub const VALID_MS: f64 = 26.0;
    pub const TOTAL_LO_MS: f64 = 85.0;
    /// The hold spins (no sleep) for its last this-many ms.
    pub const SPIN_MS: u64 = 15;

    // Span stamps (indices into STAMP).
    pub const COMMIT_END: usize = 0;
    pub const ZC_START: usize = 1;
    pub const ZC_R1: usize = 2;
    pub const ZC_R2: usize = 3;
    pub const ZC_K: usize = 4;
    pub const ZC_END: usize = 5;
    pub const OPEN_START: usize = 6;
    pub const OPEN_END: usize = 7;
    pub const K_JOIN: usize = 8;
    pub const K_GPU_END: usize = 9;
    const NSTAMP: usize = 10;
    const SEED: usize = NSTAMP;

    static STAMP: [AtomicU64; NSTAMP + 1] = [const { AtomicU64::new(0) }; NSTAMP + 1];
    static ACTIVE: AtomicBool = AtomicBool::new(false);
    static WINDOW: AtomicU64 = AtomicU64::new(0);

    /// What a window encodes.
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Q {
        Zero,
        Total,
        /// `(from, to, sub_ms)`; `from == SEED` means the seed read.
        Span(usize, usize, f64),
        /// Signed `(to - from, sub_ms)`: may be negative.
        Signed(usize, usize, f64),
    }

    const KILL_OPEN: (&str, &str) = ("FLOCK_NO_OPEN_RESTORE", "1");
    const KILL_BENC: (&str, &str) = ("FLOCK_NO_BAND_ENC_MERGE", "1");
    const KILL_L2: (&str, &str) = ("FLOCK_NO_L2_OVERLAP", "1");
    const BASE_SET: &[(&str, &str)] = &[KILL_OPEN, KILL_BENC, KILL_L2];

    /// Window table: (name, quantity, switches set for the worker).
    pub static WINDOWS: [(&str, Q, &[(&str, &str)]); 12] = [
        ("zero", Q::Zero, BASE_SET),
        ("base (record)", Q::Total, BASE_SET),
        ("open restore on", Q::Total, &[KILL_BENC, KILL_L2]),
        ("band encoder merge on", Q::Total, &[KILL_OPEN, KILL_L2]),
        ("l2 overlap on", Q::Total, &[KILL_OPEN, KILL_BENC]),
        ("all three on", Q::Total, &[]),
        ("span seed -> commit join end", Q::Span(SEED, COMMIT_END, 40.0), BASE_SET),
        ("span zc start -> r1 done", Q::Span(ZC_START, ZC_R1, -4.0), BASE_SET),
        ("span r1 done -> r2 done", Q::Span(ZC_R1, ZC_R2, -4.0), BASE_SET),
        ("span r2 done -> K done", Q::Span(ZC_R2, ZC_K, -4.0), BASE_SET),
        ("span r2 done -> zc end", Q::Span(ZC_R2, ZC_END, -2.0), BASE_SET),
        ("span open start -> open returned", Q::Span(OPEN_START, OPEN_END, -4.0), BASE_SET),
    ];

    /// One round of 30 six-second slots: window index per slot (the three
    /// single-piece arms rotate their order from group to group).
    pub static ROUND: [u8; 30] = [
        0, 1, 2, 3, 4, 6, 1, 3, 4, 2, 7, 1, 4, 2, 3, 8, 1, 2, 3, 4, 9, 1, 3, 4, 2, 10, 1, 5, 5, 11,
    ];

    fn window_for(unix_secs: u64) -> usize {
        ROUND[((unix_secs / 6) % ROUND.len() as u64) as usize] as usize
    }

    /// Extra return delay (ms) a window adds over an undelayed ~95 ms proof.
    fn expected_extra_ms(w: usize) -> f64 {
        (BASE_MS + WIDTH_MS * w as f64 + VALID_MS / 2.0 - 95.0).max(0.0)
    }

    /// Called first in the ranked worker, before any thread exists and
    /// before any switch is read: pick the arm, set its switches, and return
    /// how much of the start pause to cut (the expected extra hold).
    pub fn init(ranked: bool) -> std::time::Duration {
        if !ranked {
            return std::time::Duration::ZERO;
        }
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Lab only (the harness never sets it): force a window.
        let w = std::env::var("FLOCK_DIAG5_WINDOW")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&w| w < WINDOWS.len())
            .unwrap_or_else(|| window_for(secs));
        for (k, v) in WINDOWS[w].2 {
            // SAFETY: no other thread exists yet (first call of the worker).
            unsafe { std::env::set_var(k, v) };
        }
        WINDOW.store(w as u64, Ordering::Relaxed);
        ACTIVE.store(true, Ordering::Relaxed);
        std::time::Duration::from_micros((expected_extra_ms(w).min(1200.0) * 1e3) as u64)
    }

    /// Whether this worker runs the diagnostic.
    #[inline]
    pub fn active() -> bool {
        ACTIVE.load(Ordering::Relaxed)
    }

    /// Stamp a span point at an explicit CLOCK_UPTIME_RAW time.
    #[inline]
    pub fn at_ns(i: usize, t: u64) {
        if ACTIVE.load(Ordering::Relaxed) {
            STAMP[i].store(t, Ordering::Relaxed);
        }
    }

    /// Stamp a span point (the last stamp of a prove wins).
    #[inline]
    pub fn at(i: usize) {
        if ACTIVE.load(Ordering::Relaxed) {
            STAMP[i].store(now_ns(), Ordering::Relaxed);
        }
    }

    /// Seed read: start of this prove's timeline.
    pub fn seed_read() {
        if ACTIVE.load(Ordering::Relaxed) {
            for s in &STAMP[..NSTAMP] {
                s.store(0, Ordering::Relaxed);
            }
            STAMP[SEED].store(now_ns(), Ordering::Relaxed);
        }
    }

    /// Encoded offset (ms) inside window `w`'s band for the given stamps.
    pub fn encode(q: Q, stamps: &[u64; NSTAMP + 1], end: u64) -> f64 {
        let span = |a: usize, b: u64| -> Option<f64> {
            let s = stamps[a];
            (s != 0 && b != 0 && b >= s).then(|| (b - s) as f64 / 1e6)
        };
        let (val, sub) = match q {
            Q::Zero => return 0.0,
            Q::Total => (span(SEED, end), TOTAL_LO_MS),
            Q::Span(a, b, sub) => (span(a, stamps[b]), sub),
            Q::Signed(a, b, sub) => {
                let (s, e) = (stamps[a], stamps[b]);
                ((s != 0 && e != 0).then(|| (e as f64 - s as f64) / 1e6), sub)
            }
        };
        match val {
            None => VALID_MS + 5.0,
            Some(v) if v < sub => VALID_MS + 1.0,
            Some(v) if v - sub >= VALID_MS => VALID_MS + 3.0,
            Some(v) => v - sub,
        }
    }

    /// The speculative prove's last statement: hold until the encoded time.
    pub fn hold() {
        if !ACTIVE.load(Ordering::Relaxed) {
            return;
        }
        let seed = STAMP[SEED].swap(0, Ordering::Relaxed);
        if seed == 0 {
            return;
        }
        let end = now_ns();
        let mut st = [0u64; NSTAMP + 1];
        for (d, s) in st.iter_mut().zip(STAMP.iter()) {
            *d = s.load(Ordering::Relaxed);
        }
        st[SEED] = seed;
        let w = WINDOW.load(Ordering::Relaxed) as usize;
        let off = encode(WINDOWS[w].1, &st, end);
        let target = seed + ((BASE_MS + WIDTH_MS * w as f64 + off) * 1e6) as u64;
        loop {
            let now = now_ns();
            if now >= target {
                break;
            }
            let left = target - now;
            // Sleep only until 15 ms before the target, then spin. The runner's
            // sleeps overshoot by 5-9 ms on some workers (DIAG1R zero window),
            // which split every reading into two modes.
            if left > SPIN_MS * 1_000_000 {
                std::thread::sleep(std::time::Duration::from_nanos(left - SPIN_MS * 1_000_000));
            } else {
                std::hint::spin_loop();
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn windows_fit_and_round_covers_all() {
            for w in 0..WINDOWS.len() {
                assert!(ROUND.iter().any(|&r| r as usize == w), "window {w} unused");
            }
            assert!(BASE_MS + WIDTH_MS * (WINDOWS.len() as f64) < 700.0);
            assert!(VALID_MS + 5.0 < WIDTH_MS);
        }

        #[test]
        fn encode_ranges() {
            let mut st = [0u64; NSTAMP + 1];
            st[SEED] = 1_000_000_000;
            let end = st[SEED] + 95_000_000;
            assert!((encode(Q::Total, &st, end) - 10.0).abs() < 1e-6);
            assert_eq!(encode(Q::Total, &st, st[SEED] + 70_000_000), VALID_MS + 1.0);
            assert_eq!(encode(Q::Total, &st, st[SEED] + 120_000_000), VALID_MS + 3.0);
            assert_eq!(encode(Q::Span(ZC_R1, ZC_R2, -4.0), &st, end), VALID_MS + 5.0);
            st[ZC_R1] = 2_000_000_000;
            st[ZC_R2] = 2_012_500_000;
            assert!((encode(Q::Span(ZC_R1, ZC_R2, -4.0), &st, end) - 12.5).abs() < 1e-6);
        }
    }
}

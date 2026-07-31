//! Process-global pool for the prover's large transient `F128` buffers.
//!
//! Each prove allocates, faults in, and frees several 64–128 MB vectors
//! (the RS codeword, the round-2 fold outputs, the multilinear tail's
//! ping-pong scratch). The allocator returns such allocations to the OS on
//! free (`munmap`), so every prove re-pays soft page faults on first touch
//! and a single-threaded unmap on drop — a few ms per prove at m = 29 that
//! no kernel tuning can parallelize away.
//!
//! The pool recycles those buffers across phases and across proves: `take`
//! hands out a previously-used buffer when one with enough capacity exists,
//! `give` returns a buffer for later reuse. Contents are NOT cleared —
//! `take` has the same write-before-read contract as
//! [`crate::alloc_uninit_vec`].
//!
//! Steady-state retention is bounded by [`MAX_POOLED`] buffers (~640 MB for
//! the m = 29 prove set). Call [`clear`] to release everything to the OS,
//! e.g. after the last prove of a batch.

use crate::field::F128;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static POOL: Mutex<Vec<Vec<F128>>> = Mutex::new(Vec::new());

// ---------------------------------------------------------------------------
// Checkout instrumentation (diagnostics only, off unless FLOCK_POOL_STATS=1)
// ---------------------------------------------------------------------------
//
// `prewarm_prover`'s buffer counts were originally derived for m = 29 and were
// never re-derived for the ranked m = 32 shape, where they park 10.5 GiB. To
// size them from evidence rather than guesswork we need the actual *peak
// simultaneous checkout* per capacity class. Track live checkouts keyed by
// floor(log2(capacity)) and remember the high-water mark; dump on request.
//
// The hot path is one relaxed atomic load when disabled.

/// Largest capacity class we bucket (2^63 F128 is unreachable; 40 is plenty).
const N_CLASSES: usize = 40;

static STATS_ON: AtomicBool = AtomicBool::new(false);
static STATS_INIT: std::sync::Once = std::sync::Once::new();
static LIVE: [AtomicUsize; N_CLASSES] = [const { AtomicUsize::new(0) }; N_CLASSES];
static PEAK: [AtomicUsize; N_CLASSES] = [const { AtomicUsize::new(0) }; N_CLASSES];

#[inline]
fn stats_enabled() -> bool {
    STATS_INIT.call_once(|| {
        let on = std::env::var("FLOCK_POOL_STATS").is_ok_and(|v| v != "0" && !v.is_empty());
        STATS_ON.store(on, Ordering::Relaxed);
    });
    STATS_ON.load(Ordering::Relaxed)
}

#[inline]
fn class_of(capacity: usize) -> usize {
    if capacity == 0 {
        0
    } else {
        (usize::BITS - 1 - capacity.leading_zeros()) as usize
    }
}

#[inline]
fn note_take(capacity: usize) {
    if !stats_enabled() {
        return;
    }
    let c = class_of(capacity).min(N_CLASSES - 1);
    let live = LIVE[c].fetch_add(1, Ordering::Relaxed) + 1;
    PEAK[c].fetch_max(live, Ordering::Relaxed);
}

#[inline]
fn note_give(capacity: usize) {
    if !stats_enabled() {
        return;
    }
    let c = class_of(capacity).min(N_CLASSES - 1);
    // `give` can receive buffers that were never `take`n (a Vec built
    // elsewhere and donated). Saturate at 0 rather than wrapping.
    let _ = LIVE[c].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_sub(1))
    });
}

/// Print the peak-simultaneous-checkout high-water mark per capacity class,
/// plus the pool's current residency. No-op unless `FLOCK_POOL_STATS` is set.
/// Diagnostics for right-sizing [`prewarm_prover`].
pub fn dump_stats(tag: &str) {
    if !stats_enabled() {
        return;
    }
    let pool = POOL.lock().unwrap();
    let mut resident = 0usize;
    let mut pool_by_class = [0usize; N_CLASSES];
    for v in pool.iter() {
        resident += v.capacity() * core::mem::size_of::<F128>();
        pool_by_class[class_of(v.capacity()).min(N_CLASSES - 1)] += 1;
    }
    let pool_len = pool.len();
    drop(pool);
    eprintln!("[pool:{tag}] entries={pool_len} resident={:.2} GiB", resident as f64 / (1u64 << 30) as f64);
    eprintln!("[pool:{tag}]  class      bytes  peak_live  in_pool");
    let mut peak_bytes = 0usize;
    for c in 0..N_CLASSES {
        let p = PEAK[c].load(Ordering::Relaxed);
        if p == 0 && pool_by_class[c] == 0 {
            continue;
        }
        let bytes = (1usize << c) * core::mem::size_of::<F128>();
        peak_bytes += p * bytes;
        eprintln!(
            "[pool:{tag}]  2^{c:<3} {:>10} {p:>10} {:>8}",
            human(bytes),
            pool_by_class[c]
        );
    }
    eprintln!(
        "[pool:{tag}] peak simultaneous checkout (sum over classes) = {}",
        human(peak_bytes)
    );
}

/// Zero the live/peak counters. Call between proves so a leaked (dropped
/// rather than `give`n) buffer from an earlier prove doesn't inflate the next
/// prove's high-water mark. No-op unless `FLOCK_POOL_STATS` is set.
pub fn reset_stats() {
    if !stats_enabled() {
        return;
    }
    for c in 0..N_CLASSES {
        LIVE[c].store(0, Ordering::Relaxed);
        PEAK[c].store(0, Ordering::Relaxed);
    }
}

fn human(b: usize) -> String {
    if b >= 1 << 30 {
        format!("{:.2}G", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1}M", b as f64 / (1u64 << 20) as f64)
    } else {
        format!("{b}B")
    }
}

/// Max buffers retained. The m=29 prove cycle gives ~18 distinct buffers:
/// witness z/a/b, the L0 codeword, zerocheck's 2 fold outputs + 2 ping-pong
/// halves, ring-switch's per-claim rs_eq_ind vectors, b_combined, and
/// the PCS open's working buffers. Pooling ALL of the
/// open stage's transients matters beyond their own reuse: if they were
/// left to malloc while the earlier phases' buffers sat in the pool, the
/// open stage would fault fresh pages every prove (the pool denies malloc
/// the page reuse it would otherwise get from the freed early-phase
/// buffers) — measured as a +24% open_batch regression on M4 before this.
const MAX_POOLED: usize = 24;

/// Take a length-`n` `F128` vector, preferring a pooled buffer (smallest
/// capacity ≥ `n`); falls back to a fresh uninitialized allocation.
///
/// Contents are UNINITIALIZED in both cases — recycled buffers hold stale
/// data from a previous use. Caller MUST write every slot before reading it
/// (same contract as [`crate::alloc_uninit_vec`]).
pub fn take_f128(n: usize) -> Vec<F128> {
    if let Some(v) = try_take_f128(n) {
        return v;
    }
    let v = crate::alloc_uninit_vec(n);
    note_take(v.capacity());
    v
}

/// Pool-only variant of [`take_f128`]: returns `None` instead of falling
/// back to a fresh allocation. Lets callers branch on warm-vs-cold (e.g.
/// the commit prefault skips its page-touch thread when the pool can
/// supply an already-resident buffer).
pub(crate) fn try_take_f128(n: usize) -> Option<Vec<F128>> {
    let mut pool = POOL.lock().unwrap();
    let mut best: Option<usize> = None;
    for (i, v) in pool.iter().enumerate() {
        if v.capacity() >= n && best.is_none_or(|b| v.capacity() < pool[b].capacity()) {
            best = Some(i);
        }
    }
    if let Some(i) = best {
        let mut v = pool.swap_remove(i);
        drop(pool);
        v.clear();
        // SAFETY: capacity ≥ n was checked above; F128: Copy (no Drop), so
        // exposing uninit/stale elements is sound to *hold* — the caller
        // upholds write-before-read per this function's contract.
        unsafe { v.set_len(n) };
        note_take(v.capacity());
        return Some(v);
    }
    None
}

/// Return a buffer to the pool for reuse. When the pool is full, the
/// smallest-capacity buffer is evicted (large buffers are the expensive ones
/// to re-fault; a run that ramps problem sizes upward must not get its big
/// buffers crowded out by stale small ones).
pub fn give_f128(v: Vec<F128>) {
    if v.capacity() == 0 {
        return;
    }
    note_give(v.capacity());
    let mut pool = POOL.lock().unwrap();
    pool.push(v);
    if pool.len() > MAX_POOLED {
        let smallest = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| v.capacity())
            .map(|(i, _)| i)
            .expect("pool non-empty");
        pool.swap_remove(smallest);
    }
}

/// Pre-warm the pool for proves at witness size `2^m`: allocate and
/// first-touch the full prove-cycle buffer set once, in parallel, then park
/// it in the pool. Called from the per-hash Setup constructors, this moves
/// ALL page-fault cost off the prove path — including the first prove — so
/// proving performs no memory-management syscalls on any machine. (This is
/// the machine-independent alternative to overlapping the faults with other
/// work: a race between fault cost and the hiding window flips sign across
/// machines; eliminated work doesn't.)
///
/// ## Sizing
///
/// The counts below are derived from the **measured** peak simultaneous
/// checkout, not from a static reading of the code. Run any prover under
/// `FLOCK_POOL_STATS=1` and read the `peak_live` column of
/// [`dump_stats`]: at m = 32 (the 2^18-BLAKE3 shape) the true peak is
/// **3** buffers of the 2^(m-6) class and **6** of the 2^(m-7) class = 6.0 GiB.
///
/// The historical counts (5 large + 11 small) were derived for m = 29, where
/// they park ~1.1 GB. They were never re-derived for the 8x larger ranked
/// shape, where the same constants park **10.5 GiB** — roughly 1.9x the
/// working set, on a 36 GB machine that also runs a trusted verifier doing
/// its own multi-GiB witness regeneration between trials. Over-retention
/// there is not free: it is what pushes the OS into purging the `MADV_FREE`d
/// pages that libmalloc's large cache would otherwise hand straight back.
///
/// We provision **peak + 1 spare per class**. The spare matters because
/// [`try_take_f128`] returns the smallest capacity ≥ n: once a class is
/// exhausted, requests silently promote to the next class up and can cascade
/// into an [`crate::alloc_uninit_vec`] fallback, re-introducing exactly the
/// faults the prewarm exists to remove. Under-provisioning is therefore
/// strictly worse than over-provisioning — verify any change to these counts
/// by checking that `/usr/bin/time -l` page reclaims do **not** rise.
///
/// The set (sizes in F128s): 2^(m-6)-class — L0 codeword, zerocheck round-2
/// a/b, open-stage codeword ping-pong; 2^(m-7)-class — witness z/a/b,
/// zerocheck tail ping-pong, open-stage transients, rs_eq_ind, b_combined,
/// and the Ligerito ladder's two ping-pong scratch buffers. Release with
/// [`clear`].
pub fn prewarm_prover(m: usize) {
    use rayon::prelude::*;
    if m < 7 {
        return;
    }
    /// Peak-concurrent 2^(m-6) checkouts (3, measured) + 1 spare.
    const N_LARGE: usize = 4;
    /// Peak-concurrent 2^(m-7) checkouts (6, measured) + 1 spare.
    const N_SMALL: usize = 7;
    let small = 1usize << (m - 7);
    let large = 1usize << (m - 6);
    let mut bufs: Vec<Vec<F128>> = Vec::new();
    for _ in 0..N_LARGE {
        bufs.push(take_f128(large));
    }
    for _ in 0..N_SMALL {
        bufs.push(take_f128(small));
    }
    // First-touch every page of every buffer, all cores. Already-resident
    // (re-warmed) buffers cost a fast memset; fresh ones fault here, once.
    bufs.par_iter_mut().for_each(|b| {
        b.par_chunks_mut(1 << 16).for_each(|chunk| {
            // SAFETY: F128 is plain bytes (no Drop); zero is a valid pattern.
            unsafe { std::ptr::write_bytes(chunk.as_mut_ptr(), 0u8, chunk.len()) }
        });
    });
    for b in bufs {
        give_f128(b);
    }
    // The prewarm's own 16 take/give pairs are not a prove-cycle checkout —
    // don't let them pollute the high-water mark we size the prewarm from.
    reset_stats();
}

/// Release every pooled buffer back to the OS.
pub fn clear() {
    POOL.lock().unwrap().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_reuses_given_buffer() {
        clear();
        let mut v = take_f128(1024);
        for slot in v.iter_mut() {
            *slot = F128 { lo: 7, hi: 9 };
        }
        let ptr = v.as_ptr();
        give_f128(v);
        // Same capacity request gets the same allocation back.
        let v2 = take_f128(512);
        assert_eq!(v2.as_ptr(), ptr);
        assert_eq!(v2.len(), 512);
        clear();
    }

    #[test]
    fn pool_is_bounded() {
        clear();
        for _ in 0..(MAX_POOLED + 4) {
            give_f128(take_f128(16));
        }
        assert!(POOL.lock().unwrap().len() <= MAX_POOLED);
        clear();
    }
}

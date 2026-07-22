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
use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::Mutex;

static POOL: Mutex<Vec<Vec<F128>>> = Mutex::new(Vec::new());

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
    crate::alloc_uninit_vec(n)
}

fn take_pooled_f128(n: usize) -> Option<Vec<F128>> {
    let mut pool = POOL.lock().unwrap();
    let mut best: Option<usize> = None;
    for (i, v) in pool.iter().enumerate() {
        if v.capacity() >= n && best.is_none_or(|b| v.capacity() < pool[b].capacity()) {
            best = Some(i);
        }
    }
    best.map(|i| pool.swap_remove(i))
}

/// Pool-only variant of [`take_f128`]: returns `None` instead of falling
/// back to a fresh allocation. Lets callers branch on warm-vs-cold (e.g.
/// the commit prefault skips its page-touch thread when the pool can
/// supply an already-resident buffer).
pub(crate) fn try_take_f128(n: usize) -> Option<Vec<F128>> {
    if let Some(mut v) = take_pooled_f128(n) {
        v.clear();
        // SAFETY: capacity ≥ n was checked above; F128: Copy (no Drop), so
        // exposing uninit/stale elements is sound to *hold* — the caller
        // upholds write-before-read per this function's contract.
        unsafe { v.set_len(n) };
        return Some(v);
    }
    None
}

/// Exclusive owner for a pooled `F128` allocation while an overwrite producer
/// initializes it. Its slice is typed as `MaybeUninit`, so partitioning and
/// handing out disjoint mutable blocks is valid before any values are written.
pub struct F128OverwriteBuffer {
    buf: Vec<MaybeUninit<F128>>,
}

impl F128OverwriteBuffer {
    pub fn as_mut_slice(&mut self) -> &mut [MaybeUninit<F128>] {
        &mut self.buf
    }

    /// Convert the completed overwrite allocation back to `Vec<F128>`.
    ///
    /// # Safety
    ///
    /// Every element of `self` must have been fully initialized.
    pub unsafe fn assume_init(self) -> Vec<F128> {
        let mut buf = ManuallyDrop::new(self.buf);
        // SAFETY: guaranteed by the caller. `MaybeUninit<F128>` has the same
        // layout as `F128`, and the allocation metadata is preserved exactly.
        unsafe { Vec::from_raw_parts(buf.as_mut_ptr().cast::<F128>(), buf.len(), buf.capacity()) }
    }
}

/// Take an uninitialized length-`n` overwrite allocation without ever forming
/// a `Vec<F128>` or `&mut [F128]` over its unwritten elements. Pooled storage is
/// reused without clearing its bytes; fresh storage is allocation-only.
pub fn take_f128_overwrite(n: usize) -> F128OverwriteBuffer {
    let mut pooled = take_pooled_f128(n).unwrap_or_else(|| Vec::with_capacity(n));
    pooled.clear();
    let mut pooled = ManuallyDrop::new(pooled);
    // SAFETY: `MaybeUninit<F128>` has the same layout as `F128`; length zero
    // avoids asserting anything about prior contents. Setting the new length
    // is valid because uninitialized bytes are valid `MaybeUninit` values and
    // the selected/allotted capacity is at least n.
    let mut buf = unsafe {
        Vec::from_raw_parts(
            pooled.as_mut_ptr().cast::<MaybeUninit<F128>>(),
            0,
            pooled.capacity(),
        )
    };
    unsafe { buf.set_len(n) };
    F128OverwriteBuffer { buf }
}

/// Return a buffer to the pool for reuse. When the pool is full, the
/// smallest-capacity buffer is evicted (large buffers are the expensive ones
/// to re-fault; a run that ramps problem sizes upward must not get its big
/// buffers crowded out by stale small ones).
pub fn give_f128(v: Vec<F128>) {
    if v.capacity() == 0 {
        return;
    }
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

/// Seed the pool for proves at witness size `2^m` before the caller's warm
/// proof. The warm proof itself necessarily creates, fully writes, and
/// recycles all three `2^(m-6)` buffers needed by the measured proof: the L0
/// codeword and zerocheck's two round-2 outputs. Prewarming that size would
/// only add an earlier full-memory pass.
///
/// The measured peak needs five `2^(m-7)` buffers (witness z/a/b plus the two
/// zerocheck tail buffers). Seed seven because the warm proof consumes z and
/// b_combined in AArch64 Ligerito folds without returning their original
/// allocations to this pool; the other five remain resident for the measured
/// proof. At m = 32 this touches 3.5 GiB here instead of 10.5 GiB. Release with
/// [`clear`].
pub fn prewarm_prover(m: usize) {
    use rayon::prelude::*;
    if m < 7 {
        return;
    }
    let small = 1usize << (m - 7);
    let mut bufs: Vec<Vec<F128>> = Vec::with_capacity(7);
    for _ in 0..7 {
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
}

/// Release every pooled buffer back to the OS.
pub fn clear() {
    POOL.lock().unwrap().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn take_reuses_given_buffer() {
        let _serial = TEST_LOCK.lock().unwrap();
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
    fn overwrite_owner_reuses_without_exposing_uninitialized_f128() {
        let _serial = TEST_LOCK.lock().unwrap();
        clear();
        let mut out = take_f128_overwrite(1024);
        let ptr = out.as_mut_slice().as_mut_ptr();
        for slot in out.as_mut_slice() {
            slot.write(F128 { lo: 17, hi: 23 });
        }
        // SAFETY: every element was initialized by the loop above.
        let initialized = unsafe { out.assume_init() };
        assert!(
            initialized
                .iter()
                .all(|&slot| slot == (F128 { lo: 17, hi: 23 }))
        );
        give_f128(initialized);

        let mut reused = take_f128_overwrite(512);
        assert_eq!(reused.as_mut_slice().as_mut_ptr(), ptr);
        for slot in reused.as_mut_slice() {
            slot.write(F128 { lo: 29, hi: 31 });
        }
        // SAFETY: every element was initialized by the loop above.
        let initialized = unsafe { reused.assume_init() };
        assert!(
            initialized
                .iter()
                .all(|&slot| slot == (F128 { lo: 29, hi: 31 }))
        );
        clear();
    }

    #[test]
    fn partial_overwrite_owner_drops_without_entering_pool() {
        let _serial = TEST_LOCK.lock().unwrap();
        clear();
        let result = std::panic::catch_unwind(|| {
            let mut out = take_f128_overwrite(1024);
            out.as_mut_slice()[0].write(F128 { lo: 37, hi: 41 });
            panic!("simulated overwrite producer panic");
        });
        assert!(result.is_err());
        assert!(POOL.lock().unwrap().is_empty());
        clear();
    }

    #[test]
    fn pool_is_bounded() {
        let _serial = TEST_LOCK.lock().unwrap();
        clear();
        for _ in 0..(MAX_POOLED + 4) {
            give_f128(take_f128(16));
        }
        assert!(POOL.lock().unwrap().len() <= MAX_POOLED);
        clear();
    }
}

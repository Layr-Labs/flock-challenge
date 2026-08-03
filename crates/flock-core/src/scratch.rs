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
use core::ops::{Deref, DerefMut};
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
///
/// m=32 adds the SumcheckProver ping-pong spares: two half-size buffers per
/// recursive level (~12 more live classes), so the cap is raised to keep
/// steady-state below the eviction threshold — eviction must not fire in
/// steady state, or the small ladder buffers (the ones this pool exists to
/// keep resident) would be evicted first under the smallest-first policy.
const MAX_POOLED: usize = 48;

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
/// A traced ranked warm proof at m = 32 requests two 1 GiB F128 buffers, with
/// a six-buffer high-water at 512 MiB, then two simultaneous 256 MiB
/// zerocheck ping-pongs. Park those exact F128 classes so smallest-fit does
/// not hand the ping-pongs 512 MiB/1 GiB allocations. One 512 MiB byte buffer
/// covers the lincheck stripe; compact round two reuses its dead 512 MiB
/// F128 AB transform through [`ScratchBytes`] instead of needing another byte
/// allocation. Total ranked retention remains ~6 GiB, but with the requested
/// size classes rather than oversized substitutes. Release with [`clear`].
pub fn prewarm_prover(m: usize) {
    use rayon::prelude::*;
    if m < 7 {
        return;
    }
    let small = 1usize << (m - 7);
    let large = 1usize << (m - 6);
    let ping_pong = small / 2;
    let stripe_bytes = 1usize << (m - 3);
    let mut bufs: Vec<Vec<F128>> = Vec::new();
    for _ in 0..2 {
        bufs.push(take_f128(large));
    }
    for _ in 0..6 {
        bufs.push(take_f128(small));
    }
    if ping_pong != 0 {
        for _ in 0..2 {
            bufs.push(take_f128(ping_pong));
        }
    }
    let mut stripe = take_u8(stripe_bytes);
    // First-touch every page of every buffer, all cores. Already-resident
    // (re-warmed) buffers cost a fast memset; fresh ones fault here, once.
    bufs.par_iter_mut().for_each(|b| {
        b.par_chunks_mut(1 << 16).for_each(|chunk| {
            // SAFETY: F128 is plain bytes (no Drop); zero is a valid pattern.
            unsafe { std::ptr::write_bytes(chunk.as_mut_ptr(), 0u8, chunk.len()) }
        });
    });
    stripe.par_chunks_mut(1 << 20).for_each(|chunk| {
        // SAFETY: u8 has no invalid bit patterns and every byte is written.
        unsafe { std::ptr::write_bytes(chunk.as_mut_ptr(), 0u8, chunk.len()) }
    });
    for b in bufs {
        give_f128(b);
    }
    give_u8(stripe);
}

/// Release every pooled buffer back to the OS.
pub fn clear() {
    POOL.lock().unwrap().clear();
    POOL_U8.lock().unwrap().clear();
}

// ---------------------------------------------------------------------------
// Byte-buffer pool (the lincheck stripe).
//
// The BLAKE3 witness path builds a `2^(m-3)`-byte lincheck stripe (`512 MiB`
// at the ranked m=32) every prove and frees it after lincheck. Like the F128
// pool above, this keeps the stripe's pages resident across the worker's
// warm-up and timed proves instead of re-faulting ~32k pages per prove.
// Contents are NOT cleared; callers must write every byte before reading
// (the stripe transpose writes all of it — see
// `r1cs_hashes::common::drive_witness_packed_and_lincheck_impl`).

static POOL_U8: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Only a handful of stripe-class buffers ever exist at once.
const MAX_POOLED_U8: usize = 4;

/// Take a length-`n` byte vector, preferring a pooled buffer (smallest
/// capacity >= `n`); falls back to a fresh uninitialized allocation.
/// Contents are UNINITIALIZED in both cases (write-before-read contract,
/// same as [`take_f128`]).
pub fn take_u8(n: usize) -> Vec<u8> {
    {
        let mut pool = POOL_U8.lock().unwrap();
        let mut best: Option<usize> = None;
        for (i, v) in pool.iter().enumerate() {
            if v.capacity() >= n && best.is_none_or(|b| v.capacity() < pool[b].capacity()) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            let mut v = pool.swap_remove(i);
            // SAFETY: capacity >= n was checked above; u8: Copy (no Drop), so
            // exposing stale bytes is sound to *hold* — the caller upholds
            // write-before-read per this function's contract.
            unsafe { v.set_len(n) };
            return v;
        }
    }
    crate::alloc_uninit_vec(n)
}

/// Return a byte buffer to the pool for reuse (smallest-first eviction when
/// full, same policy as the F128 pool).
pub fn give_u8(v: Vec<u8>) {
    if v.capacity() == 0 {
        return;
    }
    let mut pool = POOL_U8.lock().unwrap();
    pool.push(v);
    if pool.len() > MAX_POOLED_U8 {
        let smallest = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| v.capacity())
            .map(|(i, _)| i)
            .expect("pool non-empty");
        pool.swap_remove(smallest);
    }
}

/// Byte-addressable scratch that retains the allocation's original element
/// type, alignment, and deallocation layout.
///
/// This is the sound seam for donating an initialized `Vec<F128>` to a
/// byte-oriented phase. It deliberately does not rebuild a `Vec<u8>` from raw
/// parts: doing so would deallocate the 16-byte-aligned F128 allocation with a
/// one-byte-aligned layout. The backing is returned to its matching pool by
/// [`Self::recycle`].
pub struct ScratchBytes {
    backing: ScratchBytesBacking,
}

enum ScratchBytesBacking {
    U8(Vec<u8>),
    F128(Vec<F128>),
}

impl ScratchBytes {
    /// Take ordinary byte storage from the byte pool.
    pub fn take(n: usize) -> Self {
        Self {
            backing: ScratchBytesBacking::U8(take_u8(n)),
        }
    }

    /// Donate fully initialized F128 storage while preserving its allocation
    /// layout. Every F128 bit pattern is valid, so its object representation
    /// may be viewed and overwritten as bytes.
    pub(crate) fn from_initialized_f128(storage: Vec<F128>) -> Self {
        Self {
            backing: ScratchBytesBacking::F128(storage),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.backing {
            ScratchBytesBacking::U8(v) => v.len(),
            ScratchBytesBacking::F128(v) => v.len() * core::mem::size_of::<F128>(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the allocation to the pool matching its original layout.
    pub fn recycle(self) {
        match self.backing {
            ScratchBytesBacking::U8(v) => give_u8(v),
            ScratchBytesBacking::F128(v) => give_f128(v),
        }
    }
}

impl Deref for ScratchBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.backing {
            ScratchBytesBacking::U8(v) => v,
            ScratchBytesBacking::F128(v) => {
                // SAFETY: F128 consists of two u64s, has no padding, and every
                // bit pattern is valid. The slice covers the initialized
                // object representation without changing ownership/layout.
                unsafe {
                    core::slice::from_raw_parts(
                        v.as_ptr().cast::<u8>(),
                        v.len() * core::mem::size_of::<F128>(),
                    )
                }
            }
        }
    }
}

impl DerefMut for ScratchBytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.backing {
            ScratchBytesBacking::U8(v) => v,
            ScratchBytesBacking::F128(v) => {
                // SAFETY: same representation argument as `Deref`; the
                // exclusive borrow of `self` makes the byte view exclusive.
                unsafe {
                    core::slice::from_raw_parts_mut(
                        v.as_mut_ptr().cast::<u8>(),
                        v.len() * core::mem::size_of::<F128>(),
                    )
                }
            }
        }
    }
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

    #[test]
    fn donated_f128_bytes_recycle_with_original_layout() {
        clear();
        let storage = vec![F128::ZERO; 64];
        let ptr = storage.as_ptr();
        let mut bytes = ScratchBytes::from_initialized_f128(storage);
        assert_eq!(bytes.len(), 64 * core::mem::size_of::<F128>());
        bytes.fill(0xa5);
        bytes.recycle();

        let recycled = take_f128(64);
        assert_eq!(recycled.as_ptr(), ptr);
        give_f128(recycled);
        clear();
    }
}

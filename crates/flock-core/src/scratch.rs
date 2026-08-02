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
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

static POOL: Mutex<Vec<Vec<F128>>> = Mutex::new(Vec::new());

/// One process-lifetime allocation whose address is consumed by a retained
/// external no-copy view. Unlike [`POOL`], this slot is deliberately immune
/// to smallest-first eviction and [`clear`]: the owner of that view calls
/// [`unpin_f128_allocation`] before the allocation may rejoin the ordinary
/// pool. While `buffer` is `None`, the exact `Vec` is checked out by prover
/// code, so ownership still exists outside this slot.
struct PinnedF128 {
    addr: usize,
    len: usize,
    buffer: Option<Vec<F128>>,
}

static PINNED_F128: Mutex<Option<PinnedF128>> = Mutex::new(None);
// Fast rejection keeps ordinary scratch traffic off the pin mutex.
static PINNED_F128_ADDR: AtomicUsize = AtomicUsize::new(0);
static PINNED_F128_LEN: AtomicUsize = AtomicUsize::new(0);

/// Second, independent pinned slot with the same semantics, registered once
/// by the recursive-Merkle GPU offload for the L1 matrix class (its no-copy
/// Metal view must keep naming one process-lifetime address so the timed
/// prove never re-pays wrap creation or page wiring). Deliberately a
/// duplicate rather than a generalization: the promoted z-pin lifecycle
/// above stays byte-for-byte untouched.
static PINNED2_F128: Mutex<Option<PinnedF128>> = Mutex::new(None);
static PINNED2_F128_ADDR: AtomicUsize = AtomicUsize::new(0);
static PINNED2_F128_LEN: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// Owner-EXCLUSIVE pins.
//
// [`PINNED_F128`] above keys only on `len`, so `take_f128(n)` hands the
// registered allocation to WHOEVER asks for that length next. That is fine for
// its owner (a single process-lifetime consumer that re-requests the same
// length every prove) but it is NOT enough for a no-copy Metal view whose
// address must be stable across proves when OTHER consumers request the same
// size class: at the ranked m = 32 the witness `a`, the witness `b`, the
// `Round1AbInner` transform and several PCS transients are all exactly
// 2^25 F128 = 512 MiB, so a same-size-class consumer poaches the allocation
// between proves and the owner comes back to a DIFFERENT address. Rebuilding a
// 512 MiB `newBufferWithBytesNoCopy` wrap (and re-wiring its pages) inside a
// timed prove costs far more than any offload can win — that exact bug is what
// a sibling GPU arm was rejected at −16.95% for.
//
// An exclusive slot is keyed by `(owner, len)` and is invisible to ordinary
// `take_f128`, so nothing but the declared owner can ever draw from it.
/// Declared owners of an exclusive scratch allocation.
///
/// Kept as a closed enum (rather than a string/type-id registry) so the set of
/// allocations withheld from the general pool is auditable at a glance: every
/// exclusive slot permanently removes one buffer from smallest-fit reuse.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExclusiveOwner {
    /// Packed `a = A · z`, read by the GPU AB-precompute prefix.
    AbPrecomputeA,
    /// Packed `b = B · z`, read by the GPU AB-precompute prefix.
    AbPrecomputeB,
    /// The `Round1AbInner` transform, written by the GPU AB-precompute prefix.
    AbPrecomputeOut,
}

impl ExclusiveOwner {
    const COUNT: usize = 3;
    #[inline]
    const fn slot(self) -> usize {
        match self {
            Self::AbPrecomputeA => 0,
            Self::AbPrecomputeB => 1,
            Self::AbPrecomputeOut => 2,
        }
    }
}

#[allow(clippy::declare_interior_mutable_const)]
const EXCLUSIVE_INIT: Mutex<Option<PinnedF128>> = Mutex::new(None);
static EXCLUSIVE_F128: [Mutex<Option<PinnedF128>>; ExclusiveOwner::COUNT] =
    [EXCLUSIVE_INIT; ExclusiveOwner::COUNT];
#[allow(clippy::declare_interior_mutable_const)]
const EXCLUSIVE_ADDR_INIT: AtomicUsize = AtomicUsize::new(0);
/// Fast rejection for [`give_f128`]: ordinary scratch traffic must not take
/// three extra mutexes on every return.
static EXCLUSIVE_F128_ADDR: [AtomicUsize; ExclusiveOwner::COUNT] =
    [EXCLUSIVE_ADDR_INIT; ExclusiveOwner::COUNT];

/// Take a length-`n` `F128` vector that belongs to `owner` alone.
///
/// The first call claims an ordinary pooled allocation and registers it; every
/// later call with the same `(owner, n)` returns that same allocation, at the
/// same address, for the life of the process. Ordinary [`take_f128`] can never
/// see it. Same UNINITIALIZED write-before-read contract as [`take_f128`].
pub fn take_f128_exclusive(owner: ExclusiveOwner, n: usize) -> Vec<F128> {
    let registered_same_len = {
        let mut slot = EXCLUSIVE_F128[owner.slot()].lock().unwrap();
        match slot.as_mut() {
            Some(pinned) if pinned.len == n => {
                if let Some(mut v) = pinned.buffer.take() {
                    debug_assert_eq!(v.as_ptr() as usize, pinned.addr);
                    debug_assert!(v.capacity() >= n);
                    v.clear();
                    // SAFETY: the registered allocation has capacity >= n and
                    // F128 is Copy; the caller upholds take_f128's
                    // write-before-read contract.
                    unsafe { v.set_len(n) };
                    return v;
                }
                // Registered, but currently checked out.
                true
            }
            _ => false,
        }
    };
    let v = take_f128(n);
    if registered_same_len {
        // A SECOND live request for the same owner and length. The ranked
        // prover does exactly this: the post-join exact-contention tuner
        // replays the AB precompute while the join's own `Round1AbInner` is
        // still alive. Hand this one an ordinary buffer and leave the
        // registration where it is — re-pointing the slot at the replay's
        // allocation would move the owner's address between proves, which is
        // the precise failure this mechanism exists to prevent.
        return v;
    }
    // First claim for this owner (or a new length class): register it. This is
    // what makes the address stable from here on.
    let addr = v.as_ptr() as usize;
    let mut slot = EXCLUSIVE_F128[owner.slot()].lock().unwrap();
    *slot = Some(PinnedF128 {
        addr,
        len: n,
        buffer: None,
    });
    EXCLUSIVE_F128_ADDR[owner.slot()].store(addr, Ordering::Release);
    v
}

/// Whether `addr` is the allocation currently registered to `owner`.
///
/// A caller holding a buffer for which this is FALSE is a second, concurrent
/// user (see [`take_f128_exclusive`]) that was handed an ordinary pooled
/// allocation. Anything keyed to the owner's stable address — a no-copy Metal
/// view above all — must sit that call out rather than treat the different
/// address as churn.
pub fn is_exclusive_allocation(owner: ExclusiveOwner, addr: usize) -> bool {
    addr != 0 && EXCLUSIVE_F128_ADDR[owner.slot()].load(Ordering::Acquire) == addr
}

/// Park `v` in its owner's exclusive slot when it is the registered
/// allocation. Returns the buffer untouched when it is not.
fn give_f128_exclusive(v: Vec<F128>) -> Option<Vec<F128>> {
    let addr = v.as_ptr() as usize;
    for (i, key) in EXCLUSIVE_F128_ADDR.iter().enumerate() {
        if key.load(Ordering::Acquire) != addr {
            continue;
        }
        let mut slot = EXCLUSIVE_F128[i].lock().unwrap();
        if let Some(pinned) = slot.as_mut()
            && pinned.addr == addr
            && v.capacity() >= pinned.len
            && pinned.buffer.is_none()
        {
            pinned.buffer = Some(v);
            return None;
        }
        return Some(v);
    }
    Some(v)
}

/// Drop every exclusive registration (tests only — a live Metal no-copy view
/// over one of these allocations would be left naming freed memory).
#[cfg(test)]
pub(crate) fn clear_exclusive_for_tests() {
    for (i, key) in EXCLUSIVE_F128_ADDR.iter().enumerate() {
        let mut slot = EXCLUSIVE_F128[i].lock().unwrap();
        key.store(0, Ordering::Release);
        *slot = None;
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
    if let Some(v) = try_take_pinned_f128(n) {
        return v;
    }
    if let Some(v) = try_take_pinned2_f128(n) {
        return v;
    }
    if let Some(v) = try_take_f128(n) {
        return v;
    }
    crate::alloc_uninit_vec(n)
}

/// [`take_f128`] variant that never returns a pinned allocation. The pinned
/// slots carry process-lifetime external no-copy Metal views; a caller that
/// intends to create its OWN no-copy view over the returned buffer (the
/// zerocheck tail products arms) must not receive a range that is already
/// wrapped — overlapping `newBufferWithBytesNoCopy` views are not legal, and
/// on a worker where the pin is parked at take time the ordinary
/// pinned-first preference hands out exactly that collision.
pub(crate) fn take_f128_unpinned(n: usize) -> Vec<F128> {
    if let Some(v) = try_take_f128(n) {
        // The evictable pool never holds a pinned allocation (give_f128
        // parks those in their dedicated slots), so this cannot alias.
        return v;
    }
    crate::alloc_uninit_vec(n)
}

/// Whether `[addr, addr + len_bytes)` overlaps either pinned registration's
/// byte range. Belt-and-braces for wrap sites: even a buffer obtained
/// through an ordinary take must refuse a second no-copy view if it aliases
/// a pinned (already-wrapped) allocation.
pub(crate) fn f128_range_overlaps_pin(addr: usize, len_bytes: usize) -> bool {
    let end = addr.saturating_add(len_bytes);
    for (a, l) in [
        (
            PINNED_F128_ADDR.load(Ordering::Acquire),
            PINNED_F128_LEN.load(Ordering::Acquire),
        ),
        (
            PINNED2_F128_ADDR.load(Ordering::Acquire),
            PINNED2_F128_LEN.load(Ordering::Acquire),
        ),
    ] {
        if a == 0 || l == 0 {
            continue;
        }
        let pin_end = a + l * core::mem::size_of::<F128>();
        if addr < pin_end && a < end {
            return true;
        }
    }
    false
}

fn try_take_pinned_f128(n: usize) -> Option<Vec<F128>> {
    if PINNED_F128_LEN.load(Ordering::Acquire) != n {
        return None;
    }
    let mut slot = PINNED_F128.lock().unwrap();
    let pinned = slot.as_mut()?;
    if pinned.len != n {
        return None;
    }
    let mut v = pinned.buffer.take()?;
    debug_assert_eq!(v.as_ptr() as usize, pinned.addr);
    debug_assert!(v.capacity() >= n);
    v.clear();
    // SAFETY: the registered allocation has capacity >= n and F128 is Copy;
    // callers retain the ordinary write-before-read contract of take_f128.
    unsafe { v.set_len(n) };
    Some(v)
}

fn try_take_pinned2_f128(n: usize) -> Option<Vec<F128>> {
    if PINNED2_F128_LEN.load(Ordering::Acquire) != n {
        return None;
    }
    let mut slot = PINNED2_F128.lock().unwrap();
    let pinned = slot.as_mut()?;
    if pinned.len != n {
        return None;
    }
    let mut v = pinned.buffer.take()?;
    debug_assert_eq!(v.as_ptr() as usize, pinned.addr);
    debug_assert!(v.capacity() >= n);
    v.clear();
    // SAFETY: the registered allocation has capacity >= n and F128 is Copy;
    // callers retain the ordinary write-before-read contract of take_f128.
    unsafe { v.set_len(n) };
    Some(v)
}

/// [`pin_f128_allocation`] for the second slot. Registered once per process
/// (the recursive-Merkle offload's init); there is no unpin — the Metal view
/// lives for the process.
pub(crate) fn pin2_f128_allocation(buffer: &[F128]) -> bool {
    if buffer.is_empty() {
        return false;
    }
    let addr = buffer.as_ptr() as usize;
    let len = buffer.len();
    let mut slot = PINNED2_F128.lock().unwrap();
    match slot.as_ref() {
        Some(pinned) => pinned.addr == addr && pinned.len == len,
        None => {
            *slot = Some(PinnedF128 {
                addr,
                len,
                buffer: None,
            });
            PINNED2_F128_ADDR.store(addr, Ordering::Release);
            PINNED2_F128_LEN.store(len, Ordering::Release);
            true
        }
    }
}

/// Quarantine the allocation behind `buffer` once it next returns through
/// [`give_f128`]. Until then the caller still owns the exact `Vec`; after it
/// returns, [`take_f128`] preferentially hands that same allocation back for
/// an equal-length request.
///
/// Only one external no-copy view is supported. Re-registering the same
/// allocation is idempotent; a different live registration fails closed.
pub(crate) fn pin_f128_allocation(buffer: &[F128]) -> bool {
    if buffer.is_empty() {
        return false;
    }
    let addr = buffer.as_ptr() as usize;
    let len = buffer.len();
    let mut slot = PINNED_F128.lock().unwrap();
    match slot.as_ref() {
        Some(pinned) => pinned.addr == addr && pinned.len == len,
        None => {
            *slot = Some(PinnedF128 {
                addr,
                len,
                buffer: None,
            });
            PINNED_F128_ADDR.store(addr, Ordering::Release);
            PINNED_F128_LEN.store(len, Ordering::Release);
            true
        }
    }
}

/// Release a registration after its external no-copy view has been released.
/// If the allocation is parked, it rejoins the ordinary evictable pool; if it
/// is checked out, its eventual [`give_f128`] follows the ordinary path.
pub(crate) fn unpin_f128_allocation(addr: usize, len: usize) -> bool {
    let parked = {
        let mut slot = PINNED_F128.lock().unwrap();
        let matches = slot
            .as_ref()
            .is_some_and(|pinned| pinned.addr == addr && pinned.len == len);
        if !matches {
            return false;
        }
        // Clear the fast-path keys while holding the slot lock. A racing
        // take/give rechecks the guarded record before acting.
        PINNED_F128_LEN.store(0, Ordering::Release);
        PINNED_F128_ADDR.store(0, Ordering::Release);
        slot.take().and_then(|pinned| pinned.buffer)
    };
    if let Some(v) = parked {
        give_f128(v);
    }
    true
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
    let Some(v) = give_f128_exclusive(v) else {
        return;
    };
    let addr = v.as_ptr() as usize;
    if PINNED_F128_ADDR.load(Ordering::Acquire) == addr {
        let mut slot = PINNED_F128.lock().unwrap();
        if let Some(pinned) = slot.as_mut()
            && pinned.addr == addr
            && v.capacity() >= pinned.len
            && pinned.buffer.is_none()
        {
            pinned.buffer = Some(v);
            return;
        }
    }
    if PINNED2_F128_ADDR.load(Ordering::Acquire) == addr {
        let mut slot = PINNED2_F128.lock().unwrap();
        if let Some(pinned) = slot.as_mut()
            && pinned.addr == addr
            && v.capacity() >= pinned.len
            && pinned.buffer.is_none()
        {
            pinned.buffer = Some(v);
            return;
        }
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

/// Release every ordinary pooled buffer back to the OS.
///
/// A buffer registered through [`pin_f128_allocation`] is intentionally not
/// released here: its external no-copy view still names that allocation.
/// [`unpin_f128_allocation`] releases that lifetime coupling first and then
/// returns any parked buffer to the ordinary pool.
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

    // Every test below mutates the same process-global pools. Serialize them
    // so pointer-identity assertions do not race another test's `clear`.
    static SCRATCH_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn take_reuses_given_buffer() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
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
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
        clear();
        for _ in 0..(MAX_POOLED + 4) {
            give_f128(take_f128(16));
        }
        assert!(POOL.lock().unwrap().len() <= MAX_POOLED);
        clear();
    }

    #[test]
    fn donated_f128_bytes_recycle_with_original_layout() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
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

    #[test]
    fn pinned_f128_is_preferred_and_survives_clear_until_unpinned() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
        const N: usize = 257;
        clear();
        let pinned = take_f128(N);
        let ptr = pinned.as_ptr();
        assert!(pin_f128_allocation(&pinned));

        // A same-size competitor returns first, but the registered allocation
        // is quarantined separately when it later comes back.
        let competitor = take_f128(N);
        assert_ne!(competitor.as_ptr(), ptr);
        give_f128(competitor);
        give_f128(pinned);
        clear();

        let reused = take_f128(N);
        assert_eq!(reused.as_ptr(), ptr);
        give_f128(reused);
        assert!(unpin_f128_allocation(ptr as usize, N));
        clear();
    }

    /// The exact poaching shape that cost a sibling GPU arm −16.95%: two
    /// consumers of the SAME size class, one of them holding a no-copy Metal
    /// view. The plain `len`-keyed pin cannot express this (it would hand the
    /// registered allocation to whichever consumer asked next); the exclusive
    /// slot must return the owner's own address every time.
    #[test]
    fn exclusive_owner_keeps_its_address_against_a_same_size_class_consumer() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
        const N: usize = 269;
        clear();
        clear_exclusive_for_tests();

        let owned = take_f128_exclusive(ExclusiveOwner::AbPrecomputeOut, N);
        let owned_ptr = owned.as_ptr();
        give_f128(owned);

        // A same-size-class consumer runs between "proves" and returns its
        // buffer to the ordinary pool.
        let poacher = take_f128(N);
        assert_ne!(
            poacher.as_ptr(),
            owned_ptr,
            "ordinary take_f128 must not see the exclusive slot"
        );
        give_f128(poacher);

        for prove in 0..4 {
            let again = take_f128_exclusive(ExclusiveOwner::AbPrecomputeOut, N);
            assert_eq!(
                again.as_ptr(),
                owned_ptr,
                "exclusive owner lost its allocation on prove {prove}"
            );
            give_f128(again);
            let competitor = take_f128(N);
            assert_ne!(competitor.as_ptr(), owned_ptr);
            give_f128(competitor);
        }

        clear_exclusive_for_tests();
        clear();
    }

    /// A second LIVE request from the same owner must not move the
    /// registration. The ranked prover does exactly this: the post-join
    /// exact-contention tuner replays the AB precompute while the join's own
    /// `Round1AbInner` is still alive, so two of them exist at once. If the
    /// replay's allocation captured the slot, the owner's address would
    /// change between proves and the no-copy wrap would churn — silently
    /// disabling the arm it exists to serve.
    #[test]
    fn a_second_live_take_does_not_move_the_registration() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
        const N: usize = 277;
        clear();
        clear_exclusive_for_tests();

        let first = take_f128_exclusive(ExclusiveOwner::AbPrecomputeOut, N);
        let owned_ptr = first.as_ptr();
        // The replay, while `first` is still alive.
        let replay = take_f128_exclusive(ExclusiveOwner::AbPrecomputeOut, N);
        assert_ne!(replay.as_ptr(), owned_ptr);
        give_f128(replay); // replay finishes first, as in the prover
        give_f128(first);

        let next_prove = take_f128_exclusive(ExclusiveOwner::AbPrecomputeOut, N);
        assert_eq!(
            next_prove.as_ptr(),
            owned_ptr,
            "the replay captured the exclusive slot; the wrap would churn"
        );
        give_f128(next_prove);

        clear_exclusive_for_tests();
        clear();
    }

    /// Distinct owners of the same size class never collide, and `clear` (the
    /// end-of-batch release) does not take an exclusive allocation away — its
    /// Metal view names that address for the life of the process.
    #[test]
    fn exclusive_owners_are_independent_and_survive_clear() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
        const N: usize = 271;
        clear();
        clear_exclusive_for_tests();

        let a = take_f128_exclusive(ExclusiveOwner::AbPrecomputeA, N);
        let b = take_f128_exclusive(ExclusiveOwner::AbPrecomputeB, N);
        let (a_ptr, b_ptr) = (a.as_ptr(), b.as_ptr());
        assert_ne!(a_ptr, b_ptr);
        give_f128(a);
        give_f128(b);
        clear();

        let a2 = take_f128_exclusive(ExclusiveOwner::AbPrecomputeA, N);
        let b2 = take_f128_exclusive(ExclusiveOwner::AbPrecomputeB, N);
        assert_eq!(a2.as_ptr(), a_ptr, "clear() dropped an exclusive allocation");
        assert_eq!(b2.as_ptr(), b_ptr, "clear() dropped an exclusive allocation");
        give_f128(a2);
        give_f128(b2);

        clear_exclusive_for_tests();
        clear();
    }

    /// The whole AB-precompute GPU arm depends on `newBufferWithBytesNoCopy`
    /// over the ranked 512 MiB F128 classes, which Metal only accepts on
    /// page-aligned memory of a page-multiple length (the byte-pool twin of
    /// this is `gpu_commit::ranked_lincheck_stripe_is_wrappable_without_a_copy`).
    #[test]
    fn ranked_ab_f128_class_is_wrappable_without_a_copy() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
        const PAGE: usize = 16384;
        // 2^25 F128 = 512 MiB: the ranked `a`, `b` and `Round1AbInner` class.
        let v = take_f128(1usize << 25);
        let (ptr, bytes) = (v.as_ptr() as usize, std::mem::size_of_val(&v[..]));
        give_f128(v);
        clear();
        assert_eq!(ptr % PAGE, 0, "ranked AB buffer base must be page-aligned");
        assert_eq!(bytes % PAGE, 0, "ranked AB length must be a page multiple");
    }

    #[test]
    fn unpin_while_checked_out_makes_the_later_return_ordinary() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
        const N: usize = 263;
        clear();
        let pinned = take_f128(N);
        let ptr = pinned.as_ptr();
        assert!(pin_f128_allocation(&pinned));
        give_f128(pinned);

        let checked_out = take_f128(N);
        assert_eq!(checked_out.as_ptr(), ptr);
        assert!(unpin_f128_allocation(ptr as usize, N));
        give_f128(checked_out);

        // The buffer is now ordinary scratch, so clear is allowed to drop it.
        assert_eq!(PINNED_F128_ADDR.load(Ordering::Acquire), 0);
        assert_eq!(PINNED_F128_LEN.load(Ordering::Acquire), 0);
        clear();
    }
}

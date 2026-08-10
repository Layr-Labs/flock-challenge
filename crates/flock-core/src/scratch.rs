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

/// An evictable pooled buffer plus its optional provenance token.
///
/// A token asserts "when this exact allocation entered pool custody, the
/// byte regions its producer declared constant held those constants, and
/// nothing has touched the buffer since" — every take path drops the token
/// with the entry, so a token can only survive parked, untouched custody.
/// A tokened entry is additionally RESERVED for its tagger: non-matching
/// takes skip it (see [`try_take_f128_inner`]), so parked custody cannot be
/// broken by an unrelated consumer that merely fits.
struct PoolEntry {
    buf: Vec<F128>,
    token: Option<u64>,
}

static POOL: Mutex<Vec<PoolEntry>> = Mutex::new(Vec::new());

/// One sampled constant-byte range a staged release token must satisfy at
/// give time (belt-and-braces against a stale stage matching a recycled
/// address): `len` bytes at `byte_off` must all equal `value`.
pub struct ReleaseProbe {
    pub byte_off: usize,
    pub len: usize,
    pub value: u8,
}

/// A release token staged by the buffer's owner while the buffer is still
/// alive (pointer identity guaranteed), consumed by the `give_f128` of that
/// exact allocation. `addr`+`len` must match the given vector exactly and
/// every probe must verify, or the give attaches no token (fail-safe).
struct StagedRelease {
    addr: usize,
    len: usize,
    tag: u64,
    probes: Vec<ReleaseProbe>,
}

static STAGED: Mutex<Vec<StagedRelease>> = Mutex::new(Vec::new());

/// Stage a provenance token for `buf`'s eventual return through
/// [`give_f128`]. Call while owning `buf` (so the address cannot be
/// recycled between stage and give of the same allocation). Replaces any
/// previously staged entry with the same `tag` — a stale stage from a
/// prove that never released (abnormal termination) is thereby retired at
/// the next prove's stage; [`take_f128_with_token`] additionally purges
/// same-tag stages before taking. `probes` are verified against the given
/// buffer's bytes at give time; any mismatch drops the token silently.
pub fn stage_f128_release_token(buf: &[F128], tag: u64, probes: Vec<ReleaseProbe>) {
    if buf.is_empty() {
        return;
    }
    let mut staged = STAGED.lock().unwrap();
    staged.retain(|s| s.tag != tag);
    staged.push(StagedRelease {
        addr: buf.as_ptr() as usize,
        len: buf.len(),
        tag,
        probes,
    });
}

/// Retire any staged token naming `addr` — called whenever a take hands out
/// (or freshly allocates) a buffer at that address, so a stale stage can
/// never be matched by a later give of a *different* allocation that
/// happens to reuse the address.
fn purge_staged_addr(addr: usize) {
    let mut staged = STAGED.lock().unwrap();
    staged.retain(|s| s.addr != addr);
}

/// Consume the staged token matching this exact give (`addr` + `len`), if
/// any, verifying its probes against the buffer contents. `None` (no stage,
/// or any probe mismatch) means the buffer parks untokened.
fn consume_staged_token(v: &[F128]) -> Option<u64> {
    let addr = v.as_ptr() as usize;
    let entry = {
        let mut staged = STAGED.lock().unwrap();
        let i = staged
            .iter()
            .position(|s| s.addr == addr && s.len == v.len())?;
        staged.swap_remove(i)
    };
    // SAFETY: F128 is plain bytes; the view covers the initialized slice.
    let bytes =
        unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), core::mem::size_of_val(v)) };
    for p in &entry.probes {
        let Some(range) = bytes.get(p.byte_off..p.byte_off + p.len) else {
            return None;
        };
        if !range.iter().all(|&b| b == p.value) {
            return None;
        }
    }
    Some(entry.tag)
}

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
    /// Provenance token attached when the allocation was last parked (same
    /// semantics as [`PoolEntry::token`]); cleared by every take.
    token: Option<u64>,
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
    take_f128_with_token_inner(n, None).0
}

/// [`take_f128`] variant carrying a provenance-token request: returns the
/// buffer plus whether the exact allocation handed out was parked through a
/// tagged release ([`stage_f128_release_token`] + `give_f128` of that same
/// live allocation) with this `tag` and length `n`, and untouched since.
/// On `true` the buffer's producer-declared constant regions are still the
/// bytes that producer wrote. On `false` the ordinary uninitialized
/// write-before-read contract applies to every slot.
///
/// Preference order matches [`take_f128`] (pinned slots first); within the
/// evictable pool an exact token match is preferred over smallest-fit.
pub fn take_f128_with_token(n: usize, tag: u64) -> (Vec<F128>, bool) {
    take_f128_with_token_inner(n, Some(tag))
}

fn take_f128_with_token_inner(n: usize, tag: Option<u64>) -> (Vec<F128>, bool) {
    if let Some(want) = tag {
        // Retire any stale stage with this tag (a prove that never
        // released) before it could match a future give at a recycled
        // address.
        STAGED.lock().unwrap().retain(|s| s.tag != want);
    }
    if let Some((v, token)) = try_take_pinned_f128(n) {
        purge_staged_addr(v.as_ptr() as usize);
        let hit = tag.is_some() && token == tag;
        return (v, hit);
    }
    if let Some((v, token)) = try_take_pinned2_f128(n) {
        purge_staged_addr(v.as_ptr() as usize);
        let hit = tag.is_some() && token == tag;
        return (v, hit);
    }
    if let Some((v, token)) = try_take_f128_inner(n, tag) {
        purge_staged_addr(v.as_ptr() as usize);
        let hit = tag.is_some() && token == tag;
        return (v, hit);
    }
    let v = crate::alloc_uninit_vec(n);
    purge_staged_addr(v.as_ptr() as usize);
    (v, false)
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
    let v = crate::alloc_uninit_vec(n);
    purge_staged_addr(v.as_ptr() as usize);
    v
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

fn try_take_pinned_f128(n: usize) -> Option<(Vec<F128>, Option<u64>)> {
    if PINNED_F128_LEN.load(Ordering::Acquire) != n {
        return None;
    }
    let mut slot = PINNED_F128.lock().unwrap();
    let pinned = slot.as_mut()?;
    if pinned.len != n {
        return None;
    }
    let mut v = pinned.buffer.take()?;
    let token = pinned.token.take();
    debug_assert_eq!(v.as_ptr() as usize, pinned.addr);
    debug_assert!(v.capacity() >= n);
    v.clear();
    // SAFETY: the registered allocation has capacity >= n and F128 is Copy;
    // callers retain the ordinary write-before-read contract of take_f128.
    unsafe { v.set_len(n) };
    Some((v, token))
}

fn try_take_pinned2_f128(n: usize) -> Option<(Vec<F128>, Option<u64>)> {
    if PINNED2_F128_LEN.load(Ordering::Acquire) != n {
        return None;
    }
    let mut slot = PINNED2_F128.lock().unwrap();
    let pinned = slot.as_mut()?;
    if pinned.len != n {
        return None;
    }
    let mut v = pinned.buffer.take()?;
    let token = pinned.token.take();
    debug_assert_eq!(v.as_ptr() as usize, pinned.addr);
    debug_assert!(v.capacity() >= n);
    v.clear();
    // SAFETY: the registered allocation has capacity >= n and F128 is Copy;
    // callers retain the ordinary write-before-read contract of take_f128.
    unsafe { v.set_len(n) };
    Some((v, token))
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
                token: None,
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
                token: None,
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
    let v = try_take_f128_inner(n, None)?.0;
    purge_staged_addr(v.as_ptr() as usize);
    Some(v)
}

/// Pool scan shared by the plain and tokened takes. When `want` is set, an
/// exact token match (`token == want` AND parked length == `n`) is
/// preferred over smallest-fit.
///
/// CUSTODY RESERVATION (constant-region elision, item B): an entry parked
/// through a tagged release belongs to its tagger until that consumer takes
/// it back — every non-matching take treats it as invisible and falls back
/// to a fresh allocation when nothing else fits. Without this, smallest-fit
/// deterministically retired the witgen a-buffer's token every prove: the
/// PCS open's small recursive-commit matrices (2^17–2^19 F128) land at a
/// moment when no untokened entry fits and grabbed the 512 MiB a-entry —
/// the first equal-capacity tokened fit in scan order — killing a's elision
/// on every subsequent prove (measured: elide-canary 0b101, the a-bit
/// permanently 0). The reserved classes are exactly the per-prove witness
/// buffers, so reservation adds no steady-state footprint: their tagger
/// re-takes them every prove, and they stay evictable (give-side
/// smallest-first) and released by [`clear`]. With elision killed
/// (`FLOCK_NO_SCRATCH_CONST_ELIDE=1`) no tokens are ever staged and this
/// policy is byte-for-byte the incumbent smallest-fit.
fn try_take_f128_inner(n: usize, want: Option<u64>) -> Option<(Vec<F128>, Option<u64>)> {
    let mut pool = POOL.lock().unwrap();
    let mut best: Option<usize> = None;
    if let Some(want) = want {
        for (i, e) in pool.iter().enumerate() {
            if e.token == Some(want) && e.buf.len() == n {
                best = Some(i);
                break;
            }
        }
    }
    if best.is_none() {
        for (i, e) in pool.iter().enumerate() {
            let cap = e.buf.capacity();
            if cap < n {
                continue;
            }
            // Reserved for another consumer (see above). A same-tag entry
            // whose parked length missed the exact-match loop is still fair
            // game for its own tagger (the token is dropped at take).
            if e.token.is_some_and(|t| Some(t) != want) {
                continue;
            }
            let better = match best {
                None => true,
                Some(b) => {
                    cap < pool[b].buf.capacity()
                        || (cap == pool[b].buf.capacity()
                            && e.token.is_none()
                            && pool[b].token.is_some())
                }
            };
            if better {
                best = Some(i);
            }
        }
    }
    if let Some(i) = best {
        let entry = pool.swap_remove(i);
        drop(pool);
        let mut v = entry.buf;
        // A token is only valid for the exact parked length (a resize
        // through set_len below would expose slots the tagged release
        // never vouched for).
        let token = entry.token.filter(|_| v.len() == n);
        v.clear();
        // SAFETY: capacity ≥ n was checked above; F128: Copy (no Drop), so
        // exposing uninit/stale elements is sound to *hold* — the caller
        // upholds write-before-read per this function's contract.
        unsafe { v.set_len(n) };
        return Some((v, token));
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
    let addr = v.as_ptr() as usize;
    // Attach the staged provenance token (if the owner staged one for this
    // exact allocation+length and its constant-region probes verify). The
    // owner held the Vec from stage to this give, so pointer identity here
    // is the same allocation, not a recycled address.
    let token = consume_staged_token(&v);
    if PINNED_F128_ADDR.load(Ordering::Acquire) == addr {
        let mut slot = PINNED_F128.lock().unwrap();
        if let Some(pinned) = slot.as_mut()
            && pinned.addr == addr
            && v.capacity() >= pinned.len
            && pinned.buffer.is_none()
        {
            pinned.buffer = Some(v);
            pinned.token = token;
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
            pinned.token = token;
            return;
        }
    }
    let mut pool = POOL.lock().unwrap();
    pool.push(PoolEntry { buf: v, token });
    if pool.len() > MAX_POOLED {
        let smallest = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.buf.capacity())
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
    STAGED.lock().unwrap().clear();
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

// ---------------------------------------------------------------------------
// Recursive-Merkle flat-tree pool.
//
// The Ligerito open builds two flat `Vec<Hash>` trees per prove (16 MiB at
// L1, 4 MiB at L2) on the FS-serial opening spine. A fresh
// `alloc_uninit_vec` pays its ~4k first-touch page faults inside the timed
// window and a single-threaded munmap on the following drop; recycling the
// two buffers across proves keeps their pages resident. Contents are NOT
// cleared; callers must write every node before reading (both builders do:
// the GPU copy-out writes the whole tree, the CPU builder writes leaves
// then every parent level).
// ---------------------------------------------------------------------------

static POOL_HASH_TREE: Mutex<Vec<Vec<crate::merkle::Hash>>> = Mutex::new(Vec::new());

/// One prove hands back all five of its level trees (L0, the recursive
/// levels, and the final level), so a four-slot cap evicted one of them —
/// smallest-first — on every prove and re-faulted it on the next. Six slots
/// hold the whole per-prove set with room for the transient test shapes.
const MAX_POOLED_HASH_TREES: usize = 6;

/// Strict kill switch for the intermediate recursive-level tree recycle in
/// the Ligerito opening spine (the one level tree that used to be dropped
/// instead of pooled). Only the exact value `1` restores the incumbent
/// drop. Pooling is allocation plumbing — it changes only which pages stay
/// resident, never proof bytes. With the recycle off, a prove returns just
/// the two trees it always returned, so the pool never reaches even the old
/// four-slot cap and the raised cap above is inert under the switch: this
/// is an exact same-binary rollback.
pub(crate) const ENV_NO_TREE_POOL_FULL: &str = "FLOCK_NO_TREE_POOL_FULL";

fn tree_pool_full_value_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("1"))
}

pub(crate) fn tree_pool_full_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        tree_pool_full_value_enabled(std::env::var_os(ENV_NO_TREE_POOL_FULL).as_deref())
    })
}

/// Take a length-`n` hash vector, preferring a pooled buffer (smallest
/// capacity >= `n`); falls back to a fresh uninitialized allocation.
/// Contents are UNINITIALIZED in both cases (write-before-read contract,
/// same as [`take_u8`]).
pub(crate) fn take_hash_tree(n: usize) -> Vec<crate::merkle::Hash> {
    {
        let mut pool = POOL_HASH_TREE.lock().unwrap();
        let mut best: Option<usize> = None;
        for (i, v) in pool.iter().enumerate() {
            if v.capacity() >= n && best.is_none_or(|b| v.capacity() < pool[b].capacity()) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            let mut v = pool.swap_remove(i);
            // SAFETY: capacity >= n was checked above; `Hash` is `[u8; 32]`
            // (Copy, no Drop), so exposing stale bytes is sound to *hold* —
            // the caller upholds write-before-read per this function's
            // contract.
            unsafe { v.set_len(n) };
            return v;
        }
    }
    crate::alloc_uninit_vec(n)
}

/// Return a flat tree to the pool for reuse (smallest-first eviction when
/// full, same policy as the byte pool).
pub(crate) fn give_hash_tree(v: Vec<crate::merkle::Hash>) {
    if v.capacity() == 0 {
        return;
    }
    let mut pool = POOL_HASH_TREE.lock().unwrap();
    pool.push(v);
    if pool.len() > MAX_POOLED_HASH_TREES {
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
    fn hash_tree_pool_is_bounded() {
        // Unconditional invariant: every `give_hash_tree` evicts back to the
        // cap, so the assertion holds whatever else the process pooled.
        for _ in 0..(MAX_POOLED_HASH_TREES + 4) {
            give_hash_tree(take_hash_tree(16));
        }
        assert!(POOL_HASH_TREE.lock().unwrap().len() <= MAX_POOLED_HASH_TREES);
    }

    #[test]
    fn exact_one_is_the_only_tree_pool_full_kill_value() {
        use std::ffi::OsStr;
        assert_eq!(ENV_NO_TREE_POOL_FULL, "FLOCK_NO_TREE_POOL_FULL");
        assert!(!tree_pool_full_value_enabled(Some(OsStr::new("1"))));
        for value in [None, Some(""), Some("0"), Some("01"), Some("true")] {
            assert!(tree_pool_full_value_enabled(value.map(OsStr::new)));
        }
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

    #[test]
    fn release_token_round_trip_then_consumed() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
        const N: usize = 1024;
        const TAG: u64 = 0xF10C_C0DE;
        clear();
        let mut v = take_f128(N);
        v.fill(F128::ZERO);
        stage_f128_release_token(
            &v,
            TAG,
            vec![ReleaseProbe {
                byte_off: 0,
                len: 64,
                value: 0,
            }],
        );
        let ptr = v.as_ptr();
        give_f128(v);
        // The tagged take returns the exact allocation with a hit...
        let (v2, hit) = take_f128_with_token(N, TAG);
        assert!(hit);
        assert_eq!(v2.as_ptr(), ptr);
        // ...and consumes the token: an un-staged re-release misses.
        give_f128(v2);
        let (v3, hit) = take_f128_with_token(N, TAG);
        assert!(!hit);
        give_f128(v3);
        clear();
    }

    #[test]
    fn release_token_dropped_by_foreign_custody_and_failed_probe() {
        let _serial = SCRATCH_TEST_LOCK.lock().unwrap();
        const N: usize = 1536;
        const TAG: u64 = 0xF10C_C0DF;
        const OTHER_TAG: u64 = 0xF10C_C0EE;
        clear();
        // Custody reservation: a plain take must NOT receive the tokened
        // entry (it falls back to a fresh allocation), so the token
        // survives for its tagger...
        let mut v = take_f128(N);
        v.fill(F128::ZERO);
        stage_f128_release_token(&v, TAG, Vec::new());
        let ptr = v.as_ptr();
        give_f128(v);
        let foreign = take_f128(N);
        assert_ne!(
            foreign.as_ptr(),
            ptr,
            "plain take must not break tagged custody"
        );
        // ...and a differently-tagged take must not receive it either.
        let (other, other_hit) = take_f128_with_token(N, OTHER_TAG);
        assert!(!other_hit);
        assert_ne!(
            other.as_ptr(),
            ptr,
            "foreign-tagged take must not break tagged custody"
        );
        give_f128(other);
        give_f128(foreign);
        let (v2, hit) = take_f128_with_token(N, TAG);
        assert!(hit, "reserved token must survive non-matching takes");
        assert_eq!(v2.as_ptr(), ptr);
        // If the tagger itself re-takes and returns without staging, the
        // next tagged take misses (token consumed exactly once).
        give_f128(v2);
        let (v2, hit) = take_f128_with_token(N, TAG);
        assert!(!hit, "token is consumed by its tagger's take");
        // Failing probe: the give must refuse to attach the token.
        let mut v2 = v2;
        v2.fill(F128::ZERO);
        stage_f128_release_token(
            &v2,
            TAG,
            vec![ReleaseProbe {
                byte_off: 0,
                len: 16,
                value: 0xFF, // buffer holds zeros — probe must fail
            }],
        );
        give_f128(v2);
        let (v3, hit) = take_f128_with_token(N, TAG);
        assert!(!hit, "failed probe must drop the token");
        // Wrong length: token is only valid for the exact staged length.
        let mut v3 = v3;
        v3.fill(F128::ZERO);
        stage_f128_release_token(&v3, TAG, Vec::new());
        give_f128(v3);
        let (v4, hit) = take_f128_with_token(N / 2, TAG);
        assert!(!hit, "length mismatch must miss");
        give_f128(v4);
        clear();
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

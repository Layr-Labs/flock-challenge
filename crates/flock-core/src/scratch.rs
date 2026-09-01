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
use std::sync::atomic::{AtomicUsize, Ordering};

/// Pool entries carry a provenance tag: `0` means "no provenance". A non-zero
/// tag asserts the buffer's contents are EXACTLY what its previous owner
/// released — attached only by [`give_f128_tagged`], dropped by every other
/// custody event ([`give_f128`] pushes tag 0; any take hands the buffer to a
/// caller whose writes void the old provenance, and re-tagging is the new
/// owner's responsibility).
static POOL: Mutex<Vec<(Vec<F128>, u64)>> = Mutex::new(Vec::new());

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
    let v = crate::alloc_uninit_vec::<F128>(n);
    void_pending_tag(v.as_ptr());
    v
}

/// Pool-only variant of [`take_f128`]: returns `None` instead of falling
/// back to a fresh allocation. Lets callers branch on warm-vs-cold (e.g.
/// the commit prefault skips its page-touch thread when the pool can
/// supply an already-resident buffer).
pub(crate) fn try_take_f128(n: usize) -> Option<Vec<F128>> {
    try_take_f128_tagged(n, 0).map(|(v, _)| v)
}

/// [`take_f128`] with provenance: prefers a pooled buffer whose tag equals
/// `tag` (smallest such capacity ≥ `n`), falling back to plain best-fit. The
/// returned flag is `true` only when the buffer's previous release attached
/// exactly this tag and no other custody event touched it since — i.e. its
/// contents are bit-identical to what the tagged releaser handed back. On a
/// `false` return the contents are stale/uninitialized as usual.
///
/// `tag` must be non-zero for the hit flag to be meaningful (tag 0 always
/// reports `false`).
pub fn take_f128_tagged(n: usize, tag: u64) -> (Vec<F128>, bool) {
    if let Some(r) = try_take_f128_tagged(n, tag) {
        return r;
    }
    let v = crate::alloc_uninit_vec::<F128>(n);
    void_pending_tag(v.as_ptr());
    (v, false)
}

fn try_take_f128_tagged(n: usize, tag: u64) -> Option<(Vec<F128>, bool)> {
    let mut pool = POOL.lock().unwrap();
    let mut best: Option<usize> = None;
    let mut best_pref = false;
    for (i, (v, t)) in pool.iter().enumerate() {
        if v.capacity() < n {
            continue;
        }
        // Tagged callers prefer their own tag; untagged callers prefer
        // untagged entries, so foreign provenance survives unrelated takes
        // whenever any other buffer fits (soft partition, never a refusal).
        let pref = if tag != 0 { *t == tag } else { *t == 0 };
        let better = match best {
            None => true,
            Some(b) => {
                (pref && !best_pref) || (pref == best_pref && v.capacity() < pool[b].0.capacity())
            }
        };
        if better {
            best = Some(i);
            best_pref = pref;
        }
    }
    let best_hit = best_pref && tag != 0;
    if let Some(i) = best {
        let (mut v, _) = pool.swap_remove(i);
        drop(pool);
        // A tag hit requires the previous release to have had the same
        // length: the tag encodes it (caller contract), and same allocation
        // + same length means `clear` + `set_len` below exposes exactly the
        // released bytes.
        v.clear();
        // SAFETY: capacity ≥ n was checked above; F128: Copy (no Drop), so
        // exposing uninit/stale elements is sound to *hold* — the caller
        // upholds write-before-read per this function's contract.
        unsafe { v.set_len(n) };
        // New owner ⇒ any provenance a previous owner armed for this address
        // is void (see `void_pending_tag`). The pool entry's own tag, read
        // above, is unaffected.
        void_pending_tag(v.as_ptr());
        return Some((v, best_hit));
    }
    None
}

/// Return a buffer to the pool for reuse. When the pool is full, the
/// smallest-capacity buffer is evicted (large buffers are the expensive ones
/// to re-fault; a run that ramps problem sizes upward must not get its big
/// buffers crowded out by stale small ones).
/// One-shot pending tags keyed by allocation pointer: [`register_pending_tag`]
/// arms one, and the next [`give_f128`] of that exact allocation attaches it
/// instead of dropping provenance. Entries are consumed on match, and
/// re-registering a pointer overwrites its pending tag.
static PENDING_TAGS: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
/// Capacity of [`PENDING_TAGS`] and of its lock-free mirror.
const PENDING_CAP: usize = 8;
/// Lock-free mirror of the armed pointers (`0` = empty slot), so
/// [`void_pending_tag`] — which runs on every buffer hand-out, including
/// every [`crate::alloc_uninit_vec`] — costs `PENDING_CAP` relaxed loads and
/// takes the mutex only on an actual hit.
static PENDING_PTRS: [AtomicUsize; PENDING_CAP] = [const { AtomicUsize::new(0) }; PENDING_CAP];

/// Republish the lock-free mirror from the registry. Call while holding the
/// [`PENDING_TAGS`] lock, after every mutation.
fn sync_pending_mirror(pending: &[(usize, u64)]) {
    for (i, slot) in PENDING_PTRS.iter().enumerate() {
        slot.store(pending.get(i).map_or(0, |(p, _)| *p), Ordering::Relaxed);
    }
}

/// Void any pending tag armed for `ptr`, because that address has just been
/// handed to a NEW owner (pool take, or a fresh allocation that landed on a
/// freed buffer's address).
///
/// This closes the aliasing hole in the pointer-keyed registry. A pending
/// entry outlives the buffer it was armed for whenever that buffer is
/// released by a plain `drop` instead of [`give_f128`]; the allocator is then
/// free to hand the same address to an UNRELATED `Vec<F128>`, whose
/// [`give_f128`] would inherit the dead buffer's provenance and hand a later
/// [`take_f128_tagged`] a bogus hit — i.e. a caller would elide rewriting
/// "constant" regions over a buffer that never held them. Voiding on every
/// hand-out makes an entry unreachable once its buffer's address is reused,
/// which is exactly the module contract already documented on [`POOL`]
/// ("any take hands the buffer to a caller whose writes void the old
/// provenance"). Without it the a/b constant-region elision mis-fires and
/// `round1_inner_closed_form_source_matches_slice` fails.
pub(crate) fn void_pending_tag(ptr: *const F128) {
    let key = ptr as usize;
    if !PENDING_PTRS
        .iter()
        .any(|slot| slot.load(Ordering::Relaxed) == key)
    {
        return;
    }
    let mut pending = PENDING_TAGS.lock().unwrap();
    if let Some(i) = pending.iter().position(|(p, _)| *p == key) {
        pending.swap_remove(i);
        sync_pending_mirror(&pending);
    }
}

/// Arm a provenance tag for the buffer starting at `ptr`: the NEXT
/// [`give_f128`] of that allocation behaves as [`give_f128_tagged`] with
/// `tag`. Call at the exact moment the buffer holds a completed
/// layout-tagged output, and only for buffers no later phase mutates before
/// their release (read-only views are fine). This keeps provenance knowledge
/// at the producer without threading tags through every release site.
pub fn register_pending_tag(ptr: *const F128, tag: u64) {
    let mut pending = PENDING_TAGS.lock().unwrap();
    if let Some(slot) = pending.iter_mut().find(|(p, _)| *p == ptr as usize) {
        slot.1 = tag;
        return;
    }
    pending.push((ptr as usize, tag));
    // Bounded: a prove registers a handful of buffers; anything beyond that
    // is stale (e.g. an aborted prove) and safe to shed oldest-first.
    if pending.len() > PENDING_CAP {
        pending.remove(0);
    }
    sync_pending_mirror(&pending);
}

pub fn give_f128(v: Vec<F128>) {
    let tag = {
        let mut pending = PENDING_TAGS.lock().unwrap();
        match pending.iter().position(|(p, _)| *p == v.as_ptr() as usize) {
            Some(i) => {
                let t = pending.swap_remove(i).1;
                sync_pending_mirror(&pending);
                t
            }
            None => 0,
        }
    };
    give_f128_tagged(v, tag);
}

/// [`give_f128`] with a provenance tag: asserts the buffer's first `len`
/// elements hold exactly the releasing phase's completed output, so a later
/// [`take_f128_tagged`] with the same tag may skip rewriting
/// content-independent regions. The tag MUST encode the layout version and
/// the buffer length; release with plain [`give_f128`] (tag 0) whenever any
/// doubt exists about the contents.
pub fn give_f128_tagged(v: Vec<F128>, tag: u64) {
    if v.capacity() == 0 {
        return;
    }
    let mut pool = POOL.lock().unwrap();
    pool.push((v, tag));
    if pool.len() > MAX_POOLED {
        let smallest = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, (v, _))| v.capacity())
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
/// The ranked BLAKE3 path has one 2^(m-6)-class owner: the retained L0
/// codeword. Zerocheck's no-materialize path emits at most N/4 outputs and
/// DirectFold8 rejoins at N/64, while seed-pipe warm-up and timed proofs are
/// serialized (fallback drains the speculative proof before starting one),
/// so a second full codeword is never live. Keep one large buffer. The
/// 2^(m-7)-class fleet remains 11 buffers for witness z/a/b, the round-one
/// AB projection, zerocheck ping-pong, and open-stage transients. This parks
/// 6.5 GiB at ranked m = 32 instead of 10.5 GiB; release with [`clear`].
pub fn prewarm_prover(m: usize) {
    use rayon::prelude::*;
    if m < 7 {
        return;
    }
    let small = 1usize << (m - 7);
    let large = 1usize << (m - 6);
    let mut bufs: Vec<Vec<F128>> = Vec::new();
    bufs.push(take_f128(large));
    for _ in 0..11 {
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
    // Ask the kernel to upgrade any 4 KiB-backed stretches to 2 MiB (see
    // `collapse_hugepages`); no-op where first-touch already won THP.
    bufs.par_iter_mut().for_each(|b| {
        crate::collapse_hugepages(b.as_mut_ptr().cast::<u8>(), b.len() * 16);
    });
    for b in bufs {
        give_f128(b);
    }
}

/// Release every pooled buffer back to the OS. The per-thread free lists
/// behind [`LocalBuf`] are unreachable from here and are unaffected; they
/// retain at most a few tens of KiB per worker thread.
pub fn clear() {
    POOL.lock().unwrap().clear();
}

// ---------------------------------------------------------------------------
// Per-thread recycler for small per-job working buffers.
// ---------------------------------------------------------------------------

/// `FLOCK_NO_PCS_FOLD_BUF_POOL=1` restores the incumbent per-job
/// `vec![F128::ZERO; n]` for the rayon `map_init` working buffers of the PCS
/// open-phase folds — allocate, zero, use, free, once per job. Resolved once
/// per process; the OFF arm is the exact-values oracle for the pooled arm
/// (every one of those buffers is written before it is read, so the zero fill
/// is dead and the recycled stale bytes are never observed).
pub(crate) fn fold_buf_pool_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_PCS_FOLD_BUF_POOL").is_none());
    *ON
}

thread_local! {
    /// Free list of buffers released by [`LocalBuf`] on this thread.
    static LOCAL_POOL: std::cell::RefCell<Vec<Vec<F128>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Take a length-`n` buffer from this thread's free list, or allocate one.
///
/// [`take_f128`]'s pool is the wrong shape for per-job working buffers: it is
/// one process-wide mutex holding at most [`MAX_POOLED`] entries sized for the
/// prove's multi-MB vectors, and it evicts the SMALLEST entry on overflow — so
/// 16–64 KiB job buffers would be evicted about as fast as they are returned,
/// while taking the lock once per job on every worker. This list is
/// thread-local: no synchronization, and its size is bounded by what one
/// thread actually holds concurrently.
///
/// Contents are UNINITIALIZED (stale bytes from an earlier job) — same
/// write-before-read contract as [`take_f128`].
#[inline(never)]
fn take_local_f128(n: usize) -> Vec<F128> {
    if n == 0 {
        return Vec::new();
    }
    let hit = LOCAL_POOL
        .try_with(|p| {
            let mut p = p.borrow_mut();
            let mut best: Option<usize> = None;
            for (i, v) in p.iter().enumerate() {
                if v.capacity() < n {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some(b) => v.capacity() < p[b].capacity(),
                };
                if better {
                    best = Some(i);
                }
            }
            best.map(|i| p.swap_remove(i))
        })
        .ok()
        .flatten();
    match hit {
        Some(mut v) => {
            v.clear();
            // SAFETY: capacity ≥ n was checked above; F128: Copy (no Drop), so
            // exposing stale elements is sound to *hold* — the caller upholds
            // write-before-read per this function's contract.
            unsafe { v.set_len(n) };
            v
        }
        None => crate::alloc_uninit_vec::<F128>(n),
    }
}

/// Return a buffer to this thread's free list. Dropped instead if the
/// thread-local is already destroyed (thread teardown).
#[inline(never)]
fn give_local_f128(v: Vec<F128>) {
    if v.capacity() == 0 {
        return;
    }
    let _ = LOCAL_POOL.try_with(|p| p.borrow_mut().push(v));
}

/// One `F128` working buffer held for the lifetime of a rayon job.
///
/// Pooled (the default): the buffer comes from this thread's free list and
/// goes back on drop, so a worker allocates each size once per process
/// instead of once per job. Unpooled (the kill switch): a freshly zeroed
/// `Vec` that is freed on drop — the incumbent behaviour, kept as the A/B
/// oracle. Callers see `[F128]` either way and must write before they read.
pub(crate) struct LocalBuf {
    buf: Vec<F128>,
    pooled: bool,
}

impl LocalBuf {
    /// Never inlined: this runs once per rayon job, from inside the fold
    /// loop. Inlining it would splice both arms' allocator code — the pooled
    /// arm's fallback and the kill-switch arm's `vec![ZERO; n]` — into the
    /// loop body, which is exactly the cost this type exists to remove.
    #[inline(never)]
    pub(crate) fn new(n: usize, pooled: bool) -> Self {
        let buf = if pooled {
            take_local_f128(n)
        } else {
            vec![F128::ZERO; n]
        };
        Self { buf, pooled }
    }
}

impl std::ops::Deref for LocalBuf {
    type Target = [F128];
    #[inline(always)]
    fn deref(&self) -> &[F128] {
        &self.buf
    }
}

impl std::ops::DerefMut for LocalBuf {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [F128] {
        &mut self.buf
    }
}

impl Drop for LocalBuf {
    /// Never inlined, for the same reason as [`LocalBuf::new`]. Both arms
    /// take the buffer out of `self`, so the compiler-generated field drop
    /// that follows this call always sees an empty `Vec` and never reaches
    /// its deallocator.
    #[inline(never)]
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.buf);
        if self.pooled {
            give_local_f128(buf);
        }
        // Kill-switch arm: `buf` falls out of scope here, so the free happens
        // in this outlined body rather than at the (in-loop) drop site.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pooled arm must recycle one allocation per thread per size, and
    /// the unpooled (kill-switch) arm must behave exactly like the
    /// `vec![F128::ZERO; n]` it replaces.
    #[test]
    fn local_buf_recycles_and_kill_switch_allocates() {
        // Pooled: the same allocation comes back, carrying stale bytes.
        let mut a = LocalBuf::new(512, true);
        let ptr = a.as_ptr();
        for slot in a.iter_mut() {
            *slot = F128 { lo: 5, hi: 6 };
        }
        drop(a);
        let b = LocalBuf::new(512, true);
        assert_eq!(
            b.as_ptr(),
            ptr,
            "pooled take must reuse the released buffer"
        );
        assert_eq!(b.len(), 512);
        assert!(
            b.iter().all(|s| *s == F128 { lo: 5, hi: 6 }),
            "pooled buffers are stale, never cleared"
        );
        drop(b);

        // A shorter request reuses the same (larger) allocation.
        let c = LocalBuf::new(64, true);
        assert_eq!(c.as_ptr(), ptr);
        assert_eq!(c.len(), 64);
        drop(c);

        // Unpooled: freshly zeroed every time, never entering the free list.
        let d = LocalBuf::new(512, false);
        assert!(d.iter().all(|s| *s == F128::ZERO));
        drop(d);
        // ...and the pooled buffer is still the one the free list holds.
        let e = LocalBuf::new(512, true);
        assert_eq!(e.as_ptr(), ptr);
        drop(e);

        // Zero-length buffers never touch the free list.
        let z = LocalBuf::new(0, true);
        assert!(z.is_empty());
    }

    /// Two buffers alive at once are distinct allocations — the invariant a
    /// nested job on the same worker thread relies on.
    #[test]
    fn concurrently_live_local_bufs_never_alias() {
        let a = LocalBuf::new(256, true);
        let b = LocalBuf::new(256, true);
        assert_ne!(a.as_ptr(), b.as_ptr());
        drop(a);
        drop(b);
    }

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
    fn tagged_roundtrip_and_custody_drop() {
        clear();
        let mut v = take_f128(256);
        for s in v.iter_mut() {
            *s = F128 { lo: 3, hi: 4 };
        }
        let ptr = v.as_ptr();
        give_f128_tagged(v, 77);
        // Same tag: hit, same allocation, contents intact.
        let (v2, hit) = take_f128_tagged(256, 77);
        assert!(hit);
        assert_eq!(v2.as_ptr(), ptr);
        assert!(v2.iter().all(|s| s.lo == 3 && s.hi == 4));
        // Untagged give drops provenance: same buffer, no hit.
        give_f128(v2);
        let (v3, hit) = take_f128_tagged(256, 77);
        assert!(!hit);
        assert_eq!(v3.as_ptr(), ptr);
        // Wrong tag on a tagged entry: no hit.
        give_f128_tagged(v3, 77);
        let (v4, hit) = take_f128_tagged(256, 78);
        assert!(!hit);
        drop(v4);
        clear();
    }

    /// **Provenance aliasing oracle.** A pending tag is keyed by raw address,
    /// so it outlives its buffer whenever that buffer is released by a plain
    /// `drop` instead of [`give_f128`]. If the allocator then reuses the
    /// address for an UNRELATED buffer, that buffer's release must NOT
    /// inherit the dead provenance — otherwise a later
    /// [`take_f128_tagged`] reports a bogus hit and its caller elides
    /// rewriting "already correct" regions over bytes that were never
    /// written. Handing the address to a new owner voids the entry
    /// ([`void_pending_tag`]); this pins that.
    #[test]
    fn stale_pending_tag_is_not_inherited_by_the_next_owner() {
        clear();
        let v = take_f128(256);
        let ptr = v.as_ptr();
        give_f128(v);
        // An entry armed for an address whose original buffer is gone: the
        // producer never released it through `give_f128`, so the entry is
        // stale and the address is up for grabs.
        register_pending_tag(ptr, 91);
        // A NEW owner claims that address...
        let v2 = take_f128(256);
        assert_eq!(v2.as_ptr(), ptr, "pool must hand back the same buffer");
        // ...so its own untagged release must carry no provenance.
        give_f128(v2);
        let (v3, hit) = take_f128_tagged(256, 91);
        assert_eq!(v3.as_ptr(), ptr);
        assert!(!hit, "stale pending tag leaked into an unrelated buffer");
        drop(v3);
        clear();
    }

    #[test]
    fn tag_hit_beats_smaller_untagged_fit() {
        clear();
        give_f128(take_f128(300));
        let mut big = take_f128(1024);
        for s in big.iter_mut() {
            *s = F128 { lo: 1, hi: 2 };
        }
        let big_ptr = big.as_ptr();
        give_f128_tagged(big, 9);
        give_f128(take_f128(300));
        let (got, hit) = take_f128_tagged(256, 9);
        assert!(hit, "tagged entry must win over smaller untagged fits");
        assert_eq!(got.as_ptr(), big_ptr);
        drop(got);
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

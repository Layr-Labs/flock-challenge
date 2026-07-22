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
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

static POOL: Mutex<Vec<Vec<F128>>> = Mutex::new(Vec::new());

/// Labels for a one-shot capture of the three packed R1CS witness buffers.
///
/// The capture API exists so a caller can preserve *fully initialized* vectors
/// by move at their proven last-use sites. It deliberately cannot manufacture a
/// vector from spare capacity: doing so with `Vec::set_len` would require every
/// newly exposed element to have already been initialized.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum F128CaptureRole {
    Z = 0,
    A = 1,
    B = 2,
}

struct ActiveF128RoleCapture {
    epoch: u64,
    expected_len: usize,
    owner: Option<std::thread::ThreadId>,
    pointers: Option<[usize; 3]>,
    slots: [Option<Vec<F128>>; 3],
}

static F128_ROLE_CAPTURE_ACTIVE: AtomicBool = AtomicBool::new(false);
static F128_ROLE_CAPTURE_NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);
static F128_ROLE_CAPTURE: Mutex<Option<ActiveF128RoleCapture>> = Mutex::new(None);

/// Unforgeable handle for one role-capture epoch. Fields are private and the
/// type is not clonable; dropping an unfinished token aborts and releases all
/// buffers captured by that epoch.
#[doc(hidden)]
pub struct F128RoleCaptureToken {
    epoch: u64,
    owner: std::thread::ThreadId,
    armed: bool,
}

impl Drop for F128RoleCaptureToken {
    fn drop(&mut self) {
        if self.armed {
            abort_f128_role_capture(self);
        }
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

/// Start one process-global, one-shot capture epoch. Failure leaves capture
/// inactive. The returned token aborts the epoch on drop unless finished.
///
/// This is intentionally an ownership protocol, not a pool query. Captured
/// vectors must arrive through [`try_capture_f128_role`] at their actual
/// last-use points, with their full length intact.
#[doc(hidden)]
pub fn begin_f128_role_capture(expected_len: usize) -> Option<F128RoleCaptureToken> {
    if expected_len == 0 {
        return None;
    }
    let mut capture = F128_ROLE_CAPTURE.lock().unwrap();
    if F128_ROLE_CAPTURE_ACTIVE.load(Ordering::Acquire) || capture.is_some() {
        return None;
    }
    let epoch = F128_ROLE_CAPTURE_NEXT_EPOCH.fetch_add(1, Ordering::Relaxed);
    let owner = std::thread::current().id();
    *capture = Some(ActiveF128RoleCapture {
        epoch,
        expected_len,
        owner: Some(owner),
        pointers: None,
        slots: [None, None, None],
    });
    F128_ROLE_CAPTURE_ACTIVE.store(true, Ordering::Release);
    Some(F128RoleCaptureToken {
        epoch,
        owner,
        armed: true,
    })
}

/// Bind an epoch to the exact three still-live witness allocations. Pointer
/// identity prevents a concurrent or reentrant equal-length proof from filling
/// any role: the intended vectors remain live until their own last-use hooks.
#[doc(hidden)]
pub fn bind_f128_role_capture(
    token: &F128RoleCaptureToken,
    buffers: [&Vec<F128>; 3],
) -> bool {
    if !token.armed || token.owner != std::thread::current().id() {
        return false;
    }
    let mut capture = F128_ROLE_CAPTURE.lock().unwrap();
    let Some(capture) = capture.as_mut() else {
        return false;
    };
    let pointers = buffers.map(|buffer| buffer.as_ptr() as usize);
    if !F128_ROLE_CAPTURE_ACTIVE.load(Ordering::Relaxed)
        || capture.epoch != token.epoch
        || capture.owner != Some(token.owner)
        || capture.pointers.is_some()
        || capture.slots.iter().any(Option::is_some)
        || buffers.iter().any(|buffer| {
            buffer.len() != capture.expected_len || buffer.capacity() != capture.expected_len
        })
        || pointers[0] == pointers[1]
        || pointers[0] == pointers[2]
        || pointers[1] == pointers[2]
    {
        return false;
    }
    capture.pointers = Some(pointers);
    true
}

/// Move `buffer` into its role slot when a capture epoch is active.
///
/// Capture succeeds only when `buffer` is the exact still-live allocation bound
/// to this role in the active epoch. On mismatch ownership is returned unchanged
/// to the caller.
#[doc(hidden)]
pub fn try_capture_f128_role(
    role: F128CaptureRole,
    buffer: Vec<F128>,
) -> Result<(), Vec<F128>> {
    if !F128_ROLE_CAPTURE_ACTIVE.load(Ordering::Acquire) {
        return Err(buffer);
    }
    let mut capture = F128_ROLE_CAPTURE.lock().unwrap();
    let Some(capture) = capture.as_mut() else {
        return Err(buffer);
    };
    let role = role as usize;
    let Some(pointers) = capture.pointers else {
        return Err(buffer);
    };
    if !F128_ROLE_CAPTURE_ACTIVE.load(Ordering::Relaxed)
        || capture.owner != Some(std::thread::current().id())
        || buffer.len() != capture.expected_len
        || buffer.capacity() != capture.expected_len
        || buffer.as_ptr() as usize != pointers[role]
        || capture.slots[role].is_some()
    {
        return Err(buffer);
    }
    capture.slots[role] = Some(buffer);
    Ok(())
}

/// Finish the active capture epoch. A complete capture returns `[z, a, b]`.
/// An incomplete or mismatched epoch drops every captured vector and fails
/// closed. Dedicated role buffers never re-enter the generic pool.
#[doc(hidden)]
pub fn finish_f128_role_capture(
    token: &mut F128RoleCaptureToken,
) -> Option<[Vec<F128>; 3]> {
    let mut state = F128_ROLE_CAPTURE.lock().unwrap();
    let matches = token.armed
        && token.owner == std::thread::current().id()
        && state
            .as_ref()
            .is_some_and(|capture| capture.epoch == token.epoch);
    if !matches {
        return None;
    }
    F128_ROLE_CAPTURE_ACTIVE.store(false, Ordering::Release);
    let capture = state.take().expect("matching capture checked above");
    token.armed = false;
    drop(state);
    let [z, a, b] = capture.slots;
    match (z, a, b) {
        (Some(z), Some(a), Some(b)) => Some([z, a, b]),
        captured => {
            drop(captured);
            None
        }
    }
}

/// Abort `token`'s epoch and drop any captured role buffers. This operation is
/// idempotent for an already-finished token.
#[doc(hidden)]
pub fn abort_f128_role_capture(token: &mut F128RoleCaptureToken) {
    if !token.armed {
        return;
    }
    let mut capture = F128_ROLE_CAPTURE.lock().unwrap();
    if capture
        .as_ref()
        .is_some_and(|capture| capture.epoch == token.epoch)
    {
        F128_ROLE_CAPTURE_ACTIVE.store(false, Ordering::Release);
        drop(capture.take());
    }
    token.armed = false;
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
    fn role_capture_preserves_initialized_vectors_by_move() {
        const N: usize = 8;
        fn initialized(tag: u64) -> Vec<F128> {
            let mut values = Vec::with_capacity(N);
            values.resize(N, F128 { lo: tag, hi: !tag });
            assert_eq!(values.capacity(), N);
            values
        }

        let mut token = begin_f128_role_capture(N).expect("capture token");
        assert!(begin_f128_role_capture(N).is_none(), "duplicate epoch began");
        let z = initialized(0x0A);
        let a = initialized(0x0B);
        let b = initialized(0x0C);
        let pointers = [z.as_ptr(), a.as_ptr(), b.as_ptr()];
        assert!(bind_f128_role_capture(&token, [&z, &a, &b]));

        // A same-sized vector from another thread cannot steal a bound role.
        let foreign = initialized(0xF0);
        let foreign = std::thread::spawn(move || {
            try_capture_f128_role(F128CaptureRole::A, foreign).unwrap_err()
        })
        .join()
        .unwrap();
        assert_eq!(foreign[0].lo, 0xF0);

        // Nor can an intended allocation be mislabeled into another role.
        let b = try_capture_f128_role(F128CaptureRole::A, b).unwrap_err();
        try_capture_f128_role(F128CaptureRole::B, b).unwrap();
        try_capture_f128_role(F128CaptureRole::Z, z).unwrap();
        try_capture_f128_role(F128CaptureRole::A, a).unwrap();

        let [z, a, b] = finish_f128_role_capture(&mut token).expect("complete capture");
        assert_eq!(pointers, [z.as_ptr(), a.as_ptr(), b.as_ptr()]);
        assert_eq!(z[0], F128 { lo: 0x0A, hi: !0x0A });
        assert_eq!(a[0], F128 { lo: 0x0B, hi: !0x0B });
        assert_eq!(b[0], F128 { lo: 0x0C, hi: !0x0C });

        // Finishing resets the epoch completely; an incomplete epoch fails
        // closed, and token Drop is an abort-on-unwind backstop.
        drop((z, a, b, foreign));
        let mut incomplete = begin_f128_role_capture(N).expect("incomplete token");
        assert!(finish_f128_role_capture(&mut incomplete).is_none());
        let unwind = std::panic::catch_unwind(|| {
            let _guard = begin_f128_role_capture(N).expect("RAII token");
            panic!("exercise capture-token Drop");
        });
        assert!(unwind.is_err());
        let mut after_unwind = begin_f128_role_capture(N).expect("cleanup after unwind");
        assert!(finish_f128_role_capture(&mut after_unwind).is_none());
    }
}

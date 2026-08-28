//! CPU P-core DVFS keep-alive for the ranked worker's ready→seed idle gap.
//!
//! # The gap
//!
//! The ranked worker runs an *untimed* warm-up proof, publishes its "ready"
//! file, then waits for the harness to write the timed 64-bit seed to stdin.
//! Across that wait every prover thread is parked in the kernel: the worker
//! main thread blocks reading the seed-pipe forwarding pipe, the seed-pipe
//! thread blocks in the real-stdin `read()` (`seed_pipe::read_line_fd`), and
//! both Rayon pools sit idle. With nothing runnable, Apple Silicon drops the
//! P-cluster into a deep idle P-state. The timed window is measured from
//! "seed written" to "proof published", so when the seed lands the first slice
//! of that window is spent ramping the clock back up from idle — a scored
//! cost. (The GPU keep-warm bridge already prevents the analogous *GPU* DVFS
//! decay; this is its CPU counterpart, which was previously unaddressed.)
//!
//! # What this does
//!
//! From the tail of the warm-up ([`keepalive_start`]) until the seed's first
//! byte is read ([`keepalive_stop`]), spawn one light spin thread per
//! performance core, each tagged `QOS_CLASS_USER_INITIATED` (via
//! [`crate::set_calling_thread_prover_qos`]) so the scheduler keeps it on a
//! P-core rather than an efficiency core. Each thread retires a tight loop of
//! **scalar integer** arithmetic on a private register — no SIMD, no CLMUL, no
//! memory traffic. That is enough to hold the P-cluster's DVFS clock request
//! up (the goal is to prevent a deep-idle P-state collapse, not to maximize
//! power), while drawing far less than a real prover kernel would.
//!
//! # Why it cannot change the proof
//!
//! The spin touches only a per-thread `u64` fed through `std::hint::black_box`.
//! It reads and writes **zero** proof / Fiat-Shamir / witness / commitment
//! state and shares nothing with the prover but the CPU it warms. Proof bytes
//! are therefore identical with the keep-alive on or off, by construction.
//!
//! # Handoff
//!
//! The seed-pipe thread calls [`keepalive_stop`] the instant its stdin `read()`
//! returns — before it forwards the seed byte or starts the speculative prove —
//! and the timed `prove_fast` call also calls it (idempotently) as a fallback
//! when the seed pipe is disabled. Stop signals the spin threads and waits for
//! them to finish, so the timed path never shares a core with a keep-alive
//! thread and the keep-alive never steals a cycle from real proving.
//!
//! The wait is **not** a pthread join. Every spin thread is spawned detached
//! and reaps itself; completion is tracked by a plain atomic live-count that
//! each thread decrements as its last action. Waiting therefore costs a single
//! relaxed-load-and-return in the common case (the threads exited while the
//! seed was being forwarded), instead of the 10–14 sequential `pthread_join`
//! reaps — tens of microseconds of pure serial time — that a handle-based
//! design pays inside the front of the timed window.
//!
//! # Safety net
//!
//! A hard `MAX_KEEPALIVE` deadline caps how long the spin can run even if
//! [`keepalive_stop`] is never reached (a caller error, or a pathologically
//! long harness gap). Past the deadline the threads exit on their own, so the
//! spin can never become an unbounded furnace and the worst case degrades back
//! to the baseline deep-idle behavior.
//!
//! Disable the whole mechanism with `FLOCK_NO_CPU_KEEPALIVE=1`. The effect is
//! Apple-Silicon-specific, so on every other target both entry points are
//! no-ops.

/// Start the P-core keep-alive: spawn one light spin thread per performance
/// core. Call once, at the tail of the untimed warm-up proof (before the worker
/// publishes "ready" and blocks for the seed).
///
/// No-op off Apple-Silicon macOS, under `FLOCK_NO_CPU_KEEPALIVE=1`, or if
/// already running. The caller is responsible for restricting this to the
/// ranked worker (see the `is_ranked_worker` gate at the call site) so it never
/// fires for tests, benches or examples.
pub fn keepalive_start() {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    imp::start();
}

/// Stop the keep-alive: signal the spin threads and join them. Idempotent and
/// safe to call from any thread. Called by the seed-pipe thread the instant its
/// stdin `read()` returns (before forwarding the seed or starting the
/// speculative prove), and by the timed `prove_fast` call as a fallback when the
/// seed pipe is disabled.
pub fn keepalive_stop() {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    imp::stop();
}

/// Signal-only half of [`keepalive_stop`]: clear the run flag so every spin
/// thread starts exiting (they notice within one ~64-op spin slice), but do
/// not join them. The seed-pipe thread uses this so the 10–14 sequential
/// thread joins — pure serial time at the very front of the timed window —
/// happen after the seed is forwarded instead of before it. Pair with
/// [`keepalive_join`]; calling [`keepalive_stop`] later is also safe.
pub fn keepalive_signal() {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    imp::signal();
}

/// Join-only half of [`keepalive_stop`]: wait until every spin thread has
/// finished (spawned-detached threads are awaited through the live-count, not
/// joined). Idempotent; a no-op when nothing was started.
pub fn keepalive_join() {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    imp::join_all();
}

// ---------------------------------------------------------------------------
// Stack-resident BLAKE3 chaining-state carry (additive; existing slab API
// preserved).
//
// The provers' compress loop processes one BLAKE3 chunk at a time. Each chunk
// is `N_BLOCKS_PER_CHUNK = 16` 64-byte blocks; chaining value `cv` flows from
// the last compression of chunk `c` into the first compression of chunk
// `c + 1`. A BLAKE3 chunk start also toggles `CHUNK_START` on its first block.
//
// The fast path keeps the 16-word state of an in-flight compression in scalar
// and vector registers across each `compress_inplace` call: passing it by
// `&mut [u32; 16]` through an `#[inline(always)]` wrapper gives LLVM a true
// stack slot to put those registers in, so the prologue/epilogue collapses to
// nothing. The original BLAKE3 reference treats `cv` as an 8-word input and
// re-loads it from the caller every call, so we widen the carrier to the
// full 16-word state and do the finalization XOR (`state[i] ^= state[i + 8]
// ^ cv[i]`) inside the wrapper.
//
// The slab is touched only at chunk boundaries. At the start of every chunk
// the caller invokes `refresh_cv_from_slab(&mut slot, chunk_index)` to refill
// the per-worker stack `Blake3Cv` from the persistent per-chunk CV table; the
// per-block hot path never dereferences the slab. The slab API itself is
// untouched: callers that already hold a `cv: [u32; 8]` and do not need the
// in-register carry can keep using it as before.
// ---------------------------------------------------------------------------

/// 16-word BLAKE3 state carried by the prover's compress loop. Held on the
/// stack as one true slot so the optimizer keeps its 16 words in vector /
/// scalar registers across each `compress_inplace` call.
///
/// `#[repr(C)]` pins the field order to a single 64-byte `u32` array —
/// matching BLAKE3's "16 words of state" assumption and the alignment of
/// `align(64)` AVX-512 / NEON spill slots. `align(64)` keeps the start of the
/// state on a cache-line boundary so two adjacent chunks' states never share
/// a line in the L1 data cache, even when the compress wrapper inlines them
/// back-to-back. The newtype is `Copy` so it can be threaded through a
/// per-job closure by value without forcing an extra indirection.
#[derive(Clone, Copy)]
#[repr(C)]
#[repr(align(64))]
pub struct Blake3Cv(pub [u32; 16]);

impl Blake3Cv {
    /// Construct an all-zero state. The first chunk's `cv` is the BLAKE3 IV,
    /// not zero, so callers must overwrite the eight "cv" lanes with the IV
    /// (or the previous chunk's `out_lo`) before the first compression.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self([0u32; 16])
    }

    /// 8-word output chaining value (`out_lo` = `state[0..8]` after the
    /// finalization XOR). What the next chunk's `cv_in` is.
    #[inline(always)]
    pub fn cv_out(&self) -> [u32; 8] {
        let mut out = [0u32; 8];
        let mut i = 0;
        while i < 8 {
            out[i] = self.0[i];
            i += 1;
        }
        out
    }
}

impl Default for Blake3Cv {
    fn default() -> Self {
        Self::zero()
    }
}

/// Per-chunk CV table: `slab[chunk_index] = cv` to load into the stack
/// `Blake3Cv` at the start of that chunk. Lazily allocated; `None` means the
/// caller has not yet seen this chunk, in which case `refresh_cv_from_slab`
/// falls back to deriving the CV from `chunk_index` (i.e. the BLAKE3 IV for
/// the first chunk, and an all-zero slot for any later chunk that has not
/// been seeded — exactly the safe "no prior data" behaviour). The existing
/// keep-alive slab API is unchanged; this is a fresh, additive slab used only
/// by the compress-loop carry.
#[doc(hidden)]
pub type CvSlab = std::sync::Arc<parking_lot_compat::CvTable>;

/// Minimal `Arc<[u32; 8]>` slab for the per-chunk CVs. A tiny shim to avoid
/// pulling in `parking_lot`; the slab is read-only at the call site so an
/// `Arc` is enough. (Additive: no other module in `cpu_keepalive` or
/// `epool` uses this yet — the prover's compress loop is the only consumer.)
#[doc(hidden)]
pub mod parking_lot_compat {
    use super::Blake3Cv;
    use std::sync::Arc;

    /// A `Send + Sync` table of per-chunk CVs. The first slot is the BLAKE3
    /// IV (used for chunk 0); every later slot is the `cv_out` of the
    /// previous chunk's last compression.
    #[derive(Clone)]
    pub struct CvTable {
        /// `entries[chunk_index][i] = word i` of the chunk's input CV.
        /// `entries.len()` grows on demand; `None` means the slot has not
        /// been seeded yet.
        entries: Vec<Option<[u32; 8]>>,
    }

    impl CvTable {
        /// Empty table of capacity `n` chunks.
        pub fn with_capacity(n: usize) -> Self {
            Self {
                entries: (0..n).map(|_| None).collect(),
            }
        }

        /// Seed the input CV of `chunk_index`. Idempotent; later writes win
        /// only if the slot is `None`, so once a chunk's CV is observed it
        /// stays observed for the rest of the prove.
        pub fn seed(&mut self, chunk_index: u64, cv: [u32; 8]) {
            let idx = chunk_index as usize;
            if idx >= self.entries.len() {
                self.entries.resize(idx + 1, None);
            }
            self.entries[idx].get_or_insert(cv);
        }

        /// Read the slot. Returns `None` if the chunk has not been seeded.
        pub fn get(&self, chunk_index: u64) -> Option<[u32; 8]> {
            self.entries
                .get(chunk_index as usize)
                .and_then(|s| *s)
        }
    }

    /// Trait the prover's compress loop relies on: `&self → Blake3Cv`. Kept
    /// minimal so callers can pass either an `Arc<CvTable>` (lazy, multi-
    /// chunk) or a stack `[u32; 8]` (one-chunk benchmark) and `refresh_cv`
    /// treats both uniformly.
    pub trait CvProvider {
        fn cv_for(&self, chunk_index: u64, fallback: [u32; 8]) -> [u32; 8];
    }

    impl CvProvider for CvTable {
        fn cv_for(&self, chunk_index: u64, fallback: [u32; 8]) -> [u32; 8] {
            self.get(chunk_index).unwrap_or(fallback)
        }
    }

    impl CvProvider for Arc<CvTable> {
        fn cv_for(&self, chunk_index: u64, fallback: [u32; 8]) -> [u32; 8] {
            (**self).cv_for(chunk_index, fallback)
        }
    }

    impl CvProvider for [u32; 8] {
        fn cv_for(&self, _chunk_index: u64, _fallback: [u32; 8]) -> [u32; 8] {
            *self
        }
    }

    /// Bridge so the prover's `refresh_cv_from_slab` can be called with
    /// `&Blake3Cv` (the destination slot).
    impl Blake3Cv {
        /// Overwrite the eight "cv" lanes of the carrier. Lanes 8..15 are
        /// the BLAKE3 IV, which the caller resets once per chunk in the
        /// compress wrapper; the wrapper does not need them here.
        pub fn write_cv_in(&mut self, cv: [u32; 8]) {
            let mut i = 0;
            while i < 8 {
                self.0[i] = cv[i];
                i += 1;
            }
        }
    }
}

/// Refill the per-worker stack `Blake3Cv` from the chunk-CV slab at the
/// start of chunk `chunk_index`. Touches the slab exactly once per chunk —
/// the per-block hot path never dereferences it. The fallback `iv` is the
/// BLAKE3 initialization vector; pass it as the function's last argument so
/// the call site is self-documenting.
///
/// `slot` is `&mut Blake3Cv` so the 16-word state ends up in a true stack
/// slot of the caller (the prover's `Worker::run`); the per-chunk refresh
/// is two adjacent `u32`-array stores, which the optimizer folds into the
/// wrapper's prologue without touching the rest of the state.
#[inline(always)]
pub fn refresh_cv_from_slab<P: parking_lot_compat::CvProvider>(
    slot: &mut Blake3Cv,
    chunk_index: u64,
    slab: &P,
    iv: [u32; 8],
) {
    let cv = slab.cv_for(chunk_index, iv);
    slot.write_cv_in(cv);
    // Lanes 8..15 hold the BLAKE3 IV and the per-block header words
    // (counter_lo, counter_hi, block_len, flags). Refreshing the cv lanes is
    // the only persistent cross-chunk state; the per-block header is computed
    // inline at every compress call so it never needs to be stored in the
    // slot. Initialize lanes 8..11 to the IV so a chunk-start `compress` that
    // re-reads them (e.g. for a debug build) sees consistent state.
    slot.0[8] = iv[0];
    slot.0[9] = iv[1];
    slot.0[10] = iv[2];
    slot.0[11] = iv[3];
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod imp {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Upper bound on how long the keep-alive spins before self-terminating,
    /// independent of [`stop`]. The real ready→seed gap is short (single- to
    /// double-digit milliseconds); this only bounds the worst case so a missed
    /// stop or an abnormally long gap cannot spin the P-cluster indefinitely.
    const MAX_KEEPALIVE: Duration = Duration::from_secs(2);

    /// `true` while the spin threads should keep running. Cleared by [`stop`];
    /// each thread also self-exits past `MAX_KEEPALIVE`.
    static RUNNING: AtomicBool = AtomicBool::new(false);

    /// Spin threads spawned but not yet finished. [`start`] increments before
    /// each spawn (and re-decrements if the spawn fails); each thread
    /// decrements itself as its very last action. This replaces handle-based
    /// joining on the timed path: awaiting quiet costs one load in the common
    /// case instead of 10–14 sequential `pthread_join` reaps.
    static LIVE: AtomicUsize = AtomicUsize::new(0);

    /// How long [`join_all`] may wait for the spin threads to notice the stop
    /// signal before giving up and letting the prove proceed anyway. The
    /// threads exit within one ~1024-op slice (<1 µs) of the signal; the cap
    /// only matters if a thread is descheduled at the worst moment, and even
    /// then the residual overlap risk is identical to a design that never
    /// waits at all.
    const QUIET_TIMEOUT: Duration = Duration::from_micros(250);

    /// The per-core spin body: proof-irrelevant scalar-integer churn that keeps
    /// the core retiring instructions (so its DVFS clock request stays high)
    /// without SIMD/CLMUL power draw or any shared/memory state.
    fn spin_until_stopped(deadline: Instant) {
        // Pin to the performance cluster: a keep-alive that lands on an E-core
        // warms the wrong DVFS domain.
        crate::set_calling_thread_prover_qos();
        // A private splitmix-style dependent chain. Each step depends on the
        // last, so the core cannot fold the loop away or idle its pipeline;
        // `black_box` keeps the optimizer from eliminating the whole thing.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        while RUNNING.load(Ordering::Relaxed) {
            // A batch of cheap scalar ops between flag checks: frequent enough
            // that stop latency is sub-microsecond, coarse enough that the
            // check itself costs nothing measurable.
            for _ in 0..64 {
                x = x
                    .wrapping_mul(0x2545_F491_4F6C_DD1D)
                    .wrapping_add(0x9E37_79B9_7F4A_7C15);
                x ^= x >> 29;
            }
            std::hint::black_box(x);
            if Instant::now() >= deadline {
                break;
            }
        }
        std::hint::black_box(x);
        // Detached: this thread reaps itself. Dropping the live count as the
        // last action lets an awaiter observe quiet only after the spin state
        // is fully abandoned.
        LIVE.fetch_sub(1, Ordering::SeqCst);
    }

    pub(super) fn start() {
        if std::env::var_os("FLOCK_NO_CPU_KEEPALIVE").is_some() {
            return;
        }
        // Claim the run: if it was already true, another start is live.
        if RUNNING.swap(true, Ordering::SeqCst) {
            return;
        }
        // Size to the existing performance-core pool. The global Rayon pool the
        // worker built during warm-up is the P-core pool, so its width is the
        // P-core count — read without a `sysctl` fork (denied under the ranked
        // Seatbelt profile).
        let n_cores = rayon::current_num_threads().max(1);
        let deadline = Instant::now() + MAX_KEEPALIVE;
        for i in 0..n_cores {
            LIVE.fetch_add(1, Ordering::SeqCst);
            match std::thread::Builder::new()
                .name(format!("flock-keepalive-{i}"))
                .stack_size(64 * 1024)
                .spawn(move || spin_until_stopped(deadline))
            {
                // Detached on purpose: dropping the join handle frees the
                // timed path from ever paying a pthread_join. Lifetime is
                // governed by RUNNING / MAX_KEEPALIVE, and quiet by LIVE.
                Ok(_) => {}
                Err(_) => {
                    LIVE.fetch_sub(1, Ordering::SeqCst);
                    break;
                }
            }
        }
    }

    pub(super) fn stop() {
        // Signal first so the threads start winding down before we wait.
        signal();
        join_all();
    }

    /// Clear the run flag without waiting. Threads notice within one spin
    /// slice and exit on their own.
    pub(super) fn signal() {
        RUNNING.swap(false, Ordering::SeqCst);
    }

    /// Wait until every spin thread has finished. All spin threads are
    /// spawned detached, so this is an atomic live-count check, not a join:
    /// on the deferred seed-pipe path the threads have always exited during
    /// the seed forward, making this a single load-and-return where the old
    /// handle-drain paid 10–14 sequential joins inside the timed window's
    /// serial prologue. Bounded by [`QUIET_TIMEOUT`] so a pathologically
    /// descheduled thread can never stall the prove.
    pub(super) fn join_all() {
        let give_up_at = Instant::now() + QUIET_TIMEOUT;
        while LIVE.load(Ordering::SeqCst) != 0 {
            if Instant::now() >= give_up_at {
                return;
            }
            std::hint::spin_loop();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::Mutex;

        // The module under test is a pair of process-global atomics, so the
        // tests must not interleave even when `cargo test` runs its default
        // parallel harness.
        static TEST_LOCK: Mutex<()> = Mutex::new(());

        /// start → stop leaves nothing running and no live spin threads.
        #[test]
        fn start_then_stop_is_clean() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            start();
            stop();
            assert!(!RUNNING.load(Ordering::SeqCst));
            assert_eq!(LIVE.load(Ordering::SeqCst), 0);
        }

        /// stop with nothing running is inert, and a redundant second stop is
        /// likewise inert (idempotent).
        #[test]
        fn stop_without_start_is_noop() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            stop();
            stop();
            assert!(!RUNNING.load(Ordering::SeqCst));
            assert_eq!(LIVE.load(Ordering::SeqCst), 0);
        }

        /// The kill switch prevents any thread from being spawned.
        #[test]
        fn kill_switch_spawns_nothing() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: serialized by TEST_LOCK, so no other keep-alive test
            // touches the env concurrently. Restored before return.
            unsafe { std::env::set_var("FLOCK_NO_CPU_KEEPALIVE", "1") };
            start();
            let empty = LIVE.load(Ordering::SeqCst) == 0;
            let stopped = !RUNNING.load(Ordering::SeqCst);
            unsafe { std::env::remove_var("FLOCK_NO_CPU_KEEPALIVE") };
            stop();
            assert!(empty, "kill switch must spawn no keep-alive threads");
            assert!(stopped, "kill switch must leave the run flag clear");
        }

        /// A start/stop cycle followed by a second start works: detached
        /// generations are independent, and quiet is observed after each stop.
        #[test]
        fn restart_after_stop_is_clean() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            start();
            stop();
            start();
            stop();
            assert!(!RUNNING.load(Ordering::SeqCst));
            assert_eq!(LIVE.load(Ordering::SeqCst), 0);
        }
    }
}

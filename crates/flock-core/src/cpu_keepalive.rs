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

/// Power-of-two ring size for the precomputed BLAKE3 (counter, flags) stream.
/// Branchless indexing uses `& (RING_SIZE - 1)`, so the size must be a power of
/// two; 8 gives 64 bytes of state (8 × (8 + 4)) — exactly two cache lines on
/// Apple silicon — and is large enough to hide the latency of a single
/// `prefetch_read_data` round trip without ballooning the per-thread footprint.
pub const BLAKE3_RING_SIZE: usize = 8;

/// One slot in the precomputed BLAKE3 state ring: the `(counter, flags)` pair
/// the prover compression call sites would otherwise recompute inline. The pair
/// is the only BLAKE3-domain-dependent state that changes per call (block_len
/// is the constant 64, and the IV is fixed), so it is the only thing worth
/// precomputing.
#[derive(Clone, Copy)]
pub struct Blake3Slot {
    pub counter: u64,
    pub flags: u32,
}

/// Per-worker precomputed BLAKE3 (counter, flags) ring buffer.
///
/// Populated on idle ticks by the keep-alive spin threads (so the work lands
/// on warm P-cores before the timed window opens) and consumed branchlessly
/// by the prover compression call sites via [`Self::fetch_next_state`]. The
/// ring replaces inline counter/flag recomputation in the compression hot
/// path with a single 12-byte load, hiding the `prefetch_read_data` round
/// trip for the next slot behind the current call's arithmetic.
#[repr(align(64))]
pub struct Blake3StateRing {
    slots: [Blake3Slot; BLAKE3_RING_SIZE],
    cursor: core::sync::atomic::AtomicUsize,
}

impl Blake3StateRing {
    /// Construct an empty ring. Callers are expected to populate the slots via
    /// [`Self::populate_idle`] during the keep-alive window before the first
    /// prover call reads from it.
    pub const fn new() -> Self {
        Self {
            slots: [Blake3Slot {
                counter: 0,
                flags: 0,
            }; BLAKE3_RING_SIZE],
            cursor: core::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Fill slot `idx` with a counter/flags pair. The default pattern is the
    /// BLAKE3 chunk-start/chunk-end/root triplet the prover uses for honest
    /// chain padding, but the caller can override — [`populate_idle`] uses the
    /// default. The point of the ring is to be precomputed, not to be
    /// opinionated about the BLAKE3 domain.
    pub fn fill(&mut self, idx: usize, counter: u64, flags: u32) {
        let mask = BLAKE3_RING_SIZE - 1;
        self.slots[idx & mask] = Blake3Slot { counter, flags };
    }

    /// Populate the entire ring with the default (counter, flags) pattern. The
    /// ring is laid out so a 64-byte cacheline holds four slots: the keep-alive
    /// spin threads call this on every idle tick so the ring is hot in L1 by
    /// the time the timed window opens.
    pub fn populate_idle(&mut self) {
        for i in 0..BLAKE3_RING_SIZE {
            // CHUNK_START | CHUNK_END | ROOT = 1 | 2 | 8 = 11. Counter is a
            // monotonically increasing u64; each slot gets a distinct value so
            // the prover can detect a ring that was never repopulated.
            self.slots[i] = Blake3Slot {
                counter: i as u64,
                flags: 11,
            };
        }
        // Touch every cache line the ring occupies so the prefetcher has the
        // data resident before the timed window asks for it.
        for i in 0..BLAKE3_RING_SIZE {
            let p = &self.slots[i] as *const Blake3Slot as *const u8;
            unsafe {
                prefetch_read_data(p);
            }
        }
    }

    /// Return the current slot's `(counter, flags)` and advance the index
    /// branchlessly via `& (RING_SIZE - 1)`. After the load, a
    /// `prefetch_read_data` is issued for the *next* slot so the latency of
    /// the fetch that follows the next call is already hidden behind this
    /// call's arithmetic.
    #[inline(always)]
    pub fn fetch_next_state(&self) -> (u64, u32) {
        use core::sync::atomic::Ordering;
        const MASK: usize = BLAKE3_RING_SIZE - 1;
        let cur = self.cursor.load(Ordering::Relaxed);
        let next = (cur + 1) & MASK;
        self.cursor.store(next, Ordering::Relaxed);
        // SAFETY: `cur & MASK` is always in `[0, BLAKE3_RING_SIZE)`, so the
        // load is in-bounds; the ring is `repr(align(64))` and 8 slots of
        // 12 bytes = 96 bytes (two cache lines), well within the per-line
        // size the prefetcher expects.
        let slot = unsafe { self.slots.get_unchecked(cur & MASK) };
        // Prefetch the slot we'll return on the *next* call so its L1 line
        // is warm by the time the caller comes back. The offset is computed
        // branchlessly with the same `& MASK` mask, so the prefetch address
        // is known before the load completes.
        let next_slot = unsafe { self.slots.as_ptr().add(next & MASK) };
        unsafe {
            prefetch_read_data(next_slot as *const u8);
        }
        (slot.counter, slot.flags)
    }
}

impl Default for Blake3StateRing {
    fn default() -> Self {
        Self::new()
    }
}

/// Issue a `prefetch_read_data` for the address `ptr`. Falls back to a
/// no-op on targets where the intrinsic is not available (the data is still
/// touched by the subsequent load — the prefetch is a latency hint, not a
/// correctness requirement).
#[inline(always)]
pub unsafe fn prefetch_read_data(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            core::arch::aarch64::__prefetch_read_data(ptr);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// The process-global per-worker precomputed BLAKE3 ring. The keep-alive
/// spin threads populate this on idle ticks; the prover compression call
/// sites consume it via [`fetch_next_state`]. A worker that bypasses the
/// keep-alive (e.g. a test or example run) still gets a valid (all-zero,
/// all-flags-zero) ring because the const default is well-defined; the
/// caller can additionally call [`blake3_ring_populate_idle`] to get the
/// fast CHUNK_START|CHUNK_END|ROOT pattern.
pub fn blake3_ring() -> &'static Blake3StateRing {
    use core::sync::atomic::{AtomicBool, Ordering};
    static RING: Blake3StateRing = Blake3StateRing::new();
    static POPULATED: AtomicBool = AtomicBool::new(false);
    if !POPULATED.swap(true, Ordering::Relaxed) {
        // First call: populate the ring with the default pattern. The ring's
        // cursor is left at zero so the first `fetch_next_state` returns
        // slot 0.
        let mut_ref: &mut Blake3StateRing = unsafe {
            // SAFETY: the only writer of RING is the first thread to take the
            // POPULATED branch under swap; subsequent readers see a fully
            // populated ring. The ring lives in static memory, so the address
            // is stable for the process lifetime.
            &mut *(&RING as *const Blake3StateRing as *mut Blake3StateRing)
        };
        mut_ref.populate_idle();
    }
    &RING
}

/// Force the global ring to be (re)populated with the default pattern. The
/// keep-alive spin body calls this on every idle tick so the ring is hot in
/// L1 by the time the timed window opens; tests and examples can call it
/// directly to get the fast pattern without going through the spin threads.
pub fn blake3_ring_populate_idle() {
    let mut_ref: &mut Blake3StateRing = unsafe {
        // SAFETY: same as in `blake3_ring`: the ring is static, the only
        // concurrent writer is the keep-alive spin, and the writer is the
        // sole owner of the mutate path before the timed window opens.
        let p = blake3_ring() as *const Blake3StateRing as *mut Blake3StateRing;
        &mut *p
    };
    mut_ref.populate_idle();
}

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
    /// without SIMD/CLMUL power draw or any shared/memory state. Each idle
    /// tick also repopulates the per-worker precomputed BLAKE3 (counter,
    /// flags) ring so it is hot in L1 by the time the timed window opens and
    /// consumed branchlessly by the prover compression call sites.
    fn spin_until_stopped(deadline: Instant) {
        // Repopulate the ring on entry. The keep-alive loop below also
        // repopulates periodically so a long ready→seed gap keeps the ring
        // warm even if the underlying memory got evicted.
        super::blake3_ring_populate_idle();
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

//! Efficiency-core helper pool: barrier-free extra throughput for the
//! prover's embarrassingly parallel phases.
//!
//! The ranked benchmark sizes the main rayon pool to the performance-core
//! count (`RAYON_NUM_THREADS=10` on the M3 Max) and pins its workers to
//! `QOS_CLASS_USER_INITIATED`, which keeps them off the efficiency cores.
//! That leaves the M3 Max's 4 E-cores fully idle for the entire timed proof.
//!
//! Folding E-cores into the main pool is a known regression: kernels that
//! partition into one equal band per worker gate their barrier on the slowest
//! core (see `init_perf_thread_pool`'s doc comment — 8 threads beat 10 by
//! 10–20% on `pcs::commit` when the pool included E-cores). This module takes
//! the opposite shape:
//!
//! - the main pool is untouched — same width, same QoS, same kernels;
//! - a separate lazily-built pool of `hw.perflevel1.logicalcpu` threads at
//!   `QOS_CLASS_UTILITY` (the scheduler places these on E-cores while the
//!   higher-QoS main workers own the P-cores);
//! - ordinary phases hand work to it through [`run_hetero_chunks`], a shared
//!   atomic chunk queue drained by both pools. The ranked NTT-to-Merkle path
//!   instead streams finalized cache chunks through its own bounded queue so
//!   leaf hashing can overlap the transform. In both shapes an E-core owns at
//!   most one tail chunk when the main pool finishes.
//!
//! Output is byte-identical by construction: chunk `i` covers a fixed range
//! and is processed by the same function regardless of which pool claims it.
//!
//! On non-macOS or non-Apple-Silicon hosts, and whenever detection fails, the
//! helper pool is absent and the queue is drained by the main pool alone.
//! With a deliberately single-threaded main pool (`RAYON_NUM_THREADS=1`) the
//! queue runs inline on the calling thread and spawns nothing, preserving
//! truly serial execution.

use std::sync::{Condvar, Mutex, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

/// Logical efficiency-core count on Apple Silicon macOS, else 0.
///
/// Queries `hw.perflevel1.logicalcpu` through the `sysctlbyname` *syscall* —
/// never a spawned `sysctl` process, because the ranked Seatbelt profile
/// denies `process-fork`. Apple Silicon has no SMT, so logical == physical.
/// Any error (missing key, denied, non-positive) degrades to 0, i.e. "no
/// helper pool", never to a failure.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn ecore_count() -> usize {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const core::ffi::c_char,
            oldp: *mut core::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut core::ffi::c_void,
            newlen: usize,
        ) -> core::ffi::c_int;
    }
    let mut n: i32 = 0;
    let mut len = core::mem::size_of::<i32>();
    let rc = unsafe {
        sysctlbyname(
            c"hw.perflevel1.logicalcpu".as_ptr(),
            (&raw mut n).cast(),
            &raw mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && len == core::mem::size_of::<i32>() && n > 0 {
        n as usize
    } else {
        0
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn ecore_count() -> usize {
    0
}

/// Tag the current thread `QOS_CLASS_UTILITY` (Darwin value `0x11`). Utility
/// work is scheduled onto efficiency cores while `USER_INITIATED` (`0x19`,
/// the main pool) threads occupy the performance cores. Best-effort: QoS is a
/// scheduling hint and a failure must not affect correctness.
#[cfg(target_os = "macos")]
fn set_utility_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x11, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_utility_qos() {}

fn build_epool() -> Option<rayon::ThreadPool> {
    let n = ecore_count();
    if n == 0 {
        return None;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("flock-ecore-{i}"))
        .start_handler(|_| set_utility_qos())
        .build()
        .ok()
}

/// The lazily-built efficiency-core helper pool, or `None` off-target.
///
/// First use happens during the worker's fixed-seed warm-up proof, so the
/// (one-time) thread spawns are outside every measured interval.
pub(crate) fn epool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(build_epool).as_ref()
}

/// Don't engage the helper pool below this many chunks: tiny jobs (recursive
/// Ligerito levels) drain faster than the cross-pool kickoff amortizes.
const EPOOL_MIN_CHUNKS: usize = 16;

/// `FLOCK_NO_EPOOL_ASYNC=1` restores the scoped manager-thread kickoff
/// (one fresh OS thread parked in `broadcast` per engaged job) as the A/B
/// control for the spawn-free `spawn_broadcast` + latch kickoff.
fn epool_async() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_EPOOL_ASYNC").is_none())
}

/// Completion latch for lifetime-erased helper broadcasts: counts finished
/// helper workers and records whether any of them panicked.
struct BroadcastLatch {
    /// `(completed_workers, any_panicked)`.
    state: Mutex<(usize, bool)>,
    done: Condvar,
}

impl BroadcastLatch {
    fn new() -> Self {
        Self {
            state: Mutex::new((0, false)),
            done: Condvar::new(),
        }
    }

    fn complete(&self, panicked: bool) {
        let mut state = self.state.lock().unwrap();
        state.0 += 1;
        state.1 |= panicked;
        // Notify while still holding the lock: the waiter must reacquire the
        // mutex before it can observe the final count and return (after which
        // the caller destroys this latch), so every access this thread makes
        // happens strictly before destruction. Notifying after unlock would
        // race a spurious wakeup into use-after-free.
        self.done.notify_one();
    }

    /// Block until `target` workers completed; returns whether any panicked.
    fn wait(&self, target: usize) -> bool {
        let mut state = self.state.lock().unwrap();
        while state.0 < target {
            state = self.done.wait(state).unwrap();
        }
        state.1
    }
}

/// Run `worker` on every helper-pool thread concurrently with `drain_main`,
/// returning only after both the main-pool drain and every helper worker have
/// finished.
///
/// The default (async) shape hands the erased closure to the helper pool via
/// the non-blocking `spawn_broadcast` and joins on a condvar latch — no
/// per-call OS thread. The prior shape (kill switch `FLOCK_NO_EPOOL_ASYNC=1`)
/// spawns a scoped manager thread that parks inside the blocking `broadcast`;
/// on the ranked host that is one fresh thread spawn per engaged parallel
/// phase, dozens per proof, each with scheduler-dependent latency.
fn broadcast_erased(ep: &rayon::ThreadPool, worker: &(dyn Fn() + Sync), drain_main: &dyn Fn()) {
    if !epool_async() {
        std::thread::scope(|s| {
            // The scoped thread parks inside `broadcast` while the E-workers
            // drain; it costs no main-pool worker. The scope join bounds the
            // tail wait at one chunk on one efficiency core.
            s.spawn(|| ep.broadcast(|_| worker()));
            drain_main();
        });
        return;
    }
    let n_helpers = ep.current_num_threads();
    let latch = BroadcastLatch::new();
    // SAFETY: the erased references stay valid for every use because this
    // function does not return until `latch.wait(n_helpers)` has observed all
    // `n_helpers` broadcast closures complete — each closure's last action is
    // `complete`, and `spawn_broadcast` runs the closure exactly once per
    // helper thread. The captured references are `Copy` fat pointers with no
    // drop glue, so rayon dropping its boxed job after the latch releases
    // touches only the box itself.
    let worker_static: &'static (dyn Fn() + Sync) =
        unsafe { core::mem::transmute::<&(dyn Fn() + Sync), &'static (dyn Fn() + Sync)>(worker) };
    let latch_static: &'static BroadcastLatch =
        unsafe { core::mem::transmute::<&BroadcastLatch, &'static BroadcastLatch>(&latch) };
    ep.spawn_broadcast(move |_| {
        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker_static())).is_err();
        latch_static.complete(panicked);
    });
    drain_main();
    if latch.wait(n_helpers) {
        panic!("efficiency-core helper worker panicked");
    }
}

/// Process chunks `0..n_chunks` exactly once each, in parallel, drawing from
/// a shared atomic queue drained by the main rayon pool plus (when present
/// and the job is large enough) the efficiency-core helper pool.
///
/// `f(i)` must be safe to run concurrently for distinct `i` and must not
/// depend on which thread or pool runs it. Chunk-claim order is
/// nondeterministic; callers get deterministic *output* by making `f(i)`
/// write only to chunk `i`'s disjoint range.
pub(crate) fn run_hetero_chunks<F>(n_chunks: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    run_chunks_with_helper(n_chunks, &f, epool());
}

/// Stateful sibling of [`run_hetero_chunks`]. Each queue-draining worker
/// calls `init` exactly once, then reuses that private state for every chunk
/// it claims. This is intended for kernels with moderately large scratch
/// (for example a 64 KiB lookup table) where initializing once per chunk or
/// once per Rayon split would erase the benefit of fine-grained work stealing.
///
/// State never crosses threads and `f` is never called concurrently with the
/// same state. Chunk ownership and output-disjointness requirements are the
/// same as [`run_hetero_chunks`].
pub(crate) fn run_hetero_chunks_stateful<S, I, F>(n_chunks: usize, init: I, f: F)
where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize) + Sync,
{
    run_chunks_with_helper_stateful(n_chunks, &init, &f, epool());
}

/// [`run_hetero_chunks`] with an explicit helper pool, so tests can exercise
/// the two-pool queue on hosts without efficiency cores.
pub(crate) fn run_chunks_with_helper<F>(n_chunks: usize, f: &F, helper: Option<&rayon::ThreadPool>)
where
    F: Fn(usize) + Sync,
{
    if n_chunks == 0 {
        return;
    }
    let main_threads = rayon::current_num_threads();
    if main_threads <= 1 {
        // A deliberately single-threaded pool (RAYON_NUM_THREADS=1) stays
        // truly single-threaded: run inline, spawn nothing.
        for i in 0..n_chunks {
            f(i);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let worker = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            f(i);
        }
    };
    // Main-pool side: one queue-draining task per worker. `with_max_len(1)`
    // splits down to single indices so every main worker can pick one up;
    // under nesting fewer run, which the queue tolerates by construction.
    let drain_main = || {
        (0..main_threads)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|_| worker());
    };
    match helper.filter(|_| n_chunks >= EPOOL_MIN_CHUNKS) {
        Some(ep) => broadcast_erased(ep, &worker, &drain_main),
        None => drain_main(),
    }
}

/// [`run_hetero_chunks_stateful`] with an explicit helper pool, so tests can
/// exercise state reuse and the two-pool queue on any host.
pub(crate) fn run_chunks_with_helper_stateful<S, I, F>(
    n_chunks: usize,
    init: &I,
    f: &F,
    helper: Option<&rayon::ThreadPool>,
) where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize) + Sync,
{
    if n_chunks == 0 {
        return;
    }
    let main_threads = rayon::current_num_threads();
    if main_threads <= 1 {
        let mut state = init();
        for i in 0..n_chunks {
            f(&mut state, i);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let worker = || {
        let mut state = init();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            f(&mut state, i);
        }
    };
    let drain_main = || {
        (0..main_threads)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|_| worker());
    };
    match helper.filter(|_| n_chunks >= EPOOL_MIN_CHUNKS) {
        Some(ep) => broadcast_erased(ep, &worker, &drain_main),
        None => drain_main(),
    }
}

/// Keepalive nudger phase: 0 = not started, 1 = handshake (nudge both the
/// main pool and the helper pool), 2 = proving (nudge only the helper pool).
static KEEPALIVE_PHASE: AtomicUsize = AtomicUsize::new(0);

/// Start a detached low-duty-cycle nudger that keeps thread pools from deep
/// idle across the benchmark worker's ready→seed handshake and across the
/// timed proof's pool-idle phase gaps.
///
/// The ranked harness launches one fresh worker per trial: after the untimed
/// warm-up proof the worker parks every pool thread while it waits for the
/// seed on stdin, and the first ~25 ms of the timed proof (input regen +
/// witness trace) never touch the efficiency-core helper pool, so the OS is
/// free to sink both clusters into deep idle whose wake latency then lands
/// inside the scored interval. The nudger posts a no-op to each pool every
/// few milliseconds: during the handshake to both pools, and once
/// [`keepalive_proving`] flips the phase, only to the utility-QoS helper pool
/// (never injecting into the main pool mid-proof). No-op nudges do no memory
/// traffic and touch no prover state, so output stays byte-identical.
///
/// Kill switch: `FLOCK_NO_EPOOL_KEEPALIVE=1` disables the nudger entirely.
pub fn keepalive_start() {
    if std::env::var_os("FLOCK_NO_EPOOL_KEEPALIVE").is_some() {
        return;
    }
    if KEEPALIVE_PHASE.swap(1, Ordering::Relaxed) != 0 {
        return; // already started
    }
    let builder = std::thread::Builder::new().name("flock-keepalive".into());
    let _ = builder.spawn(|| {
        loop {
            let handshake = KEEPALIVE_PHASE.load(Ordering::Relaxed) == 1;
            if handshake {
                rayon::spawn(|| {});
            }
            match epool() {
                Some(ep) => ep.spawn_broadcast(|_| {}),
                // No helper pool on this host: once the handshake ends there
                // is nothing left to keep warm.
                None if !handshake => return,
                None => {}
            }
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
    });
}

/// Flip the keepalive nudger from handshake mode to proving mode: stop
/// injecting no-ops into the main rayon pool for the rest of the process.
/// Call immediately after the timed seed arrives. No-op when
/// [`keepalive_start`] never ran (or was disabled).
pub fn keepalive_proving() {
    if KEEPALIVE_PHASE.load(Ordering::Relaxed) != 0 {
        KEEPALIVE_PHASE.store(2, Ordering::Relaxed);
    }
}

/// `Send + Sync` wrapper for a raw base pointer shared across the two pools.
///
/// # Safety contract (caller's)
/// Every use must derive pairwise-disjoint ranges per chunk index. Mutable
/// ranges must have one owner, and immutable producer/consumer ranges must not
/// be published until their final write is synchronized with the reader. The
/// queue handing out each index exactly once establishes ownership; individual
/// call sites establish any required happens-before edge.
#[derive(Clone, Copy)]
pub(crate) struct SyncPtr<T>(pub(crate) *mut T);

impl<T> SyncPtr<T> {
    /// The wrapped base pointer. A method call uses the whole receiver, so
    /// closures capture the `Sync` wrapper rather than (via edition-2021+
    /// precise capture) the bare non-`Sync` pointer field.
    #[inline]
    pub(crate) fn ptr(self) -> *mut T {
        self.0
    }
}

// SAFETY: SyncPtr is only a capability to derive ranges under the ownership
// and synchronization contract above; each caller upholds that contract.
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every chunk index is executed exactly once when both pools drain the
    /// queue concurrently.
    #[test]
    fn helper_queue_runs_each_chunk_exactly_once() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let n = 1000;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        run_chunks_with_helper(
            n,
            &|i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            },
            Some(&helper),
        );
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// Below the engagement threshold the helper pool is skipped but every
    /// chunk still runs exactly once on the main pool.
    #[test]
    fn small_jobs_skip_helper_but_complete() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let n = EPOOL_MIN_CHUNKS - 1;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        run_chunks_with_helper(
            n,
            &|i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            },
            Some(&helper),
        );
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// A single-threaded main pool runs the queue inline (order 0..n) and
    /// never engages the helper.
    #[test]
    fn single_threaded_pool_runs_inline_in_order() {
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        single.install(|| {
            let seen = std::sync::Mutex::new(Vec::new());
            run_chunks_with_helper(100, &|i| seen.lock().unwrap().push(i), Some(&helper));
            let seen = seen.into_inner().unwrap();
            assert_eq!(seen, (0..100).collect::<Vec<_>>());
        });
    }

    /// Zero chunks is a no-op.
    #[test]
    fn zero_chunks_is_noop() {
        run_chunks_with_helper(0, &|_| panic!("must not run"), None);
    }

    /// Stateful workers initialize private scratch once and reuse it across
    /// many queue claims while still executing every chunk exactly once.
    #[test]
    fn stateful_helper_queue_reuses_worker_state() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let n = 1000;
        let next_state = AtomicUsize::new(0);
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let state_for_chunk: Vec<AtomicUsize> =
            (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect();
        run_chunks_with_helper_stateful(
            n,
            &|| next_state.fetch_add(1, Ordering::Relaxed),
            &|state, i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
                state_for_chunk[i].store(*state, Ordering::Relaxed);
            },
            Some(&helper),
        );
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        let n_states = next_state.load(Ordering::Relaxed);
        assert!(n_states < n, "state must be per worker, not per chunk");
        let mut uses = vec![0usize; n_states];
        for state in state_for_chunk {
            uses[state.load(Ordering::Relaxed)] += 1;
        }
        assert!(uses.into_iter().any(|n_uses| n_uses > 1));
    }

    /// The stateful single-thread path preserves strict chunk order and owns
    /// exactly one state, matching the stateless dispatcher's serial contract.
    #[test]
    fn stateful_single_threaded_pool_uses_one_state_in_order() {
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        single.install(|| {
            let init_count = AtomicUsize::new(0);
            let seen = std::sync::Mutex::new(Vec::new());
            run_chunks_with_helper_stateful(
                100,
                &|| {
                    init_count.fetch_add(1, Ordering::Relaxed);
                    0usize
                },
                &|state, i| {
                    *state += 1;
                    seen.lock().unwrap().push(i);
                },
                Some(&helper),
            );
            assert_eq!(init_count.load(Ordering::Relaxed), 1);
            assert_eq!(seen.into_inner().unwrap(), (0..100).collect::<Vec<_>>());
        });
    }
}

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

use std::sync::OnceLock;
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

/// [`run_hetero_chunks`] with per-worker scratch, mirroring rayon's
/// `map_init`: `init()` runs at most **once per draining worker** (lazily, on
/// that worker's first claimed chunk) and the resulting value is reused by
/// every later chunk the same worker claims.
///
/// For kernels needing a large per-chunk scratch buffer this is the difference
/// between one allocation per worker and one per chunk. The determinism
/// contract is unchanged: `f` must still write only to chunk `i`'s disjoint
/// range, and must not let scratch *contents* carry meaning across chunks —
/// which worker claims which chunk is nondeterministic, so scratch must be
/// fully (re)initialized by `f` before it is read.
pub(crate) fn run_hetero_chunks_init<T, I, F>(n_chunks: usize, init: I, f: F)
where
    I: Fn() -> T + Sync,
    F: Fn(&mut T, usize) + Sync,
    T: Send,
{
    drain_chunks(n_chunks, &init, &f, epool());
}

/// [`run_hetero_chunks`] with an explicit helper pool, so tests can exercise
/// the two-pool queue on hosts without efficiency cores.
pub(crate) fn run_chunks_with_helper<F>(n_chunks: usize, f: &F, helper: Option<&rayon::ThreadPool>)
where
    F: Fn(usize) + Sync,
{
    drain_chunks(n_chunks, &|| (), &|(), i| f(i), helper);
}

/// Shared queue-drain body behind [`run_chunks_with_helper`] and
/// [`run_hetero_chunks_init`]: one atomic cursor handed out to the main rayon
/// pool plus (when present and the job is large enough) the efficiency-core
/// helper pool, with one lazily built scratch value per draining worker.
///
/// The stateless entry point is this with `T = ()`, so there is exactly one
/// copy of the queue logic and both APIs share its guarantees.
fn drain_chunks<T, I, F>(n_chunks: usize, init: &I, f: &F, helper: Option<&rayon::ThreadPool>)
where
    I: Fn() -> T + Sync,
    F: Fn(&mut T, usize) + Sync,
    T: Send,
{
    if n_chunks == 0 {
        return;
    }
    let main_threads = rayon::current_num_threads();
    if main_threads <= 1 {
        // A deliberately single-threaded pool (RAYON_NUM_THREADS=1) stays
        // truly single-threaded: run inline, spawn nothing.
        let mut state = init();
        for i in 0..n_chunks {
            f(&mut state, i);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let worker = || {
        // Built on the first claimed chunk, so a drain task that loses every
        // race to the queue pays nothing.
        let mut state: Option<T> = None;
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            f(state.get_or_insert_with(init), i);
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
        Some(ep) => std::thread::scope(|s| {
            // The scoped thread parks inside `broadcast` while the E-workers
            // drain; it costs no main-pool worker. The scope join bounds the
            // tail wait at one chunk on one efficiency core.
            s.spawn(|| ep.broadcast(|_| worker()));
            drain_main();
        }),
        None => drain_main(),
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

    /// The `map_init`-shaped variant visits every index exactly once with both
    /// pools draining, and builds scratch **per worker, not per chunk**.
    #[test]
    fn init_queue_runs_each_chunk_once_with_per_worker_scratch() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let n = 1000;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let inits = AtomicUsize::new(0);
        drain_chunks(
            n,
            &|| {
                inits.fetch_add(1, Ordering::Relaxed);
                // Per-worker scratch: each chunk must (re)fill it itself.
                vec![0usize; 4]
            },
            &|scratch: &mut Vec<usize>, i| {
                scratch[0] = i;
                counts[scratch[0]].fetch_add(1, Ordering::Relaxed);
            },
            Some(&helper),
        );
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        let inits = inits.load(Ordering::Relaxed);
        assert!(inits >= 1);
        // One per draining worker at most: the main pool's tasks plus the
        // helper's threads — far below one per chunk, which is the whole point.
        assert!(
            inits <= rayon::current_num_threads() + helper.current_num_threads(),
            "init ran {inits} times, expected at most one per draining worker"
        );
    }

    /// Single-threaded main pool: the init variant also runs inline in order
    /// with exactly one scratch value.
    #[test]
    fn init_queue_single_threaded_runs_inline_in_order() {
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        single.install(|| {
            let inits = AtomicUsize::new(0);
            // Inline execution means index `i` is the `i`-th body call; an
            // atomic step counter checks that without a lock.
            let step = AtomicUsize::new(0);
            let in_order = AtomicUsize::new(1);
            drain_chunks(
                100,
                &|| {
                    inits.fetch_add(1, Ordering::Relaxed);
                },
                &|(), i| {
                    if step.fetch_add(1, Ordering::Relaxed) != i {
                        in_order.store(0, Ordering::Relaxed);
                    }
                },
                Some(&helper),
            );
            assert_eq!(step.load(Ordering::Relaxed), 100);
            assert_eq!(in_order.load(Ordering::Relaxed), 1, "ran out of order");
            assert_eq!(inits.load(Ordering::Relaxed), 1);
        });
    }

    /// Zero chunks never builds scratch and never runs the body.
    #[test]
    fn init_queue_zero_chunks_is_noop() {
        drain_chunks(
            0,
            &|| panic!("must not init"),
            &|(), _| panic!("must not run"),
            None,
        );
    }
}

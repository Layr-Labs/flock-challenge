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

/// Whether this host has a live efficiency-core helper pool.
///
/// The ranked worker calls this during its warm-up proof before deciding to
/// omit work from witness generation. A missing pool therefore preserves the
/// eager path instead of falling back to a contending performance-core thread.
pub fn helper_pool_available() -> bool {
    epool().is_some()
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
/// Chunks claimed by the efficiency-core helper pool across all hetero
/// drains (diagnostic only; relaxed ordering). Read deltas around a window
/// to prove E-core engagement for that window.
static EPOOL_HELPER_CHUNKS: AtomicUsize = AtomicUsize::new(0);

/// Total chunks claimed by the helper pool so far (monotonic).
pub fn helper_chunks_claimed() -> usize {
    EPOOL_HELPER_CHUNKS.load(Ordering::Relaxed)
}

pub fn run_hetero_chunks<F>(n_chunks: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    run_chunks_with_helper(n_chunks, &f, epool());
}

/// Try to process chunks exclusively on the efficiency-core helper pool.
///
/// Unlike [`run_hetero_chunks`], the calling thread and the main Rayon pool do
/// not claim work. The caller only waits for the helper broadcast to finish,
/// making this suitable for best-effort background work that must not consume
/// a performance core. Returns `false` without running `f` when the helper
/// pool is unavailable or `max_workers` is zero, so callers can fall back to a
/// sequential implementation.
pub fn run_helper_only_chunks<F>(n_chunks: usize, max_workers: usize, f: &F) -> bool
where
    F: Fn(usize) + Sync + ?Sized,
{
    run_chunks_with_helper_only(n_chunks, max_workers, f, epool())
}

/// [`run_helper_only_chunks`] with an explicit helper pool, so tests can
/// exercise the helper-only queue on hosts without efficiency cores.
#[doc(hidden)]
pub fn run_chunks_with_helper_only<F>(
    n_chunks: usize,
    max_workers: usize,
    f: &F,
    helper: Option<&rayon::ThreadPool>,
) -> bool
where
    F: Fn(usize) + Sync + ?Sized,
{
    let Some(ep) = helper else {
        return false;
    };
    let max_workers = max_workers.min(ep.current_num_threads());
    if max_workers == 0 {
        return false;
    }
    if n_chunks == 0 {
        return true;
    }

    let next = AtomicUsize::new(0);
    ep.broadcast(|context| {
        if context.index() >= max_workers {
            return;
        }
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            EPOOL_HELPER_CHUNKS.fetch_add(1, Ordering::Relaxed);
            f(i);
        }
    });
    true
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
pub fn run_chunks_with_helper<F>(n_chunks: usize, f: &F, helper: Option<&rayon::ThreadPool>)
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
        Some(ep) => std::thread::scope(|s| {
            // The scoped thread parks inside `broadcast` while the E-workers
            // drain; it costs no main-pool worker. The scope join bounds the
            // tail wait at one chunk on one efficiency core.
            s.spawn(|| {
                ep.broadcast(|_| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n_chunks {
                        break;
                    }
                    EPOOL_HELPER_CHUNKS.fetch_add(1, Ordering::Relaxed);
                    f(i);
                })
            });
            drain_main();
        }),
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
        Some(ep) => std::thread::scope(|s| {
            s.spawn(|| {
                ep.broadcast(|_| {
                    let mut state = init();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n_chunks {
                            break;
                        }
                        EPOOL_HELPER_CHUNKS.fetch_add(1, Ordering::Relaxed);
                        f(&mut state, i);
                    }
                })
            });
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
pub struct SyncPtr<T>(pub *mut T);

impl<T> SyncPtr<T> {
    /// The wrapped base pointer. A method call uses the whole receiver, so
    /// closures capture the `Sync` wrapper rather than (via edition-2021+
    /// precise capture) the bare non-`Sync` pointer field.
    #[inline]
    pub fn ptr(self) -> *mut T {
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

    /// The helper-only drain never recruits the caller/main pool and still
    /// gives every chunk exactly one owner. Limiting the broadcast to two of
    /// four helper workers also covers the production tuning seam.
    #[test]
    fn helper_only_queue_runs_each_chunk_exactly_once() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("helper-only-test-{i}"))
            .build()
            .unwrap();
        let n = 513;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let worker_mask = AtomicUsize::new(0);
        assert!(run_chunks_with_helper_only(
            n,
            2,
            &|i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
                let worker = rayon::current_thread_index().expect("helper worker");
                worker_mask.fetch_or(1usize << worker, Ordering::Relaxed);
            },
            Some(&helper),
        ));
        assert!(counts
            .iter()
            .all(|count| count.load(Ordering::Relaxed) == 1));
        assert_eq!(worker_mask.load(Ordering::Relaxed) & !0b11, 0);

        let untouched = AtomicUsize::new(0);
        assert!(!run_chunks_with_helper_only(
            1,
            2,
            &|_| {
                untouched.fetch_add(1, Ordering::Relaxed);
            },
            None,
        ));
        assert_eq!(untouched.load(Ordering::Relaxed), 0);
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

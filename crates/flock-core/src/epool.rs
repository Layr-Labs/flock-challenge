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

/// Only a queue at least this large may absorb registered extra work — it
/// must be one of the big ranked passes, not a small helper job.
const EXTRA_WORK_MIN_HOST_CHUNKS: usize = 256;

/// Scoped side-channel that lets one caller donate an extra claim set to the
/// next large heterogeneous queue, so both are drained as ONE union by both
/// pools (single broadcast, no serialization between coordinators).
///
/// Motivation, measured the expensive way: draining the layer-1 NTT tiles and
/// the round-1 AB precompute through two *separate* queues in the same
/// commit/AB window is officially anti-additive — each E worker must finish
/// the first broadcast's whole queue before starting the second, and the
/// second scope blocks joining its broadcast, chaining the AB branch behind
/// the NTT drain. The union makes the window one claim space.
struct ExtraWork {
    n: usize,
    /// Type-erased borrowed closure. SAFETY: only dereferenced while the
    /// registering [`with_extra_work`] frame is live — the slot is cleared
    /// before that frame returns, and any queue that takes the slot completes
    /// every extra claim before returning, which happens inside that frame's
    /// dynamic extent.
    f: *const (dyn Fn(usize) + Sync),
}
unsafe impl Send for ExtraWork {}

static EXTRA_WORK: std::sync::Mutex<Option<ExtraWork>> = std::sync::Mutex::new(None);

/// Register `f_extra(0..n_extra)` as donor work for the duration of `body`.
/// Returns `(body_result, consumed)`; when `consumed` is false no queue
/// absorbed the work and the caller must run it itself.
pub(crate) fn with_extra_work<R>(
    n_extra: usize,
    f_extra: &(dyn Fn(usize) + Sync),
    body: impl FnOnce() -> R,
) -> (R, bool) {
    if n_extra == 0 {
        return (body(), true);
    }
    // SAFETY: see `ExtraWork.f` — the transmute only widens the lifetime for
    // storage; every dereference happens within `body`'s dynamic extent and
    // the slot is cleared below before this frame returns.
    let erased: *const (dyn Fn(usize) + Sync) = unsafe {
        core::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f_extra)
    };
    *EXTRA_WORK.lock().unwrap() = Some(ExtraWork { n: n_extra, f: erased });
    let out = body();
    let leftover = EXTRA_WORK.lock().unwrap().take();
    (out, leftover.is_none())
}

/// Claim the registered extra work if this queue is large enough to host it.
fn take_extra_work(n_chunks: usize) -> Option<ExtraWork> {
    if n_chunks < EXTRA_WORK_MIN_HOST_CHUNKS {
        return None;
    }
    EXTRA_WORK.lock().unwrap().take()
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
    match take_extra_work(n_chunks) {
        Some(extra) => {
            // SAFETY: within the registering frame's extent (see ExtraWork).
            let ef = unsafe { &*extra.f };
            let total = n_chunks + extra.n;
            // Proportional (Bresenham) interleave of host and donor claims.
            // The two workloads use complementary resources (the ranked host
            // pass streams DRAM; the donor AB claims are L1-lookup-bound), so
            // mixing them preserves the concurrency the old `rayon::join`
            // provided — strictly phased ranges would first saturate
            // bandwidth with cores stalling, then saturate load ports with
            // bandwidth idle.
            let union = |i: usize| {
                let e_before = i * extra.n / total;
                if (i + 1) * extra.n / total > e_before {
                    ef(e_before);
                } else {
                    f(i - e_before);
                }
            };
            run_chunks_with_helper(total, &union, epool());
        }
        None => run_chunks_with_helper(n_chunks, &f, epool()),
    }
}

/// Cooperative variant of [`run_hetero_chunks`] for jobs that run **inside an
/// overlapped region** (e.g. one branch of the commit/AB `rayon::join`).
///
/// The plain queue's main-pool side spawns one greedy drain-loop per worker;
/// a worker that picks one up holds it until the queue is empty. Standalone
/// that is ideal, but under a `join` it starves the sibling branch and can
/// serialize the overlap. Here the main-pool side instead submits **one rayon
/// task per claim** (`with_max_len(1)`), so the scheduler interleaves these
/// claims with the sibling branch's tasks exactly as it interleaves the
/// current `par_chunks` — while the efficiency-core broadcast drains the same
/// atomic queue greedily from the side. A main-pool task whose claim finds
/// the queue already exhausted is a no-op.
///
/// Same output contract as [`run_hetero_chunks`]: each chunk index runs
/// exactly once; `f(i)` must write only chunk `i`'s disjoint range. With a
/// single-threaded main pool the queue runs inline in order and spawns
/// nothing.
pub(crate) fn run_hetero_chunks_cooperative<F>(n_chunks: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if n_chunks == 0 {
        return;
    }
    if rayon::current_num_threads() <= 1 {
        for i in 0..n_chunks {
            f(i);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let drain_main = || {
        (0..n_chunks).into_par_iter().with_max_len(1).for_each(|_| {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i < n_chunks {
                f(i);
            }
        });
    };
    match epool().filter(|_| n_chunks >= EPOOL_MIN_CHUNKS) {
        Some(ep) => std::thread::scope(|s| {
            s.spawn(|| {
                ep.broadcast(|_| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n_chunks {
                            break;
                        }
                        f(i);
                    }
                })
            });
            drain_main();
        }),
        None => drain_main(),
    }
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

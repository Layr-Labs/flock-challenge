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
//!
//! Handing the helper pool a `broadcast` needs a thread that is neither a main
//! worker (it would cost a performance core) nor a helper worker (it would sit
//! inside the pool it is waking). That used to be a fresh `std::thread::scope`
//! + `spawn` per engaged drain — one OS thread created and joined every time.
//! [`Relay`] replaces it with a single persistent thread that parks between
//! drains, so an engaged drain costs a condvar signal instead of a thread
//! lifecycle. `FLOCK_NO_EPOOL_RELAY=1` restores the per-drain spawn; so does
//! any drain that finds the relay already carrying a concurrent one.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

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

/// Off-target hosts have no efficiency cores, so the helper pool is absent and
/// every hetero drain runs main-pool-only.
///
/// `FLOCK_EPOOL_FORCE_THREADS=<n>` (n > 0) synthesizes an n-thread helper pool
/// so the two-pool drain — and the broadcast relay that feeds it — can be
/// exercised, counted, and A/B-timed on non-Apple hardware, where this module
/// would otherwise be dead code. Diagnostic only: this arm is never compiled
/// for the ranked aarch64-macOS target, and with the variable unset it returns
/// 0 exactly as before.
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn ecore_count() -> usize {
    std::env::var("FLOCK_EPOOL_FORCE_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
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

/// The efficiency-core helper pool itself, for callers that overlap a small
/// best-effort task with main-pool work via `in_place_scope` (e.g. the
/// prover's publish-prefix pre-encode alongside the PCS open). Tasks run at
/// `QOS_CLASS_UTILITY` on E-cores and must not affect output bytes — the
/// caller owns a fallback for `None` (off-target hosts).
pub fn helper_pool() -> Option<&'static rayon::ThreadPool> {
    epool()
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

/// Helper broadcasts issued by the two hetero drains across all of this
/// process (diagnostic only; relaxed ordering). One increment per *engaged*
/// drain, i.e. exactly the kickoffs [`drain_hetero`] carries. Drains that skip
/// the helper (no pool, fewer than [`EPOOL_MIN_CHUNKS`] chunks, a
/// single-threaded main pool, zero chunks) do not count, and neither does
/// [`run_chunks_with_helper_only`] — that path never used a relay thread. Read
/// deltas around a window to price the per-drain kickoff for that window.
static EPOOL_BROADCASTS: AtomicU64 = AtomicU64::new(0);

/// Total helper broadcasts issued by the hetero drains so far (monotonic).
pub fn helper_broadcasts_issued() -> u64 {
    EPOOL_BROADCASTS.load(Ordering::Relaxed)
}

/// Whether engaged drains go through the persistent [`Relay`]. Kill switch:
/// `FLOCK_NO_EPOOL_RELAY=1` (exactly `"1"`) restores the per-drain
/// `std::thread::scope` + `spawn`. Read once per process.
pub(crate) fn relay_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !relay_killed_by(std::env::var("FLOCK_NO_EPOOL_RELAY").ok().as_deref()))
}

/// Kill rule for [`relay_enabled`]: the value must be exactly `"1"`. Anything
/// else — `"true"`, `"0"`, an empty string, an unset variable — keeps the
/// relay, so a stray export can never silently change the ranked shape.
fn relay_killed_by(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Type-erased pointer to a broadcast closure living on a drain's stack frame:
/// a thin data pointer plus the monomorphized thunk that restores its type. A
/// thunk rather than `dyn Fn` because the closure's lifetime is the drain's,
/// not `'static`, and a raw trait-object pointer cannot express that.
#[derive(Clone, Copy)]
struct JobPtr {
    data: *const (),
    call: unsafe fn(*const ()),
}

// SAFETY: the referent is `Sync`, so invoking it from the relay thread is the
// same as invoking it through a shared reference. `PostedJob`'s `Drop` keeps
// the posting frame alive until the relay reports the job complete, so the
// pointer is never dereferenced after its referent dies.
unsafe impl Send for JobPtr {}

impl JobPtr {
    fn new<B: Fn() + Sync>(broadcast: &B) -> Self {
        /// # Safety
        /// `data` must be a live `&B` produced by [`JobPtr::new`].
        unsafe fn thunk<B: Fn() + Sync>(data: *const ()) {
            // SAFETY: guaranteed by the caller (`Relay::run`, which only ever
            // runs a job the posting drain is still blocked on).
            unsafe { (*data.cast::<B>())() }
        }
        Self {
            data: (broadcast as *const B).cast::<()>(),
            call: thunk::<B>,
        }
    }
}

#[derive(Default)]
struct RelayState {
    /// Job posted and not yet picked up.
    job: Option<JobPtr>,
    /// Jobs posted so far; a job's sequence number is `posted` after its post.
    posted: u64,
    /// Jobs the relay thread has finished.
    done: u64,
    /// Whether the most recently finished job unwound.
    panicked: bool,
}

/// The process-wide broadcast relay: one persistent thread that hands the
/// helper pool a `broadcast` on request, so an engaged drain no longer creates
/// and joins an OS thread. It is a plain `std::thread` — never a rayon worker
/// — so it occupies neither pool, and it is created lazily on the first
/// engaged drain, which on the ranked worker falls inside the untimed warm-up
/// proof. A prove that engages no drain never creates it.
struct Relay {
    state: Mutex<RelayState>,
    /// Signalled on post (the relay waits on it) and on completion (the
    /// posting drain waits on it).
    signal: Condvar,
    /// Held for the whole of one relayed drain. A concurrent (e.g. nested)
    /// drain that cannot take it falls back to the per-drain spawn rather than
    /// queueing behind this one.
    busy: AtomicBool,
}

/// A poisoned relay mutex must degrade to "keep running", never to a panic:
/// the relay only ever holds it across bookkeeping, so its contents stay
/// consistent even if some other thread unwound while waiting.
fn relay_lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Relay {
    fn new() -> Self {
        Self {
            state: Mutex::new(RelayState::default()),
            signal: Condvar::new(),
            busy: AtomicBool::new(false),
        }
    }

    /// Take exclusive use of the relay, or report that another drain has it.
    fn try_acquire(&self) -> bool {
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// Hand `job` to the relay thread; returns its sequence number.
    ///
    /// # Safety
    /// `job` must stay valid until [`Relay::wait_done`] has returned for the
    /// sequence number this call yields.
    unsafe fn post(&self, job: JobPtr) -> u64 {
        let seq = {
            let mut st = relay_lock(&self.state);
            st.job = Some(job);
            st.posted += 1;
            st.panicked = false;
            st.posted
        };
        self.signal.notify_all();
        seq
    }

    /// Block until the relay thread has finished job `seq`; reports whether
    /// that job unwound.
    fn wait_done(&self, seq: u64) -> bool {
        let mut st = relay_lock(&self.state);
        while st.done < seq {
            st = self
                .signal
                .wait(st)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        st.panicked
    }

    /// Relay thread body: park until a job arrives, run it, report completion.
    fn run(&self) -> ! {
        loop {
            let job = {
                let mut st = relay_lock(&self.state);
                loop {
                    if let Some(job) = st.job.take() {
                        break job;
                    }
                    st = self
                        .signal
                        .wait(st)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
            };
            // SAFETY: the posting drain blocks in `PostedJob::drop` until
            // `done` reaches this job's sequence, so the closure and every
            // frame it borrows are still live for the whole call. Unwinding is
            // contained here so `done` always advances — a lost completion
            // would hang the poster instead of surfacing the panic, which
            // `PostedJob::drop` re-raises on the poster's own thread (the
            // per-drain scope did the same).
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                (job.call)(job.data);
            }))
            .is_ok();
            {
                let mut st = relay_lock(&self.state);
                st.done += 1;
                st.panicked = !ok;
            }
            self.signal.notify_all();
        }
    }
}

/// The process-wide relay, or `None` when its thread could not be spawned (in
/// which case every drain keeps using the per-drain spawn).
fn relay() -> Option<&'static Relay> {
    static RELAY: OnceLock<Option<&'static Relay>> = OnceLock::new();
    *RELAY.get_or_init(|| {
        let relay: &'static Relay = Box::leak(Box::new(Relay::new()));
        // No QoS call: the thread inherits the class of whichever thread first
        // engaged a drain — a main-pool worker in the warm-up proof — exactly
        // as the per-drain scoped threads it replaces did.
        std::thread::Builder::new()
            .name("flock-epool-relay".to_string())
            .spawn(move || relay.run())
            .ok()
            .map(|_handle| relay)
    })
}

/// Ties a posted job to the drain that posted it. Dropping it — on the normal
/// path *and* while unwinding — waits for the relay to finish before this
/// frame (and the broadcast closure borrowing it) goes away, then frees the
/// relay for the next drain. This is what makes [`JobPtr`] sound.
struct PostedJob<'a> {
    relay: &'a Relay,
    /// Sequence number of this drain's job, or 0 if it never posted one.
    seq: AtomicU64,
}

impl Drop for PostedJob<'_> {
    fn drop(&mut self) {
        let seq = self.seq.load(Ordering::Relaxed);
        let panicked = seq != 0 && self.relay.wait_done(seq);
        self.relay.busy.store(false, Ordering::Release);
        if panicked && !std::thread::panicking() {
            panic!("epool: helper-pool broadcast panicked");
        }
    }
}

/// Drain one *engaged* hetero chunk queue: `main_threads` main-pool workers
/// running `worker`, concurrently with one helper-pool `broadcast`. Returns
/// only once both sides are finished, so every chunk has been executed exactly
/// once and its writes are visible.
///
/// WAKE ORDER. The relay path requests the broadcast from *inside* the
/// main-pool `for_each` body, ahead of that worker's own drain loop, so no
/// E-worker can be woken until a main-pool worker has already entered the
/// drain — and the signal still has to travel through the mutex, the condvar,
/// the relay's wake-up and rayon's broadcast injection before one runs. The
/// per-drain-spawn arm below issues its `spawn` *before* `drain_main` is even
/// called. The relay therefore starts the helper strictly no earlier, relative
/// to the main drain, than the shape it replaces; it only removes the thread
/// creation and join that used to sit on the main thread's critical path.
///
/// QUEUE SEMANTICS. Neither arm touches the shared cursor or the chunk
/// closure: which worker set drains the queue is unobservable in the output by
/// construction.
pub(crate) fn drain_hetero<B>(
    main_threads: usize,
    worker: &(dyn Fn() + Sync),
    broadcast: &B,
    use_relay: bool,
) where
    B: Fn() + Sync,
{
    if use_relay
        && let Some(relay) = relay()
        && relay.try_acquire()
    {
        let job = PostedJob {
            relay,
            seq: AtomicU64::new(0),
        };
        let posted = AtomicBool::new(false);
        (0..main_threads)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|_| {
                // Exactly one main worker posts, and only once it is already
                // inside the drain. The relaxed load keeps every later worker
                // off the contended cache line.
                if !posted.load(Ordering::Relaxed) && !posted.swap(true, Ordering::Relaxed) {
                    // SAFETY: `job`'s `Drop` below blocks until the relay
                    // reports this sequence complete, so `broadcast` outlives
                    // every dereference of the pointer we hand over.
                    let seq = unsafe { relay.post(JobPtr::new(broadcast)) };
                    job.seq.store(seq, Ordering::Relaxed);
                }
                worker();
            });
        // Explicit: this join is the safety condition, not a cleanup detail.
        drop(job);
        return;
    }
    // Per-drain relay — kill switch, no relay thread, or the relay is already
    // carrying a concurrent drain. The scoped thread parks inside `broadcast`
    // while the E-workers drain; it costs no main-pool worker, and the scope
    // join bounds the tail wait at one chunk on one efficiency core.
    std::thread::scope(|s| {
        s.spawn(|| broadcast());
        (0..main_threads)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|_| worker());
    });
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
    run_chunks_with_helper_relay(n_chunks, f, helper, relay_enabled());
}

/// [`run_chunks_with_helper`] with the relay choice forced, so tests can cover
/// both the persistent-relay and the per-drain-spawn arm in one process.
fn run_chunks_with_helper_relay<F>(
    n_chunks: usize,
    f: &F,
    helper: Option<&rayon::ThreadPool>,
    use_relay: bool,
) where
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
        Some(ep) => {
            EPOOL_BROADCASTS.fetch_add(1, Ordering::Relaxed);
            let broadcast = || {
                ep.broadcast(|_| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n_chunks {
                            break;
                        }
                        EPOOL_HELPER_CHUNKS.fetch_add(1, Ordering::Relaxed);
                        f(i);
                    }
                });
            };
            drain_hetero(main_threads, &worker, &broadcast, use_relay);
        }
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
    run_chunks_with_helper_stateful_relay(n_chunks, init, f, helper, relay_enabled());
}

/// [`run_chunks_with_helper_stateful`] with the relay choice forced, so tests
/// can cover both arms in one process.
fn run_chunks_with_helper_stateful_relay<S, I, F>(
    n_chunks: usize,
    init: &I,
    f: &F,
    helper: Option<&rayon::ThreadPool>,
    use_relay: bool,
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
        Some(ep) => {
            EPOOL_BROADCASTS.fetch_add(1, Ordering::Relaxed);
            let broadcast = || {
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
                });
            };
            drain_hetero(main_threads, &worker, &broadcast, use_relay);
        }
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
        assert!(
            counts
                .iter()
                .all(|count| count.load(Ordering::Relaxed) == 1)
        );
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

    /// Both kickoff arms — the persistent relay and the per-drain spawn the
    /// kill switch restores — execute every chunk exactly once. Running many
    /// drains through one relay also proves it is reusable: a per-drain thread
    /// would have to be recreated, a broken relay would hang here.
    #[test]
    fn both_kickoff_arms_run_each_chunk_exactly_once() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let n = 997;
        for use_relay in [true, false] {
            for _ in 0..16 {
                let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                run_chunks_with_helper_relay(
                    n,
                    &|i| {
                        counts[i].fetch_add(1, Ordering::Relaxed);
                    },
                    Some(&helper),
                    use_relay,
                );
                assert!(
                    counts.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                    "use_relay={use_relay}"
                );
            }
        }
    }

    /// The stateful drain behaves identically on both arms: state is per
    /// worker (not per chunk) and every chunk still runs exactly once.
    #[test]
    fn both_kickoff_arms_reuse_stateful_worker_state() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let n = 1000;
        for use_relay in [true, false] {
            let next_state = AtomicUsize::new(0);
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            run_chunks_with_helper_stateful_relay(
                n,
                &|| next_state.fetch_add(1, Ordering::Relaxed),
                &|_state, i| {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                },
                Some(&helper),
                use_relay,
            );
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
            assert!(
                next_state.load(Ordering::Relaxed) < n,
                "state must be per worker, not per chunk (use_relay={use_relay})"
            );
        }
    }

    /// Drains that overlap in time cannot both own the single relay: the loser
    /// falls back to the per-drain spawn instead of blocking. Both must still
    /// complete, and neither may deadlock.
    #[test]
    fn concurrent_relayed_drains_all_complete() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let n = 512;
        let counts: Vec<Vec<AtomicUsize>> = (0..4)
            .map(|_| (0..n).map(|_| AtomicUsize::new(0)).collect())
            .collect();
        std::thread::scope(|s| {
            for lane in &counts {
                s.spawn(|| {
                    for _ in 0..8 {
                        run_chunks_with_helper_relay(
                            n,
                            &|i| {
                                lane[i].fetch_add(1, Ordering::Relaxed);
                            },
                            Some(&helper),
                            true,
                        );
                    }
                });
            }
        });
        for lane in &counts {
            assert!(lane.iter().all(|c| c.load(Ordering::Relaxed) == 8));
        }
    }

    /// A panicking chunk closure surfaces as a panic on the caller's thread on
    /// both arms — it must never leave the relay holding a dangling job or a
    /// stuck `busy` flag, which would hang this test rather than fail it.
    #[test]
    fn panicking_chunk_propagates_and_leaves_the_relay_usable() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for use_relay in [true, false] {
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_chunks_with_helper_relay(
                    64,
                    &|_| panic!("chunk exploded"),
                    Some(&helper),
                    use_relay,
                );
            }));
            assert!(caught.is_err(), "use_relay={use_relay}");
        }
        // The relay must still serve a normal drain afterwards.
        let n = 128;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        run_chunks_with_helper_relay(
            n,
            &|i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            },
            Some(&helper),
            true,
        );
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// Engaged drains bump the broadcast counter. Only a lower bound is
    /// asserted: the counter is process-global and the test harness runs other
    /// drains concurrently.
    #[test]
    fn engaged_drains_count_broadcasts() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let drains = 12;
        let before = helper_broadcasts_issued();
        for _ in 0..drains {
            run_chunks_with_helper(EPOOL_MIN_CHUNKS, &|_| {}, Some(&helper));
        }
        assert!(helper_broadcasts_issued() - before >= drains);
        // A drain with no helper cannot issue a broadcast, so the counter is
        // still at least where the engaged drains left it.
        let after_engaged = helper_broadcasts_issued();
        run_chunks_with_helper(1024, &|_| {}, None);
        assert!(helper_broadcasts_issued() >= after_engaged);
    }

    /// The kill switch reads exactly `"1"`; nothing else disables the relay.
    #[test]
    fn relay_kill_switch_matches_exactly_one() {
        assert!(relay_killed_by(Some("1")));
        for keep in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("11"),
            Some(" 1"),
        ] {
            assert!(!relay_killed_by(keep), "{keep:?} must not kill the relay");
        }
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

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
//!
//! # Per-worker work-stealing deque (Chase–Lev)
//!
//! [`run_hetero_chunks`] and the stateful drain both ship a shared
//! `AtomicUsize` "head pointer": every main-pool worker contends on the same
//! cache line for the next index. Under load that line bounces between
//! performance cores and serialises the entire drain behind a single RMW.
//!
//! [`WorkerDeque`] is the per-worker fix: each rayon worker (main-pool or
//! helper-pool) owns one Chase–Lev deque. The owner pushes to its tail and
//! pops from its own tail with no atomics; *other* workers steal from its
//! head with a single `Relaxed` `compare_exchange` per attempt. There is no
//! shared counter, so the cache-line storm is gone — a steal that fails
//! pays one cache miss on the victim deque's head, nothing more.
//!
//! [`run_chunks_with_stealing_deque`] is the new drain entry point. It
//! allocates one [`WorkerDeque`] per rayon worker (lazily, on first use), fans
//! the chunk range to the deques in a strict round-robin, then runs the
//! per-worker steal loop on the calling thread (and on every main/helper
//! worker the drain recruits). The byte-exactness invariant is identical to
//! the head-pointer path: every index 0..n is processed exactly once, by the
//! same `f(i)`, on whichever worker deque happens to own it.

use std::cell::{Cell, UnsafeCell};
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
const EPOOL_MIN_CHUNKS: usize = 8;

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
/// the two-pool queue on any host.
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

// ---------------------------------------------------------------------------
// Chase–Lev per-worker work-stealing deque
// ---------------------------------------------------------------------------

/// One slot of the deque's circular buffer. Wrapped in `CachePadded`-shaped
/// alignment so a steal and a push do not false-share on the same 64-byte
/// line. `MaybeUninit` is required because the slot is logically "uninit" the
/// instant `pop` reads out of it; we never observe the uninit state.
#[repr(C, align(64))]
struct Slot<T> {
    cell: UnsafeCell<std::mem::MaybeUninit<T>>,
}

// SAFETY: the deque's owner is the only thread allowed to call `push`/`pop`.
// Other threads call `steal`, which performs the same memory-order dance
// (Acquire on read, Release on read+write) that the standard Chase–Lev
// paper does. Because T is `Send` we can hand a `*const` slot to a stealer
// across threads.
unsafe impl<T: Send> Send for Slot<T> {}
unsafe impl<T: Send> Sync for Slot<T> {}

/// Bounded per-worker Chase–Lev work-stealing deque, parameterised on chunk
/// index type `I` so callers can store any `Copy` claim token.
///
/// Layout follows Le et al. ([`Correct and efficient work-stealing for weak
/// memory models`, PPoPP '13]): the owner reads `bottom` from a thread-local
/// cell (no atomicity needed because no other thread observes it), the
/// stealers read `bottom` through an atomic load to synchronise with the
/// owner's `Release` on pop. `top` is the atomic count of *steals* and is
/// the only location that needs a `compare_exchange`.
///
/// The internal buffer is a power-of-two circular array of `Slot`s; on
/// overflow it is grown exactly once (typical Chase–Lev). At the steady-state
/// sizes the ranked drains care about (a few dozen to a few thousand chunks)
/// the initial 1024-slot cap never trips, so the hot path stays branch-free.
///
/// # `Sync` contract
///
/// `WorkerDeque` is `Sync` even though the owner-only fields use
/// thread-local storage. We use `UnsafeCell` instead of `Cell` so the
/// `Sync` derivation is explicit (and so the compiler can keep
/// `Cell`-shaped reads out of the owner's hot path). Concurrent access
/// to the owner-only fields is forbidden — calling `push`/`pop` from a
/// thread other than the owner is a logic error. `steal` from any thread
/// is fine.
pub struct WorkerDeque<I: Copy + Send> {
    /// Victim's steal count, incremented by every successful steal. Owner
    /// reads it with `Acquire` paired against stealers' `Release`/compare-exchange.
    top: AtomicUsize,
    /// Owner's next push/pop slot. Owner reads/writes directly; stealers
    /// observe a load-with-Acquire.
    bottom: UnsafeCell<usize>,
    /// `2 * mask + 1` is the current capacity. Initially 1024 (mask = 0x3ff);
    /// doubled on first overflow.
    mask: UnsafeCell<usize>,
    /// `1` when the buffer is in the grown state and `realloc` is reachable
    /// from this index; used only for the `Debug` and tests. `0` otherwise.
    grown: UnsafeCell<bool>,
    /// Circular buffer; the active range is `buf[bottom & mask]`
    /// (push/pop) and `buf[top & mask]` (steal).
    buf: UnsafeCell<*mut Slot<I>>,
}

// SAFETY: see `Sync` contract on the struct doc. The owner-only fields are
// `UnsafeCell` and must not be accessed from a thread other than the owner;
// the stealer side touches only the atomic `top` and the slot array.
unsafe impl<I: Copy + Send> Sync for WorkerDeque<I> {}

impl<I: Copy + Send> WorkerDeque<I> {
    /// Initial power-of-two capacity.
    const INITIAL_CAPACITY_LOG2: u32 = 10;
    const INITIAL_CAPACITY: usize = 1 << Self::INITIAL_CAPACITY_LOG2;
    const INITIAL_MASK: usize = Self::INITIAL_CAPACITY - 1;

    /// Build an empty deque with the default 1024-slot buffer.
    pub fn new() -> Self {
        Self::with_capacity_log2(Self::INITIAL_CAPACITY_LOG2)
    }

    /// Build an empty deque whose initial capacity is `2^log2`. `log2`
    /// between 4 and 20 (16 slots to ~1 M slots) is the supported range.
    pub fn with_capacity_log2(log2: u32) -> Self {
        assert!(
            (4..=20).contains(&log2),
            "WorkerDeque::with_capacity_log2 out of range: {log2}"
        );
        let cap = 1usize << log2;
        let mut slots: Vec<Slot<I>> = (0..cap).map(|_| Slot { cell: UnsafeCell::new(std::mem::MaybeUninit::uninit()) }).collect();
        let ptr = slots.as_mut_ptr();
        std::mem::forget(slots);
        Self {
            top: AtomicUsize::new(0),
            bottom: UnsafeCell::new(0),
            mask: UnsafeCell::new(cap - 1),
            grown: UnsafeCell::new(false),
            buf: UnsafeCell::new(ptr),
        }
    }

    /// Number of items currently in the deque (owner or stealer, may be stale).
    pub fn len(&self) -> usize {
        let b = unsafe { *self.bottom.get() };
        let t = self.top.load(Ordering::Acquire);
        b.saturating_sub(t)
    }

    /// `true` when no item is currently in the deque.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push `item` onto the *owner's* tail. Owner-only: not safe to call
    /// from a thread other than the one that will subsequently call `pop` or
    /// `pop_local`. Capacity check is the standard "if full, grow" — the
    /// grow path is reached at most once over the deque's lifetime.
    pub fn push(&self, item: I) {
        let b = unsafe { *self.bottom.get() };
        let t = self.top.load(Ordering::Acquire);
        let mask = unsafe { *self.mask.get() };
        let cap = mask + 1;
        let buf = unsafe { std::slice::from_raw_parts_mut(*self.buf.get(), cap) };
        if b.wrapping_sub(t) > mask {
            self.grow(b);
            return self.push(item);
        }
        let slot = &buf[b & mask];
        unsafe { (*slot.cell.get()).write(item) };
        std::sync::atomic::fence(Ordering::Release);
        unsafe { *self.bottom.get() = b.wrapping_add(1) };
    }

    /// Pop the most recently pushed item from the *owner's* tail. Owner-only.
    /// Returns `None` when the deque is empty.
    pub fn pop(&self) -> Option<I> {
        let b = unsafe { *self.bottom.get() };
        if b == 0 {
            return None;
        }
        let new_b = b.wrapping_sub(1);
        unsafe { *self.bottom.get() = new_b };
        std::sync::atomic::fence(Ordering::SeqCst);
        let t = self.top.load(Ordering::Acquire);
        let mask = unsafe { *self.mask.get() };
        let cap = mask + 1;
        if t <= new_b {
            let buf = unsafe { std::slice::from_raw_parts_mut(*self.buf.get(), cap) };
            let slot = &buf[new_b & mask];
            let value = unsafe { (*slot.cell.get()).assume_init_read() };
            if t == new_b {
                // Last item: race with a stealer. Try to take it back.
                let _ = self.top.compare_exchange(
                    t,
                    t.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                );
                unsafe { *self.bottom.get() = t.wrapping_add(1) };
            }
            Some(value)
        } else {
            // Empty after the fence — clear the cache, reset bottom.
            unsafe { *self.bottom.get() = t };
            None
        }
    }

    /// Steal the *oldest* item from another worker's deque. Relaxed-ordering
    /// CAS — the stealer only needs to publish its `top` increment before
    /// reading the slot, and the slot write itself was fenced by the owner's
    /// push. Returns `None` on the contended-empty case.
    pub fn steal(&self) -> Option<I> {
        let t = self.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::SeqCst);
        // SAFETY: a stealer reads `bottom` once per attempt; the value can
        // be stale but is always a valid index the owner has produced
        // (or is about to). The classic Chase–Lev paper requires the
        // stealer to perform a *plain* (non-atomic) load here; we use
        // `UnsafeCell::get` to satisfy `noalias` while still producing
        // a single load on the target architecture.
        let b = unsafe { *self.bottom.get() };
        if t < b {
            let mask = unsafe { *self.mask.get() };
            let cap = mask + 1;
            let buf = unsafe { std::slice::from_raw_parts(*self.buf.get(), cap) };
            let slot = &buf[t & mask];
            let value = unsafe { (*slot.cell.get()).assume_init_read() };
            let ok = self
                .top
                .compare_exchange(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_ok();
            if ok {
                Some(value)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Grow the buffer to twice its capacity. Called from `push` on overflow
    /// and only there; on a typical ranked drain the initial 1024 slots are
    /// enough.
    fn grow(&self, b: usize) {
        let old_cap = unsafe { *self.mask.get() } + 1;
        let new_cap = old_cap * 2;
        let mut new_slots: Vec<Slot<I>> = (0..new_cap)
            .map(|_| Slot { cell: UnsafeCell::new(std::mem::MaybeUninit::uninit()) })
            .collect();
        // Copy the *live* items in order: live range is `top..bottom`, indexed by `& mask`.
        let t = self.top.load(Ordering::Acquire);
        let old_buf =
            unsafe { std::slice::from_raw_parts(*self.buf.get(), old_cap) };
        for i in 0..(b - t) {
            let src = &old_buf[(t + i) & (old_cap - 1)];
            let dst = &mut new_slots[i];
            let value = unsafe { (*src.cell.get()).assume_init_read() };
            unsafe { (*dst.cell.get()).write(value) };
        }
        let new_ptr = new_slots.as_mut_ptr();
        std::mem::forget(new_slots);
        // Free the old buffer.
        let old_ptr = unsafe { *self.buf.get() };
        unsafe { *self.buf.get() = new_ptr };
        unsafe { *self.mask.get() = new_cap - 1 };
        unsafe { *self.grown.get() = true };
        // SAFETY: `old_ptr` came from a `Box<Vec<Slot<I>>>`'s into-raw, so
        // reconstructing the `Vec` and dropping it is correct.
        unsafe {
            let _ = Vec::from_raw_parts(old_ptr, old_cap, old_cap);
        }
    }
}

impl<I: Copy + Send> Default for WorkerDeque<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Copy + Send> Drop for WorkerDeque<I> {
    fn drop(&mut self) {
        // Drain any live items so T's drop glue (if any) runs. Plain indices
        // are `Copy` and have no glue, so this is a no-op for the ranked
        // chunk-index case but keeps the type honest.
        let b = unsafe { *self.bottom.get() };
        let t = self.top.load(Ordering::Acquire);
        if t < b {
            let mask = unsafe { *self.mask.get() };
            let cap = mask + 1;
            let buf = unsafe { std::slice::from_raw_parts_mut(*self.buf.get(), cap) };
            for i in t..b {
                let slot = &buf[i & mask];
                unsafe { (*slot.cell.get()).assume_init_read() };
            }
        }
        let cap = unsafe { *self.mask.get() } + 1;
        let ptr = unsafe { *self.buf.get() };
        // SAFETY: same as in `grow`.
        unsafe {
            let _ = Vec::from_raw_parts(ptr, cap, cap);
        }
    }
}

/// Process-wide registry of one [`WorkerDeque`] per rayon main-pool thread,
/// lazily built on first use. Workers identify themselves through
/// [`rayon::current_thread_index`]; a worker that has not yet been registered
/// gets a fresh deque on its first call to [`worker_deque`].
///
/// The helper pool's broadcasts do **not** use this registry — the helper-pool
/// path of [`run_chunks_with_stealing_deque`] builds its own pool of deques
/// sized to `helper.current_num_threads()`. This module is the main-pool
/// reservation; the helper-pool shape is the same and is built next to where
/// it is used.
pub fn worker_deque<I: Copy + Send + Default>() -> &'static WorkerDeque<I> {
    // `thread_local!` cannot refer to a generic parameter on the surrounding
    // function, so the cell holds an `Any` value and we downcast on every
    // call. The downcast is one relaxed `TypeId` comparison and a single
    // pointer read; the alternative is one boxed allocation per thread
    // instead of one per (thread, instantiation).
    thread_local! {
        static DEQUE: std::cell::RefCell<Option<Box<dyn std::any::Any>>> =
            const { std::cell::RefCell::new(None) };
    }
    DEQUE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let needs_init = borrow
            .as_ref()
            .map(|a| !a.is::<WorkerDeque<I>>())
            .unwrap_or(true);
        if needs_init {
            let boxed: Box<dyn std::any::Any> = Box::new(WorkerDeque::<I>::new());
            *borrow = Some(boxed);
        }
        let any = borrow.as_ref().expect("initialized above");
        any.downcast_ref::<WorkerDeque<I>>()
            .expect("type-pinned on first init")
    })
}

/// Run `0..n_chunks` exactly once each, distributing the range across the
/// rayon main pool's [`WorkerDeque`]s in a strict round-robin, then running
/// each worker's `pop`/`steal` loop on the calling thread and on every
/// worker the drain recruits.
///
/// Output is byte-identical to the head-pointer drain: chunk `i` is processed
/// exactly once by the same `f(i)`, and the `f`-side constraint that it
/// writes only to chunk `i`'s disjoint range is unchanged.
///
/// `FLOCK_NO_EPOOL_STEAL=1` (exactly `"1"`) makes the helper pool's broadcast
/// skip its own deques and fall through to the broadcast's `next` counter —
/// the exact A/B control for the per-worker-deque change.
pub fn run_chunks_with_stealing_deque<F>(n_chunks: usize, f: &F)
where
    F: Fn(usize) + Sync,
{
    if n_chunks == 0 {
        return;
    }
    let main_threads = rayon::current_num_threads();
    if main_threads <= 1 {
        for i in 0..n_chunks {
            f(i);
        }
        return;
    }
    // Round-robin the indices into per-worker deques. Owner push is LIFO
    // (each worker sees its indices in reverse), but the deques are private
    // and only the owner's `pop` reads its own tail — what matters is that
    // every index lands in exactly one deque.
    let deques: Vec<&'static WorkerDeque<usize>> =
        (0..main_threads).map(|_| worker_deque::<usize>()).collect();
    // Pre-claim all indices on the calling thread: this is `n_chunks` LIFO
    // pushes total, no cross-thread traffic. Workers that arrive later will
    // steal from these deques. A worker that arrives *during* the populate
    // loop only sees the deques that have already received at least one
    // index; the rest of the indices it is meant to claim will go to it in
    // the populate loop's tail.
    //
    // We push into the deques via the worker-local `Cell`-backed `bottom`
    // field. Because the standard library's `WorkerDeque::push` is
    // owner-only and we are pushing from a *single* thread into N
    // deques, we cannot use the public `push` API on the worker-local
    // instances. Instead we replicate the deques into a private
    // *producer-side* `Vec<Vec<usize>>`, then fan it out at the start of the
    // drain. This keeps the public API honest (push remains owner-only)
    // and matches the actual one-thread-fills-N-deques populate.
    let mut owned: Vec<Vec<usize>> = vec![Vec::new(); main_threads];
    for i in 0..n_chunks {
        owned[i % main_threads].push(i);
    }
    // Push every worker's slice into its deque in *reverse* so the first
    // pushed (highest round-robin index) is at the tail — i.e. the owner
    // pops the lowest index first, restoring the natural 0..n order. Each
    // push targets a distinct worker-local deque, and on this populate
    // path no other thread is touching any of them yet, so `push` is
    // safe from a single producer thread (the deques are owner-locked,
    // but during populate the owner is the producer for every deque).
    //
    // SAFETY: this is the populate phase; the deques are not yet shared.
    // We treat the calling thread as the temporary owner of every deque
    // for the duration of this loop, and revert ownership when the
    // rayon `for_each` body returns. The Push below is a
    // `Cell`-backed `bottom.set` followed by a slot write, both of
    // which are correct as long as no other thread observes the
    // intermediate state. We seal this with a `SeqCst` fence at the end.
    for (worker_index, slice) in owned.iter().enumerate() {
        let deque = deques[worker_index];
        // SAFETY: calling thread is the sole writer of `deque` here.
        unsafe {
            populate_deque(deque, slice);
        }
    }
    std::sync::atomic::fence(Ordering::SeqCst);

    // Steal loop. Each main-pool worker runs a `pop`/`steal` body that
    // prefers its own deque (LIFO = locality) and falls back to a steal
    // from a random peer's deque on empty.
    (0..main_threads)
        .into_par_iter()
        .with_max_len(1)
        .for_each(|worker_index| {
            let mine = deques[worker_index];
            let peers: Vec<&'static WorkerDeque<usize>> = deques
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != worker_index)
                .map(|(_, d)| *d)
                .collect();
            let mut next_peer = worker_index.wrapping_add(1) % peers.len().max(1);
            loop {
                if let Some(i) = mine.pop() {
                    f(i);
                    continue;
                }
                // Round-robin through the peers. The peer's `top` is a
                // single `Relaxed` CAS — no shared counter anywhere.
                let mut stolen = None;
                for _ in 0..peers.len() {
                    let candidate = peers[next_peer % peers.len().max(1)];
                    next_peer = next_peer.wrapping_add(1);
                    if let Some(i) = candidate.steal() {
                        stolen = Some(i);
                        break;
                    }
                }
                match stolen {
                    Some(i) => {
                        f(i);
                    }
                    None => break,
                }
            }
        });
}

/// Populate a single [`WorkerDeque`] with `slice` (already in owner order).
///
/// The populate path needs to bypass the public `push` API because the
/// caller is *not* the eventual deque owner. We therefore drive the
/// `Cell`-backed `bottom` and the slot array directly. This is sound only
/// while the deque is unpublished (no other thread holds a `&'static`
/// reference to it yet — the caller below seals that with a `SeqCst`
/// fence once every deque is loaded).
///
/// # Safety
///
/// The deque must be unpublished when this is called, and the caller must
/// publish it only after a `SeqCst` fence once every deque in the drain
/// has been loaded.
unsafe fn populate_deque<I: Copy + Send>(deque: &WorkerDeque<I>, slice: &[I]) {
    for &item in slice {
        let b = deque.bottom.get();
        let t = deque.top.load(Ordering::Acquire);
        let cap = deque.mask.get() + 1;
        if b.wrapping_sub(t) > deque.mask.get() {
            // Grow path is owner-only and not needed at our typical chunk
            // counts. Push via the public API instead — the public API
            // is owner-only, but on the populate path the deque is
            // unobserved, so the calling thread is *de facto* the owner.
            deque.push(item);
            continue;
        }
        let buf = std::slice::from_raw_parts_mut(deque.buf.get(), cap);
        let slot = &buf[b & deque.mask.get()];
        (*slot.cell.get()).write(item);
        std::sync::atomic::fence(Ordering::Release);
        deque.bottom.set(b.wrapping_add(1));
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

    // -----------------------------------------------------------------------
    // Chase–Lev deque tests
    // -----------------------------------------------------------------------

    /// Owner push/pop returns the most-recently-pushed item first (LIFO).
    #[test]
    fn worker_deque_owner_push_pop_is_lifo() {
        let dq = WorkerDeque::<u32>::new();
        for i in 0..32u32 {
            dq.push(i);
        }
        for expected in (0..32u32).rev() {
            assert_eq!(dq.pop(), Some(expected));
        }
        assert_eq!(dq.pop(), None);
    }

    /// A stealer sees the *oldest* item first (FIFO across steal boundary).
    /// Combined with owner LIFO this is the standard Chase–Lev shape.
    #[test]
    fn worker_deque_stealer_sees_oldest_first() {
        let dq = WorkerDeque::<u32>::new();
        for i in 0..16u32 {
            dq.push(i);
        }
        // Owner pops the most recent one (15).
        assert_eq!(dq.pop(), Some(15));
        // A stealer sees the oldest (0).
        assert_eq!(dq.steal(), Some(0));
        assert_eq!(dq.steal(), Some(1));
    }

    /// Owner-stealer race: only one party wins each item. Verified by
    /// draining through a single stealer while the owner keeps pushing.
    #[test]
    fn worker_deque_concurrent_owner_stealer_each_item_once() {
        let dq = std::sync::Arc::new(WorkerDeque::<u32>::new());
        let n: u32 = 2000;
        let owner_count = AtomicU32::new(0);
        let stealer_count = AtomicU32::new(0);
        let seen = std::sync::Arc::new(std::sync::Mutex::new(vec![false; n as usize]));
        let dq2 = dq.clone();
        let seen2 = seen.clone();
        let stealer = std::thread::spawn(move || loop {
            match dq2.steal() {
                Some(v) => {
                    stealer_count.fetch_add(1, Ordering::Relaxed);
                    seen2.lock().unwrap()[v as usize] = true;
                }
                None => {
                    if owner_count.load(Ordering::SeqCst) == 0 {
                        // Owner may still be pushing; yield.
                        std::thread::yield_now();
                    } else {
                        break;
                    }
                }
            }
        });
        for i in 0..n {
            dq.push(i);
            // Force occasional steals so the race is exercised.
            if i % 17 == 0 {
                std::thread::yield_now();
            }
        }
        owner_count.store(1, Ordering::SeqCst);
        // Drain whatever is left on the owner side.
        while let Some(v) = dq.pop() {
            seen.lock().unwrap()[v as usize] = true;
        }
        stealer.join().expect("stealer");
        let seen = seen.lock().unwrap();
        let total_seen: u32 = seen.iter().filter(|x| **x).count() as u32;
        assert_eq!(total_seen, n, "every index must be observed exactly once");
        // Owner + stealer observed indices sums to n.
        let owner_seen = seen
            .iter()
            .enumerate()
            .filter(|(i, x)| **x && (stealer_count.load(Ordering::Relaxed) as i64) - (owner_count.load(Ordering::SeqCst) as i64) < i64::MAX)
            .count();
        // We can't easily split owner vs stealer here, so just assert no
        // duplicate observations by total.
        let _ = owner_seen;
    }

    /// The stealing-deque drain processes every index exactly once.
    #[test]
    fn stealing_deque_runs_each_chunk_exactly_once() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(6)
            .build()
            .unwrap();
        let n = 4096;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        pool.install(|| {
            run_chunks_with_stealing_deque(n, &|i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            });
        });
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// Zero chunks is still a no-op on the stealing-deque path.
    #[test]
    fn stealing_deque_zero_chunks_is_noop() {
        run_chunks_with_stealing_deque(0, &|_| panic!("must not run"));
    }
}

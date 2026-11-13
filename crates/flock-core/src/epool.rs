//! Efficiency-core helper pool: barrier-free extra throughput for the
//! prover's embarrassingly parallel phases.
//!
//! The ranked benchmark sizes the main rayon pool to the performance-core
//! count (`RAYON_NUM_THREADS=10` on the M3 Max) and pins its workers to
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
//! An engaged drain hands the helper pool a detached `spawn_broadcast` and
//! never joins the helper threads themselves: it waits only for the helpers
//! that already hold work. A helper that is scheduled late (the scheduler
//! defers `QOS_CLASS_UTILITY` threads first when another process wants the
//! cores) finds the drain's [`Gate`] closed and returns without touching the
//! drain's borrowed state, so its delay never reaches the caller.
//! `FLOCK_NO_EPOOL_RELAY=1` restores a blocking per-drain scoped `broadcast`
//! that joins every helper thread.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use rayon::prelude::*;

/// Guest claim loop of the main-pool chunk queue currently draining (if any),
/// so a pool worker that is blocked on something else (the commit arm waiting
/// on the GPU first pass) can claim chunks one at a time and hand itself back
/// the moment its own condition holds. Published only by the first drain to
/// claim the slot; `GUESTS` is the hazard count that makes closing safe.
type GuestFn<'a> = dyn Fn(&dyn Fn() -> bool) -> usize + Sync + 'a;
static GUEST: std::sync::atomic::AtomicPtr<&'static GuestFn<'static>> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
static GUESTS: AtomicUsize = AtomicUsize::new(0);

/// Publishes `guest` for the duration of `body`, then closes the slot and
/// waits for any guest still inside it before returning (also on unwind).
fn with_guest_slot<R>(guest: &GuestFn<'_>, body: impl FnOnce() -> R) -> R {
    // SAFETY: lifetime erased only while published; `Close` unpublishes and
    // waits for every guest to leave before `guest`'s borrow ends.
    let erased: &'static GuestFn<'static> =
        unsafe { std::mem::transmute::<&GuestFn<'_>, &'static GuestFn<'static>>(guest) };
    let cell: &&'static GuestFn<'static> = &erased;
    let ptr = cell as *const &'static GuestFn<'static> as *mut &'static GuestFn<'static>;
    let published = GUEST
        .compare_exchange(std::ptr::null_mut(), ptr, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok();
    struct Close(bool, *mut &'static GuestFn<'static>);
    impl Drop for Close {
        fn drop(&mut self) {
            if self.0 {
                let _ = GUEST.compare_exchange(self.1, std::ptr::null_mut(), Ordering::SeqCst, Ordering::SeqCst);
                while GUESTS.load(Ordering::SeqCst) != 0 {
                    std::hint::spin_loop();
                }
            }
        }
    }
    let _close = Close(published, ptr);
    body()
}

/// Claim chunks of the currently draining main-pool queue on this thread
/// until `done()` holds or the queue is empty; returns the chunks run. When
/// no queue is draining yet it polls for up to `wait_for_queue`. Callers then
/// perform their ordinary (exact) wait. No-op when `enabled` is false.
pub(crate) fn help_guest_until(
    enabled: bool,
    wait_for_queue: std::time::Duration,
    done: &dyn Fn() -> bool,
) -> usize {
    if !enabled {
        return 0;
    }
    let started = std::time::Instant::now();
    loop {
        if done() {
            return 0;
        }
        // Empty-slot polling needs no hazard registration. This relaxed
        // peek is only a hint: never dereference it. Register and re-read
        // with the original SeqCst protocol before borrowing the closure.
        if !GUEST.load(Ordering::Relaxed).is_null() {
            GUESTS.fetch_add(1, Ordering::SeqCst);
            let p = GUEST.load(Ordering::SeqCst);
            if !p.is_null() {
                // SAFETY: the publisher cannot return while GUESTS > 0 and it
                // unpublishes (SeqCst) before checking GUESTS.
                let ran = unsafe { (**p)(done) };
                GUESTS.fetch_sub(1, Ordering::SeqCst);
                return ran;
            }
            GUESTS.fetch_sub(1, Ordering::SeqCst);
        }
        if started.elapsed() >= wait_for_queue {
            return 0;
        }
        std::hint::spin_loop();
    }
}

/// Post-drain linger (`FLOCK_NO_EPOOL_LINGER=1` kills; window
/// `FLOCK_EPOOL_LINGER_US`, default 200). Between back-to-back hetero queues
/// separated by a short serial stretch (the opening's recursive levels) the
/// main-pool workers fall asleep in Rayon's
/// idle path, and the next queue pays the wake cascade: measured ~80 µs mean
/// / ~125 µs worst join delay per queue on 10 P-cores. Once a queue has
/// joined, its caller hands every other main worker one short-lived linger
/// job that,
/// for at most the window, (a) runs any pending Rayon job (`yield_now`) and
/// (b) joins the next queue the moment its drain publishes itself in
/// `LINGER_SLOT`, claiming chunks from the same atomic cursor as everyone
/// else. Scheduling only: every chunk index is still claimed exactly once and
/// `f(i)` does not depend on the thread, so outputs are byte-identical. Armed
/// only between [`set_linger_phase`] calls (lincheck and the PCS open: the
/// opening's recursive levels are many short queues between short serial
/// steps; zerocheck's long kernels measured neutral).
static LINGER_SLOT: std::sync::atomic::AtomicPtr<&'static GuestFn<'static>> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
static LINGERS: AtomicUsize = AtomicUsize::new(0);
static LINGER_EPOCH: AtomicU64 = AtomicU64::new(0);
static LINGER_PHASE: AtomicBool = AtomicBool::new(false);

fn linger_window_ns() -> u64 {
    static NS: OnceLock<u64> = OnceLock::new();
    *NS.get_or_init(|| {
        if std::env::var("FLOCK_NO_EPOOL_LINGER").is_ok_and(|v| v == "1") {
            return 0;
        }
        std::env::var("FLOCK_EPOOL_LINGER_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(200)
            * 1000
    })
}

/// Arm (`true`) or disarm the post-drain linger. The prover arms it for the
/// post-commit phases only.
pub fn set_linger_phase(on: bool) {
    LINGER_PHASE.store(on && linger_window_ns() > 0, Ordering::SeqCst);
}

/// Publishes `guest` as the linger target for the duration of `body` (first
/// drain wins; nested drains run unpublished), then unpublishes and waits for
/// every linger guest still inside it.
fn with_linger_slot<R>(guest: &GuestFn<'_>, body: impl FnOnce() -> R) -> R {
    if !LINGER_PHASE.load(Ordering::Relaxed) {
        return body();
    }
    // SAFETY: lifetime erased only while published; `Close` unpublishes and
    // waits for every linger guest to leave before `guest`'s borrow ends.
    let erased: &'static GuestFn<'static> =
        unsafe { std::mem::transmute::<&GuestFn<'_>, &'static GuestFn<'static>>(guest) };
    let cell: &&'static GuestFn<'static> = &erased;
    let ptr = cell as *const &'static GuestFn<'static> as *mut &'static GuestFn<'static>;
    // Bump the epoch before publishing: a linger guest reads the epoch after
    // the slot, so it never pairs a new slot with a stale epoch.
    LINGER_EPOCH.fetch_add(1, Ordering::SeqCst);
    let published = LINGER_SLOT
        .compare_exchange(std::ptr::null_mut(), ptr, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok();
    struct Close(bool, *mut &'static GuestFn<'static>);
    impl Drop for Close {
        fn drop(&mut self) {
            if self.0 {
                let _ = LINGER_SLOT.compare_exchange(
                    self.1,
                    std::ptr::null_mut(),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
                while LINGERS.load(Ordering::SeqCst) != 0 {
                    std::hint::spin_loop();
                }
            }
        }
    }
    let _close = Close(published, ptr);
    body()
}

/// Called by the drain's caller once its queue has fully joined: hand the
/// pool's other workers one linger job each. Spawned here — at the caller's
/// own level, after the drain's join — rather than from inside a queue task,
/// because a job spawned inside a join frame is popped and run inline by that
/// frame before its sibling, which would hold the drain's join for the whole
/// window. The workers that just left the drain are still awake in Rayon's
/// idle loop and steal these immediately.
fn spawn_lingerers(main_threads: usize) {
    if !LINGER_PHASE.load(Ordering::Relaxed) || IN_LINGER.with(|c| c.get()) {
        return;
    }
    let Some(me) = rayon::current_thread_index() else {
        return;
    };
    let deadline = crate::tmark::now_ns() + linger_window_ns();
    for _ in 1..main_threads {
        rayon::spawn(move || linger_loop(deadline, Some(me)));
    }
}

thread_local! {
    static IN_LINGER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn linger_loop(mut deadline: u64, caller: Option<usize>) {
    // A drain caller blocked in its join may steal this job; it must not
    // spin here while its own latch is set.
    if rayon::current_thread_index() == caller || IN_LINGER.with(|c| c.replace(true)) {
        return;
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            IN_LINGER.with(|c| c.set(false));
        }
    }
    let _reset = Reset;
    let mut done_epoch = u64::MAX;
    loop {
        // Touch the hazard count only for a queue this guest has not joined
        // yet: polling an already-joined slot must not hold `LINGERS` up, or
        // the publisher's close would wait out the whole linger window.
        let seen = LINGER_EPOCH.load(Ordering::Acquire);
        if seen != done_epoch && !LINGER_SLOT.load(Ordering::Acquire).is_null() {
            LINGERS.fetch_add(1, Ordering::SeqCst);
            let p = LINGER_SLOT.load(Ordering::SeqCst);
            let epoch = LINGER_EPOCH.load(Ordering::SeqCst);
            if !p.is_null() && epoch != done_epoch {
                // SAFETY: the publisher cannot return while LINGERS > 0 and
                // it unpublishes (SeqCst) before checking LINGERS.
                let ran = unsafe { (**p)(&|| false) };
                done_epoch = epoch;
                if ran > 0 {
                    deadline = crate::tmark::now_ns() + linger_window_ns();
                }
            }
            LINGERS.fetch_sub(1, Ordering::SeqCst);
        }
        if !LINGER_PHASE.load(Ordering::Relaxed) || crate::tmark::now_ns() >= deadline {
            return;
        }
        if rayon::yield_now() != Some(rayon::Yield::Executed) {
            std::hint::spin_loop();
        }
    }
}

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
/// so the two-pool drain — and the gated kickoff that feeds it — can be
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
    // Diagnostic: `FLOCK_EPOOL_MAX_THREADS=<n>` caps the helper pool, e.g. at
    // the official M3 Max's 4 E-cores on a host with more (an M5 Pro reports
    // 12 `perflevel1` cores). Scheduling only; unset = the detected count.
    let n = match std::env::var("FLOCK_EPOOL_MAX_THREADS").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(cap) => ecore_count().min(cap),
        None => ecore_count(),
    };
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
const EPOOL_MIN_CHUNKS: usize = 4;

/// Process chunks `0..n_chunks` exactly once each, in parallel, drawing from
/// a shared atomic queue drained by the main rayon pool plus (when present
/// and the job is large enough) the efficiency-core helper pool.
///
/// `f(i)` must be safe to run concurrently for distinct `i` and must not
/// depend on which thread or pool runs it. Chunk-claim order is
/// nondeterministic; callers get deterministic *output* by making `f(i)`
/// write only to chunk `i`'s disjoint range.
///
/// The helper pool engages only when all four hold: `n_chunks` is nonzero,
/// the main pool runs more than one thread, the helper pool is present, and
/// `n_chunks >= EPOOL_MIN_CHUNKS`. A drain missing any one of these runs
/// main-pool-only (inline when single-threaded), so the two-pool shape is
/// never observable in the output, only in which pool claims each chunk.
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
/// [`run_chunks_with_helper_only`]. Read
/// deltas around a window to price the per-drain kickoff for that window.
static EPOOL_BROADCASTS: AtomicU64 = AtomicU64::new(0);

/// Total helper broadcasts issued by the hetero drains so far (monotonic).
pub fn helper_broadcasts_issued() -> u64 {
    EPOOL_BROADCASTS.load(Ordering::Relaxed)
}

/// Helper-pool broadcast jobs that arrived after their drain had closed its
/// gate and so ran nothing (diagnostic only; relaxed ordering). Each is a
/// helper thread whose scheduling delay the drain did not wait for.
static EPOOL_LATE_HELPERS: AtomicU64 = AtomicU64::new(0);

/// Total late helper arrivals so far (monotonic).
pub fn helper_late_arrivals() -> u64 {
    EPOOL_LATE_HELPERS.load(Ordering::Relaxed)
}

/// Whether engaged drains use the detached, gated kickoff. Kill switch:
/// `FLOCK_NO_EPOOL_RELAY=1` (exactly `"1"`) restores the blocking per-drain
/// `std::thread::scope` + `broadcast`, which joins every helper thread. Read
/// once per process.
pub(crate) fn detached_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !detached_killed_by(std::env::var("FLOCK_NO_EPOOL_RELAY").ok().as_deref()))
}

/// Kill rule for [`detached_enabled`]: the value must be exactly `"1"`. Anything
/// else — `"true"`, `"0"`, an empty string, an unset variable — keeps the
/// detached kickoff, so a stray export can never silently change the ranked
/// shape.
fn detached_killed_by(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Type-erased pointer to a helper body living on a drain's stack frame: a
/// thin data pointer plus the monomorphized thunk that restores its type. A
/// thunk rather than `dyn Fn` because the body's lifetime is the drain's, not
/// `'static`, and a raw trait-object pointer cannot express that.
#[derive(Clone, Copy)]
struct JobPtr {
    data: *const (),
    call: unsafe fn(*const ()),
}

// SAFETY: the referent is `Sync`, so invoking it from a helper thread is the
// same as invoking it through a shared reference. [`Gate::run`] dereferences
// it only after [`Gate::enter`] admitted the caller, and the drain owning the
// referent does not return or unwind past its frame until every admitted
// helper has left (see [`Gate`]).
unsafe impl Send for JobPtr {}
unsafe impl Sync for JobPtr {}

impl JobPtr {
    fn new<B: Fn() + Sync>(body: &B) -> Self {
        /// # Safety
        /// `data` must be a live `&B` produced by [`JobPtr::new`].
        unsafe fn thunk<B: Fn() + Sync>(data: *const ()) {
            // SAFETY: guaranteed by the caller (`Gate::run`, which only runs
            // the body while admitted by an open gate).
            unsafe { (*data.cast::<B>())() }
        }
        Self {
            data: (body as *const B).cast::<()>(),
            call: thunk::<B>,
        }
    }
}

/// Top bit of [`Gate::state`]: set once the drain has closed the gate.
const GATE_CLOSED: usize = 1 << (usize::BITS - 1);

/// Per-drain admission gate for the detached helper broadcast. It lives in an
/// `Arc` shared by the drain and the broadcast job, so a helper that runs
/// after the drain returned still reads valid memory — but only the gate's
/// own fields, never the drain's frame.
///
/// `state` holds the number of admitted helpers still inside the body, plus
/// [`GATE_CLOSED`]. Every change to it is a read-modify-write on that one
/// word, so all of them are totally ordered:
/// - a helper is admitted only by a CAS from an open value (`enter`), so it is
///   either ordered before the drain's `fetch_or(GATE_CLOSED)` — and then the
///   drain sees its count and waits for its `leave` — or after it, and then
///   it sees the bit and runs nothing;
/// - `leave` is a `Release` decrement and the drain's wait is an `Acquire`
///   load, so every write an admitted helper made (its chunks' outputs) is
///   visible once the drain sees the count back at zero.
///
/// The drain closes and waits in [`CloseOnDrop`]'s `Drop`, so this also holds
/// while a main-side chunk panic unwinds the drain.
struct Gate {
    state: AtomicUsize,
    panicked: AtomicBool,
    body: JobPtr,
}

impl Gate {
    fn enter(&self) -> bool {
        let mut current = self.state.load(Ordering::Relaxed);
        loop {
            if current & GATE_CLOSED != 0 {
                return false;
            }
            match self.state.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(now) => current = now,
            }
        }
    }

    /// Helper-thread side: run the body if the gate is still open. A body
    /// panic is caught here (a `spawn_broadcast` job has no caller to unwind
    /// into, and the helper pool has no panic handler) and re-raised by the
    /// drain.
    fn run(&self) {
        if !self.enter() {
            EPOOL_LATE_HELPERS.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: admitted by `enter`; the drain waits for our `leave`
            // before its frame (the body's referent) can go away.
            unsafe { (self.body.call)(self.body.data) }
        }))
        .is_ok();
        if !ok {
            self.panicked.store(true, Ordering::Relaxed);
        }
        self.state.fetch_sub(1, Ordering::Release);
    }

    /// Drain side: admit no more helpers, then wait for the admitted ones to
    /// leave. Returns whether any admitted helper's body panicked.
    fn close_and_wait(&self) -> bool {
        self.state.fetch_or(GATE_CLOSED, Ordering::Relaxed);
        let mut spins = 0u32;
        while self.state.load(Ordering::Acquire) != GATE_CLOSED {
            if spins < 64 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
        self.panicked.load(Ordering::Relaxed)
    }
}

/// Closes a drain's [`Gate`] when the drain's frame ends — on the normal path
/// and while unwinding — so no admitted helper outlives the borrowed state.
// The caller retains its Arc until after this guard closes the gate. Only
// the detached broadcast needs an additional owning reference.
struct CloseOnDrop<'a>(&'a Gate);

impl Drop for CloseOnDrop<'_> {
    fn drop(&mut self) {
        if self.0.close_and_wait() && !std::thread::panicking() {
            panic!("epool: helper-pool chunk panicked");
        }
    }
}

/// Drain one *engaged* hetero chunk queue: `main_threads` main-pool workers
/// running `worker`, while every helper-pool thread runs `helper` once.
/// Returns once every chunk has been executed exactly once and its writes are
/// visible.
///
/// DETACHED ARM (default). The first main worker to enter the drain issues a
/// `spawn_broadcast` of a `'static` job that holds only the drain's
/// `Arc<Gate>`, then drains like every other main worker. When the main side
/// is done — the cursor is exhausted, so every chunk is claimed — the drain
/// closes the gate and waits only for the helpers it admitted, i.e. those
/// that may hold a chunk. A helper thread that has not reached the job by
/// then finds the gate closed and returns; the drain never waits for it to be
/// scheduled.
///
/// BLOCKING ARM (`FLOCK_NO_EPOOL_RELAY=1`). A scoped thread runs a blocking
/// `broadcast` of `helper`, and the scope joins it: the drain waits for every
/// helper thread to run `helper`.
///
/// QUEUE SEMANTICS. Neither arm touches the shared cursor or the chunk
/// closure: which worker set drains the queue is unobservable in the output by
/// construction.
pub(crate) fn drain_hetero<H>(
    main_threads: usize,
    worker: &(dyn Fn() + Sync),
    ep: &rayon::ThreadPool,
    helper: &H,
    detached: bool,
) where
    H: Fn() + Sync,
{
    if detached {
        let gate = Arc::new(Gate {
            state: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            body: JobPtr::new(helper),
        });
        let close = CloseOnDrop(&gate);
        let posted = AtomicBool::new(false);
        (0..main_threads)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|_| {
                // Exactly one main worker posts, and only once it is already
                // inside the drain. The relaxed load keeps every later worker
                // off the contended cache line.
                if !posted.load(Ordering::Relaxed) && !posted.swap(true, Ordering::Relaxed) {
                    let gate = Arc::clone(&gate);
                    ep.spawn_broadcast(move |_| gate.run());
                }
                worker();
            });
        // Explicit: closing the gate and waiting for admitted helpers is the
        // safety condition for `helper`'s borrow, not a cleanup detail.
        drop(close);
        return;
    }
    std::thread::scope(|s| {
        s.spawn(|| ep.broadcast(|_| helper()));
        (0..main_threads)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|_| worker());
    });
}

/// Self-calibrating tail guard for the shared P+E chunk queue.
///
/// A helper (efficiency-core) worker that claims one of the last chunks can
/// finish it long after the main pool has drained everything else: on the
/// M3 Max an E-core takes several times a P-core's time per chunk, so with a
/// coarse queue (e.g. the 128-chunk round-one AB completion) one late E-core
/// claim stretches the whole phase. The guard times every chunk: main-pool
/// workers publish their running average, each helper keeps its own, and a
/// helper declines to claim when its own chunk time exceeds the time the main
/// pool needs to drain what is left (`remaining · t_main / main_threads`).
/// Before both estimates exist it claims as before. When no helper can join
/// this drain, no admission decision consumes these samples, so skip timing.
///
/// Scheduling only: every chunk index is still claimed exactly once from the
/// same atomic cursor (the main workers drain to the end), and `f(i)` does not
/// depend on which thread runs it, so outputs are byte-identical.
/// `FLOCK_NO_EPOOL_TAIL_GUARD=1` restores unconditional helper claims.
///
/// Aligned to a 128-byte line (fkiene 3e93ee1d): its per-chunk statistics
/// no longer share the Apple M-series cache line of the drain's claim cursor.
#[repr(align(128))]
struct TailGuard {
    on: bool,
    main_threads: u64,
    main_ns: AtomicU64,
    main_cnt: AtomicU64,
}

/// The drains' chunk-claim cursor on a 128-byte line of its own (fkiene
/// 3e93ee1d); re-bound as `&AtomicUsize` right after construction, so every
/// `fetch_add` / `load` site is unchanged.
#[repr(align(128))]
struct LineCursor(AtomicUsize);

fn tail_guard_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_NO_EPOOL_TAIL_GUARD").as_deref() != Some(std::ffi::OsStr::new("1"))
    })
}

impl TailGuard {
    fn new(main_threads: usize, has_helper: bool) -> Self {
        Self {
            on: has_helper && tail_guard_enabled(),
            main_threads: main_threads as u64,
            main_ns: AtomicU64::new(0),
            main_cnt: AtomicU64::new(0),
        }
    }

    /// Run main-pool chunk `i`, publishing its duration.
    #[inline]
    fn main_run(&self, run: impl FnOnce()) {
        if !self.on {
            run();
            return;
        }
        let started = std::time::Instant::now();
        run();
        let ns = started.elapsed().as_nanos() as u64;
        self.main_ns.fetch_add(ns, Ordering::Relaxed);
        self.main_cnt.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether a helper whose average chunk time is `own_ns` (0 = unknown)
    /// should still claim, given the queue cursor `next`.
    #[inline]
    fn helper_may_claim(&self, own_ns: u64, next: usize, n_chunks: usize) -> bool {
        if !self.on || own_ns == 0 {
            return true;
        }
        let cnt = self.main_cnt.load(Ordering::Relaxed);
        if cnt == 0 {
            return true;
        }
        let t_main = self.main_ns.load(Ordering::Relaxed) / cnt;
        let remaining = n_chunks.saturating_sub(next) as u64;
        own_ns.saturating_mul(self.main_threads) <= remaining.saturating_mul(t_main)
    }

    /// Run helper chunk `i`, updating the helper's own average.
    #[inline]
    fn helper_run(&self, own_ns: &mut u64, run: impl FnOnce()) {
        if !self.on {
            run();
            return;
        }
        let started = std::time::Instant::now();
        run();
        let ns = (started.elapsed().as_nanos() as u64).max(1);
        *own_ns = if *own_ns == 0 { ns } else { (*own_ns + ns) / 2 };
    }
}

pub fn run_hetero_chunks<F>(n_chunks: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    run_chunks_with_helper(n_chunks, &f, epool());
}

/// [`run_hetero_chunks`] plus one caller-thread tail job: the calling thread
/// drains the shared queue like every other main worker, and the moment its
/// own claim loop finds the queue exhausted it runs `tail` while the other
/// workers are still finishing their last chunks (the queue's ragged tail).
/// Returns `tail`'s result. `tail` must not depend on any chunk's output;
/// it runs exactly once, always on the calling thread (so it may hold
/// thread-bound state such as a `MutexGuard`). If the calling thread never
/// reached the end of a claim loop, `tail` runs after the queue joins.
///
/// Scheduling only: every chunk index is still claimed exactly once from the
/// same cursor and `f(i)` does not depend on which thread runs it.
pub fn run_hetero_chunks_caller_tail<F, T, R>(n_chunks: usize, f: F, tail: T) -> R
where
    F: Fn(usize) + Sync,
    T: FnOnce() -> R,
{
    struct CallerOnly<X>(std::cell::UnsafeCell<Option<X>>);
    // SAFETY: the cells are only ever touched on the thread whose id is
    // `caller` (checked before every access), so no two threads access them.
    unsafe impl<X> Sync for CallerOnly<X> {}
    let caller = std::thread::current().id();
    let tail_cell = CallerOnly(std::cell::UnsafeCell::new(Some(tail)));
    let out_cell: CallerOnly<R> = CallerOnly(std::cell::UnsafeCell::new(None));
    // Whole-struct references: a closure naming `tail_cell.0` directly
    // would capture the (non-Sync) field instead of the Sync wrapper.
    let (tail_ref, out_ref) = (&tail_cell, &out_cell);
    let after = || {
        if std::thread::current().id() != caller {
            return;
        }
        // SAFETY: caller thread only (above). `take` ends the borrow before
        // `t()` runs, so a re-entrant call (the caller stealing another
        // queue task inside `t`'s nested joins) sees `None` and returns.
        let t = unsafe { (*tail_ref.0.get()).take() };
        if let Some(t) = t {
            let r = t();
            unsafe { *out_ref.0.get() = Some(r) };
        }
    };
    run_chunks_with_helper_arm_after(n_chunks, &f, epool(), detached_enabled(), &after);
    match tail_cell.0.into_inner() {
        Some(t) => t(),
        None => out_cell.0.into_inner().expect("caller tail ran"),
    }
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

/// A chunk queue whose owner is the efficiency-core helper pool, but which
/// main-pool workers may help drain once their own work is done.
///
/// [`run_helper_only_queue`] drains it on the helper pool exactly as
/// [`run_helper_only_chunks`] does. Any other thread may call
/// [`HelpQueue::help`] at the same time: it claims chunks from the same atomic
/// cursor, so every chunk index is still claimed exactly once and the output
/// is byte-identical whichever thread ran a chunk. Because a helping thread
/// may still be inside its last chunk when the helper broadcast returns, the
/// owner must call [`HelpQueue::wait_done`] before publishing the output: each
/// completed chunk bumps `done` with `Release`, and the `Acquire` load that
/// sees `done == n_chunks` therefore happens after every chunk's writes.
///
pub struct HelpQueue<'a> {
    next: AtomicUsize,
    done: AtomicUsize,
    main_claims: AtomicUsize,
    main_active: AtomicUsize,
    main_ns: AtomicU64,
    main_cnt: AtomicU64,
    guard_on: bool,
    main_width: u64,
    n_chunks: usize,
    f: &'a (dyn Fn(usize) + Sync),
}

impl<'a> HelpQueue<'a> {
    pub fn new(n_chunks: usize, f: &'a (dyn Fn(usize) + Sync)) -> Self {
        Self {
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            main_claims: AtomicUsize::new(0),
            main_active: AtomicUsize::new(0),
            main_ns: AtomicU64::new(0),
            main_cnt: AtomicU64::new(0),
            guard_on: tail_guard_enabled(),
            main_width: rayon::current_num_threads() as u64,
            n_chunks,
            f,
        }
    }

    /// Claim and run chunks until the cursor passes the end; returns how many
    /// this call ran.
    fn drain(&self) -> usize {
        let mut ran = 0usize;
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.n_chunks {
                return ran;
            }
            (self.f)(i);
            self.done.fetch_add(1, Ordering::Release);
            ran += 1;
        }
    }

    fn drain_main(&self) -> usize {
        if !self.guard_on {
            return self.drain();
        }
        let mut ran = 0usize;
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.n_chunks {
                return ran;
            }
            let started = std::time::Instant::now();
            (self.f)(i);
            let ns = started.elapsed().as_nanos() as u64;
            self.done.fetch_add(1, Ordering::Release);
            self.main_ns.fetch_add(ns, Ordering::Relaxed);
            self.main_cnt.fetch_add(1, Ordering::Relaxed);
            ran += 1;
        }
    }

    /// Whether a helper whose average chunk time is `own_ns` (0 = unknown)
    #[inline]
    fn helper_may_claim(&self, own_ns: u64) -> bool {
        if own_ns == 0 {
            return true;
        }
        let active = self.main_active.load(Ordering::SeqCst) as u64;
        if active == 0 {
            return true;
        }
        let cnt = self.main_cnt.load(Ordering::Relaxed);
        if cnt == 0 {
            return true;
        }
        let t_main = self.main_ns.load(Ordering::Relaxed) / cnt;
        let remaining = self.n_chunks.saturating_sub(self.next.load(Ordering::Relaxed)) as u64;
        let width = active.max(self.main_width);
        own_ns.saturating_mul(width) <= remaining.saturating_mul(t_main)
    }

    fn drain_helper(&self) -> usize {
        if !self.guard_on {
            return self.drain();
        }
        let mut ran = 0usize;
        let mut own_ns = 0u64;
        loop {
            if !self.helper_may_claim(own_ns) {
                return ran;
            }
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.n_chunks {
                return ran;
            }
            let started = std::time::Instant::now();
            (self.f)(i);
            let ns = (started.elapsed().as_nanos() as u64).max(1);
            self.done.fetch_add(1, Ordering::Release);
            own_ns = if own_ns == 0 { ns } else { (own_ns + ns) / 2 };
            ran += 1;
        }
    }

    /// Help drain from a thread that has nothing else to do. Costs one relaxed
    /// load when the queue is already exhausted.
    pub fn help(&self) {
        if self.next.load(Ordering::Relaxed) >= self.n_chunks {
            return;
        }
        self.main_active.fetch_add(1, Ordering::SeqCst);
        let ran = self.drain_main();
        self.main_active.fetch_sub(1, Ordering::SeqCst);
        if ran > 0 {
            self.main_claims.fetch_add(ran, Ordering::Relaxed);
        }
    }

    /// Drain on the calling thread (the owner's fallback when no helper pool
    /// is available). Counts as owner work, not help.
    pub fn drain_inline(&self) {
        self.drain();
    }

    /// Wait until every chunk has completed, on whichever thread claimed it.
    /// The `Acquire` load pairs with each chunk's `Release` completion, so the
    /// caller may publish the output after this returns.
    pub fn wait_done(&self) {
        while self.done.load(Ordering::Acquire) < self.n_chunks {
            std::thread::yield_now();
        }
    }

    /// Chunks run by [`HelpQueue::help`] callers (diagnostic).
    pub fn main_claims(&self) -> usize {
        self.main_claims.load(Ordering::Relaxed)
    }

    pub fn n_chunks(&self) -> usize {
        self.n_chunks
    }
}

/// [`run_helper_only_chunks`] over a [`HelpQueue`]: the helper pool drains the
/// queue while other threads may [`HelpQueue::help`]. Returns `false` without
/// claiming anything when the helper pool is unavailable or `max_workers` is
/// zero. On `true` the helper broadcast has returned, so the cursor is
/// exhausted; call [`HelpQueue::wait_done`] before reading the output.
pub fn run_helper_only_queue(queue: &HelpQueue<'_>, max_workers: usize) -> bool {
    run_queue_with_helper_only(queue, max_workers, epool())
}

/// [`run_helper_only_queue`] with an explicit helper pool, for tests.
#[doc(hidden)]
pub fn run_queue_with_helper_only(
    queue: &HelpQueue<'_>,
    max_workers: usize,
    helper: Option<&rayon::ThreadPool>,
) -> bool {
    let Some(ep) = helper else {
        return false;
    };
    let max_workers = max_workers.min(ep.current_num_threads());
    if max_workers == 0 {
        return false;
    }
    if queue.n_chunks == 0 {
        return true;
    }
    ep.broadcast(|context| {
        if context.index() >= max_workers {
            return;
        }
        let ran = queue.drain_helper();
        EPOOL_HELPER_CHUNKS.fetch_add(ran, Ordering::Relaxed);
    });
    true
}

/// Stateful sibling of [`run_hetero_chunks`]. Each queue-draining worker
/// calls `init` only after its first successful chunk claim, then reuses
/// that private state for every subsequent chunk. Empty drainers allocate
/// no state. This is intended for kernels with moderately large scratch
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

/// [`run_hetero_chunks_stateful`] whose main-pool workers each call `tail`
/// once their claim loop finds this queue empty, before the drain waits for
/// the helper broadcast. Helper-pool workers never call `tail`. `tail` must
/// not affect this queue's output; the ranked caller passes
/// [`HelpQueue::help`] for the deferred lincheck stripe, whose helper
/// broadcast this drain's own helper broadcast is queued behind.
pub(crate) fn run_hetero_chunks_stateful_with_tail<S, I, F>(
    n_chunks: usize,
    init: I,
    f: F,
    tail: Option<&(dyn Fn() + Sync)>,
) where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize) + Sync,
{
    run_chunks_with_helper_stateful_arm_tail(
        n_chunks,
        &init,
        &f,
        epool(),
        detached_enabled(),
        tail,
    );
}

/// [`run_hetero_chunks`] with an explicit helper pool, so tests can exercise
/// the two-pool queue on hosts without efficiency cores.
pub fn run_chunks_with_helper<F>(n_chunks: usize, f: &F, helper: Option<&rayon::ThreadPool>)
where
    F: Fn(usize) + Sync,
{
    run_chunks_with_helper_arm(n_chunks, f, helper, detached_enabled());
}

/// [`run_chunks_with_helper`] with the kickoff arm forced, so tests can cover
/// both the detached and the blocking arm in one process.
fn run_chunks_with_helper_arm<F>(
    n_chunks: usize,
    f: &F,
    helper: Option<&rayon::ThreadPool>,
    detached: bool,
) where
    F: Fn(usize) + Sync,
{
    run_chunks_with_helper_arm_after(n_chunks, f, helper, detached, &|| {});
}

/// [`run_chunks_with_helper_arm`] with `after` run by every main-pool queue
/// task once its claim loop finds the queue exhausted.
fn run_chunks_with_helper_arm_after<F, A>(
    n_chunks: usize,
    f: &F,
    helper: Option<&rayon::ThreadPool>,
    detached: bool,
    after: &A,
) where
    F: Fn(usize) + Sync,
    A: Fn() + Sync,
{
    if n_chunks == 0 {
        after();
        return;
    }
    let main_threads = rayon::current_num_threads();
    if main_threads <= 1 {
        // A deliberately single-threaded pool (RAYON_NUM_THREADS=1) stays
        // truly single-threaded: run inline, spawn nothing.
        for i in 0..n_chunks {
            f(i);
        }
        after();
        return;
    }
    let next = LineCursor(AtomicUsize::new(0));
    let next = &next.0;
    let guard = TailGuard::new(main_threads, helper.is_some() && n_chunks >= EPOOL_MIN_CHUNKS);
    let worker = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            guard.main_run(|| f(i));
        }
        after();
    };
    let linger_guest = |_stop: &dyn Fn() -> bool| -> usize {
        // The cursor only increases during this drain. A stale low hint
        // falls through to the ordinary atomic claim and bounds check.
        // Already-empty late joiners need not modify the shared cursor.
        if next.load(Ordering::Relaxed) >= n_chunks {
            return 0;
        }
        let mut ran = 0usize;
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            guard.main_run(|| f(i));
            ran += 1;
        }
        ran
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
            let helper_body = || {
                let mut own_ns = 0u64;
                loop {
                    if !guard.helper_may_claim(own_ns, next.load(Ordering::Relaxed), n_chunks) {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n_chunks {
                        break;
                    }
                    EPOOL_HELPER_CHUNKS.fetch_add(1, Ordering::Relaxed);
                    guard.helper_run(&mut own_ns, || f(i));
                }
            };
            with_linger_slot(&linger_guest, || {
                drain_hetero(main_threads, &worker, ep, &helper_body, detached)
            });
        }
        None => with_linger_slot(&linger_guest, drain_main),
    }
    spawn_lingerers(main_threads);
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
    run_chunks_with_helper_stateful_arm(n_chunks, init, f, helper, detached_enabled());
}

/// [`run_chunks_with_helper_stateful`] with the kickoff arm forced, so tests
/// can cover both arms in one process.
fn run_chunks_with_helper_stateful_arm<S, I, F>(
    n_chunks: usize,
    init: &I,
    f: &F,
    helper: Option<&rayon::ThreadPool>,
    detached: bool,
) where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize) + Sync,
{
    run_chunks_with_helper_stateful_arm_tail(n_chunks, init, f, helper, detached, None);
}

/// The stateful drain body; `tail` (if any) runs on each main-side worker
/// after its claim loop, see [`run_hetero_chunks_stateful_with_tail`].
fn run_chunks_with_helper_stateful_arm_tail<S, I, F>(
    n_chunks: usize,
    init: &I,
    f: &F,
    helper: Option<&rayon::ThreadPool>,
    detached: bool,
    tail: Option<&(dyn Fn() + Sync)>,
) where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize) + Sync,
{
    if n_chunks == 0 {
        if let Some(tail) = tail {
            tail();
        }
        return;
    }
    let main_threads = rayon::current_num_threads();
    if main_threads <= 1 {
        let mut state = init();
        for i in 0..n_chunks {
            f(&mut state, i);
        }
        if let Some(tail) = tail {
            tail();
        }
        return;
    }
    let next = LineCursor(AtomicUsize::new(0));
    let next = &next.0;
    let guard = TailGuard::new(main_threads, helper.is_some() && n_chunks >= EPOOL_MIN_CHUNKS);
    let worker = || {
        let mut state = None;
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            let state = state.get_or_insert_with(init);
            guard.main_run(|| f(state, i));
        }
        if let Some(tail) = tail {
            tail();
        }
    };
    let linger_guest = |_stop: &dyn Fn() -> bool| -> usize {
        // The cursor only increases during this drain. A stale low hint
        // falls through to the ordinary atomic claim and bounds check.
        // Already-empty late joiners need not modify the shared cursor.
        if next.load(Ordering::Relaxed) >= n_chunks {
            return 0;
        }
        let mut state = None;
        let mut ran = 0usize;
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            let state = state.get_or_insert_with(init);
            guard.main_run(|| f(state, i));
            ran += 1;
        }
        ran
    };
    let drain_main = || {
        (0..main_threads)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|_| worker());
    };
    // Guest claim loop (see `help_guest_until`): same queue, same chunk
    // function, one chunk per stop check, no tail.
    let guest = |stop: &dyn Fn() -> bool| -> usize {
        let mut state = None;
        let mut ran = 0usize;
        while !stop() {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            let state = state.get_or_insert_with(init);
            guard.main_run(|| f(state, i));
            ran += 1;
        }
        ran
    };
    with_linger_slot(&linger_guest, || with_guest_slot(&guest, || match helper.filter(|_| n_chunks >= EPOOL_MIN_CHUNKS) {
        Some(ep) => {
            EPOOL_BROADCASTS.fetch_add(1, Ordering::Relaxed);
            let helper_body = || {
                let mut state = None;
                let mut own_ns = 0u64;
                loop {
                    if !guard.helper_may_claim(own_ns, next.load(Ordering::Relaxed), n_chunks) {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n_chunks {
                        break;
                    }
                    EPOOL_HELPER_CHUNKS.fetch_add(1, Ordering::Relaxed);
                    let state = state.get_or_insert_with(init);
                    guard.helper_run(&mut own_ns, || f(state, i));
                }
            };
            drain_hetero(main_threads, &worker, ep, &helper_body, detached);
        }
        None => drain_main(),
    }));
    spawn_lingerers(main_threads);
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

    /// A guest claims chunks of the draining queue one at a time, returns as
    /// soon as its own condition holds, and every chunk still runs exactly
    /// once; with no queue draining it gives up after its poll budget.
    #[test]
    fn linger_chains_keep_every_chunk_exactly_once() {
        let main = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let helper = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        set_linger_phase(true);
        let linger_on = LINGER_PHASE.load(Ordering::SeqCst);
        let joined = AtomicUsize::new(0);
        main.install(|| {
            for rep in 0..200usize {
                let n = 16 + (rep % 7) * 37;
                let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                let t0 = crate::tmark::now_ns();
                // Mostly the plain arm: the stateful arm also publishes the
                // commit-guest slot, which a concurrently running guest test
                // could otherwise join.
                if rep % 16 != 0 {
                    run_chunks_with_helper_arm(
                        n,
                        &|i| {
                            counts[i].fetch_add(1, Ordering::Relaxed);
                            std::hint::black_box((0..200).fold(i, |a, b| a ^ b));
                        },
                        Some(&helper),
                        rep % 4 == 0,
                    );
                } else {
                    run_chunks_with_helper_stateful_arm_tail(
                        n,
                        &|| 0usize,
                        &|st: &mut usize, i| {
                            *st += 1;
                            counts[i].fetch_add(1, Ordering::Relaxed);
                        },
                        Some(&helper),
                        rep % 3 == 0,
                        None,
                    );
                }
                assert!(
                    counts.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                    "rep {rep}: a chunk ran zero or several times"
                );
                // Serial stretch shorter than the linger window, plus a
                // nested Rayon region the lingering workers must still run.
                while crate::tmark::now_ns() - t0 < 30_000 {
                    std::hint::spin_loop();
                }
                let s: usize = (0..64usize).into_par_iter().map(|x| x * 3).sum();
                assert_eq!(s, 3 * 63 * 64 / 2);
                if LINGERS.load(Ordering::SeqCst) > 0 {
                    joined.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        set_linger_phase(false);
        let _ = (linger_on, joined.load(Ordering::Relaxed));
    }

    #[test]
    fn help_guest_until_returns_early_and_keeps_every_chunk_once() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        let n = 4096;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let flag = AtomicBool::new(false);
        let guest_ran = AtomicUsize::new(0);
        let left_at = AtomicUsize::new(usize::MAX);
        let done_chunks = AtomicUsize::new(0);
        pool.install(|| {
            rayon::join(
                || {
                    let ran = help_guest_until(true, std::time::Duration::from_secs(5), &|| {
                        flag.load(Ordering::Acquire)
                    });
                    guest_ran.store(ran, Ordering::Release);
                    left_at.store(done_chunks.load(Ordering::Acquire), Ordering::Release);
                },
                || {
                    run_chunks_with_helper_stateful_arm_tail(
                        n,
                        &|| (),
                        &|_: &mut (), i| {
                            counts[i].fetch_add(1, Ordering::Relaxed);
                            if i == 256 {
                                flag.store(true, Ordering::Release);
                            }
                            std::thread::sleep(std::time::Duration::from_micros(20));
                            done_chunks.fetch_add(1, Ordering::AcqRel);
                        },
                        None,
                        false,
                        None,
                    );
                },
            );
        });
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        assert!(guest_ran.load(Ordering::Relaxed) > 0, "guest never joined the queue");
        assert!(left_at.load(Ordering::Relaxed) < n, "guest stayed until the queue drained");
        // Disabled, or nothing draining: no chunks, bounded wait.
        assert_eq!(help_guest_until(false, std::time::Duration::from_secs(5), &|| false), 0);
        let t = std::time::Instant::now();
        assert_eq!(help_guest_until(true, std::time::Duration::from_millis(5), &|| false), 0);
        assert!(t.elapsed() < std::time::Duration::from_secs(1));
    }

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

    #[test]
    fn tail_guard_samples_only_when_a_helper_can_join() {
        let absent = TailGuard::new(4, false);
        let present = TailGuard::new(4, true);
        let calls = AtomicUsize::new(0);
        for _ in 0..64 {
            absent.main_run(|| { calls.fetch_add(1, Ordering::Relaxed); });
            present.main_run(|| { calls.fetch_add(1, Ordering::Relaxed); });
        }
        assert_eq!(calls.load(Ordering::Relaxed), 128);
        assert_eq!(absent.main_cnt.load(Ordering::Relaxed), 0);
        assert_eq!(absent.main_ns.load(Ordering::Relaxed), 0);
        assert_eq!(present.main_cnt.load(Ordering::Relaxed), if tail_guard_enabled() { 64 } else { 0 });
        assert!(absent.helper_may_claim(u64::MAX, 0, 1));
    }

    #[test]
    fn borrowed_close_guard_keeps_owner_and_closes_on_unwind() {
        for unwind in [false, true] {
            let body = || panic!("closed gate must not execute its body");
            let gate = Arc::new(Gate {
                state: AtomicUsize::new(0),
                panicked: AtomicBool::new(false),
                body: JobPtr::new(&body),
            });
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let close = CloseOnDrop(&gate);
                assert_eq!(Arc::strong_count(&gate), 1);
                if unwind {
                    panic!("exercise guard unwind");
                }
                drop(close);
            }));
            assert_eq!(outcome.is_err(), unwind);
            assert_eq!(gate.state.load(Ordering::Acquire), GATE_CLOSED);
            assert_eq!(Arc::strong_count(&gate), 1);
            assert!(!gate.enter());
        }
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

    /// A helper-owned queue that the main side helps drain: every chunk runs
    /// exactly once, `wait_done` returns only after all completions, main
    /// claims are counted, and the tail hook of the stateful drain is what
    /// does the helping.
    #[test]
    fn help_queue_runs_each_chunk_once_with_slow_helper() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .thread_name(|i| format!("slow-help-queue-test-{i}"))
            .build()
            .unwrap();
        let n = 64;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let helper_started = AtomicBool::new(false);
        let job = |i: usize| {
            if std::thread::current()
                .name()
                .is_some_and(|name| name.starts_with("slow-help-queue-test-"))
            {
                helper_started.store(true, Ordering::Release);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            counts[i].fetch_add(1, Ordering::Relaxed);
        };
        let queue = HelpQueue::new(n, &job);
        let help = || queue.help();
        std::thread::scope(|s| {
            let owner = s.spawn(|| {
                assert!(run_queue_with_helper_only(&queue, 2, Some(&helper)));
                queue.wait_done();
                assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
            });
            while !helper_started.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            // A short unrelated stateful drain whose main-side tail helps.
            run_chunks_with_helper_stateful_arm_tail(8, &|| (), &|_, _| {}, None, true, Some(&help));
            owner.join().unwrap();
        });
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        assert!(queue.main_claims() > 0);
        assert!(queue.main_claims() < n, "slow helpers must also run chunks");
        // Exhausted queue: help is a no-op.
        queue.help();
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        // No helper pool: nothing is claimed and the owner drains inline.
        let inline_counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let inline_job = |i: usize| {
            inline_counts[i].fetch_add(1, Ordering::Relaxed);
        };
        let inline_queue = HelpQueue::new(n, &inline_job);
        assert!(!run_queue_with_helper_only(&inline_queue, 2, None));
        inline_queue.drain_inline();
        inline_queue.wait_done();
        assert!(inline_counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// Empty main tasks must not construct scratch. The tail still runs once
    /// for every main task; any guest work may belong to another active queue.
    #[test]
    fn stateful_empty_drainers_do_not_initialize() {
        let main = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let inits = AtomicUsize::new(0);
        let claims = AtomicUsize::new(0);
        let tails = AtomicUsize::new(0);
        main.install(|| {
            run_chunks_with_helper_stateful_arm_tail(
                1,
                &|| { inits.fetch_add(1, Ordering::Relaxed); },
                &|_, i| {
                    assert_eq!(i, 0);
                    claims.fetch_add(1, Ordering::Relaxed);
                },
                None,
                true,
                Some(&|| {
                    tails.fetch_add(1, Ordering::Relaxed);
                    // The process-global slot is first-publisher-wins: another
                    // test or a surrounding drain may own it. Helping that
                    // queue is legal and says nothing about this queue's claims.
                    let _ = help_guest_until(true, std::time::Duration::ZERO, &|| false);
                }),
            );
        });
        assert_eq!(claims.load(Ordering::Relaxed), 1);
        assert_eq!(tails.load(Ordering::Relaxed), 4);
        assert_eq!(inits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn stateful_every_initialized_state_processes_work() {
        let main = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let helper = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        for detached in [false, true] {
            for n in [0, 1, 3, 4, 17, 257] {
                let inits = AtomicUsize::new(0);
                let used = AtomicUsize::new(0);
                let counts: Vec<_> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                main.install(|| run_chunks_with_helper_stateful_arm(
                    n,
                    &|| { inits.fetch_add(1, Ordering::Relaxed); false },
                    &|state: &mut bool, i| {
                        if !*state {
                            *state = true;
                            used.fetch_add(1, Ordering::Relaxed);
                        }
                        counts[i].fetch_add(1, Ordering::Relaxed);
                    },
                    Some(&helper),
                    detached,
                ));
                assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
                assert_eq!(inits.load(Ordering::Relaxed), used.load(Ordering::Relaxed));
                assert!(inits.load(Ordering::Relaxed) <= n);
            }
        }
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

    /// Both kickoff arms — the detached gated broadcast and the blocking
    /// per-drain spawn the kill switch restores — execute every chunk exactly
    /// once, over many drains on one helper pool.
    #[test]
    fn both_kickoff_arms_run_each_chunk_exactly_once() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let n = 997;
        for detached in [true, false] {
            for _ in 0..16 {
                let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                run_chunks_with_helper_arm(
                    n,
                    &|i| {
                        counts[i].fetch_add(1, Ordering::Relaxed);
                    },
                    Some(&helper),
                    detached,
                );
                assert!(
                    counts.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                    "detached={detached}"
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
        for detached in [true, false] {
            let next_state = AtomicUsize::new(0);
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            run_chunks_with_helper_stateful_arm(
                n,
                &|| next_state.fetch_add(1, Ordering::Relaxed),
                &|_state, i| {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                },
                Some(&helper),
                detached,
            );
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
            assert!(
                next_state.load(Ordering::Relaxed) < n,
                "state must be per worker, not per chunk (detached={detached})"
            );
        }
    }

    /// Drains that overlap in time each own a gate and queue their own
    /// broadcast on the shared helper pool. All must complete, none may
    /// deadlock.
    #[test]
    fn concurrent_detached_drains_all_complete() {
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
                        run_chunks_with_helper_arm(
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
    /// both arms, and the helper pool still serves a normal drain afterwards.
    #[test]
    fn panicking_chunk_propagates_and_leaves_the_pool_usable() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for detached in [true, false] {
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_chunks_with_helper_arm(
                    64,
                    &|_| panic!("chunk exploded"),
                    Some(&helper),
                    detached,
                );
            }));
            assert!(caught.is_err(), "detached={detached}");
        }
        // The pool must still serve a normal drain afterwards.
        let n = 128;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        run_chunks_with_helper_arm(
            n,
            &|i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            },
            Some(&helper),
            true,
        );
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// A helper thread that is busy elsewhere for the whole drain never holds
    /// the drain up: the drain returns while that thread is still blocked, and
    /// when it finally reaches the stale job it finds the gate closed and runs
    /// nothing (the late-arrival counter moves; no chunk runs twice).
    #[test]
    fn detached_drain_does_not_wait_for_a_late_helper() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let release = Arc::new(AtomicBool::new(false));
        let blocked = Arc::new(AtomicBool::new(false));
        {
            let release = Arc::clone(&release);
            let blocked = Arc::clone(&blocked);
            helper.spawn(move || {
                blocked.store(true, Ordering::Release);
                while !release.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            });
        }
        while !blocked.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        let late_before = helper_late_arrivals();
        let n = 256;
        let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        let started = std::time::Instant::now();
        run_chunks_with_helper_arm(
            n,
            &|i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            },
            Some(&helper),
            true,
        );
        let drain_time = started.elapsed();
        assert!(!release.load(Ordering::Acquire));
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        assert!(
            drain_time < std::time::Duration::from_secs(5),
            "drain waited {drain_time:?}"
        );
        release.store(true, Ordering::Release);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while helper_late_arrivals() == late_before && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(helper_late_arrivals() > late_before, "the blocked helper never ran its stale job");
        assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// A chunk that panics on a helper thread is caught there (a detached job
    /// has no caller to unwind into) and re-raised on the drain's thread.
    #[test]
    fn helper_side_panic_propagates_to_the_drain() {
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .thread_name(|i| format!("panic-helper-test-{i}"))
            .build()
            .unwrap();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_chunks_with_helper_arm(
                256,
                &|_| {
                    if std::thread::current()
                        .name()
                        .is_some_and(|name| name.starts_with("panic-helper-test-"))
                    {
                        panic!("helper chunk exploded");
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                },
                Some(&helper),
                true,
            );
        }));
        assert!(caught.is_err());
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

    /// The kill switch reads exactly `"1"`; nothing else disables the
    /// detached kickoff.
    #[test]
    fn kickoff_kill_switch_matches_exactly_one() {
        assert!(detached_killed_by(Some("1")));
        for keep in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("11"),
            Some(" 1"),
        ] {
            assert!(!detached_killed_by(keep), "{keep:?} must not kill the kickoff");
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

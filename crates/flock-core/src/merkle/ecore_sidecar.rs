//! Persistent Apple-efficiency-core SHA sidecar.
//!
//! Four background-QoS workers and the Rayon performance workers drain the
//! same atomic tile cursor. Jobs describe contiguous equal-length SHA inputs,
//! so the queue can serve both Merkle leaves and internal child pairs without
//! owning or copying either buffer.

use super::{Hash, sha256x4};
use core::ffi::{c_char, c_void};
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};
use std::sync::{Arc, Barrier, Condvar, Mutex, OnceLock};
use std::time::Instant;

const N_ECORE_WORKERS: usize = 4;
const BACKGROUND_QOS_CLASS: u32 = 0x09;
const USER_INITIATED_QOS_CLASS: u32 = 0x19;
const TARGET_COMPRESSIONS_PER_TILE: usize = 1024;
const P_ONLY_RESERVE_TILES_PER_WORKER: usize = 12;

unsafe extern "C" {
    fn pthread_self() -> *mut c_void;
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    fn pthread_get_qos_class_np(
        thread: *mut c_void,
        qos_class: *mut u32,
        relative_priority: *mut i32,
    ) -> i32;
    fn pthread_override_qos_class_start_np(
        thread: *mut c_void,
        qos_class: u32,
        relative_priority: i32,
    ) -> *mut c_void;
    fn pthread_override_qos_class_end_np(qos_override: *mut c_void) -> i32;
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RunStats {
    pub first_ecore_claim_ns: Option<u64>,
    pub ecore_tail_ns: u64,
    pub ecore_tiles: usize,
    pub pcore_tiles: usize,
    pub tile_quads: usize,
    pub completion_owner_is_ecore: bool,
    pub worker_first_claim_ns: [Option<u64>; N_ECORE_WORKERS],
    pub worker_last_finish_ns: [Option<u64>; N_ECORE_WORKERS],
    pub worker_tiles: [usize; N_ECORE_WORKERS],
    pub qos_override_attempted: bool,
    pub qos_override_started: usize,
    pub qos_override_start_failures: usize,
    pub qos_override_end_failures: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct QosDiagnostics {
    pub classes: [u32; N_ECORE_WORKERS],
    pub set_results: [i32; N_ECORE_WORKERS],
    pub get_results: [i32; N_ECORE_WORKERS],
    pub relative_priorities: [i32; N_ECORE_WORKERS],
    pub override_probe_start_ok: [bool; N_ECORE_WORKERS],
    pub override_probe_end_results: [i32; N_ECORE_WORKERS],
}

impl QosDiagnostics {
    fn is_valid_background(self) -> bool {
        self.classes == [BACKGROUND_QOS_CLASS; N_ECORE_WORKERS]
            && self.set_results == [0; N_ECORE_WORKERS]
            && self.get_results == [0; N_ECORE_WORKERS]
            && self
                .relative_priorities
                .iter()
                .all(|&relative_priority| relative_priority <= 0)
            && self.override_probe_start_ok == [true; N_ECORE_WORKERS]
            && self.override_probe_end_results == [0; N_ECORE_WORKERS]
    }
}

#[cfg(test)]
#[derive(Default)]
struct DelayedClaimState {
    claimed: bool,
    override_started: bool,
    timed_out: bool,
}

/// Deterministic protocol hook: hold the first helper immediately after it
/// owns a tile, and hold the P drainers until that ownership is established.
/// The production post-P-drain override path releases the helper. This tests
/// the otherwise timing-sensitive claim -> rescue -> completion ordering
/// without relying on scheduler starvation to occur by chance.
#[cfg(test)]
pub(super) struct DelayedClaimHook {
    state: Mutex<DelayedClaimState>,
    changed: Condvar,
}

#[cfg(test)]
impl DelayedClaimHook {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(DelayedClaimState::default()),
            changed: Condvar::new(),
        })
    }

    fn wait_for_first_claim(&self) {
        let deadline = std::time::Duration::from_secs(30);
        let state = self.state.lock().unwrap();
        let (mut state, timeout) = self
            .changed
            .wait_timeout_while(state, deadline, |state| !state.claimed && !state.timed_out)
            .unwrap();
        if timeout.timed_out() && !state.claimed {
            state.timed_out = true;
            self.changed.notify_all();
        }
    }

    fn delay_first_claim(&self) {
        let mut state = self.state.lock().unwrap();
        if state.claimed {
            return;
        }
        state.claimed = true;
        self.changed.notify_all();
        let deadline = std::time::Duration::from_secs(30);
        let (mut state, timeout) = self
            .changed
            .wait_timeout_while(state, deadline, |state| {
                !state.override_started && !state.timed_out
            })
            .unwrap();
        if timeout.timed_out() && !state.override_started {
            state.timed_out = true;
            self.changed.notify_all();
        }
    }

    fn note_override_started(&self) {
        let mut state = self.state.lock().unwrap();
        state.override_started = true;
        self.changed.notify_all();
    }

    pub(super) fn snapshot(&self) -> (bool, bool, bool) {
        let state = self.state.lock().unwrap();
        (state.claimed, state.override_started, state.timed_out)
    }
}

#[cfg(test)]
#[derive(Default)]
struct ConcurrentSubmitState {
    owner_acquired: bool,
    release_owner: bool,
    timed_out: bool,
}

/// Deterministic concurrent-submission hook. The first caller pauses while it
/// owns `Sidecar::submit`, allowing the test's second caller to prove that it
/// takes the immediate legacy fallback instead of waiting for, or reusing, the
/// first caller's generation.
#[cfg(test)]
pub(super) struct ConcurrentSubmitHook {
    state: Mutex<ConcurrentSubmitState>,
    changed: Condvar,
}

#[cfg(test)]
impl ConcurrentSubmitHook {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(ConcurrentSubmitState::default()),
            changed: Condvar::new(),
        })
    }

    fn hold_submit_owner(&self) {
        let mut state = self.state.lock().unwrap();
        state.owner_acquired = true;
        self.changed.notify_all();
        let deadline = std::time::Duration::from_secs(30);
        let (mut state, timeout) = self
            .changed
            .wait_timeout_while(state, deadline, |state| {
                !state.release_owner && !state.timed_out
            })
            .unwrap();
        if timeout.timed_out() && !state.release_owner {
            state.timed_out = true;
            state.release_owner = true;
            self.changed.notify_all();
        }
    }

    pub(super) fn wait_for_owner(&self) -> bool {
        let deadline = std::time::Duration::from_secs(30);
        let state = self.state.lock().unwrap();
        let (mut state, timeout) = self
            .changed
            .wait_timeout_while(state, deadline, |state| {
                !state.owner_acquired && !state.timed_out
            })
            .unwrap();
        if timeout.timed_out() && !state.owner_acquired {
            state.timed_out = true;
            state.release_owner = true;
            self.changed.notify_all();
        }
        state.owner_acquired && !state.timed_out
    }

    pub(super) fn release_owner(&self) {
        let mut state = self.state.lock().unwrap();
        state.release_owner = true;
        self.changed.notify_all();
    }

    pub(super) fn snapshot(&self) -> (bool, bool, bool) {
        let state = self.state.lock().unwrap();
        (state.owner_acquired, state.release_owner, state.timed_out)
    }
}

struct Job {
    input_addr: usize,
    output_addr: usize,
    message_len: usize,
    n_quads: usize,
    tile_quads: usize,
    n_tiles: usize,
    ecore_tile_limit: usize,
    next_tile: AtomicUsize,
    remaining_tiles: AtomicUsize,
    first_ecore_claim_ns: AtomicU64,
    ecore_tiles: AtomicUsize,
    pcore_tiles: AtomicUsize,
    completion_owner: AtomicUsize,
    completed: AtomicBool,
    worker_first_claim_ns: [AtomicU64; N_ECORE_WORKERS],
    worker_last_finish_ns: [AtomicU64; N_ECORE_WORKERS],
    worker_tiles: [AtomicUsize; N_ECORE_WORKERS],
    published: OnceLock<Instant>,
    done_lock: Mutex<()>,
    done: Condvar,
    #[cfg(test)]
    delayed_claim_hook: Option<Arc<DelayedClaimHook>>,
}

impl Job {
    fn new(
        input: &[u8],
        message_len: usize,
        output: &mut [Hash],
        #[cfg(test)] delayed_claim_hook: Option<Arc<DelayedClaimHook>>,
    ) -> Self {
        assert!(message_len > 0);
        assert_eq!(output.len() % 4, 0);
        assert_eq!(input.len(), output.len() * message_len);

        let n_quads = output.len() / 4;
        let compressions_per_message = (message_len + 9).div_ceil(64);
        let compressions_per_quad = 4 * compressions_per_message;
        let tile_quads = (TARGET_COMPRESSIONS_PER_TILE / compressions_per_quad)
            .clamp(1, 64)
            .min(n_quads.max(1));
        let n_tiles = n_quads.div_ceil(tile_quads);
        let p_only_reserve = P_ONLY_RESERVE_TILES_PER_WORKER * rayon::current_num_threads().max(1);
        let ecore_tile_limit = n_tiles.saturating_sub(p_only_reserve);

        Self {
            input_addr: input.as_ptr() as usize,
            output_addr: output.as_mut_ptr() as usize,
            message_len,
            n_quads,
            tile_quads,
            n_tiles,
            ecore_tile_limit,
            next_tile: AtomicUsize::new(0),
            remaining_tiles: AtomicUsize::new(n_tiles),
            first_ecore_claim_ns: AtomicU64::new(0),
            ecore_tiles: AtomicUsize::new(0),
            pcore_tiles: AtomicUsize::new(0),
            completion_owner: AtomicUsize::new(0),
            completed: AtomicBool::new(false),
            worker_first_claim_ns: std::array::from_fn(|_| AtomicU64::new(0)),
            worker_last_finish_ns: std::array::from_fn(|_| AtomicU64::new(0)),
            worker_tiles: std::array::from_fn(|_| AtomicUsize::new(0)),
            published: OnceLock::new(),
            done_lock: Mutex::new(()),
            done: Condvar::new(),
            #[cfg(test)]
            delayed_claim_hook,
        }
    }

    #[inline]
    fn drain(&self, ecore_worker: Option<usize>) {
        let is_ecore = ecore_worker.is_some();
        #[cfg(test)]
        if !is_ecore && let Some(hook) = &self.delayed_claim_hook {
            hook.wait_for_first_claim();
        }
        loop {
            let tile = if is_ecore {
                self.claim_ecore_tile()
            } else {
                self.claim_pcore_tile()
            };
            let Some(tile) = tile else {
                break;
            };

            #[cfg(test)]
            if is_ecore && let Some(hook) = &self.delayed_claim_hook {
                hook.delay_first_claim();
            }

            if is_ecore {
                let worker_id = ecore_worker.expect("E-core worker id");
                if self.first_ecore_claim_ns.load(Ordering::Relaxed) == 0 {
                    let claimed_ns = self
                        .published
                        .get()
                        .expect("published before E-core claim")
                        .elapsed()
                        .as_nanos()
                        .max(1) as u64;
                    let _ = self.first_ecore_claim_ns.compare_exchange(
                        0,
                        claimed_ns,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                }
                if self.worker_first_claim_ns[worker_id].load(Ordering::Relaxed) == 0 {
                    let claimed_ns = self
                        .published
                        .get()
                        .expect("published before E-core claim")
                        .elapsed()
                        .as_nanos()
                        .max(1) as u64;
                    let _ = self.worker_first_claim_ns[worker_id].compare_exchange(
                        0,
                        claimed_ns,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                }
                self.ecore_tiles.fetch_add(1, Ordering::Relaxed);
                self.worker_tiles[worker_id].fetch_add(1, Ordering::Relaxed);
            } else {
                self.pcore_tiles.fetch_add(1, Ordering::Relaxed);
            }

            self.process_tile(tile);
            if let Some(worker_id) = ecore_worker {
                let finished_ns = self
                    .published
                    .get()
                    .expect("published before E-core finish")
                    .elapsed()
                    .as_nanos()
                    .max(1) as u64;
                self.worker_last_finish_ns[worker_id].store(finished_ns, Ordering::Relaxed);
            }
            if self.remaining_tiles.fetch_sub(1, Ordering::Release) == 1 {
                // Standard last-reference handoff: the final Release RMW
                // observes the counter chain, and this Acquire fence imports
                // every earlier worker's output writes before `completed`
                // republishes them to the submitting thread. An Acquire load
                // performed only by the caller would otherwise synchronize
                // with the final worker, not necessarily every predecessor.
                std::sync::atomic::fence(Ordering::Acquire);
                let _done_guard = self.done_lock.lock().unwrap();
                self.completion_owner
                    .store(if is_ecore { 2 } else { 1 }, Ordering::Relaxed);
                self.completed.store(true, Ordering::Release);
                self.done.notify_one();
            }
        }
    }

    #[inline]
    fn claim_ecore_tile(&self) -> Option<usize> {
        let mut current = self.next_tile.load(Ordering::Relaxed);
        loop {
            if current >= self.ecore_tile_limit {
                return None;
            }
            match self.next_tile.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(current),
                Err(observed) => current = observed,
            }
        }
    }

    #[inline]
    fn claim_pcore_tile(&self) -> Option<usize> {
        let tile = self.next_tile.fetch_add(1, Ordering::Relaxed);
        (tile < self.n_tiles).then_some(tile)
    }

    #[inline]
    fn process_tile(&self, tile: usize) {
        let quad_start = tile * self.tile_quads;
        let quad_end = (quad_start + self.tile_quads).min(self.n_quads);
        for quad in quad_start..quad_end {
            let message_start = 4 * quad * self.message_len;
            let output_start = 4 * quad;
            // SAFETY:
            // - `Job::new` checked input/output lengths and four-way shape;
            // - the atomic cursor gives every tile, hence every output quad,
            //   to exactly one worker;
            // - the submitting thread keeps both borrowed buffers alive and
            //   immutable/mutably exclusive until `remaining_tiles == 0`;
            // - every worker publishes through a Release decrement; the final
            //   worker's Acquire fence imports that chain and republishes it
            //   through the completed Release/Acquire handoff before the
            //   submitting thread releases its output borrow.
            unsafe {
                let input = self.input_addr as *const u8;
                let output = self.output_addr as *mut Hash;
                let m0 = core::slice::from_raw_parts(input.add(message_start), self.message_len);
                let m1 = core::slice::from_raw_parts(
                    input.add(message_start + self.message_len),
                    self.message_len,
                );
                let m2 = core::slice::from_raw_parts(
                    input.add(message_start + 2 * self.message_len),
                    self.message_len,
                );
                let m3 = core::slice::from_raw_parts(
                    input.add(message_start + 3 * self.message_len),
                    self.message_len,
                );
                let outs = core::slice::from_raw_parts_mut(output.add(output_start), 4);
                sha256x4::hash4_equal_len([m0, m1, m2, m3], outs);
            }
        }
    }
}

struct Control {
    generation: u64,
    current: Option<Arc<Job>>,
}

struct Sidecar {
    control: Mutex<Control>,
    work_available: Condvar,
    submit: Mutex<()>,
    worker_threads: [AtomicPtr<c_void>; N_ECORE_WORKERS],
    qos_classes: [AtomicU32; N_ECORE_WORKERS],
    qos_set_results: [AtomicI32; N_ECORE_WORKERS],
    qos_get_results: [AtomicI32; N_ECORE_WORKERS],
    qos_relative_priorities: [AtomicI32; N_ECORE_WORKERS],
    override_probe_start_ok: [AtomicBool; N_ECORE_WORKERS],
    override_probe_end_results: [AtomicI32; N_ECORE_WORKERS],
    healthy: AtomicBool,
    submissions: AtomicU64,
    #[cfg(test)]
    delayed_claim_hook: Mutex<Option<Arc<DelayedClaimHook>>>,
    #[cfg(test)]
    concurrent_submit_hook: Mutex<Option<Arc<ConcurrentSubmitHook>>>,
}

/// Own every successful override token until the dependency is satisfied.
/// `finish` records API failures on the normal path; `Drop` is the last-resort
/// leak guard if later code unwinds after starting an override.
struct QosOverrideGuard<'a> {
    sidecar: &'a Sidecar,
    tokens: [*mut c_void; N_ECORE_WORKERS],
}

impl QosOverrideGuard<'_> {
    fn end_all(&mut self) -> ([i32; N_ECORE_WORKERS], usize) {
        let mut results = [i32::MIN; N_ECORE_WORKERS];
        let mut failures = 0usize;
        for (worker_id, token) in self.tokens.iter_mut().enumerate() {
            let token = core::mem::replace(token, core::ptr::null_mut());
            if token.is_null() {
                continue;
            }
            // SAFETY: every non-null token was returned exactly once by
            // `pthread_override_qos_class_start_np`; replacing it with null
            // before the call makes this idempotent even if Drop follows.
            let result = unsafe { pthread_override_qos_class_end_np(token) };
            results[worker_id] = result;
            if result != 0 {
                failures += 1;
            }
        }
        if failures != 0 {
            self.sidecar.healthy.store(false, Ordering::Release);
        }
        (results, failures)
    }

    fn finish(mut self) -> ([i32; N_ECORE_WORKERS], usize) {
        self.end_all()
    }
}

impl Drop for QosOverrideGuard<'_> {
    fn drop(&mut self) {
        let _ = self.end_all();
    }
}

impl Sidecar {
    fn new() -> Arc<Self> {
        let sidecar = Arc::new(Self {
            control: Mutex::new(Control {
                generation: 0,
                current: None,
            }),
            work_available: Condvar::new(),
            submit: Mutex::new(()),
            worker_threads: std::array::from_fn(|_| AtomicPtr::new(core::ptr::null_mut())),
            qos_classes: std::array::from_fn(|_| AtomicU32::new(0)),
            qos_set_results: std::array::from_fn(|_| AtomicI32::new(i32::MIN)),
            qos_get_results: std::array::from_fn(|_| AtomicI32::new(i32::MIN)),
            qos_relative_priorities: std::array::from_fn(|_| AtomicI32::new(i32::MIN)),
            override_probe_start_ok: std::array::from_fn(|_| AtomicBool::new(false)),
            override_probe_end_results: std::array::from_fn(|_| AtomicI32::new(i32::MIN)),
            healthy: AtomicBool::new(true),
            submissions: AtomicU64::new(0),
            #[cfg(test)]
            delayed_claim_hook: Mutex::new(None),
            #[cfg(test)]
            concurrent_submit_hook: Mutex::new(None),
        });
        let handles_ready = Arc::new(Barrier::new(N_ECORE_WORKERS + 1));
        let overrides_started = Arc::new(Barrier::new(N_ECORE_WORKERS + 1));
        let workers_ready = Arc::new(Barrier::new(N_ECORE_WORKERS + 1));

        for worker_id in 0..N_ECORE_WORKERS {
            let worker_sidecar = Arc::clone(&sidecar);
            let worker_handles_ready = Arc::clone(&handles_ready);
            let worker_overrides_started = Arc::clone(&overrides_started);
            let worker_ready = Arc::clone(&workers_ready);
            std::thread::Builder::new()
                .name(format!("flock-ecore-sha-{worker_id}"))
                .spawn(move || {
                    // Publish the stable process-lifetime handle while this new
                    // thread still has inherited/default QoS. The initializer
                    // can then establish a rescue override before asking it to
                    // enter a potentially starvable BACKGROUND class.
                    // SAFETY: querying the current live pthread returns its
                    // opaque process-local handle and has no preconditions.
                    let thread = unsafe { pthread_self() };
                    worker_sidecar.worker_threads[worker_id].store(thread, Ordering::Release);
                    worker_handles_ready.wait();
                    worker_overrides_started.wait();

                    // If even one capability override failed, do not enter the
                    // low-QoS region: all workers must reach the ready barrier
                    // so initialization can fail closed instead of hanging.
                    if worker_sidecar.healthy.load(Ordering::Acquire) {
                        let qos = set_background_qos_and_read_back(thread);
                        worker_sidecar.qos_classes[worker_id].store(qos.class, Ordering::Release);
                        worker_sidecar.qos_set_results[worker_id]
                            .store(qos.set_result, Ordering::Release);
                        worker_sidecar.qos_get_results[worker_id]
                            .store(qos.get_result, Ordering::Release);
                        worker_sidecar.qos_relative_priorities[worker_id]
                            .store(qos.relative_priority, Ordering::Release);

                        // Touch the complete SHA-x4 path while the startup
                        // USER_INITIATED override is still active.
                        let messages = [[worker_id as u8; 64]; 4];
                        let mut output = [[0u8; 32]; 4];
                        sha256x4::hash4_equal_len(
                            [&messages[0], &messages[1], &messages[2], &messages[3]],
                            &mut output,
                        );
                        std::hint::black_box(output);
                    }
                    worker_ready.wait();
                    // This is the sole unwind boundary on the process-lifetime
                    // helper path. A panic after `Job::claim_ecore_tile` may
                    // leave a partially written output tile whose ownership
                    // cannot safely be transferred or recomputed while the
                    // caller's raw buffers remain published. Letting this
                    // detached thread disappear would also strand
                    // `remaining_tiles` and hang the caller forever. Recovery
                    // is therefore unsafe: fail the process immediately.
                    let _panic =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> () {
                            worker_sidecar.worker_loop(worker_id)
                        }));
                    std::process::abort();
                })
                .expect("spawn persistent E-core SHA worker");
        }

        handles_ready.wait();
        let (probe_overrides, _, _) = sidecar.start_worker_overrides();
        for worker_id in 0..N_ECORE_WORKERS {
            sidecar.override_probe_start_ok[worker_id].store(
                !probe_overrides.tokens[worker_id].is_null(),
                Ordering::Release,
            );
        }
        overrides_started.wait();
        workers_ready.wait();
        let (probe_end_results, _) = probe_overrides.finish();
        for (worker_id, result) in probe_end_results.into_iter().enumerate() {
            sidecar.override_probe_end_results[worker_id].store(result, Ordering::Release);
        }
        sidecar
    }

    fn start_worker_overrides(&self) -> (QosOverrideGuard<'_>, usize, usize) {
        let mut guard = QosOverrideGuard {
            sidecar: self,
            tokens: [core::ptr::null_mut(); N_ECORE_WORKERS],
        };
        let mut started = 0usize;
        let mut failures = 0usize;
        for worker_id in 0..N_ECORE_WORKERS {
            let thread = self.worker_threads[worker_id].load(Ordering::Acquire);
            // SAFETY: handles refer to process-lifetime persistent pthreads.
            // At startup no job exists; at runtime the caller holds `submit`,
            // so no subsequent generation can begin before tokens are ended.
            let token = if thread.is_null() {
                core::ptr::null_mut()
            } else {
                unsafe { pthread_override_qos_class_start_np(thread, USER_INITIATED_QOS_CLASS, 0) }
            };
            if token.is_null() {
                failures += 1;
            } else {
                guard.tokens[worker_id] = token;
                started += 1;
            }
        }
        if failures != 0 {
            self.healthy.store(false, Ordering::Release);
        }
        (guard, started, failures)
    }

    fn worker_loop(&self, worker_id: usize) -> ! {
        let mut seen_generation = 0u64;
        loop {
            let job = {
                let mut control = self.control.lock().unwrap();
                while control.generation == seen_generation {
                    control = self.work_available.wait(control).unwrap();
                }
                seen_generation = control.generation;
                control.current.clone()
            };
            if let Some(job) = job {
                job.drain(Some(worker_id));
            }
        }
    }

    fn run(&self, input: &[u8], message_len: usize, output: &mut [Hash]) -> Option<RunStats> {
        if !self.healthy.load(Ordering::Acquire) {
            return None;
        }
        // Concurrent Merkle callers retain the legacy Rayon path instead of
        // serializing behind this process-global sidecar.
        let Ok(_submit) = self.submit.try_lock() else {
            return None;
        };
        #[cfg(test)]
        if let Some(hook) = self.concurrent_submit_hook.lock().unwrap().take() {
            hook.hold_submit_owner();
        }
        #[cfg(test)]
        let delayed_claim_hook = self.delayed_claim_hook.lock().unwrap().take();
        let job = Arc::new(Job::new(
            input,
            message_len,
            output,
            #[cfg(test)]
            delayed_claim_hook,
        ));
        self.submissions.fetch_add(1, Ordering::Relaxed);

        {
            let mut control = self.control.lock().unwrap();
            debug_assert!(control.current.is_none());
            job.published
                .set(Instant::now())
                .expect("job published exactly once");
            control.current = Some(Arc::clone(&job));
            control.generation = control.generation.wrapping_add(1);
        }
        self.work_available.notify_all();

        // The deterministic delayed-claim test must not depend on a saturated
        // host scheduling BACKGROUND work before its P drainers intentionally
        // pause. A test-only override gets a helper to the claim point; the
        // production post-drain override below is still what releases it.
        #[cfg(test)]
        let delayed_claim_startup_override = job
            .delayed_claim_hook
            .as_ref()
            .map(|_| self.start_worker_overrides().0);

        // Give each Rayon worker one long-lived drainer. The P and E workers
        // then share only `next_tile`; no static fraction or suffix exists.
        rayon::broadcast(|_| job.drain(None));

        let pcore_done = Instant::now();
        for _ in 0..256 {
            if job.completed.load(Ordering::Acquire) {
                break;
            }
            std::hint::spin_loop();
        }
        // A false predicate after every P drainer has returned means one or
        // more helpers irrevocably own the only outstanding tiles. Temporarily
        // promote all four process-lifetime targets: selecting individual
        // workers from relaxed activity flags would introduce a missed-owner
        // race, while idle targets are harmless and are demoted moments later.
        let qos_override_attempted = !job.completed.load(Ordering::Acquire);
        let (qos_overrides, qos_override_started, qos_override_start_failures) =
            if qos_override_attempted {
                let (guard, started, failures) = self.start_worker_overrides();
                (Some(guard), started, failures)
            } else {
                (None, 0, 0)
            };
        #[cfg(test)]
        if qos_override_attempted && let Some(hook) = &job.delayed_claim_hook {
            hook.note_override_started();
        }
        // Always acquire the completion mutex, even if the bounded spin saw
        // completion. The final finisher publishes both owner and predicate
        // under this mutex, preventing a zero-counter/unstored-owner race.
        let mut done_guard = job.done_lock.lock().unwrap();
        while !job.completed.load(Ordering::Acquire) {
            done_guard = job.done.wait(done_guard).unwrap();
        }
        // The final worker's Acquire fence imports every tile's Release RMW,
        // then its completed Release publishes all disjoint output writes to
        // the Acquire predicate above. The counter load is only an invariant
        // check; it is not the publication mechanism.
        debug_assert_eq!(job.remaining_tiles.load(Ordering::Acquire), 0);
        drop(done_guard);
        // The dependency is satisfied only after the completed Acquire above.
        // End every successful token before releasing `submit`, so an override
        // can never bleed into a later sidecar generation.
        let qos_override_end_failures = qos_overrides
            .map(QosOverrideGuard::finish)
            .map_or(0, |(_, failures)| failures);
        #[cfg(test)]
        if let Some(guard) = delayed_claim_startup_override {
            let _ = guard.finish();
        }
        let ecore_tail_ns = pcore_done.elapsed().as_nanos() as u64;

        {
            let mut control = self.control.lock().unwrap();
            control.current = None;
        }

        let first = job.first_ecore_claim_ns.load(Ordering::Relaxed);
        Some(RunStats {
            first_ecore_claim_ns: (first != 0).then_some(first),
            ecore_tail_ns,
            ecore_tiles: job.ecore_tiles.load(Ordering::Relaxed),
            pcore_tiles: job.pcore_tiles.load(Ordering::Relaxed),
            tile_quads: job.tile_quads,
            completion_owner_is_ecore: job.completion_owner.load(Ordering::Relaxed) == 2,
            worker_first_claim_ns: std::array::from_fn(|i| {
                let ns = job.worker_first_claim_ns[i].load(Ordering::Relaxed);
                (ns != 0).then_some(ns)
            }),
            worker_last_finish_ns: std::array::from_fn(|i| {
                let ns = job.worker_last_finish_ns[i].load(Ordering::Relaxed);
                (ns != 0).then_some(ns)
            }),
            worker_tiles: std::array::from_fn(|i| job.worker_tiles[i].load(Ordering::Relaxed)),
            qos_override_attempted,
            qos_override_started,
            qos_override_start_failures,
            qos_override_end_failures,
        })
    }
}

#[derive(Clone, Copy)]
struct Topology {
    pcore_workers: usize,
    ecore_workers: usize,
}

fn sysctl_usize(name: &'static [u8]) -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old_value: *mut c_void,
            old_len: *mut usize,
            new_value: *mut c_void,
            new_len: usize,
        ) -> i32;
    }

    debug_assert_eq!(name.last(), Some(&0));
    let mut value = 0u32;
    let mut value_len = size_of::<u32>();
    // SAFETY: `name` is static and NUL-terminated, the output points to a
    // live `u32`, and this read-only query passes no new value.
    let result = unsafe {
        sysctlbyname(
            name.as_ptr().cast(),
            (&mut value as *mut u32).cast(),
            &mut value_len,
            core::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && value_len == size_of::<u32>() && value > 0).then_some(value as usize)
}

fn topology() -> Option<Topology> {
    static TOPOLOGY: OnceLock<Option<Topology>> = OnceLock::new();
    *TOPOLOGY.get_or_init(|| {
        Some(Topology {
            pcore_workers: sysctl_usize(b"hw.perflevel0.physicalcpu\0")?,
            ecore_workers: sysctl_usize(b"hw.perflevel1.physicalcpu\0")?,
        })
    })
}

pub(super) fn pool_shape_is_supported() -> bool {
    topology().is_some_and(|topology| {
        let rayon_workers = rayon::current_num_threads();
        (2..=topology.pcore_workers).contains(&rayon_workers)
            && topology.ecore_workers >= N_ECORE_WORKERS
    })
}

#[cfg(test)]
pub(super) fn performance_core_count() -> Option<usize> {
    topology().map(|topology| topology.pcore_workers)
}

fn sidecar_slot() -> &'static OnceLock<Option<Arc<Sidecar>>> {
    static SIDECAR: OnceLock<Option<Arc<Sidecar>>> = OnceLock::new();
    &SIDECAR
}

fn initialized_sidecar() -> Option<&'static Arc<Sidecar>> {
    sidecar_slot().get().and_then(Option::as_ref)
}

fn valid_qos(sidecar: &Sidecar) -> bool {
    sidecar.healthy.load(Ordering::Acquire) && read_qos_diagnostics(sidecar).is_valid_background()
}

pub(super) fn init() -> bool {
    let sidecar = sidecar_slot()
        .get_or_init(|| {
            let topology = topology()?;
            (topology.ecore_workers >= N_ECORE_WORKERS).then(Sidecar::new)
        })
        .as_ref();
    sidecar.is_some_and(|sidecar| valid_qos(sidecar))
}

pub(super) fn run(input: &[u8], message_len: usize, output: &mut [Hash]) -> Option<RunStats> {
    if !pool_shape_is_supported() {
        return None;
    }
    let sidecar = initialized_sidecar()?;
    valid_qos(sidecar).then_some(())?;
    sidecar.run(input, message_len, output)
}

pub(super) fn qos_diagnostics() -> Option<QosDiagnostics> {
    let sidecar = initialized_sidecar()?;
    Some(read_qos_diagnostics(sidecar))
}

fn read_qos_diagnostics(sidecar: &Sidecar) -> QosDiagnostics {
    QosDiagnostics {
        classes: std::array::from_fn(|i| sidecar.qos_classes[i].load(Ordering::Acquire)),
        set_results: std::array::from_fn(|i| sidecar.qos_set_results[i].load(Ordering::Acquire)),
        get_results: std::array::from_fn(|i| sidecar.qos_get_results[i].load(Ordering::Acquire)),
        relative_priorities: std::array::from_fn(|i| {
            sidecar.qos_relative_priorities[i].load(Ordering::Acquire)
        }),
        override_probe_start_ok: std::array::from_fn(|i| {
            sidecar.override_probe_start_ok[i].load(Ordering::Acquire)
        }),
        override_probe_end_results: std::array::from_fn(|i| {
            sidecar.override_probe_end_results[i].load(Ordering::Acquire)
        }),
    }
}

#[cfg(test)]
pub(super) fn submission_count() -> u64 {
    initialized_sidecar().map_or(0, |sidecar| sidecar.submissions.load(Ordering::Relaxed))
}

#[cfg(test)]
pub(super) fn test_serial_guard() -> std::sync::MutexGuard<'static, ()> {
    static TEST_SERIAL: Mutex<()> = Mutex::new(());
    TEST_SERIAL.lock().unwrap()
}

#[cfg(test)]
pub(super) fn install_delayed_claim_hook() -> Arc<DelayedClaimHook> {
    let sidecar = initialized_sidecar().expect("sidecar initialized before installing test hook");
    let hook = DelayedClaimHook::new();
    let mut slot = sidecar.delayed_claim_hook.lock().unwrap();
    assert!(slot.is_none(), "only one delayed-claim hook may be pending");
    *slot = Some(Arc::clone(&hook));
    hook
}

#[cfg(test)]
pub(super) fn install_concurrent_submit_hook() -> Arc<ConcurrentSubmitHook> {
    let sidecar = initialized_sidecar().expect("sidecar initialized before installing test hook");
    let hook = ConcurrentSubmitHook::new();
    let mut slot = sidecar.concurrent_submit_hook.lock().unwrap();
    assert!(
        slot.is_none(),
        "only one concurrent-submit hook may be pending"
    );
    *slot = Some(Arc::clone(&hook));
    hook
}

struct QosReadback {
    set_result: i32,
    get_result: i32,
    class: u32,
    relative_priority: i32,
}

fn set_background_qos_and_read_back(thread: *mut c_void) -> QosReadback {
    // SAFETY: these are process-local pthread APIs. The current thread is
    // alive for the process lifetime and both output pointers are valid.
    unsafe {
        let set_result = pthread_set_qos_class_self_np(BACKGROUND_QOS_CLASS, 0);
        let mut class = 0u32;
        let mut relative_priority = 0i32;
        let get_result = pthread_get_qos_class_np(thread, &mut class, &mut relative_priority);
        QosReadback {
            set_result,
            get_result,
            class,
            relative_priority,
        }
    }
}

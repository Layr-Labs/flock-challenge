use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use rayon::prelude::*;

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
    std::env::var("FLOCK_EPOOL_FORCE_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
}

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

pub(crate) fn epool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(build_epool).as_ref()
}

pub fn helper_pool_available() -> bool {
    epool().is_some()
}

pub fn helper_pool() -> Option<&'static rayon::ThreadPool> {
    epool()
}

const EPOOL_MIN_CHUNKS: usize = 8;

static EPOOL_HELPER_CHUNKS: AtomicUsize = AtomicUsize::new(0);

pub fn helper_chunks_claimed() -> usize {
    EPOOL_HELPER_CHUNKS.load(Ordering::Relaxed)
}

static EPOOL_BROADCASTS: AtomicU64 = AtomicU64::new(0);

pub fn helper_broadcasts_issued() -> u64 {
    EPOOL_BROADCASTS.load(Ordering::Relaxed)
}

pub(crate) fn relay_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !relay_killed_by(std::env::var("FLOCK_NO_EPOOL_RELAY").ok().as_deref()))
}

fn relay_killed_by(value: Option<&str>) -> bool {
    value == Some("1")
}

#[derive(Clone, Copy)]
struct JobPtr {
    data: *const (),
    call: unsafe fn(*const ()),
}

unsafe impl Send for JobPtr {}

impl JobPtr {
    fn new<B: Fn() + Sync>(broadcast: &B) -> Self {
        unsafe fn thunk<B: Fn() + Sync>(data: *const ()) {
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
    job: Option<JobPtr>,
    posted: u64,
    done: u64,
    panicked: bool,
}

struct Relay {
    state: Mutex<RelayState>,
    signal: Condvar,
    busy: AtomicBool,
}

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

    fn try_acquire(&self) -> bool {
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

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

fn relay() -> Option<&'static Relay> {
    static RELAY: OnceLock<Option<&'static Relay>> = OnceLock::new();
    *RELAY.get_or_init(|| {
        let relay: &'static Relay = Box::leak(Box::new(Relay::new()));
        std::thread::Builder::new()
            .name("flock-epool-relay".to_string())
            .spawn(move || relay.run())
            .ok()
            .map(|_handle| relay)
    })
}

struct PostedJob<'a> {
    relay: &'a Relay,
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
                if !posted.load(Ordering::Relaxed) && !posted.swap(true, Ordering::Relaxed) {
                    let seq = unsafe { relay.post(JobPtr::new(broadcast)) };
                    job.seq.store(seq, Ordering::Relaxed);
                }
                worker();
            });
        drop(job);
        return;
    }
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

pub fn run_helper_only_chunks<F>(n_chunks: usize, max_workers: usize, f: &F) -> bool
where
    F: Fn(usize) + Sync + ?Sized,
{
    run_chunks_with_helper_only(n_chunks, max_workers, f, epool())
}

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

#[allow(dead_code)]
pub(crate) fn run_hetero_chunks_stateful<S, I, F>(n_chunks: usize, init: I, f: F)
where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize) + Sync,
{
    run_chunks_with_helper_stateful(n_chunks, &init, &f, epool());
}

pub fn run_chunks_with_helper<F>(n_chunks: usize, f: &F, helper: Option<&rayon::ThreadPool>)
where
    F: Fn(usize) + Sync,
{
    run_chunks_with_helper_relay(n_chunks, f, helper, relay_enabled());
}

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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);

impl<T> SyncPtr<T> {
    #[inline]
    pub fn ptr(self) -> *mut T {
        self.0
    }
}

unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn zero_chunks_is_noop() {
        run_chunks_with_helper(0, &|_| panic!("must not run"), None);
    }

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
        let after_engaged = helper_broadcasts_issued();
        run_chunks_with_helper(1024, &|_| {}, None);
        assert!(helper_broadcasts_issued() >= after_engaged);
    }

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

pub fn keepalive_start() {
    if imp::pool_spin_start() {
        return;
    }
    imp::start();
}

pub fn pool_spin_release() {
    imp::pool_spin_release();
}

pub fn keepalive_stop() {
    imp::stop();
}

pub fn keepalive_signal() {
    imp::signal();
}

pub fn keepalive_join() {
    imp::join_all();
}

mod imp {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    const MAX_KEEPALIVE: Duration = Duration::from_secs(2);

    static RUNNING: AtomicBool = AtomicBool::new(false);

    static LIVE: AtomicUsize = AtomicUsize::new(0);

    const QUIET_TIMEOUT: Duration = Duration::from_micros(250);

    fn spin_until_stopped(deadline: Instant) {
        crate::set_calling_thread_prover_qos();
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        while RUNNING.load(Ordering::Relaxed) {
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
        LIVE.fetch_sub(1, Ordering::AcqRel);
    }

    static POOL_SPIN: AtomicBool = AtomicBool::new(false);

    pub(super) fn pool_spin_start() -> bool {
        if std::env::var_os("FLOCK_NO_CPU_KEEPALIVE").is_some()
            || std::env::var_os("FLOCK_NO_POOL_SPIN").is_some()
            || rayon::current_thread_index().is_some()
            || rayon::current_num_threads() <= 1
        {
            return false;
        }
        if POOL_SPIN.swap(true, Ordering::AcqRel) {
            return true;
        }
        let deadline = Instant::now() + MAX_KEEPALIVE;
        rayon::spawn_broadcast(move |ctx| {
            if ctx.index() + 1 == ctx.num_threads() {
                return;
            }
            pool_spin_body(deadline);
        });
        true
    }

    fn pool_spin_body(deadline: Instant) {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        while POOL_SPIN.load(Ordering::Relaxed) {
            if let Some(rayon::Yield::Executed) = rayon::yield_now() {
                continue;
            }
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
    }

    pub(super) fn pool_spin_release() {
        if POOL_SPIN.load(Ordering::Relaxed) {
            POOL_SPIN.store(false, Ordering::Release);
        }
    }

    pub(super) fn start() {
        if std::env::var_os("FLOCK_NO_CPU_KEEPALIVE").is_some() {
            return;
        }
        if RUNNING.swap(true, Ordering::AcqRel) {
            return;
        }
        let n_cores = rayon::current_num_threads().max(1);
        let deadline = Instant::now() + MAX_KEEPALIVE;
        for i in 0..n_cores {
            LIVE.fetch_add(1, Ordering::AcqRel);
            match std::thread::Builder::new()
                .name(format!("flock-keepalive-{i}"))
                .stack_size(64 * 1024)
                .spawn(move || spin_until_stopped(deadline))
            {
                Ok(_) => {}
                Err(_) => {
                    LIVE.fetch_sub(1, Ordering::AcqRel);
                    break;
                }
            }
        }
    }

    pub(super) fn stop() {
        signal();
        join_all();
    }

    pub(super) fn signal() {
        RUNNING.swap(false, Ordering::AcqRel);
    }

    pub(super) fn join_all() {
        let give_up_at = Instant::now() + QUIET_TIMEOUT;
        while LIVE.load(Ordering::Acquire) != 0 {
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

        static TEST_LOCK: Mutex<()> = Mutex::new(());

        #[test]
        fn start_then_stop_is_clean() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            start();
            stop();
            assert!(!RUNNING.load(Ordering::SeqCst));
            assert_eq!(LIVE.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn stop_without_start_is_noop() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            stop();
            stop();
            assert!(!RUNNING.load(Ordering::SeqCst));
            assert_eq!(LIVE.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn kill_switch_spawns_nothing() {
            let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { std::env::set_var("FLOCK_NO_CPU_KEEPALIVE", "1") };
            start();
            let empty = LIVE.load(Ordering::SeqCst) == 0;
            let stopped = !RUNNING.load(Ordering::SeqCst);
            unsafe { std::env::remove_var("FLOCK_NO_CPU_KEEPALIVE") };
            stop();
            assert!(empty, "kill switch must spawn no keep-alive threads");
            assert!(stopped, "kill switch must leave the run flag clear");
        }

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

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub fn enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FLOCK_GAP_TIMING").is_some());
    *ON
}

static PROC_EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

static BEGIN_NS: AtomicU64 = AtomicU64::new(0);

static LAST_NS: AtomicU64 = AtomicU64::new(0);

fn now_ns() -> u64 {
    PROC_EPOCH.elapsed().as_nanos() as u64
}

pub fn begin(label: &str) {
    if !enabled() {
        return;
    }
    let now = now_ns();
    BEGIN_NS.store(now, Ordering::Relaxed);
    LAST_NS.store(now, Ordering::Relaxed);
    eprintln!("[gap] ======== {label} ========");
}

pub fn mark(label: &str) {
    if !enabled() {
        return;
    }
    let now = now_ns();
    let last = LAST_NS.swap(now, Ordering::Relaxed);
    let begin = BEGIN_NS.load(Ordering::Relaxed);
    let d_ms = now.saturating_sub(last) as f64 / 1e6;
    let t_ms = now.saturating_sub(begin) as f64 / 1e6;
    eprintln!("[gap] +{d_ms:9.3} ms  @{t_ms:9.3} ms  {label}");
}

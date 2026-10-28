#![allow(
    dead_code,
    unused_imports,
    clippy::needless_return,
    clippy::missing_safety_doc,
    clippy::redundant_closure,
    clippy::collapsible_if,
    clippy::manual_range_contains,
    clippy::manual_is_multiple_of,
    clippy::manual_slice_size_calculation,
    clippy::manual_div_ceil,
    clippy::let_and_return,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::unnecessary_cast,
    clippy::items_after_test_module
)]

pub mod bits;
pub mod challenger;
pub mod cpu_keepalive;
#[doc(hidden)]
pub mod epool;
pub mod field;
pub mod gpu_commit;
pub mod hash;
#[allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    clashing_extern_declarations,
    unused_unsafe,
    unused_variables,
    unused_parens,
    unused_assignments,
    unreachable_code,
    clippy::all
)]
pub mod stage_ir;
pub use stage_ir as gf2_kernels;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod gf2_shaders;
pub mod lincheck;
pub mod merkle;
pub mod ntt;
pub mod pcs;
pub mod permutation;
pub mod proof;
pub mod r1cs;
pub mod scratch;
pub mod verifier;
pub mod zerocheck;

pub fn init_perf_thread_pool() -> Option<usize> {
    let n = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(perf_core_count);
    set_prover_thread_qos();
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .start_handler(|_| set_prover_thread_qos())
        .build_global()
    {
        Ok(()) => Some(n),
        Err(_) => None,
    }
}

pub fn set_calling_thread_prover_qos() {
    set_prover_thread_qos();
}

pub fn set_calling_thread_shadow_qos() {
    set_shadow_thread_qos();
}

#[cfg(target_os = "macos")]
fn set_shadow_thread_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x11, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_shadow_thread_qos() {}

#[cfg(target_os = "macos")]
fn set_prover_thread_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x19, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_prover_thread_qos() {}

#[allow(clippy::uninit_vec)]
pub(crate) fn alloc_uninit_vec<T: Copy>(n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    unsafe {
        v.set_len(n);
    }
    v
}

pub(crate) fn alloc_uninit_f128_vec(n: usize) -> Vec<crate::field::F128> {
    alloc_uninit_vec::<crate::field::F128>(n)
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn perf_core_count_cached() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(perf_core_count)
}

fn perf_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.physicalcpu"])
            .output()
            && let Ok(s) = std::str::from_utf8(&out.stdout)
            && let Ok(n) = s.trim().parse::<usize>()
            && n > 0
        {
            return n;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(n) = linux_physical_cores()
            && n > 0
        {
            let available = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            return n.min(available);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(target_os = "linux")]
fn linux_physical_cores() -> Option<usize> {
    use std::collections::HashSet;
    let mut cores: HashSet<(String, String)> = HashSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let Some(rest) = name.strip_prefix("cpu") else {
            continue;
        };
        if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
            continue; // skip "cpufreq", "cpuidle", etc.
        }
        let topo = path.join("topology");
        let core_id = std::fs::read_to_string(topo.join("core_id")).ok();
        let pkg = std::fs::read_to_string(topo.join("physical_package_id")).ok();
        if let (Some(c), Some(p)) = (core_id, pkg) {
            cores.insert((p.trim().to_owned(), c.trim().to_owned()));
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

pub mod tmark;

pub fn micro_stack_enabled() -> bool {
    std::env::var_os("FLOCK_NO_MICRO_STACK").as_deref() != Some(std::ffi::OsStr::new("1"))
}

pub const UNINIT_VEC_TAIL_PAGES: usize = 96;
pub const UNINIT_VEC_LARGE_MIN_BYTES: usize = 1 << 21;
std::thread_local! {
    pub static LAST_ALLOC_RECYCLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

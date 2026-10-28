//! `flock-core`: the protocol library and verifier for Flock's R1CS-over-GF(2)
//! sumcheck/zerocheck PIOP with a multilinear PCS.
//!
//! This crate carries everything the verifier needs. It is portable — the NEON
//! kernels in `field`, `ntt`, `lincheck`, `zerocheck`, and `merkle` have scalar
//! fallbacks — though it is tuned for Apple silicon. The end-to-end prover, the
//! hash R1CS encoders, and the CLI live in the `flock-prover` crate built on
//! top of this one.
//!
//! Protocol flow:
//!   1. Prover commits to the witness z ∈ GF(2)^n via a multilinear PCS.
//!   2. Prover computes the row-witnesses a = A·z, b = B·z, c = C·z.
//!   3. Zerocheck PIOP reduces a·b ⊕ c = 0 to evaluation claims on (â, b̂, ĉ) at ρ.
//!   4. Lincheck PIOP reduces those to a single evaluation claim ẑ(ρ') = v.
//!   5. PCS opens ẑ at ρ'.
//!
//! Workspace-wide Clippy `allow`s for the hand-tuned numeric kernels are
//! declared in `[workspace.lints.clippy]` at the repo root.

pub mod bits;
pub mod challenger;
pub mod cpu_keepalive;
// Public but hidden: flock-prover's witness driver drains its groups through
// the hetero queue (W-H1). Not a stable API surface.
#[doc(hidden)]
pub mod epool;
pub mod field;
// Public only for `note_precompute_branch_wall_ms` (the prover's join arm
// reports its wall to the hybrid-split warmup sweep); internals stay
// pub(crate).
pub mod gpu_commit;
pub mod hash;
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

/// Shared kill switch for the micro-stack batch of small prover cuts
/// (`eval_sk_at_vks` memoization, recursion-OOD optimized eq-table builder,
/// fast `pcs_open` bundle encoder). Set exactly `FLOCK_NO_MICRO_STACK=1` to
/// restore every incumbent path as a same-binary A/B control; any other value
/// (or unset) keeps the micro-stack paths. All gated paths are byte-identical
/// to the incumbents (each has its own equivalence test), so the switch exists
/// purely for screening/rollback — mirrors `FLOCK_NO_AB_COMPACT_STORE`.
pub const ENV_NO_MICRO_STACK: &str = "FLOCK_NO_MICRO_STACK";

/// Unlike `ab_compact_store_enabled` this does **not** latch the first read
/// in a `OnceLock`: every gated call site runs O(10) times per prove, so the
/// per-call `var_os` read is noise, and re-reading keeps same-process A/B
/// tests (set var → prove → unset → prove) possible.
pub fn micro_stack_enabled() -> bool {
    std::env::var_os(ENV_NO_MICRO_STACK).as_deref() != Some(std::ffi::OsStr::new("1"))
}

/// Configure rayon's global thread pool to use only performance cores on
/// Apple silicon (excluding efficiency cores).
///
/// On M-series chips the 2 efficiency cores run at ~30-40% of perf-core
/// speed and become stragglers in compute-bound parallel work — the
/// work-stealing scheduler keeps assigning them tasks that hold up the perf
/// cores at synchronization barriers. Empirically, 8 threads beats 10 by
/// ~10-20% on `pcs::commit` and similar parallel-NTT workloads.
///
/// Call this **once** at program startup, before any other parallel flock
/// code runs (rayon's global pool is set on first use; if it's already
/// created, this call is a no-op).
///
/// Respects `RAYON_NUM_THREADS` as an explicit size override while retaining
/// the platform worker setup below.
///
/// Returns the number of threads the pool was configured with, or `None`
/// if no change was made because Rayon was already initialized.
pub fn init_perf_thread_pool() -> Option<usize> {
    let n = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(perf_core_count);
    // The main/calling thread runs *all* sequential work (seed expansion,
    // Fiat-Shamir observe/sample, proof serialization, multi-proof extract)
    // but rayon's start_handler only tags the pool workers. On Apple Silicon
    // an unspecified-QoS thread is freely E-cluster-eligible, which can park
    // that serial work on efficiency cores while the P-cluster's DVFS domain
    // still participates in the QoS decision. Pin the caller first so the
    // whole prover process is USER_INITIATED before any timed trial starts.
    set_prover_thread_qos();
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .start_handler(|_| set_prover_thread_qos())
        .build_global()
    {
        Ok(()) => Some(n),
        Err(_) => None, // pool already built
    }
}

/// Apply the prover's QoS class to the calling thread.
///
/// Threads this crate's consumers spawn outside the Rayon pool (the ranked
/// worker's seed-pipeline thread) still run timed serial work, so they need
/// the same P-cluster pin [`init_perf_thread_pool`] gives the main thread.
pub fn set_calling_thread_prover_qos() {
    set_prover_thread_qos();
}

/// Move the calling thread onto the efficiency cluster for a span that nothing
/// timed is waiting on.
///
/// The comment on [`init_perf_thread_pool`] — "the main/calling thread runs
/// *all* sequential work" — predates seed pipelining and is no longer true of
/// the ranked worker. Once stdin is spliced, the protected wrapper's main
/// thread re-expands the seed and byte-compares the result *while the real
/// proof is already running on the seed-pipe thread*, so across that span it
/// holds no critical-path work at all. What it does hold is an eleventh
/// runnable thread against a Rayon pool sized to the ten performance cores,
/// plus ~29 MiB of expansion stores and ~59 MiB of comparison reads landing in
/// the store-bound witness phase. `QOS_CLASS_UTILITY` is the same class
/// [`epool`] uses to reach the E-cluster, so the scheduler parks that shadow
/// work there instead. Reverse with [`set_calling_thread_prover_qos`] before
/// any timed serial work resumes.
pub fn set_calling_thread_shadow_qos() {
    set_shadow_thread_qos();
}

#[cfg(target_os = "macos")]
fn set_shadow_thread_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // QOS_CLASS_UTILITY: the scheduler places these on efficiency cores.
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x11, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_shadow_thread_qos() {}

/// Mark Rayon prover workers as latency-sensitive on macOS. A bare Rayon pool
/// inherits default QoS, which lets sustained jobs drift onto efficiency cores
/// even when the pool was deliberately sized to the performance-core count.
#[cfg(target_os = "macos")]
fn set_prover_thread_qos() {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // QOS_CLASS_USER_INITIATED: explicit user work that should finish promptly.
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x19, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_prover_thread_qos() {}

/// Allocate a `Vec<T>` of length `n` whose contents are NOT zero-initialized.
/// Caller MUST write every slot before reading it.
///
/// Used to skip the eager zero-init of large ping-pong buffers in hot prover
/// paths (PCS open, Round-2 fold, NTT scratch, lincheck packing). At m=29 the
/// zero-fill of a fresh 128 MB `vec![T::default(); n]` runs sequentially on
/// the main thread (~22 ms), which caps the parallel speedup of those phases.
///
/// `T: Copy` ensures `T` has no Drop impl, so the leaked uninitialized
/// elements are a no-op on drop.
///
/// # Safety contract
///
/// Reading uninitialized memory is UB per Rust's memory model regardless of
/// whether all bit patterns are valid for `T`. Caller must ensure every slot
/// is written before any read.
// `uninit_vec` flags exactly this pattern; here it is the deliberate purpose of
// the function (the safety contract above is what makes it sound).
#[allow(clippy::uninit_vec)]
pub(crate) fn alloc_uninit_vec<T: Copy>(n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    // SAFETY:
    // - capacity == n was just allocated, so set_len(n) is in bounds.
    // - T: Copy implies !Drop, so leaking uninit elements is a no-op.
    // - Caller upholds write-before-read.
    unsafe {
        v.set_len(n);
    }
    v
}

/// Compatibility shim — same as `alloc_uninit_vec::<F128>(n)`.
pub(crate) fn alloc_uninit_f128_vec(n: usize) -> Vec<crate::field::F128> {
    alloc_uninit_vec::<crate::field::F128>(n)
}

/// Cached [`perf_core_count`]. The uncached version may spawn `sysctl`; this
/// memoizes it so hot paths can cheaply ask "is the current rayon pool the
/// homogeneous P-core pool?" (i.e. `current_num_threads() <= this`).
#[cfg(target_arch = "aarch64")]
pub(crate) fn perf_core_count_cached() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(perf_core_count)
}

/// Best-effort count of **physical** performance cores used to size the
/// prover's thread pool. The hot phases are CLMUL-heavy and/or
/// memory-bandwidth-bound; SMT siblings share the core's execution ports and
/// add no DRAM bandwidth, so running 2 threads per physical core only adds
/// contention (on a 32C/64T Threadripper the prove is ~16% faster at 32 threads
/// than 64). On macOS, queries `hw.perflevel0.physicalcpu` (= P-core count on
/// Apple silicon, = physical CPU count on Intel). On Linux, `available_
/// parallelism()` counts SMT siblings, so derive physical cores from `/sys`
/// topology and clamp that host-wide count to the process's affinity/cgroup
/// availability. Elsewhere, falls back to `available_parallelism()`.
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

/// Count distinct physical cores via `/sys` topology: one entry per unique
/// `(physical_package_id, core_id)` over the online `cpuN` directories. Returns
/// `None` if the topology can't be read (caller falls back to logical count).
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

//! Metal GPU offload runtime (Apple Silicon only): shared context + two
//! pipeline families.
//!
//! Season-1 resurrection, generalized. The original machinery lived in
//! `zerocheck/univariate_skip_optimized/gpu.rs` (reflog `129ab82`) and drove a
//! single URM round-1 pipeline; this module hoists the runtime (objc FFI,
//! device/queue/pipeline cache, MTLBuffer wrap-cache, prewire, failure latch)
//! to crate level and adds BLAKE3 Merkle pipelines so one wiring/warm cycle
//! per prove amortizes over two phases (Merkle leaf streaming during the NTT
//! deep pass + the URM x_hi split ~100 ms later).
//!
//! Robustness rails (unchanged from season 1):
//!   * Any runtime failure latches a permanent CPU fallback ([`is_disabled`])
//!     — callers always keep a bit-identical CPU path.
//!   * Witness/codeword/tree buffers are wrapped zero-copy
//!     (`newBufferWithBytesNoCopy`) when page-aligned and page-multiple; large
//!     (≥ 64 MiB) wraps are cached for the process (`wrap_cache`), so those
//!     allocations MUST be pool-retained (never freed) by the caller — the
//!     pooled prover buffers satisfy this. [`prewire`] warms the cache from an
//!     async thread to hide the ~25–30 ms page-wiring cost.
//!   * Env escapes: `FLOCK_NO_GPU` (everything), `FLOCK_NO_GPU_URM`,
//!     `FLOCK_NO_GPU_MERKLE`.
//!
//! Pipelines:
//!   * [`urm`] — the season-1 URM round-1 split (x_hi ∈ [0, g) on GPU, XOR
//!     merge). Shader ported verbatim; see the shader-mapping notes on the
//!     kernel source. Bit-identical to the CPU path because every accumulator
//!     is a per-lane GF(2^128) XOR and GHASH reduction commutes with XOR.
//!   * [`merkle`] — BLAKE3 Merkle tree hashing with the exact non-root
//!     chaining-value semantics of [`crate::merkle`]:
//!       leaf   = `Hasher::new().update(leaf).finalize_non_root()`
//!                (single-chunk, counter 0, `CHUNK_START` on block 0,
//!                `CHUNK_END` on the last block, never `ROOT`),
//!       parent = `merge_subtrees_non_root(l, r, Mode::Hash)`
//!                (one 64-byte block = l ‖ r, IV state, `PARENT` flag).
//!     Leaves are 64..=1024-byte multiples of 64 (always 1024 B at the ranked
//!     shape), so every block is a whole 64-byte block — no partial-block
//!     lengths anywhere. Digests land directly in the caller's tree memory
//!     (flat `merkle_tree` layout) via the zero-copy wrap: no readback copies.

use crate::field::F128;
use crate::ntt::InvNttTableByteSingleGf8;

/// Lanes per URM bank (fixed by k_skip = 6).
pub const ELL: usize = 64;

/// stderr discard: when the developer marker `/private/tmp/flock-gpu-debug`
/// exists, append `msg` to `$TMPDIR/gpu-trace.log` (inside the worker's
/// sandbox-writable scratch). No-op (one cached `exists` check) otherwise.
pub fn gpu_dbg_trace(msg: &str) {
    use std::io::Write;
    static MARKER: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::path::Path::new("/private/tmp/flock-gpu-debug").exists());
    if !*MARKER {
        return;
    }
    let path = std::env::temp_dir().join("gpu-trace.log");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "[{:?}] {}", std::thread::current().id(), msg);
    }
}

// ---------------------------------------------------------------------------
// Public facade — real implementation on macOS/aarch64, inert stub elsewhere.
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use real as imp;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use stub as imp;

pub use imp::{is_disabled, metal_available, prewire, undisable};

/// URM round-1 GPU split (season-1 pipeline, verbatim shader).
pub mod urm {
    use super::{ELL, F128, InvNttTableByteSingleGf8};

    /// Arguments for one GPU share dispatch. Mirrors what
    /// `process_one_x_hi_with_s_hat_v` needs, minus the per-worker scratch.
    pub struct ShareArgs<'a> {
        pub a_packed: &'a [u8],
        pub b_packed: &'a [u8],
        pub c_packed: &'a [u8],
        pub inv_table: &'a InvNttTableByteSingleGf8,
        pub eq_lo_scaled: &'a [F128],
        pub eq_hi: &'a [F128],
        pub b_med_counts: &'a [u8],
        pub within_outer_mask: usize,
        pub n_lo: usize,
        pub n_lo_and_inner: usize,
        /// GPU takes `x_hi ∈ [0, g)`.
        pub g: usize,
        /// Test hook: cap the per-threadgroup x_outer_lo tile (rounded up to
        /// the stream count) to exercise multi-tile grids on small inputs.
        /// `None` = production default.
        pub tile_x_outer_lo: Option<usize>,
    }

    /// GPU share result: eq_hi-folded per-bank partials for `x_hi ∈ [0, g)`,
    /// ready to XOR into the CPU reduction, plus the GPU-side elapsed seconds
    /// (command-buffer GPU timestamps) for calibration.
    pub struct ShareResult {
        pub res_ab: [F128; ELL],
        pub res_c0: [F128; ELL],
        pub res_c1: [F128; ELL],
        pub gpu_seconds: f64,
    }

    pub use super::imp::urm::{
        Job, enabled, keepalive_should_run, note_calibration, planned_g, set_enabled, start_share,
    };
}

/// BLAKE3 Merkle pipelines (leaf chaining values + parent merges).
pub mod merkle {
    pub use super::imp::merkle::{Session, available, begin, set_enabled};
}

// ---------------------------------------------------------------------------
// Stub for non-Apple targets: the GPU path never engages, everything inlines
// away. Keeps callers free of cfg noise.
// ---------------------------------------------------------------------------

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod stub {
    pub fn metal_available() -> bool {
        false
    }
    pub fn is_disabled() -> bool {
        false
    }
    pub fn undisable() {}
    pub fn prewire(_data: &[u8]) {}

    pub mod urm {
        use crate::gpu::urm::{ShareArgs, ShareResult};

        pub struct Job<'a>(core::marker::PhantomData<&'a ()>);

        impl Job<'_> {
            pub fn finish(self) -> Option<ShareResult> {
                None
            }
        }

        pub fn planned_g(_hi_size: usize, _m: usize) -> usize {
            0
        }

        pub fn start_share<'a>(_args: ShareArgs<'a>) -> Option<Job<'a>> {
            None
        }

        pub fn note_calibration(
            _g: usize,
            _cpu_count: usize,
            _gpu_seconds: f64,
            _cpu_seconds: f64,
            _wait_seconds: f64,
        ) {
        }

        /// Keepalive never engages off-Apple.
        pub fn keepalive_should_run() -> bool {
            false
        }

        pub fn set_enabled(_on: bool) {}
        pub fn enabled() -> bool {
            false
        }
    }

    pub mod merkle {
        use crate::merkle::Hash;

        /// Uninhabited off-Apple: `begin` never constructs one.
        pub enum Session {}

        impl Session {
            pub fn commit_leaves(&mut self, _leaf_lo: usize, _leaf_hi: usize) -> bool {
                match *self {}
            }
            pub fn commit_parent_levels(&mut self) -> bool {
                match *self {}
            }
            pub fn finish(self) -> Option<f64> {
                match self {}
            }
        }

        pub fn available() -> bool {
            false
        }
        pub fn set_enabled(_on: bool) {}

        /// # Safety
        /// Never dereferences anything off-Apple; always returns `None`.
        pub unsafe fn begin(
            _data: &[u8],
            _leaf_size: usize,
            _tree_ptr: *mut Hash,
            _tree_len: usize,
            _stop_nodes: usize,
        ) -> Option<Session> {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Real implementation: macOS + aarch64.
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod real {
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::LazyLock;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[cfg(test)]
    use std::sync::atomic::AtomicUsize;

    // -- env escapes ---------------------------------------------------------

    static ENV_NO_GPU: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("FLOCK_NO_GPU").is_some());
    static ENV_NO_URM: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("FLOCK_NO_GPU_URM").is_some());
    static ENV_NO_MERKLE: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("FLOCK_NO_GPU_MERKLE").is_some());

    // -- objc runtime / framework FFI ---------------------------------------

    type Id = *mut c_void;
    type Sel = *const c_void;

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const u8) -> Id;
        fn sel_registerName(name: *const u8) -> Sel;
        fn objc_msgSend();
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(ctx: *mut c_void);
    }

    #[link(name = "Metal", kind = "framework")]
    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> Id;
    }

    // Foundation provides NSString; Metal transitively loads it, the explicit
    // link keeps the dependency honest.
    #[link(name = "Foundation", kind = "framework")]
    unsafe extern "C" {}

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MTLSize {
        width: usize,
        height: usize,
        depth: usize,
    }

    /// macOS/arm64 VM page size; `newBufferWithBytesNoCopy` requires the base
    /// pointer and length to be multiples of this.
    const PAGE: usize = 16 * 1024;
    /// Largest input we are willing to copy into a fresh MTLBuffer when the
    /// zero-copy alignment requirements fail (per array).
    const MAX_COPY_BYTES: usize = 32 << 20;

    unsafe fn msg0(obj: Id, sel: Sel) -> Id {
        let f: unsafe extern "C" fn(Id, Sel) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel) }
    }

    unsafe fn msg0_void(obj: Id, sel: Sel) {
        let f: unsafe extern "C" fn(Id, Sel) =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel) }
    }

    unsafe fn msg0_usize(obj: Id, sel: Sel) -> usize {
        let f: unsafe extern "C" fn(Id, Sel) -> usize =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel) }
    }

    unsafe fn msg0_f64(obj: Id, sel: Sel) -> f64 {
        let f: unsafe extern "C" fn(Id, Sel) -> f64 =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel) }
    }

    unsafe fn msg1(obj: Id, sel: Sel, a: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, Id) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, a) }
    }

    unsafe fn msg1_void(obj: Id, sel: Sel, a: Id) {
        let f: unsafe extern "C" fn(Id, Sel, Id) =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, a) }
    }

    unsafe fn msg_str(obj: Id, sel: Sel, s: *const u8) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, *const u8) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, s) }
    }

    unsafe fn msg_lib(obj: Id, sel: Sel, src: Id, opts: Id, err: *mut Id) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, src, opts, err) }
    }

    unsafe fn msg_pso(obj: Id, sel: Sel, func: Id, err: *mut Id) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, func, err) }
    }

    unsafe fn msg_buf_bytes(obj: Id, sel: Sel, p: *const c_void, len: usize, opts: usize) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, *const c_void, usize, usize) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, p, len, opts) }
    }

    unsafe fn msg_buf_nocopy(
        obj: Id,
        sel: Sel,
        p: *const c_void,
        len: usize,
        opts: usize,
        dealloc: Id,
    ) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, *const c_void, usize, usize, Id) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, p, len, opts, dealloc) }
    }

    unsafe fn msg_buf_len(obj: Id, sel: Sel, len: usize, opts: usize) -> Id {
        let f: unsafe extern "C" fn(Id, Sel, usize, usize) -> Id =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, len, opts) }
    }

    unsafe fn msg_set_buffer(obj: Id, sel: Sel, buf: Id, offset: usize, index: usize) {
        let f: unsafe extern "C" fn(Id, Sel, Id, usize, usize) =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, buf, offset, index) }
    }

    unsafe fn msg_set_bytes(obj: Id, sel: Sel, p: *const c_void, len: usize, index: usize) {
        let f: unsafe extern "C" fn(Id, Sel, *const c_void, usize, usize) =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, p, len, index) }
    }

    unsafe fn msg_dispatch(obj: Id, sel: Sel, tgs: MTLSize, tpg: MTLSize) {
        let f: unsafe extern "C" fn(Id, Sel, MTLSize, MTLSize) =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel, tgs, tpg) }
    }

    unsafe fn msg0_ptr(obj: Id, sel: Sel) -> *mut c_void {
        let f: unsafe extern "C" fn(Id, Sel) -> *mut c_void =
            unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel) }
    }

    unsafe fn sel(name: &'static str) -> Sel {
        debug_assert!(name.ends_with('\0'));
        unsafe { sel_registerName(name.as_ptr()) }
    }

    /// Owned NSString from a NUL-terminated Rust string (autoreleased).
    unsafe fn nsstring(s: &std::ffi::CStr) -> Id {
        unsafe {
            let cls = objc_getClass(c"NSString".to_bytes_with_nul().as_ptr());
            msg_str(
                cls,
                sel("stringWithUTF8String:\0"),
                s.to_bytes_with_nul().as_ptr(),
            )
        }
    }

    // -- cached Metal context ------------------------------------------------

    struct Sels {
        new_buf_bytes: Sel,
        new_buf_nocopy: Sel,
        new_buf_len: Sel,
        command_buffer: Sel,
        compute_encoder: Sel,
        set_pso: Sel,
        set_buffer: Sel,
        set_bytes: Sel,
        dispatch_tg: Sel,
        end_encoding: Sel,
        commit: Sel,
        wait_completed: Sel,
        status: Sel,
        gpu_start_time: Sel,
        gpu_end_time: Sel,
        contents: Sel,
        retain: Sel,
        release: Sel,
    }

    struct MetalCtx {
        device: Id,
        queue: Id,
        /// URM round-1 pipeline.
        urm_pso: Id,
        /// BLAKE3 leaf chaining-value pipeline.
        leaf_pso: Id,
        /// BLAKE3 parent merge pipeline.
        parent_pso: Id,
        /// 64 KiB convert table, uploaded once (process-constant content).
        convert_buf: Id,
        /// Cache of no-copy MTLBuffer objects for the large input arrays,
        /// keyed by `(base_ptr, len)`. The scratch pool recycles the same
        /// allocations across proves, and creating a fresh MTLBuffer over
        /// 3 × 512 MiB each prove costs ~25–30 ms of command-buffer schedule
        /// latency (page wiring) — measured as `gpu_extra_wait` in fresh
        /// workers. Cached objects are created once per (ptr, len) and never
        /// released; bounded at [`WRAP_CACHE_CAP`] entries. Entries whose
        /// backing allocation the pool ever *frees* would dangle, but pooled
        /// prover buffers live for the process (MAX_POOLED retention) and the
        /// cache only admits ≥ [`WRAP_CACHE_MIN_BYTES`] page-aligned wraps —
        /// exactly the pooled shapes.
        wrap_cache: std::sync::Mutex<Vec<((usize, usize), Id)>>,
        sels: Sels,
    }

    const WRAP_CACHE_CAP: usize = 24;
    const WRAP_CACHE_MIN_BYTES: usize = 1 << 26;

    // SAFETY: MTLDevice/MTLCommandQueue/MTLComputePipelineState/MTLBuffer are
    // documented thread-safe; the raw pointers are only handles.
    unsafe impl Send for MetalCtx {}
    unsafe impl Sync for MetalCtx {}

    static CTX: LazyLock<Option<MetalCtx>> = LazyLock::new(init_ctx);
    /// Permanent CPU-fallback latch, set on any runtime failure.
    static DISABLED: AtomicBool = AtomicBool::new(false);

    #[track_caller]
    fn disable() {
        let loc = std::panic::Location::caller();
        crate::gpu::gpu_dbg_trace(&format!("DISABLE latched at {loc}"));
        if std::env::var_os("FLOCK_GPU_TRACE").is_some() {
            eprintln!("[gpu] DISABLE latched at {loc}");
        }
        DISABLED.store(true, Ordering::Relaxed);
    }

    fn ctx() -> Option<&'static MetalCtx> {
        if *ENV_NO_GPU {
            return None;
        }
        CTX.as_ref()
    }

    /// Whether a usable Metal context exists. Forces the (cached) context
    /// init — device lookup + shader compile, ~45 ms on first call — so run
    /// it during the untimed warmup prove.
    pub fn metal_available() -> bool {
        ctx().is_some()
    }

    pub fn is_disabled() -> bool {
        DISABLED.load(Ordering::Relaxed)
    }

    /// Clear the failure latch. Used ONLY by keepalive/warmup drivers: a
    /// failed warm dispatch must not condemn the real paths (which make their
    /// own attempts and latch on their own failures).
    pub fn undisable() {
        DISABLED.store(false, Ordering::Relaxed);
    }

    /// Warm the wrap cache for one large buffer: create (and cache) its
    /// no-copy MTLBuffer so the page-wiring cost is paid off the critical
    /// path. No-op unless the buffer is page-aligned, page-multiple and
    /// ≥ 64 MiB (the cacheable shapes). Safe to call from a spawned thread.
    pub fn prewire(data: &[u8]) {
        if DISABLED.load(Ordering::Relaxed) {
            return;
        }
        let Some(ctx) = ctx() else { return };
        let ptr_ok = (data.as_ptr() as usize) % PAGE == 0;
        let len_ok = !data.is_empty() && data.len() % PAGE == 0;
        if !ptr_ok || !len_ok || data.len() < WRAP_CACHE_MIN_BYTES {
            return;
        }
        unsafe {
            let pool = objc_autoreleasePoolPush();
            match wrap_input(ctx, data) {
                Some((buf, false)) => msg0_void(buf, ctx.sels.release),
                Some((_, true)) | None => {}
            }
            objc_autoreleasePoolPop(pool);
        }
    }

    fn init_ctx() -> Option<MetalCtx> {
        if *ENV_NO_GPU {
            return None;
        }
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let result = init_ctx_inner();
            objc_autoreleasePoolPop(pool);
            if result.is_none() {
                disable();
            }
            result
        }
    }

    unsafe fn init_ctx_inner() -> Option<MetalCtx> {
        unsafe {
            let device = MTLCreateSystemDefaultDevice();
            if device.is_null() {
                return None;
            }

            let src = nsstring(SHADER_SOURCE_C);
            if src.is_null() {
                return None;
            }
            let mut err: Id = ptr::null_mut();
            let lib = msg_lib(
                device,
                sel("newLibraryWithSource:options:error:\0"),
                src,
                ptr::null_mut(),
                &mut err,
            );
            if lib.is_null() {
                return None;
            }

            let make_pso = |name: &std::ffi::CStr, min_threads: usize| -> Option<Id> {
                let fname = nsstring(name);
                let func = msg1(lib, sel("newFunctionWithName:\0"), fname);
                if func.is_null() {
                    return None;
                }
                let mut err2: Id = ptr::null_mut();
                let pso = msg_pso(
                    device,
                    sel("newComputePipelineStateWithFunction:error:\0"),
                    func,
                    &mut err2,
                );
                if pso.is_null() {
                    return None;
                }
                if msg0_usize(pso, sel("maxTotalThreadsPerThreadgroup\0")) < min_threads {
                    return None;
                }
                Some(pso)
            };

            let urm_pso = make_pso(c"urm_round1", urm::TG_THREADS)?;
            let leaf_pso = make_pso(c"blake3_leaf", merkle::TG_THREADS)?;
            let parent_pso = make_pso(c"blake3_parent", merkle::TG_THREADS)?;

            let queue = msg0(device, sel("newCommandQueue\0"));
            if queue.is_null() {
                return None;
            }

            let convert = crate::zerocheck::univariate_skip_optimized::convert_table();
            let convert_buf = msg_buf_bytes(
                device,
                sel("newBufferWithBytes:length:options:\0"),
                convert.as_ptr() as *const c_void,
                std::mem::size_of_val(convert),
                0, // MTLResourceStorageModeShared
            );
            if convert_buf.is_null() {
                return None;
            }

            let sels = Sels {
                new_buf_bytes: sel("newBufferWithBytes:length:options:\0"),
                new_buf_nocopy: sel("newBufferWithBytesNoCopy:length:options:deallocator:\0"),
                new_buf_len: sel("newBufferWithLength:options:\0"),
                command_buffer: sel("commandBuffer\0"),
                compute_encoder: sel("computeCommandEncoder\0"),
                set_pso: sel("setComputePipelineState:\0"),
                set_buffer: sel("setBuffer:offset:atIndex:\0"),
                set_bytes: sel("setBytes:length:atIndex:\0"),
                dispatch_tg: sel("dispatchThreadgroups:threadsPerThreadgroup:\0"),
                end_encoding: sel("endEncoding\0"),
                commit: sel("commit\0"),
                wait_completed: sel("waitUntilCompleted\0"),
                status: sel("status\0"),
                gpu_start_time: sel("GPUStartTime\0"),
                gpu_end_time: sel("GPUEndTime\0"),
                contents: sel("contents\0"),
                retain: sel("retain\0"),
                release: sel("release\0"),
            };

            Some(MetalCtx {
                device,
                queue,
                urm_pso,
                leaf_pso,
                parent_pso,
                convert_buf,
                wrap_cache: std::sync::Mutex::new(Vec::new()),
                sels,
            })
        }
    }

    /// Test-only observability: number of input arrays bound zero-copy.
    #[cfg(test)]
    pub(super) static NOCOPY_BINDS: AtomicUsize = AtomicUsize::new(0);

    /// Wrap `data` in an MTLBuffer: zero-copy when page-aligned/page-multiple,
    /// bulk copy for small inputs, `None` otherwise.
    ///
    /// Returns `(buffer, cached)`. `cached = true` means the buffer came from
    /// (or was inserted into) the ctx-lifetime wrap cache and MUST NOT be
    /// released by the caller; `cached = false` buffers are caller-owned.
    /// Caching applies only to large page-aligned no-copy wraps — the pooled
    /// arrays whose per-prove re-wrapping otherwise costs ~25-30 ms of
    /// command-buffer schedule latency (page wiring) per fresh worker.
    unsafe fn wrap_input(ctx: &MetalCtx, data: &[u8]) -> Option<(Id, bool)> {
        unsafe {
            let ptr_ok = (data.as_ptr() as usize) % PAGE == 0;
            let len_ok = !data.is_empty() && data.len() % PAGE == 0;
            if ptr_ok && len_ok {
                #[cfg(test)]
                NOCOPY_BINDS.fetch_add(1, Ordering::Relaxed);
                wrap_nocopy(ctx, data.as_ptr(), data.len())
            } else if data.len() <= MAX_COPY_BYTES {
                let buf = msg_buf_bytes(
                    ctx.device,
                    ctx.sels.new_buf_bytes,
                    data.as_ptr() as *const c_void,
                    data.len(),
                    0,
                );
                if buf.is_null() { None } else { Some((buf, false)) }
            } else {
                None
            }
        }
    }

    /// Zero-copy wrap of a page-aligned, page-multiple region (both are the
    /// caller's obligation), with wrap-cache admission for large regions.
    /// Unlike [`wrap_input`] there is no copy fallback — used for output
    /// regions the GPU must write in place (the Merkle tree).
    unsafe fn wrap_nocopy(ctx: &MetalCtx, base: *const u8, len: usize) -> Option<(Id, bool)> {
        unsafe {
            let key = (base as usize, len);
            let cacheable = len >= WRAP_CACHE_MIN_BYTES;
            if cacheable {
                // parking_lot is not in the frozen dependency tree; the
                // cache (a Vec of Copy tuples) is valid under poisoning.
                let cache = ctx
                    .wrap_cache
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(&(_, id)) = cache.iter().find(|&&(k, _)| k == key) {
                    return Some((id, true));
                }
            }
            let buf = msg_buf_nocopy(
                ctx.device,
                ctx.sels.new_buf_nocopy,
                base as *const c_void,
                len,
                0,
                ptr::null_mut(),
            );
            if buf.is_null() {
                return None;
            }
            if cacheable {
                let mut cache = ctx
                    .wrap_cache
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if cache.len() < WRAP_CACHE_CAP && !cache.iter().any(|&(k, _)| k == key) {
                    cache.push((key, buf));
                    return Some((buf, true));
                }
            }
            Some((buf, false))
        }
    }

    // =========================================================================
    // URM round-1 pipeline (season-1, ported verbatim).
    // =========================================================================

    pub mod urm {
        use super::*;
        use crate::field::F128;
        use crate::gpu::ELL;
        use crate::gpu::urm::{ShareArgs, ShareResult};
        use std::marker::PhantomData;
        use std::sync::atomic::AtomicUsize;

        /// x_outer_lo values per threadgroup (must be a multiple of N_STREAMS).
        const TILE: usize = 512;
        /// Streams (x_outer_lo slots processed concurrently) per threadgroup.
        const N_STREAMS: usize = 8;
        /// Threads per threadgroup = N_STREAMS × 64 lanes.
        pub(super) const TG_THREADS: usize = N_STREAMS * ELL;
        /// Minimum m for which the GPU split engages in production (below this
        /// the fixed dispatch overhead is not worth it). Tests bypass via the
        /// forced-g path.
        const GPU_MIN_M: usize = 26;
        /// Initial (uncalibrated) split: GPU takes x_hi ∈ [0, 64).
        const INITIAL_G: usize = 64;
        const G_MIN: usize = 16;
        const G_MAX: usize = 112;

        /// Runtime master switch. Season 1 shipped `const false` after the
        /// standalone split measured −2.64% on the M4 Pro runner (per-prove
        /// fixed costs exceeded the round-1 gain). Season 2 shares those
        /// fixed costs with the Merkle pipeline, so it defaults ON; the
        /// warmup calibration (Blake3Setup) owns the final decision via
        /// [`set_enabled`], and `FLOCK_NO_GPU_URM` force-kills it.
        static URM_ENABLED: AtomicBool = AtomicBool::new(true);
        /// Current split point g (x_hi ∈ [0, g) on GPU) for hi_size = 128.
        static SPLIT_G: AtomicUsize = AtomicUsize::new(INITIAL_G);
        static CALIBRATED: AtomicBool = AtomicBool::new(false);

        pub fn set_enabled(on: bool) {
            URM_ENABLED.store(on, Ordering::Relaxed);
        }

        pub fn enabled() -> bool {
            URM_ENABLED.load(Ordering::Relaxed) && !*ENV_NO_GPU && !*ENV_NO_URM
        }

        // -- split planning & calibration ------------------------------------

        /// GPU share for a production call, or 0 to stay pure-CPU. Triggers
        /// the (cached) Metal context init on first use — intended to happen
        /// during the untimed warmup prove.
        pub fn planned_g(hi_size: usize, m: usize) -> usize {
            if !enabled() || m < GPU_MIN_M || hi_size != 128 || DISABLED.load(Ordering::Relaxed) {
                return 0;
            }
            if ctx().is_none() {
                return 0;
            }
            SPLIT_G.load(Ordering::Relaxed).min(hi_size)
        }

        /// Whether the pre-round-1 GPU keepalive should spin: a usable context
        /// exists, no failure latch, and calibration has not decided pure-CPU.
        pub fn keepalive_should_run() -> bool {
            if !enabled() || DISABLED.load(Ordering::Relaxed) {
                return false;
            }
            if CALIBRATED.load(Ordering::Relaxed) && SPLIT_G.load(Ordering::Relaxed) == 0 {
                return false;
            }
            ctx().is_some()
        }

        /// Record the first prove's timings and fix the split point.
        ///
        /// `wait_seconds` is the extra wall time the prover spent blocked on
        /// the GPU *after* its own CPU share finished. The effective GPU cost
        /// is the full turnaround `cpu_seconds + wait_seconds` (covers
        /// schedule latency, clock ramp, and any residual page wiring) or the
        /// raw GPU-timestamp interval, whichever is larger — calibrating on
        /// optimistic timestamps alone overshoots g and stalls the timed
        /// prove.
        pub fn note_calibration(
            g: usize,
            cpu_count: usize,
            gpu_seconds: f64,
            cpu_seconds: f64,
            wait_seconds: f64,
        ) {
            if CALIBRATED.swap(true, Ordering::Relaxed) {
                return;
            }
            if g == 0 || cpu_count == 0 {
                return;
            }
            let gpu_turnaround = gpu_seconds.max(cpu_seconds + wait_seconds.max(0.0));
            let t_gpu = gpu_turnaround / g as f64;
            let t_cpu = cpu_seconds / cpu_count as f64;
            if !(t_gpu.is_finite() && t_cpu.is_finite()) || t_gpu <= 0.0 || t_cpu <= 0.0 {
                return; // keep the initial split
            }
            if std::env::var_os("FLOCK_GPU_TRACE").is_some() {
                eprintln!(
                    "[gpu-urm] calibrate: g={g} gpu_ts={:.2}ms turnaround={:.2}ms ({:.3}ms/x_hi) cpu_share={:.2}ms ({:.3}ms/x_hi over {cpu_count}) -> new g={}",
                    gpu_seconds * 1e3,
                    gpu_turnaround * 1e3,
                    t_gpu * 1e3,
                    cpu_seconds * 1e3,
                    t_cpu * 1e3,
                    calibration_rule(t_gpu, t_cpu)
                );
            }
            SPLIT_G.store(calibration_rule(t_gpu, t_cpu), Ordering::Relaxed);
        }

        /// Pure split rule on per-x_hi rates: proportional re-balance
        /// `g = 128 · t_cpu / (t_cpu + t_gpu)` clamped to `[G_MIN, G_MAX]`.
        ///
        /// Any finite speed ratio benefits from a proportional split — the
        /// phase time is `max(g·t_gpu, (128−g)·t_cpu)`, minimized at the
        /// balance point regardless of which side is slower (e.g. a 3× slower
        /// GPU still takes 32/128 of the work and cuts the phase by ~25%).
        /// Pure CPU only when the GPU is catastrophically slow (> 6×, where
        /// the clamped minimum share stops paying) — hard failures latch
        /// through `DISABLED` instead.
        pub(in crate::gpu) fn calibration_rule(t_gpu_per_x_hi: f64, t_cpu_per_x_hi: f64) -> usize {
            if t_gpu_per_x_hi > 6.0 * t_cpu_per_x_hi {
                return 0;
            }
            let ideal = 128.0 * t_cpu_per_x_hi / (t_cpu_per_x_hi + t_gpu_per_x_hi);
            (ideal.round() as usize).clamp(G_MIN, G_MAX)
        }

        // -- dispatch ---------------------------------------------------------

        /// Kernel parameter block; layout must match `struct Params` in the
        /// MSL source (eight 32-bit words).
        #[repr(C)]
        struct Params {
            n_lo: u32,
            n_lo_and_inner: u32,
            big_lo_size: u32,
            n_iters: u32,
            n_tiles: u32,
            tile_size: u32,
            within_outer_mask: u32,
            _pad: u32,
        }

        /// In-flight GPU share: command buffer committed, not yet waited on.
        /// Not Send (raw pointers) — created and finished on the same thread,
        /// with the autorelease pool held open in between.
        pub struct Job<'a> {
            pool: *mut c_void,
            cmd: Id,
            out_buf: Id,
            /// Buffers to release in `finish` (includes `out_buf`).
            owned: Vec<Id>,
            n_slots: usize,
            _borrow: PhantomData<&'a ()>,
        }

        /// Encode + commit the GPU share. Returns `None` (and latches the CPU
        /// fallback) on any failure; the caller then runs the full range on
        /// CPU.
        pub fn start_share<'a>(args: ShareArgs<'a>) -> Option<Job<'a>> {
            if args.g == 0 || DISABLED.load(Ordering::Relaxed) || !enabled() {
                return None;
            }
            let ctx = ctx()?;
            debug_assert!(args.g <= args.eq_hi.len());
            debug_assert_eq!(args.inv_table.k, 6);

            unsafe {
                let pool = objc_autoreleasePoolPush();
                let mut owned: Vec<Id> = Vec::with_capacity(8);

                macro_rules! bail {
                    () => {{
                        disable();
                        for b in &owned {
                            msg0_void(*b, ctx.sels.release);
                        }
                        objc_autoreleasePoolPop(pool);
                        return None;
                    }};
                }

                macro_rules! own {
                    ($e:expr) => {{
                        match $e {
                            Some(b) => {
                                owned.push(b);
                                b
                            }
                            None => bail!(),
                        }
                    }};
                }

                let nz = |b: Id| if b.is_null() { None } else { Some(b) };

                // Cached wraps are ctx-lifetime objects and must not be
                // released with the per-dispatch buffers.
                macro_rules! own_wrap {
                    ($e:expr) => {{
                        match $e {
                            Some((b, true)) => b,
                            Some((b, false)) => {
                                owned.push(b);
                                b
                            }
                            None => bail!(),
                        }
                    }};
                }

                let a_buf = own_wrap!(wrap_input(ctx, args.a_packed));
                let b_buf = own_wrap!(wrap_input(ctx, args.b_packed));
                let c_buf = own_wrap!(wrap_input(ctx, args.c_packed));

                // inv-NTT original image: 256 rows × 64 bytes.
                let t0_buf = own!(nz(msg_buf_bytes(
                    ctx.device,
                    ctx.sels.new_buf_bytes,
                    args.inv_table.data_ptr() as *const c_void,
                    256 * ELL,
                    0,
                )));
                let eq_lo_buf = own!(nz(msg_buf_bytes(
                    ctx.device,
                    ctx.sels.new_buf_bytes,
                    args.eq_lo_scaled.as_ptr() as *const c_void,
                    std::mem::size_of_val(args.eq_lo_scaled),
                    0,
                )));
                let eq_hi_buf = own!(nz(msg_buf_bytes(
                    ctx.device,
                    ctx.sels.new_buf_bytes,
                    args.eq_hi.as_ptr() as *const c_void,
                    std::mem::size_of_val(args.eq_hi),
                    0,
                )));
                let counts_buf = own!(nz(msg_buf_bytes(
                    ctx.device,
                    ctx.sels.new_buf_bytes,
                    args.b_med_counts.as_ptr() as *const c_void,
                    args.b_med_counts.len(),
                    0,
                )));

                // Tile geometry: TILE x_outer_lo per threadgroup, N_STREAMS-way
                // interleaved inside the tile.
                let big = args.eq_lo_scaled.len();
                let tile_cap = args
                    .tile_x_outer_lo
                    .unwrap_or(TILE)
                    .max(N_STREAMS)
                    .next_multiple_of(N_STREAMS);
                let (tile_size, n_tiles) = if big >= tile_cap {
                    (tile_cap, big.div_ceil(tile_cap))
                } else {
                    (big.next_multiple_of(N_STREAMS), 1)
                };
                let n_iters = tile_size / N_STREAMS;
                let n_slots = n_tiles * args.g;

                let out_buf = own!(nz(msg_buf_len(
                    ctx.device,
                    ctx.sels.new_buf_len,
                    n_slots * 3 * ELL * 16,
                    0,
                )));

                let params = Params {
                    n_lo: args.n_lo as u32,
                    n_lo_and_inner: args.n_lo_and_inner as u32,
                    big_lo_size: big as u32,
                    n_iters: n_iters as u32,
                    n_tiles: n_tiles as u32,
                    tile_size: tile_size as u32,
                    within_outer_mask: args.within_outer_mask as u32,
                    _pad: 0,
                };

                let cmd = msg0(ctx.queue, ctx.sels.command_buffer);
                if cmd.is_null() {
                    bail!();
                }
                let enc = msg0(cmd, ctx.sels.compute_encoder);
                if enc.is_null() {
                    bail!();
                }
                msg1_void(enc, ctx.sels.set_pso, ctx.urm_pso);
                let bind = |buf: Id, index: usize| {
                    msg_set_buffer(enc, ctx.sels.set_buffer, buf, 0, index);
                };
                bind(a_buf, 0);
                bind(b_buf, 1);
                bind(c_buf, 2);
                bind(t0_buf, 3);
                bind(ctx.convert_buf, 4);
                bind(eq_lo_buf, 5);
                bind(eq_hi_buf, 6);
                bind(counts_buf, 7);
                bind(out_buf, 8);
                msg_set_bytes(
                    enc,
                    ctx.sels.set_bytes,
                    (&params as *const Params) as *const c_void,
                    std::mem::size_of::<Params>(),
                    9,
                );
                msg_dispatch(
                    enc,
                    ctx.sels.dispatch_tg,
                    MTLSize {
                        width: n_tiles,
                        height: args.g,
                        depth: 1,
                    },
                    MTLSize {
                        width: TG_THREADS,
                        height: 1,
                        depth: 1,
                    },
                );
                msg0_void(enc, ctx.sels.end_encoding);
                msg0_void(cmd, ctx.sels.commit);

                Some(Job {
                    pool,
                    cmd,
                    out_buf,
                    owned,
                    n_slots,
                    _borrow: PhantomData,
                })
            }
        }

        impl Job<'_> {
            /// Block until the command buffer completes, XOR-merge the per-
            /// threadgroup partials, release all per-call objects. `None` on
            /// GPU error (caller must recompute the share on CPU); also
            /// latches the permanent CPU fallback in that case.
            pub fn finish(self) -> Option<ShareResult> {
                // MTLCommandBufferStatusCompleted
                const COMPLETED: usize = 4;
                let ctx = ctx().expect("job exists ⇒ ctx exists");
                unsafe {
                    msg0_void(self.cmd, ctx.sels.wait_completed);
                    let status = msg0_usize(self.cmd, ctx.sels.status);
                    let result = if status == COMPLETED {
                        let gpu_seconds = msg0_f64(self.cmd, ctx.sels.gpu_end_time)
                            - msg0_f64(self.cmd, ctx.sels.gpu_start_time);
                        let base = msg0_ptr(self.out_buf, ctx.sels.contents) as *const F128;
                        if base.is_null() {
                            None
                        } else {
                            let mut res_ab = [F128::ZERO; ELL];
                            let mut res_c0 = [F128::ZERO; ELL];
                            let mut res_c1 = [F128::ZERO; ELL];
                            for slot in 0..self.n_slots {
                                let p = base.add(slot * 3 * ELL);
                                for lane in 0..ELL {
                                    res_ab[lane] += *p.add(lane);
                                }
                                for lane in 0..ELL {
                                    res_c0[lane] += *p.add(ELL + lane);
                                }
                                for lane in 0..ELL {
                                    res_c1[lane] += *p.add(2 * ELL + lane);
                                }
                            }
                            Some(ShareResult {
                                res_ab,
                                res_c0,
                                res_c1,
                                gpu_seconds,
                            })
                        }
                    } else {
                        None
                    };
                    for b in &self.owned {
                        msg0_void(*b, ctx.sels.release);
                    }
                    objc_autoreleasePoolPop(self.pool);
                    if result.is_none() {
                        disable();
                    }
                    result
                }
            }
        }
    }

    // =========================================================================
    // BLAKE3 Merkle pipelines.
    // =========================================================================

    pub mod merkle {
        use super::*;
        use crate::merkle::Hash;

        /// Threads per threadgroup for both BLAKE3 kernels.
        pub(super) const TG_THREADS: usize = 256;

        /// Runtime master switch (see [`super::urm::set_enabled`] for the
        /// season-2 default rationale). `FLOCK_NO_GPU_MERKLE` force-kills.
        static MERKLE_ENABLED: AtomicBool = AtomicBool::new(true);

        pub fn set_enabled(on: bool) {
            MERKLE_ENABLED.store(on, Ordering::Relaxed);
        }

        /// Whether the Merkle GPU path can engage: enabled, no failure latch,
        /// usable context (forces the cached init on first call).
        pub fn available() -> bool {
            MERKLE_ENABLED.load(Ordering::Relaxed)
                && !*ENV_NO_GPU
                && !*ENV_NO_MERKLE
                && !DISABLED.load(Ordering::Relaxed)
                && ctx().is_some()
        }

        /// Layout must match `struct LeafParams` in the MSL source.
        #[repr(C)]
        struct LeafParams {
            leaf_lo: u32,
            count: u32,
            words_per_leaf: u32,
            n_blocks: u32,
        }

        /// Layout must match `struct ParentParams` in the MSL source.
        #[repr(C)]
        struct ParentParams {
            src_base: u32,
            dst_base: u32,
            count: u32,
            _pad: u32,
        }

        /// One per-prove Merkle GPU session over a fixed leaf region and a
        /// fixed output tree region (flat `merkle_tree` layout: leaf CVs at
        /// node indices `[0, n)`, next level at `[n, 3n/2)`, …).
        ///
        /// The tree region is bound zero-copy: GPU digests appear directly in
        /// the caller's memory after [`finish`](Session::finish) — and the
        /// CPU may concurrently fill *other* leaf slots of the same region
        /// (the hybrid CPU join), as long as all CPU leaf writes happen
        /// before [`commit_parent_levels`](Session::commit_parent_levels) is
        /// called.
        pub struct Session {
            data_buf: Id,
            data_owned: bool,
            tree_buf: Id,
            tree_owned: bool,
            n_leaves: usize,
            words_per_leaf: usize,
            n_blocks: usize,
            /// Parent levels are emitted while `count >= stop_nodes`.
            stop_nodes: usize,
            /// Committed, retained command buffers, in commit order.
            cmds: Vec<Id>,
            failed: bool,
        }

        // SAFETY: all fields are thread-safe Metal object handles; methods
        // take &mut self, so the session moves between threads but is never
        // used concurrently.
        unsafe impl Send for Session {}

        /// Total nodes the GPU writes for `n` leaves with the given stop
        /// level: leaves plus every parent level with `count >= stop`.
        fn needed_nodes(n_leaves: usize, stop: usize) -> usize {
            let mut total = n_leaves;
            let mut count = n_leaves / 2;
            while count >= stop && count >= 1 {
                total += count;
                count /= 2;
            }
            total
        }

        /// Open a Merkle GPU session.
        ///
        /// * `data` — the full leaf byte region (`n_leaves · leaf_size`
        ///   bytes); `leaf_size` must be a multiple of 64 in `64..=1024`.
        /// * `tree_ptr`/`tree_len` — the output node region (flat
        ///   `merkle_tree` layout). Must be 16 KiB-page-aligned, and
        ///   `floor_page(tree_len · 32)` must cover every node the GPU will
        ///   write (`2n − s_last` nodes, where `s_last` is the smallest level
        ///   with `count ≥ stop_nodes`; just `n` for a leaf-only session).
        ///   Regions ≥ 64 MiB are wrap-cached, so they must be pool-retained
        ///   for the process lifetime.
        /// * `stop_nodes` — parent levels are computed while their node count
        ///   is ≥ `stop_nodes` (the CPU finishes the top). Pass `usize::MAX`
        ///   for a leaf-only session; `n_leaves` must be a power of two when
        ///   parent levels are requested.
        ///
        /// Returns `None` — *without* latching the failure fallback — on any
        /// shape/alignment violation (caller hashes on CPU); Metal API
        /// failures latch [`super::is_disabled`] as usual.
        ///
        /// # Safety
        /// `tree_ptr..tree_ptr + tree_len` must be valid for writes for the
        /// session's lifetime, and the GPU-written node ranges must not be
        /// read or written by the CPU between `commit_*` and `finish`
        /// (untouched slots — e.g. the CPU's leaf-join share — are fine).
        pub unsafe fn begin(
            data: &[u8],
            leaf_size: usize,
            tree_ptr: *mut Hash,
            tree_len: usize,
            stop_nodes: usize,
        ) -> Option<Session> {
            if !available() {
                return None;
            }
            let ctx = ctx()?;
            if leaf_size == 0 || leaf_size % 64 != 0 || leaf_size > 1024 {
                return None;
            }
            if data.is_empty() || data.len() % leaf_size != 0 {
                return None;
            }
            let n_leaves = data.len() / leaf_size;
            let stop = stop_nodes.max(1);
            let do_parents = stop <= n_leaves / 2;
            if do_parents && !n_leaves.is_power_of_two() {
                return None;
            }
            let needed = if do_parents {
                needed_nodes(n_leaves, stop)
            } else {
                n_leaves
            };
            if (tree_ptr as usize) % PAGE != 0 {
                return None;
            }
            let wrapped_bytes = (tree_len * 32) & !(PAGE - 1);
            if wrapped_bytes < needed * 32 {
                return None;
            }

            unsafe {
                let pool = objc_autoreleasePoolPush();
                let data_wrap = wrap_input(ctx, data);
                let Some((data_buf, data_cached)) = data_wrap else {
                    // Shape unsupported (huge misaligned input): CPU fallback
                    // without latching — the checks are cheap to re-run.
                    objc_autoreleasePoolPop(pool);
                    return None;
                };
                let tree_wrap = wrap_nocopy(ctx, tree_ptr as *const u8, wrapped_bytes);
                let Some((tree_buf, tree_cached)) = tree_wrap else {
                    if !data_cached {
                        msg0_void(data_buf, ctx.sels.release);
                    }
                    objc_autoreleasePoolPop(pool);
                    disable();
                    return None;
                };
                objc_autoreleasePoolPop(pool);

                Some(Session {
                    data_buf,
                    data_owned: !data_cached,
                    tree_buf,
                    tree_owned: !tree_cached,
                    n_leaves,
                    words_per_leaf: leaf_size / 4,
                    n_blocks: leaf_size / 64,
                    stop_nodes: stop,
                    cmds: Vec::with_capacity(16),
                    failed: false,
                })
            }
        }

        impl Session {
            pub fn n_leaves(&self) -> usize {
                self.n_leaves
            }

            /// Encode + commit one command buffer hashing leaves
            /// `[leaf_lo, leaf_hi)` into tree slots `[leaf_lo, leaf_hi)`.
            /// Non-blocking. `false` = failure (latched) — the caller must
            /// hash the range (and anything not yet finished) on the CPU.
            pub fn commit_leaves(&mut self, leaf_lo: usize, leaf_hi: usize) -> bool {
                if self.failed || DISABLED.load(Ordering::Relaxed) {
                    return false;
                }
                if leaf_lo >= leaf_hi || leaf_hi > self.n_leaves {
                    debug_assert!(false, "commit_leaves range [{leaf_lo}, {leaf_hi})");
                    return false;
                }
                let ctx = ctx().expect("session exists ⇒ ctx exists");
                let count = leaf_hi - leaf_lo;
                let params = LeafParams {
                    leaf_lo: leaf_lo as u32,
                    count: count as u32,
                    words_per_leaf: self.words_per_leaf as u32,
                    n_blocks: self.n_blocks as u32,
                };
                unsafe {
                    let pool = objc_autoreleasePoolPush();
                    let ok = (|| {
                        let cmd = msg0(ctx.queue, ctx.sels.command_buffer);
                        if cmd.is_null() {
                            return false;
                        }
                        let enc = msg0(cmd, ctx.sels.compute_encoder);
                        if enc.is_null() {
                            return false;
                        }
                        msg1_void(enc, ctx.sels.set_pso, ctx.leaf_pso);
                        msg_set_buffer(enc, ctx.sels.set_buffer, self.data_buf, 0, 0);
                        msg_set_buffer(enc, ctx.sels.set_buffer, self.tree_buf, 0, 1);
                        msg_set_bytes(
                            enc,
                            ctx.sels.set_bytes,
                            (&params as *const LeafParams) as *const c_void,
                            std::mem::size_of::<LeafParams>(),
                            2,
                        );
                        msg_dispatch(
                            enc,
                            ctx.sels.dispatch_tg,
                            MTLSize {
                                width: count.div_ceil(TG_THREADS),
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: TG_THREADS,
                                height: 1,
                                depth: 1,
                            },
                        );
                        msg0_void(enc, ctx.sels.end_encoding);
                        msg0_void(cmd, ctx.sels.commit);
                        // commandBuffer is autoreleased — retain it past the
                        // pool so `finish` (possibly on another thread) can
                        // wait on it.
                        msg0_void(cmd, ctx.sels.retain);
                        self.cmds.push(cmd);
                        true
                    })();
                    objc_autoreleasePoolPop(pool);
                    if !ok {
                        self.failed = true;
                        disable();
                    }
                    ok
                }
            }

            /// Encode + commit one command buffer computing every parent
            /// level with `count >= stop_nodes` (top of the tree is the
            /// CPU's). Call only after every `commit_leaves` call AND after
            /// all CPU-side leaf-join writes into the tree region are done.
            /// Non-blocking. `false` = failure (latched).
            ///
            /// The level dispatches are encoded on one serial compute
            /// encoder, so each level sees the previous level's writes;
            /// cross-command-buffer ordering against the leaf commits is
            /// Metal's same-queue hazard tracking on the shared tree buffer.
            pub fn commit_parent_levels(&mut self) -> bool {
                if self.failed || DISABLED.load(Ordering::Relaxed) {
                    return false;
                }
                if self.stop_nodes > self.n_leaves / 2 {
                    return true; // leaf-only session: nothing to do
                }
                let ctx = ctx().expect("session exists ⇒ ctx exists");
                unsafe {
                    let pool = objc_autoreleasePoolPush();
                    let ok = (|| {
                        let cmd = msg0(ctx.queue, ctx.sels.command_buffer);
                        if cmd.is_null() {
                            return false;
                        }
                        let enc = msg0(cmd, ctx.sels.compute_encoder);
                        if enc.is_null() {
                            return false;
                        }
                        msg1_void(enc, ctx.sels.set_pso, ctx.parent_pso);
                        msg_set_buffer(enc, ctx.sels.set_buffer, self.tree_buf, 0, 0);
                        msg_set_buffer(enc, ctx.sels.set_buffer, self.tree_buf, 0, 1);
                        let mut src_base = 0usize;
                        let mut dst_base = self.n_leaves;
                        let mut count = self.n_leaves / 2;
                        while count >= self.stop_nodes {
                            let params = ParentParams {
                                src_base: src_base as u32,
                                dst_base: dst_base as u32,
                                count: count as u32,
                                _pad: 0,
                            };
                            msg_set_bytes(
                                enc,
                                ctx.sels.set_bytes,
                                (&params as *const ParentParams) as *const c_void,
                                std::mem::size_of::<ParentParams>(),
                                2,
                            );
                            msg_dispatch(
                                enc,
                                ctx.sels.dispatch_tg,
                                MTLSize {
                                    width: count.div_ceil(TG_THREADS),
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: TG_THREADS,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                            src_base = dst_base;
                            dst_base += count;
                            count /= 2;
                        }
                        msg0_void(enc, ctx.sels.end_encoding);
                        msg0_void(cmd, ctx.sels.commit);
                        msg0_void(cmd, ctx.sels.retain);
                        self.cmds.push(cmd);
                        true
                    })();
                    objc_autoreleasePoolPop(pool);
                    if !ok {
                        self.failed = true;
                        disable();
                    }
                    ok
                }
            }

            /// Wait for every committed command buffer and verify completion.
            /// On success returns the summed command-buffer GPU time in
            /// seconds (for calibration) — the digests are already sitting in
            /// the caller's tree memory. `None` = failure (latched): the
            /// caller must rebuild the affected tree ranges on the CPU.
            pub fn finish(self) -> Option<f64> {
                // MTLCommandBufferStatusCompleted
                const COMPLETED: usize = 4;
                let ctx = ctx().expect("session exists ⇒ ctx exists");
                let mut ok = !self.failed;
                let mut gpu_seconds = 0.0f64;
                unsafe {
                    for &cmd in &self.cmds {
                        msg0_void(cmd, ctx.sels.wait_completed);
                        if msg0_usize(cmd, ctx.sels.status) == COMPLETED {
                            gpu_seconds += msg0_f64(cmd, ctx.sels.gpu_end_time)
                                - msg0_f64(cmd, ctx.sels.gpu_start_time);
                        } else {
                            ok = false;
                        }
                    }
                }
                // Drop releases cmds + owned buffers.
                drop(self);
                if ok {
                    Some(gpu_seconds)
                } else {
                    disable();
                    None
                }
            }
        }

        impl Drop for Session {
            fn drop(&mut self) {
                let Some(ctx) = ctx() else { return };
                unsafe {
                    // In-flight command buffers hold their own references to
                    // everything they use (the queue retains committed
                    // buffers until completion, encoders retain bound
                    // resources), so dropping our references early is safe
                    // even if `finish` was never called.
                    for &cmd in &self.cmds {
                        msg0_void(cmd, ctx.sels.release);
                    }
                    if self.data_owned {
                        msg0_void(self.data_buf, ctx.sels.release);
                    }
                    if self.tree_owned {
                        msg0_void(self.tree_buf, ctx.sels.release);
                    }
                }
            }
        }
    }

    // -- embedded MSL ---------------------------------------------------------
    //
    // One library, three kernels. `urm_round1` is the season-1 URM shader,
    // byte-for-byte. `blake3_leaf` / `blake3_parent` implement the exact
    // non-root chaining-value semantics of `crate::merkle` (see module docs).

    const SHADER_SOURCE_C: &std::ffi::CStr = cr#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n_lo;
    uint n_lo_and_inner;
    uint big_lo_size;
    uint n_iters;
    uint n_tiles;
    uint tile_size;
    uint within_outer_mask;
    uint pad_;
};

// GF(2^8), AES polynomial x^8+x^4+x^3+x+1: reduce a <= 16-bit carry-less
// product (exact port of flock's gf8_reduce).
inline uint gf8_reduce16(uint p) {
    uint h = p >> 8;
    uint t = (p & 0xffu) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4);
    uint h2 = t >> 8;
    return (t & 0xffu) ^ h2 ^ (h2 << 1) ^ (h2 << 3) ^ (h2 << 4);
}

// Reduced GF(2^8) multiply: bitwise clmul8 then fold.
inline uint gf8_mul(uint a, uint b) {
    uint p = 0;
    for (uint i = 0; i < 8; i++) {
        p ^= (a << i) * ((b >> i) & 1u);
    }
    return gf8_reduce16(p);
}

// 32x32 -> 64 carry-less multiply via 4-way masked integer multiplies
// (BearSSL bmul trick; column counts <= 8 fit the 4-bit gaps, so the
// integer adds never carry across groups and the mask extracts the parity).
inline ulong bmul32(uint x, uint y) {
    ulong x0 = ulong(x & 0x11111111u), x1 = ulong(x & 0x22222222u),
          x2 = ulong(x & 0x44444444u), x3 = ulong(x & 0x88888888u);
    ulong y0 = ulong(y & 0x11111111u), y1 = ulong(y & 0x22222222u),
          y2 = ulong(y & 0x44444444u), y3 = ulong(y & 0x88888888u);
    ulong z0 = (x0 * y0) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1);
    ulong z1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y3) ^ (x3 * y2);
    ulong z2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y3);
    ulong z3 = (x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0);
    return (z0 & 0x1111111111111111ul) | (z1 & 0x2222222222222222ul)
         | (z2 & 0x4444444444444444ul) | (z3 & 0x8888888888888888ul);
}

// 64x64 -> 128 carry-less multiply, Karatsuba over bmul32.
inline ulong2 clmul64(ulong a, ulong b) {
    uint a0 = uint(a & 0xfffffffful), a1 = uint(a >> 32);
    uint b0 = uint(b & 0xfffffffful), b1 = uint(b >> 32);
    ulong p0 = bmul32(a0, b0);
    ulong p2 = bmul32(a1, b1);
    ulong pm = bmul32(a0 ^ a1, b0 ^ b1) ^ p0 ^ p2;
    return ulong2(p0 ^ (pm << 32), p2 ^ (pm >> 32));
}

// 256-bit unreduced GHASH schoolbook product (r0,r1,r2,r3), exact port of
// ghash_mul_unreduced.
inline ulong4 mul_unred(ulong2 a, ulong2 b) {
    ulong2 ll = clmul64(a.x, b.x);
    ulong2 lh = clmul64(a.x, b.y);
    ulong2 hl = clmul64(a.y, b.x);
    ulong2 hh = clmul64(a.y, b.y);
    return ulong4(ll.x, ll.y ^ lh.x ^ hl.x, hh.x ^ lh.y ^ hl.y, hh.y);
}

// Reduce mod x^128 + x^7 + x^2 + x + 1 (exact port of ghash_reduce).
inline ulong2 ghash_red(ulong4 r) {
    ulong tlo = r.z ^ (r.z << 1) ^ (r.z << 2) ^ (r.z << 7);
    ulong thi = r.w ^ ((r.w << 1) | (r.z >> 63)) ^ ((r.w << 2) | (r.z >> 62))
              ^ ((r.w << 7) | (r.z >> 57));
    ulong ov = (r.w >> 63) ^ (r.w >> 62) ^ (r.w >> 57);
    ulong corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    return ulong2(r.x ^ tlo ^ corr, r.y ^ thi);
}

// One threadgroup = 8 streams x 64 lanes, one (x_hi, x_outer_lo tile).
// Grid: (n_tiles, g) threadgroups.
kernel void urm_round1(
    device const uchar*  a      [[buffer(0)]],
    device const uchar*  b      [[buffer(1)]],
    device const uchar*  c      [[buffer(2)]],
    device const ulong2* t0src  [[buffer(3)]],
    constant ulong2*     conv   [[buffer(4)]],
    device const ulong2* eq_lo  [[buffer(5)]],
    device const ulong2* eq_hi  [[buffer(6)]],
    device const uchar*  counts [[buffer(7)]],
    device ulong2*       out    [[buffer(8)]],
    constant Params&     P      [[buffer(9)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
    // 16 KiB inv-NTT byte table, cooperatively loaded; reused as the
    // stream-combine scratch after the main loop.
    threadgroup ulong2 t0mem[1024];
    for (uint i = tid; i < 1024u; i += 512u) {
        t0mem[i] = t0src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const uchar* T0 = (threadgroup const uchar*)t0mem;

    const uint lane = tid & 63u;
    const uint stream = tid >> 6;
    const uint x_hi = tg.y;
    const uint tile_base = tg.x * P.tile_size;
    // bit_transpose: out[lane] bit K = window byte [K*8 + (lane>>3)] bit (lane&7).
    const uint csh = ((lane >> 3) << 3) | (lane & 7u);

    ulong4 acc_ab = ulong4(0), acc_c0 = ulong4(0), acc_c1 = ulong4(0);

    for (uint it = 0; it < P.n_iters; it++) {
        uint xol = tile_base + it * 8u + stream;
        if (xol >= P.big_lo_size) {
            continue;
        }
        uint n_b_med = uint(counts[(xol | (x_hi << P.n_lo)) & P.within_outer_mask]);
        if (n_b_med == 0u) {
            continue;
        }
        ulong base = ((ulong(xol) << 7) | (ulong(x_hi) << P.n_lo_and_inner)) << 3;
        ulong2 conv_ab = ulong2(0), conv_c0 = ulong2(0), conv_c1 = ulong2(0);
        for (uint bm = 0; bm < n_b_med; bm++) {
            ulong wnd = base + ulong(bm << 6);
            device const ulong* aq = (device const ulong*)(a + wnd);
            device const ulong* bq = (device const ulong*)(b + wnd);
            device const ulong* cq = (device const ulong*)(c + wnd);
            // shift_reduce_inner_ab for this lane.
            uint acc = 0;
            for (uint K = 0; K < 8; K++) {
                ulong aw = aq[K], bw = bq[K];
                uint av = 0, bv = 0;
                for (uint bb = 0; bb < 8; bb++) {
                    uint idx = lane ^ (bb << 3);
                    av ^= uint(T0[(uint((aw >> (bb << 3)) & 0xfful) << 6) | idx]);
                    bv ^= uint(T0[(uint((bw >> (bb << 3)) & 0xfful) << 6) | idx]);
                }
                acc ^= gf8_mul(av, bv) << K;
            }
            uint abv = gf8_reduce16(acc);
            // bit_transpose_64bytes for this lane.
            uint cb = 0;
            for (uint K = 0; K < 8; K++) {
                cb |= uint((cq[K] >> csh) & 1ul) << K;
            }
            // Convert-table accumulation (plain 128-bit XOR).
            uint cbase = bm << 8;
            conv_ab ^= conv[cbase | abv];
            conv_c0 ^= conv[cbase | (cb & 0x55u)];
            conv_c1 ^= conv[cbase | (cb & 0xAAu)];
        }
        // eq_lo fold: XOR the 256-bit unreduced products (deferred reduction).
        ulong2 e = eq_lo[xol];
        acc_ab ^= mul_unred(conv_ab, e);
        acc_c0 ^= mul_unred(conv_c0, e);
        acc_c1 ^= mul_unred(conv_c1, e);
    }

    // Reduce once per bank, then the eq_hi outer fold (reduced multiply).
    ulong2 ehv = eq_hi[x_hi];
    ulong2 rab = ghash_red(mul_unred(ghash_red(acc_ab), ehv));
    ulong2 rc0 = ghash_red(mul_unred(ghash_red(acc_c0), ehv));
    ulong2 rc1 = ghash_red(mul_unred(ghash_red(acc_c1), ehv));

    // XOR-combine the 8 streams through threadgroup memory (t0mem is free
    // now); one 3 x 64 x 16 B slot per threadgroup.
    threadgroup ulong2* scratch = t0mem;
    ulong out_base = ulong(tg.y * P.n_tiles + tg.x) * 192ul;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = rab;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64u) {
        ulong2 v = scratch[tid];
        for (uint s = 1; s < 8u; s++) { v ^= scratch[(s << 6) | tid]; }
        out[out_base + ulong(tid)] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = rc0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64u) {
        ulong2 v = scratch[tid];
        for (uint s = 1; s < 8u; s++) { v ^= scratch[(s << 6) | tid]; }
        out[out_base + 64ul + ulong(tid)] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = rc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64u) {
        ulong2 v = scratch[tid];
        for (uint s = 1; s < 8u; s++) { v ^= scratch[(s << 6) | tid]; }
        out[out_base + 128ul + ulong(tid)] = v;
    }
}

// ---------------------------------------------------------------------------
// BLAKE3 Merkle kernels.
//
// Exact non-root chaining-value semantics (spec constants; matches
// crate::merkle::{blake3_leaf_cv, blake3_parent_cv} bit for bit):
//   leaf: single chunk (counter 0), 64-byte whole blocks chained from IV,
//         flags CHUNK_START on block 0 and CHUNK_END on the last block,
//         output = truncated (8-word) non-root CV;
//   parent: one block = left_cv || right_cv, state IV, flags PARENT.
// No ROOT flag anywhere. All block lengths are exactly 64 (leaf sizes are
// multiples of 64), so b=64 always.
// ---------------------------------------------------------------------------

constant uint B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// Per-round message word schedule (BLAKE3 permutation, unrolled).
constant uchar B3_SCHED[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

inline uint b3_rotr(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

inline void b3_g(thread uint* v, uint a, uint b, uint c, uint d, uint mx, uint my) {
    v[a] = v[a] + v[b] + mx;
    v[d] = b3_rotr(v[d] ^ v[a], 16u);
    v[c] = v[c] + v[d];
    v[b] = b3_rotr(v[b] ^ v[c], 12u);
    v[a] = v[a] + v[b] + my;
    v[d] = b3_rotr(v[d] ^ v[a], 8u);
    v[c] = v[c] + v[d];
    v[b] = b3_rotr(v[b] ^ v[c], 7u);
}

// One whole-block compression (counter 0, block_len 64), truncated output:
// cv <- first 8 words of the compressed state.
inline void b3_compress(thread uint* cv, thread const uint* m, uint flags) {
    uint v[16];
    for (uint i = 0; i < 8u; i++) {
        v[i] = cv[i];
    }
    v[8] = B3_IV[0];
    v[9] = B3_IV[1];
    v[10] = B3_IV[2];
    v[11] = B3_IV[3];
    v[12] = 0u;      // counter lo (always chunk 0 / parent)
    v[13] = 0u;      // counter hi
    v[14] = 64u;     // block length (whole blocks only)
    v[15] = flags;
    for (uint r = 0; r < 7u; r++) {
        constant const uchar* s = B3_SCHED[r];
        b3_g(v, 0u, 4u, 8u, 12u, m[s[0]], m[s[1]]);
        b3_g(v, 1u, 5u, 9u, 13u, m[s[2]], m[s[3]]);
        b3_g(v, 2u, 6u, 10u, 14u, m[s[4]], m[s[5]]);
        b3_g(v, 3u, 7u, 11u, 15u, m[s[6]], m[s[7]]);
        b3_g(v, 0u, 5u, 10u, 15u, m[s[8]], m[s[9]]);
        b3_g(v, 1u, 6u, 11u, 12u, m[s[10]], m[s[11]]);
        b3_g(v, 2u, 7u, 8u, 13u, m[s[12]], m[s[13]]);
        b3_g(v, 3u, 4u, 9u, 14u, m[s[14]], m[s[15]]);
    }
    for (uint i = 0; i < 8u; i++) {
        cv[i] = v[i] ^ v[i + 8u];
    }
}

struct LeafParams {
    uint leaf_lo;
    uint count;
    uint words_per_leaf;
    uint n_blocks;
};

// One thread per leaf: chain n_blocks whole-block compressions, write the
// 32-byte non-root CV into node slot `leaf` (flat merkle_tree layout).
kernel void blake3_leaf(
    device const uint* data  [[buffer(0)]],
    device uint*       nodes [[buffer(1)]],
    constant LeafParams& P   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= P.count) {
        return;
    }
    ulong leaf = ulong(P.leaf_lo) + ulong(gid);
    device const uint* src = data + leaf * ulong(P.words_per_leaf);
    uint cv[8];
    for (uint i = 0; i < 8u; i++) {
        cv[i] = B3_IV[i];
    }
    for (uint blk = 0; blk < P.n_blocks; blk++) {
        uint m[16];
        for (uint j = 0; j < 16u; j++) {
            m[j] = src[blk * 16u + j];
        }
        uint flags = (blk == 0u ? 1u : 0u)                 // CHUNK_START
                   | (blk + 1u == P.n_blocks ? 2u : 0u);   // CHUNK_END
        b3_compress(cv, m, flags);
    }
    device uint* dst = nodes + leaf * 8ul;
    for (uint i = 0; i < 8u; i++) {
        dst[i] = cv[i];
    }
}

struct ParentParams {
    uint src_base;
    uint dst_base;
    uint count;
    uint pad_;
};

// One thread per parent node: compress left_cv || right_cv with the PARENT
// flag from the IV state (merge_subtrees_non_root, Mode::Hash).
kernel void blake3_parent(
    device const uint* nodes_in  [[buffer(0)]],
    device uint*       nodes_out [[buffer(1)]],
    constant ParentParams& P     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= P.count) {
        return;
    }
    device const uint* src = nodes_in + (ulong(P.src_base) + 2ul * ulong(gid)) * 8ul;
    uint m[16];
    for (uint j = 0; j < 16u; j++) {
        m[j] = src[j];
    }
    uint cv[8];
    for (uint i = 0; i < 8u; i++) {
        cv[i] = B3_IV[i];
    }
    b3_compress(cv, m, 4u);   // PARENT
    device uint* dst = nodes_out + (ulong(P.dst_base) + ulong(gid)) * 8ul;
    for (uint i = 0; i < 8u; i++) {
        dst[i] = cv[i];
    }
}
"#;
}

// ---------------------------------------------------------------------------
// Tests (Apple targets; every GPU test SKIPS when Metal is unavailable).
// ---------------------------------------------------------------------------

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
    use super::{merkle, urm};
    use crate::field::F128;
    use crate::hash::HashKind;
    use crate::merkle::{Hash, hash_leaf, hash_pair};
    use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
    use crate::zerocheck::univariate_skip::{SplitEqGhash, pack_bits};
    use crate::zerocheck::univariate_skip_optimized::{
        K_SKIP, N_INNER, WorkerStateWithSHatV, build_b_med_counts, convert_table, d_inv,
        medium_challenges_ghash, process_one_x_hi_with_s_hat_v, small_challenges_ghash,
    };
    use crate::zerocheck::PaddingSpec;
    use crate::field::F8;

    // -- shared helpers -------------------------------------------------------

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bit(&mut self) -> bool {
            (self.next_u64() & 1) != 0
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.bit()).collect()
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(n);
            while out.len() < n {
                out.extend_from_slice(&self.next_u64().to_le_bytes());
            }
            out.truncate(n);
            out
        }
    }

    const PAGE: usize = 16 * 1024;

    /// Page-aligned byte region (16 KiB alignment, page-multiple length).
    struct PageBuf {
        ptr: *mut u8,
        bytes: usize,
    }

    impl PageBuf {
        fn zeroed(min_bytes: usize) -> Self {
            let bytes = min_bytes.next_multiple_of(PAGE).max(PAGE);
            let layout = std::alloc::Layout::from_size_align(bytes, PAGE).unwrap();
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Self { ptr, bytes }
        }
        fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) }
        }
        fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
        }
        fn hash_ptr(&self) -> *mut Hash {
            self.ptr as *mut Hash
        }
        fn hash_len(&self) -> usize {
            self.bytes / 32
        }
        fn hashes(&self) -> &[Hash] {
            unsafe { std::slice::from_raw_parts(self.ptr as *const Hash, self.bytes / 32) }
        }
        fn hashes_mut(&mut self) -> &mut [Hash] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut Hash, self.bytes / 32) }
        }
    }

    impl Drop for PageBuf {
        fn drop(&mut self) {
            let layout = std::alloc::Layout::from_size_align(self.bytes, PAGE).unwrap();
            unsafe { std::alloc::dealloc(self.ptr, layout) };
        }
    }

    /// CPU reference: full flat tree (leaf CVs, then levels up to the root)
    /// via the spec primitives.
    fn cpu_tree(data: &[u8], leaf_size: usize) -> Vec<Hash> {
        let n = data.len() / leaf_size;
        assert!(n.is_power_of_two());
        let mut tree = vec![[0u8; 32]; 2 * n - 1];
        for i in 0..n {
            tree[i] = hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], HashKind::Blake3);
        }
        let mut src = 0usize;
        let mut dst = n;
        let mut count = n / 2;
        while count >= 1 {
            for i in 0..count {
                let (l, r) = (tree[src + 2 * i], tree[src + 2 * i + 1]);
                tree[dst + i] = hash_pair(&l, &r, HashKind::Blake3);
            }
            src = dst;
            dst += count;
            count /= 2;
        }
        tree
    }

    // -- Merkle: leaf CV equality --------------------------------------------

    /// **GPU BLAKE3 leaf cross-check**: leaf chaining values for every
    /// (leaf count, leaf size) shape must equal `hash_leaf(_, Blake3)`
    /// byte for byte — including multi-command-buffer streaming commits.
    #[test]
    fn merkle_leaf_cvs_match_cpu() {
        if !super::metal_available() {
            eprintln!("skipping: no Metal device/pipeline available");
            return;
        }
        let mut rng = Rng::new(0x6c65_6166_u64);
        for &n in &[1usize, 5, 64, 1000, 4096] {
            for &leaf_size in &[64usize, 256, 1024] {
                let data = rng.bytes(n * leaf_size);
                let tree_len = (2 * n).next_multiple_of(512).max(512);
                let tree = PageBuf::zeroed(tree_len * 32);

                let mut s = unsafe {
                    merkle::begin(&data, leaf_size, tree.hash_ptr(), tree.hash_len(), usize::MAX)
                }
                .expect("begin must succeed when Metal is available");
                // Split into two commits to exercise multi-cb streaming.
                let mid = n / 2;
                if mid > 0 {
                    assert!(s.commit_leaves(0, mid), "commit_leaves lo n={n} ls={leaf_size}");
                    assert!(s.commit_leaves(mid, n), "commit_leaves hi n={n} ls={leaf_size}");
                } else {
                    assert!(s.commit_leaves(0, n), "commit_leaves n={n} ls={leaf_size}");
                }
                s.finish()
                    .expect("GPU leaf session must complete");

                for i in 0..n {
                    let want =
                        hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], HashKind::Blake3);
                    assert_eq!(
                        tree.hashes()[i],
                        want,
                        "leaf CV mismatch at i={i}, n={n}, leaf_size={leaf_size}"
                    );
                }
                assert!(!super::is_disabled(), "GPU latched during leaf test");
            }
        }
    }

    // -- Merkle: parent levels + full tree -------------------------------------

    /// **GPU BLAKE3 full-tree cross-check**: leaves + all parent levels down
    /// to the root (`stop_nodes = 1`) must reproduce the CPU
    /// `hash_leaf`/`hash_pair` tree exactly, for several sizes and leaf
    /// shapes.
    #[test]
    fn merkle_full_tree_matches_cpu() {
        if !super::metal_available() {
            eprintln!("skipping: no Metal device/pipeline available");
            return;
        }
        let mut rng = Rng::new(0x7472_6565_u64);
        for &(n, leaf_size) in &[(64usize, 256usize), (1024, 64), (4096, 1024)] {
            let data = rng.bytes(n * leaf_size);
            let want = cpu_tree(&data, leaf_size);
            let tree_len = (2 * n).next_multiple_of(512).max(512);
            let tree = PageBuf::zeroed(tree_len * 32);

            let mut s = unsafe {
                merkle::begin(&data, leaf_size, tree.hash_ptr(), tree.hash_len(), 1)
            }
            .expect("begin must succeed when Metal is available");
            assert!(s.commit_leaves(0, n));
            assert!(s.commit_parent_levels());
            s.finish().expect("GPU tree session must complete");

            assert_eq!(
                &tree.hashes()[..2 * n - 1],
                &want[..],
                "tree mismatch at n={n}, leaf_size={leaf_size}"
            );
        }
    }

    /// **GPU/CPU hybrid join cross-check** (the production commit.rs shape):
    /// GPU hashes the first half of the leaves, the CPU writes the second
    /// half of the leaf CVs directly into the shared tree memory, then the
    /// GPU computes parent levels down to `stop_nodes = 16` and the CPU
    /// finishes the top — the result must equal the pure-CPU tree.
    #[test]
    fn merkle_hybrid_cpu_join_matches_cpu() {
        if !super::metal_available() {
            eprintln!("skipping: no Metal device/pipeline available");
            return;
        }
        let (n, leaf_size, stop) = (2048usize, 1024usize, 16usize);
        let mut rng = Rng::new(0x6a6f_696e_u64);
        let data = rng.bytes(n * leaf_size);
        let want = cpu_tree(&data, leaf_size);
        let tree_len = (2 * n).next_multiple_of(512);
        let mut tree = PageBuf::zeroed(tree_len * 32);

        let mut s =
            unsafe { merkle::begin(&data, leaf_size, tree.hash_ptr(), tree.hash_len(), stop) }
                .expect("begin must succeed when Metal is available");
        // GPU: leaves [0, n/2).
        assert!(s.commit_leaves(0, n / 2));
        // CPU join: leaves [n/2, n) written straight into the tree region.
        for i in n / 2..n {
            tree.hashes_mut()[i] =
                hash_leaf(&data[i * leaf_size..(i + 1) * leaf_size], HashKind::Blake3);
        }
        // GPU: parent levels while count >= 16 (n .. 16), CPU takes the top.
        assert!(s.commit_parent_levels());
        s.finish().expect("GPU hybrid session must complete");

        // GPU wrote nodes [0, 2n - 16); CPU finishes levels 8, 4, 2, 1.
        {
            let nodes = tree.hashes_mut();
            let mut src = 2 * n - 32; // start of the 16-node level
            let mut dst = 2 * n - 16;
            let mut count = 8usize;
            while count >= 1 {
                for i in 0..count {
                    let (l, r) = (nodes[src + 2 * i], nodes[src + 2 * i + 1]);
                    nodes[dst + i] = hash_pair(&l, &r, HashKind::Blake3);
                }
                src = dst;
                dst += count;
                count /= 2;
            }
        }
        assert_eq!(
            &tree.hashes()[..2 * n - 1],
            &want[..],
            "hybrid tree mismatch"
        );
        assert!(!super::is_disabled(), "GPU latched during hybrid test");
    }

    /// A partial-stop session must leave the CPU-owned top slots untouched.
    #[test]
    fn merkle_partial_stop_leaves_top_untouched() {
        if !super::metal_available() {
            eprintln!("skipping: no Metal device/pipeline available");
            return;
        }
        let (n, leaf_size, stop) = (256usize, 64usize, 16usize);
        let mut rng = Rng::new(0x746f_70_u64);
        let data = rng.bytes(n * leaf_size);
        let tree_len = (2 * n).next_multiple_of(512);
        let mut tree = PageBuf::zeroed(tree_len * 32);
        let sentinel = [0xAAu8; 32];
        let gpu_nodes = 2 * n - stop; // leaves + levels 128..16
        for slot in tree.hashes_mut()[gpu_nodes..2 * n - 1].iter_mut() {
            *slot = sentinel;
        }

        let mut s =
            unsafe { merkle::begin(&data, leaf_size, tree.hash_ptr(), tree.hash_len(), stop) }
                .expect("begin must succeed when Metal is available");
        assert!(s.commit_leaves(0, n));
        assert!(s.commit_parent_levels());
        s.finish().expect("GPU session must complete");

        let want = cpu_tree(&data, leaf_size);
        assert_eq!(&tree.hashes()[..gpu_nodes], &want[..gpu_nodes]);
        for (i, slot) in tree.hashes()[gpu_nodes..2 * n - 1].iter().enumerate() {
            assert_eq!(*slot, sentinel, "top slot {i} was written by the GPU");
        }
    }

    /// Shape violations must return `None` from `begin` WITHOUT latching the
    /// global failure fallback.
    #[test]
    fn merkle_begin_shape_checks_do_not_latch() {
        if !super::metal_available() {
            eprintln!("skipping: no Metal device/pipeline available");
            return;
        }
        let tree = PageBuf::zeroed(PAGE);
        let data = vec![0u8; 4096];
        // Bad leaf sizes.
        for &ls in &[0usize, 32, 96 + 1, 2048] {
            assert!(
                unsafe { merkle::begin(&data, ls, tree.hash_ptr(), tree.hash_len(), 1) }
                    .is_none()
            );
        }
        // Non-multiple data length.
        assert!(
            unsafe { merkle::begin(&data[..1000], 64, tree.hash_ptr(), tree.hash_len(), 1) }
                .is_none()
        );
        // Misaligned tree pointer.
        let misaligned = unsafe { (tree.ptr.add(32)) as *mut Hash };
        assert!(
            unsafe { merkle::begin(&data, 64, misaligned, tree.hash_len() - 1, usize::MAX) }
                .is_none()
        );
        // Tree region too small for the requested levels (64 leaves need
        // 127 nodes to the root; wrapped region of 0 pages covers none).
        let tiny = PageBuf::zeroed(PAGE);
        assert!(
            unsafe { merkle::begin(&data, 64, tiny.hash_ptr(), 100, 1) }.is_none()
        );
        // Non-power-of-two leaves with parent levels requested.
        let data3 = vec![0u8; 3 * 64];
        assert!(
            unsafe { merkle::begin(&data3, 64, tree.hash_ptr(), tree.hash_len(), 1) }.is_none()
        );
        assert!(
            !super::is_disabled(),
            "shape checks must not latch the failure fallback"
        );
    }

    // -- URM: partial equality (season-1 tests, ported) ------------------------

    fn build_protocol_r(m: usize, outer: &[F128]) -> Vec<F128> {
        assert_eq!(outer.len(), m - K_SKIP - N_INNER);
        let mut r = vec![F128::ZERO; m];
        for (i, &small) in small_challenges_ghash().iter().enumerate() {
            r[K_SKIP + i] = small;
        }
        for (i, &med) in medium_challenges_ghash().iter().enumerate() {
            r[K_SKIP + 3 + i] = med;
        }
        for (i, &x) in outer.iter().enumerate() {
            r[K_SKIP + N_INNER + i] = x;
        }
        r
    }

    fn make_inv_table() -> InvNttTableByteSingleGf8 {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
    }

    /// **GPU single-x_hi cross-check**: the GPU share for `g = 1` must equal
    /// the CPU `process_one_x_hi_with_s_hat_v` eq_hi-folded accumulators for
    /// `x_hi = 0`, bank by bank, bit for bit.
    #[test]
    fn urm_one_x_hi_matches_cpu_partials() {
        if !super::metal_available() {
            eprintln!("skipping: no Metal device/pipeline available");
            return;
        }
        let m = 16usize;
        let mut rng = Rng::new(0x6b7531_u64);
        let a = pack_bits(&rng.bits(1 << m));
        let b = pack_bits(&rng.bits(1 << m));
        let c = pack_bits(&rng.bits(1 << m));
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let inv_table = make_inv_table();
        let padding = PaddingSpec::dense(m);

        let eq = SplitEqGhash::new(&r[K_SKIP + N_INNER..]);
        let d_inv_val = d_inv();
        let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
        let convert = convert_table();
        let (within_outer_mask, b_med_counts) = build_b_med_counts(&padding);

        // CPU: one x_hi (= 0) into a fresh state ⇒ `local_res_*` hold exactly
        // that x_hi's eq_hi-folded partials.
        let mut state = WorkerStateWithSHatV::new();
        process_one_x_hi_with_s_hat_v(
            0,
            1 << eq.n_lo,
            eq.n_lo + N_INNER,
            within_outer_mask,
            &b_med_counts,
            &a,
            &b,
            &c,
            &inv_table,
            &eq_lo_scaled,
            eq.hi[0],
            convert,
            &mut state,
        );

        // GPU: g = 1 covers x_hi = 0 only.
        let job = urm::start_share(urm::ShareArgs {
            a_packed: &a,
            b_packed: &b,
            c_packed: &c,
            inv_table: &inv_table,
            eq_lo_scaled: &eq_lo_scaled,
            eq_hi: &eq.hi,
            b_med_counts: &b_med_counts,
            within_outer_mask,
            n_lo: eq.n_lo,
            n_lo_and_inner: eq.n_lo + N_INNER,
            g: 1,
            tile_x_outer_lo: None,
        })
        .expect("start_share must succeed when Metal is available");
        let got = job.finish().expect("GPU share must complete");

        assert_eq!(got.res_ab, state.local_res_ab, "AB bank mismatch");
        assert_eq!(got.res_c0, state.local_res_c_s_0, "C bank 0 mismatch");
        assert_eq!(got.res_c1, state.local_res_c_s_1, "C bank 1 mismatch");
    }

    /// CPU reference for a whole-range GPU share: serial accumulation of
    /// `process_one_x_hi_with_s_hat_v` over all x_hi (local_res_* persist
    /// across calls, mirroring a single rayon worker owning every x_hi).
    fn cpu_reference_all_x_hi(
        a: &[u8],
        b: &[u8],
        c: &[u8],
        r: &[F128],
        inv_table: &InvNttTableByteSingleGf8,
        padding: &PaddingSpec,
    ) -> WorkerStateWithSHatV {
        let eq = SplitEqGhash::new(&r[K_SKIP + N_INNER..]);
        let d_inv_val = d_inv();
        let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
        let convert = convert_table();
        let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
        let mut state = WorkerStateWithSHatV::new();
        for x_hi in 0..(1usize << eq.n_hi) {
            process_one_x_hi_with_s_hat_v(
                x_hi,
                1 << eq.n_lo,
                eq.n_lo + N_INNER,
                within_outer_mask,
                &b_med_counts,
                a,
                b,
                c,
                inv_table,
                &eq_lo_scaled,
                eq.hi[x_hi],
                convert,
                &mut state,
            );
        }
        state
    }

    /// Run the full x_hi range on the GPU with the given tile override and
    /// compare against the serial CPU reference.
    fn assert_gpu_full_range_matches(
        m: usize,
        padding: &PaddingSpec,
        tile_x_outer_lo: Option<usize>,
        a: &[u8],
        b: &[u8],
        c: &[u8],
        r: &[F128],
        inv_table: &InvNttTableByteSingleGf8,
    ) {
        let eq = SplitEqGhash::new(&r[K_SKIP + N_INNER..]);
        let d_inv_val = d_inv();
        let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
        let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);

        let job = urm::start_share(urm::ShareArgs {
            a_packed: a,
            b_packed: b,
            c_packed: c,
            inv_table,
            eq_lo_scaled: &eq_lo_scaled,
            eq_hi: &eq.hi,
            b_med_counts: &b_med_counts,
            within_outer_mask,
            n_lo: eq.n_lo,
            n_lo_and_inner: eq.n_lo + N_INNER,
            g: 1 << eq.n_hi,
            tile_x_outer_lo,
        })
        .expect("start_share must succeed when Metal is available");
        let got = job.finish().expect("GPU share must complete");

        let state = cpu_reference_all_x_hi(a, b, c, r, inv_table, padding);
        assert_eq!(
            got.res_ab, state.local_res_ab,
            "AB mismatch at m={m}, tile={tile_x_outer_lo:?}"
        );
        assert_eq!(
            got.res_c0, state.local_res_c_s_0,
            "C bank 0 mismatch at m={m}, tile={tile_x_outer_lo:?}"
        );
        assert_eq!(
            got.res_c1, state.local_res_c_s_1,
            "C bank 1 mismatch at m={m}, tile={tile_x_outer_lo:?}"
        );
    }

    /// **GPU production-geometry cross-check**: m = 24 gives n_lo = 4
    /// (big_lo_size = 16 > stream count) and BLAKE3-style padding gives
    /// `b_med_counts = [16, 15]` with mask 1 — exercising the stream
    /// iteration loop, the odd-window b_med skip, x_outer parity via the
    /// counts mask, and (with the tile override) a multi-tile grid with
    /// per-tile output slots. Both configurations must match the CPU bit
    /// for bit.
    #[test]
    fn urm_multi_tile_padded_matches_cpu() {
        if !super::metal_available() {
            eprintln!("skipping: no Metal device/pipeline available");
            return;
        }
        let m = 24usize;
        let mut rng = Rng::new(0x7469_6c65_u64);
        let a = pack_bits(&rng.bits(1 << m));
        let b = pack_bits(&rng.bits(1 << m));
        let c = pack_bits(&rng.bits(1 << m));
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let inv_table = make_inv_table();
        // Ranked-BLAKE3 padding shape: counts = [16, 15], mask = 1.
        let padding = PaddingSpec {
            k_log: 14,
            useful_bits_per_block: 15_409,
        };

        // Default tile (16 x_outer_lo, 1 tile, 2 stream iterations).
        assert_gpu_full_range_matches(m, &padding, None, &a, &b, &c, &r, &inv_table);
        // Forced 2-tile grid (8 x_outer_lo per tile).
        assert_gpu_full_range_matches(m, &padding, Some(8), &a, &b, &c, &r, &inv_table);
    }

    /// **GPU zero-copy wrap cross-check**: page-aligned, page-multiple
    /// witness buffers must take the `newBufferWithBytesNoCopy` path (the
    /// production m = 32 route — 512 MiB per array is too large to copy)
    /// and still produce bit-identical partials.
    #[test]
    fn urm_nocopy_wrap_matches_cpu() {
        use std::sync::atomic::Ordering;

        if !super::metal_available() {
            eprintln!("skipping: no Metal device/pipeline available");
            return;
        }
        let m = 18usize; // 2^18 / 8 = 32 KiB = 2 pages per array
        let mut rng = Rng::new(0x6e6f_636f_7079_u64);
        let total = (1usize << m) / 8;
        assert_eq!(total % PAGE, 0);

        // Copy the packed arrays into guaranteed page-aligned backing.
        let mut backing: Vec<PageBuf> = Vec::new();
        for _ in 0..3 {
            let packed = pack_bits(&rng.bits(1 << m));
            let mut buf = PageBuf::zeroed(total);
            buf.as_mut_slice()[..total].copy_from_slice(&packed);
            backing.push(buf);
        }
        let a = &backing[0].as_slice()[..total];
        let b = &backing[1].as_slice()[..total];
        let c = &backing[2].as_slice()[..total];

        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let inv_table = make_inv_table();
        let padding = PaddingSpec::dense(m);

        let before = super::real::NOCOPY_BINDS.load(Ordering::Relaxed);
        assert_gpu_full_range_matches(m, &padding, None, a, b, c, &r, &inv_table);
        let after = super::real::NOCOPY_BINDS.load(Ordering::Relaxed);
        assert!(
            after - before >= 3,
            "aligned inputs must bind zero-copy (before={before}, after={after})"
        );
    }

    #[test]
    fn calibration_rule_balances_and_latches() {
        use super::real::urm::calibration_rule;
        // Equal per-x_hi rates → even split.
        assert_eq!(calibration_rule(1.0, 1.0), 64);
        // GPU 3× faster → 128 · 3/4 = 96.
        assert_eq!(calibration_rule(1.0, 3.0), 96);
        // GPU 1.5× slower: proportional 128 / 2.5 ≈ 51.
        assert_eq!(calibration_rule(1.5, 1.0), 51);
        // GPU 3× slower (M4 Pro-class GPU): 128 / 4 = 32 — still a ~25% phase
        // cut; the old 1.5× latch would have discarded it.
        assert_eq!(calibration_rule(3.0, 1.0), 32);
        // GPU 5× slower: clamps to the G_MIN floor share.
        assert_eq!(calibration_rule(5.0, 1.0), 21);
        // Catastrophically slow (> 6×) → pure CPU.
        assert_eq!(calibration_rule(6.1, 1.0), 0);
        // Extreme GPU advantage clamps at 112 (CPU keeps ≥ 16 x_hi).
        assert_eq!(calibration_rule(0.001, 1.0), 112);
    }
}

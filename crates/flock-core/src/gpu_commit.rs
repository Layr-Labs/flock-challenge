//! GPU (Metal) offload of the ranked L0 PCS commit.
//!
//! The ranked commit transforms a 1 GiB codeword (interleaved additive NTT,
//! 64 SoA lanes, `log_d = 20`) and hashes it into a BLAKE3 Merkle tree. Both
//! stages are memory-bandwidth-bound on the CPU and challenge-independent, so
//! they can run on the Apple-silicon GPU (unified memory, no PCIe copies)
//! while the P-cores run the compute-bound round-1 AB precompute.
//!
//! Design rules (each one a lesson from prior attempts):
//! - **One command buffer** for the whole commit graph — fused multi-layer
//!   NTT dispatches, then leaves, then parent levels. No per-level round
//!   trips through the CPU.
//! - **All Metal state is created once** (dlopen, shader compile, persistent
//!   buffers) and the first use happens during the worker's *untimed* warmup
//!   prove.
//! - **Latched fallback**: the warmup prove runs BOTH paths, byte-compares
//!   codeword and tree, wall-clocks both, and only latches the GPU on when it
//!   is bit-exact AND clearly faster. Any Metal failure at any point latches
//!   the CPU path — worst case is the status quo.
//! - **Bit-exactness is absolute**: GF(2^128) is carry-less (XOR/shift), and
//!   BLAKE3 is integer math, so a correct kernel is bit-identical to the CPU
//!   by construction; the warmup compare enforces it at runtime.
//!
//! No new crate dependencies: Metal and libobjc are loaded with `dlopen` and
//! driven through `objc_msgSend`, with the MSL kernel source embedded as a
//! string and compiled at init (~120 ms, absorbed by the untimed warmup).
//!
//! Kill switch: `FLOCK_NO_GPU_COMMIT=1` disables everything.

#![allow(clippy::missing_safety_doc)]

use crate::field::F128;
use crate::ntt::AdditiveNttF128;

/// Env var that disables the GPU commit path entirely.
pub const ENV_NO_GPU_COMMIT: &str = "FLOCK_NO_GPU_COMMIT";

/// Env var that latches the GPU on whenever it is bit-exact, even without a
/// wall-clock win (A/B and test tooling).
pub const ENV_GPU_COMMIT_FORCE: &str = "FLOCK_GPU_COMMIT_FORCE";

/// Env var that disables this round's NTT pass tuning (the g4 shared-table +
/// zero-region-skip from-z kernel and the half-footprint final-pass kernel),
/// restoring the incumbent kernel selection as the same-binary control.
pub const ENV_NO_NTT_PASS_TUNE: &str = "FLOCK_NO_NTT_PASS_TUNE";

/// Latched once: pass tuning enabled unless the kill switch is set.
pub(crate) fn pass_tune_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(ENV_NO_NTT_PASS_TUNE).is_none())
}

/// Wall-clock margin the GPU must beat during the warmup dual-run: latch on
/// only when `gpu_wall * 1.10 <= cpu_wall`.
const LATCH_MARGIN: f64 = 1.10;

/// The exact ranked L0 geometry the GPU graph is built for (mirrors the CPU
/// pipeline's `is_ranked_ntt_merkle_leaf_pipeline_shape`): `log_d = 20`,
/// 64 interleaved lanes, rate-1/2 entry at layer 1, 1 KiB BLAKE3 leaves.
fn is_ranked_gpu_shape(params: &crate::pcs::commit::PcsParams) -> bool {
    params.m == 32
        && params.log_inv_rate == 1
        && params.log_batch_size == 6
        && params.profile == crate::pcs::ligerito::LigeritoProfile::Fast
        && params.merkle_hash == crate::merkle::HashKind::Blake3
}

/// Build the L0 commitment tree, on the GPU when the shape matches and the
/// warmup latch decided for it; otherwise (and on any failure) via `cpu`.
///
/// State machine, decided once per process during the worker's untimed
/// warmup prove (the first ranked-shape commit):
/// - first ranked commit: run the GPU graph on a staging copy AND the CPU
///   path, byte-compare codeword + tree, wall-clock both, latch On only when
///   bit-exact and clearly faster (or `FLOCK_GPU_COMMIT_FORCE=1`).
/// - latched On: run the graph in place over the caller's codeword buffer
///   (persistent no-copy wrap) + the persistent tree buffer. On a GPU error
///   after the buffer may have been mutated, restore it via
///   `replicate_message_fill(codeword, z_packed)` and fall back to `cpu` —
///   both callers guarantee the input was exactly that replicated state.
/// - latched Off (or any init failure, non-ranked shape, kill switch): `cpu`.
pub(crate) fn commit_l0_or_fallback(
    z_packed: &[F128],
    codeword: Vec<F128>,
    params: &crate::pcs::commit::PcsParams,
    cpu: impl FnOnce(&mut [F128]) -> Vec<crate::merkle::Hash>,
) -> (crate::pcs::commit::CodewordBuf, crate::pcs::commit::MerkleTreeBuf) {
    imp::commit_l0_or_fallback(z_packed, codeword, params, cpu)
}

/// A read-only view of the transformed L0 codeword living in the GPU's
/// persistent shared staging buffer (unified memory: CPU reads during the
/// PCS open are ordinary cached reads). Dropping it releases the staging
/// back to the latched GPU state for the next prove.
pub struct GpuCodeword {
    ptr: *const F128,
    len: usize,
}

/// Read-only ranked L0 tree in the persistent shared Metal buffer.
pub struct GpuMerkleTree {
    ptr: *const crate::merkle::Hash,
    len: usize,
}
unsafe impl Send for GpuMerkleTree {}
unsafe impl Sync for GpuMerkleTree {}
impl GpuMerkleTree {
    /// SAFETY: `ptr` must point at `len` initialized Hash nodes that stay valid
    /// and un-mutated for this value's lifetime (the process-persistent tree
    /// buffer, guarded by the staging lease / latch).
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub(crate) unsafe fn new(ptr: *const crate::merkle::Hash, len: usize) -> Self {
        Self { ptr, len }
    }
}
impl core::ops::Deref for GpuMerkleTree {
    type Target = [crate::merkle::Hash];
    fn deref(&self) -> &[crate::merkle::Hash] {
        // SAFETY: contract of `new`.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

// SAFETY: the underlying memory is plain host-visible shared memory owned by
// a process-lifetime Metal buffer; the GPU only writes it between
// construction points serialized by the latch.
unsafe impl Send for GpuCodeword {}
unsafe impl Sync for GpuCodeword {}

impl GpuCodeword {
    /// SAFETY: `ptr` must point at `len` initialized F128s that stay valid
    /// and un-mutated for this value's lifetime (the process-persistent
    /// staging buffer, guarded by the in-use flag).
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub(crate) unsafe fn new(ptr: *const F128, len: usize) -> Self {
        Self { ptr, len }
    }
}

impl core::ops::Deref for GpuCodeword {
    type Target = [F128];
    fn deref(&self) -> &[F128] {
        // SAFETY: contract of `new`.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for GpuCodeword {
    fn drop(&mut self) {
        imp::staging_released();
    }
}

/// Return a ranked-size tree allocation to the GPU tree pool (no-op when the
/// GPU is unavailable/off). Keeps the 64 MiB copy-out target page-resident
/// across the worker's warmup and timed proves.
pub(crate) fn give_tree(tree: Vec<crate::merkle::Hash>) {
    imp::give_tree(tree);
}

/// Returns true when the GPU commit machinery is allowed to initialize.
pub(crate) fn gpu_commit_enabled() -> bool {
    // A/B-CONTROL: set to `false` to build an exact GPU-off control binary
    // (the benchmark harness env-clears workers, so the env kill switch
    // cannot reach them; it still serves in-process tests and tooling).
    const GPU_COMMIT_DEFAULT: bool = true;
    GPU_COMMIT_DEFAULT
        && cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && std::env::var_os(ENV_NO_GPU_COMMIT).is_none()
}

/// Build the flat breadth-first twiddle table for `log_d` layers: layer `l`
/// occupies `[2^l - 1, 2^(l+1) - 1)`. Uses the NTT's cached table when
/// present, otherwise rebuilds it (small test domains only).
pub(crate) fn flat_twiddle_table(ntt: &AdditiveNttF128, log_d: usize) -> Vec<F128> {
    let n = (1usize << log_d) - 1;
    if let Some(t) = ntt.precomputed_twiddle_table()
        && t.len() >= n
    {
        return t[..n].to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for layer in 0..log_d {
        for block in 0..1usize << layer {
            out.push(ntt.twiddle(layer, block));
        }
    }
    out
}

/// Group the layers `[start_layer, log_d)` into fused passes of at most 4
/// layers each. Each pass is one GPU dispatch; a pass of `f` layers does one
/// full read+write of the buffer for `f` butterfly layers.
pub(crate) fn plan_passes(log_d: usize, start_layer: usize) -> Vec<(usize, usize)> {
    let mut passes = Vec::new();
    let mut l = start_layer;
    while l < log_d {
        let f = (log_d - l).min(4);
        passes.push((l, f));
        l += f;
    }
    passes
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod imp {
    use super::*;
    use std::ffi::c_void;
    use std::sync::OnceLock;

    // -----------------------------------------------------------------------
    // Minimal Objective-C / Metal FFI (dlopen + objc_msgSend, no crate deps).
    // -----------------------------------------------------------------------

    pub(crate) type Id = *mut c_void;
    type Sel = *mut c_void;

    unsafe extern "C" {
        fn dlopen(path: *const i8, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
    }
    const RTLD_NOW: i32 = 2;

    pub(crate) const NIL: Id = std::ptr::null_mut();

    /// Function pointers resolved from libobjc / Metal at init.
    pub(crate) struct Api {
        msg_send: *const c_void,
        get_class: unsafe extern "C" fn(*const i8) -> Id,
        sel_register: unsafe extern "C" fn(*const i8) -> Sel,
        pool_push: unsafe extern "C" fn() -> *mut c_void,
        pool_pop: unsafe extern "C" fn(*mut c_void),
        create_system_default_device: unsafe extern "C" fn() -> Id,
        copy_all_devices: unsafe extern "C" fn() -> Id,
    }
    // SAFETY: all fields are process-global immutable function pointers.
    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}

    /// `objc_msgSend` cast to a concrete signature per call site.
    macro_rules! send {
        ($api:expr, $ty:ty, $obj:expr, $sel:expr $(, $a:expr)* $(,)?) => {{
            let f: $ty = core::mem::transmute($api.msg_send);
            f($obj, ($api.sel_register)($sel.as_ptr()) $(, $a)*)
        }};
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(crate) struct MtlSize {
        pub width: u64,
        pub height: u64,
        pub depth: u64,
    }

    impl Api {
        fn load() -> Result<Api, String> {
            unsafe {
                let objc = dlopen(c"/usr/lib/libobjc.A.dylib".as_ptr().cast(), RTLD_NOW);
                if objc.is_null() {
                    return Err("dlopen libobjc failed".into());
                }
                // Foundation first (registers NSString etc.), then Metal.
                let foundation = dlopen(
                    c"/System/Library/Frameworks/Foundation.framework/Foundation"
                        .as_ptr()
                        .cast(),
                    RTLD_NOW,
                );
                if foundation.is_null() {
                    return Err("dlopen Foundation failed".into());
                }
                let metal = dlopen(
                    c"/System/Library/Frameworks/Metal.framework/Metal"
                        .as_ptr()
                        .cast(),
                    RTLD_NOW,
                );
                if metal.is_null() {
                    return Err("dlopen Metal failed".into());
                }
                let sym = |h: *mut c_void, name: &core::ffi::CStr| -> Result<*mut c_void, String> {
                    let p = dlsym(h, name.as_ptr());
                    if p.is_null() {
                        Err(format!("dlsym {name:?} failed"))
                    } else {
                        Ok(p)
                    }
                };
                Ok(Api {
                    msg_send: sym(objc, c"objc_msgSend")?,
                    get_class: core::mem::transmute(sym(objc, c"objc_getClass")?),
                    sel_register: core::mem::transmute(sym(objc, c"sel_registerName")?),
                    pool_push: core::mem::transmute(sym(objc, c"objc_autoreleasePoolPush")?),
                    pool_pop: core::mem::transmute(sym(objc, c"objc_autoreleasePoolPop")?),
                    create_system_default_device: core::mem::transmute(sym(
                        metal,
                        c"MTLCreateSystemDefaultDevice",
                    )?),
                    copy_all_devices: core::mem::transmute(sym(
                        metal,
                        c"MTLCopyAllDevices",
                    )?),
                })
            }
        }

        pub(crate) unsafe fn nsstring(&self, s: &str) -> Result<Id, String> {
            // NSString stringWithUTF8String: (autoreleased).
            unsafe {
                let cls = (self.get_class)(c"NSString".as_ptr().cast());
                if cls.is_null() {
                    return Err("NSString class not found".into());
                }
                let bytes = s.as_bytes();
                let mut buf = Vec::with_capacity(bytes.len() + 1);
                buf.extend_from_slice(bytes);
                buf.push(0);
                let ns: Id = send!(
                    self,
                    unsafe extern "C" fn(Id, Sel, *const u8) -> Id,
                    cls,
                    c"stringWithUTF8String:",
                    buf.as_ptr()
                );
                if ns.is_null() {
                    Err("NSString creation failed".into())
                } else {
                    Ok(ns)
                }
            }
        }

        pub(crate) unsafe fn error_string(&self, err: Id) -> String {
            if err.is_null() {
                return "unknown error (nil NSError)".into();
            }
            unsafe {
                let desc: Id = send!(
                    self,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    err,
                    c"localizedDescription"
                );
                if desc.is_null() {
                    return "unknown error (nil description)".into();
                }
                let cstr: *const u8 = send!(
                    self,
                    unsafe extern "C" fn(Id, Sel) -> *const u8,
                    desc,
                    c"UTF8String"
                );
                if cstr.is_null() {
                    return "unknown error (nil UTF8String)".into();
                }
                std::ffi::CStr::from_ptr(cstr.cast())
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }

    // -----------------------------------------------------------------------
    // Metal Shading Language kernels.
    // -----------------------------------------------------------------------

    /// GF(2^128) fused-layer additive-NTT butterfly kernel + BLAKE3 tree
    /// kernels. See the extensive comments inside the source.
    const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// GF(2^128), GHASH polynomial P = x^128 + x^7 + x^2 + x + 1.
//
// F128 memory layout (little-endian struct { uint64 lo; uint64 hi; }):
// uint4 v = (lo31..0, lo63..32, hi31..0, hi63..32); bit i of the field
// element is bit (i mod 32) of word i/32.
// ===========================================================================

// v * x mod P.
static inline uint4 gf_mulx(uint4 v) {
    uint carry = v.w >> 31;
    uint4 r;
    r.w = (v.w << 1) | (v.z >> 31);
    r.z = (v.z << 1) | (v.y >> 31);
    r.y = (v.y << 1) | (v.x >> 31);
    r.x = (v.x << 1) ^ (carry * 0x87u);
    return r;
}

// a * x^8 mod P. The 8 bits shifted out (h) fold back as h * (x^7+x^2+x+1),
// which spans at most bit 14 and lands entirely in the low word.
static inline uint4 gf_shl8(uint4 a) {
    uint h = a.w >> 24;
    uint4 r;
    r.w = (a.w << 8) | (a.z >> 24);
    r.z = (a.z << 8) | (a.y >> 24);
    r.y = (a.y << 8) | (a.x >> 24);
    r.x = (a.x << 8) ^ ((h << 7) ^ (h << 2) ^ (h << 1) ^ h);
    return r;
}

// v * tw mod P via byte-wise Horner over v, using the twiddle's reduced
// nibble-multiple tables: tab[n] = n*tw, tab[16+n] = (n*x^4)*tw (n = 0..15).
// acc = ((...(b15*tw)*x^8 ^ b14*tw)*x^8 ...) accumulates v*tw exactly.
static inline uint4 gf_mul_tab(uint4 v, threadgroup const uint4* tab) {
    uint4 acc = uint4(0u);
    for (int i = 15; i >= 0; i--) {
        acc = gf_shl8(acc);
        uint b = (v[i >> 2] >> ((i & 3) * 8)) & 0xffu;
        acc ^= tab[b & 15u] ^ tab[16u + (b >> 4)];
    }
    return acc;
}

// a * x^16 mod P. The 16 bits shifted out fold back as h * 0x87 (<= bit 22).
static inline uint4 gf_shl16(uint4 a) {
    uint h = a.w >> 16;
    uint4 r;
    r.w = (a.w << 16) | (a.z >> 16);
    r.z = (a.z << 16) | (a.y >> 16);
    r.y = (a.y << 16) | (a.x >> 16);
    r.x = (a.x << 16) ^ ((h << 7) ^ (h << 2) ^ (h << 1) ^ h);
    return r;
}

// v * tw mod P, 16 bits of v per Horner step, using four reduced nibble
// tables: tab[16k + n] = (n * x^(4k)) * tw for k = 0..3, n = 0..15.
// (A dual even/odd-chain variant with shl32 steps measured ~45% slower —
// the extra live accumulator tips the kernel into register spills.)
static inline uint4 gf_mul_tab4(uint4 v, threadgroup const uint4* tab) {
    uint4 acc = uint4(0u);
    for (int i = 7; i >= 0; i--) {
        acc = gf_shl16(acc);
        uint h = (v[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
        acc ^= tab[h & 15u]
             ^ tab[16u + ((h >> 4) & 15u)]
             ^ tab[32u + ((h >> 8) & 15u)]
             ^ tab[48u + (h >> 12)];
    }
    return acc;
}

// ===========================================================================
// Fused multi-layer interleaved additive-NTT butterfly pass.
//
// Data layout matches AdditiveNttF128::forward_transform_interleaved: 64 SoA
// lanes, element (pos, lane) at flat index pos*64 + lane. At global layer L
// (log_d total layers), butterflies pair positions differing in position bit
// (log_d - L - 1); the twiddle for a pair is twiddles[(1<<L)-1 + (pos >>
// (log_d - L))] shared by all 64 lanes.
//
// One pass applies f consecutive layers l..l+f-1 to a tile of 2^f positions
// x 64 lanes staged in threadgroup memory. The tile's positions share every
// position bit except the f pair bits [log_d-l-f, log_d-l), which are
// contiguous, so tile positions are strided by S = 2^(log_d-l-f):
//     pos(e) = (B << (log_d-l)) + (e << s) + r,  tgid = B*2^s + r.
// The tile needs 2^f - 1 distinct twiddles (a small binary tree: sub-layer j
// uses 2^j of them, selected by the top j bits of e); each gets a 32-entry
// reduced nibble table built cooperatively before the butterflies.
// ===========================================================================

struct NttParams {
    uint log_d;   // log2 of positions
    uint l;       // first fused layer
    uint f;       // number of fused layers (1..=4)
    uint s;       // log_d - l - f
};

#define NTT_MAX_F 4u

kernel void ntt_fused(device uint4* data                [[buffer(0)]],
                      device const uint4* twiddles      [[buffer(1)]],
                      constant NttParams& P             [[buffer(2)]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint lid  [[thread_index_in_threadgroup]])
{
    threadgroup uint4 tile[(1u << NTT_MAX_F) * 64u];       // 16 KiB
    threadgroup uint4 tabs[((1u << NTT_MAX_F) - 1u) * 32u]; // 7.5 KiB

    const uint lane = lid & 63u;
    const uint tid  = lid >> 6;              // 0 .. 2^(f-1)-1
    const uint nf   = 1u << P.f;
    const uint nhalf = nf >> 1;
    const uint B    = tgid >> P.s;
    const uint r    = tgid & ((1u << P.s) - 1u);
    const uint pos_base = (B << (P.log_d - P.l)) + r;

    // Stage the tile (each thread loads 2 elements; lane-major = coalesced).
    for (uint e = tid; e < nf; e += nhalf) {
        tile[(e << 6) + lane] = data[((pos_base + (e << P.s)) << 6) + lane];
    }

    // Build the reduced nibble tables for the tile's 2^f - 1 twiddles.
    // Tile-local twiddle t (heap order) = sub-layer j = floor(log2(t+1)),
    // in-layer index c = t+1-2^j; its global twiddle is
    // twiddles[(1 << (l+j)) - 1 + (B << j) + c].
    const uint n_entries = (nf - 1u) * 32u;
    for (uint ei = lid; ei < n_entries; ei += nhalf << 6) {
        uint t   = ei >> 5;
        uint sub = ei & 31u;
        uint hi  = sub >> 4;
        uint n   = sub & 15u;
        uint j   = 31u - clz(t + 1u);
        uint c   = t + 1u - (1u << j);
        uint4 tw = twiddles[(1u << (P.l + j)) - 1u + (B << j) + c];
        uint4 p = tw;
        if (hi != 0u) {
            p = gf_mulx(gf_mulx(gf_mulx(gf_mulx(p))));
        }
        uint4 val = uint4(0u);
        for (uint k = 0; k < 4; k++) {
            if ((n >> k) & 1u) { val ^= p; }
            p = gf_mulx(p);
        }
        tabs[ei] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // f butterfly sub-layers over the staged tile.
    for (uint j = 0; j < P.f; j++) {
        uint bpos = P.f - 1u - j;                  // pair bit within e
        uint low  = tid & ((1u << bpos) - 1u);
        uint eu   = ((tid >> bpos) << (bpos + 1u)) | low;
        uint ev   = eu | (1u << bpos);
        uint tsel = ((1u << j) - 1u) + (eu >> (P.f - j));
        uint4 u = tile[(eu << 6) + lane];
        uint4 v = tile[(ev << 6) + lane];
        uint4 nu = u ^ gf_mul_tab(v, &tabs[tsel << 5]);
        tile[(eu << 6) + lane] = nu;
        tile[(ev << 6) + lane] = nu ^ v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write the tile back.
    for (uint e = tid; e < nf; e += nhalf) {
        data[((pos_base + (e << P.s)) << 6) + lane] = tile[(e << 6) + lane];
    }
}

// ===========================================================================
// Register-resident specializations for the production passes (f = 4, 3).
//
// One thread owns ALL 2^f tile positions of a single lane in registers, so
// the whole radix-2^f butterfly network happens in-thread: no threadgroup
// staging of data, no inter-layer barriers. A threadgroup is 64 threads =
// one or more same-B tiles (64 lanes each); their shared 2^f - 1 twiddles get
// four reduced nibble tables each (gf_mul_tab4), built cooperatively in two
// phases: first the 4 base values tw*x^(4k) per twiddle, then the 16 nibble
// multiples of each base. Same-B tiles execute sequentially, keeping the
// 64-thread occupancy and register footprint of the one-tile kernel.
// The f loops below have compile-time bounds, so the elems[] array stays in
// registers (dynamic indexing would spill it to stack memory).
// ===========================================================================

#define DEF_NTT_FUSED_REG(NAME, F_CONST, LOG_G)                                \
kernel void NAME(device uint4* data                [[buffer(0)]],              \
                 device const uint4* twiddles      [[buffer(1)]],              \
                 constant NttParams& P             [[buffer(2)]],              \
                 uint tgid [[threadgroup_position_in_grid]],                   \
                 uint lid  [[thread_index_in_threadgroup]])                    \
{                                                                              \
    constexpr uint F   = F_CONST;                                              \
    constexpr uint NF  = 1u << F;                                              \
    constexpr uint NTW = NF - 1u;                                              \
    threadgroup uint4 bases[NTW * 4u];                                         \
    threadgroup uint4 tabs[NTW * 64u];                                         \
                                                                               \
    /* LOG_G > 0: process 2^LOG_G consecutive-r tiles sequentially while    */\
    /* reusing one same-B twiddle table. Requires s >= LOG_G. */              \
    const uint lane = lid;                                                     \
    const uint B = tgid >> (P.s - LOG_G);                                      \
    const uint r_base =                                                        \
        (tgid & ((1u << (P.s - LOG_G)) - 1u)) << LOG_G;                        \
                                                                               \
    /* Phase 1: base values tw * x^(4k), one entry per thread (<= 60). */     \
    if (lid < NTW * 4u) {                                                      \
        uint t = lid >> 2;                                                     \
        uint k = lid & 3u;                                                     \
        uint j = 31u - clz(t + 1u);                                            \
        uint c = t + 1u - (1u << j);                                           \
        uint4 p = twiddles[(1u << (P.l + j)) - 1u + (B << j) + c];             \
        for (uint m = 0; m < k * 4u; m++) { p = gf_mulx(p); }                  \
        bases[lid] = p;                                                        \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
                                                                               \
    /* Phase 2: nibble multiples of each base. */                             \
    for (uint ei = lid; ei < NTW * 64u; ei += 64u) {                           \
        uint t   = ei >> 6;                                                    \
        uint sub = ei & 63u;                                                   \
        uint n   = sub & 15u;                                                  \
        uint4 p  = bases[(t << 2) | (sub >> 4)];                               \
        uint4 val = uint4(0u);                                                 \
        for (uint k = 0; k < 4u; k++) {                                        \
            if ((n >> k) & 1u) { val ^= p; }                                   \
            p = gf_mulx(p);                                                    \
        }                                                                      \
        tabs[ei] = val;                                                        \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
                                                                               \
    for (uint rr = 0; rr < (1u << LOG_G); rr++) {                              \
        const uint r = r_base + rr;                                            \
        const uint pos_base = (B << (P.log_d - P.l)) + r;                      \
        /* Load one lane's tile column into registers (coalesced per e). */    \
        uint4 elems[NF];                                                       \
        for (uint e = 0; e < NF; e++) {                                        \
            elems[e] = data[((pos_base + (e << P.s)) << 6) + lane];            \
        }                                                                      \
        /* f butterfly sub-layers, entirely in registers. */                  \
        for (uint j = 0; j < F; j++) {                                         \
            uint bpos = F - 1u - j;                                            \
            for (uint b = 0; b < (NF >> 1); b++) {                             \
                uint low = b & ((1u << bpos) - 1u);                            \
                uint eu  = ((b >> bpos) << (bpos + 1u)) | low;                 \
                uint ev  = eu | (1u << bpos);                                  \
                uint tsel = ((1u << j) - 1u) + (eu >> (F - j));                \
                uint4 nu = elems[eu]                                           \
                    ^ gf_mul_tab4(elems[ev], &tabs[tsel << 6]);                \
                elems[eu] = nu;                                                \
                elems[ev] ^= nu;                                               \
            }                                                                  \
        }                                                                      \
        for (uint e = 0; e < NF; e++) {                                        \
            data[((pos_base + (e << P.s)) << 6) + lane] = elems[e];            \
        }                                                                      \
    }                                                                          \
}

DEF_NTT_FUSED_REG(ntt_fused_reg4g4, 4u, 2u)   // 4 same-B tiles, sequential
DEF_NTT_FUSED_REG(ntt_fused_reg4,   4u, 0u)
DEF_NTT_FUSED_REG(ntt_fused_reg3,   3u, 0u)

// ===========================================================================
// Half-footprint variant for the FINAL pass (l = 16, s = 0), where every
// tile is its own block and g4 table reuse cannot apply: 32-entry byte-
// Horner tables (gf_mul_tab, the generic staged kernel's proven layout)
// instead of 64-entry 16-bit-Horner ones — ~7.7 KiB of threadgroup memory
// per 64-thread tile instead of ~16.9 KiB, so twice the tiles fit a core's
// threadgroup-memory budget (the same occupancy currency the g4 reuse
// spends). The multiply pays 16 gf_shl8 steps instead of 8 gf_shl16 for
// the same 32 table lookups. 64-thread groups, unchanged register
// footprint.
// ===========================================================================
kernel void ntt_fused_reg4h8(device uint4* data                [[buffer(0)]],
                             device const uint4* twiddles      [[buffer(1)]],
                             constant NttParams& P             [[buffer(2)]],
                             uint tgid [[threadgroup_position_in_grid]],
                             uint lid  [[thread_index_in_threadgroup]])
{
    constexpr uint F   = 4u;
    constexpr uint NF  = 1u << F;
    constexpr uint NTW = NF - 1u;
    threadgroup uint4 tabs[NTW * 32u];

    const uint lane = lid & 63u;
    const uint B = tgid >> P.s;
    const uint r = tgid & ((1u << P.s) - 1u);
    const uint pos_base = (B << (P.log_d - P.l)) + r;

    // Same table build as the generic staged kernel: tab[t*32 + n] = n*tw,
    // tab[t*32 + 16 + n] = (n*x^4)*tw.
    for (uint ei = lid; ei < NTW * 32u; ei += 64u) {
        uint t   = ei >> 5;
        uint sub = ei & 31u;
        uint hi  = sub >> 4;
        uint n   = sub & 15u;
        uint j   = 31u - clz(t + 1u);
        uint c   = t + 1u - (1u << j);
        uint4 p = twiddles[(1u << (P.l + j)) - 1u + (B << j) + c];
        if (hi != 0u) {
            p = gf_mulx(gf_mulx(gf_mulx(gf_mulx(p))));
        }
        uint4 val = uint4(0u);
        for (uint k = 0; k < 4u; k++) {
            if ((n >> k) & 1u) { val ^= p; }
            p = gf_mulx(p);
        }
        tabs[ei] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint4 elems[NF];
    for (uint e = 0; e < NF; e++) {
        elems[e] = data[((pos_base + (e << P.s)) << 6) + lane];
    }

    for (uint j = 0; j < F; j++) {
        uint bpos = F - 1u - j;
        for (uint b = 0; b < (NF >> 1); b++) {
            uint low = b & ((1u << bpos) - 1u);
            uint eu  = ((b >> bpos) << (bpos + 1u)) | low;
            uint ev  = eu | (1u << bpos);
            uint tsel = ((1u << j) - 1u) + (eu >> (F - j));
            uint4 nu = elems[eu] ^ gf_mul_tab(elems[ev], &tabs[tsel << 5]);
            elems[eu] = nu;
            elems[ev] ^= nu;
        }
    }

    for (uint e = 0; e < NF; e++) {
        data[((pos_base + (e << P.s)) << 6) + lane] = elems[e];
    }
}

// ===========================================================================
// From-z first pass: fuses the RS zero-padding into the first four layers.
//
// The commit encodes the coefficient vector [z, 0, ..., 0] (rate 1/2). With
// l = 0 and f = 4 the tile's top e-bit IS the codeword's top position bit,
// so the upper half of every tile is the zero region and the lower half is
// z itself (message positions in the same 64-lane SoA layout). This pass
// therefore reads z ONCE (512 MiB), synthesizes the zero half for free, and
// writes the full post-layer-3 codeword (1 GiB) to `data` — out of place,
// so the caller's z buffer is never mutated and any GPU failure can fall
// back to the CPU with the inputs intact. Requires P.l == 0, P.f == 4,
// log_inv_rate == 1.
// ===========================================================================
kernel void ntt_fused_reg4_from_z(device uint4* data                [[buffer(0)]],
                                  device const uint4* twiddles      [[buffer(1)]],
                                  constant NttParams& P             [[buffer(2)]],
                                  device const uint4* z             [[buffer(3)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint lid  [[thread_index_in_threadgroup]])
{
    constexpr uint F   = 4u;
    constexpr uint NF  = 1u << F;
    constexpr uint NTW = NF - 1u;
    threadgroup uint4 bases[NTW * 4u];
    threadgroup uint4 tabs[NTW * 64u];

    const uint lane = lid & 63u;
    // l = 0: a single block, B = 0; tgid enumerates r in [0, 2^s).
    const uint r = tgid;
    const uint pos_base = r;

    if (lid < NTW * 4u) {
        uint t = lid >> 2;
        uint k = lid & 3u;
        uint j = 31u - clz(t + 1u);
        uint c = t + 1u - (1u << j);
        uint4 p = twiddles[(1u << j) - 1u + c];
        for (uint m = 0; m < k * 4u; m++) { p = gf_mulx(p); }
        bases[lid] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ei = lid; ei < NTW * 64u; ei += 64u) {
        uint t   = ei >> 6;
        uint sub = ei & 63u;
        uint n   = sub & 15u;
        uint4 p  = bases[(t << 2) | (sub >> 4)];
        uint4 val = uint4(0u);
        for (uint k = 0; k < 4u; k++) {
            if ((n >> k) & 1u) { val ^= p; }
            p = gf_mulx(p);
        }
        tabs[ei] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint4 elems[NF];
    for (uint e = 0; e < NF / 2u; e++) {
        elems[e] = z[(((e << P.s) + r) << 6) + lane];
    }
    for (uint e = NF / 2u; e < NF; e++) {
        elems[e] = uint4(0u);   // the zero-padded coefficient region
    }

    for (uint j = 0; j < F; j++) {
        uint bpos = F - 1u - j;
        for (uint b = 0; b < (NF >> 1); b++) {
            uint low = b & ((1u << bpos) - 1u);
            uint eu  = ((b >> bpos) << (bpos + 1u)) | low;
            uint ev  = eu | (1u << bpos);
            uint tsel = ((1u << j) - 1u) + (eu >> (F - j));
            uint4 nu = elems[eu] ^ gf_mul_tab4(elems[ev], &tabs[tsel << 6]);
            elems[eu] = nu;
            elems[ev] ^= nu;
        }
    }

    for (uint e = 0; e < NF; e++) {
        data[((pos_base + (e << P.s)) << 6) + lane] = elems[e];
    }
}

// ===========================================================================
// From-z, tuned: the same pass with the two structural facts the plain
// kernel leaves on the table.
//
// 1. l = 0 means EVERY tile lives in block B = 0 and uses the identical
//    twiddle set, so the promoted g4 idiom applies unconditionally: one
//    64-thread group builds the tables once and completes 4 consecutive-r
//    tiles sequentially (same shape as ntt_fused_reg4g4 — 64-thread groups,
//    unchanged register footprint).
// 2. Sub-layer 0 pairs (e, e+8) across the zero-padded coefficient half:
//    v = 0 makes the butterfly nu = u, new_v = u — a pure copy. Skip its 8
//    multiplies per tile and start the butterfly network at sub-layer 1
//    (the tables for twiddle t = 0 are still built; the build loop's shape
//    is not worth specializing).
// ===========================================================================
kernel void ntt_fused_reg4_from_zg4(device uint4* data                [[buffer(0)]],
                                    device const uint4* twiddles      [[buffer(1)]],
                                    constant NttParams& P             [[buffer(2)]],
                                    device const uint4* z             [[buffer(3)]],
                                    uint tgid [[threadgroup_position_in_grid]],
                                    uint lid  [[thread_index_in_threadgroup]])
{
    constexpr uint F   = 4u;
    constexpr uint NF  = 1u << F;
    constexpr uint NTW = NF - 1u;
    constexpr uint LOG_G = 2u;
    threadgroup uint4 bases[NTW * 4u];
    threadgroup uint4 tabs[NTW * 64u];

    const uint lane = lid & 63u;
    const uint r_base = tgid << LOG_G;

    if (lid < NTW * 4u) {
        uint t = lid >> 2;
        uint k = lid & 3u;
        uint j = 31u - clz(t + 1u);
        uint c = t + 1u - (1u << j);
        uint4 p = twiddles[(1u << j) - 1u + c];
        for (uint m = 0; m < k * 4u; m++) { p = gf_mulx(p); }
        bases[lid] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ei = lid; ei < NTW * 64u; ei += 64u) {
        uint t   = ei >> 6;
        uint sub = ei & 63u;
        uint n   = sub & 15u;
        uint4 p  = bases[(t << 2) | (sub >> 4)];
        uint4 val = uint4(0u);
        for (uint k = 0; k < 4u; k++) {
            if ((n >> k) & 1u) { val ^= p; }
            p = gf_mulx(p);
        }
        tabs[ei] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint rr = 0; rr < (1u << LOG_G); rr++) {
        const uint r = r_base + rr;
        const uint pos_base = r;

        // Sub-layer 0 with v = 0 is a copy: load z once, duplicate.
        uint4 elems[NF];
        for (uint e = 0; e < NF / 2u; e++) {
            elems[e] = z[(((e << P.s) + r) << 6) + lane];
            elems[e + NF / 2u] = elems[e];
        }

        for (uint j = 1; j < F; j++) {
            uint bpos = F - 1u - j;
            for (uint b = 0; b < (NF >> 1); b++) {
                uint low = b & ((1u << bpos) - 1u);
                uint eu  = ((b >> bpos) << (bpos + 1u)) | low;
                uint ev  = eu | (1u << bpos);
                uint tsel = ((1u << j) - 1u) + (eu >> (F - j));
                uint4 nu = elems[eu] ^ gf_mul_tab4(elems[ev], &tabs[tsel << 6]);
                elems[eu] = nu;
                elems[ev] ^= nu;
            }
        }

        for (uint e = 0; e < NF; e++) {
            data[((pos_base + (e << P.s)) << 6) + lane] = elems[e];
        }
    }
}

// ===========================================================================
// BLAKE3 tree kernels (added in the Merkle milestone; kept in one library).
//
// Leaf   = BLAKE3 non-root chaining value of one 1024-byte leaf (exactly one
//          chunk: 16 blocks, counter 0, CHUNK_START on block 0, CHUNK_END on
//          block 15, never ROOT) — matches Hasher::update().finalize_non_root.
// Parent = one compression: cv = IV, block = left||right, counter 0,
//          block_len 64, flags PARENT — matches merge_subtrees_non_root.
// ===========================================================================

constant uint B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};
constant uchar B3_PERM[16] = {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};

#define B3_CHUNK_START 1u
#define B3_CHUNK_END   2u
#define B3_PARENT      4u

static void b3_compress(thread uint* cv, thread const uint* m_in,
                        uint block_len, uint flags) {
    uint v[16];
    uint m[16];
    for (int i = 0; i < 8; i++) v[i] = cv[i];
    for (int i = 0; i < 4; i++) v[8 + i] = B3_IV[i];
    v[12] = 0u;         // counter lo (always 0 for our leaves/parents)
    v[13] = 0u;         // counter hi
    v[14] = block_len;
    v[15] = flags;
    for (int i = 0; i < 16; i++) m[i] = m_in[i];
    for (int r = 0; r < 7; r++) {
        #define G(a,b,c,d,x,y) \
            v[a] = v[a] + v[b] + x; v[d] = ((v[d]^v[a])>>16)|((v[d]^v[a])<<16); \
            v[c] = v[c] + v[d];     v[b] = ((v[b]^v[c])>>12)|((v[b]^v[c])<<20); \
            v[a] = v[a] + v[b] + y; v[d] = ((v[d]^v[a])>>8) |((v[d]^v[a])<<24); \
            v[c] = v[c] + v[d];     v[b] = ((v[b]^v[c])>>7) |((v[b]^v[c])<<25);
        G(0,4,8,12,  m[0], m[1]);  G(1,5,9,13,  m[2], m[3]);
        G(2,6,10,14, m[4], m[5]);  G(3,7,11,15, m[6], m[7]);
        G(0,5,10,15, m[8], m[9]);  G(1,6,11,12, m[10],m[11]);
        G(2,7,8,13,  m[12],m[13]); G(3,4,9,14,  m[14],m[15]);
        #undef G
        if (r < 6) {
            uint t[16];
            for (int i = 0; i < 16; i++) t[i] = m[B3_PERM[i]];
            for (int i = 0; i < 16; i++) m[i] = t[i];
        }
    }
    for (int i = 0; i < 8; i++) cv[i] = v[i] ^ v[8 + i];
}

kernel void leaf_hash(device const uint* codeword [[buffer(0)]],
                      device uint* out            [[buffer(1)]],
                      uint id [[thread_position_in_grid]])
{
    device const uint* leaf = codeword + id * 256u;   // 1024 bytes
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    for (uint b = 0; b < 16u; b++) {
        uint block[16];
        for (uint i = 0; i < 16u; i++) block[i] = leaf[b * 16u + i];
        uint flags = (b == 0u ? B3_CHUNK_START : 0u) | (b == 15u ? B3_CHUNK_END : 0u);
        b3_compress(cv, block, 64u, flags);
    }
    for (int i = 0; i < 8; i++) out[id * 8u + i] = cv[i];
}

kernel void parent_hash(device const uint* children [[buffer(0)]],
                        device uint* parents        [[buffer(1)]],
                        uint id [[thread_position_in_grid]])
{
    uint block[16];
    for (uint i = 0; i < 16u; i++) block[i] = children[id * 16u + i];
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    b3_compress(cv, block, 64u, B3_PARENT);
    for (int i = 0; i < 8; i++) parents[id * 8u + i] = cv[i];
}

"#;

    // -----------------------------------------------------------------------
    // Context: device, queue, pipelines. Created once per process.
    // -----------------------------------------------------------------------

    pub(crate) struct Gpu {
        pub(crate) api: Api,
        pub(crate) device: Id,
        pub(crate) queue: Id,
        pub(crate) pso_ntt: Id,
        pub(crate) pso_ntt4g4: Id,
        pub(crate) pso_ntt4: Id,
        pub(crate) pso_ntt3: Id,
        pub(crate) pso_ntt4z: Id,
        /// Pass-tuned variants: g4 shared-table from-z with the zero-region
        /// sub-layer skipped, and the half-footprint final-pass kernel.
        pub(crate) pso_ntt4zg4: Id,
        pub(crate) pso_ntt4h8: Id,
        pub(crate) pso_leaf: Id,
        pub(crate) pso_parent: Id,
    }
    // SAFETY: MTLDevice/MTLCommandQueue/MTLComputePipelineState are
    // documented thread-safe; command buffers/encoders are created and used
    // within a single call.
    unsafe impl Send for Gpu {}
    unsafe impl Sync for Gpu {}

    static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new();

    pub(crate) fn gpu() -> Result<&'static Gpu, String> {
        if !super::gpu_commit_enabled() {
            return Err("gpu commit disabled".into());
        }
        GPU.get_or_init(init_gpu).as_ref().map_err(|e| e.clone())
    }

    fn init_gpu() -> Result<Gpu, String> {
        let api = Api::load()?;
        unsafe {
            let pool_push = api.pool_push;
            let pool_pop = api.pool_pop;
            let pool = pool_push();
            let result = (move || -> Result<Gpu, String> {
                let mut device = (api.create_system_default_device)();
                if device.is_null() {
                    // Sessions without a WindowServer bootstrap (ssh, CI)
                    // get no *default* device; MTLCopyAllDevices still
                    // enumerates the built-in GPU.
                    let all = (api.copy_all_devices)();
                    if !all.is_null() {
                        device = send!(
                            api,
                            unsafe extern "C" fn(Id, Sel) -> Id,
                            all,
                            c"firstObject"
                        );
                    }
                }
                if device.is_null() {
                    return Err("MTLCreateSystemDefaultDevice returned nil".into());
                }
                let queue: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    device,
                    c"newCommandQueue"
                );
                if queue.is_null() {
                    return Err("newCommandQueue failed".into());
                }
                let src = api.nsstring(MSL_SOURCE)?;
                let mut err: Id = NIL;
                let library: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id,
                    device,
                    c"newLibraryWithSource:options:error:",
                    src,
                    NIL,
                    &mut err
                );
                if library.is_null() {
                    return Err(format!("shader compile failed: {}", api.error_string(err)));
                }
                let pso = |name: &str| -> Result<Id, String> {
                    let ns = api.nsstring(name)?;
                    let f: Id = send!(
                        api,
                        unsafe extern "C" fn(Id, Sel, Id) -> Id,
                        library,
                        c"newFunctionWithName:",
                        ns
                    );
                    if f.is_null() {
                        return Err(format!("kernel {name} not found"));
                    }
                    let mut err: Id = NIL;
                    let p: Id = send!(
                        api,
                        unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id,
                        device,
                        c"newComputePipelineStateWithFunction:error:",
                        f,
                        &mut err
                    );
                    send!(api, unsafe extern "C" fn(Id, Sel) -> Id, f, c"release");
                    if p.is_null() {
                        return Err(format!("pipeline {name}: {}", api.error_string(err)));
                    }
                    Ok(p)
                };
                let pso_ntt = pso("ntt_fused")?;
                let pso_ntt4g4 = pso("ntt_fused_reg4g4")?;
                let pso_ntt4 = pso("ntt_fused_reg4")?;
                let pso_ntt3 = pso("ntt_fused_reg3")?;
                let pso_ntt4z = pso("ntt_fused_reg4_from_z")?;
                let pso_ntt4zg4 = pso("ntt_fused_reg4_from_zg4")?;
                let pso_ntt4h8 = pso("ntt_fused_reg4h8")?;
                let pso_leaf = pso("leaf_hash")?;
                let pso_parent = pso("parent_hash")?;
                send!(api, unsafe extern "C" fn(Id, Sel) -> Id, library, c"release");
                Ok(Gpu {
                    api,
                    device,
                    queue,
                    pso_ntt,
                    pso_ntt4g4,
                    pso_ntt4,
                    pso_ntt3,
                    pso_ntt4z,
                    pso_ntt4zg4,
                    pso_ntt4h8,
                    pso_leaf,
                    pso_parent,
                })
            })();
            pool_pop(pool);
            result
        }
    }

    // -----------------------------------------------------------------------
    // Thin typed wrappers used by both the test harness and the latched path.
    // -----------------------------------------------------------------------

    impl Gpu {
        pub(crate) unsafe fn pool_push(&self) -> *mut c_void {
            unsafe { (self.api.pool_push)() }
        }
        pub(crate) unsafe fn pool_pop(&self, p: *mut c_void) {
            unsafe { (self.api.pool_pop)(p) }
        }

        /// `newBufferWithLength:options:` — shared storage.
        pub(crate) unsafe fn new_buffer(&self, len: usize) -> Result<Id, String> {
            unsafe {
                let b: Id = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, u64, u64) -> Id,
                    self.device,
                    c"newBufferWithLength:options:",
                    len as u64,
                    0u64 // MTLResourceStorageModeShared
                );
                if b.is_null() {
                    Err(format!("newBufferWithLength {len} failed"))
                } else {
                    Ok(b)
                }
            }
        }

        /// `newBufferWithBytesNoCopy:` over caller-owned page-aligned memory.
        /// Returns Err when the pointer/length do not satisfy Metal's page
        /// requirements (caller falls back to a copy or to the CPU).
        pub(crate) unsafe fn wrap_buffer(&self, ptr: *mut u8, len: usize) -> Result<Id, String> {
            let page = 16384usize;
            if ptr as usize % page != 0 || len % page != 0 || len == 0 {
                return Err(format!(
                    "no-copy wrap needs page alignment (ptr={:p} len={len})",
                    ptr
                ));
            }
            unsafe {
                let b: Id = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, *mut c_void, u64, u64, Id) -> Id,
                    self.device,
                    c"newBufferWithBytesNoCopy:length:options:deallocator:",
                    ptr.cast(),
                    len as u64,
                    0u64,
                    NIL
                );
                if b.is_null() {
                    Err("newBufferWithBytesNoCopy failed".into())
                } else {
                    Ok(b)
                }
            }
        }

        pub(crate) unsafe fn buffer_contents(&self, buf: Id) -> *mut u8 {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel) -> *mut u8,
                    buf,
                    c"contents"
                )
            }
        }

        pub(crate) unsafe fn release(&self, obj: Id) {
            if !obj.is_null() {
                unsafe {
                    send!(self.api, unsafe extern "C" fn(Id, Sel) -> Id, obj, c"release");
                }
            }
        }

        pub(crate) unsafe fn command_buffer(&self) -> Result<Id, String> {
            unsafe {
                let cb: Id = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    self.queue,
                    c"commandBuffer"
                );
                if cb.is_null() {
                    Err("commandBuffer failed".into())
                } else {
                    Ok(cb)
                }
            }
        }

        pub(crate) unsafe fn compute_encoder(&self, cb: Id) -> Result<Id, String> {
            unsafe {
                let e: Id = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    cb,
                    c"computeCommandEncoder"
                );
                if e.is_null() {
                    Err("computeCommandEncoder failed".into())
                } else {
                    Ok(e)
                }
            }
        }

        pub(crate) unsafe fn set_pipeline(&self, enc: Id, pso: Id) {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, Id),
                    enc,
                    c"setComputePipelineState:",
                    pso
                );
            }
        }

        pub(crate) unsafe fn set_buffer(&self, enc: Id, buf: Id, offset: usize, index: usize) {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                    enc,
                    c"setBuffer:offset:atIndex:",
                    buf,
                    offset as u64,
                    index as u64
                );
            }
        }

        pub(crate) unsafe fn set_bytes(&self, enc: Id, data: &[u8], index: usize) {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64),
                    enc,
                    c"setBytes:length:atIndex:",
                    data.as_ptr().cast(),
                    data.len() as u64,
                    index as u64
                );
            }
        }

        pub(crate) unsafe fn dispatch(&self, enc: Id, groups: u64, threads_per_group: u64) {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, MtlSize, MtlSize),
                    enc,
                    c"dispatchThreadgroups:threadsPerThreadgroup:",
                    MtlSize { width: groups, height: 1, depth: 1 },
                    MtlSize { width: threads_per_group, height: 1, depth: 1 }
                );
            }
        }

        pub(crate) unsafe fn end_encoding(&self, enc: Id) {
            unsafe {
                send!(self.api, unsafe extern "C" fn(Id, Sel), enc, c"endEncoding");
            }
        }

        /// Commit and block until completion; verifies status == completed.
        /// Commit without waiting (hybrid: CPU works while the GPU runs).
        pub(crate) unsafe fn commit_async(&self, cb: Id) {
            unsafe {
                send!(self.api, unsafe extern "C" fn(Id, Sel), cb, c"commit");
            }
        }

        /// Wait for a previously `commit_async`ed buffer and check status.
        pub(crate) unsafe fn wait_cb(&self, cb: Id) -> Result<(), String> {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel),
                    cb,
                    c"waitUntilCompleted"
                );
                let status: u64 = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel) -> u64,
                    cb,
                    c"status"
                );
                if status == 4 {
                    Ok(())
                } else {
                    Err(format!("command buffer status {status} (hybrid arm)"))
                }
            }
        }

        pub(crate) unsafe fn commit_and_wait(&self, cb: Id) -> Result<(), String> {
            unsafe {
                send!(self.api, unsafe extern "C" fn(Id, Sel), cb, c"commit");
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel),
                    cb,
                    c"waitUntilCompleted"
                );
                let status: u64 = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel) -> u64,
                    cb,
                    c"status"
                );
                if status == 4 {
                    Ok(())
                } else {
                    let err: Id = send!(
                        self.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        cb,
                        c"error"
                    );
                    Err(format!(
                        "command buffer status {status}: {}",
                        self.api.error_string(err)
                    ))
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Encoding helpers.
    // -----------------------------------------------------------------------

    #[repr(C)]
    pub(crate) struct NttParams {
        pub(crate) log_d: u32,
        pub(crate) l: u32,
        pub(crate) f: u32,
        pub(crate) s: u32,
    }

    /// Encode the fused NTT passes for `layers [start_layer, log_d)` over a
    /// 64-lane interleaved buffer bound at `data_buf`.
    pub(crate) unsafe fn encode_ntt_passes(
        gpu: &Gpu,
        enc: Id,
        data_buf: Id,
        tw_buf: Id,
        log_d: usize,
        start_layer: usize,
    ) {
        unsafe {
            gpu.set_buffer(enc, data_buf, 0, 0);
            gpu.set_buffer(enc, tw_buf, 0, 1);
            let share_log = if std::env::var_os("FLOCK_NO_GPU_TABLE_REUSE").is_some() {
                0usize
            } else {
                2usize
            };
            for (l, f) in super::plan_passes(log_d, start_layer) {
                // Register-resident specializations for the production pass
                // widths; the generic staged kernel covers the rest. At
                // production passes with s >= 2, one 64-thread group builds
                // the shared twiddle table once and processes four adjacent
                // same-B tiles sequentially. This preserves the incumbent
                // register occupancy; parallel 128/256/512-thread grouping
                // loses badly because each lane keeps 16 F128s live.
                let s = log_d - l - f;
                let (pso, tpg, groups) = match f {
                    4 if share_log > 0 && s >= share_log => (
                        gpu.pso_ntt4g4,
                        64u64,
                        1u64 << (log_d - f - share_log),
                    ),
                    // s < 2 (the final pass): no same-B tiles exist to
                    // share, so spend the same occupancy currency the other
                    // way — halve the per-tile table footprint (byte-Horner
                    // 32-entry tables) so twice the tiles fit a core.
                    4 if super::pass_tune_enabled() => {
                        (gpu.pso_ntt4h8, 64u64, 1u64 << (log_d - f))
                    }
                    4 => (gpu.pso_ntt4, 64u64, 1u64 << (log_d - f)),
                    3 => (gpu.pso_ntt3, 64u64, 1u64 << (log_d - f)),
                    _ => (gpu.pso_ntt, 1u64 << (f + 5), 1u64 << (log_d - f)),
                };
                gpu.set_pipeline(enc, pso);
                let p = NttParams {
                    log_d: log_d as u32,
                    l: l as u32,
                    f: f as u32,
                    s: s as u32,
                };
                let bytes = core::slice::from_raw_parts(
                    (&p as *const NttParams).cast::<u8>(),
                    core::mem::size_of::<NttParams>(),
                );
                gpu.set_bytes(enc, bytes, 2);
                gpu.dispatch(enc, groups, tpg);
            }
        }
    }

    /// [`encode_ntt_passes`] restricted to the position prefix covering the
    /// first `prefix16` sixteenths of the codeword. Valid because the kernel
    /// derives its block index from the HIGH bits of `tgid`
    /// (`B = tgid >> (P.s - LOG_G)`), so dispatching `groups * prefix16/16`
    /// threadgroups enumerates exactly the prefix blocks of every pass with
    /// `l >= 4`.
    pub(crate) unsafe fn encode_ntt_passes_prefix(
        gpu: &Gpu,
        enc: Id,
        data_buf: Id,
        tw_buf: Id,
        log_d: usize,
        start_layer: usize,
        prefix16: u64,
    ) {
        unsafe {
            gpu.set_buffer(enc, data_buf, 0, 0);
            gpu.set_buffer(enc, tw_buf, 0, 1);
            let share_log = if std::env::var_os("FLOCK_NO_GPU_TABLE_REUSE").is_some() {
                0usize
            } else {
                2usize
            };
            for (l, f) in super::plan_passes(log_d, start_layer) {
                debug_assert!(l >= 4, "prefix passes require layer >= 4 blocks");
                let s = log_d - l - f;
                let (pso, tpg, groups) = match f {
                    4 if share_log > 0 && s >= share_log => (
                        gpu.pso_ntt4g4,
                        64u64,
                        1u64 << (log_d - f - share_log),
                    ),
                    // s < 2 (the final pass): no same-B tiles exist to
                    // share, so spend the same occupancy currency the other
                    // way — halve the per-tile table footprint (byte-Horner
                    // 32-entry tables) so twice the tiles fit a core.
                    4 if super::pass_tune_enabled() => {
                        (gpu.pso_ntt4h8, 64u64, 1u64 << (log_d - f))
                    }
                    4 => (gpu.pso_ntt4, 64u64, 1u64 << (log_d - f)),
                    3 => (gpu.pso_ntt3, 64u64, 1u64 << (log_d - f)),
                    _ => (gpu.pso_ntt, 1u64 << (f + 5), 1u64 << (log_d - f)),
                };
                gpu.set_pipeline(enc, pso);
                let p = NttParams {
                    log_d: log_d as u32,
                    l: l as u32,
                    f: f as u32,
                    s: s as u32,
                };
                let bytes = core::slice::from_raw_parts(
                    (&p as *const NttParams).cast::<u8>(),
                    core::mem::size_of::<NttParams>(),
                );
                gpu.set_bytes(enc, bytes, 2);
                debug_assert_eq!(groups % 16, 0);
                gpu.dispatch(enc, groups / 16 * prefix16, tpg);
            }
        }
    }

    /// Encode leaves + all parent levels of ONE aligned subtree
    /// (`subtree_leaves` a power of two, `leaf_start` aligned to it), writing
    /// into the subtree's slots of the GLOBAL flat tree layout.
    pub(crate) unsafe fn encode_merkle_subtree(
        gpu: &Gpu,
        enc: Id,
        codeword_buf: Id,
        tree_buf: Id,
        n_leaves_total: usize,
        leaf_start: usize,
        subtree_leaves: usize,
    ) {
        debug_assert!(subtree_leaves.is_power_of_two());
        debug_assert_eq!(leaf_start % subtree_leaves, 0);
        unsafe {
            gpu.set_pipeline(enc, gpu.pso_leaf);
            gpu.set_buffer(enc, codeword_buf, leaf_start * 1024, 0);
            gpu.set_buffer(enc, tree_buf, leaf_start * 32, 1);
            let tpg = 256u64.min(subtree_leaves as u64);
            gpu.dispatch(enc, subtree_leaves as u64 / tpg, tpg);

            gpu.set_pipeline(enc, gpu.pso_parent);
            let mut level_start = 0usize; // global node index of level base
            let mut level_len = n_leaves_total;
            let mut local_start = leaf_start;
            let mut local_len = subtree_leaves;
            while local_len > 1 {
                let write_level_start = level_start + level_len;
                let n_out = local_len / 2;
                gpu.set_buffer(enc, tree_buf, (level_start + local_start) * 32, 0);
                gpu.set_buffer(
                    enc,
                    tree_buf,
                    (write_level_start + local_start / 2) * 32,
                    1,
                );
                let tpg = 256u64.min(n_out as u64);
                gpu.dispatch(enc, n_out as u64 / tpg, tpg);
                level_start = write_level_start;
                level_len /= 2;
                local_start /= 2;
                local_len = n_out;
            }
        }
    }

    /// Encode leaf hashing (1 KiB leaves) + all parent levels into `tree_buf`
    /// (flat layout: leaves first, then parent levels, root last).
    pub(crate) unsafe fn encode_merkle(
        gpu: &Gpu,
        enc: Id,
        codeword_buf: Id,
        tree_buf: Id,
        n_leaves: usize,
    ) {
        unsafe {
            gpu.set_pipeline(enc, gpu.pso_leaf);
            gpu.set_buffer(enc, codeword_buf, 0, 0);
            gpu.set_buffer(enc, tree_buf, 0, 1);
            let tpg = 256u64.min(n_leaves as u64);
            gpu.dispatch(enc, n_leaves as u64 / tpg, tpg);

            gpu.set_pipeline(enc, gpu.pso_parent);
            let mut read_start = 0usize; // node index
            let mut read_len = n_leaves;
            while read_len > 1 {
                let write_start = read_start + read_len;
                let n_out = read_len / 2;
                gpu.set_buffer(enc, tree_buf, read_start * 32, 0);
                gpu.set_buffer(enc, tree_buf, write_start * 32, 1);
                let tpg = 256u64.min(n_out as u64);
                gpu.dispatch(enc, n_out as u64 / tpg, tpg);
                read_start = write_start;
                read_len = n_out;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Copy-in/copy-out harness (tests and the warmup dual-run).
    // -----------------------------------------------------------------------

    /// Run the fused NTT passes on a copy of `data`, writing the result back.
    /// Copy-in/copy-out; bit-gate test harness.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn gpu_ntt_interleaved_from_layer(
        ntt: &AdditiveNttF128,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) -> Result<(), String> {
        assert_eq!(num_ntts, 64, "GPU NTT kernel is specialized to 64 lanes");
        let n_total = data.len();
        assert!(n_total.is_power_of_two() && n_total >= 64);
        let log_d = (n_total / 64).trailing_zeros() as usize;
        assert_eq!(n_total, 64usize << log_d);
        assert!(start_layer <= log_d);
        if start_layer == log_d {
            return Ok(());
        }
        let gpu = gpu()?;
        let twiddles = super::flat_twiddle_table(ntt, log_d);
        unsafe {
            let pool = gpu.pool_push();
            let result = (|| -> Result<(), String> {
                let data_bytes = core::mem::size_of_val(data);
                let data_buf = gpu.new_buffer(data_bytes)?;
                let tw_bytes = core::mem::size_of_val(twiddles.as_slice()).max(16);
                let tw_buf = match gpu.new_buffer(tw_bytes) {
                    Ok(b) => b,
                    Err(e) => {
                        gpu.release(data_buf);
                        return Err(e);
                    }
                };
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().cast::<u8>(),
                    gpu.buffer_contents(data_buf),
                    data_bytes,
                );
                if !twiddles.is_empty() {
                    std::ptr::copy_nonoverlapping(
                        twiddles.as_ptr().cast::<u8>(),
                        gpu.buffer_contents(tw_buf),
                        core::mem::size_of_val(twiddles.as_slice()),
                    );
                }
                let run = (|| -> Result<(), String> {
                    let cb = gpu.command_buffer()?;
                    let enc = gpu.compute_encoder(cb)?;
                    encode_ntt_passes(gpu, enc, data_buf, tw_buf, log_d, start_layer);
                    gpu.end_encoding(enc);
                    gpu.commit_and_wait(cb)?;
                    std::ptr::copy_nonoverlapping(
                        gpu.buffer_contents(data_buf),
                        data.as_mut_ptr().cast::<u8>(),
                        data_bytes,
                    );
                    Ok(())
                })();
                gpu.release(data_buf);
                gpu.release(tw_buf);
                run
            })();
            gpu.pool_pop(pool);
            result
        }
    }

    // -----------------------------------------------------------------------
    // Latched production path.
    // -----------------------------------------------------------------------

    use crate::merkle::Hash;
    use std::sync::Mutex;

    /// Persistent Metal state owned by the latched-on path.
    struct Latched {
        /// Uploaded breadth-first twiddle table (16 MiB at the ranked shape).
        tw_buf: Id,
        /// GPU-owned flat tree buffer (leaves + parents, 64 MiB).
        tree_buf: Id,
        /// GPU-owned codeword home (1 GiB). The commit graph writes the
        /// transformed codeword here and `ProverData.codeword` derefs into
        /// it (Metal-allocated memory measured ~30% faster for the streaming
        /// graph than no-copy-wrapped malloc pages; CPU reads of shared
        /// Metal memory during the open are ordinary cached reads).
        staging: Id,
        /// No-copy read-only wraps of caller z buffers: `(ptr, len, buffer)`.
        /// The pooled z allocation is stable across the worker's warmup and
        /// timed proves, so this normally holds one entry created AND
        /// page-wired during the untimed warmup.
        wraps: Vec<(usize, usize, Id)>,
    }
    // SAFETY: Metal objects are thread-safe; access is serialized by LATCH.
    unsafe impl Send for Latched {}

    /// Whether a `GpuCodeword` handed out by `run_latched` is still alive.
    /// While true, the staging buffer's contents belong to that ProverData
    /// and a new GPU commit must fall back to the CPU (never happens in the
    /// one-prove-at-a-time worker).
    static STAGING_IN_USE: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    pub(crate) fn staging_released() {
        STAGING_IN_USE.store(false, std::sync::atomic::Ordering::Release);
    }

    enum LatchState {
        Undecided,
        On(Latched),
        Off,
    }

    static LATCH: Mutex<LatchState> = Mutex::new(LatchState::Undecided);

    /// Pool for ranked-size tree allocations (the 64 MiB copy-out target).
    static TREE_POOL: Mutex<Vec<Vec<Hash>>> = Mutex::new(Vec::new());
    /// Ranked tree node count; only allocations this large are pooled.
    const RANKED_TREE_NODES: usize = (1 << 21) - 1;

    pub(crate) fn give_tree(tree: Vec<Hash>) {
        if tree.capacity() < RANKED_TREE_NODES {
            return;
        }
        let mut pool = TREE_POOL.lock().unwrap();
        if pool.len() < 2 {
            pool.push(tree);
        }
    }

    #[allow(clippy::uninit_vec)]
    fn take_tree(n: usize) -> Vec<Hash> {
        let mut pool = TREE_POOL.lock().unwrap();
        for i in 0..pool.len() {
            if pool[i].capacity() >= n {
                let mut v = pool.swap_remove(i);
                drop(pool);
                v.clear();
                // SAFETY: capacity checked; Hash is Copy POD; caller writes
                // every slot before reading (same contract as
                // alloc_uninit_vec).
                unsafe { v.set_len(n) };
                return v;
            }
        }
        drop(pool);
        crate::alloc_uninit_vec(n)
    }

    fn debug_enabled() -> bool {
        std::env::var_os("FLOCK_COMMIT_TIMING").is_some()
            || std::env::var_os("FLOCK_GPU_COMMIT_DEBUG").is_some()
    }

    /// Parallel byte compare of a raw GPU buffer against a slice.
    fn bytes_equal_parallel(a: *const u8, b: &[u8]) -> bool {
        use rayon::prelude::*;
        let a_addr = a as usize;
        b.par_chunks(1 << 22).enumerate().all(|(i, chunk)| {
            // SAFETY: caller guarantees `a` points at least `b.len()` bytes.
            let a_chunk = unsafe {
                core::slice::from_raw_parts((a_addr as *const u8).add(i << 22), chunk.len())
            };
            a_chunk == chunk
        })
    }

    /// Parallel copy out of a raw GPU buffer.
    fn copy_bytes_parallel(src: *const u8, dst: &mut [u8]) {
        use rayon::prelude::*;
        let src_addr = src as usize;
        dst.par_chunks_mut(1 << 22).enumerate().for_each(|(i, chunk)| {
            // SAFETY: caller guarantees `src` points at least `dst.len()`
            // bytes; chunks are disjoint.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (src_addr as *const u8).add(i << 22),
                    chunk.as_mut_ptr(),
                    chunk.len(),
                );
            }
        });
    }

    /// Encode + run the full production commit graph from the message `z`:
    /// the from-z first pass (layers 0..3, reads z once, synthesizes the RS
    /// zero half) into `staging`, four more fused passes in place, then
    /// leaves + parent levels into `tree_buf`. One command buffer. Never
    /// writes `z_buf`. Requires the ranked geometry (log_d = 20, rate 1/2).
    unsafe fn run_commit_graph_from_z(
        gpu: &Gpu,
        z_buf: Id,
        staging: Id,
        tw_buf: Id,
        tree_buf: Id,
        log_d: usize,
        n_leaves: usize,
    ) -> Result<(), String> {
        unsafe {
            let pool = gpu.pool_push();
            let r = (|| {
                let cb = gpu.command_buffer()?;
                let enc = gpu.compute_encoder(cb)?;
                // Pass 1: layers 0..3 from z.
                // From-z tiles all live in block B = 0 (l = 0), so the g4
                // table-reuse idiom applies; the tuned kernel also skips the
                // zero-region sub-layer (a pure copy).
                let zg4 = super::pass_tune_enabled();
                gpu.set_pipeline(enc, if zg4 { gpu.pso_ntt4zg4 } else { gpu.pso_ntt4z });
                gpu.set_buffer(enc, staging, 0, 0);
                gpu.set_buffer(enc, tw_buf, 0, 1);
                let p = NttParams {
                    log_d: log_d as u32,
                    l: 0,
                    f: 4,
                    s: (log_d - 4) as u32,
                };
                let bytes = core::slice::from_raw_parts(
                    (&p as *const NttParams).cast::<u8>(),
                    core::mem::size_of::<NttParams>(),
                );
                gpu.set_bytes(enc, bytes, 2);
                gpu.set_buffer(enc, z_buf, 0, 3);
                if zg4 {
                    gpu.dispatch(enc, 1u64 << (log_d - 6), 64);
                } else {
                    gpu.dispatch(enc, 1u64 << (log_d - 4), 64);
                }
                // Passes 2..: layers 4..log_d in place over staging.
                encode_ntt_passes(gpu, enc, staging, tw_buf, log_d, 4);
                encode_merkle(gpu, enc, staging, tree_buf, n_leaves);
                gpu.end_encoding(enc);
                gpu.commit_and_wait(cb)
            })();
            gpu.pool_pop(pool);
            r
        }
    }

    /// Hybrid GPU/CPU commit graph: the GPU runs the shared from-z top pass
    /// (layers 0..3) over the full codeword, then owns the position prefix
    /// (first `16 - k` sixteenths: remaining NTT passes + its aligned Merkle
    /// subtrees) asynchronously while the CPU completes the suffix `k`
    /// sixteenths (layers 4.. via the bit-exact block-range driver, suffix
    /// leaves + subtree parents) directly in the shared staging and tree
    /// buffers. The top 7 tree nodes are (re)computed on the CPU after the
    /// join, covering every decomposition boundary.
    ///
    /// Bit-exact: same kernels/twiddles on both sides, every element and
    /// tree node written exactly once (top nodes twice, identically).
    unsafe fn run_commit_graph_from_z_hybrid(
        gpu: &Gpu,
        z_buf: Id,
        staging: Id,
        tw_buf: Id,
        tree_buf: Id,
        log_d: usize,
        n_leaves: usize,
        k_cpu16: usize,
    ) -> Result<(), String> {
        use rayon::prelude::*;
        debug_assert!((1..16).contains(&k_cpu16));
        unsafe {
            let pool = gpu.pool_push();
            let r = (|| {
                // cb1: shared top pass, full range.
                let cb1 = gpu.command_buffer()?;
                let enc = gpu.compute_encoder(cb1)?;
                // From-z tiles all live in block B = 0 (l = 0), so the g4
                // table-reuse idiom applies; the tuned kernel also skips the
                // zero-region sub-layer (a pure copy).
                let zg4 = super::pass_tune_enabled();
                gpu.set_pipeline(enc, if zg4 { gpu.pso_ntt4zg4 } else { gpu.pso_ntt4z });
                gpu.set_buffer(enc, staging, 0, 0);
                gpu.set_buffer(enc, tw_buf, 0, 1);
                let p = NttParams {
                    log_d: log_d as u32,
                    l: 0,
                    f: 4,
                    s: (log_d - 4) as u32,
                };
                let bytes = core::slice::from_raw_parts(
                    (&p as *const NttParams).cast::<u8>(),
                    core::mem::size_of::<NttParams>(),
                );
                gpu.set_bytes(enc, bytes, 2);
                gpu.set_buffer(enc, z_buf, 0, 3);
                if zg4 {
                    gpu.dispatch(enc, 1u64 << (log_d - 6), 64);
                } else {
                    gpu.dispatch(enc, 1u64 << (log_d - 4), 64);
                }
                gpu.end_encoding(enc);
                gpu.commit_and_wait(cb1)?;

                // cb2: GPU prefix — remaining passes + aligned subtrees.
                let prefix16 = (16 - k_cpu16) as u64;
                let cb2 = gpu.command_buffer()?;
                let enc = gpu.compute_encoder(cb2)?;
                encode_ntt_passes_prefix(gpu, enc, staging, tw_buf, log_d, 4, prefix16);
                // Greedy aligned power-of-two subtree decomposition of the
                // leaf prefix.
                let sixteenth = n_leaves / 16;
                let mut start = 0usize;
                let prefix_leaves = (16 - k_cpu16) * sixteenth;
                while start < prefix_leaves {
                    let mut size = 1usize << (prefix_leaves - start).ilog2();
                    while start % size != 0 {
                        size >>= 1;
                    }
                    encode_merkle_subtree(gpu, enc, staging, tree_buf, n_leaves, start, size);
                    start += size;
                }
                gpu.end_encoding(enc);
                gpu.commit_async(cb2);

                // CPU: suffix NTT completion + leaves + subtree parents.
                // The twiddle table is deterministic per log_d; build it once
                // per process (first call lands in the untimed warmup).
                static NTT: std::sync::OnceLock<AdditiveNttF128> = std::sync::OnceLock::new();
                let ntt = NTT.get_or_init(|| AdditiveNttF128::standard(log_d));
                debug_assert_eq!(ntt.log_domain_size(), log_d);
                let data: &mut [F128] = core::slice::from_raw_parts_mut(
                    gpu.buffer_contents(staging).cast::<F128>(),
                    n_leaves * 64,
                );
                let tree: &mut [Hash] = core::slice::from_raw_parts_mut(
                    gpu.buffer_contents(tree_buf).cast::<Hash>(),
                    2 * n_leaves - 1,
                );
                let tree_base = crate::epool::SyncPtr(tree.as_mut_ptr());
                let suffix_leaf_start = prefix_leaves;
                let suffix_leaves = n_leaves - prefix_leaves;
                if hybrid_cpu_suffix_deep_pipeline_enabled() {
                    // Publish and hash each finalized layer-10 chunk before it
                    // leaves cache.  `elem_offset` is absolute in the shared
                    // staging buffer, hence `leaf_start` lands directly in the
                    // CPU-owned suffix of the shared tree. Different callback
                    // invocations own disjoint 1,024-leaf ranges; the GPU owns
                    // only `0..prefix_leaves`.
                    let finish_chunk = |elem_offset: usize, chunk: &[F128]| {
                        debug_assert_eq!(elem_offset % 64, 0);
                        let leaf_start = elem_offset / 64;
                        let leaf_len = chunk.len() / 64;
                        debug_assert!(leaf_start >= suffix_leaf_start);
                        debug_assert!(leaf_start + leaf_len <= n_leaves);
                        // SAFETY: the NTT callback runs only after this chunk's
                        // last write. Callback ranges are pairwise disjoint and
                        // disjoint from the concurrently executing GPU prefix.
                        let bytes = core::slice::from_raw_parts(
                            chunk.as_ptr().cast::<u8>(),
                            core::mem::size_of_val(chunk),
                        );
                        let outs = core::slice::from_raw_parts_mut(
                            tree_base.ptr().add(leaf_start),
                            leaf_len,
                        );
                        crate::merkle::hash_ranked_blake3_leaf_chunk(bytes, outs);
                    };
                    ntt.forward_transform_interleaved_ranked_block_range_and_then(
                        data,
                        64,
                        4,
                        log_d,
                        16 - k_cpu16,
                        16,
                        finish_chunk,
                    );
                } else {
                    // Exact same-binary control: the original streaming suffix
                    // driver followed by a separate 4,096-leaf hash traversal.
                    ntt.forward_transform_interleaved_block_range(
                        data,
                        64,
                        4,
                        log_d,
                        16 - k_cpu16,
                        16,
                    );
                    let suffix_bytes: &[u8] = core::slice::from_raw_parts(
                        data.as_ptr().cast::<u8>().add(suffix_leaf_start * 1024),
                        suffix_leaves * 1024,
                    );
                    const LEAF_JOB: usize = 1 << 12;
                    suffix_bytes
                        .par_chunks(LEAF_JOB * 1024)
                        .enumerate()
                        .for_each(|(i, bytes)| {
                            // SAFETY: disjoint leaf output ranges per job.
                            let outs = core::slice::from_raw_parts_mut(
                                tree_base.ptr().add(suffix_leaf_start + i * LEAF_JOB),
                                bytes.len() / 1024,
                            );
                            crate::merkle::hash_ranked_blake3_leaf_chunk(bytes, outs);
                        });
                }
                // Suffix aligned subtrees' parents (greedy decomposition).
                let mut sstart = suffix_leaf_start;
                while sstart < n_leaves {
                    let mut size = 1usize << (n_leaves - sstart).ilog2();
                    while sstart % size != 0 {
                        size >>= 1;
                    }
                    let mut level_start = 0usize;
                    let mut level_len = n_leaves;
                    let mut local_start = sstart;
                    let mut local_len = size;
                    while local_len > 1 {
                        let write_level_start = level_start + level_len;
                        let (r0, w0) =
                            (level_start + local_start, write_level_start + local_start / 2);
                        let n_out = local_len / 2;
                        // ≤1024-output jobs (the parent kernel's contract),
                        // parallel across the level.
                        // SAFETY: read level fully written (leaves above /
                        // previous iteration); each job's write range is
                        // disjoint, and all are disjoint from concurrent GPU
                        // subtree ranges.
                        (0..n_out.div_ceil(1024)).into_par_iter().for_each(|j| {
                            let o = j * 1024;
                            let len = 1024.min(n_out - o);
                            let read = core::slice::from_raw_parts(
                                tree_base.ptr().add(r0 + 2 * o),
                                2 * len,
                            );
                            let write = core::slice::from_raw_parts_mut(
                                tree_base.ptr().add(w0 + o),
                                len,
                            );
                            crate::merkle::hash_ranked_blake3_parent_chunk(read, write);
                        });
                        level_start = write_level_start;
                        level_len /= 2;
                        local_start /= 2;
                        local_len /= 2;
                    }
                    sstart += size;
                }

                // Join the GPU prefix, then (re)compute every level above
                // the sixteenth-granularity roots. Every subtree on either
                // side spans ≥ one sixteenth (2^16 leaves), so the 16-node
                // level is always fully populated by subtree-internal
                // parents; the 15 nodes above it are recomputed here,
                // covering every decomposition boundary for any k.
                gpu.wait_cb(cb2)?;
                let mut level_start = 0usize;
                let mut level_len = n_leaves;
                while level_len > 16 {
                    level_start += level_len;
                    level_len /= 2;
                }
                while level_len > 1 {
                    let write_start = level_start + level_len;
                    let read =
                        core::slice::from_raw_parts(tree_base.ptr().add(level_start), level_len);
                    let write = core::slice::from_raw_parts_mut(
                        tree_base.ptr().add(write_start),
                        level_len / 2,
                    );
                    crate::merkle::hash_ranked_blake3_parent_chunk(read, write);
                    level_start = write_start;
                    level_len /= 2;
                }
                Ok(())
            })();
            gpu.pool_pop(pool);
            r
        }
    }

    /// CPU share of the hybrid commit in sixteenths of the position range.
    /// 0 disables (pure-GPU graph). Default 5 is the conservative midpoint of
    /// the cache-local suffix plateau: it retains most of the measured gain on
    /// a 10P/4E M4 Pro without assuming the benchmark's larger M3 Max GPU has
    /// the same CPU/GPU balance. `FLOCK_HYBRID_CPU_BLOCKS` remains the exact
    /// split-point override.
    fn hybrid_cpu_sixteenths() -> usize {
        use std::sync::OnceLock;
        static K: OnceLock<usize> = OnceLock::new();
        *K.get_or_init(|| {
            if std::env::var_os("FLOCK_NO_HYBRID_COMMIT").is_some() {
                return 0;
            }
            std::env::var("FLOCK_HYBRID_CPU_BLOCKS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|k| *k < 16)
                .unwrap_or(5)
        })
    }

    /// Use the ranked cache-local deep-pair CPU suffix and hash each finalized
    /// chunk before eviction. `FLOCK_NO_HYBRID_CPU_SUFFIX_DEEP=1` restores the
    /// original all-layer streaming suffix plus separate leaf-hash pass for an
    /// exact same-binary comparison.
    fn hybrid_cpu_suffix_deep_pipeline_enabled() -> bool {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("FLOCK_NO_HYBRID_CPU_SUFFIX_DEEP").is_none())
    }

    struct WarmupRun {
        latched: Latched,
        gpu_tree: Vec<Hash>,
        gpu_wall_ms: f64,
    }

    /// GPU half of the warmup dual-run: create the persistent state (twiddle
    /// upload, staging codeword home, tree buffer, read-only z wrap), run
    /// the full from-z graph once untimed (page-wires every buffer exactly
    /// as the timed prove will find them), then run it again timed with the
    /// tree copy-out included (the timed path pays that too). Never mutates
    /// z or the caller's codeword.
    fn warmup_gpu_run(
        z_packed: &[F128],
        log_d: usize,
        n_leaves: usize,
    ) -> Result<WarmupRun, String> {
        let gpu = gpu()?;
        let ntt = AdditiveNttF128::standard(log_d);
        let twiddles = super::flat_twiddle_table(&ntt, log_d);
        let total_nodes = 2 * n_leaves - 1;
        unsafe {
            let pool = gpu.pool_push();
            let mut created: Vec<Id> = Vec::new();
            let r = (|created: &mut Vec<Id>| -> Result<WarmupRun, String> {
                let tw_bytes = core::mem::size_of_val(twiddles.as_slice());
                let tw_buf = gpu.new_buffer(tw_bytes)?;
                created.push(tw_buf);
                std::ptr::copy_nonoverlapping(
                    twiddles.as_ptr().cast::<u8>(),
                    gpu.buffer_contents(tw_buf),
                    tw_bytes,
                );
                let tree_buf = gpu.new_buffer(total_nodes * 32)?;
                created.push(tree_buf);
                let staging = gpu.new_buffer(n_leaves * 1024)?;
                created.push(staging);
                // Read-only no-copy wrap of the caller's z buffer. The GPU
                // never writes it; the pooled allocation is page-aligned.
                let z_bytes = core::mem::size_of_val(z_packed);
                let z_buf =
                    gpu.wrap_buffer(z_packed.as_ptr().cast_mut().cast::<u8>(), z_bytes)?;
                created.push(z_buf);

                // Untimed wiring run, then the identical timed run.
                run_commit_graph_from_z(gpu, z_buf, staging, tw_buf, tree_buf, log_d, n_leaves)?;
                let mut gpu_tree = take_tree(total_nodes);
                copy_bytes_parallel(gpu.buffer_contents(tree_buf), {
                    core::slice::from_raw_parts_mut(
                        gpu_tree.as_mut_ptr().cast::<u8>(),
                        total_nodes * 32,
                    )
                });
                let t0 = std::time::Instant::now();
                run_commit_graph_from_z(gpu, z_buf, staging, tw_buf, tree_buf, log_d, n_leaves)?;
                copy_bytes_parallel(gpu.buffer_contents(tree_buf), {
                    core::slice::from_raw_parts_mut(
                        gpu_tree.as_mut_ptr().cast::<u8>(),
                        total_nodes * 32,
                    )
                });
                let gpu_wall_ms = t0.elapsed().as_secs_f64() * 1e3;
                created.clear(); // ownership transfers to Latched
                Ok(WarmupRun {
                    latched: Latched {
                        tw_buf,
                        tree_buf,
                        staging,
                        wraps: vec![(z_packed.as_ptr() as usize, z_bytes, z_buf)],
                    },
                    gpu_tree,
                    gpu_wall_ms,
                })
            })(&mut created);
            for id in created {
                gpu.release(id);
            }
            gpu.pool_pop(pool);
            r
        }
    }

    fn release_latched(gpu: &Gpu, latched: Latched) {
        unsafe {
            gpu.release(latched.tw_buf);
            gpu.release(latched.tree_buf);
            gpu.release(latched.staging);
            for (_, _, buf) in latched.wraps {
                gpu.release(buf);
            }
        }
    }

    /// First ranked-shape commit of the process (= the untimed warmup
    /// prove): run both paths, compare, wall-clock, and latch.
    fn warmup_and_decide(
        latch: &mut LatchState,
        z_packed: &[F128],
        mut codeword: Vec<F128>,
        params: &crate::pcs::commit::PcsParams,
        cpu: impl FnOnce(&mut [F128]) -> Vec<Hash>,
    ) -> (crate::pcs::commit::CodewordBuf, crate::pcs::commit::MerkleTreeBuf) {
        use crate::pcs::commit::{CodewordBuf, MerkleTreeBuf};
        let dbg = debug_enabled();

        // CPU first: the warmup prove's commit arm runs concurrently with the
        // round-1 AB precompute (rayon::join), exactly like the timed prove,
        // so this wall reflects the real contention the latched GPU would
        // remove. Running the GPU first was measured to bias the comparison:
        // by the time the CPU arm started, the precompute had drained and the
        // CPU commit measured ~35% faster than its production reality.
        let t0 = std::time::Instant::now();
        let cpu_tree = cpu(&mut codeword);
        let cpu_wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        let outcome = warmup_gpu_run(z_packed, params.k_code(), params.n_leaves());

        let run = match outcome {
            Ok(run) => run,
            Err(e) => {
                if dbg {
                    eprintln!("[gpu-commit] warmup: GPU unavailable ({e}); latching CPU path");
                }
                *latch = LatchState::Off;
                return (CodewordBuf::Cpu(codeword), MerkleTreeBuf::Cpu(cpu_tree));
            }
        };
        let gpu = gpu().expect("gpu() succeeded during warmup_gpu_run");

        // Bit-exactness: full codeword and full tree.
        let codeword_ok = unsafe {
            bytes_equal_parallel(
                gpu.buffer_contents(run.latched.staging),
                core::slice::from_raw_parts(
                    codeword.as_ptr().cast::<u8>(),
                    core::mem::size_of_val(codeword.as_slice()),
                ),
            )
        };
        let tree_ok = run.gpu_tree == cpu_tree;
        let exact = codeword_ok && tree_ok;
        if !exact {
            eprintln!(
                "[gpu-commit] WARMUP MISMATCH (codeword_ok={codeword_ok} tree_ok={tree_ok}); \
                 latching CPU path"
            );
        }

        let force = std::env::var_os(super::ENV_GPU_COMMIT_FORCE).is_some();
        let fast = run.gpu_wall_ms * super::LATCH_MARGIN <= cpu_wall_ms;
        let on = exact && (fast || force);
        if dbg {
            eprintln!(
                "[gpu-commit] warmup: gpu {:.2} ms vs cpu {:.2} ms, bit-exact={exact}, \
                 force={force} -> latched {}",
                run.gpu_wall_ms,
                cpu_wall_ms,
                if on { "ON" } else { "OFF" }
            );
        }
        give_tree(run.gpu_tree);
        if on {
            *latch = LatchState::On(run.latched);
        } else {
            release_latched(gpu, run.latched);
            *latch = LatchState::Off;
        }
        (CodewordBuf::Cpu(codeword), MerkleTreeBuf::Cpu(cpu_tree))
    }

    /// Timed-prove path once latched On: run the from-z graph into the
    /// persistent staging buffer (never touching the caller's z or codeword
    /// buffers), hand back a zero-copy tree view, return the pooled input
    /// codeword to the scratch pool, and hand back a `GpuCodeword` view of the
    /// staging.
    fn run_latched(
        latch: &mut LatchState,
        z_packed: &[F128],
        mut codeword: Vec<F128>,
        params: &crate::pcs::commit::PcsParams,
        cpu: impl FnOnce(&mut [F128]) -> Vec<Hash>,
    ) -> (crate::pcs::commit::CodewordBuf, crate::pcs::commit::MerkleTreeBuf) {
        use crate::pcs::commit::{CodewordBuf, MerkleTreeBuf};
        use std::sync::atomic::Ordering;
        let log_d = params.k_code();
        let n_leaves = params.n_leaves();
        let total_nodes = 2 * n_leaves - 1;
        let codeword_len = params.codeword_len_f128();
        let gpu = match gpu() {
            Ok(g) => g,
            Err(_) => {
                let tree = cpu(&mut codeword);
                return (CodewordBuf::Cpu(codeword), MerkleTreeBuf::Cpu(tree));
            }
        };

        // The staging buffer is the codeword home; if a previous prove's
        // ProverData still holds it, fall back (never happens in the
        // one-prove-at-a-time worker).
        if STAGING_IN_USE.swap(true, Ordering::Acquire) {
            if debug_enabled() {
                eprintln!("[gpu-commit] staging still in use; CPU fallback");
            }
            let tree = cpu(&mut codeword);
            return (CodewordBuf::Cpu(codeword), MerkleTreeBuf::Cpu(tree));
        }

        // Resolve the read-only z wrap (normally cached from the warmup).
        let z_ptr = z_packed.as_ptr() as usize;
        let z_bytes = core::mem::size_of_val(z_packed);
        let (tw_buf, tree_buf, staging, z_buf) = {
            let LatchState::On(state) = &mut *latch else {
                unreachable!("run_latched requires LatchState::On")
            };
            let cached = state
                .wraps
                .iter()
                .find(|(p, l, _)| *p == z_ptr && *l == z_bytes)
                .map(|&(_, _, buf)| buf);
            let z_buf = match cached {
                Some(buf) => buf,
                None => match unsafe {
                    gpu.wrap_buffer(z_packed.as_ptr().cast_mut().cast::<u8>(), z_bytes)
                } {
                    Ok(buf) => {
                        state.wraps.push((z_ptr, z_bytes, buf));
                        buf
                    }
                    Err(e) => {
                        // Inputs untouched — plain CPU fallback is safe.
                        if debug_enabled() {
                            eprintln!("[gpu-commit] z wrap failed at prove time ({e})");
                        }
                        STAGING_IN_USE.store(false, Ordering::Release);
                        let tree = cpu(&mut codeword);
                        return (CodewordBuf::Cpu(codeword), MerkleTreeBuf::Cpu(tree));
                    }
                },
            };
            (state.tw_buf, state.tree_buf, state.staging, z_buf)
        };

        let t0 = std::time::Instant::now();
        let k_cpu16 = hybrid_cpu_sixteenths();
        let run = unsafe {
            if k_cpu16 > 0 {
                run_commit_graph_from_z_hybrid(
                    gpu, z_buf, staging, tw_buf, tree_buf, log_d, n_leaves, k_cpu16,
                )
            } else {
                run_commit_graph_from_z(gpu, z_buf, staging, tw_buf, tree_buf, log_d, n_leaves)
            }
        };
        if let Err(e) = run {
            // Neither z nor the replicated codeword was written by the GPU,
            // so the plain CPU path is a bit-identical fallback.
            eprintln!("[gpu-commit] GPU failed mid-prove ({e}); falling back to CPU");
            STAGING_IN_USE.store(false, Ordering::Release);
            if let LatchState::On(state) = std::mem::replace(latch, LatchState::Off) {
                release_latched(gpu, state);
            }
            let tree = cpu(&mut codeword);
            return (CodewordBuf::Cpu(codeword), MerkleTreeBuf::Cpu(tree));
        }
        let graph_ms = t0.elapsed().as_secs_f64() * 1e3;
        // Zero-copy: opening only needs a query-dependent subset of the 64 MiB
        // tree; keep it in the persistent shared Metal buffer.
        let tree = unsafe {
            super::GpuMerkleTree::new(gpu.buffer_contents(tree_buf).cast::<Hash>(), total_nodes)
        };
        if std::env::var_os("FLOCK_COMMIT_TIMING").is_some() || debug_enabled() {
            eprintln!("[commit-timing] gpu-commit: graph {graph_ms:.2} ms + zero-copy tree");
        }
        // The replicated input codeword was never read by the from-z graph;
        // hand it straight back to the scratch pool for the next prove.
        crate::scratch::give_f128(codeword);
        let gpu_codeword = unsafe {
            super::GpuCodeword::new(gpu.buffer_contents(staging).cast::<F128>(), codeword_len)
        };
        (CodewordBuf::Gpu(gpu_codeword), MerkleTreeBuf::Gpu(tree))
    }

    pub(crate) fn commit_l0_or_fallback(
        z_packed: &[F128],
        mut codeword: Vec<F128>,
        params: &crate::pcs::commit::PcsParams,
        cpu: impl FnOnce(&mut [F128]) -> Vec<Hash>,
    ) -> (crate::pcs::commit::CodewordBuf, crate::pcs::commit::MerkleTreeBuf) {
        use crate::pcs::commit::{CodewordBuf, MerkleTreeBuf};
        if !super::gpu_commit_enabled()
            || !super::is_ranked_gpu_shape(params)
            || rayon::current_num_threads() <= 1
        {
            let tree = cpu(&mut codeword);
            return (CodewordBuf::Cpu(codeword), MerkleTreeBuf::Cpu(tree));
        }
        let mut latch = LATCH.lock().unwrap();
        match &*latch {
            LatchState::Off => {
                drop(latch);
                let tree = cpu(&mut codeword);
                (CodewordBuf::Cpu(codeword), MerkleTreeBuf::Cpu(tree))
            }
            LatchState::Undecided => {
                warmup_and_decide(&mut latch, z_packed, codeword, params, cpu)
            }
            LatchState::On(_) => run_latched(&mut latch, z_packed, codeword, params, cpu),
        }
    }

    /// Build the full BLAKE3 Merkle tree (1 KiB leaves) for `data` on the
    /// GPU. Copy-in/copy-out; bit-gate test harness.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn gpu_merkle_tree_blake3(
        data: &[u8],
        n_leaves: usize,
    ) -> Result<Vec<crate::merkle::Hash>, String> {
        assert!(n_leaves.is_power_of_two() && n_leaves > 0);
        assert_eq!(data.len(), n_leaves * 1024, "GPU leaves are 1 KiB");
        let gpu = gpu()?;
        let total_nodes = 2 * n_leaves - 1;
        unsafe {
            let pool = gpu.pool_push();
            let result = (|| -> Result<Vec<crate::merkle::Hash>, String> {
                let data_buf = gpu.new_buffer(data.len())?;
                let tree_buf = match gpu.new_buffer(total_nodes * 32) {
                    Ok(b) => b,
                    Err(e) => {
                        gpu.release(data_buf);
                        return Err(e);
                    }
                };
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    gpu.buffer_contents(data_buf),
                    data.len(),
                );
                let run = (|| -> Result<Vec<crate::merkle::Hash>, String> {
                    let cb = gpu.command_buffer()?;
                    let enc = gpu.compute_encoder(cb)?;
                    encode_merkle(gpu, enc, data_buf, tree_buf, n_leaves);
                    gpu.end_encoding(enc);
                    gpu.commit_and_wait(cb)?;
                    let mut tree: Vec<crate::merkle::Hash> =
                        crate::alloc_uninit_vec(total_nodes);
                    std::ptr::copy_nonoverlapping(
                        gpu.buffer_contents(tree_buf),
                        tree.as_mut_ptr().cast::<u8>(),
                        total_nodes * 32,
                    );
                    Ok(tree)
                })();
                gpu.release(data_buf);
                gpu.release(tree_buf);
                run
            })();
            gpu.pool_pop(pool);
            result
        }
    }
}

// Test-harness entry points (copy-in/copy-out); production goes through
// `commit_l0_or_fallback` above.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use imp::{gpu_merkle_tree_blake3, gpu_ntt_interleaved_from_layer};

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
mod imp {
    use super::*;

    pub(crate) fn gpu_ntt_interleaved_from_layer(
        _ntt: &AdditiveNttF128,
        _data: &mut [F128],
        _num_ntts: usize,
        _start_layer: usize,
    ) -> Result<(), String> {
        Err("GPU commit is only available on macOS/aarch64".into())
    }

    pub(crate) fn gpu_merkle_tree_blake3(
        _data: &[u8],
        _n_leaves: usize,
    ) -> Result<Vec<crate::merkle::Hash>, String> {
        Err("GPU commit is only available on macOS/aarch64".into())
    }

    pub(crate) fn commit_l0_or_fallback(
        _z_packed: &[F128],
        mut codeword: Vec<F128>,
        _params: &crate::pcs::commit::PcsParams,
        cpu: impl FnOnce(&mut [F128]) -> Vec<crate::merkle::Hash>,
    ) -> (crate::pcs::commit::CodewordBuf, crate::pcs::commit::MerkleTreeBuf) {
        let tree = cpu(&mut codeword);
        (
            crate::pcs::commit::CodewordBuf::Cpu(codeword),
            crate::pcs::commit::MerkleTreeBuf::Cpu(tree),
        )
    }

    pub(crate) fn give_tree(_tree: Vec<crate::merkle::Hash>) {}

    pub(crate) fn staging_released() {}
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use imp::{gpu_merkle_tree_blake3, gpu_ntt_interleaved_from_layer};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::F128;

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
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    /// Skip (with a note) when Metal is unavailable; fail on real GPU errors.
    fn gpu_or_skip<T>(r: Result<T, String>) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(e)
                if e.contains("disabled")
                    || e.contains("dlopen")
                    || e.contains("returned nil") =>
            {
                eprintln!("skipping GPU test: {e}");
                None
            }
            Err(e) => panic!("GPU error: {e}"),
        }
    }

    /// CPU oracle for exactly one interleaved butterfly layer.
    fn cpu_one_layer(ntt: &AdditiveNttF128, data: &mut [F128], num_ntts: usize, layer: usize) {
        let log_d = (data.len() / num_ntts).trailing_zeros() as usize;
        let num_blocks = 1usize << layer;
        let block_size = 1usize << (log_d - layer);
        let half = block_size >> 1;
        for block in 0..num_blocks {
            let tw = ntt.twiddle(layer, block);
            let base = block * block_size * num_ntts;
            for row in 0..half {
                for lane in 0..num_ntts {
                    let top = base + row * num_ntts + lane;
                    let bot = top + half * num_ntts;
                    let v = data[bot];
                    let nu = data[top] + v * tw;
                    data[top] = nu;
                    data[bot] = v + nu;
                }
            }
        }
    }

    /// Run only the pass (l, f) on the GPU by entering/leaving at the right
    /// layers: gpu passes are planned from `start`, so single-pass runs are
    /// exercised through `gpu_ntt_interleaved_from_layer` with log_d = l + f
    /// truncation being impossible — instead test single layers via a
    /// dedicated plan. Here we simply compare full transforms; the dedicated
    /// single-layer test below pins per-layer exactness.
    #[test]
    fn gpu_full_ntt_matches_cpu_small_shapes() {
        for (log_d, start_layer) in [(6usize, 1usize), (7, 1), (8, 2), (9, 0), (10, 1)] {
            let ntt = AdditiveNttF128::standard(log_d);
            let mut rng = Rng::new(0xD1CE + log_d as u64);
            let mut data = rng.vec(64 << log_d);
            let mut expect = data.clone();
            match gpu_or_skip(gpu_ntt_interleaved_from_layer(
                &ntt,
                &mut data,
                64,
                start_layer,
            )) {
                Some(()) => {}
                None => return,
            }
            ntt.forward_transform_interleaved_scalar_from_layer(&mut expect, 64, start_layer);
            assert_eq!(
                data, expect,
                "GPU NTT mismatch at log_d={log_d} start={start_layer}"
            );
        }
    }

    /// The hybrid commit sends only a high-block prefix through the GPU NTT
    /// encoder. Check that the grouped four-tile kernel preserves that exact
    /// range: the selected prefix matches the complete CPU transform while
    /// the CPU-owned suffix remains untouched.
    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn gpu_ntt_prefix_matches_cpu_small_shape() {
        use super::imp;

        let log_d = 10usize;
        let start_layer = 4usize;
        let prefix16 = 14u64;
        let ntt = AdditiveNttF128::standard(log_d);
        let mut rng = Rng::new(0xA11C_ED16);
        let input = rng.vec(64 << log_d);
        let mut expect = input.clone();
        ntt.forward_transform_interleaved_scalar_from_layer(&mut expect, 64, start_layer);

        let gpu = match gpu_or_skip(imp::gpu().map(|g| g as *const imp::Gpu)) {
            Some(g) => unsafe { &*g },
            None => return,
        };
        let twiddles = flat_twiddle_table(&ntt, log_d);
        unsafe {
            let pool = gpu.pool_push();
            let data_bytes = core::mem::size_of_val(input.as_slice());
            let data_buf = gpu.new_buffer(data_bytes).unwrap();
            let tw_buf = gpu
                .new_buffer(core::mem::size_of_val(twiddles.as_slice()))
                .unwrap();
            std::ptr::copy_nonoverlapping(
                input.as_ptr().cast::<u8>(),
                gpu.buffer_contents(data_buf),
                data_bytes,
            );
            std::ptr::copy_nonoverlapping(
                twiddles.as_ptr().cast::<u8>(),
                gpu.buffer_contents(tw_buf),
                core::mem::size_of_val(twiddles.as_slice()),
            );

            let cb = gpu.command_buffer().unwrap();
            let enc = gpu.compute_encoder(cb).unwrap();
            imp::encode_ntt_passes_prefix(
                gpu,
                enc,
                data_buf,
                tw_buf,
                log_d,
                start_layer,
                prefix16,
            );
            gpu.end_encoding(enc);
            gpu.commit_and_wait(cb).unwrap();

            let got = core::slice::from_raw_parts(
                gpu.buffer_contents(data_buf).cast::<F128>(),
                input.len(),
            );
            let prefix_len = input.len() / 16 * prefix16 as usize;
            assert_eq!(&got[..prefix_len], &expect[..prefix_len]);
            assert_eq!(&got[prefix_len..], &input[prefix_len..]);
            gpu.release(data_buf);
            gpu.release(tw_buf);
            gpu.pool_pop(pool);
        }
    }

    #[test]
    fn gpu_single_layers_match_cpu() {
        // Exercise every fused width f=1..4 and both shallow and deep layers
        // by running [layer, log_d) on GPU vs scalar for various layers: the
        // first GPU pass covers min(4, log_d - layer) layers.
        let log_d = 8usize;
        let ntt = AdditiveNttF128::standard(log_d);
        for layer in 0..log_d {
            let mut rng = Rng::new(0xBEEF + layer as u64);
            let mut data = rng.vec(64 << log_d);
            let mut expect = data.clone();
            match gpu_or_skip(gpu_ntt_interleaved_from_layer(&ntt, &mut data, 64, layer)) {
                Some(()) => {}
                None => return,
            }
            ntt.forward_transform_interleaved_scalar_from_layer(&mut expect, 64, layer);
            assert_eq!(data, expect, "GPU NTT mismatch from layer {layer}");
        }
    }

    #[test]
    fn cpu_one_layer_oracle_is_consistent() {
        // The per-layer oracle composed over all layers must equal the
        // library transform (validates the oracle itself).
        let log_d = 6usize;
        let ntt = AdditiveNttF128::standard(log_d);
        let mut rng = Rng::new(42);
        let mut a = rng.vec(64 << log_d);
        let mut b = a.clone();
        for layer in 1..log_d {
            cpu_one_layer(&ntt, &mut a, 64, layer);
        }
        ntt.forward_transform_interleaved_scalar_from_layer(&mut b, 64, 1);
        assert_eq!(a, b);
    }

    /// M1 gate: ONE NTT layer, GPU vs CPU, at the ranked shape
    /// (log_d=20, 64 lanes, 1 GiB). Run with `--ignored`.
    #[test]
    #[ignore = "1 GiB buffers; run explicitly with --ignored"]
    fn gpu_one_layer_matches_cpu_at_ranked_shape() {
        let log_d = 20usize;
        let ntt = AdditiveNttF128::standard(log_d);
        let mut rng = Rng::new(0x1A7C);
        let mut data = rng.vec(64 << log_d);
        let mut expect = data.clone();
        // Run only layer 19 on the GPU (single-layer pass, f=1).
        let layer = log_d - 1;
        match gpu_or_skip(gpu_ntt_interleaved_from_layer(&ntt, &mut data, 64, layer)) {
            Some(()) => {}
            None => return,
        }
        cpu_one_layer(&ntt, &mut expect, 64, layer);
        assert_eq!(data, expect, "GPU single-layer mismatch at ranked shape");
    }

    /// M2 gate: the full ranked transform (layers 1..20 at log_d=20, 64
    /// lanes, 1 GiB) bit-exact vs `forward_transform_interleaved_from_layer`.
    /// Run with `--ignored`.
    #[test]
    #[ignore = "1 GiB buffers; run explicitly with --ignored"]
    fn gpu_full_ntt_matches_cpu_at_ranked_shape() {
        let log_d = 20usize;
        let ntt = AdditiveNttF128::standard(log_d);
        let mut rng = Rng::new(0xF00D);
        let mut data = rng.vec(64 << log_d);
        let mut expect = data.clone();
        let t_gpu = std::time::Instant::now();
        match gpu_or_skip(gpu_ntt_interleaved_from_layer(&ntt, &mut data, 64, 1)) {
            Some(()) => {}
            None => return,
        }
        let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let t_cpu = std::time::Instant::now();
        ntt.forward_transform_interleaved_from_layer(&mut expect, 64, 1);
        let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "ranked-shape NTT: gpu {gpu_ms:.1} ms (incl. 2 GiB copies) vs cpu {cpu_ms:.1} ms"
        );
        assert_eq!(data, expect, "GPU full NTT mismatch at ranked shape");
    }

    #[test]
    fn gpu_merkle_matches_cpu_small() {
        for log_leaves in [0usize, 1, 4, 8, 10] {
            let n_leaves = 1usize << log_leaves;
            let mut rng = Rng::new(0x3EAF + log_leaves as u64);
            let data: Vec<u8> = (0..n_leaves * 1024)
                .map(|_| (rng.next_u64() & 0xff) as u8)
                .collect();
            let got = match gpu_or_skip(gpu_merkle_tree_blake3(&data, n_leaves)) {
                Some(t) => t,
                None => return,
            };
            let expect =
                crate::merkle::merkle_tree(&data, n_leaves, crate::merkle::HashKind::Blake3);
            assert_eq!(got, expect, "GPU Merkle mismatch at n_leaves={n_leaves}");
        }
    }

    /// M3 gate: full ranked-size tree (2^20 1 KiB leaves). Run with `--ignored`.
    #[test]
    #[ignore = "1 GiB buffers; run explicitly with --ignored"]
    fn gpu_merkle_matches_cpu_at_ranked_shape() {
        let n_leaves = 1usize << 20;
        let mut rng = Rng::new(0xACE);
        let mut data: Vec<u8> = crate::alloc_uninit_vec(n_leaves * 1024);
        for chunk in data.chunks_mut(8) {
            let v = rng.next_u64().to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
        let t_gpu = std::time::Instant::now();
        let got = match gpu_or_skip(gpu_merkle_tree_blake3(&data, n_leaves)) {
            Some(t) => t,
            None => return,
        };
        let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        let t_cpu = std::time::Instant::now();
        let expect = crate::merkle::merkle_tree(&data, n_leaves, crate::merkle::HashKind::Blake3);
        let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "ranked-shape Merkle: gpu {gpu_ms:.1} ms (incl. copies) vs cpu {cpu_ms:.1} ms"
        );
        assert_eq!(got, expect, "GPU Merkle mismatch at ranked shape");
    }

    /// Per-kernel probe at the ranked shape for the pass-tuned variants:
    /// times the final pass (l=16, s=0) as reg4 vs the half-footprint h8
    /// kernel, each in its own command buffer (min of 3). Local numbers are
    /// DIRECTIONAL ONLY — the ranked M3 Max prices threadgroup shapes
    /// differently (a 256-thread parallel variant that was 1.94x faster on
    /// an M2 lost 6.8% on the runner). Diagnostics only; bit-exactness of
    /// these kernels is pinned by the small-shape and ranked-shape oracle
    /// tests, which run the production selection. Run with `--ignored
    /// --nocapture`.
    #[test]
    #[ignore = "1 GiB buffers; run explicitly with --ignored"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn gpu_final_pass_probe_at_ranked_shape() {
        use super::imp;
        let log_d = 20usize;
        let ntt = AdditiveNttF128::standard(log_d);
        let mut rng = Rng::new(0x9A55);
        let input = rng.vec(64 << log_d);
        let gpu = match gpu_or_skip(imp::gpu().map(|g| g as *const imp::Gpu)) {
            Some(g) => unsafe { &*g },
            None => return,
        };
        let twiddles = flat_twiddle_table(&ntt, log_d);
        unsafe {
            let pool = gpu.pool_push();
            let data_bytes = core::mem::size_of_val(input.as_slice());
            let data_buf = gpu.new_buffer(data_bytes).unwrap();
            let tw_buf = gpu
                .new_buffer(core::mem::size_of_val(twiddles.as_slice()))
                .unwrap();
            std::ptr::copy_nonoverlapping(
                input.as_ptr().cast::<u8>(),
                gpu.buffer_contents(data_buf),
                data_bytes,
            );
            std::ptr::copy_nonoverlapping(
                twiddles.as_ptr().cast::<u8>(),
                gpu.buffer_contents(tw_buf),
                core::mem::size_of_val(twiddles.as_slice()),
            );
            let time_pass = |pso: imp::Id, l: usize, log_g: u64| -> f64 {
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let t = std::time::Instant::now();
                    let cb = gpu.command_buffer().unwrap();
                    let enc = gpu.compute_encoder(cb).unwrap();
                    gpu.set_buffer(enc, data_buf, 0, 0);
                    gpu.set_buffer(enc, tw_buf, 0, 1);
                    gpu.set_pipeline(enc, pso);
                    let p = imp::NttParams {
                        log_d: log_d as u32,
                        l: l as u32,
                        f: 4,
                        s: (log_d - l - 4) as u32,
                    };
                    let bytes = core::slice::from_raw_parts(
                        (&p as *const imp::NttParams).cast::<u8>(),
                        core::mem::size_of::<imp::NttParams>(),
                    );
                    gpu.set_bytes(enc, bytes, 2);
                    gpu.dispatch(enc, (1u64 << (log_d - 4)) >> log_g, 64);
                    gpu.end_encoding(enc);
                    gpu.commit_and_wait(cb).unwrap();
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                }
                best
            };
            let base = time_pass(gpu.pso_ntt4, 16, 0);
            let h8 = time_pass(gpu.pso_ntt4h8, 16, 0);
            let mid_g4 = time_pass(gpu.pso_ntt4g4, 8, 2);
            eprintln!(
                "final-pass probe l=16 s=0: reg4 {base:.2} ms, h8 {h8:.2} ms \
                 (mid-pass g4 reference l=8: {mid_g4:.2} ms)"
            );
            gpu.release(data_buf);
            gpu.release(tw_buf);
            gpu.pool_pop(pool);
        }
    }

    /// Timing probe for the full warm commit graph (5 fused NTT passes +
    /// leaves + 20 parent levels, ONE command buffer) on persistent
    /// already-touched buffers — the shape the latched production path runs.
    /// Prints per-iteration walls; also re-verifies bit-exactness of the
    /// whole graph. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "1 GiB buffers; run explicitly with --ignored"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn gpu_commit_graph_timing_at_ranked_shape() {
        use super::imp;
        let log_d = 20usize;
        let n_leaves = 1usize << log_d;
        let ntt = AdditiveNttF128::standard(log_d);
        let mut rng = Rng::new(0x717E);
        let input = rng.vec(64 << log_d);
        let gpu = match gpu_or_skip(imp::gpu().map(|g| g as *const imp::Gpu)) {
            Some(g) => unsafe { &*g },
            None => return,
        };
        let twiddles = flat_twiddle_table(&ntt, log_d);
        unsafe {
            let pool = gpu.pool_push();
            let data_bytes = core::mem::size_of_val(input.as_slice());
            let data_buf = gpu.new_buffer(data_bytes).unwrap();
            let tw_buf = gpu
                .new_buffer(core::mem::size_of_val(twiddles.as_slice()))
                .unwrap();
            let tree_buf = gpu.new_buffer((2 * n_leaves - 1) * 32).unwrap();
            std::ptr::copy_nonoverlapping(
                twiddles.as_ptr().cast::<u8>(),
                gpu.buffer_contents(tw_buf),
                core::mem::size_of_val(twiddles.as_slice()),
            );
            let mut walls = Vec::new();
            for iter in 0..4 {
                // Reset the input each iteration (untimed).
                std::ptr::copy_nonoverlapping(
                    input.as_ptr().cast::<u8>(),
                    gpu.buffer_contents(data_buf),
                    data_bytes,
                );
                // Stage split: NTT passes alone, then merkle alone (separate
                // command buffers, diagnostics only), then the fused graph
                // wall is ~their sum (verified by earlier full-graph runs).
                let t = std::time::Instant::now();
                let cb = gpu.command_buffer().unwrap();
                let enc = gpu.compute_encoder(cb).unwrap();
                imp::encode_ntt_passes(gpu, enc, data_buf, tw_buf, log_d, 1);
                gpu.end_encoding(enc);
                gpu.commit_and_wait(cb).unwrap();
                let ntt_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = std::time::Instant::now();
                let cb = gpu.command_buffer().unwrap();
                let enc = gpu.compute_encoder(cb).unwrap();
                imp::encode_merkle(gpu, enc, data_buf, tree_buf, n_leaves);
                gpu.end_encoding(enc);
                gpu.commit_and_wait(cb).unwrap();
                let merkle_ms = t.elapsed().as_secs_f64() * 1e3;
                walls.push(ntt_ms + merkle_ms);
                eprintln!(
                    "commit graph iter {iter}: ntt {ntt_ms:.2} ms + merkle {merkle_ms:.2} ms = {:.2} ms",
                    ntt_ms + merkle_ms
                );
            }
            // Bit-exactness of the final iteration against the CPU pipeline.
            let mut expect = input.clone();
            ntt.forward_transform_interleaved_from_layer(&mut expect, 64, 1);
            let got = core::slice::from_raw_parts(
                gpu.buffer_contents(data_buf).cast::<F128>(),
                expect.len(),
            );
            assert_eq!(got, expect.as_slice(), "codeword mismatch");
            let expect_bytes = core::slice::from_raw_parts(
                expect.as_ptr().cast::<u8>(),
                core::mem::size_of_val(expect.as_slice()),
            );
            let expect_tree = crate::merkle::merkle_tree(
                expect_bytes,
                n_leaves,
                crate::merkle::HashKind::Blake3,
            );
            let got_tree = core::slice::from_raw_parts(
                gpu.buffer_contents(tree_buf).cast::<crate::merkle::Hash>(),
                2 * n_leaves - 1,
            );
            assert_eq!(got_tree, expect_tree.as_slice(), "tree mismatch");
            gpu.release(data_buf);
            gpu.release(tw_buf);
            gpu.release(tree_buf);
            gpu.pool_pop(pool);
            let best = walls.iter().skip(1).cloned().fold(f64::MAX, f64::min);
            eprintln!("warm commit graph best: {best:.2} ms (NTT layers 1..20 + leaves + parents, 1 GiB)");
        }
    }

    /// M4 gate: the full latched path end-to-end at the ranked shape through
    /// the public `pcs::commit` API. First commit = warmup dual-run (GPU vs
    /// CPU compare, CPU-authoritative result); second commit = latched GPU
    /// in-place path. Roots, trees, and codewords must be identical.
    /// Run with `--ignored --test-threads 1` (uses ~4 GiB and process-global
    /// latch state).
    #[test]
    #[ignore = "multi-GiB buffers + process-global latch; run explicitly with --ignored"]
    fn gpu_latched_commit_end_to_end_at_ranked_shape() {
        // SAFETY: test runs single-threaded via --test-threads 1.
        unsafe {
            std::env::set_var(ENV_GPU_COMMIT_FORCE, "1");
            std::env::set_var("FLOCK_GPU_COMMIT_DEBUG", "1");
        }
        let params = crate::pcs::commit::PcsParams {
            m: 32,
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: crate::pcs::ligerito::LigeritoProfile::Fast,
            merkle_hash: crate::merkle::HashKind::Blake3,
        };
        let mut rng = Rng::new(0x60D0);
        let z: Vec<F128> = (0..1usize << params.log_msg_len())
            .map(|_| rng.f128())
            .collect();

        // Warmup commit: dual-run, CPU-authoritative, decides the latch.
        let (c1, pd1) = crate::pcs::commit::commit(&z, &params);
        let tree1 = pd1.merkle_tree.to_vec();
        let codeword1 = pd1.codeword.to_vec();
        drop(pd1); // returns codeword + tree to the pools, as the prover does

        // Timed-style commit: latched GPU path over the pooled buffer.
        let t0 = std::time::Instant::now();
        let (c2, pd2) = crate::pcs::commit::commit(&z, &params);
        let latched_ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("latched commit (replicate+gpu graph+zero-copy tree): {latched_ms:.2} ms");

        assert_eq!(c1.root, c2.root, "roots differ between warmup and latched");
        assert_eq!(tree1, pd2.merkle_tree, "trees differ");
        assert!(codeword1[..] == pd2.codeword[..], "codewords differ");

        // And both must equal a pure-CPU oracle from scratch.
        let mut oracle = vec![F128::ZERO; params.codeword_len_f128()];
        crate::pcs::commit::replicate_message_fill(&mut oracle, &z);
        let oracle_tree = crate::pcs::commit::cpu_transform_and_tree(&mut oracle, &params, None);
        assert!(
            oracle[..] == pd2.codeword[..],
            "codeword differs from CPU oracle"
        );
        assert_eq!(
            oracle_tree, pd2.merkle_tree,
            "tree differs from CPU oracle"
        );
    }

    #[test]
    fn plan_passes_covers_all_layers() {
        for log_d in 1..=20 {
            for start in 0..=log_d {
                let passes = plan_passes(log_d, start);
                let mut l = start;
                for &(pl, pf) in &passes {
                    assert_eq!(pl, l);
                    assert!(pf >= 1 && pf <= 4);
                    assert!(pl + pf <= log_d);
                    l += pf;
                }
                assert_eq!(l, log_d);
            }
        }
        assert_eq!(plan_passes(20, 1), vec![(1, 4), (5, 4), (9, 4), (13, 4), (17, 3)]);
    }
}

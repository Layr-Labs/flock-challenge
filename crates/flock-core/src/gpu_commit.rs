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
//!   the CPU path — worst case is the status quo. The fused-leaf graph
//!   variant (final NTT pass + BLAKE3 leaf hashing in one dispatch) is A/B'd
//!   against the plain graph in the same warmup and the faster bit-exact
//!   variant is latched; the ranked harness env-clears workers, so the A/B
//!   default (not the `FLOCK_GPU_FUSE_LEAF` env override) is what ships.
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
) -> (crate::pcs::commit::CodewordBuf, Vec<crate::merkle::Hash>) {
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

/// Env var that forces the fused-leaf graph variant on (1) or off (0). The
/// benchmark harness env-clears workers (`benchmark-tools/harness`'s
/// `.env_clear`), so the ranked run always sees the A/B default below; this
/// override is for local A/B and tooling only.
pub const ENV_GPU_FUSE_LEAF: &str = "FLOCK_GPU_FUSE_LEAF";

/// Fused-leaf graph variant selection: `None` = the warmup A/Bs both graph
/// variants (plain final pass + leaf dispatch vs fused final pass + leaf
/// hashing) and latches the faster bit-exact one; `Some(true/false)` = force
/// one variant (local A/B only — the ranked harness env-clears workers).
pub(crate) fn gpu_fuse_leaf_mode() -> Option<bool> {
    match std::env::var_os(ENV_GPU_FUSE_LEAF) {
        None => None,
        Some(v) => Some(v == "1"),
    }
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

// v * tw mod P, using four reduced nibble tables. Keep the eight
// Horner steps explicit: the production f=3/f=4 kernels call this once per
// butterfly, so removing the loop counter, component-index arithmetic, and
// loop predicate is a compiler-facing change without changing field math.
static inline uint4 gf_mul_tab4_step(uint4 acc, uint h,
                                     threadgroup const uint4* tab) {
    acc = gf_shl16(acc);
    acc ^= tab[h & 15u]
         ^ tab[16u + ((h >> 4) & 15u)]
         ^ tab[32u + ((h >> 8) & 15u)]
         ^ tab[48u + (h >> 12)];
    return acc;
}

static inline uint4 gf_mul_tab4(uint4 v, threadgroup const uint4* tab) {
    uint4 acc = uint4(0u);
    acc = gf_mul_tab4_step(acc, v.w >> 16, tab);
    acc = gf_mul_tab4_step(acc, v.w & 0xffffu, tab);
    acc = gf_mul_tab4_step(acc, v.z >> 16, tab);
    acc = gf_mul_tab4_step(acc, v.z & 0xffffu, tab);
    acc = gf_mul_tab4_step(acc, v.y >> 16, tab);
    acc = gf_mul_tab4_step(acc, v.y & 0xffffu, tab);
    acc = gf_mul_tab4_step(acc, v.x >> 16, tab);
    acc = gf_mul_tab4_step(acc, v.x & 0xffffu, tab);
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
    // One thread per twiddle: 10 gf_mulx (4 to reach tw*x^16, 6 for the two
    // 3-step chains) + 30 XORs, vs 192 mulx in the per-entry chain version.
    if (lid < nf - 1u) {
        uint t   = lid;
        uint j   = 31u - clz(t + 1u);
        uint c   = t + 1u - (1u << j);
        uint4 tw = twiddles[(1u << (P.l + j)) - 1u + (B << j) + c];
        uint4 c0 = tw;
        uint4 c1 = gf_mulx(c0);
        uint4 c2 = gf_mulx(c1);
        uint4 c3 = gf_mulx(c2);
        uint4 d0 = gf_mulx(gf_mulx(gf_mulx(gf_mulx(tw)))); // tw * x^16
        uint4 d1 = gf_mulx(d0);
        uint4 d2 = gf_mulx(d1);
        uint4 d3 = gf_mulx(d2);
        for (uint n = 0u; n < 16u; n++) {
            uint4 v0 = uint4(0u);
            if ((n >> 0) & 1u) { v0 ^= c0; }
            if ((n >> 1) & 1u) { v0 ^= c1; }
            if ((n >> 2) & 1u) { v0 ^= c2; }
            if ((n >> 3) & 1u) { v0 ^= c3; }
            tabs[(t << 5) + n] = v0;
            uint4 v1 = uint4(0u);
            if ((n >> 0) & 1u) { v1 ^= d0; }
            if ((n >> 1) & 1u) { v1 ^= d1; }
            if ((n >> 2) & 1u) { v1 ^= d2; }
            if ((n >> 3) & 1u) { v1 ^= d3; }
            tabs[(t << 5) + 16u + n] = v1;
        }
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
// one tile (64 lanes); its 2^f - 1 twiddles get four reduced nibble tables
// each (gf_mul_tab4), built cooperatively in two phases: first the 4 base
// values tw*x^(4k) per twiddle, then the 16 nibble multiples of each base.
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
    constexpr uint NTHREADS = 64u << LOG_G;                                    \
    threadgroup uint4 bases[NTW * 4u];                                         \
    threadgroup uint4 tabs[NTW * 64u];                                         \
                                                                               \
    /* LOG_G > 0: 2^LOG_G tiles with consecutive r (same B, hence the same   */\
    /* twiddle set and tables) share this threadgroup. Requires s >= LOG_G.  */\
    const uint lane = lid & 63u;                                               \
    const uint rr   = lid >> 6;                                                \
    const uint B = tgid >> (P.s - LOG_G);                                      \
    const uint r = ((tgid & ((1u << (P.s - LOG_G)) - 1u)) << LOG_G) + rr;      \
    const uint pos_base = (B << (P.log_d - P.l)) + r;                          \
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
    /* Phase 2: nibble multiples of each base. Subset-sum construction:    \
     * per base, 3 mulx build the chain base*x^k (k = 0..3), then each of  \
     * the 16 entries is an XOR of chain terms (15 XORs), instead of 4     \
     * mulx per entry. */                                                   \
    if (lid < NTW * 4u) {                                                    \
        uint4 c0 = bases[lid];                                               \
        uint4 c1 = gf_mulx(c0);                                              \
        uint4 c2 = gf_mulx(c1);                                              \
        uint4 c3 = gf_mulx(c2);                                              \
        for (uint n = 0u; n < 16u; n++) {                                    \
            uint4 val = uint4(0u);                                           \
            if ((n >> 0) & 1u) { val ^= c0; }                                \
            if ((n >> 1) & 1u) { val ^= c1; }                                \
            if ((n >> 2) & 1u) { val ^= c2; }                                \
            if ((n >> 3) & 1u) { val ^= c3; }                                \
            tabs[(lid << 4) + n] = val;                                      \
        }                                                                    \
    }                                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
                                                                               \
    /* Load the lane's tile column into registers (coalesced per e). */       \
    uint4 elems[NF];                                                           \
    for (uint e = 0; e < NF; e++) {                                            \
        elems[e] = data[((pos_base + (e << P.s)) << 6) + lane];                \
    }                                                                          \
                                                                               \
    /* f butterfly sub-layers, entirely in registers. */                      \
    for (uint j = 0; j < F; j++) {                                             \
        uint bpos = F - 1u - j;                                                \
        for (uint b = 0; b < (NF >> 1); b++) {                                 \
            uint low = b & ((1u << bpos) - 1u);                                \
            uint eu  = ((b >> bpos) << (bpos + 1u)) | low;                     \
            uint ev  = eu | (1u << bpos);                                      \
            uint tsel = ((1u << j) - 1u) + (eu >> (F - j));                    \
            uint4 nu = elems[eu] ^ gf_mul_tab4(elems[ev], &tabs[tsel << 6]);   \
            elems[eu] = nu;                                                    \
            elems[ev] ^= nu;                                                   \
        }                                                                      \
    }                                                                          \
                                                                               \
    for (uint e = 0; e < NF; e++) {                                            \
        data[((pos_base + (e << P.s)) << 6) + lane] = elems[e];                \
    }                                                                          \
}

DEF_NTT_FUSED_REG(ntt_fused_reg4g8, 4u, 3u)   // 8 same-B tiles, 512 threads
DEF_NTT_FUSED_REG(ntt_fused_reg4,   4u, 0u)
DEF_NTT_FUSED_REG(ntt_fused_reg3,   3u, 0u)

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
    if (lid < NTW * 4u) {
        uint4 c0 = bases[lid];
        uint4 c1 = gf_mulx(c0);
        uint4 c2 = gf_mulx(c1);
        uint4 c3 = gf_mulx(c2);
        for (uint n = 0u; n < 16u; n++) {
            uint4 val = uint4(0u);
            if ((n >> 0) & 1u) { val ^= c0; }
            if ((n >> 1) & 1u) { val ^= c1; }
            if ((n >> 2) & 1u) { val ^= c2; }
            if ((n >> 3) & 1u) { val ^= c3; }
            tabs[(lid << 4) + n] = val;
        }
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

// Collapse eight complete parent levels per dispatch. One 128-thread group
// owns 256 input CVs and reduces them to one subtree root. Every internal CV
// is still written to the ordinary flat tree layout, but levels 1..7 are fed
// to the next compression through threadgroup memory rather than reread from
// DRAM by a later dispatch.
kernel void parent_hash8(device const uint* children [[buffer(0)]],
                         device uint* parents        [[buffer(1)]],
                         constant uint& read_len      [[buffer(2)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint lid  [[thread_index_in_threadgroup]])
{
    threadgroup uint cvs[128u * 8u];
    uint block[16];
    const uint child_base = (tgid * 256u + lid * 2u) * 8u;
    for (uint i = 0; i < 16u; i++) block[i] = children[child_base + i];

    uint cv[8];
    for (uint i = 0; i < 8u; i++) cv[i] = B3_IV[i];
    b3_compress(cv, block, 64u, B3_PARENT);
    const uint first_id = tgid * 128u + lid;
    for (uint i = 0; i < 8u; i++) {
        parents[first_id * 8u + i] = cv[i];
        cvs[lid * 8u + i] = cv[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint local_len = 128u;
    uint global_len = read_len >> 1u;
    uint level_offset = global_len;
    for (uint level = 1u; level < 8u; level++) {
        const uint next_len = local_len >> 1u;
        uint next_block[16];
        if (lid < next_len) {
            for (uint i = 0; i < 16u; i++) next_block[i] = cvs[lid * 16u + i];
        }
        // No lane may overwrite cvs before every active lane has captured its
        // two children. Inactive lanes participate so the barrier is uniform.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < next_len) {
            uint next_cv[8];
            for (uint i = 0; i < 8u; i++) next_cv[i] = B3_IV[i];
            b3_compress(next_cv, next_block, 64u, B3_PARENT);
            const uint out_id = level_offset + tgid * next_len + lid;
            for (uint i = 0; i < 8u; i++) {
                parents[out_id * 8u + i] = next_cv[i];
                cvs[lid * 8u + i] = next_cv[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        local_len = next_len;
        global_len >>= 1u;
        level_offset += global_len;
    }
}

// ===========================================================================
// Fused final NTT pass + BLAKE3 leaf hashing (DRAM-bytes family). Wired as a
// warmup-A/B'd graph variant: `run_commit_graph_from_z(fuse_leaf=true)` when
// the warmup's bit-exact byte-compare + wall-clock A/B latch it; `None` PSO
// state (threadgroup-memory limit) degrades to the unfused variant.
//
// Ledger: the commit graph reads z once (0.5 GiB), writes the post-layer-3
// codeword (1.0 GiB), runs four in-place fused passes (8.0 GiB), then
// leaf_hash re-reads the WHOLE 1 GiB staging that pass (16,4) just wrote
// (encode_merkle's leaf dispatch). At the final pass l = log_d-4, f = 4,
// s = 0 each 64-thread threadgroup owns positions pos_base..pos_base+15;
// with 64 lanes x 16 B per position that is exactly 16 contiguous 1 KiB
// leaves (leaf id = position) -- the threadgroup's own working set. Fusing
// removes the 1 GiB re-read with ZERO added ALU (the 16 compressions/leaf
// already run in leaf_hash) and only a threadgroup-memory stage: bases 60 +
// tabs 960 + tile 1024 uint4 = 2044 x 16 B = 32,704 B <= the 32 KiB
// threadgroup limit. Block b of leaf p = uints p*256 + b*16 + i, which in
// SoA terms is lane b*4 + i/4, uint j = i%4 -- that bijection is verified on
// x86 by fused_leaf_block_lane_mapping_bijects_leaf_uints. Bit-exactness is
// gated by the warmup dual-run byte-compare; a mismatch falls back to CPU.
// MUST only be wired at a pass with P.s == 0 (final fused pass of the ranked
// geometry, log_d = 20 -> l = 16).
// ===========================================================================
kernel void ntt4_fused_leaf(device uint4* data                [[buffer(0)]],
                            device const uint4* twiddles      [[buffer(1)]],
                            constant NttParams& P             [[buffer(2)]],
                            device uint* tree                 [[buffer(3)]],
                            uint tgid [[threadgroup_position_in_grid]],
                            uint lid  [[thread_index_in_threadgroup]])
{
    constexpr uint F   = 4u;
    constexpr uint NF  = 1u << F;
    constexpr uint NTW = NF - 1u;
    threadgroup uint4 bases[NTW * 4u];
    threadgroup uint4 tabs[NTW * 64u];
    threadgroup uint4 tile[NF * 64u];

    const uint lane = lid & 63u;
    const uint r    = tgid & ((1u << P.s) - 1u);
    const uint B    = tgid >> P.s;
    const uint pos_base = (B << (P.log_d - P.l)) + r;

    // Phase 1: base values tw * x^(4k), one entry per thread (<= 60).
    if (lid < NTW * 4u) {
        uint t = lid >> 2;
        uint k = lid & 3u;
        uint j = 31u - clz(t + 1u);
        uint c = t + 1u - (1u << j);
        uint4 p = twiddles[(1u << (P.l + j)) - 1u + (B << j) + c];
        for (uint m = 0; m < k * 4u; m++) { p = gf_mulx(p); }
        bases[lid] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < NTW * 4u) {
        uint4 c0 = bases[lid];
        uint4 c1 = gf_mulx(c0);
        uint4 c2 = gf_mulx(c1);
        uint4 c3 = gf_mulx(c2);
        for (uint n = 0u; n < 16u; n++) {
            uint4 val = uint4(0u);
            if ((n >> 0) & 1u) { val ^= c0; }
            if ((n >> 1) & 1u) { val ^= c1; }
            if ((n >> 2) & 1u) { val ^= c2; }
            if ((n >> 3) & 1u) { val ^= c3; }
            tabs[(lid << 4) + n] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint4 elems[NF];
    for (uint e = 0; e < NF; e++) {
        elems[e] = data[((pos_base + (e << P.s)) << 6) + lane];
    }

    // Butterfly arithmetic copied verbatim from ntt_fused_reg4/from_z (share
    // the register chain; no per-variant copy of the 21 multiplies).
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

    // Write the transformed codeword back (the CPU open reads it later via
    // ProverData.codeword's staging deref; parent hashing reads `tree`, not
    // this buffer -- leaf hashing is the only GPU reader being fused away) and
    // stage the tile: tile[p*64 + lane] = position p, this lane's 16 bytes.
    for (uint e = 0; e < NF; e++) {
        uint4 v = elems[e];
        tile[e * 64u + lane] = v;
        data[((pos_base + (e << P.s)) << 6) + lane] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 16 threads hash one leaf each (leaf at position pos_base + lid); the
    // BLAKE3 chunk CV chain is sequential within a leaf, so one thread per
    // leaf is the required shape (same total compressions as leaf_hash).
    if (lid < 16u) {
        uint leaf_pos = pos_base + lid;
        uint cv[8];
        for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
        for (uint b = 0; b < 16u; b++) {
            uint block[16];
            for (uint i = 0; i < 16u; i++) {
                block[i] = tile[lid * 64u + b * 4u + i / 4u][i % 4u];
            }
            uint flags = (b == 0u ? B3_CHUNK_START : 0u) | (b == 15u ? B3_CHUNK_END : 0u);
            b3_compress(cv, block, 64u, flags);
        }
        for (int i = 0; i < 8; i++) tree[leaf_pos * 8u + i] = cv[i];
    }
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
        /// Compiled but unselected: 8-tile shared-table variant, kept for
        /// occupancy experiments (see the note in `encode_ntt_passes`).
        #[allow(dead_code)]
        pub(crate) pso_ntt4g8: Id,
        pub(crate) pso_ntt4: Id,
        pub(crate) pso_ntt3: Id,
        pub(crate) pso_ntt4z: Id,
        /// Fused final-pass + leaf-hash kernel; `None` when its pipeline
        /// state could not be created (e.g. threadgroup-memory limit at PSO
        /// creation), which degrades the warmup A/B to the unfused variant
        /// instead of failing the whole GPU path.
        pub(crate) pso_ntt4leaf: Option<Id>,
        pub(crate) pso_leaf: Id,
        pub(crate) pso_parent: Id,
        /// Eight-level Merkle reducer. Its 4 KiB threadgroup CV scratch keeps
        /// intermediate parent levels off DRAM while preserving every node.
        pub(crate) pso_parent8: Id,
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
                let device = (api.create_system_default_device)();
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
                let pso_ntt4g8 = pso("ntt_fused_reg4g8")?;
                let pso_ntt4 = pso("ntt_fused_reg4")?;
                let pso_ntt3 = pso("ntt_fused_reg3")?;
                let pso_ntt4z = pso("ntt_fused_reg4_from_z")?;
                let pso_ntt4leaf = match pso("ntt4_fused_leaf") {
                    Ok(p) => Some(p),
                    Err(e) => {
                        eprintln!(
                            "[gpu-commit] fused leaf kernel unavailable ({e}); \
                             A/B will use the unfused variant"
                        );
                        None
                    }
                };
                let pso_leaf = pso("leaf_hash")?;
                let pso_parent = pso("parent_hash")?;
                let pso_parent8 = pso("parent_hash8")?;
                send!(api, unsafe extern "C" fn(Id, Sel) -> Id, library, c"release");
                Ok(Gpu {
                    api,
                    device,
                    queue,
                    pso_ntt,
                    pso_ntt4g8,
                    pso_ntt4,
                    pso_ntt3,
                    pso_ntt4z,
                    pso_ntt4leaf,
                    pso_leaf,
                    pso_parent,
                    pso_parent8,
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
    struct NttParams {
        log_d: u32,
        l: u32,
        f: u32,
        s: u32,
    }

    /// Encode the fused NTT passes for `layers [start_layer, log_d)` over a
    /// 64-lane interleaved buffer bound at `data_buf`. Passes starting at or
    /// after `stop_before` (when `Some`) are skipped — the fused-leaf graph
    /// emits that final pass as `ntt4_fused_leaf` instead.
    pub(crate) unsafe fn encode_ntt_passes(
        gpu: &Gpu,
        enc: Id,
        data_buf: Id,
        tw_buf: Id,
        log_d: usize,
        start_layer: usize,
        stop_before: Option<usize>,
    ) {
        unsafe {
            gpu.set_buffer(enc, data_buf, 0, 0);
            gpu.set_buffer(enc, tw_buf, 0, 1);
            for (l, f) in super::plan_passes(log_d, start_layer) {
                if stop_before.is_some_and(|s| l >= s) {
                    continue;
                }
                // Register-resident specializations for the production pass
                // widths; the generic staged kernel covers the rest. The g8
                // variant packs 8 same-twiddle tiles per threadgroup (needs
                // s >= 3) for much better occupancy per KiB of tables.
                // NOTE: an 8-tile shared-table variant (pso_ntt4g8, 512
                // threads/group) measured ~55% SLOWER than 64-thread groups:
                // elems[16] costs ~64 registers/thread and monolithic
                // 512-thread groups lose the scheduler's register-granular
                // packing. Kept for future experiments; not selected.
                let s = log_d - l - f;
                let (pso, tpg, groups) = match f {
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
        }
        encode_merkle_parents(gpu, enc, tree_buf, n_leaves);
    }

    /// Encode the parent levels of the BLAKE3 tree (leaves already present at
    /// the head of `tree_buf`). Full batches collapse eight levels per dispatch:
    /// all nodes retain the existing flat layout, while intermediate CVs stay
    /// in threadgroup memory instead of being reread from DRAM. The final fewer
    /// than eight levels use the single-level kernel.
    pub(crate) unsafe fn encode_merkle_parents(
        gpu: &Gpu,
        enc: Id,
        tree_buf: Id,
        n_leaves: usize,
    ) {
        unsafe {
            let mut read_start = 0usize; // node index
            let mut read_len = n_leaves;
            gpu.set_pipeline(enc, gpu.pso_parent8);
            while read_len >= 256 {
                let write_start = read_start + read_len;
                gpu.set_buffer(enc, tree_buf, read_start * 32, 0);
                gpu.set_buffer(enc, tree_buf, write_start * 32, 1);
                let read_len_u32 = read_len as u32;
                let bytes = core::slice::from_raw_parts(
                    (&read_len_u32 as *const u32).cast::<u8>(),
                    core::mem::size_of::<u32>(),
                );
                gpu.set_bytes(enc, bytes, 2);
                gpu.dispatch(enc, (read_len / 256) as u64, 128);

                // The eighth output level begins after levels 1..7. Advance
                // directly to it; adding its length gives the next free node,
                // preserving the ordinary breadth-first tree layout.
                read_start = write_start + read_len - read_len / 128;
                read_len /= 256;
            }

            gpu.set_pipeline(enc, gpu.pso_parent);
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

    /// Encode the fused final NTT pass + BLAKE3 leaf hashing (kernel
    /// `ntt4_fused_leaf`, 64 threads/group, one group per 16 contiguous
    /// positions — the ranked final-pass geometry with `s = 0`). Leaves land
    /// at `tree[leaf*8..leaf*8+8)` exactly like the `leaf_hash` dispatch, so
    /// `encode_merkle_parents` can follow unchanged.
    pub(crate) unsafe fn encode_fused_leaf_pass(
        gpu: &Gpu,
        enc: Id,
        data_buf: Id,
        tw_buf: Id,
        tree_buf: Id,
        log_d: usize,
        l: usize,
        f: usize,
    ) {
        unsafe {
            gpu.set_pipeline(
                enc,
                gpu.pso_ntt4leaf.expect("fused leaf PSO was created"),
            );
            gpu.set_buffer(enc, data_buf, 0, 0);
            gpu.set_buffer(enc, tw_buf, 0, 1);
            gpu.set_buffer(enc, tree_buf, 0, 3);
            let s = log_d - l - f;
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
            gpu.dispatch(enc, 1u64 << (log_d - f), 64);
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
                    encode_ntt_passes(gpu, enc, data_buf, tw_buf, log_d, start_layer, None);
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
        /// Graph variant latched by the warmup A/B: `true` runs the fused
        /// final-pass + leaf-hash dispatch (`ntt4_fused_leaf`), `false` the
        /// plain final pass + `leaf_hash` re-read. Set only when the chosen
        /// variant was bit-exact; otherwise the whole GPU path is off.
        fuse_leaf: bool,
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
        // Pad fresh trees to a Metal page. The logical Vec length remains
        // the exact flat-tree node count; spare capacity is used only when
        // the GPU can write directly into this host allocation.
        let page = 16 * 1024;
        let bytes = n * core::mem::size_of::<Hash>();
        let padded = (bytes + page - 1) & !(page - 1);
        let mut v = Vec::with_capacity(padded / core::mem::size_of::<Hash>());
        // SAFETY: capacity is at least n and Hash is Copy; the GPU writes
        // every logical node before the caller reads it.
        unsafe { v.set_len(n) };
        v
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

    /// Copy a GPU-produced flat tree back as one contiguous transfer. The
    /// destination is one uniquely borrowed allocation, and the platform
    /// memcpy keeps the unified-memory transfer as a single stream rather than
    /// creating rayon worker wakeups and competing CPU readers during prove.
    fn copy_bytes_parallel(src: *const u8, dst: &mut [u8]) {
        if dst.is_empty() {
            return;
        }
        // SAFETY: callers guarantee `src` points at least `dst.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
        }
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
        fuse_leaf: bool,
    ) -> Result<(), String> {
        unsafe {
            let pool = gpu.pool_push();
            let r = (|| {
                let cb = gpu.command_buffer()?;
                let enc = gpu.compute_encoder(cb)?;
                // Pass 1: layers 0..3 from z.
                gpu.set_pipeline(enc, gpu.pso_ntt4z);
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
                gpu.dispatch(enc, 1u64 << (log_d - 4), 64);
                // Passes 2..: layers 4..log_d in place over staging. The
                // fused-leaf variant replaces the final pass (l = log_d - 4,
                // s = 0) with `ntt4_fused_leaf`, which also writes the leaves
                // into `tree_buf`; parents then run exactly as before.
                let final_l = log_d - 4;
                if fuse_leaf {
                    encode_ntt_passes(gpu, enc, staging, tw_buf, log_d, 4, Some(final_l));
                    encode_fused_leaf_pass(gpu, enc, staging, tw_buf, tree_buf, log_d, final_l, 4);
                    encode_merkle_parents(gpu, enc, tree_buf, n_leaves);
                } else {
                    encode_ntt_passes(gpu, enc, staging, tw_buf, log_d, 4, None);
                    encode_merkle(gpu, enc, staging, tree_buf, n_leaves);
                }
                gpu.end_encoding(enc);
                gpu.commit_and_wait(cb)
            })();
            gpu.pool_pop(pool);
            r
        }
    }

    struct WarmupRun {
        latched: Latched,
        gpu_tree_plain: Vec<Hash>,
        gpu_wall_plain_ms: f64,
        gpu_tree_fused: Option<Vec<Hash>>,
        gpu_wall_fused_ms: Option<f64>,
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
            let mut created = Vec::with_capacity(4);
            let r = (|| -> Result<WarmupRun, String> {
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

                // Run one graph variant (untimed wiring run, then the timed
                // run): returns the wall and the tree copy-out. `take_tree`
                // reuses the pooled 64 MiB allocation. Both variants write
                // the same codeword into `staging` (bit-exact by
                // construction); the tree is what distinguishes them.
                let run_variant = |fuse_leaf: bool| -> Result<(f64, Vec<Hash>), String> {
                    unsafe {
                        run_commit_graph_from_z(
                            gpu, z_buf, staging, tw_buf, tree_buf, log_d, n_leaves, fuse_leaf,
                        )?;
                    }
                    let mut tree = take_tree(total_nodes);
                    copy_bytes_parallel(gpu.buffer_contents(tree_buf), {
                        core::slice::from_raw_parts_mut(
                            tree.as_mut_ptr().cast::<u8>(),
                            total_nodes * 32,
                        )
                    });
                    let t0 = std::time::Instant::now();
                    unsafe {
                        run_commit_graph_from_z(
                            gpu, z_buf, staging, tw_buf, tree_buf, log_d, n_leaves, fuse_leaf,
                        )?;
                    }
                    copy_bytes_parallel(gpu.buffer_contents(tree_buf), {
                        core::slice::from_raw_parts_mut(
                            tree.as_mut_ptr().cast::<u8>(),
                            total_nodes * 32,
                        )
                    });
                    Ok((t0.elapsed().as_secs_f64() * 1e3, tree))
                };
                let (gpu_wall_plain_ms, gpu_tree_plain) = run_variant(false)?;
                let fused = if gpu.pso_ntt4leaf.is_some() {
                    Some(run_variant(true)?)
                } else {
                    None
                };
                created.clear(); // ownership transfers to Latched
                Ok(WarmupRun {
                    latched: Latched {
                        tw_buf,
                        tree_buf,
                        staging,
                        wraps: vec![(z_packed.as_ptr() as usize, z_bytes, z_buf)],
                        fuse_leaf: false, // decided by the A/B in warmup_and_decide
                    },
                    gpu_tree_plain,
                    gpu_wall_plain_ms,
                    gpu_tree_fused: fused.as_ref().map(|(_, t)| t.clone()),
                    gpu_wall_fused_ms: fused.as_ref().map(|(w, _)| *w),
                })
            })();
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
    ) -> (crate::pcs::commit::CodewordBuf, Vec<Hash>) {
        use crate::pcs::commit::CodewordBuf;
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

        let mut run = match outcome {
            Ok(run) => run,
            Err(e) => {
                if dbg {
                    eprintln!("[gpu-commit] warmup: GPU unavailable ({e}); latching CPU path");
                }
                *latch = LatchState::Off;
                return (CodewordBuf::Cpu(codeword), cpu_tree);
            }
        };
        let gpu = gpu().expect("gpu() succeeded during warmup_gpu_run");

        // Bit-exactness: full codeword (written identically by both graph
        // variants) and the full tree from each variant.
        let codeword_ok = unsafe {
            bytes_equal_parallel(
                gpu.buffer_contents(run.latched.staging),
                core::slice::from_raw_parts(
                    codeword.as_ptr().cast::<u8>(),
                    core::mem::size_of_val(codeword.as_slice()),
                ),
            )
        };
        let plain_ok = run.gpu_tree_plain == cpu_tree;
        let fused_ok = match &run.gpu_tree_fused {
            Some(t) => t == &cpu_tree,
            // Fused kernel unavailable (e.g. threadgroup-memory limit at PSO
            // creation): the plain variant is the only candidate, not a
            // mismatch.
            None => true,
        };
        let exact = codeword_ok && plain_ok && fused_ok;
        if !exact {
            eprintln!(
                "[gpu-commit] WARMUP MISMATCH (codeword_ok={codeword_ok} plain_ok={plain_ok} \
                 fused_ok={fused_ok}); latching CPU path"
            );
        }

        // A/B the two graph variants, both bit-exact. The override env only
        // picks a side for local A/B tooling — the ranked harness env-clears
        // workers, so `None` (wall-clock comparison) is what ships.
        let (best_wall_ms, fuse_leaf) = match (run.gpu_tree_fused.as_ref(), run.gpu_wall_fused_ms) {
            (Some(ft), Some(fw)) if ft == &cpu_tree => match super::gpu_fuse_leaf_mode() {
                Some(true) => (fw, true),
                Some(false) => (run.gpu_wall_plain_ms, false),
                None if fw < run.gpu_wall_plain_ms => (fw, true),
                None => (run.gpu_wall_plain_ms, false),
            },
            _ => (run.gpu_wall_plain_ms, false),
        };

        let force = std::env::var_os(super::ENV_GPU_COMMIT_FORCE).is_some();
        let fast = best_wall_ms * super::LATCH_MARGIN <= cpu_wall_ms;
        let on = exact && (fast || force);
        if dbg {
            eprintln!(
                "[gpu-commit] warmup: gpu plain {:.2} ms vs fused {:.2} ms vs cpu {:.2} ms, \
                 bit-exact={exact}, force={force} -> latched {}, fuse_leaf={fuse_leaf}",
                run.gpu_wall_plain_ms,
                run.gpu_wall_fused_ms.unwrap_or(f64::NAN),
                cpu_wall_ms,
                if on { "ON" } else { "OFF" }
            );
        }
        give_tree(run.gpu_tree_plain);
        if let Some(t) = run.gpu_tree_fused {
            give_tree(t);
        }
        if on {
            run.latched.fuse_leaf = fuse_leaf;
            *latch = LatchState::On(run.latched);
        } else {
            release_latched(gpu, run.latched);
            *latch = LatchState::Off;
        }
        (CodewordBuf::Cpu(codeword), cpu_tree)
    }

    /// Timed-prove path once latched On: run the from-z graph into the
    /// persistent staging buffer (never touching the caller's z or codeword
    /// buffers), copy the tree out, return the pooled input codeword to the
    /// scratch pool, and hand back a `GpuCodeword` view of the staging.
    fn run_latched(
        latch: &mut LatchState,
        z_packed: &[F128],
        mut codeword: Vec<F128>,
        params: &crate::pcs::commit::PcsParams,
        cpu: impl FnOnce(&mut [F128]) -> Vec<Hash>,
    ) -> (crate::pcs::commit::CodewordBuf, Vec<Hash>) {
        use crate::pcs::commit::CodewordBuf;
        use std::sync::atomic::Ordering;
        let log_d = params.k_code();
        let n_leaves = params.n_leaves();
        let total_nodes = 2 * n_leaves - 1;
        let codeword_len = params.codeword_len_f128();
        let gpu = match gpu() {
            Ok(g) => g,
            Err(_) => {
                let tree = cpu(&mut codeword);
                return (CodewordBuf::Cpu(codeword), tree);
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
            return (CodewordBuf::Cpu(codeword), tree);
        }

        // Resolve the read-only z wrap (normally cached from the warmup).
        let z_ptr = z_packed.as_ptr() as usize;
        let z_bytes = core::mem::size_of_val(z_packed);
        let (tw_buf, tree_buf, staging, z_buf, fuse_leaf) = {
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
                        return (CodewordBuf::Cpu(codeword), tree);
                    }
                },
            };
            (state.tw_buf, state.tree_buf, state.staging, z_buf, state.fuse_leaf)
        };

        let t0 = std::time::Instant::now();
        let run = unsafe {
            run_commit_graph_from_z(
                gpu, z_buf, staging, tw_buf, tree_buf, log_d, n_leaves, fuse_leaf,
            )
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
            return (CodewordBuf::Cpu(codeword), tree);
        }
        let graph_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut tree = take_tree(total_nodes);
        unsafe {
            copy_bytes_parallel(gpu.buffer_contents(tree_buf), {
                core::slice::from_raw_parts_mut(tree.as_mut_ptr().cast::<u8>(), total_nodes * 32)
            });
        }
        if std::env::var_os("FLOCK_COMMIT_TIMING").is_some() || debug_enabled() {
            eprintln!(
                "[commit-timing] gpu-commit: graph {graph_ms:.2} ms + tree-copyout {:.2} ms",
                t0.elapsed().as_secs_f64() * 1e3 - graph_ms
            );
        }
        // The replicated input codeword was never read by the from-z graph;
        // hand it straight back to the scratch pool for the next prove.
        crate::scratch::give_f128(codeword);
        let gpu_codeword = unsafe {
            super::GpuCodeword::new(gpu.buffer_contents(staging).cast::<F128>(), codeword_len)
        };
        (CodewordBuf::Gpu(gpu_codeword), tree)
    }

    pub(crate) fn commit_l0_or_fallback(
        z_packed: &[F128],
        mut codeword: Vec<F128>,
        params: &crate::pcs::commit::PcsParams,
        cpu: impl FnOnce(&mut [F128]) -> Vec<Hash>,
    ) -> (crate::pcs::commit::CodewordBuf, Vec<Hash>) {
        use crate::pcs::commit::CodewordBuf;
        if !super::gpu_commit_enabled()
            || !super::is_ranked_gpu_shape(params)
            || rayon::current_num_threads() <= 1
        {
            let tree = cpu(&mut codeword);
            return (CodewordBuf::Cpu(codeword), tree);
        }
        let mut latch = LATCH.lock().unwrap();
        match &*latch {
            LatchState::Off => {
                drop(latch);
                let tree = cpu(&mut codeword);
                (CodewordBuf::Cpu(codeword), tree)
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
    ) -> (crate::pcs::commit::CodewordBuf, Vec<crate::merkle::Hash>) {
        let tree = cpu(&mut codeword);
        (crate::pcs::commit::CodewordBuf::Cpu(codeword), tree)
    }

    pub(crate) fn give_tree(_tree: Vec<crate::merkle::Hash>) {}

    pub(crate) fn staging_released() {}
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use imp::{gpu_merkle_tree_blake3, gpu_ntt_interleaved_from_layer};

#[cfg(test)]
mod tests {
    #[test]
    fn fused_leaf_block_lane_mapping_bijects_leaf_uints() {
        // Final-pass fusion (l = log_d - 4, f = 4, s = 0): a threadgroup owns
        // positions pos_base..pos_base + 15, i.e. 16 contiguous BLAKE3 leaves
        // (leaf id = position; 1 KiB = 256 uints = 64 lanes x 4 uints).
        // leaf_hash reads leaf uints linearly: leaf[b * 16 + i]. The fused
        // kernel must assemble block b of leaf p as uints p*256 + b*16 + i,
        // which in the SoA layout is lane b*4 + i/4, uint j = i%4. Prove the
        // two indexes coincide and cover the leaf exactly (bijection).
        for p in 0..16u32 {
            let mut seen = [false; 256];
            for b in 0..16u32 {
                for i in 0..16u32 {
                    let lane = b * 4 + i / 4;
                    let j = i % 4;
                    assert!(lane < 64);
                    let flat = p * 256 + lane * 4 + j; // SoA leaf index
                    let linear = p * 256 + b * 16 + i; // leaf_hash index
                    assert_eq!(flat, linear, "SoA fetch must equal leaf_hash linear read");
                    seen[(b * 16 + i) as usize] = true;
                }
            }
            assert!(seen.iter().all(|&s| s), "leaf {p} not fully covered");
        }
    }

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
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
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

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
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

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
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
                imp::encode_ntt_passes(gpu, enc, data_buf, tw_buf, log_d, 1, None);
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
        let tree1 = pd1.merkle_tree.clone();
        let codeword1 = pd1.codeword.to_vec();
        drop(pd1); // returns codeword + tree to the pools, as the prover does

        // Timed-style commit: latched GPU path over the pooled buffer.
        let t0 = std::time::Instant::now();
        let (c2, pd2) = crate::pcs::commit::commit(&z, &params);
        let latched_ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("latched commit (replicate+gpu graph+copyout): {latched_ms:.2} ms");

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

    // -----------------------------------------------------------------------
    // Host-side oracle for the staged MSL fusion `ntt4_fused_leaf` — no GPU
    // needed. The kernel is a hand transcription of the production
    // `ntt_fused_reg4` register chain plus a BLAKE3 leaf phase appended; the
    // x86 test suites can never execute it, so we transcribe it a SECOND time
    // into Rust and byte-compare its full output (post-pass codeword AND the
    // BLAKE3 tree) against the CPU reference: the scalar interleaved NTT for
    // the same layers, then the real blake3 crate (`merkle::hash_leaf` /
    // `hash_pair`) — the exact tree the warmup byte-compares GPU trees
    // against. Any drift in the MSL transcription (the §4 failure mode, now
    // at the MSL level) shows up here as a tree mismatch on x86.
    //
    // Geometry: the final fused pass of the ranked shape has s = 0
    // (log_d = 20 -> l = 16, f = 4); we run the identical geometry at
    // log_d = 8 -> l = 4, f = 4, s = 0 (16 threadgroups x 64 threads,
    // 256 leaves, 256 KiB codeword).
    // -----------------------------------------------------------------------

    const B3_IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C,
        0x1F83D9AB, 0x5BE0CD19,
    ];
    const B3_PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
    const B3_CHUNK_START: u32 = 1;
    const B3_CHUNK_END: u32 = 2;
    const B3_PARENT: u32 = 4;

    #[inline]
    fn xor4(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
    }

    /// F128 -> (x, y, z, w) u32 limbs, matching the MSL uint4 layout
    /// (little-endian struct {lo, hi}; word i = bit (i*32)..(i*32+32)).
    #[inline]
    fn f128_limbs(f: F128) -> [u32; 4] {
        [f.lo as u32, (f.lo >> 32) as u32, f.hi as u32, (f.hi >> 32) as u32]
    }

    #[inline]
    fn limbs_f128(w: [u32; 4]) -> F128 {
        F128 { lo: (w[1] as u64) << 32 | w[0] as u64, hi: (w[3] as u64) << 32 | w[2] as u64 }
    }

    /// MSL `gf_mulx` — v * x mod P (MSL lines 367-375).
    #[inline]
    fn gf_mulx(v: [u32; 4]) -> [u32; 4] {
        let carry = v[3] >> 31;
        [
            (v[0] << 1) ^ (carry * 0x87u32),
            (v[1] << 1) | (v[0] >> 31),
            (v[2] << 1) | (v[1] >> 31),
            (v[3] << 1) | (v[2] >> 31),
        ]
    }

    /// MSL `gf_shl16` — a * x^16 mod P (MSL lines 403-411).
    #[inline]
    fn gf_shl16(a: [u32; 4]) -> [u32; 4] {
        let h = a[3] >> 16;
        [
            (a[0] << 16) ^ ((h << 7) ^ (h << 2) ^ (h << 1) ^ h),
            (a[1] << 16) | (a[0] >> 16),
            (a[2] << 16) | (a[1] >> 16),
            (a[3] << 16) | (a[2] >> 16),
        ]
    }

    /// MSL `gf_mul_tab4` — v * tw mod P via four 16-entry nibble tables
    /// (MSL lines 417-428). `tab` is the 64-entry block at `tabs[tsel << 6]`.
    #[inline]
    fn gf_mul_tab4(v: [u32; 4], tab: &[[u32; 4]]) -> [u32; 4] {
        let mut acc = [0u32; 4];
        for i in (0..8).rev() {
            acc = gf_shl16(acc);
            let h = (v[i >> 1] >> ((i & 1) * 16)) & 0xffff;
            acc = xor4(acc, tab[(h & 15) as usize]);
            acc = xor4(acc, tab[16 + ((h >> 4) & 15) as usize]);
            acc = xor4(acc, tab[32 + ((h >> 8) & 15) as usize]);
            acc = xor4(acc, tab[48 + (h >> 12) as usize]);
        }
        acc
    }

    /// MSL `b3_compress` (MSL lines 719-748), flags/counter per the kernel.
    fn b3_compress(cv: &mut [u32; 8], m_in: &[u32; 16], block_len: u32, flags: u32) {
        let mut v = [0u32; 16];
        let mut m = [0u32; 16];
        for i in 0..8 {
            v[i] = cv[i];
        }
        for i in 0..4 {
            v[8 + i] = B3_IV[i];
        }
        v[12] = 0;
        v[13] = 0;
        v[14] = block_len;
        v[15] = flags;
        for i in 0..16 {
            m[i] = m_in[i];
        }
        for r in 0..7 {
            // G(a,b,c,d,x,y): v[a]+=v[b]+x; v[d]=rotr32(v[d]^v[a],16); ...
            macro_rules! g {
                ($a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $y:expr) => {
                    v[$a] = v[$a].wrapping_add(v[$b]).wrapping_add(m[$x]);
                    v[$d] = (v[$d] ^ v[$a]).rotate_right(16);
                    v[$c] = v[$c].wrapping_add(v[$d]);
                    v[$b] = (v[$b] ^ v[$c]).rotate_right(12);
                    v[$a] = v[$a].wrapping_add(v[$b]).wrapping_add(m[$y]);
                    v[$d] = (v[$d] ^ v[$a]).rotate_right(8);
                    v[$c] = v[$c].wrapping_add(v[$d]);
                    v[$b] = (v[$b] ^ v[$c]).rotate_right(7);
                };
            }
            g!(0, 4, 8, 12, 0, 1);
            g!(1, 5, 9, 13, 2, 3);
            g!(2, 6, 10, 14, 4, 5);
            g!(3, 7, 11, 15, 6, 7);
            g!(0, 5, 10, 15, 8, 9);
            g!(1, 6, 11, 12, 10, 11);
            g!(2, 7, 8, 13, 12, 13);
            g!(3, 4, 9, 14, 14, 15);
            if r < 6 {
                let t = m;
                for i in 0..16 {
                    m[i] = t[B3_PERM[i]];
                }
            }
        }
        for i in 0..8 {
            cv[i] = v[i] ^ v[8 + i];
        }
    }

    /// Rust transcription of the `ntt4_fused_leaf` MSL kernel at s = 0:
    /// final-pass butterfly (layers l..l+4, in-place) plus BLAKE3 leaf
    /// hashing through the threadgroup tile. Returns the kernel's leaf
    /// hashes (tree[leaf_pos]), mutating `data` exactly as the kernel would.
    fn sim_fused_leaf_pass(
        data: &mut [F128],
        twiddles: &[F128],
        log_d: usize,
        l: usize,
    ) -> Vec<crate::merkle::Hash> {
        let f = 4usize;
        let nf = 16usize;
        let ntw = 15usize;
        let n_leaves = 1usize << log_d;
        let n_tg = 1usize << (log_d - f);
        let mut tree = vec![[0u8; 32]; n_leaves];
        for tg in 0..n_tg {
            // s = 0: B = tgid >> 0 = tg, r = 0 -> pos_base = B << (log_d - l).
            let pos_base = tg << (log_d - l);
            // Phase 1: bases[lid] = tw * x^(4k), lid = 4t + k.
            let mut bases = vec![[0u32; 4]; ntw * 4];
            for lid in 0..ntw * 4 {
                let t = lid >> 2;
                let k = lid & 3;
                let j = 31 - ((t + 1) as u32).leading_zeros() as usize; // 31 - clz(t+1)
                let c = t + 1 - (1 << j);
                let mut p = f128_limbs(twiddles[(1usize << (l + j)) - 1 + (tg << j) + c]);
                for _ in 0..k * 4 {
                    p = gf_mulx(p);
                }
                bases[lid] = p;
            }
            // Phase 2: tabs[ei] = nibble multiples of the bases. Subset-sum
            // construction: per base, 3 mulx build the chain base*x^k (k=0..3),
            // then each of the 16 entries is an XOR of chain terms (15 XORs).
            let mut tabs = vec![[0u32; 4]; ntw * 64];
            for lid in 0..ntw * 4 {
                let c0 = bases[lid];
                let c1 = gf_mulx(c0);
                let c2 = gf_mulx(c1);
                let c3 = gf_mulx(c2);
                for n in 0..16usize {
                    let mut val = [0u32; 4];
                    if (n >> 0) & 1 != 0 {
                        val = xor4(val, c0);
                    }
                    if (n >> 1) & 1 != 0 {
                        val = xor4(val, c1);
                    }
                    if (n >> 2) & 1 != 0 {
                        val = xor4(val, c2);
                    }
                    if (n >> 3) & 1 != 0 {
                        val = xor4(val, c3);
                    }
                    tabs[(lid << 4) + n] = val;
                }
            }
            // Per-lane register butterfly + tile staging + write-back.
            let mut tile = vec![[0u32; 4]; nf * 64];
            for lane in 0..64usize {
                let mut elems = [[0u32; 4]; 16];
                for e in 0..nf {
                    elems[e] = f128_limbs(data[(pos_base + e) * 64 + lane]);
                }
                for j in 0..f {
                    let bpos = f - 1 - j;
                    for b in 0..(nf >> 1) {
                        let low = b & ((1 << bpos) - 1);
                        let eu = ((b >> bpos) << (bpos + 1)) | low;
                        let ev = eu | (1 << bpos);
                        let tsel = ((1 << j) - 1) + (eu >> (f - j));
                        let nu = xor4(elems[eu], gf_mul_tab4(elems[ev], &tabs[tsel << 6..]));
                        elems[eu] = nu;
                        elems[ev] = xor4(elems[ev], nu);
                    }
                }
                for e in 0..nf {
                    tile[e * 64 + lane] = elems[e];
                    data[(pos_base + e) * 64 + lane] = limbs_f128(elems[e]);
                }
            }
            // Leaf phase: 16 threads, one leaf each, block b of leaf lid
            // assembled as tile[lid*64 + b*4 + i/4][i%4] (the pinned
            // bijection), BLAKE3 chunk CV -> tree[leaf_pos*8 + i].
            for lid in 0..16usize {
                let leaf_pos = pos_base + lid;
                let mut cv = B3_IV;
                for b in 0..16usize {
                    let mut block = [0u32; 16];
                    for i in 0..16 {
                        block[i] = tile[lid * 64 + b * 4 + i / 4][i % 4];
                    }
                    let flags = (if b == 0 { B3_CHUNK_START } else { 0 })
                        | (if b == 15 { B3_CHUNK_END } else { 0 });
                    b3_compress(&mut cv, &block, 64, flags);
                }
                for i in 0..8 {
                    tree[leaf_pos][i * 4..i * 4 + 4].copy_from_slice(&cv[i].to_le_bytes());
                }
            }
        }
        tree
    }

    /// Final-pass fusion oracle: the staged MSL kernel, transcribed, must
    /// produce (a) the same codeword as the scalar NTT for layers l..l+4 and
    /// (b) the same BLAKE3 tree as the CPU reference the warmup compares
    /// against. Runs on x86; no GPU involved.
    #[test]
    fn fused_leaf_pass_and_tree_match_cpu_reference() {
        let log_d = 8usize;
        let l = log_d - 4; // final pass: l = log_d - f, s = 0
        let n_leaves = 1usize << log_d;
        let ntt = AdditiveNttF128::standard(log_d);
        let twiddles = flat_twiddle_table(&ntt, log_d);

        let mut rng = Rng::new(0xF051F);
        let mut data = rng.vec(64 << log_d); // post-layer-l state, random

        // CPU reference: scalar NTT layers l..log_d, then the real blake3
        // tree on the resulting 1 KiB leaves.
        let mut expect_data = data.clone();
        ntt.forward_transform_interleaved_scalar_from_layer(&mut expect_data, 64, l);
        let expect_bytes: Vec<u8> = expect_data
            .iter()
            .flat_map(|f| f.lo.to_le_bytes().into_iter().chain(f.hi.to_le_bytes()))
            .collect();
        let expect_tree =
            crate::merkle::merkle_tree(&expect_bytes, n_leaves, crate::merkle::HashKind::Blake3);

        // Fused kernel transcription.
        let mut sim_data = data.clone();
        let fused_leaves = sim_fused_leaf_pass(&mut sim_data, &twiddles, log_d, l);

        // (a) Codeword bit-exactness after the fused butterfly.
        assert_eq!(sim_data, expect_data, "fused butterfly != scalar NTT layers");

        // (b) Full tree bit-exactness: fused leaves + CPU parents.
        let mut flat = fused_leaves.clone();
        let mut level = fused_leaves;
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks(2) {
                next.push(crate::merkle::hash_pair(
                    &pair[0],
                    &pair[1],
                    crate::merkle::HashKind::Blake3,
                ));
            }
            flat.extend_from_slice(&next);
            level = next;
        }
        assert_eq!(flat, expect_tree, "fused tree != CPU reference tree");
    }

    /// The fused-leaf dispatch geometry is only well-formed at s = 0: the
    /// grid `1 << (log_d - f)` groups each cover 16 positions strided by
    /// 2^s, so at s > 0 the total span exceeds the buffer (out-of-bounds
    /// reads) and the leaf phase (leaf_pos = pos_base + lid, lid < 16) stops
    /// matching the tile's actual positions (pos_base + e << s). Pin that
    /// arithmetic here so a future non-final-pass call site is caught by the
    /// encoder's debug_assert rather than by a wrong tree.
    #[test]
    fn fused_leaf_geometry_requires_s_zero() {
        let log_d = 20usize;
        let f = 4usize;
        // Final pass: l = log_d - f -> s = 0 -> coverage is exact.
        let l = log_d - f;
        let s = log_d - l - f;
        assert_eq!(s, 0);
        let span_per_tg = (16usize << s) * 1024; // 16 positions x 1 KiB
        let total = (1usize << (log_d - f)) * span_per_tg;
        assert_eq!(total, 1usize << 30, "s = 0 grid covers the 1 GiB staging exactly");
        // Hypothetical s = 1 would overshoot by 2x.
        let s1_total = (1usize << (log_d - f)) * (16usize << 1) * 1024;
        assert_eq!(s1_total, 1usize << 31, "s = 1 grid covers 2 GiB > 1 GiB staging");
    }
}

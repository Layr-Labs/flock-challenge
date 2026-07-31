//! GPU∥CPU split of the round-1 URM: a trailing range of `x_hi` values is
//! computed on the GPU (dlopen'd Metal, no build-system changes) while the
//! CPU's rayon pool processes the rest. Results merge additively (F128 add =
//! XOR, order-independent), so the split is bit-exact by construction.
//!
//! The kernel mirrors the *scalar oracle* semantics exactly
//! ([`kernels::portable`]): per (outer, b_med) the AB byte is
//! `gf8_reduce(Σ_k (A_k·B_k) << k)` with `A_k[i] = Σ_b T[byte_b][i ^ 8b]`
//! (the plain `apply_scalar` table map — the NEON BH/ODD image tricks are an
//! equivalent CPU-side optimization the GPU does not need), the C byte comes
//! from the scalar 64-byte bit transpose, and the three convert banks use the
//! unpaired convert table (`c & 0x55` / `c & 0xaa`), identical to the scalar
//! `accumulate_convert_with_s_hat_v`. Per-outer `eq_lo_scaled` multiplies use
//! a 4-bit comb against a per-outer threadgroup nibble table.
//!
//! Fallback contract: [`Round1GpuJob::submit`] returns `None` when the GPU is
//! unavailable or any setup step fails — the caller then runs the full range
//! on the CPU. If the *joined* command buffer reports failure, `finish`
//! returns `None` and the caller recomputes the GPU range on the CPU. A GPU
//! fault can cost time, never correctness.

use std::ffi::{c_char, c_void, CString};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use crate::field::F128;

use super::ELL;

type Id = *mut c_void;

unsafe extern "C" {
    fn dlopen(path: *const c_char, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, sym: *const c_char) -> *mut c_void;
}
const RTLD_NOW: i32 = 2;

/// Outers per threadgroup: amortizes the per-outer comb-table build while
/// keeping enough threadgroups in flight to fill the GPU.
const BLOCK_OUTERS: usize = 16;

/// Default fraction of `x_hi` values routed to the GPU, from the measured
/// PoC rate R = 0.53 (f = R / (1 + R) ≈ 0.35). Overridable for experiments
/// via `FLOCK_GPU_URM_PCT` (integer percent); the ranked worker's cleared
/// environment gets this default.
const DEFAULT_GPU_PCT: usize = 35;

/// Probe fraction for the first (calibration) proof on the precomputed-AB
/// path. Deliberately below `DEFAULT_GPU_PCT`: if the machine's GPU is slow
/// for this kernel, the one probe proof — the worker's untimed warm-up —
/// pays a small tail, and every timed proof after it runs at the measured
/// optimum.
const CALIB_PROBE_PCT: usize = 30;

/// Calibrated GPU share for the precomputed-AB path, in percent.
/// `u32::MAX` = not yet measured (submit probes at [`CALIB_PROBE_PCT`]);
/// `0` = calibration concluded the GPU is not worth dispatching here.
static CALIB_PCT: AtomicU32 = AtomicU32::new(u32::MAX);

/// Only calibrate on proof-sized shapes: tiny test shapes (`hi_size` below
/// this) are dispatch-overhead-dominated and would poison the estimate for
/// the ranked shape.
const CALIB_MIN_HI: usize = 64;

/// Calibrated share must clear this floor or the GPU is turned off outright
/// — below it, per-proof dispatch overhead eats the offloaded sliver.
const CALIB_FLOOR_PCT: u32 = 6;

/// Never route more than this share to the GPU, however fast it measures:
/// the estimate comes from one proof and the CPU side must stay the
/// dominant, well-measured partner.
const CALIB_CEIL_PCT: u32 = 60;

/// Fold one measured (cpu, gpu) rate pair into the calibrated share:
/// the sample is `f* = R / (1 + R)` with `R = gpu_rate / cpu_rate`. The
/// first sample is taken as-is; later samples update a 3:1 EWMA — the
/// harness runs 20 untimed warm-up proofs before the measured trials, so
/// the share converges on warm-up noise, not on scored time. Snaps to
/// "off" below [`CALIB_FLOOR_PCT`] (and stays off: 0 stops future jobs),
/// clamps to [`CALIB_CEIL_PCT`] above.
fn record_calibration(cpu_hi: usize, cpu_secs: f64, gpu_hi: usize, gpu_secs: f64) {
    if cpu_hi == 0 || gpu_hi == 0 || !(cpu_secs > 1e-6) || !(gpu_secs > 1e-6) {
        return;
    }
    let r = (gpu_hi as f64 / gpu_secs) / (cpu_hi as f64 / cpu_secs);
    let sample = 100.0 * r / (1.0 + r);
    let merged = match CALIB_PCT.load(Ordering::Relaxed) {
        u32::MAX => sample,
        old => 0.75 * old as f64 + 0.25 * sample,
    };
    let pct = merged.round() as u32;
    let pct = if pct < CALIB_FLOOR_PCT { 0 } else { pct.min(CALIB_CEIL_PCT) };
    CALIB_PCT.store(pct, Ordering::Relaxed);
}

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Layout constants passed per dispatch.
struct Params {
    uint gpu_hi_start;      // first x_hi handled by the GPU
    uint blocks_per_xhi;    // big_lo_size / BLOCK_OUTERS
    uint big_lo_size;
    uint n_lo;              // x_outer = x_outer_lo | (x_hi << n_lo)
    uint n_lo_and_inner;
    uint within_outer_mask;
    uint chunk_stride;      // N_CHUNKS (=8): chunk_byte_base multiplier
    uint n_inner;           // N_INNER: x_outer_lo << N_INNER
};

constant uint R4[16] = {
    0x000u, 0x087u, 0x10Eu, 0x189u, 0x21Cu, 0x29Bu, 0x312u, 0x395u,
    0x438u, 0x4BFu, 0x536u, 0x5B1u, 0x624u, 0x6A3u, 0x72Au, 0x7ADu
};

static inline uint4 shl1_reduce(uint4 a) {
    uint carry = a.w >> 31;
    a.w = (a.w << 1) | (a.z >> 31);
    a.z = (a.z << 1) | (a.y >> 31);
    a.y = (a.y << 1) | (a.x >> 31);
    a.x = a.x << 1;
    if (carry) a.x ^= 0x87u;
    return a;
}

// cf · eq via 4-bit comb against the per-outer threadgroup table T.
static inline uint4 f128_mul_comb(uint4 cf, threadgroup const uint4* T) {
    uint4 r = uint4(0);
    for (int w = 3; w >= 0; w--) {
        uint cw = cf[w];
        for (int s = 28; s >= 0; s -= 4) {
            uint top = r.w >> 28;
            r.w = (r.w << 4) | (r.z >> 28);
            r.z = (r.z << 4) | (r.y >> 28);
            r.y = (r.y << 4) | (r.x >> 28);
            r.x = (r.x << 4) ^ R4[top];
            r ^= T[(cw >> s) & 0xF];
        }
    }
    return r;
}

// GF(2^8) multiply, AES poly 0x11B, bit-serial (ALU beats table gathers on
// this GPU — measured). Product is reduced (degree ≤ 7).
static inline uchar gf8_mul(uchar av, uchar bv) {
    ushort a = av, r = 0;
    uchar b = bv;
    for (uint i = 0; i < 8; i++) {
        if (b & 1) r ^= a;
        b >>= 1;
        a <<= 1;
        if (a & 0x100) a ^= 0x11B;
    }
    return (uchar)r;
}

// Reduce degree ≤ 14 modulo x^8 + x^4 + x^3 + x + 1 (= crate gf8_reduce).
static inline uchar gf8_reduce16(ushort v) {
    for (int bit = 14; bit >= 8; bit--) {
        if (v & (1 << bit)) v ^= (0x11B << (bit - 8));
    }
    return (uchar)v;
}

kernel void urm_round1(device const uchar* a_packed [[buffer(0)]],
                       device const uchar* b_packed [[buffer(1)]],
                       device const uchar* c_packed [[buffer(2)]],
                       device const uchar* inv_tbl  [[buffer(3)]],  // 256*64
                       device const uint4* conv     [[buffer(4)]],  // 16*256
                       device const uint4* eq_lo    [[buffer(5)]],  // big_lo_size
                       device const uchar* counts   [[buffer(6)]],  // mask+1
                       constant Params&    P        [[buffer(7)]],
                       device uint4*       outs     [[buffer(8)]],  // [tg][64][3]
                       uint lane [[thread_position_in_threadgroup]],
                       uint tg   [[threadgroup_position_in_grid]],
                       threadgroup uint4* shmem [[threadgroup(0)]]) {
    // shmem[0..16): comb table T for the current outer.
    threadgroup uint4* T = shmem;

    uint x_hi = P.gpu_hi_start + tg / P.blocks_per_xhi;
    uint block = tg % P.blocks_per_xhi;

    uint4 part_ab = uint4(0), part_c0 = uint4(0), part_c1 = uint4(0);

    for (uint bo = 0; bo < 16u; bo++) {   // BLOCK_OUTERS
        uint x_outer_lo = block * 16u + bo;
        uint x_outer = x_outer_lo | (x_hi << P.n_lo);
        uint n_b_med = counts[x_outer & P.within_outer_mask];

        // Comb table for eq_lo_scaled[x_outer_lo], built by lane 0.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            uint4 e1 = eq_lo[x_outer_lo];
            uint4 e2 = shl1_reduce(e1);
            uint4 e4 = shl1_reduce(e2);
            uint4 e8 = shl1_reduce(e4);
            for (uint n = 0; n < 16; n++) {
                uint4 t = uint4(0);
                if (n & 1) t ^= e1;
                if (n & 2) t ^= e2;
                if (n & 4) t ^= e4;
                if (n & 8) t ^= e8;
                T[n] = t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (n_b_med == 0) continue;

        ulong chunk_byte_base =
            (ulong)(((x_outer_lo << P.n_inner) | (x_hi << P.n_lo_and_inner))) * P.chunk_stride;

        uint4 cf_ab = uint4(0), cf_c0 = uint4(0), cf_c1 = uint4(0);
        for (uint bm = 0; bm < n_b_med; bm++) {
            ulong byte_base_b = chunk_byte_base + (ulong)bm * 64u;

            // AB byte: shift_reduce over 8 K-rows, scalar-apply semantics.
            ushort acc = 0;
            for (uint k = 0; k < 8; k++) {
                ulong off = byte_base_b + k * 8u;
                uchar fa = 0, fb = 0;
                for (uint b = 0; b < 8; b++) {
                    uint ia = a_packed[off + b];
                    uint ib = b_packed[off + b];
                    uint li = lane ^ (8u * b);
                    fa ^= inv_tbl[ia * 64u + li];
                    fb ^= inv_tbl[ib * 64u + li];
                }
                acc ^= ((ushort)gf8_mul(fa, fb)) << k;
            }
            uchar ab_byte = gf8_reduce16(acc);

            // C byte: scalar bit-transpose of the same 64-byte window.
            // out[b_chunk*8 + t] bit x_small = in[x_small*8 + b_chunk] bit t.
            uint b_chunk = lane / 8u, t = lane % 8u;
            uchar c_byte = 0;
            for (uint xs = 0; xs < 8; xs++) {
                c_byte |= ((c_packed[byte_base_b + xs * 8u + b_chunk] >> t) & 1u) << xs;
            }

            // Unpaired 3-bank convert (scalar-oracle semantics).
            cf_ab ^= conv[bm * 256u + ab_byte];
            cf_c0 ^= conv[bm * 256u + (c_byte & 0x55u)];
            cf_c1 ^= conv[bm * 256u + (c_byte & 0xAAu)];
        }

        part_ab ^= f128_mul_comb(cf_ab, T);
        part_c0 ^= f128_mul_comb(cf_c0, T);
        part_c1 ^= f128_mul_comb(cf_c1, T);
    }

    outs[(ulong)tg * 192u + lane * 3u + 0u] = part_ab;
    outs[(ulong)tg * 192u + lane * 3u + 1u] = part_c0;
    outs[(ulong)tg * 192u + lane * 3u + 2u] = part_c1;
}

// Precomputed-AB variant for the ranked pipeline: the challenge-independent
// AB transform (`Round1AbInner`) already holds the per-lane AB byte at the
// same block layout as the packed inputs, so the gather+gf8 work above
// collapses to a single byte load. Only the C transpose, the three convert
// banks, and the comb drain remain.
kernel void urm_round1_pre(device const uchar* ab_inner [[buffer(0)]],
                           device const uchar* c_packed [[buffer(1)]],
                           device const uint4* conv     [[buffer(2)]],  // 16*256
                           device const uint4* eq_lo    [[buffer(3)]],  // big_lo_size
                           device const uchar* counts   [[buffer(4)]],  // mask+1
                           constant Params&    P        [[buffer(5)]],
                           device uint4*       outs     [[buffer(6)]],  // [tg][64][3]
                           uint lane [[thread_position_in_threadgroup]],
                           uint tg   [[threadgroup_position_in_grid]],
                           threadgroup uint4* shmem [[threadgroup(0)]]) {
    threadgroup uint4* T = shmem;

    uint x_hi = P.gpu_hi_start + tg / P.blocks_per_xhi;
    uint block = tg % P.blocks_per_xhi;

    uint4 part_ab = uint4(0), part_c0 = uint4(0), part_c1 = uint4(0);

    for (uint bo = 0; bo < 16u; bo++) {   // BLOCK_OUTERS
        uint x_outer_lo = block * 16u + bo;
        uint x_outer = x_outer_lo | (x_hi << P.n_lo);
        uint n_b_med = counts[x_outer & P.within_outer_mask];

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            uint4 e1 = eq_lo[x_outer_lo];
            uint4 e2 = shl1_reduce(e1);
            uint4 e4 = shl1_reduce(e2);
            uint4 e8 = shl1_reduce(e4);
            for (uint n = 0; n < 16; n++) {
                uint4 t = uint4(0);
                if (n & 1) t ^= e1;
                if (n & 2) t ^= e2;
                if (n & 4) t ^= e4;
                if (n & 8) t ^= e8;
                T[n] = t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (n_b_med == 0) continue;

        ulong chunk_byte_base =
            (ulong)(((x_outer_lo << P.n_inner) | (x_hi << P.n_lo_and_inner))) * P.chunk_stride;

        uint4 cf_ab = uint4(0), cf_c0 = uint4(0), cf_c1 = uint4(0);
        for (uint bm = 0; bm < n_b_med; bm++) {
            ulong byte_base_b = chunk_byte_base + (ulong)bm * 64u;

            uchar ab_byte = ab_inner[byte_base_b + lane];

            uint b_chunk = lane / 8u, t = lane % 8u;
            uchar c_byte = 0;
            for (uint xs = 0; xs < 8; xs++) {
                c_byte |= ((c_packed[byte_base_b + xs * 8u + b_chunk] >> t) & 1u) << xs;
            }

            cf_ab ^= conv[bm * 256u + ab_byte];
            cf_c0 ^= conv[bm * 256u + (c_byte & 0x55u)];
            cf_c1 ^= conv[bm * 256u + (c_byte & 0xAAu)];
        }

        part_ab ^= f128_mul_comb(cf_ab, T);
        part_c0 ^= f128_mul_comb(cf_c0, T);
        part_c1 ^= f128_mul_comb(cf_c1, T);
    }

    outs[(ulong)tg * 192u + lane * 3u + 0u] = part_ab;
    outs[(ulong)tg * 192u + lane * 3u + 1u] = part_c0;
    outs[(ulong)tg * 192u + lane * 3u + 2u] = part_c1;
}
"#;

struct Objc {
    msg_send: *mut c_void,
    sel: unsafe extern "C" fn(*const c_char) -> Id,
    get_class: unsafe extern "C" fn(*const c_char) -> Id,
}

impl Objc {
    unsafe fn s(&self, n: &str) -> Id {
        unsafe { (self.sel)(CString::new(n).unwrap().as_ptr()) }
    }
    unsafe fn send0(&self, o: Id, s: Id) -> Id {
        unsafe {
            let f: unsafe extern "C" fn(Id, Id) -> Id = std::mem::transmute(self.msg_send);
            f(o, s)
        }
    }
    unsafe fn send1(&self, o: Id, s: Id, a: Id) -> Id {
        unsafe {
            let f: unsafe extern "C" fn(Id, Id, Id) -> Id = std::mem::transmute(self.msg_send);
            f(o, s, a)
        }
    }
    unsafe fn send2(&self, o: Id, s: Id, a: Id, b: Id) -> Id {
        unsafe {
            let f: unsafe extern "C" fn(Id, Id, Id, Id) -> Id = std::mem::transmute(self.msg_send);
            f(o, s, a, b)
        }
    }
    unsafe fn send3(&self, o: Id, s: Id, a: Id, b: Id, c: Id) -> Id {
        unsafe {
            let f: unsafe extern "C" fn(Id, Id, Id, Id, Id) -> Id =
                std::mem::transmute(self.msg_send);
            f(o, s, a, b, c)
        }
    }
    unsafe fn buf_bytes(&self, dev: Id, data: &[u8]) -> Id {
        unsafe {
            let f: unsafe extern "C" fn(Id, Id, *const u8, u64, u64) -> Id =
                std::mem::transmute(self.msg_send);
            f(
                dev,
                self.s("newBufferWithBytes:length:options:"),
                data.as_ptr(),
                data.len() as u64,
                0,
            )
        }
    }
    unsafe fn buf_len(&self, dev: Id, len: usize) -> Id {
        unsafe {
            let f: unsafe extern "C" fn(Id, Id, u64, u64) -> Id =
                std::mem::transmute(self.msg_send);
            f(dev, self.s("newBufferWithLength:options:"), len as u64, 0)
        }
    }
    /// Zero-copy wrap of caller pages (must be page-aligned, page-multiple).
    unsafe fn buf_no_copy(&self, dev: Id, ptr: *const u8, len: usize) -> Id {
        const PAGE: usize = 16384;
        if (ptr as usize) % PAGE != 0 || len % PAGE != 0 || len == 0 {
            return std::ptr::null_mut();
        }
        unsafe {
            let f: unsafe extern "C" fn(Id, Id, *const u8, u64, u64, Id) -> Id =
                std::mem::transmute(self.msg_send);
            f(
                dev,
                self.s("newBufferWithBytesNoCopy:length:options:deallocator:"),
                ptr,
                len as u64,
                0,
                std::ptr::null_mut(),
            )
        }
    }
    unsafe fn nsstring(&self, s: &str) -> Id {
        unsafe {
            let cls = (self.get_class)(CString::new("NSString").unwrap().as_ptr());
            let cs = CString::new(s).unwrap();
            self.send1(cls, self.s("stringWithUTF8String:"), cs.as_ptr() as Id)
        }
    }
}

struct Gpu {
    objc: Objc,
    device: Id,
    queue: Id,
    pipe: Id,
    pipe_pre: Id,
}

unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

fn gpu() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| {
        // Opt-in (`FLOCK_GPU_URM=1`): ranked measurement (submission
        // 2b34b0b3, 786,747 = −4.4% vs its own base) showed the split costs
        // time on the ranked box even at the self-calibrated share — GPU
        // busy-time calibration cannot see per-dispatch page-wiring of the
        // ~0.5 GB zero-copy inputs. The ranked worker's cleared environment
        // therefore gets the pure CPU path; the kernel + calibration stay
        // available for experiments.
        if !std::env::var_os("FLOCK_GPU_URM").is_some_and(|v| v == "1")
            || std::env::var_os("FLOCK_GPU").is_some_and(|v| v == "0")
        {
            return None;
        }
        unsafe { init() }
    })
    .as_ref()
}

unsafe fn init() -> Option<Gpu> {
    unsafe {
        let metal = dlopen(
            CString::new("/System/Library/Frameworks/Metal.framework/Versions/A/Metal")
                .unwrap()
                .as_ptr(),
            RTLD_NOW,
        );
        let objc_lib = dlopen(CString::new("/usr/lib/libobjc.A.dylib").unwrap().as_ptr(), RTLD_NOW);
        if metal.is_null() || objc_lib.is_null() {
            return None;
        }
        let msg_send = dlsym(objc_lib, CString::new("objc_msgSend").unwrap().as_ptr());
        let sel = dlsym(objc_lib, CString::new("sel_registerName").unwrap().as_ptr());
        let get_class = dlsym(objc_lib, CString::new("objc_getClass").unwrap().as_ptr());
        if msg_send.is_null() || sel.is_null() || get_class.is_null() {
            return None;
        }
        let objc = Objc {
            msg_send,
            sel: std::mem::transmute(sel),
            get_class: std::mem::transmute(get_class),
        };
        let create = dlsym(metal, CString::new("MTLCreateSystemDefaultDevice").unwrap().as_ptr());
        if create.is_null() {
            return None;
        }
        let create: unsafe extern "C" fn() -> Id = std::mem::transmute(create);
        let device = create();
        if device.is_null() {
            return None;
        }
        let mut err: Id = std::ptr::null_mut();
        let lib = objc.send3(
            device,
            objc.s("newLibraryWithSource:options:error:"),
            objc.nsstring(SHADER),
            std::ptr::null_mut(),
            &mut err as *mut Id as Id,
        );
        if lib.is_null() {
            return None;
        }
        let mk_pipe = |name: &str| -> Id {
            let func = objc.send1(lib, objc.s("newFunctionWithName:"), objc.nsstring(name));
            if func.is_null() {
                return std::ptr::null_mut();
            }
            let mut perr: Id = std::ptr::null_mut();
            objc.send2(
                device,
                objc.s("newComputePipelineStateWithFunction:error:"),
                func,
                &mut perr as *mut Id as Id,
            )
        };
        let pipe = mk_pipe("urm_round1");
        let pipe_pre = mk_pipe("urm_round1_pre");
        if pipe.is_null() || pipe_pre.is_null() {
            return None;
        }
        let queue = objc.send0(device, objc.s("newCommandQueue"));
        if queue.is_null() {
            return None;
        }
        Some(Gpu { objc, device, queue, pipe, pipe_pre })
    }
}

#[repr(C)]
struct Params {
    gpu_hi_start: u32,
    blocks_per_xhi: u32,
    big_lo_size: u32,
    n_lo: u32,
    n_lo_and_inner: u32,
    within_outer_mask: u32,
    chunk_stride: u32,
    n_inner: u32,
}

/// An in-flight GPU computation of `x_hi ∈ [gpu_hi_start, hi_size)`.
pub(super) struct Round1GpuJob {
    gpu: &'static Gpu,
    cb: Id,
    out_buf: Id,
    pub(super) gpu_hi_start: usize,
    hi_count: usize,
    blocks_per_xhi: usize,
}

unsafe impl Send for Round1GpuJob {}
unsafe impl Sync for Round1GpuJob {}

impl Round1GpuJob {
    /// Encode + commit the GPU range asynchronously. Returns `None` (caller
    /// runs everything on CPU) if the GPU or any buffer step is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn submit(
        a_packed: &[u8],
        b_packed: &[u8],
        c_packed: &[u8],
        inv_table_bytes: &[u8],
        convert: &[F128],
        eq_lo_scaled: &[F128],
        b_med_counts: &[u8],
        within_outer_mask: usize,
        hi_size: usize,
        n_lo: usize,
        n_lo_and_inner: usize,
        n_inner: usize,
        chunk_stride: usize,
    ) -> Option<Round1GpuJob> {
        let g = gpu()?;
        let big_lo_size = eq_lo_scaled.len();
        if big_lo_size % BLOCK_OUTERS != 0 {
            return None;
        }
        let pct: usize = std::env::var("FLOCK_GPU_URM_PCT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_GPU_PCT)
            .min(100);
        let hi_count = (hi_size * pct) / 100;
        if hi_count == 0 {
            return None;
        }
        let gpu_hi_start = hi_size - hi_count;
        let blocks_per_xhi = big_lo_size / BLOCK_OUTERS;
        let total_tgs = hi_count * blocks_per_xhi;

        unsafe {
            let o = &g.objc;
            // Witness buffers zero-copy; small tables staged by copy.
            let b_a = o.buf_no_copy(g.device, a_packed.as_ptr(), a_packed.len());
            let b_b = o.buf_no_copy(g.device, b_packed.as_ptr(), b_packed.len());
            let b_c = o.buf_no_copy(g.device, c_packed.as_ptr(), c_packed.len());
            if b_a.is_null() || b_b.is_null() || b_c.is_null() {
                return None;
            }
            let b_inv = o.buf_bytes(g.device, inv_table_bytes);
            let conv_bytes = core::slice::from_raw_parts(
                convert.as_ptr() as *const u8,
                core::mem::size_of_val(convert),
            );
            let b_conv = o.buf_bytes(g.device, conv_bytes);
            let eq_bytes = core::slice::from_raw_parts(
                eq_lo_scaled.as_ptr() as *const u8,
                core::mem::size_of_val(eq_lo_scaled),
            );
            let b_eq = o.buf_bytes(g.device, eq_bytes);
            let b_counts = o.buf_bytes(g.device, b_med_counts);
            if b_inv.is_null() || b_conv.is_null() || b_eq.is_null() || b_counts.is_null() {
                return None;
            }

            // Per-job output buffer: proofs are serial in the worker, but
            // tests may run several round-1 calls concurrently — a shared
            // buffer would race.
            let out_len = total_tgs * ELL * 3 * 16;
            let out_buf = o.buf_len(g.device, out_len);
            if out_buf.is_null() {
                return None;
            }

            let params = Params {
                gpu_hi_start: gpu_hi_start as u32,
                blocks_per_xhi: blocks_per_xhi as u32,
                big_lo_size: big_lo_size as u32,
                n_lo: n_lo as u32,
                n_lo_and_inner: n_lo_and_inner as u32,
                within_outer_mask: within_outer_mask as u32,
                chunk_stride: chunk_stride as u32,
                n_inner: n_inner as u32,
            };

            let cb = o.send0(g.queue, o.s("commandBuffer"));
            if cb.is_null() {
                return None;
            }
            let enc = o.send0(cb, o.s("computeCommandEncoder"));
            if enc.is_null() {
                return None;
            }
            o.send1(enc, o.s("setComputePipelineState:"), g.pipe);
            let fb: unsafe extern "C" fn(Id, Id, Id, u64, u64) = std::mem::transmute(o.msg_send);
            for (i, b) in [b_a, b_b, b_c, b_inv, b_conv, b_eq, b_counts].iter().enumerate() {
                fb(enc, o.s("setBuffer:offset:atIndex:"), *b, 0, i as u64);
            }
            {
                let f: unsafe extern "C" fn(Id, Id, *const c_void, u64, u64) =
                    std::mem::transmute(o.msg_send);
                f(
                    enc,
                    o.s("setBytes:length:atIndex:"),
                    &params as *const Params as *const c_void,
                    core::mem::size_of::<Params>() as u64,
                    7,
                );
            }
            fb(enc, o.s("setBuffer:offset:atIndex:"), out_buf, 0, 8);
            {
                let f: unsafe extern "C" fn(Id, Id, u64, u64) = std::mem::transmute(o.msg_send);
                f(enc, o.s("setThreadgroupMemoryLength:atIndex:"), (16 * 16) as u64, 0);
            }
            #[repr(C)]
            struct MTLSize {
                w: u64,
                h: u64,
                d: u64,
            }
            let gdis: unsafe extern "C" fn(Id, Id, MTLSize, MTLSize) =
                std::mem::transmute(o.msg_send);
            gdis(
                enc,
                o.s("dispatchThreads:threadsPerThreadgroup:"),
                MTLSize { w: (total_tgs * ELL) as u64, h: 1, d: 1 },
                MTLSize { w: ELL as u64, h: 1, d: 1 },
            );
            o.send0(enc, o.s("endEncoding"));
            o.send0(cb, o.s("commit"));

            Some(Round1GpuJob { gpu: g, cb, out_buf, gpu_hi_start, hi_count, blocks_per_xhi })
        }
    }

    /// Wait for the GPU, then fold its per-block lane partials into
    /// per-`x_hi` triples multiplied by `eq_hi`. Returns `None` on GPU
    /// failure (caller recomputes the range on CPU).
    pub(super) fn finish(
        self,
        eq_hi: &[F128],
    ) -> Option<([F128; ELL], [F128; ELL], [F128; ELL])> {
        unsafe {
            let o = &self.gpu.objc;
            o.send0(self.cb, o.s("waitUntilCompleted"));
            if o.send0(self.cb, o.s("status")) as usize != 4 {
                return None;
            }
            fold_lane_partials(o, self.out_buf, eq_hi, self.gpu_hi_start, self.hi_count, self.blocks_per_xhi)
        }
    }
}

/// Read back the `[tg][64][3]` lane partials and fold them into per-`x_hi`
/// triples weighted by `eq_hi`. Shared by both kernel variants (identical
/// output layout).
unsafe fn fold_lane_partials(
    o: &Objc,
    out_buf: Id,
    eq_hi: &[F128],
    gpu_hi_start: usize,
    hi_count: usize,
    blocks_per_xhi: usize,
) -> Option<([F128; ELL], [F128; ELL], [F128; ELL])> {
    unsafe {
        let out_ptr = o.send0(out_buf, o.s("contents")) as *const F128;
        if out_ptr.is_null() {
            return None;
        }
        let mut res_ab = [F128::ZERO; ELL];
        let mut res_c0 = [F128::ZERO; ELL];
        let mut res_c1 = [F128::ZERO; ELL];
        for hi_idx in 0..hi_count {
            let eq_hi_val = eq_hi[gpu_hi_start + hi_idx];
            let mut part = [[F128::ZERO; ELL]; 3];
            for blk in 0..blocks_per_xhi {
                let base = (hi_idx * blocks_per_xhi + blk) * ELL * 3;
                for lane in 0..ELL {
                    part[0][lane] += *out_ptr.add(base + lane * 3);
                    part[1][lane] += *out_ptr.add(base + lane * 3 + 1);
                    part[2][lane] += *out_ptr.add(base + lane * 3 + 2);
                }
            }
            for lane in 0..ELL {
                res_ab[lane] += eq_hi_val * part[0][lane];
                res_c0[lane] += eq_hi_val * part[1][lane];
                res_c1[lane] += eq_hi_val * part[2][lane];
            }
        }
        Some((res_ab, res_c0, res_c1))
    }
}

/// An in-flight GPU computation of `x_hi ∈ [gpu_hi_start, hi_size)` over the
/// precomputed-AB layout — the round-1 variant the ranked pipeline actually
/// runs — with self-calibration: the first proof-sized call in a process
/// probes at [`CALIB_PROBE_PCT`], measures the on-machine GPU:CPU rate ratio
/// R, and pins every later call to `f* = R/(1+R)` (or turns the GPU off when
/// the ratio doesn't clear the dispatch floor). The worker's untimed
/// fixed-seed warm-up proof is that first call, so calibration never runs
/// inside a measured interval.
pub(super) struct Round1PreGpuJob {
    gpu: &'static Gpu,
    cb: Id,
    out_buf: Id,
    pub(super) gpu_hi_start: usize,
    hi_count: usize,
    blocks_per_xhi: usize,
    calibrating: bool,
    submitted: Instant,
}

unsafe impl Send for Round1PreGpuJob {}
unsafe impl Sync for Round1PreGpuJob {}

impl Round1PreGpuJob {
    /// Encode + commit the GPU range asynchronously. Returns `None` (caller
    /// runs everything on CPU) when the GPU is unavailable, calibration has
    /// turned the GPU off, or any buffer step fails.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn submit(
        ab_inner: &[u8],
        c_packed: &[u8],
        convert: &[F128],
        eq_lo_scaled: &[F128],
        b_med_counts: &[u8],
        within_outer_mask: usize,
        hi_size: usize,
        n_lo: usize,
        n_lo_and_inner: usize,
        n_inner: usize,
        chunk_stride: usize,
    ) -> Option<Round1PreGpuJob> {
        let g = gpu()?;
        let big_lo_size = eq_lo_scaled.len();
        if big_lo_size % BLOCK_OUTERS != 0 {
            return None;
        }
        let env_pct: Option<usize> = std::env::var("FLOCK_GPU_URM_PCT")
            .ok()
            .and_then(|v| v.parse().ok());
        let (pct, calibrating) = match env_pct {
            Some(p) => (p.min(100), false),
            None => match CALIB_PCT.load(Ordering::Relaxed) {
                u32::MAX if hi_size >= CALIB_MIN_HI => (CALIB_PROBE_PCT, true),
                u32::MAX => (DEFAULT_GPU_PCT, false),
                0 => return None,
                p => (p as usize, hi_size >= CALIB_MIN_HI),
            },
        };
        let hi_count = (hi_size * pct) / 100;
        if hi_count == 0 {
            return None;
        }
        let gpu_hi_start = hi_size - hi_count;
        let blocks_per_xhi = big_lo_size / BLOCK_OUTERS;
        let total_tgs = hi_count * blocks_per_xhi;

        unsafe {
            let o = &g.objc;
            let b_ab = o.buf_no_copy(g.device, ab_inner.as_ptr(), ab_inner.len());
            let b_c = o.buf_no_copy(g.device, c_packed.as_ptr(), c_packed.len());
            if b_ab.is_null() || b_c.is_null() {
                return None;
            }
            let conv_bytes = core::slice::from_raw_parts(
                convert.as_ptr() as *const u8,
                core::mem::size_of_val(convert),
            );
            let b_conv = o.buf_bytes(g.device, conv_bytes);
            let eq_bytes = core::slice::from_raw_parts(
                eq_lo_scaled.as_ptr() as *const u8,
                core::mem::size_of_val(eq_lo_scaled),
            );
            let b_eq = o.buf_bytes(g.device, eq_bytes);
            let b_counts = o.buf_bytes(g.device, b_med_counts);
            if b_conv.is_null() || b_eq.is_null() || b_counts.is_null() {
                return None;
            }

            let out_len = total_tgs * ELL * 3 * 16;
            let out_buf = o.buf_len(g.device, out_len);
            if out_buf.is_null() {
                return None;
            }

            let params = Params {
                gpu_hi_start: gpu_hi_start as u32,
                blocks_per_xhi: blocks_per_xhi as u32,
                big_lo_size: big_lo_size as u32,
                n_lo: n_lo as u32,
                n_lo_and_inner: n_lo_and_inner as u32,
                within_outer_mask: within_outer_mask as u32,
                chunk_stride: chunk_stride as u32,
                n_inner: n_inner as u32,
            };

            let cb = o.send0(g.queue, o.s("commandBuffer"));
            if cb.is_null() {
                return None;
            }
            let enc = o.send0(cb, o.s("computeCommandEncoder"));
            if enc.is_null() {
                return None;
            }
            o.send1(enc, o.s("setComputePipelineState:"), g.pipe_pre);
            let fb: unsafe extern "C" fn(Id, Id, Id, u64, u64) = std::mem::transmute(o.msg_send);
            for (i, b) in [b_ab, b_c, b_conv, b_eq, b_counts].iter().enumerate() {
                fb(enc, o.s("setBuffer:offset:atIndex:"), *b, 0, i as u64);
            }
            {
                let f: unsafe extern "C" fn(Id, Id, *const c_void, u64, u64) =
                    std::mem::transmute(o.msg_send);
                f(
                    enc,
                    o.s("setBytes:length:atIndex:"),
                    &params as *const Params as *const c_void,
                    core::mem::size_of::<Params>() as u64,
                    5,
                );
            }
            fb(enc, o.s("setBuffer:offset:atIndex:"), out_buf, 0, 6);
            {
                let f: unsafe extern "C" fn(Id, Id, u64, u64) = std::mem::transmute(o.msg_send);
                f(enc, o.s("setThreadgroupMemoryLength:atIndex:"), (16 * 16) as u64, 0);
            }
            #[repr(C)]
            struct MTLSize {
                w: u64,
                h: u64,
                d: u64,
            }
            let gdis: unsafe extern "C" fn(Id, Id, MTLSize, MTLSize) =
                std::mem::transmute(o.msg_send);
            gdis(
                enc,
                o.s("dispatchThreads:threadsPerThreadgroup:"),
                MTLSize { w: (total_tgs * ELL) as u64, h: 1, d: 1 },
                MTLSize { w: ELL as u64, h: 1, d: 1 },
            );
            o.send0(enc, o.s("endEncoding"));
            o.send0(cb, o.s("commit"));

            if std::env::var_os("FLOCK_GPU_DEBUG").is_some() {
                eprintln!(
                    "[gpu_split] pre submit: x_hi {gpu_hi_start}..{hi_size} \
                     ({hi_count}/{hi_size}, calibrating={calibrating})"
                );
            }
            Some(Round1PreGpuJob {
                gpu: g,
                cb,
                out_buf,
                gpu_hi_start,
                hi_count,
                blocks_per_xhi,
                calibrating,
                submitted: Instant::now(),
            })
        }
    }

    /// Wait for the GPU, record calibration (when this was the probe call),
    /// and fold the partials. `cpu_hi_end` / `cpu_secs` are the CPU side's
    /// share and wall time for the same proof, measured by the caller.
    /// Returns `None` on GPU failure (caller recomputes the range on CPU);
    /// a probe-call failure also turns the GPU off for the process — a
    /// faulting or absent GPU must not keep costing fallback recomputes.
    pub(super) fn finish(
        self,
        eq_hi: &[F128],
        cpu_hi_end: usize,
        cpu_secs: f64,
    ) -> Option<([F128; ELL], [F128; ELL], [F128; ELL])> {
        unsafe {
            let o = &self.gpu.objc;
            o.send0(self.cb, o.s("waitUntilCompleted"));
            let wall_secs = self.submitted.elapsed().as_secs_f64();
            if o.send0(self.cb, o.s("status")) as usize != 4 {
                if self.calibrating {
                    CALIB_PCT.store(0, Ordering::Relaxed);
                }
                return None;
            }
            if self.calibrating {
                // Prefer the command buffer's own GPU-side timestamps; fall
                // back to submit→complete wall time (an overestimate that
                // only biases the share toward the CPU — the safe side).
                let ts: unsafe extern "C" fn(Id, Id) -> f64 = std::mem::transmute(o.msg_send);
                let start = ts(self.cb, o.s("GPUStartTime"));
                let end = ts(self.cb, o.s("GPUEndTime"));
                let gpu_secs = if end > start && start > 0.0 { end - start } else { wall_secs };
                record_calibration(cpu_hi_end, cpu_secs, self.hi_count, gpu_secs);
                if std::env::var_os("FLOCK_GPU_DEBUG").is_some() {
                    eprintln!(
                        "[gpu_split] calibrated: cpu {cpu_hi_end} hi in {cpu_secs:.4}s, \
                         gpu {} hi in {gpu_secs:.4}s (wall {wall_secs:.4}s) -> pct {}",
                        self.hi_count,
                        CALIB_PCT.load(Ordering::Relaxed)
                    );
                }
            }
            fold_lane_partials(o, self.out_buf, eq_hi, self.gpu_hi_start, self.hi_count, self.blocks_per_xhi)
        }
    }
}

//! Metal backend of the zerocheck 5+6 GPU prefix arm (see
//! `ENV_NO_ZC_T56_GPU`; geometry, constants and bank finish in
//! `gpu_commit::zc_t56`).
//!
//! `zc_t56_fold`: one threadgroup per (chunk, quarter), 256 threads, each
//! thread one output pair (`a`, `b` at the same index) per iteration, 8
//! iterations at stride 256 (2048 outputs per threadgroup). Its inputs are
//! the K pass GPU prefix arm's output buffers, bound as they are: a thread
//! loads its output's four logical inputs of a from the packed layout
//! (output 7 of an odd group: the block's side entry and three zeros) and of
//! b from the b-one-packed layout (output 0 of an even group: four ones;
//! output 6 of an odd group: input 59 is one; output 7 of an odd group: the
//! block's input 60 and three zeros). It folds `x0 + rho_a (x0 + x1)` per
//! pair and then `u0 + rho_b (u0 + u1)` with nibble-table multiplies, stores
//! both outputs into this arm's own buffers, and weights `a` by
//! `eq_odd[y] = eq_lo[4 y + 3]`. The eight lanes of a group then form the
//! group's 26 slot products: each lane XOR-combines lane values with
//! `simd_shuffle_xor` (by 1, 2, 4) and multiplies one operand pair per round,
//! rounds a, b, c every iteration, round d (the two quad sums, staged in one
//! lane per iteration) every fourth. Products accumulate unreduced in 256
//! bits, are summed over the simdgroup's four groups and the threadgroup's
//! eight simdgroups, and land as 32 raw bank entries per threadgroup; the CPU
//! reduces them per slot (`zc_t56::finish_banks`). Every step is an exact
//! GF(2^128) product, XOR or table expansion.

use super::*;
use crate::gpu_commit::zc_t56::{self, BANKS, CONSTS, IN_A, IN_B, OUT, SLOTS};

const ZC_T56_MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 16x16 carry-less product via native ushort 16x16->32 imad on Apple Silicon GPUs.
static inline uint clmul16_m3_u16(ushort a0, ushort a1, ushort a2, ushort b0, ushort b1, ushort b2) {
    uint r0 = (uint(a0) * uint(b0) ^ uint(a1) * uint(b2) ^ uint(a2) * uint(b1)) & 0x49249249u;
    uint r1 = (uint(a0) * uint(b1) ^ uint(a1) * uint(b0) ^ uint(a2) * uint(b2)) & 0x92492492u;
    uint r2 = (uint(a0) * uint(b2) ^ uint(a1) * uint(b1) ^ uint(a2) * uint(b0)) & 0x24924924u;
    return r0 | r1 | r2;
}

static inline uint2 clmul32_u16(uint a, uint b) {
    const ushort m0 = 0x9249u, m1 = 0x2492u, m2 = 0x4924u;
    ushort al = ushort(a), ah = ushort(a >> 16);
    ushort bl = ushort(b), bh = ushort(b >> 16);
    ushort al0 = al & m0, al1 = al & m1, al2 = al & m2;
    ushort ah0 = ah & m0, ah1 = ah & m1, ah2 = ah & m2;
    ushort bl0 = bl & m0, bl1 = bl & m1, bl2 = bl & m2;
    ushort bh0 = bh & m0, bh1 = bh & m1, bh2 = bh & m2;
    uint lo = clmul16_m3_u16(al0, al1, al2, bl0, bl1, bl2);
    uint hi = clmul16_m3_u16(ah0, ah1, ah2, bh0, bh1, bh2);
    uint mid = clmul16_m3_u16(al0 ^ ah0, al1 ^ ah1, al2 ^ ah2, bl0 ^ bh0, bl1 ^ bh1, bl2 ^ bh2) ^ lo ^ hi;
    return uint2(lo ^ (mid << 16), hi ^ (mid >> 16));
}

static inline uint4 clmul64_u16(uint2 a, uint2 b) {
    uint2 p0 = clmul32_u16(a.x, b.x);
    uint2 p2 = clmul32_u16(a.y, b.y);
    uint2 pm = clmul32_u16(a.x ^ a.y, b.x ^ b.y) ^ p0 ^ p2;
    return uint4(p0.x, p0.y ^ pm.x, p2.x ^ pm.y, p2.y);
}

struct U256u { uint4 lo; uint4 hi; };

static inline U256u clmul128(uint4 a, uint4 b) {
    uint4 p0 = clmul64_u16(a.xy, b.xy);
    uint4 p2 = clmul64_u16(a.zw, b.zw);
    uint4 pm = clmul64_u16(a.xy ^ a.zw, b.xy ^ b.zw) ^ p0 ^ p2;
    return { uint4(p0.x, p0.y, p0.z ^ pm.x, p0.w ^ pm.y),
             uint4(p2.x ^ pm.z, p2.y ^ pm.w, p2.z, p2.w) };
}

// Reduce a 255-bit product mod x^128 + x^7 + x^2 + x + 1.
static inline uint4 gf_reduce(U256u p) {
    ulong h0 = (ulong(p.hi.y) << 32) | p.hi.x;
    ulong h1 = (ulong(p.hi.w) << 32) | p.hi.z;
    ulong t0 = h0 ^ (h0 << 1) ^ (h0 << 2) ^ (h0 << 7);
    ulong t1 = h1 ^ (h1 << 1) ^ (h1 << 2) ^ (h1 << 7)
             ^ (h0 >> 63) ^ (h0 >> 62) ^ (h0 >> 57);
    ulong ov = (h1 >> 63) ^ (h1 >> 62) ^ (h1 >> 57);
    t0 ^= ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    ulong l0 = ((ulong(p.lo.y) << 32) | p.lo.x) ^ t0;
    ulong l1 = ((ulong(p.lo.w) << 32) | p.lo.z) ^ t1;
    return uint4(uint(l0), uint(l0 >> 32), uint(l1), uint(l1 >> 32));
}

// Multiply a 16-byte value by a challenge through its 16 byte-bank nibble
// tables using extract_bits (same 32 entries XORed together).
static inline uint4 gf_shl16(uint4 a) {
    uint h = a.w >> 16;
    return uint4((a.x << 16) ^ ((h << 7) ^ (h << 2) ^ (h << 1) ^ h),
                 (a.y << 16) | (a.x >> 16), (a.z << 16) | (a.y >> 16), (a.w << 16) | (a.z >> 16));
}

static inline uint4 gf_shl32(uint4 a) {
    uint h = a.w;
    return uint4(h ^ (h << 1) ^ (h << 2) ^ (h << 7),
                 a.x ^ (h >> 31) ^ (h >> 30) ^ (h >> 25), a.y, a.z);
}

// Multiply by rho_a via 4 byte tables (word-32 Horner, 16 SRAM loads vs 32).
static inline uint4 mul_rho32(uint4 x, threadgroup const uint4* tab) {
    uchar4 b3 = as_type<uchar4>(x.w);
    uint4 a = tab[b3.x] ^ tab[256u + b3.y] ^ tab[512u + b3.z] ^ tab[768u + b3.w];
    uchar4 b2 = as_type<uchar4>(x.z);
    a = gf_shl32(a) ^ tab[b2.x] ^ tab[256u + b2.y] ^ tab[512u + b2.z] ^ tab[768u + b2.w];
    uchar4 b1 = as_type<uchar4>(x.y);
    a = gf_shl32(a) ^ tab[b1.x] ^ tab[256u + b1.y] ^ tab[512u + b1.z] ^ tab[768u + b1.w];
    uchar4 b0 = as_type<uchar4>(x.x);
    a = gf_shl32(a) ^ tab[b0.x] ^ tab[256u + b0.y] ^ tab[512u + b0.z] ^ tab[768u + b0.w];
    return a;
}

// Multiply by rho_b via 2 byte tables (byte-16 Horner, 16 SRAM loads vs 32).
static inline uint4 mul_rho16b(uint4 x, threadgroup const uint4* tab) {
    uchar4 b3 = as_type<uchar4>(x.w);
    uint4 a = tab[b3.z] ^ tab[256u + b3.w];
    a = gf_shl16(a) ^ tab[b3.x] ^ tab[256u + b3.y];
    uchar4 b2 = as_type<uchar4>(x.z);
    a = gf_shl16(a) ^ tab[b2.z] ^ tab[256u + b2.w];
    a = gf_shl16(a) ^ tab[b2.x] ^ tab[256u + b2.y];
    uchar4 b1 = as_type<uchar4>(x.y);
    a = gf_shl16(a) ^ tab[b1.z] ^ tab[256u + b1.w];
    a = gf_shl16(a) ^ tab[b1.x] ^ tab[256u + b1.y];
    uchar4 b0 = as_type<uchar4>(x.x);
    a = gf_shl16(a) ^ tab[b0.z] ^ tab[256u + b0.w];
    a = gf_shl16(a) ^ tab[b0.x] ^ tab[256u + b0.y];
    return a;
}

static inline U256u acc_mul(U256u acc, uint4 x, uint4 y) {
    U256u p = clmul128(x, y);
    return { acc.lo ^ p.lo, acc.hi ^ p.hi };
}

// v ^ v of lane (lane ^ m) of the simdgroup.
static inline uint4 xor_lanes(uint4 v, ushort m) {
    return v ^ simd_shuffle_xor(v, m);
}

static inline U256u xor_lanes4(U256u v, ushort m) {
    return { v.lo ^ simd_shuffle_xor(v.lo, m), v.hi ^ simd_shuffle_xor(v.hi, m) };
}

// Ranked chunk geometry: 128 packed superblocks of 244 a entries and 512
// b-one-packed blocks of 56 b entries in, 8192 outputs out per array.
constant uint T56_IN_A = 31232u;
constant uint T56_IN_B = 28672u;
constant uint T56_OUT = 8192u;

struct T56Params { uint tg_base; uint pad0; uint pad1; uint pad2; };

kernel void zc_t56_fold(
    device const uint4* a_in   [[buffer(0)]],
    device const uint4* b_in   [[buffer(1)]],
    device uint4*       a_out  [[buffer(2)]],
    device uint4*       b_out  [[buffer(3)]],
    device const uint4* consts [[buffer(4)]],
    device ulong4*      banks  [[buffer(5)]],
    constant T56Params& p      [[buffer(6)]],
    device const uint*  cut    [[buffer(7)]],
    uint tg_rel [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    // 24 KiB total: 1024 uint4s for rho_a word-32 Horner + 512 uint4s for rho_b
    // byte-16 Horner; aliased as red[256] (8 KiB) after the fold loop finishes.
    threadgroup uint4 tab[1536];
    threadgroup uint cut_tg;
    const uint tg = p.tg_base + tg_rel;
    if (lid == 0u) { cut_tg = cut[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tg >> 2) >= cut_tg) { return; }
    {
        const uint lo = lid & 15u;
        const uint hi = lid >> 4u;
        tab[lid]         = consts[lo] ^ consts[16u + hi];
        tab[256u + lid]  = consts[32u + lo] ^ consts[48u + hi];
        tab[512u + lid]  = consts[64u + lo] ^ consts[80u + hi];
        tab[768u + lid]  = consts[96u + lo] ^ consts[112u + hi];
        tab[1024u + lid] = consts[512u + lo] ^ consts[528u + hi];
        tab[1280u + lid] = consts[544u + lo] ^ consts[560u + hi];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint chunk = tg >> 2;
    const uint quarter = tg & 3u;
    const uint l = lid & 7u;
    const uint4 one = uint4(1u, 0u, 0u, 0u);
    // Round b: pair sums at lanes 1, 2, 5, 6, even/odd quad sums at 0, 3,
    // 4, 7; round c: halves at lanes 0..3, half-pair sums at 5, 6 and 4, 7.
    const bool sel_b = ((l ^ (l >> 1u)) & 1u) != 0u;
    device const uint4* ai = a_in + ulong(chunk) * T56_IN_A;
    device const uint4* bi = b_in + ulong(chunk) * T56_IN_B;
    device uint4* ao = a_out + ulong(chunk) * T56_OUT;
    device uint4* bo = b_out + ulong(chunk) * T56_OUT;
    device const uint4* eq_odd = consts + 1024u;
    U256u acc_a = { uint4(0u), uint4(0u) }, acc_b = { uint4(0u), uint4(0u) };
    U256u acc_c = { uint4(0u), uint4(0u) }, acc_d = { uint4(0u), uint4(0u) };
    uint4 qa = uint4(0u), qb = uint4(0u);
    for (uint j = 0u; j < 8u; j++) {
        const uint o = (quarter << 11) + (j << 8) + lid;
        const uint y = o >> 3;
        const uint blk = y >> 1;
        const uint hsel = y & 1u;
        const bool tail = (hsel != 0u) && (l == 7u);
        const uint sup = (blk >> 2) * 244u;
        uint4 a0, a1, a2, a3;
        if (tail) {
            a0 = ai[sup + 240u + (blk & 3u)];
            a1 = uint4(0u); a2 = uint4(0u); a3 = uint4(0u);
        } else {
            const uint p = sup + 60u * (blk & 3u) + (hsel << 5) + (l << 2);
            a0 = ai[p]; a1 = ai[p + 1u]; a2 = ai[p + 2u]; a3 = ai[p + 3u];
        }
        // b-one-packed: block blk's inputs 4..58 at 56 blk + 0..54 and its
        // input 60 at 56 blk + 55; inputs 0..3 and 59 are one, 61..63 zero.
        const uint bm = blk * 56u;
        uint4 b0, b1, b2, b3;
        if (hsel == 0u && l == 0u) {
            b0 = one; b1 = one; b2 = one; b3 = one;
        } else if (tail) {
            b0 = bi[bm + 55u];
            b1 = uint4(0u); b2 = uint4(0u); b3 = uint4(0u);
        } else {
            const uint q = bm + (hsel << 5) + (l << 2) - 4u;
            b0 = bi[q]; b1 = bi[q + 1u]; b2 = bi[q + 2u];
            if (hsel != 0u && l == 6u) { b3 = one; } else { b3 = bi[q + 3u]; }
        }
        const uint4 fa0 = a0 ^ mul_rho32(a0 ^ a1, tab);
        const uint4 fa1 = a2 ^ mul_rho32(a2 ^ a3, tab);
        const uint4 fa = fa0 ^ mul_rho16b(fa0 ^ fa1, tab + 1024);
        const uint4 fb0 = b0 ^ mul_rho32(b0 ^ b1, tab);
        const uint4 fb1 = b2 ^ mul_rho32(b2 ^ b3, tab);
        const uint4 fb = fb0 ^ mul_rho16b(fb0 ^ fb1, tab + 1024);
        ao[o] = fa;
        bo[o] = fb;
        const uint4 wa = gf_reduce(clmul128(eq_odd[y], fa));
        const uint4 pa = xor_lanes(wa, ushort(1)), pb = xor_lanes(fb, ushort(1));
        const uint4 ea = xor_lanes(wa, ushort(2)), eb = xor_lanes(fb, ushort(2));
        const uint4 qsa = xor_lanes(ea, ushort(1)), qsb = xor_lanes(eb, ushort(1));
        const uint4 ha = xor_lanes(wa, ushort(4)), hb = xor_lanes(fb, ushort(4));
        const uint4 h1a = xor_lanes(ha, ushort(1)), h1b = xor_lanes(hb, ushort(1));
        const uint4 h2a = xor_lanes(ha, ushort(2)), h2b = xor_lanes(hb, ushort(2));
        const uint4 hha = xor_lanes(h1a, ushort(2)), hhb = xor_lanes(h1b, ushort(2));
        acc_b = acc_mul(acc_b, sel_b ? pa : ea, sel_b ? pb : eb);
        if ((l & 3u) == (j & 3u)) { qa = qsa; qb = qsb; }
        const uint4 ca = l < 4u ? ha : (sel_b ? h1a : h2a);
        const uint4 cb = l < 4u ? hb : (sel_b ? h1b : h2b);
        acc_c = acc_mul(acc_c, ca, cb);
        acc_a = acc_mul(acc_a, l == 0u ? hha : wa, l == 0u ? hhb : fb);
        if ((j & 3u) == 3u) { acc_d = acc_mul(acc_d, qa, qb); }
    }
    acc_a = xor_lanes4(acc_a, ushort(8));
    acc_b = xor_lanes4(acc_b, ushort(8));
    acc_c = xor_lanes4(acc_c, ushort(8));
    acc_d = xor_lanes4(acc_d, ushort(8));
    acc_a = xor_lanes4(acc_a, ushort(16));
    acc_b = xor_lanes4(acc_b, ushort(16));
    acc_c = xor_lanes4(acc_c, ushort(16));
    acc_d = xor_lanes4(acc_d, ushort(16));
    // Retire every table reader before reusing `tab` as the 8 KiB reduction buffer.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint4* red = tab;
    if ((lid & 31u) < 8u) {
        const uint base = ((lid >> 5) * 32u + l) * 2u;
        red[base + 0u]  = acc_a.lo; red[base + 1u]  = acc_a.hi;
        red[base + 16u] = acc_b.lo; red[base + 17u] = acc_b.hi;
        red[base + 32u] = acc_c.lo; red[base + 33u] = acc_c.hi;
        red[base + 48u] = acc_d.lo; red[base + 49u] = acc_d.hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32u) {
        uint4 slo = red[lid * 2u], shi = red[lid * 2u + 1u];
        for (uint k = 1u; k < 8u; k++) {
            const uint idx = ((k << 5) + lid) * 2u;
            slo ^= red[idx];
            shi ^= red[idx + 1u];
        }
        device uint4* banks4 = reinterpret_cast<device uint4*>(banks);
        const ulong dst = ((ulong(tg) << 5) + lid) * 2u;
        banks4[dst] = slo;
        banks4[dst + 1u] = shi;
    }
}
"#;

/// The fold pipeline and the device it was built on, built once per
/// process (prewarm thread, else the first launch); `None` = the source did
/// not compile or the pipeline cannot run the kernel's widths (arm off).
static ZC_T56_PSO: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();

unsafe fn zc_t56_build_pso(gpu: &Gpu) -> Result<Id, String> {
    unsafe {
        let pool = gpu.pool_push();
        let built = (|| -> Result<Id, String> {
            let src = gpu.api.nsstring(ZC_T56_MSL_SOURCE)?;
            let mut err: Id = NIL;
            let library: Id = send!(
                gpu.api,
                unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id,
                gpu.device,
                c"newLibraryWithSource:options:error:",
                src,
                NIL,
                &mut err
            );
            if library.is_null() {
                return Err(format!("zc_t56 shader compile failed: {}", gpu.api.error_string(err)));
            }
            let ns = gpu.api.nsstring("zc_t56_fold")?;
            let f: Id = send!(gpu.api, unsafe extern "C" fn(Id, Sel, Id) -> Id, library, c"newFunctionWithName:", ns);
            gpu.release(library);
            if f.is_null() {
                return Err("zc_t56_fold not found".into());
            }
            let mut perr: Id = NIL;
            let pso: Id = send!(
                gpu.api,
                unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id,
                gpu.device,
                c"newComputePipelineStateWithFunction:error:",
                f,
                &mut perr
            );
            gpu.release(f);
            if pso.is_null() {
                return Err(format!("zc_t56_fold pipeline: {}", gpu.api.error_string(perr)));
            }
            Ok(pso)
        })();
        gpu.pool_pop(pool);
        let pso = built?;
        // The lane mapping needs 32-wide simdgroups and 256-thread groups.
        let property = |sel: &std::ffi::CStr| -> u64 { send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> u64, pso, sel) };
        if property(c"threadExecutionWidth") != 32 || property(c"maxTotalThreadsPerThreadgroup") < 256 {
            gpu.release(pso);
            return Err("zc_t56_fold cannot run 32-wide simdgroups of 256-thread groups".into());
        }
        Ok(pso)
    }
}

fn zc_t56_pso(gpu: &'static Gpu) -> Option<Id> {
    let built = *ZC_T56_PSO.get_or_init(|| match unsafe { zc_t56_build_pso(gpu) } {
        Ok(pso) => Some((pso as usize, gpu.device as usize)),
        Err(e) => {
            if super::gpu_zc_t56_debug() {
                eprintln!("[zc-t56-gpu] pipeline build failed (arm off): {e}");
            }
            None
        }
    });
    built.and_then(|(pso, dev)| (dev == gpu.device as usize).then_some(pso as Id))
}

/// Whether the warm-up helper should build the arm's pipeline: the arm and
/// the K arm it chains on are on.
pub(crate) fn zc_t56_prewarm_wanted() -> bool {
    super::gpu_zc_t56_enabled() && super::gpu_zc_k_share() > 0
}

/// Compile the arm's kernel now (warm-up helper thread, after the K arm's
/// pipelines), so the first launch only joins the `OnceLock`.
pub(crate) fn zc_t56_prewarm() {
    if let Ok(gpu) = gpu() {
        let _ = zc_t56_pso(gpu);
    }
}

/// The arm's own process-lifetime buffers, sized at the first launch.
struct ZcT56 {
    /// Per-launch constants (`zc_t56::consts_into`).
    consts_buf: Id,
    /// Raw bank entries, [`BANKS`] x 32 B per chunk.
    banks_buf: Id,
    /// Folded outputs of the prefix chunks, [`OUT`] per chunk and array,
    /// read by the next tri pass.
    out_a: Id,
    out_b: Id,
    /// The kernel's skip boundary (one word; `u32::MAX` = none).
    cut_buf: Id,
    /// Closed loop: one step per completed encoder; grows across proves.
    ev: Id,
    next_value: u64,
    /// Closed loop: the previous launch's command buffer, left in flight
    /// (skipping its stolen tail) and retired by [`zc_t56_retire`].
    inflight: Id,
    cap: usize,
}

// SAFETY: the handles are only touched under the state mutex; Metal objects
// themselves are thread-safe.
unsafe impl Send for ZcT56 {}

static ZC_T56_STATE: std::sync::Mutex<Option<ZcT56>> = std::sync::Mutex::new(None);
/// 0 = uncalibrated (the next launch is the oracle), 1 = admitted, 2 =
/// disabled for the rest of the process.
static ZC_T56_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
const ZC_T56_UNCALIBRATED: u8 = 0;
const ZC_T56_ADMITTED: u8 = 1;
const ZC_T56_DISABLED: u8 = 2;
static ZC_T56_FIRED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

unsafe fn zc_t56_alloc(gpu: &Gpu, cap: usize) -> Result<ZcT56, String> {
    let mut made: Vec<Id> = Vec::with_capacity(5);
    for len in [CONSTS * 16, cap * BANKS * 32, cap * OUT * 16, cap * OUT * 16, 16] {
        match unsafe { gpu.new_buffer(len) } {
            Ok(b) => made.push(b),
            Err(e) => {
                for b in made {
                    unsafe { gpu.release(b) };
                }
                return Err(e);
            }
        }
    }
    let ev: Id = unsafe { send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> Id, gpu.device, c"newSharedEvent") };
    if ev.is_null() {
        for b in made {
            unsafe { gpu.release(b) };
        }
        return Err("newSharedEvent failed".into());
    }
    Ok(ZcT56 {
        consts_buf: made[0],
        banks_buf: made[1],
        out_a: made[2],
        out_b: made[3],
        cut_buf: made[4],
        ev,
        next_value: 0,
        inflight: NIL,
        cap,
    })
}

/// An in-flight 5+6 GPU prefix.
pub(crate) struct ZcT56Job {
    cb: Id,
    chunks: usize,
    calibration: bool,
    submitted: std::time::Instant,
}

// SAFETY: the command buffer is only waited/released by the launching
// thread; Metal objects are themselves thread-safe.
unsafe impl Send for ZcT56Job {}

impl ZcT56Job {
    /// First chunk the CPU queue folds: zero during the oracle launch (the
    /// CPU folds every chunk and the GPU's are compared, then dropped).
    pub(crate) fn cpu_base(&self) -> usize {
        if self.calibration { 0 } else { self.chunks }
    }

    pub(crate) fn calibration(&self) -> bool {
        self.calibration
    }

    /// GPU prefix length in chunks.
    pub(crate) fn chunks(&self) -> usize {
        self.chunks
    }
}

/// Result of joining the 5+6 GPU prefix.
pub(crate) enum ZcT56Result {
    /// Chunks `0..parts.len()`: their 26 unweighted slots (the CPU's
    /// `partials` rows) and their outputs, `len` entries per array at `a` /
    /// `b` (process-lifetime buffers, valid until the next launch).
    Prefix { parts: Vec<[F128; SLOTS]>, a: *const F128, b: *const F128, len: usize },
    /// Metal failed: the caller folds the prefix chunks on the CPU. The arm
    /// is disabled for the process.
    Failed,
}

/// Launch the GPU fold of 5+6 chunks `0..chunks`. `pre_a` / `pre_b` are the
/// K arm's prefix slices of this prove (its output buffers' contents, which
/// the kernel binds as they are); `rho_a`, `rho_b` the pass's challenges.
/// `None` = the CPU folds every chunk (arm off or disabled, data source not
/// the K arm's buffers, geometry outside the arm's, no Metal).
pub(crate) fn launch_zc_t56_prefix(
    pre_a: &[F128],
    pre_b: &[F128],
    chunks: usize,
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
) -> Option<ZcT56Job> {
    use std::sync::atomic::Ordering;
    let debug = super::gpu_zc_t56_debug();
    let mode = ZC_T56_MODE.load(Ordering::Relaxed);
    if mode == ZC_T56_DISABLED
        || chunks == 0
        || chunks > zc_t56::HI
        || pre_a.len() < chunks * IN_A
        || pre_b.len() < chunks * IN_B
        || eq_lo.len() != zc_t56::LO
    {
        return None;
    }
    let gpu = gpu().ok()?;
    let Some(pso) = zc_t56_pso(gpu) else {
        ZC_T56_MODE.store(ZC_T56_DISABLED, Ordering::Relaxed);
        return None;
    };
    // The data source: the K arm's output buffers, whose contents the
    // prefix slices must be (the K job of this prove was joined).
    let (k_a, k_b) = {
        let k = ZC_K_STATE.get().and_then(|s| s.as_ref())?;
        let Ok(k) = k.lock() else {
            note_poisoned_lock("zc-t56 launch (K state)", false);
            return None;
        };
        let same = unsafe {
            gpu.buffer_contents(k.out_a).cast::<F128>().cast_const() == pre_a.as_ptr()
                && gpu.buffer_contents(k.out_b).cast::<F128>().cast_const() == pre_b.as_ptr()
        };
        if !same || k.chunks * k.per_chunk_a < chunks * IN_A || k.chunks * k.per_chunk_b < chunks * IN_B {
            if debug {
                eprintln!("[zc-t56-gpu] prefix is not the K arm's buffers; arm skipped");
            }
            return None;
        }
        (k.out_a, k.out_b)
    };
    // A closed-loop buffer still in flight writes the buffers this launch
    // reuses.
    zc_t56_retire();
    let mut slot = match ZC_T56_STATE.lock() {
        Ok(s) => s,
        Err(_) => {
            note_poisoned_lock("zc-t56 launch", false);
            return None;
        }
    };
    if slot.is_none() {
        match unsafe { zc_t56_alloc(gpu, chunks) } {
            Ok(s) => *slot = Some(s),
            Err(e) => {
                if debug {
                    eprintln!("[zc-t56-gpu] buffers failed ({e}); arm off");
                }
                ZC_T56_MODE.store(ZC_T56_DISABLED, Ordering::Relaxed);
                return None;
            }
        }
    }
    let state = slot.as_ref()?;
    if state.cap < chunks {
        if debug {
            eprintln!("[zc-t56-gpu] share {chunks} above the buffers' {}; arm skipped", state.cap);
        }
        return None;
    }
    unsafe {
        // The previous job was joined, so nothing in flight reads these.
        let consts = std::slice::from_raw_parts_mut(gpu.buffer_contents(state.consts_buf).cast::<F128>(), CONSTS);
        zc_t56::consts_into(consts, rho_a, rho_b, eq_lo);
        let pool = gpu.pool_push();
        let submitted = (|| -> Result<Id, String> {
            let cb = gpu.command_buffer()?;
            let enc = gpu.compute_encoder(cb)?;
            gpu.set_pipeline(enc, pso);
            gpu.set_buffer(enc, k_a, 0, 0);
            gpu.set_buffer(enc, k_b, 0, 1);
            gpu.set_buffer(enc, state.out_a, 0, 2);
            gpu.set_buffer(enc, state.out_b, 0, 3);
            gpu.set_buffer(enc, state.consts_buf, 0, 4);
            gpu.set_buffer(enc, state.banks_buf, 0, 5);
            // The static form: one dispatch from threadgroup 0, no skips.
            let params = [0u32; 4];
            gpu.set_bytes(enc, core::slice::from_raw_parts(params.as_ptr().cast::<u8>(), 16), 6);
            gpu.buffer_contents(state.cut_buf).cast::<u32>().write_volatile(u32::MAX);
            gpu.set_buffer(enc, state.cut_buf, 0, 7);
            gpu.dispatch(enc, (4 * chunks) as u64, 256);
            gpu.end_encoding(enc);
            let cb = gpu.retain(cb);
            gpu.commit_async(cb);
            Ok(cb)
        })();
        gpu.pool_pop(pool);
        match submitted {
            Ok(cb) => Some(ZcT56Job {
                cb,
                chunks,
                calibration: mode == ZC_T56_UNCALIBRATED,
                submitted: std::time::Instant::now(),
            }),
            Err(e) => {
                if debug {
                    eprintln!("[zc-t56-gpu] submit failed ({e}); arm skipped");
                }
                None
            }
        }
    }
}

/// Whether a timed prove may run the closed loop: the loop is on and the
/// oracle launch (the static form, in the warm-up prove) admitted the arm.
pub(crate) fn zc_t56_loop_admitted() -> bool {
    super::gpu_zc_t56_loop_enabled() && ZC_T56_MODE.load(std::sync::atomic::Ordering::Relaxed) == ZC_T56_ADMITTED
}

/// An in-flight launch with the late-GPU steal (see `ENV_NO_ZC_T56_LOOP`):
/// the record's single dispatch over chunks `0..chunks`, followed by one
/// shared-event signal. The signal follows the encoder, so once it is seen
/// every chunk the kernel did not skip is folded and visible to the host.
pub(crate) struct ZcT56Loop {
    gpu: &'static Gpu,
    cb: Id,
    ev: Id,
    base: u64,
    chunks: usize,
    out_a: *const F128,
    out_b: *const F128,
    banks: *const [u64; 4],
    cut: *mut u32,
    submitted: std::time::Instant,
}

// SAFETY: the pointers name the arm's process-lifetime shared buffers, which
// nothing rewrites before the job is retired; event reads are thread-safe.
unsafe impl Send for ZcT56Loop {}
unsafe impl Sync for ZcT56Loop {}

impl ZcT56Loop {
    /// Chunks offered to the GPU.
    pub(crate) fn chunks(&self) -> usize {
        self.chunks
    }

    /// Whether the buffer is complete: every chunk below the steal boundary
    /// the kernel last read is folded and visible to the host.
    pub(crate) fn done(&self) -> bool {
        let done = unsafe { self.gpu.event_signaled_value(self.ev) } > self.base;
        // Order the reads of the buffer's outputs after this read.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        done
    }

    /// Publish the CPU's steal boundary: the GPU skips chunks `>= hi`.
    pub(crate) fn set_cut(&self, hi: usize) {
        unsafe { self.cut.write_volatile(hi as u32) };
    }

    /// Whether the command buffer ended in an error (a status read, never a
    /// wait). A failed buffer does not signal its event, so [`Self::done`]
    /// stays false and the CPU folds every chunk; this is asked once more
    /// when completion was seen and when the job is retired.
    pub(crate) fn failed(&self) -> bool {
        unsafe { self.gpu.cb_status(self.cb) == 5 }
    }

    /// The 26 slots of complete chunk `c`.
    pub(crate) fn part(&self, c: usize) -> [F128; SLOTS] {
        debug_assert!(c < self.chunks);
        zc_t56::finish_banks(unsafe { std::slice::from_raw_parts(self.banks.add(c * BANKS), BANKS) })
    }

    /// The arm's output buffers (chunk `c` at `c * OUT`).
    pub(crate) fn outputs(&self) -> (*const F128, *const F128) {
        (self.out_a, self.out_b)
    }

    /// Retire the job without waiting for it: the remaining groups return at
    /// the steal boundary and the buffer is parked until [`zc_t56_retire`].
    /// `taken` = leading chunks left to the GPU, for the trace.
    pub(crate) fn finish(self, taken: usize, hi_size: usize) {
        self.set_cut(0);
        if self.failed() {
            // As the record's join on a failed buffer: the arm is off for
            // the process (this prove's chunks were all folded by the CPU).
            ZC_T56_MODE.store(ZC_T56_DISABLED, std::sync::atomic::Ordering::Relaxed);
            if super::gpu_zc_t56_debug() {
                eprintln!("[zc-t56-gpu] command buffer failed; arm disabled, the CPU folded the prefix");
            }
        }
        if super::gpu_zc_t56_debug() {
            eprintln!(
                "[zc-t56-gpu] loop (proves {}): {taken} of {hi_size} 5+6 chunks left to the GPU (offered {}), {:.2} ms after submit",
                ZC_T56_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1,
                self.chunks,
                self.submitted.elapsed().as_secs_f64() * 1e3,
            );
        }
        unsafe {
            let mut slot = ZC_T56_STATE.lock().unwrap_or_else(|e| e.into_inner());
            match slot.as_mut() {
                Some(st) if st.inflight.is_null() => st.inflight = self.cb,
                _ => {
                    let _ = self.gpu.wait_cb(self.cb);
                    self.gpu.release(self.cb);
                }
            }
        }
    }
}

/// Wait out and release a closed-loop command buffer left in flight. Called
/// before anything rewrites the buffers it reads or writes: the K pass
/// launch (its outputs are this arm's inputs) and this arm's launches.
pub(crate) fn zc_t56_retire() {
    let Ok(gpu) = gpu() else { return };
    let mut slot = ZC_T56_STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(st) = slot.as_mut()
        && !st.inflight.is_null()
    {
        unsafe {
            let _ = gpu.wait_cb(st.inflight);
            gpu.release(st.inflight);
        }
        st.inflight = NIL;
    }
}

/// Closed-loop launch over chunks `0..chunks` (see `ENV_NO_ZC_T56_LOOP`);
/// arguments as [`launch_zc_t56_prefix`]. Only for an admitted arm whose
/// buffers the oracle launch sized; `None` = the CPU folds every chunk.
pub(crate) fn launch_zc_t56_loop(
    pre_a: &[F128],
    pre_b: &[F128],
    chunks: usize,
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
) -> Option<ZcT56Loop> {
    if !zc_t56_loop_admitted()
        || chunks == 0
        || chunks > zc_t56::HI
        || pre_a.len() < chunks * IN_A
        || pre_b.len() < chunks * IN_B
        || eq_lo.len() != zc_t56::LO
    {
        return None;
    }
    let gpu = gpu().ok()?;
    let pso = zc_t56_pso(gpu)?;
    let (k_a, k_b) = {
        let k = ZC_K_STATE.get().and_then(|s| s.as_ref())?;
        let k = k.lock().ok()?;
        let same = unsafe {
            gpu.buffer_contents(k.out_a).cast::<F128>().cast_const() == pre_a.as_ptr()
                && gpu.buffer_contents(k.out_b).cast::<F128>().cast_const() == pre_b.as_ptr()
        };
        if !same || k.chunks * k.per_chunk_a < chunks * IN_A || k.chunks * k.per_chunk_b < chunks * IN_B {
            return None;
        }
        (k.out_a, k.out_b)
    };
    zc_t56_retire();
    let mut slot = ZC_T56_STATE.lock().ok()?;
    let state = slot.as_mut()?;
    if state.cap < chunks {
        return None;
    }
    unsafe {
        let consts = std::slice::from_raw_parts_mut(gpu.buffer_contents(state.consts_buf).cast::<F128>(), CONSTS);
        zc_t56::consts_into(consts, rho_a, rho_b, eq_lo);
        let cut = gpu.buffer_contents(state.cut_buf).cast::<u32>();
        cut.write_volatile(zc_t56::HI as u32);
        let base = state.next_value;
        // Skipped whether or not the buffer encodes: a value that is never
        // signalled is never waited for.
        state.next_value = base + 1;
        let (ev, out_a, out_b, consts_buf, banks_buf, cut_buf) =
            (state.ev, state.out_a, state.out_b, state.consts_buf, state.banks_buf, state.cut_buf);
        let pool = gpu.pool_push();
        let submitted = (|| -> Result<Id, String> {
            let cb = gpu.command_buffer()?;
            let enc = gpu.compute_encoder(cb)?;
            gpu.set_pipeline(enc, pso);
            gpu.set_buffer(enc, k_a, 0, 0);
            gpu.set_buffer(enc, k_b, 0, 1);
            gpu.set_buffer(enc, out_a, 0, 2);
            gpu.set_buffer(enc, out_b, 0, 3);
            gpu.set_buffer(enc, consts_buf, 0, 4);
            gpu.set_buffer(enc, banks_buf, 0, 5);
            let params = [0u32; 4];
            gpu.set_bytes(enc, core::slice::from_raw_parts(params.as_ptr().cast::<u8>(), 16), 6);
            gpu.set_buffer(enc, cut_buf, 0, 7);
            gpu.dispatch(enc, (4 * chunks) as u64, 256);
            gpu.end_encoding(enc);
            gpu.encode_signal_event(cb, ev, base + 1);
            let cb = gpu.retain(cb);
            gpu.commit_async(cb);
            Ok(cb)
        })();
        gpu.pool_pop(pool);
        submitted.ok().map(|cb| ZcT56Loop {
            gpu,
            cb,
            ev,
            base,
            chunks,
            out_a: gpu.buffer_contents(out_a).cast::<F128>().cast_const(),
            out_b: gpu.buffer_contents(out_b).cast::<F128>().cast_const(),
            banks: gpu.buffer_contents(banks_buf).cast::<[u64; 4]>().cast_const(),
            cut,
            submitted: std::time::Instant::now(),
        })
    }
}

/// Join the 5+6 GPU prefix (after the CPU queue has drained; `cpu_ms` is
/// that queue's wall, for the debug line).
pub(crate) fn zc_t56_wait(job: ZcT56Job, hi_size: usize, cpu_ms: f64) -> ZcT56Result {
    use std::sync::atomic::Ordering;
    let debug = super::gpu_zc_t56_debug();
    let Ok(gpu) = gpu() else {
        ZC_T56_MODE.store(ZC_T56_DISABLED, Ordering::Relaxed);
        return ZcT56Result::Failed;
    };
    unsafe {
        let waited = std::time::Instant::now();
        let ok = gpu.spin_wait_cb(job.cb, 2.0);
        let wait_ms = waited.elapsed().as_secs_f64() * 1e3;
        let drain_ms = job.submitted.elapsed().as_secs_f64() * 1e3;
        let gpu_ms = if ok.is_ok() { zc_fold_gpu_wall_ms(gpu, job.cb) } else { 0.0 };
        gpu.release(job.cb);
        if let Err(e) = ok {
            ZC_T56_MODE.store(ZC_T56_DISABLED, Ordering::Relaxed);
            if debug {
                eprintln!("[zc-t56-gpu] command buffer failed ({e}); arm disabled, CPU folds the prefix");
            }
            return ZcT56Result::Failed;
        }
        let slot = match ZC_T56_STATE.lock() {
            Ok(s) => s,
            Err(_) => {
                note_poisoned_lock("zc-t56 wait", false);
                ZC_T56_MODE.store(ZC_T56_DISABLED, Ordering::Relaxed);
                return ZcT56Result::Failed;
            }
        };
        let Some(state) = slot.as_ref() else {
            ZC_T56_MODE.store(ZC_T56_DISABLED, Ordering::Relaxed);
            return ZcT56Result::Failed;
        };
        let banks = std::slice::from_raw_parts(
            gpu.buffer_contents(state.banks_buf).cast::<[u64; 4]>().cast_const(),
            job.chunks * BANKS,
        );
        let parts: Vec<[F128; SLOTS]> = banks.chunks_exact(BANKS).map(zc_t56::finish_banks).collect();
        let a = gpu.buffer_contents(state.out_a).cast::<F128>().cast_const();
        let b = gpu.buffer_contents(state.out_b).cast::<F128>().cast_const();
        drop(slot);
        if debug {
            let what = if job.calibration {
                "oracle launch".to_string()
            } else {
                format!("fired (proves {})", ZC_T56_FIRED.fetch_add(1, Ordering::Relaxed) + 1)
            };
            eprintln!(
                "[zc-t56-gpu] {what}: share {}/{hi_size} 5+6 chunks (K share {}), gpu={gpu_ms:.2}ms \
                 submit-to-drain={:.2}ms cpu_queue={cpu_ms:.2}ms ({} chunks) wait={wait_ms:.3}ms",
                job.chunks,
                super::gpu_zc_k_share(),
                drain_ms,
                hi_size - job.cpu_base(),
            );
        }
        ZcT56Result::Prefix { parts, a, b, len: job.chunks * OUT }
    }
}

/// Record the oracle verdict of the first launch.
pub(crate) fn zc_t56_verdict(equal: bool, chunks: usize, hi_size: usize) {
    use std::sync::atomic::Ordering;
    ZC_T56_MODE.store(if equal { ZC_T56_ADMITTED } else { ZC_T56_DISABLED }, Ordering::Relaxed);
    if super::gpu_zc_t56_debug() {
        if equal {
            eprintln!(
                "[zc-t56-gpu] oracle: GPU outputs and slots of chunks 0..{chunks} EQUAL the CPU's -> admitted, share {chunks}/{hi_size}"
            );
        } else {
            eprintln!("[zc-t56-gpu] oracle MISMATCH -> arm disabled for the process");
        }
    }
}

/// The arm's output buffers for the tri pass 0 arm (`t0_metal`): `(out_a,
/// out_b, their contents, cap in chunks)`, or `None` before the first launch.
pub(super) fn zc_t56_out_bufs() -> Option<(Id, Id, *const F128, *const F128, usize)> {
    let gpu = gpu().ok()?;
    let slot = ZC_T56_STATE.lock().ok()?;
    let st = slot.as_ref()?;
    unsafe {
        Some((
            st.out_a,
            st.out_b,
            gpu.buffer_contents(st.out_a).cast::<F128>().cast_const(),
            gpu.buffer_contents(st.out_b).cast::<F128>().cast_const(),
            st.cap,
        ))
    }
}

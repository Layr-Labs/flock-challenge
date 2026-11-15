//! Metal backend of the tri pass 0 GPU arm (see `ENV_NO_ZC_T0_GPU`).
//!
//! Tri pass 0 is the depth-three tri pass right after the composed 5+6
//! pass: 512 chunks, each folding 8192 plain inputs per array (eight per
//! output, challenges `rho_a`, `rho_b`, `rho_c`) into 1024 outputs and the
//! chunk's 26 message slots. Its leading inputs are the 5+6 arm's outputs,
//! which stay in that arm's buffers (`t56_metal`): pass-0 chunk `x` reads
//! exactly 5+6 chunk `x`'s 8192 outputs there, so every pass-0 chunk below
//! the 5+6 arm's kept share has its whole input on the GPU.
//!
//! The pass's challenges are Fiat-Shamir samples drawn after the 5+6
//! messages, so the GPU cannot start before the 5+6 pass has joined. The
//! command buffer is therefore encoded and committed early, on a queue of
//! its own (a buffer waiting on an event blocks every later buffer of its
//! queue), behind `encodeWaitForEvent`: when pass 0 begins, the CPU writes
//! the challenges' nibble tables, the group weights and the offer into
//! buffers the kernel reads, and sets the event; the GPU starts with no
//! encode or commit on the critical path.
//!
//! `zc_t0_fold`: one threadgroup per chunk, 256 threads, four iterations of
//! one output per thread; the fold is the three-level `x0 + rho (x0 + x1)`
//! cascade with nibble-table multiplies, then the eight lanes of a group
//! form the group's 26 slot products exactly as `zc_t56_fold` does. Each
//! group of [`ZC_T0_GROUP`] chunks is one encoder followed by a shared-event
//! signal; a chunk's outputs and banks are read only after its encoder's
//! signal. A threadgroup whose chunk is at or above `min(offer, cut)` returns
//! at once (`cut` = the lowest chunk the CPU has taken, read once per
//! threadgroup into threadgroup memory, so the whole group decides alike).
//! Every step is an exact GF(2^128) product, XOR or table expansion.

use super::*;
use crate::field::F256Unreduced;
use crate::gpu_commit::zc_t56::SLOT;

/// Outputs per pass-0 chunk and array.
pub(crate) const T0_OUT: usize = 1024;
/// Inputs per pass-0 chunk and array (eight per output).
pub(crate) const T0_IN: usize = 8 * T0_OUT;
/// Chunks of the pass.
pub(crate) const T0_HI: usize = 512;
/// Groups of eight outputs per chunk (weights `eq_lo[4 y + 3]`).
pub(crate) const T0_GROUPS: usize = T0_OUT / 8;
/// Per-launch constants: three nibble tables (512 entries each, the layout
/// of `zc_t56::consts_into`) and the group weights.
pub(crate) const T0_CONSTS: usize = 1536 + T0_GROUPS;
/// Raw bank entries per chunk (one threadgroup of eight simdgroups).
pub(crate) const T0_BANKS: usize = 32;
/// Chunks per encoder (one completion signal each).
pub(crate) const ZC_T0_GROUP: usize = 32;

const ZC_T0_MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline uint clmul16_m3(uint a, uint b) {
    const uint m0 = 0x9249u, m1 = 0x2492u, m2 = 0x4924u;
    uint a0 = a & m0, a1 = a & m1, a2 = a & m2;
    uint b0 = b & m0, b1 = b & m1, b2 = b & m2;
    uint r0 = (a0 * b0 ^ a1 * b2 ^ a2 * b1) & 0x49249249u;
    uint r1 = (a0 * b1 ^ a1 * b0 ^ a2 * b2) & 0x92492492u;
    uint r2 = (a0 * b2 ^ a1 * b1 ^ a2 * b0) & 0x24924924u;
    return r0 | r1 | r2;
}

static inline ulong clmul32(uint a, uint b) {
    uint al = a & 0xffffu, ah = a >> 16, bl = b & 0xffffu, bh = b >> 16;
    uint lo = clmul16_m3(al, bl);
    uint hi = clmul16_m3(ah, bh);
    uint mid = clmul16_m3(al ^ ah, bl ^ bh) ^ lo ^ hi;
    return ulong(lo) ^ (ulong(mid) << 16) ^ (ulong(hi) << 32);
}

struct U128k { ulong lo; ulong hi; };
struct U256k { ulong r0; ulong r1; ulong r2; ulong r3; };

static inline U128k clmul64(ulong a, ulong b) {
    uint al = uint(a), ah = uint(a >> 32);
    uint bl = uint(b), bh = uint(b >> 32);
    ulong p_lo = clmul32(al, bl);
    ulong p_hi = clmul32(ah, bh);
    ulong p_mid = clmul32(al ^ ah, bl ^ bh) ^ p_lo ^ p_hi;
    U128k r;
    r.lo = p_lo ^ (p_mid << 32);
    r.hi = p_hi ^ (p_mid >> 32);
    return r;
}

static inline U256k clmul128(uint4 a, uint4 b) {
    ulong al = (ulong(a.y) << 32) | a.x, ah = (ulong(a.w) << 32) | a.z;
    ulong bl = (ulong(b.y) << 32) | b.x, bh = (ulong(b.w) << 32) | b.z;
    U128k p0 = clmul64(al, bl);
    U128k p2 = clmul64(ah, bh);
    U128k pm = clmul64(al ^ ah, bl ^ bh);
    pm.lo ^= p0.lo ^ p2.lo;
    pm.hi ^= p0.hi ^ p2.hi;
    U256k r;
    r.r0 = p0.lo;
    r.r1 = p0.hi ^ pm.lo;
    r.r2 = p2.lo ^ pm.hi;
    r.r3 = p2.hi;
    return r;
}

static inline uint4 gf_reduce(U256k p) {
    ulong h0 = p.r2, h1 = p.r3;
    ulong t0 = h0 ^ (h0 << 1) ^ (h0 << 2) ^ (h0 << 7);
    ulong t1 = h1 ^ (h1 << 1) ^ (h1 << 2) ^ (h1 << 7)
             ^ (h0 >> 63) ^ (h0 >> 62) ^ (h0 >> 57);
    ulong ov = (h1 >> 63) ^ (h1 >> 62) ^ (h1 >> 57);
    t0 ^= ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    ulong l0 = p.r0 ^ t0, l1 = p.r1 ^ t1;
    return uint4(uint(l0), uint(l0 >> 32), uint(l1), uint(l1 >> 32));
}

static inline uint4 mul_rho16(uint4 x, threadgroup const uint4* nib) {
    uint4 acc = uint4(0u);
    uint w[4] = { x.x, x.y, x.z, x.w };
    for (uint word = 0u; word < 4u; word++) {
        for (uint j = 0u; j < 4u; j++) {
            uint b = (w[word] >> (8u * j)) & 0xffu;
            uint bank = word * 4u + j;
            acc ^= nib[bank * 32u + (b & 15u)] ^ nib[bank * 32u + 16u + (b >> 4u)];
        }
    }
    return acc;
}

static inline ulong4 acc_mul(ulong4 acc, uint4 x, uint4 y) {
    U256k p = clmul128(x, y);
    return acc ^ ulong4(p.r0, p.r1, p.r2, p.r3);
}

static inline uint4 xor_lanes(uint4 v, ushort m) {
    return v ^ simd_shuffle_xor(v, m);
}

static inline ulong4 xor_lanes4(ulong4 v, ushort m) {
    uint4 lo = uint4(uint(v.x), uint(v.y), uint(v.z), uint(v.w));
    uint4 hi = uint4(uint(v.x >> 32), uint(v.y >> 32), uint(v.z >> 32), uint(v.w >> 32));
    lo = simd_shuffle_xor(lo, m);
    hi = simd_shuffle_xor(hi, m);
    return v ^ ulong4(ulong(lo.x) | (ulong(hi.x) << 32), ulong(lo.y) | (ulong(hi.y) << 32),
                      ulong(lo.z) | (ulong(hi.z) << 32), ulong(lo.w) | (ulong(hi.w) << 32));
}

// Three-level fold of eight consecutive inputs.
static inline uint4 fold8(device const uint4* v, threadgroup const uint4* nib) {
    const uint4 c0 = v[0] ^ mul_rho16(v[0] ^ v[1], nib);
    const uint4 c1 = v[2] ^ mul_rho16(v[2] ^ v[3], nib);
    const uint4 c2 = v[4] ^ mul_rho16(v[4] ^ v[5], nib);
    const uint4 c3 = v[6] ^ mul_rho16(v[6] ^ v[7], nib);
    const uint4 d0 = c0 ^ mul_rho16(c0 ^ c1, nib + 512);
    const uint4 d1 = c2 ^ mul_rho16(c2 ^ c3, nib + 512);
    return d0 ^ mul_rho16(d0 ^ d1, nib + 1024);
}

constant uint T0_IN = 8192u;
constant uint T0_OUT = 1024u;

// ctl[0] = chunks offered, ctl[1] = lowest chunk the CPU has taken.
kernel void zc_t0_fold(
    device const uint4* a_in   [[buffer(0)]],
    device const uint4* b_in   [[buffer(1)]],
    device uint4*       a_out  [[buffer(2)]],
    device uint4*       b_out  [[buffer(3)]],
    device const uint4* consts [[buffer(4)]],
    device ulong4*      banks  [[buffer(5)]],
    device const uint*  ctl    [[buffer(6)]],
    constant uint&      tg_base [[buffer(7)]],
    uint tg_rel [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup uint4 nib[1536];
    threadgroup ulong4 red[256];
    const uint chunk = tg_base + tg_rel;
    if (lid == 0u) {
        const uint lim = min(ctl[0], ctl[1]);
        red[0] = ulong4(chunk >= lim ? 1ul : 0ul, 0ul, 0ul, 0ul);
    }
    for (uint i = lid; i < 1536u; i += 256u) { nib[i] = consts[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool skip = red[0].x != 0ul;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (skip) { return; }

    const uint l = lid & 7u;
    const bool sel_b = ((l ^ (l >> 1u)) & 1u) != 0u;
    device const uint4* ai = a_in + ulong(chunk) * T0_IN;
    device const uint4* bi = b_in + ulong(chunk) * T0_IN;
    device uint4* ao = a_out + ulong(chunk) * T0_OUT;
    device uint4* bo = b_out + ulong(chunk) * T0_OUT;
    device const uint4* eq_odd = consts + 1536u;
    ulong4 acc_a = ulong4(0ul), acc_b = ulong4(0ul), acc_c = ulong4(0ul), acc_d = ulong4(0ul);
    uint4 qa = uint4(0u), qb = uint4(0u);
    for (uint j = 0u; j < 4u; j++) {
        const uint o = (j << 8) + lid;
        const uint y = o >> 3;
        const uint4 fa = fold8(ai + 8u * o, nib);
        const uint4 fb = fold8(bi + 8u * o, nib);
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
    if ((lid & 31u) < 8u) {
        const uint base = (lid >> 5) * 32u + l;
        red[base] = acc_a;
        red[base + 8u] = acc_b;
        red[base + 16u] = acc_c;
        red[base + 24u] = acc_d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32u) {
        ulong4 s = red[lid];
        for (uint k = 1u; k < 8u; k++) { s ^= red[(k << 5) + lid]; }
        banks[(ulong(chunk) << 5) + lid] = s;
    }
}
"#;

static ZC_T0_PSO: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();

unsafe fn zc_t0_build_pso(gpu: &Gpu) -> Result<Id, String> {
    unsafe {
        let pool = gpu.pool_push();
        let built = (|| -> Result<Id, String> {
            let src = gpu.api.nsstring(ZC_T0_MSL_SOURCE)?;
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
                return Err(format!("zc_t0 shader compile failed: {}", gpu.api.error_string(err)));
            }
            let ns = gpu.api.nsstring("zc_t0_fold")?;
            let f: Id = send!(gpu.api, unsafe extern "C" fn(Id, Sel, Id) -> Id, library, c"newFunctionWithName:", ns);
            gpu.release(library);
            if f.is_null() {
                return Err("zc_t0_fold not found".into());
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
                return Err(format!("zc_t0_fold pipeline: {}", gpu.api.error_string(perr)));
            }
            Ok(pso)
        })();
        gpu.pool_pop(pool);
        let pso = built?;
        let property = |sel: &std::ffi::CStr| -> u64 { send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> u64, pso, sel) };
        if property(c"threadExecutionWidth") != 32 || property(c"maxTotalThreadsPerThreadgroup") < 256 {
            gpu.release(pso);
            return Err("zc_t0_fold cannot run 32-wide simdgroups of 256-thread groups".into());
        }
        Ok(pso)
    }
}

/// The pipeline if it is built (never builds: a timed prove must not wait
/// for a compile); `zc_t0_prewarm` builds it.
fn zc_t0_pso_ready(gpu: &'static Gpu) -> Option<Id> {
    ZC_T0_PSO
        .get()
        .copied()
        .flatten()
        .and_then(|(pso, dev)| (dev == gpu.device as usize).then_some(pso as Id))
}

/// Compile the arm's kernel (the 5+6 arm's warm-up helper thread, after
/// that arm's own kernel).
pub(crate) fn zc_t0_prewarm() {
    if !super::gpu_zc_t0_enabled() {
        return;
    }
    if let Ok(gpu) = gpu() {
        let _ = ZC_T0_PSO.get_or_init(|| match unsafe { zc_t0_build_pso(gpu) } {
            Ok(pso) => Some((pso as usize, gpu.device as usize)),
            Err(e) => {
                if super::gpu_zc_t0_debug() {
                    eprintln!("[zc-t0-gpu] pipeline build failed (arm off): {e}");
                }
                None
            }
        });
    }
}

/// Process-lifetime resources, made by the first launch.
struct ZcT0 {
    queue: Id,
    ev: Id,
    next_value: u64,
    consts_buf: Id,
    banks_buf: Id,
    out_a: Id,
    out_b: Id,
    ctl_buf: Id,
    cap: usize,
    /// A committed buffer not yet fired (waits for `parked_v`).
    parked: Option<(Id, u64)>,
    /// A fired or released buffer that may still run.
    inflight: Id,
}

// SAFETY: the handles are only touched under the state mutex; Metal objects
// themselves are thread-safe.
unsafe impl Send for ZcT0 {}

static ZC_T0_STATE: std::sync::Mutex<Option<ZcT0>> = std::sync::Mutex::new(None);
/// 0 = uncalibrated (the next launch is the oracle), 1 = admitted, 2 =
/// disabled for the rest of the process.
static ZC_T0_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
const ZC_T0_UNCALIBRATED: u8 = 0;
const ZC_T0_ADMITTED: u8 = 1;
const ZC_T0_DISABLED: u8 = 2;
static ZC_T0_TRIED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn t0_lock() -> Option<std::sync::MutexGuard<'static, Option<ZcT0>>> {
    match ZC_T0_STATE.lock() {
        Ok(s) => Some(s),
        Err(_) => {
            note_poisoned_lock("zc-t0", false);
            None
        }
    }
}

unsafe fn zc_t0_alloc(gpu: &Gpu, cap: usize) -> Option<ZcT0> {
    unsafe {
        let queue: Id = send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> Id, gpu.device, c"newCommandQueue");
        if queue.is_null() {
            return None;
        }
        let ev: Id = send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> Id, gpu.device, c"newSharedEvent");
        if ev.is_null() {
            return None;
        }
        let mut made: Vec<Id> = Vec::with_capacity(5);
        for len in [T0_CONSTS * 16, cap * T0_BANKS * 32, cap * T0_OUT * 16, cap * T0_OUT * 16, 16] {
            match gpu.new_buffer(len) {
                Ok(b) => made.push(b),
                Err(_) => {
                    for b in made {
                        gpu.release(b);
                    }
                    return None;
                }
            }
        }
        Some(ZcT0 {
            queue,
            ev,
            next_value: 0,
            consts_buf: made[0],
            banks_buf: made[1],
            out_a: made[2],
            out_b: made[3],
            ctl_buf: made[4],
            cap,
            parked: None,
            inflight: NIL,
        })
    }
}

/// Wait out and release a buffer left in flight (called before anything
/// rewrites what it reads or writes).
unsafe fn retire_locked(gpu: &Gpu, st: &mut ZcT0) {
    if !st.inflight.is_null() {
        unsafe {
            let _ = gpu.wait_cb(st.inflight);
            gpu.release(st.inflight);
        }
        st.inflight = NIL;
    }
}

/// Release a parked buffer nobody fired: offer 0 (every threadgroup returns
/// at once), set its event value; it then drains as an in-flight buffer.
unsafe fn release_parked_locked(gpu: &Gpu, st: &mut ZcT0) {
    if let Some((cb, v)) = st.parked.take() {
        unsafe {
            retire_locked(gpu, st);
            let ctl = gpu.buffer_contents(st.ctl_buf).cast::<u32>();
            ctl.write_volatile(0);
            ctl.add(1).write_volatile(0);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            send!(gpu.api, unsafe extern "C" fn(Id, Sel, u64), st.ev, c"setSignaledValue:", v);
        }
        st.inflight = cb;
    }
}

/// Encode and commit the arm's buffer over chunks `0..cap` behind a wait
/// for a fresh event value; the inputs are the 5+6 arm's output buffers.
unsafe fn park_locked(gpu: &Gpu, st: &mut ZcT0, pso: Id, in_a: Id, in_b: Id) -> bool {
    unsafe {
        release_parked_locked(gpu, st);
        retire_locked(gpu, st);
        let group = ZC_T0_GROUP;
        let n_enc = st.cap.div_ceil(group);
        let v = st.next_value + 1;
        st.next_value = v + n_enc as u64;
        let pool = gpu.pool_push();
        let cb = (|| -> Option<Id> {
            let cb: Id = send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> Id, st.queue, c"commandBuffer");
            if cb.is_null() {
                return None;
            }
            send!(gpu.api, unsafe extern "C" fn(Id, Sel, Id, u64), cb, c"encodeWaitForEvent:value:", st.ev, v);
            for e in 0..n_enc {
                let lo = e * group;
                let n = group.min(st.cap - lo);
                let enc = gpu.compute_encoder(cb).ok()?;
                gpu.set_pipeline(enc, pso);
                gpu.set_buffer(enc, in_a, 0, 0);
                gpu.set_buffer(enc, in_b, 0, 1);
                gpu.set_buffer(enc, st.out_a, 0, 2);
                gpu.set_buffer(enc, st.out_b, 0, 3);
                gpu.set_buffer(enc, st.consts_buf, 0, 4);
                gpu.set_buffer(enc, st.banks_buf, 0, 5);
                gpu.set_buffer(enc, st.ctl_buf, 0, 6);
                let base = lo as u32;
                gpu.set_bytes(enc, std::slice::from_raw_parts((&raw const base).cast::<u8>(), 4), 7);
                gpu.dispatch(enc, n as u64, 256);
                gpu.end_encoding(enc);
                gpu.encode_signal_event(cb, st.ev, v + 1 + e as u64);
            }
            let cb = gpu.retain(cb);
            gpu.commit_async(cb);
            Some(cb)
        })();
        gpu.pool_pop(pool);
        match cb {
            Some(cb) => {
                st.parked = Some((cb, v));
                true
            }
            None => false,
        }
    }
}

/// Pre-commit (see `ENV_NO_ZC_T0_GPU`): park the next pass 0's buffer now,
/// before the 5+6 pass, so pass 0 only sets an event. Admitted arm and an
/// existing 5+6 buffer set only; a no-op otherwise. Also retires a buffer of
/// the previous prove left in flight, before the 5+6 launch rewrites its
/// inputs.
pub(crate) fn zc_t0_park() {
    use std::sync::atomic::Ordering;
    let Ok(gpu) = gpu() else { return };
    let Some(mut slot) = t0_lock() else { return };
    if let Some(st) = slot.as_mut() {
        unsafe {
            release_parked_locked(gpu, st);
            retire_locked(gpu, st);
        }
    }
    if ZC_T0_MODE.load(Ordering::Relaxed) != ZC_T0_ADMITTED {
        return;
    }
    let (Some(pso), Some((in_a, in_b, _, _, _))) = (zc_t0_pso_ready(gpu), super::t56_metal::zc_t56_out_bufs()) else {
        return;
    };
    if let Some(st) = slot.as_mut() {
        unsafe {
            let _ = park_locked(gpu, st, pso, in_a, in_b);
        }
    }
}

/// A fired pass-0 buffer: chunks `0..offer` in ascending encoder groups.
pub(crate) struct ZcT0Job {
    gpu: &'static Gpu,
    cb: Id,
    ev: Id,
    v: u64,
    offer: usize,
    calibration: bool,
    out_a: *const F128,
    out_b: *const F128,
    banks: *const [u64; 4],
    ctl: *mut u32,
    fired: std::time::Instant,
}

// SAFETY: the pointers name the arm's process-lifetime shared buffers, which
// nothing rewrites before the job is retired; event reads are thread-safe.
unsafe impl Send for ZcT0Job {}
unsafe impl Sync for ZcT0Job {}

impl ZcT0Job {
    pub(crate) fn calibration(&self) -> bool {
        self.calibration
    }

    pub(crate) fn offer(&self) -> usize {
        self.offer
    }

    /// Whether chunk `c`'s GPU outputs and banks are complete: its encoder's
    /// signal (encoded behind that encoder) has been observed.
    pub(crate) fn chunk_ready(&self, c: usize) -> bool {
        if c >= self.offer {
            return false;
        }
        let need = self.v + 1 + (c / ZC_T0_GROUP) as u64;
        let ok = unsafe { self.gpu.event_signaled_value(self.ev) >= need };
        if ok {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        }
        ok
    }

    /// Block until chunk `c` is complete (forced-share test hook only).
    pub(crate) fn wait_chunk(&self, c: usize) -> bool {
        let t = std::time::Instant::now();
        while !self.chunk_ready(c) {
            if unsafe { self.gpu.cb_status(self.cb) } >= 4 && !self.chunk_ready(c) {
                return false;
            }
            if t.elapsed().as_secs() > 5 {
                return false;
            }
            std::hint::spin_loop();
        }
        true
    }

    /// The CPU's steal boundary: the GPU skips chunks `>= hi`.
    pub(crate) fn set_cut(&self, hi: usize) {
        unsafe { self.ctl.add(1).write_volatile(hi as u32) };
    }

    /// Chunk `c`'s GPU outputs and its 26 slots. Caller checked
    /// [`Self::chunk_ready`].
    pub(crate) fn chunk_out(&self, c: usize) -> (&[F128], &[F128], [F128; 26]) {
        debug_assert!(c < self.offer);
        unsafe {
            let banks = std::slice::from_raw_parts(self.banks.add(c * T0_BANKS), T0_BANKS);
            let mut acc = [F256Unreduced::ZERO; 26];
            for (e, w) in banks.iter().enumerate() {
                acc[SLOT[e & 31] as usize] ^= F256Unreduced { r0: w[0], r1: w[1], r2: w[2], r3: w[3] };
            }
            (
                std::slice::from_raw_parts(self.out_a.add(c * T0_OUT), T0_OUT),
                std::slice::from_raw_parts(self.out_b.add(c * T0_OUT), T0_OUT),
                acc.map(F256Unreduced::reduce),
            )
        }
    }

    /// Block until the whole buffer has run (oracle). `false` = it failed;
    /// the arm is then off for the process.
    pub(crate) fn wait_all(&self) -> bool {
        let ok = unsafe { self.gpu.spin_wait_cb(self.cb, 2.0).is_ok() };
        if !ok {
            ZC_T0_MODE.store(ZC_T0_DISABLED, std::sync::atomic::Ordering::Relaxed);
        }
        ok
    }

    /// End of the pass: the remaining threadgroups return at once; the
    /// buffer is retired (waited out) by the next park or launch.
    pub(crate) fn finish(self, taken: usize, hi_size: usize) {
        self.set_cut(0);
        if super::gpu_zc_t0_debug() {
            let complete = (0..self.offer).filter(|&c| self.chunk_ready(c)).count();
            eprintln!(
                "[zc-t0-gpu] {}: {taken} of {hi_size} pass-0 chunks taken from the GPU (offered {}, {complete} complete), {:.3} ms after the signal",
                if self.calibration { "oracle" } else { "fired" },
                self.offer,
                self.fired.elapsed().as_secs_f64() * 1e3,
            );
        }
        if let Some(mut slot) = t0_lock() {
            match slot.as_mut() {
                Some(st) if st.inflight.is_null() => st.inflight = self.cb,
                _ => unsafe {
                    let _ = self.gpu.wait_cb(self.cb);
                    self.gpu.release(self.cb);
                },
            }
        }
    }
}

/// Record the oracle verdict of the first launch.
pub(crate) fn zc_t0_verdict(equal: bool, offer: usize) {
    use std::sync::atomic::Ordering;
    ZC_T0_MODE.store(if equal { ZC_T0_ADMITTED } else { ZC_T0_DISABLED }, Ordering::Relaxed);
    if super::gpu_zc_t0_debug() {
        if equal {
            eprintln!("[zc-t0-gpu] oracle: GPU outputs and slots of pass-0 chunks 0..{offer} EQUAL the CPU's -> admitted");
        } else {
            eprintln!("[zc-t0-gpu] oracle MISMATCH -> arm disabled for the process");
        }
    }
}

/// Start pass 0's GPU part: `pre_a` / `pre_b` are the pass's prefix slices,
/// which must be the 5+6 arm's output buffers' contents (the 5+6 pass of
/// this prove handed them over); the GPU folds the pass-0 chunks wholly
/// inside them. Fires the parked buffer (writing the challenges' tables,
/// the weights and the offer, then setting the event), or, at the arm's
/// first launch, parks and fires at once (the oracle). `None` = the CPU
/// folds every chunk.
pub(crate) fn zc_t0_fire(pre_a: &[F128], pre_b: &[F128], rhos: [F128; 3], eq_lo: &[F128]) -> Option<ZcT0Job> {
    use std::sync::atomic::Ordering;
    let mode = ZC_T0_MODE.load(Ordering::Relaxed);
    if mode == ZC_T0_DISABLED || eq_lo.len() < 4 * T0_GROUPS {
        return None;
    }
    // The oracle may run only at the arm's first launch (the warm-up prove).
    let first = !ZC_T0_TRIED.swap(true, Ordering::AcqRel);
    if mode == ZC_T0_UNCALIBRATED && !first {
        ZC_T0_MODE.store(ZC_T0_DISABLED, Ordering::Relaxed);
        return None;
    }
    let gpu = gpu().ok()?;
    let (in_a, in_b, a_ptr, b_ptr, t56_cap) = super::t56_metal::zc_t56_out_bufs()?;
    if pre_a.as_ptr() != a_ptr || pre_b.as_ptr() != b_ptr || pre_a.len() != pre_b.len() {
        if super::gpu_zc_t0_debug() {
            eprintln!("[zc-t0-gpu] prefix is not the 5+6 arm's buffers; arm skipped");
        }
        return None;
    }
    let mut offer = (pre_a.len() / T0_IN).min(t56_cap).min(T0_HI);
    if let Some(cap) = super::gpu_zc_t0_cap() {
        offer = offer.min(cap);
    }
    if offer == 0 {
        return None;
    }
    let mut slot = t0_lock()?;
    if slot.is_none() {
        // Sized for every chunk the 5+6 buffers can hold.
        match unsafe { zc_t0_alloc(gpu, t56_cap) } {
            Some(s) => *slot = Some(s),
            None => {
                ZC_T0_MODE.store(ZC_T0_DISABLED, Ordering::Relaxed);
                return None;
            }
        }
    }
    let st = slot.as_mut()?;
    if st.cap < offer {
        return None;
    }
    unsafe {
        if st.parked.is_none() {
            // Oracle (first launch), or no parked buffer this prove: encode
            // now. A timed prove without a pipeline never waits for one.
            let pso = match zc_t0_pso_ready(gpu) {
                Some(p) => p,
                None if first => {
                    zc_t0_prewarm();
                    zc_t0_pso_ready(gpu)?
                }
                None => return None,
            };
            if !park_locked(gpu, st, pso, in_a, in_b) {
                return None;
            }
        }
        let (cb, v) = st.parked.take()?;
        let consts = std::slice::from_raw_parts_mut(gpu.buffer_contents(st.consts_buf).cast::<F128>(), T0_CONSTS);
        // Three nibble tables (the layout of `zc_t56::consts_into`), then
        // the weights.
        for (k, rho) in rhos.into_iter().enumerate() {
            nib_into(&mut consts[512 * k..512 * (k + 1)], rho);
        }
        for y in 0..T0_GROUPS {
            consts[1536 + y] = eq_lo[4 * y + 3];
        }
        let ctl = gpu.buffer_contents(st.ctl_buf).cast::<u32>();
        ctl.write_volatile(offer as u32);
        ctl.add(1).write_volatile(offer as u32);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        send!(gpu.api, unsafe extern "C" fn(Id, Sel, u64), st.ev, c"setSignaledValue:", v);
        Some(ZcT0Job {
            gpu,
            cb,
            ev: st.ev,
            v,
            offer,
            calibration: mode == ZC_T0_UNCALIBRATED,
            out_a: gpu.buffer_contents(st.out_a).cast::<F128>().cast_const(),
            out_b: gpu.buffer_contents(st.out_b).cast::<F128>().cast_const(),
            banks: gpu.buffer_contents(st.banks_buf).cast::<[u64; 4]>().cast_const(),
            ctl,
            fired: std::time::Instant::now(),
        })
    }
}

/// One challenge's nibble table, the layout `mul_rho16` reads (bank `k` for
/// byte `k`: `[32 k + n] = rho * (n << 8k)`, `[32 k + 16 + n] = rho *
/// ((n << 4) << 8k)`), built exactly as `zc_t56::consts_into` builds it.
fn nib_into(out: &mut [F128], rho: F128) {
    let mut pw = [F128::ZERO; 128];
    pw[0] = rho;
    for p in 1..128 {
        pw[p] = crate::field::mul_by_x(pw[p - 1]);
    }
    let mut i = 0;
    for bank in 0..16 {
        for half in 0..2 {
            for n in 0..16usize {
                let mut v = F128::ZERO;
                for b in 0..4 {
                    if (n >> b) & 1 == 1 {
                        v += pw[8 * bank + 4 * half + b];
                    }
                }
                out[i] = v;
                i += 1;
            }
        }
    }
}

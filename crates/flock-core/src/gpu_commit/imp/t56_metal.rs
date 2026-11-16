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
//!
//! Flat fold (`zc_t56::ENV_NO_T56_SKIP`, on unless `FLOCK_NO_T56_SKIP=1`):
//! the library is this source's helpers followed by `ZC_T56_FLAT_KERNEL`,
//! the same kernel except that each output folds its four inputs `x0..x3`
//! as `x0 + rho_a (x0 + x1) + rho_b (x0 + x2) + rho_a rho_b (x0 + x1 + x2 +
//! x3)`, three table products that need nothing of each other (a third
//! nibble table, of `rho_a * rho_b`, after `eq_odd` in the constants), and
//! the reduction rows reuse the tables' threadgroup memory after a barrier.

use super::*;
use crate::gpu_commit::zc_t56::{self, BANKS, IN_A, IN_B, OUT, SLOTS};

const ZC_T56_MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 16x16 carry-less product from nine native 32-bit products of the
// bit-position-mod-3 parts: a result bit sums at most six terms, so no carry
// reaches the next bit of the same part.
static inline uint clmul16_m3(uint a, uint b) {
    const uint m0 = 0x9249u, m1 = 0x2492u, m2 = 0x4924u;
    uint a0 = a & m0, a1 = a & m1, a2 = a & m2;
    uint b0 = b & m0, b1 = b & m1, b2 = b & m2;
    uint r0 = (a0 * b0 ^ a1 * b2 ^ a2 * b1) & 0x49249249u;
    uint r1 = (a0 * b1 ^ a1 * b0 ^ a2 * b2) & 0x92492492u;
    uint r2 = (a0 * b2 ^ a1 * b1 ^ a2 * b0) & 0x24924924u;
    return r0 | r1 | r2;
}

// 32x32 carry-less product: one Karatsuba step over the 16-bit halves (no
// 64-bit integer multiplies).
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

// Reduce a 255-bit product mod x^128 + x^7 + x^2 + x + 1.
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

// Multiply a 16-byte value by a challenge through its 16 byte-bank nibble
// tables (bank b: [b*32 + n] = rho*(n at byte b), [b*32 + 16 + n] =
// rho*((n<<4) at byte b); F2-linearity).
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

// v ^ v of lane (lane ^ m) of the simdgroup.
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
    threadgroup uint4 nib[1024];
    threadgroup ulong4 red[256];
    // `tg_base`: first threadgroup of this dispatch in the pass's grid.
    // `cut[0]`: the lowest chunk the CPU has taken (late-GPU steal); a group
    // of a chunk at or above it returns at once. One thread reads the word
    // into threadgroup memory, so every thread of the group takes the same
    // branch whatever the CPU writes meanwhile (threads of one group must
    // not part ways before the barriers and SIMD shuffles below).
    threadgroup uint cut_tg;
    const uint tg = p.tg_base + tg_rel;
    if (lid == 0u) { cut_tg = cut[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tg >> 2) >= cut_tg) { return; }
    for (uint i = lid; i < 1024u; i += 256u) { nib[i] = consts[i]; }
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
    ulong4 acc_a = ulong4(0ul), acc_b = ulong4(0ul), acc_c = ulong4(0ul), acc_d = ulong4(0ul);
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
        const uint4 fa0 = a0 ^ mul_rho16(a0 ^ a1, nib);
        const uint4 fa1 = a2 ^ mul_rho16(a2 ^ a3, nib);
        const uint4 fa = fa0 ^ mul_rho16(fa0 ^ fa1, nib + 512);
        const uint4 fb0 = b0 ^ mul_rho16(b0 ^ b1, nib);
        const uint4 fb1 = b2 ^ mul_rho16(b2 ^ b3, nib);
        const uint4 fb = fb0 ^ mul_rho16(fb0 ^ fb1, nib + 512);
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
        banks[(ulong(tg) << 5) + lid] = s;
    }
}
"#;

/// The flat-fold form of `zc_t56_fold` (see `zc_t56::ENV_NO_T56_SKIP`): the
/// record kernel with each four-input fold as three independent table
/// products (`rho_a`, `rho_b`, `rho_a * rho_b`) and the reduction rows in the
/// tables' memory. Compiled after the record source's helpers.
const ZC_T56_FLAT_KERNEL: &str = r#"kernel void zc_t56_fold(
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
    // The rho_a, rho_b and rho_a * rho_b nibble tables (constants 0..1023
    // and 2048..2559), 1536 entries viewed through `nib`. After the loop and
    // a barrier, the first 256 rows hold the reduction rows (`red`): the
    // same 24 KB as the record form. Declared as `ulong4` so both views keep
    // their natural alignment.
    threadgroup ulong4 tab[768];
    threadgroup uint4* nib = (threadgroup uint4*)tab;
    threadgroup ulong4* red = tab;
    // `tg_base`: first threadgroup of this dispatch in the pass's grid.
    // `cut[0]`: the lowest chunk the CPU has taken (late-GPU steal); a group
    // of a chunk at or above it returns at once. One thread reads the word
    // into threadgroup memory, so every thread of the group takes the same
    // branch whatever the CPU writes meanwhile (threads of one group must
    // not part ways before the barriers and SIMD shuffles below).
    threadgroup uint cut_tg;
    const uint tg = p.tg_base + tg_rel;
    if (lid == 0u) { cut_tg = cut[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tg >> 2) >= cut_tg) { return; }
    for (uint i = lid; i < 1024u; i += 256u) { nib[i] = consts[i]; }
    for (uint i = lid; i < 512u; i += 256u) { nib[1024u + i] = consts[2048u + i]; }
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
    ulong4 acc_a = ulong4(0ul), acc_b = ulong4(0ul), acc_c = ulong4(0ul), acc_d = ulong4(0ul);
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
        // Flat fold: x0 + rho_a (x0 + x1) + rho_b (x0 + x2)
        // + rho_a rho_b (x0 + x1 + x2 + x3), three independent products.
        const uint4 fa = a0 ^ mul_rho16(a0 ^ a1, nib) ^ mul_rho16(a0 ^ a2, nib + 512)
                       ^ mul_rho16(a0 ^ a1 ^ a2 ^ a3, nib + 1024);
        const uint4 fb = b0 ^ mul_rho16(b0 ^ b1, nib) ^ mul_rho16(b0 ^ b2, nib + 512)
                       ^ mul_rho16(b0 ^ b1 ^ b2 ^ b3, nib + 1024);
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
    // Every simdgroup is done with the tables before `red` overwrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
        banks[(ulong(tg) << 5) + lid] = s;
    }
}
"#;

const ZC_T56_ENTRY: &str = "kernel void zc_t56_fold(";

/// The arm's library source: the record text, or (flat fold on) the record
/// text's helpers followed by [`ZC_T56_FLAT_KERNEL`], one entry point either
/// way.
fn zc_t56_source(flat: bool) -> std::borrow::Cow<'static, str> {
    match ZC_T56_MSL_SOURCE.find(ZC_T56_ENTRY) {
        Some(k) if flat => std::borrow::Cow::Owned(format!("{}{}", &ZC_T56_MSL_SOURCE[..k], ZC_T56_FLAT_KERNEL)),
        _ => std::borrow::Cow::Borrowed(ZC_T56_MSL_SOURCE),
    }
}

/// The fold pipeline and the device it was built on, built once per
/// process (prewarm thread, else the first launch); `None` = the source did
/// not compile or the pipeline cannot run the kernel's widths (arm off).
static ZC_T56_PSO: std::sync::OnceLock<Option<(usize, usize)>> = std::sync::OnceLock::new();

unsafe fn zc_t56_build_pso(gpu: &Gpu) -> Result<Id, String> {
    unsafe {
        let pool = gpu.pool_push();
        let built = (|| -> Result<Id, String> {
            let src = gpu.api.nsstring(&zc_t56_source(zc_t56::flat_enabled()))?;
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
    /// Per-launch constants (`zc_t56::launch_consts_into`).
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
    for len in [zc_t56::consts_len() * 16, cap * BANKS * 32, cap * OUT * 16, cap * OUT * 16, 16] {
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
        let consts =
            std::slice::from_raw_parts_mut(gpu.buffer_contents(state.consts_buf).cast::<F128>(), zc_t56::consts_len());
        zc_t56::launch_consts_into(consts, rho_a, rho_b, eq_lo);
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
        let consts =
            std::slice::from_raw_parts_mut(gpu.buffer_contents(state.consts_buf).cast::<F128>(), zc_t56::consts_len());
        zc_t56::launch_consts_into(consts, rho_a, rho_b, eq_lo);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Switch off: the record text exactly. On: the record helpers, then the
    /// flat kernel, with one kernel and one entry point; the flat kernel is
    /// the record kernel with only its tables, fold and reduction rows changed.
    #[test]
    fn t56_skip_source_selection() {
        let off = zc_t56_source(false);
        assert!(matches!(off, std::borrow::Cow::Borrowed(_)));
        assert_eq!(off.as_ref(), ZC_T56_MSL_SOURCE);
        let k = ZC_T56_MSL_SOURCE.find(ZC_T56_ENTRY).unwrap();
        let on = zc_t56_source(true);
        assert_eq!(on.as_ref(), format!("{}{ZC_T56_FLAT_KERNEL}", &ZC_T56_MSL_SOURCE[..k]));
        assert_eq!(on.matches("kernel void ").count(), 1);
        assert_eq!(on.matches(ZC_T56_ENTRY).count(), 1);
        assert!(ZC_T56_FLAT_KERNEL.starts_with(ZC_T56_ENTRY));
        assert_eq!(ZC_T56_FLAT_KERNEL.matches("mul_rho16(").count(), 6);
        assert_eq!(ZC_T56_FLAT_KERNEL.matches("nib + 1024").count(), 2);
        assert_eq!(ZC_T56_FLAT_KERNEL.matches("consts[2048u + i]").count(), 1);
    }
}

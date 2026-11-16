
#include <metal_stdlib>
using namespace metal;

// 16x16 carry-less product from 9 native 32-bit integer products: each
// factor split by bit position mod 3, so a product of two masked parts has at
// most 6 overlapping terms per result bit (16 bits / spacing 3) and its sums
// stay below the next same-residue bit (no carry crosses into a kept bit).
static inline uint zck_clmul16(uint a, uint b) {
    const uint m0 = 0x9249u, m1 = 0x2492u, m2 = 0x4924u;
    uint a0 = a & m0, a1 = a & m1, a2 = a & m2;
    uint b0 = b & m0, b1 = b & m1, b2 = b & m2;
    uint r0 = (a0 * b0 ^ a1 * b2 ^ a2 * b1) & 0x49249249u;
    uint r1 = (a0 * b1 ^ a1 * b0 ^ a2 * b2) & 0x92492492u;
    uint r2 = (a0 * b2 ^ a1 * b1 ^ a2 * b0) & 0x24924924u;
    return r0 | r1 | r2;
}

// 32x32 by one Karatsuba step over 16-bit halves (27 32-bit products instead
// of the embedded kernel's 16 64-bit ones).
static inline ulong zck_clmul32(uint a, uint b) {
    uint al = a & 0xffffu, ah = a >> 16, bl = b & 0xffffu, bh = b >> 16;
    uint lo = zck_clmul16(al, bl);
    uint hi = zck_clmul16(ah, bh);
    uint mid = zck_clmul16(al ^ ah, bl ^ bh) ^ lo ^ hi;
    return ulong(lo) ^ (ulong(mid) << 16) ^ (ulong(hi) << 32);
}

struct ZckU128 { ulong lo; ulong hi; };

static inline ZckU128 zck_clmul64(ulong a, ulong b) {
    uint al = uint(a), ah = uint(a >> 32);
    uint bl = uint(b), bh = uint(b >> 32);
    ulong p_lo = zck_clmul32(al, bl);
    ulong p_hi = zck_clmul32(ah, bh);
    ulong p_mid = zck_clmul32(al ^ ah, bl ^ bh) ^ p_lo ^ p_hi;
    ZckU128 r;
    r.lo = p_lo ^ (p_mid << 32);
    r.hi = p_hi ^ (p_mid >> 32);
    return r;
}

// a * b mod x^128 + x^7 + x^2 + x + 1 (Karatsuba product, then reduction).
static inline uint4 zck_mul(uint4 a, uint4 b) {
    ulong al = (ulong(a.y) << 32) | a.x, ah = (ulong(a.w) << 32) | a.z;
    ulong bl = (ulong(b.y) << 32) | b.x, bh = (ulong(b.w) << 32) | b.z;
    ZckU128 p0 = zck_clmul64(al, bl);
    ZckU128 p2 = zck_clmul64(ah, bh);
    ZckU128 pm = zck_clmul64(al ^ ah, bl ^ bh);
    pm.lo ^= p0.lo ^ p2.lo;
    pm.hi ^= p0.hi ^ p2.hi;
    ulong r0 = p0.lo, r1 = p0.hi ^ pm.lo, h0 = p2.lo ^ pm.hi, h1 = p2.hi;
    ulong t0 = h0 ^ (h0 << 1) ^ (h0 << 2) ^ (h0 << 7);
    ulong t1 = h1 ^ (h1 << 1) ^ (h1 << 2) ^ (h1 << 7)
             ^ (h0 >> 63) ^ (h0 >> 62) ^ (h0 >> 57);
    ulong ov = (h1 >> 63) ^ (h1 >> 62) ^ (h1 >> 57);
    t0 ^= ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    ulong l0 = r0 ^ t0, l1 = r1 ^ t1;
    return uint4(uint(l0), uint(l0 >> 32), uint(l1), uint(l1 >> 32));
}

// T(code) for one 8-byte delta code (lo = bytes 0..3, hi = bytes 4..7).
static inline uint4 zck_fold8(uint lo, uint hi, threadgroup const uint4* nib) {
    uint4 acc = uint4(0u);
    ulong code = (ulong(hi) << 32) | ulong(lo);
    #pragma clang loop unroll(disable)
    for (uint j = 0u; j < 8u; j++) {
        uint v = uint(code >> (8u * j)) & 0xffu;
        acc ^= nib[j * 32u + (v & 15u)] ^ nib[j * 32u + 16u + (v >> 4u)];
    }
    return acc;
}

// x * rho2 for the pass's fixed rho2: rho2 * x is F2-linear in x, so it is
// the XOR over the 32 nibbles of x of the precomputed rho2 * (v << 4k)
// (`rt[16k + v]`, built on the host per prove, 8 KiB in threadgroup memory).
static inline uint4 zck_mulc(uint4 x, threadgroup const uint4* rt) {
    uint4 acc = uint4(0u);
    for (uint w = 0u; w < 4u; w++) {
        uint v = x[w];
        for (uint j = 0u; j < 8u; j++) {
            acc ^= rt[(w * 8u + j) * 16u + ((v >> (4u * j)) & 15u)];
        }
    }
    return acc;
}

struct ZcKParams {
    uint sb_per_chunk;
    uint eq_hi_off;
    // Compact-state layout (tail-compacted, optionally head-compacted; see
    // `UniSkipCompactFold::anchor_index` / `delta_offset`): pair p = 128 B + q
    // of block B. Non-head pairs: [a, b] anchors at a_base + a_stride B +
    // 2q - a_shift, 16 delta bytes at uint4 d_base + d_stride B + q - d_shift.
    // Head pairs (hc, q < 8): a anchor at a_base + a_stride B + q, 8 a delta
    // bytes at byte 16 (d_base + d_stride B) + 8q; b anchor head_b, b delta 0.
    uint hc;
    uint a_base;
    uint a_stride;
    uint a_shift;
    uint d_base;
    uint d_stride;
    uint d_shift;
    uint pad0;
    uint pad1;
    uint pad2;
};

// One threadgroup per 256-output superblock (four dead-tail blocks of 64);
// thread `lid` = output 64k + i of block B = 4 tg + k (pairs 2i, 2i + 1 of
// the block). Outputs i = 61..63 are structural zeros (dead pairs): never
// loaded or stored; output 60 has only its first pair live. The a outputs
// land in the packed layout (244 per superblock: 60 per block, then the
// four outputs 60); the b outputs in the b-one-packed layout (56 per block:
// outputs 4..58, then output 60). The b outputs 0..3 and 59 are not stored:
// each must be one, and any that is not sets `flag`. `consts[0]` = rho2,
// `consts[1]` = head_b, `consts[2 + t]` = w_t = eq_lo[2t + 1],
// `consts[eq_hi_off + c]` = eq_hi[c].
kernel void zc_k_fold_v2(
    device const uint4* anchors [[buffer(0)]],
    device const uint4* deltas  [[buffer(1)]],
    device const uint4* nib_dev [[buffer(2)]],
    device const uint4* consts  [[buffer(3)]],
    device uint4*       out_a   [[buffer(4)]],
    device uint4*       out_b   [[buffer(5)]],
    device uint4*       tg_part [[buffer(6)]],
    constant ZcKParams& p       [[buffer(7)]],
    device atomic_uint* flag    [[buffer(8)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    ushort sl [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint4 nib[256];
    threadgroup uint4 rtab[512];
    threadgroup uint4 sred[8][8];
    nib[lid] = nib_dev[lid];
    rtab[lid] = nib_dev[256u + lid];
    rtab[256u + lid] = nib_dev[512u + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = lid & 63u;
    uint k = lid >> 6u;
    ulong blk = ulong(tg) * 4ul + ulong(k);
    uint4 a = uint4(0u);
    uint4 b = uint4(0u);
    if (i <= 60u) {
        uint4 xa0, xb0;
        uint4 xa1 = uint4(0u);
        uint4 xb1 = uint4(0u);
        ulong ab = ulong(p.a_base) + blk * ulong(p.a_stride);
        ulong db = ulong(p.d_base) + blk * ulong(p.d_stride);
        // One fold sequence for head and non-head lanes: a simdgroup that
        // holds both kinds runs four folds, not six.
        bool head = p.hc != 0u && i < 4u;
        bool has1 = !head && i < 60u;
        ulong ia = head ? ab + 2ul * ulong(i) : ab + 4ul * ulong(i) - ulong(p.a_shift);
        ulong id = head ? db + ulong(i) : db + 2ul * ulong(i) - ulong(p.d_shift);
        uint4 d0 = deltas[id];
        uint4 d1 = uint4(0u);
        if (has1) { d1 = deltas[id + 1ul]; }
        // Values in order a0, b0, a1, b1 through one fold body.
        uint2 c0 = d0.xy, c1 = d0.zw, c2 = head ? d0.zw : d1.xy, c3 = d1.zw;
        uint offs = head ? 0x3110u : 0x3210u;
        uint act = head ? 0x5u : (has1 ? 0xfu : 0x3u);
        xa0 = uint4(0u); xb0 = uint4(0u);
        #pragma clang loop unroll(disable)
        for (uint s = 0u; s < 4u; s++) {
            uint4 v = uint4(0u);
            if (((act >> s) & 1u) != 0u) {
                v = anchors[ia + ulong((offs >> (4u * s)) & 15u)] ^ zck_fold8(c0.x, c0.y, nib);
            }
            c0 = c1; c1 = c2; c2 = c3;
            xa0 = xb0; xb0 = xa1; xa1 = xb1; xb1 = v;
        }
        if (head) {
            xb0 = consts[1];
            xb1 = consts[1];
        }
        a = xa0 ^ zck_mulc(xa0 ^ xa1, rtab);
        b = xb0 ^ zck_mulc(xb0 ^ xb1, rtab);
        out_a[ulong(tg) * 244ul + ulong(i < 60u ? 60u * k + i : 240u + k)] = a;
        if (i < 4u || i == 59u) {
            if (any(b != uint4(1u, 0u, 0u, 0u))) {
                atomic_fetch_or_explicit(flag, 1u, memory_order_relaxed);
            }
        } else {
            out_b[blk * 56ul + ulong(i == 60u ? 55u : i - 4u)] = b;
        }
    }
    uint t = ((tg % p.sb_per_chunk) * 256u + lid) >> 2u;
    uint4 aw = zck_mul(consts[2u + t], a);
    ushort qb = sl & ushort(0xFFFC);
    uint4 aw0 = simd_shuffle(aw, qb);
    uint4 aw1 = simd_shuffle(aw, ushort(qb + 1));
    uint4 aw2 = simd_shuffle(aw, ushort(qb + 2));
    uint4 aw3 = simd_shuffle(aw, ushort(qb + 3));
    uint4 b0 = simd_shuffle(b, qb);
    uint4 b1 = simd_shuffle(b, ushort(qb + 1));
    uint4 b2 = simd_shuffle(b, ushort(qb + 2));
    uint4 b3 = simd_shuffle(b, ushort(qb + 3));
    // Quad lane q owns aggregate slots 2q and 2q + 1.
    uint q = lid & 3u;
    uint4 x0, y0, x1, y1;
    if (q == 0u) {
        x0 = aw1; y0 = b1; x1 = aw0 ^ aw1; y1 = b0 ^ b1;
    } else if (q == 1u) {
        x0 = aw3; y0 = b3; x1 = aw2 ^ aw3; y1 = b2 ^ b3;
    } else if (q == 2u) {
        x0 = aw2; y0 = b2; x1 = aw0 ^ aw2; y1 = b0 ^ b2;
    } else {
        x0 = aw1 ^ aw3; y0 = b1 ^ b3;
        x1 = aw0 ^ aw1 ^ aw2 ^ aw3; y1 = b0 ^ b1 ^ b2 ^ b3;
    }
    uint4 r0 = zck_mul(x0, y0);
    uint4 r1 = zck_mul(x1, y1);
    for (ushort off = 4; off < 32; off <<= 1) {
        r0 ^= simd_shuffle_xor(r0, off);
        r1 ^= simd_shuffle_xor(r1, off);
    }
    if (sl < 4) {
        sred[sg][2u * uint(sl)] = r0;
        sred[sg][2u * uint(sl) + 1u] = r1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 8u) {
        uint4 s = uint4(0u);
        for (uint j = 0u; j < 8u; j++) { s ^= sred[j][lid]; }
        tg_part[ulong(tg) * 8ul + ulong(lid)] = s;
    }
}


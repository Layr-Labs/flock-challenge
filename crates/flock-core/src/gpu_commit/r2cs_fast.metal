// R2-CS kernel, lean form (`FLOCK_NO_R2CS_FAST=1` restores the original
// kernel text). Same entry point, buffers, launch shape and stores; every
// output is the same exact GF(2^128) value or byte. Four changes, each
// order-free because every partial is an XOR sum over the threadgroup (a
// fifth, the skip boundary of the late-GPU steal, is described at its read):
//  - Parity-sorted map: SIMD group s of a threadgroup (lanes 32s..32s+31)
//    takes sixteen windows of one parity (s & 1) and one class pair
//    {2t, 2t + 1}, t = (s >> 1) & 1. Pair partners stay lanes lid and
//    lid ^ 1 and the lane parity stays the class parity, so the parity
//    split and the shuffles are unchanged. Any SIMD width gives the same
//    bijection; only how often the skip below fires depends on the width.
//  - A step whose pairs are all padded in the whole SIMD group (medium 15
//    of the odd windows, medium 14 of their classes 2 and 3) does no
//    loads, folds, stores or products: its products are zero, so only the
//    four Horner shifts run.
//  - Each pass of the shared product body forms its operands from the
//    lane's own four folds (and one partner shuffle per side for the
//    lookahead terms) instead of rotating four precomputed operand pairs.
//  - The 32x32 products split their 16-bit halves into residue planes once
//    and form the Karatsuba middle planes by XOR (masking is F2-linear);
//    the weighing pass reduces the four accumulators first and stores
//    each SIMD sum as soon as it is formed.

// Nine products of the bit-position-mod-3 planes of two 16-bit values
// (the same products and masks as zc_r2_cs_clmul16).
static inline uint zc_r2_cs_leaf3(uint a0, uint a1, uint a2, uint b0, uint b1, uint b2) {
    uint r0 = (a0 * b0 ^ a1 * b2 ^ a2 * b1) & 0x49249249u;
    uint r1 = (a0 * b1 ^ a1 * b0 ^ a2 * b2) & 0x92492492u;
    uint r2 = (a0 * b2 ^ a1 * b1 ^ a2 * b0) & 0x24924924u;
    return r0 | r1 | r2;
}

// zc_r2_cs_clmul32 with the middle operand's planes formed from the
// halves' planes.
static inline ulong zc_r2_cs_clmul32p(uint a, uint b) {
    const uint m0 = 0x9249u, m1 = 0x2492u, m2 = 0x4924u;
    uint ah = a >> 16, bh = b >> 16;
    uint al0 = a & m0, al1 = a & m1, al2 = a & m2;
    uint ah0 = ah & m0, ah1 = ah & m1, ah2 = ah & m2;
    uint bl0 = b & m0, bl1 = b & m1, bl2 = b & m2;
    uint bh0 = bh & m0, bh1 = bh & m1, bh2 = bh & m2;
    uint lo = zc_r2_cs_leaf3(al0, al1, al2, bl0, bl1, bl2);
    uint hi = zc_r2_cs_leaf3(ah0, ah1, ah2, bh0, bh1, bh2);
    uint mid = zc_r2_cs_leaf3(al0 ^ ah0, al1 ^ ah1, al2 ^ ah2, bl0 ^ bh0, bl1 ^ bh1, bl2 ^ bh2) ^ lo ^ hi;
    return ulong(lo) ^ (ulong(mid) << 16) ^ (ulong(hi) << 32);
}

static inline U128k zc_r2_cs_clmul64p(ulong a, ulong b) {
    uint al = uint(a), ah = uint(a >> 32);
    uint bl = uint(b), bh = uint(b >> 32);
    ulong p_lo = zc_r2_cs_clmul32p(al, bl);
    ulong p_hi = zc_r2_cs_clmul32p(ah, bh);
    ulong p_mid = zc_r2_cs_clmul32p(al ^ ah, bl ^ bh) ^ p_lo ^ p_hi;
    U128k r;
    r.lo = p_lo ^ (p_mid << 32);
    r.hi = p_hi ^ (p_mid >> 32);
    return r;
}

static inline U256k zc_r2_cs_clmul128p(uint4 a, uint4 b) {
    ulong al = (ulong(a.y) << 32) | a.x, ah = (ulong(a.w) << 32) | a.z;
    ulong bl = (ulong(b.y) << 32) | b.x, bh = (ulong(b.w) << 32) | b.z;
    U128k p0 = zc_r2_cs_clmul64p(al, bl);
    U128k p2 = zc_r2_cs_clmul64p(ah, bh);
    U128k pm = zc_r2_cs_clmul64p(al ^ ah, bl ^ bh);
    pm.lo ^= p0.lo ^ p2.lo;
    pm.hi ^= p0.hi ^ p2.hi;
    U256k r;
    r.r0 = p0.lo;
    r.r1 = p0.hi ^ pm.lo;
    r.r2 = p2.lo ^ pm.hi;
    r.r3 = p2.hi;
    return r;
}

kernel void zc_r2_products_horner_cs(
    device const uint4* a_in  [[buffer(0)]],
    device const uint4* b_in  [[buffer(1)]],
    device const uint4* eq_lo [[buffer(2)]],
    device const uint4* eq_hi [[buffer(3)]],
    device const uint4* nib_tab_dev [[buffer(4)]],
    device uint4*       partials    [[buffer(5)]],
    constant ZcR2Params& p          [[buffer(6)]],
    device uint4*       cs_anchors  [[buffer(7)]],
    device uint4*       cs_deltas   [[buffer(8)]],
    constant ZcR2CsLayout& cl       [[buffer(9)]],
    device atomic_uint* cs_flag     [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]],
    ushort simd_width [[threads_per_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]],
    ushort simd_lane  [[thread_index_in_simdgroup]])
{
    threadgroup ulong4 red[256];
    threadgroup uint4* nib = reinterpret_cast<threadgroup uint4*>(red);
    // Skip boundary (late-GPU steal): word 1 of the flag buffer holds the
    // ones' complement of the lowest chunk the CPU has taken (0 = none); a
    // group of a chunk at or above it returns at once. One thread reads the
    // word into threadgroup memory, so every thread of the group takes the
    // same branch whatever the CPU writes meanwhile (the threads of a group
    // must not part ways before the barriers and SIMD shuffles below).
    threadgroup uint cut_tg;
    if (lid == 0u) { cut_tg = ~atomic_load_explicit(cs_flag + 1, memory_order_relaxed); }
    nib[lid] = nib_tab_dev[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint hi_chunk = tgid >> 2;
    if (hi_chunk >= cut_tg) { return; }
    // lid bits -> window bit 0 = bit 5, window bits 1..4 = bits 1..4,
    // window bit 5 = bit 7, class bit 0 = bit 0, class bit 1 = bit 6.
    uint block = lid >> 5;
    uint window = ((tgid & 3u) << 6) | ((block >> 2) << 5) | (lid & 30u) | (block & 1u);
    uint small_class = (((block >> 1) & 1u) << 1) | (lid & 1u);
    bool odd_lane = (small_class & 1u) != 0u;
    U256k h1 = {0ul, 0ul, 0ul, 0ul};
    U256k hi = {0ul, 0ul, 0ul, 0ul};
    U256k wa = {0ul, 0ul, 0ul, 0ul};
    U256k wb = {0ul, 0ul, 0ul, 0ul};
    bool head_bad = false;
    for (uint step = 0u; step < (p.xpt & 0x7fffffffu); step++) {
        uint medium = 15u - step;
        uint x_lo = (window << 6) | (medium << 2) | small_class;
        uint pair_idx = hi_chunk * p.lo_size + x_lo;
        // A padded pair folds to zero, so every product it enters is zero;
        // a group whose even pair is padded has only zero operands (the
        // walk's boundary group pairs pair 120 with 121).
        bool pad = (pair_idx & p.mask) >= p.useful;
        if (simd_all(pad)) {
            zc_r2_cs_horner_shift(h1);
            zc_r2_cs_horner_shift(hi);
            zc_r2_cs_horner_shift(wa);
            zc_r2_cs_horner_shift(wb);
            continue;
        }
        uint4 a0 = uint4(0u);
        uint4 a1 = uint4(0u);
        uint4 b0 = uint4(0u);
        uint4 b1 = uint4(0u);
        if (!pad) {
            uint4 ar = a_in[pair_idx];
            uint4 br = b_in[pair_idx];
            a0 = zc_r2_fold<ZC_R2_PL>(ar.x, ar.y, nib);
            a1 = zc_r2_fold<ZC_R2_PL>(ar.z, ar.w, nib);
            b0 = zc_r2_fold<ZC_R2_PL>(br.x, br.y, nib);
            b1 = zc_r2_fold<ZC_R2_PL>(br.z, br.w, nib);
            zc_r2_cs_store(pair_idx, ar, br, a0, b0, cs_anchors, cs_deltas, cl);
            // Head-compacted state: a head pair's b rows must be all ones.
            if (cl.hc != 0u && (pair_idx & 127u) < 8u) {
                head_bad = head_bad || ((br.x & br.z) & (br.y & br.w)) != 0xffffffffu;
            }
        }
        // Four passes through one product body; pass k takes the
        // accumulator in h1 one Horner step, then the accumulators rotate.
        //   k = 0: a1 b1 -> h1;  k = 1: s_a s_b -> hi, s = a0 + a1;
        //   k = 2: odd lane a0 b0 (W0), even lane e_a e_b (W3) -> wa,
        //          e = a0 + partner a0;
        //   k = 3: odd lane t_a t_b (W5), t = s + partner s, even lane
        //          o_a o_b (W4), o = a1 + partner a1 -> wb.
        #pragma clang loop unroll(disable)
        for (uint k = 0u; k < 4u; k++) {
            uint4 x = a1;
            uint4 y = b1;
            if (k == 1u) {
                x = a0 ^ a1;
                y = b0 ^ b1;
            } else if (k >= 2u) {
                // k = 2: every lane sends a0; k = 3: the odd lane sends a1,
                // the even lane s.
                bool send_a1 = k == 3u && odd_lane;
                uint4 su = send_a1 ? a1 : (k == 3u ? a0 ^ a1 : a0);
                uint4 sv = send_a1 ? b1 : (k == 3u ? b0 ^ b1 : b0);
                uint4 ru = simd_shuffle_xor(su, ushort(1));
                uint4 rv = simd_shuffle_xor(sv, ushort(1));
                if (k == 2u) {
                    x = odd_lane ? a0 : a0 ^ ru;
                    y = odd_lane ? b0 : b0 ^ rv;
                } else {
                    x = odd_lane ? (a0 ^ a1) ^ ru : a1 ^ ru;
                    y = odd_lane ? (b0 ^ b1) ^ rv : b1 ^ rv;
                }
            }
            zc_r2_cs_horner_shift(h1);
            zc_r2_cs_horner_add(h1, zc_r2_cs_clmul128p(x, y));
            U256k th = h1; h1 = hi; hi = wa; wa = wb; wb = th;
        }
    }
    if (head_bad) {
        atomic_store_explicit(cs_flag, 1u, memory_order_relaxed);
    }
    uint4 e = eq_lo[window * 4u + small_class];
    uint4 lam = eq_lo[window * 4u + (small_class | 1u)];
    // Retire every table reader before the sums reuse the storage.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // [p1_even, pinf_even, p1_odd, pinf_odd, W0, W3, W4, W5] per physical
    // group, as the original kernel: each accumulator reduced, weighed by
    // its eq_lo weight (e for h1 and hi, lam for wa and wb), summed per lane
    // parity inside the SIMD group and stored at
    // (group * 4 + accumulator) * 2 + parity; lanes 0..7 then sum and weigh
    // one output slot each.
    {
        uint4 r0 = gf_reduce(h1);
        uint4 r1 = gf_reduce(hi);
        uint4 r2 = gf_reduce(wa);
        uint4 r3 = gf_reduce(wb);
        #pragma clang loop unroll(disable)
        for (uint k = 0u; k < 4u; k++) {
            U256k t = zc_r2_cs_simd_parity_xor(zc_r2_cs_clmul128(k < 2u ? e : lam, r0), simd_width);
            uint4 tr = r0; r0 = r1; r1 = r2; r2 = r3; r3 = tr;
            if (simd_lane < 2u) {
                red[(uint(simd_group) * 4u + k) * 2u + uint(simd_lane)] = ulong4(t.r0, t.r1, t.r2, t.r3);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 8u) {
        // Output slot -> (accumulator, parity): p1_even (h1, 0),
        // pinf_even (hi, 0), p1_odd (h1, 1), pinf_odd (hi, 1), W0 (wa, 1),
        // W3 (wa, 0), W4 (wb, 0), W5 (wb, 1).
        uint acc = (0x33221010u >> (4u * lid)) & 0xfu;
        uint par = (0x10011100u >> (4u * lid)) & 0xfu;
        ulong4 s = ulong4(0ul);
        for (uint g = 0u; g < 256u / uint(simd_width); g++) {
            s ^= red[(g * 4u + acc) * 2u + par];
        }
        partials[tgid * 8u + lid] = zc_r2_cs_weigh(eq_hi[hi_chunk], s);
    }
}

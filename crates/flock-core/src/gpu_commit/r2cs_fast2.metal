// R2-CS kernel, lean form plus the b == 1 step (`FLOCK_NO_R2CS_FAST2=1`
// compiles the lean kernel of r2cs_fast.metal instead; `FLOCK_NO_R2CS_FAST=1`
// the original kernel text). Built after the lean form's product helpers, with
// the same entry point, buffers, launch shape and stores; every output is the
// same exact GF(2^128) value or byte.
//
// The b == 1 step. The all-ones row folds to ONE whenever the nibble table is
// a partition of unity (checked once per threadgroup: `c_one`). A step in
// which every pair of the SIMD group is padded or has all-ones b rows
// (`c_one` required for the latter) then has b0 = b1 = 1 for each live pair
// and b0 = b1 = 0 for each padded one, for every lane and its partner, so its
// four products are known without a multiplication:
//   p1 = a1 * b1 = a1;  pinf = s_a * (b0 + b1) = 0;
//   odd lane:  W0 = a0 * b0 = a0;  W5 = t_a * (s_b + s_b') = 0;
//   even lane: W3 = (a0 + a0') * (b0 + b0'), W4 = (a1 + a1') * (b1 + b1'),
//              whose b factor is 1 when exactly one pair of the group is live
//              and 0 otherwise.
// A padded pair folds to zero, so these hold for padded lanes too. The step
// still folds a0 and a1 of each live pair and makes its compact stores (b
// anchor ONE, the fold of the all-ones row); it skips the two b folds and the
// four products. The general step is unchanged, and also covers a SIMD group
// whose pairs are all padded (the lean form's skip) as a b == 1 step with no
// live pair.

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
    nib[lid] = nib_tab_dev[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Whether the all-ones row folds to exactly ONE (the table is a
    // partition of unity): the premise of the b == 1 step for live pairs.
    bool c_one = all(zc_r2_fold<ZC_R2_PL>(0xffffffffu, 0xffffffffu, nib) == uint4(1u, 0u, 0u, 0u));

    uint hi_chunk = tgid >> 2;
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
        // A padded pair folds to zero (no loads, folds or stores).
        bool pad = (pair_idx & p.mask) >= p.useful;
        uint4 a0 = uint4(0u);
        uint4 a1 = uint4(0u);
        uint4 b0 = uint4(0u);
        uint4 b1 = uint4(0u);
        uint4 ar = uint4(0u);
        uint4 br = uint4(0u);
        if (!pad) {
            ar = a_in[pair_idx];
            br = b_in[pair_idx];
        }
        // b == 1 step (see the header): every pair of the SIMD group is padded
        // or has all-ones b rows folding to ONE. p1 = a1, pinf = 0, W0 = a0,
        // W5 = 0, and W3, W4 are the group sums of a0 and a1 when exactly one
        // pair of the group is live, else 0: no products. A group with only
        // padded pairs adds nothing (only the four Horner shifts run).
        if (simd_all(pad || (c_one && ((br.x & br.z) & (br.y & br.w)) == 0xffffffffu))) {
            if (!pad) {
                a0 = zc_r2_fold<ZC_R2_PL>(ar.x, ar.y, nib);
                a1 = zc_r2_fold<ZC_R2_PL>(ar.z, ar.w, nib);
                zc_r2_cs_store(pair_idx, ar, br, a0, uint4(1u, 0u, 0u, 0u), cs_anchors, cs_deltas, cl);
            }
            // Group partners are lanes lid and lid ^ 1, pairs pair_idx and
            // pair_idx ^ 1; lone = exactly one of the two is live.
            uint4 pa0 = simd_shuffle_xor(a0, ushort(1));
            uint4 pa1 = simd_shuffle_xor(a1, ushort(1));
            bool lone = pad != (((pair_idx ^ 1u) & p.mask) >= p.useful);
            uint4 w0 = odd_lane ? a0 : (lone ? a0 ^ pa0 : uint4(0u));
            uint4 w1 = (!odd_lane && lone) ? a1 ^ pa1 : uint4(0u);
            zc_r2_cs_horner_shift(h1);
            h1.r0 ^= (ulong(a1.y) << 32) | a1.x;
            h1.r1 ^= (ulong(a1.w) << 32) | a1.z;
            zc_r2_cs_horner_shift(hi);
            zc_r2_cs_horner_shift(wa);
            wa.r0 ^= (ulong(w0.y) << 32) | w0.x;
            wa.r1 ^= (ulong(w0.w) << 32) | w0.z;
            zc_r2_cs_horner_shift(wb);
            wb.r0 ^= (ulong(w1.y) << 32) | w1.x;
            wb.r1 ^= (ulong(w1.w) << 32) | w1.z;
            continue;
        }
        if (!pad) {
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

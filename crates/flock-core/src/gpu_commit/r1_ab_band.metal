// Derived from the Apache-2.0 Flock kernel in PR #1857, commit 6269ea1a.
// The ranked CPU producer publishes each input row before this dispatch.

#include <metal_stdlib>
using namespace metal;

// Four binary coefficients in even bit positions act as base-4 digits.
// The only digit overflow is 0x55 * 0x55. Its carry changes the parity
// at degrees 8 and 10, so restore those bits before extracting parity.
static inline uint clpoly_even4(uint a, uint b) {
    uint fix = select(0u, 0x500u, (a & b) == 0x55u);
    return ((a * b) ^ fix) & 0x1555u;
}

// Write each polynomial as even(x) + x*odd(x) and use Karatsuba.
static inline uint clpoly8(uint a, uint b) {
    uint a_even = a & 0x55u, a_odd = (a >> 1) & 0x55u;
    uint b_even = b & 0x55u, b_odd = (b >> 1) & 0x55u;
    uint lo = clpoly_even4(a_even, b_even);
    uint hi = clpoly_even4(a_odd, b_odd);
    uint mid = clpoly_even4(a_even ^ a_odd, b_even ^ b_odd) ^ lo ^ hi;
    return lo ^ (mid << 1) ^ (hi << 2);
}

// Wide reduce for the deferred accumulator (<=21-bit input): each fold
// shaves ~4 bits above 8 (the h<<4 term dominates), so four folds fully
// reduce 21 bits. F2-linearity makes the single deferred reduce equal to
// the scalar's per-k reduce chain.
static inline uint gf8_reduce_wide(uint p) {
    for (uint r = 0; r < 4u; ++r) {
        uint h = p >> 8;
        p = (p & 0xFFu) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4);
    }
    return p;
}

// One threadgroup (256 threads) per x_outer chunk: 16 b_med rows x 16
// four-lane thread slots. The two windows use the ranked materialized rows.
kernel void ab_pre_rows(
    device const uchar* a        [[buffer(0)]],
    device const uchar* b        [[buffer(1)]],
    constant uint*      table    [[buffer(2)]],   // 256 x 64 bytes, plain
    device uchar*       out      [[buffer(3)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint x = tg;
    uint row = tid >> 4;
    uint quad = tid & 15u;
    // The ranked transform omits even rows 0–1 and odd rows 14–15.
    if ((x & 1u) == 0u ? row < 2u : row >= 14u) { return; }
    ulong base = ulong(x) * 1024ul + ulong(row) * 64ul;

    // Reduce once after accumulating the shifted carry-less products.
    uint acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0; // per-lane <=21-bit accs
    for (uint k = 0; k < 8u; ++k) {
        device const uchar* ap = a + base + ulong(k) * 8ul;
        device const uchar* bp = b + base + ulong(k) * 8ul;
        uint acol = 0, bcol = 0; // 4 lanes packed LE
        for (uint j = 0; j < 8u; ++j) {
            uint idx4 = ((quad * 4u) ^ (8u * j)) >> 2;
            acol ^= table[uint(ap[j]) * 16u + idx4];
            bcol ^= table[uint(bp[j]) * 16u + idx4];
        }
        acc0 ^= clpoly8(acol & 0xFFu,         bcol & 0xFFu)         << k;
        acc1 ^= clpoly8((acol >> 8) & 0xFFu,  (bcol >> 8) & 0xFFu)  << k;
        acc2 ^= clpoly8((acol >> 16) & 0xFFu, (bcol >> 16) & 0xFFu) << k;
        acc3 ^= clpoly8(acol >> 24,           bcol >> 24)           << k;
    }
    uint outw = gf8_reduce_wide(acc0)
              | (gf8_reduce_wide(acc1) << 8)
              | (gf8_reduce_wide(acc2) << 16)
              | (gf8_reduce_wide(acc3) << 24);
    ((device uint*)(out + base))[quad] = outw;
}

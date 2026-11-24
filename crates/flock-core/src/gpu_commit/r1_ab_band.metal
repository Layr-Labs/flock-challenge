// Derived from the Apache-2.0 Flock kernel in PR #1857, commit 6269ea1a.
// Standalone experiment for the four-lane arithmetic in the 26916aeb band.

#include <metal_stdlib>
using namespace metal;

static inline uint times_x(uint word) {
    uint overflow = (word >> 7) & 0x01010101u;
    return ((word << 1) & 0xfefefefeu) ^ (overflow * 0x1bu);
}

// For shifts up to four, the overflow times 0x1b stays within each byte.
template <uint Shift>
static inline uint times_x_pow(uint word) {
    const uint low_bits = (1u << Shift) - 1u;
    uint overflow = (word >> (8u - Shift)) & (0x01010101u * low_bits);
    uint lo = (word << Shift) & (0x01010101u * (0xffu ^ low_bits));
    return lo ^ overflow ^ (overflow << 1) ^ (overflow << 3) ^ (overflow << 4);
}

// Each base-256 digit is 0 or 1, so the 255 multiple has no carry.
static inline uint byte_mask(uint word) {
    uint ones = word & 0x01010101u;
    return (ones << 8) - ones;
}

static inline uint two_bits(uint a, uint xa, uint bits) {
    return (a & byte_mask(bits)) ^ (xa & byte_mask(bits >> 1));
}

static inline uint mul4(uint a, uint b) {
    uint xa = times_x(a);
    uint lo = two_bits(a, xa, b) ^ times_x_pow<2>(two_bits(a, xa, b >> 2));
    uint hi = two_bits(a, xa, b >> 4) ^ times_x_pow<2>(two_bits(a, xa, b >> 6));
    return lo ^ times_x_pow<4>(hi);
}

// One threadgroup (256 threads) per x_outer chunk: 16 b_med rows x 16
// four-lane thread slots. The two windows use the ranked materialized rows.
kernel void ab_pre_rows(
    device const uchar* a        [[buffer(0)]],
    device const uchar* b        [[buffer(1)]],
    constant uint*      table    [[buffer(2)]],   // 256 x 64 bytes, plain
    device uchar*       out      [[buffer(3)]],
    constant uint&      omit_mixed_ones [[buffer(4)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint x = tg;
    uint row = tid >> 4;
    uint quad = tid & 15u;
    bool odd = (x & 1u) != 0u;
    // The ranked transform omits even rows 0–1 and odd rows 14–15.
    if (odd ? row >= 14u : row < 2u) { return; }
    ulong base = ulong(x) * 1024ul + ulong(row) * 64ul;

    // The C fold restores the omitted B=1 terms of the two mixed rows.
    uint first_k = omit_mixed_ones != 0u && !odd && row == 2u ? 2u : 0u;
    uint end_k = omit_mixed_ones != 0u && odd && row == 13u ? 4u : 8u;
    // Horner evaluates the x^k weights in GF(2^8), four lanes at once.
    uint outw = 0;
    for (uint remaining = end_k; remaining > first_k; --remaining) {
        uint k = remaining - 1u;
        device const uchar* ap = a + base + ulong(k) * 8ul;
        device const uchar* bp = b + base + ulong(k) * 8ul;
        uint acol = 0, bcol = 0; // 4 lanes packed LE
        for (uint j = 0; j < 8u; ++j) {
            uint idx4 = ((quad * 4u) ^ (8u * j)) >> 2;
            acol ^= table[uint(ap[j]) * 16u + idx4];
            bcol ^= table[uint(bp[j]) * 16u + idx4];
        }
        outw = times_x(outw) ^ mul4(acol, bcol);
    }
    // The even mixed row still has two trailing zero Horner steps.
    if (first_k != 0u) { outw = times_x_pow<2>(outw); }
    ((device uint*)(out + base))[quad] = outw;
}

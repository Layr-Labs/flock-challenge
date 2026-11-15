// Each aligned plane contains eight radix-16 coefficients. Their integer
// product has coefficients at most eight, so no carry crosses a digit.
static inline ulong clmul_plane_pair(uint a0, uint a1, uint b0, uint b1) {
    const ulong MASK = 0x1111111111111111UL;
    ulong p0 = (ulong(a0) * ulong(b0)) & MASK;
    ulong p2 = (ulong(a1) * ulong(b1)) & MASK;
    ulong pm = (ulong(a0 ^ a1) * ulong(b0 ^ b1)) & MASK;
    return p0 ^ ((pm ^ p0 ^ p2) << 1) ^ (p2 << 2);
}

// Two Karatsuba levels use nine integer products. The result ends at bit 62.
static inline ulong clmul32(uint a, uint b) {
    const uint MASK = 0x11111111u;
    uint a0 = a & MASK, a1 = (a >> 1) & MASK;
    uint a2 = (a >> 2) & MASK, a3 = (a >> 3) & MASK;
    uint b0 = b & MASK, b1 = (b >> 1) & MASK;
    uint b2 = (b >> 2) & MASK, b3 = (b >> 3) & MASK;
    ulong lo = clmul_plane_pair(a0, a1, b0, b1);
    ulong hi = clmul_plane_pair(a2, a3, b2, b3);
    ulong mid = clmul_plane_pair(a0 ^ a2, a1 ^ a3, b0 ^ b2, b1 ^ b3);
    return lo ^ ((mid ^ lo ^ hi) << 2) ^ (hi << 4);
}

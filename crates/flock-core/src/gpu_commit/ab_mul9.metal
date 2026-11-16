static inline uint ab_clmul16_leaf(uint a0, uint a1, uint a2, uint b0, uint b1, uint b2) {
    uint r0 = (a0 * b0 ^ a1 * b2 ^ a2 * b1) & 0x49249249u;
    uint r1 = (a0 * b1 ^ a1 * b0 ^ a2 * b2) & 0x92492492u;
    uint r2 = (a0 * b2 ^ a1 * b1 ^ a2 * b0) & 0x24924924u;
    return r0 | r1 | r2;
}

static inline ulong clmul32(uint a, uint b) {
    const uint m0 = 0x9249u, m1 = 0x2492u, m2 = 0x4924u;
    uint ah = a >> 16, bh = b >> 16;
    uint al0 = a & m0, al1 = a & m1, al2 = a & m2;
    uint ah0 = ah & m0, ah1 = ah & m1, ah2 = ah & m2;
    uint bl0 = b & m0, bl1 = b & m1, bl2 = b & m2;
    uint bh0 = bh & m0, bh1 = bh & m1, bh2 = bh & m2;
    uint lo = ab_clmul16_leaf(al0, al1, al2, bl0, bl1, bl2);
    uint hi = ab_clmul16_leaf(ah0, ah1, ah2, bh0, bh1, bh2);
    uint mid = ab_clmul16_leaf(al0 ^ ah0, al1 ^ ah1, al2 ^ ah2, bl0 ^ bh0, bl1 ^ bh1, bl2 ^ bh2) ^ lo ^ hi;
    return ulong(lo) ^ (ulong(mid) << 16) ^ (ulong(hi) << 32);
}

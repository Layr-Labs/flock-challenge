// 16x16 carry-less product from 9 native 16x16->32 ushort multiplies.
static inline uint ab_clmul16_u16(ushort a, ushort b) {
    ushort a0 = a & 0x9249u, a1 = a & 0x2492u, a2 = a & 0x4924u;
    ushort b0 = b & 0x9249u, b1 = b & 0x2492u, b2 = b & 0x4924u;
    uint r0 = (uint(a0) * uint(b0) ^ uint(a1) * uint(b2) ^ uint(a2) * uint(b1)) & 0x49249249u;
    uint r1 = (uint(a0) * uint(b1) ^ uint(a1) * uint(b0) ^ uint(a2) * uint(b2)) & 0x92492492u;
    uint r2 = (uint(a0) * uint(b2) ^ uint(a1) * uint(b1) ^ uint(a2) * uint(b0)) & 0x24924924u;
    return r0 | r1 | r2;
}

// 32x32 carry-less product via 1-level Karatsuba over 16x16->32 ushort leaves.
static inline ulong clmul32(uint a, uint b) {
    ushort al = ushort(a), ah = ushort(a >> 16);
    ushort bl = ushort(b), bh = ushort(b >> 16);
    uint lo = ab_clmul16_u16(al, bl);
    uint hi = ab_clmul16_u16(ah, bh);
    uint mid = ab_clmul16_u16(al ^ ah, bl ^ bh) ^ lo ^ hi;
    return ulong(lo) ^ (ulong(mid) << 16) ^ (ulong(hi) << 32);
}

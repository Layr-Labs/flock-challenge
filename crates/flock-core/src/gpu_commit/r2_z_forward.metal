#include <metal_stdlib>
using namespace metal;

struct R2ZWriter {
    device uint* a;
    device uint* b;
    uint aPending;
    uint bPending;
};

template <uint Bit>
inline uint r2z_read(device const uint* z) {
    constexpr uint shift = Bit % 32u;
    if (shift == 0u) return z[Bit / 32u];
    return (z[Bit / 32u] >> shift) | (z[Bit / 32u + 1u] << ((32u - shift) & 31u));
}

template <uint Bit, uint Width>
inline void r2z_append(thread R2ZWriter& w, uint a, uint b) {
    constexpr uint shift = Bit % 32u;
    uint av = w.aPending | (a << shift);
    uint bv = w.bPending | (b << shift);
    if (shift + Width >= 32u) {
        w.a[Bit / 32u] = av;
        w.b[Bit / 32u] = bv;
        w.aPending = shift == 0u ? 0u : a >> ((32u - shift) & 31u);
        w.bPending = shift == 0u ? 0u : b >> ((32u - shift) & 31u);
    } else {
        w.aPending = av;
        w.bPending = bv;
    }
}

template <uint Bit>
inline uint r2z_add(thread R2ZWriter& w, uint x, uint y) {
    uint sum = x + y;
    r2z_append<Bit, 31u>(w, (sum ^ y) & 0x7fffffffu, (sum ^ x) & 0x7fffffffu);
    return sum;
}

template <uint N>
inline uint r2z_rotr(uint x) { return (x >> N) | (x << (32u - N)); }

template <uint G>
inline void r2z_g(thread R2ZWriter& w, thread uint& a, thread uint& b,
                  thread uint& c, thread uint& d, uint x, uint y) {
    constexpr uint base = 1153u + G * 250u;
    uint t0 = r2z_add<base>(w, a, b);
    a = r2z_add<base + 31u>(w, t0, x);
    d = r2z_rotr<16u>(d ^ a);
    c = r2z_add<base + 62u>(w, c, d);
    b = r2z_rotr<12u>(b ^ c);
    uint t1 = r2z_add<base + 93u>(w, a, b);
    a = r2z_add<base + 124u>(w, t1, y);
    d = r2z_rotr<8u>(d ^ a);
    c = r2z_add<base + 155u>(w, c, d);
    b = r2z_rotr<7u>(b ^ c);
    r2z_append<base + 186u, 32u>(w, b, 0xffffffffu);
    r2z_append<base + 218u, 32u>(w, d, 0xffffffffu);
}

kernel void r2_reconstruct_raw_from_z(
    device const uint* zRows [[buffer(0)]],
    device uint* aRows [[buffer(1)]], device uint* bRows [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint block [[thread_position_in_grid]])
{
    if (block >= count) return;
    constexpr uint rowWords = 512u;
    device const uint* z = zRows + block * rowWords;
    R2ZWriter w = {aRows + block * rowWords, bRows + block * rowWords, z[36] & 1u, 1u};
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 36u; i++) {
        w.a[i] = z[i];
        w.b[i] = 0xffffffffu;
    }
    uint s[16] = {z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7],
                  0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                  r2z_read<1025u>(z), r2z_read<1057u>(z), r2z_read<1089u>(z), r2z_read<1121u>(z)};
    uint m[16] = {r2z_read<513u>(z), r2z_read<545u>(z), r2z_read<577u>(z), r2z_read<609u>(z),
                  r2z_read<641u>(z), r2z_read<673u>(z), r2z_read<705u>(z), r2z_read<737u>(z),
                  r2z_read<769u>(z), r2z_read<801u>(z), r2z_read<833u>(z), r2z_read<865u>(z),
                  r2z_read<897u>(z), r2z_read<929u>(z), r2z_read<961u>(z), r2z_read<993u>(z)};

#define R2Z_ROUND(R, M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15) \
    r2z_g<8u * R + 0u>(w, s[0], s[4], s[8], s[12], m[M0], m[M1]); \
    r2z_g<8u * R + 1u>(w, s[1], s[5], s[9], s[13], m[M2], m[M3]); \
    r2z_g<8u * R + 2u>(w, s[2], s[6], s[10], s[14], m[M4], m[M5]); \
    r2z_g<8u * R + 3u>(w, s[3], s[7], s[11], s[15], m[M6], m[M7]); \
    r2z_g<8u * R + 4u>(w, s[0], s[5], s[10], s[15], m[M8], m[M9]); \
    r2z_g<8u * R + 5u>(w, s[1], s[6], s[11], s[12], m[M10], m[M11]); \
    r2z_g<8u * R + 6u>(w, s[2], s[7], s[8], s[13], m[M12], m[M13]); \
    r2z_g<8u * R + 7u>(w, s[3], s[4], s[9], s[14], m[M14], m[M15]);
    R2Z_ROUND(0u, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    R2Z_ROUND(1u, 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
    R2Z_ROUND(2u, 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1)
    R2Z_ROUND(3u, 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6)
    R2Z_ROUND(4u, 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4)
    R2Z_ROUND(5u, 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7)
    R2Z_ROUND(6u, 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13)
#undef R2Z_ROUND

    r2z_append<15153u, 32u>(w, s[8] ^ z[0], 0xffffffffu);
    r2z_append<15185u, 32u>(w, s[9] ^ z[1], 0xffffffffu);
    r2z_append<15217u, 32u>(w, s[10] ^ z[2], 0xffffffffu);
    r2z_append<15249u, 32u>(w, s[11] ^ z[3], 0xffffffffu);
    r2z_append<15281u, 32u>(w, s[12] ^ z[4], 0xffffffffu);
    r2z_append<15313u, 32u>(w, s[13] ^ z[5], 0xffffffffu);
    r2z_append<15345u, 32u>(w, s[14] ^ z[6], 0xffffffffu);
    r2z_append<15377u, 32u>(w, s[15] ^ z[7], 0xffffffffu);
    w.a[15409u / 32u] = w.aPending;
    w.b[15409u / 32u] = w.bPending;
    for (uint i = 15409u / 32u + 1u; i < rowWords; i++) {
        w.a[i] = 0u;
        w.b[i] = 0u;
    }
}

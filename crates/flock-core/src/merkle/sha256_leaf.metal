#include <metal_stdlib>

using namespace metal;

// Test-only bounded Merkle leaf checksum gate. Each thread hashes exactly one
// 1,024-byte leaf as an independent standard SHA-256 message. The caller
// exposes page-aligned no-copy VM segments and proves every signed index below
// remains within the currently bound buffers.

inline uint sha256_rotr(uint value, uint distance)
{
    return (value >> distance) | (value << (32u - distance));
}

inline uint sha256_big_sigma0(uint value)
{
    return sha256_rotr(value, 2u) ^ sha256_rotr(value, 13u) ^
           sha256_rotr(value, 22u);
}

inline uint sha256_big_sigma1(uint value)
{
    return sha256_rotr(value, 6u) ^ sha256_rotr(value, 11u) ^
           sha256_rotr(value, 25u);
}

inline uint sha256_small_sigma0(uint value)
{
    return sha256_rotr(value, 7u) ^ sha256_rotr(value, 18u) ^ (value >> 3u);
}

inline uint sha256_small_sigma1(uint value)
{
    return sha256_rotr(value, 17u) ^ sha256_rotr(value, 19u) ^ (value >> 10u);
}

inline uint sha256_choose(uint x, uint y, uint z)
{
    return (x & y) ^ (~x & z);
}

inline uint sha256_majority(uint x, uint y, uint z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint sha256_byte_swap(uint value)
{
    return ((value & 0x000000ffu) << 24u) |
           ((value & 0x0000ff00u) << 8u) |
           ((value & 0x00ff0000u) >> 8u) |
           ((value & 0xff000000u) >> 24u);
}

// One round updates the variables that become the next round's E and A.
// Rotating the macro arguments performs the remaining six register renames.
#define SHA256_ROUND(A, B, C, D, E, F, G, H, K, W)                         \
    do {                                                                    \
        (H) += sha256_big_sigma1(E) + sha256_choose((E), (F), (G)) +       \
               (K) + (W);                                                   \
        (D) += (H);                                                         \
        (H) += sha256_big_sigma0(A) + sha256_majority((A), (B), (C));      \
    } while (false)

[[max_total_threads_per_threadgroup(32)]]
kernel void sha256_leaf_checksum(
    const device uint4 *input [[buffer(0)]],
    device uint4 *output [[buffer(1)]],
    constant uint &global_leaf_base [[buffer(2)]],
    constant uint &local_leaf_count [[buffer(3)]],
    constant uint &input_uint4_offset [[buffer(4)]],
    constant uint &output_uint4_offset [[buffer(5)]],
    uint local_gid [[thread_position_in_grid]])
{
    constexpr uint kLeafCount = 1u << 20;
    if (local_gid >= local_leaf_count) {
        return;
    }
    const uint global_gid = global_leaf_base + local_gid;
    if (global_gid >= kLeafCount) {
        return;
    }

    // One leaf is 64 uint4 values and one digest is two uint4 values. The host
    // proves the bases and final touched indices fit signed int.
    const int leaf = int(local_gid);
    const int input_base = int(input_uint4_offset) + (leaf << 6);
    const int output_base = int(output_uint4_offset) + (leaf << 1);

    // Eight named scalar state words; no private state or schedule arrays.
    uint a = 0x6a09e667u;
    uint b = 0xbb67ae85u;
    uint c = 0x3c6ef372u;
    uint d = 0xa54ff53au;
    uint e = 0x510e527fu;
    uint f = 0x9b05688cu;
    uint g = 0x1f83d9abu;
    uint h = 0x5be0cd19u;

    // Sixteen data compression blocks consume the full 1,024-byte leaf.
    // Rounds are source-unrolled and use sixteen named rolling schedule words.
    for (int block = 0; block < 16; ++block) {
        const int vector_base = input_base + (block << 2);
        const uint4 input0 = input[vector_base];
        const uint4 input1 = input[vector_base + 1];
        const uint4 input2 = input[vector_base + 2];
        const uint4 input3 = input[vector_base + 3];

        uint w0 = sha256_byte_swap(input0.x);
        uint w1 = sha256_byte_swap(input0.y);
        uint w2 = sha256_byte_swap(input0.z);
        uint w3 = sha256_byte_swap(input0.w);
        uint w4 = sha256_byte_swap(input1.x);
        uint w5 = sha256_byte_swap(input1.y);
        uint w6 = sha256_byte_swap(input1.z);
        uint w7 = sha256_byte_swap(input1.w);
        uint w8 = sha256_byte_swap(input2.x);
        uint w9 = sha256_byte_swap(input2.y);
        uint w10 = sha256_byte_swap(input2.z);
        uint w11 = sha256_byte_swap(input2.w);
        uint w12 = sha256_byte_swap(input3.x);
        uint w13 = sha256_byte_swap(input3.y);
        uint w14 = sha256_byte_swap(input3.z);
        uint w15 = sha256_byte_swap(input3.w);

        const uint saved_a = a;
        const uint saved_b = b;
        const uint saved_c = c;
        const uint saved_d = d;
        const uint saved_e = e;
        const uint saved_f = f;
        const uint saved_g = g;
        const uint saved_h = h;

        SHA256_ROUND(a, b, c, d, e, f, g, h, 0x428a2f98u, w0);
        SHA256_ROUND(h, a, b, c, d, e, f, g, 0x71374491u, w1);
        SHA256_ROUND(g, h, a, b, c, d, e, f, 0xb5c0fbcfu, w2);
        SHA256_ROUND(f, g, h, a, b, c, d, e, 0xe9b5dba5u, w3);
        SHA256_ROUND(e, f, g, h, a, b, c, d, 0x3956c25bu, w4);
        SHA256_ROUND(d, e, f, g, h, a, b, c, 0x59f111f1u, w5);
        SHA256_ROUND(c, d, e, f, g, h, a, b, 0x923f82a4u, w6);
        SHA256_ROUND(b, c, d, e, f, g, h, a, 0xab1c5ed5u, w7);
        SHA256_ROUND(a, b, c, d, e, f, g, h, 0xd807aa98u, w8);
        SHA256_ROUND(h, a, b, c, d, e, f, g, 0x12835b01u, w9);
        SHA256_ROUND(g, h, a, b, c, d, e, f, 0x243185beu, w10);
        SHA256_ROUND(f, g, h, a, b, c, d, e, 0x550c7dc3u, w11);
        SHA256_ROUND(e, f, g, h, a, b, c, d, 0x72be5d74u, w12);
        SHA256_ROUND(d, e, f, g, h, a, b, c, 0x80deb1feu, w13);
        SHA256_ROUND(c, d, e, f, g, h, a, b, 0x9bdc06a7u, w14);
        SHA256_ROUND(b, c, d, e, f, g, h, a, 0xc19bf174u, w15);

        w0 += sha256_small_sigma1(w14) + w9 + sha256_small_sigma0(w1);
        SHA256_ROUND(a, b, c, d, e, f, g, h, 0xe49b69c1u, w0);
        w1 += sha256_small_sigma1(w15) + w10 + sha256_small_sigma0(w2);
        SHA256_ROUND(h, a, b, c, d, e, f, g, 0xefbe4786u, w1);
        w2 += sha256_small_sigma1(w0) + w11 + sha256_small_sigma0(w3);
        SHA256_ROUND(g, h, a, b, c, d, e, f, 0x0fc19dc6u, w2);
        w3 += sha256_small_sigma1(w1) + w12 + sha256_small_sigma0(w4);
        SHA256_ROUND(f, g, h, a, b, c, d, e, 0x240ca1ccu, w3);
        w4 += sha256_small_sigma1(w2) + w13 + sha256_small_sigma0(w5);
        SHA256_ROUND(e, f, g, h, a, b, c, d, 0x2de92c6fu, w4);
        w5 += sha256_small_sigma1(w3) + w14 + sha256_small_sigma0(w6);
        SHA256_ROUND(d, e, f, g, h, a, b, c, 0x4a7484aau, w5);
        w6 += sha256_small_sigma1(w4) + w15 + sha256_small_sigma0(w7);
        SHA256_ROUND(c, d, e, f, g, h, a, b, 0x5cb0a9dcu, w6);
        w7 += sha256_small_sigma1(w5) + w0 + sha256_small_sigma0(w8);
        SHA256_ROUND(b, c, d, e, f, g, h, a, 0x76f988dau, w7);
        w8 += sha256_small_sigma1(w6) + w1 + sha256_small_sigma0(w9);
        SHA256_ROUND(a, b, c, d, e, f, g, h, 0x983e5152u, w8);
        w9 += sha256_small_sigma1(w7) + w2 + sha256_small_sigma0(w10);
        SHA256_ROUND(h, a, b, c, d, e, f, g, 0xa831c66du, w9);
        w10 += sha256_small_sigma1(w8) + w3 + sha256_small_sigma0(w11);
        SHA256_ROUND(g, h, a, b, c, d, e, f, 0xb00327c8u, w10);
        w11 += sha256_small_sigma1(w9) + w4 + sha256_small_sigma0(w12);
        SHA256_ROUND(f, g, h, a, b, c, d, e, 0xbf597fc7u, w11);
        w12 += sha256_small_sigma1(w10) + w5 + sha256_small_sigma0(w13);
        SHA256_ROUND(e, f, g, h, a, b, c, d, 0xc6e00bf3u, w12);
        w13 += sha256_small_sigma1(w11) + w6 + sha256_small_sigma0(w14);
        SHA256_ROUND(d, e, f, g, h, a, b, c, 0xd5a79147u, w13);
        w14 += sha256_small_sigma1(w12) + w7 + sha256_small_sigma0(w15);
        SHA256_ROUND(c, d, e, f, g, h, a, b, 0x06ca6351u, w14);
        w15 += sha256_small_sigma1(w13) + w8 + sha256_small_sigma0(w0);
        SHA256_ROUND(b, c, d, e, f, g, h, a, 0x14292967u, w15);

        w0 += sha256_small_sigma1(w14) + w9 + sha256_small_sigma0(w1);
        SHA256_ROUND(a, b, c, d, e, f, g, h, 0x27b70a85u, w0);
        w1 += sha256_small_sigma1(w15) + w10 + sha256_small_sigma0(w2);
        SHA256_ROUND(h, a, b, c, d, e, f, g, 0x2e1b2138u, w1);
        w2 += sha256_small_sigma1(w0) + w11 + sha256_small_sigma0(w3);
        SHA256_ROUND(g, h, a, b, c, d, e, f, 0x4d2c6dfcu, w2);
        w3 += sha256_small_sigma1(w1) + w12 + sha256_small_sigma0(w4);
        SHA256_ROUND(f, g, h, a, b, c, d, e, 0x53380d13u, w3);
        w4 += sha256_small_sigma1(w2) + w13 + sha256_small_sigma0(w5);
        SHA256_ROUND(e, f, g, h, a, b, c, d, 0x650a7354u, w4);
        w5 += sha256_small_sigma1(w3) + w14 + sha256_small_sigma0(w6);
        SHA256_ROUND(d, e, f, g, h, a, b, c, 0x766a0abbu, w5);
        w6 += sha256_small_sigma1(w4) + w15 + sha256_small_sigma0(w7);
        SHA256_ROUND(c, d, e, f, g, h, a, b, 0x81c2c92eu, w6);
        w7 += sha256_small_sigma1(w5) + w0 + sha256_small_sigma0(w8);
        SHA256_ROUND(b, c, d, e, f, g, h, a, 0x92722c85u, w7);
        w8 += sha256_small_sigma1(w6) + w1 + sha256_small_sigma0(w9);
        SHA256_ROUND(a, b, c, d, e, f, g, h, 0xa2bfe8a1u, w8);
        w9 += sha256_small_sigma1(w7) + w2 + sha256_small_sigma0(w10);
        SHA256_ROUND(h, a, b, c, d, e, f, g, 0xa81a664bu, w9);
        w10 += sha256_small_sigma1(w8) + w3 + sha256_small_sigma0(w11);
        SHA256_ROUND(g, h, a, b, c, d, e, f, 0xc24b8b70u, w10);
        w11 += sha256_small_sigma1(w9) + w4 + sha256_small_sigma0(w12);
        SHA256_ROUND(f, g, h, a, b, c, d, e, 0xc76c51a3u, w11);
        w12 += sha256_small_sigma1(w10) + w5 + sha256_small_sigma0(w13);
        SHA256_ROUND(e, f, g, h, a, b, c, d, 0xd192e819u, w12);
        w13 += sha256_small_sigma1(w11) + w6 + sha256_small_sigma0(w14);
        SHA256_ROUND(d, e, f, g, h, a, b, c, 0xd6990624u, w13);
        w14 += sha256_small_sigma1(w12) + w7 + sha256_small_sigma0(w15);
        SHA256_ROUND(c, d, e, f, g, h, a, b, 0xf40e3585u, w14);
        w15 += sha256_small_sigma1(w13) + w8 + sha256_small_sigma0(w0);
        SHA256_ROUND(b, c, d, e, f, g, h, a, 0x106aa070u, w15);

        w0 += sha256_small_sigma1(w14) + w9 + sha256_small_sigma0(w1);
        SHA256_ROUND(a, b, c, d, e, f, g, h, 0x19a4c116u, w0);
        w1 += sha256_small_sigma1(w15) + w10 + sha256_small_sigma0(w2);
        SHA256_ROUND(h, a, b, c, d, e, f, g, 0x1e376c08u, w1);
        w2 += sha256_small_sigma1(w0) + w11 + sha256_small_sigma0(w3);
        SHA256_ROUND(g, h, a, b, c, d, e, f, 0x2748774cu, w2);
        w3 += sha256_small_sigma1(w1) + w12 + sha256_small_sigma0(w4);
        SHA256_ROUND(f, g, h, a, b, c, d, e, 0x34b0bcb5u, w3);
        w4 += sha256_small_sigma1(w2) + w13 + sha256_small_sigma0(w5);
        SHA256_ROUND(e, f, g, h, a, b, c, d, 0x391c0cb3u, w4);
        w5 += sha256_small_sigma1(w3) + w14 + sha256_small_sigma0(w6);
        SHA256_ROUND(d, e, f, g, h, a, b, c, 0x4ed8aa4au, w5);
        w6 += sha256_small_sigma1(w4) + w15 + sha256_small_sigma0(w7);
        SHA256_ROUND(c, d, e, f, g, h, a, b, 0x5b9cca4fu, w6);
        w7 += sha256_small_sigma1(w5) + w0 + sha256_small_sigma0(w8);
        SHA256_ROUND(b, c, d, e, f, g, h, a, 0x682e6ff3u, w7);
        w8 += sha256_small_sigma1(w6) + w1 + sha256_small_sigma0(w9);
        SHA256_ROUND(a, b, c, d, e, f, g, h, 0x748f82eeu, w8);
        w9 += sha256_small_sigma1(w7) + w2 + sha256_small_sigma0(w10);
        SHA256_ROUND(h, a, b, c, d, e, f, g, 0x78a5636fu, w9);
        w10 += sha256_small_sigma1(w8) + w3 + sha256_small_sigma0(w11);
        SHA256_ROUND(g, h, a, b, c, d, e, f, 0x84c87814u, w10);
        w11 += sha256_small_sigma1(w9) + w4 + sha256_small_sigma0(w12);
        SHA256_ROUND(f, g, h, a, b, c, d, e, 0x8cc70208u, w11);
        w12 += sha256_small_sigma1(w10) + w5 + sha256_small_sigma0(w13);
        SHA256_ROUND(e, f, g, h, a, b, c, d, 0x90befffau, w12);
        w13 += sha256_small_sigma1(w11) + w6 + sha256_small_sigma0(w14);
        SHA256_ROUND(d, e, f, g, h, a, b, c, 0xa4506cebu, w13);
        w14 += sha256_small_sigma1(w12) + w7 + sha256_small_sigma0(w15);
        SHA256_ROUND(c, d, e, f, g, h, a, b, 0xbef9a3f7u, w14);
        w15 += sha256_small_sigma1(w13) + w8 + sha256_small_sigma0(w0);
        SHA256_ROUND(b, c, d, e, f, g, h, a, 0xc67178f2u, w15);

        a += saved_a;
        b += saved_b;
        c += saved_c;
        d += saved_d;
        e += saved_e;
        f += saved_f;
        g += saved_g;
        h += saved_h;
    }

    // The seventeenth block is fixed for every 1,024-byte message: one 1 bit,
    // zero fill, then the 64-bit big-endian length 8,192 (0x2000). Its complete
    // schedule is frozen as literals, so no padding schedule object exists.
    const uint saved_a = a;
    const uint saved_b = b;
    const uint saved_c = c;
    const uint saved_d = d;
    const uint saved_e = e;
    const uint saved_f = f;
    const uint saved_g = g;
    const uint saved_h = h;

    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x428a2f98u, 0x80000000u);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x71374491u, 0x00000000u);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0xb5c0fbcfu, 0x00000000u);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0xe9b5dba5u, 0x00000000u);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x3956c25bu, 0x00000000u);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x59f111f1u, 0x00000000u);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x923f82a4u, 0x00000000u);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0xab1c5ed5u, 0x00000000u);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0xd807aa98u, 0x00000000u);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x12835b01u, 0x00000000u);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x243185beu, 0x00000000u);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x550c7dc3u, 0x00000000u);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x72be5d74u, 0x00000000u);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x80deb1feu, 0x00000000u);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x9bdc06a7u, 0x00000000u);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0xc19bf174u, 0x00002000u);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0xe49b69c1u, 0x80000000u);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0xefbe4786u, 0x14000008u);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x0fc19dc6u, 0x00205000u);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x240ca1ccu, 0x00000880u);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x2de92c6fu, 0x22000800u);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x4a7484aau, 0x05500002u);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x5cb0a9dcu, 0x0508b542u);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x76f988dau, 0x80001602u);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x983e5152u, 0x60080010u);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0xa831c66du, 0x0a016005u);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0xb00327c8u, 0x00124685u);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0xbf597fc7u, 0xbe00ac18u);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0xc6e00bf3u, 0x70e2249cu);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0xd5a79147u, 0x48a97e2du);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x06ca6351u, 0xdec1a926u);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x14292967u, 0x01c9672eu);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x27b70a85u, 0x7e2b69d8u);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x2e1b2138u, 0xc78943b9u);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x4d2c6dfcu, 0x9a09b723u);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x53380d13u, 0x00807011u);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x650a7354u, 0x5c9ce3a9u);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x766a0abbu, 0xc417aff4u);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x81c2c92eu, 0x05096241u);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x92722c85u, 0x4d26873cu);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0xa2bfe8a1u, 0x18509086u);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0xa81a664bu, 0xd3192a17u);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0xc24b8b70u, 0x68a2ccf7u);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0xc76c51a3u, 0x8af60248u);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0xd192e819u, 0x81087599u);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0xd6990624u, 0x4b2ecbd9u);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0xf40e3585u, 0x66392173u);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x106aa070u, 0x3bee6356u);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x19a4c116u, 0x41633094u);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x1e376c08u, 0x67b8f4cau);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x2748774cu, 0x615df87eu);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x34b0bcb5u, 0x482f6955u);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x391c0cb3u, 0x05ec4721u);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x4ed8aa4au, 0x5e1e57fdu);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x5b9cca4fu, 0xbd5f2d91u);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x682e6ff3u, 0x9abb704eu);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x748f82eeu, 0x72a027efu);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x78a5636fu, 0x5b78199au);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x84c87814u, 0xc8c24449u);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x8cc70208u, 0xdf108058u);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x90befffau, 0x52cc91a5u);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0xa4506cebu, 0xfaf63996u);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0xbef9a3f7u, 0x0fc6e033u);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0xc67178f2u, 0x76e15b1bu);

    a += saved_a;
    b += saved_b;
    c += saved_c;
    d += saved_d;
    e += saved_e;
    f += saved_f;
    g += saved_g;
    h += saved_h;

    // SHA-256 digest bytes are big-endian. uint4 stores are native-endian on
    // Apple silicon, so byte-swap every state word before the two vector writes.
    output[output_base] = uint4(
        sha256_byte_swap(a), sha256_byte_swap(b),
        sha256_byte_swap(c), sha256_byte_swap(d));
    output[output_base + 1] = uint4(
        sha256_byte_swap(e), sha256_byte_swap(f),
        sha256_byte_swap(g), sha256_byte_swap(h));
}

#undef SHA256_ROUND

#include <metal_stdlib>
using namespace metal;

// Ranked from-z radix-256 prefix. One thread owns sixteen adjacent positions
// for four of the 64 field lanes. The first three nontrivial layers exchange
// those register columns inside a SIMD group; the final four layers are
// entirely thread-local. No codeword tile is staged in threadgroup memory.

static inline uint4 gf_shl8_r256(uint4 a) {
    uint h = a.w >> 24;
    uint4 r;
    r.w = (a.w << 8) | (a.z >> 24);
    r.z = (a.z << 8) | (a.y >> 24);
    r.y = (a.y << 8) | (a.x >> 24);
    r.x = (a.x << 8) ^ ((h << 7) ^ (h << 2) ^ (h << 1) ^ h);
    return r;
}

// `tabs` is the compact heap-ordered image of the first eight NTT layers:
// 255 twiddles x one 256-entry byte table.
static inline uint4 gf_mul_tab8_r256(
    uint4 v,
    device const uint4* tab)
{
    uint4 acc = uint4(0u);
    for (int i = 15; i >= 0; i--) {
        acc = gf_shl8_r256(acc);
        uint h = (v[i >> 2] >> ((i & 3) * 8)) & 0xffu;
        acc ^= tab[h];
    }
    return acc;
}

// All 32 lanes execute both shuffles. Only the upper butterfly endpoint
// performs the field product; its new-u is shuffled to the lower endpoint,
// which forms new-v = old-v ^ new-u.
#define R256_CROSS_ONE(E, MASK, CHUNK_BIT, TABLE, IS_ZERO)                     \
    {                                                                          \
        uint4 old_v = (E);                                                      \
        uint4 peer_v = simd_shuffle_xor(old_v, ushort(MASK));                   \
        uint4 new_u = old_v;                                                    \
        if ((chunk & (CHUNK_BIT)) == 0u) {                                      \
            if (!(IS_ZERO)) {                                                   \
                new_u ^= gf_mul_tab8_r256(peer_v, (TABLE));                     \
            }                                                                  \
        }                                                                      \
        uint4 peer_new_u = simd_shuffle_xor(new_u, ushort(MASK));               \
        (E) = ((chunk & (CHUNK_BIT)) == 0u) ? new_u : (old_v ^ peer_new_u);     \
    }

#define R256_CROSS_LAYER(MASK, CHUNK_BIT, TBASE, CEXPR)                        \
    {                                                                          \
        const uint cross_c = (CEXPR);                                           \
        const bool cross_zero = cross_c == 0u;                                  \
        device const uint4* cross_tab = &tabs[((TBASE) + cross_c) << 8];        \
        R256_CROSS_ONE(e0,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e1,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e2,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e3,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e4,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e5,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e6,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e7,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e8,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e9,  MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e10, MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e11, MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e12, MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e13, MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e14, MASK, CHUNK_BIT, cross_tab, cross_zero)             \
        R256_CROSS_ONE(e15, MASK, CHUNK_BIT, cross_tab, cross_zero)             \
    }

#define R256_LOCAL_BFLY(U, V, TABLE, IS_ZERO)                                 \
    {                                                                          \
        uint4 new_u = (U);                                                      \
        if (!(IS_ZERO)) {                                                       \
            new_u ^= gf_mul_tab8_r256((V), (TABLE));                            \
        }                                                                      \
        (U) = new_u;                                                            \
        (V) ^= new_u;                                                           \
    }

[[max_total_threads_per_threadgroup(64)]]
kernel void ntt_from_z_radix256_register_simd(
    device uint4* data             [[buffer(0)]],
    device const uint4* tabs       [[buffer(1)]],
    device const uint4* z          [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]])
{
    constexpr uint S = 12u;
    // 16 lane-quads cover the 64 independent F128 columns for one r tile.
    const uint lane_quad = tgid & 15u;
    const uint r = tgid >> 4;
    const uint lane = (lane_quad << 2) | (lid & 3u);
    const uint chunk = lid >> 2;       // high four bits of e, 0..15
    const uint source_chunk = chunk & 7u;
    const uint source_base = source_chunk << 4;
    const uint output_base = chunk << 4;

    // Layer zero crosses the rate-1/2 zero boundary. Its twiddle is zero, so
    // both outputs equal the lower message value; load that value directly
    // in both paired chunks.
    #define R256_LOAD(K)                                                        \
        uint4 e##K = z[((((source_base + (K)) << S) + r) << 6) + lane];
    R256_LOAD(0)  R256_LOAD(1)  R256_LOAD(2)  R256_LOAD(3)
    R256_LOAD(4)  R256_LOAD(5)  R256_LOAD(6)  R256_LOAD(7)
    R256_LOAD(8)  R256_LOAD(9)  R256_LOAD(10) R256_LOAD(11)
    R256_LOAD(12) R256_LOAD(13) R256_LOAD(14) R256_LOAD(15)
    #undef R256_LOAD

    // Global layers 1..3 pair chunk bits 2,1,0. With four field lanes per
    // chunk those are SIMD xor masks 16,8,4 respectively.
    R256_CROSS_LAYER(16u, 4u, 1u, chunk >> 3)
    R256_CROSS_LAYER(8u,  2u, 3u, chunk >> 2)
    R256_CROSS_LAYER(4u,  1u, 7u, chunk >> 1)

    // Global layer 4, local pair bit 3.
    {
        const bool zero = chunk == 0u;
        device const uint4* tab = &tabs[(15u + chunk) << 8];
        R256_LOCAL_BFLY(e0, e8, tab, zero)
        R256_LOCAL_BFLY(e1, e9, tab, zero)
        R256_LOCAL_BFLY(e2, e10, tab, zero)
        R256_LOCAL_BFLY(e3, e11, tab, zero)
        R256_LOCAL_BFLY(e4, e12, tab, zero)
        R256_LOCAL_BFLY(e5, e13, tab, zero)
        R256_LOCAL_BFLY(e6, e14, tab, zero)
        R256_LOCAL_BFLY(e7, e15, tab, zero)
    }

    // Global layer 5, local pair bit 2.
    {
        const uint c0 = chunk << 1;
        const bool zero0 = c0 == 0u;
        device const uint4* tab0 = &tabs[(31u + c0) << 8];
        R256_LOCAL_BFLY(e0, e4, tab0, zero0)
        R256_LOCAL_BFLY(e1, e5, tab0, zero0)
        R256_LOCAL_BFLY(e2, e6, tab0, zero0)
        R256_LOCAL_BFLY(e3, e7, tab0, zero0)
        device const uint4* tab1 = &tabs[(32u + c0) << 8];
        R256_LOCAL_BFLY(e8, e12, tab1, false)
        R256_LOCAL_BFLY(e9, e13, tab1, false)
        R256_LOCAL_BFLY(e10, e14, tab1, false)
        R256_LOCAL_BFLY(e11, e15, tab1, false)
    }

    // Global layer 6, local pair bit 1.
    {
        const uint c0 = chunk << 2;
        const bool zero0 = c0 == 0u;
        device const uint4* tab0 = &tabs[(63u + c0) << 8];
        device const uint4* tab1 = tab0 + 256u;
        device const uint4* tab2 = tab1 + 256u;
        device const uint4* tab3 = tab2 + 256u;
        R256_LOCAL_BFLY(e0, e2, tab0, zero0)
        R256_LOCAL_BFLY(e1, e3, tab0, zero0)
        R256_LOCAL_BFLY(e4, e6, tab1, false)
        R256_LOCAL_BFLY(e5, e7, tab1, false)
        R256_LOCAL_BFLY(e8, e10, tab2, false)
        R256_LOCAL_BFLY(e9, e11, tab2, false)
        R256_LOCAL_BFLY(e12, e14, tab3, false)
        R256_LOCAL_BFLY(e13, e15, tab3, false)
    }

    // Global layer 7, local pair bit 0.
    {
        const uint c0 = chunk << 3;
        const bool zero0 = c0 == 0u;
        device const uint4* tab0 = &tabs[(127u + c0) << 8];
        R256_LOCAL_BFLY(e0, e1, tab0 + 0u * 256u, zero0)
        R256_LOCAL_BFLY(e2, e3, tab0 + 1u * 256u, false)
        R256_LOCAL_BFLY(e4, e5, tab0 + 2u * 256u, false)
        R256_LOCAL_BFLY(e6, e7, tab0 + 3u * 256u, false)
        R256_LOCAL_BFLY(e8, e9, tab0 + 4u * 256u, false)
        R256_LOCAL_BFLY(e10, e11, tab0 + 5u * 256u, false)
        R256_LOCAL_BFLY(e12, e13, tab0 + 6u * 256u, false)
        R256_LOCAL_BFLY(e14, e15, tab0 + 7u * 256u, false)
    }

    #define R256_STORE(K)                                                       \
        data[((((output_base + (K)) << S) + r) << 6) + lane] = e##K;
    R256_STORE(0)  R256_STORE(1)  R256_STORE(2)  R256_STORE(3)
    R256_STORE(4)  R256_STORE(5)  R256_STORE(6)  R256_STORE(7)
    R256_STORE(8)  R256_STORE(9)  R256_STORE(10) R256_STORE(11)
    R256_STORE(12) R256_STORE(13) R256_STORE(14) R256_STORE(15)
    #undef R256_STORE
}

#undef R256_LOCAL_BFLY
#undef R256_CROSS_LAYER
#undef R256_CROSS_ONE

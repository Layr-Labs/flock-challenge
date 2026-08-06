#include <metal_stdlib>
using namespace metal;

// Ranked from-z radix-256 prefix. One threadgroup owns four of the 64 field
// lanes and one complete 256-position tile. The tile remains in threadgroup
// memory through layers 1..7, so the codeword is written to device memory
// only after all eight prefix layers are complete.

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

// Layer L pairs position bit (7-L). Each of the 64 threads owns eight of the
// 512 lane-specific butterflies, and every tile location has exactly one
// writer. The barrier publishes the completed layer before the next one.
#define R256_SHARED_LAYER(L, DIST, TBASE)                                      \
    {                                                                          \
        for (uint j = lid; j < 512u; j += 64u) {                               \
            const uint bf = j >> 2;                                            \
            const uint lane4 = j & 3u;                                         \
            const uint low = bf & ((DIST) - 1u);                               \
            const uint u_e = ((bf - low) << 1) | low;                          \
            const uint v_e = u_e + (DIST);                                     \
            const uint c = u_e >> (8u - (L));                                  \
            uint4 u = tile[(u_e << 2) | lane4];                                \
            uint4 v = tile[(v_e << 2) | lane4];                                \
            if (c != 0u) {                                                     \
                u ^= gf_mul_tab8_r256(v, &tabs[((TBASE) + c) << 8]);            \
            }                                                                  \
            v ^= u;                                                            \
            tile[(u_e << 2) | lane4] = u;                                      \
            tile[(v_e << 2) | lane4] = v;                                      \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
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
    threadgroup uint4 tile[1024];

    // Sixteen lane-quads cover the 64 independent F128 columns. Layer zero
    // crosses the rate-1/2 zero boundary with a zero twiddle, so both outputs
    // equal the corresponding lower-half message value.
    const uint lane_quad = tgid & 15u;
    const uint r = tgid >> 4;
    const uint lane4 = lid & 3u;
    const uint lane = (lane_quad << 2) | lane4;
    const uint output_base = (lid >> 2) << 4;
    for (uint k = 0u; k < 16u; k++) {
        const uint e = output_base + k;
        const uint source_e = e & 127u;
        tile[(e << 2) | lane4] = z[(((source_e << S) + r) << 6) + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    R256_SHARED_LAYER(1u, 64u, 1u)
    R256_SHARED_LAYER(2u, 32u, 3u)
    R256_SHARED_LAYER(3u, 16u, 7u)
    R256_SHARED_LAYER(4u, 8u, 15u)
    R256_SHARED_LAYER(5u, 4u, 31u)
    R256_SHARED_LAYER(6u, 2u, 63u)
    R256_SHARED_LAYER(7u, 1u, 127u)

    for (uint k = 0u; k < 16u; k++) {
        const uint e = output_base + k;
        data[(((e << S) + r) << 6) + lane] = tile[(e << 2) | lane4];
    }
}

#undef R256_SHARED_LAYER

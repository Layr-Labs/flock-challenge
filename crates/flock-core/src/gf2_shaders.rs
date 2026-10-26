pub const SOURCE: &str = r####"
#include <metal_stdlib>
using namespace metal;

// 256-Bit EVM Word (4 x 64-bit limbs) & Carry-Chain Arithmetic for Metal 3.0
struct ln_u256 { ulong l0; ulong l1; ulong l2; ulong l3; };

inline ln_u256 ln_u256_add(ln_u256 a, ln_u256 b) {
    ulong s0 = a.l0 + b.l0; ulong c0 = (s0 < a.l0) ? 1UL : 0UL;
    ulong t1 = a.l1 + b.l1; ulong c1a = (t1 < a.l1) ? 1UL : 0UL;
    ulong s1 = t1 + c0; ulong c1b = (s1 < t1) ? 1UL : 0UL; ulong c1 = c1a + c1b;
    ulong t2 = a.l2 + b.l2; ulong c2a = (t2 < a.l2) ? 1UL : 0UL;
    ulong s2 = t2 + c1; ulong c2b = (s2 < t2) ? 1UL : 0UL; ulong c2 = c2a + c2b;
    ulong s3 = a.l3 + b.l3 + c2; ln_u256 r = { s0, s1, s2, s3 }; return r;
}
inline ln_u256 ln_u256_sub(ln_u256 a, ln_u256 b) {
    ulong d0 = a.l0 - b.l0; ulong b0 = (a.l0 < b.l0) ? 1UL : 0UL;
    ulong t1 = a.l1 - b.l1; ulong b1a = (a.l1 < b.l1) ? 1UL : 0UL;
    ulong d1 = t1 - b0; ulong b1b = (t1 < b0) ? 1UL : 0UL; ulong b1 = b1a + b1b;
    ulong t2 = a.l2 - b.l2; ulong b2a = (a.l2 < b.l2) ? 1UL : 0UL;
    ulong d2 = t2 - b1; ulong b2b = (t2 < b1) ? 1UL : 0UL; ulong b2 = b2a + b2b;
    ulong d3 = a.l3 - b.l3 - b2; ln_u256 r = { d0, d1, d2, d3 }; return r;
}
inline ulong ln_mul64_hi(ulong x, ulong y) {
    ulong x_lo = x & 0xffffffffUL; ulong x_hi = x >> 32UL;
    ulong y_lo = y & 0xffffffffUL; ulong y_hi = y >> 32UL;
    ulong p0 = x_lo * y_lo; ulong p1 = x_lo * y_hi; ulong p2 = x_hi * y_lo; ulong p3 = x_hi * y_hi;
    ulong cy = ((p0 >> 32UL) + (p1 & 0xffffffffUL) + (p2 & 0xffffffffUL)) >> 32UL;
    return p3 + (p1 >> 32UL) + (p2 >> 32UL) + cy;
}
inline ln_u256 ln_u256_mul_lo(ln_u256 a, ln_u256 b) {
    ulong r0 = a.l0 * b.l0;
    ulong r1 = ln_mul64_hi(a.l0, b.l0) + a.l0 * b.l1 + a.l1 * b.l0;
    ulong r2 = a.l0 * b.l2 + a.l1 * b.l1 + a.l2 * b.l0;
    ulong r3 = a.l0 * b.l3 + a.l1 * b.l2 + a.l2 * b.l1 + a.l3 * b.l0;
    ln_u256 r = { r0, r1, r2, r3 }; return r;
}
inline ulong ln_clmul64(ulong a, ulong b) {
    if (b == 0UL || a == 0UL) { return 0UL; }
    if (b == 0x87UL) { return a ^ (a << 1UL) ^ (a << 2UL) ^ (a << 7UL); }
    ulong r = 0UL;
    for (uint i = 0u; i < 64u; ++i) {
        r ^= a & (0UL - ((b >> i) & 1UL));
        a <<= 1UL;
    }
    return r;
}
inline ulong ln_clmul64_hi(ulong a, ulong b) {
    if (b == 0UL || a == 0UL) { return 0UL; }
    if (b == 0x87UL) { return (a >> 63UL) ^ (a >> 62UL) ^ (a >> 57UL); }
    ulong r = 0UL;
    for (uint i = 1u; i < 64u; ++i) {
        r ^= (a >> (64u - i)) & (0UL - ((b >> i) & 1UL));
    }
    return r;
}
inline uint4 ln_gf128_mulx(uint4 v) {
    uint c0 = v.x >> 31u; uint c1 = v.y >> 31u; uint c2 = v.z >> 31u; uint c3 = v.w >> 31u;
    return uint4((v.x << 1u) ^ (c3 * 0x87u), (v.y << 1u) | c0, (v.z << 1u) | c1, (v.w << 1u) | c2);
}
inline uint4 ln_gf128_shl8(uint4 a) {
    uint hi = a.w >> 24u;
    uint4 s = uint4(a.x << 8u, (a.y << 8u) | (a.x >> 24u), (a.z << 8u) | (a.y >> 24u), (a.w << 8u) | (a.z >> 24u));
    uint r_lo = (hi << 7u) ^ (hi << 2u) ^ (hi << 1u) ^ hi;
    uint r_hi = (hi >> 25u) ^ (hi >> 30u) ^ (hi >> 31u);
    return uint4(s.x ^ r_lo, s.y ^ r_hi, s.z, s.w);
}
inline uint4 ln_gf128_shl16(uint4 a) {
    uint hi = a.w >> 16u;
    uint4 s = uint4(a.x << 16u, (a.y << 16u) | (a.x >> 16u), (a.z << 16u) | (a.y >> 16u), (a.w << 16u) | (a.z >> 16u));
    uint r_lo = (hi << 7u) ^ (hi << 2u) ^ (hi << 1u) ^ hi;
    uint r_hi = (hi >> 25u) ^ (hi >> 30u) ^ (hi >> 31u);
    return uint4(s.x ^ r_lo, s.y ^ r_hi, s.z, s.w);
}
inline ln_u256 ln_u256_clz(ln_u256 a) {
    ulong z = 256UL;
    if (a.l3 != 0UL) { z = (ulong)clz(a.l3); }
    else if (a.l2 != 0UL) { z = 64UL + (ulong)clz(a.l2); }
    else if (a.l1 != 0UL) { z = 128UL + (ulong)clz(a.l1); }
    else if (a.l0 != 0UL) { z = 192UL + (ulong)clz(a.l0); }
    ln_u256 r = { z, 0UL, 0UL, 0UL }; return r;
}

kernel void k0_kernel_map_scale_add(
    device const ulong* in [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& scale [[buffer(2)]],
    constant ulong& bias [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = in[gid] * scale + bias;
    }
}

kernel void k0_kernel_zip_add(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = in0[gid] + in1[gid];
    }
}

kernel void k0_kernel_zip_mul(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = in0[gid] * in1[gid];
    }
}

kernel void k0_kernel_saxpy(
    device const ulong* x [[buffer(0)]],
    device const ulong* y [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& alpha [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = alpha * x[gid] + y[gid];
    }
}

kernel void k0_kernel_stencil1d(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& wl [[buffer(2)]],
    constant ulong& wc [[buffer(3)]],
    constant ulong& wr [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong i = (ulong)gid;
    if (i < n) {
        if (i == 0UL || i + 1UL >= n) { out[i] = in0[i]; }
        else { out[i] = wl * in0[i - 1UL] + wc * in0[i] + wr * in0[i + 1UL]; }
    }
}

kernel void k0_kernel_dot_reduce_u64(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* block_partials [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup ulong sdata[8];
    ulong acc = ((ulong)gid < n) ? (in0[gid] * in1[gid]) : 0UL;
    ulong sg_sum = simd_sum(acc);
    if (simd_lane == 0u) { sdata[simd_id] = sg_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        ulong final_val = (simd_lane < 8u) ? sdata[simd_lane] : 0UL;
        ulong block_sum = simd_sum(final_val);
        if (simd_lane == 0u) { block_partials[bid] = block_sum; }
    }
}

kernel void k0_kernel_matmul_tiled_u64(
    device const ulong* A [[buffer(0)]],
    device const ulong* B [[buffer(1)]],
    device ulong* C [[buffer(2)]],
    constant ulong& dim [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 ctaid [[threadgroup_position_in_grid]]
) {
    // Bank-conflict-free +1 padded threadgroup tiles (16 x 17 = 272 ulongs)
    threadgroup ulong tileA[272];
    threadgroup ulong tileB[272];
    ulong row = (ulong)(ctaid.y * 16u + tid.y);
    ulong col = (ulong)(ctaid.x * 16u + tid.x);
    ulong acc = 0UL;
    ulong num_tiles = (dim + 15UL) / 16UL;
    for (ulong t = 0UL; t < num_tiles; ++t) {
        ulong a_col = t * 16UL + (ulong)tid.x;
        ulong b_row = t * 16UL + (ulong)tid.y;
        tileA[tid.y * 17u + tid.x] = (row < dim && a_col < dim) ? A[row * dim + a_col] : 0UL;
        tileB[tid.y * 17u + tid.x] = (b_row < dim && col < dim) ? B[b_row * dim + col] : 0UL;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < 16u; ++k) {
            acc += tileA[tid.y * 17u + k] * tileB[k * 17u + tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < dim && col < dim) {
        C[row * dim + col] = acc;
    }
}

inline ulong gf2_kern_gf8_mul_composite(ulong v0, ulong v1) {
    ulong v2 = 15UL;
    ulong v3 = (v0 & 15UL);
    ulong v4 = 4UL;
    ulong v5 = (v0 >> (4UL & 63UL));
    ulong v6 = (v5 & 15UL);
    ulong v7 = (v1 & 15UL);
    ulong v8 = (v1 >> (4UL & 63UL));
    ulong v9 = (v8 & 15UL);
    ulong v10 = (v3 & 15UL);
    ulong v11 = (v7 & 15UL);
    ulong v12 = ln_clmul64(v10, v11);
    ulong v13 = 127UL;
    ulong v14 = (v12 & 127UL);
    ulong v15 = (v14 >> (4UL & 63UL));
    ulong v16 = 1UL;
    ulong v17 = (v15 << (1UL & 63UL));
    ulong v18 = (v14 ^ v17);
    ulong v19 = (v18 ^ v15);
    ulong v20 = (v19 & 15UL);
    ulong v21 = (v6 & 15UL);
    ulong v22 = (v9 & 15UL);
    ulong v23 = ln_clmul64(v21, v22);
    ulong v24 = (v23 & 127UL);
    ulong v25 = (v24 >> (4UL & 63UL));
    ulong v26 = (v25 << (1UL & 63UL));
    ulong v27 = (v24 ^ v26);
    ulong v28 = (v27 ^ v25);
    ulong v29 = (v28 & 15UL);
    ulong v30 = (v3 ^ v6);
    ulong v31 = (v30 & 15UL);
    ulong v32 = (v7 ^ v9);
    ulong v33 = (v32 & 15UL);
    ulong v34 = ln_clmul64(v31, v33);
    ulong v35 = (v34 & 127UL);
    ulong v36 = (v35 >> (4UL & 63UL));
    ulong v37 = (v36 << (1UL & 63UL));
    ulong v38 = (v35 ^ v37);
    ulong v39 = (v38 ^ v36);
    ulong v40 = (v39 & 15UL);
    ulong v41 = 9UL;
    ulong v42 = (v29 & 15UL);
    ulong v43 = ln_clmul64(9UL, v42);
    ulong v44 = (v43 & 127UL);
    ulong v45 = (v44 >> (4UL & 63UL));
    ulong v46 = (v45 << (1UL & 63UL));
    ulong v47 = (v44 ^ v46);
    ulong v48 = (v47 ^ v45);
    ulong v49 = (v48 & 15UL);
    ulong v50 = (v20 ^ v49);
    ulong v51 = (v40 ^ v20);
    ulong v52 = (v51 << (4UL & 63UL));
    ulong v53 = (v50 | v52);
    return v53;
}

kernel void k0_metal_decl_gf2_kern_gf8_mul_composite(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_mul_composite(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_clmul64_hi(ulong v0, ulong v1) {
    ulong v2 = ln_clmul64_hi(v0, v1);
    return v2;
}

kernel void k0_metal_decl_gf2_kern_clmul64_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_clmul64_hi(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_bit_deposit_lowest_one(ulong v0, ulong v1) {
    ulong v2 = 0UL;
    ulong v3 = (0UL - v1);
    ulong v4 = (v1 & v3);
    ulong v5 = 1UL;
    ulong v6 = (v0 & 1UL);
    ulong v7 = (0UL - v6);
    ulong v8 = (v4 & v7);
    return v8;
}

kernel void k0_metal_decl_gf2_kern_bit_deposit_lowest_one(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_bit_deposit_lowest_one(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_blake3_add_sum(ulong v0, ulong v1) {
    ulong v2 = 4294967295UL;
    ulong v3 = (v0 & 4294967295UL);
    ulong v4 = (v1 & 4294967295UL);
    ulong v5 = (v3 + v4);
    ulong v6 = (v5 & 4294967295UL);
    return v6;
}

kernel void k0_metal_decl_gf2_kern_blake3_add_sum(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_blake3_add_sum(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_gf8_mul(ulong v0, ulong v1) {
    ulong v2 = 255UL;
    ulong v3 = (v0 & 255UL);
    ulong v4 = (v1 & 255UL);
    ulong v5 = ln_clmul64(v3, v4);
    ulong v6 = 65535UL;
    ulong v7 = (v5 & 65535UL);
    ulong v8 = 8UL;
    ulong v9 = (v7 >> (8UL & 63UL));
    ulong v10 = (v7 & 255UL);
    ulong v11 = (v10 ^ v9);
    ulong v12 = 1UL;
    ulong v13 = (v9 << (1UL & 63UL));
    ulong v14 = (v11 ^ v13);
    ulong v15 = 3UL;
    ulong v16 = (v9 << (3UL & 63UL));
    ulong v17 = (v14 ^ v16);
    ulong v18 = 4UL;
    ulong v19 = (v9 << (4UL & 63UL));
    ulong v20 = (v17 ^ v19);
    ulong v21 = (v20 >> (8UL & 63UL));
    ulong v22 = (v20 & 255UL);
    ulong v23 = (v22 ^ v21);
    ulong v24 = (v21 << (1UL & 63UL));
    ulong v25 = (v23 ^ v24);
    ulong v26 = (v21 << (3UL & 63UL));
    ulong v27 = (v25 ^ v26);
    ulong v28 = (v21 << (4UL & 63UL));
    ulong v29 = (v27 ^ v28);
    ulong v30 = (v29 & 255UL);
    return v30;
}

kernel void k0_metal_decl_gf2_kern_gf8_mul(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_mul(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_blake3_add_left(ulong v0, ulong v1) {
    ulong v2 = 4294967295UL;
    ulong v3 = 2147483647UL;
    ulong v4 = (v0 & 4294967295UL);
    ulong v5 = (v1 & 4294967295UL);
    ulong v6 = (v4 + v5);
    ulong v7 = (v6 & 4294967295UL);
    ulong v8 = (v7 ^ v4);
    ulong v9 = (v8 ^ v5);
    ulong v10 = (v4 ^ v9);
    ulong v11 = (v10 & 2147483647UL);
    return v11;
}

kernel void k0_metal_decl_gf2_kern_blake3_add_left(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_blake3_add_left(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_ntt_radix4_comb_d(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4, ulong v5, ulong v6) {
    ulong v7 = (v3 ^ v1);
    ulong v8 = (v7 ^ v5);
    ulong v9 = (v8 ^ v2);
    ulong v10 = (v9 ^ v0);
    ulong v11 = (v10 ^ v4);
    ulong v12 = (v11 ^ v6);
    return v12;
}

kernel void k0_metal_decl_gf2_kern_ntt_radix4_comb_d(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device const ulong* in5 [[buffer(5)]],
    device const ulong* in6 [[buffer(6)]],
    device ulong* out [[buffer(7)]],
    constant ulong& n [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ntt_radix4_comb_d(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid], in5[gid], in6[gid]);
    }
}

inline ulong gf2_kern_mle_fold_lo(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4, ulong v5) {
    ulong v6 = (v0 ^ v2);
    ulong v7 = (v1 ^ v3);
    ulong v8 = ln_clmul64(v6, v4);
    ulong v9 = ln_clmul64_hi(v6, v5);
    ulong v10 = ln_clmul64_hi(v7, v4);
    ulong v11 = ln_clmul64(v7, v5);
    ulong v12 = ln_clmul64_hi(v7, v5);
    ulong v13 = 135UL;
    ulong v14 = ln_clmul64_hi(v12, 135UL);
    ulong v15 = (v9 ^ v10);
    ulong v16 = (v15 ^ v14);
    ulong v17 = (v11 ^ v16);
    ulong v18 = ln_clmul64(v17, 135UL);
    ulong v19 = (v8 ^ v18);
    ulong v20 = (v0 ^ v19);
    return v20;
}

kernel void k0_metal_decl_gf2_kern_mle_fold_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device const ulong* in5 [[buffer(5)]],
    device ulong* out [[buffer(6)]],
    constant ulong& n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_mle_fold_lo(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid], in5[gid]);
    }
}

inline ulong gf2_kern_slot_indicator_lo(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4) {
    ulong v5 = 1UL;
    ulong v6 = (v0 & 1UL);
    ulong v7 = (v6 ^ 1UL);
    ulong v8 = (v1 ^ v7);
    ulong v9 = (v0 >> (1UL & 63UL));
    ulong v10 = (v9 & 1UL);
    ulong v11 = (v10 ^ 1UL);
    ulong v12 = (v3 ^ v11);
    ulong v1017 = ln_clmul64(v8, v12);
    ulong v1018 = ln_clmul64_hi(v8, v4);
    ulong v1019 = ln_clmul64_hi(v2, v12);
    ulong v1020 = ln_clmul64(v2, v4);
    ulong v1021 = ln_clmul64_hi(v2, v4);
    ulong v1022 = 135UL;
    ulong v1023 = ln_clmul64_hi(v1021, 135UL);
    ulong v1024 = (v1018 ^ v1019);
    ulong v1025 = (v1024 ^ v1023);
    ulong v1026 = (v1020 ^ v1025);
    ulong v1027 = ln_clmul64(v1026, 135UL);
    ulong v1028 = (v1017 ^ v1027);
    ulong v13 = v1028;
    return v13;
}

kernel void k0_metal_decl_gf2_kern_slot_indicator_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device ulong* out [[buffer(5)]],
    constant ulong& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_slot_indicator_lo(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid]);
    }
}

inline ulong gf2_kern_gf8_unreduced_odd_x2(ulong v0, ulong v1) {
    ulong v2 = 8UL;
    ulong v3 = (v0 >> (8UL & 63UL));
    ulong v4 = 16711935UL;
    ulong v5 = (v3 & 16711935UL);
    ulong v6 = (v1 >> (8UL & 63UL));
    ulong v7 = (v6 & 16711935UL);
    ulong v8 = ln_clmul64(v5, v7);
    ulong v9 = 281470681808895UL;
    ulong v10 = (v8 & 281470681808895UL);
    return v10;
}

kernel void k0_metal_decl_gf2_kern_gf8_unreduced_odd_x2(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_unreduced_odd_x2(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_slot_indicator_hi(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4) {
    ulong v5 = 1UL;
    ulong v6 = (v0 & 1UL);
    ulong v7 = (v6 ^ 1UL);
    ulong v8 = (v1 ^ v7);
    ulong v9 = (v0 >> (1UL & 63UL));
    ulong v10 = (v9 & 1UL);
    ulong v11 = (v10 ^ 1UL);
    ulong v12 = (v3 ^ v11);
    ulong v1017 = ln_clmul64_hi(v8, v12);
    ulong v1018 = ln_clmul64(v8, v4);
    ulong v1019 = ln_clmul64_hi(v8, v4);
    ulong v1020 = ln_clmul64(v2, v12);
    ulong v1021 = ln_clmul64_hi(v2, v12);
    ulong v1022 = ln_clmul64(v2, v4);
    ulong v1023 = ln_clmul64_hi(v2, v4);
    ulong v1024 = 135UL;
    ulong v1025 = ln_clmul64(v1023, 135UL);
    ulong v1026 = ln_clmul64_hi(v1023, 135UL);
    ulong v1027 = (v1019 ^ v1021);
    ulong v1028 = (v1027 ^ v1026);
    ulong v1029 = (v1022 ^ v1028);
    ulong v1030 = ln_clmul64_hi(v1029, 135UL);
    ulong v1031 = (v1018 ^ v1020);
    ulong v1032 = (v1031 ^ v1025);
    ulong v1033 = (v1017 ^ v1030);
    ulong v1034 = (v1032 ^ v1033);
    ulong v13 = v1034;
    return v13;
}

kernel void k0_metal_decl_gf2_kern_slot_indicator_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device ulong* out [[buffer(5)]],
    constant ulong& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_slot_indicator_hi(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid]);
    }
}

inline ulong gf2_kern_ntt_btf_v_lo(ulong v0, ulong v1, ulong v2) {
    ulong v3 = (v0 ^ v2);
    ulong v4 = (v1 ^ v3);
    return v4;
}

kernel void k0_metal_decl_gf2_kern_ntt_btf_v_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ntt_btf_v_lo(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_blake3_add_sum_carry(ulong v0, ulong v1) {
    ulong v2 = 4294967295UL;
    ulong v3 = 2147483647UL;
    ulong v4 = (v0 & 4294967295UL);
    ulong v5 = (v1 & 4294967295UL);
    ulong v6 = (v4 + v5);
    ulong v7 = (v6 & 4294967295UL);
    ulong v8 = (v7 ^ v4);
    ulong v9 = (v8 ^ v5);
    ulong v10 = (v4 ^ v9);
    ulong v11 = (v10 & 2147483647UL);
    ulong v12 = (v5 ^ v9);
    ulong v13 = (v12 & 2147483647UL);
    ulong v14 = (v11 & v13);
    ulong v15 = 32UL;
    ulong v16 = (v14 << (32UL & 63UL));
    ulong v17 = (v7 | v16);
    return v17;
}

kernel void k0_metal_decl_gf2_kern_blake3_add_sum_carry(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_blake3_add_sum_carry(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_transpose_8x8_u64(ulong v0) {
    ulong v1 = 47851476196393130UL;
    ulong v2 = 225176545447116UL;
    ulong v3 = 4042322160UL;
    ulong v4 = 7UL;
    ulong v5 = (v0 >> (7UL & 63UL));
    ulong v6 = (v0 ^ v5);
    ulong v7 = (v6 & 47851476196393130UL);
    ulong v8 = (v0 ^ v7);
    ulong v9 = (v7 << (7UL & 63UL));
    ulong v10 = (v8 ^ v9);
    ulong v11 = 14UL;
    ulong v12 = (v10 >> (14UL & 63UL));
    ulong v13 = (v10 ^ v12);
    ulong v14 = (v13 & 225176545447116UL);
    ulong v15 = (v10 ^ v14);
    ulong v16 = (v14 << (14UL & 63UL));
    ulong v17 = (v15 ^ v16);
    ulong v18 = 28UL;
    ulong v19 = (v17 >> (28UL & 63UL));
    ulong v20 = (v17 ^ v19);
    ulong v21 = (v20 & 4042322160UL);
    ulong v22 = (v17 ^ v21);
    ulong v23 = (v21 << (28UL & 63UL));
    ulong v24 = (v22 ^ v23);
    return v24;
}

kernel void k0_metal_decl_gf2_kern_transpose_8x8_u64(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_transpose_8x8_u64(in0[gid]);
    }
}

inline ulong gf2_kern_nibble_lo(ulong v0) {
    ulong v1 = 15UL;
    ulong v2 = (v0 & 15UL);
    return v2;
}

kernel void k0_metal_decl_gf2_kern_nibble_lo(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_nibble_lo(in0[gid]);
    }
}

inline ulong gf2_kern_gf8_to_composite(ulong v0) {
    ulong v1 = 0UL;
    ulong v2 = 1UL;
    ulong v3 = (v0 & 1UL);
    ulong v4 = (0UL - v3);
    ulong v5 = (v0 >> (1UL & 63UL));
    ulong v6 = (v5 & 1UL);
    ulong v7 = (0UL - v6);
    ulong v8 = 2UL;
    ulong v9 = (v0 >> (2UL & 63UL));
    ulong v10 = (v9 & 1UL);
    ulong v11 = (0UL - v10);
    ulong v12 = 3UL;
    ulong v13 = (v0 >> (3UL & 63UL));
    ulong v14 = (v13 & 1UL);
    ulong v15 = (0UL - v14);
    ulong v16 = 4UL;
    ulong v17 = (v0 >> (4UL & 63UL));
    ulong v18 = (v17 & 1UL);
    ulong v19 = (0UL - v18);
    ulong v20 = 5UL;
    ulong v21 = (v0 >> (5UL & 63UL));
    ulong v22 = (v21 & 1UL);
    ulong v23 = (0UL - v22);
    ulong v24 = 6UL;
    ulong v25 = (v0 >> (6UL & 63UL));
    ulong v26 = (v25 & 1UL);
    ulong v27 = (0UL - v26);
    ulong v28 = 7UL;
    ulong v29 = (v0 >> (7UL & 63UL));
    ulong v30 = (v29 & 1UL);
    ulong v31 = (0UL - v30);
    ulong v32 = (v4 & 1UL);
    ulong v33 = 73UL;
    ulong v34 = (v7 & 73UL);
    ulong v35 = (v32 ^ v34);
    ulong v36 = 53UL;
    ulong v37 = (v11 & 53UL);
    ulong v38 = (v35 ^ v37);
    ulong v39 = 61UL;
    ulong v40 = (v15 & 61UL);
    ulong v41 = (v38 ^ v40);
    ulong v42 = 89UL;
    ulong v43 = (v19 & 89UL);
    ulong v44 = (v41 ^ v43);
    ulong v45 = 231UL;
    ulong v46 = (v23 & 231UL);
    ulong v47 = (v44 ^ v46);
    ulong v48 = 85UL;
    ulong v49 = (v27 & 85UL);
    ulong v50 = (v47 ^ v49);
    ulong v51 = 177UL;
    ulong v52 = (v31 & 177UL);
    ulong v53 = (v50 ^ v52);
    return v53;
}

kernel void k0_metal_decl_gf2_kern_gf8_to_composite(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_to_composite(in0[gid]);
    }
}

inline ulong gf2_kern_ntt_btf_u_lo(ulong v0, ulong v1) {
    ulong v2 = (v0 ^ v1);
    return v2;
}

kernel void k0_metal_decl_gf2_kern_ntt_btf_u_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ntt_btf_u_lo(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_eq_bit_mul_hi(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4) {
    ulong v5 = 1UL;
    ulong v6 = (v0 & 1UL);
    ulong v7 = (v6 ^ 1UL);
    ulong v8 = (v3 ^ v7);
    ulong v1013 = ln_clmul64_hi(v1, v8);
    ulong v1014 = ln_clmul64(v1, v4);
    ulong v1015 = ln_clmul64_hi(v1, v4);
    ulong v1016 = ln_clmul64(v2, v8);
    ulong v1017 = ln_clmul64_hi(v2, v8);
    ulong v1018 = ln_clmul64(v2, v4);
    ulong v1019 = ln_clmul64_hi(v2, v4);
    ulong v1020 = 135UL;
    ulong v1021 = ln_clmul64(v1019, 135UL);
    ulong v1022 = ln_clmul64_hi(v1019, 135UL);
    ulong v1023 = (v1015 ^ v1017);
    ulong v1024 = (v1023 ^ v1022);
    ulong v1025 = (v1018 ^ v1024);
    ulong v1026 = ln_clmul64_hi(v1025, 135UL);
    ulong v1027 = (v1014 ^ v1016);
    ulong v1028 = (v1027 ^ v1021);
    ulong v1029 = (v1013 ^ v1026);
    ulong v1030 = (v1028 ^ v1029);
    ulong v9 = v1030;
    return v9;
}

kernel void k0_metal_decl_gf2_kern_eq_bit_mul_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device ulong* out [[buffer(5)]],
    constant ulong& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_eq_bit_mul_hi(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid]);
    }
}

inline ulong gf2_kern_gf8_unreduced_x4(ulong v0, ulong v1) {
    ulong v2 = 16711935UL;
    ulong v3 = (v0 & 16711935UL);
    ulong v4 = (v1 & 16711935UL);
    ulong v5 = 8UL;
    ulong v6 = (v0 >> (8UL & 63UL));
    ulong v7 = (v6 & 16711935UL);
    ulong v8 = (v1 >> (8UL & 63UL));
    ulong v9 = (v8 & 16711935UL);
    ulong v10 = ln_clmul64(v3, v4);
    ulong v11 = 281470681808895UL;
    ulong v12 = (v10 & 281470681808895UL);
    ulong v13 = ln_clmul64(v7, v9);
    ulong v14 = (v13 & 281470681808895UL);
    ulong v15 = 16UL;
    ulong v16 = (v14 << (16UL & 63UL));
    ulong v17 = (v12 | v16);
    return v17;
}

kernel void k0_metal_decl_gf2_kern_gf8_unreduced_x4(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_unreduced_x4(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_blake3_add_lr(ulong v0, ulong v1) {
    ulong v2 = 4294967295UL;
    ulong v3 = 2147483647UL;
    ulong v4 = (v0 & 4294967295UL);
    ulong v5 = (v1 & 4294967295UL);
    ulong v6 = (v4 + v5);
    ulong v7 = (v6 & 4294967295UL);
    ulong v8 = (v7 ^ v4);
    ulong v9 = (v8 ^ v5);
    ulong v10 = (v4 ^ v9);
    ulong v11 = (v10 & 2147483647UL);
    ulong v12 = (v5 ^ v9);
    ulong v13 = (v12 & 2147483647UL);
    ulong v14 = 32UL;
    ulong v15 = (v13 << (32UL & 63UL));
    ulong v16 = (v11 | v15);
    return v16;
}

kernel void k0_metal_decl_gf2_kern_blake3_add_lr(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_blake3_add_lr(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_ghash_reduce_hi(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 135UL;
    ulong v4 = ln_clmul64_hi(v2, 135UL);
    ulong v5 = (v1 ^ v4);
    ulong v6 = ln_clmul64(v2, 135UL);
    ulong v7 = (v0 ^ v6);
    ulong v8 = ln_clmul64_hi(v5, 135UL);
    ulong v9 = (v7 ^ v8);
    return v9;
}

kernel void k0_metal_decl_gf2_kern_ghash_reduce_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ghash_reduce_hi(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_ntt_radix4_comb_c(ulong v0, ulong v1, ulong v2, ulong v3) {
    ulong v4 = (v1 ^ v0);
    ulong v5 = (v4 ^ v2);
    ulong v6 = (v5 ^ v3);
    return v6;
}

kernel void k0_metal_decl_gf2_kern_ntt_radix4_comb_c(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device ulong* out [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ntt_radix4_comb_c(in0[gid], in1[gid], in2[gid], in3[gid]);
    }
}

inline ulong gf2_kern_ghash_mul_hi(ulong v0, ulong v1, ulong v2, ulong v3) {
    ulong v4 = ln_clmul64_hi(v0, v2);
    ulong v5 = ln_clmul64(v0, v3);
    ulong v6 = ln_clmul64_hi(v0, v3);
    ulong v7 = ln_clmul64(v1, v2);
    ulong v8 = ln_clmul64_hi(v1, v2);
    ulong v9 = ln_clmul64(v1, v3);
    ulong v10 = ln_clmul64_hi(v1, v3);
    ulong v11 = 135UL;
    ulong v12 = ln_clmul64(v10, 135UL);
    ulong v13 = ln_clmul64_hi(v10, 135UL);
    ulong v14 = (v6 ^ v8);
    ulong v15 = (v14 ^ v13);
    ulong v16 = (v9 ^ v15);
    ulong v17 = ln_clmul64_hi(v16, 135UL);
    ulong v18 = (v5 ^ v7);
    ulong v19 = (v18 ^ v12);
    ulong v20 = (v4 ^ v17);
    ulong v21 = (v19 ^ v20);
    return v21;
}

kernel void k0_metal_decl_gf2_kern_ghash_mul_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device ulong* out [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ghash_mul_hi(in0[gid], in1[gid], in2[gid], in3[gid]);
    }
}

inline ulong gf2_kern_gf8_unreduced_even_x2(ulong v0, ulong v1) {
    ulong v2 = 16711935UL;
    ulong v3 = (v0 & 16711935UL);
    ulong v4 = (v1 & 16711935UL);
    ulong v5 = ln_clmul64(v3, v4);
    ulong v6 = 281470681808895UL;
    ulong v7 = (v5 & 281470681808895UL);
    return v7;
}

kernel void k0_metal_decl_gf2_kern_gf8_unreduced_even_x2(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_unreduced_even_x2(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_f128_reduce_deg191_hi(ulong v0, ulong v1, ulong v2) {
    ulong v3 = (v0 ^ v1);
    ulong v4 = 135UL;
    ulong v5 = ln_clmul64_hi(v2, 135UL);
    ulong v6 = (v3 ^ v5);
    return v6;
}

kernel void k0_metal_decl_gf2_kern_f128_reduce_deg191_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_f128_reduce_deg191_hi(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_split_even_odd_bits_u8(ulong v0) {
    ulong v1 = 85UL;
    ulong v2 = (v0 & 85UL);
    ulong v3 = 170UL;
    ulong v4 = (v0 & 170UL);
    ulong v5 = 32UL;
    ulong v6 = (v4 << (32UL & 63UL));
    ulong v7 = (v2 | v6);
    return v7;
}

kernel void k0_metal_decl_gf2_kern_split_even_odd_bits_u8(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_split_even_odd_bits_u8(in0[gid]);
    }
}

inline ulong gf2_kern_delta_swap_lo(ulong v0, ulong v1, ulong v2, ulong v3) {
    ulong v4 = 63UL;
    ulong v5 = (v2 & 63UL);
    ulong v6 = (v0 >> (v5 & 63UL));
    ulong v7 = (v6 ^ v1);
    ulong v8 = (v7 & v3);
    ulong v9 = (v8 << (v5 & 63UL));
    ulong v10 = (v0 ^ v9);
    return v10;
}

kernel void k0_metal_decl_gf2_kern_delta_swap_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device ulong* out [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_delta_swap_lo(in0[gid], in1[gid], in2[gid], in3[gid]);
    }
}

inline ulong gf2_kern_sha256_big_sigma1(ulong v0) {
    ulong v1 = 4294967295UL;
    ulong v2 = (v0 & 4294967295UL);
    ulong v3 = 6UL;
    ulong v4 = (v2 >> (6UL & 63UL));
    ulong v5 = 26UL;
    ulong v6 = (v2 << (26UL & 63UL));
    ulong v7 = (v4 | v6);
    ulong v8 = (v7 & 4294967295UL);
    ulong v9 = 11UL;
    ulong v10 = (v2 >> (11UL & 63UL));
    ulong v11 = 21UL;
    ulong v12 = (v2 << (21UL & 63UL));
    ulong v13 = (v10 | v12);
    ulong v14 = (v13 & 4294967295UL);
    ulong v15 = 25UL;
    ulong v16 = (v2 >> (25UL & 63UL));
    ulong v17 = 7UL;
    ulong v18 = (v2 << (7UL & 63UL));
    ulong v19 = (v16 | v18);
    ulong v20 = (v19 & 4294967295UL);
    ulong v21 = (v8 ^ v14);
    ulong v22 = (v21 ^ v20);
    return v22;
}

kernel void k0_metal_decl_gf2_kern_sha256_big_sigma1(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_sha256_big_sigma1(in0[gid]);
    }
}

inline ulong gf2_kern_gf8_fft_btf_pair(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 255UL;
    ulong v4 = (v2 & 255UL);
    ulong v5 = (v1 & 255UL);
    ulong v6 = ln_clmul64(v4, v5);
    ulong v7 = 65535UL;
    ulong v8 = (v6 & 65535UL);
    ulong v9 = 8UL;
    ulong v10 = (v8 >> (8UL & 63UL));
    ulong v11 = (v8 & 255UL);
    ulong v12 = (v11 ^ v10);
    ulong v13 = 1UL;
    ulong v14 = (v10 << (1UL & 63UL));
    ulong v15 = (v12 ^ v14);
    ulong v16 = 3UL;
    ulong v17 = (v10 << (3UL & 63UL));
    ulong v18 = (v15 ^ v17);
    ulong v19 = 4UL;
    ulong v20 = (v10 << (4UL & 63UL));
    ulong v21 = (v18 ^ v20);
    ulong v22 = (v21 >> (8UL & 63UL));
    ulong v23 = (v21 & 255UL);
    ulong v24 = (v23 ^ v22);
    ulong v25 = (v22 << (1UL & 63UL));
    ulong v26 = (v24 ^ v25);
    ulong v27 = (v22 << (3UL & 63UL));
    ulong v28 = (v26 ^ v27);
    ulong v29 = (v22 << (4UL & 63UL));
    ulong v30 = (v28 ^ v29);
    ulong v31 = (v30 & 255UL);
    ulong v32 = (v0 & 255UL);
    ulong v33 = (v32 ^ v31);
    ulong v34 = (v5 ^ v33);
    ulong v35 = 32UL;
    ulong v36 = (v34 << (32UL & 63UL));
    ulong v37 = (v33 | v36);
    return v37;
}

kernel void k0_metal_decl_gf2_kern_gf8_fft_btf_pair(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_fft_btf_pair(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_sha256_big_sigma0(ulong v0) {
    ulong v1 = 4294967295UL;
    ulong v2 = (v0 & 4294967295UL);
    ulong v3 = 2UL;
    ulong v4 = (v2 >> (2UL & 63UL));
    ulong v5 = 30UL;
    ulong v6 = (v2 << (30UL & 63UL));
    ulong v7 = (v4 | v6);
    ulong v8 = (v7 & 4294967295UL);
    ulong v9 = 13UL;
    ulong v10 = (v2 >> (13UL & 63UL));
    ulong v11 = 19UL;
    ulong v12 = (v2 << (19UL & 63UL));
    ulong v13 = (v10 | v12);
    ulong v14 = (v13 & 4294967295UL);
    ulong v15 = 22UL;
    ulong v16 = (v2 >> (22UL & 63UL));
    ulong v17 = 10UL;
    ulong v18 = (v2 << (10UL & 63UL));
    ulong v19 = (v16 | v18);
    ulong v20 = (v19 & 4294967295UL);
    ulong v21 = (v8 ^ v14);
    ulong v22 = (v21 ^ v20);
    return v22;
}

kernel void k0_metal_decl_gf2_kern_sha256_big_sigma0(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_sha256_big_sigma0(in0[gid]);
    }
}

inline ulong gf2_kern_gf8_subspace_w2(ulong v0) {
    ulong v1 = 255UL;
    ulong v2 = (v0 & 255UL);
    ulong v3 = (v2 & 255UL);
    ulong v4 = ln_clmul64(v3, v3);
    ulong v5 = 65535UL;
    ulong v6 = (v4 & 65535UL);
    ulong v7 = 8UL;
    ulong v8 = (v6 >> (8UL & 63UL));
    ulong v9 = (v6 & 255UL);
    ulong v10 = (v9 ^ v8);
    ulong v11 = 1UL;
    ulong v12 = (v8 << (1UL & 63UL));
    ulong v13 = (v10 ^ v12);
    ulong v14 = 3UL;
    ulong v15 = (v8 << (3UL & 63UL));
    ulong v16 = (v13 ^ v15);
    ulong v17 = 4UL;
    ulong v18 = (v8 << (4UL & 63UL));
    ulong v19 = (v16 ^ v18);
    ulong v20 = (v19 >> (8UL & 63UL));
    ulong v21 = (v19 & 255UL);
    ulong v22 = (v21 ^ v20);
    ulong v23 = (v20 << (1UL & 63UL));
    ulong v24 = (v22 ^ v23);
    ulong v25 = (v20 << (3UL & 63UL));
    ulong v26 = (v24 ^ v25);
    ulong v27 = (v20 << (4UL & 63UL));
    ulong v28 = (v26 ^ v27);
    ulong v29 = (v28 & 255UL);
    ulong v30 = (v29 ^ v2);
    ulong v31 = (v30 & 255UL);
    ulong v32 = 6UL;
    ulong v33 = (v30 ^ 6UL);
    ulong v34 = (v33 & 255UL);
    ulong v35 = ln_clmul64(v31, v34);
    ulong v36 = (v35 & 65535UL);
    ulong v37 = (v36 >> (8UL & 63UL));
    ulong v38 = (v36 & 255UL);
    ulong v39 = (v38 ^ v37);
    ulong v40 = (v37 << (1UL & 63UL));
    ulong v41 = (v39 ^ v40);
    ulong v42 = (v37 << (3UL & 63UL));
    ulong v43 = (v41 ^ v42);
    ulong v44 = (v37 << (4UL & 63UL));
    ulong v45 = (v43 ^ v44);
    ulong v46 = (v45 >> (8UL & 63UL));
    ulong v47 = (v45 & 255UL);
    ulong v48 = (v47 ^ v46);
    ulong v49 = (v46 << (1UL & 63UL));
    ulong v50 = (v48 ^ v49);
    ulong v51 = (v46 << (3UL & 63UL));
    ulong v52 = (v50 ^ v51);
    ulong v53 = (v46 << (4UL & 63UL));
    ulong v54 = (v52 ^ v53);
    ulong v55 = (v54 & 255UL);
    return v55;
}

kernel void k0_metal_decl_gf2_kern_gf8_subspace_w2(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_subspace_w2(in0[gid]);
    }
}

inline ulong l_Gf2Kernel_IR_clmul64(ulong v0, ulong v1) {
    ulong v2 = ln_clmul64(v0, v1);
    return v2;
}

kernel void k0_metal_decl_l_Gf2Kernel_IR_clmul64(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = l_Gf2Kernel_IR_clmul64(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_urm_pack_c_pair_bank1(ulong v0, ulong v1) {
    ulong v2 = 12297829382473034410UL;
    ulong v3 = (v0 & 12297829382473034410UL);
    ulong v4 = 1UL;
    ulong v5 = (v3 >> (1UL & 63UL));
    ulong v6 = (v1 & 12297829382473034410UL);
    ulong v7 = (v5 | v6);
    return v7;
}

kernel void k0_metal_decl_gf2_kern_urm_pack_c_pair_bank1(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_urm_pack_c_pair_bank1(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_f128_karatsuba_unred_r1(ulong v0, ulong v1, ulong v2, ulong v3) {
    ulong v4 = (v3 ^ v0);
    ulong v5 = (v4 ^ v2);
    ulong v6 = (v1 ^ v5);
    return v6;
}

kernel void k0_metal_decl_gf2_kern_f128_karatsuba_unred_r1(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device ulong* out [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_f128_karatsuba_unred_r1(in0[gid], in1[gid], in2[gid], in3[gid]);
    }
}

inline ulong gf2_kern_sha256_ch_and_out(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 4294967295UL;
    ulong v4 = (v1 ^ v2);
    ulong v5 = (v0 & v4);
    ulong v6 = (v5 & 4294967295UL);
    ulong v7 = (v2 & 4294967295UL);
    ulong v8 = (v6 ^ v7);
    ulong v9 = 32UL;
    ulong v10 = (v8 << (32UL & 63UL));
    ulong v11 = (v6 | v10);
    return v11;
}

kernel void k0_metal_decl_gf2_kern_sha256_ch_and_out(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_sha256_ch_and_out(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_urm_pack_c_pair_bank0(ulong v0, ulong v1) {
    ulong v2 = 6148914691236517205UL;
    ulong v3 = (v0 & 6148914691236517205UL);
    ulong v4 = (v1 & 6148914691236517205UL);
    ulong v5 = 1UL;
    ulong v6 = (v4 << (1UL & 63UL));
    ulong v7 = (v3 | v6);
    return v7;
}

kernel void k0_metal_decl_gf2_kern_urm_pack_c_pair_bank0(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_urm_pack_c_pair_bank0(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_gf8_reduce_deg21_pair_x4(ulong v0, ulong v1) {
    ulong v2 = 281470681808895UL;
    ulong v3 = (v0 & 281470681808895UL);
    ulong v4 = (v1 & 281470681808895UL);
    ulong v5 = 16UL;
    ulong v6 = (v4 << (16UL & 63UL));
    ulong v7 = (v3 ^ v6);
    ulong v8 = (v0 >> (16UL & 63UL));
    ulong v9 = 270582939711UL;
    ulong v10 = (v8 & 270582939711UL);
    ulong v11 = 17732923536900096UL;
    ulong v12 = (v1 & 17732923536900096UL);
    ulong v13 = (v10 ^ v12);
    ulong v14 = 6UL;
    ulong v15 = (v13 << (6UL & 63UL));
    ulong v16 = (v7 ^ v15);
    ulong v17 = 4UL;
    ulong v18 = (v13 << (4UL & 63UL));
    ulong v19 = (v16 ^ v18);
    ulong v20 = 3UL;
    ulong v21 = (v13 << (3UL & 63UL));
    ulong v22 = (v19 ^ v21);
    ulong v23 = 2UL;
    ulong v24 = (v13 << (2UL & 63UL));
    ulong v25 = (v22 ^ v24);
    ulong v26 = 1UL;
    ulong v27 = (v13 << (1UL & 63UL));
    ulong v28 = (v25 ^ v27);
    ulong v29 = 71777214294589695UL;
    ulong v30 = 8UL;
    ulong v31 = (v28 >> (8UL & 63UL));
    ulong v32 = (v31 & 71777214294589695UL);
    ulong v33 = (v28 & 71777214294589695UL);
    ulong v34 = (v33 ^ v32);
    ulong v35 = (v32 << (1UL & 63UL));
    ulong v36 = (v34 ^ v35);
    ulong v37 = (v32 << (3UL & 63UL));
    ulong v38 = (v36 ^ v37);
    ulong v39 = (v32 << (4UL & 63UL));
    ulong v40 = (v38 ^ v39);
    ulong v41 = (v40 >> (8UL & 63UL));
    ulong v42 = (v41 & 71777214294589695UL);
    ulong v43 = (v40 & 71777214294589695UL);
    ulong v44 = (v43 ^ v42);
    ulong v45 = (v42 << (1UL & 63UL));
    ulong v46 = (v44 ^ v45);
    ulong v47 = (v42 << (3UL & 63UL));
    ulong v48 = (v46 ^ v47);
    ulong v49 = (v42 << (4UL & 63UL));
    ulong v50 = (v48 ^ v49);
    ulong v51 = (v50 & 71777214294589695UL);
    return v51;
}

kernel void k0_metal_decl_gf2_kern_gf8_reduce_deg21_pair_x4(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_reduce_deg21_pair_x4(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_delta_swap_hi(ulong v0, ulong v1, ulong v2, ulong v3) {
    ulong v4 = 63UL;
    ulong v5 = (v2 & 63UL);
    ulong v6 = (v0 >> (v5 & 63UL));
    ulong v7 = (v6 ^ v1);
    ulong v8 = (v7 & v3);
    ulong v9 = (v1 ^ v8);
    return v9;
}

kernel void k0_metal_decl_gf2_kern_delta_swap_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device ulong* out [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_delta_swap_hi(in0[gid], in1[gid], in2[gid], in3[gid]);
    }
}

inline ulong gf2_kern_inv_ntt_cyclic_offset_lo(ulong v0) {
    ulong v1 = 2UL;
    ulong v2 = (v0 & 2UL);
    ulong v3 = 3UL;
    ulong v4 = (v2 << (3UL & 63UL));
    ulong v5 = 4UL;
    ulong v6 = (v0 & 4UL);
    ulong v7 = 12UL;
    ulong v8 = (v6 * 12UL);
    ulong v9 = (v4 + v8);
    return v9;
}

kernel void k0_metal_decl_gf2_kern_inv_ntt_cyclic_offset_lo(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_inv_ntt_cyclic_offset_lo(in0[gid]);
    }
}

inline ulong gf2_kern_ntt_dual_block_src_offset(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4) {
    ulong v5 = (v3 * v0);
    ulong v6 = (v5 + v2);
    ulong v7 = (v6 * v1);
    ulong v8 = (v7 + v4);
    return v8;
}

kernel void k0_metal_decl_gf2_kern_ntt_dual_block_src_offset(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device ulong* out [[buffer(5)]],
    constant ulong& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ntt_dual_block_src_offset(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid]);
    }
}

inline ulong gf2_kern_pack8_bool_bytes(ulong v0) {
    ulong v1 = 72340172838076673UL;
    ulong v2 = (v0 & 72340172838076673UL);
    ulong v3 = 72624976668147840UL;
    ulong v4 = (v2 * 72624976668147840UL);
    ulong v5 = 56UL;
    ulong v6 = (v4 >> (56UL & 63UL));
    return v6;
}

kernel void k0_metal_decl_gf2_kern_pack8_bool_bytes(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_pack8_bool_bytes(in0[gid]);
    }
}

inline ulong gf2_kern_gf8_from_composite(ulong v0) {
    ulong v1 = 0UL;
    ulong v2 = 1UL;
    ulong v3 = (v0 & 1UL);
    ulong v4 = (0UL - v3);
    ulong v5 = (v0 >> (1UL & 63UL));
    ulong v6 = (v5 & 1UL);
    ulong v7 = (0UL - v6);
    ulong v8 = 2UL;
    ulong v9 = (v0 >> (2UL & 63UL));
    ulong v10 = (v9 & 1UL);
    ulong v11 = (0UL - v10);
    ulong v12 = 3UL;
    ulong v13 = (v0 >> (3UL & 63UL));
    ulong v14 = (v13 & 1UL);
    ulong v15 = (0UL - v14);
    ulong v16 = 4UL;
    ulong v17 = (v0 >> (4UL & 63UL));
    ulong v18 = (v17 & 1UL);
    ulong v19 = (0UL - v18);
    ulong v20 = 5UL;
    ulong v21 = (v0 >> (5UL & 63UL));
    ulong v22 = (v21 & 1UL);
    ulong v23 = (0UL - v22);
    ulong v24 = 6UL;
    ulong v25 = (v0 >> (6UL & 63UL));
    ulong v26 = (v25 & 1UL);
    ulong v27 = (0UL - v26);
    ulong v28 = 7UL;
    ulong v29 = (v0 >> (7UL & 63UL));
    ulong v30 = (v29 & 1UL);
    ulong v31 = (0UL - v30);
    ulong v32 = (v4 & 1UL);
    ulong v33 = 225UL;
    ulong v34 = (v7 & 225UL);
    ulong v35 = (v32 ^ v34);
    ulong v36 = 92UL;
    ulong v37 = (v11 & 92UL);
    ulong v38 = (v35 ^ v37);
    ulong v39 = 12UL;
    ulong v40 = (v15 & 12UL);
    ulong v41 = (v38 ^ v40);
    ulong v42 = 18UL;
    ulong v43 = (v19 & 18UL);
    ulong v44 = (v41 ^ v43);
    ulong v45 = 75UL;
    ulong v46 = (v23 & 75UL);
    ulong v47 = (v44 ^ v46);
    ulong v48 = 15UL;
    ulong v49 = (v27 & 15UL);
    ulong v50 = (v47 ^ v49);
    ulong v51 = 216UL;
    ulong v52 = (v31 & 216UL);
    ulong v53 = (v50 ^ v52);
    return v53;
}

kernel void k0_metal_decl_gf2_kern_gf8_from_composite(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_from_composite(in0[gid]);
    }
}

inline ulong gf2_kern_gf8_ifft_btf_pair(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 255UL;
    ulong v4 = (v0 & 255UL);
    ulong v5 = (v1 & 255UL);
    ulong v6 = (v5 ^ v4);
    ulong v7 = (v2 & 255UL);
    ulong v8 = (v6 & 255UL);
    ulong v9 = ln_clmul64(v7, v8);
    ulong v10 = 65535UL;
    ulong v11 = (v9 & 65535UL);
    ulong v12 = 8UL;
    ulong v13 = (v11 >> (8UL & 63UL));
    ulong v14 = (v11 & 255UL);
    ulong v15 = (v14 ^ v13);
    ulong v16 = 1UL;
    ulong v17 = (v13 << (1UL & 63UL));
    ulong v18 = (v15 ^ v17);
    ulong v19 = 3UL;
    ulong v20 = (v13 << (3UL & 63UL));
    ulong v21 = (v18 ^ v20);
    ulong v22 = 4UL;
    ulong v23 = (v13 << (4UL & 63UL));
    ulong v24 = (v21 ^ v23);
    ulong v25 = (v24 >> (8UL & 63UL));
    ulong v26 = (v24 & 255UL);
    ulong v27 = (v26 ^ v25);
    ulong v28 = (v25 << (1UL & 63UL));
    ulong v29 = (v27 ^ v28);
    ulong v30 = (v25 << (3UL & 63UL));
    ulong v31 = (v29 ^ v30);
    ulong v32 = (v25 << (4UL & 63UL));
    ulong v33 = (v31 ^ v32);
    ulong v34 = (v33 & 255UL);
    ulong v35 = (v4 ^ v34);
    ulong v36 = 32UL;
    ulong v37 = (v6 << (32UL & 63UL));
    ulong v38 = (v35 | v37);
    return v38;
}

kernel void k0_metal_decl_gf2_kern_gf8_ifft_btf_pair(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_ifft_btf_pair(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_blake3_add_right(ulong v0, ulong v1) {
    ulong v2 = 4294967295UL;
    ulong v3 = 2147483647UL;
    ulong v4 = (v0 & 4294967295UL);
    ulong v5 = (v1 & 4294967295UL);
    ulong v6 = (v4 + v5);
    ulong v7 = (v6 & 4294967295UL);
    ulong v8 = (v7 ^ v4);
    ulong v9 = (v8 ^ v5);
    ulong v10 = (v5 ^ v9);
    ulong v11 = (v10 & 2147483647UL);
    return v11;
}

kernel void k0_metal_decl_gf2_kern_blake3_add_right(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_blake3_add_right(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_gf8_ntt_next_s(ulong v0, ulong v1) {
    ulong v2 = 255UL;
    ulong v3 = (v0 & 255UL);
    ulong v4 = (v0 ^ v1);
    ulong v5 = (v4 & 255UL);
    ulong v6 = ln_clmul64(v3, v5);
    ulong v7 = 65535UL;
    ulong v8 = (v6 & 65535UL);
    ulong v9 = 8UL;
    ulong v10 = (v8 >> (8UL & 63UL));
    ulong v11 = (v8 & 255UL);
    ulong v12 = (v11 ^ v10);
    ulong v13 = 1UL;
    ulong v14 = (v10 << (1UL & 63UL));
    ulong v15 = (v12 ^ v14);
    ulong v16 = 3UL;
    ulong v17 = (v10 << (3UL & 63UL));
    ulong v18 = (v15 ^ v17);
    ulong v19 = 4UL;
    ulong v20 = (v10 << (4UL & 63UL));
    ulong v21 = (v18 ^ v20);
    ulong v22 = (v21 >> (8UL & 63UL));
    ulong v23 = (v21 & 255UL);
    ulong v24 = (v23 ^ v22);
    ulong v25 = (v22 << (1UL & 63UL));
    ulong v26 = (v24 ^ v25);
    ulong v27 = (v22 << (3UL & 63UL));
    ulong v28 = (v26 ^ v27);
    ulong v29 = (v22 << (4UL & 63UL));
    ulong v30 = (v28 ^ v29);
    ulong v31 = (v30 & 255UL);
    return v31;
}

kernel void k0_metal_decl_gf2_kern_gf8_ntt_next_s(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_ntt_next_s(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_ntt_radix4_comb_b(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4) {
    ulong v5 = (v1 ^ v3);
    ulong v6 = (v5 ^ v0);
    ulong v7 = (v6 ^ v2);
    ulong v8 = (v7 ^ v4);
    return v8;
}

kernel void k0_metal_decl_gf2_kern_ntt_radix4_comb_b(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device ulong* out [[buffer(5)]],
    constant ulong& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ntt_radix4_comb_b(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid]);
    }
}

inline ulong gf2_kern_bitrecord_flush_word(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 63UL;
    ulong v4 = (v2 & 63UL);
    ulong v5 = (v0 << (v4 & 63UL));
    ulong v6 = (v5 | v1);
    return v6;
}

kernel void k0_metal_decl_gf2_kern_bitrecord_flush_word(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_bitrecord_flush_word(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_or_u32_lo(ulong v0, ulong v1) {
    ulong v2 = 4294967295UL;
    ulong v3 = (v0 & 4294967295UL);
    ulong v4 = 63UL;
    ulong v5 = (v1 & 63UL);
    ulong v6 = (v3 << (v5 & 63UL));
    return v6;
}

kernel void k0_metal_decl_gf2_kern_or_u32_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_or_u32_lo(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_eq_bit_mul_lo(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4) {
    ulong v5 = 1UL;
    ulong v6 = (v0 & 1UL);
    ulong v7 = (v6 ^ 1UL);
    ulong v8 = (v3 ^ v7);
    ulong v1013 = ln_clmul64(v1, v8);
    ulong v1014 = ln_clmul64_hi(v1, v4);
    ulong v1015 = ln_clmul64_hi(v2, v8);
    ulong v1016 = ln_clmul64(v2, v4);
    ulong v1017 = ln_clmul64_hi(v2, v4);
    ulong v1018 = 135UL;
    ulong v1019 = ln_clmul64_hi(v1017, 135UL);
    ulong v1020 = (v1014 ^ v1015);
    ulong v1021 = (v1020 ^ v1019);
    ulong v1022 = (v1016 ^ v1021);
    ulong v1023 = ln_clmul64(v1022, 135UL);
    ulong v1024 = (v1013 ^ v1023);
    ulong v9 = v1024;
    return v9;
}

kernel void k0_metal_decl_gf2_kern_eq_bit_mul_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device ulong* out [[buffer(5)]],
    constant ulong& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_eq_bit_mul_lo(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid]);
    }
}

inline ulong gf2_kern_f128_pre_reduce_const_hi(ulong v0, ulong v1) {
    ulong v2 = 135UL;
    ulong v3 = ln_clmul64_hi(v1, 135UL);
    ulong v4 = (v0 ^ v3);
    return v4;
}

kernel void k0_metal_decl_gf2_kern_f128_pre_reduce_const_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_f128_pre_reduce_const_hi(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_unpack_even_bytes_u32(ulong v0) {
    ulong v1 = 4294967295UL;
    ulong v2 = (v0 & 4294967295UL);
    ulong v3 = 16UL;
    ulong v4 = (v2 << (16UL & 63UL));
    ulong v5 = (v2 | v4);
    ulong v6 = 281470681808895UL;
    ulong v7 = (v5 & 281470681808895UL);
    ulong v8 = 8UL;
    ulong v9 = (v7 << (8UL & 63UL));
    ulong v10 = (v7 | v9);
    ulong v11 = 71777214294589695UL;
    ulong v12 = (v10 & 71777214294589695UL);
    return v12;
}

kernel void k0_metal_decl_gf2_kern_unpack_even_bytes_u32(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_unpack_even_bytes_u32(in0[gid]);
    }
}

inline ulong gf2_kern_sha256_maj_and_out(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 4294967295UL;
    ulong v4 = (v0 ^ v1);
    ulong v5 = (v0 ^ v2);
    ulong v6 = (v4 & v5);
    ulong v7 = (v6 & 4294967295UL);
    ulong v8 = (v0 & 4294967295UL);
    ulong v9 = (v7 ^ v8);
    ulong v10 = 32UL;
    ulong v11 = (v9 << (32UL & 63UL));
    ulong v12 = (v7 | v11);
    return v12;
}

kernel void k0_metal_decl_gf2_kern_sha256_maj_and_out(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_sha256_maj_and_out(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_gf8_mul_x4(ulong v0, ulong v1) {
    ulong v2 = 16711935UL;
    ulong v3 = (v0 & 16711935UL);
    ulong v4 = (v1 & 16711935UL);
    ulong v5 = 8UL;
    ulong v6 = (v0 >> (8UL & 63UL));
    ulong v7 = (v6 & 16711935UL);
    ulong v8 = (v1 >> (8UL & 63UL));
    ulong v9 = (v8 & 16711935UL);
    ulong v10 = ln_clmul64(v3, v4);
    ulong v11 = 281470681808895UL;
    ulong v12 = (v10 & 281470681808895UL);
    ulong v13 = ln_clmul64(v7, v9);
    ulong v14 = (v13 & 281470681808895UL);
    ulong v15 = 16UL;
    ulong v16 = (v14 << (16UL & 63UL));
    ulong v17 = (v12 | v16);
    ulong v18 = 71777214294589695UL;
    ulong v19 = (v17 >> (8UL & 63UL));
    ulong v20 = (v19 & 71777214294589695UL);
    ulong v21 = (v17 & 71777214294589695UL);
    ulong v22 = (v21 ^ v20);
    ulong v23 = 1UL;
    ulong v24 = (v20 << (1UL & 63UL));
    ulong v25 = (v22 ^ v24);
    ulong v26 = 3UL;
    ulong v27 = (v20 << (3UL & 63UL));
    ulong v28 = (v25 ^ v27);
    ulong v29 = 4UL;
    ulong v30 = (v20 << (4UL & 63UL));
    ulong v31 = (v28 ^ v30);
    ulong v32 = (v31 >> (8UL & 63UL));
    ulong v33 = (v32 & 71777214294589695UL);
    ulong v34 = (v31 & 71777214294589695UL);
    ulong v35 = (v34 ^ v33);
    ulong v36 = (v33 << (1UL & 63UL));
    ulong v37 = (v35 ^ v36);
    ulong v38 = (v33 << (3UL & 63UL));
    ulong v39 = (v37 ^ v38);
    ulong v40 = (v33 << (4UL & 63UL));
    ulong v41 = (v39 ^ v40);
    ulong v42 = (v41 & 71777214294589695UL);
    return v42;
}

kernel void k0_metal_decl_gf2_kern_gf8_mul_x4(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_mul_x4(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_pack_even_bytes_x8(ulong v0, ulong v1) {
    ulong v2 = 71777214294589695UL;
    ulong v3 = (v0 & 71777214294589695UL);
    ulong v4 = 8UL;
    ulong v5 = (v3 >> (8UL & 63UL));
    ulong v6 = (v3 | v5);
    ulong v7 = 281470681808895UL;
    ulong v8 = (v6 & 281470681808895UL);
    ulong v9 = 16UL;
    ulong v10 = (v8 >> (16UL & 63UL));
    ulong v11 = (v8 | v10);
    ulong v12 = 4294967295UL;
    ulong v13 = (v11 & 4294967295UL);
    ulong v14 = (v1 & 71777214294589695UL);
    ulong v15 = (v14 >> (8UL & 63UL));
    ulong v16 = (v14 | v15);
    ulong v17 = (v16 & 281470681808895UL);
    ulong v18 = (v17 >> (16UL & 63UL));
    ulong v19 = (v17 | v18);
    ulong v20 = 32UL;
    ulong v21 = (v19 << (32UL & 63UL));
    ulong v22 = (v13 | v21);
    return v22;
}

kernel void k0_metal_decl_gf2_kern_pack_even_bytes_x8(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_pack_even_bytes_x8(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_blake3_xor_rotr32(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 4294967295UL;
    ulong v4 = (v0 ^ v1);
    ulong v5 = (v4 & 4294967295UL);
    ulong v6 = 31UL;
    ulong v7 = (v2 & 31UL);
    ulong v8 = (v5 >> (v7 & 63UL));
    ulong v9 = 32UL;
    ulong v10 = (32UL - v7);
    ulong v11 = (v10 & 31UL);
    ulong v12 = (v5 << (v11 & 63UL));
    ulong v13 = (v8 | v12);
    ulong v14 = (v13 & 4294967295UL);
    return v14;
}

kernel void k0_metal_decl_gf2_kern_blake3_xor_rotr32(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_blake3_xor_rotr32(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_ntt_dual_block_dst_offset(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4, ulong v5) {
    ulong v6 = 4UL;
    ulong v7 = (v0 << (4UL & 63UL));
    ulong v8 = (v7 * v1);
    ulong v9 = (v2 * v8);
    ulong v10 = (v4 * v0);
    ulong v11 = (v10 + v3);
    ulong v12 = (v11 * v1);
    ulong v13 = (v12 + v5);
    ulong v14 = (v9 + v13);
    return v14;
}

kernel void k0_metal_decl_gf2_kern_ntt_dual_block_dst_offset(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device const ulong* in5 [[buffer(5)]],
    device ulong* out [[buffer(6)]],
    constant ulong& n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ntt_dual_block_dst_offset(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid], in5[gid]);
    }
}

inline ulong gf2_kern_gf8_reduce(ulong v0) {
    ulong v1 = 65535UL;
    ulong v2 = (v0 & 65535UL);
    ulong v3 = 8UL;
    ulong v4 = (v2 >> (8UL & 63UL));
    ulong v5 = 255UL;
    ulong v6 = (v2 & 255UL);
    ulong v7 = (v6 ^ v4);
    ulong v8 = 1UL;
    ulong v9 = (v4 << (1UL & 63UL));
    ulong v10 = (v7 ^ v9);
    ulong v11 = 3UL;
    ulong v12 = (v4 << (3UL & 63UL));
    ulong v13 = (v10 ^ v12);
    ulong v14 = 4UL;
    ulong v15 = (v4 << (4UL & 63UL));
    ulong v16 = (v13 ^ v15);
    ulong v17 = (v16 >> (8UL & 63UL));
    ulong v18 = (v16 & 255UL);
    ulong v19 = (v18 ^ v17);
    ulong v20 = (v17 << (1UL & 63UL));
    ulong v21 = (v19 ^ v20);
    ulong v22 = (v17 << (3UL & 63UL));
    ulong v23 = (v21 ^ v22);
    ulong v24 = (v17 << (4UL & 63UL));
    ulong v25 = (v23 ^ v24);
    ulong v26 = (v25 & 255UL);
    return v26;
}

kernel void k0_metal_decl_gf2_kern_gf8_reduce(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_reduce(in0[gid]);
    }
}

inline ulong gf2_kern_f128_karatsuba_unred_r2(ulong v0, ulong v1, ulong v2, ulong v3) {
    ulong v4 = (v3 ^ v0);
    ulong v5 = (v4 ^ v2);
    ulong v6 = (v1 ^ v5);
    return v6;
}

kernel void k0_metal_decl_gf2_kern_f128_karatsuba_unred_r2(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device ulong* out [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_f128_karatsuba_unred_r2(in0[gid], in1[gid], in2[gid], in3[gid]);
    }
}

inline ulong gf2_kern_blake3_chunk_flags(ulong v0, ulong v1) {
    ulong v10 = 0UL;
    ulong v11 = ((v0 == 0UL) ? 1UL : 0UL);
    ulong v12 = ((v11 == 0UL) ? 1UL : 0UL);
    if (v12 != 0UL) {
        ulong v3 = 0UL;
        ulong v4 = 1UL;
        ulong v5 = (v0 + 1UL);
        ulong v6 = ((v5 == v1) ? 1UL : 0UL);
        ulong v7 = ((v6 == 0UL) ? 1UL : 0UL);
        if (v7 != 0UL) {
            return v3;
        } else {
            ulong v8 = 2UL;
            ulong v9 = 2UL;
            return v9;
        }
    } else {
        ulong v13 = 1UL;
        return v13;
    }
}

kernel void k0_metal_decl_gf2_kern_blake3_chunk_flags(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_blake3_chunk_flags(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_blake3_finalize_word_pair(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 4294967295UL;
    ulong v4 = (v0 ^ v1);
    ulong v5 = (v4 & 4294967295UL);
    ulong v6 = (v1 ^ v2);
    ulong v7 = (v6 & 4294967295UL);
    ulong v8 = 32UL;
    ulong v9 = (v7 << (32UL & 63UL));
    ulong v10 = (v5 | v9);
    return v10;
}

kernel void k0_metal_decl_gf2_kern_blake3_finalize_word_pair(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_blake3_finalize_word_pair(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_unpack8_bool_bytes(ulong v0) {
    ulong v1 = 127UL;
    ulong v2 = (v0 & 127UL);
    ulong v3 = 4432676798593UL;
    ulong v4 = (v2 * 4432676798593UL);
    ulong v5 = 128UL;
    ulong v6 = (v0 & 128UL);
    ulong v7 = 49UL;
    ulong v8 = (v6 << (49UL & 63UL));
    ulong v9 = (v4 | v8);
    ulong v10 = 72340172838076673UL;
    ulong v11 = (v9 & 72340172838076673UL);
    return v11;
}

kernel void k0_metal_decl_gf2_kern_unpack8_bool_bytes(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_unpack8_bool_bytes(in0[gid]);
    }
}

inline ulong gf2_kern_ghash_mul_unred_r1(ulong v0, ulong v1, ulong v2, ulong v3) {
    ulong v4 = ln_clmul64_hi(v0, v2);
    ulong v5 = ln_clmul64(v0, v3);
    ulong v6 = ln_clmul64(v1, v2);
    ulong v7 = (v5 ^ v6);
    ulong v8 = (v4 ^ v7);
    return v8;
}

kernel void k0_metal_decl_gf2_kern_ghash_mul_unred_r1(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device ulong* out [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ghash_mul_unred_r1(in0[gid], in1[gid], in2[gid], in3[gid]);
    }
}

inline ulong gf2_kern_mul_by_x_lo(ulong v0, ulong v1) {
    ulong v2 = 63UL;
    ulong v3 = (v1 >> (63UL & 63UL));
    ulong v4 = 0UL;
    ulong v5 = (0UL - v3);
    ulong v6 = 1UL;
    ulong v7 = (v0 << (1UL & 63UL));
    ulong v8 = 135UL;
    ulong v9 = (135UL & v5);
    ulong v10 = (v7 ^ v9);
    return v10;
}

kernel void k0_metal_decl_gf2_kern_mul_by_x_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_mul_by_x_lo(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_monomial_basis128_lo(ulong v0) {
    ulong v1 = 6UL;
    ulong v2 = (v0 >> (6UL & 63UL));
    ulong v3 = 1UL;
    ulong v4 = (v2 & 1UL);
    ulong v5 = (v4 ^ 1UL);
    ulong v6 = 63UL;
    ulong v7 = (v0 & 63UL);
    ulong v8 = (v5 << (v7 & 63UL));
    return v8;
}

kernel void k0_metal_decl_gf2_kern_monomial_basis128_lo(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_monomial_basis128_lo(in0[gid]);
    }
}

inline ulong gf2_kern_gf8_mul_by_x(ulong v0) {
    ulong v1 = 255UL;
    ulong v2 = (v0 & 255UL);
    ulong v3 = 7UL;
    ulong v4 = (v2 >> (7UL & 63UL));
    ulong v5 = 1UL;
    ulong v6 = (v4 & 1UL);
    ulong v7 = 0UL;
    ulong v8 = (0UL - v6);
    ulong v9 = (v2 << (1UL & 63UL));
    ulong v10 = 27UL;
    ulong v11 = (v8 & 27UL);
    ulong v12 = (v9 ^ v11);
    ulong v13 = (v12 & 255UL);
    return v13;
}

kernel void k0_metal_decl_gf2_kern_gf8_mul_by_x(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_mul_by_x(in0[gid]);
    }
}

inline ulong gf2_kern_isolate_lowest_one(ulong v0) {
    ulong v1 = 0UL;
    ulong v2 = (0UL - v0);
    ulong v3 = (v0 & v2);
    return v3;
}

kernel void k0_metal_decl_gf2_kern_isolate_lowest_one(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_isolate_lowest_one(in0[gid]);
    }
}

inline ulong gf2_kern_sha256_small_sigma0(ulong v0) {
    ulong v1 = 4294967295UL;
    ulong v2 = (v0 & 4294967295UL);
    ulong v3 = 7UL;
    ulong v4 = (v2 >> (7UL & 63UL));
    ulong v5 = 25UL;
    ulong v6 = (v2 << (25UL & 63UL));
    ulong v7 = (v4 | v6);
    ulong v8 = (v7 & 4294967295UL);
    ulong v9 = 18UL;
    ulong v10 = (v2 >> (18UL & 63UL));
    ulong v11 = 14UL;
    ulong v12 = (v2 << (14UL & 63UL));
    ulong v13 = (v10 | v12);
    ulong v14 = (v13 & 4294967295UL);
    ulong v15 = 3UL;
    ulong v16 = (v2 >> (3UL & 63UL));
    ulong v17 = (v8 ^ v14);
    ulong v18 = (v17 ^ v16);
    return v18;
}

kernel void k0_metal_decl_gf2_kern_sha256_small_sigma0(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_sha256_small_sigma0(in0[gid]);
    }
}

inline ulong gf2_kern_jagged_transition(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4) {
    ulong v5 = 1UL;
    ulong v6 = (v0 & 1UL);
    ulong v7 = (v1 & 1UL);
    ulong v8 = (v2 & 1UL);
    ulong v9 = (v3 & 1UL);
    ulong v10 = (v4 & 1UL);
    ulong v11 = (v4 >> (1UL & 63UL));
    ulong v12 = (v11 & 1UL);
    ulong v13 = (v6 ^ v10);
    ulong v14 = (v13 ^ v8);
    ulong v15 = (v14 ^ v7);
    ulong v16 = (v6 & v10);
    ulong v17 = (v6 & v8);
    ulong v18 = (v16 | v17);
    ulong v19 = (v10 & v8);
    ulong v20 = (v18 | v19);
    ulong v21 = (v7 ^ v9);
    ulong v22 = (v21 & v9);
    ulong v23 = (v21 ^ 1UL);
    ulong v24 = (v23 & v12);
    ulong v25 = (v22 | v24);
    ulong v26 = (v25 << (1UL & 63UL));
    ulong v27 = (v20 | v26);
    ulong v28 = (v15 - 1UL);
    ulong v29 = (v27 & v28);
    ulong v30 = 2UL;
    ulong v31 = (v15 << (2UL & 63UL));
    ulong v32 = (v29 | v31);
    return v32;
}

kernel void k0_metal_decl_gf2_kern_jagged_transition(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device ulong* out [[buffer(5)]],
    constant ulong& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_jagged_transition(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid]);
    }
}

inline ulong gf2_kern_keccak_chi(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 18446744073709551615UL;
    ulong v4 = (v1 ^ 18446744073709551615UL);
    ulong v5 = (v4 & v2);
    ulong v6 = (v0 ^ v5);
    return v6;
}

kernel void k0_metal_decl_gf2_kern_keccak_chi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_keccak_chi(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_gf16_log_mod15_signed_reduce(ulong v0) {
    ulong v1 = 255UL;
    ulong v2 = (v0 & 255UL);
    ulong v3 = 15UL;
    ulong v4 = (15UL - v2);
    ulong v5 = 63UL;
    ulong v6 = (v4 >> (63UL & 63UL));
    ulong v7 = 7UL;
    ulong v8 = (v2 >> (7UL & 63UL));
    ulong v9 = 1UL;
    ulong v10 = (v8 ^ 1UL);
    ulong v11 = 0UL;
    ulong v12 = (v6 & v10);
    ulong v13 = (0UL - v12);
    ulong v14 = (v2 - v13);
    ulong v15 = (v14 & 255UL);
    return v15;
}

kernel void k0_metal_decl_gf2_kern_gf16_log_mod15_signed_reduce(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf16_log_mod15_signed_reduce(in0[gid]);
    }
}

inline ulong gf2_kern_keccak_chi_and(ulong v0, ulong v1) {
    ulong v2 = 18446744073709551615UL;
    ulong v3 = (v0 ^ 18446744073709551615UL);
    ulong v4 = (v3 & v1);
    return v4;
}

kernel void k0_metal_decl_gf2_kern_keccak_chi_and(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_keccak_chi_and(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_pack_even_bytes_u32(ulong v0) {
    ulong v1 = 71777214294589695UL;
    ulong v2 = (v0 & 71777214294589695UL);
    ulong v3 = 8UL;
    ulong v4 = (v2 >> (8UL & 63UL));
    ulong v5 = (v2 | v4);
    ulong v6 = 281470681808895UL;
    ulong v7 = (v5 & 281470681808895UL);
    ulong v8 = 16UL;
    ulong v9 = (v7 >> (16UL & 63UL));
    ulong v10 = (v7 | v9);
    ulong v11 = 4294967295UL;
    ulong v12 = (v10 & 4294967295UL);
    return v12;
}

kernel void k0_metal_decl_gf2_kern_pack_even_bytes_u32(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_pack_even_bytes_u32(in0[gid]);
    }
}

inline ulong gf2_kern_keccak_theta_d(ulong v0, ulong v1) {
    ulong v2 = 1UL;
    ulong v3 = (v1 << (1UL & 63UL));
    ulong v4 = 63UL;
    ulong v5 = (v1 >> (63UL & 63UL));
    ulong v6 = (v3 | v5);
    ulong v7 = (v0 ^ v6);
    return v7;
}

kernel void k0_metal_decl_gf2_kern_keccak_theta_d(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_keccak_theta_d(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_gf8_square(ulong v0) {
    ulong v1 = 255UL;
    ulong v2 = (v0 & 255UL);
    ulong v3 = 4UL;
    ulong v4 = (v2 << (4UL & 63UL));
    ulong v5 = (v2 ^ v4);
    ulong v6 = 3855UL;
    ulong v7 = (v5 & 3855UL);
    ulong v8 = 2UL;
    ulong v9 = (v7 << (2UL & 63UL));
    ulong v10 = (v7 ^ v9);
    ulong v11 = 13107UL;
    ulong v12 = (v10 & 13107UL);
    ulong v13 = 1UL;
    ulong v14 = (v12 << (1UL & 63UL));
    ulong v15 = (v12 ^ v14);
    ulong v16 = 21845UL;
    ulong v17 = (v15 & 21845UL);
    ulong v18 = 8UL;
    ulong v19 = (v17 >> (8UL & 63UL));
    ulong v20 = (v17 & 255UL);
    ulong v21 = (v20 ^ v19);
    ulong v22 = (v19 << (1UL & 63UL));
    ulong v23 = (v21 ^ v22);
    ulong v24 = 3UL;
    ulong v25 = (v19 << (3UL & 63UL));
    ulong v26 = (v23 ^ v25);
    ulong v27 = (v19 << (4UL & 63UL));
    ulong v28 = (v26 ^ v27);
    ulong v29 = (v28 >> (8UL & 63UL));
    ulong v30 = (v28 & 255UL);
    ulong v31 = (v30 ^ v29);
    ulong v32 = (v29 << (1UL & 63UL));
    ulong v33 = (v31 ^ v32);
    ulong v34 = (v29 << (3UL & 63UL));
    ulong v35 = (v33 ^ v34);
    ulong v36 = (v29 << (4UL & 63UL));
    ulong v37 = (v35 ^ v36);
    ulong v38 = (v37 & 255UL);
    return v38;
}

kernel void k0_metal_decl_gf2_kern_gf8_square(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_square(in0[gid]);
    }
}

inline ulong l_Gf2Kernel_IR_clmul64Hi(ulong v0, ulong v1) {
    ulong v2 = ln_clmul64_hi(v0, v1);
    return v2;
}

kernel void k0_metal_decl_l_Gf2Kernel_IR_clmul64Hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = l_Gf2Kernel_IR_clmul64Hi(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_blake3_add_carry_aux(ulong v0, ulong v1) {
    ulong v2 = 4294967295UL;
    ulong v3 = 2147483647UL;
    ulong v4 = (v0 & 4294967295UL);
    ulong v5 = (v1 & 4294967295UL);
    ulong v6 = (v4 + v5);
    ulong v7 = (v6 & 4294967295UL);
    ulong v8 = (v7 ^ v4);
    ulong v9 = (v8 ^ v5);
    ulong v10 = (v4 ^ v9);
    ulong v11 = (v10 & 2147483647UL);
    ulong v12 = (v5 ^ v9);
    ulong v13 = (v12 & 2147483647UL);
    ulong v14 = (v11 & v13);
    return v14;
}

kernel void k0_metal_decl_gf2_kern_blake3_add_carry_aux(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_blake3_add_carry_aux(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_ghash_mul_unred_r2(ulong v0, ulong v1, ulong v2, ulong v3) {
    ulong v4 = ln_clmul64(v1, v3);
    ulong v5 = ln_clmul64_hi(v0, v3);
    ulong v6 = ln_clmul64_hi(v1, v2);
    ulong v7 = (v5 ^ v6);
    ulong v8 = (v4 ^ v7);
    return v8;
}

kernel void k0_metal_decl_gf2_kern_ghash_mul_unred_r2(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device ulong* out [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ghash_mul_unred_r2(in0[gid], in1[gid], in2[gid], in3[gid]);
    }
}

inline ulong gf2_kern_inv_ntt_cyclic_offset_hi(ulong v0) {
    ulong v1 = 2UL;
    ulong v2 = (v0 & 2UL);
    ulong v3 = 3UL;
    ulong v4 = (v2 << (3UL & 63UL));
    ulong v5 = 4UL;
    ulong v6 = (v0 ^ 4UL);
    ulong v7 = (v6 & 4UL);
    ulong v8 = 12UL;
    ulong v9 = (v7 * 12UL);
    ulong v10 = (v4 + v9);
    return v10;
}

kernel void k0_metal_decl_gf2_kern_inv_ntt_cyclic_offset_hi(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_inv_ntt_cyclic_offset_hi(in0[gid]);
    }
}

inline ulong gf2_kern_nibble_hi(ulong v0) {
    ulong v1 = 4UL;
    ulong v2 = (v0 >> (4UL & 63UL));
    ulong v3 = 15UL;
    ulong v4 = (v2 & 15UL);
    return v4;
}

kernel void k0_metal_decl_gf2_kern_nibble_hi(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_nibble_hi(in0[gid]);
    }
}

inline ulong gf2_kern_ntt_radix4_comb_a(ulong v0, ulong v1, ulong v2) {
    ulong v3 = (v0 ^ v1);
    ulong v4 = (v3 ^ v2);
    return v4;
}

kernel void k0_metal_decl_gf2_kern_ntt_radix4_comb_a(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ntt_radix4_comb_a(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_mul_by_x_hi(ulong v0, ulong v1) {
    ulong v2 = 1UL;
    ulong v3 = (v1 << (1UL & 63UL));
    ulong v4 = 63UL;
    ulong v5 = (v0 >> (63UL & 63UL));
    ulong v6 = (v3 | v5);
    return v6;
}

kernel void k0_metal_decl_gf2_kern_mul_by_x_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_mul_by_x_hi(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_ghash_mul_lo(ulong v0, ulong v1, ulong v2, ulong v3) {
    ulong v4 = ln_clmul64(v0, v2);
    ulong v5 = ln_clmul64_hi(v0, v3);
    ulong v6 = ln_clmul64_hi(v1, v2);
    ulong v7 = ln_clmul64(v1, v3);
    ulong v8 = ln_clmul64_hi(v1, v3);
    ulong v9 = 135UL;
    ulong v10 = ln_clmul64_hi(v8, 135UL);
    ulong v11 = (v5 ^ v6);
    ulong v12 = (v11 ^ v10);
    ulong v13 = (v7 ^ v12);
    ulong v14 = ln_clmul64(v13, 135UL);
    ulong v15 = (v4 ^ v14);
    return v15;
}

kernel void k0_metal_decl_gf2_kern_ghash_mul_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device ulong* out [[buffer(4)]],
    constant ulong& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ghash_mul_lo(in0[gid], in1[gid], in2[gid], in3[gid]);
    }
}

inline ulong gf2_kern_monomial_basis128_hi(ulong v0) {
    ulong v1 = 6UL;
    ulong v2 = (v0 >> (6UL & 63UL));
    ulong v3 = 1UL;
    ulong v4 = (v2 & 1UL);
    ulong v5 = 63UL;
    ulong v6 = (v0 & 63UL);
    ulong v7 = (v4 << (v6 & 63UL));
    return v7;
}

kernel void k0_metal_decl_gf2_kern_monomial_basis128_hi(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_monomial_basis128_hi(in0[gid]);
    }
}

inline ulong gf2_kern_gf8_reduce_x4(ulong v0) {
    ulong v1 = 71777214294589695UL;
    ulong v2 = 8UL;
    ulong v3 = (v0 >> (8UL & 63UL));
    ulong v4 = (v3 & 71777214294589695UL);
    ulong v5 = (v0 & 71777214294589695UL);
    ulong v6 = (v5 ^ v4);
    ulong v7 = 1UL;
    ulong v8 = (v4 << (1UL & 63UL));
    ulong v9 = (v6 ^ v8);
    ulong v10 = 3UL;
    ulong v11 = (v4 << (3UL & 63UL));
    ulong v12 = (v9 ^ v11);
    ulong v13 = 4UL;
    ulong v14 = (v4 << (4UL & 63UL));
    ulong v15 = (v12 ^ v14);
    ulong v16 = (v15 >> (8UL & 63UL));
    ulong v17 = (v16 & 71777214294589695UL);
    ulong v18 = (v15 & 71777214294589695UL);
    ulong v19 = (v18 ^ v17);
    ulong v20 = (v17 << (1UL & 63UL));
    ulong v21 = (v19 ^ v20);
    ulong v22 = (v17 << (3UL & 63UL));
    ulong v23 = (v21 ^ v22);
    ulong v24 = (v17 << (4UL & 63UL));
    ulong v25 = (v23 ^ v24);
    ulong v26 = (v25 & 71777214294589695UL);
    return v26;
}

kernel void k0_metal_decl_gf2_kern_gf8_reduce_x4(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_gf8_reduce_x4(in0[gid]);
    }
}

inline ulong gf2_kern_clmul64_lo(ulong v0, ulong v1) {
    ulong v2 = ln_clmul64(v0, v1);
    return v2;
}

kernel void k0_metal_decl_gf2_kern_clmul64_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_clmul64_lo(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_clear_lowest_one(ulong v0) {
    ulong v1 = 1UL;
    ulong v2 = (v0 - 1UL);
    ulong v3 = (v0 & v2);
    return v3;
}

kernel void k0_metal_decl_gf2_kern_clear_lowest_one(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_clear_lowest_one(in0[gid]);
    }
}

inline ulong gf2_kern_f128_pre_reduce_const_lo(ulong v0) {
    ulong v1 = 135UL;
    ulong v2 = ln_clmul64(v0, 135UL);
    return v2;
}

kernel void k0_metal_decl_gf2_kern_f128_pre_reduce_const_lo(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_f128_pre_reduce_const_lo(in0[gid]);
    }
}

inline ulong gf2_kern_bitrecord_flush_spill(ulong v0, ulong v1) {
    ulong v2 = 1UL;
    ulong v3 = (v0 >> (1UL & 63UL));
    ulong v4 = 63UL;
    ulong v5 = (v1 & 63UL);
    ulong v6 = (63UL - v5);
    ulong v7 = (v3 >> (v6 & 63UL));
    return v7;
}

kernel void k0_metal_decl_gf2_kern_bitrecord_flush_spill(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_bitrecord_flush_spill(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_f128_fma_hi(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4) {
    ulong v5 = ln_clmul64_hi(v0, v2);
    ulong v6 = ln_clmul64(v0, v3);
    ulong v7 = ln_clmul64_hi(v0, v3);
    ulong v8 = ln_clmul64(v1, v2);
    ulong v9 = ln_clmul64_hi(v1, v2);
    ulong v10 = ln_clmul64(v1, v3);
    ulong v11 = ln_clmul64_hi(v1, v3);
    ulong v12 = 135UL;
    ulong v13 = ln_clmul64(v11, 135UL);
    ulong v14 = ln_clmul64_hi(v11, 135UL);
    ulong v15 = (v7 ^ v9);
    ulong v16 = (v15 ^ v14);
    ulong v17 = (v10 ^ v16);
    ulong v18 = ln_clmul64_hi(v17, 135UL);
    ulong v19 = (v6 ^ v8);
    ulong v20 = (v19 ^ v13);
    ulong v21 = (v5 ^ v18);
    ulong v22 = (v20 ^ v21);
    ulong v23 = (v22 ^ v4);
    return v23;
}

kernel void k0_metal_decl_gf2_kern_f128_fma_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device ulong* out [[buffer(5)]],
    constant ulong& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_f128_fma_hi(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid]);
    }
}

inline ulong gf2_kern_keccak_rol64(ulong v0, ulong v1) {
    ulong v2 = 63UL;
    ulong v3 = (v1 & 63UL);
    ulong v4 = (v0 << (v3 & 63UL));
    ulong v5 = 1UL;
    ulong v6 = (v0 >> (1UL & 63UL));
    ulong v7 = (63UL - v3);
    ulong v8 = (v6 >> (v7 & 63UL));
    ulong v9 = (v4 | v8);
    return v9;
}

kernel void k0_metal_decl_gf2_kern_keccak_rol64(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_keccak_rol64(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_or_u32_hi(ulong v0, ulong v1) {
    ulong v2 = 63UL;
    ulong v3 = (v1 & 63UL);
    ulong v4 = 32UL;
    ulong v5 = ((32UL < v3) ? 1UL : 0UL);
    ulong v6 = ((v5 == 0UL) ? 1UL : 0UL);
    if (v6 != 0UL) {
        ulong v7 = 0UL;
        return v7;
    } else {
        ulong v8 = 4294967295UL;
        ulong v9 = (v0 & 4294967295UL);
        ulong v10 = 64UL;
        ulong v11 = (64UL - v3);
        ulong v12 = (v9 >> (v11 & 63UL));
        return v12;
    }
}

kernel void k0_metal_decl_gf2_kern_or_u32_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_or_u32_hi(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_sha256_small_sigma1(ulong v0) {
    ulong v1 = 4294967295UL;
    ulong v2 = (v0 & 4294967295UL);
    ulong v3 = 17UL;
    ulong v4 = (v2 >> (17UL & 63UL));
    ulong v5 = 15UL;
    ulong v6 = (v2 << (15UL & 63UL));
    ulong v7 = (v4 | v6);
    ulong v8 = (v7 & 4294967295UL);
    ulong v9 = 19UL;
    ulong v10 = (v2 >> (19UL & 63UL));
    ulong v11 = 13UL;
    ulong v12 = (v2 << (13UL & 63UL));
    ulong v13 = (v10 | v12);
    ulong v14 = (v13 & 4294967295UL);
    ulong v15 = 10UL;
    ulong v16 = (v2 >> (10UL & 63UL));
    ulong v17 = (v8 ^ v14);
    ulong v18 = (v17 ^ v16);
    return v18;
}

kernel void k0_metal_decl_gf2_kern_sha256_small_sigma1(
    device const ulong* in0 [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant ulong& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_sha256_small_sigma1(in0[gid]);
    }
}

inline ulong gf2_kern_ghash_reduce_lo(ulong v0, ulong v1, ulong v2) {
    ulong v3 = 135UL;
    ulong v4 = ln_clmul64_hi(v2, 135UL);
    ulong v5 = (v1 ^ v4);
    ulong v6 = ln_clmul64(v5, 135UL);
    ulong v7 = (v0 ^ v6);
    return v7;
}

kernel void k0_metal_decl_gf2_kern_ghash_reduce_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device ulong* out [[buffer(3)]],
    constant ulong& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ghash_reduce_lo(in0[gid], in1[gid], in2[gid]);
    }
}

inline ulong gf2_kern_f128_reduce_deg191_lo(ulong v0, ulong v1) {
    ulong v2 = 135UL;
    ulong v3 = ln_clmul64(v1, 135UL);
    ulong v4 = (v0 ^ v3);
    return v4;
}

kernel void k0_metal_decl_gf2_kern_f128_reduce_deg191_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_f128_reduce_deg191_lo(in0[gid], in1[gid]);
    }
}

inline ulong gf2_kern_mle_fold_hi(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4, ulong v5) {
    ulong v6 = (v0 ^ v2);
    ulong v7 = (v1 ^ v3);
    ulong v8 = ln_clmul64_hi(v6, v4);
    ulong v9 = ln_clmul64(v6, v5);
    ulong v10 = ln_clmul64_hi(v6, v5);
    ulong v11 = ln_clmul64(v7, v4);
    ulong v12 = ln_clmul64_hi(v7, v4);
    ulong v13 = ln_clmul64(v7, v5);
    ulong v14 = ln_clmul64_hi(v7, v5);
    ulong v15 = 135UL;
    ulong v16 = ln_clmul64(v14, 135UL);
    ulong v17 = ln_clmul64_hi(v14, 135UL);
    ulong v18 = (v10 ^ v12);
    ulong v19 = (v18 ^ v17);
    ulong v20 = (v13 ^ v19);
    ulong v21 = ln_clmul64_hi(v20, 135UL);
    ulong v22 = (v9 ^ v11);
    ulong v23 = (v22 ^ v16);
    ulong v24 = (v8 ^ v21);
    ulong v25 = (v23 ^ v24);
    ulong v26 = (v1 ^ v25);
    return v26;
}

kernel void k0_metal_decl_gf2_kern_mle_fold_hi(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device const ulong* in5 [[buffer(5)]],
    device ulong* out [[buffer(6)]],
    constant ulong& n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_mle_fold_hi(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid], in5[gid]);
    }
}

inline ulong gf2_kern_f128_fma_lo(ulong v0, ulong v1, ulong v2, ulong v3, ulong v4) {
    ulong v5 = ln_clmul64(v0, v2);
    ulong v6 = ln_clmul64_hi(v0, v3);
    ulong v7 = ln_clmul64_hi(v1, v2);
    ulong v8 = ln_clmul64(v1, v3);
    ulong v9 = ln_clmul64_hi(v1, v3);
    ulong v10 = 135UL;
    ulong v11 = ln_clmul64_hi(v9, 135UL);
    ulong v12 = (v6 ^ v7);
    ulong v13 = (v12 ^ v11);
    ulong v14 = (v8 ^ v13);
    ulong v15 = ln_clmul64(v14, 135UL);
    ulong v16 = (v5 ^ v15);
    ulong v17 = (v16 ^ v4);
    return v17;
}

kernel void k0_metal_decl_gf2_kern_f128_fma_lo(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device const ulong* in2 [[buffer(2)]],
    device const ulong* in3 [[buffer(3)]],
    device const ulong* in4 [[buffer(4)]],
    device ulong* out [[buffer(5)]],
    constant ulong& n [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_f128_fma_lo(in0[gid], in1[gid], in2[gid], in3[gid], in4[gid]);
    }
}

inline ulong gf2_kern_ntt_replicated_msg_idx(ulong v0, ulong v1) {
    ulong v2 = 1UL;
    ulong v3 = 63UL;
    ulong v4 = (v1 & 63UL);
    ulong v5 = (1UL << (v4 & 63UL));
    ulong v6 = (v5 - 1UL);
    ulong v7 = (v0 & v6);
    return v7;
}

kernel void k0_metal_decl_gf2_kern_ntt_replicated_msg_idx(
    device const ulong* in0 [[buffer(0)]],
    device const ulong* in1 [[buffer(1)]],
    device ulong* out [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        out[gid] = gf2_kern_ntt_replicated_msg_idx(in0[gid], in1[gid]);
    }
}

kernel void k0_kernel_ntt_butterfly_f128(
    device ulong2* top [[buffer(0)]],
    device ulong2* bot [[buffer(1)]],
    constant ulong2& twiddle [[buffer(2)]],
    constant ulong& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((ulong)gid < n) {
        ulong2 u = top[gid];
        ulong2 v = bot[gid];
        ulong p_lo = gf2_kern_ghash_mul_lo(v.x, v.y, twiddle.x, twiddle.y);
        ulong p_hi = gf2_kern_ghash_mul_hi(v.x, v.y, twiddle.x, twiddle.y);
        ulong nu_lo = gf2_kern_ntt_btf_u_lo(u.x, p_lo);
        ulong nu_hi = gf2_kern_ntt_btf_u_lo(u.y, p_hi);
        ulong nv_lo = gf2_kern_ntt_btf_v_lo(u.x, v.x, p_lo);
        ulong nv_hi = gf2_kern_ntt_btf_v_lo(u.y, v.y, p_hi);
        top[gid] = ulong2(nu_lo, nu_hi);
        bot[gid] = ulong2(nv_lo, nv_hi);
    }
}

kernel void k0_kernel_ntt_interleaved_layer_f128(
    device ulong2* data [[buffer(0)]],
    device const ulong2* layer_twiddles [[buffer(1)]],
    constant uint4& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint log_d = params.x;
    uint layer = params.y;
    uint log_num_ntts = params.z;
    uint total_threads = params.w;
    if (gid < total_threads) {
        uint num_ntts_mask = (1u << log_num_ntts) - 1u;
        uint lane = gid & num_ntts_mask;
        uint pair_idx = gid >> log_num_ntts;
        uint shift_half = log_d - 1u - layer;
        uint block_size_half = 1u << shift_half;
        uint block = pair_idx >> shift_half;
        uint row = pair_idx & (block_size_half - 1u);
        ulong pos_top = ((ulong)block << (log_d - layer)) | (ulong)row;
        ulong pos_bot = pos_top | (ulong)block_size_half;
        ulong off_top = (pos_top << log_num_ntts) | (ulong)lane;
        ulong off_bot = (pos_bot << log_num_ntts) | (ulong)lane;
        ulong2 u = data[off_top];
        ulong2 v = data[off_bot];
        ulong2 tw = layer_twiddles[block];
        ulong p_lo = gf2_kern_ghash_mul_lo(v.x, v.y, tw.x, tw.y);
        ulong p_hi = gf2_kern_ghash_mul_hi(v.x, v.y, tw.x, tw.y);
        ulong nu_lo = gf2_kern_ntt_btf_u_lo(u.x, p_lo);
        ulong nu_hi = gf2_kern_ntt_btf_u_lo(u.y, p_hi);
        ulong nv_lo = gf2_kern_ntt_btf_v_lo(u.x, v.x, p_lo);
        ulong nv_hi = gf2_kern_ntt_btf_v_lo(u.y, v.y, p_hi);
        data[off_top] = ulong2(nu_lo, nu_hi);
        data[off_bot] = ulong2(nv_lo, nv_hi);
    }
}

kernel void k0_kernel_ntt_radix4_interleaved_f128(
    device ulong2* data [[buffer(0)]],
    device const ulong2* twiddles_l0 [[buffer(1)]],
    device const ulong2* twiddles_l1 [[buffer(2)]],
    constant uint4& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint log_d = params.x;
    uint layer0 = params.y;
    uint log_num_ntts = params.z;
    uint total_quads = params.w;
    if (gid < total_quads) {
        uint num_ntts_mask = (1u << log_num_ntts) - 1u;
        uint lane = gid & num_ntts_mask;
        uint quad_idx = gid >> log_num_ntts;
        uint shift_q = log_d - 2u - layer0;
        uint quarter = 1u << shift_q;
        uint block = quad_idx >> shift_q;
        uint row = quad_idx & (quarter - 1u);
        ulong base_pos = ((ulong)block << (log_d - layer0)) | (ulong)row;
        ulong o0 = (base_pos << log_num_ntts) | (ulong)lane;
        ulong o1 = ((base_pos | (ulong)quarter) << log_num_ntts) | (ulong)lane;
        ulong o2 = ((base_pos | ((ulong)quarter << 1UL)) << log_num_ntts) | (ulong)lane;
        ulong o3 = ((base_pos | ((ulong)quarter * 3UL)) << log_num_ntts) | (ulong)lane;
        ulong2 a = data[o0]; ulong2 b = data[o1]; ulong2 c = data[o2]; ulong2 d = data[o3];
        ulong2 tw_out = twiddles_l0[block];
        ulong2 tw_ia = twiddles_l1[block << 1u];
        ulong2 tw_ib = twiddles_l1[(block << 1u) | 1u];
        ulong pc_lo = gf2_kern_ghash_mul_lo(c.x, c.y, tw_out.x, tw_out.y);
        ulong pc_hi = gf2_kern_ghash_mul_hi(c.x, c.y, tw_out.x, tw_out.y);
        ulong pd_lo = gf2_kern_ghash_mul_lo(d.x, d.y, tw_out.x, tw_out.y);
        ulong pd_hi = gf2_kern_ghash_mul_hi(d.x, d.y, tw_out.x, tw_out.y);
        ulong2 na = ulong2(a.x ^ pc_lo, a.y ^ pc_hi); ulong2 nc = ulong2(c.x ^ na.x, c.y ^ na.y);
        ulong2 nb = ulong2(b.x ^ pd_lo, b.y ^ pd_hi); ulong2 nd = ulong2(d.x ^ nb.x, d.y ^ nb.y);
        ulong pb2_lo = gf2_kern_ghash_mul_lo(nb.x, nb.y, tw_ia.x, tw_ia.y);
        ulong pb2_hi = gf2_kern_ghash_mul_hi(nb.x, nb.y, tw_ia.x, tw_ia.y);
        ulong pd2_lo = gf2_kern_ghash_mul_lo(nd.x, nd.y, tw_ib.x, tw_ib.y);
        ulong pd2_hi = gf2_kern_ghash_mul_hi(nd.x, nd.y, tw_ib.x, tw_ib.y);
        ulong2 fa = ulong2(na.x ^ pb2_lo, na.y ^ pb2_hi); ulong2 fb = ulong2(nb.x ^ fa.x, nb.y ^ fa.y);
        ulong2 fc = ulong2(nc.x ^ pd2_lo, nc.y ^ pd2_hi); ulong2 fd = ulong2(nd.x ^ fc.x, nd.y ^ fc.y);
        data[o0] = fa; data[o1] = fb; data[o2] = fc; data[o3] = fd;
    }
}

inline void ln_blake3_g(thread uint& a, thread uint& b, thread uint& c, thread uint& d, uint mx, uint my) {
    uint t0 = (uint)gf2_kern_blake3_add_sum((ulong)a, (ulong)b);
    a = (uint)gf2_kern_blake3_add_sum((ulong)t0, (ulong)mx);
    d = (uint)gf2_kern_blake3_xor_rotr32((ulong)d, (ulong)a, 16UL);
    c = (uint)gf2_kern_blake3_add_sum((ulong)c, (ulong)d);
    b = (uint)gf2_kern_blake3_xor_rotr32((ulong)b, (ulong)c, 12UL);
    uint t1 = (uint)gf2_kern_blake3_add_sum((ulong)a, (ulong)b);
    a = (uint)gf2_kern_blake3_add_sum((ulong)t1, (ulong)my);
    d = (uint)gf2_kern_blake3_xor_rotr32((ulong)d, (ulong)a, 8UL);
    c = (uint)gf2_kern_blake3_add_sum((ulong)c, (ulong)d);
    b = (uint)gf2_kern_blake3_xor_rotr32((ulong)b, (ulong)c, 7UL);
}

inline void ln_blake3_round(thread uint s[16], thread const uint m[16]) {
    ln_blake3_g(s[0], s[4], s[8],  s[12], m[0],  m[1]);
    ln_blake3_g(s[1], s[5], s[9],  s[13], m[2],  m[3]);
    ln_blake3_g(s[2], s[6], s[10], s[14], m[4],  m[5]);
    ln_blake3_g(s[3], s[7], s[11], s[15], m[6],  m[7]);
    ln_blake3_g(s[0], s[5], s[10], s[15], m[8],  m[9]);
    ln_blake3_g(s[1], s[6], s[11], s[12], m[10], m[11]);
    ln_blake3_g(s[2], s[7], s[8],  s[13], m[12], m[13]);
    ln_blake3_g(s[3], s[4], s[9],  s[14], m[14], m[15]);
}

inline void ln_blake3_permute(thread uint m[16]) {
    uint t[16];
    t[0] = m[2];  t[1] = m[6];  t[2] = m[3];  t[3] = m[10];
    t[4] = m[7];  t[5] = m[0];  t[6] = m[4];  t[7] = m[13];
    t[8] = m[1];  t[9] = m[11]; t[10] = m[12]; t[11] = m[5];
    t[12] = m[9]; t[13] = m[14]; t[14] = m[15]; t[15] = m[8];
    for (uint i = 0u; i < 16u; ++i) { m[i] = t[i]; }
}

inline void ln_blake3_compress(thread uint cv[8], thread uint m[16], uint block_len, uint flags) {
    uint s[16] = {
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0u, 0u, block_len, flags
    };
    for (uint r = 0u; r < 6u; ++r) {
        ln_blake3_round(s, m);
        ln_blake3_permute(m);
    }
    ln_blake3_round(s, m);
    for (uint i = 0u; i < 8u; ++i) {
        ulong pair = gf2_kern_blake3_finalize_word_pair((ulong)s[i], (ulong)s[i + 8u], (ulong)cv[i]);
        cv[i] = (uint)(pair & 0xffffffffUL);
    }
}

kernel void k0_kernel_blake3_merkle_leaves(
    device const uint* data [[buffer(0)]],
    device uint* out_leaves [[buffer(1)]],
    constant uint2& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint num_leaves = params.x;
    uint blocks_per_leaf = params.y;
    if (gid < num_leaves) {
        uint cv[8] = {
            0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
            0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
        };
        ulong base_word = (ulong)gid * (ulong)blocks_per_leaf * 16UL;
        for (uint blk = 0u; blk < blocks_per_leaf; ++blk) {
            uint m[16];
            ulong blk_off = base_word + (ulong)blk * 16UL;
            for (uint i = 0u; i < 16u; ++i) { m[i] = data[blk_off + (ulong)i]; }
            uint flags = (uint)gf2_kern_blake3_chunk_flags((ulong)blk, (ulong)blocks_per_leaf);
            ln_blake3_compress(cv, m, 64u, flags);
        }
        ulong out_off = (ulong)gid * 8UL;
        for (uint i = 0u; i < 8u; ++i) { out_leaves[out_off + (ulong)i] = cv[i]; }
    }
}

kernel void k0_kernel_blake3_merkle_parents(
    device const uint* in_children [[buffer(0)]],
    device uint* out_parents [[buffer(1)]],
    constant uint& num_parents [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < num_parents) {
        uint cv[8] = {
            0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
            0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
        };
        uint m[16];
        ulong in_off = (ulong)gid * 16UL;
        for (uint i = 0u; i < 16u; ++i) { m[i] = in_children[in_off + (ulong)i]; }
        ln_blake3_compress(cv, m, 64u, 4u);
        ulong out_off = (ulong)gid * 8UL;
        for (uint i = 0u; i < 8u; ++i) { out_parents[out_off + (ulong)i] = cv[i]; }
    }
}"####;

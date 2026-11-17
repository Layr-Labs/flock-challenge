// bigab: GPU arm of the round-one AB transform (gpu_commit::imp::bigab_metal; FLOCK_NO_GPU_AB_ARM).
#include <metal_stdlib>
using namespace metal;

struct BigAbGenP {
    uint inp;                    // input record of thread 0 (25 words each: cv[8], m[16], counter_lo)
    uint pad0;
    uint first;                  // instance index of thread 0
    uint count;                  // instances in this batch
    uint stride;                 // instances per staging slab
    uint pad1; uint pad2; uint pad3;
};
// Rolling LSB-first writer of the a and b rows: `fill` (< 32) pending bits in (aa, ab); one complete word of each
// row per store, a at p[0], b at p[stride]. Every thread has the same bit positions, so control flow is uniform.
struct BigAbW { uint aa; uint ab; uint fill; device uint* p; uint stride; };
static inline void bigab_put(thread BigAbW& w, uint va, uint vb, uint n) {
    w.aa |= va << w.fill;
    w.ab |= vb << w.fill;
    uint f = w.fill + n;
    if (f >= 32u) {
        w.p[0] = w.aa;
        w.p[w.stride] = w.ab;
        w.p += 2u * w.stride;
        uint used = 32u - w.fill;
        w.aa = used < 32u ? va >> used : 0u;
        w.ab = used < 32u ? vb >> used : 0u;
        f -= 32u;
    }
    w.fill = f;
}
// One witness ADD: a row = x ^ cin, b row = y ^ cin (31 bits each); returns the sum.
static inline uint bigab_add(thread BigAbW& w, uint x, uint y) {
    uint s = x + y;
    uint c = s ^ x ^ y;
    bigab_put(w, (x ^ c) & 0x7fffffffu, (y ^ c) & 0x7fffffffu, 31u);
    return s;
}
static inline uint bigab_rotr(uint x, uint r) { return (x >> r) | (x << (32u - r)); }
// One G with its 250-bit record: six ADDs, then b_new and d_new (a = value, b = ones).
static inline void bigab_g(thread BigAbW& w, thread uint& a, thread uint& b, thread uint& c, thread uint& d,
                           uint mx, uint my) {
    uint t0 = bigab_add(w, a, b);
    uint a1 = bigab_add(w, t0, mx);
    uint d1 = bigab_rotr(d ^ a1, 16u);
    uint c1 = bigab_add(w, c, d1);
    uint b1 = bigab_rotr(b ^ c1, 12u);
    uint t1 = bigab_add(w, a1, b1);
    uint a2 = bigab_add(w, t1, my);
    uint d2 = bigab_rotr(d1 ^ a2, 8u);
    uint c2 = bigab_add(w, c1, d2);
    uint bn = bigab_rotr(b1 ^ c2, 7u);
    bigab_put(w, bn, 0xffffffffu, 32u);
    bigab_put(w, d2, 0xffffffffu, 32u);
    a = a2; b = bn; c = c2; d = d2;
}
// ---- bigab_gen: one thread per instance; live a/b words (bits [1024, 15360)) word-interleaved ----
kernel void bigab_gen(device uint* S [[buffer(0)]], constant BigAbGenP& P [[buffer(1)]],
                      device const uint* I [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= P.count) return;
    // The compression's inputs (counter_hi 0, block_len 64, flags 11 are the benchmark's constants).
    device const uint* in = I + (P.inp + t) * 25u;
    uint cv[8], m[16];
    for (uint k = 0u; k < 8u; k++) cv[k] = in[k];
    for (uint k = 0u; k < 16u; k++) m[k] = in[8u + k];
    uint ctr = in[24];
    BigAbW w = { 0u, 0u, 0u, S + t, P.stride };
    const uint ones = 0xffffffffu;
    bigab_put(w, m[15] >> 31, 1u, 1u);     // bit 1024: m[15] bit 31
    bigab_put(w, ctr, ones, 32u);          // counter_lo
    bigab_put(w, 0u, ones, 32u);           // counter_hi
    bigab_put(w, 64u, ones, 32u);          // block_len
    bigab_put(w, 11u, ones, 32u);          // flags
    // State lanes and message words through small private arrays: one G body for all 56 G's keeps the
    // backend compile short (the arm builds its pipelines in every worker process).
    uint v[16] = { cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                   0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, ctr, 0u, 64u, 11u };
    const ushort lanes[8] = { 0xc840u, 0xd951u, 0xea62u, 0xfb73u, 0xfa50u, 0xcb61u, 0xd872u, 0xe943u };
    const uchar msg[56] = { 0x10u, 0x32u, 0x54u, 0x76u, 0x98u, 0xbau, 0xdcu, 0xfeu, 0x62u, 0xa3u, 0x07u, 0xd4u, 0xb1u, 0x5cu, 0xe9u, 0x8fu, 0x43u, 0xcau, 0x2du, 0xe7u, 0x56u, 0x09u, 0xfbu, 0x18u, 0x7au, 0x9cu, 0x3eu, 0xfdu, 0x04u, 0x2bu, 0x85u, 0x61u, 0xdcu, 0xb9u, 0xafu, 0x8eu, 0x27u, 0x35u, 0x10u, 0x46u, 0xe9u, 0x5bu, 0xc8u, 0x1fu, 0x3du, 0xa0u, 0x62u, 0x74u, 0xfbu, 0x05u, 0x91u, 0x68u, 0xaeu, 0xc2u, 0x43u, 0xd7u };
    #pragma clang loop unroll(disable)
    for (uint gi = 0u; gi < 56u; gi++) {
        uint l = lanes[gi & 7u], mi = msg[gi];
        uint ia = l & 15u, ib = (l >> 4) & 15u, ic = (l >> 8) & 15u, id = l >> 12;
        uint a = v[ia], b = v[ib], c = v[ic], d = v[id];
        bigab_g(w, a, b, c, d, m[mi & 15u], m[mi >> 4]);
        v[ia] = a; v[ib] = b; v[ic] = c; v[id] = d;
    }
    uint v8 = v[8], v9 = v[9], v10 = v[10], v11 = v[11], v12 = v[12], v13 = v[13], v14 = v[14];
    // out_hi words 0..6 (the live range ends at bit 15360, inside word 6).
    bigab_put(w, v8 ^ cv[0], ones, 32u);
    bigab_put(w, v9 ^ cv[1], ones, 32u);
    bigab_put(w, v10 ^ cv[2], ones, 32u);
    bigab_put(w, v11 ^ cv[3], ones, 32u);
    bigab_put(w, v12 ^ cv[4], ones, 32u);
    bigab_put(w, v13 ^ cv[5], ones, 32u);
    bigab_put(w, v14 ^ cv[6], ones, 32u);
}

// ---- bigab_xform: thread per (instance, live block, 8-lane chunk): byte table + SWAR GF(2^8) products ----
static inline uint2 bigab_xtime(uint2 a) {
    uint2 hi = (a >> 7) & 0x01010101u;
    return ((a & 0x7f7f7f7fu) << 1) ^ (hi * 0x1bu);
}
static inline uint2 bigab_mul(uint2 a, uint2 b) {
    uint2 r = uint2(0u);
    for (uint i = 0u; i < 8u; i++) {
        uint2 m = (b >> i) & 0x01010101u;
        r ^= a & ((m << 8) - m);
        a = bigab_xtime(a);
    }
    return r;
}
struct BigAbXformP {
    uint stride;    // instances per staging slab (word w of a: S[2 w stride + i], of b: S[(2 w + 1) stride + i])
    uint count;     // instances in this batch
    uint out_inst;  // output instance index of staging instance 0 (relative to the output buffer)
    uint iters;     // 32-item groups per threadgroup (amortises the table load)
};
kernel void bigab_xform(device const uint* S [[buffer(0)]], device const uint2* tab [[buffer(1)]],
                        device uint2* out [[buffer(2)]], constant BigAbXformP& P [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup uint2 T[2048];
    for (uint i = lid; i < 2048u; i += 256u) T[i] = tab[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = lid & 7u;
    uint stride = P.stride;
    uint items = P.count * 28u;
    for (uint it = 0u; it < P.iters; it++) {
        uint item = (tg * P.iters + it) * 32u + (lid >> 3);
        if (item >= items) break;
        uint j = item / P.count;
        uint i = item - j * P.count;
        device const uint* s = S + i + (32u * j) * stride;
        uint2 acc = uint2(0u);
        for (int k = 7; k >= 0; k--) {
            uint w0 = 2u * uint(k);
            uint2 av = uint2(s[(2u * w0) * stride], s[(2u * w0 + 2u) * stride]);
            uint2 bv = uint2(s[(2u * w0 + 1u) * stride], s[(2u * w0 + 3u) * stride]);
            uint2 x = uint2(0u), y = uint2(0u);
            for (uint by = 0u; by < 8u; by++) {
                uint xa = (av[by >> 2] >> (8u * (by & 3u))) & 0xffu;
                uint xb = (bv[by >> 2] >> (8u * (by & 3u))) & 0xffu;
                x ^= T[xa * 8u + (n ^ by)];
                y ^= T[xb * 8u + (n ^ by)];
            }
            acc = bigab_xtime(acc) ^ bigab_mul(x, y);
        }
        out[(P.out_inst + i) * 256u + 16u + 8u * j + n] = acc;
    }
}

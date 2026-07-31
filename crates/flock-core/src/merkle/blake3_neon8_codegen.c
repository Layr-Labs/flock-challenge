#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

// Source for blake3_neon8_macos.S. This translation unit is intentionally
// self-contained so the checked-in assembly can be reproduced with Apple
// clang without depending on Cargo's copy of the BLAKE3 C sources.
//
// Regenerate with Apple clang 21, then remove the host-specific
// `.build_version` directive before checking in the output:
//   xcrun clang -S -O3 -mcpu=apple-m3 -fno-stack-protector \
//     -fno-asynchronous-unwind-tables -fvisibility=hidden \
//     blake3_neon8_codegen.c -o blake3_neon8_macos.S

static const uint32_t IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

static const uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

static inline uint32x4_t rot16(uint32x4_t x) {
  return vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x)));
}

static inline uint32x4_t rot12(uint32x4_t x) {
  return vsriq_n_u32(vshlq_n_u32(x, 20), x, 12);
}

static inline uint32x4_t rot8(uint32x4_t x) {
  return vreinterpretq_u32_u8(__builtin_shufflevector(
      vreinterpretq_u8_u32(x), vreinterpretq_u8_u32(x),
      1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12));
}

static inline uint32x4_t rot7(uint32x4_t x) {
  return vsriq_n_u32(vshlq_n_u32(x, 25), x, 7);
}

static inline void transpose4(uint32x4_t vecs[4]) {
  uint32x4x2_t rows01 = vtrnq_u32(vecs[0], vecs[1]);
  uint32x4x2_t rows23 = vtrnq_u32(vecs[2], vecs[3]);
  vecs[0] = vcombine_u32(vget_low_u32(rows01.val[0]),
                         vget_low_u32(rows23.val[0]));
  vecs[1] = vcombine_u32(vget_low_u32(rows01.val[1]),
                         vget_low_u32(rows23.val[1]));
  vecs[2] = vcombine_u32(vget_high_u32(rows01.val[0]),
                         vget_high_u32(rows23.val[0]));
  vecs[3] = vcombine_u32(vget_high_u32(rows01.val[1]),
                         vget_high_u32(rows23.val[1]));
}

static inline void transpose_block4(const uint8_t *i0, const uint8_t *i1,
                                    const uint8_t *i2, const uint8_t *i3,
                                    size_t block_offset,
                                    uint32x4_t out[16]) {
  const uint8_t *inputs[4] = {i0, i1, i2, i3};
  for (size_t quarter = 0; quarter < 4; quarter++) {
    const size_t offset = block_offset + quarter * 16;
    out[quarter * 4 + 0] = vreinterpretq_u32_u8(vld1q_u8(inputs[0] + offset));
    out[quarter * 4 + 1] = vreinterpretq_u32_u8(vld1q_u8(inputs[1] + offset));
    out[quarter * 4 + 2] = vreinterpretq_u32_u8(vld1q_u8(inputs[2] + offset));
    out[quarter * 4 + 3] = vreinterpretq_u32_u8(vld1q_u8(inputs[3] + offset));
    transpose4(&out[quarter * 4]);
  }
}

#define G2(a, b, c, d, x, y)                                                \
  do {                                                                        \
    v0[a] = vaddq_u32(vaddq_u32(v0[a], v0[b]), m0[x]);                        \
    v1[a] = vaddq_u32(vaddq_u32(v1[a], v1[b]), m1[x]);                        \
    v0[d] = rot16(veorq_u32(v0[d], v0[a]));                                   \
    v1[d] = rot16(veorq_u32(v1[d], v1[a]));                                   \
    v0[c] = vaddq_u32(v0[c], v0[d]);                                          \
    v1[c] = vaddq_u32(v1[c], v1[d]);                                          \
    v0[b] = rot12(veorq_u32(v0[b], v0[c]));                                   \
    v1[b] = rot12(veorq_u32(v1[b], v1[c]));                                   \
    v0[a] = vaddq_u32(vaddq_u32(v0[a], v0[b]), m0[y]);                        \
    v1[a] = vaddq_u32(vaddq_u32(v1[a], v1[b]), m1[y]);                        \
    v0[d] = rot8(veorq_u32(v0[d], v0[a]));                                    \
    v1[d] = rot8(veorq_u32(v1[d], v1[a]));                                    \
    v0[c] = vaddq_u32(v0[c], v0[d]);                                          \
    v1[c] = vaddq_u32(v1[c], v1[d]);                                          \
    v0[b] = rot7(veorq_u32(v0[b], v0[c]));                                    \
    v1[b] = rot7(veorq_u32(v1[b], v1[c]));                                    \
  } while (0)

static __attribute__((always_inline)) inline void
round2(uint32x4_t v0[16], uint32x4_t v1[16],
       const uint32x4_t m0[16], const uint32x4_t m1[16], size_t round) {
  const uint8_t *s = MSG_SCHEDULE[round];
  G2(0, 4, 8, 12, s[0], s[1]);
  G2(1, 5, 9, 13, s[2], s[3]);
  G2(2, 6, 10, 14, s[4], s[5]);
  G2(3, 7, 11, 15, s[6], s[7]);
  G2(0, 5, 10, 15, s[8], s[9]);
  G2(1, 6, 11, 12, s[10], s[11]);
  G2(2, 7, 8, 13, s[12], s[13]);
  G2(3, 4, 9, 14, s[14], s[15]);
}

#undef G2

static inline void init_state(uint32x4_t v[16], const uint32x4_t h[8],
                              uint32_t flags) {
  for (size_t i = 0; i < 8; i++) {
    v[i] = h[i];
  }
  v[8] = vdupq_n_u32(IV[0]);
  v[9] = vdupq_n_u32(IV[1]);
  v[10] = vdupq_n_u32(IV[2]);
  v[11] = vdupq_n_u32(IV[3]);
  v[12] = vdupq_n_u32(0);
  v[13] = vdupq_n_u32(0);
  v[14] = vdupq_n_u32(64);
  v[15] = vdupq_n_u32(flags);
}

static inline void finish_block(uint32x4_t h[8], const uint32x4_t v[16]) {
  for (size_t i = 0; i < 8; i++) {
    h[i] = veorq_u32(v[i], v[i + 8]);
  }
}

static inline void store_cv4(uint32x4_t h[8], uint8_t *out) {
  transpose4(&h[0]);
  transpose4(&h[4]);
  vst1q_u8(out + 0, vreinterpretq_u8_u32(h[0]));
  vst1q_u8(out + 16, vreinterpretq_u8_u32(h[4]));
  vst1q_u8(out + 32, vreinterpretq_u8_u32(h[1]));
  vst1q_u8(out + 48, vreinterpretq_u8_u32(h[5]));
  vst1q_u8(out + 64, vreinterpretq_u8_u32(h[2]));
  vst1q_u8(out + 80, vreinterpretq_u8_u32(h[6]));
  vst1q_u8(out + 96, vreinterpretq_u8_u32(h[3]));
  vst1q_u8(out + 112, vreinterpretq_u8_u32(h[7]));
}

__attribute__((visibility("hidden")))
void flock_blake3_hash8_neon_1024(const uint8_t *data, uint8_t *out,
                                  size_t groups) {
  for (size_t group = 0; group < groups; group++) {
    const uint8_t *input = data + group * 8 * 1024;
    uint32x4_t h0[8], h1[8];
    for (size_t i = 0; i < 8; i++) {
      h0[i] = h1[i] = vdupq_n_u32(IV[i]);
    }

    for (size_t block = 0; block < 16; block++) {
      const size_t block_offset = block * 64;
      uint32x4_t m0[16], m1[16];
      uint32x4_t v0[16], v1[16];
      transpose_block4(input + 0 * 1024, input + 1 * 1024,
                       input + 2 * 1024, input + 3 * 1024,
                       block_offset, m0);
      transpose_block4(input + 4 * 1024, input + 5 * 1024,
                       input + 6 * 1024, input + 7 * 1024,
                       block_offset, m1);
      uint32_t flags = (block == 0 ? 1 : 0) | (block == 15 ? 2 : 0);
      init_state(v0, h0, flags);
      init_state(v1, h1, flags);

      round2(v0, v1, m0, m1, 0);
      round2(v0, v1, m0, m1, 1);
      round2(v0, v1, m0, m1, 2);
      round2(v0, v1, m0, m1, 3);
      round2(v0, v1, m0, m1, 4);
      round2(v0, v1, m0, m1, 5);
      round2(v0, v1, m0, m1, 6);

      finish_block(h0, v0);
      finish_block(h1, v1);
    }

    store_cv4(h0, out + group * 8 * 32 + 0 * 4 * 32);
    store_cv4(h1, out + group * 8 * 32 + 1 * 4 * 32);
  }
}

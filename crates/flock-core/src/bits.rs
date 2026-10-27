//! Small bit-manipulation primitives shared across modules.

/// Hacker's Delight (Sec. 7-3) 8×8 bit-matrix transpose stored in a `u64`.
///
/// The input holds 8 bytes representing 8 rows of 8 bits each; the output holds
/// the transposed matrix (bit `r·8 + c` of input → bit `c·8 + r` of output).
///
/// Shared by the lincheck byte-stripe builder (`flock_prover::r1cs_hashes::common`)
/// and the PCS ring-switch `fold_1b` kernels ([`crate::pcs::ring_switch`]).
#[inline(always)]
pub(crate) fn transpose_8x8_bits(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AAu64;
    x = x ^ t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCCu64;
    x = x ^ t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0u64;
    x = x ^ t ^ (t << 28);
    x
}

/// Bit-transpose 8 little-endian `u64` lanes (the 64-byte block they form) into
/// a 64-byte output stripe.
///
/// The 8 LE u64s viewed as 64 bytes are exactly the input shape of the NEON
/// [`bit_transpose_64bytes`] kernel (input byte `r·8 + c` = byte `c` of lane
/// `r`; output byte `c·8 + t` bit `r` = that byte's bit `t`), so this delegates
/// to it — ~5× fewer ops than the scalar per-column loop. Shared by the
/// lincheck byte-stripe builder (`flock_prover::r1cs_hashes::common`) and the
/// core R1CS matrix-apply ([`crate::r1cs`]).
///
/// [`bit_transpose_64bytes`]: crate::zerocheck::univariate_skip_optimized::bit_transpose_64bytes
#[inline(always)]
pub fn transpose_8_u64s_to_64_bytes(lanes: &[u64; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 64);
    // SAFETY: [u64; 8] is 64 bytes with no padding; u8 has weaker alignment.
    let input: &[u8; 64] = unsafe { &*(lanes.as_ptr() as *const [u8; 64]) };
    let out64: &mut [u8; 64] = out.try_into().expect("64-byte stripe slice");
    crate::zerocheck::univariate_skip_optimized::bit_transpose_64bytes(input, out64);
}

/// [`transpose_8_u64s_to_64_bytes`] of the eight words `base[l * stride]`,
/// `l = 0..8`, read straight into vector lanes (`ldr d` + `ld1 {v.d}[1]` per
/// lane pair) instead of through general registers. Same output bytes.
///
/// # Safety
/// `base.add(l * stride)` must be valid for an 8-byte read for every `l < 8`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn transpose_8_strided_u64s_to_64_bytes(base: *const u64, stride: usize, out: &mut [u8; 64]) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn lane_pair(lo: *const u64, hi: *const u64) -> uint8x16_t {
        // SAFETY: the caller's contract covers both 8-byte reads.
        unsafe { vreinterpretq_u8_u64(vcombine_u64(vld1_u64(lo), vld1_u64(hi))) }
    }

    // SAFETY: the caller guarantees all eight reads; `out` is 64 writable bytes.
    unsafe {
        let v0 = lane_pair(base, base.add(stride));
        let v1 = lane_pair(base.add(2 * stride), base.add(3 * stride));
        let v2 = lane_pair(base.add(4 * stride), base.add(5 * stride));
        let v3 = lane_pair(base.add(6 * stride), base.add(7 * stride));
        let [y0, y1, y2, y3] =
            crate::zerocheck::univariate_skip_optimized::bit_transpose_64bytes_neon_regs(v0, v1, v2, v3);
        let out_ptr = out.as_mut_ptr();
        vst1q_u8(out_ptr, vreinterpretq_u8_u64(y0));
        vst1q_u8(out_ptr.add(16), vreinterpretq_u8_u64(y1));
        vst1q_u8(out_ptr.add(32), vreinterpretq_u8_u64(y2));
        vst1q_u8(out_ptr.add(48), vreinterpretq_u8_u64(y3));
    }
}

/// [`transpose_8_strided_u64s_to_64_bytes`] with the bit transpose done in
/// row form and the output left in four registers (output bytes 0..16, 16..32,
/// 32..48, 48..64). Same bytes.
///
/// Each 64-bit half holds one source word, so byte `c` of every half is a row
/// of 8x8 matrix `c`. The three block swaps of the transpose (row distance 4,
/// 2, 1) pair whole halves, so they need no per-lane shift-and-mask: distance
/// 4 is one shift-insert per output register, distance 2 a shift plus a bit
/// select, distance 1 a half swap, a per-byte shift and a bit select. One
/// byte table lookup per output register then puts row `t` of matrix `c` at
/// output byte `8c + t`, the order [`transpose_8_u64s_to_64_bytes`] writes.
///
/// # Safety
/// `base.add(l * stride)` must be valid for an 8-byte read for every `l < 8`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn transpose_8_strided_u64s_to_regs(
    base: *const u64,
    stride: usize,
) -> [core::arch::aarch64::uint8x16_t; 4] {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn lane_pair(lo: *const u64, hi: *const u64) -> uint8x16_t {
        // SAFETY: the caller's contract covers both 8-byte reads.
        unsafe { vreinterpretq_u8_u64(vcombine_u64(vld1_u64(lo), vld1_u64(hi))) }
    }

    // Output byte 8c + t comes from row t (register t / 2, half t % 2), byte c.
    const IDX0: [u8; 16] = [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57];
    const IDX1: [u8; 16] = [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59];
    const IDX2: [u8; 16] = [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61];
    const IDX3: [u8; 16] = [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63];
    // Distance 1: the low half keeps bits 0x55 and takes the high half's row
    // shifted up one; the high half keeps 0xAA and takes the low row shifted
    // down one (a negative per-byte shift count shifts right).
    const KEEP1: [u8; 16] = [
        0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
        0xAA,
    ];
    const SHIFT1: [i8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1];

    #[inline(always)]
    unsafe fn swap1(v: uint8x16_t, keep: uint8x16_t, shift: int8x16_t) -> uint8x16_t {
        unsafe {
            let moved = vshlq_u8(vextq_u8::<8>(v, v), shift);
            vbslq_u8(keep, v, moved)
        }
    }

    // SAFETY: the caller guarantees all eight reads.
    unsafe {
        // Rows 0..8 = the eight source words, two per register.
        let r01 = lane_pair(base, base.add(stride));
        let r23 = lane_pair(base.add(2 * stride), base.add(3 * stride));
        let r45 = lane_pair(base.add(4 * stride), base.add(5 * stride));
        let r67 = lane_pair(base.add(6 * stride), base.add(7 * stride));

        // Distance 4: row r keeps bits 0..4 and takes row r+4's bits 0..4 as
        // its bits 4..8; row r+4 keeps bits 4..8 and takes row r's bits 4..8.
        let a01 = vsliq_n_u8::<4>(r01, r45);
        let a45 = vsriq_n_u8::<4>(r45, r01);
        let a23 = vsliq_n_u8::<4>(r23, r67);
        let a67 = vsriq_n_u8::<4>(r67, r23);

        // Distance 2: rows r and r+2 sit in the same half of two registers.
        let keep2 = vdupq_n_u8(0x33);
        let b01 = vbslq_u8(keep2, a01, vshlq_n_u8::<2>(a23));
        let b23 = vbslq_u8(keep2, vshrq_n_u8::<2>(a01), a23);
        let b45 = vbslq_u8(keep2, a45, vshlq_n_u8::<2>(a67));
        let b67 = vbslq_u8(keep2, vshrq_n_u8::<2>(a45), a67);

        // Distance 1: rows r and r+1 are the two halves of one register.
        let keep1 = vld1q_u8(KEEP1.as_ptr());
        let shift1 = vld1q_s8(SHIFT1.as_ptr());
        let c01 = swap1(b01, keep1, shift1);
        let c23 = swap1(b23, keep1, shift1);
        let c45 = swap1(b45, keep1, shift1);
        let c67 = swap1(b67, keep1, shift1);

        let rows = uint8x16x4_t(c01, c23, c45, c67);
        [
            vqtbl4q_u8(rows, vld1q_u8(IDX0.as_ptr())),
            vqtbl4q_u8(rows, vld1q_u8(IDX1.as_ptr())),
            vqtbl4q_u8(rows, vld1q_u8(IDX2.as_ptr())),
            vqtbl4q_u8(rows, vld1q_u8(IDX3.as_ptr())),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference for [`transpose_8_u64s_to_64_bytes`] — test oracle only.
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn transpose_8_u64s_to_64_bytes_scalar(lanes: &[u64; 8], out: &mut [u8]) {
        debug_assert_eq!(out.len(), 64);
        for c in 0..8 {
            let shift = c * 8;
            let mut packed: u64 = 0;
            packed |= ((lanes[0] >> shift) & 0xFF) << (0 * 8);
            packed |= ((lanes[1] >> shift) & 0xFF) << (1 * 8);
            packed |= ((lanes[2] >> shift) & 0xFF) << (2 * 8);
            packed |= ((lanes[3] >> shift) & 0xFF) << (3 * 8);
            packed |= ((lanes[4] >> shift) & 0xFF) << (4 * 8);
            packed |= ((lanes[5] >> shift) & 0xFF) << (5 * 8);
            packed |= ((lanes[6] >> shift) & 0xFF) << (6 * 8);
            packed |= ((lanes[7] >> shift) & 0xFF) << (7 * 8);
            let transposed = transpose_8x8_bits(packed);
            out[c * 8..c * 8 + 8].copy_from_slice(&transposed.to_le_bytes());
        }
    }

    /// The NEON-delegating transpose must match the scalar per-column oracle
    /// bit-for-bit on varied inputs.
    #[test]
    fn transpose_8_u64s_matches_scalar() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        for _ in 0..100 {
            let lanes: [u64; 8] = std::array::from_fn(|_| next());
            let mut fast = [0u8; 64];
            let mut oracle = [0u8; 64];
            transpose_8_u64s_to_64_bytes(&lanes, &mut fast);
            transpose_8_u64s_to_64_bytes_scalar(&lanes, &mut oracle);
            assert_eq!(fast, oracle);
        }
        // Edge patterns.
        for lanes in [[0u64; 8], [u64::MAX; 8], std::array::from_fn(|i| 1u64 << i)] {
            let mut fast = [0u8; 64];
            let mut oracle = [0u8; 64];
            transpose_8_u64s_to_64_bytes(&lanes, &mut fast);
            transpose_8_u64s_to_64_bytes_scalar(&lanes, &mut oracle);
            assert_eq!(fast, oracle, "lanes={lanes:?}");
        }
    }

    /// The strided lane-load entry matches the array entry bit-for-bit.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn strided_transpose_matches_array_transpose() {
        let mut state = 0x0DDB_A11C_AFE5_7EEDu64;
        let mut next = || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        for stride in [1usize, 2, 3, 7, 64, 256] {
            let words: Vec<u64> = (0..8 * stride + 5).map(|_| next()).collect();
            for word in 0..stride.min(5) {
                let lanes: [u64; 8] = std::array::from_fn(|l| words[l * stride + word]);
                let mut expected = [0u8; 64];
                transpose_8_u64s_to_64_bytes(&lanes, &mut expected);
                let mut actual = [0u8; 64];
                // SAFETY: the highest read is 7 * stride + word < words.len().
                unsafe {
                    transpose_8_strided_u64s_to_64_bytes(words.as_ptr().add(word), stride, &mut actual)
                };
                assert_eq!(actual, expected, "stride={stride} word={word}");
            }
        }
    }

    /// The row-form register transpose matches the strided entry and the
    /// scalar oracle bit-for-bit, on random words, single set bits and
    /// all-zero/all-one words, at several strides.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn row_form_strided_transpose_matches_oracle() {
        let mut state = 0x5EED_0F_B175_u64;
        let mut next = || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let check = |words: &[u64], stride: usize, word: usize| {
            let lanes: [u64; 8] = std::array::from_fn(|l| words[l * stride + word]);
            let mut oracle = [0u8; 64];
            transpose_8_u64s_to_64_bytes_scalar(&lanes, &mut oracle);
            let mut strided = [0u8; 64];
            let mut regs = [0u8; 64];
            // SAFETY: the highest read is 7 * stride + word < words.len().
            unsafe {
                transpose_8_strided_u64s_to_64_bytes(words.as_ptr().add(word), stride, &mut strided);
                let out = transpose_8_strided_u64s_to_regs(words.as_ptr().add(word), stride);
                for (k, v) in out.iter().enumerate() {
                    core::arch::aarch64::vst1q_u8(regs.as_mut_ptr().add(16 * k), *v);
                }
            }
            assert_eq!(regs, oracle, "stride={stride} word={word}");
            assert_eq!(regs, strided, "stride={stride} word={word}");
        };
        for stride in [1usize, 2, 3, 7, 64, 256] {
            let words: Vec<u64> = (0..8 * stride + 5).map(|_| next()).collect();
            for word in 0..stride.min(5) {
                check(&words, stride, word);
            }
        }
        let stride = 256;
        for _ in 0..40 {
            let words: Vec<u64> = (0..8 * stride).map(|_| next()).collect();
            for word in 0..stride {
                check(&words, stride, word);
            }
        }
        // Every single set bit of the 512-bit input lands where the oracle puts it.
        for bit in 0..512 {
            let mut words = [0u64; 8];
            words[bit / 64] = 1u64 << (bit % 64);
            check(&words, 1, 0);
            let mut inv = [u64::MAX; 8];
            inv[bit / 64] ^= 1u64 << (bit % 64);
            check(&inv, 1, 0);
        }
        check(&[0u64; 8], 1, 0);
        check(&[u64::MAX; 8], 1, 0);
    }

    /// Transposing twice is the identity.
    #[test]
    fn transpose_is_involution() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..256 {
            state = state.wrapping_mul(0x2545_F491_4F6C_DD1D).rotate_left(31);
            assert_eq!(transpose_8x8_bits(transpose_8x8_bits(state)), state);
        }
    }

    /// Cross-check against a naive bit-by-bit transpose of the 8×8 matrix.
    #[test]
    fn matches_naive() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..256 {
            state = state.wrapping_mul(0x2545_F491_4F6C_DD1D).rotate_left(17);
            let got = transpose_8x8_bits(state);
            let mut want = 0u64;
            for r in 0..8 {
                for c in 0..8 {
                    if (state >> (r * 8 + c)) & 1 == 1 {
                        want |= 1u64 << (c * 8 + r);
                    }
                }
            }
            assert_eq!(got, want, "input={state:016x}");
        }
    }
}

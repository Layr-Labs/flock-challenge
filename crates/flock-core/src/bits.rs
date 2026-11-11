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
    // SAFETY: the caller guarantees all eight reads.
    unsafe {
        transpose_8_row_pairs_to_regs(
            lane_pair(base, base.add(stride)),
            lane_pair(base.add(2 * stride), base.add(3 * stride)),
            lane_pair(base.add(4 * stride), base.add(5 * stride)),
            lane_pair(base.add(6 * stride), base.add(7 * stride)),
        )
    }
}

/// [`transpose_8_strided_u64s_to_regs`] of the eight words
/// `a[l * stride] & b[l * stride]`: the AND is taken in the row registers, so
/// a caller holding `a` and `b` (with `z = a AND b`) needs no copy of `z`.
/// Same bytes as transposing the ANDed words.
///
/// # Safety
/// `a.add(l * stride)` and `b.add(l * stride)` must be valid for an 8-byte
/// read for every `l < 8`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn transpose_8_strided_and_u64s_to_regs(
    a: *const u64,
    b: *const u64,
    stride: usize,
) -> [core::arch::aarch64::uint8x16_t; 4] {
    use core::arch::aarch64::vandq_u8;
    // SAFETY: the caller guarantees all sixteen reads.
    unsafe {
        let pair = |l: usize| {
            vandq_u8(
                lane_pair(a.add(l * stride), a.add((l + 1) * stride)),
                lane_pair(b.add(l * stride), b.add((l + 1) * stride)),
            )
        };
        transpose_8_row_pairs_to_regs(pair(0), pair(2), pair(4), pair(6))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn lane_pair(lo: *const u64, hi: *const u64) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    // SAFETY: the caller's contract covers both 8-byte reads.
    unsafe { vreinterpretq_u8_u64(vcombine_u64(vld1_u64(lo), vld1_u64(hi))) }
}

/// The bit transpose of [`transpose_8_strided_u64s_to_regs`] on its four row
/// registers (rows `2i` and `2i + 1` in the halves of register `i`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_8_row_pairs_to_regs(
    r01: core::arch::aarch64::uint8x16_t,
    r23: core::arch::aarch64::uint8x16_t,
    r45: core::arch::aarch64::uint8x16_t,
    r67: core::arch::aarch64::uint8x16_t,
) -> [core::arch::aarch64::uint8x16_t; 4] {
    use core::arch::aarch64::*;

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

    // SAFETY: register arithmetic and loads of the constant tables only.
    unsafe {
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

/// Kill switch for the interleaved-row stripe transpose
/// ([`transpose_8_strided_u64s_to_regs_zip`]): `FLOCK_NO_STRIPE_ZIP_TRANSPOSE=1`
/// restores the half-register form with its four four-register table lookups.
pub const ENV_NO_STRIPE_ZIP_TRANSPOSE: &str = "FLOCK_NO_STRIPE_ZIP_TRANSPOSE";

/// Whether the stripe fills use the interleaved-row transpose (default on).
pub fn stripe_zip_transpose_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os(ENV_NO_STRIPE_ZIP_TRANSPOSE).is_none_or(|v| v != "1")
    });
    *ON
}

/// Rows `lo` and `hi` byte-interleaved in one register: byte `2c` is byte `c`
/// of `*lo`, byte `2c + 1` is byte `c` of `*hi`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn lane_zip(lo: *const u64, hi: *const u64) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    // SAFETY: the caller's contract covers both 8-byte reads.
    unsafe {
        let zero = vdup_n_u64(0);
        vzip1q_u8(
            vreinterpretq_u8_u64(vcombine_u64(vld1_u64(lo), zero)),
            vreinterpretq_u8_u64(vcombine_u64(vld1_u64(hi), zero)),
        )
    }
}

/// [`transpose_8_strided_u64s_to_regs`] with the row pairs held
/// byte-interleaved, so the final byte placement is eight two-register zips
/// instead of four four-register table lookups. Same bytes.
///
/// # Safety
/// As [`transpose_8_strided_u64s_to_regs`].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn transpose_8_strided_u64s_to_regs_zip(
    base: *const u64,
    stride: usize,
) -> [core::arch::aarch64::uint8x16_t; 4] {
    // SAFETY: the caller guarantees all eight reads.
    unsafe {
        transpose_8_zipped_rows_to_regs(
            lane_zip(base, base.add(stride)),
            lane_zip(base.add(2 * stride), base.add(3 * stride)),
            lane_zip(base.add(4 * stride), base.add(5 * stride)),
            lane_zip(base.add(6 * stride), base.add(7 * stride)),
        )
    }
}

/// [`transpose_8_strided_and_u64s_to_regs`] in the interleaved-row form of
/// [`transpose_8_strided_u64s_to_regs_zip`]. Same bytes.
///
/// # Safety
/// As [`transpose_8_strided_and_u64s_to_regs`].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn transpose_8_strided_and_u64s_to_regs_zip(
    a: *const u64,
    b: *const u64,
    stride: usize,
) -> [core::arch::aarch64::uint8x16_t; 4] {
    use core::arch::aarch64::*;
    // SAFETY: the caller guarantees all sixteen reads.
    unsafe {
        let zero = vdup_n_u64(0);
        let row = |l: usize| {
            vreinterpretq_u8_u64(vcombine_u64(
                vand_u64(vld1_u64(a.add(l * stride)), vld1_u64(b.add(l * stride))),
                zero,
            ))
        };
        let pair = |l: usize| vzip1q_u8(row(l), row(l + 1));
        transpose_8_zipped_rows_to_regs(pair(0), pair(2), pair(4), pair(6))
    }
}

/// The bit transpose of [`transpose_8_row_pairs_to_regs`] on byte-interleaved
/// row pairs (register `i`: byte `2c` = row `2i` byte `c`, byte `2c + 1` =
/// row `2i + 1` byte `c`). The distance-4 and distance-2 swaps pair bytes at
/// equal positions of two registers, exactly as in the half-register form;
/// the distance-1 swap pairs the two bytes of each 16-bit lane. The result
/// rows are already interleaved in pairs, so two rounds of zips (16-bit, then
/// 32-bit lanes) put row `t` of matrix `c` at output byte `8c + t`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_8_zipped_rows_to_regs(
    i01: core::arch::aarch64::uint8x16_t,
    i23: core::arch::aarch64::uint8x16_t,
    i45: core::arch::aarch64::uint8x16_t,
    i67: core::arch::aarch64::uint8x16_t,
) -> [core::arch::aarch64::uint8x16_t; 4] {
    use core::arch::aarch64::*;

    // SAFETY: register arithmetic only.
    unsafe {
        // Distance 4.
        let a01 = vsliq_n_u8::<4>(i01, i45);
        let a45 = vsriq_n_u8::<4>(i45, i01);
        let a23 = vsliq_n_u8::<4>(i23, i67);
        let a67 = vsriq_n_u8::<4>(i67, i23);

        // Distance 2.
        let keep2 = vdupq_n_u8(0x33);
        let b01 = vbslq_u8(keep2, a01, vshlq_n_u8::<2>(a23));
        let b23 = vbslq_u8(keep2, vshrq_n_u8::<2>(a01), a23);
        let b45 = vbslq_u8(keep2, a45, vshlq_n_u8::<2>(a67));
        let b67 = vbslq_u8(keep2, vshrq_n_u8::<2>(a45), a67);

        // Distance 1: even bytes (row 2i) keep 0x55 and take the odd byte's
        // bits shifted up one; odd bytes (row 2i + 1) keep 0xAA and take the
        // even byte's bits shifted down one.
        let keep1 = vreinterpretq_u8_u16(vdupq_n_u16(0xAA55));
        let shift1 = vreinterpretq_s8_u16(vdupq_n_u16(0xFF01));
        let swap1 = |v: uint8x16_t| vbslq_u8(keep1, v, vshlq_u8(vrev16q_u8(v), shift1));
        let c01 = vreinterpretq_u16_u8(swap1(b01));
        let c23 = vreinterpretq_u16_u8(swap1(b23));
        let c45 = vreinterpretq_u16_u8(swap1(b45));
        let c67 = vreinterpretq_u16_u8(swap1(b67));

        // Rows 0..4 and rows 4..8 of matrices 0..4 (lo) and 4..8 (hi).
        let q_lo = vreinterpretq_u32_u16(vzip1q_u16(c01, c23));
        let q_hi = vreinterpretq_u32_u16(vzip2q_u16(c01, c23));
        let p_lo = vreinterpretq_u32_u16(vzip1q_u16(c45, c67));
        let p_hi = vreinterpretq_u32_u16(vzip2q_u16(c45, c67));
        [
            vreinterpretq_u8_u32(vzip1q_u32(q_lo, p_lo)),
            vreinterpretq_u8_u32(vzip2q_u32(q_lo, p_lo)),
            vreinterpretq_u8_u32(vzip1q_u32(q_hi, p_hi)),
            vreinterpretq_u8_u32(vzip2q_u32(q_hi, p_hi)),
        ]
    }
}

/// Two adjacent stripe rows at once: the transposes of the eight words
/// `a[l * stride] & b[l * stride]` (registers 0..4 of the result) and of the
/// eight words `a[l * stride + 1] & b[l * stride + 1]` (registers 4..8), each
/// exactly the bytes of [`transpose_8_strided_and_u64s_to_regs`].
///
/// # Safety
/// `a.add(l * stride)` and `b.add(l * stride)` must be valid for a 16-byte
/// read for every `l < 8`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn transpose_8_strided_and_u64_pairs_to_regs(
    a: *const u64,
    b: *const u64,
    stride: usize,
) -> [core::arch::aarch64::uint8x16_t; 8] {
    use core::arch::aarch64::*;
    // SAFETY: the caller guarantees all sixteen 16-byte reads.
    unsafe {
        let row = |l: usize| {
            vandq_u8(
                vld1q_u8(a.add(l * stride).cast::<u8>()),
                vld1q_u8(b.add(l * stride).cast::<u8>()),
            )
        };
        transpose_8_word_pair_rows_to_regs([
            row(0),
            row(1),
            row(2),
            row(3),
            row(4),
            row(5),
            row(6),
            row(7),
        ])
    }
}

/// [`transpose_8_strided_and_u64_pairs_to_regs`] for two rows that are not
/// both whole words. With `z[i] = a[i] & b[i]` and `mask = [m0, m1]`, block
/// `l` contributes `(z[l * stride] & m0) | (z[l * stride + 1] & !m0)` to the
/// first row (registers 0..4) and
/// `(z[l * stride + 1] & m1) | (z[l * stride + 2] & !m1)` to the second
/// (registers 4..8): one bit select between the block's two overlapping
/// 16-byte word pairs, then the same transpose.
///
/// # Safety
/// `a.add(l * stride)` and `b.add(l * stride)` must be valid for a 24-byte
/// read for every `l < 8`; `mask` for a 16-byte read.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn transpose_8_strided_and_u64_pairs_select_to_regs(
    a: *const u64,
    b: *const u64,
    stride: usize,
    mask: *const u64,
) -> [core::arch::aarch64::uint8x16_t; 8] {
    use core::arch::aarch64::*;
    // SAFETY: the caller guarantees all thirty-two 16-byte reads and the mask.
    unsafe {
        let m = vld1q_u8(mask.cast::<u8>());
        let row = |l: usize| {
            let first = vandq_u8(
                vld1q_u8(a.add(l * stride).cast::<u8>()),
                vld1q_u8(b.add(l * stride).cast::<u8>()),
            );
            let second = vandq_u8(
                vld1q_u8(a.add(l * stride + 1).cast::<u8>()),
                vld1q_u8(b.add(l * stride + 1).cast::<u8>()),
            );
            vbslq_u8(m, first, second)
        };
        transpose_8_word_pair_rows_to_regs([
            row(0),
            row(1),
            row(2),
            row(3),
            row(4),
            row(5),
            row(6),
            row(7),
        ])
    }
}

/// The bit transposes of two stripe rows held as eight registers, register
/// `l` carrying block `l`'s word of the first row in its low half and of the
/// second row in its high half. Result registers 0..4 are the first row's 64
/// bytes, 4..8 the second row's, in the byte order of
/// [`transpose_8_u64s_to_64_bytes`].
///
/// Every block is a whole register, so all three block swaps pair whole
/// registers: distance 4 is one shift-insert per register, distances 2 and 1
/// a shift plus a bit select. Three rounds of zips (bytes, 16-bit, 32-bit
/// lanes) then place row `t` of matrix `c` at output byte `8c + t` of its
/// word.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_8_word_pair_rows_to_regs(
    v: [core::arch::aarch64::uint8x16_t; 8],
) -> [core::arch::aarch64::uint8x16_t; 8] {
    use core::arch::aarch64::*;
    // SAFETY: register arithmetic only.
    unsafe {
        let [v0, v1, v2, v3, v4, v5, v6, v7] = v;

        // Distance 4: row r keeps bits 0..4 and takes row r+4's bits 0..4 as
        // its bits 4..8; row r+4 keeps bits 4..8 and takes row r's bits 4..8.
        let x0 = vsliq_n_u8::<4>(v0, v4);
        let x4 = vsriq_n_u8::<4>(v4, v0);
        let x1 = vsliq_n_u8::<4>(v1, v5);
        let x5 = vsriq_n_u8::<4>(v5, v1);
        let x2 = vsliq_n_u8::<4>(v2, v6);
        let x6 = vsriq_n_u8::<4>(v6, v2);
        let x3 = vsliq_n_u8::<4>(v3, v7);
        let x7 = vsriq_n_u8::<4>(v7, v3);

        // Distance 2: row r keeps 0x33 and takes row r+2's 0x33 bits shifted
        // up two; row r+2 keeps 0xCC and takes row r's 0xCC bits shifted down.
        let keep2 = vdupq_n_u8(0x33);
        let y0 = vbslq_u8(keep2, x0, vshlq_n_u8::<2>(x2));
        let y2 = vbslq_u8(keep2, vshrq_n_u8::<2>(x0), x2);
        let y1 = vbslq_u8(keep2, x1, vshlq_n_u8::<2>(x3));
        let y3 = vbslq_u8(keep2, vshrq_n_u8::<2>(x1), x3);
        let y4 = vbslq_u8(keep2, x4, vshlq_n_u8::<2>(x6));
        let y6 = vbslq_u8(keep2, vshrq_n_u8::<2>(x4), x6);
        let y5 = vbslq_u8(keep2, x5, vshlq_n_u8::<2>(x7));
        let y7 = vbslq_u8(keep2, vshrq_n_u8::<2>(x5), x7);

        // Distance 1: row r keeps 0x55 and takes row r+1's 0x55 bits shifted
        // up one; row r+1 keeps 0xAA and takes row r's 0xAA bits shifted down.
        let keep1 = vdupq_n_u8(0x55);
        let z0 = vbslq_u8(keep1, y0, vshlq_n_u8::<1>(y1));
        let z1 = vbslq_u8(keep1, vshrq_n_u8::<1>(y0), y1);
        let z2 = vbslq_u8(keep1, y2, vshlq_n_u8::<1>(y3));
        let z3 = vbslq_u8(keep1, vshrq_n_u8::<1>(y2), y3);
        let z4 = vbslq_u8(keep1, y4, vshlq_n_u8::<1>(y5));
        let z5 = vbslq_u8(keep1, vshrq_n_u8::<1>(y4), y5);
        let z6 = vbslq_u8(keep1, y6, vshlq_n_u8::<1>(y7));
        let z7 = vbslq_u8(keep1, vshrq_n_u8::<1>(y6), y7);

        // Bytes: rows (2i, 2i + 1) interleaved, first row's word (zip1) and
        // second row's word (zip2).
        let p = |lo: uint8x16_t, hi: uint8x16_t| {
            (
                vreinterpretq_u16_u8(vzip1q_u8(lo, hi)),
                vreinterpretq_u16_u8(vzip2q_u8(lo, hi)),
            )
        };
        let (p0, s0) = p(z0, z1);
        let (p1, s1) = p(z2, z3);
        let (p2, s2) = p(z4, z5);
        let (p3, s3) = p(z6, z7);
        // 16-bit lanes: rows 0..4 and rows 4..8 of matrices 0..4 / 4..8;
        // 32-bit lanes: all eight rows.
        let word = |p0: uint16x8_t, p1: uint16x8_t, p2: uint16x8_t, p3: uint16x8_t| {
            let q_lo = vreinterpretq_u32_u16(vzip1q_u16(p0, p1));
            let q_hi = vreinterpretq_u32_u16(vzip2q_u16(p0, p1));
            let r_lo = vreinterpretq_u32_u16(vzip1q_u16(p2, p3));
            let r_hi = vreinterpretq_u32_u16(vzip2q_u16(p2, p3));
            [
                vreinterpretq_u8_u32(vzip1q_u32(q_lo, r_lo)),
                vreinterpretq_u8_u32(vzip2q_u32(q_lo, r_lo)),
                vreinterpretq_u8_u32(vzip1q_u32(q_hi, r_hi)),
                vreinterpretq_u8_u32(vzip2q_u32(q_hi, r_hi)),
            ]
        };
        let [o0, o1, o2, o3] = word(p0, p1, p2, p3);
        let [o4, o5, o6, o7] = word(s0, s1, s2, s3);
        [o0, o1, o2, o3, o4, o5, o6, o7]
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
            // AND form: split each word w into a = w | m, b = w | !m for a
            // pseudo-random mask m, so a AND b = w.
            let m: Vec<u64> = words.iter().map(|w| w.rotate_left(17) ^ 0xA5A5_5A5A_0FF0_F00F).collect();
            let a: Vec<u64> = words.iter().zip(&m).map(|(w, m)| w | m).collect();
            let b: Vec<u64> = words.iter().zip(&m).map(|(w, m)| w | !m).collect();
            let mut anded = [0u8; 64];
            // SAFETY: as above, `a` and `b` have the length of `words`.
            unsafe {
                let out = transpose_8_strided_and_u64s_to_regs(a.as_ptr().add(word), b.as_ptr().add(word), stride);
                for (k, v) in out.iter().enumerate() {
                    core::arch::aarch64::vst1q_u8(anded.as_mut_ptr().add(16 * k), *v);
                }
            }
            assert_eq!(anded, oracle, "and form stride={stride} word={word}");
            // Interleaved-row (zip) forms: the same 64 bytes.
            let mut zip = [0u8; 64];
            let mut zip_and = [0u8; 64];
            // SAFETY: as above.
            unsafe {
                let out = transpose_8_strided_u64s_to_regs_zip(words.as_ptr().add(word), stride);
                for (k, v) in out.iter().enumerate() {
                    core::arch::aarch64::vst1q_u8(zip.as_mut_ptr().add(16 * k), *v);
                }
                let out =
                    transpose_8_strided_and_u64s_to_regs_zip(a.as_ptr().add(word), b.as_ptr().add(word), stride);
                for (k, v) in out.iter().enumerate() {
                    core::arch::aarch64::vst1q_u8(zip_and.as_mut_ptr().add(16 * k), *v);
                }
            }
            // Pair form: this word and the next one of every block (only when
            // the next word of the last block exists).
            if 7 * stride + word + 1 < words.len() {
                let lanes1: [u64; 8] = std::array::from_fn(|l| words[l * stride + word + 1]);
                let mut oracle1 = [0u8; 64];
                transpose_8_u64s_to_64_bytes_scalar(&lanes1, &mut oracle1);
                let mut pair = [0u8; 128];
                // SAFETY: the highest read ends at word 7 * stride + word + 1.
                unsafe {
                    let out = transpose_8_strided_and_u64_pairs_to_regs(
                        a.as_ptr().add(word),
                        b.as_ptr().add(word),
                        stride,
                    );
                    for (k, v) in out.iter().enumerate() {
                        core::arch::aarch64::vst1q_u8(pair.as_mut_ptr().add(16 * k), *v);
                    }
                }
                assert_eq!(pair[..64], oracle, "pair form, first word, stride={stride} word={word}");
                assert_eq!(pair[64..], oracle1, "pair form, second word, stride={stride} word={word}");
            }
            // Select form: two rows cut from three adjacent words by masks.
            if 7 * stride + word + 2 < words.len() {
                for (m0, m1) in [
                    (u64::MAX, u64::MAX),
                    (0, 0),
                    (0x0000_0000_03ff_ffff, 0),
                    (u64::MAX, 0x0000_1fff_ffff_ffff),
                    (0x1, 0x7fff_ffff_ffff_ffff),
                    (words[0], !words[0].rotate_left(9)),
                ] {
                    let z = |l: usize, k: usize| words[l * stride + word + k];
                    let r0: [u64; 8] = std::array::from_fn(|l| (z(l, 0) & m0) | (z(l, 1) & !m0));
                    let r1: [u64; 8] = std::array::from_fn(|l| (z(l, 1) & m1) | (z(l, 2) & !m1));
                    let (mut o0, mut o1) = ([0u8; 64], [0u8; 64]);
                    transpose_8_u64s_to_64_bytes_scalar(&r0, &mut o0);
                    transpose_8_u64s_to_64_bytes_scalar(&r1, &mut o1);
                    let mask = [m0, m1];
                    let mut got = [0u8; 128];
                    // SAFETY: the highest read ends at word 7 * stride + word + 2.
                    unsafe {
                        let out = transpose_8_strided_and_u64_pairs_select_to_regs(
                            a.as_ptr().add(word),
                            b.as_ptr().add(word),
                            stride,
                            mask.as_ptr(),
                        );
                        for (k, v) in out.iter().enumerate() {
                            core::arch::aarch64::vst1q_u8(got.as_mut_ptr().add(16 * k), *v);
                        }
                    }
                    assert_eq!(got[..64], o0, "select form, first row, stride={stride} word={word} m0={m0:#x}");
                    assert_eq!(got[64..], o1, "select form, second row, stride={stride} word={word} m1={m1:#x}");
                }
            }
            assert_eq!(zip, oracle, "zip form stride={stride} word={word}");
            assert_eq!(zip_and, oracle, "zip and form stride={stride} word={word}");
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
    /// One P-core, L1-resident: the AND-form stripe transpose over one group
    /// (eight 256-word blocks of a and of b, 176 rows, 64 output bytes each),
    /// half-register form against interleaved-row form, alternated in one
    /// process (ignored timing probe). Outputs are compared.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "timing probe"]
    fn stripe_transpose_zip_timing() {
        use core::arch::aarch64::*;
        const STRIDE: usize = 256;
        const ROWS: usize = 176;
        let rounds: usize = std::env::var("PROBE_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(101);
        let mut x = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let a: Vec<u64> = (0..8 * STRIDE).map(|_| next()).collect();
        let b: Vec<u64> = (0..8 * STRIDE).map(|_| next()).collect();
        let mut out_old = vec![0u8; ROWS * 64];
        let mut out_new = vec![0u8; ROWS * 64];
        let mut out_pair = vec![0u8; ROWS * 64];
        let mut pair_ns: Vec<f64> = Vec::new();
        #[inline(never)]
        fn run_pairs(a: &[u64], b: &[u64], out: &mut [u8], reps: usize) {
            for _ in 0..reps {
                for j in (0..ROWS).step_by(2) {
                    // SAFETY: words j, j + 1 of all eight blocks; 128 bytes of `out` per pair.
                    unsafe {
                        let y = transpose_8_strided_and_u64_pairs_to_regs(a.as_ptr().add(j), b.as_ptr().add(j), STRIDE);
                        let d = out.as_mut_ptr().add(64 * j);
                        for (k, v) in y.iter().enumerate() {
                            vst1q_u8(d.add(16 * k), *v);
                        }
                    }
                }
                std::hint::black_box(&mut *out);
            }
        }
        #[inline(never)]
        fn run<const ZIP: bool>(a: &[u64], b: &[u64], out: &mut [u8], reps: usize) {
            for _ in 0..reps {
                for j in 0..ROWS {
                    // SAFETY: word j + 7 * STRIDE < 8 * STRIDE; 64 bytes of `out` per row.
                    unsafe {
                        let [y0, y1, y2, y3] = if ZIP {
                            transpose_8_strided_and_u64s_to_regs_zip(a.as_ptr().add(j), b.as_ptr().add(j), STRIDE)
                        } else {
                            transpose_8_strided_and_u64s_to_regs(a.as_ptr().add(j), b.as_ptr().add(j), STRIDE)
                        };
                        let d = out.as_mut_ptr().add(64 * j);
                        vst1q_u8(d, y0);
                        vst1q_u8(d.add(16), y1);
                        vst1q_u8(d.add(32), y2);
                        vst1q_u8(d.add(48), y3);
                    }
                }
                std::hint::black_box(&mut *out);
            }
        }
        const REPS: usize = 2000;
        let (mut old_ns, mut new_ns, mut lower) = (Vec::new(), Vec::new(), 0usize);
        for round in 0..rounds {
            let mut t = [0f64; 2];
            for k in 0..2 {
                let zip = (k == 1) ^ (round % 2 == 1);
                let t0 = std::time::Instant::now();
                if zip {
                    run::<true>(&a, &b, &mut out_new, REPS);
                } else {
                    run::<false>(&a, &b, &mut out_old, REPS);
                }
                t[usize::from(zip)] = t0.elapsed().as_nanos() as f64 / (REPS * ROWS) as f64;
            }
            assert_eq!(out_old, out_new);
            let t0 = std::time::Instant::now();
            run_pairs(&a, &b, &mut out_pair, REPS);
            pair_ns.push(t0.elapsed().as_nanos() as f64 / (REPS * ROWS) as f64);
            assert_eq!(out_old, out_pair);
            old_ns.push(t[0]);
            new_ns.push(t[1]);
            lower += usize::from(t[1] < t[0]);
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|p, q| p.partial_cmp(q).unwrap());
            (v[v.len() / 4], v[v.len() / 2])
        };
        let (oq, om) = med(&mut old_ns);
        let (nq, nm) = med(&mut new_ns);
        let (pq, pm) = med(&mut pair_ns);
        eprintln!(
            "stripe transpose probe: half-register {om:.3} ns/row (q25 {oq:.3}), interleaved {nm:.3} ns/row (q25 {nq:.3}), {:+.1}% at the median, lower in {lower} of {rounds}; two rows per pass {pm:.3} ns/row (q25 {pq:.3}), {:+.1}%",
            (nm / om - 1.0) * 100.0,
            (pm / om - 1.0) * 100.0
        );
    }
}

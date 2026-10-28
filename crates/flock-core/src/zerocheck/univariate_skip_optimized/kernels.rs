use super::{F8, InvNttTableByteSingleGf8};

mod portable;

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) use portable::bit_transpose_64bytes_scalar;
#[cfg(all(test, any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(super) use portable::shift_reduce_inner_ab_scalar;

#[cfg(target_arch = "aarch64")]
pub(super) mod aarch64;

#[cfg(target_arch = "x86_64")]
pub(super) mod x86_64;

#[inline(always)]
pub(super) fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        aarch64::bit_transpose_64bytes_neon(input, output);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    ))]
    unsafe {
        x86_64::bit_transpose_64bytes_avx512(input, output);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi"
        ))
    ))]
    unsafe {
        if crate::gf2_kernels::k0_has_avx512_gfni() {
            x86_64::bit_transpose_64bytes_avx512(input, output);
            return;
        }
        use core::arch::x86_64::*;
        let p = input.as_ptr();
        let a01 = _mm_loadu_si128(p as *const __m128i);
        let a23 = _mm_loadu_si128(p.add(16) as *const __m128i);
        let a45 = _mm_loadu_si128(p.add(32) as *const __m128i);
        let a67 = _mm_loadu_si128(p.add(48) as *const __m128i);
        let p01 = _mm_unpacklo_epi8(a01, _mm_srli_si128::<8>(a01));
        let p23 = _mm_unpacklo_epi8(a23, _mm_srli_si128::<8>(a23));
        let p45 = _mm_unpacklo_epi8(a45, _mm_srli_si128::<8>(a45));
        let p67 = _mm_unpacklo_epi8(a67, _mm_srli_si128::<8>(a67));
        let q0123_lo = _mm_unpacklo_epi16(p01, p23);
        let q0123_hi = _mm_unpackhi_epi16(p01, p23);
        let q4567_lo = _mm_unpacklo_epi16(p45, p67);
        let q4567_hi = _mm_unpackhi_epi16(p45, p67);
        let c01 = _mm_unpacklo_epi32(q0123_lo, q4567_lo);
        let c23 = _mm_unpackhi_epi32(q0123_lo, q4567_lo);
        let c45 = _mm_unpacklo_epi32(q0123_hi, q4567_hi);
        let c67 = _mm_unpackhi_epi32(q0123_hi, q4567_hi);
        let mut y0 = _mm256_set_m128i(c23, c01);
        let mut y1 = _mm256_set_m128i(c67, c45);
        let m1 = _mm256_set1_epi64x(0x00AA00AA00AA00AAu64 as i64);
        let m2 = _mm256_set1_epi64x(0x0000CCCC0000CCCCu64 as i64);
        let m3 = _mm256_set1_epi64x(0x00000000F0F0F0F0u64 as i64);
        let t0 = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<7>(y0)), m1);
        let t1 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<7>(y1)), m1);
        y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t0, _mm256_slli_epi64::<7>(t0)));
        y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t1, _mm256_slli_epi64::<7>(t1)));
        let t0 = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<14>(y0)), m2);
        let t1 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<14>(y1)), m2);
        y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t0, _mm256_slli_epi64::<14>(t0)));
        y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t1, _mm256_slli_epi64::<14>(t1)));
        let t0 = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<28>(y0)), m3);
        let t1 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<28>(y1)), m3);
        y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t0, _mm256_slli_epi64::<28>(t0)));
        y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t1, _mm256_slli_epi64::<28>(t1)));
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, y0);
        _mm256_storeu_si256(output.as_mut_ptr().add(32) as *mut __m256i, y1);
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx2"),
        not(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi"
        ))
    ))]
    unsafe {
        crate::gf2_kernels::k0_f128_bit_transpose_64bytes(input.as_ptr(), output.as_mut_ptr());
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    portable::bit_transpose_64bytes_scalar(input, output);
}

#[inline(always)]
pub(super) fn bit_transpose_and_pack_c_pair_64bytes(
    c0: &[u8; 64],
    c1: Option<&[u8; 64]>,
    out_bank0: &mut [u8; 64],
    out_bank1: &mut [u8; 64],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let a01 = _mm_loadu_si128(c0.as_ptr() as *const __m128i);
        let a23 = _mm_loadu_si128(c0.as_ptr().add(16) as *const __m128i);
        let a45 = _mm_loadu_si128(c0.as_ptr().add(32) as *const __m128i);
        let a67 = _mm_loadu_si128(c0.as_ptr().add(48) as *const __m128i);
        let (b01, b23, b45, b67) = match c1 {
            Some(p1) => (
                _mm_loadu_si128(p1.as_ptr() as *const __m128i),
                _mm_loadu_si128(p1.as_ptr().add(16) as *const __m128i),
                _mm_loadu_si128(p1.as_ptr().add(32) as *const __m128i),
                _mm_loadu_si128(p1.as_ptr().add(48) as *const __m128i),
            ),
            None => {
                let z = _mm_setzero_si128();
                (z, z, z, z)
            }
        };
        let m1 = _mm256_set1_epi64x(0x00AA00AA00AA00AAu64 as i64);
        let m2 = _mm256_set1_epi64x(0x0000CCCC0000CCCCu64 as i64);
        let m3 = _mm256_set1_epi64x(0x00000000F0F0F0F0u64 as i64);

        macro_rules! finish_transpose_64 {
            ($p01:expr, $p23:expr, $p45:expr, $p67:expr, $out:expr) => {{
                let q0123_lo = _mm_unpacklo_epi16($p01, $p23);
                let q0123_hi = _mm_unpackhi_epi16($p01, $p23);
                let q4567_lo = _mm_unpacklo_epi16($p45, $p67);
                let q4567_hi = _mm_unpackhi_epi16($p45, $p67);
                let c01 = _mm_unpacklo_epi32(q0123_lo, q4567_lo);
                let c23 = _mm_unpackhi_epi32(q0123_lo, q4567_lo);
                let c45 = _mm_unpacklo_epi32(q0123_hi, q4567_hi);
                let c67 = _mm_unpackhi_epi32(q0123_hi, q4567_hi);
                let mut y0 = _mm256_set_m128i(c23, c01);
                let mut y1 = _mm256_set_m128i(c67, c45);
                let t0 = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<7>(y0)), m1);
                let t1 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<7>(y1)), m1);
                y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t0, _mm256_slli_epi64::<7>(t0)));
                y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t1, _mm256_slli_epi64::<7>(t1)));
                let t0 = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<14>(y0)), m2);
                let t1 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<14>(y1)), m2);
                y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t0, _mm256_slli_epi64::<14>(t0)));
                y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t1, _mm256_slli_epi64::<14>(t1)));
                let t0 = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<28>(y0)), m3);
                let t1 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<28>(y1)), m3);
                y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t0, _mm256_slli_epi64::<28>(t0)));
                y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t1, _mm256_slli_epi64::<28>(t1)));
                _mm256_storeu_si256($out.as_mut_ptr() as *mut __m256i, y0);
                _mm256_storeu_si256($out.as_mut_ptr().add(32) as *mut __m256i, y1);
            }};
        }

        finish_transpose_64!(
            _mm_unpacklo_epi8(a01, b01),
            _mm_unpacklo_epi8(a23, b23),
            _mm_unpacklo_epi8(a45, b45),
            _mm_unpacklo_epi8(a67, b67),
            out_bank0
        );
        finish_transpose_64!(
            _mm_unpackhi_epi8(a01, b01),
            _mm_unpackhi_epi8(a23, b23),
            _mm_unpackhi_epi8(a45, b45),
            _mm_unpackhi_epi8(a67, b67),
            out_bank1
        );
        return;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let a01 = vld1q_u8(c0.as_ptr());
        let a23 = vld1q_u8(c0.as_ptr().add(16));
        let a45 = vld1q_u8(c0.as_ptr().add(32));
        let a67 = vld1q_u8(c0.as_ptr().add(48));
        let (b01, b23, b45, b67) = match c1 {
            Some(p1) => (
                vld1q_u8(p1.as_ptr()),
                vld1q_u8(p1.as_ptr().add(16)),
                vld1q_u8(p1.as_ptr().add(32)),
                vld1q_u8(p1.as_ptr().add(48)),
            ),
            None => {
                let z = vdupq_n_u8(0);
                (z, z, z, z)
            }
        };
        let m1 = vdupq_n_u64(0x00AA00AA00AA00AAu64);
        let m2 = vdupq_n_u64(0x0000CCCC0000CCCCu64);
        let m3 = vdupq_n_u64(0x00000000F0F0F0F0u64);

        macro_rules! finish_transpose_64_neon {
            ($p01:expr, $p23:expr, $p45:expr, $p67:expr, $out:expr) => {{
                let q0123_lo = vzip1q_u16(vreinterpretq_u16_u8($p01), vreinterpretq_u16_u8($p23));
                let q0123_hi = vzip2q_u16(vreinterpretq_u16_u8($p01), vreinterpretq_u16_u8($p23));
                let q4567_lo = vzip1q_u16(vreinterpretq_u16_u8($p45), vreinterpretq_u16_u8($p67));
                let q4567_hi = vzip2q_u16(vreinterpretq_u16_u8($p45), vreinterpretq_u16_u8($p67));
                let mut c01 = vreinterpretq_u64_u32(vzip1q_u32(
                    vreinterpretq_u32_u16(q0123_lo),
                    vreinterpretq_u32_u16(q4567_lo),
                ));
                let mut c23 = vreinterpretq_u64_u32(vzip2q_u32(
                    vreinterpretq_u32_u16(q0123_lo),
                    vreinterpretq_u32_u16(q4567_lo),
                ));
                let mut c45 = vreinterpretq_u64_u32(vzip1q_u32(
                    vreinterpretq_u32_u16(q0123_hi),
                    vreinterpretq_u32_u16(q4567_hi),
                ));
                let mut c67 = vreinterpretq_u64_u32(vzip2q_u32(
                    vreinterpretq_u32_u16(q0123_hi),
                    vreinterpretq_u32_u16(q4567_hi),
                ));
                macro_rules! bfly {
                    ($v:ident) => {{
                        let t = vandq_u64(veorq_u64($v, vshrq_n_u64::<7>($v)), m1);
                        $v = crate::gf2_kernels::k0_xor3_neon($v, t, vshlq_n_u64::<7>(t));
                        let t = vandq_u64(veorq_u64($v, vshrq_n_u64::<14>($v)), m2);
                        $v = crate::gf2_kernels::k0_xor3_neon($v, t, vshlq_n_u64::<14>(t));
                        let t = vandq_u64(veorq_u64($v, vshrq_n_u64::<28>($v)), m3);
                        $v = crate::gf2_kernels::k0_xor3_neon($v, t, vshlq_n_u64::<28>(t));
                    }};
                }
                bfly!(c01);
                bfly!(c23);
                bfly!(c45);
                bfly!(c67);
                let dst = $out.as_mut_ptr();
                vst1q_u8(dst, vreinterpretq_u8_u64(c01));
                vst1q_u8(dst.add(16), vreinterpretq_u8_u64(c23));
                vst1q_u8(dst.add(32), vreinterpretq_u8_u64(c45));
                vst1q_u8(dst.add(48), vreinterpretq_u8_u64(c67));
            }};
        }

        finish_transpose_64_neon!(
            vzip1q_u8(a01, b01),
            vzip1q_u8(a23, b23),
            vzip1q_u8(a45, b45),
            vzip1q_u8(a67, b67),
            out_bank0
        );
        finish_transpose_64_neon!(
            vzip2q_u8(a01, b01),
            vzip2q_u8(a23, b23),
            vzip2q_u8(a45, b45),
            vzip2q_u8(a67, b67),
            out_bank1
        );
        return;
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    {
        let mut t0 = [0u8; 64];
        let mut t1 = [0u8; 64];
        bit_transpose_64bytes(c0, &mut t0);
        if let Some(p1) = c1 {
            bit_transpose_64bytes(p1, &mut t1);
        }
        for l in 0..64 {
            let x0 = t0[l];
            let x1 = t1[l];
            out_bank0[l] = (x0 & 0x55) | ((x1 & 0x55) << 1);
            out_bank1[l] = ((x0 & 0xaa) >> 1) | (x1 & 0xaa);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn shift_reduce_inner_ab(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (a_col, b_col);
        aarch64::shift_reduce_inner_ab_fused_neon(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
        );
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        let _ = (a_col, b_col);
        unsafe {
            x86_64::shift_reduce_inner_ab_x86_avx512(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
            );
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    unsafe {
        x86_64::shift_reduce_inner_ab_x86_sse(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            a_col,
            b_col,
        );
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "gfni")
    )))]
    {
        #[cfg(target_arch = "x86_64")]
        if crate::gf2_kernels::k0_has_avx512_gfni() {
            unsafe {
                x86_64::shift_reduce_inner_ab_x86_avx512(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    out,
                );
            }
            return;
        }
        portable::shift_reduce_inner_ab_scalar(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            a_col,
            b_col,
        );
    }
}

#[inline(always)]
pub(super) fn has_avx512_gfni() -> bool {
    crate::gf2_kernels::k0_has_avx512_gfni()
}

#[inline]
pub(super) fn build_ab_eq_fold_mats(
    eq_lo_val: super::F128,
    convert: &[super::F128],
    out_mats: &mut [[u64; 16]; 16],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        x86_64::build_ab_eq_fold_mats_x86_gfni(eq_lo_val, convert, out_mats);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (eq_lo_val, convert, out_mats);
}

#[inline]
pub(super) fn convert_ab_gfni_nomul(
    ab_rows: &[[u8; 64]],
    n_b_med: usize,
    mats: &[[u64; 16]; 16],
    out_banks: &mut [[u8; 64]; 16],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        x86_64::convert_ab_gfni_nomul(ab_rows, n_b_med, mats, out_banks);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (ab_rows, n_b_med, mats, out_banks);
}

#[inline]
pub(super) fn fold_ab_plane_banks(
    plane_banks: &[[[u8; 64]; 16]],
    weights: &[super::F128],
    n_banks: usize,
    out: &mut [super::F128; 64],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        x86_64::fold_ab_plane_banks_x86_avx512(plane_banks, weights, n_banks, out);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (plane_banks, weights, n_banks, out);
}

#[allow(dead_code, clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    partial_ab: &mut [super::KaratsubaAcc; 64],
    partial_c: &mut [super::KaratsubaAcc; 64],
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        aarch64::accumulate_convert(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    portable::accumulate_convert(
        chunk_ab_bytes,
        chunk_c_bytes,
        n_b_med,
        convert,
        eq_lo_val,
        partial_ab,
        partial_c,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert_batch(
    batch_ab_bytes: &[[[u8; 64]; 16]; super::URM_BATCH],
    batch_c_bytes: &[[[u8; 64]; 16]; super::URM_BATCH],
    batch_n_b_med: &[u8; super::URM_BATCH],
    batch_eq_lo: &[super::F128; super::URM_BATCH],
    batch_len: usize,
    convert: &[super::F128],
    partial_ab: &mut [super::KaratsubaAcc; 64],
    partial_c: &mut [super::KaratsubaAcc; 64],
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        aarch64::accumulate_convert_batch(
            batch_ab_bytes,
            batch_c_bytes,
            batch_n_b_med,
            batch_eq_lo,
            batch_len,
            convert,
            partial_ab,
            partial_c,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    portable::accumulate_convert_batch(
        batch_ab_bytes,
        batch_c_bytes,
        batch_n_b_med,
        batch_eq_lo,
        batch_len,
        convert,
        partial_ab,
        partial_c,
    );
}

#[allow(dead_code, clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    partial_ab: &mut [super::KaratsubaAcc; 64],
    partial_c_0: &mut [super::KaratsubaAcc; 64],
    partial_c_1: &mut [super::KaratsubaAcc; 64],
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        aarch64::accumulate_convert_with_s_hat_v(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c_0,
            partial_c_1,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    portable::accumulate_convert_with_s_hat_v(
        chunk_ab_bytes,
        chunk_c_bytes,
        n_b_med,
        convert,
        eq_lo_val,
        partial_ab,
        partial_c_0,
        partial_c_1,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert_batch_with_s_hat_v(
    batch_ab_bytes: &[[[u8; 64]; 16]; super::URM_BATCH],
    batch_c_bytes: &[[[u8; 64]; 16]; super::URM_BATCH],
    batch_n_b_med: &[u8; super::URM_BATCH],
    batch_eq_lo: &[super::F128; super::URM_BATCH],
    batch_len: usize,
    convert: &[super::F128],
    convert_c_pair: &[super::F128],
    partial_ab: &mut [super::KaratsubaAcc; 64],
    partial_c_0: &mut [super::KaratsubaAcc; 64],
    partial_c_1: &mut [super::KaratsubaAcc; 64],
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        aarch64::accumulate_convert_batch_with_s_hat_v(
            batch_ab_bytes,
            batch_c_bytes,
            batch_n_b_med,
            batch_eq_lo,
            batch_len,
            convert,
            convert_c_pair,
            partial_ab,
            partial_c_0,
            partial_c_1,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    portable::accumulate_convert_batch_with_s_hat_v(
        batch_ab_bytes,
        batch_c_bytes,
        batch_n_b_med,
        batch_eq_lo,
        batch_len,
        convert,
        convert_c_pair,
        partial_ab,
        partial_c_0,
        partial_c_1,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert_batch_ab_only(
    batch_ab_bytes: &[[[u8; 64]; 16]; super::URM_BATCH],
    batch_n_b_med: &[u8; super::URM_BATCH],
    batch_eq_lo: &[super::F128; super::URM_BATCH],
    batch_len: usize,
    convert: &[super::F128],
    partial_ab: &mut [super::KaratsubaAcc; 64],
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        aarch64::accumulate_convert_batch_ab_only(
            batch_ab_bytes,
            batch_n_b_med,
            batch_eq_lo,
            batch_len,
            convert,
            partial_ab,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    portable::accumulate_convert_batch_ab_only(
        batch_ab_bytes,
        batch_n_b_med,
        batch_eq_lo,
        batch_len,
        convert,
        partial_ab,
    );
}

#[cfg(target_feature = "gfni")]
use super::super::F8;
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
use super::super::N_MEDIUM;
#[cfg(target_arch = "x86_64")]
use super::super::{ELL, F128, InvNttTableByteSingleGf8, N_CHUNKS};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vbmi,avx512bw,avx512f")]
pub(crate) unsafe fn bit_transpose_64bytes_avx512(input: &[u8; 64], output: &mut [u8; 64]) {
    use core::arch::x86_64::*;
    #[rustfmt::skip]
    const IDX: [i8; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56,  1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,  3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,  5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,  7, 15, 23, 31, 39, 47, 55, 63,
    ];
    unsafe {
        let inp = _mm512_loadu_si512(input.as_ptr() as *const __m512i);
        let idx = _mm512_loadu_si512(IDX.as_ptr() as *const __m512i);
        let mut y = _mm512_permutexvar_epi8(idx, inp);

        let mask1 = _mm512_set1_epi64(0x00AA00AA00AA00AAu64 as i64);
        let mask2 = _mm512_set1_epi64(0x0000CCCC0000CCCCu64 as i64);
        let mask3 = _mm512_set1_epi64(0x00000000F0F0F0F0u64 as i64);

        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<7>(y)), mask1);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<7>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<14>(y)), mask2);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<14>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<28>(y)), mask3);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<28>(t)));

        _mm512_storeu_si512(output.as_mut_ptr() as *mut __m512i, y);
    }
}

#[inline]
#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", target_feature = "gfni"))]
#[target_feature(enable = "gfni,sse2")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_sse(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    use core::arch::x86_64::*;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    unsafe {
        let mut acc = [_mm_setzero_si128(); 4];
        for k in 0..8usize {
            let chunk_off = byte_base_b + k * N_CHUNKS;
            let bw = (b_packed.as_ptr().add(chunk_off) as *const u64).read_unaligned();
            if bw == 0 {
                continue;
            }
            let aw = (a_packed.as_ptr().add(chunk_off) as *const u64).read_unaligned();
            if aw == 0 {
                continue;
            }
            let xk = _mm_set1_epi8((1u8 << k) as i8);
            if bw == !0u64 {
                inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
                let a_ptr = a_col.as_ptr() as *const u8;
                for c in 0..4usize {
                    let av = _mm_loadu_si128(a_ptr.add(c * 16) as *const __m128i);
                    let y = if k == 0 {
                        av
                    } else {
                        _mm_gf2p8mul_epi8(av, xk)
                    };
                    acc[c] = _mm_xor_si128(acc[c], y);
                }
            } else if aw == !0u64 {
                inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
                let b_ptr = b_col.as_ptr() as *const u8;
                for c in 0..4usize {
                    let bv = _mm_loadu_si128(b_ptr.add(c * 16) as *const __m128i);
                    let y = if k == 0 {
                        bv
                    } else {
                        _mm_gf2p8mul_epi8(bv, xk)
                    };
                    acc[c] = _mm_xor_si128(acc[c], y);
                }
            } else {
                inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
                inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
                let a_ptr = a_col.as_ptr() as *const u8;
                let b_ptr = b_col.as_ptr() as *const u8;
                for c in 0..4usize {
                    let av = _mm_loadu_si128(a_ptr.add(c * 16) as *const __m128i);
                    let bv = _mm_loadu_si128(b_ptr.add(c * 16) as *const __m128i);
                    let prod = _mm_gf2p8mul_epi8(av, bv);
                    let y = if k == 0 {
                        prod
                    } else {
                        _mm_gf2p8mul_epi8(prod, xk)
                    };
                    acc[c] = _mm_xor_si128(acc[c], y);
                }
            }
        }
        let out_ptr = out.as_mut_ptr();
        for c in 0..4usize {
            _mm_storeu_si128(out_ptr.add(c * 16) as *mut __m128i, acc[c]);
        }
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    unsafe {
        crate::gf2_kernels::k0_gf8_shift_reduce_inner_ab_gfni_avx512_slice(
            a_packed.as_ptr().add(byte_base_b),
            b_packed.as_ptr().add(byte_base_b),
            inv_table.data_ptr(),
            out.as_mut_ptr(),
        );
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")]
pub(crate) unsafe fn build_ab_eq_fold_mats_x86_gfni(
    eq_lo_val: F128,
    convert: &[F128],
    out_mats: &mut [[u64; 16]; 16],
) {
    debug_assert!(convert.len() >= 16 * 256);
    unsafe {
        crate::gf2_kernels::k0_f128_build_ab_eq_fold_mats_gfni_slice(
            eq_lo_val.lo,
            eq_lo_val.hi,
            convert.as_ptr() as *const u64,
            out_mats.as_mut_ptr() as *mut u64,
        );
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,gfni")]
pub(crate) unsafe fn convert_ab_gfni_nomul(
    ab_rows: &[[u8; ELL]],
    n_b_med: usize,
    mats: &[[u64; 16]; 16],
    out_banks: &mut [[u8; 64]; 16],
) {
    debug_assert!(n_b_med <= ab_rows.len() && n_b_med <= 16);
    unsafe {
        crate::gf2_kernels::k0_gf8_urm_convert_ab_nomul_gfni_slice(
            ab_rows.as_ptr() as *const u8,
            n_b_med,
            mats.as_ptr() as *const u64,
            out_banks.as_mut_ptr() as *mut u8,
        );
    }
}

#[inline]
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn c_plane_bank_to_f128_x86_avx512(
    plane_bank: &[[u8; 64]; 16],
    out: &mut [F128; ELL],
) {
    unsafe {
        crate::gf2_kernels::k0_f128_plane_bank_to_f128_slice(
            plane_bank.as_ptr() as *const u8,
            out.as_mut_ptr() as *mut u64,
        );
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,vpclmulqdq")]
pub(crate) unsafe fn fold_ab_plane_banks_x86_avx512(
    plane_banks: &[[[u8; 64]; 16]],
    weights: &[F128],
    n_banks: usize,
    out: &mut [F128; ELL],
) {
    debug_assert!(n_banks <= plane_banks.len() && n_banks <= weights.len() && n_banks <= 8);
    unsafe {
        crate::gf2_kernels::k0_f128_fold_ab_plane_banks_slice(
            plane_banks.as_ptr() as *const u8,
            weights.as_ptr() as *const u64,
            n_banks,
            out.as_mut_ptr() as *mut u64,
        );
    }
}

#[inline]
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")]
pub(crate) unsafe fn build_gf2_c4_fold_mats_x86_gfni(
    table: &[F128],
    coeffs: &[F128; 4],
    out_bcast_mats: &mut [[u64; 8]; 64],
) {
    debug_assert!(table.len() >= 8 * 256);
    unsafe {
        crate::gf2_kernels::k0_f128_build_gf2_c4_fold_mats_gfni_slice(
            table.as_ptr() as *const u64,
            coeffs.as_ptr() as *const u64,
            out_bcast_mats.as_mut_ptr() as *mut u64,
        );
    }
}

#[inline]
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")]
pub(crate) unsafe fn fold_64_gf2_rows_gfni_c4(
    rows: &[u64; 64],
    bcast_mats: &[[u64; 8]; 64],
    out: &mut [F128; 16],
    dead_lines: u8,
) {
    unsafe {
        crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_c4_slice(
            rows.as_ptr() as *const u8,
            bcast_mats.as_ptr() as *const u64,
            out.as_mut_ptr() as *mut u64,
            dead_lines,
        );
    }
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_convert_with_s_hat_v_x86_avx512(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    chunk_c_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; ELL],
    partial_c_0: &mut [F128; ELL],
    partial_c_1: &mut [F128; ELL],
) {
    use crate::field::gf2_128::x86_64::{f128x4_set, ghash_mul_x4};
    use core::arch::x86_64::*;
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert_eq!(ELL % 4, 0);

    unsafe {
        let eq = f128x4_set(eq_lo_val, eq_lo_val, eq_lo_val, eq_lo_val);
        for lane in (0..ELL).step_by(4) {
            let mut cf_ab = [F128::ZERO; 4];
            let mut cf_c_0 = [F128::ZERO; 4];
            let mut cf_c_1 = [F128::ZERO; 4];
            for b_med in 0..n_b_med {
                let table_base = b_med * 256;
                for j in 0..4 {
                    let v_ab = chunk_ab_bytes[b_med][lane + j] as usize;
                    let v_c = chunk_c_bytes[b_med][lane + j] as usize;
                    cf_ab[j] += convert[table_base + v_ab];
                    cf_c_0[j] += convert[table_base + (v_c & 0x55)];
                    cf_c_1[j] += convert[table_base + (v_c & 0xAA)];
                }
            }

            let scaled_ab = ghash_mul_x4(f128x4_set(cf_ab[0], cf_ab[1], cf_ab[2], cf_ab[3]), eq);
            let scaled_c_0 =
                ghash_mul_x4(f128x4_set(cf_c_0[0], cf_c_0[1], cf_c_0[2], cf_c_0[3]), eq);
            let scaled_c_1 =
                ghash_mul_x4(f128x4_set(cf_c_1[0], cf_c_1[1], cf_c_1[2], cf_c_1[3]), eq);

            let ab_ptr = partial_ab.as_mut_ptr().add(lane) as *mut __m512i;
            let c0_ptr = partial_c_0.as_mut_ptr().add(lane) as *mut __m512i;
            let c1_ptr = partial_c_1.as_mut_ptr().add(lane) as *mut __m512i;
            _mm512_storeu_si512(
                ab_ptr,
                _mm512_xor_si512(_mm512_loadu_si512(ab_ptr), scaled_ab),
            );
            _mm512_storeu_si512(
                c0_ptr,
                _mm512_xor_si512(_mm512_loadu_si512(c0_ptr), scaled_c_0),
            );
            _mm512_storeu_si512(
                c1_ptr,
                _mm512_xor_si512(_mm512_loadu_si512(c1_ptr), scaled_c_1),
            );
        }
    }
}

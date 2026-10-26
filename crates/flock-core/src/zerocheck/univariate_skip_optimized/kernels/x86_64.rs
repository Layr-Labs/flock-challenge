#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
use super::super::{ELL, F128, N_MEDIUM};
#[cfg(target_feature = "gfni")]
use super::super::{F8, InvNttTableByteSingleGf8, N_CHUNKS};

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
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
            inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
            inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
            let a_ptr = a_col.as_ptr() as *const u8;
            let b_ptr = b_col.as_ptr() as *const u8;
            let xk = _mm_set1_epi8((1u8 << k) as i8);
            for c in 0..4usize {
                let av = _mm_loadu_si128(a_ptr.add(c * 16) as *const __m128i);
                let bv = _mm_loadu_si128(b_ptr.add(c * 16) as *const __m128i);

                let y = _mm_gf2p8mul_epi8(_mm_gf2p8mul_epi8(av, bv), xk);
                acc[c] = _mm_xor_si128(acc[c], y);
            }
        }
        let out_ptr = out.as_mut_ptr();
        for c in 0..4usize {
            _mm_storeu_si128(out_ptr.add(c * 16) as *mut __m128i, acc[c]);
        }
    }
}

#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    use core::arch::x86_64::*;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    unsafe {
        let mut acc = _mm512_setzero_si512();
        for k in 0..8usize {
            let off = byte_base_b + k * N_CHUNKS;
            let av = inv_table.apply_x86_avx512_register_unchecked(a_packed.as_ptr().add(off));
            let bv = inv_table.apply_x86_avx512_register_unchecked(b_packed.as_ptr().add(off));
            let product = _mm512_gf2p8mul_epi8(av, bv);

            let scaled = if k == 0 {
                product
            } else {
                _mm512_gf2p8mul_epi8(product, _mm512_set1_epi8((1u8 << k) as i8))
            };
            acc = _mm512_xor_si512(acc, scaled);
        }
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
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

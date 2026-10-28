use crate::field::F128;

#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let idx_even = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let idx_odd = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let lanes = dst.len() & !3;
        let mut t = 0;
        while t < lanes {
            let s = 2 * (base + t);
            let lo = _mm512_loadu_si512(src.as_ptr().add(s) as *const __m512i);
            let hi = _mm512_loadu_si512(src.as_ptr().add(s + 4) as *const __m512i);
            let even = _mm512_permutex2var_epi64(lo, idx_even, hi);
            let odd = _mm512_permutex2var_epi64(lo, idx_odd, hi);
            let new = _mm512_xor_si512(even, ghash_mul_x4(r_bcast, _mm512_xor_si512(even, odd)));
            _mm512_storeu_si512(dst.as_mut_ptr().add(t) as *mut __m512i, new);
            t += 4;
        }
        portable_tail(src, base, dst, r, t);
    }
}

#[inline]
fn portable_tail(src: &[F128], base: usize, dst: &mut [F128], r: F128, mut t: usize) {
    while t < dst.len() {
        let s = 2 * (base + t);
        let v0 = src[s];
        let v1 = src[s + 1];
        dst[t] = v0 + (v0 + v1) * r;
        t += 1;
    }
}

#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(super) unsafe fn fold_pairs_and_msg(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    unsafe {
        let r_bcast = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        let idx_even = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let idx_odd = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let idx_dup_even = _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0);
        let mut acc_vec = _mm512_setzero_si512();

        let len = fc.len();
        let lanes = len & !3;
        let mut t = 0;
        while t < lanes {
            let s = 2 * (base + t);
            let f_lo = _mm512_loadu_si512(f.as_ptr().add(s) as *const __m512i);
            let f_hi = _mm512_loadu_si512(f.as_ptr().add(s + 4) as *const __m512i);
            let f_even = _mm512_permutex2var_epi64(f_lo, idx_even, f_hi);
            let f_odd = _mm512_permutex2var_epi64(f_lo, idx_odd, f_hi);
            let nf = _mm512_xor_si512(f_even, ghash_mul_x4(r_bcast, _mm512_xor_si512(f_even, f_odd)));
            _mm512_storeu_si512(fc.as_mut_ptr().add(t) as *mut __m512i, nf);

            let b_lo = _mm512_loadu_si512(b.as_ptr().add(s) as *const __m512i);
            let b_hi = _mm512_loadu_si512(b.as_ptr().add(s + 4) as *const __m512i);
            let b_even = _mm512_permutex2var_epi64(b_lo, idx_even, b_hi);
            let b_odd = _mm512_permutex2var_epi64(b_lo, idx_odd, b_hi);
            let nb = _mm512_xor_si512(b_even, ghash_mul_x4(r_bcast, _mm512_xor_si512(b_even, b_odd)));
            _mm512_storeu_si512(bc.as_mut_ptr().add(t) as *mut __m512i, nb);

            let nf_prep = _mm512_xor_si512(nf, _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nf));
            let nb_prep = _mm512_xor_si512(nb, _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nb));
            acc_vec = _mm512_xor_si512(acc_vec, ghash_mul_x4(nf_prep, nb_prep));
            t += 4;
        }

        let mut acc_arr = [F128::ZERO; 4];
        _mm512_storeu_si512(acc_arr.as_mut_ptr() as *mut __m512i, acc_vec);
        let mut u0 = acc_arr[0] + acc_arr[2];
        let mut u2 = acc_arr[1] + acc_arr[3];

        portable_tail(f, base, fc, r, t);
        portable_tail(b, base, bc, r, t);
        while t + 1 < len {
            let f0 = fc[t];
            let f1 = fc[t + 1];
            let b0 = bc[t];
            let b1 = bc[t + 1];
            u0 += f0 * b0;
            u2 += (f0 + f1) * (b0 + b1);
            t += 2;
        }
        (u0, u2)
    }
}

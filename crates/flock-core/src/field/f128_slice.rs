use super::F128;

#[cfg(any(test, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
mod portable;

#[allow(dead_code)]
#[inline]
pub(crate) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    assert!(
        base <= src.len() / 2 && dst.len() <= src.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        crate::gf2_kernels::k0_f128_fold_pairs_slice(
            src.as_ptr() as *const u64,
            base,
            dst.as_mut_ptr() as *mut u64,
            dst.len(),
            r.lo,
            r.hi,
        );
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    portable::fold_pairs(src, base, dst, r);
}

#[inline]
pub(crate) fn fold_pairs_and_msg(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
) -> (F128, F128) {
    assert_eq!(fc.len(), bc.len());
    assert!(
        base <= f.len() / 2
            && fc.len() <= f.len() / 2 - base
            && base <= b.len() / 2
            && bc.len() <= b.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let (u0_lo, u0_hi, u2_lo, u2_hi) = crate::gf2_kernels::k0_f128_fold_pairs_and_msg_slice(
            f.as_ptr() as *const u64,
            b.as_ptr() as *const u64,
            base,
            fc.as_mut_ptr() as *mut u64,
            bc.as_mut_ptr() as *mut u64,
            fc.len(),
            r.lo,
            r.hi,
        );
        return (
            F128 {
                lo: u0_lo,
                hi: u0_hi,
            },
            F128 {
                lo: u2_lo,
                hi: u2_hi,
            },
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    portable::fold_pairs_and_msg(f, b, base, fc, bc, r)
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn mul_4(a: [F128; 4], b: [F128; 4]) -> [F128; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let va = _mm512_loadu_si512(a.as_ptr() as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr() as *const __m512i);
        let vr = crate::gf2_kernels::k0_f128_mul_zmm(va, vb);
        let mut out = [F128::ZERO; 4];
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, vr);
        return out;
    }

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))
    ))]
    unsafe {
        let (p0_lo, p0_hi, p1_lo, p1_hi) = crate::gf2_kernels::k0_f128_mul_vec2_raw(
            a[0].lo, a[0].hi, a[1].lo, a[1].hi, b[0].lo, b[0].hi, b[1].lo, b[1].hi,
        );
        let (p2_lo, p2_hi, p3_lo, p3_hi) = crate::gf2_kernels::k0_f128_mul_vec2_raw(
            a[2].lo, a[2].hi, a[3].lo, a[3].hi, b[2].lo, b[2].hi, b[3].lo, b[3].hi,
        );
        return [
            F128 {
                lo: p0_lo,
                hi: p0_hi,
            },
            F128 {
                lo: p1_lo,
                hi: p1_hi,
            },
            F128 {
                lo: p2_lo,
                hi: p2_hi,
            },
            F128 {
                lo: p3_lo,
                hi: p3_hi,
            },
        ];
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

#[inline]
pub fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    assert_eq!(a.len(), b.len(), "inner_product: slice lengths must match");
    let _n = a.len();

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let (lo, hi) = crate::gf2_kernels::k0_f128_inner_product_slice(
            a.as_ptr() as *const u64,
            b.as_ptr() as *const u64,
            _n,
        );
        return F128 { lo, hi };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut acc = crate::field::F256Unreduced::ZERO;
        for (&x, &y) in a.iter().zip(b.iter()) {
            acc ^= x.mul_unreduced(y);
        }
        acc.reduce()
    }
}

#[inline]
pub fn xor_assign(dst: &mut [F128], src: &[F128]) {
    assert_eq!(dst.len(), src.len(), "xor_assign: slice lengths must match");
    let _n = dst.len();

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        crate::gf2_kernels::k0_f128_xor_assign_slice(
            dst.as_mut_ptr() as *mut u64,
            src.as_ptr() as *const u64,
            _n,
        );
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

#[inline]
pub(crate) fn scalar_mul_into(dst: &mut [F128], src: &[F128], alpha: F128) {
    assert_eq!(
        dst.len(),
        src.len(),
        "scalar_mul_into: slice lengths must match"
    );
    let _n = dst.len();

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        crate::gf2_kernels::k0_f128_scalar_mul_into_slice(
            dst.as_mut_ptr() as *mut u64,
            src.as_ptr() as *const u64,
            _n,
            alpha.lo,
            alpha.hi,
        );
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = alpha * s;
    }
}

#[inline]
pub(crate) fn scalar_mul_add_assign(dst: &mut [F128], src: &[F128], alpha: F128) {
    assert_eq!(
        dst.len(),
        src.len(),
        "scalar_mul_add_assign: slice lengths must match"
    );
    let _n = dst.len();

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        crate::gf2_kernels::k0_f128_scalar_mul_add_slice(
            dst.as_mut_ptr() as *mut u64,
            src.as_ptr() as *const u64,
            _n,
            alpha.lo,
            alpha.hi,
        );
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = alpha.fma(s, *d);
    }
}

#[allow(dead_code)]
#[inline]
pub(crate) fn fold_two_and_msg_in_place(
    f: &mut Vec<F128>,
    b: &mut Vec<F128>,
    r: F128,
) -> (F128, F128) {
    assert_eq!(f.len(), b.len());
    assert!(f.len().is_multiple_of(4));

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let half = f.len() / 2;
        let (u0_lo, u0_hi, u2_lo, u2_hi) = crate::gf2_kernels::k0_f128_fold_pairs_and_msg_slice(
            f.as_ptr() as *const u64,
            b.as_ptr() as *const u64,
            0,
            f.as_mut_ptr() as *mut u64,
            b.as_mut_ptr() as *mut u64,
            half,
            r.lo,
            r.hi,
        );
        f.truncate(half);
        b.truncate(half);
        return (
            F128 {
                lo: u0_lo,
                hi: u0_hi,
            },
            F128 {
                lo: u2_lo,
                hi: u2_hi,
            },
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fold_two_and_msg_in_place_scalar(f, b, r)
}

#[allow(dead_code)]
pub(crate) fn fold_two_and_msg_in_place_scalar(
    f: &mut Vec<F128>,
    b: &mut Vec<F128>,
    r: F128,
) -> (F128, F128) {
    let half = f.len() / 2;
    let mut u0 = crate::field::F256Unreduced::ZERO;
    let mut u2 = crate::field::F256Unreduced::ZERO;
    let mut t = 0usize;
    while t < half {
        let source = 2 * t;
        let f0 = f[source] + r * (f[source] + f[source + 1]);
        let f1 = f[source + 2] + r * (f[source + 2] + f[source + 3]);
        let b0 = b[source] + r * (b[source] + b[source + 1]);
        let b1 = b[source + 2] + r * (b[source + 2] + b[source + 3]);
        f[t] = f0;
        f[t + 1] = f1;
        b[t] = b0;
        b[t + 1] = b1;
        u0 ^= f0.mul_unreduced(b0);
        u2 ^= (f0 + f1).mul_unreduced(b0 + b1);
        t += 2;
    }
    f.truncate(half);
    b.truncate(half);
    (u0.reduce(), u2.reduce())
}

#[allow(dead_code)]
#[inline]
pub(crate) fn fold4_nested(src: &[F128], dst: &mut [F128], r0: F128, r1: F128) {
    assert_eq!(
        src.len(),
        4 * dst.len(),
        "fold4 source must contain four elements for every destination slot"
    );
    for (t, value) in dst.iter_mut().enumerate() {
        let a0 = src[4 * t];
        let a1 = src[4 * t + 1];
        let a2 = src[4 * t + 2];
        let a3 = src[4 * t + 3];
        let low = a0 + r0 * (a0 + a1);
        let high = a2 + r0 * (a2 + a3);
        *value = low + r1 * (low + high);
    }
}

#[allow(dead_code)]
#[inline]
pub(crate) fn fold16_banked(src: &[F128], dst: &mut [F128], w: &[F128; 16]) {
    assert_eq!(
        src.len(),
        16 * dst.len(),
        "fold16 source must contain sixteen elements for every destination slot"
    );
    for (t, value) in dst.iter_mut().enumerate() {
        let base = 16 * t;
        let mut acc = crate::field::F256Unreduced::ZERO;
        for b in 0..16 {
            acc ^= w[b].mul_unreduced(src[base + b]);
        }
        *value = acc.reduce();
    }
}

#[allow(dead_code)]
#[inline]
pub(crate) fn fold64_banked_nested(
    src: &[F128],
    dst: &mut [F128],
    w: &[F128; 16],
    r0: F128,
    r1: F128,
) {
    assert_eq!(
        src.len(),
        64 * dst.len(),
        "fold64 source must contain sixty-four elements for every destination slot"
    );

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        crate::gf2_kernels::k0_f128_fold64_banked_nested_slice(
            src.as_ptr() as *const u64,
            dst.as_mut_ptr() as *mut u64,
            dst.len(),
            w.as_ptr() as *const u64,
            r0.lo,
            r0.hi,
            r1.lo,
            r1.hi,
        );
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for (t, value) in dst.iter_mut().enumerate() {
        let base_t = 64 * t;
        let mut m = [F128::ZERO; 4];
        for (s_idx, m_slot) in m.iter_mut().enumerate() {
            let mut acc = crate::field::F256Unreduced::ZERO;
            for b in 0..16 {
                acc ^= w[b].mul_unreduced(src[base_t + (b << 2) + s_idx]);
            }
            *m_slot = acc.reduce();
        }
        let low = m[0] + r0 * (m[0] + m[1]);
        let high = m[2] + r0 * (m[2] + m[3]);
        *value = low + r1 * (low + high);
    }
}

#[allow(dead_code)]
#[inline]
pub(crate) fn bind_split_half(lo: &mut [F128], hi: &[F128], r: F128) {
    assert!(
        hi.len() >= lo.len(),
        "split bind needs one high slot per low"
    );
    for i in 0..lo.len() {
        lo[i] = lo[i] + r * (hi[i] + lo[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_fold_matches_portable_with_offset_and_tail() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let src: Vec<F128> = (0..30)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let r = F128 {
            lo: next(),
            hi: next(),
        };
        let mut expected = vec![F128::ZERO; 9];
        let mut actual = vec![F128::ZERO; 9];

        portable::fold_pairs(&src, 3, &mut expected, r);
        fold_pairs(&src, 3, &mut actual, r);

        assert_eq!(actual, expected);
    }
}

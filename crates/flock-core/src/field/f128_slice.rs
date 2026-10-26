use super::F128;

#[cfg(any(test, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
mod portable;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod x86_64;

#[allow(dead_code)]
#[inline]
pub(crate) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    assert!(
        base <= src.len() / 2 && dst.len() <= src.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        x86_64::fold_pairs(src, base, dst, r);
        return;
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
        crate::gf2_kernels::k0_f128_fold_pairs_slice(
            src.as_ptr() as *const u64,
            base,
            dst.as_mut_ptr() as *mut u64,
            dst.len(),
            r.lo,
            r.hi,
        );
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

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        return x86_64::fold_pairs_and_msg(f, b, base, fc, bc, r);
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
        (
            F128 {
                lo: u0_lo,
                hi: u0_hi,
            },
            F128 {
                lo: u2_lo,
                hi: u2_hi,
            },
        )
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
        let vr = crate::field::gf2_128::x86_64::ghash_mul_x4(va, vb);
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
        [
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
        ]
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

#[inline]
pub fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    assert_eq!(a.len(), b.len(), "inner_product: slice lengths must match");
    let _n = a.len();

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        let mut acc4 = crate::field::gf2_128::x86_64::WideGhashX4::zero();
        let mut i = 0;
        while i + 4 <= _n {
            let va = crate::field::gf2_128::x86_64::f128x4_loadu(a.as_ptr().add(i));
            let vb = crate::field::gf2_128::x86_64::f128x4_loadu(b.as_ptr().add(i));
            acc4.mul_acc(va, vb);
            i += 4;
        }
        let mut acc = acc4.fold();
        while i < _n {
            acc ^= a[i].mul_unreduced(b[i]);
            i += 1;
        }
        return acc.reduce();
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
        let (lo, hi) = crate::gf2_kernels::k0_f128_inner_product_slice(
            a.as_ptr() as *const u64,
            b.as_ptr() as *const u64,
            _n,
        );
        F128 { lo, hi }
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

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0;
        while i + 4 <= _n {
            let vd = _mm512_loadu_si512(dst.as_ptr().add(i) as *const __m512i);
            let vs = _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i);
            _mm512_storeu_si512(
                dst.as_mut_ptr().add(i) as *mut __m512i,
                _mm512_xor_si512(vd, vs),
            );
            i += 4;
        }
        while i < _n {
            dst[i] += src[i];
            i += 1;
        }
        return;
    }

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(all(target_arch = "x86_64", target_feature = "avx512f"))
    ))]
    unsafe {
        crate::gf2_kernels::k0_f128_xor_assign_slice(
            dst.as_mut_ptr() as *mut u64,
            src.as_ptr() as *const u64,
            _n,
        );
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

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let valpha = crate::field::gf2_128::x86_64::f128x4_set(alpha, alpha, alpha, alpha);
        let mut i = 0;
        while i + 4 <= _n {
            let vs = _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i);
            let vp = crate::field::gf2_128::x86_64::ghash_mul_x4(vs, valpha);
            _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut __m512i, vp);
            i += 4;
        }
        while i < _n {
            dst[i] = alpha * src[i];
            i += 1;
        }
        return;
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
        crate::gf2_kernels::k0_f128_scalar_mul_into_slice(
            dst.as_mut_ptr() as *mut u64,
            src.as_ptr() as *const u64,
            _n,
            alpha.lo,
            alpha.hi,
        );
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

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let valpha = crate::field::gf2_128::x86_64::f128x4_set(alpha, alpha, alpha, alpha);
        let mut i = 0;
        while i + 4 <= _n {
            let vd = _mm512_loadu_si512(dst.as_ptr().add(i) as *const __m512i);
            let vs = _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i);
            let vp = crate::field::gf2_128::x86_64::ghash_mul_x4(vs, valpha);
            _mm512_storeu_si512(
                dst.as_mut_ptr().add(i) as *mut __m512i,
                _mm512_xor_si512(vd, vp),
            );
            i += 4;
        }
        while i < _n {
            dst[i] = alpha.fma(src[i], dst[i]);
            i += 1;
        }
        return;
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
        crate::gf2_kernels::k0_f128_scalar_mul_add_slice(
            dst.as_mut_ptr() as *mut u64,
            src.as_ptr() as *const u64,
            _n,
            alpha.lo,
            alpha.hi,
        );
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
        (
            F128 {
                lo: u0_lo,
                hi: u0_hi,
            },
            F128 {
                lo: u2_lo,
                hi: u2_hi,
            },
        )
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

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let src_ptr = src.as_ptr() as *const __m128i;
        let dst_ptr = dst.as_mut_ptr() as *mut __m128i;
        let mut vw = [_mm_setzero_si128(); 16];
        for b in 0..16 {
            vw[b] = _mm_loadu_si128(w.as_ptr().add(b) as *const __m128i);
        }
        let vr0 = _mm_set_epi64x(r0.hi as i64, r0.lo as i64);
        let (rr0_0, rr0_1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vr0);
        let vr1 = _mm_set_epi64x(r1.hi as i64, r1.lo as i64);
        let (rr1_0, rr1_1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vr1);

        let n = dst.len();
        let mut t = 0usize;
        while t + 2 <= n {
            let base0 = 64 * t;
            let base1 = 64 * (t + 1);

            macro_rules! fold_slot_m128i {
                ($base:expr) => {{
                    let z = _mm_setzero_si128();
                    let (mut ll0, mut mid0, mut hh0) = (z, z, z);
                    let (mut ll1, mut mid1, mut hh1) = (z, z, z);
                    let (mut ll2, mut mid2, mut hh2) = (z, z, z);
                    let (mut ll3, mut mid3, mut hh3) = (z, z, z);
                    for b in 0..16 {
                        let wb = vw[b];
                        let ptr = src_ptr.add($base + (b << 2));
                        let s0 = _mm_loadu_si128(ptr);
                        let s1 = _mm_loadu_si128(ptr.add(1));
                        let s2 = _mm_loadu_si128(ptr.add(2));
                        let s3 = _mm_loadu_si128(ptr.add(3));
                        crate::gf2_kernels::k0_f128_mul_acc_m128i(
                            wb, s0, &mut ll0, &mut mid0, &mut hh0,
                        );
                        crate::gf2_kernels::k0_f128_mul_acc_m128i(
                            wb, s1, &mut ll1, &mut mid1, &mut hh1,
                        );
                        crate::gf2_kernels::k0_f128_mul_acc_m128i(
                            wb, s2, &mut ll2, &mut mid2, &mut hh2,
                        );
                        crate::gf2_kernels::k0_f128_mul_acc_m128i(
                            wb, s3, &mut ll3, &mut mid3, &mut hh3,
                        );
                    }
                    let m0 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(ll0, mid0, hh0);
                    let m1 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(ll1, mid1, hh1);
                    let m2 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(ll2, mid2, hh2);
                    let m3 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(ll3, mid3, hh3);
                    let (p0, p2) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        _mm_xor_si128(m0, m1),
                        _mm_xor_si128(m2, m3),
                        rr0_0,
                        rr0_1,
                        rr0_0,
                        rr0_1,
                    );
                    let low = _mm_xor_si128(m0, p0);
                    let high = _mm_xor_si128(m2, p2);
                    (low, _mm_xor_si128(low, high))
                }};
            }

            let (low0, diff0) = fold_slot_m128i!(base0);
            let (low1, diff1) = fold_slot_m128i!(base1);
            let (q0, q1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                diff0, diff1, rr1_0, rr1_1, rr1_0, rr1_1,
            );
            _mm_storeu_si128(dst_ptr.add(t), _mm_xor_si128(low0, q0));
            _mm_storeu_si128(dst_ptr.add(t + 1), _mm_xor_si128(low1, q1));
            t += 2;
        }
        while t < n {
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
            dst[t] = low + r1 * (low + high);
            t += 1;
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let src_ptr = src.as_ptr() as *const u64;
        let dst_ptr = dst.as_mut_ptr() as *mut u64;
        let z = vdupq_n_u64(0);
        let mut vw = [z; 16];
        for b in 0..16 {
            vw[b] = vld1q_u64((w.as_ptr().add(b)) as *const u64);
        }
        let vr0 = vcombine_u64(vcreate_u64(r0.lo), vcreate_u64(r0.hi));
        let (rr0_0, rr0_1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vr0);
        let vr1 = vcombine_u64(vcreate_u64(r1.lo), vcreate_u64(r1.hi));
        let (rr1_0, rr1_1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vr1);

        let n = dst.len();
        let mut t = 0usize;
        while t + 2 <= n {
            let base0 = 64 * t;
            let base1 = 64 * (t + 1);

            macro_rules! fold_slot_neon {
                ($base:expr) => {{
                    let (mut ll0, mut mid0, mut hh0) = (z, z, z);
                    let (mut ll1, mut mid1, mut hh1) = (z, z, z);
                    let (mut ll2, mut mid2, mut hh2) = (z, z, z);
                    let (mut ll3, mut mid3, mut hh3) = (z, z, z);
                    for b in 0..16 {
                        let wb = vw[b];
                        let ptr = src_ptr.add(2 * ($base + (b << 2)));
                        let s0 = vld1q_u64(ptr);
                        let s1 = vld1q_u64(ptr.add(2));
                        let s2 = vld1q_u64(ptr.add(4));
                        let s3 = vld1q_u64(ptr.add(6));
                        crate::gf2_kernels::k0_f128_mul_acc_neon(
                            wb, s0, &mut ll0, &mut mid0, &mut hh0,
                        );
                        crate::gf2_kernels::k0_f128_mul_acc_neon(
                            wb, s1, &mut ll1, &mut mid1, &mut hh1,
                        );
                        crate::gf2_kernels::k0_f128_mul_acc_neon(
                            wb, s2, &mut ll2, &mut mid2, &mut hh2,
                        );
                        crate::gf2_kernels::k0_f128_mul_acc_neon(
                            wb, s3, &mut ll3, &mut mid3, &mut hh3,
                        );
                    }
                    let m0 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(ll0, mid0, hh0);
                    let m1 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(ll1, mid1, hh1);
                    let m2 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(ll2, mid2, hh2);
                    let m3 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(ll3, mid3, hh3);
                    let (p0, p2) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        veorq_u64(m0, m1),
                        veorq_u64(m2, m3),
                        rr0_0,
                        rr0_1,
                        rr0_0,
                        rr0_1,
                    );
                    let low = veorq_u64(m0, p0);
                    let high = veorq_u64(m2, p2);
                    (low, veorq_u64(low, high))
                }};
            }

            let (low0, diff0) = fold_slot_neon!(base0);
            let (low1, diff1) = fold_slot_neon!(base1);
            let (q0, q1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                diff0, diff1, rr1_0, rr1_1, rr1_0, rr1_1,
            );
            vst1q_u64(dst_ptr.add(2 * t), veorq_u64(low0, q0));
            vst1q_u64(dst_ptr.add(2 * (t + 1)), veorq_u64(low1, q1));
            t += 2;
        }
        while t < n {
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
            dst[t] = low + r1 * (low + high);
            t += 1;
        }
        return;
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes"),
    )))]
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

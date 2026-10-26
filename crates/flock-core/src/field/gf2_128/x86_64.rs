use super::{F128, F256Unreduced, ghash_reduce};
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn pmull(a: u64, b: u64) -> __m128i {
    let va = _mm_set_epi64x(0, a as i64);
    let vb = _mm_set_epi64x(0, b as i64);

    _mm_clmulepi64_si128::<0x00>(va, vb)
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane0(v: __m128i) -> u64 {
    _mm_extract_epi64::<0>(v) as u64
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn lane1(v: __m128i) -> u64 {
    _mm_extract_epi64::<1>(v) as u64
}

#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_schoolbook(a: F128, b: F128) -> F128 {
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let cr_lo = lane0(cross);
        let cr_hi = lane1(cross);

        ghash_reduce(
            lane0(p_ll),
            lane1(p_ll) ^ cr_lo,
            lane0(p_hh) ^ cr_hi,
            lane1(p_hh),
        )
    }
}

#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_binius(a: F128, b: F128) -> F128 {
    unsafe {
        let t0 = pmull(a.lo, b.lo);
        let t1a = pmull(a.lo, b.hi);
        let t1b = pmull(a.hi, b.lo);
        let t2 = pmull(a.hi, b.hi);
        let mut t1 = _mm_xor_si128(t1a, t1b);

        let t2_shifted = _mm_slli_si128::<8>(t2);
        t1 = _mm_xor_si128(t1, t2_shifted);
        let t2_red = pmull(lane1(t2), 0x87);
        t1 = _mm_xor_si128(t1, t2_red);

        let t1_shifted = _mm_slli_si128::<8>(t1);
        let mut t0 = _mm_xor_si128(t0, t1_shifted);
        let t1_red = pmull(lane1(t1), 0x87);
        t0 = _mm_xor_si128(t0, t1_red);

        F128 {
            lo: lane0(t0),
            hi: lane1(t0),
        }
    }
}

#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba(a: F128, b: F128) -> F128 {
    unsafe {
        let p0 = pmull(a.lo, b.lo);
        let p1 = pmull(a.hi, b.hi);
        let pm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);

        let p0_lo = lane0(p0);
        let p0_hi = lane1(p0);
        let p1_lo = lane0(p1);
        let p1_hi = lane1(p1);
        let pm_lo = lane0(pm);
        let pm_hi = lane1(pm);

        let cross_lo = pm_lo ^ p0_lo ^ p1_lo;
        let cross_hi = pm_hi ^ p0_hi ^ p1_hi;

        ghash_reduce(p0_lo, p0_hi ^ cross_lo, p1_lo ^ cross_hi, p1_hi)
    }
}

#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_karatsuba_barrett(a: F128, b: F128) -> F128 {
    unsafe {
        let d0 = pmull(a.lo, b.lo);
        let d2 = pmull(a.hi, b.hi);
        let dm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);
        let d1 = _mm_xor_si128(_mm_xor_si128(dm, d0), d2);

        let d0_lo = lane0(d0);
        let d0_hi = lane1(d0);
        let d1_lo = lane0(d1);
        let d1_hi = lane1(d1);
        let d2_lo = lane0(d2);
        let d2_hi = lane1(d2);

        let lo_lo = d0_lo;
        let lo_hi = d0_hi ^ d1_lo;
        let hi_lo = d2_lo ^ d1_hi;
        let hi_hi = d2_hi;

        let r_hi = pmull(hi_hi, 0x87);
        let r_lo = pmull(hi_lo, 0x87);

        let r_lo_lo = lane0(r_lo);
        let r_lo_hi = lane1(r_lo);
        let r_hi_lo = lane0(r_hi);
        let r_hi_hi = lane1(r_hi);

        let ov = r_hi_hi;
        let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);

        F128 {
            lo: lo_lo ^ r_lo_lo ^ corr,
            hi: lo_hi ^ r_lo_hi ^ r_hi_lo,
        }
    }
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn ghash_mul_m128i(x: __m128i, y: __m128i) -> __m128i {
    let poly = _mm_set1_epi64x(0x87);
    let t0 = _mm_clmulepi64_si128::<0x00>(x, y);
    let t1a = _mm_clmulepi64_si128::<0x10>(x, y);
    let t1b = _mm_clmulepi64_si128::<0x01>(x, y);
    let t2 = _mm_clmulepi64_si128::<0x11>(x, y);
    let mut t1 = _mm_xor_si128(t1a, t1b);

    let t2_shifted = _mm_bslli_si128::<8>(t2);
    t1 = _mm_xor_si128(t1, t2_shifted);
    let t2_red = _mm_clmulepi64_si128::<0x01>(t2, poly);
    t1 = _mm_xor_si128(t1, t2_red);

    let t1_shifted = _mm_bslli_si128::<8>(t1);
    let mut res = _mm_xor_si128(t0, t1_shifted);
    let t1_red = _mm_clmulepi64_si128::<0x01>(t1, poly);
    res = _mm_xor_si128(res, t1_red);
    res
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub unsafe fn ghash_mul_unreduced_x86(a: F128, b: F128) -> F256Unreduced {
    unsafe {
        let va = _mm_set_epi64x(a.hi as i64, a.lo as i64);
        let vb = _mm_set_epi64x(b.hi as i64, b.lo as i64);
        let p_ll = _mm_clmulepi64_si128::<0x00>(va, vb);
        let p_lh = _mm_clmulepi64_si128::<0x10>(va, vb);
        let p_hl = _mm_clmulepi64_si128::<0x01>(va, vb);
        let p_hh = _mm_clmulepi64_si128::<0x11>(va, vb);

        let cross = _mm_xor_si128(p_lh, p_hl);
        let lo = _mm_xor_si128(p_ll, _mm_bslli_si128::<8>(cross));
        let hi = _mm_xor_si128(p_hh, _mm_bsrli_si128::<8>(cross));

        F256Unreduced {
            r0: lane0(lo),
            r1: lane1(lo),
            r2: lane0(hi),
            r3: lane1(hi),
        }
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn ghash_poly_x4() -> __m512i {
    _mm512_set_epi64(0, 0x87, 0, 0x87, 0, 0x87, 0, 0x87)
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn gf2_128_reduce_x4(mut t0: __m512i, t1: __m512i) -> __m512i {
    unsafe {
        let poly = ghash_poly_x4();
        t0 = _mm512_xor_si512(t0, _mm512_bslli_epi128::<8>(t1));
        t0 = _mm512_xor_si512(t0, _mm512_clmulepi64_epi128::<0x01>(t1, poly));
        t0
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub unsafe fn ghash_mul_x4(x: __m512i, y: __m512i) -> __m512i {
    unsafe {
        let t1a = _mm512_clmulepi64_epi128::<0x01>(x, y);
        let t1b = _mm512_clmulepi64_epi128::<0x10>(x, y);
        let mut t1 = _mm512_xor_si512(t1a, t1b);

        let t2 = _mm512_clmulepi64_epi128::<0x11>(x, y);
        t1 = gf2_128_reduce_x4(t1, t2);

        let t0 = _mm512_clmulepi64_epi128::<0x00>(x, y);
        gf2_128_reduce_x4(t0, t1)
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_loadu(p: *const F128) -> __m512i {
    unsafe { _mm512_loadu_si512(p as *const __m512i) }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn f128x4_set(a: F128, b: F128, c: F128, d: F128) -> __m512i {
    _mm512_set_epi64(
        d.hi as i64,
        d.lo as i64,
        c.hi as i64,
        c.lo as i64,
        b.hi as i64,
        b.lo as i64,
        a.hi as i64,
        a.lo as i64,
    )
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xor4_lanes(v: __m512i) -> __m128i {
    let l0 = _mm512_extracti32x4_epi32::<0>(v);
    let l1 = _mm512_extracti32x4_epi32::<1>(v);
    let l2 = _mm512_extracti32x4_epi32::<2>(v);
    let l3 = _mm512_extracti32x4_epi32::<3>(v);
    _mm_xor_si128(_mm_xor_si128(l0, l1), _mm_xor_si128(l2, l3))
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[derive(Clone, Copy)]
pub struct WideGhashX4 {
    lo: __m512i,
    hi: __m512i,
    mid: __m512i,
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
impl WideGhashX4 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn zero() -> Self {
        let z = _mm512_setzero_si512();
        Self {
            lo: z,
            hi: z,
            mid: z,
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f,vpclmulqdq")]
    pub unsafe fn mul_acc(&mut self, x: __m512i, y: __m512i) {
        self.lo = _mm512_xor_si512(self.lo, _mm512_clmulepi64_epi128::<0x00>(x, y));
        self.hi = _mm512_xor_si512(self.hi, _mm512_clmulepi64_epi128::<0x11>(x, y));
        let m = _mm512_xor_si512(
            _mm512_clmulepi64_epi128::<0x01>(x, y),
            _mm512_clmulepi64_epi128::<0x10>(x, y),
        );
        self.mid = _mm512_xor_si512(self.mid, m);
    }

    #[inline]
    #[target_feature(enable = "avx512f,sse4.1")]
    pub unsafe fn fold(self) -> F256Unreduced {
        unsafe {
            let lo = xor4_lanes(self.lo);
            let hi = xor4_lanes(self.hi);
            let mid = xor4_lanes(self.mid);
            let lo_lo = _mm_extract_epi64::<0>(lo) as u64;
            let lo_hi = _mm_extract_epi64::<1>(lo) as u64;
            let hi_lo = _mm_extract_epi64::<0>(hi) as u64;
            let hi_hi = _mm_extract_epi64::<1>(hi) as u64;
            let mid_lo = _mm_extract_epi64::<0>(mid) as u64;
            let mid_hi = _mm_extract_epi64::<1>(mid) as u64;
            F256Unreduced {
                r0: lo_lo,
                r1: lo_hi ^ mid_lo,
                r2: hi_lo ^ mid_hi,
                r3: hi_hi,
            }
        }
    }
}

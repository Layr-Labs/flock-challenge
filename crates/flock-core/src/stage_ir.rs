#![allow(
    unused_unsafe,
    dead_code,
    unused_imports,
    unused_variables,
    unused_parens,
    non_snake_case,
    non_camel_case_types,
    clippy::all
)]
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Gf2KernelFlatSlice {
    pub data: *const core::ffi::c_void,
    pub len: usize,
    pub stride_bytes: usize,
}
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Gf2KernelMutSlice {
    pub data: *mut core::ffi::c_void,
    pub len: usize,
    pub cap: usize,
}
pub type Gf2KernelSliceOutScanFn = extern "C" fn(
    ptr: *const u8,
    len: usize,
    out_spans: *mut u64,
    cap: usize,
    state_io: *mut u64,
) -> u32;
#[inline]
pub const fn k0_pack_span(start: u32, len: u32, kind: u8) -> u64 {
    ((start as u64) << 32) | (((len & 0xFF_FFFF) as u64) << 8) | (kind as u64)
}
#[inline(always)]
pub unsafe fn k0_clmul64_lo(a: u64, b: u64) -> u64 {
    if b == 0x87 {
        return a ^ (a << 1) ^ (a << 2) ^ (a << 7);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let va = _mm_set_epi64x(0, a as i64);
        let vb = _mm_set_epi64x(0, b as i64);
        let mut product = va;
        core::arch::asm!(
            "pclmulqdq {product}, {vb}, 0",
            product = inout(xmm_reg) product,
            vb = in(xmm_reg) vb,
            options(pure, nomem, nostack, preserves_flags),
        );
        return _mm_extract_epi64::<0>(product) as u64;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let prod = vmull_p64(a, b);
        return vgetq_lane_u64::<0>(vreinterpretq_u64_p128(prod));
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    {
        let mut acc = 0u64;
        let mut i = 0u32;
        while i < 64 {
            if ((b >> i) & 1) != 0 {
                acc ^= a << i;
            }
            i += 1;
        }
        acc
    }
}
#[inline(always)]
pub unsafe fn k0_clmul64_hi(a: u64, b: u64) -> u64 {
    if b == 0x87 {
        return (a >> 63) ^ (a >> 62) ^ (a >> 57);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let va = _mm_set_epi64x(0, a as i64);
        let vb = _mm_set_epi64x(0, b as i64);
        let mut product = va;
        core::arch::asm!(
            "pclmulqdq {product}, {vb}, 0",
            product = inout(xmm_reg) product,
            vb = in(xmm_reg) vb,
            options(pure, nomem, nostack, preserves_flags),
        );
        return _mm_extract_epi64::<1>(product) as u64;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let prod = vmull_p64(a, b);
        return vgetq_lane_u64::<1>(vreinterpretq_u64_p128(prod));
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    {
        let mut acc = 0u64;
        let mut i = 1u32;
        while i < 64 {
            if ((b >> i) & 1) != 0 {
                acc ^= a >> (64 - i);
            }
            i += 1;
        }
        acc
    }
}
#[inline(always)]
pub unsafe fn k0_f128_reduce_256_raw(r0: u64, r1: u64, r2: u64, r3: u64) -> (u64, u64) {
    let s1_lo = r2 << 1;
    let s1_hi = (r3 << 1) | (r2 >> 63);
    let s2_lo = r2 << 2;
    let s2_hi = (r3 << 2) | (r2 >> 62);
    let s7_lo = r2 << 7;
    let s7_hi = (r3 << 7) | (r2 >> 57);
    let t_lo = r2 ^ s1_lo ^ s2_lo ^ s7_lo;
    let t_hi = r3 ^ s1_hi ^ s2_hi ^ s7_hi;
    let ov = (r3 >> 63) ^ (r3 >> 62) ^ (r3 >> 57);
    let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    (r0 ^ t_lo ^ corr, r1 ^ t_hi)
}
#[inline(always)]
pub unsafe fn k0_f128_mul_unreduced_raw(
    a_lo: u64,
    a_hi: u64,
    b_lo: u64,
    b_hi: u64,
) -> (u64, u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let va = _mm_set_epi64x(a_hi as i64, a_lo as i64);
        let vb = _mm_set_epi64x(b_hi as i64, b_lo as i64);
        let p_ll = _mm_clmulepi64_si128::<0x00>(va, vb);
        let p_hh = _mm_clmulepi64_si128::<0x11>(va, vb);
        let vm = _mm_set_epi64x((b_lo ^ b_hi) as i64, (a_lo ^ a_hi) as i64);
        let p_mid = _mm_clmulepi64_si128::<0x01>(vm, vm);
        let cross = _mm_xor_si128(p_mid, _mm_xor_si128(p_ll, p_hh));
        let lo = _mm_xor_si128(p_ll, _mm_bslli_si128::<8>(cross));
        let hi = _mm_xor_si128(p_hh, _mm_bsrli_si128::<8>(cross));
        return (
            _mm_extract_epi64::<0>(lo) as u64,
            _mm_extract_epi64::<1>(lo) as u64,
            _mm_extract_epi64::<0>(hi) as u64,
            _mm_extract_epi64::<1>(hi) as u64,
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let p_ll = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
        let p_hh = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));
        let p_mid = vreinterpretq_u64_p128(vmull_p64(a_lo ^ a_hi, b_lo ^ b_hi));
        let cross = veorq_u64(p_mid, veorq_u64(p_ll, p_hh));
        return (
            vgetq_lane_u64::<0>(p_ll),
            vgetq_lane_u64::<1>(p_ll) ^ vgetq_lane_u64::<0>(cross),
            vgetq_lane_u64::<0>(p_hh) ^ vgetq_lane_u64::<1>(cross),
            vgetq_lane_u64::<1>(p_hh),
        );
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let ll_lo = k0_clmul64_lo(a_lo, b_lo);
        let ll_hi = k0_clmul64_hi(a_lo, b_lo);
        let lh_lo = k0_clmul64_lo(a_lo, b_hi);
        let lh_hi = k0_clmul64_hi(a_lo, b_hi);
        let hl_lo = k0_clmul64_lo(a_hi, b_lo);
        let hl_hi = k0_clmul64_hi(a_hi, b_lo);
        let hh_lo = k0_clmul64_lo(a_hi, b_hi);
        let hh_hi = k0_clmul64_hi(a_hi, b_hi);
        (ll_lo, ll_hi ^ lh_lo ^ hl_lo, hh_lo ^ lh_hi ^ hl_hi, hh_hi)
    }
}
#[inline(always)]
pub unsafe fn k0_f128_mul_raw(a_lo: u64, a_hi: u64, b_lo: u64, b_hi: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let va = _mm_set_epi64x(a_hi as i64, a_lo as i64);
        let vb = _mm_set_epi64x(b_hi as i64, b_lo as i64);
        let res = k0_f128_mul_m128i(va, vb);
        return (
            _mm_extract_epi64::<0>(res) as u64,
            _mm_extract_epi64::<1>(res) as u64,
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let t0 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
        let mid = vreinterpretq_u64_p128(vmull_p64(a_lo ^ a_hi, b_lo ^ b_hi));
        let t2 = vreinterpretq_u64_p128(vmull_p64(a_hi, b_hi));
        let mut t1 = veorq_u64(mid, veorq_u64(t0, t2));
        t1 = veorq_u64(t1, vextq_u64::<1>(z, t2));
        t1 = veorq_u64(
            t1,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t2), poly)),
        );
        let mut res = veorq_u64(t0, vextq_u64::<1>(z, t1));
        res = veorq_u64(
            res,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1), poly)),
        );
        return (vgetq_lane_u64::<0>(res), vgetq_lane_u64::<1>(res));
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let (r0, r1, r2, r3) = k0_f128_mul_unreduced_raw(a_lo, a_hi, b_lo, b_hi);
        k0_f128_reduce_256_raw(r0, r1, r2, r3)
    }
}
#[inline(always)]
pub unsafe fn k0_f128_mul_vec2_raw(
    a0_lo: u64,
    a0_hi: u64,
    a1_lo: u64,
    a1_hi: u64,
    b0_lo: u64,
    b0_hi: u64,
    b1_lo: u64,
    b1_hi: u64,
) -> (u64, u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let va0 = _mm_set_epi64x(a0_hi as i64, a0_lo as i64);
        let vb0 = _mm_set_epi64x(b0_hi as i64, b0_lo as i64);
        let va1 = _mm_set_epi64x(a1_hi as i64, a1_lo as i64);
        let vb1 = _mm_set_epi64x(b1_hi as i64, b1_lo as i64);
        let (res0, res1) = k0_f128_mul_vec2_m128i(va0, va1, vb0, vb1);
        return (
            _mm_extract_epi64::<0>(res0) as u64,
            _mm_extract_epi64::<1>(res0) as u64,
            _mm_extract_epi64::<0>(res1) as u64,
            _mm_extract_epi64::<1>(res1) as u64,
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let p0_00 = vreinterpretq_u64_p128(vmull_p64(a0_lo, b0_lo));
        let p1_00 = vreinterpretq_u64_p128(vmull_p64(a1_lo, b1_lo));
        let m0 = vreinterpretq_u64_p128(vmull_p64(a0_lo ^ a0_hi, b0_lo ^ b0_hi));
        let m1 = vreinterpretq_u64_p128(vmull_p64(a1_lo ^ a1_hi, b1_lo ^ b1_hi));
        let p0_11 = vreinterpretq_u64_p128(vmull_p64(a0_hi, b0_hi));
        let p1_11 = vreinterpretq_u64_p128(vmull_p64(a1_hi, b1_hi));
        let mut t1_0 = veorq_u64(m0, veorq_u64(p0_00, p0_11));
        let mut t1_1 = veorq_u64(m1, veorq_u64(p1_00, p1_11));
        t1_0 = veorq_u64(t1_0, vextq_u64::<1>(z, p0_11));
        t1_1 = veorq_u64(t1_1, vextq_u64::<1>(z, p1_11));
        t1_0 = veorq_u64(
            t1_0,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p0_11), poly)),
        );
        t1_1 = veorq_u64(
            t1_1,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p1_11), poly)),
        );
        let mut res0 = veorq_u64(p0_00, vextq_u64::<1>(z, t1_0));
        let mut res1 = veorq_u64(p1_00, vextq_u64::<1>(z, t1_1));
        res0 = veorq_u64(
            res0,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1_0), poly)),
        );
        res1 = veorq_u64(
            res1,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1_1), poly)),
        );
        return (
            vgetq_lane_u64::<0>(res0),
            vgetq_lane_u64::<1>(res0),
            vgetq_lane_u64::<0>(res1),
            vgetq_lane_u64::<1>(res1),
        );
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let (o0_lo, o0_hi) = k0_f128_mul_raw(a0_lo, a0_hi, b0_lo, b0_hi);
        let (o1_lo, o1_hi) = k0_f128_mul_raw(a1_lo, a1_hi, b1_lo, b1_hi);
        (o0_lo, o0_hi, o1_lo, o1_hi)
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_m128i(
    va: core::arch::x86_64::__m128i,
    vb: core::arch::x86_64::__m128i,
) -> core::arch::x86_64::__m128i {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm_set1_epi64x(0x87);
        let t0 = _mm_clmulepi64_si128::<0x00>(va, vb);
        let t2 = _mm_clmulepi64_si128::<0x11>(va, vb);
        let a_sum = _mm_xor_si128(va, _mm_shuffle_epi32::<0x4e>(va));
        let b_sum = _mm_xor_si128(vb, _mm_shuffle_epi32::<0x4e>(vb));
        let mid = _mm_clmulepi64_si128::<0x00>(a_sum, b_sum);
        let mut t1 = _mm_xor_si128(mid, _mm_xor_si128(t0, t2));
        t1 = _mm_xor_si128(t1, _mm_bslli_si128::<8>(t2));
        t1 = _mm_xor_si128(t1, _mm_clmulepi64_si128::<0x01>(t2, poly));
        let mut res = _mm_xor_si128(t0, _mm_bslli_si128::<8>(t1));
        res = _mm_xor_si128(res, _mm_clmulepi64_si128::<0x01>(t1, poly));
        res
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_vec2_m128i(
    va0: core::arch::x86_64::__m128i,
    va1: core::arch::x86_64::__m128i,
    vb0: core::arch::x86_64::__m128i,
    vb1: core::arch::x86_64::__m128i,
) -> (core::arch::x86_64::__m128i, core::arch::x86_64::__m128i) {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm_set1_epi64x(0x87);
        let p0_00 = _mm_clmulepi64_si128::<0x00>(va0, vb0);
        let p1_00 = _mm_clmulepi64_si128::<0x00>(va1, vb1);
        let p0_11 = _mm_clmulepi64_si128::<0x11>(va0, vb0);
        let p1_11 = _mm_clmulepi64_si128::<0x11>(va1, vb1);
        let a0_sum = _mm_xor_si128(va0, _mm_shuffle_epi32::<0x4e>(va0));
        let b0_sum = _mm_xor_si128(vb0, _mm_shuffle_epi32::<0x4e>(vb0));
        let a1_sum = _mm_xor_si128(va1, _mm_shuffle_epi32::<0x4e>(va1));
        let b1_sum = _mm_xor_si128(vb1, _mm_shuffle_epi32::<0x4e>(vb1));
        let m0 = _mm_clmulepi64_si128::<0x00>(a0_sum, b0_sum);
        let m1 = _mm_clmulepi64_si128::<0x00>(a1_sum, b1_sum);
        let mut t1_0 = _mm_xor_si128(m0, _mm_xor_si128(p0_00, p0_11));
        let mut t1_1 = _mm_xor_si128(m1, _mm_xor_si128(p1_00, p1_11));
        t1_0 = _mm_xor_si128(t1_0, _mm_bslli_si128::<8>(p0_11));
        t1_1 = _mm_xor_si128(t1_1, _mm_bslli_si128::<8>(p1_11));
        t1_0 = _mm_xor_si128(t1_0, _mm_clmulepi64_si128::<0x01>(p0_11, poly));
        t1_1 = _mm_xor_si128(t1_1, _mm_clmulepi64_si128::<0x01>(p1_11, poly));
        let mut res0 = _mm_xor_si128(p0_00, _mm_bslli_si128::<8>(t1_0));
        let mut res1 = _mm_xor_si128(p1_00, _mm_bslli_si128::<8>(t1_1));
        res0 = _mm_xor_si128(res0, _mm_clmulepi64_si128::<0x01>(t1_0, poly));
        res1 = _mm_xor_si128(res1, _mm_clmulepi64_si128::<0x01>(t1_1, poly));
        (res0, res1)
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_pre_reduce_const_m128i(
    vr: core::arch::x86_64::__m128i,
) -> (core::arch::x86_64::__m128i, core::arch::x86_64::__m128i) {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm_set1_epi64x(0x87);
        let p = _mm_clmulepi64_si128::<0x01>(vr, poly);
        let vr0 = _mm_unpacklo_epi64(vr, p);
        let p_xor = _mm_xor_si128(p, _mm_shuffle_epi32::<0x44>(vr));
        let vr1 = _mm_unpackhi_epi64(vr, p_xor);
        (vr0, vr1)
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_const_m128i(
    vx: core::arch::x86_64::__m128i,
    vr0: core::arch::x86_64::__m128i,
    vr1: core::arch::x86_64::__m128i,
) -> core::arch::x86_64::__m128i {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm_set1_epi64x(0x87);
        let l0 = _mm_clmulepi64_si128::<0x00>(vx, vr0);
        let l1 = _mm_clmulepi64_si128::<0x11>(vx, vr0);
        let h0 = _mm_clmulepi64_si128::<0x00>(vx, vr1);
        let h1 = _mm_clmulepi64_si128::<0x11>(vx, vr1);
        let l = _mm_xor_si128(l0, l1);
        let h = _mm_xor_si128(h0, h1);
        let red = _mm_clmulepi64_si128::<0x01>(h, poly);
        _mm_xor_si128(_mm_xor_si128(l, _mm_bslli_si128::<8>(h)), red)
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_const_vec2_m128i(
    vx0: core::arch::x86_64::__m128i,
    vx1: core::arch::x86_64::__m128i,
    vr00: core::arch::x86_64::__m128i,
    vr01: core::arch::x86_64::__m128i,
    vr10: core::arch::x86_64::__m128i,
    vr11: core::arch::x86_64::__m128i,
) -> (core::arch::x86_64::__m128i, core::arch::x86_64::__m128i) {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm_set1_epi64x(0x87);
        let l0_0 = _mm_clmulepi64_si128::<0x00>(vx0, vr00);
        let l1_0 = _mm_clmulepi64_si128::<0x00>(vx1, vr10);
        let l0_1 = _mm_clmulepi64_si128::<0x11>(vx0, vr00);
        let l1_1 = _mm_clmulepi64_si128::<0x11>(vx1, vr10);
        let h0_0 = _mm_clmulepi64_si128::<0x00>(vx0, vr01);
        let h1_0 = _mm_clmulepi64_si128::<0x00>(vx1, vr11);
        let h0_1 = _mm_clmulepi64_si128::<0x11>(vx0, vr01);
        let h1_1 = _mm_clmulepi64_si128::<0x11>(vx1, vr11);
        let l0 = _mm_xor_si128(l0_0, l0_1);
        let l1 = _mm_xor_si128(l1_0, l1_1);
        let h0 = _mm_xor_si128(h0_0, h0_1);
        let h1 = _mm_xor_si128(h1_0, h1_1);
        let red0 = _mm_clmulepi64_si128::<0x01>(h0, poly);
        let red1 = _mm_clmulepi64_si128::<0x01>(h1, poly);
        (
            _mm_xor_si128(_mm_xor_si128(l0, _mm_bslli_si128::<8>(h0)), red0),
            _mm_xor_si128(_mm_xor_si128(l1, _mm_bslli_si128::<8>(h1)), red1),
        )
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_acc_m128i(
    va: core::arch::x86_64::__m128i,
    vb: core::arch::x86_64::__m128i,
    ll: &mut core::arch::x86_64::__m128i,
    mid: &mut core::arch::x86_64::__m128i,
    hh: &mut core::arch::x86_64::__m128i,
) {
    unsafe {
        use core::arch::x86_64::*;
        *ll = _mm_xor_si128(*ll, _mm_clmulepi64_si128::<0x00>(va, vb));
        *hh = _mm_xor_si128(*hh, _mm_clmulepi64_si128::<0x11>(va, vb));
        let a_sum = _mm_xor_si128(va, _mm_shuffle_epi32::<0x4e>(va));
        let b_sum = _mm_xor_si128(vb, _mm_shuffle_epi32::<0x4e>(vb));
        *mid = _mm_xor_si128(*mid, _mm_clmulepi64_si128::<0x00>(a_sum, b_sum));
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_acc_pair_m128i(
    veq: core::arch::x86_64::__m128i,
    vb0: core::arch::x86_64::__m128i,
    vb1: core::arch::x86_64::__m128i,
    ll0: &mut core::arch::x86_64::__m128i,
    mid0: &mut core::arch::x86_64::__m128i,
    hh0: &mut core::arch::x86_64::__m128i,
    ll1: &mut core::arch::x86_64::__m128i,
    mid1: &mut core::arch::x86_64::__m128i,
    hh1: &mut core::arch::x86_64::__m128i,
) {
    unsafe {
        use core::arch::x86_64::*;
        let eq_sum = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4e>(veq));
        *ll0 = _mm_xor_si128(*ll0, _mm_clmulepi64_si128::<0x00>(veq, vb0));
        *ll1 = _mm_xor_si128(*ll1, _mm_clmulepi64_si128::<0x00>(veq, vb1));
        *hh0 = _mm_xor_si128(*hh0, _mm_clmulepi64_si128::<0x11>(veq, vb0));
        *hh1 = _mm_xor_si128(*hh1, _mm_clmulepi64_si128::<0x11>(veq, vb1));
        let b0_sum = _mm_xor_si128(vb0, _mm_shuffle_epi32::<0x4e>(vb0));
        let b1_sum = _mm_xor_si128(vb1, _mm_shuffle_epi32::<0x4e>(vb1));
        *mid0 = _mm_xor_si128(*mid0, _mm_clmulepi64_si128::<0x00>(eq_sum, b0_sum));
        *mid1 = _mm_xor_si128(*mid1, _mm_clmulepi64_si128::<0x00>(eq_sum, b1_sum));
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_reduce_3lane_m128i(
    ll: core::arch::x86_64::__m128i,
    mid: core::arch::x86_64::__m128i,
    hh: core::arch::x86_64::__m128i,
) -> core::arch::x86_64::__m128i {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm_set1_epi64x(0x87);
        let cross = _mm_xor_si128(mid, _mm_xor_si128(ll, hh));
        let mut t1 = _mm_xor_si128(cross, _mm_bslli_si128::<8>(hh));
        t1 = _mm_xor_si128(t1, _mm_clmulepi64_si128::<0x01>(hh, poly));
        let mut res = _mm_xor_si128(ll, _mm_bslli_si128::<8>(t1));
        res = _mm_xor_si128(res, _mm_clmulepi64_si128::<0x01>(t1, poly));
        res
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn k0_f128_pre_reduce_const_neon(
    vr: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let p = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(vr), poly));
        let vr0 = vzip1q_u64(vr, p);
        let p_xor = veorq_u64(p, vdupq_laneq_u64::<0>(vr));
        let vr1 = vzip2q_u64(vr, p_xor);
        (vr0, vr1)
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn k0_f128_mul_const_neon(
    vx: core::arch::aarch64::uint64x2_t,
    vr0: core::arch::aarch64::uint64x2_t,
    vr1: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let vx_lo = vgetq_lane_u64::<0>(vx);
        let l0 = vreinterpretq_u64_p128(vmull_p64(vx_lo, vgetq_lane_u64::<0>(vr0)));
        let l1 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(vx),
            vreinterpretq_p64_u64(vr0),
        ));
        let h0 = vreinterpretq_u64_p128(vmull_p64(vx_lo, vgetq_lane_u64::<0>(vr1)));
        let h1 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(vx),
            vreinterpretq_p64_u64(vr1),
        ));
        let l = veorq_u64(l0, l1);
        let h = veorq_u64(h0, h1);
        let red = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(h), poly));
        veorq_u64(veorq_u64(l, vextq_u64::<1>(z, h)), red)
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn k0_f128_mul_const_vec2_neon(
    vx0: core::arch::aarch64::uint64x2_t,
    vx1: core::arch::aarch64::uint64x2_t,
    vr00: core::arch::aarch64::uint64x2_t,
    vr01: core::arch::aarch64::uint64x2_t,
    vr10: core::arch::aarch64::uint64x2_t,
    vr11: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let vx0_lo = vgetq_lane_u64::<0>(vx0);
        let vx1_lo = vgetq_lane_u64::<0>(vx1);
        let l0_0 = vreinterpretq_u64_p128(vmull_p64(vx0_lo, vgetq_lane_u64::<0>(vr00)));
        let l1_0 = vreinterpretq_u64_p128(vmull_p64(vx1_lo, vgetq_lane_u64::<0>(vr10)));
        let l0_1 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(vx0),
            vreinterpretq_p64_u64(vr00),
        ));
        let l1_1 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(vx1),
            vreinterpretq_p64_u64(vr10),
        ));
        let h0_0 = vreinterpretq_u64_p128(vmull_p64(vx0_lo, vgetq_lane_u64::<0>(vr01)));
        let h1_0 = vreinterpretq_u64_p128(vmull_p64(vx1_lo, vgetq_lane_u64::<0>(vr11)));
        let h0_1 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(vx0),
            vreinterpretq_p64_u64(vr01),
        ));
        let h1_1 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(vx1),
            vreinterpretq_p64_u64(vr11),
        ));
        let l0 = veorq_u64(l0_0, l0_1);
        let l1 = veorq_u64(l1_0, l1_1);
        let h0 = veorq_u64(h0_0, h0_1);
        let h1 = veorq_u64(h1_0, h1_1);
        let red0 = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(h0), poly));
        let red1 = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(h1), poly));
        (
            veorq_u64(veorq_u64(l0, vextq_u64::<1>(z, h0)), red0),
            veorq_u64(veorq_u64(l1, vextq_u64::<1>(z, h1)), red1),
        )
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn k0_f128_mul_neon(
    va: core::arch::aarch64::uint64x2_t,
    vb: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let a_lo = vgetq_lane_u64::<0>(va);
        let b_lo = vgetq_lane_u64::<0>(vb);
        let a_sum = a_lo ^ vgetq_lane_u64::<1>(va);
        let b_sum = b_lo ^ vgetq_lane_u64::<1>(vb);
        let t0 = vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo));
        let mid = vreinterpretq_u64_p128(vmull_p64(a_sum, b_sum));
        let t2 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(va),
            vreinterpretq_p64_u64(vb),
        ));
        let mut t1 = veorq_u64(mid, veorq_u64(t0, t2));
        t1 = veorq_u64(t1, vextq_u64::<1>(z, t2));
        t1 = veorq_u64(
            t1,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t2), poly)),
        );
        let mut res = veorq_u64(t0, vextq_u64::<1>(z, t1));
        res = veorq_u64(
            res,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1), poly)),
        );
        res
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn k0_f128_mul_vec2_neon(
    va0: core::arch::aarch64::uint64x2_t,
    va1: core::arch::aarch64::uint64x2_t,
    vb0: core::arch::aarch64::uint64x2_t,
    vb1: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let a0_lo = vgetq_lane_u64::<0>(va0);
        let b0_lo = vgetq_lane_u64::<0>(vb0);
        let a1_lo = vgetq_lane_u64::<0>(va1);
        let b1_lo = vgetq_lane_u64::<0>(vb1);
        let a0_sum = a0_lo ^ vgetq_lane_u64::<1>(va0);
        let b0_sum = b0_lo ^ vgetq_lane_u64::<1>(vb0);
        let a1_sum = a1_lo ^ vgetq_lane_u64::<1>(va1);
        let b1_sum = b1_lo ^ vgetq_lane_u64::<1>(vb1);
        let p0_00 = vreinterpretq_u64_p128(vmull_p64(a0_lo, b0_lo));
        let p1_00 = vreinterpretq_u64_p128(vmull_p64(a1_lo, b1_lo));
        let m0 = vreinterpretq_u64_p128(vmull_p64(a0_sum, b0_sum));
        let m1 = vreinterpretq_u64_p128(vmull_p64(a1_sum, b1_sum));
        let p0_11 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(va0),
            vreinterpretq_p64_u64(vb0),
        ));
        let p1_11 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(va1),
            vreinterpretq_p64_u64(vb1),
        ));
        let mut t1_0 = veorq_u64(m0, veorq_u64(p0_00, p0_11));
        let mut t1_1 = veorq_u64(m1, veorq_u64(p1_00, p1_11));
        t1_0 = veorq_u64(t1_0, vextq_u64::<1>(z, p0_11));
        t1_1 = veorq_u64(t1_1, vextq_u64::<1>(z, p1_11));
        t1_0 = veorq_u64(
            t1_0,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p0_11), poly)),
        );
        t1_1 = veorq_u64(
            t1_1,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p1_11), poly)),
        );
        let mut res0 = veorq_u64(p0_00, vextq_u64::<1>(z, t1_0));
        let mut res1 = veorq_u64(p1_00, vextq_u64::<1>(z, t1_1));
        res0 = veorq_u64(
            res0,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1_0), poly)),
        );
        res1 = veorq_u64(
            res1,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1_1), poly)),
        );
        (res0, res1)
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn k0_f128_mul_unreduced_neon(
    va: core::arch::aarch64::uint64x2_t,
    vb: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let mut ll = vdupq_n_u64(0);
        let mut mid = vdupq_n_u64(0);
        let mut hh = vdupq_n_u64(0);
        k0_f128_mul_acc_neon(va, vb, &mut ll, &mut mid, &mut hh);
        (ll, mid, hh)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn k0_f128_mul_acc_neon(
    va: core::arch::aarch64::uint64x2_t,
    vb: core::arch::aarch64::uint64x2_t,
    ll: &mut core::arch::aarch64::uint64x2_t,
    mid: &mut core::arch::aarch64::uint64x2_t,
    hh: &mut core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let a_lo = vgetq_lane_u64::<0>(va);
        let b_lo = vgetq_lane_u64::<0>(vb);
        let a_sum = a_lo ^ vgetq_lane_u64::<1>(va);
        let b_sum = b_lo ^ vgetq_lane_u64::<1>(vb);
        *ll = veorq_u64(*ll, vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo)));
        *mid = veorq_u64(*mid, vreinterpretq_u64_p128(vmull_p64(a_sum, b_sum)));
        *hh = veorq_u64(
            *hh,
            vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(va),
                vreinterpretq_p64_u64(vb),
            )),
        );
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn k0_f128_mul_acc_pair_neon(
    veq: core::arch::aarch64::uint64x2_t,
    vb0: core::arch::aarch64::uint64x2_t,
    vb1: core::arch::aarch64::uint64x2_t,
    ll0: &mut core::arch::aarch64::uint64x2_t,
    mid0: &mut core::arch::aarch64::uint64x2_t,
    hh0: &mut core::arch::aarch64::uint64x2_t,
    ll1: &mut core::arch::aarch64::uint64x2_t,
    mid1: &mut core::arch::aarch64::uint64x2_t,
    hh1: &mut core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let eq_lo = vgetq_lane_u64::<0>(veq);
        let eq_sum = eq_lo ^ vgetq_lane_u64::<1>(veq);
        let b0_lo = vgetq_lane_u64::<0>(vb0);
        let b1_lo = vgetq_lane_u64::<0>(vb1);
        let b0_sum = b0_lo ^ vgetq_lane_u64::<1>(vb0);
        let b1_sum = b1_lo ^ vgetq_lane_u64::<1>(vb1);
        *ll0 = veorq_u64(*ll0, vreinterpretq_u64_p128(vmull_p64(eq_lo, b0_lo)));
        *ll1 = veorq_u64(*ll1, vreinterpretq_u64_p128(vmull_p64(eq_lo, b1_lo)));
        *mid0 = veorq_u64(*mid0, vreinterpretq_u64_p128(vmull_p64(eq_sum, b0_sum)));
        *mid1 = veorq_u64(*mid1, vreinterpretq_u64_p128(vmull_p64(eq_sum, b1_sum)));
        *hh0 = veorq_u64(
            *hh0,
            vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(veq),
                vreinterpretq_p64_u64(vb0),
            )),
        );
        *hh1 = veorq_u64(
            *hh1,
            vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(veq),
                vreinterpretq_p64_u64(vb1),
            )),
        );
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn k0_f128_reduce_3lane_neon(
    ll: core::arch::aarch64::uint64x2_t,
    mid: core::arch::aarch64::uint64x2_t,
    hh: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let cross = veorq_u64(mid, veorq_u64(ll, hh));
        let mut t1 = veorq_u64(cross, vextq_u64::<1>(z, hh));
        t1 = veorq_u64(
            t1,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(hh), poly)),
        );
        let mut res = veorq_u64(ll, vextq_u64::<1>(z, t1));
        res = veorq_u64(
            res,
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1), poly)),
        );
        res
    }
}
#[inline(always)]
pub unsafe fn k0_ntt_butterfly_row_pair_slice(
    top_ptr: *mut u64,
    bot_ptr: *mut u64,
    n_f128: usize,
    tw_lo: u64,
    tw_hi: u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let tw = _mm_set_epi64x(tw_hi as i64, tw_lo as i64);
        let (tw0, tw1) = k0_f128_pre_reduce_const_m128i(tw);
        let mut lane = 0usize;
        while lane + 2 <= n_f128 {
            let idx0 = lane * 2;
            let idx1 = idx0 + 2;
            let u0 = _mm_loadu_si128(top_ptr.add(idx0) as *const __m128i);
            let v0 = _mm_loadu_si128(bot_ptr.add(idx0) as *const __m128i);
            let u1 = _mm_loadu_si128(top_ptr.add(idx1) as *const __m128i);
            let v1 = _mm_loadu_si128(bot_ptr.add(idx1) as *const __m128i);
            let (p0, p1) = k0_f128_mul_const_vec2_m128i(v0, v1, tw0, tw1, tw0, tw1);
            let nu0 = _mm_xor_si128(u0, p0);
            let nv0 = _mm_xor_si128(v0, nu0);
            let nu1 = _mm_xor_si128(u1, p1);
            let nv1 = _mm_xor_si128(v1, nu1);
            _mm_storeu_si128(top_ptr.add(idx0) as *mut __m128i, nu0);
            _mm_storeu_si128(bot_ptr.add(idx0) as *mut __m128i, nv0);
            _mm_storeu_si128(top_ptr.add(idx1) as *mut __m128i, nu1);
            _mm_storeu_si128(bot_ptr.add(idx1) as *mut __m128i, nv1);
            lane += 2;
        }
        while lane < n_f128 {
            let idx = lane * 2;
            let u = _mm_loadu_si128(top_ptr.add(idx) as *const __m128i);
            let v = _mm_loadu_si128(bot_ptr.add(idx) as *const __m128i);
            let p = k0_f128_mul_const_m128i(v, tw0, tw1);
            let nu = _mm_xor_si128(u, p);
            _mm_storeu_si128(top_ptr.add(idx) as *mut __m128i, nu);
            _mm_storeu_si128(bot_ptr.add(idx) as *mut __m128i, _mm_xor_si128(v, nu));
            lane += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let tw_arr = [tw_lo, tw_hi];
        let tw = vld1q_u64(tw_arr.as_ptr());
        let (tw0, tw1) = k0_f128_pre_reduce_const_neon(tw);
        let mut lane = 0usize;
        while lane + 2 <= n_f128 {
            let idx0 = lane * 2;
            let idx1 = idx0 + 2;
            let u0 = vld1q_u64(top_ptr.add(idx0));
            let v0 = vld1q_u64(bot_ptr.add(idx0));
            let u1 = vld1q_u64(top_ptr.add(idx1));
            let v1 = vld1q_u64(bot_ptr.add(idx1));
            let (p0, p1) = k0_f128_mul_const_vec2_neon(v0, v1, tw0, tw1, tw0, tw1);
            let nu0 = veorq_u64(u0, p0);
            let nv0 = veorq_u64(v0, nu0);
            let nu1 = veorq_u64(u1, p1);
            let nv1 = veorq_u64(v1, nu1);
            vst1q_u64(top_ptr.add(idx0), nu0);
            vst1q_u64(bot_ptr.add(idx0), nv0);
            vst1q_u64(top_ptr.add(idx1), nu1);
            vst1q_u64(bot_ptr.add(idx1), nv1);
            lane += 2;
        }
        while lane < n_f128 {
            let idx = lane * 2;
            let u = vld1q_u64(top_ptr.add(idx));
            let v = vld1q_u64(bot_ptr.add(idx));
            let p = k0_f128_mul_const_neon(v, tw0, tw1);
            let nu = veorq_u64(u, p);
            vst1q_u64(top_ptr.add(idx), nu);
            vst1q_u64(bot_ptr.add(idx), veorq_u64(v, nu));
            lane += 1;
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let mut lane = 0usize;
        while lane + 2 <= n_f128 {
            let idx0 = lane * 2;
            let idx1 = idx0 + 2;
            let u0_lo = *top_ptr.add(idx0);
            let u0_hi = *top_ptr.add(idx0 + 1);
            let v0_lo = *bot_ptr.add(idx0);
            let v0_hi = *bot_ptr.add(idx0 + 1);
            let u1_lo = *top_ptr.add(idx1);
            let u1_hi = *top_ptr.add(idx1 + 1);
            let v1_lo = *bot_ptr.add(idx1);
            let v1_hi = *bot_ptr.add(idx1 + 1);
            let (p0_lo, p0_hi, p1_lo, p1_hi) =
                k0_f128_mul_vec2_raw(v0_lo, v0_hi, v1_lo, v1_hi, tw_lo, tw_hi, tw_lo, tw_hi);
            let nu0_lo = u0_lo ^ p0_lo;
            let nu0_hi = u0_hi ^ p0_hi;
            let nv0_lo = v0_lo ^ nu0_lo;
            let nv0_hi = v0_hi ^ nu0_hi;
            let nu1_lo = u1_lo ^ p1_lo;
            let nu1_hi = u1_hi ^ p1_hi;
            let nv1_lo = v1_lo ^ nu1_lo;
            let nv1_hi = v1_hi ^ nu1_hi;
            *top_ptr.add(idx0) = nu0_lo;
            *top_ptr.add(idx0 + 1) = nu0_hi;
            *bot_ptr.add(idx0) = nv0_lo;
            *bot_ptr.add(idx0 + 1) = nv0_hi;
            *top_ptr.add(idx1) = nu1_lo;
            *top_ptr.add(idx1 + 1) = nu1_hi;
            *bot_ptr.add(idx1) = nv1_lo;
            *bot_ptr.add(idx1 + 1) = nv1_hi;
            lane += 2;
        }
        while lane < n_f128 {
            let idx = lane * 2;
            let u_lo = *top_ptr.add(idx);
            let u_hi = *top_ptr.add(idx + 1);
            let v_lo = *bot_ptr.add(idx);
            let v_hi = *bot_ptr.add(idx + 1);
            let (p_lo, p_hi) = k0_f128_mul_raw(v_lo, v_hi, tw_lo, tw_hi);
            let nu_lo = u_lo ^ p_lo;
            let nu_hi = u_hi ^ p_hi;
            *top_ptr.add(idx) = nu_lo;
            *top_ptr.add(idx + 1) = nu_hi;
            *bot_ptr.add(idx) = v_lo ^ nu_lo;
            *bot_ptr.add(idx + 1) = v_hi ^ nu_hi;
            lane += 1;
        }
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_ntt_butterfly_fused_2layer_slice(
    a_ptr: *mut u64,
    b_ptr: *mut u64,
    c_ptr: *mut u64,
    d_ptr: *mut u64,
    n_f128: usize,
    to_lo: u64,
    to_hi: u64,
    tia_lo: u64,
    tia_hi: u64,
    tib_lo: u64,
    tib_hi: u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let to = _mm_set_epi64x(to_hi as i64, to_lo as i64);
        let tia = _mm_set_epi64x(tia_hi as i64, tia_lo as i64);
        let tib = _mm_set_epi64x(tib_hi as i64, tib_lo as i64);
        let (to0, to1) = k0_f128_pre_reduce_const_m128i(to);
        let (tia0, tia1) = k0_f128_pre_reduce_const_m128i(tia);
        let (tib0, tib1) = k0_f128_pre_reduce_const_m128i(tib);
        for lane in 0..n_f128 {
            let idx = lane * 2;
            let xa = _mm_loadu_si128(a_ptr.add(idx) as *const __m128i);
            let xb = _mm_loadu_si128(b_ptr.add(idx) as *const __m128i);
            let xc = _mm_loadu_si128(c_ptr.add(idx) as *const __m128i);
            let xd = _mm_loadu_si128(d_ptr.add(idx) as *const __m128i);
            let (pc, pd) = k0_f128_mul_const_vec2_m128i(xc, xd, to0, to1, to0, to1);
            let na = _mm_xor_si128(xa, pc);
            let nc = _mm_xor_si128(xc, na);
            let nb = _mm_xor_si128(xb, pd);
            let nd = _mm_xor_si128(xd, nb);
            let (pb2, pd2) = k0_f128_mul_const_vec2_m128i(nb, nd, tia0, tia1, tib0, tib1);
            let fa = _mm_xor_si128(na, pb2);
            let fb = _mm_xor_si128(nb, fa);
            let fc = _mm_xor_si128(nc, pd2);
            let fd = _mm_xor_si128(nd, fc);
            _mm_storeu_si128(a_ptr.add(idx) as *mut __m128i, fa);
            _mm_storeu_si128(b_ptr.add(idx) as *mut __m128i, fb);
            _mm_storeu_si128(c_ptr.add(idx) as *mut __m128i, fc);
            _mm_storeu_si128(d_ptr.add(idx) as *mut __m128i, fd);
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let to_arr = [to_lo, to_hi];
        let tia_arr = [tia_lo, tia_hi];
        let tib_arr = [tib_lo, tib_hi];
        let to = vld1q_u64(to_arr.as_ptr());
        let tia = vld1q_u64(tia_arr.as_ptr());
        let tib = vld1q_u64(tib_arr.as_ptr());
        let (to0, to1) = k0_f128_pre_reduce_const_neon(to);
        let (tia0, tia1) = k0_f128_pre_reduce_const_neon(tia);
        let (tib0, tib1) = k0_f128_pre_reduce_const_neon(tib);
        for lane in 0..n_f128 {
            let idx = lane * 2;
            let xa = vld1q_u64(a_ptr.add(idx));
            let xb = vld1q_u64(b_ptr.add(idx));
            let xc = vld1q_u64(c_ptr.add(idx));
            let xd = vld1q_u64(d_ptr.add(idx));
            let (pc, pd) = k0_f128_mul_const_vec2_neon(xc, xd, to0, to1, to0, to1);
            let na = veorq_u64(xa, pc);
            let nc = veorq_u64(xc, na);
            let nb = veorq_u64(xb, pd);
            let nd = veorq_u64(xd, nb);
            let (pb2, pd2) = k0_f128_mul_const_vec2_neon(nb, nd, tia0, tia1, tib0, tib1);
            let fa = veorq_u64(na, pb2);
            let fb = veorq_u64(nb, fa);
            let fc = veorq_u64(nc, pd2);
            let fd = veorq_u64(nd, fc);
            vst1q_u64(a_ptr.add(idx), fa);
            vst1q_u64(b_ptr.add(idx), fb);
            vst1q_u64(c_ptr.add(idx), fc);
            vst1q_u64(d_ptr.add(idx), fd);
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        for lane in 0..n_f128 {
            let idx = lane * 2;
            let xa_lo = *a_ptr.add(idx);
            let xa_hi = *a_ptr.add(idx + 1);
            let xb_lo = *b_ptr.add(idx);
            let xb_hi = *b_ptr.add(idx + 1);
            let xc_lo = *c_ptr.add(idx);
            let xc_hi = *c_ptr.add(idx + 1);
            let xd_lo = *d_ptr.add(idx);
            let xd_hi = *d_ptr.add(idx + 1);
            let (pc_lo, pc_hi, pd_lo, pd_hi) =
                k0_f128_mul_vec2_raw(xc_lo, xc_hi, xd_lo, xd_hi, to_lo, to_hi, to_lo, to_hi);
            let na_lo = xa_lo ^ pc_lo;
            let na_hi = xa_hi ^ pc_hi;
            let nc_lo = xc_lo ^ na_lo;
            let nc_hi = xc_hi ^ na_hi;
            let nb_lo = xb_lo ^ pd_lo;
            let nb_hi = xb_hi ^ pd_hi;
            let nd_lo = xd_lo ^ nb_lo;
            let nd_hi = xd_hi ^ nb_hi;
            let (pb2_lo, pb2_hi, pd2_lo, pd2_hi) =
                k0_f128_mul_vec2_raw(nb_lo, nb_hi, nd_lo, nd_hi, tia_lo, tia_hi, tib_lo, tib_hi);
            let fa_lo = na_lo ^ pb2_lo;
            let fa_hi = na_hi ^ pb2_hi;
            let fb_lo = nb_lo ^ fa_lo;
            let fb_hi = nb_hi ^ fa_hi;
            let fc_lo = nc_lo ^ pd2_lo;
            let fc_hi = nc_hi ^ pd2_hi;
            let fd_lo = nd_lo ^ fc_lo;
            let fd_hi = nd_hi ^ fc_hi;
            *a_ptr.add(idx) = fa_lo;
            *a_ptr.add(idx + 1) = fa_hi;
            *b_ptr.add(idx) = fb_lo;
            *b_ptr.add(idx + 1) = fb_hi;
            *c_ptr.add(idx) = fc_lo;
            *c_ptr.add(idx + 1) = fc_hi;
            *d_ptr.add(idx) = fd_lo;
            *d_ptr.add(idx + 1) = fd_hi;
        }
    }
}
#[inline(always)]
pub unsafe fn k0_ntt_butterfly_fused_4layer_rows_slice(
    ptr: *mut u64,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    tw_ptr: *const u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut tw_r0 = [_mm_setzero_si128(); 15];
        let mut tw_r1 = [_mm_setzero_si128(); 15];
        for i in 0..15 {
            let t = _mm_loadu_si128(tw_ptr.add(2 * i) as *const __m128i);
            let (r0, r1) = k0_f128_pre_reduce_const_m128i(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        macro_rules! btf2_x86 {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = k0_f128_mul_const_vec2_m128i(
                    $v[$v0], $v[$v1], $tw0[$t0], $tw1[$t0], $tw0[$t1], $tw1[$t1],
                );
                let nu0 = _mm_xor_si128($v[$u0], p0);
                $v[$v0] = _mm_xor_si128($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = _mm_xor_si128($v[$u1], p1);
                $v[$v1] = _mm_xor_si128($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v = [_mm_setzero_si128(); 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = _mm_loadu_si128(ptr.add(off) as *const __m128i);
                }
                btf2_x86!(v, tw_r0, tw_r1, 0, 8, 0, 1, 9, 0);
                btf2_x86!(v, tw_r0, tw_r1, 2, 10, 0, 3, 11, 0);
                btf2_x86!(v, tw_r0, tw_r1, 4, 12, 0, 5, 13, 0);
                btf2_x86!(v, tw_r0, tw_r1, 6, 14, 0, 7, 15, 0);
                btf2_x86!(v, tw_r0, tw_r1, 0, 4, 1, 1, 5, 1);
                btf2_x86!(v, tw_r0, tw_r1, 2, 6, 1, 3, 7, 1);
                btf2_x86!(v, tw_r0, tw_r1, 8, 12, 2, 9, 13, 2);
                btf2_x86!(v, tw_r0, tw_r1, 10, 14, 2, 11, 15, 2);
                btf2_x86!(v, tw_r0, tw_r1, 0, 2, 3, 1, 3, 3);
                btf2_x86!(v, tw_r0, tw_r1, 4, 6, 4, 5, 7, 4);
                btf2_x86!(v, tw_r0, tw_r1, 8, 10, 5, 9, 11, 5);
                btf2_x86!(v, tw_r0, tw_r1, 12, 14, 6, 13, 15, 6);
                btf2_x86!(v, tw_r0, tw_r1, 0, 1, 7, 2, 3, 8);
                btf2_x86!(v, tw_r0, tw_r1, 4, 5, 9, 6, 7, 10);
                btf2_x86!(v, tw_r0, tw_r1, 8, 9, 11, 10, 11, 12);
                btf2_x86!(v, tw_r0, tw_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    _mm_storeu_si128(ptr.add(off) as *mut __m128i, v[i]);
                }
            }
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let mut tw_r0 = [z; 15];
        let mut tw_r1 = [z; 15];
        for i in 0..15 {
            let t = vld1q_u64(tw_ptr.add(2 * i));
            let (r0, r1) = k0_f128_pre_reduce_const_neon(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        macro_rules! btf2_neon {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = k0_f128_mul_const_vec2_neon(
                    $v[$v0], $v[$v1], $tw0[$t0], $tw1[$t0], $tw0[$t1], $tw1[$t1],
                );
                let nu0 = veorq_u64($v[$u0], p0);
                $v[$v0] = veorq_u64($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = veorq_u64($v[$u1], p1);
                $v[$v1] = veorq_u64($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v = [z; 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = vld1q_u64(ptr.add(off));
                }
                btf2_neon!(v, tw_r0, tw_r1, 0, 8, 0, 1, 9, 0);
                btf2_neon!(v, tw_r0, tw_r1, 2, 10, 0, 3, 11, 0);
                btf2_neon!(v, tw_r0, tw_r1, 4, 12, 0, 5, 13, 0);
                btf2_neon!(v, tw_r0, tw_r1, 6, 14, 0, 7, 15, 0);
                btf2_neon!(v, tw_r0, tw_r1, 0, 4, 1, 1, 5, 1);
                btf2_neon!(v, tw_r0, tw_r1, 2, 6, 1, 3, 7, 1);
                btf2_neon!(v, tw_r0, tw_r1, 8, 12, 2, 9, 13, 2);
                btf2_neon!(v, tw_r0, tw_r1, 10, 14, 2, 11, 15, 2);
                btf2_neon!(v, tw_r0, tw_r1, 0, 2, 3, 1, 3, 3);
                btf2_neon!(v, tw_r0, tw_r1, 4, 6, 4, 5, 7, 4);
                btf2_neon!(v, tw_r0, tw_r1, 8, 10, 5, 9, 11, 5);
                btf2_neon!(v, tw_r0, tw_r1, 12, 14, 6, 13, 15, 6);
                btf2_neon!(v, tw_r0, tw_r1, 0, 1, 7, 2, 3, 8);
                btf2_neon!(v, tw_r0, tw_r1, 4, 5, 9, 6, 7, 10);
                btf2_neon!(v, tw_r0, tw_r1, 8, 9, 11, 10, 11, 12);
                btf2_neon!(v, tw_r0, tw_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    vst1q_u64(ptr.add(off), v[i]);
                }
            }
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let mut tw = [(0u64, 0u64); 15];
        for i in 0..15 {
            tw[i] = (*tw_ptr.add(2 * i), *tw_ptr.add(2 * i + 1));
        }
        macro_rules! btf2 {
            ($v:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0_lo, p0_hi, p1_lo, p1_hi) = k0_f128_mul_vec2_raw(
                    $v[$v0].0, $v[$v0].1, $v[$v1].0, $v[$v1].1, tw[$t0].0, tw[$t0].1, tw[$t1].0,
                    tw[$t1].1,
                );
                let nu0 = ($v[$u0].0 ^ p0_lo, $v[$u0].1 ^ p0_hi);
                $v[$v0] = ($v[$v0].0 ^ nu0.0, $v[$v0].1 ^ nu0.1);
                $v[$u0] = nu0;
                let nu1 = ($v[$u1].0 ^ p1_lo, $v[$u1].1 ^ p1_hi);
                $v[$v1] = ($v[$v1].0 ^ nu1.0, $v[$v1].1 ^ nu1.1);
                $v[$u1] = nu1;
            }};
        }
        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v = [(0u64, 0u64); 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = (*ptr.add(off), *ptr.add(off + 1));
                }
                btf2!(v, 0, 8, 0, 1, 9, 0);
                btf2!(v, 2, 10, 0, 3, 11, 0);
                btf2!(v, 4, 12, 0, 5, 13, 0);
                btf2!(v, 6, 14, 0, 7, 15, 0);
                btf2!(v, 0, 4, 1, 1, 5, 1);
                btf2!(v, 2, 6, 1, 3, 7, 1);
                btf2!(v, 8, 12, 2, 9, 13, 2);
                btf2!(v, 10, 14, 2, 11, 15, 2);
                btf2!(v, 0, 2, 3, 1, 3, 3);
                btf2!(v, 4, 6, 4, 5, 7, 4);
                btf2!(v, 8, 10, 5, 9, 11, 5);
                btf2!(v, 12, 14, 6, 13, 15, 6);
                btf2!(v, 0, 1, 7, 2, 3, 8);
                btf2!(v, 4, 5, 9, 6, 7, 10);
                btf2!(v, 8, 9, 11, 10, 11, 12);
                btf2!(v, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    *ptr.add(off) = v[i].0;
                    *ptr.add(off + 1) = v[i].1;
                }
            }
        }
    }
}
#[inline(always)]
pub unsafe fn k0_ntt_butterfly_fused_4layer_row_slice(
    ptr: *mut u64,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    tw_ptr: *const u64,
) {
    unsafe {
        k0_ntt_butterfly_fused_4layer_rows_slice(ptr, sixteenth, num_ntts, r, 1, tw_ptr);
    }
}
#[inline(always)]
pub unsafe fn k0_ntt_butterfly_fused_4layer_dual_from_src_rows_slice(
    src_ptr: *const u64,
    dst0_ptr: *mut u64,
    dst1_ptr: *mut u64,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    tw0_ptr: *const u64,
    tw1_ptr: *const u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut tw0_r0 = [_mm_setzero_si128(); 15];
        let mut tw0_r1 = [_mm_setzero_si128(); 15];
        let mut tw1_r0 = [_mm_setzero_si128(); 15];
        let mut tw1_r1 = [_mm_setzero_si128(); 15];
        for i in 0..15 {
            let t0 = _mm_loadu_si128(tw0_ptr.add(2 * i) as *const __m128i);
            let (r00, r01) = k0_f128_pre_reduce_const_m128i(t0);
            tw0_r0[i] = r00;
            tw0_r1[i] = r01;
            let t1 = _mm_loadu_si128(tw1_ptr.add(2 * i) as *const __m128i);
            let (r10, r11) = k0_f128_pre_reduce_const_m128i(t1);
            tw1_r0[i] = r10;
            tw1_r1[i] = r11;
        }
        macro_rules! btf2_x86 {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = k0_f128_mul_const_vec2_m128i(
                    $v[$v0], $v[$v1], $tw0[$t0], $tw1[$t0], $tw0[$t1], $tw1[$t1],
                );
                let nu0 = _mm_xor_si128($v[$u0], p0);
                $v[$v0] = _mm_xor_si128($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = _mm_xor_si128($v[$u1], p1);
                $v[$v1] = _mm_xor_si128($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v0 = [_mm_setzero_si128(); 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v0[i] = _mm_loadu_si128(src_ptr.add(off) as *const __m128i);
                }
                let mut v1 = v0;
                btf2_x86!(v0, tw0_r0, tw0_r1, 0, 8, 0, 1, 9, 0);
                btf2_x86!(v0, tw0_r0, tw0_r1, 2, 10, 0, 3, 11, 0);
                btf2_x86!(v0, tw0_r0, tw0_r1, 4, 12, 0, 5, 13, 0);
                btf2_x86!(v0, tw0_r0, tw0_r1, 6, 14, 0, 7, 15, 0);
                btf2_x86!(v0, tw0_r0, tw0_r1, 0, 4, 1, 1, 5, 1);
                btf2_x86!(v0, tw0_r0, tw0_r1, 2, 6, 1, 3, 7, 1);
                btf2_x86!(v0, tw0_r0, tw0_r1, 8, 12, 2, 9, 13, 2);
                btf2_x86!(v0, tw0_r0, tw0_r1, 10, 14, 2, 11, 15, 2);
                btf2_x86!(v0, tw0_r0, tw0_r1, 0, 2, 3, 1, 3, 3);
                btf2_x86!(v0, tw0_r0, tw0_r1, 4, 6, 4, 5, 7, 4);
                btf2_x86!(v0, tw0_r0, tw0_r1, 8, 10, 5, 9, 11, 5);
                btf2_x86!(v0, tw0_r0, tw0_r1, 12, 14, 6, 13, 15, 6);
                btf2_x86!(v0, tw0_r0, tw0_r1, 0, 1, 7, 2, 3, 8);
                btf2_x86!(v0, tw0_r0, tw0_r1, 4, 5, 9, 6, 7, 10);
                btf2_x86!(v0, tw0_r0, tw0_r1, 8, 9, 11, 10, 11, 12);
                btf2_x86!(v0, tw0_r0, tw0_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    _mm_storeu_si128(dst0_ptr.add(off) as *mut __m128i, v0[i]);
                }
                btf2_x86!(v1, tw1_r0, tw1_r1, 0, 8, 0, 1, 9, 0);
                btf2_x86!(v1, tw1_r0, tw1_r1, 2, 10, 0, 3, 11, 0);
                btf2_x86!(v1, tw1_r0, tw1_r1, 4, 12, 0, 5, 13, 0);
                btf2_x86!(v1, tw1_r0, tw1_r1, 6, 14, 0, 7, 15, 0);
                btf2_x86!(v1, tw1_r0, tw1_r1, 0, 4, 1, 1, 5, 1);
                btf2_x86!(v1, tw1_r0, tw1_r1, 2, 6, 1, 3, 7, 1);
                btf2_x86!(v1, tw1_r0, tw1_r1, 8, 12, 2, 9, 13, 2);
                btf2_x86!(v1, tw1_r0, tw1_r1, 10, 14, 2, 11, 15, 2);
                btf2_x86!(v1, tw1_r0, tw1_r1, 0, 2, 3, 1, 3, 3);
                btf2_x86!(v1, tw1_r0, tw1_r1, 4, 6, 4, 5, 7, 4);
                btf2_x86!(v1, tw1_r0, tw1_r1, 8, 10, 5, 9, 11, 5);
                btf2_x86!(v1, tw1_r0, tw1_r1, 12, 14, 6, 13, 15, 6);
                btf2_x86!(v1, tw1_r0, tw1_r1, 0, 1, 7, 2, 3, 8);
                btf2_x86!(v1, tw1_r0, tw1_r1, 4, 5, 9, 6, 7, 10);
                btf2_x86!(v1, tw1_r0, tw1_r1, 8, 9, 11, 10, 11, 12);
                btf2_x86!(v1, tw1_r0, tw1_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    _mm_storeu_si128(dst1_ptr.add(off) as *mut __m128i, v1[i]);
                }
            }
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let mut tw0_r0 = [z; 15];
        let mut tw0_r1 = [z; 15];
        let mut tw1_r0 = [z; 15];
        let mut tw1_r1 = [z; 15];
        for i in 0..15 {
            let t0 = vld1q_u64(tw0_ptr.add(2 * i));
            let (r00, r01) = k0_f128_pre_reduce_const_neon(t0);
            tw0_r0[i] = r00;
            tw0_r1[i] = r01;
            let t1 = vld1q_u64(tw1_ptr.add(2 * i));
            let (r10, r11) = k0_f128_pre_reduce_const_neon(t1);
            tw1_r0[i] = r10;
            tw1_r1[i] = r11;
        }
        macro_rules! btf2_neon {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = k0_f128_mul_const_vec2_neon(
                    $v[$v0], $v[$v1], $tw0[$t0], $tw1[$t0], $tw0[$t1], $tw1[$t1],
                );
                let nu0 = veorq_u64($v[$u0], p0);
                $v[$v0] = veorq_u64($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = veorq_u64($v[$u1], p1);
                $v[$v1] = veorq_u64($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v0 = [z; 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v0[i] = vld1q_u64(src_ptr.add(off));
                }
                let mut v1 = v0;
                btf2_neon!(v0, tw0_r0, tw0_r1, 0, 8, 0, 1, 9, 0);
                btf2_neon!(v0, tw0_r0, tw0_r1, 2, 10, 0, 3, 11, 0);
                btf2_neon!(v0, tw0_r0, tw0_r1, 4, 12, 0, 5, 13, 0);
                btf2_neon!(v0, tw0_r0, tw0_r1, 6, 14, 0, 7, 15, 0);
                btf2_neon!(v0, tw0_r0, tw0_r1, 0, 4, 1, 1, 5, 1);
                btf2_neon!(v0, tw0_r0, tw0_r1, 2, 6, 1, 3, 7, 1);
                btf2_neon!(v0, tw0_r0, tw0_r1, 8, 12, 2, 9, 13, 2);
                btf2_neon!(v0, tw0_r0, tw0_r1, 10, 14, 2, 11, 15, 2);
                btf2_neon!(v0, tw0_r0, tw0_r1, 0, 2, 3, 1, 3, 3);
                btf2_neon!(v0, tw0_r0, tw0_r1, 4, 6, 4, 5, 7, 4);
                btf2_neon!(v0, tw0_r0, tw0_r1, 8, 10, 5, 9, 11, 5);
                btf2_neon!(v0, tw0_r0, tw0_r1, 12, 14, 6, 13, 15, 6);
                btf2_neon!(v0, tw0_r0, tw0_r1, 0, 1, 7, 2, 3, 8);
                btf2_neon!(v0, tw0_r0, tw0_r1, 4, 5, 9, 6, 7, 10);
                btf2_neon!(v0, tw0_r0, tw0_r1, 8, 9, 11, 10, 11, 12);
                btf2_neon!(v0, tw0_r0, tw0_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    vst1q_u64(dst0_ptr.add(off), v0[i]);
                }
                btf2_neon!(v1, tw1_r0, tw1_r1, 0, 8, 0, 1, 9, 0);
                btf2_neon!(v1, tw1_r0, tw1_r1, 2, 10, 0, 3, 11, 0);
                btf2_neon!(v1, tw1_r0, tw1_r1, 4, 12, 0, 5, 13, 0);
                btf2_neon!(v1, tw1_r0, tw1_r1, 6, 14, 0, 7, 15, 0);
                btf2_neon!(v1, tw1_r0, tw1_r1, 0, 4, 1, 1, 5, 1);
                btf2_neon!(v1, tw1_r0, tw1_r1, 2, 6, 1, 3, 7, 1);
                btf2_neon!(v1, tw1_r0, tw1_r1, 8, 12, 2, 9, 13, 2);
                btf2_neon!(v1, tw1_r0, tw1_r1, 10, 14, 2, 11, 15, 2);
                btf2_neon!(v1, tw1_r0, tw1_r1, 0, 2, 3, 1, 3, 3);
                btf2_neon!(v1, tw1_r0, tw1_r1, 4, 6, 4, 5, 7, 4);
                btf2_neon!(v1, tw1_r0, tw1_r1, 8, 10, 5, 9, 11, 5);
                btf2_neon!(v1, tw1_r0, tw1_r1, 12, 14, 6, 13, 15, 6);
                btf2_neon!(v1, tw1_r0, tw1_r1, 0, 1, 7, 2, 3, 8);
                btf2_neon!(v1, tw1_r0, tw1_r1, 4, 5, 9, 6, 7, 10);
                btf2_neon!(v1, tw1_r0, tw1_r1, 8, 9, 11, 10, 11, 12);
                btf2_neon!(v1, tw1_r0, tw1_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    vst1q_u64(dst1_ptr.add(off), v1[i]);
                }
            }
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        for i in 0..(16 * sixteenth * num_ntts * 2) {
            let val = *src_ptr.add(i);
            *dst0_ptr.add(i) = val;
            *dst1_ptr.add(i) = val;
        }
        k0_ntt_butterfly_fused_4layer_rows_slice(
            dst0_ptr, sixteenth, num_ntts, r_start, r_count, tw0_ptr,
        );
        k0_ntt_butterfly_fused_4layer_rows_slice(
            dst1_ptr, sixteenth, num_ntts, r_start, r_count, tw1_ptr,
        );
    }
}
#[inline(always)]
pub unsafe fn k0_ntt_butterfly_fused_4layer_dual_from_src_row_slice(
    src_ptr: *const u64,
    dst0_ptr: *mut u64,
    dst1_ptr: *mut u64,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    tw0_ptr: *const u64,
    tw1_ptr: *const u64,
) {
    unsafe {
        k0_ntt_butterfly_fused_4layer_dual_from_src_rows_slice(
            src_ptr, dst0_ptr, dst1_ptr, sixteenth, num_ntts, r, 1, tw0_ptr, tw1_ptr,
        );
    }
}
#[inline(always)]
pub unsafe fn k0_ntt_butterfly_fused_3layer_row_slice(
    ptr: *mut u64,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    tw_ptr: *const u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut tw_r0 = [_mm_setzero_si128(); 7];
        let mut tw_r1 = [_mm_setzero_si128(); 7];
        for i in 0..7 {
            let t = _mm_loadu_si128(tw_ptr.add(2 * i) as *const __m128i);
            let (r0, r1) = k0_f128_pre_reduce_const_m128i(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        macro_rules! btf2_x86 {
            ($v:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = k0_f128_mul_const_vec2_m128i(
                    $v[$v0], $v[$v1], tw_r0[$t0], tw_r1[$t0], tw_r0[$t1], tw_r1[$t1],
                );
                let nu0 = _mm_xor_si128($v[$u0], p0);
                $v[$v0] = _mm_xor_si128($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = _mm_xor_si128($v[$u1], p1);
                $v[$v1] = _mm_xor_si128($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        for lane in 0..num_ntts {
            let mut v = [_mm_setzero_si128(); 8];
            for i in 0..8 {
                let off = 2 * ((i * eighth + r) * num_ntts + lane);
                v[i] = _mm_loadu_si128(ptr.add(off) as *const __m128i);
            }
            btf2_x86!(v, 0, 4, 0, 1, 5, 0);
            btf2_x86!(v, 2, 6, 0, 3, 7, 0);
            btf2_x86!(v, 0, 2, 1, 1, 3, 1);
            btf2_x86!(v, 4, 6, 2, 5, 7, 2);
            btf2_x86!(v, 0, 1, 3, 2, 3, 4);
            btf2_x86!(v, 4, 5, 5, 6, 7, 6);
            for i in 0..8 {
                let off = 2 * ((i * eighth + r) * num_ntts + lane);
                _mm_storeu_si128(ptr.add(off) as *mut __m128i, v[i]);
            }
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let mut tw_r0 = [z; 7];
        let mut tw_r1 = [z; 7];
        for i in 0..7 {
            let t = vld1q_u64(tw_ptr.add(2 * i));
            let (r0, r1) = k0_f128_pre_reduce_const_neon(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        macro_rules! btf2_neon {
            ($v:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = k0_f128_mul_const_vec2_neon(
                    $v[$v0], $v[$v1], tw_r0[$t0], tw_r1[$t0], tw_r0[$t1], tw_r1[$t1],
                );
                let nu0 = veorq_u64($v[$u0], p0);
                $v[$v0] = veorq_u64($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = veorq_u64($v[$u1], p1);
                $v[$v1] = veorq_u64($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        for lane in 0..num_ntts {
            let mut v = [z; 8];
            for i in 0..8 {
                let off = 2 * ((i * eighth + r) * num_ntts + lane);
                v[i] = vld1q_u64(ptr.add(off));
            }
            btf2_neon!(v, 0, 4, 0, 1, 5, 0);
            btf2_neon!(v, 2, 6, 0, 3, 7, 0);
            btf2_neon!(v, 0, 2, 1, 1, 3, 1);
            btf2_neon!(v, 4, 6, 2, 5, 7, 2);
            btf2_neon!(v, 0, 1, 3, 2, 3, 4);
            btf2_neon!(v, 4, 5, 5, 6, 7, 6);
            for i in 0..8 {
                let off = 2 * ((i * eighth + r) * num_ntts + lane);
                vst1q_u64(ptr.add(off), v[i]);
            }
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let mut tw = [(0u64, 0u64); 7];
        for i in 0..7 {
            tw[i] = (*tw_ptr.add(2 * i), *tw_ptr.add(2 * i + 1));
        }
        macro_rules! btf2 {
            ($v:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0_lo, p0_hi, p1_lo, p1_hi) = k0_f128_mul_vec2_raw(
                    $v[$v0].0, $v[$v0].1, $v[$v1].0, $v[$v1].1, tw[$t0].0, tw[$t0].1, tw[$t1].0,
                    tw[$t1].1,
                );
                let nu0 = ($v[$u0].0 ^ p0_lo, $v[$u0].1 ^ p0_hi);
                $v[$v0] = ($v[$v0].0 ^ nu0.0, $v[$v0].1 ^ nu0.1);
                $v[$u0] = nu0;
                let nu1 = ($v[$u1].0 ^ p1_lo, $v[$u1].1 ^ p1_hi);
                $v[$v1] = ($v[$v1].0 ^ nu1.0, $v[$v1].1 ^ nu1.1);
                $v[$u1] = nu1;
            }};
        }
        for lane in 0..num_ntts {
            let mut v = [(0u64, 0u64); 8];
            for i in 0..8 {
                let off = 2 * ((i * eighth + r) * num_ntts + lane);
                v[i] = (*ptr.add(off), *ptr.add(off + 1));
            }
            btf2!(v, 0, 4, 0, 1, 5, 0);
            btf2!(v, 2, 6, 0, 3, 7, 0);
            btf2!(v, 0, 2, 1, 1, 3, 1);
            btf2!(v, 4, 6, 2, 5, 7, 2);
            btf2!(v, 0, 1, 3, 2, 3, 4);
            btf2!(v, 4, 5, 5, 6, 7, 6);
            for i in 0..8 {
                let off = 2 * ((i * eighth + r) * num_ntts + lane);
                *ptr.add(off) = v[i].0;
                *ptr.add(off + 1) = v[i].1;
            }
        }
    }
}
#[inline(always)]
pub unsafe fn k0_f128_fold_pairs_slice(
    src_ptr: *const u64,
    base: usize,
    dst_ptr: *mut u64,
    dst_len: usize,
    r_lo: u64,
    r_hi: u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let vr = _mm_set_epi64x(r_hi as i64, r_lo as i64);
        let (vr0, vr1) = k0_f128_pre_reduce_const_m128i(vr);
        let lanes = dst_len & !1;
        let mut t = 0usize;
        while t < lanes {
            let s = 4 * (base + t);
            let e0 = _mm_loadu_si128(src_ptr.add(s) as *const __m128i);
            let o0 = _mm_loadu_si128(src_ptr.add(s + 2) as *const __m128i);
            let e1 = _mm_loadu_si128(src_ptr.add(s + 4) as *const __m128i);
            let o1 = _mm_loadu_si128(src_ptr.add(s + 6) as *const __m128i);
            let (p0, p1) = k0_f128_mul_const_vec2_m128i(
                _mm_xor_si128(e0, o0),
                _mm_xor_si128(e1, o1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let d = 2 * t;
            _mm_storeu_si128(dst_ptr.add(d) as *mut __m128i, _mm_xor_si128(e0, p0));
            _mm_storeu_si128(dst_ptr.add(d + 2) as *mut __m128i, _mm_xor_si128(e1, p1));
            t += 2;
        }
        while t < dst_len {
            let s = 4 * (base + t);
            let e = _mm_loadu_si128(src_ptr.add(s) as *const __m128i);
            let o = _mm_loadu_si128(src_ptr.add(s + 2) as *const __m128i);
            let p = k0_f128_mul_const_m128i(_mm_xor_si128(e, o), vr0, vr1);
            let d = 2 * t;
            _mm_storeu_si128(dst_ptr.add(d) as *mut __m128i, _mm_xor_si128(e, p));
            t += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let r_arr = [r_lo, r_hi];
        let vr = vld1q_u64(r_arr.as_ptr());
        let (vr0, vr1) = k0_f128_pre_reduce_const_neon(vr);
        let lanes = dst_len & !1;
        let mut t = 0usize;
        while t < lanes {
            let s = 4 * (base + t);
            let e0 = vld1q_u64(src_ptr.add(s));
            let o0 = vld1q_u64(src_ptr.add(s + 2));
            let e1 = vld1q_u64(src_ptr.add(s + 4));
            let o1 = vld1q_u64(src_ptr.add(s + 6));
            let (p0, p1) = k0_f128_mul_const_vec2_neon(
                veorq_u64(e0, o0),
                veorq_u64(e1, o1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let d = 2 * t;
            vst1q_u64(dst_ptr.add(d), veorq_u64(e0, p0));
            vst1q_u64(dst_ptr.add(d + 2), veorq_u64(e1, p1));
            t += 2;
        }
        while t < dst_len {
            let s = 4 * (base + t);
            let e = vld1q_u64(src_ptr.add(s));
            let o = vld1q_u64(src_ptr.add(s + 2));
            let p = k0_f128_mul_const_neon(veorq_u64(e, o), vr0, vr1);
            let d = 2 * t;
            vst1q_u64(dst_ptr.add(d), veorq_u64(e, p));
            t += 1;
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let lanes = dst_len & !1;
        let mut t = 0usize;
        while t < lanes {
            let s = 4 * (base + t);
            let e0_lo = *src_ptr.add(s);
            let e0_hi = *src_ptr.add(s + 1);
            let o0_lo = *src_ptr.add(s + 2);
            let o0_hi = *src_ptr.add(s + 3);
            let e1_lo = *src_ptr.add(s + 4);
            let e1_hi = *src_ptr.add(s + 5);
            let o1_lo = *src_ptr.add(s + 6);
            let o1_hi = *src_ptr.add(s + 7);
            let (p0_lo, p0_hi, p1_lo, p1_hi) = k0_f128_mul_vec2_raw(
                e0_lo ^ o0_lo,
                e0_hi ^ o0_hi,
                e1_lo ^ o1_lo,
                e1_hi ^ o1_hi,
                r_lo,
                r_hi,
                r_lo,
                r_hi,
            );
            let d = 2 * t;
            *dst_ptr.add(d) = e0_lo ^ p0_lo;
            *dst_ptr.add(d + 1) = e0_hi ^ p0_hi;
            *dst_ptr.add(d + 2) = e1_lo ^ p1_lo;
            *dst_ptr.add(d + 3) = e1_hi ^ p1_hi;
            t += 2;
        }
        while t < dst_len {
            let s = 4 * (base + t);
            let e_lo = *src_ptr.add(s);
            let e_hi = *src_ptr.add(s + 1);
            let o_lo = *src_ptr.add(s + 2);
            let o_hi = *src_ptr.add(s + 3);
            let (p_lo, p_hi) = k0_f128_mul_raw(e_lo ^ o_lo, e_hi ^ o_hi, r_lo, r_hi);
            let d = 2 * t;
            *dst_ptr.add(d) = e_lo ^ p_lo;
            *dst_ptr.add(d + 1) = e_hi ^ p_hi;
            t += 1;
        }
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_fold_pairs_and_msg_slice(
    f_ptr: *const u64,
    b_ptr: *const u64,
    base: usize,
    fc_ptr: *mut u64,
    bc_ptr: *mut u64,
    len: usize,
    r_lo: u64,
    r_hi: u64,
) -> (u64, u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let vr = _mm_set_epi64x(r_hi as i64, r_lo as i64);
        let (vr0, vr1) = k0_f128_pre_reduce_const_m128i(vr);
        let lanes = len & !1;
        let (mut u0_ll, mut u0_mid, mut u0_hh) = (
            _mm_setzero_si128(),
            _mm_setzero_si128(),
            _mm_setzero_si128(),
        );
        let (mut u2_ll, mut u2_mid, mut u2_hh) = (
            _mm_setzero_si128(),
            _mm_setzero_si128(),
            _mm_setzero_si128(),
        );
        let mut t = 0usize;
        while t < lanes {
            let s = 4 * (base + t);
            let fe0 = _mm_loadu_si128(f_ptr.add(s) as *const __m128i);
            let fo0 = _mm_loadu_si128(f_ptr.add(s + 2) as *const __m128i);
            let fe1 = _mm_loadu_si128(f_ptr.add(s + 4) as *const __m128i);
            let fo1 = _mm_loadu_si128(f_ptr.add(s + 6) as *const __m128i);
            let (fp0, fp1) = k0_f128_mul_const_vec2_m128i(
                _mm_xor_si128(fe0, fo0),
                _mm_xor_si128(fe1, fo1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let nf0 = _mm_xor_si128(fe0, fp0);
            let nf1 = _mm_xor_si128(fe1, fp1);
            let d = 2 * t;
            _mm_storeu_si128(fc_ptr.add(d) as *mut __m128i, nf0);
            _mm_storeu_si128(fc_ptr.add(d + 2) as *mut __m128i, nf1);
            let be0 = _mm_loadu_si128(b_ptr.add(s) as *const __m128i);
            let bo0 = _mm_loadu_si128(b_ptr.add(s + 2) as *const __m128i);
            let be1 = _mm_loadu_si128(b_ptr.add(s + 4) as *const __m128i);
            let bo1 = _mm_loadu_si128(b_ptr.add(s + 6) as *const __m128i);
            let (bp0, bp1) = k0_f128_mul_const_vec2_m128i(
                _mm_xor_si128(be0, bo0),
                _mm_xor_si128(be1, bo1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let nb0 = _mm_xor_si128(be0, bp0);
            let nb1 = _mm_xor_si128(be1, bp1);
            _mm_storeu_si128(bc_ptr.add(d) as *mut __m128i, nb0);
            _mm_storeu_si128(bc_ptr.add(d + 2) as *mut __m128i, nb1);
            k0_f128_mul_acc_m128i(nf0, nb0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            k0_f128_mul_acc_m128i(
                _mm_xor_si128(nf0, nf1),
                _mm_xor_si128(nb0, nb1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            t += 2;
        }
        while t < len {
            let s = 4 * (base + t);
            let fe = _mm_loadu_si128(f_ptr.add(s) as *const __m128i);
            let fo = _mm_loadu_si128(f_ptr.add(s + 2) as *const __m128i);
            let fp = k0_f128_mul_const_m128i(_mm_xor_si128(fe, fo), vr0, vr1);
            let be = _mm_loadu_si128(b_ptr.add(s) as *const __m128i);
            let bo = _mm_loadu_si128(b_ptr.add(s + 2) as *const __m128i);
            let bp = k0_f128_mul_const_m128i(_mm_xor_si128(be, bo), vr0, vr1);
            let d = 2 * t;
            _mm_storeu_si128(fc_ptr.add(d) as *mut __m128i, _mm_xor_si128(fe, fp));
            _mm_storeu_si128(bc_ptr.add(d) as *mut __m128i, _mm_xor_si128(be, bp));
            t += 1;
        }
        let u0 = k0_f128_reduce_3lane_m128i(u0_ll, u0_mid, u0_hh);
        let u2 = k0_f128_reduce_3lane_m128i(u2_ll, u2_mid, u2_hh);
        return (
            _mm_extract_epi64::<0>(u0) as u64,
            _mm_extract_epi64::<1>(u0) as u64,
            _mm_extract_epi64::<0>(u2) as u64,
            _mm_extract_epi64::<1>(u2) as u64,
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let r_arr = [r_lo, r_hi];
        let vr = vld1q_u64(r_arr.as_ptr());
        let (vr0, vr1) = k0_f128_pre_reduce_const_neon(vr);
        let lanes = len & !1;
        let z = vdupq_n_u64(0);
        let (mut u0_ll, mut u0_mid, mut u0_hh) = (z, z, z);
        let (mut u2_ll, mut u2_mid, mut u2_hh) = (z, z, z);
        let mut t = 0usize;
        while t < lanes {
            let s = 4 * (base + t);
            let fe0 = vld1q_u64(f_ptr.add(s));
            let fo0 = vld1q_u64(f_ptr.add(s + 2));
            let fe1 = vld1q_u64(f_ptr.add(s + 4));
            let fo1 = vld1q_u64(f_ptr.add(s + 6));
            let (fp0, fp1) = k0_f128_mul_const_vec2_neon(
                veorq_u64(fe0, fo0),
                veorq_u64(fe1, fo1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let nf0 = veorq_u64(fe0, fp0);
            let nf1 = veorq_u64(fe1, fp1);
            let d = 2 * t;
            vst1q_u64(fc_ptr.add(d), nf0);
            vst1q_u64(fc_ptr.add(d + 2), nf1);
            let be0 = vld1q_u64(b_ptr.add(s));
            let bo0 = vld1q_u64(b_ptr.add(s + 2));
            let be1 = vld1q_u64(b_ptr.add(s + 4));
            let bo1 = vld1q_u64(b_ptr.add(s + 6));
            let (bp0, bp1) = k0_f128_mul_const_vec2_neon(
                veorq_u64(be0, bo0),
                veorq_u64(be1, bo1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let nb0 = veorq_u64(be0, bp0);
            let nb1 = veorq_u64(be1, bp1);
            vst1q_u64(bc_ptr.add(d), nb0);
            vst1q_u64(bc_ptr.add(d + 2), nb1);
            k0_f128_mul_acc_neon(nf0, nb0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            k0_f128_mul_acc_neon(
                veorq_u64(nf0, nf1),
                veorq_u64(nb0, nb1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            t += 2;
        }
        while t < len {
            let s = 4 * (base + t);
            let fe = vld1q_u64(f_ptr.add(s));
            let fo = vld1q_u64(f_ptr.add(s + 2));
            let fp = k0_f128_mul_const_neon(veorq_u64(fe, fo), vr0, vr1);
            let be = vld1q_u64(b_ptr.add(s));
            let bo = vld1q_u64(b_ptr.add(s + 2));
            let bp = k0_f128_mul_const_neon(veorq_u64(be, bo), vr0, vr1);
            let d = 2 * t;
            vst1q_u64(fc_ptr.add(d), veorq_u64(fe, fp));
            vst1q_u64(bc_ptr.add(d), veorq_u64(be, bp));
            t += 1;
        }
        let u0 = k0_f128_reduce_3lane_neon(u0_ll, u0_mid, u0_hh);
        let u2 = k0_f128_reduce_3lane_neon(u2_ll, u2_mid, u2_hh);
        return (
            vgetq_lane_u64::<0>(u0),
            vgetq_lane_u64::<1>(u0),
            vgetq_lane_u64::<0>(u2),
            vgetq_lane_u64::<1>(u2),
        );
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let lanes = len & !1;
        let (mut u0_r0, mut u0_r1, mut u0_r2, mut u0_r3) = (0u64, 0u64, 0u64, 0u64);
        let (mut u2_r0, mut u2_r1, mut u2_r2, mut u2_r3) = (0u64, 0u64, 0u64, 0u64);
        let mut t = 0usize;
        while t < lanes {
            let s = 4 * (base + t);
            let fe0_lo = *f_ptr.add(s);
            let fe0_hi = *f_ptr.add(s + 1);
            let fo0_lo = *f_ptr.add(s + 2);
            let fo0_hi = *f_ptr.add(s + 3);
            let fe1_lo = *f_ptr.add(s + 4);
            let fe1_hi = *f_ptr.add(s + 5);
            let fo1_lo = *f_ptr.add(s + 6);
            let fo1_hi = *f_ptr.add(s + 7);
            let (fp0_lo, fp0_hi, fp1_lo, fp1_hi) = k0_f128_mul_vec2_raw(
                fe0_lo ^ fo0_lo,
                fe0_hi ^ fo0_hi,
                fe1_lo ^ fo1_lo,
                fe1_hi ^ fo1_hi,
                r_lo,
                r_hi,
                r_lo,
                r_hi,
            );
            let nf0_lo = fe0_lo ^ fp0_lo;
            let nf0_hi = fe0_hi ^ fp0_hi;
            let nf1_lo = fe1_lo ^ fp1_lo;
            let nf1_hi = fe1_hi ^ fp1_hi;
            let d = 2 * t;
            *fc_ptr.add(d) = nf0_lo;
            *fc_ptr.add(d + 1) = nf0_hi;
            *fc_ptr.add(d + 2) = nf1_lo;
            *fc_ptr.add(d + 3) = nf1_hi;
            let be0_lo = *b_ptr.add(s);
            let be0_hi = *b_ptr.add(s + 1);
            let bo0_lo = *b_ptr.add(s + 2);
            let bo0_hi = *b_ptr.add(s + 3);
            let be1_lo = *b_ptr.add(s + 4);
            let be1_hi = *b_ptr.add(s + 5);
            let bo1_lo = *b_ptr.add(s + 6);
            let bo1_hi = *b_ptr.add(s + 7);
            let (bp0_lo, bp0_hi, bp1_lo, bp1_hi) = k0_f128_mul_vec2_raw(
                be0_lo ^ bo0_lo,
                be0_hi ^ bo0_hi,
                be1_lo ^ bo1_lo,
                be1_hi ^ bo1_hi,
                r_lo,
                r_hi,
                r_lo,
                r_hi,
            );
            let nb0_lo = be0_lo ^ bp0_lo;
            let nb0_hi = be0_hi ^ bp0_hi;
            let nb1_lo = be1_lo ^ bp1_lo;
            let nb1_hi = be1_hi ^ bp1_hi;
            *bc_ptr.add(d) = nb0_lo;
            *bc_ptr.add(d + 1) = nb0_hi;
            *bc_ptr.add(d + 2) = nb1_lo;
            *bc_ptr.add(d + 3) = nb1_hi;
            let (m0_0, m0_1, m0_2, m0_3) =
                k0_f128_mul_unreduced_raw(nf0_lo, nf0_hi, nb0_lo, nb0_hi);
            let (m2_0, m2_1, m2_2, m2_3) = k0_f128_mul_unreduced_raw(
                nf0_lo ^ nf1_lo,
                nf0_hi ^ nf1_hi,
                nb0_lo ^ nb1_lo,
                nb0_hi ^ nb1_hi,
            );
            u0_r0 ^= m0_0;
            u0_r1 ^= m0_1;
            u0_r2 ^= m0_2;
            u0_r3 ^= m0_3;
            u2_r0 ^= m2_0;
            u2_r1 ^= m2_1;
            u2_r2 ^= m2_2;
            u2_r3 ^= m2_3;
            t += 2;
        }
        while t < len {
            let s = 4 * (base + t);
            let fe_lo = *f_ptr.add(s);
            let fe_hi = *f_ptr.add(s + 1);
            let fo_lo = *f_ptr.add(s + 2);
            let fo_hi = *f_ptr.add(s + 3);
            let (fp_lo, fp_hi) = k0_f128_mul_raw(fe_lo ^ fo_lo, fe_hi ^ fo_hi, r_lo, r_hi);
            let be_lo = *b_ptr.add(s);
            let be_hi = *b_ptr.add(s + 1);
            let bo_lo = *b_ptr.add(s + 2);
            let bo_hi = *b_ptr.add(s + 3);
            let (bp_lo, bp_hi) = k0_f128_mul_raw(be_lo ^ bo_lo, be_hi ^ bo_hi, r_lo, r_hi);
            let d = 2 * t;
            *fc_ptr.add(d) = fe_lo ^ fp_lo;
            *fc_ptr.add(d + 1) = fe_hi ^ fp_hi;
            *bc_ptr.add(d) = be_lo ^ bp_lo;
            *bc_ptr.add(d + 1) = be_hi ^ bp_hi;
            t += 1;
        }
        let (u0_lo, u0_hi) = k0_f128_reduce_256_raw(u0_r0, u0_r1, u0_r2, u0_r3);
        let (u2_lo, u2_hi) = k0_f128_reduce_256_raw(u2_r0, u2_r1, u2_r2, u2_r3);
        (u0_lo, u0_hi, u2_lo, u2_hi)
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_fold_and_msg_round_slice(
    a_in_ptr: *const u64,
    b_in_ptr: *const u64,
    a_out_ptr: *mut u64,
    b_out_ptr: *mut u64,
    eq_lo_ptr: *const u64,
    lo_size: usize,
    r_lo: u64,
    r_hi: u64,
) -> (u64, u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let vr = _mm_set_epi64x(r_hi as i64, r_lo as i64);
        let (vr0, vr1) = k0_f128_pre_reduce_const_m128i(vr);
        let (mut p1_ll, mut p1_mid, mut p1_hh) = (
            _mm_setzero_si128(),
            _mm_setzero_si128(),
            _mm_setzero_si128(),
        );
        let (mut pi_ll, mut pi_mid, mut pi_hh) = (
            _mm_setzero_si128(),
            _mm_setzero_si128(),
            _mm_setzero_si128(),
        );
        for x_lo in 0..lo_size {
            let s = 8 * x_lo;
            let ae0 = _mm_loadu_si128(a_in_ptr.add(s) as *const __m128i);
            let ao0 = _mm_loadu_si128(a_in_ptr.add(s + 2) as *const __m128i);
            let ae1 = _mm_loadu_si128(a_in_ptr.add(s + 4) as *const __m128i);
            let ao1 = _mm_loadu_si128(a_in_ptr.add(s + 6) as *const __m128i);
            let (ap0, ap1) = k0_f128_mul_const_vec2_m128i(
                _mm_xor_si128(ae0, ao0),
                _mm_xor_si128(ae1, ao1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let a0 = _mm_xor_si128(ae0, ap0);
            let a1 = _mm_xor_si128(ae1, ap1);
            let be0 = _mm_loadu_si128(b_in_ptr.add(s) as *const __m128i);
            let bo0 = _mm_loadu_si128(b_in_ptr.add(s + 2) as *const __m128i);
            let be1 = _mm_loadu_si128(b_in_ptr.add(s + 4) as *const __m128i);
            let bo1 = _mm_loadu_si128(b_in_ptr.add(s + 6) as *const __m128i);
            let (bp0, bp1) = k0_f128_mul_const_vec2_m128i(
                _mm_xor_si128(be0, bo0),
                _mm_xor_si128(be1, bo1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let b0 = _mm_xor_si128(be0, bp0);
            let b1 = _mm_xor_si128(be1, bp1);
            let d = 4 * x_lo;
            _mm_storeu_si128(a_out_ptr.add(d) as *mut __m128i, a0);
            _mm_storeu_si128(a_out_ptr.add(d + 2) as *mut __m128i, a1);
            _mm_storeu_si128(b_out_ptr.add(d) as *mut __m128i, b0);
            _mm_storeu_si128(b_out_ptr.add(d + 2) as *mut __m128i, b1);
            let (g1, gi) =
                k0_f128_mul_vec2_m128i(a1, _mm_xor_si128(a0, a1), b1, _mm_xor_si128(b0, b1));
            let eq = _mm_loadu_si128(eq_lo_ptr.add(2 * x_lo) as *const __m128i);
            k0_f128_mul_acc_m128i(eq, g1, &mut p1_ll, &mut p1_mid, &mut p1_hh);
            k0_f128_mul_acc_m128i(eq, gi, &mut pi_ll, &mut pi_mid, &mut pi_hh);
        }
        let p1 = k0_f128_reduce_3lane_m128i(p1_ll, p1_mid, p1_hh);
        let pi = k0_f128_reduce_3lane_m128i(pi_ll, pi_mid, pi_hh);
        return (
            _mm_extract_epi64::<0>(p1) as u64,
            _mm_extract_epi64::<1>(p1) as u64,
            _mm_extract_epi64::<0>(pi) as u64,
            _mm_extract_epi64::<1>(pi) as u64,
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let r_arr = [r_lo, r_hi];
        let vr = vld1q_u64(r_arr.as_ptr());
        let (vr0, vr1) = k0_f128_pre_reduce_const_neon(vr);
        let z = vdupq_n_u64(0);
        let (mut p1_ll, mut p1_mid, mut p1_hh) = (z, z, z);
        let (mut pi_ll, mut pi_mid, mut pi_hh) = (z, z, z);
        for x_lo in 0..lo_size {
            let s = 8 * x_lo;
            let ae0 = vld1q_u64(a_in_ptr.add(s));
            let ao0 = vld1q_u64(a_in_ptr.add(s + 2));
            let ae1 = vld1q_u64(a_in_ptr.add(s + 4));
            let ao1 = vld1q_u64(a_in_ptr.add(s + 6));
            let (ap0, ap1) = k0_f128_mul_const_vec2_neon(
                veorq_u64(ae0, ao0),
                veorq_u64(ae1, ao1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let a0 = veorq_u64(ae0, ap0);
            let a1 = veorq_u64(ae1, ap1);
            let be0 = vld1q_u64(b_in_ptr.add(s));
            let bo0 = vld1q_u64(b_in_ptr.add(s + 2));
            let be1 = vld1q_u64(b_in_ptr.add(s + 4));
            let bo1 = vld1q_u64(b_in_ptr.add(s + 6));
            let (bp0, bp1) = k0_f128_mul_const_vec2_neon(
                veorq_u64(be0, bo0),
                veorq_u64(be1, bo1),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let b0 = veorq_u64(be0, bp0);
            let b1 = veorq_u64(be1, bp1);
            let d = 4 * x_lo;
            vst1q_u64(a_out_ptr.add(d), a0);
            vst1q_u64(a_out_ptr.add(d + 2), a1);
            vst1q_u64(b_out_ptr.add(d), b0);
            vst1q_u64(b_out_ptr.add(d + 2), b1);
            let (g1, gi) = k0_f128_mul_vec2_neon(a1, veorq_u64(a0, a1), b1, veorq_u64(b0, b1));
            let eq = vld1q_u64(eq_lo_ptr.add(2 * x_lo));
            k0_f128_mul_acc_neon(eq, g1, &mut p1_ll, &mut p1_mid, &mut p1_hh);
            k0_f128_mul_acc_neon(eq, gi, &mut pi_ll, &mut pi_mid, &mut pi_hh);
        }
        let p1 = k0_f128_reduce_3lane_neon(p1_ll, p1_mid, p1_hh);
        let pi = k0_f128_reduce_3lane_neon(pi_ll, pi_mid, pi_hh);
        return (
            vgetq_lane_u64::<0>(p1),
            vgetq_lane_u64::<1>(p1),
            vgetq_lane_u64::<0>(pi),
            vgetq_lane_u64::<1>(pi),
        );
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let (mut p1_r0, mut p1_r1, mut p1_r2, mut p1_r3) = (0u64, 0u64, 0u64, 0u64);
        let (mut pi_r0, mut pi_r1, mut pi_r2, mut pi_r3) = (0u64, 0u64, 0u64, 0u64);
        for x_lo in 0..lo_size {
            let s = 8 * x_lo;
            let ae0_lo = *a_in_ptr.add(s);
            let ae0_hi = *a_in_ptr.add(s + 1);
            let ao0_lo = *a_in_ptr.add(s + 2);
            let ao0_hi = *a_in_ptr.add(s + 3);
            let ae1_lo = *a_in_ptr.add(s + 4);
            let ae1_hi = *a_in_ptr.add(s + 5);
            let ao1_lo = *a_in_ptr.add(s + 6);
            let ao1_hi = *a_in_ptr.add(s + 7);
            let (ap0_lo, ap0_hi, ap1_lo, ap1_hi) = k0_f128_mul_vec2_raw(
                ae0_lo ^ ao0_lo,
                ae0_hi ^ ao0_hi,
                ae1_lo ^ ao1_lo,
                ae1_hi ^ ao1_hi,
                r_lo,
                r_hi,
                r_lo,
                r_hi,
            );
            let a0_lo = ae0_lo ^ ap0_lo;
            let a0_hi = ae0_hi ^ ap0_hi;
            let a1_lo = ae1_lo ^ ap1_lo;
            let a1_hi = ae1_hi ^ ap1_hi;
            let be0_lo = *b_in_ptr.add(s);
            let be0_hi = *b_in_ptr.add(s + 1);
            let bo0_lo = *b_in_ptr.add(s + 2);
            let bo0_hi = *b_in_ptr.add(s + 3);
            let be1_lo = *b_in_ptr.add(s + 4);
            let be1_hi = *b_in_ptr.add(s + 5);
            let bo1_lo = *b_in_ptr.add(s + 6);
            let bo1_hi = *b_in_ptr.add(s + 7);
            let (bp0_lo, bp0_hi, bp1_lo, bp1_hi) = k0_f128_mul_vec2_raw(
                be0_lo ^ bo0_lo,
                be0_hi ^ bo0_hi,
                be1_lo ^ bo1_lo,
                be1_hi ^ bo1_hi,
                r_lo,
                r_hi,
                r_lo,
                r_hi,
            );
            let b0_lo = be0_lo ^ bp0_lo;
            let b0_hi = be0_hi ^ bp0_hi;
            let b1_lo = be1_lo ^ bp1_lo;
            let b1_hi = be1_hi ^ bp1_hi;
            let d = 4 * x_lo;
            *a_out_ptr.add(d) = a0_lo;
            *a_out_ptr.add(d + 1) = a0_hi;
            *a_out_ptr.add(d + 2) = a1_lo;
            *a_out_ptr.add(d + 3) = a1_hi;
            *b_out_ptr.add(d) = b0_lo;
            *b_out_ptr.add(d + 1) = b0_hi;
            *b_out_ptr.add(d + 2) = b1_lo;
            *b_out_ptr.add(d + 3) = b1_hi;
            let (g1_lo, g1_hi, gi_lo, gi_hi) = k0_f128_mul_vec2_raw(
                a1_lo,
                a1_hi,
                a0_lo ^ a1_lo,
                a0_hi ^ a1_hi,
                b1_lo,
                b1_hi,
                b0_lo ^ b1_lo,
                b0_hi ^ b1_hi,
            );
            let eq_lo = *eq_lo_ptr.add(2 * x_lo);
            let eq_hi = *eq_lo_ptr.add(2 * x_lo + 1);
            let (u1_0, u1_1, u1_2, u1_3) = k0_f128_mul_unreduced_raw(eq_lo, eq_hi, g1_lo, g1_hi);
            let (ui_0, ui_1, ui_2, ui_3) = k0_f128_mul_unreduced_raw(eq_lo, eq_hi, gi_lo, gi_hi);
            p1_r0 ^= u1_0;
            p1_r1 ^= u1_1;
            p1_r2 ^= u1_2;
            p1_r3 ^= u1_3;
            pi_r0 ^= ui_0;
            pi_r1 ^= ui_1;
            pi_r2 ^= ui_2;
            pi_r3 ^= ui_3;
        }
        let (p1_lo, p1_hi) = k0_f128_reduce_256_raw(p1_r0, p1_r1, p1_r2, p1_r3);
        let (pi_lo, pi_hi) = k0_f128_reduce_256_raw(pi_r0, pi_r1, pi_r2, pi_r3);
        (p1_lo, p1_hi, pi_lo, pi_hi)
    }
}
#[inline(always)]
pub unsafe fn k0_f128_inner_product_slice(
    a_ptr: *const u64,
    b_ptr: *const u64,
    n_f128: usize,
) -> (u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let (mut ll, mut mid, mut hh) = (
            _mm_setzero_si128(),
            _mm_setzero_si128(),
            _mm_setzero_si128(),
        );
        for i in 0..n_f128 {
            let idx = 2 * i;
            let va = _mm_loadu_si128(a_ptr.add(idx) as *const __m128i);
            let vb = _mm_loadu_si128(b_ptr.add(idx) as *const __m128i);
            k0_f128_mul_acc_m128i(va, vb, &mut ll, &mut mid, &mut hh);
        }
        let res = k0_f128_reduce_3lane_m128i(ll, mid, hh);
        return (
            _mm_extract_epi64::<0>(res) as u64,
            _mm_extract_epi64::<1>(res) as u64,
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let (mut ll, mut mid, mut hh) = (z, z, z);
        for i in 0..n_f128 {
            let idx = 2 * i;
            let va = vld1q_u64(a_ptr.add(idx));
            let vb = vld1q_u64(b_ptr.add(idx));
            k0_f128_mul_acc_neon(va, vb, &mut ll, &mut mid, &mut hh);
        }
        let res = k0_f128_reduce_3lane_neon(ll, mid, hh);
        return (vgetq_lane_u64::<0>(res), vgetq_lane_u64::<1>(res));
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let (mut r0, mut r1, mut r2, mut r3) = (0u64, 0u64, 0u64, 0u64);
        for i in 0..n_f128 {
            let idx = 2 * i;
            let (u0, u1, u2, u3) = k0_f128_mul_unreduced_raw(
                *a_ptr.add(idx),
                *a_ptr.add(idx + 1),
                *b_ptr.add(idx),
                *b_ptr.add(idx + 1),
            );
            r0 ^= u0;
            r1 ^= u1;
            r2 ^= u2;
            r3 ^= u3;
        }
        k0_f128_reduce_256_raw(r0, r1, r2, r3)
    }
}
#[inline(always)]
pub unsafe fn k0_f128_scalar_mul_into_slice(
    dst_ptr: *mut u64,
    src_ptr: *const u64,
    n_f128: usize,
    alpha_lo: u64,
    alpha_hi: u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let valpha = _mm_set_epi64x(alpha_hi as i64, alpha_lo as i64);
        let (va0, va1) = k0_f128_pre_reduce_const_m128i(valpha);
        let mut i = 0usize;
        while i + 2 <= n_f128 {
            let idx = 2 * i;
            let s0 = _mm_loadu_si128(src_ptr.add(idx) as *const __m128i);
            let s1 = _mm_loadu_si128(src_ptr.add(idx + 2) as *const __m128i);
            let (p0, p1) = k0_f128_mul_const_vec2_m128i(s0, s1, va0, va1, va0, va1);
            _mm_storeu_si128(dst_ptr.add(idx) as *mut __m128i, p0);
            _mm_storeu_si128(dst_ptr.add(idx + 2) as *mut __m128i, p1);
            i += 2;
        }
        while i < n_f128 {
            let idx = 2 * i;
            let s0 = _mm_loadu_si128(src_ptr.add(idx) as *const __m128i);
            let p0 = k0_f128_mul_const_m128i(s0, va0, va1);
            _mm_storeu_si128(dst_ptr.add(idx) as *mut __m128i, p0);
            i += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let alpha_arr = [alpha_lo, alpha_hi];
        let valpha = vld1q_u64(alpha_arr.as_ptr());
        let (va0, va1) = k0_f128_pre_reduce_const_neon(valpha);
        let mut i = 0usize;
        while i + 2 <= n_f128 {
            let idx = 2 * i;
            let s0 = vld1q_u64(src_ptr.add(idx));
            let s1 = vld1q_u64(src_ptr.add(idx + 2));
            let (p0, p1) = k0_f128_mul_const_vec2_neon(s0, s1, va0, va1, va0, va1);
            vst1q_u64(dst_ptr.add(idx), p0);
            vst1q_u64(dst_ptr.add(idx + 2), p1);
            i += 2;
        }
        while i < n_f128 {
            let idx = 2 * i;
            let s0 = vld1q_u64(src_ptr.add(idx));
            let p0 = k0_f128_mul_const_neon(s0, va0, va1);
            vst1q_u64(dst_ptr.add(idx), p0);
            i += 1;
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let mut i = 0usize;
        while i + 2 <= n_f128 {
            let idx = 2 * i;
            let (p0_lo, p0_hi, p1_lo, p1_hi) = k0_f128_mul_vec2_raw(
                *src_ptr.add(idx),
                *src_ptr.add(idx + 1),
                *src_ptr.add(idx + 2),
                *src_ptr.add(idx + 3),
                alpha_lo,
                alpha_hi,
                alpha_lo,
                alpha_hi,
            );
            *dst_ptr.add(idx) = p0_lo;
            *dst_ptr.add(idx + 1) = p0_hi;
            *dst_ptr.add(idx + 2) = p1_lo;
            *dst_ptr.add(idx + 3) = p1_hi;
            i += 2;
        }
        while i < n_f128 {
            let idx = 2 * i;
            let (p_lo, p_hi) =
                k0_f128_mul_raw(*src_ptr.add(idx), *src_ptr.add(idx + 1), alpha_lo, alpha_hi);
            *dst_ptr.add(idx) = p_lo;
            *dst_ptr.add(idx + 1) = p_hi;
            i += 1;
        }
    }
}
#[inline(always)]
pub unsafe fn k0_f128_scalar_mul_add_slice(
    dst_ptr: *mut u64,
    src_ptr: *const u64,
    n_f128: usize,
    alpha_lo: u64,
    alpha_hi: u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let valpha = _mm_set_epi64x(alpha_hi as i64, alpha_lo as i64);
        let (va0, va1) = k0_f128_pre_reduce_const_m128i(valpha);
        let mut i = 0usize;
        while i + 2 <= n_f128 {
            let idx = 2 * i;
            let s0 = _mm_loadu_si128(src_ptr.add(idx) as *const __m128i);
            let s1 = _mm_loadu_si128(src_ptr.add(idx + 2) as *const __m128i);
            let d0 = _mm_loadu_si128(dst_ptr.add(idx) as *const __m128i);
            let d1 = _mm_loadu_si128(dst_ptr.add(idx + 2) as *const __m128i);
            let (p0, p1) = k0_f128_mul_const_vec2_m128i(s0, s1, va0, va1, va0, va1);
            _mm_storeu_si128(dst_ptr.add(idx) as *mut __m128i, _mm_xor_si128(d0, p0));
            _mm_storeu_si128(dst_ptr.add(idx + 2) as *mut __m128i, _mm_xor_si128(d1, p1));
            i += 2;
        }
        while i < n_f128 {
            let idx = 2 * i;
            let s0 = _mm_loadu_si128(src_ptr.add(idx) as *const __m128i);
            let d0 = _mm_loadu_si128(dst_ptr.add(idx) as *const __m128i);
            let p0 = k0_f128_mul_const_m128i(s0, va0, va1);
            _mm_storeu_si128(dst_ptr.add(idx) as *mut __m128i, _mm_xor_si128(d0, p0));
            i += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let alpha_arr = [alpha_lo, alpha_hi];
        let valpha = vld1q_u64(alpha_arr.as_ptr());
        let (va0, va1) = k0_f128_pre_reduce_const_neon(valpha);
        let mut i = 0usize;
        while i + 2 <= n_f128 {
            let idx = 2 * i;
            let s0 = vld1q_u64(src_ptr.add(idx));
            let s1 = vld1q_u64(src_ptr.add(idx + 2));
            let d0 = vld1q_u64(dst_ptr.add(idx));
            let d1 = vld1q_u64(dst_ptr.add(idx + 2));
            let (p0, p1) = k0_f128_mul_const_vec2_neon(s0, s1, va0, va1, va0, va1);
            vst1q_u64(dst_ptr.add(idx), veorq_u64(d0, p0));
            vst1q_u64(dst_ptr.add(idx + 2), veorq_u64(d1, p1));
            i += 2;
        }
        while i < n_f128 {
            let idx = 2 * i;
            let s0 = vld1q_u64(src_ptr.add(idx));
            let d0 = vld1q_u64(dst_ptr.add(idx));
            let p0 = k0_f128_mul_const_neon(s0, va0, va1);
            vst1q_u64(dst_ptr.add(idx), veorq_u64(d0, p0));
            i += 1;
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let mut i = 0usize;
        while i + 2 <= n_f128 {
            let idx = 2 * i;
            let (p0_lo, p0_hi, p1_lo, p1_hi) = k0_f128_mul_vec2_raw(
                *src_ptr.add(idx),
                *src_ptr.add(idx + 1),
                *src_ptr.add(idx + 2),
                *src_ptr.add(idx + 3),
                alpha_lo,
                alpha_hi,
                alpha_lo,
                alpha_hi,
            );
            *dst_ptr.add(idx) ^= p0_lo;
            *dst_ptr.add(idx + 1) ^= p0_hi;
            *dst_ptr.add(idx + 2) ^= p1_lo;
            *dst_ptr.add(idx + 3) ^= p1_hi;
            i += 2;
        }
        while i < n_f128 {
            let idx = 2 * i;
            let (p_lo, p_hi) =
                k0_f128_mul_raw(*src_ptr.add(idx), *src_ptr.add(idx + 1), alpha_lo, alpha_hi);
            *dst_ptr.add(idx) ^= p_lo;
            *dst_ptr.add(idx + 1) ^= p_hi;
            i += 1;
        }
    }
}
#[inline(always)]
pub unsafe fn k0_f128_xor_assign_slice(dst_ptr: *mut u64, src_ptr: *const u64, n_f128: usize) {
    unsafe {
        let n_u64 = 2 * n_f128;
        let mut i = 0usize;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use core::arch::x86_64::*;
            while i + 4 <= n_u64 {
                let d = _mm256_loadu_si256(dst_ptr.add(i) as *const __m256i);
                let s = _mm256_loadu_si256(src_ptr.add(i) as *const __m256i);
                _mm256_storeu_si256(dst_ptr.add(i) as *mut __m256i, _mm256_xor_si256(d, s));
                i += 4;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::*;
            while i + 2 <= n_u64 {
                let d = vld1q_u64(dst_ptr.add(i));
                let s = vld1q_u64(src_ptr.add(i));
                vst1q_u64(dst_ptr.add(i), veorq_u64(d, s));
                i += 2;
            }
        }
        while i < n_u64 {
            *dst_ptr.add(i) ^= *src_ptr.add(i);
            i += 1;
        }
    }
}
#[inline(always)]
pub unsafe fn k0_f128_bit_transpose_64bytes(input: *const u8, output: *mut u8) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let r01 = _mm_loadu_si128(input as *const __m128i);
        let r23 = _mm_loadu_si128(input.add(16) as *const __m128i);
        let r45 = _mm_loadu_si128(input.add(32) as *const __m128i);
        let r67 = _mm_loadu_si128(input.add(48) as *const __m128i);
        let p01 = _mm_unpacklo_epi8(r01, _mm_srli_si128::<8>(r01));
        let p23 = _mm_unpacklo_epi8(r23, _mm_srli_si128::<8>(r23));
        let p45 = _mm_unpacklo_epi8(r45, _mm_srli_si128::<8>(r45));
        let p67 = _mm_unpacklo_epi8(r67, _mm_srli_si128::<8>(r67));
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
        _mm256_storeu_si256(output as *mut __m256i, y0);
        _mm256_storeu_si256(output.add(32) as *mut __m256i, y1);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v0 = vld1q_u8(input);
        let v1 = vld1q_u8(input.add(16));
        let v2 = vld1q_u8(input.add(32));
        let v3 = vld1q_u8(input.add(48));
        let table = uint8x16x4_t(v0, v1, v2, v3);
        const IDX0: [u8; 16] = [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57];
        const IDX1: [u8; 16] = [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59];
        const IDX2: [u8; 16] = [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61];
        const IDX3: [u8; 16] = [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63];
        let mut y0 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX0.as_ptr())));
        let mut y1 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX1.as_ptr())));
        let mut y2 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX2.as_ptr())));
        let mut y3 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX3.as_ptr())));
        let m1 = vdupq_n_u64(0x00AA00AA00AA00AAu64);
        let m2 = vdupq_n_u64(0x0000CCCC0000CCCCu64);
        let m3 = vdupq_n_u64(0x00000000F0F0F0F0u64);
        macro_rules! swar2 {
            ($y:ident) => {{
                let t1 = vandq_u64(veorq_u64($y, vshrq_n_u64::<7>($y)), m1);
                $y = veorq_u64($y, veorq_u64(t1, vshlq_n_u64::<7>(t1)));
                let t2 = vandq_u64(veorq_u64($y, vshrq_n_u64::<14>($y)), m2);
                $y = veorq_u64($y, veorq_u64(t2, vshlq_n_u64::<14>(t2)));
                let t3 = vandq_u64(veorq_u64($y, vshrq_n_u64::<28>($y)), m3);
                $y = veorq_u64($y, veorq_u64(t3, vshlq_n_u64::<28>(t3)));
            }};
        }
        swar2!(y0);
        swar2!(y1);
        swar2!(y2);
        swar2!(y3);
        vst1q_u8(output, vreinterpretq_u8_u64(y0));
        vst1q_u8(output.add(16), vreinterpretq_u8_u64(y1));
        vst1q_u8(output.add(32), vreinterpretq_u8_u64(y2));
        vst1q_u8(output.add(48), vreinterpretq_u8_u64(y3));
        return;
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_lincheck_bind_both_and_eval_slice(
    cq0: *mut u64,
    cq1: *mut u64,
    cq2: *const u64,
    cq3: *const u64,
    zq0: *mut u64,
    zq1: *mut u64,
    zq2: *const u64,
    zq3: *const u64,
    len_f128: usize,
    r_lo: u64,
    r_hi: u64,
) -> (u64, u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let vr = _mm_set_epi64x(r_hi as i64, r_lo as i64);
        let (vr0, vr1) = k0_f128_pre_reduce_const_m128i(vr);
        let (mut e1_ll, mut e1_mid, mut e1_hh) = (
            _mm_setzero_si128(),
            _mm_setzero_si128(),
            _mm_setzero_si128(),
        );
        let (mut ei_ll, mut ei_mid, mut ei_hh) = (
            _mm_setzero_si128(),
            _mm_setzero_si128(),
            _mm_setzero_si128(),
        );
        for i in 0..len_f128 {
            let idx = 2 * i;
            let c0 = _mm_loadu_si128(cq0.add(idx) as *const __m128i);
            let c1 = _mm_loadu_si128(cq1.add(idx) as *const __m128i);
            let c2 = _mm_loadu_si128(cq2.add(idx) as *const __m128i);
            let c3 = _mm_loadu_si128(cq3.add(idx) as *const __m128i);
            let (fc0, fc1) = k0_f128_mul_const_vec2_m128i(
                _mm_xor_si128(c0, c2),
                _mm_xor_si128(c1, c3),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let lo = _mm_xor_si128(c0, fc0);
            let hi = _mm_xor_si128(c1, fc1);
            let z0 = _mm_loadu_si128(zq0.add(idx) as *const __m128i);
            let z1 = _mm_loadu_si128(zq1.add(idx) as *const __m128i);
            let z2 = _mm_loadu_si128(zq2.add(idx) as *const __m128i);
            let z3 = _mm_loadu_si128(zq3.add(idx) as *const __m128i);
            let (fz0, fz1) = k0_f128_mul_const_vec2_m128i(
                _mm_xor_si128(z0, z2),
                _mm_xor_si128(z1, z3),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let zlo = _mm_xor_si128(z0, fz0);
            let zhi = _mm_xor_si128(z1, fz1);
            _mm_storeu_si128(cq0.add(idx) as *mut __m128i, lo);
            _mm_storeu_si128(cq1.add(idx) as *mut __m128i, hi);
            _mm_storeu_si128(zq0.add(idx) as *mut __m128i, zlo);
            _mm_storeu_si128(zq1.add(idx) as *mut __m128i, zhi);
            k0_f128_mul_acc_m128i(hi, zhi, &mut e1_ll, &mut e1_mid, &mut e1_hh);
            k0_f128_mul_acc_m128i(
                _mm_xor_si128(hi, lo),
                _mm_xor_si128(zhi, zlo),
                &mut ei_ll,
                &mut ei_mid,
                &mut ei_hh,
            );
        }
        let e1 = k0_f128_reduce_3lane_m128i(e1_ll, e1_mid, e1_hh);
        let ei = k0_f128_reduce_3lane_m128i(ei_ll, ei_mid, ei_hh);
        return (
            _mm_extract_epi64::<0>(e1) as u64,
            _mm_extract_epi64::<1>(e1) as u64,
            _mm_extract_epi64::<0>(ei) as u64,
            _mm_extract_epi64::<1>(ei) as u64,
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let r_arr = [r_lo, r_hi];
        let vr = vld1q_u64(r_arr.as_ptr());
        let (vr0, vr1) = k0_f128_pre_reduce_const_neon(vr);
        let z = vdupq_n_u64(0);
        let (mut e1_ll, mut e1_mid, mut e1_hh) = (z, z, z);
        let (mut ei_ll, mut ei_mid, mut ei_hh) = (z, z, z);
        for i in 0..len_f128 {
            let idx = 2 * i;
            let c0 = vld1q_u64(cq0.add(idx));
            let c1 = vld1q_u64(cq1.add(idx));
            let c2 = vld1q_u64(cq2.add(idx));
            let c3 = vld1q_u64(cq3.add(idx));
            let (fc0, fc1) = k0_f128_mul_const_vec2_neon(
                veorq_u64(c0, c2),
                veorq_u64(c1, c3),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let lo = veorq_u64(c0, fc0);
            let hi = veorq_u64(c1, fc1);
            let z0 = vld1q_u64(zq0.add(idx));
            let z1 = vld1q_u64(zq1.add(idx));
            let z2 = vld1q_u64(zq2.add(idx));
            let z3 = vld1q_u64(zq3.add(idx));
            let (fz0, fz1) = k0_f128_mul_const_vec2_neon(
                veorq_u64(z0, z2),
                veorq_u64(z1, z3),
                vr0,
                vr1,
                vr0,
                vr1,
            );
            let zlo = veorq_u64(z0, fz0);
            let zhi = veorq_u64(z1, fz1);
            vst1q_u64(cq0.add(idx), lo);
            vst1q_u64(cq1.add(idx), hi);
            vst1q_u64(zq0.add(idx), zlo);
            vst1q_u64(zq1.add(idx), zhi);
            k0_f128_mul_acc_neon(hi, zhi, &mut e1_ll, &mut e1_mid, &mut e1_hh);
            k0_f128_mul_acc_neon(
                veorq_u64(hi, lo),
                veorq_u64(zhi, zlo),
                &mut ei_ll,
                &mut ei_mid,
                &mut ei_hh,
            );
        }
        let e1 = k0_f128_reduce_3lane_neon(e1_ll, e1_mid, e1_hh);
        let ei = k0_f128_reduce_3lane_neon(ei_ll, ei_mid, ei_hh);
        return (
            vgetq_lane_u64::<0>(e1),
            vgetq_lane_u64::<1>(e1),
            vgetq_lane_u64::<0>(ei),
            vgetq_lane_u64::<1>(ei),
        );
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let (mut e1_r0, mut e1_r1, mut e1_r2, mut e1_r3) = (0u64, 0u64, 0u64, 0u64);
        let (mut ei_r0, mut ei_r1, mut ei_r2, mut ei_r3) = (0u64, 0u64, 0u64, 0u64);
        for i in 0..len_f128 {
            let idx = 2 * i;
            let c0_lo = *cq0.add(idx);
            let c0_hi = *cq0.add(idx + 1);
            let c1_lo = *cq1.add(idx);
            let c1_hi = *cq1.add(idx + 1);
            let c2_lo = *cq2.add(idx);
            let c2_hi = *cq2.add(idx + 1);
            let c3_lo = *cq3.add(idx);
            let c3_hi = *cq3.add(idx + 1);
            let z0_lo = *zq0.add(idx);
            let z0_hi = *zq0.add(idx + 1);
            let z1_lo = *zq1.add(idx);
            let z1_hi = *zq1.add(idx + 1);
            let z2_lo = *zq2.add(idx);
            let z2_hi = *zq2.add(idx + 1);
            let z3_lo = *zq3.add(idx);
            let z3_hi = *zq3.add(idx + 1);
            let (fc0_lo, fc0_hi, fc1_lo, fc1_hi) = k0_f128_mul_vec2_raw(
                c0_lo ^ c2_lo,
                c0_hi ^ c2_hi,
                c1_lo ^ c3_lo,
                c1_hi ^ c3_hi,
                r_lo,
                r_hi,
                r_lo,
                r_hi,
            );
            let lo_lo = c0_lo ^ fc0_lo;
            let lo_hi = c0_hi ^ fc0_hi;
            let hi_lo = c1_lo ^ fc1_lo;
            let hi_hi = c1_hi ^ fc1_hi;
            let (fz0_lo, fz0_hi, fz1_lo, fz1_hi) = k0_f128_mul_vec2_raw(
                z0_lo ^ z2_lo,
                z0_hi ^ z2_hi,
                z1_lo ^ z3_lo,
                z1_hi ^ z3_hi,
                r_lo,
                r_hi,
                r_lo,
                r_hi,
            );
            let zlo_lo = z0_lo ^ fz0_lo;
            let zlo_hi = z0_hi ^ fz0_hi;
            let zhi_lo = z1_lo ^ fz1_lo;
            let zhi_hi = z1_hi ^ fz1_hi;
            *cq0.add(idx) = lo_lo;
            *cq0.add(idx + 1) = lo_hi;
            *cq1.add(idx) = hi_lo;
            *cq1.add(idx + 1) = hi_hi;
            *zq0.add(idx) = zlo_lo;
            *zq0.add(idx + 1) = zlo_hi;
            *zq1.add(idx) = zhi_lo;
            *zq1.add(idx + 1) = zhi_hi;
            let (m1_0, m1_1, m1_2, m1_3) = k0_f128_mul_unreduced_raw(hi_lo, hi_hi, zhi_lo, zhi_hi);
            let (mi_0, mi_1, mi_2, mi_3) = k0_f128_mul_unreduced_raw(
                hi_lo ^ lo_lo,
                hi_hi ^ lo_hi,
                zhi_lo ^ zlo_lo,
                zhi_hi ^ zlo_hi,
            );
            e1_r0 ^= m1_0;
            e1_r1 ^= m1_1;
            e1_r2 ^= m1_2;
            e1_r3 ^= m1_3;
            ei_r0 ^= mi_0;
            ei_r1 ^= mi_1;
            ei_r2 ^= mi_2;
            ei_r3 ^= mi_3;
        }
        let (e1_lo, e1_hi) = k0_f128_reduce_256_raw(e1_r0, e1_r1, e1_r2, e1_r3);
        let (ei_lo, ei_hi) = k0_f128_reduce_256_raw(ei_r0, ei_r1, ei_r2, ei_r3);
        (e1_lo, e1_hi, ei_lo, ei_hi)
    }
}
pub mod c_abi {
    unsafe extern "C" {
        pub fn gf2_kern_mul_by_x_lo(lo: u64, hi: u64) -> u64;
        pub fn gf2_kern_mul_by_x_hi(lo: u64, hi: u64) -> u64;
        pub fn gf2_kern_ghash_reduce_lo(r0: u64, r2: u64, r3: u64) -> u64;
        pub fn gf2_kern_ghash_reduce_hi(r1: u64, r2: u64, r3: u64) -> u64;
        pub fn gf2_kern_clmul64_lo(a: u64, b: u64) -> u64;
        pub fn gf2_kern_clmul64_hi(a: u64, b: u64) -> u64;
        pub fn gf2_kern_ghash_mul_lo(aLo: u64, aHi: u64, bLo: u64, bHi: u64) -> u64;
        pub fn gf2_kern_ghash_mul_hi(aLo: u64, aHi: u64, bLo: u64, bHi: u64) -> u64;
        pub fn gf2_kern_ntt_btf_u_lo(uLo: u64, prodLo: u64) -> u64;
        pub fn gf2_kern_ntt_btf_v_lo(uLo: u64, vLo: u64, prodLo: u64) -> u64;
        pub fn gf2_kern_blake3_add_sum(x: u64, y: u64) -> u64;
        pub fn gf2_kern_blake3_add_left(x: u64, y: u64) -> u64;
        pub fn gf2_kern_blake3_add_right(x: u64, y: u64) -> u64;
        pub fn gf2_kern_blake3_add_carry_aux(x: u64, y: u64) -> u64;
        pub fn gf2_kern_or_u32_lo(val: u64, shift: u64) -> u64;
        pub fn gf2_kern_or_u32_hi(val: u64, shift: u64) -> u64;
        pub fn gf2_kern_transpose_8x8_u64(y: u64) -> u64;
        pub fn gf2_kern_gf8_reduce(p: u64) -> u64;
        pub fn gf2_kern_gf8_reduce_x4(p4: u64) -> u64;
        pub fn gf2_kern_gf8_mul(a: u64, b: u64) -> u64;
        pub fn gf2_kern_gf8_unreduced_x4(a4: u64, b4: u64) -> u64;
        pub fn gf2_kern_gf8_mul_x4(a4: u64, b4: u64) -> u64;
        pub fn gf2_kern_blake3_add_sum_carry(x: u64, y: u64) -> u64;
        pub fn gf2_kern_blake3_add_lr(x: u64, y: u64) -> u64;
        pub fn gf2_kern_gf8_unreduced_even_x2(a4: u64, b4: u64) -> u64;
        pub fn gf2_kern_gf8_unreduced_odd_x2(a4: u64, b4: u64) -> u64;
        pub fn gf2_kern_gf8_reduce_deg21_pair_x4(accEven: u64, accOdd: u64) -> u64;
        pub fn gf2_kern_blake3_xor_rotr32(x: u64, y: u64, n: u64) -> u64;
        pub fn gf2_kern_blake3_finalize_word_pair(sw: u64, sw8: u64, cvw: u64) -> u64;
        pub fn gf2_kern_gf8_to_composite(x: u64) -> u64;
        pub fn gf2_kern_gf8_from_composite(c: u64) -> u64;
        pub fn gf2_kern_gf8_mul_composite(ca: u64, cb: u64) -> u64;
        pub fn gf2_kern_gf16_log_mod15_signed_reduce(s: u64) -> u64;
        pub fn gf2_kern_f128_pre_reduce_const_lo(r1: u64) -> u64;
        pub fn gf2_kern_f128_pre_reduce_const_hi(r0: u64, r1: u64) -> u64;
        pub fn gf2_kern_f128_reduce_deg191_lo(lLo: u64, hHi: u64) -> u64;
        pub fn gf2_kern_f128_reduce_deg191_hi(lHi: u64, hLo: u64, hHi: u64) -> u64;
        pub fn gf2_kern_f128_karatsuba_unred_r1(llLo: u64, llHi: u64, hhLo: u64, pmLo: u64) -> u64;
        pub fn gf2_kern_f128_karatsuba_unred_r2(llHi: u64, hhLo: u64, hhHi: u64, pmHi: u64) -> u64;
        pub fn gf2_kern_urm_pack_c_pair_bank0(x0: u64, x1: u64) -> u64;
        pub fn gf2_kern_urm_pack_c_pair_bank1(x0: u64, x1: u64) -> u64;
        pub fn gf2_kern_f128_fma_lo(
            alphaLo: u64,
            alphaHi: u64,
            aLo: u64,
            aHi: u64,
            bLo: u64,
        ) -> u64;
        pub fn gf2_kern_f128_fma_hi(
            alphaLo: u64,
            alphaHi: u64,
            aLo: u64,
            aHi: u64,
            bHi: u64,
        ) -> u64;
        pub fn gf2_kern_ghash_mul_unred_r1(aLo: u64, aHi: u64, bLo: u64, bHi: u64) -> u64;
        pub fn gf2_kern_ghash_mul_unred_r2(aLo: u64, aHi: u64, bLo: u64, bHi: u64) -> u64;
        pub fn gf2_kern_mle_fold_lo(
            v0Lo: u64,
            v0Hi: u64,
            v1Lo: u64,
            v1Hi: u64,
            rLo: u64,
            rHi: u64,
        ) -> u64;
        pub fn gf2_kern_mle_fold_hi(
            v0Lo: u64,
            v0Hi: u64,
            v1Lo: u64,
            v1Hi: u64,
            rLo: u64,
            rHi: u64,
        ) -> u64;
        pub fn gf2_kern_pack8_bool_bytes(x: u64) -> u64;
        pub fn gf2_kern_unpack8_bool_bytes(b: u64) -> u64;
        pub fn gf2_kern_blake3_chunk_flags(blockIdx: u64, numBlocks: u64) -> u64;
        pub fn gf2_kern_gf8_mul_by_x(a: u64) -> u64;
        pub fn gf2_kern_gf8_square(a: u64) -> u64;
        pub fn gf2_kern_gf8_ntt_next_s(s: u64, sAtRoot: u64) -> u64;
        pub fn gf2_kern_bitrecord_flush_word(w: u64, spill: u64, s: u64) -> u64;
        pub fn gf2_kern_bitrecord_flush_spill(w: u64, s: u64) -> u64;
        pub fn gf2_kern_delta_swap_lo(ak: u64, akj: u64, j: u64, m: u64) -> u64;
        pub fn gf2_kern_delta_swap_hi(ak: u64, akj: u64, j: u64, m: u64) -> u64;
        pub fn gf2_kern_ntt_replicated_msg_idx(dstIdx: u64, logMsgLen: u64) -> u64;
        pub fn gf2_kern_inv_ntt_cyclic_offset_lo(b: u64) -> u64;
        pub fn gf2_kern_inv_ntt_cyclic_offset_hi(b: u64) -> u64;
        pub fn gf2_kern_ntt_dual_block_src_offset(
            sixteenth: u64,
            numNtts: u64,
            r: u64,
            i: u64,
            lane: u64,
        ) -> u64;
        pub fn gf2_kern_ntt_dual_block_dst_offset(
            sixteenth: u64,
            numNtts: u64,
            block: u64,
            r: u64,
            i: u64,
            lane: u64,
        ) -> u64;
        pub fn gf2_kern_sha256_ch_and_out(e: u64, f: u64, g: u64) -> u64;
        pub fn gf2_kern_sha256_maj_and_out(a: u64, b: u64, c: u64) -> u64;
        pub fn gf2_kern_sha256_big_sigma0(x: u64) -> u64;
        pub fn gf2_kern_sha256_big_sigma1(x: u64) -> u64;
        pub fn gf2_kern_sha256_small_sigma0(x: u64) -> u64;
        pub fn gf2_kern_sha256_small_sigma1(x: u64) -> u64;
        pub fn gf2_kern_keccak_rol64(x: u64, r: u64) -> u64;
        pub fn gf2_kern_keccak_theta_d(cm1: u64, cp1: u64) -> u64;
        pub fn gf2_kern_keccak_chi_and(b1: u64, b2: u64) -> u64;
        pub fn gf2_kern_keccak_chi(b0: u64, b1: u64, b2: u64) -> u64;
        pub fn gf2_kern_pack_even_bytes_u32(r0: u64) -> u64;
        pub fn gf2_kern_pack_even_bytes_x8(r0: u64, r1: u64) -> u64;
        pub fn gf2_kern_unpack_even_bytes_u32(x: u64) -> u64;
        pub fn gf2_kern_split_even_odd_bits_u8(c: u64) -> u64;
        pub fn gf2_kern_gf8_fft_btf_pair(u: u64, w: u64, lambda: u64) -> u64;
        pub fn gf2_kern_gf8_ifft_btf_pair(uPrime: u64, wPrime: u64, lambda: u64) -> u64;
        pub fn gf2_kern_isolate_lowest_one(w: u64) -> u64;
        pub fn gf2_kern_clear_lowest_one(w: u64) -> u64;
        pub fn gf2_kern_bit_deposit_lowest_one(c: u64, m: u64) -> u64;
        pub fn gf2_kern_gf8_subspace_w2(s: u64) -> u64;
        pub fn gf2_kern_nibble_lo(b: u64) -> u64;
        pub fn gf2_kern_nibble_hi(b: u64) -> u64;
        pub fn gf2_kern_jagged_transition(
            row: u64,
            index: u64,
            curr: u64,
            next: u64,
            state: u64,
        ) -> u64;
        pub fn gf2_kern_slot_indicator_lo(
            targetSlot: u64,
            ssLo: u64,
            ssHi: u64,
            sdLo: u64,
            sdHi: u64,
        ) -> u64;
        pub fn gf2_kern_slot_indicator_hi(
            targetSlot: u64,
            ssLo: u64,
            ssHi: u64,
            sdLo: u64,
            sdHi: u64,
        ) -> u64;
        pub fn gf2_kern_eq_bit_mul_lo(bit: u64, vLo: u64, vHi: u64, rLo: u64, rHi: u64) -> u64;
        pub fn gf2_kern_eq_bit_mul_hi(bit: u64, vLo: u64, vHi: u64, rLo: u64, rHi: u64) -> u64;
        pub fn gf2_kern_monomial_basis128_lo(i: u64) -> u64;
        pub fn gf2_kern_monomial_basis128_hi(i: u64) -> u64;
        pub fn gf2_kern_ntt_radix4_comb_a(a: u64, pOutA: u64, pInA: u64) -> u64;
        pub fn gf2_kern_ntt_radix4_comb_b(a: u64, b: u64, pOutA: u64, pOutB: u64, pInA: u64)
        -> u64;
        pub fn gf2_kern_ntt_radix4_comb_c(a: u64, c: u64, pOutA: u64, pInB: u64) -> u64;
        pub fn gf2_kern_ntt_radix4_comb_d(
            a: u64,
            b: u64,
            c: u64,
            d: u64,
            pOutA: u64,
            pOutB: u64,
            pInB: u64,
        ) -> u64;
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_mul_by_x_lo(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 63u64;
        let v3: u64 = (v1.wrapping_shr((63u64 & 63) as u32));
        let v4: u64 = 0u64;
        let v5: u64 = (0u64.wrapping_sub(v3));
        let v6: u64 = 1u64;
        let v7: u64 = (v0.wrapping_shl((1u64 & 63) as u32));
        let v8: u64 = 135u64;
        let v9: u64 = (135u64 & v5);
        let v10: u64 = (v7 ^ v9);
        v10
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_mul_by_x_hi(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 1u64;
        let v3: u64 = (v1.wrapping_shl((1u64 & 63) as u32));
        let v4: u64 = 63u64;
        let v5: u64 = (v0.wrapping_shr((63u64 & 63) as u32));
        let v6: u64 = (v3 | v5);
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ghash_reduce_lo(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 135u64;
        let v4: u64 = k0_clmul64_hi(v2, 135u64);
        let v5: u64 = (v1 ^ v4);
        let v6: u64 = k0_clmul64_lo(v5, 135u64);
        let v7: u64 = (v0 ^ v6);
        v7
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ghash_reduce_hi(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 135u64;
        let v4: u64 = k0_clmul64_hi(v2, 135u64);
        let v5: u64 = (v1 ^ v4);
        let v6: u64 = k0_clmul64_lo(v2, 135u64);
        let v7: u64 = (v0 ^ v6);
        let v8: u64 = k0_clmul64_hi(v5, 135u64);
        let v9: u64 = (v7 ^ v8);
        v9
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_clmul64_lo(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = k0_clmul64_lo(v0, v1);
        v2
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_clmul64_hi(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = k0_clmul64_hi(v0, v1);
        v2
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ghash_mul_lo(v0: u64, v1: u64, v2: u64, v3: u64) -> u64 {
    unsafe {
        let v4: u64 = k0_clmul64_lo(v0, v2);
        let v5: u64 = k0_clmul64_hi(v0, v3);
        let v6: u64 = k0_clmul64_hi(v1, v2);
        let v7: u64 = k0_clmul64_lo(v1, v3);
        let v8: u64 = k0_clmul64_hi(v1, v3);
        let v9: u64 = 135u64;
        let v10: u64 = k0_clmul64_hi(v8, 135u64);
        let v11: u64 = (v5 ^ v6);
        let v12: u64 = (v11 ^ v10);
        let v13: u64 = (v7 ^ v12);
        let v14: u64 = k0_clmul64_lo(v13, 135u64);
        let v15: u64 = (v4 ^ v14);
        v15
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ghash_mul_hi(v0: u64, v1: u64, v2: u64, v3: u64) -> u64 {
    unsafe {
        let v4: u64 = k0_clmul64_hi(v0, v2);
        let v5: u64 = k0_clmul64_lo(v0, v3);
        let v6: u64 = k0_clmul64_hi(v0, v3);
        let v7: u64 = k0_clmul64_lo(v1, v2);
        let v8: u64 = k0_clmul64_hi(v1, v2);
        let v9: u64 = k0_clmul64_lo(v1, v3);
        let v10: u64 = k0_clmul64_hi(v1, v3);
        let v11: u64 = 135u64;
        let v12: u64 = k0_clmul64_lo(v10, 135u64);
        let v13: u64 = k0_clmul64_hi(v10, 135u64);
        let v14: u64 = (v6 ^ v8);
        let v15: u64 = (v14 ^ v13);
        let v16: u64 = (v9 ^ v15);
        let v17: u64 = k0_clmul64_hi(v16, 135u64);
        let v18: u64 = (v5 ^ v7);
        let v19: u64 = (v18 ^ v12);
        let v20: u64 = (v4 ^ v17);
        let v21: u64 = (v19 ^ v20);
        v21
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ntt_btf_u_lo(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = (v0 ^ v1);
        v2
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ntt_btf_v_lo(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = (v0 ^ v2);
        let v4: u64 = (v1 ^ v3);
        v4
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_blake3_add_sum(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 4294967295u64;
        let v3: u64 = (v0 & 4294967295u64);
        let v4: u64 = (v1 & 4294967295u64);
        let v5: u64 = (v3.wrapping_add(v4));
        let v6: u64 = (v5 & 4294967295u64);
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_blake3_add_left(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 4294967295u64;
        let v3: u64 = 2147483647u64;
        let v4: u64 = (v0 & 4294967295u64);
        let v5: u64 = (v1 & 4294967295u64);
        let v6: u64 = (v4.wrapping_add(v5));
        let v7: u64 = (v6 & 4294967295u64);
        let v8: u64 = (v7 ^ v4);
        let v9: u64 = (v8 ^ v5);
        let v10: u64 = (v4 ^ v9);
        let v11: u64 = (v10 & 2147483647u64);
        v11
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_blake3_add_right(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 4294967295u64;
        let v3: u64 = 2147483647u64;
        let v4: u64 = (v0 & 4294967295u64);
        let v5: u64 = (v1 & 4294967295u64);
        let v6: u64 = (v4.wrapping_add(v5));
        let v7: u64 = (v6 & 4294967295u64);
        let v8: u64 = (v7 ^ v4);
        let v9: u64 = (v8 ^ v5);
        let v10: u64 = (v5 ^ v9);
        let v11: u64 = (v10 & 2147483647u64);
        v11
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_blake3_add_carry_aux(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 4294967295u64;
        let v3: u64 = 2147483647u64;
        let v4: u64 = (v0 & 4294967295u64);
        let v5: u64 = (v1 & 4294967295u64);
        let v6: u64 = (v4.wrapping_add(v5));
        let v7: u64 = (v6 & 4294967295u64);
        let v8: u64 = (v7 ^ v4);
        let v9: u64 = (v8 ^ v5);
        let v10: u64 = (v4 ^ v9);
        let v11: u64 = (v10 & 2147483647u64);
        let v12: u64 = (v5 ^ v9);
        let v13: u64 = (v12 & 2147483647u64);
        let v14: u64 = (v11 & v13);
        v14
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_or_u32_lo(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 4294967295u64;
        let v3: u64 = (v0 & 4294967295u64);
        let v4: u64 = 63u64;
        let v5: u64 = (v1 & 63u64);
        let v6: u64 = (v3.wrapping_shl((v5 & 63) as u32));
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_or_u32_hi(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 63u64;
        let v3: u64 = (v1 & 63u64);
        let v4: u64 = 32u64;
        let v5: u64 = (if 32u64 < v3 { 1u64 } else { 0u64 });
        let v6: u64 = (if v5 == 0u64 { 1u64 } else { 0u64 });
        if v6 != 0 {
            let v7: u64 = 0u64;
            v7
        } else {
            let v8: u64 = 4294967295u64;
            let v9: u64 = (v0 & 4294967295u64);
            let v10: u64 = 64u64;
            let v11: u64 = (64u64.wrapping_sub(v3));
            let v12: u64 = (v9.wrapping_shr((v11 & 63) as u32));
            v12
        }
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_transpose_8x8_u64(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 47851476196393130u64;
        let v2: u64 = 225176545447116u64;
        let v3: u64 = 4042322160u64;
        let v4: u64 = 7u64;
        let v5: u64 = (v0.wrapping_shr((7u64 & 63) as u32));
        let v6: u64 = (v0 ^ v5);
        let v7: u64 = (v6 & 47851476196393130u64);
        let v8: u64 = (v0 ^ v7);
        let v9: u64 = (v7.wrapping_shl((7u64 & 63) as u32));
        let v10: u64 = (v8 ^ v9);
        let v11: u64 = 14u64;
        let v12: u64 = (v10.wrapping_shr((14u64 & 63) as u32));
        let v13: u64 = (v10 ^ v12);
        let v14: u64 = (v13 & 225176545447116u64);
        let v15: u64 = (v10 ^ v14);
        let v16: u64 = (v14.wrapping_shl((14u64 & 63) as u32));
        let v17: u64 = (v15 ^ v16);
        let v18: u64 = 28u64;
        let v19: u64 = (v17.wrapping_shr((28u64 & 63) as u32));
        let v20: u64 = (v17 ^ v19);
        let v21: u64 = (v20 & 4042322160u64);
        let v22: u64 = (v17 ^ v21);
        let v23: u64 = (v21.wrapping_shl((28u64 & 63) as u32));
        let v24: u64 = (v22 ^ v23);
        v24
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_reduce(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 65535u64;
        let v2: u64 = (v0 & 65535u64);
        let v3: u64 = 8u64;
        let v4: u64 = (v2.wrapping_shr((8u64 & 63) as u32));
        let v5: u64 = 255u64;
        let v6: u64 = (v2 & 255u64);
        let v7: u64 = (v6 ^ v4);
        let v8: u64 = 1u64;
        let v9: u64 = (v4.wrapping_shl((1u64 & 63) as u32));
        let v10: u64 = (v7 ^ v9);
        let v11: u64 = 3u64;
        let v12: u64 = (v4.wrapping_shl((3u64 & 63) as u32));
        let v13: u64 = (v10 ^ v12);
        let v14: u64 = 4u64;
        let v15: u64 = (v4.wrapping_shl((4u64 & 63) as u32));
        let v16: u64 = (v13 ^ v15);
        let v17: u64 = (v16.wrapping_shr((8u64 & 63) as u32));
        let v18: u64 = (v16 & 255u64);
        let v19: u64 = (v18 ^ v17);
        let v20: u64 = (v17.wrapping_shl((1u64 & 63) as u32));
        let v21: u64 = (v19 ^ v20);
        let v22: u64 = (v17.wrapping_shl((3u64 & 63) as u32));
        let v23: u64 = (v21 ^ v22);
        let v24: u64 = (v17.wrapping_shl((4u64 & 63) as u32));
        let v25: u64 = (v23 ^ v24);
        let v26: u64 = (v25 & 255u64);
        v26
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_reduce_x4(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 71777214294589695u64;
        let v2: u64 = 8u64;
        let v3: u64 = (v0.wrapping_shr((8u64 & 63) as u32));
        let v4: u64 = (v3 & 71777214294589695u64);
        let v5: u64 = (v0 & 71777214294589695u64);
        let v6: u64 = (v5 ^ v4);
        let v7: u64 = 1u64;
        let v8: u64 = (v4.wrapping_shl((1u64 & 63) as u32));
        let v9: u64 = (v6 ^ v8);
        let v10: u64 = 3u64;
        let v11: u64 = (v4.wrapping_shl((3u64 & 63) as u32));
        let v12: u64 = (v9 ^ v11);
        let v13: u64 = 4u64;
        let v14: u64 = (v4.wrapping_shl((4u64 & 63) as u32));
        let v15: u64 = (v12 ^ v14);
        let v16: u64 = (v15.wrapping_shr((8u64 & 63) as u32));
        let v17: u64 = (v16 & 71777214294589695u64);
        let v18: u64 = (v15 & 71777214294589695u64);
        let v19: u64 = (v18 ^ v17);
        let v20: u64 = (v17.wrapping_shl((1u64 & 63) as u32));
        let v21: u64 = (v19 ^ v20);
        let v22: u64 = (v17.wrapping_shl((3u64 & 63) as u32));
        let v23: u64 = (v21 ^ v22);
        let v24: u64 = (v17.wrapping_shl((4u64 & 63) as u32));
        let v25: u64 = (v23 ^ v24);
        let v26: u64 = (v25 & 71777214294589695u64);
        v26
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_mul(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 255u64;
        let v3: u64 = (v0 & 255u64);
        let v4: u64 = (v1 & 255u64);
        let v5: u64 = k0_clmul64_lo(v3, v4);
        let v6: u64 = 65535u64;
        let v7: u64 = (v5 & 65535u64);
        let v8: u64 = 8u64;
        let v9: u64 = (v7.wrapping_shr((8u64 & 63) as u32));
        let v10: u64 = (v7 & 255u64);
        let v11: u64 = (v10 ^ v9);
        let v12: u64 = 1u64;
        let v13: u64 = (v9.wrapping_shl((1u64 & 63) as u32));
        let v14: u64 = (v11 ^ v13);
        let v15: u64 = 3u64;
        let v16: u64 = (v9.wrapping_shl((3u64 & 63) as u32));
        let v17: u64 = (v14 ^ v16);
        let v18: u64 = 4u64;
        let v19: u64 = (v9.wrapping_shl((4u64 & 63) as u32));
        let v20: u64 = (v17 ^ v19);
        let v21: u64 = (v20.wrapping_shr((8u64 & 63) as u32));
        let v22: u64 = (v20 & 255u64);
        let v23: u64 = (v22 ^ v21);
        let v24: u64 = (v21.wrapping_shl((1u64 & 63) as u32));
        let v25: u64 = (v23 ^ v24);
        let v26: u64 = (v21.wrapping_shl((3u64 & 63) as u32));
        let v27: u64 = (v25 ^ v26);
        let v28: u64 = (v21.wrapping_shl((4u64 & 63) as u32));
        let v29: u64 = (v27 ^ v28);
        let v30: u64 = (v29 & 255u64);
        v30
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_unreduced_x4(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 16711935u64;
        let v3: u64 = (v0 & 16711935u64);
        let v4: u64 = (v1 & 16711935u64);
        let v5: u64 = 8u64;
        let v6: u64 = (v0.wrapping_shr((8u64 & 63) as u32));
        let v7: u64 = (v6 & 16711935u64);
        let v8: u64 = (v1.wrapping_shr((8u64 & 63) as u32));
        let v9: u64 = (v8 & 16711935u64);
        let v10: u64 = k0_clmul64_lo(v3, v4);
        let v11: u64 = 281470681808895u64;
        let v12: u64 = (v10 & 281470681808895u64);
        let v13: u64 = k0_clmul64_lo(v7, v9);
        let v14: u64 = (v13 & 281470681808895u64);
        let v15: u64 = 16u64;
        let v16: u64 = (v14.wrapping_shl((16u64 & 63) as u32));
        let v17: u64 = (v12 | v16);
        v17
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_mul_x4(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 16711935u64;
        let v3: u64 = (v0 & 16711935u64);
        let v4: u64 = (v1 & 16711935u64);
        let v5: u64 = 8u64;
        let v6: u64 = (v0.wrapping_shr((8u64 & 63) as u32));
        let v7: u64 = (v6 & 16711935u64);
        let v8: u64 = (v1.wrapping_shr((8u64 & 63) as u32));
        let v9: u64 = (v8 & 16711935u64);
        let v10: u64 = k0_clmul64_lo(v3, v4);
        let v11: u64 = 281470681808895u64;
        let v12: u64 = (v10 & 281470681808895u64);
        let v13: u64 = k0_clmul64_lo(v7, v9);
        let v14: u64 = (v13 & 281470681808895u64);
        let v15: u64 = 16u64;
        let v16: u64 = (v14.wrapping_shl((16u64 & 63) as u32));
        let v17: u64 = (v12 | v16);
        let v18: u64 = 71777214294589695u64;
        let v19: u64 = (v17.wrapping_shr((8u64 & 63) as u32));
        let v20: u64 = (v19 & 71777214294589695u64);
        let v21: u64 = (v17 & 71777214294589695u64);
        let v22: u64 = (v21 ^ v20);
        let v23: u64 = 1u64;
        let v24: u64 = (v20.wrapping_shl((1u64 & 63) as u32));
        let v25: u64 = (v22 ^ v24);
        let v26: u64 = 3u64;
        let v27: u64 = (v20.wrapping_shl((3u64 & 63) as u32));
        let v28: u64 = (v25 ^ v27);
        let v29: u64 = 4u64;
        let v30: u64 = (v20.wrapping_shl((4u64 & 63) as u32));
        let v31: u64 = (v28 ^ v30);
        let v32: u64 = (v31.wrapping_shr((8u64 & 63) as u32));
        let v33: u64 = (v32 & 71777214294589695u64);
        let v34: u64 = (v31 & 71777214294589695u64);
        let v35: u64 = (v34 ^ v33);
        let v36: u64 = (v33.wrapping_shl((1u64 & 63) as u32));
        let v37: u64 = (v35 ^ v36);
        let v38: u64 = (v33.wrapping_shl((3u64 & 63) as u32));
        let v39: u64 = (v37 ^ v38);
        let v40: u64 = (v33.wrapping_shl((4u64 & 63) as u32));
        let v41: u64 = (v39 ^ v40);
        let v42: u64 = (v41 & 71777214294589695u64);
        v42
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_blake3_add_sum_carry(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 4294967295u64;
        let v3: u64 = 2147483647u64;
        let v4: u64 = (v0 & 4294967295u64);
        let v5: u64 = (v1 & 4294967295u64);
        let v6: u64 = (v4.wrapping_add(v5));
        let v7: u64 = (v6 & 4294967295u64);
        let v8: u64 = (v7 ^ v4);
        let v9: u64 = (v8 ^ v5);
        let v10: u64 = (v4 ^ v9);
        let v11: u64 = (v10 & 2147483647u64);
        let v12: u64 = (v5 ^ v9);
        let v13: u64 = (v12 & 2147483647u64);
        let v14: u64 = (v11 & v13);
        let v15: u64 = 32u64;
        let v16: u64 = (v14.wrapping_shl((32u64 & 63) as u32));
        let v17: u64 = (v7 | v16);
        v17
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_blake3_add_lr(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 4294967295u64;
        let v3: u64 = 2147483647u64;
        let v4: u64 = (v0 & 4294967295u64);
        let v5: u64 = (v1 & 4294967295u64);
        let v6: u64 = (v4.wrapping_add(v5));
        let v7: u64 = (v6 & 4294967295u64);
        let v8: u64 = (v7 ^ v4);
        let v9: u64 = (v8 ^ v5);
        let v10: u64 = (v4 ^ v9);
        let v11: u64 = (v10 & 2147483647u64);
        let v12: u64 = (v5 ^ v9);
        let v13: u64 = (v12 & 2147483647u64);
        let v14: u64 = 32u64;
        let v15: u64 = (v13.wrapping_shl((32u64 & 63) as u32));
        let v16: u64 = (v11 | v15);
        v16
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_unreduced_even_x2(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 16711935u64;
        let v3: u64 = (v0 & 16711935u64);
        let v4: u64 = (v1 & 16711935u64);
        let v5: u64 = k0_clmul64_lo(v3, v4);
        let v6: u64 = 281470681808895u64;
        let v7: u64 = (v5 & 281470681808895u64);
        v7
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_unreduced_odd_x2(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 8u64;
        let v3: u64 = (v0.wrapping_shr((8u64 & 63) as u32));
        let v4: u64 = 16711935u64;
        let v5: u64 = (v3 & 16711935u64);
        let v6: u64 = (v1.wrapping_shr((8u64 & 63) as u32));
        let v7: u64 = (v6 & 16711935u64);
        let v8: u64 = k0_clmul64_lo(v5, v7);
        let v9: u64 = 281470681808895u64;
        let v10: u64 = (v8 & 281470681808895u64);
        v10
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_reduce_deg21_pair_x4(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 281470681808895u64;
        let v3: u64 = (v0 & 281470681808895u64);
        let v4: u64 = (v1 & 281470681808895u64);
        let v5: u64 = 16u64;
        let v6: u64 = (v4.wrapping_shl((16u64 & 63) as u32));
        let v7: u64 = (v3 ^ v6);
        let v8: u64 = (v0.wrapping_shr((16u64 & 63) as u32));
        let v9: u64 = 270582939711u64;
        let v10: u64 = (v8 & 270582939711u64);
        let v11: u64 = 17732923536900096u64;
        let v12: u64 = (v1 & 17732923536900096u64);
        let v13: u64 = (v10 ^ v12);
        let v14: u64 = 6u64;
        let v15: u64 = (v13.wrapping_shl((6u64 & 63) as u32));
        let v16: u64 = (v7 ^ v15);
        let v17: u64 = 4u64;
        let v18: u64 = (v13.wrapping_shl((4u64 & 63) as u32));
        let v19: u64 = (v16 ^ v18);
        let v20: u64 = 3u64;
        let v21: u64 = (v13.wrapping_shl((3u64 & 63) as u32));
        let v22: u64 = (v19 ^ v21);
        let v23: u64 = 2u64;
        let v24: u64 = (v13.wrapping_shl((2u64 & 63) as u32));
        let v25: u64 = (v22 ^ v24);
        let v26: u64 = 1u64;
        let v27: u64 = (v13.wrapping_shl((1u64 & 63) as u32));
        let v28: u64 = (v25 ^ v27);
        let v29: u64 = 71777214294589695u64;
        let v30: u64 = 8u64;
        let v31: u64 = (v28.wrapping_shr((8u64 & 63) as u32));
        let v32: u64 = (v31 & 71777214294589695u64);
        let v33: u64 = (v28 & 71777214294589695u64);
        let v34: u64 = (v33 ^ v32);
        let v35: u64 = (v32.wrapping_shl((1u64 & 63) as u32));
        let v36: u64 = (v34 ^ v35);
        let v37: u64 = (v32.wrapping_shl((3u64 & 63) as u32));
        let v38: u64 = (v36 ^ v37);
        let v39: u64 = (v32.wrapping_shl((4u64 & 63) as u32));
        let v40: u64 = (v38 ^ v39);
        let v41: u64 = (v40.wrapping_shr((8u64 & 63) as u32));
        let v42: u64 = (v41 & 71777214294589695u64);
        let v43: u64 = (v40 & 71777214294589695u64);
        let v44: u64 = (v43 ^ v42);
        let v45: u64 = (v42.wrapping_shl((1u64 & 63) as u32));
        let v46: u64 = (v44 ^ v45);
        let v47: u64 = (v42.wrapping_shl((3u64 & 63) as u32));
        let v48: u64 = (v46 ^ v47);
        let v49: u64 = (v42.wrapping_shl((4u64 & 63) as u32));
        let v50: u64 = (v48 ^ v49);
        let v51: u64 = (v50 & 71777214294589695u64);
        v51
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_blake3_xor_rotr32(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 4294967295u64;
        let v4: u64 = (v0 ^ v1);
        let v5: u64 = (v4 & 4294967295u64);
        let v6: u64 = 31u64;
        let v7: u64 = (v2 & 31u64);
        let v8: u64 = (v5.wrapping_shr((v7 & 63) as u32));
        let v9: u64 = 32u64;
        let v10: u64 = (32u64.wrapping_sub(v7));
        let v11: u64 = (v10 & 31u64);
        let v12: u64 = (v5.wrapping_shl((v11 & 63) as u32));
        let v13: u64 = (v8 | v12);
        let v14: u64 = (v13 & 4294967295u64);
        v14
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_blake3_finalize_word_pair(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 4294967295u64;
        let v4: u64 = (v0 ^ v1);
        let v5: u64 = (v4 & 4294967295u64);
        let v6: u64 = (v1 ^ v2);
        let v7: u64 = (v6 & 4294967295u64);
        let v8: u64 = 32u64;
        let v9: u64 = (v7.wrapping_shl((32u64 & 63) as u32));
        let v10: u64 = (v5 | v9);
        v10
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_to_composite(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 0u64;
        let v2: u64 = 1u64;
        let v3: u64 = (v0 & 1u64);
        let v4: u64 = (0u64.wrapping_sub(v3));
        let v5: u64 = (v0.wrapping_shr((1u64 & 63) as u32));
        let v6: u64 = (v5 & 1u64);
        let v7: u64 = (0u64.wrapping_sub(v6));
        let v8: u64 = 2u64;
        let v9: u64 = (v0.wrapping_shr((2u64 & 63) as u32));
        let v10: u64 = (v9 & 1u64);
        let v11: u64 = (0u64.wrapping_sub(v10));
        let v12: u64 = 3u64;
        let v13: u64 = (v0.wrapping_shr((3u64 & 63) as u32));
        let v14: u64 = (v13 & 1u64);
        let v15: u64 = (0u64.wrapping_sub(v14));
        let v16: u64 = 4u64;
        let v17: u64 = (v0.wrapping_shr((4u64 & 63) as u32));
        let v18: u64 = (v17 & 1u64);
        let v19: u64 = (0u64.wrapping_sub(v18));
        let v20: u64 = 5u64;
        let v21: u64 = (v0.wrapping_shr((5u64 & 63) as u32));
        let v22: u64 = (v21 & 1u64);
        let v23: u64 = (0u64.wrapping_sub(v22));
        let v24: u64 = 6u64;
        let v25: u64 = (v0.wrapping_shr((6u64 & 63) as u32));
        let v26: u64 = (v25 & 1u64);
        let v27: u64 = (0u64.wrapping_sub(v26));
        let v28: u64 = 7u64;
        let v29: u64 = (v0.wrapping_shr((7u64 & 63) as u32));
        let v30: u64 = (v29 & 1u64);
        let v31: u64 = (0u64.wrapping_sub(v30));
        let v32: u64 = (v4 & 1u64);
        let v33: u64 = 73u64;
        let v34: u64 = (v7 & 73u64);
        let v35: u64 = (v32 ^ v34);
        let v36: u64 = 53u64;
        let v37: u64 = (v11 & 53u64);
        let v38: u64 = (v35 ^ v37);
        let v39: u64 = 61u64;
        let v40: u64 = (v15 & 61u64);
        let v41: u64 = (v38 ^ v40);
        let v42: u64 = 89u64;
        let v43: u64 = (v19 & 89u64);
        let v44: u64 = (v41 ^ v43);
        let v45: u64 = 231u64;
        let v46: u64 = (v23 & 231u64);
        let v47: u64 = (v44 ^ v46);
        let v48: u64 = 85u64;
        let v49: u64 = (v27 & 85u64);
        let v50: u64 = (v47 ^ v49);
        let v51: u64 = 177u64;
        let v52: u64 = (v31 & 177u64);
        let v53: u64 = (v50 ^ v52);
        v53
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_from_composite(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 0u64;
        let v2: u64 = 1u64;
        let v3: u64 = (v0 & 1u64);
        let v4: u64 = (0u64.wrapping_sub(v3));
        let v5: u64 = (v0.wrapping_shr((1u64 & 63) as u32));
        let v6: u64 = (v5 & 1u64);
        let v7: u64 = (0u64.wrapping_sub(v6));
        let v8: u64 = 2u64;
        let v9: u64 = (v0.wrapping_shr((2u64 & 63) as u32));
        let v10: u64 = (v9 & 1u64);
        let v11: u64 = (0u64.wrapping_sub(v10));
        let v12: u64 = 3u64;
        let v13: u64 = (v0.wrapping_shr((3u64 & 63) as u32));
        let v14: u64 = (v13 & 1u64);
        let v15: u64 = (0u64.wrapping_sub(v14));
        let v16: u64 = 4u64;
        let v17: u64 = (v0.wrapping_shr((4u64 & 63) as u32));
        let v18: u64 = (v17 & 1u64);
        let v19: u64 = (0u64.wrapping_sub(v18));
        let v20: u64 = 5u64;
        let v21: u64 = (v0.wrapping_shr((5u64 & 63) as u32));
        let v22: u64 = (v21 & 1u64);
        let v23: u64 = (0u64.wrapping_sub(v22));
        let v24: u64 = 6u64;
        let v25: u64 = (v0.wrapping_shr((6u64 & 63) as u32));
        let v26: u64 = (v25 & 1u64);
        let v27: u64 = (0u64.wrapping_sub(v26));
        let v28: u64 = 7u64;
        let v29: u64 = (v0.wrapping_shr((7u64 & 63) as u32));
        let v30: u64 = (v29 & 1u64);
        let v31: u64 = (0u64.wrapping_sub(v30));
        let v32: u64 = (v4 & 1u64);
        let v33: u64 = 225u64;
        let v34: u64 = (v7 & 225u64);
        let v35: u64 = (v32 ^ v34);
        let v36: u64 = 92u64;
        let v37: u64 = (v11 & 92u64);
        let v38: u64 = (v35 ^ v37);
        let v39: u64 = 12u64;
        let v40: u64 = (v15 & 12u64);
        let v41: u64 = (v38 ^ v40);
        let v42: u64 = 18u64;
        let v43: u64 = (v19 & 18u64);
        let v44: u64 = (v41 ^ v43);
        let v45: u64 = 75u64;
        let v46: u64 = (v23 & 75u64);
        let v47: u64 = (v44 ^ v46);
        let v48: u64 = 15u64;
        let v49: u64 = (v27 & 15u64);
        let v50: u64 = (v47 ^ v49);
        let v51: u64 = 216u64;
        let v52: u64 = (v31 & 216u64);
        let v53: u64 = (v50 ^ v52);
        v53
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_mul_composite(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 15u64;
        let v3: u64 = (v0 & 15u64);
        let v4: u64 = 4u64;
        let v5: u64 = (v0.wrapping_shr((4u64 & 63) as u32));
        let v6: u64 = (v5 & 15u64);
        let v7: u64 = (v1 & 15u64);
        let v8: u64 = (v1.wrapping_shr((4u64 & 63) as u32));
        let v9: u64 = (v8 & 15u64);
        let v10: u64 = (v3 & 15u64);
        let v11: u64 = (v7 & 15u64);
        let v12: u64 = k0_clmul64_lo(v10, v11);
        let v13: u64 = 127u64;
        let v14: u64 = (v12 & 127u64);
        let v15: u64 = (v14.wrapping_shr((4u64 & 63) as u32));
        let v16: u64 = 1u64;
        let v17: u64 = (v15.wrapping_shl((1u64 & 63) as u32));
        let v18: u64 = (v14 ^ v17);
        let v19: u64 = (v18 ^ v15);
        let v20: u64 = (v19 & 15u64);
        let v21: u64 = (v6 & 15u64);
        let v22: u64 = (v9 & 15u64);
        let v23: u64 = k0_clmul64_lo(v21, v22);
        let v24: u64 = (v23 & 127u64);
        let v25: u64 = (v24.wrapping_shr((4u64 & 63) as u32));
        let v26: u64 = (v25.wrapping_shl((1u64 & 63) as u32));
        let v27: u64 = (v24 ^ v26);
        let v28: u64 = (v27 ^ v25);
        let v29: u64 = (v28 & 15u64);
        let v30: u64 = (v3 ^ v6);
        let v31: u64 = (v30 & 15u64);
        let v32: u64 = (v7 ^ v9);
        let v33: u64 = (v32 & 15u64);
        let v34: u64 = k0_clmul64_lo(v31, v33);
        let v35: u64 = (v34 & 127u64);
        let v36: u64 = (v35.wrapping_shr((4u64 & 63) as u32));
        let v37: u64 = (v36.wrapping_shl((1u64 & 63) as u32));
        let v38: u64 = (v35 ^ v37);
        let v39: u64 = (v38 ^ v36);
        let v40: u64 = (v39 & 15u64);
        let v41: u64 = 9u64;
        let v42: u64 = (v29 & 15u64);
        let v43: u64 = k0_clmul64_lo(9u64, v42);
        let v44: u64 = (v43 & 127u64);
        let v45: u64 = (v44.wrapping_shr((4u64 & 63) as u32));
        let v46: u64 = (v45.wrapping_shl((1u64 & 63) as u32));
        let v47: u64 = (v44 ^ v46);
        let v48: u64 = (v47 ^ v45);
        let v49: u64 = (v48 & 15u64);
        let v50: u64 = (v20 ^ v49);
        let v51: u64 = (v40 ^ v20);
        let v52: u64 = (v51.wrapping_shl((4u64 & 63) as u32));
        let v53: u64 = (v50 | v52);
        v53
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf16_log_mod15_signed_reduce(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 255u64;
        let v2: u64 = (v0 & 255u64);
        let v3: u64 = 15u64;
        let v4: u64 = (15u64.wrapping_sub(v2));
        let v5: u64 = 63u64;
        let v6: u64 = (v4.wrapping_shr((63u64 & 63) as u32));
        let v7: u64 = 7u64;
        let v8: u64 = (v2.wrapping_shr((7u64 & 63) as u32));
        let v9: u64 = 1u64;
        let v10: u64 = (v8 ^ 1u64);
        let v11: u64 = 0u64;
        let v12: u64 = (v6 & v10);
        let v13: u64 = (0u64.wrapping_sub(v12));
        let v14: u64 = (v2.wrapping_sub(v13));
        let v15: u64 = (v14 & 255u64);
        v15
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_f128_pre_reduce_const_lo(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 135u64;
        let v2: u64 = k0_clmul64_lo(v0, 135u64);
        v2
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_f128_pre_reduce_const_hi(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 135u64;
        let v3: u64 = k0_clmul64_hi(v1, 135u64);
        let v4: u64 = (v0 ^ v3);
        v4
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_f128_reduce_deg191_lo(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 135u64;
        let v3: u64 = k0_clmul64_lo(v1, 135u64);
        let v4: u64 = (v0 ^ v3);
        v4
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_f128_reduce_deg191_hi(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = (v0 ^ v1);
        let v4: u64 = 135u64;
        let v5: u64 = k0_clmul64_hi(v2, 135u64);
        let v6: u64 = (v3 ^ v5);
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_f128_karatsuba_unred_r1(v0: u64, v1: u64, v2: u64, v3: u64) -> u64 {
    unsafe {
        let v4: u64 = (v3 ^ v0);
        let v5: u64 = (v4 ^ v2);
        let v6: u64 = (v1 ^ v5);
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_f128_karatsuba_unred_r2(v0: u64, v1: u64, v2: u64, v3: u64) -> u64 {
    unsafe {
        let v4: u64 = (v3 ^ v0);
        let v5: u64 = (v4 ^ v2);
        let v6: u64 = (v1 ^ v5);
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_urm_pack_c_pair_bank0(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 6148914691236517205u64;
        let v3: u64 = (v0 & 6148914691236517205u64);
        let v4: u64 = (v1 & 6148914691236517205u64);
        let v5: u64 = 1u64;
        let v6: u64 = (v4.wrapping_shl((1u64 & 63) as u32));
        let v7: u64 = (v3 | v6);
        v7
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_urm_pack_c_pair_bank1(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 12297829382473034410u64;
        let v3: u64 = (v0 & 12297829382473034410u64);
        let v4: u64 = 1u64;
        let v5: u64 = (v3.wrapping_shr((1u64 & 63) as u32));
        let v6: u64 = (v1 & 12297829382473034410u64);
        let v7: u64 = (v5 | v6);
        v7
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_f128_fma_lo(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    unsafe {
        let v5: u64 = k0_clmul64_lo(v0, v2);
        let v6: u64 = k0_clmul64_hi(v0, v3);
        let v7: u64 = k0_clmul64_hi(v1, v2);
        let v8: u64 = k0_clmul64_lo(v1, v3);
        let v9: u64 = k0_clmul64_hi(v1, v3);
        let v10: u64 = 135u64;
        let v11: u64 = k0_clmul64_hi(v9, 135u64);
        let v12: u64 = (v6 ^ v7);
        let v13: u64 = (v12 ^ v11);
        let v14: u64 = (v8 ^ v13);
        let v15: u64 = k0_clmul64_lo(v14, 135u64);
        let v16: u64 = (v5 ^ v15);
        let v17: u64 = (v16 ^ v4);
        v17
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_f128_fma_hi(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    unsafe {
        let v5: u64 = k0_clmul64_hi(v0, v2);
        let v6: u64 = k0_clmul64_lo(v0, v3);
        let v7: u64 = k0_clmul64_hi(v0, v3);
        let v8: u64 = k0_clmul64_lo(v1, v2);
        let v9: u64 = k0_clmul64_hi(v1, v2);
        let v10: u64 = k0_clmul64_lo(v1, v3);
        let v11: u64 = k0_clmul64_hi(v1, v3);
        let v12: u64 = 135u64;
        let v13: u64 = k0_clmul64_lo(v11, 135u64);
        let v14: u64 = k0_clmul64_hi(v11, 135u64);
        let v15: u64 = (v7 ^ v9);
        let v16: u64 = (v15 ^ v14);
        let v17: u64 = (v10 ^ v16);
        let v18: u64 = k0_clmul64_hi(v17, 135u64);
        let v19: u64 = (v6 ^ v8);
        let v20: u64 = (v19 ^ v13);
        let v21: u64 = (v5 ^ v18);
        let v22: u64 = (v20 ^ v21);
        let v23: u64 = (v22 ^ v4);
        v23
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ghash_mul_unred_r1(v0: u64, v1: u64, v2: u64, v3: u64) -> u64 {
    unsafe {
        let v4: u64 = k0_clmul64_hi(v0, v2);
        let v5: u64 = k0_clmul64_lo(v0, v3);
        let v6: u64 = k0_clmul64_lo(v1, v2);
        let v7: u64 = (v5 ^ v6);
        let v8: u64 = (v4 ^ v7);
        v8
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ghash_mul_unred_r2(v0: u64, v1: u64, v2: u64, v3: u64) -> u64 {
    unsafe {
        let v4: u64 = k0_clmul64_lo(v1, v3);
        let v5: u64 = k0_clmul64_hi(v0, v3);
        let v6: u64 = k0_clmul64_hi(v1, v2);
        let v7: u64 = (v5 ^ v6);
        let v8: u64 = (v4 ^ v7);
        v8
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_mle_fold_lo(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64, v5: u64) -> u64 {
    unsafe {
        let v6: u64 = (v0 ^ v2);
        let v7: u64 = (v1 ^ v3);
        let v8: u64 = k0_clmul64_lo(v6, v4);
        let v9: u64 = k0_clmul64_hi(v6, v5);
        let v10: u64 = k0_clmul64_hi(v7, v4);
        let v11: u64 = k0_clmul64_lo(v7, v5);
        let v12: u64 = k0_clmul64_hi(v7, v5);
        let v13: u64 = 135u64;
        let v14: u64 = k0_clmul64_hi(v12, 135u64);
        let v15: u64 = (v9 ^ v10);
        let v16: u64 = (v15 ^ v14);
        let v17: u64 = (v11 ^ v16);
        let v18: u64 = k0_clmul64_lo(v17, 135u64);
        let v19: u64 = (v8 ^ v18);
        let v20: u64 = (v0 ^ v19);
        v20
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_mle_fold_hi(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64, v5: u64) -> u64 {
    unsafe {
        let v6: u64 = (v0 ^ v2);
        let v7: u64 = (v1 ^ v3);
        let v8: u64 = k0_clmul64_hi(v6, v4);
        let v9: u64 = k0_clmul64_lo(v6, v5);
        let v10: u64 = k0_clmul64_hi(v6, v5);
        let v11: u64 = k0_clmul64_lo(v7, v4);
        let v12: u64 = k0_clmul64_hi(v7, v4);
        let v13: u64 = k0_clmul64_lo(v7, v5);
        let v14: u64 = k0_clmul64_hi(v7, v5);
        let v15: u64 = 135u64;
        let v16: u64 = k0_clmul64_lo(v14, 135u64);
        let v17: u64 = k0_clmul64_hi(v14, 135u64);
        let v18: u64 = (v10 ^ v12);
        let v19: u64 = (v18 ^ v17);
        let v20: u64 = (v13 ^ v19);
        let v21: u64 = k0_clmul64_hi(v20, 135u64);
        let v22: u64 = (v9 ^ v11);
        let v23: u64 = (v22 ^ v16);
        let v24: u64 = (v8 ^ v21);
        let v25: u64 = (v23 ^ v24);
        let v26: u64 = (v1 ^ v25);
        v26
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_pack8_bool_bytes(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 72340172838076673u64;
        let v2: u64 = (v0 & 72340172838076673u64);
        let v3: u64 = 72624976668147840u64;
        let v4: u64 = (v2.wrapping_mul(72624976668147840u64));
        let v5: u64 = 56u64;
        let v6: u64 = (v4.wrapping_shr((56u64 & 63) as u32));
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_unpack8_bool_bytes(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 127u64;
        let v2: u64 = (v0 & 127u64);
        let v3: u64 = 4432676798593u64;
        let v4: u64 = (v2.wrapping_mul(4432676798593u64));
        let v5: u64 = 128u64;
        let v6: u64 = (v0 & 128u64);
        let v7: u64 = 49u64;
        let v8: u64 = (v6.wrapping_shl((49u64 & 63) as u32));
        let v9: u64 = (v4 | v8);
        let v10: u64 = 72340172838076673u64;
        let v11: u64 = (v9 & 72340172838076673u64);
        v11
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_blake3_chunk_flags(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v10: u64 = 0u64;
        let v11: u64 = (if v0 == 0u64 { 1u64 } else { 0u64 });
        let v12: u64 = (if v11 == 0u64 { 1u64 } else { 0u64 });
        if v12 != 0 {
            let v3: u64 = 0u64;
            let v4: u64 = 1u64;
            let v5: u64 = (v0.wrapping_add(1u64));
            let v6: u64 = (if v5 == v1 { 1u64 } else { 0u64 });
            let v7: u64 = (if v6 == 0u64 { 1u64 } else { 0u64 });
            if v7 != 0 {
                v3
            } else {
                let v8: u64 = 2u64;
                let v9: u64 = 2u64;
                v9
            }
        } else {
            let v13: u64 = 1u64;
            v13
        }
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_mul_by_x(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 255u64;
        let v2: u64 = (v0 & 255u64);
        let v3: u64 = 7u64;
        let v4: u64 = (v2.wrapping_shr((7u64 & 63) as u32));
        let v5: u64 = 1u64;
        let v6: u64 = (v4 & 1u64);
        let v7: u64 = 0u64;
        let v8: u64 = (0u64.wrapping_sub(v6));
        let v9: u64 = (v2.wrapping_shl((1u64 & 63) as u32));
        let v10: u64 = 27u64;
        let v11: u64 = (v8 & 27u64);
        let v12: u64 = (v9 ^ v11);
        let v13: u64 = (v12 & 255u64);
        v13
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_square(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 255u64;
        let v2: u64 = (v0 & 255u64);
        let v3: u64 = 4u64;
        let v4: u64 = (v2.wrapping_shl((4u64 & 63) as u32));
        let v5: u64 = (v2 ^ v4);
        let v6: u64 = 3855u64;
        let v7: u64 = (v5 & 3855u64);
        let v8: u64 = 2u64;
        let v9: u64 = (v7.wrapping_shl((2u64 & 63) as u32));
        let v10: u64 = (v7 ^ v9);
        let v11: u64 = 13107u64;
        let v12: u64 = (v10 & 13107u64);
        let v13: u64 = 1u64;
        let v14: u64 = (v12.wrapping_shl((1u64 & 63) as u32));
        let v15: u64 = (v12 ^ v14);
        let v16: u64 = 21845u64;
        let v17: u64 = (v15 & 21845u64);
        let v18: u64 = 8u64;
        let v19: u64 = (v17.wrapping_shr((8u64 & 63) as u32));
        let v20: u64 = (v17 & 255u64);
        let v21: u64 = (v20 ^ v19);
        let v22: u64 = (v19.wrapping_shl((1u64 & 63) as u32));
        let v23: u64 = (v21 ^ v22);
        let v24: u64 = 3u64;
        let v25: u64 = (v19.wrapping_shl((3u64 & 63) as u32));
        let v26: u64 = (v23 ^ v25);
        let v27: u64 = (v19.wrapping_shl((4u64 & 63) as u32));
        let v28: u64 = (v26 ^ v27);
        let v29: u64 = (v28.wrapping_shr((8u64 & 63) as u32));
        let v30: u64 = (v28 & 255u64);
        let v31: u64 = (v30 ^ v29);
        let v32: u64 = (v29.wrapping_shl((1u64 & 63) as u32));
        let v33: u64 = (v31 ^ v32);
        let v34: u64 = (v29.wrapping_shl((3u64 & 63) as u32));
        let v35: u64 = (v33 ^ v34);
        let v36: u64 = (v29.wrapping_shl((4u64 & 63) as u32));
        let v37: u64 = (v35 ^ v36);
        let v38: u64 = (v37 & 255u64);
        v38
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_ntt_next_s(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 255u64;
        let v3: u64 = (v0 & 255u64);
        let v4: u64 = (v0 ^ v1);
        let v5: u64 = (v4 & 255u64);
        let v6: u64 = k0_clmul64_lo(v3, v5);
        let v7: u64 = 65535u64;
        let v8: u64 = (v6 & 65535u64);
        let v9: u64 = 8u64;
        let v10: u64 = (v8.wrapping_shr((8u64 & 63) as u32));
        let v11: u64 = (v8 & 255u64);
        let v12: u64 = (v11 ^ v10);
        let v13: u64 = 1u64;
        let v14: u64 = (v10.wrapping_shl((1u64 & 63) as u32));
        let v15: u64 = (v12 ^ v14);
        let v16: u64 = 3u64;
        let v17: u64 = (v10.wrapping_shl((3u64 & 63) as u32));
        let v18: u64 = (v15 ^ v17);
        let v19: u64 = 4u64;
        let v20: u64 = (v10.wrapping_shl((4u64 & 63) as u32));
        let v21: u64 = (v18 ^ v20);
        let v22: u64 = (v21.wrapping_shr((8u64 & 63) as u32));
        let v23: u64 = (v21 & 255u64);
        let v24: u64 = (v23 ^ v22);
        let v25: u64 = (v22.wrapping_shl((1u64 & 63) as u32));
        let v26: u64 = (v24 ^ v25);
        let v27: u64 = (v22.wrapping_shl((3u64 & 63) as u32));
        let v28: u64 = (v26 ^ v27);
        let v29: u64 = (v22.wrapping_shl((4u64 & 63) as u32));
        let v30: u64 = (v28 ^ v29);
        let v31: u64 = (v30 & 255u64);
        v31
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_bitrecord_flush_word(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 63u64;
        let v4: u64 = (v2 & 63u64);
        let v5: u64 = (v0.wrapping_shl((v4 & 63) as u32));
        let v6: u64 = (v5 | v1);
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_bitrecord_flush_spill(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 1u64;
        let v3: u64 = (v0.wrapping_shr((1u64 & 63) as u32));
        let v4: u64 = 63u64;
        let v5: u64 = (v1 & 63u64);
        let v6: u64 = (63u64.wrapping_sub(v5));
        let v7: u64 = (v3.wrapping_shr((v6 & 63) as u32));
        v7
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_delta_swap_lo(v0: u64, v1: u64, v2: u64, v3: u64) -> u64 {
    unsafe {
        let v4: u64 = 63u64;
        let v5: u64 = (v2 & 63u64);
        let v6: u64 = (v0.wrapping_shr((v5 & 63) as u32));
        let v7: u64 = (v6 ^ v1);
        let v8: u64 = (v7 & v3);
        let v9: u64 = (v8.wrapping_shl((v5 & 63) as u32));
        let v10: u64 = (v0 ^ v9);
        v10
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_delta_swap_hi(v0: u64, v1: u64, v2: u64, v3: u64) -> u64 {
    unsafe {
        let v4: u64 = 63u64;
        let v5: u64 = (v2 & 63u64);
        let v6: u64 = (v0.wrapping_shr((v5 & 63) as u32));
        let v7: u64 = (v6 ^ v1);
        let v8: u64 = (v7 & v3);
        let v9: u64 = (v1 ^ v8);
        v9
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ntt_replicated_msg_idx(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 1u64;
        let v3: u64 = 63u64;
        let v4: u64 = (v1 & 63u64);
        let v5: u64 = (1u64.wrapping_shl((v4 & 63) as u32));
        let v6: u64 = (v5.wrapping_sub(1u64));
        let v7: u64 = (v0 & v6);
        v7
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_inv_ntt_cyclic_offset_lo(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 2u64;
        let v2: u64 = (v0 & 2u64);
        let v3: u64 = 3u64;
        let v4: u64 = (v2.wrapping_shl((3u64 & 63) as u32));
        let v5: u64 = 4u64;
        let v6: u64 = (v0 & 4u64);
        let v7: u64 = 12u64;
        let v8: u64 = (v6.wrapping_mul(12u64));
        let v9: u64 = (v4.wrapping_add(v8));
        v9
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_inv_ntt_cyclic_offset_hi(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 2u64;
        let v2: u64 = (v0 & 2u64);
        let v3: u64 = 3u64;
        let v4: u64 = (v2.wrapping_shl((3u64 & 63) as u32));
        let v5: u64 = 4u64;
        let v6: u64 = (v0 ^ 4u64);
        let v7: u64 = (v6 & 4u64);
        let v8: u64 = 12u64;
        let v9: u64 = (v7.wrapping_mul(12u64));
        let v10: u64 = (v4.wrapping_add(v9));
        v10
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ntt_dual_block_src_offset(
    v0: u64,
    v1: u64,
    v2: u64,
    v3: u64,
    v4: u64,
) -> u64 {
    unsafe {
        let v5: u64 = (v3.wrapping_mul(v0));
        let v6: u64 = (v5.wrapping_add(v2));
        let v7: u64 = (v6.wrapping_mul(v1));
        let v8: u64 = (v7.wrapping_add(v4));
        v8
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ntt_dual_block_dst_offset(
    v0: u64,
    v1: u64,
    v2: u64,
    v3: u64,
    v4: u64,
    v5: u64,
) -> u64 {
    unsafe {
        let v6: u64 = 4u64;
        let v7: u64 = (v0.wrapping_shl((4u64 & 63) as u32));
        let v8: u64 = (v7.wrapping_mul(v1));
        let v9: u64 = (v2.wrapping_mul(v8));
        let v10: u64 = (v4.wrapping_mul(v0));
        let v11: u64 = (v10.wrapping_add(v3));
        let v12: u64 = (v11.wrapping_mul(v1));
        let v13: u64 = (v12.wrapping_add(v5));
        let v14: u64 = (v9.wrapping_add(v13));
        v14
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_sha256_ch_and_out(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 4294967295u64;
        let v4: u64 = (v1 ^ v2);
        let v5: u64 = (v0 & v4);
        let v6: u64 = (v5 & 4294967295u64);
        let v7: u64 = (v2 & 4294967295u64);
        let v8: u64 = (v6 ^ v7);
        let v9: u64 = 32u64;
        let v10: u64 = (v8.wrapping_shl((32u64 & 63) as u32));
        let v11: u64 = (v6 | v10);
        v11
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_sha256_maj_and_out(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 4294967295u64;
        let v4: u64 = (v0 ^ v1);
        let v5: u64 = (v0 ^ v2);
        let v6: u64 = (v4 & v5);
        let v7: u64 = (v6 & 4294967295u64);
        let v8: u64 = (v0 & 4294967295u64);
        let v9: u64 = (v7 ^ v8);
        let v10: u64 = 32u64;
        let v11: u64 = (v9.wrapping_shl((32u64 & 63) as u32));
        let v12: u64 = (v7 | v11);
        v12
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_sha256_big_sigma0(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 4294967295u64;
        let v2: u64 = (v0 & 4294967295u64);
        let v3: u64 = 2u64;
        let v4: u64 = (v2.wrapping_shr((2u64 & 63) as u32));
        let v5: u64 = 30u64;
        let v6: u64 = (v2.wrapping_shl((30u64 & 63) as u32));
        let v7: u64 = (v4 | v6);
        let v8: u64 = (v7 & 4294967295u64);
        let v9: u64 = 13u64;
        let v10: u64 = (v2.wrapping_shr((13u64 & 63) as u32));
        let v11: u64 = 19u64;
        let v12: u64 = (v2.wrapping_shl((19u64 & 63) as u32));
        let v13: u64 = (v10 | v12);
        let v14: u64 = (v13 & 4294967295u64);
        let v15: u64 = 22u64;
        let v16: u64 = (v2.wrapping_shr((22u64 & 63) as u32));
        let v17: u64 = 10u64;
        let v18: u64 = (v2.wrapping_shl((10u64 & 63) as u32));
        let v19: u64 = (v16 | v18);
        let v20: u64 = (v19 & 4294967295u64);
        let v21: u64 = (v8 ^ v14);
        let v22: u64 = (v21 ^ v20);
        v22
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_sha256_big_sigma1(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 4294967295u64;
        let v2: u64 = (v0 & 4294967295u64);
        let v3: u64 = 6u64;
        let v4: u64 = (v2.wrapping_shr((6u64 & 63) as u32));
        let v5: u64 = 26u64;
        let v6: u64 = (v2.wrapping_shl((26u64 & 63) as u32));
        let v7: u64 = (v4 | v6);
        let v8: u64 = (v7 & 4294967295u64);
        let v9: u64 = 11u64;
        let v10: u64 = (v2.wrapping_shr((11u64 & 63) as u32));
        let v11: u64 = 21u64;
        let v12: u64 = (v2.wrapping_shl((21u64 & 63) as u32));
        let v13: u64 = (v10 | v12);
        let v14: u64 = (v13 & 4294967295u64);
        let v15: u64 = 25u64;
        let v16: u64 = (v2.wrapping_shr((25u64 & 63) as u32));
        let v17: u64 = 7u64;
        let v18: u64 = (v2.wrapping_shl((7u64 & 63) as u32));
        let v19: u64 = (v16 | v18);
        let v20: u64 = (v19 & 4294967295u64);
        let v21: u64 = (v8 ^ v14);
        let v22: u64 = (v21 ^ v20);
        v22
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_sha256_small_sigma0(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 4294967295u64;
        let v2: u64 = (v0 & 4294967295u64);
        let v3: u64 = 7u64;
        let v4: u64 = (v2.wrapping_shr((7u64 & 63) as u32));
        let v5: u64 = 25u64;
        let v6: u64 = (v2.wrapping_shl((25u64 & 63) as u32));
        let v7: u64 = (v4 | v6);
        let v8: u64 = (v7 & 4294967295u64);
        let v9: u64 = 18u64;
        let v10: u64 = (v2.wrapping_shr((18u64 & 63) as u32));
        let v11: u64 = 14u64;
        let v12: u64 = (v2.wrapping_shl((14u64 & 63) as u32));
        let v13: u64 = (v10 | v12);
        let v14: u64 = (v13 & 4294967295u64);
        let v15: u64 = 3u64;
        let v16: u64 = (v2.wrapping_shr((3u64 & 63) as u32));
        let v17: u64 = (v8 ^ v14);
        let v18: u64 = (v17 ^ v16);
        v18
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_sha256_small_sigma1(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 4294967295u64;
        let v2: u64 = (v0 & 4294967295u64);
        let v3: u64 = 17u64;
        let v4: u64 = (v2.wrapping_shr((17u64 & 63) as u32));
        let v5: u64 = 15u64;
        let v6: u64 = (v2.wrapping_shl((15u64 & 63) as u32));
        let v7: u64 = (v4 | v6);
        let v8: u64 = (v7 & 4294967295u64);
        let v9: u64 = 19u64;
        let v10: u64 = (v2.wrapping_shr((19u64 & 63) as u32));
        let v11: u64 = 13u64;
        let v12: u64 = (v2.wrapping_shl((13u64 & 63) as u32));
        let v13: u64 = (v10 | v12);
        let v14: u64 = (v13 & 4294967295u64);
        let v15: u64 = 10u64;
        let v16: u64 = (v2.wrapping_shr((10u64 & 63) as u32));
        let v17: u64 = (v8 ^ v14);
        let v18: u64 = (v17 ^ v16);
        v18
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_keccak_rol64(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 63u64;
        let v3: u64 = (v1 & 63u64);
        let v4: u64 = (v0.wrapping_shl((v3 & 63) as u32));
        let v5: u64 = 1u64;
        let v6: u64 = (v0.wrapping_shr((1u64 & 63) as u32));
        let v7: u64 = (63u64.wrapping_sub(v3));
        let v8: u64 = (v6.wrapping_shr((v7 & 63) as u32));
        let v9: u64 = (v4 | v8);
        v9
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_keccak_theta_d(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 1u64;
        let v3: u64 = (v1.wrapping_shl((1u64 & 63) as u32));
        let v4: u64 = 63u64;
        let v5: u64 = (v1.wrapping_shr((63u64 & 63) as u32));
        let v6: u64 = (v3 | v5);
        let v7: u64 = (v0 ^ v6);
        v7
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_keccak_chi_and(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 18446744073709551615u64;
        let v3: u64 = (v0 ^ 18446744073709551615u64);
        let v4: u64 = (v3 & v1);
        v4
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_keccak_chi(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 18446744073709551615u64;
        let v4: u64 = (v1 ^ 18446744073709551615u64);
        let v5: u64 = (v4 & v2);
        let v6: u64 = (v0 ^ v5);
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_pack_even_bytes_u32(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 71777214294589695u64;
        let v2: u64 = (v0 & 71777214294589695u64);
        let v3: u64 = 8u64;
        let v4: u64 = (v2.wrapping_shr((8u64 & 63) as u32));
        let v5: u64 = (v2 | v4);
        let v6: u64 = 281470681808895u64;
        let v7: u64 = (v5 & 281470681808895u64);
        let v8: u64 = 16u64;
        let v9: u64 = (v7.wrapping_shr((16u64 & 63) as u32));
        let v10: u64 = (v7 | v9);
        let v11: u64 = 4294967295u64;
        let v12: u64 = (v10 & 4294967295u64);
        v12
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_pack_even_bytes_x8(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 71777214294589695u64;
        let v3: u64 = (v0 & 71777214294589695u64);
        let v4: u64 = 8u64;
        let v5: u64 = (v3.wrapping_shr((8u64 & 63) as u32));
        let v6: u64 = (v3 | v5);
        let v7: u64 = 281470681808895u64;
        let v8: u64 = (v6 & 281470681808895u64);
        let v9: u64 = 16u64;
        let v10: u64 = (v8.wrapping_shr((16u64 & 63) as u32));
        let v11: u64 = (v8 | v10);
        let v12: u64 = 4294967295u64;
        let v13: u64 = (v11 & 4294967295u64);
        let v14: u64 = (v1 & 71777214294589695u64);
        let v15: u64 = (v14.wrapping_shr((8u64 & 63) as u32));
        let v16: u64 = (v14 | v15);
        let v17: u64 = (v16 & 281470681808895u64);
        let v18: u64 = (v17.wrapping_shr((16u64 & 63) as u32));
        let v19: u64 = (v17 | v18);
        let v20: u64 = 32u64;
        let v21: u64 = (v19.wrapping_shl((32u64 & 63) as u32));
        let v22: u64 = (v13 | v21);
        v22
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_unpack_even_bytes_u32(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 4294967295u64;
        let v2: u64 = (v0 & 4294967295u64);
        let v3: u64 = 16u64;
        let v4: u64 = (v2.wrapping_shl((16u64 & 63) as u32));
        let v5: u64 = (v2 | v4);
        let v6: u64 = 281470681808895u64;
        let v7: u64 = (v5 & 281470681808895u64);
        let v8: u64 = 8u64;
        let v9: u64 = (v7.wrapping_shl((8u64 & 63) as u32));
        let v10: u64 = (v7 | v9);
        let v11: u64 = 71777214294589695u64;
        let v12: u64 = (v10 & 71777214294589695u64);
        v12
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_split_even_odd_bits_u8(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 85u64;
        let v2: u64 = (v0 & 85u64);
        let v3: u64 = 170u64;
        let v4: u64 = (v0 & 170u64);
        let v5: u64 = 32u64;
        let v6: u64 = (v4.wrapping_shl((32u64 & 63) as u32));
        let v7: u64 = (v2 | v6);
        v7
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_fft_btf_pair(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 255u64;
        let v4: u64 = (v2 & 255u64);
        let v5: u64 = (v1 & 255u64);
        let v6: u64 = k0_clmul64_lo(v4, v5);
        let v7: u64 = 65535u64;
        let v8: u64 = (v6 & 65535u64);
        let v9: u64 = 8u64;
        let v10: u64 = (v8.wrapping_shr((8u64 & 63) as u32));
        let v11: u64 = (v8 & 255u64);
        let v12: u64 = (v11 ^ v10);
        let v13: u64 = 1u64;
        let v14: u64 = (v10.wrapping_shl((1u64 & 63) as u32));
        let v15: u64 = (v12 ^ v14);
        let v16: u64 = 3u64;
        let v17: u64 = (v10.wrapping_shl((3u64 & 63) as u32));
        let v18: u64 = (v15 ^ v17);
        let v19: u64 = 4u64;
        let v20: u64 = (v10.wrapping_shl((4u64 & 63) as u32));
        let v21: u64 = (v18 ^ v20);
        let v22: u64 = (v21.wrapping_shr((8u64 & 63) as u32));
        let v23: u64 = (v21 & 255u64);
        let v24: u64 = (v23 ^ v22);
        let v25: u64 = (v22.wrapping_shl((1u64 & 63) as u32));
        let v26: u64 = (v24 ^ v25);
        let v27: u64 = (v22.wrapping_shl((3u64 & 63) as u32));
        let v28: u64 = (v26 ^ v27);
        let v29: u64 = (v22.wrapping_shl((4u64 & 63) as u32));
        let v30: u64 = (v28 ^ v29);
        let v31: u64 = (v30 & 255u64);
        let v32: u64 = (v0 & 255u64);
        let v33: u64 = (v32 ^ v31);
        let v34: u64 = (v5 ^ v33);
        let v35: u64 = 32u64;
        let v36: u64 = (v34.wrapping_shl((32u64 & 63) as u32));
        let v37: u64 = (v33 | v36);
        v37
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_ifft_btf_pair(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = 255u64;
        let v4: u64 = (v0 & 255u64);
        let v5: u64 = (v1 & 255u64);
        let v6: u64 = (v5 ^ v4);
        let v7: u64 = (v2 & 255u64);
        let v8: u64 = (v6 & 255u64);
        let v9: u64 = k0_clmul64_lo(v7, v8);
        let v10: u64 = 65535u64;
        let v11: u64 = (v9 & 65535u64);
        let v12: u64 = 8u64;
        let v13: u64 = (v11.wrapping_shr((8u64 & 63) as u32));
        let v14: u64 = (v11 & 255u64);
        let v15: u64 = (v14 ^ v13);
        let v16: u64 = 1u64;
        let v17: u64 = (v13.wrapping_shl((1u64 & 63) as u32));
        let v18: u64 = (v15 ^ v17);
        let v19: u64 = 3u64;
        let v20: u64 = (v13.wrapping_shl((3u64 & 63) as u32));
        let v21: u64 = (v18 ^ v20);
        let v22: u64 = 4u64;
        let v23: u64 = (v13.wrapping_shl((4u64 & 63) as u32));
        let v24: u64 = (v21 ^ v23);
        let v25: u64 = (v24.wrapping_shr((8u64 & 63) as u32));
        let v26: u64 = (v24 & 255u64);
        let v27: u64 = (v26 ^ v25);
        let v28: u64 = (v25.wrapping_shl((1u64 & 63) as u32));
        let v29: u64 = (v27 ^ v28);
        let v30: u64 = (v25.wrapping_shl((3u64 & 63) as u32));
        let v31: u64 = (v29 ^ v30);
        let v32: u64 = (v25.wrapping_shl((4u64 & 63) as u32));
        let v33: u64 = (v31 ^ v32);
        let v34: u64 = (v33 & 255u64);
        let v35: u64 = (v4 ^ v34);
        let v36: u64 = 32u64;
        let v37: u64 = (v6.wrapping_shl((32u64 & 63) as u32));
        let v38: u64 = (v35 | v37);
        v38
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_isolate_lowest_one(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 0u64;
        let v2: u64 = (0u64.wrapping_sub(v0));
        let v3: u64 = (v0 & v2);
        v3
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_clear_lowest_one(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 1u64;
        let v2: u64 = (v0.wrapping_sub(1u64));
        let v3: u64 = (v0 & v2);
        v3
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_bit_deposit_lowest_one(v0: u64, v1: u64) -> u64 {
    unsafe {
        let v2: u64 = 0u64;
        let v3: u64 = (0u64.wrapping_sub(v1));
        let v4: u64 = (v1 & v3);
        let v5: u64 = 1u64;
        let v6: u64 = (v0 & 1u64);
        let v7: u64 = (0u64.wrapping_sub(v6));
        let v8: u64 = (v4 & v7);
        v8
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_gf8_subspace_w2(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 255u64;
        let v2: u64 = (v0 & 255u64);
        let v3: u64 = (v2 & 255u64);
        let v4: u64 = k0_clmul64_lo(v3, v3);
        let v5: u64 = 65535u64;
        let v6: u64 = (v4 & 65535u64);
        let v7: u64 = 8u64;
        let v8: u64 = (v6.wrapping_shr((8u64 & 63) as u32));
        let v9: u64 = (v6 & 255u64);
        let v10: u64 = (v9 ^ v8);
        let v11: u64 = 1u64;
        let v12: u64 = (v8.wrapping_shl((1u64 & 63) as u32));
        let v13: u64 = (v10 ^ v12);
        let v14: u64 = 3u64;
        let v15: u64 = (v8.wrapping_shl((3u64 & 63) as u32));
        let v16: u64 = (v13 ^ v15);
        let v17: u64 = 4u64;
        let v18: u64 = (v8.wrapping_shl((4u64 & 63) as u32));
        let v19: u64 = (v16 ^ v18);
        let v20: u64 = (v19.wrapping_shr((8u64 & 63) as u32));
        let v21: u64 = (v19 & 255u64);
        let v22: u64 = (v21 ^ v20);
        let v23: u64 = (v20.wrapping_shl((1u64 & 63) as u32));
        let v24: u64 = (v22 ^ v23);
        let v25: u64 = (v20.wrapping_shl((3u64 & 63) as u32));
        let v26: u64 = (v24 ^ v25);
        let v27: u64 = (v20.wrapping_shl((4u64 & 63) as u32));
        let v28: u64 = (v26 ^ v27);
        let v29: u64 = (v28 & 255u64);
        let v30: u64 = (v29 ^ v2);
        let v31: u64 = (v30 & 255u64);
        let v32: u64 = 6u64;
        let v33: u64 = (v30 ^ 6u64);
        let v34: u64 = (v33 & 255u64);
        let v35: u64 = k0_clmul64_lo(v31, v34);
        let v36: u64 = (v35 & 65535u64);
        let v37: u64 = (v36.wrapping_shr((8u64 & 63) as u32));
        let v38: u64 = (v36 & 255u64);
        let v39: u64 = (v38 ^ v37);
        let v40: u64 = (v37.wrapping_shl((1u64 & 63) as u32));
        let v41: u64 = (v39 ^ v40);
        let v42: u64 = (v37.wrapping_shl((3u64 & 63) as u32));
        let v43: u64 = (v41 ^ v42);
        let v44: u64 = (v37.wrapping_shl((4u64 & 63) as u32));
        let v45: u64 = (v43 ^ v44);
        let v46: u64 = (v45.wrapping_shr((8u64 & 63) as u32));
        let v47: u64 = (v45 & 255u64);
        let v48: u64 = (v47 ^ v46);
        let v49: u64 = (v46.wrapping_shl((1u64 & 63) as u32));
        let v50: u64 = (v48 ^ v49);
        let v51: u64 = (v46.wrapping_shl((3u64 & 63) as u32));
        let v52: u64 = (v50 ^ v51);
        let v53: u64 = (v46.wrapping_shl((4u64 & 63) as u32));
        let v54: u64 = (v52 ^ v53);
        let v55: u64 = (v54 & 255u64);
        v55
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_nibble_lo(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 15u64;
        let v2: u64 = (v0 & 15u64);
        v2
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_nibble_hi(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 4u64;
        let v2: u64 = (v0.wrapping_shr((4u64 & 63) as u32));
        let v3: u64 = 15u64;
        let v4: u64 = (v2 & 15u64);
        v4
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_jagged_transition(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    unsafe {
        let v5: u64 = 1u64;
        let v6: u64 = (v0 & 1u64);
        let v7: u64 = (v1 & 1u64);
        let v8: u64 = (v2 & 1u64);
        let v9: u64 = (v3 & 1u64);
        let v10: u64 = (v4 & 1u64);
        let v11: u64 = (v4.wrapping_shr((1u64 & 63) as u32));
        let v12: u64 = (v11 & 1u64);
        let v13: u64 = (v6 ^ v10);
        let v14: u64 = (v13 ^ v8);
        let v15: u64 = (v14 ^ v7);
        let v16: u64 = (v6 & v10);
        let v17: u64 = (v6 & v8);
        let v18: u64 = (v16 | v17);
        let v19: u64 = (v10 & v8);
        let v20: u64 = (v18 | v19);
        let v21: u64 = (v7 ^ v9);
        let v22: u64 = (v21 & v9);
        let v23: u64 = (v21 ^ 1u64);
        let v24: u64 = (v23 & v12);
        let v25: u64 = (v22 | v24);
        let v26: u64 = (v25.wrapping_shl((1u64 & 63) as u32));
        let v27: u64 = (v20 | v26);
        let v28: u64 = (v15.wrapping_sub(1u64));
        let v29: u64 = (v27 & v28);
        let v30: u64 = 2u64;
        let v31: u64 = (v15.wrapping_shl((2u64 & 63) as u32));
        let v32: u64 = (v29 | v31);
        v32
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_slot_indicator_lo(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    unsafe {
        let v5: u64 = 1u64;
        let v6: u64 = (v0 & 1u64);
        let v7: u64 = (v6 ^ 1u64);
        let v8: u64 = (v1 ^ v7);
        let v9: u64 = (v0.wrapping_shr((1u64 & 63) as u32));
        let v10: u64 = (v9 & 1u64);
        let v11: u64 = (v10 ^ 1u64);
        let v12: u64 = (v3 ^ v11);
        let v1017: u64 = k0_clmul64_lo(v8, v12);
        let v1018: u64 = k0_clmul64_hi(v8, v4);
        let v1019: u64 = k0_clmul64_hi(v2, v12);
        let v1020: u64 = k0_clmul64_lo(v2, v4);
        let v1021: u64 = k0_clmul64_hi(v2, v4);
        let v1022: u64 = 135u64;
        let v1023: u64 = k0_clmul64_hi(v1021, 135u64);
        let v1024: u64 = (v1018 ^ v1019);
        let v1025: u64 = (v1024 ^ v1023);
        let v1026: u64 = (v1020 ^ v1025);
        let v1027: u64 = k0_clmul64_lo(v1026, 135u64);
        let v1028: u64 = (v1017 ^ v1027);
        let v13: u64 = v1028;
        v13
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_slot_indicator_hi(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    unsafe {
        let v5: u64 = 1u64;
        let v6: u64 = (v0 & 1u64);
        let v7: u64 = (v6 ^ 1u64);
        let v8: u64 = (v1 ^ v7);
        let v9: u64 = (v0.wrapping_shr((1u64 & 63) as u32));
        let v10: u64 = (v9 & 1u64);
        let v11: u64 = (v10 ^ 1u64);
        let v12: u64 = (v3 ^ v11);
        let v1017: u64 = k0_clmul64_hi(v8, v12);
        let v1018: u64 = k0_clmul64_lo(v8, v4);
        let v1019: u64 = k0_clmul64_hi(v8, v4);
        let v1020: u64 = k0_clmul64_lo(v2, v12);
        let v1021: u64 = k0_clmul64_hi(v2, v12);
        let v1022: u64 = k0_clmul64_lo(v2, v4);
        let v1023: u64 = k0_clmul64_hi(v2, v4);
        let v1024: u64 = 135u64;
        let v1025: u64 = k0_clmul64_lo(v1023, 135u64);
        let v1026: u64 = k0_clmul64_hi(v1023, 135u64);
        let v1027: u64 = (v1019 ^ v1021);
        let v1028: u64 = (v1027 ^ v1026);
        let v1029: u64 = (v1022 ^ v1028);
        let v1030: u64 = k0_clmul64_hi(v1029, 135u64);
        let v1031: u64 = (v1018 ^ v1020);
        let v1032: u64 = (v1031 ^ v1025);
        let v1033: u64 = (v1017 ^ v1030);
        let v1034: u64 = (v1032 ^ v1033);
        let v13: u64 = v1034;
        v13
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_eq_bit_mul_lo(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    unsafe {
        let v5: u64 = 1u64;
        let v6: u64 = (v0 & 1u64);
        let v7: u64 = (v6 ^ 1u64);
        let v8: u64 = (v3 ^ v7);
        let v1013: u64 = k0_clmul64_lo(v1, v8);
        let v1014: u64 = k0_clmul64_hi(v1, v4);
        let v1015: u64 = k0_clmul64_hi(v2, v8);
        let v1016: u64 = k0_clmul64_lo(v2, v4);
        let v1017: u64 = k0_clmul64_hi(v2, v4);
        let v1018: u64 = 135u64;
        let v1019: u64 = k0_clmul64_hi(v1017, 135u64);
        let v1020: u64 = (v1014 ^ v1015);
        let v1021: u64 = (v1020 ^ v1019);
        let v1022: u64 = (v1016 ^ v1021);
        let v1023: u64 = k0_clmul64_lo(v1022, 135u64);
        let v1024: u64 = (v1013 ^ v1023);
        let v9: u64 = v1024;
        v9
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_eq_bit_mul_hi(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    unsafe {
        let v5: u64 = 1u64;
        let v6: u64 = (v0 & 1u64);
        let v7: u64 = (v6 ^ 1u64);
        let v8: u64 = (v3 ^ v7);
        let v1013: u64 = k0_clmul64_hi(v1, v8);
        let v1014: u64 = k0_clmul64_lo(v1, v4);
        let v1015: u64 = k0_clmul64_hi(v1, v4);
        let v1016: u64 = k0_clmul64_lo(v2, v8);
        let v1017: u64 = k0_clmul64_hi(v2, v8);
        let v1018: u64 = k0_clmul64_lo(v2, v4);
        let v1019: u64 = k0_clmul64_hi(v2, v4);
        let v1020: u64 = 135u64;
        let v1021: u64 = k0_clmul64_lo(v1019, 135u64);
        let v1022: u64 = k0_clmul64_hi(v1019, 135u64);
        let v1023: u64 = (v1015 ^ v1017);
        let v1024: u64 = (v1023 ^ v1022);
        let v1025: u64 = (v1018 ^ v1024);
        let v1026: u64 = k0_clmul64_hi(v1025, 135u64);
        let v1027: u64 = (v1014 ^ v1016);
        let v1028: u64 = (v1027 ^ v1021);
        let v1029: u64 = (v1013 ^ v1026);
        let v1030: u64 = (v1028 ^ v1029);
        let v9: u64 = v1030;
        v9
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_monomial_basis128_lo(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 6u64;
        let v2: u64 = (v0.wrapping_shr((6u64 & 63) as u32));
        let v3: u64 = 1u64;
        let v4: u64 = (v2 & 1u64);
        let v5: u64 = (v4 ^ 1u64);
        let v6: u64 = 63u64;
        let v7: u64 = (v0 & 63u64);
        let v8: u64 = (v5.wrapping_shl((v7 & 63) as u32));
        v8
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_monomial_basis128_hi(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 6u64;
        let v2: u64 = (v0.wrapping_shr((6u64 & 63) as u32));
        let v3: u64 = 1u64;
        let v4: u64 = (v2 & 1u64);
        let v5: u64 = 63u64;
        let v6: u64 = (v0 & 63u64);
        let v7: u64 = (v4.wrapping_shl((v6 & 63) as u32));
        v7
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ntt_radix4_comb_a(v0: u64, v1: u64, v2: u64) -> u64 {
    unsafe {
        let v3: u64 = (v0 ^ v1);
        let v4: u64 = (v3 ^ v2);
        v4
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ntt_radix4_comb_b(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    unsafe {
        let v5: u64 = (v1 ^ v3);
        let v6: u64 = (v5 ^ v0);
        let v7: u64 = (v6 ^ v2);
        let v8: u64 = (v7 ^ v4);
        v8
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ntt_radix4_comb_c(v0: u64, v1: u64, v2: u64, v3: u64) -> u64 {
    unsafe {
        let v4: u64 = (v1 ^ v0);
        let v5: u64 = (v4 ^ v2);
        let v6: u64 = (v5 ^ v3);
        v6
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_ntt_radix4_comb_d(
    v0: u64,
    v1: u64,
    v2: u64,
    v3: u64,
    v4: u64,
    v5: u64,
    v6: u64,
) -> u64 {
    unsafe {
        let v7: u64 = (v3 ^ v1);
        let v8: u64 = (v7 ^ v5);
        let v9: u64 = (v8 ^ v2);
        let v10: u64 = (v9 ^ v0);
        let v11: u64 = (v10 ^ v4);
        let v12: u64 = (v11 ^ v6);
        v12
    }
}

pub mod s1 {

    use rayon::prelude::*;

    type Compression = ([u32; 8], [u32; 16], u64, u32, u32);

    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

    const DRAWS_PER_BLOCK: usize = 25;
    const ZERO_COMPRESSION: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

    #[inline(always)]
    pub fn mix(mut z: u64) -> u32 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }

    #[inline(always)]
    pub fn generator_init(log2_size: u32, seed: u64) -> u64 {
        seed ^ u64::from(log2_size).rotate_left(29)
    }

    pub fn generate_compressions_par(log2_size: u32, seed: u64) -> Vec<Compression> {
        let mut out = vec![ZERO_COMPRESSION; 1usize << log2_size];
        fill_compressions_par(&mut out, log2_size, seed);
        out
    }

    pub fn fill_compressions_par(out: &mut [Compression], log2_size: u32, seed: u64) {
        fill_compressions_from_init(out, generator_init(log2_size, seed));
    }

    pub fn fill_compressions_from_init(out: &mut [Compression], init: u64) {
        let ilp = !gen_block_ilp_killed();
        out.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(chunk_index, dst)| {
                let base = chunk_index * 4096;
                for (offset, slot) in dst.iter_mut().enumerate() {
                    *slot = gen_block_with(init, base + offset, ilp);
                }
            });
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn gen_block(init: u64, block: usize) -> Compression {
        gen_block_with(init, block, !gen_block_ilp_killed())
    }

    pub fn gen_block_ilp_killed() -> bool {
        std::env::var("FLOCK_NO_GEN_BLOCK_SIMD").is_ok_and(|v| v == "1")
    }

    #[inline(always)]
    pub fn gen_block_with(init: u64, block: usize, ilp: bool) -> Compression {
        if ilp {
            gen_block_ilp(init, block)
        } else {
            gen_block_scalar(init, block)
        }
    }

    #[inline(always)]
    pub fn gen_block_scalar(init: u64, block: usize) -> Compression {
        let mut s = init.wrapping_add(((DRAWS_PER_BLOCK * block) as u64).wrapping_mul(GOLDEN));
        let mut cv = [0u32; 8];
        for word in cv.iter_mut() {
            s = s.wrapping_add(GOLDEN);
            *word = mix(s);
        }
        let mut message = [0u32; 16];
        for word in message.iter_mut() {
            s = s.wrapping_add(GOLDEN);
            *word = mix(s);
        }
        s = s.wrapping_add(GOLDEN);
        (cv, message, u64::from(mix(s)), 64, 11)
    }

    #[inline(always)]
    pub fn gen_block_ilp(init: u64, block: usize) -> Compression {
        let base = init.wrapping_add(((DRAWS_PER_BLOCK * block) as u64).wrapping_mul(GOLDEN));
        #[inline(always)]
        fn draw(base: u64, d: usize) -> u32 {
            mix(base.wrapping_add(((d + 1) as u64).wrapping_mul(GOLDEN)))
        }
        let mut out = [0u32; DRAWS_PER_BLOCK];
        let mut d = 0usize;
        while d + 4 <= DRAWS_PER_BLOCK {
            let (o0, o1, o2, o3) = (
                draw(base, d),
                draw(base, d + 1),
                draw(base, d + 2),
                draw(base, d + 3),
            );
            out[d] = o0;
            out[d + 1] = o1;
            out[d + 2] = o2;
            out[d + 3] = o3;
            d += 4;
        }
        while d < DRAWS_PER_BLOCK {
            out[d] = draw(base, d);
            d += 1;
        }
        let cv: [u32; 8] = out[0..8].try_into().expect("draw layout");
        let message: [u32; 16] = out[8..24].try_into().expect("draw layout");
        (cv, message, u64::from(out[24]), 64, 11)
    }

    #[allow(dead_code)]
    #[repr(align(16))]
    pub struct CompressionQuadSoa {
        pub cv: [[u32; 4]; 8],
        pub message: [[u32; 4]; 16],
        pub counter_lo: [u32; 4],
        pub counter_hi: [u32; 4],
        pub block_len: [u32; 4],
        pub flags: [u32; 4],
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn gen_quad_soa(init: u64, first_block: usize) -> CompressionQuadSoa {
        let first_state =
            init.wrapping_add(((DRAWS_PER_BLOCK * first_block) as u64).wrapping_mul(GOLDEN));
        let block_stride = (DRAWS_PER_BLOCK as u64).wrapping_mul(GOLDEN);
        let mut states = [
            first_state,
            first_state.wrapping_add(block_stride),
            first_state.wrapping_add(block_stride.wrapping_mul(2)),
            first_state.wrapping_add(block_stride.wrapping_mul(3)),
        ];

        #[inline(always)]
        fn draw_row(states: &mut [u64; 4]) -> [u32; 4] {
            states[0] = states[0].wrapping_add(GOLDEN);
            states[1] = states[1].wrapping_add(GOLDEN);
            states[2] = states[2].wrapping_add(GOLDEN);
            states[3] = states[3].wrapping_add(GOLDEN);
            [
                mix(states[0]),
                mix(states[1]),
                mix(states[2]),
                mix(states[3]),
            ]
        }

        let cv = std::array::from_fn(|_| draw_row(&mut states));
        let message = std::array::from_fn(|_| draw_row(&mut states));
        let counter_lo = draw_row(&mut states);
        CompressionQuadSoa {
            cv,
            message,
            counter_lo,
            counter_hi: [0; 4],
            block_len: [64; 4],
            flags: [11; 4],
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn generate_compressions_ref(log2_size: u32, seed: u64) -> Vec<Compression> {
            let mut s = seed ^ u64::from(log2_size).rotate_left(29);
            let mut next_u32 = || {
                s = s.wrapping_add(GOLDEN);
                mix(s)
            };
            let count = 1usize << log2_size;
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let cv = std::array::from_fn(|_| next_u32());
                let message = std::array::from_fn(|_| next_u32());
                let counter = u64::from(next_u32());
                out.push((cv, message, counter, 64, 11));
            }
            out
        }

        #[test]
        fn s1_matches_reference_generator() {
            for log2_size in [8u32, 12] {
                for seed in [0u64, 1, 0x00C0_FFEE_BEEF_D15C, 0xDEAD_BEEF_CAFE_BABE] {
                    let expected = generate_compressions_ref(log2_size, seed);
                    let actual = generate_compressions_par(log2_size, seed);
                    assert_eq!(actual, expected);
                }
            }
        }

        #[test]
        fn gen_quad_soa_matches_four_reference_blocks() {
            let init = generator_init(10, 0x00C0_FFEE_BEEF_D15C);
            for first in (0..64).step_by(4) {
                let quad = gen_quad_soa(init, first);
                for lane in 0..4 {
                    let (cv, message, counter, block_len, flags) =
                        gen_block_scalar(init, first + lane);
                    for w in 0..8 {
                        assert_eq!(quad.cv[w][lane], cv[w]);
                    }
                    for w in 0..16 {
                        assert_eq!(quad.message[w][lane], message[w]);
                    }
                    assert_eq!(u64::from(quad.counter_lo[lane]), counter);
                    assert_eq!(quad.block_len[lane], block_len);
                    assert_eq!(quad.flags[lane], flags);
                }
            }
        }
    }
}

pub mod s2 {

    pub fn keepalive_start() {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        imp::start();
    }

    pub fn keepalive_stop() {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        imp::stop();
    }

    pub fn keepalive_signal() {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        imp::signal();
    }

    pub fn keepalive_join() {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        imp::join_all();
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    mod imp {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::{Duration, Instant};

        const MAX_KEEPALIVE: Duration = Duration::from_secs(2);

        static RUNNING: AtomicBool = AtomicBool::new(false);
        static LIVE: AtomicUsize = AtomicUsize::new(0);

        const QUIET_TIMEOUT: Duration = Duration::from_millis(50);

        fn spin_until_stopped(deadline: Instant) {
            crate::set_calling_thread_prover_qos();
            let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
            while RUNNING.load(Ordering::Relaxed) {
                for _ in 0..8 {
                    x = x
                        .wrapping_mul(0x2545_F491_4F6C_DD1D)
                        .wrapping_add(0x9E37_79B9_7F4A_7C15);
                    x ^= x >> 29;
                    std::hint::spin_loop();
                }
                std::hint::black_box(x);
                if Instant::now() >= deadline {
                    break;
                }
            }
            std::hint::black_box(x);
            LIVE.fetch_sub(1, Ordering::SeqCst);
        }

        pub(super) fn start() {
            if std::env::var_os("FLOCK_NO_S2").is_some() {
                return;
            }
            if RUNNING.swap(true, Ordering::SeqCst) {
                return;
            }
            let n_threads = rayon::current_num_threads();
            let n_cores = if cfg!(target_arch = "x86_64") {
                (n_threads / 2).saturating_sub(1).max(1)
            } else {
                n_threads.saturating_sub(1).max(1)
            };
            let deadline = Instant::now() + MAX_KEEPALIVE;
            for i in 0..n_cores {
                LIVE.fetch_add(1, Ordering::SeqCst);
                match std::thread::Builder::new()
                    .name(format!("flock-keepalive-{i}"))
                    .stack_size(64 * 1024)
                    .spawn(move || spin_until_stopped(deadline))
                {
                    Ok(_) => {}
                    Err(_) => {
                        LIVE.fetch_sub(1, Ordering::SeqCst);
                        break;
                    }
                }
            }
        }

        pub(super) fn stop() {
            signal();
            join_all();
        }

        pub(super) fn signal() {
            RUNNING.swap(false, Ordering::SeqCst);
        }

        pub(super) fn join_all() {
            let give_up_at = Instant::now() + QUIET_TIMEOUT;
            while LIVE.load(Ordering::SeqCst) != 0 {
                if Instant::now() >= give_up_at {
                    return;
                }
                std::hint::spin_loop();
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use std::sync::Mutex;

            static TEST_LOCK: Mutex<()> = Mutex::new(());

            #[test]
            fn start_then_stop_is_clean() {
                let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                start();
                stop();
                assert!(!RUNNING.load(Ordering::SeqCst));
                assert_eq!(LIVE.load(Ordering::SeqCst), 0);
            }

            #[test]
            fn stop_without_start_is_noop() {
                let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                stop();
                stop();
                assert!(!RUNNING.load(Ordering::SeqCst));
                assert_eq!(LIVE.load(Ordering::SeqCst), 0);
            }

            #[test]
            fn kill_switch_spawns_nothing() {
                let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                unsafe { std::env::set_var("FLOCK_NO_S2", "1") };
                start();
                let empty = LIVE.load(Ordering::SeqCst) == 0;
                let stopped = !RUNNING.load(Ordering::SeqCst);
                unsafe { std::env::remove_var("FLOCK_NO_S2") };
                stop();
                assert!(empty, "kill switch must spawn no keep-alive threads");
                assert!(stopped, "kill switch must leave the run flag clear");
            }

            #[test]
            fn restart_after_stop_is_clean() {
                let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                start();
                stop();
                start();
                stop();
                assert!(!RUNNING.load(Ordering::SeqCst));
                assert_eq!(LIVE.load(Ordering::SeqCst), 0);
            }
        }
    }
}

pub mod s3 {

    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex, OnceLock};

    use rayon::prelude::*;

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn ecore_count() -> usize {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const core::ffi::c_char,
                oldp: *mut core::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut core::ffi::c_void,
                newlen: usize,
            ) -> core::ffi::c_int;
        }
        let mut n: i32 = 0;
        let mut len = core::mem::size_of::<i32>();
        let rc = unsafe {
            sysctlbyname(
                c"hw.perflevel1.logicalcpu".as_ptr(),
                (&raw mut n).cast(),
                &raw mut len,
                core::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && len == core::mem::size_of::<i32>() && n > 0 {
            n as usize
        } else {
            0
        }
    }

    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    fn ecore_count() -> usize {
        std::env::var("FLOCK_HELPER_FORCE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    }

    #[cfg(target_os = "macos")]
    fn set_utility_qos() {
        unsafe extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        unsafe {
            let _ = pthread_set_qos_class_self_np(0x11, 0);
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn set_utility_qos() {}

    fn build_s3() -> Option<rayon::ThreadPool> {
        let n = ecore_count();
        if n == 0 {
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("flock-ecore-{i}"))
            .start_handler(|_| set_utility_qos())
            .build()
            .ok()
    }

    pub(crate) fn s3() -> Option<&'static rayon::ThreadPool> {
        static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
        POOL.get_or_init(build_s3).as_ref()
    }

    pub fn helper_pool_available() -> bool {
        s3().is_some()
    }

    pub fn helper_pool() -> Option<&'static rayon::ThreadPool> {
        s3()
    }

    const HELPER_MIN_CHUNKS: usize = 8;

    static HELPER_CHUNKS: AtomicUsize = AtomicUsize::new(0);

    pub fn helper_chunks_claimed() -> usize {
        HELPER_CHUNKS.load(Ordering::Relaxed)
    }

    static HELPER_BROADCASTS: AtomicU64 = AtomicU64::new(0);

    pub fn helper_broadcasts_issued() -> u64 {
        HELPER_BROADCASTS.load(Ordering::Relaxed)
    }

    pub(crate) fn relay_enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| !relay_killed_by(std::env::var("FLOCK_NO_HELPER_RELAY").ok().as_deref()))
    }

    fn relay_killed_by(value: Option<&str>) -> bool {
        value == Some("1")
    }

    #[derive(Clone, Copy)]
    struct JobPtr {
        data: *const (),
        call: unsafe fn(*const ()),
    }

    unsafe impl Send for JobPtr {}

    impl JobPtr {
        fn new<B: Fn() + Sync>(broadcast: &B) -> Self {
            unsafe fn thunk<B: Fn() + Sync>(data: *const ()) {
                unsafe { (*data.cast::<B>())() }
            }
            Self {
                data: (broadcast as *const B).cast::<()>(),
                call: thunk::<B>,
            }
        }
    }

    #[derive(Default)]
    struct RelayState {
        job: Option<JobPtr>,

        posted: u64,

        done: u64,

        panicked: bool,
    }

    struct Relay {
        state: Mutex<RelayState>,

        signal: Condvar,

        busy: AtomicBool,
    }

    fn relay_lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    impl Relay {
        fn new() -> Self {
            Self {
                state: Mutex::new(RelayState::default()),
                signal: Condvar::new(),
                busy: AtomicBool::new(false),
            }
        }

        fn try_acquire(&self) -> bool {
            self.busy
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        }

        unsafe fn post(&self, job: JobPtr) -> u64 {
            let seq = {
                let mut st = relay_lock(&self.state);
                st.job = Some(job);
                st.posted += 1;
                st.panicked = false;
                st.posted
            };
            self.signal.notify_all();
            seq
        }

        fn wait_done(&self, seq: u64) -> bool {
            let mut st = relay_lock(&self.state);
            while st.done < seq {
                st = self
                    .signal
                    .wait(st)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            st.panicked
        }

        fn run(&self) -> ! {
            loop {
                let job = {
                    let mut st = relay_lock(&self.state);
                    loop {
                        if let Some(job) = st.job.take() {
                            break job;
                        }
                        st = self
                            .signal
                            .wait(st)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                };

                let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    (job.call)(job.data);
                }))
                .is_ok();
                {
                    let mut st = relay_lock(&self.state);
                    st.done += 1;
                    st.panicked = !ok;
                }
                self.signal.notify_all();
            }
        }
    }

    fn relay() -> Option<&'static Relay> {
        static RELAY: OnceLock<Option<&'static Relay>> = OnceLock::new();
        *RELAY.get_or_init(|| {
            let relay: &'static Relay = Box::leak(Box::new(Relay::new()));

            std::thread::Builder::new()
                .name("flock-s3-relay".to_string())
                .spawn(move || relay.run())
                .ok()
                .map(|_handle| relay)
        })
    }

    struct PostedJob<'a> {
        relay: &'a Relay,

        seq: AtomicU64,
    }

    impl Drop for PostedJob<'_> {
        fn drop(&mut self) {
            let seq = self.seq.load(Ordering::Relaxed);
            let panicked = seq != 0 && self.relay.wait_done(seq);
            self.relay.busy.store(false, Ordering::Release);
            if panicked && !std::thread::panicking() {
                panic!("s3: helper-pool broadcast panicked");
            }
        }
    }

    pub(crate) fn drain_hetero<B>(
        main_threads: usize,
        worker: &(dyn Fn() + Sync),
        broadcast: &B,
        use_relay: bool,
    ) where
        B: Fn() + Sync,
    {
        if use_relay
            && let Some(relay) = relay()
            && relay.try_acquire()
        {
            let job = PostedJob {
                relay,
                seq: AtomicU64::new(0),
            };
            let posted = AtomicBool::new(false);
            (0..main_threads)
                .into_par_iter()
                .with_max_len(1)
                .for_each(|_| {
                    if !posted.load(Ordering::Relaxed) && !posted.swap(true, Ordering::Relaxed) {
                        let seq = unsafe { relay.post(JobPtr::new(broadcast)) };
                        job.seq.store(seq, Ordering::Relaxed);
                    }
                    worker();
                });

            drop(job);
            return;
        }

        std::thread::scope(|s| {
            s.spawn(broadcast);
            (0..main_threads)
                .into_par_iter()
                .with_max_len(1)
                .for_each(|_| worker());
        });
    }

    pub fn run_hetero_chunks<F>(n_chunks: usize, f: F)
    where
        F: Fn(usize) + Sync,
    {
        run_chunks_with_helper(n_chunks, &f, s3());
    }

    pub fn run_helper_only_chunks<F>(n_chunks: usize, max_workers: usize, f: &F) -> bool
    where
        F: Fn(usize) + Sync + ?Sized,
    {
        run_chunks_with_helper_only(n_chunks, max_workers, f, s3())
    }

    #[doc(hidden)]
    pub fn run_chunks_with_helper_only<F>(
        n_chunks: usize,
        max_workers: usize,
        f: &F,
        helper: Option<&rayon::ThreadPool>,
    ) -> bool
    where
        F: Fn(usize) + Sync + ?Sized,
    {
        let Some(ep) = helper else {
            return false;
        };
        let max_workers = max_workers.min(ep.current_num_threads());
        if max_workers == 0 {
            return false;
        }
        if n_chunks == 0 {
            return true;
        }

        let next = AtomicUsize::new(0);
        ep.broadcast(|context| {
            if context.index() >= max_workers {
                return;
            }
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n_chunks {
                    break;
                }
                HELPER_CHUNKS.fetch_add(1, Ordering::Relaxed);
                f(i);
            }
        });
        true
    }

    #[allow(dead_code)]
    pub(crate) fn run_hetero_chunks_stateful<S, I, F>(n_chunks: usize, init: I, f: F)
    where
        I: Fn() -> S + Sync,
        F: Fn(&mut S, usize) + Sync,
    {
        run_chunks_with_helper_stateful(n_chunks, &init, &f, s3());
    }

    pub fn run_chunks_with_helper<F>(n_chunks: usize, f: &F, helper: Option<&rayon::ThreadPool>)
    where
        F: Fn(usize) + Sync,
    {
        run_chunks_with_helper_relay(n_chunks, f, helper, relay_enabled());
    }

    fn run_chunks_with_helper_relay<F>(
        n_chunks: usize,
        f: &F,
        helper: Option<&rayon::ThreadPool>,
        use_relay: bool,
    ) where
        F: Fn(usize) + Sync,
    {
        if n_chunks == 0 {
            return;
        }
        let main_threads = rayon::current_num_threads();
        if main_threads <= 1 {
            for i in 0..n_chunks {
                f(i);
            }
            return;
        }
        let next = AtomicUsize::new(0);
        let worker = || {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n_chunks {
                    break;
                }
                f(i);
            }
        };

        let drain_main = || {
            (0..main_threads)
                .into_par_iter()
                .with_max_len(1)
                .for_each(|_| worker());
        };
        match helper.filter(|_| n_chunks >= HELPER_MIN_CHUNKS) {
            Some(ep) => {
                HELPER_BROADCASTS.fetch_add(1, Ordering::Relaxed);
                let broadcast = || {
                    ep.broadcast(|_| {
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= n_chunks {
                                break;
                            }
                            HELPER_CHUNKS.fetch_add(1, Ordering::Relaxed);
                            f(i);
                        }
                    });
                };
                drain_hetero(main_threads, &worker, &broadcast, use_relay);
            }
            None => drain_main(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn run_chunks_with_helper_stateful<S, I, F>(
        n_chunks: usize,
        init: &I,
        f: &F,
        helper: Option<&rayon::ThreadPool>,
    ) where
        I: Fn() -> S + Sync,
        F: Fn(&mut S, usize) + Sync,
    {
        run_chunks_with_helper_stateful_relay(n_chunks, init, f, helper, relay_enabled());
    }

    #[allow(dead_code)]
    fn run_chunks_with_helper_stateful_relay<S, I, F>(
        n_chunks: usize,
        init: &I,
        f: &F,
        helper: Option<&rayon::ThreadPool>,
        use_relay: bool,
    ) where
        I: Fn() -> S + Sync,
        F: Fn(&mut S, usize) + Sync,
    {
        if n_chunks == 0 {
            return;
        }
        let main_threads = rayon::current_num_threads();
        if main_threads <= 1 {
            let mut state = init();
            for i in 0..n_chunks {
                f(&mut state, i);
            }
            return;
        }
        let next = AtomicUsize::new(0);
        let worker = || {
            let mut state = init();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n_chunks {
                    break;
                }
                f(&mut state, i);
            }
        };
        let drain_main = || {
            (0..main_threads)
                .into_par_iter()
                .with_max_len(1)
                .for_each(|_| worker());
        };
        match helper.filter(|_| n_chunks >= HELPER_MIN_CHUNKS) {
            Some(ep) => {
                HELPER_BROADCASTS.fetch_add(1, Ordering::Relaxed);
                let broadcast = || {
                    ep.broadcast(|_| {
                        let mut state = init();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= n_chunks {
                                break;
                            }
                            HELPER_CHUNKS.fetch_add(1, Ordering::Relaxed);
                            f(&mut state, i);
                        }
                    });
                };
                drain_hetero(main_threads, &worker, &broadcast, use_relay);
            }
            None => drain_main(),
        }
    }

    #[derive(Clone, Copy)]
    pub struct SyncPtr<T>(pub *mut T);

    impl<T> SyncPtr<T> {
        #[inline]
        pub fn ptr(self) -> *mut T {
            self.0
        }
    }

    unsafe impl<T> Send for SyncPtr<T> {}
    unsafe impl<T> Sync for SyncPtr<T> {}

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn helper_queue_runs_each_chunk_exactly_once() {
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let n = 1000;
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            run_chunks_with_helper(
                n,
                &|i| {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                },
                Some(&helper),
            );
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        }

        #[test]
        fn small_jobs_skip_helper_but_complete() {
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let n = HELPER_MIN_CHUNKS - 1;
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            run_chunks_with_helper(
                n,
                &|i| {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                },
                Some(&helper),
            );
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        }

        #[test]
        fn single_threaded_pool_runs_inline_in_order() {
            let single = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            single.install(|| {
                let seen = std::sync::Mutex::new(Vec::new());
                run_chunks_with_helper(100, &|i| seen.lock().unwrap().push(i), Some(&helper));
                let seen = seen.into_inner().unwrap();
                assert_eq!(seen, (0..100).collect::<Vec<_>>());
            });
        }

        #[test]
        fn zero_chunks_is_noop() {
            run_chunks_with_helper(0, &|_| panic!("must not run"), None);
        }

        #[test]
        fn helper_only_queue_runs_each_chunk_exactly_once() {
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .thread_name(|i| format!("helper-only-test-{i}"))
                .build()
                .unwrap();
            let n = 513;
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let worker_mask = AtomicUsize::new(0);
            assert!(run_chunks_with_helper_only(
                n,
                2,
                &|i| {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                    let worker = rayon::current_thread_index().expect("helper worker");
                    worker_mask.fetch_or(1usize << worker, Ordering::Relaxed);
                },
                Some(&helper),
            ));
            assert!(
                counts
                    .iter()
                    .all(|count| count.load(Ordering::Relaxed) == 1)
            );
            assert_eq!(worker_mask.load(Ordering::Relaxed) & !0b11, 0);

            let untouched = AtomicUsize::new(0);
            assert!(!run_chunks_with_helper_only(
                1,
                2,
                &|_| {
                    untouched.fetch_add(1, Ordering::Relaxed);
                },
                None,
            ));
            assert_eq!(untouched.load(Ordering::Relaxed), 0);
        }

        #[test]
        fn stateful_helper_queue_reuses_worker_state() {
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let n = 1000;
            let next_state = AtomicUsize::new(0);
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let state_for_chunk: Vec<AtomicUsize> =
                (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect();
            run_chunks_with_helper_stateful(
                n,
                &|| next_state.fetch_add(1, Ordering::Relaxed),
                &|state, i| {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                    state_for_chunk[i].store(*state, Ordering::Relaxed);
                },
                Some(&helper),
            );
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
            let n_states = next_state.load(Ordering::Relaxed);
            assert!(n_states < n, "state must be per worker, not per chunk");
            let mut uses = vec![0usize; n_states];
            for state in state_for_chunk {
                uses[state.load(Ordering::Relaxed)] += 1;
            }
            assert!(uses.into_iter().any(|n_uses| n_uses > 1));
        }

        #[test]
        fn both_kickoff_arms_run_each_chunk_exactly_once() {
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let n = 997;
            for use_relay in [true, false] {
                for _ in 0..16 {
                    let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                    run_chunks_with_helper_relay(
                        n,
                        &|i| {
                            counts[i].fetch_add(1, Ordering::Relaxed);
                        },
                        Some(&helper),
                        use_relay,
                    );
                    assert!(
                        counts.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                        "use_relay={use_relay}"
                    );
                }
            }
        }

        #[test]
        fn both_kickoff_arms_reuse_stateful_worker_state() {
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let n = 1000;
            for use_relay in [true, false] {
                let next_state = AtomicUsize::new(0);
                let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                run_chunks_with_helper_stateful_relay(
                    n,
                    &|| next_state.fetch_add(1, Ordering::Relaxed),
                    &|_state, i| {
                        counts[i].fetch_add(1, Ordering::Relaxed);
                    },
                    Some(&helper),
                    use_relay,
                );
                assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
                assert!(
                    next_state.load(Ordering::Relaxed) < n,
                    "state must be per worker, not per chunk (use_relay={use_relay})"
                );
            }
        }

        #[test]
        fn concurrent_relayed_drains_all_complete() {
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let n = 512;
            let counts: Vec<Vec<AtomicUsize>> = (0..4)
                .map(|_| (0..n).map(|_| AtomicUsize::new(0)).collect())
                .collect();
            std::thread::scope(|s| {
                for lane in &counts {
                    s.spawn(|| {
                        for _ in 0..8 {
                            run_chunks_with_helper_relay(
                                n,
                                &|i| {
                                    lane[i].fetch_add(1, Ordering::Relaxed);
                                },
                                Some(&helper),
                                true,
                            );
                        }
                    });
                }
            });
            for lane in &counts {
                assert!(lane.iter().all(|c| c.load(Ordering::Relaxed) == 8));
            }
        }

        #[test]
        fn panicking_chunk_propagates_and_leaves_the_relay_usable() {
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            for use_relay in [true, false] {
                let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_chunks_with_helper_relay(
                        64,
                        &|_| panic!("chunk exploded"),
                        Some(&helper),
                        use_relay,
                    );
                }));
                assert!(caught.is_err(), "use_relay={use_relay}");
            }

            let n = 128;
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            run_chunks_with_helper_relay(
                n,
                &|i| {
                    counts[i].fetch_add(1, Ordering::Relaxed);
                },
                Some(&helper),
                true,
            );
            assert!(counts.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        }

        #[test]
        fn engaged_drains_count_broadcasts() {
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            let drains = 12;
            let before = helper_broadcasts_issued();
            for _ in 0..drains {
                run_chunks_with_helper(HELPER_MIN_CHUNKS, &|_| {}, Some(&helper));
            }
            assert!(helper_broadcasts_issued() - before >= drains);

            let after_engaged = helper_broadcasts_issued();
            run_chunks_with_helper(1024, &|_| {}, None);
            assert!(helper_broadcasts_issued() >= after_engaged);
        }

        #[test]
        fn relay_kill_switch_matches_exactly_one() {
            assert!(relay_killed_by(Some("1")));
            for keep in [
                None,
                Some(""),
                Some("0"),
                Some("true"),
                Some("11"),
                Some(" 1"),
            ] {
                assert!(!relay_killed_by(keep), "{keep:?} must not kill the relay");
            }
        }

        #[test]
        fn stateful_single_threaded_pool_uses_one_state_in_order() {
            let single = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let helper = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap();
            single.install(|| {
                let init_count = AtomicUsize::new(0);
                let seen = std::sync::Mutex::new(Vec::new());
                run_chunks_with_helper_stateful(
                    100,
                    &|| {
                        init_count.fetch_add(1, Ordering::Relaxed);
                        0usize
                    },
                    &|state, i| {
                        *state += 1;
                        seen.lock().unwrap().push(i);
                    },
                    Some(&helper),
                );
                assert_eq!(init_count.load(Ordering::Relaxed), 1);
                assert_eq!(seen.into_inner().unwrap(), (0..100).collect::<Vec<_>>());
            });
        }
    }
}

#[cfg(test)]
mod k0_asm_tests {
    #[test]
    fn carryless_register_lowering_matches_polynomial_product() {
        let mut state = 0x9e3779b97f4a7c15u64;
        for i in 0..4096 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let a = state;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let b = if i < 64 { 1u64 << i } else { state };
            let mut expected = 0u128;
            for bit in 0..64 {
                if (b >> bit) & 1 != 0 {
                    expected ^= (a as u128) << bit;
                }
            }
            let actual = unsafe {
                (super::k0_clmul64_lo(a, b) as u128) | ((super::k0_clmul64_hi(a, b) as u128) << 64)
            };
            assert_eq!(actual, expected);
        }
    }
}

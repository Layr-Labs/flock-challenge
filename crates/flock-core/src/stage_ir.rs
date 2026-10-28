#![allow(
    unused_unsafe,
    dead_code,
    unused_imports,
    unused_variables,
    unused_parens,
    unused_assignments,
    unreachable_code,
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
        return _mm_extract_epi64::<0>(_mm_clmulepi64_si128::<0x00>(va, vb)) as u64;
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
        return _mm_extract_epi64::<1>(_mm_clmulepi64_si128::<0x00>(va, vb)) as u64;
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
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_xor3_m128i(
    a: core::arch::x86_64::__m128i,
    b: core::arch::x86_64::__m128i,
    c: core::arch::x86_64::__m128i,
) -> core::arch::x86_64::__m128i {
    #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
    unsafe {
        return core::arch::x86_64::_mm_ternarylogic_epi64::<0x96>(a, b, c);
    }
    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512vl")))]
    unsafe {
        core::arch::x86_64::_mm_xor_si128(core::arch::x86_64::_mm_xor_si128(a, b), c)
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
pub unsafe fn k0_xor3_neon(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    c: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    #[cfg(target_feature = "sha3")]
    unsafe {
        return core::arch::aarch64::veor3q_u64(a, b, c);
    }
    #[cfg(not(target_feature = "sha3"))]
    unsafe {
        core::arch::aarch64::veorq_u64(core::arch::aarch64::veorq_u64(a, b), c)
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
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
        let a_lo = vgetq_lane_u64::<0>(va);
        let b_lo = vgetq_lane_u64::<0>(vb);
        let a_sum = a_lo ^ vgetq_lane_u64::<1>(va);
        let b_sum = b_lo ^ vgetq_lane_u64::<1>(vb);
        (
            vreinterpretq_u64_p128(vmull_p64(a_lo, b_lo)),
            vreinterpretq_u64_p128(vmull_p64(a_sum, b_sum)),
            vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(va),
                vreinterpretq_p64_u64(vb),
            )),
        )
    }
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
        let cross = k0_xor3_neon(mid, t0, t2);
        let t1 = k0_xor3_neon(
            cross,
            vextq_u64::<1>(z, t2),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t2), poly)),
        );
        let res = k0_xor3_neon(
            t0,
            vextq_u64::<1>(z, t1),
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
        let c0 = k0_xor3_neon(m0, p0_00, p0_11);
        let c1 = k0_xor3_neon(m1, p1_00, p1_11);
        let t1_0 = k0_xor3_neon(
            c0,
            vextq_u64::<1>(z, p0_11),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p0_11), poly)),
        );
        let t1_1 = k0_xor3_neon(
            c1,
            vextq_u64::<1>(z, p1_11),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p1_11), poly)),
        );
        let res0 = k0_xor3_neon(
            p0_00,
            vextq_u64::<1>(z, t1_0),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1_0), poly)),
        );
        let res1 = k0_xor3_neon(
            p1_00,
            vextq_u64::<1>(z, t1_1),
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
        let cross = k0_xor3_m128i(mid, t0, t2);
        let t1 = k0_xor3_m128i(
            cross,
            _mm_bslli_si128::<8>(t2),
            _mm_clmulepi64_si128::<0x01>(t2, poly),
        );
        k0_xor3_m128i(
            t0,
            _mm_bslli_si128::<8>(t1),
            _mm_clmulepi64_si128::<0x01>(t1, poly),
        )
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
        let c0 = k0_xor3_m128i(m0, p0_00, p0_11);
        let c1 = k0_xor3_m128i(m1, p1_00, p1_11);
        let t1_0 = k0_xor3_m128i(
            c0,
            _mm_bslli_si128::<8>(p0_11),
            _mm_clmulepi64_si128::<0x01>(p0_11, poly),
        );
        let t1_1 = k0_xor3_m128i(
            c1,
            _mm_bslli_si128::<8>(p1_11),
            _mm_clmulepi64_si128::<0x01>(p1_11, poly),
        );
        let res0 = k0_xor3_m128i(
            p0_00,
            _mm_bslli_si128::<8>(t1_0),
            _mm_clmulepi64_si128::<0x01>(t1_0, poly),
        );
        let res1 = k0_xor3_m128i(
            p1_00,
            _mm_bslli_si128::<8>(t1_1),
            _mm_clmulepi64_si128::<0x01>(t1_1, poly),
        );
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
        k0_xor3_m128i(l, _mm_bslli_si128::<8>(h), red)
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
            k0_xor3_m128i(l0, _mm_bslli_si128::<8>(h0), red0),
            k0_xor3_m128i(l1, _mm_bslli_si128::<8>(h1), red1),
        )
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_low_const_m128i(
    vx: core::arch::x86_64::__m128i,
    vtw_lo: core::arch::x86_64::__m128i,
) -> core::arch::x86_64::__m128i {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm_set1_epi64x(0x87);
        let p0 = _mm_clmulepi64_si128::<0x00>(vx, vtw_lo);
        let p1 = _mm_clmulepi64_si128::<0x01>(vx, vtw_lo);
        let red = _mm_clmulepi64_si128::<0x01>(p1, poly);
        k0_xor3_m128i(p0, _mm_bslli_si128::<8>(p1), red)
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_low_const_vec2_m128i(
    vx0: core::arch::x86_64::__m128i,
    vx1: core::arch::x86_64::__m128i,
    vtw0: core::arch::x86_64::__m128i,
    vtw1: core::arch::x86_64::__m128i,
) -> (core::arch::x86_64::__m128i, core::arch::x86_64::__m128i) {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm_set1_epi64x(0x87);
        let p0_0 = _mm_clmulepi64_si128::<0x00>(vx0, vtw0);
        let p1_0 = _mm_clmulepi64_si128::<0x00>(vx1, vtw1);
        let p0_1 = _mm_clmulepi64_si128::<0x01>(vx0, vtw0);
        let p1_1 = _mm_clmulepi64_si128::<0x01>(vx1, vtw1);
        let red0 = _mm_clmulepi64_si128::<0x01>(p0_1, poly);
        let red1 = _mm_clmulepi64_si128::<0x01>(p1_1, poly);
        (
            k0_xor3_m128i(p0_0, _mm_bslli_si128::<8>(p0_1), red0),
            k0_xor3_m128i(p1_0, _mm_bslli_si128::<8>(p1_1), red1),
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
pub unsafe fn k0_f128_mul_acc2_m128i(
    va0: core::arch::x86_64::__m128i,
    vb0: core::arch::x86_64::__m128i,
    va1: core::arch::x86_64::__m128i,
    vb1: core::arch::x86_64::__m128i,
    ll: &mut core::arch::x86_64::__m128i,
    mid: &mut core::arch::x86_64::__m128i,
    hh: &mut core::arch::x86_64::__m128i,
) {
    unsafe {
        use core::arch::x86_64::*;
        let p0_ll = _mm_clmulepi64_si128::<0x00>(va0, vb0);
        let p1_ll = _mm_clmulepi64_si128::<0x00>(va1, vb1);
        *ll = k0_xor3_m128i(*ll, p0_ll, p1_ll);
        let p0_hh = _mm_clmulepi64_si128::<0x11>(va0, vb0);
        let p1_hh = _mm_clmulepi64_si128::<0x11>(va1, vb1);
        *hh = k0_xor3_m128i(*hh, p0_hh, p1_hh);
        let a0_sum = _mm_xor_si128(va0, _mm_shuffle_epi32::<0x4e>(va0));
        let b0_sum = _mm_xor_si128(vb0, _mm_shuffle_epi32::<0x4e>(vb0));
        let a1_sum = _mm_xor_si128(va1, _mm_shuffle_epi32::<0x4e>(va1));
        let b1_sum = _mm_xor_si128(vb1, _mm_shuffle_epi32::<0x4e>(vb1));
        let p0_mid = _mm_clmulepi64_si128::<0x00>(a0_sum, b0_sum);
        let p1_mid = _mm_clmulepi64_si128::<0x00>(a1_sum, b1_sum);
        *mid = k0_xor3_m128i(*mid, p0_mid, p1_mid);
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
        let cross = k0_xor3_m128i(mid, ll, hh);
        let t1 = k0_xor3_m128i(
            cross,
            _mm_bslli_si128::<8>(hh),
            _mm_clmulepi64_si128::<0x01>(hh, poly),
        );
        k0_xor3_m128i(
            ll,
            _mm_bslli_si128::<8>(t1),
            _mm_clmulepi64_si128::<0x01>(t1, poly),
        )
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_xor3_zmm(
    a: core::arch::x86_64::__m512i,
    b: core::arch::x86_64::__m512i,
    c: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    unsafe { core::arch::x86_64::_mm512_ternarylogic_epi64::<0x96>(a, b, c) }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_pre_reduce_const_zmm(
    vr: core::arch::x86_64::__m512i,
) -> (core::arch::x86_64::__m512i, core::arch::x86_64::__m512i) {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm512_set1_epi64(0x87);
        let p = _mm512_clmulepi64_epi128::<0x01>(vr, poly);
        let vr0 = _mm512_unpacklo_epi64(vr, p);
        let p_xor = _mm512_xor_si512(p, _mm512_shuffle_epi32::<0x44>(vr));
        let vr1 = _mm512_unpackhi_epi64(vr, p_xor);
        (vr0, vr1)
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_mul_const_zmm(
    vx: core::arch::x86_64::__m512i,
    vr0: core::arch::x86_64::__m512i,
    vr1: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm512_set1_epi64(0x87);
        let l0 = _mm512_clmulepi64_epi128::<0x00>(vx, vr0);
        let l1 = _mm512_clmulepi64_epi128::<0x11>(vx, vr0);
        let h0 = _mm512_clmulepi64_epi128::<0x00>(vx, vr1);
        let h1 = _mm512_clmulepi64_epi128::<0x11>(vx, vr1);
        let h = _mm512_xor_si512(h0, h1);
        let red = _mm512_clmulepi64_epi128::<0x01>(h, poly);
        let lo = k0_xor3_zmm(l0, l1, red);
        _mm512_xor_si512(lo, _mm512_bslli_epi128::<8>(h))
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_mul_low_const_zmm(
    vx: core::arch::x86_64::__m512i,
    vtw_lo: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm512_set1_epi64(0x87);
        let p0 = _mm512_clmulepi64_epi128::<0x00>(vx, vtw_lo);
        let p1 = _mm512_clmulepi64_epi128::<0x01>(vx, vtw_lo);
        let red = _mm512_clmulepi64_epi128::<0x01>(p1, poly);
        k0_xor3_zmm(p0, _mm512_bslli_epi128::<8>(p1), red)
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_mul_zmm(
    va: core::arch::x86_64::__m512i,
    vb: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm512_set1_epi64(0x87);
        let t0 = _mm512_clmulepi64_epi128::<0x00>(va, vb);
        let t2 = _mm512_clmulepi64_epi128::<0x11>(va, vb);
        let a_sum = _mm512_xor_si512(va, _mm512_shuffle_epi32::<0x4e>(va));
        let b_sum = _mm512_xor_si512(vb, _mm512_shuffle_epi32::<0x4e>(vb));
        let mid = _mm512_clmulepi64_epi128::<0x00>(a_sum, b_sum);
        let cross = k0_xor3_zmm(mid, t0, t2);
        let t1 = k0_xor3_zmm(
            cross,
            _mm512_bslli_epi128::<8>(t2),
            _mm512_clmulepi64_epi128::<0x01>(t2, poly),
        );
        k0_xor3_zmm(
            t0,
            _mm512_bslli_epi128::<8>(t1),
            _mm512_clmulepi64_epi128::<0x01>(t1, poly),
        )
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_mul_acc_zmm(
    va: core::arch::x86_64::__m512i,
    vb: core::arch::x86_64::__m512i,
    ll: &mut core::arch::x86_64::__m512i,
    mid: &mut core::arch::x86_64::__m512i,
    hh: &mut core::arch::x86_64::__m512i,
) {
    unsafe {
        use core::arch::x86_64::*;
        *ll = _mm512_xor_si512(*ll, _mm512_clmulepi64_epi128::<0x00>(va, vb));
        *hh = _mm512_xor_si512(*hh, _mm512_clmulepi64_epi128::<0x11>(va, vb));
        let a_sum = _mm512_xor_si512(va, _mm512_shuffle_epi32::<0x4e>(va));
        let b_sum = _mm512_xor_si512(vb, _mm512_shuffle_epi32::<0x4e>(vb));
        *mid = _mm512_xor_si512(*mid, _mm512_clmulepi64_epi128::<0x00>(a_sum, b_sum));
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_mul_acc2_zmm(
    va0: core::arch::x86_64::__m512i,
    vb0: core::arch::x86_64::__m512i,
    va1: core::arch::x86_64::__m512i,
    vb1: core::arch::x86_64::__m512i,
    ll: &mut core::arch::x86_64::__m512i,
    mid: &mut core::arch::x86_64::__m512i,
    hh: &mut core::arch::x86_64::__m512i,
) {
    unsafe {
        use core::arch::x86_64::*;
        let p0_ll = _mm512_clmulepi64_epi128::<0x00>(va0, vb0);
        let p1_ll = _mm512_clmulepi64_epi128::<0x00>(va1, vb1);
        *ll = k0_xor3_zmm(*ll, p0_ll, p1_ll);
        let p0_hh = _mm512_clmulepi64_epi128::<0x11>(va0, vb0);
        let p1_hh = _mm512_clmulepi64_epi128::<0x11>(va1, vb1);
        *hh = k0_xor3_zmm(*hh, p0_hh, p1_hh);
        let a0_sum = _mm512_xor_si512(va0, _mm512_shuffle_epi32::<0x4e>(va0));
        let b0_sum = _mm512_xor_si512(vb0, _mm512_shuffle_epi32::<0x4e>(vb0));
        let a1_sum = _mm512_xor_si512(va1, _mm512_shuffle_epi32::<0x4e>(va1));
        let b1_sum = _mm512_xor_si512(vb1, _mm512_shuffle_epi32::<0x4e>(vb1));
        let p0_mid = _mm512_clmulepi64_epi128::<0x00>(a0_sum, b0_sum);
        let p1_mid = _mm512_clmulepi64_epi128::<0x00>(a1_sum, b1_sum);
        *mid = k0_xor3_zmm(*mid, p0_mid, p1_mid);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_hsum_zmm(v: core::arch::x86_64::__m512i) -> core::arch::x86_64::__m128i {
    unsafe {
        use core::arch::x86_64::*;
        let lo256 = _mm512_castsi512_si256(v);
        let hi256 = _mm512_extracti64x4_epi64::<1>(v);
        let sum256 = _mm256_xor_si256(lo256, hi256);
        let lo128 = _mm256_castsi256_si128(sum256);
        let hi128 = _mm256_extracti128_si256::<1>(sum256);
        _mm_xor_si128(lo128, hi128)
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_reduce_3lane_zmm(
    ll: core::arch::x86_64::__m512i,
    mid: core::arch::x86_64::__m512i,
    hh: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m128i {
    unsafe {
        use core::arch::x86_64::*;
        let ll_128 = k0_f128_hsum_zmm(ll);
        let mid_128 = k0_f128_hsum_zmm(mid);
        let hh_128 = k0_f128_hsum_zmm(hh);
        let poly = _mm_set1_epi64x(0x87);
        let cross = _mm_xor_si128(mid_128, _mm_xor_si128(ll_128, hh_128));
        let t1 = _mm_xor_si128(
            cross,
            _mm_xor_si128(
                _mm_bslli_si128::<8>(hh_128),
                _mm_clmulepi64_si128::<0x01>(hh_128, poly),
            ),
        );
        _mm_xor_si128(
            ll_128,
            _mm_xor_si128(
                _mm_bslli_si128::<8>(t1),
                _mm_clmulepi64_si128::<0x01>(t1, poly),
            ),
        )
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_reduce_lanes_zmm(
    ll: core::arch::x86_64::__m512i,
    mid: core::arch::x86_64::__m512i,
    hh: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    unsafe {
        use core::arch::x86_64::*;
        let poly = _mm512_set1_epi64(0x87);
        let cross = k0_xor3_zmm(mid, ll, hh);
        let t1 = k0_xor3_zmm(
            cross,
            _mm512_bslli_epi128::<8>(hh),
            _mm512_clmulepi64_epi128::<0x01>(hh, poly),
        );
        k0_xor3_zmm(
            ll,
            _mm512_bslli_epi128::<8>(t1),
            _mm512_clmulepi64_epi128::<0x01>(t1, poly),
        )
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_fold_even_odd_zmm(
    v0: core::arch::x86_64::__m512i,
    v1: core::arch::x86_64::__m512i,
    vr0: core::arch::x86_64::__m512i,
    vr1: core::arch::x86_64::__m512i,
) -> core::arch::x86_64::__m512i {
    unsafe {
        use core::arch::x86_64::*;
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let fe = _mm512_permutex2var_epi64(v0, even_idx, v1);
        let fo = _mm512_permutex2var_epi64(v0, odd_idx, v1);
        let fp = k0_f128_mul_const_zmm(_mm512_xor_si512(fe, fo), vr0, vr1);
        _mm512_xor_si512(fe, fp)
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
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
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
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
        k0_xor3_neon(l, vextq_u64::<1>(z, h), red)
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
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
            k0_xor3_neon(l0, vextq_u64::<1>(z, h0), red0),
            k0_xor3_neon(l1, vextq_u64::<1>(z, h1), red1),
        )
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_low_const_neon(
    vx: core::arch::aarch64::uint64x2_t,
    tw_lo: u64,
) -> core::arch::aarch64::uint64x2_t {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let p0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(vx), tw_lo));
        let p1 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<1>(vx), tw_lo));
        let red = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p1), poly));
        k0_xor3_neon(p0, vextq_u64::<1>(z, p1), red)
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_low_const_vec2_neon(
    vx0: core::arch::aarch64::uint64x2_t,
    vx1: core::arch::aarch64::uint64x2_t,
    tw0_lo: u64,
    tw1_lo: u64,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let p0_0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(vx0), tw0_lo));
        let p1_0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(vx1), tw1_lo));
        let p0_1 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<1>(vx0), tw0_lo));
        let p1_1 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<1>(vx1), tw1_lo));
        let red0 = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p0_1), poly));
        let red1 = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p1_1), poly));
        (
            k0_xor3_neon(p0_0, vextq_u64::<1>(z, p0_1), red0),
            k0_xor3_neon(p1_0, vextq_u64::<1>(z, p1_1), red1),
        )
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_neon(
    va: core::arch::aarch64::uint64x2_t,
    vb: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let (t0, mid, t2) = k0_f128_mul_unreduced_neon(va, vb);
        let cross = k0_xor3_neon(mid, t0, t2);
        let t1 = k0_xor3_neon(
            cross,
            vextq_u64::<1>(z, t2),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t2), poly)),
        );
        k0_xor3_neon(
            t0,
            vextq_u64::<1>(z, t1),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1), poly)),
        )
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
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
        let va0_sum = veorq_u64(va0, vextq_u64::<1>(va0, va0));
        let vb0_sum = veorq_u64(vb0, vextq_u64::<1>(vb0, vb0));
        let va1_sum = veorq_u64(va1, vextq_u64::<1>(va1, va1));
        let vb1_sum = veorq_u64(vb1, vextq_u64::<1>(vb1, vb1));
        let p0_00 = vreinterpretq_u64_p128(vmull_p64(
            vgetq_lane_u64::<0>(va0),
            vgetq_lane_u64::<0>(vb0),
        ));
        let p1_00 = vreinterpretq_u64_p128(vmull_p64(
            vgetq_lane_u64::<0>(va1),
            vgetq_lane_u64::<0>(vb1),
        ));
        let m0 = vreinterpretq_u64_p128(vmull_p64(
            vgetq_lane_u64::<0>(va0_sum),
            vgetq_lane_u64::<0>(vb0_sum),
        ));
        let m1 = vreinterpretq_u64_p128(vmull_p64(
            vgetq_lane_u64::<0>(va1_sum),
            vgetq_lane_u64::<0>(vb1_sum),
        ));
        let p0_11 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(va0),
            vreinterpretq_p64_u64(vb0),
        ));
        let p1_11 = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(va1),
            vreinterpretq_p64_u64(vb1),
        ));
        let c0 = k0_xor3_neon(m0, p0_00, p0_11);
        let c1 = k0_xor3_neon(m1, p1_00, p1_11);
        let t1_0 = k0_xor3_neon(
            c0,
            vextq_u64::<1>(z, p0_11),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p0_11), poly)),
        );
        let t1_1 = k0_xor3_neon(
            c1,
            vextq_u64::<1>(z, p1_11),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p1_11), poly)),
        );
        let res0 = k0_xor3_neon(
            p0_00,
            vextq_u64::<1>(z, t1_0),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1_0), poly)),
        );
        let res1 = k0_xor3_neon(
            p1_00,
            vextq_u64::<1>(z, t1_1),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1_1), poly)),
        );
        (res0, res1)
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_acc_neon(
    va: core::arch::aarch64::uint64x2_t,
    vb: core::arch::aarch64::uint64x2_t,
    ll: &mut core::arch::aarch64::uint64x2_t,
    mid: &mut core::arch::aarch64::uint64x2_t,
    hh: &mut core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let (p_ll, p_mid, p_hh) = k0_f128_mul_unreduced_neon(va, vb);
        *ll = veorq_u64(*ll, p_ll);
        *mid = veorq_u64(*mid, p_mid);
        *hh = veorq_u64(*hh, p_hh);
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
pub unsafe fn k0_f128_mul_acc2_neon(
    va0: core::arch::aarch64::uint64x2_t,
    vb0: core::arch::aarch64::uint64x2_t,
    va1: core::arch::aarch64::uint64x2_t,
    vb1: core::arch::aarch64::uint64x2_t,
    ll: &mut core::arch::aarch64::uint64x2_t,
    mid: &mut core::arch::aarch64::uint64x2_t,
    hh: &mut core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        let (p0_ll, p0_mid, p0_hh) = k0_f128_mul_unreduced_neon(va0, vb0);
        let (p1_ll, p1_mid, p1_hh) = k0_f128_mul_unreduced_neon(va1, vb1);
        *ll = k0_xor3_neon(*ll, p0_ll, p1_ll);
        *mid = k0_xor3_neon(*mid, p0_mid, p1_mid);
        *hh = k0_xor3_neon(*hh, p0_hh, p1_hh);
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
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
        let veq_sum = veorq_u64(veq, vextq_u64::<1>(veq, veq));
        let vb0_sum = veorq_u64(vb0, vextq_u64::<1>(vb0, vb0));
        let vb1_sum = veorq_u64(vb1, vextq_u64::<1>(vb1, vb1));
        let eq_sum = vgetq_lane_u64::<0>(veq_sum);
        *ll0 = veorq_u64(
            *ll0,
            vreinterpretq_u64_p128(vmull_p64(eq_lo, vgetq_lane_u64::<0>(vb0))),
        );
        *ll1 = veorq_u64(
            *ll1,
            vreinterpretq_u64_p128(vmull_p64(eq_lo, vgetq_lane_u64::<0>(vb1))),
        );
        *mid0 = veorq_u64(
            *mid0,
            vreinterpretq_u64_p128(vmull_p64(eq_sum, vgetq_lane_u64::<0>(vb0_sum))),
        );
        *mid1 = veorq_u64(
            *mid1,
            vreinterpretq_u64_p128(vmull_p64(eq_sum, vgetq_lane_u64::<0>(vb1_sum))),
        );
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
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
pub unsafe fn k0_f128_reduce_3lane_neon(
    ll: core::arch::aarch64::uint64x2_t,
    mid: core::arch::aarch64::uint64x2_t,
    hh: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let cross = k0_xor3_neon(mid, ll, hh);
        let t1 = k0_xor3_neon(
            cross,
            vextq_u64::<1>(z, hh),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(hh), poly)),
        );
        k0_xor3_neon(
            ll,
            vextq_u64::<1>(z, t1),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1), poly)),
        )
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
pub unsafe fn k0_f128_tri_acc_neon(
    va: core::arch::aarch64::uint64x2_t,
    vb: core::arch::aarch64::uint64x2_t,
    lo: &mut core::arch::aarch64::uint64x2_t,
    mid: &mut core::arch::aarch64::uint64x2_t,
    hi: &mut core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let vb_sw = vextq_u64::<1>(vb, vb);
        let a_lo = vgetq_lane_u64::<0>(va);
        let p_ll = vreinterpretq_u64_p128(vmull_p64(a_lo, vgetq_lane_u64::<0>(vb)));
        let p_hh = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(va),
            vreinterpretq_p64_u64(vb),
        ));
        let p_lh = vreinterpretq_u64_p128(vmull_p64(a_lo, vgetq_lane_u64::<0>(vb_sw)));
        let p_hl = vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(va),
            vreinterpretq_p64_u64(vb_sw),
        ));
        *lo = veorq_u64(*lo, p_ll);
        *hi = veorq_u64(*hi, p_hh);
        *mid = k0_xor3_neon(*mid, p_lh, p_hl);
    }
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
pub unsafe fn k0_f128_reduce_tri_neon(
    lo: core::arch::aarch64::uint64x2_t,
    mid: core::arch::aarch64::uint64x2_t,
    hi: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let t1 = k0_xor3_neon(
            mid,
            vextq_u64::<1>(z, hi),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(hi), poly)),
        );
        k0_xor3_neon(
            lo,
            vextq_u64::<1>(z, t1),
            vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(t1), poly)),
        )
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
        let mut lane = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            if tw_hi == 0 {
                let vtw_lo = _mm512_set1_epi64(tw_lo as i64);
                while lane + 4 <= n_f128 {
                    let idx = lane * 2;
                    let u = _mm512_loadu_si512(top_ptr.add(idx) as *const __m512i);
                    let v = _mm512_loadu_si512(bot_ptr.add(idx) as *const __m512i);
                    let p = k0_f128_mul_low_const_zmm(v, vtw_lo);
                    let nu = _mm512_xor_si512(u, p);
                    _mm512_storeu_si512(top_ptr.add(idx) as *mut __m512i, nu);
                    _mm512_storeu_si512(bot_ptr.add(idx) as *mut __m512i, _mm512_xor_si512(v, nu));
                    lane += 4;
                }
            } else {
                let tw128 = _mm_set_epi64x(tw_hi as i64, tw_lo as i64);
                let tw = _mm512_broadcast_i32x4(tw128);
                let (tw0, tw1) = k0_f128_pre_reduce_const_zmm(tw);
                while lane + 4 <= n_f128 {
                    let idx = lane * 2;
                    let u = _mm512_loadu_si512(top_ptr.add(idx) as *const __m512i);
                    let v = _mm512_loadu_si512(bot_ptr.add(idx) as *const __m512i);
                    let p = k0_f128_mul_const_zmm(v, tw0, tw1);
                    let nu = _mm512_xor_si512(u, p);
                    _mm512_storeu_si512(top_ptr.add(idx) as *mut __m512i, nu);
                    _mm512_storeu_si512(bot_ptr.add(idx) as *mut __m512i, _mm512_xor_si512(v, nu));
                    lane += 4;
                }
            }
        }
        if tw_hi == 0 {
            let vtw_lo = _mm_set1_epi64x(tw_lo as i64);
            while lane + 2 <= n_f128 {
                let idx0 = lane * 2;
                let idx1 = idx0 + 2;
                let u0 = _mm_loadu_si128(top_ptr.add(idx0) as *const __m128i);
                let v0 = _mm_loadu_si128(bot_ptr.add(idx0) as *const __m128i);
                let u1 = _mm_loadu_si128(top_ptr.add(idx1) as *const __m128i);
                let v1 = _mm_loadu_si128(bot_ptr.add(idx1) as *const __m128i);
                let (p0, p1) = k0_f128_mul_low_const_vec2_m128i(v0, v1, vtw_lo, vtw_lo);
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
                let p = k0_f128_mul_low_const_m128i(v, vtw_lo);
                let nu = _mm_xor_si128(u, p);
                _mm_storeu_si128(top_ptr.add(idx) as *mut __m128i, nu);
                _mm_storeu_si128(bot_ptr.add(idx) as *mut __m128i, _mm_xor_si128(v, nu));
                lane += 1;
            }
        } else {
            let tw = _mm_set_epi64x(tw_hi as i64, tw_lo as i64);
            let (tw0, tw1) = k0_f128_pre_reduce_const_m128i(tw);
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
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let mut lane = 0usize;
        if tw_hi == 0 {
            while lane + 2 <= n_f128 {
                let idx0 = lane * 2;
                let idx1 = idx0 + 2;
                let u0 = vld1q_u64(top_ptr.add(idx0));
                let v0 = vld1q_u64(bot_ptr.add(idx0));
                let u1 = vld1q_u64(top_ptr.add(idx1));
                let v1 = vld1q_u64(bot_ptr.add(idx1));
                let (p0, p1) = k0_f128_mul_low_const_vec2_neon(v0, v1, tw_lo, tw_lo);
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
                let p = k0_f128_mul_low_const_neon(v, tw_lo);
                let nu = veorq_u64(u, p);
                vst1q_u64(top_ptr.add(idx), nu);
                vst1q_u64(bot_ptr.add(idx), veorq_u64(v, nu));
                lane += 1;
            }
        } else {
            let tw_arr = [tw_lo, tw_hi];
            let tw = vld1q_u64(tw_arr.as_ptr());
            let (tw0, tw1) = k0_f128_pre_reduce_const_neon(tw);
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
        let mut lane = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            if (to_hi | tia_hi | tib_hi) == 0 {
                let vto = _mm512_set1_epi64(to_lo as i64);
                let vtia = _mm512_set1_epi64(tia_lo as i64);
                let vtib = _mm512_set1_epi64(tib_lo as i64);
                while lane + 4 <= n_f128 {
                    let idx = lane * 2;
                    let xa = _mm512_loadu_si512(a_ptr.add(idx) as *const __m512i);
                    let xb = _mm512_loadu_si512(b_ptr.add(idx) as *const __m512i);
                    let xc = _mm512_loadu_si512(c_ptr.add(idx) as *const __m512i);
                    let xd = _mm512_loadu_si512(d_ptr.add(idx) as *const __m512i);
                    let pc = k0_f128_mul_low_const_zmm(xc, vto);
                    let pd = k0_f128_mul_low_const_zmm(xd, vto);
                    let na = _mm512_xor_si512(xa, pc);
                    let nc = _mm512_xor_si512(xc, na);
                    let nb = _mm512_xor_si512(xb, pd);
                    let nd = _mm512_xor_si512(xd, nb);
                    let pb2 = k0_f128_mul_low_const_zmm(nb, vtia);
                    let pd2 = k0_f128_mul_low_const_zmm(nd, vtib);
                    let fa = _mm512_xor_si512(na, pb2);
                    let fb = _mm512_xor_si512(nb, fa);
                    let fc = _mm512_xor_si512(nc, pd2);
                    let fd = _mm512_xor_si512(nd, fc);
                    _mm512_storeu_si512(a_ptr.add(idx) as *mut __m512i, fa);
                    _mm512_storeu_si512(b_ptr.add(idx) as *mut __m512i, fb);
                    _mm512_storeu_si512(c_ptr.add(idx) as *mut __m512i, fc);
                    _mm512_storeu_si512(d_ptr.add(idx) as *mut __m512i, fd);
                    lane += 4;
                }
            } else {
                let to = _mm512_broadcast_i32x4(_mm_set_epi64x(to_hi as i64, to_lo as i64));
                let tia = _mm512_broadcast_i32x4(_mm_set_epi64x(tia_hi as i64, tia_lo as i64));
                let tib = _mm512_broadcast_i32x4(_mm_set_epi64x(tib_hi as i64, tib_lo as i64));
                let (to0, to1) = k0_f128_pre_reduce_const_zmm(to);
                let (tia0, tia1) = k0_f128_pre_reduce_const_zmm(tia);
                let (tib0, tib1) = k0_f128_pre_reduce_const_zmm(tib);
                while lane + 4 <= n_f128 {
                    let idx = lane * 2;
                    let xa = _mm512_loadu_si512(a_ptr.add(idx) as *const __m512i);
                    let xb = _mm512_loadu_si512(b_ptr.add(idx) as *const __m512i);
                    let xc = _mm512_loadu_si512(c_ptr.add(idx) as *const __m512i);
                    let xd = _mm512_loadu_si512(d_ptr.add(idx) as *const __m512i);
                    let pc = k0_f128_mul_const_zmm(xc, to0, to1);
                    let pd = k0_f128_mul_const_zmm(xd, to0, to1);
                    let na = _mm512_xor_si512(xa, pc);
                    let nc = _mm512_xor_si512(xc, na);
                    let nb = _mm512_xor_si512(xb, pd);
                    let nd = _mm512_xor_si512(xd, nb);
                    let pb2 = k0_f128_mul_const_zmm(nb, tia0, tia1);
                    let pd2 = k0_f128_mul_const_zmm(nd, tib0, tib1);
                    let fa = _mm512_xor_si512(na, pb2);
                    let fb = _mm512_xor_si512(nb, fa);
                    let fc = _mm512_xor_si512(nc, pd2);
                    let fd = _mm512_xor_si512(nd, fc);
                    _mm512_storeu_si512(a_ptr.add(idx) as *mut __m512i, fa);
                    _mm512_storeu_si512(b_ptr.add(idx) as *mut __m512i, fb);
                    _mm512_storeu_si512(c_ptr.add(idx) as *mut __m512i, fc);
                    _mm512_storeu_si512(d_ptr.add(idx) as *mut __m512i, fd);
                    lane += 4;
                }
            }
        }
        if (to_hi | tia_hi | tib_hi) == 0 {
            let vto = _mm_set1_epi64x(to_lo as i64);
            let vtia = _mm_set1_epi64x(tia_lo as i64);
            let vtib = _mm_set1_epi64x(tib_lo as i64);
            while lane < n_f128 {
                let idx = lane * 2;
                let xa = _mm_loadu_si128(a_ptr.add(idx) as *const __m128i);
                let xb = _mm_loadu_si128(b_ptr.add(idx) as *const __m128i);
                let xc = _mm_loadu_si128(c_ptr.add(idx) as *const __m128i);
                let xd = _mm_loadu_si128(d_ptr.add(idx) as *const __m128i);
                let (pc, pd) = k0_f128_mul_low_const_vec2_m128i(xc, xd, vto, vto);
                let na = _mm_xor_si128(xa, pc);
                let nc = _mm_xor_si128(xc, na);
                let nb = _mm_xor_si128(xb, pd);
                let nd = _mm_xor_si128(xd, nb);
                let (pb2, pd2) = k0_f128_mul_low_const_vec2_m128i(nb, nd, vtia, vtib);
                let fa = _mm_xor_si128(na, pb2);
                let fb = _mm_xor_si128(nb, fa);
                let fc = _mm_xor_si128(nc, pd2);
                let fd = _mm_xor_si128(nd, fc);
                _mm_storeu_si128(a_ptr.add(idx) as *mut __m128i, fa);
                _mm_storeu_si128(b_ptr.add(idx) as *mut __m128i, fb);
                _mm_storeu_si128(c_ptr.add(idx) as *mut __m128i, fc);
                _mm_storeu_si128(d_ptr.add(idx) as *mut __m128i, fd);
                lane += 1;
            }
        } else {
            let to = _mm_set_epi64x(to_hi as i64, to_lo as i64);
            let tia = _mm_set_epi64x(tia_hi as i64, tia_lo as i64);
            let tib = _mm_set_epi64x(tib_hi as i64, tib_lo as i64);
            let (to0, to1) = k0_f128_pre_reduce_const_m128i(to);
            let (tia0, tia1) = k0_f128_pre_reduce_const_m128i(tia);
            let (tib0, tib1) = k0_f128_pre_reduce_const_m128i(tib);
            while lane < n_f128 {
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
                lane += 1;
            }
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        if (to_hi | tia_hi | tib_hi) == 0 {
            for lane in 0..n_f128 {
                let idx = lane * 2;
                let xa = vld1q_u64(a_ptr.add(idx));
                let xb = vld1q_u64(b_ptr.add(idx));
                let xc = vld1q_u64(c_ptr.add(idx));
                let xd = vld1q_u64(d_ptr.add(idx));
                let (pc, pd) = k0_f128_mul_low_const_vec2_neon(xc, xd, to_lo, to_lo);
                let na = veorq_u64(xa, pc);
                let nc = veorq_u64(xc, na);
                let nb = veorq_u64(xb, pd);
                let nd = veorq_u64(xd, nb);
                let (pb2, pd2) = k0_f128_mul_low_const_vec2_neon(nb, nd, tia_lo, tib_lo);
                let fa = veorq_u64(na, pb2);
                let fb = veorq_u64(nb, fa);
                let fc = veorq_u64(nc, pd2);
                let fd = veorq_u64(nd, fc);
                vst1q_u64(a_ptr.add(idx), fa);
                vst1q_u64(b_ptr.add(idx), fb);
                vst1q_u64(c_ptr.add(idx), fc);
                vst1q_u64(d_ptr.add(idx), fd);
            }
        } else {
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
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        let (tw512_r0, tw512_r1) = {
            let mut z0 = [_mm512_setzero_si512(); 15];
            let mut z1 = [_mm512_setzero_si512(); 15];
            for i in 0..15 {
                z0[i] = _mm512_broadcast_i32x4(tw_r0[i]);
                z1[i] = _mm512_broadcast_i32x4(tw_r1[i]);
            }
            (z0, z1)
        };
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        macro_rules! btf2_zmm {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let p0 = k0_f128_mul_const_zmm($v[$v0], $tw0[$t0], $tw1[$t0]);
                let p1 = k0_f128_mul_const_zmm($v[$v1], $tw0[$t1], $tw1[$t1]);
                let nu0 = _mm512_xor_si512($v[$u0], p0);
                $v[$v0] = _mm512_xor_si512($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = _mm512_xor_si512($v[$u1], p1);
                $v[$v1] = _mm512_xor_si512($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
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
            let mut lane = 0usize;
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            while lane + 4 <= num_ntts {
                let mut v = [_mm512_setzero_si512(); 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = _mm512_loadu_si512(ptr.add(off) as *const __m512i);
                }
                btf2_zmm!(v, tw512_r0, tw512_r1, 0, 8, 0, 1, 9, 0);
                btf2_zmm!(v, tw512_r0, tw512_r1, 2, 10, 0, 3, 11, 0);
                btf2_zmm!(v, tw512_r0, tw512_r1, 4, 12, 0, 5, 13, 0);
                btf2_zmm!(v, tw512_r0, tw512_r1, 6, 14, 0, 7, 15, 0);
                btf2_zmm!(v, tw512_r0, tw512_r1, 0, 4, 1, 1, 5, 1);
                btf2_zmm!(v, tw512_r0, tw512_r1, 2, 6, 1, 3, 7, 1);
                btf2_zmm!(v, tw512_r0, tw512_r1, 8, 12, 2, 9, 13, 2);
                btf2_zmm!(v, tw512_r0, tw512_r1, 10, 14, 2, 11, 15, 2);
                btf2_zmm!(v, tw512_r0, tw512_r1, 0, 2, 3, 1, 3, 3);
                btf2_zmm!(v, tw512_r0, tw512_r1, 4, 6, 4, 5, 7, 4);
                btf2_zmm!(v, tw512_r0, tw512_r1, 8, 10, 5, 9, 11, 5);
                btf2_zmm!(v, tw512_r0, tw512_r1, 12, 14, 6, 13, 15, 6);
                btf2_zmm!(v, tw512_r0, tw512_r1, 0, 1, 7, 2, 3, 8);
                btf2_zmm!(v, tw512_r0, tw512_r1, 4, 5, 9, 6, 7, 10);
                btf2_zmm!(v, tw512_r0, tw512_r1, 8, 9, 11, 10, 11, 12);
                btf2_zmm!(v, tw512_r0, tw512_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    _mm512_storeu_si512(ptr.add(off) as *mut __m512i, v[i]);
                }
                lane += 4;
            }
            while lane < num_ntts {
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
                lane += 1;
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
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        let (tw0_z0, tw0_z1, tw1_z0, tw1_z1) = {
            let mut a0 = [_mm512_setzero_si512(); 15];
            let mut a1 = [_mm512_setzero_si512(); 15];
            let mut b0 = [_mm512_setzero_si512(); 15];
            let mut b1 = [_mm512_setzero_si512(); 15];
            for i in 0..15 {
                a0[i] = _mm512_broadcast_i32x4(tw0_r0[i]);
                a1[i] = _mm512_broadcast_i32x4(tw0_r1[i]);
                b0[i] = _mm512_broadcast_i32x4(tw1_r0[i]);
                b1[i] = _mm512_broadcast_i32x4(tw1_r1[i]);
            }
            (a0, a1, b0, b1)
        };
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        macro_rules! btf2_zmm {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let p0 = k0_f128_mul_const_zmm($v[$v0], $tw0[$t0], $tw1[$t0]);
                let p1 = k0_f128_mul_const_zmm($v[$v1], $tw0[$t1], $tw1[$t1]);
                let nu0 = _mm512_xor_si512($v[$u0], p0);
                $v[$v0] = _mm512_xor_si512($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = _mm512_xor_si512($v[$u1], p1);
                $v[$v1] = _mm512_xor_si512($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
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
            let mut lane = 0usize;
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            while lane + 4 <= num_ntts {
                let mut v0 = [_mm512_setzero_si512(); 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v0[i] = _mm512_loadu_si512(src_ptr.add(off) as *const __m512i);
                }
                let mut v1 = v0;
                btf2_zmm!(v0, tw0_z0, tw0_z1, 0, 8, 0, 1, 9, 0);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 2, 10, 0, 3, 11, 0);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 4, 12, 0, 5, 13, 0);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 6, 14, 0, 7, 15, 0);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 0, 4, 1, 1, 5, 1);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 2, 6, 1, 3, 7, 1);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 8, 12, 2, 9, 13, 2);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 10, 14, 2, 11, 15, 2);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 0, 2, 3, 1, 3, 3);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 4, 6, 4, 5, 7, 4);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 8, 10, 5, 9, 11, 5);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 12, 14, 6, 13, 15, 6);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 0, 1, 7, 2, 3, 8);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 4, 5, 9, 6, 7, 10);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 8, 9, 11, 10, 11, 12);
                btf2_zmm!(v0, tw0_z0, tw0_z1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    _mm512_storeu_si512(dst0_ptr.add(off) as *mut __m512i, v0[i]);
                }
                btf2_zmm!(v1, tw1_z0, tw1_z1, 0, 8, 0, 1, 9, 0);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 2, 10, 0, 3, 11, 0);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 4, 12, 0, 5, 13, 0);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 6, 14, 0, 7, 15, 0);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 0, 4, 1, 1, 5, 1);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 2, 6, 1, 3, 7, 1);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 8, 12, 2, 9, 13, 2);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 10, 14, 2, 11, 15, 2);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 0, 2, 3, 1, 3, 3);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 4, 6, 4, 5, 7, 4);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 8, 10, 5, 9, 11, 5);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 12, 14, 6, 13, 15, 6);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 0, 1, 7, 2, 3, 8);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 4, 5, 9, 6, 7, 10);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 8, 9, 11, 10, 11, 12);
                btf2_zmm!(v1, tw1_z0, tw1_z1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    _mm512_storeu_si512(dst1_ptr.add(off) as *mut __m512i, v1[i]);
                }
                lane += 4;
            }
            while lane < num_ntts {
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
                lane += 1;
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
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        let (tw512_r0, tw512_r1) = {
            let mut z0 = [_mm512_setzero_si512(); 7];
            let mut z1 = [_mm512_setzero_si512(); 7];
            for i in 0..7 {
                z0[i] = _mm512_broadcast_i32x4(tw_r0[i]);
                z1[i] = _mm512_broadcast_i32x4(tw_r1[i]);
            }
            (z0, z1)
        };
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        macro_rules! btf2_zmm {
            ($v:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let p0 = k0_f128_mul_const_zmm($v[$v0], tw512_r0[$t0], tw512_r1[$t0]);
                let p1 = k0_f128_mul_const_zmm($v[$v1], tw512_r0[$t1], tw512_r1[$t1]);
                let nu0 = _mm512_xor_si512($v[$u0], p0);
                $v[$v0] = _mm512_xor_si512($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = _mm512_xor_si512($v[$u1], p1);
                $v[$v1] = _mm512_xor_si512($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
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
        let mut lane = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        while lane + 4 <= num_ntts {
            let mut v = [_mm512_setzero_si512(); 8];
            for i in 0..8 {
                let off = 2 * ((i * eighth + r) * num_ntts + lane);
                v[i] = _mm512_loadu_si512(ptr.add(off) as *const __m512i);
            }
            btf2_zmm!(v, 0, 4, 0, 1, 5, 0);
            btf2_zmm!(v, 2, 6, 0, 3, 7, 0);
            btf2_zmm!(v, 0, 2, 1, 1, 3, 1);
            btf2_zmm!(v, 4, 6, 2, 5, 7, 2);
            btf2_zmm!(v, 0, 1, 3, 2, 3, 4);
            btf2_zmm!(v, 4, 5, 5, 6, 7, 6);
            for i in 0..8 {
                let off = 2 * ((i * eighth + r) * num_ntts + lane);
                _mm512_storeu_si512(ptr.add(off) as *mut __m512i, v[i]);
            }
            lane += 4;
        }
        while lane < num_ntts {
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
            lane += 1;
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
        let mut t = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let zr0 = _mm512_broadcast_i32x4(vr0);
            let zr1 = _mm512_broadcast_i32x4(vr1);
            while t + 4 <= dst_len {
                let s = 4 * (base + t);
                let v0 = _mm512_loadu_si512(src_ptr.add(s) as *const __m512i);
                let v1 = _mm512_loadu_si512(src_ptr.add(s + 8) as *const __m512i);
                let nf = k0_f128_fold_even_odd_zmm(v0, v1, zr0, zr1);
                _mm512_storeu_si512(dst_ptr.add(2 * t) as *mut __m512i, nf);
                t += 4;
            }
        }
        let lanes = dst_len & !1;
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
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let zr0 = _mm512_broadcast_i32x4(vr0);
            let zr1 = _mm512_broadcast_i32x4(vr1);
            let idx_dup_even = _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0);
            let (mut z_ll, mut z_mid, mut z_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            while t + 8 <= lanes {
                let s = 4 * (base + t);
                let d = 2 * t;
                let f0 = _mm512_loadu_si512(f_ptr.add(s) as *const __m512i);
                let f1 = _mm512_loadu_si512(f_ptr.add(s + 8) as *const __m512i);
                let f2 = _mm512_loadu_si512(f_ptr.add(s + 16) as *const __m512i);
                let f3 = _mm512_loadu_si512(f_ptr.add(s + 24) as *const __m512i);
                let nf_a = k0_f128_fold_even_odd_zmm(f0, f1, zr0, zr1);
                let nf_b = k0_f128_fold_even_odd_zmm(f2, f3, zr0, zr1);
                _mm512_storeu_si512(fc_ptr.add(d) as *mut __m512i, nf_a);
                _mm512_storeu_si512(fc_ptr.add(d + 8) as *mut __m512i, nf_b);
                let b0 = _mm512_loadu_si512(b_ptr.add(s) as *const __m512i);
                let b1 = _mm512_loadu_si512(b_ptr.add(s + 8) as *const __m512i);
                let b2 = _mm512_loadu_si512(b_ptr.add(s + 16) as *const __m512i);
                let b3 = _mm512_loadu_si512(b_ptr.add(s + 24) as *const __m512i);
                let nb_a = k0_f128_fold_even_odd_zmm(b0, b1, zr0, zr1);
                let nb_b = k0_f128_fold_even_odd_zmm(b2, b3, zr0, zr1);
                _mm512_storeu_si512(bc_ptr.add(d) as *mut __m512i, nb_a);
                _mm512_storeu_si512(bc_ptr.add(d + 8) as *mut __m512i, nb_b);
                let nf_a_prep = _mm512_xor_si512(
                    nf_a,
                    _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nf_a),
                );
                let nb_a_prep = _mm512_xor_si512(
                    nb_a,
                    _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nb_a),
                );
                let nf_b_prep = _mm512_xor_si512(
                    nf_b,
                    _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nf_b),
                );
                let nb_b_prep = _mm512_xor_si512(
                    nb_b,
                    _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nb_b),
                );
                k0_f128_mul_acc2_zmm(
                    nf_a_prep, nb_a_prep, nf_b_prep, nb_b_prep, &mut z_ll, &mut z_mid, &mut z_hh,
                );
                t += 8;
            }
            while t + 4 <= lanes {
                let s = 4 * (base + t);
                let d = 2 * t;
                let f0 = _mm512_loadu_si512(f_ptr.add(s) as *const __m512i);
                let f1 = _mm512_loadu_si512(f_ptr.add(s + 8) as *const __m512i);
                let nf_a = k0_f128_fold_even_odd_zmm(f0, f1, zr0, zr1);
                _mm512_storeu_si512(fc_ptr.add(d) as *mut __m512i, nf_a);
                let b0 = _mm512_loadu_si512(b_ptr.add(s) as *const __m512i);
                let b1 = _mm512_loadu_si512(b_ptr.add(s + 8) as *const __m512i);
                let nb_a = k0_f128_fold_even_odd_zmm(b0, b1, zr0, zr1);
                _mm512_storeu_si512(bc_ptr.add(d) as *mut __m512i, nb_a);
                let nf_a_prep = _mm512_xor_si512(
                    nf_a,
                    _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nf_a),
                );
                let nb_a_prep = _mm512_xor_si512(
                    nb_a,
                    _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nb_a),
                );
                k0_f128_mul_acc_zmm(nf_a_prep, nb_a_prep, &mut z_ll, &mut z_mid, &mut z_hh);
                t += 4;
            }
            let ll_256 = _mm256_xor_si256(
                _mm512_castsi512_si256(z_ll),
                _mm512_extracti64x4_epi64::<1>(z_ll),
            );
            let mid_256 = _mm256_xor_si256(
                _mm512_castsi512_si256(z_mid),
                _mm512_extracti64x4_epi64::<1>(z_mid),
            );
            let hh_256 = _mm256_xor_si256(
                _mm512_castsi512_si256(z_hh),
                _mm512_extracti64x4_epi64::<1>(z_hh),
            );
            u0_ll = _mm256_castsi256_si128(ll_256);
            u0_mid = _mm256_castsi256_si128(mid_256);
            u0_hh = _mm256_castsi256_si128(hh_256);
            u2_ll = _mm256_extracti128_si256::<1>(ll_256);
            u2_mid = _mm256_extracti128_si256::<1>(mid_256);
            u2_hh = _mm256_extracti128_si256::<1>(hh_256);
        }
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
            k0_f128_tri_acc_neon(nf0, nb0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            k0_f128_tri_acc_neon(
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
        let u0 = k0_f128_reduce_tri_neon(u0_ll, u0_mid, u0_hh);
        let u2 = k0_f128_reduce_tri_neon(u2_ll, u2_mid, u2_hh);
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
        let mut x_lo = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let zr0 = _mm512_broadcast_i32x4(vr0);
            let zr1 = _mm512_broadcast_i32x4(vr1);
            let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
            let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
            let (mut zp1_ll, mut zp1_mid, mut zp1_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            let (mut zpi_ll, mut zpi_mid, mut zpi_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            while x_lo + 4 <= lo_size {
                let s = 8 * x_lo;
                let d = 4 * x_lo;
                let va0 = _mm512_loadu_si512(a_in_ptr.add(s) as *const __m512i);
                let va1 = _mm512_loadu_si512(a_in_ptr.add(s + 8) as *const __m512i);
                let va2 = _mm512_loadu_si512(a_in_ptr.add(s + 16) as *const __m512i);
                let va3 = _mm512_loadu_si512(a_in_ptr.add(s + 24) as *const __m512i);
                let na_a = k0_f128_fold_even_odd_zmm(va0, va1, zr0, zr1);
                let na_b = k0_f128_fold_even_odd_zmm(va2, va3, zr0, zr1);
                _mm512_storeu_si512(a_out_ptr.add(d) as *mut __m512i, na_a);
                _mm512_storeu_si512(a_out_ptr.add(d + 8) as *mut __m512i, na_b);
                let vb0 = _mm512_loadu_si512(b_in_ptr.add(s) as *const __m512i);
                let vb1 = _mm512_loadu_si512(b_in_ptr.add(s + 8) as *const __m512i);
                let vb2 = _mm512_loadu_si512(b_in_ptr.add(s + 16) as *const __m512i);
                let vb3 = _mm512_loadu_si512(b_in_ptr.add(s + 24) as *const __m512i);
                let nb_a = k0_f128_fold_even_odd_zmm(vb0, vb1, zr0, zr1);
                let nb_b = k0_f128_fold_even_odd_zmm(vb2, vb3, zr0, zr1);
                _mm512_storeu_si512(b_out_ptr.add(d) as *mut __m512i, nb_a);
                _mm512_storeu_si512(b_out_ptr.add(d + 8) as *mut __m512i, nb_b);
                let a0_vec = _mm512_permutex2var_epi64(na_a, even_idx, na_b);
                let a1_vec = _mm512_permutex2var_epi64(na_a, odd_idx, na_b);
                let b0_vec = _mm512_permutex2var_epi64(nb_a, even_idx, nb_b);
                let b1_vec = _mm512_permutex2var_epi64(nb_a, odd_idx, nb_b);
                let g1 = k0_f128_mul_zmm(a1_vec, b1_vec);
                let gi = k0_f128_mul_zmm(
                    _mm512_xor_si512(a0_vec, a1_vec),
                    _mm512_xor_si512(b0_vec, b1_vec),
                );
                let eq = _mm512_loadu_si512(eq_lo_ptr.add(2 * x_lo) as *const __m512i);
                k0_f128_mul_acc_zmm(eq, g1, &mut zp1_ll, &mut zp1_mid, &mut zp1_hh);
                k0_f128_mul_acc_zmm(eq, gi, &mut zpi_ll, &mut zpi_mid, &mut zpi_hh);
                x_lo += 4;
            }
            p1_ll = k0_f128_hsum_zmm(zp1_ll);
            p1_mid = k0_f128_hsum_zmm(zp1_mid);
            p1_hh = k0_f128_hsum_zmm(zp1_hh);
            pi_ll = k0_f128_hsum_zmm(zpi_ll);
            pi_mid = k0_f128_hsum_zmm(zpi_mid);
            pi_hh = k0_f128_hsum_zmm(zpi_hh);
        }
        while x_lo < lo_size {
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
            x_lo += 1;
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
            k0_f128_tri_acc_neon(eq, g1, &mut p1_ll, &mut p1_mid, &mut p1_hh);
            k0_f128_tri_acc_neon(eq, gi, &mut pi_ll, &mut pi_mid, &mut pi_hh);
        }
        let p1 = k0_f128_reduce_tri_neon(p1_ll, p1_mid, p1_hh);
        let pi = k0_f128_reduce_tri_neon(pi_ll, pi_mid, pi_hh);
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
        let mut i = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let (mut z_ll, mut z_mid, mut z_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            while i + 8 <= n_f128 {
                let idx = 2 * i;
                let va0 = _mm512_loadu_si512(a_ptr.add(idx) as *const __m512i);
                let vb0 = _mm512_loadu_si512(b_ptr.add(idx) as *const __m512i);
                let va1 = _mm512_loadu_si512(a_ptr.add(idx + 8) as *const __m512i);
                let vb1 = _mm512_loadu_si512(b_ptr.add(idx + 8) as *const __m512i);
                k0_f128_mul_acc2_zmm(va0, vb0, va1, vb1, &mut z_ll, &mut z_mid, &mut z_hh);
                i += 8;
            }
            while i + 4 <= n_f128 {
                let idx = 2 * i;
                let va = _mm512_loadu_si512(a_ptr.add(idx) as *const __m512i);
                let vb = _mm512_loadu_si512(b_ptr.add(idx) as *const __m512i);
                k0_f128_mul_acc_zmm(va, vb, &mut z_ll, &mut z_mid, &mut z_hh);
                i += 4;
            }
            ll = k0_f128_hsum_zmm(z_ll);
            mid = k0_f128_hsum_zmm(z_mid);
            hh = k0_f128_hsum_zmm(z_hh);
        }
        while i + 2 <= n_f128 {
            let idx = 2 * i;
            let va0 = _mm_loadu_si128(a_ptr.add(idx) as *const __m128i);
            let vb0 = _mm_loadu_si128(b_ptr.add(idx) as *const __m128i);
            let va1 = _mm_loadu_si128(a_ptr.add(idx + 2) as *const __m128i);
            let vb1 = _mm_loadu_si128(b_ptr.add(idx + 2) as *const __m128i);
            k0_f128_mul_acc2_m128i(va0, vb0, va1, vb1, &mut ll, &mut mid, &mut hh);
            i += 2;
        }
        while i < n_f128 {
            let idx = 2 * i;
            let va = _mm_loadu_si128(a_ptr.add(idx) as *const __m128i);
            let vb = _mm_loadu_si128(b_ptr.add(idx) as *const __m128i);
            k0_f128_mul_acc_m128i(va, vb, &mut ll, &mut mid, &mut hh);
            i += 1;
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
        let (mut lo, mut mid, mut hi) = (z, z, z);
        for i in 0..n_f128 {
            let idx = 2 * i;
            let va = vld1q_u64(a_ptr.add(idx));
            let vb = vld1q_u64(b_ptr.add(idx));
            k0_f128_tri_acc_neon(va, vb, &mut lo, &mut mid, &mut hi);
        }
        let res = k0_f128_reduce_tri_neon(lo, mid, hi);
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
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let za0 = _mm512_broadcast_i32x4(va0);
            let za1 = _mm512_broadcast_i32x4(va1);
            while i + 4 <= n_f128 {
                let idx = 2 * i;
                let s = _mm512_loadu_si512(src_ptr.add(idx) as *const __m512i);
                let p = k0_f128_mul_const_zmm(s, za0, za1);
                _mm512_storeu_si512(dst_ptr.add(idx) as *mut __m512i, p);
                i += 4;
            }
        }
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
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let za0 = _mm512_broadcast_i32x4(va0);
            let za1 = _mm512_broadcast_i32x4(va1);
            while i + 4 <= n_f128 {
                let idx = 2 * i;
                let s = _mm512_loadu_si512(src_ptr.add(idx) as *const __m512i);
                let d = _mm512_loadu_si512(dst_ptr.add(idx) as *const __m512i);
                let p = k0_f128_mul_const_zmm(s, za0, za1);
                _mm512_storeu_si512(dst_ptr.add(idx) as *mut __m512i, _mm512_xor_si512(d, p));
                i += 4;
            }
        }
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
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_fold_alpha_four_slice(
    dst_ptr: *mut u64,
    s0_ptr: *const u64,
    s1_ptr: *const u64,
    s2_ptr: *const u64,
    s3_ptr: *const u64,
    n_f128: usize,
    a0_lo: u64,
    a0_hi: u64,
    a1_lo: u64,
    a1_hi: u64,
    a2_lo: u64,
    a2_hi: u64,
    a3_lo: u64,
    a3_hi: u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let (va0_0, va0_1) =
            k0_f128_pre_reduce_const_m128i(_mm_set_epi64x(a0_hi as i64, a0_lo as i64));
        let (va1_0, va1_1) =
            k0_f128_pre_reduce_const_m128i(_mm_set_epi64x(a1_hi as i64, a1_lo as i64));
        let (va2_0, va2_1) =
            k0_f128_pre_reduce_const_m128i(_mm_set_epi64x(a2_hi as i64, a2_lo as i64));
        let (va3_0, va3_1) =
            k0_f128_pre_reduce_const_m128i(_mm_set_epi64x(a3_hi as i64, a3_lo as i64));
        let mut i = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let poly = _mm512_set1_epi64(0x87);
            let za0_0 = _mm512_broadcast_i32x4(va0_0);
            let za0_1 = _mm512_broadcast_i32x4(va0_1);
            let za1_0 = _mm512_broadcast_i32x4(va1_0);
            let za1_1 = _mm512_broadcast_i32x4(va1_1);
            let za2_0 = _mm512_broadcast_i32x4(va2_0);
            let za2_1 = _mm512_broadcast_i32x4(va2_1);
            let za3_0 = _mm512_broadcast_i32x4(va3_0);
            let za3_1 = _mm512_broadcast_i32x4(va3_1);
            while i + 4 <= n_f128 {
                let idx = 2 * i;
                let x0 = _mm512_loadu_si512(s0_ptr.add(idx) as *const __m512i);
                let x1 = _mm512_loadu_si512(s1_ptr.add(idx) as *const __m512i);
                let x2 = _mm512_loadu_si512(s2_ptr.add(idx) as *const __m512i);
                let x3 = _mm512_loadu_si512(s3_ptr.add(idx) as *const __m512i);
                let mut l = _mm512_xor_si512(
                    _mm512_clmulepi64_epi128::<0x00>(x0, za0_0),
                    _mm512_clmulepi64_epi128::<0x11>(x0, za0_0),
                );
                let mut h = _mm512_xor_si512(
                    _mm512_clmulepi64_epi128::<0x00>(x0, za0_1),
                    _mm512_clmulepi64_epi128::<0x11>(x0, za0_1),
                );
                l = k0_xor3_zmm(
                    l,
                    _mm512_clmulepi64_epi128::<0x00>(x1, za1_0),
                    _mm512_clmulepi64_epi128::<0x11>(x1, za1_0),
                );
                h = k0_xor3_zmm(
                    h,
                    _mm512_clmulepi64_epi128::<0x00>(x1, za1_1),
                    _mm512_clmulepi64_epi128::<0x11>(x1, za1_1),
                );
                l = k0_xor3_zmm(
                    l,
                    _mm512_clmulepi64_epi128::<0x00>(x2, za2_0),
                    _mm512_clmulepi64_epi128::<0x11>(x2, za2_0),
                );
                h = k0_xor3_zmm(
                    h,
                    _mm512_clmulepi64_epi128::<0x00>(x2, za2_1),
                    _mm512_clmulepi64_epi128::<0x11>(x2, za2_1),
                );
                l = k0_xor3_zmm(
                    l,
                    _mm512_clmulepi64_epi128::<0x00>(x3, za3_0),
                    _mm512_clmulepi64_epi128::<0x11>(x3, za3_0),
                );
                h = k0_xor3_zmm(
                    h,
                    _mm512_clmulepi64_epi128::<0x00>(x3, za3_1),
                    _mm512_clmulepi64_epi128::<0x11>(x3, za3_1),
                );
                let red = _mm512_clmulepi64_epi128::<0x01>(h, poly);
                _mm512_storeu_si512(
                    dst_ptr.add(idx) as *mut __m512i,
                    k0_xor3_zmm(l, _mm512_bslli_epi128::<8>(h), red),
                );
                i += 4;
            }
        }
        let poly = _mm_set1_epi64x(0x87);
        while i < n_f128 {
            let idx = 2 * i;
            let x0 = _mm_loadu_si128(s0_ptr.add(idx) as *const __m128i);
            let x1 = _mm_loadu_si128(s1_ptr.add(idx) as *const __m128i);
            let x2 = _mm_loadu_si128(s2_ptr.add(idx) as *const __m128i);
            let x3 = _mm_loadu_si128(s3_ptr.add(idx) as *const __m128i);
            let mut l = _mm_xor_si128(
                _mm_clmulepi64_si128::<0x00>(x0, va0_0),
                _mm_clmulepi64_si128::<0x11>(x0, va0_0),
            );
            let mut h = _mm_xor_si128(
                _mm_clmulepi64_si128::<0x00>(x0, va0_1),
                _mm_clmulepi64_si128::<0x11>(x0, va0_1),
            );
            l = k0_xor3_m128i(
                l,
                _mm_clmulepi64_si128::<0x00>(x1, va1_0),
                _mm_clmulepi64_si128::<0x11>(x1, va1_0),
            );
            h = k0_xor3_m128i(
                h,
                _mm_clmulepi64_si128::<0x00>(x1, va1_1),
                _mm_clmulepi64_si128::<0x11>(x1, va1_1),
            );
            l = k0_xor3_m128i(
                l,
                _mm_clmulepi64_si128::<0x00>(x2, va2_0),
                _mm_clmulepi64_si128::<0x11>(x2, va2_0),
            );
            h = k0_xor3_m128i(
                h,
                _mm_clmulepi64_si128::<0x00>(x2, va2_1),
                _mm_clmulepi64_si128::<0x11>(x2, va2_1),
            );
            l = k0_xor3_m128i(
                l,
                _mm_clmulepi64_si128::<0x00>(x3, va3_0),
                _mm_clmulepi64_si128::<0x11>(x3, va3_0),
            );
            h = k0_xor3_m128i(
                h,
                _mm_clmulepi64_si128::<0x00>(x3, va3_1),
                _mm_clmulepi64_si128::<0x11>(x3, va3_1),
            );
            let red = _mm_clmulepi64_si128::<0x01>(h, poly);
            _mm_storeu_si128(
                dst_ptr.add(idx) as *mut __m128i,
                k0_xor3_m128i(l, _mm_bslli_si128::<8>(h), red),
            );
            i += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let poly = vreinterpretq_p64_u64(vdupq_n_u64(0x87));
        let (va0_0, va0_1) = k0_f128_pre_reduce_const_neon(vld1q_u64([a0_lo, a0_hi].as_ptr()));
        let (va1_0, va1_1) = k0_f128_pre_reduce_const_neon(vld1q_u64([a1_lo, a1_hi].as_ptr()));
        let (va2_0, va2_1) = k0_f128_pre_reduce_const_neon(vld1q_u64([a2_lo, a2_hi].as_ptr()));
        let (va3_0, va3_1) = k0_f128_pre_reduce_const_neon(vld1q_u64([a3_lo, a3_hi].as_ptr()));
        for i in 0..n_f128 {
            let idx = 2 * i;
            let x0 = vld1q_u64(s0_ptr.add(idx));
            let x0_lo = vgetq_lane_u64::<0>(x0);
            let x1 = vld1q_u64(s1_ptr.add(idx));
            let x1_lo = vgetq_lane_u64::<0>(x1);
            let x2 = vld1q_u64(s2_ptr.add(idx));
            let x2_lo = vgetq_lane_u64::<0>(x2);
            let x3 = vld1q_u64(s3_ptr.add(idx));
            let x3_lo = vgetq_lane_u64::<0>(x3);
            let mut l = veorq_u64(
                vreinterpretq_u64_p128(vmull_p64(x0_lo, vgetq_lane_u64::<0>(va0_0))),
                vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(x0),
                    vreinterpretq_p64_u64(va0_0),
                )),
            );
            let mut h = veorq_u64(
                vreinterpretq_u64_p128(vmull_p64(x0_lo, vgetq_lane_u64::<0>(va0_1))),
                vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(x0),
                    vreinterpretq_p64_u64(va0_1),
                )),
            );
            l = k0_xor3_neon(
                l,
                vreinterpretq_u64_p128(vmull_p64(x1_lo, vgetq_lane_u64::<0>(va1_0))),
                vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(x1),
                    vreinterpretq_p64_u64(va1_0),
                )),
            );
            h = k0_xor3_neon(
                h,
                vreinterpretq_u64_p128(vmull_p64(x1_lo, vgetq_lane_u64::<0>(va1_1))),
                vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(x1),
                    vreinterpretq_p64_u64(va1_1),
                )),
            );
            l = k0_xor3_neon(
                l,
                vreinterpretq_u64_p128(vmull_p64(x2_lo, vgetq_lane_u64::<0>(va2_0))),
                vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(x2),
                    vreinterpretq_p64_u64(va2_0),
                )),
            );
            h = k0_xor3_neon(
                h,
                vreinterpretq_u64_p128(vmull_p64(x2_lo, vgetq_lane_u64::<0>(va2_1))),
                vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(x2),
                    vreinterpretq_p64_u64(va2_1),
                )),
            );
            l = k0_xor3_neon(
                l,
                vreinterpretq_u64_p128(vmull_p64(x3_lo, vgetq_lane_u64::<0>(va3_0))),
                vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(x3),
                    vreinterpretq_p64_u64(va3_0),
                )),
            );
            h = k0_xor3_neon(
                h,
                vreinterpretq_u64_p128(vmull_p64(x3_lo, vgetq_lane_u64::<0>(va3_1))),
                vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(x3),
                    vreinterpretq_p64_u64(va3_1),
                )),
            );
            let red = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(h), poly));
            vst1q_u64(dst_ptr.add(idx), k0_xor3_neon(l, vextq_u64::<1>(z, h), red));
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        k0_f128_scalar_mul_into_slice(dst_ptr, s0_ptr, n_f128, a0_lo, a0_hi);
        k0_f128_scalar_mul_add_slice(dst_ptr, s1_ptr, n_f128, a1_lo, a1_hi);
        k0_f128_scalar_mul_add_slice(dst_ptr, s2_ptr, n_f128, a2_lo, a2_hi);
        k0_f128_scalar_mul_add_slice(dst_ptr, s3_ptr, n_f128, a3_lo, a3_hi);
    }
}
#[inline(always)]
pub unsafe fn k0_f128_xor_assign_slice(dst_ptr: *mut u64, src_ptr: *const u64, n_f128: usize) {
    unsafe {
        let n_u64 = 2 * n_f128;
        let mut i = 0usize;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            use core::arch::x86_64::*;
            while i + 8 <= n_u64 {
                let d = _mm512_loadu_si512(dst_ptr.add(i) as *const __m512i);
                let s = _mm512_loadu_si512(src_ptr.add(i) as *const __m512i);
                _mm512_storeu_si512(dst_ptr.add(i) as *mut __m512i, _mm512_xor_si512(d, s));
                i += 8;
            }
        }
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
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_fold64_banked_nested_slice(
    src_ptr: *const u64,
    dst_ptr: *mut u64,
    n_dst: usize,
    w_ptr: *const u64,
    r0_lo: u64,
    r0_hi: u64,
    r1_lo: u64,
    r1_hi: u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut vw = [_mm_setzero_si128(); 16];
        let mut vw_sum = [_mm_setzero_si128(); 16];
        for b in 0..16 {
            let wb = _mm_loadu_si128(w_ptr.add(2 * b) as *const __m128i);
            vw[b] = wb;
            vw_sum[b] = _mm_xor_si128(wb, _mm_srli_si128::<8>(wb));
        }
        let vr0 = _mm_set_epi64x(r0_hi as i64, r0_lo as i64);
        let (rr0_0, rr0_1) = k0_f128_pre_reduce_const_m128i(vr0);
        let vr1 = _mm_set_epi64x(r1_hi as i64, r1_lo as i64);
        let (rr1_0, rr1_1) = k0_f128_pre_reduce_const_m128i(vr1);
        let mut t = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let mut zvw = [_mm512_setzero_si512(); 16];
            for b in 0..16 {
                zvw[b] = _mm512_broadcast_i32x4(vw[b]);
            }
            let zrr0_0 = _mm512_broadcast_i32x4(rr0_0);
            let zrr0_1 = _mm512_broadcast_i32x4(rr0_1);
            let zrr1_0 = _mm512_broadcast_i32x4(rr1_0);
            let zrr1_1 = _mm512_broadcast_i32x4(rr1_1);
            macro_rules! fold_slot_zmm {
                ($base:expr) => {{
                    let mut z_ll = _mm512_setzero_si512();
                    let mut z_mid = _mm512_setzero_si512();
                    let mut z_hh = _mm512_setzero_si512();
                    let mut b = 0usize;
                    while b < 16 {
                        let s0 = _mm512_loadu_si512(
                            src_ptr.add(2 * ($base + (b << 2))) as *const __m512i
                        );
                        let s1 = _mm512_loadu_si512(
                            src_ptr.add(2 * ($base + ((b + 1) << 2))) as *const __m512i
                        );
                        k0_f128_mul_acc2_zmm(
                            zvw[b],
                            s0,
                            zvw[b + 1],
                            s1,
                            &mut z_ll,
                            &mut z_mid,
                            &mut z_hh,
                        );
                        b += 2;
                    }
                    k0_f128_reduce_lanes_zmm(z_ll, z_mid, z_hh)
                }};
            }
            while t + 4 <= n_dst {
                let m0 = fold_slot_zmm!(64 * t);
                let m1 = fold_slot_zmm!(64 * (t + 1));
                let m2 = fold_slot_zmm!(64 * (t + 2));
                let m3 = fold_slot_zmm!(64 * (t + 3));
                let lh01 = k0_f128_fold_even_odd_zmm(m0, m1, zrr0_0, zrr0_1);
                let lh23 = k0_f128_fold_even_odd_zmm(m2, m3, zrr0_0, zrr0_1);
                let out = k0_f128_fold_even_odd_zmm(lh01, lh23, zrr1_0, zrr1_1);
                _mm512_storeu_si512(dst_ptr.add(2 * t) as *mut __m512i, out);
                t += 4;
            }
        }
        macro_rules! acc1_pre_m128i {
            ($wb:expr, $ws:expr, $s:expr, $ll:ident, $mid:ident, $hh:ident) => {{
                let s = $s;
                let s_sum = _mm_xor_si128(s, _mm_srli_si128::<8>(s));
                $ll = _mm_xor_si128($ll, _mm_clmulepi64_si128($wb, s, 0x00));
                $mid = _mm_xor_si128($mid, _mm_clmulepi64_si128($ws, s_sum, 0x00));
                $hh = _mm_xor_si128($hh, _mm_clmulepi64_si128($wb, s, 0x11));
            }};
        }
        while t < n_dst {
            let base = 64 * t;
            let z = _mm_setzero_si128();
            let (mut ll0, mut mid0, mut hh0) = (z, z, z);
            let (mut ll1, mut mid1, mut hh1) = (z, z, z);
            let (mut ll2, mut mid2, mut hh2) = (z, z, z);
            let (mut ll3, mut mid3, mut hh3) = (z, z, z);
            for b in 0..16 {
                let wb = vw[b];
                let ws = vw_sum[b];
                let ptr = src_ptr.add(2 * (base + (b << 2)));
                acc1_pre_m128i!(
                    wb,
                    ws,
                    _mm_loadu_si128(ptr as *const __m128i),
                    ll0,
                    mid0,
                    hh0
                );
                acc1_pre_m128i!(
                    wb,
                    ws,
                    _mm_loadu_si128(ptr.add(2) as *const __m128i),
                    ll1,
                    mid1,
                    hh1
                );
                acc1_pre_m128i!(
                    wb,
                    ws,
                    _mm_loadu_si128(ptr.add(4) as *const __m128i),
                    ll2,
                    mid2,
                    hh2
                );
                acc1_pre_m128i!(
                    wb,
                    ws,
                    _mm_loadu_si128(ptr.add(6) as *const __m128i),
                    ll3,
                    mid3,
                    hh3
                );
            }
            let m0 = k0_f128_reduce_3lane_m128i(ll0, mid0, hh0);
            let m1 = k0_f128_reduce_3lane_m128i(ll1, mid1, hh1);
            let m2 = k0_f128_reduce_3lane_m128i(ll2, mid2, hh2);
            let m3 = k0_f128_reduce_3lane_m128i(ll3, mid3, hh3);
            let (p0, p2) = k0_f128_mul_const_vec2_m128i(
                _mm_xor_si128(m0, m1),
                _mm_xor_si128(m2, m3),
                rr0_0,
                rr0_1,
                rr0_0,
                rr0_1,
            );
            let low = _mm_xor_si128(m0, p0);
            let high = _mm_xor_si128(m2, p2);
            let q0 = k0_f128_mul_const_m128i(_mm_xor_si128(low, high), rr1_0, rr1_1);
            _mm_storeu_si128(dst_ptr.add(2 * t) as *mut __m128i, _mm_xor_si128(low, q0));
            t += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let mut vw = [z; 16];
        let mut vw_lo = [0u64; 16];
        let mut vw_sum = [0u64; 16];
        for b in 0..16 {
            let wb = vld1q_u64(w_ptr.add(2 * b));
            vw[b] = wb;
            vw_lo[b] = vgetq_lane_u64::<0>(wb);
            vw_sum[b] = vgetq_lane_u64::<0>(veorq_u64(wb, vextq_u64::<1>(wb, wb)));
        }
        let vr0 = vld1q_u64([r0_lo, r0_hi].as_ptr());
        let (rr0_0, rr0_1) = k0_f128_pre_reduce_const_neon(vr0);
        let vr1 = vld1q_u64([r1_lo, r1_hi].as_ptr());
        let (rr1_0, rr1_1) = k0_f128_pre_reduce_const_neon(vr1);
        macro_rules! acc2_w_neon {
            ($w0:expr, $wl0:expr, $ws0:expr, $s0:expr, $w1:expr, $wl1:expr, $ws1:expr, $s1:expr, $ll:ident, $mid:ident, $hh:ident) => {{
                let s0 = $s0;
                let s1 = $s1;
                let s0_sum = veorq_u64(s0, vextq_u64::<1>(s0, s0));
                let s1_sum = veorq_u64(s1, vextq_u64::<1>(s1, s1));
                $ll = k0_xor3_neon(
                    $ll,
                    vreinterpretq_u64_p128(vmull_p64($wl0, vgetq_lane_u64::<0>(s0))),
                    vreinterpretq_u64_p128(vmull_p64($wl1, vgetq_lane_u64::<0>(s1))),
                );
                $mid = k0_xor3_neon(
                    $mid,
                    vreinterpretq_u64_p128(vmull_p64($ws0, vgetq_lane_u64::<0>(s0_sum))),
                    vreinterpretq_u64_p128(vmull_p64($ws1, vgetq_lane_u64::<0>(s1_sum))),
                );
                $hh = k0_xor3_neon(
                    $hh,
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64($w0),
                        vreinterpretq_p64_u64(s0),
                    )),
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64($w1),
                        vreinterpretq_p64_u64(s1),
                    )),
                );
            }};
        }
        macro_rules! fold_slot_neon {
            ($base:expr) => {{
                let (mut ll0, mut mid0, mut hh0) = (z, z, z);
                let (mut ll1, mut mid1, mut hh1) = (z, z, z);
                let (mut ll2, mut mid2, mut hh2) = (z, z, z);
                let (mut ll3, mut mid3, mut hh3) = (z, z, z);
                let mut b = 0usize;
                while b < 16 {
                    let wb0 = vw[b];
                    let wl0 = vw_lo[b];
                    let ws0 = vw_sum[b];
                    let wb1 = vw[b + 1];
                    let wl1 = vw_lo[b + 1];
                    let ws1 = vw_sum[b + 1];
                    let ptr0 = src_ptr.add(2 * ($base + (b << 2)));
                    let ptr1 = src_ptr.add(2 * ($base + ((b + 1) << 2)));
                    acc2_w_neon!(
                        wb0,
                        wl0,
                        ws0,
                        vld1q_u64(ptr0),
                        wb1,
                        wl1,
                        ws1,
                        vld1q_u64(ptr1),
                        ll0,
                        mid0,
                        hh0
                    );
                    acc2_w_neon!(
                        wb0,
                        wl0,
                        ws0,
                        vld1q_u64(ptr0.add(2)),
                        wb1,
                        wl1,
                        ws1,
                        vld1q_u64(ptr1.add(2)),
                        ll1,
                        mid1,
                        hh1
                    );
                    acc2_w_neon!(
                        wb0,
                        wl0,
                        ws0,
                        vld1q_u64(ptr0.add(4)),
                        wb1,
                        wl1,
                        ws1,
                        vld1q_u64(ptr1.add(4)),
                        ll2,
                        mid2,
                        hh2
                    );
                    acc2_w_neon!(
                        wb0,
                        wl0,
                        ws0,
                        vld1q_u64(ptr0.add(6)),
                        wb1,
                        wl1,
                        ws1,
                        vld1q_u64(ptr1.add(6)),
                        ll3,
                        mid3,
                        hh3
                    );
                    b += 2;
                }
                let m0 = k0_f128_reduce_3lane_neon(ll0, mid0, hh0);
                let m1 = k0_f128_reduce_3lane_neon(ll1, mid1, hh1);
                let m2 = k0_f128_reduce_3lane_neon(ll2, mid2, hh2);
                let m3 = k0_f128_reduce_3lane_neon(ll3, mid3, hh3);
                let (p0, p2) = k0_f128_mul_const_vec2_neon(
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
        let mut t = 0usize;
        while t + 2 <= n_dst {
            let (low0, diff0) = fold_slot_neon!(64 * t);
            let (low1, diff1) = fold_slot_neon!(64 * (t + 1));
            let (q0, q1) = k0_f128_mul_const_vec2_neon(diff0, diff1, rr1_0, rr1_1, rr1_0, rr1_1);
            vst1q_u64(dst_ptr.add(2 * t), veorq_u64(low0, q0));
            vst1q_u64(dst_ptr.add(2 * (t + 1)), veorq_u64(low1, q1));
            t += 2;
        }
        while t < n_dst {
            let (low0, diff0) = fold_slot_neon!(64 * t);
            let q0 = k0_f128_mul_const_neon(diff0, rr1_0, rr1_1);
            vst1q_u64(dst_ptr.add(2 * t), veorq_u64(low0, q0));
            t += 1;
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        for t in 0..n_dst {
            let base_t = 64 * t;
            let mut m_lo = [0u64; 4];
            let mut m_hi = [0u64; 4];
            for s_idx in 0..4 {
                let (mut r0, mut r1, mut r2, mut r3) = (0u64, 0u64, 0u64, 0u64);
                for b in 0..16 {
                    let s_off = 2 * (base_t + (b << 2) + s_idx);
                    let (u0, u1, u2, u3) = k0_f128_mul_unreduced_raw(
                        *w_ptr.add(2 * b),
                        *w_ptr.add(2 * b + 1),
                        *src_ptr.add(s_off),
                        *src_ptr.add(s_off + 1),
                    );
                    r0 ^= u0;
                    r1 ^= u1;
                    r2 ^= u2;
                    r3 ^= u3;
                }
                let (ml, mh) = k0_f128_reduce_256_raw(r0, r1, r2, r3);
                m_lo[s_idx] = ml;
                m_hi[s_idx] = mh;
            }
            let (p0_lo, p0_hi, p2_lo, p2_hi) = k0_f128_mul_vec2_raw(
                m_lo[0] ^ m_lo[1],
                m_hi[0] ^ m_hi[1],
                m_lo[2] ^ m_lo[3],
                m_hi[2] ^ m_hi[3],
                r0_lo,
                r0_hi,
                r0_lo,
                r0_hi,
            );
            let low_lo = m_lo[0] ^ p0_lo;
            let low_hi = m_hi[0] ^ p0_hi;
            let high_lo = m_lo[2] ^ p2_lo;
            let high_hi = m_hi[2] ^ p2_hi;
            let (q_lo, q_hi) = k0_f128_mul_raw(low_lo ^ high_lo, low_hi ^ high_hi, r1_lo, r1_hi);
            *dst_ptr.add(2 * t) = low_lo ^ q_lo;
            *dst_ptr.add(2 * t + 1) = low_hi ^ q_hi;
        }
    }
}
#[inline(always)]
pub unsafe fn k0_f128_transpose_butterfly_slice(
    top_ptr: *mut u64,
    bot_ptr: *mut u64,
    n_f128: usize,
    tw_lo: u64,
    tw_hi: u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let mut i = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            if tw_hi == 0 {
                let vtw_lo = _mm512_set1_epi64(tw_lo as i64);
                while i + 4 <= n_f128 {
                    let idx = 2 * i;
                    let a = _mm512_loadu_si512(top_ptr.add(idx) as *const __m512i);
                    let b = _mm512_loadu_si512(bot_ptr.add(idx) as *const __m512i);
                    let s = _mm512_xor_si512(a, b);
                    let p = k0_f128_mul_low_const_zmm(s, vtw_lo);
                    _mm512_storeu_si512(top_ptr.add(idx) as *mut __m512i, s);
                    _mm512_storeu_si512(bot_ptr.add(idx) as *mut __m512i, _mm512_xor_si512(p, b));
                    i += 4;
                }
            } else {
                let vt = _mm_set_epi64x(tw_hi as i64, tw_lo as i64);
                let (tr0, tr1) = k0_f128_pre_reduce_const_m128i(vt);
                let ztr0 = _mm512_broadcast_i32x4(tr0);
                let ztr1 = _mm512_broadcast_i32x4(tr1);
                while i + 4 <= n_f128 {
                    let idx = 2 * i;
                    let a = _mm512_loadu_si512(top_ptr.add(idx) as *const __m512i);
                    let b = _mm512_loadu_si512(bot_ptr.add(idx) as *const __m512i);
                    let s = _mm512_xor_si512(a, b);
                    let p = k0_f128_mul_const_zmm(s, ztr0, ztr1);
                    _mm512_storeu_si512(top_ptr.add(idx) as *mut __m512i, s);
                    _mm512_storeu_si512(bot_ptr.add(idx) as *mut __m512i, _mm512_xor_si512(p, b));
                    i += 4;
                }
            }
        }
        if tw_hi == 0 {
            let vtw_lo = _mm_set1_epi64x(tw_lo as i64);
            while i + 2 <= n_f128 {
                let idx = 2 * i;
                let a0 = _mm_loadu_si128(top_ptr.add(idx) as *const __m128i);
                let b0 = _mm_loadu_si128(bot_ptr.add(idx) as *const __m128i);
                let a1 = _mm_loadu_si128(top_ptr.add(idx + 2) as *const __m128i);
                let b1 = _mm_loadu_si128(bot_ptr.add(idx + 2) as *const __m128i);
                let s0 = _mm_xor_si128(a0, b0);
                let s1 = _mm_xor_si128(a1, b1);
                let (p0, p1) = k0_f128_mul_low_const_vec2_m128i(s0, s1, vtw_lo, vtw_lo);
                _mm_storeu_si128(top_ptr.add(idx) as *mut __m128i, s0);
                _mm_storeu_si128(top_ptr.add(idx + 2) as *mut __m128i, s1);
                _mm_storeu_si128(bot_ptr.add(idx) as *mut __m128i, _mm_xor_si128(p0, b0));
                _mm_storeu_si128(bot_ptr.add(idx + 2) as *mut __m128i, _mm_xor_si128(p1, b1));
                i += 2;
            }
            while i < n_f128 {
                let idx = 2 * i;
                let a = _mm_loadu_si128(top_ptr.add(idx) as *const __m128i);
                let b = _mm_loadu_si128(bot_ptr.add(idx) as *const __m128i);
                let s = _mm_xor_si128(a, b);
                let p = k0_f128_mul_low_const_m128i(s, vtw_lo);
                _mm_storeu_si128(top_ptr.add(idx) as *mut __m128i, s);
                _mm_storeu_si128(bot_ptr.add(idx) as *mut __m128i, _mm_xor_si128(p, b));
                i += 1;
            }
        } else {
            let vt = _mm_set_epi64x(tw_hi as i64, tw_lo as i64);
            let (tr0, tr1) = k0_f128_pre_reduce_const_m128i(vt);
            while i + 2 <= n_f128 {
                let idx = 2 * i;
                let a0 = _mm_loadu_si128(top_ptr.add(idx) as *const __m128i);
                let b0 = _mm_loadu_si128(bot_ptr.add(idx) as *const __m128i);
                let a1 = _mm_loadu_si128(top_ptr.add(idx + 2) as *const __m128i);
                let b1 = _mm_loadu_si128(bot_ptr.add(idx + 2) as *const __m128i);
                let s0 = _mm_xor_si128(a0, b0);
                let s1 = _mm_xor_si128(a1, b1);
                let (p0, p1) = k0_f128_mul_const_vec2_m128i(s0, s1, tr0, tr1, tr0, tr1);
                _mm_storeu_si128(top_ptr.add(idx) as *mut __m128i, s0);
                _mm_storeu_si128(top_ptr.add(idx + 2) as *mut __m128i, s1);
                _mm_storeu_si128(bot_ptr.add(idx) as *mut __m128i, _mm_xor_si128(p0, b0));
                _mm_storeu_si128(bot_ptr.add(idx + 2) as *mut __m128i, _mm_xor_si128(p1, b1));
                i += 2;
            }
            while i < n_f128 {
                let idx = 2 * i;
                let a = _mm_loadu_si128(top_ptr.add(idx) as *const __m128i);
                let b = _mm_loadu_si128(bot_ptr.add(idx) as *const __m128i);
                let s = _mm_xor_si128(a, b);
                let p = k0_f128_mul_const_m128i(s, tr0, tr1);
                _mm_storeu_si128(top_ptr.add(idx) as *mut __m128i, s);
                _mm_storeu_si128(bot_ptr.add(idx) as *mut __m128i, _mm_xor_si128(p, b));
                i += 1;
            }
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let mut i = 0usize;
        if tw_hi == 0 {
            while i + 2 <= n_f128 {
                let idx = 2 * i;
                let a0 = vld1q_u64(top_ptr.add(idx));
                let b0 = vld1q_u64(bot_ptr.add(idx));
                let a1 = vld1q_u64(top_ptr.add(idx + 2));
                let b1 = vld1q_u64(bot_ptr.add(idx + 2));
                let s0 = veorq_u64(a0, b0);
                let s1 = veorq_u64(a1, b1);
                let (p0, p1) = k0_f128_mul_low_const_vec2_neon(s0, s1, tw_lo, tw_lo);
                vst1q_u64(top_ptr.add(idx), s0);
                vst1q_u64(top_ptr.add(idx + 2), s1);
                vst1q_u64(bot_ptr.add(idx), veorq_u64(p0, b0));
                vst1q_u64(bot_ptr.add(idx + 2), veorq_u64(p1, b1));
                i += 2;
            }
            while i < n_f128 {
                let idx = 2 * i;
                let a = vld1q_u64(top_ptr.add(idx));
                let b = vld1q_u64(bot_ptr.add(idx));
                let s = veorq_u64(a, b);
                let p = k0_f128_mul_low_const_neon(s, tw_lo);
                vst1q_u64(top_ptr.add(idx), s);
                vst1q_u64(bot_ptr.add(idx), veorq_u64(p, b));
                i += 1;
            }
        } else {
            let vt = vld1q_u64([tw_lo, tw_hi].as_ptr());
            let (tr0, tr1) = k0_f128_pre_reduce_const_neon(vt);
            while i + 2 <= n_f128 {
                let idx = 2 * i;
                let a0 = vld1q_u64(top_ptr.add(idx));
                let b0 = vld1q_u64(bot_ptr.add(idx));
                let a1 = vld1q_u64(top_ptr.add(idx + 2));
                let b1 = vld1q_u64(bot_ptr.add(idx + 2));
                let s0 = veorq_u64(a0, b0);
                let s1 = veorq_u64(a1, b1);
                let (p0, p1) = k0_f128_mul_const_vec2_neon(s0, s1, tr0, tr1, tr0, tr1);
                vst1q_u64(top_ptr.add(idx), s0);
                vst1q_u64(top_ptr.add(idx + 2), s1);
                vst1q_u64(bot_ptr.add(idx), veorq_u64(p0, b0));
                vst1q_u64(bot_ptr.add(idx + 2), veorq_u64(p1, b1));
                i += 2;
            }
            while i < n_f128 {
                let idx = 2 * i;
                let a = vld1q_u64(top_ptr.add(idx));
                let b = vld1q_u64(bot_ptr.add(idx));
                let s = veorq_u64(a, b);
                let p = k0_f128_mul_const_neon(s, tr0, tr1);
                vst1q_u64(top_ptr.add(idx), s);
                vst1q_u64(bot_ptr.add(idx), veorq_u64(p, b));
                i += 1;
            }
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        for i in 0..n_f128 {
            let idx = 2 * i;
            let a_lo = *top_ptr.add(idx);
            let a_hi = *top_ptr.add(idx + 1);
            let b_lo = *bot_ptr.add(idx);
            let b_hi = *bot_ptr.add(idx + 1);
            let s_lo = a_lo ^ b_lo;
            let s_hi = a_hi ^ b_hi;
            let (p_lo, p_hi) = k0_f128_mul_raw(s_lo, s_hi, tw_lo, tw_hi);
            *top_ptr.add(idx) = s_lo;
            *top_ptr.add(idx + 1) = s_hi;
            *bot_ptr.add(idx) = p_lo ^ b_lo;
            *bot_ptr.add(idx + 1) = p_hi ^ b_hi;
        }
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_transpose_butterfly_fused_2layer_slice(
    q0_ptr: *mut u64,
    q1_ptr: *mut u64,
    q2_ptr: *mut u64,
    q3_ptr: *mut u64,
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
        let mut i = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            if (to_hi | tia_hi | tib_hi) == 0 {
                let vto = _mm512_set1_epi64(to_lo as i64);
                let vtia = _mm512_set1_epi64(tia_lo as i64);
                let vtib = _mm512_set1_epi64(tib_lo as i64);
                while i + 4 <= n_f128 {
                    let idx = 2 * i;
                    let x0 = _mm512_loadu_si512(q0_ptr.add(idx) as *const __m512i);
                    let x1 = _mm512_loadu_si512(q1_ptr.add(idx) as *const __m512i);
                    let x2 = _mm512_loadu_si512(q2_ptr.add(idx) as *const __m512i);
                    let x3 = _mm512_loadu_si512(q3_ptr.add(idx) as *const __m512i);
                    let s01 = _mm512_xor_si512(x0, x1);
                    let s23 = _mm512_xor_si512(x2, x3);
                    let m01 = k0_f128_mul_low_const_zmm(s01, vtia);
                    let m23 = k0_f128_mul_low_const_zmm(s23, vtib);
                    let y1 = _mm512_xor_si512(m01, x1);
                    let y3 = _mm512_xor_si512(m23, x3);
                    let z0 = _mm512_xor_si512(s01, s23);
                    let z1 = _mm512_xor_si512(y1, y3);
                    let m02 = k0_f128_mul_low_const_zmm(z0, vto);
                    let m13 = k0_f128_mul_low_const_zmm(z1, vto);
                    _mm512_storeu_si512(q0_ptr.add(idx) as *mut __m512i, z0);
                    _mm512_storeu_si512(q1_ptr.add(idx) as *mut __m512i, z1);
                    _mm512_storeu_si512(
                        q2_ptr.add(idx) as *mut __m512i,
                        _mm512_xor_si512(m02, s23),
                    );
                    _mm512_storeu_si512(q3_ptr.add(idx) as *mut __m512i, _mm512_xor_si512(m13, y3));
                    i += 4;
                }
            } else {
                let (to0, to1) = k0_f128_pre_reduce_const_zmm(_mm512_broadcast_i32x4(
                    _mm_set_epi64x(to_hi as i64, to_lo as i64),
                ));
                let (tia0, tia1) = k0_f128_pre_reduce_const_zmm(_mm512_broadcast_i32x4(
                    _mm_set_epi64x(tia_hi as i64, tia_lo as i64),
                ));
                let (tib0, tib1) = k0_f128_pre_reduce_const_zmm(_mm512_broadcast_i32x4(
                    _mm_set_epi64x(tib_hi as i64, tib_lo as i64),
                ));
                while i + 4 <= n_f128 {
                    let idx = 2 * i;
                    let x0 = _mm512_loadu_si512(q0_ptr.add(idx) as *const __m512i);
                    let x1 = _mm512_loadu_si512(q1_ptr.add(idx) as *const __m512i);
                    let x2 = _mm512_loadu_si512(q2_ptr.add(idx) as *const __m512i);
                    let x3 = _mm512_loadu_si512(q3_ptr.add(idx) as *const __m512i);
                    let s01 = _mm512_xor_si512(x0, x1);
                    let s23 = _mm512_xor_si512(x2, x3);
                    let m01 = k0_f128_mul_const_zmm(s01, tia0, tia1);
                    let m23 = k0_f128_mul_const_zmm(s23, tib0, tib1);
                    let y1 = _mm512_xor_si512(m01, x1);
                    let y3 = _mm512_xor_si512(m23, x3);
                    let z0 = _mm512_xor_si512(s01, s23);
                    let z1 = _mm512_xor_si512(y1, y3);
                    let m02 = k0_f128_mul_const_zmm(z0, to0, to1);
                    let m13 = k0_f128_mul_const_zmm(z1, to0, to1);
                    _mm512_storeu_si512(q0_ptr.add(idx) as *mut __m512i, z0);
                    _mm512_storeu_si512(q1_ptr.add(idx) as *mut __m512i, z1);
                    _mm512_storeu_si512(
                        q2_ptr.add(idx) as *mut __m512i,
                        _mm512_xor_si512(m02, s23),
                    );
                    _mm512_storeu_si512(q3_ptr.add(idx) as *mut __m512i, _mm512_xor_si512(m13, y3));
                    i += 4;
                }
            }
        }
        if (to_hi | tia_hi | tib_hi) == 0 {
            let vto = _mm_set1_epi64x(to_lo as i64);
            let vtia = _mm_set1_epi64x(tia_lo as i64);
            let vtib = _mm_set1_epi64x(tib_lo as i64);
            while i < n_f128 {
                let idx = 2 * i;
                let x0 = _mm_loadu_si128(q0_ptr.add(idx) as *const __m128i);
                let x1 = _mm_loadu_si128(q1_ptr.add(idx) as *const __m128i);
                let x2 = _mm_loadu_si128(q2_ptr.add(idx) as *const __m128i);
                let x3 = _mm_loadu_si128(q3_ptr.add(idx) as *const __m128i);
                let s01 = _mm_xor_si128(x0, x1);
                let s23 = _mm_xor_si128(x2, x3);
                let (m01, m23) = k0_f128_mul_low_const_vec2_m128i(s01, s23, vtia, vtib);
                let y1 = _mm_xor_si128(m01, x1);
                let y3 = _mm_xor_si128(m23, x3);
                let z0 = _mm_xor_si128(s01, s23);
                let z1 = _mm_xor_si128(y1, y3);
                let (m02, m13) = k0_f128_mul_low_const_vec2_m128i(z0, z1, vto, vto);
                _mm_storeu_si128(q0_ptr.add(idx) as *mut __m128i, z0);
                _mm_storeu_si128(q1_ptr.add(idx) as *mut __m128i, z1);
                _mm_storeu_si128(q2_ptr.add(idx) as *mut __m128i, _mm_xor_si128(m02, s23));
                _mm_storeu_si128(q3_ptr.add(idx) as *mut __m128i, _mm_xor_si128(m13, y3));
                i += 1;
            }
        } else {
            let (to0, to1) =
                k0_f128_pre_reduce_const_m128i(_mm_set_epi64x(to_hi as i64, to_lo as i64));
            let (tia0, tia1) =
                k0_f128_pre_reduce_const_m128i(_mm_set_epi64x(tia_hi as i64, tia_lo as i64));
            let (tib0, tib1) =
                k0_f128_pre_reduce_const_m128i(_mm_set_epi64x(tib_hi as i64, tib_lo as i64));
            while i < n_f128 {
                let idx = 2 * i;
                let x0 = _mm_loadu_si128(q0_ptr.add(idx) as *const __m128i);
                let x1 = _mm_loadu_si128(q1_ptr.add(idx) as *const __m128i);
                let x2 = _mm_loadu_si128(q2_ptr.add(idx) as *const __m128i);
                let x3 = _mm_loadu_si128(q3_ptr.add(idx) as *const __m128i);
                let s01 = _mm_xor_si128(x0, x1);
                let s23 = _mm_xor_si128(x2, x3);
                let (m01, m23) = k0_f128_mul_const_vec2_m128i(s01, s23, tia0, tia1, tib0, tib1);
                let y1 = _mm_xor_si128(m01, x1);
                let y3 = _mm_xor_si128(m23, x3);
                let z0 = _mm_xor_si128(s01, s23);
                let z1 = _mm_xor_si128(y1, y3);
                let (m02, m13) = k0_f128_mul_const_vec2_m128i(z0, z1, to0, to1, to0, to1);
                _mm_storeu_si128(q0_ptr.add(idx) as *mut __m128i, z0);
                _mm_storeu_si128(q1_ptr.add(idx) as *mut __m128i, z1);
                _mm_storeu_si128(q2_ptr.add(idx) as *mut __m128i, _mm_xor_si128(m02, s23));
                _mm_storeu_si128(q3_ptr.add(idx) as *mut __m128i, _mm_xor_si128(m13, y3));
                i += 1;
            }
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        if (to_hi | tia_hi | tib_hi) == 0 {
            for i in 0..n_f128 {
                let idx = 2 * i;
                let x0 = vld1q_u64(q0_ptr.add(idx));
                let x1 = vld1q_u64(q1_ptr.add(idx));
                let x2 = vld1q_u64(q2_ptr.add(idx));
                let x3 = vld1q_u64(q3_ptr.add(idx));
                let s01 = veorq_u64(x0, x1);
                let s23 = veorq_u64(x2, x3);
                let (m01, m23) = k0_f128_mul_low_const_vec2_neon(s01, s23, tia_lo, tib_lo);
                let y1 = veorq_u64(m01, x1);
                let y3 = veorq_u64(m23, x3);
                let z0 = veorq_u64(s01, s23);
                let z1 = veorq_u64(y1, y3);
                let (m02, m13) = k0_f128_mul_low_const_vec2_neon(z0, z1, to_lo, to_lo);
                vst1q_u64(q0_ptr.add(idx), z0);
                vst1q_u64(q1_ptr.add(idx), z1);
                vst1q_u64(q2_ptr.add(idx), veorq_u64(m02, s23));
                vst1q_u64(q3_ptr.add(idx), veorq_u64(m13, y3));
            }
        } else {
            let (to0, to1) = k0_f128_pre_reduce_const_neon(vld1q_u64([to_lo, to_hi].as_ptr()));
            let (tia0, tia1) = k0_f128_pre_reduce_const_neon(vld1q_u64([tia_lo, tia_hi].as_ptr()));
            let (tib0, tib1) = k0_f128_pre_reduce_const_neon(vld1q_u64([tib_lo, tib_hi].as_ptr()));
            for i in 0..n_f128 {
                let idx = 2 * i;
                let x0 = vld1q_u64(q0_ptr.add(idx));
                let x1 = vld1q_u64(q1_ptr.add(idx));
                let x2 = vld1q_u64(q2_ptr.add(idx));
                let x3 = vld1q_u64(q3_ptr.add(idx));
                let s01 = veorq_u64(x0, x1);
                let s23 = veorq_u64(x2, x3);
                let (m01, m23) = k0_f128_mul_const_vec2_neon(s01, s23, tia0, tia1, tib0, tib1);
                let y1 = veorq_u64(m01, x1);
                let y3 = veorq_u64(m23, x3);
                let z0 = veorq_u64(s01, s23);
                let z1 = veorq_u64(y1, y3);
                let (m02, m13) = k0_f128_mul_const_vec2_neon(z0, z1, to0, to1, to0, to1);
                vst1q_u64(q0_ptr.add(idx), z0);
                vst1q_u64(q1_ptr.add(idx), z1);
                vst1q_u64(q2_ptr.add(idx), veorq_u64(m02, s23));
                vst1q_u64(q3_ptr.add(idx), veorq_u64(m13, y3));
            }
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        for i in 0..n_f128 {
            let idx = 2 * i;
            let x0_lo = *q0_ptr.add(idx);
            let x0_hi = *q0_ptr.add(idx + 1);
            let x1_lo = *q1_ptr.add(idx);
            let x1_hi = *q1_ptr.add(idx + 1);
            let x2_lo = *q2_ptr.add(idx);
            let x2_hi = *q2_ptr.add(idx + 1);
            let x3_lo = *q3_ptr.add(idx);
            let x3_hi = *q3_ptr.add(idx + 1);
            let s01_lo = x0_lo ^ x1_lo;
            let s01_hi = x0_hi ^ x1_hi;
            let s23_lo = x2_lo ^ x3_lo;
            let s23_hi = x2_hi ^ x3_hi;
            let (m01_lo, m01_hi, m23_lo, m23_hi) = k0_f128_mul_vec2_raw(
                s01_lo, s01_hi, s23_lo, s23_hi, tia_lo, tia_hi, tib_lo, tib_hi,
            );
            let y1_lo = m01_lo ^ x1_lo;
            let y1_hi = m01_hi ^ x1_hi;
            let y3_lo = m23_lo ^ x3_lo;
            let y3_hi = m23_hi ^ x3_hi;
            let z0_lo = s01_lo ^ s23_lo;
            let z0_hi = s01_hi ^ s23_hi;
            let z1_lo = y1_lo ^ y3_lo;
            let z1_hi = y1_hi ^ y3_hi;
            let (m02_lo, m02_hi, m13_lo, m13_hi) =
                k0_f128_mul_vec2_raw(z0_lo, z0_hi, z1_lo, z1_hi, to_lo, to_hi, to_lo, to_hi);
            *q0_ptr.add(idx) = z0_lo;
            *q0_ptr.add(idx + 1) = z0_hi;
            *q1_ptr.add(idx) = z1_lo;
            *q1_ptr.add(idx + 1) = z1_hi;
            *q2_ptr.add(idx) = m02_lo ^ s23_lo;
            *q2_ptr.add(idx + 1) = m02_hi ^ s23_hi;
            *q3_ptr.add(idx) = m13_lo ^ y3_lo;
            *q3_ptr.add(idx + 1) = m13_hi ^ y3_hi;
        }
    }
}
#[inline(always)]
pub unsafe fn k0_f128_eq_expand_slice(
    lo_ptr: *mut u64,
    hi_ptr: *mut u64,
    n_f128: usize,
    r_lo: u64,
    r_hi: u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let vr = _mm_set_epi64x(r_hi as i64, r_lo as i64);
        let (rr0, rr1) = k0_f128_pre_reduce_const_m128i(vr);
        let mut x = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let zrr0 = _mm512_broadcast_i32x4(rr0);
            let zrr1 = _mm512_broadcast_i32x4(rr1);
            while x + 4 <= n_f128 {
                let idx = 2 * x;
                let old = _mm512_loadu_si512(lo_ptr.add(idx) as *const __m512i);
                let h = k0_f128_mul_const_zmm(old, zrr0, zrr1);
                _mm512_storeu_si512(hi_ptr.add(idx) as *mut __m512i, h);
                _mm512_storeu_si512(lo_ptr.add(idx) as *mut __m512i, _mm512_xor_si512(old, h));
                x += 4;
            }
        }
        while x + 2 <= n_f128 {
            let idx = 2 * x;
            let old0 = _mm_loadu_si128(lo_ptr.add(idx) as *const __m128i);
            let old1 = _mm_loadu_si128(lo_ptr.add(idx + 2) as *const __m128i);
            let (h0, h1) = k0_f128_mul_const_vec2_m128i(old0, old1, rr0, rr1, rr0, rr1);
            _mm_storeu_si128(hi_ptr.add(idx) as *mut __m128i, h0);
            _mm_storeu_si128(hi_ptr.add(idx + 2) as *mut __m128i, h1);
            _mm_storeu_si128(lo_ptr.add(idx) as *mut __m128i, _mm_xor_si128(old0, h0));
            _mm_storeu_si128(lo_ptr.add(idx + 2) as *mut __m128i, _mm_xor_si128(old1, h1));
            x += 2;
        }
        while x < n_f128 {
            let idx = 2 * x;
            let old0 = _mm_loadu_si128(lo_ptr.add(idx) as *const __m128i);
            let h0 = k0_f128_mul_const_m128i(old0, rr0, rr1);
            _mm_storeu_si128(hi_ptr.add(idx) as *mut __m128i, h0);
            _mm_storeu_si128(lo_ptr.add(idx) as *mut __m128i, _mm_xor_si128(old0, h0));
            x += 1;
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let vr = vld1q_u64([r_lo, r_hi].as_ptr());
        let (rr0, rr1) = k0_f128_pre_reduce_const_neon(vr);
        let mut x = 0usize;
        while x + 2 <= n_f128 {
            let idx = 2 * x;
            let old0 = vld1q_u64(lo_ptr.add(idx));
            let old1 = vld1q_u64(lo_ptr.add(idx + 2));
            let (h0, h1) = k0_f128_mul_const_vec2_neon(old0, old1, rr0, rr1, rr0, rr1);
            vst1q_u64(hi_ptr.add(idx), h0);
            vst1q_u64(hi_ptr.add(idx + 2), h1);
            vst1q_u64(lo_ptr.add(idx), veorq_u64(old0, h0));
            vst1q_u64(lo_ptr.add(idx + 2), veorq_u64(old1, h1));
            x += 2;
        }
        while x < n_f128 {
            let idx = 2 * x;
            let old0 = vld1q_u64(lo_ptr.add(idx));
            let h0 = k0_f128_mul_const_neon(old0, rr0, rr1);
            vst1q_u64(hi_ptr.add(idx), h0);
            vst1q_u64(lo_ptr.add(idx), veorq_u64(old0, h0));
            x += 1;
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        for x in 0..n_f128 {
            let idx = 2 * x;
            let o_lo = *lo_ptr.add(idx);
            let o_hi = *lo_ptr.add(idx + 1);
            let (h_lo, h_hi) = k0_f128_mul_raw(o_lo, o_hi, r_lo, r_hi);
            *hi_ptr.add(idx) = h_lo;
            *hi_ptr.add(idx + 1) = h_hi;
            *lo_ptr.add(idx) = o_lo ^ h_lo;
            *lo_ptr.add(idx + 1) = o_hi ^ h_hi;
        }
    }
}
#[inline(always)]
pub unsafe fn k0_f128_round_msg_lsb_slice(
    f_ptr: *const u64,
    b_ptr: *const u64,
    n_f128: usize,
) -> (u64, u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
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
        let mut k = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let idx_dup_even = _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0);
            let (mut z_ll, mut z_mid, mut z_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            while k + 8 <= n_f128 {
                let idx = 2 * k;
                let vf0 = _mm512_loadu_si512(f_ptr.add(idx) as *const __m512i);
                let vb0 = _mm512_loadu_si512(b_ptr.add(idx) as *const __m512i);
                let vf1 = _mm512_loadu_si512(f_ptr.add(idx + 8) as *const __m512i);
                let vb1 = _mm512_loadu_si512(b_ptr.add(idx + 8) as *const __m512i);
                let vf0_p =
                    _mm512_xor_si512(vf0, _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, vf0));
                let vb0_p =
                    _mm512_xor_si512(vb0, _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, vb0));
                let vf1_p =
                    _mm512_xor_si512(vf1, _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, vf1));
                let vb1_p =
                    _mm512_xor_si512(vb1, _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, vb1));
                k0_f128_mul_acc2_zmm(vf0_p, vb0_p, vf1_p, vb1_p, &mut z_ll, &mut z_mid, &mut z_hh);
                k += 8;
            }
            while k + 4 <= n_f128 {
                let idx = 2 * k;
                let vf0 = _mm512_loadu_si512(f_ptr.add(idx) as *const __m512i);
                let vb0 = _mm512_loadu_si512(b_ptr.add(idx) as *const __m512i);
                let vf0_p =
                    _mm512_xor_si512(vf0, _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, vf0));
                let vb0_p =
                    _mm512_xor_si512(vb0, _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, vb0));
                k0_f128_mul_acc_zmm(vf0_p, vb0_p, &mut z_ll, &mut z_mid, &mut z_hh);
                k += 4;
            }
            let ll_256 = _mm256_xor_si256(
                _mm512_castsi512_si256(z_ll),
                _mm512_extracti64x4_epi64::<1>(z_ll),
            );
            let mid_256 = _mm256_xor_si256(
                _mm512_castsi512_si256(z_mid),
                _mm512_extracti64x4_epi64::<1>(z_mid),
            );
            let hh_256 = _mm256_xor_si256(
                _mm512_castsi512_si256(z_hh),
                _mm512_extracti64x4_epi64::<1>(z_hh),
            );
            u0_ll = _mm256_castsi256_si128(ll_256);
            u0_mid = _mm256_castsi256_si128(mid_256);
            u0_hh = _mm256_castsi256_si128(hh_256);
            u2_ll = _mm256_extracti128_si256::<1>(ll_256);
            u2_mid = _mm256_extracti128_si256::<1>(mid_256);
            u2_hh = _mm256_extracti128_si256::<1>(hh_256);
        }
        while k + 1 < n_f128 {
            let idx = 2 * k;
            let f0 = _mm_loadu_si128(f_ptr.add(idx) as *const __m128i);
            let f1 = _mm_loadu_si128(f_ptr.add(idx + 2) as *const __m128i);
            let b0 = _mm_loadu_si128(b_ptr.add(idx) as *const __m128i);
            let b1 = _mm_loadu_si128(b_ptr.add(idx + 2) as *const __m128i);
            k0_f128_mul_acc_m128i(f0, b0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            k0_f128_mul_acc_m128i(
                _mm_xor_si128(f0, f1),
                _mm_xor_si128(b0, b1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            k += 2;
        }
        let r0 = k0_f128_reduce_3lane_m128i(u0_ll, u0_mid, u0_hh);
        let r2 = k0_f128_reduce_3lane_m128i(u2_ll, u2_mid, u2_hh);
        return (
            _mm_extract_epi64::<0>(r0) as u64,
            _mm_extract_epi64::<1>(r0) as u64,
            _mm_extract_epi64::<0>(r2) as u64,
            _mm_extract_epi64::<1>(r2) as u64,
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let (mut u0_ll, mut u0_mid, mut u0_hh) = (z, z, z);
        let (mut u2_ll, mut u2_mid, mut u2_hh) = (z, z, z);
        let mut k = 0usize;
        while k + 3 < n_f128 {
            let idx = 2 * k;
            let f0 = vld1q_u64(f_ptr.add(idx));
            let f1 = vld1q_u64(f_ptr.add(idx + 2));
            let f2 = vld1q_u64(f_ptr.add(idx + 4));
            let f3 = vld1q_u64(f_ptr.add(idx + 6));
            let b0 = vld1q_u64(b_ptr.add(idx));
            let b1 = vld1q_u64(b_ptr.add(idx + 2));
            let b2 = vld1q_u64(b_ptr.add(idx + 4));
            let b3 = vld1q_u64(b_ptr.add(idx + 6));
            k0_f128_mul_acc2_neon(f0, b0, f2, b2, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            k0_f128_mul_acc2_neon(
                veorq_u64(f0, f1),
                veorq_u64(b0, b1),
                veorq_u64(f2, f3),
                veorq_u64(b2, b3),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            k += 4;
        }
        while k + 1 < n_f128 {
            let idx = 2 * k;
            let f0 = vld1q_u64(f_ptr.add(idx));
            let f1 = vld1q_u64(f_ptr.add(idx + 2));
            let b0 = vld1q_u64(b_ptr.add(idx));
            let b1 = vld1q_u64(b_ptr.add(idx + 2));
            k0_f128_mul_acc_neon(f0, b0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            k0_f128_mul_acc_neon(
                veorq_u64(f0, f1),
                veorq_u64(b0, b1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            k += 2;
        }
        let r0 = k0_f128_reduce_3lane_neon(u0_ll, u0_mid, u0_hh);
        let r2 = k0_f128_reduce_3lane_neon(u2_ll, u2_mid, u2_hh);
        return (
            vgetq_lane_u64::<0>(r0),
            vgetq_lane_u64::<1>(r0),
            vgetq_lane_u64::<0>(r2),
            vgetq_lane_u64::<1>(r2),
        );
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let (mut u0_0, mut u0_1, mut u0_2, mut u0_3) = (0u64, 0u64, 0u64, 0u64);
        let (mut u2_0, mut u2_1, mut u2_2, mut u2_3) = (0u64, 0u64, 0u64, 0u64);
        let mut k = 0usize;
        while k + 1 < n_f128 {
            let idx = 2 * k;
            let f0_lo = *f_ptr.add(idx);
            let f0_hi = *f_ptr.add(idx + 1);
            let f1_lo = *f_ptr.add(idx + 2);
            let f1_hi = *f_ptr.add(idx + 3);
            let b0_lo = *b_ptr.add(idx);
            let b0_hi = *b_ptr.add(idx + 1);
            let b1_lo = *b_ptr.add(idx + 2);
            let b1_hi = *b_ptr.add(idx + 3);
            let (a0, a1, a2, a3) = k0_f128_mul_unreduced_raw(f0_lo, f0_hi, b0_lo, b0_hi);
            let (c0, c1, c2, c3) = k0_f128_mul_unreduced_raw(
                f0_lo ^ f1_lo,
                f0_hi ^ f1_hi,
                b0_lo ^ b1_lo,
                b0_hi ^ b1_hi,
            );
            u0_0 ^= a0;
            u0_1 ^= a1;
            u0_2 ^= a2;
            u0_3 ^= a3;
            u2_0 ^= c0;
            u2_1 ^= c1;
            u2_2 ^= c2;
            u2_3 ^= c3;
            k += 2;
        }
        let (r0_lo, r0_hi) = k0_f128_reduce_256_raw(u0_0, u0_1, u0_2, u0_3);
        let (r2_lo, r2_hi) = k0_f128_reduce_256_raw(u2_0, u2_1, u2_2, u2_3);
        (r0_lo, r0_hi, r2_lo, r2_hi)
    }
}
#[inline(always)]
pub unsafe fn k0_f128_round_msg_and_eval_lsb_slice(
    f_ptr: *const u64,
    b_ptr: *const u64,
    n_f128: usize,
) -> (u64, u64, u64, u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
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
        let (mut y1_ll, mut y1_mid, mut y1_hh) = (
            _mm_setzero_si128(),
            _mm_setzero_si128(),
            _mm_setzero_si128(),
        );
        let mut k = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
            let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
            let (mut z01_ll, mut z01_mid, mut z01_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            let (mut z2_ll, mut z2_mid, mut z2_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            while k + 8 <= n_f128 {
                let idx = 2 * k;
                let vf0 = _mm512_loadu_si512(f_ptr.add(idx) as *const __m512i);
                let vb0 = _mm512_loadu_si512(b_ptr.add(idx) as *const __m512i);
                let vf1 = _mm512_loadu_si512(f_ptr.add(idx + 8) as *const __m512i);
                let vb1 = _mm512_loadu_si512(b_ptr.add(idx + 8) as *const __m512i);
                k0_f128_mul_acc2_zmm(vf0, vb0, vf1, vb1, &mut z01_ll, &mut z01_mid, &mut z01_hh);
                let fe = _mm512_permutex2var_epi64(vf0, even_idx, vf1);
                let fo = _mm512_permutex2var_epi64(vf0, odd_idx, vf1);
                let be = _mm512_permutex2var_epi64(vb0, even_idx, vb1);
                let bo = _mm512_permutex2var_epi64(vb0, odd_idx, vb1);
                k0_f128_mul_acc_zmm(
                    _mm512_xor_si512(fe, fo),
                    _mm512_xor_si512(be, bo),
                    &mut z2_ll,
                    &mut z2_mid,
                    &mut z2_hh,
                );
                k += 8;
            }
            let ll01_256 = _mm256_xor_si256(
                _mm512_castsi512_si256(z01_ll),
                _mm512_extracti64x4_epi64::<1>(z01_ll),
            );
            let mid01_256 = _mm256_xor_si256(
                _mm512_castsi512_si256(z01_mid),
                _mm512_extracti64x4_epi64::<1>(z01_mid),
            );
            let hh01_256 = _mm256_xor_si256(
                _mm512_castsi512_si256(z01_hh),
                _mm512_extracti64x4_epi64::<1>(z01_hh),
            );
            u0_ll = _mm256_castsi256_si128(ll01_256);
            u0_mid = _mm256_castsi256_si128(mid01_256);
            u0_hh = _mm256_castsi256_si128(hh01_256);
            y1_ll = _mm256_extracti128_si256::<1>(ll01_256);
            y1_mid = _mm256_extracti128_si256::<1>(mid01_256);
            y1_hh = _mm256_extracti128_si256::<1>(hh01_256);
            u2_ll = k0_f128_hsum_zmm(z2_ll);
            u2_mid = k0_f128_hsum_zmm(z2_mid);
            u2_hh = k0_f128_hsum_zmm(z2_hh);
        }
        while k + 1 < n_f128 {
            let idx = 2 * k;
            let f0 = _mm_loadu_si128(f_ptr.add(idx) as *const __m128i);
            let f1 = _mm_loadu_si128(f_ptr.add(idx + 2) as *const __m128i);
            let b0 = _mm_loadu_si128(b_ptr.add(idx) as *const __m128i);
            let b1 = _mm_loadu_si128(b_ptr.add(idx + 2) as *const __m128i);
            k0_f128_mul_acc_m128i(f0, b0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            k0_f128_mul_acc_m128i(
                _mm_xor_si128(f0, f1),
                _mm_xor_si128(b0, b1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            k0_f128_mul_acc_m128i(f1, b1, &mut y1_ll, &mut y1_mid, &mut y1_hh);
            k += 2;
        }
        let r0 = k0_f128_reduce_3lane_m128i(u0_ll, u0_mid, u0_hh);
        let r2 = k0_f128_reduce_3lane_m128i(u2_ll, u2_mid, u2_hh);
        let ry = k0_f128_reduce_3lane_m128i(y1_ll, y1_mid, y1_hh);
        return (
            _mm_extract_epi64::<0>(r0) as u64,
            _mm_extract_epi64::<1>(r0) as u64,
            _mm_extract_epi64::<0>(r2) as u64,
            _mm_extract_epi64::<1>(r2) as u64,
            _mm_extract_epi64::<0>(ry) as u64,
            _mm_extract_epi64::<1>(ry) as u64,
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let (mut u0_ll, mut u0_mid, mut u0_hh) = (z, z, z);
        let (mut u2_ll, mut u2_mid, mut u2_hh) = (z, z, z);
        let (mut y1_ll, mut y1_mid, mut y1_hh) = (z, z, z);
        let mut k = 0usize;
        while k + 3 < n_f128 {
            let idx = 2 * k;
            let f0 = vld1q_u64(f_ptr.add(idx));
            let f1 = vld1q_u64(f_ptr.add(idx + 2));
            let f2 = vld1q_u64(f_ptr.add(idx + 4));
            let f3 = vld1q_u64(f_ptr.add(idx + 6));
            let b0 = vld1q_u64(b_ptr.add(idx));
            let b1 = vld1q_u64(b_ptr.add(idx + 2));
            let b2 = vld1q_u64(b_ptr.add(idx + 4));
            let b3 = vld1q_u64(b_ptr.add(idx + 6));
            k0_f128_mul_acc2_neon(f0, b0, f2, b2, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            k0_f128_mul_acc2_neon(
                veorq_u64(f0, f1),
                veorq_u64(b0, b1),
                veorq_u64(f2, f3),
                veorq_u64(b2, b3),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            k0_f128_mul_acc2_neon(f1, b1, f3, b3, &mut y1_ll, &mut y1_mid, &mut y1_hh);
            k += 4;
        }
        while k + 1 < n_f128 {
            let idx = 2 * k;
            let f0 = vld1q_u64(f_ptr.add(idx));
            let f1 = vld1q_u64(f_ptr.add(idx + 2));
            let b0 = vld1q_u64(b_ptr.add(idx));
            let b1 = vld1q_u64(b_ptr.add(idx + 2));
            k0_f128_mul_acc_neon(f0, b0, &mut u0_ll, &mut u0_mid, &mut u0_hh);
            k0_f128_mul_acc_neon(
                veorq_u64(f0, f1),
                veorq_u64(b0, b1),
                &mut u2_ll,
                &mut u2_mid,
                &mut u2_hh,
            );
            k0_f128_mul_acc_neon(f1, b1, &mut y1_ll, &mut y1_mid, &mut y1_hh);
            k += 2;
        }
        let r0 = k0_f128_reduce_3lane_neon(u0_ll, u0_mid, u0_hh);
        let r2 = k0_f128_reduce_3lane_neon(u2_ll, u2_mid, u2_hh);
        let ry = k0_f128_reduce_3lane_neon(y1_ll, y1_mid, y1_hh);
        return (
            vgetq_lane_u64::<0>(r0),
            vgetq_lane_u64::<1>(r0),
            vgetq_lane_u64::<0>(r2),
            vgetq_lane_u64::<1>(r2),
            vgetq_lane_u64::<0>(ry),
            vgetq_lane_u64::<1>(ry),
        );
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        let (mut u0_0, mut u0_1, mut u0_2, mut u0_3) = (0u64, 0u64, 0u64, 0u64);
        let (mut u2_0, mut u2_1, mut u2_2, mut u2_3) = (0u64, 0u64, 0u64, 0u64);
        let (mut y1_0, mut y1_1, mut y1_2, mut y1_3) = (0u64, 0u64, 0u64, 0u64);
        let mut k = 0usize;
        while k + 1 < n_f128 {
            let idx = 2 * k;
            let f0_lo = *f_ptr.add(idx);
            let f0_hi = *f_ptr.add(idx + 1);
            let f1_lo = *f_ptr.add(idx + 2);
            let f1_hi = *f_ptr.add(idx + 3);
            let b0_lo = *b_ptr.add(idx);
            let b0_hi = *b_ptr.add(idx + 1);
            let b1_lo = *b_ptr.add(idx + 2);
            let b1_hi = *b_ptr.add(idx + 3);
            let (a0, a1, a2, a3) = k0_f128_mul_unreduced_raw(f0_lo, f0_hi, b0_lo, b0_hi);
            let (c0, c1, c2, c3) = k0_f128_mul_unreduced_raw(
                f0_lo ^ f1_lo,
                f0_hi ^ f1_hi,
                b0_lo ^ b1_lo,
                b0_hi ^ b1_hi,
            );
            let (d0, d1, d2, d3) = k0_f128_mul_unreduced_raw(f1_lo, f1_hi, b1_lo, b1_hi);
            u0_0 ^= a0;
            u0_1 ^= a1;
            u0_2 ^= a2;
            u0_3 ^= a3;
            u2_0 ^= c0;
            u2_1 ^= c1;
            u2_2 ^= c2;
            u2_3 ^= c3;
            y1_0 ^= d0;
            y1_1 ^= d1;
            y1_2 ^= d2;
            y1_3 ^= d3;
            k += 2;
        }
        let (r0_lo, r0_hi) = k0_f128_reduce_256_raw(u0_0, u0_1, u0_2, u0_3);
        let (r2_lo, r2_hi) = k0_f128_reduce_256_raw(u2_0, u2_1, u2_2, u2_3);
        let (ry_lo, ry_hi) = k0_f128_reduce_256_raw(y1_0, y1_1, y1_2, y1_3);
        (r0_lo, r0_hi, r2_lo, r2_hi, ry_lo, ry_hi)
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_fold_pairs_and_msg_ex_slice(
    f_ptr: *const u64,
    b_ptr: *const u64,
    base: usize,
    fc_ptr: *mut u64,
    bc_ptr: *mut u64,
    len: usize,
    r_lo: u64,
    r_hi: u64,
    stream_f: bool,
    stream_b: bool,
) -> (u64, u64, u64, u64) {
    unsafe {
        if !stream_f && !stream_b {
            return k0_f128_fold_pairs_and_msg_slice(
                f_ptr, b_ptr, base, fc_ptr, bc_ptr, len, r_lo, r_hi,
            );
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
        {
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
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            {
                let zr0 = _mm512_broadcast_i32x4(vr0);
                let zr1 = _mm512_broadcast_i32x4(vr1);
                let idx_dup_even = _mm512_set_epi64(5, 4, 5, 4, 1, 0, 1, 0);
                let (mut z_ll, mut z_mid, mut z_hh) = (
                    _mm512_setzero_si512(),
                    _mm512_setzero_si512(),
                    _mm512_setzero_si512(),
                );
                macro_rules! store_zmm_stream {
                    ($ptr:expr, $v:expr) => {{
                        let p = $ptr as *mut __m128i;
                        let v = $v;
                        _mm_stream_si128(p, _mm512_castsi512_si128(v));
                        _mm_stream_si128(p.add(1), _mm512_extracti32x4_epi32::<1>(v));
                        _mm_stream_si128(p.add(2), _mm512_extracti32x4_epi32::<2>(v));
                        _mm_stream_si128(p.add(3), _mm512_extracti32x4_epi32::<3>(v));
                    }};
                }
                while t + 8 <= lanes {
                    let s = 4 * (base + t);
                    let d = 2 * t;
                    let f0 = _mm512_loadu_si512(f_ptr.add(s) as *const __m512i);
                    let f1 = _mm512_loadu_si512(f_ptr.add(s + 8) as *const __m512i);
                    let f2 = _mm512_loadu_si512(f_ptr.add(s + 16) as *const __m512i);
                    let f3 = _mm512_loadu_si512(f_ptr.add(s + 24) as *const __m512i);
                    let nf_a = k0_f128_fold_even_odd_zmm(f0, f1, zr0, zr1);
                    let nf_b = k0_f128_fold_even_odd_zmm(f2, f3, zr0, zr1);
                    if stream_f {
                        store_zmm_stream!(fc_ptr.add(d), nf_a);
                        store_zmm_stream!(fc_ptr.add(d + 8), nf_b);
                    } else {
                        _mm512_storeu_si512(fc_ptr.add(d) as *mut __m512i, nf_a);
                        _mm512_storeu_si512(fc_ptr.add(d + 8) as *mut __m512i, nf_b);
                    }
                    let b0 = _mm512_loadu_si512(b_ptr.add(s) as *const __m512i);
                    let b1 = _mm512_loadu_si512(b_ptr.add(s + 8) as *const __m512i);
                    let b2 = _mm512_loadu_si512(b_ptr.add(s + 16) as *const __m512i);
                    let b3 = _mm512_loadu_si512(b_ptr.add(s + 24) as *const __m512i);
                    let nb_a = k0_f128_fold_even_odd_zmm(b0, b1, zr0, zr1);
                    let nb_b = k0_f128_fold_even_odd_zmm(b2, b3, zr0, zr1);
                    if stream_b {
                        store_zmm_stream!(bc_ptr.add(d), nb_a);
                        store_zmm_stream!(bc_ptr.add(d + 8), nb_b);
                    } else {
                        _mm512_storeu_si512(bc_ptr.add(d) as *mut __m512i, nb_a);
                        _mm512_storeu_si512(bc_ptr.add(d + 8) as *mut __m512i, nb_b);
                    }
                    let nf_a_prep = _mm512_xor_si512(
                        nf_a,
                        _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nf_a),
                    );
                    let nb_a_prep = _mm512_xor_si512(
                        nb_a,
                        _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nb_a),
                    );
                    let nf_b_prep = _mm512_xor_si512(
                        nf_b,
                        _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nf_b),
                    );
                    let nb_b_prep = _mm512_xor_si512(
                        nb_b,
                        _mm512_maskz_permutexvar_epi64(0xCC, idx_dup_even, nb_b),
                    );
                    k0_f128_mul_acc2_zmm(
                        nf_a_prep, nb_a_prep, nf_b_prep, nb_b_prep, &mut z_ll, &mut z_mid,
                        &mut z_hh,
                    );
                    t += 8;
                }
                let ll_256 = _mm256_xor_si256(
                    _mm512_castsi512_si256(z_ll),
                    _mm512_extracti64x4_epi64::<1>(z_ll),
                );
                let mid_256 = _mm256_xor_si256(
                    _mm512_castsi512_si256(z_mid),
                    _mm512_extracti64x4_epi64::<1>(z_mid),
                );
                let hh_256 = _mm256_xor_si256(
                    _mm512_castsi512_si256(z_hh),
                    _mm512_extracti64x4_epi64::<1>(z_hh),
                );
                u0_ll = _mm256_castsi256_si128(ll_256);
                u0_mid = _mm256_castsi256_si128(mid_256);
                u0_hh = _mm256_castsi256_si128(hh_256);
                u2_ll = _mm256_extracti128_si256::<1>(ll_256);
                u2_mid = _mm256_extracti128_si256::<1>(mid_256);
                u2_hh = _mm256_extracti128_si256::<1>(hh_256);
            }
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
                if stream_f {
                    _mm_stream_si128(fc_ptr.add(d) as *mut __m128i, nf0);
                    _mm_stream_si128(fc_ptr.add(d + 2) as *mut __m128i, nf1);
                } else {
                    _mm_storeu_si128(fc_ptr.add(d) as *mut __m128i, nf0);
                    _mm_storeu_si128(fc_ptr.add(d + 2) as *mut __m128i, nf1);
                }
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
                if stream_b {
                    _mm_stream_si128(bc_ptr.add(d) as *mut __m128i, nb0);
                    _mm_stream_si128(bc_ptr.add(d + 2) as *mut __m128i, nb1);
                } else {
                    _mm_storeu_si128(bc_ptr.add(d) as *mut __m128i, nb0);
                    _mm_storeu_si128(bc_ptr.add(d + 2) as *mut __m128i, nb1);
                }
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
            _mm_sfence();
            let u0 = k0_f128_reduce_3lane_m128i(u0_ll, u0_mid, u0_hh);
            let u2 = k0_f128_reduce_3lane_m128i(u2_ll, u2_mid, u2_hh);
            return (
                _mm_extract_epi64::<0>(u0) as u64,
                _mm_extract_epi64::<1>(u0) as u64,
                _mm_extract_epi64::<0>(u2) as u64,
                _mm_extract_epi64::<1>(u2) as u64,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
        k0_f128_fold_pairs_and_msg_slice(f_ptr, b_ptr, base, fc_ptr, bc_ptr, len, r_lo, r_hi)
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
                $y = k0_xor3_neon($y, t1, vshlq_n_u64::<7>(t1));
                let t2 = vandq_u64(veorq_u64($y, vshrq_n_u64::<14>($y)), m2);
                $y = k0_xor3_neon($y, t2, vshlq_n_u64::<14>(t2));
                let t3 = vandq_u64(veorq_u64($y, vshrq_n_u64::<28>($y)), m3);
                $y = k0_xor3_neon($y, t3, vshlq_n_u64::<28>(t3));
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
        let mut i = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let zr0 = _mm512_broadcast_i32x4(vr0);
            let zr1 = _mm512_broadcast_i32x4(vr1);
            let (mut ze1_ll, mut ze1_mid, mut ze1_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            let (mut zei_ll, mut zei_mid, mut zei_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            while i + 4 <= len_f128 {
                let idx = 2 * i;
                let c0 = _mm512_loadu_si512(cq0.add(idx) as *const __m512i);
                let c1 = _mm512_loadu_si512(cq1.add(idx) as *const __m512i);
                let c2 = _mm512_loadu_si512(cq2.add(idx) as *const __m512i);
                let c3 = _mm512_loadu_si512(cq3.add(idx) as *const __m512i);
                let fc0 = k0_f128_mul_const_zmm(_mm512_xor_si512(c0, c2), zr0, zr1);
                let fc1 = k0_f128_mul_const_zmm(_mm512_xor_si512(c1, c3), zr0, zr1);
                let lo = _mm512_xor_si512(c0, fc0);
                let hi = _mm512_xor_si512(c1, fc1);
                let z0 = _mm512_loadu_si512(zq0.add(idx) as *const __m512i);
                let z1 = _mm512_loadu_si512(zq1.add(idx) as *const __m512i);
                let z2 = _mm512_loadu_si512(zq2.add(idx) as *const __m512i);
                let z3 = _mm512_loadu_si512(zq3.add(idx) as *const __m512i);
                let fz0 = k0_f128_mul_const_zmm(_mm512_xor_si512(z0, z2), zr0, zr1);
                let fz1 = k0_f128_mul_const_zmm(_mm512_xor_si512(z1, z3), zr0, zr1);
                let zlo = _mm512_xor_si512(z0, fz0);
                let zhi = _mm512_xor_si512(z1, fz1);
                _mm512_storeu_si512(cq0.add(idx) as *mut __m512i, lo);
                _mm512_storeu_si512(cq1.add(idx) as *mut __m512i, hi);
                _mm512_storeu_si512(zq0.add(idx) as *mut __m512i, zlo);
                _mm512_storeu_si512(zq1.add(idx) as *mut __m512i, zhi);
                k0_f128_mul_acc_zmm(hi, zhi, &mut ze1_ll, &mut ze1_mid, &mut ze1_hh);
                k0_f128_mul_acc_zmm(
                    _mm512_xor_si512(hi, lo),
                    _mm512_xor_si512(zhi, zlo),
                    &mut zei_ll,
                    &mut zei_mid,
                    &mut zei_hh,
                );
                i += 4;
            }
            e1_ll = k0_f128_hsum_zmm(ze1_ll);
            e1_mid = k0_f128_hsum_zmm(ze1_mid);
            e1_hh = k0_f128_hsum_zmm(ze1_hh);
            ei_ll = k0_f128_hsum_zmm(zei_ll);
            ei_mid = k0_f128_hsum_zmm(zei_mid);
            ei_hh = k0_f128_hsum_zmm(zei_hh);
        }
        while i < len_f128 {
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
            i += 1;
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
            k0_f128_tri_acc_neon(hi, zhi, &mut e1_ll, &mut e1_mid, &mut e1_hh);
            k0_f128_tri_acc_neon(
                veorq_u64(hi, lo),
                veorq_u64(zhi, zlo),
                &mut ei_ll,
                &mut ei_mid,
                &mut ei_hh,
            );
        }
        let e1 = k0_f128_reduce_tri_neon(e1_ll, e1_mid, e1_hh);
        let ei = k0_f128_reduce_tri_neon(ei_ll, ei_mid, ei_hh);
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
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_partial_fold_tile4_slice(
    z_base: *const u8,
    k: usize,
    bs_lo: usize,
    bs_hi: usize,
    tables_ptr: *const u64,
    out_ptr: *mut u64,
    init: u8,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let ta0 = tables_ptr as *const __m128i;
        let mut bs = bs_lo;
        if init != 0 {
            while bs + 8 <= bs_hi {
                let base_ptr = z_base.add(bs);
                let out_b = out_ptr.add(2 * bs) as *mut __m128i;
                let w0 = (base_ptr as *const u64).read_unaligned() as usize;
                let mut a0 = _mm_load_si128(ta0.add(w0 & 0xFF));
                let mut a1 = _mm_load_si128(ta0.add((w0 >> 8) & 0xFF));
                let mut a2 = _mm_load_si128(ta0.add((w0 >> 16) & 0xFF));
                let mut a3 = _mm_load_si128(ta0.add((w0 >> 24) & 0xFF));
                let mut a4 = _mm_load_si128(ta0.add((w0 >> 32) & 0xFF));
                let mut a5 = _mm_load_si128(ta0.add((w0 >> 40) & 0xFF));
                let mut a6 = _mm_load_si128(ta0.add((w0 >> 48) & 0xFF));
                let mut a7 = _mm_load_si128(ta0.add(w0 >> 56));
                for t in 1..4 {
                    let w = (base_ptr.add(t * k) as *const u64).read_unaligned() as usize;
                    let ta = ta0.add(t * 256);
                    a0 = _mm_xor_si128(a0, _mm_load_si128(ta.add(w & 0xFF)));
                    a1 = _mm_xor_si128(a1, _mm_load_si128(ta.add((w >> 8) & 0xFF)));
                    a2 = _mm_xor_si128(a2, _mm_load_si128(ta.add((w >> 16) & 0xFF)));
                    a3 = _mm_xor_si128(a3, _mm_load_si128(ta.add((w >> 24) & 0xFF)));
                    a4 = _mm_xor_si128(a4, _mm_load_si128(ta.add((w >> 32) & 0xFF)));
                    a5 = _mm_xor_si128(a5, _mm_load_si128(ta.add((w >> 40) & 0xFF)));
                    a6 = _mm_xor_si128(a6, _mm_load_si128(ta.add((w >> 48) & 0xFF)));
                    a7 = _mm_xor_si128(a7, _mm_load_si128(ta.add(w >> 56)));
                }
                _mm_store_si128(out_b, a0);
                _mm_store_si128(out_b.add(1), a1);
                _mm_store_si128(out_b.add(2), a2);
                _mm_store_si128(out_b.add(3), a3);
                _mm_store_si128(out_b.add(4), a4);
                _mm_store_si128(out_b.add(5), a5);
                _mm_store_si128(out_b.add(6), a6);
                _mm_store_si128(out_b.add(7), a7);
                bs += 8;
            }
        } else {
            while bs + 8 <= bs_hi {
                let base_ptr = z_base.add(bs);
                let out_b = out_ptr.add(2 * bs) as *mut __m128i;
                let w0 = (base_ptr as *const u64).read_unaligned() as usize;
                let mut a0 =
                    _mm_xor_si128(_mm_load_si128(out_b), _mm_load_si128(ta0.add(w0 & 0xFF)));
                let mut a1 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(1)),
                    _mm_load_si128(ta0.add((w0 >> 8) & 0xFF)),
                );
                let mut a2 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(2)),
                    _mm_load_si128(ta0.add((w0 >> 16) & 0xFF)),
                );
                let mut a3 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(3)),
                    _mm_load_si128(ta0.add((w0 >> 24) & 0xFF)),
                );
                let mut a4 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(4)),
                    _mm_load_si128(ta0.add((w0 >> 32) & 0xFF)),
                );
                let mut a5 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(5)),
                    _mm_load_si128(ta0.add((w0 >> 40) & 0xFF)),
                );
                let mut a6 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(6)),
                    _mm_load_si128(ta0.add((w0 >> 48) & 0xFF)),
                );
                let mut a7 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(7)),
                    _mm_load_si128(ta0.add(w0 >> 56)),
                );
                for t in 1..4 {
                    let w = (base_ptr.add(t * k) as *const u64).read_unaligned() as usize;
                    let ta = ta0.add(t * 256);
                    a0 = _mm_xor_si128(a0, _mm_load_si128(ta.add(w & 0xFF)));
                    a1 = _mm_xor_si128(a1, _mm_load_si128(ta.add((w >> 8) & 0xFF)));
                    a2 = _mm_xor_si128(a2, _mm_load_si128(ta.add((w >> 16) & 0xFF)));
                    a3 = _mm_xor_si128(a3, _mm_load_si128(ta.add((w >> 24) & 0xFF)));
                    a4 = _mm_xor_si128(a4, _mm_load_si128(ta.add((w >> 32) & 0xFF)));
                    a5 = _mm_xor_si128(a5, _mm_load_si128(ta.add((w >> 40) & 0xFF)));
                    a6 = _mm_xor_si128(a6, _mm_load_si128(ta.add((w >> 48) & 0xFF)));
                    a7 = _mm_xor_si128(a7, _mm_load_si128(ta.add(w >> 56)));
                }
                _mm_store_si128(out_b, a0);
                _mm_store_si128(out_b.add(1), a1);
                _mm_store_si128(out_b.add(2), a2);
                _mm_store_si128(out_b.add(3), a3);
                _mm_store_si128(out_b.add(4), a4);
                _mm_store_si128(out_b.add(5), a5);
                _mm_store_si128(out_b.add(6), a6);
                _mm_store_si128(out_b.add(7), a7);
                bs += 8;
            }
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let ta0 = tables_ptr;
        let mut bs = bs_lo;
        if init != 0 {
            while bs + 8 <= bs_hi {
                let base_ptr = z_base.add(bs);
                let out_b = out_ptr.add(2 * bs);
                let w0 = (base_ptr as *const u64).read_unaligned() as usize;
                let w1 = (base_ptr.add(k) as *const u64).read_unaligned() as usize;
                let w2 = (base_ptr.add(2 * k) as *const u64).read_unaligned() as usize;
                let w3 = (base_ptr.add(3 * k) as *const u64).read_unaligned() as usize;
                let ta1 = ta0.add(512);
                let ta2 = ta0.add(1024);
                let ta3 = ta0.add(1536);
                let a0 = veorq_u64(
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * (w0 & 0xFF))),
                        vld1q_u64(ta1.add(2 * (w1 & 0xFF))),
                        vld1q_u64(ta2.add(2 * (w2 & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * (w3 & 0xFF))),
                );
                let a1 = veorq_u64(
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 8) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 8) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 8) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 8) & 0xFF))),
                );
                let a2 = veorq_u64(
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 16) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 16) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 16) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 16) & 0xFF))),
                );
                let a3 = veorq_u64(
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 24) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 24) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 24) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 24) & 0xFF))),
                );
                let a4 = veorq_u64(
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 32) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 32) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 32) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 32) & 0xFF))),
                );
                let a5 = veorq_u64(
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 40) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 40) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 40) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 40) & 0xFF))),
                );
                let a6 = veorq_u64(
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 48) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 48) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 48) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 48) & 0xFF))),
                );
                let a7 = veorq_u64(
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * (w0 >> 56))),
                        vld1q_u64(ta1.add(2 * (w1 >> 56))),
                        vld1q_u64(ta2.add(2 * (w2 >> 56))),
                    ),
                    vld1q_u64(ta3.add(2 * (w3 >> 56))),
                );
                vst1q_u64(out_b, a0);
                vst1q_u64(out_b.add(2), a1);
                vst1q_u64(out_b.add(4), a2);
                vst1q_u64(out_b.add(6), a3);
                vst1q_u64(out_b.add(8), a4);
                vst1q_u64(out_b.add(10), a5);
                vst1q_u64(out_b.add(12), a6);
                vst1q_u64(out_b.add(14), a7);
                bs += 8;
            }
        } else {
            while bs + 8 <= bs_hi {
                let base_ptr = z_base.add(bs);
                let out_b = out_ptr.add(2 * bs);
                let w0 = (base_ptr as *const u64).read_unaligned() as usize;
                let w1 = (base_ptr.add(k) as *const u64).read_unaligned() as usize;
                let w2 = (base_ptr.add(2 * k) as *const u64).read_unaligned() as usize;
                let w3 = (base_ptr.add(3 * k) as *const u64).read_unaligned() as usize;
                let ta1 = ta0.add(512);
                let ta2 = ta0.add(1024);
                let ta3 = ta0.add(1536);
                let a0 = k0_xor3_neon(
                    vld1q_u64(out_b),
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * (w0 & 0xFF))),
                        vld1q_u64(ta1.add(2 * (w1 & 0xFF))),
                        vld1q_u64(ta2.add(2 * (w2 & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * (w3 & 0xFF))),
                );
                let a1 = k0_xor3_neon(
                    vld1q_u64(out_b.add(2)),
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 8) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 8) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 8) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 8) & 0xFF))),
                );
                let a2 = k0_xor3_neon(
                    vld1q_u64(out_b.add(4)),
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 16) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 16) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 16) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 16) & 0xFF))),
                );
                let a3 = k0_xor3_neon(
                    vld1q_u64(out_b.add(6)),
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 24) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 24) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 24) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 24) & 0xFF))),
                );
                let a4 = k0_xor3_neon(
                    vld1q_u64(out_b.add(8)),
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 32) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 32) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 32) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 32) & 0xFF))),
                );
                let a5 = k0_xor3_neon(
                    vld1q_u64(out_b.add(10)),
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 40) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 40) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 40) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 40) & 0xFF))),
                );
                let a6 = k0_xor3_neon(
                    vld1q_u64(out_b.add(12)),
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * ((w0 >> 48) & 0xFF))),
                        vld1q_u64(ta1.add(2 * ((w1 >> 48) & 0xFF))),
                        vld1q_u64(ta2.add(2 * ((w2 >> 48) & 0xFF))),
                    ),
                    vld1q_u64(ta3.add(2 * ((w3 >> 48) & 0xFF))),
                );
                let a7 = k0_xor3_neon(
                    vld1q_u64(out_b.add(14)),
                    k0_xor3_neon(
                        vld1q_u64(ta0.add(2 * (w0 >> 56))),
                        vld1q_u64(ta1.add(2 * (w1 >> 56))),
                        vld1q_u64(ta2.add(2 * (w2 >> 56))),
                    ),
                    vld1q_u64(ta3.add(2 * (w3 >> 56))),
                );
                vst1q_u64(out_b, a0);
                vst1q_u64(out_b.add(2), a1);
                vst1q_u64(out_b.add(4), a2);
                vst1q_u64(out_b.add(6), a3);
                vst1q_u64(out_b.add(8), a4);
                vst1q_u64(out_b.add(10), a5);
                vst1q_u64(out_b.add(12), a6);
                vst1q_u64(out_b.add(14), a7);
                bs += 8;
            }
        }
        return;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unsafe {
        for bs in bs_lo..bs_hi {
            let mut lo = if init != 0 {
                0u64
            } else {
                *out_ptr.add(2 * bs)
            };
            let mut hi = if init != 0 {
                0u64
            } else {
                *out_ptr.add(2 * bs + 1)
            };
            for t in 0..4 {
                let b = *z_base.add(t * k + bs) as usize;
                let tp = tables_ptr.add(2 * (t * 256 + b));
                lo ^= *tp;
                hi ^= *tp.add(1);
            }
            *out_ptr.add(2 * bs) = lo;
            *out_ptr.add(2 * bs + 1) = hi;
        }
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_partial_fold_tile8_slice(
    z_base: *const u8,
    k: usize,
    bs_lo: usize,
    bs_hi: usize,
    tables_ptr: *const u64,
    out_ptr: *mut u64,
    init: u8,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let ta0 = tables_ptr as *const __m128i;
        let mut bs = bs_lo;
        if init != 0 {
            while bs + 8 <= bs_hi {
                let base_ptr = z_base.add(bs);
                let out_b = out_ptr.add(2 * bs) as *mut __m128i;
                let w0 = (base_ptr as *const u64).read_unaligned() as usize;
                let mut a0 = _mm_load_si128(ta0.add(w0 & 0xFF));
                let mut a1 = _mm_load_si128(ta0.add((w0 >> 8) & 0xFF));
                let mut a2 = _mm_load_si128(ta0.add((w0 >> 16) & 0xFF));
                let mut a3 = _mm_load_si128(ta0.add((w0 >> 24) & 0xFF));
                let mut a4 = _mm_load_si128(ta0.add((w0 >> 32) & 0xFF));
                let mut a5 = _mm_load_si128(ta0.add((w0 >> 40) & 0xFF));
                let mut a6 = _mm_load_si128(ta0.add((w0 >> 48) & 0xFF));
                let mut a7 = _mm_load_si128(ta0.add(w0 >> 56));
                for t in 1..8 {
                    let w = (base_ptr.add(t * k) as *const u64).read_unaligned() as usize;
                    let ta = ta0.add(t * 256);
                    a0 = _mm_xor_si128(a0, _mm_load_si128(ta.add(w & 0xFF)));
                    a1 = _mm_xor_si128(a1, _mm_load_si128(ta.add((w >> 8) & 0xFF)));
                    a2 = _mm_xor_si128(a2, _mm_load_si128(ta.add((w >> 16) & 0xFF)));
                    a3 = _mm_xor_si128(a3, _mm_load_si128(ta.add((w >> 24) & 0xFF)));
                    a4 = _mm_xor_si128(a4, _mm_load_si128(ta.add((w >> 32) & 0xFF)));
                    a5 = _mm_xor_si128(a5, _mm_load_si128(ta.add((w >> 40) & 0xFF)));
                    a6 = _mm_xor_si128(a6, _mm_load_si128(ta.add((w >> 48) & 0xFF)));
                    a7 = _mm_xor_si128(a7, _mm_load_si128(ta.add(w >> 56)));
                }
                _mm_store_si128(out_b, a0);
                _mm_store_si128(out_b.add(1), a1);
                _mm_store_si128(out_b.add(2), a2);
                _mm_store_si128(out_b.add(3), a3);
                _mm_store_si128(out_b.add(4), a4);
                _mm_store_si128(out_b.add(5), a5);
                _mm_store_si128(out_b.add(6), a6);
                _mm_store_si128(out_b.add(7), a7);
                bs += 8;
            }
        } else {
            while bs + 8 <= bs_hi {
                let base_ptr = z_base.add(bs);
                let out_b = out_ptr.add(2 * bs) as *mut __m128i;
                let w0 = (base_ptr as *const u64).read_unaligned() as usize;
                let mut a0 =
                    _mm_xor_si128(_mm_load_si128(out_b), _mm_load_si128(ta0.add(w0 & 0xFF)));
                let mut a1 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(1)),
                    _mm_load_si128(ta0.add((w0 >> 8) & 0xFF)),
                );
                let mut a2 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(2)),
                    _mm_load_si128(ta0.add((w0 >> 16) & 0xFF)),
                );
                let mut a3 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(3)),
                    _mm_load_si128(ta0.add((w0 >> 24) & 0xFF)),
                );
                let mut a4 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(4)),
                    _mm_load_si128(ta0.add((w0 >> 32) & 0xFF)),
                );
                let mut a5 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(5)),
                    _mm_load_si128(ta0.add((w0 >> 40) & 0xFF)),
                );
                let mut a6 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(6)),
                    _mm_load_si128(ta0.add((w0 >> 48) & 0xFF)),
                );
                let mut a7 = _mm_xor_si128(
                    _mm_load_si128(out_b.add(7)),
                    _mm_load_si128(ta0.add(w0 >> 56)),
                );
                for t in 1..8 {
                    let w = (base_ptr.add(t * k) as *const u64).read_unaligned() as usize;
                    let ta = ta0.add(t * 256);
                    a0 = _mm_xor_si128(a0, _mm_load_si128(ta.add(w & 0xFF)));
                    a1 = _mm_xor_si128(a1, _mm_load_si128(ta.add((w >> 8) & 0xFF)));
                    a2 = _mm_xor_si128(a2, _mm_load_si128(ta.add((w >> 16) & 0xFF)));
                    a3 = _mm_xor_si128(a3, _mm_load_si128(ta.add((w >> 24) & 0xFF)));
                    a4 = _mm_xor_si128(a4, _mm_load_si128(ta.add((w >> 32) & 0xFF)));
                    a5 = _mm_xor_si128(a5, _mm_load_si128(ta.add((w >> 40) & 0xFF)));
                    a6 = _mm_xor_si128(a6, _mm_load_si128(ta.add((w >> 48) & 0xFF)));
                    a7 = _mm_xor_si128(a7, _mm_load_si128(ta.add(w >> 56)));
                }
                _mm_store_si128(out_b, a0);
                _mm_store_si128(out_b.add(1), a1);
                _mm_store_si128(out_b.add(2), a2);
                _mm_store_si128(out_b.add(3), a3);
                _mm_store_si128(out_b.add(4), a4);
                _mm_store_si128(out_b.add(5), a5);
                _mm_store_si128(out_b.add(6), a6);
                _mm_store_si128(out_b.add(7), a7);
                bs += 8;
            }
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let ta0 = tables_ptr;
        let mut bs = bs_lo;
        if init != 0 {
            while bs + 8 <= bs_hi {
                let base_ptr = z_base.add(bs);
                let out_b = out_ptr.add(2 * bs);
                let w0 = (base_ptr as *const u64).read_unaligned() as usize;
                let mut a0 = vld1q_u64(ta0.add(2 * (w0 & 0xFF)));
                let mut a1 = vld1q_u64(ta0.add(2 * ((w0 >> 8) & 0xFF)));
                let mut a2 = vld1q_u64(ta0.add(2 * ((w0 >> 16) & 0xFF)));
                let mut a3 = vld1q_u64(ta0.add(2 * ((w0 >> 24) & 0xFF)));
                let mut a4 = vld1q_u64(ta0.add(2 * ((w0 >> 32) & 0xFF)));
                let mut a5 = vld1q_u64(ta0.add(2 * ((w0 >> 40) & 0xFF)));
                let mut a6 = vld1q_u64(ta0.add(2 * ((w0 >> 48) & 0xFF)));
                let mut a7 = vld1q_u64(ta0.add(2 * (w0 >> 56)));
                for t in 1..8 {
                    let w = (base_ptr.add(t * k) as *const u64).read_unaligned() as usize;
                    let ta = ta0.add(t * 512);
                    a0 = veorq_u64(a0, vld1q_u64(ta.add(2 * (w & 0xFF))));
                    a1 = veorq_u64(a1, vld1q_u64(ta.add(2 * ((w >> 8) & 0xFF))));
                    a2 = veorq_u64(a2, vld1q_u64(ta.add(2 * ((w >> 16) & 0xFF))));
                    a3 = veorq_u64(a3, vld1q_u64(ta.add(2 * ((w >> 24) & 0xFF))));
                    a4 = veorq_u64(a4, vld1q_u64(ta.add(2 * ((w >> 32) & 0xFF))));
                    a5 = veorq_u64(a5, vld1q_u64(ta.add(2 * ((w >> 40) & 0xFF))));
                    a6 = veorq_u64(a6, vld1q_u64(ta.add(2 * ((w >> 48) & 0xFF))));
                    a7 = veorq_u64(a7, vld1q_u64(ta.add(2 * (w >> 56))));
                }
                vst1q_u64(out_b, a0);
                vst1q_u64(out_b.add(2), a1);
                vst1q_u64(out_b.add(4), a2);
                vst1q_u64(out_b.add(6), a3);
                vst1q_u64(out_b.add(8), a4);
                vst1q_u64(out_b.add(10), a5);
                vst1q_u64(out_b.add(12), a6);
                vst1q_u64(out_b.add(14), a7);
                bs += 8;
            }
        } else {
            while bs + 8 <= bs_hi {
                let base_ptr = z_base.add(bs);
                let out_b = out_ptr.add(2 * bs);
                let w0 = (base_ptr as *const u64).read_unaligned() as usize;
                let mut a0 = veorq_u64(vld1q_u64(out_b), vld1q_u64(ta0.add(2 * (w0 & 0xFF))));
                let mut a1 = veorq_u64(
                    vld1q_u64(out_b.add(2)),
                    vld1q_u64(ta0.add(2 * ((w0 >> 8) & 0xFF))),
                );
                let mut a2 = veorq_u64(
                    vld1q_u64(out_b.add(4)),
                    vld1q_u64(ta0.add(2 * ((w0 >> 16) & 0xFF))),
                );
                let mut a3 = veorq_u64(
                    vld1q_u64(out_b.add(6)),
                    vld1q_u64(ta0.add(2 * ((w0 >> 24) & 0xFF))),
                );
                let mut a4 = veorq_u64(
                    vld1q_u64(out_b.add(8)),
                    vld1q_u64(ta0.add(2 * ((w0 >> 32) & 0xFF))),
                );
                let mut a5 = veorq_u64(
                    vld1q_u64(out_b.add(10)),
                    vld1q_u64(ta0.add(2 * ((w0 >> 40) & 0xFF))),
                );
                let mut a6 = veorq_u64(
                    vld1q_u64(out_b.add(12)),
                    vld1q_u64(ta0.add(2 * ((w0 >> 48) & 0xFF))),
                );
                let mut a7 =
                    veorq_u64(vld1q_u64(out_b.add(14)), vld1q_u64(ta0.add(2 * (w0 >> 56))));
                for t in 1..8 {
                    let w = (base_ptr.add(t * k) as *const u64).read_unaligned() as usize;
                    let ta = ta0.add(t * 512);
                    a0 = veorq_u64(a0, vld1q_u64(ta.add(2 * (w & 0xFF))));
                    a1 = veorq_u64(a1, vld1q_u64(ta.add(2 * ((w >> 8) & 0xFF))));
                    a2 = veorq_u64(a2, vld1q_u64(ta.add(2 * ((w >> 16) & 0xFF))));
                    a3 = veorq_u64(a3, vld1q_u64(ta.add(2 * ((w >> 24) & 0xFF))));
                    a4 = veorq_u64(a4, vld1q_u64(ta.add(2 * ((w >> 32) & 0xFF))));
                    a5 = veorq_u64(a5, vld1q_u64(ta.add(2 * ((w >> 40) & 0xFF))));
                    a6 = veorq_u64(a6, vld1q_u64(ta.add(2 * ((w >> 48) & 0xFF))));
                    a7 = veorq_u64(a7, vld1q_u64(ta.add(2 * (w >> 56))));
                }
                vst1q_u64(out_b, a0);
                vst1q_u64(out_b.add(2), a1);
                vst1q_u64(out_b.add(4), a2);
                vst1q_u64(out_b.add(6), a3);
                vst1q_u64(out_b.add(8), a4);
                vst1q_u64(out_b.add(10), a5);
                vst1q_u64(out_b.add(12), a6);
                vst1q_u64(out_b.add(14), a7);
                bs += 8;
            }
        }
        return;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unsafe {
        for bs in bs_lo..bs_hi {
            let mut lo = if init != 0 {
                0u64
            } else {
                *out_ptr.add(2 * bs)
            };
            let mut hi = if init != 0 {
                0u64
            } else {
                *out_ptr.add(2 * bs + 1)
            };
            for t in 0..8 {
                let b = *z_base.add(t * k + bs) as usize;
                let tp = tables_ptr.add(2 * (t * 256 + b));
                lo ^= *tp;
                hi ^= *tp.add(1);
            }
            *out_ptr.add(2 * bs) = lo;
            *out_ptr.add(2 * bs + 1) = hi;
        }
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_adjoint_inject_and_sweep_slice(
    acc_ptr: *mut u64,
    eq_ptr: *const u64,
    a_roots_ptr: *const u32,
    b_roots_ptr: *const u32,
    n_rows: usize,
    alpha_lo: u64,
    alpha_hi: u64,
    children_ptr: *const u32,
    n_internal: usize,
    adj_base: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let valpha = _mm_set_epi64x(alpha_hi as i64, alpha_lo as i64);
        let (va0, va1) = k0_f128_pre_reduce_const_m128i(valpha);
        let mut r = 0usize;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let za0 = _mm512_broadcast_i32x4(va0);
            let za1 = _mm512_broadcast_i32x4(va1);
            while r + 4 <= n_rows {
                let veq = _mm512_loadu_si512(eq_ptr.add(2 * r) as *const __m512i);
                let vea = k0_f128_mul_const_zmm(veq, za0, za1);
                let ea0 = _mm512_castsi512_si128(vea);
                let ea1 = _mm512_extracti32x4_epi32::<1>(vea);
                let ea2 = _mm512_extracti32x4_epi32::<2>(vea);
                let ea3 = _mm512_extracti32x4_epi32::<3>(vea);
                let eq0 = _mm512_castsi512_si128(veq);
                let eq1 = _mm512_extracti32x4_epi32::<1>(veq);
                let eq2 = _mm512_extracti32x4_epi32::<2>(veq);
                let eq3 = _mm512_extracti32x4_epi32::<3>(veq);
                macro_rules! scatter_row {
                    ($idx:expr, $ea:expr, $eq:expr) => {{
                        let ar = *a_roots_ptr.add($idx) as usize;
                        let pa = acc_ptr.add(2 * ar) as *mut __m128i;
                        _mm_storeu_si128(pa, _mm_xor_si128(_mm_loadu_si128(pa), $ea));
                        let br = *b_roots_ptr.add($idx);
                        if br != u32::MAX {
                            let pb = acc_ptr.add(2 * (br as usize)) as *mut __m128i;
                            _mm_storeu_si128(pb, _mm_xor_si128(_mm_loadu_si128(pb), $eq));
                        }
                    }};
                }
                scatter_row!(r, ea0, eq0);
                scatter_row!(r + 1, ea1, eq1);
                scatter_row!(r + 2, ea2, eq2);
                scatter_row!(r + 3, ea3, eq3);
                r += 4;
            }
        }
        while r + 2 <= n_rows {
            let eq0 = _mm_loadu_si128(eq_ptr.add(2 * r) as *const __m128i);
            let eq1 = _mm_loadu_si128(eq_ptr.add(2 * (r + 1)) as *const __m128i);
            let (ea0, ea1) = k0_f128_mul_const_vec2_m128i(eq0, eq1, va0, va1, va0, va1);
            let ar0 = *a_roots_ptr.add(r) as usize;
            let pa0 = acc_ptr.add(2 * ar0) as *mut __m128i;
            _mm_storeu_si128(pa0, _mm_xor_si128(_mm_loadu_si128(pa0), ea0));
            let br0 = *b_roots_ptr.add(r);
            if br0 != u32::MAX {
                let pb0 = acc_ptr.add(2 * (br0 as usize)) as *mut __m128i;
                _mm_storeu_si128(pb0, _mm_xor_si128(_mm_loadu_si128(pb0), eq0));
            }
            let ar1 = *a_roots_ptr.add(r + 1) as usize;
            let pa1 = acc_ptr.add(2 * ar1) as *mut __m128i;
            _mm_storeu_si128(pa1, _mm_xor_si128(_mm_loadu_si128(pa1), ea1));
            let br1 = *b_roots_ptr.add(r + 1);
            if br1 != u32::MAX {
                let pb1 = acc_ptr.add(2 * (br1 as usize)) as *mut __m128i;
                _mm_storeu_si128(pb1, _mm_xor_si128(_mm_loadu_si128(pb1), eq1));
            }
            r += 2;
        }
        while r < n_rows {
            let eq0 = _mm_loadu_si128(eq_ptr.add(2 * r) as *const __m128i);
            let ea0 = k0_f128_mul_const_m128i(eq0, va0, va1);
            let ar0 = *a_roots_ptr.add(r) as usize;
            let pa0 = acc_ptr.add(2 * ar0) as *mut __m128i;
            _mm_storeu_si128(pa0, _mm_xor_si128(_mm_loadu_si128(pa0), ea0));
            let br0 = *b_roots_ptr.add(r);
            if br0 != u32::MAX {
                let pb0 = acc_ptr.add(2 * (br0 as usize)) as *mut __m128i;
                _mm_storeu_si128(pb0, _mm_xor_si128(_mm_loadu_si128(pb0), eq0));
            }
            r += 1;
        }
        for idx in (0..n_internal).rev() {
            let p_node = acc_ptr.add(2 * (adj_base + idx)) as *const __m128i;
            let v = _mm_loadu_si128(p_node);
            let c0 = *children_ptr.add(2 * idx) as usize;
            let c1 = *children_ptr.add(2 * idx + 1) as usize;
            let p0 = acc_ptr.add(2 * c0) as *mut __m128i;
            let p1 = acc_ptr.add(2 * c1) as *mut __m128i;
            _mm_storeu_si128(p0, _mm_xor_si128(_mm_loadu_si128(p0), v));
            _mm_storeu_si128(p1, _mm_xor_si128(_mm_loadu_si128(p1), v));
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let valpha = vld1q_u64([alpha_lo, alpha_hi].as_ptr());
        let (va0, va1) = k0_f128_pre_reduce_const_neon(valpha);
        let mut r = 0usize;
        while r + 2 <= n_rows {
            let eq0 = vld1q_u64(eq_ptr.add(2 * r));
            let eq1 = vld1q_u64(eq_ptr.add(2 * (r + 1)));
            let (ea0, ea1) = k0_f128_mul_const_vec2_neon(eq0, eq1, va0, va1, va0, va1);
            let ar0 = *a_roots_ptr.add(r) as usize;
            let pa0 = acc_ptr.add(2 * ar0);
            vst1q_u64(pa0, veorq_u64(vld1q_u64(pa0), ea0));
            let br0 = *b_roots_ptr.add(r);
            if br0 != u32::MAX {
                let pb0 = acc_ptr.add(2 * (br0 as usize));
                vst1q_u64(pb0, veorq_u64(vld1q_u64(pb0), eq0));
            }
            let ar1 = *a_roots_ptr.add(r + 1) as usize;
            let pa1 = acc_ptr.add(2 * ar1);
            vst1q_u64(pa1, veorq_u64(vld1q_u64(pa1), ea1));
            let br1 = *b_roots_ptr.add(r + 1);
            if br1 != u32::MAX {
                let pb1 = acc_ptr.add(2 * (br1 as usize));
                vst1q_u64(pb1, veorq_u64(vld1q_u64(pb1), eq1));
            }
            r += 2;
        }
        while r < n_rows {
            let eq0 = vld1q_u64(eq_ptr.add(2 * r));
            let ea0 = k0_f128_mul_const_neon(eq0, va0, va1);
            let ar0 = *a_roots_ptr.add(r) as usize;
            let pa0 = acc_ptr.add(2 * ar0);
            vst1q_u64(pa0, veorq_u64(vld1q_u64(pa0), ea0));
            let br0 = *b_roots_ptr.add(r);
            if br0 != u32::MAX {
                let pb0 = acc_ptr.add(2 * (br0 as usize));
                vst1q_u64(pb0, veorq_u64(vld1q_u64(pb0), eq0));
            }
            r += 1;
        }
        for idx in (0..n_internal).rev() {
            let p_node = acc_ptr.add(2 * (adj_base + idx));
            let v = vld1q_u64(p_node);
            let c0 = *children_ptr.add(2 * idx) as usize;
            let c1 = *children_ptr.add(2 * idx + 1) as usize;
            let p0 = acc_ptr.add(2 * c0);
            let p1 = acc_ptr.add(2 * c1);
            vst1q_u64(p0, veorq_u64(vld1q_u64(p0), v));
            vst1q_u64(p1, veorq_u64(vld1q_u64(p1), v));
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        for r in 0..n_rows {
            let eq_lo = *eq_ptr.add(2 * r);
            let eq_hi = *eq_ptr.add(2 * r + 1);
            let (ea_lo, ea_hi) = k0_f128_mul_raw(eq_lo, eq_hi, alpha_lo, alpha_hi);
            let ar = *a_roots_ptr.add(r) as usize;
            *acc_ptr.add(2 * ar) ^= ea_lo;
            *acc_ptr.add(2 * ar + 1) ^= ea_hi;
            let br = *b_roots_ptr.add(r);
            if br != u32::MAX {
                let b = br as usize;
                *acc_ptr.add(2 * b) ^= eq_lo;
                *acc_ptr.add(2 * b + 1) ^= eq_hi;
            }
        }
        for idx in (0..n_internal).rev() {
            let n_off = 2 * (adj_base + idx);
            let v_lo = *acc_ptr.add(n_off);
            let v_hi = *acc_ptr.add(n_off + 1);
            let c0 = *children_ptr.add(2 * idx) as usize;
            let c1 = *children_ptr.add(2 * idx + 1) as usize;
            *acc_ptr.add(2 * c0) ^= v_lo;
            *acc_ptr.add(2 * c0 + 1) ^= v_hi;
            *acc_ptr.add(2 * c1) ^= v_lo;
            *acc_ptr.add(2 * c1 + 1) ^= v_hi;
        }
    }
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn k0_f128_urm_accumulate_convert_batch_ab_slice(
    batch_ab_ptr: *const u8,
    batch_n_b_med_ptr: *const u8,
    batch_eq_lo_ptr: *const u64,
    batch_len: usize,
    convert_ptr: *const u64,
    partial_ab_ptr: *mut u64,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        if k0_has_avx512_gfni() {
            k0_f128_urm_accumulate_convert_batch_ab_gfni_avx512_slice(
                batch_ab_ptr,
                batch_n_b_med_ptr,
                batch_eq_lo_ptr,
                batch_len,
                convert_ptr,
                partial_ab_ptr,
            );
            return;
        }
        let conv_m128 = convert_ptr as *const __m128i;
        let mut conv_buf0_uninit = core::mem::MaybeUninit::<[[__m128i; 64]; 8]>::uninit();
        let conv_buf0 = &mut *conv_buf0_uninit.as_mut_ptr();
        let mut veqs_uninit = core::mem::MaybeUninit::<[__m128i; 8]>::uninit();
        let veqs = &mut *veqs_uninit.as_mut_ptr();
        let mut veq_sums_uninit = core::mem::MaybeUninit::<[__m128i; 8]>::uninit();
        let veq_sums = &mut *veq_sums_uninit.as_mut_ptr();
        for b_idx in 0..batch_len {
            let veq = _mm_loadu_si128(batch_eq_lo_ptr.add(2 * b_idx) as *const __m128i);
            veqs[b_idx] = veq;
            veq_sums[b_idx] = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4e>(veq));
        }
        for q in 0..4 {
            let b_base = q * 4;
            let t0 = conv_m128.add(b_base * 256);
            let t1 = conv_m128.add((b_base + 1) * 256);
            let t2 = conv_m128.add((b_base + 2) * 256);
            let t3 = conv_m128.add((b_base + 3) * 256);
            for b_idx in 0..batch_len {
                let n_b_med = *batch_n_b_med_ptr.add(b_idx) as usize;
                let dst_ab = &mut conv_buf0[b_idx];
                if b_base >= n_b_med {
                    if q == 0 {
                        dst_ab.fill(_mm_load_si128(t0));
                    }
                    continue;
                }
                let ab_rows_ptr = batch_ab_ptr.add(b_idx * 1024);
                let rem = n_b_med - b_base;
                if rem >= 4 {
                    let r0 = ab_rows_ptr.add(b_base * 64);
                    let r1 = ab_rows_ptr.add((b_base + 1) * 64);
                    let r2 = ab_rows_ptr.add((b_base + 2) * 64);
                    let r3 = ab_rows_ptr.add((b_base + 3) * 64);
                    if q == 0 {
                        for tile in 0..16 {
                            let l0 = tile * 4;
                            let l1 = l0 + 1;
                            let l2 = l0 + 2;
                            let l3 = l0 + 3;
                            let mut ab0 = _mm_load_si128(t0.add(*r0.add(l0) as usize));
                            let mut ab1 = _mm_load_si128(t0.add(*r0.add(l1) as usize));
                            let mut ab2 = _mm_load_si128(t0.add(*r0.add(l2) as usize));
                            let mut ab3 = _mm_load_si128(t0.add(*r0.add(l3) as usize));
                            ab0 = _mm_xor_si128(ab0, _mm_load_si128(t1.add(*r1.add(l0) as usize)));
                            ab1 = _mm_xor_si128(ab1, _mm_load_si128(t1.add(*r1.add(l1) as usize)));
                            ab2 = _mm_xor_si128(ab2, _mm_load_si128(t1.add(*r1.add(l2) as usize)));
                            ab3 = _mm_xor_si128(ab3, _mm_load_si128(t1.add(*r1.add(l3) as usize)));
                            ab0 = _mm_xor_si128(ab0, _mm_load_si128(t2.add(*r2.add(l0) as usize)));
                            ab1 = _mm_xor_si128(ab1, _mm_load_si128(t2.add(*r2.add(l1) as usize)));
                            ab2 = _mm_xor_si128(ab2, _mm_load_si128(t2.add(*r2.add(l2) as usize)));
                            ab3 = _mm_xor_si128(ab3, _mm_load_si128(t2.add(*r2.add(l3) as usize)));
                            dst_ab[l0] =
                                _mm_xor_si128(ab0, _mm_load_si128(t3.add(*r3.add(l0) as usize)));
                            dst_ab[l1] =
                                _mm_xor_si128(ab1, _mm_load_si128(t3.add(*r3.add(l1) as usize)));
                            dst_ab[l2] =
                                _mm_xor_si128(ab2, _mm_load_si128(t3.add(*r3.add(l2) as usize)));
                            dst_ab[l3] =
                                _mm_xor_si128(ab3, _mm_load_si128(t3.add(*r3.add(l3) as usize)));
                        }
                    } else {
                        for tile in 0..16 {
                            let l0 = tile * 4;
                            let l1 = l0 + 1;
                            let l2 = l0 + 2;
                            let l3 = l0 + 3;
                            let mut ab0 = _mm_xor_si128(
                                dst_ab[l0],
                                _mm_load_si128(t0.add(*r0.add(l0) as usize)),
                            );
                            let mut ab1 = _mm_xor_si128(
                                dst_ab[l1],
                                _mm_load_si128(t0.add(*r0.add(l1) as usize)),
                            );
                            let mut ab2 = _mm_xor_si128(
                                dst_ab[l2],
                                _mm_load_si128(t0.add(*r0.add(l2) as usize)),
                            );
                            let mut ab3 = _mm_xor_si128(
                                dst_ab[l3],
                                _mm_load_si128(t0.add(*r0.add(l3) as usize)),
                            );
                            ab0 = _mm_xor_si128(ab0, _mm_load_si128(t1.add(*r1.add(l0) as usize)));
                            ab1 = _mm_xor_si128(ab1, _mm_load_si128(t1.add(*r1.add(l1) as usize)));
                            ab2 = _mm_xor_si128(ab2, _mm_load_si128(t1.add(*r1.add(l2) as usize)));
                            ab3 = _mm_xor_si128(ab3, _mm_load_si128(t1.add(*r1.add(l3) as usize)));
                            ab0 = _mm_xor_si128(ab0, _mm_load_si128(t2.add(*r2.add(l0) as usize)));
                            ab1 = _mm_xor_si128(ab1, _mm_load_si128(t2.add(*r2.add(l1) as usize)));
                            ab2 = _mm_xor_si128(ab2, _mm_load_si128(t2.add(*r2.add(l2) as usize)));
                            ab3 = _mm_xor_si128(ab3, _mm_load_si128(t2.add(*r2.add(l3) as usize)));
                            dst_ab[l0] =
                                _mm_xor_si128(ab0, _mm_load_si128(t3.add(*r3.add(l0) as usize)));
                            dst_ab[l1] =
                                _mm_xor_si128(ab1, _mm_load_si128(t3.add(*r3.add(l1) as usize)));
                            dst_ab[l2] =
                                _mm_xor_si128(ab2, _mm_load_si128(t3.add(*r3.add(l2) as usize)));
                            dst_ab[l3] =
                                _mm_xor_si128(ab3, _mm_load_si128(t3.add(*r3.add(l3) as usize)));
                        }
                    }
                } else if rem == 3 {
                    let r0 = ab_rows_ptr.add(b_base * 64);
                    let r1 = ab_rows_ptr.add((b_base + 1) * 64);
                    let r2 = ab_rows_ptr.add((b_base + 2) * 64);
                    if q == 0 {
                        for tile in 0..16 {
                            let l0 = tile * 4;
                            let l1 = l0 + 1;
                            let l2 = l0 + 2;
                            let l3 = l0 + 3;
                            let mut ab0 = _mm_load_si128(t0.add(*r0.add(l0) as usize));
                            let mut ab1 = _mm_load_si128(t0.add(*r0.add(l1) as usize));
                            let mut ab2 = _mm_load_si128(t0.add(*r0.add(l2) as usize));
                            let mut ab3 = _mm_load_si128(t0.add(*r0.add(l3) as usize));
                            ab0 = _mm_xor_si128(ab0, _mm_load_si128(t1.add(*r1.add(l0) as usize)));
                            ab1 = _mm_xor_si128(ab1, _mm_load_si128(t1.add(*r1.add(l1) as usize)));
                            ab2 = _mm_xor_si128(ab2, _mm_load_si128(t1.add(*r1.add(l2) as usize)));
                            ab3 = _mm_xor_si128(ab3, _mm_load_si128(t1.add(*r1.add(l3) as usize)));
                            dst_ab[l0] =
                                _mm_xor_si128(ab0, _mm_load_si128(t2.add(*r2.add(l0) as usize)));
                            dst_ab[l1] =
                                _mm_xor_si128(ab1, _mm_load_si128(t2.add(*r2.add(l1) as usize)));
                            dst_ab[l2] =
                                _mm_xor_si128(ab2, _mm_load_si128(t2.add(*r2.add(l2) as usize)));
                            dst_ab[l3] =
                                _mm_xor_si128(ab3, _mm_load_si128(t2.add(*r2.add(l3) as usize)));
                        }
                    } else {
                        for tile in 0..16 {
                            let l0 = tile * 4;
                            let l1 = l0 + 1;
                            let l2 = l0 + 2;
                            let l3 = l0 + 3;
                            let mut ab0 = _mm_xor_si128(
                                dst_ab[l0],
                                _mm_load_si128(t0.add(*r0.add(l0) as usize)),
                            );
                            let mut ab1 = _mm_xor_si128(
                                dst_ab[l1],
                                _mm_load_si128(t0.add(*r0.add(l1) as usize)),
                            );
                            let mut ab2 = _mm_xor_si128(
                                dst_ab[l2],
                                _mm_load_si128(t0.add(*r0.add(l2) as usize)),
                            );
                            let mut ab3 = _mm_xor_si128(
                                dst_ab[l3],
                                _mm_load_si128(t0.add(*r0.add(l3) as usize)),
                            );
                            ab0 = _mm_xor_si128(ab0, _mm_load_si128(t1.add(*r1.add(l0) as usize)));
                            ab1 = _mm_xor_si128(ab1, _mm_load_si128(t1.add(*r1.add(l1) as usize)));
                            ab2 = _mm_xor_si128(ab2, _mm_load_si128(t1.add(*r1.add(l2) as usize)));
                            ab3 = _mm_xor_si128(ab3, _mm_load_si128(t1.add(*r1.add(l3) as usize)));
                            dst_ab[l0] =
                                _mm_xor_si128(ab0, _mm_load_si128(t2.add(*r2.add(l0) as usize)));
                            dst_ab[l1] =
                                _mm_xor_si128(ab1, _mm_load_si128(t2.add(*r2.add(l1) as usize)));
                            dst_ab[l2] =
                                _mm_xor_si128(ab2, _mm_load_si128(t2.add(*r2.add(l2) as usize)));
                            dst_ab[l3] =
                                _mm_xor_si128(ab3, _mm_load_si128(t2.add(*r2.add(l3) as usize)));
                        }
                    }
                } else if q == 0 {
                    for l in 0..64 {
                        let mut ab =
                            _mm_loadu_si128(t0.add(*ab_rows_ptr.add(b_base * 64 + l) as usize));
                        for r in 1..rem {
                            ab = _mm_xor_si128(
                                ab,
                                _mm_loadu_si128(conv_m128.add(
                                    (b_base + r) * 256
                                        + (*ab_rows_ptr.add((b_base + r) * 64 + l) as usize),
                                )),
                            );
                        }
                        dst_ab[l] = ab;
                    }
                } else {
                    for l in 0..64 {
                        let mut ab = _mm_xor_si128(
                            dst_ab[l],
                            _mm_loadu_si128(t0.add(*ab_rows_ptr.add(b_base * 64 + l) as usize)),
                        );
                        for r in 1..rem {
                            ab = _mm_xor_si128(
                                ab,
                                _mm_loadu_si128(conv_m128.add(
                                    (b_base + r) * 256
                                        + (*ab_rows_ptr.add((b_base + r) * 64 + l) as usize),
                                )),
                            );
                        }
                        dst_ab[l] = ab;
                    }
                }
            }
        }
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let mut zveqs = [_mm512_setzero_si512(); 8];
            for b_idx in 0..batch_len {
                zveqs[b_idx] = _mm512_broadcast_i32x4(veqs[b_idx]);
            }
            for group in 0..16 {
                let l_base = group * 4;
                let (mut z_ll, mut z_mid, mut z_hh) = (
                    _mm512_setzero_si512(),
                    _mm512_setzero_si512(),
                    _mm512_setzero_si512(),
                );
                let mut b_idx = 0usize;
                while b_idx + 2 <= batch_len {
                    let ab_a =
                        _mm512_loadu_si512(conv_buf0[b_idx].as_ptr().add(l_base) as *const __m512i);
                    let ab_b = _mm512_loadu_si512(
                        conv_buf0[b_idx + 1].as_ptr().add(l_base) as *const __m512i
                    );
                    k0_f128_mul_acc2_zmm(
                        ab_a,
                        zveqs[b_idx],
                        ab_b,
                        zveqs[b_idx + 1],
                        &mut z_ll,
                        &mut z_mid,
                        &mut z_hh,
                    );
                    b_idx += 2;
                }
                while b_idx < batch_len {
                    let ab_a =
                        _mm512_loadu_si512(conv_buf0[b_idx].as_ptr().add(l_base) as *const __m512i);
                    k0_f128_mul_acc_zmm(ab_a, zveqs[b_idx], &mut z_ll, &mut z_mid, &mut z_hh);
                    b_idx += 1;
                }
                macro_rules! store_lane {
                    ($lane:literal) => {{
                        let p = partial_ab_ptr.add(6 * (l_base + $lane)) as *mut __m128i;
                        let ll_l = _mm512_extracti32x4_epi32::<$lane>(z_ll);
                        let mid_l = _mm512_extracti32x4_epi32::<$lane>(z_mid);
                        let hh_l = _mm512_extracti32x4_epi32::<$lane>(z_hh);
                        _mm_storeu_si128(p, _mm_xor_si128(_mm_loadu_si128(p), ll_l));
                        _mm_storeu_si128(p.add(1), _mm_xor_si128(_mm_loadu_si128(p.add(1)), mid_l));
                        _mm_storeu_si128(p.add(2), _mm_xor_si128(_mm_loadu_si128(p.add(2)), hh_l));
                    }};
                }
                store_lane!(0);
                store_lane!(1);
                store_lane!(2);
                store_lane!(3);
            }
            return;
        }
        for pair in 0..32 {
            let l0 = pair * 2;
            let l1 = l0 + 1;
            let ptr0 = partial_ab_ptr.add(6 * l0) as *mut __m128i;
            let ptr1 = partial_ab_ptr.add(6 * l1) as *mut __m128i;
            let mut ll0 = _mm_loadu_si128(ptr0);
            let mut mid0 = _mm_loadu_si128(ptr0.add(1));
            let mut hh0 = _mm_loadu_si128(ptr0.add(2));
            let mut ll1 = _mm_loadu_si128(ptr1);
            let mut mid1 = _mm_loadu_si128(ptr1.add(1));
            let mut hh1 = _mm_loadu_si128(ptr1.add(2));
            for b_idx in 0..batch_len {
                let veq = veqs[b_idx];
                let veq_sum = veq_sums[b_idx];
                let src_ab = &conv_buf0[b_idx];
                let ab0 = src_ab[l0];
                let ab1 = src_ab[l1];
                let ab01_sum =
                    _mm_xor_si128(_mm_unpacklo_epi64(ab0, ab1), _mm_unpackhi_epi64(ab0, ab1));
                ll0 = _mm_xor_si128(ll0, _mm_clmulepi64_si128::<0x00>(ab0, veq));
                mid0 = _mm_xor_si128(mid0, _mm_clmulepi64_si128::<0x00>(ab01_sum, veq_sum));
                hh0 = _mm_xor_si128(hh0, _mm_clmulepi64_si128::<0x11>(ab0, veq));
                ll1 = _mm_xor_si128(ll1, _mm_clmulepi64_si128::<0x00>(ab1, veq));
                mid1 = _mm_xor_si128(mid1, _mm_clmulepi64_si128::<0x01>(ab01_sum, veq_sum));
                hh1 = _mm_xor_si128(hh1, _mm_clmulepi64_si128::<0x11>(ab1, veq));
            }
            _mm_storeu_si128(ptr0, ll0);
            _mm_storeu_si128(ptr0.add(1), mid0);
            _mm_storeu_si128(ptr0.add(2), hh0);
            _mm_storeu_si128(ptr1, ll1);
            _mm_storeu_si128(ptr1.add(1), mid1);
            _mm_storeu_si128(ptr1.add(2), hh1);
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let mut conv_ab_uninit = core::mem::MaybeUninit::<[[uint64x2_t; 64]; 8]>::uninit();
        let conv_ab = &mut *conv_ab_uninit.as_mut_ptr();
        for q in 0..4 {
            let b_base = q * 4;
            let t0 = convert_ptr.add(b_base * 512);
            let t1 = convert_ptr.add((b_base + 1) * 512);
            let t2 = convert_ptr.add((b_base + 2) * 512);
            let t3 = convert_ptr.add((b_base + 3) * 512);
            for b_idx in 0..batch_len {
                let n_b_med = *batch_n_b_med_ptr.add(b_idx) as usize;
                let dst_ab = &mut conv_ab[b_idx];
                if b_base >= n_b_med {
                    if q == 0 {
                        dst_ab.fill(z);
                    }
                    continue;
                }
                let ab_rows_ptr = batch_ab_ptr.add(b_idx * 1024);
                let rem = n_b_med - b_base;
                if rem >= 4 {
                    let r0 = ab_rows_ptr.add(b_base * 64);
                    let r1 = ab_rows_ptr.add((b_base + 1) * 64);
                    let r2 = ab_rows_ptr.add((b_base + 2) * 64);
                    let r3 = ab_rows_ptr.add((b_base + 3) * 64);
                    if q == 0 {
                        for tile in 0..16 {
                            let l0 = tile * 4;
                            let l1 = l0 + 1;
                            let l2 = l0 + 2;
                            let l3 = l0 + 3;
                            dst_ab[l0] = veorq_u64(
                                k0_xor3_neon(
                                    vld1q_u64(t0.add(2 * (*r0.add(l0) as usize))),
                                    vld1q_u64(t1.add(2 * (*r1.add(l0) as usize))),
                                    vld1q_u64(t2.add(2 * (*r2.add(l0) as usize))),
                                ),
                                vld1q_u64(t3.add(2 * (*r3.add(l0) as usize))),
                            );
                            dst_ab[l1] = veorq_u64(
                                k0_xor3_neon(
                                    vld1q_u64(t0.add(2 * (*r0.add(l1) as usize))),
                                    vld1q_u64(t1.add(2 * (*r1.add(l1) as usize))),
                                    vld1q_u64(t2.add(2 * (*r2.add(l1) as usize))),
                                ),
                                vld1q_u64(t3.add(2 * (*r3.add(l1) as usize))),
                            );
                            dst_ab[l2] = veorq_u64(
                                k0_xor3_neon(
                                    vld1q_u64(t0.add(2 * (*r0.add(l2) as usize))),
                                    vld1q_u64(t1.add(2 * (*r1.add(l2) as usize))),
                                    vld1q_u64(t2.add(2 * (*r2.add(l2) as usize))),
                                ),
                                vld1q_u64(t3.add(2 * (*r3.add(l2) as usize))),
                            );
                            dst_ab[l3] = veorq_u64(
                                k0_xor3_neon(
                                    vld1q_u64(t0.add(2 * (*r0.add(l3) as usize))),
                                    vld1q_u64(t1.add(2 * (*r1.add(l3) as usize))),
                                    vld1q_u64(t2.add(2 * (*r2.add(l3) as usize))),
                                ),
                                vld1q_u64(t3.add(2 * (*r3.add(l3) as usize))),
                            );
                        }
                    } else {
                        for tile in 0..16 {
                            let l0 = tile * 4;
                            let l1 = l0 + 1;
                            let l2 = l0 + 2;
                            let l3 = l0 + 3;
                            let ab0 = veorq_u64(
                                k0_xor3_neon(
                                    vld1q_u64(t0.add(2 * (*r0.add(l0) as usize))),
                                    vld1q_u64(t1.add(2 * (*r1.add(l0) as usize))),
                                    vld1q_u64(t2.add(2 * (*r2.add(l0) as usize))),
                                ),
                                vld1q_u64(t3.add(2 * (*r3.add(l0) as usize))),
                            );
                            let ab1 = veorq_u64(
                                k0_xor3_neon(
                                    vld1q_u64(t0.add(2 * (*r0.add(l1) as usize))),
                                    vld1q_u64(t1.add(2 * (*r1.add(l1) as usize))),
                                    vld1q_u64(t2.add(2 * (*r2.add(l1) as usize))),
                                ),
                                vld1q_u64(t3.add(2 * (*r3.add(l1) as usize))),
                            );
                            let ab2 = veorq_u64(
                                k0_xor3_neon(
                                    vld1q_u64(t0.add(2 * (*r0.add(l2) as usize))),
                                    vld1q_u64(t1.add(2 * (*r1.add(l2) as usize))),
                                    vld1q_u64(t2.add(2 * (*r2.add(l2) as usize))),
                                ),
                                vld1q_u64(t3.add(2 * (*r3.add(l2) as usize))),
                            );
                            let ab3 = veorq_u64(
                                k0_xor3_neon(
                                    vld1q_u64(t0.add(2 * (*r0.add(l3) as usize))),
                                    vld1q_u64(t1.add(2 * (*r1.add(l3) as usize))),
                                    vld1q_u64(t2.add(2 * (*r2.add(l3) as usize))),
                                ),
                                vld1q_u64(t3.add(2 * (*r3.add(l3) as usize))),
                            );
                            dst_ab[l0] = veorq_u64(dst_ab[l0], ab0);
                            dst_ab[l1] = veorq_u64(dst_ab[l1], ab1);
                            dst_ab[l2] = veorq_u64(dst_ab[l2], ab2);
                            dst_ab[l3] = veorq_u64(dst_ab[l3], ab3);
                        }
                    }
                } else if rem == 3 {
                    let r0 = ab_rows_ptr.add(b_base * 64);
                    let r1 = ab_rows_ptr.add((b_base + 1) * 64);
                    let r2 = ab_rows_ptr.add((b_base + 2) * 64);
                    if q == 0 {
                        for tile in 0..16 {
                            let l0 = tile * 4;
                            let l1 = l0 + 1;
                            let l2 = l0 + 2;
                            let l3 = l0 + 3;
                            dst_ab[l0] = k0_xor3_neon(
                                vld1q_u64(t0.add(2 * (*r0.add(l0) as usize))),
                                vld1q_u64(t1.add(2 * (*r1.add(l0) as usize))),
                                vld1q_u64(t2.add(2 * (*r2.add(l0) as usize))),
                            );
                            dst_ab[l1] = k0_xor3_neon(
                                vld1q_u64(t0.add(2 * (*r0.add(l1) as usize))),
                                vld1q_u64(t1.add(2 * (*r1.add(l1) as usize))),
                                vld1q_u64(t2.add(2 * (*r2.add(l1) as usize))),
                            );
                            dst_ab[l2] = k0_xor3_neon(
                                vld1q_u64(t0.add(2 * (*r0.add(l2) as usize))),
                                vld1q_u64(t1.add(2 * (*r1.add(l2) as usize))),
                                vld1q_u64(t2.add(2 * (*r2.add(l2) as usize))),
                            );
                            dst_ab[l3] = k0_xor3_neon(
                                vld1q_u64(t0.add(2 * (*r0.add(l3) as usize))),
                                vld1q_u64(t1.add(2 * (*r1.add(l3) as usize))),
                                vld1q_u64(t2.add(2 * (*r2.add(l3) as usize))),
                            );
                        }
                    } else {
                        for tile in 0..16 {
                            let l0 = tile * 4;
                            let l1 = l0 + 1;
                            let l2 = l0 + 2;
                            let l3 = l0 + 3;
                            let ab0 = k0_xor3_neon(
                                vld1q_u64(t0.add(2 * (*r0.add(l0) as usize))),
                                vld1q_u64(t1.add(2 * (*r1.add(l0) as usize))),
                                vld1q_u64(t2.add(2 * (*r2.add(l0) as usize))),
                            );
                            let ab1 = k0_xor3_neon(
                                vld1q_u64(t0.add(2 * (*r0.add(l1) as usize))),
                                vld1q_u64(t1.add(2 * (*r1.add(l1) as usize))),
                                vld1q_u64(t2.add(2 * (*r2.add(l1) as usize))),
                            );
                            let ab2 = k0_xor3_neon(
                                vld1q_u64(t0.add(2 * (*r0.add(l2) as usize))),
                                vld1q_u64(t1.add(2 * (*r1.add(l2) as usize))),
                                vld1q_u64(t2.add(2 * (*r2.add(l2) as usize))),
                            );
                            let ab3 = k0_xor3_neon(
                                vld1q_u64(t0.add(2 * (*r0.add(l3) as usize))),
                                vld1q_u64(t1.add(2 * (*r1.add(l3) as usize))),
                                vld1q_u64(t2.add(2 * (*r2.add(l3) as usize))),
                            );
                            dst_ab[l0] = veorq_u64(dst_ab[l0], ab0);
                            dst_ab[l1] = veorq_u64(dst_ab[l1], ab1);
                            dst_ab[l2] = veorq_u64(dst_ab[l2], ab2);
                            dst_ab[l3] = veorq_u64(dst_ab[l3], ab3);
                        }
                    }
                } else if q == 0 {
                    for l in 0..64 {
                        let mut ab =
                            vld1q_u64(t0.add(2 * (*ab_rows_ptr.add(b_base * 64 + l) as usize)));
                        for r in 1..rem {
                            ab = veorq_u64(
                                ab,
                                vld1q_u64(convert_ptr.add(
                                    2 * ((b_base + r) * 256
                                        + (*ab_rows_ptr.add((b_base + r) * 64 + l) as usize)),
                                )),
                            );
                        }
                        dst_ab[l] = ab;
                    }
                } else {
                    for l in 0..64 {
                        let mut ab = veorq_u64(
                            dst_ab[l],
                            vld1q_u64(t0.add(2 * (*ab_rows_ptr.add(b_base * 64 + l) as usize))),
                        );
                        for r in 1..rem {
                            ab = veorq_u64(
                                ab,
                                vld1q_u64(convert_ptr.add(
                                    2 * ((b_base + r) * 256
                                        + (*ab_rows_ptr.add((b_base + r) * 64 + l) as usize)),
                                )),
                            );
                        }
                        dst_ab[l] = ab;
                    }
                }
            }
        }
        let mut veqs = [z; 8];
        let mut eq_los = [0u64; 8];
        let mut eq_sums = [0u64; 8];
        for b_idx in 0..batch_len {
            let veq = vld1q_u64(batch_eq_lo_ptr.add(2 * b_idx));
            let eq_lo = vgetq_lane_u64::<0>(veq);
            veqs[b_idx] = veq;
            eq_los[b_idx] = eq_lo;
            eq_sums[b_idx] = vgetq_lane_u64::<0>(veorq_u64(veq, vextq_u64::<1>(veq, veq)));
        }
        for l in 0..64 {
            let (mut ab_ll, mut ab_mid, mut ab_hh) = (z, z, z);
            let mut b_idx = 0usize;
            while b_idx + 2 <= batch_len {
                let vab0 = conv_ab[b_idx][l];
                let vab1 = conv_ab[b_idx + 1][l];
                let ab0_sum = veorq_u64(vab0, vextq_u64::<1>(vab0, vab0));
                let ab1_sum = veorq_u64(vab1, vextq_u64::<1>(vab1, vab1));
                ab_ll = k0_xor3_neon(
                    ab_ll,
                    vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(vab0), eq_los[b_idx])),
                    vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(vab1), eq_los[b_idx + 1])),
                );
                ab_mid = k0_xor3_neon(
                    ab_mid,
                    vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(ab0_sum), eq_sums[b_idx])),
                    vreinterpretq_u64_p128(vmull_p64(
                        vgetq_lane_u64::<0>(ab1_sum),
                        eq_sums[b_idx + 1],
                    )),
                );
                ab_hh = k0_xor3_neon(
                    ab_hh,
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64(vab0),
                        vreinterpretq_p64_u64(veqs[b_idx]),
                    )),
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64(vab1),
                        vreinterpretq_p64_u64(veqs[b_idx + 1]),
                    )),
                );
                b_idx += 2;
            }
            while b_idx < batch_len {
                let vab0 = conv_ab[b_idx][l];
                let ab0_sum = veorq_u64(vab0, vextq_u64::<1>(vab0, vab0));
                ab_ll = veorq_u64(
                    ab_ll,
                    vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(vab0), eq_los[b_idx])),
                );
                ab_mid = veorq_u64(
                    ab_mid,
                    vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(ab0_sum), eq_sums[b_idx])),
                );
                ab_hh = veorq_u64(
                    ab_hh,
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64(vab0),
                        vreinterpretq_p64_u64(veqs[b_idx]),
                    )),
                );
                b_idx += 1;
            }
            let ptr_ab = partial_ab_ptr.add(6 * l);
            vst1q_u64(ptr_ab, veorq_u64(vld1q_u64(ptr_ab), ab_ll));
            vst1q_u64(ptr_ab.add(2), veorq_u64(vld1q_u64(ptr_ab.add(2)), ab_mid));
            vst1q_u64(ptr_ab.add(4), veorq_u64(vld1q_u64(ptr_ab.add(4)), ab_hh));
        }
        return;
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    unsafe {
        for b_idx in 0..batch_len {
            let eq_lo = *batch_eq_lo_ptr.add(2 * b_idx);
            let eq_hi = *batch_eq_lo_ptr.add(2 * b_idx + 1);
            let n_b_med = *batch_n_b_med_ptr.add(b_idx) as usize;
            let ab_rows_ptr = batch_ab_ptr.add(b_idx * 1024);
            for lane in 0..64 {
                let mut c_lo = 0u64;
                let mut c_hi = 0u64;
                for b_med in 0..n_b_med {
                    let v = *ab_rows_ptr.add(b_med * 64 + lane) as usize;
                    let cp = convert_ptr.add(2 * (b_med * 256 + v));
                    c_lo ^= *cp;
                    c_hi ^= *cp.add(1);
                }
                let (p_lo, p_hi) = k0_f128_mul_raw(c_lo, c_hi, eq_lo, eq_hi);
                *partial_ab_ptr.add(6 * lane) ^= p_lo;
                *partial_ab_ptr.add(6 * lane + 1) ^= p_hi;
            }
        }
    }
}
#[inline(always)]
pub unsafe fn k0_f128_pre_reduce_eq_lo_slice(
    eq_lo_ptr: *const u64,
    eq_prered_out_ptr: *mut u64,
    n_elems: usize,
) {
    unsafe {
        #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
        {
            use core::arch::x86_64::*;
            let src = eq_lo_ptr as *const __m128i;
            let dst = eq_prered_out_ptr as *mut __m128i;
            for i in 0..n_elems {
                let veq = _mm_loadu_si128(src.add(i));
                let (r0, r1) = k0_f128_pre_reduce_const_m128i(veq);
                _mm_storeu_si128(dst.add(2 * i), r0);
                _mm_storeu_si128(dst.add(2 * i + 1), r1);
            }
            return;
        }
        #[allow(unreachable_code)]
        for i in 0..n_elems {
            let lo = *eq_lo_ptr.add(2 * i);
            let hi = *eq_lo_ptr.add(2 * i + 1);
            let p_lo = k0_clmul64_lo(hi, 0x87);
            *eq_prered_out_ptr.add(4 * i) = lo;
            *eq_prered_out_ptr.add(4 * i + 1) = p_lo;
            *eq_prered_out_ptr.add(4 * i + 2) = hi;
            *eq_prered_out_ptr.add(4 * i + 3) = p_lo ^ lo;
        }
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn k0_gf8_inv_ntt_avx2_pair_composite_w(
    base: *const u8,
    w: usize,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    unsafe {
        use core::arch::x86_64::*;
        let row0 = base.add((w & 0xFF) << 6);
        let row4 = base.add(((w >> 32) & 0xFF) << 6);
        let r04_lo = _mm256_xor_si256(
            _mm256_load_si256(row0 as *const __m256i),
            _mm256_load_si256(row4.add(32) as *const __m256i),
        );
        let r04_hi = _mm256_xor_si256(
            _mm256_load_si256(row0.add(32) as *const __m256i),
            _mm256_load_si256(row4 as *const __m256i),
        );
        let row1 = base.add(((w >> 8) & 0xFF) << 6);
        let row5 = base.add(((w >> 40) & 0xFF) << 6);
        let r15_lo = _mm256_xor_si256(
            _mm256_load_si256(row1 as *const __m256i),
            _mm256_load_si256(row5.add(32) as *const __m256i),
        );
        let even_lo = _mm256_xor_si256(r04_lo, _mm256_shuffle_epi32::<0x4E>(r15_lo));
        let r15_hi = _mm256_xor_si256(
            _mm256_load_si256(row1.add(32) as *const __m256i),
            _mm256_load_si256(row5 as *const __m256i),
        );
        let even_hi = _mm256_xor_si256(r04_hi, _mm256_shuffle_epi32::<0x4E>(r15_hi));
        let row2 = base.add(((w >> 16) & 0xFF) << 6);
        let row6 = base.add(((w >> 48) & 0xFF) << 6);
        let row3 = base.add(((w >> 24) & 0xFF) << 6);
        let row7 = base.add((w >> 56) << 6);
        let r26_lo = _mm256_xor_si256(
            _mm256_load_si256(row2 as *const __m256i),
            _mm256_load_si256(row6.add(32) as *const __m256i),
        );
        let r37_lo = _mm256_xor_si256(
            _mm256_load_si256(row3 as *const __m256i),
            _mm256_load_si256(row7.add(32) as *const __m256i),
        );
        let odd_lo = _mm256_xor_si256(r26_lo, _mm256_shuffle_epi32::<0x4E>(r37_lo));
        let lo = _mm256_xor_si256(even_lo, _mm256_permute4x64_epi64::<0x4E>(odd_lo));
        let r26_hi = _mm256_xor_si256(
            _mm256_load_si256(row2.add(32) as *const __m256i),
            _mm256_load_si256(row6 as *const __m256i),
        );
        let r37_hi = _mm256_xor_si256(
            _mm256_load_si256(row3.add(32) as *const __m256i),
            _mm256_load_si256(row7 as *const __m256i),
        );
        let odd_hi = _mm256_xor_si256(r26_hi, _mm256_shuffle_epi32::<0x4E>(r37_hi));
        let hi = _mm256_xor_si256(even_hi, _mm256_permute4x64_epi64::<0x4E>(odd_hi));
        (lo, hi)
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn k0_gf8_inv_ntt_avx2_pair_composite(
    base: *const u8,
    bytes: *const u8,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    unsafe {
        let w = (bytes as *const u64).read_unaligned() as usize;
        k0_gf8_inv_ntt_avx2_pair_composite_w(base, w)
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn k0_gf8_inv_ntt_avx2_pair_composite_u32(
    base: *const u8,
    w: usize,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    unsafe {
        use core::arch::x86_64::*;
        let row0 = base.add((w & 0xFF) << 6);
        let r04_lo = _mm256_load_si256(row0 as *const __m256i);
        let r04_hi = _mm256_load_si256(row0.add(32) as *const __m256i);
        let row1 = base.add(((w >> 8) & 0xFF) << 6);
        let r15_lo = _mm256_load_si256(row1 as *const __m256i);
        let even_lo = _mm256_xor_si256(r04_lo, _mm256_shuffle_epi32::<0x4E>(r15_lo));
        let r15_hi = _mm256_load_si256(row1.add(32) as *const __m256i);
        let even_hi = _mm256_xor_si256(r04_hi, _mm256_shuffle_epi32::<0x4E>(r15_hi));
        let row2 = base.add(((w >> 16) & 0xFF) << 6);
        let row3 = base.add(((w >> 24) & 0xFF) << 6);
        let r26_lo = _mm256_load_si256(row2 as *const __m256i);
        let r37_lo = _mm256_load_si256(row3 as *const __m256i);
        let odd_lo = _mm256_xor_si256(r26_lo, _mm256_shuffle_epi32::<0x4E>(r37_lo));
        let lo = _mm256_xor_si256(even_lo, _mm256_permute4x64_epi64::<0x4E>(odd_lo));
        let r26_hi = _mm256_load_si256(row2.add(32) as *const __m256i);
        let r37_hi = _mm256_load_si256(row3.add(32) as *const __m256i);
        let odd_hi = _mm256_xor_si256(r26_hi, _mm256_shuffle_epi32::<0x4E>(r37_hi));
        let hi = _mm256_xor_si256(even_hi, _mm256_permute4x64_epi64::<0x4E>(odd_hi));
        (lo, hi)
    }
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn k0_gf8_shift_reduce_inner_ab_slice(
    a_ptr: *const u8,
    b_ptr: *const u8,
    inv_std_base: *const u8,
    inv_comp_base: *const u8,
    out_ptr: *mut u8,
) {
    unsafe {
        use core::arch::x86_64::*;
        if k0_has_avx512_gfni() {
            k0_gf8_shift_reduce_inner_ab_gfni_avx512_slice(a_ptr, b_ptr, inv_std_base, out_ptr);
            return;
        }
        let v_0f = _mm256_set1_epi8(0x0F);
        let v_log = _mm256_broadcastsi128_si256(_mm_setr_epi8(
            -1, 0x00, 0x01, 0x04, 0x02, 0x08, 0x05, 0x0a, 0x03, 0x0e, 0x09, 0x07, 0x06, 0x0d, 0x0b,
            0x0c,
        ));
        let v_exp0 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
            0x13,
            0xaa_u8 as i8,
            0x53,
            0xd4_u8 as i8,
            0xb9_u8 as i8,
            0xf9_u8 as i8,
            0x87_u8 as i8,
            0x6d,
            0x40,
            0x7e,
            0xea_u8 as i8,
            0x2d,
            0x3e,
            0x94_u8 as i8,
            0xc7_u8 as i8,
            0x13,
        ));
        let v_exp1 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
            0x0d,
            0x01,
            0xe1_u8 as i8,
            0x5c,
            0x0c,
            0xe0_u8 as i8,
            0xbd_u8 as i8,
            0x50,
            0xec_u8 as i8,
            0x5d,
            0xed_u8 as i8,
            0xbc_u8 as i8,
            0xb1_u8 as i8,
            0xb0_u8 as i8,
            0x51,
            0x0d,
        ));
        let v_exp01 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
            0x12,
            0x4b,
            0x0f,
            0xd8_u8 as i8,
            0x59,
            0x44,
            0xd7_u8 as i8,
            0x81_u8 as i8,
            0x1d,
            0x93_u8 as i8,
            0x56,
            0x9c_u8 as i8,
            0x8e_u8 as i8,
            0xc5_u8 as i8,
            0xcau8 as i8,
            0x12,
        ));
        let v_poly = _mm256_set1_epi8(0x1b);
        macro_rules! gf8_mul_composite_32 {
            ($a:expr, $b:expr) => {{
                let a0 = _mm256_and_si256($a, v_0f);
                let b0 = _mm256_and_si256($b, v_0f);
                let s0 = _mm256_adds_epu8(
                    _mm256_shuffle_epi8(v_log, a0),
                    _mm256_shuffle_epi8(v_log, b0),
                );
                let r0 = _mm256_sub_epi8(s0, _mm256_cmpgt_epi8(s0, v_0f));
                let p0 = _mm256_shuffle_epi8(v_exp0, r0);
                let a1 = _mm256_srli_epi16::<4>($a);
                let b1 = _mm256_srli_epi16::<4>($b);
                let a01 = _mm256_and_si256(_mm256_xor_si256($a, a1), v_0f);
                let b01 = _mm256_and_si256(_mm256_xor_si256($b, b1), v_0f);
                let a1 = _mm256_and_si256(a1, v_0f);
                let b1 = _mm256_and_si256(b1, v_0f);
                let s1 = _mm256_adds_epu8(
                    _mm256_shuffle_epi8(v_log, a1),
                    _mm256_shuffle_epi8(v_log, b1),
                );
                let r1 = _mm256_sub_epi8(s1, _mm256_cmpgt_epi8(s1, v_0f));
                let p01_acc = _mm256_xor_si256(p0, _mm256_shuffle_epi8(v_exp1, r1));
                let s01 = _mm256_adds_epu8(
                    _mm256_shuffle_epi8(v_log, a01),
                    _mm256_shuffle_epi8(v_log, b01),
                );
                let r01 = _mm256_sub_epi8(s01, _mm256_cmpgt_epi8(s01, v_0f));
                _mm256_xor_si256(p01_acc, _mm256_shuffle_epi8(v_exp01, r01))
            }};
        }
        macro_rules! gf8_mul_x_32 {
            ($x:expr) => {{
                let shifted = _mm256_add_epi8($x, $x);
                let red = _mm256_xor_si256(_mm256_shuffle_epi8(v_poly, $x), v_poly);
                _mm256_xor_si256(shifted, red)
            }};
        }
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();
        let b_base_ptr = b_ptr as *const u64;
        let b0 = b_base_ptr.read_unaligned();
        let b1 = b_base_ptr.add(1).read_unaligned();
        let b2 = b_base_ptr.add(2).read_unaligned();
        let b3 = b_base_ptr.add(3).read_unaligned();
        let b4 = b_base_ptr.add(4).read_unaligned();
        let b5 = b_base_ptr.add(5).read_unaligned();
        let b6 = b_base_ptr.add(6).read_unaligned();
        let b7 = b_base_ptr.add(7).read_unaligned();
        let all_general = (b0.wrapping_add(1) > 1)
            & (b1.wrapping_add(1) > 1)
            & (b2.wrapping_add(1) > 1)
            & (b3.wrapping_add(1) > 1)
            & (b4.wrapping_add(1) > 1)
            & (b5.wrapping_add(1) > 1)
            & (b6.wrapping_add(1) > 1)
            & (b7.wrapping_add(1) > 1);
        if all_general {
            let (a_lo, a_hi) = k0_gf8_inv_ntt_avx2_pair_composite(inv_comp_base, a_ptr.add(56));
            let (b_lo, b_hi) = k0_gf8_inv_ntt_avx2_pair_composite_w(inv_comp_base, b7 as usize);
            acc_lo = gf8_mul_composite_32!(a_lo, b_lo);
            acc_hi = gf8_mul_composite_32!(a_hi, b_hi);
            for k in (0..7).rev() {
                let off = k * 8;
                let (a_lo, a_hi) =
                    k0_gf8_inv_ntt_avx2_pair_composite(inv_comp_base, a_ptr.add(off));
                let (b_lo, b_hi) =
                    k0_gf8_inv_ntt_avx2_pair_composite(inv_comp_base, b_ptr.add(off));
                let y_lo = gf8_mul_composite_32!(a_lo, b_lo);
                acc_lo = _mm256_xor_si256(gf8_mul_x_32!(acc_lo), y_lo);
                let y_hi = gf8_mul_composite_32!(a_hi, b_hi);
                acc_hi = _mm256_xor_si256(gf8_mul_x_32!(acc_hi), y_hi);
            }
        } else if (b0 & b1 & b2 & b3 & b4 & b5 & b6 & b7) == !0u64 {
            (acc_lo, acc_hi) = k0_gf8_inv_ntt_avx2_pair_composite(inv_std_base, a_ptr.add(56));
            for k in (0..7).rev() {
                let (y_lo, y_hi) =
                    k0_gf8_inv_ntt_avx2_pair_composite(inv_std_base, a_ptr.add(k * 8));
                acc_lo = _mm256_xor_si256(gf8_mul_x_32!(acc_lo), y_lo);
                acc_hi = _mm256_xor_si256(gf8_mul_x_32!(acc_hi), y_hi);
            }
        } else if (b4 & b5 & b6 & b7) == !0u64
            && (b0.wrapping_add(1) > 1)
            && (b1.wrapping_add(1) > 1)
            && (b2.wrapping_add(1) > 1)
            && (b3.wrapping_add(1) > 1)
        {
            (acc_lo, acc_hi) = k0_gf8_inv_ntt_avx2_pair_composite(inv_std_base, a_ptr.add(56));
            for k in (4..7).rev() {
                let (y_lo, y_hi) =
                    k0_gf8_inv_ntt_avx2_pair_composite(inv_std_base, a_ptr.add(k * 8));
                acc_lo = _mm256_xor_si256(gf8_mul_x_32!(acc_lo), y_lo);
                acc_hi = _mm256_xor_si256(gf8_mul_x_32!(acc_hi), y_hi);
            }
            for k in (0..4).rev() {
                let off = k * 8;
                let (a_lo, a_hi) =
                    k0_gf8_inv_ntt_avx2_pair_composite(inv_comp_base, a_ptr.add(off));
                let (b_lo, b_hi) =
                    k0_gf8_inv_ntt_avx2_pair_composite(inv_comp_base, b_ptr.add(off));
                let y_lo = gf8_mul_composite_32!(a_lo, b_lo);
                acc_lo = _mm256_xor_si256(gf8_mul_x_32!(acc_lo), y_lo);
                let y_hi = gf8_mul_composite_32!(a_hi, b_hi);
                acc_hi = _mm256_xor_si256(gf8_mul_x_32!(acc_hi), y_hi);
            }
        } else if (b2 | b3 | b4 | b5 | b6 | b7) == 0 {
            if b1 != 0 {
                if b1 == !0u64 {
                    (acc_lo, acc_hi) =
                        k0_gf8_inv_ntt_avx2_pair_composite(inv_std_base, a_ptr.add(8));
                } else {
                    let a1 = (a_ptr.add(8) as *const u64).read_unaligned();
                    let (a_lo, a_hi, b_lo, b_hi) = if ((a1 | b1) >> 32) == 0 {
                        let (al, ah) =
                            k0_gf8_inv_ntt_avx2_pair_composite_u32(inv_comp_base, a1 as usize);
                        let (bl, bh) =
                            k0_gf8_inv_ntt_avx2_pair_composite_u32(inv_comp_base, b1 as usize);
                        (al, ah, bl, bh)
                    } else {
                        let (al, ah) =
                            k0_gf8_inv_ntt_avx2_pair_composite_w(inv_comp_base, a1 as usize);
                        let (bl, bh) =
                            k0_gf8_inv_ntt_avx2_pair_composite_w(inv_comp_base, b1 as usize);
                        (al, ah, bl, bh)
                    };
                    acc_lo = gf8_mul_composite_32!(a_lo, b_lo);
                    acc_hi = gf8_mul_composite_32!(a_hi, b_hi);
                }
                acc_lo = gf8_mul_x_32!(acc_lo);
                acc_hi = gf8_mul_x_32!(acc_hi);
                if b0 != 0 {
                    if b0 == !0u64 {
                        let (y_lo, y_hi) = k0_gf8_inv_ntt_avx2_pair_composite(inv_std_base, a_ptr);
                        acc_lo = _mm256_xor_si256(acc_lo, y_lo);
                        acc_hi = _mm256_xor_si256(acc_hi, y_hi);
                    } else {
                        let (a_lo, a_hi) = k0_gf8_inv_ntt_avx2_pair_composite(inv_comp_base, a_ptr);
                        let (b_lo, b_hi) =
                            k0_gf8_inv_ntt_avx2_pair_composite_w(inv_comp_base, b0 as usize);
                        acc_lo = _mm256_xor_si256(acc_lo, gf8_mul_composite_32!(a_lo, b_lo));
                        acc_hi = _mm256_xor_si256(acc_hi, gf8_mul_composite_32!(a_hi, b_hi));
                    }
                }
            } else if b0 != 0 {
                if b0 == !0u64 {
                    (acc_lo, acc_hi) = k0_gf8_inv_ntt_avx2_pair_composite(inv_std_base, a_ptr);
                } else {
                    let (a_lo, a_hi) = k0_gf8_inv_ntt_avx2_pair_composite(inv_comp_base, a_ptr);
                    let (b_lo, b_hi) =
                        k0_gf8_inv_ntt_avx2_pair_composite_w(inv_comp_base, b0 as usize);
                    acc_lo = gf8_mul_composite_32!(a_lo, b_lo);
                    acc_hi = gf8_mul_composite_32!(a_hi, b_hi);
                }
            }
        } else {
            let b_words = [b0, b1, b2, b3, b4, b5, b6, b7];
            let mut started = false;
            for k in (0..8).rev() {
                let off = k * 8;
                let bw = b_words[k];
                if bw == 0 {
                    if started {
                        acc_lo = gf8_mul_x_32!(acc_lo);
                        acc_hi = gf8_mul_x_32!(acc_hi);
                    }
                } else {
                    let (y_lo, y_hi) = if bw == !0u64 {
                        k0_gf8_inv_ntt_avx2_pair_composite(inv_std_base, a_ptr.add(off))
                    } else {
                        let (a_lo, a_hi) =
                            k0_gf8_inv_ntt_avx2_pair_composite(inv_comp_base, a_ptr.add(off));
                        let (b_lo, b_hi) =
                            k0_gf8_inv_ntt_avx2_pair_composite_w(inv_comp_base, bw as usize);
                        (
                            gf8_mul_composite_32!(a_lo, b_lo),
                            gf8_mul_composite_32!(a_hi, b_hi),
                        )
                    };
                    if !started {
                        acc_lo = y_lo;
                        acc_hi = y_hi;
                        started = true;
                    } else {
                        acc_lo = _mm256_xor_si256(gf8_mul_x_32!(acc_lo), y_lo);
                        acc_hi = _mm256_xor_si256(gf8_mul_x_32!(acc_hi), y_hi);
                    }
                }
            }
        }
        _mm256_storeu_si256(out_ptr as *mut __m256i, acc_lo);
        _mm256_storeu_si256(out_ptr.add(32) as *mut __m256i, acc_hi);
    }
}
#[inline(always)]
pub fn k0_has_avx512_gfni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use core::sync::atomic::{AtomicU8, Ordering};
        static HAS_GFNI: AtomicU8 = AtomicU8::new(0);
        let v = HAS_GFNI.load(Ordering::Relaxed);
        if v != 0 {
            return v == 1;
        }
        let ok = std::env::var_os("FLOCK_NO_GFNI").is_none()
            && std::is_x86_feature_detected!("gfni")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("vpclmulqdq");
        HAS_GFNI.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
        ok
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn k0_gf8_inv_ntt_avx512_register_w(
    base: *const u8,
    w: usize,
) -> core::arch::x86_64::__m512i {
    unsafe {
        use core::arch::x86_64::*;
        let row0 = _mm512_loadu_si512(base.add((w & 0xFF) << 6) as *const __m512i);
        let row1 = _mm512_loadu_si512(base.add(((w >> 8) & 0xFF) << 6) as *const __m512i);
        let row2 = _mm512_loadu_si512(base.add(((w >> 16) & 0xFF) << 6) as *const __m512i);
        let row3 = _mm512_loadu_si512(base.add(((w >> 24) & 0xFF) << 6) as *const __m512i);
        if (w >> 32) == 0 {
            let even = _mm512_xor_si512(row0, _mm512_shuffle_epi32::<0x4E>(row1));
            let odd = _mm512_xor_si512(row2, _mm512_shuffle_epi32::<0x4E>(row3));
            return _mm512_xor_si512(even, _mm512_shuffle_i64x2::<0xB1>(odd, odd));
        }
        let row4 = _mm512_loadu_si512(base.add(((w >> 32) & 0xFF) << 6) as *const __m512i);
        let row5 = _mm512_loadu_si512(base.add(((w >> 40) & 0xFF) << 6) as *const __m512i);
        let row6 = _mm512_loadu_si512(base.add(((w >> 48) & 0xFF) << 6) as *const __m512i);
        let row7 = _mm512_loadu_si512(base.add((w >> 56) << 6) as *const __m512i);
        let r04 = _mm512_xor_si512(row0, _mm512_shuffle_i64x2::<0x4E>(row4, row4));
        let r15 = _mm512_xor_si512(row1, _mm512_shuffle_i64x2::<0x4E>(row5, row5));
        let r26 = _mm512_xor_si512(row2, _mm512_shuffle_i64x2::<0x4E>(row6, row6));
        let r37 = _mm512_xor_si512(row3, _mm512_shuffle_i64x2::<0x4E>(row7, row7));
        let even = _mm512_xor_si512(r04, _mm512_shuffle_epi32::<0x4E>(r15));
        let odd = _mm512_xor_si512(r26, _mm512_shuffle_epi32::<0x4E>(r37));
        _mm512_xor_si512(even, _mm512_shuffle_i64x2::<0xB1>(odd, odd))
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,gfni")]
#[inline]
pub unsafe fn k0_gf8_shift_reduce_inner_ab_gfni_avx512_slice(
    a_ptr: *const u8,
    b_ptr: *const u8,
    inv_std_base: *const u8,
    out_ptr: *mut u8,
) {
    unsafe {
        use core::arch::x86_64::*;
        let mut acc = _mm512_setzero_si512();
        let a_words = a_ptr as *const u64;
        let b_words = b_ptr as *const u64;
        for k in 0..8usize {
            let bw = b_words.add(k).read_unaligned();
            if bw == 0 {
                continue;
            }
            let aw = a_words.add(k).read_unaligned();
            if aw == 0 {
                continue;
            }
            let xk = _mm512_set1_epi8((1u8 << k) as i8);
            let scaled = if bw == !0u64 {
                let av = k0_gf8_inv_ntt_avx512_register_w(inv_std_base, aw as usize);
                if k == 0 {
                    av
                } else {
                    _mm512_gf2p8mul_epi8(av, xk)
                }
            } else if aw == !0u64 {
                let bv = k0_gf8_inv_ntt_avx512_register_w(inv_std_base, bw as usize);
                if k == 0 {
                    bv
                } else {
                    _mm512_gf2p8mul_epi8(bv, xk)
                }
            } else {
                let av = k0_gf8_inv_ntt_avx512_register_w(inv_std_base, aw as usize);
                let av_scaled = if k == 0 {
                    av
                } else {
                    _mm512_gf2p8mul_epi8(av, xk)
                };
                let bv = k0_gf8_inv_ntt_avx512_register_w(inv_std_base, bw as usize);
                _mm512_gf2p8mul_epi8(av_scaled, bv)
            };
            acc = _mm512_xor_si512(acc, scaled);
        }
        _mm512_storeu_si512(out_ptr as *mut __m512i, acc);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
pub unsafe fn k0_byte_transpose_8x64(
    src_ptr: *const u8,
) -> (
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
    core::arch::x86_64::__m512i,
) {
    unsafe {
        use core::arch::x86_64::*;
        const BT: [i8; 64] = [
            0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50,
            58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37,
            45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
        ];
        let bt = _mm512_loadu_si512(BT.as_ptr() as *const __m512i);
        let p = src_ptr as *const __m512i;
        let t0 = _mm512_loadu_si512(p);
        let t1 = _mm512_loadu_si512(p.add(1));
        let t2 = _mm512_loadu_si512(p.add(2));
        let t3 = _mm512_loadu_si512(p.add(3));
        let t4 = _mm512_loadu_si512(p.add(4));
        let t5 = _mm512_loadu_si512(p.add(5));
        let t6 = _mm512_loadu_si512(p.add(6));
        let t7 = _mm512_loadu_si512(p.add(7));
        let e01 = _mm512_unpacklo_epi64(t0, t1);
        let o01 = _mm512_unpackhi_epi64(t0, t1);
        let e23 = _mm512_unpacklo_epi64(t2, t3);
        let o23 = _mm512_unpackhi_epi64(t2, t3);
        let e45 = _mm512_unpacklo_epi64(t4, t5);
        let o45 = _mm512_unpackhi_epi64(t4, t5);
        let e67 = _mm512_unpacklo_epi64(t6, t7);
        let o67 = _mm512_unpackhi_epi64(t6, t7);
        let s2_lo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        let s2_hi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
        let s3_lo = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        let s3_hi = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
        let h02_a = _mm512_permutex2var_epi64(e01, s2_lo, e23);
        let h46_a = _mm512_permutex2var_epi64(e01, s2_hi, e23);
        let h13_a = _mm512_permutex2var_epi64(o01, s2_lo, o23);
        let h57_a = _mm512_permutex2var_epi64(o01, s2_hi, o23);
        let h02_b = _mm512_permutex2var_epi64(e45, s2_lo, e67);
        let h46_b = _mm512_permutex2var_epi64(e45, s2_hi, e67);
        let h13_b = _mm512_permutex2var_epi64(o45, s2_lo, o67);
        let h57_b = _mm512_permutex2var_epi64(o45, s2_hi, o67);
        (
            _mm512_permutexvar_epi8(bt, _mm512_permutex2var_epi64(h02_a, s3_lo, h02_b)),
            _mm512_permutexvar_epi8(bt, _mm512_permutex2var_epi64(h13_a, s3_lo, h13_b)),
            _mm512_permutexvar_epi8(bt, _mm512_permutex2var_epi64(h02_a, s3_hi, h02_b)),
            _mm512_permutexvar_epi8(bt, _mm512_permutex2var_epi64(h13_a, s3_hi, h13_b)),
            _mm512_permutexvar_epi8(bt, _mm512_permutex2var_epi64(h46_a, s3_lo, h46_b)),
            _mm512_permutexvar_epi8(bt, _mm512_permutex2var_epi64(h57_a, s3_lo, h57_b)),
            _mm512_permutexvar_epi8(bt, _mm512_permutex2var_epi64(h46_a, s3_hi, h46_b)),
            _mm512_permutexvar_epi8(bt, _mm512_permutex2var_epi64(h57_a, s3_hi, h57_b)),
        )
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
#[inline]
pub unsafe fn k0_f128_transpose_8_to_16_gfni_mats(basis_ptr: *const u64, out_mats_ptr: *mut u64) {
    unsafe {
        use core::arch::x86_64::*;
        const I: [u8; 64] = [
            56, 48, 40, 32, 24, 16, 8, 0, 57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18, 10,
            2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36, 28, 20, 12, 4, 61, 53, 45, 37, 29,
            21, 13, 5, 62, 54, 46, 38, 30, 22, 14, 6, 63, 55, 47, 39, 31, 23, 15, 7,
        ];
        const BSWAP: [u8; 64] = [
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 23, 22, 21, 20, 19, 18, 17, 16,
            31, 30, 29, 28, 27, 26, 25, 24, 39, 38, 37, 36, 35, 34, 33, 32, 47, 46, 45, 44, 43, 42,
            41, 40, 55, 54, 53, 52, 51, 50, 49, 48, 63, 62, 61, 60, 59, 58, 57, 56,
        ];
        let p = basis_ptr as *const __m512i;
        let v0 = _mm512_loadu_si512(p);
        let v1 = _mm512_loadu_si512(p.add(1));
        let lo_idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let hi_idx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        let lo = _mm512_permutex2var_epi64(v0, lo_idx, v1);
        let hi = _mm512_permutex2var_epi64(v0, hi_idx, v1);
        let i = _mm512_loadu_si512(I.as_ptr() as *const __m512i);
        let bswap = _mm512_loadu_si512(BSWAP.as_ptr() as *const __m512i);
        let id = _mm512_set1_epi64(0x8040201008040201u64 as i64);
        let dst = out_mats_ptr as *mut __m512i;
        let t_lo = _mm512_gf2p8affine_epi64_epi8::<0>(id, _mm512_permutexvar_epi8(i, lo));
        let t_hi = _mm512_gf2p8affine_epi64_epi8::<0>(id, _mm512_permutexvar_epi8(i, hi));
        _mm512_storeu_si512(dst, _mm512_shuffle_epi8(t_lo, bswap));
        _mm512_storeu_si512(dst.add(1), _mm512_shuffle_epi8(t_hi, bswap));
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_build_ab_eq_fold_mats_gfni_slice(
    eq_lo: u64,
    eq_hi: u64,
    convert_ptr: *const u64,
    out_mats_ptr: *mut u64,
) {
    unsafe {
        use core::arch::x86_64::*;
        let zveq = _mm512_broadcast_i32x4(_mm_set_epi64x(eq_hi as i64, eq_lo as i64));
        let (za0, za1) = k0_f128_pre_reduce_const_zmm(zveq);
        let mut basis = [0u64; 16];
        for b in 0..16usize {
            let tbl = convert_ptr.add(b * 512);
            for i in 0..8usize {
                let off = 2 * (1usize << i);
                basis[2 * i] = *tbl.add(off);
                basis[2 * i + 1] = *tbl.add(off + 1);
            }
            let bp = basis.as_mut_ptr() as *mut __m512i;
            let s0 = k0_f128_mul_const_zmm(_mm512_loadu_si512(bp), za0, za1);
            let s1 = k0_f128_mul_const_zmm(_mm512_loadu_si512(bp.add(1)), za0, za1);
            _mm512_storeu_si512(bp, s0);
            _mm512_storeu_si512(bp.add(1), s1);
            k0_f128_transpose_8_to_16_gfni_mats(basis.as_ptr(), out_mats_ptr.add(b * 16));
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,gfni")]
#[inline]
pub unsafe fn k0_gf8_urm_convert_ab_nomul_gfni_slice(
    ab_rows_ptr: *const u8,
    n_b_med: usize,
    mats_ptr: *const u64,
    out_banks_ptr: *mut u8,
) {
    unsafe {
        use core::arch::x86_64::*;
        if n_b_med == 0 {
            return;
        }
        for plane_pair in 0..8usize {
            let p0 = plane_pair * 2;
            let p1 = p0 + 1;
            let mut acc0 = _mm512_setzero_si512();
            let mut acc1 = _mm512_setzero_si512();
            let mut b = 0usize;
            while b + 2 <= n_b_med {
                let v0 = _mm512_loadu_si512(ab_rows_ptr.add(b * 64) as *const __m512i);
                let v1 = _mm512_loadu_si512(ab_rows_ptr.add((b + 1) * 64) as *const __m512i);
                let m00 = _mm512_set1_epi64(*mats_ptr.add(b * 16 + p0) as i64);
                let m10 = _mm512_set1_epi64(*mats_ptr.add((b + 1) * 16 + p0) as i64);
                let m01 = _mm512_set1_epi64(*mats_ptr.add(b * 16 + p1) as i64);
                let m11 = _mm512_set1_epi64(*mats_ptr.add((b + 1) * 16 + p1) as i64);
                let t00 = _mm512_gf2p8affine_epi64_epi8::<0>(v0, m00);
                let t10 = _mm512_gf2p8affine_epi64_epi8::<0>(v1, m10);
                let t01 = _mm512_gf2p8affine_epi64_epi8::<0>(v0, m01);
                let t11 = _mm512_gf2p8affine_epi64_epi8::<0>(v1, m11);
                acc0 = _mm512_ternarylogic_epi64::<0x96>(acc0, t00, t10);
                acc1 = _mm512_ternarylogic_epi64::<0x96>(acc1, t01, t11);
                b += 2;
            }
            if b < n_b_med {
                let v0 = _mm512_loadu_si512(ab_rows_ptr.add(b * 64) as *const __m512i);
                let m00 = _mm512_set1_epi64(*mats_ptr.add(b * 16 + p0) as i64);
                let m01 = _mm512_set1_epi64(*mats_ptr.add(b * 16 + p1) as i64);
                acc0 = _mm512_xor_si512(acc0, _mm512_gf2p8affine_epi64_epi8::<0>(v0, m00));
                acc1 = _mm512_xor_si512(acc1, _mm512_gf2p8affine_epi64_epi8::<0>(v0, m01));
            }
            let dst0 = out_banks_ptr.add(p0 * 64) as *mut __m512i;
            let dst1 = out_banks_ptr.add(p1 * 64) as *mut __m512i;
            _mm512_storeu_si512(dst0, _mm512_xor_si512(_mm512_loadu_si512(dst0), acc0));
            _mm512_storeu_si512(dst1, _mm512_xor_si512(_mm512_loadu_si512(dst1), acc1));
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
pub unsafe fn k0_f128_plane_bank_to_f128_slice(plane_bank_ptr: *const u8, out_f128_ptr: *mut u64) {
    unsafe {
        use core::arch::x86_64::*;
        let (lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7) = k0_byte_transpose_8x64(plane_bank_ptr);
        let (hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7) =
            k0_byte_transpose_8x64(plane_bank_ptr.add(512));
        let il_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        let il_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        let dst = out_f128_ptr as *mut __m512i;
        macro_rules! store_pair {
            ($idx:expr, $lo:expr, $hi:expr) => {{
                _mm512_storeu_si512(
                    dst.add(2 * $idx),
                    _mm512_permutex2var_epi64($lo, il_lo, $hi),
                );
                _mm512_storeu_si512(
                    dst.add(2 * $idx + 1),
                    _mm512_permutex2var_epi64($lo, il_hi, $hi),
                );
            }};
        }
        store_pair!(0, lo0, hi0);
        store_pair!(1, lo1, hi1);
        store_pair!(2, lo2, hi2);
        store_pair!(3, lo3, hi3);
        store_pair!(4, lo4, hi4);
        store_pair!(5, lo5, hi5);
        store_pair!(6, lo6, hi6);
        store_pair!(7, lo7, hi7);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_fold_ab_plane_banks_slice(
    plane_banks_ptr: *const u8,
    weights_ptr: *const u64,
    n_banks: usize,
    out_f128_ptr: *mut u64,
) {
    unsafe {
        use core::arch::x86_64::*;
        if n_banks == 0 {
            return;
        }
        let mut bank_f128_uninit = core::mem::MaybeUninit::<[[u64; 128]; 8]>::uninit();
        let bank_f128 = &mut *bank_f128_uninit.as_mut_ptr();
        let mut c0 = [_mm512_setzero_si512(); 8];
        let mut c1 = [_mm512_setzero_si512(); 8];
        for idx in 0..n_banks {
            k0_f128_plane_bank_to_f128_slice(
                plane_banks_ptr.add(idx * 1024),
                bank_f128[idx].as_mut_ptr(),
            );
            let w_lo = *weights_ptr.add(2 * idx);
            let w_hi = *weights_ptr.add(2 * idx + 1);
            let zvw = _mm512_broadcast_i32x4(_mm_set_epi64x(w_hi as i64, w_lo as i64));
            let (r0, r1) = k0_f128_pre_reduce_const_zmm(zvw);
            c0[idx] = r0;
            c1[idx] = r1;
        }
        for chunk in 0..16usize {
            let off = chunk * 8;
            let dst = out_f128_ptr.add(off) as *mut __m512i;
            let mut acc = _mm512_loadu_si512(dst);
            for idx in 0..n_banks {
                let v = _mm512_loadu_si512(bank_f128[idx].as_ptr().add(off) as *const __m512i);
                let prod = k0_f128_mul_const_zmm(v, c0[idx], c1[idx]);
                acc = _mm512_xor_si512(acc, prod);
            }
            _mm512_storeu_si512(dst, acc);
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_urm_accumulate_convert_batch_ab_gfni_avx512_slice(
    batch_ab_ptr: *const u8,
    batch_n_b_med_ptr: *const u8,
    batch_eq_lo_ptr: *const u64,
    batch_len: usize,
    convert_ptr: *const u64,
    partial_ab_ptr: *mut u64,
) {
    unsafe {
        use core::arch::x86_64::*;
        static UNIT_MATS_CACHE: std::sync::OnceLock<(usize, [u64; 256])> =
            std::sync::OnceLock::new();
        let cached = UNIT_MATS_CACHE.get_or_init(|| {
            let mut m = [0u64; 256];
            k0_f128_build_ab_eq_fold_mats_gfni_slice(1, 0, convert_ptr, m.as_mut_ptr());
            (convert_ptr as usize, m)
        });
        let mut local_mats: [u64; 256];
        let mats_ptr = if cached.0 == convert_ptr as usize {
            cached.1.as_ptr()
        } else {
            local_mats = [0u64; 256];
            k0_f128_build_ab_eq_fold_mats_gfni_slice(1, 0, convert_ptr, local_mats.as_mut_ptr());
            local_mats.as_ptr()
        };
        let mut conv_buf0_uninit = core::mem::MaybeUninit::<[[__m128i; 64]; 8]>::uninit();
        let conv_buf0 = &mut *conv_buf0_uninit.as_mut_ptr();
        let mut zveqs = [_mm512_setzero_si512(); 8];
        for b_idx in 0..batch_len {
            let veq = _mm_loadu_si128(batch_eq_lo_ptr.add(2 * b_idx) as *const __m128i);
            zveqs[b_idx] = _mm512_broadcast_i32x4(veq);
            let n_b_med = *batch_n_b_med_ptr.add(b_idx) as usize;
            if n_b_med == 0 {
                conv_buf0[b_idx].fill(_mm_setzero_si128());
            } else {
                let mut plane_bank = [[0u8; 64]; 16];
                k0_gf8_urm_convert_ab_nomul_gfni_slice(
                    batch_ab_ptr.add(b_idx * 1024),
                    n_b_med,
                    mats_ptr,
                    plane_bank.as_mut_ptr() as *mut u8,
                );
                k0_f128_plane_bank_to_f128_slice(
                    plane_bank.as_ptr() as *const u8,
                    conv_buf0[b_idx].as_mut_ptr() as *mut u64,
                );
            }
        }
        for group in 0..16usize {
            let l_base = group * 4;
            let (mut z_ll, mut z_mid, mut z_hh) = (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            let mut b_idx = 0usize;
            while b_idx + 2 <= batch_len {
                let ab_a =
                    _mm512_loadu_si512(conv_buf0[b_idx].as_ptr().add(l_base) as *const __m512i);
                let ab_b =
                    _mm512_loadu_si512(conv_buf0[b_idx + 1].as_ptr().add(l_base) as *const __m512i);
                k0_f128_mul_acc2_zmm(
                    ab_a,
                    zveqs[b_idx],
                    ab_b,
                    zveqs[b_idx + 1],
                    &mut z_ll,
                    &mut z_mid,
                    &mut z_hh,
                );
                b_idx += 2;
            }
            while b_idx < batch_len {
                let ab_a =
                    _mm512_loadu_si512(conv_buf0[b_idx].as_ptr().add(l_base) as *const __m512i);
                k0_f128_mul_acc_zmm(ab_a, zveqs[b_idx], &mut z_ll, &mut z_mid, &mut z_hh);
                b_idx += 1;
            }
            macro_rules! store_lane {
                ($lane:literal) => {{
                    let p = partial_ab_ptr.add(6 * (l_base + $lane)) as *mut __m128i;
                    let ll_l = _mm512_extracti32x4_epi32::<$lane>(z_ll);
                    let mid_l = _mm512_extracti32x4_epi32::<$lane>(z_mid);
                    let hh_l = _mm512_extracti32x4_epi32::<$lane>(z_hh);
                    _mm_storeu_si128(p, _mm_xor_si128(_mm_loadu_si128(p), ll_l));
                    _mm_storeu_si128(p.add(1), _mm_xor_si128(_mm_loadu_si128(p.add(1)), mid_l));
                    _mm_storeu_si128(p.add(2), _mm_xor_si128(_mm_loadu_si128(p.add(2)), hh_l));
                }};
            }
            store_lane!(0);
            store_lane!(1);
            store_lane!(2);
            store_lane!(3);
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq,pclmulqdq")]
#[inline]
pub unsafe fn k0_f128_build_gf2_c4_fold_mats_gfni_slice(
    table_ptr: *const u64,
    coeffs_ptr: *const u64,
    out_bcast_mats_ptr: *mut u64,
) {
    unsafe {
        use core::arch::x86_64::*;
        let mut base_cols = [[0u64; 16]; 8];
        for j in 0..8usize {
            let tbl = table_ptr.add(j * 512);
            for b in 0..8usize {
                let off = 2 * (1usize << b);
                base_cols[j][2 * b] = *tbl.add(off);
                base_cols[j][2 * b + 1] = *tbl.add(off + 1);
            }
        }
        let mut per_a = [0u64; 128];
        let mut scaled_cols = [0u64; 16];
        for a in 0..4usize {
            let c_lo = *coeffs_ptr.add(2 * a);
            let c_hi = *coeffs_ptr.add(2 * a + 1);
            let zvc = _mm512_broadcast_i32x4(_mm_set_epi64x(c_hi as i64, c_lo as i64));
            let (za0, za1) = k0_f128_pre_reduce_const_zmm(zvc);
            for j in 0..8usize {
                let src = base_cols[j].as_ptr() as *const __m512i;
                let dst = scaled_cols.as_mut_ptr() as *mut __m512i;
                _mm512_storeu_si512(
                    dst,
                    k0_f128_mul_const_zmm(_mm512_loadu_si512(src), za0, za1),
                );
                _mm512_storeu_si512(
                    dst.add(1),
                    k0_f128_mul_const_zmm(_mm512_loadu_si512(src.add(1)), za0, za1),
                );
                k0_f128_transpose_8_to_16_gfni_mats(
                    scaled_cols.as_ptr(),
                    per_a.as_mut_ptr().add(16 * j),
                );
            }
            for j in 0..8usize {
                let dst_lo = out_bcast_mats_ptr.add(8 * (8 * a + j));
                let dst_hi = out_bcast_mats_ptr.add(8 * (32 + 8 * a + j));
                core::ptr::copy_nonoverlapping(per_a.as_ptr().add(16 * j), dst_lo, 8);
                core::ptr::copy_nonoverlapping(per_a.as_ptr().add(16 * j + 8), dst_hi, 8);
            }
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")]
#[inline]
pub unsafe fn k0_gf2_fold_64_rows_gfni_c4_slice(
    rows_ptr: *const u8,
    bcast_mats_ptr: *const u64,
    out_f128_ptr: *mut u64,
    dead_lines: u8,
) {
    unsafe {
        use core::arch::x86_64::*;
        let mut z = [_mm512_setzero_si512(); 8];
        if dead_lines == 0 {
            for (i, slot) in z.iter_mut().enumerate() {
                *slot = _mm512_loadu_si512(rows_ptr.add(64 * i) as *const __m512i);
            }
        } else {
            for (i, slot) in z.iter_mut().enumerate() {
                if dead_lines & (1u8 << i) == 0 {
                    *slot = _mm512_loadu_si512(rows_ptr.add(64 * i) as *const __m512i);
                }
            }
        }
        const BT: [i8; 64] = [
            0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50,
            58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37,
            45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
        ];
        let bt = _mm512_loadu_si512(BT.as_ptr() as *const __m512i);
        #[repr(C, align(64))]
        struct Octs([u64; 64]);
        let mut octs = core::mem::MaybeUninit::<Octs>::uninit();
        let op = octs.as_mut_ptr().cast::<u64>();
        let gather = |h: usize, s0: __m512i, s1: __m512i, s2: __m512i, s3: __m512i| {
            let g1_lo = _mm512_setr_epi64(0, 4, 8, 12, 1, 5, 9, 13);
            let g1_hi = _mm512_setr_epi64(2, 6, 10, 14, 3, 7, 11, 15);
            let s3_lo = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
            let s3_hi = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
            let p01 = _mm512_permutex2var_epi64(s0, g1_lo, s1);
            let p23 = _mm512_permutex2var_epi64(s0, g1_hi, s1);
            let p45 = _mm512_permutex2var_epi64(s2, g1_lo, s3);
            let p67 = _mm512_permutex2var_epi64(s2, g1_hi, s3);
            let base = op.add(32 * h);
            let oct = |a: usize, v: __m512i| {
                _mm512_storeu_si512(
                    base.add(8 * a) as *mut __m512i,
                    _mm512_permutexvar_epi8(bt, v),
                );
            };
            oct(0, _mm512_permutex2var_epi64(p01, s3_lo, p45));
            oct(1, _mm512_permutex2var_epi64(p01, s3_hi, p45));
            oct(2, _mm512_permutex2var_epi64(p23, s3_lo, p67));
            oct(3, _mm512_permutex2var_epi64(p23, s3_hi, p67));
        };
        gather(0, z[0], z[1], z[2], z[3]);
        gather(1, z[4], z[5], z[6], z[7]);
        let rp: *const u64 = core::hint::black_box(op as *const u64);
        let mp = bcast_mats_ptr;
        let dst = out_f128_ptr as *mut __m512i;
        macro_rules! pass {
            ($h:expr) => {{
                const H: usize = $h;
                let mp = core::hint::black_box(mp);
                let mut accp = [_mm512_setzero_si512(); 2];
                let mut accq = [_mm512_setzero_si512(); 2];
                for a in 0..4usize {
                    let b: [__m512i; 8] = core::array::from_fn(|j| {
                        _mm512_set1_epi64(*rp.add(32 * H + 8 * a + j) as i64)
                    });
                    let f: [[__m512i; 8]; 2] = core::array::from_fn(|hh| {
                        core::array::from_fn(|j| {
                            _mm512_gf2p8affine_epi64_epi8::<0>(
                                b[j],
                                _mm512_loadu_si512(
                                    mp.add(8 * (32 * hh + 8 * a + j)) as *const __m512i
                                ),
                            )
                        })
                    });
                    for hh in 0..2usize {
                        let g = f[hh];
                        let v1 = _mm512_ternarylogic_epi64::<0x96>(g[0], g[1], g[2]);
                        let v2 = _mm512_ternarylogic_epi64::<0x96>(g[3], g[4], g[5]);
                        let v3 = _mm512_ternarylogic_epi64::<0x96>(g[6], g[7], v1);
                        if a == 0 {
                            accp[hh] = v2;
                            accq[hh] = v3;
                        } else {
                            accp[hh] = _mm512_ternarylogic_epi64::<0x96>(accp[hh], accq[hh], v2);
                            accq[hh] = v3;
                        }
                    }
                }
                let il_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
                let il_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
                let a0 = _mm512_permutexvar_epi8(bt, _mm512_xor_si512(accp[0], accq[0]));
                let a1 = _mm512_permutexvar_epi8(bt, _mm512_xor_si512(accp[1], accq[1]));
                _mm512_storeu_si512(dst.add(2 * H), _mm512_permutex2var_epi64(a0, il_lo, a1));
                _mm512_storeu_si512(dst.add(2 * H + 1), _mm512_permutex2var_epi64(a0, il_hi, a1));
            }};
        }
        pass!(0);
        pass!(1);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
#[inline]
pub unsafe fn k0_gf2_fold_64_rows_gfni_slice(
    rows_ptr: *const u8,
    mats_ptr: *const u64,
    out_f128_ptr: *mut u64,
) {
    unsafe {
        use core::arch::x86_64::*;
        const BT: [i8; 64] = [
            0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50,
            58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37,
            45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
        ];
        let bt = _mm512_loadu_si512(BT.as_ptr() as *const __m512i);
        let rp = rows_ptr as *const __m512i;
        let t0 = _mm512_permutexvar_epi8(bt, _mm512_loadu_si512(rp));
        let t1 = _mm512_permutexvar_epi8(bt, _mm512_loadu_si512(rp.add(1)));
        let t2 = _mm512_permutexvar_epi8(bt, _mm512_loadu_si512(rp.add(2)));
        let t3 = _mm512_permutexvar_epi8(bt, _mm512_loadu_si512(rp.add(3)));
        let t4 = _mm512_permutexvar_epi8(bt, _mm512_loadu_si512(rp.add(4)));
        let t5 = _mm512_permutexvar_epi8(bt, _mm512_loadu_si512(rp.add(5)));
        let t6 = _mm512_permutexvar_epi8(bt, _mm512_loadu_si512(rp.add(6)));
        let t7 = _mm512_permutexvar_epi8(bt, _mm512_loadu_si512(rp.add(7)));
        let e01 = _mm512_unpacklo_epi64(t0, t1);
        let o01 = _mm512_unpackhi_epi64(t0, t1);
        let e23 = _mm512_unpacklo_epi64(t2, t3);
        let o23 = _mm512_unpackhi_epi64(t2, t3);
        let e45 = _mm512_unpacklo_epi64(t4, t5);
        let o45 = _mm512_unpackhi_epi64(t4, t5);
        let e67 = _mm512_unpacklo_epi64(t6, t7);
        let o67 = _mm512_unpackhi_epi64(t6, t7);
        let s2_lo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        let s2_hi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
        let s3_lo = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        let s3_hi = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
        let h02_a = _mm512_permutex2var_epi64(e01, s2_lo, e23);
        let h46_a = _mm512_permutex2var_epi64(e01, s2_hi, e23);
        let h13_a = _mm512_permutex2var_epi64(o01, s2_lo, o23);
        let h57_a = _mm512_permutex2var_epi64(o01, s2_hi, o23);
        let h02_b = _mm512_permutex2var_epi64(e45, s2_lo, e67);
        let h46_b = _mm512_permutex2var_epi64(e45, s2_hi, e67);
        let h13_b = _mm512_permutex2var_epi64(o45, s2_lo, o67);
        let h57_b = _mm512_permutex2var_epi64(o45, s2_hi, o67);
        let p = [
            _mm512_permutex2var_epi64(h02_a, s3_lo, h02_b),
            _mm512_permutex2var_epi64(h13_a, s3_lo, h13_b),
            _mm512_permutex2var_epi64(h02_a, s3_hi, h02_b),
            _mm512_permutex2var_epi64(h13_a, s3_hi, h13_b),
            _mm512_permutex2var_epi64(h46_a, s3_lo, h46_b),
            _mm512_permutex2var_epi64(h57_a, s3_lo, h57_b),
            _mm512_permutex2var_epi64(h46_a, s3_hi, h46_b),
            _mm512_permutex2var_epi64(h57_a, s3_hi, h57_b),
        ];
        let mut planes = [[0u8; 64]; 16];
        for k in 0..16usize {
            let g0 = _mm512_gf2p8affine_epi64_epi8::<0>(
                p[0],
                _mm512_set1_epi64(*mats_ptr.add(k) as i64),
            );
            let g1 = _mm512_gf2p8affine_epi64_epi8::<0>(
                p[1],
                _mm512_set1_epi64(*mats_ptr.add(16 + k) as i64),
            );
            let g2 = _mm512_gf2p8affine_epi64_epi8::<0>(
                p[2],
                _mm512_set1_epi64(*mats_ptr.add(32 + k) as i64),
            );
            let g3 = _mm512_gf2p8affine_epi64_epi8::<0>(
                p[3],
                _mm512_set1_epi64(*mats_ptr.add(48 + k) as i64),
            );
            let g4 = _mm512_gf2p8affine_epi64_epi8::<0>(
                p[4],
                _mm512_set1_epi64(*mats_ptr.add(64 + k) as i64),
            );
            let g5 = _mm512_gf2p8affine_epi64_epi8::<0>(
                p[5],
                _mm512_set1_epi64(*mats_ptr.add(80 + k) as i64),
            );
            let g6 = _mm512_gf2p8affine_epi64_epi8::<0>(
                p[6],
                _mm512_set1_epi64(*mats_ptr.add(96 + k) as i64),
            );
            let g7 = _mm512_gf2p8affine_epi64_epi8::<0>(
                p[7],
                _mm512_set1_epi64(*mats_ptr.add(112 + k) as i64),
            );
            let v1 = _mm512_ternarylogic_epi64::<0x96>(g0, g1, g2);
            let v2 = _mm512_ternarylogic_epi64::<0x96>(g3, g4, g5);
            let v3 = _mm512_ternarylogic_epi64::<0x96>(g6, g7, v1);
            _mm512_storeu_si512(
                planes[k].as_mut_ptr() as *mut __m512i,
                _mm512_xor_si512(v2, v3),
            );
        }
        k0_f128_plane_bank_to_f128_slice(planes.as_ptr() as *const u8, out_f128_ptr);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
pub unsafe fn k0_blake3_pow_scan_avx512(
    state_digest: &[u8; 32],
    start: u64,
    len: u64,
    bits: u32,
) -> Option<u64> {
    unsafe {
        use core::arch::x86_64::*;
        const IV: [u32; 8] = [
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
            0x5BE0CD19,
        ];
        const FLAGS: i32 = (1 | 2 | 8) as i32;
        const MSG_SCHEDULE: [[usize; 16]; 7] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
            [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
            [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
            [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
            [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
            [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
        ];
        #[derive(Clone, Copy)]
        struct Msg {
            dig: [__m512i; 8],
            n_lo: __m512i,
            n_hi: __m512i,
            zero: __m512i,
        }
        impl Msg {
            #[inline(always)]
            fn w(&self, j: usize) -> __m512i {
                match j {
                    0..=7 => self.dig[j],
                    8 => self.n_lo,
                    9 => self.n_hi,
                    _ => self.zero,
                }
            }
        }
        macro_rules! g {
            ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
                $v[$a] = _mm512_add_epi32(_mm512_add_epi32($v[$a], $v[$b]), $mx);
                $v[$d] = _mm512_ror_epi32::<16>(_mm512_xor_si512($v[$d], $v[$a]));
                $v[$c] = _mm512_add_epi32($v[$c], $v[$d]);
                $v[$b] = _mm512_ror_epi32::<12>(_mm512_xor_si512($v[$b], $v[$c]));
                $v[$a] = _mm512_add_epi32(_mm512_add_epi32($v[$a], $v[$b]), $my);
                $v[$d] = _mm512_ror_epi32::<8>(_mm512_xor_si512($v[$d], $v[$a]));
                $v[$c] = _mm512_add_epi32($v[$c], $v[$d]);
                $v[$b] = _mm512_ror_epi32::<7>(_mm512_xor_si512($v[$b], $v[$c]));
            }};
        }
        macro_rules! g_no_b {
            ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
                $v[$a] = _mm512_add_epi32(_mm512_add_epi32($v[$a], $v[$b]), $mx);
                $v[$d] = _mm512_ror_epi32::<16>(_mm512_xor_si512($v[$d], $v[$a]));
                $v[$c] = _mm512_add_epi32($v[$c], $v[$d]);
                let b1 = _mm512_ror_epi32::<12>(_mm512_xor_si512($v[$b], $v[$c]));
                $v[$a] = _mm512_add_epi32(_mm512_add_epi32($v[$a], b1), $my);
                $v[$d] = _mm512_ror_epi32::<8>(_mm512_xor_si512($v[$d], $v[$a]));
                $v[$c] = _mm512_add_epi32($v[$c], $v[$d]);
            }};
        }
        macro_rules! g_a_only {
            ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
                $v[$a] = _mm512_add_epi32(_mm512_add_epi32($v[$a], $v[$b]), $mx);
                let d1 = _mm512_ror_epi32::<16>(_mm512_xor_si512($v[$d], $v[$a]));
                let c1 = _mm512_add_epi32($v[$c], d1);
                let b1 = _mm512_ror_epi32::<12>(_mm512_xor_si512($v[$b], c1));
                $v[$a] = _mm512_add_epi32(_mm512_add_epi32($v[$a], b1), $my);
            }};
        }
        let zero = _mm512_setzero_si512();
        let mut dig = [zero; 8];
        for i in 0..8 {
            dig[i] = _mm512_set1_epi32(u32::from_le_bytes(
                state_digest[4 * i..4 * i + 4].try_into().unwrap(),
            ) as i32);
        }
        let mut prefix: [__m512i; 16] = [
            _mm512_set1_epi32(IV[0] as i32),
            _mm512_set1_epi32(IV[1] as i32),
            _mm512_set1_epi32(IV[2] as i32),
            _mm512_set1_epi32(IV[3] as i32),
            _mm512_set1_epi32(IV[4] as i32),
            _mm512_set1_epi32(IV[5] as i32),
            _mm512_set1_epi32(IV[6] as i32),
            _mm512_set1_epi32(IV[7] as i32),
            _mm512_set1_epi32(IV[0] as i32),
            _mm512_set1_epi32(IV[1] as i32),
            _mm512_set1_epi32(IV[2] as i32),
            _mm512_set1_epi32(IV[3] as i32),
            zero,
            zero,
            _mm512_set1_epi32(64),
            _mm512_set1_epi32(FLAGS),
        ];
        g!(prefix, 0, 4, 8, 12, dig[0], dig[1]);
        g!(prefix, 1, 5, 9, 13, dig[2], dig[3]);
        g!(prefix, 2, 6, 10, 14, dig[4], dig[5]);
        g!(prefix, 3, 7, 11, 15, dig[6], dig[7]);
        g!(prefix, 1, 6, 11, 12, zero, zero);
        g!(prefix, 2, 7, 8, 13, zero, zero);
        g!(prefix, 3, 4, 9, 14, zero, zero);
        let thresh = _mm512_set1_epi32(
            (if bits == 32 {
                1u32
            } else {
                1u32 << (32 - bits)
            }) as i32,
        );
        let bswap = _mm512_set4_epi32(0x0C0D_0E0F, 0x0809_0A0B, 0x0405_0607, 0x0001_0203);
        let lane_idx: [u32; 16] = core::array::from_fn(|i| i as u32);
        let lane_idx_v = _mm512_loadu_si512(lane_idx.as_ptr().cast());
        let end = start.saturating_add(len);
        let mut base = start;
        while base < end {
            let n = (end - base).min(32) as u32;
            let ms: [Msg; 2] = core::array::from_fn(|k| {
                let gb = base.wrapping_add((16 * k) as u64);
                let lo_base = _mm512_set1_epi32(gb as u32 as i32);
                let lo = _mm512_add_epi32(lo_base, lane_idx_v);
                let carry = _mm512_cmplt_epu32_mask(lo, lo_base);
                let hi_base = _mm512_set1_epi32((gb >> 32) as u32 as i32);
                let hi = _mm512_mask_add_epi32(hi_base, carry, hi_base, _mm512_set1_epi32(1));
                Msg {
                    dig,
                    n_lo: lo,
                    n_hi: hi,
                    zero,
                }
            });
            let mut vs = [prefix; 2];
            for k in 0..2 {
                g!(vs[k], 0, 5, 10, 15, ms[k].n_lo, ms[k].n_hi);
            }
            for r in 1..=5 {
                let s = &MSG_SCHEDULE[r];
                for k in 0..2 {
                    g!(vs[k], 0, 4, 8, 12, ms[k].w(s[0]), ms[k].w(s[1]));
                }
                for k in 0..2 {
                    g!(vs[k], 1, 5, 9, 13, ms[k].w(s[2]), ms[k].w(s[3]));
                }
                for k in 0..2 {
                    g!(vs[k], 2, 6, 10, 14, ms[k].w(s[4]), ms[k].w(s[5]));
                }
                for k in 0..2 {
                    g!(vs[k], 3, 7, 11, 15, ms[k].w(s[6]), ms[k].w(s[7]));
                }
                for k in 0..2 {
                    g!(vs[k], 0, 5, 10, 15, ms[k].w(s[8]), ms[k].w(s[9]));
                }
                for k in 0..2 {
                    g!(vs[k], 1, 6, 11, 12, ms[k].w(s[10]), ms[k].w(s[11]));
                }
                for k in 0..2 {
                    g!(vs[k], 2, 7, 8, 13, ms[k].w(s[12]), ms[k].w(s[13]));
                }
                for k in 0..2 {
                    g!(vs[k], 3, 4, 9, 14, ms[k].w(s[14]), ms[k].w(s[15]));
                }
            }
            let s = &MSG_SCHEDULE[6];
            for k in 0..2 {
                g_no_b!(vs[k], 0, 4, 8, 12, ms[k].w(s[0]), ms[k].w(s[1]));
            }
            for k in 0..2 {
                g!(vs[k], 1, 5, 9, 13, ms[k].w(s[2]), ms[k].w(s[3]));
            }
            for k in 0..2 {
                g_no_b!(vs[k], 2, 6, 10, 14, ms[k].w(s[4]), ms[k].w(s[5]));
            }
            for k in 0..2 {
                g!(vs[k], 3, 7, 11, 15, ms[k].w(s[6]), ms[k].w(s[7]));
            }
            for k in 0..2 {
                g_a_only!(vs[k], 0, 5, 10, 15, ms[k].w(s[8]), ms[k].w(s[9]));
            }
            for k in 0..2 {
                g_no_b!(vs[k], 2, 7, 8, 13, ms[k].w(s[12]), ms[k].w(s[13]));
            }
            let mut mask: u32 = 0;
            for k in 0..2 {
                let h = _mm512_xor_si512(vs[k][0], vs[k][8]);
                let rev = _mm512_shuffle_epi8(h, bswap);
                let m = _mm512_cmplt_epu32_mask(rev, thresh);
                mask |= (m as u32) << (16 * k);
            }
            if n < 32 {
                mask &= (1u32 << n) - 1;
            }
            if mask != 0 {
                return Some(base + u64::from(mask.trailing_zeros()));
            }
            base += u64::from(n);
        }
        None
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn k0_blake3_pow_scan_avx2(
    state_digest: &[u8; 32],
    start: u64,
    len: u64,
    bits: u32,
) -> Option<u64> {
    unsafe {
        use core::arch::x86_64::*;
        const IV: [u32; 8] = [
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
            0x5BE0CD19,
        ];
        const FLAGS: i32 = (1 | 2 | 8) as i32;
        const MSG_SCHEDULE: [[usize; 16]; 7] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
            [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
            [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
            [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
            [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
            [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
        ];
        #[derive(Clone, Copy)]
        struct Msg {
            dig: [__m256i; 8],
            n_lo: __m256i,
            n_hi: __m256i,
            zero: __m256i,
        }
        impl Msg {
            #[inline(always)]
            fn w(&self, j: usize) -> __m256i {
                match j {
                    0..=7 => self.dig[j],
                    8 => self.n_lo,
                    9 => self.n_hi,
                    _ => self.zero,
                }
            }
        }
        let r16 = _mm256_setr_epi8(
            2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11,
            8, 9, 14, 15, 12, 13,
        );
        let r8 = _mm256_setr_epi8(
            1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10,
            11, 8, 13, 14, 15, 12,
        );
        macro_rules! ror12 {
            ($x:expr) => {
                _mm256_or_si256(_mm256_srli_epi32::<12>($x), _mm256_slli_epi32::<20>($x))
            };
        }
        macro_rules! ror7 {
            ($x:expr) => {
                _mm256_or_si256(_mm256_srli_epi32::<7>($x), _mm256_slli_epi32::<25>($x))
            };
        }
        macro_rules! g {
            ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
                $v[$a] = _mm256_add_epi32(_mm256_add_epi32($v[$a], $v[$b]), $mx);
                $v[$d] = _mm256_shuffle_epi8(_mm256_xor_si256($v[$d], $v[$a]), r16);
                $v[$c] = _mm256_add_epi32($v[$c], $v[$d]);
                $v[$b] = ror12!(_mm256_xor_si256($v[$b], $v[$c]));
                $v[$a] = _mm256_add_epi32(_mm256_add_epi32($v[$a], $v[$b]), $my);
                $v[$d] = _mm256_shuffle_epi8(_mm256_xor_si256($v[$d], $v[$a]), r8);
                $v[$c] = _mm256_add_epi32($v[$c], $v[$d]);
                $v[$b] = ror7!(_mm256_xor_si256($v[$b], $v[$c]));
            }};
        }
        macro_rules! g_no_b {
            ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
                $v[$a] = _mm256_add_epi32(_mm256_add_epi32($v[$a], $v[$b]), $mx);
                $v[$d] = _mm256_shuffle_epi8(_mm256_xor_si256($v[$d], $v[$a]), r16);
                $v[$c] = _mm256_add_epi32($v[$c], $v[$d]);
                let b1 = ror12!(_mm256_xor_si256($v[$b], $v[$c]));
                $v[$a] = _mm256_add_epi32(_mm256_add_epi32($v[$a], b1), $my);
                $v[$d] = _mm256_shuffle_epi8(_mm256_xor_si256($v[$d], $v[$a]), r8);
                $v[$c] = _mm256_add_epi32($v[$c], $v[$d]);
            }};
        }
        macro_rules! g_a_only {
            ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
                $v[$a] = _mm256_add_epi32(_mm256_add_epi32($v[$a], $v[$b]), $mx);
                let d1 = _mm256_shuffle_epi8(_mm256_xor_si256($v[$d], $v[$a]), r16);
                let c1 = _mm256_add_epi32($v[$c], d1);
                let b1 = ror12!(_mm256_xor_si256($v[$b], c1));
                $v[$a] = _mm256_add_epi32(_mm256_add_epi32($v[$a], b1), $my);
            }};
        }
        let zero = _mm256_setzero_si256();
        let mut dig = [zero; 8];
        for i in 0..8 {
            dig[i] = _mm256_set1_epi32(u32::from_le_bytes(
                state_digest[4 * i..4 * i + 4].try_into().unwrap(),
            ) as i32);
        }
        let mut prefix: [__m256i; 16] = [
            _mm256_set1_epi32(IV[0] as i32),
            _mm256_set1_epi32(IV[1] as i32),
            _mm256_set1_epi32(IV[2] as i32),
            _mm256_set1_epi32(IV[3] as i32),
            _mm256_set1_epi32(IV[4] as i32),
            _mm256_set1_epi32(IV[5] as i32),
            _mm256_set1_epi32(IV[6] as i32),
            _mm256_set1_epi32(IV[7] as i32),
            _mm256_set1_epi32(IV[0] as i32),
            _mm256_set1_epi32(IV[1] as i32),
            _mm256_set1_epi32(IV[2] as i32),
            _mm256_set1_epi32(IV[3] as i32),
            zero,
            zero,
            _mm256_set1_epi32(64),
            _mm256_set1_epi32(FLAGS),
        ];
        g!(prefix, 0, 4, 8, 12, dig[0], dig[1]);
        g!(prefix, 1, 5, 9, 13, dig[2], dig[3]);
        g!(prefix, 2, 6, 10, 14, dig[4], dig[5]);
        g!(prefix, 3, 7, 11, 15, dig[6], dig[7]);
        g!(prefix, 1, 6, 11, 12, zero, zero);
        g!(prefix, 2, 7, 8, 13, zero, zero);
        g!(prefix, 3, 4, 9, 14, zero, zero);
        let sign_bias = _mm256_set1_epi32(0x8000_0000u32 as i32);
        let thresh_biased = _mm256_xor_si256(
            _mm256_set1_epi32(
                (if bits == 32 {
                    1u32
                } else {
                    1u32 << (32 - bits)
                }) as i32,
            ),
            sign_bias,
        );
        let bswap = _mm256_setr_epi8(
            3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10,
            9, 8, 15, 14, 13, 12,
        );
        let end = start.saturating_add(len);
        let mut base = start;
        while base < end {
            let n = (end - base).min(16) as u32;
            let mut lo = [0u32; 16];
            let mut hi = [0u32; 16];
            for i in 0..16 {
                let x = base.wrapping_add(i as u64);
                lo[i] = x as u32;
                hi[i] = (x >> 32) as u32;
            }
            let ms: [Msg; 2] = core::array::from_fn(|k| Msg {
                dig,
                n_lo: _mm256_loadu_si256(lo.as_ptr().add(8 * k) as *const __m256i),
                n_hi: _mm256_loadu_si256(hi.as_ptr().add(8 * k) as *const __m256i),
                zero,
            });
            let mut vs = [prefix; 2];
            for k in 0..2 {
                g!(vs[k], 0, 5, 10, 15, ms[k].n_lo, ms[k].n_hi);
            }
            for r in 1..=5 {
                let s = &MSG_SCHEDULE[r];
                for k in 0..2 {
                    g!(vs[k], 0, 4, 8, 12, ms[k].w(s[0]), ms[k].w(s[1]));
                }
                for k in 0..2 {
                    g!(vs[k], 1, 5, 9, 13, ms[k].w(s[2]), ms[k].w(s[3]));
                }
                for k in 0..2 {
                    g!(vs[k], 2, 6, 10, 14, ms[k].w(s[4]), ms[k].w(s[5]));
                }
                for k in 0..2 {
                    g!(vs[k], 3, 7, 11, 15, ms[k].w(s[6]), ms[k].w(s[7]));
                }
                for k in 0..2 {
                    g!(vs[k], 0, 5, 10, 15, ms[k].w(s[8]), ms[k].w(s[9]));
                }
                for k in 0..2 {
                    g!(vs[k], 1, 6, 11, 12, ms[k].w(s[10]), ms[k].w(s[11]));
                }
                for k in 0..2 {
                    g!(vs[k], 2, 7, 8, 13, ms[k].w(s[12]), ms[k].w(s[13]));
                }
                for k in 0..2 {
                    g!(vs[k], 3, 4, 9, 14, ms[k].w(s[14]), ms[k].w(s[15]));
                }
            }
            let s = &MSG_SCHEDULE[6];
            for k in 0..2 {
                g_no_b!(vs[k], 0, 4, 8, 12, ms[k].w(s[0]), ms[k].w(s[1]));
            }
            for k in 0..2 {
                g!(vs[k], 1, 5, 9, 13, ms[k].w(s[2]), ms[k].w(s[3]));
            }
            for k in 0..2 {
                g_no_b!(vs[k], 2, 6, 10, 14, ms[k].w(s[4]), ms[k].w(s[5]));
            }
            for k in 0..2 {
                g!(vs[k], 3, 7, 11, 15, ms[k].w(s[6]), ms[k].w(s[7]));
            }
            for k in 0..2 {
                g_a_only!(vs[k], 0, 5, 10, 15, ms[k].w(s[8]), ms[k].w(s[9]));
            }
            for k in 0..2 {
                g_no_b!(vs[k], 2, 7, 8, 13, ms[k].w(s[12]), ms[k].w(s[13]));
            }
            let mut mask: u32 = 0;
            for k in 0..2 {
                let h = _mm256_xor_si256(vs[k][0], vs[k][8]);
                let rev_biased = _mm256_xor_si256(_mm256_shuffle_epi8(h, bswap), sign_bias);
                let cmp = _mm256_cmpgt_epi32(thresh_biased, rev_biased);
                let m = _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32;
                mask |= (m & 0xFF) << (8 * k);
            }
            if n < 16 {
                mask &= (1u32 << n) - 1;
            }
            if mask != 0 {
                return Some(base + u64::from(mask.trailing_zeros()));
            }
            base += u64::from(n);
        }
        None
    }
}
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn k0_blake3_pow_scan_neon(
    state_digest: &[u8; 32],
    start: u64,
    len: u64,
    bits: u32,
) -> Option<u64> {
    unsafe {
        use core::arch::aarch64::*;
        const IV: [u32; 8] = [
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
            0x5BE0CD19,
        ];
        const FLAGS: u32 = 1 | 2 | 8;
        const MSG_SCHEDULE: [[usize; 16]; 7] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
            [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
            [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
            [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
            [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
            [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
        ];
        #[derive(Clone, Copy)]
        struct Msg {
            dig: [uint32x4_t; 8],
            n_lo: uint32x4_t,
            n_hi: uint32x4_t,
            zero: uint32x4_t,
        }
        impl Msg {
            #[inline(always)]
            fn w(&self, j: usize) -> uint32x4_t {
                match j {
                    0..=7 => self.dig[j],
                    8 => self.n_lo,
                    9 => self.n_hi,
                    _ => self.zero,
                }
            }
        }
        const IDX8: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
        let idx8 = vld1q_u8(IDX8.as_ptr());
        macro_rules! ror16 {
            ($x:expr) => {
                vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32($x)))
            };
        }
        macro_rules! ror12 {
            ($x:expr) => {
                vsriq_n_u32(vshlq_n_u32($x, 20), $x, 12)
            };
        }
        macro_rules! ror8 {
            ($x:expr) => {
                vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32($x), idx8))
            };
        }
        macro_rules! ror7 {
            ($x:expr) => {
                vsriq_n_u32(vshlq_n_u32($x, 25), $x, 7)
            };
        }
        macro_rules! g {
            ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
                $v[$a] = vaddq_u32(vaddq_u32($v[$a], $v[$b]), $mx);
                $v[$d] = ror16!(veorq_u32($v[$d], $v[$a]));
                $v[$c] = vaddq_u32($v[$c], $v[$d]);
                $v[$b] = ror12!(veorq_u32($v[$b], $v[$c]));
                $v[$a] = vaddq_u32(vaddq_u32($v[$a], $v[$b]), $my);
                $v[$d] = ror8!(veorq_u32($v[$d], $v[$a]));
                $v[$c] = vaddq_u32($v[$c], $v[$d]);
                $v[$b] = ror7!(veorq_u32($v[$b], $v[$c]));
            }};
        }
        macro_rules! g_no_b {
            ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
                $v[$a] = vaddq_u32(vaddq_u32($v[$a], $v[$b]), $mx);
                $v[$d] = ror16!(veorq_u32($v[$d], $v[$a]));
                $v[$c] = vaddq_u32($v[$c], $v[$d]);
                let b1 = ror12!(veorq_u32($v[$b], $v[$c]));
                $v[$a] = vaddq_u32(vaddq_u32($v[$a], b1), $my);
                $v[$d] = ror8!(veorq_u32($v[$d], $v[$a]));
                $v[$c] = vaddq_u32($v[$c], $v[$d]);
            }};
        }
        macro_rules! g_a_only {
            ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $mx:expr, $my:expr) => {{
                $v[$a] = vaddq_u32(vaddq_u32($v[$a], $v[$b]), $mx);
                let d1 = ror16!(veorq_u32($v[$d], $v[$a]));
                let c1 = vaddq_u32($v[$c], d1);
                let b1 = ror12!(veorq_u32($v[$b], c1));
                $v[$a] = vaddq_u32(vaddq_u32($v[$a], b1), $my);
            }};
        }
        let zero = vdupq_n_u32(0);
        let mut dig = [zero; 8];
        for i in 0..8 {
            dig[i] = vdupq_n_u32(u32::from_le_bytes(
                state_digest[4 * i..4 * i + 4].try_into().unwrap(),
            ));
        }
        let mut prefix: [uint32x4_t; 16] = [
            vdupq_n_u32(IV[0]),
            vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]),
            vdupq_n_u32(IV[3]),
            vdupq_n_u32(IV[4]),
            vdupq_n_u32(IV[5]),
            vdupq_n_u32(IV[6]),
            vdupq_n_u32(IV[7]),
            vdupq_n_u32(IV[0]),
            vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]),
            vdupq_n_u32(IV[3]),
            zero,
            zero,
            vdupq_n_u32(64),
            vdupq_n_u32(FLAGS),
        ];
        g!(prefix, 0, 4, 8, 12, dig[0], dig[1]);
        g!(prefix, 1, 5, 9, 13, dig[2], dig[3]);
        g!(prefix, 2, 6, 10, 14, dig[4], dig[5]);
        g!(prefix, 3, 7, 11, 15, dig[6], dig[7]);
        g!(prefix, 1, 6, 11, 12, zero, zero);
        g!(prefix, 2, 7, 8, 13, zero, zero);
        g!(prefix, 3, 4, 9, 14, zero, zero);
        let thresh = vdupq_n_u32(if bits == 32 { 1 } else { 1u32 << (32 - bits) });
        let lane_bits = vld1q_u32([1u32, 2, 4, 8].as_ptr());
        let end = start.saturating_add(len);
        let mut base = start;
        while base < end {
            let n = (end - base).min(16) as u32;
            let mut lo = [0u32; 16];
            let mut hi = [0u32; 16];
            for i in 0..16 {
                let x = base.wrapping_add(i as u64);
                lo[i] = x as u32;
                hi[i] = (x >> 32) as u32;
            }
            let ms: [Msg; 4] = core::array::from_fn(|k| Msg {
                dig,
                n_lo: vld1q_u32(lo.as_ptr().add(4 * k)),
                n_hi: vld1q_u32(hi.as_ptr().add(4 * k)),
                zero,
            });
            let mut vs = [prefix; 4];
            for k in 0..4 {
                g!(vs[k], 0, 5, 10, 15, ms[k].n_lo, ms[k].n_hi);
            }
            for r in 1..=5 {
                let s = &MSG_SCHEDULE[r];
                for k in 0..4 {
                    g!(vs[k], 0, 4, 8, 12, ms[k].w(s[0]), ms[k].w(s[1]));
                }
                for k in 0..4 {
                    g!(vs[k], 1, 5, 9, 13, ms[k].w(s[2]), ms[k].w(s[3]));
                }
                for k in 0..4 {
                    g!(vs[k], 2, 6, 10, 14, ms[k].w(s[4]), ms[k].w(s[5]));
                }
                for k in 0..4 {
                    g!(vs[k], 3, 7, 11, 15, ms[k].w(s[6]), ms[k].w(s[7]));
                }
                for k in 0..4 {
                    g!(vs[k], 0, 5, 10, 15, ms[k].w(s[8]), ms[k].w(s[9]));
                }
                for k in 0..4 {
                    g!(vs[k], 1, 6, 11, 12, ms[k].w(s[10]), ms[k].w(s[11]));
                }
                for k in 0..4 {
                    g!(vs[k], 2, 7, 8, 13, ms[k].w(s[12]), ms[k].w(s[13]));
                }
                for k in 0..4 {
                    g!(vs[k], 3, 4, 9, 14, ms[k].w(s[14]), ms[k].w(s[15]));
                }
            }
            let s = &MSG_SCHEDULE[6];
            for k in 0..4 {
                g_no_b!(vs[k], 0, 4, 8, 12, ms[k].w(s[0]), ms[k].w(s[1]));
            }
            for k in 0..4 {
                g!(vs[k], 1, 5, 9, 13, ms[k].w(s[2]), ms[k].w(s[3]));
            }
            for k in 0..4 {
                g_no_b!(vs[k], 2, 6, 10, 14, ms[k].w(s[4]), ms[k].w(s[5]));
            }
            for k in 0..4 {
                g!(vs[k], 3, 7, 11, 15, ms[k].w(s[6]), ms[k].w(s[7]));
            }
            for k in 0..4 {
                g_a_only!(vs[k], 0, 5, 10, 15, ms[k].w(s[8]), ms[k].w(s[9]));
            }
            for k in 0..4 {
                g_no_b!(vs[k], 2, 7, 8, 13, ms[k].w(s[12]), ms[k].w(s[13]));
            }
            let cmp: [uint32x4_t; 4] = core::array::from_fn(|k| {
                let h = veorq_u32(vs[k][0], vs[k][8]);
                vcltq_u32(
                    vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(h))),
                    thresh,
                )
            });
            let any = vorrq_u32(vorrq_u32(cmp[0], cmp[1]), vorrq_u32(cmp[2], cmp[3]));
            if vmaxvq_u32(any) != 0 {
                let mut mask = 0u32;
                for (k, &c) in cmp.iter().enumerate() {
                    mask |= vaddvq_u32(vandq_u32(c, lane_bits)) << (4 * k);
                }
                if n < 16 {
                    mask &= (1u32 << n) - 1;
                }
                if mask != 0 {
                    return Some(base + u64::from(mask.trailing_zeros()));
                }
            }
            base += u64::from(n);
        }
        None
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

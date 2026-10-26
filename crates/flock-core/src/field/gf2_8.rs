// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::ops::{Add, AddAssign, Mul, MulAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F8(pub u8);

impl F8 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    #[inline]
    pub const fn new(v: u8) -> Self {
        Self(v)
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn square(self) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            Self(crate::gf2_kernels::gf2_kern_gf8_square(self.0 as u64) as u8)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self * self
        }
    }

    #[inline]
    pub fn mul_by_x(self) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            Self(crate::gf2_kernels::gf2_kern_gf8_mul_by_x(self.0 as u64) as u8)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self * Self(0x02)
        }
    }

    pub fn inv(self) -> Self {
        let mut result = Self::ONE;
        let mut sq = self;
        for i in 0..8 {
            if (0xFEu8 >> i) & 1 != 0 {
                result *= sq;
            }
            sq = sq.square();
        }
        result
    }

    #[inline]
    pub fn unpack_even_bytes_u32(x: u32) -> u64 {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            crate::gf2_kernels::gf2_kern_unpack_even_bytes_u32(x as u64)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let t0 = x as u64;
            let t1 = (t0 | (t0 << 16)) & 0x0000_FFFF_0000_FFFF;
            (t1 | (t1 << 8)) & 0x00FF_00FF_00FF_00FF
        }
    }

    #[inline]
    pub fn pack_even_bytes_u32(r0: u64) -> u32 {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            crate::gf2_kernels::gf2_kern_pack_even_bytes_u32(r0) as u32
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let m0 = r0 & 0x00FF_00FF_00FF_00FF;
            let t0 = (m0 | (m0 >> 8)) & 0x0000_FFFF_0000_FFFF;
            ((t0 | (t0 >> 16)) & 0x0000_0000_FFFF_FFFF) as u32
        }
    }

    #[inline]
    pub fn subspace_w2(self) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            Self(crate::gf2_kernels::gf2_kern_gf8_subspace_w2(self.0 as u64) as u8)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let q = self.square() + self;
            q * (q + Self(6))
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for F8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl AddAssign for F8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Mul for F8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            Self(crate::gf2_kernels::gf2_kern_gf8_mul(self.0 as u64, rhs.0 as u64) as u8)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self(gf8_reduce(clmul8(self.0, rhs.0)))
        }
    }
}

impl MulAssign for F8 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[allow(dead_code)]
#[inline]
fn clmul8(a: u8, b: u8) -> u16 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        unsafe { clmul8_neon(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        clmul8_software(a, b)
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes")]
#[inline]
unsafe fn clmul8_neon(a: u8, b: u8) -> u16 {
    use core::arch::aarch64::*;
    let va = vdup_n_p8(a);
    let vb = vdup_n_p8(b);
    let prod = vmull_p8(va, vb);
    vgetq_lane_u16::<0>(vreinterpretq_u16_p16(prod))
}

#[allow(dead_code)]
#[inline]
const fn clmul8_software(a: u8, b: u8) -> u16 {
    let b16 = b as u16;
    let mut acc: u16 = 0;
    let mut i = 0;
    while i < 8 {
        if (a >> i) & 1 != 0 {
            acc ^= b16 << i;
        }
        i += 1;
    }
    acc
}

#[inline]
pub(crate) fn gf8_reduce(p: u16) -> u8 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        crate::gf2_kernels::gf2_kern_gf8_reduce(p as u64) as u8
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let h: u16 = p >> 8;
        let t: u16 = (p & 0xff) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4);
        let h2: u16 = t >> 8;
        ((t & 0xff) ^ h2 ^ (h2 << 1) ^ (h2 << 3) ^ (h2 << 4)) as u8
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use core::arch::aarch64::*;
    use core::mem::transmute;

    #[inline]
    pub unsafe fn gf8_reduce_vec16(c0: uint8x16_t, c1: uint8x16_t) -> uint8x16_t {
        unsafe {
            let q_plus_rsh1: poly8x8_t = transmute::<u64, poly8x8_t>(0x8d8d8d8d8d8d8d8d_u64);
            let q_star: poly8x8_t = transmute::<u64, poly8x8_t>(0x1b1b1b1b1b1b1b1b_u64);

            let cl = vuzp1q_u8(c0, c1);
            let ch = vuzp2q_u8(c0, c1);

            let t0 = vreinterpretq_u8_u16(vshlq_n_u16::<1>(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(ch)),
                q_plus_rsh1,
            ))));
            let t1 = vreinterpretq_u8_u16(vshlq_n_u16::<1>(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(ch)),
                q_plus_rsh1,
            ))));

            let tmp_hi = vuzp2q_u8(t0, t1);
            let r0 = vreinterpretq_u8_u16(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(tmp_hi)),
                q_star,
            )));
            let r1 = vreinterpretq_u8_u16(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(tmp_hi)),
                q_star,
            )));

            veorq_u8(cl, vuzp1q_u8(r0, r1))
        }
    }

    #[inline]
    pub unsafe fn gf8_mul_vec16(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        unsafe {
            let c0 = vreinterpretq_u8_u16(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(a)),
                transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(b)),
            )));
            let c1 = vreinterpretq_u8_u16(vreinterpretq_u16_p16(vmull_p8(
                transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(a)),
                transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(b)),
            )));
            gf8_reduce_vec16(c0, c1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn add_is_xor() {
        assert_eq!(F8(0x53) + F8(0xCA), F8(0x53 ^ 0xCA));
        assert_eq!(F8(0xFF) + F8(0xFF), F8::ZERO);
    }

    #[test]
    fn mul_identities() {
        for v in 0u8..=255 {
            let a = F8(v);
            assert_eq!(a * F8::ZERO, F8::ZERO);
            assert_eq!(a * F8::ONE, a);
        }
    }

    #[test]
    fn mul_known_values() {
        let x = F8(0x02);
        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        assert_eq!(x2, F8(0x04));
        assert_eq!(x4, F8(0x10));
        assert_eq!(x8, F8(0x1B));
    }

    #[test]
    fn inv_roundtrip() {
        for v in 1u8..=255 {
            let a = F8(v);
            assert_eq!(a * a.inv(), F8::ONE, "v={}", v);
        }
    }

    #[test]
    fn software_matches_neon() {
        let mut rng = Rng::new(0xDEADBEEF);
        for _ in 0..1024 {
            let a = (rng.next_u64() & 0xff) as u8;
            let b = (rng.next_u64() & 0xff) as u8;
            assert_eq!(clmul8(a, b), clmul8_software(a, b));
        }
    }

    #[test]
    fn associativity_random() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..256 {
            let a = F8((rng.next_u64() & 0xff) as u8);
            let b = F8((rng.next_u64() & 0xff) as u8);
            let c = F8((rng.next_u64() & 0xff) as u8);
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn mul_commutativity_exhaustive() {
        for a in 0u8..=255 {
            for b in 0u8..=255 {
                assert_eq!(F8(a) * F8(b), F8(b) * F8(a));
            }
        }
    }

    #[test]
    fn fips_197_test_vectors() {
        assert_eq!(F8(0x57) * F8(0x13), F8(0xfe), "FIPS-197: 57·13");
        assert_eq!(F8(0x57) * F8(0x83), F8(0xc1), "FIPS-197: 57·83");

        for a in 0u8..=255 {
            let expected = if a & 0x80 != 0 {
                (a << 1) ^ 0x1b
            } else {
                a << 1
            };
            assert_eq!(
                (F8(a) * F8(0x02)).0,
                expected,
                "xtime mismatch at a=0x{a:02x}"
            );
            assert_eq!(
                F8(a).mul_by_x().0,
                expected,
                "F8::mul_by_x mismatch at a=0x{a:02x}"
            );
            assert_eq!(
                F8(a).square(),
                F8(a) * F8(a),
                "F8::square mismatch at a=0x{a:02x}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_gf8_mul_vec16_matches_scalar() {
        use core::arch::aarch64::*;
        use core::mem::transmute;

        let mut rng = Rng::new(0xBADC0FFEE);
        for _ in 0..256 {
            let mut a_arr = [0u8; 16];
            let mut b_arr = [0u8; 16];
            for i in 0..16 {
                a_arr[i] = (rng.next_u64() & 0xff) as u8;
                b_arr[i] = (rng.next_u64() & 0xff) as u8;
            }

            let mut expected = [0u8; 16];
            for i in 0..16 {
                expected[i] = (F8(a_arr[i]) * F8(b_arr[i])).0;
            }

            let result_vec = unsafe {
                let a_v = vld1q_u8(a_arr.as_ptr());
                let b_v = vld1q_u8(b_arr.as_ptr());
                neon::gf8_mul_vec16(a_v, b_v)
            };
            let result: [u8; 16] = unsafe { transmute(result_vec) };
            assert_eq!(result, expected, "a={:02x?}, b={:02x?}", a_arr, b_arr);
        }
    }

    #[test]
    fn fermat_little_theorem() {
        for v in 1u8..=255 {
            let a = F8(v);
            let mut p = F8::ONE;
            for _ in 0..255 {
                p *= a;
            }
            assert_eq!(p, F8::ONE, "a^255 != 1 for a=0x{v:02x}");
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn k0_gf8_swar_kernels_match_scalar() {
        let mut rng = Rng::new(0xCAFEBABE);
        for _ in 0..256 {
            let a4 = rng.next_u64() & 0xFFFF_FFFF;
            let b4 = rng.next_u64() & 0xFFFF_FFFF;
            unsafe {
                let u4 = crate::gf2_kernels::gf2_kern_gf8_unreduced_x4(a4, b4);
                let r4 = crate::gf2_kernels::gf2_kern_gf8_reduce_x4(u4);
                let m4 = crate::gf2_kernels::gf2_kern_gf8_mul_x4(a4, b4);
                let ev = crate::gf2_kernels::gf2_kern_gf8_unreduced_even_x2(a4, b4);
                let od = crate::gf2_kernels::gf2_kern_gf8_unreduced_odd_x2(a4, b4);
                let deg21 = crate::gf2_kernels::gf2_kern_gf8_reduce_deg21_pair_x4(ev, od);
                assert_eq!(r4, m4);
                assert_eq!(deg21, m4);
            }
        }
    }
}

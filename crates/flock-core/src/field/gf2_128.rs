use core::ops::{Add, AddAssign, BitXor, BitXorAssign, Mul, MulAssign};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct F128 {
    pub lo: u64,
    pub hi: u64,
}

impl F128 {
    pub const ZERO: Self = Self { lo: 0, hi: 0 };
    pub const ONE: Self = Self { lo: 1, hi: 0 };

    #[inline]
    pub const fn new(lo: u64, hi: u64) -> Self {
        Self { lo, hi }
    }

    #[inline]
    pub const fn generator() -> Self {
        Self { lo: 2, hi: 0 }
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.lo == 0 && self.hi == 0
    }

    #[inline]
    pub fn mul_unreduced(self, rhs: Self) -> F256Unreduced {
        ghash_mul_unreduced(self, rhs)
    }

    #[inline(always)]
    pub fn mle_fold(self, v1: Self, r: Self) -> Self {
        self + (self + v1) * r
    }

    #[inline(always)]
    pub fn mle_fold_kernel(self, v1: Self, r: Self) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            return Self {
                lo: crate::gf2_kernels::gf2_kern_mle_fold_lo(
                    self.lo, self.hi, v1.lo, v1.hi, r.lo, r.hi,
                ),
                hi: crate::gf2_kernels::gf2_kern_mle_fold_hi(
                    self.lo, self.hi, v1.lo, v1.hi, r.lo, r.hi,
                ),
            };
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self + (self + v1) * r
        }
    }

    #[inline(always)]
    pub fn fma(self, a: Self, b: Self) -> Self {
        (self * a) + b
    }

    #[inline(always)]
    pub fn fma_kernel(self, a: Self, b: Self) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            return Self {
                lo: crate::gf2_kernels::gf2_kern_f128_fma_lo(self.lo, self.hi, a.lo, a.hi, b.lo),
                hi: crate::gf2_kernels::gf2_kern_f128_fma_hi(self.lo, self.hi, a.lo, a.hi, b.hi),
            };
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            (self * a) + b
        }
    }

    pub fn inv(self) -> Self {
        let mut r = Self::ONE;
        let mut cur = self * self;
        for _ in 1..128 {
            r *= cur;
            cur = cur * cur;
        }
        r
    }
}

impl Add for F128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo ^ rhs.lo,
            hi: self.hi ^ rhs.hi,
        }
    }
}

impl AddAssign for F128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.lo ^= rhs.lo;
        self.hi ^= rhs.hi;
    }
}

impl Mul for F128 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            let (lo, hi) = crate::gf2_kernels::k0_f128_mul_raw(self.lo, self.hi, rhs.lo, rhs.hi);
            return Self { lo, hi };
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            software::ghash_mul(self, rhs)
        }
    }
}

impl MulAssign for F128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[inline]
pub fn mul_by_x(z: F128) -> F128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return F128 {
            lo: crate::gf2_kernels::gf2_kern_mul_by_x_lo(z.lo, z.hi),
            hi: crate::gf2_kernels::gf2_kern_mul_by_x_hi(z.lo, z.hi),
        };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let carry = z.hi >> 63;
        let mask = 0u64.wrapping_sub(carry);
        F128 {
            lo: (z.lo << 1) ^ (0x87 & mask),
            hi: (z.hi << 1) | (z.lo >> 63),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F256Unreduced {
    pub r0: u64,
    pub r1: u64,
    pub r2: u64,
    pub r3: u64,
}

impl F256Unreduced {
    pub const ZERO: Self = Self {
        r0: 0,
        r1: 0,
        r2: 0,
        r3: 0,
    };

    #[inline]
    pub fn reduce(self) -> F128 {
        ghash_reduce(self.r0, self.r1, self.r2, self.r3)
    }
}

impl BitXor for F256Unreduced {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self {
            r0: self.r0 ^ rhs.r0,
            r1: self.r1 ^ rhs.r1,
            r2: self.r2 ^ rhs.r2,
            r3: self.r3 ^ rhs.r3,
        }
    }
}

impl BitXorAssign for F256Unreduced {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.r0 ^= rhs.r0;
        self.r1 ^= rhs.r1;
        self.r2 ^= rhs.r2;
        self.r3 ^= rhs.r3;
    }
}

#[inline]
pub fn ghash_reduce(r0: u64, r1: u64, r2: u64, r3: u64) -> F128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return F128 {
            lo: crate::gf2_kernels::gf2_kern_ghash_reduce_lo(r0, r2, r3),
            hi: crate::gf2_kernels::gf2_kern_ghash_reduce_hi(r1, r2, r3),
        };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
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

        F128 {
            lo: r0 ^ t_lo ^ corr,
            hi: r1 ^ t_hi,
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub mod aarch64;

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
pub mod x86_64;

#[path = "gf2_128/portable.rs"]
pub mod software;

#[inline(always)]
fn ghash_mul_unreduced(a: F128, b: F128) -> F256Unreduced {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let (r0, r1, r2, r3) =
            crate::gf2_kernels::k0_f128_mul_unreduced_raw(a.lo, a.hi, b.lo, b.hi);
        return F256Unreduced { r0, r1, r2, r3 };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        software::ghash_mul_unreduced(a, b)
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
        fn next_f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    #[test]
    fn add_identities() {
        let mut rng = Rng::new(1);
        for _ in 0..64 {
            let a = rng.next_f128();
            assert_eq!(a + F128::ZERO, a);
            assert_eq!(a + a, F128::ZERO);
        }
    }

    #[test]
    fn mul_identities() {
        let mut rng = Rng::new(2);
        for _ in 0..64 {
            let a = rng.next_f128();
            assert_eq!(a * F128::ZERO, F128::ZERO);
            assert_eq!(a * F128::ONE, a);
        }
    }

    #[test]
    fn mul_by_x_matches_mul_by_gen() {
        let mut rng = Rng::new(3);
        for _ in 0..256 {
            let a = rng.next_f128();
            assert_eq!(mul_by_x(a), a * F128::generator());
        }
    }

    #[test]
    fn deferred_reduction_matches_direct() {
        let mut rng = Rng::new(4);
        for _ in 0..64 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let direct = a * b;
            let deferred = a.mul_unreduced(b).reduce();
            assert_eq!(direct, deferred);
        }
    }

    #[test]
    fn deferred_xor_commutes_with_reduction() {
        let mut rng = Rng::new(5);
        let n = 16;
        let pairs: Vec<(F128, F128)> = (0..n).map(|_| (rng.next_f128(), rng.next_f128())).collect();

        let direct: F128 = pairs.iter().fold(F128::ZERO, |acc, (a, b)| acc + *a * *b);

        let mut acc = F256Unreduced::ZERO;
        for (a, b) in &pairs {
            acc ^= a.mul_unreduced(*b);
        }
        assert_eq!(direct, acc.reduce());
    }

    #[test]
    fn inverse_roundtrip() {
        let mut rng = Rng::new(6);
        for _ in 0..16 {
            let a = rng.next_f128();
            if a.is_zero() {
                continue;
            }
            assert_eq!(a * a.inv(), F128::ONE);
        }
    }

    #[test]
    fn associativity_random() {
        let mut rng = Rng::new(7);
        for _ in 0..64 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let c = rng.next_f128();
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn mul_commutativity() {
        let mut rng = Rng::new(91);
        for _ in 0..256 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            assert_eq!(a * b, b * a);
        }
    }

    #[test]
    fn ghash_reduction_smoking_gun() {
        let x = F128::generator();
        let x_127 = F128 {
            lo: 0,
            hi: 1u64 << 63,
        };
        assert_eq!(x * x_127, F128 { lo: 0x87, hi: 0 }, "x · x^127");

        let x_63 = F128 {
            lo: 1u64 << 63,
            hi: 0,
        };
        assert_eq!(x * x_63, F128 { lo: 0, hi: 1 }, "x · x^63 = x^64");

        let x_64 = F128 { lo: 0, hi: 1 };
        assert_eq!(x_64 * x_64, F128 { lo: 0x87, hi: 0 }, "x^64 · x^64");

        assert_eq!(x * x, F128 { lo: 4, hi: 0 }, "x^2");
    }

    #[test]
    fn high_bit_inputs_reduce_correctly() {
        let high = F128 {
            lo: 0,
            hi: 1u64 << 63,
        };
        assert_eq!(high * high.inv(), F128::ONE);
        let almost_max = F128 {
            lo: u64::MAX,
            hi: u64::MAX,
        };
        assert_eq!(almost_max * almost_max.inv(), F128::ONE);
        let just_top = F128 {
            lo: 0,
            hi: u64::MAX,
        };
        assert_eq!(just_top * just_top.inv(), F128::ONE);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_mul_vec2_matches_scalar() {
        let mut rng = Rng::new(11);
        for _ in 0..128 {
            let a0 = rng.next_f128();
            let a1 = rng.next_f128();
            let b0 = rng.next_f128();
            let b1 = rng.next_f128();
            let expected = [a0 * b0, a1 * b1];
            let result = unsafe { aarch64::ghash_mul_vec2_neon([a0, a1], [b0, b1]) };
            assert_eq!(result[0], expected[0], "lane 0");
            assert_eq!(result[1], expected[1], "lane 1");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn all_neon_variants_agree() {
        let mut rng = Rng::new(8);
        for _ in 0..128 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let sw = software::ghash_mul(a, b);
            let sb = unsafe { aarch64::ghash_mul_schoolbook(a, b) };
            let ka = unsafe { aarch64::ghash_mul_karatsuba(a, b) };
            let kb = unsafe { aarch64::ghash_mul_karatsuba_barrett(a, b) };
            let bi = unsafe { aarch64::ghash_mul_binius(a, b) };
            assert_eq!(sw, sb);
            assert_eq!(sw, ka);
            assert_eq!(sw, kb);
            assert_eq!(sw, bi);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    #[test]
    fn all_x86_variants_agree() {
        let mut rng = Rng::new(8);
        for _ in 0..128 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            let sw = software::ghash_mul(a, b);
            let sb = unsafe { x86_64::ghash_mul_schoolbook(a, b) };
            let ka = unsafe { x86_64::ghash_mul_karatsuba(a, b) };
            let kb = unsafe { x86_64::ghash_mul_karatsuba_barrett(a, b) };
            let bi = unsafe { x86_64::ghash_mul_binius(a, b) };
            let un = unsafe { x86_64::ghash_mul_unreduced_x86(a, b) }.reduce();
            assert_eq!(sw, sb, "schoolbook");
            assert_eq!(sw, ka, "karatsuba");
            assert_eq!(sw, kb, "karatsuba_barrett");
            assert_eq!(sw, bi, "binius");
            assert_eq!(sw, un, "unreduced");
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn ghash_mul_x4_matches_scalar() {
        use core::arch::x86_64::*;
        let mut rng = Rng::new(0x4A4_C0DE);
        for _ in 0..256 {
            let xs = [
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
            ];
            let ys = [
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
                rng.next_f128(),
            ];
            let got: [F128; 4] = unsafe {
                let x = _mm512_loadu_si512(xs.as_ptr() as *const __m512i);
                let y = _mm512_loadu_si512(ys.as_ptr() as *const __m512i);
                let r = x86_64::ghash_mul_x4(x, y);
                let mut out = [F128::ZERO; 4];
                _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, r);
                out
            };
            for lane in 0..4 {
                assert_eq!(
                    got[lane],
                    xs[lane] * ys[lane],
                    "lane {lane}: x4 != scalar mul"
                );
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn wide_ghash_x4_matches_scalar_deferred() {
        let mut rng = Rng::new(0xDEF_E44);
        for _ in 0..128 {
            let mut wide = unsafe { x86_64::WideGhashX4::zero() };
            let mut scalar = F256Unreduced::ZERO;
            for _ in 0..5 {
                let xs = [
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                ];
                let ys = [
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                    rng.next_f128(),
                ];
                unsafe {
                    let xv = x86_64::f128x4_loadu(xs.as_ptr());
                    let yv = x86_64::f128x4_set(ys[0], ys[1], ys[2], ys[3]);
                    wide.mul_acc(xv, yv);
                }
                for i in 0..4 {
                    scalar ^= xs[i].mul_unreduced(ys[i]);
                }
            }
            let folded = unsafe { wide.fold() };
            assert_eq!(folded, scalar, "wide fold != scalar deferred accumulator");
            assert_eq!(folded.reduce(), scalar.reduce(), "reduced values differ");
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn k0_kernels_equivalence_against_frozen_reference() {
        use crate::gf2_kernels::*;

        fn ref_mul_by_x(z: F128) -> F128 {
            let carry = z.hi >> 63;
            let mask = 0u64.wrapping_sub(carry);
            F128 {
                lo: (z.lo << 1) ^ (0x87 & mask),
                hi: (z.hi << 1) | (z.lo >> 63),
            }
        }

        fn ref_ghash_reduce(r0: u64, r1: u64, r2: u64, r3: u64) -> F128 {
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
            F128 {
                lo: r0 ^ t_lo ^ corr,
                hi: r1 ^ t_hi,
            }
        }

        fn ref_clmul64(a: u64, b: u64) -> (u64, u64) {
            let mut lo = 0u64;
            let mut hi = 0u64;
            for i in 0..64 {
                if (a >> i) & 1 != 0 {
                    lo ^= b << i;
                    if i != 0 {
                        hi ^= b >> (64 - i);
                    }
                }
            }
            (lo, hi)
        }

        fn ref_ghash_mul(a: F128, b: F128) -> F128 {
            let (ll_lo, ll_hi) = ref_clmul64(a.lo, b.lo);
            let (lh_lo, lh_hi) = ref_clmul64(a.lo, b.hi);
            let (hl_lo, hl_hi) = ref_clmul64(a.hi, b.lo);
            let (hh_lo, hh_hi) = ref_clmul64(a.hi, b.hi);
            let r0 = ll_lo;
            let r1 = ll_hi ^ lh_lo ^ hl_lo;
            let r2 = hh_lo ^ lh_hi ^ hl_hi;
            let r3 = hh_hi;
            ref_ghash_reduce(r0, r1, r2, r3)
        }

        fn ref_add_carry_parts(x: u32, y: u32) -> (u32, u32, u32, u32) {
            let sum = x.wrapping_add(y);
            let cin = sum ^ x ^ y;
            const MASK_LO31: u32 = 0x7FFF_FFFF;
            let left = (x ^ cin) & MASK_LO31;
            let right = (y ^ cin) & MASK_LO31;
            let carry_aux = left & right;
            (sum, left, right, carry_aux)
        }

        unsafe {
            assert_eq!(gf2_kern_mul_by_x_lo(1, 0), 2);
            assert_eq!(gf2_kern_mul_by_x_hi(1, 0), 0);
            assert_eq!(gf2_kern_mul_by_x_lo(0, 1u64 << 63), 0x87);
            assert_eq!(gf2_kern_mul_by_x_hi(1u64 << 63, 0), 1);
            assert_eq!(gf2_kern_ghash_reduce_lo(0, 1, 0), 0x87);
            assert_eq!(gf2_kern_ghash_reduce_hi(0, 1, 0), 0);
            assert_eq!(gf2_kern_clmul64_lo(3, 3), 5);
            assert_eq!(gf2_kern_clmul64_hi(1u64 << 63, 2), 1);
            assert_eq!(gf2_kern_ghash_mul_lo(1, 0, 1, 0), 1);
            assert_eq!(gf2_kern_ghash_mul_hi(1, 0, 1, 0), 0);
            assert_eq!(gf2_kern_ghash_mul_lo(0, 1u64 << 63, 2, 0), 0x87);
            assert_eq!(gf2_kern_ghash_mul_hi(0, 1u64 << 63, 2, 0), 0);
            assert_eq!(gf2_kern_ntt_btf_u_lo(10, 6), 12);
            assert_eq!(gf2_kern_ntt_btf_v_lo(10, 3, 6), 15);
            assert_eq!(gf2_kern_blake3_add_sum(1, 1), 2);
            assert_eq!(gf2_kern_blake3_add_carry_aux(1, 1), 3);
            assert_eq!(gf2_kern_or_u32_lo(0xdeadbeef, 8), 0xdeadbeef00);
            assert_eq!(gf2_kern_or_u32_hi(0xdeadbeef, 48), 0xdead);
        }

        let boundaries: [u64; 12] = [
            0,
            1,
            2,
            3,
            0x87,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFFF,
            1u64 << 57,
            1u64 << 62,
            1u64 << 63,
            u64::MAX,
        ];
        for &a in &boundaries {
            for &b in &boundaries {
                let z = F128 { lo: a, hi: b };
                assert_eq!(mul_by_x(z), ref_mul_by_x(z));
                assert_eq!(software::clmul64(a, b), ref_clmul64(a, b));
                let w = F128 {
                    lo: b.rotate_left(17),
                    hi: a.rotate_right(29),
                };
                assert_eq!(z * w, ref_ghash_mul(z, w));
                assert_eq!(
                    ghash_reduce(a, b, a ^ b, a.wrapping_add(b)),
                    ref_ghash_reduce(a, b, a ^ b, a.wrapping_add(b))
                );
                let x32 = a as u32;
                let y32 = b as u32;
                let (r_sum, r_left, r_right, r_aux) = ref_add_carry_parts(x32, y32);
                unsafe {
                    assert_eq!(
                        gf2_kern_blake3_add_sum(x32 as u64, y32 as u64) as u32,
                        r_sum
                    );
                    assert_eq!(
                        gf2_kern_blake3_add_left(x32 as u64, y32 as u64) as u32,
                        r_left
                    );
                    assert_eq!(
                        gf2_kern_blake3_add_right(x32 as u64, y32 as u64) as u32,
                        r_right
                    );
                    assert_eq!(
                        gf2_kern_blake3_add_carry_aux(x32 as u64, y32 as u64) as u32,
                        r_aux
                    );
                }
            }
        }

        let mut rng = Rng::new(0x5A17_F128_2026_0001);
        for _ in 0..10_000 {
            let a = rng.next_f128();
            let b = rng.next_f128();
            assert_eq!(mul_by_x(a), ref_mul_by_x(a));
            assert_eq!(a * b, ref_ghash_mul(a, b));
            let r0 = rng.next_u64();
            let r1 = rng.next_u64();
            let r2 = rng.next_u64();
            let r3 = rng.next_u64();
            assert_eq!(
                ghash_reduce(r0, r1, r2, r3),
                ref_ghash_reduce(r0, r1, r2, r3)
            );
            let x32 = r0 as u32;
            let y32 = r1 as u32;
            let (r_sum, r_left, r_right, r_aux) = ref_add_carry_parts(x32, y32);
            let shift = r2 & 63;
            unsafe {
                assert_eq!(
                    gf2_kern_blake3_add_sum(x32 as u64, y32 as u64) as u32,
                    r_sum
                );
                assert_eq!(
                    gf2_kern_blake3_add_left(x32 as u64, y32 as u64) as u32,
                    r_left
                );
                assert_eq!(
                    gf2_kern_blake3_add_right(x32 as u64, y32 as u64) as u32,
                    r_right
                );
                assert_eq!(
                    gf2_kern_blake3_add_carry_aux(x32 as u64, y32 as u64) as u32,
                    r_aux
                );
                let exp_lo = (x32 as u64) << shift;
                let exp_hi = if shift > 32 {
                    (x32 as u64) >> (64 - shift)
                } else {
                    0
                };
                assert_eq!(gf2_kern_or_u32_lo(x32 as u64, shift), exp_lo);
                assert_eq!(gf2_kern_or_u32_hi(x32 as u64, shift), exp_hi);
            }
        }
    }
}

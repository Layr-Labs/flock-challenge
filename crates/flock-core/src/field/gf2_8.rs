// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// The NEON 16-wide multiplier (`gf8_mul_vec16` / `gf8_reduce_vec16`) is a
// port of `packed_aes_16x8b_multiply` from binius64
// (https://github.com/binius-zk/binius64,
// `crates/field/src/arch/aarch64/simd_arithmetic.rs`).

//! GF(2^8) with the AES irreducible polynomial x^8 + x^4 + x^3 + x + 1.
//!
//! Reduction: x^8 ≡ x^4 + x^3 + x + 1, so the upper byte h folds back as
//!   h ^ (h<<1) ^ (h<<3) ^ (h<<4).

use core::ops::{Add, AddAssign, Mul, MulAssign};

use super::{F128, phi8};

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

    /// Multiplicative inverse via Fermat: x^254 = x^{-1} in F_{2^8}.
    /// Exponent bit pattern 0xFE = 0b11111110 — 7 squarings + 6 multiplies.
    pub fn inv(self) -> Self {
        let mut result = Self::ONE;
        let mut sq = self;
        for i in 0..8 {
            if (0xFEu8 >> i) & 1 != 0 {
                result *= sq;
            }
            sq *= sq;
        }
        result
    }
}

// In GF(2⁸), addition is bitwise XOR by definition — the `^` is correct, not a
// typo for `+` (which is what these Clippy lints guard against).
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
        Self(gf8_reduce(clmul8(self.0, rhs.0)))
    }
}

impl MulAssign for F8 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Carry-less product of two bytes; result fits in 15 bits.
#[inline]
fn clmul8(a: u8, b: u8) -> u16 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: `aes` target feature is enabled at compile time.
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

/// Software fallback / test oracle. Used when `aes` is off, and as the
/// cross-check oracle inside the `software_matches_neon` unit test.
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

/// Reduce a polynomial of degree ≤ 14 modulo x^8 + x^4 + x^3 + x + 1.
/// Two-step fold: first turns 15-bit input into ≤12-bit, second into ≤8-bit.
///
/// Exposed `pub(crate)` so the URM shift_reduce inner kernel can reuse it.
#[inline]
pub(crate) const fn gf8_reduce(p: u16) -> u8 {
    let h: u16 = p >> 8;
    let t: u16 = (p & 0xff) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4);
    let h2: u16 = t >> 8;
    ((t & 0xff) ^ h2 ^ (h2 << 1) ^ (h2 << 3) ^ (h2 << 4)) as u8
}

// ---------------------------------------------------------------------------
// Const-N unrolled gf2_8 → gf2_128 reduction paths.
//
// The byte-by-byte F8 mul then phi8-lift is the dominant serial cost around
// the BLAKE3 challenge sampling: every byte of the input runs through
// `clmul8` (PMULL on aarch64+`aes`, software schoolbook elsewhere) plus
// `gf8_reduce`, then is fed to a second table to lift the reduced F8 into
// an F128.  Two dependent table loads, two shifts, and an XOR-fold per
// byte.
//
// The const-N unrolled paths below fuse the F8 multiply and the F128 lift
// into a single 64 KB precomputed table `MUL_PHI_TABLE[a][b] = phi8(a*b)`
// so the per-iteration work drops to a single 16-byte table load (or 4 /
// 8 in the unrolled variants).  The `gf2_8->gf2_128` "reduction" in the
// method-family sense is therefore amortized across 4 or 8 F8 products per
// loop step, with 4 or 8 pre-lifted 64-bit limbs emitted per iter ready to
// be XOR-accumulated into the wider F256Unreduced accumulator or directly
// folded into the per-lane F128 bank.
// ---------------------------------------------------------------------------

/// `MUL_PHI_TABLE[a*256 + b] = φ_8(F8(a) * F8(b))`.  Fuses the
/// carry-less-mul-then-reduce of the F8 product with the F8→F128 lift in
/// one table, so a single 16-byte load per F8 product replaces the
/// `gf8_reduce(clmul8(a,b))` followed by `phi8(..)` chain.
///
/// 64 KiB = 256 * 256 * 16 B; one load, no arithmetic, no second table
/// hop.  Built once via [`MUL_PHI_TABLE_INIT`] at first use; the
/// initialization is `const`-safe.
pub static MUL_PHI_TABLE: [F128; 65536] = MUL_PHI_TABLE_INIT;

/// `const`-time initializer for [`MUL_PHI_TABLE`].  Lifted out so the
/// `const fn` body does not blow the Rust const-eval stack limit.
const MUL_PHI_TABLE_INIT: [F128; 65536] = {
    let mut table = [F128 { lo: 0, hi: 0 }; 65536];
    let mut a: usize = 0;
    while a < 256 {
        let mut b: usize = 0;
        while b < 256 {
            let fa = F8(a as u8);
            let fb = F8(b as u8);
            table[a * 256 + b] = phi8(fa * fb);
            b += 1;
        }
        a += 1;
    }
    table
};

/// Compute 4 F8 products and lift each through φ_8, returning 4 F128
/// elements (8 pre-lifted `u64` limbs) in a single fully-unrolled step.
///
/// The output is laid out as `[u64; 8]` with `[lo_i, hi_i]` for `i =
/// 0..4`, matching the wider inner loop's expectation that one iteration
/// emits 8 pre-lifted 64-bit limbs.
#[inline(always)]
pub fn gf8_mul_phi_x4(a: [u8; 4], b: [u8; 4]) -> [u64; 8] {
    // Four independent MUL_PHI_TABLE loads.  Each table entry is the
    // fused F8 product + φ_8 lift, so the caller never has to run
    // `clmul8` or `gf8_reduce` at all.
    let r0 = &MUL_PHI_TABLE[a[0] as usize * 256 + b[0] as usize];
    let r1 = &MUL_PHI_TABLE[a[1] as usize * 256 + b[1] as usize];
    let r2 = &MUL_PHI_TABLE[a[2] as usize * 256 + b[2] as usize];
    let r3 = &MUL_PHI_TABLE[a[3] as usize * 256 + b[3] as usize];
    [r0.lo, r0.hi, r1.lo, r1.hi, r2.lo, r2.hi, r3.lo, r3.hi]
}

/// Compute 8 F8 products and lift each through φ_8, returning 8 F128
/// elements (16 pre-lifted `u64` limbs) in a single fully-unrolled step.
#[inline(always)]
pub fn gf8_mul_phi_x8(a: [u8; 8], b: [u8; 8]) -> [u64; 16] {
    let r0 = &MUL_PHI_TABLE[a[0] as usize * 256 + b[0] as usize];
    let r1 = &MUL_PHI_TABLE[a[1] as usize * 256 + b[1] as usize];
    let r2 = &MUL_PHI_TABLE[a[2] as usize * 256 + b[2] as usize];
    let r3 = &MUL_PHI_TABLE[a[3] as usize * 256 + b[3] as usize];
    let r4 = &MUL_PHI_TABLE[a[4] as usize * 256 + b[4] as usize];
    let r5 = &MUL_PHI_TABLE[a[5] as usize * 256 + b[5] as usize];
    let r6 = &MUL_PHI_TABLE[a[6] as usize * 256 + b[6] as usize];
    let r7 = &MUL_PHI_TABLE[a[7] as usize * 256 + b[7] as usize];
    [
        r0.lo, r0.hi, r1.lo, r1.hi, r2.lo, r2.hi, r3.lo, r3.hi, r4.lo, r4.hi, r5.lo, r5.hi, r6.lo,
        r6.hi, r7.lo, r7.hi,
    ]
}

/// Compute 4 F8 products and lift each through φ_8, returning 4 F128
/// elements packed as a `[F128; 4]`.  Equivalent to `gf8_mul_phi_x4` but
/// returns the F128 struct form for callers that already operate on F128
/// arrays.
#[inline(always)]
pub fn gf8_mul_phi_x4_f128(a: [u8; 4], b: [u8; 4]) -> [F128; 4] {
    [
        MUL_PHI_TABLE[a[0] as usize * 256 + b[0] as usize],
        MUL_PHI_TABLE[a[1] as usize * 256 + b[1] as usize],
        MUL_PHI_TABLE[a[2] as usize * 256 + b[2] as usize],
        MUL_PHI_TABLE[a[3] as usize * 256 + b[3] as usize],
    ]
}

/// Compute 8 F8 products and lift each through φ_8, returning 8 F128
/// elements packed as a `[F128; 8]`.
#[inline(always)]
pub fn gf8_mul_phi_x8_f128(a: [u8; 8], b: [u8; 8]) -> [F128; 8] {
    [
        MUL_PHI_TABLE[a[0] as usize * 256 + b[0] as usize],
        MUL_PHI_TABLE[a[1] as usize * 256 + b[1] as usize],
        MUL_PHI_TABLE[a[2] as usize * 256 + b[2] as usize],
        MUL_PHI_TABLE[a[3] as usize * 256 + b[3] as usize],
        MUL_PHI_TABLE[a[4] as usize * 256 + b[4] as usize],
        MUL_PHI_TABLE[a[5] as usize * 256 + b[5] as usize],
        MUL_PHI_TABLE[a[6] as usize * 256 + b[6] as usize],
        MUL_PHI_TABLE[a[7] as usize * 256 + b[7] as usize],
    ]
}

/// Compute 4 F8 products as F128 elements and accumulate them into `acc`
/// (a `[F128; 4]` scratch), emitting 4 pre-lifted 64-bit limbs per call.
/// The caller XORs many of these into the wider F128 bank and reduces
/// once at the end.
///
/// This is the const-N widening of the byte-by-byte F8-mul-then-phi8
/// pattern: 4 F8 products per iteration instead of 1, with the F128
/// lift already done.
#[inline]
pub fn gf8_mul_phi_x4_unreduced(
    a: [u8; 4],
    b: [u8; 4],
    acc: &mut [F128; 4],
) {
    acc[0] += MUL_PHI_TABLE[a[0] as usize * 256 + b[0] as usize];
    acc[1] += MUL_PHI_TABLE[a[1] as usize * 256 + b[1] as usize];
    acc[2] += MUL_PHI_TABLE[a[2] as usize * 256 + b[2] as usize];
    acc[3] += MUL_PHI_TABLE[a[3] as usize * 256 + b[3] as usize];
}

/// Compute 8 F8 products and accumulate them into `acc` (a `[F128; 8]`
/// scratch) in a single fully-unrolled step.  8 pre-lifted 64-bit limbs
/// per iter.
#[inline]
pub fn gf8_mul_phi_x8_unreduced(
    a: [u8; 8],
    b: [u8; 8],
    acc: &mut [F128; 8],
) {
    acc[0] += MUL_PHI_TABLE[a[0] as usize * 256 + b[0] as usize];
    acc[1] += MUL_PHI_TABLE[a[1] as usize * 256 + b[1] as usize];
    acc[2] += MUL_PHI_TABLE[a[2] as usize * 256 + b[2] as usize];
    acc[3] += MUL_PHI_TABLE[a[3] as usize * 256 + b[3] as usize];
    acc[4] += MUL_PHI_TABLE[a[4] as usize * 256 + b[4] as usize];
    acc[5] += MUL_PHI_TABLE[a[5] as usize * 256 + b[5] as usize];
    acc[6] += MUL_PHI_TABLE[a[6] as usize * 256 + b[6] as usize];
    acc[7] += MUL_PHI_TABLE[a[7] as usize * 256 + b[7] as usize];
}

/// Reduce 4 F8 products in a single fully-unrolled step and return the
/// F2 sum of the 4 lifted F128 elements, ready to feed straight into
/// the wider sumcheck or zerocheck bank fold.
#[inline]
pub fn gf8_mul_phi_x4_reduce(a: [u8; 4], b: [u8; 4]) -> F128 {
    let r0 = MUL_PHI_TABLE[a[0] as usize * 256 + b[0] as usize];
    let r1 = MUL_PHI_TABLE[a[1] as usize * 256 + b[1] as usize];
    let r2 = MUL_PHI_TABLE[a[2] as usize * 256 + b[2] as usize];
    let r3 = MUL_PHI_TABLE[a[3] as usize * 256 + b[3] as usize];
    r0 + r1 + r2 + r3
}

/// Slice-level widening fold.  Computes `phi8(a[i] * b[i])` for
/// `i = 0..a.len()` in chunks of 4, returning a `Vec<F128>` of length
/// `a.len()`.  Tail (lengths that are not multiples of 4) is processed
/// with the scalar `F8 * F8` then `phi8` form.
#[inline]
pub fn gf8_mul_phi_chunk4(a: &[u8], b: &[u8]) -> Vec<F128> {
    assert_eq!(a.len(), b.len(), "gf8_mul_phi_chunk4: length mismatch");
    let n = a.len();
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i + 4 <= n {
        let limbs = gf8_mul_phi_x4(
            [a[i], a[i + 1], a[i + 2], a[i + 3]],
            [b[i], b[i + 1], b[i + 2], b[i + 3]],
        );
        for k in 0..4 {
            out.push(F128 {
                lo: limbs[2 * k],
                hi: limbs[2 * k + 1],
            });
        }
        i += 4;
    }
    while i < n {
        out.push(phi8(F8(a[i]) * F8(b[i])));
        i += 1;
    }
    out
}

/// Slice-level widening fold in chunks of 8.
#[inline]
pub fn gf8_mul_phi_chunk8(a: &[u8], b: &[u8]) -> Vec<F128> {
    assert_eq!(a.len(), b.len(), "gf8_mul_phi_chunk8: length mismatch");
    let n = a.len();
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i + 8 <= n {
        let limbs = gf8_mul_phi_x8(
            [
                a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4], a[i + 5], a[i + 6], a[i + 7],
            ],
            [
                b[i], b[i + 1], b[i + 2], b[i + 3], b[i + 4], b[i + 5], b[i + 6], b[i + 7],
            ],
        );
        for k in 0..8 {
            out.push(F128 {
                lo: limbs[2 * k],
                hi: limbs[2 * k + 1],
            });
        }
        i += 8;
    }
    while i < n {
        out.push(phi8(F8(a[i]) * F8(b[i])));
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// aarch64 NEON helpers: 16-lane GF(2^8) mul and reduce.
//
// These are the building blocks for the round-1 URM shift_reduce inner kernel.
//
// `vmull_p8` is a baseline NEON instruction (no aes feature needed), so the
// only cfg gate is `target_arch = "aarch64"`.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use core::arch::aarch64::*;
    use core::mem::transmute;

    /// Fold the coefficients at degrees 8..15 below degree 8 modulo the AES
    /// polynomial, for 16 high bytes in parallel.
    ///
    /// The tables contain the subset sums of:
    /// - x^8..x^11 mod p = 0x1b, 0x36, 0x6c, 0xd8
    /// - x^12..x^15 mod p = 0xab, 0x4d, 0x9a, 0x2f
    ///
    /// # Safety
    /// Uses `core::arch::aarch64` NEON intrinsics; only call on `aarch64`.
    #[inline]
    pub(crate) unsafe fn gf8_fold_high_bytes_vec16(ch: uint8x16_t) -> uint8x16_t {
        const RED_LO: [u8; 16] = [
            0x00, 0x1b, 0x36, 0x2d, 0x6c, 0x77, 0x5a, 0x41, 0xd8, 0xc3, 0xee, 0xf5, 0xb4, 0xaf,
            0x82, 0x99,
        ];
        const RED_HI: [u8; 16] = [
            0x00, 0xab, 0x4d, 0xe6, 0x9a, 0x31, 0xd7, 0x7c, 0x2f, 0x84, 0x62, 0xc9, 0xb5, 0x1e,
            0xf8, 0x53,
        ];

        unsafe {
            let lo_nibble = vandq_u8(ch, vdupq_n_u8(0x0f));
            let hi_nibble = vshrq_n_u8::<4>(ch);
            let red_lo = vld1q_u8(RED_LO.as_ptr());
            let red_hi = vld1q_u8(RED_HI.as_ptr());
            veorq_u8(vqtbl1q_u8(red_lo, lo_nibble), vqtbl1q_u8(red_hi, hi_nibble))
        }
    }

    /// Reduce 16 polynomial products (in interleaved layout `[lo0,hi0, lo1,hi1, ...]`,
    /// passed as `(c0, c1)`) modulo `x^8 + x^4 + x^3 + x + 1`, returning 16 reduced
    /// GF(2^8) values.
    ///
    /// The low byte is already reduced. The high byte is split into nibbles,
    /// folded through two 16-entry tables, and XORed with the low byte.
    ///
    /// # Safety
    /// Uses `core::arch::aarch64` NEON intrinsics; only call on `aarch64`.
    #[inline]
    pub unsafe fn gf8_reduce_vec16(c0: uint8x16_t, c1: uint8x16_t) -> uint8x16_t {
        unsafe {
            let cl = vuzp1q_u8(c0, c1); // low bytes of all 16 products
            let ch = vuzp2q_u8(c0, c1); // high bytes of all 16 products

            veorq_u8(cl, gf8_fold_high_bytes_vec16(ch))
        }
    }

    /// Element-wise multiply 16 pairs of GF(2^8) values (binius64 13-op NEON kernel).
    ///
    /// # Safety
    /// Uses `core::arch::aarch64` NEON intrinsics (PMULL); only call on `aarch64`.
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

    /// Deterministic splitmix64 PRNG for test reproducibility.
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
        // x = F8(0x02). x^2 = 0x04. x^4 = 0x10.
        // x^8 mod p = x^4 + x^3 + x + 1 = 0x1B.
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
        // If we are on aarch64+aes, sanity-check that the software path agrees.
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
        // Trivially symmetric in the formula, but free to assert over all pairs.
        for a in 0u8..=255 {
            for b in 0u8..=255 {
                assert_eq!(F8(a) * F8(b), F8(b) * F8(a));
            }
        }
    }

    #[test]
    fn fips_197_test_vectors() {
        // FIPS 197 § 4.2 (AES specification) publishes these products
        // for the GF(2^8) multiplication used by AES.
        assert_eq!(F8(0x57) * F8(0x13), F8(0xfe), "FIPS-197: 57·13");
        assert_eq!(F8(0x57) * F8(0x83), F8(0xc1), "FIPS-197: 57·83");
        // xtime: a · 0x02 (used by MixColumns), exhaustively cross-check
        // against the spec'd formula: xtime(a) = (a << 1) ^ (0x1B if a high bit).
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
            // Scalar reference: lane-wise F8 mul.
            let mut expected = [0u8; 16];
            for i in 0..16 {
                expected[i] = (F8(a_arr[i]) * F8(b_arr[i])).0;
            }
            // NEON result.
            let result_vec = unsafe {
                let a_v = vld1q_u8(a_arr.as_ptr());
                let b_v = vld1q_u8(b_arr.as_ptr());
                neon::gf8_mul_vec16(a_v, b_v)
            };
            let result: [u8; 16] = unsafe { transmute(result_vec) };
            assert_eq!(result, expected, "a={:02x?}, b={:02x?}", a_arr, b_arr);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_gf8_nibble_fold_matches_scalar_exhaustive() {
        use core::arch::aarch64::*;
        use core::mem::transmute;

        for base in (0u16..=255).step_by(16) {
            let mut high = [0u8; 16];
            let mut expected = [0u8; 16];
            for lane in 0..16 {
                high[lane] = (base + lane as u16) as u8;
                expected[lane] = gf8_reduce((high[lane] as u16) << 8);
            }

            let folded = unsafe {
                let high = vld1q_u8(high.as_ptr());
                neon::gf8_fold_high_bytes_vec16(high)
            };
            let actual: [u8; 16] = unsafe { transmute(folded) };
            assert_eq!(
                actual,
                expected,
                "high bytes {base:#04x}..={:#04x}",
                base + 15
            );
        }
    }

    #[test]
    fn fermat_little_theorem() {
        // F_{2^8}\{0} has order 255, so a^{255} = 1 for every nonzero a.
        // Strong structural check: catches any single-bit error in the
        // reduction logic, since wrong reduction breaks the cyclic group.
        for v in 1u8..=255 {
            let a = F8(v);
            let mut p = F8::ONE;
            for _ in 0..255 {
                p *= a;
            }
            assert_eq!(p, F8::ONE, "a^255 != 1 for a=0x{v:02x}");
        }
    }

    #[test]
    fn mul_phi_table_holds_phi8_of_product() {
        // Smoke check: the table must equal phi8(F8(a) * F8(b)) for every
        // (a, b).  Sample a few hot values plus zero/one corners.
        for &(a, b) in &[
            (0u8, 0u8),
            (0, 1),
            (1, 0),
            (1, 1),
            (0x57, 0x13),
            (0x57, 0x83),
            (0xFF, 0xFF),
            (0x02, 0x02),
            (0xAA, 0x55),
        ] {
            let expected = phi8(F8(a) * F8(b));
            let got = MUL_PHI_TABLE[a as usize * 256 + b as usize];
            assert_eq!(got, expected, "a={a:02x}, b={b:02x}");
        }
    }

    #[test]
    fn gf8_mul_phi_x4_matches_scalar() {
        let a = [0x12u8, 0x57, 0x83, 0xFF];
        let b = [0x34u8, 0x13, 0xAA, 0x55];
        let limbs = gf8_mul_phi_x4(a, b);
        for k in 0..4 {
            let expected = phi8(F8(a[k]) * F8(b[k]));
            assert_eq!(limbs[2 * k], expected.lo, "a={:02x} b={:02x} lo", a[k], b[k]);
            assert_eq!(limbs[2 * k + 1], expected.hi, "a={:02x} b={:02x} hi", a[k], b[k]);
        }
    }

    #[test]
    fn gf8_mul_phi_x8_matches_scalar() {
        let a = [0x01u8, 0x02, 0x03, 0x57, 0x83, 0xAA, 0xDE, 0xFF];
        let b = [0x10u8, 0x20, 0x30, 0x13, 0xC1, 0x55, 0xAD, 0x01];
        let limbs = gf8_mul_phi_x8(a, b);
        for k in 0..8 {
            let expected = phi8(F8(a[k]) * F8(b[k]));
            assert_eq!(limbs[2 * k], expected.lo, "k={k} lo");
            assert_eq!(limbs[2 * k + 1], expected.hi, "k={k} hi");
        }
    }

    #[test]
    fn gf8_mul_phi_x4_f128_matches_scalar() {
        let a = [0x12u8, 0x57, 0x83, 0xFF];
        let b = [0x34u8, 0x13, 0xAA, 0x55];
        let got = gf8_mul_phi_x4_f128(a, b);
        for k in 0..4 {
            let expected = phi8(F8(a[k]) * F8(b[k]));
            assert_eq!(got[k], expected, "k={k}");
        }
    }

    #[test]
    fn gf8_mul_phi_x8_f128_matches_scalar() {
        let a = [0x01u8, 0x02, 0x03, 0x57, 0x83, 0xAA, 0xDE, 0xFF];
        let b = [0x10u8, 0x20, 0x30, 0x13, 0xC1, 0x55, 0xAD, 0x01];
        let got = gf8_mul_phi_x8_f128(a, b);
        for k in 0..8 {
            let expected = phi8(F8(a[k]) * F8(b[k]));
            assert_eq!(got[k], expected, "k={k}");
        }
    }

    #[test]
    fn gf8_mul_phi_x4_reduce_matches_scalar_sum() {
        let a = [0x12u8, 0x57, 0x83, 0xFF];
        let b = [0x34u8, 0x13, 0xAA, 0x55];
        let expected = phi8(F8(a[0]) * F8(b[0]))
            + phi8(F8(a[1]) * F8(b[1]))
            + phi8(F8(a[2]) * F8(b[2]))
            + phi8(F8(a[3]) * F8(b[3]));
        assert_eq!(gf8_mul_phi_x4_reduce(a, b), expected);
    }

    #[test]
    fn gf8_mul_phi_x4_unreduced_accumulates() {
        let a = [0x12u8, 0x57, 0x83, 0xFF];
        let b = [0x34u8, 0x13, 0xAA, 0x55];
        let mut acc = [F128::ZERO; 4];
        gf8_mul_phi_x4_unreduced(a, b, &mut acc);
        for k in 0..4 {
            assert_eq!(acc[k], phi8(F8(a[k]) * F8(b[k])), "k={k}");
        }
    }

    #[test]
    fn gf8_mul_phi_x8_unreduced_accumulates() {
        let a = [0x01u8, 0x02, 0x03, 0x57, 0x83, 0xAA, 0xDE, 0xFF];
        let b = [0x10u8, 0x20, 0x30, 0x13, 0xC1, 0x55, 0xAD, 0x01];
        let mut acc = [F128::ZERO; 8];
        gf8_mul_phi_x8_unreduced(a, b, &mut acc);
        for k in 0..8 {
            assert_eq!(acc[k], phi8(F8(a[k]) * F8(b[k])), "k={k}");
        }
    }

    #[test]
    fn gf8_mul_phi_chunk4_and_chunk8_aligned_match_scalar() {
        // 256 distinct values so we cross all byte pair classes.
        let a: Vec<u8> = (0u8..=255).collect();
        let b: Vec<u8> = (0u8..=255).map(|x| x ^ 0xA5).collect();
        let got4 = gf8_mul_phi_chunk4(&a, &b);
        let got8 = gf8_mul_phi_chunk8(&a, &b);
        assert_eq!(got4.len(), a.len());
        assert_eq!(got8.len(), a.len());
        for i in 0..a.len() {
            let expected = phi8(F8(a[i]) * F8(b[i]));
            assert_eq!(got4[i], expected, "chunk4 i={i}");
            assert_eq!(got8[i], expected, "chunk8 i={i}");
        }
    }

    #[test]
    fn gf8_mul_phi_chunk4_and_chunk8_handle_tail() {
        // Lengths that are not multiples of 4 or 8 — tail must fall
        // through to the scalar form and still match.
        for n in 1usize..=11 {
            let a: Vec<u8> = (0u8..n as u8).collect();
            let b: Vec<u8> = (0u8..n as u8).map(|x| x.wrapping_mul(7)).collect();
            let got4 = gf8_mul_phi_chunk4(&a, &b);
            let got8 = gf8_mul_phi_chunk8(&a, &b);
            assert_eq!(got4.len(), n);
            assert_eq!(got8.len(), n);
            for i in 0..n {
                let expected = phi8(F8(a[i]) * F8(b[i]));
                assert_eq!(got4[i], expected, "n={n} chunk4 i={i}");
                assert_eq!(got8[i], expected, "n={n} chunk8 i={i}");
            }
        }
    }
}

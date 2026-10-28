use super::{F128, F256Unreduced, ghash_reduce};
use core::arch::aarch64::*;
use core::mem::transmute;

// The SHA3 extension includes EOR3; retain the two-EOR form for generic
// AArch64 builds that do not enable it.
#[cfg(target_feature = "sha3")]
#[inline(always)]
unsafe fn xor3_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    unsafe { veor3q_u64(a, b, c) }
}

#[cfg(not(target_feature = "sha3"))]
#[inline(always)]
unsafe fn xor3_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    unsafe { veorq_u64(a, veorq_u64(b, c)) }
}

/// 64×64 carry-less product, returned as a 128-bit vector.
///
/// # Safety
/// Caller must ensure the `aes` target feature is enabled (statically
/// satisfied here because every caller is itself `#[target_feature(enable = "aes")]`).
#[inline]
#[target_feature(enable = "aes")]
unsafe fn pmull(a: u64, b: u64) -> uint64x2_t {
    let prod = vmull_p64(a, b);
    // SAFETY: u128 and uint64x2_t are both 128-bit, 16-byte-aligned values;
    // transmute is a bit-level reinterpret with no UB.
    unsafe { transmute::<u128, uint64x2_t>(prod) }
}

/// Schoolbook 4 PMULL — fully independent products, then scalar reduction.
///
/// # Safety
/// Requires the `aes` target feature (compiles to PMULL); only call where
/// `aes` is statically enabled or has been runtime-detected.
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_schoolbook(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the aes target feature; helper calls below
    // require that and nothing else.
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let ll_lo = vgetq_lane_u64::<0>(p_ll);
        let ll_hi = vgetq_lane_u64::<1>(p_ll);
        let hh_lo = vgetq_lane_u64::<0>(p_hh);
        let hh_hi = vgetq_lane_u64::<1>(p_hh);
        let cross = veorq_u64(p_lh, p_hl);
        let cr_lo = vgetq_lane_u64::<0>(cross);
        let cr_hi = vgetq_lane_u64::<1>(cross);

        ghash_reduce(ll_lo, ll_hi ^ cr_lo, hh_lo ^ cr_hi, hh_hi)
    }
}

/// Karatsuba 3 PMULL — middle term depends on XOR of inputs (one stall on
/// CPUs with 2 PMULL units).
///
/// # Safety
/// Requires the `aes` target feature (compiles to PMULL); only call where
/// `aes` is statically enabled or has been runtime-detected.
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_karatsuba(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the aes target feature.
    unsafe {
        let p0 = pmull(a.lo, b.lo);
        let p1 = pmull(a.hi, b.hi);
        let pm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);

        let p0_lo = vgetq_lane_u64::<0>(p0);
        let p0_hi = vgetq_lane_u64::<1>(p0);
        let p1_lo = vgetq_lane_u64::<0>(p1);
        let p1_hi = vgetq_lane_u64::<1>(p1);
        let pm_lo = vgetq_lane_u64::<0>(pm);
        let pm_hi = vgetq_lane_u64::<1>(pm);

        let cross_lo = pm_lo ^ p0_lo ^ p1_lo;
        let cross_hi = pm_hi ^ p0_hi ^ p1_hi;

        ghash_reduce(p0_lo, p0_hi ^ cross_lo, p1_lo ^ cross_hi, p1_hi)
    }
}

/// Karatsuba 3 PMULL + Barrett 2 PMULL = 5 PMULL total.
/// `r_hi = hi_hi · 0x87` depends only on `d2`, not `d1`, so it can issue
/// in parallel with the cross-term computation.
///
/// # Safety
/// Requires the `aes` target feature (compiles to PMULL); only call where
/// `aes` is statically enabled or has been runtime-detected.
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_karatsuba_barrett(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the aes target feature.
    unsafe {
        let d0 = pmull(a.lo, b.lo);
        let d2 = pmull(a.hi, b.hi);
        let dm = pmull(a.lo ^ a.hi, b.lo ^ b.hi);
        let d1 = xor3_u64(dm, d0, d2);

        let d0_lo = vgetq_lane_u64::<0>(d0);
        let d0_hi = vgetq_lane_u64::<1>(d0);
        let d1_lo = vgetq_lane_u64::<0>(d1);
        let d1_hi = vgetq_lane_u64::<1>(d1);
        let d2_lo = vgetq_lane_u64::<0>(d2);
        let d2_hi = vgetq_lane_u64::<1>(d2);

        let lo_lo = d0_lo;
        let lo_hi = d0_hi ^ d1_lo;
        let hi_lo = d2_lo ^ d1_hi;
        let hi_hi = d2_hi;

        let r_hi = pmull(hi_hi, 0x87);
        let r_lo = pmull(hi_lo, 0x87);

        let r_lo_lo = vgetq_lane_u64::<0>(r_lo);
        let r_lo_hi = vgetq_lane_u64::<1>(r_lo);
        let r_hi_lo = vgetq_lane_u64::<0>(r_hi);
        let r_hi_hi = vgetq_lane_u64::<1>(r_hi);

        // hi_hi · 0x87 has degree ≤ 70, so r_hi_hi has at most 7 bits.
        let ov = r_hi_hi;
        let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);

        F128 {
            lo: lo_lo ^ r_lo_lo ^ corr,
            hi: lo_hi ^ r_lo_hi ^ r_hi_lo,
        }
    }
}

/// Binius-style: schoolbook 4 PMULL + recursive 2-stage reduction (2 PMULL).
/// Each stage keeps the intermediate ≤128 bits — no separate 7-bit overflow
/// term required. Total 6 PMULL but fewer scalar shifts in the dep chain.
/// Memory recorded this as the best of the four variants on M-series.
///
/// # Safety
/// Requires the `aes` target feature (compiles to PMULL); only call where
/// `aes` is statically enabled or has been runtime-detected.
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_binius(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the aes target feature.
    unsafe {
        let zero = vdupq_n_u64(0);

        let t0 = pmull(a.lo, b.lo);
        let t1a = pmull(a.lo, b.hi);
        let t1b = pmull(a.hi, b.lo);
        let t2 = pmull(a.hi, b.hi);
        let t1_cross = veorq_u64(t1a, t1b);

        // First reduce: t1 = t1 + x^64 · t2 (mod p).
        // vextq_u64::<1>(zero, t2) = {0, t2.lo} — places t2.lo into t1.hi.
        let t2_shifted = vextq_u64::<1>(zero, t2);
        let t2_hi_s = vgetq_lane_u64::<1>(t2);
        let t2_red = pmull(t2_hi_s, 0x87);
        let t1 = xor3_u64(t1_cross, t2_shifted, t2_red);

        // Second reduce: t0 = t0 + x^64 · t1 (mod p).
        let t1_shifted = vextq_u64::<1>(zero, t1);
        let t1_hi_s = vgetq_lane_u64::<1>(t1);
        let t1_red = pmull(t1_hi_s, 0x87);
        let t0 = xor3_u64(t0, t1_shifted, t1_red);

        F128 {
            lo: vgetq_lane_u64::<0>(t0),
            hi: vgetq_lane_u64::<1>(t0),
        }
    }
}

/// Batch multiply 2× F128 in parallel.
///
/// Strategy: 8 schoolbook PMULLs (4 per mul, all independent), repack the
/// four unreduced 64-bit words `(r0, r1, r2, r3)` of each product into
/// lane-paired `uint64x2_t` registers, then run the GHASH shift-XOR
/// reduction once with each NEON op handling both muls' lanes. Trades
/// the binius variant's 4 reduction-stage PMULLs (2 per mul × 2 muls)
/// for a vectorised XOR-based reduction. Worth it because PMULL is the
/// scarce resource on M-class (2 units, 1/cycle each).
///
/// # Safety
/// Requires the `aes` target feature (compiles to PMULL); only call where
/// `aes` is statically enabled or has been runtime-detected.
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_vec2_neon(a: [F128; 2], b: [F128; 2]) -> [F128; 2] {
    // SAFETY: function carries the aes target feature; pmull requires it.
    unsafe {
        // 8 independent schoolbook PMULLs.
        let p0_ll = pmull(a[0].lo, b[0].lo);
        let p0_lh = pmull(a[0].lo, b[0].hi);
        let p0_hl = pmull(a[0].hi, b[0].lo);
        let p0_hh = pmull(a[0].hi, b[0].hi);
        let p1_ll = pmull(a[1].lo, b[1].lo);
        let p1_lh = pmull(a[1].lo, b[1].hi);
        let p1_hl = pmull(a[1].hi, b[1].lo);
        let p1_hh = pmull(a[1].hi, b[1].hi);

        // Per-mul cross terms (lh + hl).
        let c0 = veorq_u64(p0_lh, p0_hl);
        let c1 = veorq_u64(p1_lh, p1_hl);

        // Lane-paired (mul0, mul1) layout for each word position.
        //   r0 = ll_lo
        //   r1 = ll_hi ^ cross_lo
        //   r2 = hh_lo ^ cross_hi
        //   r3 = hh_hi
        let r0 = vzip1q_u64(p0_ll, p1_ll);
        let ll_hi = vzip2q_u64(p0_ll, p1_ll);
        let c_lo = vzip1q_u64(c0, c1);
        let r1 = veorq_u64(ll_hi, c_lo);
        let hh_lo = vzip1q_u64(p0_hh, p1_hh);
        let c_hi = vzip2q_u64(c0, c1);
        let r2 = veorq_u64(hh_lo, c_hi);
        let r3 = vzip2q_u64(p0_hh, p1_hh);

        // Vectorised GHASH reduction: fold (r2, r3) into (r0, r1) mod p,
        // where p = x^128 + x^7 + x^2 + x + 1. r(x) = x^7 + x^2 + x + 1.
        // Each shift produces (lo_part, overflow); the overflow goes into
        // the next-higher word.
        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));

        let t_lo = xor3_u64(r2, s1_lo, veorq_u64(s2_lo, s7_lo));
        let t_hi = xor3_u64(r3, s1_hi, veorq_u64(s2_hi, s7_hi));

        // Bits of r3 that overflowed past position 127 in the three shifts.
        let ov = xor3_u64(
            vshrq_n_u64::<63>(r3),
            vshrq_n_u64::<62>(r3),
            vshrq_n_u64::<57>(r3),
        );
        let corr = xor3_u64(
            ov,
            vshlq_n_u64::<1>(ov),
            veorq_u64(vshlq_n_u64::<2>(ov), vshlq_n_u64::<7>(ov)),
        );

        let final_lo = xor3_u64(r0, t_lo, corr);
        let final_hi = veorq_u64(r1, t_hi);

        // Unpack: lane 0 → mul0, lane 1 → mul1.
        [
            F128 {
                lo: vgetq_lane_u64::<0>(final_lo),
                hi: vgetq_lane_u64::<0>(final_hi),
            },
            F128 {
                lo: vgetq_lane_u64::<1>(final_lo),
                hi: vgetq_lane_u64::<1>(final_hi),
            },
        ]
    }
}

/// Batch multiply two arbitrary field elements by constants whose high
/// 64-bit limbs are zero.
///
/// Each product needs only `value.lo * constant.lo` and
/// `value.hi * constant.lo`, cutting the unreduced product from four PMULLs
/// to two. The two independent products are then reduced lane-wise with NEON.
///
/// # Safety
/// Requires the `aes` target feature, and both constants must have `hi == 0`.
#[target_feature(enable = "aes")]
#[inline]
pub unsafe fn ghash_mul_low_constants_vec2_neon(
    constants: [F128; 2],
    values: [F128; 2],
) -> [F128; 2] {
    debug_assert_eq!(constants[0].hi, 0);
    debug_assert_eq!(constants[1].hi, 0);

    // SAFETY: function carries the aes target feature; pmull requires it.
    unsafe {
        let p0_ll = pmull(values[0].lo, constants[0].lo);
        let p0_hl = pmull(values[0].hi, constants[0].lo);
        let p1_ll = pmull(values[1].lo, constants[1].lo);
        let p1_hl = pmull(values[1].hi, constants[1].lo);

        // With constant.hi == 0, the unreduced words are:
        //   r0 = ll.lo
        //   r1 = ll.hi ^ hl.lo
        //   r2 = hl.hi
        //   r3 = 0
        let r0 = vzip1q_u64(p0_ll, p1_ll);
        let r1 = veorq_u64(vzip2q_u64(p0_ll, p1_ll), vzip1q_u64(p0_hl, p1_hl));
        let r2 = vzip2q_u64(p0_hl, p1_hl);

        // Fold r2*x^128 with x^128 = x^7 + x^2 + x + 1. Since r3 is zero,
        // there is no second overflow correction.
        let s1_lo = vshlq_n_u64::<1>(r2);
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s7_lo = vshlq_n_u64::<7>(r2);
        let folded_lo = xor3_u64(r2, s1_lo, veorq_u64(s2_lo, s7_lo));
        let folded_hi = xor3_u64(
            vshrq_n_u64::<63>(r2),
            vshrq_n_u64::<62>(r2),
            vshrq_n_u64::<57>(r2),
        );
        let out_lo = veorq_u64(r0, folded_lo);
        let out_hi = veorq_u64(r1, folded_hi);

        [
            F128 {
                lo: vgetq_lane_u64::<0>(out_lo),
                hi: vgetq_lane_u64::<0>(out_hi),
            },
            F128 {
                lo: vgetq_lane_u64::<1>(out_lo),
                hi: vgetq_lane_u64::<1>(out_hi),
            },
        ]
    }
}

/// Batch multiply 2× F128 by a SHARED constant `c`, Karatsuba variant of
/// [`ghash_mul_vec2_neon`]: 6 PMULL instead of 8 (3 per product), reusing
/// the same lane-paired vectorised GHASH reduction. PMULL is the scarce
/// resource on M-class (2 units, 1/cycle each), so dropping a quarter of the
/// products matters in multiplier-bound loops such as the sumcheck fold
/// (where `c` is the round challenge).
///
/// # Safety
/// Requires the `aes` target feature (compiles to PMULL); only call where
/// `aes` is statically enabled or has been runtime-detected.
#[inline]
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_const_vec2_neon(c: F128, b: [F128; 2]) -> [F128; 2] {
    // SAFETY: function carries the aes target feature; pmull requires it.
    unsafe {
        // Karatsuba per product: ll, hh, and (b_lo ^ b_hi)·(c_lo ^ c_hi);
        // cross = mm ^ ll ^ hh (char 2).
        let c_mid = c.lo ^ c.hi;
        let p0_ll = pmull(b[0].lo, c.lo);
        let p0_hh = pmull(b[0].hi, c.hi);
        let p0_mm = pmull(b[0].lo ^ b[0].hi, c_mid);
        let p1_ll = pmull(b[1].lo, c.lo);
        let p1_hh = pmull(b[1].hi, c.hi);
        let p1_mm = pmull(b[1].lo ^ b[1].hi, c_mid);

        let c0 = xor3_u64(p0_mm, p0_ll, p0_hh);
        let c1 = xor3_u64(p1_mm, p1_ll, p1_hh);

        // Lane-paired (mul0, mul1) layout for each word position, identical
        // to `ghash_mul_vec2_neon`:
        //   r0 = ll_lo
        //   r1 = ll_hi ^ cross_lo
        //   r2 = hh_lo ^ cross_hi
        //   r3 = hh_hi
        let r0 = vzip1q_u64(p0_ll, p1_ll);
        let ll_hi = vzip2q_u64(p0_ll, p1_ll);
        let c_lo = vzip1q_u64(c0, c1);
        let r1 = veorq_u64(ll_hi, c_lo);
        let hh_lo = vzip1q_u64(p0_hh, p1_hh);
        let c_hi = vzip2q_u64(c0, c1);
        let r2 = veorq_u64(hh_lo, c_hi);
        let r3 = vzip2q_u64(p0_hh, p1_hh);

        // Vectorised GHASH reduction: fold (r2, r3) into (r0, r1) mod p,
        // where p = x^128 + x^7 + x^2 + x + 1. r(x) = x^7 + x^2 + x + 1.
        // Each shift produces (lo_part, overflow); the overflow goes into
        // the next-higher word.
        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));

        let t_lo = xor3_u64(r2, s1_lo, veorq_u64(s2_lo, s7_lo));
        let t_hi = xor3_u64(r3, s1_hi, veorq_u64(s2_hi, s7_hi));

        // Bits of r3 that overflowed past position 127 in the three shifts.
        let ov = xor3_u64(
            vshrq_n_u64::<63>(r3),
            vshrq_n_u64::<62>(r3),
            vshrq_n_u64::<57>(r3),
        );
        let corr = xor3_u64(
            ov,
            vshlq_n_u64::<1>(ov),
            veorq_u64(vshlq_n_u64::<2>(ov), vshlq_n_u64::<7>(ov)),
        );

        let final_lo = xor3_u64(r0, t_lo, corr);
        let final_hi = veorq_u64(r1, t_hi);

        // Unpack: lane 0 → mul0, lane 1 → mul1.
        [
            F128 {
                lo: vgetq_lane_u64::<0>(final_lo),
                hi: vgetq_lane_u64::<0>(final_hi),
            },
            F128 {
                lo: vgetq_lane_u64::<1>(final_lo),
                hi: vgetq_lane_u64::<1>(final_hi),
            },
        ]
    }
}

/// Full 256-bit carry-less product `a · b`, no mod-p reduction. The standard
/// middle-cross fold is baked in: r1 = ll_hi ^ cross_lo, r2 = hh_lo ^ cross_hi.
///
/// # Safety
/// Requires the `aes` target feature (compiles to PMULL); only call where
/// `aes` is statically enabled or has been runtime-detected.
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_unreduced_neon(a: F128, b: F128) -> F256Unreduced {
    // SAFETY: function carries the aes target feature.
    unsafe {
        let p_ll = pmull(a.lo, b.lo);
        let p_lh = pmull(a.lo, b.hi);
        let p_hl = pmull(a.hi, b.lo);
        let p_hh = pmull(a.hi, b.hi);

        let ll_lo = vgetq_lane_u64::<0>(p_ll);
        let ll_hi = vgetq_lane_u64::<1>(p_ll);
        let hh_lo = vgetq_lane_u64::<0>(p_hh);
        let hh_hi = vgetq_lane_u64::<1>(p_hh);
        let cross = veorq_u64(p_lh, p_hl);
        let cr_lo = vgetq_lane_u64::<0>(cross);
        let cr_hi = vgetq_lane_u64::<1>(cross);

        F256Unreduced {
            r0: ll_lo,
            r1: ll_hi ^ cr_lo,
            r2: hh_lo ^ cr_hi,
            r3: hh_hi,
        }
    }
}

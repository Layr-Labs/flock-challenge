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
#[inline(always)]
#[target_feature(enable = "aes")]
unsafe fn pmull(a: u64, b: u64) -> uint64x2_t {
    let prod = vmull_p64(a, b);
    // SAFETY: u128 and uint64x2_t are both 128-bit, 16-byte-aligned values;
    // transmute is a bit-level reinterpret with no UB.
    unsafe { transmute::<u128, uint64x2_t>(prod) }
}

/// High-qword PMULL2: returns `a.hi · b.hi` as a 128-bit vector.
/// Uses the AES2 `vmull_high_p64` instruction (128-bit×128-bit→128-bit
/// with carry-less semantics on the high qword).
///
/// # Safety
/// Caller must ensure the `aes` (and `aes2` for the high variant) target
/// features are enabled. `vmull_high_p64` is part of the FEAT_PMULL
/// extension, gated by `aes` in Rust's target_feature list.
#[inline(always)]
#[target_feature(enable = "aes")]
unsafe fn pmull2(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    // vmull_high_p64 takes poly64x2_t; the bit pattern is the same as
    // uint64x2_t.
    let prod = vmull_high_p64(transmute(a), transmute(b));
    unsafe { transmute::<u128, uint64x2_t>(prod) }
}

// ---------------------------------------------------------------------------
// Reduction constant for the branchless Barrett fold. Same polynomial as
// the x86_64 path: `B = (0xC200_0000_0000_0000, 0x0000_0000_0000_0001)`.
// Stored as a `uint64x2_t` (low qword first) so the compiler can load it
// from `.rodata` and reuse it across calls.
// ---------------------------------------------------------------------------
const POLY_B: uint64x2_t =
    unsafe { transmute([0x0000_0000_0000_0001u64, 0xC200_0000_0000_0000u64]) };

/// 2-PMULL branchless Barrett reduction of `(lo, hi)` (each a 128-bit
/// vector laid out as {lo qword, hi qword}) into the 128-bit result of
/// `(lo + hi·x^128) mod g(x)`. The polynomial constant `B` is a
/// precomputed `uint64x2_t` so no `0x87`-style constant materialisation
/// is on the runtime dep chain.
#[inline(always)]
#[target_feature(enable = "aes")]
unsafe fn barrett_reduce(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
    // D = hi · B (lo qword of hi × lo qword of B = hi.lo × 1, plus
    // hi.lo × 0xC2_00... in the high half).
    let d = pmull(vgetq_lane_u64::<0>(hi), vgetq_lane_u64::<0>(POLY_B));
    // E = D.hi · B.lo (PMULL2 on the high half). This is the second
    // Barrett quotient — completes the mod-g fold without a scalar
    // shift-XOR chain.
    let e = pmull(vgetq_lane_u64::<1>(d), vgetq_lane_u64::<0>(POLY_B));

    // 7-bit overflow correction: hi half of D has up to 7 set bits
    // (because the polynomial is degree 7) and they overflow past bit
    // 127; fold them in via the canonical x^7 + x^2 + x + 1 pattern.
    let d_hi = vgetq_lane_u64::<1>(d);
    let ov = d_hi;
    let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    let corr_v = vsetq_lane_u64::<0>(corr, vdupq_n_u64(0));

    // T_lo ^ D_lo (cancels hi*poly's low 64 bits) ^ E (the second
    // Barrett quotient in the low half) ^ corr (7-bit correction).
    xor3_u64(lo, d, veorq_u64(e, corr_v))
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

/// Default aarch64 GF(2^128) multiplication: PMULL+PMULL2 schoolbook +
/// branchless Barrett reduction using a precomputed polynomial constant
/// `B = (0xC200_0000_0000_0000, 0x0000_0000_0000_0001)`.
///
/// The schoolbook portion is one `vmull_p64` (a.lo·b.lo) plus one
/// `vmull_high_p64` (a.hi·b.hi, the PMULL2 instruction) for the
/// independent products, and one `vmull_p64` plus one lane-swapped
/// `vmull_p64` for the cross terms (a.hi·b.lo + a.lo·b.hi). The Barrett
/// fold is two `vmull_p64`s against the precomputed poly constant. The
/// runtime dep chain is shorter than the binius variant because the
/// poly constant `B` is a `.rodata` load, not a runtime-extracted
/// `0x87` byte. The routine is **inlined and unrolled 2x for ILP** via
/// the companion `ghash_mul_pmull_barrett_x2` helper, so callers
/// amortise the Barrett epilogue over two products.
#[inline]
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_pmull_barrett(a: F128, b: F128) -> F128 {
    // SAFETY: function carries the aes target feature; pmull/pmull2 require it.
    unsafe {
        // 4-PMULL schoolbook:
        //   p_ll   = a.lo · b.lo          (PMULL)
        //   p_hh   = a.hi · b.hi          (PMULL2 — vmull_high_p64)
        //   p_hl_x = a.hi · b.lo          (PMULL)
        //   p_lh_x = a.lo · b.hi          (PMULL after vext-swapping b)
        // then XOR p_hl_x ^ p_lh_x for the cross term.
        let va = vsetq_lane_u64(a.lo, vdupq_n_u64(a.hi));
        let vb = vsetq_lane_u64(b.lo, vdupq_n_u64(b.hi));

        let p_ll = pmull(a.lo, b.lo);
        let p_hh = pmull2(va, vb);
        let p_hl = pmull(a.hi, b.lo);
        let vb_swap = vextq_u64::<1>(vb, vdupq_n_u64(0));
        let p_lh = pmull(a.lo, vgetq_lane_u64::<0>(vb_swap));
        let cross = veorq_u64(p_hl, p_lh);

        // Pack as 2-vector: lo-limb = (p_hh.hi, p_ll.lo), hi-limb = cross
        // mirrored into both lanes.
        let lo_limb = vsetq_lane_u64(vgetq_lane_u64::<1>(p_hh), p_ll);
        let cross_swapped = vextq_u64::<1>(cross, cross);
        let hi_limb = veorq_u64(cross, cross_swapped);

        let reduced = barrett_reduce(lo_limb, hi_limb);
        F128 {
            lo: vgetq_lane_u64::<0>(reduced),
            hi: vgetq_lane_u64::<1>(reduced),
        }
    }
}

/// 2x-unrolled batch multiplication for ILP: returns
/// `[a0*b0, a1*b1]` with the second product's PMULLs overlapping the
/// first's Barrett epilogue. Exposed publicly so hot callers (e.g. the
/// field-slice butterfly path) can amortise the Barrett fold over a
/// pair of products.
#[inline]
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_pmull_barrett_x2(
    a0: F128,
    b0: F128,
    a1: F128,
    b1: F128,
) -> [F128; 2] {
    // SAFETY: function carries the aes target feature.
    unsafe {
        // Issue both products' PMULLs back-to-back before either Barrett,
        // so the PMULL latency overlaps the Barrett epilogue.
        let va0 = vsetq_lane_u64(a0.lo, vdupq_n_u64(a0.hi));
        let vb0 = vsetq_lane_u64(b0.lo, vdupq_n_u64(b0.hi));
        let p_ll0 = pmull(a0.lo, b0.lo);
        let p_hh0 = pmull2(va0, vb0);
        let p_hl0 = pmull(a0.hi, b0.lo);
        let vb0_swap = vextq_u64::<1>(vb0, vdupq_n_u64(0));
        let p_lh0 = pmull(a0.lo, vgetq_lane_u64::<0>(vb0_swap));
        let cross0 = veorq_u64(p_hl0, p_lh0);

        let va1 = vsetq_lane_u64(a1.lo, vdupq_n_u64(a1.hi));
        let vb1 = vsetq_lane_u64(b1.lo, vdupq_n_u64(b1.hi));
        let p_ll1 = pmull(a1.lo, b1.lo);
        let p_hh1 = pmull2(va1, vb1);
        let p_hl1 = pmull(a1.hi, b1.lo);
        let vb1_swap = vextq_u64::<1>(vb1, vdupq_n_u64(0));
        let p_lh1 = pmull(a1.lo, vgetq_lane_u64::<0>(vb1_swap));
        let cross1 = veorq_u64(p_hl1, p_lh1);

        let lo_limb0 = vsetq_lane_u64(vgetq_lane_u64::<1>(p_hh0), p_ll0);
        let cross0_swapped = vextq_u64::<1>(cross0, cross0);
        let hi_limb0 = veorq_u64(cross0, cross0_swapped);
        let lo_limb1 = vsetq_lane_u64(vgetq_lane_u64::<1>(p_hh1), p_ll1);
        let cross1_swapped = vextq_u64::<1>(cross1, cross1);
        let hi_limb1 = veorq_u64(cross1, cross1_swapped);

        let r0 = barrett_reduce(lo_limb0, hi_limb0);
        let r1 = barrett_reduce(lo_limb1, hi_limb1);
        [
            F128 {
                lo: vgetq_lane_u64::<0>(r0),
                hi: vgetq_lane_u64::<1>(r0),
            },
            F128 {
                lo: vgetq_lane_u64::<0>(r1),
                hi: vgetq_lane_u64::<1>(r1),
            },
        ]
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

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
        let c87 = vdupq_n_u64(0x87);

        let t0 = pmull(a.lo, b.lo);
        let t1a = pmull(a.lo, b.hi);
        let t1b = pmull(a.hi, b.lo);
        let t2 = pmull(a.hi, b.hi);

        // Put t0.high in the middle low lane; its high lane stays unchanged.
        let s2 = xor3_u64(t1a, t1b, vextq_u64::<1>(t0, t2));
        let t1: uint64x2_t;
        core::arch::asm!(
            "pmull2 {o:v}.1q, {m:v}.2d, {c:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {s:v}.16b",
            m = in(vreg) t2,
            c = in(vreg) c87,
            s = in(vreg) s2,
            o = out(vreg) t1,
            options(pure, nomem, nostack, preserves_flags),
        );

        // The corrected low lane lets ZIP form the final base without zero.
        let s1 = vzip1q_u64(t0, t1);
        let out: uint64x2_t;
        core::arch::asm!(
            "pmull2 {o:v}.1q, {m:v}.2d, {c:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {s:v}.16b",
            m = in(vreg) t1,
            c = in(vreg) c87,
            s = in(vreg) s1,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );

        F128 {
            lo: vgetq_lane_u64::<0>(out),
            hi: vgetq_lane_u64::<1>(out),
        }
    }
}

/// Batch multiply 2× F128 in parallel via Karatsuba 6-PMULL.
///
/// Strategy: 6 Karatsuba PMULLs (3 per product: ll, hh, and (a_lo^a_hi)·(b_lo^b_hi)),
/// cross-terms c = mm ^ ll ^ hh via `xor3_u64`, repack the four unreduced 64-bit
/// words `(r0, r1, r2, r3)` of each product into lane-paired `uint64x2_t` registers,
/// then run the GHASH shift-XOR reduction once with each NEON op handling both
/// muls' lanes. Drops PMULL count from 8 to 6, conserving the scarce PMULL units
/// on M-class (2 units, 1/cycle each).
///
/// # Safety
/// Requires the `aes` target feature (compiles to PMULL); only call where
/// `aes` is statically enabled or has been runtime-detected.
#[target_feature(enable = "aes")]
pub unsafe fn ghash_mul_vec2_neon(a: [F128; 2], b: [F128; 2]) -> [F128; 2] {
    // SAFETY: function carries the aes target feature; pmull requires it.
    unsafe {
        // 6 Karatsuba PMULLs (3 per product).
        let p0_ll = pmull(a[0].lo, b[0].lo);
        let p0_hh = pmull(a[0].hi, b[0].hi);
        let p0_mm = pmull(a[0].lo ^ a[0].hi, b[0].lo ^ b[0].hi);
        let p1_ll = pmull(a[1].lo, b[1].lo);
        let p1_hh = pmull(a[1].hi, b[1].hi);
        let p1_mm = pmull(a[1].lo ^ a[1].hi, b[1].lo ^ b[1].hi);

        // Per-mul cross terms mm ^ ll ^ hh.
        let c0 = xor3_u64(p0_mm, p0_ll, p0_hh);
        let c1 = xor3_u64(p1_mm, p1_ll, p1_hh);
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
        let r2 = veorq_u64(
            r2,
            xor3_u64(
                vshrq_n_u64::<63>(r3),
                vshrq_n_u64::<62>(r3),
                vshrq_n_u64::<57>(r3),
            ),
        );
        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));

        let t_lo = xor3_u64(r2, s1_lo, veorq_u64(s2_lo, s7_lo));
        let t_hi = xor3_u64(r3, s1_hi, veorq_u64(s2_hi, s7_hi));

        let final_lo = veorq_u64(r0, t_lo);
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

// ---------------------------------------------------------------------------
// Fixed-multiplier ("prepared") products for the sumcheck folds.
//
// The sumcheck folds multiply a whole slice by one challenge `t`. For
// a fixed `t`, precompute `R = t.hi * 0x87` (so `x^128 * t.hi = R` modulo
// `x^128 + x^7 + x^2 + x + 1`) and pack two operands:
//
//   p.0 = (t.lo, R.lo)          p.1 = (t.hi, t.lo ^ R.hi)
//
// Then `v * t = L + x^64 * M` with `L = v.lo*t.lo + v.hi*R.lo` and
// `M = v.lo*t.hi + v.hi*(t.lo + R.hi)`: four data PMULLs, and one more PMULL
// folds `x^64 * M`. The value never leaves the vector register file and no
// Karatsuba middle terms or 256-bit shift reduction are formed. Both `L` and
// `M` are GF(2)-linear in `v`, so a sum of products by different constants
// shares the single fold. Same field products, canonical (fully reduced)
// outputs: bit-identical to the Karatsuba kernels they replace.
// ---------------------------------------------------------------------------

pub(crate) type ConstPrep = (uint64x2_t, uint64x2_t);

#[inline(always)]
pub(crate) unsafe fn const_prep(t: uint64x2_t) -> ConstPrep {
    unsafe {
        let r = transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(t),
            vreinterpretq_p64_u64(vdupq_n_u64(0x87)),
        ));
        let lo = vzip1q_u64(t, r);
        let mid = veorq_u64(vextq_u64::<1>(t, t), vzip2q_u64(vdupq_n_u64(0), r));
        (lo, mid)
    }
}

/// `(x.lo * y.lo) ^ (x.hi * y.hi)`.
#[inline(always)]
unsafe fn pmull_pair_sum(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    let out: uint64x2_t;
    unsafe {
        core::arch::asm!(
            "pmull {p:v}.1q, {x:v}.1d, {y:v}.1d",
            "pmull2 {o:v}.1q, {x:v}.2d, {y:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {p:v}.16b",
            x = in(vreg) x,
            y = in(vreg) y,
            p = out(vreg) _,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// `acc ^ (x.lo * y.lo) ^ (x.hi * y.hi)`, each PMULL adjacent to its EOR.
#[inline(always)]
unsafe fn pmull_pair_acc(acc: uint64x2_t, x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    let out: uint64x2_t;
    unsafe {
        core::arch::asm!(
            "pmull {p:v}.1q, {x:v}.1d, {y:v}.1d",
            "eor {p:v}.16b, {p:v}.16b, {c:v}.16b",
            "pmull2 {o:v}.1q, {x:v}.2d, {y:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {p:v}.16b",
            x = in(vreg) x,
            y = in(vreg) y,
            c = in(vreg) acc,
            p = out(vreg) _,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// Reduce `L + x^64 * M`: `M.lo` moves to the high limb and
/// `M.hi * x^128 = M.hi * 0x87` (at most 71 bits) lands in the low limbs.
#[inline(always)]
unsafe fn fold_lm(l: uint64x2_t, m: uint64x2_t) -> uint64x2_t {
    unsafe {
        let s = veorq_u64(l, vextq_u64::<1>(vdupq_n_u64(0), m));
        let out: uint64x2_t;
        core::arch::asm!(
            "pmull2 {o:v}.1q, {m:v}.2d, {k:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {s:v}.16b",
            m = in(vreg) m,
            k = in(vreg) vdupq_n_u64(0x87),
            s = in(vreg) s,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
        out
    }
}

/// `x * t` for the prepared constant `p = const_prep(t)`, fully reduced.
#[inline(always)]
pub(crate) unsafe fn mul_const_prep(p: ConstPrep, x: uint64x2_t) -> uint64x2_t {
    unsafe { fold_lm(pmull_pair_sum(x, p.0), pmull_pair_sum(x, p.1)) }
}

/// `x * c + y * d` for two prepared constants, with one shared fold.
#[inline(always)]
pub(crate) unsafe fn mul_two_const_sum_prep(
    pc: ConstPrep,
    x: uint64x2_t,
    pd: ConstPrep,
    y: uint64x2_t,
) -> uint64x2_t {
    unsafe {
        let l = pmull_pair_acc(pmull_pair_sum(x, pc.0), y, pd.0);
        let m = pmull_pair_acc(pmull_pair_sum(x, pc.1), y, pd.1);
        fold_lm(l, m)
    }
}

/// Prepared form of the shared-constant product pair;
/// `FLOCK_NO_CONST_VEC2_PREP=1` restores the Karatsuba kernel.
#[inline]
fn use_const_vec2_prep() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_CONST_VEC2_PREP").is_none());
    *ON
}

/// [`ghash_mul_const_vec2_neon`] on the prepared fixed-multiplier product:
/// one PMULL prepares `c`, then each product is four data PMULLs and one
/// fold, with no lane-paired shift reduction. When the call is inlined into a
/// loop over a slice the preparation is loop-invariant.
#[inline(always)]
unsafe fn ghash_mul_const_vec2_prepared(c: F128, b: [F128; 2]) -> [F128; 2] {
    // SAFETY: `F128` is `repr(C, align(16))` over `{ lo, hi }`, the
    // little-endian lane order of `uint64x2_t`; callers supply `aes`.
    unsafe {
        let p = const_prep(transmute::<F128, uint64x2_t>(c));
        [
            transmute::<uint64x2_t, F128>(mul_const_prep(p, transmute::<F128, uint64x2_t>(b[0]))),
            transmute::<uint64x2_t, F128>(mul_const_prep(p, transmute::<F128, uint64x2_t>(b[1]))),
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
    // SAFETY: the caller supplies `aes`.
    unsafe {
        if use_const_vec2_prep() {
            ghash_mul_const_vec2_prepared(c, b)
        } else {
            ghash_mul_const_vec2_karatsuba(c, b)
        }
    }
}

#[cfg(test)]
pub(crate) unsafe fn ghash_mul_const_vec2_prepared_for_test(c: F128, b: [F128; 2]) -> [F128; 2] {
    unsafe { ghash_mul_const_vec2_prepared(c, b) }
}

#[cfg(test)]
pub(crate) unsafe fn ghash_mul_const_vec2_karatsuba_for_test(c: F128, b: [F128; 2]) -> [F128; 2] {
    unsafe { ghash_mul_const_vec2_karatsuba(c, b) }
}

/// The Karatsuba form of [`ghash_mul_const_vec2_neon`].
#[inline]
#[target_feature(enable = "aes")]
unsafe fn ghash_mul_const_vec2_karatsuba(c: F128, b: [F128; 2]) -> [F128; 2] {
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

        let r2 = veorq_u64(
            r2,
            xor3_u64(
                vshrq_n_u64::<63>(r3),
                vshrq_n_u64::<62>(r3),
                vshrq_n_u64::<57>(r3),
            ),
        );
        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));

        let t_lo = xor3_u64(r2, s1_lo, veorq_u64(s2_lo, s7_lo));
        let t_hi = xor3_u64(r3, s1_hi, veorq_u64(s2_hi, s7_hi));

        let final_lo = veorq_u64(r0, t_lo);
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

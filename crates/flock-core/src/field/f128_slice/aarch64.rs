use crate::field::F128;
use crate::field::gf2_128::aarch64::{
    const_prep, mul_const_prep, mul_const_prep_xor, mul_four_const_sum_prep_xor,
    mul_three_const_sum_prep_xor, mul_two_const_sum_prep_xor,
};
#[cfg(test)]
use crate::field::gf2_128::aarch64::mul_two_const_sum_prep;
use core::arch::aarch64::*;
use core::mem::transmute;

#[derive(Clone, Copy)]
struct WideNeon {
    lo: uint64x2_t,
    hi: uint64x2_t,
}

/// Unreduced schoolbook product accumulator `(ll, cross, hh)` updated with
/// Apple Silicon macro-fused `pmull+eor` / `pmull2+eor` pairs. Reconstructing
/// `WideNeon` is deferred once per scan instead of per product.
#[derive(Clone, Copy)]
struct TriNeon {
    ll: uint64x2_t,
    cross: uint64x2_t,
    hh: uint64x2_t,
}

impl TriNeon {
    #[inline(always)]
    unsafe fn zero() -> Self {
        let z = unsafe { vdupq_n_u64(0) };
        Self {
            ll: z,
            cross: z,
            hh: z,
        }
    }
}

/// Unreduced Karatsuba product accumulator `(ll, hh, mm)` updated with
/// Apple Silicon macro-fused `pmull+eor` / `pmull2+eor` pairs. Used in
/// shared-weight multi-slot/multi-lane kernels where unzipping two adjacent
/// weights `(w0, w1)` into `(w_lo, w_hi, w_mid)` amortizes across slots and
/// cuts PMULL issue pressure by 25% (3 vs 4 PMULLs/product) while keeping
/// accumulator chains at most 2-deep.
#[derive(Clone, Copy)]
struct KaraNeon {
    ll: uint64x2_t,
    hh: uint64x2_t,
    mm: uint64x2_t,
}

impl KaraNeon {
    #[inline(always)]
    unsafe fn zero() -> Self {
        let z = unsafe { vdupq_n_u64(0) };
        Self {
            ll: z,
            hh: z,
            mm: z,
        }
    }
}

#[inline(always)]
unsafe fn tri_to_wide(value: TriNeon) -> WideNeon {
    unsafe {
        let zero = vdupq_n_u64(0);
        WideNeon {
            lo: veorq_u64(value.ll, vextq_u64::<1>(zero, value.cross)),
            hi: veorq_u64(value.hh, vextq_u64::<1>(value.cross, zero)),
        }
    }
}

#[inline(always)]
unsafe fn kara_to_wide(value: KaraNeon) -> WideNeon {
    unsafe {
        let zero = vdupq_n_u64(0);
        let cross = xor3_u64(value.mm, value.ll, value.hh);
        WideNeon {
            lo: veorq_u64(value.ll, vextq_u64::<1>(zero, cross)),
            hi: veorq_u64(value.hh, vextq_u64::<1>(cross, zero)),
        }
    }
}

#[inline(always)]
unsafe fn fused_pmull_lo_acc(acc: uint64x2_t, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let out: uint64x2_t;
    // SAFETY: register-only carry-less multiply and XOR; caller supplies `aes`.
    unsafe {
        core::arch::asm!(
            "pmull {o:v}.1q, {a:v}.1d, {b:v}.1d",
            "eor {o:v}.16b, {o:v}.16b, {c:v}.16b",
            a = in(vreg) a,
            b = in(vreg) b,
            c = in(vreg) acc,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

#[inline(always)]
unsafe fn fused_pmull_hi_acc(acc: uint64x2_t, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let out: uint64x2_t;
    // SAFETY: register-only carry-less multiply and XOR; caller supplies `aes`.
    unsafe {
        core::arch::asm!(
            "pmull2 {o:v}.1q, {a:v}.2d, {b:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {c:v}.16b",
            a = in(vreg) a,
            b = in(vreg) b,
            c = in(vreg) acc,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

#[inline(always)]
unsafe fn fused_cross_acc(acc: uint64x2_t, a_sw: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let mut out = acc;
    // SAFETY: register-only carry-less multiplies and XORs; caller supplies `aes`.
    unsafe {
        core::arch::asm!(
            "pmull {p:v}.1q, {t:v}.1d, {b:v}.1d",
            "eor {p:v}.16b, {p:v}.16b, {o:v}.16b",
            "pmull2 {o:v}.1q, {t:v}.2d, {b:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {p:v}.16b",
            t = in(vreg) a_sw,
            b = in(vreg) b,
            p = out(vreg) _,
            o = inout(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// `acc += x0 * w0 + x1 * w1` in [`KaraNeon`] form using pre-unzipped weight
/// lanes `(w_lo, w_hi, w_mid)`: six macro-fused `pmull+eor` / `pmull2+eor`
/// pairs (three PMULLs per product) with 2-deep per-accumulator chains.
#[inline(always)]
unsafe fn kara_acc_two_banks(
    acc: &mut KaraNeon,
    x0: uint64x2_t,
    x1: uint64x2_t,
    w_lo: uint64x2_t,
    w_hi: uint64x2_t,
    w_mid: uint64x2_t,
) {
    unsafe {
        let x_lo = vuzp1q_u64(x0, x1);
        let x_hi = vuzp2q_u64(x0, x1);
        let x_mid = veorq_u64(x_lo, x_hi);
        acc.ll = fused_cross_acc(acc.ll, x_lo, w_lo);
        acc.hh = fused_cross_acc(acc.hh, x_hi, w_hi);
        acc.mm = fused_cross_acc(acc.mm, x_mid, w_mid);
    }
}

/// `acc += x * w` in [`KaraNeon`] form for a single tail bank, given `w` and
/// `w_mid = w ^ swap_halves(w)`.
#[inline(always)]
unsafe fn kara_acc_single(
    acc: &mut KaraNeon,
    x: uint64x2_t,
    w: uint64x2_t,
    w_mid: uint64x2_t,
) {
    unsafe {
        let x_mid = veorq_u64(x, vextq_u64::<1>(x, x));
        acc.ll = fused_pmull_lo_acc(acc.ll, x, w);
        acc.hh = fused_pmull_hi_acc(acc.hh, x, w);
        acc.mm = fused_pmull_lo_acc(acc.mm, x_mid, w_mid);
    }
}

/// `acc += a * b` in [`TriNeon`] form using pre-swapped `a_sw = swap_halves(a)`:
/// four macro-fused `pmull+eor` / `pmull2+eor` SIMD issue slots.
#[inline(always)]
unsafe fn tri_acc_sw(acc: &mut TriNeon, a: uint64x2_t, a_sw: uint64x2_t, b: uint64x2_t) {
    unsafe {
        acc.ll = fused_pmull_lo_acc(acc.ll, a, b);
        acc.hh = fused_pmull_hi_acc(acc.hh, a, b);
        acc.cross = fused_cross_acc(acc.cross, a_sw, b);
    }
}

/// `acc += a * b` in [`TriNeon`] form: one `ext` plus four macro-fused
/// `pmull+eor` / `pmull2+eor` SIMD issue slots (five slots total).
#[inline(always)]
unsafe fn tri_acc_q(acc: &mut TriNeon, a: uint64x2_t, b: uint64x2_t) {
    unsafe {
        let a_sw = swap_halves(a);
        tri_acc_sw(acc, a, a_sw, b);
    }
}

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

#[inline]
#[target_feature(enable = "aes")]
unsafe fn pmull(a: u64, b: u64) -> uint64x2_t {
    // SAFETY: the caller provides the `aes` target feature; both types are
    // 128-bit bit containers with compatible alignment.
    unsafe { transmute::<u128, uint64x2_t>(vmull_p64(a, b)) }
}

/// Prepared fixed-multiplier products in the fused sumcheck folds;
/// `FLOCK_NO_FOLD_CONST_PREP=1` restores the Karatsuba constant kernels.
#[inline]
fn use_fold_const_prep() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_FOLD_CONST_PREP").is_none());
    *ON
}

/// Multiply two independent values by the same constant and reduce them in
/// lane-paired form. Constant-r Karatsuba needs six PMULLs total rather than
/// the generic two-product schoolbook kernel's eight.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_const_vec2(r: uint64x2_t, x0: uint64x2_t, x1: uint64x2_t) -> [uint64x2_t; 2] {
    unsafe {
        let r_lo = vgetq_lane_u64::<0>(r);
        let r_hi = vgetq_lane_u64::<1>(r);
        let r_mid = veorq_u64(r, vextq_u64::<1>(r, r));
        let x0_mid = veorq_u64(x0, vextq_u64::<1>(x0, x0));
        let x1_mid = veorq_u64(x1, vextq_u64::<1>(x1, x1));

        let p0_ll = pmull(vgetq_lane_u64::<0>(x0), r_lo);
        let p0_hh = pmull(vgetq_lane_u64::<1>(x0), r_hi);
        let p0_mm = pmull(vgetq_lane_u64::<0>(x0_mid), vgetq_lane_u64::<0>(r_mid));
        let p1_ll = pmull(vgetq_lane_u64::<0>(x1), r_lo);
        let p1_hh = pmull(vgetq_lane_u64::<1>(x1), r_hi);
        let p1_mm = pmull(vgetq_lane_u64::<0>(x1_mid), vgetq_lane_u64::<0>(r_mid));
        let c0 = xor3_u64(p0_mm, p0_ll, p0_hh);
        let c1 = xor3_u64(p1_mm, p1_ll, p1_hh);

        // Pack product 0/1 into lanes and reduce both together.
        let r0 = vzip1q_u64(p0_ll, p1_ll);
        let r1 = veorq_u64(vzip2q_u64(p0_ll, p1_ll), vzip1q_u64(c0, c1));
        let r2 = veorq_u64(vzip1q_u64(p0_hh, p1_hh), vzip2q_u64(c0, c1));
        let r3 = vzip2q_u64(p0_hh, p1_hh);

        let r2 = veorq_u64(r2, r3_overflow(r3));
        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));
        let t_lo = xor3_u64(r2, s1_lo, veorq_u64(s2_lo, s7_lo));
        let t_hi = xor3_u64(r3, s1_hi, veorq_u64(s2_hi, s7_hi));
        let out_lo = veorq_u64(r0, t_lo);
        let out_hi = veorq_u64(r1, t_hi);
        [vzip1q_u64(out_lo, out_hi), vzip2q_u64(out_lo, out_hi)]
    }
}

/// Compute two independent sums of constant products and reduce each sum
/// once: `[c*x0 + d*y0, c*x1 + d*y1]`.
///
/// The two Karatsuba products contributing to each output remain in the
/// 256-bit product domain until after their `ll`, `hh`, and `mm` components
/// have been XORed. Packing the two resulting product sums into NEON lanes
/// then shares one vectorized reduction across both outputs. Compared with
/// two calls to [`mul_const_vec2`], this preserves the twelve-PMULL count but
/// removes one complete paired reduction.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_two_const_sum_vec2(
    c: uint64x2_t,
    x0: uint64x2_t,
    x1: uint64x2_t,
    d: uint64x2_t,
    y0: uint64x2_t,
    y1: uint64x2_t,
) -> [uint64x2_t; 2] {
    unsafe {
        let c_lo = vgetq_lane_u64::<0>(c);
        let c_hi = vgetq_lane_u64::<1>(c);
        let c_mid = c_lo ^ c_hi;
        let d_lo = vgetq_lane_u64::<0>(d);
        let d_hi = vgetq_lane_u64::<1>(d);
        let d_mid = d_lo ^ d_hi;

        let x0_lo = vgetq_lane_u64::<0>(x0);
        let x0_hi = vgetq_lane_u64::<1>(x0);
        let y0_lo = vgetq_lane_u64::<0>(y0);
        let y0_hi = vgetq_lane_u64::<1>(y0);
        let p0_ll = veorq_u64(pmull(x0_lo, c_lo), pmull(y0_lo, d_lo));
        let p0_hh = veorq_u64(pmull(x0_hi, c_hi), pmull(y0_hi, d_hi));
        let p0_mm = veorq_u64(pmull(x0_lo ^ x0_hi, c_mid), pmull(y0_lo ^ y0_hi, d_mid));

        let x1_lo = vgetq_lane_u64::<0>(x1);
        let x1_hi = vgetq_lane_u64::<1>(x1);
        let y1_lo = vgetq_lane_u64::<0>(y1);
        let y1_hi = vgetq_lane_u64::<1>(y1);
        let p1_ll = veorq_u64(pmull(x1_lo, c_lo), pmull(y1_lo, d_lo));
        let p1_hh = veorq_u64(pmull(x1_hi, c_hi), pmull(y1_hi, d_hi));
        let p1_mm = veorq_u64(pmull(x1_lo ^ x1_hi, c_mid), pmull(y1_lo ^ y1_hi, d_mid));

        let cross0 = xor3_u64(p0_mm, p0_ll, p0_hh);
        let cross1 = xor3_u64(p1_mm, p1_ll, p1_hh);

        // Pack output 0/1 into lanes and reduce both product sums together.
        let r0 = vzip1q_u64(p0_ll, p1_ll);
        let r1 = veorq_u64(vzip2q_u64(p0_ll, p1_ll), vzip1q_u64(cross0, cross1));
        let r2 = veorq_u64(vzip1q_u64(p0_hh, p1_hh), vzip2q_u64(cross0, cross1));
        let r3 = vzip2q_u64(p0_hh, p1_hh);

        let r2 = veorq_u64(r2, r3_overflow(r3));
        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));
        let t_lo = xor3_u64(r2, s1_lo, veorq_u64(s2_lo, s7_lo));
        let t_hi = xor3_u64(r3, s1_hi, veorq_u64(s2_hi, s7_hi));
        let out_lo = veorq_u64(r0, t_lo);
        let out_hi = veorq_u64(r1, t_hi);
        [vzip1q_u64(out_lo, out_hi), vzip2q_u64(out_lo, out_hi)]
    }
}

#[inline(always)]
unsafe fn xor_wide(acc: &mut WideNeon, value: WideNeon) {
    unsafe {
        acc.lo = veorq_u64(acc.lo, value.lo);
        acc.hi = veorq_u64(acc.hi, value.hi);
    }
}

#[inline(always)]
unsafe fn r3_overflow(r3: uint64x2_t) -> uint64x2_t {
    unsafe {
        xor3_u64(
            vshrq_n_u64::<63>(r3),
            vshrq_n_u64::<62>(r3),
            vshrq_n_u64::<57>(r3),
        )
    }
}

#[inline(always)]
unsafe fn reduce_wide(value: WideNeon) -> uint64x2_t {
    unsafe {
        let zero = vdupq_n_u64(0);
        let high = veorq_u64(value.hi, r3_overflow(vextq_u64::<1>(value.hi, zero)));
        let shift1 = veorq_u64(
            vshlq_n_u64::<1>(high),
            vextq_u64::<1>(zero, vshrq_n_u64::<63>(high)),
        );
        let shift2 = veorq_u64(
            vshlq_n_u64::<2>(high),
            vextq_u64::<1>(zero, vshrq_n_u64::<62>(high)),
        );
        let shift7 = veorq_u64(
            vshlq_n_u64::<7>(high),
            vextq_u64::<1>(zero, vshrq_n_u64::<57>(high)),
        );
        let folded = xor3_u64(high, shift1, veorq_u64(shift2, shift7));
        veorq_u64(value.lo, folded)
    }
}

#[inline(always)]
unsafe fn reduce_wide_pair(first: WideNeon, second: WideNeon) -> [uint64x2_t; 2] {
    unsafe {
        let r0 = vzip1q_u64(first.lo, second.lo);
        let r1 = vzip2q_u64(first.lo, second.lo);
        let r2 = vzip1q_u64(first.hi, second.hi);
        let r3 = vzip2q_u64(first.hi, second.hi);

        let r2 = veorq_u64(r2, r3_overflow(r3));
        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));
        let t_lo = xor3_u64(r2, s1_lo, veorq_u64(s2_lo, s7_lo));
        let t_hi = xor3_u64(r3, s1_hi, veorq_u64(s2_hi, s7_hi));
        let out_lo = veorq_u64(r0, t_lo);
        let out_hi = veorq_u64(r1, t_hi);
        [vzip1q_u64(out_lo, out_hi), vzip2q_u64(out_lo, out_hi)]
    }
}

/// Accumulate the ranked opening's round-zero message and round-one
/// lookahead without reducing every product individually.
///
/// The scalar expression has eight independent product sums per four input
/// slots. Carry-less multiplication is linear over XOR, so each sum may stay
/// in the 256-bit product domain for the full slice and be reduced once at
/// the end. This preserves the exact field result while removing the
/// per-product reduction/shuttle from the hot scan.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `witness` and `basis` must have
/// equal lengths divisible by four.
#[target_feature(enable = "aes")]
pub(super) unsafe fn round0_and_round1_lookahead(
    witness: &[F128],
    basis: &[F128],
) -> ((F128, F128), [F128; 6]) {
    unsafe {
        debug_assert_eq!(witness.len(), basis.len());
        debug_assert!(witness.len().is_multiple_of(4));

        let mut a_c0 = TriNeon::zero();
        let mut a_c1_endpoint = TriNeon::zero();
        let mut a_c2 = TriNeon::zero();
        let mut a_u0_second = TriNeon::zero();
        let mut a_u2_second = TriNeon::zero();
        let mut a_c3 = TriNeon::zero();
        let mut a_c4_endpoint = TriNeon::zero();
        let mut a_c5 = TriNeon::zero();

        let mut i = 0usize;
        while i < witness.len() {
            let a0 = vld1q_u64(witness.as_ptr().add(i).cast::<u64>());
            let a1 = vld1q_u64(witness.as_ptr().add(i + 1).cast::<u64>());
            let a2 = vld1q_u64(witness.as_ptr().add(i + 2).cast::<u64>());
            let a3 = vld1q_u64(witness.as_ptr().add(i + 3).cast::<u64>());
            let b0 = vld1q_u64(basis.as_ptr().add(i).cast::<u64>());
            let b1 = vld1q_u64(basis.as_ptr().add(i + 1).cast::<u64>());
            let b2 = vld1q_u64(basis.as_ptr().add(i + 2).cast::<u64>());
            let b3 = vld1q_u64(basis.as_ptr().add(i + 3).cast::<u64>());

            let sa0 = veorq_u64(a0, a1);
            let sb0 = veorq_u64(b0, b1);
            let sa1 = veorq_u64(a2, a3);
            let sb1 = veorq_u64(b2, b3);
            tri_acc_q(&mut a_c0, a0, b0);
            tri_acc_q(&mut a_c1_endpoint, a1, b1);
            tri_acc_q(&mut a_c2, sa0, sb0);
            tri_acc_q(&mut a_u0_second, a2, b2);
            tri_acc_q(&mut a_u2_second, sa1, sb1);

            let even_a = veorq_u64(a0, a2);
            let even_b = veorq_u64(b0, b2);
            let odd_a = veorq_u64(a1, a3);
            let odd_b = veorq_u64(b1, b3);
            let sum_a = veorq_u64(even_a, odd_a);
            let sum_b = veorq_u64(even_b, odd_b);
            tri_acc_q(&mut a_c3, even_a, even_b);
            tri_acc_q(&mut a_c4_endpoint, odd_a, odd_b);
            tri_acc_q(&mut a_c5, sum_a, sum_b);
            i += 4;
        }

        let red = |value: WideNeon| transmute::<uint64x2_t, F128>(reduce_wide(value));
        let c0 = red(tri_to_wide(a_c0));
        let c2 = red(tri_to_wide(a_c2));
        let c3 = red(tri_to_wide(a_c3));
        let c5 = red(tri_to_wide(a_c5));
        let c1 = red(tri_to_wide(a_c1_endpoint)) + c0 + c2;
        let c4 = red(tri_to_wide(a_c4_endpoint)) + c3 + c5;
        let u0 = c0 + red(tri_to_wide(a_u0_second));
        let u2 = c2 + red(tri_to_wide(a_u2_second));
        ((u0, u2), [c0, c1, c2, c3, c4, c5])
    }
}

/// Banked deferred-reduction fold of a single output slot:
/// `Σ_{k<BANKS} weight[k] · input[k]`.
///
/// Carry-less multiplication is F2-linear and reduction mod p is F2-linear,
/// so the whole bank sum may stay in the 256-bit product domain and be
/// reduced exactly once instead of once per bank. Each product accumulates via
/// macro-fused `pmull+eor` / `pmull2+eor` into four independent [`TriNeon`]
/// streams so XOR chains stay off the critical path.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `input` must hold at least
/// `BANKS` elements.
#[inline]
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_banked_slot<const BANKS: usize>(
    weight: &[F128; BANKS],
    input: &[F128],
) -> F128 {
    unsafe {
        debug_assert!(input.len() >= BANKS);
        let mut a0 = TriNeon::zero();
        let mut a1 = TriNeon::zero();
        let mut a2 = TriNeon::zero();
        let mut a3 = TriNeon::zero();

        let w = weight.as_ptr();
        let x = input.as_ptr();
        let main = BANKS & !3;
        let mut k = 0usize;
        while k < main {
            let (w0, w1) = load_pair_whole(w.add(k));
            let (w2, w3) = load_pair_whole(w.add(k + 2));
            let (x0, x1) = load_pair_whole(x.add(k));
            let (x2, x3) = load_pair_whole(x.add(k + 2));
            tri_acc_q(&mut a0, w0, x0);
            tri_acc_q(&mut a1, w1, x1);
            tri_acc_q(&mut a2, w2, x2);
            tri_acc_q(&mut a3, w3, x3);
            k += 4;
        }
        while k < BANKS {
            let wk = vld1q_u64(w.add(k).cast::<u64>());
            let xk = vld1q_u64(x.add(k).cast::<u64>());
            tri_acc_q(&mut a0, wk, xk);
            k += 1;
        }

        let mut wide = tri_to_wide(a0);
        xor_wide(&mut wide, tri_to_wide(a1));
        xor_wide(&mut wide, tri_to_wide(a2));
        xor_wide(&mut wide, tri_to_wide(a3));
        transmute::<uint64x2_t, F128>(reduce_wide(wide))
    }
}

/// Banked deferred-reduction folds of two adjacent output slots. Two
/// consecutive weights are unzipped into `(w_lo, w_hi, w_mid)` once and shared
/// across both slots, accumulating into four independent macro-fused
/// [`KaraNeon`] streams (three PMULLs/product, 1-deep chains) with one paired
/// reduction.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `input` must hold at least
/// `2 * BANKS` elements, with the second slot starting at `input[BANKS]`.
#[inline]
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_banked_slots2<const BANKS: usize>(
    weight: &[F128; BANKS],
    input: &[F128],
) -> [F128; 2] {
    unsafe {
        debug_assert!(input.len() >= 2 * BANKS);
        let mut first0 = KaraNeon::zero();
        let mut first1 = KaraNeon::zero();
        let mut second0 = KaraNeon::zero();
        let mut second1 = KaraNeon::zero();

        let w = weight.as_ptr();
        let x0 = input.as_ptr();
        let x1 = input.as_ptr().add(BANKS);
        let main = BANKS & !1;
        let mut bank = 0usize;
        while bank < main {
            let (w0, w1) = load_pair_whole(w.add(bank));
            let w_lo = vuzp1q_u64(w0, w1);
            let w_hi = vuzp2q_u64(w0, w1);
            let w_mid = veorq_u64(w_lo, w_hi);
            let (x00, x01) = load_pair_whole(x0.add(bank));
            let e_lo = vuzp1q_u64(x00, x01);
            let e_hi = vuzp2q_u64(x00, x01);
            let e_mid = veorq_u64(e_lo, e_hi);
            first0.ll = fused_pmull_lo_acc(first0.ll, e_lo, w_lo);
            first1.ll = fused_pmull_hi_acc(first1.ll, e_lo, w_lo);
            first0.hh = fused_pmull_lo_acc(first0.hh, e_hi, w_hi);
            first1.hh = fused_pmull_hi_acc(first1.hh, e_hi, w_hi);
            first0.mm = fused_pmull_lo_acc(first0.mm, e_mid, w_mid);
            first1.mm = fused_pmull_hi_acc(first1.mm, e_mid, w_mid);
            let (x10, x11) = load_pair_whole(x1.add(bank));
            let o_lo = vuzp1q_u64(x10, x11);
            let o_hi = vuzp2q_u64(x10, x11);
            let o_mid = veorq_u64(o_lo, o_hi);
            second0.ll = fused_pmull_lo_acc(second0.ll, o_lo, w_lo);
            second1.ll = fused_pmull_hi_acc(second1.ll, o_lo, w_lo);
            second0.hh = fused_pmull_lo_acc(second0.hh, o_hi, w_hi);
            second1.hh = fused_pmull_hi_acc(second1.hh, o_hi, w_hi);
            second0.mm = fused_pmull_lo_acc(second0.mm, o_mid, w_mid);
            second1.mm = fused_pmull_hi_acc(second1.mm, o_mid, w_mid);
            bank += 2;
        }
        if bank < BANKS {
            let wk = load_whole(w.add(bank));
            let wk_mid = veorq_u64(wk, vextq_u64::<1>(wk, wk));
            kara_acc_single(&mut first0, load_whole(x0.add(bank)), wk, wk_mid);
            kara_acc_single(&mut second0, load_whole(x1.add(bank)), wk, wk_mid);
        }

        first0.ll = veorq_u64(first0.ll, first1.ll);
        first0.hh = veorq_u64(first0.hh, first1.hh);
        first0.mm = veorq_u64(first0.mm, first1.mm);
        second0.ll = veorq_u64(second0.ll, second1.ll);
        second0.hh = veorq_u64(second0.hh, second1.hh);
        second0.mm = veorq_u64(second0.mm, second1.mm);
        let reduced = reduce_wide_pair(kara_to_wide(first0), kara_to_wide(second0));
        [
            transmute::<uint64x2_t, F128>(reduced[0]),
            transmute::<uint64x2_t, F128>(reduced[1]),
        ]
    }
}

///
/// # Safety
/// Requires the `aes` target feature (PMULL). `input` must hold at least
#[inline]
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_banked_slots4<const BANKS: usize>(
    weight: &[F128; BANKS],
    input: &[F128],
) -> [F128; 4] {
    unsafe {
        debug_assert!(input.len() >= 4 * BANKS);
        if BANKS % 2 == 0 {
            return fold_banked_slots4_schoolbook::<BANKS>(weight, input);
        }
        let mut acc = [TriNeon::zero(); 4];

        let w = weight.as_ptr();
        let x = input.as_ptr();
        let mut bank = 0usize;
        while bank < BANKS {
            let wv = vld1q_u64(w.add(bank).cast::<u64>());
            let wv_sw = swap_halves(wv);
            for slot in 0..4 {
                let xv = vld1q_u64(x.add(slot * BANKS + bank).cast::<u64>());
                tri_acc_sw(&mut acc[slot], wv, wv_sw, xv);
            }
            bank += 1;
        }

        let low = reduce_wide_pair(tri_to_wide(acc[0]), tri_to_wide(acc[1]));
        let high = reduce_wide_pair(tri_to_wide(acc[2]), tri_to_wide(acc[3]));
        [
            transmute::<uint64x2_t, F128>(low[0]),
            transmute::<uint64x2_t, F128>(low[1]),
            transmute::<uint64x2_t, F128>(high[0]),
            transmute::<uint64x2_t, F128>(high[1]),
        ]
    }
}

/// `[v.hi, v.lo]` as one `EXT`, kept opaque so LLVM cannot turn the
/// matching-lane products below back into lane `DUP`s.
#[inline(always)]
unsafe fn swap_halves(v: uint64x2_t) -> uint64x2_t {
    let out: uint64x2_t;
    // SAFETY: register-only lane rotate; no memory, stack or flags.
    unsafe {
        core::arch::asm!(
            "ext {out:v}.16b, {v:v}.16b, {v:v}.16b, #8",
            v = in(vreg) v,
            out = lateout(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// [`fold_banked_slots4`] for an even bank count: two consecutive weights are
/// unzipped into `(w_lo, w_hi, w_mid)` once and shared across all four slots,
/// accumulating via macro-fused [`KaraNeon`] (3 PMULLs/product, 12 live
/// accumulator registers, 2-deep chains).
#[inline(always)]
unsafe fn fold_banked_slots4_schoolbook<const BANKS: usize>(
    weight: &[F128; BANKS],
    input: &[F128],
) -> [F128; 4] {
    unsafe {
        let mut acc = [KaraNeon::zero(); 4];

        let w = weight.as_ptr();
        let x = input.as_ptr();
        let mut bank = 0usize;
        while bank < BANKS {
            let (w0, w1) = load_pair_whole(w.add(bank));
            let w_lo = vuzp1q_u64(w0, w1);
            let w_hi = vuzp2q_u64(w0, w1);
            let w_mid = veorq_u64(w_lo, w_hi);
            for slot in 0..4 {
                let (x0, x1) = load_pair_whole(x.add(slot * BANKS + bank));
                kara_acc_two_banks(&mut acc[slot], x0, x1, w_lo, w_hi, w_mid);
            }
            bank += 2;
        }

        let low = reduce_wide_pair(kara_to_wide(acc[0]), kara_to_wide(acc[1]));
        let high = reduce_wide_pair(kara_to_wide(acc[2]), kara_to_wide(acc[3]));
        [
            transmute::<uint64x2_t, F128>(low[0]),
            transmute::<uint64x2_t, F128>(low[1]),
            transmute::<uint64x2_t, F128>(high[0]),
            transmute::<uint64x2_t, F128>(high[1]),
        ]
    }
}

/// One 32-byte pair load (`ldp q, q`) that the optimizer may not narrow or
/// split into 8-byte lane loads / `ld1r` broadcasts while still allowing
/// cross-iteration scheduling (`pure, readonly`).
#[inline(always)]
unsafe fn load_pair_whole(p: *const F128) -> (uint64x2_t, uint64x2_t) {
    let x0: uint64x2_t;
    let x1: uint64x2_t;
    // SAFETY (caller): `p` is valid for a 32-byte read of two adjacent `F128`s.
    unsafe {
        core::arch::asm!(
            "ldp {x0:q}, {x1:q}, [{p}]",
            p = in(reg) p,
            x0 = out(vreg) x0,
            x1 = out(vreg) x1,
            options(pure, readonly, nostack, preserves_flags),
        );
    }
    (x0, x1)
}

/// One 32-byte pair store (`stp q, q`) matching [`load_pair_whole`].
#[inline(always)]
unsafe fn store_pair_whole(p: *mut F128, x0: uint64x2_t, x1: uint64x2_t) {
    // SAFETY (caller): `p` is valid for a 32-byte write of two adjacent `F128`s.
    unsafe {
        core::arch::asm!(
            "stp {x0:q}, {x1:q}, [{p}]",
            p = in(reg) p,
            x0 = in(vreg) x0,
            x1 = in(vreg) x1,
            options(nostack, preserves_flags),
        );
    }
}

/// One whole 16-byte load of an input element (`ldr q`) that the optimizer may
/// not narrow or split.
#[inline(always)]
unsafe fn load_whole(p: *const F128) -> uint64x2_t {
    let x: uint64x2_t;
    // SAFETY (caller): `p` is valid for a 16-byte read.
    unsafe {
        core::arch::asm!(
            "ldr {x:q}, [{p}]",
            p = in(reg) p,
            x = out(vreg) x,
            options(pure, readonly, nostack, preserves_flags),
        );
    }
    x
}

/// [`fold_banked_slots4`] for an even bank count with whole-pair loads
/// ([`load_pair_whole`]) and macro-fused `pmull+eor` accumulation, writing the
/// four folded slots to `out[0..4]`.
///
/// `odd_live` is the number of leading banks that can be nonzero in the
/// second and fourth slot (`BANKS` when nothing is known). Banks at and above
/// it, rounded up to a bank pair, are neither loaded nor multiplied for those
/// two slots: the caller promises they are zero, and a zero input adds
/// nothing to any of the three product sums.
///
/// `odd_lo_tail` (used only with an odd `odd_live < BANKS`) adds the promise
/// that bank `odd_live - 1` of those two slots has a zero high limb. Its
/// pair partner is then zero too, so of that pair's eight limb products
/// only `x.lo * w.lo` and `x.lo * w.hi` can be nonzero: just those two run.
///
/// Same products (less the promised-zero ones), same accumulators and the
/// same reduction as [`fold_banked_slots4_schoolbook`]: the four outputs are
/// bit-identical whenever the promise holds.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `BANKS` must be even, `input`
/// must hold at least `4 * BANKS` elements and `out` at least four.
#[inline]
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_banked_slots4_whole_into<const BANKS: usize>(
    weight: &[F128; BANKS],
    input: &[F128],
    out: &mut [F128],
    odd_live: usize,
    odd_lo_tail: bool,
) {
    unsafe {
        debug_assert!(BANKS % 2 == 0);
        debug_assert!(input.len() >= 4 * BANKS);
        debug_assert!(out.len() >= 4);
        let mut acc = [KaraNeon::zero(); 4];

        let w = weight.as_ptr();
        let x = input.as_ptr();
        let mut lo_tail = odd_lo_tail && odd_live < BANKS && odd_live % 2 == 1;
        let split = if odd_live >= BANKS {
            BANKS
        } else if lo_tail {
            odd_live - 1
        } else {
            (odd_live + 1) & !1
        };
        let mut bank = 0usize;
        while bank < split {
            let (w0, w1) = load_pair_whole(w.add(bank));
            let w_lo = vuzp1q_u64(w0, w1);
            let w_hi = vuzp2q_u64(w0, w1);
            let w_mid = veorq_u64(w_lo, w_hi);
            for slot in 0..4 {
                let (x0, x1) = load_pair_whole(x.add(slot * BANKS + bank));
                kara_acc_two_banks(&mut acc[slot], x0, x1, w_lo, w_hi, w_mid);
            }
            bank += 2;
        }
        // Banks whose second- and fourth-slot inputs are promised zero.
        while bank < BANKS {
            let (w0, w1) = load_pair_whole(w.add(bank));
            let w_lo = vuzp1q_u64(w0, w1);
            let w_hi = vuzp2q_u64(w0, w1);
            let w_mid = veorq_u64(w_lo, w_hi);
            for slot in [0usize, 2] {
                let (x0, x1) = load_pair_whole(x.add(slot * BANKS + bank));
                kara_acc_two_banks(&mut acc[slot], x0, x1, w_lo, w_hi, w_mid);
            }
            if lo_tail {
                // Bank `bank` of the second and fourth slot: low limb only
                // (`x_tail.hi == 0`), and bank `bank + 1` is zero.
                // `ll ^= x.lo * w0.lo`, `mm ^= x.lo * (w0.lo ^ w0.hi)` so
                // `cross = mm ^ ll ^ hh` gains `x.lo * w0.hi`.
                for slot in [1usize, 3] {
                    let x_tail = load_whole(x.add(slot * BANKS + bank));
                    acc[slot].ll = fused_pmull_lo_acc(acc[slot].ll, x_tail, w_lo);
                    acc[slot].mm = fused_pmull_lo_acc(acc[slot].mm, x_tail, w_mid);
                }
                lo_tail = false;
            }
            bank += 2;
        }

        let low = reduce_wide_pair(kara_to_wide(acc[0]), kara_to_wide(acc[1]));
        let high = reduce_wide_pair(kara_to_wide(acc[2]), kara_to_wide(acc[3]));
        let o = out.as_mut_ptr().cast::<u64>();
        vst1q_u64(o, low[0]);
        vst1q_u64(o.add(2), low[1]);
        vst1q_u64(o.add(4), high[0]);
        vst1q_u64(o.add(6), high[1]);
    }
}

/// Fixed 128-bank by 64-lane bank-major fold for the ranked zerocheck AB
/// completion. Two consecutive weights are unzipped once and shared across
/// four adjacent lanes with macro-fused [`KaraNeon`] accumulators.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `input` must contain exactly
/// 128 consecutive rows of 64 [`F128`] values.
#[inline(never)]
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_bank_major_128x64_rows2(
    weight: &[F128; 128],
    input: &[F128],
    output: &mut [F128; 64],
) {
    unsafe {
        debug_assert_eq!(input.len(), 128 * 64);
        let wp = weight.as_ptr();
        let xp = input.as_ptr();
        let out = output.as_mut_ptr();

        let mut lane = 0usize;
        while lane < 64 {
            let mut acc = [KaraNeon::zero(); 4];
            let mut bank = 0usize;
            let mut row = xp.add(lane);
            while bank < 128 {
                let (w0, w1) = load_pair_whole(wp.add(bank));
                let w_lo = vuzp1q_u64(w0, w1);
                let w_hi = vuzp2q_u64(w0, w1);
                let w_mid = veorq_u64(w_lo, w_hi);
                let (r0_0, r0_1) = load_pair_whole(row);
                let (r0_2, r0_3) = load_pair_whole(row.add(2));
                let (r1_0, r1_1) = load_pair_whole(row.add(64));
                let (r1_2, r1_3) = load_pair_whole(row.add(66));
                kara_acc_two_banks(&mut acc[0], r0_0, r1_0, w_lo, w_hi, w_mid);
                kara_acc_two_banks(&mut acc[1], r0_1, r1_1, w_lo, w_hi, w_mid);
                kara_acc_two_banks(&mut acc[2], r0_2, r1_2, w_lo, w_hi, w_mid);
                kara_acc_two_banks(&mut acc[3], r0_3, r1_3, w_lo, w_hi, w_mid);
                bank += 2;
                row = row.add(128);
            }

            let lo = reduce_wide_pair(kara_to_wide(acc[0]), kara_to_wide(acc[1]));
            let hi = reduce_wide_pair(kara_to_wide(acc[2]), kara_to_wide(acc[3]));
            vst1q_u64(out.add(lane).cast::<u64>(), lo[0]);
            vst1q_u64(out.add(lane + 1).cast::<u64>(), lo[1]);
            vst1q_u64(out.add(lane + 2).cast::<u64>(), hi[0]);
            vst1q_u64(out.add(lane + 3).cast::<u64>(), hi[1]);
            lane += 4;
        }
    }
}

/// Fixed 64-bank by 128-lane bank-major collapse for ring-switch Fold8.
/// Two consecutive weights are unzipped once and shared across four adjacent
/// lanes with macro-fused [`KaraNeon`] accumulators.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `input` must contain exactly
/// 64 consecutive rows of 128 [`F128`] values.
#[inline(never)]
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_bank_major_64x128_rows2(
    weight: &[F128; 64],
    input: &[F128],
    output: &mut [F128; 128],
) {
    unsafe {
        debug_assert_eq!(input.len(), 64 * 128);
        let wp = weight.as_ptr();
        let xp = input.as_ptr();
        let out = output.as_mut_ptr();

        let mut lane = 0usize;
        while lane < 128 {
            let mut acc = [KaraNeon::zero(); 4];
            let mut bank = 0usize;
            let mut row = xp.add(lane);
            while bank < 64 {
                let (w0, w1) = load_pair_whole(wp.add(bank));
                let w_lo = vuzp1q_u64(w0, w1);
                let w_hi = vuzp2q_u64(w0, w1);
                let w_mid = veorq_u64(w_lo, w_hi);
                let (r0_0, r0_1) = load_pair_whole(row);
                let (r0_2, r0_3) = load_pair_whole(row.add(2));
                let (r1_0, r1_1) = load_pair_whole(row.add(128));
                let (r1_2, r1_3) = load_pair_whole(row.add(130));
                kara_acc_two_banks(&mut acc[0], r0_0, r1_0, w_lo, w_hi, w_mid);
                kara_acc_two_banks(&mut acc[1], r0_1, r1_1, w_lo, w_hi, w_mid);
                kara_acc_two_banks(&mut acc[2], r0_2, r1_2, w_lo, w_hi, w_mid);
                kara_acc_two_banks(&mut acc[3], r0_3, r1_3, w_lo, w_hi, w_mid);
                bank += 2;
                row = row.add(256);
            }

            let lo = reduce_wide_pair(kara_to_wide(acc[0]), kara_to_wide(acc[1]));
            let hi = reduce_wide_pair(kara_to_wide(acc[2]), kara_to_wide(acc[3]));
            vst1q_u64(out.add(lane).cast::<u64>(), lo[0]);
            vst1q_u64(out.add(lane + 1).cast::<u64>(), lo[1]);
            vst1q_u64(out.add(lane + 2).cast::<u64>(), hi[0]);
            vst1q_u64(out.add(lane + 3).cast::<u64>(), hi[1]);
            lane += 4;
        }
    }
}

/// Deferred-reduction round-zero message `(u_0, u_2)` over paired slots.
///
/// Bitwise-identical to the scalar pair loop: both accumulate
/// `u0 = Σ a[2i]·b[2i]` and `u2 = Σ (a[2i]+a[2i+1])·(b[2i]+b[2i+1])`, and
/// reduction commutes with the XOR sum.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `witness` and `basis` must have
/// equal lengths divisible by two.
#[target_feature(enable = "aes")]
pub(super) unsafe fn round0(witness: &[F128], basis: &[F128]) -> (F128, F128) {
    unsafe {
        debug_assert_eq!(witness.len(), basis.len());
        debug_assert!(witness.len().is_multiple_of(2));

        let mut u0_a = TriNeon::zero();
        let mut u0_b = TriNeon::zero();
        let mut u2_a = TriNeon::zero();
        let mut u2_b = TriNeon::zero();

        let n = witness.len();
        let main = n & !3;
        let wp = witness.as_ptr();
        let bp = basis.as_ptr();
        let mut i = 0usize;
        while i < main {
            let (a0, a1) = load_pair_whole(wp.add(i));
            let (a2, a3) = load_pair_whole(wp.add(i + 2));
            let (b0, b1) = load_pair_whole(bp.add(i));
            let (b2, b3) = load_pair_whole(bp.add(i + 2));
            tri_acc_q(&mut u0_a, a0, b0);
            tri_acc_q(&mut u0_b, a2, b2);
            tri_acc_q(&mut u2_a, veorq_u64(a0, a1), veorq_u64(b0, b1));
            tri_acc_q(&mut u2_b, veorq_u64(a2, a3), veorq_u64(b2, b3));
            i += 4;
        }
        while i < n {
            let (a0, a1) = load_pair_whole(wp.add(i));
            let (b0, b1) = load_pair_whole(bp.add(i));
            tri_acc_q(&mut u0_a, a0, b0);
            tri_acc_q(&mut u2_a, veorq_u64(a0, a1), veorq_u64(b0, b1));
            i += 2;
        }

        for (left, right) in [(&mut u0_a, u0_b), (&mut u2_a, u2_b)] {
            left.ll = veorq_u64(left.ll, right.ll);
            left.hh = veorq_u64(left.hh, right.hh);
            left.cross = veorq_u64(left.cross, right.cross);
        }
        let reduced = reduce_wide_pair(tri_to_wide(u0_a), tri_to_wide(u2_a));
        (
            transmute::<uint64x2_t, F128>(reduced[0]),
            transmute::<uint64x2_t, F128>(reduced[1]),
        )
    }
}

/// [`round0`] fused with the final additive basis update: for every slot,
/// `basis[i] += delta(i)` is stored and the updated value feeds four
/// macro-fused [`TriNeon`] streams, so the completed basis is never re-read.
/// `delta` is called once per slot in increasing slot order.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `witness` and `basis` must have
/// equal lengths divisible by two. With `COPY`, `dst` must be writable for
/// `witness.len()` elements and disjoint from both slices.
#[target_feature(enable = "aes")]
pub(super) unsafe fn round0_add_basis<D: FnMut(usize) -> F128, const COPY: bool>(
    witness: &[F128],
    basis: &mut [F128],
    mut delta: D,
    dst: *mut F128,
) -> (F128, F128) {
    unsafe {
        debug_assert_eq!(witness.len(), basis.len());
        debug_assert!(witness.len().is_multiple_of(2));

        let mut u0_a = TriNeon::zero();
        let mut u0_b = TriNeon::zero();
        let mut u2_a = TriNeon::zero();
        let mut u2_b = TriNeon::zero();

        let n = witness.len();
        let main = n & !3;
        let wp = witness.as_ptr();
        let bp = basis.as_mut_ptr();
        // SAFETY (closure): `j < n = basis.len()` at every call site below.
        let mut update = |j: usize| -> uint64x2_t {
            let v = *bp.add(j) + delta(j);
            *bp.add(j) = v;
            transmute::<F128, uint64x2_t>(v)
        };
        let mut i = 0usize;
        while i < main {
            let (a0, a1) = load_pair_whole(wp.add(i));
            let (a2, a3) = load_pair_whole(wp.add(i + 2));
            if COPY {
                store_pair_whole(dst.add(i), a0, a1);
                store_pair_whole(dst.add(i + 2), a2, a3);
            }
            let b0 = update(i);
            let b1 = update(i + 1);
            let b2 = update(i + 2);
            let b3 = update(i + 3);
            tri_acc_q(&mut u0_a, a0, b0);
            tri_acc_q(&mut u0_b, a2, b2);
            tri_acc_q(&mut u2_a, veorq_u64(a0, a1), veorq_u64(b0, b1));
            tri_acc_q(&mut u2_b, veorq_u64(a2, a3), veorq_u64(b2, b3));
            i += 4;
        }
        while i < n {
            let (a0, a1) = load_pair_whole(wp.add(i));
            if COPY {
                store_pair_whole(dst.add(i), a0, a1);
            }
            let b0 = update(i);
            let b1 = update(i + 1);
            tri_acc_q(&mut u0_a, a0, b0);
            tri_acc_q(&mut u2_a, veorq_u64(a0, a1), veorq_u64(b0, b1));
            i += 2;
        }

        for (left, right) in [(&mut u0_a, u0_b), (&mut u2_a, u2_b)] {
            left.ll = veorq_u64(left.ll, right.ll);
            left.hh = veorq_u64(left.hh, right.hh);
            left.cross = veorq_u64(left.cross, right.cross);
        }
        let reduced = reduce_wide_pair(tri_to_wide(u0_a), tri_to_wide(u2_a));
        (
            transmute::<uint64x2_t, F128>(reduced[0]),
            transmute::<uint64x2_t, F128>(reduced[1]),
        )
    }
}

/// Deferred-reduction sufficient statistics for a factorized LSB equality
/// basis. Two consecutive tail weights are unzipped into `(w_lo, w_hi, w_mid)`
/// once and shared across even/odd witness pairs into four independent
/// macro-fused [`KaraNeon`] streams (3 PMULLs/product, 1-deep chains); after
/// one paired reduction, `s = a + odd` recovers the pair-sum statistic.
///
/// # Safety
/// Requires the `aes` target feature. `f.len()` must equal
/// `2 * eq_tail.len()`.
#[target_feature(enable = "aes")]
pub(super) unsafe fn round0_factorized_eq(f: &[F128], eq_tail: &[F128]) -> (F128, F128) {
    unsafe {
        debug_assert_eq!(f.len(), 2 * eq_tail.len());

        let mut a0 = KaraNeon::zero();
        let mut a1 = KaraNeon::zero();
        let mut o0 = KaraNeon::zero();
        let mut o1 = KaraNeon::zero();

        let fp = f.as_ptr();
        let wp = eq_tail.as_ptr();
        let main = eq_tail.len() & !1;
        let mut j = 0usize;
        while j < main {
            let (f_00, f_01) = load_pair_whole(fp.add(2 * j));
            let (f_10, f_11) = load_pair_whole(fp.add(2 * j + 2));
            let (w_0, w_1) = load_pair_whole(wp.add(j));
            let w_lo = vuzp1q_u64(w_0, w_1);
            let w_hi = vuzp2q_u64(w_0, w_1);
            let w_mid = veorq_u64(w_lo, w_hi);
            let e_lo = vuzp1q_u64(f_00, f_10);
            let e_hi = vuzp2q_u64(f_00, f_10);
            let e_mid = veorq_u64(e_lo, e_hi);
            a0.ll = fused_pmull_lo_acc(a0.ll, e_lo, w_lo);
            a1.ll = fused_pmull_hi_acc(a1.ll, e_lo, w_lo);
            a0.hh = fused_pmull_lo_acc(a0.hh, e_hi, w_hi);
            a1.hh = fused_pmull_hi_acc(a1.hh, e_hi, w_hi);
            a0.mm = fused_pmull_lo_acc(a0.mm, e_mid, w_mid);
            a1.mm = fused_pmull_hi_acc(a1.mm, e_mid, w_mid);
            let o_lo = vuzp1q_u64(f_01, f_11);
            let o_hi = vuzp2q_u64(f_01, f_11);
            let o_mid = veorq_u64(o_lo, o_hi);
            o0.ll = fused_pmull_lo_acc(o0.ll, o_lo, w_lo);
            o1.ll = fused_pmull_hi_acc(o1.ll, o_lo, w_lo);
            o0.hh = fused_pmull_lo_acc(o0.hh, o_hi, w_hi);
            o1.hh = fused_pmull_hi_acc(o1.hh, o_hi, w_hi);
            o0.mm = fused_pmull_lo_acc(o0.mm, o_mid, w_mid);
            o1.mm = fused_pmull_hi_acc(o1.mm, o_mid, w_mid);
            j += 2;
        }
        if j < eq_tail.len() {
            let (f_0, f_1) = load_pair_whole(fp.add(2 * j));
            let w = load_whole(wp.add(j));
            let w_mid = veorq_u64(w, vextq_u64::<1>(w, w));
            kara_acc_single(&mut a0, f_0, w, w_mid);
            kara_acc_single(&mut o0, f_1, w, w_mid);
        }

        a0.ll = veorq_u64(a0.ll, a1.ll);
        a0.hh = veorq_u64(a0.hh, a1.hh);
        a0.mm = veorq_u64(a0.mm, a1.mm);
        o0.ll = veorq_u64(o0.ll, o1.ll);
        o0.hh = veorq_u64(o0.hh, o1.hh);
        o0.mm = veorq_u64(o0.mm, o1.mm);
        let reduced = reduce_wide_pair(kara_to_wide(a0), kara_to_wide(o0));
        let s_reduced = veorq_u64(reduced[0], reduced[1]);
        (transmute(reduced[0]), transmute(s_reduced))
    }
}

/// Expand one equality-table level with a shared multiplier `r`.
///
/// For every old value `v = lo[i]`, writes `hi[i] = v * r` and
/// `lo[i] = v + hi[i]`. When prepared constants are enabled, `const_prep(r)`
/// is hoisted once per level and four elements are processed per iteration
/// using whole-pair loads/stores purely in NEON vector registers.
///
/// # Safety
/// Requires the `aes` target feature. `lo` and `hi` must have equal lengths.
#[inline]
#[target_feature(enable = "aes")]
pub(super) unsafe fn expand_eq_table_level(lo: &mut [F128], hi: &mut [F128], r: F128) {
    use crate::field::gf2_128::aarch64::{ghash_mul_const_vec2_neon, use_const_vec2_prep};

    debug_assert_eq!(lo.len(), hi.len());
    let n = lo.len();
    if use_const_vec2_prep() {
        unsafe {
            let p = const_prep(transmute::<F128, uint64x2_t>(r));
            let lp = lo.as_mut_ptr();
            let hp = hi.as_mut_ptr();
            let main = n & !3;
            let mut i = 0usize;
            while i < main {
                let (v0, v1) = load_pair_whole(lp.add(i));
                let (v2, v3) = load_pair_whole(lp.add(i + 2));
                let h0 = mul_const_prep(p, v0);
                let h1 = mul_const_prep(p, v1);
                let h2 = mul_const_prep(p, v2);
                let h3 = mul_const_prep(p, v3);
                store_pair_whole(hp.add(i), h0, h1);
                store_pair_whole(hp.add(i + 2), h2, h3);
                store_pair_whole(lp.add(i), veorq_u64(v0, h0), veorq_u64(v1, h1));
                store_pair_whole(lp.add(i + 2), veorq_u64(v2, h2), veorq_u64(v3, h3));
                i += 4;
            }
            while i < n {
                let v = load_whole(lp.add(i));
                let h = mul_const_prep(p, v);
                vst1q_u64(hp.add(i).cast::<u64>(), h);
                vst1q_u64(lp.add(i).cast::<u64>(), veorq_u64(v, h));
                i += 1;
            }
        }
        return;
    }

    let paired = n & !1;
    let mut i = 0usize;
    while i < paired {
        let values = [lo[i], lo[i + 1]];
        // SAFETY: the architecture-selecting caller supplies `aes`.
        let products = unsafe { ghash_mul_const_vec2_neon(r, values) };
        hi[i] = products[0];
        hi[i + 1] = products[1];
        lo[i] = values[0] + products[0];
        lo[i + 1] = values[1] + products[1];
        i += 2;
    }

    if i < n {
        let value = lo[i];
        let product = value * r;
        hi[i] = product;
        lo[i] = value + product;
    }
}

/// Two-lane pair fold using NEON and PMULL.
///
/// # Safety
/// Requires the `aes` target feature.
#[allow(dead_code)]
pub(super) unsafe fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_const_vec2_neon;

    let lanes = dst.len() & !1;
    let mut t = 0;
    while t < lanes {
        let s = 2 * (base + t);
        let e0 = src[s];
        let o0 = src[s + 1];
        let e1 = src[s + 2];
        let o1 = src[s + 3];
        let x0 = F128 {
            lo: e0.lo ^ o0.lo,
            hi: e0.hi ^ o0.hi,
        };
        let x1 = F128 {
            lo: e1.lo ^ o1.lo,
            hi: e1.hi ^ o1.hi,
        };
        // Constant-multiplier Karatsuba: 6 PMULL for the two products
        // instead of 8 schoolbook (PMULL is the scarce resource).
        // SAFETY: caller guarantees the aes target feature.
        let prod = unsafe { ghash_mul_const_vec2_neon(r, [x0, x1]) };
        dst[t] = F128 {
            lo: e0.lo ^ prod[0].lo,
            hi: e0.hi ^ prod[0].hi,
        };
        dst[t + 1] = F128 {
            lo: e1.lo ^ prod[1].lo,
            hi: e1.hi ^ prod[1].hi,
        };
        t += 2;
    }

    let one_plus_r = F128::ONE + r;
    while t < dst.len() {
        let s = 2 * (base + t);
        dst[t] = src[s] * one_plus_r + src[s + 1] * r;
        t += 1;
    }
}

/// Fold two polynomials and accumulate the next sumcheck message in one pass.
/// Each message product stays unreduced until the end of the chunk.
///
/// # Safety
/// Requires the `aes` target feature. Bounds and even destination length are
/// checked by the architecture-selecting wrapper.
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_two_and_msg(
    f: &[F128],
    b: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
) -> (F128, F128) {
    // SAFETY: same contract as the caller; the two arms differ only in how
    // the fixed-multiplier products are formed.
    unsafe {
        if use_fold_const_prep() {
            fold_two_and_msg_impl::<true>(f, b, base, nf, nb, r)
        } else {
            fold_two_and_msg_impl::<false>(f, b, base, nf, nb, r)
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fold_two_and_msg_impl<const PREP: bool>(
    f: &[F128],
    b: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
) -> (F128, F128) {
    unsafe {
        let r_q = transmute::<F128, uint64x2_t>(r);
        let r_prep = const_prep(r_q);
        let mut u0 = TriNeon::zero();
        let mut u2 = TriNeon::zero();
        let mut t = 0;
        while t < nf.len() {
            let source = 2 * (base + t);
            let (f_even0, f_odd0) = load_pair_whole(f.as_ptr().add(source));
            let (f_even1, f_odd1) = load_pair_whole(f.as_ptr().add(source + 2));
            let (b_even0, b_odd0) = load_pair_whole(b.as_ptr().add(source));
            let (b_even1, b_odd1) = load_pair_whole(b.as_ptr().add(source + 2));

            let (f0, f1) = if PREP {
                (
                    mul_const_prep_xor(r_prep, veorq_u64(f_even0, f_odd0), f_even0),
                    mul_const_prep_xor(r_prep, veorq_u64(f_even1, f_odd1), f_even1),
                )
            } else {
                let folded_f = mul_const_vec2_pmred(
                    r_q,
                    veorq_u64(f_even0, f_odd0),
                    veorq_u64(f_even1, f_odd1),
                );
                (
                    veorq_u64(f_even0, folded_f[0]),
                    veorq_u64(f_even1, folded_f[1]),
                )
            };
            let (b0, b1) = if PREP {
                (
                    mul_const_prep_xor(r_prep, veorq_u64(b_even0, b_odd0), b_even0),
                    mul_const_prep_xor(r_prep, veorq_u64(b_even1, b_odd1), b_even1),
                )
            } else {
                let folded_b = mul_const_vec2_pmred(
                    r_q,
                    veorq_u64(b_even0, b_odd0),
                    veorq_u64(b_even1, b_odd1),
                );
                (
                    veorq_u64(b_even0, folded_b[0]),
                    veorq_u64(b_even1, folded_b[1]),
                )
            };

            vst1q_u64(nf.as_mut_ptr().add(t).cast::<u64>(), f0);
            vst1q_u64(nf.as_mut_ptr().add(t + 1).cast::<u64>(), f1);
            vst1q_u64(nb.as_mut_ptr().add(t).cast::<u64>(), b0);
            vst1q_u64(nb.as_mut_ptr().add(t + 1).cast::<u64>(), b1);

            tri_acc_q(&mut u0, f0, b0);
            tri_acc_q(&mut u2, veorq_u64(f0, f1), veorq_u64(b0, b1));
            t += 2;
        }
        let reduced = reduce_wide_pair(tri_to_wide(u0), tri_to_wide(u2));
        (
            transmute::<uint64x2_t, F128>(reduced[0]),
            transmute::<uint64x2_t, F128>(reduced[1]),
        )
    }
}

/// In-place DirectFold8 counterpart of [`fold_two_and_msg`]. The arithmetic
/// and reduction order intentionally mirror that kernel exactly.
///
/// # Safety
/// Requires the `aes` target feature and equal input lengths divisible by
/// four. Each iteration loads source slots `2t..2t+4` from both states before
/// writing output slots `t..t+2`. All earlier writes end below every later
/// source read, so folding into the lower half cannot clobber unread input.
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_two_and_msg_in_place(
    f: &mut [F128],
    b: &mut [F128],
    r: F128,
) -> (F128, F128) {
    // SAFETY: same contract as the caller; the two arms differ only in how
    // the fixed-multiplier products are formed.
    unsafe {
        if use_fold_const_prep() {
            fold_two_and_msg_in_place_impl::<true>(f, b, r)
        } else {
            fold_two_and_msg_in_place_impl::<false>(f, b, r)
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fold_two_and_msg_in_place_impl<const PREP: bool>(
    f: &mut [F128],
    b: &mut [F128],
    r: F128,
) -> (F128, F128) {
    unsafe {
        let f_ptr = f.as_mut_ptr();
        let b_ptr = b.as_mut_ptr();
        let half = f.len() / 2;
        let r_q = transmute::<F128, uint64x2_t>(r);
        let r_prep = const_prep(r_q);
        let mut u0 = TriNeon::zero();
        let mut u2 = TriNeon::zero();
        let mut t = 0;
        while t < half {
            let source = 2 * t;
            let (f_even0, f_odd0) = load_pair_whole(f_ptr.add(source));
            let (f_even1, f_odd1) = load_pair_whole(f_ptr.add(source + 2));
            let (b_even0, b_odd0) = load_pair_whole(b_ptr.add(source));
            let (b_even1, b_odd1) = load_pair_whole(b_ptr.add(source + 2));

            let (f0, f1) = if PREP {
                (
                    mul_const_prep_xor(r_prep, veorq_u64(f_even0, f_odd0), f_even0),
                    mul_const_prep_xor(r_prep, veorq_u64(f_even1, f_odd1), f_even1),
                )
            } else {
                let folded_f = mul_const_vec2_pmred(
                    r_q,
                    veorq_u64(f_even0, f_odd0),
                    veorq_u64(f_even1, f_odd1),
                );
                (
                    veorq_u64(f_even0, folded_f[0]),
                    veorq_u64(f_even1, folded_f[1]),
                )
            };
            let (b0, b1) = if PREP {
                (
                    mul_const_prep_xor(r_prep, veorq_u64(b_even0, b_odd0), b_even0),
                    mul_const_prep_xor(r_prep, veorq_u64(b_even1, b_odd1), b_even1),
                )
            } else {
                let folded_b = mul_const_vec2_pmred(
                    r_q,
                    veorq_u64(b_even0, b_odd0),
                    veorq_u64(b_even1, b_odd1),
                );
                (
                    veorq_u64(b_even0, folded_b[0]),
                    veorq_u64(b_even1, folded_b[1]),
                )
            };

            vst1q_u64(f_ptr.add(t).cast::<u64>(), f0);
            vst1q_u64(f_ptr.add(t + 1).cast::<u64>(), f1);
            vst1q_u64(b_ptr.add(t).cast::<u64>(), b0);
            vst1q_u64(b_ptr.add(t + 1).cast::<u64>(), b1);

            tri_acc_q(&mut u0, f0, b0);
            tri_acc_q(&mut u2, veorq_u64(f0, f1), veorq_u64(b0, b1));
            t += 2;
        }
        let reduced = reduce_wide_pair(tri_to_wide(u0), tri_to_wide(u2));
        (
            transmute::<uint64x2_t, F128>(reduced[0]),
            transmute::<uint64x2_t, F128>(reduced[1]),
        )
    }
}

/// Fold two polynomials, inject a scaled folded-domain basis addend, and
/// accumulate the next sumcheck message over the corrected outputs.
///
/// # Safety
/// Requires the `aes` target feature. The architecture-selecting wrapper
/// checks equal polynomial/output lengths, even output alignment, global
/// source coverage for `[base, base + nf.len())`, and local addend coverage
/// for `[0, nf.len())`.
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_two_and_msg_with_scaled_basis_addend(
    f: &[F128],
    b: &[F128],
    basis_addend: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
    scale: F128,
) -> (F128, F128) {
    // SAFETY: same contract as the caller; the two arms differ only in how
    // the fixed-multiplier products are formed.
    unsafe {
        if use_fold_const_prep() {
            fold_two_and_msg_with_scaled_basis_addend_impl::<true>(f, b, basis_addend, base, nf, nb, r, scale)
        } else {
            fold_two_and_msg_with_scaled_basis_addend_impl::<false>(f, b, basis_addend, base, nf, nb, r, scale)
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fold_two_and_msg_with_scaled_basis_addend_impl<const PREP: bool>(
    f: &[F128],
    b: &[F128],
    basis_addend: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
    scale: F128,
) -> (F128, F128) {
    unsafe {
        let r_q = transmute::<F128, uint64x2_t>(r);
        let r_prep = const_prep(r_q);
        let scale_q = transmute::<F128, uint64x2_t>(scale);
        let scale_prep = const_prep(scale_q);
        let mut u0 = TriNeon::zero();
        let mut u2 = TriNeon::zero();
        let mut t = 0usize;
        while t < nf.len() {
            let source = 2 * (base + t);
            let (f_even0, f_odd0) = load_pair_whole(f.as_ptr().add(source));
            let (f_even1, f_odd1) = load_pair_whole(f.as_ptr().add(source + 2));
            let (f0, f1) = if PREP {
                (
                    mul_const_prep_xor(r_prep, veorq_u64(f_even0, f_odd0), f_even0),
                    mul_const_prep_xor(r_prep, veorq_u64(f_even1, f_odd1), f_even1),
                )
            } else {
                let folded_f = mul_const_vec2_pmred(
                    r_q,
                    veorq_u64(f_even0, f_odd0),
                    veorq_u64(f_even1, f_odd1),
                );
                (
                    veorq_u64(f_even0, folded_f[0]),
                    veorq_u64(f_even1, folded_f[1]),
                )
            };

            let (b_even0, b_odd0) = load_pair_whole(b.as_ptr().add(source));
            let (b_even1, b_odd1) = load_pair_whole(b.as_ptr().add(source + 2));
            let (addend0, addend1) = load_pair_whole(basis_addend.as_ptr().add(t));
            let (b0, b1) = if PREP {
                (
                    mul_two_const_sum_prep_xor(
                        r_prep,
                        veorq_u64(b_even0, b_odd0),
                        scale_prep,
                        addend0,
                        b_even0,
                    ),
                    mul_two_const_sum_prep_xor(
                        r_prep,
                        veorq_u64(b_even1, b_odd1),
                        scale_prep,
                        addend1,
                        b_even1,
                    ),
                )
            } else {
                let folded_b_with_addend = mul_two_const_sum_vec2_pmred(
                    r_q,
                    veorq_u64(b_even0, b_odd0),
                    veorq_u64(b_even1, b_odd1),
                    scale_q,
                    addend0,
                    addend1,
                );
                (
                    veorq_u64(b_even0, folded_b_with_addend[0]),
                    veorq_u64(b_even1, folded_b_with_addend[1]),
                )
            };

            vst1q_u64(nf.as_mut_ptr().add(t).cast::<u64>(), f0);
            vst1q_u64(nf.as_mut_ptr().add(t + 1).cast::<u64>(), f1);
            vst1q_u64(nb.as_mut_ptr().add(t).cast::<u64>(), b0);
            vst1q_u64(nb.as_mut_ptr().add(t + 1).cast::<u64>(), b1);

            tri_acc_q(&mut u0, f0, b0);
            tri_acc_q(&mut u2, veorq_u64(f0, f1), veorq_u64(b0, b1));
            t += 2;
        }
        let reduced = reduce_wide_pair(tri_to_wide(u0), tri_to_wide(u2));
        (
            transmute::<uint64x2_t, F128>(reduced[0]),
            transmute::<uint64x2_t, F128>(reduced[1]),
        )
    }
}

/// Fold two incumbent polynomials, consume an input-domain deferred basis,
/// inject a scaled chunk-local addend, and accumulate the next message.
///
/// With prepared constants enabled, `alpha * d_even + alpha_r * (d_even + d_odd)`
/// rewrites as `(alpha + alpha_r) * d_even + alpha_r * d_odd`, so all four
/// basis products plus `b_even` share a single `fold_lm` without reloading
/// `deferred_basis`, and the message accumulates via macro-fused [`TriNeon`].
///
/// # Safety
/// Requires the `aes` target feature. The architecture-selecting wrapper
/// checks equal input domains, global source coverage for
/// `[base, base + nf.len())`, exact local-addend/output coverage, and even
/// output/base alignment. `alpha_r` must equal `alpha * r`.
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn fold_two_and_msg_with_deferred_basis_and_scaled_local_addend(
    f: &[F128],
    b: &[F128],
    deferred_basis: &[F128],
    local_addend: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
    alpha: F128,
    alpha_r: F128,
    gamma: F128,
) -> (F128, F128) {
    unsafe {
        if use_fold_const_prep() {
            fold_two_and_msg_with_deferred_basis_and_scaled_local_addend_impl::<true>(
                f, b, deferred_basis, local_addend, base, nf, nb, r, alpha, alpha_r, gamma,
            )
        } else {
            fold_two_and_msg_with_deferred_basis_and_scaled_local_addend_impl::<false>(
                f, b, deferred_basis, local_addend, base, nf, nb, r, alpha, alpha_r, gamma,
            )
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fold_two_and_msg_with_deferred_basis_and_scaled_local_addend_impl<const PREP: bool>(
    f: &[F128],
    b: &[F128],
    deferred_basis: &[F128],
    local_addend: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
    alpha: F128,
    alpha_r: F128,
    gamma: F128,
) -> (F128, F128) {
    unsafe {
        let r_q = transmute::<F128, uint64x2_t>(r);
        let r_prep = const_prep(r_q);
        let mut t = 0usize;
        while t < nf.len() {
            let source = 2 * (base + t);
            let (f_even0, f_odd0) = load_pair_whole(f.as_ptr().add(source));
            let (f_even1, f_odd1) = load_pair_whole(f.as_ptr().add(source + 2));
            let (f0, f1) = if PREP {
                (
                    mul_const_prep_xor(r_prep, veorq_u64(f_even0, f_odd0), f_even0),
                    mul_const_prep_xor(r_prep, veorq_u64(f_even1, f_odd1), f_even1),
                )
            } else {
                let folded_f =
                    mul_const_vec2(r_q, veorq_u64(f_even0, f_odd0), veorq_u64(f_even1, f_odd1));
                (
                    veorq_u64(f_even0, folded_f[0]),
                    veorq_u64(f_even1, folded_f[1]),
                )
            };
            vst1q_u64(nf.as_mut_ptr().add(t).cast::<u64>(), f0);
            vst1q_u64(nf.as_mut_ptr().add(t + 1).cast::<u64>(), f1);
            t += 2;
        }

        let alpha_q = transmute::<F128, uint64x2_t>(alpha);
        let alpha_r_q = transmute::<F128, uint64x2_t>(alpha_r);
        let gamma_q = transmute::<F128, uint64x2_t>(gamma);
        if PREP {
            let alpha_one_plus_r_prep = const_prep(veorq_u64(alpha_q, alpha_r_q));
            let alpha_r_prep = const_prep(alpha_r_q);
            let gamma_prep = const_prep(gamma_q);
            let mut t = 0usize;
            while t < nb.len() {
                let source = 2 * (base + t);
                let (b_even0, b_odd0) = load_pair_whole(b.as_ptr().add(source));
                let (b_even1, b_odd1) = load_pair_whole(b.as_ptr().add(source + 2));
                let (d_even0, d_odd0) = load_pair_whole(deferred_basis.as_ptr().add(source));
                let (d_even1, d_odd1) = load_pair_whole(deferred_basis.as_ptr().add(source + 2));
                let (addend0, addend1) = load_pair_whole(local_addend.as_ptr().add(t));
                let b0 = mul_four_const_sum_prep_xor(
                    r_prep,
                    veorq_u64(b_even0, b_odd0),
                    alpha_one_plus_r_prep,
                    d_even0,
                    alpha_r_prep,
                    d_odd0,
                    gamma_prep,
                    addend0,
                    b_even0,
                );
                let b1 = mul_four_const_sum_prep_xor(
                    r_prep,
                    veorq_u64(b_even1, b_odd1),
                    alpha_one_plus_r_prep,
                    d_even1,
                    alpha_r_prep,
                    d_odd1,
                    gamma_prep,
                    addend1,
                    b_even1,
                );
                vst1q_u64(nb.as_mut_ptr().add(t).cast::<u64>(), b0);
                vst1q_u64(nb.as_mut_ptr().add(t + 1).cast::<u64>(), b1);
                t += 2;
            }
        } else {
            let mut t = 0usize;
            while t < nb.len() {
                let source = 2 * (base + t);
                let b_even0 = vld1q_u64(b.as_ptr().add(source).cast::<u64>());
                let b_odd0 = vld1q_u64(b.as_ptr().add(source + 1).cast::<u64>());
                let b_even1 = vld1q_u64(b.as_ptr().add(source + 2).cast::<u64>());
                let b_odd1 = vld1q_u64(b.as_ptr().add(source + 3).cast::<u64>());
                let deferred_even0 = vld1q_u64(deferred_basis.as_ptr().add(source).cast::<u64>());
                let deferred_even1 = vld1q_u64(deferred_basis.as_ptr().add(source + 2).cast::<u64>());
                let incumbent_and_deferred_even = mul_two_const_sum_vec2(
                    r_q,
                    veorq_u64(b_even0, b_odd0),
                    veorq_u64(b_even1, b_odd1),
                    alpha_q,
                    deferred_even0,
                    deferred_even1,
                );
                let partial_b0 = veorq_u64(b_even0, incumbent_and_deferred_even[0]);
                let partial_b1 = veorq_u64(b_even1, incumbent_and_deferred_even[1]);

                let deferred_even0 = vld1q_u64(deferred_basis.as_ptr().add(source).cast::<u64>());
                let deferred_odd0 = vld1q_u64(deferred_basis.as_ptr().add(source + 1).cast::<u64>());
                let deferred_even1 = vld1q_u64(deferred_basis.as_ptr().add(source + 2).cast::<u64>());
                let deferred_odd1 = vld1q_u64(deferred_basis.as_ptr().add(source + 3).cast::<u64>());
                let addend0 = vld1q_u64(local_addend.as_ptr().add(t).cast::<u64>());
                let addend1 = vld1q_u64(local_addend.as_ptr().add(t + 1).cast::<u64>());
                let deferred_delta_and_addend = mul_two_const_sum_vec2(
                    alpha_r_q,
                    veorq_u64(deferred_even0, deferred_odd0),
                    veorq_u64(deferred_even1, deferred_odd1),
                    gamma_q,
                    addend0,
                    addend1,
                );
                let b0 = veorq_u64(partial_b0, deferred_delta_and_addend[0]);
                let b1 = veorq_u64(partial_b1, deferred_delta_and_addend[1]);
                vst1q_u64(nb.as_mut_ptr().add(t).cast::<u64>(), b0);
                vst1q_u64(nb.as_mut_ptr().add(t + 1).cast::<u64>(), b1);
                t += 2;
            }
        }

        let mut u0_a = TriNeon::zero();
        let mut u0_b = TriNeon::zero();
        let mut u2_a = TriNeon::zero();
        let mut u2_b = TriNeon::zero();
        let main = nf.len() & !3;
        let mut t = 0usize;
        while t < main {
            let (f0, f1) = load_pair_whole(nf.as_ptr().add(t));
            let (f2, f3) = load_pair_whole(nf.as_ptr().add(t + 2));
            let (b0, b1) = load_pair_whole(nb.as_ptr().add(t));
            let (b2, b3) = load_pair_whole(nb.as_ptr().add(t + 2));
            tri_acc_q(&mut u0_a, f0, b0);
            tri_acc_q(&mut u0_b, f2, b2);
            tri_acc_q(&mut u2_a, veorq_u64(f0, f1), veorq_u64(b0, b1));
            tri_acc_q(&mut u2_b, veorq_u64(f2, f3), veorq_u64(b2, b3));
            t += 4;
        }
        while t < nf.len() {
            let (f0, f1) = load_pair_whole(nf.as_ptr().add(t));
            let (b0, b1) = load_pair_whole(nb.as_ptr().add(t));
            tri_acc_q(&mut u0_a, f0, b0);
            tri_acc_q(&mut u2_a, veorq_u64(f0, f1), veorq_u64(b0, b1));
            t += 2;
        }
        for (left, right) in [(&mut u0_a, u0_b), (&mut u2_a, u2_b)] {
            left.ll = veorq_u64(left.ll, right.ll);
            left.hh = veorq_u64(left.hh, right.hh);
            left.cross = veorq_u64(left.cross, right.cross);
        }
        let reduced = reduce_wide_pair(tri_to_wide(u0_a), tri_to_wide(u2_a));
        (
            transmute::<uint64x2_t, F128>(reduced[0]),
            transmute::<uint64x2_t, F128>(reduced[1]),
        )
    }
}

/// Ticket-14 NEON: two-challenge fused fold. Per output group of 4, loads 16
/// source pairs per polynomial, binds `(r_a, r_b)` in a single-fold bilinear
/// prepared step when `prep_on` is enabled (`fe0 ^ r_a*(fe0^fo0) ^
/// r_b*(fe0^fe1) ^ (r_a*r_b)*(fe0^fo0^fe1^fo1)`), stores 4 outputs per
/// polynomial, and accumulates the direct message plus the 6 lookahead
/// coefficients via macro-fused [`TriNeon`].
///
/// # Safety
/// Caller guarantees PMULL and: `f.len() == b.len()`, `wf.len() == wb.len()`,
/// `wf.len() % 4 == 0`, `base % 4 == 0`, and `4 * (base + wf.len()) <= f.len()`.
unsafe fn fold2_two_and_msgs_impl(
    f: &[F128],
    b: &[F128],
    base: usize,
    wf: &mut [F128],
    wb: &mut [F128],
    r_a: F128,
    r_b: F128,
    nt_stores: bool,
    prep_on: bool,
) -> (F128, F128, [F128; 6]) {
    #[inline(always)]
    unsafe fn store_pair_nt(dst: *mut F128, x: uint64x2_t, y: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {x:q}, {y:q}, [{dst}]",
                dst = in(reg) dst,
                x = in(vreg) x,
                y = in(vreg) y,
                options(nostack, preserves_flags),
            );
        }
    }
    unsafe {
        let zero = vdupq_n_u64(0);
        let ra_q = transmute::<F128, uint64x2_t>(r_a);
        let rb_q = transmute::<F128, uint64x2_t>(r_b);
        let rab_q = transmute::<F128, uint64x2_t>(r_a * r_b);
        let ra_p = const_prep(ra_q);
        let rb_p = const_prep(rb_q);
        let rab_p = const_prep(rab_q);
        let mut a_u0a = TriNeon::zero();
        let mut a_u0b = TriNeon::zero();
        let mut a_u2a = TriNeon::zero();
        let mut a_u2b = TriNeon::zero();
        let mut a_c1 = TriNeon::zero();
        let mut a_c3 = TriNeon::zero();
        let mut a_c4 = TriNeon::zero();
        let mut a_c5 = TriNeon::zero();

        let mut w_regs_f = [zero; 4];
        let mut w_regs_b = [zero; 4];
        let mut t = 0;
        while t < wf.len() {
            for q in 0..4 {
                let src = 4 * (base + t + q);
                let (fe0, fo0) = load_pair_whole(f.as_ptr().add(src));
                let (fe1, fo1) = load_pair_whole(f.as_ptr().add(src + 2));
                let (be0, bo0) = load_pair_whole(b.as_ptr().add(src));
                let (be1, bo1) = load_pair_whole(b.as_ptr().add(src + 2));

                let (wq_f, wq_b) = if prep_on {
                    let df0 = veorq_u64(fe0, fo0);
                    let df1 = veorq_u64(fe1, fo1);
                    let db0 = veorq_u64(be0, bo0);
                    let db1 = veorq_u64(be1, bo1);
                    (
                        mul_three_const_sum_prep_xor(
                            ra_p,
                            df0,
                            rb_p,
                            veorq_u64(fe0, fe1),
                            rab_p,
                            veorq_u64(df0, df1),
                            fe0,
                        ),
                        mul_three_const_sum_prep_xor(
                            ra_p,
                            db0,
                            rb_p,
                            veorq_u64(be0, be1),
                            rab_p,
                            veorq_u64(db0, db1),
                            be0,
                        ),
                    )
                } else {
                    let pf = mul_const_vec2(ra_q, veorq_u64(fe0, fo0), veorq_u64(fe1, fo1));
                    let vf0 = veorq_u64(fe0, pf[0]);
                    let vf1 = veorq_u64(fe1, pf[1]);
                    let pb = mul_const_vec2(ra_q, veorq_u64(be0, bo0), veorq_u64(be1, bo1));
                    let vb0 = veorq_u64(be0, pb[0]);
                    let vb1 = veorq_u64(be1, pb[1]);
                    let pw = mul_const_vec2(rb_q, veorq_u64(vf0, vf1), veorq_u64(vb0, vb1));
                    (veorq_u64(vf0, pw[0]), veorq_u64(vb0, pw[1]))
                };
                if !nt_stores {
                    vst1q_u64(wf.as_mut_ptr().add(t + q).cast::<u64>(), wq_f);
                    vst1q_u64(wb.as_mut_ptr().add(t + q).cast::<u64>(), wq_b);
                }
                w_regs_f[q] = wq_f;
                w_regs_b[q] = wq_b;
            }
            if nt_stores {
                store_pair_nt(wf.as_mut_ptr().add(t), w_regs_f[0], w_regs_f[1]);
                store_pair_nt(wf.as_mut_ptr().add(t + 2), w_regs_f[2], w_regs_f[3]);
                store_pair_nt(wb.as_mut_ptr().add(t), w_regs_b[0], w_regs_b[1]);
                store_pair_nt(wb.as_mut_ptr().add(t + 2), w_regs_b[2], w_regs_b[3]);
            }
            let s0f = veorq_u64(w_regs_f[0], w_regs_f[1]);
            let s0b = veorq_u64(w_regs_b[0], w_regs_b[1]);
            let s1f = veorq_u64(w_regs_f[2], w_regs_f[3]);
            let s1b = veorq_u64(w_regs_b[2], w_regs_b[3]);
            tri_acc_q(&mut a_u0a, w_regs_f[0], w_regs_b[0]);
            tri_acc_q(&mut a_u0b, w_regs_f[2], w_regs_b[2]);
            tri_acc_q(&mut a_u2a, s0f, s0b);
            tri_acc_q(&mut a_u2b, s1f, s1b);
            tri_acc_q(&mut a_c1, w_regs_f[1], w_regs_b[1]);
            let e_f = veorq_u64(w_regs_f[0], w_regs_f[2]);
            let o_f = veorq_u64(w_regs_f[1], w_regs_f[3]);
            let e_b = veorq_u64(w_regs_b[0], w_regs_b[2]);
            let o_b = veorq_u64(w_regs_b[1], w_regs_b[3]);
            let se_f = veorq_u64(e_f, o_f);
            let se_b = veorq_u64(e_b, o_b);
            tri_acc_q(&mut a_c3, e_f, e_b);
            tri_acc_q(&mut a_c4, o_f, o_b);
            tri_acc_q(&mut a_c5, se_f, se_b);
            t += 4;
        }
        let red = |w: TriNeon| transmute::<uint64x2_t, F128>(reduce_wide(tri_to_wide(w)));
        let c0 = red(a_u0a);
        let c2v = red(a_u2a);
        let c1 = red(a_c1) + c0 + c2v;
        let c3 = red(a_c3);
        let c5 = red(a_c5);
        let c4 = red(a_c4) + c3 + c5;
        let u_0 = c0 + red(a_u0b);
        let u_2 = c2v + red(a_u2b);
        (u_0, u_2, [c0, c1, c2v, c3, c4, c5])
    }
}

/// Final two-challenge Ligerito fold. This is the direct-message-only sibling
/// of [`fold2_two_and_msgs`]: after the last initial-lane pair there is no
/// following lookahead to evaluate, so retaining its four extra endpoint
/// products and accumulators is dead work.
///
/// # Safety
/// Caller guarantees PMULL and: `f.len() == b.len()`, `wf.len() == wb.len()`,
/// `wf.len() % 4 == 0`, `base % 4 == 0`, and `4 * (base + wf.len()) <= f.len()`.
unsafe fn fold2_two_and_msg_impl(
    f: &[F128],
    b: &[F128],
    base: usize,
    wf: &mut [F128],
    wb: &mut [F128],
    r_a: F128,
    r_b: F128,
    nt_stores: bool,
    prep_on: bool,
) -> (F128, F128) {
    #[inline(always)]
    unsafe fn store_pair_nt(dst: *mut F128, x: uint64x2_t, y: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {x:q}, {y:q}, [{dst}]",
                dst = in(reg) dst,
                x = in(vreg) x,
                y = in(vreg) y,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        let zero = vdupq_n_u64(0);
        let ra_q = transmute::<F128, uint64x2_t>(r_a);
        let rb_q = transmute::<F128, uint64x2_t>(r_b);
        let rab_q = transmute::<F128, uint64x2_t>(r_a * r_b);
        let ra_p = const_prep(ra_q);
        let rb_p = const_prep(rb_q);
        let rab_p = const_prep(rab_q);
        let mut a_u0a = TriNeon::zero();
        let mut a_u0b = TriNeon::zero();
        let mut a_u2a = TriNeon::zero();
        let mut a_u2b = TriNeon::zero();

        let mut w_regs_f = [zero; 4];
        let mut w_regs_b = [zero; 4];
        let mut t = 0;
        while t < wf.len() {
            for q in 0..4 {
                let src = 4 * (base + t + q);
                let (fe0, fo0) = load_pair_whole(f.as_ptr().add(src));
                let (fe1, fo1) = load_pair_whole(f.as_ptr().add(src + 2));
                let (be0, bo0) = load_pair_whole(b.as_ptr().add(src));
                let (be1, bo1) = load_pair_whole(b.as_ptr().add(src + 2));

                let (wq_f, wq_b) = if prep_on {
                    let df0 = veorq_u64(fe0, fo0);
                    let df1 = veorq_u64(fe1, fo1);
                    let db0 = veorq_u64(be0, bo0);
                    let db1 = veorq_u64(be1, bo1);
                    (
                        mul_three_const_sum_prep_xor(
                            ra_p,
                            df0,
                            rb_p,
                            veorq_u64(fe0, fe1),
                            rab_p,
                            veorq_u64(df0, df1),
                            fe0,
                        ),
                        mul_three_const_sum_prep_xor(
                            ra_p,
                            db0,
                            rb_p,
                            veorq_u64(be0, be1),
                            rab_p,
                            veorq_u64(db0, db1),
                            be0,
                        ),
                    )
                } else {
                    let pf = mul_const_vec2(ra_q, veorq_u64(fe0, fo0), veorq_u64(fe1, fo1));
                    let vf0 = veorq_u64(fe0, pf[0]);
                    let vf1 = veorq_u64(fe1, pf[1]);
                    let pb = mul_const_vec2(ra_q, veorq_u64(be0, bo0), veorq_u64(be1, bo1));
                    let vb0 = veorq_u64(be0, pb[0]);
                    let vb1 = veorq_u64(be1, pb[1]);
                    let pw = mul_const_vec2(rb_q, veorq_u64(vf0, vf1), veorq_u64(vb0, vb1));
                    (veorq_u64(vf0, pw[0]), veorq_u64(vb0, pw[1]))
                };
                if !nt_stores {
                    vst1q_u64(wf.as_mut_ptr().add(t + q).cast::<u64>(), wq_f);
                    vst1q_u64(wb.as_mut_ptr().add(t + q).cast::<u64>(), wq_b);
                }
                w_regs_f[q] = wq_f;
                w_regs_b[q] = wq_b;
            }
            if nt_stores {
                store_pair_nt(wf.as_mut_ptr().add(t), w_regs_f[0], w_regs_f[1]);
                store_pair_nt(wf.as_mut_ptr().add(t + 2), w_regs_f[2], w_regs_f[3]);
                store_pair_nt(wb.as_mut_ptr().add(t), w_regs_b[0], w_regs_b[1]);
                store_pair_nt(wb.as_mut_ptr().add(t + 2), w_regs_b[2], w_regs_b[3]);
            }

            let s0f = veorq_u64(w_regs_f[0], w_regs_f[1]);
            let s0b = veorq_u64(w_regs_b[0], w_regs_b[1]);
            let s1f = veorq_u64(w_regs_f[2], w_regs_f[3]);
            let s1b = veorq_u64(w_regs_b[2], w_regs_b[3]);
            tri_acc_q(&mut a_u0a, w_regs_f[0], w_regs_b[0]);
            tri_acc_q(&mut a_u0b, w_regs_f[2], w_regs_b[2]);
            tri_acc_q(&mut a_u2a, s0f, s0b);
            tri_acc_q(&mut a_u2b, s1f, s1b);
            t += 4;
        }

        for (left, right) in [(&mut a_u0a, a_u0b), (&mut a_u2a, a_u2b)] {
            left.ll = veorq_u64(left.ll, right.ll);
            left.hh = veorq_u64(left.hh, right.hh);
            left.cross = veorq_u64(left.cross, right.cross);
        }
        let reduced = reduce_wide_pair(tri_to_wide(a_u0a), tri_to_wide(a_u2a));
        (
            transmute::<uint64x2_t, F128>(reduced[0]),
            transmute::<uint64x2_t, F128>(reduced[1]),
        )
    }
}

/// Prepared fixed-multiplier products in the two-challenge fold kernels;
/// `FLOCK_NO_FOLD2_CONST_PREP=1` restores the Karatsuba constant kernel.
#[inline]
fn use_fold2_const_prep() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_FOLD2_CONST_PREP").is_none());
    *ON
}

/// See [`fold2_two_and_msgs_impl`]; selects the constant-product form once.
///
/// # Safety
/// As [`fold2_two_and_msgs_impl`].
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn fold2_two_and_msgs(
    f: &[F128],
    b: &[F128],
    base: usize,
    wf: &mut [F128],
    wb: &mut [F128],
    r_a: F128,
    r_b: F128,
    nt_stores: bool,
) -> (F128, F128, [F128; 6]) {
    unsafe { fold2_two_and_msgs_impl(f, b, base, wf, wb, r_a, r_b, nt_stores, use_fold2_const_prep()) }
}

/// See [`fold2_two_and_msg_impl`]; selects the constant-product form once.
///
/// # Safety
/// As [`fold2_two_and_msg_impl`].
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn fold2_two_and_msg(
    f: &[F128],
    b: &[F128],
    base: usize,
    wf: &mut [F128],
    wb: &mut [F128],
    r_a: F128,
    r_b: F128,
    nt_stores: bool,
) -> (F128, F128) {
    unsafe { fold2_two_and_msg_impl(f, b, base, wf, wb, r_a, r_b, nt_stores, use_fold2_const_prep()) }
}

/// `out[i] = z0[i] + r * (z0[i] + z1[i])` with `r` prepared once and `z0[i]`
/// absorbed into [`mul_const_prep_xor`] at zero extra SIMD issue slots.
///
/// # Safety
/// Requires the `aes` target feature and equal slice lengths.
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_half_into(z0: &[F128], z1: &[F128], out: &mut [F128], r: F128) {
    use crate::field::gf2_128::aarch64::{ghash_mul_const_vec2_neon, use_const_vec2_prep};

    debug_assert_eq!(z0.len(), z1.len());
    debug_assert_eq!(z0.len(), out.len());
    let n = z0.len();
    if use_const_vec2_prep() {
        unsafe {
            let p = const_prep(transmute::<F128, uint64x2_t>(r));
            let p0 = z0.as_ptr();
            let p1 = z1.as_ptr();
            let po = out.as_mut_ptr();
            let main = n & !3;
            let mut i = 0usize;
            while i < main {
                let (a0, a1) = load_pair_whole(p0.add(i));
                let (a2, a3) = load_pair_whole(p0.add(i + 2));
                let (b0, b1) = load_pair_whole(p1.add(i));
                let (b2, b3) = load_pair_whole(p1.add(i + 2));
                let o0 = mul_const_prep_xor(p, veorq_u64(a0, b0), a0);
                let o1 = mul_const_prep_xor(p, veorq_u64(a1, b1), a1);
                let o2 = mul_const_prep_xor(p, veorq_u64(a2, b2), a2);
                let o3 = mul_const_prep_xor(p, veorq_u64(a3, b3), a3);
                store_pair_whole(po.add(i), o0, o1);
                store_pair_whole(po.add(i + 2), o2, o3);
                i += 4;
            }
            while i < n {
                let a = load_whole(p0.add(i));
                let b = load_whole(p1.add(i));
                vst1q_u64(
                    po.add(i).cast::<u64>(),
                    mul_const_prep_xor(p, veorq_u64(a, b), a),
                );
                i += 1;
            }
        }
        return;
    }

    let paired = n & !1;
    let mut i = 0usize;
    while i < paired {
        let a0 = z0[i];
        let a1 = z0[i + 1];
        let prod = unsafe { ghash_mul_const_vec2_neon(r, [a0 + z1[i], a1 + z1[i + 1]]) };
        out[i] = a0 + prod[0];
        out[i + 1] = a1 + prod[1];
        i += 2;
    }
    if i < n {
        let a = z0[i];
        out[i] = a + r * (a + z1[i]);
    }
}

/// `out[i] = src[i] * s` with the multiplier prepared once, unrolled 4-wide
/// with whole-pair loads and stores.
///
/// # Safety
/// Requires the `aes` target feature and `out.len() == src.len()`.
#[target_feature(enable = "aes")]
pub(super) unsafe fn scale_slice_prepared(out: &mut [F128], src: &[F128], s: F128) {
    unsafe {
        let p = const_prep(transmute::<F128, uint64x2_t>(s));
        let n = out.len().min(src.len());
        let (po, ps) = (out.as_mut_ptr(), src.as_ptr());
        let main = n & !3;
        let mut i = 0usize;
        while i < main {
            let (v0, v1) = load_pair_whole(ps.add(i));
            let (v2, v3) = load_pair_whole(ps.add(i + 2));
            let o0 = mul_const_prep(p, v0);
            let o1 = mul_const_prep(p, v1);
            let o2 = mul_const_prep(p, v2);
            let o3 = mul_const_prep(p, v3);
            store_pair_whole(po.add(i), o0, o1);
            store_pair_whole(po.add(i + 2), o2, o3);
            i += 4;
        }
        while i < n {
            let v = load_whole(ps.add(i));
            vst1q_u64(po.add(i).cast::<u64>(), mul_const_prep(p, v));
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prepared row scale equals the generic field multiply, entry by
    /// entry, for edge multipliers and odd lengths.
    #[test]
    fn prepared_scale_matches_generic() {
        let mut state = 0x5343_414C_4553_4C43u64;
        let consts = test_field_cases(&mut state);
        for &n in &[1usize, 2, 3, 127, 128, 1000] {
            for &sc in consts.iter().step_by(3) {
                let src: Vec<F128> =
                    (0..n).map(|_| F128::new(test_xorshift(&mut state), test_xorshift(&mut state))).collect();
                let mut got = vec![F128::ONE; n];
                unsafe { scale_slice_prepared(&mut got, &src, sc) };
                let want: Vec<F128> = src.iter().map(|&l| l * sc).collect();
                assert_eq!(got, want, "n={n}");
            }
        }
    }

    /// Both two-challenge fold kernels give the same folded slices and
    /// messages with prepared and with Karatsuba constant products.
    #[test]
    fn prepared_fold2_matches_incumbent() {
        let mut state = 0x464F_4C44_3250_5245u64;
        let rand = |state: &mut u64| F128::new(test_xorshift(state), test_xorshift(state));
        for &outputs in &[4usize, 8, 64, 512] {
            for case in 0..5 {
                let base = 4 * (case % 2);
                let f: Vec<F128> = (0..4 * (base + outputs)).map(|_| rand(&mut state)).collect();
                let b: Vec<F128> = (0..4 * (base + outputs)).map(|_| rand(&mut state)).collect();
                let (r_a, r_b) = match case {
                    0 => (F128::ZERO, F128::ONE),
                    1 => (F128::new(0, u64::MAX), F128::new(u64::MAX, 0)),
                    _ => (rand(&mut state), rand(&mut state)),
                };
                for nt in [false, true] {
                    unsafe {
                        let run = |prep: bool| {
                            let mut wf = vec![F128::ZERO; outputs];
                            let mut wb = vec![F128::ZERO; outputs];
                            let msg = fold2_two_and_msgs_impl(&f, &b, base, &mut wf, &mut wb, r_a, r_b, nt, prep);
                            (wf, wb, msg)
                        };
                        assert_eq!(run(true), run(false), "fold2 msgs outputs={outputs} case={case}");
                        let run = |prep: bool| {
                            let mut wf = vec![F128::ZERO; outputs];
                            let mut wb = vec![F128::ZERO; outputs];
                            let msg = fold2_two_and_msg_impl(&f, &b, base, &mut wf, &mut wb, r_a, r_b, nt, prep);
                            (wf, wb, msg)
                        };
                        assert_eq!(run(true), run(false), "fold2 msg outputs={outputs} case={case}");
                    }
                }
            }
        }
    }

    /// Local timing probe (not a correctness test): the equality-table outer
    /// product row (128 and 256 entries) and the two-challenge fold kernel.
    #[test]
    #[ignore]
    fn prepared_scale_fold2_timing() {
        let mut state = 0x5449_4D45_5343_414Cu64;
        let rand = |state: &mut u64| F128::new(test_xorshift(state), test_xorshift(state));
        for &row in &[128usize, 256] {
            let src: Vec<F128> = (0..row).map(|_| rand(&mut state)).collect();
            let rows: Vec<F128> = (0..256).map(|_| rand(&mut state)).collect();
            let mut out = vec![F128::ZERO; row * rows.len()];
            let mut best = [f64::MAX; 2];
            for _trial in 0..15 {
                for which in 0..2 {
                    let t0 = std::time::Instant::now();
                    for _ in 0..50 {
                        for (chunk, &sc) in out.chunks_mut(row).zip(rows.iter()) {
                            if which == 0 {
                                for (o, &l) in chunk.iter_mut().zip(&src) {
                                    *o = l * sc;
                                }
                            } else {
                                unsafe { scale_slice_prepared(chunk, &src, sc) };
                            }
                        }
                        std::hint::black_box(&mut out);
                    }
                    let ns = t0.elapsed().as_secs_f64() * 1e9 / (50 * out.len()) as f64;
                    best[which] = best[which].min(ns);
                }
            }
            eprintln!(
                "[scale-timing] row={row} ns per entry (best of 15): generic {:.3}, prepared {:.3} ({:+.1}%)",
                best[0], best[1], (best[1] / best[0] - 1.0) * 100.0
            );
        }
        const OUT: usize = 512;
        let f: Vec<F128> = (0..4 * OUT).map(|_| rand(&mut state)).collect();
        let b: Vec<F128> = (0..4 * OUT).map(|_| rand(&mut state)).collect();
        let (r_a, r_b) = (rand(&mut state), rand(&mut state));
        let mut wf = vec![F128::ZERO; OUT];
        let mut wb = vec![F128::ZERO; OUT];
        let mut best = [f64::MAX; 2];
        for _trial in 0..15 {
            for which in 0..2 {
                let t0 = std::time::Instant::now();
                let mut sink = F128::ZERO;
                for _ in 0..4000 {
                    let msg = unsafe {
                        fold2_two_and_msgs_impl(&f, &b, 0, &mut wf, &mut wb, r_a, r_b, false, which == 1)
                    };
                    sink += msg.0;
                    std::hint::black_box((&mut wf, &mut wb));
                }
                std::hint::black_box(sink);
                let ns = t0.elapsed().as_secs_f64() * 1e9 / (4000 * OUT) as f64;
                best[which] = best[which].min(ns);
            }
        }
        eprintln!(
            "[fold2-timing] ns per output pair (best of 15): incumbent {:.3}, prepared {:.3} ({:+.1}%)",
            best[0], best[1], (best[1] / best[0] - 1.0) * 100.0
        );
    }

    #[test]
    fn wide_reduction_pair_matches_two_serial_reductions() {
        unsafe fn from_words(words: [u64; 4]) -> WideNeon {
            unsafe {
                WideNeon {
                    lo: transmute::<[u64; 2], uint64x2_t>([words[0], words[1]]),
                    hi: transmute::<[u64; 2], uint64x2_t>([words[2], words[3]]),
                }
            }
        }

        unsafe fn assert_pair(first: [u64; 4], second: [u64; 4], label: &str) {
            unsafe {
                let first = from_words(first);
                let second = from_words(second);
                let actual = reduce_wide_pair(first, second);
                let expected = [reduce_wide(first), reduce_wide(second)];
                assert_eq!(
                    transmute::<uint64x2_t, [u64; 2]>(actual[0]),
                    transmute::<uint64x2_t, [u64; 2]>(expected[0]),
                    "first output: {label}"
                );
                assert_eq!(
                    transmute::<uint64x2_t, [u64; 2]>(actual[1]),
                    transmute::<uint64x2_t, [u64; 2]>(expected[1]),
                    "second output: {label}"
                );
            }
        }

        unsafe {
            for bit in 0..256 {
                let mut one_hot = [0u64; 4];
                one_hot[bit / 64] = 1 << (bit % 64);
                assert_pair(one_hot, [0; 4], "one-hot first");
                assert_pair([0; 4], one_hot, "one-hot second");
            }

            let directed = [
                ([0; 4], [0; 4]),
                ([u64::MAX; 4], [u64::MAX; 4]),
                ([0, 0, 0, 1 << 57], [0, 0, 0, 1 << 62]),
                ([0, 0, 0, 1 << 63], [0, 0, 0, u64::MAX]),
                ([u64::MAX, 0, u64::MAX, 0], [0, u64::MAX, 0, u64::MAX]),
            ];
            for (case, (first, second)) in directed.into_iter().enumerate() {
                assert_pair(first, second, &format!("directed case {case}"));
            }

            let mut state = 0x5749_4445_5041_4952u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            for case in 0..16_384 {
                let first = [next(), next(), next(), next()];
                let second = [next(), next(), next(), next()];
                assert_pair(first, second, &format!("random case {case}"));
            }
        }
    }

    fn test_xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn test_field_cases(state: &mut u64) -> Vec<F128> {
        let mut cases = vec![
            F128::ZERO,
            F128::ONE,
            F128::new(0, 1),
            F128::new(u64::MAX, 0),
            F128::new(0, u64::MAX),
            F128::new(u64::MAX, u64::MAX),
            F128::new(0x87, 1 << 63),
            F128::new(1 << 63, 1 << 63),
        ];
        cases.extend((0..160).map(|_| F128::new(test_xorshift(state), test_xorshift(state))));
        cases
    }

    /// The prepared fixed-multiplier products equal the generic field product
    /// and the Karatsuba constant kernels, bit for bit, including the edge
    /// multipliers (zero, one, single limbs, all ones, top bits).
    #[test]
    fn prepared_const_products_match_reference() {
        let mut state = 0x5052_4550_434F_4E53u64;
        let consts = test_field_cases(&mut state);
        for (case, &c) in consts.iter().enumerate() {
            let d = consts[(case * 7 + 3) % consts.len()];
            for &x in consts.iter().step_by(5) {
                let y = F128::new(test_xorshift(&mut state), test_xorshift(&mut state));
                unsafe {
                    let c_q = transmute::<F128, uint64x2_t>(c);
                    let d_q = transmute::<F128, uint64x2_t>(d);
                    let x_q = transmute::<F128, uint64x2_t>(x);
                    let y_q = transmute::<F128, uint64x2_t>(y);
                    let got = transmute::<uint64x2_t, F128>(mul_const_prep(const_prep(c_q), x_q));
                    assert_eq!(got, x * c, "single product case={case}");
                    let karatsuba = mul_const_vec2(c_q, x_q, y_q);
                    assert_eq!(got, transmute::<uint64x2_t, F128>(karatsuba[0]));
                    let sum = transmute::<uint64x2_t, F128>(mul_two_const_sum_prep(
                        const_prep(c_q),
                        x_q,
                        const_prep(d_q),
                        y_q,
                    ));
                    assert_eq!(sum, x * c + y * d, "sum of products case={case}");
                }
            }
        }
    }

    /// Each fused fold kernel returns the same folded slices and the same
    /// message with the prepared products as with the incumbent ones.
    #[test]
    fn prepared_folds_match_incumbent_folds() {
        let mut state = 0x464F_4C44_5052_4550u64;
        let rand = |state: &mut u64| F128::new(test_xorshift(state), test_xorshift(state));
        for &outputs in &[2usize, 4, 6, 64, 2048] {
            for case in 0..6 {
                let f: Vec<F128> = (0..2 * outputs + 8).map(|_| rand(&mut state)).collect();
                let b: Vec<F128> = (0..2 * outputs + 8).map(|_| rand(&mut state)).collect();
                let addend: Vec<F128> = (0..outputs).map(|_| rand(&mut state)).collect();
                let (r, scale) = match case {
                    0 => (F128::ZERO, rand(&mut state)),
                    1 => (F128::ONE, F128::ZERO),
                    2 => (F128::new(0, u64::MAX), F128::ONE),
                    _ => (rand(&mut state), rand(&mut state)),
                };
                let base = 2 * (case % 2);
                unsafe {
                    let run = |prep: bool| {
                        let mut nf = vec![F128::ZERO; outputs];
                        let mut nb = vec![F128::ZERO; outputs];
                        let msg = if prep {
                            fold_two_and_msg_impl::<true>(&f, &b, base, &mut nf, &mut nb, r)
                        } else {
                            fold_two_and_msg_impl::<false>(&f, &b, base, &mut nf, &mut nb, r)
                        };
                        (nf, nb, msg)
                    };
                    assert_eq!(run(true), run(false), "fold_two_and_msg outputs={outputs} case={case}");

                    let run = |prep: bool| {
                        let mut nf = vec![F128::ZERO; outputs];
                        let mut nb = vec![F128::ZERO; outputs];
                        let msg = if prep {
                            fold_two_and_msg_with_scaled_basis_addend_impl::<true>(
                                &f, &b, &addend, 0, &mut nf, &mut nb, r, scale,
                            )
                        } else {
                            fold_two_and_msg_with_scaled_basis_addend_impl::<false>(
                                &f, &b, &addend, 0, &mut nf, &mut nb, r, scale,
                            )
                        };
                        (nf, nb, msg)
                    };
                    assert_eq!(run(true), run(false), "scaled addend outputs={outputs} case={case}");

                    if outputs.is_multiple_of(2) {
                        let run = |prep: bool| {
                            let mut ff = f[..2 * outputs].to_vec();
                            let mut bb = b[..2 * outputs].to_vec();
                            let msg = if prep {
                                fold_two_and_msg_in_place_impl::<true>(&mut ff, &mut bb, r)
                            } else {
                                fold_two_and_msg_in_place_impl::<false>(&mut ff, &mut bb, r)
                            };
                            (ff[..outputs].to_vec(), bb[..outputs].to_vec(), msg)
                        };
                        assert_eq!(run(true), run(false), "in place outputs={outputs} case={case}");
                    }
                }
            }
        }
    }

    /// Local timing probe (not a correctness test) at the ranked chunk shape:
    /// 2048 outputs per call. Run with
    /// `cargo test --profile challenge -p flock-core --lib prepared_fold_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn prepared_fold_timing() {
        let mut state = 0x5449_4D45_464F_4C44u64;
        let rand = |state: &mut u64| F128::new(test_xorshift(state), test_xorshift(state));
        const OUT: usize = 2048;
        let f: Vec<F128> = (0..2 * OUT).map(|_| rand(&mut state)).collect();
        let b: Vec<F128> = (0..2 * OUT).map(|_| rand(&mut state)).collect();
        let addend: Vec<F128> = (0..OUT).map(|_| rand(&mut state)).collect();
        let (r, scale) = (rand(&mut state), rand(&mut state));
        let mut nf = vec![F128::ZERO; OUT];
        let mut nb = vec![F128::ZERO; OUT];
        const REPS: usize = 2000;
        let mut best = [f64::MAX; 4];
        for _trial in 0..15 {
            for which in 0..4 {
                let t0 = std::time::Instant::now();
                let mut sink = F128::ZERO;
                for _ in 0..REPS {
                    let msg = unsafe {
                        match which {
                            0 => fold_two_and_msg_impl::<false>(&f, &b, 0, &mut nf, &mut nb, r),
                            1 => fold_two_and_msg_impl::<true>(&f, &b, 0, &mut nf, &mut nb, r),
                            2 => fold_two_and_msg_with_scaled_basis_addend_impl::<false>(
                                &f, &b, &addend, 0, &mut nf, &mut nb, r, scale,
                            ),
                            _ => fold_two_and_msg_with_scaled_basis_addend_impl::<true>(
                                &f, &b, &addend, 0, &mut nf, &mut nb, r, scale,
                            ),
                        }
                    };
                    sink += msg.0 + msg.1;
                    std::hint::black_box((&mut nf, &mut nb));
                }
                std::hint::black_box(sink);
                let ns = t0.elapsed().as_secs_f64() * 1e9 / (REPS * OUT) as f64;
                best[which] = best[which].min(ns);
            }
        }
        eprintln!(
            "[fold-timing] ns per output pair (best of 15): fold_two_and_msg incumbent {:.3}, prepared {:.3} ({:+.1}%); scaled addend incumbent {:.3}, prepared {:.3} ({:+.1}%)",
            best[0],
            best[1],
            (best[1] / best[0] - 1.0) * 100.0,
            best[2],
            best[3],
            (best[3] / best[2] - 1.0) * 100.0
        );
    }

    #[test]
    fn two_const_sum_vec2_matches_two_reduced_products() {
        let mut state = 0x5457_4F43_4F4E_5354u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut cases = vec![
            (F128::ZERO, F128::ZERO),
            (F128::ZERO, F128::ONE),
            (F128::ONE, F128::ZERO),
            (F128::ONE, F128::ONE),
        ];
        cases.extend((0..128).map(|_| (F128::new(next(), next()), F128::new(next(), next()))));

        for (case, (c, d)) in cases.into_iter().enumerate() {
            let x0 = F128::new(next(), next());
            let x1 = F128::new(next(), next());
            let y0 = F128::new(next(), next());
            let y1 = F128::new(next(), next());
            unsafe {
                let c_q = transmute::<F128, uint64x2_t>(c);
                let d_q = transmute::<F128, uint64x2_t>(d);
                let x0_q = transmute::<F128, uint64x2_t>(x0);
                let x1_q = transmute::<F128, uint64x2_t>(x1);
                let y0_q = transmute::<F128, uint64x2_t>(y0);
                let y1_q = transmute::<F128, uint64x2_t>(y1);

                let cx = mul_const_vec2(c_q, x0_q, x1_q);
                let dy = mul_const_vec2(d_q, y0_q, y1_q);
                let expected = [veorq_u64(cx[0], dy[0]), veorq_u64(cx[1], dy[1])];
                let actual = mul_two_const_sum_vec2(c_q, x0_q, x1_q, d_q, y0_q, y1_q);
                assert_eq!(
                    transmute::<uint64x2_t, F128>(actual[0]),
                    transmute::<uint64x2_t, F128>(expected[0]),
                    "lane=0 case={case}"
                );
                assert_eq!(
                    transmute::<uint64x2_t, F128>(actual[1]),
                    transmute::<uint64x2_t, F128>(expected[1]),
                    "lane=1 case={case}"
                );
            }
        }
    }
}

// Reduce a 256-bit polynomial using carry-less multiplication by x^7+x^2+x+1.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn reduce_pmred(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
    unsafe {
        let p0 = pmull(vgetq_lane_u64::<0>(hi), 0x87);
        let p1 = pmull(vgetq_lane_u64::<1>(hi), 0x87);
        let correction = pmull(vgetq_lane_u64::<1>(p1), 0x87);
        let low = vgetq_lane_u64::<0>(lo) ^ vgetq_lane_u64::<0>(p0) ^ vgetq_lane_u64::<0>(correction);
        let high = vgetq_lane_u64::<1>(lo) ^ vgetq_lane_u64::<1>(p0) ^ vgetq_lane_u64::<0>(p1) ^ vgetq_lane_u64::<1>(correction);
        vcombine_u64(vcreate_u64(low), vcreate_u64(high))
    }
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_const_vec2_pmred(r: uint64x2_t, x0: uint64x2_t, x1: uint64x2_t) -> [uint64x2_t; 2] {
    unsafe {
        let r_lo = vgetq_lane_u64::<0>(r);
        let r_hi = vgetq_lane_u64::<1>(r);
        let r_mid = veorq_u64(r, vextq_u64::<1>(r, r));
        let x0_mid = veorq_u64(x0, vextq_u64::<1>(x0, x0));
        let x1_mid = veorq_u64(x1, vextq_u64::<1>(x1, x1));

        let p0_ll = pmull(vgetq_lane_u64::<0>(x0), r_lo);
        let p0_hh = pmull(vgetq_lane_u64::<1>(x0), r_hi);
        let p0_mm = pmull(vgetq_lane_u64::<0>(x0_mid), vgetq_lane_u64::<0>(r_mid));
        let p1_ll = pmull(vgetq_lane_u64::<0>(x1), r_lo);
        let p1_hh = pmull(vgetq_lane_u64::<1>(x1), r_hi);
        let p1_mm = pmull(vgetq_lane_u64::<0>(x1_mid), vgetq_lane_u64::<0>(r_mid));
        let c0 = xor3_u64(p0_mm, p0_ll, p0_hh);
        let c1 = xor3_u64(p1_mm, p1_ll, p1_hh);

        let zero = vdupq_n_u64(0);
        let lo0 = veorq_u64(p0_ll, vextq_u64::<1>(zero, c0));
        let hi0 = veorq_u64(p0_hh, vextq_u64::<1>(c0, zero));
        let lo1 = veorq_u64(p1_ll, vextq_u64::<1>(zero, c1));
        let hi1 = veorq_u64(p1_hh, vextq_u64::<1>(c1, zero));
        [reduce_pmred(lo0,hi0), reduce_pmred(lo1,hi1)]
    }
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_two_const_sum_vec2_pmred(
    c: uint64x2_t,
    x0: uint64x2_t,
    x1: uint64x2_t,
    d: uint64x2_t,
    y0: uint64x2_t,
    y1: uint64x2_t,
) -> [uint64x2_t; 2] {
    unsafe {
        let c_lo = vgetq_lane_u64::<0>(c);
        let c_hi = vgetq_lane_u64::<1>(c);
        let c_mid = c_lo ^ c_hi;
        let d_lo = vgetq_lane_u64::<0>(d);
        let d_hi = vgetq_lane_u64::<1>(d);
        let d_mid = d_lo ^ d_hi;

        let x0_lo = vgetq_lane_u64::<0>(x0);
        let x0_hi = vgetq_lane_u64::<1>(x0);
        let y0_lo = vgetq_lane_u64::<0>(y0);
        let y0_hi = vgetq_lane_u64::<1>(y0);
        let p0_ll = veorq_u64(pmull(x0_lo, c_lo), pmull(y0_lo, d_lo));
        let p0_hh = veorq_u64(pmull(x0_hi, c_hi), pmull(y0_hi, d_hi));
        let p0_mm = veorq_u64(pmull(x0_lo ^ x0_hi, c_mid), pmull(y0_lo ^ y0_hi, d_mid));

        let x1_lo = vgetq_lane_u64::<0>(x1);
        let x1_hi = vgetq_lane_u64::<1>(x1);
        let y1_lo = vgetq_lane_u64::<0>(y1);
        let y1_hi = vgetq_lane_u64::<1>(y1);
        let p1_ll = veorq_u64(pmull(x1_lo, c_lo), pmull(y1_lo, d_lo));
        let p1_hh = veorq_u64(pmull(x1_hi, c_hi), pmull(y1_hi, d_hi));
        let p1_mm = veorq_u64(pmull(x1_lo ^ x1_hi, c_mid), pmull(y1_lo ^ y1_hi, d_mid));

        let cross0 = xor3_u64(p0_mm, p0_ll, p0_hh);
        let cross1 = xor3_u64(p1_mm, p1_ll, p1_hh);

        let zero = vdupq_n_u64(0);
        let lo0 = veorq_u64(p0_ll, vextq_u64::<1>(zero, cross0));
        let hi0 = veorq_u64(p0_hh, vextq_u64::<1>(cross0, zero));
        let lo1 = veorq_u64(p1_ll, vextq_u64::<1>(zero, cross1));
        let hi1 = veorq_u64(p1_hh, vextq_u64::<1>(cross1, zero));
        [reduce_pmred(lo0,hi0), reduce_pmred(lo1,hi1)]
    }
}

#[cfg(test)]
#[test]
fn pmred_matches_shift_reduction_for_wide_values() {
    let mut x = 0x9e3779b97f4a7c15u64;
    let mut next = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
    for i in 0..16388 {
        let words = match i {
            0 => [0, 0, 0, 0],
            1 => [u64::MAX; 4],
            2 => [0, 0, 0, 1 << 63],
            3 => [u64::MAX, 0, u64::MAX, 0],
            _ => [next(), next(), next(), next()],
        };
        unsafe {
            let lo = vcombine_u64(vcreate_u64(words[0]), vcreate_u64(words[1]));
            let hi = vcombine_u64(vcreate_u64(words[2]), vcreate_u64(words[3]));
            let got: [u64; 2] = transmute(reduce_pmred(lo, hi));
            let expected: [u64; 2] = transmute(reduce_wide(WideNeon { lo, hi }));
            assert_eq!(got, expected, "wide reduction case {i}");
        }
    }
}

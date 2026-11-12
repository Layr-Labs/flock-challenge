use crate::field::F128;
// Used by the generic-AArch64 (no SHA3) `ProdAcc` twin and cfg-gated paths.
#[allow(unused_imports)]
use crate::field::F256Unreduced;

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct WideNeon {
    lo: core::arch::aarch64::uint64x2_t,
    hi: core::arch::aarch64::uint64x2_t,
}

/// Unreduced product sum kept as the three 128-bit partial products of
/// [`clmul_products_q`]. Combining them into a [`WideNeon`] costs two EXT and
/// two EOR, so an accumulator that only ever XORs products in defers that to
/// one combine per chunk instead of one per product.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct TriNeon {
    ll: core::arch::aarch64::uint64x2_t,
    cross: core::arch::aarch64::uint64x2_t,
    hh: core::arch::aarch64::uint64x2_t,
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reduce_tri_q(value: TriNeon) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        reduce_wide_q(WideNeon {
            lo: veorq_u64(value.ll, vextq_u64::<1>(zero, value.cross)),
            hi: veorq_u64(value.hh, vextq_u64::<1>(value.cross, zero)),
        })
    }
}

// The SHA3 extension includes EOR3; retain the two-EOR form for generic
// AArch64 builds that do not enable it.
#[cfg(target_feature = "sha3")]
#[inline(always)]
unsafe fn xor3_u64(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    c: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe { core::arch::aarch64::veor3q_u64(a, b, c) }
}

#[cfg(not(target_feature = "sha3"))]
#[inline(always)]
unsafe fn xor3_u64(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    c: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe { core::arch::aarch64::veorq_u64(a, core::arch::aarch64::veorq_u64(b, c)) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn pmull_lane(a: u64, b: u64) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe { core::mem::transmute::<u128, uint64x2_t>(vmull_p64(a, b)) }
}

/// `x.lo*y.lo ^ x.hi*y.hi` (the cross term of a schoolbook product when one
/// operand is half-swapped) as `PMULL` + one fused `PMULL2`+`EOR` µop: two µops
/// and three instructions instead of three µops (PMULL, PMULL2, EOR).
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn cross_sum_q(
    x: core::arch::aarch64::uint64x2_t,
    y: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    // SAFETY: register-only carry-less multiplies and XOR; `p` and `o` are
    // early clobbers so neither overwrites `x` or `y` before their last read.
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

#[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn cross_sum_q(
    x: core::arch::aarch64::uint64x2_t,
    y: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        veorq_u64(
            pmull_lane(vgetq_lane_u64::<0>(x), vgetq_lane_u64::<0>(y)),
            pmull_lane(vgetq_lane_u64::<1>(x), vgetq_lane_u64::<1>(y)),
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
/// The three 128-bit partial products of `a * b`: `(lo*lo, lo*hi + hi*lo,
/// hi*hi)`. Schoolbook, not Karatsuba: the fourth PMULL is cheaper on Apple
/// Silicon than the two operand sums, the EXT each needs and the three-way
/// XOR that Karatsuba spends to avoid it, and the vector pipes, not the PMULL
/// units, are what the fused sumcheck kernels saturate.
unsafe fn clmul_products_q(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let p0 = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let p2 = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
        let b_swap = vextq_u64::<1>(b, b);
        (p0, cross_sum_q(a, b_swap), p2)
    }
}

/// GHASH multiply with both operands and the result kept in q registers.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_q(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let (t0, mut t1, t2) = clmul_products_q(a, b);

        t1 = xor3_u64(
            t1,
            vextq_u64::<1>(zero, t2),
            pmull_lane(vgetq_lane_u64::<1>(t2), 0x87),
        );

        xor3_u64(
            t0,
            vextq_u64::<1>(zero, t1),
            pmull_lane(vgetq_lane_u64::<1>(t1), 0x87),
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_unreduced_q(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> WideNeon {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let (ll, cross, hh) = clmul_products_q(a, b);
        WideNeon {
            lo: veorq_u64(ll, vextq_u64::<1>(zero, cross)),
            hi: veorq_u64(hh, vextq_u64::<1>(cross, zero)),
        }
    }
}

/// `[v.hi, v.lo]` as one `EXT`. Kept opaque so LLVM cannot fold it back into
/// the two lane `DUP`s it otherwise emits for every product's cross term.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn swap_halves_q(v: core::arch::aarch64::uint64x2_t) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
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

/// [`clmul_products_q`] with `a_sw = swap_halves_q(a)` supplied by the caller:
/// both cross products then read matching lanes (`PMULL`/`PMULL2` of `a_sw`
/// and `b`), so the swap is one `EXT` that a shared operand can reuse.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul_products_sw_q(
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let p0 = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let p2 = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
        let p10 = pmull_lane(vgetq_lane_u64::<0>(a_sw), vgetq_lane_u64::<0>(b));
        let p01 = pmull_lane(vgetq_lane_u64::<1>(a_sw), vgetq_lane_u64::<1>(b));
        (p0, p01, p10, p2)
    }
}

// ---------------------------------------------------------------------------
// PMULL+EOR macro-fused accumulation.
//
// Apple P-cores issue `pmull A, B, C ; eor A, A, D` (also `pmull2`, and the
// `eor A, D, A` operand order) in ONE SIMD issue slot when the two
// instructions are adjacent (still two pipeline µops); `EOR3` never fuses
// with PMULL. In the comments below, "µop" counts SIMD issue slots. Measured on an M4 P-core with a
// cycle-calibrated microbenchmark: a fused pair issues at 4/cycle (0.25 cyc),
// an unfused pair at 0.50 cyc, the fused pair's latency from the accumulator
// input is 3 cycles, and vector `mov` is eliminated at rename. So an
// accumulating product costs 1 µop when the PMULL writes a fresh register and
// the adjacent EOR folds the old accumulator into it. The new accumulator
// value lives in that fresh register; the old one dies, so register pressure
// is unchanged and any loop-carried copy is a free rename.
//
// XOR is associative and commutative, so every accumulator holds exactly the
// same bits as the EOR/EOR3 forms.
// ---------------------------------------------------------------------------

/// `acc ^ (a.lo * b.lo)` as one fused µop.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn fused_pmull_lo_acc(
    acc: core::arch::aarch64::uint64x2_t,
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    // SAFETY: register-only carry-less multiply and XOR; `out` is an early
    // clobber so the PMULL never overwrites `acc`, `a` or `b` before use.
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

/// `acc ^ (a.hi * b.hi)` as one fused µop.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn fused_pmull_hi_acc(
    acc: core::arch::aarch64::uint64x2_t,
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    // SAFETY: as `fused_pmull_lo_acc`.
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

/// `acc ^ (a_sw.lo * b.lo) ^ (a_sw.hi * b.hi)` as two chained fused µops
/// (the cross term of `a * b` when `a_sw = swap_halves_q(a)`).
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn fused_cross_acc(
    acc: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    // SAFETY: register-only; `p` and `o` are early clobbers.
    unsafe {
        core::arch::asm!(
            "pmull {p:v}.1q, {t:v}.1d, {b:v}.1d",
            "eor {p:v}.16b, {p:v}.16b, {c:v}.16b",
            "pmull2 {o:v}.1q, {t:v}.2d, {b:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {p:v}.16b",
            t = in(vreg) a_sw,
            b = in(vreg) b,
            c = in(vreg) acc,
            p = out(vreg) _,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// `acc += a*b` in [`TriNeon`] form; the two cross products enter the cross
/// accumulator through one `EOR3`.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn tri_acc_sw(
    acc: &mut TriNeon,
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) {
    // Fused form: four µops (four fused PMULL+EOR pairs).
    unsafe {
        acc.ll = fused_pmull_lo_acc(acc.ll, a, b);
        acc.hh = fused_pmull_hi_acc(acc.hh, a, b);
        acc.cross = fused_cross_acc(acc.cross, a_sw, b);
    }
}

/// `acc += a*b` in [`TriNeon`] form; the two cross products enter the cross
/// accumulator through one `EOR3`.
#[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn tri_acc_sw(
    acc: &mut TriNeon,
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let (p0, p01, p10, p2) = clmul_products_sw_q(a, a_sw, b);
        acc.ll = veorq_u64(acc.ll, p0);
        acc.cross = xor3_u64(acc.cross, p01, p10);
        acc.hh = veorq_u64(acc.hh, p2);
    }
}

/// `acc += a*b` in [`TriNeon`] form as one fixed sequence: one `EXT` for the
/// swapped operand, four `PMULL`s and three accumulating XORs with only two
/// scratch registers, so the scheduler cannot widen the live set around the
/// spilled accumulators.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn tri_acc_q(
    acc: &mut TriNeon,
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) {
    // Fused form: `ext` + four fused PMULL+EOR pairs = 5 µops per product
    // (was 8: ext, 4 PMULL, 2 EOR, 1 EOR3 with nothing fused). Each
    // accumulator is touched once per call, so the 3-cycle fused latency
    // (6 on the cross chain) is far below the per-group issue time.
    unsafe {
        let a_sw = swap_halves_q(a);
        acc.ll = fused_pmull_lo_acc(acc.ll, a, b);
        acc.hh = fused_pmull_hi_acc(acc.hh, a, b);
        acc.cross = fused_cross_acc(acc.cross, a_sw, b);
    }
}

/// `acc += a*b` in [`TriNeon`] form as kaankolcu's single fixed sequence: one
/// `EXT`, four `PMULL`s and three accumulating XORs (`EOR3` for the cross
/// term) with only two scratch registers. Not fused, but its register
/// footprint is minimal, which the cascade K pass needs: it holds eight
/// [`TriNeon`] accumulators plus its table rows, and the fused form's fresh
/// destinations measured 25% slower there.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn tri_acc_q_compact(
    acc: &mut TriNeon,
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) {
    // SAFETY: register-only carry-less products and XORs (AES and SHA3 are
    // enabled for this build); no memory, stack or flags.
    unsafe {
        core::arch::asm!(
            "ext {t:v}.16b, {a:v}.16b, {a:v}.16b, #8",
            "pmull {p:v}.1q, {a:v}.1d, {b:v}.1d",
            "eor {ll:v}.16b, {ll:v}.16b, {p:v}.16b",
            "pmull2 {p:v}.1q, {a:v}.2d, {b:v}.2d",
            "eor {hh:v}.16b, {hh:v}.16b, {p:v}.16b",
            "pmull {p:v}.1q, {t:v}.1d, {b:v}.1d",
            "pmull2 {t:v}.1q, {t:v}.2d, {b:v}.2d",
            "eor3 {cr:v}.16b, {cr:v}.16b, {p:v}.16b, {t:v}.16b",
            a = in(vreg) a,
            b = in(vreg) b,
            t = out(vreg) _,
            p = out(vreg) _,
            ll = inout(vreg) acc.ll,
            cr = inout(vreg) acc.cross,
            hh = inout(vreg) acc.hh,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
}

#[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
#[inline(always)]
unsafe fn tri_acc_q_compact(
    acc: &mut TriNeon,
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) {
    unsafe { tri_acc_sw(acc, a, swap_halves_q(a), b) }
}

#[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
#[inline(always)]
unsafe fn tri_acc_q(
    acc: &mut TriNeon,
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) {
    unsafe { tri_acc_sw(acc, a, swap_halves_q(a), b) }
}

/// [`mul_q`] with the caller's `a_sw = swap_halves_q(a)`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_sw_q(
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let t0 = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let t2 = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
        let t1 = xor3_u64(
            cross_sum_q(a_sw, b),
            vextq_u64::<1>(zero, t2),
            pmull_lane(vgetq_lane_u64::<1>(t2), 0x87),
        );
        xor3_u64(
            t0,
            vextq_u64::<1>(zero, t1),
            pmull_lane(vgetq_lane_u64::<1>(t1), 0x87),
        )
    }
}

/// [`mul_unreduced_q`] with the caller's `a_sw = swap_halves_q(a)`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_unreduced_sw_q(
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> WideNeon {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let ll = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let hh = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
        let cross = cross_sum_q(a_sw, b);
        WideNeon {
            lo: veorq_u64(ll, vextq_u64::<1>(zero, cross)),
            hi: veorq_u64(hh, vextq_u64::<1>(cross, zero)),
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn wide_xor(acc: &mut WideNeon, value: WideNeon) {
    use core::arch::aarch64::*;
    unsafe {
        acc.lo = veorq_u64(acc.lo, value.lo);
        acc.hi = veorq_u64(acc.hi, value.hi);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reduce_wide_q(value: WideNeon) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let r3 = vextq_u64::<1>(value.hi, zero);
        let ov = xor3_u64(
            vshrq_n_u64::<63>(r3),
            vshrq_n_u64::<62>(r3),
            vshrq_n_u64::<57>(r3),
        );
        let hi = veorq_u64(value.hi, ov);
        let shift1 = veorq_u64(
            vshlq_n_u64::<1>(hi),
            vextq_u64::<1>(zero, vshrq_n_u64::<63>(hi)),
        );
        let shift2 = veorq_u64(
            vshlq_n_u64::<2>(hi),
            vextq_u64::<1>(zero, vshrq_n_u64::<62>(hi)),
        );
        let shift7 = veorq_u64(
            vshlq_n_u64::<7>(hi),
            vextq_u64::<1>(zero, vshrq_n_u64::<57>(hi)),
        );
        let folded = xor3_u64(hi, shift1, veorq_u64(shift2, shift7));
        veorq_u64(value.lo, folded)
    }
}

/// Compose two adjacent four-value folds with one final reduction per output.
///
/// For each group `(v0, v1, v2, v3)`, the sequential fold is expanded as
///
/// `v0 + ra*(v0+v1) + rb*(v0+v2) + (ra*rb)*(v0+v1+v2+v3)`.
///
/// The three products stay unreduced until their XOR sum is complete. Pairing
/// adjacent outputs shares constant setup and interleaves the two independent
/// recursive-Binius reductions without using the rejected shift reducer.
///
/// # Safety
/// `values` must point to eight initialized, readable `F128` values. The AES
/// target feature must be available.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn fold4_pair_expanded_binius_q(
    values: *const F128,
    rho_a: core::arch::aarch64::uint64x2_t,
    rho_b: core::arch::aarch64::uint64x2_t,
    rho_ab: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe { fold4_pair_expanded_tri_q(values, &Fold4RhoQ::new(rho_a, rho_b, rho_ab)) }
}

/// The three fold constants of [`fold4_pair_expanded_tri_q`] with their
/// half-swapped copies, built once per pass so every constant product reads
/// matching lanes (`PMULL`/`PMULL2`) and needs no lane move.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[derive(Clone, Copy)]
struct Fold4RhoQ {
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    b_sw: core::arch::aarch64::uint64x2_t,
    ab: core::arch::aarch64::uint64x2_t,
    ab_sw: core::arch::aarch64::uint64x2_t,
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
impl Fold4RhoQ {
    #[inline(always)]
    unsafe fn new(
        a: core::arch::aarch64::uint64x2_t,
        b: core::arch::aarch64::uint64x2_t,
        ab: core::arch::aarch64::uint64x2_t,
    ) -> Self {
        unsafe {
            Self {
                a,
                a_sw: swap_halves_q(a),
                b,
                b_sw: swap_halves_q(b),
                ab,
                ab_sw: swap_halves_q(ab),
            }
        }
    }
}

/// Recursive-Binius reduction of the [`TriNeon`] sum `(ll, cross, hh)` plus
/// `extra`, without first forming the wide `(lo, hi)` pair. With
/// `lo = ll + [0, cross.lo]` and `hi = hh + [cross.hi, 0]`, the reducer's
/// middle word `[lo.hi, hi.lo]` is `ext(ll, hh) + cross`, `hi.hi = hh.hi` and
/// `lo.lo = ll.lo`, so the result equals the two-stage recursive Binius
/// reduction of the wide sum (the default `F128` multiplier's), XOR `extra`.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn reduce_tri_binius_xor_q(
    ll: core::arch::aarch64::uint64x2_t,
    cross: core::arch::aarch64::uint64x2_t,
    hh: core::arch::aarch64::uint64x2_t,
    extra: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let t1 = xor3_u64(
            vextq_u64::<1>(ll, hh),
            cross,
            pmull_lane(vgetq_lane_u64::<1>(hh), 0x87),
        );
        xor3_u64(
            vzip1q_u64(ll, t1),
            pmull_lane(vgetq_lane_u64::<1>(t1), 0x87),
            extra,
        )
    }
}

/// One expanded fold4 output in [`TriNeon`] form: the three constant products
/// (four matching-lane PMULLs each against the hoisted swapped constants) XOR
/// into one `(ll, cross, hh)` sum, followed by one tri-form reduction.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn fold4_expanded_tri_one_q(
    v0: core::arch::aarch64::uint64x2_t,
    v1: core::arch::aarch64::uint64x2_t,
    v2: core::arch::aarch64::uint64x2_t,
    v3: core::arch::aarch64::uint64x2_t,
    rho: &Fold4RhoQ,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let delta_a = veorq_u64(v0, v1);
        let delta_b = veorq_u64(v0, v2);
        let delta_ab = xor3_u64(v0, v1, veorq_u64(v2, v3));
        let (ll_a, ca0, ca1, hh_a) = clmul_products_sw_q(rho.a, rho.a_sw, delta_a);
        let (ll_b, cb0, cb1, hh_b) = clmul_products_sw_q(rho.b, rho.b_sw, delta_b);
        let (ll_ab, cab0, cab1, hh_ab) = clmul_products_sw_q(rho.ab, rho.ab_sw, delta_ab);
        let ll = xor3_u64(ll_a, ll_b, ll_ab);
        let hh = xor3_u64(hh_a, hh_b, hh_ab);
        let cross = veorq_u64(xor3_u64(xor3_u64(ca0, ca1, cb0), cb1, cab0), cab1);
        reduce_tri_binius_xor_q(ll, cross, hh, v0)
    }
}

/// [`fold4_pair_expanded_binius_q`] with the swapped constants supplied by
/// the caller. Same products, same field values; only the XOR association and
/// the reduction's word bookkeeping differ.
///
/// # Safety
/// `values` must point to eight initialized, readable `F128` values. The AES
/// target feature must be available.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn fold4_pair_expanded_tri_q(
    values: *const F128,
    rho: &Fold4RhoQ,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let v0 = vld1q_u64(values.add(0).cast::<u64>());
        let v1 = vld1q_u64(values.add(1).cast::<u64>());
        let v2 = vld1q_u64(values.add(2).cast::<u64>());
        let v3 = vld1q_u64(values.add(3).cast::<u64>());
        let v4 = vld1q_u64(values.add(4).cast::<u64>());
        let v5 = vld1q_u64(values.add(5).cast::<u64>());
        let v6 = vld1q_u64(values.add(6).cast::<u64>());
        let v7 = vld1q_u64(values.add(7).cast::<u64>());
        (
            fold4_expanded_tri_one_q(v0, v1, v2, v3, rho),
            fold4_expanded_tri_one_q(v4, v5, v6, v7, rho),
        )
    }
}

/// `a * b` in the field with the caller's `a_sw = swap_halves_q(a)`, reduced
/// in tri form (see [`reduce_tri_binius_xor_q`]).
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn mul_sw_tri_binius_q(
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let (ll, p01, p10, hh) = clmul_products_sw_q(a, a_sw, b);
        let t1 = xor3_u64(
            vextq_u64::<1>(ll, hh),
            veorq_u64(p01, p10),
            pmull_lane(vgetq_lane_u64::<1>(hh), 0x87),
        );
        veorq_u64(
            vzip1q_u64(ll, t1),
            pmull_lane(vgetq_lane_u64::<1>(t1), 0x87),
        )
    }
}

/// [`mul_sw_tri_binius_q`] with Karatsuba cross terms: the two square products
/// plus the diagonal product (three `PMULL`s) instead of four schoolbook,
/// trading one `PMULL` for two lane `DUP`s, one scalar `EOR` and the recombine
/// `EOR3`. Bit-identical: `mid ^ ll ^ hh == p01 ^ p10` as values over GF(2),
/// so the tri words match exactly. Used only for the four shared-`wt` products
/// in [`LookaheadTriAcc::add_group`], where the caller's hoisted lane sum
/// covers the repeated operand; every other product keeps the schoolbook form.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn mul_sw_tri_binius_karatsuba_q(
    a: core::arch::aarch64::uint64x2_t,
    a_sum: u64,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let ll = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let hh = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
        let b_sum = vgetq_lane_u64::<0>(b) ^ vgetq_lane_u64::<1>(b);
        let mid = pmull_lane(a_sum, b_sum);
        let cross = xor3_u64(mid, ll, hh);
        let t1 = xor3_u64(
            vextq_u64::<1>(ll, hh),
            cross,
            pmull_lane(vgetq_lane_u64::<1>(hh), 0x87),
        );
        veorq_u64(
            vzip1q_u64(ll, t1),
            pmull_lane(vgetq_lane_u64::<1>(t1), 0x87),
        )
    }
}

/// The eight lookahead message accumulators in [`TriNeon`] form, in the slot
/// order of the incumbent `F256Unreduced` array.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[derive(Clone, Copy)]
struct LookaheadTriAcc([TriNeon; 8]);

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
impl LookaheadTriAcc {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe {
            let zero = core::arch::aarch64::vdupq_n_u64(0);
            Self(
                [TriNeon {
                    ll: zero,
                    cross: zero,
                    hh: zero,
                }; 8],
            )
        }
    }

    /// One group's message contribution: the incumbent algebra
    /// `wt*a_i`, then the eight unreduced products, with the four weight
    /// products sharing one swap of `wt` and every product accumulated in
    /// tri form.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn add_group(
        &mut self,
        wt: core::arch::aarch64::uint64x2_t,
        a0: core::arch::aarch64::uint64x2_t,
        a1: core::arch::aarch64::uint64x2_t,
        a2: core::arch::aarch64::uint64x2_t,
        a3: core::arch::aarch64::uint64x2_t,
        b0: core::arch::aarch64::uint64x2_t,
        b1: core::arch::aarch64::uint64x2_t,
        b2: core::arch::aarch64::uint64x2_t,
        b3: core::arch::aarch64::uint64x2_t,
    ) {
        unsafe { self.add_group_impl::<true>(wt, a0, a1, a2, a3, b0, b1, b2, b3) }
    }

    /// [`Self::add_group`] through [`tri_acc_q_compact`] (cascade K pass).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn add_group_compact(
        &mut self,
        wt: core::arch::aarch64::uint64x2_t,
        a0: core::arch::aarch64::uint64x2_t,
        a1: core::arch::aarch64::uint64x2_t,
        a2: core::arch::aarch64::uint64x2_t,
        a3: core::arch::aarch64::uint64x2_t,
        b0: core::arch::aarch64::uint64x2_t,
        b1: core::arch::aarch64::uint64x2_t,
        b2: core::arch::aarch64::uint64x2_t,
        b3: core::arch::aarch64::uint64x2_t,
    ) {
        unsafe { self.add_group_impl::<false>(wt, a0, a1, a2, a3, b0, b1, b2, b3) }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn add_group_impl<const FUSED: bool>(
        &mut self,
        wt: core::arch::aarch64::uint64x2_t,
        a0: core::arch::aarch64::uint64x2_t,
        a1: core::arch::aarch64::uint64x2_t,
        a2: core::arch::aarch64::uint64x2_t,
        a3: core::arch::aarch64::uint64x2_t,
        b0: core::arch::aarch64::uint64x2_t,
        b1: core::arch::aarch64::uint64x2_t,
        b2: core::arch::aarch64::uint64x2_t,
        b3: core::arch::aarch64::uint64x2_t,
    ) {
        use core::arch::aarch64::*;
        let acc_one = |t: &mut TriNeon, x: uint64x2_t, y: uint64x2_t| {
            // SAFETY: register-only asm; the AES/SHA3 features hold for this build.
            unsafe {
                if FUSED {
                    tri_acc_q(t, x, y)
                } else {
                    tri_acc_q_compact(t, x, y)
                }
            }
        };
        unsafe {
            let wt_sum = vgetq_lane_u64::<0>(wt) ^ vgetq_lane_u64::<1>(wt);
            let a0w = mul_sw_tri_binius_karatsuba_q(wt, wt_sum, a0);
            let a1w = mul_sw_tri_binius_karatsuba_q(wt, wt_sum, a1);
            let a2w = mul_sw_tri_binius_karatsuba_q(wt, wt_sum, a2);
            let a3w = mul_sw_tri_binius_karatsuba_q(wt, wt_sum, a3);
            let acc = &mut self.0;
            acc_one(&mut acc[0], a1w, b1);
            acc_one(&mut acc[1], veorq_u64(a0w, a1w), veorq_u64(b0, b1));
            acc_one(&mut acc[2], a3w, b3);
            acc_one(&mut acc[3], veorq_u64(a2w, a3w), veorq_u64(b2, b3));
            acc_one(&mut acc[4], a2w, b2);
            let (e_aw, e_b) = (veorq_u64(a0w, a2w), veorq_u64(b0, b2));
            let (o_aw, o_b) = (veorq_u64(a1w, a3w), veorq_u64(b1, b3));
            acc_one(&mut acc[5], e_aw, e_b);
            acc_one(&mut acc[6], o_aw, o_b);
            acc_one(&mut acc[7], veorq_u64(e_aw, o_aw), veorq_u64(e_b, o_b));
        }
    }

    #[inline(always)]
    unsafe fn reduce(self) -> [F128; 8] {
        unsafe {
            self.0
                .map(|acc| core::mem::transmute::<core::arch::aarch64::uint64x2_t, F128>(reduce_tri_q(acc)))
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fold_row_q(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        let entry = |chunk: usize| {
            table_data.add(
                chunk * STRIDE + (*bytes_ptr.add(chunk)) as usize * core::mem::size_of::<F128>(),
            )
        };
        let a = xor3_u64(
            vreinterpretq_u64_u8(vld1q_u8(entry(0))),
            vreinterpretq_u64_u8(vld1q_u8(entry(1))),
            vreinterpretq_u64_u8(vld1q_u8(entry(2))),
        );
        let b = xor3_u64(
            vreinterpretq_u64_u8(vld1q_u8(entry(3))),
            vreinterpretq_u64_u8(vld1q_u8(entry(4))),
            vreinterpretq_u64_u8(vld1q_u8(entry(5))),
        );
        let c = veorq_u64(
            vreinterpretq_u64_u8(vld1q_u8(entry(6))),
            vreinterpretq_u64_u8(vld1q_u8(entry(7))),
        );
        xor3_u64(a, b, c)
    }
}

/// Build-time choice of the instruction pair behind [`bank_load_q`]. Every
/// form reads the same 16 bytes; only the address arithmetic differs.
///
/// * `1` (default): the byte index is produced already multiplied by 16
///   (`and idx, mask, row, lsr #(SHIFT - 4)` with `mask = 0xff0`, or `ubfiz
///   idx, row, #4, #8` for byte 0) and the load uses a plain register offset,
///   `ldr q, [bank, idx]`.
/// * `0` (`FLOCK_Q_LOAD_FORM=0` in the build environment): the parent's pair,
///   `ubfx idx, row, #SHIFT, #8` and `ldr q, [bank, idx, lsl #4]`.
/// * `2` (`FLOCK_Q_LOAD_FORM=2`): an alternative kept for measurement on
///   other cores. The row is masked once per lane parity (`row &
///   0x00ff00ff00ff00ff` and `row & 0xff00ff00ff00ff00`), then `ubfx idx,
///   masked, #(SHIFT - 4), #12` and the plain register-offset load. It
///   needs no mask register and no two-issue `and`, at two extra
///   instructions per row. On an M4 it measured slightly behind form 1.
///
/// Forms 0 and 1 are two instructions per lookup. The public Firestorm
/// (Apple M1 performance core) instruction tables list the 128-bit load
/// with a shifted register offset as two uops (one integer, one load) and
/// the same load with an unshifted register offset as one, so on such a core
/// form 0 takes three pipeline slots per lookup and form 1 takes two. An M4
/// performance core measures the same way (`q_load_form_uop_probe`).
const Q_LOAD_FORM: u8 = match option_env!("FLOCK_Q_LOAD_FORM") {
    Some(v) => {
        let v = v.as_bytes();
        if v.len() == 1 && v[0] == b'0' {
            0
        } else if v.len() == 1 && v[0] == b'2' {
            2
        } else {
            1
        }
    }
    None => 1,
};

/// `0xff << 4`: selects one table byte index that is already multiplied by
/// the 16-byte entry size.
const Q_INDEX_MASK: u64 = 0xff0;
/// Byte lanes 0, 2, 4, 6 and 1, 3, 5, 7 of a packed row (form 2 only).
const Q_EVEN_LANES: u64 = 0x00ff_00ff_00ff_00ff;
const Q_ODD_LANES: u64 = 0xff00_ff00_ff00_ff00;

/// Entry `(row >> SHIFT) & 0xff` of the 256 x 16-byte table bank at `bank`.
///
/// Two instructions in every form selected by [`Q_LOAD_FORM`]. From the plain
/// expression LLVM folds the `<< 4` into the mask, `(row >> (SHIFT-4)) &
/// 0xff0`, as a separate shift and mask: a third integer op per lookup in the
/// issue-bound round-two fold. The default form computes that same value in
/// one instruction (`and` with a shifted register operand against a mask held
/// in a register) and so keeps the register-offset load unscaled.
///
/// # Safety
/// `bank` must point to 256 readable 16-byte entries (4 KiB).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bank_load_q<const SHIFT: u32>(
    bank: *const u8,
    row: u64,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    // SAFETY: the byte index is at most 255 (as a byte offset: at most 4080,
    // a multiple of 16), so the 16-byte load stays inside the caller's 4 KiB
    // bank; the asm only reads that memory.
    unsafe {
        if Q_LOAD_FORM == 0 {
            core::arch::asm!(
                "ubfx {idx}, {row}, #{shift}, #8",
                "ldr {out:q}, [{bank}, {idx}, lsl #4]",
                shift = const SHIFT,
                row = in(reg) row,
                bank = in(reg) bank,
                idx = out(reg) _,
                out = lateout(vreg) out,
                options(pure, readonly, nostack, preserves_flags),
            );
        } else if SHIFT == 0 {
            // idx = (row & 0xff) << 4
            core::arch::asm!(
                "ubfiz {idx}, {row}, #4, #8",
                "ldr {out:q}, [{bank}, {idx}]",
                row = in(reg) row,
                bank = in(reg) bank,
                idx = out(reg) _,
                out = lateout(vreg) out,
                options(pure, readonly, nostack, preserves_flags),
            );
        } else if Q_LOAD_FORM == 2 {
            // With the neighbouring byte lanes cleared, the 12-bit field that
            // starts four bits below the byte is the byte times 16.
            let lanes = if SHIFT & 8 == 0 {
                Q_EVEN_LANES
            } else {
                Q_ODD_LANES
            };
            core::arch::asm!(
                "ubfx {idx}, {row}, #{shift}, #{width}",
                "ldr {out:q}, [{bank}, {idx}]",
                shift = const (SHIFT + 60) % 64,
                width = const if SHIFT == 0 { 4 } else { 12 },
                row = in(reg) row & lanes,
                bank = in(reg) bank,
                idx = out(reg) _,
                out = lateout(vreg) out,
                options(pure, readonly, nostack, preserves_flags),
            );
        } else {
            // idx = (row >> (SHIFT - 4)) & 0xff0 = ((row >> SHIFT) & 0xff) << 4
            core::arch::asm!(
                "and {idx}, {mask}, {row}, lsr #{shift}",
                "ldr {out:q}, [{bank}, {idx}]",
                shift = const (SHIFT + 60) % 64,
                mask = in(reg) Q_INDEX_MASK,
                row = in(reg) row,
                bank = in(reg) bank,
                idx = out(reg) _,
                out = lateout(vreg) out,
                options(pure, readonly, nostack, preserves_flags),
            );
        }
    }
    out
}

/// Fold four already-loaded packed rows together. Grouping the independent
/// lookup chains by byte bank exposes four L1 loads at a time, and loading each
/// row as one `u64` lets the compact round-two path reuse it for its raw delta.
///
/// `table_data` must point to the eight 4 KiB byte banks of a fold table.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fold_four_row_codes_q(
    table_data: *const u8,
    row0: u64,
    row1: u64,
    row2: u64,
    row3: u64,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        macro_rules! load {
            ($row:expr, $chunk:literal) => {
                bank_load_q::<{ 8 * $chunk }>(table_data.add($chunk * STRIDE), $row)
            };
        }

        // Retain the four independent row streams while consuming two new
        // table rows per dependent accumulator update.
        let mut acc0 = load!(row0, 0);
        let mut acc1 = load!(row1, 0);
        let mut acc2 = load!(row2, 0);
        let mut acc3 = load!(row3, 0);
        macro_rules! step {
            ($c0:literal, $c1:literal) => {
                acc0 = xor3_u64(acc0, load!(row0, $c0), load!(row0, $c1));
                acc1 = xor3_u64(acc1, load!(row1, $c0), load!(row1, $c1));
                acc2 = xor3_u64(acc2, load!(row2, $c0), load!(row2, $c1));
                acc3 = xor3_u64(acc3, load!(row3, $c0), load!(row3, $c1));
            };
        }
        step!(1, 2);
        step!(3, 4);
        step!(5, 6);
        acc0 = veorq_u64(acc0, load!(row0, 7));
        acc1 = veorq_u64(acc1, load!(row1, 7));
        acc2 = veorq_u64(acc2, load!(row2, 7));
        acc3 = veorq_u64(acc3, load!(row3, 7));
        (acc0, acc1, acc2, acc3)
    }
}

/// Two-row variant of [`fold_four_row_codes_q`] for the degenerate-pair path
/// (the two b rows are skipped, so only the a rows need folding).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fold_two_row_codes_q(
    table_data: *const u8,
    row0: u64,
    row1: u64,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        macro_rules! load {
            ($row:expr, $chunk:literal) => {
                bank_load_q::<{ 8 * $chunk }>(table_data.add($chunk * STRIDE), $row)
            };
        }
        let mut acc0 = load!(row0, 0);
        let mut acc1 = load!(row1, 0);
        macro_rules! step {
            ($c0:literal, $c1:literal) => {
                acc0 = xor3_u64(acc0, load!(row0, $c0), load!(row0, $c1));
                acc1 = xor3_u64(acc1, load!(row1, $c0), load!(row1, $c1));
            };
        }
        step!(1, 2);
        step!(3, 4);
        step!(5, 6);
        acc0 = veorq_u64(acc0, load!(row0, 7));
        acc1 = veorq_u64(acc1, load!(row1, 7));
        (acc0, acc1)
    }
}

/// One-row variant of [`fold_two_row_codes_q`].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fold_row_code_q(table_data: *const u8, row: u64) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        macro_rules! load {
            ($chunk:literal) => {
                bank_load_q::<{ 8 * $chunk }>(table_data.add($chunk * STRIDE), row)
            };
        }
        let acc = xor3_u64(load!(0), load!(1), load!(2));
        let acc = xor3_u64(acc, load!(3), load!(4));
        let acc = xor3_u64(acc, load!(5), load!(6));
        veorq_u64(acc, load!(7))
    }
}

/// Returns `true` iff a 128-bit vector is all-zero.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn is_zero_q(v: core::arch::aarch64::uint64x2_t) -> bool {
    use core::arch::aarch64::*;
    unsafe { vmaxvq_u32(vreinterpretq_u32_u64(v)) == 0 }
}

/// Complete q-register-native round-two worker chunk. Four folded rows stay in
/// vector registers through output stores, reduced GHASH products, and the
/// eq-weighted unreduced accumulators; only the two final chunk sums cross back
/// to scalar `F128`. This removes the per-pair fmov/mov.d/dup round trips from
/// the original inlined scalar-struct path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_round2_chunk_neon_unchecked_8(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_pair_nt(dst: *mut F128, x: uint64x2_t, y: uint64x2_t) {
        // Round two emits two 1 GiB F128 tables that are not consumed until
        // the producer's Rayon barrier. They cannot remain cache-resident, so
        // ordinary stores only add write-allocate traffic and evict the 32 KiB
        // fold table. `stnp` is the same best-effort non-temporal pair hint
        // used by the batch-major witness producer.
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
        let mut p1_acc = WideNeon { lo: zero, hi: zero };
        let mut pinf_acc = WideNeon { lo: zero, hi: zero };

        for x_lo in 0..lo_size {
            let out = 2 * x_lo;
            if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                store_pair_nt(a_out.add(out), zero, zero);
                store_pair_nt(b_out.add(out), zero, zero);
                continue;
            }

            let row0 = 2 * x_lo;
            let row1 = row0 + 1;
            let a0 = fold_row_q(table_data, a_packed.add(row0 * 8));
            let b0 = fold_row_q(table_data, b_packed.add(row0 * 8));
            let a1 = fold_row_q(table_data, a_packed.add(row1 * 8));
            let b1 = fold_row_q(table_data, b_packed.add(row1 * 8));

            store_pair_nt(a_out.add(out), a0, a1);
            store_pair_nt(b_out.add(out), b0, b1);

            let g1 = mul_q(a1, b1);
            let g_inf = mul_q(veorq_u64(a0, a1), veorq_u64(b0, b1));
            let eq_l = vld1q_u64(eq_lo.add(x_lo).cast::<u64>());
            wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
            wide_xor(&mut pinf_acc, mul_unreduced_q(eq_l, g_inf));
        }

        (
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1_acc)),
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf_acc)),
        )
    }
}

/// Compact round-two worker: retain only the even-row folded anchor and the
/// packed adjacent-row delta for each pair.  The odd folded row is still kept
/// in registers long enough to form the round message, but is not written.
///
/// The later challenge-specialized fold reconstructs
/// `anchor + rho * fold(delta)` from this 24-byte-per-polynomial
/// representation, instead of materializing both 16-byte folded rows.
///
/// `degen` enables the b≡1 chunk-class degeneration: when both b row codes of
/// a pair are all-ones (statically true for the input/output/const regions of
/// the BLAKE3 block R1CS — 11 of 121 useful pairs per block), both b folds
/// equal `Σ_i L_i(z) = 1`, so the 16 b-table lookups are skipped, the b
/// anchor is the precomputed all-ones fold, `G(∞)`'s `(b0 + b1)` factor
/// vanishes (that mul chain is skipped), and — when the all-ones fold is
/// bit-exactly `F128::ONE` (partition of unity; always in practice) —
/// `a1 · b1` degenerates to `a1`. Value-forced, therefore bit-exact.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_round2_compact_chunk_neon_unchecked_8(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    degen: bool,
) -> (F128, F128) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_anchor_pair_nt(dst: *mut F128, a: uint64x2_t, b: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{dst}]",
                dst = in(reg) dst,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        let zero = vdupq_n_u64(0);
        let mut p1_acc = WideNeon { lo: zero, hi: zero };
        let mut pinf_acc = WideNeon { lo: zero, hi: zero };

        // Fold of the all-ones row code = Σ_i L_i(z) — exactly the XOR the
        // per-pair path would compute, so storing it as the b anchor is
        // bit-identical. Partition of unity makes it F128::ONE; the `a1·b1 →
        // a1` shortcut is additionally gated on that being bit-exact.
        let ones_bytes = [0xFFu8; 8];
        let b_ones = fold_row_q(table_data, ones_bytes.as_ptr());
        let ones_is_one = is_zero_q(veorq_u64(
            b_ones,
            core::mem::transmute::<F128, uint64x2_t>(F128::ONE),
        ));

        for x_lo in 0..lo_size {
            if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                continue;
            }

            let row0 = 2 * x_lo;
            let row1 = row0 + 1;
            let a0_ptr = a_packed.add(row0 * 8);
            let a1_ptr = a_packed.add(row1 * 8);
            let b0_ptr = b_packed.add(row0 * 8);
            let b1_ptr = b_packed.add(row1 * 8);
            let a0_code = u64::from_le(core::ptr::read_unaligned(a0_ptr.cast::<u64>()));
            let a1_code = u64::from_le(core::ptr::read_unaligned(a1_ptr.cast::<u64>()));
            let b0_code = u64::from_le(core::ptr::read_unaligned(b0_ptr.cast::<u64>()));
            let b1_code = u64::from_le(core::ptr::read_unaligned(b1_ptr.cast::<u64>()));

            if degen && (b0_code & b1_code) == u64::MAX {
                // b ≡ 1 pair: b0 = b1 = fold(all-ones) = 1. Skip the 16
                // b-table lookups; `b0 + b1 = 0` zeroes the G(∞) term.
                let (a0, a1) = fold_two_row_codes_q(table_data, a0_code, a1_code);
                store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b_ones);
                let delta_pair =
                    core::mem::transmute::<[u64; 2], uint64x2_t>([a0_code ^ a1_code, 0]);
                vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);

                let g1 = if ones_is_one { a1 } else { mul_q(a1, b_ones) };
                let eq_l = vld1q_u64(eq_lo.add(x_lo).cast::<u64>());
                wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
                continue;
            }

            let (a0, a1, b0, b1) =
                fold_four_row_codes_q(table_data, a0_code, a1_code, b0_code, b1_code);

            store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b0);
            let da = a0_code ^ a1_code;
            let db = b0_code ^ b1_code;
            let delta_pair = core::mem::transmute::<[u64; 2], uint64x2_t>([da, db]);
            vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);

            let g1 = mul_q(a1, b1);
            let g_inf = mul_q(veorq_u64(a0, a1), veorq_u64(b0, b1));
            let eq_l = vld1q_u64(eq_lo.add(x_lo).cast::<u64>());
            wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
            wide_xor(&mut pinf_acc, mul_unreduced_q(eq_l, g_inf));
        }

        (
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1_acc)),
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf_acc)),
        )
    }
}

/// Anchors-and-deltas-only sibling of
/// [`fold_round2_compact_chunk_neon_unchecked_8`]: identical anchor stores,
/// identical delta bytes, identical padded-pair zeroing — but no `a1`/`b1`
/// folds and no message products. Used for the chunks whose products the
/// round-two GPU arm computes concurrently; byte-identical output is
/// guaranteed because every store below is the same expression the fused
/// kernel stores (the degen branch there is a value-preserving shortcut:
/// `fold(all-ones) == fold(b0_code)` when `b0_code` is all-ones, and its
/// delta lanes are `[a0^a1, 0] == [a0^a1, b0^b1]`).
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn fold_round2_compact_chunk_neon_anchors_only_8(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_anchor_pair_nt(dst: *mut F128, a: uint64x2_t, b: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{dst}]",
                dst = in(reg) dst,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        let zero = vdupq_n_u64(0);
        for x_lo in 0..lo_size {
            if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                continue;
            }
            let row0 = 2 * x_lo;
            let row1 = row0 + 1;
            let a0_code = u64::from_le(core::ptr::read_unaligned(
                a_packed.add(row0 * 8).cast::<u64>(),
            ));
            let a1_code = u64::from_le(core::ptr::read_unaligned(
                a_packed.add(row1 * 8).cast::<u64>(),
            ));
            let b0_code = u64::from_le(core::ptr::read_unaligned(
                b_packed.add(row0 * 8).cast::<u64>(),
            ));
            let b1_code = u64::from_le(core::ptr::read_unaligned(
                b_packed.add(row1 * 8).cast::<u64>(),
            ));
            let (a0, b0) = fold_two_row_codes_q(table_data, a0_code, b0_code);
            store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b0);
            let delta_pair = core::mem::transmute::<[u64; 2], uint64x2_t>([
                a0_code ^ a1_code,
                b0_code ^ b1_code,
            ]);
            vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);
        }
    }
}

/// Fold one round-two pair and emit its compact anchor + delta stores.
///
/// Every store below is the *same expression* the incumbent
/// [`fold_round2_compact_chunk_neon_unchecked_8`] emits for the same pair
/// (padded pairs zeroed, degenerate pairs shortcut with the identical
/// `[a0^a1, 0] == [a0^a1, b0^b1]` delta), so the compact state produced by
/// the lookahead kernel is byte-identical to the incumbent's.
///
///
/// Returns the four folded rows plus whether the b≡1 shortcut was taken.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn r2_pair_fold_and_store(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    x_lo: usize,
    padded: bool,
    degen: bool,
    b_ones: core::arch::aarch64::uint64x2_t,
    skip_dead: bool,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    bool,
) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_anchor_pair_nt(dst: *mut F128, a: uint64x2_t, b: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{dst}]",
                dst = in(reg) dst,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        let zero = vdupq_n_u64(0);
        if padded {
            // `skip_dead`: the cascade K pass treats padded pairs as known
            // zeros and never reads their slots, so the zero stores are dead.
            if !skip_dead {
                store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
            }
            return (zero, zero, zero, zero, false);
        }

        let row0 = 2 * x_lo;
        let row1 = row0 + 1;
        let a0_code = u64::from_le(core::ptr::read_unaligned(
            a_packed.add(row0 * 8).cast::<u64>(),
        ));
        let a1_code = u64::from_le(core::ptr::read_unaligned(
            a_packed.add(row1 * 8).cast::<u64>(),
        ));
        let b0_code = u64::from_le(core::ptr::read_unaligned(
            b_packed.add(row0 * 8).cast::<u64>(),
        ));
        let b1_code = u64::from_le(core::ptr::read_unaligned(
            b_packed.add(row1 * 8).cast::<u64>(),
        ));

        if degen && (b0_code & b1_code) == u64::MAX {
            let (a0, a1) = fold_two_row_codes_q(table_data, a0_code, a1_code);
            store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b_ones);
            store_delta_a_only_gpr(deltas.add(x_lo * 16), a0_code, a1_code);
            return (a0, a1, b_ones, b_ones, true);
        }

        let (a0, a1, b0, b1) =
            fold_four_row_codes_q(table_data, a0_code, a1_code, b0_code, b1_code);
        store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b0);
        store_delta_pair_gpr(deltas.add(x_lo * 16), a0_code, a1_code, b0_code, b1_code);
        (a0, a1, b0, b1, false)
    }
}

/// Round-two compact producer extended with the deferred round-three
/// lookahead accumulators (variant K).
///
/// Anchors, deltas, padded-pair zeroing and the round-two products are the
/// same expressions the incumbent kernel emits; the only structural change is
/// that pairs are visited two at a time — one round-three group `y = x'/2` —
/// so a group's four rows are live together and the six deferred round-three
/// aggregates can be formed with **no extra table lookups**.
///
/// **One weight per group, one scaling per row.** The two eq2 lanes of a group
/// and eq3 itself are constant multiples of one another —
/// `eq2(2y) = (1+r1)*eq3(y)` and `eq2(2y+1) = r1*eq3(y)` — so the whole group is
/// accumulated against the single weight `w = eq2_lo[2u+1]`, the four rows are
/// pre-scaled by `w` with **four** reduced multiplies, and all eight products
/// then cost one unreduced multiply each: 44 PMULL per group against the
/// incumbent's 32 for half as many products. The constant rescalings
/// (`kappa = (1+r1)/r1` on the even sums, `r1^-1` on the six deferred
/// aggregates) are applied once by the driver, off the hot path.
///
/// Writes eight F128 slots to `out`, every one of them `w`-weighted:
///
/// | slot | value |
/// |---|---|
/// | 0, 1 | `sum_{x' even} w*a1*b1`, `sum_{x' even} w*(a0+a1)(b0+b1)` |
/// | 2, 3 | the same two sums over **odd** `x'` — i.e. `r1*W1`, `r1*W2` |
/// | 4..8 | `r1*W0`, `r1*W3`, `r1*W4`, `r1*W5` |
///
/// With `FULL = false` slots 0/1 stay zero: the round-two GPU arm owns this
/// chunk's even-parity products, so the even pair's two products are skipped.
///
/// With `ODD_ON_GPU = true` slots 2/3 stay zero as well. The GPU arm computes
/// both parities anyway — it used to XOR them together in the last step of its
/// threadgroup reduction and hand back only the sum, which forced the CPU to
/// recompute the odd half for the lookahead's `W1`/`W2`. Keeping that split
/// costs the GPU nothing, and lets this kernel drop two more of its eight
/// products: **32 PMULL per group instead of 38** on every offloaded chunk.
/// The four `mul_q` row weightings survive because `W0`/`W3`/`W4`/`W5` still
/// need all four scaled rows.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_round2_compact_chunk_neon_lookahead_8<
    const FULL: bool,
    const ODD_ON_GPU: bool,
>(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    degen: bool,
    periodic_padding: bool,
    skip_dead: bool,
    out: *mut F128,
) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_anchor_pair_nt(dst: *mut F128, a: uint64x2_t, b: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{dst}]",
                dst = in(reg) dst,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        let zero = vdupq_n_u64(0);
        let tri_zero = TriNeon {
            ll: zero,
            cross: zero,
            hh: zero,
        };
        let mut p1_even = tri_zero;
        let mut pinf_even = tri_zero;
        let mut p1_odd = tri_zero;
        let mut pinf_odd = tri_zero;
        let mut w0 = tri_zero;
        let mut w3 = tri_zero;
        let mut w4 = tri_zero;
        let mut w5 = tri_zero;

        // Row weighting `w * a`: the cascade K pass's fused PMULL+EOR
        // schoolbook product with `0x87` hoisted into a register, the same
        // field value as `mul_sw_q` in fewer issue slots.
        #[cfg(target_feature = "sha3")]
        let k87 = vdupq_n_u64(0x87);
        macro_rules! wmul {
            ($w:expr, $w_sw:expr, $a:expr) => {{
                #[cfg(target_feature = "sha3")]
                let v = mul_sw_fused_q($w, $w_sw, $a, k87);
                #[cfg(not(target_feature = "sha3"))]
                let v = mul_sw_q($w, $w_sw, $a);
                v
            }};
        }

        let ones_bytes = [0xFFu8; 8];
        let b_ones = fold_row_q(table_data, ones_bytes.as_ptr());
        let ones_is_one = is_zero_q(veorq_u64(
            b_ones,
            core::mem::transmute::<F128, uint64x2_t>(F128::ONE),
        ));

        macro_rules! zero_group {
            ($u:expr) => {{
                if !skip_dead {
                    let x_lo0 = 2 * $u;
                    let x_lo1 = x_lo0 + 1;
                    store_anchor_pair_nt(anchors.add(2 * x_lo0), zero, zero);
                    vst1q_u64(deltas.add(x_lo0 * 16).cast::<u64>(), zero);
                    store_anchor_pair_nt(anchors.add(2 * x_lo1), zero, zero);
                    vst1q_u64(deltas.add(x_lo1 * 16).cast::<u64>(), zero);
                }
            }};
        }

        macro_rules! process_group {
            ($u:expr, $pad0:expr, $pad1:expr) => {{
                let x_lo0 = 2 * $u;
                let x_lo1 = x_lo0 + 1;
                let pad0 = $pad0;
                let pad1 = $pad1;

                let (a0, a1, b0, b1, deg0) = r2_pair_fold_and_store(
                    table_data, a_packed, b_packed, anchors, deltas, x_lo0, pad0, degen, b_ones,
                    skip_dead,
                );
                let (a2, a3, b2, b3, deg1) = r2_pair_fold_and_store(
                    table_data, a_packed, b_packed, anchors, deltas, x_lo1, pad1, degen, b_ones,
                    skip_dead,
                );

                // The odd lane's weight drives the whole group; see the doc above.
                let w = vld1q_u64(eq_lo.add(x_lo1).cast::<u64>());
                let w_sw = swap_halves_q(w);

                if deg0 & deg1 & ones_is_one {
                    // b === 1 across the group: e_b = o_b = 0 kills W3/W4/W5 and
                    // both G(inf) chains, and every surviving product is `w * a_i`.
                    if FULL {
                        tri_acc_sw(&mut p1_even, w, w_sw, a1);
                    }
                    if !ODD_ON_GPU {
                        tri_acc_sw(&mut p1_odd, w, w_sw, a3);
                    }
                    tri_acc_sw(&mut w0, w, w_sw, a2);
                } else {
                    let a0w = wmul!(w, w_sw, a0);
                    let a1w = wmul!(w, w_sw, a1);
                    let (a2w, a3w) = if pad1 {
                        (zero, zero)
                    } else {
                        (wmul!(w, w_sw, a2), wmul!(w, w_sw, a3))
                    };
                    macro_rules! acc {
                        ($acc:expr, $a:expr, $b:expr) => {{
                            // The GPU-shared (`!FULL`) body keeps fewer live
                            // accumulators and measures faster with them updated
                            // in place; the full body keeps the all-fused form.
                            #[cfg(target_feature = "sha3")]
                            if FULL {
                                tri_acc_q(&mut $acc, $a, $b);
                            } else {
                                tri_acc_q_crossfused(&mut $acc, $a, $b);
                            }
                            #[cfg(not(target_feature = "sha3"))]
                            tri_acc_q(&mut $acc, $a, $b);
                        }};
                    }

                    // ---- round-two products, split by the parity of x' ----
                    if FULL {
                        acc!(p1_even, a1w, b1);
                        if !deg0 {
                            acc!(pinf_even, veorq_u64(a0w, a1w), veorq_u64(b0, b1));
                        }
                    }
                    if !pad1 {
                        if !ODD_ON_GPU {
                            acc!(p1_odd, a3w, b3);
                            if !deg1 {
                                acc!(pinf_odd, veorq_u64(a2w, a3w), veorq_u64(b2, b3));
                            }
                        }
                        acc!(w0, a2w, b2);
                    }

                    // ---- deferred round-three aggregates (no extra lookups) ----
                    let e_aw = veorq_u64(a0w, a2w);
                    let o_aw = veorq_u64(a1w, a3w);
                    let e_b = veorq_u64(b0, b2);
                    let o_b = veorq_u64(b1, b3);
                    acc!(w3, e_aw, e_b);
                    acc!(w4, o_aw, o_b);
                    acc!(w5, veorq_u64(e_aw, o_aw), veorq_u64(e_b, o_b));
                }
            }};
        }

        let n_groups = lo_size / 2;
        let ranked_periodic = periodic_padding
            && pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
            && (pair_idx_base & 127) == 0
            && n_groups.is_multiple_of(64);
        if ranked_periodic {
            for block in 0..n_groups / 64 {
                let first = block * 64;
                for u in first..first + 60 {
                    process_group!(u, false, false);
                }
                process_group!(first + 60, false, true);
                for u in first + 61..first + 64 {
                    zero_group!(u);
                }
            }
        } else {
            for u in 0..n_groups {
                let x_lo0 = 2 * u;
                let x_lo1 = x_lo0 + 1;
                let pad0 = ((pair_idx_base + x_lo0) & pair_in_block_mask) >= useful_pairs_inclusive;
                let pad1 = ((pair_idx_base + x_lo1) & pair_in_block_mask) >= useful_pairs_inclusive;
                if pad0 & pad1 {
                    zero_group!(u);
                    continue;
                }
                process_group!(u, pad0, pad1);
            }
        }

        *out.add(0) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(p1_even));
        *out.add(1) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(pinf_even));
        *out.add(2) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(p1_odd));
        *out.add(3) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(pinf_odd));
        *out.add(4) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(w0));
        *out.add(5) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(w3));
        *out.add(6) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(w4));
        *out.add(7) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(w5));
    }
}

#[cfg(target_arch = "aarch64")]
fn r2_fixed_pads_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_OFF_R2_FIXED_PADS").is_none())
}

/// GPU-covered round-two lookahead with the four fixed medium weights
/// deferred until after a 16-position polynomial Horner sum.
///
/// The caller checks that the medium coordinates are the protocol constants.
/// For a 32-group-aligned `first`, `u = first + 2*b + t` has weight
/// `eq_lo[2*u+1] = eq_lo[2*(first+t)+1] * x^b`. Each raw product has degree
/// at most 254, so all 16 positions fit in 270 bits. The four fifth limbs
/// share two q registers; anchors and deltas retain their canonical form.
///
/// `table_data` must be GF(2)-linear in each byte (`fold(x ^ y) = fold(x) +
/// fold(y)`), as [`UniSkipFoldTable::new`] and `scaled_linear` build it: the
/// odd rows are folded only as `a1 ^ a3` and `b1 ^ b3`.
///
/// `SKIP_FIRST_SHIFT` peels the first group of every 16-position window: the
/// ten accumulator registers are still zero there, so the two `shift_pair`
/// calls in front of it shift zeros. The peeled group omits them; every
/// later group is unchanged. Outputs are bit-identical for both values.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_round2_compact_chunk_neon_horner_8<const SKIP_FIRST_SHIFT: bool>(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    degen: bool,
    skip_dead: bool,
    out: *mut F128,
) {
    unsafe {
        if r2_gpr_delta_enabled() {
            fold_round2_compact_chunk_neon_horner_8_impl::<SKIP_FIRST_SHIFT, true>(
                table_data,
                a_packed,
                b_packed,
                anchors,
                deltas,
                eq_lo,
                lo_size,
                pair_idx_base,
                pair_in_block_mask,
                useful_pairs_inclusive,
                degen,
                skip_dead,
                out,
            )
        } else {
            fold_round2_compact_chunk_neon_horner_8_impl::<SKIP_FIRST_SHIFT, false>(
                table_data,
                a_packed,
                b_packed,
                anchors,
                deltas,
                eq_lo,
                lo_size,
                pair_idx_base,
                pair_in_block_mask,
                useful_pairs_inclusive,
                degen,
                skip_dead,
                out,
            )
        }
    }
}

/// Kill switch for the unreduced window sums of the pre-scaled round-two W
/// kernel: `FLOCK_NO_R2_WINDOW_TRI` (any value) restores one reduced product
/// per window. Same four sums either way; read once per process.
#[cfg(target_arch = "aarch64")]
fn r2_window_tri_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_R2_WINDOW_TRI").is_none())
}

#[cfg(target_arch = "aarch64")]
fn r2_gpr_delta_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_OFF_R2_GPR_DELTA").is_none())
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_code_pair_gpr(p: *const u64) -> (u64, u64) {
    let (lo, hi): (u64, u64);
    unsafe {
        core::arch::asm!(
            "ldp {lo}, {hi}, [{p}]",
            p = in(reg) p,
            lo = lateout(reg) lo,
            hi = lateout(reg) hi,
            options(pure, readonly, nostack, preserves_flags),
        );
    }
    (lo, hi)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_delta_pair_gpr(dst: *mut u8, a0: u64, a1: u64, b0: u64, b1: u64) {
    unsafe {
        core::arch::asm!(
            "eor {ta}, {a0}, {a1}",
            "eor {tb}, {b0}, {b1}",
            "stp {ta}, {tb}, [{dst}]",
            dst = in(reg) dst,
            a0 = in(reg) a0,
            a1 = in(reg) a1,
            b0 = in(reg) b0,
            b1 = in(reg) b1,
            ta = out(reg) _,
            tb = out(reg) _,
            options(nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_delta_a_only_gpr(dst: *mut u8, a0: u64, a1: u64) {
    unsafe {
        core::arch::asm!(
            "eor {ta}, {a0}, {a1}",
            "stp {ta}, xzr, [{dst}]",
            dst = in(reg) dst,
            a0 = in(reg) a0,
            a1 = in(reg) a1,
            ta = out(reg) _,
            options(nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
unsafe fn fold_round2_compact_chunk_neon_horner_8_impl<
    const SKIP_FIRST_SHIFT: bool,
    const GPR_DELTA: bool,
>(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    degen: bool,
    skip_dead: bool,
    out: *mut F128,
) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_anchor_pair_nt(dst: *mut F128, a: uint64x2_t, b: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{dst}]",
                dst = in(reg) dst,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack, preserves_flags),
            );
        }
    }

    #[inline(always)]
    unsafe fn fold_odd_codes(
        table_data: *const u8,
        row0: u64,
        row1: u64,
    ) -> (uint64x2_t, uint64x2_t) {
        unsafe {
            const STRIDE: usize = 256 * 16;
            let mut acc0 = bank_load_q::<0>(table_data, row0);
            let mut acc1 = bank_load_q::<0>(table_data, row1);
            // Keep each pair of loads and its accumulator update together.
            // Independent pure loads otherwise keep all 16 table values live.
            macro_rules! step_impl {
                ($c0:literal, $c1:literal, $xor0:literal, $xor1:literal) => {
                    // SAFETY: each byte index is at most 255. Both banks have
                    // 256 initialized q entries. Only the accumulators change.
                    // The three arms read the same entries (see `Q_LOAD_FORM`).
                    if Q_LOAD_FORM == 0 {
                        core::arch::asm!(
                            "ubfx {idx}, {row0}, #{shift0}, #8",
                            "ldr {tmp0:q}, [{bank0}, {idx}, lsl #4]",
                            "ubfx {idx}, {row0}, #{shift1}, #8",
                            "ldr {tmp1:q}, [{bank1}, {idx}, lsl #4]",
                            $xor0,
                            "ubfx {idx}, {row1}, #{shift0}, #8",
                            "ldr {tmp0:q}, [{bank0}, {idx}, lsl #4]",
                            "ubfx {idx}, {row1}, #{shift1}, #8",
                            "ldr {tmp1:q}, [{bank1}, {idx}, lsl #4]",
                            $xor1,
                            shift0 = const 8 * $c0,
                            shift1 = const 8 * $c1,
                            bank0 = in(reg) table_data.add($c0 * STRIDE),
                            bank1 = in(reg) table_data.add($c1 * STRIDE),
                            row0 = in(reg) row0,
                            row1 = in(reg) row1,
                            acc0 = inout(vreg) acc0,
                            acc1 = inout(vreg) acc1,
                            idx = out(reg) _,
                            tmp0 = out(vreg) _,
                            tmp1 = out(vreg) _,
                            options(pure, readonly, nostack, preserves_flags),
                        );
                    } else if Q_LOAD_FORM == 2 {
                        // `$c0` is an odd byte lane and `$c1` an even one.
                        core::arch::asm!(
                            "ubfx {idx}, {row0o}, #{shift0}, #12",
                            "ldr {tmp0:q}, [{bank0}, {idx}]",
                            "ubfx {idx}, {row0e}, #{shift1}, #12",
                            "ldr {tmp1:q}, [{bank1}, {idx}]",
                            $xor0,
                            "ubfx {idx}, {row1o}, #{shift0}, #12",
                            "ldr {tmp0:q}, [{bank0}, {idx}]",
                            "ubfx {idx}, {row1e}, #{shift1}, #12",
                            "ldr {tmp1:q}, [{bank1}, {idx}]",
                            $xor1,
                            shift0 = const 8 * $c0 - 4,
                            shift1 = const 8 * $c1 - 4,
                            bank0 = in(reg) table_data.add($c0 * STRIDE),
                            bank1 = in(reg) table_data.add($c1 * STRIDE),
                            row0o = in(reg) row0 & Q_ODD_LANES,
                            row0e = in(reg) row0 & Q_EVEN_LANES,
                            row1o = in(reg) row1 & Q_ODD_LANES,
                            row1e = in(reg) row1 & Q_EVEN_LANES,
                            acc0 = inout(vreg) acc0,
                            acc1 = inout(vreg) acc1,
                            idx = out(reg) _,
                            tmp0 = out(vreg) _,
                            tmp1 = out(vreg) _,
                            options(pure, readonly, nostack, preserves_flags),
                        );
                    } else {
                        core::arch::asm!(
                            "and {idx}, {mask}, {row0}, lsr #{shift0}",
                            "ldr {tmp0:q}, [{bank0}, {idx}]",
                            "and {idx}, {mask}, {row0}, lsr #{shift1}",
                            "ldr {tmp1:q}, [{bank1}, {idx}]",
                            $xor0,
                            "and {idx}, {mask}, {row1}, lsr #{shift0}",
                            "ldr {tmp0:q}, [{bank0}, {idx}]",
                            "and {idx}, {mask}, {row1}, lsr #{shift1}",
                            "ldr {tmp1:q}, [{bank1}, {idx}]",
                            $xor1,
                            shift0 = const 8 * $c0 - 4,
                            shift1 = const 8 * $c1 - 4,
                            mask = in(reg) Q_INDEX_MASK,
                            bank0 = in(reg) table_data.add($c0 * STRIDE),
                            bank1 = in(reg) table_data.add($c1 * STRIDE),
                            row0 = in(reg) row0,
                            row1 = in(reg) row1,
                            acc0 = inout(vreg) acc0,
                            acc1 = inout(vreg) acc1,
                            idx = out(reg) _,
                            tmp0 = out(vreg) _,
                            tmp1 = out(vreg) _,
                            options(pure, readonly, nostack, preserves_flags),
                        );
                    }
                };
            }
            macro_rules! step {
                ($c0:literal, $c1:literal) => {
                    #[cfg(target_feature = "sha3")]
                    step_impl!(
                        $c0, $c1,
                        "eor3 {acc0:v}.16b, {acc0:v}.16b, {tmp0:v}.16b, {tmp1:v}.16b",
                        "eor3 {acc1:v}.16b, {acc1:v}.16b, {tmp0:v}.16b, {tmp1:v}.16b"
                    );
                    #[cfg(not(target_feature = "sha3"))]
                    step_impl!(
                        $c0, $c1,
                        "eor {acc0:v}.16b, {acc0:v}.16b, {tmp0:v}.16b\neor {acc0:v}.16b, {acc0:v}.16b, {tmp1:v}.16b",
                        "eor {acc1:v}.16b, {acc1:v}.16b, {tmp0:v}.16b\neor {acc1:v}.16b, {acc1:v}.16b, {tmp1:v}.16b"
                    );
                };
            }
            step!(1, 2);
            step!(3, 4);
            step!(5, 6);
            acc0 = veorq_u64(acc0, bank_load_q::<56>(table_data.add(7 * STRIDE), row0));
            acc1 = veorq_u64(acc1, bank_load_q::<56>(table_data.add(7 * STRIDE), row1));
            (acc0, acc1)
        }
    }

    #[inline(always)]
    unsafe fn shift_pair(a: &mut WideNeon, b: &mut WideNeon, top: &mut uint64x2_t) {
        unsafe {
            let zero = vdupq_n_u64(0);
            *top = vsriq_n_u64::<63>(vshlq_n_u64::<1>(*top), vzip2q_u64(a.hi, b.hi));
            a.hi = vsriq_n_u64::<63>(vshlq_n_u64::<1>(a.hi), vextq_u64::<1>(a.lo, a.hi));
            b.hi = vsriq_n_u64::<63>(vshlq_n_u64::<1>(b.hi), vextq_u64::<1>(b.lo, b.hi));
            a.lo = vsriq_n_u64::<63>(vshlq_n_u64::<1>(a.lo), vextq_u64::<1>(zero, a.lo));
            b.lo = vsriq_n_u64::<63>(vshlq_n_u64::<1>(b.lo), vextq_u64::<1>(zero, b.lo));
        }
    }

    #[inline(always)]
    unsafe fn reduce_pair(
        mut a: WideNeon,
        mut b: WideNeon,
        top: uint64x2_t,
    ) -> (uint64x2_t, uint64x2_t) {
        unsafe {
            let zero = vdupq_n_u64(0);
            // x^256 = 1 + x^2 + x^4 + x^14 modulo x^128+x^7+x^2+x+1.
            // Each top limb has at most 14 bits, so this cannot overflow.
            let folded = veorq_u64(
                xor3_u64(top, vshlq_n_u64::<2>(top), vshlq_n_u64::<4>(top)),
                vshlq_n_u64::<14>(top),
            );
            a.lo = veorq_u64(a.lo, vzip1q_u64(folded, zero));
            b.lo = veorq_u64(b.lo, vzip2q_u64(folded, zero));
            (reduce_wide_q(a), reduce_wide_q(b))
        }
    }

    unsafe {
        let zero = vdupq_n_u64(0);
        let mut sum0 = zero;
        let mut sum3 = zero;
        let mut sum4 = zero;
        let mut sum5 = zero;
        let b_ones = fold_row_q(table_data, [0xffu8; 8].as_ptr());
        let ones_is_one = is_zero_q(veorq_u64(
            b_ones,
            core::mem::transmute::<F128, uint64x2_t>(F128::ONE),
        ));

        let aligned_padding = pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
            && (pair_idx_base & 127) == 0
            && lo_size.is_multiple_of(128);
        let fixed_pads = aligned_padding && r2_fixed_pads_enabled();
        for first in (0..lo_size / 2).step_by(32) {
            for t in 0..2 {
                let mut w0 = WideNeon { lo: zero, hi: zero };
                let mut w3 = WideNeon { lo: zero, hi: zero };
                let mut w4 = WideNeon { lo: zero, hi: zero };
                let mut w5 = WideNeon { lo: zero, hi: zero };
                let mut top03 = zero;
                let mut top45 = zero;
                // Groups 61..64 are leading zero coefficients in this
                // sub-polynomial. Its ten q values are still zero here.
                let b_end = if aligned_padding && (first & 63) == 32 {
                    15 - t
                } else {
                    16
                };
                if !skip_dead {
                    for b in (b_end..16).rev() {
                        let x_lo = 2 * (first + 2 * b + t);
                        store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                        vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                        store_anchor_pair_nt(anchors.add(2 * (x_lo + 1)), zero, zero);
                        vst1q_u64(deltas.add((x_lo + 1) * 16).cast::<u64>(), zero);
                    }
                }
                // One group at Horner position `$b`. `$shift` is false only
                // for the peeled first group of a window, whose accumulators
                // are all zero.
                macro_rules! horner_group {
                    ($b:expr, $shift:expr, $pads:expr) => {
                        'group: {
                            if $shift {
                                shift_pair(&mut w0, &mut w3, &mut top03);
                                shift_pair(&mut w4, &mut w5, &mut top45);
                            }
                            let u = first + 2 * $b + t;
                            let x_lo0 = 2 * u;
                            let x_lo1 = x_lo0 + 1;
                            let (pad0, pad1): (bool, bool) = ($pads)(x_lo0, x_lo1);
                            // One pair: fold and store its even rows, hand back its odd
                            // row codes. A padded pair keeps its zero store and reads as
                            // all-zero codes.
                            let pair = |x_lo: usize, pad: bool| {
                                if pad {
                                    if !skip_dead {
                                        store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                                        vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                                    }
                                    return (zero, zero, 0u64, 0u64, false);
                                }
                                let a_row = a_packed.add(2 * x_lo * 8).cast::<u64>();
                                let b_row = b_packed.add(2 * x_lo * 8).cast::<u64>();
                                let (a_even, a_odd, b_even, b_odd) = if GPR_DELTA {
                                    let (a_even, a_odd) = load_code_pair_gpr(a_row);
                                    let (b_even, b_odd) = load_code_pair_gpr(b_row);
                                    (a_even, a_odd, b_even, b_odd)
                                } else {
                                    (
                                        u64::from_le(core::ptr::read_unaligned(a_row)),
                                        u64::from_le(core::ptr::read_unaligned(a_row.add(1))),
                                        u64::from_le(core::ptr::read_unaligned(b_row)),
                                        u64::from_le(core::ptr::read_unaligned(b_row.add(1))),
                                    )
                                };
                                let deg = degen && (b_even & b_odd) == u64::MAX;
                                let (a, b) = if deg {
                                    (fold_row_code_q(table_data, a_even), b_ones)
                                } else {
                                    fold_two_row_codes_q(table_data, a_even, b_even)
                                };
                                store_anchor_pair_nt(anchors.add(2 * x_lo), a, b);
                                if GPR_DELTA {
                                    store_delta_pair_gpr(
                                        deltas.add(x_lo * 16),
                                        a_even,
                                        a_odd,
                                        b_even,
                                        b_odd,
                                    );
                                } else {
                                    let delta_pair = core::mem::transmute::<[u64; 2], uint64x2_t>([
                                        a_even ^ a_odd,
                                        b_even ^ b_odd,
                                    ]);
                                    vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);
                                }
                                (a, b, a_odd, b_odd, deg)
                            };
                            let (a0, b0, a1c, b1c, deg0) = pair(x_lo0, pad0);
                            let (a2, b2, a3c, b3c, deg1) = pair(x_lo1, pad1);
                            if pad0 & pad1 {
                                break 'group;
                            }
                            if deg0 & deg1 & ones_is_one {
                                // b == 1 on all four rows: only `a2 * b2 = a2` survives.
                                w0.lo = veorq_u64(w0.lo, a2);
                            } else {
                                let e_a = veorq_u64(a0, a2);
                                let e_b = veorq_u64(b0, b2);
                                if !pad1 {
                                    wide_xor(&mut w0, mul_unreduced_sw_q(a2, swap_halves_q(a2), b2));
                                }
                                wide_xor(&mut w3, mul_unreduced_sw_q(e_a, swap_halves_q(e_a), e_b));
                                // Consume the individual even rows before the odd
                                // lookup chains need their accumulator registers.
                                let (o_a, o_b) = fold_odd_codes(table_data, a1c ^ a3c, b1c ^ b3c);
                                wide_xor(&mut w4, mul_unreduced_sw_q(o_a, swap_halves_q(o_a), o_b));
                                let all_a = veorq_u64(e_a, o_a);
                                let all_b = veorq_u64(e_b, o_b);
                                wide_xor(
                                    &mut w5,
                                    mul_unreduced_sw_q(all_a, swap_halves_q(all_a), all_b),
                                );
                            }
                        }
                    };
                }
                let generic_pads = |x0: usize, x1: usize| {
                    (
                        ((pair_idx_base + x0) & pair_in_block_mask) >= useful_pairs_inclusive,
                        ((pair_idx_base + x1) & pair_in_block_mask) >= useful_pairs_inclusive,
                    )
                };
                if SKIP_FIRST_SHIFT {
                    // `b_end` is at least 14, so the peeled group exists.
                    if fixed_pads {
                        let boundary = b_end == 15;
                        horner_group!(b_end - 1, false, |_: usize, _: usize| (false, boundary));
                        for b in (0..b_end - 1).rev() {
                            horner_group!(b, true, |_: usize, _: usize| (false, false));
                        }
                    } else {
                        horner_group!(b_end - 1, false, generic_pads);
                        for b in (0..b_end - 1).rev() {
                            horner_group!(b, true, generic_pads);
                        }
                    }
                } else {
                    for b in (0..b_end).rev() {
                        horner_group!(b, true, generic_pads);
                    }
                }
                let lambda = vld1q_u64(eq_lo.add(2 * (first + t) + 1).cast::<u64>());
                let (w0, w3) = reduce_pair(w0, w3, top03);
                let (w4, w5) = reduce_pair(w4, w5, top45);
                sum0 = veorq_u64(sum0, mul_q(lambda, w0));
                sum3 = veorq_u64(sum3, mul_q(lambda, w3));
                sum4 = veorq_u64(sum4, mul_q(lambda, w4));
                sum5 = veorq_u64(sum5, mul_q(lambda, w5));
            }
        }
        for i in 0..4 {
            *out.add(i) = F128::ZERO;
        }
        *out.add(4) = core::mem::transmute::<uint64x2_t, F128>(sum0);
        *out.add(5) = core::mem::transmute::<uint64x2_t, F128>(sum3);
        *out.add(6) = core::mem::transmute::<uint64x2_t, F128>(sum4);
        *out.add(7) = core::mem::transmute::<uint64x2_t, F128>(sum5);
    }
}

///
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn mul_xb_q(
    v: core::arch::aarch64::uint64x2_t,
    xb: core::arch::aarch64::uint64x2_t,
    k: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    unsafe {
        core::arch::asm!(
            "pmull2 {u:v}.1q, {v:v}.2d, {xb:v}.2d",
            "pmull2 {r:v}.1q, {u:v}.2d, {k:v}.2d",
            "zip1 {r:v}.2d, {r:v}.2d, {u:v}.2d",
            "pmull {o:v}.1q, {v:v}.1d, {xb:v}.1d",
            "eor {o:v}.16b, {o:v}.16b, {r:v}.16b",
            v = in(vreg) v,
            xb = in(vreg) xb,
            k = in(vreg) k,
            u = out(vreg) _,
            r = out(vreg) _,
            o = lateout(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn fold_two_codes_paired_q(
    table_data: *const u8,
    row0: u64,
    row1: u64,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    if Q_LOAD_FORM != 1 {
        return unsafe { fold_two_row_codes_q(table_data, row0, row1) };
    }
    unsafe {
        const STRIDE: usize = 256 * 16;
        let mut acc0 = bank_load_q::<0>(table_data, row0);
        let mut acc1 = bank_load_q::<0>(table_data, row1);
        macro_rules! step {
            ($c0:literal, $c1:literal) => {
                core::arch::asm!(
                    "and {idx}, {mask}, {row0}, lsr #{shift0}",
                    "ldr {tmp0:q}, [{bank0}, {idx}]",
                    "and {idx}, {mask}, {row0}, lsr #{shift1}",
                    "ldr {tmp1:q}, [{bank1}, {idx}]",
                    "eor3 {acc0:v}.16b, {acc0:v}.16b, {tmp0:v}.16b, {tmp1:v}.16b",
                    "and {idx}, {mask}, {row1}, lsr #{shift0}",
                    "ldr {tmp0:q}, [{bank0}, {idx}]",
                    "and {idx}, {mask}, {row1}, lsr #{shift1}",
                    "ldr {tmp1:q}, [{bank1}, {idx}]",
                    "eor3 {acc1:v}.16b, {acc1:v}.16b, {tmp0:v}.16b, {tmp1:v}.16b",
                    shift0 = const 8 * $c0 - 4,
                    shift1 = const 8 * $c1 - 4,
                    mask = in(reg) Q_INDEX_MASK,
                    bank0 = in(reg) table_data.add($c0 * STRIDE),
                    bank1 = in(reg) table_data.add($c1 * STRIDE),
                    row0 = in(reg) row0,
                    row1 = in(reg) row1,
                    acc0 = inout(vreg) acc0,
                    acc1 = inout(vreg) acc1,
                    idx = out(reg) _,
                    tmp0 = out(vreg) _,
                    tmp1 = out(vreg) _,
                    options(pure, readonly, nostack, preserves_flags),
                );
            };
        }
        step!(1, 2);
        step!(3, 4);
        step!(5, 6);
        acc0 = veorq_u64(acc0, bank_load_q::<56>(table_data.add(7 * STRIDE), row0));
        acc1 = veorq_u64(acc1, bank_load_q::<56>(table_data.add(7 * STRIDE), row1));
        (acc0, acc1)
    }
}

///
///
///
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold_round2_compact_chunk_neon_prescaled_8(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    degen: bool,
    skip_dead: bool,
    out: *mut F128,
) {
    unsafe {
        match (r2_gpr_delta_enabled(), r2_fixed_pads_enabled()) {
            (true, true) => fold_round2_compact_chunk_neon_prescaled_8_impl::<true, true>(
                table_data,
                a_packed,
                b_packed,
                anchors,
                deltas,
                eq_lo,
                lo_size,
                pair_idx_base,
                pair_in_block_mask,
                useful_pairs_inclusive,
                degen,
                skip_dead,
                out,
            ),
            (true, false) => fold_round2_compact_chunk_neon_prescaled_8_impl::<true, false>(
                table_data,
                a_packed,
                b_packed,
                anchors,
                deltas,
                eq_lo,
                lo_size,
                pair_idx_base,
                pair_in_block_mask,
                useful_pairs_inclusive,
                degen,
                skip_dead,
                out,
            ),
            (false, true) => fold_round2_compact_chunk_neon_prescaled_8_impl::<false, true>(
                table_data,
                a_packed,
                b_packed,
                anchors,
                deltas,
                eq_lo,
                lo_size,
                pair_idx_base,
                pair_in_block_mask,
                useful_pairs_inclusive,
                degen,
                skip_dead,
                out,
            ),
            (false, false) => fold_round2_compact_chunk_neon_prescaled_8_impl::<false, false>(
                table_data,
                a_packed,
                b_packed,
                anchors,
                deltas,
                eq_lo,
                lo_size,
                pair_idx_base,
                pair_in_block_mask,
                useful_pairs_inclusive,
                degen,
                skip_dead,
                out,
            ),
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
unsafe fn fold_round2_compact_chunk_neon_prescaled_8_impl<
    const GPR_DELTA: bool,
    const FIXED_PADS: bool,
>(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    degen: bool,
    skip_dead: bool,
    out: *mut F128,
) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_anchor_pair_nt(dst: *mut F128, a: uint64x2_t, b: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{dst}]",
                dst = in(reg) dst,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        let zero = vdupq_n_u64(0);
        let k87 = vdupq_n_u64(0x87);
        let tri_zero = TriNeon {
            ll: zero,
            cross: zero,
            hh: zero,
        };
        // Window sums: with `r2_window_tri_enabled` each window's weighted
        // value is accumulated unreduced (three-limb form) and the four sums
        // are reduced once per chunk; otherwise one reduced product per window.
        let window_tri = r2_window_tri_enabled();
        let mut sum0 = tri_zero;
        let mut sum3 = tri_zero;
        let mut sum4 = tri_zero;
        let mut sum5 = tri_zero;
        let b_ones = fold_row_q(table_data, [0xffu8; 8].as_ptr());
        let ones_is_one = is_zero_q(veorq_u64(
            b_ones,
            core::mem::transmute::<F128, uint64x2_t>(F128::ONE),
        ));

        let aligned_padding = pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
            && (pair_idx_base & 127) == 0
            && lo_size.is_multiple_of(128);
        let fixed_pads = FIXED_PADS && aligned_padding;
        for first in (0..lo_size / 2).step_by(32) {
            for t in 0..2 {
                let mut w0 = tri_zero;
                let mut w3 = tri_zero;
                let mut w4 = tri_zero;
                let mut w5 = tri_zero;
                let b_end = if aligned_padding && (first & 63) == 32 {
                    15 - t
                } else {
                    16
                };
                if !skip_dead {
                    for pos in (b_end..16).rev() {
                        let x_lo = 2 * (first + 2 * pos + t);
                        store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                        vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                        store_anchor_pair_nt(anchors.add(2 * (x_lo + 1)), zero, zero);
                        vst1q_u64(deltas.add((x_lo + 1) * 16).cast::<u64>(), zero);
                    }
                }
                let mut xb = vdupq_n_u64(1u64 << (b_end - 1));
                macro_rules! group {
                    ($pos:expr, $pads:expr) => {
                        'group: {
                            let u = first + 2 * $pos + t;
                            let x_lo0 = 2 * u;
                            let x_lo1 = x_lo0 + 1;
                            let (pad0, pad1): (bool, bool) = ($pads)(x_lo0, x_lo1);
                            let pair = |x_lo: usize, pad: bool| {
                                if pad {
                                    if !skip_dead {
                                        store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                                        vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                                    }
                                    return (zero, zero, 0u64, 0u64, false);
                                }
                                let a_row = a_packed.add(2 * x_lo * 8).cast::<u64>();
                                let b_row = b_packed.add(2 * x_lo * 8).cast::<u64>();
                                let (a_even, a_odd, b_even, b_odd) = if GPR_DELTA {
                                    let (a_even, a_odd) = load_code_pair_gpr(a_row);
                                    let (b_even, b_odd) = load_code_pair_gpr(b_row);
                                    (a_even, a_odd, b_even, b_odd)
                                } else {
                                    (
                                        u64::from_le(core::ptr::read_unaligned(a_row)),
                                        u64::from_le(core::ptr::read_unaligned(a_row.add(1))),
                                        u64::from_le(core::ptr::read_unaligned(b_row)),
                                        u64::from_le(core::ptr::read_unaligned(b_row.add(1))),
                                    )
                                };
                                let deg = degen && (b_even & b_odd) == u64::MAX;
                                let (a, b) = if deg {
                                    (fold_row_code_q(table_data, a_even), b_ones)
                                } else {
                                    fold_two_row_codes_q(table_data, a_even, b_even)
                                };
                                store_anchor_pair_nt(anchors.add(2 * x_lo), a, b);
                                if GPR_DELTA {
                                    store_delta_pair_gpr(
                                        deltas.add(x_lo * 16),
                                        a_even,
                                        a_odd,
                                        b_even,
                                        b_odd,
                                    );
                                } else {
                                    let delta_pair = core::mem::transmute::<[u64; 2], uint64x2_t>([
                                        a_even ^ a_odd,
                                        b_even ^ b_odd,
                                    ]);
                                    vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);
                                }
                                (a, b, a_odd, b_odd, deg)
                            };
                            let (a0, b0, a1c, b1c, deg0) = pair(x_lo0, pad0);
                            let (a2, b2, a3c, b3c, deg1) = pair(x_lo1, pad1);
                            if pad0 & pad1 {
                                break 'group;
                            }
                            if deg0 & deg1 & ones_is_one {
                                w0.ll = veorq_u64(w0.ll, mul_xb_q(a2, xb, k87));
                            } else {
                                let e_a = veorq_u64(a0, a2);
                                let e_b = veorq_u64(b0, b2);
                                if !pad1 {
                                    tri_acc_q_crossfused(&mut w0, mul_xb_q(a2, xb, k87), b2);
                                }
                                let e_s = mul_xb_q(e_a, xb, k87);
                                tri_acc_q_crossfused(&mut w3, e_s, e_b);
                                let (o_a, o_b) =
                                    fold_two_codes_paired_q(table_data, a1c ^ a3c, b1c ^ b3c);
                                let o_s = mul_xb_q(o_a, xb, k87);
                                tri_acc_q_crossfused(&mut w4, o_s, o_b);
                                tri_acc_q_crossfused(
                                    &mut w5,
                                    veorq_u64(e_s, o_s),
                                    veorq_u64(e_b, o_b),
                                );
                            }
                        }
                        xb = vshrq_n_u64::<1>(xb);
                    };
                }
                let generic_pads = |x0: usize, x1: usize| {
                    (
                        ((pair_idx_base + x0) & pair_in_block_mask) >= useful_pairs_inclusive,
                        ((pair_idx_base + x1) & pair_in_block_mask) >= useful_pairs_inclusive,
                    )
                };
                if fixed_pads {
                    let boundary = b_end == 15;
                    group!(b_end - 1, |_: usize, _: usize| (false, boundary));
                    for pos in (0..b_end - 1).rev() {
                        group!(pos, |_: usize, _: usize| (false, false));
                    }
                } else {
                    for pos in (0..b_end).rev() {
                        group!(pos, generic_pads);
                    }
                }
                let lambda = vld1q_u64(eq_lo.add(2 * (first + t) + 1).cast::<u64>());
                if window_tri {
                    tri_acc_q_crossfused(&mut sum0, lambda, reduce_tri_binius_xor_q(w0.ll, w0.cross, w0.hh, zero));
                    tri_acc_q_crossfused(&mut sum3, lambda, reduce_tri_binius_xor_q(w3.ll, w3.cross, w3.hh, zero));
                    tri_acc_q_crossfused(&mut sum4, lambda, reduce_tri_binius_xor_q(w4.ll, w4.cross, w4.hh, zero));
                    tri_acc_q_crossfused(&mut sum5, lambda, reduce_tri_binius_xor_q(w5.ll, w5.cross, w5.hh, zero));
                } else {
                    sum0.ll = veorq_u64(sum0.ll, mul_q(lambda, reduce_tri_q(w0)));
                    sum3.ll = veorq_u64(sum3.ll, mul_q(lambda, reduce_tri_q(w3)));
                    sum4.ll = veorq_u64(sum4.ll, mul_q(lambda, reduce_tri_q(w4)));
                    sum5.ll = veorq_u64(sum5.ll, mul_q(lambda, reduce_tri_q(w5)));
                }
            }
        }
        for i in 0..4 {
            *out.add(i) = F128::ZERO;
        }
        *out.add(4) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(sum0));
        *out.add(5) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(sum3));
        *out.add(6) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(sum4));
        *out.add(7) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(sum5));
    }
}

/// Bind **both** ρ₁ and ρ₂ straight out of the compact round-two state and
/// emit the round-four message in the same pass (variant K).
///
/// `A''[y] = [anc0 + ρ₂(anc0+anc1)] + fold_{λ₁}(δ0) + fold_{λ₃}(δ1)` with
/// `λ₁ = ρ₁(1+ρ₂)`, `λ₃ = ρ₁ρ₂` — in characteristic two `λ₀+λ₁ = 1+ρ₂` and
/// `λ₂+λ₃ = ρ₂`, so the two anchors collapse into one ordinary ρ₂ fold and
/// only the two deltas need λ-scaled tables. Padding needs no predicate: the
/// compact state already carries zero anchors and zero deltas there, and a
/// zero delta code folds to zero through the table's zero entry.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold2_compact_and_round4_chunk_neon_8(
    table_l1: *const u8,
    table_l3: *const u8,
    rho2: F128,
    anchors: *const F128,
    deltas: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    out_pairs: usize,
    degen: bool,
) -> (F128, F128) {
    use core::arch::aarch64::*;

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
        let rho2_q = core::mem::transmute::<F128, uint64x2_t>(rho2);
        let one_q = core::mem::transmute::<F128, uint64x2_t>(F128::ONE);
        let mut p1_acc = TriNeon { ll: zero, cross: zero, hh: zero };
        let mut pinf_acc = TriNeon { ll: zero, cross: zero, hh: zero };

        for u in 0..out_pairs {
            let mut av = [zero; 2];
            let mut bv = [zero; 2];
            let mut b_flat = true;
            for lane in 0..2usize {
                let g = 2 * u + lane;
                let ap = anchors.add(4 * g).cast::<u64>();
                let anc_a0 = vld1q_u64(ap);
                let anc_b0 = vld1q_u64(ap.add(2));
                let anc_a1 = vld1q_u64(ap.add(4));
                let anc_b1 = vld1q_u64(ap.add(6));

                let dp = deltas.add(32 * g);
                let da0 = u64::from_le(core::ptr::read_unaligned(dp.cast::<u64>()));
                let db0 = u64::from_le(core::ptr::read_unaligned(dp.add(8).cast::<u64>()));
                let da1 = u64::from_le(core::ptr::read_unaligned(dp.add(16).cast::<u64>()));
                let db1 = u64::from_le(core::ptr::read_unaligned(dp.add(24).cast::<u64>()));

                let a_delta = veorq_u64(
                    lookup_lanes_q::<8>(table_l1, da0, 0),
                    lookup_lanes_q::<8>(table_l3, da1, 0),
                );
                av[lane] = xor3_u64(anc_a0, mul_q(rho2_q, veorq_u64(anc_a0, anc_a1)), a_delta);

                if degen && (db0 | db1) == 0 {
                    // b rows are constant across the group: zero deltas mean
                    // both b halves equal their anchors.
                    let bd = veorq_u64(anc_b0, anc_b1);
                    bv[lane] = if is_zero_q(bd) {
                        anc_b0
                    } else {
                        b_flat = false;
                        veorq_u64(anc_b0, mul_q(rho2_q, bd))
                    };
                } else {
                    b_flat = false;
                    let b_delta = veorq_u64(
                        lookup_lanes_q::<8>(table_l1, db0, 0),
                        lookup_lanes_q::<8>(table_l3, db1, 0),
                    );
                    bv[lane] = xor3_u64(anc_b0, mul_q(rho2_q, veorq_u64(anc_b0, anc_b1)), b_delta);
                }
            }

            store_pair_nt(a_out.add(2 * u), av[0], av[1]);
            store_pair_nt(b_out.add(2 * u), bv[0], bv[1]);

            let eq_l = vld1q_u64(eq_lo.add(u).cast::<u64>());
            if b_flat {
                // Value-forced shortcuts on the static b≡1 mass, gated on the
                // loaded values so they stay bit-exact.
                let g1 = if is_zero_q(veorq_u64(bv[1], one_q)) {
                    av[1]
                } else {
                    mul_q(av[1], bv[1])
                };
                tri_acc_q(&mut p1_acc, eq_l, g1);
                let b_sum = veorq_u64(bv[0], bv[1]);
                if !is_zero_q(b_sum) {
                    let g_inf = mul_q(veorq_u64(av[0], av[1]), b_sum);
                    tri_acc_q(&mut pinf_acc, eq_l, g_inf);
                }
            } else {
                let g1 = mul_q(av[1], bv[1]);
                let g_inf = mul_q(veorq_u64(av[0], av[1]), veorq_u64(bv[0], bv[1]));
                tri_acc_q(&mut p1_acc, eq_l, g1);
                tri_acc_q(&mut pinf_acc, eq_l, g_inf);
            }
        }

        (
            core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(p1_acc)),
            core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(pinf_acc)),
        )
    }
}

/// Cascade variant of [`fold2_compact_and_round4_chunk_neon_8`]: identical
/// per-output value computation and byte-identical stores, but K outputs are
/// visited FOUR at a time — one round-**five** group `y' = out/4` (two
/// adjacent round-4 pairs) — so the six deferred round-five aggregates can be
/// formed while the group's outputs are still live in registers, exactly the
/// position the round-2 sweep exploited for the round-3 lookahead.
///
/// **One weight per group, one scaling per row** (the same reshape as
/// [`fold_round2_compact_chunk_neon_lookahead_8`]): with `r' = r_next4[1]`,
/// `eq4(2y') = (1+r')·eq5(y')` and `eq4(2y'+1) = r'·eq5(y')`, so the whole
/// group accumulates against the single odd-lane weight `w = eq4_lo[2t+1]`,
/// the four A outputs are pre-scaled by `w` with four reduced multiplies, and
/// all eight products cost one unreduced multiply each. The constant
/// rescalings (`kappa = (1+r')/r'` on the even round-4 sums, `r'^-1` on the
/// six deferred aggregates) are applied once by the driver, off the hot path.
///
/// Writes eight F128 slots to `out`, every one of them `w`-weighted:
///
/// | slot | value |
/// |---|---|
/// | 0, 1 | `sum_{u even} w·a1·b1`, `sum_{u even} w·(a0+a1)(b0+b1)` |
/// | 2, 3 | the same two sums over **odd** round-4 pairs `u` — i.e. `r'·W1'`, `r'·W2'` |
/// | 4..8 | `r'·W0'`, `r'·W3'`, `r'·W4'`, `r'·W5'` |
///
/// with `W0' = Σ eq5·a2b2`, `W3'/W4'/W5'` the even/odd K-output-parity split
/// (`e = out[4y']+out[4y'+2]`, `o = out[4y'+1]+out[4y'+3]`), mirroring the
/// round-3 lookahead's aggregate shape one level deeper.
///
/// The incumbent kernel's per-pair `b_flat` product shortcuts are replaced by
/// a group-level `b ≡ 1` shortcut (all four `b` outputs equal ONE, value
/// gated): every difference product vanishes (`b0+b1 = b2+b3 = e_b = o_b =
/// 0`) and the three survivors are `w·a1`, `w·a3`, `w·a2` — three unreduced
/// multiplies for the whole group. Mixed groups take the full path, which is
/// value-identical (products of genuinely-zero factors XOR in as zero).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold2_compact_and_round45_chunk_neon_8<const DEAD_TAIL: bool>(
    table_l1: *const u8,
    table_l3: *const u8,
    rho2: F128,
    anchors: *const F128,
    deltas: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    out_pairs: usize,
    degen: bool,
    out: *mut F128,
) {
    use core::arch::aarch64::*;

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
        let rho2_q = core::mem::transmute::<F128, uint64x2_t>(rho2);
        // Swapped once per pass: every rho2 product reads matching lanes.
        let rho2_sw = swap_halves_q(rho2_q);
        let one_q = core::mem::transmute::<F128, uint64x2_t>(F128::ONE);
        // Slots: p1_even, pinf_even, p1_odd, pinf_odd, W0', W3', W4', W5'.
        let mut acc = LookaheadTriAcc::zero();

        debug_assert!(out_pairs >= 2 && out_pairs.is_multiple_of(2));
        let n5 = out_pairs / 2;
        // One round-five group `t` (four K outputs, eight pairs).
        macro_rules! k_group {
            ($t:expr) => {{
                let t = $t;
                let mut av = [zero; 4];
                let mut bv = [zero; 4];
                let mut b_flat = true;
                for lane in 0..4usize {
                    let g = 4 * t + lane;
                    let ap = anchors.add(4 * g).cast::<u64>();
                    let anc_a0 = vld1q_u64(ap);
                    let anc_b0 = vld1q_u64(ap.add(2));
                    let anc_a1 = vld1q_u64(ap.add(4));
                    let anc_b1 = vld1q_u64(ap.add(6));

                    let dp = deltas.add(32 * g);
                    let da0 = u64::from_le(core::ptr::read_unaligned(dp.cast::<u64>()));
                    let db0 = u64::from_le(core::ptr::read_unaligned(dp.add(8).cast::<u64>()));
                    let da1 = u64::from_le(core::ptr::read_unaligned(dp.add(16).cast::<u64>()));
                    let db1 = u64::from_le(core::ptr::read_unaligned(dp.add(24).cast::<u64>()));

                    let a_delta = veorq_u64(
                        lookup_lanes_q::<8>(table_l1, da0, 0),
                        lookup_lanes_q::<8>(table_l3, da1, 0),
                    );
                    av[lane] = xor3_u64(
                        anc_a0,
                        mul_sw_tri_binius_q(rho2_q, rho2_sw, veorq_u64(anc_a0, anc_a1)),
                        a_delta,
                    );

                    if degen && (db0 | db1) == 0 {
                        // b rows are constant across the group: zero deltas mean
                        // both b halves equal their anchors.
                        let bd = veorq_u64(anc_b0, anc_b1);
                        bv[lane] = if is_zero_q(bd) {
                            anc_b0
                        } else {
                            b_flat = false;
                            veorq_u64(anc_b0, mul_sw_tri_binius_q(rho2_q, rho2_sw, bd))
                        };
                    } else {
                        b_flat = false;
                        let b_delta = veorq_u64(
                            lookup_lanes_q::<8>(table_l1, db0, 0),
                            lookup_lanes_q::<8>(table_l3, db1, 0),
                        );
                        bv[lane] = xor3_u64(
                            anc_b0,
                            mul_sw_tri_binius_q(rho2_q, rho2_sw, veorq_u64(anc_b0, anc_b1)),
                            b_delta,
                        );
                    }
                }

                store_pair_nt(a_out.add(4 * t), av[0], av[1]);
                store_pair_nt(a_out.add(4 * t + 2), av[2], av[3]);
                store_pair_nt(b_out.add(4 * t), bv[0], bv[1]);
                store_pair_nt(b_out.add(4 * t + 2), bv[2], bv[3]);

                // The odd round-4 pair's weight drives the whole group; see doc.
                let w = vld1q_u64(eq_lo.add(2 * t + 1).cast::<u64>());

                if b_flat {
                    // Every b output equals its anchor; check the b≡1 mass.
                    let ones_miss = vorrq_u64(
                        vorrq_u64(veorq_u64(bv[0], one_q), veorq_u64(bv[1], one_q)),
                        vorrq_u64(veorq_u64(bv[2], one_q), veorq_u64(bv[3], one_q)),
                    );
                    if is_zero_q(ones_miss) {
                        tri_acc_q_compact(&mut acc.0[0], w, av[1]);
                        tri_acc_q_compact(&mut acc.0[2], w, av[3]);
                        tri_acc_q_compact(&mut acc.0[4], w, av[2]);
                        continue;
                    }
                }

                // Weight-sharing reshape: four reduced row scalings sharing
                // one swap of w, then eight unreduced products in tri form.
                acc.add_group_compact(w, av[0], av[1], av[2], av[3], bv[0], bv[1], bv[2], bv[3]);
            }};
        }

        if DEAD_TAIL {
            // Round two left the dead-pair slots unwritten: in every 128-pair
            // block, pairs 121..128 are structural padding (the last group:
            // lane 0's odd pair and lanes 1..3). Their dense values are known
            // zeros (zero anchors, zero deltas, table entry 0 = 0), so the
            // tail group is formed from lane 0's live pair alone; every
            // product with a zero factor is dropped (XOR of zero is exact).
            debug_assert!(n5.is_multiple_of(16));
            for blk in 0..n5 / 16 {
                let t0 = 16 * blk;
                for t in t0..t0 + 15 {
                    k_group!(t);
                }
                let t = t0 + 15;
                let g = 4 * t;
                let ap = anchors.add(4 * g).cast::<u64>();
                let anc_a0 = vld1q_u64(ap);
                let anc_b0 = vld1q_u64(ap.add(2));
                let dp = deltas.add(32 * g);
                let da0 = u64::from_le(core::ptr::read_unaligned(dp.cast::<u64>()));
                let db0 = u64::from_le(core::ptr::read_unaligned(dp.add(8).cast::<u64>()));
                // Dense lane 0 with anc1 = 0 and delta1 = 0: anc0 + rho2*anc0
                // + T_l1(delta0) (+ T_l3(0) = 0). This one formula equals both
                // dense b branches (degenerate and full).
                let av0 = xor3_u64(
                    anc_a0,
                    mul_sw_tri_binius_q(rho2_q, rho2_sw, anc_a0),
                    lookup_lanes_q::<8>(table_l1, da0, 0),
                );
                let bv0 = xor3_u64(
                    anc_b0,
                    mul_sw_tri_binius_q(rho2_q, rho2_sw, anc_b0),
                    lookup_lanes_q::<8>(table_l1, db0, 0),
                );
                store_pair_nt(a_out.add(4 * t), av0, zero);
                store_pair_nt(a_out.add(4 * t + 2), zero, zero);
                store_pair_nt(b_out.add(4 * t), bv0, zero);
                store_pair_nt(b_out.add(4 * t + 2), zero, zero);
                // Lanes 1..3 are zero, so the dense group's only non-zero
                // products are (w*a0)*b0 in pinf_even, W3' and W5'.
                let w = vld1q_u64(eq_lo.add(2 * t + 1).cast::<u64>());
                let (ll, cross, hh) = clmul_products_q(mul_q(w, av0), bv0);
                for slot in [1usize, 5, 7] {
                    let t = &mut acc.0[slot];
                    t.ll = veorq_u64(t.ll, ll);
                    t.cross = veorq_u64(t.cross, cross);
                    t.hh = veorq_u64(t.hh, hh);
                }
            }
        } else {
            for t in 0..n5 {
                k_group!(t);
            }
        }

        let sums = acc.reduce();
        for (k, v) in sums.into_iter().enumerate() {
            *out.add(k) = v;
        }
    }
}

/// `a*b + e1 + e2` in the field, with `a_sw = swap_halves_q(a)` and `k` =
/// `0x87` in both lanes. The same four schoolbook `PMULL`s and the same
/// two-step reduction as [`mul_sw_tri_binius_q`] (so the same field value),
/// but every `PMULL` that feeds an XOR is issued as an adjacent PMULL+EOR
/// fused pair and the reduction constant is a hoisted register: 9 SIMD issue
/// slots plus one `EOR3` for the addends, instead of 12 plus a `DUP` of 0x87.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn mul_sw_fused_xor3_q(
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    k: core::arch::aarch64::uint64x2_t,
    e1: core::arch::aarch64::uint64x2_t,
    e2: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    // SAFETY: register-only carry-less products and XORs (AES and SHA3 are
    // enabled for this build); every scratch is an early clobber, so no
    // input is overwritten before its last read.
    unsafe {
        core::arch::asm!(
            "pmull {ll:v}.1q, {a:v}.1d, {b:v}.1d",
            "pmull2 {hh:v}.1q, {a:v}.2d, {b:v}.2d",
            "ext {x:v}.16b, {ll:v}.16b, {hh:v}.16b, #8",
            "pmull {p:v}.1q, {s:v}.1d, {b:v}.1d",
            "eor {p:v}.16b, {p:v}.16b, {x:v}.16b",
            "pmull2 {x:v}.1q, {s:v}.2d, {b:v}.2d",
            "eor {x:v}.16b, {x:v}.16b, {p:v}.16b",
            "pmull2 {p:v}.1q, {hh:v}.2d, {k:v}.2d",
            "eor {p:v}.16b, {p:v}.16b, {x:v}.16b",
            "zip1 {x:v}.2d, {ll:v}.2d, {p:v}.2d",
            "eor3 {x:v}.16b, {x:v}.16b, {e1:v}.16b, {e2:v}.16b",
            "pmull2 {hh:v}.1q, {p:v}.2d, {k:v}.2d",
            "eor {hh:v}.16b, {hh:v}.16b, {x:v}.16b",
            a = in(vreg) a,
            s = in(vreg) a_sw,
            b = in(vreg) b,
            k = in(vreg) k,
            e1 = in(vreg) e1,
            e2 = in(vreg) e2,
            ll = out(vreg) _,
            hh = out(vreg) out,
            x = out(vreg) _,
            p = out(vreg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// [`mul_sw_fused_xor3_q`] without the addends: `a*b` in 8 issue slots.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn mul_sw_fused_q(
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    k: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    // SAFETY: as `mul_sw_fused_xor3_q`.
    unsafe {
        core::arch::asm!(
            "pmull {ll:v}.1q, {a:v}.1d, {b:v}.1d",
            "pmull2 {hh:v}.1q, {a:v}.2d, {b:v}.2d",
            "ext {x:v}.16b, {ll:v}.16b, {hh:v}.16b, #8",
            "pmull {p:v}.1q, {s:v}.1d, {b:v}.1d",
            "eor {p:v}.16b, {p:v}.16b, {x:v}.16b",
            "pmull2 {x:v}.1q, {s:v}.2d, {b:v}.2d",
            "eor {x:v}.16b, {x:v}.16b, {p:v}.16b",
            "pmull2 {p:v}.1q, {hh:v}.2d, {k:v}.2d",
            "eor {p:v}.16b, {p:v}.16b, {x:v}.16b",
            "zip1 {x:v}.2d, {ll:v}.2d, {p:v}.2d",
            "pmull2 {hh:v}.1q, {p:v}.2d, {k:v}.2d",
            "eor {hh:v}.16b, {hh:v}.16b, {x:v}.16b",
            a = in(vreg) a,
            s = in(vreg) a_sw,
            b = in(vreg) b,
            k = in(vreg) k,
            ll = out(vreg) _,
            hh = out(vreg) out,
            x = out(vreg) _,
            p = out(vreg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// `acc += a*b` in [`TriNeon`] form with only the cross pair fused: `ext`,
/// the two cross `PMULL`s as fused pairs chained back into `cross`, and
/// plain `PMULL`+`EOR` for `ll`/`hh` so every accumulator stays in its own
/// register (no loop-carried moves, which the all-fused [`tri_acc_q`] needs).
/// 7 issue slots and 9 instructions vs [`tri_acc_q_compact`]'s 8 and 8.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn tri_acc_q_crossfused(
    acc: &mut TriNeon,
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) {
    // SAFETY: register-only; the scratch registers are early clobbers and
    // `cross` is overwritten only after its old value is folded into `p`.
    unsafe {
        core::arch::asm!(
            "ext {t:v}.16b, {a:v}.16b, {a:v}.16b, #8",
            "pmull {p:v}.1q, {t:v}.1d, {b:v}.1d",
            "eor {p:v}.16b, {p:v}.16b, {cr:v}.16b",
            "pmull2 {cr:v}.1q, {t:v}.2d, {b:v}.2d",
            "eor {cr:v}.16b, {cr:v}.16b, {p:v}.16b",
            "pmull {t:v}.1q, {a:v}.1d, {b:v}.1d",
            "pmull2 {p:v}.1q, {a:v}.2d, {b:v}.2d",
            "eor {ll:v}.16b, {ll:v}.16b, {t:v}.16b",
            "eor {hh:v}.16b, {hh:v}.16b, {p:v}.16b",
            a = in(vreg) a,
            b = in(vreg) b,
            t = out(vreg) _,
            p = out(vreg) _,
            ll = inout(vreg) acc.ll,
            cr = inout(vreg) acc.cross,
            hh = inout(vreg) acc.hh,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
}

/// `init` XOR the eight byte-lane entries of `code` in the 8 x 4 KiB `table`
/// (four `EOR3`s).
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn lookup8_acc_q(
    table: *const u8,
    code: u64,
    init: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe {
        const STRIDE: usize = 256 * 16;
        macro_rules! bank {
            ($lane:literal) => {
                bank_load_q::<{ 8 * $lane }>(table.add($lane * STRIDE), code)
            };
        }
        let acc = xor3_u64(init, bank!(0), bank!(1));
        let acc = xor3_u64(acc, bank!(2), bank!(3));
        let acc = xor3_u64(acc, bank!(4), bank!(5));
        xor3_u64(acc, bank!(6), bank!(7))
    }
}

/// The eight byte-lane entries of `code` XORed (three `EOR3`s and an `EOR`).
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn lookup8_q(table: *const u8, code: u64) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        macro_rules! bank {
            ($lane:literal) => {
                bank_load_q::<{ 8 * $lane }>(table.add($lane * STRIDE), code)
            };
        }
        let acc = xor3_u64(bank!(0), bank!(1), bank!(2));
        let acc = xor3_u64(acc, bank!(3), bank!(4));
        let acc = xor3_u64(acc, bank!(5), bank!(6));
        veorq_u64(acc, bank!(7))
    }
}

/// [`fold2_compact_and_round45_chunk_neon_8`] with the same stores and the
/// same eight reduced slots, rebuilt around the fused PMULL+EOR forms the
/// one-body kernel could not afford next to its compact accumulators:
///
/// - each lane's two `rho2` products are [`mul_sw_fused_xor3_q`] with the
///   anchor and both lookup trees as its addends and `0x87` hoisted into a
///   register (the incumbent re-`DUP`s it from a GPR for every product);
/// - the four `w`-scaled rows are fused schoolbook [`mul_sw_fused_q`] sharing
///   one swap of `w`, replacing the Karatsuba middle terms that went through
///   GPRs (two lane moves, a scalar `EOR` and a move back per row);
/// - the eight products use [`tri_acc_q_crossfused`], whose accumulators
///   stay in place.
///
/// M4 P-core, real ranked compact state, 2^12 pairs per chunk, dead tail:
/// 18.1–18.4 vs 20.9–21.4 cycles per K output (paired 0.86–0.87, 31/31);
/// the fold's ~124 table lookups per group are now the bound (fold alone
/// 14.5). Every product is the incumbent's or a field-equal one, and the tri
/// sums are reduced once, so the output bytes and slots are identical.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold2_compact_and_round45_chunk_neon_8_fused<const DEAD_TAIL: bool>(
    table_l1: *const u8,
    table_l3: *const u8,
    rho2: F128,
    anchors: *const F128,
    deltas: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    out_pairs: usize,
    degen: bool,
    out: *mut F128,
) {
    use core::arch::aarch64::*;

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
        let k87 = vdupq_n_u64(0x87);
        let rho2_q = core::mem::transmute::<F128, uint64x2_t>(rho2);
        let rho2_sw = swap_halves_q(rho2_q);
        let one_q = core::mem::transmute::<F128, uint64x2_t>(F128::ONE);
        // Slots: p1_even, pinf_even, p1_odd, pinf_odd, W0', W3', W4', W5'.
        let mut acc = LookaheadTriAcc::zero();

        debug_assert!(out_pairs >= 2 && out_pairs.is_multiple_of(2));
        let n5 = out_pairs / 2;

        // One dense lane `g`: the incumbent's a/b values (and its `b_flat`
        // bookkeeping), each `rho2` product fused with its addends.
        macro_rules! fold_lane {
            ($g:expr, $b_flat:ident) => {{
                let g = $g;
                let ap = anchors.add(4 * g).cast::<u64>();
                let anc_a0 = vld1q_u64(ap);
                let anc_b0 = vld1q_u64(ap.add(2));
                let anc_a1 = vld1q_u64(ap.add(4));
                let anc_b1 = vld1q_u64(ap.add(6));
                let dp = deltas.add(32 * g);
                let da0 = u64::from_le(core::ptr::read_unaligned(dp.cast::<u64>()));
                let db0 = u64::from_le(core::ptr::read_unaligned(dp.add(8).cast::<u64>()));
                let da1 = u64::from_le(core::ptr::read_unaligned(dp.add(16).cast::<u64>()));
                let db1 = u64::from_le(core::ptr::read_unaligned(dp.add(24).cast::<u64>()));

                // anc_a0 + rho2*(anc_a0 + anc_a1) + T_l1(da0) + T_l3(da1).
                let av = mul_sw_fused_xor3_q(
                    rho2_q,
                    rho2_sw,
                    veorq_u64(anc_a0, anc_a1),
                    k87,
                    lookup8_acc_q(table_l1, da0, anc_a0),
                    lookup8_q(table_l3, da1),
                );
                let bv = if degen && (db0 | db1) == 0 {
                    let bd = veorq_u64(anc_b0, anc_b1);
                    if is_zero_q(bd) {
                        anc_b0
                    } else {
                        $b_flat = false;
                        mul_sw_fused_xor3_q(rho2_q, rho2_sw, bd, k87, anc_b0, zero)
                    }
                } else {
                    $b_flat = false;
                    mul_sw_fused_xor3_q(
                        rho2_q,
                        rho2_sw,
                        veorq_u64(anc_b0, anc_b1),
                        k87,
                        lookup8_acc_q(table_l1, db0, anc_b0),
                        lookup8_q(table_l3, db1),
                    )
                };
                (av, bv)
            }};
        }

        macro_rules! k_group {
            ($t:expr) => {{
                let t = $t;
                let mut b_flat = true;
                let (a0, b0) = fold_lane!(4 * t, b_flat);
                let (a1, b1) = fold_lane!(4 * t + 1, b_flat);
                store_pair_nt(a_out.add(4 * t), a0, a1);
                store_pair_nt(b_out.add(4 * t), b0, b1);
                let (a2, b2) = fold_lane!(4 * t + 2, b_flat);
                let (a3, b3) = fold_lane!(4 * t + 3, b_flat);
                store_pair_nt(a_out.add(4 * t + 2), a2, a3);
                store_pair_nt(b_out.add(4 * t + 2), b2, b3);
                // The odd round-4 pair's weight drives the whole group.
                let w = vld1q_u64(eq_lo.add(2 * t + 1).cast::<u64>());
                let acc = &mut acc.0;
                if b_flat
                    && is_zero_q(vorrq_u64(
                        vorrq_u64(veorq_u64(b0, one_q), veorq_u64(b1, one_q)),
                        vorrq_u64(veorq_u64(b2, one_q), veorq_u64(b3, one_q)),
                    ))
                {
                    // b ≡ 1: the three surviving products are w·a1, w·a3, w·a2.
                    tri_acc_q_crossfused(&mut acc[0], w, a1);
                    tri_acc_q_crossfused(&mut acc[2], w, a3);
                    tri_acc_q_crossfused(&mut acc[4], w, a2);
                } else {
                    let w_sw = swap_halves_q(w);
                    let a0w = mul_sw_fused_q(w, w_sw, a0, k87);
                    let a1w = mul_sw_fused_q(w, w_sw, a1, k87);
                    let a2w = mul_sw_fused_q(w, w_sw, a2, k87);
                    let a3w = mul_sw_fused_q(w, w_sw, a3, k87);
                    tri_acc_q_crossfused(&mut acc[0], a1w, b1);
                    tri_acc_q_crossfused(&mut acc[1], veorq_u64(a0w, a1w), veorq_u64(b0, b1));
                    tri_acc_q_crossfused(&mut acc[2], a3w, b3);
                    tri_acc_q_crossfused(&mut acc[3], veorq_u64(a2w, a3w), veorq_u64(b2, b3));
                    tri_acc_q_crossfused(&mut acc[4], a2w, b2);
                    let (e_aw, e_b) = (veorq_u64(a0w, a2w), veorq_u64(b0, b2));
                    let (o_aw, o_b) = (veorq_u64(a1w, a3w), veorq_u64(b1, b3));
                    tri_acc_q_crossfused(&mut acc[5], e_aw, e_b);
                    tri_acc_q_crossfused(&mut acc[6], o_aw, o_b);
                    tri_acc_q_crossfused(&mut acc[7], veorq_u64(e_aw, o_aw), veorq_u64(e_b, o_b));
                }
            }};
        }

        if DEAD_TAIL {
            // Same dead-pair geometry as the incumbent: in every 128-pair
            // block the last round-five group holds only lane 0's live pair.
            debug_assert!(n5.is_multiple_of(16));
            for blk in 0..n5 / 16 {
                let t0 = 16 * blk;
                for t in t0..t0 + 15 {
                    k_group!(t);
                }
                let t = t0 + 15;
                let g = 4 * t;
                let ap = anchors.add(4 * g).cast::<u64>();
                let anc_a0 = vld1q_u64(ap);
                let anc_b0 = vld1q_u64(ap.add(2));
                let dp = deltas.add(32 * g);
                let da0 = u64::from_le(core::ptr::read_unaligned(dp.cast::<u64>()));
                let db0 = u64::from_le(core::ptr::read_unaligned(dp.add(8).cast::<u64>()));
                // anc0 + rho2*anc0 + T_l1(delta0), as the incumbent.
                let av0 = mul_sw_fused_xor3_q(
                    rho2_q,
                    rho2_sw,
                    anc_a0,
                    k87,
                    anc_a0,
                    lookup8_q(table_l1, da0),
                );
                let bv0 = mul_sw_fused_xor3_q(
                    rho2_q,
                    rho2_sw,
                    anc_b0,
                    k87,
                    anc_b0,
                    lookup8_q(table_l1, db0),
                );
                store_pair_nt(a_out.add(4 * t), av0, zero);
                store_pair_nt(a_out.add(4 * t + 2), zero, zero);
                store_pair_nt(b_out.add(4 * t), bv0, zero);
                store_pair_nt(b_out.add(4 * t + 2), zero, zero);
                // Lanes 1..3 are zero: (w·a0)·b0 into pinf_even, W3' and W5'.
                let w = vld1q_u64(eq_lo.add(2 * t + 1).cast::<u64>());
                let a0w = mul_sw_fused_q(w, swap_halves_q(w), av0, k87);
                // All three slots add the same raw product.
                let (ll, cross, hh) = clmul_products_q(a0w, bv0);
                for slot in [1usize, 5, 7] {
                    let t = &mut acc.0[slot];
                    t.ll = veorq_u64(t.ll, ll);
                    t.cross = veorq_u64(t.cross, cross);
                    t.hh = veorq_u64(t.hh, hh);
                }
            }
        } else {
            for t in 0..n5 {
                k_group!(t);
            }
        }

        let sums = acc.reduce();
        for (k, v) in sums.into_iter().enumerate() {
            *out.add(k) = v;
        }
    }
}

/// Blocks of 64 K outputs per superblock of the dead-tail-packed layout.
pub(crate) const PACK_BLOCKS: usize = 4;
/// Entries per superblock of the dead-tail-packed layout: four blocks'
/// outputs 0..60 (block `w` at `60 * w`), then their four outputs 60 (at
/// `240 + w`); outputs 61..63, which are zero, have no entry. 244 entries
/// are 61 cache lines, so every superblock starts on a line.
pub(crate) const PACK_SUPER: usize = 61 * PACK_BLOCKS;

/// `a*b + e1` in the field: [`mul_sw_fused_xor3_q`] with one addend (`eor`
/// in place of the `eor3`). Same four schoolbook `PMULL`s, same two-step
/// reduction, same field value; 13 instructions.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn mul_sw_fused_xor1_q(
    a: core::arch::aarch64::uint64x2_t,
    a_sw: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    k: core::arch::aarch64::uint64x2_t,
    e1: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    // SAFETY: as `mul_sw_fused_xor3_q`.
    unsafe {
        core::arch::asm!(
            "pmull {ll:v}.1q, {a:v}.1d, {b:v}.1d",
            "pmull2 {hh:v}.1q, {a:v}.2d, {b:v}.2d",
            "ext {x:v}.16b, {ll:v}.16b, {hh:v}.16b, #8",
            "pmull {p:v}.1q, {s:v}.1d, {b:v}.1d",
            "eor {p:v}.16b, {p:v}.16b, {x:v}.16b",
            "pmull2 {x:v}.1q, {s:v}.2d, {b:v}.2d",
            "eor {x:v}.16b, {x:v}.16b, {p:v}.16b",
            "pmull2 {p:v}.1q, {hh:v}.2d, {k:v}.2d",
            "eor {p:v}.16b, {p:v}.16b, {x:v}.16b",
            "zip1 {x:v}.2d, {ll:v}.2d, {p:v}.2d",
            "eor {x:v}.16b, {x:v}.16b, {e1:v}.16b",
            "pmull2 {hh:v}.1q, {p:v}.2d, {k:v}.2d",
            "eor {hh:v}.16b, {hh:v}.16b, {x:v}.16b",
            a = in(vreg) a,
            s = in(vreg) a_sw,
            b = in(vreg) b,
            k = in(vreg) k,
            e1 = in(vreg) e1,
            ll = out(vreg) _,
            hh = out(vreg) out,
            x = out(vreg) _,
            p = out(vreg) _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

///
/// The incumbent lane value is `anc0 + ρ₂(anc0+anc1) + T_λ₁(δ0) + T_λ₃(δ1)`
/// with `λ₁ = ρ₁(1+ρ₂)`, `λ₃ = ρ₁ρ₂`. With `x0 = anc0 + T_ρ₁(δ0)` and
/// `x1 = anc1 + T_ρ₁(δ1)` (the two round-three values, `T_ρ₁ = ρ₁·T`, `T`
/// GF(2)-linear per byte) this is the sequential fold `x0 + ρ₂(x0 + x1)`:
/// `x0 + ρ₂(x0+x1) = anc0 + ρ₂(anc0+anc1) + ρ₁(1+ρ₂)T(δ0) + ρ₁ρ₂T(δ1)`.
/// Same field element, so every store and every product below is the
/// incumbent's. Per lane side: the same 16 lookups (two `lookup8_acc_q`
/// trees of four `EOR3` each instead of one `lookup8_acc_q` plus one
/// `lookup8_q`), one `EOR` (`x0^x1` instead of `anc0^anc1`), one fused
/// multiply with one addend (13 instructions, the incumbent's has two
/// addends in its `EOR3`). The table set shrinks from 16 to 8 bank bases
/// (64 → 32 KiB), which is what frees the general registers the incumbent
/// spills (`ldr x?, [sp, ..]` per lane side for one λ₃ bank base), and the
/// stores use immediate offsets (two address `ADD`s per group instead of
/// six). The message half is unchanged.
///
/// `PACK` (with `DEAD_TAIL` only): the outputs are written in the
/// dead-tail-packed layout ([`PACK_SUPER`]): outputs 61..63 of every block of
/// 64, which are zero, are not written. Same values, same slots. The chunk
/// must hold a whole number of four-block superblocks.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold2_compact_and_round45_chunk_neon_8_fused_r1<
    const DEAD_TAIL: bool,
    const PACK: bool,
>(
    table_r1: *const u8,
    rho2: F128,
    anchors: *const F128,
    deltas: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    out_pairs: usize,
    degen: bool,
    out: *mut F128,
) {
    const { assert!(DEAD_TAIL || !PACK) };
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_pair_nt_off<const OFF: usize>(dst: *mut F128, x: uint64x2_t, y: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {x:q}, {y:q}, [{dst}, #{off}]",
                dst = in(reg) dst,
                x = in(vreg) x,
                y = in(vreg) y,
                off = const OFF,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        let zero = vdupq_n_u64(0);
        let k87 = vdupq_n_u64(0x87);
        let rho2_q = core::mem::transmute::<F128, uint64x2_t>(rho2);
        let rho2_sw = swap_halves_q(rho2_q);
        let one_q = core::mem::transmute::<F128, uint64x2_t>(F128::ONE);
        // Slots: p1_even, pinf_even, p1_odd, pinf_odd, W0', W3', W4', W5'.
        let mut acc = LookaheadTriAcc::zero();

        debug_assert!(out_pairs >= 2 && out_pairs.is_multiple_of(2));
        let n5 = out_pairs / 2;

        macro_rules! fold_lane {
            ($g:expr, $b_flat:ident) => {{
                let g = $g;
                let ap = anchors.add(4 * g).cast::<u64>();
                let anc_a0 = vld1q_u64(ap);
                let anc_b0 = vld1q_u64(ap.add(2));
                let anc_a1 = vld1q_u64(ap.add(4));
                let anc_b1 = vld1q_u64(ap.add(6));
                let dp = deltas.add(32 * g);
                let da0 = u64::from_le(core::ptr::read_unaligned(dp.cast::<u64>()));
                let db0 = u64::from_le(core::ptr::read_unaligned(dp.add(8).cast::<u64>()));
                let da1 = u64::from_le(core::ptr::read_unaligned(dp.add(16).cast::<u64>()));
                let db1 = u64::from_le(core::ptr::read_unaligned(dp.add(24).cast::<u64>()));

                let xa0 = lookup8_acc_q(table_r1, da0, anc_a0);
                let xa1 = lookup8_acc_q(table_r1, da1, anc_a1);
                let av = mul_sw_fused_xor1_q(rho2_q, rho2_sw, veorq_u64(xa0, xa1), k87, xa0);
                let bv = if degen && (db0 | db1) == 0 {
                    let bd = veorq_u64(anc_b0, anc_b1);
                    if is_zero_q(bd) {
                        anc_b0
                    } else {
                        $b_flat = false;
                        mul_sw_fused_xor1_q(rho2_q, rho2_sw, bd, k87, anc_b0)
                    }
                } else {
                    $b_flat = false;
                    let xb0 = lookup8_acc_q(table_r1, db0, anc_b0);
                    let xb1 = lookup8_acc_q(table_r1, db1, anc_b1);
                    mul_sw_fused_xor1_q(rho2_q, rho2_sw, veorq_u64(xb0, xb1), k87, xb0)
                };
                (av, bv)
            }};
        }

        // `$o`: the index of the group's first output in `a_out` / `b_out`.
        macro_rules! k_group {
            ($t:expr, $o:expr) => {{
                let t = $t;
                let mut b_flat = true;
                let ao = a_out.add($o);
                let bo = b_out.add($o);
                let (a0, b0) = fold_lane!(4 * t, b_flat);
                let (a1, b1) = fold_lane!(4 * t + 1, b_flat);
                store_pair_nt_off::<0>(ao, a0, a1);
                store_pair_nt_off::<0>(bo, b0, b1);
                let (a2, b2) = fold_lane!(4 * t + 2, b_flat);
                let (a3, b3) = fold_lane!(4 * t + 3, b_flat);
                store_pair_nt_off::<32>(ao, a2, a3);
                store_pair_nt_off::<32>(bo, b2, b3);
                let w = vld1q_u64(eq_lo.add(2 * t + 1).cast::<u64>());
                let acc = &mut acc.0;
                if b_flat
                    && is_zero_q(vorrq_u64(
                        vorrq_u64(veorq_u64(b0, one_q), veorq_u64(b1, one_q)),
                        vorrq_u64(veorq_u64(b2, one_q), veorq_u64(b3, one_q)),
                    ))
                {
                    tri_acc_q_crossfused(&mut acc[0], w, a1);
                    tri_acc_q_crossfused(&mut acc[2], w, a3);
                    tri_acc_q_crossfused(&mut acc[4], w, a2);
                } else {
                    let w_sw = swap_halves_q(w);
                    let a0w = mul_sw_fused_q(w, w_sw, a0, k87);
                    let a1w = mul_sw_fused_q(w, w_sw, a1, k87);
                    let a2w = mul_sw_fused_q(w, w_sw, a2, k87);
                    let a3w = mul_sw_fused_q(w, w_sw, a3, k87);
                    tri_acc_q_crossfused(&mut acc[0], a1w, b1);
                    tri_acc_q_crossfused(&mut acc[1], veorq_u64(a0w, a1w), veorq_u64(b0, b1));
                    tri_acc_q_crossfused(&mut acc[2], a3w, b3);
                    tri_acc_q_crossfused(&mut acc[3], veorq_u64(a2w, a3w), veorq_u64(b2, b3));
                    tri_acc_q_crossfused(&mut acc[4], a2w, b2);
                    let (e_aw, e_b) = (veorq_u64(a0w, a2w), veorq_u64(b0, b2));
                    let (o_aw, o_b) = (veorq_u64(a1w, a3w), veorq_u64(b1, b3));
                    tri_acc_q_crossfused(&mut acc[5], e_aw, e_b);
                    tri_acc_q_crossfused(&mut acc[6], o_aw, o_b);
                    tri_acc_q_crossfused(&mut acc[7], veorq_u64(e_aw, o_aw), veorq_u64(e_b, o_b));
                }
            }};
        }

        // Packed: output 60 of each block is staged here until the
        // superblock's four are written as one 64-byte line.
        let mut stage_a = [zero; 4];
        let mut stage_b = [zero; 4];
        if DEAD_TAIL {
            debug_assert!(n5.is_multiple_of(16));
            debug_assert!(!PACK || (n5 / 16).is_multiple_of(PACK_BLOCKS));
            for blk in 0..n5 / 16 {
                let t0 = 16 * blk;
                // Packed: block `blk` starts at
                // `PACK_SUPER * (blk / 4) + 60 * (blk % 4)`.
                let o0 = if PACK {
                    PACK_SUPER * (blk / PACK_BLOCKS) + 60 * (blk % PACK_BLOCKS)
                } else {
                    4 * t0
                };
                for t in t0..t0 + 15 {
                    k_group!(t, o0 + 4 * (t - t0));
                }
                let t = t0 + 15;
                let g = 4 * t;
                let ap = anchors.add(4 * g).cast::<u64>();
                let anc_a0 = vld1q_u64(ap);
                let anc_b0 = vld1q_u64(ap.add(2));
                let dp = deltas.add(32 * g);
                let da0 = u64::from_le(core::ptr::read_unaligned(dp.cast::<u64>()));
                let db0 = u64::from_le(core::ptr::read_unaligned(dp.add(8).cast::<u64>()));
                let xa0 = lookup8_acc_q(table_r1, da0, anc_a0);
                let xb0 = lookup8_acc_q(table_r1, db0, anc_b0);
                let av0 = mul_sw_fused_xor1_q(rho2_q, rho2_sw, xa0, k87, xa0);
                let bv0 = mul_sw_fused_xor1_q(rho2_q, rho2_sw, xb0, k87, xb0);
                if PACK {
                    let slot = blk % PACK_BLOCKS;
                    stage_a[slot] = av0;
                    stage_b[slot] = bv0;
                    if slot == PACK_BLOCKS - 1 {
                        let line = PACK_SUPER * (blk / PACK_BLOCKS) + 60 * PACK_BLOCKS;
                        let (sa, sb) = (a_out.add(line), b_out.add(line));
                        store_pair_nt_off::<0>(sa, stage_a[0], stage_a[1]);
                        store_pair_nt_off::<32>(sa, stage_a[2], stage_a[3]);
                        store_pair_nt_off::<0>(sb, stage_b[0], stage_b[1]);
                        store_pair_nt_off::<32>(sb, stage_b[2], stage_b[3]);
                    }
                } else {
                    let ao = a_out.add(4 * t);
                    let bo = b_out.add(4 * t);
                    store_pair_nt_off::<0>(ao, av0, zero);
                    store_pair_nt_off::<32>(ao, zero, zero);
                    store_pair_nt_off::<0>(bo, bv0, zero);
                    store_pair_nt_off::<32>(bo, zero, zero);
                }
                let w = vld1q_u64(eq_lo.add(2 * t + 1).cast::<u64>());
                let a0w = mul_sw_fused_q(w, swap_halves_q(w), av0, k87);
                let (ll, cross, hh) = clmul_products_q(a0w, bv0);
                for slot in [1usize, 5, 7] {
                    let t = &mut acc.0[slot];
                    t.ll = veorq_u64(t.ll, ll);
                    t.cross = veorq_u64(t.cross, cross);
                    t.hh = veorq_u64(t.hh, hh);
                }
            }
        } else {
            for t in 0..n5 {
                k_group!(t, 4 * t);
            }
        }

        let sums = acc.reduce();
        for (k, v) in sums.into_iter().enumerate() {
            *out.add(k) = v;
        }
    }
}

/// XOR of the `L` table entries for byte lanes `lane0..lane0 + L` of `code`.
/// One 16-byte L1 load per lane; the tree shape matches `fold_row_q` so the
/// full-width (`L = 8`, `lane0 = 0`) case is instruction-equivalent.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn lookup_lanes_q<const L: usize>(
    table_data: *const u8,
    code: u64,
    lane0: usize,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        let load = |j: usize| {
            let lane = lane0 + j;
            let index = ((code >> (8 * lane)) & 0xff) as usize;
            vld1q_u64(
                table_data
                    .add(lane * STRIDE + index * core::mem::size_of::<F128>())
                    .cast::<u64>(),
            )
        };
        if L == 8 && lane0 == 0 {
            // Every full-width caller passes a literal 0, so after inlining
            // this is the only path: the same eight loads in the same XOR
            // tree through the two-instruction lookup of `bank_load_q`.
            macro_rules! bank {
                ($lane:literal) => {
                    bank_load_q::<{ 8 * $lane }>(table_data.add($lane * STRIDE), code)
                };
            }
            // Byte 0 goes through the same helper unless the parent's load
            // form is selected, which keeps the parent's plain expression.
            let first = if Q_LOAD_FORM == 0 {
                load(0)
            } else {
                bank!(0)
            };
            let acc = xor3_u64(first, bank!(1), bank!(2));
            let acc = xor3_u64(acc, bank!(3), bank!(4));
            let acc = xor3_u64(acc, bank!(5), bank!(6));
            return veorq_u64(acc, bank!(7));
        }
        let mut acc = load(0);
        let mut j = 1;
        while j + 1 < L {
            acc = xor3_u64(acc, load(j), load(j + 1));
            j += 2;
        }
        if j < L {
            acc = veorq_u64(acc, load(j));
        }
        acc
    }
}

/// Pairs per tile for the byte-lane-outer streaming kernels. Accumulator
/// footprint is `4 × 16 B × STREAM_TILE_PAIRS` (8 KiB) plus the row codes
/// (4 KiB) — L1-resident alongside the active 4·L KiB of table rows.
#[cfg(target_arch = "aarch64")]
const STREAM_TILE_PAIRS: usize = 128;

/// Byte-lane-outer streaming variant of
/// [`fold_round2_compact_chunk_neon_unchecked_8`]. Identical outputs (XOR
/// reassociation only). `L` byte lanes are consumed per pass; each tile of
/// [`STREAM_TILE_PAIRS`] pairs keeps its four fold accumulators in an
/// L1-resident stack array, re-reading the packed row codes once per pass.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_round2_compact_stream_chunk_neon<const L: usize>(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_anchor_pair_nt(dst: *mut F128, a: uint64x2_t, b: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{dst}]",
                dst = in(reg) dst,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack, preserves_flags),
            );
        }
    }

    debug_assert!(L == 1 || L == 2 || L == 4 || L == 8);
    let n_passes = 8 / L;

    unsafe {
        let zero = vdupq_n_u64(0);
        let mut p1_acc = WideNeon { lo: zero, hi: zero };
        let mut pinf_acc = WideNeon { lo: zero, hi: zero };
        // Per-tile fold accumulators: [a0, a1, b0, b1] per pair.
        let mut acc = [[zero; 4]; STREAM_TILE_PAIRS];

        let mut t0 = 0usize;
        while t0 < lo_size {
            let tlen = STREAM_TILE_PAIRS.min(lo_size - t0);

            for pass in 0..n_passes {
                let lane0 = pass * L;
                for i in 0..tlen {
                    let x_lo = t0 + i;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        if pass == 0 {
                            store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                            vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                            acc[i] = [zero; 4];
                        }
                        continue;
                    }
                    let row0 = 2 * x_lo;
                    let a0_code =
                        u64::from_le(core::ptr::read_unaligned(a_packed.add(row0 * 8).cast()));
                    let a1_code = u64::from_le(core::ptr::read_unaligned(
                        a_packed.add((row0 + 1) * 8).cast(),
                    ));
                    let b0_code =
                        u64::from_le(core::ptr::read_unaligned(b_packed.add(row0 * 8).cast()));
                    let b1_code = u64::from_le(core::ptr::read_unaligned(
                        b_packed.add((row0 + 1) * 8).cast(),
                    ));

                    let va0 = lookup_lanes_q::<L>(table_data, a0_code, lane0);
                    let va1 = lookup_lanes_q::<L>(table_data, a1_code, lane0);
                    let vb0 = lookup_lanes_q::<L>(table_data, b0_code, lane0);
                    let vb1 = lookup_lanes_q::<L>(table_data, b1_code, lane0);

                    if pass == 0 {
                        acc[i] = [va0, va1, vb0, vb1];
                        let delta_pair = core::mem::transmute::<[u64; 2], uint64x2_t>([
                            a0_code ^ a1_code,
                            b0_code ^ b1_code,
                        ]);
                        vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);
                    } else {
                        acc[i][0] = veorq_u64(acc[i][0], va0);
                        acc[i][1] = veorq_u64(acc[i][1], va1);
                        acc[i][2] = veorq_u64(acc[i][2], vb0);
                        acc[i][3] = veorq_u64(acc[i][3], vb1);
                    }
                }
            }

            // Finalize the tile: anchor stores + round message.
            for i in 0..tlen {
                let x_lo = t0 + i;
                if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                    continue;
                }
                let [a0, a1, b0, b1] = acc[i];
                store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b0);
                let g1 = mul_q(a1, b1);
                let g_inf = mul_q(veorq_u64(a0, a1), veorq_u64(b0, b1));
                let eq_l = vld1q_u64(eq_lo.add(x_lo).cast::<u64>());
                wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
                wide_xor(&mut pinf_acc, mul_unreduced_q(eq_l, g_inf));
            }

            t0 += tlen;
        }

        (
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1_acc)),
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf_acc)),
        )
    }
}

/// Byte-lane-outer streaming variant of [`fold_compact_chunk_neon_unchecked_8`]
/// (compact reconstruction). Identical outputs; same tile/pass schedule as
/// [`fold_round2_compact_stream_chunk_neon`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_compact_stream_chunk_neon<const L: usize>(
    scaled_table: *const u8,
    anchors: *const F128,
    deltas: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    lo_size: usize,
) -> (F128, F128) {
    use core::arch::aarch64::*;

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

    debug_assert!(L == 1 || L == 2 || L == 4 || L == 8);
    let n_passes = 8 / L;

    unsafe {
        let zero = vdupq_n_u64(0);
        let mut p1_acc = WideNeon { lo: zero, hi: zero };
        let mut pinf_acc = WideNeon { lo: zero, hi: zero };
        // Per-tile delta-fold accumulators: [a0, a1, b0, b1] per pair.
        let mut acc = [[zero; 4]; STREAM_TILE_PAIRS];

        let mut t0 = 0usize;
        while t0 < lo_size {
            let tlen = STREAM_TILE_PAIRS.min(lo_size - t0);

            for pass in 0..n_passes {
                let lane0 = pass * L;
                for i in 0..tlen {
                    let x_lo = t0 + i;
                    let out = 2 * x_lo;
                    let delta = deltas.add(out * 16);
                    let a0_code = u64::from_le(core::ptr::read_unaligned(delta.cast::<u64>()));
                    let b0_code =
                        u64::from_le(core::ptr::read_unaligned(delta.add(8).cast::<u64>()));
                    let a1_code =
                        u64::from_le(core::ptr::read_unaligned(delta.add(16).cast::<u64>()));
                    let b1_code =
                        u64::from_le(core::ptr::read_unaligned(delta.add(24).cast::<u64>()));

                    let va0 = lookup_lanes_q::<L>(scaled_table, a0_code, lane0);
                    let va1 = lookup_lanes_q::<L>(scaled_table, a1_code, lane0);
                    let vb0 = lookup_lanes_q::<L>(scaled_table, b0_code, lane0);
                    let vb1 = lookup_lanes_q::<L>(scaled_table, b1_code, lane0);

                    if pass == 0 {
                        acc[i] = [va0, va1, vb0, vb1];
                    } else {
                        acc[i][0] = veorq_u64(acc[i][0], va0);
                        acc[i][1] = veorq_u64(acc[i][1], va1);
                        acc[i][2] = veorq_u64(acc[i][2], vb0);
                        acc[i][3] = veorq_u64(acc[i][3], vb1);
                    }
                }
            }

            // Finalize: add anchors, store outputs, accumulate the message.
            for i in 0..tlen {
                let x_lo = t0 + i;
                let out = 2 * x_lo;
                let a0 = veorq_u64(vld1q_u64(anchors.add(2 * out).cast::<u64>()), acc[i][0]);
                let a1 = veorq_u64(
                    vld1q_u64(anchors.add(2 * (out + 1)).cast::<u64>()),
                    acc[i][1],
                );
                let b0 = veorq_u64(vld1q_u64(anchors.add(2 * out + 1).cast::<u64>()), acc[i][2]);
                let b1 = veorq_u64(
                    vld1q_u64(anchors.add(2 * (out + 1) + 1).cast::<u64>()),
                    acc[i][3],
                );

                store_pair_nt(a_out.add(out), a0, a1);
                store_pair_nt(b_out.add(out), b0, b1);

                let g1 = mul_q(a1, b1);
                let g_inf = mul_q(veorq_u64(a0, a1), veorq_u64(b0, b1));
                let eq_l = vld1q_u64(eq_lo.add(x_lo).cast::<u64>());
                wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
                wide_xor(&mut pinf_acc, mul_unreduced_q(eq_l, g_inf));
            }

            t0 += tlen;
        }

        (
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1_acc)),
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf_acc)),
        )
    }
}

/// Reconstruction-only sibling of [`fold_compact_chunk_neon_unchecked_8`]:
/// writes byte-identical `a_out`/`b_out` values and skips the message
/// products entirely. Used for the chunk prefix whose products the GPU T3
/// arm owns — the CPU must still materialize every reconstructed value for
/// the later tail rounds, but the two message muls and two eq-weight muls
/// per pair move to the GPU.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_compact_chunk_neon_reconstruct_only_8(
    scaled_table: *const u8,
    anchors: *const F128,
    deltas: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    lo_size: usize,
) {
    use core::arch::aarch64::*;

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
        for x_lo in 0..lo_size {
            let out = 2 * x_lo;
            let delta = deltas.add(out * 16);
            let a0_code = u64::from_le(core::ptr::read_unaligned(delta.cast::<u64>()));
            let b0_code = u64::from_le(core::ptr::read_unaligned(delta.add(8).cast::<u64>()));
            let a1_code = u64::from_le(core::ptr::read_unaligned(delta.add(16).cast::<u64>()));
            let b1_code = u64::from_le(core::ptr::read_unaligned(delta.add(24).cast::<u64>()));

            let a0 = veorq_u64(
                vld1q_u64(anchors.add(2 * out).cast::<u64>()),
                lookup_lanes_q::<8>(scaled_table, a0_code, 0),
            );
            let b0 = veorq_u64(
                vld1q_u64(anchors.add(2 * out + 1).cast::<u64>()),
                lookup_lanes_q::<8>(scaled_table, b0_code, 0),
            );
            let a1 = veorq_u64(
                vld1q_u64(anchors.add(2 * (out + 1)).cast::<u64>()),
                lookup_lanes_q::<8>(scaled_table, a1_code, 0),
            );
            let b1 = veorq_u64(
                vld1q_u64(anchors.add(2 * (out + 1) + 1).cast::<u64>()),
                lookup_lanes_q::<8>(scaled_table, b1_code, 0),
            );

            store_pair_nt(a_out.add(out), a0, a1);
            store_pair_nt(b_out.add(out), b0, b1);
        }
    }
}

/// Reconstruct one compact round-two level at the sampled challenge and form
/// the next sumcheck message. `scaled_table` is the univariate fold table with
/// every entry multiplied by that challenge, so reconstruction needs only
/// cache-resident table loads and XORs:
///
/// `folded = anchor + scaled_table_fold(packed_row0 XOR packed_row1)`.
///
/// `degen` enables the b≡1 chunk-class degeneration: pairs whose two b delta
/// codes are both zero reconstruct `b = anchor` with no b-table lookups; the
/// value-forced shortcuts (`b0 + b1 = 0` ⇒ skip the `G(∞)` mul chain,
/// `b1 = 1` ⇒ `a1·b1 = a1`) are additionally gated on the loaded anchor
/// values, so they are bit-exact by construction.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_compact_chunk_neon_unchecked_8(
    scaled_table: *const u8,
    anchors: *const F128,
    deltas: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    lo_size: usize,
    degen: bool,
) -> (F128, F128) {
    use core::arch::aarch64::*;

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
        let mut p1_acc = WideNeon { lo: zero, hi: zero };
        let mut pinf_acc = WideNeon { lo: zero, hi: zero };

        let one_q = core::mem::transmute::<F128, uint64x2_t>(F128::ONE);

        for x_lo in 0..lo_size {
            let out = 2 * x_lo;
            let delta = deltas.add(out * 16);
            let a0_code = u64::from_le(core::ptr::read_unaligned(delta.cast::<u64>()));
            let b0_code = u64::from_le(core::ptr::read_unaligned(delta.add(8).cast::<u64>()));
            let a1_code = u64::from_le(core::ptr::read_unaligned(delta.add(16).cast::<u64>()));
            let b1_code = u64::from_le(core::ptr::read_unaligned(delta.add(24).cast::<u64>()));

            if degen && (b0_code | b1_code) == 0 {
                // Zero b deltas: b rows fold to zero (table entry 0 is zero),
                // so b = anchor with no lookups. On the static b≡1 mass both
                // anchors are 1: G(∞)'s (b0 + b1) factor vanishes and
                // a1·b1 = a1; both shortcuts are value-gated below.
                let (a0_delta, a1_delta) = fold_two_row_codes_q(scaled_table, a0_code, a1_code);
                let a0 = veorq_u64(vld1q_u64(anchors.add(2 * out).cast::<u64>()), a0_delta);
                let a1 = veorq_u64(
                    vld1q_u64(anchors.add(2 * (out + 1)).cast::<u64>()),
                    a1_delta,
                );
                let b0 = vld1q_u64(anchors.add(2 * out + 1).cast::<u64>());
                let b1 = vld1q_u64(anchors.add(2 * (out + 1) + 1).cast::<u64>());

                store_pair_nt(a_out.add(out), a0, a1);
                store_pair_nt(b_out.add(out), b0, b1);

                let eq_l = vld1q_u64(eq_lo.add(x_lo).cast::<u64>());
                let g1 = if is_zero_q(veorq_u64(b1, one_q)) {
                    a1
                } else {
                    mul_q(a1, b1)
                };
                wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
                let b_sum = veorq_u64(b0, b1);
                if !is_zero_q(b_sum) {
                    let g_inf = mul_q(veorq_u64(a0, a1), b_sum);
                    wide_xor(&mut pinf_acc, mul_unreduced_q(eq_l, g_inf));
                }
                continue;
            }

            let (a0_delta, a1_delta, b0_delta, b1_delta) =
                fold_four_row_codes_q(scaled_table, a0_code, a1_code, b0_code, b1_code);
            let a0 = veorq_u64(vld1q_u64(anchors.add(2 * out).cast::<u64>()), a0_delta);
            let a1 = veorq_u64(
                vld1q_u64(anchors.add(2 * (out + 1)).cast::<u64>()),
                a1_delta,
            );
            let b0 = veorq_u64(vld1q_u64(anchors.add(2 * out + 1).cast::<u64>()), b0_delta);
            let b1 = veorq_u64(
                vld1q_u64(anchors.add(2 * (out + 1) + 1).cast::<u64>()),
                b1_delta,
            );

            store_pair_nt(a_out.add(out), a0, a1);
            store_pair_nt(b_out.add(out), b0, b1);

            let g1 = mul_q(a1, b1);
            let g_inf = mul_q(veorq_u64(a0, a1), veorq_u64(b0, b1));
            let eq_l = vld1q_u64(eq_lo.add(x_lo).cast::<u64>());
            wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
            wide_xor(&mut pinf_acc, mul_unreduced_q(eq_l, g_inf));
        }

        (
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1_acc)),
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf_acc)),
        )
    }
}

/// NEON one-row fold: 8 aligned 16-byte loads + 8 XORs, hand-unrolled for
/// `n_chunks = 8` (the k_skip=6 protocol size). Returns the folded F128.
///
/// The table is `Vec<F128>` with each entry 16-byte aligned (F128 is
/// `repr(C, align(16))`), so every `vld1q_u8` lands on an aligned address.
///
/// # Safety
/// Caller must guarantee `table_data` points to ≥ 8 × 256 × 16 valid bytes
/// (an `n_chunks ≥ 8` table) and `bytes_ptr` to ≥ 8 valid bytes.
#[cfg(all(target_arch = "aarch64", test))]
#[inline(always)]
pub(crate) unsafe fn fold_one_row_neon_unchecked_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> F128 {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        let mut acc = vld1q_u8(table_data.add((*bytes_ptr) as usize * 16));
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(1 * STRIDE + (*bytes_ptr.add(1)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(2 * STRIDE + (*bytes_ptr.add(2)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(3 * STRIDE + (*bytes_ptr.add(3)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(4 * STRIDE + (*bytes_ptr.add(4)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(5 * STRIDE + (*bytes_ptr.add(5)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(6 * STRIDE + (*bytes_ptr.add(6)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(7 * STRIDE + (*bytes_ptr.add(7)) as usize * 16)),
        );
        let acc_u64 = vreinterpretq_u64_u8(acc);
        F128 {
            lo: vgetq_lane_u64::<0>(acc_u64),
            hi: vgetq_lane_u64::<1>(acc_u64),
        }
    }
}

///
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
fn fold_two_by_rho(even0: F128, odd0: F128, even1: F128, odd1: F128, rho: F128) -> (F128, F128) {
    let products = unsafe {
        crate::field::gf2_128::aarch64::ghash_mul_const_vec2_neon(rho, [even0 + odd0, even1 + odd1])
    };
    (even0 + products[0], even1 + products[1])
}

#[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
#[inline(always)]
fn fold_two_by_rho(even0: F128, odd0: F128, even1: F128, odd1: F128, rho: F128) -> (F128, F128) {
    (
        even0 + rho * (even0 + odd0),
        even1 + rho * (even1 + odd1),
    )
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fold4_two(v: &[F128], i: usize, rho_a: F128, rho_b: F128, rho_ab: F128) -> (F128, F128) {
    #[cfg(target_feature = "aes")]
    {
        unsafe {
            let (x, y) = fold4_pair_expanded_binius_q(
                v.as_ptr().add(i),
                core::mem::transmute::<F128, core::arch::aarch64::uint64x2_t>(rho_a),
                core::mem::transmute::<F128, core::arch::aarch64::uint64x2_t>(rho_b),
                core::mem::transmute::<F128, core::arch::aarch64::uint64x2_t>(rho_ab),
            );
            return (
                core::mem::transmute::<core::arch::aarch64::uint64x2_t, F128>(x),
                core::mem::transmute::<core::arch::aarch64::uint64x2_t, F128>(y),
            );
        }
    }
    #[cfg(not(target_feature = "aes"))]
    {
        let _ = rho_ab;
        let one = |j: usize| {
            let t0 = v[j] + rho_a * (v[j] + v[j + 1]);
            let t1 = v[j + 2] + rho_a * (v[j + 2] + v[j + 3]);
            t0 + rho_b * (t0 + t1)
        };
        (one(i), one(i + 4))
    }
}

/// Fuse one multilinear tail fold with construction of the following round's
/// message. The previous AArch64 path first streamed all of `a_in`/`b_in` into
/// `a_out`/`b_out`, then immediately reread both outputs in a second pass.
/// Keeping each four-value folded pair live until its message contribution is
/// accumulated removes that full output readback while preserving the exact
/// canonical output tables for the next round.
#[cfg(target_arch = "aarch64")]
pub(crate) fn fold_and_message_aarch64(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
    nt_stores: bool,
) -> (F128, F128) {
    // `nt_stores` is decided once per round by the driver (round output past
    // LLC size ⇒ ping-pong writes not read until the next barrier ⇒ `stnp`
    // elides the write-allocate RFO reads; small rounds keep normal stores so
    // LLC-resident outputs stay hot). Per-chunk callers must not decide this
    // from their sub-slice length.
    if nt_stores {
        fold_and_message_body::<true>(a_in, b_in, a_out, b_out, r_fold, eq_lo)
    } else {
        fold_and_message_body::<false>(a_in, b_in, a_out, b_out, r_fold, eq_lo)
    }
}

#[inline(always)]
fn fold_and_message_body<const NT: bool>(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use core::arch::aarch64::uint64x2_t;

    debug_assert_eq!(a_in.len(), 2 * a_out.len());
    debug_assert_eq!(b_in.len(), 2 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());

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

    let mut p1_acc = ProdAcc::ZERO;
    let mut pinf_acc = ProdAcc::ZERO;

    let a_out_ptr = a_out.as_mut_ptr();
    let b_out_ptr = b_out.as_mut_ptr();

    for (x_lo, &eq_l) in eq_lo.iter().enumerate() {
        let i = 4 * x_lo;
        let o = 2 * x_lo;

        let a_even_0 = a_in[i];
        let a_odd_0 = a_in[i + 1];
        let a_even_1 = a_in[i + 2];
        let a_odd_1 = a_in[i + 3];
        let b_even_0 = b_in[i];
        let b_odd_0 = b_in[i + 1];
        let b_even_1 = b_in[i + 2];
        let b_odd_1 = b_in[i + 3];

        let (a0, a1) = fold_two_by_rho(a_even_0, a_odd_0, a_even_1, a_odd_1, r_fold);
        let (b0, b1) = fold_two_by_rho(b_even_0, b_odd_0, b_even_1, b_odd_1, r_fold);

        if NT {
            // SAFETY: `o + 1 < a_out.len()` by the len contract above; F128
            // is repr(C, align(16)) two u64s, bit-compatible with uint64x2_t;
            // each 32-byte pair store is 16-byte aligned and lands in this
            // iteration's disjoint output slots.
            unsafe {
                store_pair_nt(
                    a_out_ptr.add(o),
                    core::mem::transmute::<F128, uint64x2_t>(a0),
                    core::mem::transmute::<F128, uint64x2_t>(a1),
                );
                store_pair_nt(
                    b_out_ptr.add(o),
                    core::mem::transmute::<F128, uint64x2_t>(b0),
                    core::mem::transmute::<F128, uint64x2_t>(b1),
                );
            }
        } else {
            a_out[o] = a0;
            a_out[o + 1] = a1;
            b_out[o] = b0;
            b_out[o + 1] = b1;
        }

        p1_acc.fma(eq_l, a1 * b1);
        pinf_acc.fma(eq_l, (a0 + a1) * (b0 + b1));
    }

    (p1_acc.reduce(), pinf_acc.reduce())
}

/// Two-challenge composed variant of [`fold_and_message_aarch64`] (cascade):
/// binds `rho_a` then `rho_b` in ONE pass — quartering the tables instead of
/// halving them — and emits the following round's message from the composed
/// outputs. Replaces two sequential fused tail rounds: same fold ALU (three
/// reduced multiplies per output vs 2+1 across the two passes), but the
/// intermediate half-size tables are never written or re-read.
///
/// Value-identical to `fold(rho_a)` then `fold(rho_b)`: the composition
/// `t0 + rho_b·(t0+t1)` over the two rho_a-folded halves IS the sequential
/// result, term for term — no reassociation at all, just deleted traffic.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_and_message_aarch64(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
    nt_stores: bool,
) -> (F128, F128) {
    // Same NT policy as `fold_and_message_aarch64`: decided once per round by
    // the driver from the ROUND's output size, not per chunk.
    if nt_stores {
        fold2_and_message_body::<true>(a_in, b_in, a_out, b_out, rho_a, rho_b, eq_lo)
    } else {
        fold2_and_message_body::<false>(a_in, b_in, a_out, b_out, rho_a, rho_b, eq_lo)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fold2_and_message_body<const NT: bool>(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use core::arch::aarch64::uint64x2_t;

    debug_assert_eq!(a_in.len(), 4 * a_out.len());
    debug_assert_eq!(b_in.len(), 4 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());

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

    let rho_ab = rho_a * rho_b;

    let mut p1_acc = ProdAcc::ZERO;
    let mut pinf_acc = ProdAcc::ZERO;

    let a_out_ptr = a_out.as_mut_ptr();
    let b_out_ptr = b_out.as_mut_ptr();

    for (x_lo, &eq_l) in eq_lo.iter().enumerate() {
        let i = 8 * x_lo;
        let o = 2 * x_lo;

        let (a0, a1) = fold4_two(a_in, i, rho_a, rho_b, rho_ab);
        let (b0, b1) = fold4_two(b_in, i, rho_a, rho_b, rho_ab);

        if NT {
            // SAFETY: `o + 1 < a_out.len()` by the len contract above; F128
            // is repr(C, align(16)) two u64s, bit-compatible with uint64x2_t;
            // each 32-byte pair store is 16-byte aligned and lands in this
            // iteration's disjoint output slots.
            unsafe {
                store_pair_nt(
                    a_out_ptr.add(o),
                    core::mem::transmute::<F128, uint64x2_t>(a0),
                    core::mem::transmute::<F128, uint64x2_t>(a1),
                );
                store_pair_nt(
                    b_out_ptr.add(o),
                    core::mem::transmute::<F128, uint64x2_t>(b0),
                    core::mem::transmute::<F128, uint64x2_t>(b1),
                );
            }
        } else {
            a_out[o] = a0;
            a_out[o + 1] = a1;
            b_out[o] = b0;
            b_out[o + 1] = b1;
        }

        p1_acc.fma(eq_l, a1 * b1);
        pinf_acc.fma(eq_l, (a0 + a1) * (b0 + b1));
    }

    (p1_acc.reduce(), pinf_acc.reduce())
}

/// Ordinary-store direct composed pass using the expanded fold4 pair. This is
/// the final cascade rung (ranked rounds 11/12); the incumbent direct kernel
/// above and both E038/E039 lookahead specializations remain separate.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_and_message_normal_expanded_aarch64(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    rho_ab: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use core::arch::aarch64::*;

    let expected_out = eq_lo
        .len()
        .checked_mul(2)
        .expect("direct composed output length overflow");
    let expected_in = expected_out
        .checked_mul(4)
        .expect("direct composed input length overflow");
    assert_eq!(a_out.len(), expected_out);
    assert_eq!(b_out.len(), expected_out);
    assert_eq!(a_in.len(), expected_in);
    assert_eq!(b_in.len(), expected_in);

    let rho = unsafe {
        Fold4RhoQ::new(
            core::mem::transmute::<F128, uint64x2_t>(rho_a),
            core::mem::transmute::<F128, uint64x2_t>(rho_b),
            core::mem::transmute::<F128, uint64x2_t>(rho_ab),
        )
    };
    let mut p1_acc = ProdAcc::ZERO;
    let mut pinf_acc = ProdAcc::ZERO;

    for (x_lo, &eq_l) in eq_lo.iter().enumerate() {
        let i = 8 * x_lo;
        let o = 2 * x_lo;
        let (a0, a1, b0, b1) = unsafe {
            // SAFETY: the release-mode geometry assertions and loop bound put
            // both eight-value pairs wholly inside their input tables.
            let (a0, a1) =
                fold4_pair_expanded_tri_q(a_in.as_ptr().add(i), &rho);
            let (b0, b1) =
                fold4_pair_expanded_tri_q(b_in.as_ptr().add(i), &rho);
            (
                core::mem::transmute::<uint64x2_t, F128>(a0),
                core::mem::transmute::<uint64x2_t, F128>(a1),
                core::mem::transmute::<uint64x2_t, F128>(b0),
                core::mem::transmute::<uint64x2_t, F128>(b1),
            )
        };

        // Ordinary stores retain the 2 MiB combined output in cache for the
        // next serial tail round.
        a_out[o] = a0;
        a_out[o + 1] = a1;
        b_out[o] = b0;
        b_out[o + 1] = b1;

        // Exactly the incumbent direct message algebra.
        p1_acc.fma(eq_l, a1 * b1);
        pinf_acc.fma(eq_l, (a0 + a1) * (b0 + b1));
    }

    (p1_acc.reduce(), pinf_acc.reduce())
}

/// [`fold2_and_message_aarch64`] **plus** the deferred round-seven aggregates
/// (cascade level three). The composed outputs and every store are identical;
/// the sweep consumes composed outputs four at a time (two round-6 message
/// pairs = one round-7 group) on the shared odd-lane weight `eq_lo[2t+1]` and
/// returns the eight per-chunk accumulators
/// `[p1_even, pinf_even, p1_odd, pinf_odd, W0'', W3'', W4'', W5'']` — the
/// exact slot order of `fold2_compact_and_round45_chunk_neon_8`. The driver
/// reconstructs the round-six message via κ and rescales the six lookahead
/// aggregates by `r''⁻¹`, mirroring the cascade K pass.
#[inline(never)]
pub(crate) fn fold2_and_message_lookahead_aarch64(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
    nt_stores: bool,
) -> [F128; 8] {
    if nt_stores {
        fold2_and_message_lookahead_body::<true>(a_in, b_in, a_out, b_out, rho_a, rho_b, eq_lo)
    } else {
        fold2_and_message_lookahead_body::<false>(a_in, b_in, a_out, b_out, rho_a, rho_b, eq_lo)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fold2_and_message_lookahead_body<const NT: bool>(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
) -> [F128; 8] {
    use core::arch::aarch64::uint64x2_t;

    debug_assert_eq!(a_in.len(), 4 * a_out.len());
    debug_assert_eq!(b_in.len(), 4 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());
    debug_assert_eq!(eq_lo.len() % 2, 0);

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

    let rho_ab = rho_a * rho_b;

    let mut acc = [ProdAcc::ZERO; 8];

    let a_out_ptr = a_out.as_mut_ptr();
    let b_out_ptr = b_out.as_mut_ptr();

    for t in 0..eq_lo.len() / 2 {
        let i = 16 * t;
        let o = 4 * t;

        let (a0, a1) = fold4_two(a_in, i, rho_a, rho_b, rho_ab);
        let (a2, a3) = fold4_two(a_in, i + 8, rho_a, rho_b, rho_ab);
        let (b0, b1) = fold4_two(b_in, i, rho_a, rho_b, rho_ab);
        let (b2, b3) = fold4_two(b_in, i + 8, rho_a, rho_b, rho_ab);

        if NT {
            // SAFETY: `o + 3 < a_out.len()` by the len contract above; F128 is
            // repr(C, align(16)) two u64s, bit-compatible with uint64x2_t;
            // each 32-byte pair store is 16-byte aligned and lands in this
            // iteration's disjoint output slots.
            unsafe {
                store_pair_nt(
                    a_out_ptr.add(o),
                    core::mem::transmute::<F128, uint64x2_t>(a0),
                    core::mem::transmute::<F128, uint64x2_t>(a1),
                );
                store_pair_nt(
                    a_out_ptr.add(o + 2),
                    core::mem::transmute::<F128, uint64x2_t>(a2),
                    core::mem::transmute::<F128, uint64x2_t>(a3),
                );
                store_pair_nt(
                    b_out_ptr.add(o),
                    core::mem::transmute::<F128, uint64x2_t>(b0),
                    core::mem::transmute::<F128, uint64x2_t>(b1),
                );
                store_pair_nt(
                    b_out_ptr.add(o + 2),
                    core::mem::transmute::<F128, uint64x2_t>(b2),
                    core::mem::transmute::<F128, uint64x2_t>(b3),
                );
            }
        } else {
            a_out[o] = a0;
            a_out[o + 1] = a1;
            a_out[o + 2] = a2;
            a_out[o + 3] = a3;
            b_out[o] = b0;
            b_out[o + 1] = b1;
            b_out[o + 2] = b2;
            b_out[o + 3] = b3;
        }

        // One weight per round-7 group: the odd lane. Weighting the a-side
        // once makes `W1''`/`W2''` (the odd round-6 pair sums) free — the same
        // zero-extra-multiplies parity trick as the cascade K pass.
        let wt = eq_lo[2 * t + 1];
        let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
        acc[0].fma(a1w, b1);
        acc[1].fma(a0w + a1w, b0 + b1);
        acc[2].fma(a3w, b3);
        acc[3].fma(a2w + a3w, b2 + b3);
        acc[4].fma(a2w, b2);
        let (e_aw, e_b) = (a0w + a2w, b0 + b2);
        let (o_aw, o_b) = (a1w + a3w, b1 + b3);
        acc[5].fma(e_aw, e_b);
        acc[6].fma(o_aw, o_b);
        acc[7].fma(e_aw + o_aw, e_b + o_b);
    }

    [
        acc[0].reduce(),
        acc[1].reduce(),
        acc[2].reduce(),
        acc[3].reduce(),
        acc[4].reduce(),
        acc[5].reduce(),
        acc[6].reduce(),
        acc[7].reduce(),
    ]
}

/// Ordinary-store sibling of
/// [`fold2_and_message_lookahead_nt_expanded_aarch64`]. This intentionally has
/// its own loop body: the screened NT kernel remains unchanged, and this rung
/// can be inspected and rolled back without perturbing its code generation.
/// The ranked driver selects this only for rounds 7/8 and 9/10.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_and_message_lookahead_normal_expanded_aarch64(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    rho_ab: F128,
    eq_lo: &[F128],
) -> [F128; 8] {
    use core::arch::aarch64::*;

    let expected_out = eq_lo
        .len()
        .checked_mul(2)
        .expect("lookahead output length overflow");
    let expected_in = expected_out
        .checked_mul(4)
        .expect("lookahead input length overflow");
    assert_eq!(eq_lo.len() % 2, 0);
    assert_eq!(a_out.len(), expected_out);
    assert_eq!(b_out.len(), expected_out);
    assert_eq!(a_in.len(), expected_in);
    assert_eq!(b_in.len(), expected_in);

    let rho = unsafe {
        Fold4RhoQ::new(
            core::mem::transmute::<F128, uint64x2_t>(rho_a),
            core::mem::transmute::<F128, uint64x2_t>(rho_b),
            core::mem::transmute::<F128, uint64x2_t>(rho_ab),
        )
    };
    let mut acc = unsafe { LookaheadTriAcc::zero() };

    for t in 0..eq_lo.len() / 2 {
        let i = 16 * t;
        let o = 4 * t;

        let (a0, a1, a2, a3, b0, b1, b2, b3) = unsafe {
            // SAFETY: the geometry assertions and loop bound place each pair
            // wholly inside its corresponding 16-element source group.
            let (a0, a1) =
                fold4_pair_expanded_tri_q(a_in.as_ptr().add(i), &rho);
            let (a2, a3) =
                fold4_pair_expanded_tri_q(a_in.as_ptr().add(i + 8), &rho);
            let (b0, b1) =
                fold4_pair_expanded_tri_q(b_in.as_ptr().add(i), &rho);
            let (b2, b3) =
                fold4_pair_expanded_tri_q(b_in.as_ptr().add(i + 8), &rho);
            (
                core::mem::transmute::<uint64x2_t, F128>(a0),
                core::mem::transmute::<uint64x2_t, F128>(a1),
                core::mem::transmute::<uint64x2_t, F128>(a2),
                core::mem::transmute::<uint64x2_t, F128>(a3),
                core::mem::transmute::<uint64x2_t, F128>(b0),
                core::mem::transmute::<uint64x2_t, F128>(b1),
                core::mem::transmute::<uint64x2_t, F128>(b2),
                core::mem::transmute::<uint64x2_t, F128>(b3),
            )
        };

        // These ordinary stores intentionally mirror the incumbent false arm;
        // the output tables remain hot for the next LLC-resident cascade pass.
        a_out[o] = a0;
        a_out[o + 1] = a1;
        a_out[o + 2] = a2;
        a_out[o + 3] = a3;
        b_out[o] = b0;
        b_out[o + 1] = b1;
        b_out[o + 2] = b2;
        b_out[o + 3] = b3;

        // Byte-for-byte incumbent lookahead message algebra.
        // The incumbent message algebra (`wt*a_i`, then the eight unreduced
        // products) in q registers and tri-form accumulators.
        unsafe {
            let q = |v: F128| core::mem::transmute::<F128, uint64x2_t>(v);
            acc.add_group(
                q(eq_lo[2 * t + 1]),
                q(a0),
                q(a1),
                q(a2),
                q(a3),
                q(b0),
                q(b1),
                q(b2),
                q(b3),
            );
        }
    }

    unsafe { acc.reduce() }
}

/// NT-store specialization of the lookahead pass using the expanded,
/// deferred-reduction fold4 pair. The ranked cascade reaches this entry only
/// for rounds 5/6; the ordinary-store and direct composed kernels stay on the
/// incumbent implementations above.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_and_message_lookahead_nt_expanded_aarch64(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    rho_ab: F128,
    eq_lo: &[F128],
) -> [F128; 8] {
    use core::arch::aarch64::*;

    // The private fold helper uses unchecked input pointers and the stores
    // below use raw output pointers. Establish their complete release-mode
    // geometry without overflow before entering the loop.
    let expected_out = eq_lo
        .len()
        .checked_mul(2)
        .expect("lookahead output length overflow");
    let expected_in = expected_out
        .checked_mul(4)
        .expect("lookahead input length overflow");
    assert_eq!(eq_lo.len() % 2, 0);
    assert_eq!(a_out.len(), expected_out);
    assert_eq!(b_out.len(), expected_out);
    assert_eq!(a_in.len(), expected_in);
    assert_eq!(b_in.len(), expected_in);

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

    let rho = unsafe {
        Fold4RhoQ::new(
            core::mem::transmute::<F128, uint64x2_t>(rho_a),
            core::mem::transmute::<F128, uint64x2_t>(rho_b),
            core::mem::transmute::<F128, uint64x2_t>(rho_ab),
        )
    };
    let mut acc = unsafe { LookaheadTriAcc::zero() };
    let a_out_ptr = a_out.as_mut_ptr();
    let b_out_ptr = b_out.as_mut_ptr();

    for t in 0..eq_lo.len() / 2 {
        let i = 16 * t;
        let o = 4 * t;

        let (a0, a1, a2, a3, b0, b1, b2, b3) = unsafe {
            // SAFETY: the geometry assertions and loop bound place each pair
            // wholly inside its corresponding 16-element source group.
            let (a0, a1) =
                fold4_pair_expanded_tri_q(a_in.as_ptr().add(i), &rho);
            let (a2, a3) =
                fold4_pair_expanded_tri_q(a_in.as_ptr().add(i + 8), &rho);
            let (b0, b1) =
                fold4_pair_expanded_tri_q(b_in.as_ptr().add(i), &rho);
            let (b2, b3) =
                fold4_pair_expanded_tri_q(b_in.as_ptr().add(i + 8), &rho);
            (
                core::mem::transmute::<uint64x2_t, F128>(a0),
                core::mem::transmute::<uint64x2_t, F128>(a1),
                core::mem::transmute::<uint64x2_t, F128>(a2),
                core::mem::transmute::<uint64x2_t, F128>(a3),
                core::mem::transmute::<uint64x2_t, F128>(b0),
                core::mem::transmute::<uint64x2_t, F128>(b1),
                core::mem::transmute::<uint64x2_t, F128>(b2),
                core::mem::transmute::<uint64x2_t, F128>(b3),
            )
        };

        // SAFETY: `o + 3 < expected_out`; every pair store is 16-byte aligned
        // and this iteration owns four disjoint output slots per array.
        unsafe {
            store_pair_nt(
                a_out_ptr.add(o),
                core::mem::transmute::<F128, uint64x2_t>(a0),
                core::mem::transmute::<F128, uint64x2_t>(a1),
            );
            store_pair_nt(
                a_out_ptr.add(o + 2),
                core::mem::transmute::<F128, uint64x2_t>(a2),
                core::mem::transmute::<F128, uint64x2_t>(a3),
            );
            store_pair_nt(
                b_out_ptr.add(o),
                core::mem::transmute::<F128, uint64x2_t>(b0),
                core::mem::transmute::<F128, uint64x2_t>(b1),
            );
            store_pair_nt(
                b_out_ptr.add(o + 2),
                core::mem::transmute::<F128, uint64x2_t>(b2),
                core::mem::transmute::<F128, uint64x2_t>(b3),
            );
        }

        // This is byte-for-byte the incumbent lookahead message algebra; only
        // construction of a0..a3/b0..b3 differs.
        // The incumbent message algebra (`wt*a_i`, then the eight unreduced
        // products) in q registers and tri-form accumulators.
        unsafe {
            let q = |v: F128| core::mem::transmute::<F128, uint64x2_t>(v);
            acc.add_group(
                q(eq_lo[2 * t + 1]),
                q(a0),
                q(a1),
                q(a2),
                q(a3),
                q(b0),
                q(b1),
                q(b2),
                q(b3),
            );
        }
    }

    unsafe { acc.reduce() }
}

pub(crate) const TRI_SLOTS: usize = 26;

///
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_tri_and_message_aarch64<const DEPTH: usize, const NT: bool, const ABL: u32>(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    rho_c: F128,
    eq_lo: &[F128],
) -> [F128; TRI_SLOTS] {
    fold_tri_and_message_impl::<DEPTH, NT, ABL, false>(
        a_in, b_in, a_out, b_out, rho_a, rho_b, rho_c, eq_lo,
    )
}

/// [`fold_tri_and_message_aarch64`] (depth two, no probes) over an input in
/// the dead-tail-packed layout ([`PACK_SUPER`]): logical inputs 61..63 of
/// every block of 64 have no entry and enter the same fold as zero
/// registers, so every output and slot is the one the full input (with its
/// zeros) gives.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_tri_and_message_packed_aarch64<const NT: bool>(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    eq_lo: &[F128],
) -> [F128; TRI_SLOTS] {
    fold_tri_and_message_impl::<2, NT, 0, true>(
        a_in, b_in, a_out, b_out, rho_a, rho_b, F128::ZERO, eq_lo,
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fold_tri_and_message_impl<const DEPTH: usize, const NT: bool, const ABL: u32, const PACK: bool>(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_a: F128,
    rho_b: F128,
    rho_c: F128,
    eq_lo: &[F128],
) -> [F128; TRI_SLOTS] {
    use core::arch::aarch64::*;
    assert!(DEPTH == 2 || DEPTH == 3);
    assert!(!PACK || (DEPTH == 2 && ABL == 0));
    let n_out = a_out.len();
    assert_eq!(b_out.len(), n_out);
    assert_eq!(n_out, 2 * eq_lo.len());
    assert!(n_out.is_multiple_of(8));
    if PACK {
        let n_in = n_out << DEPTH;
        assert!(n_in.is_multiple_of(64 * PACK_BLOCKS));
        assert_eq!(a_in.len(), n_in / (64 * PACK_BLOCKS) * PACK_SUPER);
        assert_eq!(b_in.len(), n_in / (64 * PACK_BLOCKS) * PACK_SUPER);
    } else {
        assert_eq!(a_in.len(), n_out << DEPTH);
        assert_eq!(b_in.len(), n_out << DEPTH);
    }

    #[inline(always)]
    unsafe fn store_pair<const NT: bool>(dst: *mut F128, x: uint64x2_t, y: uint64x2_t) {
        unsafe {
            if NT {
                core::arch::asm!(
                    "stnp {x:q}, {y:q}, [{dst}]",
                    dst = in(reg) dst,
                    x = in(vreg) x,
                    y = in(vreg) y,
                    options(nostack, preserves_flags),
                );
            } else {
                vst1q_u64(dst.cast::<u64>(), x);
                vst1q_u64(dst.add(1).cast::<u64>(), y);
            }
        }
    }

    unsafe {
        let q = |v: F128| core::mem::transmute::<F128, uint64x2_t>(v);
        let rho = Fold4RhoQ::new(q(rho_a), q(rho_b), q(rho_a * rho_b));
        let rc = q(rho_c);
        let rc_sw = swap_halves_q(rc);
        let zero = vdupq_n_u64(0);
        let mut acc = [TriNeon {
            ll: zero,
            cross: zero,
            hh: zero,
        }; TRI_SLOTS];
        let mut sink = zero;
        let a_ip = a_in.as_ptr();
        let b_ip = b_in.as_ptr();
        let a_op = a_out.as_mut_ptr();
        let b_op = b_out.as_mut_ptr();

        #[inline(always)]
        unsafe fn fold_eight<const DEPTH: usize>(
            src: *const F128,
            rho: &Fold4RhoQ,
            rc: uint64x2_t,
            rc_sw: uint64x2_t,
        ) -> [uint64x2_t; 8] {
            unsafe {
                let mut v = [vdupq_n_u64(0); 8];
                if DEPTH == 2 {
                    for h in 0..4 {
                        let (x, y) = fold4_pair_expanded_tri_q(src.add(8 * h), rho);
                        v[2 * h] = x;
                        v[2 * h + 1] = y;
                    }
                } else {
                    for (h, slot) in v.iter_mut().enumerate() {
                        let (t0, t1) = fold4_pair_expanded_tri_q(src.add(8 * h), rho);
                        *slot = veorq_u64(t0, mul_sw_tri_binius_q(rc, rc_sw, veorq_u64(t0, t1)));
                    }
                }
                v
            }
        }

        /// [`fold_eight`] (depth two) of a packed block's second half:
        /// logical inputs 32..60 at `src`, input 60 = `*side`, inputs
        /// 61..63 zero.
        #[inline(always)]
        unsafe fn fold_eight_packed_tail(
            src: *const F128,
            side: *const F128,
            rho: &Fold4RhoQ,
        ) -> [uint64x2_t; 8] {
            unsafe {
                let mut v = [vdupq_n_u64(0); 8];
                for h in 0..3 {
                    let (x, y) = fold4_pair_expanded_tri_q(src.add(8 * h), rho);
                    v[2 * h] = x;
                    v[2 * h + 1] = y;
                }
                let ld = |p: *const F128| vld1q_u64(p.cast::<u64>());
                let zero = vdupq_n_u64(0);
                v[6] = fold4_expanded_tri_one_q(
                    ld(src.add(24)),
                    ld(src.add(25)),
                    ld(src.add(26)),
                    ld(src.add(27)),
                    rho,
                );
                v[7] = fold4_expanded_tri_one_q(ld(side), zero, zero, zero, rho);
                v
            }
        }
        // Group `y` (32 logical inputs): packed, block `y / 2` of superblock
        // `y / 8`, first half from its main entries, second half through the
        // tail with the block's output-60 entry.
        let fold_group = |ip: *const F128, y: usize| -> [uint64x2_t; 8] {
            if PACK {
                let (blk, w) = (y / 2, (y / 2) % PACK_BLOCKS);
                let sup = ip.add(PACK_SUPER * (blk / PACK_BLOCKS));
                let main = sup.add(60 * w);
                if y % 2 == 0 {
                    fold_eight::<2>(main, &rho, rc, rc_sw)
                } else {
                    fold_eight_packed_tail(main.add(32), sup.add(60 * PACK_BLOCKS + w), &rho)
                }
            } else {
                fold_eight::<DEPTH>(ip.add((8 * y) << DEPTH), &rho, rc, rc_sw)
            }
        };

        const TG: usize = 16;
        let ng = n_out / 8;
        let mut ta = [[zero; 8]; TG];
        let mut tb = [[zero; 8]; TG];
        let mut g0 = 0usize;
        while g0 < ng {
            let gn = (ng - g0).min(TG);
            for t in 0..gn {
                let y = g0 + t;
                let a = fold_group(a_ip, y);
                let b = fold_group(b_ip, y);
                if ABL & 2 == 0 {
                    for h in 0..4 {
                        store_pair::<NT>(a_op.add(8 * y + 2 * h), a[2 * h], a[2 * h + 1]);
                        store_pair::<NT>(b_op.add(8 * y + 2 * h), b[2 * h], b[2 * h + 1]);
                    }
                } else {
                    for h in 0..8 {
                        sink = xor3_u64(sink, a[h], b[h]);
                    }
                }
                if ABL & 1 != 0 {
                    sink = xor3_u64(sink, a[0], b[7]);
                    continue;
                }
                let w = vld1q_u64(eq_lo.as_ptr().add(4 * y + 3).cast::<u64>());
                let w_sum = vgetq_lane_u64::<0>(w) ^ vgetq_lane_u64::<1>(w);
                for i in 0..8 {
                    ta[t][i] = mul_sw_tri_binius_karatsuba_q(w, w_sum, a[i]);
                }
                tb[t] = b;
            }
            g0 += gn;
            if ABL & 1 != 0 {
                continue;
            }
            {
                let mut e = [acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]];
                for t in 0..gn {
                    let (aw, b) = (&ta[t], &tb[t]);
                    for j in 0..4 {
                        tri_acc_q(&mut e[j], aw[2 * j + 1], b[2 * j + 1]);
                        tri_acc_q(
                            &mut e[4 + j],
                            veorq_u64(aw[2 * j], aw[2 * j + 1]),
                            veorq_u64(b[2 * j], b[2 * j + 1]),
                        );
                    }
                }
                acc[..8].copy_from_slice(&e);
            }
            {
                let mut l = [
                    acc[8], acc[9], acc[10], acc[11], acc[12], acc[13], acc[14], acc[15],
                ];
                for t in 0..gn {
                    let (aw, b) = (&ta[t], &tb[t]);
                    for k in 0..2 {
                        let o = 4 * k;
                        tri_acc_q(&mut l[k], aw[o + 2], b[o + 2]);
                        let (ea, eb) = (veorq_u64(aw[o], aw[o + 2]), veorq_u64(b[o], b[o + 2]));
                        let (oa, ob) =
                            (veorq_u64(aw[o + 1], aw[o + 3]), veorq_u64(b[o + 1], b[o + 3]));
                        tri_acc_q(&mut l[2 + k], ea, eb);
                        tri_acc_q(&mut l[4 + k], oa, ob);
                        tri_acc_q(&mut l[6 + k], veorq_u64(ea, oa), veorq_u64(eb, ob));
                    }
                }
                acc[8..16].copy_from_slice(&l);
            }
            {
                let mut g = [acc[16], acc[17], acc[18], acc[20], acc[21]];
                for t in 0..gn {
                    let (aw, b) = (&ta[t], &tb[t]);
                    tri_acc_q(&mut g[0], aw[4], b[4]);
                    let (s0a, s0b) = (veorq_u64(aw[0], aw[4]), veorq_u64(b[0], b[4]));
                    let (s1a, s1b) = (veorq_u64(aw[1], aw[5]), veorq_u64(b[1], b[5]));
                    let (s2a, s2b) = (veorq_u64(aw[2], aw[6]), veorq_u64(b[2], b[6]));
                    let (s3a, s3b) = (veorq_u64(aw[3], aw[7]), veorq_u64(b[3], b[7]));
                    tri_acc_q(&mut g[1], s0a, s0b);
                    tri_acc_q(&mut g[2], s2a, s2b);
                    tri_acc_q(&mut g[3], s1a, s1b);
                    tri_acc_q(&mut g[4], s3a, s3b);
                }
                acc[16] = g[0];
                acc[17] = g[1];
                acc[18] = g[2];
                acc[20] = g[3];
                acc[21] = g[4];
            }
            {
                let mut g = [acc[19], acc[22], acc[23], acc[24], acc[25]];
                for t in 0..gn {
                    let (aw, b) = (&ta[t], &tb[t]);
                    let (s0a, s0b) = (veorq_u64(aw[0], aw[4]), veorq_u64(b[0], b[4]));
                    let (s1a, s1b) = (veorq_u64(aw[1], aw[5]), veorq_u64(b[1], b[5]));
                    let (s2a, s2b) = (veorq_u64(aw[2], aw[6]), veorq_u64(b[2], b[6]));
                    let (s3a, s3b) = (veorq_u64(aw[3], aw[7]), veorq_u64(b[3], b[7]));
                    let (s02a, s02b) = (veorq_u64(s0a, s2a), veorq_u64(s0b, s2b));
                    let (s13a, s13b) = (veorq_u64(s1a, s3a), veorq_u64(s1b, s3b));
                    let (s01a, s01b) = (veorq_u64(s0a, s1a), veorq_u64(s0b, s1b));
                    let (s23a, s23b) = (veorq_u64(s2a, s3a), veorq_u64(s2b, s3b));
                    tri_acc_q(&mut g[0], s02a, s02b);
                    tri_acc_q(&mut g[1], s13a, s13b);
                    tri_acc_q(&mut g[2], s01a, s01b);
                    tri_acc_q(&mut g[3], s23a, s23b);
                    tri_acc_q(&mut g[4], veorq_u64(s01a, s23a), veorq_u64(s01b, s23b));
                }
                acc[19] = g[0];
                acc[22] = g[1];
                acc[23] = g[2];
                acc[24] = g[3];
                acc[25] = g[4];
            }
        }
        if ABL != 0 {
            core::hint::black_box(sink);
        }
        acc.map(|t| core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(t)))
    }
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "aes"))]
mod tests {
    use super::*;
    use core::arch::aarch64::uint64x2_t;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Every byte lane and every byte value: the selected load form returns
    /// exactly entry `(row >> SHIFT) & 0xff` of the bank, whatever the other
    /// 56 bits of the row hold.
    #[test]
    fn bank_load_q_reads_the_indexed_entry() {
        // The bank sits between two guard banks with different contents, so a
        // wrong offset cannot return the right value by accident.
        let entries: Vec<[u64; 2]> = (0..3 * 256u64)
            .map(|j| [j.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xa5a5, !j])
            .collect();
        let bank = unsafe { entries.as_ptr().add(256).cast::<u8>() };
        let mut state = 0x4c4f_4144_464f_524d;
        macro_rules! check_lane {
            ($lane:literal) => {
                for byte in 0..256u64 {
                    let field = 0xffu64 << (8 * $lane);
                    for case in 0..6 {
                        let other = match case {
                            0 => 0,
                            1 => u64::MAX,
                            _ => splitmix64(&mut state),
                        };
                        let row = (other & !field) | (byte << (8 * $lane));
                        let got: [u64; 2] = unsafe {
                            core::mem::transmute(bank_load_q::<{ 8 * $lane }>(bank, row))
                        };
                        assert_eq!(
                            got,
                            entries[256 + byte as usize],
                            "lane {} byte {byte} row {row:#018x}",
                            $lane
                        );
                    }
                }
            };
        }
        check_lane!(0);
        check_lane!(1);
        check_lane!(2);
        check_lane!(3);
        check_lane!(4);
        check_lane!(5);
        check_lane!(6);
        check_lane!(7);
    }

    /// The eight-lane lookups built on `bank_load_q` equal the XOR of the
    /// eight plainly indexed entries.
    #[test]
    fn eight_lane_lookups_match_plain_indexing() {
        let mut state = 0x4c4f_4f4b_5550_3858;
        let table: Vec<[u64; 2]> = (0..8 * 256)
            .map(|_| [splitmix64(&mut state), splitmix64(&mut state)])
            .collect();
        let ptr = table.as_ptr().cast::<u8>();
        let plain = |code: u64| -> [u64; 2] {
            let mut acc = [0u64; 2];
            for lane in 0..8 {
                let e = table[lane * 256 + ((code >> (8 * lane)) & 0xff) as usize];
                acc[0] ^= e[0];
                acc[1] ^= e[1];
            }
            acc
        };
        let as_words = |v: uint64x2_t| -> [u64; 2] { unsafe { core::mem::transmute(v) } };
        for case in 0..4096 {
            let codes: [u64; 4] = match case {
                0 => [0; 4],
                1 => [u64::MAX; 4],
                _ => core::array::from_fn(|_| splitmix64(&mut state)),
            };
            unsafe {
                assert_eq!(as_words(lookup_lanes_q::<8>(ptr, codes[0], 0)), plain(codes[0]));
                assert_eq!(as_words(fold_row_code_q(ptr, codes[1])), plain(codes[1]));
                let (x, y) = fold_two_row_codes_q(ptr, codes[0], codes[1]);
                assert_eq!([as_words(x), as_words(y)], [plain(codes[0]), plain(codes[1])]);
                let (w, x, y, z) = fold_four_row_codes_q(ptr, codes[0], codes[1], codes[2], codes[3]);
                assert_eq!(
                    [as_words(w), as_words(x), as_words(y), as_words(z)],
                    [plain(codes[0]), plain(codes[1]), plain(codes[2]), plain(codes[3])]
                );
                #[cfg(target_feature = "sha3")]
                {
                    assert_eq!(as_words(lookup8_q(ptr, codes[2])), plain(codes[2]));
                    let init = core::mem::transmute::<[u64; 2], uint64x2_t>([codes[0], codes[1]]);
                    let want = plain(codes[3]);
                    assert_eq!(
                        as_words(lookup8_acc_q(ptr, codes[3], init)),
                        [want[0] ^ codes[0], want[1] ^ codes[1]]
                    );
                }
            }
        }
    }

    /// One pass of the standalone lookup loop for load form `form` over `n`
    /// eight-byte rows: eight table loads per row, XORed into one sum.
    ///
    /// Forms: 0 `ubfx` + scaled-index load (the parent); 1 `and` with a
    /// shifted register + plain register-offset load (this tree); 2 two lane
    /// masks per row, then `ubfx` of twelve bits + plain load; 3 `ubfx`,
    /// `lsl #4`, plain load (three single instructions, the calibration for
    /// "one more pipeline slot per lookup"). With `filler`, sixteen
    /// independent non-load instructions per row are added so that the loop
    /// is bound by pipeline width rather than by the load units.
    #[inline(never)]
    unsafe fn q_load_uop_pass(
        form: u8,
        table: *const u8,
        rows: *const u64,
        n: usize,
        filler: bool,
    ) -> [u64; 2] {
        use core::arch::aarch64::vdupq_n_u64;
        unsafe {
            let end = rows.add(n);
            let acc0 = vdupq_n_u64(0);
            let mut acc = acc0;
            // SAFETY: every index is a byte times 16 inside a 4 KiB bank and
            // `rows..end` is readable; the loop only reads memory.
            match form {
                0 => core::arch::asm!(
                        "2:",
                        "ldr {row}, [{p}], #8",
                        "ubfx {i}, {row}, #0, #8",
                        "ldr q0, [{b0}, {i}, lsl #4]",
                        "ubfx {i}, {row}, #8, #8",
                        "ldr q1, [{b1}, {i}, lsl #4]",
                        "ubfx {i}, {row}, #16, #8",
                        "ldr q2, [{b2}, {i}, lsl #4]",
                        "ubfx {i}, {row}, #24, #8",
                        "ldr q3, [{b3}, {i}, lsl #4]",
                        "ubfx {i}, {row}, #32, #8",
                        "ldr q4, [{b4}, {i}, lsl #4]",
                        "ubfx {i}, {row}, #40, #8",
                        "ldr q5, [{b5}, {i}, lsl #4]",
                        "ubfx {i}, {row}, #48, #8",
                        "ldr q6, [{b6}, {i}, lsl #4]",
                        "ubfx {i}, {row}, #56, #8",
                        "ldr q7, [{b7}, {i}, lsl #4]",
                        "eor v0.16b, v0.16b, v1.16b",
                        "eor v2.16b, v2.16b, v3.16b",
                        "eor v4.16b, v4.16b, v5.16b",
                        "eor v6.16b, v6.16b, v7.16b",
                        "eor v0.16b, v0.16b, v2.16b",
                        "eor v4.16b, v4.16b, v6.16b",
                        "eor v0.16b, v0.16b, v4.16b",
                        "eor v16.16b, v16.16b, v0.16b",
                        "cbz {fill}, 3f",
                        "add {f}, {g}, #1",
                        "add {f}, {g}, #2",
                        "add {f}, {g}, #3",
                        "add {f}, {g}, #4",
                        "add {f}, {g}, #5",
                        "add {f}, {g}, #6",
                        "add {f}, {g}, #7",
                        "add {f}, {g}, #8",
                        "add {f}, {g}, #9",
                        "add {f}, {g}, #10",
                        "add {f}, {g}, #11",
                        "add {f}, {g}, #12",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "3:",
                        "cmp {p}, {end}",
                        "b.lo 2b",
                        p = inout(reg) rows => _,
                        end = in(reg) end,
                        row = out(reg) _,
                        i = out(reg) _,
                        fill = in(reg) filler as u64,
                        f = out(reg) _,
                        g = in(reg) 7u64,
                        b0 = in(reg) table,
                        b1 = in(reg) table.add(4096),
                        b2 = in(reg) table.add(2 * 4096),
                        b3 = in(reg) table.add(3 * 4096),
                        b4 = in(reg) table.add(4 * 4096),
                        b5 = in(reg) table.add(5 * 4096),
                        b6 = in(reg) table.add(6 * 4096),
                        b7 = in(reg) table.add(7 * 4096),
                        inout("v16") acc,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                        out("v17") _, in("v18") acc0, in("v19") acc0,
                        options(readonly, nostack),
                    ),
                1 => core::arch::asm!(
                        "2:",
                        "ldr {row}, [{p}], #8",
                        "ubfiz {i}, {row}, #4, #8",
                        "ldr q0, [{b0}, {i}]",
                        "and {i}, {mask}, {row}, lsr #4",
                        "ldr q1, [{b1}, {i}]",
                        "and {i}, {mask}, {row}, lsr #12",
                        "ldr q2, [{b2}, {i}]",
                        "and {i}, {mask}, {row}, lsr #20",
                        "ldr q3, [{b3}, {i}]",
                        "and {i}, {mask}, {row}, lsr #28",
                        "ldr q4, [{b4}, {i}]",
                        "and {i}, {mask}, {row}, lsr #36",
                        "ldr q5, [{b5}, {i}]",
                        "and {i}, {mask}, {row}, lsr #44",
                        "ldr q6, [{b6}, {i}]",
                        "and {i}, {mask}, {row}, lsr #52",
                        "ldr q7, [{b7}, {i}]",
                        "eor v0.16b, v0.16b, v1.16b",
                        "eor v2.16b, v2.16b, v3.16b",
                        "eor v4.16b, v4.16b, v5.16b",
                        "eor v6.16b, v6.16b, v7.16b",
                        "eor v0.16b, v0.16b, v2.16b",
                        "eor v4.16b, v4.16b, v6.16b",
                        "eor v0.16b, v0.16b, v4.16b",
                        "eor v16.16b, v16.16b, v0.16b",
                        "cbz {fill}, 3f",
                        "add {f}, {g}, #1",
                        "add {f}, {g}, #2",
                        "add {f}, {g}, #3",
                        "add {f}, {g}, #4",
                        "add {f}, {g}, #5",
                        "add {f}, {g}, #6",
                        "add {f}, {g}, #7",
                        "add {f}, {g}, #8",
                        "add {f}, {g}, #9",
                        "add {f}, {g}, #10",
                        "add {f}, {g}, #11",
                        "add {f}, {g}, #12",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "3:",
                        "cmp {p}, {end}",
                        "b.lo 2b",
                        mask = in(reg) 0xff0u64,
                        p = inout(reg) rows => _,
                        end = in(reg) end,
                        row = out(reg) _,
                        i = out(reg) _,
                        fill = in(reg) filler as u64,
                        f = out(reg) _,
                        g = in(reg) 7u64,
                        b0 = in(reg) table,
                        b1 = in(reg) table.add(4096),
                        b2 = in(reg) table.add(2 * 4096),
                        b3 = in(reg) table.add(3 * 4096),
                        b4 = in(reg) table.add(4 * 4096),
                        b5 = in(reg) table.add(5 * 4096),
                        b6 = in(reg) table.add(6 * 4096),
                        b7 = in(reg) table.add(7 * 4096),
                        inout("v16") acc,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                        out("v17") _, in("v18") acc0, in("v19") acc0,
                        options(readonly, nostack),
                    ),
                2 => core::arch::asm!(
                        "2:",
                        "ldr {row}, [{p}], #8",
                        "and {re}, {row}, #0x00ff00ff00ff00ff",
                        "and {ro}, {row}, #0xff00ff00ff00ff00",
                        "ubfiz {i}, {row}, #4, #8",
                        "ldr q0, [{b0}, {i}]",
                        "ubfx {i}, {ro}, #4, #12",
                        "ldr q1, [{b1}, {i}]",
                        "ubfx {i}, {re}, #12, #12",
                        "ldr q2, [{b2}, {i}]",
                        "ubfx {i}, {ro}, #20, #12",
                        "ldr q3, [{b3}, {i}]",
                        "ubfx {i}, {re}, #28, #12",
                        "ldr q4, [{b4}, {i}]",
                        "ubfx {i}, {ro}, #36, #12",
                        "ldr q5, [{b5}, {i}]",
                        "ubfx {i}, {re}, #44, #12",
                        "ldr q6, [{b6}, {i}]",
                        "ubfx {i}, {ro}, #52, #12",
                        "ldr q7, [{b7}, {i}]",
                        "eor v0.16b, v0.16b, v1.16b",
                        "eor v2.16b, v2.16b, v3.16b",
                        "eor v4.16b, v4.16b, v5.16b",
                        "eor v6.16b, v6.16b, v7.16b",
                        "eor v0.16b, v0.16b, v2.16b",
                        "eor v4.16b, v4.16b, v6.16b",
                        "eor v0.16b, v0.16b, v4.16b",
                        "eor v16.16b, v16.16b, v0.16b",
                        "cbz {fill}, 3f",
                        "add {f}, {g}, #1",
                        "add {f}, {g}, #2",
                        "add {f}, {g}, #3",
                        "add {f}, {g}, #4",
                        "add {f}, {g}, #5",
                        "add {f}, {g}, #6",
                        "add {f}, {g}, #7",
                        "add {f}, {g}, #8",
                        "add {f}, {g}, #9",
                        "add {f}, {g}, #10",
                        "add {f}, {g}, #11",
                        "add {f}, {g}, #12",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "3:",
                        "cmp {p}, {end}",
                        "b.lo 2b",
                        re = out(reg) _, ro = out(reg) _,
                        p = inout(reg) rows => _,
                        end = in(reg) end,
                        row = out(reg) _,
                        i = out(reg) _,
                        fill = in(reg) filler as u64,
                        f = out(reg) _,
                        g = in(reg) 7u64,
                        b0 = in(reg) table,
                        b1 = in(reg) table.add(4096),
                        b2 = in(reg) table.add(2 * 4096),
                        b3 = in(reg) table.add(3 * 4096),
                        b4 = in(reg) table.add(4 * 4096),
                        b5 = in(reg) table.add(5 * 4096),
                        b6 = in(reg) table.add(6 * 4096),
                        b7 = in(reg) table.add(7 * 4096),
                        inout("v16") acc,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                        out("v17") _, in("v18") acc0, in("v19") acc0,
                        options(readonly, nostack),
                    ),
                3 => core::arch::asm!(
                        "2:",
                        "ldr {row}, [{p}], #8",
                        "ubfx {i}, {row}, #0, #8",
                        "lsl {i}, {i}, #4",
                        "ldr q0, [{b0}, {i}]",
                        "ubfx {i}, {row}, #8, #8",
                        "lsl {i}, {i}, #4",
                        "ldr q1, [{b1}, {i}]",
                        "ubfx {i}, {row}, #16, #8",
                        "lsl {i}, {i}, #4",
                        "ldr q2, [{b2}, {i}]",
                        "ubfx {i}, {row}, #24, #8",
                        "lsl {i}, {i}, #4",
                        "ldr q3, [{b3}, {i}]",
                        "ubfx {i}, {row}, #32, #8",
                        "lsl {i}, {i}, #4",
                        "ldr q4, [{b4}, {i}]",
                        "ubfx {i}, {row}, #40, #8",
                        "lsl {i}, {i}, #4",
                        "ldr q5, [{b5}, {i}]",
                        "ubfx {i}, {row}, #48, #8",
                        "lsl {i}, {i}, #4",
                        "ldr q6, [{b6}, {i}]",
                        "ubfx {i}, {row}, #56, #8",
                        "lsl {i}, {i}, #4",
                        "ldr q7, [{b7}, {i}]",
                        "eor v0.16b, v0.16b, v1.16b",
                        "eor v2.16b, v2.16b, v3.16b",
                        "eor v4.16b, v4.16b, v5.16b",
                        "eor v6.16b, v6.16b, v7.16b",
                        "eor v0.16b, v0.16b, v2.16b",
                        "eor v4.16b, v4.16b, v6.16b",
                        "eor v0.16b, v0.16b, v4.16b",
                        "eor v16.16b, v16.16b, v0.16b",
                        "cbz {fill}, 3f",
                        "add {f}, {g}, #1",
                        "add {f}, {g}, #2",
                        "add {f}, {g}, #3",
                        "add {f}, {g}, #4",
                        "add {f}, {g}, #5",
                        "add {f}, {g}, #6",
                        "add {f}, {g}, #7",
                        "add {f}, {g}, #8",
                        "add {f}, {g}, #9",
                        "add {f}, {g}, #10",
                        "add {f}, {g}, #11",
                        "add {f}, {g}, #12",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "eor v17.16b, v18.16b, v19.16b",
                        "3:",
                        "cmp {p}, {end}",
                        "b.lo 2b",
                        p = inout(reg) rows => _,
                        end = in(reg) end,
                        row = out(reg) _,
                        i = out(reg) _,
                        fill = in(reg) filler as u64,
                        f = out(reg) _,
                        g = in(reg) 7u64,
                        b0 = in(reg) table,
                        b1 = in(reg) table.add(4096),
                        b2 = in(reg) table.add(2 * 4096),
                        b3 = in(reg) table.add(3 * 4096),
                        b4 = in(reg) table.add(4 * 4096),
                        b5 = in(reg) table.add(5 * 4096),
                        b6 = in(reg) table.add(6 * 4096),
                        b7 = in(reg) table.add(7 * 4096),
                        inout("v16") acc,
                        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                        out("v17") _, in("v18") acc0, in("v19") acc0,
                        options(readonly, nostack),
                    ),
                _ => unreachable!(),
            }
            core::mem::transmute(acc)
        }
    }

    /// Standalone answer to one question about the core this runs on: does a
    /// 128-bit load with a shifted register offset cost one more pipeline slot
    /// than the same load with a plain register offset? Compare form 0 with
    /// form 1 (same instruction count) and with form 3 (one more instruction
    /// per lookup) in the `filler` rows. All four forms must return the same
    /// sum. Single thread, a few hundred milliseconds.
    #[test]
    #[ignore = "timing probe"]
    fn q_load_form_uop_probe() {
        const ROWS: usize = 2048;
        const PASSES: usize = 64;
        const SAMPLES: usize = 31;
        let mut state = 0x554f_5050_524f_4245;
        let table: Vec<[u64; 2]> = (0..8 * 256)
            .map(|_| [splitmix64(&mut state), splitmix64(&mut state)])
            .collect();
        let rows: Vec<u64> = (0..ROWS).map(|_| splitmix64(&mut state)).collect();
        let run = |form: u8, filler: bool| -> (f64, [u64; 2]) {
            let mut sum = [0u64; 2];
            let t = std::time::Instant::now();
            for _ in 0..PASSES {
                sum = unsafe {
                    q_load_uop_pass(form, table.as_ptr().cast::<u8>(), rows.as_ptr(), ROWS, filler)
                };
            }
            (t.elapsed().as_nanos() as f64 / (PASSES * ROWS) as f64, sum)
        };
        let want = run(0, false).1;
        for filler in [false, true] {
            let mut ns: [Vec<f64>; 4] = Default::default();
            for _ in 0..SAMPLES {
                for form in 0..4u8 {
                    let (t, sum) = run(form, filler);
                    assert_eq!(sum, want, "form {form}");
                    ns[form as usize].push(t);
                }
            }
            for v in ns.iter_mut() {
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            }
            let q = |v: &Vec<f64>| (v[0], v[SAMPLES / 4], v[SAMPLES / 2]);
            for (form, v) in ns.iter().enumerate() {
                let (min, q1, med) = q(v);
                println!(
                    "UOPPROBE filler={} form={form} ns_per_row min {min:.4} q1 {q1:.4} med {med:.4} (min vs form 0: {:+.2}%)",
                    filler as u8,
                    (min / ns[0][0] - 1.0) * 100.0,
                );
            }
        }
    }

    /// Single-thread cost of the two table-lookup kernels of the ranked path
    /// for the load form this binary was built with (`FLOCK_Q_LOAD_FORM`).
    /// Build one test binary per form and alternate them; the checksums must
    /// agree between forms.
    #[test]
    #[ignore = "timing probe"]
    fn table_load_form_timing() {
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        use crate::zerocheck::univariate_skip::build_eq;
        use crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash;

        let mut state = 0x4c4f_4144_5449_4d45;
        let z = F128::new(splitmix64(&mut state), splitmix64(&mut state));
        let table = UniSkipFoldTable::new(6, z);
        let mix = |sum: &mut u64, v: &F128| {
            let w: [u64; 2] = unsafe { core::mem::transmute(*v) };
            *sum = (*sum ^ w[0]).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ w[1];
        };

        // Round two: the Horner kernel as the ranked proof calls it (peeled
        // first group, 128-pair blocks with 121 live pairs).
        const LO: usize = 1 << 14;
        const R2_CHUNKS: usize = 2;
        const R2_PASSES: usize = 12;
        let mut r: Vec<F128> = (0..14)
            .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
            .collect();
        r[2..6].copy_from_slice(&medium_challenges_ghash());
        let eq_lo = build_eq(&r);
        let mut a = vec![0u8; 16 * LO * R2_CHUNKS];
        let mut b = vec![0u8; 16 * LO * R2_CHUNKS];
        for pair in 0..LO * R2_CHUNKS {
            if pair & 127 >= 121 {
                continue;
            }
            for byte in a[16 * pair..16 * pair + 16]
                .iter_mut()
                .chain(b[16 * pair..16 * pair + 16].iter_mut())
            {
                *byte = splitmix64(&mut state) as u8;
            }
        }
        let mut anchors = vec![F128::ZERO; 2 * LO * R2_CHUNKS];
        let mut deltas = vec![0u8; 16 * LO * R2_CHUNKS];
        let mut r2_pass = || -> (f64, u64) {
            let mut sum = 0u64;
            let t = std::time::Instant::now();
            for _ in 0..R2_PASSES {
                for c in 0..R2_CHUNKS {
                    let mut out = [F128::ZERO; 8];
                    unsafe {
                        fold_round2_compact_chunk_neon_horner_8::<true>(
                            table.data.as_ptr().cast::<u8>(),
                            a.as_ptr().add(16 * LO * c),
                            b.as_ptr().add(16 * LO * c),
                            anchors.as_mut_ptr().add(2 * LO * c),
                            deltas.as_mut_ptr().add(16 * LO * c),
                            eq_lo.as_ptr(),
                            LO,
                            c * LO,
                            127,
                            121,
                            true,
                            true,
                            out.as_mut_ptr(),
                        );
                    }
                    for v in &out[4..] {
                        mix(&mut sum, v);
                    }
                }
            }
            let ns = t.elapsed().as_nanos() as f64 / (R2_PASSES * R2_CHUNKS * LO) as f64;
            for v in anchors.iter().step_by(61) {
                mix(&mut sum, v);
            }
            (ns, sum)
        };

        // K pass: the fused kernel on 2^12 lanes per call, dense data.
        #[cfg(target_feature = "sha3")]
        const K_OUT_PAIRS: usize = 1 << 11;
        #[cfg(target_feature = "sha3")]
        const K_PASSES: usize = 96;
        #[cfg(target_feature = "sha3")]
        let (k_l1, k_l3, k_rho2, k_anchors, k_deltas, k_eq) = {
            let rho = |state: &mut u64| F128::new(splitmix64(state), splitmix64(state));
            let l1 = table.scaled_linear(rho(&mut state));
            let l3 = table.scaled_linear(rho(&mut state));
            let rho2 = rho(&mut state);
            let anchors: Vec<F128> = (0..8 * K_OUT_PAIRS).map(|_| rho(&mut state)).collect();
            let deltas: Vec<u8> = (0..64 * K_OUT_PAIRS)
                .map(|_| splitmix64(&mut state) as u8)
                .collect();
            let eq: Vec<F128> = (0..K_OUT_PAIRS).map(|_| rho(&mut state)).collect();
            (l1, l3, rho2, anchors, deltas, eq)
        };
        #[cfg(target_feature = "sha3")]
        let mut k_a = vec![F128::ZERO; 2 * K_OUT_PAIRS];
        #[cfg(target_feature = "sha3")]
        let mut k_b = vec![F128::ZERO; 2 * K_OUT_PAIRS];
        #[cfg(target_feature = "sha3")]
        let mut k_pass = || -> (f64, u64) {
            let mut sum = 0u64;
            let t = std::time::Instant::now();
            for _ in 0..K_PASSES {
                let mut out = [F128::ZERO; 8];
                unsafe {
                    fold2_compact_and_round45_chunk_neon_8_fused::<false>(
                        k_l1.as_ptr().cast::<u8>(),
                        k_l3.as_ptr().cast::<u8>(),
                        k_rho2,
                        k_anchors.as_ptr(),
                        k_deltas.as_ptr(),
                        k_a.as_mut_ptr(),
                        k_b.as_mut_ptr(),
                        k_eq.as_ptr(),
                        K_OUT_PAIRS,
                        true,
                        out.as_mut_ptr(),
                    );
                }
                for v in &out {
                    mix(&mut sum, v);
                }
            }
            let ns = t.elapsed().as_nanos() as f64 / (K_PASSES * 2 * K_OUT_PAIRS) as f64;
            for v in k_a.iter().chain(k_b.iter()).step_by(37) {
                mix(&mut sum, v);
            }
            (ns, sum)
        };
        #[cfg(not(target_feature = "sha3"))]
        let k_pass = || -> (f64, u64) { (0.0, 0) };

        r2_pass();
        k_pass();
        const SAMPLES: usize = 41;
        let mut r2_ns = Vec::with_capacity(SAMPLES);
        let mut k_ns = Vec::with_capacity(SAMPLES);
        let (mut r2_sum, mut k_sum) = (0u64, 0u64);
        for _ in 0..SAMPLES {
            let (ns, sum) = r2_pass();
            r2_ns.push(ns);
            r2_sum = sum;
            let (ns, sum) = k_pass();
            k_ns.push(ns);
            k_sum = sum;
        }
        r2_ns.sort_by(|x, y| x.partial_cmp(y).unwrap());
        k_ns.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!(
            "LOADFORM form={} r2_ns_per_pair min {:.4} q1 {:.4} med {:.4} k_ns_per_lane min {:.4} q1 {:.4} med {:.4} sums {:016x} {:016x}",
            Q_LOAD_FORM,
            r2_ns[0], r2_ns[SAMPLES / 4], r2_ns[SAMPLES / 2],
            k_ns[0], k_ns[SAMPLES / 4], k_ns[SAMPLES / 2],
            r2_sum, k_sum,
        );
    }

    #[test]
    fn round2_horner_weighted_sums_match_full_cpu() {
        use crate::field::PHI_8_TABLE;
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        use crate::zerocheck::univariate_skip::build_eq;
        use crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash;

        const LO_SIZE: usize = 256;
        let mut state = 0x5232_484f_524e_4552;
        let mut a_packed = vec![0u8; 16 * LO_SIZE];
        let mut b_packed = vec![0u8; 16 * LO_SIZE];
        for byte in a_packed.iter_mut().chain(b_packed.iter_mut()) {
            *byte = splitmix64(&mut state) as u8;
        }
        for instance in 0..2 {
            let base = instance * 2048;
            for group in [0, 1, 2, 3, 59] {
                b_packed[base + group * 32..base + (group + 1) * 32].fill(0xff);
            }
            // Also test a mixed group, and the 49 useful bits in row 240.
            b_packed[base + 4 * 32..base + 4 * 32 + 16].fill(0xff);
            for packed in [&mut a_packed, &mut b_packed] {
                packed[base + 1926] &= 1;
                packed[base + 1927..base + 1936].fill(0);
            }
        }
        let random_z = F128::new(splitmix64(&mut state), splitmix64(&mut state));
        for (z_case, z) in [F128::ZERO, F128::ONE, PHI_8_TABLE[64], random_z]
            .into_iter()
            .enumerate()
        {
            let mut table = UniSkipFoldTable::new(6, z);
            // A random linear table exercises the guard when fold(all-ones) != 1.
            if z_case == 3 {
                for bank in table.data.chunks_mut(256) {
                    for bit in 0..8 {
                        bank[1 << bit] = F128::new(splitmix64(&mut state), splitmix64(&mut state));
                    }
                    for value in 3usize..256 {
                        if !value.is_power_of_two() {
                            let low = value & value.wrapping_neg();
                            bank[value] = bank[value ^ low] + bank[low];
                        }
                    }
                }
            }
            for challenge_case in 0..6 {
                let mut r: Vec<F128> = (0..8)
                    .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                    .collect();
                r[2..6].copy_from_slice(&medium_challenges_ghash());
                match challenge_case {
                    0 => r[0] = F128::ZERO,
                    1 => r[0] = F128::ONE,
                    2 => r[1] = F128::ZERO,
                    3 => r[1] = F128::ONE,
                    4 => r[6..8].copy_from_slice(&[F128::ZERO, F128::ONE]),
                    _ => {}
                }
                let eq_lo = build_eq(&r);
                for (mask, useful) in [(usize::MAX, usize::MAX), (127, 121)] {
                    for (degen, skip_dead) in
                        [(false, false), (true, false), (false, true), (true, true)]
                    {
                        let run = |horner: bool| {
                            let poison = F128::new(u64::MAX, u64::MAX);
                            let mut anchors = vec![poison; 2 * LO_SIZE];
                            let mut deltas = vec![0xa5u8; 16 * LO_SIZE];
                            let mut out = [poison; 8];
                            unsafe {
                                if horner {
                                    fold_round2_compact_chunk_neon_horner_8::<true>(
                                        table.data.as_ptr().cast::<u8>(),
                                        a_packed.as_ptr(),
                                        b_packed.as_ptr(),
                                        anchors.as_mut_ptr(),
                                        deltas.as_mut_ptr(),
                                        eq_lo.as_ptr(),
                                        LO_SIZE,
                                        0,
                                        mask,
                                        useful,
                                        degen,
                                        skip_dead,
                                        out.as_mut_ptr(),
                                    );
                                } else {
                                    fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                                        table.data.as_ptr().cast::<u8>(),
                                        a_packed.as_ptr(),
                                        b_packed.as_ptr(),
                                        anchors.as_mut_ptr(),
                                        deltas.as_mut_ptr(),
                                        eq_lo.as_ptr(),
                                        LO_SIZE,
                                        0,
                                        mask,
                                        useful,
                                        degen,
                                        true,
                                        skip_dead,
                                        out.as_mut_ptr(),
                                    );
                                }
                            }
                            (anchors, deltas, out)
                        };
                        let cpu = run(false);
                        let horner = run(true);
                        assert_eq!(
                            horner.0, cpu.0,
                            "anchor mismatch at {z_case}/{challenge_case}"
                        );
                        assert_eq!(
                            horner.1, cpu.1,
                            "delta mismatch at {z_case}/{challenge_case}"
                        );
                        assert_eq!(&horner.2[..4], &[F128::ZERO; 4]);
                        assert_eq!(
                            &horner.2[4..],
                            &cpu.2[4..],
                            "sum mismatch at {z_case}/{challenge_case}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn round2_horner_gpr_delta_matches_vector_delta() {
        use crate::field::PHI_8_TABLE;
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        use crate::zerocheck::univariate_skip::build_eq;
        use crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash;

        const LO_SIZE: usize = 512;
        let mut state = 0x4750_525f_4445_4c54;
        let mut a_packed = vec![0u8; 16 * LO_SIZE];
        let mut b_packed = vec![0u8; 16 * LO_SIZE];
        for byte in a_packed.iter_mut().chain(b_packed.iter_mut()) {
            *byte = splitmix64(&mut state) as u8;
        }
        for group in [0usize, 15, 16, 31, 47, 60, 64, 127, 200] {
            b_packed[group * 32..(group + 1) * 32].fill(0xff);
        }
        b_packed[5 * 32..5 * 32 + 16].fill(0xff);
        let random_z = F128::new(splitmix64(&mut state), splitmix64(&mut state));
        for z in [F128::ONE, PHI_8_TABLE[64], random_z] {
            let table = UniSkipFoldTable::new(6, z);
            let mut r: Vec<F128> = (0..9)
                .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                .collect();
            r[2..6].copy_from_slice(&medium_challenges_ghash());
            let eq_lo = build_eq(&r);
            for (mask, useful) in [(usize::MAX, usize::MAX), (127, 121)] {
                for (degen, skip_dead) in [(false, false), (true, false), (true, true)] {
                    for base in [0usize, 64, 121, 3 * LO_SIZE] {
                        let run = |gpr: bool, peel: bool| {
                            let poison = F128::new(u64::MAX, u64::MAX);
                            let mut anchors = vec![poison; 2 * LO_SIZE];
                            let mut deltas = vec![0xa5u8; 16 * LO_SIZE];
                            let mut out = [poison; 8];
                            unsafe {
                                let kernel = match (gpr, peel) {
                                    (true, true) => fold_round2_compact_chunk_neon_horner_8_impl::<true, true>,
                                    (false, true) => fold_round2_compact_chunk_neon_horner_8_impl::<true, false>,
                                    (true, false) => fold_round2_compact_chunk_neon_horner_8_impl::<false, true>,
                                    (false, false) => fold_round2_compact_chunk_neon_horner_8_impl::<false, false>,
                                };
                                kernel(
                                    table.data.as_ptr().cast::<u8>(),
                                    a_packed.as_ptr(),
                                    b_packed.as_ptr(),
                                    anchors.as_mut_ptr(),
                                    deltas.as_mut_ptr(),
                                    eq_lo.as_ptr(),
                                    LO_SIZE,
                                    base,
                                    mask,
                                    useful,
                                    degen,
                                    skip_dead,
                                    out.as_mut_ptr(),
                                );
                            }
                            (anchors, deltas, out)
                        };
                        for peel in [true, false] {
                            let vector = run(false, peel);
                            let gpr = run(true, peel);
                            let at = format!("mask {mask} base {base} degen {degen} skip {skip_dead} peel {peel}");
                            assert_eq!(gpr.0, vector.0, "anchors: {at}");
                            assert_eq!(gpr.1, vector.1, "deltas: {at}");
                            assert_eq!(gpr.2, vector.2, "slots: {at}");
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_feature = "sha3")]
    #[test]
    fn mul_xb_matches_mul_by_x_ladder() {
        use core::arch::aarch64::*;
        let mut state = 0x7862_5f6c_6164_6465;
        let k = unsafe { vdupq_n_u64(0x87) };
        for case in 0..512 {
            let hi = splitmix64(&mut state) | if case & 1 == 0 { 1u64 << 63 } else { 0 };
            let v = F128::new(splitmix64(&mut state), hi);
            let mut want = v;
            for b in 0..16u32 {
                let got = unsafe {
                    core::mem::transmute::<uint64x2_t, F128>(mul_xb_q(
                        core::mem::transmute::<F128, uint64x2_t>(v),
                        vdupq_n_u64(1u64 << b),
                        k,
                    ))
                };
                assert_eq!(got, want, "case {case} b={b}");
                want = crate::field::mul_by_x(want);
            }
        }
    }

    #[cfg(target_feature = "sha3")]
    #[test]
    fn round2_prescaled_matches_horner() {
        use crate::field::PHI_8_TABLE;
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        use crate::zerocheck::univariate_skip::build_eq;
        use crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash;

        const LO_SIZE: usize = 512;
        let mut state = 0x5052_4553_4341_4c45;
        let mut a_packed = vec![0u8; 16 * LO_SIZE];
        let mut b_packed = vec![0u8; 16 * LO_SIZE];
        for byte in a_packed.iter_mut().chain(b_packed.iter_mut()) {
            *byte = splitmix64(&mut state) as u8;
        }
        for group in [0usize, 15, 16, 31, 47, 59, 60, 64, 127, 200] {
            b_packed[group * 32..(group + 1) * 32].fill(0xff);
        }
        b_packed[5 * 32..5 * 32 + 16].fill(0xff);
        let random_z = F128::new(splitmix64(&mut state), splitmix64(&mut state));
        for (z_case, z) in [F128::ZERO, F128::ONE, PHI_8_TABLE[64], random_z]
            .into_iter()
            .enumerate()
        {
            let mut table = UniSkipFoldTable::new(6, z);
            if z_case == 3 {
                for bank in table.data.chunks_mut(256) {
                    for bit in 0..8 {
                        bank[1 << bit] = F128::new(splitmix64(&mut state), splitmix64(&mut state));
                    }
                    for value in 3usize..256 {
                        if !value.is_power_of_two() {
                            let low = value & value.wrapping_neg();
                            bank[value] = bank[value ^ low] + bank[low];
                        }
                    }
                }
            }
            for challenge_case in 0..3 {
                let mut r: Vec<F128> = (0..9)
                    .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                    .collect();
                r[2..6].copy_from_slice(&medium_challenges_ghash());
                match challenge_case {
                    0 => r[0] = F128::ZERO,
                    1 => r[1] = F128::ONE,
                    _ => {}
                }
                let eq_lo = build_eq(&r);
                assert_eq!(eq_lo.len(), LO_SIZE);
                for (mask, useful) in [(usize::MAX, usize::MAX), (127, 121)] {
                    for (degen, skip_dead) in
                        [(false, false), (true, false), (false, true), (true, true)]
                    {
                        for base in [0usize, 64, 121, 3 * LO_SIZE] {
                            let run = |variant: usize| {
                                let poison = F128::new(u64::MAX, u64::MAX);
                                let mut anchors = vec![poison; 2 * LO_SIZE];
                                let mut deltas = vec![0xa5u8; 16 * LO_SIZE];
                                let mut out = [poison; 8];
                                unsafe {
                                    let kernel = match variant {
                                        0 => fold_round2_compact_chunk_neon_horner_8_impl::<true, false>,
                                        1 => fold_round2_compact_chunk_neon_prescaled_8_impl::<false, false>,
                                        2 => fold_round2_compact_chunk_neon_prescaled_8_impl::<true, false>,
                                        3 => fold_round2_compact_chunk_neon_prescaled_8_impl::<false, true>,
                                        _ => fold_round2_compact_chunk_neon_prescaled_8_impl::<true, true>,
                                    };
                                    kernel(
                                        table.data.as_ptr().cast::<u8>(),
                                        a_packed.as_ptr(),
                                        b_packed.as_ptr(),
                                        anchors.as_mut_ptr(),
                                        deltas.as_mut_ptr(),
                                        eq_lo.as_ptr(),
                                        LO_SIZE,
                                        base,
                                        mask,
                                        useful,
                                        degen,
                                        skip_dead,
                                        out.as_mut_ptr(),
                                    );
                                }
                                (anchors, deltas, out)
                            };
                            let horner = run(0);
                            for variant in 1..=4 {
                                let prescaled = run(variant);
                                let at = format!("v {variant} z {z_case} r {challenge_case} mask {mask} base {base} degen {degen} skip {skip_dead}");
                                assert_eq!(prescaled.0, horner.0, "anchors: {at}");
                                assert_eq!(prescaled.1, horner.1, "deltas: {at}");
                                assert_eq!(prescaled.2, horner.2, "slots: {at}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// Calls the Horner kernel with or without the peeled first group.
    #[allow(clippy::too_many_arguments)]
    unsafe fn horner_first_shift_variant(
        skip: bool,
        table: &[F128],
        a: *const u8,
        b: *const u8,
        anchors: *mut F128,
        deltas: *mut u8,
        eq_lo: *const F128,
        lo_size: usize,
        base: usize,
        mask: usize,
        useful: usize,
        degen: bool,
        skip_dead: bool,
        out: *mut F128,
    ) {
        unsafe {
            let kernel = if skip {
                fold_round2_compact_chunk_neon_horner_8::<true>
            } else {
                fold_round2_compact_chunk_neon_horner_8::<false>
            };
            kernel(
                table.as_ptr().cast::<u8>(),
                a,
                b,
                anchors,
                deltas,
                eq_lo,
                lo_size,
                base,
                mask,
                useful,
                degen,
                skip_dead,
                out,
            );
        }
    }

    #[test]
    fn round2_horner_first_shift_skip_matches_unpeeled() {
        use crate::field::PHI_8_TABLE;
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        use crate::zerocheck::univariate_skip::build_eq;
        use crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash;

        const LO_SIZE: usize = 512;
        let mut state = 0x4649_5253_5453_4b50;
        let mut a_packed = vec![0u8; 16 * LO_SIZE];
        let mut b_packed = vec![0u8; 16 * LO_SIZE];
        for byte in a_packed.iter_mut().chain(b_packed.iter_mut()) {
            *byte = splitmix64(&mut state) as u8;
        }
        // Some all-ones B groups (the degenerate shortcut), including the
        // first group of a window, and one mixed group.
        for group in [0usize, 15, 16, 31, 47, 60, 64, 127, 200] {
            b_packed[group * 32..(group + 1) * 32].fill(0xff);
        }
        b_packed[5 * 32..5 * 32 + 16].fill(0xff);
        let random_z = F128::new(splitmix64(&mut state), splitmix64(&mut state));
        for z in [F128::ZERO, F128::ONE, PHI_8_TABLE[64], random_z] {
            let table = UniSkipFoldTable::new(6, z);
            for challenge_case in 0..3 {
                let mut r: Vec<F128> = (0..9)
                    .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                    .collect();
                r[2..6].copy_from_slice(&medium_challenges_ghash());
                match challenge_case {
                    0 => r[0] = F128::ZERO,
                    1 => r[1] = F128::ONE,
                    _ => {}
                }
                let eq_lo = build_eq(&r);
                assert_eq!(eq_lo.len(), LO_SIZE);
                for (mask, useful) in [(usize::MAX, usize::MAX), (127, 121)] {
                    for (degen, skip_dead) in
                        [(false, false), (true, false), (false, true), (true, true)]
                    {
                        for base in [0usize, 64, 121, 3 * LO_SIZE] {
                            let run = |skip: bool| {
                                let poison = F128::new(u64::MAX, u64::MAX);
                                let mut anchors = vec![poison; 2 * LO_SIZE];
                                let mut deltas = vec![0xa5u8; 16 * LO_SIZE];
                                let mut out = [poison; 8];
                                unsafe {
                                    horner_first_shift_variant(
                                        skip,
                                        &table.data,
                                        a_packed.as_ptr(),
                                        b_packed.as_ptr(),
                                        anchors.as_mut_ptr(),
                                        deltas.as_mut_ptr(),
                                        eq_lo.as_ptr(),
                                        LO_SIZE,
                                        base,
                                        mask,
                                        useful,
                                        degen,
                                        skip_dead,
                                        out.as_mut_ptr(),
                                    );
                                }
                                (anchors, deltas, out)
                            };
                            let plain = run(false);
                            let peeled = run(true);
                            let tag = format!("{challenge_case}/{mask}/{degen}/{skip_dead}/{base}");
                            assert_eq!(peeled.0, plain.0, "anchors at {tag}");
                            assert_eq!(peeled.1, plain.1, "deltas at {tag}");
                            assert_eq!(peeled.2, plain.2, "sums at {tag}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "timing probe"]
    fn round2_horner_first_shift_timing() {
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        use crate::zerocheck::univariate_skip::build_eq;
        use crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash;

        const LO: usize = 1 << 14;
        let mut state = 0x5036_5449_4d45_0001;
        let z = F128::new(splitmix64(&mut state), splitmix64(&mut state));
        let table = UniSkipFoldTable::new(6, z);
        let mut r: Vec<F128> = (0..14)
            .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
            .collect();
        r[2..6].copy_from_slice(&medium_challenges_ghash());
        let eq_lo = build_eq(&r);
        for (chunks, passes) in [(2usize, 24usize), (48, 1)] {
            let mut a = vec![0u8; 16 * LO * chunks];
            let mut b = vec![0u8; 16 * LO * chunks];
            for pair in 0..LO * chunks {
                if pair & 127 >= 121 {
                    continue;
                }
                for byte in a[16 * pair..16 * pair + 16]
                    .iter_mut()
                    .chain(b[16 * pair..16 * pair + 16].iter_mut())
                {
                    *byte = splitmix64(&mut state) as u8;
                }
            }
            let mut anchors = vec![F128::ZERO; 2 * LO * chunks];
            let mut deltas = vec![0u8; 16 * LO * chunks];
            let mut pass = |skip: bool| -> (f64, u64) {
                let mut sum = 0u64;
                let t = std::time::Instant::now();
                for _ in 0..passes {
                    for c in 0..chunks {
                        let mut out = [F128::ZERO; 8];
                        unsafe {
                            horner_first_shift_variant(
                                skip,
                                &table.data,
                                a.as_ptr().add(16 * LO * c),
                                b.as_ptr().add(16 * LO * c),
                                anchors.as_mut_ptr().add(2 * LO * c),
                                deltas.as_mut_ptr().add(16 * LO * c),
                                eq_lo.as_ptr(),
                                LO,
                                c * LO,
                                127,
                                121,
                                true,
                                true,
                                out.as_mut_ptr(),
                            );
                        }
                        for v in &out[4..] {
                            let w: [u64; 2] = unsafe { core::mem::transmute(*v) };
                            sum = (sum ^ w[0]).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ w[1];
                        }
                    }
                }
                let ns = t.elapsed().as_nanos() as f64 / (passes * chunks * LO) as f64;
                (ns, sum)
            };
            pass(false);
            let mut all: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let mut sums = [0u64; 2];
            for _ in 0..9 {
                for (i, skip) in [false, true].into_iter().enumerate() {
                    let (ns, sum) = pass(skip);
                    all[i].push(ns);
                    sums[i] = sum;
                }
            }
            assert_eq!(sums[0], sums[1]);
            for v in all.iter_mut() {
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            }
            println!(
                "r2 horner first-shift chunks={chunks}: min plain {:.3} peeled {:.3} ns/pair ({:+.2}%), median plain {:.3} peeled {:.3} ({:+.2}%)",
                all[0][0], all[1][0], (all[1][0] / all[0][0] - 1.0) * 100.0,
                all[0][4], all[1][4], (all[1][4] / all[0][4] - 1.0) * 100.0,
            );
        }
    }

    // Exact parent form from 70417ff326ebfde4d370b970ebdcf8f9edc69ace.
    #[target_feature(enable = "aes")]
    unsafe fn fold_round2_horner_parent_8(
        table_data: *const u8,
        a_packed: *const u8,
        b_packed: *const u8,
        anchors: *mut F128,
        deltas: *mut u8,
        eq_lo: *const F128,
        lo_size: usize,
        pair_idx_base: usize,
        pair_in_block_mask: usize,
        useful_pairs_inclusive: usize,
        degen: bool,
        skip_dead: bool,
        out: *mut F128,
    ) {
        use core::arch::aarch64::*;

        #[inline(always)]
        unsafe fn store_anchor_pair_nt(dst: *mut F128, a: uint64x2_t, b: uint64x2_t) {
            unsafe {
                core::arch::asm!(
                    "stnp {a:q}, {b:q}, [{dst}]",
                    dst = in(reg) dst,
                    a = in(vreg) a,
                    b = in(vreg) b,
                    options(nostack, preserves_flags),
                );
            }
        }

        #[inline(always)]
        unsafe fn shift_pair(a: &mut WideNeon, b: &mut WideNeon, top: &mut uint64x2_t) {
            unsafe {
                let zero = vdupq_n_u64(0);
                *top = vsriq_n_u64::<63>(vshlq_n_u64::<1>(*top), vzip2q_u64(a.hi, b.hi));
                a.hi = vsriq_n_u64::<63>(vshlq_n_u64::<1>(a.hi), vextq_u64::<1>(a.lo, a.hi));
                b.hi = vsriq_n_u64::<63>(vshlq_n_u64::<1>(b.hi), vextq_u64::<1>(b.lo, b.hi));
                a.lo = vsriq_n_u64::<63>(vshlq_n_u64::<1>(a.lo), vextq_u64::<1>(zero, a.lo));
                b.lo = vsriq_n_u64::<63>(vshlq_n_u64::<1>(b.lo), vextq_u64::<1>(zero, b.lo));
            }
        }

        #[inline(always)]
        unsafe fn reduce_pair(
            mut a: WideNeon,
            mut b: WideNeon,
            top: uint64x2_t,
        ) -> (uint64x2_t, uint64x2_t) {
            unsafe {
                let zero = vdupq_n_u64(0);
                // x^256 = 1 + x^2 + x^4 + x^14 modulo x^128+x^7+x^2+x+1.
                // Each top limb has at most 14 bits, so this cannot overflow.
                let folded = veorq_u64(
                    xor3_u64(top, vshlq_n_u64::<2>(top), vshlq_n_u64::<4>(top)),
                    vshlq_n_u64::<14>(top),
                );
                a.lo = veorq_u64(a.lo, vzip1q_u64(folded, zero));
                b.lo = veorq_u64(b.lo, vzip2q_u64(folded, zero));
                (reduce_wide_q(a), reduce_wide_q(b))
            }
        }

        unsafe {
            let zero = vdupq_n_u64(0);
            let mut sum0 = zero;
            let mut sum3 = zero;
            let mut sum4 = zero;
            let mut sum5 = zero;
            let b_ones = fold_row_q(table_data, [0xffu8; 8].as_ptr());
            let ones_is_one = is_zero_q(veorq_u64(
                b_ones,
                core::mem::transmute::<F128, uint64x2_t>(F128::ONE),
            ));

            for first in (0..lo_size / 2).step_by(32) {
                for t in 0..2 {
                    let mut w0 = WideNeon { lo: zero, hi: zero };
                    let mut w3 = WideNeon { lo: zero, hi: zero };
                    let mut w4 = WideNeon { lo: zero, hi: zero };
                    let mut w5 = WideNeon { lo: zero, hi: zero };
                    let mut top03 = zero;
                    let mut top45 = zero;
                    for b in (0..16).rev() {
                        shift_pair(&mut w0, &mut w3, &mut top03);
                        shift_pair(&mut w4, &mut w5, &mut top45);
                        let u = first + 2 * b + t;
                        let x_lo0 = 2 * u;
                        let x_lo1 = x_lo0 + 1;
                        let pad0 =
                            ((pair_idx_base + x_lo0) & pair_in_block_mask) >= useful_pairs_inclusive;
                        let pad1 =
                            ((pair_idx_base + x_lo1) & pair_in_block_mask) >= useful_pairs_inclusive;
                        // One pair: fold and store its even rows, hand back its odd
                        // row codes. A padded pair keeps its zero store and reads as
                        // all-zero codes.
                        let pair = |x_lo: usize, pad: bool| {
                            if pad {
                                if !skip_dead {
                                    store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                                    vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                                }
                                return (zero, zero, 0u64, 0u64, false);
                            }
                            let a_row = a_packed.add(2 * x_lo * 8).cast::<u64>();
                            let b_row = b_packed.add(2 * x_lo * 8).cast::<u64>();
                            let a_even = u64::from_le(core::ptr::read_unaligned(a_row));
                            let a_odd = u64::from_le(core::ptr::read_unaligned(a_row.add(1)));
                            let b_even = u64::from_le(core::ptr::read_unaligned(b_row));
                            let b_odd = u64::from_le(core::ptr::read_unaligned(b_row.add(1)));
                            let deg = degen && (b_even & b_odd) == u64::MAX;
                            let (a, b) = if deg {
                                (fold_row_code_q(table_data, a_even), b_ones)
                            } else {
                                fold_two_row_codes_q(table_data, a_even, b_even)
                            };
                            store_anchor_pair_nt(anchors.add(2 * x_lo), a, b);
                            let delta_pair = core::mem::transmute::<[u64; 2], uint64x2_t>([
                                a_even ^ a_odd,
                                b_even ^ b_odd,
                            ]);
                            vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);
                            (a, b, a_odd, b_odd, deg)
                        };
                        let (a0, b0, a1c, b1c, deg0) = pair(x_lo0, pad0);
                        let (a2, b2, a3c, b3c, deg1) = pair(x_lo1, pad1);
                        if pad0 & pad1 {
                            continue;
                        }
                        if deg0 & deg1 & ones_is_one {
                            // b == 1 on all four rows: only `a2 * b2 = a2` survives.
                            w0.lo = veorq_u64(w0.lo, a2);
                        } else {
                            // The fold is GF(2)-linear in the row code, so the odd
                            // rows enter only through fold(a1 ^ a3) and fold(b1 ^ b3):
                            // six row folds instead of eight. A degenerate pair's b
                            // rows are all-ones codes, which fold to `b_ones`.
                            let (o_a, o_b) = fold_two_row_codes_q(table_data, a1c ^ a3c, b1c ^ b3c);
                            let e_a = veorq_u64(a0, a2);
                            let e_b = veorq_u64(b0, b2);
                            if !pad1 {
                                wide_xor(&mut w0, mul_unreduced_sw_q(a2, swap_halves_q(a2), b2));
                            }
                            let all_a = veorq_u64(e_a, o_a);
                            let all_b = veorq_u64(e_b, o_b);
                            wide_xor(&mut w3, mul_unreduced_sw_q(e_a, swap_halves_q(e_a), e_b));
                            wide_xor(&mut w4, mul_unreduced_sw_q(o_a, swap_halves_q(o_a), o_b));
                            wide_xor(&mut w5, mul_unreduced_sw_q(all_a, swap_halves_q(all_a), all_b));
                        }
                    }
                    let lambda = vld1q_u64(eq_lo.add(2 * (first + t) + 1).cast::<u64>());
                    let (w0, w3) = reduce_pair(w0, w3, top03);
                    let (w4, w5) = reduce_pair(w4, w5, top45);
                    sum0 = veorq_u64(sum0, mul_q(lambda, w0));
                    sum3 = veorq_u64(sum3, mul_q(lambda, w3));
                    sum4 = veorq_u64(sum4, mul_q(lambda, w4));
                    sum5 = veorq_u64(sum5, mul_q(lambda, w5));
                }
            }
            for i in 0..4 {
                *out.add(i) = F128::ZERO;
            }
            *out.add(4) = core::mem::transmute::<uint64x2_t, F128>(sum0);
            *out.add(5) = core::mem::transmute::<uint64x2_t, F128>(sum3);
            *out.add(6) = core::mem::transmute::<uint64x2_t, F128>(sum4);
            *out.add(7) = core::mem::transmute::<uint64x2_t, F128>(sum5);
        }
    }

    #[test]
    fn round2_horner_leading_padding_matches_parent_outputs() {
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        use crate::zerocheck::univariate_skip::build_eq;
        use crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash;

        type Horner = unsafe fn(
            *const u8,
            *const u8,
            *const u8,
            *mut F128,
            *mut u8,
            *const F128,
            usize,
            usize,
            usize,
            usize,
            bool,
            bool,
            *mut F128,
        );
        let kernels: [Horner; 2] = [
            fold_round2_horner_parent_8,
            fold_round2_compact_chunk_neon_horner_8::<true>,
        ];
        let mut cases = Vec::new();
        for lo_size in [64usize, 128, 256] {
            for base in [0usize, 1, 63, 64, 120, 121, 122, 127, 128, 129, 255, 256] {
                cases.push((lo_size, base, 127usize, 121usize));
            }
        }
        for useful in [0usize, 120, 122, 128] {
            cases.push((256, 0, 127, useful));
        }
        cases.push((256, 0, usize::MAX, usize::MAX));
        for base in [0usize, 1, 1 << 14] {
            cases.push((1 << 14, base, 127, 121));
        }
        let mut state = 0x5232_4c45_4144_5041;
        for (lo_size, base, mask, useful) in cases {
            let mut a = vec![0u8; 16 * lo_size];
            let mut b = vec![0u8; 16 * lo_size];
            for byte in a.iter_mut().chain(b.iter_mut()) {
                *byte = splitmix64(&mut state) as u8;
            }
            for group in 0..lo_size / 2 {
                match group & 63 {
                    0 | 1 | 2 | 3 | 59 => b[group * 32..group * 32 + 32].fill(0xff),
                    4 => b[group * 32..group * 32 + 16].fill(0xff),
                    5 => b[group * 32 + 16..group * 32 + 32].fill(0xff),
                    _ => {}
                }
            }
            for instance in 0..lo_size / 128 {
                let offset = instance * 2048;
                for packed in [&mut a, &mut b] {
                    packed[offset + 1926] &= 1;
                    packed[offset + 1927..offset + 1936].fill(0);
                }
            }
            let mut r: Vec<F128> = (0..lo_size.ilog2())
                .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                .collect();
            r[2..6].copy_from_slice(&medium_challenges_ghash());
            let eq = build_eq(&r);
            for table_case in 0..3 {
                let mut table = UniSkipFoldTable::new(6, F128::new(0x1234_5678, 0x9abc_def0));
                if table_case == 1 {
                    for value in &mut table.data {
                        *value = *value * F128::new(3, 0);
                    }
                } else if table_case == 2 {
                    table.data.fill(F128::ZERO);
                }
                for degen in [false, true] {
                    for skip_dead in [false, true] {
                        let poison = F128::new(u64::MAX, u64::MAX);
                        let run = |kernel: Horner| {
                            // Include guards and compare every stored slot.
                            let mut anchors = vec![poison; 2 * lo_size + 4];
                            let mut deltas = vec![0xa5u8; 16 * lo_size + 32];
                            let mut out = [poison; 12];
                            unsafe {
                                kernel(
                                    table.data.as_ptr().cast::<u8>(),
                                    a.as_ptr(),
                                    b.as_ptr(),
                                    anchors.as_mut_ptr().add(2),
                                    deltas.as_mut_ptr().add(16),
                                    eq.as_ptr(),
                                    lo_size,
                                    base,
                                    mask,
                                    useful,
                                    degen,
                                    skip_dead,
                                    out.as_mut_ptr().add(2),
                                );
                            }
                            (anchors, deltas, out)
                        };
                        let expected = run(kernels[0]);
                        let actual = run(kernels[1]);
                        assert_eq!(
                            actual.0, expected.0,
                            "anchors: lo={lo_size} base={base} useful={useful} table={table_case} degen={degen} dead={skip_dead}"
                        );
                        assert_eq!(
                            actual.1, expected.1,
                            "deltas: lo={lo_size} base={base} useful={useful} table={table_case} degen={degen} dead={skip_dead}"
                        );
                        assert_eq!(
                            actual.2, expected.2,
                            "accumulators: lo={lo_size} base={base} useful={useful} table={table_case} degen={degen} dead={skip_dead}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn round2_horner_odd_schedule_matches_parent_outputs() {
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        use crate::zerocheck::univariate_skip::build_eq;
        use crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash;

        type Horner = unsafe fn(
            *const u8,
            *const u8,
            *const u8,
            *mut F128,
            *mut u8,
            *const F128,
            usize,
            usize,
            usize,
            usize,
            bool,
            bool,
            *mut F128,
        );
        let kernels: [Horner; 2] = [
            fold_round2_horner_parent_8,
            fold_round2_compact_chunk_neon_horner_8::<true>,
        ];
        let mut cases = Vec::new();
        for lo_size in [64usize, 128, 256] {
            for base in [0usize, 1, 63, 64, 120, 121, 122, 127, 128, 129, 255, 256] {
                cases.push((lo_size, base, 127usize, 121usize));
            }
        }
        for useful in [0usize, 120, 122, 128] {
            cases.push((256, 0, 127, useful));
        }
        cases.push((256, 0, usize::MAX, usize::MAX));
        for base in [0usize, 1, 1 << 14] {
            cases.push((1 << 14, base, 127, 121));
        }
        let mut state = 0x5232_4c45_4144_5041;
        for (lo_size, base, mask, useful) in cases {
            let mut a = vec![0u8; 16 * lo_size];
            let mut b = vec![0u8; 16 * lo_size];
            for byte in a.iter_mut().chain(b.iter_mut()) {
                *byte = splitmix64(&mut state) as u8;
            }
            for group in 0..lo_size / 2 {
                match group & 63 {
                    0 | 1 | 2 | 3 | 59 => b[group * 32..group * 32 + 32].fill(0xff),
                    4 => b[group * 32..group * 32 + 16].fill(0xff),
                    5 => b[group * 32 + 16..group * 32 + 32].fill(0xff),
                    _ => {}
                }
            }
            for instance in 0..lo_size / 128 {
                let offset = instance * 2048;
                for packed in [&mut a, &mut b] {
                    packed[offset + 1926] &= 1;
                    packed[offset + 1927..offset + 1936].fill(0);
                }
            }
            let mut r: Vec<F128> = (0..lo_size.ilog2())
                .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                .collect();
            r[2..6].copy_from_slice(&medium_challenges_ghash());
            let eq = build_eq(&r);
            for table_case in 0..3 {
                let mut table = UniSkipFoldTable::new(6, F128::new(0x1234_5678, 0x9abc_def0));
                if table_case == 1 {
                    for value in &mut table.data {
                        *value = *value * F128::new(3, 0);
                    }
                } else if table_case == 2 {
                    table.data.fill(F128::ZERO);
                }
                for degen in [false, true] {
                    for skip_dead in [false, true] {
                        let poison = F128::new(u64::MAX, u64::MAX);
                        let run = |kernel: Horner| {
                            // Include guards and compare every stored slot.
                            let mut anchors = vec![poison; 2 * lo_size + 4];
                            let mut deltas = vec![0xa5u8; 16 * lo_size + 32];
                            let mut out = [poison; 12];
                            unsafe {
                                kernel(
                                    table.data.as_ptr().cast::<u8>(),
                                    a.as_ptr(),
                                    b.as_ptr(),
                                    anchors.as_mut_ptr().add(2),
                                    deltas.as_mut_ptr().add(16),
                                    eq.as_ptr(),
                                    lo_size,
                                    base,
                                    mask,
                                    useful,
                                    degen,
                                    skip_dead,
                                    out.as_mut_ptr().add(2),
                                );
                            }
                            (anchors, deltas, out)
                        };
                        let expected = run(kernels[0]);
                        let actual = run(kernels[1]);
                        assert_eq!(
                            actual.0, expected.0,
                            "anchors: lo={lo_size} base={base} useful={useful} table={table_case} degen={degen} dead={skip_dead}"
                        );
                        assert_eq!(
                            actual.1, expected.1,
                            "deltas: lo={lo_size} base={base} useful={useful} table={table_case} degen={degen} dead={skip_dead}"
                        );
                        assert_eq!(
                            actual.2, expected.2,
                            "accumulators: lo={lo_size} base={base} useful={useful} table={table_case} degen={degen} dead={skip_dead}"
                        );
                    }
                }
            }
        }
    }

    fn fold4_scalar(values: &[F128], i: usize, rho_a: F128, rho_b: F128) -> F128 {
        let t0 = values[i] + rho_a * (values[i] + values[i + 1]);
        let t1 = values[i + 2] + rho_a * (values[i + 2] + values[i + 3]);
        t0 + rho_b * (t0 + t1)
    }

    #[test]
    fn clmul_q_products_match_scalar_field_products() {
        let mut state = 0x4b41_5241_5453_5542;

        unsafe {
            for _ in 0..256 {
                let a = F128::new(splitmix64(&mut state), splitmix64(&mut state));
                let b = F128::new(splitmix64(&mut state), splitmix64(&mut state));
                let a_q = core::mem::transmute::<F128, uint64x2_t>(a);
                let b_q = core::mem::transmute::<F128, uint64x2_t>(b);

                let (p0, cross, p2) = clmul_products_q(a_q, b_q);
                let product = WideNeon {
                    lo: core::arch::aarch64::veorq_u64(
                        p0,
                        core::arch::aarch64::vextq_u64::<1>(
                            core::arch::aarch64::vdupq_n_u64(0),
                            cross,
                        ),
                    ),
                    hi: core::arch::aarch64::veorq_u64(
                        p2,
                        core::arch::aarch64::vextq_u64::<1>(
                            cross,
                            core::arch::aarch64::vdupq_n_u64(0),
                        ),
                    ),
                };
                let product_lo = core::mem::transmute::<uint64x2_t, [u64; 2]>(product.lo);
                let product_hi = core::mem::transmute::<uint64x2_t, [u64; 2]>(product.hi);
                let expected_unreduced = a.mul_unreduced(b);

                assert_eq!(
                    F256Unreduced {
                        r0: product_lo[0],
                        r1: product_lo[1],
                        r2: product_hi[0],
                        r3: product_hi[1],
                    },
                    expected_unreduced
                );
                assert_eq!(
                    core::mem::transmute::<uint64x2_t, F128>(mul_q(a_q, b_q)),
                    a * b
                );

                let unreduced = mul_unreduced_q(a_q, b_q);
                let unreduced_lo = core::mem::transmute::<uint64x2_t, [u64; 2]>(unreduced.lo);
                let unreduced_hi = core::mem::transmute::<uint64x2_t, [u64; 2]>(unreduced.hi);
                assert_eq!(
                    F256Unreduced {
                        r0: unreduced_lo[0],
                        r1: unreduced_lo[1],
                        r2: unreduced_hi[0],
                        r3: unreduced_hi[1],
                    },
                    expected_unreduced
                );
            }
        }
    }

    #[test]
    fn expanded_fold4_pair_matches_nested_scalar() {
        let mut state = 0x4558_5041_4e44_3446;

        for case in 0..256 {
            let rho_a = match case {
                0 => F128::ZERO,
                1 => F128::ONE,
                _ => F128::new(splitmix64(&mut state), splitmix64(&mut state)),
            };
            let rho_b = match case {
                2 => F128::ZERO,
                3 => F128::ONE,
                _ => F128::new(splitmix64(&mut state), splitmix64(&mut state)),
            };
            let mut values = [F128::ZERO; 8];
            for value in &mut values {
                *value = F128::new(splitmix64(&mut state), splitmix64(&mut state));
            }

            let (actual0, actual1) = unsafe {
                fold4_pair_expanded_binius_q(
                    values.as_ptr(),
                    core::mem::transmute::<F128, uint64x2_t>(rho_a),
                    core::mem::transmute::<F128, uint64x2_t>(rho_b),
                    core::mem::transmute::<F128, uint64x2_t>(rho_a * rho_b),
                )
            };
            let actual = unsafe {
                [
                    core::mem::transmute::<uint64x2_t, F128>(actual0),
                    core::mem::transmute::<uint64x2_t, F128>(actual1),
                ]
            };
            assert_eq!(
                actual,
                [
                    fold4_scalar(&values, 0, rho_a, rho_b),
                    fold4_scalar(&values, 4, rho_a, rho_b),
                ],
                "expanded pair mismatch at case {case}"
            );
        }
    }

    #[test]
    fn expanded_direct_normal_matches_scalar_composed_pass() {
        const LO_SIZE: usize = 8;
        let mut state = 0x4449_5245_4354_3446;

        for case in 0..16 {
            let rho_a = match case {
                0 => F128::ZERO,
                1 => F128::ONE,
                _ => F128::new(splitmix64(&mut state), splitmix64(&mut state)),
            };
            let rho_b = match case {
                2 => F128::ZERO,
                3 => F128::ONE,
                _ => F128::new(splitmix64(&mut state), splitmix64(&mut state)),
            };
            let a_in: Vec<F128> = (0..8 * LO_SIZE)
                .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                .collect();
            let b_in: Vec<F128> = (0..8 * LO_SIZE)
                .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                .collect();
            let eq_lo: Vec<F128> = (0..LO_SIZE)
                .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                .collect();
            let mut actual_a = vec![F128::ZERO; 2 * LO_SIZE];
            let mut actual_b = vec![F128::ZERO; 2 * LO_SIZE];
            let actual = fold2_and_message_normal_expanded_aarch64(
                &a_in,
                &b_in,
                &mut actual_a,
                &mut actual_b,
                rho_a,
                rho_b,
                rho_a * rho_b,
                &eq_lo,
            );

            let mut expected_a = vec![F128::ZERO; 2 * LO_SIZE];
            let mut expected_b = vec![F128::ZERO; 2 * LO_SIZE];
            let mut p1_acc = F256Unreduced::ZERO;
            let mut pinf_acc = F256Unreduced::ZERO;
            for (x_lo, &eq_l) in eq_lo.iter().enumerate() {
                let i = 8 * x_lo;
                let o = 2 * x_lo;
                let a0 = fold4_scalar(&a_in, i, rho_a, rho_b);
                let a1 = fold4_scalar(&a_in, i + 4, rho_a, rho_b);
                let b0 = fold4_scalar(&b_in, i, rho_a, rho_b);
                let b1 = fold4_scalar(&b_in, i + 4, rho_a, rho_b);
                expected_a[o] = a0;
                expected_a[o + 1] = a1;
                expected_b[o] = b0;
                expected_b[o + 1] = b1;
                p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
            }
            let expected = (p1_acc.reduce(), pinf_acc.reduce());

            assert_eq!(actual_a, expected_a, "A mismatch at case {case}");
            assert_eq!(actual_b, expected_b, "B mismatch at case {case}");
            assert_eq!(actual, expected, "message mismatch at case {case}");
        }
    }

    #[test]
    fn expanded_lookahead_store_variants_match_incumbent_pass() {
        const LO_SIZE: usize = 8;
        let mut state = 0x4e54_4c4f_4f4b_4148;

        for case in 0..16 {
            let rho_a = match case {
                0 => F128::ZERO,
                1 => F128::ONE,
                _ => F128::new(splitmix64(&mut state), splitmix64(&mut state)),
            };
            let rho_b = match case {
                2 => F128::ZERO,
                3 => F128::ONE,
                _ => F128::new(splitmix64(&mut state), splitmix64(&mut state)),
            };
            let a_in: Vec<F128> = (0..8 * LO_SIZE)
                .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                .collect();
            let b_in: Vec<F128> = (0..8 * LO_SIZE)
                .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                .collect();
            let eq_lo: Vec<F128> = (0..LO_SIZE)
                .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
                .collect();
            let mut candidate_a = vec![F128::ZERO; 2 * LO_SIZE];
            let mut candidate_b = vec![F128::ZERO; 2 * LO_SIZE];
            let mut normal_a = vec![F128::ZERO; 2 * LO_SIZE];
            let mut normal_b = vec![F128::ZERO; 2 * LO_SIZE];
            let mut incumbent_a = vec![F128::ZERO; 2 * LO_SIZE];
            let mut incumbent_b = vec![F128::ZERO; 2 * LO_SIZE];
            let rho_ab = rho_a * rho_b;

            let candidate = fold2_and_message_lookahead_nt_expanded_aarch64(
                &a_in,
                &b_in,
                &mut candidate_a,
                &mut candidate_b,
                rho_a,
                rho_b,
                rho_ab,
                &eq_lo,
            );
            let normal = fold2_and_message_lookahead_normal_expanded_aarch64(
                &a_in,
                &b_in,
                &mut normal_a,
                &mut normal_b,
                rho_a,
                rho_b,
                rho_ab,
                &eq_lo,
            );
            let incumbent = fold2_and_message_lookahead_aarch64(
                &a_in,
                &b_in,
                &mut incumbent_a,
                &mut incumbent_b,
                rho_a,
                rho_b,
                &eq_lo,
                false,
            );

            assert_eq!(candidate_a, incumbent_a, "A mismatch at case {case}");
            assert_eq!(candidate_b, incumbent_b, "B mismatch at case {case}");
            assert_eq!(candidate, incumbent, "message mismatch at case {case}");
            assert_eq!(normal_a, incumbent_a, "normal A mismatch at case {case}");
            assert_eq!(normal_b, incumbent_b, "normal B mismatch at case {case}");
            assert_eq!(normal, incumbent, "normal message mismatch at case {case}");
        }
    }

    #[test]
    fn round2_periodic_padding_matches_generic_schedule() {
        const LO_SIZE: usize = 128;
        const N_CHUNKS: usize = 8;

        let mut state = 0x5045_5249_4f44_4943;
        let table: Vec<F128> = (0..N_CHUNKS * 256)
            .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
            .collect();
        let mut a_packed = vec![0u8; 2 * LO_SIZE * N_CHUNKS];
        let mut b_packed = vec![0u8; 2 * LO_SIZE * N_CHUNKS];
        for byte in a_packed.iter_mut().chain(b_packed.iter_mut()) {
            *byte = splitmix64(&mut state) as u8;
        }
        let eq_lo: Vec<F128> = (0..LO_SIZE)
            .map(|_| F128::new(splitmix64(&mut state), splitmix64(&mut state)))
            .collect();

        let run = |periodic_padding: bool, skip_dead: bool| {
            let poison = F128::new(0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555);
            let mut anchors = vec![poison; 2 * LO_SIZE];
            let mut deltas = vec![0xa5u8; 2 * LO_SIZE * N_CHUNKS];
            let mut out = [poison; 8];
            unsafe {
                fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                    table.as_ptr().cast::<u8>(),
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    anchors.as_mut_ptr(),
                    deltas.as_mut_ptr(),
                    eq_lo.as_ptr(),
                    LO_SIZE,
                    0,
                    127,
                    121,
                    true,
                    periodic_padding,
                    skip_dead,
                    out.as_mut_ptr(),
                );
            }
            (anchors, deltas, out)
        };

        let periodic = run(true, false);
        let generic = run(false, false);
        assert_eq!(periodic.0, generic.0, "anchor schedule mismatch");
        assert_eq!(periodic.1, generic.1, "delta schedule mismatch");
        assert_eq!(periodic.2, generic.2, "message/lookahead mismatch");

        // Dead-pair skip: both schedules leave every padded slot untouched
        // (poison survives), write the live slots identically to the dense
        // run, and produce the same message/lookahead sums.
        let poison = F128::new(0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555);
        for periodic_padding in [true, false] {
            let sparse = run(periodic_padding, true);
            assert_eq!(sparse.2, periodic.2, "skip message/lookahead mismatch");
            for x_lo in 0..LO_SIZE {
                let dead = (x_lo & 127) >= 121;
                let anc = &sparse.0[2 * x_lo..2 * x_lo + 2];
                let del = &sparse.1[16 * x_lo..16 * x_lo + 16];
                if dead {
                    assert_eq!(anc, &[poison; 2], "dead anchor written at {x_lo}");
                    assert!(del.iter().all(|&b| b == 0xa5), "dead delta written at {x_lo}");
                } else {
                    assert_eq!(anc, &periodic.0[2 * x_lo..2 * x_lo + 2], "anchor at {x_lo}");
                    assert_eq!(del, &periodic.1[16 * x_lo..16 * x_lo + 16], "delta at {x_lo}");
                }
            }
        }
    }

    /// The Karatsuba cross recombination used by [`LookaheadTriAcc::add_group`]
    /// is bit-identical to the schoolbook [`mul_sw_tri_binius_q`]: 512 random
    /// pairs plus zero/one edges, compared lane by lane.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn mul_sw_tri_binius_karatsuba_matches_schoolbook() {
        use core::arch::aarch64::*;
        let mut state = 0x8E5C_AD3E_1ECA_57ED;
        let mut next = || splitmix64(&mut state);
        let mut check = |a: [u64; 2], b: [u64; 2]| unsafe {
            let av: uint64x2_t = std::mem::transmute(a);
            let bv: uint64x2_t = std::mem::transmute(b);
            let expected = mul_sw_tri_binius_q(av, swap_halves_q(av), bv);
            let a_sum = vgetq_lane_u64::<0>(av) ^ vgetq_lane_u64::<1>(av);
            let actual = mul_sw_tri_binius_karatsuba_q(av, a_sum, bv);
            assert_eq!(
                (
                    vgetq_lane_u64::<0>(actual),
                    vgetq_lane_u64::<1>(actual)
                ),
                (
                    vgetq_lane_u64::<0>(expected),
                    vgetq_lane_u64::<1>(expected)
                ),
            );
        };
        for _ in 0..512 {
            check([next(), next()], [next(), next()]);
        }
        for edge in [[0, 0], [1, 0], [0, 1], [u64::MAX, u64::MAX], [1, 1]] {
            check(edge, [next(), next()]);
            check([next(), next()], edge);
        }
    }

    #[cfg(target_feature = "sha3")]
    #[test]
    fn k_fused_one_table_matches_two_tables() {
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        let mut state = 0x4b31_5441_424c_4521;
        let rho = |state: &mut u64| F128::new(splitmix64(state), splitmix64(state));
        let table = UniSkipFoldTable::new(6, rho(&mut state));
        const OUT_PAIRS: usize = 2 * 16 * 3;
        for case in 0..6 {
            let (rho1, rho2) = match case {
                0 => (F128::ONE, F128::ONE),
                1 => (rho(&mut state), F128::ZERO),
                _ => (rho(&mut state), rho(&mut state)),
            };
            let l1 = table.scaled_linear(rho1 * (F128::ONE + rho2));
            let l3 = table.scaled_linear(rho1 * rho2);
            let r1 = table.scaled_linear(rho1);
            let mut anchors: Vec<F128> = (0..8 * OUT_PAIRS).map(|_| rho(&mut state)).collect();
            let mut deltas: Vec<u8> =
                (0..64 * OUT_PAIRS).map(|_| splitmix64(&mut state) as u8).collect();
            let eq: Vec<F128> = (0..OUT_PAIRS).map(|_| rho(&mut state)).collect();
            for g in 0..4 * OUT_PAIRS / 2 {
                let class = if (g / 4) % 7 == 3 { 1 } else { splitmix64(&mut state) % 5 };
                match class {
                    0 => {
                        anchors[4 * g + 3] = anchors[4 * g + 1];
                        deltas[32 * g + 8..32 * g + 16].fill(0);
                        deltas[32 * g + 24..32 * g + 32].fill(0);
                    }
                    1 => {
                        anchors[4 * g + 1] = F128::ONE;
                        anchors[4 * g + 3] = F128::ONE;
                        deltas[32 * g + 8..32 * g + 16].fill(0);
                        deltas[32 * g + 24..32 * g + 32].fill(0);
                    }
                    2 => {
                        deltas[32 * g + 8..32 * g + 16].fill(0);
                        deltas[32 * g + 24..32 * g + 32].fill(0);
                    }
                    _ => {}
                }
            }
            for dead_tail in [false, true] {
                let run = |two: bool| {
                    let mut a = vec![F128::new(7, 7); 2 * OUT_PAIRS];
                    let mut b = vec![F128::new(9, 9); 2 * OUT_PAIRS];
                    let mut out = [F128::ZERO; 8];
                    unsafe {
                        match (two, dead_tail) {
                            (true, false) => fold2_compact_and_round45_chunk_neon_8_fused::<false>(
                                l1.as_ptr().cast(), l3.as_ptr().cast(), rho2, anchors.as_ptr(),
                                deltas.as_ptr(), a.as_mut_ptr(), b.as_mut_ptr(), eq.as_ptr(),
                                OUT_PAIRS, true, out.as_mut_ptr(),
                            ),
                            (true, true) => fold2_compact_and_round45_chunk_neon_8_fused::<true>(
                                l1.as_ptr().cast(), l3.as_ptr().cast(), rho2, anchors.as_ptr(),
                                deltas.as_ptr(), a.as_mut_ptr(), b.as_mut_ptr(), eq.as_ptr(),
                                OUT_PAIRS, true, out.as_mut_ptr(),
                            ),
                            (false, false) => fold2_compact_and_round45_chunk_neon_8_fused_r1::<false, false>(
                                r1.as_ptr().cast(), rho2, anchors.as_ptr(), deltas.as_ptr(),
                                a.as_mut_ptr(), b.as_mut_ptr(), eq.as_ptr(), OUT_PAIRS, true,
                                out.as_mut_ptr(),
                            ),
                            (false, true) => fold2_compact_and_round45_chunk_neon_8_fused_r1::<true, false>(
                                r1.as_ptr().cast(), rho2, anchors.as_ptr(), deltas.as_ptr(),
                                a.as_mut_ptr(), b.as_mut_ptr(), eq.as_ptr(), OUT_PAIRS, true,
                                out.as_mut_ptr(),
                            ),
                        }
                    }
                    (a, b, out)
                };
                let (a2, b2, o2) = run(true);
                let (a1, b1, o1) = run(false);
                assert_eq!(a1, a2, "a outputs, case {case}, dead_tail {dead_tail}");
                assert_eq!(b1, b2, "b outputs, case {case}, dead_tail {dead_tail}");
                assert_eq!(o1, o2, "slots, case {case}, dead_tail {dead_tail}");
            }
        }
    }

    /// Logical input `i` of a dead-tail-packed array (`PACK_SUPER` layout),
    /// or `None` for the unstored zeros (outputs 61..63 of a block).
    fn packed_index(i: usize) -> Option<usize> {
        let (blk, j) = (i / 64, i % 64);
        let (sup, w) = (blk / PACK_BLOCKS, blk % PACK_BLOCKS);
        match j {
            0..60 => Some(PACK_SUPER * sup + 60 * w + j),
            60 => Some(PACK_SUPER * sup + 60 * PACK_BLOCKS + w),
            _ => None,
        }
    }

    /// The dead-tail-packed K output (`PACK`) holds exactly the incumbent's
    /// outputs (and the incumbent's outputs 61..63 of every block are zero);
    /// the eight slots are identical; entries past the packed length are not
    /// written.
    #[cfg(target_feature = "sha3")]
    #[test]
    fn k_dead_pack_matches_full_output() {
        use crate::zerocheck::multilinear::UniSkipFoldTable;
        let mut state = 0x4b31_5041_434b_3631;
        let rho = |state: &mut u64| F128::new(splitmix64(state), splitmix64(state));
        let table = UniSkipFoldTable::new(6, rho(&mut state));
        for case in 0..12 {
            // Two and one superblocks of four 64-output blocks.
            let out_pairs = if case % 2 == 0 { 2 * 16 * 8 } else { 2 * 16 * 4 };
            let n_out = 2 * out_pairs;
            let (rho1, rho2) = match case / 2 {
                0 => (F128::ONE, F128::ONE),
                1 => (rho(&mut state), F128::ZERO),
                _ => (rho(&mut state), rho(&mut state)),
            };
            let r1 = table.scaled_linear(rho1);
            let mut anchors: Vec<F128> = (0..8 * out_pairs).map(|_| rho(&mut state)).collect();
            let mut deltas: Vec<u8> =
                (0..64 * out_pairs).map(|_| splitmix64(&mut state) as u8).collect();
            let eq: Vec<F128> = (0..out_pairs).map(|_| rho(&mut state)).collect();
            for g in 0..4 * out_pairs / 2 {
                let class = if (g / 4) % 7 == 3 { 1 } else { splitmix64(&mut state) % 5 };
                match class {
                    0 => {
                        anchors[4 * g + 3] = anchors[4 * g + 1];
                        deltas[32 * g + 8..32 * g + 16].fill(0);
                        deltas[32 * g + 24..32 * g + 32].fill(0);
                    }
                    1 => {
                        anchors[4 * g + 1] = F128::ONE;
                        anchors[4 * g + 3] = F128::ONE;
                        deltas[32 * g + 8..32 * g + 16].fill(0);
                        deltas[32 * g + 24..32 * g + 32].fill(0);
                    }
                    2 => {
                        deltas[32 * g + 8..32 * g + 16].fill(0);
                        deltas[32 * g + 24..32 * g + 32].fill(0);
                    }
                    _ => {}
                }
            }
            for degen in [false, true] {
                let poison = F128::new(0xdead, 0xbeef);
                let (mut a, mut b) = (vec![poison; n_out], vec![poison; n_out]);
                let (mut ap, mut bp) = (vec![poison; n_out], vec![poison; n_out]);
                let (mut o_full, mut o_pack) = ([F128::ZERO; 8], [F128::ZERO; 8]);
                unsafe {
                    fold2_compact_and_round45_chunk_neon_8_fused_r1::<true, false>(
                        r1.as_ptr().cast(), rho2, anchors.as_ptr(), deltas.as_ptr(),
                        a.as_mut_ptr(), b.as_mut_ptr(), eq.as_ptr(), out_pairs, degen,
                        o_full.as_mut_ptr(),
                    );
                    fold2_compact_and_round45_chunk_neon_8_fused_r1::<true, true>(
                        r1.as_ptr().cast(), rho2, anchors.as_ptr(), deltas.as_ptr(),
                        ap.as_mut_ptr(), bp.as_mut_ptr(), eq.as_ptr(), out_pairs, degen,
                        o_pack.as_mut_ptr(),
                    );
                }
                assert_eq!(o_pack, o_full, "slots, case {case}, degen {degen}");
                for i in 0..n_out {
                    match packed_index(i) {
                        Some(k) => {
                            assert_eq!(ap[k], a[i], "a output {i}, case {case}");
                            assert_eq!(bp[k], b[i], "b output {i}, case {case}");
                        }
                        None => {
                            assert_eq!(a[i], F128::ZERO, "a output {i} not zero");
                            assert_eq!(b[i], F128::ZERO, "b output {i} not zero");
                        }
                    }
                }
                let packed_len = n_out / 256 * PACK_SUPER;
                assert!(ap[packed_len..].iter().all(|&v| v == poison));
                assert!(bp[packed_len..].iter().all(|&v| v == poison));
            }
        }
    }

    /// The tri pass over a dead-tail-packed input equals the tri pass over
    /// the full input with zeros at logical inputs 61..63 of every block:
    /// every output and all 26 slots, both store flavours.
    #[test]
    fn tri_packed_input_matches_full_input() {
        let mut state = 0x7472_6970_6163_6b31;
        let rho = |state: &mut u64| F128::new(splitmix64(state), splitmix64(state));
        for case in 0..4 {
            let n_out = 64 << case;
            let n_in = 4 * n_out;
            let mut a: Vec<F128> = (0..n_in).map(|_| rho(&mut state)).collect();
            let mut b: Vec<F128> = (0..n_in).map(|_| rho(&mut state)).collect();
            for blk in 0..n_in / 64 {
                a[64 * blk + 61..64 * blk + 64].fill(F128::ZERO);
                b[64 * blk + 61..64 * blk + 64].fill(F128::ZERO);
            }
            let pack = |v: &[F128]| -> Vec<F128> {
                let mut p = vec![F128::new(1, 2); n_in / 256 * PACK_SUPER];
                for (i, &x) in v.iter().enumerate() {
                    if let Some(k) = packed_index(i) {
                        p[k] = x;
                    }
                }
                p
            };
            let (ap_in, bp_in) = (pack(&a), pack(&b));
            let eq: Vec<F128> = (0..n_out / 2).map(|_| rho(&mut state)).collect();
            let (ra, rb) = (rho(&mut state), rho(&mut state));
            for nt in [false, true] {
                let (mut ao, mut bo) = (vec![F128::ZERO; n_out], vec![F128::ZERO; n_out]);
                let (mut ap, mut bp) = (vec![F128::ZERO; n_out], vec![F128::ZERO; n_out]);
                let (full, packed) = if nt {
                    (
                        fold_tri_and_message_aarch64::<2, true, 0>(&a, &b, &mut ao, &mut bo, ra, rb, F128::ZERO, &eq),
                        fold_tri_and_message_packed_aarch64::<true>(&ap_in, &bp_in, &mut ap, &mut bp, ra, rb, &eq),
                    )
                } else {
                    (
                        fold_tri_and_message_aarch64::<2, false, 0>(&a, &b, &mut ao, &mut bo, ra, rb, F128::ZERO, &eq),
                        fold_tri_and_message_packed_aarch64::<false>(&ap_in, &bp_in, &mut ap, &mut bp, ra, rb, &eq),
                    )
                };
                assert_eq!(packed, full, "slots, case {case}, nt {nt}");
                assert_eq!(ap, ao, "a outputs, case {case}, nt {nt}");
                assert_eq!(bp, bo, "b outputs, case {case}, nt {nt}");
            }
        }
    }
}

/// Accumulator of unreduced F128 products for the generic sumcheck kernels:
/// `acc.fma(a, b)` is `acc ^= a.mul_unreduced(b)` and `acc.reduce()` is
/// `F256Unreduced::reduce`. Here it keeps the three partial-product sums of
/// [`TriNeon`] and accumulates with the fused [`tri_acc_q`] (5 µops per
/// product). The reduced value is identical: the tri-to-wide map is linear
/// over XOR and `reduce_wide_q` is the same GHASH reduction.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[derive(Clone, Copy)]
pub(crate) struct ProdAcc(TriNeon);

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
impl ProdAcc {
    // SAFETY: an all-zero bit pattern is a valid `uint64x2_t`.
    const Z: core::arch::aarch64::uint64x2_t =
        unsafe { core::mem::transmute::<[u64; 2], core::arch::aarch64::uint64x2_t>([0, 0]) };
    pub(crate) const ZERO: Self = Self(TriNeon {
        ll: Self::Z,
        cross: Self::Z,
        hh: Self::Z,
    });

    #[inline(always)]
    pub(crate) fn fma(&mut self, a: F128, b: F128) {
        // SAFETY: F128 and uint64x2_t are both 16 plain bytes; register-only asm.
        unsafe {
            tri_acc_q(
                &mut self.0,
                core::mem::transmute::<F128, core::arch::aarch64::uint64x2_t>(a),
                core::mem::transmute::<F128, core::arch::aarch64::uint64x2_t>(b),
            );
        }
    }

    #[inline(always)]
    pub(crate) fn reduce(self) -> F128 {
        // SAFETY: as above.
        unsafe { core::mem::transmute::<core::arch::aarch64::uint64x2_t, F128>(reduce_tri_q(self.0)) }
    }
}

/// Generic-AArch64 (no SHA3) twin: the plain `F256Unreduced` accumulator.
#[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
#[derive(Clone, Copy)]
pub(crate) struct ProdAcc(F256Unreduced);

#[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
impl ProdAcc {
    pub(crate) const ZERO: Self = Self(F256Unreduced::ZERO);

    #[inline(always)]
    pub(crate) fn fma(&mut self, a: F128, b: F128) {
        self.0 ^= a.mul_unreduced(b);
    }

    #[inline(always)]
    pub(crate) fn reduce(self) -> F128 {
        self.0.reduce()
    }
}

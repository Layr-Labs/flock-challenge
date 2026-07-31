use crate::field::F128;
use core::arch::aarch64::*;
use core::mem::transmute;

#[derive(Clone, Copy)]
struct WideNeon {
    lo: uint64x2_t,
    hi: uint64x2_t,
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn pmull(a: u64, b: u64) -> uint64x2_t {
    // SAFETY: the caller provides the `aes` target feature; both types are
    // 128-bit bit containers with compatible alignment.
    unsafe { transmute::<u128, uint64x2_t>(vmull_p64(a, b)) }
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
        let c0 = veorq_u64(veorq_u64(p0_mm, p0_ll), p0_hh);
        let c1 = veorq_u64(veorq_u64(p1_mm, p1_ll), p1_hh);

        // Pack product 0/1 into lanes and reduce both together.
        let r0 = vzip1q_u64(p0_ll, p1_ll);
        let r1 = veorq_u64(vzip2q_u64(p0_ll, p1_ll), vzip1q_u64(c0, c1));
        let r2 = veorq_u64(vzip1q_u64(p0_hh, p1_hh), vzip2q_u64(c0, c1));
        let r3 = vzip2q_u64(p0_hh, p1_hh);

        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));
        let t_lo = veorq_u64(veorq_u64(r2, s1_lo), veorq_u64(s2_lo, s7_lo));
        let t_hi = veorq_u64(veorq_u64(r3, s1_hi), veorq_u64(s2_hi, s7_hi));
        let overflow = veorq_u64(
            veorq_u64(vshrq_n_u64::<63>(r3), vshrq_n_u64::<62>(r3)),
            vshrq_n_u64::<57>(r3),
        );
        let correction = veorq_u64(
            veorq_u64(overflow, vshlq_n_u64::<1>(overflow)),
            veorq_u64(vshlq_n_u64::<2>(overflow), vshlq_n_u64::<7>(overflow)),
        );
        let out_lo = veorq_u64(veorq_u64(r0, t_lo), correction);
        let out_hi = veorq_u64(r1, t_hi);
        [vzip1q_u64(out_lo, out_hi), vzip2q_u64(out_lo, out_hi)]
    }
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_unreduced(a: uint64x2_t, b: uint64x2_t) -> WideNeon {
    unsafe {
        let ll = pmull(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let hh = pmull(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
        let a_mid = veorq_u64(a, vextq_u64::<1>(a, a));
        let b_mid = veorq_u64(b, vextq_u64::<1>(b, b));
        let middle = pmull(vgetq_lane_u64::<0>(a_mid), vgetq_lane_u64::<0>(b_mid));
        let cross = veorq_u64(veorq_u64(middle, ll), hh);
        let zero = vdupq_n_u64(0);
        WideNeon {
            lo: veorq_u64(ll, vextq_u64::<1>(zero, cross)),
            hi: veorq_u64(hh, vextq_u64::<1>(cross, zero)),
        }
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
unsafe fn reduce_wide(value: WideNeon) -> uint64x2_t {
    unsafe {
        let zero = vdupq_n_u64(0);
        let high = value.hi;
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
        let folded = veorq_u64(veorq_u64(high, shift1), veorq_u64(shift2, shift7));
        let high_word = vextq_u64::<1>(high, zero);
        let overflow = veorq_u64(
            veorq_u64(vshrq_n_u64::<63>(high_word), vshrq_n_u64::<62>(high_word)),
            vshrq_n_u64::<57>(high_word),
        );
        let correction = veorq_u64(
            veorq_u64(overflow, vshlq_n_u64::<1>(overflow)),
            veorq_u64(vshlq_n_u64::<2>(overflow), vshlq_n_u64::<7>(overflow)),
        );
        veorq_u64(value.lo, veorq_u64(folded, correction))
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
    unsafe {
        let zero = vdupq_n_u64(0);
        let r_q = transmute::<F128, uint64x2_t>(r);
        let mut u0 = WideNeon { lo: zero, hi: zero };
        let mut u2 = WideNeon { lo: zero, hi: zero };
        let mut t = 0;
        while t < nf.len() {
            let source = 2 * (base + t);
            let f_even0 = vld1q_u64(f.as_ptr().add(source).cast::<u64>());
            let f_odd0 = vld1q_u64(f.as_ptr().add(source + 1).cast::<u64>());
            let f_even1 = vld1q_u64(f.as_ptr().add(source + 2).cast::<u64>());
            let f_odd1 = vld1q_u64(f.as_ptr().add(source + 3).cast::<u64>());
            let b_even0 = vld1q_u64(b.as_ptr().add(source).cast::<u64>());
            let b_odd0 = vld1q_u64(b.as_ptr().add(source + 1).cast::<u64>());
            let b_even1 = vld1q_u64(b.as_ptr().add(source + 2).cast::<u64>());
            let b_odd1 = vld1q_u64(b.as_ptr().add(source + 3).cast::<u64>());

            let folded_f =
                mul_const_vec2(r_q, veorq_u64(f_even0, f_odd0), veorq_u64(f_even1, f_odd1));
            let f0 = veorq_u64(f_even0, folded_f[0]);
            let f1 = veorq_u64(f_even1, folded_f[1]);
            let folded_b =
                mul_const_vec2(r_q, veorq_u64(b_even0, b_odd0), veorq_u64(b_even1, b_odd1));
            let b0 = veorq_u64(b_even0, folded_b[0]);
            let b1 = veorq_u64(b_even1, folded_b[1]);

            vst1q_u64(nf.as_mut_ptr().add(t).cast::<u64>(), f0);
            vst1q_u64(nf.as_mut_ptr().add(t + 1).cast::<u64>(), f1);
            vst1q_u64(nb.as_mut_ptr().add(t).cast::<u64>(), b0);
            vst1q_u64(nb.as_mut_ptr().add(t + 1).cast::<u64>(), b1);

            xor_wide(&mut u0, mul_unreduced(f0, b0));
            xor_wide(&mut u2, mul_unreduced(veorq_u64(f0, f1), veorq_u64(b0, b1)));
            t += 2;
        }
        (
            transmute::<uint64x2_t, F128>(reduce_wide(u0)),
            transmute::<uint64x2_t, F128>(reduce_wide(u2)),
        )
    }
}

use crate::field::F128;
use core::arch::aarch64::*;
use core::mem::transmute;

#[derive(Clone, Copy)]
struct WideNeon {
    lo: uint64x2_t,
    hi: uint64x2_t,
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

        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));
        let t_lo = xor3_u64(r2, s1_lo, veorq_u64(s2_lo, s7_lo));
        let t_hi = xor3_u64(r3, s1_hi, veorq_u64(s2_hi, s7_hi));
        let overflow = xor3_u64(
            vshrq_n_u64::<63>(r3),
            vshrq_n_u64::<62>(r3),
            vshrq_n_u64::<57>(r3),
        );
        let correction = xor3_u64(
            overflow,
            vshlq_n_u64::<1>(overflow),
            veorq_u64(vshlq_n_u64::<2>(overflow), vshlq_n_u64::<7>(overflow)),
        );
        let out_lo = xor3_u64(r0, t_lo, correction);
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
        let cross = xor3_u64(middle, ll, hh);
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
        let folded = xor3_u64(high, shift1, veorq_u64(shift2, shift7));
        let high_word = vextq_u64::<1>(high, zero);
        let overflow = xor3_u64(
            vshrq_n_u64::<63>(high_word),
            vshrq_n_u64::<62>(high_word),
            vshrq_n_u64::<57>(high_word),
        );
        let correction = xor3_u64(
            overflow,
            vshlq_n_u64::<1>(overflow),
            veorq_u64(vshlq_n_u64::<2>(overflow), vshlq_n_u64::<7>(overflow)),
        );
        xor3_u64(value.lo, folded, correction)
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

        let zero = vdupq_n_u64(0);
        let mut a_c0 = WideNeon { lo: zero, hi: zero };
        let mut a_c1_endpoint = WideNeon { lo: zero, hi: zero };
        let mut a_c2 = WideNeon { lo: zero, hi: zero };
        let mut a_u0_second = WideNeon { lo: zero, hi: zero };
        let mut a_u2_second = WideNeon { lo: zero, hi: zero };
        let mut a_c3 = WideNeon { lo: zero, hi: zero };
        let mut a_c4_endpoint = WideNeon { lo: zero, hi: zero };
        let mut a_c5 = WideNeon { lo: zero, hi: zero };

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
            xor_wide(&mut a_c0, mul_unreduced(a0, b0));
            xor_wide(&mut a_c1_endpoint, mul_unreduced(a1, b1));
            xor_wide(&mut a_c2, mul_unreduced(sa0, sb0));
            xor_wide(&mut a_u0_second, mul_unreduced(a2, b2));
            xor_wide(&mut a_u2_second, mul_unreduced(sa1, sb1));

            let even_a = veorq_u64(a0, a2);
            let even_b = veorq_u64(b0, b2);
            let odd_a = veorq_u64(a1, a3);
            let odd_b = veorq_u64(b1, b3);
            let sum_a = veorq_u64(even_a, odd_a);
            let sum_b = veorq_u64(even_b, odd_b);
            xor_wide(&mut a_c3, mul_unreduced(even_a, even_b));
            xor_wide(&mut a_c4_endpoint, mul_unreduced(odd_a, odd_b));
            xor_wide(&mut a_c5, mul_unreduced(sum_a, sum_b));
            i += 4;
        }

        let red = |value: WideNeon| transmute::<uint64x2_t, F128>(reduce_wide(value));
        let c0 = red(a_c0);
        let c2 = red(a_c2);
        let c3 = red(a_c3);
        let c5 = red(a_c5);
        let c1 = red(a_c1_endpoint) + c0 + c2;
        let c4 = red(a_c4_endpoint) + c3 + c5;
        let u0 = c0 + red(a_u0_second);
        let u2 = c2 + red(a_u2_second);
        ((u0, u2), [c0, c1, c2, c3, c4, c5])
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
/// Ticket-14 NEON: two-challenge fused fold. Per output group of 4, loads 16
/// source pairs per polynomial, binds `r_a` then `r_b` entirely in registers,
/// stores 4 outputs per polynomial, and accumulates the direct message plus
/// the 6 lookahead coefficients. Product reuse: the direct message's first
/// u_0/u_2 terms are identical to lookahead c0/c2, so each is computed once.
///
/// # Safety
/// Caller guarantees PMULL and: `f.len() == b.len()`, `wf.len() == wb.len()`,
/// `wf.len() % 4 == 0`, `base % 4 == 0`, and `4 * (base + wf.len()) <= f.len()`.
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
    // `nt_stores` is decided once per fold round by the driver (round output
    // past LLC size ⇒ the w arrays are not read until the next fold pair's
    // barrier ⇒ `stnp` elides write-allocate RFO reads). Per-chunk callers
    // must not decide this from their sub-slice length. (Resample draw 2 of
    // this re-land; mechanism unchanged — see submission note.)
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
        // Accumulators: a_u0a = c0 (also u_0 term 1), a_u0b = u_0 term 2,
        // a_u2a = c2 (also u_2 term 1), a_u2b = u_2 term 2. a_c1/a_c4
        // hold the complementary endpoint products for Karatsuba recovery;
        // a_c3/a_c5 hold the other lookahead endpoint products.
        let mut a_u0a = WideNeon { lo: zero, hi: zero };
        let mut a_u0b = WideNeon { lo: zero, hi: zero };
        let mut a_u2a = WideNeon { lo: zero, hi: zero };
        let mut a_u2b = WideNeon { lo: zero, hi: zero };
        let mut a_c1 = WideNeon { lo: zero, hi: zero };
        let mut a_c3 = WideNeon { lo: zero, hi: zero };
        let mut a_c4 = WideNeon { lo: zero, hi: zero };
        let mut a_c5 = WideNeon { lo: zero, hi: zero };

        let mut w_regs_f = [zero; 4];
        let mut w_regs_b = [zero; 4];
        let mut t = 0;
        while t < wf.len() {
            for q in 0..4 {
                let src = 4 * (base + t + q);
                let fe0 = vld1q_u64(f.as_ptr().add(src).cast::<u64>());
                let fo0 = vld1q_u64(f.as_ptr().add(src + 1).cast::<u64>());
                let fe1 = vld1q_u64(f.as_ptr().add(src + 2).cast::<u64>());
                let fo1 = vld1q_u64(f.as_ptr().add(src + 3).cast::<u64>());
                let be0 = vld1q_u64(b.as_ptr().add(src).cast::<u64>());
                let bo0 = vld1q_u64(b.as_ptr().add(src + 1).cast::<u64>());
                let be1 = vld1q_u64(b.as_ptr().add(src + 2).cast::<u64>());
                let bo1 = vld1q_u64(b.as_ptr().add(src + 3).cast::<u64>());

                // First bind at r_a: v = even ^ r_a*(even^odd), two v per poly.
                let pf = mul_const_vec2(ra_q, veorq_u64(fe0, fo0), veorq_u64(fe1, fo1));
                let vf0 = veorq_u64(fe0, pf[0]);
                let vf1 = veorq_u64(fe1, pf[1]);
                let pb = mul_const_vec2(ra_q, veorq_u64(be0, bo0), veorq_u64(be1, bo1));
                let vb0 = veorq_u64(be0, pb[0]);
                let vb1 = veorq_u64(be1, pb[1]);

                // Second bind at r_b, both polynomials in one paired multiply.
                let pw = mul_const_vec2(rb_q, veorq_u64(vf0, vf1), veorq_u64(vb0, vb1));
                let wq_f = veorq_u64(vf0, pw[0]);
                let wq_b = veorq_u64(vb0, pw[1]);
                if !nt_stores {
                    vst1q_u64(wf.as_mut_ptr().add(t + q).cast::<u64>(), wq_f);
                    vst1q_u64(wb.as_mut_ptr().add(t + q).cast::<u64>(), wq_b);
                }
                w_regs_f[q] = wq_f;
                w_regs_b[q] = wq_b;
            }
            if nt_stores {
                // The four group outputs are adjacent: two 32-byte pair
                // stores per polynomial. Same values as the per-q stores —
                // only the cacheability hint differs.
                store_pair_nt(wf.as_mut_ptr().add(t), w_regs_f[0], w_regs_f[1]);
                store_pair_nt(wf.as_mut_ptr().add(t + 2), w_regs_f[2], w_regs_f[3]);
                store_pair_nt(wb.as_mut_ptr().add(t), w_regs_b[0], w_regs_b[1]);
                store_pair_nt(wb.as_mut_ptr().add(t + 2), w_regs_b[2], w_regs_b[3]);
            }
            // Direct message over pairs (w0,w1), (w2,w3); lookahead over the
            // quad. Shared products accumulated once.
            let s0f = veorq_u64(w_regs_f[0], w_regs_f[1]);
            let s0b = veorq_u64(w_regs_b[0], w_regs_b[1]);
            let s1f = veorq_u64(w_regs_f[2], w_regs_f[3]);
            let s1b = veorq_u64(w_regs_b[2], w_regs_b[3]);
            xor_wide(&mut a_u0a, mul_unreduced(w_regs_f[0], w_regs_b[0])); // = c0 + u0 term
            xor_wide(&mut a_u0b, mul_unreduced(w_regs_f[2], w_regs_b[2]));
            xor_wide(&mut a_u2a, mul_unreduced(s0f, s0b)); // = c2 + u2 term
            xor_wide(&mut a_u2b, mul_unreduced(s1f, s1b));
            // c1's cross term is recovered after reduction as
            // w1f*w1b ^ c0 ^ c2: one product instead of the direct two.
            xor_wide(&mut a_c1, mul_unreduced(w_regs_f[1], w_regs_b[1]));
            // e = w0 + w2, o = w1 + w3, se = e + o
            let e_f = veorq_u64(w_regs_f[0], w_regs_f[2]);
            let o_f = veorq_u64(w_regs_f[1], w_regs_f[3]);
            let e_b = veorq_u64(w_regs_b[0], w_regs_b[2]);
            let o_b = veorq_u64(w_regs_b[1], w_regs_b[3]);
            let se_f = veorq_u64(e_f, o_f);
            let se_b = veorq_u64(e_b, o_b);
            xor_wide(&mut a_c3, mul_unreduced(e_f, e_b));
            // c4 uses the same Karatsuba identity with the odd aggregate as
            // the complementary endpoint.
            xor_wide(&mut a_c4, mul_unreduced(o_f, o_b));
            xor_wide(&mut a_c5, mul_unreduced(se_f, se_b));
            t += 4;
        }
        let red = |w: WideNeon| transmute::<uint64x2_t, F128>(reduce_wide(w));
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
/// products and wide accumulators is dead work.
///
/// # Safety
/// Caller guarantees PMULL and: `f.len() == b.len()`, `wf.len() == wb.len()`,
/// `wf.len() % 4 == 0`, `base % 4 == 0`, and `4 * (base + wf.len()) <= f.len()`.
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
        let mut a_u0a = WideNeon { lo: zero, hi: zero };
        let mut a_u0b = WideNeon { lo: zero, hi: zero };
        let mut a_u2a = WideNeon { lo: zero, hi: zero };
        let mut a_u2b = WideNeon { lo: zero, hi: zero };

        let mut w_regs_f = [zero; 4];
        let mut w_regs_b = [zero; 4];
        let mut t = 0;
        while t < wf.len() {
            for q in 0..4 {
                let src = 4 * (base + t + q);
                let fe0 = vld1q_u64(f.as_ptr().add(src).cast::<u64>());
                let fo0 = vld1q_u64(f.as_ptr().add(src + 1).cast::<u64>());
                let fe1 = vld1q_u64(f.as_ptr().add(src + 2).cast::<u64>());
                let fo1 = vld1q_u64(f.as_ptr().add(src + 3).cast::<u64>());
                let be0 = vld1q_u64(b.as_ptr().add(src).cast::<u64>());
                let bo0 = vld1q_u64(b.as_ptr().add(src + 1).cast::<u64>());
                let be1 = vld1q_u64(b.as_ptr().add(src + 2).cast::<u64>());
                let bo1 = vld1q_u64(b.as_ptr().add(src + 3).cast::<u64>());

                let pf = mul_const_vec2(ra_q, veorq_u64(fe0, fo0), veorq_u64(fe1, fo1));
                let vf0 = veorq_u64(fe0, pf[0]);
                let vf1 = veorq_u64(fe1, pf[1]);
                let pb = mul_const_vec2(ra_q, veorq_u64(be0, bo0), veorq_u64(be1, bo1));
                let vb0 = veorq_u64(be0, pb[0]);
                let vb1 = veorq_u64(be1, pb[1]);

                let pw = mul_const_vec2(rb_q, veorq_u64(vf0, vf1), veorq_u64(vb0, vb1));
                let wq_f = veorq_u64(vf0, pw[0]);
                let wq_b = veorq_u64(vb0, pw[1]);
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
            xor_wide(&mut a_u0a, mul_unreduced(w_regs_f[0], w_regs_b[0]));
            xor_wide(&mut a_u0b, mul_unreduced(w_regs_f[2], w_regs_b[2]));
            xor_wide(&mut a_u2a, mul_unreduced(s0f, s0b));
            xor_wide(&mut a_u2b, mul_unreduced(s1f, s1b));
            t += 4;
        }

        let red = |w: WideNeon| transmute::<uint64x2_t, F128>(reduce_wide(w));
        (red(a_u0a) + red(a_u0b), red(a_u2a) + red(a_u2b))
    }
}

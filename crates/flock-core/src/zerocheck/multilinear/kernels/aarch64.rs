use crate::field::{F128, F256Unreduced};

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct WideNeon {
    lo: core::arch::aarch64::uint64x2_t,
    hi: core::arch::aarch64::uint64x2_t,
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn pmull_lane(a: u64, b: u64) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe { core::mem::transmute::<u128, uint64x2_t>(vmull_p64(a, b)) }
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
        let t0 = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let t1a = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<1>(b));
        let t1b = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<0>(b));
        let t2 = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
        let mut t1 = veorq_u64(t1a, t1b);

        t1 = veorq_u64(t1, vextq_u64::<1>(zero, t2));
        t1 = veorq_u64(t1, pmull_lane(vgetq_lane_u64::<1>(t2), 0x87));

        let mut out = veorq_u64(t0, vextq_u64::<1>(zero, t1));
        out = veorq_u64(out, pmull_lane(vgetq_lane_u64::<1>(t1), 0x87));
        out
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
        let ll = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let lh = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<1>(b));
        let hl = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<0>(b));
        let hh = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
        let cross = veorq_u64(lh, hl);
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
        let hi = value.hi;
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
        let folded = veorq_u64(veorq_u64(hi, shift1), veorq_u64(shift2, shift7));

        // Only r3 (the high lane) can overflow the 128-bit fold. Move it to
        // the low lane so the correction lands in result coefficient 0.
        let r3 = vextq_u64::<1>(hi, zero);
        let ov = veorq_u64(
            veorq_u64(vshrq_n_u64::<63>(r3), vshrq_n_u64::<62>(r3)),
            vshrq_n_u64::<57>(r3),
        );
        let corr = veorq_u64(
            veorq_u64(ov, vshlq_n_u64::<1>(ov)),
            veorq_u64(vshlq_n_u64::<2>(ov), vshlq_n_u64::<7>(ov)),
        );
        veorq_u64(value.lo, veorq_u64(folded, corr))
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
        let mut acc = vreinterpretq_u64_u8(vld1q_u8(table_data.add((*bytes_ptr) as usize * 16)));
        for chunk in 1..8 {
            let entry = table_data.add(
                chunk * STRIDE + (*bytes_ptr.add(chunk)) as usize * core::mem::size_of::<F128>(),
            );
            acc = veorq_u64(acc, vreinterpretq_u64_u8(vld1q_u8(entry)));
        }
        acc
    }
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
    unsafe {
        let zero = vdupq_n_u64(0);
        let mut p1_acc = WideNeon { lo: zero, hi: zero };
        let mut pinf_acc = WideNeon { lo: zero, hi: zero };

        for x_lo in 0..lo_size {
            let out = 2 * x_lo;
            if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                vst1q_u64(a_out.add(out).cast::<u64>(), zero);
                vst1q_u64(a_out.add(out + 1).cast::<u64>(), zero);
                vst1q_u64(b_out.add(out).cast::<u64>(), zero);
                vst1q_u64(b_out.add(out + 1).cast::<u64>(), zero);
                continue;
            }

            let row0 = 2 * x_lo;
            let row1 = row0 + 1;
            let a0 = fold_row_q(table_data, a_packed.add(row0 * 8));
            let b0 = fold_row_q(table_data, b_packed.add(row0 * 8));
            let a1 = fold_row_q(table_data, a_packed.add(row1 * 8));
            let b1 = fold_row_q(table_data, b_packed.add(row1 * 8));

            vst1q_u64(a_out.add(out).cast::<u64>(), a0);
            vst1q_u64(a_out.add(out + 1).cast::<u64>(), a1);
            vst1q_u64(b_out.add(out).cast::<u64>(), b0);
            vst1q_u64(b_out.add(out + 1).cast::<u64>(), b1);

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

/// Fuse one multilinear tail fold with construction of the following round's
/// message. The previous AArch64 path first streamed all of `a_in`/`b_in` into
/// `a_out`/`b_out`, then immediately reread both outputs in a second pass.
/// Keeping each four-value folded pair live until its message contribution is
/// accumulated removes that full output readback while preserving the exact
/// canonical output tables for the next round.
///
/// # Measured dead ends — do not retry these
///
/// This loop looks like an obvious target for PMULL reduction and for NEON
/// rewrites. Three variants were built and timed against this scalar loop on
/// identical inputs; **all three lost** at the geometry that actually matters.
///
/// | variant | PMULL/elem | result |
/// |---|---|---|
/// | batch products via `ghash_mul_vec2_neon` | 32 | ~21–30% slower |
/// | `mul_unreduced_q` + `reduce_wide_q` everywhere | 32 | ~57% slower |
/// | ditto, only off the store critical path | 40 | ~29% slower |
/// | fully q-register-resident, `mul_q` (same math) | 44 | see below |
///
/// `ghash_mul_vec2_neon` takes and returns `[F128; 2]` *by value* through
/// general-purpose registers, so wrapping q-resident code in it adds more lane
/// extract/reinsert traffic than the saved PMULLs buy. Composing
/// `mul_unreduced_q` + `reduce_wide_q` trades 2 PMULL for a ~10-op shift/XOR
/// network, which is a bad deal here. **This loop is not PMULL-bound, so
/// counting PMULL does not predict its speed.**
///
/// The q-register-resident variant is the instructive one. It kept the same 44
/// PMULL and the same values, and deleted only the GPR↔vector crossings
/// (20 `fmov` + 10 `mov.d` → 2 + 0; 203 → 120 hot-loop instructions). It won
/// by 6–7% on small cache-resident inputs — and **lost** on the real ones:
///
/// | round size | 2^20 | 2^21 | 2^22 | 2^23 | 2^24 | 2^25 | 2^26 |
/// |---|---|---|---|---|---|---|---|
/// | q-resident vs this | +4.7% | +8.8% | +2.5% | +1.9% | −1.3% | −2.0% | −2.3% |
///
/// The ranked tail starts at `log_n = 26` and halves; rounds ≥ 2^24 carry
/// ~87% of all tail work, so over the full 17-round tail the q-resident kernel
/// measured **−0.2% to −1.9%** (33.6 ms → 34.2 ms). Once the working set
/// leaves cache the loop is memory-bound and saved instructions buy nothing,
/// while the wider vector live-range costs a little. Any future attempt here
/// must be timed on the **full descending tail from 2^26**, not on one small
/// cache-resident size, or it will report a win that does not exist.
#[cfg(target_arch = "aarch64")]
pub(crate) fn fold_and_message_aarch64(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    debug_assert_eq!(a_in.len(), 2 * a_out.len());
    debug_assert_eq!(b_in.len(), 2 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());

    let mut p1_acc = F256Unreduced::ZERO;
    let mut pinf_acc = F256Unreduced::ZERO;

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

        let a0 = a_even_0 + r_fold * (a_even_0 + a_odd_0);
        let a1 = a_even_1 + r_fold * (a_even_1 + a_odd_1);
        let b0 = b_even_0 + r_fold * (b_even_0 + b_odd_0);
        let b1 = b_even_1 + r_fold * (b_even_1 + b_odd_1);

        a_out[o] = a0;
        a_out[o + 1] = a1;
        b_out[o] = b0;
        b_out[o + 1] = b1;

        p1_acc ^= eq_l.mul_unreduced(a1 * b1);
        pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
    }

    (p1_acc.reduce(), pinf_acc.reduce())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 PRNG, deterministic — matches the repo's test convention.
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
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    /// Verbatim scalar formula the batched kernel must reproduce bit-for-bit.
    fn reference(
        a_in: &[F128],
        b_in: &[F128],
        a_out: &mut [F128],
        b_out: &mut [F128],
        r_fold: F128,
        eq_lo: &[F128],
    ) -> (F128, F128) {
        let mut p1_acc = F256Unreduced::ZERO;
        let mut pinf_acc = F256Unreduced::ZERO;
        for (x_lo, &eq_l) in eq_lo.iter().enumerate() {
            let i = 4 * x_lo;
            let o = 2 * x_lo;
            let a0 = a_in[i] + r_fold * (a_in[i] + a_in[i + 1]);
            let a1 = a_in[i + 2] + r_fold * (a_in[i + 2] + a_in[i + 3]);
            let b0 = b_in[i] + r_fold * (b_in[i] + b_in[i + 1]);
            let b1 = b_in[i + 2] + r_fold * (b_in[i + 2] + b_in[i + 3]);
            a_out[o] = a0;
            a_out[o + 1] = a1;
            b_out[o] = b0;
            b_out[o + 1] = b1;
            p1_acc ^= eq_l.mul_unreduced(a1 * b1);
            pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
        }
        (p1_acc.reduce(), pinf_acc.reduce())
    }

    #[test]
    fn fold_and_message_matches_scalar_reference() {
        // Even and odd `lo_size` both matter: odd sizes exercise the scalar
        // trailing iteration after the 2-wide batched pass.
        for &lo_size in &[1usize, 2, 3, 4, 5, 7, 8, 9, 16, 17] {
            let mut rng = Rng::new(0x5EED_0000 ^ lo_size as u64);
            let a_in = rng.f128_vec(4 * lo_size);
            let b_in = rng.f128_vec(4 * lo_size);
            let eq_lo = rng.f128_vec(lo_size);
            let r_fold = rng.f128();

            let mut a_got = vec![F128::ZERO; 2 * lo_size];
            let mut b_got = vec![F128::ZERO; 2 * lo_size];
            let got =
                fold_and_message_aarch64(&a_in, &b_in, &mut a_got, &mut b_got, r_fold, &eq_lo);

            let mut a_want = vec![F128::ZERO; 2 * lo_size];
            let mut b_want = vec![F128::ZERO; 2 * lo_size];
            let want = reference(&a_in, &b_in, &mut a_want, &mut b_want, r_fold, &eq_lo);

            assert_eq!(a_got, a_want, "a_out mismatch at lo_size={lo_size}");
            assert_eq!(b_got, b_want, "b_out mismatch at lo_size={lo_size}");
            assert_eq!(got.0, want.0, "p1 mismatch at lo_size={lo_size}");
            assert_eq!(got.1, want.1, "p_inf mismatch at lo_size={lo_size}");
        }
    }

    /// Edge inputs (zero rows, r_fold = 0/1) must also agree exactly.
    #[test]
    fn fold_and_message_matches_scalar_reference_edge_scalars() {
        let lo_size = 6usize;
        let mut rng = Rng::new(0xC0FFEE);
        let a_in = rng.f128_vec(4 * lo_size);
        let b_in = rng.f128_vec(4 * lo_size);
        let eq_lo = rng.f128_vec(lo_size);

        for r_fold in [F128::ZERO, F128::ONE, F128::generator()] {
            let mut a_got = vec![F128::ZERO; 2 * lo_size];
            let mut b_got = vec![F128::ZERO; 2 * lo_size];
            let got =
                fold_and_message_aarch64(&a_in, &b_in, &mut a_got, &mut b_got, r_fold, &eq_lo);

            let mut a_want = vec![F128::ZERO; 2 * lo_size];
            let mut b_want = vec![F128::ZERO; 2 * lo_size];
            let want = reference(&a_in, &b_in, &mut a_want, &mut b_want, r_fold, &eq_lo);

            assert_eq!((a_got, b_got, got), (a_want, b_want, want));
        }
    }
}

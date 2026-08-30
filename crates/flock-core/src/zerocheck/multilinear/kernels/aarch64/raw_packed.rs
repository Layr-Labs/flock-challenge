//! Reconstruct the first two multilinear folds from their original bit rows.
//! Only the retained representation changes; the round-four/five accumulators
//! below use the same odd-lane weight and products as the compact kernel.

use super::{F128, WideNeon, is_zero_q, lookup_lanes_q, mul_q, mul_unreduced_q};
use super::{reduce_wide_q, wide_xor, xor3_u64};
use core::arch::aarch64::*;

/// Read four raw rows, applying the producer's pair-level padding before
/// looking at memory. A boundary group may have one live and one dead pair.
#[inline(always)]
unsafe fn load_padded_group(
    packed: *const u8,
    group: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [u64; 4] {
    let mut rows = [0u64; 4];
    for pair in 0..2 {
        if ((pair_idx_base + 2 * group + pair) & pair_in_block_mask)
            < useful_pairs_inclusive
        {
            // SAFETY: the caller supplies four rows per local group. Dead
            // pairs are never read, even if their input bytes are nonzero.
            unsafe {
                let src = packed.add(group * 32 + pair * 16);
                rows[2 * pair] = u64::from_le(core::ptr::read_unaligned(src.cast::<u64>()));
                rows[2 * pair + 1] =
                    u64::from_le(core::ptr::read_unaligned(src.add(8).cast::<u64>()));
            }
        }
    }
    rows
}

#[inline(always)]
unsafe fn fold_weighted_rows(tables: &[*const u8; 4], rows: &[u64; 4]) -> uint64x2_t {
    unsafe {
        let lo = veorq_u64(
            lookup_lanes_q::<8>(tables[0], rows[0], 0),
            lookup_lanes_q::<8>(tables[1], rows[1], 0),
        );
        xor3_u64(
            lo,
            lookup_lanes_q::<8>(tables[2], rows[2], 0),
            lookup_lanes_q::<8>(tables[3], rows[3], 0),
        )
    }
}

/// `tables[j] = lambda[j] * T_z`, with the four ordinary multilinear
/// equality weights for `[rho1, rho2]`. Each output is the weighted fold
/// of four original rows, so no anchor or delta buffer is required.
///
/// SAFETY: packed inputs cover `8 * out_pairs` rows of eight bytes each;
/// outputs cover `2 * out_pairs` F128 values; eq_lo covers `out_pairs`
/// values and out covers eight. Every table has 8 * 256 F128 entries.
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold2_packed_and_round45_chunk_neon_8(
    tables: [*const u8; 4],
    folded_ones: F128,
    a_packed: *const u8,
    b_packed: *const u8,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    out_pairs: usize,
    degen: bool,
    out: *mut F128,
) {
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
        let one_q = core::mem::transmute::<F128, uint64x2_t>(F128::ONE);
        let folded_ones_q = core::mem::transmute::<F128, uint64x2_t>(folded_ones);
        let mut p1_even = WideNeon { lo: zero, hi: zero };
        let mut pinf_even = WideNeon { lo: zero, hi: zero };
        let mut p1_odd = WideNeon { lo: zero, hi: zero };
        let mut pinf_odd = WideNeon { lo: zero, hi: zero };
        let mut w0 = WideNeon { lo: zero, hi: zero };
        let mut w3 = WideNeon { lo: zero, hi: zero };
        let mut w4 = WideNeon { lo: zero, hi: zero };
        let mut w5 = WideNeon { lo: zero, hi: zero };

        debug_assert!(out_pairs >= 2 && out_pairs.is_multiple_of(2));
        for t in 0..out_pairs / 2 {
            let mut av = [zero; 4];
            let mut bv = [zero; 4];
            let mut b_flat = true;
            for lane in 0..4 {
                let group = 4 * t + lane;
                let ar = load_padded_group(
                    a_packed,
                    group,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                );
                let br = load_padded_group(
                    b_packed,
                    group,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                );
                av[lane] = if (ar[0] | ar[1] | ar[2] | ar[3]) == 0 {
                    zero
                } else {
                    fold_weighted_rows(&tables, &ar)
                };
                // Short-circuit BEFORE any B lookup. All-ones/zero codes
                // have sum(lambda)=1; padding is already masked to zero,
                // so a mixed live/dead group cannot take the ones arm.
                if degen && (br[0] & br[1] & br[2] & br[3]) == u64::MAX {
                    bv[lane] = folded_ones_q;
                } else if degen && (br[0] | br[1] | br[2] | br[3]) == 0 {
                    bv[lane] = zero;
                } else {
                    b_flat = false;
                    bv[lane] = fold_weighted_rows(&tables, &br);
                }
            }

            store_pair_nt(a_out.add(4 * t), av[0], av[1]);
            store_pair_nt(a_out.add(4 * t + 2), av[2], av[3]);
            store_pair_nt(b_out.add(4 * t), bv[0], bv[1]);
            store_pair_nt(b_out.add(4 * t + 2), bv[2], bv[3]);

            let w = vld1q_u64(eq_lo.add(2 * t + 1).cast::<u64>());
            if b_flat {
                let ones_miss = vorrq_u64(
                    vorrq_u64(veorq_u64(bv[0], one_q), veorq_u64(bv[1], one_q)),
                    vorrq_u64(veorq_u64(bv[2], one_q), veorq_u64(bv[3], one_q)),
                );
                if is_zero_q(ones_miss) {
                    wide_xor(&mut p1_even, mul_unreduced_q(w, av[1]));
                    wide_xor(&mut p1_odd, mul_unreduced_q(w, av[3]));
                    wide_xor(&mut w0, mul_unreduced_q(w, av[2]));
                    continue;
                }
            }

            let a0w = mul_q(w, av[0]);
            let a1w = mul_q(w, av[1]);
            let a2w = mul_q(w, av[2]);
            let a3w = mul_q(w, av[3]);
            wide_xor(&mut p1_even, mul_unreduced_q(a1w, bv[1]));
            wide_xor(
                &mut pinf_even,
                mul_unreduced_q(veorq_u64(a0w, a1w), veorq_u64(bv[0], bv[1])),
            );
            wide_xor(&mut p1_odd, mul_unreduced_q(a3w, bv[3]));
            wide_xor(
                &mut pinf_odd,
                mul_unreduced_q(veorq_u64(a2w, a3w), veorq_u64(bv[2], bv[3])),
            );
            wide_xor(&mut w0, mul_unreduced_q(a2w, bv[2]));
            let e_aw = veorq_u64(a0w, a2w);
            let o_aw = veorq_u64(a1w, a3w);
            let e_b = veorq_u64(bv[0], bv[2]);
            let o_b = veorq_u64(bv[1], bv[3]);
            wide_xor(&mut w3, mul_unreduced_q(e_aw, e_b));
            wide_xor(&mut w4, mul_unreduced_q(o_aw, o_b));
            wide_xor(
                &mut w5,
                mul_unreduced_q(veorq_u64(e_aw, o_aw), veorq_u64(e_b, o_b)),
            );
        }

        *out.add(0) = core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1_even));
        *out.add(1) = core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf_even));
        *out.add(2) = core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1_odd));
        *out.add(3) = core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf_odd));
        *out.add(4) = core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(w0));
        *out.add(5) = core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(w3));
        *out.add(6) = core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(w4));
        *out.add(7) = core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(w5));
    }
}

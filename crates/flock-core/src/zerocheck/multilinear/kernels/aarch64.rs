use crate::field::{F128, F256Unreduced};

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct WideNeon {
    lo: core::arch::aarch64::uint64x2_t,
    hi: core::arch::aarch64::uint64x2_t,
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

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn karatsuba_products_q(
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
        let a_sum = veorq_u64(a, vextq_u64::<1>(a, a));
        let b_sum = veorq_u64(b, vextq_u64::<1>(b, b));
        let pm = pmull_lane(vgetq_lane_u64::<0>(a_sum), vgetq_lane_u64::<0>(b_sum));
        let cross = xor3_u64(pm, p0, p2);
        (p0, cross, p2)
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
        let (t0, mut t1, t2) = karatsuba_products_q(a, b);

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
        let (ll, cross, hh) = karatsuba_products_q(a, b);
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
        let folded = xor3_u64(hi, shift1, veorq_u64(shift2, shift7));

        // Only r3 (the high lane) can overflow the 128-bit fold. Move it to
        // the low lane so the correction lands in result coefficient 0.
        let r3 = vextq_u64::<1>(hi, zero);
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
        xor3_u64(value.lo, folded, corr)
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

/// Fold four already-loaded packed rows together. Grouping the independent
/// lookup chains by byte bank exposes four L1 loads at a time, and loading each
/// row as one `u64` lets the compact round-two path reuse it for its raw delta.
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
        let load = |row: u64, chunk: usize| {
            let shift = 8 * chunk;
            let offset = chunk * STRIDE;
            let index = ((row >> shift) & 0xff) as usize;
            vld1q_u64(
                table_data
                    .add(offset + index * core::mem::size_of::<F128>())
                    .cast::<u64>(),
            )
        };

        // Retain the four independent row streams while consuming two new
        // table rows per dependent accumulator update.
        let mut acc0 = load(row0, 0);
        let mut acc1 = load(row1, 0);
        let mut acc2 = load(row2, 0);
        let mut acc3 = load(row3, 0);
        for chunk in (1..7).step_by(2) {
            acc0 = xor3_u64(acc0, load(row0, chunk), load(row0, chunk + 1));
            acc1 = xor3_u64(acc1, load(row1, chunk), load(row1, chunk + 1));
            acc2 = xor3_u64(acc2, load(row2, chunk), load(row2, chunk + 1));
            acc3 = xor3_u64(acc3, load(row3, chunk), load(row3, chunk + 1));
        }
        acc0 = veorq_u64(acc0, load(row0, 7));
        acc1 = veorq_u64(acc1, load(row1, 7));
        acc2 = veorq_u64(acc2, load(row2, 7));
        acc3 = veorq_u64(acc3, load(row3, 7));
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
        let load = |row: u64, chunk: usize| {
            let shift = 8 * chunk;
            let offset = chunk * STRIDE;
            let index = ((row >> shift) & 0xff) as usize;
            vld1q_u64(
                table_data
                    .add(offset + index * core::mem::size_of::<F128>())
                    .cast::<u64>(),
            )
        };
        let mut acc0 = load(row0, 0);
        let mut acc1 = load(row1, 0);
        for chunk in (1..7).step_by(2) {
            acc0 = xor3_u64(acc0, load(row0, chunk), load(row0, chunk + 1));
            acc1 = xor3_u64(acc1, load(row1, chunk), load(row1, chunk + 1));
        }
        acc0 = veorq_u64(acc0, load(row0, 7));
        acc1 = veorq_u64(acc1, load(row1, 7));
        (acc0, acc1)
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
    unsafe fn store_anchor_pair_nt(
        dst: *mut F128,
        a: uint64x2_t,
        b: uint64x2_t,
    ) {
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
            let delta_pair =
                core::mem::transmute::<[u64; 2], uint64x2_t>([a0_code ^ a1_code, b0_code ^ b1_code]);
            vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);
        }
    }
}

/// In-place-delta counterpart of
/// [`fold_round2_compact_chunk_neon_unchecked_8`].  Each packed pair is laid
/// out as `[row0, row0 XOR row1]`; the odd row is reconstructed in registers
/// for the round message, and only anchors are written because the deltas
/// already reside in A/B.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_round2_in_place_deltas_chunk_neon_unchecked_8(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
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
        let ones_bytes = [0xFFu8; 8];
        let b_ones = fold_row_q(table_data, ones_bytes.as_ptr());
        let ones_is_one = is_zero_q(veorq_u64(
            b_ones,
            core::mem::transmute::<F128, uint64x2_t>(F128::ONE),
        ));

        for x_lo in 0..lo_size {
            if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                continue;
            }
            let pair_off = x_lo * 16;
            let a0_code = u64::from_le(core::ptr::read_unaligned(
                a_packed.add(pair_off).cast::<u64>(),
            ));
            let da = u64::from_le(core::ptr::read_unaligned(
                a_packed.add(pair_off + 8).cast::<u64>(),
            ));
            let b0_code = u64::from_le(core::ptr::read_unaligned(
                b_packed.add(pair_off).cast::<u64>(),
            ));
            let db = u64::from_le(core::ptr::read_unaligned(
                b_packed.add(pair_off + 8).cast::<u64>(),
            ));
            let a1_code = a0_code ^ da;
            let b1_code = b0_code ^ db;

            if degen && (b0_code & b1_code) == u64::MAX {
                let (a0, a1) = fold_two_row_codes_q(table_data, a0_code, a1_code);
                store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b_ones);
                let g1 = if ones_is_one { a1 } else { mul_q(a1, b_ones) };
                let eq_l = vld1q_u64(eq_lo.add(x_lo).cast::<u64>());
                wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
                continue;
            }

            let (a0, a1, b0, b1) =
                fold_four_row_codes_q(table_data, a0_code, a1_code, b0_code, b1_code);
            store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b0);
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

/// Anchors-only CPU sibling for an in-place-delta GPU prefix.  Odd rows are
/// irrelevant to anchors, so this path reads just the retained even rows.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn fold_round2_in_place_deltas_chunk_neon_anchors_only_8(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
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
                continue;
            }
            let pair_off = x_lo * 16;
            let a0_code = u64::from_le(core::ptr::read_unaligned(
                a_packed.add(pair_off).cast::<u64>(),
            ));
            let b0_code = u64::from_le(core::ptr::read_unaligned(
                b_packed.add(pair_off).cast::<u64>(),
            ));
            let (a0, b0) = fold_two_row_codes_q(table_data, a0_code, b0_code);
            store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b0);
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
                let a1 = veorq_u64(vld1q_u64(anchors.add(2 * (out + 1)).cast::<u64>()), a1_delta);
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

            let (a0_delta, a1_delta, b0_delta, b1_delta) = fold_four_row_codes_q(
                scaled_table,
                a0_code,
                a1_code,
                b0_code,
                b1_code,
            );
            let a0 = veorq_u64(
                vld1q_u64(anchors.add(2 * out).cast::<u64>()),
                a0_delta,
            );
            let a1 = veorq_u64(
                vld1q_u64(anchors.add(2 * (out + 1)).cast::<u64>()),
                a1_delta,
            );
            let b0 = veorq_u64(
                vld1q_u64(anchors.add(2 * out + 1).cast::<u64>()),
                b0_delta,
            );
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

/// First-tail reconstruction from anchors plus strided deltas retained in the
/// odd rows of the original packed A/B buffers.  For compact index `i`, its
/// delta is the second eight-byte half of packed pair `i`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_in_place_deltas_chunk_neon_unchecked_8(
    scaled_table: *const u8,
    anchors: *const F128,
    a_pairs: *const u8,
    b_pairs: *const u8,
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
            let a0_code = u64::from_le(core::ptr::read_unaligned(
                a_pairs.add(out * 16 + 8).cast::<u64>(),
            ));
            let b0_code = u64::from_le(core::ptr::read_unaligned(
                b_pairs.add(out * 16 + 8).cast::<u64>(),
            ));
            let a1_code = u64::from_le(core::ptr::read_unaligned(
                a_pairs.add((out + 1) * 16 + 8).cast::<u64>(),
            ));
            let b1_code = u64::from_le(core::ptr::read_unaligned(
                b_pairs.add((out + 1) * 16 + 8).cast::<u64>(),
            ));

            if degen && (b0_code | b1_code) == 0 {
                let (a0_delta, a1_delta) =
                    fold_two_row_codes_q(scaled_table, a0_code, a1_code);
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
                    wide_xor(
                        &mut pinf_acc,
                        mul_unreduced_q(eq_l, mul_q(veorq_u64(a0, a1), b_sum)),
                    );
                }
                continue;
            }

            let (a0_delta, a1_delta, b0_delta, b1_delta) = fold_four_row_codes_q(
                scaled_table,
                a0_code,
                a1_code,
                b0_code,
                b1_code,
            );
            let a0 = veorq_u64(vld1q_u64(anchors.add(2 * out).cast::<u64>()), a0_delta);
            let a1 = veorq_u64(
                vld1q_u64(anchors.add(2 * (out + 1)).cast::<u64>()),
                a1_delta,
            );
            let b0 = veorq_u64(
                vld1q_u64(anchors.add(2 * out + 1).cast::<u64>()),
                b0_delta,
            );
            let b1 = veorq_u64(
                vld1q_u64(anchors.add(2 * (out + 1) + 1).cast::<u64>()),
                b1_delta,
            );
            store_pair_nt(a_out.add(out), a0, a1);
            store_pair_nt(b_out.add(out), b0, b1);
            let eq_l = vld1q_u64(eq_lo.add(x_lo).cast::<u64>());
            wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, mul_q(a1, b1)));
            wide_xor(
                &mut pinf_acc,
                mul_unreduced_q(
                    eq_l,
                    mul_q(veorq_u64(a0, a1), veorq_u64(b0, b1)),
                ),
            );
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

    let mut p1_acc = F256Unreduced::ZERO;
    let mut pinf_acc = F256Unreduced::ZERO;

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

        let a0 = a_even_0 + r_fold * (a_even_0 + a_odd_0);
        let a1 = a_even_1 + r_fold * (a_even_1 + a_odd_1);
        let b0 = b_even_0 + r_fold * (b_even_0 + b_odd_0);
        let b1 = b_even_1 + r_fold * (b_even_1 + b_odd_1);

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

        p1_acc ^= eq_l.mul_unreduced(a1 * b1);
        pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
    }

    (p1_acc.reduce(), pinf_acc.reduce())
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

    #[test]
    fn karatsuba_q_products_match_scalar_field_products() {
        let mut state = 0x4b41_5241_5453_5542;

        unsafe {
            for _ in 0..256 {
                let a = F128::new(splitmix64(&mut state), splitmix64(&mut state));
                let b = F128::new(splitmix64(&mut state), splitmix64(&mut state));
                let a_q = core::mem::transmute::<F128, uint64x2_t>(a);
                let b_q = core::mem::transmute::<F128, uint64x2_t>(b);

                let (p0, cross, p2) = karatsuba_products_q(a_q, b_q);
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
}

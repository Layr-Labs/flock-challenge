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

/// Reduce one arbitrary 256-bit polynomial with the same two-stage recursive
/// Binius reduction used by the default `F128` multiplier.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn reduce_wide_binius_q(value: WideNeon) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);

        // First reduce r3 into the adjacent (r1, r2) words, then reduce the
        // resulting high word into r0. This is the arbitrary-wide form of
        // `ghash_mul_binius`'s two recursive reduction stages.
        let middle = vextq_u64::<1>(value.lo, value.hi);
        let t1 = veorq_u64(middle, pmull_lane(vgetq_lane_u64::<1>(value.hi), 0x87));
        let low_word = vsetq_lane_u64::<1>(0, value.lo);
        xor3_u64(
            low_word,
            vextq_u64::<1>(zero, t1),
            pmull_lane(vgetq_lane_u64::<1>(t1), 0x87),
        )
    }
}

/// Two unreduced Karatsuba products by one shared constant. Keeping both
/// products live exposes their six independent PMULLs to the scheduler and
/// lets the caller XOR several constant-product pairs before reducing.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn mul_const_unreduced_pair_q(
    constant: core::arch::aarch64::uint64x2_t,
    x0: core::arch::aarch64::uint64x2_t,
    x1: core::arch::aarch64::uint64x2_t,
) -> (WideNeon, WideNeon) {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let constant_lo = vgetq_lane_u64::<0>(constant);
        let constant_hi = vgetq_lane_u64::<1>(constant);
        let constant_mid = constant_lo ^ constant_hi;
        let x0_mid = veorq_u64(x0, vextq_u64::<1>(x0, x0));
        let x1_mid = veorq_u64(x1, vextq_u64::<1>(x1, x1));

        let ll0 = pmull_lane(vgetq_lane_u64::<0>(x0), constant_lo);
        let hh0 = pmull_lane(vgetq_lane_u64::<1>(x0), constant_hi);
        let mm0 = pmull_lane(vgetq_lane_u64::<0>(x0_mid), constant_mid);
        let ll1 = pmull_lane(vgetq_lane_u64::<0>(x1), constant_lo);
        let hh1 = pmull_lane(vgetq_lane_u64::<1>(x1), constant_hi);
        let mm1 = pmull_lane(vgetq_lane_u64::<0>(x1_mid), constant_mid);
        let cross0 = xor3_u64(mm0, ll0, hh0);
        let cross1 = xor3_u64(mm1, ll1, hh1);

        (
            WideNeon {
                lo: veorq_u64(ll0, vextq_u64::<1>(zero, cross0)),
                hi: veorq_u64(hh0, vextq_u64::<1>(cross0, zero)),
            },
            WideNeon {
                lo: veorq_u64(ll1, vextq_u64::<1>(zero, cross1)),
                hi: veorq_u64(hh1, vextq_u64::<1>(cross1, zero)),
            },
        )
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

        let delta_a0 = veorq_u64(v0, v1);
        let delta_a1 = veorq_u64(v4, v5);
        let delta_b0 = veorq_u64(v0, v2);
        let delta_b1 = veorq_u64(v4, v6);
        let delta_ab0 = xor3_u64(v0, v1, veorq_u64(v2, v3));
        let delta_ab1 = xor3_u64(v4, v5, veorq_u64(v6, v7));

        let (mut acc0, mut acc1) = mul_const_unreduced_pair_q(rho_a, delta_a0, delta_a1);
        let (product0, product1) = mul_const_unreduced_pair_q(rho_b, delta_b0, delta_b1);
        wide_xor(&mut acc0, product0);
        wide_xor(&mut acc1, product1);
        let (product0, product1) = mul_const_unreduced_pair_q(rho_ab, delta_ab0, delta_ab1);
        wide_xor(&mut acc0, product0);
        wide_xor(&mut acc1, product1);
        acc0.lo = veorq_u64(acc0.lo, v0);
        acc1.lo = veorq_u64(acc1.lo, v4);

        (reduce_wide_binius_q(acc0), reduce_wide_binius_q(acc1))
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
            store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
            vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
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
            let delta_pair = core::mem::transmute::<[u64; 2], uint64x2_t>([a0_code ^ a1_code, 0]);
            vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);
            return (a0, a1, b_ones, b_ones, true);
        }

        let (a0, a1, b0, b1) =
            fold_four_row_codes_q(table_data, a0_code, a1_code, b0_code, b1_code);
        store_anchor_pair_nt(anchors.add(2 * x_lo), a0, b0);
        let delta_pair =
            core::mem::transmute::<[u64; 2], uint64x2_t>([a0_code ^ a1_code, b0_code ^ b1_code]);
        vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);
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
        let mut p1_even = WideNeon { lo: zero, hi: zero };
        let mut pinf_even = WideNeon { lo: zero, hi: zero };
        let mut p1_odd = WideNeon { lo: zero, hi: zero };
        let mut pinf_odd = WideNeon { lo: zero, hi: zero };
        let mut w0 = WideNeon { lo: zero, hi: zero };
        let mut w3 = WideNeon { lo: zero, hi: zero };
        let mut w4 = WideNeon { lo: zero, hi: zero };
        let mut w5 = WideNeon { lo: zero, hi: zero };

        let ones_bytes = [0xFFu8; 8];
        let b_ones = fold_row_q(table_data, ones_bytes.as_ptr());
        let ones_is_one = is_zero_q(veorq_u64(
            b_ones,
            core::mem::transmute::<F128, uint64x2_t>(F128::ONE),
        ));

        macro_rules! zero_group {
            ($u:expr) => {{
                let x_lo0 = 2 * $u;
                let x_lo1 = x_lo0 + 1;
                store_anchor_pair_nt(anchors.add(2 * x_lo0), zero, zero);
                vst1q_u64(deltas.add(x_lo0 * 16).cast::<u64>(), zero);
                store_anchor_pair_nt(anchors.add(2 * x_lo1), zero, zero);
                vst1q_u64(deltas.add(x_lo1 * 16).cast::<u64>(), zero);
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
                );
                let (a2, a3, b2, b3, deg1) = r2_pair_fold_and_store(
                    table_data, a_packed, b_packed, anchors, deltas, x_lo1, pad1, degen, b_ones,
                );

                // The odd lane's weight drives the whole group; see the doc above.
                let w = vld1q_u64(eq_lo.add(x_lo1).cast::<u64>());

                if deg0 & deg1 & ones_is_one {
                    // b === 1 across the group: e_b = o_b = 0 kills W3/W4/W5 and
                    // both G(inf) chains, and every surviving product is `w * a_i`.
                    if FULL {
                        wide_xor(&mut p1_even, mul_unreduced_q(w, a1));
                    }
                    if !ODD_ON_GPU {
                        wide_xor(&mut p1_odd, mul_unreduced_q(w, a3));
                    }
                    wide_xor(&mut w0, mul_unreduced_q(w, a2));
                } else {
                    let a0w = mul_q(w, a0);
                    let a1w = mul_q(w, a1);
                    let (a2w, a3w) = if pad1 {
                        (zero, zero)
                    } else {
                        (mul_q(w, a2), mul_q(w, a3))
                    };

                    // ---- round-two products, split by the parity of x' ----
                    if FULL {
                        wide_xor(&mut p1_even, mul_unreduced_q(a1w, b1));
                        if !deg0 {
                            wide_xor(
                                &mut pinf_even,
                                mul_unreduced_q(veorq_u64(a0w, a1w), veorq_u64(b0, b1)),
                            );
                        }
                    }
                    if !pad1 {
                        if !ODD_ON_GPU {
                            wide_xor(&mut p1_odd, mul_unreduced_q(a3w, b3));
                            if !deg1 {
                                wide_xor(
                                    &mut pinf_odd,
                                    mul_unreduced_q(veorq_u64(a2w, a3w), veorq_u64(b2, b3)),
                                );
                            }
                        }
                        wide_xor(&mut w0, mul_unreduced_q(a2w, b2));
                    }

                    // ---- deferred round-three aggregates (no extra lookups) ----
                    let e_aw = veorq_u64(a0w, a2w);
                    let o_aw = veorq_u64(a1w, a3w);
                    let e_b = veorq_u64(b0, b2);
                    let o_b = veorq_u64(b1, b3);
                    wide_xor(&mut w3, mul_unreduced_q(e_aw, e_b));
                    wide_xor(&mut w4, mul_unreduced_q(o_aw, o_b));
                    wide_xor(
                        &mut w5,
                        mul_unreduced_q(veorq_u64(e_aw, o_aw), veorq_u64(e_b, o_b)),
                    );
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
        let mut p1_acc = WideNeon { lo: zero, hi: zero };
        let mut pinf_acc = WideNeon { lo: zero, hi: zero };

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
                wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
                let b_sum = veorq_u64(bv[0], bv[1]);
                if !is_zero_q(b_sum) {
                    let g_inf = mul_q(veorq_u64(av[0], av[1]), b_sum);
                    wide_xor(&mut pinf_acc, mul_unreduced_q(eq_l, g_inf));
                }
            } else {
                let g1 = mul_q(av[1], bv[1]);
                let g_inf = mul_q(veorq_u64(av[0], av[1]), veorq_u64(bv[0], bv[1]));
                wide_xor(&mut p1_acc, mul_unreduced_q(eq_l, g1));
                wide_xor(&mut pinf_acc, mul_unreduced_q(eq_l, g_inf));
            }
        }

        (
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1_acc)),
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf_acc)),
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
pub(crate) unsafe fn fold2_compact_and_round45_chunk_neon_8(
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
        let one_q = core::mem::transmute::<F128, uint64x2_t>(F128::ONE);
        let mut p1_even = WideNeon { lo: zero, hi: zero };
        let mut pinf_even = WideNeon { lo: zero, hi: zero };
        let mut p1_odd = WideNeon { lo: zero, hi: zero };
        let mut pinf_odd = WideNeon { lo: zero, hi: zero };
        let mut w0 = WideNeon { lo: zero, hi: zero };
        let mut w3 = WideNeon { lo: zero, hi: zero };
        let mut w4 = WideNeon { lo: zero, hi: zero };
        let mut w5 = WideNeon { lo: zero, hi: zero };

        debug_assert!(out_pairs >= 2 && out_pairs.is_multiple_of(2));
        let n5 = out_pairs / 2;
        for t in 0..n5 {
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
                    wide_xor(&mut p1_even, mul_unreduced_q(w, av[1]));
                    wide_xor(&mut p1_odd, mul_unreduced_q(w, av[3]));
                    wide_xor(&mut w0, mul_unreduced_q(w, av[2]));
                    continue;
                }
            }

            // Weight-sharing reshape: four reduced row scalings, then every
            // product below is one unreduced multiply.
            let a0w = mul_q(w, av[0]);
            let a1w = mul_q(w, av[1]);
            let a2w = mul_q(w, av[2]);
            let a3w = mul_q(w, av[3]);

            // ---- round-four products, split by round-4-pair parity ----
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

            // ---- deferred round-five aggregates (no extra loads) ----
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

    // One composed output from four consecutive inputs: two rho_a folds, one
    // rho_b fold. Exactly the value the two sequential passes produce.
    #[inline(always)]
    fn fold4(v: &[F128], i: usize, rho_a: F128, rho_b: F128) -> F128 {
        let t0 = v[i] + rho_a * (v[i] + v[i + 1]);
        let t1 = v[i + 2] + rho_a * (v[i + 2] + v[i + 3]);
        t0 + rho_b * (t0 + t1)
    }

    let mut p1_acc = F256Unreduced::ZERO;
    let mut pinf_acc = F256Unreduced::ZERO;

    let a_out_ptr = a_out.as_mut_ptr();
    let b_out_ptr = b_out.as_mut_ptr();

    for (x_lo, &eq_l) in eq_lo.iter().enumerate() {
        let i = 8 * x_lo;
        let o = 2 * x_lo;

        let a0 = fold4(a_in, i, rho_a, rho_b);
        let a1 = fold4(a_in, i + 4, rho_a, rho_b);
        let b0 = fold4(b_in, i, rho_a, rho_b);
        let b1 = fold4(b_in, i + 4, rho_a, rho_b);

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

    let rho_a_q = unsafe { core::mem::transmute::<F128, uint64x2_t>(rho_a) };
    let rho_b_q = unsafe { core::mem::transmute::<F128, uint64x2_t>(rho_b) };
    let rho_ab_q = unsafe { core::mem::transmute::<F128, uint64x2_t>(rho_ab) };
    let mut p1_acc = F256Unreduced::ZERO;
    let mut pinf_acc = F256Unreduced::ZERO;

    for (x_lo, &eq_l) in eq_lo.iter().enumerate() {
        let i = 8 * x_lo;
        let o = 2 * x_lo;
        let (a0, a1, b0, b1) = unsafe {
            // SAFETY: the release-mode geometry assertions and loop bound put
            // both eight-value pairs wholly inside their input tables.
            let (a0, a1) =
                fold4_pair_expanded_binius_q(a_in.as_ptr().add(i), rho_a_q, rho_b_q, rho_ab_q);
            let (b0, b1) =
                fold4_pair_expanded_binius_q(b_in.as_ptr().add(i), rho_a_q, rho_b_q, rho_ab_q);
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
        p1_acc ^= eq_l.mul_unreduced(a1 * b1);
        pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
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

    // One composed output from four consecutive inputs — identical expression
    // to `fold2_and_message_body`, so the stored tables are value-identical.
    #[inline(always)]
    fn fold4(v: &[F128], i: usize, rho_a: F128, rho_b: F128) -> F128 {
        let t0 = v[i] + rho_a * (v[i] + v[i + 1]);
        let t1 = v[i + 2] + rho_a * (v[i + 2] + v[i + 3]);
        t0 + rho_b * (t0 + t1)
    }

    let mut acc = [F256Unreduced::ZERO; 8];

    let a_out_ptr = a_out.as_mut_ptr();
    let b_out_ptr = b_out.as_mut_ptr();

    for t in 0..eq_lo.len() / 2 {
        let i = 16 * t;
        let o = 4 * t;

        let a0 = fold4(a_in, i, rho_a, rho_b);
        let a1 = fold4(a_in, i + 4, rho_a, rho_b);
        let a2 = fold4(a_in, i + 8, rho_a, rho_b);
        let a3 = fold4(a_in, i + 12, rho_a, rho_b);
        let b0 = fold4(b_in, i, rho_a, rho_b);
        let b1 = fold4(b_in, i + 4, rho_a, rho_b);
        let b2 = fold4(b_in, i + 8, rho_a, rho_b);
        let b3 = fold4(b_in, i + 12, rho_a, rho_b);

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
        acc[0] ^= a1w.mul_unreduced(b1);
        acc[1] ^= (a0w + a1w).mul_unreduced(b0 + b1);
        acc[2] ^= a3w.mul_unreduced(b3);
        acc[3] ^= (a2w + a3w).mul_unreduced(b2 + b3);
        acc[4] ^= a2w.mul_unreduced(b2);
        let (e_aw, e_b) = (a0w + a2w, b0 + b2);
        let (o_aw, o_b) = (a1w + a3w, b1 + b3);
        acc[5] ^= e_aw.mul_unreduced(e_b);
        acc[6] ^= o_aw.mul_unreduced(o_b);
        acc[7] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
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

    let rho_a_q = unsafe { core::mem::transmute::<F128, uint64x2_t>(rho_a) };
    let rho_b_q = unsafe { core::mem::transmute::<F128, uint64x2_t>(rho_b) };
    let rho_ab_q = unsafe { core::mem::transmute::<F128, uint64x2_t>(rho_ab) };
    let mut acc = [F256Unreduced::ZERO; 8];

    for t in 0..eq_lo.len() / 2 {
        let i = 16 * t;
        let o = 4 * t;

        let (a0, a1, a2, a3, b0, b1, b2, b3) = unsafe {
            // SAFETY: the geometry assertions and loop bound place each pair
            // wholly inside its corresponding 16-element source group.
            let (a0, a1) =
                fold4_pair_expanded_binius_q(a_in.as_ptr().add(i), rho_a_q, rho_b_q, rho_ab_q);
            let (a2, a3) =
                fold4_pair_expanded_binius_q(a_in.as_ptr().add(i + 8), rho_a_q, rho_b_q, rho_ab_q);
            let (b0, b1) =
                fold4_pair_expanded_binius_q(b_in.as_ptr().add(i), rho_a_q, rho_b_q, rho_ab_q);
            let (b2, b3) =
                fold4_pair_expanded_binius_q(b_in.as_ptr().add(i + 8), rho_a_q, rho_b_q, rho_ab_q);
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
        let wt = eq_lo[2 * t + 1];
        let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
        acc[0] ^= a1w.mul_unreduced(b1);
        acc[1] ^= (a0w + a1w).mul_unreduced(b0 + b1);
        acc[2] ^= a3w.mul_unreduced(b3);
        acc[3] ^= (a2w + a3w).mul_unreduced(b2 + b3);
        acc[4] ^= a2w.mul_unreduced(b2);
        let (e_aw, e_b) = (a0w + a2w, b0 + b2);
        let (o_aw, o_b) = (a1w + a3w, b1 + b3);
        acc[5] ^= e_aw.mul_unreduced(e_b);
        acc[6] ^= o_aw.mul_unreduced(o_b);
        acc[7] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
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

    let rho_a_q = unsafe { core::mem::transmute::<F128, uint64x2_t>(rho_a) };
    let rho_b_q = unsafe { core::mem::transmute::<F128, uint64x2_t>(rho_b) };
    let rho_ab_q = unsafe { core::mem::transmute::<F128, uint64x2_t>(rho_ab) };
    let mut acc = [F256Unreduced::ZERO; 8];
    let a_out_ptr = a_out.as_mut_ptr();
    let b_out_ptr = b_out.as_mut_ptr();

    for t in 0..eq_lo.len() / 2 {
        let i = 16 * t;
        let o = 4 * t;

        let (a0, a1, a2, a3, b0, b1, b2, b3) = unsafe {
            // SAFETY: the geometry assertions and loop bound place each pair
            // wholly inside its corresponding 16-element source group.
            let (a0, a1) =
                fold4_pair_expanded_binius_q(a_in.as_ptr().add(i), rho_a_q, rho_b_q, rho_ab_q);
            let (a2, a3) =
                fold4_pair_expanded_binius_q(a_in.as_ptr().add(i + 8), rho_a_q, rho_b_q, rho_ab_q);
            let (b0, b1) =
                fold4_pair_expanded_binius_q(b_in.as_ptr().add(i), rho_a_q, rho_b_q, rho_ab_q);
            let (b2, b3) =
                fold4_pair_expanded_binius_q(b_in.as_ptr().add(i + 8), rho_a_q, rho_b_q, rho_ab_q);
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
        let wt = eq_lo[2 * t + 1];
        let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
        acc[0] ^= a1w.mul_unreduced(b1);
        acc[1] ^= (a0w + a1w).mul_unreduced(b0 + b1);
        acc[2] ^= a3w.mul_unreduced(b3);
        acc[3] ^= (a2w + a3w).mul_unreduced(b2 + b3);
        acc[4] ^= a2w.mul_unreduced(b2);
        let (e_aw, e_b) = (a0w + a2w, b0 + b2);
        let (o_aw, o_b) = (a1w + a3w, b1 + b3);
        acc[5] ^= e_aw.mul_unreduced(e_b);
        acc[6] ^= o_aw.mul_unreduced(o_b);
        acc[7] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
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

    fn fold4_scalar(values: &[F128], i: usize, rho_a: F128, rho_b: F128) -> F128 {
        let t0 = values[i] + rho_a * (values[i] + values[i + 1]);
        let t1 = values[i + 2] + rho_a * (values[i + 2] + values[i + 3]);
        t0 + rho_b * (t0 + t1)
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

        let run = |periodic_padding: bool| {
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
                    out.as_mut_ptr(),
                );
            }
            (anchors, deltas, out)
        };

        let periodic = run(true);
        let generic = run(false);
        assert_eq!(periodic.0, generic.0, "anchor schedule mismatch");
        assert_eq!(periodic.1, generic.1, "delta schedule mismatch");
        assert_eq!(periodic.2, generic.2, "message/lookahead mismatch");
    }
}

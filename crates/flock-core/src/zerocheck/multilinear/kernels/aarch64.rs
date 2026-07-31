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

/// Reconstruct one compact round-two level at the sampled challenge and form
/// the next sumcheck message. `scaled_table` is the univariate fold table with
/// every entry multiplied by that challenge, so reconstruction needs only
/// cache-resident table loads and XORs:
///
/// `folded = anchor + scaled_table_fold(packed_row0 XOR packed_row1)`.
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

        for x_lo in 0..lo_size {
            let out = 2 * x_lo;
            let delta = deltas.add(out * 16);
            let a0_code = u64::from_le(core::ptr::read_unaligned(delta.cast::<u64>()));
            let b0_code = u64::from_le(core::ptr::read_unaligned(delta.add(8).cast::<u64>()));
            let a1_code = u64::from_le(core::ptr::read_unaligned(delta.add(16).cast::<u64>()));
            let b1_code = u64::from_le(core::ptr::read_unaligned(delta.add(24).cast::<u64>()));
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

/// Multiply two independent values by the same constant and reduce them in
/// lane-paired form. Constant-r Karatsuba needs six PMULLs total rather than
/// the generic two-product schoolbook kernel's eight, and the paired
/// shift-XOR reduction spends zero PMULLs. Mirrors
/// `field::f128_slice::aarch64::mul_const_vec2`.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_const_vec2_q(
    r: core::arch::aarch64::uint64x2_t,
    x0: core::arch::aarch64::uint64x2_t,
    x1: core::arch::aarch64::uint64x2_t,
) -> [core::arch::aarch64::uint64x2_t; 2] {
    use core::arch::aarch64::*;
    unsafe {
        let r_lo = vgetq_lane_u64::<0>(r);
        let r_hi = vgetq_lane_u64::<1>(r);
        let r_mid = veorq_u64(r, vextq_u64::<1>(r, r));
        let x0_mid = veorq_u64(x0, vextq_u64::<1>(x0, x0));
        let x1_mid = veorq_u64(x1, vextq_u64::<1>(x1, x1));

        let p0_ll = pmull_lane(vgetq_lane_u64::<0>(x0), r_lo);
        let p0_hh = pmull_lane(vgetq_lane_u64::<1>(x0), r_hi);
        let p0_mm = pmull_lane(vgetq_lane_u64::<0>(x0_mid), vgetq_lane_u64::<0>(r_mid));
        let p1_ll = pmull_lane(vgetq_lane_u64::<0>(x1), r_lo);
        let p1_hh = pmull_lane(vgetq_lane_u64::<1>(x1), r_hi);
        let p1_mm = pmull_lane(vgetq_lane_u64::<0>(x1_mid), vgetq_lane_u64::<0>(r_mid));
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

/// Q-register-native tail fold + message. **Measured 1.4x SLOWER than the
/// scalar-struct path on M4 Max** (`fold_and_message_timing`: 0.69x at
/// eq_len=2^14..2^17) despite dropping ~44 PMULL per 2-output group to 28 and
/// removing all GPR<->FPR round trips. The wide OoO core hides the scalar
/// version's fmov traffic, while this version's lane-zip constant-Karatsuba
/// reduction adds serial vector-ALU latency the PMULL savings do not repay.
/// Kept (unwired) so the experiment is not repeated.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[target_feature(enable = "aes")]
unsafe fn fold_and_message_aarch64_q(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use core::arch::aarch64::*;

    assert_eq!(a_in.len(), 2 * a_out.len());
    assert_eq!(b_in.len(), 2 * b_out.len());
    assert_eq!(a_out.len(), 2 * eq_lo.len());

    unsafe {
        let zero = vdupq_n_u64(0);
        let r_q = core::mem::transmute::<F128, uint64x2_t>(r_fold);
        let ap = a_in.as_ptr();
        let bp = b_in.as_ptr();
        let aop = a_out.as_mut_ptr();
        let bop = b_out.as_mut_ptr();
        let eqp = eq_lo.as_ptr();

        let mut p1 = WideNeon { lo: zero, hi: zero };
        let mut pinf = WideNeon { lo: zero, hi: zero };

        for x_lo in 0..eq_lo.len() {
            let i = 4 * x_lo;
            let o = 2 * x_lo;

            let ae0 = vld1q_u64(ap.add(i).cast::<u64>());
            let ao0 = vld1q_u64(ap.add(i + 1).cast::<u64>());
            let ae1 = vld1q_u64(ap.add(i + 2).cast::<u64>());
            let ao1 = vld1q_u64(ap.add(i + 3).cast::<u64>());
            let be0 = vld1q_u64(bp.add(i).cast::<u64>());
            let bo0 = vld1q_u64(bp.add(i + 1).cast::<u64>());
            let be1 = vld1q_u64(bp.add(i + 2).cast::<u64>());
            let bo1 = vld1q_u64(bp.add(i + 3).cast::<u64>());

            // Fold at r: v = even ^ r*(even^odd), both values of each
            // polynomial through one paired constant-r multiply.
            let pa = mul_const_vec2_q(r_q, veorq_u64(ae0, ao0), veorq_u64(ae1, ao1));
            let a0 = veorq_u64(ae0, pa[0]);
            let a1 = veorq_u64(ae1, pa[1]);
            let pb = mul_const_vec2_q(r_q, veorq_u64(be0, bo0), veorq_u64(be1, bo1));
            let b0 = veorq_u64(be0, pb[0]);
            let b1 = veorq_u64(be1, pb[1]);

            vst1q_u64(aop.add(o).cast::<u64>(), a0);
            vst1q_u64(aop.add(o + 1).cast::<u64>(), a1);
            vst1q_u64(bop.add(o).cast::<u64>(), b0);
            vst1q_u64(bop.add(o + 1).cast::<u64>(), b1);

            // Message contributions: eq * (a1·b1) and eq * ((a0+a1)·(b0+b1)),
            // with the inner product reduced and the eq product left wide.
            let g1 = mul_q(a1, b1);
            let ginf = mul_q(veorq_u64(a0, a1), veorq_u64(b0, b1));
            let eq = vld1q_u64(eqp.add(x_lo).cast::<u64>());
            wide_xor(&mut p1, mul_unreduced_q(eq, g1));
            wide_xor(&mut pinf, mul_unreduced_q(eq, ginf));
        }

        (
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1)),
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf)),
        )
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
) -> (F128, F128) {
    // The q-register rewrite measured slower (see fold_and_message_aarch64_q);
    // the scalar-struct body remains the fastest known form on Apple Silicon.
    fold_and_message_scalar_ref(a_in, b_in, a_out, b_out, r_fold, eq_lo)
}

/// Scalar-struct implementation of the fused tail fold + message: the live
/// path, and the bit-exactness oracle for the (slower, unwired)
/// [`fold_and_message_aarch64_q`] in tests.
#[cfg(target_arch = "aarch64")]
pub(crate) fn fold_and_message_scalar_ref(
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

    #[test]
    fn mul_const_vec2_q_matches_scalar_products() {
        let mut state = 0x636f_6e73_745f_7632;
        unsafe {
            for _ in 0..256 {
                let r = F128::new(splitmix64(&mut state), splitmix64(&mut state));
                let x0 = F128::new(splitmix64(&mut state), splitmix64(&mut state));
                let x1 = F128::new(splitmix64(&mut state), splitmix64(&mut state));
                let out = mul_const_vec2_q(
                    core::mem::transmute::<F128, uint64x2_t>(r),
                    core::mem::transmute::<F128, uint64x2_t>(x0),
                    core::mem::transmute::<F128, uint64x2_t>(x1),
                );
                assert_eq!(core::mem::transmute::<uint64x2_t, F128>(out[0]), r * x0);
                assert_eq!(core::mem::transmute::<uint64x2_t, F128>(out[1]), r * x1);
            }
        }
    }

    #[test]
    fn fold_and_message_q_matches_scalar_reference() {
        let mut state = 0x666f_6c64_6d73_6721;
        for eq_len in [1usize, 2, 3, 8, 33, 128] {
            let n_in = 4 * eq_len;
            let n_out = 2 * eq_len;
            let rand_f128 =
                |state: &mut u64| F128::new(splitmix64(state), splitmix64(state));

            let a_in: Vec<F128> = (0..n_in).map(|_| rand_f128(&mut state)).collect();
            let b_in: Vec<F128> = (0..n_in).map(|_| rand_f128(&mut state)).collect();
            let eq_lo: Vec<F128> = (0..eq_len).map(|_| rand_f128(&mut state)).collect();
            let r_fold = rand_f128(&mut state);

            let mut a_out_ref = vec![F128::ZERO; n_out];
            let mut b_out_ref = vec![F128::ZERO; n_out];
            let expected = fold_and_message_scalar_ref(
                &a_in,
                &b_in,
                &mut a_out_ref,
                &mut b_out_ref,
                r_fold,
                &eq_lo,
            );

            let mut a_out_q = vec![F128::ZERO; n_out];
            let mut b_out_q = vec![F128::ZERO; n_out];
            let got = unsafe {
                fold_and_message_aarch64_q(
                    &a_in,
                    &b_in,
                    &mut a_out_q,
                    &mut b_out_q,
                    r_fold,
                    &eq_lo,
                )
            };

            assert_eq!(got, expected, "messages diverge at eq_len={eq_len}");
            assert_eq!(a_out_q, a_out_ref, "a_out diverges at eq_len={eq_len}");
            assert_eq!(b_out_q, b_out_ref, "b_out diverges at eq_len={eq_len}");
        }
    }

    /// Timing harness: `cargo test --profile challenge -p flock-core --lib -- \
    ///   --ignored --nocapture fold_and_message_timing`
    #[test]
    #[ignore]
    fn fold_and_message_timing() {
        let mut state = 0x7469_6d69_6e67_2121;
        let rand_f128 = |state: &mut u64| F128::new(splitmix64(state), splitmix64(state));

        for log_eq in [10usize, 14, 17] {
            let eq_len = 1 << log_eq;
            let n_in = 4 * eq_len;
            let n_out = 2 * eq_len;
            let a_in: Vec<F128> = (0..n_in).map(|_| rand_f128(&mut state)).collect();
            let b_in: Vec<F128> = (0..n_in).map(|_| rand_f128(&mut state)).collect();
            let eq_lo: Vec<F128> = (0..eq_len).map(|_| rand_f128(&mut state)).collect();
            let r_fold = rand_f128(&mut state);
            let mut a_out = vec![F128::ZERO; n_out];
            let mut b_out = vec![F128::ZERO; n_out];

            let reps = (1usize << 24) / eq_len;
            let time = |f: &mut dyn FnMut() -> (F128, F128)| {
                let mut acc = F128::ZERO;
                let start = std::time::Instant::now();
                for _ in 0..reps {
                    let (x, y) = f();
                    acc = acc + x + y;
                }
                (start.elapsed().as_nanos() as f64 / reps as f64, acc)
            };

            let (t_scalar, c_scalar) = time(&mut || {
                fold_and_message_scalar_ref(&a_in, &b_in, &mut a_out, &mut b_out, r_fold, &eq_lo)
            });
            let (t_q, c_q) = time(&mut || unsafe {
                fold_and_message_aarch64_q(&a_in, &b_in, &mut a_out, &mut b_out, r_fold, &eq_lo)
            });
            assert_eq!(c_scalar, c_q);
            println!(
                "eq_len=2^{log_eq}: scalar {:>10.1} ns/call  q {:>10.1} ns/call  ({:.2}x)",
                t_scalar,
                t_q,
                t_scalar / t_q
            );
        }
    }
}

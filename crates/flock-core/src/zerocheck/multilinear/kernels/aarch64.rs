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

/// Binius-structured GHASH multiply, operands and result in q registers.
///
/// [`mul_q`] above is Karatsuba: three product PMULLs plus two reduction
/// PMULLs. Five is fewer than binius's six, but `ghash_mul_binius`'s own doc
/// comment records why the scalar field layer picked binius on M-series anyway
/// — "fewer scalar shifts in the dep chain", i.e. it wins on latency, not
/// throughput. Karatsuba's `pm = (a.lo^a.hi)·(b.lo^b.hi)` serialises two XORs
/// ahead of its third PMULL, and the tail fold's multiply sits directly on the
/// critical path (`a0 = a_even + r_fold·(a_even + a_odd)` feeds both the store
/// and the next multiply), so latency is what matters here.
///
/// Used only by [`fold_and_message_q`]; the round-two kernels keep [`mul_q`],
/// whose lookup-table structure has different balance and which is already
/// promoted.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn mul_q_binius(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);

        let t0 = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let t1a = pmull_lane(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<1>(b));
        let t1b = pmull_lane(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<0>(b));
        let t2 = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ));
        let t1_cross = veorq_u64(t1a, t1b);

        // t1 += x^64 · t2 (mod p).
        let t1 = xor3_u64(
            t1_cross,
            vextq_u64::<1>(zero, t2),
            pmull_lane(vgetq_lane_u64::<1>(t2), 0x87),
        );

        // t0 += x^64 · t1 (mod p).
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

            let g1 = mul_q_binius(a1, b1);
            let g_inf = mul_q_binius(veorq_u64(a0, a1), veorq_u64(b0, b1));
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
    debug_assert_eq!(a_in.len(), 2 * a_out.len());
    debug_assert_eq!(b_in.len(), 2 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());

    // The vector-resident path is the same pattern the round-two kernels in
    // this file already use (`WideNeon` accumulators, `mul_q`,
    // `mul_unreduced_q`, `reduce_wide_q`); this kernel was the one that never
    // adopted it. `FLOCK_NO_ZC_TAIL_NEON=1` restores the scalar chain below in
    // the same binary.
    #[cfg(target_feature = "aes")]
    if vector_resident_tail() {
        // SAFETY: the cfg gate supplies `aes`; the length contract asserted
        // above is exactly what the vector form indexes.
        return unsafe { fold_and_message_q(a_in, b_in, a_out, b_out, r_fold, eq_lo) };
    }

    fold_and_message_scalar(a_in, b_in, a_out, b_out, r_fold, eq_lo)
}

/// Scalar `F128`/`F256Unreduced` form, retained as the non-`aes` fallback, as
/// the `FLOCK_NO_ZC_TAIL_NEON` control, and as the equivalence oracle for
/// [`fold_and_message_q`].
///
/// `inline(never)` keeps this cold body out of the hot chunk closure: when the
/// vector path is active this code never runs, and inlining it only inflated
/// the closure's instruction footprint.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn fold_and_message_scalar(
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

/// Whether the vector-resident multilinear tail kernel is used.
///
/// `FLOCK_NO_ZC_TAIL_NEON=1` restores the scalar `F128`/`F256Unreduced` chain
/// in the same binary, so a candidate/control pair differs only in this
/// dispatch. Read once per process.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
fn vector_resident_tail() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_TAIL_NEON").is_none())
}

/// Vector-resident form of [`fold_and_message_aarch64`].
///
/// The scalar chain above accumulates into `F256Unreduced`, a four-word struct,
/// so each `^=` is four scalar `u64` XORs and each `mul_unreduced` repacks a
/// NEON result into it — 48 scalar XORs and 45 general-purpose/vector moves per
/// emitted chunk against 56 PMULLs. Here both deferred accumulators stay in
/// `WideNeon` (two q registers) and the four folded values never leave vector
/// registers between load and store.
///
/// The multiply count, the butterfly order, the deferred-reduction structure
/// and the single final `reduce` are all unchanged, so the folded tables and
/// both returned message coordinates are bit-identical: `reduce` is F2-linear,
/// XOR is associative, and the same products are summed in the same order.
///
/// # Safety
/// Requires the `aes` target feature. `a_in`/`b_in` must hold `4 * eq_lo.len()`
/// elements and `a_out`/`b_out` exactly `2 * eq_lo.len()`.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes")]
unsafe fn fold_and_message_q(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let rf = vld1q_u64((&raw const r_fold).cast::<u64>());
        let mut p1_acc = WideNeon { lo: zero, hi: zero };
        let mut pinf_acc = WideNeon { lo: zero, hi: zero };

        let ap = a_in.as_ptr().cast::<u64>();
        let bp = b_in.as_ptr().cast::<u64>();
        let aq = a_out.as_mut_ptr().cast::<u64>();
        let bq = b_out.as_mut_ptr().cast::<u64>();

        for (x_lo, eq_l) in eq_lo.iter().enumerate() {
            let i = 8 * x_lo; // 4 F128 inputs, 2 u64 lanes each
            let o = 4 * x_lo; // 2 F128 outputs

            let a_even_0 = vld1q_u64(ap.add(i));
            let a_odd_0 = vld1q_u64(ap.add(i + 2));
            let a_even_1 = vld1q_u64(ap.add(i + 4));
            let a_odd_1 = vld1q_u64(ap.add(i + 6));
            let b_even_0 = vld1q_u64(bp.add(i));
            let b_odd_0 = vld1q_u64(bp.add(i + 2));
            let b_even_1 = vld1q_u64(bp.add(i + 4));
            let b_odd_1 = vld1q_u64(bp.add(i + 6));

            let a0 = veorq_u64(a_even_0, mul_q_binius(rf, veorq_u64(a_even_0, a_odd_0)));
            let a1 = veorq_u64(a_even_1, mul_q_binius(rf, veorq_u64(a_even_1, a_odd_1)));
            let b0 = veorq_u64(b_even_0, mul_q_binius(rf, veorq_u64(b_even_0, b_odd_0)));
            let b1 = veorq_u64(b_even_1, mul_q_binius(rf, veorq_u64(b_even_1, b_odd_1)));

            vst1q_u64(aq.add(o), a0);
            vst1q_u64(aq.add(o + 2), a1);
            vst1q_u64(bq.add(o), b0);
            vst1q_u64(bq.add(o + 2), b1);

            let eq = vld1q_u64((&raw const *eq_l).cast::<u64>());
            let g1 = mul_q(a1, b1);
            let g_inf = mul_q(veorq_u64(a0, a1), veorq_u64(b0, b1));
            wide_xor(&mut p1_acc, mul_unreduced_q(eq, g1));
            wide_xor(&mut pinf_acc, mul_unreduced_q(eq, g_inf));
        }

        (
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(p1_acc)),
            core::mem::transmute::<uint64x2_t, F128>(reduce_wide_q(pinf_acc)),
        )
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

    fn rand_f128(state: &mut u64) -> F128 {
        F128 {
            lo: splitmix64(state),
            hi: splitmix64(state),
        }
    }

    /// The vector-resident multilinear tail kernel must be bit-identical to the
    /// scalar `F128`/`F256Unreduced` chain: both folded output tables and both
    /// returned message coordinates, across the `lo_size` values the ranked
    /// tail uses.
    #[test]
    fn fold_and_message_q_matches_scalar() {
        let mut state = 0x5441_494C_5F4E_454F;
        for &lo_size in &[1usize, 2, 4, 8, 64, 256, 512] {
            let a_in: Vec<F128> = (0..4 * lo_size).map(|_| rand_f128(&mut state)).collect();
            let b_in: Vec<F128> = (0..4 * lo_size).map(|_| rand_f128(&mut state)).collect();
            let eq_lo: Vec<F128> = (0..lo_size).map(|_| rand_f128(&mut state)).collect();
            let r_fold = rand_f128(&mut state);

            let mut aw = vec![F128::ZERO; 2 * lo_size];
            let mut bw = vec![F128::ZERO; 2 * lo_size];
            let want = fold_and_message_scalar(&a_in, &b_in, &mut aw, &mut bw, r_fold, &eq_lo);

            let mut ag = vec![F128::ZERO; 2 * lo_size];
            let mut bg = vec![F128::ZERO; 2 * lo_size];
            // SAFETY: this module carries `aes` via cfg; lengths satisfy the
            // documented 4:2:1 in/out/eq contract.
            let got = unsafe {
                fold_and_message_q(&a_in, &b_in, &mut ag, &mut bg, r_fold, &eq_lo)
            };

            assert_eq!(ag, aw, "a_out mismatch at lo_size={lo_size}");
            assert_eq!(bg, bw, "b_out mismatch at lo_size={lo_size}");
            assert_eq!(got, want, "message mismatch at lo_size={lo_size}");
        }
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

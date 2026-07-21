use crate::field::{F128, F256Unreduced};

/// Non-temporal store of two adjacent `F128` values via `stnp`. Bypasses the
/// read-for-ownership fetch on write-allocate, which is pure waste for outputs
/// that exceed every cache and are read back only once, sequentially, next
/// round. Gated conservatively to the very largest tail rounds.
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn stnp_pair(a: F128, b: F128, dst: *mut F128) {
    use core::arch::aarch64::uint8x16_t;
    // SAFETY: F128 and uint8x16_t are both 16-byte, 16-byte-aligned POD; the
    // reinterpret is a bit-level view with no UB.
    let av: uint8x16_t = unsafe { core::mem::transmute(a) };
    let bv: uint8x16_t = unsafe { core::mem::transmute(b) };
    unsafe {
        std::arch::asm!(
            "stnp {a:q}, {b:q}, [{dst}]",
            a = in(vreg) av,
            b = in(vreg) bv,
            dst = in(reg) dst,
            options(nostack, preserves_flags),
        );
    }
}

/// Fold two adjacent output values while keeping both results in registers.
/// `out_base` is measured in folded/output elements, so the four source
/// elements begin at `2 * out_base`.
///
/// # Safety
/// Requires the `aes` target feature and four in-bounds source elements.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn fold_two(src: &[F128], out_base: usize, r: F128) -> [F128; 2] {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    let s = 2 * out_base;
    let e0 = src[s];
    let o0 = src[s + 1];
    let e1 = src[s + 2];
    let o1 = src[s + 3];
    // SAFETY: this helper carries the required target feature.
    let prod = unsafe { ghash_mul_vec2_neon([r, r], [e0 + o0, e1 + o1]) };
    [e0 + prod[0], e1 + prod[1]]
}

/// ARM fused tail kernel: fold `a_in`/`b_in` and form the next sumcheck
/// message without rereading the freshly written output arrays.
///
/// The generic ARM path first writes all folded values, then streams both
/// output arrays again. At the benchmark's first tail round that is 64 MiB of
/// avoidable reads. This kernel uses the same two-output PMULL fold primitive,
/// but consumes each pair directly from registers before moving on.
///
/// When `a_out` exceeds every cache (the first 1–3 tail rounds: 64–512 MiB),
/// the folded outputs are written with non-temporal `stnp` stores: the next
/// round reads them once, sequentially, so the write-allocate RFO fetch is
/// pure waste on those overflowing sizes. Smaller rounds keep regular stores
/// so the data stays cache-hot for the immediate next-round read.
///
/// # Safety
/// Requires the `aes` target feature. The caller must provide
/// `a_in.len() == b_in.len() == 2 * a_out.len()`, equal output lengths, and
/// `eq_lo.len() * 2 == a_out.len()` with an even `eq_lo.len()`.
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_and_message_neon(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    debug_assert_eq!(a_in.len(), b_in.len());
    debug_assert_eq!(a_out.len(), b_out.len());
    debug_assert_eq!(a_in.len(), 2 * a_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());
    debug_assert_eq!(eq_lo.len() & 1, 0);

    // Non-temporal stores only when the output exceeds every cache: the M4 Pro
    // L2 is ~16 MiB per cluster, so ≥ 64 MiB (4×) is safely past all levels.
    // Below that, regular stores keep the next round's read cache-hot.
    #[cfg(target_arch = "aarch64")]
    let use_nt = a_out.len() * core::mem::size_of::<F128>() >= (1 << 26); // ≥ 64 MiB
    #[cfg(not(target_arch = "aarch64"))]
    let use_nt = false;

    let mut p1_acc = F256Unreduced::ZERO;
    let mut pinf_acc = F256Unreduced::ZERO;
    let mut x_lo = 0;
    while x_lo < eq_lo.len() {
        let o = 2 * x_lo;
        // Two x_lo points = four adjacent folded values per witness. Keeping
        // this at two points limits register pressure while exposing four
        // independent two-lane fold products to the out-of-order engine.
        let aa = unsafe { fold_two(a_in, o, r_fold) };
        let ab = unsafe { fold_two(a_in, o + 2, r_fold) };
        let ba = unsafe { fold_two(b_in, o, r_fold) };
        let bb = unsafe { fold_two(b_in, o + 2, r_fold) };

        if use_nt {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                let ap = a_out.as_mut_ptr().add(o);
                let bp = b_out.as_mut_ptr().add(o);
                stnp_pair(aa[0], aa[1], ap);
                stnp_pair(ab[0], ab[1], ap.add(2));
                stnp_pair(ba[0], ba[1], bp);
                stnp_pair(bb[0], bb[1], bp.add(2));
            }
        } else {
            a_out[o] = aa[0];
            a_out[o + 1] = aa[1];
            a_out[o + 2] = ab[0];
            a_out[o + 3] = ab[1];
            b_out[o] = ba[0];
            b_out[o + 1] = ba[1];
            b_out[o + 2] = bb[0];
            b_out[o + 3] = bb[1];
        }

        let g1_a = aa[1] * ba[1];
        let g1_b = ab[1] * bb[1];
        let g_inf_a = (aa[0] + aa[1]) * (ba[0] + ba[1]);
        let g_inf_b = (ab[0] + ab[1]) * (bb[0] + bb[1]);
        p1_acc ^= eq_lo[x_lo].mul_unreduced(g1_a);
        p1_acc ^= eq_lo[x_lo + 1].mul_unreduced(g1_b);
        pinf_acc ^= eq_lo[x_lo].mul_unreduced(g_inf_a);
        pinf_acc ^= eq_lo[x_lo + 1].mul_unreduced(g_inf_b);

        x_lo += 2;
    }

    (p1_acc.reduce(), pinf_acc.reduce())
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
#[cfg(target_arch = "aarch64")]
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

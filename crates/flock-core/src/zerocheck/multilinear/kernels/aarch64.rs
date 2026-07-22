use crate::field::{F128, F256Unreduced};

/// Explicit C layout matching the promoted Rust leaf's hidden result pointer:
/// the leaf stores `p1` at byte 0 and `pinf` at byte 16.
#[cfg(target_vendor = "apple")]
#[repr(C)]
struct Round2ChunkOutput {
    p1: F128,
    pinf: F128,
}

#[cfg(target_vendor = "apple")]
#[allow(clippy::too_many_arguments)]
#[unsafe(naked)]
unsafe extern "C" fn round2_chunk_raw_neon(
    _output: *mut Round2ChunkOutput,
    _table_data: *const u8,
    _a_packed: *const u8,
    _b_packed: *const u8,
    _a_out: *mut F128,
    _b_out: *mut F128,
    _eq_lo: *const F128,
    _lo_size: usize,
    _pair_idx_base: usize,
    _pair_in_block_mask: usize,
    _useful_pairs_inclusive: usize,
) {
    core::arch::naked_asm!(include_str!("round2_chunk_raw_neon_fused.S"));
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

        a_out[o] = aa[0];
        a_out[o + 1] = aa[1];
        a_out[o + 2] = ab[0];
        a_out[o + 3] = ab[1];
        b_out[o] = ba[0];
        b_out[o + 1] = ba[1];
        b_out[o + 2] = bb[0];
        b_out[o + 3] = bb[1];

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

/// Raw-pointer leaf for one round-2 `x_hi` chunk at the protocol-fixed
/// `k_skip = 6` / eight-table-chunk geometry.
///
/// The caller advances every pointer to the beginning of one Rayon chunk.
/// Keeping this loop in a noinline leaf prevents the closure's slice and
/// capture state from occupying registers across the table-lookup and GHASH
/// dependency chains.
///
/// # Safety
/// Requires the `aes` target feature and all of the following:
///
/// - `table_data` addresses at least `8 * 256 * size_of::<F128>()` bytes;
/// - `a_packed` and `b_packed` each address `2 * lo_size * 8` bytes;
/// - `a_out` and `b_out` each address `2 * lo_size` writable `F128`s;
/// - `eq_lo` addresses `lo_size` initialized `F128`s;
/// - `pair_idx_base + lo_size` does not overflow `usize`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_vendor = "apple", allow(dead_code))]
#[inline(never)]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn round2_chunk_raw_neon_baseline(
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
    unsafe {
        let mut p1_acc = F256Unreduced::ZERO;
        let mut pinf_acc = F256Unreduced::ZERO;

        let mut a_src = a_packed;
        let mut b_src = b_packed;
        let mut a_dst = a_out;
        let mut b_dst = b_out;
        let mut eq_ptr = eq_lo;
        let mut pair_idx = pair_idx_base;
        let mut remaining = lo_size;

        while remaining != 0 {
            if (pair_idx & pair_in_block_mask) >= useful_pairs_inclusive {
                a_dst.write(F128::ZERO);
                a_dst.add(1).write(F128::ZERO);
                b_dst.write(F128::ZERO);
                b_dst.add(1).write(F128::ZERO);
            } else {
                let a0 = fold_one_row_neon_unchecked_8(table_data, a_src);
                let b0 = fold_one_row_neon_unchecked_8(table_data, b_src);
                let a1 = fold_one_row_neon_unchecked_8(table_data, a_src.add(8));
                let b1 = fold_one_row_neon_unchecked_8(table_data, b_src.add(8));

                a_dst.write(a0);
                a_dst.add(1).write(a1);
                b_dst.write(b0);
                b_dst.add(1).write(b1);

                let eq_l = eq_ptr.read();
                let g1 = a1 * b1;
                p1_acc ^= eq_l.mul_unreduced(g1);
                let g_inf = (a0 + a1) * (b0 + b1);
                pinf_acc ^= eq_l.mul_unreduced(g_inf);
            }

            a_src = a_src.add(16);
            b_src = b_src.add(16);
            a_dst = a_dst.add(2);
            b_dst = b_dst.add(2);
            eq_ptr = eq_ptr.add(1);
            pair_idx += 1;
            remaining -= 1;
        }

        (p1_acc.reduce(), pinf_acc.reduce())
    }
}

/// Apple-only byte-frozen clone of [`round2_chunk_raw_neon_baseline`].
///
/// Its two GHASH windows differ only in physical SIMD register fields, making
/// all six adjacent PMULL/PMULL2-to-EOR pairs satisfy Apple's destructive-
/// destination issue-fusion condition. The external symbol deliberately takes
/// an explicit output pointer so its C ABI exactly matches the hidden-result
/// layout observed for the promoted Rust leaf.
///
/// # Safety
/// The requirements are identical to [`round2_chunk_raw_neon_baseline`].
#[cfg(target_vendor = "apple")]
#[inline(always)]
pub(crate) unsafe fn round2_chunk_raw_neon_fused(
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
    let mut output = core::mem::MaybeUninit::<Round2ChunkOutput>::uninit();
    unsafe {
        round2_chunk_raw_neon(
            output.as_mut_ptr(),
            table_data,
            a_packed,
            b_packed,
            a_out,
            b_out,
            eq_lo,
            lo_size,
            pair_idx_base,
            pair_in_block_mask,
            useful_pairs_inclusive,
        );
        let output = output.assume_init();
        (output.p1, output.pinf)
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

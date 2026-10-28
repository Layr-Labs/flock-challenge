use super::{C_MASK_TABLE_STRIDE, F8, InvNttTableByteSingleGf8};

mod portable;

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) use portable::bit_transpose_64bytes_scalar;
#[cfg(all(
    test,
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "gfni")
    )
))]
pub(super) use portable::shift_reduce_inner_ab_scalar;

#[cfg(target_arch = "aarch64")]
pub(super) mod aarch64;

#[cfg(target_arch = "aarch64")]
pub(super) use aarch64::StaticBContext;

#[cfg(not(target_arch = "aarch64"))]
#[derive(Clone, Copy)]
pub(super) struct StaticBContext;

#[cfg(target_arch = "x86_64")]
pub(super) mod x86_64;

/// Resolve the process-wide static-B policy and inverse-NTT partial table once
/// per AB precompute, rather than once for every hot `(window, b_med)` call.
#[inline]
pub(super) fn prepare_static_b_context(
    inv_table: &InvNttTableByteSingleGf8,
    blake3_static_layout: bool,
) -> Option<StaticBContext> {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::prepare_static_b_context(inv_table, blake3_static_layout)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (inv_table, blake3_static_layout);
        None
    }
}

#[inline]
pub(super) fn static_b_context_is_prepared(context: Option<StaticBContext>) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        matches!(context, Some(StaticBContext::Prepared { .. }))
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = context;
        false
    }
}

#[inline]
pub(super) fn fast_shift_reduce_enabled() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::fast_shift_reduce_enabled()
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub(super) const AB_FAST_POLICY_PROCESS: u8 = 0;
pub(super) const AB_FAST_POLICY_FORCE_FAST: u8 = 1;

#[inline]
pub(super) fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON.
    unsafe {
        aarch64::bit_transpose_64bytes_neon(input, output);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    ))]
    // SAFETY: all target features required by the kernel are enabled.
    unsafe {
        x86_64::bit_transpose_64bytes_avx512(input, output);
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi"
        )
    )))]
    portable::bit_transpose_64bytes_scalar(input, output);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn shift_reduce_inner_ab<const FAST_POLICY: u8>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
    check_all_ones: bool,
    check_single_k0: bool,
    const_one_mask: u8,
    bstatic_w: usize,
    static_b_context: Option<StaticBContext>,
    nt_store: bool,
) {
    #[cfg(not(target_arch = "aarch64"))]
    let _ = nt_store;
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (a_col, b_col);
        aarch64::shift_reduce_inner_ab_fused_neon_checked_with_fast_policy::<FAST_POLICY>(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            check_all_ones,
            check_single_k0,
            const_one_mask,
            bstatic_w,
            static_b_context,
            nt_store,
        );
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        let _ = (
            a_col,
            b_col,
            check_all_ones,
            check_single_k0,
            const_one_mask,
            bstatic_w,
            static_b_context,
        );
        // SAFETY: all required target features are enabled at compile time.
        unsafe {
            x86_64::shift_reduce_inner_ab_x86_avx512(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
            );
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    // SAFETY: gfni is enabled at compile time; SSE2 is baseline on x86_64.
    unsafe {
        let _ = (
            check_all_ones,
            check_single_k0,
            const_one_mask,
            bstatic_w,
            static_b_context,
        );
        x86_64::shift_reduce_inner_ab_x86_sse(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            a_col,
            b_col,
        );
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "gfni")
    )))]
    {
        let _ = (
            check_all_ones,
            check_single_k0,
            const_one_mask,
            bstatic_w,
            static_b_context,
        );
        portable::shift_reduce_inner_ab_scalar(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            a_col,
            b_col,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    partial_ab: &mut [super::F128; 64],
    partial_c: &mut [super::F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // all table-selected loads.
    unsafe {
        aarch64::accumulate_convert(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    portable::accumulate_convert(
        chunk_ab_bytes,
        chunk_c_bytes,
        n_b_med,
        convert,
        eq_lo_val,
        partial_ab,
        partial_c,
    );
}

/// The eight α-free single-bit-`K` C banks for one `x_outer_lo`.
///
/// Bank `s` accumulates `Σ_b γ^b · bit_s(chunk_c_bytes[b][lane])`. For `b < 16`
/// `γ^b = F128 { lo: 1 << b }` — `gamma_pow` is repeated `mul_by_x` from ONE and
/// never reduces below bit 128 — so the entire bank value is the u16 mask of
/// bit `s` across the 16 `b_med` C bytes. The C side therefore needs **no
/// convert table, no gathers and no F128 XOR accumulators at all**.
///
/// The remaining `F128 { lo: mask } * eq_lo_val` is F₂-linear in the mask bits,
/// so it is a table lookup rather than a multiply: `mask_tables` is this
/// `x_outer_lo`'s [`C_MASK_TABLE_STRIDE`]-entry slice with
/// `[v] = F128 { lo: v } * eq_lo` and `[256 + v] = F128 { lo: v << 8 } * eq_lo`,
/// built once per prove by [`super::build_c_mask_tables`]. **Zero multiplies in
/// the hot loop.**
///
/// The banks are **α-free** and carry **no `C_2`**: `α^K` and `C_2` are exactly
/// the small-eq weights that `low_eq = build_eq([r[k_skip+1], r[k_skip+2]])`
/// re-applies downstream — in `collapse_s_hat_v_quad` on intake *and* in
/// `build_direct_fold2_table` at materialize time. Folding either constant in
/// here would double-count it. The wire `s_hat_v_c` re-applies them once, at
/// reduction time (`α^K` per bank, then `C_2 · α^{-b_3[0]}`).
///
/// Padding: with `n_b_med < 16` the missing `b_med` simply leave their mask bit
/// clear, which is the same field element the per-`b_med` form would have
/// accumulated (those windows are all-zero witness).
#[inline]
pub(super) fn accumulate_c_banks(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_tables: &[super::F128],
    partial_c: &mut [[super::F128; 64]; 8],
) {
    accumulate_c_banks_with_policy(c_block, n_b_med, mask_tables, partial_c, true);
}

#[inline]
pub(super) fn accumulate_c_banks_with_policy(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_tables: &[super::F128],
    partial_c: &mut [[super::F128; 64]; 8],
    drain4: bool,
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON; the fixed-size arrays cover
    // every load/store and the table halves bound every `u8`-scaled index.
    unsafe {
        aarch64::accumulate_c_banks(c_block, n_b_med, mask_tables, partial_c, drain4);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = drain4;
        accumulate_c_banks_scalar(c_block, n_b_med, mask_tables, partial_c);
    }
}

/// Thirty-two-bank C drain for the experimental direct-fold4 capture.
/// `q = b_med & 3` is retained alongside the incumbent eight small-bit banks;
/// the 16-entry table folds only `h = b_med >> 2`.
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn accumulate_c_fold4_banks(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_table: &[super::F128],
    partial_c: &mut [[super::F128; 64]; 32],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON; fixed-size arrays cover
    // every load/store and each mask-table index is a four-bit nibble.
    unsafe {
        aarch64::accumulate_c_fold4_banks(c_block, n_b_med, mask_table, partial_c);
    }

    #[cfg(not(target_arch = "aarch64"))]
    super::accumulate_c_fold4_banks_scalar(c_block, n_b_med, mask_table, partial_c);
}

/// Pair-fused Fold4 drain with the original 32-bank accumulator shape.  This
/// is the first A/B candidate: it halves drain traffic without changing the
/// 128-job schedule or allocating additional partials.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_c_fold4_pair_banks(
    c_block_even: &[u8; 16 * 64],
    n_b_med_even: usize,
    c_block_odd: &[u8; 16 * 64],
    n_b_med_odd: usize,
    pair_mask_table: &[super::F128; 256],
    partial_c: &mut [[super::F128; 64]; 32],
) {
    for q in 0..4 {
        let q_banks: &mut [[super::F128; 64]; 8] = (&mut partial_c[q * 8..(q + 1) * 8])
            .try_into()
            .expect("eight Fold4 banks per retained q group");
        accumulate_c_fold4_q_pair_banks(
            c_block_even,
            n_b_med_even,
            c_block_odd,
            n_b_med_odd,
            q,
            pair_mask_table,
            q_banks,
        );
    }
}

/// Four-block Fold4 drain for the monolithic 32-bank candidate. Two composed
/// pair tables cover blocks (0,1) and (2,3); the architecture kernel combines
/// their contributions before one accumulator read/modify/write.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_c_fold4_four_banks(
    c_blocks: [&[u8; 16 * 64]; 4],
    n_b_med: [usize; 4],
    pair_mask_tables: [&[super::F128; 256]; 2],
    partial_c: &mut [[super::F128; 64]; 32],
) {
    assert!(
        n_b_med.into_iter().all(|count| count <= 16),
        "Fold4 padding count exceeds 16 rows"
    );

    #[cfg(target_arch = "aarch64")]
    // SAFETY: fixed arrays cover all selected rows and both table indices are
    // assembled from two four-bit masks.
    unsafe {
        aarch64::accumulate_c_fold4_four_banks(c_blocks, n_b_med, pair_mask_tables, partial_c);
    }

    #[cfg(not(target_arch = "aarch64"))]
    for q in 0..4 {
        let q_banks: &mut [[super::F128; 64]; 8] = (&mut partial_c[q * 8..(q + 1) * 8])
            .try_into()
            .expect("eight Fold4 banks per retained q group");
        super::accumulate_c_fold4_q_pair_banks_scalar(
            c_blocks[0],
            n_b_med[0],
            c_blocks[1],
            n_b_med[1],
            q,
            pair_mask_tables[0],
            q_banks,
        );
        super::accumulate_c_fold4_q_pair_banks_scalar(
            c_blocks[2],
            n_b_med[2],
            c_blocks[3],
            n_b_med[3],
            q,
            pair_mask_tables[1],
            q_banks,
        );
    }
}

/// Pair-fused q-local Fold4 drain. The low and high nibbles come from two
/// adjacent low-coordinate witness blocks with independent padding bounds;
/// the 256-entry table contains their already-added field contributions.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate_c_fold4_q_pair_banks(
    c_block_even: &[u8; 16 * 64],
    n_b_med_even: usize,
    c_block_odd: &[u8; 16 * 64],
    n_b_med_odd: usize,
    q: usize,
    pair_mask_table: &[super::F128; 256],
    partial_c: &mut [[super::F128; 64]; 8],
) {
    assert!(
        n_b_med_even <= 16,
        "even Fold4 padding count exceeds 16 rows"
    );
    assert!(n_b_med_odd <= 16, "odd Fold4 padding count exceeds 16 rows");
    assert!(q < 4, "Fold4 retained-medium group must be in 0..4");
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON; both fixed arrays cover
    // every selected row, q is checked by the kernel, and the two four-bit
    // masks form a bounded eight-bit table index.
    unsafe {
        aarch64::accumulate_c_fold4_q_pair_banks(
            c_block_even,
            n_b_med_even,
            c_block_odd,
            n_b_med_odd,
            q,
            pair_mask_table,
            partial_c,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    super::accumulate_c_fold4_q_pair_banks_scalar(
        c_block_even,
        n_b_med_even,
        c_block_odd,
        n_b_med_odd,
        q,
        pair_mask_table,
        partial_c,
    );
}

/// AB-only half of [`accumulate_convert_with_s_hat_v`].  Keeping this as a
/// separate entry point lets the ranked precomputed-AB path finish a whole
/// `x_hi` band with the 64 KiB convert table resident before switching to the
/// unrelated C-mask tables.
#[inline]
pub(super) fn accumulate_convert_ab(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    partial_ab: &mut [super::F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // every table-selected load.
    unsafe {
        aarch64::accumulate_convert_ab(chunk_ab_bytes, n_b_med, convert, eq_lo_val, partial_ab);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the SIMD features and the fixed arrays
    // cover every four-lane load/store.
    unsafe {
        x86_64::accumulate_convert_ab_x86_avx512(
            chunk_ab_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
        );
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )
    )))]
    portable::accumulate_convert_ab(chunk_ab_bytes, n_b_med, convert, eq_lo_val, partial_ab);
}

/// Multiply-free twin of [`accumulate_convert_ab`], for the banked `eq_lo`
/// tensor fold.
///
/// `convert` is the pre-scaled table `T_w[i] = convert[i] * eq_top[w]` for the
/// current chunk's high `x_lo` bits, and `bank` is the accumulator for its low
/// `x_lo` bits; the caller applies `eq_bot[u]` to each bank once per `x_hi`
/// band. Same gathers as the incumbent, zero GHASH multiplies.
#[inline]
pub(super) fn accumulate_convert_ab_nomul(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    bank: &mut [super::F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // every table-selected load and every bank slot drained.
    unsafe {
        aarch64::accumulate_convert_ab_nomul(chunk_ab_bytes, n_b_med, convert, bank);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the SIMD features and the fixed arrays
    // cover every four-lane load/store.
    unsafe {
        x86_64::accumulate_convert_ab_nomul_x86_avx512(chunk_ab_bytes, n_b_med, convert, bank);
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )
    )))]
    portable::accumulate_convert_ab_nomul(chunk_ab_bytes, n_b_med, convert, bank);
}

/// Portable reference for [`accumulate_c_banks`], and the shape the aarch64
/// kernel is tested against.
///
/// Deliberately written as the **composition** the fused kernel replaces —
/// per-`b_med` [`bit_transpose_64bytes`] into a scratch row, then the
/// cross-`b_med` mask scatter — so the differential test compares two
/// genuinely different routes to the same bytes rather than two spellings of
/// one route.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(super) fn accumulate_c_banks_scalar(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_tables: &[super::F128],
    partial_c: &mut [[super::F128; 64]; 8],
) {
    debug_assert!(n_b_med <= 16);
    debug_assert_eq!(mask_tables.len(), C_MASK_TABLE_STRIDE);
    let (t_lo, t_hi) = mask_tables.split_at(256);

    let mut transposed = [[0u8; 64]; 16];
    for b_med in 0..n_b_med {
        let row: &[u8; 64] = c_block[b_med * 64..(b_med + 1) * 64]
            .try_into()
            .expect("64 c-bytes per medium position");
        bit_transpose_64bytes(row, &mut transposed[b_med]);
    }

    for lane in 0..64 {
        let mut masks = [0u16; 8];
        for (b_med, row) in transposed.iter().enumerate().take(n_b_med) {
            let c = row[lane];
            for (s, mask) in masks.iter_mut().enumerate() {
                *mask |= u16::from((c >> s) & 1) << b_med;
            }
        }
        for (bank, mask) in partial_c.iter_mut().zip(masks) {
            bank[lane] += t_lo[usize::from(mask & 0xff)] + t_hi[usize::from(mask >> 8)];
        }
    }
}

#[inline]
pub(super) fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    mask_tables: &[super::F128],
    partial_ab: &mut [super::F128; 64],
    partial_c: &mut [[super::F128; 64]; 8],
) {
    accumulate_convert_ab(chunk_ab_bytes, n_b_med, convert, eq_lo_val, partial_ab);
    accumulate_c_banks(c_block, n_b_med, mask_tables, partial_c);
}

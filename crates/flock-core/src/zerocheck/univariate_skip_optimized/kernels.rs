use super::{C_MASK_TABLE_STRIDE, InvNttTableByteSingleGf8, F8};

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
pub(super) fn shift_reduce_inner_ab(
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
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (a_col, b_col);
        aarch64::shift_reduce_inner_ab_fused_neon_checked(
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

/// Preconvert one outer block's AB medium rows into 64 F128 lane values
/// (fixed medium-coordinate conversion, challenge-independent). NEON uses
/// the same 4-lane paired-table EOR3 schedule as [`accumulate_convert_ab`]
/// with a storing drain; the portable form is the scalar sum the oracle
/// tests pin.
#[inline]
pub(super) fn preconvert_ab_block(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    out: &mut [super::F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // every table-selected load.
    unsafe {
        aarch64::preconvert_ab_block(chunk_ab_bytes, n_b_med, convert, out);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        out.fill(super::F128::ZERO);
        for b_med in 0..n_b_med {
            let table = &convert[b_med * 256..(b_med + 1) * 256];
            let src = &chunk_ab_bytes[b_med];
            for lane in 0..64 {
                out[lane] += table[src[lane] as usize];
            }
        }
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
    accumulate_c_banks_with_drain_lanes(
        c_block,
        n_b_med,
        mask_tables,
        partial_c,
        if drain4 { 4 } else { 1 },
    );
}

#[inline]
pub(super) fn accumulate_c_banks_with_drain_lanes(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_tables: &[super::F128],
    partial_c: &mut [[super::F128; 64]; 8],
    drain_lanes: usize,
) {
    debug_assert!(matches!(drain_lanes, 1 | 4 | 8));

    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON; the fixed-size arrays cover
    // every load/store and the table halves bound every `u8`-scaled index.
    unsafe {
        aarch64::accumulate_c_banks(
            c_block,
            n_b_med,
            mask_tables,
            partial_c,
            drain_lanes,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = drain_lanes;
        accumulate_c_banks_scalar(c_block, n_b_med, mask_tables, partial_c);
    }
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
        aarch64::accumulate_convert_ab(
            chunk_ab_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
        );
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

//! Compile-time-selected leaf kernels for the F128 additive NTT.
//!
//! Transform scheduling and cache-blocking policy stay in the parent module;
//! this module owns the architecture-specific operations on blocks of data.
//!
//! ## Dead end: q-form (shared-twiddle Karatsuba + Barrett) butterfly leaves
//!
//! A full rewrite of `butterfly_row_pair` / `butterfly_fused_2layer` (and the
//! seed row-group kernels) in the promoted zerocheck/open q-form — hoisted
//! `lo/hi/lo⊕hi` twiddle broadcasts, 6 Karatsuba PMULL per lane pair, EOR3
//! cross terms, per-lane Barrett `hi·0x87` reduction, `ldp/stp q` I/O, zero
//! GPR round-trips — was measured **18-23% SLOWER** than these portable
//! per-lane loops (ST and 10T, m=25 and m=29 shapes, `ntt_butterfly_probe`
//! paired A/B; e2e `[commit-timing] ntt` 57 → 68 ms). Reason: under
//! `-C target-cpu=native` LLVM already compiles the portable lane loop
//! (binius mul) to all-NEON with EOR3 — ~15 NEON-pipe ops + 2 transfer-unit
//! `fmov` per butterfly, i.e. already AT the 4-pipe issue floor (~3.9
//! cyc/butterfly measured). Karatsuba+Barrett needs the same 6 PMULL per mul
//! as binius (3+3 vs 4+2), and the SoA zip/ext glue ADDS ~1.5 NEON-pipe
//! ops/lane. The wave-4 q-form wins came from replacing GPR-mixed leaves and
//! ~26-op vectorised shift reductions; neither disease exists here. Do not
//! re-attempt without first cutting PMULL count below 6/mul or NEON glue
//! below the current form.

use crate::field::F128;

mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod x86_64;

#[inline]
pub(super) fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    debug_assert_eq!(top.len(), bot.len());

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features.
    unsafe {
        x86_64::butterfly_row_pair(top, bot, twiddle);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    portable::butterfly_row_pair(top, bot, twiddle);
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), d.len());

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features.
    unsafe {
        x86_64::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    portable::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
}

/// Process one fused-two-layer row group from a separate source buffer.
///
/// # Safety
/// The caller must ensure the four selected source rows are valid, the four
/// selected destination rows are valid, and concurrent calls write disjoint
/// destination row groups. Source and destination must not overlap.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 3],
) {
    // SAFETY: forwarded caller contract.
    unsafe { portable::butterfly_fused_2layer_row_from(src, dst, quarter, num_ntts, r, twiddles) }
}

/// Process the sparse-twiddle first output block of the rate-1/2 layer-2 seed.
///
/// Its layer-1 and left layer-2 twiddles are zero; `right_twiddle` is the only
/// non-zero tree value.
///
/// # Safety
/// Same source/destination validity, non-aliasing, and disjoint-write contract
/// as [`butterfly_fused_2layer_row_from`].
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    right_twiddle: F128,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_2layer_row_from_sparse(
            src,
            dst,
            quarter,
            num_ntts,
            r,
            right_twiddle,
        )
    }
}

/// Process one fused-four-layer row group across every interleaved NTT lane.
///
/// # Safety
/// The caller must ensure the 16 row slices selected by `r` are valid and
/// disjoint from any row group being processed concurrently.
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_row(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: target features are guaranteed by cfg; the caller owns the row
    // geometry and disjointness contract.
    unsafe {
        x86_64::butterfly_fused_4layer_row(ptr, sixteenth, num_ntts, r, twiddles);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_4layer_row(ptr, sixteenth, num_ntts, r, twiddles);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe { aarch64::butterfly_block(chunk, twiddle, half) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block_pair(
    data: &mut [F128],
    base: usize,
    t_a: F128,
    t_b: F128,
) {
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe { aarch64::butterfly_block_pair(&mut data[base..base + 4], t_a, t_b) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block_pair_chunk(chunk: &mut [F128], t_a: F128, t_b: F128) {
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe { aarch64::butterfly_block_pair(chunk, t_a, t_b) }
}

/// Largest interleaved-lane count [`seed_fused_2layer_row_group_nt`] accepts.
/// Bounds the stack staging block at 8 rows × 64 lanes × 16 B = 8 KiB.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(super) const SEED_NT_MAX_NTTS: usize = 64;

/// Process one rate-1/2 seed row group (both codeword halves) through an
/// 8-row stack staging block, publishing each output row with q-form `stnp`
/// non-temporal pairs. Byte-identical to calling
/// [`butterfly_fused_2layer_row_from_sparse`] then
/// [`butterfly_fused_2layer_row_from`] on the two halves.
///
/// # Safety
/// Same source/destination validity, non-aliasing, and disjoint-write
/// contract as the unstaged pair; additionally `num_ntts` must be a multiple
/// of 8 and at most [`SEED_NT_MAX_NTTS`], and both codeword halves must start
/// 128-byte aligned.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn seed_fused_2layer_row_group_nt(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    half_len: usize,
    r: usize,
    right_twiddle: F128,
    twiddles: &[F128; 3],
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        aarch64::seed_fused_2layer_row_group_nt(
            src,
            dst,
            quarter,
            num_ntts,
            half_len,
            r,
            right_twiddle,
            twiddles,
        )
    }
}

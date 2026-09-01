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

#[allow(dead_code)] // Portable fallbacks remain available for rollback builds.
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
    // A twiddle whose high limb is zero kills both limb products that involve
    // it, and the reduction step that folds them. Dispatch once per row pair,
    // outside the lane loop; `low_twiddle_disabled()` restores the general
    // kernel for a same-binary A/B.
    let low = twiddle.hi == 0 && !low_twiddle_disabled();

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features, and `low`
    // is exactly the zero-high-limb precondition.
    unsafe {
        if low {
            x86_64::butterfly_row_pair_gen::<true>(top, bot, twiddle);
        } else {
            x86_64::butterfly_row_pair_gen::<false>(top, bot, twiddle);
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    if low {
        portable::butterfly_row_pair_gen::<true>(top, bot, twiddle);
    } else {
        portable::butterfly_row_pair_gen::<false>(top, bot, twiddle);
    }
}

/// `FLOCK_NO_LOW_TWIDDLE=1` restores the general field-multiply kernel in the
/// same binary, so a candidate/control pair differs only in this dispatch.
#[inline]
fn low_twiddle_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_LOW_TWIDDLE").is_some())
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

    // The two layers carry independent twiddles, so specialize them
    // independently: the pair is dispatched once here, never per lane.
    let off = low_twiddle_disabled();
    let outer_low = t_outer.hi == 0 && !off;
    let inner_low = t_inner_a.hi == 0 && t_inner_b.hi == 0 && !off;

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features, and the
    // flags above are exactly the zero-high-limb preconditions.
    unsafe {
        match (outer_low, inner_low) {
            (false, false) => x86_64::butterfly_fused_2layer_gen::<false, false>(
                a, b, c, d, t_outer, t_inner_a, t_inner_b,
            ),
            (false, true) => x86_64::butterfly_fused_2layer_gen::<false, true>(
                a, b, c, d, t_outer, t_inner_a, t_inner_b,
            ),
            (true, false) => x86_64::butterfly_fused_2layer_gen::<true, false>(
                a, b, c, d, t_outer, t_inner_a, t_inner_b,
            ),
            (true, true) => x86_64::butterfly_fused_2layer_gen::<true, true>(
                a, b, c, d, t_outer, t_inner_a, t_inner_b,
            ),
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    match (outer_low, inner_low) {
        (false, false) => portable::butterfly_fused_2layer_gen::<false, false>(
            a, b, c, d, t_outer, t_inner_a, t_inner_b,
        ),
        (false, true) => portable::butterfly_fused_2layer_gen::<false, true>(
            a, b, c, d, t_outer, t_inner_a, t_inner_b,
        ),
        (true, false) => portable::butterfly_fused_2layer_gen::<true, false>(
            a, b, c, d, t_outer, t_inner_a, t_inner_b,
        ),
        (true, true) => portable::butterfly_fused_2layer_gen::<true, true>(
            a, b, c, d, t_outer, t_inner_a, t_inner_b,
        ),
    }
}

/// Final fused-two-layer butterfly whose four rows are published directly to
/// four non-temporal codeword destinations instead of written back to scratch.
///
/// This is deliberately x86 AVX-512-only: the caller retains the incumbent
/// in-place butterfly plus scatter on every other target and geometry.
///
/// # Safety
///
/// `src` must expose four readable 64-element rows at `src + i * src_step`.
/// `dst_a` through `dst_d` must each expose 64 writable, 16-byte-aligned
/// elements, be mutually disjoint, and not overlap `src`. `lanes` must be 60
/// or 64. When `ALIGNED_ZMM`, every destination must be 64-byte aligned; the
/// fallback only requires 16-byte alignment. The cfg gate guarantees the
/// required target features.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_publish_nt<const ALIGNED_ZMM: bool>(
    src: *const F128,
    src_step: usize,
    dst_a: *mut F128,
    dst_b: *mut F128,
    dst_c: *mut F128,
    dst_d: *mut F128,
    lanes: usize,
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    debug_assert!(lanes == 60 || lanes == 64);
    debug_assert_eq!(dst_a as usize % 16, 0);
    debug_assert_eq!(dst_b as usize % 16, 0);
    debug_assert_eq!(dst_c as usize % 16, 0);
    debug_assert_eq!(dst_d as usize % 16, 0);
    debug_assert!(!ALIGNED_ZMM || dst_a as usize % 64 == 0);
    debug_assert!(!ALIGNED_ZMM || dst_b as usize % 64 == 0);
    debug_assert!(!ALIGNED_ZMM || dst_c as usize % 64 == 0);
    debug_assert!(!ALIGNED_ZMM || dst_d as usize % 64 == 0);

    // Match `butterfly_fused_2layer` exactly: both low-twiddle decisions are
    // made once outside the lane loop, and the x86 leaf makes the same
    // process-cached mul-diet choice.
    let off = low_twiddle_disabled();
    let outer_low = t_outer.hi == 0 && !off;
    let inner_low = t_inner_a.hi == 0 && t_inner_b.hi == 0 && !off;

    // SAFETY: forwarded caller contract; the flags are exactly the
    // zero-high-limb preconditions of the selected specializations.
    unsafe {
        match (outer_low, inner_low) {
            (false, false) => {
                x86_64::butterfly_fused_2layer_publish_nt_gen::<false, false, ALIGNED_ZMM>(
                    src, src_step, dst_a, dst_b, dst_c, dst_d, lanes, t_outer, t_inner_a, t_inner_b,
                )
            }
            (false, true) => {
                x86_64::butterfly_fused_2layer_publish_nt_gen::<false, true, ALIGNED_ZMM>(
                    src, src_step, dst_a, dst_b, dst_c, dst_d, lanes, t_outer, t_inner_a, t_inner_b,
                )
            }
            (true, false) => {
                x86_64::butterfly_fused_2layer_publish_nt_gen::<true, false, ALIGNED_ZMM>(
                    src, src_step, dst_a, dst_b, dst_c, dst_d, lanes, t_outer, t_inner_a, t_inner_b,
                )
            }
            (true, true) => {
                x86_64::butterfly_fused_2layer_publish_nt_gen::<true, true, ALIGNED_ZMM>(
                    src, src_step, dst_a, dst_b, dst_c, dst_d, lanes, t_outer, t_inner_a, t_inner_b,
                )
            }
        }
    }
}

/// Process one fused-two-layer row group from a separate source buffer.
///
/// # Safety
/// The caller must ensure the four selected source rows are valid, the four
/// selected destination rows are valid, and concurrent calls write disjoint
/// destination row groups. Source and destination must not overlap.
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 3],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: cfg gate guarantees the required target features.
    unsafe {
        x86_64::butterfly_fused_2layer_row_from(src, dst, quarter, num_ntts, r, twiddles);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_2layer_row_from(src, dst, quarter, num_ntts, r, twiddles);
    }
}

/// NT-publish twin of [`butterfly_fused_2layer_row_from`]: XMM `MOVNTDQ`
/// dest stores. x86 AVX-512 only; dest 16-byte aligned, `num_ntts` a
/// multiple of 4.
///
/// # Safety
/// Same as [`butterfly_fused_2layer_row_from`], plus the NT constraints.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from_nt(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 3],
) {
    // SAFETY: forwarded; identical src/dst geometry.
    unsafe {
        x86_64::butterfly_fused_2layer_row_from_geo_nt(
            src, quarter, r, dst, quarter, r, num_ntts, twiddles,
        );
    }
}

/// [`butterfly_fused_2layer_row_from`] with independent source/destination
/// row geometry (source rows `i·src_quarter + src_r`, destination rows
/// `i·dst_quarter + dst_r`).
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    twiddles: &[F128; 3],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: cfg gate guarantees the required target features.
    unsafe {
        x86_64::butterfly_fused_2layer_row_from_geo(
            src,
            src_quarter,
            src_r,
            dst,
            dst_quarter,
            dst_r,
            num_ntts,
            twiddles,
        );
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_2layer_row_from_geo(
            src,
            src_quarter,
            src_r,
            dst,
            dst_quarter,
            dst_r,
            num_ntts,
            twiddles,
        );
    }
}

/// [`butterfly_fused_2layer_row_from_sparse`] with independent
/// source/destination row geometry.
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: cfg gate guarantees the required target features.
    unsafe {
        x86_64::butterfly_fused_2layer_row_from_sparse_geo(
            src,
            src_quarter,
            src_r,
            dst,
            dst_quarter,
            dst_r,
            num_ntts,
            right_twiddle,
        );
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_2layer_row_from_sparse_geo(
            src,
            src_quarter,
            src_r,
            dst,
            dst_quarter,
            dst_r,
            num_ntts,
            right_twiddle,
        )
    }
}

/// [`butterfly_fused_2layer_row_from_sparse_geo`] that also asks for one line
/// of each of the four rows starting at `pf_src` on every lane step. Portable
/// builds ignore the hints.
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from_sparse_geo`]; the four
/// rows `pf_src + i * src_quarter * num_ntts` must also lie inside the source
/// buffer.
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_geo_pf(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
    pf_src: *const F128,
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: cfg gate guarantees the required target features.
    unsafe {
        x86_64::butterfly_fused_2layer_row_from_sparse_geo_pf(
            src,
            src_quarter,
            src_r,
            dst,
            dst_quarter,
            dst_r,
            num_ntts,
            right_twiddle,
            pf_src,
        );
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        let _ = pf_src;
        portable::butterfly_fused_2layer_row_from_sparse_geo(
            src,
            src_quarter,
            src_r,
            dst,
            dst_quarter,
            dst_r,
            num_ntts,
            right_twiddle,
        )
    }
}

/// One four-row message load, both seed staging groups. x86 AVX-512 holds the
/// four ZMMs; other builds run the two-call form (same bytes, two gathers).
///
/// # Safety
/// Union of the sparse-geo and dense-geo contracts on the shared source and
/// the two destinations. Destinations must not alias.
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[allow(clippy::too_many_arguments)]
#[inline]
#[allow(dead_code)] // Retained fused-kernel rollback/oracle entry point.
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_dense_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst_sparse: *mut F128,
    dst_dense: *mut F128,
    dst_quarter: usize,
    num_ntts: usize,
    right_twiddle: F128,
    dense_tw: &[F128; 3],
    pf_src: *const F128,
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        x86_64::butterfly_fused_2layer_row_from_sparse_dense_geo(
            src,
            src_quarter,
            src_r,
            dst_sparse,
            dst_dense,
            dst_quarter,
            num_ntts,
            right_twiddle,
            dense_tw,
            pf_src,
        );
        return;
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    unsafe {
        let _ = pf_src;
        portable::butterfly_fused_2layer_row_from_sparse_geo(
            src,
            src_quarter,
            src_r,
            dst_sparse,
            dst_quarter,
            0,
            num_ntts,
            right_twiddle,
        );
        portable::butterfly_fused_2layer_row_from_geo(
            src,
            src_quarter,
            src_r,
            dst_dense,
            dst_quarter,
            0,
            num_ntts,
            dense_tw,
        );
    }
}

/// Process the sparse-twiddle first output block of the rate-1/2 layer-2 seed.
///
/// Its layer-1 and left layer-2 twiddles are zero; `right_twiddle` is the only
/// non-zero tree value.
///
/// # Safety
/// Same source/destination validity, non-aliasing, and disjoint-write contract
/// as [`butterfly_fused_2layer_row_from`].
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    right_twiddle: F128,
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: cfg gate guarantees the required target features.
    unsafe {
        x86_64::butterfly_fused_2layer_row_from_sparse(
            src,
            dst,
            quarter,
            num_ntts,
            r,
            right_twiddle,
        );
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_2layer_row_from_sparse(
            src,
            dst,
            quarter,
            num_ntts,
            r,
            right_twiddle,
        );
    }
}

/// NT-publish twin of [`butterfly_fused_2layer_row_from_sparse`].
///
/// # Safety
/// Same as [`butterfly_fused_2layer_row_from_nt`].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_nt(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    right_twiddle: F128,
) {
    // SAFETY: forwarded; identical src/dst geometry.
    unsafe {
        x86_64::butterfly_fused_2layer_row_from_sparse_geo_nt(
            src,
            quarter,
            r,
            dst,
            quarter,
            r,
            num_ntts,
            right_twiddle,
        );
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
    lanes: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    debug_assert!(lanes <= num_ntts);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: target features are guaranteed by cfg; the caller owns the row
    // geometry and disjointness contract. The shaped arms only fire when the
    // constants equal the runtime shape, so they are value-identical.
    unsafe {
        if super::ntt_shaped_enabled() && num_ntts == 64 {
            match sixteenth {
                128 => {
                    return x86_64::butterfly_fused_4layer_row_shaped::<128, 64, 0>(
                        ptr, lanes, r, twiddles, 0,
                    );
                }
                8 => {
                    return x86_64::butterfly_fused_4layer_row_shaped::<8, 64, 0>(
                        ptr, lanes, r, twiddles, 0,
                    );
                }
                1 => {
                    return x86_64::butterfly_fused_4layer_row_shaped::<1, 64, 0>(
                        ptr, lanes, r, twiddles, 0,
                    );
                }
                _ => {}
            }
        }
        x86_64::butterfly_fused_4layer_row(ptr, sixteenth, num_ntts, lanes, r, twiddles);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_4layer_row(ptr, sixteenth, num_ntts, lanes, r, twiddles);
    }
}

/// [`butterfly_fused_4layer_row`] with one line hint per row of row group
/// `pf_r` issued at every lane step. `H` selects the hint level (1 = L1,
/// 2 = L2). Portable builds ignore the hint.
///
/// # Safety
/// Same contract as [`butterfly_fused_4layer_row`]; row group `pf_r` must
/// also lie inside the block.
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_row_pf<const H: u8>(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    lanes: usize,
    r: usize,
    twiddles: &[F128; 15],
    pf_r: usize,
) {
    debug_assert!(lanes <= num_ntts);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: target features are guaranteed by cfg; the caller owns the row
    // geometry and disjointness contract. The shaped arms only fire when the
    // constants equal the runtime shape, so they are value-identical.
    unsafe {
        if super::ntt_shaped_enabled() && num_ntts == 64 {
            match sixteenth {
                128 => {
                    return x86_64::butterfly_fused_4layer_row_shaped::<128, 64, H>(
                        ptr, lanes, r, twiddles, pf_r,
                    );
                }
                8 => {
                    return x86_64::butterfly_fused_4layer_row_shaped::<8, 64, H>(
                        ptr, lanes, r, twiddles, pf_r,
                    );
                }
                _ => {}
            }
        }
        x86_64::butterfly_fused_4layer_row_pf::<H>(
            ptr, sixteenth, num_ntts, lanes, r, twiddles, pf_r,
        );
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        let _ = pf_r;
        portable::butterfly_fused_4layer_row(ptr, sixteenth, num_ntts, lanes, r, twiddles);
    }
}

/// Process one fused-three-layer group of eight consecutive rows.
///
/// Rows `0..8` start at `ptr + i · num_ntts`. Lanes `0..dense_lanes` get the
/// full three-layer network; on lanes `dense_lanes..num_ntts` the group's odd
/// rows are known to be zero and the reduced network runs instead.
///
/// # Safety
/// The caller must ensure the eight rows are valid and disjoint from any row
/// group being processed concurrently, that `dense_lanes <= num_ntts`, and
/// that rows 1, 3, 5 and 7 hold zero on lanes `dense_lanes..num_ntts`.
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_rows(
    ptr: *mut F128,
    num_ntts: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    debug_assert!(dense_lanes <= num_ntts);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: target features are guaranteed by cfg; the caller owns the row
    // geometry, disjointness and zero-tail contract. The shaped arm only
    // fires when the constant equals the runtime shape (value-identical).
    unsafe {
        if super::ntt_shaped_enabled() && num_ntts == 64 {
            x86_64::butterfly_fused_3layer_rows_shaped::<64>(ptr, dense_lanes, twiddles);
            return;
        }
        x86_64::butterfly_fused_3layer_rows(ptr, num_ntts, dense_lanes, twiddles);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_3layer_rows(ptr, num_ntts, dense_lanes, twiddles);
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

#[cfg(test)]
mod low_twiddle_tests {
    use super::*;
    use crate::field::F128;

    fn rng(seed: u64) -> impl FnMut() -> F128 {
        let mut st = seed | 1;
        move || {
            let mut n = || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st
            };
            F128 { lo: n(), hi: n() }
        }
    }

    /// The scalar low-multiplier path must equal the general product for
    /// every multiplier with a zero high limb, including zero and one.
    #[test]
    fn mul_low_rhs_matches_general() {
        let mut next = rng(0xA11CE_5EED);
        for _ in 0..4096 {
            let a = next();
            let b = F128 {
                lo: next().lo,
                hi: 0,
            };
            assert_eq!(
                crate::field::gf2_128::mul_low_rhs(a, b),
                a * b,
                "a={a:?} b={b:?}"
            );
        }
        for a in [F128::ZERO, F128::ONE, next()] {
            for b in [
                F128::ZERO,
                F128::ONE,
                F128 {
                    lo: u64::MAX,
                    hi: 0,
                },
            ] {
                assert_eq!(crate::field::gf2_128::mul_low_rhs(a, b), a * b);
            }
        }
    }

    /// The specialized row-pair kernel must be bit-identical to the general
    /// one on any low twiddle, through whichever backend this host dispatches.
    #[test]
    fn row_pair_low_matches_general() {
        let mut next = rng(0xB0B_5EED);
        for len in [1usize, 3, 4, 7, 8, 64] {
            let twiddle = F128 {
                lo: next().lo,
                hi: 0,
            };
            let top: Vec<F128> = (0..len).map(|_| next()).collect();
            let bot: Vec<F128> = (0..len).map(|_| next()).collect();

            let (mut t1, mut b1) = (top.clone(), bot.clone());
            let (mut t2, mut b2) = (top.clone(), bot.clone());

            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            // SAFETY: cfg supplies the features; the twiddle's high limb is 0.
            unsafe {
                x86_64::butterfly_row_pair_gen::<true>(&mut t1, &mut b1, twiddle);
                x86_64::butterfly_row_pair_gen::<false>(&mut t2, &mut b2, twiddle);
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            {
                portable::butterfly_row_pair_gen::<true>(&mut t1, &mut b1, twiddle);
                portable::butterfly_row_pair_gen::<false>(&mut t2, &mut b2, twiddle);
            }
            assert_eq!(t1, t2, "len={len}");
            assert_eq!(b1, b2, "len={len}");
        }
    }

    /// Same for the fused pair, across all four low/general monomorphizations.
    #[test]
    fn fused_2layer_low_matches_general() {
        let mut next = rng(0xC0FFEE_5EED);
        for len in [1usize, 4, 5, 64] {
            let t_outer = F128 {
                lo: next().lo,
                hi: 0,
            };
            let t_a = F128 {
                lo: next().lo,
                hi: 0,
            };
            let t_b = F128 {
                lo: next().lo,
                hi: 0,
            };
            let rows: Vec<Vec<F128>> = (0..4).map(|_| (0..len).map(|_| next()).collect()).collect();

            let run = |outer_low: bool, inner_low: bool| -> Vec<Vec<F128>> {
                let mut r: Vec<Vec<F128>> = rows.clone();
                let (a, rest) = r.split_at_mut(1);
                let (b, rest) = rest.split_at_mut(1);
                let (c, d) = rest.split_at_mut(1);
                let (a, b, c, d) = (&mut a[0], &mut b[0], &mut c[0], &mut d[0]);
                macro_rules! call {
                    ($o:expr, $i:expr) => {{
                        #[cfg(all(
                            target_arch = "x86_64",
                            target_feature = "avx512f",
                            target_feature = "vpclmulqdq"
                        ))]
                        // SAFETY: cfg supplies features; both twiddle sets
                        // have zero high limbs.
                        unsafe {
                            x86_64::butterfly_fused_2layer_gen::<$o, $i>(
                                a, b, c, d, t_outer, t_a, t_b,
                            )
                        }
                        #[cfg(not(all(
                            target_arch = "x86_64",
                            target_feature = "avx512f",
                            target_feature = "vpclmulqdq"
                        )))]
                        portable::butterfly_fused_2layer_gen::<$o, $i>(
                            a, b, c, d, t_outer, t_a, t_b,
                        )
                    }};
                }
                match (outer_low, inner_low) {
                    (false, false) => call!(false, false),
                    (false, true) => call!(false, true),
                    (true, false) => call!(true, false),
                    (true, true) => call!(true, true),
                }
                r
            };

            let reference = run(false, false);
            for (o, i) in [(false, true), (true, false), (true, true)] {
                assert_eq!(
                    run(o, i),
                    reference,
                    "len={len} outer_low={o} inner_low={i}"
                );
            }
        }
    }
}

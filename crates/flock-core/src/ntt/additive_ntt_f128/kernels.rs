//! Compile-time-selected leaf kernels for the F128 additive NTT.
//!
//! Transform scheduling and cache-blocking policy stay in the parent module;
//! this module owns the architecture-specific operations on blocks of data.

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

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate supplies `aes`; slice lengths are asserted above.
    unsafe {
        if vector_resident_rows() {
            aarch64::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
            return;
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    portable::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
}

/// AArch64 block form of [`butterfly_fused_2layer`] that retains the three
/// twiddles across all row groups. Returns `false` when the incumbent row
/// dispatcher should be used instead.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn try_butterfly_fused_2layer_rows(
    block: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    quarter: usize,
    num_ntts: usize,
    odd_tail: usize,
) -> bool {
    debug_assert!(quarter > 0);
    debug_assert!(num_ntts > 0);
    debug_assert!(odd_tail < num_ntts);
    debug_assert!(odd_tail == 0 || quarter.is_multiple_of(2));
    debug_assert_eq!(block.len(), 4 * quarter * num_ntts);

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate supplies `aes`; the block geometry is established
    // by the deep fused-pair scheduler and asserted above.
    unsafe {
        if vector_resident_rows() {
            aarch64::butterfly_fused_2layer_rows(
                block, t_outer, t_inner_a, t_inner_b, quarter, num_ntts, odd_tail,
            );
            return true;
        }
    }

    false
}

/// AArch64 specialization for a fused pair whose three twiddles all have a
/// zero high limb. Other targets retain the ordinary field-multiply kernel.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn butterfly_fused_2layer_low_twiddles(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    debug_assert_eq!(t_outer.hi, 0);
    debug_assert_eq!(t_inner_a.hi, 0);
    debug_assert_eq!(t_inner_b.hi, 0);

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate guarantees PMULL through the aes feature.
    unsafe {
        aarch64::butterfly_fused_2layer_low_twiddles(a, b, c, d, t_outer, t_inner_a, t_inner_b);
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    portable::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
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

/// Process one fused-three-layer row group across every interleaved NTT lane.
///
/// # Safety
/// The caller must ensure the 8 row slices selected by `r` are valid and
/// disjoint from any row group being processed concurrently.
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    lanes: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: forwarded caller contract; the cfg gate supplies `aes`.
    unsafe {
        if vector_resident_rows() {
            aarch64::butterfly_fused_3layer_row(ptr, eighth, num_ntts, lanes, r, twiddles);
            return;
        }
    }

    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_3layer_row(ptr, eighth, num_ntts, lanes, r, twiddles);
    }
}

#[inline]
pub(super) unsafe fn butterfly_fused_3layer_rows(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    row_start: usize,
    row_end: usize,
    twiddles: &[F128; 7],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        if vector_resident_rows() {
            aarch64::butterfly_fused_3layer_rows(
                ptr, eighth, num_ntts, row_start, row_end, twiddles,
            );
            return;
        }
    }
    unsafe {
        for r in row_start..row_end {
            portable::butterfly_fused_3layer_row(ptr, eighth, num_ntts, num_ntts, r, twiddles);
        }
    }
}

/// Whether the AArch64 vector-resident radix-8 row kernels are used.
///
/// `FLOCK_NO_NTT_NEON_ROWS=1` restores the portable `F128`-typed chain in the
/// same binary, so a candidate/control pair differs only in this dispatch —
/// not in compilation or source revision. Read once per process; the ranked
/// harness sets its environment before the worker starts.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
fn vector_resident_rows() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_NTT_NEON_ROWS").is_none())
}

/// Rate-1/2 first-pass row kernel: one load of the radix-8 row group from
/// `src`, two fused-3 evaluations — zero-root set to `dst0`, general set to
/// `dst1`. The AArch64 arm stages outputs in L1 and emits each destination
/// row as a sequential `stnp` burst; see that kernel's doc for why.
///
/// # Safety
/// Row/lane geometry valid for all three pointers, disjoint row groups
/// across concurrent calls, `num_ntts ≤ 64` even, and
/// `t_zero[0] == t_zero[1] == t_zero[3] == 0`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_3layer_dual_from_src_row(
    src: *const F128,
    dst0: *mut F128,
    dst1: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    t_zero: &[F128; 7],
    t_gen: &[F128; 7],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: forwarded caller contract; the cfg gate supplies `aes`.
    unsafe {
        if vector_resident_rows() {
            aarch64::butterfly_fused_3layer_dual_from_src_row(
                src, dst0, dst1, eighth, num_ntts, r, t_zero, t_gen,
            );
            return;
        }
    }

    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_3layer_dual_from_src_row(
            src, dst0, dst1, eighth, num_ntts, r, t_zero, t_gen,
        );
    }
}

/// Out-of-place fused-three-layer row for recursive from-message transforms.
///
/// # Safety
/// Row/lane geometry must be valid for both pointers and concurrent calls
/// must own disjoint destination row groups.
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_from_src_row(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: forwarded caller contract; the cfg gate supplies `aes`.
    unsafe {
        if vector_resident_rows() {
            aarch64::butterfly_fused_3layer_from_src_row(src, dst, eighth, num_ntts, r, twiddles);
            return;
        }
    }

    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_3layer_from_src_row(src, dst, eighth, num_ntts, r, twiddles);
    }
}

/// Zero-root specialization of [`butterfly_fused_3layer_from_src_row`].
///
/// # Safety
/// As the general form, plus `twiddles[0] == twiddles[1] == twiddles[3] == 0`.
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_zero_root_from_src_row(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: forwarded caller contract; the cfg gate supplies `aes`.
    unsafe {
        if vector_resident_rows() {
            aarch64::butterfly_fused_3layer_zero_root_from_src_row(
                src, dst, eighth, num_ntts, r, twiddles,
            );
            return;
        }
    }

    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_3layer_zero_root_from_src_row(
            src, dst, eighth, num_ntts, r, twiddles,
        );
    }
}

/// Root-block specialization of [`butterfly_fused_3layer_row`].
///
/// # Safety
/// In addition to the ordinary row-geometry contract, the caller must ensure
/// that twiddles 0, 1, and 3 are zero.
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_zero_root_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    lanes: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: forwarded caller contract; the cfg gate supplies `aes`.
    unsafe {
        if vector_resident_rows() {
            aarch64::butterfly_fused_3layer_zero_root_row(
                ptr, eighth, num_ntts, lanes, r, twiddles,
            );
            return;
        }
    }

    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_3layer_zero_root_row(ptr, eighth, num_ntts, lanes, r, twiddles);
    }
}

#[inline]
pub(super) unsafe fn butterfly_fused_3layer_zero_root_rows(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    row_start: usize,
    row_end: usize,
    twiddles: &[F128; 7],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        if vector_resident_rows() {
            aarch64::butterfly_fused_3layer_zero_root_rows(
                ptr, eighth, num_ntts, row_start, row_end, twiddles,
            );
            return;
        }
    }
    unsafe {
        for r in row_start..row_end {
            portable::butterfly_fused_3layer_zero_root_row(
                ptr, eighth, num_ntts, num_ntts, r, twiddles,
            );
        }
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

#[cfg(all(test, target_arch = "aarch64", target_feature = "aes"))]
mod aarch64_row_tests {
    use super::*;

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_f128(state: &mut u64) -> F128 {
        F128 {
            lo: splitmix(state),
            hi: splitmix(state),
        }
    }

    /// Build a random row-group buffer plus twiddles for one geometry.
    fn fixture(
        state: &mut u64,
        eighth: usize,
        num_ntts: usize,
        zero_root: bool,
    ) -> (Vec<F128>, [F128; 7]) {
        let buf: Vec<F128> = (0..8 * eighth * num_ntts)
            .map(|_| rand_f128(state))
            .collect();
        let mut tw: [F128; 7] = core::array::from_fn(|_| rand_f128(state));
        if zero_root {
            tw[0] = F128::ZERO;
            tw[1] = F128::ZERO;
            tw[3] = F128::ZERO;
        }
        (buf, tw)
    }

    /// The vector-resident radix-8 row kernel must be **bit-identical** to the
    /// portable `F128`-typed chain across the row geometries the ranked
    /// transform uses (`num_ntts = 64` is the ranked interleaving). Both run
    /// on identical random input; every word of all eight row streams is
    /// compared.
    #[test]
    fn neon_fused_3layer_row_matches_portable() {
        let mut state = 0x4E54_5F52_4F57_5300;
        for &(eighth, num_ntts) in &[(1usize, 1usize), (1, 64), (2, 64), (4, 16), (3, 5)] {
            for r in 0..eighth.min(2) {
                for lanes in [num_ntts, num_ntts.saturating_sub(7).max(1)] {
                    let (base, tw) = fixture(&mut state, eighth, num_ntts, false);

                    let mut want = base.clone();
                    // SAFETY: buffer is 8 * eighth * num_ntts long, r < eighth,
                    // lanes <= num_ntts.
                    unsafe {
                        portable::butterfly_fused_3layer_row(
                            want.as_mut_ptr(),
                            eighth,
                            num_ntts,
                            lanes,
                            r,
                            &tw,
                        );
                    }

                    let mut got = base.clone();
                    // SAFETY: same geometry; this module carries `aes` via cfg.
                    unsafe {
                        aarch64::butterfly_fused_3layer_row(
                            got.as_mut_ptr(),
                            eighth,
                            num_ntts,
                            lanes,
                            r,
                            &tw,
                        );
                    }

                    assert_eq!(
                        got, want,
                        "fused3 row mismatch at eighth={eighth} num_ntts={num_ntts} \
                         lanes={lanes} r={r}"
                    );
                    // Lanes beyond the processed prefix must be untouched.
                    for row in 0..8 {
                        for lane in lanes..num_ntts {
                            let idx = (row * eighth + r) * num_ntts + lane;
                            assert_eq!(got[idx], base[idx], "lane {lane} was written");
                        }
                    }
                }
            }
        }
    }

    /// Same equivalence for the zero-root spine specialization, including its
    /// claim that row stream zero is left untouched.
    #[test]
    fn neon_fused_3layer_zero_root_row_matches_portable() {
        let mut state = 0x5A45_524F_524F_4F54;
        for &(eighth, num_ntts) in &[(1usize, 1usize), (1, 64), (2, 64), (4, 16), (3, 5)] {
            for r in 0..eighth.min(2) {
                for lanes in [num_ntts, num_ntts.saturating_sub(7).max(1)] {
                    let (base, tw) = fixture(&mut state, eighth, num_ntts, true);

                    let mut want = base.clone();
                    // SAFETY: geometry as above; twiddles 0/1/3 are zero.
                    unsafe {
                        portable::butterfly_fused_3layer_zero_root_row(
                            want.as_mut_ptr(),
                            eighth,
                            num_ntts,
                            lanes,
                            r,
                            &tw,
                        );
                    }

                    let mut got = base.clone();
                    // SAFETY: geometry as above; twiddles 0/1/3 are zero.
                    unsafe {
                        aarch64::butterfly_fused_3layer_zero_root_row(
                            got.as_mut_ptr(),
                            eighth,
                            num_ntts,
                            lanes,
                            r,
                            &tw,
                        );
                    }

                    assert_eq!(
                        got, want,
                        "zero-root row mismatch at eighth={eighth} num_ntts={num_ntts} \
                         lanes={lanes} r={r}"
                    );
                    // Lanes beyond the processed prefix must be untouched.
                    for row in 0..8 {
                        for lane in lanes..num_ntts {
                            let idx = (row * eighth + r) * num_ntts + lane;
                            assert_eq!(got[idx], base[idx], "lane {lane} was written");
                        }
                    }
                }
            }
        }
    }

    /// The vector-resident deep fused-pair kernel must be **bit-identical**
    /// to the portable `F128`-typed chain across the row widths the ranked
    /// deep transform uses (`num_ntts = 64` rows) plus narrow/odd widths.
    #[test]
    fn neon_fused_2layer_matches_portable() {
        let mut state = 0x4645_5350_4149_5232;
        for &width in &[1usize, 2, 5, 16, 64, 96] {
            for _ in 0..4 {
                let mut mk =
                    |n: usize| -> Vec<F128> { (0..n).map(|_| rand_f128(&mut state)).collect() };
                let (a0, b0, c0, d0) = (mk(width), mk(width), mk(width), mk(width));
                let t_outer = rand_f128(&mut state);
                let t_inner_a = rand_f128(&mut state);
                let t_inner_b = rand_f128(&mut state);

                let (mut wa, mut wb, mut wc, mut wd) =
                    (a0.clone(), b0.clone(), c0.clone(), d0.clone());
                portable::butterfly_fused_2layer(
                    &mut wa, &mut wb, &mut wc, &mut wd, t_outer, t_inner_a, t_inner_b,
                );

                let (mut ga, mut gb, mut gc, mut gd) = (a0, b0, c0, d0);
                // SAFETY: this module carries `aes` via cfg; equal lengths.
                unsafe {
                    aarch64::butterfly_fused_2layer(
                        &mut ga, &mut gb, &mut gc, &mut gd, t_outer, t_inner_a, t_inner_b,
                    );
                }

                assert_eq!(ga, wa, "row a mismatch at width={width}");
                assert_eq!(gb, wb, "row b mismatch at width={width}");
                assert_eq!(gc, wc, "row c mismatch at width={width}");
                assert_eq!(gd, wd, "row d mismatch at width={width}");
            }
        }
    }
}

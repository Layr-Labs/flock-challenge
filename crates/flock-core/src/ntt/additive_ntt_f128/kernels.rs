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

    // Measured at ranked geometry: routing these butterflies through
    // `ghash_mul_vec2_neon` REGRESSED the commit encode phase by ~16% despite
    // using 4 PMULL per product instead of 6. The helper takes and returns
    // `[F128; 2]` by value through general-purpose registers, so each call
    // pays lane-extract and reinsert traffic around values the compiler was
    // otherwise keeping in q registers. That move traffic exceeds the two
    // saved PMULLs. Keep the portable path until a q-in/q-out formulation is
    // measured to win.
    portable::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
}

/// Apply two forward layers from four immutable source rows into four
/// disjoint destination rows. Source and destination must not overlap.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn butterfly_fused_2layer_out_of_place(
    src_a: &[F128],
    src_b: &[F128],
    src_c: &[F128],
    src_d: &[F128],
    dst_a: &mut [F128],
    dst_b: &mut [F128],
    dst_c: &mut [F128],
    dst_d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    debug_assert_eq!(src_a.len(), src_b.len());
    debug_assert_eq!(src_a.len(), src_c.len());
    debug_assert_eq!(src_a.len(), src_d.len());
    debug_assert_eq!(src_a.len(), dst_a.len());
    debug_assert_eq!(src_a.len(), dst_b.len());
    debug_assert_eq!(src_a.len(), dst_c.len());
    debug_assert_eq!(src_a.len(), dst_d.len());

    // See the note on `butterfly_fused_2layer`: the NEON vec2 formulation was
    // measured slower at ranked geometry.
    portable::butterfly_fused_2layer_out_of_place(
        src_a, src_b, src_c, src_d, dst_a, dst_b, dst_c, dst_d, t_outer, t_inner_a, t_inner_b,
    );
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
    r: usize,
    twiddles: &[F128; 7],
) {
    // See the note on `butterfly_fused_2layer`: the NEON vec2 formulation of
    // this radix-8 row was measured slower at ranked geometry.
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_3layer_row(ptr, eighth, num_ntts, r, twiddles);
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
mod neon_tests {
    use super::{aarch64, portable};
    use crate::field::F128;

    /// SplitMix64 PRNG, deterministic.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    #[test]
    fn neon_fused_3layer_row_matches_portable() {
        let mut rng = Rng::new(0x3_1AE_8);
        for (eighth, num_ntts) in [(1usize, 1usize), (1, 2), (3, 5), (2, 8), (5, 64)] {
            let src = rng.vec(8 * eighth * num_ntts);
            let twiddles: [F128; 7] = core::array::from_fn(|_| rng.f128());

            let mut want = src.clone();
            let mut got = src.clone();
            for r in 0..eighth {
                // SAFETY: `want`/`got` each hold exactly 8 * eighth * num_ntts
                // elements, so every `(i * eighth + r) * num_ntts + lane`
                // touched for i < 8, r < eighth, lane < num_ntts is in bounds.
                // The two buffers are distinct allocations.
                unsafe {
                    portable::butterfly_fused_3layer_row(
                        want.as_mut_ptr(),
                        eighth,
                        num_ntts,
                        r,
                        &twiddles,
                    );
                    aarch64::butterfly_fused_3layer_row(
                        got.as_mut_ptr(),
                        eighth,
                        num_ntts,
                        r,
                        &twiddles,
                    );
                }
            }
            assert_eq!(
                got, want,
                "radix-8 NEON diverged at eighth={eighth} num_ntts={num_ntts}"
            );
            assert_ne!(got, src, "test never transformed anything");
        }
    }

    #[test]
    fn neon_fused_2layer_matches_portable() {
        let mut rng = Rng::new(0x2_1AE_4);
        // Odd lane counts exercise the two-lane loop's scalar tail.
        for lanes in [1usize, 2, 3, 7, 8, 33] {
            let a = rng.vec(lanes);
            let b = rng.vec(lanes);
            let c = rng.vec(lanes);
            let d = rng.vec(lanes);
            let (t_outer, t_inner_a, t_inner_b) = (rng.f128(), rng.f128(), rng.f128());

            let (mut wa, mut wb, mut wc, mut wd) = (a.clone(), b.clone(), c.clone(), d.clone());
            portable::butterfly_fused_2layer(
                &mut wa, &mut wb, &mut wc, &mut wd, t_outer, t_inner_a, t_inner_b,
            );

            let (mut ga, mut gb, mut gc, mut gd) = (a.clone(), b.clone(), c.clone(), d.clone());
            // SAFETY: the module cfg gate guarantees the `aes` target feature.
            unsafe {
                aarch64::butterfly_fused_2layer(
                    &mut ga, &mut gb, &mut gc, &mut gd, t_outer, t_inner_a, t_inner_b,
                );
            }
            assert_eq!(
                (&ga, &gb, &gc, &gd),
                (&wa, &wb, &wc, &wd),
                "in-place lanes={lanes}"
            );
            assert_ne!(ga, a, "test never transformed anything");

            // The out-of-place portable kernel is the dispatcher's non-NEON
            // path, so it is the oracle for the out-of-place NEON kernel.
            let mut pa = vec![F128::ZERO; lanes];
            let mut pb = vec![F128::ZERO; lanes];
            let mut pc = vec![F128::ZERO; lanes];
            let mut pd = vec![F128::ZERO; lanes];
            portable::butterfly_fused_2layer_out_of_place(
                &a, &b, &c, &d, &mut pa, &mut pb, &mut pc, &mut pd, t_outer, t_inner_a, t_inner_b,
            );
            assert_eq!(
                (&pa, &pb, &pc, &pd),
                (&wa, &wb, &wc, &wd),
                "portable in-place/out-of-place disagree at lanes={lanes}"
            );

            let mut oa = vec![F128::ZERO; lanes];
            let mut ob = vec![F128::ZERO; lanes];
            let mut oc = vec![F128::ZERO; lanes];
            let mut od = vec![F128::ZERO; lanes];
            // SAFETY: the module cfg gate guarantees the `aes` target feature;
            // sources and destinations are distinct allocations of `lanes`.
            unsafe {
                aarch64::butterfly_fused_2layer_out_of_place(
                    &a, &b, &c, &d, &mut oa, &mut ob, &mut oc, &mut od, t_outer, t_inner_a,
                    t_inner_b,
                );
            }
            assert_eq!(
                (&oa, &ob, &oc, &od),
                (&pa, &pb, &pc, &pd),
                "out-of-place lanes={lanes}"
            );
        }
    }
}

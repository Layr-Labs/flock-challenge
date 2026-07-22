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

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    portable::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
}

/// Process one fused-three-layer row group across the interleaved lane prefix
/// `0..active_lanes`; the remaining lanes are left untouched.
///
/// # Safety
/// The caller must ensure the eight selected rows are valid and that
/// concurrent calls use disjoint row groups.
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
    active_lanes: usize,
) {
    debug_assert!(active_lanes <= num_ntts);
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    if twiddles[0].is_zero() && twiddles[1].is_zero() && twiddles[3].is_zero() {
        // Block zero of every fused pass has the seed's sparse tree shape.
        let sparse_twiddles = [twiddles[2], twiddles[4], twiddles[5], twiddles[6]];
        // SAFETY: the sparse kernel loads a lane's eight values before writing
        // them, so exact source/destination overlap preserves the contract.
        unsafe {
            portable::butterfly_fused_3layer_row_from_sparse(
                ptr,
                ptr,
                eighth,
                num_ntts,
                r,
                &sparse_twiddles,
                active_lanes,
            )
        }
        return;
    }

    #[cfg(all(
        target_arch = "aarch64",
        target_vendor = "apple",
        target_feature = "aes",
        target_feature = "sha3"
    ))]
    {
        // SAFETY: forwarded caller contract. The sparse block-zero shape was
        // dispatched above, so this leaf owns only the dense in-place graph.
        unsafe {
            aarch64::butterfly_fused_3layer_row_dense_qresident(
                ptr,
                eighth,
                num_ntts,
                r,
                twiddles,
                active_lanes,
            )
        }
        return;
    }

    #[cfg(not(all(
        target_arch = "aarch64",
        target_vendor = "apple",
        target_feature = "aes",
        target_feature = "sha3"
    )))]
    {
        // SAFETY: forwarded caller contract.
        unsafe {
            portable::butterfly_fused_3layer_row(ptr, eighth, num_ntts, r, twiddles, active_lanes)
        }
    }
}

/// Test/analysis oracle retained on the Q-resident target. Production dense
/// calls use the private leaf, while differential tests need the unchanged
/// portable graph in the same linked artifact.
#[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    target_feature = "aes",
    target_feature = "sha3"
))]
#[allow(dead_code)]
pub(super) unsafe fn butterfly_fused_3layer_row_portable_oracle(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
    active_lanes: usize,
) {
    unsafe {
        portable::butterfly_fused_3layer_row(ptr, eighth, num_ntts, r, twiddles, active_lanes)
    }
}

/// Process one fused-three-layer row group from a separate source buffer for
/// lanes `0..active_lanes`; the remaining destination lanes are untouched.
///
/// # Safety
/// The caller must ensure the eight selected source rows are valid, the eight
/// selected destination rows are valid, and concurrent calls write disjoint
/// destination row groups. Source and destination must not overlap.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row_from(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
    active_lanes: usize,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_3layer_row_from(
            src,
            dst,
            eighth,
            num_ntts,
            r,
            twiddles,
            active_lanes,
        )
    }
}

/// Process the sparse-twiddle first output block of the rate-1/2 seed.
///
/// Its layer-1 twiddle, left layer-2 twiddle, and left layer-3 twiddle are
/// zero. `twiddles` contains only the four remaining non-zero tree values;
/// only lanes `0..active_lanes` are written.
///
/// # Safety
/// Same source/destination validity, non-aliasing, and disjoint-write contract
/// as [`butterfly_fused_3layer_row_from`].
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row_from_sparse(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 4],
    active_lanes: usize,
) {
    // SAFETY: forwarded caller contract.
    unsafe {
        portable::butterfly_fused_3layer_row_from_sparse(
            src,
            dst,
            eighth,
            num_ntts,
            r,
            twiddles,
            active_lanes,
        )
    }
}

/// Zero an uncomputed lane suffix in the eight destination rows selected by
/// one fused-three-layer row group.
///
/// # Safety
/// The caller must provide valid destination geometry and exclusive access to
/// the selected row suffixes.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn zero_fused_3layer_row_tail(
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    dense_lanes: usize,
) {
    debug_assert!(dense_lanes <= num_ntts);
    unsafe {
        for i in 0..8 {
            std::ptr::write_bytes(
                dst.add((i * eighth + r) * num_ntts + dense_lanes),
                0,
                num_ntts - dense_lanes,
            );
        }
    }
}

/// Fused final three layers for an interleaved buffer with an exact zero
/// suffix on every odd input row.
///
/// # Safety
/// The caller must provide the single eight-row group at the deepest three
/// layers and exclusive access to it.
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row_final_odd_zero(
    ptr: *mut F128,
    num_ntts: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    debug_assert!(dense_lanes <= num_ntts);
    // SAFETY: forwarded caller contract. The dense prefix uses the unchanged
    // fused-3 kernel, including its sparse-twiddle dispatch.
    unsafe {
        butterfly_fused_3layer_row(ptr, 1, num_ntts, 0, twiddles, dense_lanes);
        portable::butterfly_fused_3layer_row_final_odd_zero_tail(
            ptr,
            num_ntts,
            dense_lanes,
            twiddles,
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

#[cfg(all(
    test,
    target_arch = "aarch64",
    target_vendor = "apple",
    target_feature = "aes",
    target_feature = "sha3"
))]
mod qresident_tests {
    use super::*;

    #[inline]
    fn next_u64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[inline]
    fn next_f128(state: &mut u64) -> F128 {
        F128 {
            lo: next_u64(state),
            hi: next_u64(state),
        }
    }

    fn compare_case(
        input: &[F128],
        eighth: usize,
        num_ntts: usize,
        r: usize,
        twiddles: &[F128; 7],
        active_lanes: usize,
    ) {
        let mut candidate = input.to_vec();
        let mut oracle = input.to_vec();
        unsafe {
            aarch64::butterfly_fused_3layer_row_dense_qresident(
                candidate.as_mut_ptr(),
                eighth,
                num_ntts,
                r,
                twiddles,
                active_lanes,
            );
            butterfly_fused_3layer_row_portable_oracle(
                oracle.as_mut_ptr(),
                eighth,
                num_ntts,
                r,
                twiddles,
                active_lanes,
            );
        }
        assert_eq!(
            candidate, oracle,
            "eighth={eighth} num_ntts={num_ntts} r={r} active_lanes={active_lanes}"
        );
    }

    #[test]
    fn qresident_dense_matches_portable_all_active_lanes_and_geometries() {
        let mut state = 0x243f_6a88_85a3_08d3;
        let twiddles = core::array::from_fn(|_| next_f128(&mut state));

        let eighth = 2;
        let num_ntts = 64;
        let input: Vec<_> = (0..8 * eighth * num_ntts)
            .map(|_| next_f128(&mut state))
            .collect();
        for active_lanes in 0..=num_ntts {
            compare_case(&input, eighth, num_ntts, 1, &twiddles, active_lanes);
        }

        for eighth in [1, 2, 4, 8, 16, 64] {
            let num_ntts = 64;
            let input: Vec<_> = (0..8 * eighth * num_ntts)
                .map(|_| next_f128(&mut state))
                .collect();
            let rows: Vec<_> = if eighth <= 8 {
                (0..eighth).collect()
            } else {
                vec![0, 1, eighth / 2, eighth - 1]
            };
            for r in rows {
                for active_lanes in [0, 1, 2, 57, 63, 64] {
                    compare_case(&input, eighth, num_ntts, r, &twiddles, active_lanes);
                }
            }
        }
    }

    #[test]
    fn qresident_dense_matches_portable_edge_and_one_hot_fields() {
        let mut state = 0x1319_8a2e_0370_7344;
        let dense_twiddles = core::array::from_fn(|_| next_f128(&mut state));

        for fill in [
            F128::ZERO,
            F128 {
                lo: u64::MAX,
                hi: u64::MAX,
            },
            F128 {
                lo: 0xaaaa_aaaa_aaaa_aaaa,
                hi: 0x5555_5555_5555_5555,
            },
        ] {
            compare_case(&[fill; 8], 1, 1, 0, &dense_twiddles, 1);
        }

        for value_index in 0..8 {
            for bit in 0..128 {
                let mut input = [F128::ZERO; 8];
                if bit < 64 {
                    input[value_index].lo = 1u64 << bit;
                } else {
                    input[value_index].hi = 1u64 << (bit - 64);
                }
                compare_case(&input, 1, 1, 0, &dense_twiddles, 1);
            }
        }

        let input: [F128; 8] = core::array::from_fn(|_| next_f128(&mut state));
        for twiddle_index in 0..7 {
            for bit in 0..128 {
                let mut twiddles = [F128::ZERO; 7];
                if bit < 64 {
                    twiddles[twiddle_index].lo = 1u64 << bit;
                } else {
                    twiddles[twiddle_index].hi = 1u64 << (bit - 64);
                }
                compare_case(&input, 1, 1, 0, &twiddles, 1);
            }
        }
    }

    #[test]
    fn qresident_dense_matches_portable_ten_thousand_random_cases() {
        let mut state = 0xa409_3822_299f_31d0;
        for case in 0..10_000 {
            let input: [F128; 8] = core::array::from_fn(|_| next_f128(&mut state));
            let mut twiddles = core::array::from_fn(|_| next_f128(&mut state));
            if case % 17 == 0 {
                twiddles[case % 7] = F128::ZERO;
            }
            compare_case(&input, 1, 1, 0, &twiddles, 1);
        }
    }

    #[test]
    fn qresident_dense_preserves_aapcs64_d8_through_d15() {
        use core::arch::asm;

        let mut state = 0x082e_fa98_ec4e_6c89;
        let mut values: [F128; 8] = core::array::from_fn(|_| next_f128(&mut state));
        let twiddles: [F128; 7] = core::array::from_fn(|_| next_f128(&mut state));
        let before = [
            0x0123_4567_89ab_cdef,
            0x1023_4567_89ab_cdef,
            0x2023_4567_89ab_cdef,
            0x3023_4567_89ab_cdef,
            0x4023_4567_89ab_cdef,
            0x5023_4567_89ab_cdef,
            0x6023_4567_89ab_cdef,
            0x7023_4567_89ab_cdef,
        ];
        let mut after = [0u64; 8];

        unsafe {
            asm!(
                "stp x19, x20, [sp, #-16]!",
                "mov x19, x4",
                "mov x20, x5",
                "ldp d8, d9, [x19, #0]",
                "ldp d10, d11, [x19, #16]",
                "ldp d12, d13, [x19, #32]",
                "ldp d14, d15, [x19, #48]",
                "bl _flock_ntt_fused3_dense_qresident",
                "stp d8, d9, [x20, #0]",
                "stp d10, d11, [x20, #16]",
                "stp d12, d13, [x20, #32]",
                "stp d14, d15, [x20, #48]",
                "ldp x19, x20, [sp], #16",
                in("x0") values.as_mut_ptr(),
                in("x1") core::mem::size_of::<F128>(),
                in("x2") 1usize,
                in("x3") twiddles.as_ptr(),
                in("x4") before.as_ptr(),
                in("x5") after.as_mut_ptr(),
                clobber_abi("C"),
            );
        }

        assert_eq!(after, before);
    }

    #[test]
    #[ignore = "diagnostic timing; run explicitly with --release --ignored --nocapture"]
    fn qresident_dense_real_layout_32k_seam() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const EIGHTH: usize = 4;
        const NUM_NTTS: usize = 64;
        const ITERATIONS: usize = 2_000;
        const PAIRS: usize = 12;

        fn run(
            data: &mut [F128],
            twiddles: &[F128; 7],
            candidate: bool,
            iterations: usize,
        ) -> Duration {
            let start = Instant::now();
            for _ in 0..iterations {
                for r in 0..EIGHTH {
                    unsafe {
                        if candidate {
                            aarch64::butterfly_fused_3layer_row_dense_qresident(
                                black_box(data.as_mut_ptr()),
                                EIGHTH,
                                NUM_NTTS,
                                r,
                                black_box(twiddles),
                                NUM_NTTS,
                            );
                        } else {
                            butterfly_fused_3layer_row_portable_oracle(
                                black_box(data.as_mut_ptr()),
                                EIGHTH,
                                NUM_NTTS,
                                r,
                                black_box(twiddles),
                                NUM_NTTS,
                            );
                        }
                    }
                }
            }
            black_box(&data[0]);
            start.elapsed()
        }

        let mut state = 0x4528_21e6_38d0_1377;
        let initial: Vec<_> = (0..8 * EIGHTH * NUM_NTTS)
            .map(|_| next_f128(&mut state))
            .collect();
        assert_eq!(initial.len() * core::mem::size_of::<F128>(), 32 * 1024);
        let twiddles: [F128; 7] = core::array::from_fn(|_| next_f128(&mut state));

        let mut warm_baseline = initial.clone();
        let mut warm_candidate = initial.clone();
        run(&mut warm_baseline, &twiddles, false, 100);
        run(&mut warm_candidate, &twiddles, true, 100);
        assert_eq!(warm_candidate, warm_baseline);

        let mut baseline_ns = Vec::with_capacity(PAIRS);
        let mut candidate_ns = Vec::with_capacity(PAIRS);
        let mut candidate_wins = 0usize;
        for pair in 0..PAIRS {
            let mut baseline = initial.clone();
            let mut candidate = initial.clone();
            let (baseline_elapsed, candidate_elapsed) = if pair & 1 == 0 {
                let baseline_elapsed = run(&mut baseline, &twiddles, false, ITERATIONS);
                let candidate_elapsed = run(&mut candidate, &twiddles, true, ITERATIONS);
                (baseline_elapsed, candidate_elapsed)
            } else {
                let candidate_elapsed = run(&mut candidate, &twiddles, true, ITERATIONS);
                let baseline_elapsed = run(&mut baseline, &twiddles, false, ITERATIONS);
                (baseline_elapsed, candidate_elapsed)
            };
            assert_eq!(candidate, baseline);
            let baseline = baseline_elapsed.as_nanos();
            let candidate = candidate_elapsed.as_nanos();
            candidate_wins += usize::from(candidate < baseline);
            baseline_ns.push(baseline);
            candidate_ns.push(candidate);
            eprintln!(
                "QRESIDENT_32K pair={pair} baseline_ns={baseline} candidate_ns={candidate} delta_ns={}",
                baseline as i128 - candidate as i128
            );
        }

        baseline_ns.sort_unstable();
        candidate_ns.sort_unstable();
        let baseline_median = (baseline_ns[PAIRS / 2 - 1] + baseline_ns[PAIRS / 2]) / 2;
        let candidate_median = (candidate_ns[PAIRS / 2 - 1] + candidate_ns[PAIRS / 2]) / 2;
        let reduction =
            100.0 * (baseline_median as f64 - candidate_median as f64) / baseline_median as f64;
        eprintln!(
            "QRESIDENT_32K_SUMMARY pairs={PAIRS} wins={candidate_wins} baseline_median_ns={baseline_median} candidate_median_ns={candidate_median} reduction_pct={reduction:.6}"
        );
    }
}

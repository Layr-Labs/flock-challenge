//! Architecture-selected kernels over contiguous [`F128`] slices.

use super::F128;

#[cfg(any(
    test,
    not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ),
        all(target_arch = "aarch64", target_feature = "aes")
    ))
))]
mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod x86_64;

/// AArch64 deferred-reduction kernel for the ranked opening lookahead scan.
/// The caller retains the scalar implementation as the portable and exact
/// same-binary fallback.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(crate) fn round0_and_round1_lookahead(
    witness: &[F128],
    basis: &[F128],
) -> ((F128, F128), [F128; 6]) {
    assert_eq!(witness.len(), basis.len());
    assert!(witness.len().is_multiple_of(4));
    // SAFETY: the cfg gate supplies PMULL through `aes`; the checks above are
    // the complete slice-shape contract of the architecture kernel.
    unsafe { aarch64::round0_and_round1_lookahead(witness, basis) }
}

/// Deferred-reduction round-zero message `(u_0, u_2)` over paired slots.
/// Bit-identical to the fully-reduced scalar pair loop, since reduction is
/// F2-linear and commutes with the XOR product sum.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(crate) fn round0(witness: &[F128], basis: &[F128]) -> (F128, F128) {
    assert_eq!(witness.len(), basis.len());
    assert!(witness.len().is_multiple_of(2));
    // SAFETY: the cfg gate supplies PMULL through `aes`; the checks above are
    // the complete slice-shape contract of the architecture kernel.
    unsafe { aarch64::round0(witness, basis) }
}

/// Fold one banked output slot with deferred reduction:
/// `Σ_{k<BANKS} weight[k] · input[k]`, reduced once instead of `BANKS` times.
///
/// Bit-identical to
/// `weight.iter().zip(input).fold(ZERO, |a, (w, x)| a + *w * *x)`. AArch64
/// uses the NEON kernel; other targets keep the portable [`F256Unreduced`]
/// accumulator, which is the same algebra with the portable primitive.
///
/// [`F256Unreduced`]: super::F256Unreduced
#[inline]
pub(crate) fn fold_banked_slot<const BANKS: usize>(weight: &[F128; BANKS], input: &[F128]) -> F128 {
    debug_assert!(input.len() >= BANKS);
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: the cfg gate supplies PMULL through `aes`; the caller's
        // sub-slice guarantees at least `BANKS` readable elements.
        unsafe { aarch64::fold_banked_slot::<BANKS>(weight, input) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        let mut acc = super::F256Unreduced::ZERO;
        for (k, w) in weight.iter().enumerate() {
            acc ^= w.mul_unreduced(input[k]);
        }
        acc.reduce()
    }
}

/// Fold adjacent pairs from `src` into `dst`, starting at pair `base`.
///
/// Computes `dst[t] = src[2j] * (1 + r) + src[2j + 1] * r`, where
/// `j = base + t`. Architecture selection is resolved at compile time.
#[inline]
#[allow(dead_code)]
pub(crate) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    assert!(
        base <= src.len() / 2 && dst.len() <= src.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features and the
    // bounds check above guarantees both source elements for every output.
    unsafe {
        x86_64::fold_pairs(src, base, dst, r);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate guarantees PMULL support through the aes feature;
    // the bounds check above guarantees both source elements for every output.
    unsafe {
        aarch64::fold_pairs(src, base, dst, r);
    }

    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    portable::fold_pairs(src, base, dst, r);
}

/// AArch64 fused kernel for the Ligerito sumcheck hot path: fold `f` and `b`
/// at the same challenge and return the next round's `(u_0, u_2)` message.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(crate) fn fold_two_and_msg(
    f: &[F128],
    b: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
) -> (F128, F128) {
    assert_eq!(f.len(), b.len());
    assert_eq!(nf.len(), nb.len());
    assert!(base.is_multiple_of(2));
    assert!(nf.len().is_multiple_of(2));
    assert!(
        base <= f.len() / 2 && nf.len() <= f.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );
    // SAFETY: the cfg gate supplies PMULL, and the checks above establish all
    // source/destination bounds plus the message-pair alignment.
    unsafe { aarch64::fold_two_and_msg(f, b, base, nf, nb, r) }
}

/// AArch64 fused kernel for two consecutive Ligerito folds plus the direct
/// and one-round-lookahead messages. The lookahead lets the prover preserve
/// Fiat-Shamir order without materializing the intermediate half-sized state.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(crate) fn fold2_two_and_msgs(
    f: &[F128],
    b: &[F128],
    base: usize,
    wf: &mut [F128],
    wb: &mut [F128],
    r_a: F128,
    r_b: F128,
    nt_stores: bool,
) -> (F128, F128, [F128; 6]) {
    assert_eq!(f.len(), b.len());
    assert_eq!(wf.len(), wb.len());
    assert!(base.is_multiple_of(4));
    assert!(wf.len().is_multiple_of(4));
    assert!(4 * (base + wf.len()) <= f.len());
    // SAFETY: cfg supplies PMULL; the checks establish bounds and alignment.
    unsafe { aarch64::fold2_two_and_msgs(f, b, base, wf, wb, r_a, r_b, nt_stores) }
}

/// AArch64 fused final-pair kernel: bind two challenges and return only the
/// direct next message. The final initial-lane pair has no consumer for the
/// ordinary six-coefficient lookahead.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(crate) fn fold2_two_and_msg(
    f: &[F128],
    b: &[F128],
    base: usize,
    wf: &mut [F128],
    wb: &mut [F128],
    r_a: F128,
    r_b: F128,
    nt_stores: bool,
) -> (F128, F128) {
    assert_eq!(f.len(), b.len());
    assert_eq!(wf.len(), wb.len());
    assert!(base.is_multiple_of(4));
    assert!(wf.len().is_multiple_of(4));
    assert!(4 * (base + wf.len()) <= f.len());
    // SAFETY: cfg supplies PMULL; the checks establish bounds and alignment.
    unsafe { aarch64::fold2_two_and_msg(f, b, base, wf, wb, r_a, r_b, nt_stores) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_fold_matches_portable_with_offset_and_tail() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let src: Vec<F128> = (0..30)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let r = F128 {
            lo: next(),
            hi: next(),
        };
        let mut expected = vec![F128::ZERO; 9];
        let mut actual = vec![F128::ZERO; 9];

        portable::fold_pairs(&src, 3, &mut expected, r);
        fold_pairs(&src, 3, &mut actual, r);

        assert_eq!(actual, expected);
    }

    /// The `nt_stores` arm of the fold2 kernel is size-gated to beyond-LLC
    /// rounds in production and therefore unreachable at test sizes — force
    /// it here and require bit-identical outputs vs the normal-store arm.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn fold2_nt_store_arm_matches_normal_stores() {
        let mut state = 0x0932_2284_e49c_db2d_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for trial in 0..32 {
            let n_out = 4 * (1 + (trial % 7)); // multiples of 4, incl. multi-group
            let f: Vec<F128> = (0..8 * n_out).map(|_| F128::new(next(), next())).collect();
            let b: Vec<F128> = (0..8 * n_out).map(|_| F128::new(next(), next())).collect();
            let r_a = F128::new(next(), next());
            let r_b = F128::new(next(), next());
            let mut wf_normal = vec![F128::ZERO; n_out];
            let mut wb_normal = vec![F128::ZERO; n_out];
            let mut wf_nt = vec![F128::ZERO; n_out];
            let mut wb_nt = vec![F128::ZERO; n_out];
            let (u0_n, u2_n, c_n) =
                fold2_two_and_msgs(&f, &b, 0, &mut wf_normal, &mut wb_normal, r_a, r_b, false);
            let (u0_t, u2_t, c_t) =
                fold2_two_and_msgs(&f, &b, 0, &mut wf_nt, &mut wb_nt, r_a, r_b, true);
            assert_eq!(wf_normal, wf_nt, "wf trial={trial}");
            assert_eq!(wb_normal, wb_nt, "wb trial={trial}");
            assert_eq!((u0_n, u2_n, c_n), (u0_t, u2_t, c_t), "msgs trial={trial}");
        }
    }

    /// The final-pair kernel must materialize the same state and direct
    /// message as the full fold2 kernel; only the unused lookahead is absent.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn fold2_direct_only_matches_full_kernel() {
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for trial in 0..64 {
            let base = 4 * (trial % 3);
            let n_out = 4 * (1 + (trial % 9));
            let input_len = 4 * (base + n_out);
            let f: Vec<F128> = (0..input_len).map(|_| F128::new(next(), next())).collect();
            let b: Vec<F128> = (0..input_len).map(|_| F128::new(next(), next())).collect();
            let r_a = F128::new(next(), next());
            let r_b = F128::new(next(), next());
            let mut full_f = vec![F128::ZERO; n_out];
            let mut full_b = vec![F128::ZERO; n_out];
            let mut direct_f = vec![F128::ZERO; n_out];
            let mut direct_b = vec![F128::ZERO; n_out];
            let nt_stores = trial % 2 == 1;

            let (want_u0, want_u2, _) =
                fold2_two_and_msgs(&f, &b, base, &mut full_f, &mut full_b, r_a, r_b, nt_stores);
            let got = fold2_two_and_msg(
                &f,
                &b,
                base,
                &mut direct_f,
                &mut direct_b,
                r_a,
                r_b,
                nt_stores,
            );
            assert_eq!(direct_f, full_f, "f trial={trial}");
            assert_eq!(direct_b, full_b, "b trial={trial}");
            assert_eq!(got, (want_u0, want_u2), "message trial={trial}");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn fused_two_fold_and_message_matches_separate_oracle() {
        let mut state = 0x1319_8a2e_0370_7344_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for trial in 0..64 {
            let f: Vec<F128> = (0..46).map(|_| F128::new(next(), next())).collect();
            let b: Vec<F128> = (0..46).map(|_| F128::new(next(), next())).collect();
            let r = F128::new(next(), next());
            let mut expected_f = vec![F128::ZERO; 16];
            let mut expected_b = vec![F128::ZERO; 16];
            portable::fold_pairs(&f, 4, &mut expected_f, r);
            portable::fold_pairs(&b, 4, &mut expected_b, r);
            let mut expected_u0 = F128::ZERO;
            let mut expected_u2 = F128::ZERO;
            for k in (0..expected_f.len()).step_by(2) {
                expected_u0 += expected_f[k] * expected_b[k];
                expected_u2 +=
                    (expected_f[k] + expected_f[k + 1]) * (expected_b[k] + expected_b[k + 1]);
            }

            let mut got_f = vec![F128::ZERO; expected_f.len()];
            let mut got_b = vec![F128::ZERO; expected_b.len()];
            let (got_u0, got_u2) = fold_two_and_msg(&f, &b, 4, &mut got_f, &mut got_b, r);
            assert_eq!(got_f, expected_f, "f trial={trial}");
            assert_eq!(got_b, expected_b, "b trial={trial}");
            assert_eq!(got_u0, expected_u0, "u0 trial={trial}");
            assert_eq!(got_u2, expected_u2, "u2 trial={trial}");
        }
    }

    /// Oracle for the deferred-reduction banked slot fold used by the
    /// direct-fold4 (`fold16`) and direct-fold8 (`fold64`) materializers:
    /// one reduction per slot must produce exactly the bits of the
    /// per-bank fully-reduced accumulation.
    #[test]
    fn banked_slot_fold_matches_fully_reduced_oracle() {
        let mut state = 0xbb67_ae85_84ca_a73b_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        fn oracle<const BANKS: usize>(weight: &[F128; BANKS], input: &[F128]) -> F128 {
            let mut value = F128::ZERO;
            for bank in 0..BANKS {
                value += weight[bank] * input[bank];
            }
            value
        }

        for trial in 0..64 {
            // Production bank counts, plus a non-multiple-of-4 width to
            // exercise the kernel's scalar tail.
            let w16: [F128; 16] = std::array::from_fn(|_| F128::new(next(), next()));
            let w64: [F128; 64] = std::array::from_fn(|_| F128::new(next(), next()));
            let w6: [F128; 6] = std::array::from_fn(|_| F128::new(next(), next()));
            // Slot windows are read out of a longer buffer, exactly as the
            // materializers slice `input[64 * slot..]`.
            let buf: Vec<F128> = (0..256).map(|_| F128::new(next(), next())).collect();
            let off16 = 16 * (trial % 8);
            let off64 = 64 * (trial % 3);
            let off6 = trial % 11;

            assert_eq!(
                fold_banked_slot::<16>(&w16, &buf[off16..off16 + 16]),
                oracle(&w16, &buf[off16..off16 + 16]),
                "banks=16 trial={trial}"
            );
            assert_eq!(
                fold_banked_slot::<64>(&w64, &buf[off64..off64 + 64]),
                oracle(&w64, &buf[off64..off64 + 64]),
                "banks=64 trial={trial}"
            );
            assert_eq!(
                fold_banked_slot::<6>(&w6, &buf[off6..off6 + 6]),
                oracle(&w6, &buf[off6..off6 + 6]),
                "banks=6 trial={trial}"
            );
        }
    }

    /// Oracle for the deferred-reduction round-zero kernel that replaces the
    /// scalar pair loop closing each direct-fold8 block.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn round0_deferred_matches_fully_reduced_oracle() {
        let mut state = 0x3c6e_f372_fe94_f82b_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for trial in 0..64 {
            // Even lengths only; include a non-multiple-of-4 length so the
            // kernel's two-slot tail is exercised.
            let n = 2 * (1 + trial % 17);
            let f: Vec<F128> = (0..n).map(|_| F128::new(next(), next())).collect();
            let b: Vec<F128> = (0..n).map(|_| F128::new(next(), next())).collect();
            let mut want_u0 = F128::ZERO;
            let mut want_u2 = F128::ZERO;
            for k in (0..n).step_by(2) {
                want_u0 += f[k] * b[k];
                want_u2 += (f[k] + f[k + 1]) * (b[k] + b[k + 1]);
            }
            assert_eq!(round0(&f, &b), (want_u0, want_u2), "n={n} trial={trial}");
        }
    }
}

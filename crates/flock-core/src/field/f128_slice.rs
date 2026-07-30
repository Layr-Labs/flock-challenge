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

/// Fold adjacent pairs from `src` into `dst`, starting at pair `base`.
///
/// Computes `dst[t] = src[2j] * (1 + r) + src[2j + 1] * r`, where
/// `j = base + t`. Architecture selection is resolved at compile time.
#[inline]
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

/// Fold adjacent pairs from `f` and `b` and accumulate the next sumcheck
/// message over pairs of folded outputs.
///
/// The destination length must be even so each `(2k, 2k + 1)` message pair is
/// wholly contained in this call. Returns
/// `(Σ f[2k]b[2k], Σ(f[2k]+f[2k+1])(b[2k]+b[2k+1]))` over the folded slices.
#[inline]
pub(crate) fn fold_pairs_and_message(
    f: &[F128],
    b: &[F128],
    base: usize,
    folded_f: &mut [F128],
    folded_b: &mut [F128],
    r: F128,
) -> (F128, F128) {
    debug_assert_eq!(f.len(), b.len(), "fold inputs must have equal length");
    debug_assert_eq!(
        folded_f.len(),
        folded_b.len(),
        "fold outputs must have equal length"
    );
    debug_assert!(
        folded_f.len().is_multiple_of(2),
        "message pairs must not straddle fold chunks"
    );
    debug_assert!(
        base <= f.len() / 2 && folded_f.len() <= f.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate guarantees PMULL support. All memory access in the
    // specialized kernel remains bounds-checked.
    unsafe {
        return aarch64::fold_pairs_and_message(f, b, base, folded_f, folded_b, r);
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        fold_pairs(f, base, folded_f, r);
        fold_pairs(b, base, folded_b, r);

        let mut u_0 = F128::ZERO;
        let mut u_2 = F128::ZERO;
        for (fs, bs) in folded_f.chunks_exact(2).zip(folded_b.chunks_exact(2)) {
            u_0 += fs[0] * bs[0];
            u_2 += (fs[0] + fs[1]) * (bs[0] + bs[1]);
        }
        (u_0, u_2)
    }
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

    #[test]
    fn selected_fold_and_message_matches_scalar_oracle() {
        let mut state = 0x1319_8a2e_0370_7344_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            F128 {
                lo: state,
                hi: state.rotate_left(29),
            }
        };
        let f: Vec<F128> = (0..48).map(|_| next()).collect();
        let b: Vec<F128> = (0..48).map(|_| next()).collect();
        let r = next();
        let mut expected_f = vec![F128::ZERO; 10];
        let mut expected_b = vec![F128::ZERO; 10];
        let mut actual_f = vec![F128::ZERO; 10];
        let mut actual_b = vec![F128::ZERO; 10];

        portable::fold_pairs(&f, 4, &mut expected_f, r);
        portable::fold_pairs(&b, 4, &mut expected_b, r);
        let mut expected = (F128::ZERO, F128::ZERO);
        for (fs, bs) in expected_f.chunks_exact(2).zip(expected_b.chunks_exact(2)) {
            expected.0 += fs[0] * bs[0];
            expected.1 += (fs[0] + fs[1]) * (bs[0] + bs[1]);
        }

        let actual = fold_pairs_and_message(&f, &b, 4, &mut actual_f, &mut actual_b, r);

        assert_eq!(actual_f, expected_f);
        assert_eq!(actual_b, expected_b);
        assert_eq!(actual, expected);
    }
}

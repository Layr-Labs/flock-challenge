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
}

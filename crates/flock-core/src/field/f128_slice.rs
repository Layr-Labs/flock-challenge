//! Architecture-selected kernels over contiguous [`F128`] slices.

use super::F128;

#[cfg(any(test, not(all(target_arch = "aarch64", target_feature = "aes"))))]
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

/// Accumulate the two sufficient statistics for a factorized LSB equality
/// basis:
///
/// `a = sum_j f[2j] * eq_tail[j]`
/// `s = sum_j (f[2j] + f[2j + 1]) * eq_tail[j]`.
///
/// The AArch64 kernel keeps both product sums unreduced for the complete
/// slice, then reduces each once. Other targets retain the fully-reduced
/// portable loop. Reduction is F2-linear, so both paths are bit-identical.
#[inline]
pub(crate) fn round0_factorized_eq(f: &[F128], eq_tail: &[F128]) -> (F128, F128) {
    assert_eq!(
        f.len(),
        2 * eq_tail.len(),
        "factorized equality tail must cover every witness pair"
    );

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate supplies PMULL, and the hard length check above
    // establishes the architecture kernel's complete slice contract.
    unsafe {
        aarch64::round0_factorized_eq(f, eq_tail)
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    portable::round0_factorized_eq(f, eq_tail)
}

/// Expand one level of an equality table from its populated low half into an
/// equally sized high half. Both architecture paths implement, exactly,
/// `hi[i] = lo_old[i] * r` and `lo[i] = lo_old[i] + hi[i]`.
#[inline]
pub(crate) fn expand_eq_table_level(lo: &mut [F128], hi: &mut [F128], r: F128) {
    assert_eq!(
        lo.len(),
        hi.len(),
        "equality-table level halves must have equal lengths"
    );

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate supplies PMULL, and the hard length check above is
    // the architecture kernel's complete slice-shape contract.
    unsafe {
        aarch64::expand_eq_table_level(lo, hi, r)
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    portable::expand_eq_table_level(lo, hi, r)
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

/// Fold two adjacent banked output slots while loading each shared weight
/// once. Each output remains an independent deferred-reduction sum, so the
/// returned values are bit-identical to two [`fold_banked_slot`] calls.
#[inline]
pub(crate) fn fold_banked_slots2<const BANKS: usize>(
    weight: &[F128; BANKS],
    input: &[F128],
) -> [F128; 2] {
    debug_assert!(input.len() >= 2 * BANKS);
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: the cfg gate supplies PMULL through `aes`; the caller's
        // sub-slice guarantees two complete adjacent banked slots.
        unsafe { aarch64::fold_banked_slots2::<BANKS>(weight, input) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        let mut first = super::F256Unreduced::ZERO;
        let mut second = super::F256Unreduced::ZERO;
        for (bank, w) in weight.iter().enumerate() {
            first ^= w.mul_unreduced(input[bank]);
            second ^= w.mul_unreduced(input[BANKS + bank]);
        }
        [first.reduce(), second.reduce()]
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

/// Fold two same-sized states into their own lower halves and return the next
/// round's message. The allocation and capacity of both vectors are retained.
#[inline]
pub(crate) fn fold_two_and_msg_in_place(
    f: &mut Vec<F128>,
    b: &mut Vec<F128>,
    r: F128,
) -> (F128, F128) {
    assert_eq!(f.len(), b.len());
    assert!(f.len().is_multiple_of(4));
    let half = f.len() / 2;

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate supplies PMULL and the length checks establish the
    // complete in-place kernel shape. The kernel's raw-pointer loop loads each
    // source group before overwriting its lower-half output slots.
    let message = unsafe { aarch64::fold_two_and_msg_in_place(f, b, r) };

    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    let message = portable::fold_two_and_msg_in_place(f, b, r);

    f.truncate(half);
    b.truncate(half);
    message
}

/// Fold `f` and `b`, add `scale * basis_addend` to the folded basis, and
/// accumulate the next-round message over the corrected `(nf, nb)` state.
///
/// `base` is an index in the folded output domain. Consequently, output slot
/// `t` reads source pair `2 * (base + t)` and addend slot `base + t`. Keeping
/// the addend in the same global index space lets parallel callers share one
/// immutable slice without manufacturing per-chunk subslices.
#[inline]
#[cfg(test)]
pub(crate) fn fold_two_and_msg_with_scaled_basis_addend(
    f: &[F128],
    b: &[F128],
    basis_addend: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
    scale: F128,
) -> (F128, F128) {
    fold_two_and_msg_with_scaled_basis_addend_at(f, b, basis_addend, base, base, nf, nb, r, scale)
}

/// Fold-and-message variant whose addend is a chunk-local table. Source pair
/// `t` still comes from global folded-domain slot `base + t`, while its addend
/// comes from `basis_addend[t]`.
///
/// The ranked lazy-OOD split representation uses this to reuse one 2,048-slot
/// low equality factor for every high-factor chunk without materializing the
/// full tensor product.
#[inline]
pub(crate) fn fold_two_and_msg_with_scaled_local_basis_addend(
    f: &[F128],
    b: &[F128],
    basis_addend: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
    scale: F128,
) -> (F128, F128) {
    fold_two_and_msg_with_scaled_basis_addend_at(f, b, basis_addend, base, 0, nf, nb, r, scale)
}

/// Fold an incumbent witness/basis pair while consuming two deferred basis
/// corrections, then accumulate the next-round message in the same pass.
///
/// `deferred_basis` shares the input-domain indexing of `f` and `b`, while
/// `local_addend` starts at folded output slot zero for this chunk. For output
/// slot `t`, with `source = 2 * (base + t)`, the corrected basis is
///
/// ```text
/// b' = b[source]
///    + r       * (b[source] + b[source + 1])
///    + alpha   * deferred_basis[source]
///    + alpha_r * (deferred_basis[source] + deferred_basis[source + 1])
///    + gamma   * local_addend[t].
/// ```
///
/// The caller must precompute `alpha_r = alpha * r`. Passing it explicitly
/// lets every output consume the deferred ordinary glue without another
/// reduced multiplication in the hot loop.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_two_and_msg_with_deferred_basis_and_scaled_local_addend(
    f: &[F128],
    b: &[F128],
    deferred_basis: &[F128],
    local_addend: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
    alpha: F128,
    alpha_r: F128,
    gamma: F128,
) -> (F128, F128) {
    assert_eq!(f.len(), b.len(), "fold input lengths must match");
    assert_eq!(
        deferred_basis.len(),
        f.len(),
        "deferred basis must share the fold input domain"
    );
    assert_eq!(nf.len(), nb.len(), "fold output lengths must match");
    assert_eq!(
        local_addend.len(),
        nf.len(),
        "local addend must cover exactly this output chunk"
    );
    assert!(f.len().is_multiple_of(2), "fold input length must be even");
    assert!(
        base.is_multiple_of(2),
        "fold output base must preserve message pairs"
    );
    assert!(
        !nf.is_empty() && nf.len().is_multiple_of(2),
        "fold output must contain complete message pairs"
    );
    assert!(
        base <= f.len() / 2 && nf.len() <= f.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate supplies PMULL, and the hard checks above establish
    // every global source, local addend, and output bound plus pair alignment.
    unsafe {
        return aarch64::fold_two_and_msg_with_deferred_basis_and_scaled_local_addend(
            f,
            b,
            deferred_basis,
            local_addend,
            base,
            nf,
            nb,
            r,
            alpha,
            alpha_r,
            gamma,
        );
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    portable::fold_two_and_msg_with_deferred_basis_and_scaled_local_addend(
        f,
        b,
        deferred_basis,
        local_addend,
        base,
        nf,
        nb,
        r,
        alpha,
        alpha_r,
        gamma,
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn fold_two_and_msg_with_scaled_basis_addend_at(
    f: &[F128],
    b: &[F128],
    basis_addend: &[F128],
    base: usize,
    addend_base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
    scale: F128,
) -> (F128, F128) {
    assert_eq!(f.len(), b.len(), "fold input lengths must match");
    assert_eq!(nf.len(), nb.len(), "fold output lengths must match");
    assert!(f.len().is_multiple_of(2), "fold input length must be even");
    assert!(
        base.is_multiple_of(2),
        "fold output base must preserve message pairs"
    );
    assert!(
        !nf.is_empty() && nf.len().is_multiple_of(2),
        "fold output must contain complete message pairs"
    );
    assert!(
        base <= f.len() / 2 && nf.len() <= f.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );
    assert!(
        addend_base <= basis_addend.len() && nf.len() <= basis_addend.len() - addend_base,
        "scaled basis addend must cover every destination slot"
    );
    let basis_addend = &basis_addend[addend_base..addend_base + nf.len()];

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate supplies PMULL, and the hard checks above establish
    // every source, addend, and destination bound plus message-pair alignment.
    unsafe {
        return aarch64::fold_two_and_msg_with_scaled_basis_addend(
            f,
            b,
            basis_addend,
            base,
            nf,
            nb,
            r,
            scale,
        );
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        let one_plus_r = F128::ONE + r;
        let mut u_0 = F128::ZERO;
        let mut u_2 = F128::ZERO;
        let mut t = 0usize;
        while t < nf.len() {
            let source = 2 * (base + t);
            let f_0 = f[source] * one_plus_r + f[source + 1] * r;
            let f_1 = f[source + 2] * one_plus_r + f[source + 3] * r;
            let b_0 = b[source] * one_plus_r + b[source + 1] * r + scale * basis_addend[t];
            let b_1 = b[source + 2] * one_plus_r + b[source + 3] * r + scale * basis_addend[t + 1];
            nf[t] = f_0;
            nf[t + 1] = f_1;
            nb[t] = b_0;
            nb[t + 1] = b_1;
            u_0 += f_0 * b_0;
            u_2 += (f_0 + f_1) * (b_0 + b_1);
            t += 2;
        }
        (u_0, u_2)
    }
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
    fn eq_table_level_matches_portable_oracle() {
        let mut state = 0x4551_5441_424c_4531u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for &len in &[0usize, 1, 2, 3, 4, 17, 256, 1025] {
            let input: Vec<F128> = (0..len).map(|_| F128::new(next(), next())).collect();
            let r = F128::new(next(), next());
            let mut expected_lo = input.clone();
            let mut expected_hi = vec![F128::new(u64::MAX, u64::MAX); len];
            let mut actual_lo = input;
            let mut actual_hi = vec![F128::new(u64::MAX, u64::MAX); len];

            portable::expand_eq_table_level(&mut expected_lo, &mut expected_hi, r);
            expand_eq_table_level(&mut actual_lo, &mut actual_hi, r);

            assert_eq!(actual_lo, expected_lo, "low half, len={len}");
            assert_eq!(actual_hi, expected_hi, "high half, len={len}");
        }
    }

    /// Architecture selection must preserve the fully-reduced scalar result
    /// for empty, odd-tail, even-tail, and production-chunk geometries.
    #[test]
    fn factorized_eq_round0_matches_portable_oracle() {
        let mut state = 0x4641_4354_4F52_4551u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for &tail_len in &[0usize, 1, 2, 3, 7, 31, 2048, 2051] {
            let f: Vec<F128> = (0..2 * tail_len)
                .map(|_| F128::new(next(), next()))
                .collect();
            let eq_tail: Vec<F128> = (0..tail_len).map(|_| F128::new(next(), next())).collect();
            let expected = portable::round0_factorized_eq(&f, &eq_tail);
            let actual = round0_factorized_eq(&f, &eq_tail);
            assert_eq!(actual, expected, "tail_len={tail_len}");
        }
    }

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
    fn direct_fold8_in_place_five_rounds_match_allocating_oracle() {
        let mut state = 0x494E_504C_4143_4538u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for trial in 0..8 {
            let mut actual_f: Vec<F128> = (0..8192).map(|_| F128::new(next(), next())).collect();
            let mut actual_b: Vec<F128> = (0..8192).map(|_| F128::new(next(), next())).collect();
            let mut expected_f = actual_f.clone();
            let mut expected_b = actual_b.clone();
            let actual_f_ptr = actual_f.as_ptr();
            let actual_b_ptr = actual_b.as_ptr();
            let actual_f_capacity = actual_f.capacity();
            let actual_b_capacity = actual_b.capacity();

            for round in 0..5 {
                let r = F128::new(next(), next());
                let half = expected_f.len() / 2;
                let mut next_f = vec![F128::ZERO; half];
                let mut next_b = vec![F128::ZERO; half];
                portable::fold_pairs(&expected_f, 0, &mut next_f, r);
                portable::fold_pairs(&expected_b, 0, &mut next_b, r);
                let mut expected_message = (F128::ZERO, F128::ZERO);
                for t in (0..half).step_by(2) {
                    expected_message.0 += next_f[t] * next_b[t];
                    expected_message.1 += (next_f[t] + next_f[t + 1]) * (next_b[t] + next_b[t + 1]);
                }

                let actual_message = fold_two_and_msg_in_place(&mut actual_f, &mut actual_b, r);
                assert_eq!(actual_f, next_f, "f trial={trial} round={round}");
                assert_eq!(actual_b, next_b, "b trial={trial} round={round}");
                assert_eq!(
                    actual_message, expected_message,
                    "message trial={trial} round={round}"
                );
                assert_eq!(actual_f.as_ptr(), actual_f_ptr);
                assert_eq!(actual_b.as_ptr(), actual_b_ptr);
                assert_eq!(actual_f.capacity(), actual_f_capacity);
                assert_eq!(actual_b.capacity(), actual_b_capacity);
                expected_f = next_f;
                expected_b = next_b;
            }
        }
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

    /// The lazy-OOD fold kernel must be exactly the incumbent pair fold,
    /// followed by `nb += scale * addend`, followed by the ordinary message.
    /// Offset cases verify that the source and addend share the folded-domain
    /// `base`; the production-sized chunk exercises the complete hot loop.
    #[test]
    fn fused_fold_with_scaled_basis_addend_matches_two_pass_oracle() {
        let mut state = 0x4C41_5A59_4F4F_445Fu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for &(base, n_out) in &[(0usize, 2usize), (2, 4), (6, 10), (0, 2048), (6, 2048)] {
            let input_len = 2 * (base + n_out + 2);
            let addend_len = base + n_out + 3;
            let f: Vec<F128> = (0..input_len).map(|_| F128::new(next(), next())).collect();
            let b: Vec<F128> = (0..input_len).map(|_| F128::new(next(), next())).collect();
            let addend: Vec<F128> = (0..addend_len).map(|_| F128::new(next(), next())).collect();
            let challenges = [
                (F128::ZERO, F128::ZERO),
                (F128::ZERO, F128::ONE),
                (F128::ONE, F128::ZERO),
                (F128::ONE, F128::ONE),
                (F128::new(next(), next()), F128::new(next(), next())),
                (F128::new(next(), next()), F128::new(next(), next())),
            ];

            for (case, &(r, scale)) in challenges.iter().enumerate() {
                let mut expected_f = vec![F128::ZERO; n_out];
                let mut expected_b = vec![F128::ZERO; n_out];
                portable::fold_pairs(&f, base, &mut expected_f, r);
                portable::fold_pairs(&b, base, &mut expected_b, r);
                for t in 0..n_out {
                    expected_b[t] += scale * addend[base + t];
                }
                let mut expected_u_0 = F128::ZERO;
                let mut expected_u_2 = F128::ZERO;
                for t in (0..n_out).step_by(2) {
                    expected_u_0 += expected_f[t] * expected_b[t];
                    expected_u_2 +=
                        (expected_f[t] + expected_f[t + 1]) * (expected_b[t] + expected_b[t + 1]);
                }

                let sentinel = F128::new(u64::MAX, 0xA5A5_A5A5_A5A5_A5A5);
                let mut actual_f = vec![sentinel; n_out];
                let mut actual_b = vec![sentinel; n_out];
                let actual_msg = fold_two_and_msg_with_scaled_basis_addend(
                    &f,
                    &b,
                    &addend,
                    base,
                    &mut actual_f,
                    &mut actual_b,
                    r,
                    scale,
                );
                assert_eq!(actual_f, expected_f, "f base={base} n={n_out} case={case}");
                assert_eq!(actual_b, expected_b, "b base={base} n={n_out} case={case}");
                assert_eq!(
                    actual_msg,
                    (expected_u_0, expected_u_2),
                    "message base={base} n={n_out} case={case}"
                );
            }
        }
    }

    /// Deferred ordinary glue and the retained lazy-OOD correction must be
    /// consumed before the next message is accumulated. Nonzero `base` cases
    /// distinguish the global deferred-basis domain from the local addend.
    #[test]
    fn fused_fold_with_deferred_basis_and_local_addend_matches_oracle() {
        let mut state = 0x4445_4645_5252_4544u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for &(base, n_out) in &[(0usize, 2usize), (2, 4), (6, 10), (0, 2048), (6, 2048)] {
            let input_len = 2 * (base + n_out + 2);
            let f: Vec<F128> = (0..input_len).map(|_| F128::new(next(), next())).collect();
            let b: Vec<F128> = (0..input_len).map(|_| F128::new(next(), next())).collect();
            let deferred_basis: Vec<F128> =
                (0..input_len).map(|_| F128::new(next(), next())).collect();
            let local_addend: Vec<F128> = (0..n_out).map(|_| F128::new(next(), next())).collect();
            let challenges = [
                (F128::ZERO, F128::ZERO, F128::ZERO),
                (F128::ZERO, F128::ONE, F128::ONE),
                (F128::ONE, F128::ZERO, F128::ONE),
                (F128::ONE, F128::ONE, F128::ZERO),
                (F128::ONE, F128::ONE, F128::ONE),
                (
                    F128::new(next(), next()),
                    F128::new(next(), next()),
                    F128::new(next(), next()),
                ),
                (
                    F128::new(next(), next()),
                    F128::new(next(), next()),
                    F128::new(next(), next()),
                ),
            ];

            for (case, &(r, alpha, gamma)) in challenges.iter().enumerate() {
                let alpha_r = alpha * r;
                let mut expected_f = vec![F128::ZERO; n_out];
                let mut expected_b = vec![F128::ZERO; n_out];
                let mut folded_deferred = vec![F128::ZERO; n_out];
                portable::fold_pairs(&f, base, &mut expected_f, r);
                portable::fold_pairs(&b, base, &mut expected_b, r);
                portable::fold_pairs(&deferred_basis, base, &mut folded_deferred, r);
                for t in 0..n_out {
                    expected_b[t] += alpha * folded_deferred[t] + gamma * local_addend[t];
                }
                let mut expected_u_0 = F128::ZERO;
                let mut expected_u_2 = F128::ZERO;
                for t in (0..n_out).step_by(2) {
                    expected_u_0 += expected_f[t] * expected_b[t];
                    expected_u_2 +=
                        (expected_f[t] + expected_f[t + 1]) * (expected_b[t] + expected_b[t + 1]);
                }
                let expected_msg = (expected_u_0, expected_u_2);

                let sentinel = F128::new(u64::MAX, 0xA5A5_A5A5_A5A5_A5A5);
                let mut portable_f = vec![sentinel; n_out];
                let mut portable_b = vec![sentinel; n_out];
                let portable_msg =
                    portable::fold_two_and_msg_with_deferred_basis_and_scaled_local_addend(
                        &f,
                        &b,
                        &deferred_basis,
                        &local_addend,
                        base,
                        &mut portable_f,
                        &mut portable_b,
                        r,
                        alpha,
                        alpha_r,
                        gamma,
                    );
                assert_eq!(
                    portable_f, expected_f,
                    "portable f base={base} n={n_out} case={case}"
                );
                assert_eq!(
                    portable_b, expected_b,
                    "portable b base={base} n={n_out} case={case}"
                );
                assert_eq!(
                    portable_msg, expected_msg,
                    "portable msg base={base} n={n_out} case={case}"
                );

                let mut actual_f = vec![sentinel; n_out];
                let mut actual_b = vec![sentinel; n_out];
                let actual_msg = fold_two_and_msg_with_deferred_basis_and_scaled_local_addend(
                    &f,
                    &b,
                    &deferred_basis,
                    &local_addend,
                    base,
                    &mut actual_f,
                    &mut actual_b,
                    r,
                    alpha,
                    alpha_r,
                    gamma,
                );
                assert_eq!(
                    actual_f, expected_f,
                    "selected f base={base} n={n_out} case={case}"
                );
                assert_eq!(
                    actual_b, expected_b,
                    "selected b base={base} n={n_out} case={case}"
                );
                assert_eq!(
                    actual_msg, expected_msg,
                    "selected msg base={base} n={n_out} case={case}"
                );
            }
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
            assert_eq!(
                fold_banked_slots2::<16>(&w16, &buf[off16..off16 + 32]),
                [
                    oracle(&w16, &buf[off16..off16 + 16]),
                    oracle(&w16, &buf[off16 + 16..off16 + 32]),
                ],
                "pair banks=16 trial={trial}"
            );
            assert_eq!(
                fold_banked_slots2::<64>(&w64, &buf[off64..off64 + 128]),
                [
                    oracle(&w64, &buf[off64..off64 + 64]),
                    oracle(&w64, &buf[off64 + 64..off64 + 128]),
                ],
                "pair banks=64 trial={trial}"
            );
            assert_eq!(
                fold_banked_slots2::<6>(&w6, &buf[off6..off6 + 12]),
                [
                    oracle(&w6, &buf[off6..off6 + 6]),
                    oracle(&w6, &buf[off6 + 6..off6 + 12]),
                ],
                "pair banks=6 trial={trial}"
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

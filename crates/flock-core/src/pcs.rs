//! Polynomial commitment scheme for the bit-MLE witness `ẑ` over GF(2).
//!
//! Construction: Binius-style PCS with F_{2^128} packing.
//!
//! - **Commit**: pack the 2^m Boolean witness into 2^(m−7) F_{2^128} elements
//!   (one bit per polynomial-basis coordinate of F_{2^128}), batch RS-encode
//!   via additive NTT, Merkle-commit the codeword.
//! - **Open**: at a QuirkyPoint (z_skip, x_outer) from the zerocheck/lincheck:
//!   1. [`ring_switch::prove`] sends 128 partial-evaluations `s_hat_v` and
//!      produces a sumcheck target `(rs_eq_ind, sumcheck_claim)`.
//!   2. [`ligerito::recursive_prover_with_basis`] discharges the combined
//!      claim `⟨packed_witness, b_combined⟩ = target_combined` via the
//!      recursive Ligerito argument, reusing the commit-time codeword and
//!      Merkle tree as Ligerito's L0 commitment.
//! - **Verify**: the verifier replays ring-switching succinctly, then drives
//!   the succinct recursive Ligerito verifier, evaluating the combined basis
//!   at the residual point (see [`verify_opening_batch_ligerito_mixed`]).
//!
//! See [DP24](https://eprint.iacr.org/2024/504) (ring-switching) and the
//! ligerito module docs for the recursion.

pub mod commit;
pub mod jagged;
pub mod ligerito;
pub mod pack;
pub mod ring_switch;
pub mod tensor_algebra;

/// Whether untimed warmup permanently selected the ranked Metal commit.
/// Callers may omit CPU-only speculative buffers once this is true.
pub fn ranked_gpu_commit_latched_on() -> bool {
    crate::gpu_commit::gpu_commit_latched_on()
}

pub use commit::{
    Commitment, PcsParams, ProverData, commit, commit_from_streamed_first_pass, commit_into,
    commit_preinitialized, prefault_codeword_during, use_ranked_from_message_commit,
};
pub use pack::{LOG_PACKING, pack_witness, unpack_witness};
pub use ring_switch::{RingSwitchProof, SparseEqTensor};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::zerocheck::PaddingSpec;
use serde::{Deserialize, Serialize};

/// Batched opening proof: ring-switching frontend + Ligerito backend.
/// The combined `b_combined` + target_combined feed
/// [`ligerito::recursive_prover_with_basis`] (see ligerito module docs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchOpeningProofLigerito {
    pub ring_switches: Vec<RingSwitchProof>,
    pub ligerito: ligerito::LigeritoProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    RingSwitch(ring_switch::VerifyError),
    /// The Ligerito recursive verifier rejected the proof.
    Ligerito,
}

/// `eq_ind` representation for a packed-direct claim. The contributed value at
/// scattered index `j` is the tensor entry — for the dense variant the index
/// is the array offset; for the sparse variant it's reconstructed via
/// [`SparseEqTensor::scatter_idx`].
#[derive(Clone, Debug)]
pub enum DirectEqInd {
    /// Fully-materialized `eq_ind(point)` of length `2^L`.
    Dense(Vec<F128>),
    /// Sparse representation — non-zero entries at scattered indices.
    /// Built from a claim point with one or more exactly-zero coords via
    /// [`ring_switch::build_eq_sparse`].
    Sparse(SparseEqTensor),
}

/// A packed-MLE evaluation claim: `ẑ_packed(point) = value`. Unlike a
/// ring-switched claim, this is opened directly without going through the
/// bit-MLE ↔ packed-MLE bridge (no `s_hat_v`, no φ_8 weighting).
///
/// Use case: protocols whose sumcheck output is naturally a packed-MLE
/// evaluation (e.g. the chain shift sumcheck operating on packed columns
/// instead of bit-folded scalars). Skips the ring-switch step for this claim,
/// saving the `fold_1b_rows` + per-opening-tail work at the prover and the
/// ring-switch verify + φ_8 reconstruction at the verifier.
///
/// The claim-combine step adds `γ_k · eq_ind(point)` to `b_combined` and
/// `γ_k · value` to the target; the verifier's residual check contributes
/// `γ_k · eq_eval(point, residual_challenges)`.
#[derive(Clone, Debug)]
pub struct PackedDirectClaim {
    /// Multilinear point of length `L = m − 7`.
    pub point: Vec<F128>,
    /// Claimed `ẑ_packed(point)` value.
    pub value: F128,
    /// `eq_ind(point)` in dense or sparse form. Caller responsibility to
    /// match the claim's `point` — the contribution to `b_combined` is read
    /// directly from this tensor.
    pub eq_ind: DirectEqInd,
}

/// Mixed-claim batched open: supports both **ring-switched** claims (bit-MLE
/// openings reduced via `ring_switch::prove_batched`, with optional per-claim
/// precomputed `s_hat_v`) and **packed-direct** claims (packed-MLE openings
/// that skip ring-switch). Runs the ring_switch + b_combined computation, then
/// routes to [`ligerito::recursive_prover_with_basis`] using the existing
/// `prover_data`'s codeword + tree as Ligerito's L0 commit (no L0 re-commit).
///
/// `lig_config.initial_k` must equal `commitment.params.log_batch_size` so that
/// `prover_data`'s codeword/tree shape matches what Ligerito expects for L0.
#[allow(clippy::too_many_arguments)]
pub fn open_batch_mixed_ligerito_with_precomputed_s_hat_v<Ch: Challenger>(
    packed_witness: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    lig_config: &ligerito::ProverConfig,
    challenger: &mut Ch,
) -> BatchOpeningProofLigerito {
    let trace =
        std::env::var("PCS_TRACE").is_ok() || std::env::var_os("FLOCK_OPEN_TIMING").is_some();
    let t_total = std::time::Instant::now();

    assert_eq!(
        lig_config.initial_k, commitment.params.log_batch_size,
        "ligerito initial_k ({}) must match PcsParams.log_batch_size ({}) for L0 reuse",
        lig_config.initial_k, commitment.params.log_batch_size,
    );
    assert_eq!(
        lig_config.log_inv_rates[0], commitment.params.log_inv_rate,
        "ligerito log_inv_rates[0] ({}) must match PcsParams.log_inv_rate ({}) for L0 reuse",
        lig_config.log_inv_rates[0], commitment.params.log_inv_rate,
    );

    let combined = compute_combined_basis_and_target(
        &packed_witness,
        x_outers,
        precomputed_s_hat_v,
        packed_direct,
        padding,
        challenger,
        trace,
        ligerito::ranked_fold2_enabled(packed_witness.len(), lig_config.initial_k),
    );

    let t = std::time::Instant::now();
    let ligerito_proof = if let Some(direct) = combined.direct_fold8 {
        ligerito::recursive_prover_with_basis_direct_fold8(
            lig_config,
            packed_witness,
            combined.b_combined,
            direct,
            combined.target_combined,
            &prover_data.codeword,
            &*prover_data.merkle_tree,
            combined.round0_prime,
            challenger,
        )
    } else if let Some(direct) = combined.direct_fold4 {
        ligerito::recursive_prover_with_basis_direct_fold4(
            lig_config,
            packed_witness,
            combined.b_combined,
            direct,
            combined.target_combined,
            &prover_data.codeword,
            &*prover_data.merkle_tree,
            combined.round0_prime,
            combined
                .round1_lookahead
                .expect("direct-fold4 requires round-1 lookahead"),
            combined
                .round2_lookahead
                .expect("direct-fold4 requires round-2 lookahead"),
            combined
                .round3_lookahead
                .expect("direct-fold4 requires round-3 lookahead"),
            challenger,
        )
    } else if let Some(direct) = combined.direct_fold2 {
        ligerito::recursive_prover_with_basis_direct_ab_fold2(
            lig_config,
            packed_witness,
            combined.b_combined,
            direct,
            combined.target_combined,
            &prover_data.codeword,
            &*prover_data.merkle_tree,
            combined.round0_prime,
            combined
                .round1_lookahead
                .expect("direct AB fold2 requires round-1 lookahead"),
            challenger,
        )
    } else {
        ligerito::recursive_prover_with_basis_precomputed_round0(
            lig_config,
            packed_witness,
            combined.b_combined,
            combined.target_combined,
            &prover_data.codeword,
            &*prover_data.merkle_tree,
            combined.round0_prime,
            combined.round1_lookahead,
            challenger,
        )
    };
    if trace {
        eprintln!(
            "  [open_batch] ligerito::recursive_prover_with_basis: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        eprintln!(
            "  [open_batch] TOTAL: {:6.2} ms",
            t_total.elapsed().as_secs_f64() * 1e3
        );
    }

    BatchOpeningProofLigerito {
        ring_switches: combined.ring_switches,
        ligerito: ligerito_proof,
    }
}

/// What ring_switch + claim-combination produces, fed to the Ligerito backend.
struct CombinedClaim {
    ring_switches: Vec<RingSwitchProof>,
    b_combined: Vec<F128>,
    target_combined: F128,
    /// Round-0 sumcheck `(u_0, u_2)` prime over `packed_witness · b_combined`,
    /// consumed by `recursive_prover_with_basis_precomputed_round0`.
    round0_prime: (F128, F128),
    /// Round-1 message as two quadratics in the first fold challenge. Present
    /// only for the exact ranked two-challenge cadence.
    round1_lookahead: Option<[F128; 6]>,
    /// Experimental direct-fold4 round-2 bivariate lookahead.
    round2_lookahead: Option<Fold4Lookahead2>,
    /// Experimental direct-fold4 round-3 trivariate lookahead.
    round3_lookahead: Option<Fold4Lookahead3>,
    /// Per-claim sufficient statistics for direct materialization after rounds
    /// 0/1. `b_combined` still contains every ordinary claim (currently C) —
    /// unless deferred-C is active, in which case C rides along here as a
    /// second claim (products zeroed) and `b_combined` is empty, or the
    /// direct-C completion is active, in which case C rides along with real
    /// `products` and there is no basis sweep at all.
    direct_fold2: Option<Vec<ring_switch::DirectFold2Factors>>,
    /// Sixteen-bank direct factors. This is populated only behind the strict
    /// experimental opt-in and leaves the frontier path unchanged by default.
    direct_fold4: Option<Vec<ring_switch::DirectFold4Factors>>,
    /// Sixty-four-bank direct factors, populated when both claims carry an
    /// honest 64-bank precompute and the fold8 gate is on.
    direct_fold8: Option<Vec<ring_switch::DirectFold8Factors>>,
}

/// Compute the ordinary round-zero message and the following message as two
/// quadratics in the first challenge. The latter lets the ranked prover sample
/// its second challenge before binding the first, so both binds share one pass.
#[inline]
fn use_ranked_open_lookahead_neon(ranked_shape: bool, len: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        target_feature = "aes"
    )) && ranked_shape
        && len == (1usize << 15)
        && *ON.get_or_init(|| std::env::var_os("FLOCK_NO_OPEN_LOOKAHEAD_NEON").is_none())
}

#[inline]
#[cfg(test)]
fn round0_and_round1_lookahead(witness: &[F128], basis: &[F128]) -> ((F128, F128), [F128; 6]) {
    assert_eq!(witness.len(), basis.len());
    assert!(witness.len().is_multiple_of(4));

    round0_and_round1_lookahead_scalar(witness, basis)
}

/// Ranked-shape dispatcher for the deferred-reduction AArch64 kernel. Keeping
/// the generic helper scalar makes promotion an explicit property of the two
/// benchmark geometry call sites rather than an accidental property of a
/// local 2^15-slot slice.
#[inline]
fn round0_and_round1_lookahead_ranked(
    witness: &[F128],
    basis: &[F128],
    ranked_shape: bool,
) -> ((F128, F128), [F128; 6]) {
    assert_eq!(witness.len(), basis.len());
    assert!(witness.len().is_multiple_of(4));

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    if use_ranked_open_lookahead_neon(ranked_shape, witness.len()) {
        return crate::field::f128_slice::round0_and_round1_lookahead(witness, basis);
    }

    round0_and_round1_lookahead_scalar(witness, basis)
}

/// Kill switch for the direct-materializer deferred-reduction fast paths:
/// the banked `fold16`/`fold64` slot folds and the round-0 message kernels
/// that close each direct fold block. Every one of them is bit-identical to
/// the fully-reduced scalar loop it replaces (reduction mod p is F2-linear,
/// so it commutes with the XOR product sum), so this is a pure A/B escape
/// hatch rather than a correctness condition.
#[inline]
pub(crate) fn use_fold_deferred_reduce() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_FOLD_DEFERRED_REDUCE").is_none());
    *ON
}

/// Deferred-reduction dispatcher for the direct-fold4 materializer's block
/// tail. The `_ranked` sibling is pinned to the 2^15-slot combine geometry;
/// the direct-fold4 materializer's blocks are 2^13 slots, so it needs its own
/// gate. Both arms return the same bits.
#[inline]
pub(crate) fn round0_and_round1_lookahead_deferred(
    witness: &[F128],
    basis: &[F128],
) -> ((F128, F128), [F128; 6]) {
    assert_eq!(witness.len(), basis.len());
    assert!(witness.len().is_multiple_of(4));

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    if use_fold_deferred_reduce() {
        return crate::field::f128_slice::round0_and_round1_lookahead(witness, basis);
    }

    round0_and_round1_lookahead_scalar(witness, basis)
}

/// Deferred-reduction dispatcher for the direct-fold8 materializer's block
/// tail. Routing fold8 through the six-coefficient lookahead kernel would be
/// a regression: fold8 consumes only `(u_0, u_2)`, and the lookahead scan
/// spends eight unreduced products per four slots where round-zero needs
/// four.
#[inline]
pub(crate) fn round0_deferred(witness: &[F128], basis: &[F128]) -> (F128, F128) {
    assert_eq!(witness.len(), basis.len());

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    if use_fold_deferred_reduce() && witness.len().is_multiple_of(2) {
        return crate::field::f128_slice::round0(witness, basis);
    }

    round0_scalar(witness, basis)
}

/// Round-0 message `(u_0, u_2)` over paired slots, without the round-1
/// lookahead coefficients. Sums the same pair products as
/// [`round0_and_round1_lookahead_scalar`], so the values agree with it
/// bitwise (char-2 regrouping). Used by the fold8 materialize, whose M6 is
/// the last initial message — no lookahead follows.
#[inline]
pub(crate) fn round0_scalar(witness: &[F128], basis: &[F128]) -> (F128, F128) {
    assert_eq!(witness.len(), basis.len());
    let mut u0 = F128::ZERO;
    let mut u2 = F128::ZERO;
    for i in (0..witness.len()).step_by(2) {
        let a0 = witness[i];
        let a1 = witness[i + 1];
        let b0 = basis[i];
        let b1 = basis[i + 1];
        u0 += a0 * b0;
        u2 += (a0 + a1) * (b0 + b1);
    }
    (u0, u2)
}

#[inline]
fn round0_and_round1_lookahead_scalar(
    witness: &[F128],
    basis: &[F128],
) -> ((F128, F128), [F128; 6]) {
    let mut u0 = F128::ZERO;
    let mut u2 = F128::ZERO;
    let mut c = [F128::ZERO; 6];
    for i in (0..witness.len()).step_by(4) {
        let a0 = witness[i];
        let a1 = witness[i + 1];
        let a2 = witness[i + 2];
        let a3 = witness[i + 3];
        let b0 = basis[i];
        let b1 = basis[i + 1];
        let b2 = basis[i + 2];
        let b3 = basis[i + 3];
        let sa0 = a0 + a1;
        let sb0 = b0 + b1;
        let sa1 = a2 + a3;
        let sb1 = b2 + b3;
        let p_even0 = a0 * b0;
        let p_sum0 = sa0 * sb0;
        u0 += p_even0 + a2 * b2;
        u2 += p_sum0 + sa1 * sb1;
        c[0] += p_even0;
        // Karatsuba cross term: a0*sb0 + b0*sa0 equals
        // a1*b1 + (a0*b0) + (sa0*sb0). The endpoint products are already
        // live for c0/c2, so this costs one product instead of two.
        c[1] += a1 * b1 + p_even0 + p_sum0;
        c[2] += p_sum0;
        let e_a = a0 + a2;
        let e_b = b0 + b2;
        let se_a = sa0 + sa1;
        let se_b = sb0 + sb1;
        let p_even = e_a * e_b;
        let p_sum = se_a * se_b;
        // Same identity for the even/odd grouped pair. Here the complementary
        // endpoint is (a1+a3, b1+b3).
        let p_odd = (se_a + e_a) * (se_b + e_b);
        c[3] += p_even;
        c[4] += p_odd + p_even + p_sum;
        c[5] += p_sum;
    }
    ((u0, u2), c)
}
fn messages_from_direct_products(
    products: &[ring_switch::DirectFold2Factors],
) -> ((F128, F128), [F128; 6]) {
    let mut h = [F128::ZERO; 16];
    for claim in products {
        for (out, value) in h.iter_mut().zip(claim.products) {
            *out += value;
        }
    }
    let at = |e: usize, d: usize| h[4 * e + d];
    let block_sum = |es: &[usize], ds: &[usize]| {
        let mut sum = F128::ZERO;
        for &e in es {
            for &d in ds {
                sum += at(e, d);
            }
        }
        sum
    };
    let round0 = (
        at(0, 0) + at(2, 2),
        block_sum(&[0, 1], &[0, 1]) + block_sum(&[2, 3], &[2, 3]),
    );
    let lookahead = [
        at(0, 0),
        at(0, 1) + at(1, 0),
        block_sum(&[0, 1], &[0, 1]),
        block_sum(&[0, 2], &[0, 2]),
        block_sum(&[0, 2], &[1, 3]) + block_sum(&[1, 3], &[0, 2]),
        block_sum(&[0, 1, 2, 3], &[0, 1, 2, 3]),
    ];
    (round0, lookahead)
}

pub(crate) type Fold4Lookahead2 = [F128; 18];
pub(crate) type Fold4Lookahead3 = [F128; 54];
pub(crate) type Fold8Lookahead4 = [F128; 162];
pub(crate) type Fold8Lookahead5 = [F128; 486];

#[inline(always)]
fn quadratic_coefficients([at_zero, at_one, leading]: [F128; 3]) -> [F128; 3] {
    [at_zero, at_zero + at_one + leading, leading]
}

/// Convert an evaluation tensor over `{0, 1, leading}^variables` into the
/// matching degree-at-most-two coefficient tensor, in row-major coordinate
/// order. This is the fold3 interpolation algebra generalized to three prior
/// challenges for the direct-fold4 scaffold.
fn tensor_quadratic_coefficients(values: &mut [F128], variables: usize) {
    debug_assert_eq!(values.len(), 3usize.pow(variables as u32));
    for axis in 0..variables {
        let stride = 3usize.pow((variables - axis - 1) as u32);
        let period = 3 * stride;
        for block in (0..values.len()).step_by(period) {
            for offset in 0..stride {
                let indices = [
                    block + offset,
                    block + stride + offset,
                    block + 2 * stride + offset,
                ];
                let coefficients = quadratic_coefficients([
                    values[indices[0]],
                    values[indices[1]],
                    values[indices[2]],
                ]);
                for (index, coefficient) in indices.into_iter().zip(coefficients) {
                    values[index] = coefficient;
                }
            }
        }
    }
}

/// Build the two message-polynomial coefficient tensors at `round` from a
/// 16×16 bilinear product matrix. Prior-coordinate grid digit 0 selects the
/// zero endpoint, 1 the one endpoint, and 2 the quadratic leading term (the
/// sum of both endpoint banks). The current-coordinate `u_2` grid similarly
/// selects both halves. Higher coordinates are summed independently.
fn direct_fold4_message_coefficients(h: &[F128; 256], round: usize) -> (Vec<F128>, Vec<F128>) {
    debug_assert!(round < 4);
    let grid_len = 3usize.pow(round as u32);
    let mut endpoints = vec![F128::ZERO; 2 * grid_len];

    let product = |mask: u16| {
        let mut sum = F128::ZERO;
        for e in 0..16 {
            if mask & (1 << e) == 0 {
                continue;
            }
            for d in 0..16 {
                if mask & (1 << d) != 0 {
                    sum += h[16 * e + d];
                }
            }
        }
        sum
    };

    for current_leading in 0..2 {
        for grid_index in 0..grid_len {
            let mut total = F128::ZERO;
            let high_assignments = 1usize << (3 - round);
            for high in 0..high_assignments {
                let mut mask = 0u16;
                'bank: for bank in 0..16 {
                    if current_leading == 0 && ((bank >> round) & 1) != 0 {
                        continue;
                    }
                    for bit in 0..round {
                        let divisor = 3usize.pow((round - bit - 1) as u32);
                        let mode = (grid_index / divisor) % 3;
                        if mode < 2 && ((bank >> bit) & 1) != mode {
                            continue 'bank;
                        }
                    }
                    if bank >> (round + 1) != high {
                        continue;
                    }
                    mask |= 1 << bank;
                }
                total += product(mask);
            }
            endpoints[current_leading * grid_len + grid_index] = total;
        }
    }

    let (u0, u2) = endpoints.split_at_mut(grid_len);
    tensor_quadratic_coefficients(u0, round);
    tensor_quadratic_coefficients(u2, round);
    (u0.to_vec(), u2.to_vec())
}

/// Derive the first four transcript messages from sixteen-bank direct
/// sufficient statistics, without materializing either N-sized polynomial.
/// The returned lookaheads are respectively uni-, bi-, and trivariate
/// quadratics in the already-sampled challenges.
pub(crate) fn messages_from_direct_products_fold4(
    factors: &[ring_switch::DirectFold4Factors],
) -> ((F128, F128), [F128; 6], Fold4Lookahead2, Fold4Lookahead3) {
    let mut h = [F128::ZERO; 256];
    for claim in factors {
        for (out, value) in h.iter_mut().zip(claim.products) {
            *out += value;
        }
    }

    let (round0_u0, round0_u2) = direct_fold4_message_coefficients(&h, 0);
    let (round1_u0, round1_u2) = direct_fold4_message_coefficients(&h, 1);
    let (round2_u0, round2_u2) = direct_fold4_message_coefficients(&h, 2);
    let (round3_u0, round3_u2) = direct_fold4_message_coefficients(&h, 3);

    let mut round1 = [F128::ZERO; 6];
    round1[..3].copy_from_slice(&round1_u0);
    round1[3..].copy_from_slice(&round1_u2);
    let mut round2 = [F128::ZERO; 18];
    round2[..9].copy_from_slice(&round2_u0);
    round2[9..].copy_from_slice(&round2_u2);
    let mut round3 = [F128::ZERO; 54];
    round3[..27].copy_from_slice(&round3_u0);
    round3[27..].copy_from_slice(&round3_u2);

    ((round0_u0[0], round0_u2[0]), round1, round2, round3)
}

/// Build the two message-polynomial coefficient tensors at `round` from a
/// 64×64 bilinear product matrix. Same algebra as
/// [`direct_fold4_message_coefficients`] one level wider: prior-coordinate
/// grid digit 0/1 selects the zero/one endpoint, 2 the quadratic leading
/// term (both endpoint banks); higher coordinates are summed independently.
/// Selected banks always form a subcube, so the product sum iterates set
/// mask bits only (Σ_r 2·3^r·2^(5−r) configs × E|selected|² = 2^(r+1) ≈ 47K
/// F128 adds total — scalar-negligible).
#[cfg(test)]
fn direct_fold8_message_coefficients(h: &[F128; 4096], round: usize) -> (Vec<F128>, Vec<F128>) {
    debug_assert!(round < 6);
    let grid_len = 3usize.pow(round as u32);
    let mut endpoints = vec![F128::ZERO; 2 * grid_len];

    let product = |mask: u64| {
        let mut sum = F128::ZERO;
        let mut e_bits = mask;
        while e_bits != 0 {
            let e = e_bits.trailing_zeros() as usize;
            e_bits &= e_bits - 1;
            let row = &h[64 * e..64 * e + 64];
            let mut d_bits = mask;
            while d_bits != 0 {
                let d = d_bits.trailing_zeros() as usize;
                d_bits &= d_bits - 1;
                sum += row[d];
            }
        }
        sum
    };

    for current_leading in 0..2 {
        for grid_index in 0..grid_len {
            let mut total = F128::ZERO;
            let high_assignments = 1usize << (5 - round);
            for high in 0..high_assignments {
                let mut mask = 0u64;
                'bank: for bank in 0..64usize {
                    if current_leading == 0 && ((bank >> round) & 1) != 0 {
                        continue;
                    }
                    for bit in 0..round {
                        let divisor = 3usize.pow((round - bit - 1) as u32);
                        let mode = (grid_index / divisor) % 3;
                        if mode < 2 && ((bank >> bit) & 1) != mode {
                            continue 'bank;
                        }
                    }
                    if bank >> (round + 1) != high {
                        continue;
                    }
                    mask |= 1 << bank;
                }
                total += product(mask);
            }
            endpoints[current_leading * grid_len + grid_index] = total;
        }
    }

    let (u0, u2) = endpoints.split_at_mut(grid_len);
    tensor_quadratic_coefficients(u0, round);
    tensor_quadratic_coefficients(u2, round);
    (u0.to_vec(), u2.to_vec())
}

/// Derive the first six transcript messages from sixty-four-bank direct
/// sufficient statistics, without materializing either N-sized polynomial.
/// The returned lookaheads are respectively uni-, bi-, tri-, quadri-, and
/// quintivariate quadratics in the already-sampled challenges.
#[cfg(test)]
pub(crate) fn messages_from_direct_products_fold8(
    h: &[F128; 4096],
) -> (
    (F128, F128),
    [F128; 6],
    Fold4Lookahead2,
    Fold4Lookahead3,
    Fold8Lookahead4,
    Fold8Lookahead5,
) {
    let (round0_u0, round0_u2) = direct_fold8_message_coefficients(h, 0);
    let (round1_u0, round1_u2) = direct_fold8_message_coefficients(h, 1);
    let (round2_u0, round2_u2) = direct_fold8_message_coefficients(h, 2);
    let (round3_u0, round3_u2) = direct_fold8_message_coefficients(h, 3);
    let (round4_u0, round4_u2) = direct_fold8_message_coefficients(h, 4);
    let (round5_u0, round5_u2) = direct_fold8_message_coefficients(h, 5);

    let mut round1 = [F128::ZERO; 6];
    round1[..3].copy_from_slice(&round1_u0);
    round1[3..].copy_from_slice(&round1_u2);
    let mut round2 = [F128::ZERO; 18];
    round2[..9].copy_from_slice(&round2_u0);
    round2[9..].copy_from_slice(&round2_u2);
    let mut round3 = [F128::ZERO; 54];
    round3[..27].copy_from_slice(&round3_u0);
    round3[27..].copy_from_slice(&round3_u2);
    let mut round4 = [F128::ZERO; 162];
    round4[..81].copy_from_slice(&round4_u0);
    round4[81..].copy_from_slice(&round4_u2);
    let mut round5 = [F128::ZERO; 486];
    round5[..243].copy_from_slice(&round5_u0);
    round5[243..].copy_from_slice(&round5_u2);

    (
        (round0_u0[0], round0_u2[0]),
        round1,
        round2,
        round3,
        round4,
        round5,
    )
}

/// Round-zero message from the factorized sixty-four-bank state. Each claim's
/// contribution is cached by its parallel ring-switch tail, so this step only
/// sums the tuples into the message for their combined 64x64 product matrix.
fn message_from_direct_factors_fold8(factors: &[ring_switch::DirectFold8Factors]) -> (F128, F128) {
    factors
        .iter()
        .map(|claim| {
            assert_eq!(claim.a_state.len(), (1 << LOG_PACKING) * 64);
            assert_eq!(claim.w_state.len(), claim.a_state.len());
            claim.round0
        })
        .fold((F128::ZERO, F128::ZERO), |(a0, a2), (b0, b2)| {
            (a0 + b0, a2 + b2)
        })
}

/// Exact ranked shape for the heterogeneous combined-basis queue. The gate is
/// deliberately narrower than the algebraic fast path: two ring-switched
/// BLAKE3 claims, no packed-direct claim, 2^25 packed slots split into 2^15
/// slot blocks (1024 independent 512 KiB jobs).
#[inline]
fn is_ranked_hetero_open_combine_shape(l: usize, b: usize, n_rs: usize, n_pd: usize) -> bool {
    l == (1usize << 25) && b == (1usize << 15) && n_rs == 2 && n_pd == 0
}

/// Exact ranked direct-fold2 materialization shapes. The first arm is the
/// frontier AB claim plus ordinary C basis; the second is deferred C encoded
/// as a second direct claim with no materialized ordinary basis.
#[inline]
fn is_ranked_direct_fold2_lookahead_shape(
    packed_len: usize,
    block_len: usize,
    claim_count: usize,
    has_ordinary: bool,
) -> bool {
    packed_len == (1usize << 25)
        && block_len == (1usize << 15)
        && ((claim_count == 1 && has_ordinary) || (claim_count == 2 && !has_ordinary))
}

#[inline]
fn use_ranked_hetero_open_combine(l: usize, b: usize, n_rs: usize, n_pd: usize) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && is_ranked_hetero_open_combine_shape(l, b, n_rs, n_pd)
        && std::env::var_os("FLOCK_NO_HETERO_OPEN_COMBINE").is_none()
        && crate::epool::epool().is_some()
}

/// Exact ranked direct-fold8 materializer shape: 2^25 packed slots folding
/// 64:1 into 2^19 outputs, two direct claims (AB + direct-C), no ordinary
/// basis. The deferred split (`deferred_split_n_lo(19) = 11`) makes the
/// materializer's blocks 2^11 slots, i.e. 256 independent jobs that each
/// read a disjoint 2 MiB witness stripe and write a disjoint 64 KiB output
/// pair — the same block-owns-its-range contract as the hetero combine
/// queue above.
#[inline]
fn is_ranked_open_mat_hetero_shape(
    packed_len: usize,
    block_len: usize,
    claim_count: usize,
    has_ordinary: bool,
) -> bool {
    packed_len == (1usize << 25) && block_len == (1usize << 11) && claim_count == 2 && !has_ordinary
}

/// Heterogeneous (P+E) drain for the direct-fold8 materializer — the single
/// 512 MiB witness pass of the ranked opening (2^19 slots × 64 banked muls
/// = 2^25 ≈ 33.5M deferred products, plus 2 × 2^19 byte-table basis folds).
/// The pass is product-dense, not bandwidth-starved: at the ranked shape the
/// P-cores retire pmull products far below memory saturation, which is the
/// same census that made the combined-basis queue
/// ([`use_ranked_hetero_open_combine`]) and the witgen drain
/// (`r1cs_hashes::common::run_hetero_chunks`) profitable on efficiency
/// cores — and the opposite of the sumcheck fold sweeps, whose byte/mul
/// ratio measured hetero-negative. Default **on** for the ranked worker;
/// `FLOCK_NO_OPEN_MAT_HETERO=1` (exact '1', per the grind-reg precedent)
/// restores the incumbent rayon-only pass. Both drains compute identical
/// per-block values and the block partials are XOR sums (order-free in
/// char 2), so the emitted message and outputs are bit-identical either way.
#[inline]
pub(crate) fn use_open_mat_hetero(
    packed_len: usize,
    block_len: usize,
    claim_count: usize,
    has_ordinary: bool,
) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && is_ranked_open_mat_hetero_shape(packed_len, block_len, claim_count, has_ordinary)
        && !std::env::var("FLOCK_NO_OPEN_MAT_HETERO").is_ok_and(|v| v == "1")
        && crate::epool::epool().is_some()
}

/// Drain fixed-size output blocks through the stateful P/E queue and reduce
/// one `(u0, u2)` partial per block after the synchronous join. `fold_block`
/// owns the complete output block for its queue index; `init` supplies private
/// worker scratch that persists across all blocks claimed by that worker.
fn run_hetero_open_combine_blocks<S, I, F>(
    out: &mut [F128],
    block_len: usize,
    init: I,
    fold_block: F,
) -> (F128, F128)
where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &mut [F128]) -> (F128, F128) + Sync,
{
    assert!(block_len > 0 && out.len().is_multiple_of(block_len));
    let n_blocks = out.len() / block_len;
    let mut partials = vec![(F128::ZERO, F128::ZERO); n_blocks];
    let out_base = crate::epool::SyncPtr(out.as_mut_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks_stateful(n_blocks, init, |state, block| {
        // SAFETY: queue index `block` is claimed exactly once. It owns disjoint
        // output `[block*block_len, (block+1)*block_len)` and one partial slot;
        // the synchronous two-pool join publishes both before reduce.
        unsafe {
            let out_block =
                core::slice::from_raw_parts_mut(out_base.ptr().add(block * block_len), block_len);
            partials_base
                .ptr()
                .add(block)
                .write(fold_block(state, block, out_block));
        }
    });
    partials
        .into_iter()
        .fold((F128::ZERO, F128::ZERO), |(x0, x2), (y0, y2)| {
            (x0 + y0, x2 + y2)
        })
}

/// Lookahead-bearing sibling of [`run_hetero_open_combine_blocks`]. Kept
/// separate so `FLOCK_NO_LIG_FOLD2` executes the frontier's exact pair-only
/// allocation, worker closure, and reduction path.
fn run_hetero_open_combine_blocks_lookahead<S, I, F>(
    out: &mut [F128],
    block_len: usize,
    init: I,
    fold_block: F,
) -> (F128, F128, [F128; 6])
where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &mut [F128]) -> (F128, F128, [F128; 6]) + Sync,
{
    assert!(block_len > 0 && out.len().is_multiple_of(block_len));
    let n_blocks = out.len() / block_len;
    let mut partials = vec![(F128::ZERO, F128::ZERO, [F128::ZERO; 6]); n_blocks];
    let out_base = crate::epool::SyncPtr(out.as_mut_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks_stateful(n_blocks, init, |state, block| {
        // SAFETY: as above, each claimed queue index owns one disjoint output
        // block and one partial slot until the synchronous two-pool join.
        unsafe {
            let out_block =
                core::slice::from_raw_parts_mut(out_base.ptr().add(block * block_len), block_len);
            partials_base
                .ptr()
                .add(block)
                .write(fold_block(state, block, out_block));
        }
    });
    partials.into_iter().fold(
        (F128::ZERO, F128::ZERO, [F128::ZERO; 6]),
        |(x0, x2, mut xc), (y0, y2, yc)| {
            for (x, y) in xc.iter_mut().zip(yc) {
                *x += y;
            }
            (x0 + y0, x2 + y2, xc)
        },
    )
}

/// Deferred-C sibling of [`run_hetero_open_combine_blocks_lookahead`]: no
/// L-sized output — each worker owns a private block-sized scratch buffer
/// that `fold_block` fully rewrites for every block it claims. Identical
/// per-block value sequence and block-indexed reduction, so the returned
/// prime + lookahead are bit-identical to the materializing variant.
fn run_hetero_open_combine_scratch_lookahead<S, I, F>(
    n_blocks: usize,
    block_len: usize,
    init: I,
    fold_block: F,
) -> (F128, F128, [F128; 6])
where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &mut [F128]) -> (F128, F128, [F128; 6]) + Sync,
{
    assert!(block_len > 0 && n_blocks > 0);
    let mut partials = vec![(F128::ZERO, F128::ZERO, [F128::ZERO; 6]); n_blocks];
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks_stateful(
        n_blocks,
        || (init(), vec![F128::ZERO; block_len]),
        |(state, scratch), block| {
            // SAFETY: queue index `block` is claimed exactly once; it owns
            // one partial slot until the synchronous two-pool join. The
            // scratch buffer is worker-private.
            unsafe {
                partials_base
                    .ptr()
                    .add(block)
                    .write(fold_block(state, block, scratch));
            }
        },
    );
    partials.into_iter().fold(
        (F128::ZERO, F128::ZERO, [F128::ZERO; 6]),
        |(x0, x2, mut xc), (y0, y2, yc)| {
            for (x, y) in xc.iter_mut().zip(yc) {
                *x += y;
            }
            (x0 + y0, x2 + y2, xc)
        },
    )
}

#[inline]
fn direct_ab_claim_mix_supported(
    rs_results: &[(RingSwitchProof, ring_switch::RingSwitchBatchOutput)],
) -> bool {
    matches!(
        rs_results,
        [(_, ab), (_, c)]
            if ab.direct_fold2.is_some()
                && c.direct_fold2.is_none()
                && matches!(&c.rs_eq_ind, ring_switch::RsEqInd::DeferredDense { .. })
    )
}

/// Completed direct path: **both** claims carry their own four-bank factor
/// bundle, so every round-0/round-1 statistic comes from `products` and the
/// L-sized combine sweep has nothing left to compute.
#[inline]
fn direct_all_claim_mix_supported(
    rs_results: &[(RingSwitchProof, ring_switch::RingSwitchBatchOutput)],
) -> bool {
    matches!(
        rs_results,
        [(_, ab), (_, c)]
            if ab.direct_fold2.is_some()
                && c.direct_fold2.is_some()
                && matches!(&ab.rs_eq_ind, ring_switch::RsEqInd::DeferredDense { .. })
                && matches!(&c.rs_eq_ind, ring_switch::RsEqInd::DeferredDense { .. })
    )
}

/// Experimental sixteen-bank route: both ranked claims must expose a complete
/// direct-fold4 bundle, so no ordinary basis sweep or duplicate statistics
/// path is needed.
#[inline]
fn direct_fold4_all_claim_mix_supported(
    rs_results: &[(RingSwitchProof, ring_switch::RingSwitchBatchOutput)],
) -> bool {
    matches!(
        rs_results,
        [(_, ab), (_, c)]
            if ab.direct_fold4.is_some()
                && c.direct_fold4.is_some()
                && matches!(&ab.rs_eq_ind, ring_switch::RsEqInd::DeferredDense { .. })
                && matches!(&c.rs_eq_ind, ring_switch::RsEqInd::DeferredDense { .. })
    )
}

/// Direct-fold8 route: both ranked claims must expose a complete
/// direct-fold8 bundle, so no ordinary basis sweep or duplicate statistics
/// path is needed.
#[inline]
fn direct_fold8_all_claim_mix_supported(
    rs_results: &[(RingSwitchProof, ring_switch::RingSwitchBatchOutput)],
) -> bool {
    matches!(
        rs_results,
        [(_, ab), (_, c)]
            if ab.direct_fold8.is_some()
                && c.direct_fold8.is_some()
                && matches!(&ab.rs_eq_ind, ring_switch::RsEqInd::DeferredDense { .. })
                && matches!(&c.rs_eq_ind, ring_switch::RsEqInd::DeferredDense { .. })
    )
}

/// Direct-fold4 enable, latched once per process.
///
/// Default **enabled** for the ranked worker. `FLOCK_NO_OPEN_DIRECT_FOLD4=1`
/// restores the previous direct-C route bit-for-bit. The retained-coordinate
/// producers and this consumer share this predicate so they cannot silently
/// disagree.
#[inline]
pub fn ranked_direct_fold4_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_OPEN_DIRECT_FOLD4").is_none())
}

/// Direct-fold8 enable, latched once per process.
///
/// Default **enabled** for the ranked worker on top of DirectFold4.
/// `FLOCK_NO_OPEN_DIRECT_FOLD8=1` restores the exact incumbent fold4 route;
/// the fold4 kill switch also disables fold8 (fold8 is a strict widening of
/// the fold4 chain). The stripe-C/AB producers and this consumer share this
/// predicate so they cannot silently disagree.
#[inline]
pub fn ranked_direct_fold8_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        ranked_direct_fold4_enabled() && std::env::var_os("FLOCK_NO_OPEN_DIRECT_FOLD8").is_none()
    });
    *ON
}

/// Direct-C completion kill switch, latched once per process.
///
/// Default **enabled**: the ranked worker's environment is cleared down to
/// `RAYON_NUM_THREADS` + `TMPDIR`, so an env opt-out never reaches it and
/// default-on is the shipped behaviour. `FLOCK_NO_OPEN_DIRECT_C=1` restores the
/// deferred-C path bit-for-bit in the same binary. Read by both the zerocheck-side
/// quad handoff (`flock_prover::prover`) and `use_direct_all` below, so the
/// capture and its consumer can never disagree; the structural predicate above
/// stays the final authority, so a mismatch degrades instead of panicking.
#[inline]
pub fn ranked_direct_c_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_OPEN_DIRECT_C").is_none())
}

/// Runs ring_switch over RS claims, observes packed-direct claim values +
/// samples their gammas, then builds `b_combined` (the γ-weighted linear
/// combination of all `rs_eq_ind`s and `eq_ind`s) and `target_combined`.
/// Also computes the round-0 prime as a side effect (cheap since it shares
/// the b_combined pass).
#[allow(clippy::too_many_arguments)]
fn compute_combined_basis_and_target<Ch: Challenger>(
    packed_witness: &[F128],
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    challenger: &mut Ch,
    trace: bool,
    enable_fold2: bool,
) -> CombinedClaim {
    let n_rs = x_outers.len();
    let n_pd = packed_direct.len();
    assert!(n_rs + n_pd > 0, "open_batch_mixed: need at least one claim");
    assert!(
        precomputed_s_hat_v.is_empty() || precomputed_s_hat_v.len() == n_rs,
        "precomputed_s_hat_v: must be empty or length {n_rs}, got {}",
        precomputed_s_hat_v.len(),
    );

    challenger.observe_label(b"flock-pcs-open-batch-v0");

    // 1. Ring-switching for all x_outers.
    let t = std::time::Instant::now();
    let (mut rs_results, gammas_rs): (
        Vec<(RingSwitchProof, ring_switch::RingSwitchBatchOutput)>,
        Vec<F128>,
    ) = if n_rs > 0 {
        ring_switch::prove_batched_padded_with_precomputed(
            packed_witness,
            x_outers,
            precomputed_s_hat_v,
            padding,
            challenger,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    if trace {
        eprintln!(
            "  [open_batch] ring_switch::prove_batched ×{}: {:6.2} ms",
            n_rs,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 2. Observe packed-direct claim values + sample γ_pd.
    for pd in packed_direct {
        challenger.observe_label(b"flock-pcs-packed-direct-v0");
        challenger.observe_f128(pd.value);
    }
    let gammas_pd: Vec<F128> = (0..n_pd).map(|_| challenger.sample_f128()).collect();

    let t = std::time::Instant::now();
    use rayon::prelude::*;

    let l = if let Some((_, out)) = rs_results.first() {
        out.rs_eq_ind.len()
    } else {
        1usize << packed_direct[0].point.len()
    };
    debug_assert!(rs_results.iter().all(|(_, o)| o.rs_eq_ind.len() == l));
    debug_assert!(
        packed_direct.iter().all(|pd| 1usize << pd.point.len() == l),
        "all packed-direct claims must share L (= packed witness length)"
    );

    let mut target_combined = F128::ZERO;
    for ((_, output), g) in rs_results.iter().zip(gammas_rs.iter()) {
        target_combined += *g * output.sumcheck_claim;
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        target_combined += *g * pd.value;
    }
    // The ranked direct path takes claim 0's four-bank sufficient statistic
    // unconditionally; when claim 1 (C) also carries one, `use_direct_all`
    // takes it too and no claim is left on the basis sweep. Either opt-out
    // restores the corresponding incumbent implementation.
    let direct_common = enable_fold2
        && l == (1usize << 25)
        && n_rs == 2
        && n_pd == 0
        && std::env::var_os("FLOCK_NO_OPEN_DIRECT_AB").is_none();
    let use_direct_fold8 = direct_common
        && ranked_direct_fold8_enabled()
        && direct_fold8_all_claim_mix_supported(&rs_results);
    let use_direct_fold4 = !use_direct_fold8
        && direct_common
        && ranked_direct_fold4_enabled()
        && direct_fold4_all_claim_mix_supported(&rs_results);
    let use_direct_all = !use_direct_fold4
        && direct_common
        && ranked_direct_c_enabled()
        && direct_all_claim_mix_supported(&rs_results);
    let use_direct_ab =
        direct_common && (use_direct_all || direct_ab_claim_mix_supported(&rs_results));
    let direct_fold8 = if use_direct_fold8 {
        Some(vec![
            rs_results[0]
                .1
                .direct_fold8
                .take()
                .expect("direct-fold8 gate checked claim zero"),
            rs_results[1]
                .1
                .direct_fold8
                .take()
                .expect("direct-fold8 gate checked claim one"),
        ])
    } else {
        None
    };
    let direct_fold4 = if use_direct_fold4 {
        Some(vec![
            rs_results[0]
                .1
                .direct_fold4
                .take()
                .expect("direct-fold4 gate checked claim zero"),
            rs_results[1]
                .1
                .direct_fold4
                .take()
                .expect("direct-fold4 gate checked claim one"),
        ])
    } else {
        None
    };
    let mut direct_fold2 = if use_direct_all {
        Some(vec![
            rs_results[0]
                .1
                .direct_fold2
                .take()
                .expect("direct-all gate checked claim zero"),
            rs_results[1]
                .1
                .direct_fold2
                .take()
                .expect("direct-all gate checked claim one"),
        ])
    } else if use_direct_ab {
        Some(vec![
            rs_results[0]
                .1
                .direct_fold2
                .take()
                .expect("direct AB gate checked claim zero"),
        ])
    } else {
        None
    };
    let direct_count = direct_fold8.as_ref().map_or_else(
        || {
            direct_fold4
                .as_ref()
                .map_or_else(|| direct_fold2.as_ref().map_or(0, Vec::len), Vec::len)
        },
        Vec::len,
    );
    // Deferred-C candidate: taken here, before `rs_baked`/`rs_deferred`
    // borrow `rs_results`; confirmed below once the ranked sweep shape is
    // known (dropped — incumbent path — otherwise). Mutually exclusive with
    // the completed path, which needs no sweep at all.
    let mut deferred_c_candidate = if use_direct_ab
        && !use_direct_all
        && std::env::var_os("FLOCK_NO_OPEN_DEFERRED_C").is_none()
    {
        rs_results[1].1.deferred_c_fold2.take()
    } else {
        None
    };

    let rs_baked: Vec<&[F128]> = rs_results
        .iter()
        .enumerate()
        .filter_map(|(index, (_, output))| {
            // Direct claims are always taken from the front (0, then 1).
            if index < direct_count {
                return None;
            }
            match &output.rs_eq_ind {
                ring_switch::RsEqInd::Dense(values) => Some(values.as_slice()),
                _ => None,
            }
        })
        .collect();
    // Deferred-dense claims (fused fast path): the per-claim `γ_k·B_k` buffer
    // was never materialized — fold each slot on the fly below and accumulate
    // straight into `b_combined`, saving a 2^(m-7) materialize + readback per
    // claim. Carries (eq_lo, eq_hi, γ-baked table, log₂ B).
    let rs_deferred: Vec<(&[F128], &[F128], &[F128], usize)> = rs_results
        .iter()
        .enumerate()
        .filter_map(|(index, (_, output))| {
            // Direct claims are always taken from the front (0, then 1).
            if index < direct_count {
                return None;
            }
            match &output.rs_eq_ind {
                ring_switch::RsEqInd::DeferredDense {
                    eq_lo,
                    eq_hi,
                    table,
                } => Some((
                    eq_lo.as_slice(),
                    eq_hi.as_slice(),
                    table.as_slice(),
                    eq_lo.len().trailing_zeros() as usize,
                )),
                _ => None,
            }
        })
        .collect();
    let pd_dense: Vec<(&[F128], F128)> = packed_direct
        .iter()
        .zip(gammas_pd.iter())
        .filter_map(|(pd, g)| match &pd.eq_ind {
            DirectEqInd::Dense(v) => Some((v.as_slice(), *g)),
            _ => None,
        })
        .collect();

    // Fast path (compression-proof open: claims ab, c; also chain/merkle): every
    // RS claim is a fused DeferredDense fold and no DENSE packed-direct claim
    // needs the per-element combine. Fold all claims block-by-block straight into
    // b_combined — each claim's `e_hi` hoisted once per block, exactly as in
    // `fold_b128_elems_split` — and fuse the round-0 prime in the same pass.
    // Neither the per-claim `γ_k·B_k` buffer nor a combine readback is ever
    // materialized (saves ~2·L writes + 2·L reads of the 2^(m-7) basis).
    //
    // SPARSE packed-direct claims (the chain/merkle I/O claim) do NOT disable
    // this path: they're scatter-added onto b_combined after the fold (with an
    // incremental round-0 prime adjustment), so they only require
    // `pd_dense.is_empty()`, not `packed_direct.is_empty()`. This keeps the two
    // big ab/c claims on the fused fold instead of materializing them.
    let use_fast = !rs_deferred.is_empty()
        && rs_deferred.len() + direct_count == rs_results.len()
        && pd_dense.is_empty();

    let want_round1_lookahead = use_fast && enable_fold2;

    // Deferred-C: on the ranked direct-AB shape, when C also carries its
    // factor bundle, `b_combined` is never materialized — the combine sweep
    // below writes per-worker scratch blocks (same per-block value sequence,
    // same block-indexed reduction, so the prime + lookahead come out
    // bit-identical) and C joins AB as a second direct claim at materialize
    // time. `FLOCK_NO_OPEN_DEFERRED_C=1` restores the incumbent path.
    let deferred_c = if want_round1_lookahead
        && use_ranked_hetero_open_combine(l, rs_deferred[0].0.len(), n_rs, n_pd)
    {
        deferred_c_candidate.take()
    } else {
        None
    };

    let use_deferred_c = deferred_c.is_some();

    // ---- Build b_combined (γ-weighted sum of all rs_eq_ind + eq_ind) and the
    //      round-0 prime (u_0, u_2 over packed_witness · b_combined).
    let t_alloc = std::time::Instant::now();
    let mut b_combined: Vec<F128> =
        if use_deferred_c || use_direct_all || use_direct_fold4 || use_direct_fold8 {
            Vec::new()
        } else {
            crate::scratch::take_f128(l)
        };
    let alloc_ms = t_alloc.elapsed().as_secs_f64() * 1e3;
    let t_fold = std::time::Instant::now();
    let (mut round0_u0, mut round0_u2, round1_lookahead) = if use_direct_fold8 {
        // Fold8's round-zero message comes from its factor state below. Later
        // messages are derived online after their preceding challenges.
        (F128::ZERO, F128::ZERO, None)
    } else if use_direct_fold4 {
        // All four initial messages come from the two claims' 16x16 product
        // matrices below; no L-sized basis exists to sweep.
        (F128::ZERO, F128::ZERO, Some([F128::ZERO; 6]))
    } else if use_direct_all {
        // Every claim's round-0 and round-1 contribution comes from its own
        // `products` (added below, from `messages_from_direct_products`), so
        // there is no basis to sweep: no L-sized allocation, no per-slot fold,
        // no witness streaming pass. Seeding the lookahead with zeros — rather
        // than `None` — is what keeps the accumulate below a plain add.
        (F128::ZERO, F128::ZERO, Some([F128::ZERO; 6]))
    } else if use_fast {
        let b = rs_deferred[0].0.len(); // eq_lo.len(); shared across claims (same split)
        debug_assert!(b >= 2 && b.is_multiple_of(2));
        debug_assert!(rs_deferred.iter().all(|d| d.0.len() == b));
        // Composed-table sweep. `fold_one_slot(·, T)` is F₂-linear, so the
        // per-slot map `lo ↦ fold_one_slot(lo·e_hi, T)` collapses into ONE
        // per-claim-per-block byte table (`compose_fold_byte_table_into`),
        // deleting the per-slot field multiply from the L-sized sweep —
        // 1 GF mul × L × n_claims of counted work — for a per-block table
        // build (~4.3k ops) amortized over b slots. Bit-identical: the
        // composed table encodes exactly the same F₂-linear map (see the
        // helper's docs). One 64 KiB table is live per thread at a time
        // (claims composed and swept sequentially — table reused in place),
        // preserving the claim-sequential L1 footprint.
        //
        // Small blocks (tiny test shapes) keep the direct slot-multiply
        // sweep: the table build wouldn't amortize below ~2^12 slots.
        const COMPOSE_MIN_BLOCK: usize = 1 << 12;
        let composed = b >= COMPOSE_MIN_BLOCK;
        let fill_block = |ctable: &mut Vec<F128>, hi: usize, out_block: &mut [F128]| {
            // Accumulate each claim's block: first claim writes, rest add.
            // `e_hi` is folded into the composed table once per claim per
            // block, then swept over eq_lo with no per-slot multiply.
            for (ci, (eq_lo, eq_hi, table, _)) in rs_deferred.iter().enumerate() {
                let e_hi = eq_hi[hi];
                if composed {
                    ring_switch::compose_fold_byte_table_into(e_hi, table, ctable);
                    if ci == 0 {
                        for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                            *slot = ring_switch::fold_one_slot(lo, ctable);
                        }
                    } else {
                        for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                            *slot += ring_switch::fold_one_slot(lo, ctable);
                        }
                    }
                } else if ci == 0 {
                    for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                        *slot = ring_switch::fold_one_slot(lo * e_hi, table);
                    }
                } else {
                    for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                        *slot += ring_switch::fold_one_slot(lo * e_hi, table);
                    }
                }
            }
        };
        let fold_pair_block = |ctable: &mut Vec<F128>, hi: usize, out_block: &mut [F128]| {
            fill_block(ctable, hi, out_block);
            // Round-0 prime over this block's pairs (b is even, base is even).
            // Keep this separate from the last claim's streaming store: the
            // pairwise stepping variant is slower even though it removes a pass.
            let base = hi * b;
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            for t in 0..(b / 2) {
                let s0 = out_block[2 * t];
                let s1 = out_block[2 * t + 1];
                let a0 = packed_witness[base + 2 * t];
                let a1 = packed_witness[base + 2 * t + 1];
                u0 += a0 * s0;
                u2 += (a0 + a1) * (s0 + s1);
            }
            (u0, u2)
        };
        let ranked_lookahead_neon =
            enable_fold2 && is_ranked_hetero_open_combine_shape(l, b, n_rs, n_pd);
        let fold_lookahead_block = |ctable: &mut Vec<F128>, hi: usize, out_block: &mut [F128]| {
            fill_block(ctable, hi, out_block);
            let base = hi * b;
            debug_assert!(b.is_multiple_of(4));
            let ((u0, u2), lookahead) = round0_and_round1_lookahead_ranked(
                &packed_witness[base..base + b],
                out_block,
                ranked_lookahead_neon,
            );
            (u0, u2, lookahead)
        };
        let init_ctable = || {
            if composed {
                vec![F128::ZERO; ring_switch::FOLD_TABLE_LEN]
            } else {
                Vec::new()
            }
        };

        let ranked_hetero = use_ranked_hetero_open_combine(l, b, n_rs, n_pd);
        if want_round1_lookahead {
            let fast = if deferred_c.is_some() {
                run_hetero_open_combine_scratch_lookahead(
                    l / b,
                    b,
                    init_ctable,
                    |ctable, hi, out_block| fold_lookahead_block(ctable, hi, out_block),
                )
            } else if ranked_hetero {
                run_hetero_open_combine_blocks_lookahead(
                    &mut b_combined,
                    b,
                    init_ctable,
                    |ctable, hi, out_block| fold_lookahead_block(ctable, hi, out_block),
                )
            } else {
                b_combined
                    .par_chunks_mut(b)
                    .enumerate()
                    .map_init(init_ctable, |ctable, (hi, out_block)| {
                        fold_lookahead_block(ctable, hi, out_block)
                    })
                    .reduce(
                        || (F128::ZERO, F128::ZERO, [F128::ZERO; 6]),
                        |(x0, x2, mut xc), (y0, y2, yc)| {
                            for (x, y) in xc.iter_mut().zip(yc) {
                                *x += y;
                            }
                            (x0 + y0, x2 + y2, xc)
                        },
                    )
            };
            (fast.0, fast.1, Some(fast.2))
        } else {
            // Exact opt-out: retain the frontier's pair-only worker payload,
            // allocation, map_init ownership, and reduction.
            let fast = if ranked_hetero {
                run_hetero_open_combine_blocks(
                    &mut b_combined,
                    b,
                    init_ctable,
                    |ctable, hi, out_block| fold_pair_block(ctable, hi, out_block),
                )
            } else {
                b_combined
                    .par_chunks_mut(b)
                    .enumerate()
                    .map_init(init_ctable, |ctable, (hi, out_block)| {
                        fold_pair_block(ctable, hi, out_block)
                    })
                    .reduce(
                        || (F128::ZERO, F128::ZERO),
                        |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
                    )
            };
            (fast.0, fast.1, None)
        }
    } else {
        // General path (mixed / sparse / packed-direct): materialize any
        // deferred-dense claims (parallel block fold), then the per-element
        // combine over all dense buffers + packed-direct, matching the
        // original behavior.
        let materialized: Vec<Vec<F128>> = rs_results
            .iter()
            .filter_map(|(_, o)| match &o.rs_eq_ind {
                ring_switch::RsEqInd::DeferredDense {
                    eq_lo,
                    eq_hi,
                    table,
                } => Some(ring_switch::fold_b128_from_table(eq_lo, eq_hi, table)),
                _ => None,
            })
            .collect();
        let mut rs_dense_all: Vec<&[F128]> = rs_baked.clone();
        rs_dense_all.extend(materialized.iter().map(|v| v.as_slice()));
        let prime = b_combined
            .par_chunks_mut(2)
            .enumerate()
            .map(|(i, chunk)| {
                let mut b0 = F128::ZERO;
                let mut b1 = F128::ZERO;
                for v in rs_dense_all.iter() {
                    b0 += v[2 * i];
                    b1 += v[2 * i + 1];
                }
                for (v, g) in pd_dense.iter() {
                    b0 += *g * v[2 * i];
                    b1 += *g * v[2 * i + 1];
                }
                chunk[0] = b0;
                chunk[1] = b1;
                let a0 = packed_witness[2 * i];
                let a1 = packed_witness[2 * i + 1];
                (a0 * b0, (a0 + a1) * (b0 + b1))
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
            );
        for v in materialized {
            crate::scratch::give_f128(v);
        }
        (prime.0, prime.1, None)
    };
    let mut round1_lookahead = round1_lookahead;
    let mut round2_lookahead = None;
    let mut round3_lookahead = None;
    if let Some(direct) = direct_fold8.as_ref() {
        let direct_round0 = message_from_direct_factors_fold8(direct);
        round0_u0 += direct_round0.0;
        round0_u2 += direct_round0.1;
    }
    if let Some(direct) = direct_fold4.as_ref() {
        let (direct_round0, direct_round1, direct_round2, direct_round3) =
            messages_from_direct_products_fold4(direct);
        round0_u0 += direct_round0.0;
        round0_u2 += direct_round0.1;
        let combined_round1 = round1_lookahead
            .as_mut()
            .expect("direct-fold4 gate requires round-1 lookahead storage");
        for (out, value) in combined_round1.iter_mut().zip(direct_round1) {
            *out += value;
        }
        round2_lookahead = Some(direct_round2);
        round3_lookahead = Some(direct_round3);
    }
    if let Some(direct) = direct_fold2.as_ref() {
        let (direct_round0, direct_lookahead) = messages_from_direct_products(direct);
        round0_u0 += direct_round0.0;
        round0_u2 += direct_round0.1;
        let combined_lookahead = round1_lookahead
            .as_mut()
            .expect("direct fold2 gate requires a round-1 lookahead");
        for (out, value) in combined_lookahead.iter_mut().zip(direct_lookahead) {
            *out += value;
        }
    }
    if let Some(c) = deferred_c {
        // C's round-0/1 contribution already came from the sweep above; its
        // `products` are zeroed by construction, so it joins only the
        // materialize-time claims.
        direct_fold2
            .as_mut()
            .expect("deferred-C requires the direct AB bundle")
            .push(c);
    }
    let fold_ms = t_fold.elapsed().as_secs_f64() * 1e3;
    let t_sparse = std::time::Instant::now();
    let mut adjust_prime_for_delta = |idx: usize, delta: F128| {
        let pair = idx / 2;
        let a0 = packed_witness[2 * pair];
        let a1 = packed_witness[2 * pair + 1];
        if idx & 1 == 0 {
            round0_u0 += a0 * delta;
        }
        round0_u2 += (a0 + a1) * delta;
    };
    for (_, output) in rs_results.iter() {
        if let ring_switch::RsEqInd::Sparse { entries, .. } = &output.rs_eq_ind {
            round1_lookahead = None;
            for &(idx, val) in entries {
                b_combined[idx] += val;
                adjust_prime_for_delta(idx, val);
            }
        }
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        if let DirectEqInd::Sparse(eq) = &pd.eq_ind {
            round1_lookahead = None;
            // Scatter-add the sparse claim and fold its round-0 prime
            // contribution in the SAME pass (O(live positions)), instead of a
            // full O(L) re-pass over b_combined. The prime is linear in
            // b_combined, so the delta from scattering `g·eq` equals
            // Σ adjust_prime_for_delta(idx, g·val) over the live positions.
            let (du0, du2) = sparse_scatter_add_parallel(&mut b_combined, packed_witness, eq, *g);
            round0_u0 += du0;
            round0_u2 += du2;
        }
    }
    if trace {
        eprintln!(
            "  [open_batch] combine rs_eq_ind (L={}, rs×{}, pd×{}, fast={}, deferred_c={}): alloc {:6.2} ms, fold+prime {:6.2} ms, sparse {:6.2} ms, total {:6.2} ms",
            l,
            n_rs,
            n_pd,
            use_fast,
            use_deferred_c,
            alloc_ms,
            fold_ms,
            t_sparse.elapsed().as_secs_f64() * 1e3,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    CombinedClaim {
        ring_switches: rs_results
            .into_iter()
            .map(|(p, o)| {
                // The per-claim rs_eq_ind (L F128s) dies here — recycle it.
                if let ring_switch::RsEqInd::Dense(v) = o.rs_eq_ind {
                    crate::scratch::give_f128(v);
                }
                p
            })
            .collect(),
        b_combined,
        target_combined,
        round0_prime: (round0_u0, round0_u2),
        round1_lookahead,
        round2_lookahead,
        round3_lookahead,
        direct_fold2,
        direct_fold4,
        direct_fold8,
    }
}

/// Parallel sparse scatter-add: `b_combined[scatter_idx(c)] += gamma * eq.live_tensor[c]`
/// for every `c`. Partitions `c`-space across rayon threads; since
/// [`SparseEqTensor::scatter_idx`] is monotonic in `c` (live_positions sorted
/// ascending), each thread's scattered indices fall in a contiguous, disjoint
/// range of `b_combined`. Splits `b_combined` at the chunk boundaries via
/// `split_at_mut`, then writes scatter-adds into the disjoint mutable slices —
/// safe rust, no atomics.
/// Scatter-add `gamma · eq` into `b_combined` and return the resulting
/// round-0 prime delta `(Δu0, Δu2)`. Because the prime is linear in
/// `b_combined`, adding `delta = gamma·val` at index `idx` changes the prime by
/// `Δu0 += a0·delta` (if `idx` even) and `Δu2 += (a0+a1)·delta`, where
/// `a0 = packed_witness[2·pair]`, `a1 = packed_witness[2·pair+1]`,
/// `pair = idx/2`. Computing it here (O(live positions)) avoids a full O(L)
/// re-pass over `b_combined` at the call site.
fn sparse_scatter_add_parallel(
    b_combined: &mut [F128],
    packed_witness: &[F128],
    eq: &SparseEqTensor,
    gamma: F128,
) -> (F128, F128) {
    use rayon::prelude::*;

    let c_total = eq.live_tensor.len();
    if c_total == 0 {
        return (F128::ZERO, F128::ZERO);
    }
    let n_threads = rayon::current_num_threads().max(1);
    let c_per_chunk = c_total.div_ceil(n_threads).max(1);
    let actual_n_chunks = c_total.div_ceil(c_per_chunk);

    // Boundaries in `b_combined` index space. `b_boundaries[i]` is where chunk
    // `i` starts. `b_boundaries[i+1] − b_boundaries[i]` is chunk `i`'s slice
    // length. The last chunk extends to `b_combined.len()` to absorb any tail
    // positions beyond the maximum scatter idx (those contain only dense
    // contributions from the parallel pass).
    let b_boundaries: Vec<usize> = (0..=actual_n_chunks)
        .map(|i| {
            if i == 0 {
                0
            } else if i == actual_n_chunks {
                b_combined.len()
            } else {
                eq.scatter_idx(i * c_per_chunk)
            }
        })
        .collect();
    debug_assert!(b_boundaries.windows(2).all(|w| w[0] <= w[1]));

    // Disjoint mutable slices via repeated split_at_mut.
    let mut remaining: &mut [F128] = b_combined;
    let mut slices: Vec<&mut [F128]> = Vec::with_capacity(actual_n_chunks);
    for i in 1..actual_n_chunks {
        let split_at = b_boundaries[i] - b_boundaries[i - 1];
        let (left, right) = remaining.split_at_mut(split_at);
        slices.push(left);
        remaining = right;
    }
    slices.push(remaining);
    debug_assert_eq!(slices.len(), actual_n_chunks);

    slices
        .into_par_iter()
        .enumerate()
        .map(|(t, slice)| {
            let c_lo = t * c_per_chunk;
            let c_hi = ((t + 1) * c_per_chunk).min(c_total);
            let b_lo = b_boundaries[t];
            let mut du0 = F128::ZERO;
            let mut du2 = F128::ZERO;
            for c in c_lo..c_hi {
                let val = eq.live_tensor[c];
                let idx = eq.scatter_idx(c);
                let delta = gamma * val;
                slice[idx - b_lo] += delta;
                // Round-0 prime delta for this scattered position.
                let pair = idx / 2;
                let a0 = packed_witness[2 * pair];
                let a1 = packed_witness[2 * pair + 1];
                if idx & 1 == 0 {
                    du0 += a0 * delta;
                }
                du2 += (a0 + a1) * delta;
            }
            (du0, du2)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        )
}

/// Verifier reference to a packed-direct claim: the multilinear point at
/// which `ẑ_packed` was claimed equal to `value`. The verifier owns the data
/// (it appears in the public statement of whatever produced the claim, e.g.
/// the chain shift sumcheck output).
#[derive(Clone, Copy, Debug)]
pub struct PackedDirectClaimRef<'a> {
    pub point: &'a [F128],
    pub value: F128,
}

/// Verify a mixed-claim batched opening (mirror of
/// [`open_batch_mixed_ligerito_with_precomputed_s_hat_v`]). Uses
/// `ring_switch::verify_succinct` per claim (no dense `rs_eq_ind`
/// materialization), then drives the succinct recursive Ligerito verifier,
/// evaluating the combined basis only at the residual point.
#[allow(clippy::too_many_arguments)]
pub fn verify_opening_batch_ligerito_mixed<Ch: Challenger>(
    commitment: &Commitment,
    claims: &[F128],
    z_skips: &[F128],
    x_outers: &[&[F128]],
    packed_direct: &[PackedDirectClaimRef<'_>],
    proof: &BatchOpeningProofLigerito,
    lig_config: &ligerito::VerifierConfig,
    challenger: &mut Ch,
) -> Result<(), VerifyError> {
    let n_rs = claims.len();
    let n_pd = packed_direct.len();
    assert_eq!(z_skips.len(), n_rs);
    assert_eq!(x_outers.len(), n_rs);
    assert_eq!(proof.ring_switches.len(), n_rs);
    assert!(n_rs + n_pd > 0);

    challenger.observe_label(b"flock-pcs-open-batch-v0");

    // 1. Ring-switch SUCCINCT verify per claim — gets sumcheck_claim and a
    //    length-128 `eq_r_dprime` instead of the dense `rs_eq_ind`. Saves
    //    ~16 MB allocation at m=29.
    let mut rs_outputs = Vec::with_capacity(n_rs);
    for i in 0..n_rs {
        let out = ring_switch::verify_succinct(
            claims[i],
            z_skips[i],
            x_outers[i],
            &proof.ring_switches[i],
            challenger,
        )
        .map_err(VerifyError::RingSwitch)?;
        rs_outputs.push(out);
    }
    let gammas_rs: Vec<F128> = (0..n_rs).map(|_| challenger.sample_f128()).collect();

    // 2. PD claim values + γ_pd.
    for pd in packed_direct {
        challenger.observe_label(b"flock-pcs-packed-direct-v0");
        challenger.observe_f128(pd.value);
    }
    let gammas_pd: Vec<F128> = (0..n_pd).map(|_| challenger.sample_f128()).collect();

    // 3. target_combined from succinct rs claims + PD values.
    let mut target_combined = F128::ZERO;
    for (out, g) in rs_outputs.iter().zip(gammas_rs.iter()) {
        target_combined += *g * out.sumcheck_claim;
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        target_combined += *g * pd.value;
    }

    // 4. Batch evaluator: returns b_combined at all yr positions in one call.
    //    For RS claims, precompute the ring_switch tensor PREFIX once (over
    //    the ris part) and only re-do the yr_log_n-step suffix per y.
    //    For PD claims, precompute eq prefix factors over ris and finish per y.
    //    For BLAKE3 m=30: ris is 19 dims, yr is 4 dims → 19× prefix reuse.
    let log_n = commitment.params.m - LOG_PACKING;
    let eval_b_residual = |ris: &[F128], yr_log_n: usize| -> Vec<F128> {
        use crate::zerocheck::multilinear::eq_eval;
        let yr_len = 1usize << yr_log_n;
        let prefix_len = ris.len();

        // ---- RS claim prefixes ----
        let rs_prefixes: Vec<crate::pcs::tensor_algebra::TensorAlgebra> = rs_outputs
            .iter()
            .zip(x_outers.iter())
            .map(|(_out, x_outer)| {
                // x_outer[1..] has length log_n; we feed only the ris prefix.
                ring_switch::eval_rs_eq_prefix(&x_outer[1..1 + prefix_len], ris)
            })
            .collect();

        // ---- PD claim prefix scalars ----
        // eq(pd.point, point) factors over coordinates; precompute the prefix product.
        let pd_prefix_scalars: Vec<F128> = packed_direct
            .iter()
            .map(|pd| eq_eval(&pd.point[..prefix_len], ris))
            .collect();

        // ---- Per-y assembly (parallel over yr positions; each y is independent).
        //      y_suffix is binary (bits of y), so we use the binary-query
        //      specializations of eval_rs_eq_finish / eq_eval — each suffix
        //      step collapses to a single scale_vertical / scalar product.
        use rayon::prelude::*;
        debug_assert!(yr_log_n <= 32, "yr_log_n > 32 not supported by binary path");
        (0..yr_len)
            .into_par_iter()
            .map(|y| {
                let y_bits = y as u32;
                let mut sum = F128::ZERO;
                for (((out, g), x_outer), prefix) in rs_outputs
                    .iter()
                    .zip(gammas_rs.iter())
                    .zip(x_outers.iter())
                    .zip(rs_prefixes.iter())
                {
                    sum += *g
                        * ring_switch::eval_rs_eq_finish_from_prefix_binary_q(
                            prefix,
                            &x_outer[1 + prefix_len..],
                            y_bits,
                            &out.eq_r_dprime,
                        );
                }
                for ((pd, g), prefix_scalar) in packed_direct
                    .iter()
                    .zip(gammas_pd.iter())
                    .zip(pd_prefix_scalars.iter())
                {
                    sum += *g
                        * *prefix_scalar
                        * crate::zerocheck::multilinear::eq_eval_binary_x(
                            &pd.point[prefix_len..],
                            y_bits,
                        );
                }
                sum
            })
            .collect()
    };

    // 5. Drive ligerito SUCCINCT verifier — eval_b_residual is called ONCE
    //    at the residual check (returns all yr_len values in one batch).
    let ok = ligerito::recursive_verifier_with_basis_succinct(
        lig_config,
        &proof.ligerito,
        log_n,
        target_combined,
        &commitment.root,
        eval_b_residual,
        challenger,
    );
    if !ok {
        return Err(VerifyError::Ligerito);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;
    use crate::zerocheck::multilinear::lagrange_weights_naive;
    use crate::zerocheck::univariate_skip::build_eq;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    #[test]
    fn ranked_hetero_open_combine_shape_gate_is_narrow() {
        assert!(is_ranked_hetero_open_combine_shape(1 << 25, 1 << 15, 2, 0));
        assert!(!is_ranked_hetero_open_combine_shape(1 << 24, 1 << 15, 2, 0));
        assert!(!is_ranked_hetero_open_combine_shape(1 << 25, 1 << 14, 2, 0));
        assert!(!is_ranked_hetero_open_combine_shape(1 << 25, 1 << 15, 1, 0));
        assert!(!is_ranked_hetero_open_combine_shape(1 << 25, 1 << 15, 2, 1));
    }

    #[test]
    fn direct_fold8_cached_round0_matches_two_fresh_claim_scans() {
        let mut rng = Rng::new(0xD1CE_CACE);
        let factors: Vec<_> = (0..2)
            .map(|_| {
                let len = (1usize << LOG_PACKING) * 64;
                let a_state: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
                let w_state: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
                let round0 = round0_deferred(&a_state, &w_state);
                ring_switch::DirectFold8Factors {
                    eq_lo: vec![F128::ONE],
                    eq_hi: vec![F128::ONE],
                    a_state,
                    w_state,
                    round0,
                }
            })
            .collect();
        let expected = factors
            .iter()
            .map(|claim| round0_deferred(&claim.a_state, &claim.w_state))
            .fold((F128::ZERO, F128::ZERO), |(a0, a2), (b0, b2)| {
                (a0 + b0, a2 + b2)
            });

        assert_eq!(message_from_direct_factors_fold8(&factors), expected);
    }

    #[test]
    fn direct_ab_gate_rejects_sparse_c_without_consuming_ab_state() {
        let direct = || ring_switch::DirectFold2Factors {
            eq_lo: Vec::new(),
            eq_hi: Vec::new(),
            low_eq: [F128::ZERO; 4],
            table: Vec::new(),
            products: [F128::ZERO; 16],
        };
        let proof = || RingSwitchProof {
            s_hat_v: Vec::new(),
        };
        let sparse_c = vec![
            (
                proof(),
                ring_switch::RingSwitchBatchOutput {
                    rs_eq_ind: ring_switch::RsEqInd::DeferredDense {
                        eq_lo: vec![F128::ONE],
                        eq_hi: vec![F128::ONE],
                        table: Vec::new(),
                    },
                    sumcheck_claim: F128::ZERO,
                    direct_fold2: Some(direct()),
                    direct_fold4: None,
                    direct_fold8: None,
                    deferred_c_fold2: None,
                },
            ),
            (
                proof(),
                ring_switch::RingSwitchBatchOutput {
                    rs_eq_ind: ring_switch::RsEqInd::Sparse {
                        len: 1,
                        entries: Vec::new(),
                    },
                    sumcheck_claim: F128::ZERO,
                    direct_fold2: None,
                    direct_fold4: None,
                    direct_fold8: None,
                    deferred_c_fold2: None,
                },
            ),
        ];
        assert!(!direct_ab_claim_mix_supported(&sparse_c));
        assert!(
            sparse_c[0].1.direct_fold2.is_some(),
            "fallback must leave AB state untouched for the ordinary transcript"
        );

        let ordinary_c = vec![
            (
                proof(),
                ring_switch::RingSwitchBatchOutput {
                    rs_eq_ind: ring_switch::RsEqInd::DeferredDense {
                        eq_lo: vec![F128::ONE],
                        eq_hi: vec![F128::ONE],
                        table: Vec::new(),
                    },
                    sumcheck_claim: F128::ZERO,
                    direct_fold2: Some(direct()),
                    direct_fold4: None,
                    direct_fold8: None,
                    deferred_c_fold2: None,
                },
            ),
            (
                proof(),
                ring_switch::RingSwitchBatchOutput {
                    rs_eq_ind: ring_switch::RsEqInd::DeferredDense {
                        eq_lo: vec![F128::ONE],
                        eq_hi: vec![F128::ONE],
                        table: Vec::new(),
                    },
                    sumcheck_claim: F128::ZERO,
                    direct_fold2: None,
                    direct_fold4: None,
                    direct_fold8: None,
                    deferred_c_fold2: None,
                },
            ),
        ];
        assert!(direct_ab_claim_mix_supported(&ordinary_c));
    }

    /// Compact ownership/reduction oracle for the production stateful block
    /// dispatcher. It uses the same 64 KiB private-state size as the ranked
    /// composed table while keeping the output small enough for an ordinary
    /// unit test.
    #[test]
    fn hetero_open_combine_block_dispatch_matches_serial() {
        const N_BLOCKS: usize = 256;
        const BLOCK_LEN: usize = 32;
        let mut got = vec![F128::ZERO; N_BLOCKS * BLOCK_LEN];
        let got_uv = run_hetero_open_combine_blocks(
            &mut got,
            BLOCK_LEN,
            || vec![F128::ZERO; ring_switch::FOLD_TABLE_LEN],
            |scratch, block, out_block| {
                scratch[0] = F128::new(block as u64 ^ 0xA55A, (block as u64) << 32);
                let mut u0 = F128::ZERO;
                let mut u2 = F128::ZERO;
                for (offset, slot) in out_block.iter_mut().enumerate() {
                    *slot = scratch[0] + F128::new(offset as u64, offset as u64 * 3);
                    if offset.is_multiple_of(2) {
                        u0 += *slot;
                    }
                    u2 += *slot;
                }
                (u0, u2)
            },
        );

        let mut expected = vec![F128::ZERO; got.len()];
        let mut expected_uv = (F128::ZERO, F128::ZERO);
        for (block, out_block) in expected.chunks_mut(BLOCK_LEN).enumerate() {
            let seed = F128::new(block as u64 ^ 0xA55A, (block as u64) << 32);
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            for (offset, slot) in out_block.iter_mut().enumerate() {
                *slot = seed + F128::new(offset as u64, offset as u64 * 3);
                if offset.is_multiple_of(2) {
                    u0 += *slot;
                }
                u2 += *slot;
            }
            expected_uv.0 += u0;
            expected_uv.1 += u2;
        }
        assert_eq!(got, expected);
        assert_eq!(got_uv, expected_uv);
    }

    #[test]
    fn round1_lookahead_matches_materialized_fold() {
        let mut rng = Rng::new(0xF01D_2001);
        for log_n in [4usize, 9, 13] {
            let n = 1usize << log_n;
            let witness: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let basis: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let r = rng.f128();
            let ((u0, u2), c) = round0_and_round1_lookahead(&witness, &basis);

            let mut oracle0 = (F128::ZERO, F128::ZERO);
            for i in (0..n).step_by(2) {
                oracle0.0 += witness[i] * basis[i];
                oracle0.1 += (witness[i] + witness[i + 1]) * (basis[i] + basis[i + 1]);
            }
            assert_eq!((u0, u2), oracle0, "round zero at log_n={log_n}");

            let one_plus_r = F128::ONE + r;
            let folded_witness: Vec<F128> = witness
                .chunks_exact(2)
                .map(|p| p[0] * one_plus_r + p[1] * r)
                .collect();
            let folded_basis: Vec<F128> = basis
                .chunks_exact(2)
                .map(|p| p[0] * one_plus_r + p[1] * r)
                .collect();
            let mut oracle1 = (F128::ZERO, F128::ZERO);
            for i in (0..folded_witness.len()).step_by(2) {
                oracle1.0 += folded_witness[i] * folded_basis[i];
                oracle1.1 += (folded_witness[i] + folded_witness[i + 1])
                    * (folded_basis[i] + folded_basis[i + 1]);
            }
            let r2 = r * r;
            let evaluated = (c[0] + c[1] * r + c[2] * r2, c[3] + c[4] * r + c[5] * r2);
            assert_eq!(evaluated, oracle1, "round one at log_n={log_n}");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn ranked_open_lookahead_neon_matches_scalar() {
        let mut rng = Rng::new(0x10CA_AEAD);
        for log_n in [2usize, 5, 9, 15] {
            let n = 1usize << log_n;
            let witness: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let basis: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let expected = round0_and_round1_lookahead_scalar(&witness, &basis);
            let actual = crate::field::f128_slice::round0_and_round1_lookahead(&witness, &basis);
            assert_eq!(actual, expected, "deferred reduction at log_n={log_n}");
        }
    }

    /// Oracle for the two rerouted direct-materializer block tails. Both
    /// dispatchers must return exactly the bits their scalar predecessors
    /// returned, at the block widths the fold4/fold8 materializers actually
    /// use (2^13 and 2^11 slots) as well as at small widths.
    ///
    /// It also pins the semantic precondition of the fold8 reroute:
    /// `round0_scalar` and the `(u_0, u_2)` half of the lookahead scan agree
    /// bitwise, so swapping in a round-0-only deferred kernel cannot change
    /// the transcript.
    #[test]
    fn fold_deferred_reduce_dispatchers_match_scalar() {
        let mut rng = Rng::new(0xF01D_DEFD);
        for log_n in [2usize, 5, 9, 11, 13] {
            let n = 1usize << log_n;
            let witness: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let basis: Vec<F128> = (0..n).map(|_| rng.f128()).collect();

            let want_lookahead = round0_and_round1_lookahead_scalar(&witness, &basis);
            assert_eq!(
                round0_and_round1_lookahead_deferred(&witness, &basis),
                want_lookahead,
                "fold4 block tail at log_n={log_n}"
            );

            let want_round0 = round0_scalar(&witness, &basis);
            assert_eq!(
                want_round0, want_lookahead.0,
                "round0_scalar must agree with the lookahead (u_0, u_2) at log_n={log_n}"
            );
            assert_eq!(
                round0_deferred(&witness, &basis),
                want_round0,
                "fold8 block tail at log_n={log_n}"
            );
        }
    }

    #[test]
    fn fold_deferred_reduce_gate_follows_env() {
        assert_eq!(
            use_fold_deferred_reduce(),
            std::env::var_os("FLOCK_NO_FOLD_DEFERRED_REDUCE").is_none()
        );
    }

    #[test]
    fn ranked_open_lookahead_neon_gate_is_exact() {
        let expected = cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            target_feature = "aes"
        )) && std::env::var_os("FLOCK_NO_OPEN_LOOKAHEAD_NEON").is_none();
        assert_eq!(use_ranked_open_lookahead_neon(true, 1usize << 15), expected);
        assert!(!use_ranked_open_lookahead_neon(false, 1usize << 15));
        assert!(!use_ranked_open_lookahead_neon(true, 1usize << 14));
        assert!(!use_ranked_open_lookahead_neon(true, 1usize << 16));

        assert!(is_ranked_direct_fold2_lookahead_shape(
            1 << 25,
            1 << 15,
            1,
            true,
        ));
        assert!(is_ranked_direct_fold2_lookahead_shape(
            1 << 25,
            1 << 15,
            2,
            false,
        ));
        assert!(!is_ranked_direct_fold2_lookahead_shape(
            1 << 24,
            1 << 15,
            2,
            false,
        ));
        assert!(!is_ranked_direct_fold2_lookahead_shape(
            1 << 25,
            1 << 14,
            2,
            false,
        ));
        assert!(!is_ranked_direct_fold2_lookahead_shape(
            1 << 25,
            1 << 15,
            1,
            false,
        ));
        assert!(!is_ranked_direct_fold2_lookahead_shape(
            1 << 25,
            1 << 15,
            2,
            true,
        ));
    }

    #[test]
    fn direct_products_reproduce_round0_and_lookahead() {
        let mut rng = Rng::new(0xD1CE_0002);
        let mut witness = [F128::ZERO; 4];
        let mut basis = [F128::ZERO; 4];
        for value in witness.iter_mut().chain(basis.iter_mut()) {
            *value = rng.f128();
        }
        let mut products = [F128::ZERO; 16];
        for e in 0..4 {
            for d in 0..4 {
                products[4 * e + d] = witness[e] * basis[d];
            }
        }
        let factors = ring_switch::DirectFold2Factors {
            eq_lo: Vec::new(),
            eq_hi: Vec::new(),
            low_eq: [F128::ZERO; 4],
            table: Vec::new(),
            products,
        };
        assert_eq!(
            messages_from_direct_products(&[factors]),
            round0_and_round1_lookahead(&witness, &basis),
        );
    }

    fn zhat_skip_reference(z: &[bool], m: usize, z_skip: F128, x_outer: &[F128]) -> F128 {
        const K_SKIP: usize = 6;
        let ell = 1usize << K_SKIP;
        let lambda = lagrange_weights_naive(K_SKIP, z_skip);
        let eq_outer = build_eq(x_outer);
        let mut acc = F128::ZERO;
        for i_outer in 0..(1usize << (m - K_SKIP)) {
            let base = i_outer * ell;
            let mut inner = F128::ZERO;
            for i_skip in 0..ell {
                if z[base + i_skip] {
                    inner += lambda[i_skip];
                }
            }
            acc += eq_outer[i_outer] * inner;
        }
        acc
    }

    /// End-to-end Ligerito backend roundtrip through pcs::open_batch_mixed_ligerito
    /// and verify_opening_batch_ligerito_mixed. Single ring-switched claim
    /// (no PD — PD path is task #11).
    #[test]
    #[ignore] // Heavier — ~50-100 ms; run with `cargo test pcs_ligerito_roundtrip -- --ignored --nocapture`
    fn pcs_ligerito_backend_roundtrip() {
        let m = 22usize;
        let mut rng = Rng::new(0x11_6E_2170);
        let z = rng.bits(1 << m);
        let z_skip = rng.f128();
        let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
        let rs_claim = zhat_skip_reference(&z, m, z_skip, &x_outer);

        // PcsParams MUST set log_batch_size = ligerito_initial_k for L0 reuse.
        let initial_k = 6;
        let params = PcsParams {
            m,
            log_inv_rate: 1,
            log_batch_size: initial_k,
            profile: Default::default(),
            merkle_hash: Default::default(),
        };
        let z_packed = pack_witness(&z, m);
        let (commitment, prover_data) = commit(&z_packed, &params);

        let recursive_ks = vec![3usize, 3, 3];
        let log_inv_rates = vec![1usize, 3, 4, 6];
        let queries: Vec<usize> = log_inv_rates
            .iter()
            .map(|&r| crate::pcs::ligerito::udr_queries(r))
            .collect();
        let grinding_bits = vec![0usize; log_inv_rates.len()];
        let n_levels = log_inv_rates.len();
        let lig_p_cfg = crate::pcs::ligerito::ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: (m - LOG_PACKING) - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![6, 3, 0],
            recursive_ks: recursive_ks.clone(),
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; n_levels],
            ood_samples: vec![0; n_levels],
            merkle_hash: Default::default(),
        };
        let lig_v_cfg = crate::pcs::ligerito::VerifierConfig {
            log_inv_rates,
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: (m - LOG_PACKING) - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![6, 3, 0],
            recursive_ks,
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; n_levels],
            ood_samples: vec![0; n_levels],
            merkle_hash: Default::default(),
        };

        let mut ch_p = FsChallenger::new(b"flock-test-lig-v0");
        let proof = open_batch_mixed_ligerito_with_precomputed_s_hat_v(
            z_packed.clone(),
            &prover_data,
            &commitment,
            &[x_outer.as_slice()],
            &[],
            &[],
            &PaddingSpec::dense(m),
            &lig_p_cfg,
            &mut ch_p,
        );

        let mut ch_v = FsChallenger::new(b"flock-test-lig-v0");
        verify_opening_batch_ligerito_mixed(
            &commitment,
            &[rs_claim],
            &[z_skip],
            &[x_outer.as_slice()],
            &[],
            &proof,
            &lig_v_cfg,
            &mut ch_v,
        )
        .unwrap_or_else(|e| panic!("ligerito verify rejected honest proof: {e:?}"));
    }
}

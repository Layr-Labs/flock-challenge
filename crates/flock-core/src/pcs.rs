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

pub use commit::{
    Commitment, PcsParams, ProverData, commit, commit_into, prefault_codeword_during,
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

/// Run the `b_combined` build on the all-logical-cores commit pool
/// ([`commit::wide_hash_pool`]). The combine is DRAM-store + L1-table bound
/// and scales with the E-cores' extra issue capacity (paired same-session
/// A/B, m=32 M4 Max, 18 opens/arm: 23.7 → 20.4 ms median), unlike the rest
/// of the open phase, whose short statically-chunked sections regress behind
/// E-core stragglers (lig-prove total +4.3 ms when the WHOLE phase runs
/// wide: induce_sumcheck_poly +2.2, initial folds +1.0, recursive commits
/// +0.9) — so only this section is widened. Gated to large opens
/// (`l ≥ 2^22`, the m=29 ranked floor) and to configs where the global pool
/// is actually narrower; `FLOCK_NO_OPEN_POOL=1` is the kill switch (local
/// diagnostics; the ranked worker's cleared env never sets it). Pool width
/// cannot change wire bytes: every parallel reduction here is an XOR sum.
fn in_wide_combine_pool<R: Send>(_l: usize, op: impl FnOnce() -> R + Send) -> R {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if _l >= (1 << 22)
        && std::thread::available_parallelism()
            .is_ok_and(|n| n.get() > rayon::current_num_threads())
        && std::env::var_os("FLOCK_NO_OPEN_POOL").is_none()
    {
        return commit::wide_hash_pool().install(op);
    }
    op()
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
#[allow(clippy::explicit_auto_deref, clippy::too_many_arguments)]
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
    let trace = std::env::var("PCS_TRACE").is_ok();
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

    // Fold arena: all `initial_k` L0 fold rounds' output sizes are known
    // right now — allocate one exact-size arena (2·(l − l/2^k) F128s; 1008 MiB
    // at the ranked m=32 shape) and prefault it on a background thread WHILE
    // `compute_combined_basis_and_target` runs (its b_combined pass is partly
    // load-issue-bound, leaving spare DRAM bandwidth for the kernel's
    // zero-fill). The fold rounds then carve slices instead of paying ~1 GiB
    // of zero-fill + ~62k page faults inside the serial Fiat–Shamir chain.
    // Strictly per-open (no cross-prove retention): moved into the sumcheck
    // prover, freed when it drops. `FLOCK_NO_FOLD_ARENA` is a
    // local-diagnostics escape hatch; the ranked worker's cleared environment
    // never sets it.
    #[cfg(target_arch = "aarch64")]
    let fold_arena = {
        let l = packed_witness.len();
        let k = lig_config.initial_k;
        // Sub-2^13 opens fold serially anyway (no carve) — skip the arena.
        if k >= 1 && l >= (1 << 13) && std::env::var_os("FLOCK_NO_FOLD_ARENA").is_none() {
            Some(ligerito::FoldArena::new_prefaulted(
                ligerito::FoldArena::capacity_for(l, k),
            ))
        } else {
            None
        }
    };
    // x86_64 keeps its prewarmed scratch-pool fold path unchanged.
    #[cfg(not(target_arch = "aarch64"))]
    let fold_arena: Option<ligerito::FoldArena> = None;
    crate::gaptime::mark("open: fold arena ready");

    let combined = compute_combined_basis_and_target(
        &packed_witness,
        x_outers,
        precomputed_s_hat_v,
        packed_direct,
        padding,
        challenger,
        trace,
    );
    crate::gaptime::mark("open: combined basis + target done");

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
            fold_arena,
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
            &prover_data.merkle_tree,
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
            fold_arena,
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
            &prover_data.merkle_tree,
            combined.round0_prime,
            combined
                .round1_lookahead
                .expect("direct AB fold2 requires round-1 lookahead"),
            fold_arena,
            challenger,
        )
    } else {
        ligerito::recursive_prover_with_basis_precomputed_round0(
            lig_config,
            packed_witness,
            combined.b_combined,
            combined.target_combined,
            &prover_data.codeword,
            &prover_data.merkle_tree,
            combined.round0_prime,
            fold_arena,
            challenger,
        )
    };
    crate::gaptime::mark("open: ligerito recursive prover done");
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
    /// Round-1 message as two quadratics in the first challenge.
    round1_lookahead: Option<[F128; 6]>,
    /// Direct-fold4 round-2 bivariate lookahead (coefficients in r0, r1).
    round2_lookahead: Option<Fold4Lookahead2>,
    /// Direct-fold4 round-3 trivariate lookahead (coefficients in r0, r1, r2).
    round3_lookahead: Option<Fold4Lookahead3>,
    /// Direct-fold8 round-4 quadrivariate lookahead.
    #[allow(dead_code)] // Reserved for the rollback DirectFold8 lookahead path.
    round4_lookahead: Option<Fold8Lookahead4>,
    /// Direct-fold8 round-5 quintivariate lookahead.
    #[allow(dead_code)] // Reserved for the rollback DirectFold8 lookahead path.
    round5_lookahead: Option<Fold8Lookahead5>,
    /// AB sufficient statistics for direct materialization after rounds 0/1.
    /// `b_combined` contains the ordinary C contribution only.
    direct_fold2: Option<Vec<ring_switch::DirectFold2Factors>>,
    /// Sixteen-bank direct factors for BOTH ranked claims: rounds 0..3 come
    /// from the 16×16 product matrices and the state materializes once at N/16.
    direct_fold4: Option<Vec<ring_switch::DirectFold4Factors>>,
    /// Sixty-four-bank direct factors, populated when both claims carry an
    /// honest 64-bank precompute and the fold8 gate is on.
    direct_fold8: Option<Vec<ring_switch::DirectFold8Factors>>,
}

/// Compute the ordinary round-zero message and the following message as two
/// quadratics in the first challenge.
#[cfg(test)]
fn round0_and_round1_lookahead(witness: &[F128], basis: &[F128]) -> ((F128, F128), [F128; 6]) {
    assert_eq!(witness.len(), basis.len());
    assert!(witness.len().is_multiple_of(4));
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
        c[1] += a1 * b1 + p_even0 + p_sum0;
        c[2] += p_sum0;
        let e_a = a0 + a2;
        let e_b = b0 + b2;
        let se_a = sa0 + sa1;
        let se_b = sb0 + sb1;
        let p_even = e_a * e_b;
        let p_sum = se_a * se_b;
        let p_odd = (se_a + e_a) * (se_b + e_b);
        c[3] += p_even;
        c[4] += p_odd + p_even + p_sum;
        c[5] += p_sum;
    }
    ((u0, u2), c)
}

/// Round-0 message `(u_0, u_2)` over paired slots, without a following
/// lookahead. DirectFold8 uses this for its sixth and final retained round.
#[inline]
pub(crate) fn round0_scalar(witness: &[F128], basis: &[F128]) -> (F128, F128) {
    assert_eq!(witness.len(), basis.len());
    assert!(witness.len().is_multiple_of(2));
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

fn messages_from_direct_products(
    products: &[ring_switch::DirectFold2Factors],
) -> ((F128, F128), [F128; 6]) {
    let mut h = [F128::ZERO; 16];
    for claim in products {
        if let Some(products) = claim.products {
            for (out, value) in h.iter_mut().zip(products) {
                *out += value;
            }
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
#[allow(dead_code)] // Scalar oracle for the factorized DirectFold8 path.
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
#[cfg(any())]
pub(crate) fn messages_from_direct_products_fold8(
    factors: &[ring_switch::DirectFold8Factors],
) -> (
    (F128, F128),
    [F128; 6],
    Fold4Lookahead2,
    Fold4Lookahead3,
    Fold8Lookahead4,
    Fold8Lookahead5,
) {
    let mut h = [F128::ZERO; 4096];
    for claim in factors {
        for (out, value) in h.iter_mut().zip(claim.products) {
            *out += value;
        }
    }

    let (round0_u0, round0_u2) = direct_fold8_message_coefficients(&h, 0);
    let (round1_u0, round1_u2) = direct_fold8_message_coefficients(&h, 1);
    let (round2_u0, round2_u2) = direct_fold8_message_coefficients(&h, 2);
    let (round3_u0, round3_u2) = direct_fold8_message_coefficients(&h, 3);
    let (round4_u0, round4_u2) = direct_fold8_message_coefficients(&h, 4);
    let (round5_u0, round5_u2) = direct_fold8_message_coefficients(&h, 5);

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

/// Sixteen-bank route: both ranked claims must expose a complete
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

/// Deferred reduction in the direct-fold4 sixteen-bank materializer's
/// witness fold (`FLOCK_NO_FOLD_DEFERRED_REDUCE=1` restores the two nested
/// pair-fold passes; same field values either way). Latched once per process.
#[inline]
pub(crate) fn fold_deferred_reduce_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_FOLD_DEFERRED_REDUCE").is_none())
}

/// Direct-fold4 enable, latched once per process.
///
/// Default **enabled** for the ranked worker. `FLOCK_NO_OPEN_DIRECT_FOLD4=1`
/// restores the direct-C (fold2) route bit-for-bit. The retained-coordinate
/// producers (zerocheck round-1 C capture, the AB `z_vec` fold) and this
/// consumer share this predicate so they cannot silently disagree.
#[inline]
pub fn ranked_direct_fold4_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_OPEN_DIRECT_FOLD4").is_none())
}

#[inline]
fn direct_ab_claim_mix_supported(
    rs_results: &[(RingSwitchProof, ring_switch::RingSwitchBatchOutput)],
) -> bool {
    matches!(
        rs_results,
        [(_, ab), (_, c)]
            if ab.direct_fold2.as_ref().is_some_and(|direct| direct.products.is_some())
                && matches!(&c.rs_eq_ind, ring_switch::RsEqInd::DeferredDense { .. })
    )
}

#[inline]
fn direct_all_claim_mix_supported(
    rs_results: &[(RingSwitchProof, ring_switch::RingSwitchBatchOutput)],
) -> bool {
    matches!(
        rs_results,
        [(_, ab), (_, c)]
            if ab.direct_fold2.as_ref().is_some_and(|direct| direct.products.is_some())
                && c.direct_fold2.as_ref().is_some_and(|direct| direct.products.is_some())
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
    // `basis_elidable` is exactly the `direct_common` predicate evaluated
    // below, minus the parts that depend on ring_switch's own output. If it
    // holds AND every claim comes back with a direct_fold8 bundle and a
    // DeferredDense rs_eq_ind, `use_direct_fold8` is taken, `direct_count ==
    // n_rs` filters every claim out of both `rs_baked` and `rs_deferred`,
    // `direct_c_stats` is None and `b_combined` stays empty -- so no
    // `rs_eq_ind` basis factor is read anywhere. ring_switch decides the
    // second half of that antecedent itself and skips building them.
    let basis_elidable = cfg!(target_arch = "x86_64")
        && n_rs == 2
        && n_pd == 0
        && packed_witness.len() == (1usize << 25)
        && std::env::var_os("FLOCK_NO_OPEN_DIRECT_AB").is_none();
    let (mut rs_results, gammas_rs): (
        Vec<(RingSwitchProof, ring_switch::RingSwitchBatchOutput)>,
        Vec<F128>,
    ) = if n_rs > 0 {
        ring_switch::prove_batched_padded_with_precomputed_elidable(
            packed_witness,
            x_outers,
            precomputed_s_hat_v,
            padding,
            challenger,
            basis_elidable,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    crate::gaptime::mark("open: ring_switch done");
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

    // Production-only x86 route: retain AB as four low-coordinate banks and
    // materialize it only after the first two sumcheck challenges. The next
    // experiment retains C too, eliminating the remaining full basis buffer.
    let direct_common = cfg!(target_arch = "x86_64")
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
    let use_direct_ab = !use_direct_fold8
        && !use_direct_fold4
        && direct_common
        && (use_direct_all || direct_ab_claim_mix_supported(&rs_results));
    let use_direct_c = use_direct_ab
        && !use_direct_all
        && ranked_direct_c_enabled()
        && rs_results[1].1.direct_fold2.is_some();
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
    let direct_fold2 = if use_direct_all {
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
        let mut direct = vec![
            rs_results[0]
                .1
                .direct_fold2
                .take()
                .expect("direct AB gate checked claim zero"),
        ];
        if use_direct_c {
            direct.push(
                rs_results[1]
                    .1
                    .direct_fold2
                    .take()
                    .expect("direct C gate checked claim one"),
            );
        }
        Some(direct)
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

    let direct_c_stats = if use_direct_c {
        match &rs_results[1].1.rs_eq_ind {
            ring_switch::RsEqInd::DeferredDense {
                eq_lo,
                eq_hi,
                table,
                ..
            } => Some((eq_lo.as_slice(), eq_hi.as_slice(), table.as_slice())),
            _ => None,
        }
    } else {
        None
    };

    let rs_baked: Vec<&[F128]> = rs_results
        .iter()
        .enumerate()
        .filter_map(|(index, (_, output))| {
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
            if index < direct_count {
                return None;
            }
            match &output.rs_eq_ind {
                ring_switch::RsEqInd::DeferredDense {
                    eq_lo,
                    eq_hi,
                    table,
                    ..
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

    // ---- Build b_combined (γ-weighted sum of all rs_eq_ind + eq_ind) and the
    //      round-0 prime (u_0, u_2 over packed_witness · b_combined).
    let mut b_combined: Vec<F128> =
        if use_direct_c || use_direct_all || use_direct_fold4 || use_direct_fold8 {
            Vec::new()
        } else {
            crate::scratch::take_f128(l)
        };
    crate::gaptime::mark("open: b_combined taken");

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

    let ((mut round0_u0, mut round0_u2), mut round1_lookahead) = in_wide_combine_pool(l, || {
        if use_direct_fold8 {
            // M0 comes from the cached factor-state statistics below; subsequent
            // initial messages are generated online after each sampled challenge.
            ((F128::ZERO, F128::ZERO), None)
        } else if use_direct_fold4 {
            // All four initial messages come from retained product matrices.
            ((F128::ZERO, F128::ZERO), Some([F128::ZERO; 6]))
        } else if use_direct_all {
            ((F128::ZERO, F128::ZERO), Some([F128::ZERO; 6]))
        } else if let Some((eq_lo, eq_hi, table)) = direct_c_stats {
            let (prime, lookahead) = deferred_stats_lookahead(packed_witness, eq_lo, eq_hi, table);
            (prime, Some(lookahead))
        } else if use_fast {
            let b = rs_deferred[0].0.len(); // eq_lo.len(); shared across claims (same split)
            debug_assert!(b >= 2 && b.is_multiple_of(2));
            debug_assert!(rs_deferred.iter().all(|d| d.0.len() == b));
            // aarch64 ranked path: accumulate the leading claims into a 64 KiB
            // per-task stage and publish the final (s0, s1) pairs with `stnp q,q`
            // (see `fused_fast_combine_staged_nt`) — kills the 512 MiB
            // write-allocate read on the cold b_combined lines. Value-identical
            // to `fused_fast_combine`; `FLOCK_NO_BCOMB_NT` is a local-diagnostics
            // escape hatch (the ranked worker's cleared environment never sets it).
            // COMBINE-MERGE AUTOPSY (2026-07-28, M4 Max, m=32, b=4096, paired
            // same-session A/B, PCS_TRACE combine ms over 6 opens each):
            //   staged two-pass (this default):        22.3-27.0 ms
            //   fused2 single-pass (no stage, static
            //     tables, 2 mult+fold per elem):        25.5-27.5 ms
            //   merged (per-block composed tables,
            //     0 mults, 1 fused pass):               33.3-42.0 ms
            // The phase sits at the ONE-64KiB-table-live L1 floor: co-residing
            // both claims' random-access tables (128 KiB = all of L1d, on top of
            // the eq_lo/witness streams) thrashes, and the merged kernel's
            // per-block 128 KiB table rebuild adds stores that dwarf the two
            // saved GF mults/elem. Fewer fold evaluations per element is
            // algebraically impossible for the production claim pair: the ab
            // point (lincheck + AB-sumcheck challenges) and the c point (the
            // zerocheck eq challenge r_rest) share no coordinates, so the two
            // folds never share an argument (see fused_fast_combine_merged_nt
            // docs for the exact non-commuting step). FLOCK_COMBINE_MERGE=1
            // opts back into the merged kernel for re-measurement.
            #[cfg(target_arch = "aarch64")]
            let prime = if std::env::var_os("FLOCK_NO_BCOMB_NT").is_none() {
                if rs_deferred.len() <= 2
                    && b >= MERGE_MIN_BLOCK
                    && std::env::var_os("FLOCK_COMBINE_MERGE").is_some()
                {
                    fused_fast_combine_merged_nt(&mut b_combined, packed_witness, &rs_deferred, b)
                } else {
                    fused_fast_combine_staged_nt(&mut b_combined, packed_witness, &rs_deferred, b)
                }
            } else {
                fused_fast_combine(&mut b_combined, packed_witness, &rs_deferred, b)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let (prime, lookahead) = if use_direct_ab {
                let (prime, lookahead) =
                    fused_fast_combine_lookahead(&mut b_combined, packed_witness, &rs_deferred, b);
                (prime, Some(lookahead))
            } else {
                (
                    fused_fast_combine(&mut b_combined, packed_witness, &rs_deferred, b),
                    None,
                )
            };
            #[cfg(target_arch = "aarch64")]
            let lookahead = None;
            (prime, lookahead)
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
                        ..
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
            (prime, None)
        }
    });
    crate::gaptime::mark("open: combine kernel done");

    let mut round2_lookahead = None;
    let mut round3_lookahead = None;
    let round4_lookahead = None;
    let round5_lookahead = None;
    if let Some(direct) = direct_fold8.as_ref() {
        for claim in direct {
            round0_u0 += claim.round0.0;
            round0_u2 += claim.round0.1;
        }
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
            .expect("direct AB gate requires ordinary C lookahead");
        for (out, value) in combined_lookahead.iter_mut().zip(direct_lookahead) {
            *out += value;
        }
    }
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
            for &(idx, val) in entries {
                b_combined[idx] += val;
                adjust_prime_for_delta(idx, val);
            }
        }
    }
    for (pd, g) in packed_direct.iter().zip(gammas_pd.iter()) {
        if let DirectEqInd::Sparse(eq) = &pd.eq_ind {
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
            "  [open_batch] combine rs_eq_ind (L={}, rs×{}, pd×{}): {:6.2} ms",
            l,
            n_rs,
            n_pd,
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
        round4_lookahead,
        round5_lookahead,
        direct_fold2,
        direct_fold4,
        direct_fold8,
    }
}

/// Fused fast-path combine: fold every DeferredDense claim block-by-block
/// straight into `b_combined` — each claim's `e_hi` hoisted once per
/// `b = eq_lo.len()`-slot block, exactly as in `fold_b128_elems_split` — and
/// compute the round-0 prime `(u_0, u_2)` in the same pass. Every claim but
/// the last writes/RMWs `out_block`; the last claim is handled pairwise so
/// its result feeds the prime directly from registers rather than rereading
/// all of `out_block`.
fn fused_fast_combine(
    b_combined: &mut [F128],
    packed_witness: &[F128],
    rs_deferred: &[(&[F128], &[F128], &[F128], usize)],
    b: usize,
) -> (F128, F128) {
    use rayon::prelude::*;
    b_combined
        .par_chunks_mut(b)
        .enumerate()
        .map(|(hi, out_block)| {
            // Accumulate every claim except the last: first claim writes,
            // subsequent claims add.
            let last = rs_deferred.len() - 1;
            for (ci, (eq_lo, eq_hi, table, _)) in rs_deferred[..last].iter().enumerate() {
                let e_hi = eq_hi[hi];
                if ci == 0 {
                    for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                        *slot = ring_switch::fold_one_slot(lo * e_hi, table);
                    }
                } else {
                    for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                        *slot += ring_switch::fold_one_slot(lo * e_hi, table);
                    }
                }
            }

            // Final claim + round-0 prime over this block's pairs. This is
            // the same scalar operation and pair order as the former final
            // prime loop, but removes one full b_combined cache read pass.
            let (eq_lo, eq_hi, table, _) = rs_deferred[last];
            let e_hi = eq_hi[hi];
            let base = hi * b;
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            for t in 0..(b / 2) {
                let i0 = 2 * t;
                let i1 = i0 + 1;
                let v0 = ring_switch::fold_one_slot(eq_lo[i0] * e_hi, table);
                let v1 = ring_switch::fold_one_slot(eq_lo[i1] * e_hi, table);
                let (s0, s1) = if last == 0 {
                    (v0, v1)
                } else {
                    (out_block[i0] + v0, out_block[i1] + v1)
                };
                out_block[i0] = s0;
                out_block[i1] = s1;
                let a0 = packed_witness[base + i0];
                let a1 = packed_witness[base + i1];
                u0 += a0 * s0;
                u2 += (a0 + a1) * (s0 + s1);
            }
            (u0, u2)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        )
}

/// Direct-AB variant of [`fused_fast_combine`]. In addition to writing the
/// ordinary (C) basis and its round-zero message, accumulate the round-one
/// message as two quadratics in the first challenge. Groups of four are used
/// so no extra pass over the full witness/basis is needed.
fn fused_fast_combine_lookahead(
    b_combined: &mut [F128],
    packed_witness: &[F128],
    rs_deferred: &[(&[F128], &[F128], &[F128], usize)],
    b: usize,
) -> ((F128, F128), [F128; 6]) {
    use rayon::prelude::*;
    debug_assert!(b >= 4 && b.is_multiple_of(4));
    b_combined
        .par_chunks_mut(b)
        .enumerate()
        .map(|(hi, out_block)| {
            let last = rs_deferred.len() - 1;
            for (ci, (eq_lo, eq_hi, table, _)) in rs_deferred[..last].iter().enumerate() {
                let e_hi = eq_hi[hi];
                if ci == 0 {
                    for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                        *slot = ring_switch::fold_one_slot(lo * e_hi, table);
                    }
                } else {
                    for (slot, &lo) in out_block.iter_mut().zip(eq_lo.iter()) {
                        *slot += ring_switch::fold_one_slot(lo * e_hi, table);
                    }
                }
            }

            let (eq_lo, eq_hi, table, _) = rs_deferred[last];
            let e_hi = eq_hi[hi];
            let base = hi * b;
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            let mut c = [F128::ZERO; 6];
            for t in 0..(b / 4) {
                let i = 4 * t;
                let v0 = ring_switch::fold_one_slot(eq_lo[i] * e_hi, table);
                let v1 = ring_switch::fold_one_slot(eq_lo[i + 1] * e_hi, table);
                let v2 = ring_switch::fold_one_slot(eq_lo[i + 2] * e_hi, table);
                let v3 = ring_switch::fold_one_slot(eq_lo[i + 3] * e_hi, table);
                let (b0, b1, b2, b3) = if last == 0 {
                    (v0, v1, v2, v3)
                } else {
                    (
                        out_block[i] + v0,
                        out_block[i + 1] + v1,
                        out_block[i + 2] + v2,
                        out_block[i + 3] + v3,
                    )
                };
                out_block[i] = b0;
                out_block[i + 1] = b1;
                out_block[i + 2] = b2;
                out_block[i + 3] = b3;

                let a0 = packed_witness[base + i];
                let a1 = packed_witness[base + i + 1];
                let a2 = packed_witness[base + i + 2];
                let a3 = packed_witness[base + i + 3];
                let sa0 = a0 + a1;
                let sb0 = b0 + b1;
                let sa1 = a2 + a3;
                let sb1 = b2 + b3;
                let p_even0 = a0 * b0;
                let p_sum0 = sa0 * sb0;
                u0 += p_even0 + a2 * b2;
                u2 += p_sum0 + sa1 * sb1;
                c[0] += p_even0;
                c[1] += a1 * b1 + p_even0 + p_sum0;
                c[2] += p_sum0;
                let e_a = a0 + a2;
                let e_b = b0 + b2;
                let se_a = sa0 + sa1;
                let se_b = sb0 + sb1;
                let p_even = e_a * e_b;
                let p_sum = se_a * se_b;
                let p_odd = (se_a + e_a) * (se_b + e_b);
                c[3] += p_even;
                c[4] += p_odd + p_even + p_sum;
                c[5] += p_sum;
            }
            ((u0, u2), c)
        })
        .reduce(
            || ((F128::ZERO, F128::ZERO), [F128::ZERO; 6]),
            |((x0, x2), mut xc), ((y0, y2), yc)| {
                for (x, y) in xc.iter_mut().zip(yc) {
                    *x += y;
                }
                ((x0 + y0, x2 + y2), xc)
            },
        )
}

/// Evaluate one deferred-dense basis only long enough to obtain its first two
/// sumcheck messages. No length-`L` basis is written: C is reconstructed
/// directly at `L/4` once both challenges are known.
fn deferred_stats_lookahead(
    packed_witness: &[F128],
    eq_lo: &[F128],
    eq_hi: &[F128],
    table: &[F128],
) -> ((F128, F128), [F128; 6]) {
    use rayon::prelude::*;
    let b = eq_lo.len();
    debug_assert!(b >= 4 && b.is_multiple_of(4));
    debug_assert_eq!(packed_witness.len(), b * eq_hi.len());
    // Hoist e_hi multiply into one composed 64 KiB table per hi-block.
    // Removes one GF mult per element from the full-length C statistics pass.
    // map_init reuses the composed buffer across hi-blocks (no per-block alloc).
    eq_hi
        .par_iter()
        .enumerate()
        .map_init(
            || vec![F128::ZERO; ring_switch::FOLD_TABLE_TOTAL],
            |composed, (hi, &e_hi)| {
                let base = hi * b;
                ring_switch::compose_block_table(table, e_hi, composed);
                let mut u0 = F128::ZERO;
                let mut u2 = F128::ZERO;
                let mut c = [F128::ZERO; 6];
                for t in 0..(b / 4) {
                    let i = 4 * t;
                    let b0 = ring_switch::fold_one_slot(eq_lo[i], composed);
                    let b1 = ring_switch::fold_one_slot(eq_lo[i + 1], composed);
                    let b2 = ring_switch::fold_one_slot(eq_lo[i + 2], composed);
                    let b3 = ring_switch::fold_one_slot(eq_lo[i + 3], composed);
                    let a0 = packed_witness[base + i];
                    let a1 = packed_witness[base + i + 1];
                    let a2 = packed_witness[base + i + 2];
                    let a3 = packed_witness[base + i + 3];
                    let sa0 = a0 + a1;
                    let sb0 = b0 + b1;
                    let sa1 = a2 + a3;
                    let sb1 = b2 + b3;
                    let p_even0 = a0 * b0;
                    let p_sum0 = sa0 * sb0;
                    u0 += p_even0 + a2 * b2;
                    u2 += p_sum0 + sa1 * sb1;
                    c[0] += p_even0;
                    c[1] += a1 * b1 + p_even0 + p_sum0;
                    c[2] += p_sum0;
                    let e_a = a0 + a2;
                    let e_b = b0 + b2;
                    let se_a = sa0 + sa1;
                    let se_b = sb0 + sb1;
                    let p_even = e_a * e_b;
                    let p_sum = se_a * se_b;
                    let p_odd = (se_a + e_a) * (se_b + e_b);
                    c[3] += p_even;
                    c[4] += p_odd + p_even + p_sum;
                    c[5] += p_sum;
                }
                ((u0, u2), c)
            },
        )
        .reduce(
            || ((F128::ZERO, F128::ZERO), [F128::ZERO; 6]),
            |((x0, x2), mut xc), ((y0, y2), yc)| {
                for (x, y) in xc.iter_mut().zip(yc) {
                    *x += y;
                }
                ((x0 + y0, x2 + y2), xc)
            },
        )
}

/// `stnp q,q` publish of two adjacent F128s (32 B) from NEON registers —
/// non-temporal, so the destination line's write-allocate RFO read is
/// skipped. No Rust intrinsic emits `stnp`; raw asm, same 3-line wrapper as
/// the promoted q-form stnp family.
///
/// # Safety
/// `dst` must be valid for 32 bytes of writes and 16-byte aligned.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_nt_f128_pair(dst: *mut F128, v0: F128, v1: F128) {
    // SAFETY: F128 is repr(C, align(16)) {lo, hi} little-endian — the same
    // 16 bytes a q-register store publishes.
    unsafe {
        let q0: core::arch::aarch64::uint8x16_t = core::mem::transmute(v0);
        let q1: core::arch::aarch64::uint8x16_t = core::mem::transmute(v1);
        core::arch::asm!(
            "stnp {a:q}, {b:q}, [{p}]",
            a = in(vreg) q0,
            b = in(vreg) q1,
            p = in(reg) dst,
            options(nostack, preserves_flags)
        );
    }
}

/// [`fused_fast_combine`] with staged accumulation + non-temporal publish.
///
/// The leading claims' first pass no longer touches the cold `b_combined`
/// pool lines (512 MiB of write-allocate RFO at the ranked shape): it
/// accumulates into a reused 64 KiB per-task stage buffer that stays L1/L2
/// hot. The final pairwise loop reads the stage, forms the final `(s0, s1)`
/// in registers, and publishes them with `stnp q,q` 32-byte pairs — every
/// store to a `b_combined` line is non-temporal (blocks are line-aligned),
/// so no allocate happens at all. `b_combined` is next read only at fold
/// round 0, after a 19-bit Fiat–Shamir grind — fully DRAM-cold, so no SLC
/// free-hits are lost.
///
/// The `fold_one_slot` table-read ORDER is exactly `fused_fast_combine`'s —
/// only the store targets change — and the round-0 prime is fed from the
/// same register values, so the result (and the transcript) is identical.
#[cfg(target_arch = "aarch64")]
fn fused_fast_combine_staged_nt(
    b_combined: &mut [F128],
    packed_witness: &[F128],
    rs_deferred: &[(&[F128], &[F128], &[F128], usize)],
    b: usize,
) -> (F128, F128) {
    use rayon::prelude::*;
    b_combined
        .par_chunks_mut(b)
        .enumerate()
        .map_init(
            // Per-task stage, reused across this task's blocks. Uninit is
            // safe: when `last > 0` claim 0 writes every slot before any
            // read; when `last == 0` the stage is never touched.
            || crate::alloc_uninit_f128_vec(b),
            |stage, (hi, out_block)| {
                let last = rs_deferred.len() - 1;
                for (ci, (eq_lo, eq_hi, table, _)) in rs_deferred[..last].iter().enumerate() {
                    let e_hi = eq_hi[hi];
                    if ci == 0 {
                        for (slot, &lo) in stage.iter_mut().zip(eq_lo.iter()) {
                            *slot = ring_switch::fold_one_slot(lo * e_hi, table);
                        }
                    } else {
                        for (slot, &lo) in stage.iter_mut().zip(eq_lo.iter()) {
                            *slot += ring_switch::fold_one_slot(lo * e_hi, table);
                        }
                    }
                }

                let (eq_lo, eq_hi, table, _) = rs_deferred[last];
                let e_hi = eq_hi[hi];
                let base = hi * b;
                let mut u0 = F128::ZERO;
                let mut u2 = F128::ZERO;
                let out_ptr = out_block.as_mut_ptr();
                for t in 0..(b / 2) {
                    let i0 = 2 * t;
                    let i1 = i0 + 1;
                    let v0 = ring_switch::fold_one_slot(eq_lo[i0] * e_hi, table);
                    let v1 = ring_switch::fold_one_slot(eq_lo[i1] * e_hi, table);
                    let (s0, s1) = if last == 0 {
                        (v0, v1)
                    } else {
                        (stage[i0] + v0, stage[i1] + v1)
                    };
                    // SAFETY: i1 < b = out_block.len(), so out_ptr + i0 is
                    // valid for 32 bytes; out_block is a par_chunks_mut(b)
                    // block of a 16-aligned F128 buffer with b even, so the
                    // address is 16-byte aligned.
                    unsafe { store_nt_f128_pair(out_ptr.add(i0), s0, s1) };
                    let a0 = packed_witness[base + i0];
                    let a1 = packed_witness[base + i1];
                    u0 += a0 * s0;
                    u2 += (a0 + a1) * (s0 + s1);
                }
                (u0, u2)
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        )
}

/// Minimum `b = eq_lo.len()` for [`fused_fast_combine_merged_nt`]: the
/// per-block composed-table rebuild costs ~128 GF doublings + 128 folds +
/// 4096 XORs per claim, which must amortize over the block's `b` fold
/// evaluations. Production (m = 32) has b = 4096.
#[cfg(target_arch = "aarch64")]
const MERGE_MIN_BLOCK: usize = 2048;

/// OPT-IN (`FLOCK_COMBINE_MERGE=1`) — measured SLOWER than the staged
/// default at the production shape; kept as the reproducible negative +
/// the record of why no algebraic claim-merging exists. See the autopsy
/// comment at the call site for the paired numbers.
///
/// # Why the element loop cannot be halved (impossibility record)
/// `fold_one_slot(x, T)` is the F2-linear map `M_T(x) = Σ_b x_b·col_T[b]` —
/// linear in BOTH the table and the element. Claim merging
/// (`fold(x,T1) + fold(x,T2) = fold(x, T1⊕T2)`) therefore needs a COMMON
/// argument `x`. The production pair folds
/// `x_ab(j) = eq_lo_ab[i]·eq_hi_ab[h]` and `x_c(j) = eq_lo_c[i]·eq_hi_c[h]`
/// where the ab point is built from lincheck + AB-sumcheck bind challenges
/// (`r_inner_*`, `mlv_challenges`) and the c point from the zerocheck eq
/// challenge (`r_rest`) — independent Fiat–Shamir samples with zero shared
/// coordinates (`r1cs::ab_claim_point` / `c_claim_point`, both layouts), so
/// `x_ab(j) ≠ x_c(j)` and no merged table exists. Blockwise hi-factoring
/// (`M(e_hi·x) = e_hi·M(x)`) fails because `M_T` is F2-linear but not
/// GF(2^128)-linear: commuting with `mul_{e_hi}` for an `e_hi` generating
/// the field forces `M = mul_{M(1)}` (the field is its own centralizer in
/// End_F2), i.e. `eq(b, r'') = α·X^b` for all `b` — a measure-zero event
/// for a Fiat–Shamir `r''`. And the fold inputs are products of uniform
/// field elements, so no byte-level repetition exists to memoize.
///
/// # What this kernel does instead
/// [`fused_fast_combine_staged_nt`] with the per-element GF multiply
/// algebraically absorbed into per-block composed tables, and all (≤ 2)
/// claims fused into ONE pass.
///
/// Identity used (exact, not approximate): `fold_one_slot(·, T)` is the
/// F2-linear map `M_T`, and GF multiplication is F2-bilinear, so per block
/// `hi` the map `x ↦ M_T(x · e_hi)` is itself F2-linear and is represented
/// by a composed byte table built in O(4096) per block
/// ([`ring_switch::compose_block_table`]). Then
/// `b[hi·b + i] = Σ_c fold_one_slot(eq_lo_c[i], T'_{c,hi})` — no per-element
/// multiply, no NEON→GPR byte-extract on the table-address critical path
/// (`eq_lo_c[i]` loads straight into GPRs), and no 64 KiB stage buffer
/// (both claims are folded in the same iteration, so `s0`/`s1` form in
/// registers). The `stnp` publish and the round-0 prime feed are identical
/// to the staged kernel.
///
/// Value-identity: every `s0`/`s1` is the same field element the staged
/// kernel computes (exact F2 algebra), and the prime is an XOR-reduction
/// (associative + commutative), so `b_combined` bytes and the transcript
/// are unchanged. Oracle-tested against [`fused_fast_combine`].
#[cfg(target_arch = "aarch64")]
fn fused_fast_combine_merged_nt(
    b_combined: &mut [F128],
    packed_witness: &[F128],
    rs_deferred: &[(&[F128], &[F128], &[F128], usize)],
    b: usize,
) -> (F128, F128) {
    use rayon::prelude::*;
    const TAB: usize = ring_switch::FOLD_TABLE_TOTAL;
    let n_claims = rs_deferred.len();
    assert!(n_claims == 1 || n_claims == 2);
    b_combined
        .par_chunks_mut(b)
        .enumerate()
        .map_init(
            // Per-task composed-table storage (64 KiB per claim), rebuilt for
            // every block. Uninit is safe: `compose_block_table` writes every
            // slot before any read.
            || crate::alloc_uninit_f128_vec(n_claims * TAB),
            |tabs, (hi, out_block)| {
                for (c, (_, eq_hi, table, _)) in rs_deferred.iter().enumerate() {
                    ring_switch::compose_block_table(
                        table,
                        eq_hi[hi],
                        &mut tabs[c * TAB..(c + 1) * TAB],
                    );
                }
                let base = hi * b;
                let mut u0 = F128::ZERO;
                let mut u2 = F128::ZERO;
                let out_ptr = out_block.as_mut_ptr();
                match rs_deferred {
                    [(lo0, _, _, _)] => {
                        let t0 = &tabs[..TAB];
                        for t in 0..(b / 2) {
                            let i0 = 2 * t;
                            let i1 = i0 + 1;
                            let s0 = ring_switch::fold_one_slot(lo0[i0], t0);
                            let s1 = ring_switch::fold_one_slot(lo0[i1], t0);
                            // SAFETY: i1 < b = out_block.len(); out_block is a
                            // par_chunks_mut(b) block of a 16-aligned F128
                            // buffer with b even, so the address is 16-byte
                            // aligned and valid for 32 bytes.
                            unsafe { store_nt_f128_pair(out_ptr.add(i0), s0, s1) };
                            let a0 = packed_witness[base + i0];
                            let a1 = packed_witness[base + i1];
                            u0 += a0 * s0;
                            u2 += (a0 + a1) * (s0 + s1);
                        }
                    }
                    [(lo0, _, _, _), (lo1, _, _, _)] => {
                        let (t0, t1) = tabs.split_at(TAB);
                        for t in 0..(b / 2) {
                            let i0 = 2 * t;
                            let i1 = i0 + 1;
                            let s0 = ring_switch::fold_one_slot(lo0[i0], t0)
                                + ring_switch::fold_one_slot(lo1[i0], t1);
                            let s1 = ring_switch::fold_one_slot(lo0[i1], t0)
                                + ring_switch::fold_one_slot(lo1[i1], t1);
                            // SAFETY: as above.
                            unsafe { store_nt_f128_pair(out_ptr.add(i0), s0, s1) };
                            let a0 = packed_witness[base + i0];
                            let a1 = packed_witness[base + i1];
                            u0 += a0 * s0;
                            u2 += (a0 + a1) * (s0 + s1);
                        }
                    }
                    _ => unreachable!("gated to 1 or 2 claims"),
                }
                (u0, u2)
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        )
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
    /// Oracle: `compose_block_table` — folding `x` through the composed
    /// table must be bit-identical to folding `x * e_hi` through the base
    /// table, for random tables/elements.
    #[test]
    fn compose_block_table_matches_mult_then_fold() {
        let mut rng = Rng::new(0xC0_4B_1E5);
        for trial in 0..4 {
            let eq_r_dprime: Vec<F128> = (0..128).map(|_| rng.f128()).collect();
            let base = ring_switch::build_fold_byte_table(&eq_r_dprime);
            let e_hi = rng.f128();
            let mut composed = vec![F128::ZERO; ring_switch::FOLD_TABLE_TOTAL];
            ring_switch::compose_block_table(&base, e_hi, &mut composed);
            for _ in 0..256 {
                let x = rng.f128();
                assert_eq!(
                    ring_switch::fold_one_slot(x, &composed),
                    ring_switch::fold_one_slot(x * e_hi, &base),
                    "trial {trial}"
                );
            }
            // Edge elements.
            for x in [F128::ZERO, F128::ONE, F128::new(!0, !0)] {
                assert_eq!(
                    ring_switch::fold_one_slot(x, &composed),
                    ring_switch::fold_one_slot(x * e_hi, &base),
                );
            }
        }
    }

    #[test]
    fn combine_lookahead_matches_materialized_oracle() {
        let mut rng = Rng::new(0xD1CE_C001);
        for &(b, n_hi, n_claims) in &[(16usize, 8usize, 1usize), (64, 4, 2)] {
            let l = b * n_hi;
            let packed_witness: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let claims: Vec<(Vec<F128>, Vec<F128>, Vec<F128>)> = (0..n_claims)
                .map(|_| {
                    let eq_lo: Vec<F128> = (0..b).map(|_| rng.f128()).collect();
                    let eq_hi: Vec<F128> = (0..n_hi).map(|_| rng.f128()).collect();
                    let eq_r_dprime: Vec<F128> = (0..128).map(|_| rng.f128()).collect();
                    (
                        eq_lo,
                        eq_hi,
                        ring_switch::build_fold_byte_table(&eq_r_dprime),
                    )
                })
                .collect();
            let deferred: Vec<(&[F128], &[F128], &[F128], usize)> = claims
                .iter()
                .map(|(lo, hi, table)| {
                    (
                        lo.as_slice(),
                        hi.as_slice(),
                        table.as_slice(),
                        b.trailing_zeros() as usize,
                    )
                })
                .collect();
            let mut got_basis = vec![F128::ZERO; l];
            let (got_round0, got_lookahead) =
                fused_fast_combine_lookahead(&mut got_basis, &packed_witness, &deferred, b);
            let mut want_basis = vec![F128::ZERO; l];
            let want_round0 = fused_fast_combine(&mut want_basis, &packed_witness, &deferred, b);
            let (oracle_round0, oracle_lookahead) =
                round0_and_round1_lookahead(&packed_witness, &want_basis);
            assert_eq!(got_basis, want_basis);
            assert_eq!(got_round0, want_round0);
            assert_eq!(got_round0, oracle_round0);
            assert_eq!(got_lookahead, oracle_lookahead);
            if n_claims == 1 {
                let (stats_round0, stats_lookahead) = deferred_stats_lookahead(
                    &packed_witness,
                    deferred[0].0,
                    deferred[0].1,
                    deferred[0].2,
                );
                assert_eq!(stats_round0, oracle_round0);
                assert_eq!(stats_lookahead, oracle_lookahead);
            }
        }
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
            products: Some(products),
        };
        assert_eq!(
            messages_from_direct_products(&[factors]),
            round0_and_round1_lookahead(&witness, &basis),
        );

        let mut basis_c = [F128::ZERO; 4];
        for value in &mut basis_c {
            *value = rng.f128();
        }
        let mut products_ab = [F128::ZERO; 16];
        let mut products_c = [F128::ZERO; 16];
        for e in 0..4 {
            for d in 0..4 {
                products_ab[4 * e + d] = witness[e] * basis[d];
                products_c[4 * e + d] = witness[e] * basis_c[d];
            }
        }
        let make_factors = |products| ring_switch::DirectFold2Factors {
            eq_lo: Vec::new(),
            eq_hi: Vec::new(),
            low_eq: [F128::ZERO; 4],
            table: Vec::new(),
            products: Some(products),
        };
        let mut combined_basis = basis;
        for (out, value) in combined_basis.iter_mut().zip(basis_c) {
            *out += value;
        }
        assert_eq!(
            messages_from_direct_products(&[make_factors(products_ab), make_factors(products_c),]),
            round0_and_round1_lookahead(&witness, &combined_basis),
        );
    }

    /// Oracle: `fused_fast_combine_merged_nt` must produce byte-identical
    /// `b_combined` and an identical round-0 prime vs the reference
    /// `fused_fast_combine` (and the staged kernel), for 1- and 2-claim
    /// shapes at several block sizes including the production-like split.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn merged_combine_matches_reference() {
        let mut rng = Rng::new(0x6E_46_D3);
        for &(b, n_hi, n_claims) in &[
            (16usize, 8usize, 2usize),
            (64, 4, 1),
            (2048, 4, 2),
            (4096, 2, 2),
            (4096, 2, 1),
        ] {
            let l = b * n_hi;
            let packed_witness: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let claims: Vec<(Vec<F128>, Vec<F128>, Vec<F128>)> = (0..n_claims)
                .map(|_| {
                    let eq_lo: Vec<F128> = (0..b).map(|_| rng.f128()).collect();
                    let eq_hi: Vec<F128> = (0..n_hi).map(|_| rng.f128()).collect();
                    let eq_r_dprime: Vec<F128> = (0..128).map(|_| rng.f128()).collect();
                    let table = ring_switch::build_fold_byte_table(&eq_r_dprime);
                    (eq_lo, eq_hi, table)
                })
                .collect();
            let rs_deferred: Vec<(&[F128], &[F128], &[F128], usize)> = claims
                .iter()
                .map(|(lo, hi, t)| {
                    (
                        lo.as_slice(),
                        hi.as_slice(),
                        t.as_slice(),
                        b.trailing_zeros() as usize,
                    )
                })
                .collect();

            let mut b_ref = vec![F128::ZERO; l];
            let prime_ref = fused_fast_combine(&mut b_ref, &packed_witness, &rs_deferred, b);

            let mut b_merged = vec![F128::ZERO; l];
            let prime_merged =
                fused_fast_combine_merged_nt(&mut b_merged, &packed_witness, &rs_deferred, b);

            assert_eq!(b_ref, b_merged, "b={b} n_hi={n_hi} claims={n_claims}");
            assert_eq!(
                prime_ref, prime_merged,
                "b={b} n_hi={n_hi} claims={n_claims}"
            );

            let mut b_staged = vec![F128::ZERO; l];
            let prime_staged =
                fused_fast_combine_staged_nt(&mut b_staged, &packed_witness, &rs_deferred, b);
            assert_eq!(b_staged, b_merged);
            assert_eq!(prime_staged, prime_merged);
        }
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

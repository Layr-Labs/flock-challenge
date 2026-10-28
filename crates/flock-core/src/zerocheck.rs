//! Zerocheck PIOP: prove a(y) · b(y) ⊕ c(y) = 0 for all y ∈ {0,1}^m.
//!
//! Inputs are three bit vectors of length 2^m. Output is an evaluation claim
//! on the multilinear extensions â, b̂, ĉ at the protocol-derived point.
//!
//! Protocol shape (m = log_n, k_skip = [`K_SKIP`] = 6):
//!   1. Verifier samples `r ∈ F_{2^128}^m` (the zerocheck challenge).
//!   2. Prover sends `P^{AB}(λ)` and `P^C(λ)` for λ ∈ Λ, |Λ| = 2^k_skip.
//!   3. Verifier samples `z ∈ F_{2^128}` (univariate-skip fold point).
//!   4. For each of the `m - k_skip` multilinear rounds, prover sends
//!      `(P_r(1), P_r(∞))` and verifier samples `ρ_r`.
//!   5. Prover sends final MLE evaluations `(â, b̂, ĉ)` at the resulting point.
//!
//! Both `prove` and `verify` are wired end-to-end. The prove→verify roundtrip
//! is tested on honest witnesses; verify also rejects byte-mutated proofs and
//! shape-corrupted ones.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::ntt::InvNttTableByteSingleGf8;
use serde::{Deserialize, Serialize};

pub mod multilinear;
pub mod univariate_skip;
pub mod univariate_skip_deg4;
pub mod univariate_skip_deg4_optimized;
pub mod univariate_skip_optimized;

use multilinear::{
    UniSkipFoldTable, eval_round3_lookahead, fold_and_compute_round_pair_into,
    fold_compact_and_compute_round_pair, fold_in_place_pair, fold2_compact_and_round4_into,
    fold2_compact_and_round45_into, fold2_plain_and_round6_into, fold2_plain_and_round67_into,
    interpolate_at_z_combined, interpolate_at_z_on_lambda, round_pair_naive,
    uni_skip_fold_and_round_pair_compact_padded_lookahead,
    uni_skip_fold_and_round_pair_compact_padded_with_deltas,
};
use univariate_skip_optimized::{
    c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed_padded,
    small_challenges_ghash,
};

/// Test-only forced-off latch for the two-challenge lookahead. Production
/// reads `FLOCK_NO_ZC_LOOKAHEAD`; the transcript-identity test flips this
/// instead so it never has to mutate the process environment. Flipping it
/// cannot make a concurrently running test wrong — both routes emit the same
/// transcript, which is exactly what that test asserts.
#[cfg(test)]
pub(crate) static ZC_LOOKAHEAD_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[inline]
fn lookahead_off() -> bool {
    #[cfg(test)]
    if ZC_LOOKAHEAD_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    std::env::var_os("FLOCK_NO_ZC_LOOKAHEAD").is_some()
}

/// Test-only forced-off latch for the second-level cascade (rounds 5+6),
/// mirroring [`ZC_LOOKAHEAD_FORCED_OFF`]: the transcript-identity test flips
/// this instead of mutating the process environment. Flipping it cannot make
/// a concurrently running test wrong — both routes emit the same transcript.
#[cfg(test)]
pub(crate) static ZC_CASCADE2_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Kill switch for the cascaded rounds-5/6 lookahead: `FLOCK_NO_ZC_CASCADE2=1`
/// (exact '1') restores the incumbent i=2/i=3 tail route within the same
/// binary. Bit-identical either way — the cascade is a pure reassociation of
/// exact F128 arithmetic, which is what the transcript-identity test asserts.
#[inline]
fn cascade2_off() -> bool {
    #[cfg(test)]
    if ZC_CASCADE2_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    std::env::var_os("FLOCK_NO_ZC_CASCADE2").is_some_and(|v| v == *"1")
}

/// Test-only forced-off latch for the third-level cascade (rounds 7+8),
/// mirroring [`ZC_CASCADE2_FORCED_OFF`]: the transcript-identity test flips
/// this instead of mutating the process environment. Flipping it cannot make
/// a concurrently running test wrong — both routes emit the same transcript.
#[cfg(test)]
pub(crate) static ZC_CASCADE3_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Kill switch for the cascaded rounds-7/8 lookahead: `FLOCK_NO_ZC_CASCADE3=1`
/// (exact '1') restores the cascade2 i=4/i=5 tail route within the same
/// binary (and with cascade2 ALSO off, the incumbent route). OnceLock-latched:
/// the environment is read once per process. Bit-identical either way — same
/// pure-reassociation argument as the levels above, asserted by the cascade3
/// transcript-identity test.
#[inline]
fn cascade3_off() -> bool {
    #[cfg(test)]
    if ZC_CASCADE3_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_CASCADE3").is_some_and(|v| v == *"1"))
}

/// Test-only forced-off latch for the fourth-level cascade (rounds 9+10),
/// mirroring [`ZC_CASCADE3_FORCED_OFF`]: the transcript-identity test flips
/// this instead of mutating the process environment. Flipping it cannot make
/// a concurrently running test wrong — both routes emit the same transcript.
#[cfg(test)]
pub(crate) static ZC_CASCADE4_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Kill switch for the cascaded rounds-9/10 lookahead: `FLOCK_NO_ZC_CASCADE4=1`
/// (exact '1') restores the cascade3 i=6/i=7 tail route within the same
/// binary (and with the deeper levels off, their respective routes).
/// OnceLock-latched: the environment is read once per process. Bit-identical
/// either way — same pure-reassociation argument as the levels above,
/// asserted by the cascade4 transcript-identity test.
#[inline]
fn cascade4_off() -> bool {
    #[cfg(test)]
    if ZC_CASCADE4_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_CASCADE4").is_some_and(|v| v == *"1"))
}

/// Test-only forced-off latch for the fifth-level cascade (rounds 11+12),
/// mirroring [`ZC_CASCADE4_FORCED_OFF`]. Both routes emit the same transcript.
#[cfg(test)]
pub(crate) static ZC_CASCADE5_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `FLOCK_NO_ZC_CASCADE5=1` restores the cascade4 i=8/i=9 tail route for an
/// exact same-binary A/B. Production defaults to the fifth cascade.
#[inline]
fn cascade5_off() -> bool {
    #[cfg(test)]
    if ZC_CASCADE5_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_CASCADE5").is_some_and(|v| v == *"1"))
}

/// Number of variables folded in round 1 via the additive-NTT univariate skip.
/// |Λ| = 2^K_SKIP = 64 elements; the round-1 prover message is two length-64
/// vectors of F128.
pub const K_SKIP: usize = 6;

/// Witness padding descriptor for URM work-skipping.
///
/// The witness is a sequence of `2^(m - k_log)` blocks of `2^k_log` bits each;
/// inside each block, bits `[0, useful_bits_per_block)` carry real data and
/// bits `[useful_bits_per_block, 2^k_log)` are zero padding. URM contributions
/// from a chunk of all-zero bits are themselves zero, so we can skip those
/// chunks and produce byte-identical output.
///
/// Use [`PaddingSpec::dense`] when the witness has no padding holes.
#[derive(Clone, Copy, Debug)]
pub struct PaddingSpec {
    pub k_log: usize,
    pub useful_bits_per_block: usize,
}

impl PaddingSpec {
    /// "No padding": every bit of the witness is treated as useful. Equivalent
    /// to the legacy URM path with no skipping.
    pub fn dense(m: usize) -> Self {
        Self {
            k_log: m,
            useful_bits_per_block: 1usize << m,
        }
    }
}

// ---------------------------------------------------------------------------
// Public types: claim, proof, error.
// ---------------------------------------------------------------------------

/// Evaluation claims on the multilinear extensions of a, b, c. **Note that
/// `a_eval`/`b_eval` and `c_eval` are claimed at *different points*** —
/// extract_c separates C from the AB sumcheck:
///
/// - `a_eval`, `b_eval` are at `(z, mlv_challenges)` — the AB sumcheck binds
///   the rest variables one at a time to fresh `ρ_r` challenges.
/// - `c_eval` is at `(z, r_rest)` — C is linear, so its eq-weighted sum
///   collapses immediately to an MLE evaluation at the original eq weights;
///   no per-round folding needed. Here `r_rest = r[K_SKIP..m]` from the
///   zerocheck challenge.
///
/// The downstream caller (R1CS prover + PCS) opens each commitment at its
/// own claim point. Two openings for a, b at the same point; one for c at
/// a different point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZerocheckClaim {
    /// Univariate-skip challenge sampled after round 1 (binds the K_SKIP
    /// skip variables).
    pub z: F128,
    /// AB sumcheck bind challenges, one per multilinear round; length = `m - K_SKIP`.
    pub mlv_challenges: Vec<F128>,
    /// Eq weights for the rest variables = the zerocheck challenge restricted
    /// to `r[K_SKIP..m]`. This is the *rest part of the c-claim's point*.
    /// Length = `m - K_SKIP`.
    pub r_rest: Vec<F128>,
    /// `â(z, mlv_challenges)`.
    pub a_eval: F128,
    /// `b̂(z, mlv_challenges)`.
    pub b_eval: F128,
    /// `ĉ(z, r_rest)` — at a *different point* than a_eval, b_eval.
    pub c_eval: F128,
}

/// All round messages the prover sends, in order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZerocheckProof {
    /// Round 1 (univariate skip): `P^{AB}(λ)` for λ ∈ Λ, length 2^K_SKIP.
    pub round1_ab: Vec<F128>,
    /// Round 1 (extract_c): `P^C(λ)` for λ ∈ Λ, length 2^K_SKIP. Sent separately
    /// from `round1_ab` so the verifier can evaluate the C-claim immediately
    /// and skip the C-column in all subsequent rounds.
    pub round1_c: Vec<F128>,
    /// Multilinear sumcheck rounds: each entry is `(P_r(1), P_r(∞))` via the
    /// Karatsuba ∞-trick. Length = `m - K_SKIP`.
    pub multilinear_rounds: Vec<(F128, F128)>,
    /// Final MLE evaluations sent at the end of the protocol.
    pub final_a_eval: F128,
    pub final_b_eval: F128,
    pub final_c_eval: F128,
}

/// Reasons the verifier may reject a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// `log_n` doesn't satisfy `log_n >= K_SKIP`.
    LogNTooSmall { log_n: usize, k_skip: usize },
    /// Round-1 messages have the wrong length (expected `2^K_SKIP`).
    BadRound1Length { expected: usize, got: usize },
    /// Wrong number of multilinear-round messages (expected `log_n - K_SKIP`).
    BadMultilinearRoundsLength { expected: usize, got: usize },
    /// `proof.final_c_eval` doesn't match the verifier's reconstruction
    /// `C_s · interpolate_at_z_on_lambda(round1_c, k_skip, z)`. Catches
    /// dishonesty in the round-1 C message or in the final c-eval claim.
    CEvalMismatch,
    /// The AB sumcheck final consistency check failed: the inner running
    /// claim after all rounds should equal `final_a_eval · final_b_eval`.
    /// Any inconsistency in `round1_ab`, in a multilinear round's
    /// `(P_r(1), P_r(∞))`, or in `final_a_eval` / `final_b_eval` propagates
    /// to this check.
    SumcheckFinalFailed,
}

// ---------------------------------------------------------------------------
// API: prove / verify.
// ---------------------------------------------------------------------------

/// Prove that `a(y) · b(y) ⊕ c(y) = 0` for all `y ∈ {0,1}^m`.
///
/// Inputs are LSB-first bit-packed byte vectors (each of length `2^m / 8`).
/// `m ≥ K_SKIP + N_INNER` (= 13). `challenger` supplies all verifier
/// randomness; the prover absorbs each of its messages into the challenger
/// before sampling the next challenge so the verifier (using the same
/// challenger implementation in lockstep) derives identical challenges.
///
/// Returns:
///   - the [`ZerocheckProof`] (raw round messages), and
///   - the [`ZerocheckClaim`] the higher-level caller will pass to its PCS.
pub fn prove_packed<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    prove_packed_padded(
        a_packed,
        b_packed,
        c_packed,
        m,
        &PaddingSpec::dense(m),
        challenger,
    )
}

/// Same as [`prove_packed`] but lets the caller declare a per-block padding
/// pattern so URM can skip work for chunks that fall entirely in the zero
/// padding of every block. Output is byte-identical to the dense path when
/// the padding bits are honestly zero.
pub fn prove_packed_padded<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    let (proof, claim, _) = prove_packed_padded_inner(
        a_packed, b_packed, c_packed, m, padding, false, None, None, challenger,
    );
    (proof, claim)
}

/// Variant of [`prove_packed_padded`] that ALSO returns the canonical
/// `s_hat_v_c` produced by the fused two-bank round-1 kernel. The downstream
/// PCS open uses this to skip `fold_1b_rows` for the c-claim — see
/// [`crate::pcs::ring_switch::round1_shift_reduce_extract_c_packed_padded_with_s_hat_v`].
///
/// Wire output `(ZerocheckProof, ZerocheckClaim)` is byte-identical to
/// [`prove_packed_padded`].
pub fn prove_packed_padded_capture_s_hat_v_c<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, CapturedSHatVC) {
    let (proof, claim, captured) = prove_packed_padded_inner(
        a_packed, b_packed, c_packed, m, padding, true, None, None, challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce s_hat_v_c"),
    )
}

/// The c-claim opening statistics round 1 captures for free.
///
/// The incumbent pair comes out of the same eight α-free banks: `s_hat_v_c` is the canonical
/// length-128 vector `fold_1b_rows` would produce, `quad` the length-512
/// four-bank form that lets the PCS open take C's `products` directly instead
/// of sweeping the combined basis. `collapse_s_hat_v_quad(quad, suffix[..2])`
/// reproduces `s_hat_v_c` exactly, so shipping either keeps the transcript.
/// The experimental `fold4` tensor has sixteen 128-element banks and likewise
/// collapses under `suffix[..4]`; it is present only behind the shared strict
/// DirectFold4 opt-in.
pub struct CapturedSHatVC {
    pub s_hat_v_c: Vec<F128>,
    pub quad: Vec<F128>,
    pub fold4: Option<Vec<F128>>,
    /// Sixty-four-bank form for the direct-fold8 route; collapses under
    /// `suffix[..6]` to `s_hat_v_c` exactly like `fold4` does under
    /// `suffix[..4]`. Present only behind the shared DirectFold8 opt-in.
    pub fold8: Option<Vec<F128>>,
}

/// Capture-`s_hat_v_c` prover that consumes a challenge-independent AB inner
/// transform prepared while the witness commitment was being built. The
/// original A and B buffers are still required and remain untouched for the
/// challenge-dependent round-2 fold.
pub fn prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    ab_inner: univariate_skip_optimized::Round1AbInner,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, CapturedSHatVC) {
    let (proof, claim, captured) = prove_packed_padded_inner(
        a_packed,
        b_packed,
        c_packed,
        m,
        padding,
        true,
        Some(ab_inner),
        None,
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce s_hat_v_c"),
    )
}

/// Ranked identity-C specialization of
/// [`prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab`]. The extra
/// buffer is the already-built lincheck stripe for C (= z); it lets round one
/// derive the legacy C message and all RingSwitch captures after a single
/// outer fold instead of draining the row-major witness into 32 field banks.
/// The proof and transcript remain byte-identical to the ordinary Fold4 path.
#[allow(clippy::too_many_arguments)]
pub fn prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab_and_lincheck_c<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    c_lincheck: &[u8],
    m: usize,
    padding: &PaddingSpec,
    ab_inner: univariate_skip_optimized::Round1AbInner,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, CapturedSHatVC) {
    let (proof, claim, captured) = prove_packed_padded_inner(
        a_packed,
        b_packed,
        c_packed,
        m,
        padding,
        true,
        Some(ab_inner),
        Some(c_lincheck),
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce s_hat_v_c"),
    )
}

/// Commit-tail fill, staging half (`FLOCK_NO_COMMIT_TAIL_FILL=1` kills; see
/// `gpu_commit::ENV_NO_COMMIT_TAIL_FILL`): derive the zerocheck challenges on
/// a forked challenger the moment the commit graph completes — the Merkle
/// root is the only commit-derived transcript input — and submit the
/// round-one C fold's GPU prefix inside the commit window's AB-arm tail.
/// The staged dispatch is the incumbent dispatch, merely earlier; zerocheck
/// entry consumes it only after the real transcript reproduces the identical
/// challenge vector, and abandons it otherwise (its output is then never
/// read, so the fill is byte-inert by construction).
///
/// Must be called on the thread that will later run zerocheck (the staged
/// job pins the fold-state lock to this thread). The caller guarantees
/// `c_lincheck` is fully written (release/acquire on the stripe-complete
/// flag) and stable for the rest of the prove.
pub fn stage_commit_tail_fill<C: Challenger>(
    forked: C,
    r1cs: &crate::r1cs::BlockR1cs,
    commitment: &crate::pcs::Commitment,
    c_lincheck: &[u8],
    padding: &PaddingSpec,
) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let mut forked = forked;
        let m = r1cs.m;
        let k_skip = K_SKIP;
        const N_INNER: usize = 7;
        if m < k_skip + N_INNER {
            return;
        }
        let t_stage = std::time::Instant::now();
        crate::proof::bind_statement(&mut forked, r1cs, commitment);
        forked.observe_label(b"flock-zerocheck-v0");
        // Mirror of the sampling block in `prove_packed_padded_inner`
        // below — same order, same constants, byte-identical challenges.
        let r_skip = forked.sample_f128_vec(k_skip);
        let r_outer = forked.sample_f128_vec(m - k_skip - N_INNER);
        let mut r = vec![F128::ZERO; m];
        r[..k_skip].copy_from_slice(&r_skip);
        for (i, val) in small_challenges_ghash().iter().enumerate() {
            r[k_skip + i] = *val;
        }
        for (i, val) in medium_challenges_ghash().iter().enumerate() {
            r[k_skip + 3 + i] = *val;
        }
        r[k_skip + N_INNER..].copy_from_slice(&r_outer);
        let staged = univariate_skip_optimized::stage_c_prelude_for_tail_fill(
            c_lincheck,
            m,
            padding.k_log,
            padding.useful_bits_per_block,
            &r,
        );
        if std::env::var_os("FLOCK_ZC_TIMING").is_some() {
            eprintln!(
                "[commit-tail-fill] stage at graph completion: staged={staged} {:.2} ms",
                t_stage.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (forked, r1cs, commitment, c_lincheck, padding);
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_packed_padded_inner<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    capture_s_hat_v_c: bool,
    precomputed_ab: Option<univariate_skip_optimized::Round1AbInner>,
    c_lincheck: Option<&[u8]>,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Option<CapturedSHatVC>) {
    let k_skip = K_SKIP;
    const N_INNER: usize = 7; // 3 small + 4 medium fixed-constant eq dims
    assert!(
        m >= k_skip + N_INNER,
        "prove requires m >= k_skip + N_INNER (= {})",
        k_skip + N_INNER
    );
    let expected_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), expected_bytes);
    assert_eq!(b_packed.len(), expected_bytes);
    assert_eq!(c_packed.len(), expected_bytes);
    let n_mlv = m - k_skip;

    challenger.observe_label(b"flock-zerocheck-v0");

    // ---- 1. Sample r (with protocol-fixed constants in the inner 7 dims) ----
    //
    // r layout:
    //   r[0..k_skip]                — sampled (used by verifier for the
    //                                  final check at S; not by the URM)
    //   r[k_skip..k_skip+3]         — protocol small-eq constants φ_8(0xF7..)
    //   r[k_skip+3..k_skip+7]       — protocol medium-eq constants β_i
    //   r[k_skip+7..m]              — sampled (the "outer" eq weights for
    //                                  the URM and multilinear rounds)
    let r_skip = challenger.sample_f128_vec(k_skip);
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    // ---- 3. Round 1: URM (extract_c, parallel) ----
    //
    // The optimized URM drops a `C_s = φ_8(0x1C)` scalar from its accumulators
    // (a prover-side optimization tied to the small-eq trick — see the
    // C_s factor analysis in `univariate_skip_optimized`). The wire format
    // must be in "naive" convention so the verifier doesn't need to know
    // about this internal optimization; we restore the C_s factor here.
    let zc_timing = std::env::var_os("FLOCK_ZC_TIMING").is_some();
    let cpu_r1 = crate::pcs::commit::commit_cpu_ms();
    let t_round1 = std::time::Instant::now();
    debug_assert_eq!(k_skip, 6, "ranked protocol fixes k_skip=6");
    let inv_table = InvNttTableByteSingleGf8::cached_standard_k6();
    let (round1_ab_opt, round1_c_opt, s_hat_v_c) = if let Some(ab_inner) = precomputed_ab.as_ref() {
        assert!(
            capture_s_hat_v_c,
            "precomputed AB path currently requires s_hat_v capture"
        );
        if let Some(c_lincheck) = c_lincheck {
            assert!(m == 32 || cfg!(test), "lincheck C reuse is ranked-only");
            assert_eq!(padding.k_log, 14, "lincheck C reuse fixes k_log=14");
            assert!(
                crate::pcs::ranked_direct_fold4_enabled() || cfg!(test),
                "lincheck C reuse requires ranked DirectFold4"
            );
            // Submit the C fold's GPU prefix BEFORE the AB completion. The
            // GPU is idle for the whole zerocheck window and round one has no
            // Fiat-Shamir dependency inside it (r was sampled above, the
            // transcript only advances after the round-one messages), so the
            // prefix runs concurrently with the AB completion AND with the
            // CPU's own share of the C fold.
            let c_prelude = crate::zerocheck::univariate_skip_optimized::round1_c_prelude(
                c_lincheck,
                m,
                padding.k_log,
                padding.useful_bits_per_block,
                &r,
            );
            // ZC-window GPU idle fill (`FLOCK_NO_ZC_IDLE_FILL=1` kills):
            // with the C fold's GPU prefix in flight, stage round two's
            // GPU-arm window setup behind it. Round two's eq split derives
            // from `r[k_skip+1..]` — bound above, strictly before the fold
            // submit — and a/b are the round-one operands, so every staged
            // input is Fiat-Shamir-available here. (The LINCHECK window's
            // eq table is NOT: its point is the mlv bind-challenge tail,
            // sampled only after the round messages below are observed.)
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            if c_prelude.gpu_in_flight()
                && crate::gpu_commit::zc_idle_fill_enabled()
                && crate::gpu_commit::zc_r2_idle_fill_viable()
            {
                crate::zerocheck::multilinear::stage_round2_gpu_window_from_r1_challenges(
                    a_packed, b_packed, m, k_skip, &r, padding,
                );
            }
            let cpu_ab = crate::pcs::commit::commit_cpu_ms();
            let t_ab = std::time::Instant::now();
            let ab = crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_ab_packed_padded_with_precomputed(
                ab_inner,
                m,
                k_skip,
                &r,
                padding,
            );
            if zc_timing {
                eprintln!(
                    "[zc-timing] round1 AB completion: {:.2} ms cpu={:.1}",
                    t_ab.elapsed().as_secs_f64() * 1e3,
                    crate::pcs::commit::commit_cpu_ms() - cpu_ab,
                );
            }
            let cpu_c = crate::pcs::commit::commit_cpu_ms();
            let t_c = std::time::Instant::now();
            if crate::pcs::ranked_direct_fold8_enabled() {
                let (c, s_hat_v_c, quad, fold8) =
                    crate::zerocheck::univariate_skip_optimized::round1_c_fold8_from_lincheck_stripe(
                        c_lincheck,
                        m,
                        padding.k_log,
                        k_skip,
                        padding.useful_bits_per_block,
                        &r,
                        inv_table,
                        c_prelude,
                    );
                if zc_timing {
                    eprintln!(
                        "[zc-timing] round1 lincheck-stripe C (fold8): {:.2} ms cpu={:.1}",
                        t_c.elapsed().as_secs_f64() * 1e3,
                        crate::pcs::commit::commit_cpu_ms() - cpu_c,
                    );
                }
                (
                    ab,
                    c,
                    Some(CapturedSHatVC {
                        s_hat_v_c,
                        quad,
                        fold4: None,
                        fold8: Some(fold8),
                    }),
                )
            } else {
                let (c, s_hat_v_c, quad, fold4) =
                    crate::zerocheck::univariate_skip_optimized::round1_c_fold4_from_lincheck_stripe(
                        c_lincheck,
                        m,
                        padding.k_log,
                        k_skip,
                        padding.useful_bits_per_block,
                        &r,
                        inv_table,
                        c_prelude,
                    );
                if zc_timing {
                    eprintln!(
                        "[zc-timing] round1 lincheck-stripe C: {:.2} ms cpu={:.1}",
                        t_c.elapsed().as_secs_f64() * 1e3,
                        crate::pcs::commit::commit_cpu_ms() - cpu_c,
                    );
                }
                (
                    ab,
                    c,
                    Some(CapturedSHatVC {
                        s_hat_v_c,
                        quad,
                        fold4: Some(fold4),
                        fold8: None,
                    }),
                )
            }
        } else if m == 32 && crate::pcs::ranked_direct_fold4_enabled() {
            let (ab, c, s_hat_v_c, quad, fold4) =
                crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4(
                    ab_inner,
                    c_packed,
                    m,
                    k_skip,
                    &r,
                    inv_table,
                    padding,
                );
            (
                ab,
                c,
                Some(CapturedSHatVC {
                    s_hat_v_c,
                    quad,
                    fold4: Some(fold4),
                    fold8: None,
                }),
            )
        } else {
            let (ab, c, s_hat_v_c, quad) =
                crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab(
                    ab_inner,
                    c_packed,
                    m,
                    k_skip,
                    &r,
                    inv_table,
                    padding,
                );
            (
                ab,
                c,
                Some(CapturedSHatVC {
                    s_hat_v_c,
                    quad,
                    fold4: None,
                    fold8: None,
                }),
            )
        }
    } else if capture_s_hat_v_c {
        let (ab, c, s_hat_v_c, quad) =
            crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad(
                a_packed,
                b_packed,
                c_packed,
                m,
                k_skip,
                &r,
                inv_table,
                padding,
            );
        (
            ab,
            c,
            Some(CapturedSHatVC {
                s_hat_v_c,
                quad,
                fold4: None,
                fold8: None,
            }),
        )
    } else {
        let (ab, c) = round1_shift_reduce_extract_c_packed_padded(
            a_packed, b_packed, c_packed, m, k_skip, &r, inv_table, padding,
        );
        (ab, c, None)
    };
    // The A-sized transform is dead after the round-1 message. Its byte length
    // exactly matches compact round two's delta storage, so retain its F128
    // allocation/layout and donate it instead of taking a fresh Vec<u8>.
    let compact_deltas =
        precomputed_ab.map(univariate_skip_optimized::Round1AbInner::into_scratch_bytes);
    let c_s = c_s_f128();
    let round1_ab: Vec<F128> = round1_ab_opt.iter().map(|x| c_s * *x).collect();
    let round1_c: Vec<F128> = round1_c_opt.iter().map(|x| c_s * *x).collect();
    if zc_timing {
        eprintln!(
            "[zc-timing] round1 URM: {:.2} ms cpu={:.1}",
            t_round1.elapsed().as_secs_f64() * 1e3,
            crate::pcs::commit::commit_cpu_ms() - cpu_r1
        );
    }

    // ---- 4. Observe round-1 message, sample z (URM fold point) ----
    challenger.observe_f128_slice(&round1_ab);
    challenger.observe_f128_slice(&round1_c);
    let z = challenger.sample_f128();

    // ---- 5. c_eval = ĉ(z, r_rest) via interpolation of round1_c at z ----
    //
    // round1_c (now in naive convention) carries `P^C(λ) = Σ_x eq(r_rest, x) · ĉ(λ, x)`
    // as its 2^k_skip evaluations on Λ. Interpolating to λ=z gives
    // `ĉ(z, r_rest)` directly (the eq-weighted sum collapses to the MLE
    // evaluation because ĉ is linear). This is **the c-claim** — at point
    // `(z, r_rest)`, *not* `(z, ρ-values)`. ~64 F128 muls + Lagrange weights.
    let final_c_eval = interpolate_at_z_on_lambda(&round1_c, k_skip, z);

    // ---- 6. Round 2: fused fold + first multilinear message ----
    //
    // Convention A wrapping: pass `mlv_arg[0] = ONE` so the function's output
    // `mlv_arg[0] · G(1)` becomes the bare `G(1)` we send on the wire. The
    // verifier samples ρ_1 after observing this message.
    let cpu_r2 = crate::pcs::commit::commit_cpu_ms();
    let t_round2 = std::time::Instant::now();
    let fold_table = UniSkipFoldTable::new(k_skip, z);
    let mut mlv_arg = vec![F128::ONE; n_mlv];
    mlv_arg[1..].copy_from_slice(&r[k_skip + 1..]);

    // Two-challenge symbolic lookahead (variant K): round three's message is a
    // quadratic in ρ₁, so its six coefficients ride along inside round two's
    // existing memory stall and rounds 3+4 collapse into a single double-fold
    // pass out of the compact state. Value-identical by construction — F128 is
    // exact and the transcript order below is untouched — so the kill switch
    // exists purely for same-binary A/B screening.
    //
    // `r[k_skip+1] = 0` (probability 2⁻¹²⁸) makes W1/W2 unrecoverable from the
    // parity split; that case falls back to the incumbent route, which stays
    // in the tree as the oracle anyway.
    let use_lookahead =
        (m == 32 || cfg!(test)) && n_mlv >= 6 && r[k_skip + 1] != F128::ZERO && !lookahead_off();

    let (compact_mlv, msg_1, msg_inf, lookahead) = if use_lookahead {
        let (compact, m1, mi, la) = uni_skip_fold_and_round_pair_compact_padded_lookahead(
            a_packed,
            b_packed,
            m,
            k_skip,
            &fold_table,
            &mlv_arg,
            padding,
            compact_deltas,
        );
        (compact, m1, mi, Some(la))
    } else {
        let (compact, m1, mi) = uni_skip_fold_and_round_pair_compact_padded_with_deltas(
            a_packed,
            b_packed,
            m,
            k_skip,
            &fold_table,
            &mlv_arg,
            padding,
            compact_deltas,
        );
        (compact, m1, mi, None)
    };

    if zc_timing {
        eprintln!(
            "[zc-timing] round2 fused fold: {:.2} ms cpu={:.1}",
            t_round2.elapsed().as_secs_f64() * 1e3,
            crate::pcs::commit::commit_cpu_ms() - cpu_r2
        );
    }
    let cpu_tail = crate::pcs::commit::commit_cpu_ms();
    let t_tail = std::time::Instant::now();
    let mut multilinear_msgs = Vec::with_capacity(n_mlv);
    multilinear_msgs.push((msg_1, msg_inf));
    challenger.observe_f128(msg_1);
    challenger.observe_f128(msg_inf);
    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);
    mlv_rhos.push(challenger.sample_f128());

    // ---- 7. Rounds 3..(n_mlv + 1) — AB only (c is done) ----
    //
    // Iter i: fold (a, b) at ρ_{i+1}, compute round (i+3) message, sample
    // ρ_{i+2}. Use the fused parallel path while log_n ≥ 15; below that the
    // 12..14 are structurally valid but open a fixed 128-chunk Rayon region
    // over too little work; below 12, SplitEqGhash cannot form lo_size ≥ 2
    // under MAX_N_HI = 9 at all. Fall back to fold_in_place_pair +
    // round_pair_naive for this serial tail.
    //
    // The first challenge is applied directly to round two's compact
    // anchor+packed-delta representation.  Composing rho into the 32 KiB byte
    // table removes the two field multiplications per output that the generic
    // pair fold would require, while materializing exactly the ordinary
    // post-fold tables expected by all subsequent rounds.
    let tail_round_timing = std::env::var_os("FLOCK_ZC_TAIL_ROUND_TIMING").is_some();

    // Cascade the lookahead one level deeper (rounds 5+6, see
    // `fold2_compact_and_round45_into`): the K pass materializes each round-4
    // output group in registers before its store — the same position round
    // two was in before the round-3 promotion — so round five's message rides
    // it as a deferred quadratic in the not-yet-sampled ρ₃, and rounds 5+6
    // then collapse into one plain composed double-fold, deleting tail
    // iterations i = 2 and i = 3 (their DRAM passes and their FS-serialized
    // round boundaries). Same value-identity argument as the lookahead: pure
    // reassociation of exact F128 arithmetic, transcript order untouched.
    //
    // `r[k_skip+3] = 0` would make W1'/W2' unrecoverable from the eq parity
    // split (at the ranked shape that slot is the protocol constant β₀ ≠ 0;
    // for a sampled slot it is probability 2⁻¹²⁸); that case falls back to
    // the incumbent route, which stays in the tree as the oracle anyway.
    // n_mlv ≥ 7 keeps every eq split at lo_size ≥ 2 and the composed input
    // ≥ 32. Kill switch: FLOCK_NO_ZC_CASCADE2=1 (exact '1').
    let use_cascade = use_lookahead && n_mlv >= 7 && r[k_skip + 3] != F128::ZERO && !cascade2_off();

    // Cascade one level deeper still (rounds 7+8, see
    // `fold2_plain_and_round67_into`): the composed 5+6 pass materializes each
    // output group in registers before its store — the same position the K
    // pass was in before the round-5 promotion — so round seven's message
    // rides it as a deferred quadratic in the not-yet-sampled ρ₅, and rounds
    // 7+8 then collapse into one more plain composed double-fold, deleting
    // tail iterations i = 4 and i = 5 (their DRAM passes and one of their
    // FS-serialized round boundaries). Same value-identity argument again:
    // pure reassociation of exact F128 arithmetic, transcript order untouched.
    //
    // `r[k_skip+5] = 0` would make W1''/W2'' unrecoverable from the eq parity
    // split (at every shape with m ≥ 13 that slot is the protocol constant
    // β₂ ≠ 0; for a sampled slot it is probability 2⁻¹²⁸); that case falls
    // back to the cascade2 route, which stays in the tree as the oracle
    // anyway. n_mlv ≥ 8 keeps the composed-7/8 input ≥ 16 (its own floor) and
    // every eq split at lo_size ≥ 2. Kill switch: FLOCK_NO_ZC_CASCADE3=1
    // (exact '1').
    let use_cascade3 = use_cascade && n_mlv >= 8 && r[k_skip + 5] != F128::ZERO && !cascade3_off();

    // Cascade one level deeper still (rounds 9+10, see the cascade3 comment
    // above — the induction step is identical): the composed 7+8 pass
    // materializes each output group in registers before its store, so round
    // nine's message rides it as a deferred quadratic in the not-yet-sampled
    // ρ₇, and rounds 9+10 then collapse into one more plain composed
    // double-fold, deleting tail iterations i = 6 and i = 7 (their 32 MiB +
    // 16 MiB reads and 16 MiB + 8 MiB writes become one 32 MiB read + 8 MiB
    // write, plus one fewer FS-serialized round boundary). Same value-identity
    // argument again: pure reassociation of exact F128 arithmetic, transcript
    // order untouched.
    //
    // `r[k_skip+7] = 0` would make the parity split unrecoverable (protocol
    // constant ≠ 0 at every ranked-relevant shape; probability 2⁻¹²⁸ for a
    // sampled slot); that case falls back to the cascade3 route, which stays
    // in the tree as the oracle anyway. n_mlv ≥ 10 keeps the composed-9/10
    // input ≥ 16 and every eq split at lo_size ≥ 2. Kill switch:
    // FLOCK_NO_ZC_CASCADE4=1 (exact '1').
    let use_cascade4 =
        use_cascade3 && n_mlv >= 10 && r[k_skip + 7] != F128::ZERO && !cascade4_off();

    // Extend the same round-agnostic lookahead through rounds 11+12. The
    // rounds-9/10 pass carries round eleven's deferred quadratic, then one
    // final composed fold replaces loop iterations i=8 and i=9. n_mlv >= 12
    // keeps that final composed input at the kernel's 16-element floor.
    let use_cascade5 =
        use_cascade4 && n_mlv >= 12 && r[k_skip + 9] != F128::ZERO && !cascade5_off();

    // `loop_start` is the first tail iteration this route has not already
    // produced. The loop body's `r_next[1..] = r[k_skip + i + 2..]` is already
    // indexed by `i`, so starting at 2 (or 4) needs no other change.
    let (mut a_mlv, mut b_mlv, loop_start) = if let Some(la) = lookahead {
        // Round three: evaluate the deferred quadratic. No pass at all.
        let (first_m1, first_mi) = eval_round3_lookahead(&la, mlv_rhos[0]);
        multilinear_msgs.push((first_m1, first_mi));
        challenger.observe_f128(first_m1);
        challenger.observe_f128(first_mi);
        mlv_rhos.push(challenger.sample_f128());

        // Rounds three and four now fold together in one pass over the compact
        // state, replacing the T3 reconstruction *and* tail iteration i = 1.
        let t_k = std::time::Instant::now();
        let n_groups = compact_mlv.len() / 2;
        let mut r_next4 = vec![F128::ONE; n_mlv - 2];
        r_next4[1..].copy_from_slice(&r[k_skip + 3..]);
        // Unpinned: these become the loop-round arm's no-copy wrap targets
        // next round, and the pinned slots already carry process-lifetime
        // Metal views (see `fold_compact_and_compute_round_pair`).
        let mut a_out = crate::scratch::take_f128_unpinned(n_groups);
        let mut b_out = crate::scratch::take_f128_unpinned(n_groups);
        if use_cascade {
            let (m4_1, m4_inf, la5) = fold2_compact_and_round45_into(
                &compact_mlv,
                &fold_table,
                mlv_rhos[0],
                mlv_rhos[1],
                &r_next4,
                &mut a_out,
                &mut b_out,
            );
            if tail_round_timing {
                eprintln!(
                    "[zc-tail-rounds] K double fold + round4 (cascade +W', out n={n_groups}): {:.2} ms",
                    t_k.elapsed().as_secs_f64() * 1e3
                );
            }
            compact_mlv.recycle();
            multilinear_msgs.push((m4_1, m4_inf));
            challenger.observe_f128(m4_1);
            challenger.observe_f128(m4_inf);
            mlv_rhos.push(challenger.sample_f128());

            // Round five: evaluate the deferred quadratic at ρ₃. No pass.
            let (m5_1, m5_inf) = eval_round3_lookahead(&la5, mlv_rhos[2]);
            multilinear_msgs.push((m5_1, m5_inf));
            challenger.observe_f128(m5_1);
            challenger.observe_f128(m5_inf);
            mlv_rhos.push(challenger.sample_f128());

            // Rounds five and six now fold together in one plain composed
            // pass (ρ₃ and ρ₄ at once), replacing tail iterations i = 2 and
            // i = 3: their 512 MiB + 256 MiB reads and 256 MiB + 128 MiB
            // writes become one 512 MiB read + 128 MiB write.
            let t_c = std::time::Instant::now();
            let quarter = n_groups / 4;
            let mut r_next6 = vec![F128::ONE; n_mlv - 4];
            r_next6[1..].copy_from_slice(&r[k_skip + 5..]);
            // Unpinned for the same reason as the K outputs above.
            let mut a2_out = crate::scratch::take_f128_unpinned(quarter);
            let mut b2_out = crate::scratch::take_f128_unpinned(quarter);
            if use_cascade3 {
                // Level three: the composed 5+6 pass additionally carries the
                // deferred round-seven quadratic — zero extra traversals.
                let (m6_1, m6_inf, la7) = fold2_plain_and_round67_into(
                    &a_out,
                    &b_out,
                    &mut a2_out,
                    &mut b2_out,
                    mlv_rhos[2],
                    mlv_rhos[3],
                    &r_next6,
                );
                if tail_round_timing {
                    eprintln!(
                        "[zc-tail-rounds] composed rounds 5+6 fold (cascade +W'', out n={quarter}): {:.2} ms",
                        t_c.elapsed().as_secs_f64() * 1e3
                    );
                }
                crate::scratch::give_f128(a_out);
                crate::scratch::give_f128(b_out);
                multilinear_msgs.push((m6_1, m6_inf));
                challenger.observe_f128(m6_1);
                challenger.observe_f128(m6_inf);
                mlv_rhos.push(challenger.sample_f128());

                // Round seven: evaluate the deferred quadratic at ρ₅. No pass.
                let (m7_1, m7_inf) = eval_round3_lookahead(&la7, mlv_rhos[4]);
                multilinear_msgs.push((m7_1, m7_inf));
                challenger.observe_f128(m7_1);
                challenger.observe_f128(m7_inf);
                mlv_rhos.push(challenger.sample_f128());

                // Rounds seven and eight fold together in one more plain
                // composed pass (ρ₅ and ρ₆ at once), replacing tail
                // iterations i = 4 and i = 5: their 128 MiB + 64 MiB reads
                // and 64 MiB + 32 MiB writes become one 128 MiB read +
                // 32 MiB write. Under cascade4 the same pass additionally
                // carries the deferred round-nine quadratic — zero extra
                // traversals.
                let t_c3 = std::time::Instant::now();
                let sixteenth = n_groups / 16;
                let mut r_next8 = vec![F128::ONE; n_mlv - 6];
                r_next8[1..].copy_from_slice(&r[k_skip + 7..]);
                // Unpinned for the same reason as the K outputs above.
                let mut a3_out = crate::scratch::take_f128_unpinned(sixteenth);
                let mut b3_out = crate::scratch::take_f128_unpinned(sixteenth);
                let (m8_1, m8_inf, la9) = if use_cascade4 {
                    let (m8_1, m8_inf, la9) = fold2_plain_and_round67_into(
                        &a2_out,
                        &b2_out,
                        &mut a3_out,
                        &mut b3_out,
                        mlv_rhos[4],
                        mlv_rhos[5],
                        &r_next8,
                    );
                    (m8_1, m8_inf, Some(la9))
                } else {
                    let (m8_1, m8_inf) = fold2_plain_and_round6_into(
                        &a2_out,
                        &b2_out,
                        &mut a3_out,
                        &mut b3_out,
                        mlv_rhos[4],
                        mlv_rhos[5],
                        &r_next8,
                    );
                    (m8_1, m8_inf, None)
                };
                if tail_round_timing {
                    eprintln!(
                        "[zc-tail-rounds] composed rounds 7+8 fold (out n={sixteenth}, cascade4={}): {:.2} ms",
                        la9.is_some(),
                        t_c3.elapsed().as_secs_f64() * 1e3
                    );
                }
                crate::scratch::give_f128(a2_out);
                crate::scratch::give_f128(b2_out);
                multilinear_msgs.push((m8_1, m8_inf));
                challenger.observe_f128(m8_1);
                challenger.observe_f128(m8_inf);
                mlv_rhos.push(challenger.sample_f128());
                if let Some(la9) = la9 {
                    // Round nine: evaluate the deferred quadratic at ρ₇. No
                    // pass at all.
                    let (m9_1, m9_inf) = eval_round3_lookahead(&la9, mlv_rhos[6]);
                    multilinear_msgs.push((m9_1, m9_inf));
                    challenger.observe_f128(m9_1);
                    challenger.observe_f128(m9_inf);
                    mlv_rhos.push(challenger.sample_f128());

                    // Rounds nine and ten fold together in one more plain
                    // composed pass (ρ₇ and ρ₈ at once), replacing tail
                    // iterations i = 6 and i = 7. Under cascade5 this pass
                    // also carries round eleven's deferred quadratic.
                    let t_c4 = std::time::Instant::now();
                    let sixtyfourth = n_groups / 64;
                    let mut r_next10 = vec![F128::ONE; n_mlv - 8];
                    r_next10[1..].copy_from_slice(&r[k_skip + 9..]);
                    // Unpinned for the same reason as the K outputs above.
                    let mut a4_out = crate::scratch::take_f128_unpinned(sixtyfourth);
                    let mut b4_out = crate::scratch::take_f128_unpinned(sixtyfourth);
                    let (m10_1, m10_inf, la11) = if use_cascade5 {
                        let (m10_1, m10_inf, la11) = fold2_plain_and_round67_into(
                            &a3_out,
                            &b3_out,
                            &mut a4_out,
                            &mut b4_out,
                            mlv_rhos[6],
                            mlv_rhos[7],
                            &r_next10,
                        );
                        (m10_1, m10_inf, Some(la11))
                    } else {
                        let (m10_1, m10_inf) = fold2_plain_and_round6_into(
                            &a3_out,
                            &b3_out,
                            &mut a4_out,
                            &mut b4_out,
                            mlv_rhos[6],
                            mlv_rhos[7],
                            &r_next10,
                        );
                        (m10_1, m10_inf, None)
                    };
                    if tail_round_timing {
                        eprintln!(
                            "[zc-tail-rounds] composed rounds 9+10 fold (out n={sixtyfourth}, cascade5={}): {:.2} ms",
                            la11.is_some(),
                            t_c4.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    crate::scratch::give_f128(a3_out);
                    crate::scratch::give_f128(b3_out);
                    multilinear_msgs.push((m10_1, m10_inf));
                    challenger.observe_f128(m10_1);
                    challenger.observe_f128(m10_inf);
                    mlv_rhos.push(challenger.sample_f128());
                    if let Some(la11) = la11 {
                        // Round eleven: evaluate the deferred quadratic at
                        // ρ₉ without traversing the tables.
                        let (m11_1, m11_inf) = eval_round3_lookahead(&la11, mlv_rhos[8]);
                        multilinear_msgs.push((m11_1, m11_inf));
                        challenger.observe_f128(m11_1);
                        challenger.observe_f128(m11_inf);
                        mlv_rhos.push(challenger.sample_f128());

                        // Bind ρ₉ and ρ₁₀ together and emit round twelve,
                        // replacing tail iterations i = 8 and i = 9.
                        let t_c5 = std::time::Instant::now();
                        let twofiftysixth = n_groups / 256;
                        let mut r_next12 = vec![F128::ONE; n_mlv - 10];
                        r_next12[1..].copy_from_slice(&r[k_skip + 11..]);
                        let mut a5_out = crate::scratch::take_f128_unpinned(twofiftysixth);
                        let mut b5_out = crate::scratch::take_f128_unpinned(twofiftysixth);
                        let (m12_1, m12_inf) = fold2_plain_and_round6_into(
                            &a4_out,
                            &b4_out,
                            &mut a5_out,
                            &mut b5_out,
                            mlv_rhos[8],
                            mlv_rhos[9],
                            &r_next12,
                        );
                        if tail_round_timing {
                            eprintln!(
                                "[zc-tail-rounds] composed rounds 11+12 fold (out n={twofiftysixth}): {:.2} ms",
                                t_c5.elapsed().as_secs_f64() * 1e3
                            );
                        }
                        crate::scratch::give_f128(a4_out);
                        crate::scratch::give_f128(b4_out);
                        multilinear_msgs.push((m12_1, m12_inf));
                        challenger.observe_f128(m12_1);
                        challenger.observe_f128(m12_inf);
                        mlv_rhos.push(challenger.sample_f128());
                        (a5_out, b5_out, 10usize)
                    } else {
                        (a4_out, b4_out, 8usize)
                    }
                } else {
                    (a3_out, b3_out, 6usize)
                }
            } else {
                let (m6_1, m6_inf) = fold2_plain_and_round6_into(
                    &a_out,
                    &b_out,
                    &mut a2_out,
                    &mut b2_out,
                    mlv_rhos[2],
                    mlv_rhos[3],
                    &r_next6,
                );
                if tail_round_timing {
                    eprintln!(
                        "[zc-tail-rounds] composed rounds 5+6 fold (out n={quarter}): {:.2} ms",
                        t_c.elapsed().as_secs_f64() * 1e3
                    );
                }
                crate::scratch::give_f128(a_out);
                crate::scratch::give_f128(b_out);
                multilinear_msgs.push((m6_1, m6_inf));
                challenger.observe_f128(m6_1);
                challenger.observe_f128(m6_inf);
                mlv_rhos.push(challenger.sample_f128());
                (a2_out, b2_out, 4usize)
            }
        } else {
            let (m4_1, m4_inf) = fold2_compact_and_round4_into(
                &compact_mlv,
                &fold_table,
                mlv_rhos[0],
                mlv_rhos[1],
                &r_next4,
                &mut a_out,
                &mut b_out,
            );
            if tail_round_timing {
                eprintln!(
                    "[zc-tail-rounds] K double fold + round4 (out n={n_groups}): {:.2} ms",
                    t_k.elapsed().as_secs_f64() * 1e3
                );
            }
            compact_mlv.recycle();
            multilinear_msgs.push((m4_1, m4_inf));
            challenger.observe_f128(m4_1);
            challenger.observe_f128(m4_inf);
            mlv_rhos.push(challenger.sample_f128());
            (a_out, b_out, 2usize)
        }
    } else {
        let t_t3 = std::time::Instant::now();
        let mut first_r_next = vec![F128::ONE; n_mlv - 1];
        first_r_next[1..].copy_from_slice(&r[k_skip + 2..]);
        let (a_mlv, b_mlv, first_m1, first_mi) = fold_compact_and_compute_round_pair(
            &compact_mlv,
            &fold_table,
            mlv_rhos[0],
            &first_r_next,
        );
        if tail_round_timing {
            eprintln!(
                "[zc-tail-rounds] T3 compact fold (out n={}): {:.2} ms",
                a_mlv.len(),
                t_t3.elapsed().as_secs_f64() * 1e3
            );
        }
        compact_mlv.recycle();
        multilinear_msgs.push((first_m1, first_mi));
        challenger.observe_f128(first_m1);
        challenger.observe_f128(first_mi);
        mlv_rhos.push(challenger.sample_f128());
        (a_mlv, b_mlv, 1usize)
    };

    // Ping-pong scratch buffers for the remaining fused path: each fused round folds
    // (a_mlv, b_mlv) of size N into size N/2. Rather than allocating — and,
    // worse, `munmap`-ing, which is single-threaded and caps the tail's
    // parallel speedup — a fresh 64 MB buffer per round, we alternate between
    // two persistent buffers. Scratch capacity = N/2 (the largest fused
    // output); only needed when the first round is actually fused.
    let n_in = a_mlv.len();
    let (mut a_nxt, mut b_nxt) = if n_in >= 1024 {
        (
            crate::scratch::take_f128(n_in / 2),
            crate::scratch::take_f128(n_in / 2),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    // H2 engagement evidence: E-core chunks claimed across the loop rounds
    // (T3's hetero drain is already behind us, so the delta is loop-only).
    let hetero_trace = std::env::var_os("FLOCK_ZC_TAIL_HETERO_TRACE").is_some();
    let hetero_claimed_before = crate::epool::helper_chunks_claimed();
    for i in loop_start..(n_mlv - 1) {
        let t_round_i = std::time::Instant::now();
        let rho_prev = mlv_rhos[i];
        let log_n_before = a_mlv.len().trailing_zeros() as usize;

        // r_next for the next round's message: length log_n_before - 1.
        // r_next[0] = ONE (Convention A factor); r_next[1..] are the eq
        // weights for the remaining variables = r[k_skip + i + 2..m].
        let mut r_next = vec![F128::ONE; log_n_before - 1];
        r_next[1..].copy_from_slice(&r[k_skip + i + 2..]);

        let (m1, mi) = if log_n_before >= 15 {
            let half = a_mlv.len() / 2;
            let (m1, mi) = fold_and_compute_round_pair_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..half],
                &mut b_nxt[..half],
                rho_prev,
                &r_next,
            );
            // Swap current <-> scratch, then shrink the new current to the
            // folded size. The old (larger) buffer becomes scratch; we only
            // ever write its leading `half` slots next round, so its stale
            // length is harmless.
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(half);
            b_mlv.truncate(half);
            (m1, mi)
        } else {
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
            round_pair_naive(&a_mlv, &b_mlv, &r_next)
        };

        multilinear_msgs.push((m1, mi));
        challenger.observe_f128(m1);
        challenger.observe_f128(mi);
        mlv_rhos.push(challenger.sample_f128());
        if tail_round_timing {
            eprintln!(
                "[zc-tail-rounds] loop i={i} (log_n {log_n_before}): {:.2} ms",
                t_round_i.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    if hetero_trace {
        eprintln!(
            "[zc-tail] hetero loop rounds: {} chunks claimed by E-cores",
            crate::epool::helper_chunks_claimed() - hetero_claimed_before
        );
    }

    // ---- 8. Final binding at ρ_{n_mlv} (the last challenge) ----
    let rho_last = *mlv_rhos.last().expect("at least one ρ sampled");
    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_last);
    debug_assert_eq!(a_mlv.len(), 1);
    debug_assert_eq!(b_mlv.len(), 1);

    let final_a_eval = a_mlv[0];
    let final_b_eval = b_mlv[0];

    // ---- Fiat–Shamir: bind the final â, b̂ claims into the transcript ----
    //
    // These two claims are reduced downstream by lincheck via a *single*
    // random-linear-combination check with coefficient α (`target = α·v_a + v_b`,
    // see `lincheck`). That batching is only sound if α is sampled *after*
    // (v_a, v_b) are committed to the transcript — otherwise a prover that knows
    // α can pick (v_a, v_b) to satisfy the one batched equation while violating
    // the individual checks. So observe them here, before any later challenge
    // (the next one drawn is lincheck's α). `final_c_eval` needs no observe — the
    // verifier recomputes it from the already-absorbed `round1_c`/`z` and rejects
    // on mismatch (see `verify`), so it is already transcript-bound.
    challenger.observe_f128(final_a_eval);
    challenger.observe_f128(final_b_eval);

    // Recycle the four tail buffers (the two len-1 survivors still own their
    // full round-2 capacity) for the next phase/prove.
    crate::scratch::give_f128(a_mlv);
    crate::scratch::give_f128(b_mlv);
    crate::scratch::give_f128(a_nxt);
    crate::scratch::give_f128(b_nxt);

    if zc_timing {
        eprintln!(
            "[zc-timing] rounds 3+ tail: {:.2} ms cpu={:.1}",
            t_tail.elapsed().as_secs_f64() * 1e3,
            crate::pcs::commit::commit_cpu_ms() - cpu_tail
        );
    }

    let r_rest: Vec<F128> = r[k_skip..].to_vec();

    let proof = ZerocheckProof {
        round1_ab,
        round1_c,
        multilinear_rounds: multilinear_msgs,
        final_a_eval,
        final_b_eval,
        final_c_eval,
    };
    let claim = ZerocheckClaim {
        z,
        mlv_challenges: mlv_rhos,
        r_rest,
        a_eval: final_a_eval,
        b_eval: final_b_eval,
        c_eval: final_c_eval,
    };
    (proof, claim, s_hat_v_c)
}

/// Verify a zerocheck proof for an instance over `{0,1}^log_n`.
///
/// Walks the challenger in lockstep with the prover, samples the same
/// challenges, and checks every round's consistency equation.
///
/// On accept: returns the [`ZerocheckClaim`] the caller must check against
/// its PCS opening of `â`, `b̂`, `ĉ`.
/// On reject: returns a [`VerifyError`] indicating which check failed.
pub fn verify<C: Challenger>(
    log_n: usize,
    proof: &ZerocheckProof,
    challenger: &mut C,
) -> Result<ZerocheckClaim, VerifyError> {
    let m = log_n;
    let k_skip = K_SKIP;
    const N_INNER: usize = 7;

    if m < k_skip + N_INNER {
        return Err(VerifyError::LogNTooSmall { log_n: m, k_skip });
    }
    let n_mlv = m - k_skip;
    let ell = 1usize << k_skip;

    // ---- Shape checks ----
    if proof.round1_ab.len() != ell {
        return Err(VerifyError::BadRound1Length {
            expected: ell,
            got: proof.round1_ab.len(),
        });
    }
    if proof.round1_c.len() != ell {
        return Err(VerifyError::BadRound1Length {
            expected: ell,
            got: proof.round1_c.len(),
        });
    }
    if proof.multilinear_rounds.len() != n_mlv {
        return Err(VerifyError::BadMultilinearRoundsLength {
            expected: n_mlv,
            got: proof.multilinear_rounds.len(),
        });
    }

    challenger.observe_label(b"flock-zerocheck-v0");

    // ---- Re-derive r (in lockstep with prove_packed) ----
    let r_skip = challenger.sample_f128_vec(k_skip);
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    // ---- Observe round-1 messages, sample z ----
    challenger.observe_f128_slice(&proof.round1_ab);
    challenger.observe_f128_slice(&proof.round1_c);
    let z = challenger.sample_f128();

    // ---- Reconstruct ĉ(z, r_rest) from round1_c ----
    //
    // P^C has degree < 2^k_skip in λ (C is linear, summed against eq); ell
    // evaluations on Λ uniquely interpolate to z. round1_c is in naive
    // convention (the prover restored the C_s factor before sending), so
    // `ĉ(z, r_rest) = P^C(z)` directly.
    let computed_c_eval = interpolate_at_z_on_lambda(&proof.round1_c, k_skip, z);
    if computed_c_eval != proof.final_c_eval {
        return Err(VerifyError::CEvalMismatch);
    }

    // ---- Reconstruct the initial AB running claim ----
    //
    // P^{AB}(z) requires the polynomial in λ of degree < 2·ell to be evaluated
    // at z. The prover sent only ell evaluations on Λ — not enough on its own.
    // The verifier uses the **zerocheck assumption** `P^{AB}(λ) + P^C(λ) = 0`
    // for `λ ∈ S`. Together with the ell Λ-evaluations of the combined
    // polynomial, that's 2·ell evaluations — enough to interpolate the
    // combined polynomial at z. Then `P^{AB}(z) = P^{combined}(z) − P^C(z)`,
    // which in char-2 is `P^{combined}(z) + P^C(z)`.
    //
    // If the prover's witness is dishonest the S-zero assumption fails, the
    // reconstructed c_0 is wrong, and the running-claim chain ends at a value
    // inconsistent with `â · b̂`. We catch that at the final sumcheck check.
    let combined_at_lambda: Vec<F128> = proof
        .round1_ab
        .iter()
        .zip(&proof.round1_c)
        .map(|(x, y)| *x + *y)
        .collect();
    let combined_at_z = interpolate_at_z_combined(&combined_at_lambda, k_skip, z);
    let p_c_at_z = interpolate_at_z_on_lambda(&proof.round1_c, k_skip, z);
    let mut c_running = combined_at_z + p_c_at_z;

    // ---- Multilinear sumcheck chain ----
    //
    // The propagated running claim is the *inner* polynomial value G(ρ),
    // not the full per-round polynomial P(ρ) = eq(r_eq, ρ) · G(ρ). The eq
    // factor for the just-bound variable is absorbed by the next round's
    // consistency check via the identity
    //   G_{r-1}(ρ_{r-1}) = (1 + r_eq_r) · G_r(0) + r_eq_r · G_r(1).
    //
    // Round r (0-indexed i = r − 2) binds the i-th rest variable with eq weight
    // r[k_skip + i]. The prover sends `(G(1), G(∞))` (Convention A — no
    // factor). Verifier:
    //   1. reconstruct G(0) from consistency `c_running = (1+r_eq)·G(0) + r_eq·G(1)`,
    //   2. observe message, sample ρ_i,
    //   3. update `c_running ← G(ρ_i)`,
    //      where `G(X) = G(0)·(1+X) + G(1)·X + G(∞)·X·(X+1)` (char-2 quadratic
    //      interpolation through G(0), G(1), G(∞)).
    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);
    for (i, &(msg_1, msg_inf)) in proof.multilinear_rounds.iter().enumerate() {
        let r_eq = r[k_skip + i];
        let one_plus_r_eq = F128::ONE + r_eq;

        let g1 = msg_1;
        let g_inf = msg_inf;
        let g0 = (c_running + r_eq * g1) * one_plus_r_eq.inv();

        challenger.observe_f128(msg_1);
        challenger.observe_f128(msg_inf);
        let rho = challenger.sample_f128();
        mlv_rhos.push(rho);

        let one_plus_rho = F128::ONE + rho;
        // G(ρ) = G(0)·(1+ρ) + G(1)·ρ + G(∞)·ρ·(1+ρ).
        c_running = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
    }

    // ---- AB sumcheck final consistency ----
    //
    // After all variables are bound, the inner running claim is just the
    // polynomial without the eq weighting:
    //   G_final(ρ_all) = â(z, ρ) · b̂(z, ρ) = final_a_eval · final_b_eval.
    // (The eq factors were absorbed round-by-round into the consistency checks,
    // never accumulating into the running claim.)
    let r_rest: Vec<F128> = r[k_skip..].to_vec();
    let expected_final = proof.final_a_eval * proof.final_b_eval;
    if c_running != expected_final {
        return Err(VerifyError::SumcheckFinalFailed);
    }

    // ---- Fiat–Shamir: bind the final â, b̂ claims (mirrors `prove_packed_padded_inner`) ----
    //
    // Must observe at the same transcript position as the prover, before the
    // next challenge (lincheck's α) is drawn, so the α-batched reduction of
    // these two claims is sound. `final_c_eval` is already bound via the
    // recompute-and-compare above, so it is not observed.
    challenger.observe_f128(proof.final_a_eval);
    challenger.observe_f128(proof.final_b_eval);

    Ok(ZerocheckClaim {
        z,
        mlv_challenges: mlv_rhos,
        r_rest,
        a_eval: proof.final_a_eval,
        b_eval: proof.final_b_eval,
        c_eval: proof.final_c_eval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

    /// SplitMix64 PRNG, deterministic.
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
    }

    /// Pack three Boolean vectors into the (a_packed, b_packed, c_packed)
    /// shape that `prove_packed` consumes.
    fn pack_abc(a: &[bool], b: &[bool], c: &[bool]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        use univariate_skip::pack_bits;
        (pack_bits(a), pack_bits(b), pack_bits(c))
    }

    /// `prove` runs end-to-end at the smallest valid m (= k_skip + N_INNER = 13)
    /// without panicking, and produces output of the right shape.
    ///
    /// We can't yet check the proof is *accepted* (verify is a stub), but the
    /// structural sanity here catches:
    ///   - mismatched challenger observe/sample sequence
    ///   - wrong slice lengths in r / mlv_arg / r_next at any round
    ///   - any unreachable assert in the underlying functions
    #[test]
    fn prove_runs_end_to_end() {
        for &m in &[13usize, 14, 15, 16] {
            let mut rng = Rng::new(m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            // Honest witness: c = a AND b, so a·b ⊕ c = 0 on the hypercube.
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut challenger = FsChallenger::new(b"flock-test-v0");
            let (proof, claim) = prove_packed(&a_p, &b_p, &c_p, m, &mut challenger);

            // Shape checks.
            assert_eq!(proof.round1_ab.len(), 1usize << K_SKIP, "m={m}");
            assert_eq!(proof.round1_c.len(), 1usize << K_SKIP, "m={m}");
            assert_eq!(proof.multilinear_rounds.len(), m - K_SKIP, "m={m}");
            assert_eq!(claim.mlv_challenges.len(), m - K_SKIP, "m={m}");

            // Claim's eval fields agree with the proof's final evals.
            assert_eq!(claim.a_eval, proof.final_a_eval, "m={m}");
            assert_eq!(claim.b_eval, proof.final_b_eval, "m={m}");
            assert_eq!(claim.c_eval, proof.final_c_eval, "m={m}");
        }
    }

    /// **Prove→verify roundtrip**: an honest proof verifies cleanly, and the
    /// claim returned by `verify` is byte-for-byte equal to the claim returned
    /// by `prove`.
    #[test]
    fn prove_verify_roundtrip_honest() {
        for &m in &[13usize, 14, 15, 16] {
            let mut rng = Rng::new(1000 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, claim_p) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let result = verify(m, &proof, &mut ch_verify);
            let claim_v = result.unwrap_or_else(|e| panic!("verify rejected at m={m}: {e:?}"));

            assert_eq!(claim_p, claim_v, "claim mismatch at m={m}");
        }
    }

    /// End-to-end transcript gate for the ranked identity-C producer. The
    /// alternate lincheck-stripe path must emit the exact legacy zerocheck
    /// proof and claim, not merely an algebraically equivalent C opening.
    #[test]
    fn lincheck_stripe_c_proof_is_byte_identical() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .stack_size(16 << 20)
            .build()
            .unwrap();
        pool.install(lincheck_stripe_c_proof_is_byte_identical_inner);
    }

    fn lincheck_stripe_c_proof_is_byte_identical_inner() {
        const M: usize = 17;
        const K_LOG: usize = 14;
        let padding = PaddingSpec {
            k_log: K_LOG,
            useful_bits_per_block: 1 << K_LOG,
        };
        let mut rng = Rng::new(0x1D_C5_7A1E);
        let a = rng.bits(1 << M);
        let b = rng.bits(1 << M);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let inv_table = InvNttTableByteSingleGf8::cached_standard_k6();

        let old_ab = univariate_skip_optimized::precompute_round1_ab_inner_packed_padded(
            &a_p, &b_p, M, K_SKIP, inv_table, &padding,
        );
        let mut old_ch = FsChallenger::new(b"flock-c-stripe-proof-v0");
        let (old_proof, old_claim, old_capture) = prove_packed_padded_inner(
            &a_p,
            &b_p,
            &c_p,
            M,
            &padding,
            true,
            Some(old_ab),
            None,
            &mut old_ch,
        );

        let c_words: Vec<F128> = c_p
            .chunks_exact(16)
            .map(|bytes| F128 {
                lo: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                hi: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
            })
            .collect();
        let c_lincheck = crate::lincheck::pack_z_lincheck_from_packed(&c_words, M, K_LOG);
        let new_ab = univariate_skip_optimized::precompute_round1_ab_inner_packed_padded(
            &a_p, &b_p, M, K_SKIP, inv_table, &padding,
        );
        let mut new_ch = FsChallenger::new(b"flock-c-stripe-proof-v0");
        let (new_proof, new_claim, new_capture) = prove_packed_padded_inner(
            &a_p,
            &b_p,
            &c_p,
            M,
            &padding,
            true,
            Some(new_ab),
            Some(&c_lincheck),
            &mut new_ch,
        );

        assert_eq!(new_proof, old_proof);
        assert_eq!(new_claim, old_claim);
        let old_capture = old_capture.expect("legacy C capture");
        let new_capture = new_capture.expect("stripe C capture");
        assert_eq!(new_capture.s_hat_v_c, old_capture.s_hat_v_c);
        assert_eq!(new_capture.quad, old_capture.quad);

        let mut verifier = FsChallenger::new(b"flock-c-stripe-proof-v0");
        assert_eq!(verify(M, &new_proof, &mut verifier), Ok(new_claim));
    }

    /// Same transcript gate as above, at a shape large enough for the GPU
    /// round-one C-fold arm to actually submit (`n_outer = 2^13` ⇒ 1024
    /// stripes ⇒ 2 tile claims, so the GPU takes a prefix and the CPU the
    /// suffix). The proof, claim and every RingSwitch capture must be
    /// byte-identical to the pure-CPU 32-bank drain.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn lincheck_stripe_c_proof_is_byte_identical_with_gpu_prefix() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .stack_size(16 << 20)
            .build()
            .unwrap();
        pool.install(|| {
            const M: usize = 27;
            const K_LOG: usize = 14;
            let padding = PaddingSpec {
                k_log: K_LOG,
                useful_bits_per_block: (1 << K_LOG) - 975,
            };
            let mut rng = Rng::new(0x7A_C0_DE_51);
            let a = rng.bits(1 << M);
            let b = rng.bits(1 << M);
            let mut c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            // Honest padding: rows past `useful_bits` are zero in every block.
            for (i, bit) in c.iter_mut().enumerate() {
                if i % (1 << K_LOG) >= padding.useful_bits_per_block {
                    *bit = false;
                }
            }
            let a: Vec<bool> = a
                .iter()
                .enumerate()
                .map(|(i, v)| *v && i % (1 << K_LOG) < padding.useful_bits_per_block)
                .collect();
            let b: Vec<bool> = b
                .iter()
                .enumerate()
                .map(|(i, v)| *v && i % (1 << K_LOG) < padding.useful_bits_per_block)
                .collect();
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let inv_table = InvNttTableByteSingleGf8::cached_standard_k6();

            let old_ab = univariate_skip_optimized::precompute_round1_ab_inner_packed_padded(
                &a_p, &b_p, M, K_SKIP, inv_table, &padding,
            );
            let mut old_ch = FsChallenger::new(b"flock-c-stripe-gpu-v0");
            let (old_proof, old_claim, old_capture) = prove_packed_padded_inner(
                &a_p,
                &b_p,
                &c_p,
                M,
                &padding,
                true,
                Some(old_ab),
                None,
                &mut old_ch,
            );

            let c_words: Vec<F128> = c_p
                .chunks_exact(16)
                .map(|bytes| F128 {
                    lo: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                    hi: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
                })
                .collect();
            let c_lincheck = crate::lincheck::pack_z_lincheck_from_packed(&c_words, M, K_LOG);
            let new_ab = univariate_skip_optimized::precompute_round1_ab_inner_packed_padded(
                &a_p, &b_p, M, K_SKIP, inv_table, &padding,
            );
            let submits_before = crate::gpu_commit::zerocheck_gpu_submits();
            let mut new_ch = FsChallenger::new(b"flock-c-stripe-gpu-v0");
            let (new_proof, new_claim, new_capture) = prove_packed_padded_inner(
                &a_p,
                &b_p,
                &c_p,
                M,
                &padding,
                true,
                Some(new_ab),
                Some(&c_lincheck),
                &mut new_ch,
            );
            assert!(
                crate::gpu_commit::zerocheck_gpu_submits() > submits_before,
                "GPU round-one C prefix never submitted — the oracle proved nothing"
            );

            assert_eq!(new_proof, old_proof);
            assert_eq!(new_claim, old_claim);
            let old_capture = old_capture.expect("legacy C capture");
            let new_capture = new_capture.expect("stripe C capture");
            assert_eq!(new_capture.s_hat_v_c, old_capture.s_hat_v_c);
            assert_eq!(new_capture.quad, old_capture.quad);

            let mut verifier = FsChallenger::new(b"flock-c-stripe-gpu-v0");
            assert_eq!(verify(M, &new_proof, &mut verifier), Ok(new_claim));
        });
    }

    /// **Verify rejects byte-mutated proofs.** Walk each component of the
    /// proof and flip one F128 entry; the verifier must return an `Err`
    /// (rather than panicking or silently accepting).
    #[test]
    fn verify_rejects_mutations() {
        let m = 14;
        let mut rng = Rng::new(5050);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let _seed: u64 = 0xDEAD_BEEF;
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // Each closure returns a mutated copy; verify must reject all of them.
        let mutations: Vec<(&str, Box<dyn Fn(&ZerocheckProof) -> ZerocheckProof>)> = vec![
            (
                "round1_ab[0] bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.round1_ab[0].lo ^= 1;
                    q
                }),
            ),
            (
                "round1_c[5] bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.round1_c[5].lo ^= 1;
                    q
                }),
            ),
            (
                "multilinear_rounds[0].0 bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.multilinear_rounds[0].0.lo ^= 1;
                    q
                }),
            ),
            (
                "multilinear_rounds[2].1 bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    let last = q.multilinear_rounds.len() / 2;
                    q.multilinear_rounds[last].1.hi ^= 1;
                    q
                }),
            ),
            (
                "final_a_eval bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.final_a_eval.lo ^= 1;
                    q
                }),
            ),
            (
                "final_c_eval bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.final_c_eval.hi ^= 1;
                    q
                }),
            ),
        ];

        for (label, mutate) in mutations {
            let bad = mutate(&proof);
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let result = verify(m, &bad, &mut ch);
            assert!(
                result.is_err(),
                "verify accepted mutated proof ({label}) — should have rejected"
            );
        }
    }

    /// Shape rejections: too-short round1, wrong number of multilinear rounds.
    #[test]
    fn verify_rejects_shape_errors() {
        let m = 14;
        let mut rng = Rng::new(606);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // Truncate round1_ab.
        let mut bad = proof.clone();
        bad.round1_ab.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, &bad, &mut ch),
            Err(VerifyError::BadRound1Length { .. })
        ));

        // Truncate multilinear rounds.
        let mut bad = proof.clone();
        bad.multilinear_rounds.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, &bad, &mut ch),
            Err(VerifyError::BadMultilinearRoundsLength { .. })
        ));

        // log_n too small.
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(K_SKIP + 6, &proof, &mut ch),
            Err(VerifyError::LogNTooSmall { .. })
        ));
    }

    /// AUDIT: a FALSE statement (c ≠ a·b at some hypercube point) must be
    /// rejected, even though the prover follows the honest algorithm on its
    /// (dishonest) witness.
    #[test]
    fn audit_false_statement_rejected() {
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(7777 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            // Correct c, then corrupt ONE bit so a·b ⊕ c ≠ 0 somewhere.
            let mut c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            c[3] = !c[3];

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &proof, &mut ch_verify);
            assert!(
                res.is_err(),
                "verify ACCEPTED a false statement at m={m}: {res:?}"
            );
        }
    }

    /// AUDIT: flipping any round's `msg_inf` (the degree-2 / ∞ coefficient)
    /// must be rejected. `msg_inf` is observed into the transcript, so the
    /// tamper both reshuffles subsequent ρ challenges and breaks the
    /// running-claim chain — either way the final check fails.
    #[test]
    fn audit_round_msg_inf_tamper_rejected() {
        let m = 14;
        let mut rng = Rng::new(424242);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // For each round, flip msg_inf to a different value. Because msg_inf
        // is observed into the transcript, this reshuffles subsequent rho's;
        // a sound verifier should reject (overwhelming probability).
        for idx in 0..proof.multilinear_rounds.len() {
            let mut bad = proof.clone();
            bad.multilinear_rounds[idx].1 += F128::ONE;
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &bad, &mut ch);
            assert!(res.is_err(), "msg_inf tamper at round {idx} ACCEPTED");
        }
    }

    /// AUDIT: the LAST round's `msg_inf` must be constrained — a common
    /// off-by-one is to leave the final round's leading coefficient unchecked.
    /// Kept separate from the all-rounds loop above so a regression here points
    /// straight at the final-round binding.
    #[test]
    fn audit_last_round_inf_constrained() {
        let m = 13;
        let mut rng = Rng::new(98765);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        let last = proof.multilinear_rounds.len() - 1;
        let mut bad = proof.clone();
        bad.multilinear_rounds[last].1 += F128::ONE;
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &bad, &mut ch).is_err(),
            "last-round msg_inf unconstrained"
        );
    }

    /// AUDIT (Fiat–Shamir binding of the final â, b̂ claims). Regression test
    /// for the gap where `final_a_eval`/`final_b_eval` were not observed into
    /// the transcript.
    ///
    /// Downstream, lincheck reduces these two claims via a *single* random-
    /// linear-combination check (`target = α·v_a + v_b`). That batching is only
    /// sound if α is sampled *after* the claims are bound to the transcript —
    /// otherwise a prover that already knows α can pick (v_a, v_b) to satisfy
    /// the one batched equation while violating the individual ties.
    ///
    /// A *product-preserving* tamper `(â, b̂) → (â·t, b̂·t⁻¹)` leaves the
    /// zerocheck's own final check `c_running == â·b̂` satisfied, so `verify`
    /// still returns `Ok` — the zerocheck alone is blind to it. The defense is
    /// that both claims are now observed last in the transcript, so the next
    /// challenge (the slot lincheck draws α from) must diverge from the honest
    /// run. This assertion FAILS before the observe was added (identical
    /// post-state) and passes now.
    #[test]
    fn audit_final_ab_claims_bound_to_transcript() {
        let m = 14;
        let mut rng = Rng::new(0xF1A7_5A11);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        // Honest verify, then capture the next challenge the transcript feeds
        // downstream — this is exactly the slot lincheck samples α from.
        let mut ch_honest = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &proof, &mut ch_honest).is_ok(),
            "honest verify rejected"
        );
        let alpha_honest = ch_honest.sample_f128();

        // Product-preserving tamper: â' = â·t, b̂' = b̂·t⁻¹ ⇒ â'·b̂' = â·b̂, so the
        // zerocheck's `c_running == â·b̂` check still holds for the tampered pair.
        let t = F128 {
            lo: 0x0123_4567_89ab_cdef,
            hi: 0xfedc_ba98_7654_3210,
        };
        assert!(t != F128::ZERO && t != F128::ONE, "t must be nontrivial");
        let mut bad = proof.clone();
        bad.final_a_eval *= t;
        bad.final_b_eval *= t.inv();
        assert_ne!(bad.final_a_eval, proof.final_a_eval, "tamper must change â");
        assert_ne!(bad.final_b_eval, proof.final_b_eval, "tamper must change b̂");
        assert_eq!(
            bad.final_a_eval * bad.final_b_eval,
            proof.final_a_eval * proof.final_b_eval,
            "tamper must preserve the product",
        );

        // The zerocheck's own checks are blind to a product-preserving tamper:
        // verify still ACCEPTS. This is precisely the gap the FS binding closes —
        // the tamper is caught only because the claims now move the transcript.
        let mut ch_tampered = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &bad, &mut ch_tampered).is_ok(),
            "product-preserving tamper rejected by zerocheck's own checks (unexpected)",
        );
        let alpha_tampered = ch_tampered.sample_f128();

        // The fix: observing â, b̂ makes the downstream challenge depend on them,
        // so lincheck's α (and everything after) diverges and rejects the
        // tampered pair. Before the fix these challenges were equal.
        assert_ne!(
            alpha_honest, alpha_tampered,
            "final â/b̂ claims are NOT bound into the transcript: a product-preserving \
             tamper leaves the downstream challenge unchanged, breaking lincheck's \
             α-batched reduction of (v_a, v_b)",
        );
    }

    /// AUDIT: many random false witnesses must all be rejected. Stronger than a
    /// single corruption — exercises the full prove→verify path on statements
    /// that are false at varying numbers of hypercube points.
    #[test]
    fn audit_many_false_statements_rejected() {
        let m = 13;
        for seed in 0..20u64 {
            let mut rng = Rng::new(0xBADC0DE ^ seed);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let mut c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            // Flip a random number of bits (1..=4).
            let nflip = 1 + (rng.next_u64() as usize % 4);
            for _ in 0..nflip {
                let idx = rng.next_u64() as usize % c.len();
                c[idx] = !c[idx];
            }
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);
            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &proof, &mut ch_verify);
            assert!(
                res.is_err(),
                "false statement (seed={seed}) ACCEPTED: {res:?}"
            );
        }
    }

    /// AUDIT: tamper msg_1 in each round; must reject.
    #[test]
    fn audit_round_msg_1_tamper_rejected() {
        let m = 14;
        let mut rng = Rng::new(31415);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);
        for idx in 0..proof.multilinear_rounds.len() {
            let mut bad = proof.clone();
            bad.multilinear_rounds[idx].0 += F128::ONE;
            let mut ch = FsChallenger::new(b"flock-test-v0");
            assert!(
                verify(m, &bad, &mut ch).is_err(),
                "msg_1 tamper round {idx} ACCEPTED"
            );
        }
    }

    /// Determinism: same witness + same challenger seed → same proof.
    #[test]
    fn prove_deterministic() {
        let m = 14;
        let mut rng = Rng::new(99);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch1 = FsChallenger::new(b"flock-test-v0");
        let mut ch2 = FsChallenger::new(b"flock-test-v0");
        let (proof1, claim1) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch1);
        let (proof2, claim2) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch2);

        assert_eq!(proof1.round1_ab, proof2.round1_ab);
        assert_eq!(proof1.round1_c, proof2.round1_c);
        assert_eq!(proof1.multilinear_rounds, proof2.multilinear_rounds);
        assert_eq!(proof1.final_a_eval, proof2.final_a_eval);
        assert_eq!(proof1.final_b_eval, proof2.final_b_eval);
        assert_eq!(proof1.final_c_eval, proof2.final_c_eval);
        assert_eq!(claim1.z, claim2.z);
        assert_eq!(claim1.mlv_challenges, claim2.mlv_challenges);
    }

    /// T5 — the only test that proves *transcript* identity rather than
    /// message identity: prove twice from the same witness with the
    /// two-challenge lookahead engaged and disabled, and compare the complete
    /// `ZerocheckProof` plus the claim's challenge vector.
    #[test]
    fn prove_transcript_identical_with_and_without_lookahead() {
        use std::sync::atomic::Ordering;
        for m in [13usize, 14, 16] {
            let mut rng = Rng::new(0x5EED ^ m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

            ZC_LOOKAHEAD_FORCED_OFF.store(false, Ordering::Relaxed);
            let mut ch_on = FsChallenger::new(b"flock-test-v0");
            let (proof_on, claim_on) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_on);

            ZC_LOOKAHEAD_FORCED_OFF.store(true, Ordering::Relaxed);
            let mut ch_off = FsChallenger::new(b"flock-test-v0");
            let (proof_off, claim_off) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_off);
            ZC_LOOKAHEAD_FORCED_OFF.store(false, Ordering::Relaxed);

            assert_eq!(proof_on.round1_ab, proof_off.round1_ab, "round1_ab m={m}");
            assert_eq!(proof_on.round1_c, proof_off.round1_c, "round1_c m={m}");
            assert_eq!(
                proof_on.multilinear_rounds, proof_off.multilinear_rounds,
                "multilinear_rounds m={m}"
            );
            assert_eq!(
                proof_on.final_a_eval, proof_off.final_a_eval,
                "a_eval m={m}"
            );
            assert_eq!(
                proof_on.final_b_eval, proof_off.final_b_eval,
                "b_eval m={m}"
            );
            assert_eq!(
                proof_on.final_c_eval, proof_off.final_c_eval,
                "c_eval m={m}"
            );
            assert_eq!(claim_on.z, claim_off.z, "z m={m}");
            assert_eq!(
                claim_on.mlv_challenges, claim_off.mlv_challenges,
                "mlv_challenges m={m}"
            );
        }
    }

    /// C-T5 — transcript identity for the second-level cascade (rounds 5+6):
    /// prove twice from the same witness with the cascade engaged and
    /// disabled (the round-3 lookahead stays on in both — the cascade's
    /// fallback IS the incumbent lookahead route), and compare the complete
    /// `ZerocheckProof` plus the claim's challenge vector.
    #[test]
    fn prove_transcript_identical_with_and_without_cascade2() {
        use std::sync::atomic::Ordering;
        for m in [13usize, 14, 16] {
            let mut rng = Rng::new(0xCA5C ^ m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

            ZC_CASCADE2_FORCED_OFF.store(false, Ordering::Relaxed);
            let mut ch_on = FsChallenger::new(b"flock-test-v0");
            let (proof_on, claim_on) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_on);

            ZC_CASCADE2_FORCED_OFF.store(true, Ordering::Relaxed);
            let mut ch_off = FsChallenger::new(b"flock-test-v0");
            let (proof_off, claim_off) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_off);
            ZC_CASCADE2_FORCED_OFF.store(false, Ordering::Relaxed);

            assert_eq!(proof_on.round1_ab, proof_off.round1_ab, "round1_ab m={m}");
            assert_eq!(proof_on.round1_c, proof_off.round1_c, "round1_c m={m}");
            assert_eq!(
                proof_on.multilinear_rounds, proof_off.multilinear_rounds,
                "multilinear_rounds m={m}"
            );
            assert_eq!(
                proof_on.final_a_eval, proof_off.final_a_eval,
                "a_eval m={m}"
            );
            assert_eq!(
                proof_on.final_b_eval, proof_off.final_b_eval,
                "b_eval m={m}"
            );
            assert_eq!(
                proof_on.final_c_eval, proof_off.final_c_eval,
                "c_eval m={m}"
            );
            assert_eq!(claim_on.z, claim_off.z, "z m={m}");
            assert_eq!(
                claim_on.mlv_challenges, claim_off.mlv_challenges,
                "mlv_challenges m={m}"
            );
        }
    }

    /// C3-T5 — transcript identity for the third-level cascade (rounds 7+8):
    /// prove twice from the same witness with cascade3 engaged and disabled
    /// (the lookahead and cascade2 stay on in both — cascade3's fallback IS
    /// the cascade2 route), and compare the complete `ZerocheckProof` plus
    /// the claim's challenge vector. m=13 (n_mlv=7) sits below cascade3's
    /// n_mlv ≥ 8 floor, so it also exercises the disengaged boundary.
    #[test]
    fn prove_transcript_identical_with_and_without_cascade3() {
        use std::sync::atomic::Ordering;
        for m in [13usize, 14, 16] {
            let mut rng = Rng::new(0xCA53 ^ m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

            ZC_CASCADE3_FORCED_OFF.store(false, Ordering::Relaxed);
            let mut ch_on = FsChallenger::new(b"flock-test-v0");
            let (proof_on, claim_on) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_on);

            ZC_CASCADE3_FORCED_OFF.store(true, Ordering::Relaxed);
            let mut ch_off = FsChallenger::new(b"flock-test-v0");
            let (proof_off, claim_off) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_off);
            ZC_CASCADE3_FORCED_OFF.store(false, Ordering::Relaxed);

            assert_eq!(proof_on.round1_ab, proof_off.round1_ab, "round1_ab m={m}");
            assert_eq!(proof_on.round1_c, proof_off.round1_c, "round1_c m={m}");
            assert_eq!(
                proof_on.multilinear_rounds, proof_off.multilinear_rounds,
                "multilinear_rounds m={m}"
            );
            assert_eq!(
                proof_on.final_a_eval, proof_off.final_a_eval,
                "a_eval m={m}"
            );
            assert_eq!(
                proof_on.final_b_eval, proof_off.final_b_eval,
                "b_eval m={m}"
            );
            assert_eq!(
                proof_on.final_c_eval, proof_off.final_c_eval,
                "c_eval m={m}"
            );
            assert_eq!(claim_on.z, claim_off.z, "z m={m}");
            assert_eq!(
                claim_on.mlv_challenges, claim_off.mlv_challenges,
                "mlv_challenges m={m}"
            );
        }
    }

    /// C4-T1 — transcript identity for the fourth-level cascade (rounds
    /// 9+10): prove twice from the same witness with cascade4 engaged and
    /// disabled (the lookahead, cascade2, and cascade3 stay on in both —
    /// cascade4's fallback IS the cascade3 route), and compare the complete
    /// `ZerocheckProof` plus the claim's challenge vector. m=15 (n_mlv=9)
    /// sits below cascade4's n_mlv ≥ 10 floor, so it also exercises the
    /// disengaged boundary.
    #[test]
    fn prove_transcript_identical_with_and_without_cascade4() {
        use std::sync::atomic::Ordering;
        for m in [15usize, 16, 18] {
            let mut rng = Rng::new(0xCA54 ^ m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

            ZC_CASCADE4_FORCED_OFF.store(false, Ordering::Relaxed);
            let mut ch_on = FsChallenger::new(b"flock-test-v0");
            let (proof_on, claim_on) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_on);

            ZC_CASCADE4_FORCED_OFF.store(true, Ordering::Relaxed);
            let mut ch_off = FsChallenger::new(b"flock-test-v0");
            let (proof_off, claim_off) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_off);
            ZC_CASCADE4_FORCED_OFF.store(false, Ordering::Relaxed);

            assert_eq!(proof_on.round1_ab, proof_off.round1_ab, "round1_ab m={m}");
            assert_eq!(proof_on.round1_c, proof_off.round1_c, "round1_c m={m}");
            assert_eq!(
                proof_on.multilinear_rounds, proof_off.multilinear_rounds,
                "multilinear_rounds m={m}"
            );
            assert_eq!(
                proof_on.final_a_eval, proof_off.final_a_eval,
                "a_eval m={m}"
            );
            assert_eq!(
                proof_on.final_b_eval, proof_off.final_b_eval,
                "b_eval m={m}"
            );
            assert_eq!(
                proof_on.final_c_eval, proof_off.final_c_eval,
                "c_eval m={m}"
            );
            assert_eq!(claim_on.z, claim_off.z, "z m={m}");
            assert_eq!(
                claim_on.mlv_challenges, claim_off.mlv_challenges,
                "mlv_challenges m={m}"
            );
        }
    }

    /// C5-T1 — transcript identity for the fifth-level cascade (rounds
    /// 11+12). m=17 is immediately below the n_mlv >= 12 engagement floor;
    /// m=18 and m=20 exercise the ranked route at smaller test shapes.
    #[test]
    fn prove_transcript_identical_with_and_without_cascade5() {
        use std::sync::atomic::Ordering;
        for m in [17usize, 18, 20] {
            let mut rng = Rng::new(0xCA55 ^ m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

            ZC_CASCADE5_FORCED_OFF.store(false, Ordering::Relaxed);
            let mut ch_on = FsChallenger::new(b"flock-test-v0");
            let (proof_on, claim_on) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_on);

            ZC_CASCADE5_FORCED_OFF.store(true, Ordering::Relaxed);
            let mut ch_off = FsChallenger::new(b"flock-test-v0");
            let (proof_off, claim_off) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_off);
            ZC_CASCADE5_FORCED_OFF.store(false, Ordering::Relaxed);

            assert_eq!(proof_on.round1_ab, proof_off.round1_ab, "round1_ab m={m}");
            assert_eq!(proof_on.round1_c, proof_off.round1_c, "round1_c m={m}");
            assert_eq!(
                proof_on.multilinear_rounds, proof_off.multilinear_rounds,
                "multilinear_rounds m={m}"
            );
            assert_eq!(
                proof_on.final_a_eval, proof_off.final_a_eval,
                "a_eval m={m}"
            );
            assert_eq!(
                proof_on.final_b_eval, proof_off.final_b_eval,
                "b_eval m={m}"
            );
            assert_eq!(
                proof_on.final_c_eval, proof_off.final_c_eval,
                "c_eval m={m}"
            );
            assert_eq!(claim_on.z, claim_off.z, "z m={m}");
            assert_eq!(
                claim_on.mlv_challenges, claim_off.mlv_challenges,
                "mlv_challenges m={m}"
            );
        }
    }
}

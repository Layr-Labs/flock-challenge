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
use crate::field::{F8, F128};
use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use serde::{Deserialize, Serialize};

pub mod multilinear;
pub mod univariate_skip;
pub mod univariate_skip_deg4;
pub mod univariate_skip_deg4_optimized;
pub mod univariate_skip_optimized;

use multilinear::{
    UniSkipFoldTable, eval_round3_lookahead, fold_and_compute_round_pair_into, fold_in_place_pair,
    fold2_from_packed_and_round_pair_lookahead_into_with_eq,
    fold2_plain_and_round_pair_lookahead_into, fold2_plain_and_round4_into,
    interpolate_at_z_combined, interpolate_at_z_on_lambda, marginalize_eq_low2,
    packed_round2_split_eq, round_pair_naive, uni_skip_fold_and_round_pair_optimized_packed_padded,
    uni_skip_fold_and_round_pair_optimized_packed_padded_lookahead,
    uni_skip_round_pair_lookahead_nomat_packed_padded_with_eq,
};
use univariate_skip_optimized::{
    c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed_padded,
    small_challenges_ghash,
};

/// Number of variables folded in round 1 via the additive-NTT univariate skip.
/// |Λ| = 2^K_SKIP = 64 elements; the round-1 prover message is two length-64
/// vectors of F128.
pub const K_SKIP: usize = 6;

/// Test-only forced-off latch for the two-challenge lookahead. Production
/// reads `FLOCK_NO_ZC_LOOKAHEAD`; the transcript-identity test flips this
/// instead so it never has to mutate the process environment. Flipping it
/// cannot make a concurrently running test wrong — both routes emit the same
/// transcript, which is exactly what that test asserts.
#[cfg(test)]
pub(crate) static ZC_LOOKAHEAD_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `FLOCK_NO_ZC_LOOKAHEAD=1` restores the incumbent rounds-2..4 route
/// (round-2 sweep, then two fused tail passes). Same-binary A/B control and
/// emergency fallback; the ranked worker's cleared environment never sets it,
/// so the shipped behavior is lookahead-ON at the ranked shape.
#[inline]
fn lookahead_off() -> bool {
    #[cfg(test)]
    if ZC_LOOKAHEAD_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    std::env::var_os("FLOCK_NO_ZC_LOOKAHEAD").is_some()
}

/// Test-only: number of lookahead/cascade levels the last prove ran with, so
/// the transcript-identity test can assert each arm really engaged.
#[cfg(test)]
pub(crate) static ZC_LEVELS_LAST: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Test-only forced-off latches for the cascade levels (rounds 5+6, 7+8).
#[cfg(test)]
pub(crate) static ZC_CASCADE2_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
pub(crate) static ZC_CASCADE3_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `FLOCK_NO_ZC_CASCADE2=1` stops the lookahead cascade after rounds 3+4
/// (tail iterations i ≥ 2 run the incumbent fused passes). Independent
/// same-binary control; the ranked worker's cleared environment never sets
/// it. Cascade level 2 requires the lookahead itself to be on.
#[inline]
fn cascade2_off() -> bool {
    #[cfg(test)]
    if ZC_CASCADE2_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    std::env::var_os("FLOCK_NO_ZC_CASCADE2").is_some()
}

/// Test-only forced-off latch for the no-materialize sweep.
#[cfg(test)]
pub(crate) static ZC_NOMAT_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// Test-only: whether the last prove ran the no-materialize route.
#[cfg(test)]
pub(crate) static ZC_NOMAT_LAST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `FLOCK_NO_ZC_SWEEP_NOMAT=1` makes the round-two sweep materialize the two
/// folded tables again (the `5d4d2a9` lookahead behavior); the default
/// no-materialize route computes only the round-two message and the deferred
/// round-three coefficients in the sweep and re-derives the folded rows from
/// the packed witness inside the composed rounds-3+4 pass. Independent
/// same-binary control; the ranked worker's cleared environment never sets it.
#[inline]
fn nomat_off() -> bool {
    #[cfg(test)]
    if ZC_NOMAT_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    std::env::var_os("FLOCK_NO_ZC_SWEEP_NOMAT").is_some()
}

/// `FLOCK_NO_ZC_CASCADE3=1` stops the cascade after rounds 5+6.
#[inline]
fn cascade3_off() -> bool {
    #[cfg(test)]
    if ZC_CASCADE3_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    std::env::var_os("FLOCK_NO_ZC_CASCADE3").is_some()
}

/// Test-only latches for cascade levels 3 and 4 (rounds 9+10, 11+12). Both
/// levels now ship on, so each latch forces the matching level *off*. Either
/// way both routes emit the same transcript — that is what the
/// `prove_transcript_identical_*` tests assert.
#[cfg(test)]
pub(crate) static ZC_CASCADE4_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
pub(crate) static ZC_CASCADE5_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `FLOCK_NO_ZC_CASCADE4=1` stops the cascade after rounds 7+8 (the incumbent
/// frontier behavior); the default runs one more composed pass over rounds
/// 9+10. Independent same-binary control; the ranked worker's cleared
/// environment never sets it.
#[inline]
fn cascade4_off() -> bool {
    #[cfg(test)]
    if ZC_CASCADE4_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    std::env::var_os("FLOCK_NO_ZC_CASCADE4").is_some()
}

/// Cascade level 4 (rounds 11+12) ships on: `FLOCK_NO_ZC_CASCADE5=1` restores
/// the prior opt-in-off incumbent. Zen 5 priced this as a 0.4 ms regression
/// vs the calling-thread serial tail (0.54+0.25 ms vs a 0.57–0.99 ms rayon
/// composed pass). Official SPR priced the earlier tail-fanout patch at
/// −0.66% because rayon regions are cheap there — the same inversion that
/// makes serial tail a local win can make this composed pass a runner win.
/// One mechanism, kill-switch default ON; the ranked worker's cleared env
/// never sets the flag.
#[inline]
fn cascade5_off() -> bool {
    #[cfg(test)]
    if ZC_CASCADE5_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    std::env::var_os("FLOCK_NO_ZC_CASCADE5").is_some()
}

/// `FLOCK_NO_ZC_DEEP_CASCADE=1` restores the promoted five-level cap. The
/// default keeps composing the remaining value-identical round pairs through
/// the ranked 26-variable tail.
#[inline]
fn deep_cascade_off() -> bool {
    std::env::var_os("FLOCK_NO_ZC_DEEP_CASCADE").is_some()
}

fn build_urm_inv_table(k_skip: usize) -> InvNttTableByteSingleGf8 {
    let ntt_s = AdditiveNttGf8::new(k_skip, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(k_skip, F8(1u8 << k_skip));
    InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
}

/// Transcript-independent inverse-NTT data for the protocol-fixed skip.
/// The worker's mandatory untimed proof initializes it before measurements.
static URM_INV_TABLE_K_SKIP: std::sync::LazyLock<InvNttTableByteSingleGf8> =
    std::sync::LazyLock::new(|| build_urm_inv_table(K_SKIP));

#[inline]
pub fn shared_urm_inv_table() -> &'static InvNttTableByteSingleGf8 {
    &URM_INV_TABLE_K_SKIP
}

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

pub struct CapturedSHatVC {
    pub s_hat_v_c: Vec<F128>,
    pub quad: Vec<F128>,
    /// Sixteen-bank direct-fold4 C tensor (`bank = e_small + 4·q`, 128 entries
    /// each), present only when the shared `ranked_direct_fold4_enabled()`
    /// gate is on and the round-1 split admits the four-window producer. It
    /// collapses under `suffix[..4]` to `s_hat_v_c` exactly.
    pub fold4: Option<Vec<F128>>,
    /// Sixty-four-bank form for the direct-fold8 route; collapses under
    /// `suffix[..6]` to `s_hat_v_c` exactly like `fold4` does under
    /// `suffix[..4]`. Present only behind the shared DirectFold8 opt-in.
    pub fold8: Option<Vec<F128>>,
}

/// Capture-`s_hat_v_c` prover that consumes a challenge-independent AB inner
/// transform prepared while the witness commitment was being built.  The
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
/// buffer is the packed witness itself (C = z at the ranked shape); it lets
/// round one derive the legacy C message and every RingSwitch capture from a
/// single block-major outer fold instead of draining the witness into 32 field
/// banks. Proof and transcript stay byte-identical to the Fold4 path.
#[allow(clippy::too_many_arguments)]
pub fn prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab_and_identity_c<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    c_identity_z: &[F128],
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
        Some(c_identity_z),
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce s_hat_v_c"),
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_packed_padded_inner<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    capture_s_hat_v_c: bool,
    mut precomputed_ab: Option<univariate_skip_optimized::Round1AbInner>,
    c_identity_z: Option<&[F128]>,
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
    let t_round1 = std::time::Instant::now();
    let inv_table_owned;
    let inv_table: &InvNttTableByteSingleGf8 = if k_skip == K_SKIP {
        &URM_INV_TABLE_K_SKIP
    } else {
        inv_table_owned = build_urm_inv_table(k_skip);
        &inv_table_owned
    };
    let (round1_ab_opt, round1_c_opt, s_hat_v_c) = if let Some(ab_inner) = precomputed_ab.as_mut() {
        assert!(
            capture_s_hat_v_c,
            "precomputed AB path currently requires s_hat_v capture"
        );
        if c_identity_z.is_none() {
            // Fold4/quad fallback consumers require dense AB. In particular,
            // disabling identity-C must not expose the producer's omitted
            // one-row windows as though they had been initialized.
            ab_inner.restore_full_if_ranked_one_rows_elided(a_packed, b_packed, inv_table);
        }
        if let Some(c_identity_z) = c_identity_z {
            // Ranked identity-C: AB completes without touching `c_packed`, and
            // C's message plus all three capture tensors come from one
            // block-major outer fold of the witness. Both halves are
            // bit-identical to the fused Fold4 kernel's outputs; the caller's
            // gate pins the shape.
            assert_eq!(padding.k_log, 14, "identity-C reuse fixes k_log=14");
            assert!(
                crate::pcs::ranked_direct_fold4_enabled(),
                "identity-C reuse requires ranked DirectFold4"
            );
            let ranked_one_rows = ab_inner.ranked_one_rows_elided();
            // The two halves are independent (round one has no Fiat-Shamir
            // dependency inside it), so run them concurrently rather than
            // back to back: each alone reaches only ~35 GB/s, while the pair
            // interleaved recovers the fused kernel's stream-level
            // parallelism over the same total bytes.
            let t_r1 = std::time::Instant::now();
            let ab_closure = || {
                let t = std::time::Instant::now();
                let ab = crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_ab_packed_padded_with_precomputed(
                    ab_inner, a_packed, b_packed, m, k_skip, &r, inv_table, padding,
                );
                (ab, t.elapsed().as_secs_f64() * 1e3)
            };
            let c_closure = || {
                let t = std::time::Instant::now();
                let (c, s_hat_v_c, quad, fold4, fold8, one_ab) =
                    crate::zerocheck::univariate_skip_optimized::round1_c_fold4_from_block_major_z(
                        c_identity_z,
                        m,
                        padding.k_log,
                        k_skip,
                        padding.useful_bits_per_block,
                        &r,
                        inv_table,
                        ranked_one_rows,
                    );
                (
                    c,
                    s_hat_v_c,
                    quad,
                    fold4,
                    fold8,
                    one_ab,
                    t.elapsed().as_secs_f64() * 1e3,
                )
            };
            // Per-core SMT arm partition. When the half-pools exist (ranked
            // shape; built at process start in `crate::smt_split`), AB runs on
            // sibling 0 of every physical core and C on sibling 1, so each core
            // pairs one of each kernel instead of whatever the work-stealer
            // happened to co-schedule. Schedule only: identical closures over
            // identical inputs, so the proof bytes are unchanged. Falls back to
            // the incumbent `rayon::join` whenever the pools are absent.
            let ((mut ab, t_ab_ms), (c, s_hat_v_c, quad, fold4, fold8, one_ab, t_c_ms)) =
                match crate::smt_split::zc_r1_pools() {
                    Some((ab_pool, c_pool)) => {
                        rayon::join(|| ab_pool.install(ab_closure), || c_pool.install(c_closure))
                    }
                    None => rayon::join(ab_closure, c_closure),
                };
            if let Some(one_ab) = one_ab {
                debug_assert_eq!(ab.len(), one_ab.len());
                // The merge is a post-join serial tail on round one's
                // critical path: it adds the C-half's one-row AB
                // contribution into the AB-half's shift-reduced tensor
                // element-wise. Addition is order-independent, so spread
                // the pass over the ambient pool instead of one core —
                // bit-identical output, shorter tail.
                use rayon::prelude::*;
                ab.par_iter_mut()
                    .zip(one_ab.par_iter())
                    .for_each(|(dst, src)| {
                        *dst += *src;
                    });
            }
            if zc_timing {
                eprintln!(
                    "[zc-timing] round1 AB {t_ab_ms:.2} ms || identity-C fold {t_c_ms:.2} ms -> {:.2} ms",
                    t_r1.elapsed().as_secs_f64() * 1e3
                );
            }
            (
                ab,
                c,
                Some(CapturedSHatVC {
                    s_hat_v_c,
                    quad,
                    fold4: (!fold4.is_empty()).then_some(fold4),
                    fold8: (!fold8.is_empty()).then_some(fold8),
                }),
            )
        } else if crate::pcs::ranked_direct_fold4_enabled()
            && crate::zerocheck::univariate_skip_optimized::c_fold4_capture_available(m, k_skip)
        {
            let (ab, c, s_hat_v_c, quad, fold4) =
                crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4(
                    ab_inner, a_packed, b_packed, c_packed, m, k_skip, &r, inv_table, padding,
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
                crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_quad(
                    ab_inner, a_packed, b_packed, c_packed, m, k_skip, &r, inv_table, padding,
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
                a_packed, b_packed, c_packed, m, k_skip, &r, inv_table, padding,
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
    // The A-sized transform is dead after the round-1 message and can return
    // to the scratch pool before the much larger round-2 fold allocations.
    drop(precomputed_ab);
    let c_s = c_s_f128();
    let round1_ab: Vec<F128> = round1_ab_opt.iter().map(|x| c_s * *x).collect();
    let round1_c: Vec<F128> = round1_c_opt.iter().map(|x| c_s * *x).collect();
    crate::gaptime::mark("zc: round1 done (URM + C_s restore)");
    if zc_timing {
        eprintln!(
            "[zc-timing] round1 URM: {:.2} ms",
            t_round1.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- 4. Observe round-1 message, sample z (URM fold point) ----
    challenger.observe_f128_slice(&round1_ab);
    challenger.observe_f128_slice(&round1_c);
    let z = challenger.sample_f128();
    crate::gaptime::mark("zc: round1 observed + z sampled");

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
    let t_round2 = std::time::Instant::now();
    let fold_table = UniSkipFoldTable::new(k_skip, z);
    crate::gaptime::mark("zc: fold_table built");
    let mut mlv_arg = vec![F128::ONE; n_mlv];
    mlv_arg[1..].copy_from_slice(&r[k_skip + 1..]);
    // Diagnostics kill switch: FLOCK_NO_TAIL_FUSION=1 routes every tail round
    // through the unfused two-pass path (fold_in_place_pair, then a separate
    // round_pair_naive read of the folded arrays) for one-process A/B runs —
    // and, being the "no fusion at all" oracle, also disables the lookahead.
    // Transcript bytes are identical either way; the ranked worker's cleared
    // environment never sets it.
    let no_tail_fusion = std::env::var_os("FLOCK_NO_TAIL_FUSION").is_some();

    // Two-challenge symbolic lookahead: round three's message is a quadratic
    // in ρ₁, so its six coefficients ride along inside round two's sweep and
    // rounds 3+4 collapse into a single composed double-fold pass out of the
    // round-two tables — deleting the first (largest) tail pass. Value-
    // identical by construction — F128 is exact and the transcript order
    // below is untouched — so the kill switch exists purely for same-binary
    // A/B screening.
    //
    // `r[k_skip+1] = 0` (probability 2⁻¹²⁸ for a sampled slot; at the ranked
    // shape it is the protocol constant φ₈(0x53) ≠ 0) makes W1/W2
    // unrecoverable from the parity split; that case falls back to the
    // incumbent route, which stays in the tree as the oracle anyway.
    // `n_mlv ≥ 6` keeps the composed pass's input ≥ 16 and every eq split at
    // lo_size ≥ 2. `m ≥ 20` covers the ranked shape (m = 32) and the local
    // smoke geometry; tiny shapes stay on the incumbent outside tests.
    let use_lookahead = (m >= 20 || cfg!(test))
        && n_mlv >= 6
        && r[k_skip + 1] != F128::ZERO
        && !no_tail_fusion
        && !lookahead_off();

    // No-materialize sweep: with the lookahead on, the round-two sweep no
    // longer needs to write its two folded tables (2 GiB of stores at the
    // ranked shape) — the composed rounds-3+4 pass re-derives the folded rows
    // it consumes straight from the packed witness through the same byte-
    // table gathers (1 GiB of reads instead of 2 GiB of dense F128), and the
    // peak zerocheck footprint drops by 2 GiB. The pass always emits the
    // round-five lookahead too, so it needs `r[k_skip+3] ≠ 0` (β₀ at the
    // ranked shape). Kill switch: FLOCK_NO_ZC_SWEEP_NOMAT.
    let use_nomat = use_lookahead && r[k_skip + 3] != F128::ZERO && !nomat_off();
    #[cfg(test)]
    ZC_NOMAT_LAST.store(use_nomat, std::sync::atomic::Ordering::Relaxed);

    // Round two and the packed rounds-3+4 pass share the same 13 high eq
    // coordinates at the ranked split. Keep round two's small tensors until
    // that pass; its low tensor loses exactly the first two coordinates.
    let mut packed_eq = use_nomat
        .then(|| packed_round2_split_eq(&mlv_arg))
        // The composed packed leaf requires at least one low variable after
        // the two-coordinate marginalization. Tiny test shapes keep their
        // independently built split.
        .filter(|eq| eq.n_lo >= 3);

    let (mut a_mlv, mut b_mlv, msg_1, msg_inf, lookahead) = if use_nomat {
        let (m1, mi, la) = uni_skip_round_pair_lookahead_nomat_packed_padded_with_eq(
            a_packed,
            b_packed,
            m,
            k_skip,
            &fold_table,
            &mlv_arg,
            padding,
            packed_eq.as_ref(),
        );
        (Vec::new(), Vec::new(), m1, mi, Some(la))
    } else if use_lookahead {
        let (a, b, m1, mi, la) = uni_skip_fold_and_round_pair_optimized_packed_padded_lookahead(
            a_packed,
            b_packed,
            m,
            k_skip,
            &fold_table,
            &mlv_arg,
            padding,
        );
        (a, b, m1, mi, Some(la))
    } else {
        let (a, b, m1, mi) = uni_skip_fold_and_round_pair_optimized_packed_padded(
            a_packed,
            b_packed,
            m,
            k_skip,
            &fold_table,
            &mlv_arg,
            padding,
        );
        (a, b, m1, mi, None)
    };
    crate::gaptime::mark("zc: round2 fused fold done");

    if zc_timing {
        eprintln!(
            "[zc-timing] round2 fused fold: {:.2} ms",
            t_round2.elapsed().as_secs_f64() * 1e3
        );
    }
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
    // ρ_{i+2}. Use the fused parallel path while log_n ≥ 10; below that the
    // SplitEqGhash inner can't form lo_size ≥ 2, so we fall back to
    // fold_in_place_pair + round_pair_naive.
    //
    // Ping-pong scratch buffers for the fused path: each fused round folds
    // (a_mlv, b_mlv) of size N into size N/2. Rather than allocating — and,
    // worse, `munmap`-ing, which is single-threaded and caps the tail's
    // parallel speedup — a fresh 64 MB buffer per round, we alternate between
    // two persistent buffers. Scratch capacity = N/2 (the largest fused
    // output); only needed when the first round is actually fused.
    //
    // With the lookahead, the first output of the tail is the composed pass's
    // quarter-size table, so the scratch pair is taken at N/4 (the pool's
    // best-fit hands back the same 2^(m-7)-class buffer either way).
    //
    // No-materialize: `a_mlv`/`b_mlv` are empty; the composed pass writes its
    // N/4 outputs into freshly taken buffers and the scratch pair is sized for
    // the fold after it (N/8).
    let n_in = 1usize << n_mlv;
    let first_out = if use_nomat {
        n_in / 8
    } else if lookahead.is_some() {
        n_in / 4
    } else {
        n_in / 2
    };
    let (mut a_nxt, mut b_nxt) = if n_in >= 1024 || lookahead.is_some() {
        (
            crate::scratch::take_f128(first_out),
            crate::scratch::take_f128(first_out),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    crate::gaptime::mark("zc: tail ping-pong buffers taken");
    let mut tail_round_ms: Vec<(usize, f64)> = Vec::new();

    // Cascade the lookahead deeper (rounds 5+6, then 7+8): every composed
    // pass materializes each output group in registers before its store —
    // the same position round two was in before the round-3 promotion — so
    // the next round's message rides it as a deferred quadratic in the
    // not-yet-sampled challenge, and the following two rounds collapse into
    // one more plain composed double-fold. Level L deletes tail iterations
    // i = 2L and i = 2L+1 (their DRAM passes and one FS-serialized round
    // boundary). Same value-identity argument as the lookahead: pure
    // reassociation of exact F128 arithmetic, transcript order untouched.
    //
    // Level L's lookahead needs the parity split of eq for round 2L+2, i.e.
    // `r[k_skip + 2L + 1] ≠ 0` (at the ranked shape those slots are the
    // protocol constants β₀, β₂ ≠ 0; probability 2⁻¹²⁸ for a sampled slot);
    // otherwise the chain stops one level earlier and the incumbent tail
    // continues from there. `n_mlv ≥ 7` (level 2) / `≥ 8` (level 3) keep
    // every composed input ≥ 16 and every eq split at lo_size ≥ 2.
    // Kill switches: FLOCK_NO_ZC_CASCADE2 / FLOCK_NO_ZC_CASCADE3.
    let use_cascade2 =
        use_lookahead && n_mlv >= 7 && r[k_skip + 3] != F128::ZERO && !cascade2_off();
    let use_cascade3 = use_cascade2 && n_mlv >= 8 && r[k_skip + 5] != F128::ZERO && !cascade3_off();
    // Levels 3 and 4 (rounds 9+10 and 11+12) extend the same chain. Their
    // parity weights r[k_skip+7] / r[k_skip+9] are sampled slots (the seven
    // protocol constants end at k_skip+6), so they are non-zero except with
    // probability 2⁻¹²⁸; the chain simply stops a level earlier otherwise.
    // The n_mlv floors keep the last composed pass's input at ≥ 16 elements
    // (level L consumes 2^(n_mlv−2L)) with a level of margin, matching the
    // Apple track's shipped gating.
    //
    // Level 3 ships on (kill switch FLOCK_NO_ZC_CASCADE4): at the ranked shape
    // it turns tail rounds log_n 20 and 19 — 2.0 + 1.2 ms of rayon regions —
    // into one 1.7 ms composed pass and deletes an FS round boundary.
    // Level 4 ships on (kill switch FLOCK_NO_ZC_CASCADE5); see `cascade5_off`.
    let use_cascade4 =
        use_cascade3 && n_mlv >= 10 && r[k_skip + 7] != F128::ZERO && !cascade4_off();
    let use_cascade5 =
        use_cascade4 && n_mlv >= 12 && r[k_skip + 9] != F128::ZERO && !cascade5_off();
    let use_cascade6 =
        use_cascade5 && n_mlv >= 14 && r[k_skip + 11] != F128::ZERO && !deep_cascade_off();
    let use_cascade7 = use_cascade6 && n_mlv >= 16 && r[k_skip + 13] != F128::ZERO;
    let use_cascade8 = use_cascade7 && n_mlv >= 18 && r[k_skip + 15] != F128::ZERO;
    let use_cascade9 = use_cascade8 && n_mlv >= 20 && r[k_skip + 17] != F128::ZERO;
    let use_cascade10 = use_cascade9 && n_mlv >= 22 && r[k_skip + 19] != F128::ZERO;
    let use_cascade11 = use_cascade10 && n_mlv >= 24 && r[k_skip + 21] != F128::ZERO;
    let use_cascade12 = use_cascade11 && n_mlv >= 26 && r[k_skip + 23] != F128::ZERO;
    let n_levels = match (
        use_lookahead,
        use_cascade2,
        use_cascade3,
        use_cascade4,
        use_cascade5,
        use_cascade6,
        use_cascade7,
        use_cascade8,
        use_cascade9,
        use_cascade10,
        use_cascade11,
        use_cascade12,
    ) {
        (false, ..) => 0,
        (true, false, ..) => 1,
        (true, true, false, ..) => 2,
        (true, true, true, false, ..) => 3,
        (true, true, true, true, false, ..) => 4,
        (true, true, true, true, true, false, ..) => 5,
        (true, true, true, true, true, true, false, ..) => 6,
        (true, true, true, true, true, true, true, false, ..) => 7,
        (true, true, true, true, true, true, true, true, false, ..) => 8,
        (true, true, true, true, true, true, true, true, true, false, ..) => 9,
        (true, true, true, true, true, true, true, true, true, true, false, ..) => 10,
        (true, true, true, true, true, true, true, true, true, true, true, false) => 11,
        (true, true, true, true, true, true, true, true, true, true, true, true) => 12,
    };
    #[cfg(test)]
    ZC_LEVELS_LAST.store(n_levels, std::sync::atomic::Ordering::Relaxed);

    // `loop_start` is the first tail iteration this route has not already
    // produced. The loop body's `r_next[1..] = r[k_skip + i + 2..]` is already
    // indexed by `i`, so starting at 2·levels needs no other change.
    let mut la = lookahead;
    for level in 0..n_levels {
        let la_cur = la.take().expect("cascade level without a deferred message");
        // Round 2L+3: evaluate the deferred quadratic at ρ_{2L+1}. No pass.
        let (m_odd_1, m_odd_inf) = eval_round3_lookahead(&la_cur, mlv_rhos[2 * level]);
        multilinear_msgs.push((m_odd_1, m_odd_inf));
        challenger.observe_f128(m_odd_1);
        challenger.observe_f128(m_odd_inf);
        mlv_rhos.push(challenger.sample_f128());

        // Rounds 2L+3 and 2L+4 fold together in one pass (ρ_{2L+1} and
        // ρ_{2L+2} at once), replacing tail iterations i = 2L and i = 2L+1.
        // At the ranked shape level 0 turns 2 GiB + 1 GiB of reads and 1 GiB
        // + 512 MiB of writes into one 2 GiB read + 512 MiB write; level 1
        // turns 512 + 256 MiB reads / 256 + 128 MiB writes into 512 / 128.
        let t_round = std::time::Instant::now();
        let n_cur = if level == 0 { n_in } else { a_mlv.len() };
        let log_n_cur = n_cur.trailing_zeros() as usize;
        let quarter = n_cur / 4;
        let mut r_next = vec![F128::ONE; log_n_cur - 2];
        r_next[1..].copy_from_slice(&r[k_skip + 2 * level + 3..]);
        let (m_even_1, m_even_inf) = if level == 0 && use_nomat {
            // Rounds 3+4 straight from the packed witness (see above); the
            // outputs land in freshly taken N/4 buffers, and the old (empty)
            // pair is dropped. Also yields the round-five lookahead.
            let mut a4 = crate::scratch::take_f128(quarter);
            let mut b4 = crate::scratch::take_f128(quarter);
            let round2_eq = packed_eq.take();
            let eq_lo = round2_eq.as_ref().map(|eq| marginalize_eq_low2(&eq.lo));
            let eq_override = match (&eq_lo, &round2_eq) {
                (Some(lo), Some(eq)) => Some((&lo[..], &eq.hi[..])),
                _ => None,
            };
            let (m1, mi, la_next) = fold2_from_packed_and_round_pair_lookahead_into_with_eq(
                a_packed,
                b_packed,
                m,
                k_skip,
                &fold_table,
                padding,
                &mut a4,
                &mut b4,
                mlv_rhos[0],
                mlv_rhos[1],
                &r_next,
                eq_override,
            );
            a_mlv = a4;
            b_mlv = b4;
            if level + 1 < n_levels {
                la = Some(la_next);
            }
            (m1, mi)
        } else if level + 1 < n_levels {
            let (m1, mi, la_next) = fold2_plain_and_round_pair_lookahead_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..quarter],
                &mut b_nxt[..quarter],
                mlv_rhos[2 * level],
                mlv_rhos[2 * level + 1],
                &r_next,
            );
            la = Some(la_next);
            (m1, mi)
        } else {
            fold2_plain_and_round4_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..quarter],
                &mut b_nxt[..quarter],
                mlv_rhos[2 * level],
                mlv_rhos[2 * level + 1],
                &r_next,
            )
        };
        if !(level == 0 && use_nomat) {
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(quarter);
            b_mlv.truncate(quarter);
        }
        if level == 0 {
            crate::gaptime::mark("zc: rounds 3+4 composed fold done");
        }
        if zc_timing {
            tail_round_ms.push((log_n_cur, t_round.elapsed().as_secs_f64() * 1e3));
        }
        multilinear_msgs.push((m_even_1, m_even_inf));
        challenger.observe_f128(m_even_1);
        challenger.observe_f128(m_even_inf);
        mlv_rhos.push(challenger.sample_f128());
    }
    let loop_start = 2 * n_levels;

    for i in loop_start..(n_mlv - 1) {
        let t_round = std::time::Instant::now();
        let rho_prev = mlv_rhos[i];
        let log_n_before = a_mlv.len().trailing_zeros() as usize;

        // r_next for the next round's message: length log_n_before - 1.
        // r_next[0] = ONE (Convention A factor); r_next[1..] are the eq
        // weights for the remaining variables = r[k_skip + i + 2..m].
        let mut r_next = vec![F128::ONE; log_n_before - 1];
        r_next[1..].copy_from_slice(&r[k_skip + i + 2..]);

        let (m1, mi) = if log_n_before >= 10 && !no_tail_fusion {
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
        if zc_timing {
            tail_round_ms.push((log_n_before, t_round.elapsed().as_secs_f64() * 1e3));
        }
        multilinear_msgs.push((m1, mi));
        challenger.observe_f128(m1);
        challenger.observe_f128(mi);
        mlv_rhos.push(challenger.sample_f128());
    }
    // Last ML ρ is in. RowMajor `x_outer = mlv[k_log − k_skip ..]` is
    // complete here (not URM `r`). Kick today's one-shot leftover z-fold
    // so it can run on the full rayon pool while this thread finishes
    // serial FS (final bind + observe â/b̂). Compute only. No-op if the
    // ranked BlockMajor path did not prepare, or if `mlv` is short
    // (kicking during rounds 2–27 is REJECT). Kick after this function
    // returns is a no-op — the final bind has already run.
    crate::lincheck::kick_last_rho_z_fold(&mlv_rhos);
    crate::gaptime::mark("zc: tail rounds done");

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
    crate::gaptime::mark("zc: tail buffers recycled");

    if zc_timing && !tail_round_ms.is_empty() {
        let per_round: String = tail_round_ms
            .iter()
            .map(|(log_n, ms)| format!("n{log_n}:{ms:.2}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("[zc-timing] tail rounds (log_n:ms): {per_round}");
    }
    if zc_timing {
        eprintln!(
            "[zc-timing] rounds 3+ tail: {:.2} ms",
            t_tail.elapsed().as_secs_f64() * 1e3
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
    /// **Prove-level tail-fusion oracle**: the fused tail (fold r + message
    /// r+1 in one pass, q-form NEON kernels) produces a proof and claim
    /// byte-identical to the unfused two-pass path
    /// (`FLOCK_NO_TAIL_FUSION=1`: `fold_in_place_pair` then a separate
    /// `round_pair_naive` sweep every round). m ≥ 17 so several rounds run
    /// through the fused parallel kernel (log_n ≥ 10).
    #[test]
    fn prove_fused_tail_matches_unfused_two_pass() {
        for &m in &[17usize, 18] {
            let mut rng = Rng::new(4200 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

            let mut ch_fused = FsChallenger::new(b"flock-test-v0");
            let (proof_fused, claim_fused) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_fused);

            // SAFETY: test-only env toggle; the flag selects between
            // value-identical code paths, so even a concurrently running
            // prove observes no behavioral difference.
            unsafe { std::env::set_var("FLOCK_NO_TAIL_FUSION", "1") };
            let mut ch_unfused = FsChallenger::new(b"flock-test-v0");
            let (proof_unfused, claim_unfused) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_unfused);
            // SAFETY: as above.
            unsafe { std::env::remove_var("FLOCK_NO_TAIL_FUSION") };

            assert_eq!(proof_fused, proof_unfused, "proof bytes diverge at m={m}");
            assert_eq!(claim_fused, claim_unfused, "claim diverges at m={m}");
        }
    }

    /// **Lookahead / cascade transcript identity**: the two-challenge
    /// lookahead route (deferred round-3 quadratic + composed rounds-3/4
    /// double fold) and its cascades (rounds 5+6, 7+8, 9+10, 11+12) each emit
    /// a proof and claim byte-identical to the incumbent route — dense and
    /// BLAKE3-padded (k_log=14, useful=15409), m ∈ {13, 14, 17, 18, 19}
    /// (n_mlv = 7 enables level 2 only, 8 three, 10 four, 12 five) — and the
    /// full-cascade proof verifies. Each arm caps the cascade one level lower
    /// than the previous, so every level is its own successor’s oracle.
    /// Toggles the test latches, not the process env.
    #[test]
    fn prove_transcript_identical_with_and_without_lookahead() {
        use std::sync::atomic::Ordering;
        for &(m, padded) in &[
            (13usize, false),
            (14, false),
            (17, false),
            (18, false),
            (19, false),
            (17, true),
            (18, true),
        ] {
            let mut rng = Rng::new(7700 + m as u64 + padded as u64 * 1000);
            let mut a = rng.bits(1 << m);
            let mut b = rng.bits(1 << m);
            let padding = if padded {
                let (k_log, useful) = (14usize, 15_409usize);
                let block = 1usize << k_log;
                for blk in 0..((1usize << m) / block) {
                    for j in useful..block {
                        a[blk * block + j] = false;
                        b[blk * block + j] = false;
                    }
                }
                PaddingSpec {
                    k_log,
                    useful_bits_per_block: useful,
                }
            } else {
                PaddingSpec::dense(m)
            };
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

            // Arms: (lookahead, cascade2, cascade3, cascade4, cascade5,
            // nomat) forced-off flags.
            let arms = [
                (false, false, false, false, false, false), // full: nomat sweep + every level
                (false, false, false, false, false, true),  // materializing sweep + every level
                (false, false, false, false, true, false),  // cascade capped at level 3
                (false, false, false, true, true, false),   // capped at level 2 (frontier)
                (false, false, true, true, true, false),    // nomat + lookahead + cascade2
                (false, true, true, true, true, false),     // nomat + lookahead only
                (false, true, true, true, true, true), // materializing lookahead only (5d4d2a9)
                (true, true, true, true, true, true),  // incumbent
            ];
            let n_mlv = m - K_SKIP;
            let all = if n_mlv >= 12 {
                5
            } else if n_mlv >= 10 {
                4
            } else if n_mlv >= 8 {
                3
            } else if n_mlv >= 7 {
                2
            } else {
                1
            };
            let expect_levels = [all, all, all.min(4), all.min(3), all.min(2), 1, 1, 0];
            let expect_nomat = [true, false, true, true, true, true, false, false];
            let mut results = Vec::new();
            for (k, &(la_off, c2_off, c3_off, c4_off, c5_off, nm_off)) in arms.iter().enumerate() {
                ZC_LOOKAHEAD_FORCED_OFF.store(la_off, Ordering::Relaxed);
                ZC_CASCADE2_FORCED_OFF.store(c2_off, Ordering::Relaxed);
                ZC_CASCADE3_FORCED_OFF.store(c3_off, Ordering::Relaxed);
                ZC_CASCADE4_FORCED_OFF.store(c4_off, Ordering::Relaxed);
                ZC_CASCADE5_FORCED_OFF.store(c5_off, Ordering::Relaxed);
                ZC_NOMAT_FORCED_OFF.store(nm_off, Ordering::Relaxed);
                let mut ch = FsChallenger::new(b"flock-test-v0");
                results.push(prove_packed_padded(&a_p, &b_p, &c_p, m, &padding, &mut ch));
                assert_eq!(
                    ZC_LEVELS_LAST.load(Ordering::Relaxed),
                    expect_levels[k],
                    "arm {k} ran the wrong number of levels at m={m}"
                );
                assert_eq!(
                    ZC_NOMAT_LAST.load(Ordering::Relaxed),
                    expect_nomat[k],
                    "arm {k} nomat engagement wrong at m={m}"
                );
            }
            ZC_LOOKAHEAD_FORCED_OFF.store(false, Ordering::Relaxed);
            ZC_CASCADE2_FORCED_OFF.store(false, Ordering::Relaxed);
            ZC_CASCADE3_FORCED_OFF.store(false, Ordering::Relaxed);
            ZC_CASCADE4_FORCED_OFF.store(false, Ordering::Relaxed);
            ZC_CASCADE5_FORCED_OFF.store(false, Ordering::Relaxed);
            ZC_NOMAT_FORCED_OFF.store(false, Ordering::Relaxed);

            let (proof_full, claim_full) = &results[0];
            for (k, (proof, claim)) in results.iter().enumerate().skip(1) {
                assert_eq!(
                    proof_full, proof,
                    "proof diverges arm {k} at m={m} padded={padded}"
                );
                assert_eq!(
                    claim_full, claim,
                    "claim diverges arm {k} at m={m} padded={padded}"
                );
            }

            let mut ch_v = FsChallenger::new(b"flock-test-v0");
            let claim_v = verify(m, proof_full, &mut ch_v)
                .unwrap_or_else(|e| panic!("verify rejected cascade proof at m={m}: {e:?}"));
            assert_eq!(&claim_v, claim_full, "verify claim mismatch at m={m}");
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
}

//! Top-level R1CS prover: composes zerocheck + lincheck for block-diagonal
//! circuit R1CS instances. Outputs **two** z-claims at different quirky
//! points that the PCS layer (when it lands) will verify against `z`'s
//! commitment.
//!
//! Flow:
//! ```text
//!     witness z ──► pack ──► a = A·z, b = B·z, c = z (since C=I)
//!         │
//!         │       ┌─────────────┐
//!         │       │  zerocheck  │  reduces a·b ⊕ c = 0 to MLE claims:
//!         │       │             │  • â(z, mlv_challenges) = v_a
//!         │       │             │  • b̂(z, mlv_challenges) = v_b
//!         │       │             │  • ĉ(z, r_rest)         = v_c  ← directly a z-claim
//!         │       └─────────────┘
//!         │
//!         │       ┌─────────────┐
//!         │ ─► z ─►  lincheck   │  reduces â, b̂ claims (same point) to a
//!         │       │             │  single z-claim at (r_inner_skip,
//!         │       │             │                      r_inner_rest,
//!         │       │             │                      x_ab.x_outer).
//!         │       └─────────────┘
//!         │
//!         ▼
//!     R1csClaim { ab: z-claim from lincheck,  c: z-claim from extract_c }
//! ```

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::{self, QuirkyPoint, pack_z_lincheck_from_packed};
use flock_core::pcs::{self, Commitment, PcsParams};
use flock_core::proof::{R1csClaim, R1csProofLigerito, ZClaim, bind_statement};
use flock_core::r1cs::{BlockR1cs, Round1PinnedOnePrefixStatement};
use flock_core::zerocheck;
use crate::r1cs_hashes::blake3::Blake3IdentityPrefixWitness;

/// Exact rollback for C1. A cleared ranked-worker environment selects the
/// structurally guarded omission; literal `1` retains the complete AB
/// producer and its original completion reads.
pub const ENV_NO_ZC_AB_IDENTITY_PREFIX_OMIT: &str = "FLOCK_NO_ZC_AB_IDENTITY_PREFIX_OMIT";

/// Literal `1` retains C1 while computing window 30 normally.
pub const ENV_NO_ZC_AB_OUTPUT_WINDOW_OMIT: &str = "FLOCK_NO_ZC_AB_OUTPUT_WINDOW_OMIT";

#[inline]
fn ranked_direct_ab_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && r1cs.m == 32
        && r1cs.k_log >= pcs::LOG_PACKING + 2
        && std::env::var_os("FLOCK_NO_OPEN_DIRECT_AB").is_none()
        && std::env::var_os("FLOCK_NO_LIG_FOLD2").is_none()
}

/// Whether the c-claim ships its four-bank sufficient statistic instead of the
/// canonical `s_hat_v_c`. Same shape conditions as the AB quad plus the single
/// process-wide direct-C predicate `pcs`'s consumer reads, so capture and
/// consumer can never disagree; a shape miss simply keeps the 128-vector and
/// the incumbent deferred-C path.
#[inline]
fn ranked_direct_c_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    ranked_direct_ab_precompute_enabled(r1cs) && pcs::ranked_direct_c_enabled()
}

#[inline]
fn ranked_direct_fold4_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    ranked_direct_ab_precompute_enabled(r1cs) && pcs::ranked_direct_fold4_enabled()
}

/// Direct-fold8 capture/consumer predicate: the fold4 chain plus the shared
/// fold8 latch and six retainable tail coordinates (k_log >= k_skip + 7).
/// Read by BOTH the AB/C producers and (transitively) the pcs consumer gate,
/// so capture and consumer cannot disagree; a shape miss simply keeps the
/// 16-bank tensor and the incumbent fold4 route.
#[inline]
fn ranked_direct_fold8_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    ranked_direct_fold4_precompute_enabled(r1cs)
        && pcs::ranked_direct_fold8_enabled()
        && r1cs.k_log >= r1cs.k_skip + 7
}

/// Exact-shape gate for deriving identity C from the already-materialized
/// lincheck stripe. Keep this narrower than the generic DirectFold4 gate:
/// the shortcut relies on ranked BLAKE3's block geometry and honest padding,
/// while every miss retains the incumbent row-major C producer.
#[inline]
fn ranked_lincheck_c_reuse_enabled(r1cs: &BlockR1cs) -> bool {
    ranked_direct_fold4_precompute_enabled(r1cs)
        && r1cs.m == 32
        && r1cs.k_log == 14
        && r1cs.k_skip == zerocheck::K_SKIP
        && r1cs.useful_bits == 15_409
        && r1cs.layout == flock_core::r1cs::WitnessLayout::RowMajor
        && r1cs.c0_is_identity()
        && std::env::var_os("FLOCK_NO_ZC_LINCHECK_C_REUSE").is_none()
}

/// C1 needs both statement properties and the exact row-major BLAKE3
/// producer's value guarantee. Generic witnesses never receive the latter.
#[inline]
fn ranked_identity_prefix_omission_contract(
    r1cs: &BlockR1cs,
    witness: Option<&Blake3IdentityPrefixWitness>,
    z: &[F128],
    a: &[F128],
    b: &[F128],
) -> Option<Round1PinnedOnePrefixStatement> {
    let witness = witness?;
    if !ranked_lincheck_c_reuse_enabled(r1cs)
        || std::env::var_os(ENV_NO_ZC_AB_IDENTITY_PREFIX_OMIT).as_deref()
            == Some(std::ffi::OsStr::new("1"))
        || !witness.matches(z, a, b)
        || z.len() != 1usize << (r1cs.m - 7)
    {
        return None;
    }
    if std::env::var_os(ENV_NO_ZC_AB_OUTPUT_WINDOW_OMIT).as_deref()
        != Some(std::ffi::OsStr::new("1"))
        && let Some(statement) = r1cs.round1_pinned_one_prefix_and_output_statement()
    {
        return Some(statement);
    }
    r1cs.round1_pinned_one_prefix_statement()
}

fn precompute_ab_s_hat_v(
    r1cs: &BlockR1cs,
    z_vec: &[F128],
    inner_rest_tail: &[F128],
) -> Option<Vec<F128>> {
    if ranked_direct_fold8_precompute_enabled(r1cs) {
        Some(pcs::ring_switch::s_hat_v_fold8_from_z_vec(
            z_vec,
            inner_rest_tail,
        ))
    } else if ranked_direct_fold4_precompute_enabled(r1cs) {
        Some(pcs::ring_switch::s_hat_v_fold4_from_z_vec(
            z_vec,
            inner_rest_tail,
        ))
    } else if ranked_direct_ab_precompute_enabled(r1cs) {
        Some(pcs::ring_switch::s_hat_v_quad_from_z_vec(
            z_vec,
            inner_rest_tail,
        ))
    } else if r1cs.k_log >= pcs::LOG_PACKING {
        Some(pcs::ring_switch::s_hat_v_from_z_vec(z_vec, inner_rest_tail))
    } else {
        None
    }
}

/// Pick C's precomputed slot. The DirectFold8 route takes the sixty-four-bank
/// tensor; the strict DirectFold4 experiment takes the sixteen-bank tensor;
/// the incumbent ranked path takes the four-bank tensor; every other shape
/// takes the canonical transcript-visible statistic. Falls back through
/// narrower captures by presence, so a producer shape miss (e.g. no lincheck
/// stripe) degrades to a still-correct route instead of panicking.
#[inline]
fn pre_c_slot<'a>(r1cs: &BlockR1cs, captured: &'a zerocheck::CapturedSHatVC) -> Option<&'a [F128]> {
    Some(
        if ranked_direct_fold8_precompute_enabled(r1cs) && captured.fold8.is_some() {
            captured.fold8.as_deref().unwrap()
        } else if ranked_direct_fold4_precompute_enabled(r1cs) && captured.fold4.is_some() {
            captured.fold4.as_deref().unwrap()
        } else if ranked_direct_c_precompute_enabled(r1cs) {
            captured.quad.as_slice()
        } else {
            captured.s_hat_v_c.as_slice()
        },
    )
}

/// Construct a multilinear `x_outer_full` of length `m − k_skip` from a
/// QuirkyPoint: concatenate `x_inner_rest` and `x_outer`. This is the format
/// the PCS expects (k_skip = 6 absorbed via `z_skip`; everything else is
/// multilinear).
pub(crate) fn quirky_x_outer_full(point: &QuirkyPoint) -> Vec<F128> {
    let mut v = Vec::with_capacity(point.x_inner_rest.len() + point.x_outer.len());
    v.extend_from_slice(&point.x_inner_rest);
    v.extend_from_slice(&point.x_outer);
    v
}

/// Batched PCS open over an arbitrary list of `ẑ`-evaluation claims. This is
/// the generic seam: the base R1CS proof opens `[ab, c]`; relation wrappers
/// (e.g. the hash chain) append their own claims and open `[ab, c, …]`.
/// Per-claim optional precomputed `s_hat_v` is passed through to ring-switch:
/// when `Some(v)`, the claim skips `fold_1b_rows` and uses `v` directly.
/// Caller responsibility: each `Some(v)` MUST equal what `fold_1b_rows` would
/// produce on `z_packed` against the claim's suffix — see
/// [`pcs::ring_switch::s_hat_v_from_z_vec`] for the AB-claim derivation.
///
/// Must be called at the same transcript position as the verifier's
/// [`flock_core::verifier::verify_claims_ligerito`].
pub(crate) fn open_claims_with_precomputed_ligerito<Ch: Challenger>(
    z_packed: Vec<F128>,
    prover_data: &pcs::ProverData,
    commitment: &Commitment,
    claims: &[ZClaim],
    precomputed_s_hat_v: &[Option<&[F128]>],
    padding: &zerocheck::PaddingSpec,
    lig_config: &pcs::ligerito::ProverConfig,
    challenger: &mut Ch,
) -> pcs::BatchOpeningProofLigerito {
    let x_fulls: Vec<Vec<F128>> = claims
        .iter()
        .map(|c| quirky_x_outer_full(&c.point))
        .collect();
    let x_refs: Vec<&[F128]> = x_fulls.iter().map(|v| v.as_slice()).collect();
    let out = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        prover_data,
        commitment,
        &x_refs,
        precomputed_s_hat_v,
        &[],
        padding,
        lig_config,
        challenger,
    );
    flock_core::tmark::mark("open body end");
    out
}

/// Run the full R1CS proof on an F_{2^128}-packed witness.
///
/// The witness is in the canonical packed form (polynomial basis: bit `r` of
/// `z_packed[i]` = logical bit `i·128 + r`), length `2^(m - 7)`. The prover
/// never unpacks; downstream R1CS/zerocheck/lincheck/PCS all consume packed
/// representations.
///
/// Returns the proof bundle, the witness commitment, and the two claims (which
/// the verifier needs to know to check the openings).
pub fn prove_ligerito<Ch: Challenger>(
    r1cs: &BlockR1cs,
    z_packed: Vec<F128>,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    assert_eq!(
        r1cs.layout,
        flock_core::r1cs::WitnessLayout::RowMajor,
        "the generic matrix-driven provers assume the row-major layout \
         (block-diagonal apply + lincheck stripe packing); batch-major \
         setups must use the per-hash prove_fast paths"
    );
    assert_eq!(z_packed.len(), 1usize << (r1cs.m - 7));
    assert_eq!(pcs_params.m, r1cs.m);

    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito default config; bump m for tiny instances");

    let (commitment, prover_data) = pcs::commit(&z_packed, pcs_params);
    bind_statement(challenger, r1cs, &commitment);

    // a = A·z, b = B·z; for the C = I convention c aliases z.
    let a_packed_f128 = r1cs.apply_a_packed(&z_packed);
    let b_packed_f128 = r1cs.apply_b_packed(&z_packed);
    let c_packed_f128: Vec<F128> = if r1cs.c0_is_identity() {
        Vec::new()
    } else {
        r1cs.apply_c_packed(&z_packed)
    };
    let cast = |v: &[F128]| -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    };
    let a_packed: &[u8] = cast(&a_packed_f128);
    let b_packed: &[u8] = cast(&b_packed_f128);
    let c_packed: &[u8] = if c_packed_f128.is_empty() {
        cast(&z_packed)
    } else {
        cast(&c_packed_f128)
    };
    let z_packed_lincheck = pack_z_lincheck_from_packed(&z_packed, r1cs.m, r1cs.k_log);

    let padding = r1cs.padding_spec();
    let (zc_proof, zc_claim, s_hat_v_c) = zerocheck::prove_packed_padded_capture_s_hat_v_c(
        a_packed, b_packed, c_packed, r1cs.m, &padding, challenger,
    );

    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

    let lc_circuit =
        lincheck::SparseMatrixCircuit::new(&r1cs.a_0, &r1cs.b_0).with_const_pin(r1cs.const_pin);
    let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_padded_capture_z_vec(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        &lc_circuit,
        &x_ab,
        challenger,
    );

    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };

    let s_hat_v_ab = precompute_ab_s_hat_v(r1cs, &z_vec_pre, &lc_claim.r_inner_rest[1..]);
    // z_vec_pre only fed s_hat_v_ab; recycle before PCS open residency.
    flock_core::scratch::give_f128(z_vec_pre);
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = pre_c_slot(r1cs, &s_hat_v_c);
    let pcs_open = open_claims_with_precomputed_ligerito(
        z_packed,
        &prover_data,
        &commitment,
        &[ab.clone(), c.clone()],
        &[pre_ab, pre_c],
        &padding,
        &lig_config,
        challenger,
    );

    let proof = R1csProofLigerito {
        zerocheck: zc_proof,
        lincheck: lc_proof,
        pcs_open,
    };
    let claim = R1csClaim { ab, c };
    (proof, commitment, claim)
}

/// Shared `prove_fast` pipeline for the monolithic hash R1CS modules. Takes
/// the four packed buffers produced by the per-hash
/// `generate_witness_with_ab_packed_and_lincheck` and runs commit → zerocheck
/// → lincheck → PCS-open. Uses the c-aliasing trick (`C = I` → `c == z`
/// byte-for-byte). Used by per-hash modules' `prove_fast_ligerito` methods.
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_from_witness<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    prove_fast_ligerito_from_witness_with_identity_prefix(
        r1cs, pcs_params, z_packed, a_packed_f128, b_packed_f128,
        z_packed_lincheck, lincheck_circuit, prefaulted_codeword, None, challenger,
    )
}

/// BLAKE3-only counterpart: the receipt is minted after its witness writer
/// joins and is bound to these three allocations. No matrix-only inference.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_fast_ligerito_from_witness_with_identity_prefix<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    identity_prefix_witness: Option<Blake3IdentityPrefixWitness>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    let commit_codeword = match prefaulted_codeword {
        Some(codeword) => CommitCodeword::NeedsReplication(codeword),
        None => CommitCodeword::Allocate,
    };
    prove_fast_ligerito_from_witness_with_commit_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        LincheckStripeInput::Ready(z_packed_lincheck),
        lincheck_circuit,
        commit_codeword,
        identity_prefix_witness,
        None,
        challenger,
    )
}

/// Ranked row-major counterpart of [`prove_fast_ligerito_from_witness`].
/// `codeword` already contains the post-trivial-layer state, so commit starts
/// directly at the remaining NTT layers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_fast_ligerito_from_preinitialized_codeword<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    codeword: Vec<F128>,
    identity_prefix_witness: Option<Blake3IdentityPrefixWitness>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    prove_fast_ligerito_from_witness_with_commit_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        LincheckStripeInput::Ready(z_packed_lincheck),
        lincheck_circuit,
        CommitCodeword::Preinitialized(codeword),
        identity_prefix_witness,
        None,
        challenger,
    )
}

/// Ranked from-message path whose first GPU NTT pass was launched in bands
/// during witness generation. The remaining graph still runs in the commit
/// arm concurrently with round-1 AB precomputation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_fast_ligerito_from_streamed_first_pass<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    codeword: Vec<F128>,
    stream: flock_core::gpu_commit::FromZFirstPassStream,
    identity_prefix_witness: Option<Blake3IdentityPrefixWitness>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    prove_fast_ligerito_from_witness_with_commit_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        LincheckStripeInput::Ready(z_packed_lincheck),
        lincheck_circuit,
        CommitCodeword::StreamedFirstPass(codeword, stream),
        identity_prefix_witness,
        None,
        challenger,
    )
}

/// Ranked path that moves the 512 MiB lincheck transpose out of
/// witness generation. Two utility-pool workers transpose the immutable
/// packed witness behind a scoped coordinator while commit, round-1 AB
/// preprocessing, and zerocheck run; the scope is joined immediately before
/// lincheck can read the stripe. The caller's exact-shape/helper-availability
/// gate is deliberately kept in the BLAKE3 witness driver, where omission of
/// the eager stripe is decided.
///
/// `ab_producer` is `Some` when the witness driver deferred the A/B rows:
/// `a_packed_f128` / `b_packed_f128` then hold only their per-block
/// constants, and the round-one AB precompute produces every row (see
/// [`zerocheck::univariate_skip_optimized::Round1AbProducer`]) before any
/// other reader of A/B runs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_fast_ligerito_from_streamed_first_pass_deferred_stripe<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    codeword: Vec<F128>,
    stream: flock_core::gpu_commit::FromZFirstPassStream,
    identity_prefix_witness: Option<Blake3IdentityPrefixWitness>,
    ab_producer: Option<zerocheck::univariate_skip_optimized::Round1AbProducer<'_>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    prove_fast_ligerito_from_witness_with_commit_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        LincheckStripeInput::DeferredRanked,
        lincheck_circuit,
        CommitCodeword::StreamedFirstPass(codeword, stream),
        identity_prefix_witness,
        ab_producer,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_fast_ligerito_from_witness_with_commit_codeword<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: LincheckStripeInput,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    commit_codeword: CommitCodeword,
    identity_prefix_witness: Option<Blake3IdentityPrefixWitness>,
    ab_producer: Option<zerocheck::univariate_skip_optimized::Round1AbProducer<'_>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito default config; bump m for tiny instances");

    let ProveCore {
        zc_proof,
        lc_proof,
        ab,
        c,
        commitment,
        prover_data,
        z_packed,
        s_hat_v_ab,
        s_hat_v_c,
    } = prove_fast_core_with_commit_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        z_packed_lincheck,
        lincheck_circuit,
        commit_codeword,
        identity_prefix_witness,
        ab_producer,
        challenger,
    );

    let padding = r1cs.padding_spec();
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = pre_c_slot(r1cs, &s_hat_v_c);
    let phase_timing = std::env::var_os("FLOCK_PHASE_TIMING").is_some();
    flock_core::tmark::mark("open start");
    let cpu_open0 = phase_timing.then(process_cpu_ms);
    let t_open = std::time::Instant::now();
    // Publish-prefix pre-encode: `commitment` / `zc_proof` / `lc_proof` are
    // transcript-final here — the open below only produces `pcs_open` — so
    // the publish-tail's 450 kB output allocation and ~4.3 kB prefix encode
    // run on one idle E-core (UTILITY QoS) while the ~19 ms open owns the
    // P-cores + GPU. `in_place_scope` keeps the open itself on this thread
    // unchanged; its end-of-scope join is free because the spawned task
    // finishes orders of magnitude before the open does. Worst-case
    // contention is one E-core busy for tens of µs if an epool broadcast
    // lands in that window — queue semantics (and therefore bytes) are
    // unaffected. A deliberately single-threaded main pool spawns nothing,
    // preserving truly serial execution (same contract as epool's hetero
    // queue); publish then takes the incumbent full-encode path, as it does
    // on hosts without a helper pool or under FLOCK_NO_PRE_ENCODE=1.
    let stash_pool = (crate::proof_io::pre_encode_enabled() && rayon::current_num_threads() > 1)
        .then(flock_core::epool::helper_pool)
        .flatten();
    let claims = [ab, c];
    let pcs_open = match stash_pool {
        Some(pool) => pool.in_place_scope(|s| {
            s.spawn(|_| {
                crate::proof_io::stash_pre_encoded_prefix(&commitment, &zc_proof, &lc_proof)
            });
            // The open runs on a main-pool worker so its parallel regions
            // start as nested joins (`flock_core::in_pool`); the helper scope
            // and its spawned prefix encode are unchanged.
            flock_core::in_pool(|| {
                flock_core::tmark::mark("open body start");
                open_claims_with_precomputed_ligerito(
                    z_packed,
                    &prover_data,
                    &commitment,
                    &claims,
                    &[pre_ab, pre_c],
                    &padding,
                    &lig_config,
                    challenger,
                )
            })
        }),
        None => flock_core::in_pool(|| {
            open_claims_with_precomputed_ligerito(
                z_packed,
                &prover_data,
                &commitment,
                &claims,
                &[pre_ab, pre_c],
                &padding,
                &lig_config,
                challenger,
            )
        }),
    };
    let [ab, c] = claims;
    flock_core::tmark::mark("open returned");
    if phase_timing {
        let wall = t_open.elapsed().as_secs_f64() * 1e3;
        let cpu = process_cpu_ms() - cpu_open0.unwrap_or(0.0);
        eprintln!(
            "[phase-timing] pcs-open: {wall:.2} ms cpu={cpu:.1} util={:.1}",
            cpu / wall
        );
        eprintln!("[minflt] at open end: {}", flock_core::pcs::commit::process_minflt());
    }

    let proof = R1csProofLigerito {
        zerocheck: zc_proof,
        lincheck: lc_proof,
        pcs_open,
    };
    let claim = R1csClaim { ab, c };
    (proof, commitment, claim)
}

/// Everything the prover produces *before* the PCS open: the zerocheck +
/// lincheck sub-proofs, the two base z-claims (`ab`, `c`), and the retained
/// commitment / prover-data / packed witness needed to open more claims.
///
/// The generic seam: `prove_fast_ligerito_from_witness` = `prove_fast_core` +
/// `open_claims([ab, c])`; a relation wrapper (e.g. the hash chain) runs the
/// same core, derives extra z-claims, and calls `open_claims([ab, c, …])`.
pub struct ProveCore {
    pub zc_proof: zerocheck::ZerocheckProof,
    pub lc_proof: lincheck::LincheckProof,
    pub ab: ZClaim,
    pub c: ZClaim,
    pub commitment: Commitment,
    pub prover_data: pcs::ProverData,
    pub z_packed: Vec<F128>,
    /// Precomputed `s_hat_v` for the AB claim — derived from lincheck's
    /// pre-sumcheck `z_vec` via [`pcs::ring_switch::s_hat_v_from_z_vec`].
    /// Skips `fold_1b_rows` for the AB claim at PCS-open time.
    ///
    /// `None` when `k_log < LOG_PACKING` (the kernel needs `z_vec.len() ==
    /// 2^LOG_PACKING * 2^tail.len()`, which requires `k_log >= LOG_PACKING`).
    /// Real R1CS instances have `k_log >= 16` so this branch only fires in
    /// tiny test setups.
    pub s_hat_v_ab: Option<Vec<F128>>,
    /// Precomputed opening statistics for the C claim — produced by zerocheck
    /// round 1's eight-bank fusion kernel. Skips `fold_1b_rows` for the C claim
    /// at PCS-open time; [`ranked_direct_c_precompute_enabled`] decides which of
    /// the two forms goes into the open.
    pub s_hat_v_c: zerocheck::CapturedSHatVC,
}

/// Ownership and initialization state of the commit codeword buffer.
///
/// Keeping these states distinct prevents the ranked path from accidentally
/// replicating the witness over a buffer its witness workers already filled.
enum CommitCodeword {
    Allocate,
    NeedsReplication(Vec<F128>),
    Preinitialized(Vec<F128>),
    StreamedFirstPass(Vec<F128>, flock_core::gpu_commit::FromZFirstPassStream),
}

/// Lincheck stripe ownership at the commit boundary. `DeferredRanked` is
/// constructed only by the exact BLAKE3 streamed-GPU selector; all generic
/// and fallback paths carry the already-materialized byte stripe.
enum LincheckStripeInput {
    Ready(Vec<u8>),
    DeferredRanked,
}

const DEFERRED_STRIPE_GROUPS_PER_JOB: usize = 32;

fn fill_deferred_lincheck_stripe_group(
    z_packed: &[F128],
    stripe: &mut [u8],
    group: usize,
    group_f128: usize,
    u64_per_block: usize,
    useful_words: usize,
) {
    let group_start = group * group_f128;
    let z_group = &z_packed[group_start..group_start + group_f128];
    // SAFETY: F128 is repr(C) as two little-endian u64 halves; this is the
    // same read-only view used by the eager witness transpose.
    let z_u64: &[u64] =
        unsafe { std::slice::from_raw_parts(z_group.as_ptr().cast::<u64>(), z_group.len() * 2) };
    #[cfg(not(target_arch = "aarch64"))]
    let mut transposed = [0u8; 64];
    // Every read below is `lane * u64_per_block + word` with `lane < 8` and
    // `word < useful_words <= u64_per_block`, inside the group's 8 blocks.
    assert!(useful_words <= u64_per_block && z_u64.len() == 8 * u64_per_block);
    for word in 0..useful_words {
        let dst = &mut stripe[word * 64..word * 64 + 64];
        // aarch64: the eight lane words go from memory straight into vector
        // lanes, the transpose stays in registers, and the four result
        // registers are stored directly; same bytes as the array path.
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the reads are bounded by the assert above; `dst` is 64
        // writable bytes.
        unsafe {
            // Prefetch ahead in each input lane, keeping every hint within
            // the useful part of this group.
            if word % 16 == 0 && word + 16 < useful_words {
                for lane in 0..8 {
                    let address = z_u64.as_ptr().add(lane * u64_per_block + word + 16);
                    core::arch::asm!("prfm pldl1keep, [{p}]", p=in(reg) address, options(nostack, readonly, preserves_flags));
                }
            }
            let [y0, y1, y2, y3] = flock_core::bits::transpose_8_strided_u64s_to_regs(
                z_u64.as_ptr().add(word),
                u64_per_block,
            );
            // The stripe is not read until after commit+zerocheck, so use the
            // same cache-bypassing store flavor as the eager SIMD witness
            // path. `stnp` permits these 64-byte slices without an extra
            // alignment contract.
            core::arch::asm!(
                "stnp {y0:q}, {y1:q}, [{dst}]",
                "stnp {y2:q}, {y3:q}, [{dst}, #32]",
                dst = in(reg) dst.as_mut_ptr(),
                y0 = in(vreg) y0,
                y1 = in(vreg) y1,
                y2 = in(vreg) y2,
                y3 = in(vreg) y3,
                options(nostack, preserves_flags)
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let lanes: [u64; 8] = std::array::from_fn(|lane| z_u64[lane * u64_per_block + word]);
            flock_core::bits::transpose_8_u64s_to_64_bytes(&lanes, &mut transposed);
            dst.copy_from_slice(&transposed);
        }
    }
    // The ranked padded fold never observes the tail. Keeping the honest
    // full-stripe contract in tests catches accidental selector widening.
    #[cfg(test)]
    stripe[useful_words * 64..].fill(0);
}

/// Root-compacted sibling of [`fill_deferred_lincheck_stripe_group`]: writes
/// the group's 177 root rows (`r1cs_hashes::blake3::RootRow`) instead of its
/// 241 useful words. Row `j`'s eight lane words are
/// `(w[lo] & lo_mask) | (w[lo + 1] & hi_mask)`; a full row (all 64 bits from
/// one word) takes the incumbent strided-transpose fast path. Bytes past row
/// 177 of the group are never read (every fold runs over `N_ROOT_COLS`).
fn fill_root_lincheck_stripe_group(
    z_packed: &[F128],
    stripe: &mut [u8],
    group: usize,
    group_f128: usize,
    u64_per_block: usize,
    rows: &[crate::r1cs_hashes::blake3::RootRow],
) {
    let group_start = group * group_f128;
    let z_group = &z_packed[group_start..group_start + group_f128];
    // SAFETY: F128 is repr(C) as two little-endian u64 halves.
    let z_u64: &[u64] =
        unsafe { std::slice::from_raw_parts(z_group.as_ptr().cast::<u64>(), z_group.len() * 2) };
    assert!(z_u64.len() == 8 * u64_per_block && rows.len() * 64 <= stripe.len());
    // Words each lane reads: through the last row's `lo_word`, plus one when
    // that row also takes bits from the next word.
    #[cfg(target_arch = "aarch64")]
    let read_words = rows
        .last()
        .map_or(0, |row| row.lo_word + 1 + usize::from(row.hi_mask != 0));
    #[cfg(target_arch = "aarch64")]
    let mut prefetched_line = usize::MAX;
    for (j, row) in rows.iter().enumerate() {
        let full = row.lo_mask == u64::MAX;
        // Reads stay inside the group: lo_word (+1 iff hi_mask != 0) < u64_per_block.
        assert!(row.lo_word + usize::from(row.hi_mask != 0) < u64_per_block);
        let lanes: [u64; 8] = if full {
            [0; 8]
        } else {
            std::array::from_fn(|lane| {
                let base = lane * u64_per_block + row.lo_word;
                let hi = if row.hi_mask != 0 { z_u64[base + 1] & row.hi_mask } else { 0 };
                (z_u64[base] & row.lo_mask) | hi
            })
        };
        let dst = &mut stripe[j * 64..j * 64 + 64];
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the strided reads are bounded by the asserts above (or read
        // the 8-word local array with stride 1); `dst` is 64 writable bytes.
        unsafe {
            // Prefetch each input lane 16 words ahead, once per 16-word line,
            // keeping every hint within the words this group reads.
            let ahead = row.lo_word + 16;
            if ahead < read_words && ahead / 16 != prefetched_line {
                prefetched_line = ahead / 16;
                for lane in 0..8 {
                    let address = z_u64.as_ptr().add(lane * u64_per_block + ahead);
                    core::arch::asm!("prfm pldl1keep, [{p}]", p=in(reg) address, options(nostack, readonly, preserves_flags));
                }
            }
            let [y0, y1, y2, y3] = if full {
                flock_core::bits::transpose_8_strided_u64s_to_regs(
                    z_u64.as_ptr().add(row.lo_word),
                    u64_per_block,
                )
            } else {
                flock_core::bits::transpose_8_strided_u64s_to_regs(lanes.as_ptr(), 1)
            };
            core::arch::asm!(
                "stnp {y0:q}, {y1:q}, [{dst}]",
                "stnp {y2:q}, {y3:q}, [{dst}, #32]",
                dst = in(reg) dst.as_mut_ptr(),
                y0 = in(vreg) y0,
                y1 = in(vreg) y1,
                y2 = in(vreg) y2,
                y3 = in(vreg) y3,
                options(nostack, preserves_flags)
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let lanes: [u64; 8] = if full {
                std::array::from_fn(|lane| z_u64[lane * u64_per_block + row.lo_word])
            } else {
                lanes
            };
            let mut transposed = [0u8; 64];
            flock_core::bits::transpose_8_u64s_to_64_bytes(&lanes, &mut transposed);
            dst.copy_from_slice(&transposed);
        }
    }
}

/// Most root rows any stripe uses (`N_ROOT_COLS.div_ceil(64)`).
const AB_STRIPE_MAX_ROWS: usize = 177;

/// Distinguishes emitters, so a thread-local group left by an earlier prove
/// is never taken for this prove's even quad.
static AB_STRIPE_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Per-thread copy of one stripe group's z words (8 blocks, the packed
/// layout of `z_packed`), rebuilt as a AND b from the producer's scratch.
struct AbStripeZGroup {
    epoch: u64,
    group: usize,
    words: [F128; 8 * AB_STRIPE_WORDS / 2],
}

thread_local! {
    static AB_STRIPE_ZGROUP: std::cell::RefCell<AbStripeZGroup> = const {
        std::cell::RefCell::new(AbStripeZGroup {
            epoch: u64::MAX,
            group: usize::MAX,
            words: [F128::ZERO; 8 * AB_STRIPE_WORDS / 2],
        })
    };
}

/// Emits the root lincheck stripe from the deferred A/B producer's scratch.
///
/// A stripe group is 8 consecutive blocks = quads `2g` and `2g + 1`. The
/// fused AB queue hands each job a run of whole, even-aligned units in
/// ascending order on one thread, so the even quad's words are stashed
/// thread-locally and the group is transposed and stored when the odd quad
/// arrives. Any group whose two quads did not meet on one thread (none on
/// the ranked queue; every group when the producer runs without scratch) is
/// left to [`AbStripeEmitter::finish`], which fills it from `z` exactly as
/// the E-core stripe arm does. Every group is written by exactly one path,
/// and both write the bytes of [`fill_root_lincheck_stripe_group`].
struct AbStripeEmitter<'a> {
    stripe: flock_core::epool::SyncPtr<u8>,
    group_bytes: usize,
    n_groups: usize,
    rows: &'static [crate::r1cs_hashes::blake3::RootRow],
    fixed_inputs: bool,
    mismatch: &'a std::sync::atomic::AtomicBool,
    emitted: Box<[std::sync::atomic::AtomicU64]>,
    epoch: u64,
    /// Words of a block the rows and the fixed-input checks read.
    read_words: usize,
}

impl<'a> AbStripeEmitter<'a> {
    fn new(
        stripe: flock_core::epool::SyncPtr<u8>,
        stripe_len: usize,
        group_bytes: usize,
        rows: &'static [crate::r1cs_hashes::blake3::RootRow],
        fixed_inputs: bool,
        mismatch: &'a std::sync::atomic::AtomicBool,
    ) -> Self {
        // One group is 8 blocks of `group_bytes / 8` bytes = 2 KiB of packed
        // u64 words per block (the producer's quad layout).
        assert_eq!(group_bytes, 1 << 14);
        assert!(rows.len() <= AB_STRIPE_MAX_ROWS && rows.len() * 64 <= group_bytes);
        assert!(rows.iter().all(|r| r.lo_word + usize::from(r.hi_mask != 0) < AB_STRIPE_WORDS));
        let n_groups = stripe_len / group_bytes;
        Self {
            stripe,
            group_bytes,
            n_groups,
            rows,
            fixed_inputs,
            mismatch,
            emitted: (0..n_groups.div_ceil(64))
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            epoch: AB_STRIPE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            read_words: rows
                .iter()
                .map(|r| r.lo_word + 1 + usize::from(r.hi_mask != 0))
                .chain(crate::r1cs_hashes::blake3::FIXED_INPUT_CHECKS.iter().map(|c| c.0 + 1))
                .max()
                .unwrap_or(0),
        }
    }

    /// Called right after the producer wrote quad `unit`'s complete a and b
    /// blocks to the two scratch buffers (null: no scratch, nothing to do).
    fn emit(&self, unit: usize, a_scratch: *mut u8, b_scratch: *mut u8) {
        if a_scratch.is_null() || b_scratch.is_null() {
            return;
        }
        let group = unit / 2;
        let half = unit % 2;
        if group >= self.n_groups {
            return;
        }
        let a = a_scratch.cast_const().cast::<u64>();
        let b = b_scratch.cast_const().cast::<u64>();
        AB_STRIPE_ZGROUP.with_borrow_mut(|st| {
            if half == 1 && (st.epoch != self.epoch || st.group != group) {
                return;
            }
            // z = a AND b over the words the rows (and checks) read.
            // SAFETY: the scratch holds 4 complete blocks of
            // `AB_STRIPE_WORDS` words; `read_words <= AB_STRIPE_WORDS`.
            let zw: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(st.words.as_mut_ptr().cast::<u64>(), 8 * AB_STRIPE_WORDS)
            };
            for c in 0..4 {
                let (a_blk, b_blk) = unsafe {
                    (
                        std::slice::from_raw_parts(a.add(c * AB_STRIPE_WORDS), self.read_words),
                        std::slice::from_raw_parts(b.add(c * AB_STRIPE_WORDS), self.read_words),
                    )
                };
                let base = (4 * half + c) * AB_STRIPE_WORDS;
                for ((d, x), y) in zw[base..base + self.read_words].iter_mut().zip(a_blk).zip(b_blk) {
                    *d = x & y;
                }
            }
            if half == 0 {
                st.epoch = self.epoch;
                st.group = group;
                return;
            }
            st.group = usize::MAX;
            // SAFETY: group `group` owns its `group_bytes` of the stripe;
            // only this call writes them.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    self.stripe.ptr().add(group * self.group_bytes),
                    self.group_bytes,
                )
            };
            let group_f128 = 8 * AB_STRIPE_WORDS / 2;
            fill_root_lincheck_stripe_group(&st.words, dst, 0, group_f128, AB_STRIPE_WORDS, self.rows);
            if self.fixed_inputs && !fixed_inputs_hold(&st.words, 0, group_f128, AB_STRIPE_WORDS) {
                self.mismatch.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            self.emitted[group / 64].fetch_or(1 << (group % 64), std::sync::atomic::Ordering::Relaxed);
        });
    }

    /// After every unit was produced (the AB arm has joined its workers):
    /// fill the groups no thread emitted with the incumbent fill job (one
    /// job = `DEFERRED_STRIPE_GROUPS_PER_JOB` groups; jobs with no missing
    /// group are skipped, partially missing ones are refilled whole, which
    /// rewrites identical bytes). Returns the number of missing groups.
    fn finish(&self, _z_packed: &[F128], job: &(dyn Fn(usize) + Sync)) -> usize {
        use rayon::prelude::*;
        let per_job = DEFERRED_STRIPE_GROUPS_PER_JOB;
        let missing = |g: usize| {
            self.emitted[g / 64].load(std::sync::atomic::Ordering::Relaxed) & (1 << (g % 64)) == 0
        };
        let n_missing = (0..self.n_groups).filter(|&g| missing(g)).count();
        if n_missing > 0 {
            let jobs: Vec<usize> = (0..self.n_groups.div_ceil(per_job))
                .filter(|&j| (j * per_job..((j + 1) * per_job).min(self.n_groups)).any(missing))
                .collect();
            jobs.par_iter().for_each(|&j| job(j));
        }
        n_missing
    }
}

/// Packed u64 words per block at the ranked shape (`2^14 / 64`).
const AB_STRIPE_WORDS: usize = 256;

/// Whether all eight instances of stripe group `group` carry the ranked
/// `block_len`/`flags` (`r1cs_hashes::blake3::FIXED_INPUT_CHECKS`). Reads the
/// two packed words the root rows of this group already loaded.
fn fixed_inputs_hold(z_packed: &[F128], group: usize, group_f128: usize, u64_per_block: usize) -> bool {
    let group_start = group * group_f128;
    let z_group = &z_packed[group_start..group_start + group_f128];
    // SAFETY: F128 is repr(C) as two little-endian u64 halves.
    let z_u64: &[u64] =
        unsafe { std::slice::from_raw_parts(z_group.as_ptr().cast::<u64>(), z_group.len() * 2) };
    let mut bad = 0u64;
    for lane in 0..8 {
        for &(word, mask, want) in crate::r1cs_hashes::blake3::FIXED_INPUT_CHECKS.iter() {
            bad |= (z_u64[lane * u64_per_block + word] & mask) ^ want;
        }
    }
    bad == 0
}

/// The deferred stripe fill as a chunk job: returns the job count and the job
/// closure, so the closure can outlive one dispatch call and be shared by the
/// helper pool and main-pool help drains. Job `j` writes exactly the stripe
/// groups `j*64..(j+1)*64` (clamped), each one disjoint `k`-byte range, and
/// reads only `z_packed`.
///
/// The closure writes through a raw pointer to `z_lincheck`: the caller must
/// keep that buffer alive and untouched until every job has completed.
fn deferred_lincheck_stripe_job<'a>(
    z_packed: &'a [F128],
    z_lincheck: &mut [u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    root_rows: Option<&'static [crate::r1cs_hashes::blake3::RootRow]>,
    input_mismatch: Option<&'a std::sync::atomic::AtomicBool>,
) -> (usize, impl Fn(usize) + Sync + use<'a>) {
    assert!(k_log >= 7, "packed block must contain whole F128 words");
    assert!(
        m >= k_log + 3,
        "lincheck layout needs at least eight blocks"
    );
    let k = 1usize << k_log;
    let n_total = 1usize << m;
    assert!(useful_bits <= k);
    let f128_per_block = k / 128;
    let u64_per_block = k / 64;
    let group_f128 = 8 * f128_per_block;
    let useful_words = useful_bits.div_ceil(64);
    assert_eq!(z_packed.len(), n_total / 128);
    assert_eq!(z_lincheck.len(), n_total / 8);

    let n_groups = z_lincheck.len() / k;
    let n_jobs = n_groups.div_ceil(DEFERRED_STRIPE_GROUPS_PER_JOB);
    let stripe_base = flock_core::epool::SyncPtr(z_lincheck.as_mut_ptr());
    let job = move |job: usize| {
        let group_start = job * DEFERRED_STRIPE_GROUPS_PER_JOB;
        let group_end = (group_start + DEFERRED_STRIPE_GROUPS_PER_JOB).min(n_groups);
        for group in group_start..group_end {
            // SAFETY: every dispatched job owns a distinct contiguous set of
            // groups, and each group maps to one disjoint k-byte stripe.
            let stripe =
                unsafe { std::slice::from_raw_parts_mut(stripe_base.ptr().add(group * k), k) };
            match root_rows {
                Some(rows) => {
                    fill_root_lincheck_stripe_group(
                        z_packed,
                        stripe,
                        group,
                        group_f128,
                        u64_per_block,
                        rows,
                    );
                    // Fixed-input stripe: the omitted block_len/flags columns
                    // must hold the ranked constants in all eight instances;
                    // the producer falls back to the 177-row stripe otherwise.
                    if let Some(flag) = input_mismatch {
                        if !fixed_inputs_hold(z_packed, group, group_f128, u64_per_block) {
                            flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
                None => fill_deferred_lincheck_stripe_group(
                    z_packed,
                    stripe,
                    group,
                    group_f128,
                    u64_per_block,
                    useful_words,
                ),
            }
        }
    };
    (n_jobs, job)
}

/// Materialize a BLAKE3-style lincheck stripe from immutable row-major packed
/// `z`, dispatching coarse, pairwise-disjoint stripe slabs through `dispatch`.
/// Production calls only the asserted ranked geometry below; compact
/// congruent shapes give the exact fill oracle a cheap unit-test surface.
fn fill_deferred_lincheck_stripe_with_dispatch(
    z_packed: &[F128],
    z_lincheck: &mut [u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    dispatch: impl FnOnce(usize, &(dyn Fn(usize) + Sync)) -> bool,
) -> bool {
    let (n_jobs, job) =
        deferred_lincheck_stripe_job(z_packed, z_lincheck, m, k_log, useful_bits, None, None);
    dispatch(n_jobs, &job)
}

/// Sequential reference fill (the test oracle for the dispatched shapes).
#[cfg(test)]
fn fill_deferred_lincheck_stripe(
    z_packed: &[F128],
    z_lincheck: &mut [u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
) {
    let completed = fill_deferred_lincheck_stripe_with_dispatch(
        z_packed,
        z_lincheck,
        m,
        k_log,
        useful_bits,
        |n_jobs, job| {
            for i in 0..n_jobs {
                job(i);
            }
            true
        },
    );
    debug_assert!(completed);
}

#[cfg(test)]
mod deferred_stripe_tests {
    use super::*;

    #[test]
    fn deferred_fill_modes_match_canonical_packed_stripe() {
        const M: usize = 20;
        const K_LOG: usize = 8;
        let z: Vec<F128> = (0..1usize << (M - 7))
            .map(|i| {
                let x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                F128::new(x ^ x.rotate_left(17), (!x).rotate_right(11))
            })
            .collect();
        let expected = pack_z_lincheck_from_packed(&z, M, K_LOG);
        let mut actual = vec![0xa5; expected.len()];
        fill_deferred_lincheck_stripe(&z, &mut actual, M, K_LOG, 1 << K_LOG);
        assert_eq!(actual, expected);

        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut helper_actual = vec![0xa5; expected.len()];
        assert!(fill_deferred_lincheck_stripe_with_dispatch(
            &z,
            &mut helper_actual,
            M,
            K_LOG,
            1 << K_LOG,
            |n_jobs, job| flock_core::epool::run_chunks_with_helper_only(
                n_jobs,
                4,
                job,
                Some(&helper),
            ),
        ));
        assert_eq!(helper_actual, expected);
    }

    /// Fixed-input root stripe (176 rows) at the ranked block width: the fill
    /// flags no mismatch on ranked `block_len`/`flags`, flags one when any
    /// instance differs, and the 177-row fallback run over the same buffer
    /// afterwards yields exactly the 177-row stripe.
    #[test]
    fn fixed_input_stripe_flags_mismatch_and_fallback_is_exact() {
        use crate::r1cs_hashes::blake3 as b3;
        const K_LOG: usize = 14;
        const M: usize = K_LOG + 3; // one group of eight instances
        let full_rows = b3::root_stripe_rows().expect("root stripe on by default");
        let fixed_rows = b3::fixed_input_root_stripe_rows().expect("fixed-input stripe on by default");
        assert_eq!((full_rows.len(), fixed_rows.len()), (177, 176));
        let build = |bad_lane: Option<(usize, u32, u32)>| -> Vec<F128> {
            let mut words: Vec<u64> = Vec::with_capacity(1 << (M - 6));
            for lane in 0..8u32 {
                let cv: [u32; 8] = std::array::from_fn(|i| (lane + 1).wrapping_mul(0x9e37_79b9) ^ i as u32);
                let msg: [u32; 16] = std::array::from_fn(|i| (lane * 16 + i as u32).wrapping_mul(0x85eb_ca6b));
                let (bl, fl) = match bad_lane {
                    Some((l, bl, fl)) if l == lane as usize => (bl, fl),
                    _ => (b3::FIXED_BLOCK_LEN, b3::FIXED_FLAGS),
                };
                words.extend(b3::test_block_witness_z(&cv, &msg, u64::from(lane) << 33, bl, fl));
            }
            words.chunks(2).map(|w| F128::new(w[0], w[1])).collect()
        };
        let fill = |z: &[F128], rows, flag: Option<&std::sync::atomic::AtomicBool>, stripe: &mut [u8]| {
            let (n_jobs, job) =
                deferred_lincheck_stripe_job(z, stripe, M, K_LOG, 15_409, Some(rows), flag);
            for j in 0..n_jobs {
                job(j);
            }
        };
        let cases = [
            None,
            Some((0, 63, b3::FIXED_FLAGS)),
            Some((5, b3::FIXED_BLOCK_LEN, 10)),
            Some((7, b3::FIXED_BLOCK_LEN | (1 << 31), b3::FIXED_FLAGS)),
        ];
        for bad in cases {
            let z = build(bad);
            let mut expected = vec![0xa5u8; 1 << (M - 3)];
            fill(&z, full_rows, None, &mut expected);
            let flag = std::sync::atomic::AtomicBool::new(false);
            let mut stripe = vec![0xa5u8; 1 << (M - 3)];
            fill(&z, fixed_rows, Some(&flag), &mut stripe);
            let mismatch = flag.load(std::sync::atomic::Ordering::Acquire);
            assert_eq!(mismatch, bad.is_some(), "case {bad:?}");
            fill(&z, full_rows, None, &mut stripe);
            assert_eq!(stripe, expected, "fallback must equal the 177-row stripe ({bad:?})");
        }
    }

    /// AB-producer stripe: units produced in queue order (pairs on one
    /// thread), out of order (every group falls back to the fill job), and
    /// on a 4-thread pool with even-aligned jobs, for both root-row sets; the
    /// bytes equal the incumbent fill from z = a AND b and the fixed-input
    /// mismatch flag agrees with the incumbent fill's.
    #[test]
    fn ab_stripe_emitter_matches_incumbent_fill() {
        use crate::r1cs_hashes::blake3 as b3;
        const K_LOG: usize = 14;
        const M: usize = K_LOG + 3 + 3; // eight groups, sixteen quads
        const UNIT_U64: usize = 4 * AB_STRIPE_WORDS;
        let n_units = 1usize << (M - K_LOG - 2);
        let mut x = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let a: Vec<u64> = (0..n_units * UNIT_U64).map(|_| next()).collect();
        let mut b: Vec<u64> = (0..n_units * UNIT_U64).map(|_| next() | next()).collect();
        // Make every instance of the first four quads carry the ranked
        // block_len/flags (b = all ones, a = z there), the rest random.
        let mut a = a;
        for unit in 0..4 {
            for c in 0..4 {
                for &(word, mask, want) in b3::FIXED_INPUT_CHECKS.iter() {
                    let i = unit * UNIT_U64 + c * AB_STRIPE_WORDS + word;
                    a[i] = (a[i] & !mask) | want;
                    b[i] |= mask;
                }
            }
        }
        let z_u64: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x & y).collect();
        let z: Vec<F128> = z_u64.chunks(2).map(|w| F128::new(w[0], w[1])).collect();
        for rows in [
            b3::root_stripe_rows().expect("root stripe on by default"),
            b3::fixed_input_root_stripe_rows().expect("fixed-input stripe on by default"),
        ] {
            let fixed = rows.len() == 176;
            let ref_flag = std::sync::atomic::AtomicBool::new(false);
            let mut expected = vec![0xa5u8; 1 << (M - 3)];
            {
                let (n_jobs, job) = deferred_lincheck_stripe_job(
                    &z, &mut expected, M, K_LOG, 15_409, Some(rows), fixed.then_some(&ref_flag),
                );
                (0..n_jobs).for_each(&job);
            }
            let orders: [Vec<usize>; 3] = [
                (0..n_units).collect(),
                (0..n_units).map(|u| u ^ 1).collect(),
                vec![],
            ];
            for (case, order) in orders.iter().enumerate() {
                let flag = std::sync::atomic::AtomicBool::new(false);
                let mut stripe = vec![0xa5u8; 1 << (M - 3)];
                let (stripe_ptr, stripe_len) = (stripe.as_mut_ptr(), stripe.len());
                let (n_jobs, job) = deferred_lincheck_stripe_job(
                    &z, &mut stripe, M, K_LOG, 15_409, Some(rows), fixed.then_some(&flag),
                );
                let emitter = AbStripeEmitter::new(
                    flock_core::epool::SyncPtr(stripe_ptr),
                    stripe_len,
                    1 << K_LOG,
                    rows,
                    fixed,
                    &flag,
                );
                let produce = |unit: usize| {
                    let mut sa = a[unit * UNIT_U64..(unit + 1) * UNIT_U64].to_vec();
                    let mut sb = b[unit * UNIT_U64..(unit + 1) * UNIT_U64].to_vec();
                    emitter.emit(unit, sa.as_mut_ptr().cast(), sb.as_mut_ptr().cast());
                };
                if case < 2 {
                    order.iter().for_each(|&u| produce(u));
                } else {
                    // Two-unit jobs on a 4-thread pool, as the fused queue.
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
                    pool.install(|| {
                        use rayon::prelude::*;
                        (0..n_units / 2).into_par_iter().for_each(|j| {
                            produce(2 * j);
                            produce(2 * j + 1);
                        })
                    });
                }
                let missing = emitter.finish(&z, &job);
                assert_eq!(missing, if case == 1 { n_units / 2 } else { 0 }, "case {case}");
                let _ = n_jobs;
                assert_eq!(stripe, expected, "rows={} case {case}", rows.len());
                assert_eq!(
                    flag.load(std::sync::atomic::Ordering::Relaxed),
                    ref_flag.load(std::sync::atomic::Ordering::Relaxed),
                    "mismatch flag rows={} case {case}",
                    rows.len()
                );
            }
            if fixed {
                assert!(ref_flag.load(std::sync::atomic::Ordering::Relaxed), "random tail must mismatch");
            }
        }
    }

    /// Single-thread cost of the AB stripe emitter (scratch in L1) and
    /// of the incumbent fill from a large z (ignored timing probe).
    #[test]
    #[ignore = "timing probe"]
    fn ab_stripe_emit_timing() {
        use crate::r1cs_hashes::blake3 as b3;
        const K_LOG: usize = 14;
        const G: usize = 4096;
        const UNIT_U64: usize = 4 * AB_STRIPE_WORDS;
        let rows = b3::fixed_input_root_stripe_rows().unwrap();
        let mut x = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut sa: Vec<u64> = (0..2 * UNIT_U64).map(|_| next()).collect();
        let mut sb: Vec<u64> = (0..2 * UNIT_U64).map(|_| next()).collect();
        let z: Vec<F128> = (0..G * 1024).map(|_| F128::new(next(), next())).collect();
        let mut stripe = vec![0u8; G << K_LOG];
        let flag = std::sync::atomic::AtomicBool::new(false);
        for round in 0..5 {
            {
                let e = AbStripeEmitter::new(
                    flock_core::epool::SyncPtr(stripe.as_mut_ptr()),
                    stripe.len(),
                    1 << K_LOG,
                    rows,
                    true,
                    &flag,
                );
                let t = std::time::Instant::now();
                for g in 0..G {
                    e.emit(2 * g, sa.as_mut_ptr().cast(), sb.as_mut_ptr().cast());
                    e.emit(2 * g + 1, sa[UNIT_U64..].as_mut_ptr().cast(), sb[UNIT_U64..].as_mut_ptr().cast());
                }
                let ns = t.elapsed().as_nanos() as f64 / G as f64;
                eprintln!("round {round} emit: {ns:.0} ns/group");
            }
            let t = std::time::Instant::now();
            for g in 0..G {
                let dst = &mut stripe[g << K_LOG..(g + 1) << K_LOG];
                fill_root_lincheck_stripe_group(&z, dst, g, 1024, AB_STRIPE_WORDS, rows);
            }
            let ns = t.elapsed().as_nanos() as f64 / G as f64;
            eprintln!("round {round} incumbent fill from DRAM z: {ns:.0} ns/group");
            let t = std::time::Instant::now();
            for g in 0..G {
                let dst = &mut stripe[g << K_LOG..(g + 1) << K_LOG];
                fill_root_lincheck_stripe_group(&z[..1024], dst, 0, 1024, AB_STRIPE_WORDS, rows);
            }
            let ns = t.elapsed().as_nanos() as f64 / G as f64;
            eprintln!("round {round} incumbent fill from L1 z: {ns:.0} ns/group");
        }
    }

    /// Slow efficiency cores: the helper-only broadcast runs on a pool whose
    /// workers sleep before every chunk (a test-side delay, only in this test),
    /// while main-pool workers help drain the same queue as the AB arm does in
    /// production. The bytes must equal the canonical stripe, every job must
    /// complete exactly once, and the main pool must have claimed jobs.
    #[test]
    fn deferred_fill_help_drain_with_slow_helper_matches_canonical_stripe() {
        const M: usize = 20;
        const K_LOG: usize = 8;
        let z: Vec<F128> = (0..1usize << (M - 7))
            .map(|i| {
                let x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                F128::new(x ^ x.rotate_left(17), (!x).rotate_right(11))
            })
            .collect();
        let expected = pack_z_lincheck_from_packed(&z, M, K_LOG);
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("slow-ecore-test-{i}"))
            .build()
            .unwrap();
        let main = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for delay_us in [0u64, 20_000] {
            let mut stripe = vec![0xa5u8; expected.len()];
            let (n_jobs, job) =
                deferred_lincheck_stripe_job(&z, &mut stripe, M, K_LOG, 1 << K_LOG, None, None);
            assert!(n_jobs >= 4, "test shape needs several jobs, got {n_jobs}");
            let runs: Vec<std::sync::atomic::AtomicUsize> =
                (0..n_jobs).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect();
            let helper_started = std::sync::atomic::AtomicBool::new(false);
            let delayed = |i: usize| {
                let on_slow_helper = std::thread::current()
                    .name()
                    .is_some_and(|name| name.starts_with("slow-ecore-test-"));
                if on_slow_helper {
                    helper_started.store(true, std::sync::atomic::Ordering::Release);
                    if delay_us > 0 {
                        std::thread::sleep(std::time::Duration::from_micros(delay_us));
                    }
                }
                runs[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                job(i);
            };
            let queue = flock_core::epool::HelpQueue::new(n_jobs, &delayed);
            let ready = std::sync::atomic::AtomicBool::new(false);
            std::thread::scope(|scope| {
                let owner = scope.spawn(|| {
                    assert!(flock_core::epool::run_queue_with_helper_only(
                        &queue,
                        4,
                        Some(&helper),
                    ));
                    queue.wait_done();
                    ready.store(true, std::sync::atomic::Ordering::Release);
                });
                // Main-pool help, as the AB queue's workers run it. With the
                // delay, start helping only once a slow helper holds a job,
                // so the two sides really interleave.
                if delay_us > 0 {
                    while !helper_started.load(std::sync::atomic::Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                }
                main.install(|| {
                    use rayon::prelude::*;
                    (0..4).into_par_iter().with_max_len(1).for_each(|_| queue.help());
                });
                owner.join().unwrap();
            });
            assert!(ready.load(std::sync::atomic::Ordering::Acquire));
            assert!(
                runs.iter()
                    .all(|r| r.load(std::sync::atomic::Ordering::Relaxed) == 1),
                "every stripe job runs exactly once (delay {delay_us} us)"
            );
            if delay_us > 0 {
                assert!(
                    queue.main_claims() > 0,
                    "main-pool helpers must claim stripe jobs from a slow helper pool"
                );
                assert!(
                    queue.main_claims() < n_jobs,
                    "the slow helper pool must also have run jobs"
                );
            }
            assert_eq!(stripe, expected, "delay {delay_us} us");
        }
    }
}

/// Build the witness commitment and the challenge-independent half of
/// zerocheck round 1 on the same fixed Rayon pool. A/B are only borrowed:
/// their original packed values remain live for zerocheck round 2.
/// Commit-tail fill hook: runs in the commit arm's closure immediately after
/// the GPU graph completes (arm A of the `rayon::join` executes on the
/// calling thread, and the Merkle root is final the instant `pcs::commit*`
/// returns), while the AB precompute arm is typically still running. See
/// `zerocheck::stage_commit_tail_fill`.
type CommitTailFillHook<'a> = Box<dyn FnOnce(&Commitment) + Send + 'a>;

/// Build the commit-tail-fill hook for one prove: `None` when the fill is
/// killed (`FLOCK_NO_COMMIT_TAIL_FILL=1`), off the ranked lincheck-C-reuse
/// shape, or the challenger cannot fork. The hook — run at commit-graph
/// completion inside the commit arm — acquires `stripe_ready`, reconstructs
/// the stripe view, and stages the round-one C fold's GPU prefix (see
/// `zerocheck::stage_commit_tail_fill`).
fn make_commit_tail_fill_hook<'a, Ch: Challenger>(
    challenger: &Ch,
    r1cs: &'a BlockR1cs,
    padding: &'a zerocheck::PaddingSpec,
    stripe_ready: &'a std::sync::atomic::AtomicBool,
    stripe_ptr: usize,
    stripe_len: usize,
) -> Option<CommitTailFillHook<'a>>
where
    Ch: 'a,
{
    if !flock_core::gpu_commit::commit_tail_fill_enabled() || !ranked_lincheck_c_reuse_enabled(r1cs)
    {
        return None;
    }
    let forked = challenger.fork()?;
    Some(Box::new(move |commitment: &Commitment| {
        if !stripe_ready.load(std::sync::atomic::Ordering::Acquire) {
            // The stripe fill lost the race to the graph; staging would
            // read torn bytes. Skip — the incumbent zerocheck-entry
            // submit runs.
            if std::env::var_os("FLOCK_ZC_TIMING").is_some() {
                eprintln!("[commit-tail-fill] skipped: stripe not ready");
            }
            return;
        }
        // SAFETY: the acquire above pairs with the fill worker's
        // release-store, which follows `HelpQueue::wait_done`'s acquire of
        // every chunk's release completion, i.e. the last stripe write by
        // any thread; the buffer is not written again until after
        // zerocheck consumed the stash.
        let stripe: &[u8] =
            unsafe { std::slice::from_raw_parts(stripe_ptr as *const u8, stripe_len) };
        flock_core::zerocheck::stage_commit_tail_fill(forked, r1cs, commitment, stripe, padding);
    }))
}

fn commit_with_round1_ab_precompute(
    z_packed: &[F128],
    a_packed_f128: &[F128],
    b_packed_f128: &[F128],
    pcs_params: &PcsParams,
    padding: &zerocheck::PaddingSpec,
    commit_codeword: CommitCodeword,
    tail_fill: Option<CommitTailFillHook<'_>>,
    identity_prefix_contract: Option<Round1PinnedOnePrefixStatement>,
    stripe_help: Option<&(dyn Fn() + Sync)>,
    ab_producer: Option<zerocheck::univariate_skip_optimized::Round1AbProducer<'_>>,
    after_ab: Option<&(dyn Fn() + Sync)>,
) -> (
    (Commitment, pcs::ProverData),
    zerocheck::univariate_skip_optimized::Round1AbInner,
) {
    let as_bytes = |v: &[F128]| -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    };
    let a_packed = as_bytes(a_packed_f128);
    let b_packed = as_bytes(b_packed_f128);
    let k_skip = zerocheck::K_SKIP;
    debug_assert_eq!(k_skip, 6, "ranked protocol fixes k_skip=6");
    let inv_table = flock_core::ntt::InvNttTableByteSingleGf8::cached_standard_k6();

    // `stripe_help` (deferred-stripe proves only): each main-pool worker of
    // the AB queue drains leftover stripe chunks once the AB queue is empty.
    // The AB queue's helper broadcast is queued behind the stripe's on the
    // efficiency-core pool, so this arm cannot finish before the stripe does;
    // helping moves the stripe's remainder onto otherwise idle P-workers.
    let precompute_ab = || {
        if let Some(producer) = ab_producer {
            // Deferred A/B rows: this arm is their only writer, and it has
            // produced every unit by the time it returns (the replay below
            // and round two read complete buffers).
            // SAFETY: both buffers are live, `2^m / 8` bytes long, and
            // written only by the producer during this call.
            return unsafe {
                zerocheck::univariate_skip_optimized::precompute_round1_ab_inner_packed_padded_produced(
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    pcs_params.m,
                    k_skip,
                    inv_table,
                    padding,
                    identity_prefix_contract,
                    stripe_help,
                    producer,
                )
            };
        }
        zerocheck::univariate_skip_optimized::precompute_round1_ab_inner_packed_padded_with_tail(
            a_packed,
            b_packed,
            pcs_params.m,
            k_skip,
            inv_table,
            padding,
            identity_prefix_contract,
            stripe_help,
        )
    };
    let precompute_ab_replay = || {
        zerocheck::univariate_skip_optimized::precompute_round1_ab_inner_packed_padded_for_exact_tune(
            a_packed,
            b_packed,
            pcs_params.m,
            k_skip,
            inv_table,
            padding,
            identity_prefix_contract,
        )
    };
    // `Blake3Setup::prove_fast` issues this ticket before call-zero witness
    // generation. A valid cache hit may satisfy it inside the commit arm;
    // otherwise the post-join callback claims it and replays this exact A/B
    // closure beside the broad split sweep.
    let run_ranked_exact_tune = flock_core::gpu_commit::ranked_exact_contention_tune_pending();

    // The BLAKE3 padding rows force every block's tail words to zero, which
    // in the SoA codeword is a static all-zero pattern on the top lanes of
    // every odd position. Publish it for the duration of this commitment so
    // the NTT can drop those butterflies; the guard restores the previous
    // value (recursive PCS commits run at eight lanes and never match).
    let _zero_lane_skip = flock_core::ntt::additive_ntt_f128::ZeroOddTailLanes::scope(
        pcs_params.num_ntts(),
        flock_core::ntt::additive_ntt_f128::ZeroOddTailLanes::lanes_for_padding(
            pcs_params.num_ntts(),
            padding.k_log,
            padding.useful_bits_per_block,
        ),
    );

    flock_core::tmark::mark("commit join start");
    let result = rayon::join(
        || {
            flock_core::tmark::mark("commit arm start");
            let pre = match commit_codeword {
                CommitCodeword::Allocate => pcs::commit(z_packed, pcs_params),
                CommitCodeword::NeedsReplication(buf) => {
                    pcs::commit_into(z_packed, pcs_params, buf)
                }
                CommitCodeword::Preinitialized(buf) => {
                    pcs::commit_preinitialized(z_packed, buf, pcs_params)
                }
                CommitCodeword::StreamedFirstPass(buf, stream) => {
                    pcs::commit_from_streamed_first_pass(z_packed, buf, pcs_params, stream)
                }
            };
            // Graph complete, root final — the AB arm is (typically) still
            // running. Fire the tail-fill hook here so its staged GPU work
            // executes in the arm-tail idle.
            flock_core::tmark::mark("commit arm root");
            if let Some(hook) = tail_fill {
                hook(&pre.0);
            }
            flock_core::tmark::mark("commit arm end");
            pre
        },
        || {
            flock_core::tmark::mark("ab arm start");
            let t = std::time::Instant::now();
            let r = precompute_ab();
            // Inside the arm, so its completion is ordered before the
            // tail-fill hook can observe a published stripe.
            if let Some(after_ab) = after_ab {
                after_ab();
            }
            let wall_ms = t.elapsed().as_secs_f64() * 1e3;
            // The hybrid-commit warmup sweep sizes its contention emulation
            // from this arm's measured wall (an Instant read is free; the
            // store is one relaxed atomic per prove).
            flock_core::tmark::mark("ab arm end");
            flock_core::gpu_commit::note_precompute_branch_wall_ms(wall_ms);
            if std::env::var_os("FLOCK_PHASE_TIMING").is_some() {
                eprintln!("[phase-timing] ab-precompute branch wall: {wall_ms:.2} ms");
            }
            r
        },
    );

    flock_core::tmark::mark("commit join end");
    if run_ranked_exact_tune {
        flock_core::gpu_commit::retune_ranked_hybrid_with_exact_contention(
            pcs_params,
            &result.0.1.codeword,
            &result.0.1.merkle_tree,
            || {
                let replayed = precompute_ab_replay();
                std::hint::black_box(&replayed);
            },
        );
    }
    result
}

/// Run commit → bind → zerocheck → lincheck and build the base claims, stopping
/// just before the PCS open. See [`ProveCore`].
pub fn prove_fast_core<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> ProveCore {
    prove_fast_core_with_commit_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        LincheckStripeInput::Ready(z_packed_lincheck),
        lincheck_circuit,
        CommitCodeword::Allocate,
        None,
        None,
        challenger,
    )
}

/// [`prove_fast_core`] with an optional pre-faulted codeword buffer (see
/// [`pcs::prefault_codeword_during`]). When `Some`, the commit reuses it via
/// [`pcs::commit_into`] instead of allocating — the alloc was already done,
/// overlapped with witness generation. When `None`, behaves exactly like
/// [`prove_fast_core`] (commit allocates inline).
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_core_with_codeword<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> ProveCore {
    let commit_codeword = match prefaulted_codeword {
        Some(codeword) => CommitCodeword::NeedsReplication(codeword),
        None => CommitCodeword::Allocate,
    };
    prove_fast_core_with_commit_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        LincheckStripeInput::Ready(z_packed_lincheck),
        lincheck_circuit,
        commit_codeword,
        None,
        None,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_fast_core_with_commit_codeword<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: LincheckStripeInput,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    commit_codeword: CommitCodeword,
    identity_prefix_witness: Option<Blake3IdentityPrefixWitness>,
    ab_producer: Option<zerocheck::univariate_skip_optimized::Round1AbProducer<'_>>,
    challenger: &mut Ch,
) -> ProveCore {
    let padding = r1cs.padding_spec();
    let identity_prefix_contract = ranked_identity_prefix_omission_contract(
        r1cs, identity_prefix_witness.as_ref(), &z_packed, &a_packed_f128, &b_packed_f128,
    );
    let phase_timing = std::env::var_os("FLOCK_PHASE_TIMING").is_some();
    let run_commit = |tail_fill: Option<CommitTailFillHook<'_>>,
                      stripe_help: Option<&(dyn Fn() + Sync)>,
                      producer: Option<zerocheck::univariate_skip_optimized::Round1AbProducer<'_>>,
                      after_ab: Option<&(dyn Fn() + Sync)>| {
        let cpu0 = phase_timing.then(process_cpu_ms);
        let helper0 = flock_core::epool::helper_chunks_claimed();
        let t_commit = std::time::Instant::now();
        let ((commitment, prover_data), ab_inner) = commit_with_round1_ab_precompute(
            &z_packed,
            &a_packed_f128,
            &b_packed_f128,
            pcs_params,
            &padding,
            commit_codeword,
            tail_fill,
            identity_prefix_contract,
            stripe_help,
            producer,
            after_ab,
        );
        if phase_timing {
            let wall = t_commit.elapsed().as_secs_f64() * 1e3;
            let cpu = process_cpu_ms() - cpu0.unwrap_or(0.0);
            eprintln!(
                "[phase-timing] commit+ab-precompute: {wall:.2} ms cpu={cpu:.1} util={:.1}",
                cpu / wall
            );
            eprintln!(
                "[phase-timing] epool helper chunks in commit: {}",
                flock_core::epool::helper_chunks_claimed() - helper0
            );
        }
        (commitment, prover_data, ab_inner)
    };

    // Publish the stripe before zerocheck: the ranked identity-C producer now
    // consumes it immediately after the commitment challenge is available.
    // Deferred fill is shorter than commit+AB at the target shape, so moving
    // this join earlier adds no measured tail while eliminating C's 32-bank
    // row-major drain.
    // Root-compacted stripe registration (ranked deferred path only); held
    // until lincheck has consumed the stripe.
    let mut root_stripe_registration: Option<flock_core::lincheck::RootStripeRegistration> =
        None;
    let (pre_zerocheck, z_packed_lincheck) = match z_packed_lincheck {
        LincheckStripeInput::Ready(stripe) => {
            // The stripe is fully materialized, so the tail fill can stage
            // unconditionally at graph completion. Engaging here keeps the
            // warmup prove's split calibration representative of the timed
            // prove's earlier-submit head timing.
            let stripe_ready = std::sync::atomic::AtomicBool::new(true);
            let hook = make_commit_tail_fill_hook(
                &*challenger,
                r1cs,
                &padding,
                &stripe_ready,
                stripe.as_ptr() as usize,
                stripe.len(),
            );
            (run_commit(hook, None, ab_producer, None), stripe)
        }
        LincheckStripeInput::DeferredRanked => {
            assert_eq!(r1cs.m, 32);
            assert_eq!(r1cs.k_log, 14);
            assert_eq!(r1cs.useful_bits, 15_409);
            let (m, k_log, useful_bits) = (r1cs.m, r1cs.k_log, r1cs.useful_bits);
            let mut stripe = flock_core::scratch::take_u8(1usize << (m - 3));
            // Root stripe (`FLOCK_NO_ROOT_STRIPE=1` restores the full 241-row
            // stripe): store only the 11,313 root columns; every stripe fold
            // consults the registration and expands its result exactly.
            // Default: the fixed-input stripe (176 rows; block_len/flags
            // omitted, `FLOCK_NO_FIXED_INPUT_ROOTS=1` keeps them), verified on
            // every instance by the fill; on any mismatch the stripe thread
            // refills with the 177-row stripe before publishing it.
            let full_root_rows = crate::r1cs_hashes::blake3::root_stripe_rows();
            let fixed_root_rows = crate::r1cs_hashes::blake3::fixed_input_root_stripe_rows();
            let root_rows = fixed_root_rows.or(full_root_rows);
            let input_mismatch = std::sync::atomic::AtomicBool::new(false);
            if root_rows.is_some() {
                root_stripe_registration = Some(flock_core::lincheck::register_root_stripe(
                    &stripe,
                    if fixed_root_rows.is_some() {
                        crate::r1cs_hashes::blake3::fixed_input_root_stripe_descriptor()
                    } else {
                        crate::r1cs_hashes::blake3::root_stripe_descriptor()
                    },
                ));
            }
            // E4 (all four E-workers), revisited 2026-08-05. The E3 default
            // was tuned in the GPU-bound-window regime (welttowelt `4e30884`
            // +0.95% ranked over E2; "E4 steals bandwidth from commit" was
            // measured when the commit GRAPH bound the join). Post-byte16 the
            // window is bound by the AB precompute ARM (arm 58.2 ms ≈ window
            // 58.3 ms; GPU graph 41–53 ms, 0.00 ms host wait), and the QS5
            // hetero precompute queue lets E-workers continue onto arm chunks
            // the moment the stripe broadcast returns. A wider stripe both
            // finishes the fill earlier (~40 ms vs ~52 ms) and releases the
            // E-cluster to the arm-bound queue sooner; the bandwidth-theft
            // argument no longer applies to a CPU-compute-bound binder. The
            // 1..=4 override stays for controlled same-binary diagnostics
            // (E3 = the prior shipped behavior).
            const DEFER_STRIPE_EPOOL_THREADS_DEFAULT: usize = 4;
            let epool_workers = std::env::var_os("FLOCK_DEFER_STRIPE_EPOOL_THREADS")
                .and_then(|value| value.to_str()?.parse::<usize>().ok())
                .filter(|workers| (1..=4).contains(workers))
                .unwrap_or(DEFER_STRIPE_EPOOL_THREADS_DEFAULT);
            // Commit-tail fill (`FLOCK_NO_COMMIT_TAIL_FILL=1` kills): the
            // hook stages only after acquiring `stripe_ready`, whose paired
            // release follows the last stripe write by any thread (see
            // `HelpQueue::wait_done` below).
            let stripe_ready = std::sync::atomic::AtomicBool::new(false);
            let stripe_ptr = stripe.as_ptr() as usize;
            let stripe_len = stripe.len();
            let tail_fill_hook = make_commit_tail_fill_hook(
                &*challenger,
                r1cs,
                &padding,
                &stripe_ready,
                stripe_ptr,
                stripe_len,
            );
            // One shared chunk queue for the stripe. The E-workers own it
            // (helper-only broadcast, as before); main-pool workers that
            // finish the AB precompute queue help drain what is left, so a
            // slow efficiency cluster delays the commit window by at most the
            // chunks it already holds, not the whole stripe remainder. Each
            // job index is claimed exactly once from one cursor and writes
            // only its own groups, so the bytes do not depend on who ran it.
            // `stripe` is not touched again until after the scope below.
            let (n_stripe_jobs, stripe_job_fn) =
                deferred_lincheck_stripe_job(
                    &z_packed,
                    &mut stripe,
                    m,
                    k_log,
                    useful_bits,
                    root_rows,
                    fixed_root_rows.map(|_| &input_mismatch),
                );
            // Exact fallback for the fixed-input stripe (never taken on ranked
            // inputs): the 177-row fill, run only after every first-pass chunk
            // has completed, so the two jobs never touch the stripe at once.
            let fallback_job = fixed_root_rows.and(full_root_rows).map(|rows| {
                deferred_lincheck_stripe_job(
                    &z_packed,
                    &mut stripe,
                    m,
                    k_log,
                    useful_bits,
                    Some(rows),
                    None,
                )
            });
            let stripe_queue = flock_core::epool::HelpQueue::new(n_stripe_jobs, &stripe_job_fn);
            let stripe_help = || stripe_queue.help();
            // Stripe from the AB producer (`FLOCK_NO_STRIPE_IN_AB` kills): the
            // fused round-one AB queue holds each quad's complete a and b
            // blocks in L1, and z = a AND b word for word (C = I), so the
            // queue emits the root stripe of every 8-block group from the
            // two quads it produces back to back. No E-core stripe arm and no
            // 512 MiB re-read of z inside the commit window.
            let stripe_in_ab = root_rows.is_some()
                && ab_producer.is_some()
                && zerocheck::univariate_skip_optimized::zc_ab_producer_fused_active()
                && std::env::var_os("FLOCK_NO_STRIPE_IN_AB").is_none();
            let pre = if stripe_in_ab {
                let base = ab_producer.expect("checked above");
                let emitter = AbStripeEmitter::new(
                    flock_core::epool::SyncPtr(stripe_ptr as *mut u8),
                    stripe_len,
                    1usize << k_log,
                    root_rows.expect("checked above"),
                    fixed_root_rows.is_some(),
                    &input_mismatch,
                );
                // `primed`: the scratch pair has been filled before by this
                // producer, which may then leave its per-block constants in
                // place.  Either way the pair holds the unit's complete a and
                // b blocks when `emit` reads it.
                let produce = |unit: usize, a: *mut u8, b: *mut u8, primed: bool| {
                    (base.produce)(unit, a, b, primed);
                    emitter.emit(unit, a, b);
                };
                let finish = || {
                    let t = std::time::Instant::now();
                    let missing = emitter.finish(&z_packed, &stripe_job_fn);
                    let mismatch = input_mismatch.load(std::sync::atomic::Ordering::Acquire);
                    if mismatch {
                        if let Some((n_jobs, job)) = &fallback_job {
                            use rayon::prelude::*;
                            (0..*n_jobs).into_par_iter().for_each(|j| job(j));
                            let replaced = flock_core::lincheck::replace_root_stripe(
                                stripe_ptr,
                                stripe_len,
                                crate::r1cs_hashes::blake3::root_stripe_descriptor(),
                            );
                            assert!(replaced, "root stripe registration vanished");
                        }
                    }
                    stripe_ready.store(true, std::sync::atomic::Ordering::Release);
                    if phase_timing {
                        eprintln!(
                            "[phase-timing] ab-stripe finish: {:.2} ms fallback_groups={missing} mismatch={mismatch}",
                            t.elapsed().as_secs_f64() * 1e3
                        );
                    }
                };
                run_commit(
                    tail_fill_hook,
                    None,
                    Some(zerocheck::univariate_skip_optimized::Round1AbProducer {
                        unit_bytes: base.unit_bytes,
                        produce: &produce,
                    }),
                    Some(&finish),
                )
            } else {
                std::thread::scope(|scope| {
                    let stripe_ready = &stripe_ready;
                    let stripe_queue = &stripe_queue;
                    let input_mismatch = &input_mismatch;
                    let fallback_job = &fallback_job;
                    let stripe_job = scope.spawn(move || {
                        let started = std::time::Instant::now();
                        let filled_on_epool =
                            flock_core::epool::run_helper_only_queue(stripe_queue, epool_workers);
                        if !filled_on_epool {
                            // No helper pool: drain the same queue here (main-pool
                            // help may already own some chunks).
                            stripe_queue.drain_inline();
                        }
                        // Chunks claimed by main-pool helpers may still be running
                        // after the broadcast returns; wait for every completion
                        // (Release per chunk, Acquire here) before publishing.
                        stripe_queue.wait_done();
                        if input_mismatch.load(std::sync::atomic::Ordering::Acquire) {
                            if let Some((n_jobs, job)) = fallback_job {
                                for j in 0..*n_jobs {
                                    job(j);
                                }
                                let replaced = flock_core::lincheck::replace_root_stripe(
                                    stripe_ptr,
                                    stripe_len,
                                    crate::r1cs_hashes::blake3::root_stripe_descriptor(),
                                );
                                assert!(replaced, "root stripe registration vanished");
                            }
                        }
                        stripe_ready.store(true, std::sync::atomic::Ordering::Release);
                        if phase_timing {
                            eprintln!(
                                "[phase-timing] deferred lincheck stripe: {:.2} ms mode={} workers={} main_help_jobs={}",
                                started.elapsed().as_secs_f64() * 1e3,
                                if filled_on_epool {
                                    "epool"
                                } else {
                                    "sequential"
                                },
                                if filled_on_epool { epool_workers } else { 0 },
                                stripe_queue.main_claims(),
                            );
                        }
                    });
                    let pre = run_commit(tail_fill_hook, Some(&stripe_help), ab_producer, None);
                    // The calling thread is idle until the join; help first (one
                    // relaxed load when the queue is already exhausted).
                    stripe_queue.help();
                    let join_started = std::time::Instant::now();
                    // Spin-family completion, prover side: this is the last timed
                    // production join in the prove path (the gpu_commit.rs joins
                    // already poll command-buffer status). The deferred fill is
                    // scheduled to finish before commit+AB at the target shape,
                    // so the worker is usually ALREADY complete when this join
                    // runs — yet a blocking scoped-thread join still pays the
                    // park + completion-wake tail whenever the worker races the
                    // join call. Poll `is_finished` first (zero cost when done),
                    // then yield for a bounded budget (yield, not spin: the
                    // worker is a sibling CPU thread and must keep its core —
                    // same rationale as the warmup AB-wait), then degrade to the
                    // exact incumbent blocking join. Byte-identical either way:
                    // the join consumes the same completed thread.
                    if !stripe_job.is_finished() {
                        let spin_deadline =
                            std::time::Instant::now() + std::time::Duration::from_millis(2);
                        while !stripe_job.is_finished() && std::time::Instant::now() < spin_deadline {
                            std::thread::yield_now();
                        }
                    }
                    stripe_job
                        .join()
                        .expect("deferred ranked lincheck stripe worker panicked");
                    if phase_timing {
                        eprintln!(
                            "[phase-timing] deferred lincheck stripe join tail: {:.2} ms",
                            join_started.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    pre
                })
            };
            (pre, stripe)
        }
    };
    let (commitment, prover_data, ab_inner) = pre_zerocheck;
    bind_statement(challenger, r1cs, &commitment);
    flock_core::tmark::mark("zc start");
    let cpu_zc0 = phase_timing.then(process_cpu_ms);
    let t_zc = std::time::Instant::now();

    // Zerocheck runs on a main-pool worker so its parallel regions start as
    // nested joins (`flock_core::in_pool`). Same work, same transcript order.
    let (zc_proof, zc_claim, s_hat_v_c) = flock_core::in_pool(|| {
        flock_core::tmark::mark("zc body start");
        // Zero-cost &[u8] views of the F128 buffers; c aliases z (C = I).
        let a_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                a_packed_f128.as_ptr() as *const u8,
                a_packed_f128.len() * core::mem::size_of::<F128>(),
            )
        };
        let b_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                b_packed_f128.as_ptr() as *const u8,
                b_packed_f128.len() * core::mem::size_of::<F128>(),
            )
        };
        let c_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                z_packed.as_ptr() as *const u8,
                z_packed.len() * core::mem::size_of::<F128>(),
            )
        };
        if ranked_lincheck_c_reuse_enabled(r1cs) {
            zerocheck::prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab_and_lincheck_c(
                a_packed,
                b_packed,
                c_packed,
                &z_packed_lincheck,
                r1cs.m,
                &padding,
                ab_inner,
                challenger,
            )
        } else {
            zerocheck::prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab(
                a_packed, b_packed, c_packed, r1cs.m, &padding, ab_inner, challenger,
            )
        }
    });
    if phase_timing {
        let wall = t_zc.elapsed().as_secs_f64() * 1e3;
        let cpu = process_cpu_ms() - cpu_zc0.unwrap_or(0.0);
        eprintln!(
            "[phase-timing] zerocheck: {wall:.2} ms cpu={cpu:.1} util={:.1}",
            cpu / wall
        );
    }
    // Nothing downstream reads a/b (zerocheck consumed them in rounds 1–2);
    // recycle the two buffers (2 × 2^(m-3) bytes — 128 MB at m = 29) instead
    // of carrying them through lincheck and the PCS open.
    flock_core::scratch::give_f128(a_packed_f128);
    flock_core::scratch::give_f128(b_packed_f128);

    flock_core::tmark::mark("zc end");
    let cpu_lc0 = phase_timing.then(process_cpu_ms);
    let t_lc = std::time::Instant::now();
    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

    // Capture lincheck's pre-sumcheck z_vec so the PCS open can derive the
    // AB-claim's `s_hat_v` from it (skips fold_1b_rows for AB).
    // Lincheck runs on a main-pool worker (`flock_core::in_pool`), as above.
    flock_core::tmark::mark("lc call");
    let (lc_proof, lc_claim, z_vec_pre) = flock_core::in_pool(|| {
        flock_core::tmark::mark("lc body start");
        lincheck::prove_padded_capture_z_vec(
            &z_packed_lincheck,
            r1cs.m,
            r1cs.k_log,
            r1cs.k_skip,
            r1cs.useful_bits,
            lincheck_circuit,
            &x_ab,
            challenger,
        )
    });
    // The lincheck stripe copy of z is dead from here on; return it to the
    // scratch byte pool before the PCS open (2^(m-3) bytes — 512 MB at
    // m = 32) so the next prove reuses its resident pages.
    flock_core::tmark::mark("lc returned");
    drop(root_stripe_registration);
    flock_core::scratch::give_u8(z_packed_lincheck);

    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };

    // Strided fold of z_vec_pre against the AB-claim suffix's inner-rest tail
    // (everything past prefix0). Byte-identical to `fold_1b_rows` on the AB
    // suffix tensor — see `s_hat_v_from_z_vec`. Skip when k_log < LOG_PACKING
    // (only test setups; real R1CS has k_log >= 16).
    let s_hat_v_ab = precompute_ab_s_hat_v(r1cs, &z_vec_pre, &lc_claim.r_inner_rest[1..]);
    // z_vec_pre only fed s_hat_v_ab; recycle before PCS open residency.
    flock_core::scratch::give_lincheck_eq(z_vec_pre);
    if phase_timing {
        let wall = t_lc.elapsed().as_secs_f64() * 1e3;
        let cpu = process_cpu_ms() - cpu_lc0.unwrap_or(0.0);
        eprintln!(
            "[phase-timing] lincheck+s_hat_v: {wall:.2} ms cpu={cpu:.1} util={:.1}",
            cpu / wall
        );
    }

    ProveCore {
        zc_proof,
        lc_proof,
        ab,
        c,
        commitment,
        prover_data,
        z_packed,
        s_hat_v_ab,
        s_hat_v_c,
    }
}

/// Process CPU time (user+system) in ms, for FLOCK_PHASE_TIMING per-phase
/// parallelism diagnostics. Diagnostics-only; returns 0.0 off macOS.
pub(crate) fn process_cpu_ms() -> f64 {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct Timeval {
            sec: i64,
            usec: i32,
        }
        #[repr(C)]
        struct Rusage {
            utime: Timeval,
            stime: Timeval,
            other: [i64; 14],
        }
        unsafe extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        let mut ru = Rusage {
            utime: Timeval { sec: 0, usec: 0 },
            stime: Timeval { sec: 0, usec: 0 },
            other: [0; 14],
        };
        if unsafe { getrusage(0, &mut ru) } == 0 {
            return (ru.utime.sec + ru.stime.sec) as f64 * 1e3
                + (ru.utime.usec + ru.stime.usec) as f64 * 1e-3;
        }
    }
    0.0
}

/// Per-phase wall-clock timings (seconds) of the Ligerito fast prover, for
/// benchmark cost breakdowns. See [`prove_fast_ligerito_timed`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ProvePhaseTimings {
    /// Witness generation. Filled by the per-setup `prove_fast_timed` wrappers
    /// (not by [`prove_fast_ligerito_timed`], which takes the witness as input).
    pub witness_s: f64,
    pub commit_s: f64,
    pub zerocheck_s: f64,
    /// Lincheck prove + the small post-lincheck base-claim / `s_hat_v` setup.
    pub lincheck_s: f64,
    /// The real Ligerito recursive PCS open (`open_claims_…_ligerito`).
    pub open_s: f64,
}

/// [`prove_fast_ligerito_from_witness`] with per-phase timers. Inlines the same
/// calls in the same order as `prove_fast_core_with_codeword` + the Ligerito
/// open, wrapping each phase in an `Instant`, so the returned
/// [`ProvePhaseTimings`] decompose the *real* Ligerito prover --- including its
/// recursive opening. Kept in lockstep
/// with `prove_fast_ligerito_from_witness`; benchmark-only.
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_timed<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim, ProvePhaseTimings) {
    let commit_codeword = match prefaulted_codeword {
        Some(codeword) => CommitCodeword::NeedsReplication(codeword),
        None => CommitCodeword::Allocate,
    };
    prove_fast_ligerito_timed_with_commit_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        z_packed_lincheck,
        lincheck_circuit,
        commit_codeword,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_fast_ligerito_timed_from_preinitialized_codeword<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    codeword: Vec<F128>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim, ProvePhaseTimings) {
    prove_fast_ligerito_timed_with_commit_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        z_packed_lincheck,
        lincheck_circuit,
        CommitCodeword::Preinitialized(codeword),
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_fast_ligerito_timed_with_commit_codeword<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    z_packed_lincheck: Vec<u8>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    commit_codeword: CommitCodeword,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim, ProvePhaseTimings) {
    use std::time::Instant;
    let mut t = ProvePhaseTimings::default();

    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito default config; bump m for tiny instances");

    let padding = r1cs.padding_spec();

    // --- PCS commit + challenge-independent zerocheck AB preprocessing ---
    let t0 = Instant::now();
    let ((commitment, prover_data), ab_inner) = commit_with_round1_ab_precompute(
        &z_packed,
        &a_packed_f128,
        &b_packed_f128,
        pcs_params,
        &padding,
        commit_codeword,
        None,
        None,
        None,
        None,
        None,
    );
    t.commit_s = t0.elapsed().as_secs_f64();
    bind_statement(challenger, r1cs, &commitment);

    // --- zerocheck ---
    let t0 = Instant::now();
    let (zc_proof, zc_claim, s_hat_v_c) = {
        let a_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                a_packed_f128.as_ptr() as *const u8,
                a_packed_f128.len() * core::mem::size_of::<F128>(),
            )
        };
        let b_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                b_packed_f128.as_ptr() as *const u8,
                b_packed_f128.len() * core::mem::size_of::<F128>(),
            )
        };
        let c_packed: &[u8] = unsafe {
            std::slice::from_raw_parts(
                z_packed.as_ptr() as *const u8,
                z_packed.len() * core::mem::size_of::<F128>(),
            )
        };
        zerocheck::prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab(
            a_packed, b_packed, c_packed, r1cs.m, &padding, ab_inner, challenger,
        )
    };
    t.zerocheck_s = t0.elapsed().as_secs_f64();
    flock_core::scratch::give_f128(a_packed_f128);
    flock_core::scratch::give_f128(b_packed_f128);

    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

    // --- lincheck + base-claim / s_hat_v setup ---
    let t0 = Instant::now();
    let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_padded_capture_z_vec(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        lincheck_circuit,
        &x_ab,
        challenger,
    );
    flock_core::scratch::give_u8(z_packed_lincheck);
    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };
    let s_hat_v_ab = precompute_ab_s_hat_v(r1cs, &z_vec_pre, &lc_claim.r_inner_rest[1..]);
    // z_vec_pre only fed s_hat_v_ab; recycle before PCS open residency.
    flock_core::scratch::give_f128(z_vec_pre);
    t.lincheck_s = t0.elapsed().as_secs_f64();

    // --- Ligerito recursive PCS open ---
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = pre_c_slot(r1cs, &s_hat_v_c);
    let t0 = Instant::now();
    let pcs_open = open_claims_with_precomputed_ligerito(
        z_packed,
        &prover_data,
        &commitment,
        &[ab.clone(), c.clone()],
        &[pre_ab, pre_c],
        &padding,
        &lig_config,
        challenger,
    );
    t.open_s = t0.elapsed().as_secs_f64();

    let proof = R1csProofLigerito {
        zerocheck: zc_proof,
        lincheck: lc_proof,
        pcs_open,
    };
    let claim = R1csClaim { ab, c };
    (proof, commitment, claim, t)
}

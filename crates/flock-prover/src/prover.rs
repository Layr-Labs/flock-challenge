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
use flock_core::r1cs::BlockR1cs;
use flock_core::zerocheck;
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
    pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        prover_data,
        commitment,
        &x_refs,
        precomputed_s_hat_v,
        &[],
        padding,
        lig_config,
        challenger,
    )
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
        challenger,
    );

    let padding = r1cs.padding_spec();
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = pre_c_slot(r1cs, &s_hat_v_c);
    let phase_timing = std::env::var_os("FLOCK_PHASE_TIMING").is_some();
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
    let pcs_open = match stash_pool {
        Some(pool) => pool.in_place_scope(|s| {
            s.spawn(|_| {
                crate::proof_io::stash_pre_encoded_prefix(&commitment, &zc_proof, &lc_proof)
            });
            open_claims_with_precomputed_ligerito(
                z_packed,
                &prover_data,
                &commitment,
                &[ab.clone(), c.clone()],
                &[pre_ab, pre_c],
                &padding,
                &lig_config,
                challenger,
            )
        }),
        None => open_claims_with_precomputed_ligerito(
            z_packed,
            &prover_data,
            &commitment,
            &[ab.clone(), c.clone()],
            &[pre_ab, pre_c],
            &padding,
            &lig_config,
            challenger,
        ),
    };
    if phase_timing {
        let wall = t_open.elapsed().as_secs_f64() * 1e3;
        let cpu = process_cpu_ms() - cpu_open0.unwrap_or(0.0);
        eprintln!(
            "[phase-timing] pcs-open: {wall:.2} ms cpu={cpu:.1} util={:.1}",
            cpu / wall
        );
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

const DEFERRED_STRIPE_GROUPS_PER_JOB: usize = 64;

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
    let mut transposed = [0u8; 64];
    for word in 0..useful_words {
        let lanes: [u64; 8] = std::array::from_fn(|lane| z_u64[lane * u64_per_block + word]);
        flock_core::bits::transpose_8_u64s_to_64_bytes(&lanes, &mut transposed);
        let dst = &mut stripe[word * 64..word * 64 + 64];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // The stripe is not read until after commit+zerocheck, so use the
            // same cache-bypassing store flavor as the eager SIMD witness
            // path. `ldp/stnp` permit these 64-byte slices and the stack
            // temporary without an extra alignment contract.
            core::arch::asm!(
                "ldp {t0:q}, {t1:q}, [{src}]",
                "stnp {t0:q}, {t1:q}, [{dst}]",
                "ldp {t0:q}, {t1:q}, [{src}, #32]",
                "stnp {t0:q}, {t1:q}, [{dst}, #32]",
                src = in(reg) transposed.as_ptr(),
                dst = in(reg) dst.as_mut_ptr(),
                t0 = out(vreg) _,
                t1 = out(vreg) _,
                options(nostack)
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        dst.copy_from_slice(&transposed);
    }
    // The ranked padded fold never observes the tail. Keeping the honest
    // full-stripe contract in tests catches accidental selector widening.
    #[cfg(test)]
    stripe[useful_words * 64..].fill(0);
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
    dispatch(n_jobs, &|job| {
        let group_start = job * DEFERRED_STRIPE_GROUPS_PER_JOB;
        let group_end = (group_start + DEFERRED_STRIPE_GROUPS_PER_JOB).min(n_groups);
        for group in group_start..group_end {
            // SAFETY: every dispatched job owns a distinct contiguous set of
            // groups, and each group maps to one disjoint k-byte stripe.
            let stripe =
                unsafe { std::slice::from_raw_parts_mut(stripe_base.ptr().add(group * k), k) };
            fill_deferred_lincheck_stripe_group(
                z_packed,
                stripe,
                group,
                group_f128,
                u64_per_block,
                useful_words,
            );
        }
    })
}

/// Safe sequential fallback if the utility pool becomes unavailable after the
/// witness driver selected deferred materialization.
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
        // release-store, which follows its last stripe write; the buffer
        // is not written again until after zerocheck consumed the stash.
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

    let precompute_ab = || {
        zerocheck::univariate_skip_optimized::precompute_round1_ab_inner_packed_padded(
            a_packed,
            b_packed,
            pcs_params.m,
            k_skip,
            inv_table,
            padding,
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

    let result = rayon::join(
        || {
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
            if let Some(hook) = tail_fill {
                hook(&pre.0);
            }
            pre
        },
        || {
            let t = std::time::Instant::now();
            let r = precompute_ab();
            let wall_ms = t.elapsed().as_secs_f64() * 1e3;
            // The hybrid-commit warmup sweep sizes its contention emulation
            // from this arm's measured wall (an Instant read is free; the
            // store is one relaxed atomic per prove).
            flock_core::gpu_commit::note_precompute_branch_wall_ms(wall_ms);
            if std::env::var_os("FLOCK_PHASE_TIMING").is_some() {
                eprintln!("[phase-timing] ab-precompute branch wall: {wall_ms:.2} ms");
            }
            r
        },
    );

    if run_ranked_exact_tune {
        flock_core::gpu_commit::retune_ranked_hybrid_with_exact_contention(
            pcs_params,
            &result.0.1.codeword,
            &result.0.1.merkle_tree,
            || {
                let replayed = precompute_ab();
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
    challenger: &mut Ch,
) -> ProveCore {
    let padding = r1cs.padding_spec();
    let phase_timing = std::env::var_os("FLOCK_PHASE_TIMING").is_some();
    let run_commit = |tail_fill: Option<CommitTailFillHook<'_>>| {
        let cpu0 = phase_timing.then(process_cpu_ms);
        let t_commit = std::time::Instant::now();
        let ((commitment, prover_data), ab_inner) = commit_with_round1_ab_precompute(
            &z_packed,
            &a_packed_f128,
            &b_packed_f128,
            pcs_params,
            &padding,
            commit_codeword,
            tail_fill,
        );
        if phase_timing {
            let wall = t_commit.elapsed().as_secs_f64() * 1e3;
            let cpu = process_cpu_ms() - cpu0.unwrap_or(0.0);
            eprintln!(
                "[phase-timing] commit+ab-precompute: {wall:.2} ms cpu={cpu:.1} util={:.1}",
                cpu / wall
            );
        }
        (commitment, prover_data, ab_inner)
    };

    // Publish the stripe before zerocheck: the ranked identity-C producer now
    // consumes it immediately after the commitment challenge is available.
    // Deferred fill is shorter than commit+AB at the target shape, so moving
    // this join earlier adds no measured tail while eliminating C's 32-bank
    // row-major drain.
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
            (run_commit(hook), stripe)
        }
        LincheckStripeInput::DeferredRanked => {
            assert_eq!(r1cs.m, 32);
            assert_eq!(r1cs.k_log, 14);
            assert_eq!(r1cs.useful_bits, 15_409);
            let (m, k_log, useful_bits) = (r1cs.m, r1cs.k_log, r1cs.useful_bits);
            let mut stripe = flock_core::scratch::take_u8(1usize << (m - 3));
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
            // release follows the fill worker's last stripe write.
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
            let pre = std::thread::scope(|scope| {
                let stripe_ready = &stripe_ready;
                let stripe_job = scope.spawn(|| {
                    let started = std::time::Instant::now();
                    let filled_on_epool = fill_deferred_lincheck_stripe_with_dispatch(
                        &z_packed,
                        &mut stripe,
                        m,
                        k_log,
                        useful_bits,
                        |n_jobs, job| {
                            flock_core::epool::run_helper_only_chunks(n_jobs, epool_workers, job)
                        },
                    );
                    if !filled_on_epool {
                        fill_deferred_lincheck_stripe(
                            &z_packed,
                            &mut stripe,
                            m,
                            k_log,
                            useful_bits,
                        );
                    }
                    // Last stripe write is above; the release-store publishes
                    // the bytes to the commit arm's tail-fill hook (acquire).
                    stripe_ready.store(true, std::sync::atomic::Ordering::Release);
                    if phase_timing {
                        eprintln!(
                            "[phase-timing] deferred lincheck stripe: {:.2} ms mode={} workers={}",
                            started.elapsed().as_secs_f64() * 1e3,
                            if filled_on_epool {
                                "epool"
                            } else {
                                "sequential"
                            },
                            if filled_on_epool { epool_workers } else { 0 },
                        );
                    }
                });
                let pre = run_commit(tail_fill_hook);
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
            });
            (pre, stripe)
        }
    };
    let (commitment, prover_data, ab_inner) = pre_zerocheck;
    bind_statement(challenger, r1cs, &commitment);
    let cpu_zc0 = phase_timing.then(process_cpu_ms);
    let t_zc = std::time::Instant::now();

    let (zc_proof, zc_claim, s_hat_v_c) = {
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
    };
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

    let cpu_lc0 = phase_timing.then(process_cpu_ms);
    let t_lc = std::time::Instant::now();
    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

    // Capture lincheck's pre-sumcheck z_vec so the PCS open can derive the
    // AB-claim's `s_hat_v` from it (skips fold_1b_rows for AB).
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
    // The lincheck stripe copy of z is dead from here on; return it to the
    // scratch byte pool before the PCS open (2^(m-3) bytes — 512 MB at
    // m = 32) so the next prove reuses its resident pages.
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
    flock_core::scratch::give_f128(z_vec_pre);
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

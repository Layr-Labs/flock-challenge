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
use flock_core::field::{F8, F128};
use flock_core::lincheck::{self, QuirkyPoint, pack_z_lincheck_from_packed};
use flock_core::pcs::{self, Commitment, PcsParams};
use flock_core::proof::{R1csClaim, R1csProofLigerito, ZClaim, bind_statement};
use flock_core::r1cs::BlockR1cs;
use flock_core::zerocheck;

#[inline]
fn ranked_direct_ab_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    cfg!(target_arch = "x86_64")
        && r1cs.m == 32
        && r1cs.k_log >= pcs::LOG_PACKING + 2
        && std::env::var_os("FLOCK_NO_OPEN_DIRECT_AB").is_none()
}

#[inline]
fn ranked_direct_c_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    ranked_direct_ab_precompute_enabled(r1cs) && pcs::ranked_direct_c_enabled()
}

/// Direct-fold4 (sixteen-bank) precompute: needs the ranked direct-AB shape,
/// the shared `ranked_direct_fold4_enabled()` latch, and a round-1 C capture
/// that actually produced the fold4 tensor.
#[inline]
fn ranked_direct_fold4_precompute_enabled(
    r1cs: &BlockR1cs,
    captured: &zerocheck::CapturedSHatVC,
) -> bool {
    ranked_direct_ab_precompute_enabled(r1cs)
        && pcs::ranked_direct_fold4_enabled()
        && captured.fold4.is_some()
        && r1cs.k_log >= pcs::LOG_PACKING + 4
}

/// Memoised [`BlockR1cs::c0_is_identity`].
///
/// `c0_is_identity` walks all `2^k_log` rows of `c_0` and compares each row
/// against `[i]`. At the ranked shape that is 16,384 separately
/// heap-allocated `Vec<usize>`, i.e. 16,384 dependent pointer chases, and it
/// runs once per prove inside the serial Fiat-Shamir chain to answer a
/// question that is a fixed property of the statement. Measured cost of the
/// enclosing predicate: 0.166 ms warm median (gap seam
/// `zerocheck: pool entered`, n=48 warm proves, resolution ~0.007 ms).
///
/// The answer is keyed on `statement_digest()`, itself a `OnceLock` that
/// absorbs `a_0`, `b_0` and `c_0`, so two different statements can never
/// share a cached answer and a mutated `c_0` changes the key. The digest is
/// already materialised by `bind_statement` on the first prove, so the warm
/// lookup is a `OnceLock` load plus one 32-byte compare.
///
/// `FLOCK_NO_R1CS_C0_ID_CACHE=1` restores the per-prove walk. The env var is
/// read once into a `LazyLock`, never inside a loop.
fn c0_is_identity_cached(r1cs: &BlockR1cs) -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_R1CS_C0_ID_CACHE").is_none());
    if !*ON {
        return r1cs.c0_is_identity();
    }
    static CACHE: std::sync::Mutex<Option<([u8; 32], bool)>> = std::sync::Mutex::new(None);
    let digest = r1cs.statement_digest();
    let mut slot = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((cached_digest, value)) = *slot
        && cached_digest == digest
    {
        return value;
    }
    let value = r1cs.c0_is_identity();
    *slot = Some((digest, value));
    value
}

/// Exact-shape gate for deriving identity C from one block-major outer fold of
/// the witness instead of draining it into 32 field banks in zerocheck round
/// one. Deliberately narrower than the generic DirectFold4 gate — the shortcut
/// relies on ranked BLAKE3's block geometry and honest padding, and every miss
/// keeps the incumbent producer.
///
/// The fold4 capture is checked by shape rather than by a `CapturedSHatVC`,
/// because this gate runs *before* round one produces one; the stripe path
/// always emits the fold4 tensor, so the downstream consumer gate still agrees.
#[inline]
fn ranked_identity_c_fold_enabled(r1cs: &BlockR1cs) -> bool {
    ranked_direct_ab_precompute_enabled(r1cs)
        && pcs::ranked_direct_fold4_enabled()
        && r1cs.k_log >= pcs::LOG_PACKING + 4
        && r1cs.m == 32
        && r1cs.k_log == 14
        && r1cs.k_skip == zerocheck::K_SKIP
        && r1cs.useful_bits == 15_409
        && r1cs.layout == flock_core::r1cs::WitnessLayout::RowMajor
        && c0_is_identity_cached(r1cs)
        && std::env::var_os("FLOCK_NO_ZC_IDENTITY_C").is_none()
}

/// Direct-fold8 capture/consumer predicate: the fold4 chain plus the shared
/// fold8 latch and six retainable tail coordinates (k_log >= k_skip + 7).
#[inline]
fn ranked_direct_fold8_precompute_enabled(
    r1cs: &BlockR1cs,
    captured: &zerocheck::CapturedSHatVC,
) -> bool {
    ranked_direct_fold4_precompute_enabled(r1cs, captured)
        && pcs::ranked_direct_fold8_enabled()
        && captured.fold8.is_some()
        && r1cs.k_log >= r1cs.k_skip + 7
        && r1cs.k_log >= pcs::LOG_PACKING + 6
}

/// AB claim precompute from lincheck's pre-sumcheck `z_vec`: sixty-four banks
/// when fold8 is live, sixteen banks on fold4, four banks on direct-C, and the
/// canonical 128-vector otherwise.
fn precompute_ab_s_hat_v(
    r1cs: &BlockR1cs,
    captured: &zerocheck::CapturedSHatVC,
    z_vec: &[F128],
    inner_rest_tail: &[F128],
) -> Option<Vec<F128>> {
    if ranked_direct_fold8_precompute_enabled(r1cs, captured) {
        Some(pcs::ring_switch::s_hat_v_fold8_from_z_vec(
            z_vec,
            inner_rest_tail,
        ))
    } else if ranked_direct_fold4_precompute_enabled(r1cs, captured) {
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

/// Select the first-bind Fold8 capture only for the exact ranked geometry.
///
/// `FLOCK_NO_R1_FOL8_CAPTURE_REUSE=1` restores the incumbent pre-sumcheck
/// clone plus the existing `s_hat_v_fold8_from_z_vec` consumer. The selector
/// is cached so the hot path does not repeat an environment lookup.
#[inline]
fn ab_capture_mode(
    r1cs: &BlockR1cs,
    captured: &zerocheck::CapturedSHatVC,
) -> lincheck::ZCaptureMode {
    static REUSE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_R1_FOL8_CAPTURE_REUSE").is_none());
    if *REUSE
        && ranked_direct_fold8_precompute_enabled(r1cs, captured)
        && r1cs.k_log.checked_sub(r1cs.k_skip) == Some(8)
    {
        lincheck::ZCaptureMode::RankedFold8Tail
    } else {
        lincheck::ZCaptureMode::PreSumcheck
    }
}

/// Finish AB's precomputed statistic, keeping every non-Fold8 route on the
/// incumbent helper. A successful first-bind capture is already the exact
/// sixty-four-bank Fold8 tensor consumed by the PCS open.
#[inline]
fn finish_ab_s_hat_v(
    r1cs: &BlockR1cs,
    captured: &zerocheck::CapturedSHatVC,
    z_vec: Vec<F128>,
    actual: lincheck::ZCaptureMode,
    inner_rest_tail: &[F128],
) -> Option<Vec<F128>> {
    if actual == lincheck::ZCaptureMode::RankedFold8Tail {
        debug_assert!(ranked_direct_fold8_precompute_enabled(r1cs, captured));
        debug_assert_eq!(z_vec.len(), 64 * (1usize << pcs::LOG_PACKING));
        Some(z_vec)
    } else {
        precompute_ab_s_hat_v(r1cs, captured, &z_vec, inner_rest_tail)
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
    Some(if ranked_direct_fold8_precompute_enabled(r1cs, captured) {
        captured.fold8.as_deref().unwrap()
    } else if ranked_direct_fold4_precompute_enabled(r1cs, captured) {
        captured.fold4.as_deref().unwrap()
    } else if ranked_direct_c_precompute_enabled(r1cs) {
        captured.quad.as_slice()
    } else {
        captured.s_hat_v_c.as_slice()
    })
}

enum FastLincheckInput {
    Stripe(Vec<u8>),
    BlockMajor,
}

const PHASE_POOLS_DISABLE_ENV: &str = "FLOCK_NO_PHASE_POOLS";
const WITNESS_PHASE_POOL_DISABLE_ENV: &str = "FLOCK_NO_WITNESS_PHASE_POOL";

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn phase_pool_widths(m: usize) -> Option<(usize, usize)> {
    if m < 29 || std::env::var_os(PHASE_POOLS_DISABLE_ENV).is_some() {
        return None;
    }
    let global = rayon::current_num_threads().max(1);
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(global);
    if available <= global {
        return None;
    }
    Some(((global + 2).min(available), available))
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn phase_pool_widths(m: usize) -> Option<(usize, usize)> {
    let _ = (m, PHASE_POOLS_DISABLE_ENV);
    None
}

fn commit_phase_pool(m: usize) -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    let (width, _) = phase_pool_widths(m)?;
    Some(POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .thread_name(|i| format!("flock-commit-{i}"))
            .build()
            .expect("build cached commit pool")
    }))
}

fn zerocheck_phase_pool(m: usize) -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    let (_, width) = phase_pool_widths(m)?;
    Some(POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .thread_name(|i| format!("flock-zerocheck-{i}"))
            .build()
            .expect("build cached zerocheck pool")
    }))
}

fn in_commit_phase_pool<R: Send>(m: usize, op: impl FnOnce() -> R + Send) -> R {
    match commit_phase_pool(m) {
        Some(pool) => pool.install(op),
        None => op(),
    }
}

fn in_zerocheck_phase_pool<R: Send>(m: usize, op: impl FnOnce() -> R + Send) -> R {
    match zerocheck_phase_pool(m) {
        Some(pool) => pool.install(op),
        None => op(),
    }
}

/// Run the BLAKE3 row-major witness/projection phase in the existing cached
/// all-available pool. The pool's platform and large-instance guards remain
/// centralized in `zerocheck_phase_pool`; this adds only a phase-local opt-out.
pub(crate) fn in_witness_phase_pool<R: Send>(m: usize, op: impl FnOnce() -> R + Send) -> R {
    if std::env::var_os(WITNESS_PHASE_POOL_DISABLE_ENV).is_some() {
        return op();
    }
    in_zerocheck_phase_pool(m, op)
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
    // NOTE: the open runs on the caller's (global 10T) pool ON PURPOSE. A
    // whole-phase wide (P+E) pool was measured (paired same-session A/B,
    // m=32, 18 opens/arm) and is a wash: the combine gains −3.3 ms but every
    // short statically-chunked ligerito section loses to E-core stragglers
    // (lig-prove total +4.3 ms). Only the combine is widened, inside
    // `pcs::in_wide_combine_pool` (kill switch FLOCK_NO_OPEN_POOL=1).
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
    let c_packed_f128: Vec<F128> = if c0_is_identity_cached(r1cs) {
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
    let (lc_proof, lc_claim, z_vec_pre, z_mode) = lincheck::prove_padded_capture_z_vec_mode(
        &z_packed_lincheck,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        r1cs.useful_bits,
        &lc_circuit,
        &x_ab,
        ab_capture_mode(r1cs, &s_hat_v_c),
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

    let s_hat_v_ab = finish_ab_s_hat_v(
        r1cs,
        &s_hat_v_c,
        z_vec_pre,
        z_mode,
        &lc_claim.r_inner_rest[1..],
    );
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c = pre_c_slot(r1cs, &s_hat_v_c);
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
    prove_fast_ligerito_from_witness_inner(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        None,
        FastLincheckInput::Stripe(z_packed_lincheck),
        lincheck_circuit,
        prefaulted_codeword,
        challenger,
    )
}

/// Stripe-free fast path for a canonical row-major witness. `z_packed`
/// remains owned by the pipeline for the PCS open; lincheck borrows it only
/// after the outer challenge is available.
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_from_block_major_witness<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    prove_fast_ligerito_from_witness_inner(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        None,
        FastLincheckInput::BlockMajor,
        lincheck_circuit,
        prefaulted_codeword,
        challenger,
    )
}

/// Block-major fast path whose challenge-independent round-one AB projection
/// was emitted during witness generation. Commitment runs directly, avoiding
/// a post-witness pass over packed A/B; the operands remain live for round 2.
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_from_block_major_witness_with_precomputed_ab<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    ab_inner: zerocheck::univariate_skip_optimized::Round1AbInner,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    prove_fast_ligerito_from_witness_inner(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        Some(ab_inner),
        FastLincheckInput::BlockMajor,
        lincheck_circuit,
        prefaulted_codeword,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_fast_ligerito_from_witness_inner<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    ab_inner: Option<zerocheck::univariate_skip_optimized::Round1AbInner>,
    lincheck_input: FastLincheckInput,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim) {
    flock_core::gaptime::mark("inner: enter");
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
    } = prove_fast_core_with_codeword_inner(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        ab_inner,
        lincheck_input,
        lincheck_circuit,
        prefaulted_codeword,
        challenger,
    );
    flock_core::gaptime::mark("core: returned");

    let padding = r1cs.padding_spec();
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c = pre_c_slot(r1cs, &s_hat_v_c);
    flock_core::gaptime::mark("open: begin");
    // Publish-prefix pre-encode: commitment / zerocheck / lincheck are
    // transcript-final here, so their serialization (plus the 460 kB output
    // allocation) runs on a detached helper thread concurrently with the
    // ~20 ms open instead of inside the measured publish tail. Tens of µs of
    // work; the fingerprint gate in `proof_io` makes a stale or missing
    // stash fall back to the incumbent full encode, byte-identically.
    let stash = if crate::proof_io::pre_encode_enabled() {
        let commitment_c = commitment.clone();
        let zc_c = zc_proof.clone();
        let lc_c = lc_proof.clone();
        Some(std::thread::spawn(move || {
            crate::proof_io::stash_pre_encoded_prefix(&commitment_c, &zc_c, &lc_c);
            (commitment_c, zc_c, lc_c)
        }))
    } else {
        None
    };
    let pcs_open = flock_core::in_pool(|| open_claims_with_precomputed_ligerito(
        z_packed,
        &prover_data,
        &commitment,
        &[ab.clone(), c.clone()],
        &[pre_ab, pre_c],
        &padding,
        &lig_config,
        challenger,
    ));
    if let Some(handle) = stash {
        // Finished long ago (µs vs the ~20 ms open); join keeps the thread
        // from outliving the prove.
        let _ = handle.join();
    }
    flock_core::gaptime::mark("open: returned");

    let proof = R1csProofLigerito {
        zerocheck: zc_proof,
        lincheck: lc_proof,
        pcs_open,
    };
    let claim = R1csClaim { ab, c };
    flock_core::gaptime::mark("proof assembled");
    drop(s_hat_v_ab);
    drop(s_hat_v_c);
    flock_core::gaptime::mark("s_hat_v dropped");
    drop(prover_data);
    flock_core::gaptime::mark("prover_data dropped");
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
    /// Precomputed `s_hat_v` for the C claim — produced by zerocheck round 1's
    /// two-bank fusion kernel (one extra `vld1q+veorq` per chunk-lane-b_med
    /// vs the original single-bank C-side). Skips `fold_1b_rows` for the C
    /// claim at PCS-open time.
    pub s_hat_v_c: zerocheck::CapturedSHatVC,
}

/// Build the witness commitment and the challenge-independent half of
/// zerocheck round 1 on the same fixed Rayon pool.  A/B are only borrowed:
/// their original packed values remain live for zerocheck round 2.
fn commit_with_round1_ab_precompute(
    z_packed: &[F128],
    a_packed_f128: &[F128],
    b_packed_f128: &[F128],
    pcs_params: &PcsParams,
    padding: &zerocheck::PaddingSpec,
    prefaulted_codeword: Option<Vec<F128>>,
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
    let ntt_s = flock_core::ntt::AdditiveNttGf8::new(k_skip, F8::ZERO);
    let ntt_l = flock_core::ntt::AdditiveNttGf8::new(k_skip, F8(1u8 << k_skip));
    let inv_table = flock_core::ntt::InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);

    rayon::join(
        || match prefaulted_codeword {
            Some(buf) => pcs::commit_into(z_packed, pcs_params, buf),
            None => pcs::commit(z_packed, pcs_params),
        },
        || {
            zerocheck::univariate_skip_optimized::precompute_round1_ab_inner_packed_padded(
                a_packed,
                b_packed,
                pcs_params.m,
                k_skip,
                &inv_table,
                padding,
            )
        },
    )
}

/// Fire-and-forget async prewire of the round-1 GPU inputs — the witness
/// buffers a, b, c (= z) plus, when it already exists, the codeword buffer
/// the commit will encode into (the tree buffer is prewired inside
/// `pcs::commit`'s GPU stream, where it is allocated).
///
/// Season-1 measurement: without this the driver's page wiring for the big
/// arrays stalls the round-1 GPU command buffer ~23 ms. All availability/env
/// checks (and the lazy ~45 ms Metal context init) happen on the spawned
/// thread, never on the prove path.
///
/// **No clock-keepalive thread this season** (season-1 had one): the GPU
/// Merkle stream inside commit lands a command buffer every few milliseconds
/// until the end of commit, which keeps the GPU out of its power-gated idle
/// state across the ~100 ms gap to the round-1 URM dispatch — that was the
/// keepalive's entire job.
///
/// The buffers are scratch-pool-retained (never unmapped for the process
/// lifetime), so the detached thread cannot outlive their allocations.
/// Non-Apple builds compile this helper to a no-op: their `gpu::prewire`
/// implementation is empty, so spawning a thread would be pure timed work.
fn gpu_prewire_round1_inputs(a: &[F128], b: &[F128], z: &[F128], codeword: Option<&[F128]>) {
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (a, b, z, codeword);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let view = |v: &[F128]| (v.as_ptr() as usize, std::mem::size_of_val(v));
        let mut bufs = [view(a), view(b), view(z), (0usize, 0usize)];
        if let Some(cw) = codeword {
            bufs[3] = view(cw);
        }
        let _ = std::thread::Builder::new()
            .name("flock-gpu-prewire".into())
            .spawn(move || {
                for (addr, len) in bufs {
                    if len == 0 {
                        continue;
                    }
                    // SAFETY: scratch-pool allocations stay mapped for the
                    // process lifetime (the pool parks, never frees, them).
                    let bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, len) };
                    flock_core::gpu::prewire(bytes);
                }
            });
    }
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
    prove_fast_core_with_codeword(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        z_packed_lincheck,
        lincheck_circuit,
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
    prove_fast_core_with_codeword_inner(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        None,
        FastLincheckInput::Stripe(z_packed_lincheck),
        lincheck_circuit,
        prefaulted_codeword,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_fast_core_with_codeword_inner<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    ab_inner: Option<zerocheck::univariate_skip_optimized::Round1AbInner>,
    lincheck_input: FastLincheckInput,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> ProveCore {
    if matches!(&lincheck_input, FastLincheckInput::BlockMajor) {
        assert_eq!(
            r1cs.layout,
            flock_core::r1cs::WitnessLayout::RowMajor,
            "direct lincheck fold requires canonical block-major z"
        );
    }
    let padding = r1cs.padding_spec();
    flock_core::gaptime::mark("core: padding built");
    gpu_prewire_round1_inputs(
        &a_packed_f128,
        &b_packed_f128,
        &z_packed,
        prefaulted_codeword.as_deref(),
    );
    flock_core::gaptime::mark("core: gpu prewire spawned");
    // The BLAKE3 padding rows force every block's tail words to zero, which in
    // the SoA codeword is a static all-zero pattern on the top lanes of every
    // odd position. Publish it for the duration of this commitment so the NTT
    // can drop those butterflies; the guard restores the previous value on
    // drop, and recursive Ligerito commits run at a different lane count and
    // never match the ranked gate.
    let _zero_lane_skip = flock_core::ntt::additive_ntt_f128::ZeroOddTailLanes::scope(
        pcs_params.num_ntts(),
        flock_core::ntt::additive_ntt_f128::ZeroOddTailLanes::lanes_for_padding(
            pcs_params.num_ntts(),
            padding.k_log,
            padding.useful_bits_per_block,
        ),
    );
    let ((commitment, prover_data), ab_inner) = if let Some(ab_inner) = ab_inner {
        let committed = in_commit_phase_pool(r1cs.m, || {
            flock_core::gaptime::mark("commit: pool entered");
            let r = match prefaulted_codeword {
                Some(buf) => pcs::commit_into(&z_packed, pcs_params, buf),
                None => pcs::commit(&z_packed, pcs_params),
            };
            flock_core::gaptime::mark("commit: work done");
            r
        });
        flock_core::gaptime::mark("commit: pool exited");
        (committed, ab_inner)
    } else {
        let r = in_commit_phase_pool(r1cs.m, || {
            flock_core::gaptime::mark("commit: pool entered");
            let r = commit_with_round1_ab_precompute(
                &z_packed,
                &a_packed_f128,
                &b_packed_f128,
                pcs_params,
                &padding,
                prefaulted_codeword,
            );
            flock_core::gaptime::mark("commit: work done");
            r
        });
        flock_core::gaptime::mark("commit: pool exited");
        r
    };
    bind_statement(challenger, r1cs, &commitment);
    flock_core::gaptime::mark("bind_statement done");

    // Last-ρ leftover z-fold: arm before zerocheck so the last ML ρ can
    // start today's one-shot fold. Guard keeps `z_packed` live until
    // lincheck waits (before fold_alpha). Stripe path does not prepare.
    let _last_rho = if matches!(&lincheck_input, FastLincheckInput::BlockMajor) {
        Some(lincheck::prepare_last_rho_z_fold(
            &z_packed,
            r1cs.m,
            r1cs.k_log,
            r1cs.useful_bits,
            r1cs.k_log - r1cs.k_skip,
        ))
    } else {
        None
    };

    // Ranked identity-C (`FLOCK_NO_ZC_IDENTITY_C=1` restores the 32-bank
    // row-major drain): round one folds the packed witness directly.
    let c_identity_z: Option<&[F128]> =
        ranked_identity_c_fold_enabled(r1cs).then_some(z_packed.as_slice());
    let (zc_proof, zc_claim, s_hat_v_c) =
        in_zerocheck_phase_pool(r1cs.m, || flock_core::in_pool(|| {
            flock_core::gaptime::mark("zerocheck: pool entered");
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
            flock_core::gaptime::mark("zerocheck: views built");
            let r = match c_identity_z {
            Some(c_identity_z) => {
                zerocheck::prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab_and_identity_c(
                    a_packed, b_packed, c_packed, c_identity_z, r1cs.m, &padding, ab_inner,
                    challenger,
                )
            }
            None => zerocheck::prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab(
                a_packed, b_packed, c_packed, r1cs.m, &padding, ab_inner, challenger,
            ),
        };
            flock_core::gaptime::mark("zerocheck: work done");
            r
        }));
    flock_core::gaptime::mark("zerocheck: pool exited");
    // Nothing downstream reads a/b (zerocheck consumed them in rounds 1–2);
    // recycle the two buffers (2 × 2^(m-3) bytes — 128 MB at m = 29) instead
    // of carrying them through lincheck and the PCS open.
    flock_core::scratch::give_f128(a_packed_f128);
    flock_core::scratch::give_f128(b_packed_f128);
    flock_core::gaptime::mark("a/b recycled to scratch pool");

    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);
    flock_core::gaptime::mark("x_ab built");

    // Capture lincheck's pre-sumcheck z_vec so the PCS open can derive the
    // AB-claim's `s_hat_v` from it (skips fold_1b_rows for AB).
    //
    // Lincheck stays on the ambient (10T global) pool: two independent
    // official bisections on THIS lineage measured the 14T zerocheck-pool
    // placement at −0.6% (its z-scan is L1-load-latency-bound and the
    // E-core stragglers dominate the barrier), overriding the earlier
    // ff7f68b-era 14T datapoint from the pre-NT-kernel lineage.
    let (lc_proof, lc_claim, z_vec_pre, z_mode) = flock_core::in_pool(|| match lincheck_input {
        FastLincheckInput::Stripe(z_packed_lincheck) => {
            let result = lincheck::prove_padded_capture_z_vec_mode(
                &z_packed_lincheck,
                r1cs.m,
                r1cs.k_log,
                r1cs.k_skip,
                r1cs.useful_bits,
                lincheck_circuit,
                &x_ab,
                ab_capture_mode(r1cs, &s_hat_v_c),
                challenger,
            );
            // The stripe copy is dead after lincheck. Stripe-based callers
            // retain their existing allocation lifecycle.
            drop(z_packed_lincheck);
            result
        }
        FastLincheckInput::BlockMajor => lincheck::prove_padded_capture_z_vec_block_major_mode(
            &z_packed,
            r1cs.m,
            r1cs.k_log,
            r1cs.k_skip,
            r1cs.useful_bits,
            lincheck_circuit,
            &x_ab,
            ab_capture_mode(r1cs, &s_hat_v_c),
            challenger,
        ),
    });
    flock_core::gaptime::mark("lincheck: done");

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
    let s_hat_v_ab = finish_ab_s_hat_v(
        r1cs,
        &s_hat_v_c,
        z_vec_pre,
        z_mode,
        &lc_claim.r_inner_rest[1..],
    );
    flock_core::gaptime::mark("s_hat_v_ab built (core exit)");

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
    prove_fast_ligerito_timed_inner(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        FastLincheckInput::Stripe(z_packed_lincheck),
        lincheck_circuit,
        prefaulted_codeword,
        challenger,
    )
}

/// Timed counterpart of [`prove_fast_ligerito_from_block_major_witness`].
#[allow(clippy::too_many_arguments)]
pub fn prove_fast_ligerito_timed_from_block_major_witness<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim, ProvePhaseTimings) {
    prove_fast_ligerito_timed_inner(
        r1cs,
        pcs_params,
        z_packed,
        a_packed_f128,
        b_packed_f128,
        FastLincheckInput::BlockMajor,
        lincheck_circuit,
        prefaulted_codeword,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_fast_ligerito_timed_inner<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    z_packed: Vec<F128>,
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    lincheck_input: FastLincheckInput,
    lincheck_circuit: &dyn lincheck::LincheckCircuit,
    prefaulted_codeword: Option<Vec<F128>>,
    challenger: &mut Ch,
) -> (R1csProofLigerito, Commitment, R1csClaim, ProvePhaseTimings) {
    use std::time::Instant;
    if matches!(&lincheck_input, FastLincheckInput::BlockMajor) {
        assert_eq!(
            r1cs.layout,
            flock_core::r1cs::WitnessLayout::RowMajor,
            "direct lincheck fold requires canonical block-major z"
        );
    }
    let mut t = ProvePhaseTimings::default();

    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito default config; bump m for tiny instances");

    let padding = r1cs.padding_spec();
    gpu_prewire_round1_inputs(
        &a_packed_f128,
        &b_packed_f128,
        &z_packed,
        prefaulted_codeword.as_deref(),
    );

    // --- PCS commit + challenge-independent zerocheck AB preprocessing ---
    let t0 = Instant::now();
    let ((commitment, prover_data), ab_inner) = in_commit_phase_pool(r1cs.m, || {
        commit_with_round1_ab_precompute(
            &z_packed,
            &a_packed_f128,
            &b_packed_f128,
            pcs_params,
            &padding,
            prefaulted_codeword,
        )
    });
    t.commit_s = t0.elapsed().as_secs_f64();
    bind_statement(challenger, r1cs, &commitment);

    let _last_rho = if matches!(&lincheck_input, FastLincheckInput::BlockMajor) {
        Some(lincheck::prepare_last_rho_z_fold(
            &z_packed,
            r1cs.m,
            r1cs.k_log,
            r1cs.useful_bits,
            r1cs.k_log - r1cs.k_skip,
        ))
    } else {
        None
    };

    // --- zerocheck ---
    let t0 = Instant::now();
    // Ranked identity-C (`FLOCK_NO_ZC_IDENTITY_C=1` restores the 32-bank
    // row-major drain): round one folds the packed witness directly.
    let c_identity_z: Option<&[F128]> =
        ranked_identity_c_fold_enabled(r1cs).then_some(z_packed.as_slice());
    let (zc_proof, zc_claim, s_hat_v_c) =
        in_zerocheck_phase_pool(r1cs.m, || {
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
            match c_identity_z {
            Some(c_identity_z) => {
                zerocheck::prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab_and_identity_c(
                    a_packed, b_packed, c_packed, c_identity_z, r1cs.m, &padding, ab_inner,
                    challenger,
                )
            }
            None => zerocheck::prove_packed_padded_capture_s_hat_v_c_with_precomputed_ab(
                a_packed, b_packed, c_packed, r1cs.m, &padding, ab_inner, challenger,
            ),
        }
        });
    t.zerocheck_s = t0.elapsed().as_secs_f64();
    flock_core::scratch::give_f128(a_packed_f128);
    flock_core::scratch::give_f128(b_packed_f128);

    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

    // --- lincheck + base-claim / s_hat_v setup ---
    let t0 = Instant::now();
    let (lc_proof, lc_claim, z_vec_pre, z_mode) = match lincheck_input {
        FastLincheckInput::Stripe(z_packed_lincheck) => {
            let result = lincheck::prove_padded_capture_z_vec_mode(
                &z_packed_lincheck,
                r1cs.m,
                r1cs.k_log,
                r1cs.k_skip,
                r1cs.useful_bits,
                lincheck_circuit,
                &x_ab,
                ab_capture_mode(r1cs, &s_hat_v_c),
                challenger,
            );
            drop(z_packed_lincheck);
            result
        }
        FastLincheckInput::BlockMajor => lincheck::prove_padded_capture_z_vec_block_major_mode(
            &z_packed,
            r1cs.m,
            r1cs.k_log,
            r1cs.k_skip,
            r1cs.useful_bits,
            lincheck_circuit,
            &x_ab,
            ab_capture_mode(r1cs, &s_hat_v_c),
            challenger,
        ),
    };
    let ab = ZClaim {
        point: r1cs.ab_claim_point(lc_claim.r_inner_skip, &lc_claim.r_inner_rest, &x_ab.x_outer),
        value: lc_claim.w,
    };
    let c = ZClaim {
        point: r1cs.c_claim_point(zc_claim.z, &zc_claim.r_rest),
        value: zc_claim.c_eval,
    };
    let s_hat_v_ab = finish_ab_s_hat_v(
        r1cs,
        &s_hat_v_c,
        z_vec_pre,
        z_mode,
        &lc_claim.r_inner_rest[1..],
    );
    t.lincheck_s = t0.elapsed().as_secs_f64();

    // --- Ligerito recursive PCS open ---
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c = pre_c_slot(r1cs, &s_hat_v_c);
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
// gin-x86-spawn-free-resample-01: independent timing sample of the target-dead thread deletion.
// gin-x86-spawn-free-resample-02: second independent timing sample of the target-dead thread deletion.

// zarar-x86-resample-03: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-07: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-10: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-14: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-29: independent official timing sample of the promoted source; no executable change.

// newjordan-x86-refire-29: independent official timing sample of the promoted
// source; no executable change. (drawn against fbe181b frontier, 1410759.47)

// newjordan-x86-refire-11: independent official timing sample of the promoted
// source; no executable change. (drawn against a3b828b frontier)

// newjordan-x86-refire-19: independent official timing sample of the
// promoted source; no executable change. (drawn against ebdd5d5 frontier)

// zarar-x86-resample-29: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-30: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-32: independent official timing sample of the promoted source; no executable change.

// refire-45: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-47: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-110: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-111: independent official timing sample of the promoted source; no executable change.

// zarar-x86-resample-170: independent official timing sample of the promoted source; no executable change.

// zarar-x86-draw-20260826T211753Z: independent official timing sample; no executable change.

// zarar-x86-draw-20260828T215451Z: independent official timing sample; no executable change.

// zarar-x86-draw-20260828T223447Z: independent official timing sample; no executable change.

// zarar-x86-draw-20260828T225139Z: independent official timing sample; no executable change.

// zarar-x86-draw-20260828T230904Z: independent official timing sample; no executable change.

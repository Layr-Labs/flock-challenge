
use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::{self, QuirkyPoint, pack_z_lincheck_from_packed};
use flock_core::pcs::{self, Commitment, PcsParams};
use flock_core::proof::{R1csClaim, R1csProofLigerito, ZClaim, bind_statement};
use flock_core::r1cs::{BlockR1cs, Round1PinnedOnePrefixStatement};
use flock_core::zerocheck;
use crate::r1cs_hashes::blake3::Blake3IdentityPrefixWitness;

pub const ENV_NO_ZC_AB_IDENTITY_PREFIX_OMIT: &str = "FLOCK_NO_ZC_AB_IDENTITY_PREFIX_OMIT";

pub const ENV_NO_ZC_AB_OUTPUT_WINDOW_OMIT: &str = "FLOCK_NO_ZC_AB_OUTPUT_WINDOW_OMIT";

#[inline]
fn ranked_direct_ab_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && r1cs.m == 32
        && r1cs.k_log >= pcs::LOG_PACKING + 2
        && std::env::var_os("FLOCK_NO_OPEN_DIRECT_AB").is_none()
        && std::env::var_os("FLOCK_NO_LIG_FOLD2").is_none()
}

#[inline]
fn ranked_direct_c_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    ranked_direct_ab_precompute_enabled(r1cs) && pcs::ranked_direct_c_enabled()
}

#[inline]
fn ranked_direct_fold4_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    ranked_direct_ab_precompute_enabled(r1cs) && pcs::ranked_direct_fold4_enabled()
}

#[inline]
fn ranked_direct_fold8_precompute_enabled(r1cs: &BlockR1cs) -> bool {
    ranked_direct_fold4_precompute_enabled(r1cs)
        && pcs::ranked_direct_fold8_enabled()
        && r1cs.k_log >= r1cs.k_skip + 7
}

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

pub(crate) fn quirky_x_outer_full(point: &QuirkyPoint) -> Vec<F128> {
    let mut v = Vec::with_capacity(point.x_inner_rest.len() + point.x_outer.len());
    v.extend_from_slice(&point.x_inner_rest);
    v.extend_from_slice(&point.x_outer);
    v
}

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
    if crate::seed_pipe::is_timed_speculative_prove() {
        std::mem::forget(s_hat_v_ab);
        std::mem::forget(s_hat_v_c);
        std::mem::forget(prover_data);
    }
    (proof, commitment, claim)
}

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
        challenger,
    )
}

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
        challenger,
    )
}

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
        challenger,
    )
}

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
        challenger,
    );

    let padding = r1cs.padding_spec();
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = pre_c_slot(r1cs, &s_hat_v_c);
    let phase_timing = std::env::var_os("FLOCK_PHASE_TIMING").is_some();
    flock_core::tmark::mark("open start");
    let cpu_open0 = phase_timing.then(process_cpu_ms);
    let t_open = std::time::Instant::now();
    let stash_pool = (crate::proof_io::pre_encode_enabled() && rayon::current_num_threads() > 1)
        .then(flock_core::epool::helper_pool)
        .flatten();
    let pcs_open = match stash_pool {
        Some(pool) => pool.in_place_scope(|s| {
            s.spawn(|_| {
                crate::proof_io::stash_pre_encoded_prefix(&commitment, &zc_proof, &lc_proof)
            });
            flock_core::in_pool(|| {
                flock_core::tmark::mark("open body start");
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
            })
        }),
        None => flock_core::in_pool(|| {
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
    };
    flock_core::tmark::mark("open returned");
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
    if crate::seed_pipe::is_timed_speculative_prove() {
        std::mem::forget(s_hat_v_ab);
        std::mem::forget(s_hat_v_c);
        std::mem::forget(prover_data);
    }
    (proof, commitment, claim)
}

pub struct ProveCore {
    pub zc_proof: zerocheck::ZerocheckProof,
    pub lc_proof: lincheck::LincheckProof,
    pub ab: ZClaim,
    pub c: ZClaim,
    pub commitment: Commitment,
    pub prover_data: pcs::ProverData,
    pub z_packed: Vec<F128>,
    pub s_hat_v_ab: Option<Vec<F128>>,
    pub s_hat_v_c: zerocheck::CapturedSHatVC,
}

enum CommitCodeword {
    Allocate,
    NeedsReplication(Vec<F128>),
    Preinitialized(Vec<F128>),
    StreamedFirstPass(Vec<F128>, flock_core::gpu_commit::FromZFirstPassStream),
}

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
    let z_u64: &[u64] =
        unsafe { std::slice::from_raw_parts(z_group.as_ptr().cast::<u64>(), z_group.len() * 2) };
    #[cfg(not(target_arch = "aarch64"))]
    let mut transposed = [0u8; 64];
    assert!(useful_words <= u64_per_block && z_u64.len() == 8 * u64_per_block);
    for word in 0..useful_words {
        let dst = &mut stripe[word * 64..word * 64 + 64];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let [y0, y1, y2, y3] = flock_core::bits::transpose_8_strided_u64s_to_regs(
                z_u64.as_ptr().add(word),
                u64_per_block,
            );
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
    #[cfg(test)]
    stripe[useful_words * 64..].fill(0);
}

fn deferred_lincheck_stripe_job<'a>(
    z_packed: &'a [F128],
    z_lincheck: &mut [u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
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
    };
    (n_jobs, job)
}

fn fill_deferred_lincheck_stripe_with_dispatch(
    z_packed: &[F128],
    z_lincheck: &mut [u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    dispatch: impl FnOnce(usize, &(dyn Fn(usize) + Sync)) -> bool,
) -> bool {
    let (n_jobs, job) = deferred_lincheck_stripe_job(z_packed, z_lincheck, m, k_log, useful_bits);
    dispatch(n_jobs, &job)
}

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
            let (n_jobs, job) = deferred_lincheck_stripe_job(&z, &mut stripe, M, K_LOG, 1 << K_LOG);
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

type CommitTailFillHook<'a> = Box<dyn FnOnce(&Commitment) + Send + 'a>;

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
            if std::env::var_os("FLOCK_ZC_TIMING").is_some() {
                eprintln!("[commit-tail-fill] skipped: stripe not ready");
            }
            return;
        }
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
    let run_ranked_exact_tune = flock_core::gpu_commit::ranked_exact_contention_tune_pending();

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
            if let Some(hook) = tail_fill {
                hook(&pre.0);
            }
            pre
        },
        || {
            let t = std::time::Instant::now();
            let r = precompute_ab();
            let wall_ms = t.elapsed().as_secs_f64() * 1e3;
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
                let replayed = precompute_ab_replay();
                std::hint::black_box(&replayed);
            },
        );
    }
    result
}

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
        challenger,
    )
}

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
    challenger: &mut Ch,
) -> ProveCore {
    let padding = r1cs.padding_spec();
    let identity_prefix_contract = ranked_identity_prefix_omission_contract(
        r1cs, identity_prefix_witness.as_ref(), &z_packed, &a_packed_f128, &b_packed_f128,
    );
    let phase_timing = std::env::var_os("FLOCK_PHASE_TIMING").is_some();
    let run_commit = |tail_fill: Option<CommitTailFillHook<'_>>,
                      stripe_help: Option<&(dyn Fn() + Sync)>| {
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
            identity_prefix_contract,
            stripe_help,
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

    let (pre_zerocheck, z_packed_lincheck) = match z_packed_lincheck {
        LincheckStripeInput::Ready(stripe) => {
            let stripe_ready = std::sync::atomic::AtomicBool::new(true);
            let hook = make_commit_tail_fill_hook(
                &*challenger,
                r1cs,
                &padding,
                &stripe_ready,
                stripe.as_ptr() as usize,
                stripe.len(),
            );
            (run_commit(hook, None), stripe)
        }
        LincheckStripeInput::DeferredRanked => {
            assert_eq!(r1cs.m, 32);
            assert_eq!(r1cs.k_log, 14);
            assert_eq!(r1cs.useful_bits, 15_409);
            let (m, k_log, useful_bits) = (r1cs.m, r1cs.k_log, r1cs.useful_bits);
            let mut stripe = flock_core::scratch::take_u8(1usize << (m - 3));
            const DEFER_STRIPE_EPOOL_THREADS_DEFAULT: usize = 4;
            let epool_workers = std::env::var_os("FLOCK_DEFER_STRIPE_EPOOL_THREADS")
                .and_then(|value| value.to_str()?.parse::<usize>().ok())
                .filter(|workers| (1..=4).contains(workers))
                .unwrap_or(DEFER_STRIPE_EPOOL_THREADS_DEFAULT);
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
            let (n_stripe_jobs, stripe_job_fn) =
                deferred_lincheck_stripe_job(&z_packed, &mut stripe, m, k_log, useful_bits);
            let stripe_queue = flock_core::epool::HelpQueue::new(n_stripe_jobs, &stripe_job_fn);
            let stripe_help = || stripe_queue.help();
            let pre = std::thread::scope(|scope| {
                let stripe_ready = &stripe_ready;
                let stripe_queue = &stripe_queue;
                let stripe_job = scope.spawn(move || {
                    let started = std::time::Instant::now();
                    let filled_on_epool =
                        flock_core::epool::run_helper_only_queue(stripe_queue, epool_workers);
                    if !filled_on_epool {
                        stripe_queue.drain_inline();
                    }
                    stripe_queue.wait_done();
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
                let pre = run_commit(tail_fill_hook, Some(&stripe_help));
                stripe_queue.help();
                let join_started = std::time::Instant::now();
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
    flock_core::tmark::mark("zc start");
    let cpu_zc0 = phase_timing.then(process_cpu_ms);
    let t_zc = std::time::Instant::now();

    let (zc_proof, zc_claim, s_hat_v_c) = flock_core::in_pool(|| {
        flock_core::tmark::mark("zc body start");
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
    flock_core::scratch::give_f128(a_packed_f128);
    flock_core::scratch::give_f128(b_packed_f128);

    flock_core::tmark::mark("zc end");
    let cpu_lc0 = phase_timing.then(process_cpu_ms);
    let t_lc = std::time::Instant::now();
    let x_ab = r1cs.x_ab_from_mlv(zc_claim.z, &zc_claim.mlv_challenges);

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
    flock_core::tmark::mark("lc returned");
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

#[derive(Clone, Copy, Debug, Default)]
pub struct ProvePhaseTimings {
    pub witness_s: f64,
    pub commit_s: f64,
    pub zerocheck_s: f64,
    pub lincheck_s: f64,
    pub open_s: f64,
}

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
    );
    t.commit_s = t0.elapsed().as_secs_f64();
    bind_statement(challenger, r1cs, &commitment);

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
    flock_core::scratch::give_f128(z_vec_pre);
    t.lincheck_s = t0.elapsed().as_secs_f64();

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

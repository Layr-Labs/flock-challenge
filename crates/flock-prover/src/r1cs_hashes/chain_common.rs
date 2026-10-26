use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::pcs::{
    Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef, PcsParams,
};
use flock_core::r1cs::BlockR1cs;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct ChainLayout {
    pub k_log: usize,

    pub k_skip: usize,

    pub region_log: usize,

    pub region_bits: usize,

    pub input_byte_off: usize,

    pub output_byte_off: usize,
}

impl ChainLayout {
    #[inline]
    pub fn tau_pos_len(&self) -> usize {
        self.region_log - LOG_PACKING
    }

    #[inline]
    pub fn high_zeros(&self) -> usize {
        self.k_log - self.region_log - 1
    }
}

#[derive(Clone, Debug)]
pub struct ChainFold {
    pub tau_pos: Vec<F128>,
}

impl ChainFold {
    pub fn new(layout: &ChainLayout, tau_pos: Vec<F128>) -> Self {
        assert_eq!(
            tau_pos.len(),
            layout.tau_pos_len(),
            "τ_pos length must be region_log − LOG_PACKING"
        );
        Self { tau_pos }
    }

    pub fn fold_public_phys(&self, phys_bits: &[bool]) -> F128 {
        let bits_per_packed = 1usize << LOG_PACKING;
        let n_packed = 1usize << self.tau_pos.len();
        let slot_bits = n_packed * bits_per_packed;
        assert!(
            phys_bits.len() <= slot_bits,
            "fold_public_phys: phys_bits length {} > slot bits {}",
            phys_bits.len(),
            slot_bits,
        );

        let eq_tau = build_eq_table(&self.tau_pos);
        let mut acc = F128::ZERO;
        for pos in 0..n_packed {
            let mut packed = F128::ZERO;
            for b in 0..bits_per_packed {
                let bit_idx = pos * bits_per_packed + b;
                if bit_idx < phys_bits.len() && phys_bits[bit_idx] {
                    if b < 64 {
                        packed.lo |= 1u64 << b;
                    } else {
                        packed.hi |= 1u64 << (b - 64);
                    }
                }
            }
            acc += eq_tau[pos] * packed;
        }
        acc
    }
}

pub fn fold_in_out(
    layout: &ChainLayout,
    wl: flock_core::r1cs::WitnessLayout,
    packed: &[F128],
    fold: &ChainFold,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;

    let bits_per_packed = 1usize << LOG_PACKING;
    let n_packed_per_region = 1usize << fold.tau_pos.len();
    let block_packed = (1usize << layout.k_log) / bits_per_packed;
    let in_pos_base = (layout.input_byte_off * 8) / bits_per_packed;
    let out_pos_base = (layout.output_byte_off * 8) / bits_per_packed;
    assert_eq!(
        packed.len() % block_packed,
        0,
        "packed witness length must be a whole number of blocks"
    );
    let n_inst = packed.len() / block_packed;
    let n_log = n_inst.trailing_zeros() as usize;

    let eq_tau = build_eq_table(&fold.tau_pos);

    let word_addr = move |i: usize, w: usize| -> usize {
        match wl {
            flock_core::r1cs::WitnessLayout::RowMajor => i * block_packed + w,
            flock_core::r1cs::WitnessLayout::BatchMajor => (w << n_log) + i,
        }
    };

    let fold_one = |i: usize, base: usize| -> F128 {
        match wl {
            flock_core::r1cs::WitnessLayout::RowMajor => {
                let start = i * block_packed + base;
                flock_core::field::f128_slice::inner_product(
                    &eq_tau,
                    &packed[start..start + n_packed_per_region],
                )
            }
            flock_core::r1cs::WitnessLayout::BatchMajor => {
                use flock_core::field::gf2_128::F256Unreduced;
                let mut acc = F256Unreduced::ZERO;
                for pos in 0..n_packed_per_region {
                    acc ^= eq_tau[pos].mul_unreduced(packed[word_addr(i, base + pos)]);
                }
                acc.reduce()
            }
        }
    };

    let in_vals: Vec<F128> = (0..n_inst)
        .into_par_iter()
        .map(|i| fold_one(i, in_pos_base))
        .collect();
    let out_vals: Vec<F128> = (0..n_inst)
        .into_par_iter()
        .map(|i| fold_one(i, out_pos_base))
        .collect();

    (in_vals, out_vals)
}

pub fn assemble_chain_claim(
    layout: &ChainLayout,
    wl: flock_core::r1cs::WitnessLayout,
    fold: &ChainFold,
    claims: &crate::chain::ChainClaims,
) -> PackedDirectClaim {
    let point = build_chain_claim_point(layout, wl, fold, claims);
    let sparse_eq = flock_core::pcs::ring_switch::build_eq_sparse(&point);
    PackedDirectClaim {
        point,
        value: claims.value,
        eq_ind: DirectEqInd::Sparse(sparse_eq),
    }
}

fn build_chain_claim_point(
    layout: &ChainLayout,
    wl: flock_core::r1cs::WitnessLayout,
    fold: &ChainFold,
    claims: &crate::chain::ChainClaims,
) -> Vec<F128> {
    let high = layout.high_zeros();
    let point_len = fold.tau_pos.len() + 1 + high + claims.instance_point.len();
    let mut point = Vec::with_capacity(point_len);
    if wl == flock_core::r1cs::WitnessLayout::BatchMajor {
        point.extend_from_slice(&claims.instance_point);
    }
    point.extend_from_slice(&fold.tau_pos);
    point.push(claims.sel0);
    point.extend(std::iter::repeat_n(F128::ZERO, high));
    if wl == flock_core::r1cs::WitnessLayout::RowMajor {
        point.extend_from_slice(&claims.instance_point);
    }
    debug_assert_eq!(point.len(), point_len);
    point
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainProofLigerito {
    pub zerocheck: flock_core::zerocheck::ZerocheckProof,
    pub lincheck: flock_core::lincheck::LincheckProof,
    pub shift: crate::chain::ChainShiftProof,
    pub pcs_open: flock_core::pcs::BatchOpeningProofLigerito,
}

#[derive(Debug)]
pub enum ChainVerifyError {
    R1cs(flock_core::verifier::VerifyError),

    Shift(crate::chain::ChainError),

    Pcs(flock_core::pcs::VerifyError),
}

#[allow(clippy::too_many_arguments)]
pub fn prove_chain_ligerito_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    layout: &ChainLayout,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> (ChainProofLigerito, Commitment) {
    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito default config for chain prove; bump m for tiny instances");

    let core = crate::prover::prove_fast_core(
        r1cs,
        pcs_params,
        z_packed,
        a_packed,
        b_packed,
        z_lincheck,
        lincheck_circuit,
        challenger,
    );

    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = ChainFold::new(layout, tau_pos);
    let (in_vals, out_vals) = fold_in_out(layout, r1cs.layout, &core.z_packed, &fold);

    let (shift, claims) = crate::chain::prove_chain_shift(&in_vals, &out_vals, challenger);
    let chain_claim = assemble_chain_claim(layout, r1cs.layout, &fold, &claims);

    let padding = r1cs.padding_spec();
    let ab_x_outer = crate::prover::quirky_x_outer_full(&core.ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&core.c.point);

    let crate::prover::ProveCore {
        zc_proof,
        lc_proof,
        commitment,
        prover_data,
        z_packed,
        s_hat_v_ab,
        s_hat_v_c,
        ..
    } = core;
    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let pcs_open = flock_core::pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        &prover_data,
        &commitment,
        &[ab_x_outer.as_slice(), c_x_outer.as_slice()],
        &[pre_ab, pre_c],
        std::slice::from_ref(&chain_claim),
        &padding,
        &lig_config,
        challenger,
    );

    (
        ChainProofLigerito {
            zerocheck: zc_proof,
            lincheck: lc_proof,
            shift,
            pcs_open,
        },
        commitment,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_chain_ligerito_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    layout: &ChainLayout,
    commitment: &Commitment,
    proof: &ChainProofLigerito,
    n_log: usize,
    x0_phys: &[bool],
    xlast_phys: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(), ChainVerifyError> {
    let (ab, c) = flock_core::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )
    .map_err(ChainVerifyError::R1cs)?;

    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = ChainFold::new(layout, tau_pos);

    let x0_r = fold.fold_public_phys(x0_phys);
    let xlast_r = fold.fold_public_phys(xlast_phys);
    let claims = crate::chain::verify_chain_shift(&proof.shift, x0_r, xlast_r, n_log, challenger)
        .map_err(ChainVerifyError::Shift)?;

    let chain_point = build_chain_claim_point(layout, r1cs.layout, &fold, &claims);
    let ab_x_outer = crate::prover::quirky_x_outer_full(&ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&c.point);
    let pd_ref = PackedDirectClaimRef {
        point: &chain_point,
        value: claims.value,
    };

    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito default verifier config for chain verify");

    flock_core::pcs::verify_opening_batch_ligerito_mixed(
        commitment,
        &[ab.value, c.value],
        &[ab.point.z_skip, c.point.z_skip],
        &[ab_x_outer.as_slice(), c_x_outer.as_slice()],
        std::slice::from_ref(&pd_ref),
        &proof.pcs_open,
        &lig_v_config,
        challenger,
    )
    .map_err(ChainVerifyError::Pcs)?;

    Ok(())
}

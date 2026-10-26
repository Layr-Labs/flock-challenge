use crate::merkle_path::{MerklePathShiftProof, SlotLayout};
use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::pcs::{
    Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef, PcsParams,
};
use flock_core::r1cs::BlockR1cs;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct MerkleLayout {
    pub k_log: usize,

    pub k_skip: usize,

    pub region_log: usize,

    pub region_bits: usize,

    pub slot_base_byte_off: usize,

    pub z_slot: u8,

    pub x_l_slot: u8,

    pub x_r_slot: u8,
}

impl MerkleLayout {
    #[inline]
    pub fn tau_pos_len(&self) -> usize {
        self.region_log - LOG_PACKING
    }

    #[inline]
    pub fn high_zeros(&self) -> usize {
        self.k_log - self.region_log - 2
    }

    pub fn other_slot(&self) -> u8 {
        (0..4)
            .find(|&s| s != self.z_slot && s != self.x_l_slot && s != self.x_r_slot)
            .expect("3 of 4 slots are assigned; the fourth always exists")
    }

    pub fn slot_layout(&self) -> SlotLayout {
        SlotLayout {
            z_slot: self.z_slot,
            x_l_slot: self.x_l_slot,
            x_r_slot: self.x_r_slot,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MerklePathFold {
    pub tau_pos: Vec<F128>,
}

impl MerklePathFold {
    pub fn new(layout: &MerkleLayout, tau_pos: Vec<F128>) -> Self {
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

pub fn fold_all_slots(
    layout: &MerkleLayout,
    wl: flock_core::r1cs::WitnessLayout,
    packed: &[F128],
    fold: &MerklePathFold,
) -> [Vec<F128>; 4] {
    use rayon::prelude::*;

    let bits_per_packed = 1usize << LOG_PACKING;
    let n_packed_per_slot = 1usize << fold.tau_pos.len();
    let block_packed = (1usize << layout.k_log) / bits_per_packed;
    let slot_base_packed = (layout.slot_base_byte_off * 8) / bits_per_packed;
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
                    &packed[start..start + n_packed_per_slot],
                )
            }
            flock_core::r1cs::WitnessLayout::BatchMajor => {
                use flock_core::field::gf2_128::F256Unreduced;
                let mut acc = F256Unreduced::ZERO;
                for pos in 0..n_packed_per_slot {
                    acc ^= eq_tau[pos].mul_unreduced(packed[word_addr(i, base + pos)]);
                }
                acc.reduce()
            }
        }
    };

    let mut results: [Vec<F128>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (slot_idx, vec_out) in results.iter_mut().enumerate() {
        let slot_offset = slot_base_packed + slot_idx * n_packed_per_slot;
        *vec_out = (0..n_inst)
            .into_par_iter()
            .map(|i| fold_one(i, slot_offset))
            .collect();
    }
    results
}

pub fn assemble_merkle_path_claim(
    layout: &MerkleLayout,
    wl: flock_core::r1cs::WitnessLayout,
    fold: &MerklePathFold,
    claims: &crate::merkle_path::MerklePathClaims,
) -> PackedDirectClaim {
    let point = build_merkle_claim_point(layout, wl, fold, claims);
    let sparse_eq = flock_core::pcs::ring_switch::build_eq_sparse(&point);
    PackedDirectClaim {
        point,
        value: claims.value,
        eq_ind: DirectEqInd::Sparse(sparse_eq),
    }
}

fn build_merkle_claim_point(
    layout: &MerkleLayout,
    wl: flock_core::r1cs::WitnessLayout,
    fold: &MerklePathFold,
    claims: &crate::merkle_path::MerklePathClaims,
) -> Vec<F128> {
    let high = layout.high_zeros();
    let point_len = fold.tau_pos.len() + 2 + high + claims.instance_point.len();
    let mut point = Vec::with_capacity(point_len);
    if wl == flock_core::r1cs::WitnessLayout::BatchMajor {
        point.extend_from_slice(&claims.instance_point);
    }
    point.extend_from_slice(&fold.tau_pos);
    point.push(claims.sel_slot);
    point.push(claims.side);
    point.extend(std::iter::repeat_n(F128::ZERO, high));
    if wl == flock_core::r1cs::WitnessLayout::RowMajor {
        point.extend_from_slice(&claims.instance_point);
    }
    debug_assert_eq!(point.len(), point_len);
    point
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerklePathProofLigerito {
    pub zerocheck: flock_core::zerocheck::ZerocheckProof,
    pub lincheck: flock_core::lincheck::LincheckProof,
    pub shift: MerklePathShiftProof,
    pub pcs_open: flock_core::pcs::BatchOpeningProofLigerito,
}

#[derive(Debug)]
pub enum MerklePathVerifyError {
    R1cs(flock_core::verifier::VerifyError),

    Shift(crate::merkle_path::MerklePathError),

    Pcs(flock_core::pcs::VerifyError),
}

#[allow(clippy::too_many_arguments)]
pub fn prove_merkle_paths_ligerito_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    pcs_params: &PcsParams,
    layout: &MerkleLayout,
    path_log: usize,
    z_packed: Vec<F128>,
    a_packed: Vec<F128>,
    b_packed: Vec<F128>,
    z_lincheck: Vec<u8>,
    b_bits: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    challenger: &mut Ch,
) -> (MerklePathProofLigerito, Commitment) {
    let trace = std::env::var("MERKLE_TRACE").is_ok();

    let lig_config = pcs_params
        .ligerito_prover_config()
        .expect("Ligerito config for merkle-path prove; bump m for tiny instances");

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
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
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "base_r1cs (zc+lc)",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = MerklePathFold::new(layout, tau_pos);
    let slot_vals = fold_all_slots(layout, r1cs.layout, &core.z_packed, &fold);
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "fold_slots",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let x_l_vals = &slot_vals[layout.x_l_slot as usize];
    let x_r_vals = &slot_vals[layout.x_r_slot as usize];
    let z_vals = &slot_vals[layout.z_slot as usize];
    let iv_vals = &slot_vals[layout.other_slot() as usize];
    let (shift, claims) = crate::merkle_path::prove_merkle_path_shift(
        path_log,
        x_l_vals,
        x_r_vals,
        z_vals,
        iv_vals,
        b_bits,
        layout.slot_layout(),
        challenger,
    );
    let merkle_claim = assemble_merkle_path_claim(layout, r1cs.layout, &fold, &claims);
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "shift_sumcheck",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
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
        std::slice::from_ref(&merkle_claim),
        &padding,
        &lig_config,
        challenger,
    );
    if let Some(t) = t {
        eprintln!(
            "[merkle] {:<18} {:>8.2} ms",
            "open_batch",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (
        MerklePathProofLigerito {
            zerocheck: zc_proof,
            lincheck: lc_proof,
            shift,
            pcs_open,
        },
        commitment,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_merkle_paths_ligerito_generic<Ch: Challenger>(
    r1cs: &BlockR1cs,
    layout: &MerkleLayout,
    path_log: usize,
    commitment: &Commitment,
    proof: &MerklePathProofLigerito,
    n_log: usize,
    leaves_phys: &[&[bool]],
    root_phys: &[bool],
    b_bits: &[bool],
    lincheck_circuit: &dyn flock_core::lincheck::LincheckCircuit,
    pcs_params: &PcsParams,
    challenger: &mut Ch,
) -> Result<(), MerklePathVerifyError> {
    assert!(path_log <= n_log, "path_log must be ≤ n_log");
    let n_paths = 1usize << path_log;
    assert_eq!(
        leaves_phys.len(),
        n_paths,
        "leaves_phys must have length 2^path_log"
    );

    let (ab, c) = flock_core::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        commitment,
        lincheck_circuit,
        challenger,
    )
    .map_err(MerklePathVerifyError::R1cs)?;

    let tau_pos = challenger.sample_f128_vec(layout.tau_pos_len());
    let fold = MerklePathFold::new(layout, tau_pos);

    let leaf_evals: Vec<F128> = leaves_phys
        .iter()
        .map(|lp| fold.fold_public_phys(lp))
        .collect();
    let root_r = fold.fold_public_phys(root_phys);

    let claims = crate::merkle_path::verify_merkle_path_shift(
        path_log,
        &proof.shift,
        &leaf_evals,
        root_r,
        b_bits,
        n_log,
        layout.slot_layout(),
        challenger,
    )
    .map_err(MerklePathVerifyError::Shift)?;

    let merkle_point = build_merkle_claim_point(layout, r1cs.layout, &fold, &claims);
    let ab_x_outer = crate::prover::quirky_x_outer_full(&ab.point);
    let c_x_outer = crate::prover::quirky_x_outer_full(&c.point);
    let pd_ref = PackedDirectClaimRef {
        point: &merkle_point,
        value: claims.value,
    };

    let lig_v_config = pcs_params
        .ligerito_verifier_config()
        .expect("Ligerito verifier config for merkle-path verify");

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
    .map_err(MerklePathVerifyError::Pcs)?;

    Ok(())
}

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::zerocheck::multilinear::eq_eval;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct SlotLayout {
    pub z_slot: u8,

    pub x_l_slot: u8,

    pub x_r_slot: u8,
}

impl SlotLayout {
    pub fn validate(&self) {
        assert!(
            self.z_slot < 4 && self.x_l_slot < 4 && self.x_r_slot < 4,
            "slot out of range"
        );
        assert_ne!(self.z_slot, self.x_l_slot);
        assert_ne!(self.z_slot, self.x_r_slot);
        assert_ne!(self.x_l_slot, self.x_r_slot);
    }

    pub fn other_slot(&self) -> u8 {
        (0..4)
            .find(|&s| s != self.z_slot && s != self.x_l_slot && s != self.x_r_slot)
            .expect("must exist since 3 slots are assigned out of 4")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundMsg {
    pub q_1: F128,
    pub q_omega: F128,
    pub q_omega_plus_1: F128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerklePathShiftProof {
    pub rounds: Vec<RoundMsg>,
    pub g_at_point: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerklePathClaims {
    pub instance_point: Vec<F128>,
    pub sel_slot: F128,
    pub side: F128,
    pub value: F128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MerklePathError {
    MalformedProof,
    SumcheckFinal,
}

const OMEGA: F128 = F128 { lo: 2, hi: 0 };

#[inline]
fn omega_plus_1() -> F128 {
    OMEGA + F128::ONE
}

#[inline]
fn lagrange_eval_degree3(
    q_0: F128,
    q_1: F128,
    q_omega: F128,
    q_omega_plus_1: F128,
    r: F128,
) -> F128 {
    let one = F128::ONE;
    let omega = OMEGA;
    let opo = omega_plus_1();

    let n0 = (r + one) * (r + omega) * (r + opo);
    let d0 = omega * opo;
    let l0 = n0 * d0.inv();

    let n1 = r * (r + omega) * (r + opo);
    let d1 = (one + omega) * omega;
    let l1 = n1 * d1.inv();

    let n2 = r * (r + one) * (r + opo);
    let d2 = omega * opo;
    let l2 = n2 * d2.inv();

    let n3 = r * (r + one) * (r + omega);
    let d3 = opo * omega;
    let l3 = n3 * d3.inv();

    q_0 * l0 + q_1 * l1 + q_omega * l2 + q_omega_plus_1 * l3
}

#[inline]
fn slot_indicator(target_slot: u8, ss: F128, sd: F128) -> F128 {
    use flock_core::gf2_kernels::{gf2_kern_slot_indicator_hi, gf2_kern_slot_indicator_lo};
    let s = target_slot as u64;

    unsafe {
        F128 {
            lo: gf2_kern_slot_indicator_lo(s, ss.lo, ss.hi, sd.lo, sd.hi),
            hi: gf2_kern_slot_indicator_hi(s, ss.lo, ss.hi, sd.lo, sd.hi),
        }
    }
}

#[inline]
fn shift_mle(a: &[F128], b: &[F128]) -> F128 {
    crate::chain::shift_mle(a, b)
}

fn eval_bit_mle(b_bits: &[bool], r: &[F128]) -> F128 {
    let eq_r = build_eq_table(r);
    let mut acc = F128::ZERO;
    for (y, &bit) in b_bits.iter().enumerate() {
        if bit {
            acc += eq_r[y];
        }
    }
    acc
}

#[allow(clippy::too_many_arguments)]
pub fn prove_merkle_path_shift<Ch: Challenger>(
    path_log: usize,
    x_l_vals: &[F128],
    x_r_vals: &[F128],
    z_vals: &[F128],
    iv_vals: &[F128],
    b_bits: &[bool],
    layout: SlotLayout,
    challenger: &mut Ch,
) -> (MerklePathShiftProof, MerklePathClaims) {
    layout.validate();
    let n_total = z_vals.len();
    assert!(n_total.is_power_of_two(), "N must be a power of two");
    assert_eq!(x_l_vals.len(), n_total);
    assert_eq!(x_r_vals.len(), n_total);
    assert_eq!(iv_vals.len(), n_total);
    assert_eq!(b_bits.len(), n_total);
    let n = n_total.trailing_zeros() as usize;
    assert!(path_log <= n, "path_log must be ≤ n");
    let pos_log = n - path_log;
    let n_paths = 1usize << path_log;
    let n_pos = 1usize << pos_log;

    let tau = challenger.sample_f128_vec(n);
    let alpha = challenger.sample_f128();

    let tau_q = &tau[..pos_log];
    let tau_p = &tau[pos_log..n];

    let eq_tau = build_eq_table(&tau);
    let eq_tau_p = build_eq_table(tau_p);
    let eq_tau_q = build_eq_table(tau_q);

    let mut t_shift_alpha = vec![F128::ZERO; n_total];
    for i_p in 0..n_paths {
        let weight_p = eq_tau_p[i_p];
        let row_base = i_p << pos_log;
        for i_q in 0..n_pos {
            let shift_iq = if i_q == 0 {
                F128::ZERO
            } else {
                eq_tau_q[i_q - 1]
            };
            let leaf_iq = if i_q == 0 { alpha } else { F128::ZERO };
            t_shift_alpha[row_base | i_q] = weight_p * (shift_iq + leaf_iq);
        }
    }
    let mut t_eq = eq_tau.clone();
    let mut t_b: Vec<F128> = b_bits
        .iter()
        .map(|&b| if b { F128::ONE } else { F128::ZERO })
        .collect();

    for i_p in 0..n_paths {
        t_b[i_p << pos_log] = F128::ZERO;
    }

    let mut g_z = z_vals.to_vec();
    let mut g_xl = x_l_vals.to_vec();
    let mut g_xr = x_r_vals.to_vec();
    let mut g_other = iv_vals.to_vec();

    let mut rounds: Vec<RoundMsg> = Vec::with_capacity(n + 2);
    let mut r_pts: Vec<F128> = Vec::with_capacity(n + 2);
    let eval_pts = [F128::ONE, OMEGA, omega_plus_1()];

    for _round_idx in 0..n {
        let half = t_b.len() / 2;
        let mut sums = [F128::ZERO; 3];

        for (e, &xx) in eval_pts.iter().enumerate() {
            use flock_core::field::gf2_128::F256Unreduced;
            let mut acc_unred = F256Unreduced::ZERO;
            for i in 0..half {
                let ta = t_shift_alpha[i].mle_fold(t_shift_alpha[i + half], xx);
                let te = t_eq[i].mle_fold(t_eq[i + half], xx);
                let tb = t_b[i].mle_fold(t_b[i + half], xx);

                let gz = g_z[i].mle_fold(g_z[i + half], xx);
                let gxl = g_xl[i].mle_fold(g_xl[i + half], xx);
                let gxr = g_xr[i].mle_fold(g_xr[i + half], xx);
                let g_sel = gxl.mle_fold(gxr, tb);

                acc_unred ^= te.mul_unreduced(gz) ^ ta.mul_unreduced(g_sel);
            }
            sums[e] = acc_unred.reduce();
        }

        let msg = RoundMsg {
            q_1: sums[0],
            q_omega: sums[1],
            q_omega_plus_1: sums[2],
        };
        challenger.observe_f128(msg.q_1);
        challenger.observe_f128(msg.q_omega);
        challenger.observe_f128(msg.q_omega_plus_1);
        let r = challenger.sample_f128();
        rounds.push(msg);
        r_pts.push(r);

        for i in 0..half {
            t_shift_alpha[i] = t_shift_alpha[i].mle_fold(t_shift_alpha[i + half], r);
            t_eq[i] = t_eq[i].mle_fold(t_eq[i + half], r);
            t_b[i] = t_b[i].mle_fold(t_b[i + half], r);
            g_z[i] = g_z[i].mle_fold(g_z[i + half], r);
            g_xl[i] = g_xl[i].mle_fold(g_xl[i + half], r);
            g_xr[i] = g_xr[i].mle_fold(g_xr[i + half], r);
            g_other[i] = g_other[i].mle_fold(g_other[i + half], r);
        }
        t_shift_alpha.truncate(half);
        t_eq.truncate(half);
        t_b.truncate(half);
        g_z.truncate(half);
        g_xl.truncate(half);
        g_xr.truncate(half);
        g_other.truncate(half);
    }

    let ta = t_shift_alpha[0];
    let te = t_eq[0];
    let tb = t_b[0];

    let mut w_table = [F128::ZERO; 4];
    let mut g_table = [F128::ZERO; 4];
    let z_slot = layout.z_slot as usize;
    let xl_slot = layout.x_l_slot as usize;
    let xr_slot = layout.x_r_slot as usize;
    let other_slot = layout.other_slot() as usize;
    w_table[z_slot] = te;
    w_table[xl_slot] = unsafe {
        F128 {
            lo: flock_core::gf2_kernels::gf2_kern_eq_bit_mul_lo(0, ta.lo, ta.hi, tb.lo, tb.hi),
            hi: flock_core::gf2_kernels::gf2_kern_eq_bit_mul_hi(0, ta.lo, ta.hi, tb.lo, tb.hi),
        }
    };
    w_table[xr_slot] = unsafe {
        F128 {
            lo: flock_core::gf2_kernels::gf2_kern_eq_bit_mul_lo(1, ta.lo, ta.hi, tb.lo, tb.hi),
            hi: flock_core::gf2_kernels::gf2_kern_eq_bit_mul_hi(1, ta.lo, ta.hi, tb.lo, tb.hi),
        }
    };

    g_table[z_slot] = g_z[0];
    g_table[xl_slot] = g_xl[0];
    g_table[xr_slot] = g_xr[0];
    g_table[other_slot] = g_other[0];

    let mut w_vec = w_table.to_vec();
    let mut g_vec = g_table.to_vec();

    for _round_idx in 0..2 {
        let half = w_vec.len() / 2;
        let mut sums = [F128::ZERO; 3];
        for (e, &xx) in eval_pts.iter().enumerate() {
            use flock_core::field::gf2_128::F256Unreduced;
            let mut acc_unred = F256Unreduced::ZERO;
            for i in 0..half {
                let w_at_xx = w_vec[i].mle_fold(w_vec[i + half], xx);
                let g_at_xx = g_vec[i].mle_fold(g_vec[i + half], xx);
                acc_unred ^= w_at_xx.mul_unreduced(g_at_xx);
            }
            sums[e] = acc_unred.reduce();
        }
        let msg = RoundMsg {
            q_1: sums[0],
            q_omega: sums[1],
            q_omega_plus_1: sums[2],
        };
        challenger.observe_f128(msg.q_1);
        challenger.observe_f128(msg.q_omega);
        challenger.observe_f128(msg.q_omega_plus_1);
        let r = challenger.sample_f128();
        rounds.push(msg);
        r_pts.push(r);
        for i in 0..half {
            w_vec[i] = w_vec[i].mle_fold(w_vec[i + half], r);
            g_vec[i] = g_vec[i].mle_fold(g_vec[i + half], r);
        }
        w_vec.truncate(half);
        g_vec.truncate(half);
    }

    let g_at_point = g_vec[0];

    let mut instance_point = vec![F128::ZERO; n];
    for j in 0..n {
        instance_point[j] = r_pts[n - 1 - j];
    }
    let side = r_pts[n];
    let sel_slot = r_pts[n + 1];

    (
        MerklePathShiftProof { rounds, g_at_point },
        MerklePathClaims {
            instance_point,
            sel_slot,
            side,
            value: g_at_point,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_merkle_path_shift<Ch: Challenger>(
    path_log: usize,
    proof: &MerklePathShiftProof,
    leaf_evals: &[F128],
    root_r: F128,
    b_bits: &[bool],
    n: usize,
    layout: SlotLayout,
    challenger: &mut Ch,
) -> Result<MerklePathClaims, MerklePathError> {
    layout.validate();
    let d = n + 2;
    if proof.rounds.len() != d {
        return Err(MerklePathError::MalformedProof);
    }
    assert!(path_log <= n, "path_log must be ≤ n");
    let pos_log = n - path_log;
    let n_paths = 1usize << path_log;
    assert_eq!(b_bits.len(), 1usize << n);
    assert_eq!(
        leaf_evals.len(),
        n_paths,
        "leaf_evals must have length 2^path_log"
    );

    let tau = challenger.sample_f128_vec(n);
    let alpha = challenger.sample_f128();
    let tau_q = &tau[..pos_log];
    let tau_p = &tau[pos_log..n];

    let eq_tau_q_ones = tau_q.iter().copied().fold(F128::ONE, |acc, t| acc * t);
    let eq_tau_p_table = build_eq_table(tau_p);
    let combined_leaf: F128 = eq_tau_p_table
        .iter()
        .zip(leaf_evals.iter())
        .map(|(&w, &v)| w * v)
        .fold(F128::ZERO, |a, b| a + b);
    let mut claim = eq_tau_q_ones * root_r + alpha * combined_leaf;

    let mut r_pts: Vec<F128> = Vec::with_capacity(d);
    for msg in &proof.rounds {
        challenger.observe_f128(msg.q_1);
        challenger.observe_f128(msg.q_omega);
        challenger.observe_f128(msg.q_omega_plus_1);
        let r = challenger.sample_f128();

        let q_0 = claim + msg.q_1;

        claim = lagrange_eval_degree3(q_0, msg.q_1, msg.q_omega, msg.q_omega_plus_1, r);
        r_pts.push(r);
    }

    let mut instance_point = vec![F128::ZERO; n];
    for j in 0..n {
        instance_point[j] = r_pts[n - 1 - j];
    }
    let side = r_pts[n];
    let sel_slot = r_pts[n + 1];

    let tau_y_q = &instance_point[..pos_log];
    let tau_y_p = &instance_point[pos_log..n];
    let eq_taup_tauyp = eq_eval(tau_p, tau_y_p);
    let shift_q = shift_mle(tau_q, tau_y_q);

    let eq_tauyq_zero = tau_y_q
        .iter()
        .copied()
        .fold(F128::ONE, |acc, t| acc * (F128::ONE + t));
    let t_shift_alpha = eq_taup_tauyp * (shift_q + alpha * eq_tauyq_zero);
    let t_eq = eq_eval(&tau, &instance_point);

    let mut b_local = b_bits.to_vec();
    for i_p in 0..n_paths {
        b_local[i_p << pos_log] = false;
    }
    let t_b = eval_bit_mle(&b_local, &instance_point);
    let one_plus_t_b = F128::ONE + t_b;

    let w_z_contrib = slot_indicator(layout.z_slot, sel_slot, side) * t_eq;
    let w_xl_contrib =
        slot_indicator(layout.x_l_slot, sel_slot, side) * t_shift_alpha * one_plus_t_b;
    let w_xr_contrib = slot_indicator(layout.x_r_slot, sel_slot, side) * t_shift_alpha * t_b;
    let w_final = w_z_contrib + w_xl_contrib + w_xr_contrib;

    if claim != w_final * proof.g_at_point {
        return Err(MerklePathError::SumcheckFinal);
    }

    Ok(MerklePathClaims {
        instance_point,
        sel_slot,
        side,
        value: proof.g_at_point,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::FsChallenger;

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
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn bit(&mut self) -> bool {
            self.next_u64() & 1 == 1
        }
    }

    fn build_honest_scenario(
        n: usize,
        seed: u64,
    ) -> (
        Vec<F128>,
        Vec<F128>,
        Vec<F128>,
        Vec<F128>,
        Vec<bool>,
        F128,
        F128,
    ) {
        let mut rng = Rng::new(seed);
        let k = 1usize << n;

        let z_vals: Vec<F128> = (0..k).map(|_| rng.f128()).collect();

        let mut b_bits = vec![false; k];
        for i in 1..k {
            b_bits[i] = rng.bit();
        }
        let mut x_l = vec![F128::ZERO; k];
        let mut x_r = vec![F128::ZERO; k];

        let leaf = rng.f128();
        x_l[0] = leaf;
        x_r[0] = rng.f128();

        for i in 1..k {
            if !b_bits[i] {
                x_l[i] = z_vals[i - 1];
                x_r[i] = rng.f128();
            } else {
                x_r[i] = z_vals[i - 1];
                x_l[i] = rng.f128();
            }
        }
        let iv: Vec<F128> = (0..k).map(|_| rng.f128()).collect();
        let root = z_vals[k - 1];
        (x_l, x_r, z_vals, iv, b_bits, leaf, root)
    }

    fn canonical_layout() -> SlotLayout {
        SlotLayout {
            z_slot: 0,
            x_l_slot: 1,
            x_r_slot: 2,
        }
    }

    #[test]
    fn honest_roundtrip_accepts() {
        for &n in &[3usize, 4, 5] {
            let (x_l, x_r, z, iv, b, leaf, root) = build_honest_scenario(n, 0xC0FFEE + n as u64);
            let layout = canonical_layout();

            let mut ch_p = FsChallenger::new(b"merkle-test-v0");
            let (proof, claims_p) =
                prove_merkle_path_shift(0, &x_l, &x_r, &z, &iv, &b, layout, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"merkle-test-v0");
            let claims_v =
                verify_merkle_path_shift(0, &proof, &[leaf], root, &b, n, layout, &mut ch_v)
                    .unwrap_or_else(|e| panic!("verify rejected honest proof at n={n}: {e:?}"));

            assert_eq!(claims_v, claims_p, "claims must match at n={n}");
        }
    }

    #[test]
    fn broken_chain_rejects() {
        let n = 4;
        let (mut x_l, x_r, z, iv, b, leaf, root) = build_honest_scenario(n, 0xBAD);

        for i in 1..(1 << n) {
            if !b[i] {
                x_l[i] += F128::ONE;
                break;
            }
        }
        let layout = canonical_layout();
        let mut ch_p = FsChallenger::new(b"merkle-test-v0");
        let (proof, _) = prove_merkle_path_shift(0, &x_l, &x_r, &z, &iv, &b, layout, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"merkle-test-v0");
        let res = verify_merkle_path_shift(0, &proof, &[leaf], root, &b, n, layout, &mut ch_v);
        assert!(matches!(res, Err(MerklePathError::SumcheckFinal)));
    }

    #[test]
    fn wrong_leaf_rejects() {
        let n = 4;
        let (x_l, x_r, z, iv, b, leaf, root) = build_honest_scenario(n, 0xC0DE);
        let layout = canonical_layout();
        let mut ch_p = FsChallenger::new(b"merkle-test-v0");
        let (proof, _) = prove_merkle_path_shift(0, &x_l, &x_r, &z, &iv, &b, layout, &mut ch_p);

        let mut bad_leaf = leaf;
        bad_leaf.lo ^= 1;
        let mut ch_v = FsChallenger::new(b"merkle-test-v0");
        let res = verify_merkle_path_shift(0, &proof, &[bad_leaf], root, &b, n, layout, &mut ch_v);
        assert!(matches!(res, Err(MerklePathError::SumcheckFinal)));
    }

    #[test]
    fn wrong_root_rejects() {
        let n = 4;
        let (x_l, x_r, z, iv, b, leaf, root) = build_honest_scenario(n, 0xDEAD);
        let layout = canonical_layout();
        let mut ch_p = FsChallenger::new(b"merkle-test-v0");
        let (proof, _) = prove_merkle_path_shift(0, &x_l, &x_r, &z, &iv, &b, layout, &mut ch_p);
        let mut bad_root = root;
        bad_root.lo ^= 1;
        let mut ch_v = FsChallenger::new(b"merkle-test-v0");
        let res = verify_merkle_path_shift(0, &proof, &[leaf], bad_root, &b, n, layout, &mut ch_v);
        assert!(matches!(res, Err(MerklePathError::SumcheckFinal)));
    }
}

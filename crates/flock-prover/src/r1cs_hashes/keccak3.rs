use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{Commitment, PcsParams};
use flock_core::proof::R1csClaim;
use flock_core::r1cs::BlockR1cs;
use flock_core::verifier;

use super::keccak::{
    LANE_BITS, Lanes, N_LANES, N_ROUNDS, N_T, ROUND_CONSTANTS, STATE_BITS, STATE_SIZE_BITS, State,
    apply_phi_bool, apply_phi_t, iota_lanes, rho_pi_lanes, state_idx, state_to_lanes, theta_lanes,
    theta_rho_pi_preimage,
};

pub const K_LOG: usize = 17;
pub const K: usize = 1 << K_LOG;

pub const K_SKIP: usize = 6;

pub const N_SUB: usize = 3;

pub const SLOT_BITS: usize = 2048;

#[inline]
fn state0_bit_base(i: usize) -> usize {
    (2 * i) * SLOT_BITS
}
#[inline]
fn state24_bit_base(i: usize) -> usize {
    (2 * i + 1) * SLOT_BITS
}

pub const Z_CONST: usize = 2 * N_SUB * SLOT_BITS;
const Z_CONST_U64: usize = Z_CONST / LANE_BITS;
pub const T_PACKED_BIT_BASE: usize = Z_CONST + LANE_BITS;

pub const USEFUL_BITS: usize = T_PACKED_BIT_BASE + N_SUB * N_T * STATE_SIZE_BITS;

pub const U64_PER_BLOCK: usize = K / 64;

const _: () = {
    assert!(USEFUL_BITS <= K);
};

#[inline]
fn within_lane_contiguous(j: usize) -> usize {
    let lane_xy = j % N_LANES;
    let z_in_lane = j / N_LANES;
    LANE_BITS * lane_xy + z_in_lane
}

#[inline]
fn state_u64_base(i: usize, r: usize) -> usize {
    match r {
        0 => state0_bit_base(i) / LANE_BITS,
        24 => state24_bit_base(i) / LANE_BITS,
        _ => panic!("only state_0 and state_24 are materialized, got r={r}"),
    }
}

#[inline]
pub fn z_pos_state(i: usize, r: usize, j: usize) -> usize {
    debug_assert!(j < STATE_BITS);
    let base = match r {
        0 => state0_bit_base(i),
        24 => state24_bit_base(i),
        _ => panic!("only state_0 and state_24 are materialized, got r={r}"),
    };
    base + within_lane_contiguous(j)
}

#[inline]
fn t_u64_base(i: usize, r: usize) -> usize {
    debug_assert!(i < N_SUB && r < N_T);
    (T_PACKED_BIT_BASE / LANE_BITS) + (i * N_T + r) * N_LANES
}

#[inline]
pub fn z_pos_t(i: usize, r: usize, j: usize) -> usize {
    debug_assert!(i < N_SUB && r < N_T && j < STATE_BITS);
    T_PACKED_BIT_BASE + (i * N_T + r) * STATE_SIZE_BITS + within_lane_contiguous(j)
}

pub fn min_n_blocks_log(n_keccaks: usize) -> usize {
    assert!(n_keccaks >= 1, "n_keccaks must be ≥ 1");
    let n_blocks = n_keccaks.div_ceil(N_SUB);
    let n = n_blocks.max(8);
    (n.next_power_of_two().trailing_zeros() as usize).max(3)
}

pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
    super::common::build_block_r1cs_empty_stub(n_blocks_log, K_LOG, K_SKIP, USEFUL_BITS)
}

fn fill_subkeccak(
    i: usize,
    initial: &State,
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    let mut state_lanes: Lanes = state_to_lanes(initial);
    let s0_base = state_u64_base(i, 0);
    for lane in 0..N_LANES {
        let pos = s0_base + lane;
        let v = state_lanes[lane];
        z_u64[pos] = v;
        a_u64[pos] = v;
        b_u64[pos] = u64::MAX;
    }

    for r in 0..N_ROUNDS {
        let mut b_state: Lanes = state_lanes;
        theta_lanes(&mut b_state);
        let b_state: Lanes = rho_pi_lanes(&b_state);

        let mut t_lanes: Lanes = [0u64; 25];
        let mut next: Lanes = [0u64; 25];
        for y in 0..5 {
            let b0 = b_state[5 * y];
            let b1 = b_state[1 + 5 * y];
            let b2 = b_state[2 + 5 * y];
            let b3 = b_state[3 + 5 * y];
            let b4 = b_state[4 + 5 * y];
            unsafe {
                t_lanes[5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b1, b2);
                t_lanes[1 + 5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b2, b3);
                t_lanes[2 + 5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b3, b4);
                t_lanes[3 + 5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b4, b0);
                t_lanes[4 + 5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b0, b1);
                next[5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi(b0, b1, b2);
                next[1 + 5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi(b1, b2, b3);
                next[2 + 5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi(b2, b3, b4);
                next[3 + 5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi(b3, b4, b0);
                next[4 + 5 * y] = flock_core::gf2_kernels::gf2_kern_keccak_chi(b4, b0, b1);
            }
        }
        iota_lanes(&mut next, r);

        let t_base = t_u64_base(i, r);
        for y in 0..5 {
            for x in 0..5 {
                let lane = x + 5 * y;
                let pos = t_base + lane;
                z_u64[pos] = t_lanes[lane];
                a_u64[pos] = !b_state[(x + 1) % 5 + 5 * y];
                b_u64[pos] = b_state[(x + 2) % 5 + 5 * y];
            }
        }

        state_lanes = next;
    }

    let s24_base = state_u64_base(i, 24);
    for lane in 0..N_LANES {
        let pos = s24_base + lane;
        let v = state_lanes[lane];
        z_u64[pos] = v;
        a_u64[pos] = v;
        b_u64[pos] = u64::MAX;
    }
}

fn build_block_witness_into(
    triple: &[State; N_SUB],
    z_u64: &mut [u64],
    a_u64: &mut [u64],
    b_u64: &mut [u64],
) {
    debug_assert_eq!(z_u64.len(), U64_PER_BLOCK);
    debug_assert_eq!(a_u64.len(), U64_PER_BLOCK);
    debug_assert_eq!(b_u64.len(), U64_PER_BLOCK);

    z_u64[Z_CONST_U64] = 1;
    a_u64[Z_CONST_U64] = 1;
    b_u64[Z_CONST_U64] = 1;

    for i in 0..N_SUB {
        fill_subkeccak(i, &triple[i], z_u64, a_u64, b_u64);
    }
}

pub fn generate_witness_with_ab_packed_and_lincheck(
    initial_states: &[State],
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    let n_blocks = initial_states.len().div_ceil(N_SUB);
    let zero: State = [false; STATE_BITS];
    let triples: Vec<[State; N_SUB]> = (0..n_blocks)
        .map(|blk| {
            let mut t = [zero; N_SUB];
            for (i, slot) in t.iter_mut().enumerate() {
                let idx = blk * N_SUB + i;
                if idx < initial_states.len() {
                    *slot = initial_states[idx];
                }
            }
            t
        })
        .collect();

    let padding: [State; N_SUB] = [zero; N_SUB];

    super::common::drive_witness_packed_and_lincheck(
        &triples,
        Some(&padding),
        n_blocks_log,
        K_LOG,
        build_block_witness_into,
    )
}

fn accumulate_subkeccak(i: usize, alpha: F128, eq_inner: &[F128], comb: &mut [F128]) {
    for j in 0..STATE_BITS {
        let row = z_pos_state(i, 0, j);
        let e = eq_inner[row];
        comb[row] = alpha.fma(e, comb[row]);
        comb[Z_CONST] += e;
    }

    let mut vec_pin: Vec<F128> = vec![F128::ZERO; STATE_BITS];
    let mut sum_eq_pin = F128::ZERO;
    for j in 0..STATE_BITS {
        let row = z_pos_state(i, 24, j);
        let e = eq_inner[row];
        vec_pin[j] = e;
        sum_eq_pin += e;
    }
    comb[Z_CONST] += sum_eq_pin; // B-side from pin's B = [Z_CONST]

    use rayon::prelude::*;
    let chi: Vec<(Vec<F128>, Vec<F128>, F128)> = (0..N_T)
        .into_par_iter()
        .map(|r| {
            let mut ca = vec![F128::ZERO; STATE_BITS];
            let mut cb = vec![F128::ZERO; STATE_BITS];
            let mut se = F128::ZERO;
            for zpos in 0..64 {
                for y in 0..5 {
                    for x in 0..5 {
                        let row = z_pos_t(i, r, state_idx(x, y, zpos));
                        let e = eq_inner[row];
                        se += e;
                        for &s in theta_rho_pi_preimage((x + 1) % 5, y, zpos).iter() {
                            ca[s] += e;
                        }
                        for &s in theta_rho_pi_preimage((x + 2) % 5, y, zpos).iter() {
                            cb[s] += e;
                        }
                    }
                }
            }
            (ca, cb, se)
        })
        .collect();
    let mut chi_a: Vec<Vec<F128>> = Vec::with_capacity(N_T);
    let mut chi_b: Vec<Vec<F128>> = Vec::with_capacity(N_T);
    let mut sum_eq_t = F128::ZERO;
    for (ca, cb, se) in chi {
        chi_a.push(ca);
        chi_b.push(cb);
        sum_eq_t += se;
    }
    comb[Z_CONST] += alpha * sum_eq_t;

    let mut rc = [false; STATE_BITS];
    let mut rc_a = F128::ZERO;
    let mut rc_b = F128::ZERO;
    for r in 0..N_T {
        for s in 0..STATE_BITS {
            if rc[s] {
                rc_a += chi_a[r][s];
                rc_b += chi_b[r][s];
            }
        }
        rc = apply_phi_bool(&rc);
        for zpos in 0..64 {
            if (ROUND_CONSTANTS[r] >> zpos) & 1 == 1 {
                let s = state_idx(0, 0, zpos);
                rc[s] ^= true;
            }
        }
    }
    let mut rc_pin = F128::ZERO;
    for s in 0..STATE_BITS {
        if rc[s] {
            rc_pin += vec_pin[s];
        }
    }
    comb[Z_CONST] += alpha * rc_a;
    comb[Z_CONST] += rc_b;
    comb[Z_CONST] += alpha * rc_pin;

    let t23_base = t_u64_base(i, N_T - 1);
    for s in 0..STATE_BITS {
        let pos = (t23_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
        comb[pos] = alpha.fma(vec_pin[s], comb[pos]);
    }
    let mut k_a = apply_phi_t(&vec_pin);
    flock_core::field::f128_slice::xor_assign(&mut k_a, &chi_a[N_T - 1]);
    for r in (1..N_T).rev() {
        let t_base = t_u64_base(i, r - 1);
        for s in 0..STATE_BITS {
            let pos = (t_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
            comb[pos] = alpha.fma(k_a[s], comb[pos]);
        }
        let mut new_k = apply_phi_t(&k_a);
        flock_core::field::f128_slice::xor_assign(&mut new_k, &chi_a[r - 1]);
        k_a = new_k;
    }
    let s0_base = state_u64_base(i, 0);
    for s in 0..STATE_BITS {
        let pos = (s0_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
        comb[pos] = alpha.fma(k_a[s], comb[pos]);
    }

    let mut k_b = chi_b[N_T - 1].clone();
    for r in (1..N_T).rev() {
        let t_base = t_u64_base(i, r - 1);
        for s in 0..STATE_BITS {
            let pos = (t_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
            comb[pos] += k_b[s];
        }
        let mut new_k = apply_phi_t(&k_b);
        flock_core::field::f128_slice::xor_assign(&mut new_k, &chi_b[r - 1]);
        k_b = new_k;
    }
    for s in 0..STATE_BITS {
        let pos = (s0_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
        comb[pos] += k_b[s];
    }
}

pub struct KeccakLincheckCircuit;

impl LincheckCircuit for KeccakLincheckCircuit {
    fn n_cols(&self) -> usize {
        K
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(Z_CONST)
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        use rayon::prelude::*;
        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");

        let mut combs: Vec<Vec<F128>> = (0..N_SUB)
            .into_par_iter()
            .map(|i| {
                let mut comb = vec![F128::ZERO; K];
                accumulate_subkeccak(i, alpha, eq_inner, &mut comb);
                comb
            })
            .collect();

        let mut comb = combs.swap_remove(0);
        for other in &combs {
            comb.par_chunks_mut(1 << 13)
                .zip(other.par_chunks(1 << 13))
                .for_each(|(dst, src)| {
                    flock_core::field::f128_slice::xor_assign(dst, src);
                });
        }

        let e0 = eq_inner[Z_CONST];
        comb[Z_CONST] += alpha * e0;
        comb[Z_CONST] += e0;

        comb
    }
}

#[derive(Clone, Debug)]
pub struct KeccakSetup {
    pub n_keccaks: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: PcsParams,
}

impl KeccakSetup {
    pub fn new(n_keccaks: usize) -> Self {
        Self::with_log_inv_rate(n_keccaks, 1)
    }

    pub fn with_log_inv_rate(n_keccaks: usize, log_inv_rate: usize) -> Self {
        let profile = match log_inv_rate {
            1 => flock_core::pcs::ligerito::LigeritoProfile::Fast,
            2 => flock_core::pcs::ligerito::LigeritoProfile::Slim,
            _ => flock_core::pcs::ligerito::LigeritoProfile::Fast,
        };
        Self::with_profile_and_rate(n_keccaks, profile, log_inv_rate)
    }

    pub fn with_profile(
        n_keccaks: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
    ) -> Self {
        Self::with_profile_and_rate(n_keccaks, profile, profile.log_inv_rate())
    }

    fn with_profile_and_rate(
        n_keccaks: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
        log_inv_rate: usize,
    ) -> Self {
        assert!(n_keccaks >= 1);
        let n_blocks_log = min_n_blocks_log(n_keccaks);
        let r1cs = build_block_r1cs(n_blocks_log);
        let pcs_params = PcsParams {
            m: r1cs.m,
            log_inv_rate,
            log_batch_size: 6,
            profile,
            merkle_hash: Default::default(),
        };
        Self {
            n_keccaks,
            r1cs,
            pcs_params,
        }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_blocks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
    pub fn n_block_slots(&self) -> usize {
        1usize << self.n_blocks_log()
    }

    pub fn prove_fast<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (flock_core::proof::R1csProofLigerito, Commitment, R1csClaim) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        let (codeword, (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                generate_witness_with_ab_packed_and_lincheck(initial_states, self.n_blocks_log())
            });
        crate::prover::prove_fast_ligerito_from_witness(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            &KeccakLincheckCircuit,
            codeword,
            challenger,
        )
    }

    pub fn verify<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &flock_core::proof::R1csProofLigerito,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, verifier::VerifyError> {
        verifier::verify_ligerito(
            &self.r1cs,
            commitment,
            proof,
            &KeccakLincheckCircuit,
            &self.pcs_params,
            challenger,
        )
    }

    pub fn prove_fast_timed<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        Commitment,
        R1csClaim,
        crate::prover::ProvePhaseTimings,
    ) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        let t0 = std::time::Instant::now();
        let (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck) =
            generate_witness_with_ab_packed_and_lincheck(initial_states, self.n_blocks_log());
        let witness_s = t0.elapsed().as_secs_f64();
        let (proof, commitment, claim, mut timings) = crate::prover::prove_fast_ligerito_timed(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            &KeccakLincheckCircuit,
            None,
            challenger,
        );
        timings.witness_s = witness_s;
        (proof, commitment, claim, timings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs_hashes::keccak::{keccak_f, lanes_to_state};

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
    }

    fn random_state(rng: &mut Rng) -> State {
        let mut s = [false; STATE_BITS];
        let mut i = 0;
        while i < STATE_BITS {
            let w = rng.next_u64();
            for b in 0..64 {
                if i + b < STATE_BITS {
                    s[i + b] = (w >> b) & 1 == 1;
                }
            }
            i += 64;
        }
        s
    }

    #[test]
    fn layout_constants_consistent() {
        assert_eq!(K, 131_072);
        assert_eq!(USEFUL_BITS, 12_352 + 3 * 24 * 1600);
        assert_eq!(USEFUL_BITS, 127_552);
        assert!(USEFUL_BITS <= K);
        assert_eq!(Z_CONST, 12_288);
        assert_eq!(Z_CONST_U64, 192);
        assert_eq!(T_PACKED_BIT_BASE, 12_352);
        assert_eq!(U64_PER_BLOCK, 2048);
        assert_eq!(state_u64_base(0, 0), 0);
        assert_eq!(state_u64_base(0, 24), 32);
        assert_eq!(state_u64_base(2, 24), 160);
        assert!(state_u64_base(2, 24) + N_LANES <= Z_CONST_U64);
        assert_eq!(t_u64_base(0, 0), 193);
        assert!(t_u64_base(N_SUB - 1, N_T - 1) + N_LANES <= U64_PER_BLOCK);
    }

    #[test]
    fn witness_layout_round_trip() {
        let mut rng = Rng::new(0xCAB1E_FA17);
        let triple: [State; N_SUB] = [
            random_state(&mut rng),
            random_state(&mut rng),
            random_state(&mut rng),
        ];

        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        build_block_witness_into(&triple, &mut z_u64, &mut a_u64, &mut b_u64);

        assert_eq!(z_u64[Z_CONST_U64] & 1, 1);

        for i in 0..N_SUB {
            let mut s0_lanes: Lanes = [0u64; 25];
            let s0_base = state_u64_base(i, 0);
            for lane in 0..N_LANES {
                s0_lanes[lane] = z_u64[s0_base + lane];
            }
            assert_eq!(lanes_to_state(&s0_lanes), triple[i], "sub {i} state_0");

            let mut s24_lanes: Lanes = [0u64; 25];
            let s24_base = state_u64_base(i, 24);
            for lane in 0..N_LANES {
                s24_lanes[lane] = z_u64[s24_base + lane];
            }
            let mut native = triple[i];
            keccak_f(&mut native);
            assert_eq!(
                lanes_to_state(&s24_lanes),
                native,
                "sub {i} state_24 != keccak_f(state_0)"
            );
        }

        for pos in 0..U64_PER_BLOCK {
            assert_eq!(
                a_u64[pos] & b_u64[pos],
                z_u64[pos],
                "a·b ≠ z at u64 pos {pos}"
            );
        }
    }

    #[test]
    fn fold_matches_witness_consistency() {
        let mut rng = Rng::new(0xFA7E_C001_BABE);
        let triple: [State; N_SUB] = [
            random_state(&mut rng),
            random_state(&mut rng),
            random_state(&mut rng),
        ];

        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        build_block_witness_into(&triple, &mut z_u64, &mut a_u64, &mut b_u64);

        let alpha = F128 {
            lo: rng.next_u64(),
            hi: rng.next_u64(),
        };
        let eq_inner: Vec<F128> = (0..K)
            .map(|_| F128 {
                lo: rng.next_u64(),
                hi: rng.next_u64(),
            })
            .collect();

        let mut v_a = F128::ZERO;
        let mut v_b = F128::ZERO;
        for i in 0..K {
            let u = i / 64;
            let bit = i % 64;
            if (a_u64[u] >> bit) & 1 == 1 {
                v_a += eq_inner[i];
            }
            if (b_u64[u] >> bit) & 1 == 1 {
                v_b += eq_inner[i];
            }
        }
        let expected = alpha * v_a + v_b;

        let comb = KeccakLincheckCircuit.fold_alpha_batched(alpha, &eq_inner);

        let mut got = F128::ZERO;
        for c in 0..K {
            let u = c / 64;
            let bit = c % 64;
            if (z_u64[u] >> bit) & 1 == 1 {
                got += comb[c];
            }
        }

        assert_eq!(got, expected, "fold consistency: comb·z ≠ α·v_a + v_b");
    }

    #[test]
    fn setup_sizes_correctly() {
        for &(n, blk_log) in &[(1usize, 3), (24, 3), (25, 4), (4096, 11)] {
            let setup = KeccakSetup::new(n);
            assert_eq!(setup.n_blocks_log(), blk_log, "n={n}");
            assert_eq!(setup.m(), K_LOG + blk_log, "n={n}");
        }
    }

    #[test]
    #[ignore = "timing comparison; run manually with --nocapture"]
    fn pack_win() {
        use crate::r1cs_hashes::keccak as k1;
        use flock_core::challenger::FsChallenger;
        use std::time::Instant;

        let n = 6144usize;
        let mut rng = Rng::new(0xC0FFEE_BEEF);
        let inputs: Vec<State> = (0..n).map(|_| random_state(&mut rng)).collect();

        let s1 = k1::KeccakSetup::new(n);
        let s3 = KeccakSetup::new(n);
        println!(
            "n={n}: single m={} (2^{} bits)  |  3-wide m={} (2^{} bits)",
            s1.m(),
            s1.m(),
            s3.m(),
            s3.m()
        );

        let mut t = Instant::now();
        let mut ch = FsChallenger::new(b"pack-bench");
        let (p1, _, _) = s1.prove_fast(&inputs, &mut ch);
        let single = t.elapsed().as_secs_f64();
        std::hint::black_box(&p1);

        t = Instant::now();
        let mut ch = FsChallenger::new(b"pack-bench");
        let (p3, _, _) = s3.prove_fast(&inputs, &mut ch);
        let wide = t.elapsed().as_secs_f64();
        std::hint::black_box(&p3);

        println!(
            "prove_fast: single={:.1} ms  3-wide={:.1} ms  ({:.2}x)",
            single * 1e3,
            wide * 1e3,
            single / wide
        );
    }

    #[test]
    #[ignore]
    fn prove_fast_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let n_keccaks = 49;
        let mut rng = Rng::new(0x21111_2170);
        let inputs: Vec<State> = (0..n_keccaks).map(|_| random_state(&mut rng)).collect();
        let setup = KeccakSetup::new(n_keccaks);
        let mut ch_p = FsChallenger::new(b"k3-lig-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"k3-lig-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("ligerito verify rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    #[test]
    #[ignore]
    fn prove_fast_timed_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let n_keccaks = 49;
        let mut rng = Rng::new(0x21111_2171);
        let inputs: Vec<State> = (0..n_keccaks).map(|_| random_state(&mut rng)).collect();
        let setup = KeccakSetup::new(n_keccaks);
        let mut ch_p = FsChallenger::new(b"k3-lig-v0");
        let (proof, commitment, claim_p, t) = setup.prove_fast_timed(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"k3-lig-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("timed ligerito verify rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
        assert!(t.witness_s > 0.0 && t.commit_s > 0.0 && t.zerocheck_s > 0.0);
        assert!(t.lincheck_s > 0.0 && t.open_s > 0.0);
    }

    #[test]
    fn prove_fast_roundtrip() {
        use flock_core::challenger::FsChallenger;

        let n_keccaks = 49;
        let mut rng = Rng::new(0xFAB1E_F011);
        let inputs: Vec<State> = (0..n_keccaks).map(|_| random_state(&mut rng)).collect();

        let setup = KeccakSetup::new(n_keccaks);
        let mut ch_p = FsChallenger::new(b"keccak3-test-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);

        let mut ch_v = FsChallenger::new(b"keccak3-test-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("prove_fast: verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    #[test]
    fn all_zero_witness_rejected() {
        use crate::r1cs_hashes::keccak::keccak_f;
        use flock_core::challenger::FsChallenger;

        let n_keccaks = 49;
        let setup = KeccakSetup::new(n_keccaks);

        let mut s0 = [false; STATE_BITS];
        keccak_f(&mut s0);
        assert!(s0.iter().any(|&b| b), "keccak_f(0) must be nonzero");

        let inputs: Vec<State> = vec![[false; STATE_BITS]; n_keccaks];
        let (mut z, mut a, mut b, mut zlc) =
            generate_witness_with_ab_packed_and_lincheck(&inputs, setup.n_blocks_log());
        z.iter_mut().for_each(|v| *v = F128::ZERO);
        a.iter_mut().for_each(|v| *v = F128::ZERO);
        b.iter_mut().for_each(|v| *v = F128::ZERO);
        zlc.iter_mut().for_each(|v| *v = 0);

        let mut ch_p = FsChallenger::new(b"poc");
        let (proof, commitment, _claim) = crate::prover::prove_fast_ligerito_from_witness(
            &setup.r1cs,
            &setup.pcs_params,
            z,
            a,
            b,
            zlc,
            &KeccakLincheckCircuit,
            None,
            &mut ch_p,
        );

        let mut ch_v = FsChallenger::new(b"poc");
        let res = setup.verify(&commitment, &proof, &mut ch_v);
        assert!(
            matches!(res, Err(verifier::VerifyError::Lincheck(_))),
            "all-zero witness must be rejected by the constant-wire pin; got {res:?}"
        );
    }
}

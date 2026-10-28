use crate::r1cs_hashes::chain_common::{ChainLayout, ChainVerifyError};
use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use flock_core::pcs::{Commitment, PcsParams};
use flock_core::proof::R1csClaim;
use flock_core::r1cs::BlockR1cs;
use flock_core::verifier;

pub const STATE_BITS: usize = 1600;

pub const N_LANES: usize = 25;
pub const LANE_BITS: usize = 64;
pub const STATE_SIZE_BITS: usize = N_LANES * LANE_BITS;
pub const N_T: usize = 24;
pub const N_ROUNDS: usize = 24;
pub const K_SKIP: usize = 6;

const RHO_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

pub const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

#[inline]
pub fn state_idx(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < 5 && y < 5 && z < 64);
    x + 5 * y + 25 * z
}

pub fn theta_rho_pi_preimage(x: usize, y: usize, z: usize) -> [usize; 11] {
    let a = (x + 3 * y) % 5;
    let b = x;
    let r = (RHO_OFFSETS[a][b] as usize) % 64;
    let c = (z + 64 - r) % 64;
    let c_prev = (c + 63) % 64;
    let a_minus = (a + 4) % 5;
    let a_plus = (a + 1) % 5;

    let mut bits = [0usize; 11];
    bits[0] = state_idx(a, b, c);
    for yp in 0..5 {
        bits[1 + yp] = state_idx(a_minus, yp, c);
    }
    for yp in 0..5 {
        bits[6 + yp] = state_idx(a_plus, yp, c_prev);
    }
    bits
}

pub type State = [bool; STATE_BITS];

pub type Lanes = [u64; 25];

#[inline]
fn lane_idx(x: usize, y: usize) -> usize {
    debug_assert!(x < 5 && y < 5);
    x + 5 * y
}

pub fn state_to_lanes(s: &State) -> Lanes {
    let mut lanes = [0u64; 25];
    for z in 0..64 {
        for y in 0..5 {
            for x in 0..5 {
                if s[state_idx(x, y, z)] {
                    lanes[lane_idx(x, y)] |= 1u64 << z;
                }
            }
        }
    }
    lanes
}

pub fn lanes_to_state(lanes: &Lanes) -> State {
    let mut s = [false; STATE_BITS];
    for y in 0..5 {
        for x in 0..5 {
            let lane = lanes[lane_idx(x, y)];
            for z in 0..64 {
                s[state_idx(x, y, z)] = (lane >> z) & 1 == 1;
            }
        }
    }
    s
}

#[inline]
pub fn theta_lanes(s: &mut Lanes) {
    let c = [
        s[lane_idx(0, 0)]
            ^ s[lane_idx(0, 1)]
            ^ s[lane_idx(0, 2)]
            ^ s[lane_idx(0, 3)]
            ^ s[lane_idx(0, 4)],
        s[lane_idx(1, 0)]
            ^ s[lane_idx(1, 1)]
            ^ s[lane_idx(1, 2)]
            ^ s[lane_idx(1, 3)]
            ^ s[lane_idx(1, 4)],
        s[lane_idx(2, 0)]
            ^ s[lane_idx(2, 1)]
            ^ s[lane_idx(2, 2)]
            ^ s[lane_idx(2, 3)]
            ^ s[lane_idx(2, 4)],
        s[lane_idx(3, 0)]
            ^ s[lane_idx(3, 1)]
            ^ s[lane_idx(3, 2)]
            ^ s[lane_idx(3, 3)]
            ^ s[lane_idx(3, 4)],
        s[lane_idx(4, 0)]
            ^ s[lane_idx(4, 1)]
            ^ s[lane_idx(4, 2)]
            ^ s[lane_idx(4, 3)]
            ^ s[lane_idx(4, 4)],
    ];
    let d = [
        unsafe { flock_core::gf2_kernels::gf2_kern_keccak_theta_d(c[4], c[1]) },
        unsafe { flock_core::gf2_kernels::gf2_kern_keccak_theta_d(c[0], c[2]) },
        unsafe { flock_core::gf2_kernels::gf2_kern_keccak_theta_d(c[1], c[3]) },
        unsafe { flock_core::gf2_kernels::gf2_kern_keccak_theta_d(c[2], c[4]) },
        unsafe { flock_core::gf2_kernels::gf2_kern_keccak_theta_d(c[3], c[0]) },
    ];
    for y in 0..5 {
        for x in 0..5 {
            s[lane_idx(x, y)] ^= d[x];
        }
    }
}

#[inline]
pub fn rho_pi_lanes(s_in: &Lanes) -> Lanes {
    let mut s_out = [0u64; 25];
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            let b = x;
            let r = (RHO_OFFSETS[a][b] % 64) as u64;
            s_out[lane_idx(x, y)] =
                unsafe { flock_core::gf2_kernels::gf2_kern_keccak_rol64(s_in[lane_idx(a, b)], r) };
        }
    }
    s_out
}

#[inline]
pub fn chi_lanes(s_in: &Lanes) -> Lanes {
    let mut s_out = [0u64; 25];
    for y in 0..5 {
        let a0 = s_in[lane_idx(0, y)];
        let a1 = s_in[lane_idx(1, y)];
        let a2 = s_in[lane_idx(2, y)];
        let a3 = s_in[lane_idx(3, y)];
        let a4 = s_in[lane_idx(4, y)];
        s_out[lane_idx(0, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi(a0, a1, a2) };
        s_out[lane_idx(1, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi(a1, a2, a3) };
        s_out[lane_idx(2, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi(a2, a3, a4) };
        s_out[lane_idx(3, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi(a3, a4, a0) };
        s_out[lane_idx(4, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi(a4, a0, a1) };
    }
    s_out
}

#[inline]
pub fn iota_lanes(s: &mut Lanes, round_idx: usize) {
    s[lane_idx(0, 0)] ^= ROUND_CONSTANTS[round_idx];
}

#[inline]
pub fn keccak_round_lanes(s: &mut Lanes, round_idx: usize) {
    theta_lanes(s);
    *s = rho_pi_lanes(s);
    *s = chi_lanes(s);
    iota_lanes(s, round_idx);
}

#[inline]
pub fn compute_round_t_lanes(state_in: &Lanes) -> Lanes {
    let mut b = *state_in;
    theta_lanes(&mut b);
    let b = rho_pi_lanes(&b);
    let mut t = [0u64; 25];
    for y in 0..5 {
        let b1 = b[lane_idx(1, y)];
        let b2 = b[lane_idx(2, y)];
        let b3 = b[lane_idx(3, y)];
        let b4 = b[lane_idx(4, y)];
        let b0 = b[lane_idx(0, y)];
        t[lane_idx(0, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b1, b2) };
        t[lane_idx(1, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b2, b3) };
        t[lane_idx(2, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b3, b4) };
        t[lane_idx(3, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b4, b0) };
        t[lane_idx(4, y)] = unsafe { flock_core::gf2_kernels::gf2_kern_keccak_chi_and(b0, b1) };
    }
    t
}

pub fn theta(s: &mut State) {
    let mut c = [[false; 64]; 5];
    for x in 0..5 {
        for z in 0..64 {
            let mut acc = false;
            for y in 0..5 {
                acc ^= s[state_idx(x, y, z)];
            }
            c[x][z] = acc;
        }
    }
    let mut d = [[false; 64]; 5];
    for x in 0..5 {
        for z in 0..64 {
            d[x][z] = c[(x + 4) % 5][z] ^ c[(x + 1) % 5][(z + 63) % 64];
        }
    }
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                s[state_idx(x, y, z)] ^= d[x][z];
            }
        }
    }
}

pub fn rho_pi(s_in: &State) -> State {
    let mut s_out = [false; STATE_BITS];
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                let a = (x + 3 * y) % 5;
                let b = x;
                let r = (RHO_OFFSETS[a][b] as usize) % 64;
                let c = (z + 64 - r) % 64;
                s_out[state_idx(x, y, z)] = s_in[state_idx(a, b, c)];
            }
        }
    }
    s_out
}

pub fn chi(s_in: &State) -> State {
    let mut s_out = [false; STATE_BITS];
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                let a = s_in[state_idx(x, y, z)];
                let b = s_in[state_idx((x + 1) % 5, y, z)];
                let c = s_in[state_idx((x + 2) % 5, y, z)];
                s_out[state_idx(x, y, z)] = a ^ ((!b) & c);
            }
        }
    }
    s_out
}

pub fn iota(s: &mut State, round_idx: usize) {
    let rc = ROUND_CONSTANTS[round_idx];
    for z in 0..64 {
        if (rc >> z) & 1 != 0 {
            s[state_idx(0, 0, z)] ^= true;
        }
    }
}

pub fn keccak_round(s: &mut State, round_idx: usize) {
    let mut lanes = state_to_lanes(s);
    keccak_round_lanes(&mut lanes, round_idx);
    *s = lanes_to_state(&lanes);
}

pub fn keccak_f(s: &mut State) {
    for r in 0..24 {
        keccak_round(s, r);
    }
}

pub const K_LOG: usize = 16;
pub const K: usize = 1 << K_LOG;

pub const SLOT_BITS: usize = 2048;

pub const STATE0_BIT_BASE: usize = 0;
pub const STATE24_BIT_BASE: usize = SLOT_BITS;
pub const Z_CONST: usize = 2 * SLOT_BITS;
pub const T_PACKED_BIT_BASE: usize = Z_CONST + LANE_BITS;

pub const USEFUL_BITS: usize = T_PACKED_BIT_BASE + N_T * STATE_SIZE_BITS;

const _: () = {
    assert!(USEFUL_BITS <= K);
};

pub const U64_PER_BLOCK: usize = K / 64;

#[inline]
fn state_u64_base(r: usize) -> usize {
    match r {
        0 => STATE0_BIT_BASE / LANE_BITS,
        24 => STATE24_BIT_BASE / LANE_BITS,
        _ => panic!("keccak_chain only materializes state_0 and state_24, got r={r}"),
    }
}

#[inline]
fn t_u64_base(r: usize) -> usize {
    debug_assert!(r < N_T);
    (T_PACKED_BIT_BASE / LANE_BITS) + r * N_LANES
}

const Z_CONST_U64: usize = Z_CONST / LANE_BITS;

#[inline]
fn within_lane_contiguous(j: usize) -> usize {
    let lane_xy = j % N_LANES;
    let z_in_lane = j / N_LANES;
    LANE_BITS * lane_xy + z_in_lane
}

#[inline]
pub fn z_pos_state(r: usize, j: usize) -> usize {
    debug_assert!(j < STATE_BITS);
    match r {
        0 => STATE0_BIT_BASE + within_lane_contiguous(j),
        24 => STATE24_BIT_BASE + within_lane_contiguous(j),
        _ => panic!("keccak_chain only materializes state_0 and state_24, got r={r}"),
    }
}

#[inline]
pub fn z_pos_t(r: usize, j: usize) -> usize {
    debug_assert!(r < N_T);
    debug_assert!(j < STATE_BITS);
    T_PACKED_BIT_BASE + r * STATE_SIZE_BITS + within_lane_contiguous(j)
}

pub fn min_n_keccaks_log(n_keccaks: usize) -> usize {
    assert!(n_keccaks >= 1, "n_keccaks must be ≥ 1");
    let n = n_keccaks.max(8);
    let bits = n.next_power_of_two().trailing_zeros() as usize;
    bits.max(3)
}

pub(crate) fn apply_phi_t(v: &[F128]) -> Vec<F128> {
    debug_assert_eq!(v.len(), STATE_BITS);
    let mut out = vec![F128::ZERO; STATE_BITS];
    for s_out in 0..STATE_BITS {
        let z_p = s_out / N_LANES;
        let xy = s_out % N_LANES;
        let y_p = xy / 5;
        let x_p = xy % 5;
        let preim = theta_rho_pi_preimage(x_p, y_p, z_p);
        let val = v[s_out];
        for &s_in in preim.iter() {
            out[s_in] += val;
        }
    }
    out
}

pub(crate) fn apply_phi_bool(v: &[bool; STATE_BITS]) -> [bool; STATE_BITS] {
    let mut out = [false; STATE_BITS];
    for s_out in 0..STATE_BITS {
        let z_p = s_out / N_LANES;
        let xy = s_out % N_LANES;
        let y_p = xy / 5;
        let x_p = xy % 5;
        let preim = theta_rho_pi_preimage(x_p, y_p, z_p);
        let mut bit = false;
        for &s_in in preim.iter() {
            bit ^= v[s_in];
        }
        out[s_out] = bit;
    }
    out
}

pub fn build_block_r1cs(n_keccaks_log: usize) -> BlockR1cs {
    super::common::build_block_r1cs_empty_stub(n_keccaks_log, K_LOG, K_SKIP, USEFUL_BITS)
}

fn build_chain_witness_ab_packed_into(
    initial: &State,
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

    let mut state_lanes: Lanes = state_to_lanes(initial);
    let s0_base = state_u64_base(0);
    for lane_idx in 0..N_LANES {
        let pos = s0_base + lane_idx;
        let v = state_lanes[lane_idx];
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

        let t_base = t_u64_base(r);
        for y in 0..5 {
            for x in 0..5 {
                let lane_idx = x + 5 * y;
                let pos = t_base + lane_idx;
                z_u64[pos] = t_lanes[lane_idx];
                a_u64[pos] = !b_state[(x + 1) % 5 + 5 * y];
                b_u64[pos] = b_state[(x + 2) % 5 + 5 * y];
            }
        }

        state_lanes = next;
    }

    let s24_base = state_u64_base(24);
    for lane_idx in 0..N_LANES {
        let pos = s24_base + lane_idx;
        let v = state_lanes[lane_idx];
        z_u64[pos] = v;
        a_u64[pos] = v;
        b_u64[pos] = u64::MAX;
    }
}

pub fn generate_witness_with_ab_packed_and_lincheck(
    initial_states: &[State],
    n_keccaks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    let padding: State = [false; STATE_BITS];
    super::common::drive_witness_packed_and_lincheck(
        initial_states,
        Some(&padding),
        n_keccaks_log,
        K_LOG,
        build_chain_witness_ab_packed_into,
    )
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
        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");
        let mut comb = vec![F128::ZERO; K];

        let e0 = eq_inner[Z_CONST];
        comb[Z_CONST] = alpha.fma(e0, comb[Z_CONST] + e0);

        for j in 0..STATE_BITS {
            let row = z_pos_state(0, j);
            let e = eq_inner[row];
            comb[row] = alpha.fma(e, comb[row]);
            comb[Z_CONST] += e;
        }

        let mut vec_pin: Vec<F128> = vec![F128::ZERO; STATE_BITS];
        let mut sum_eq_pin = F128::ZERO;
        for j in 0..STATE_BITS {
            let row = z_pos_state(24, j);
            let e = eq_inner[row];
            vec_pin[j] = e;
            sum_eq_pin += e;
        }
        comb[Z_CONST] += sum_eq_pin; // B-side from pin's B = [Z_CONST]

        let mut chi_a: Vec<Vec<F128>> = (0..N_T).map(|_| vec![F128::ZERO; STATE_BITS]).collect();
        let mut chi_b: Vec<Vec<F128>> = (0..N_T).map(|_| vec![F128::ZERO; STATE_BITS]).collect();
        let mut sum_eq_t = F128::ZERO;

        for r in 0..N_T {
            for zpos in 0..64 {
                for y in 0..5 {
                    for x in 0..5 {
                        let row = z_pos_t(r, state_idx(x, y, zpos));
                        let e = eq_inner[row];
                        sum_eq_t += e;
                        for &s in theta_rho_pi_preimage((x + 1) % 5, y, zpos).iter() {
                            chi_a[r][s] += e;
                        }
                        for &s in theta_rho_pi_preimage((x + 2) % 5, y, zpos).iter() {
                            chi_b[r][s] += e;
                        }
                    }
                }
            }
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

        let t23_base = t_u64_base(N_T - 1);
        for s in 0..STATE_BITS {
            let pos = (t23_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
            comb[pos] = alpha.fma(vec_pin[s], comb[pos]);
        }
        let mut k_a = apply_phi_t(&vec_pin);
        flock_core::field::f128_slice::xor_assign(&mut k_a, &chi_a[N_T - 1]);
        for r in (1..N_T).rev() {
            let t_base = t_u64_base(r - 1);
            for s in 0..STATE_BITS {
                let pos = (t_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
                comb[pos] = alpha.fma(k_a[s], comb[pos]);
            }
            let mut new_k = apply_phi_t(&k_a);
            flock_core::field::f128_slice::xor_assign(&mut new_k, &chi_a[r - 1]);
            k_a = new_k;
        }
        let s0_base = state_u64_base(0);
        for s in 0..STATE_BITS {
            let pos = (s0_base + s % N_LANES) * LANE_BITS + (s / N_LANES);
            comb[pos] = alpha.fma(k_a[s], comb[pos]);
        }

        let mut k_b = chi_b[N_T - 1].clone();
        for r in (1..N_T).rev() {
            let t_base = t_u64_base(r - 1);
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

        comb
    }
}

pub const CHAIN_LAYOUT: ChainLayout = ChainLayout {
    k_log: K_LOG,
    k_skip: K_SKIP,
    region_log: 11,
    region_bits: STATE_BITS,
    input_byte_off: STATE0_BIT_BASE / 8,
    output_byte_off: STATE24_BIT_BASE / 8,
};

pub fn state_to_phys_bits(x: &State) -> Vec<bool> {
    let mut phys = vec![false; STATE_BITS];
    for j in 0..STATE_BITS {
        phys[within_lane_contiguous(j)] = x[j];
    }
    phys
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
        let n_log = min_n_keccaks_log(n_keccaks);
        let r1cs = build_block_r1cs(n_log);
        flock_core::scratch::prewarm_prover(r1cs.m);
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

    pub fn new_batch_major(n_keccaks: usize) -> Self {
        let mut s = Self::new(n_keccaks);
        s.r1cs.layout = flock_core::r1cs::WitnessLayout::BatchMajor;
        s
    }

    fn generate_witness(
        &self,
        initial_states: &[State],
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        match self.r1cs.layout {
            flock_core::r1cs::WitnessLayout::RowMajor => {
                generate_witness_with_ab_packed_and_lincheck(initial_states, self.n_keccaks_log())
            }
            flock_core::r1cs::WitnessLayout::BatchMajor => {
                generate_witness_batch_major(initial_states, self.n_keccaks_log())
            }
        }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_keccaks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
    pub fn n_keccak_slots(&self) -> usize {
        1usize << self.n_keccaks_log()
    }

    pub fn prove_fast<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (flock_core::proof::R1csProofLigerito, Commitment, R1csClaim) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        let (codeword, (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                self.generate_witness(initial_states)
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

    pub fn prove_chain<Ch: Challenger>(
        &self,
        initial_states: &[State],
        challenger: &mut Ch,
    ) -> (
        crate::r1cs_hashes::chain_common::ChainProofLigerito,
        Commitment,
    ) {
        assert_eq!(initial_states.len(), self.n_keccaks);
        assert_eq!(self.n_keccaks, self.n_keccak_slots());
        let (z_packed, a_packed, b_packed, z_lincheck) = self.generate_witness(initial_states);
        crate::r1cs_hashes::chain_common::prove_chain_ligerito_generic(
            &self.r1cs,
            &self.pcs_params,
            &CHAIN_LAYOUT,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            &KeccakLincheckCircuit,
            challenger,
        )
    }

    pub fn verify_chain<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &crate::r1cs_hashes::chain_common::ChainProofLigerito,
        x_0: &State,
        x_last: &State,
        challenger: &mut Ch,
    ) -> Result<(), ChainVerifyError> {
        assert_eq!(self.n_keccaks, self.n_keccak_slots());
        let n_log = self.n_keccaks_log();
        let x0_phys = state_to_phys_bits(x_0);
        let xlast_phys = state_to_phys_bits(x_last);
        crate::r1cs_hashes::chain_common::verify_chain_ligerito_generic(
            &self.r1cs,
            &CHAIN_LAYOUT,
            commitment,
            proof,
            n_log,
            &x0_phys,
            &xlast_phys,
            &KeccakLincheckCircuit,
            &self.pcs_params,
            challenger,
        )
    }
}

use super::common::{BM_V, SendPtr, nt_store_row};

type VLane = [u64; BM_V];
type VLanes = [[u64; BM_V]; N_LANES];

#[inline(always)]
fn theta_v(s: &mut VLanes) {
    let mut c = [[0u64; BM_V]; 5];
    for x in 0..5 {
        for y in 0..5 {
            for j in 0..BM_V {
                c[x][j] ^= s[x + 5 * y][j];
            }
        }
    }
    let mut d = [[0u64; BM_V]; 5];
    for x in 0..5 {
        for j in 0..BM_V {
            d[x][j] = unsafe {
                flock_core::gf2_kernels::gf2_kern_keccak_theta_d(
                    c[(x + 4) % 5][j],
                    c[(x + 1) % 5][j],
                )
            };
        }
    }
    for i in 0..N_LANES {
        let x = i % 5;
        for j in 0..BM_V {
            s[i][j] ^= d[x][j];
        }
    }
}

#[inline(always)]
fn rho_pi_v(s_in: &VLanes) -> VLanes {
    let mut out = [[0u64; BM_V]; N_LANES];
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            let b = x;
            let r = (RHO_OFFSETS[a][b] % 64) as u64;
            for j in 0..BM_V {
                out[x + 5 * y][j] = unsafe {
                    flock_core::gf2_kernels::gf2_kern_keccak_rol64(s_in[a + 5 * b][j], r)
                };
            }
        }
    }
    out
}

struct RowWriter {
    dest: *mut u64,
    o0_x2: usize,
    n_log: usize,
    pending_w: usize,
    pending: VLane,
}

impl RowWriter {
    fn new(dest: *mut u64, o0: usize, n_log: usize) -> Self {
        Self {
            dest,
            o0_x2: o0 * 2,
            n_log,
            pending_w: usize::MAX,
            pending: [0u64; BM_V],
        }
    }

    #[inline(always)]
    unsafe fn emit(&mut self, c: usize, even: &VLane, odd: &VLane) {
        let mut buf = [0u64; 2 * BM_V];
        for j in 0..BM_V {
            buf[2 * j] = even[j];
            buf[2 * j + 1] = odd[j];
        }
        unsafe {
            nt_store_row(
                buf.as_ptr(),
                self.dest.add((c << self.n_log) * 2 + self.o0_x2),
            );
        }
    }

    #[inline(always)]
    unsafe fn push(&mut self, w: usize, vals: &VLane) {
        if w & 1 == 1 {
            debug_assert_eq!(self.pending_w, w - 1, "odd half without its even partner");
            let pending = self.pending;
            self.pending_w = usize::MAX;
            unsafe { self.emit(w >> 1, &pending, vals) };
        } else {
            unsafe { self.flush() };
            self.pending_w = w;
            self.pending = *vals;
        }
    }

    #[inline(always)]
    unsafe fn flush(&mut self) {
        if self.pending_w != usize::MAX {
            let (w, pending) = (self.pending_w, self.pending);
            self.pending_w = usize::MAX;
            unsafe { self.emit(w >> 1, &pending, &[0u64; BM_V]) };
        }
    }
}

unsafe fn build_group_batch_major(
    states: [&State; BM_V],
    o0: usize,
    n_log: usize,
    z: *mut u64,
    a: *mut u64,
    b: *mut u64,
    stripe: *mut u8,
) {
    use flock_core::bits::transpose_8_u64s_to_64_bytes;

    let mut wz = RowWriter::new(z, o0, n_log);
    let mut wa = RowWriter::new(a, o0, n_log);
    let mut wb = RowWriter::new(b, o0, n_log);
    let ones: VLane = [u64::MAX; BM_V];
    let one: VLane = [1u64; BM_V];
    let zeros: VLane = [0u64; BM_V];

    let stripe_base = unsafe { stripe.add((o0 / 8) * U64_PER_BLOCK * 64) };
    macro_rules! push_z {
        ($w:expr, $vals:expr) => {{
            let w: usize = $w;
            let vals: &VLane = $vals;
            let out = std::slice::from_raw_parts_mut(stripe_base.add(w * 64), 64);
            transpose_8_u64s_to_64_bytes(vals, out);
            wz.push(w, vals);
        }};
    }

    let mut lanes: VLanes = [[0u64; BM_V]; N_LANES];
    for (j, s) in states.iter().enumerate() {
        let l = state_to_lanes(s);
        for i in 0..N_LANES {
            lanes[i][j] = l[i];
        }
    }

    let s0_w = state_u64_base(0);
    let s24_w = state_u64_base(24);
    unsafe {
        for i in 0..N_LANES {
            push_z!(s0_w + i, &lanes[i]);
            wa.push(s0_w + i, &lanes[i]);
            wb.push(s0_w + i, &ones);
        }
        wz.flush();
        wa.flush();
        wb.flush();

        for w in (s0_w + N_LANES + 1)..s24_w {
            push_z!(w, &zeros);
            wa.push(w, &zeros);
            wb.push(w, &zeros);
        }
        wz.flush();
        wa.flush();
        wb.flush();

        push_z!(Z_CONST_U64, &one);
        wa.push(Z_CONST_U64, &one);
        wb.push(Z_CONST_U64, &one);

        for r in 0..N_ROUNDS {
            let mut b_state = lanes;
            theta_v(&mut b_state);
            let b_state = rho_pi_v(&b_state);

            let mut t = [[0u64; BM_V]; N_LANES];
            let mut next = [[0u64; BM_V]; N_LANES];
            for y in 0..5 {
                for x in 0..5 {
                    let i = x + 5 * y;
                    let i1 = (x + 1) % 5 + 5 * y;
                    let i2 = (x + 2) % 5 + 5 * y;
                    for j in 0..BM_V {
                        t[i][j] = flock_core::gf2_kernels::gf2_kern_keccak_chi_and(
                            b_state[i1][j],
                            b_state[i2][j],
                        );
                        next[i][j] = b_state[i][j] ^ t[i][j];
                    }
                }
            }
            for j in 0..BM_V {
                next[0][j] ^= ROUND_CONSTANTS[r];
            }

            let t_base = t_u64_base(r);
            for y in 0..5 {
                for x in 0..5 {
                    let i = x + 5 * y;
                    let w = t_base + i;
                    push_z!(w, &t[i]);
                    let mut av = [0u64; BM_V];
                    let i1 = (x + 1) % 5 + 5 * y;
                    for j in 0..BM_V {
                        av[j] = !b_state[i1][j];
                    }
                    wa.push(w, &av);
                    wb.push(w, &b_state[(x + 2) % 5 + 5 * y]);
                }
            }

            lanes = next;
        }
        wz.flush();
        wa.flush();
        wb.flush();

        for i in 0..N_LANES {
            push_z!(s24_w + i, &lanes[i]);
            wa.push(s24_w + i, &lanes[i]);
            wb.push(s24_w + i, &ones);
        }
        wz.flush();
        wa.flush();
        wb.flush();

        for w in (s24_w + N_LANES + 1)..Z_CONST_U64 {
            push_z!(w, &zeros);
            wa.push(w, &zeros);
            wb.push(w, &zeros);
        }
        wz.flush();
        wa.flush();
        wb.flush();
    }
}

pub fn generate_witness_batch_major(
    initial_states: &[State],
    n_keccaks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    use rayon::prelude::*;

    let n_total = 1usize << n_keccaks_log;
    assert!(initial_states.len() <= n_total);
    assert!(n_total >= BM_V, "batch-major needs n_total >= 8");
    let useful_chunks = USEFUL_BITS.div_ceil(128);
    let total_f128 = n_total * (U64_PER_BLOCK / 2);

    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let stripe = vec![0u8; n_total * U64_PER_BLOCK * 8];
    let tail = useful_chunks << n_keccaks_log;
    for buf in [&mut z, &mut a, &mut b] {
        buf[tail..]
            .par_chunks_mut(1 << 16)
            .for_each(|c| c.fill(F128::ZERO));
    }

    let (zp, ap, bp) = (
        SendPtr(z.as_mut_ptr() as *mut u64),
        SendPtr(a.as_mut_ptr() as *mut u64),
        SendPtr(b.as_mut_ptr() as *mut u64),
    );
    let sp = SendPtr(stripe.as_ptr() as *mut u64);
    let padding: State = [false; STATE_BITS];
    let states_ref = initial_states;
    let padding_ref = &padding;

    (0..n_total / BM_V).into_par_iter().for_each(move |g| {
        let o0 = g * BM_V;
        let group: [&State; BM_V] =
            std::array::from_fn(|j| states_ref.get(o0 + j).unwrap_or(padding_ref));
        unsafe {
            build_group_batch_major(
                group,
                o0,
                n_keccaks_log,
                zp.get(),
                ap.get(),
                bp.get(),
                sp.get() as *mut u8,
            )
        }
    });

    (z, a, b, stripe)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn keccak_f_u64(a: &mut [[u64; 5]; 5]) {
        for round in 0..24 {
            let mut c = [0u64; 5];
            for x in 0..5 {
                c[x] = a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4];
            }
            let mut d = [0u64; 5];
            for x in 0..5 {
                d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            }
            for x in 0..5 {
                for y in 0..5 {
                    a[x][y] ^= d[x];
                }
            }
            let mut b = [[0u64; 5]; 5];
            for x in 0..5 {
                for y in 0..5 {
                    b[y][(2 * x + 3 * y) % 5] = a[x][y].rotate_left(RHO_OFFSETS[x][y]);
                }
            }
            for x in 0..5 {
                for y in 0..5 {
                    a[x][y] = b[x][y] ^ ((!b[(x + 1) % 5][y]) & b[(x + 2) % 5][y]);
                }
            }
            a[0][0] ^= ROUND_CONSTANTS[round];
        }
    }

    fn bit_state_to_lanes(s: &State) -> [[u64; 5]; 5] {
        let mut out = [[0u64; 5]; 5];
        for x in 0..5 {
            for y in 0..5 {
                let mut lane = 0u64;
                for z in 0..64 {
                    if s[state_idx(x, y, z)] {
                        lane |= 1u64 << z;
                    }
                }
                out[x][y] = lane;
            }
        }
        out
    }

    fn lanes_to_bit_state(lanes: &[[u64; 5]; 5]) -> State {
        let mut s = [false; STATE_BITS];
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    s[state_idx(x, y, z)] = (lanes[x][y] >> z) & 1 == 1;
                }
            }
        }
        s
    }

    #[test]
    fn bit_keccak_f_matches_u64_lane() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..3 {
            let s = random_state(&mut rng);
            let mut s_bit = s;
            keccak_f(&mut s_bit);

            let mut lanes = bit_state_to_lanes(&s);
            keccak_f_u64(&mut lanes);
            let s_lane = lanes_to_bit_state(&lanes);

            assert_eq!(
                s_bit
                    .iter()
                    .zip(s_lane.iter())
                    .filter(|(a, b)| a != b)
                    .count(),
                0,
                "bit-level Keccak-f disagrees with u64-lane reference"
            );
        }
    }

    #[test]
    fn theta_rho_pi_preimage_matches_pipeline() {
        let mut rng = Rng::new(42);
        let state_in = random_state(&mut rng);
        let mut step = state_in;
        theta(&mut step);
        let b = rho_pi(&step);

        for z in 0..64 {
            for y in 0..5 {
                for x in 0..5 {
                    let preimage = theta_rho_pi_preimage(x, y, z);
                    let xor: bool = preimage.iter().fold(false, |acc, &i| acc ^ state_in[i]);
                    assert_eq!(
                        xor,
                        b[state_idx(x, y, z)],
                        "preimage mismatch at (x,y,z)=({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn theta_rho_pi_preimage_has_no_duplicates() {
        for z in 0..64 {
            for y in 0..5 {
                for x in 0..5 {
                    let p = theta_rho_pi_preimage(x, y, z);
                    let mut sorted = p;
                    sorted.sort();
                    for w in sorted.windows(2) {
                        assert_ne!(
                            w[0], w[1],
                            "duplicate preimage bit at (x,y,z)=({x},{y},{z}): {sorted:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn layout_constants_consistent() {
        assert_eq!(K, 65_536);
        assert_eq!(USEFUL_BITS, 4160 + 24 * 1600);
        assert_eq!(USEFUL_BITS, 42_560);
        assert!(USEFUL_BITS < K);
        assert_eq!(STATE0_BIT_BASE, 0);
        assert_eq!(STATE24_BIT_BASE, 2048);
        assert_eq!(Z_CONST, 4096);
        assert_eq!(T_PACKED_BIT_BASE, 4160);
        assert_eq!(U64_PER_BLOCK, 1024);
        assert_eq!(state_u64_base(0), 0);
        assert_eq!(state_u64_base(24), 32);
        assert_eq!(t_u64_base(0), 65);
        assert_eq!(t_u64_base(23), 65 + 23 * N_LANES);
    }

    #[test]
    fn witness_layout_round_trip() {
        let mut rng = Rng::new(0xCAB1E_FA17);
        let initial = random_state(&mut rng);

        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        build_chain_witness_ab_packed_into(&initial, &mut z_u64, &mut a_u64, &mut b_u64);

        assert_eq!(z_u64[Z_CONST_U64] & 1, 1);

        let mut s0_lanes: Lanes = [0u64; 25];
        let s0_base = state_u64_base(0);
        for lane_idx in 0..N_LANES {
            s0_lanes[lane_idx] = z_u64[s0_base + lane_idx];
        }
        assert_eq!(lanes_to_state(&s0_lanes), initial);

        let mut s24_lanes: Lanes = [0u64; 25];
        let s24_base = state_u64_base(24);
        for lane_idx in 0..N_LANES {
            s24_lanes[lane_idx] = z_u64[s24_base + lane_idx];
        }
        let s24_recovered = lanes_to_state(&s24_lanes);
        let mut native = initial;
        keccak_f(&mut native);
        assert_eq!(s24_recovered, native, "state_24 != keccak_f(initial)");

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
        let initial = random_state(&mut rng);

        let mut z_u64 = vec![0u64; U64_PER_BLOCK];
        let mut a_u64 = vec![0u64; U64_PER_BLOCK];
        let mut b_u64 = vec![0u64; U64_PER_BLOCK];
        build_chain_witness_ab_packed_into(&initial, &mut z_u64, &mut a_u64, &mut b_u64);

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

        let walker = KeccakLincheckCircuit;
        let comb = walker.fold_alpha_batched(alpha, &eq_inner);

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
        for &n in &[1usize, 8, 4096] {
            let setup = KeccakSetup::new(n);
            let expected_log = min_n_keccaks_log(n);
            assert_eq!(setup.n_keccaks_log(), expected_log);
            assert_eq!(setup.m(), K_LOG + expected_log);
            assert_eq!(setup.n_keccak_slots(), 1 << expected_log);
        }
    }

    #[test]
    #[ignore]
    fn prove_fast_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let setup = KeccakSetup::new(64);
        let mut rng = Rng::new(0x21111_2170);
        let inputs: Vec<State> = (0..64).map(|_| random_state(&mut rng)).collect();
        let mut ch_p = FsChallenger::new(b"flock-lig-keccak-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-lig-keccak-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("prove_fast: verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    #[test]
    fn batch_major_witness_matches_row_major_transposed() {
        for (n_inputs, n_log) in [(8usize, 3usize), (13, 4)] {
            let mut rng = Rng::new(0xBA7C_4 + n_log as u64);
            let inputs: Vec<State> = (0..n_inputs).map(|_| random_state(&mut rng)).collect();

            let (z_r, a_r, b_r, stripe_r) =
                generate_witness_with_ab_packed_and_lincheck(&inputs, n_log);
            let (z_b, a_b, b_b, stripe_b) = generate_witness_batch_major(&inputs, n_log);

            assert_eq!(stripe_b, stripe_r, "stripe diverged (n_log={n_log})");

            let chunks_per_block = U64_PER_BLOCK / 2;
            let transpose = |row: &[F128]| -> Vec<F128> {
                let mut out = vec![F128::ZERO; row.len()];
                for o in 0..1usize << n_log {
                    for c in 0..chunks_per_block {
                        out[(c << n_log) + o] = row[o * chunks_per_block + c];
                    }
                }
                out
            };
            assert_eq!(z_b, transpose(&z_r), "z diverged (n_log={n_log})");
            assert_eq!(a_b, transpose(&a_r), "a diverged (n_log={n_log})");
            assert_eq!(b_b, transpose(&b_r), "b diverged (n_log={n_log})");
        }
    }

    #[test]
    #[ignore]
    fn batch_major_prove_fast_roundtrip() {
        use flock_core::challenger::FsChallenger;

        let setup = KeccakSetup::new_batch_major(64);
        let mut rng = Rng::new(0xBA7C_2170);
        let inputs: Vec<State> = (0..64).map(|_| random_state(&mut rng)).collect();
        let mut ch_p = FsChallenger::new(b"flock-lig-keccak-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-lig-keccak-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("batch-major Ligerito verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    #[test]
    #[ignore]
    fn const_pin_all_zero_rejected() {
        use flock_core::challenger::FsChallenger;

        let n = 60;
        let setup = KeccakSetup::new(n);

        let mut rng = Rng::new(0x5EED_C0DE);
        let inputs: Vec<State> = (0..n).map(|_| random_state(&mut rng)).collect();
        let mut ch_p = FsChallenger::new(b"honest");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"honest");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("honest padded proof rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        let zeros: Vec<State> = vec![[false; STATE_BITS]; n];
        let (mut z, mut a, mut b, mut zlc) =
            generate_witness_with_ab_packed_and_lincheck(&zeros, setup.n_keccaks_log());
        z.iter_mut().for_each(|v| *v = F128::ZERO);
        a.iter_mut().for_each(|v| *v = F128::ZERO);
        b.iter_mut().for_each(|v| *v = F128::ZERO);
        zlc.iter_mut().for_each(|v| *v = 0);
        let mut ch_p = FsChallenger::new(b"poc");
        let (proof, commitment, _) = crate::prover::prove_fast_ligerito_from_witness(
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
            matches!(res, Err(flock_core::verifier::VerifyError::Lincheck(_))),
            "all-zero witness must be rejected by the constant-wire pin; got {res:?}"
        );
    }

    #[test]
    fn batch_major_chain_fold_matches_row_major() {
        use crate::r1cs_hashes::chain_common::{ChainFold, fold_in_out};
        let n_log = 3;
        let mut rng = Rng::new(0xF01D_BA7C);
        let inputs: Vec<State> = (0..8).map(|_| random_state(&mut rng)).collect();
        let (z_r, _, _, _) = generate_witness_with_ab_packed_and_lincheck(&inputs, n_log);
        let (z_b, _, _, _) = generate_witness_batch_major(&inputs, n_log);

        let tau_pos: Vec<F128> = (0..CHAIN_LAYOUT.tau_pos_len())
            .map(|i| F128 {
                lo: rng.next_u64() ^ i as u64,
                hi: rng.next_u64(),
            })
            .collect();
        let fold = ChainFold::new(&CHAIN_LAYOUT, tau_pos);
        let (in_r, out_r) = fold_in_out(
            &CHAIN_LAYOUT,
            flock_core::r1cs::WitnessLayout::RowMajor,
            &z_r,
            &fold,
        );
        let (in_b, out_b) = fold_in_out(
            &CHAIN_LAYOUT,
            flock_core::r1cs::WitnessLayout::BatchMajor,
            &z_b,
            &fold,
        );
        assert_eq!(in_b, in_r, "In fold diverged across layouts");
        assert_eq!(out_b, out_r, "Out fold diverged across layouts");
    }

    #[test]
    #[ignore]
    fn batch_major_prove_chain_ligerito_roundtrip() {
        use flock_core::challenger::RandomChallenger;
        let setup = KeccakSetup::new_batch_major(64);
        let mut rng = Rng::new(0xBA7C_11C7);
        let x0 = random_state(&mut rng);
        let mut inputs = Vec::with_capacity(64);
        let mut cur = x0;
        for _ in 0..64 {
            inputs.push(cur);
            keccak_f(&mut cur);
        }
        let x_last = cur;
        let mut chp = RandomChallenger::new(0xCA2);
        let (proof, comm) = setup.prove_chain(&inputs, &mut chp);
        let mut chv = RandomChallenger::new(0xCA2);
        setup
            .verify_chain(&comm, &proof, &x0, &x_last, &mut chv)
            .expect("batch-major ligerito chain must verify");
    }

    #[test]
    #[ignore]
    fn prove_chain_roundtrip() {
        use flock_core::challenger::RandomChallenger;
        let setup = KeccakSetup::new(64);
        let n_inst = setup.n_keccak_slots();
        let mut rng = Rng::new(0xC0DE_5170);
        let x0 = random_state(&mut rng);
        let mut inputs = Vec::with_capacity(n_inst);
        let mut cur = x0;
        for _ in 0..n_inst {
            inputs.push(cur);
            keccak_f(&mut cur);
        }
        let x_last = cur;
        let mut chp = RandomChallenger::new(0xCA1);
        let (proof, comm) = setup.prove_chain(&inputs, &mut chp);
        let mut chv = RandomChallenger::new(0xCA1);
        setup
            .verify_chain(&comm, &proof, &x0, &x_last, &mut chv)
            .expect("chain must verify");
    }
}

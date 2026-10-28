use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::zerocheck::multilinear::eq_eval;
use serde::{Deserialize, Serialize};

pub fn shift_mle(a: &[F128], b: &[F128]) -> F128 {
    let n = a.len();
    assert_eq!(b.len(), n, "shift_mle: arity mismatch");

    let mut pre = vec![F128::ONE; n + 1];
    for j in 0..n {
        let aj_not_bj = unsafe {
            F128 {
                lo: flock_core::gf2_kernels::gf2_kern_eq_bit_mul_lo(
                    0, a[j].lo, a[j].hi, b[j].lo, b[j].hi,
                ),
                hi: flock_core::gf2_kernels::gf2_kern_eq_bit_mul_hi(
                    0, a[j].lo, a[j].hi, b[j].lo, b[j].hi,
                ),
            }
        };
        pre[j + 1] = pre[j] * aj_not_bj;
    }
    let mut eqsuf = vec![F128::ONE; n + 1];
    for j in (0..n).rev() {
        let eq_l = F128::ONE + a[j] + b[j];
        eqsuf[j] = eqsuf[j + 1] * eq_l;
    }

    let mut acc = F128::ZERO;
    for j in 0..n {
        let mid = unsafe {
            F128 {
                lo: flock_core::gf2_kernels::gf2_kern_eq_bit_mul_lo(
                    0, b[j].lo, b[j].hi, a[j].lo, a[j].hi,
                ),
                hi: flock_core::gf2_kernels::gf2_kern_eq_bit_mul_hi(
                    0, b[j].lo, b[j].hi, a[j].lo, a[j].hi,
                ),
            }
        };
        acc += pre[j] * mid * eqsuf[j + 1];
    }
    acc
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainShiftProof {
    pub rounds: Vec<(F128, F128)>,
    pub g_at_point: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainClaims {
    pub instance_point: Vec<F128>,
    pub sel0: F128,
    pub value: F128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainError {
    MalformedProof,
    SumcheckFinal,
}

#[cfg(test)]
fn dot(eq: &[F128], vals: &[F128]) -> F128 {
    debug_assert_eq!(eq.len(), vals.len());
    let mut acc = F128::ZERO;
    for i in 0..eq.len() {
        acc += eq[i] * vals[i];
    }
    acc
}

pub fn prove_chain_shift<Ch: Challenger>(
    in_vals: &[F128],
    out_vals: &[F128],
    challenger: &mut Ch,
) -> (ChainShiftProof, ChainClaims) {
    let n_total = in_vals.len();
    assert!(n_total.is_power_of_two(), "n_total must be a power of two");
    assert_eq!(out_vals.len(), n_total, "In/Out length mismatch");
    let n = n_total.trailing_zeros() as usize;

    let tau = challenger.sample_f128_vec(n);
    let alpha = challenger.sample_f128();
    let eqtau = build_eq_table(&tau);

    let mut wt = vec![F128::ZERO; 2 * n_total];
    for y in 1..n_total {
        wt[y] = eqtau[y - 1];
    }
    wt[0] += alpha;
    wt[n_total..].copy_from_slice(&eqtau);

    let mut g = Vec::with_capacity(2 * n_total);
    g.extend_from_slice(in_vals);
    g.extend_from_slice(out_vals);

    let d = n + 1;
    let mut rounds = Vec::with_capacity(d);
    let mut r_pts = Vec::with_capacity(d);
    for _ in 0..d {
        use flock_core::field::gf2_128::F256Unreduced;
        let half = g.len() / 2;
        let mut e1_unred = F256Unreduced::ZERO;
        let mut einf_unred = F256Unreduced::ZERO;
        for i in 0..half {
            let (wlo, whi) = (wt[i], wt[i + half]);
            let (glo, ghi) = (g[i], g[i + half]);
            e1_unred ^= whi.mul_unreduced(ghi);
            einf_unred ^= (whi + wlo).mul_unreduced(ghi + glo);
        }
        let e1 = e1_unred.reduce();
        let einf = einf_unred.reduce();
        challenger.observe_f128(e1);
        challenger.observe_f128(einf);
        let r = challenger.sample_f128();
        for i in 0..half {
            wt[i] = wt[i].mle_fold(wt[i + half], r);
            g[i] = g[i].mle_fold(g[i + half], r);
        }
        wt.truncate(half);
        g.truncate(half);
        rounds.push((e1, einf));
        r_pts.push(r);
    }

    let mut full = vec![F128::ZERO; d];
    for (k, &r) in r_pts.iter().enumerate() {
        full[d - 1 - k] = r;
    }
    let claims = ChainClaims {
        instance_point: full[..n].to_vec(),
        sel0: full[n],
        value: g[0],
    };
    (
        ChainShiftProof {
            rounds,
            g_at_point: g[0],
        },
        claims,
    )
}

pub fn verify_chain_shift<Ch: Challenger>(
    proof: &ChainShiftProof,
    x0_r: F128,
    xlast_r: F128,
    n: usize,
    challenger: &mut Ch,
) -> Result<ChainClaims, ChainError> {
    let d = n + 1;
    if proof.rounds.len() != d {
        return Err(ChainError::MalformedProof);
    }

    let tau = challenger.sample_f128_vec(n);
    let alpha = challenger.sample_f128();
    let eq_tau_ones = tau.iter().copied().fold(F128::ONE, |acc, t| acc * t);
    let mut claim = eq_tau_ones * xlast_r + alpha * x0_r;

    let mut r_pts = Vec::with_capacity(d);
    for &(e1, einf) in &proof.rounds {
        challenger.observe_f128(e1);
        challenger.observe_f128(einf);
        let r = challenger.sample_f128();
        let e0 = claim + e1;
        let c1 = e0 + e1 + einf;
        claim = einf * r * r + c1 * r + e0;
        r_pts.push(r);
    }

    let mut full = vec![F128::ZERO; d];
    for (k, &r) in r_pts.iter().enumerate() {
        full[d - 1 - k] = r;
    }
    let taup: Vec<F128> = full[..n].to_vec(); // τ' (instance coords)
    let s0 = full[n];

    let s = shift_mle(&tau, &taup);
    let eq_tt = eq_eval(&tau, &taup);
    let zero_n = vec![F128::ZERO; n];
    let eq_t0 = eq_eval(&taup, &zero_n); // eq(τ', 0ⁿ) = Π_j (1+τ'_j)
    let one_plus_s0 = F128::ONE + s0;
    let w_final = s * one_plus_s0 + eq_tt * s0 + alpha * eq_t0 * one_plus_s0;

    if claim != w_final * proof.g_at_point {
        return Err(ChainError::SumcheckFinal);
    }

    Ok(ChainClaims {
        instance_point: taup,
        sel0: s0,
        value: proof.g_at_point,
    })
}

#[inline]
pub fn read_packed_bit(packed: &[F128], g: usize) -> bool {
    let elem = packed[g >> 7];
    let i_skip = g & 127;
    if i_skip < 64 {
        (elem.lo >> i_skip) & 1 == 1
    } else {
        (elem.hi >> (i_skip - 64)) & 1 == 1
    }
}

pub fn fold_region_naive(packed: &[F128], k_log: usize, taps: &[(usize, F128)]) -> Vec<F128> {
    let total_bits = packed.len() << 7;
    let block = 1usize << k_log;
    assert!(
        total_bits.is_multiple_of(block),
        "packed witness not a whole number of blocks"
    );
    let n_inst = total_bits >> k_log;

    (0..n_inst)
        .map(|i| {
            let block_base = i * block;
            let mut acc = F128::ZERO;
            for &(pos, w) in taps {
                if read_packed_bit(packed, block_base + pos) {
                    acc += w;
                }
            }
            acc
        })
        .collect()
}

pub fn fold_contiguous_regions(
    packed: &[F128],
    k_log: usize,
    region_byte_offsets: &[usize],
    region_weights: &[F128],
) -> Vec<Vec<F128>> {
    use rayon::prelude::*;

    let region_bits = region_weights.len();
    assert!(
        region_bits.is_multiple_of(8),
        "region weights must be a whole number of bytes"
    );
    let n_bytes = region_bits / 8;

    let block = 1usize << k_log;
    assert!(block.is_multiple_of(8), "block must be byte-aligned");
    let block_bytes = block / 8;
    let total_bits = packed.len() << 7;
    assert!(
        total_bits.is_multiple_of(block),
        "packed witness not a whole number of blocks"
    );
    let n_inst = total_bits >> k_log;

    let mut tab = vec![[F128::ZERO; 256]; n_bytes];
    for bo in 0..n_bytes {
        let t = &mut tab[bo];
        for v in 1usize..256 {
            let lsb =
                unsafe { flock_core::gf2_kernels::gf2_kern_isolate_lowest_one(v as u64) } as usize;
            let bit = lsb.trailing_zeros() as usize;
            t[v] = t[v ^ lsb] + region_weights[8 * bo + bit];
        }
    }

    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 16) };

    let n_regions = region_byte_offsets.len();
    let flat: Vec<F128> = (0..n_inst)
        .into_par_iter()
        .flat_map_iter(|i| {
            let instance_base = i * block_bytes;
            let mut row = [F128::ZERO; 8];
            assert!(
                n_regions <= row.len(),
                "fold_contiguous_regions: too many regions"
            );
            for bo in 0..n_bytes {
                let t_bo = &tab[bo];
                for (r_idx, &off) in region_byte_offsets.iter().enumerate() {
                    row[r_idx] += t_bo[bytes[instance_base + off + bo] as usize];
                }
            }
            (0..n_regions).map(move |r_idx| row[r_idx])
        })
        .collect();

    (0..n_regions)
        .map(|r_idx| (0..n_inst).map(|i| flat[i * n_regions + r_idx]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::RandomChallenger;

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
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    fn bool_point(idx: usize, n: usize) -> Vec<F128> {
        (0..n)
            .map(|j| {
                if (idx >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                }
            })
            .collect()
    }

    fn pack(z: &[bool]) -> Vec<F128> {
        assert!(z.len().is_multiple_of(128));
        (0..z.len() / 128)
            .map(|i_rest| {
                let base = i_rest * 128;
                let mut lo = 0u64;
                let mut hi = 0u64;
                for r in 0..64 {
                    if z[base + r] {
                        lo |= 1 << r;
                    }
                    if z[base + 64 + r] {
                        hi |= 1 << r;
                    }
                }
                F128 { lo, hi }
            })
            .collect()
    }

    #[test]
    fn fold_region_naive_matches_direct() {
        let mut rng = Rng::new(0xF01D);
        let k_log = 5usize;
        let n = 3usize;
        let block = 1usize << k_log;
        let total = (1usize << n) * block;
        let z: Vec<bool> = (0..total).map(|_| rng.next_u64() & 1 == 1).collect();
        let packed = pack(&z);

        let taps: Vec<(usize, F128)> = (0..10).map(|t| (3 * t % block, rng.f128())).collect();

        let got = fold_region_naive(&packed, k_log, &taps);
        assert_eq!(got.len(), 1 << n);
        for i in 0..(1 << n) {
            let mut want = F128::ZERO;
            for &(pos, w) in &taps {
                if z[i * block + pos] {
                    want += w;
                }
            }
            assert_eq!(got[i], want, "instance {i}");
        }
    }

    #[test]
    fn fold_contiguous_regions_matches_per_region() {
        let mut rng = Rng::new(0xC0DE_F00D);
        let k_log = 6usize;
        let n = 4usize;
        let block = 1usize << k_log;
        let total = (1usize << n) * block;
        let z: Vec<bool> = (0..total).map(|_| rng.next_u64() & 1 == 1).collect();
        let packed = pack(&z);

        let region_bits = 16;
        let weights: Vec<F128> = (0..region_bits).map(|_| rng.f128()).collect();

        for &offs in &[&[0usize] as &[usize], &[0, 2], &[0, 2, 4]] {
            let got = fold_contiguous_regions(&packed, k_log, offs, &weights);
            assert_eq!(got.len(), offs.len());
            for (r_idx, &off) in offs.iter().enumerate() {
                let taps: Vec<(usize, F128)> = (0..region_bits)
                    .map(|p| (off * 8 + p, weights[p]))
                    .collect();
                let want = fold_region_naive(&packed, k_log, &taps);
                assert_eq!(got[r_idx], want, "region {r_idx} (offset {off})");
            }
        }
    }

    #[test]
    fn shift_mle_boolean_is_successor() {
        for n in 1..=6 {
            let n_total = 1usize << n;
            for a in 0..n_total {
                for b in 0..n_total {
                    let av = bool_point(a, n);
                    let bv = bool_point(b, n);
                    let got = shift_mle(&av, &bv);
                    let want = if b == a + 1 { F128::ONE } else { F128::ZERO };
                    assert_eq!(got, want, "shift({a},{b}) n={n}");
                }
            }
        }
    }

    #[test]
    fn shift_mle_top_has_no_successor() {
        let mut rng = Rng::new(7);
        for n in 1..=5 {
            let a = bool_point((1 << n) - 1, n);
            let b = rng.f128_vec(n);
            assert_eq!(shift_mle(&a, &b), F128::ZERO);
        }
    }

    #[test]
    fn shift_equals_shifted_eq() {
        let mut rng = Rng::new(11);
        for n in 1..=5 {
            let n_total = 1usize << n;
            let tau = rng.f128_vec(n);
            let eqtau = build_eq_table(&tau);
            for y in 0..n_total {
                let yv = bool_point(y, n);
                let got = shift_mle(&tau, &yv);
                let want = if y == 0 { F128::ZERO } else { eqtau[y - 1] };
                assert_eq!(got, want, "y={y} n={n}");
            }
        }
    }

    #[test]
    fn honest_roundtrip_accepts() {
        for n in 3..=8 {
            let n_total = 1usize << n;
            let mut rng = Rng::new(100 + n as u64);
            let chain: Vec<F128> = rng.f128_vec(n_total + 1);
            let in_vals: Vec<F128> = chain[..n_total].to_vec();
            let out_vals: Vec<F128> = chain[1..].to_vec();
            let x0_r = chain[0];
            let xlast_r = chain[n_total];

            let mut chp = RandomChallenger::new(42);
            let (proof, _claims) = prove_chain_shift(&in_vals, &out_vals, &mut chp);

            let mut chv = RandomChallenger::new(42);
            let claims = verify_chain_shift(&proof, x0_r, xlast_r, n, &mut chv)
                .expect("honest proof should verify");

            let eq_taup = build_eq_table(&claims.instance_point);
            let in_true = dot(&eq_taup, &in_vals);
            let out_true = dot(&eq_taup, &out_vals);
            let g_true = (F128::ONE + claims.sel0) * in_true + claims.sel0 * out_true;
            assert_eq!(claims.value, g_true, "merged claim n={n}");
        }
    }

    #[test]
    fn broken_chain_rejects() {
        let n = 6;
        let n_total = 1usize << n;
        let mut rng = Rng::new(2024);
        let chain: Vec<F128> = rng.f128_vec(n_total + 1);
        let in_vals: Vec<F128> = chain[..n_total].to_vec();
        let mut out_vals: Vec<F128> = chain[1..].to_vec();
        let x0_r = chain[0];
        let xlast_r = chain[n_total];

        out_vals[3] += F128::ONE;

        let mut chp = RandomChallenger::new(9);
        let (proof, _claims) = prove_chain_shift(&in_vals, &out_vals, &mut chp);
        let mut chv = RandomChallenger::new(9);
        let res = verify_chain_shift(&proof, x0_r, xlast_r, n, &mut chv);
        assert_eq!(res, Err(ChainError::SumcheckFinal));
    }

    #[test]
    fn wrong_input_endpoint_rejects() {
        let n = 5;
        let n_total = 1usize << n;
        let mut rng = Rng::new(555);
        let chain: Vec<F128> = rng.f128_vec(n_total + 1);
        let in_vals: Vec<F128> = chain[..n_total].to_vec();
        let out_vals: Vec<F128> = chain[1..].to_vec();
        let xlast_r = chain[n_total];
        let wrong_x0 = chain[0] + F128::ONE;

        let mut chp = RandomChallenger::new(3);
        let (proof, _claims) = prove_chain_shift(&in_vals, &out_vals, &mut chp);
        let mut chv = RandomChallenger::new(3);
        let res = verify_chain_shift(&proof, wrong_x0, xlast_r, n, &mut chv);
        assert_eq!(res, Err(ChainError::SumcheckFinal));
    }

    #[test]
    fn wrong_output_endpoint_rejects() {
        let n = 5;
        let n_total = 1usize << n;
        let mut rng = Rng::new(777);
        let chain: Vec<F128> = rng.f128_vec(n_total + 1);
        let in_vals: Vec<F128> = chain[..n_total].to_vec();
        let out_vals: Vec<F128> = chain[1..].to_vec();
        let x0_r = chain[0];
        let wrong_xlast = chain[n_total] + F128::ONE;

        let mut chp = RandomChallenger::new(1);
        let (proof, _claims) = prove_chain_shift(&in_vals, &out_vals, &mut chp);
        let mut chv = RandomChallenger::new(1);
        let res = verify_chain_shift(&proof, x0_r, wrong_xlast, n, &mut chv);
        assert_eq!(res, Err(ChainError::SumcheckFinal));
    }
}

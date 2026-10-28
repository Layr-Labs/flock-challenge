
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::challenger::Challenger;
use crate::field::F128;
use crate::merkle::Hash;
use crate::pcs::ligerito::{ProverConfig, VerifierConfig};
use crate::pcs::{
    self, Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef, PcsParams,
    ProverData, commit,
};
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::univariate_skip::{SplitEqGhash, build_eq};

const DOMAIN: &[u8] = b"flock-perm-v0";


#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutationProof {
    pub v_root: Hash,
    pub claimed_product: F128,
    pub rounds: Vec<(F128, F128)>,
    pub f_eval: F128,
    pub g_eval: F128,
    pub s_sigma_eval: F128,
    pub v_x0: F128, // v(ρ, 0)
    pub v_x1: F128, // v(ρ, 1)
    pub v_1x: F128, // v(1, ρ)
    pub v_0x: F128, // v(0, ρ) — the leaf value ℓ(ρ), used in relation A
    pub v_open: pcs::BatchOpeningProofLigerito,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutationClaim {
    pub rho: Vec<F128>,
    pub f_eval: F128,
    pub g_eval: F128,
    pub s_sigma_eval: F128,
    pub v_x0: F128,
    pub v_x1: F128,
    pub v_1x: F128,
    pub v_0x: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    SumcheckFinalFailed,
    RootNotOne,
    PcsOpen(pcs::VerifyError),
}


fn batch_inverse(values: &[F128]) -> Vec<F128> {
    const CHUNK: usize = 1 << 14;
    let mut out = vec![F128::ZERO; values.len()];
    out.par_chunks_mut(CHUNK)
        .zip(values.par_chunks(CHUNK))
        .for_each(|(out_c, val_c)| {
            let mut acc = F128::ONE;
            for (o, v) in out_c.iter_mut().zip(val_c) {
                debug_assert!(!v.is_zero(), "batch_inverse: zero input");
                *o = acc; // prefix product within this chunk
                acc *= *v;
            }
            acc = acc.inv(); // (∏ chunk)⁻¹
            for (o, v) in out_c.iter_mut().zip(val_c).rev() {
                *o *= acc;
                acc *= *v;
            }
        });
    out
}

fn s_id_basis(mu: usize) -> Vec<F128> {
    assert!(mu <= 128, "s_id needs μ ≤ 128 distinct bit positions");
    (0..mu)
        .map(|i| {
            let idx = i as u64;
            unsafe {
                F128::new(
                    crate::gf2_kernels::gf2_kern_monomial_basis128_lo(idx),
                    crate::gf2_kernels::gf2_kern_monomial_basis128_hi(idx),
                )
            }
        })
        .collect()
}

#[cfg(test)]
fn s_id_value(idx: usize, basis: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for (i, b) in basis.iter().enumerate() {
        if (idx >> i) & 1 == 1 {
            acc += *b;
        }
    }
    acc
}

fn s_id_eval(basis: &[F128], rho: &[F128]) -> F128 {
    let mut acc = F128::ZERO;
    for (b, r) in basis.iter().zip(rho) {
        acc += *b * *r;
    }
    acc
}

fn build_grand_product(h: &[F128]) -> Vec<F128> {
    let n = h.len();
    assert!(n.is_power_of_two() && n >= 2);
    let mut v = vec![F128::ZERO; 2 * n];
    v[..n].copy_from_slice(h);
    let mut read = 0usize;
    let mut size = n;
    let mut write = n;
    while size >= 2 {
        let half = size / 2;
        let (done, rest) = v.split_at_mut(write);
        let src = &done[read..read + size];
        rest[..half]
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, d)| *d = src[2 * j] * src[2 * j + 1]);
        read = write;
        write += half;
        size = half;
    }
    v
}

const PAR_THRESHOLD: usize = 1 << 12;

fn fold_in_place(u: &mut Vec<F128>, rho: F128) {
    let half = u.len() / 2;
    let one_minus = F128::ONE + rho;
    if half >= PAR_THRESHOLD {
        let mut out = crate::scratch::take_f128(half);
        out.par_iter_mut().enumerate().for_each(|(x, o)| {
            *o = u[2 * x] * one_minus + u[2 * x + 1] * rho;
        });
        let old = std::mem::replace(u, out);
        crate::scratch::give_f128(old);
    } else {
        for x in 0..half {
            u[x] = u[2 * x] * one_minus + u[2 * x + 1] * rho;
        }
        u.truncate(half);
    }
}

#[cfg(test)]
fn mle_eval(table: &[F128], point: &[F128]) -> F128 {
    let mut t = table.to_vec();
    for &r in point {
        fold_in_place(&mut t, r);
    }
    t[0]
}

const PCS_LOG_INV_RATE: usize = 1;
const PCS_LOG_BATCH_SIZE: usize = 1;

fn pcs_params(num_vars: usize) -> PcsParams {
    PcsParams {
        m: num_vars + LOG_PACKING,
        log_inv_rate: PCS_LOG_INV_RATE,
        log_batch_size: PCS_LOG_BATCH_SIZE,
        profile: Default::default(),
        merkle_hash: Default::default(),
    }
}

fn ligerito_prover_config(num_vars: usize) -> ProverConfig {
    pcs::ligerito::default_config(num_vars, PCS_LOG_BATCH_SIZE, PCS_LOG_INV_RATE)
        .expect("Ligerito config for the aux poly v; permutation check requires μ ≥ 7")
}

fn ligerito_verifier_config(num_vars: usize) -> VerifierConfig {
    pcs::ligerito::default_verifier_config(num_vars, PCS_LOG_BATCH_SIZE, PCS_LOG_INV_RATE)
        .expect("Ligerito verifier config for the aux poly v; permutation check requires μ ≥ 7")
}

fn open_v<C: Challenger>(
    poly: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    claims: &[PackedDirectClaim],
    ch: &mut C,
) -> pcs::BatchOpeningProofLigerito {
    let num_vars = commitment.params.log_msg_len();
    let padding = PaddingSpec::dense(commitment.params.m);
    let cfg = ligerito_prover_config(num_vars);
    pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        poly,
        prover_data,
        commitment,
        &[],
        &[],
        claims,
        &padding,
        &cfg,
        ch,
    )
}

fn verify_v<C: Challenger>(
    commitment: &Commitment,
    claims: &[PackedDirectClaimRef<'_>],
    open: &pcs::BatchOpeningProofLigerito,
    ch: &mut C,
) -> Result<(), VerifyError> {
    let num_vars = commitment.params.log_msg_len();
    let cfg = ligerito_verifier_config(num_vars);
    pcs::verify_opening_batch_ligerito_mixed(commitment, &[], &[], &[], claims, open, &cfg, ch)
        .map_err(VerifyError::PcsOpen)
}

fn v_open_points(rho: &[F128]) -> [Vec<F128>; 5] {
    let mu = rho.len();
    let with_low = |bit: F128| {
        let mut p = Vec::with_capacity(mu + 1);
        p.push(bit);
        p.extend_from_slice(rho);
        p
    };
    let with_high = |bit: F128| {
        let mut p = rho.to_vec();
        p.push(bit);
        p
    };
    let mut root = vec![F128::ONE; mu + 1];
    root[0] = F128::ZERO;
    [
        with_low(F128::ZERO),  // v(ρ, 0)
        with_low(F128::ONE),   // v(ρ, 1)
        with_high(F128::ONE),  // v(1, ρ)
        with_high(F128::ZERO), // v(0, ρ)
        root,                  // v[2N-2]
    ]
}


#[allow(clippy::too_many_arguments)]
fn round_message(
    f: &[F128],
    g: &[F128],
    s_sig: &[F128],
    a: &[F128],
    b: &[F128],
    c: &[F128],
    v0: &[F128],
    beta: F128,
    gamma: F128,
    alpha: F128,
    eq: &SplitEqGhash,
) -> (F128, F128) {
    let lo = &eq.lo;
    let hi = &eq.hi;
    let block = lo.len(); // 2^n_lo  (x_lo per x_hi)
    let n_blocks = hi.len(); // 2^n_hi
    debug_assert_eq!(block * n_blocks, f.len() / 2);

    let block_fn = |x_hi: usize| -> (F128, F128) {
        let x_base = x_hi * block;
        let (mut s1, mut s_inf) = (F128::ZERO, F128::ZERO);
        for x_lo in 0..block {
            let xp = x_base + x_lo;
            let (i0, i1) = (2 * xp, 2 * xp + 1);

            let p1 = f[i1] + gamma;
            let q0 = g[i0] + beta * s_sig[i0] + gamma;
            let q1 = g[i1] + beta * s_sig[i1] + gamma;

            let (l0, l1) = (v0[i0], v0[i1]);
            let ga1 = l1 * q1 + p1;
            let ga_inf = (l0 + l1) * (q0 + q1);

            let (a0, a1, b0, b1) = (a[i0], a[i1], b[i0], b[i1]);
            let gb1 = c[i1] + a1 * b1;
            let gb_inf = (a0 + a1) * (b0 + b1);

            let el = lo[x_lo];
            s1 += el * (ga1 + alpha * gb1);
            s_inf += el * (ga_inf + alpha * gb_inf);
        }
        let eh = hi[x_hi];
        (eh * s1, eh * s_inf)
    };

    if block * n_blocks >= PAR_THRESHOLD {
        (0..n_blocks).into_par_iter().map(block_fn).reduce(
            || (F128::ZERO, F128::ZERO),
            |(o0, i0), (o1, i1)| (o0 + o1, i0 + i1),
        )
    } else {
        let (mut g_one, mut g_inf) = (F128::ZERO, F128::ZERO);
        for x_hi in 0..n_blocks {
            let (o, i) = block_fn(x_hi);
            g_one += o;
            g_inf += i;
        }
        (g_one, g_inf)
    }
}


pub fn prove<C: Challenger>(
    f: &[F128],
    g: &[F128],
    sigma: &[usize],
    ch: &mut C,
) -> (PermutationProof, PermutationClaim) {
    let n = f.len();
    assert_eq!(g.len(), n);
    assert_eq!(sigma.len(), n);
    assert!(n.is_power_of_two() && n >= 2, "need N = 2^μ ≥ 2");
    let mu = n.trailing_zeros() as usize;

    let trace = std::env::var("PERM_TRACE").is_ok();
    let mut t = std::time::Instant::now();
    let mut tp = |label: &str| {
        if trace {
            eprintln!(
                "  [perm-prove] {label:<14} {:8.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
            t = std::time::Instant::now();
        }
    };

    ch.observe_label(DOMAIN);
    let beta = ch.sample_f128();
    let gamma = ch.sample_f128();

    let basis = s_id_basis(mu);
    let mut s_id_vec = vec![F128::ZERO; n];
    for (k, &bk) in basis.iter().enumerate() {
        let half = 1usize << k;
        let (lo, hi) = s_id_vec.split_at_mut(half);
        if half >= (1 << 12) {
            hi[..half]
                .par_iter_mut()
                .zip(lo.par_iter())
                .for_each(|(dst, src)| *dst = *src + bk);
        } else {
            for (dst, src) in hi.iter_mut().zip(lo.iter()) {
                *dst = *src + bk;
            }
        }
    }
    let s_sig_vec: Vec<F128> = sigma.par_iter().map(|&sx| s_id_vec[sx]).collect();
    let p: Vec<F128> = f
        .par_iter()
        .zip(&s_id_vec)
        .map(|(fx, sx)| *fx + beta * *sx + gamma)
        .collect();
    let q: Vec<F128> = g
        .par_iter()
        .zip(&s_sig_vec)
        .map(|(gx, sx)| *gx + beta * *sx + gamma)
        .collect();
    let q_inv = batch_inverse(&q);
    let leaves: Vec<F128> = p.par_iter().zip(&q_inv).map(|(px, qx)| *px * *qx).collect();

    let v = build_grand_product(&leaves);
    let a: Vec<F128> = (0..n).into_par_iter().map(|i| v[2 * i]).collect();
    let b: Vec<F128> = (0..n).into_par_iter().map(|i| v[2 * i + 1]).collect();
    let c: Vec<F128> = v[n..2 * n].to_vec(); // c[i] = v[n+i]
    let v0: Vec<F128> = v[..n].to_vec();
    let claimed_product = v[2 * n - 2];
    tp("witness+v");

    let params_v = pcs_params(mu + 1);
    let (commitment_v, pdata_v) = commit(&v, &params_v);
    tp("commit(v)");
    ch.observe_bytes(&commitment_v.root);
    ch.observe_f128(claimed_product);
    let alpha = ch.sample_f128();
    let r = ch.sample_f128_vec(mu);

    let mut sid_suffix = vec![F128::ZERO; mu + 1];
    for i in (0..mu).rev() {
        sid_suffix[i] = basis[i] * r[i] + sid_suffix[i + 1];
    }

    let (mut wf, mut wg) = (f.to_vec(), g.to_vec());
    let mut wssig = s_sig_vec;
    let (mut wa, mut wb, mut wc, mut wv0) = (a, b, c, v0);

    let mut rounds = Vec::with_capacity(mu);
    let mut rho = Vec::with_capacity(mu);
    let mut c_prefix = F128::ZERO; // Σ_{k<i} basis_k·ρ_k
    for i in 0..mu {
        let eq = SplitEqGhash::new(&r[i + 1..mu]);
        let (g1_core, g_inf) = round_message(
            &wf, &wg, &wssig, &wa, &wb, &wc, &wv0, beta, gamma, alpha, &eq,
        );
        let g1 = g1_core + beta * ((c_prefix + basis[i]) + sid_suffix[i + 1]);
        ch.observe_f128(g1);
        ch.observe_f128(g_inf);
        let rho_i = ch.sample_f128();
        rho.push(rho_i);
        rounds.push((g1, g_inf));
        c_prefix += basis[i] * rho_i;

        for u in [
            &mut wf, &mut wg, &mut wssig, &mut wa, &mut wb, &mut wc, &mut wv0,
        ] {
            fold_in_place(u, rho_i);
        }
    }
    debug_assert_eq!(c_prefix, s_id_eval(&basis, &rho));
    tp("sumcheck");

    let (f_eval, g_eval, s_sigma_eval) = (wf[0], wg[0], wssig[0]);
    let (v_x0, v_x1, v_1x, v_0x) = (wa[0], wb[0], wc[0], wv0[0]);

    observe_evals(ch, &[f_eval, g_eval, s_sigma_eval, v_x0, v_x1, v_1x, v_0x]);

    let v_points = v_open_points(&rho);
    let v_values = [v_x0, v_x1, v_1x, v_0x, claimed_product];
    let v_claims: Vec<PackedDirectClaim> = v_points
        .iter()
        .zip(v_values)
        .map(|(point, value)| PackedDirectClaim {
            point: point.clone(),
            value,
            eq_ind: DirectEqInd::Dense(build_eq(point)),
        })
        .collect();
    let v_open = open_v(v, &pdata_v, &commitment_v, &v_claims, ch);
    tp("open(v)");

    let proof = PermutationProof {
        v_root: commitment_v.root,
        claimed_product,
        rounds,
        f_eval,
        g_eval,
        s_sigma_eval,
        v_x0,
        v_x1,
        v_1x,
        v_0x,
        v_open,
    };

    let claim = PermutationClaim {
        rho,
        f_eval,
        g_eval,
        s_sigma_eval,
        v_x0,
        v_x1,
        v_1x,
        v_0x,
    };
    (proof, claim)
}


pub fn verify<C: Challenger>(
    mu: usize,
    proof: &PermutationProof,
    ch: &mut C,
) -> Result<PermutationClaim, VerifyError> {
    assert_eq!(proof.rounds.len(), mu);

    ch.observe_label(DOMAIN);
    let beta = ch.sample_f128();
    let gamma = ch.sample_f128();

    let commitment_v = Commitment {
        root: proof.v_root,
        params: pcs_params(mu + 1),
    };
    ch.observe_bytes(&commitment_v.root);
    ch.observe_f128(proof.claimed_product);
    let alpha = ch.sample_f128();
    let r = ch.sample_f128_vec(mu);

    let mut c_running = F128::ZERO;
    let mut rho = Vec::with_capacity(mu);
    for i in 0..mu {
        let (g1, g_inf) = proof.rounds[i];
        let r_eq = r[i];
        let one_plus_r_eq = F128::ONE + r_eq;
        let g0 = (c_running + r_eq * g1) * one_plus_r_eq.inv();

        ch.observe_f128(g1);
        ch.observe_f128(g_inf);
        let rho_i = ch.sample_f128();
        rho.push(rho_i);

        let one_plus_rho = F128::ONE + rho_i;
        c_running = g0 * one_plus_rho + g1 * rho_i + g_inf * rho_i * one_plus_rho;
    }

    let basis = s_id_basis(mu);
    let s_id_rho = s_id_eval(&basis, &rho);
    let p_rho = proof.f_eval + beta * s_id_rho + gamma;
    let q_rho = proof.g_eval + beta * proof.s_sigma_eval + gamma;
    let f_a = proof.v_0x * q_rho + p_rho;
    let f_b = proof.v_1x + proof.v_x0 * proof.v_x1;
    let expected = f_a + alpha * f_b;
    if c_running != expected {
        return Err(VerifyError::SumcheckFinalFailed);
    }
    if proof.claimed_product != F128::ONE {
        return Err(VerifyError::RootNotOne);
    }

    observe_evals(
        ch,
        &[
            proof.f_eval,
            proof.g_eval,
            proof.s_sigma_eval,
            proof.v_x0,
            proof.v_x1,
            proof.v_1x,
            proof.v_0x,
        ],
    );

    let v_points = v_open_points(&rho);
    let v_values = [
        proof.v_x0,
        proof.v_x1,
        proof.v_1x,
        proof.v_0x,
        proof.claimed_product,
    ];
    let v_refs: Vec<PackedDirectClaimRef<'_>> = v_points
        .iter()
        .zip(v_values)
        .map(|(point, value)| PackedDirectClaimRef { point, value })
        .collect();
    verify_v(&commitment_v, &v_refs, &proof.v_open, ch)?;

    Ok(PermutationClaim {
        rho,
        f_eval: proof.f_eval,
        g_eval: proof.g_eval,
        s_sigma_eval: proof.s_sigma_eval,
        v_x0: proof.v_x0,
        v_x1: proof.v_x1,
        v_1x: proof.v_1x,
        v_0x: proof.v_0x,
    })
}

fn observe_evals<C: Challenger>(ch: &mut C, evals: &[F128; 7]) {
    for e in evals {
        ch.observe_f128(*e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128::new(self.next_u64(), self.next_u64())
        }
        fn permutation(&mut self, n: usize) -> Vec<usize> {
            let mut p: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (self.next_u64() % (i as u64 + 1)) as usize;
                p.swap(i, j);
            }
            p
        }
    }

    fn invert(sigma: &[usize]) -> Vec<usize> {
        let mut inv = vec![0usize; sigma.len()];
        for (x, &sx) in sigma.iter().enumerate() {
            inv[sx] = x;
        }
        inv
    }

    fn honest_instance(mu: usize, seed: u64) -> (Vec<F128>, Vec<F128>, Vec<usize>) {
        let n = 1usize << mu;
        let mut rng = Rng::new(seed);
        let g: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
        let sigma = rng.permutation(n);
        let sinv = invert(&sigma);
        let f: Vec<F128> = (0..n).map(|x| g[sinv[x]]).collect();
        (f, g, sigma)
    }

    fn bind<C: Challenger>(ch: &mut C, f: &[F128], g: &[F128], sigma: &[usize]) {
        ch.observe_f128_slice(f);
        ch.observe_f128_slice(g);
        for &s in sigma {
            ch.observe_f128(F128::new(s as u64, 0));
        }
    }

    fn run_prove(f: &[F128], g: &[F128], sigma: &[usize]) -> (PermutationProof, PermutationClaim) {
        let mut ch = FsChallenger::new(b"perm-test");
        bind(&mut ch, f, g, sigma);
        prove(f, g, sigma, &mut ch)
    }

    fn run_verify(
        mu: usize,
        f: &[F128],
        g: &[F128],
        sigma: &[usize],
        proof: &PermutationProof,
    ) -> Result<PermutationClaim, VerifyError> {
        let mut ch = FsChallenger::new(b"perm-test");
        bind(&mut ch, f, g, sigma);
        verify(mu, proof, &mut ch)
    }

    #[test]
    fn honest_roundtrip_and_claim_match() {
        for mu in 7..=8 {
            let (f, g, sigma) = honest_instance(mu, 0xC0FFEE ^ mu as u64);
            let (proof, claim_p) = run_prove(&f, &g, &sigma);
            assert_eq!(proof.claimed_product, F128::ONE, "μ={mu}: ∏ℓ ≠ 1");
            let claim_v = run_verify(mu, &f, &g, &sigma, &proof).expect("verify");
            assert_eq!(claim_p, claim_v, "μ={mu}: prover/verifier claim mismatch");
        }
    }

    #[test]
    fn claim_matches_direct_mle() {
        let mu = 7;
        let (f, g, sigma) = honest_instance(mu, 0xABCD);
        let (_proof, claim) = run_prove(&f, &g, &sigma);

        let mut ch = FsChallenger::new(b"perm-test");
        bind(&mut ch, &f, &g, &sigma);
        ch.observe_label(DOMAIN);
        let beta = ch.sample_f128();
        let gamma = ch.sample_f128();
        let n = 1usize << mu;
        let basis = s_id_basis(mu);
        let s_id_vec: Vec<F128> = (0..n).map(|x| s_id_value(x, &basis)).collect();
        let s_sig_vec: Vec<F128> = (0..n).map(|x| s_id_value(sigma[x], &basis)).collect();
        let p: Vec<F128> = (0..n).map(|x| f[x] + beta * s_id_vec[x] + gamma).collect();
        let q: Vec<F128> = (0..n).map(|x| g[x] + beta * s_sig_vec[x] + gamma).collect();
        let q_inv = batch_inverse(&q);
        let leaves: Vec<F128> = (0..n).map(|x| p[x] * q_inv[x]).collect();
        let v = build_grand_product(&leaves);

        let rho = &claim.rho;
        assert_eq!(claim.f_eval, mle_eval(&f, rho));
        assert_eq!(claim.g_eval, mle_eval(&g, rho));
        assert_eq!(claim.s_sigma_eval, mle_eval(&s_sig_vec, rho));
        assert_eq!(claim.v_0x, mle_eval(&leaves, rho));

        let mut pt_x0 = vec![F128::ZERO];
        pt_x0.extend_from_slice(rho);
        let mut pt_x1 = vec![F128::ONE];
        pt_x1.extend_from_slice(rho);
        let mut pt_1x = rho.clone();
        pt_1x.push(F128::ONE);
        let mut pt_0x = rho.clone();
        pt_0x.push(F128::ZERO);
        assert_eq!(claim.v_x0, mle_eval(&v, &pt_x0));
        assert_eq!(claim.v_x1, mle_eval(&v, &pt_x1));
        assert_eq!(claim.v_1x, mle_eval(&v, &pt_1x));
        assert_eq!(claim.v_0x, mle_eval(&v, &pt_0x));
        assert_eq!(v[2 * n - 2], F128::ONE);
    }

    #[test]
    fn tampered_witness_rejected() {
        let mu = 7;
        let (f, g, sigma) = honest_instance(mu, 0x1234);
        let (proof, _) = run_prove(&f, &g, &sigma);

        let mut f_bad = f.clone();
        f_bad[3] += F128::ONE;
        let res = run_verify(mu, &f_bad, &g, &sigma, &proof);
        assert!(
            matches!(res, Err(VerifyError::SumcheckFinalFailed)),
            "got {res:?}"
        );
    }

    #[test]
    fn non_permutation_relation_rejected() {
        let mu = 7;
        let n = 1usize << mu;
        let mut rng = Rng::new(0x9999);
        let g: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
        let f: Vec<F128> = (0..n).map(|_| rng.f128()).collect(); // unrelated
        let sigma: Vec<usize> = (0..n).collect(); // identity tag
        let (proof, _) = run_prove(&f, &g, &sigma);
        assert_ne!(proof.claimed_product, F128::ONE, "expected ∏h ≠ 1");
        let res = run_verify(mu, &f, &g, &sigma, &proof);
        assert_eq!(res, Err(VerifyError::RootNotOne));
    }

    #[test]
    fn tampered_pcs_opening_rejected() {
        let mu = 7;
        let (f, g, sigma) = honest_instance(mu, 0x2468);
        let (mut proof, _) = run_prove(&f, &g, &sigma);
        proof.v_open.ligerito.sumcheck_transcript[0].u_0.lo ^= 1;
        let res = run_verify(mu, &f, &g, &sigma, &proof);
        assert!(matches!(res, Err(VerifyError::PcsOpen(_))), "got {res:?}");
    }
}

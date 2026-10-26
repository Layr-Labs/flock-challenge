use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::build_eq_table;

#[derive(Clone, Debug)]
pub struct JaggedParams {
    pub n: usize,

    pub k: usize,

    pub m: usize,

    pub col_prefix_sums: Vec<u64>,
}

impl JaggedParams {
    pub fn from_heights(heights: &[u64], n: usize, m: usize) -> Self {
        assert!(
            heights.len().is_power_of_two(),
            "number of columns must be a power of two (zero-pad)"
        );
        let k = heights.len().trailing_zeros() as usize;
        let mut col_prefix_sums = Vec::with_capacity(heights.len() + 1);
        let mut acc: u64 = 0;
        col_prefix_sums.push(0);
        for &h in heights {
            assert!(h <= (1u64 << n), "column height exceeds 2^n");
            acc += h;
            col_prefix_sums.push(acc);
        }
        assert!(acc <= (1u64 << m), "total area exceeds 2^m");
        JaggedParams {
            n,
            k,
            m,
            col_prefix_sums,
        }
    }

    pub fn area(&self) -> u64 {
        *self.col_prefix_sums.last().unwrap()
    }

    pub fn unrank(&self, i: u64) -> (usize, usize) {
        debug_assert!(i < self.area());

        let col = self.col_prefix_sums.partition_point(|&t| t <= i) - 1;
        let row = i - self.col_prefix_sums[col];
        (row as usize, col)
    }
}

#[inline]
fn point_bit(z: &[F128], layer: usize) -> F128 {
    if layer < z.len() {
        z[layer]
    } else {
        F128::ZERO
    }
}

#[inline]
fn int_bit(t: u64, layer: usize) -> F128 {
    if (t >> layer) & 1 == 1 {
        F128::ONE
    } else {
        F128::ZERO
    }
}

#[inline]
fn transition(row: bool, index: bool, curr: bool, next: bool, state: usize) -> Option<usize> {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let out = crate::gf2_kernels::gf2_kern_jagged_transition(
            row as u64,
            index as u64,
            curr as u64,
            next as u64,
            state as u64,
        ) as usize;
        if out < 4 { Some(out) } else { None }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let carry = state & 1;
        let comparison = (state >> 1) & 1;

        let sum = row as usize + carry + curr as usize;
        if (index as usize) != (sum & 1) {
            return None;
        }
        let new_carry = sum >> 1;

        let new_comparison = if index == next {
            comparison
        } else {
            next as usize
        };
        Some(new_carry + (new_comparison << 1))
    }
}

const STATE_INITIAL: usize = 0;
const STATE_SUCCESS: usize = 2;

fn g_hat_eval(z_row: &[F128], z_index: &[F128], t_c: u64, t_next: u64, m: usize) -> F128 {
    let mut dp = [F128::ZERO; 4];
    dp[STATE_SUCCESS] = F128::ONE;
    for layer in (0..=m).rev() {
        let eq16 = build_eq_table(&[
            point_bit(z_row, layer),
            point_bit(z_index, layer),
            int_bit(t_c, layer),
            int_bit(t_next, layer),
        ]);
        let mut new_dp = [F128::ZERO; 4];
        for (s, slot) in new_dp.iter_mut().enumerate() {
            let mut acc_unred = crate::field::F256Unreduced::ZERO;
            for (idx, &w) in eq16.iter().enumerate() {
                let row = idx & 1 != 0;
                let index = (idx >> 1) & 1 != 0;
                let curr = (idx >> 2) & 1 != 0;
                let next = (idx >> 3) & 1 != 0;
                if let Some(out) = transition(row, index, curr, next, s) {
                    acc_unred ^= w.mul_unreduced(dp[out]);
                }
            }
            *slot = acc_unred.reduce();
        }
        dp = new_dp;
    }
    dp[STATE_INITIAL]
}

pub fn f_hat_t(params: &JaggedParams, z_row: &[F128], z_col: &[F128], z_index: &[F128]) -> F128 {
    assert_eq!(z_row.len(), params.n);
    assert_eq!(z_col.len(), params.k);
    assert_eq!(z_index.len(), params.m);
    let eq_col = build_eq_table(z_col);
    let cols = 1usize << params.k;
    let mut acc_unred = crate::field::F256Unreduced::ZERO;
    for c in 0..cols {
        let g = g_hat_eval(
            z_row,
            z_index,
            params.col_prefix_sums[c],
            params.col_prefix_sums[c + 1],
            params.m,
        );
        acc_unred ^= eq_col[c].mul_unreduced(g);
    }
    acc_unred.reduce()
}

#[derive(Clone, Debug)]
pub struct JaggedSumcheckProof {
    pub rounds: Vec<(F128, F128)>,
    pub q_eval: F128,
}

#[derive(Clone, Debug)]
pub struct DenseClaim {
    pub point: Vec<F128>,
    pub alpha: F128,
}

fn generate_f_and_claim(
    params: &JaggedParams,
    q: &[F128],
    z_row: &[F128],
    z_col: &[F128],
) -> (Vec<F128>, F128) {
    use rayon::prelude::*;
    let len = 1usize << params.m;
    let area = params.area() as usize;
    let eq_row = build_eq_table(z_row);
    let eq_col = build_eq_table(z_col);
    let prefix = &params.col_prefix_sums;
    let mut b = crate::alloc_uninit_f128_vec(len);

    const CHUNK: usize = 1 << 16;
    let v = b
        .par_chunks_mut(CHUNK)
        .enumerate()
        .map(|(ci, b_chunk)| {
            let g0 = ci * CHUNK;
            let q_chunk = &q[g0..g0 + b_chunk.len()];

            let mut col = prefix
                .partition_point(|&t| t <= g0 as u64)
                .saturating_sub(1);
            let mut acc_unred = crate::field::F256Unreduced::ZERO;
            for (local, slot) in b_chunk.iter_mut().enumerate() {
                let i = g0 + local;
                if i >= area {
                    *slot = F128::ZERO;
                    continue;
                }
                while (i as u64) >= prefix[col + 1] {
                    col += 1;
                }
                let row = i - prefix[col] as usize;
                let bi = eq_row[row] * eq_col[col];
                *slot = bi;
                acc_unred ^= q_chunk[local].mul_unreduced(bi);
            }
            acc_unred.reduce()
        })
        .reduce(|| F128::ZERO, |x, y| x + y);
    (b, v)
}

pub fn prove<C: Challenger>(
    params: &JaggedParams,
    q: &[F128],
    z_row: &[F128],
    z_col: &[F128],
    challenger: &mut C,
) -> (JaggedSumcheckProof, F128) {
    let m = params.m;
    let len = 1usize << m;
    assert_eq!(q.len(), len, "q must have 2^m entries");
    assert_eq!(z_row.len(), params.n);
    assert_eq!(z_col.len(), params.k);
    challenger.observe_label(b"flock-jagged-v0");

    let (b, v) = generate_f_and_claim(params, q, z_row, z_col);

    let mut a = q.to_vec();
    let mut bb = b;
    let mut sa = crate::alloc_uninit_f128_vec(len / 2);
    let mut sb = crate::alloc_uninit_f128_vec(len / 2);
    let mut cur = len;
    let mut rounds = Vec::with_capacity(m);
    let (mut g_one, mut g_inf) = round_msg_par(&a[..cur], &bb[..cur]);
    for _ in 0..m {
        let half = cur / 2;
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        rounds.push((g_one, g_inf));
        if cur > 2 {
            (g_one, g_inf) =
                fold_and_round_oop_par(&a[..cur], &bb[..cur], r, &mut sa[..half], &mut sb[..half]);
        } else {
            fold_oop_par(&a[..cur], &bb[..cur], r, &mut sa[..half], &mut sb[..half]);
        }
        std::mem::swap(&mut a, &mut sa);
        std::mem::swap(&mut bb, &mut sb);
        cur = half;
    }

    debug_assert_eq!(cur, 1);
    let proof = JaggedSumcheckProof {
        rounds,
        q_eval: a[0],
    };
    (proof, v)
}

pub fn verify<C: Challenger>(
    params: &JaggedParams,
    z_row: &[F128],
    z_col: &[F128],
    claim_v: F128,
    proof: &JaggedSumcheckProof,
    challenger: &mut C,
) -> Option<DenseClaim> {
    let m = params.m;
    if proof.rounds.len() != m {
        return None;
    }
    challenger.observe_label(b"flock-jagged-v0");

    let mut claim = claim_v;
    let mut point = Vec::with_capacity(m);
    for &(g_one, g_inf) in &proof.rounds {
        challenger.observe_f128(g_one);
        challenger.observe_f128(g_inf);
        let r = challenger.sample_f128();
        claim = fold_round_claim(claim, g_one, g_inf, r);
        point.push(r);
    }

    let beta = f_hat_t(params, z_row, z_col, &point);
    if claim == proof.q_eval * beta {
        Some(DenseClaim {
            point,
            alpha: proof.q_eval,
        })
    } else {
        None
    }
}

#[inline]
fn fold_round_claim(claim: F128, g_one: F128, g_inf: F128, r: F128) -> F128 {
    let g0 = claim + g_one;

    g0 + (g_one + g0 + g_inf) * r + g_inf * (r * r)
}

#[allow(dead_code)]
#[inline]
fn round_msg(a: &[F128], b: &[F128]) -> (F128, F128) {
    let half = a.len() / 2;
    let mut g_one = F128::ZERO;
    let mut g_inf = F128::ZERO;
    for x in 0..half {
        let (a0, a1) = (a[2 * x], a[2 * x + 1]);
        let (b0, b1) = (b[2 * x], b[2 * x + 1]);
        g_one += a1 * b1;
        g_inf += (a0 + a1) * (b0 + b1);
    }
    (g_one, g_inf)
}

#[allow(dead_code)]
fn fold_and_round_fused(a: &mut Vec<F128>, b: &mut Vec<F128>, r: F128) -> (F128, F128) {
    let n = a.len();
    debug_assert!(n >= 4 && n.is_power_of_two());
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    let pairs = half / 2;
    let mut g_one = F128::ZERO;
    let mut g_inf = F128::ZERO;
    for xp in 0..pairs {
        let base = 4 * xp;

        let na0 = a[base] + r * (a[base + 1] + a[base]);
        let na1 = a[base + 2] + r * (a[base + 3] + a[base + 2]);
        let nb0 = b[base] + r * (b[base + 1] + b[base]);
        let nb1 = b[base + 2] + r * (b[base + 3] + b[base + 2]);
        a[2 * xp] = na0;
        a[2 * xp + 1] = na1;
        b[2 * xp] = nb0;
        b[2 * xp + 1] = nb1;

        g_one += na1 * nb1;
        g_inf += (na0 + na1) * (nb0 + nb1);
    }
    a.truncate(half);
    b.truncate(half);
    (g_one, g_inf)
}

fn round_msg_par(a: &[F128], b: &[F128]) -> (F128, F128) {
    use rayon::prelude::*;
    const C: usize = 1 << 14;
    a.par_chunks(C)
        .zip(b.par_chunks(C))
        .map(|(ac, bc)| {
            let mut g1_unred = crate::field::F256Unreduced::ZERO;
            let mut gi_unred = crate::field::F256Unreduced::ZERO;
            for (ap, bp) in ac
                .as_chunks::<2>()
                .0
                .iter()
                .zip(bc.as_chunks::<2>().0.iter())
            {
                g1_unred ^= ap[1].mul_unreduced(bp[1]);
                gi_unred ^= (ap[0] + ap[1]).mul_unreduced(bp[0] + bp[1]);
            }
            (g1_unred.reduce(), gi_unred.reduce())
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t))
}

fn fold_oop_par(a: &[F128], b: &[F128], r: F128, ao: &mut [F128], bo: &mut [F128]) {
    use rayon::prelude::*;
    ao.par_iter_mut()
        .zip(bo.par_iter_mut())
        .enumerate()
        .for_each(|(x, (oa, ob))| {
            *oa = a[2 * x].mle_fold(a[2 * x + 1], r);
            *ob = b[2 * x].mle_fold(b[2 * x + 1], r);
        });
}

fn fold_and_round_oop_par(
    a: &[F128],
    b: &[F128],
    r: F128,
    ao: &mut [F128],
    bo: &mut [F128],
) -> (F128, F128) {
    use rayon::prelude::*;
    debug_assert_eq!(a.len(), 2 * ao.len());
    debug_assert!(a.len() >= 4);

    const CO: usize = 1 << 13;
    ao.par_chunks_mut(CO)
        .zip(bo.par_chunks_mut(CO))
        .zip(a.par_chunks(2 * CO))
        .zip(b.par_chunks(2 * CO))
        .map(|(((oa, ob), ain), bin)| {
            let mut g1_unred = crate::field::F256Unreduced::ZERO;
            let mut gi_unred = crate::field::F256Unreduced::ZERO;
            for (((op, opb), aq), bq) in oa
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(ob.as_chunks_mut::<2>().0.iter_mut())
                .zip(ain.as_chunks::<4>().0.iter())
                .zip(bin.as_chunks::<4>().0.iter())
            {
                let na0 = aq[0].mle_fold(aq[1], r);
                let na1 = aq[2].mle_fold(aq[3], r);
                let nb0 = bq[0].mle_fold(bq[1], r);
                let nb1 = bq[2].mle_fold(bq[3], r);
                op[0] = na0;
                op[1] = na1;
                opb[0] = nb0;
                opb[1] = nb1;
                g1_unred ^= na1.mul_unreduced(nb1);
                gi_unred ^= (na0 + na1).mul_unreduced(nb0 + nb1);
            }
            (g1_unred.reduce(), gi_unred.reduce())
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::{FsChallenger, RandomChallenger};
    use crate::zerocheck::multilinear::fold_in_place_pair;

    fn sample_vec(ch: &mut RandomChallenger, n: usize) -> Vec<F128> {
        (0..n).map(|_| ch.sample_f128()).collect()
    }

    fn f_hat_t_bruteforce(
        params: &JaggedParams,
        z_row: &[F128],
        z_col: &[F128],
        z_index: &[F128],
    ) -> F128 {
        let eq_row = build_eq_table(z_row);
        let eq_col = build_eq_table(z_col);
        let eq_idx = build_eq_table(z_index);
        let mut acc = F128::ZERO;
        for i in 0..params.area() {
            let (row, col) = params.unrank(i);
            acc += eq_row[row] * eq_col[col] * eq_idx[i as usize];
        }
        acc
    }

    fn mle_eval(q: &[F128], point: &[F128]) -> F128 {
        let eq = build_eq_table(point);
        q.iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |s, x| s + x)
    }

    fn random_instance(
        ch: &mut RandomChallenger,
        n: usize,
        k: usize,
        m: usize,
    ) -> (JaggedParams, Vec<F128>) {
        let cols = 1usize << k;
        let cap = 1u64 << m;
        let max_h = 1u64 << n;

        let mut heights = vec![0u64; cols];
        let mut remaining = cap;
        for h in heights.iter_mut() {
            let r = ch.sample_f128().lo % (max_h + 1);
            let take = r.min(remaining);
            *h = take;
            remaining -= take;
        }
        let params = JaggedParams::from_heights(&heights, n, m);

        let mut q = vec![F128::ZERO; 1usize << m];
        for qi in q.iter_mut().take(params.area() as usize) {
            *qi = ch.sample_f128();
        }
        (params, q)
    }

    #[test]
    fn f_hat_t_matches_bruteforce() {
        let mut ch = RandomChallenger::new(0x1A66_ED12);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (2, 4, 6), (5, 1, 5)] {
            for _ in 0..8 {
                let (params, _q) = random_instance(&mut ch, n, k, m);
                let z_row = sample_vec(&mut ch, n);
                let z_col = sample_vec(&mut ch, k);
                let z_idx = sample_vec(&mut ch, m);
                let got = f_hat_t(&params, &z_row, &z_col, &z_idx);
                let want = f_hat_t_bruteforce(&params, &z_row, &z_col, &z_idx);
                assert_eq!(got, want, "f̂_t mismatch for n={n} k={k} m={m}");
            }
        }
    }

    #[test]
    fn f_hat_t_eq4_at_boolean_points() {
        let mut ch = RandomChallenger::new(0xB001_2345);
        let (params, _q) = random_instance(&mut ch, 4, 3, 7);
        let z_row = sample_vec(&mut ch, 4);
        let z_col = sample_vec(&mut ch, 3);
        let eq_row = build_eq_table(&z_row);
        let eq_col = build_eq_table(&z_col);
        for i in 0..params.area() {
            let z_idx: Vec<F128> = (0..params.m).map(|bit| int_bit(i, bit)).collect();
            let got = f_hat_t(&params, &z_row, &z_col, &z_idx);
            let (row, col) = params.unrank(i);
            let want = eq_row[row] * eq_col[col];
            assert_eq!(got, want, "Eq.4 failed at boolean i={i}");
        }
    }

    #[test]
    fn sumcheck_roundtrip() {
        let mut ch = RandomChallenger::new(0x5C4E_CC01);
        for &(n, k, m) in &[(3usize, 2usize, 5usize), (4, 3, 7), (2, 4, 6)] {
            for _ in 0..5 {
                let (params, q) = random_instance(&mut ch, n, k, m);
                let z_row = sample_vec(&mut ch, n);
                let z_col = sample_vec(&mut ch, k);

                let mut pch = FsChallenger::new(b"flock-jagged-test");
                let (proof, v) = prove(&params, &q, &z_row, &z_col, &mut pch);

                let mut vch = FsChallenger::new(b"flock-jagged-test");
                let claim = verify(&params, &z_row, &z_col, v, &proof, &mut vch)
                    .expect("honest proof must verify");

                assert_eq!(claim.alpha, mle_eval(&q, &claim.point), "alpha ≠ q̂(i*)");
            }
        }
    }

    #[test]
    fn sumcheck_rejects_wrong_value() {
        let mut ch = RandomChallenger::new(0xBAD0_C1A1);
        let (params, q) = random_instance(&mut ch, 4, 3, 7);
        let z_row = sample_vec(&mut ch, 4);
        let z_col = sample_vec(&mut ch, 3);

        let mut pch = FsChallenger::new(b"flock-jagged-test");
        let (proof, v) = prove(&params, &q, &z_row, &z_col, &mut pch);

        let mut vch = FsChallenger::new(b"flock-jagged-test");
        let bad = v + F128::ONE;
        assert!(
            verify(&params, &z_row, &z_col, bad, &proof, &mut vch).is_none(),
            "verifier must reject a wrong claim value"
        );
    }

    #[test]
    #[ignore = "heavy benchmark; run explicitly with --release --ignored --nocapture"]
    fn runtime_m25() {
        use std::time::Instant;

        let _ = crate::init_perf_thread_pool();
        let (n, k, m) = (13usize, 12usize, 25usize);
        let cols = 1usize << k;
        let height = (1u64 << m) / cols as u64;
        let params = JaggedParams::from_heights(&vec![height; cols], n, m);
        assert_eq!(params.area(), 1u64 << m);

        let len = 1usize << m;
        let mut q = vec![F128::ZERO; len];
        for (i, qi) in q.iter_mut().enumerate() {
            *qi = F128 {
                lo: i as u64,
                hi: (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            };
        }
        let mut rc = RandomChallenger::new(0x0B7A_4225);
        let z_row = sample_vec(&mut rc, n);
        let z_col = sample_vec(&mut rc, k);

        let mb = (len * std::mem::size_of::<F128>()) as f64 / (1024.0 * 1024.0);
        eprintln!("\n[jagged runtime] m={m} ({len} F128 = {mb:.0} MB), n={n}, k={k}, cols={cols}");

        const REPS: usize = 3;

        let mut t_gen_ser = std::time::Duration::MAX;
        let mut t_gen_par = std::time::Duration::MAX;
        let (mut b, mut v) = (Vec::new(), F128::ZERO);
        for _ in 0..REPS {
            let t0 = Instant::now();
            let eq_row = build_eq_table(&z_row);
            let eq_col = build_eq_table(&z_col);
            let mut bs = vec![F128::ZERO; len];
            for col in 0..cols {
                let start = params.col_prefix_sums[col] as usize;
                let end = params.col_prefix_sums[col + 1] as usize;
                let ec = eq_col[col];
                for (row, slot) in bs[start..end].iter_mut().enumerate() {
                    *slot = eq_row[row] * ec;
                }
            }
            let mut vs = F128::ZERO;
            for (qi, bi) in q.iter().zip(bs.iter()) {
                vs += *qi * *bi;
            }
            t_gen_ser = t_gen_ser.min(t0.elapsed());
            std::hint::black_box(&bs);

            let t1 = Instant::now();
            let (bp, vp) = generate_f_and_claim(&params, &q, &z_row, &z_col);
            t_gen_par = t_gen_par.min(t1.elapsed());
            assert_eq!(vs, vp, "parallel gen must match serial");
            b = bp;
            v = vp;
        }
        let _ = v;

        let run_serial = |fused: bool| -> std::time::Duration {
            let mut a = q.clone();
            let mut bb = b.clone();
            let mut ch = FsChallenger::new(b"flock-jagged-bench");
            ch.observe_label(b"flock-jagged-v0");
            let t = Instant::now();
            if fused {
                let (mut g1, mut gi) = round_msg(&a, &bb);
                for _ in 0..m {
                    ch.observe_f128(g1);
                    ch.observe_f128(gi);
                    let r = ch.sample_f128();
                    if a.len() > 2 {
                        (g1, gi) = fold_and_round_fused(&mut a, &mut bb, r);
                    } else {
                        fold_in_place_pair(&mut a, &mut bb, r);
                    }
                }
            } else {
                for _ in 0..m {
                    let (g1, gi) = round_msg(&a, &bb);
                    ch.observe_f128(g1);
                    ch.observe_f128(gi);
                    let r = ch.sample_f128();
                    fold_in_place_pair(&mut a, &mut bb, r);
                }
            }
            std::hint::black_box(a[0]);
            t.elapsed()
        };

        let run_par = |fused: bool| -> std::time::Duration {
            let mut a = q.clone();
            let mut bb = b.clone();
            let mut sa = vec![F128::ZERO; len / 2];
            let mut sb = vec![F128::ZERO; len / 2];
            let mut cur = len;
            let mut ch = FsChallenger::new(b"flock-jagged-bench");
            ch.observe_label(b"flock-jagged-v0");
            let t = Instant::now();
            let (mut g1, mut gi) = if fused {
                round_msg_par(&a[..cur], &bb[..cur])
            } else {
                (F128::ZERO, F128::ZERO)
            };
            for _ in 0..m {
                let half = cur / 2;
                if !fused {
                    let (m1, mi) = round_msg_par(&a[..cur], &bb[..cur]);
                    g1 = m1;
                    gi = mi;
                }
                ch.observe_f128(g1);
                ch.observe_f128(gi);
                let r = ch.sample_f128();
                if fused && cur > 2 {
                    let (n1, ni) = fold_and_round_oop_par(
                        &a[..cur],
                        &bb[..cur],
                        r,
                        &mut sa[..half],
                        &mut sb[..half],
                    );
                    g1 = n1;
                    gi = ni;
                } else {
                    fold_oop_par(&a[..cur], &bb[..cur], r, &mut sa[..half], &mut sb[..half]);
                }
                std::mem::swap(&mut a, &mut sa);
                std::mem::swap(&mut bb, &mut sb);
                cur = half;
            }
            std::hint::black_box(a[0]);
            t.elapsed()
        };

        let mut s_unf = std::time::Duration::MAX;
        let mut s_fus = std::time::Duration::MAX;
        let mut p_unf = std::time::Duration::MAX;
        let mut p_fus = std::time::Duration::MAX;
        for _ in 0..REPS {
            s_unf = s_unf.min(run_serial(false));
            s_fus = s_fus.min(run_serial(true));
            p_unf = p_unf.min(run_par(false));
            p_fus = p_fus.min(run_par(true));
        }

        let point: Vec<F128> = (0..m).map(|_| rc.sample_f128()).collect();
        let t2 = Instant::now();
        let beta = f_hat_t(&params, &z_row, &z_col, &point);
        std::hint::black_box(beta);
        let t_ver = t2.elapsed();

        let ratio = |unf: std::time::Duration, fus: std::time::Duration| {
            unf.as_secs_f64() / fus.as_secs_f64()
        };
        eprintln!("  threads: {}", rayon::current_num_threads());
        eprintln!(
            "  f̂_t-gen (B + claim) serial {:>8.1?} → parallel {:>8.1?}   ({:.2}x)",
            t_gen_ser,
            t_gen_par,
            ratio(t_gen_ser, t_gen_par)
        );
        eprintln!("                          unfused      fused     fusion");
        eprintln!(
            "  sumcheck serial   : {:>9.1?}  {:>9.1?}   {:.2}x",
            s_unf,
            s_fus,
            ratio(s_unf, s_fus)
        );
        eprintln!(
            "  sumcheck parallel : {:>9.1?}  {:>9.1?}   {:.2}x   (vs serial unfused {:.2}x)",
            p_unf,
            p_fus,
            ratio(p_unf, p_fus),
            ratio(s_unf, p_fus)
        );
        eprintln!("  verifier f̂_t eval            : {:>9.3?}", t_ver);
        let best = p_unf.min(p_fus);
        eprintln!(
            "  best prover total (gen + best sumcheck): {:.1?} ({:.2} ns/elem)\n",
            t_gen_par + best,
            (t_gen_par + best).as_nanos() as f64 / len as f64
        );
    }

    #[test]
    #[ignore = "diagnostic; run with --release --ignored --nocapture"]
    fn scaling_diag() {
        use rayon::prelude::*;
        use std::time::{Duration, Instant};
        let _ = crate::init_perf_thread_pool();
        let m = 25usize;
        let len = 1usize << m;
        let half = len / 2;
        let a: Vec<F128> = (0..len)
            .map(|i| F128 {
                lo: i as u64,
                hi: i as u64,
            })
            .collect();
        let b = a.clone();
        let r = F128 {
            lo: 0x9E37,
            hi: 0x1234,
        };
        const REPS: usize = 6;
        const CHUNK: usize = 1 << 13;

        let bench = |f: &mut dyn FnMut()| {
            let mut t = Duration::MAX;
            for _ in 0..REPS {
                let t0 = Instant::now();
                f();
                t = t.min(t0.elapsed());
            }
            t
        };
        let sp = |s: Duration, p: Duration| s.as_secs_f64() / p.as_secs_f64();
        let gbps = |bytes: usize, t: Duration| bytes as f64 / t.as_secs_f64() / 1e9;

        eprintln!(
            "\n[scaling diag] m={m}, threads={}",
            rayon::current_num_threads()
        );

        let mut dst = crate::alloc_uninit_f128_vec(len);
        let ts = bench(&mut || dst.copy_from_slice(&a));
        let tp = bench(&mut || {
            dst.par_chunks_mut(CHUNK)
                .enumerate()
                .for_each(|(ci, d)| d.copy_from_slice(&a[ci * CHUNK..ci * CHUNK + d.len()]));
        });
        let bytes = len * 32;
        eprintln!(
            "  memcpy        : serial {:>7.1?} ({:>4.0} GB/s)  parallel {:>7.1?} ({:>4.0} GB/s)  {:.2}x",
            ts,
            gbps(bytes, ts),
            tp,
            gbps(bytes, tp),
            sp(ts, tp)
        );

        let mut out = crate::alloc_uninit_f128_vec(half);
        let ts = bench(&mut || {
            for x in 0..half {
                out[x] = a[2 * x] + r * (a[2 * x + 1] + a[2 * x]);
            }
        });
        let tp_fine = bench(&mut || {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(x, o)| *o = a[2 * x] + r * (a[2 * x + 1] + a[2 * x]));
        });
        let tp_coarse = bench(&mut || {
            out.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, oc)| {
                let x0 = ci * CHUNK;
                for (j, o) in oc.iter_mut().enumerate() {
                    let x = x0 + j;
                    *o = a[2 * x] + r * (a[2 * x + 1] + a[2 * x]);
                }
            });
        });
        let bytes = half * 16 * 3;
        eprintln!(
            "  fold   serial {:>7.1?} ({:>4.0} GB/s)  par.fine {:>7.1?} {:.2}x  par.coarse {:>7.1?} {:.2}x",
            ts,
            gbps(bytes, ts),
            tp_fine,
            sp(ts, tp_fine),
            tp_coarse,
            sp(ts, tp_coarse)
        );

        let ts = bench(&mut || {
            std::hint::black_box(round_msg(&a, &b));
        });
        let tp_fine = bench(&mut || {
            std::hint::black_box(round_msg_par(&a, &b));
        });

        let tp_coarse = bench(&mut || {
            let acc = (0..half)
                .into_par_iter()
                .with_min_len(CHUNK)
                .fold(
                    || (F128::ZERO, F128::ZERO),
                    |(g1, gi), x| {
                        let (a0, a1) = (a[2 * x], a[2 * x + 1]);
                        let (b0, b1) = (b[2 * x], b[2 * x + 1]);
                        (g1 + a1 * b1, gi + (a0 + a1) * (b0 + b1))
                    },
                )
                .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t));
            std::hint::black_box(acc);
        });
        let rd_bytes = len * 16 * 2;
        eprintln!(
            "  round_msg serial {:>7.1?} ({:>4.0} GB/s)  par.fine {:>7.1?} {:.2}x  par.coarse {:>7.1?} {:.2}x",
            ts,
            gbps(rd_bytes, ts),
            tp_fine,
            sp(ts, tp_fine),
            tp_coarse,
            sp(ts, tp_coarse)
        );

        let tp_slice = bench(&mut || {
            let acc = a
                .par_chunks(2 * CHUNK)
                .zip(b.par_chunks(2 * CHUNK))
                .map(|(ac, bc)| {
                    let mut g1 = F128::ZERO;
                    let mut gi = F128::ZERO;
                    for (ap, bp) in ac
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(bc.as_chunks::<2>().0.iter())
                    {
                        g1 += ap[1] * bp[1];
                        gi += (ap[0] + ap[1]) * (bp[0] + bp[1]);
                    }
                    (g1, gi)
                })
                .reduce(|| (F128::ZERO, F128::ZERO), |(p, q), (s, t)| (p + s, q + t));
            std::hint::black_box(acc);
        });
        eprintln!(
            "  round_msg par.slice(chunks_exact)   {:>7.1?} ({:>4.0} GB/s)  {:.2}x",
            tp_slice,
            gbps(rd_bytes, tp_slice),
            sp(ts, tp_slice)
        );

        let mut sa = crate::alloc_uninit_f128_vec(half);
        let mut sb = crate::alloc_uninit_f128_vec(half);
        let mut round_t = vec![Duration::MAX; m + 1];
        for _ in 0..REPS {
            let mut av = a.clone();
            let mut bv = b.clone();
            let mut cur = len;
            let t0 = Instant::now();
            let (mut _g1, mut _gi) = round_msg_par(&av[..cur], &bv[..cur]);
            round_t[0] = round_t[0].min(t0.elapsed());
            for rd in 0..m {
                let half_r = cur / 2;
                let t = Instant::now();
                if cur > 2 {
                    (_g1, _gi) = fold_and_round_oop_par(
                        &av[..cur],
                        &bv[..cur],
                        r,
                        &mut sa[..half_r],
                        &mut sb[..half_r],
                    );
                } else {
                    fold_oop_par(
                        &av[..cur],
                        &bv[..cur],
                        r,
                        &mut sa[..half_r],
                        &mut sb[..half_r],
                    );
                }
                std::mem::swap(&mut av, &mut sa);
                std::mem::swap(&mut bv, &mut sb);
                round_t[rd + 1] = round_t[rd + 1].min(t.elapsed());
                cur = half_r;
            }
        }
        let total: Duration = round_t.iter().sum();
        let tail: Duration = round_t[6..].iter().sum();
        eprintln!(
            "  sumcheck per-round: total {:.1?} | r0 {:.1?} r1 {:.1?} r2 {:.1?} | tail(r6+, cur≤2^19) {:.2?} ({:.0}%)",
            total,
            round_t[1],
            round_t[2],
            round_t[3],
            tail,
            100.0 * tail.as_secs_f64() / total.as_secs_f64()
        );
        std::hint::black_box(&out);
        std::hint::black_box(&dst);
    }

    #[test]
    fn sumcheck_rejects_tampered_proof() {
        let mut ch = RandomChallenger::new(0xDEAD_BEEF);
        let (params, q) = random_instance(&mut ch, 3, 3, 6);
        let z_row = sample_vec(&mut ch, 3);
        let z_col = sample_vec(&mut ch, 3);

        let mut pch = FsChallenger::new(b"flock-jagged-test");
        let (mut proof, v) = prove(&params, &q, &z_row, &z_col, &mut pch);
        proof.q_eval += F128::ONE;

        let mut vch = FsChallenger::new(b"flock-jagged-test");
        assert!(
            verify(&params, &z_row, &z_col, v, &proof, &mut vch).is_none(),
            "verifier must reject a tampered q_eval"
        );
    }
}

// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// EXP-0016 test-only, depth-two specialization of the sparse-dual algebra in
// the pinned x86 ligerito.rs (c75aeceb34f8f33ff49c06a293b05dd7e00d463b).
// Uses Apple's existing transpose and an explicit merge, not x86 fold plumbing.

use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Default, Serialize)]
struct Work {
    constructor_products: usize,
    row_reads: usize,
    message_products: [usize; 3],
    scaling_products: usize,
    reduced_coeff_products: usize,
    reduced_transforms: usize,
    reduced_log_d: usize,
    reduced_active_windows: usize,
    merge_additions: usize,
}

fn mul(a: F128, b: F128, count: &mut usize) -> F128 {
    *count += 1;
    a * b
}

fn dot(a: &[F128], b: &[F128], count: &mut usize) -> F128 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(F128::ZERO, |s, (&x, &y)| s + mul(x, y, count))
}

/// An exact row of a local forward NTT, generated without a dense matrix.
fn local_row(
    ntt: &AdditiveNttF128,
    log_d: usize,
    k: usize,
    query: usize,
    count: &mut usize,
) -> [F128; 8] {
    let mut row = [F128::ZERO; 8];
    row[0] = F128::ONE;
    let base = query & !((1usize << k) - 1);
    let p = query - base;
    for s in 0..k {
        let layer = log_d - 1 - s;
        let block = (base >> (s + 1)) + (p >> (s + 1));
        let mut scale = ntt.twiddle(layer, block);
        if p & (1 << s) != 0 {
            scale += F128::ONE;
        }
        for i in 0..1usize << s {
            row[(1 << s) + i] = mul(row[i], scale, count);
        }
    }
    row
}

struct Dual {
    ntt: AdditiveNttF128,
    log_d: usize,
    log_inv_rate: usize,
    queries: Vec<usize>,
    weights: Vec<F128>,
    residues: Vec<[F128; 8]>,
    work: Work,
}

impl Dual {
    fn new<'a>(
        log_d: usize,
        log_inv_rate: usize,
        lane_challenges: &[F128],
        queries: &[usize],
        alpha: &[F128],
        mut row: impl FnMut(usize) -> &'a [F128],
    ) -> (Self, F128) {
        assert!(log_inv_rate + 2 < log_d);
        assert!(queries.iter().all(|&q| q < 1usize << log_d));
        let lane_weights = build_eq_table(lane_challenges);
        let weights: Vec<F128> = if queries.is_empty() {
            Vec::new()
        } else {
            build_eq_table(alpha)
                .into_iter()
                .take(queries.len())
                .collect()
        };
        assert_eq!(queries.len(), weights.len());
        let ntt = AdditiveNttF128::standard(log_d);
        let mut work = Work::default();
        work.constructor_products = lane_weights.len() - 1
            + if queries.is_empty() {
                0
            } else {
                (1usize << alpha.len()) - 1
            };
        let mut residues = Vec::with_capacity(queries.len());
        let mut enforced = F128::ZERO;
        for (&q, &alpha) in queries.iter().zip(&weights) {
            let base = q & !7usize;
            let mut cache = [F128::ZERO; 8];
            for j in 0..8 {
                cache[j] = dot(row(base + j), &lane_weights, &mut work.constructor_products);
                work.row_reads += 1;
            }
            enforced += mul(alpha, cache[q - base], &mut work.constructor_products);
            // Invert the final three forward layers of the valid codeword.
            for layer in ((log_d - 3)..log_d).rev() {
                let size = 1usize << (log_d - layer);
                let half = size / 2;
                for off in (0..8).step_by(size) {
                    let twiddle = ntt.twiddle(layer, (base + off) / size);
                    for j in 0..half {
                        let y0 = cache[off + j];
                        let v = y0 + cache[off + half + j];
                        cache[off + j] = y0 + mul(v, twiddle, &mut work.constructor_products);
                        cache[off + half + j] = v;
                    }
                }
            }
            residues.push(cache);
        }
        (
            Self {
                ntt,
                log_d,
                log_inv_rate,
                queries: queries.to_vec(),
                weights,
                residues,
                work,
            },
            enforced,
        )
    }

    fn message(&mut self, rs: &[F128]) -> SumcheckMessage {
        assert!(rs.len() <= 2);
        let k = rs.len() + 1;
        let size = 1usize << k;
        let half = size / 2;
        let weights = build_eq_table(rs);
        let count = &mut self.work.message_products[rs.len()];
        *count += weights.len() - 1;
        let mut msg = SumcheckMessage {
            u_0: F128::ZERO,
            u_2: F128::ZERO,
        };
        for i in 0..self.queries.len() {
            let q = self.queries[i];
            let cache_base = q & !7usize;
            let mut cache = self.residues[i];
            for layer in (self.log_d - 3)..(self.log_d - k) {
                let local_size = 1usize << (self.log_d - layer);
                let local_half = local_size / 2;
                for off in (0..8).step_by(local_size) {
                    let twiddle = self.ntt.twiddle(layer, (cache_base + off) / local_size);
                    for j in 0..local_half {
                        let v = cache[off + local_half + j];
                        let u = cache[off + j] + mul(v, twiddle, count);
                        cache[off + j] = u;
                        cache[off + local_half + j] = v + u;
                    }
                }
            }
            let off = (q & !(size - 1)) - cache_base;
            let f0 = dot(&cache[off..off + half], &weights, count);
            let f1 = dot(&cache[off + half..off + size], &weights, count);
            let row = local_row(&self.ntt, self.log_d, k, q, count);
            let dot0 = dot(&row[..half], &weights, count);
            let mut dot2 = F128::ZERO;
            for j in 0..half {
                dot2 += mul(row[j] + row[half + j], weights[j], count);
            }
            let a0 = mul(self.weights[i], f0, count);
            let a2 = mul(self.weights[i], f0 + f1, count);
            msg.u_0 += mul(a0, dot0, count);
            msg.u_2 += mul(a2, dot2, count);
        }
        msg
    }

    fn scale(&mut self, beta: F128) {
        for w in &mut self.weights {
            *w = mul(*w, beta, &mut self.work.scaling_products);
        }
    }

    fn materialize(&mut self, rs: &[F128]) -> Vec<F128> {
        assert_eq!(rs.len(), 2);
        assert_eq!(self.work.reduced_transforms, 0, "materialized twice");
        let weights = build_eq_table(rs);
        let count = &mut self.work.reduced_coeff_products;
        *count += 3;
        let mut positions = Vec::with_capacity(self.queries.len());
        let mut values = Vec::with_capacity(self.queries.len());
        for (&q, &alpha) in self.queries.iter().zip(&self.weights) {
            let row = local_row(&self.ntt, self.log_d, 2, q, count);
            let v = dot(&row[..4], &weights, count);
            positions.push(q >> 2);
            values.push(mul(alpha, v, count));
        }
        let reduced_log_d = self.log_d - 2;
        // Keep the original NTT's twiddles. A fresh NTT at the smaller
        // dimension would describe a different linear map.
        let mut basis =
            transpose_forward_ntt_sparse(&self.ntt, &positions, &values, reduced_log_d, false);
        self.work.reduced_transforms += 1;
        self.work.reduced_log_d = reduced_log_d;
        self.work.reduced_active_windows = positions
            .iter()
            .map(|p| p >> 8)
            .collect::<BTreeSet<_>>()
            .len();
        basis.truncate(1usize << (reduced_log_d - self.log_inv_rate));
        basis
    }
}

fn artifact_fields(dir: &Path, name: &str, values: &[F128]) {
    let mut data = Vec::with_capacity(16 * values.len());
    for v in values {
        data.extend_from_slice(&v.lo.to_le_bytes());
        data.extend_from_slice(&v.hi.to_le_bytes());
    }
    std::fs::write(dir.join(format!("{name}.bin")), data).unwrap();
}

fn artifact_json(dir: &Path, name: &str, value: &impl Serialize) {
    std::fs::write(
        dir.join(format!("{name}.json")),
        serde_json::to_vec_pretty(value).unwrap(),
    )
    .unwrap();
}

fn same_fields(dir: &Path, name: &str, got: &[F128], want: &[F128]) {
    artifact_fields(dir, name, want);
    if got != want {
        artifact_fields(dir, &format!("{name}-FAILED-candidate"), got);
    }
    assert!(
        got == want,
        "{name}: field-vector mismatch (full vectors saved)"
    );
}

struct Fixture {
    log_msg: usize,
    log_rate: usize,
    lanes: Vec<F128>,
    alpha: Vec<F128>,
    queries: Vec<usize>,
    rows: BTreeMap<usize, Vec<F128>>,
    f: Vec<F128>,
}

fn compact_fixture(shape: usize, case: usize, dir: &Path) -> Fixture {
    let (log_msg, log_rate, log_lanes) = [(5usize, 1usize, 0usize), (7, 1, 2), (10, 2, 6)][shape];
    let seed = 0xC2_0016_1000 + (shape * 16 + case) as u64;
    let mut ch = crate::challenger::RandomChallenger::new(seed);
    let mut poly = ch.sample_f128_vec((1usize << log_msg) * (1usize << log_lanes));
    if case == 9 {
        poly.fill(F128::ZERO);
    }
    let mut lanes = ch.sample_f128_vec(log_lanes);
    if case == 8 {
        lanes.fill(F128::ONE);
    }
    let f = partial_eval_lsb(&poly, &lanes);
    let n = 1usize << (log_msg + log_rate);
    let mut codeword = vec![F128::ZERO; n << log_lanes];
    codeword[..poly.len()].copy_from_slice(&poly);
    AdditiveNttF128::standard(log_msg + log_rate)
        .forward_transform_interleaved(&mut codeword, 1 << log_lanes);
    let mut queries = vec![0, 1, 2, 3, 6, 7, 8, 9, n / 2 - 1, n / 2, n - 2, n - 1];
    if case == 10 {
        queries.reverse();
    }
    if case == 11 {
        queries.clear();
    }
    let mut alpha = ch.sample_f128_vec(ceil_log2(queries.len().max(1)));
    if case == 8 {
        alpha.fill(F128::ZERO);
    }
    let needed: BTreeSet<usize> = queries
        .iter()
        .flat_map(|q| (q & !7)..(q & !7) + 8)
        .collect();
    let rows = needed
        .into_iter()
        .map(|q| (q, codeword[q << log_lanes..(q + 1) << log_lanes].to_vec()))
        .collect();
    artifact_fields(dir, "input-interleaved-message", &poly);
    artifact_json(
        dir,
        "fixture",
        &serde_json::json!({"seed":seed, "log_msg":log_msg,"log_rate":log_rate,"log_lanes":log_lanes,"case":case,"kind":"full_random_interleaved"}),
    );
    Fixture {
        log_msg,
        log_rate,
        lanes,
        alpha,
        queries,
        rows,
        f,
    }
}

fn ranked_fixture(dir: &Path) -> Fixture {
    const SEED: u64 = 0xC2_0016_AAAA;
    let mut ch = crate::challenger::RandomChallenger::new(SEED);
    let log_msg = 19;
    let log_rate = 1;
    let n = 1usize << (log_msg + log_rate);
    let p0 = ch.sample_f128_vec(n / 2);
    let p1 = ch.sample_f128_vec(n / 2);
    let v0 = ch.sample_f128_vec(64);
    let v1 = ch.sample_f128_vec(64);
    let lanes = ch.sample_f128_vec(6);
    let eq = build_eq_table(&lanes);
    let mut ignored = 0;
    let w0 = dot(&v0, &eq, &mut ignored);
    let w1 = dot(&v1, &eq, &mut ignored);
    let f = p0.iter().zip(&p1).map(|(&a, &b)| a * w0 + b * w1).collect();
    let mut c0 = vec![F128::ZERO; n];
    let mut c1 = vec![F128::ZERO; n];
    c0[..n / 2].copy_from_slice(&p0);
    c1[..n / 2].copy_from_slice(&p1);
    let ntt = AdditiveNttF128::standard(20);
    ntt.forward_transform(&mut c0);
    ntt.forward_transform(&mut c1);
    // Each lane is v0[lane]*p0 + v1[lane]*p1. Linearity makes these exact
    // rows of a valid 64-lane codeword without allocating its other 1 GiB.
    let mut queries = BTreeSet::from([0usize, 1, 2, 3, 6, 7, 8, 9, n / 2 - 1, n / 2, n - 2, n - 1]);
    let mut i = 0;
    while queries.len() < 218 {
        queries.insert((i * 4_801 + i * i * 17 + 29) & (n - 1));
        i += 1;
    }
    let queries: Vec<usize> = queries.into_iter().collect();
    let needed: BTreeSet<usize> = queries
        .iter()
        .flat_map(|q| (q & !7)..(q & !7) + 8)
        .collect();
    let rows = needed
        .into_iter()
        .map(|q| {
            (
                q,
                (0..64)
                    .map(|lane| c0[q] * v0[lane] + c1[q] * v1[lane])
                    .collect(),
            )
        })
        .collect();
    let alpha = ch.sample_f128_vec(8);
    artifact_fields(dir, "input-column0", &p0);
    artifact_fields(dir, "input-column1", &p1);
    artifact_fields(dir, "input-lane-vector0", &v0);
    artifact_fields(dir, "input-lane-vector1", &v1);
    artifact_json(
        dir,
        "fixture",
        &serde_json::json!({"seed":SEED,"log_msg":19,"log_rate":1,"log_lanes":6,"kind":"valid_rank_two_interleaved"}),
    );
    Fixture {
        log_msg,
        log_rate,
        lanes,
        alpha,
        queries,
        rows,
        f,
    }
}

fn check_case(fx: Fixture, case: usize, ranked: bool, dir: &Path) {
    check_production(&fx, case);
    artifact_fields(dir, "input-f", &fx.f);
    artifact_fields(dir, "input-lane-challenges", &fx.lanes);
    artifact_fields(dir, "input-alpha", &fx.alpha);
    artifact_json(dir, "queries", &fx.queries);
    artifact_json(
        dir,
        "neighborhood-row-indices",
        &fx.rows.keys().copied().collect::<Vec<_>>(),
    );
    artifact_fields(
        dir,
        "input-neighborhoods",
        &fx.rows.values().flatten().copied().collect::<Vec<_>>(),
    );
    let opened: Vec<Vec<F128>> = fx.queries.iter().map(|q| fx.rows[q].clone()).collect();
    let before_parent = TEST_INDUCE_FUSED_MSG_HITS.with(|n| n.get());
    let before_child = TEST_L0_FACTORED_FINAL3_HITS.with(|n| n.get());
    let sks = if ranked {
        Vec::new()
    } else {
        eval_sk_at_vks(fx.log_msg)
    };
    let (basis, enforced, precomputed) = induce_sumcheck_poly_auto_with_ranked_msg(
        fx.log_msg,
        fx.log_rate,
        &sks,
        &opened,
        &fx.lanes,
        &fx.queries,
        &fx.alpha,
        &fx.f,
    );
    let parent_hits = TEST_INDUCE_FUSED_MSG_HITS.with(|n| n.get()) - before_parent;
    let child_hits = TEST_L0_FACTORED_FINAL3_HITS.with(|n| n.get()) - before_child;
    if ranked {
        assert_eq!(
            (parent_hits, child_hits),
            (1, 1),
            "incumbent optimized path was not selected"
        );
        assert!(use_ranked_fused_densify_first(20, 8, 218));
        assert!(!use_ranked_fused_densify_first(18, 8, 218));
    }
    let (mut dual, got_enforced) = Dual::new(
        fx.log_msg + fx.log_rate,
        fx.log_rate,
        &fx.lanes,
        &fx.queries,
        &fx.alpha,
        |q| &fx.rows[&q],
    );
    assert_eq!(got_enforced, enforced, "enforced sum");
    let direct_intro = dual.message(&[]);
    let expected_intro = precomputed.unwrap_or_else(|| round_msg_lsb(&fx.f, &basis));
    artifact_json(
        dir,
        "introduction",
        &(enforced, direct_intro, expected_intro),
    );
    assert_eq!(direct_intro, expected_intro, "introduction message");
    artifact_fields(dir, "control-induced-basis", &basis);

    let mut ch = crate::challenger::RandomChallenger::new(0xC2_0016_B000 + case as u64);
    let initial_basis = ch.sample_f128_vec(fx.f.len());
    let h =
        fx.f.iter()
            .zip(&initial_basis)
            .fold(F128::ZERO, |s, (&x, &y)| s + x * y);
    let (mut control, m0) = SumcheckProver::new(fx.f.clone(), initial_basis.clone(), h);
    let (mut candidate, m1) = SumcheckProver::new(fx.f.clone(), initial_basis.clone(), h);
    assert_eq!(m0, m1);
    let mut z = ch.sample_f128_vec(fx.log_msg);
    z[0] = if case & 1 == 0 { F128::ZERO } else { F128::ONE };
    let ood0 = control.introduce_new_ood_factorized(&z).unwrap();
    let ood1 = candidate.introduce_new_ood_factorized(&z).unwrap();
    assert_eq!(ood0, ood1);
    let ood_beta = ch.sample_f128();
    control.glue_factorized_ood(ood_beta);
    candidate.glue_factorized_ood(ood_beta);
    let mut beta = ch.sample_f128();
    let mut rs = ch.sample_f128_vec(4);
    if !ranked && case < 8 {
        beta = if case & 1 == 0 { F128::ZERO } else { F128::ONE };
        rs[0] = if case & 2 == 0 { F128::ZERO } else { F128::ONE };
        rs[1] = if case & 4 == 0 { F128::ZERO } else { F128::ONE };
    }
    artifact_fields(dir, "input-initial-basis", &initial_basis);
    artifact_fields(dir, "input-ood-z", &z);
    artifact_fields(dir, "input-fold-challenges", &rs);
    artifact_fields(dir, "input-glue-challenges", &[ood_beta, beta]);
    let expected = if let Some(msg) = precomputed {
        control.introduce_new_with_precomputed_msg(basis.clone(), enforced, msg)
    } else {
        control.introduce_new(basis.clone(), enforced)
    };
    assert_eq!(expected, direct_intro);
    assert!(candidate.pending_glue.is_none() && candidate.pending_fold_basis.is_none());
    candidate.transcript.push(direct_intro);
    control.glue_deferred_into_lazy_ood_fold(beta);
    candidate.t_r += beta * got_enforced;
    dual.scale(beta);
    assert_eq!(candidate.t_r, control.t_r);
    assert_eq!(candidate.transcript, control.transcript);
    let scaled_basis: Vec<F128> = basis.iter().map(|&b| beta * b).collect();
    let mut negative_message_checks = 0;
    for depth in 1..=2 {
        let want = control.fold(rs[depth - 1]);
        let uncorrected = candidate.fold(rs[depth - 1]);
        let correction = dual.message(&rs[..depth]);
        let last = candidate.transcript.last_mut().unwrap();
        last.u_0 += correction.u_0;
        last.u_2 += correction.u_2;
        let got = *last;
        artifact_json(
            dir,
            &format!("fold-{depth}-messages"),
            &(want, got, uncorrected, correction),
        );
        assert_eq!(got, want, "fold {depth} message");
        if correction.u_0 != F128::ZERO || correction.u_2 != F128::ZERO {
            assert_ne!(uncorrected, want, "missing correction not detected");
            let doubled = SumcheckMessage {
                u_0: got.u_0 + correction.u_0,
                u_2: got.u_2 + correction.u_2,
            };
            assert_ne!(doubled, want, "duplicate correction not detected");
            negative_message_checks += 2;
        }
        assert_eq!(candidate.transcript, control.transcript);
        assert_eq!(candidate.t_r, control.t_r);
        assert_eq!(candidate.f, control.f);
        assert!(candidate.pending_ood_eq.is_none() && control.pending_ood_eq.is_none());
        assert!(candidate.pending_fold_basis.is_none() && control.pending_fold_basis.is_none());
        let delta = partial_eval_lsb(&scaled_basis, &rs[..depth]);
        let logical_basis: Vec<F128> = candidate
            .combined_basis
            .iter()
            .zip(&delta)
            .map(|(&a, &b)| a + b)
            .collect();
        same_fields(
            dir,
            &format!("fold-{depth}-logical-basis"),
            &logical_basis,
            &control.combined_basis,
        );
    }
    let reduced = dual.materialize(&rs[..2]);
    let want_reduced = partial_eval_lsb(&scaled_basis, &rs[..2]);
    same_fields(dir, "reduced-basis", &reduced, &want_reduced);
    let nonzero = reduced.iter().any(|&v| v != F128::ZERO);
    if nonzero {
        assert_ne!(
            candidate.combined_basis, control.combined_basis,
            "omitted merge not detected"
        );
    }
    assert!(
        candidate.pending_glue.is_none()
            && candidate.pending_ood_eq.is_none()
            && candidate.pending_fold_basis.is_none()
    );
    for (dst, &src) in candidate.combined_basis.iter_mut().zip(&reduced) {
        *dst += src;
        dual.work.merge_additions += 1;
    }
    same_fields(
        dir,
        "merged-basis",
        &candidate.combined_basis,
        &control.combined_basis,
    );
    if nonzero {
        assert!(
            candidate
                .combined_basis
                .iter()
                .zip(&reduced)
                .zip(&control.combined_basis)
                .any(|((&a, &b), &want)| a + b != want),
            "duplicate merge not detected"
        );
    }
    let m0 = control.fold(rs[2]);
    let m1 = candidate.fold(rs[2]);
    artifact_json(dir, "fold-3-messages", &(m0, m1));
    assert_eq!(m0, m1, "third fold after merge");
    same_fields(
        dir,
        "fold-3-basis",
        &candidate.combined_basis,
        &control.combined_basis,
    );
    same_fields(dir, "fold-3-f", &candidate.f, &control.f);
    // An ordinary introduction after this level must see complete state.
    let next_basis = ch.sample_f128_vec(control.f.len());
    let next_h = control
        .f
        .iter()
        .zip(&next_basis)
        .fold(F128::ZERO, |s, (&x, &y)| s + x * y);
    assert_eq!(
        candidate.introduce_new(next_basis.clone(), next_h),
        control.introduce_new(next_basis.clone(), next_h)
    );
    assert_eq!(candidate.pending_glue, control.pending_glue);
    let next_beta = ch.sample_f128();
    candidate.glue(next_beta);
    control.glue(next_beta);
    assert_eq!(candidate.fold(rs[3]), control.fold(rs[3]));
    same_fields(
        dir,
        "subsequent-basis",
        &candidate.combined_basis,
        &control.combined_basis,
    );
    same_fields(dir, "subsequent-f", &candidate.f, &control.f);
    assert_eq!(candidate.t_r, control.t_r);
    assert_eq!(candidate.transcript, control.transcript);
    assert!(
        candidate.pending_glue.is_none()
            && candidate.pending_fold_basis.is_none()
            && candidate.pending_ood_eq.is_none()
    );
    artifact_fields(dir, "input-subsequent-basis", &next_basis);
    artifact_fields(dir, "input-subsequent-beta", &[next_beta]);
    artifact_json(dir, "transcript", &candidate.transcript);
    assert_eq!(
        TEST_INDUCE_FUSED_MSG_HITS.with(|n| n.get()) - before_parent,
        parent_hits,
        "candidate invoked full induction"
    );
    assert_eq!(dual.work.reduced_transforms, 1);
    if ranked {
        assert_eq!(dual.work.constructor_products, 114_768);
        assert_eq!(dual.work.message_products, [3_706, 4_143, 5_889]);
        assert_eq!(dual.work.reduced_coeff_products, 1_747);
        assert_eq!(dual.work.merge_additions, 131_072);
    }
    artifact_json(
        dir,
        "result",
        &serde_json::json!({
            "passed":true,"ranked":ranked,"queries":fx.queries.len(),"parent_hits":parent_hits,"factored_hits":child_hits,
            "old_active_windows":fx.queries.iter().map(|q|q>>8).collect::<BTreeSet<_>>().len(),
            "old_active_densify_blocks":fx.queries.iter().map(|q|q>>11).collect::<BTreeSet<_>>().len(),
            "work":dual.work,"negative_message_checks":negative_message_checks,"negative_merge_checks":if nonzero {2} else {0},
            "owned_capacities_bytes":{"residues":dual.residues.capacity()*128,"queries":dual.queries.capacity()*8,"weights":dual.weights.capacity()*16,"reduced_basis":reduced.capacity()*16},
            "full_basis_fields":basis.len(),"reduced_basis_fields":reduced.len(),"stored_transcript_equal":true,"subsequent_state_equal":true
        }),
    );
    eprintln!(
        "C2 case_pass path={} ranked={} queries={} full_fields={} reduced_fields={}",
        dir.display(),
        ranked,
        fx.queries.len(),
        basis.len(),
        reduced.len()
    );
}

#[test]
#[ignore = "bounded EXP-0016 component oracle; run via its recorder"]
fn depth_two_component_oracle() {
    let root = std::path::PathBuf::from(
        std::env::var_os("FLOCK_C2_ARTIFACT_DIR").expect("run only via EXP-0016 recorder"),
    );
    for shape in 0..3 {
        for case in 0..12 {
            let dir = root.join(format!("compact-{shape}-{case:02}"));
            std::fs::create_dir(&dir).unwrap();
            let fx = compact_fixture(shape, case, &dir);
            check_case(fx, case, false, &dir);
        }
    }
    let dir = root.join("ranked");
    std::fs::create_dir(&dir).unwrap();
    let fx = ranked_fixture(&dir);
    check_case(fx, 12, true, &dir);
    eprintln!("C2 summary compact_cases=36 ranked_cases=1 proofs=0");
}

// Exercise the actual production state transitions against the complete basis,
// separately from the counted research prototype above.
fn check_production(fx: &Fixture, case: usize) {
    let opened: Vec<Vec<F128>> = fx.queries.iter().map(|q| fx.rows[q].clone()).collect();
    let (basis, enforced) = induce_sumcheck_poly_auto(
        fx.log_msg,
        fx.log_rate,
        &eval_sk_at_vks(fx.log_msg),
        &opened,
        &fx.lanes,
        &fx.queries,
        &fx.alpha,
    );
    let mut ch = crate::challenger::RandomChallenger::new(0xC200_2026 + case as u64);
    let initial = ch.sample_f128_vec(fx.f.len());
    let h =
        fx.f.iter()
            .zip(&initial)
            .fold(F128::ZERO, |s, (&a, &b)| s + a * b);
    let (mut control, _) = SumcheckProver::new(fx.f.clone(), initial.clone(), h);
    let (mut candidate, _) = SumcheckProver::new(fx.f.clone(), initial, h);
    let z = ch.sample_f128_vec(fx.log_msg);
    assert_eq!(
        control.introduce_new_ood_factorized(&z),
        candidate.introduce_new_ood_factorized(&z)
    );
    let beta_ood = ch.sample_f128();
    control.glue_factorized_ood(beta_ood);
    candidate.glue_factorized_ood(beta_ood);
    // The constructor cannot retain row borrows: this copy is dropped before
    // introduction, folding, or materialization.
    let rows = fx.rows.clone();
    let (dual, got_h) = c2::Dual::new(
        fx.log_msg + fx.log_rate,
        fx.log_rate,
        &fx.lanes,
        &fx.queries,
        &fx.alpha,
        |q| &rows[&q],
    );
    drop(rows);
    assert_eq!(got_h, enforced);
    let (mut pending, intro) = c2::Pending::introduce(dual, got_h, &mut candidate);
    assert_eq!(intro, control.introduce_new(basis, enforced));
    let beta = match case % 3 {
        0 => F128::ZERO,
        1 => F128::ONE,
        _ => ch.sample_f128(),
    };
    pending.glue(beta, &mut candidate);
    control.glue_deferred_into_lazy_ood_fold(beta);
    assert_eq!(candidate.t_r, control.t_r);
    assert_eq!(candidate.transcript, control.transcript);
    for depth in 1..=2 {
        let r = match (case + depth) % 3 {
            0 => F128::ZERO,
            1 => F128::ONE,
            _ => ch.sample_f128(),
        };
        let (msg, complete) = pending.fold(r, &mut candidate);
        assert_eq!(complete, depth == 2);
        assert_eq!(msg, control.fold(r), "production message at depth {depth}");
        assert_eq!(candidate.transcript, control.transcript);
        assert_eq!(candidate.f, control.f);
        assert_eq!(candidate.t_r, control.t_r);
        assert!(candidate.pending_ood_eq.is_none());
        assert!(candidate.pending_fold_basis.is_none());
    }
    drop(pending);
    assert_eq!(
        candidate.combined_basis, control.combined_basis,
        "once-only production merge"
    );
    let r = ch.sample_f128();
    assert_eq!(candidate.fold(r), control.fold(r));
    let basis = ch.sample_f128_vec(candidate.f.len());
    let h = candidate
        .f
        .iter()
        .zip(&basis)
        .fold(F128::ZERO, |s, (&a, &b)| s + a * b);
    assert_eq!(
        candidate.introduce_new(basis.clone(), h),
        control.introduce_new(basis, h)
    );
    let beta = ch.sample_f128();
    candidate.glue(beta);
    control.glue(beta);
    let r = ch.sample_f128();
    assert_eq!(candidate.fold(r), control.fold(r));
    assert_eq!(candidate.transcript, control.transcript);
    assert_eq!(candidate.combined_basis, control.combined_basis);
}

#[test]
fn production_gate_and_optout() {
    use std::ffi::OsStr;
    assert!(c2::enabled(true, None));
    assert!(c2::enabled(true, Some(OsStr::new("0"))));
    assert!(c2::enabled(true, Some(OsStr::new(""))));
    assert!(!c2::enabled(true, Some(OsStr::new("1"))));
    for value in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
        assert!(!c2::enabled(false, value));
    }
    let mut config = prover_config_for(25, 6, LigeritoProfile::Fast).unwrap();
    config.merkle_hash = HashKind::Blake3;
    let selected = |c: &ProverConfig, log_n, n1, count, len, direct, platform, disabled| {
        c2::enabled(
            ranked_l1_lazy_ood_eq_selected(c, log_n, n1, count, len, direct, platform, disabled),
            None,
        )
    };
    assert!(selected(&config, 25, 19, 1, 1 << 19, true, true, false));
    assert!(!selected(&config, 24, 19, 1, 1 << 19, true, true, false));
    assert!(!selected(&config, 25, 18, 1, 1 << 19, true, true, false));
    assert!(!selected(&config, 25, 19, 2, 1 << 19, true, true, false));
    assert!(!selected(&config, 25, 19, 1, 1 << 18, true, true, false));
    assert!(!selected(&config, 25, 19, 1, 1 << 19, false, true, false));
    assert!(!selected(&config, 25, 19, 1, 1 << 19, true, false, false));
    assert!(!selected(&config, 25, 19, 1, 1 << 19, true, true, true));
    config.queries[0] -= 1;
    assert!(!selected(&config, 25, 19, 1, 1 << 19, true, true, false));
}

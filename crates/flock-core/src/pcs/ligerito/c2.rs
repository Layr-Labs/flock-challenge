// Copyright (c) 2026 Bain Capital Crypto, LP and Ron Rothblum
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Delayed L0 induction: retain sparse query residues through two folds.

use super::*;

#[inline]
fn mul(a: F128, b: F128) -> F128 {
    a * b
}

#[inline]
fn dot(a: &[F128], b: &[F128]) -> F128 {
    assert_eq!(a.len(), b.len());
    match b.len() {
        64 => crate::field::f128_slice::fold_banked_slot::<64>(b.try_into().unwrap(), a),
        8 => crate::field::f128_slice::fold_banked_slot::<8>(b.try_into().unwrap(), a),
        4 => crate::field::f128_slice::fold_banked_slot::<4>(b.try_into().unwrap(), a),
        2 => crate::field::f128_slice::fold_banked_slot::<2>(b.try_into().unwrap(), a),
        1 => mul(a[0], b[0]),
        _ => {
            let mut acc = crate::field::F256Unreduced::ZERO;
            for (&x, &y) in a.iter().zip(b) {
                acc ^= x.mul_unreduced(y);
            }
            acc.reduce()
        }
    }
}

/// Builds the row one NTT layer at a time: `row[0]` is one, and layer `s`
/// extends entries `0..2^s` into `2^s..2^{s+1}` by multiplying with that
/// layer's twiddle `scale` (plus one when the query's bit `s` is set), so
/// after `k` layers the row holds the query's exact forward-NTT weights.
/// The `block` index folds the query's block prefix with its in-block
/// position, keeping the whole row O(k) twiddle reads, no dense matrix.
/// An exact row of a local forward NTT, generated without a dense matrix.
#[inline(always)]
fn local_row(ntt: &AdditiveNttF128, log_d: usize, k: usize, query: usize, row: &mut [F128; 16]) {
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
        let width = 1usize << s;
        let (lower, upper) = row.split_at_mut(width);
        for (dst, value) in upper[..width].iter_mut().zip(lower) {
            *dst = mul(*value, scale);
        }
    }
}

pub(super) struct Dual {
    ntt: AdditiveNttF128,
    log_d: usize,
    log_inv_rate: usize,
    depth: usize,
    cache_len: usize,
    queries: Vec<usize>,
    weights: Vec<F128>,
    residues: Vec<[F128; 16]>,
}

impl Dual {
    pub(super) fn new<'a>(
        log_d: usize,
        log_inv_rate: usize,
        lane_challenges: &[F128],
        queries: &[usize],
        alpha: &[F128],
        row: impl FnMut(usize) -> &'a [F128],
    ) -> (Self, F128) {
        Self::new_with_depth(2, log_d, log_inv_rate, lane_challenges, queries, alpha, row)
    }

    pub(super) fn new_with_depth<'a>(
        depth: usize,
        log_d: usize,
        log_inv_rate: usize,
        lane_challenges: &[F128],
        queries: &[usize],
        alpha: &[F128],
        mut row: impl FnMut(usize) -> &'a [F128],
    ) -> (Self, F128) {
        assert!((2..=3).contains(&depth));
        assert!(log_inv_rate + depth < log_d);
        assert!(queries.iter().all(|&q| q < 1usize << log_d));
        let cache_len = 1usize << (depth + 1);
        let mask = cache_len - 1;
        let inv_layers = depth + 1;
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
        let mut residues = Vec::with_capacity(queries.len());
        let mut enforced = F128::ZERO;
        for (&q, &alpha) in queries.iter().zip(&weights) {
            let base = q & !mask;
            let mut cache = [F128::ZERO; 16];
            for j in 0..cache_len {
                cache[j] = dot(row(base + j), &lane_weights);
            }
            enforced += mul(alpha, cache[q - base]);
            for layer in ((log_d - inv_layers)..log_d).rev() {
                let size = 1usize << (log_d - layer);
                let half = size / 2;
                for off in (0..cache_len).step_by(size) {
                    let twiddle = ntt.twiddle(layer, (base + off) / size);
                    for j in 0..half {
                        let y0 = cache[off + j];
                        let v = y0 + cache[off + half + j];
                        cache[off + j] = y0 + mul(v, twiddle);
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
                depth,
                cache_len,
                queries: queries.to_vec(),
                weights,
                residues,
            },
            enforced,
        )
    }

    pub(super) fn new_sync<'a>(
        depth: usize,
        log_d: usize,
        log_inv_rate: usize,
        lane_challenges: &[F128],
        queries: &[usize],
        alpha: &[F128],
        row: impl Fn(usize) -> &'a [F128] + Sync,
    ) -> (Self, F128) {
        assert!((2..=3).contains(&depth));
        assert!(log_inv_rate + depth < log_d);
        assert!(queries.iter().all(|&q| q < 1usize << log_d));
        let cache_len = 1usize << (depth + 1);
        let mask = cache_len - 1;
        let inv_layers = depth + 1;
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
        let build_one = |(&q, &w): (&usize, &F128)| -> ([F128; 16], F128) {
            let base = q & !mask;
            let mut cache = [F128::ZERO; 16];
            for j in 0..cache_len {
                cache[j] = dot(row(base + j), &lane_weights);
            }
            let enf = mul(w, cache[q - base]);
            for layer in ((log_d - inv_layers)..log_d).rev() {
                let size = 1usize << (log_d - layer);
                let half = size / 2;
                for off in (0..cache_len).step_by(size) {
                    let twiddle = ntt.twiddle(layer, (base + off) / size);
                    for j in 0..half {
                        let y0 = cache[off + j];
                        let v = y0 + cache[off + half + j];
                        cache[off + j] = y0 + mul(v, twiddle);
                        cache[off + half + j] = v;
                    }
                }
            }
            (cache, enf)
        };
        let mut residues = Vec::with_capacity(queries.len());
        let mut enforced = F128::ZERO;
        if queries.len() * lane_weights.len() >= 1024 {
            use rayon::prelude::*;
            let pairs: Vec<([F128; 16], F128)> =
                queries.par_iter().zip(&weights).map(build_one).collect();
            for (cache, enf) in pairs {
                residues.push(cache);
                enforced += enf;
            }
        } else {
            for pair in queries.iter().zip(&weights) {
                let (cache, enf) = build_one(pair);
                residues.push(cache);
                enforced += enf;
            }
        }
        (
            Self {
                ntt,
                log_d,
                log_inv_rate,
                depth,
                cache_len,
                queries: queries.to_vec(),
                weights,
                residues,
            },
            enforced,
        )
    }

    pub(super) fn message(&mut self, rs: &[F128]) -> SumcheckMessage {
        assert!(rs.len() <= self.depth);
        let k = rs.len() + 1;
        let size = 1usize << k;
        let half = size / 2;
        let cache_len = self.cache_len;
        let mask = cache_len - 1;
        let inv_layers = self.depth + 1;
        let weights = build_eq_table(rs);
        let mut msg = SumcheckMessage {
            u_0: F128::ZERO,
            u_2: F128::ZERO,
        };
        for i in 0..self.queries.len() {
            let q = self.queries[i];
            let cache_base = q & !mask;
            let mut cache = self.residues[i];
            for layer in (self.log_d - inv_layers)..(self.log_d - k) {
                let local_size = 1usize << (self.log_d - layer);
                let local_half = local_size / 2;
                for off in (0..cache_len).step_by(local_size) {
                    let twiddle = self.ntt.twiddle(layer, (cache_base + off) / local_size);
                    for j in 0..local_half {
                        let v = cache[off + local_half + j];
                        let u = cache[off + j] + mul(v, twiddle);
                        cache[off + j] = u;
                        cache[off + local_half + j] = v + u;
                    }
                }
            }
            let off = (q & !(size - 1)) - cache_base;
            let f0 = dot(&cache[off..off + half], &weights);
            let f1 = dot(&cache[off + half..off + size], &weights);
            let mut row = [F128::ZERO; 16];
            local_row(&self.ntt, self.log_d, k, q, &mut row);
            let dot0 = dot(&row[..half], &weights);
            let dot2 = mul(F128::ONE + row[half], dot0);
            let a0 = mul(self.weights[i], f0);
            let a2 = mul(self.weights[i], f0 + f1);
            msg.u_0 += mul(a0, dot0);
            msg.u_2 += mul(a2, dot2);
        }
        msg
    }

    pub(super) fn scale(&mut self, beta: F128) {
        for w in &mut self.weights {
            *w = mul(*w, beta);
        }
    }

    pub(super) fn materialize(&mut self, rs: &[F128]) -> Vec<F128> {
        let d = self.depth;
        assert_eq!(rs.len(), d);
        let half = 1usize << d;
        let weights = build_eq_table(rs);
        let mut positions = Vec::with_capacity(self.queries.len());
        let mut values = Vec::with_capacity(self.queries.len());
        for (&q, &alpha) in self.queries.iter().zip(&self.weights) {
            let mut row = [F128::ZERO; 16];
            local_row(&self.ntt, self.log_d, d, q, &mut row);
            let v = dot(&row[..half], &weights);
            positions.push(q >> d);
            values.push(mul(alpha, v));
        }
        let reduced_log_d = self.log_d - d;
        // Keep the original NTT's twiddles. A fresh NTT at the smaller
        // dimension would describe a different linear map.
        let (mut basis, _) = transpose_forward_ntt_sparse_impl(
            &self.ntt,
            &positions,
            &values,
            reduced_log_d,
            false,
            None,
            false,
        );
        basis.truncate(1usize << (reduced_log_d - self.log_inv_rate));
        basis
    }
}

/// Owns the sparse term separately from the one-fold lazy OOD slot.
/// The caller drops this state immediately after `fold` completes `dual.depth`.
pub(super) struct Pending {
    dual: Dual,
    enforced: F128,
    challenges: Vec<F128>,
    glued: bool,
}

impl Pending {
    pub(super) fn introduce(
        dual: Dual,
        enforced: F128,
        sc: &mut SumcheckProver,
    ) -> (Self, SumcheckMessage) {
        assert!(sc.pending_glue.is_none() && sc.pending_fold_basis.is_none());
        assert!(!matches!(
            sc.pending_ood_eq,
            Some(PendingOodEq::Introduced { .. })
        ));
        let cap = dual.depth;
        let mut state = Self {
            dual,
            enforced,
            challenges: Vec::with_capacity(cap),
            glued: false,
        };
        let message = state.dual.message(&[]);
        sc.transcript.push(message);
        (state, message)
    }

    pub(super) fn glue(&mut self, beta: F128, sc: &mut SumcheckProver) {
        assert!(
            !self.glued && self.challenges.is_empty(),
            "C2 glue applied twice"
        );
        sc.t_r += beta * self.enforced;
        self.dual.scale(beta);
        self.glued = true;
    }

    /// Correct both copies of the message before the next challenge, and
    /// materialize/merge# once before any subsequent introduction.
    pub(super) fn fold(&mut self, r: F128, sc: &mut SumcheckProver) -> (SumcheckMessage, bool) {
        let d = self.dual.depth;
        assert!(self.glued && self.challenges.len() < d);
        let mut message = sc.fold(r);
        self.challenges.push(r);
        let correction = self.dual.message(&self.challenges);
        message.u_0 += correction.u_0;
        message.u_2 += correction.u_2;
        *sc.transcript.last_mut().expect("fold records a message") = message;
        let complete = self.challenges.len() == d;
        if complete {
            assert!(
                sc.pending_glue.is_none()
                    && sc.pending_fold_basis.is_none()
                    && sc.pending_ood_eq.is_none()
            );
            let basis = self.dual.materialize(&self.challenges);
            assert_eq!(basis.len(), sc.combined_basis.len());
            if sc.combined_basis.len() >= 16_384 {
                use rayon::prelude::*;
                sc.combined_basis
                    .par_chunks_mut(4096)
                    .zip(basis.par_chunks(4096))
                    .for_each(|(dst_chunk, src_chunk)| {
                        for (dst, &src) in dst_chunk.iter_mut().zip(src_chunk) {
                            *dst += src;
                        }
                    });
            } else {
                for (dst, &src) in sc.combined_basis.iter_mut().zip(&basis) {
                    *dst += src;
                }
            }
            crate::scratch::give_sparse_ntt_data(basis);
        }
        (message, complete)
    }
}

pub(super) fn enabled(lazy_l1_selected: bool, disabled: Option<&std::ffi::OsStr>) -> bool {
    lazy_l1_selected && disabled != Some(std::ffi::OsStr::new("1"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::{Challenger, RandomChallenger};

    fn parent_message(dual: &Dual, rs: &[F128]) -> SumcheckMessage {
        assert!(rs.len() <= dual.depth);
        let k = rs.len() + 1;
        let size = 1usize << k;
        let half = size / 2;
        let cache_len = dual.cache_len;
        let mask = cache_len - 1;
        let inv_layers = dual.depth + 1;
        let weights = build_eq_table(rs);
        let mut msg = SumcheckMessage {
            u_0: F128::ZERO,
            u_2: F128::ZERO,
        };
        for i in 0..dual.queries.len() {
            let q = dual.queries[i];
            let cache_base = q & !mask;
            let mut cache = dual.residues[i];
            for layer in (dual.log_d - inv_layers)..(dual.log_d - k) {
                let local_size = 1usize << (dual.log_d - layer);
                let local_half = local_size / 2;
                for off in (0..cache_len).step_by(local_size) {
                    let twiddle = dual.ntt.twiddle(layer, (cache_base + off) / local_size);
                    for j in 0..local_half {
                        let v = cache[off + local_half + j];
                        let u = cache[off + j] + mul(v, twiddle);
                        cache[off + j] = u;
                        cache[off + local_half + j] = v + u;
                    }
                }
            }
            let off = (q & !(size - 1)) - cache_base;
            let f0 = dot(&cache[off..off + half], &weights);
            let f1 = dot(&cache[off + half..off + size], &weights);
            let row = {
                let ntt = &dual.ntt;
                let log_d = dual.log_d;
                let query = q;
                let mut row = [F128::ZERO; 16];
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
                        row[(1 << s) + i] = mul(row[i], scale);
                    }
                }
                row
            };
            let dot0 = dot(&row[..half], &weights);
            let dot2 = mul(F128::ONE + row[half], dot0);
            let a0 = mul(dual.weights[i], f0);
            let a2 = mul(dual.weights[i], f0 + f1);
            msg.u_0 += mul(a0, dot0);
            msg.u_2 += mul(a2, dot2);
        }
        msg
    }

    const LOG_D: usize = 5;
    const LOG_RATE: usize = 1;
    const LOG_LANES: usize = 6;

    fn dual_fixture(
        ntt: AdditiveNttF128,
        depth: usize,
        codeword: &[F128],
        lanes: &[F128],
        alpha: &[F128],
    ) -> Dual {
        let cache_len = 1usize << (depth + 1);
        let lane_weights = build_eq_table(lanes);
        let queries: Vec<usize> = (0..1usize << LOG_D).collect();
        let mut residues = Vec::with_capacity(queries.len());
        for &q in &queries {
            let base = q & !(cache_len - 1);
            let mut cache = [F128::ZERO; 16];
            for j in 0..cache_len {
                let start = (base + j) << LOG_LANES;
                cache[j] = dot(&codeword[start..start + (1 << LOG_LANES)], &lane_weights);
            }
            for layer in ((LOG_D - depth - 1)..LOG_D).rev() {
                let size = 1usize << (LOG_D - layer);
                let half = size / 2;
                for off in (0..cache_len).step_by(size) {
                    let twiddle = ntt.twiddle(layer, (base + off) / size);
                    for j in 0..half {
                        let y0 = cache[off + j];
                        let v = y0 + cache[off + half + j];
                        cache[off + j] = y0 + mul(v, twiddle);
                        cache[off + half + j] = v;
                    }
                }
            }
            residues.push(cache);
        }
        Dual {
            ntt,
            log_d: LOG_D,
            log_inv_rate: LOG_RATE,
            depth,
            cache_len,
            queries,
            weights: build_eq_table(alpha),
            residues,
        }
    }

    #[test]
    fn dot2_factored_depths_match_current_parent() {
        let mut rng = RandomChallenger::new(0xC2_D072_9992_0340);
        for custom in [false, true] {
            let scale = F128::new(0x9e37_79b9, 0x0123_4567_89ab_cdef);
            let basis: Vec<F128> = (0..LOG_D)
                .map(|i| mul(F128::new(1u64 << i, 0), scale))
                .collect();
            let ntt = if custom {
                AdditiveNttF128::new(&basis)
            } else {
                AdditiveNttF128::standard(LOG_D)
            };
            for depth in [2, 3] {
                for k in 1..=depth + 1 {
                    let half = 1usize << (k - 1);
                    for q in 0..1usize << LOG_D {
                        let mut row = [F128::ZERO; 16];
                        local_row(&ntt, LOG_D, k, q, &mut row);
                        assert_eq!(row[0], F128::ONE);
                        for j in 0..half {
                            assert_eq!(row[j + half], mul(row[half], row[j]));
                        }
                    }
                }
                for case in 0..3 {
                    let lanes = match case {
                        0 => vec![F128::ZERO; LOG_LANES],
                        1 => vec![F128::ONE; LOG_LANES],
                        _ => rng.sample_f128_vec(LOG_LANES),
                    };
                    let alpha = match case {
                        0 => vec![F128::ZERO; LOG_D],
                        1 => vec![F128::ONE; LOG_D],
                        _ => rng.sample_f128_vec(LOG_D),
                    };
                    let mut codeword = vec![F128::ZERO; 1 << (LOG_D + LOG_LANES)];
                    for value in &mut codeword[..1 << (LOG_D - LOG_RATE + LOG_LANES)] {
                        *value = match case {
                            0 => F128::ZERO,
                            1 => F128::ONE,
                            _ => rng.sample_f128(),
                        };
                    }
                    ntt.forward_transform_interleaved(&mut codeword, 1 << LOG_LANES);
                    let mut dual = dual_fixture(ntt.clone(), depth, &codeword, &lanes, &alpha);
                    let expected_intro = parent_message(&dual, &[]);
                    assert_eq!(dual.message(&[]), expected_intro);
                    let initial_weights = dual.weights.clone();
                    let initial_residues = dual.residues.clone();
                    let mut random = rng.sample_f128();
                    random.hi |= 1;
                    let challenges = [F128::ZERO, F128::ONE, random];
                    for &beta in &challenges {
                        dual.weights.copy_from_slice(&initial_weights);
                        dual.scale(beta);
                        for mut choice in 0..3usize.pow(depth as u32) {
                            let rs: Vec<F128> = (0..depth)
                                .map(|_| {
                                    let r = challenges[choice % 3];
                                    choice /= 3;
                                    r
                                })
                                .collect();
                            for count in 1..=depth {
                                let point = &rs[..count];
                                let expected = parent_message(&dual, point);
                                assert_eq!(
                                    dual.message(point),
                                    expected,
                                    "custom={custom}, depth={depth}, case={case}, point={point:?}, beta={beta:?}"
                                );
                            }
                        }
                        assert_eq!(dual.residues, initial_residues);
                        if depth == 2 {
                            assert!(
                                dual.residues
                                    .iter()
                                    .all(|r| r[8..].iter().all(|x| x.is_zero()))
                            );
                        }
                    }
                }
            }
        }
    }
}

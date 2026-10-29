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

#[inline]
fn one_plus_scale(ntt: &AdditiveNttF128, log_d: usize, s: usize, query: usize) -> F128 {
    let mut val = ntt.twiddle(log_d - 1 - s, query >> (s + 1));
    if (query >> s) & 1 == 0 {
        val += F128::ONE;
    }
    val
}

pub(super) struct Dual {
    ntt: AdditiveNttF128,
    log_d: usize,
    log_inv_rate: usize,
    depth: usize,
    cache_len: usize,
    current_k: usize,
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
        let layer0 = log_d - 1;
        for (&q, &alpha) in queries.iter().zip(&weights) {
            let base = q & !mask;
            let mut cache = [F128::ZERO; 16];
            for j in 0..cache_len {
                cache[j] = dot(row(base + j), &lane_weights);
            }
            enforced += mul(alpha, cache[q - base]);
            for off in (0..cache_len).step_by(2) {
                let twiddle = ntt.twiddle(layer0, (base + off) / 2);
                let y0 = cache[off];
                let v = y0 + cache[off + 1];
                cache[off] = y0 + mul(v, twiddle);
                cache[off + 1] = v;
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
                current_k: 1,
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
        let layer0 = log_d - 1;
        let build_one = |(&q, &w): (&usize, &F128)| -> ([F128; 16], F128) {
            let base = q & !mask;
            let mut cache = [F128::ZERO; 16];
            for j in 0..cache_len {
                cache[j] = dot(row(base + j), &lane_weights);
            }
            let enf = mul(w, cache[q - base]);
            for off in (0..cache_len).step_by(2) {
                let twiddle = ntt.twiddle(layer0, (base + off) / 2);
                let y0 = cache[off];
                let v = y0 + cache[off + 1];
                cache[off] = y0 + mul(v, twiddle);
                cache[off + 1] = v;
            }
            (cache, enf)
        };
        let mut residues = vec![[F128::ZERO; 16]; queries.len()];
        let mut enforced = F128::ZERO;
        if queries.len() * lane_weights.len() >= 1024 {
            use rayon::prelude::*;
            enforced = residues
                .par_iter_mut()
                .zip(queries.par_iter().zip(&weights))
                .map(|(slot, pair)| {
                    let (cache, enf) = build_one(pair);
                    *slot = cache;
                    enf
                })
                .reduce(|| F128::ZERO, |a, b| a + b);
        } else {
            for (slot, pair) in residues.iter_mut().zip(queries.iter().zip(&weights)) {
                let (cache, enf) = build_one(pair);
                *slot = cache;
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
                current_k: 1,
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
        let weights = build_eq_table(rs);
        let mut msg = SumcheckMessage {
            u_0: F128::ZERO,
            u_2: F128::ZERO,
        };
        let prev_k = self.current_k;
        for i in 0..self.queries.len() {
            let q = self.queries[i];
            let cache_base = q & !mask;
            let cache = &mut self.residues[i];
            if k > prev_k {
                for step_k in (prev_k + 1)..=k {
                    let layer = self.log_d - step_k;
                    let step_size = 1usize << step_k;
                    let step_half = step_size / 2;
                    for off in (0..cache_len).step_by(step_size) {
                        let twiddle = self.ntt.twiddle(layer, (cache_base + off) / step_size);
                        for j in 0..step_half {
                            let y0 = cache[off + j];
                            let v = y0 + cache[off + step_half + j];
                            cache[off + j] = y0 + mul(v, twiddle);
                            cache[off + step_half + j] = v;
                        }
                    }
                }
            } else if k < prev_k {
                for step_k in ((k + 1)..=prev_k).rev() {
                    let layer = self.log_d - step_k;
                    let step_size = 1usize << step_k;
                    let step_half = step_size / 2;
                    for off in (0..cache_len).step_by(step_size) {
                        let twiddle = self.ntt.twiddle(layer, (cache_base + off) / step_size);
                        for j in 0..step_half {
                            let v = cache[off + step_half + j];
                            let u = cache[off + j] + mul(v, twiddle);
                            cache[off + j] = u;
                            cache[off + step_half + j] = v + u;
                        }
                    }
                }
            }
            let off = (q & !(size - 1)) - cache_base;
            let f0 = dot(&cache[off..off + half], &weights);
            let f1 = dot(&cache[off + half..off + size], &weights);
            let mut dot0 = F128::ONE;
            for (s, &r_s) in rs.iter().enumerate() {
                dot0 = mul(dot0, F128::ONE + mul(r_s, one_plus_scale(&self.ntt, self.log_d, s, q)));
            }
            let w0 = mul(self.weights[i], dot0);
            let w2 = mul(w0, one_plus_scale(&self.ntt, self.log_d, k - 1, q));
            msg.u_0 += mul(w0, f0);
            msg.u_2 += mul(w2, f0 + f1);
        }
        self.current_k = k;
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
        let mut positions = Vec::with_capacity(self.queries.len());
        let mut values = Vec::with_capacity(self.queries.len());
        for (&q, &alpha) in self.queries.iter().zip(&self.weights) {
            let mut v = alpha;
            for (s, &r_s) in rs.iter().enumerate() {
                v = mul(v, F128::ONE + mul(r_s, one_plus_scale(&self.ntt, self.log_d, s, q)));
            }
            positions.push(q >> d);
            values.push(v);
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

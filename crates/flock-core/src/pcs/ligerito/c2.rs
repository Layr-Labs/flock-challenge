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

fn dot(a: &[F128], b: &[F128]) -> F128 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(F128::ZERO, |s, (&x, &y)| s + mul(x, y))
}

/// Builds the row one NTT layer at a time: `row[0]` is one, and layer `s`
/// extends entries `0..2^s` into `2^s..2^{s+1}` by multiplying with that
/// layer's twiddle `scale` (plus one when the query's bit `s` is set), so
/// after `k` layers the row holds the query's exact forward-NTT weights.
/// The `block` index folds the query's block prefix with its in-block
/// position, keeping the whole row O(k) twiddle reads, no dense matrix.
/// An exact row of a local forward NTT, generated without a dense matrix.
fn local_row(ntt: &AdditiveNttF128, log_d: usize, k: usize, query: usize) -> [F128; 8] {
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
            row[(1 << s) + i] = mul(row[i], scale);
        }
    }
    row
}

pub(super) struct Dual {
    ntt: AdditiveNttF128,
    log_d: usize,
    log_inv_rate: usize,
    queries: Vec<usize>,
    weights: Vec<F128>,
    residues: Vec<[F128; 8]>,
}

impl Dual {
    pub(super) fn new<'a>(
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
        let mut residues = Vec::with_capacity(queries.len());
        let mut enforced = F128::ZERO;
        for (&q, &alpha) in queries.iter().zip(&weights) {
            let base = q & !7usize;
            let mut cache = [F128::ZERO; 8];
            for j in 0..8 {
                cache[j] = dot(row(base + j), &lane_weights);
            }
            enforced += mul(alpha, cache[q - base]);
            // Invert the final three forward layers of the valid codeword.
            for layer in ((log_d - 3)..log_d).rev() {
                let size = 1usize << (log_d - layer);
                let half = size / 2;
                for off in (0..8).step_by(size) {
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
                queries: queries.to_vec(),
                weights,
                residues,
            },
            enforced,
        )
    }

    pub(super) fn message(&mut self, rs: &[F128]) -> SumcheckMessage {
        assert!(rs.len() <= 2);
        let k = rs.len() + 1;
        let size = 1usize << k;
        let half = size / 2;
        let weights = build_eq_table(rs);
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
                        let u = cache[off + j] + mul(v, twiddle);
                        cache[off + j] = u;
                        cache[off + local_half + j] = v + u;
                    }
                }
            }
            let off = (q & !(size - 1)) - cache_base;
            let f0 = dot(&cache[off..off + half], &weights);
            let f1 = dot(&cache[off + half..off + size], &weights);
            let row = local_row(&self.ntt, self.log_d, k, q);
            let dot0 = dot(&row[..half], &weights);
            let mut dot2 = F128::ZERO;
            for j in 0..half {
                dot2 += mul(row[j] + row[half + j], weights[j]);
            }
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
        assert_eq!(rs.len(), 2);
        let weights = build_eq_table(rs);
        let mut positions = Vec::with_capacity(self.queries.len());
        let mut values = Vec::with_capacity(self.queries.len());
        for (&q, &alpha) in self.queries.iter().zip(&self.weights) {
            let row = local_row(&self.ntt, self.log_d, 2, q);
            let v = dot(&row[..4], &weights);
            positions.push(q >> 2);
            values.push(mul(alpha, v));
        }
        let reduced_log_d = self.log_d - 2;
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
/// The caller drops this state immediately after `fold` completes depth two.
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
        assert!(matches!(
            sc.pending_ood_eq,
            Some(PendingOodEq::Glued { .. })
        ));
        let mut state = Self {
            dual,
            enforced,
            challenges: Vec::with_capacity(2),
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
    /// materialize/merge exactly once before any subsequent introduction.
    pub(super) fn fold(&mut self, r: F128, sc: &mut SumcheckProver) -> (SumcheckMessage, bool) {
        assert!(self.glued && self.challenges.len() < 2);
        let mut message = sc.fold(r);
        self.challenges.push(r);
        let correction = self.dual.message(&self.challenges);
        message.u_0 += correction.u_0;
        message.u_2 += correction.u_2;
        *sc.transcript.last_mut().expect("fold records a message") = message;
        let complete = self.challenges.len() == 2;
        if complete {
            assert!(
                sc.pending_glue.is_none()
                    && sc.pending_fold_basis.is_none()
                    && sc.pending_ood_eq.is_none()
            );
            let basis = self.dual.materialize(&self.challenges);
            assert_eq!(basis.len(), sc.combined_basis.len());
            for (dst, src) in sc.combined_basis.iter_mut().zip(basis) {
                *dst += src;
            }
        }
        (message, complete)
    }
}

pub(super) fn enabled(lazy_l1_selected: bool, disabled: Option<&std::ffi::OsStr>) -> bool {
    lazy_l1_selected && disabled != Some(std::ffi::OsStr::new("1"))
}

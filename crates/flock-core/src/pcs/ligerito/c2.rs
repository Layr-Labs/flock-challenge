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
fn dot_pair(local: &[F128], weights: &[F128]) -> (F128, F128) {
    let half = weights.len();
    debug_assert_eq!(local.len(), 2 * half);
    match half {
        8 => {
            let w: &[F128; 8] = weights.try_into().unwrap();
            let [f0, f1] = crate::field::f128_slice::fold_banked_slots2::<8>(w, local);
            (f0, f1)
        }
        4 => {
            let w: &[F128; 4] = weights.try_into().unwrap();
            let [f0, f1] = crate::field::f128_slice::fold_banked_slots2::<4>(w, local);
            (f0, f1)
        }
        2 => {
            let w: &[F128; 2] = weights.try_into().unwrap();
            let [f0, f1] = crate::field::f128_slice::fold_banked_slots2::<2>(w, local);
            (f0, f1)
        }
        1 => (mul(local[0], weights[0]), mul(local[1], weights[0])),
        _ => (dot(&local[..half], weights), dot(&local[half..], weights)),
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

#[allow(clippy::too_many_arguments)]
#[inline]
fn build_query<'a>(
    ntt: &AdditiveNttF128,
    log_d: usize,
    depth: usize,
    inv_layers: usize,
    cache_len: usize,
    mask: usize,
    lane_weights: &[F128],
    q: usize,
    w: F128,
    mut row_fn: impl FnMut(usize) -> &'a [F128],
    out_rows: &mut [[F128; 16]],
) -> (
    crate::field::F256Unreduced,
    crate::field::F256Unreduced,
    crate::field::F256Unreduced,
) {
    let base = q & !mask;
    let qoff = q - base;
    let mut cache = [F128::ZERO; 16];
    for j in 0..cache_len {
        cache[j] = dot(row_fn(base + j), lane_weights);
    }
    let enf = w.mul_unreduced(cache[qoff]);
    let mut qscales = [F128::ZERO; 4];
    for (s, slot) in qscales.iter_mut().enumerate().take(depth + 1) {
        *slot = one_plus_scale(ntt, log_d, s, q);
    }
    let mut u0 = crate::field::F256Unreduced::ZERO;
    let mut u2 = crate::field::F256Unreduced::ZERO;
    for (stage, layer) in ((log_d - inv_layers)..log_d).rev().enumerate() {
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
        let len = 2usize << stage;
        let start = qoff & !(len - 1);
        match stage {
            0 => {
                let f0 = cache[start];
                let f1 = cache[start + 1];
                let a2 = mul(w, f0 + f1);
                u0 = w.mul_unreduced(f0);
                u2 = a2.mul_unreduced(qscales[0]);
            }
            1 => {
                out_rows[0][0..4].copy_from_slice(&cache[start..start + 4]);
            }
            2 => {
                out_rows[0][4..12].copy_from_slice(&cache[start..start + 8]);
                out_rows[0][12..16].copy_from_slice(&qscales);
            }
            3 => {
                out_rows[1].copy_from_slice(&cache[start..start + 16]);
            }
            _ => unreachable!(),
        }
    }
    (enf, u0, u2)
}

pub(super) struct Dual {
    ntt: AdditiveNttF128,
    log_d: usize,
    log_inv_rate: usize,
    depth: usize,
    queries: Vec<usize>,
    weights: Vec<F128>,
    rows: Vec<[F128; 16]>,
    msg0: SumcheckMessage,
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
        let stride = depth - 1;
        let mut rows = crate::alloc_uninit_vec::<[F128; 16]>(queries.len() * stride);
        let mut enf_acc = crate::field::F256Unreduced::ZERO;
        let mut u0_acc = crate::field::F256Unreduced::ZERO;
        let mut u2_acc = crate::field::F256Unreduced::ZERO;
        for ((&q, &w), out_rows) in queries
            .iter()
            .zip(&weights)
            .zip(rows.chunks_exact_mut(stride))
        {
            let (enf, u0, u2) = build_query(
                &ntt,
                log_d,
                depth,
                inv_layers,
                cache_len,
                mask,
                &lane_weights,
                q,
                w,
                &mut row,
                out_rows,
            );
            enf_acc ^= enf;
            u0_acc ^= u0;
            u2_acc ^= u2;
        }
        (
            Self {
                ntt,
                log_d,
                log_inv_rate,
                depth,
                queries: queries.to_vec(),
                weights,
                rows,
                msg0: SumcheckMessage {
                    u_0: u0_acc.reduce(),
                    u_2: u2_acc.reduce(),
                },
            },
            enf_acc.reduce(),
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
        let n_queries = queries.len();
        let stride = depth - 1;
        let mut rows = crate::alloc_uninit_vec::<[F128; 16]>(n_queries * stride);
        let (enf_acc, u0_acc, u2_acc) = if n_queries * lane_weights.len() >= 1024 {
            use rayon::prelude::*;
            queries
                .par_iter()
                .zip(&weights)
                .zip(rows.par_chunks_exact_mut(stride))
                .map(|((&q, &w), out_rows)| {
                    build_query(
                        &ntt,
                        log_d,
                        depth,
                        inv_layers,
                        cache_len,
                        mask,
                        &lane_weights,
                        q,
                        w,
                        &row,
                        out_rows,
                    )
                })
                .reduce(
                    || {
                        (
                            crate::field::F256Unreduced::ZERO,
                            crate::field::F256Unreduced::ZERO,
                            crate::field::F256Unreduced::ZERO,
                        )
                    },
                    |a, b| (a.0 ^ b.0, a.1 ^ b.1, a.2 ^ b.2),
                )
        } else {
            let mut enf_sum = crate::field::F256Unreduced::ZERO;
            let mut u0_sum = crate::field::F256Unreduced::ZERO;
            let mut u2_sum = crate::field::F256Unreduced::ZERO;
            for ((&q, &w), out_rows) in queries
                .iter()
                .zip(&weights)
                .zip(rows.chunks_exact_mut(stride))
            {
                let (enf, u0, u2) = build_query(
                    &ntt,
                    log_d,
                    depth,
                    inv_layers,
                    cache_len,
                    mask,
                    &lane_weights,
                    q,
                    w,
                    &row,
                    out_rows,
                );
                enf_sum ^= enf;
                u0_sum ^= u0;
                u2_sum ^= u2;
            }
            (enf_sum, u0_sum, u2_sum)
        };
        (
            Self {
                ntt,
                log_d,
                log_inv_rate,
                depth,
                queries: queries.to_vec(),
                weights,
                rows,
                msg0: SumcheckMessage {
                    u_0: u0_acc.reduce(),
                    u_2: u2_acc.reduce(),
                },
            },
            enf_acc.reduce(),
        )
    }

    pub(super) fn message(&mut self, rs: &[F128]) -> SumcheckMessage {
        assert!(rs.len() <= self.depth);
        if rs.is_empty() {
            return self.msg0;
        }
        let k = rs.len() + 1;
        let half = 1usize << rs.len();
        let mut weights_buf = [F128::ZERO; 8];
        let weights = &mut weights_buf[..half];
        crate::lincheck::build_eq_table_into(rs, weights);
        let stride = self.depth - 1;
        let mut u0_acc = crate::field::F256Unreduced::ZERO;
        let mut u2_acc = crate::field::F256Unreduced::ZERO;
        for i in 0..self.queries.len() {
            let hdr = &self.rows[i * stride];
            let local: &[F128] = match k {
                2 => &hdr[0..4],
                3 => &hdr[4..12],
                4 => &self.rows[i * stride + 1],
                _ => unreachable!(),
            };
            let (f0, f1) = dot_pair(local, weights);
            let qscales = &hdr[12..16];
            let mut w0 = mul(self.weights[i], F128::ONE + mul(rs[0], qscales[0]));
            for (s, &r_s) in rs.iter().enumerate().skip(1) {
                w0 = mul(w0, F128::ONE + mul(r_s, qscales[s]));
            }
            let w2 = mul(w0, qscales[k - 1]);
            u0_acc ^= w0.mul_unreduced(f0);
            u2_acc ^= w2.mul_unreduced(f0 + f1);
        }
        SumcheckMessage {
            u_0: u0_acc.reduce(),
            u_2: u2_acc.reduce(),
        }
    }

    pub(super) fn message_and_materialize(
        &mut self,
        rs: &[F128],
    ) -> (SumcheckMessage, Vec<F128>) {
        let d = self.depth;
        assert_eq!(rs.len(), d);
        let k = d + 1;
        let half = 1usize << d;
        let mut weights_buf = [F128::ZERO; 8];
        let weights = &mut weights_buf[..half];
        crate::lincheck::build_eq_table_into(rs, weights);
        let stride = d - 1;
        let n = self.queries.len();
        let reduced_log_d = self.log_d - d;
        let mut u0_acc = crate::field::F256Unreduced::ZERO;
        let mut u2_acc = crate::field::F256Unreduced::ZERO;
        let mut basis = if n <= 256 {
            let mut pos_stack = [0usize; 256];
            let mut val_stack = [F128::ZERO; 256];
            for i in 0..n {
                let hdr = &self.rows[i * stride];
                let local: &[F128] = match k {
                    2 => &hdr[0..4],
                    3 => &hdr[4..12],
                    4 => &self.rows[i * stride + 1],
                    _ => unreachable!(),
                };
                let (f0, f1) = dot_pair(local, weights);
                let qscales = &hdr[12..16];
                let mut w0 = mul(self.weights[i], F128::ONE + mul(rs[0], qscales[0]));
                for (s, &r_s) in rs.iter().enumerate().skip(1) {
                    w0 = mul(w0, F128::ONE + mul(r_s, qscales[s]));
                }
                pos_stack[i] = self.queries[i] >> d;
                val_stack[i] = w0;
                let w2 = mul(w0, qscales[k - 1]);
                u0_acc ^= w0.mul_unreduced(f0);
                u2_acc ^= w2.mul_unreduced(f0 + f1);
            }
            transpose_forward_ntt_sparse_impl(
                &self.ntt,
                &pos_stack[..n],
                &val_stack[..n],
                reduced_log_d,
                false,
                None,
                false,
            )
            .0
        } else {
            let mut positions = Vec::with_capacity(n);
            let mut values = Vec::with_capacity(n);
            for i in 0..n {
                let hdr = &self.rows[i * stride];
                let local: &[F128] = match k {
                    2 => &hdr[0..4],
                    3 => &hdr[4..12],
                    4 => &self.rows[i * stride + 1],
                    _ => unreachable!(),
                };
                let (f0, f1) = dot_pair(local, weights);
                let qscales = &hdr[12..16];
                let mut w0 = mul(self.weights[i], F128::ONE + mul(rs[0], qscales[0]));
                for (s, &r_s) in rs.iter().enumerate().skip(1) {
                    w0 = mul(w0, F128::ONE + mul(r_s, qscales[s]));
                }
                positions.push(self.queries[i] >> d);
                values.push(w0);
                let w2 = mul(w0, qscales[k - 1]);
                u0_acc ^= w0.mul_unreduced(f0);
                u2_acc ^= w2.mul_unreduced(f0 + f1);
            }
            transpose_forward_ntt_sparse_impl(
                &self.ntt,
                &positions,
                &values,
                reduced_log_d,
                false,
                None,
                false,
            )
            .0
        };
        basis.truncate(1usize << (reduced_log_d - self.log_inv_rate));
        (
            SumcheckMessage {
                u_0: u0_acc.reduce(),
                u_2: u2_acc.reduce(),
            },
            basis,
        )
    }

    pub(super) fn scale(&mut self, beta: F128) {
        self.msg0.u_0 = mul(self.msg0.u_0, beta);
        self.msg0.u_2 = mul(self.msg0.u_2, beta);
        for w in &mut self.weights {
            *w = mul(*w, beta);
        }
    }

    pub(super) fn materialize(&mut self, rs: &[F128]) -> Vec<F128> {
        let d = self.depth;
        assert_eq!(rs.len(), d);
        let n = self.queries.len();
        let stride = d - 1;
        let reduced_log_d = self.log_d - d;
        // Keep the original NTT's twiddles. A fresh NTT at the smaller
        // dimension would describe a different linear map.
        let mut basis = if n <= 256 {
            let mut pos_stack = [0usize; 256];
            let mut val_stack = [F128::ZERO; 256];
            for i in 0..n {
                let q = self.queries[i];
                let alpha = self.weights[i];
                let qscales = &self.rows[i * stride][12..16];
                let mut v = alpha;
                for (s, &r_s) in rs.iter().enumerate() {
                    v = mul(v, F128::ONE + mul(r_s, qscales[s]));
                }
                pos_stack[i] = q >> d;
                val_stack[i] = v;
            }
            transpose_forward_ntt_sparse_impl(
                &self.ntt,
                &pos_stack[..n],
                &val_stack[..n],
                reduced_log_d,
                false,
                None,
                false,
            )
            .0
        } else {
            let mut positions = Vec::with_capacity(n);
            let mut values = Vec::with_capacity(n);
            for i in 0..n {
                let q = self.queries[i];
                let alpha = self.weights[i];
                let qscales = &self.rows[i * stride][12..16];
                let mut v = alpha;
                for (s, &r_s) in rs.iter().enumerate() {
                    v = mul(v, F128::ONE + mul(r_s, qscales[s]));
                }
                positions.push(q >> d);
                values.push(v);
            }
            transpose_forward_ntt_sparse_impl(
                &self.ntt,
                &positions,
                &values,
                reduced_log_d,
                false,
                None,
                false,
            )
            .0
        };
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
        let complete = self.challenges.len() == d;
        let (correction, materialized) = if complete {
            let (corr, basis) = self.dual.message_and_materialize(&self.challenges);
            (corr, Some(basis))
        } else {
            (self.dual.message(&self.challenges), None)
        };
        message.u_0 += correction.u_0;
        message.u_2 += correction.u_2;
        *sc.transcript.last_mut().expect("fold records a message") = message;
        if let Some(basis) = materialized {
            assert!(
                sc.pending_glue.is_none()
                    && sc.pending_fold_basis.is_none()
                    && sc.pending_ood_eq.is_none()
            );
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

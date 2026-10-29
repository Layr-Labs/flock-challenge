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
fn dot2(local: &[F128], weights: &[F128], half: usize) -> [F128; 2] {
    debug_assert_eq!(local.len(), 2 * half);
    debug_assert_eq!(weights.len(), half);
    match half {
        1 => [local[0], local[1]],
        2 => crate::field::f128_slice::fold_banked_slots2::<2>(weights.try_into().unwrap(), local),
        4 => crate::field::f128_slice::fold_banked_slots2::<4>(weights.try_into().unwrap(), local),
        8 => crate::field::f128_slice::fold_banked_slots2::<8>(weights.try_into().unwrap(), local),
        _ => [dot(&local[..half], weights), dot(&local[half..2 * half], weights)],
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

#[inline]
fn invert_local_staged(
    ntt: &AdditiveNttF128,
    log_d: usize,
    inv_layers: usize,
    cache_len: usize,
    base: usize,
    qoff: usize,
    cache: &mut [F128; 16],
) -> [F128; 30] {
    let mut stages = [F128::ZERO; 30];
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
        stages[len - 2..2 * len - 2].copy_from_slice(&cache[start..start + len]);
    }
    stages
}

pub(super) struct Dual {
    ntt: AdditiveNttF128,
    log_d: usize,
    log_inv_rate: usize,
    depth: usize,
    positions: Vec<usize>,
    weights: Vec<F128>,
    w0_pows: Vec<F128>,
    w2_pows: Vec<F128>,
    step_depth: usize,
    stages: Vec<[F128; 30]>,
    scales: Vec<[F128; 4]>,
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
        let mut stages = Vec::with_capacity(queries.len());
        let mut scales = Vec::with_capacity(queries.len());
        let mut positions = Vec::with_capacity(queries.len());
        let mut w2_pows = Vec::with_capacity(queries.len());
        let mut enforced = crate::field::F256Unreduced::ZERO;
        for (&q, &alpha) in queries.iter().zip(&weights) {
            let base = q & !mask;
            let qoff = q - base;
            let mut cache = [F128::ZERO; 16];
            for j in 0..cache_len {
                cache[j] = dot(row(base + j), &lane_weights);
            }
            enforced ^= alpha.mul_unreduced(cache[qoff]);
            stages.push(invert_local_staged(
                &ntt,
                log_d,
                inv_layers,
                cache_len,
                base,
                qoff,
                &mut cache,
            ));
            let mut qscales = [F128::ZERO; 4];
            for s in 0..=depth {
                qscales[s] = one_plus_scale(&ntt, log_d, s, q);
            }
            w2_pows.push(mul(alpha, qscales[0]));
            scales.push(qscales);
            positions.push(q >> depth);
        }
        let w0_pows = weights.clone();
        (
            Self {
                ntt,
                log_d,
                log_inv_rate,
                depth,
                positions,
                weights,
                w0_pows,
                w2_pows,
                step_depth: 0,
                stages,
                scales,
            },
            enforced.reduce(),
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
        let mut stages = crate::alloc_uninit_vec::<[F128; 30]>(n_queries);
        let mut scales = crate::alloc_uninit_vec::<[F128; 4]>(n_queries);
        let mut positions = crate::alloc_uninit_vec::<usize>(n_queries);
        let mut w2_pows = crate::alloc_uninit_vec::<F128>(n_queries);
        let enforced = if n_queries * lane_weights.len() >= 1024 {
            use rayon::prelude::*;
            queries
                .par_iter()
                .zip(&weights)
                .zip(stages.par_iter_mut())
                .zip(scales.par_iter_mut())
                .zip(positions.par_iter_mut())
                .zip(w2_pows.par_iter_mut())
                .map(|(((((&q, &w), stage_out), scale_out), pos_out), w2_out)| {
                    let base = q & !mask;
                    let qoff = q - base;
                    let mut cache = [F128::ZERO; 16];
                    for j in 0..cache_len {
                        cache[j] = dot(row(base + j), &lane_weights);
                    }
                    let enf = w.mul_unreduced(cache[qoff]);
                    *stage_out = invert_local_staged(
                        &ntt,
                        log_d,
                        inv_layers,
                        cache_len,
                        base,
                        qoff,
                        &mut cache,
                    );
                    for s in 0..=depth {
                        scale_out[s] = one_plus_scale(&ntt, log_d, s, q);
                    }
                    *pos_out = q >> depth;
                    *w2_out = mul(w, scale_out[0]);
                    enf
                })
                .reduce(
                    || crate::field::F256Unreduced::ZERO,
                    |mut a, b| {
                        a ^= b;
                        a
                    },
                )
                .reduce()
        } else {
            let mut enf_sum = crate::field::F256Unreduced::ZERO;
            for (((((&q, &w), stage_out), scale_out), pos_out), w2_out) in queries
                .iter()
                .zip(&weights)
                .zip(stages.iter_mut())
                .zip(scales.iter_mut())
                .zip(positions.iter_mut())
                .zip(w2_pows.iter_mut())
            {
                let base = q & !mask;
                let qoff = q - base;
                let mut cache = [F128::ZERO; 16];
                for j in 0..cache_len {
                    cache[j] = dot(row(base + j), &lane_weights);
                }
                enf_sum ^= w.mul_unreduced(cache[qoff]);
                *stage_out = invert_local_staged(
                    &ntt,
                    log_d,
                    inv_layers,
                    cache_len,
                    base,
                    qoff,
                    &mut cache,
                );
                for s in 0..=depth {
                    scale_out[s] = one_plus_scale(&ntt, log_d, s, q);
                }
                *pos_out = q >> depth;
                *w2_out = mul(w, scale_out[0]);
            }
            enf_sum.reduce()
        };
        let w0_pows = weights.clone();
        (
            Self {
                ntt,
                log_d,
                log_inv_rate,
                depth,
                positions,
                weights,
                w0_pows,
                w2_pows,
                step_depth: 0,
                stages,
                scales,
            },
            enforced,
        )
    }

    pub(super) fn message(&mut self, rs: &[F128]) -> SumcheckMessage {
        assert!(rs.len() <= self.depth);
        if rs.is_empty() && self.step_depth == 0 {
            let mut u0 = crate::field::F256Unreduced::ZERO;
            let mut u2 = crate::field::F256Unreduced::ZERO;
            for ((stage, &c0), &c2) in self.stages.iter().zip(&self.w0_pows).zip(&self.w2_pows) {
                let f0 = stage[0];
                let f1 = stage[1];
                u0 ^= c0.mul_unreduced(f0);
                u2 ^= c2.mul_unreduced(f0 + f1);
            }
            return SumcheckMessage {
                u_0: u0.reduce(),
                u_2: u2.reduce(),
            };
        }
        let k = rs.len() + 1;
        let size = 1usize << k;
        let half = size / 2;
        let eq_weights = build_eq_table(rs);
        let mut u0 = crate::field::F256Unreduced::ZERO;
        let mut u2 = crate::field::F256Unreduced::ZERO;
        if rs.len() == self.step_depth + 1 {
            let r_prev = rs[self.step_depth];
            for (((stage, qscales), w0), w2) in self
                .stages
                .iter()
                .zip(&self.scales)
                .zip(&mut self.w0_pows)
                .zip(&mut self.w2_pows)
            {
                let local = &stage[size - 2..2 * size - 2];
                let [f0, f1] = dot2(local, &eq_weights, half);
                let c0 = *w0 + mul(r_prev, *w2);
                let c2 = mul(c0, qscales[k - 1]);
                *w0 = c0;
                *w2 = c2;
                u0 ^= c0.mul_unreduced(f0);
                u2 ^= c2.mul_unreduced(f0 + f1);
            }
            self.step_depth = rs.len();
        } else {
            for ((stage, qscales), &w_init) in
                self.stages.iter().zip(&self.scales).zip(&self.weights)
            {
                let local = &stage[size - 2..2 * size - 2];
                let [f0, f1] = dot2(local, &eq_weights, half);
                let mut c0 = w_init;
                for (s, &r_s) in rs.iter().enumerate() {
                    c0 = mul(c0, F128::ONE + mul(r_s, qscales[s]));
                }
                let c2 = mul(c0, qscales[k - 1]);
                u0 ^= c0.mul_unreduced(f0);
                u2 ^= c2.mul_unreduced(f0 + f1);
            }
        }
        SumcheckMessage {
            u_0: u0.reduce(),
            u_2: u2.reduce(),
        }
    }

    pub(super) fn scale(&mut self, beta: F128) {
        for (((w, w0), w2), qscales) in self
            .weights
            .iter_mut()
            .zip(&mut self.w0_pows)
            .zip(&mut self.w2_pows)
            .zip(&self.scales)
        {
            let nw = mul(*w, beta);
            *w = nw;
            *w0 = nw;
            *w2 = mul(nw, qscales[0]);
        }
        self.step_depth = 0;
    }

    pub(super) fn materialize(&mut self, rs: &[F128]) -> Vec<F128> {
        let d = self.depth;
        assert_eq!(rs.len(), d);
        if self.step_depth != d {
            for ((&alpha, qscales), w0) in
                self.weights.iter().zip(&self.scales).zip(&mut self.w0_pows)
            {
                let mut v = alpha;
                for (s, &r_s) in rs.iter().enumerate() {
                    v = mul(v, F128::ONE + mul(r_s, qscales[s]));
                }
                *w0 = v;
            }
        }
        let reduced_log_d = self.log_d - d;
        let (mut basis, _) = transpose_forward_ntt_sparse_impl(
            &self.ntt,
            &self.positions,
            &self.w0_pows,
            reduced_log_d,
            false,
            None,
            false,
        );
        basis.truncate(1usize << (reduced_log_d - self.log_inv_rate));
        basis
    }
}

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

    pub(super) fn fold(&mut self, r: F128, sc: &mut SumcheckProver) -> (SumcheckMessage, bool) {
        let d = self.dual.depth;
        assert!(self.glued && self.challenges.len() < d);
        self.challenges.push(r);
        let complete = self.challenges.len() == d;
        if complete {
            assert!(
                sc.pending_glue.is_none()
                    && sc.pending_fold_basis.is_none()
                    && sc.pending_ood_eq.is_none()
            );
            let dual = &mut self.dual;
            let challenges = &self.challenges;
            let (mut message, (correction, basis)) = rayon::join(
                || sc.fold(r),
                || {
                    let corr = dual.message(challenges);
                    let b = dual.materialize(challenges);
                    (corr, b)
                },
            );
            message.u_0 += correction.u_0;
            message.u_2 += correction.u_2;
            *sc.transcript.last_mut().expect("fold records a message") = message;
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
            (message, true)
        } else {
            let dual = &mut self.dual;
            let challenges = &self.challenges;
            let (mut message, correction) =
                rayon::join(|| sc.fold(r), || dual.message(challenges));
            message.u_0 += correction.u_0;
            message.u_2 += correction.u_2;
            *sc.transcript.last_mut().expect("fold records a message") = message;
            (message, false)
        }
    }
}

pub(super) fn enabled(lazy_l1_selected: bool, disabled: Option<&std::ffi::OsStr>) -> bool {
    lazy_l1_selected && disabled != Some(std::ffi::OsStr::new("1"))
}

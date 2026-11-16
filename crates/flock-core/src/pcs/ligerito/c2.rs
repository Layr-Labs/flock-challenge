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

/// The part of one query of [`Dual::new_sync`] that does not depend on the
/// α weights: the query's own lane-folded entry, its local inverse stages and
/// its scale factors. Computed by [`DualPrep::query`] while the queries are
/// still being drawn (query overlap, `FLOCK_NO_QUERY_OVERLAP=1` keeps
/// `new_sync`).
pub(super) struct DualQuery {
    cache_q: F128,
    stage: [F128; 30],
    scale: [F128; 4],
}

/// What [`Dual::new_sync`] derives before its per-query loop, minus the α
/// weights (α is drawn after the queries).
pub(super) struct DualPrep {
    ntt: AdditiveNttF128,
    log_d: usize,
    depth: usize,
    lane_weights: Vec<F128>,
}

impl DualPrep {
    pub(super) fn new(depth: usize, log_d: usize, lane_challenges: &[F128]) -> Self {
        assert!((2..=3).contains(&depth));
        Self {
            ntt: AdditiveNttF128::standard(log_d),
            log_d,
            depth,
            lane_weights: build_eq_table(lane_challenges),
        }
    }

    /// The per-query body of [`Dual::new_sync`] for query `q`, minus its two
    /// products by the query's α weight (applied in [`Dual::from_prepared`]).
    pub(super) fn query<'a>(&self, q: usize, row: impl Fn(usize) -> &'a [F128]) -> DualQuery {
        let cache_len = 1usize << (self.depth + 1);
        let mask = cache_len - 1;
        let inv_layers = self.depth + 1;
        let base = q & !mask;
        let qoff = q - base;
        let mut cache = [F128::ZERO; 16];
        for j in 0..cache_len {
            cache[j] = dot(row(base + j), &self.lane_weights);
        }
        let cache_q = cache[qoff];
        let stage = invert_local_staged(
            &self.ntt,
            self.log_d,
            inv_layers,
            cache_len,
            base,
            qoff,
            &mut cache,
        );
        let mut scale = [F128::ZERO; 4];
        for s in 0..=self.depth {
            scale[s] = one_plus_scale(&self.ntt, self.log_d, s, q);
        }
        DualQuery {
            cache_q,
            stage,
            scale,
        }
    }
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

    /// [`Self::new_sync`] for the sorted `queries`, from the per-query parts
    /// [`DualPrep::query`] computed for them (`parts[i]` belongs to
    /// `queries[i]`). Same fields, same enforced sum: the α products are the
    /// two `new_sync` takes per query, and the enforced sum is an XOR of the
    /// same unreduced products, reduced once. Large inputs fill the fields on
    /// the pool, as `new_sync` does.
    pub(super) fn from_prepared(
        prep: &DualPrep,
        log_inv_rate: usize,
        queries: &[usize],
        alpha: &[F128],
        parts: &[&DualQuery],
    ) -> (Self, F128) {
        let (log_d, depth) = (prep.log_d, prep.depth);
        assert!(log_inv_rate + depth < log_d);
        assert!(queries.iter().all(|&q| q < 1usize << log_d));
        assert_eq!(parts.len(), queries.len());
        let ntt = AdditiveNttF128::standard(log_d);
        let weights: Vec<F128> = if queries.is_empty() {
            Vec::new()
        } else {
            build_eq_table(alpha)
                .into_iter()
                .take(queries.len())
                .collect()
        };
        assert_eq!(queries.len(), weights.len());
        let n_queries = queries.len();
        let (stages, scales, positions, w2_pows, enforced) = if n_queries >= 64 {
            use rayon::prelude::*;
            let mut stages = crate::alloc_uninit_vec::<[F128; 30]>(n_queries);
            let mut scales = crate::alloc_uninit_vec::<[F128; 4]>(n_queries);
            let mut positions = crate::alloc_uninit_vec::<usize>(n_queries);
            let mut w2_pows = crate::alloc_uninit_vec::<F128>(n_queries);
            let enforced = queries
                .par_iter()
                .zip(&weights)
                .zip(parts)
                .zip(stages.par_iter_mut())
                .zip(scales.par_iter_mut())
                .zip(positions.par_iter_mut())
                .zip(w2_pows.par_iter_mut())
                .with_min_len(16)
                .map(|((((((&q, &w), part), stage_out), scale_out), pos_out), w2_out)| {
                    *stage_out = part.stage;
                    *scale_out = part.scale;
                    *pos_out = q >> depth;
                    *w2_out = mul(w, part.scale[0]);
                    w.mul_unreduced(part.cache_q)
                })
                .reduce(
                    || crate::field::F256Unreduced::ZERO,
                    |mut a, b| {
                        a ^= b;
                        a
                    },
                );
            (stages, scales, positions, w2_pows, enforced)
        } else {
            let mut stages = Vec::with_capacity(n_queries);
            let mut scales = Vec::with_capacity(n_queries);
            let mut positions = Vec::with_capacity(n_queries);
            let mut w2_pows = Vec::with_capacity(n_queries);
            let mut enforced = crate::field::F256Unreduced::ZERO;
            for ((&q, &w), part) in queries.iter().zip(&weights).zip(parts) {
                enforced ^= w.mul_unreduced(part.cache_q);
                stages.push(part.stage);
                w2_pows.push(mul(w, part.scale[0]));
                scales.push(part.scale);
                positions.push(q >> depth);
            }
            (stages, scales, positions, w2_pows, enforced)
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
        let par_floor = if std::env::var("FLOCK_NO_OP_DUAL_PAR").map_or(true, |v| v != "1") {
            256
        } else {
            1024
        };
        let enforced = if n_queries * lane_weights.len() >= par_floor {
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
        if c2_retained_transpose_enabled() {
            if let Some(basis) = transpose_forward_ntt_sparse_retained(
                &self.ntt,
                &self.positions,
                &self.w0_pows,
                reduced_log_d,
                self.log_inv_rate,
            ) {
                debug_assert_eq!(
                    basis.len(),
                    1usize << (reduced_log_d - self.log_inv_rate)
                );
                return basis;
            }
        }
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

#[cfg(test)]
mod prepared_tests {
    use super::*;

    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f128(state: &mut u64) -> F128 {
        F128 {
            lo: next(state),
            hi: next(state),
        }
    }

    /// `from_prepared` over `DualPrep::query` parts against `new_sync`:
    /// every field, the enforced sum, and the messages of the folds that
    /// follow, at the ranked L0 (64 lanes, depth 2 and 3) and level (8 lanes,
    /// depth 3) geometries, on both the serial and the parallel arm of
    /// `new_sync`.
    #[test]
    fn from_prepared_matches_new_sync() {
        // (depth, log_d, log_inv_rate, log_lanes, queries)
        let cases = [
            (2usize, 12usize, 2usize, 6usize, 218usize),
            (3, 12, 2, 6, 218),
            (3, 11, 3, 3, 106),
            (3, 9, 3, 3, 93),
            (3, 8, 3, 3, 60),
            (2, 8, 2, 1, 5),
            (3, 7, 3, 3, 0),
        ];
        for (case, &(depth, log_d, log_inv_rate, log_lanes, n_queries)) in cases.iter().enumerate() {
            let mut st = 0xC2_D0A1_0000 + case as u64;
            let lanes = 1usize << log_lanes;
            let rows: Vec<F128> = (0..(lanes << log_d)).map(|_| f128(&mut st)).collect();
            let row = |q: usize| -> &[F128] { &rows[q * lanes..(q + 1) * lanes] };
            let lane_challenges: Vec<F128> = (0..log_lanes).map(|_| f128(&mut st)).collect();
            let mut queries = std::collections::BTreeSet::new();
            while queries.len() < n_queries {
                queries.insert((next(&mut st) as usize) % (1usize << log_d));
            }
            let queries: Vec<usize> = queries.into_iter().collect();
            let alpha: Vec<F128> = (0..n_queries.next_power_of_two().trailing_zeros())
                .map(|_| f128(&mut st))
                .collect();

            let (mut want, want_enforced) =
                Dual::new_sync(depth, log_d, log_inv_rate, &lane_challenges, &queries, &alpha, row);
            let prep = DualPrep::new(depth, log_d, &lane_challenges);
            let parts: Vec<DualQuery> = queries.iter().map(|&q| prep.query(q, row)).collect();
            let refs: Vec<&DualQuery> = parts.iter().collect();
            let (mut got, got_enforced) =
                Dual::from_prepared(&prep, log_inv_rate, &queries, &alpha, &refs);

            assert_eq!(got_enforced, want_enforced, "case {case}: enforced sum");
            assert_eq!(got.log_d, want.log_d);
            assert_eq!(got.log_inv_rate, want.log_inv_rate);
            assert_eq!(got.depth, want.depth);
            assert_eq!(got.step_depth, want.step_depth);
            assert_eq!(got.positions, want.positions, "case {case}: positions");
            assert_eq!(got.weights, want.weights, "case {case}: weights");
            assert_eq!(got.w0_pows, want.w0_pows, "case {case}: w0");
            assert_eq!(got.w2_pows, want.w2_pows, "case {case}: w2");
            assert_eq!(got.stages, want.stages, "case {case}: stages");
            assert_eq!(got.scales.len(), want.scales.len());
            for (g, w) in got.scales.iter().zip(&want.scales) {
                // `new_sync`'s parallel arm leaves the slots past `depth`
                // unwritten; only `0..=depth` is ever read.
                assert_eq!(g[..=depth], w[..=depth], "case {case}: scales");
            }
            // The messages the dual answers during the folds that follow.
            let rs: Vec<F128> = (0..depth).map(|_| f128(&mut st)).collect();
            for k in 0..=depth {
                assert_eq!(
                    got.message(&rs[..k]),
                    want.message(&rs[..k]),
                    "case {case}: message after {k} folds"
                );
            }
        }
    }
}

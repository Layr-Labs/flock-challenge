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

/// Exact rollback for the four-row query-window dots of
/// [`Dual::new_sync`]: `FLOCK_NO_C2_QUAD_ROWS` (any value) restores one
/// [`dot`] per row. Same bytes either way.
fn quad_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_C2_QUAD_ROWS").is_none())
}

/// `cache[j] = dot(row(base + j), weights)` for `j < cache_len`. When the
/// caller hands the flat row-major matrix behind `row` (`row(q)` is
/// `flat[q·W..(q+1)·W]` with `W = weights.len()`), each aligned group of
/// four adjacent rows is folded by one four-slot banked kernel that loads
/// every weight once for the four rows. Each output is the same exact field
/// sum `Σ_b w_b·x_b`, so every cache value is bit-identical to [`dot`].
#[inline]
fn window_dots<'a>(
    row: &impl Fn(usize) -> &'a [F128],
    flat: Option<&'a [F128]>,
    base: usize,
    cache_len: usize,
    weights: &[F128],
    cache: &mut [F128; 16],
) {
    let w = weights.len();
    if let Some(flat) = flat
        && cache_len.is_multiple_of(4)
        && (w == 64 || w == 8)
        && (base + cache_len) * w <= flat.len()
        && core::ptr::eq(row(base).as_ptr(), flat[base * w..].as_ptr())
        && quad_rows_enabled()
    {
        let rows = &flat[base * w..(base + cache_len) * w];
        for (j, quad) in rows.chunks_exact(4 * w).enumerate() {
            let out = &mut cache[4 * j..4 * j + 4];
            if w == 64 {
                crate::field::f128_slice::fold_banked_slots4_whole_into::<64>(
                    weights.try_into().unwrap(),
                    quad,
                    out,
                    64,
                );
            } else {
                crate::field::f128_slice::fold_banked_slots4_whole_into::<8>(
                    weights.try_into().unwrap(),
                    quad,
                    out,
                    8,
                );
            }
        }
        return;
    }
    for j in 0..cache_len {
        cache[j] = dot(row(base + j), weights);
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_sync<'a>(
        depth: usize,
        log_d: usize,
        log_inv_rate: usize,
        lane_challenges: &[F128],
        queries: &[usize],
        alpha: &[F128],
        row: impl Fn(usize) -> &'a [F128] + Sync,
        flat: Option<&'a [F128]>,
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
                    window_dots(&row, flat, base, cache_len, &lane_weights, &mut cache);
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
                window_dots(&row, flat, base, cache_len, &lane_weights, &mut cache);
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
        match self.materialize_deferred(rs, false) {
            Materialized::Basis(basis) => basis,
            Materialized::Prefix(..) => unreachable!("prefix only on request"),
        }
    }

    /// [`Self::materialize`], except that with `defer_tail` the retained
    /// transpose stops before its last stage and hands the stage back, so
    /// the caller can run it straight into its destination
    /// ([`transpose_forward_ntt_sparse_retained_finish_add`]).
    pub(super) fn materialize_deferred(&mut self, rs: &[F128], defer_tail: bool) -> Materialized {
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
        if c2_retained_transpose_enabled() && defer_tail {
            if let Some((data, tail)) = transpose_forward_ntt_sparse_retained_prefix(
                &self.ntt,
                &self.positions,
                &self.w0_pows,
                reduced_log_d,
                self.log_inv_rate,
            ) {
                return Materialized::Prefix(data, tail, reduced_log_d);
            }
        }
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
                return Materialized::Basis(basis);
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
        Materialized::Basis(basis)
    }
}

pub(super) enum Materialized {
    Basis(Vec<F128>),
    /// Retained-transpose data, its owed last stage, and the transform's
    /// `log_d`.
    Prefix(Vec<F128>, RetainedTail, usize),
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
            let fused_add = c2_fused_add_enabled();
            let (mut message, (correction, materialized)) = rayon::join(
                || sc.fold(r),
                || {
                    let corr = dual.message(challenges);
                    let b = dual.materialize_deferred(challenges, fused_add);
                    (corr, b)
                },
            );
            message.u_0 += correction.u_0;
            message.u_2 += correction.u_2;
            *sc.transcript.last_mut().expect("fold records a message") = message;
            let basis = match materialized {
                Materialized::Basis(basis) => basis,
                Materialized::Prefix(data, tail, log_d) => {
                    transpose_forward_ntt_sparse_retained_finish_add(
                        &dual.ntt,
                        data,
                        log_d,
                        tail,
                        &mut sc.combined_basis,
                    );
                    return (message, true);
                }
            };
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

/// Exact rollback for overlapping each query's alpha-independent C2 setup
/// with the serial query sampling: `FLOCK_NO_L0_QUERY_OVERLAP` (any value)
/// samples every query first and then runs [`Dual::new_sync`]. Same bytes
/// either way.
pub(super) fn query_overlap_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_L0_QUERY_OVERLAP").is_none())
}

/// Opt-in for the query overlap at the deep C2 levels
/// (`FLOCK_DEEP_QUERY_OVERLAP`, any value). Off by default: on mini1 the
/// 106-query L1 opening measured ~0.02 ms slower with it (spawn and
/// path-touch overhead against a small dual) and L2 ~0.02 ms faster, so the
/// deep levels net out to nothing; L0 keeps it.
pub(super) fn deep_query_overlap_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        query_overlap_enabled() && std::env::var_os("FLOCK_DEEP_QUERY_OVERLAP").is_some()
    })
}

/// The alpha-independent part of one query's [`Dual::new_sync`] work.
#[derive(Clone, Copy)]
pub(super) struct PreparedQuery {
    stage: [F128; 30],
    scales: [F128; 4],
    /// `cache[qoff]` before the local inversion (the opened value the
    /// enforced sum weights by alpha).
    at_query: F128,
}

const PREPARED_ZERO: PreparedQuery = PreparedQuery {
    stage: [F128::ZERO; 30],
    scales: [F128::ZERO; 4],
    at_query: F128::ZERO,
};

/// Exactly the per-query body of [`Dual::new_sync`] minus the two values
/// that need the query weight (`enforced` term and `w2_pows`).
#[inline]
fn prepare_query<'a>(
    ntt: &AdditiveNttF128,
    log_d: usize,
    depth: usize,
    row: &(impl Fn(usize) -> &'a [F128] + Sync),
    flat: Option<&'a [F128]>,
    lane_weights: &[F128],
    q: usize,
) -> PreparedQuery {
    let cache_len = 1usize << (depth + 1);
    let mask = cache_len - 1;
    let inv_layers = depth + 1;
    let base = q & !mask;
    let qoff = q - base;
    let mut cache = [F128::ZERO; 16];
    window_dots(row, flat, base, cache_len, lane_weights, &mut cache);
    let at_query = cache[qoff];
    let stage = invert_local_staged(ntt, log_d, inv_layers, cache_len, base, qoff, &mut cache);
    let mut scales = [F128::ZERO; 4];
    for s in 0..=depth {
        scales[s] = one_plus_scale(ntt, log_d, s, q);
    }
    PreparedQuery {
        stage,
        scales,
        at_query,
    }
}

/// Read every sibling hash the multi-proof of leaf `q` can emit (one per
/// level of the flat tree), so the serial `merkle_multi_proof` that follows
/// the sampling finds them in cache. Loads only; nothing is written.
#[inline]
fn touch_merkle_path(tree: &[Hash], num_leaves: usize, q: usize) {
    let mut level_start = 0usize;
    let mut level_len = num_leaves;
    let mut p = q;
    let mut acc = 0u8;
    while level_len > 1 {
        acc ^= tree[level_start + (p ^ 1)][0];
        level_start += level_len;
        level_len >>= 1;
        p >>= 1;
    }
    core::hint::black_box(acc);
}

/// Sorted queries with their prepared setups, awaiting the alpha challenge.
pub(super) struct PreparedDual {
    ntt: AdditiveNttF128,
    log_d: usize,
    log_inv_rate: usize,
    depth: usize,
    prepared: Vec<PreparedQuery>,
}

/// Sample `count` distinct queries in `[0, block_len)` exactly as
/// `sample_distinct_queries` does (same challenger calls, same dedup, sorted
/// output) while the pool's other workers prepare each newly sampled query's
/// alpha-independent C2 setup ([`prepare_query`]). The helper jobs are
/// spawned into the current pool, where the open's hot-spinning workers pick
/// them up through `rayon::yield_now`; a helper that claims a slot not yet
/// sampled spins until the sampler publishes it. The sampler never waits on a
/// helper, so a single-thread pool simply runs the helpers after sampling.
#[allow(clippy::too_many_arguments)]
pub(super) fn sample_and_prepare<'a, Ch: Challenger, R>(
    challenger: &mut Ch,
    block_len: usize,
    count: usize,
    depth: usize,
    log_d: usize,
    log_inv_rate: usize,
    lane_challenges: &[F128],
    row: &R,
    flat: Option<&'a [F128]>,
    tree: &[Hash],
) -> (Vec<usize>, PreparedDual, Vec<Vec<F128>>)
where
    R: Fn(usize) -> &'a [F128] + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering};
    assert!(
        count <= block_len,
        "sample_and_prepare: count ({count}) > block_len ({block_len})"
    );
    assert!((2..=3).contains(&depth));
    assert!(log_inv_rate + depth < log_d);
    assert!(block_len <= 1usize << log_d);
    let lane_weights = build_eq_table(lane_challenges);
    let ntt = AdditiveNttF128::standard(log_d);
    let raw_q: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
    let published = AtomicUsize::new(0);
    let next = AtomicUsize::new(0);
    let mut prepared = vec![PREPARED_ZERO; count];
    let out = crate::epool::SyncPtr(prepared.as_mut_ptr());
    let mut rows: Vec<Vec<F128>> = (0..count).map(|_| Vec::new()).collect();
    let rows_addr = rows.as_mut_ptr() as usize;
    assert_eq!(tree.len(), 2 * block_len - 1);
    let work = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= count {
                break;
            }
            while published.load(Ordering::Acquire) <= i {
                std::hint::spin_loop();
            }
            let q = raw_q[i].load(Ordering::Relaxed);
            let p = prepare_query(&ntt, log_d, depth, row, flat, &lane_weights, q);
            let opened = row(q).to_vec();
            touch_merkle_path(tree, block_len, q);
            // SAFETY: slot `i < count` is claimed by exactly one caller of
            // `work` (unique `fetch_add` ticket); the scope joins every
            // writer before `prepared`/`rows` are read again.
            unsafe {
                out.ptr().add(i).write(p);
                *(rows_addr as *mut Vec<F128>).add(i) = opened;
            }
        }
    };
    let raw: Vec<usize> = rayon::scope(|s| {
        let helpers = rayon::current_num_threads().saturating_sub(1);
        for _ in 0..helpers {
            s.spawn(|_| work());
        }
        let mut seen = std::collections::HashSet::new();
        let mut raw = Vec::with_capacity(count);
        while raw.len() < count {
            let v = challenger.sample_f128();
            let q = (v.lo as usize) % block_len;
            if seen.insert(q) {
                raw_q[raw.len()].store(q, Ordering::Relaxed);
                raw.push(q);
                published.store(raw.len(), Ordering::Release);
            }
        }
        work();
        raw
    });
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_unstable_by_key(|&i| raw[i]);
    let queries: Vec<usize> = order.iter().map(|&i| raw[i]).collect();
    let prepared: Vec<PreparedQuery> = order.iter().map(|&i| prepared[i]).collect();
    let rows: Vec<Vec<F128>> = order
        .iter()
        .map(|&i| std::mem::take(&mut rows[i]))
        .collect();
    (
        queries,
        PreparedDual {
            ntt,
            log_d,
            log_inv_rate,
            depth,
            prepared,
        },
        rows,
    )
}

impl PreparedDual {
    /// Finish [`Dual::new_sync`] from the prepared setups: the same weights,
    /// positions, stages, scales, `w2_pows` and enforced sum (an XOR sum of
    /// the same exact products, reduced once).
    pub(super) fn into_dual(self, queries: &[usize], alpha: &[F128]) -> (Dual, F128) {
        let Self {
            ntt,
            log_d,
            log_inv_rate,
            depth,
            prepared,
        } = self;
        assert_eq!(queries.len(), prepared.len());
        let weights: Vec<F128> = if queries.is_empty() {
            Vec::new()
        } else {
            build_eq_table(alpha)
                .into_iter()
                .take(queries.len())
                .collect()
        };
        assert_eq!(queries.len(), weights.len());
        let mut enforced = crate::field::F256Unreduced::ZERO;
        let mut stages = Vec::with_capacity(queries.len());
        let mut scales = Vec::with_capacity(queries.len());
        let mut positions = Vec::with_capacity(queries.len());
        let mut w2_pows = Vec::with_capacity(queries.len());
        for ((&q, &w), p) in queries.iter().zip(&weights).zip(&prepared) {
            enforced ^= w.mul_unreduced(p.at_query);
            stages.push(p.stage);
            scales.push(p.scales);
            positions.push(q >> depth);
            w2_pows.push(mul(w, p.scales[0]));
        }
        let w0_pows = weights.clone();
        (
            Dual {
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
}

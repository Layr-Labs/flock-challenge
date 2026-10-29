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
    queries: Vec<usize>,
    weights: Vec<F128>,
    stages: Vec<[F128; 30]>,
    scales: Vec<[F128; 4]>,
}

static C2_STAGES_POOL: std::sync::Mutex<Option<Vec<[F128; 30]>>> = std::sync::Mutex::new(None);
static C2_SCALES_POOL: std::sync::Mutex<Option<Vec<[F128; 4]>>> = std::sync::Mutex::new(None);

fn take_c2_stages(len: usize) -> Vec<[F128; 30]> {
    if let Ok(mut guard) = C2_STAGES_POOL.lock() {
        if let Some(mut v) = guard.take() {
            if v.capacity() >= len {
                unsafe { v.set_len(len) };
                return v;
            }
            *guard = Some(v);
        }
    }
    let cap = len.max(256);
    let mut v = crate::alloc_uninit_vec::<[F128; 30]>(cap);
    unsafe { v.set_len(len) };
    v
}

fn take_c2_scales(len: usize) -> Vec<[F128; 4]> {
    if let Ok(mut guard) = C2_SCALES_POOL.lock() {
        if let Some(mut v) = guard.take() {
            if v.capacity() >= len {
                unsafe { v.set_len(len) };
                return v;
            }
            *guard = Some(v);
        }
    }
    let cap = len.max(256);
    let mut v = crate::alloc_uninit_vec::<[F128; 4]>(cap);
    unsafe { v.set_len(len) };
    v
}

impl Drop for Dual {
    fn drop(&mut self) {
        let mut stages = core::mem::take(&mut self.stages);
        if stages.capacity() >= 256 {
            stages.clear();
            if let Ok(mut guard) = C2_STAGES_POOL.lock() {
                if guard.as_ref().map_or(true, |old| stages.capacity() > old.capacity()) {
                    *guard = Some(stages);
                }
            }
        }
        let mut scales = core::mem::take(&mut self.scales);
        if scales.capacity() >= 256 {
            scales.clear();
            if let Ok(mut guard) = C2_SCALES_POOL.lock() {
                if guard.as_ref().map_or(true, |old| scales.capacity() > old.capacity()) {
                    *guard = Some(scales);
                }
            }
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
        let mut enforced = F128::ZERO;
        for (&q, &alpha) in queries.iter().zip(&weights) {
            let base = q & !mask;
            let qoff = q - base;
            let mut cache = [F128::ZERO; 16];
            for j in 0..cache_len {
                cache[j] = dot(row(base + j), &lane_weights);
            }
            enforced += mul(alpha, cache[qoff]);
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
            scales.push(qscales);
        }
        (
            Self {
                ntt,
                log_d,
                log_inv_rate,
                depth,
                queries: queries.to_vec(),
                weights,
                stages,
                scales,
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
        let n_queries = queries.len();
        let mut stages = take_c2_stages(n_queries);
        let mut scales = take_c2_scales(n_queries);
        let enforced = if n_queries * lane_weights.len() >= 1024 {
            use rayon::prelude::*;
            queries
                .par_iter()
                .zip(&weights)
                .zip(stages.par_iter_mut())
                .zip(scales.par_iter_mut())
                .map(|(((&q, &w), stage_out), scale_out)| {
                    let base = q & !mask;
                    let qoff = q - base;
                    let mut cache = [F128::ZERO; 16];
                    for j in 0..cache_len {
                        cache[j] = dot(row(base + j), &lane_weights);
                    }
                    let enf = mul(w, cache[qoff]);
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
                    enf
                })
                .reduce(|| F128::ZERO, |a, b| a + b)
        } else {
            let mut enf_sum = F128::ZERO;
            for (((&q, &w), stage_out), scale_out) in queries
                .iter()
                .zip(&weights)
                .zip(stages.iter_mut())
                .zip(scales.iter_mut())
            {
                let base = q & !mask;
                let qoff = q - base;
                let mut cache = [F128::ZERO; 16];
                for j in 0..cache_len {
                    cache[j] = dot(row(base + j), &lane_weights);
                }
                enf_sum += mul(w, cache[qoff]);
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
            }
            enf_sum
        };
        (
            Self {
                ntt,
                log_d,
                log_inv_rate,
                depth,
                queries: queries.to_vec(),
                weights,
                stages,
                scales,
            },
            enforced,
        )
    }

    pub(super) fn message(&mut self, rs: &[F128]) -> SumcheckMessage {
        assert!(rs.len() <= self.depth);
        let k = rs.len() + 1;
        let size = 1usize << k;
        let half = size / 2;
        let weights = build_eq_table(rs);
        let mut u0_acc = crate::field::F256Unreduced::ZERO;
        let mut u2_acc = crate::field::F256Unreduced::ZERO;
        for i in 0..self.queries.len() {
            let local = &self.stages[i][size - 2..2 * size - 2];
            let [f0, f1] = dot2(local, &weights, half);
            let qscales = &self.scales[i];
            let mut c0 = self.weights[i];
            for (s, &r_s) in rs.iter().enumerate() {
                c0 = mul(c0, F128::ONE + mul(r_s, qscales[s]));
            }
            let c2 = mul(c0, qscales[k - 1]);
            u0_acc ^= c0.mul_unreduced(f0);
            u2_acc ^= c2.mul_unreduced(f0 + f1);
        }
        SumcheckMessage {
            u_0: u0_acc.reduce(),
            u_2: u2_acc.reduce(),
        }
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
        for ((&q, &alpha), qscales) in self.queries.iter().zip(&self.weights).zip(&self.scales) {
            let mut v = alpha;
            for (s, &r_s) in rs.iter().enumerate() {
                v = mul(v, F128::ONE + mul(r_s, qscales[s]));
            }
            positions.push(q >> d);
            values.push(v);
        }
        let reduced_log_d = self.log_d - d;
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

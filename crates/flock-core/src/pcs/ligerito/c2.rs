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

/// Lanes of an L0 C2 row before the shared tail.
const ROW_HEAD: usize = 57;

/// Kill switch for the shared tail of adjacent C2 rows:
/// `FLOCK_NO_C2_SHARED_TAIL=1` (any value) forms both full dots. Read once
/// per process.
fn c2_shared_tail_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_C2_SHARED_TAIL").is_none())
}

/// Kill switch for four-row weight-sharing Karatsuba NEON lane dots in C2
/// dual construction (`FLOCK_NO_C2_ROW_DOT_KARA`, any value restores the
/// incumbent per-pair `row_dot_pair` `fold_banked_slot` calls). Read once per
/// process.
pub const ENV_NO_C2_ROW_DOT_KARA: &str = "FLOCK_NO_C2_ROW_DOT_KARA";

#[inline]
fn c2_row_dot_kara_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(ENV_NO_C2_ROW_DOT_KARA).is_none())
}

/// The lane dots `[a · w, b · w]` of two adjacent C2 rows. When both have
/// 64 lanes and agree on lanes 57..64, that tail's seven products are formed
/// once and added to each row's 57-lane head: the same field sums. At L0
/// these lanes are the witness padding (banks 57..63 of every odd 64-word
/// slot are zero), so each such lane's message has zero odd coefficients and
/// its codeword takes equal values at positions `2i` and `2i + 1`.
#[inline]
fn row_dot_pair(a: &[F128], b: &[F128], w: &[F128]) -> [F128; 2] {
    if c2_shared_tail_enabled()
        && w.len() == 64
        && a.len() == 64
        && b.len() == 64
        && a[ROW_HEAD..] == b[ROW_HEAD..]
    {
        use crate::field::f128_slice::fold_banked_slot;
        let head: &[F128; ROW_HEAD] = w[..ROW_HEAD].try_into().unwrap();
        let tail = fold_banked_slot::<{ 64 - ROW_HEAD }>(w[ROW_HEAD..].try_into().unwrap(), &a[ROW_HEAD..]);
        return [
            fold_banked_slot::<ROW_HEAD>(head, a) + tail,
            fold_banked_slot::<ROW_HEAD>(head, b) + tail,
        ];
    }
    [dot(a, w), dot(b, w)]
}

/// The lane dots `[r0 · w, r1 · w, r2 · w, r3 · w]` of four consecutive C2
/// rows. Shares each weight pair across all four rows via Karatsuba NEON
/// (`fold_banked_quad_disjoint` / `fold_banked_pair_disjoint`), preserving the
/// exact L0 shared-tail decomposition (`57 + 7`) and bit-identical field sums.
#[inline]
fn row_dot_quad(r0: &[F128], r1: &[F128], r2: &[F128], r3: &[F128], w: &[F128]) -> [F128; 4] {
    if !c2_row_dot_kara_enabled() {
        let [d0, d1] = row_dot_pair(r0, r1, w);
        let [d2, d3] = row_dot_pair(r2, r3, w);
        return [d0, d1, d2, d3];
    }
    use crate::field::f128_slice::{fold_banked_pair_disjoint, fold_banked_quad_disjoint};
    if c2_shared_tail_enabled()
        && w.len() == 64
        && r0.len() == 64
        && r1.len() == 64
        && r2.len() == 64
        && r3.len() == 64
        && r0[ROW_HEAD..] == r1[ROW_HEAD..]
        && r2[ROW_HEAD..] == r3[ROW_HEAD..]
    {
        let head: &[F128; ROW_HEAD] = w[..ROW_HEAD].try_into().unwrap();
        let tail_w: &[F128; 64 - ROW_HEAD] = w[ROW_HEAD..].try_into().unwrap();
        let [t01, t23] =
            fold_banked_pair_disjoint::<{ 64 - ROW_HEAD }>(tail_w, &r0[ROW_HEAD..], &r2[ROW_HEAD..]);
        let [h0, h1, h2, h3] = fold_banked_quad_disjoint::<ROW_HEAD>(head, r0, r1, r2, r3);
        return [h0 + t01, h1 + t01, h2 + t23, h3 + t23];
    }
    if r0.len() == w.len() && r1.len() == w.len() && r2.len() == w.len() && r3.len() == w.len() {
        match w.len() {
            64 => return fold_banked_quad_disjoint::<64>(w.try_into().unwrap(), r0, r1, r2, r3),
            8 => return fold_banked_quad_disjoint::<8>(w.try_into().unwrap(), r0, r1, r2, r3),
            4 => return fold_banked_quad_disjoint::<4>(w.try_into().unwrap(), r0, r1, r2, r3),
            2 => return fold_banked_quad_disjoint::<2>(w.try_into().unwrap(), r0, r1, r2, r3),
            _ => {}
        }
    }
    let [d0, d1] = row_dot_pair(r0, r1, w);
    let [d2, d3] = row_dot_pair(r2, r3, w);
    [d0, d1, d2, d3]
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

/// Kill switch for prepared NEON constant multiplications in C2 dual helpers
/// (`invert_local_staged`, `Dual::message`, `Dual::scale`): setting
/// `FLOCK_NO_C2_DUAL_PREP` restores the incumbent scalar `F128 * F128` loops.
/// Read once per process.
#[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
pub const ENV_NO_C2_DUAL_PREP: &str = "FLOCK_NO_C2_DUAL_PREP";

#[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
#[inline]
fn c2_dual_prep_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(ENV_NO_C2_DUAL_PREP).is_none())
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
    #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
    if c2_dual_prep_enabled() {
        use core::arch::aarch64::{uint64x2_t, veorq_u64};
        use crate::field::gf2_128::aarch64::{const_prep, mul_const_prep_xor};

        // SAFETY: AES and little-endian F128 layout hold by cfg; `cache_len <= 16`.
        unsafe {
            let mut c: [uint64x2_t; 16] =
                core::array::from_fn(|i| core::mem::transmute::<F128, uint64x2_t>(cache[i]));
            for (stage, layer) in ((log_d - inv_layers)..log_d).rev().enumerate() {
                let size = 1usize << (log_d - layer);
                let half = size / 2;
                for off in (0..cache_len).step_by(size) {
                    let twiddle = ntt.twiddle(layer, (base + off) / size);
                    let prep = const_prep(core::mem::transmute::<F128, uint64x2_t>(twiddle));
                    for j in 0..half {
                        let y0 = c[off + j];
                        let v = veorq_u64(y0, c[off + half + j]);
                        c[off + j] = mul_const_prep_xor(prep, v, y0);
                        c[off + half + j] = v;
                    }
                }
                let len = 2usize << stage;
                let start = qoff & !(len - 1);
                for j in 0..len {
                    stages[len - 2 + j] = core::mem::transmute::<uint64x2_t, F128>(c[start + j]);
                }
            }
            for i in 0..cache_len {
                cache[i] = core::mem::transmute::<uint64x2_t, F128>(c[i]);
            }
        }
        return stages;
    }
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
            for j in (0..cache_len).step_by(4) {
                [cache[j], cache[j + 1], cache[j + 2], cache[j + 3]] = row_dot_quad(
                    row(base + j),
                    row(base + j + 1),
                    row(base + j + 2),
                    row(base + j + 3),
                    &lane_weights,
                );
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
        let par_floor = if op_dual_par_env_on() {
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
                    for j in (0..cache_len).step_by(4) {
                        [cache[j], cache[j + 1], cache[j + 2], cache[j + 3]] = row_dot_quad(
                            row(base + j),
                            row(base + j + 1),
                            row(base + j + 2),
                            row(base + j + 3),
                            &lane_weights,
                        );
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
                for j in (0..cache_len).step_by(4) {
                    [cache[j], cache[j + 1], cache[j + 2], cache[j + 3]] = row_dot_quad(
                        row(base + j),
                        row(base + j + 1),
                        row(base + j + 2),
                        row(base + j + 3),
                        &lane_weights,
                    );
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
            #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
            let r_prev_prep = if c2_dual_prep_enabled() {
                use core::arch::aarch64::uint64x2_t;
                use crate::field::gf2_128::aarch64::const_prep;
                // SAFETY: AES and little-endian F128 layout hold by cfg.
                Some(unsafe { const_prep(core::mem::transmute::<F128, uint64x2_t>(r_prev)) })
            } else {
                None
            };
            for (((stage, qscales), w0), w2) in self
                .stages
                .iter()
                .zip(&self.scales)
                .zip(&mut self.w0_pows)
                .zip(&mut self.w2_pows)
            {
                let local = &stage[size - 2..2 * size - 2];
                let [f0, f1] = dot2(local, &eq_weights, half);
                #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
                let c0 = if let Some(prep) = r_prev_prep {
                    use core::arch::aarch64::uint64x2_t;
                    use crate::field::gf2_128::aarch64::mul_const_prep_xor;
                    // SAFETY: AES and little-endian F128 layout hold by cfg.
                    unsafe {
                        core::mem::transmute::<uint64x2_t, F128>(mul_const_prep_xor(
                            prep,
                            core::mem::transmute::<F128, uint64x2_t>(*w2),
                            core::mem::transmute::<F128, uint64x2_t>(*w0),
                        ))
                    }
                } else {
                    *w0 + mul(r_prev, *w2)
                };
                #[cfg(not(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little")))]
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
        #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
        let beta_prep = if c2_dual_prep_enabled() {
            use core::arch::aarch64::uint64x2_t;
            use crate::field::gf2_128::aarch64::const_prep;
            // SAFETY: AES and little-endian F128 layout hold by cfg.
            Some(unsafe { const_prep(core::mem::transmute::<F128, uint64x2_t>(beta)) })
        } else {
            None
        };
        for (((w, w0), w2), qscales) in self
            .weights
            .iter_mut()
            .zip(&mut self.w0_pows)
            .zip(&mut self.w2_pows)
            .zip(&self.scales)
        {
            #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
            let nw = if let Some(prep) = beta_prep {
                use core::arch::aarch64::uint64x2_t;
                use crate::field::gf2_128::aarch64::mul_const_prep;
                // SAFETY: AES and little-endian F128 layout hold by cfg.
                unsafe {
                    core::mem::transmute::<uint64x2_t, F128>(mul_const_prep(
                        prep,
                        core::mem::transmute::<F128, uint64x2_t>(*w),
                    ))
                }
            } else {
                mul(*w, beta)
            };
            #[cfg(not(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little")))]
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

    pub(super) fn materialize_into(&mut self, rs: &[F128], dst: &mut [F128]) {
        let d = self.depth;
        assert_eq!(rs.len(), d);
        let reduced_log_d = self.log_d - d;
        assert_eq!(dst.len(), 1usize << (reduced_log_d - self.log_inv_rate));
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
        if c2_retained_transpose_enabled()
            && transpose_forward_ntt_sparse_retained_into(
                &self.ntt,
                &self.positions,
                &self.w0_pows,
                reduced_log_d,
                self.log_inv_rate,
                dst,
            )
        {
            return;
        }
        let basis = self.materialize(rs);
        assert_eq!(basis.len(), dst.len());
        for (o, &src) in dst.iter_mut().zip(&basis) {
            *o += src;
        }
        crate::scratch::give_sparse_ntt_data(basis);
    }
}

/// Largest prover state whose C2 fold skips the Rayon join (see
/// [`Pending::fold`]).
const C2_SEQ_MAX_STATE: usize = 1 << 15;

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
        // Small states (`FLOCK_NO_SMALL_SERIAL=1` kills): the fold and the
        // dual's message / materialize run one after the other on this
        // thread instead of as a Rayon join. On the ranked open these are
        // the folds of the 2^15..2^5 states, where the join's wake and wait
        // cost about as much as the dual's work (M4: the c2 folds of one
        // prove take 1.04 -> 0.98 ms with 4 threads, 1.34 -> 1.20 ms with
        // 10). Same operations on the same inputs, combined in the same
        // order, so the message, the basis and the transcript are unchanged.
        if sc.f.len() <= C2_SEQ_MAX_STATE && crate::small_serial_enabled() {
            let mut message = sc.fold(r);
            let correction = self.dual.message(&self.challenges);
            message.u_0 += correction.u_0;
            message.u_2 += correction.u_2;
            *sc.transcript.last_mut().expect("fold records a message") = message;
            if !complete {
                return (message, false);
            }
            assert!(
                sc.pending_glue.is_none()
                    && sc.pending_fold_basis.is_none()
                    && sc.pending_ood_eq.is_none()
            );
            self.dual
                .materialize_into(&self.challenges, &mut sc.combined_basis);
            return (message, true);
        }
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

    /// The completing fold when it is the last fold of the open
    /// (`FLOCK_NO_C2_TERMINAL_SKIP=1` kills). [`Self::fold`] then
    /// materializes the dual's basis and adds it into `combined_basis` for
    /// later rounds, but after the last fold the prover reads only `f` (the
    /// final `yr`) and the transcript before it drops its state, so that
    /// basis is never read. Same fold, same dual correction, same message in
    /// the same transcript slot; only the materialize and the merge go.
    pub(super) fn fold_terminal(&mut self, r: F128, sc: &mut SumcheckProver) -> SumcheckMessage {
        assert!(self.glued && self.challenges.len() + 1 == self.dual.depth);
        self.challenges.push(r);
        let mut message = sc.fold(r);
        let correction = self.dual.message(&self.challenges);
        message.u_0 += correction.u_0;
        message.u_2 += correction.u_2;
        *sc.transcript.last_mut().expect("fold records a message") = message;
        assert!(
            sc.pending_glue.is_none()
                && sc.pending_fold_basis.is_none()
                && sc.pending_ood_eq.is_none()
        );
        message
    }
}

/// Kill switch for [`Pending::fold_terminal`]: `FLOCK_NO_C2_TERMINAL_SKIP`
/// (any value) keeps the materialize and merge on the last fold. Read once
/// per process.
pub(super) fn terminal_skip_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_C2_TERMINAL_SKIP").is_none())
}

/// The `FLOCK_NO_OP_DUAL_PAR` (exact `'1'`) term of the dual-opening
/// parallel floor. The process environment does not change during a prove,
/// so the first answer is kept; `FLOCK_NO_LIGERITO_ENV_ONCE` (any value)
/// restores the per-call lookup.
#[inline]
fn op_dual_par_env_on() -> bool {
    if super::ligerito_env_once_enabled() {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var("FLOCK_NO_OP_DUAL_PAR").map_or(true, |v| v != "1")
        });
        return *ON;
    }
    std::env::var("FLOCK_NO_OP_DUAL_PAR").map_or(true, |v| v != "1")
}

pub(super) fn enabled(lazy_l1_selected: bool, disabled: Option<&std::ffi::OsStr>) -> bool {
    lazy_l1_selected && disabled != Some(std::ffi::OsStr::new("1"))
}

#[cfg(test)]
mod terminal_tests {
    use super::*;
    use crate::challenger::Challenger;

    /// The terminal fold returns the same messages, transcript, folded `f` and
    /// target as the merging fold; it only leaves `combined_basis` unmerged.
    /// Shapes include the ranked deep one (log_msg 7, rate 5, 8 lanes, depth 3).
    #[test]
    fn terminal_fold_matches_merging_fold_except_the_unread_basis() {
        for (case, (log_msg, log_rate, log_lanes, depth)) in
            [(5usize, 1usize, 0usize, 2usize), (7, 1, 2, 2), (7, 2, 2, 3), (7, 5, 3, 3), (10, 2, 3, 3)]
                .into_iter()
                .enumerate()
        {
            let mut ch = crate::challenger::RandomChallenger::new(0xC2_7E40_0000 + case as u64);
            let poly = ch.sample_f128_vec((1usize << log_msg) << log_lanes);
            let lanes = ch.sample_f128_vec(log_lanes);
            let f = partial_eval_lsb(&poly, &lanes);
            let n = 1usize << (log_msg + log_rate);
            let mut codeword = vec![F128::ZERO; n << log_lanes];
            codeword[..poly.len()].copy_from_slice(&poly);
            AdditiveNttF128::standard(log_msg + log_rate)
                .forward_transform_interleaved(&mut codeword, 1 << log_lanes);
            let queries = vec![0, 1, 2, 3, 6, 7, 8, 9, n / 2 - 1, n / 2, n - 2, n - 1];
            let alpha = ch.sample_f128_vec(ceil_log2(queries.len()));
            let initial = ch.sample_f128_vec(f.len());
            let h = f.iter().zip(&initial).fold(F128::ZERO, |s, (&a, &b)| s + a * b);
            let beta = ch.sample_f128();
            let rs = ch.sample_f128_vec(depth);
            let run = |terminal: bool| {
                let (mut sc, _) = SumcheckProver::new(f.clone(), initial.clone(), h);
                let (dual, enforced) = Dual::new_with_depth(
                    depth,
                    log_msg + log_rate,
                    log_rate,
                    &lanes,
                    &queries,
                    &alpha,
                    |q| &codeword[q << log_lanes..(q + 1) << log_lanes],
                );
                let (mut pending, _) = Pending::introduce(dual, enforced, &mut sc);
                pending.glue(beta, &mut sc);
                let mut msgs = Vec::new();
                for (s, &r) in rs.iter().enumerate() {
                    if terminal && s + 1 == depth {
                        msgs.push(pending.fold_terminal(r, &mut sc));
                    } else {
                        let (msg, complete) = pending.fold(r, &mut sc);
                        assert_eq!(complete, s + 1 == depth);
                        msgs.push(msg);
                    }
                }
                (msgs, sc.transcript.clone(), sc.f.clone(), sc.t_r)
            };
            assert_eq!(run(true), run(false), "case {case}");
        }
    }
}

#[cfg(test)]
mod shared_tail_tests {
    use super::*;

    fn rng_f128(state: &mut u64) -> F128 {
        let mut next = || {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *state ^ (*state >> 29)
        };
        F128::new(next(), next())
    }

    /// A rate-1/2 codeword of 64-lane rows whose message is zero at every odd
    /// position in lanes 57..64 (the L0 witness padding) repeats those lanes on
    /// rows 2i and 2i+1; the pair dot then shares the tail and equals the two
    /// full dots and the plain field sum. Rows that differ in the tail take the
    /// full dots.
    #[test]
    fn padded_lanes_repeat_on_adjacent_rows_and_pair_dots_match() {
        let mut state = 0xC2_7A11_0001u64;
        let log_d = 10;
        let positions = 1usize << (log_d - 1);
        let ntt = AdditiveNttF128::standard(log_d);
        let mut msg: Vec<F128> = (0..positions * 64).map(|_| rng_f128(&mut state)).collect();
        for s in (1..positions).step_by(2) {
            msg[64 * s + ROW_HEAD..64 * s + 64].fill(F128::ZERO);
        }
        let mut cw = msg.clone();
        cw.extend_from_slice(&msg);
        ntt.forward_transform_interleaved_from_layer(&mut cw, 64, 1);
        let w: Vec<F128> = (0..64).map(|_| rng_f128(&mut state)).collect();
        let oracle = |r: &[F128]| r.iter().zip(&w).fold(F128::ZERO, |acc, (x, y)| acc + *x * *y);
        for i in 0..(1usize << log_d) / 2 {
            let a = &cw[128 * i..128 * i + 64];
            let b = &cw[128 * i + 64..128 * i + 128];
            assert_eq!(a[ROW_HEAD..], b[ROW_HEAD..], "rows {} and {}", 2 * i, 2 * i + 1);
            assert_eq!(row_dot_pair(a, b, &w), [dot(a, &w), dot(b, &w)]);
            assert_eq!(row_dot_pair(a, b, &w), [oracle(a), oracle(b)]);
        }
        let a: Vec<F128> = (0..64).map(|_| rng_f128(&mut state)).collect();
        let mut b = a.clone();
        b[63] = rng_f128(&mut state);
        assert_eq!(row_dot_pair(&a, &b, &w), [oracle(&a), oracle(&b)]);
        let narrow: Vec<F128> = w[..8].to_vec();
        assert_eq!(row_dot_pair(&a[..8], &a[..8], &narrow), [dot(&a[..8], &narrow); 2]);
    }
}

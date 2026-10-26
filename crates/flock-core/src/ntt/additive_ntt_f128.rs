// Copyright 2024-2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::field::F128;

mod kernels;

fn generate_evals_from_subspace(basis: &[F128]) -> Vec<Vec<F128>> {
    let l = basis.len();
    let mut evals: Vec<Vec<F128>> = Vec::with_capacity(l);

    evals.push(basis.to_vec());

    for i in 1..l {
        let mut row = Vec::with_capacity(l - i);
        for k in 1..evals[i - 1].len() {
            let val = evals[i - 1][k] * (evals[i - 1][k] + evals[i - 1][0]);
            row.push(val);
        }
        evals.push(row);
    }

    for row in evals.iter_mut() {
        let inv = row[0].inv();
        for v in row.iter_mut() {
            *v *= inv;
        }
    }

    evals
}

#[inline]
fn span_get(basis: &[F128], idx: usize) -> F128 {
    let mut lo = 0u64;
    let mut hi = 0u64;
    let len = basis.len();
    let mut m = if len >= usize::BITS as usize {
        idx
    } else {
        idx & ((1usize << len) - 1)
    };
    let ptr = basis.as_ptr();
    while m != 0 {
        let j = m.trailing_zeros() as usize;
        m &= m - 1;

        let b = unsafe { *ptr.add(j) };
        lo ^= b.lo;
        hi ^= b.hi;
    }
    F128 { lo, hi }
}

#[derive(Clone, Debug)]
pub struct AdditiveNttF128 {
    evals: Vec<Vec<F128>>,
}

impl AdditiveNttF128 {
    pub fn new(basis: &[F128]) -> Self {
        Self {
            evals: generate_evals_from_subspace(basis),
        }
    }

    pub fn standard(dim: usize) -> Self {
        assert!(dim <= 64, "standard NTT requires dim ≤ 64");
        let basis: Vec<F128> = (0..dim).map(|i| F128::new(1u64 << i, 0)).collect();
        Self::new(&basis)
    }

    pub fn log_domain_size(&self) -> usize {
        self.evals.len()
    }

    pub fn twiddle(&self, layer: usize, block: usize) -> F128 {
        let v = &self.evals[self.log_domain_size() - layer - 1];
        span_get(&v[1..], block)
    }

    #[inline]
    pub fn twiddles_3layer(&self, layer: usize, block: usize) -> [F128; 7] {
        let ell = self.log_domain_size();
        let v0 = &self.evals[ell - layer - 1];
        let v1 = &self.evals[ell - layer - 2];
        let v2 = &self.evals[ell - layer - 3];
        let tw0 = span_get(&v0[1..], block);
        let tw1 = span_get(&v1[2..], block);
        let tw2 = tw1 + v1[1];
        let tw3 = span_get(&v2[3..], block);
        let d20 = v2[1];
        let d21 = v2[2];
        let tw4 = tw3 + d20;
        let tw5 = tw3 + d21;
        let tw6 = tw4 + d21;
        [tw0, tw1, tw2, tw3, tw4, tw5, tw6]
    }

    #[inline]
    pub fn twiddles_4layer(&self, layer: usize, block: usize) -> [F128; 15] {
        let ell = self.log_domain_size();
        let v0 = &self.evals[ell - layer - 1];
        let v1 = &self.evals[ell - layer - 2];
        let v2 = &self.evals[ell - layer - 3];
        let v3 = &self.evals[ell - layer - 4];
        let tw0 = span_get(&v0[1..], block);
        let tw1 = span_get(&v1[2..], block);
        let tw2 = tw1 + v1[1];
        let tw3 = span_get(&v2[3..], block);
        let d20 = v2[1];
        let d21 = v2[2];
        let tw4 = tw3 + d20;
        let tw5 = tw3 + d21;
        let tw6 = tw4 + d21;
        let tw7 = span_get(&v3[4..], block);
        let d30 = v3[1];
        let d31 = v3[2];
        let d32 = v3[3];
        let tw8 = tw7 + d30;
        let tw9 = tw7 + d31;
        let tw10 = tw8 + d31;
        let tw11 = tw7 + d32;
        let tw12 = tw8 + d32;
        let tw13 = tw9 + d32;
        let tw14 = tw10 + d32;
        [
            tw0, tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8, tw9, tw10, tw11, tw12, tw13, tw14,
        ]
    }

    pub fn forward_transform(&self, data: &mut [F128]) {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            self.forward_transform_batched(data);
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            self.forward_transform_scalar(data);
        }
    }

    pub fn forward_transform_interleaved(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_from_layer(data, num_ntts, 0);
    }

    pub fn forward_transform_interleaved_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        assert!(num_ntts.is_power_of_two() && num_ntts > 0);
        let n_total = data.len();
        assert_eq!(n_total % num_ntts, 0);
        let log_d = log2_pow2(n_total / num_ntts);
        assert!(log_d <= self.log_domain_size());
        assert!(start_layer <= log_d);

        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        ))]
        {
            self.forward_transform_interleaved_parallel_from_layer(data, num_ntts, start_layer);
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        )))]
        {
            self.forward_transform_interleaved_scalar_from_layer(data, num_ntts, start_layer);
        }
    }

    pub fn forward_transform_interleaved_from_replicated_msg(
        &self,
        data: &mut [F128],
        msg: &[F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        self.forward_transform_interleaved_from_replicated_msg_with_cb(
            data,
            msg,
            num_ntts,
            start_layer,
            None,
        );
    }

    pub fn forward_transform_interleaved_from_replicated_msg_on_sub_done(
        &self,
        data: &mut [F128],
        msg: &[F128],
        num_ntts: usize,
        start_layer: usize,
        on_sub_done: &(dyn Fn(core::ops::Range<usize>, &[F128]) + Sync),
    ) {
        self.forward_transform_interleaved_from_replicated_msg_with_cb(
            data,
            msg,
            num_ntts,
            start_layer,
            Some(on_sub_done),
        );
    }

    fn forward_transform_interleaved_from_replicated_msg_with_cb(
        &self,
        data: &mut [F128],
        msg: &[F128],
        num_ntts: usize,
        start_layer: usize,
        on_sub_done: Option<&(dyn Fn(core::ops::Range<usize>, &[F128]) + Sync)>,
    ) {
        assert!(num_ntts.is_power_of_two() && num_ntts > 0);
        let n_total = data.len();
        assert_eq!(n_total % num_ntts, 0);
        assert_eq!(msg.len() << start_layer, n_total);
        assert!(msg.len().is_power_of_two());
        let log_d = log2_pow2(n_total / num_ntts);
        assert!(log_d <= self.log_domain_size());
        assert!(start_layer <= log_d);

        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        ))]
        {
            const TARGET_SUBGROUP_LOG_BYTES: usize = 21;
            let log_bytes_per_position = 4 + log2_pow2(num_ntts);
            let target_log_positions =
                TARGET_SUBGROUP_LOG_BYTES.saturating_sub(log_bytes_per_position);
            let cache_n_top = log_d.saturating_sub(target_log_positions);
            const PARALLEL_FLOOR_LOG_D: usize = 12;
            const MIN_SUB_LOG: usize = 8;
            let n_top = if log_d >= PARALLEL_FLOOR_LOG_D {
                let want_subs_log = log2_pow2(rayon::current_num_threads().next_power_of_two());
                let max_n_top = log_d.saturating_sub(MIN_SUB_LOG);
                cache_n_top.max(want_subs_log.min(max_n_top))
            } else {
                cache_n_top
            };
            let fused4_ok = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));
            let n_top = if fused4_ok && n_top > start_layer {
                let rem4 = (n_top - start_layer) & 3;
                if rem4 == 0 {
                    n_top
                } else {
                    let n_top_up = n_top + (4 - rem4);
                    if n_top_up + MIN_SUB_LOG <= log_d {
                        n_top_up
                    } else if rem4 == 1 {
                        n_top - 1
                    } else {
                        n_top
                    }
                }
            } else {
                n_top
            };
            let block_size = 1usize << (log_d - start_layer);
            if fused4_ok
                && n_top > 0
                && log_d >= 8
                && start_layer + 3 < n_top
                && block_size >= 16
                && num_ntts <= 64
            {
                let num_blocks = 1usize << start_layer;
                let block_bytes = block_size * num_ntts;
                let sixteenth = block_size >> 4;
                if start_layer == 1 {
                    let tw0 = self.twiddles_4layer(1, 0);
                    let tw1 = self.twiddles_4layer(1, 1);
                    butterfly_interleaved_fused_4layer_dual_par_rows_from_src(
                        data,
                        msg,
                        block_bytes,
                        &tw0,
                        &tw1,
                        sixteenth,
                        num_ntts,
                    );
                } else {
                    for block in 0..num_blocks {
                        let tw = self.twiddles_4layer(start_layer, block);
                        let start = block * block_bytes;
                        butterfly_interleaved_fused_4layer_par_rows_from_replicated_msg(
                            &mut data[start..start + block_bytes],
                            msg,
                            start,
                            &tw,
                            sixteenth,
                            num_ntts,
                        );
                    }
                }
                self.forward_transform_interleaved_parallel_from_layer_with_n_top(
                    data,
                    num_ntts,
                    start_layer + 4,
                    n_top,
                    on_sub_done,
                );
                return;
            }
        }

        crate::pcs::commit::replicate_message_fill(data, msg);
        self.forward_transform_interleaved_from_layer(data, num_ntts, start_layer);
        if let Some(cb) = on_sub_done {
            cb(0..(n_total / num_ntts), data);
        }
    }

    pub fn forward_transform_interleaved_scalar(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_scalar_from_layer(data, num_ntts, 0);
    }

    pub fn forward_transform_interleaved_scalar_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        let n_total = data.len();
        let log_d = log2_pow2(n_total / num_ntts);

        for layer in start_layer..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;
            let block_size_bytes = block_size * num_ntts;
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block * block_size_bytes;

                for row in 0..block_size_half {
                    let off_top = block_start + row * num_ntts;
                    let off_bot = off_top + block_size_half * num_ntts;
                    for lane in 0..num_ntts {
                        let v = data[off_bot + lane];
                        let new_u = data[off_top + lane] + v * twiddle;
                        data[off_top + lane] = new_u;
                        data[off_bot + lane] = v + new_u;
                    }
                }
            }
        }
    }

    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    pub fn forward_transform_interleaved_parallel(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_parallel_from_layer(data, num_ntts, 0);
    }

    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    pub fn forward_transform_interleaved_parallel_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        let n_total = data.len();
        let log_d = log2_pow2(n_total / num_ntts);

        const TARGET_SUBGROUP_LOG_BYTES: usize = 21;
        let log_bytes_per_position = 4 + log2_pow2(num_ntts);
        let target_log_positions = TARGET_SUBGROUP_LOG_BYTES.saturating_sub(log_bytes_per_position);
        let cache_n_top = log_d.saturating_sub(target_log_positions);

        const PARALLEL_FLOOR_LOG_D: usize = 12;
        const MIN_SUB_LOG: usize = 8;
        let n_top = if log_d >= PARALLEL_FLOOR_LOG_D {
            let want_subs_log = log2_pow2(rayon::current_num_threads().next_power_of_two());
            let max_n_top = log_d.saturating_sub(MIN_SUB_LOG);
            cache_n_top.max(want_subs_log.min(max_n_top))
        } else {
            cache_n_top
        };
        let fused4_ok = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));
        let n_top = if fused4_ok && n_top > start_layer {
            let rem4 = (n_top - start_layer) & 3;
            if rem4 == 0 {
                n_top
            } else {
                let n_top_up = n_top + (4 - rem4);
                if n_top_up + MIN_SUB_LOG <= log_d {
                    n_top_up
                } else if rem4 == 1 {
                    n_top - 1
                } else {
                    n_top
                }
            }
        } else {
            n_top
        };
        self.forward_transform_interleaved_parallel_from_layer_with_n_top(
            data,
            num_ntts,
            start_layer,
            n_top,
            None,
        );
    }

    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    fn forward_sub_depth_first(
        &self,
        blk: &mut [F128],
        layer: usize,
        log_d: usize,
        global_block: usize,
        num_ntts: usize,
        on_sub_done: Option<&(dyn Fn(core::ops::Range<usize>, &[F128]) + Sync)>,
    ) -> bool {
        if layer >= log_d {
            return false;
        }
        let rem = log_d - layer;
        let block_size = 1usize << rem;
        let fused4_ok = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));

        if fused4_ok && rem >= 4 && block_size >= 16 {
            let sixteenth = block_size >> 4;
            let tw = self.twiddles_4layer(layer, global_block);
            let base = blk.as_mut_ptr();
            unsafe {
                kernels::butterfly_fused_4layer_rows(base, sixteenth, num_ntts, 0, sixteenth, &tw);
            }
            if layer + 4 < log_d {
                let child_len = sixteenth * num_ntts;
                if rem == 7 {
                    let ell = self.log_domain_size();
                    let layer3 = layer + 4;
                    let v0 = &self.evals[ell - layer3 - 1];
                    let v1 = &self.evals[ell - layer3 - 2];
                    let v2 = &self.evals[ell - layer3 - 3];
                    let mut tw0 = span_get(&v0[5..], global_block);
                    let mut tw1 = span_get(&v1[6..], global_block);
                    let mut tw2 = tw1 + v1[1];
                    let mut tw3 = span_get(&v2[7..], global_block);
                    let d20 = v2[1];
                    let d21 = v2[2];
                    let mut tw4 = tw3 + d20;
                    let mut tw5 = tw3 + d21;
                    let mut tw6 = tw4 + d21;
                    for g in 0..16usize {
                        let c = g ^ (g >> 1);
                        let tw3_leaf = [tw0, tw1, tw2, tw3, tw4, tw5, tw6];
                        let child_base = unsafe { base.add(c * child_len) };
                        unsafe {
                            kernels::butterfly_fused_3layer_row(
                                child_base, 1, num_ntts, 0, &tw3_leaf,
                            );
                        }
                        if g < 15 {
                            let bit = (g + 1).trailing_zeros() as usize;
                            let dv0 = v0[1 + bit];
                            let dv1 = v1[2 + bit];
                            let dv2 = v2[3 + bit];
                            tw0 += dv0;
                            tw1 += dv1;
                            tw2 += dv1;
                            tw3 += dv2;
                            tw4 += dv2;
                            tw5 += dv2;
                            tw6 += dv2;
                        }
                    }
                    if let Some(cb) = on_sub_done {
                        let lo = global_block * block_size;
                        cb(lo..lo + block_size, blk);
                        return true;
                    }
                    return false;
                } else {
                    let mut handled = false;
                    for c in 0..16 {
                        handled |= self.forward_sub_depth_first(
                            &mut blk[c * child_len..(c + 1) * child_len],
                            layer + 4,
                            log_d,
                            (global_block << 4) | c,
                            num_ntts,
                            on_sub_done,
                        );
                    }
                    return handled;
                }
            }
            false
        } else if fused4_ok && rem == 3 && block_size >= 8 {
            let eighth = block_size >> 3;
            let tw = self.twiddles_3layer(layer, global_block);
            let base = blk.as_mut_ptr();
            for r in 0..eighth {
                unsafe {
                    kernels::butterfly_fused_3layer_row(base, eighth, num_ntts, r, &tw);
                }
            }
            false
        } else if rem >= 2 && block_size >= 4 {
            let quarter = block_size >> 2;
            let stride = quarter * num_ntts;
            let t_outer = self.twiddle(layer, global_block);
            let t_inner_a = self.twiddle(layer + 1, 2 * global_block);
            let t_inner_b = self.twiddle(layer + 1, 2 * global_block + 1);
            let (top_half, bot_half) = blk.split_at_mut(2 * stride);
            let (q1, q2) = top_half.split_at_mut(stride);
            let (q3, q4) = bot_half.split_at_mut(stride);
            for r in 0..quarter {
                let off = r * num_ntts;
                kernels::butterfly_fused_2layer(
                    &mut q1[off..off + num_ntts],
                    &mut q2[off..off + num_ntts],
                    &mut q3[off..off + num_ntts],
                    &mut q4[off..off + num_ntts],
                    t_outer,
                    t_inner_a,
                    t_inner_b,
                );
            }
            if layer + 2 < log_d {
                let mut handled = false;
                for c in 0..4 {
                    handled |= self.forward_sub_depth_first(
                        &mut blk[c * stride..(c + 1) * stride],
                        layer + 2,
                        log_d,
                        (global_block << 2) | c,
                        num_ntts,
                        on_sub_done,
                    );
                }
                return handled;
            }
            false
        } else {
            let block_size_half = block_size >> 1;
            let twiddle = self.twiddle(layer, global_block);
            butterfly_interleaved_block(blk, twiddle, block_size_half, num_ntts);
            if layer + 1 < log_d {
                let half_len = block_size_half * num_ntts;
                let h0 = self.forward_sub_depth_first(
                    &mut blk[..half_len],
                    layer + 1,
                    log_d,
                    global_block << 1,
                    num_ntts,
                    on_sub_done,
                );
                let h1 = self.forward_sub_depth_first(
                    &mut blk[half_len..],
                    layer + 1,
                    log_d,
                    (global_block << 1) | 1,
                    num_ntts,
                    on_sub_done,
                );
                return h0 || h1;
            }
            false
        }
    }

    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    fn forward_transform_interleaved_parallel_from_layer_with_n_top(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        n_top: usize,
        on_sub_done: Option<&(dyn Fn(core::ops::Range<usize>, &[F128]) + Sync)>,
    ) {
        use rayon::prelude::*;
        let n_total = data.len();
        let num_leaves = n_total / num_ntts;
        let log_d = log2_pow2(num_leaves);
        let fused4_ok = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));
        if n_top == 0 || log_d < 8 {
            let mut handled = false;
            if start_layer < log_d {
                let num_blocks = 1usize << start_layer;
                let block_bytes = (1usize << (log_d - start_layer)) * num_ntts;
                for block in 0..num_blocks {
                    handled |= self.forward_sub_depth_first(
                        &mut data[block * block_bytes..(block + 1) * block_bytes],
                        start_layer,
                        log_d,
                        block,
                        num_ntts,
                        on_sub_done,
                    );
                }
            }
            if !handled && let Some(cb) = on_sub_done {
                cb(0..num_leaves, data);
            }
            return;
        }
        let mut layer = start_layer.min(n_top);
        while layer < n_top {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_bytes = block_size * num_ntts;

            if fused4_ok && layer + 3 < n_top && block_size >= 16 {
                let sixteenth = block_size >> 4;
                if num_blocks >= 8 {
                    data.par_chunks_mut(block_bytes)
                        .enumerate()
                        .for_each(|(block, blk)| {
                            let tw = self.twiddles_4layer(layer, block);
                            let base = blk.as_mut_ptr();
                            unsafe {
                                kernels::butterfly_fused_4layer_rows(
                                    base, sixteenth, num_ntts, 0, sixteenth, &tw,
                                );
                            }
                        });
                } else {
                    for block in 0..num_blocks {
                        let tw = self.twiddles_4layer(layer, block);
                        let start = block * block_bytes;
                        butterfly_interleaved_fused_4layer_par_rows(
                            &mut data[start..start + block_bytes],
                            &tw,
                            sixteenth,
                            num_ntts,
                        );
                    }
                }
                layer += 4;
            } else if layer + 1 < n_top && block_size >= 4 {
                let quarter = block_size >> 2;
                for block in 0..num_blocks {
                    let t_outer = self.twiddle(layer, block);
                    let t_inner_a = self.twiddle(layer + 1, 2 * block);
                    let t_inner_b = self.twiddle(layer + 1, 2 * block + 1);
                    let start = block * block_bytes;
                    butterfly_interleaved_fused_2layer_par_rows(
                        &mut data[start..start + block_bytes],
                        t_outer,
                        t_inner_a,
                        t_inner_b,
                        quarter,
                        num_ntts,
                    );
                }
                layer += 2;
            } else {
                let block_size_half = block_size >> 1;
                for block in 0..num_blocks {
                    let t = self.twiddle(layer, block);
                    let start = block * block_bytes;
                    butterfly_interleaved_block_par_rows(
                        &mut data[start..start + block_bytes],
                        t,
                        block_size_half,
                        num_ntts,
                    );
                }
                layer += 1;
            }
        }

        let sub_size_positions = 1usize << (log_d - n_top);
        let sub_bytes = sub_size_positions * num_ntts;

        data.par_chunks_mut(sub_bytes)
            .enumerate()
            .for_each(|(sub_idx, sub_data)| {
                let mut handled = false;
                if start_layer <= n_top {
                    handled = self.forward_sub_depth_first(
                        sub_data,
                        n_top,
                        log_d,
                        sub_idx,
                        num_ntts,
                        on_sub_done,
                    );
                } else if start_layer < log_d {
                    let extra = start_layer - n_top;
                    let num_blocks_in_sub = 1usize << extra;
                    let blk_len = (1usize << (log_d - start_layer)) * num_ntts;
                    for b in 0..num_blocks_in_sub {
                        handled |= self.forward_sub_depth_first(
                            &mut sub_data[b * blk_len..(b + 1) * blk_len],
                            start_layer,
                            log_d,
                            (sub_idx << extra) | b,
                            num_ntts,
                            on_sub_done,
                        );
                    }
                }
                if !handled && let Some(cb) = on_sub_done {
                    let lo = sub_idx * sub_size_positions;
                    cb(lo..lo + sub_size_positions, sub_data);
                }
            });
    }

    pub fn forward_transform_scalar(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in 0..log_d {
            let num_blocks = 1usize << layer;
            let block_size_half = 1usize << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = idx0 | block_size_half;

                    let v = data[idx1];
                    let new_u = data[idx0] + v * twiddle;
                    data[idx0] = new_u;
                    data[idx1] = v + new_u;
                }
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_neon(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in 0..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;

            unsafe {
                if block_size_half >= 2 {
                    for block in 0..num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        let chunk = &mut data[block_start..block_start + block_size];
                        kernels::butterfly_neon_block(chunk, twiddle, block_size_half);
                    }
                } else {
                    debug_assert_eq!(block_size_half, 1);
                    let mut block = 0;
                    while block + 1 < num_blocks {
                        let t_a = self.twiddle(layer, block);
                        let t_b = self.twiddle(layer, block + 1);
                        kernels::butterfly_neon_block_pair(data, block * 2, t_a, t_b);
                        block += 2;
                    }

                    while block < num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let idx0 = block * 2;
                        let idx1 = idx0 + 1;
                        let v = data[idx1];
                        let new_u = data[idx0] + v * twiddle;
                        data[idx0] = new_u;
                        data[idx1] = v + new_u;
                        block += 1;
                    }
                }
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_parallel(&self, data: &mut [F128]) {
        use rayon::prelude::*;
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        const PARALLEL_THRESHOLD_LOG: usize = 14;
        if log_d <= PARALLEL_THRESHOLD_LOG {
            self.forward_transform_neon(data);
            return;
        }

        for layer in 0..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;

            if num_blocks >= 4 && block_size_half >= 2 {
                let twiddles: Vec<F128> = (0..num_blocks).map(|b| self.twiddle(layer, b)).collect();
                data.par_chunks_mut(block_size)
                    .zip(twiddles.par_iter())
                    .for_each(|(chunk, &twiddle)| {
                        unsafe { kernels::butterfly_neon_block(chunk, twiddle, block_size_half) };
                    });
            } else if block_size_half >= 2 {
                unsafe {
                    for block in 0..num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        kernels::butterfly_neon_block(
                            &mut data[block_start..block_start + block_size],
                            twiddle,
                            block_size_half,
                        );
                    }
                }
            } else {
                debug_assert_eq!(block_size_half, 1);
                if num_blocks >= 2 {
                    let twiddles: Vec<F128> =
                        (0..num_blocks).map(|b| self.twiddle(layer, b)).collect();
                    data.par_chunks_mut(4).zip(twiddles.par_chunks(2)).for_each(
                        |(chunk, twiddle_pair)| {
                            unsafe {
                                kernels::butterfly_neon_block_pair_chunk(
                                    chunk,
                                    twiddle_pair[0],
                                    twiddle_pair[1],
                                )
                            };
                        },
                    );
                } else {
                    let twiddle = self.twiddle(layer, 0);
                    let v = data[1];
                    let new_u = data[0] + v * twiddle;
                    data[0] = new_u;
                    data[1] = v + new_u;
                }
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_batched(&self, data: &mut [F128]) {
        use rayon::prelude::*;
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        const TARGET_SUB_NTT_LOG: usize = 17;
        if log_d <= TARGET_SUB_NTT_LOG {
            self.forward_transform_parallel(data);
            return;
        }
        let n_top = log_d - TARGET_SUB_NTT_LOG;
        let sub_ntt_size = 1usize << (log_d - n_top);

        for layer in 0..n_top {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;

            if num_blocks >= 4 {
                let twiddles: Vec<F128> = (0..num_blocks).map(|b| self.twiddle(layer, b)).collect();
                data.par_chunks_mut(block_size)
                    .zip(twiddles.par_iter())
                    .for_each(|(chunk, &t)| {
                        unsafe { kernels::butterfly_neon_block(chunk, t, block_size_half) };
                    });
            } else {
                unsafe {
                    for block in 0..num_blocks {
                        let t = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        kernels::butterfly_neon_block(
                            &mut data[block_start..block_start + block_size],
                            t,
                            block_size_half,
                        );
                    }
                }
            }
        }

        data.par_chunks_mut(sub_ntt_size)
            .enumerate()
            .for_each(|(sub_idx, sub_data)| {
                for layer in n_top..log_d {
                    let layer_in_sub = layer - n_top;
                    let num_blocks_in_sub = 1usize << layer_in_sub;
                    let block_size = 1usize << (log_d - layer);
                    let block_size_half = block_size >> 1;

                    for block_in_sub in 0..num_blocks_in_sub {
                        let global_block = sub_idx * num_blocks_in_sub + block_in_sub;
                        let twiddle = self.twiddle(layer, global_block);
                        let block_start = block_in_sub * block_size;
                        let block = &mut sub_data[block_start..block_start + block_size];
                        if block_size_half >= 2 {
                            unsafe {
                                kernels::butterfly_neon_block(block, twiddle, block_size_half)
                            };
                        } else {
                            let v = block[1];
                            let new_u = block[0] + v * twiddle;
                            block[0] = new_u;
                            block[1] = v + new_u;
                        }
                    }
                }
            });
    }

    pub fn inverse_transform(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in (0..log_d).rev() {
            let num_blocks = 1usize << layer;
            let block_size_half = 1usize << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = idx0 | block_size_half;

                    let u = data[idx0];
                    let new_v = data[idx1] + u;
                    data[idx1] = new_v;
                    data[idx0] = u + new_v * twiddle;
                }
            }
        }
    }
}

#[inline]
fn butterfly_interleaved_block_par_rows(
    block: &mut [F128],
    twiddle: F128,
    block_size_half: usize,
    num_ntts: usize,
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 512;
    if block_size_half < PARALLEL_ROW_THRESHOLD {
        butterfly_interleaved_block(block, twiddle, block_size_half, num_ntts);
        return;
    }
    let half_offset = block_size_half * num_ntts;
    let (top, bot) = block.split_at_mut(half_offset);
    top.par_chunks_mut(num_ntts)
        .zip(bot.par_chunks_mut(num_ntts))
        .for_each(|(top_row, bot_row)| {
            kernels::butterfly_row_pair(top_row, bot_row, twiddle);
        });
}

#[inline]
fn butterfly_interleaved_fused_2layer_par_rows(
    block: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    quarter: usize,
    num_ntts: usize,
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    let stride = quarter * num_ntts;
    debug_assert_eq!(block.len(), 4 * stride);

    let do_one = |row_a: &mut [F128],
                  row_b: &mut [F128],
                  row_c: &mut [F128],
                  row_d: &mut [F128]| {
        kernels::butterfly_fused_2layer(row_a, row_b, row_c, row_d, t_outer, t_inner_a, t_inner_b);
    };

    let (top_half, bot_half) = block.split_at_mut(2 * stride);
    let (q1, q2) = top_half.split_at_mut(stride);
    let (q3, q4) = bot_half.split_at_mut(stride);

    if quarter < PARALLEL_ROW_THRESHOLD {
        for r in 0..quarter {
            let off = r * num_ntts;
            let (q1r, q1_rest) = q1[off..].split_at_mut(num_ntts);
            let _ = q1_rest;
            let (q2r, _) = q2[off..].split_at_mut(num_ntts);
            let (q3r, _) = q3[off..].split_at_mut(num_ntts);
            let (q4r, _) = q4[off..].split_at_mut(num_ntts);
            do_one(q1r, q2r, q3r, q4r);
        }
    } else {
        q1.par_chunks_mut(num_ntts)
            .zip(q2.par_chunks_mut(num_ntts))
            .zip(q3.par_chunks_mut(num_ntts))
            .zip(q4.par_chunks_mut(num_ntts))
            .for_each(|(((row_a, row_b), row_c), row_d)| {
                do_one(row_a, row_b, row_c, row_d);
            });
    }
}

#[inline]
fn butterfly_interleaved_block(
    block: &mut [F128],
    twiddle: F128,
    block_size_half: usize,
    num_ntts: usize,
) {
    let off_bot = block_size_half * num_ntts;
    let (top, bot) = block.split_at_mut(off_bot);
    for r in 0..block_size_half {
        let o = r * num_ntts;
        kernels::butterfly_row_pair(
            &mut top[o..o + num_ntts],
            &mut bot[o..o + num_ntts],
            twiddle,
        );
    }
}

#[inline]
fn butterfly_interleaved_fused_4layer_par_rows(
    block: &mut [F128],
    t: &[F128; 15],
    sixteenth: usize,
    num_ntts: usize,
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    debug_assert_eq!(block.len(), 16 * sixteenth * num_ntts);

    let base = block.as_mut_ptr() as usize;
    if sixteenth < PARALLEL_ROW_THRESHOLD {
        unsafe {
            kernels::butterfly_fused_4layer_rows(
                base as *mut F128,
                sixteenth,
                num_ntts,
                0,
                sixteenth,
                t,
            )
        };
    } else {
        const ROW_CHUNK: usize = 64;
        let num_chunks = sixteenth.div_ceil(ROW_CHUNK);
        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let r_start = chunk_idx * ROW_CHUNK;
            let r_count = (sixteenth - r_start).min(ROW_CHUNK);

            unsafe {
                kernels::butterfly_fused_4layer_rows(
                    base as *mut F128,
                    sixteenth,
                    num_ntts,
                    r_start,
                    r_count,
                    t,
                )
            };
        });
    }
}

#[inline]
fn butterfly_interleaved_fused_4layer_dual_par_rows_from_src(
    data: &mut [F128],
    msg: &[F128],
    block_bytes: usize,
    tw0: &[F128; 15],
    tw1: &[F128; 15],
    sixteenth: usize,
    num_ntts: usize,
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    debug_assert_eq!(block_bytes, 16 * sixteenth * num_ntts);
    debug_assert_eq!(msg.len(), block_bytes);
    debug_assert_eq!(data.len(), 2 * block_bytes);
    let dst0 = data.as_mut_ptr() as usize;
    let dst1 = unsafe { data.as_mut_ptr().add(block_bytes) } as usize;
    let src = msg.as_ptr() as usize;
    if sixteenth < PARALLEL_ROW_THRESHOLD {
        unsafe {
            kernels::butterfly_fused_4layer_dual_from_src_rows(
                src as *const F128,
                dst0 as *mut F128,
                dst1 as *mut F128,
                sixteenth,
                num_ntts,
                0,
                sixteenth,
                tw0,
                tw1,
            );
        }
    } else {
        const ROW_CHUNK: usize = 64;
        let num_chunks = sixteenth.div_ceil(ROW_CHUNK);
        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let r_start = chunk_idx * ROW_CHUNK;
            let r_count = (sixteenth - r_start).min(ROW_CHUNK);
            unsafe {
                kernels::butterfly_fused_4layer_dual_from_src_rows(
                    src as *const F128,
                    dst0 as *mut F128,
                    dst1 as *mut F128,
                    sixteenth,
                    num_ntts,
                    r_start,
                    r_count,
                    tw0,
                    tw1,
                );
            }
        });
    }
}

#[inline]
fn butterfly_interleaved_fused_4layer_par_rows_from_replicated_msg(
    block: &mut [F128],
    msg: &[F128],
    dst_off: usize,
    t: &[F128; 15],
    sixteenth: usize,
    num_ntts: usize,
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    debug_assert_eq!(block.len(), 16 * sixteenth * num_ntts);
    let base = block.as_mut_ptr() as usize;
    let msg_ptr = msg.as_ptr() as usize;
    let msg_len = msg.len();
    if sixteenth < PARALLEL_ROW_THRESHOLD {
        for r in 0..sixteenth {
            unsafe {
                kernels::butterfly_fused_4layer_row_from_replicated_msg(
                    msg_ptr as *const F128,
                    msg_len,
                    dst_off,
                    base as *mut F128,
                    sixteenth,
                    num_ntts,
                    r,
                    t,
                );
            }
        }
    } else {
        (0..sixteenth).into_par_iter().for_each(|r| unsafe {
            kernels::butterfly_fused_4layer_row_from_replicated_msg(
                msg_ptr as *const F128,
                msg_len,
                dst_off,
                base as *mut F128,
                sixteenth,
                num_ntts,
                r,
                t,
            );
        });
    }
}

#[inline]
fn log2_pow2(n: usize) -> usize {
    assert!(
        n.is_power_of_two() && n > 0,
        "length must be a positive power of 2"
    );
    n.trailing_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    fn rand_vec(rng: &mut Rng, n: usize) -> Vec<F128> {
        (0..n).map(|_| rng.f128()).collect()
    }

    #[test]
    fn forward_inverse_roundtrip() {
        let mut rng = Rng::new(0xAB1);
        for log_d in [1usize, 2, 3, 4, 6, 8] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v = original.clone();
            ntt.forward_transform(&mut v);
            ntt.inverse_transform(&mut v);
            assert_eq!(v, original, "roundtrip failed at log_d={log_d}");
        }
    }

    #[test]
    fn inverse_forward_roundtrip() {
        let mut rng = Rng::new(0xAB2);
        for log_d in [1usize, 2, 3, 4, 6, 8] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v = original.clone();
            ntt.inverse_transform(&mut v);
            ntt.forward_transform(&mut v);
            assert_eq!(
                v, original,
                "inverse∘forward roundtrip failed at log_d={log_d}"
            );
        }
    }

    #[test]
    fn forward_is_linear() {
        let mut rng = Rng::new(0xAB3);
        for log_d in [1usize, 2, 3, 5] {
            let ntt = AdditiveNttF128::standard(log_d);
            let n = 1 << log_d;
            let a = rand_vec(&mut rng, n);
            let b = rand_vec(&mut rng, n);
            let ab: Vec<F128> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();

            let mut fa = a.clone();
            ntt.forward_transform(&mut fa);
            let mut fb = b.clone();
            ntt.forward_transform(&mut fb);
            let mut fab = ab.clone();
            ntt.forward_transform(&mut fab);

            for i in 0..n {
                assert_eq!(
                    fa[i] + fb[i],
                    fab[i],
                    "linearity fails at i={i}, log_d={log_d}"
                );
            }
        }
    }

    #[test]
    fn ntt_of_zero_is_zero() {
        for log_d in [1usize, 2, 3, 6] {
            let ntt = AdditiveNttF128::standard(log_d);
            let mut v = vec![F128::ZERO; 1 << log_d];
            ntt.forward_transform(&mut v);
            assert!(v.iter().all(|&x| x == F128::ZERO));
        }
    }

    #[test]
    fn twiddle_at_layer_0_uses_full_basis_minus_one() {
        let ntt = AdditiveNttF128::standard(4);
        assert_eq!(ntt.twiddle(0, 0), F128::ZERO);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_matches_scalar() {
        let mut rng = Rng::new(0xBB1);
        for log_d in 1..=10 {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v_scalar = original.clone();
            ntt.forward_transform_scalar(&mut v_scalar);
            let mut v_neon = original.clone();
            ntt.forward_transform_neon(&mut v_neon);
            assert_eq!(
                v_neon, v_scalar,
                "NEON disagrees with scalar at log_d={log_d}"
            );
        }
    }

    #[test]
    fn interleaved_matches_per_lane() {
        let mut rng = Rng::new(0xCC1);

        for log_d in [3usize, 4, 8] {
            for num_ntts in [1usize, 2, 4, 8] {
                let ntt = AdditiveNttF128::standard(log_d);
                let n_total = (1 << log_d) * num_ntts;
                let original = rand_vec(&mut rng, n_total);

                let mut v_inter = original.clone();
                ntt.forward_transform_interleaved_scalar(&mut v_inter, num_ntts);

                let mut v_ref = original.clone();
                for lane in 0..num_ntts {
                    let mut sub: Vec<F128> = (0..(1 << log_d))
                        .map(|pos| v_ref[pos * num_ntts + lane])
                        .collect();
                    ntt.forward_transform_scalar(&mut sub);
                    for pos in 0..(1 << log_d) {
                        v_ref[pos * num_ntts + lane] = sub[pos];
                    }
                }

                assert_eq!(
                    v_inter, v_ref,
                    "interleaved mismatch at log_d={log_d}, num_ntts={num_ntts}"
                );
            }
        }
    }

    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ))]
    #[test]
    fn interleaved_parallel_matches_scalar() {
        let mut rng = Rng::new(0xCC2);
        for log_d in [4usize, 10, 14, 17, 19] {
            for &num_ntts in &[2usize, 8, 32] {
                let ntt = AdditiveNttF128::standard(log_d);
                let n_total = (1 << log_d) * num_ntts;
                let original = rand_vec(&mut rng, n_total);
                let mut v_scalar = original.clone();
                ntt.forward_transform_interleaved_scalar(&mut v_scalar, num_ntts);
                let mut v_par = original.clone();
                ntt.forward_transform_interleaved_parallel(&mut v_par, num_ntts);
                assert_eq!(
                    v_par, v_scalar,
                    "interleaved parallel mismatch at log_d={log_d}, num_ntts={num_ntts}"
                );
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn batched_matches_scalar() {
        let mut rng = Rng::new(0xBB4);

        for log_d in [4usize, 8, 12, 17, 18, 19, 20] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v_scalar = original.clone();
            ntt.forward_transform_scalar(&mut v_scalar);
            let mut v_batched = original.clone();
            ntt.forward_transform_batched(&mut v_batched);
            assert_eq!(
                v_batched, v_scalar,
                "batched disagrees with scalar at log_d={log_d}"
            );
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn parallel_matches_scalar() {
        let mut rng = Rng::new(0xBB2);
        for log_d in [4usize, 8, 12, 15, 16] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v_scalar = original.clone();
            ntt.forward_transform_scalar(&mut v_scalar);
            let mut v_par = original.clone();
            ntt.forward_transform_parallel(&mut v_par);
            assert_eq!(
                v_par, v_scalar,
                "parallel disagrees with scalar at log_d={log_d}"
            );
        }
    }

    #[test]
    fn deepest_layer_twiddle_count() {
        let log_d = 4;
        let ntt = AdditiveNttF128::standard(log_d);

        for b in 0..8 {
            let _t = ntt.twiddle(log_d - 1, b);
        }
    }
}

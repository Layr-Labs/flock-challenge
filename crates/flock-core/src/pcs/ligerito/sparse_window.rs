use super::{
    AdditiveNttF128, F128, expand_singleton_into, transpose_butterfly,
    transpose_forward_ntt_window_dense_slice,
};

pub(super) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_INDUCE_SPARSE_TRIE").is_none());
    *ON
}

pub(super) fn expand(
    ntt: &AdditiveNttF128,
    log_d: usize,
    k: usize,
    w: usize,
    positions: &[usize],
    values: &[F128],
    order: &[u32],
    out: &mut [F128],
) {
    debug_assert!(k <= log_d);
    debug_assert_eq!(out.len(), 1usize << k);
    debug_assert_eq!(positions.len(), values.len());
    debug_assert!(
        order
            .windows(2)
            .all(|pair| { positions[pair[0] as usize] <= positions[pair[1] as usize] })
    );
    debug_assert!(order.iter().all(|&i| positions[i as usize] >> k == w));

    if order.is_empty() {
        out.fill(F128::ZERO);
        return;
    }

    let first = order[0] as usize;
    let last = *order.last().unwrap() as usize;
    if positions[first] == positions[last] {
        let value = order
            .iter()
            .fold(F128::ZERO, |sum, &i| sum + values[i as usize]);
        if k == 0 {
            out[0] = value;
        } else {
            expand_singleton_into(
                ntt,
                log_d,
                k,
                w,
                positions[first] & (out.len() - 1),
                value,
                out,
            );
        }
        return;
    }

    if k <= 4 && order.len() >= out.len() / 2 {
        out.fill(F128::ZERO);
        for &i in order {
            let i = i as usize;
            out[positions[i] & (out.len() - 1)] += values[i];
        }
        transpose_forward_ntt_window_dense_slice(ntt, log_d, k, w, out);
        return;
    }

    let half = out.len() / 2;
    let split_position = (w << k) + half;
    let split = order.partition_point(|&i| positions[i as usize] < split_position);
    let twiddle = ntt.twiddle(log_d - k, w);
    let (top, bot) = out.split_at_mut(half);
    if split == 0 || split == order.len() {
        let right = split == 0;
        expand(
            ntt,
            log_d,
            k - 1,
            2 * w + usize::from(right),
            positions,
            values,
            order,
            top,
        );
        let scale = twiddle + if right { F128::ONE } else { F128::ZERO };
        scale_into(bot, top, scale);
    } else {
        expand(
            ntt,
            log_d,
            k - 1,
            2 * w,
            positions,
            values,
            &order[..split],
            top,
        );
        expand(
            ntt,
            log_d,
            k - 1,
            2 * w + 1,
            positions,
            values,
            &order[split..],
            bot,
        );
        transpose_butterfly(top, bot, twiddle);
    }
}

fn scale_into(out: &mut [F128], input: &[F128], scale: F128) {
    if scale == F128::ZERO {
        out.fill(F128::ZERO);
    } else if scale == F128::ONE {
        out.copy_from_slice(input);
    } else {
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        unsafe {
            super::eq_expand_block_x4(out, input, scale);
        }
        #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
        for (dst, &src) in out.iter_mut().zip(input) {
            *dst = src * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare(log_d: usize, k: usize, w: usize, entries: &[(usize, F128)]) {
        let ntt = AdditiveNttF128::standard(log_d);
        let positions: Vec<_> = entries.iter().map(|&(p, _)| (w << k) + p).collect();
        let values: Vec<_> = entries.iter().map(|&(_, v)| v).collect();
        let mut order: Vec<_> = (0..entries.len() as u32).collect();
        order.sort_by_key(|&i| positions[i as usize]);
        let mut want = vec![F128::ZERO; 1usize << k];
        for &(p, v) in entries {
            want[p] += v;
        }
        if k > 0 {
            transpose_forward_ntt_window_dense_slice(&ntt, log_d, k, w, &mut want);
        }
        let mut got = vec![F128::new(u64::MAX, 0xabad_1dea); 1usize << k];
        expand(&ntt, log_d, k, w, &positions, &values, &order, &mut got);
        assert_eq!(
            got,
            want,
            "log_d={log_d}, k={k}, w={w}, entries={}",
            entries.len()
        );
    }

    #[test]
    fn sparse_trie_empty_singleton_and_duplicate_windows() {
        for k in 0..=9 {
            let n = 1usize << k;
            for w in [0usize, 1, 3] {
                compare(k + 2, k, w, &[]);
                for p in [0, n / 2, n - 1] {
                    for v in [F128::ZERO, F128::ONE, F128::new(7, u64::MAX)] {
                        compare(k + 2, k, w, &[(p, v)]);
                        compare(k + 2, k, w, &[(p, v), (p, v)]);
                        compare(k + 2, k, w, &[(p, v), (p, F128::ONE), (p, v)]);
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_trie_all_small_pairs() {
        for k in 1..=5 {
            for a in 0..1usize << k {
                for b in 0..1usize << k {
                    compare(
                        k + 1,
                        k,
                        1,
                        &[
                            (a, F128::new(0x1234, u64::MAX)),
                            (b, F128::new(u64::MAX, 0x5678)),
                        ],
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_trie_mixed_density_and_cosets() {
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for k in 1..=12 {
            let n = 1usize << k;
            for count in [2, 3, 6, 14, 32, n / 2, n] {
                let entries: Vec<_> = (0..count)
                    .map(|_| ((random() as usize) & (n - 1), F128::new(random(), random())))
                    .collect();
                for w in [0usize, 1, 5] {
                    compare(k + 3, k, w, &entries);
                }
                let entries: Vec<_> = entries.into_iter().rev().map(|(p, v)| (p / 4, v)).collect();
                compare(k + 3, k, 7, &entries);
            }
        }
    }

    #[test]
    fn sparse_trie_all_occupied_leaves() {
        for k in 1..=9 {
            let entries: Vec<_> = (0..1usize << k)
                .rev()
                .map(|p| (p, F128::new(p as u64, !(p as u64))))
                .collect();
            compare(k + 1, k, 1, &entries);
        }
    }

    #[test]
    fn sparse_trie_complete_transform_and_dead_outputs() {
        use super::super::transpose_forward_ntt_sparse_truncated;
        use rayon::ThreadPoolBuilder;

        for threads in [1, 4] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for (log_d, _k, queries) in
                    [(12, 6, 71), (14, 10, 106), (16, 12, 218), (20, 12, 218)]
                {
                    let n = 1usize << log_d;
                    let ntt = AdditiveNttF128::standard(log_d);
                    let positions: Vec<_> = (0..queries)
                        .rev()
                        .map(|i| ((i / 2) * 7919 + 17) & (n - 1))
                        .collect();
                    let values: Vec<_> = (0..queries)
                        .map(|i| F128::new((i as u64 + 1) * 0x1234, !(i as u64)))
                        .collect();
                    for dead in [0, 1, 3, 5] {
                        let got = transpose_forward_ntt_sparse_truncated(
                            &ntt, &positions, &values, log_d, dead,
                        );
                        let want = transpose_forward_ntt_sparse_truncated(
                            &ntt, &positions, &values, log_d, 0,
                        );
                        let kept = n >> dead;
                        assert_eq!(
                            &got[..kept],
                            &want[..kept],
                            "threads={threads}, log_d={log_d}, dead={dead}"
                        );
                    }
                }
            });
        }
    }

    fn product_bound(k: usize, positions: &[usize]) -> usize {
        let n = 1usize << k;
        if positions.is_empty() {
            return 0;
        }
        if positions[0] == *positions.last().unwrap() {
            return n - 1;
        }
        if k <= 4 && positions.len() >= n / 2 {
            return k * n / 2;
        }
        let split = positions.partition_point(|&p| p < n / 2);
        let right: Vec<_> = positions[split..].iter().map(|&p| p - n / 2).collect();
        n / 2 + product_bound(k - 1, &positions[..split]) + product_bound(k - 1, &right)
    }

    #[test]
    fn sparse_trie_shares_ancestor_products() {
        for k in 5..=12 {
            let n = 1usize << k;
            for count in [2, 3, 6, 14, 32] {
                let positions: Vec<_> = (0..count).map(|i| i * n / count).collect();
                let shared = product_bound(k, &positions);
                let dense = k * n / 2;
                let separate = count * (n - 1);
                assert!(shared <= dense);
                assert!(shared < separate);
            }
        }
        let positions: Vec<_> = (0..14).map(|i| i * 4096 / 14).collect();
        let shared = product_bound(12, &positions);
        assert!(shared < 24576 / 2);
        eprintln!(
            "14 queries / 4096 slots: shared upper bound {shared}, dense 24576, separate 57330 products"
        );
    }
}

use super::*;

enum MergeBasis {
    Retained(open_restore::RetainedBasis),
    Dense(Vec<F128>),
}

impl Pending {
    pub(super) fn fold_complete_restored(
        &mut self,
        r: F128,
        sc: &mut SumcheckProver,
    ) -> (SumcheckMessage, bool) {
        let dual = &mut self.dual;
        let challenges = &self.challenges;
        let (mut message, (correction, basis)) = rayon::join(
            || sc.fold(r),
            || {
                let correction = dual.message(challenges);
                dual.prepare_materialization(challenges);
                let retained = if c2_retained_transpose_enabled() {
                    prepare_sparse_retained(
                        &dual.ntt,
                        &dual.positions,
                        &dual.w0_pows,
                        dual.log_d - dual.depth,
                        dual.log_inv_rate,
                        c2_lean_windows_enabled(),
                    )
                } else {
                    None
                };
                let basis = match retained {
                    Some(basis) => MergeBasis::Retained(basis),
                    None => MergeBasis::Dense(dual.materialize_prepared()),
                };
                (correction, basis)
            },
        );
        message.u_0 += correction.u_0;
        message.u_2 += correction.u_2;
        *sc.transcript.last_mut().expect("fold records a message") = message;
        match basis {
            MergeBasis::Retained(basis) => basis.add_into(&dual.ntt, &mut sc.combined_basis),
            MergeBasis::Dense(basis) => {
                assert_eq!(basis.len(), sc.combined_basis.len());
                if basis.len() >= 16_384 {
                    use rayon::prelude::*;
                    sc.combined_basis.par_chunks_mut(4096)
                        .zip(basis.par_chunks(4096))
                        .for_each(|(dst, src)| {
                            for (d, &s) in dst.iter_mut().zip(src) {
                                *d += s;
                            }
                        });
                } else {
                    for (d, &s) in sc.combined_basis.iter_mut().zip(&basis) {
                        *d += s;
                    }
                }
                crate::scratch::give_sparse_ntt_data(basis);
            }
        }
        (message, true)
    }
}

struct QuerySetup {
    query: usize,
    stage: [F128; 30],
    scales: [F128; 4],
    value: F128,
    row: Vec<F128>,
}

impl Dual {
    /// Samples on the caller; workers see only already accepted positions.
    pub(in crate::pcs::ligerito) fn sample_with_setup<'a, Ch: Challenger>(
        depth: usize,
        log_d: usize,
        log_inv_rate: usize,
        lane_challenges: &[F128],
        block_len: usize,
        count: usize,
        challenger: &mut Ch,
        tree: &[Hash],
        row: impl Fn(usize) -> &'a [F128] + Sync,
    ) -> (Vec<usize>, Vec<F128>, (Self, F128), (Vec<Hash>, Vec<Vec<F128>>)) {
        assert!((2..=3).contains(&depth));
        assert!(log_inv_rate + depth < log_d);
        assert_eq!(block_len, 1usize << log_d);
        let lane_weights = build_eq_table(lane_challenges);
        let ntt = AdditiveNttF128::standard(log_d);
        let cache_len = 1usize << (depth + 1);
        let mask = cache_len - 1;
        let mut setups: Vec<Option<QuerySetup>> = (0..count).map(|_| None).collect();
        let (queries, alpha, proof) = rayon::in_place_scope(|scope| {
            let mut slots = setups.iter_mut();
            let queries = sample_distinct_queries_visit(challenger, block_len, count, |q| {
                let slot = slots.next().expect("one setup per distinct query");
                let ntt = &ntt;
                let lane_weights = &lane_weights;
                let row = &row;
                scope.spawn(move |_| {
                    let base = q & !mask;
                    let qoff = q - base;
                    let mut cache = [F128::ZERO; 16];
                    for j in 0..cache_len {
                        cache[j] = dot(row(base + j), lane_weights);
                    }
                    let value = cache[qoff];
                    let stage = invert_local_staged(
                        ntt, log_d, depth + 1, cache_len, base, qoff, &mut cache,
                    );
                    let mut scales = [F128::ZERO; 4];
                    for s in 0..=depth {
                        scales[s] = one_plus_scale(ntt, log_d, s, q);
                    }
                    *slot = Some(QuerySetup { query: q, stage, scales, value, row: row(q).to_vec() });
                });
            });
            let alpha = challenger.sample_f128_vec(ceil_log2(count));
            let proof = merkle_multi_proof_for(tree, block_len, &queries);
            (queries, alpha, proof)
        });
        // Reassociate with the sorted order before applying alpha's weights.
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_unstable_by_key(|&i| setups[i].as_ref().expect("scope joined").query);
        let weights: Vec<F128> = if queries.is_empty() {
            Vec::new()
        } else {
            build_eq_table(&alpha).into_iter().take(count).collect()
        };
        assert_eq!(weights.len(), count);
        let mut stages = Vec::with_capacity(count);
        let mut scales = Vec::with_capacity(count);
        let mut positions = Vec::with_capacity(count);
        let mut w2_pows = Vec::with_capacity(count);
        let mut rows = Vec::with_capacity(count);
        let mut enforced = crate::field::F256Unreduced::ZERO;
        for ((i, &q), &w) in order.into_iter().zip(&queries).zip(&weights) {
            let setup = setups[i].take().expect("scope joined");
            debug_assert_eq!(setup.query, q);
            enforced ^= w.mul_unreduced(setup.value);
            w2_pows.push(mul(w, setup.scales[0]));
            positions.push(q >> depth);
            stages.push(setup.stage);
            scales.push(setup.scales);
            rows.push(setup.row);
        }
        let w0_pows = weights.clone();
        let dual = Self {
            ntt, log_d, log_inv_rate, depth, positions, weights, w0_pows,
            w2_pows, step_depth: 0, stages, scales,
        };
        (queries, alpha, (dual, enforced.reduce()), (proof, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restored_query_setup_matches_serial_transcript_and_state() {
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                for depth in [2, 3] {
                    for lane_bits in [0, 2, 6] {
                        let log_d = 9;
                        let block_len = 1usize << log_d;
                        let lanes = 1usize << lane_bits;
                        let lane_challenges = vec![F128::new(9, 17); lane_bits];
                        let mat: Vec<_> = (0..block_len * lanes)
                            .map(|i| F128::new(i as u64, (i as u64).wrapping_mul(37)))
                            .collect();
                        let row = |q| &mat[q * lanes..(q + 1) * lanes];
                        let tree = vec![[0u8; 32]; 2 * block_len - 1];
                        for count in [1, 19, 218] {
                            let mut old_ch = crate::challenger::FsChallenger::new(b"setup equality");
                            let mut new_ch = crate::challenger::FsChallenger::new(b"setup equality");
                            let queries = sample_distinct_queries(&mut old_ch, block_len, count);
                            let alpha = old_ch.sample_f128_vec(ceil_log2(count));
                            let (old, enforced) = Dual::new_sync(
                                depth, log_d, 1, &lane_challenges, &queries, &alpha, row,
                            );
                            let (new_queries, new_alpha, (new, new_enforced), (proof, rows)) =
                                Dual::sample_with_setup(
                                    depth, log_d, 1, &lane_challenges, block_len, count,
                                    &mut new_ch, &tree, row,
                                );
                            assert_eq!(queries, new_queries);
                            assert_eq!(alpha, new_alpha);
                            assert_eq!(old_ch.sample_f128(), new_ch.sample_f128());
                            assert_eq!(enforced, new_enforced);
                            assert_eq!(old.positions, new.positions);
                            assert_eq!(old.weights, new.weights);
                            assert_eq!(old.w0_pows, new.w0_pows);
                            assert_eq!(old.w2_pows, new.w2_pows);
                            assert_eq!(old.stages, new.stages);
                            assert_eq!(old.scales, new.scales);
                            assert_eq!(proof, merkle_multi_proof_for(&tree, block_len, &queries));
                            assert_eq!(rows, queries.iter().map(|&q| row(q).to_vec()).collect::<Vec<_>>());
                        }
                    }
                }
            });
        }
    }
}

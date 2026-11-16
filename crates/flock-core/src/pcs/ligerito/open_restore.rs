use super::*;

pub(super) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_OPEN_RESTORE").as_deref()
            != Some(std::ffi::OsStr::new("1"))
    });
    *ON
}

/// The sparse transpose with only its final output operation deferred.
pub(super) struct RetainedBasis {
    pub(super) data: Vec<F128>,
    pub(super) log_d: usize,
    pub(super) drop_log: usize,
    pub(super) low_half: bool,
}

impl RetainedBasis {
    pub(super) fn materialize(mut self, ntt: &AdditiveNttF128) -> Vec<F128> {
        if self.low_half {
            transpose_forward_ntt_factored_final_3layer_low_half(ntt, &mut self.data, self.log_d);
            self.data.truncate(1usize << (self.log_d - 1));
            self.data
        } else {
            let out = xor_collapse_cosets(&self.data, self.log_d - self.drop_log);
            crate::scratch::give_sparse_ntt_data(self.data);
            out
        }
    }

    /// Called only after the ordinary fold has finished writing `dst`.
    pub(super) fn add_into(self, ntt: &AdditiveNttF128, dst: &mut [F128]) {
        assert_eq!(dst.len(), 1usize << (self.log_d - self.drop_log));
        if self.low_half {
            add_final_low_half(ntt, &self.data, self.log_d, dst);
        } else {
            use rayon::prelude::*;
            let keep = dst.len();
            let cosets = self.data.len() / keep;
            let add = |base: usize, chunk: &mut [F128]| {
                for (j, out) in chunk.iter_mut().enumerate() {
                    let mut value = self.data[base + j];
                    for coset in 1..cosets {
                        value += self.data[coset * keep + base + j];
                    }
                    *out += value;
                }
            };
            if keep >= 2 * SMALL_NTT_MIN_ELEMS {
                dst.par_chunks_mut(SMALL_NTT_MIN_ELEMS)
                    .enumerate()
                    .for_each(|(ci, chunk)| add(ci * SMALL_NTT_MIN_ELEMS, chunk));
            } else {
                add(0, dst);
            }
        }
        crate::scratch::give_sparse_ntt_data(self.data);
    }
}

fn add_final_low_half(ntt: &AdditiveNttF128, data: &[F128], log_d: usize, dst: &mut [F128]) {
    use rayon::prelude::*;
    assert!(log_d >= 3);
    assert_eq!(data.len(), 1usize << log_d);
    assert_eq!(dst.len(), data.len() >> 1);
    let eighth = data.len() >> 3;
    let layer_1_twiddles = [ntt.twiddle(1, 0), ntt.twiddle(1, 1)];
    let layer_2_twiddles: [F128; 4] = core::array::from_fn(|block| ntt.twiddle(2, block));
    assert_eq!(layer_1_twiddles[0], F128::ZERO);
    assert_eq!(layer_2_twiddles[0], F128::ZERO);
    assert_eq!(layer_2_twiddles[3], layer_2_twiddles[1] + layer_2_twiddles[2]);
    let constants = L0FactoredFinal3Constants {
        a: layer_2_twiddles[1],
        d: layer_2_twiddles[2],
        a_plus_d: layer_2_twiddles[3],
        b: layer_1_twiddles[1],
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
    let constants = {
        use core::arch::aarch64::uint64x2_t;
        use crate::field::gf2_128::aarch64::const_prep;
        // SAFETY: AES and the little-endian field lane layout are required.
        unsafe {
            [constants.a, constants.d, constants.a_plus_d, constants.b]
                .map(|t| const_prep(core::mem::transmute::<F128, uint64x2_t>(t)))
        }
    };
    let dst_ptr = dst.as_mut_ptr() as usize;
    (0..eighth)
        .into_par_iter()
        .with_min_len(small_ntt_min_jobs(8))
        .for_each(|row| {
            let values = core::array::from_fn(|i| data[row + i * eighth]);
            #[cfg(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little"))]
            let retained = l0_factored_final3_prepared_row(values, constants);
            #[cfg(not(all(target_arch = "aarch64", target_feature = "aes", target_endian = "little")))]
            let retained = l0_factored_final3_retained_row(values, constants);
            // SAFETY: each row owns one distinct entry in each output quarter;
            // the fold joined before this call, and `data` cannot alias `dst`.
            unsafe {
                let ptr = dst_ptr as *mut F128;
                for (i, value) in retained.into_iter().enumerate() {
                    *ptr.add(row + i * eighth) += value;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restored_retained_add_matches_materialization() {
        for (log_d, drop_log) in [(6, 1), (9, 2), (12, 3), (13, 2), (17, 1), (15, 2)] {
            let ntt = AdditiveNttF128::standard(log_d);
            let mut positions = vec![0, 0, 3, 7, 8, (1 << log_d) - 1];
            positions.sort_unstable();
            let values: Vec<_> = (0..positions.len())
                .map(|i| if i % 3 == 0 { F128::ZERO } else { F128::new(i as u64, 19) })
                .collect();
            for lean in [false, true] {
                let basis = transpose_forward_ntt_sparse_retained_with(
                    &ntt, &positions, &values, log_d, drop_log, lean,
                ).unwrap();
                let mut actual: Vec<_> = (0..basis.len())
                    .map(|i| F128::new((i as u64).wrapping_mul(0x9e3779b97f4a7c15), !i as u64))
                    .collect();
                let expected: Vec<_> = actual.iter().zip(&basis).map(|(&a, &b)| a + b).collect();
                prepare_sparse_retained(&ntt, &positions, &values, log_d, drop_log, lean)
                    .unwrap().add_into(&ntt, &mut actual);
                assert_eq!(actual, expected, "log={log_d} drop={drop_log} lean={lean}");
            }
        }
    }
}

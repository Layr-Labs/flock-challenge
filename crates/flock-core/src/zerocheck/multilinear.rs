#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use crate::field::gf2_128::x86_64::{WideGhashX4, f128x4_loadu, f128x4_set, ghash_mul_x4};
use crate::field::{F128, F256Unreduced, PHI_8_TABLE};
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::univariate_skip::{SplitEqGhash, build_eq, pack_bits};

mod kernels;

#[cfg(target_arch = "aarch64")]
use kernels::aarch64::{
    fold_2vars_and_lookahead_chunk_neon_unchecked, fold_2vars_and_round_pair_chunk_neon_unchecked,
    fold_one_row_neon_unchecked_8, round2_msg_only_chunk_neon_unchecked_8,
    round3_fold_in_place_chunk_neon_unchecked_8, round4_fold_2vars_into_chunk_neon_unchecked_8,
    round23_lookahead_chunk_neon_unchecked_8,
    round45_fold_2vars_into_and_lookahead_chunk_neon_unchecked_8,
};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use kernels::x86_64::fold_and_message_x86_avx512;
#[cfg(target_arch = "x86_64")]
use kernels::x86_64::{
    fold_2vars_and_lookahead_chunk_x86_unchecked, fold_2vars_and_round_pair_chunk_x86_unchecked,
    fold_round2_pair_x86_unchecked_8, round2_msg_only_chunk_x86_unchecked_8,
    round3_fold_in_place_chunk_x86_unchecked_8, round4_fold_2vars_into_chunk_x86_unchecked_8,
    round23_lookahead_chunk_x86_unchecked_8,
    round45_fold_2vars_into_and_lookahead_chunk_x86_unchecked_8,
};

fn round2_pair_skip(padding: &PaddingSpec, k_skip: usize) -> (usize, usize) {
    if padding.k_log <= k_skip + 1 {
        return (0, usize::MAX);
    }
    let pairs_per_block = 1usize << (padding.k_log - k_skip - 1);
    let chunk_bits = 1usize << k_skip;
    let useful_pairs = padding.useful_bits_per_block.div_ceil(2 * chunk_bits);
    if useful_pairs >= pairs_per_block {
        return (0, usize::MAX);
    }
    (pairs_per_block - 1, useful_pairs)
}

pub fn lagrange_weights_naive(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    assert!(ell <= 256, "k_skip > 8 would exceed PHI_8_TABLE");
    let mut weights = vec![F128::ZERO; ell];
    for i in 0..ell {
        let si = PHI_8_TABLE[i];
        let mut num = F128::ONE;
        let mut den = F128::ONE;
        for j in 0..ell {
            if j == i {
                continue;
            }
            let sj = PHI_8_TABLE[j];
            num *= z + sj;
            den *= si + sj;
        }
        weights[i] = num * den.inv();
    }
    weights
}

pub fn lagrange_weights_lambda_naive(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    let mut weights = vec![F128::ZERO; ell];
    for i in 0..ell {
        let si = PHI_8_TABLE[ell + i];
        let mut num = F128::ONE;
        let mut den = F128::ONE;
        for j in 0..ell {
            if j == i {
                continue;
            }
            let sj = PHI_8_TABLE[ell + j];
            num *= z + sj;
            den *= si + sj;
        }
        weights[i] = num * den.inv();
    }
    weights
}

pub fn interpolate_at_z_on_lambda(values: &[F128], k_skip: usize, z: F128) -> F128 {
    let ell = 1usize << k_skip;
    assert_eq!(values.len(), ell);
    let weights = lagrange_weights_lambda_naive(k_skip, z);
    crate::field::f128_slice::inner_product(&weights, values)
}

pub fn interpolate_at_z_combined(values_on_lambda: &[F128], k_skip: usize, z: F128) -> F128 {
    let ell = 1usize << k_skip;
    assert_eq!(values_on_lambda.len(), ell);
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    let n_total = 2 * ell;
    let mut acc = F128::ZERO;
    for i in 0..ell {
        let node_idx = ell + i;
        let si = PHI_8_TABLE[node_idx];
        let mut num = F128::ONE;
        let mut den = F128::ONE;
        for j in 0..n_total {
            if j == node_idx {
                continue;
            }
            let sj = PHI_8_TABLE[j];
            num *= z + sj;
            den *= si + sj;
        }
        let weight = num * den.inv();
        acc += weight * values_on_lambda[i];
    }
    acc
}

pub fn eq_eval(r: &[F128], x: &[F128]) -> F128 {
    assert_eq!(r.len(), x.len());
    let mut acc = F128::ONE;
    for i in 0..r.len() {
        acc *= F128::ONE + r[i] + x[i];
    }
    acc
}

pub fn eq_eval_binary_x(r: &[F128], x_bits: u32) -> F128 {
    debug_assert!(r.len() <= 32, "x_bits is u32; r > 32 dims not supported");
    let mut acc = F128::ONE;
    for (i, &r_i) in r.iter().enumerate() {
        let factor = if (x_bits >> i) & 1 == 1 {
            r_i
        } else {
            F128::ONE + r_i
        };
        acc *= factor;
    }
    acc
}

pub fn fold_at_z_naive(witness: &[bool], m: usize, k_skip: usize, weights: &[F128]) -> Vec<F128> {
    assert!(k_skip <= m);
    let ell = 1usize << k_skip;
    let n_rest = 1usize << (m - k_skip);
    assert_eq!(witness.len(), 1usize << m);
    assert_eq!(weights.len(), ell);

    let mut folded = vec![F128::ZERO; n_rest];
    for x_rest in 0..n_rest {
        let base = x_rest * ell;
        let mut acc = F128::ZERO;
        for s in 0..ell {
            if witness[base + s] {
                acc += weights[s];
            }
        }
        folded[x_rest] = acc;
    }
    folded
}

pub fn round_pair_naive(a_mlv: &[F128], b_mlv: &[F128], r: &[F128]) -> (F128, F128) {
    let n = a_mlv.len();
    assert_eq!(b_mlv.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r.len(), log_n);

    let eq_remaining = build_eq(&r[1..]);
    assert_eq!(eq_remaining.len(), half);

    let mut g_one = F256Unreduced::ZERO;
    let mut g_inf = F256Unreduced::ZERO;
    for x_prime in 0..half {
        let a0 = a_mlv[2 * x_prime];
        let a1 = a_mlv[2 * x_prime + 1];
        let b0 = b_mlv[2 * x_prime];
        let b1 = b_mlv[2 * x_prime + 1];
        let eq_x = eq_remaining[x_prime];
        g_one ^= eq_x.mul_unreduced(a1 * b1);

        g_inf ^= eq_x.mul_unreduced((a0 + a1) * (b0 + b1));
    }
    (r[0] * g_one.reduce(), g_inf.reduce())
}

pub fn uni_skip_fold_and_round_pair_naive(
    a: &[bool],
    b: &[bool],
    m: usize,
    k_skip: usize,
    z: F128,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    assert!(
        m > k_skip,
        "need at least one multilinear variable past the skip"
    );
    assert_eq!(mlv_challenges.len(), m - k_skip);

    let weights = lagrange_weights_naive(k_skip, z);
    let a_mlv = fold_at_z_naive(a, m, k_skip, &weights);
    let b_mlv = fold_at_z_naive(b, m, k_skip, &weights);
    let (msg_1, msg_inf) = round_pair_naive(&a_mlv, &b_mlv, mlv_challenges);
    (a_mlv, b_mlv, msg_1, msg_inf)
}

#[derive(Clone, Debug)]
pub struct UniSkipFoldTable {
    pub n_chunks: usize,
    pub data: Vec<F128>,
}

impl UniSkipFoldTable {
    pub fn new(k_skip: usize, z: F128) -> Self {
        let ell = 1usize << k_skip;
        assert_eq!(ell % 8, 0, "k_skip must be ≥ 3 (need ell divisible by 8)");
        let n_chunks = ell / 8;
        let weights = lagrange_weights_naive(k_skip, z);

        let mut data = vec![F128::ZERO; n_chunks * 256];
        for j in 0..n_chunks {
            let basis = &weights[8 * j..8 * j + 8];

            for b in 0..8 {
                data[j * 256 + (1 << b)] = basis[b];
            }

            for v in 3usize..256 {
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                let (lo_bit, parent) = unsafe {
                    let lo = crate::gf2_kernels::gf2_kern_isolate_lowest_one(v as u64) as usize;
                    let par = crate::gf2_kernels::gf2_kern_clear_lowest_one(v as u64) as usize;
                    (lo, par)
                };
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                let (lo_bit, parent) = {
                    let lo = v.isolate_lowest_one();
                    (lo, v ^ lo)
                };
                if parent == 0 {
                    continue;
                }
                data[j * 256 + v] = data[j * 256 + parent] + data[j * 256 + lo_bit];
            }
        }
        Self { n_chunks, data }
    }

    pub fn scaled_pair(&self, rho_0: F128) -> (Self, Self) {
        let n_chunks = self.n_chunks;
        let one_plus_rho = F128::ONE + rho_0;
        let mut data0 = vec![F128::ZERO; n_chunks * 256];
        let mut data1 = vec![F128::ZERO; n_chunks * 256];
        for j in 0..n_chunks {
            let base = j * 256;
            for b in 0..8 {
                let idx = 1usize << b;
                let w = self.data[base + idx];
                data0[base + idx] = one_plus_rho * w;
                data1[base + idx] = rho_0 * w;
            }
            for v in 3usize..256 {
                let lo_bit = v & v.wrapping_neg();
                let parent = v ^ lo_bit;
                if parent == 0 {
                    continue;
                }
                data0[base + v] = data0[base + parent] + data0[base + lo_bit];
                data1[base + v] = data1[base + parent] + data1[base + lo_bit];
            }
        }
        (
            Self {
                n_chunks,
                data: data0,
            },
            Self {
                n_chunks,
                data: data1,
            },
        )
    }

    #[inline]
    pub fn fold_one_row(&self, bytes: &[u8]) -> F128 {
        assert_eq!(bytes.len(), self.n_chunks);
        if self.n_chunks == 8 && self.data.len() >= 8 * 256 {
            unsafe {
                return *self.data.get_unchecked(bytes[0] as usize)
                    + *self.data.get_unchecked(256 + bytes[1] as usize)
                    + *self.data.get_unchecked(512 + bytes[2] as usize)
                    + *self.data.get_unchecked(768 + bytes[3] as usize)
                    + *self.data.get_unchecked(1024 + bytes[4] as usize)
                    + *self.data.get_unchecked(1280 + bytes[5] as usize)
                    + *self.data.get_unchecked(1536 + bytes[6] as usize)
                    + *self.data.get_unchecked(1792 + bytes[7] as usize);
            }
        }
        let mut acc = F128::ZERO;
        for j in 0..self.n_chunks {
            acc += self.data[j * 256 + bytes[j] as usize];
        }
        acc
    }
}

pub fn uni_skip_fold_and_round_pair_optimized_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    uni_skip_fold_and_round_pair_optimized_packed_padded(
        a_packed,
        b_packed,
        m,
        k_skip,
        table,
        mlv_challenges,
        &PaddingSpec::dense(m),
    )
}

pub fn uni_skip_fold_and_round_pair_optimized_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    use rayon::prelude::*;

    assert_eq!(
        k_skip, 6,
        "optimized fold-and-round_pair variant is k_skip=6 only"
    );
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);

    let mut a_folded: Vec<F128> = crate::scratch::take_f128(n_out);
    let mut b_folded: Vec<F128> = crate::scratch::take_f128(n_out);

    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_out);

    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);

    let (sum1, sum_inf) = a_folded
        .par_chunks_mut(chunk_size)
        .zip(b_folded.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(x_hi, (a_chunk, b_chunk))| {
            let mut p1_acc = F256Unreduced::ZERO;
            let mut pinf_acc = F256Unreduced::ZERO;
            let pair_idx_base = x_hi * lo_size;

            #[cfg(target_arch = "aarch64")]
            unsafe {
                let table_ptr = table.data.as_ptr() as *const u8;
                let a_pkt_ptr = a_packed.as_ptr();
                let b_pkt_ptr = b_packed.as_ptr();
                let base = x_hi * chunk_size;

                for x_lo in 0..lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        continue;
                    }
                    let x0g = base + 2 * x_lo;
                    let x1g = x0g + 1;

                    let a0 = fold_one_row_neon_unchecked_8(table_ptr, a_pkt_ptr.add(x0g * 8));
                    let b0 = fold_one_row_neon_unchecked_8(table_ptr, b_pkt_ptr.add(x0g * 8));
                    let a1 = fold_one_row_neon_unchecked_8(table_ptr, a_pkt_ptr.add(x1g * 8));
                    let b1 = fold_one_row_neon_unchecked_8(table_ptr, b_pkt_ptr.add(x1g * 8));

                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;

                    let eq_l = eq_lo[x_lo];
                    let ad = a0 + a1;
                    let bd = b0 + b1;
                    let (g1_lo, g1_hi, ginf_lo, ginf_hi) = crate::gf2_kernels::k0_f128_mul_vec2_raw(
                        a1.lo, a1.hi, ad.lo, ad.hi, b1.lo, b1.hi, bd.lo, bd.hi,
                    );
                    p1_acc ^= eq_l.mul_unreduced(F128 {
                        lo: g1_lo,
                        hi: g1_hi,
                    });
                    pinf_acc ^= eq_l.mul_unreduced(F128 {
                        lo: ginf_lo,
                        hi: ginf_hi,
                    });
                }
            }
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            unsafe {
                let table_ptr = table.data.as_ptr();
                let a_pkt_ptr = a_packed.as_ptr();
                let b_pkt_ptr = b_packed.as_ptr();
                let base = x_hi * chunk_size;
                let mut p1_wide = WideGhashX4::zero();
                let mut pinf_wide = WideGhashX4::zero();
                let mut x_lo = 0;

                while x_lo + 4 <= lo_size {
                    let mut a0 = [F128::ZERO; 4];
                    let mut a1 = [F128::ZERO; 4];
                    let mut b0 = [F128::ZERO; 4];
                    let mut b1 = [F128::ZERO; 4];

                    for lane in 0..4 {
                        let pair = x_lo + lane;
                        let x0l = 2 * pair;
                        let x1l = x0l + 1;
                        if ((pair_idx_base + pair) & pair_in_block_mask) >= useful_pairs_inclusive {
                            a_chunk[x0l] = F128::ZERO;
                            a_chunk[x1l] = F128::ZERO;
                            b_chunk[x0l] = F128::ZERO;
                            b_chunk[x1l] = F128::ZERO;
                            continue;
                        }

                        let x0g = base + x0l;
                        let x1g = x0g + 1;
                        let folded = fold_round2_pair_x86_unchecked_8(
                            table_ptr,
                            a_pkt_ptr.add(x0g * 8),
                            a_pkt_ptr.add(x1g * 8),
                            b_pkt_ptr.add(x0g * 8),
                            b_pkt_ptr.add(x1g * 8),
                        );
                        [a0[lane], a1[lane], b0[lane], b1[lane]] = folded;
                        a_chunk[x0l] = a0[lane];
                        a_chunk[x1l] = a1[lane];
                        b_chunk[x0l] = b0[lane];
                        b_chunk[x1l] = b1[lane];
                    }

                    let a1x4 = f128x4_loadu(a1.as_ptr());
                    let b1x4 = f128x4_loadu(b1.as_ptr());
                    let a_sum_x4 =
                        f128x4_set(a0[0] + a1[0], a0[1] + a1[1], a0[2] + a1[2], a0[3] + a1[3]);
                    let b_sum_x4 =
                        f128x4_set(b0[0] + b1[0], b0[1] + b1[1], b0[2] + b1[2], b0[3] + b1[3]);
                    let g1x4 = ghash_mul_x4(a1x4, b1x4);
                    let g_inf_x4 = ghash_mul_x4(a_sum_x4, b_sum_x4);
                    let eqx4 = f128x4_loadu(eq_lo[x_lo..].as_ptr());
                    p1_wide.mul_acc(eqx4, g1x4);
                    pinf_wide.mul_acc(eqx4, g_inf_x4);
                    x_lo += 4;
                }

                while x_lo < lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        x_lo += 1;
                        continue;
                    }

                    let x0g = base + x0l;
                    let x1g = x0g + 1;
                    let [a0, a1, b0, b1] = fold_round2_pair_x86_unchecked_8(
                        table_ptr,
                        a_pkt_ptr.add(x0g * 8),
                        a_pkt_ptr.add(x1g * 8),
                        b_pkt_ptr.add(x0g * 8),
                        b_pkt_ptr.add(x1g * 8),
                    );
                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;
                    let eq_l = eq_lo[x_lo];
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                    x_lo += 1;
                }

                p1_acc ^= p1_wide.fold();
                pinf_acc ^= pinf_wide.fold();
            }
            #[cfg(all(
                target_arch = "x86_64",
                not(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))
            ))]
            unsafe {
                let table_ptr = table.data.as_ptr();
                let a_pkt_ptr = a_packed.as_ptr();
                let b_pkt_ptr = b_packed.as_ptr();
                let base = x_hi * chunk_size;
                for x_lo in 0..lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        continue;
                    }
                    let x0g = base + x0l;
                    let x1g = x0g + 1;
                    let [a0, a1, b0, b1] = fold_round2_pair_x86_unchecked_8(
                        table_ptr,
                        a_pkt_ptr.add(x0g * 8),
                        a_pkt_ptr.add(x1g * 8),
                        b_pkt_ptr.add(x0g * 8),
                        b_pkt_ptr.add(x1g * 8),
                    );
                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;
                    let eq_l = eq_lo[x_lo];
                    let ad = a0 + a1;
                    let bd = b0 + b1;
                    let (g1_lo, g1_hi, ginf_lo, ginf_hi) = crate::gf2_kernels::k0_f128_mul_vec2_raw(
                        a1.lo, a1.hi, ad.lo, ad.hi, b1.lo, b1.hi, bd.lo, bd.hi,
                    );
                    p1_acc ^= eq_l.mul_unreduced(F128 {
                        lo: g1_lo,
                        hi: g1_hi,
                    });
                    pinf_acc ^= eq_l.mul_unreduced(F128 {
                        lo: ginf_lo,
                        hi: ginf_hi,
                    });
                }
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64",)))]
            {
                let base = x_hi * chunk_size;
                for x_lo in 0..lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        continue;
                    }
                    let x0g = base + 2 * x_lo;
                    let x1g = x0g + 1;
                    let a0 = table.fold_one_row(&a_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let b0 = table.fold_one_row(&b_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let a1 = table.fold_one_row(&a_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    let b1 = table.fold_one_row(&b_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;
                    let eq_l = eq_lo[x_lo];
                    let ad = a0 + a1;
                    let bd = b0 + b1;
                    let (g1_lo, g1_hi, ginf_lo, ginf_hi) = crate::gf2_kernels::k0_f128_mul_vec2_raw(
                        a1.lo, a1.hi, ad.lo, ad.hi, b1.lo, b1.hi, bd.lo, bd.hi,
                    );
                    p1_acc ^= eq_l.mul_unreduced(F128 {
                        lo: g1_lo,
                        hi: g1_hi,
                    });
                    pinf_acc ^= eq_l.mul_unreduced(F128 {
                        lo: ginf_lo,
                        hi: ginf_hi,
                    });
                }
            }

            let p1 = p1_acc.reduce();
            let pinf = pinf_acc.reduce();
            let eq_h = eq_hi[x_hi];
            (eq_h.mul_unreduced(p1), eq_h.mul_unreduced(pinf))
        })
        .reduce(
            || (F256Unreduced::ZERO, F256Unreduced::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 ^ c1, sinf ^ cinf),
        );

    (
        a_folded,
        b_folded,
        mlv_challenges[0] * sum1.reduce(),
        sum_inf.reduce(),
    )
}

pub fn uni_skip_round2_msg_only_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (F128, F128) {
    use rayon::prelude::*;

    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);

    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_out);

    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);

    let (sum1, sum_inf) = (0..hi_size)
        .into_par_iter()
        .map(|x_hi| {
            let pair_idx_base = x_hi * lo_size;
            let base = x_hi * chunk_size;

            #[cfg(target_arch = "aarch64")]
            let (p1, pinf) = unsafe {
                round2_msg_only_chunk_neon_unchecked_8(
                    table.data.as_ptr() as *const u8,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    eq_lo,
                    base,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(target_arch = "x86_64")]
            let (p1, pinf) = unsafe {
                round2_msg_only_chunk_x86_unchecked_8(
                    table.data.as_ptr(),
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    eq_lo,
                    base,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            let (p1, pinf) = {
                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                for x_lo in 0..lo_size {
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        continue;
                    }
                    let x0g = base + 2 * x_lo;
                    let x1g = x0g + 1;
                    let a0 = table.fold_one_row(&a_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let b0 = table.fold_one_row(&b_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let a1 = table.fold_one_row(&a_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    let b1 = table.fold_one_row(&b_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    let eq_l = eq_lo[x_lo];
                    let ad = a0 + a1;
                    let bd = b0 + b1;
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced(ad * bd);
                }
                (p1_acc.reduce(), pinf_acc.reduce())
            };

            let eq_h = eq_hi[x_hi];
            (eq_h.mul_unreduced(p1), eq_h.mul_unreduced(pinf))
        })
        .reduce(
            || (F256Unreduced::ZERO, F256Unreduced::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 ^ c1, sinf ^ cinf),
        );

    let s1 = sum1.reduce();
    let msg1 = if mlv_challenges[0] == F128::ONE {
        s1
    } else {
        mlv_challenges[0] * s1
    };
    (msg1, sum_inf.reduce())
}

#[allow(clippy::too_many_arguments)]
pub fn uni_skip_fold_in_place_and_round3_msg_packed_padded(
    a_vec: &mut [F128],
    b_vec: &mut [F128],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    rho_0: F128,
    r_next_round3: &[F128],
    padding: &PaddingSpec,
) -> (F128, F128) {
    use rayon::prelude::*;

    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_half = 1usize << (m - k_skip - 1);
    assert_eq!(a_vec.len(), n_half);
    assert_eq!(b_vec.len(), n_half);
    assert_eq!(r_next_round3.len(), m - k_skip - 1);

    let eq = SplitEqGhash::new(&r_next_round3[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_half);

    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let (table0, table1) = table.scaled_pair(rho_0);

    let (sum1, sum_inf) = a_vec
        .par_chunks_mut(chunk_size)
        .zip(b_vec.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(x_hi, (a_chunk, b_chunk))| {
            let base_j = x_hi * chunk_size;

            #[cfg(target_arch = "aarch64")]
            let (p1, pinf) = unsafe {
                round3_fold_in_place_chunk_neon_unchecked_8(
                    table.data.as_ptr() as *const u8,
                    rho_0,
                    a_chunk,
                    b_chunk,
                    eq_lo,
                    base_j,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(target_arch = "x86_64")]
            let (p1, pinf) = unsafe {
                round3_fold_in_place_chunk_x86_unchecked_8(
                    table.data.as_ptr(),
                    rho_0,
                    a_chunk,
                    b_chunk,
                    eq_lo,
                    base_j,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            let (p1, pinf) = {
                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                for x_lo in 0..lo_size {
                    let j0l = 2 * x_lo;
                    let j1l = j0l + 1;
                    let j0g = base_j + j0l;
                    let j1g = j0g + 1;

                    let (a_j0, b_j0) = if (j0g & pair_in_block_mask) >= useful_pairs_inclusive {
                        (F128::ZERO, F128::ZERO)
                    } else {
                        let a_bytes = a_chunk[j0l].to_le_bytes();
                        let b_bytes = b_chunk[j0l].to_le_bytes();
                        (
                            table0.fold_one_row(&a_bytes[0..8])
                                + table1.fold_one_row(&a_bytes[8..16]),
                            table0.fold_one_row(&b_bytes[0..8])
                                + table1.fold_one_row(&b_bytes[8..16]),
                        )
                    };

                    let (a_j1, b_j1) = if (j1g & pair_in_block_mask) >= useful_pairs_inclusive {
                        (F128::ZERO, F128::ZERO)
                    } else {
                        let a_bytes = a_chunk[j1l].to_le_bytes();
                        let b_bytes = b_chunk[j1l].to_le_bytes();
                        (
                            table0.fold_one_row(&a_bytes[0..8])
                                + table1.fold_one_row(&a_bytes[8..16]),
                            table0.fold_one_row(&b_bytes[0..8])
                                + table1.fold_one_row(&b_bytes[8..16]),
                        )
                    };

                    a_chunk[j0l] = a_j0;
                    a_chunk[j1l] = a_j1;
                    b_chunk[j0l] = b_j0;
                    b_chunk[j1l] = b_j1;

                    let eq_l = eq_lo[x_lo];
                    let ad = a_j0 + a_j1;
                    let bd = b_j0 + b_j1;
                    p1_acc ^= eq_l.mul_unreduced(a_j1 * b_j1);
                    pinf_acc ^= eq_l.mul_unreduced(ad * bd);
                }
                (p1_acc.reduce(), pinf_acc.reduce())
            };

            let eq_h = eq_hi[x_hi];
            (eq_h.mul_unreduced(p1), eq_h.mul_unreduced(pinf))
        })
        .reduce(
            || (F256Unreduced::ZERO, F256Unreduced::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 ^ c1, sinf ^ cinf),
        );

    let s1 = sum1.reduce();
    let msg1 = if r_next_round3[0] == F128::ONE {
        s1
    } else {
        r_next_round3[0] * s1
    };
    (msg1, sum_inf.reduce())
}

#[derive(Clone, Copy, Debug)]
pub struct Round23Lookahead {
    pub msg2_1: F128,
    pub msg2_inf: F128,
    pub s01: F128,
    pub s11: F128,
    pub si1: F128,
    pub t0: F128,
    pub t1: F128,
    pub ti: F128,
}

impl Round23Lookahead {
    #[inline]
    pub fn eval_round3(&self, rho_0: F128, r_next0: F128) -> (F128, F128) {
        let opr0 = F128::ONE + rho_0;
        let cross = rho_0 * opr0;
        let g1 = self.s01.mle_fold(self.s11, rho_0) + cross * self.si1;
        let ginf = self.t0.mle_fold(self.t1, rho_0) + cross * self.ti;
        let msg1 = if r_next0 == F128::ONE {
            g1
        } else {
            r_next0 * g1
        };
        (msg1, ginf)
    }
}

pub fn uni_skip_round2_and_round3_lookahead_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> Round23Lookahead {
    use rayon::prelude::*;

    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);

    let r1 = mlv_challenges[1];
    let eq3 = SplitEqGhash::new(&mlv_challenges[2..]);
    let lo_size = 1usize << eq3.n_lo;
    let hi_size = 1usize << eq3.n_hi;
    assert_eq!(lo_size * hi_size * 4, n_out);

    let eq_hi = &eq3.hi;
    let eq_lo = &eq3.lo;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);

    let sums = (0..hi_size)
        .into_par_iter()
        .map(|x_hi| {
            let q_base = x_hi * lo_size;

            #[cfg(target_arch = "aarch64")]
            let p = unsafe {
                round23_lookahead_chunk_neon_unchecked_8(
                    table.data.as_ptr() as *const u8,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    eq_lo,
                    q_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(target_arch = "x86_64")]
            let p = unsafe {
                round23_lookahead_chunk_x86_unchecked_8(
                    table.data.as_ptr(),
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    eq_lo,
                    q_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            let p = {
                let mut acc = [F256Unreduced::ZERO; 8];
                for q_lo in 0..lo_size {
                    let qg = q_base + q_lo;
                    let j0 = 2 * qg;
                    let j1 = j0 + 1;
                    if (j0 & pair_in_block_mask) >= useful_pairs_inclusive {
                        continue;
                    }
                    let r0 = 4 * qg;
                    let a00 = table.fold_one_row(&a_packed[r0 * n_chunks..(r0 + 1) * n_chunks]);
                    let b00 = table.fold_one_row(&b_packed[r0 * n_chunks..(r0 + 1) * n_chunks]);
                    let a10 =
                        table.fold_one_row(&a_packed[(r0 + 1) * n_chunks..(r0 + 2) * n_chunks]);
                    let b10 =
                        table.fold_one_row(&b_packed[(r0 + 1) * n_chunks..(r0 + 2) * n_chunks]);
                    let (a01, b01, a11, b11) = if (j1 & pair_in_block_mask) < useful_pairs_inclusive
                    {
                        (
                            table.fold_one_row(&a_packed[(r0 + 2) * n_chunks..(r0 + 3) * n_chunks]),
                            table.fold_one_row(&b_packed[(r0 + 2) * n_chunks..(r0 + 3) * n_chunks]),
                            table.fold_one_row(&a_packed[(r0 + 3) * n_chunks..(r0 + 4) * n_chunks]),
                            table.fold_one_row(&b_packed[(r0 + 3) * n_chunks..(r0 + 4) * n_chunks]),
                        )
                    } else {
                        (F128::ZERO, F128::ZERO, F128::ZERO, F128::ZERO)
                    };
                    let eq_l = eq_lo[q_lo];
                    let da0 = a00 + a10;
                    let db0 = b00 + b10;
                    let da1 = a01 + a11;
                    let db1 = b01 + b11;
                    let da_0 = a00 + a01;
                    let db_0 = b00 + b01;
                    let da_1 = a10 + a11;
                    let db_1 = b10 + b11;
                    let da_i = da0 + da1;
                    let db_i = db0 + db1;
                    acc[0] ^= eq_l.mul_unreduced(a10 * b10);
                    acc[1] ^= eq_l.mul_unreduced(da0 * db0);
                    acc[2] ^= eq_l.mul_unreduced(a11 * b11);
                    acc[3] ^= eq_l.mul_unreduced(da1 * db1);
                    acc[4] ^= eq_l.mul_unreduced(a01 * b01);
                    acc[5] ^= eq_l.mul_unreduced(da_0 * db_0);
                    acc[6] ^= eq_l.mul_unreduced(da_1 * db_1);
                    acc[7] ^= eq_l.mul_unreduced(da_i * db_i);
                }
                acc.map(|u| u.reduce())
            };

            let eq_h = eq_hi[x_hi];
            p.map(|v| eq_h.mul_unreduced(v))
        })
        .reduce(
            || [F256Unreduced::ZERO; 8],
            |mut a, b| {
                for k in 0..8 {
                    a[k] ^= b[k];
                }
                a
            },
        );

    let [s10, si0, s11, si1, s01, t0, t1, ti] = sums.map(|u| u.reduce());
    let g2_1 = s10.mle_fold(s11, r1);
    let msg2_inf = si0.mle_fold(si1, r1);
    let msg2_1 = if mlv_challenges[0] == F128::ONE {
        g2_1
    } else {
        mlv_challenges[0] * g2_1
    };

    Round23Lookahead {
        msg2_1,
        msg2_inf,
        s01,
        s11,
        si1,
        t0,
        t1,
        ti,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn uni_skip_fold_2vars_into_and_round4_msg_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    a_out: &mut [F128],
    b_out: &mut [F128],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    rho_0: F128,
    rho_1: F128,
    r_next_round4: &[F128],
    padding: &PaddingSpec,
) -> (F128, F128) {
    use rayon::prelude::*;

    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let n_quarter = 1usize << (m - k_skip - 2);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(a_out.len(), n_quarter);
    assert_eq!(b_out.len(), n_quarter);
    assert_eq!(r_next_round4.len(), m - k_skip - 2);

    let eq4 = SplitEqGhash::new(&r_next_round4[1..]);
    let lo_size = 1usize << eq4.n_lo;
    let hi_size = 1usize << eq4.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_quarter);

    let chunk_size = 2 * lo_size;
    let eq_hi = &eq4.hi;
    let eq_lo = &eq4.lo;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);
    #[cfg(target_arch = "x86_64")]
    let use_stream = n_quarter >= (1 << 19)
        && ((a_out.as_ptr() as usize) & 15) == 0
        && ((b_out.as_ptr() as usize) & 15) == 0;

    let (sum1, sum_inf) = a_out
        .par_chunks_mut(chunk_size)
        .zip(b_out.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(x_hi, (a_out_chunk, b_out_chunk))| {
            let q_base = x_hi * chunk_size;

            #[cfg(target_arch = "aarch64")]
            let (p1, pinf) = unsafe {
                round4_fold_2vars_into_chunk_neon_unchecked_8(
                    table.data.as_ptr() as *const u8,
                    rho_0,
                    rho_1,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    a_out_chunk,
                    b_out_chunk,
                    eq_lo,
                    q_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(target_arch = "x86_64")]
            let (p1, pinf) = unsafe {
                round4_fold_2vars_into_chunk_x86_unchecked_8(
                    table.data.as_ptr(),
                    rho_0,
                    rho_1,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    a_out_chunk,
                    b_out_chunk,
                    eq_lo,
                    q_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                    use_stream,
                )
            };
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            let (p1, pinf) = {
                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                let fold_q = |qg: usize| -> (F128, F128) {
                    let j0 = 2 * qg;
                    let j1 = j0 + 1;
                    if (j0 & pair_in_block_mask) >= useful_pairs_inclusive {
                        return (F128::ZERO, F128::ZERO);
                    }
                    let r0 = 4 * qg;
                    let a00 = table.fold_one_row(&a_packed[r0 * n_chunks..(r0 + 1) * n_chunks]);
                    let b00 = table.fold_one_row(&b_packed[r0 * n_chunks..(r0 + 1) * n_chunks]);
                    let a10 =
                        table.fold_one_row(&a_packed[(r0 + 1) * n_chunks..(r0 + 2) * n_chunks]);
                    let b10 =
                        table.fold_one_row(&b_packed[(r0 + 1) * n_chunks..(r0 + 2) * n_chunks]);
                    let a_j0 = a00.mle_fold(a10, rho_0);
                    let b_j0 = b00.mle_fold(b10, rho_0);
                    let (a_j1, b_j1) = if (j1 & pair_in_block_mask) < useful_pairs_inclusive {
                        let a01 =
                            table.fold_one_row(&a_packed[(r0 + 2) * n_chunks..(r0 + 3) * n_chunks]);
                        let b01 =
                            table.fold_one_row(&b_packed[(r0 + 2) * n_chunks..(r0 + 3) * n_chunks]);
                        let a11 =
                            table.fold_one_row(&a_packed[(r0 + 3) * n_chunks..(r0 + 4) * n_chunks]);
                        let b11 =
                            table.fold_one_row(&b_packed[(r0 + 3) * n_chunks..(r0 + 4) * n_chunks]);
                        (a01.mle_fold(a11, rho_0), b01.mle_fold(b11, rho_0))
                    } else {
                        (F128::ZERO, F128::ZERO)
                    };
                    (a_j0.mle_fold(a_j1, rho_1), b_j0.mle_fold(b_j1, rho_1))
                };
                for x_lo in 0..lo_size {
                    let q0l = 2 * x_lo;
                    let q1l = q0l + 1;
                    let q0g = q_base + q0l;
                    let (a_q0, b_q0) = fold_q(q0g);
                    let (a_q1, b_q1) = fold_q(q0g + 1);
                    a_out_chunk[q0l] = a_q0;
                    a_out_chunk[q1l] = a_q1;
                    b_out_chunk[q0l] = b_q0;
                    b_out_chunk[q1l] = b_q1;
                    let eq_l = eq_lo[x_lo];
                    p1_acc ^= eq_l.mul_unreduced(a_q1 * b_q1);
                    pinf_acc ^= eq_l.mul_unreduced((a_q0 + a_q1) * (b_q0 + b_q1));
                }
                (p1_acc.reduce(), pinf_acc.reduce())
            };

            let eq_h = eq_hi[x_hi];
            (eq_h.mul_unreduced(p1), eq_h.mul_unreduced(pinf))
        })
        .reduce(
            || (F256Unreduced::ZERO, F256Unreduced::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 ^ c1, sinf ^ cinf),
        );

    let s1 = sum1.reduce();
    let msg1 = if r_next_round4[0] == F128::ONE {
        s1
    } else {
        r_next_round4[0] * s1
    };
    (msg1, sum_inf.reduce())
}

#[allow(clippy::too_many_arguments)]
pub fn uni_skip_fold_2vars_into_and_round45_lookahead_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    a_out: &mut [F128],
    b_out: &mut [F128],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    rho_0: F128,
    rho_1: F128,
    r_next_round4: &[F128],
    padding: &PaddingSpec,
) -> Round23Lookahead {
    use rayon::prelude::*;

    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let n_quarter = 1usize << (m - k_skip - 2);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(a_out.len(), n_quarter);
    assert_eq!(b_out.len(), n_quarter);
    assert_eq!(r_next_round4.len(), m - k_skip - 2);

    let r1 = r_next_round4[1];
    let eq5 = SplitEqGhash::new(&r_next_round4[2..]);
    let lo_size = 1usize << eq5.n_lo;
    let hi_size = 1usize << eq5.n_hi;
    assert_eq!(lo_size * hi_size * 4, n_quarter);

    let chunk_size = 4 * lo_size;
    let eq_hi = &eq5.hi;
    let eq_lo = &eq5.lo;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);
    #[cfg(target_arch = "x86_64")]
    let use_stream = n_quarter >= (1 << 19)
        && ((a_out.as_ptr() as usize) & 15) == 0
        && ((b_out.as_ptr() as usize) & 15) == 0;

    let sums = a_out
        .par_chunks_mut(chunk_size)
        .zip(b_out.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(x_hi, (a_out_chunk, b_out_chunk))| {
            let q_base = x_hi * chunk_size;

            #[cfg(target_arch = "aarch64")]
            let p = unsafe {
                round45_fold_2vars_into_and_lookahead_chunk_neon_unchecked_8(
                    table.data.as_ptr() as *const u8,
                    rho_0,
                    rho_1,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    a_out_chunk,
                    b_out_chunk,
                    eq_lo,
                    q_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                )
            };
            #[cfg(target_arch = "x86_64")]
            let p = unsafe {
                round45_fold_2vars_into_and_lookahead_chunk_x86_unchecked_8(
                    table.data.as_ptr(),
                    rho_0,
                    rho_1,
                    a_packed.as_ptr(),
                    b_packed.as_ptr(),
                    a_out_chunk,
                    b_out_chunk,
                    eq_lo,
                    q_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                    use_stream,
                )
            };
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            let p = {
                let mut acc = [F256Unreduced::ZERO; 8];
                let fold_q = |qg: usize| -> (F128, F128) {
                    let j0 = 2 * qg;
                    let j1 = j0 + 1;
                    if (j0 & pair_in_block_mask) >= useful_pairs_inclusive {
                        return (F128::ZERO, F128::ZERO);
                    }
                    let r0 = 4 * qg;
                    let a00 = table.fold_one_row(&a_packed[r0 * n_chunks..(r0 + 1) * n_chunks]);
                    let b00 = table.fold_one_row(&b_packed[r0 * n_chunks..(r0 + 1) * n_chunks]);
                    let a10 =
                        table.fold_one_row(&a_packed[(r0 + 1) * n_chunks..(r0 + 2) * n_chunks]);
                    let b10 =
                        table.fold_one_row(&b_packed[(r0 + 1) * n_chunks..(r0 + 2) * n_chunks]);
                    let a_j0 = a00.mle_fold(a10, rho_0);
                    let b_j0 = b00.mle_fold(b10, rho_0);
                    let (a_j1, b_j1) = if (j1 & pair_in_block_mask) < useful_pairs_inclusive {
                        let a01 =
                            table.fold_one_row(&a_packed[(r0 + 2) * n_chunks..(r0 + 3) * n_chunks]);
                        let b01 =
                            table.fold_one_row(&b_packed[(r0 + 2) * n_chunks..(r0 + 3) * n_chunks]);
                        let a11 =
                            table.fold_one_row(&a_packed[(r0 + 3) * n_chunks..(r0 + 4) * n_chunks]);
                        let b11 =
                            table.fold_one_row(&b_packed[(r0 + 3) * n_chunks..(r0 + 4) * n_chunks]);
                        (a01.mle_fold(a11, rho_0), b01.mle_fold(b11, rho_0))
                    } else {
                        (F128::ZERO, F128::ZERO)
                    };
                    (a_j0.mle_fold(a_j1, rho_1), b_j0.mle_fold(b_j1, rho_1))
                };
                for p_lo in 0..lo_size {
                    let q0l = 4 * p_lo;
                    let q0g = q_base + q0l;
                    let (a00, b00) = fold_q(q0g);
                    let (a10, b10) = fold_q(q0g + 1);
                    let (a01, b01) = fold_q(q0g + 2);
                    let (a11, b11) = fold_q(q0g + 3);
                    a_out_chunk[q0l] = a00;
                    a_out_chunk[q0l + 1] = a10;
                    a_out_chunk[q0l + 2] = a01;
                    a_out_chunk[q0l + 3] = a11;
                    b_out_chunk[q0l] = b00;
                    b_out_chunk[q0l + 1] = b10;
                    b_out_chunk[q0l + 2] = b01;
                    b_out_chunk[q0l + 3] = b11;
                    let eq_l = eq_lo[p_lo];
                    let da0 = a00 + a10;
                    let db0 = b00 + b10;
                    let da1 = a01 + a11;
                    let db1 = b01 + b11;
                    let da_0 = a00 + a01;
                    let db_0 = b00 + b01;
                    let da_1 = a10 + a11;
                    let db_1 = b10 + b11;
                    let da_i = da0 + da1;
                    let db_i = db0 + db1;
                    acc[0] ^= eq_l.mul_unreduced(a10 * b10);
                    acc[1] ^= eq_l.mul_unreduced(da0 * db0);
                    acc[2] ^= eq_l.mul_unreduced(a11 * b11);
                    acc[3] ^= eq_l.mul_unreduced(da1 * db1);
                    acc[4] ^= eq_l.mul_unreduced(a01 * b01);
                    acc[5] ^= eq_l.mul_unreduced(da_0 * db_0);
                    acc[6] ^= eq_l.mul_unreduced(da_1 * db_1);
                    acc[7] ^= eq_l.mul_unreduced(da_i * db_i);
                }
                acc.map(|u| u.reduce())
            };

            let eq_h = eq_hi[x_hi];
            p.map(|v| eq_h.mul_unreduced(v))
        })
        .reduce(
            || [F256Unreduced::ZERO; 8],
            |mut a, b| {
                for k in 0..8 {
                    a[k] ^= b[k];
                }
                a
            },
        );

    let [s10, si0, s11, si1, s01, t0, t1, ti] = sums.map(|u| u.reduce());
    let g4_1 = s10.mle_fold(s11, r1);
    let msg2_inf = si0.mle_fold(si1, r1);
    let msg2_1 = if r_next_round4[0] == F128::ONE {
        g4_1
    } else {
        r_next_round4[0] * g4_1
    };

    Round23Lookahead {
        msg2_1,
        msg2_inf,
        s01,
        s11,
        si1,
        t0,
        t1,
        ti,
    }
}

pub fn fold_2vars_and_compute_round_lookahead_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_0: F128,
    rho_1: F128,
    r_next: &[F128],
) -> Round23Lookahead {
    use rayon::prelude::*;

    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 64);
    let quarter = n / 4;
    assert_eq!(a_out.len(), quarter);
    assert_eq!(b_out.len(), quarter);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 2);

    let r1 = r_next[1];
    let eq = SplitEqGhash::new(&r_next[2..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 4, quarter);

    let chunk_in = 16 * lo_size;
    let chunk_out = 4 * lo_size;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;
    #[cfg(target_arch = "x86_64")]
    let use_stream = quarter >= (1 << 19)
        && ((a_out.as_ptr() as usize) & 15) == 0
        && ((b_out.as_ptr() as usize) & 15) == 0;

    let sums = a_out
        .par_chunks_mut(chunk_out)
        .zip(b_out.par_chunks_mut(chunk_out))
        .enumerate()
        .map(|(x_hi, (a_out_chunk, b_out_chunk))| {
            let a_in = &a[x_hi * chunk_in..(x_hi + 1) * chunk_in];
            let b_in = &b[x_hi * chunk_in..(x_hi + 1) * chunk_in];

            #[cfg(target_arch = "aarch64")]
            let p = unsafe {
                fold_2vars_and_lookahead_chunk_neon_unchecked(
                    a_in,
                    b_in,
                    a_out_chunk,
                    b_out_chunk,
                    rho_0,
                    rho_1,
                    eq_lo,
                )
            };
            #[cfg(target_arch = "x86_64")]
            let p = unsafe {
                fold_2vars_and_lookahead_chunk_x86_unchecked(
                    a_in,
                    b_in,
                    a_out_chunk,
                    b_out_chunk,
                    rho_0,
                    rho_1,
                    eq_lo,
                    use_stream,
                )
            };
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            let p = {
                let mut acc = [F256Unreduced::ZERO; 8];
                let fold_q = |slice: &[F128], base: usize| -> F128 {
                    let w0 = slice[base].mle_fold(slice[base + 1], rho_0);
                    let w1 = slice[base + 2].mle_fold(slice[base + 3], rho_0);
                    w0.mle_fold(w1, rho_1)
                };
                for p_lo in 0..lo_size {
                    let s = 16 * p_lo;
                    let d = 4 * p_lo;
                    let a00 = fold_q(a_in, s);
                    let b00 = fold_q(b_in, s);
                    let a10 = fold_q(a_in, s + 4);
                    let b10 = fold_q(b_in, s + 4);
                    let a01 = fold_q(a_in, s + 8);
                    let b01 = fold_q(b_in, s + 8);
                    let a11 = fold_q(a_in, s + 12);
                    let b11 = fold_q(b_in, s + 12);
                    a_out_chunk[d] = a00;
                    a_out_chunk[d + 1] = a10;
                    a_out_chunk[d + 2] = a01;
                    a_out_chunk[d + 3] = a11;
                    b_out_chunk[d] = b00;
                    b_out_chunk[d + 1] = b10;
                    b_out_chunk[d + 2] = b01;
                    b_out_chunk[d + 3] = b11;
                    let eq_l = eq_lo[p_lo];
                    let da0 = a00 + a10;
                    let db0 = b00 + b10;
                    let da1 = a01 + a11;
                    let db1 = b01 + b11;
                    let da_0 = a00 + a01;
                    let db_0 = b00 + b01;
                    let da_1 = a10 + a11;
                    let db_1 = b10 + b11;
                    let da_i = da0 + da1;
                    let db_i = db0 + db1;
                    acc[0] ^= eq_l.mul_unreduced(a10 * b10);
                    acc[1] ^= eq_l.mul_unreduced(da0 * db0);
                    acc[2] ^= eq_l.mul_unreduced(a11 * b11);
                    acc[3] ^= eq_l.mul_unreduced(da1 * db1);
                    acc[4] ^= eq_l.mul_unreduced(a01 * b01);
                    acc[5] ^= eq_l.mul_unreduced(da_0 * db_0);
                    acc[6] ^= eq_l.mul_unreduced(da_1 * db_1);
                    acc[7] ^= eq_l.mul_unreduced(da_i * db_i);
                }
                acc.map(|u| u.reduce())
            };

            let eq_h = eq_hi[x_hi];
            p.map(|v| eq_h.mul_unreduced(v))
        })
        .reduce(
            || [F256Unreduced::ZERO; 8],
            |mut a, b| {
                for k in 0..8 {
                    a[k] ^= b[k];
                }
                a
            },
        );

    let [s10, si0, s11, si1, s01, t0, t1, ti] = sums.map(|u| u.reduce());
    let g_1 = s10.mle_fold(s11, r1);
    let msg2_inf = si0.mle_fold(si1, r1);
    let msg2_1 = if r_next[0] == F128::ONE {
        g_1
    } else {
        r_next[0] * g_1
    };

    Round23Lookahead {
        msg2_1,
        msg2_inf,
        s01,
        s11,
        si1,
        t0,
        t1,
        ti,
    }
}

pub fn fold_2vars_and_compute_round_pair_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_0: F128,
    rho_1: F128,
    r_next: &[F128],
) -> (F128, F128) {
    use rayon::prelude::*;

    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 32);
    let quarter = n / 4;
    assert_eq!(a_out.len(), quarter);
    assert_eq!(b_out.len(), quarter);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 2);

    let eq = SplitEqGhash::new(&r_next[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, quarter);

    let chunk_in = 8 * lo_size;
    let chunk_out = 2 * lo_size;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;
    #[cfg(target_arch = "x86_64")]
    let use_stream = quarter >= (1 << 19)
        && ((a_out.as_ptr() as usize) & 15) == 0
        && ((b_out.as_ptr() as usize) & 15) == 0;

    let (sum1, sum_inf) = a_out
        .par_chunks_mut(chunk_out)
        .zip(b_out.par_chunks_mut(chunk_out))
        .enumerate()
        .map(|(x_hi, (a_out_chunk, b_out_chunk))| {
            let a_in = &a[x_hi * chunk_in..(x_hi + 1) * chunk_in];
            let b_in = &b[x_hi * chunk_in..(x_hi + 1) * chunk_in];

            #[cfg(target_arch = "aarch64")]
            let (p1, pinf) = unsafe {
                fold_2vars_and_round_pair_chunk_neon_unchecked(
                    a_in,
                    b_in,
                    a_out_chunk,
                    b_out_chunk,
                    rho_0,
                    rho_1,
                    eq_lo,
                )
            };
            #[cfg(target_arch = "x86_64")]
            let (p1, pinf) = unsafe {
                fold_2vars_and_round_pair_chunk_x86_unchecked(
                    a_in,
                    b_in,
                    a_out_chunk,
                    b_out_chunk,
                    rho_0,
                    rho_1,
                    eq_lo,
                    use_stream,
                )
            };
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            let (p1, pinf) = {
                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                let fold_q = |slice: &[F128], base: usize| -> F128 {
                    let w0 = slice[base].mle_fold(slice[base + 1], rho_0);
                    let w1 = slice[base + 2].mle_fold(slice[base + 3], rho_0);
                    w0.mle_fold(w1, rho_1)
                };
                for x_lo in 0..lo_size {
                    let s = 8 * x_lo;
                    let d = 2 * x_lo;
                    let a0 = fold_q(a_in, s);
                    let b0 = fold_q(b_in, s);
                    let a1 = fold_q(a_in, s + 4);
                    let b1 = fold_q(b_in, s + 4);
                    a_out_chunk[d] = a0;
                    a_out_chunk[d + 1] = a1;
                    b_out_chunk[d] = b0;
                    b_out_chunk[d + 1] = b1;
                    let eq_l = eq_lo[x_lo];
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                }
                (p1_acc.reduce(), pinf_acc.reduce())
            };

            let eq_h = eq_hi[x_hi];
            (eq_h.mul_unreduced(p1), eq_h.mul_unreduced(pinf))
        })
        .reduce(
            || (F256Unreduced::ZERO, F256Unreduced::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 ^ c1, sinf ^ cinf),
        );

    let s1 = sum1.reduce();
    let msg1 = if r_next[0] == F128::ONE {
        s1
    } else {
        r_next[0] * s1
    };
    (msg1, sum_inf.reduce())
}

pub fn fold_in_place_single(a: &mut Vec<F128>, challenge: F128) {
    let n = a.len();
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    for x in 0..half {
        let a0 = a[2 * x];
        let a1 = a[2 * x + 1];
        a[x] = a0.mle_fold(a1, challenge);
    }
    a.truncate(half);
}

pub fn fold_packed_witness_at_z(
    witness_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
) -> Vec<F128> {
    use rayon::prelude::*;
    assert_eq!(witness_packed.len(), (1usize << m) / 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let mut out = vec![F128::ZERO; n_out];
    out.par_iter_mut().enumerate().for_each(|(x_rest, slot)| {
        *slot = table.fold_one_row(&witness_packed[x_rest * n_chunks..(x_rest + 1) * n_chunks]);
    });
    out
}

pub fn fold_in_place_pair(a: &mut Vec<F128>, b: &mut Vec<F128>, challenge: F128) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    for x in 0..half {
        let a0 = a[2 * x];
        let a1 = a[2 * x + 1];
        let b0 = b[2 * x];
        let b1 = b[2 * x + 1];
        a[x] = a0.mle_fold(a1, challenge);
        b[x] = b0.mle_fold(b1, challenge);
    }
    a.truncate(half);
    b.truncate(half);
}

pub fn fold_and_compute_round_pair_optimized(
    a: &[F128],
    b: &[F128],
    r_fold: F128,
    r_next: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    let half = a.len() / 2;

    let mut a_new = crate::alloc_uninit_f128_vec(half);
    let mut b_new = crate::alloc_uninit_f128_vec(half);
    let (m1, mi) = fold_and_compute_round_pair_into(a, b, &mut a_new, &mut b_new, r_fold, r_next);
    (a_new, b_new, m1, mi)
}

pub fn fold_and_compute_round_pair_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    r_next: &[F128],
) -> (F128, F128) {
    use rayon::prelude::*;

    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 8);
    let half = n / 2;
    assert_eq!(a_out.len(), half);
    assert_eq!(b_out.len(), half);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 1);

    let eq = SplitEqGhash::new(&r_next[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 2, "fold_and_compute requires lo_size ≥ 2");

    assert_eq!(lo_size * hi_size * 2, half);

    let chunk_in = 4 * lo_size;
    let chunk_out = 2 * lo_size;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    let (sum1, sum_inf) = a_out
        .par_chunks_mut(chunk_out)
        .zip(b_out.par_chunks_mut(chunk_out))
        .enumerate()
        .map(|(x_hi, (a_out, b_out))| {
            let a_in = &a[x_hi * chunk_in..(x_hi + 1) * chunk_in];
            let b_in = &b[x_hi * chunk_in..(x_hi + 1) * chunk_in];

            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            let (p1, pinf) =
                unsafe { fold_and_message_x86_avx512(a_in, b_in, a_out, b_out, r_fold, eq_lo) };

            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "aarch64"),
                not(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "vpclmulqdq"
                ))
            ))]
            let (p1, pinf) = unsafe {
                assert!(lo_size & 1 == 0, "lo_size must be even");
                #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
                {
                    use core::arch::x86_64::*;
                    let a_in_ptr = a_in.as_ptr() as *const u64;
                    let b_in_ptr = b_in.as_ptr() as *const u64;
                    let a_out_ptr = a_out.as_mut_ptr() as *mut u64;
                    let b_out_ptr = b_out.as_mut_ptr() as *mut u64;
                    let eq_lo_ptr = eq_lo.as_ptr() as *const u64;
                    let vr = _mm_set_epi64x(r_fold.hi as i64, r_fold.lo as i64);
                    let (vr0, vr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vr);
                    let z = _mm_setzero_si128();
                    let v_one = _mm_set_epi64x(0, 1);
                    let (mut p1_ll, mut p1_mid, mut p1_hh) = (z, z, z);
                    let (mut pi_ll, mut pi_mid, mut pi_hh) = (z, z, z);
                    macro_rules! fold_step_x86 {
                        ($x_lo:expr, $store_fn:ident, $check_special:literal) => {{
                            let x_lo = $x_lo;
                            let s = 8 * x_lo;
                            let d = 4 * x_lo;
                            let ae0 = _mm_loadu_si128(a_in_ptr.add(s) as *const __m128i);
                            let ao0 = _mm_loadu_si128(a_in_ptr.add(s + 2) as *const __m128i);
                            let ae1 = _mm_loadu_si128(a_in_ptr.add(s + 4) as *const __m128i);
                            let ao1 = _mm_loadu_si128(a_in_ptr.add(s + 6) as *const __m128i);
                            let be0 = _mm_loadu_si128(b_in_ptr.add(s) as *const __m128i);
                            let bo0 = _mm_loadu_si128(b_in_ptr.add(s + 2) as *const __m128i);
                            let be1 = _mm_loadu_si128(b_in_ptr.add(s + 4) as *const __m128i);
                            let bo1 = _mm_loadu_si128(b_in_ptr.add(s + 6) as *const __m128i);
                            if $check_special {
                                let tail_or = _mm_or_si128(
                                    _mm_or_si128(ao0, _mm_or_si128(ae1, ao1)),
                                    _mm_or_si128(bo0, _mm_or_si128(be1, bo1)),
                                );
                                if _mm_testz_si128(tail_or, tail_or) != 0 {
                                    let head_or = _mm_or_si128(ae0, be0);
                                    if _mm_testz_si128(head_or, head_or) != 0 {
                                        $store_fn(a_out_ptr.add(d) as *mut __m128i, z);
                                        $store_fn(a_out_ptr.add(d + 2) as *mut __m128i, z);
                                        $store_fn(b_out_ptr.add(d) as *mut __m128i, z);
                                        $store_fn(b_out_ptr.add(d + 2) as *mut __m128i, z);
                                    } else {
                                        let (ap0, bp0) =
                                            crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                                ae0, be0, vr0, vr1, vr0, vr1,
                                            );
                                        let a0 = _mm_xor_si128(ae0, ap0);
                                        let b0 = _mm_xor_si128(be0, bp0);
                                        $store_fn(a_out_ptr.add(d) as *mut __m128i, a0);
                                        $store_fn(a_out_ptr.add(d + 2) as *mut __m128i, z);
                                        $store_fn(b_out_ptr.add(d) as *mut __m128i, b0);
                                        $store_fn(b_out_ptr.add(d + 2) as *mut __m128i, z);
                                        let gi = crate::gf2_kernels::k0_f128_mul_m128i(a0, b0);
                                        let eq = _mm_loadu_si128(
                                            eq_lo_ptr.add(2 * x_lo) as *const __m128i
                                        );
                                        crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                            eq,
                                            gi,
                                            &mut pi_ll,
                                            &mut pi_mid,
                                            &mut pi_hh,
                                        );
                                    }
                                } else {
                                    let b_ones_diff = _mm_or_si128(
                                        _mm_or_si128(
                                            _mm_xor_si128(be0, v_one),
                                            _mm_xor_si128(bo0, v_one),
                                        ),
                                        _mm_or_si128(
                                            _mm_xor_si128(be1, v_one),
                                            _mm_xor_si128(bo1, v_one),
                                        ),
                                    );
                                    if _mm_testz_si128(b_ones_diff, b_ones_diff) != 0 {
                                        let (ap0, ap1) =
                                            crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                                _mm_xor_si128(ae0, ao0),
                                                _mm_xor_si128(ae1, ao1),
                                                vr0,
                                                vr1,
                                                vr0,
                                                vr1,
                                            );
                                        let a0 = _mm_xor_si128(ae0, ap0);
                                        let a1 = _mm_xor_si128(ae1, ap1);
                                        $store_fn(a_out_ptr.add(d) as *mut __m128i, a0);
                                        $store_fn(a_out_ptr.add(d + 2) as *mut __m128i, a1);
                                        $store_fn(b_out_ptr.add(d) as *mut __m128i, v_one);
                                        $store_fn(b_out_ptr.add(d + 2) as *mut __m128i, v_one);
                                        let eq = _mm_loadu_si128(
                                            eq_lo_ptr.add(2 * x_lo) as *const __m128i
                                        );
                                        crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                            eq,
                                            a1,
                                            &mut p1_ll,
                                            &mut p1_mid,
                                            &mut p1_hh,
                                        );
                                    } else {
                                        let (ap0, ap1) =
                                            crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                                _mm_xor_si128(ae0, ao0),
                                                _mm_xor_si128(ae1, ao1),
                                                vr0,
                                                vr1,
                                                vr0,
                                                vr1,
                                            );
                                        let a0 = _mm_xor_si128(ae0, ap0);
                                        let a1 = _mm_xor_si128(ae1, ap1);
                                        let (bp0, bp1) =
                                            crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                                _mm_xor_si128(be0, bo0),
                                                _mm_xor_si128(be1, bo1),
                                                vr0,
                                                vr1,
                                                vr0,
                                                vr1,
                                            );
                                        let b0 = _mm_xor_si128(be0, bp0);
                                        let b1 = _mm_xor_si128(be1, bp1);
                                        $store_fn(a_out_ptr.add(d) as *mut __m128i, a0);
                                        $store_fn(a_out_ptr.add(d + 2) as *mut __m128i, a1);
                                        $store_fn(b_out_ptr.add(d) as *mut __m128i, b0);
                                        $store_fn(b_out_ptr.add(d + 2) as *mut __m128i, b1);
                                        let (g1, gi) = crate::gf2_kernels::k0_f128_mul_vec2_m128i(
                                            a1,
                                            _mm_xor_si128(a0, a1),
                                            b1,
                                            _mm_xor_si128(b0, b1),
                                        );
                                        let eq = _mm_loadu_si128(
                                            eq_lo_ptr.add(2 * x_lo) as *const __m128i
                                        );
                                        crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
                                            eq,
                                            g1,
                                            gi,
                                            &mut p1_ll,
                                            &mut p1_mid,
                                            &mut p1_hh,
                                            &mut pi_ll,
                                            &mut pi_mid,
                                            &mut pi_hh,
                                        );
                                    }
                                }
                            } else {
                                let (ap0, ap1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                    _mm_xor_si128(ae0, ao0),
                                    _mm_xor_si128(ae1, ao1),
                                    vr0,
                                    vr1,
                                    vr0,
                                    vr1,
                                );
                                let a0 = _mm_xor_si128(ae0, ap0);
                                let a1 = _mm_xor_si128(ae1, ap1);
                                let (bp0, bp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                    _mm_xor_si128(be0, bo0),
                                    _mm_xor_si128(be1, bo1),
                                    vr0,
                                    vr1,
                                    vr0,
                                    vr1,
                                );
                                let b0 = _mm_xor_si128(be0, bp0);
                                let b1 = _mm_xor_si128(be1, bp1);
                                $store_fn(a_out_ptr.add(d) as *mut __m128i, a0);
                                $store_fn(a_out_ptr.add(d + 2) as *mut __m128i, a1);
                                $store_fn(b_out_ptr.add(d) as *mut __m128i, b0);
                                $store_fn(b_out_ptr.add(d + 2) as *mut __m128i, b1);
                                let (g1, gi) = crate::gf2_kernels::k0_f128_mul_vec2_m128i(
                                    a1,
                                    _mm_xor_si128(a0, a1),
                                    b1,
                                    _mm_xor_si128(b0, b1),
                                );
                                let eq = _mm_loadu_si128(eq_lo_ptr.add(2 * x_lo) as *const __m128i);
                                crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
                                    eq,
                                    g1,
                                    gi,
                                    &mut p1_ll,
                                    &mut p1_mid,
                                    &mut p1_hh,
                                    &mut pi_ll,
                                    &mut pi_mid,
                                    &mut pi_hh,
                                );
                            }
                        }};
                    }
                    if half >= (1 << 19) {
                        if (lo_size & 31) == 0 {
                            let mut blk = 0usize;
                            while blk < lo_size {
                                fold_step_x86!(blk, _mm_stream_si128, true);
                                fold_step_x86!(blk + 1, _mm_stream_si128, true);
                                for x_lo in (blk + 2)..(blk + 30) {
                                    fold_step_x86!(x_lo, _mm_stream_si128, false);
                                }
                                fold_step_x86!(blk + 30, _mm_stream_si128, true);
                                fold_step_x86!(blk + 31, _mm_stream_si128, true);
                                blk += 32;
                            }
                        } else {
                            for x_lo in 0..lo_size {
                                fold_step_x86!(x_lo, _mm_stream_si128, false);
                            }
                        }
                        _mm_sfence();
                    } else {
                        for x_lo in 0..lo_size {
                            fold_step_x86!(x_lo, _mm_storeu_si128, false);
                        }
                    }
                    let p1 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(p1_ll, p1_mid, p1_hh);
                    let pi = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(pi_ll, pi_mid, pi_hh);
                    let eq_h = eq_hi[x_hi];
                    return (
                        eq_h.mul_unreduced(core::mem::transmute::<__m128i, F128>(p1)),
                        eq_h.mul_unreduced(core::mem::transmute::<__m128i, F128>(pi)),
                    );
                }
                #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
                {
                    use core::arch::aarch64::*;
                    let a_in_ptr = a_in.as_ptr() as *const u64;
                    let b_in_ptr = b_in.as_ptr() as *const u64;
                    let a_out_ptr = a_out.as_mut_ptr() as *mut u64;
                    let b_out_ptr = b_out.as_mut_ptr() as *mut u64;
                    let eq_lo_ptr = eq_lo.as_ptr() as *const u64;
                    let vr = vcombine_u64(vcreate_u64(r_fold.lo), vcreate_u64(r_fold.hi));
                    let (vr0, vr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vr);
                    let z = vdupq_n_u64(0);
                    let v_one = vcombine_u64(vcreate_u64(1), vcreate_u64(0));
                    let (mut p1_ll, mut p1_mid, mut p1_hh) = (z, z, z);
                    let (mut pi_ll, mut pi_mid, mut pi_hh) = (z, z, z);
                    macro_rules! fold_step_neon {
                        ($x_lo:expr, $check_special:literal) => {{
                            let x_lo = $x_lo;
                            let s = 8 * x_lo;
                            let d = 4 * x_lo;
                            let ae0 = vld1q_u64(a_in_ptr.add(s));
                            let ao0 = vld1q_u64(a_in_ptr.add(s + 2));
                            let ae1 = vld1q_u64(a_in_ptr.add(s + 4));
                            let ao1 = vld1q_u64(a_in_ptr.add(s + 6));
                            let be0 = vld1q_u64(b_in_ptr.add(s));
                            let bo0 = vld1q_u64(b_in_ptr.add(s + 2));
                            let be1 = vld1q_u64(b_in_ptr.add(s + 4));
                            let bo1 = vld1q_u64(b_in_ptr.add(s + 6));
                            if $check_special {
                                let tail_or = vorrq_u64(
                                    vorrq_u64(ao0, vorrq_u64(ae1, ao1)),
                                    vorrq_u64(bo0, vorrq_u64(be1, bo1)),
                                );
                                if (vgetq_lane_u64::<0>(tail_or) | vgetq_lane_u64::<1>(tail_or))
                                    == 0
                                {
                                    let head_or = vorrq_u64(ae0, be0);
                                    if (vgetq_lane_u64::<0>(head_or) | vgetq_lane_u64::<1>(head_or))
                                        == 0
                                    {
                                        vst1q_u64(a_out_ptr.add(d), z);
                                        vst1q_u64(a_out_ptr.add(d + 2), z);
                                        vst1q_u64(b_out_ptr.add(d), z);
                                        vst1q_u64(b_out_ptr.add(d + 2), z);
                                    } else {
                                        let (ap0, bp0) =
                                            crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                                                ae0, be0, vr0, vr1, vr0, vr1,
                                            );
                                        let a0 = veorq_u64(ae0, ap0);
                                        let b0 = veorq_u64(be0, bp0);
                                        vst1q_u64(a_out_ptr.add(d), a0);
                                        vst1q_u64(a_out_ptr.add(d + 2), z);
                                        vst1q_u64(b_out_ptr.add(d), b0);
                                        vst1q_u64(b_out_ptr.add(d + 2), z);
                                        let gi = crate::gf2_kernels::k0_f128_mul_neon(a0, b0);
                                        let eq = vld1q_u64(eq_lo_ptr.add(2 * x_lo));
                                        crate::gf2_kernels::k0_f128_mul_acc_neon(
                                            eq,
                                            gi,
                                            &mut pi_ll,
                                            &mut pi_mid,
                                            &mut pi_hh,
                                        );
                                    }
                                } else {
                                    let b_ones_diff = vorrq_u64(
                                        vorrq_u64(veorq_u64(be0, v_one), veorq_u64(bo0, v_one)),
                                        vorrq_u64(veorq_u64(be1, v_one), veorq_u64(bo1, v_one)),
                                    );
                                    if (vgetq_lane_u64::<0>(b_ones_diff)
                                        | vgetq_lane_u64::<1>(b_ones_diff))
                                        == 0
                                    {
                                        let (ap0, ap1) =
                                            crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                                                veorq_u64(ae0, ao0),
                                                veorq_u64(ae1, ao1),
                                                vr0,
                                                vr1,
                                                vr0,
                                                vr1,
                                            );
                                        let a0 = veorq_u64(ae0, ap0);
                                        let a1 = veorq_u64(ae1, ap1);
                                        vst1q_u64(a_out_ptr.add(d), a0);
                                        vst1q_u64(a_out_ptr.add(d + 2), a1);
                                        vst1q_u64(b_out_ptr.add(d), v_one);
                                        vst1q_u64(b_out_ptr.add(d + 2), v_one);
                                        let eq = vld1q_u64(eq_lo_ptr.add(2 * x_lo));
                                        crate::gf2_kernels::k0_f128_mul_acc_neon(
                                            eq,
                                            a1,
                                            &mut p1_ll,
                                            &mut p1_mid,
                                            &mut p1_hh,
                                        );
                                    } else {
                                        let (ap0, ap1) =
                                            crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                                                veorq_u64(ae0, ao0),
                                                veorq_u64(ae1, ao1),
                                                vr0,
                                                vr1,
                                                vr0,
                                                vr1,
                                            );
                                        let a0 = veorq_u64(ae0, ap0);
                                        let a1 = veorq_u64(ae1, ap1);
                                        let (bp0, bp1) =
                                            crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                                                veorq_u64(be0, bo0),
                                                veorq_u64(be1, bo1),
                                                vr0,
                                                vr1,
                                                vr0,
                                                vr1,
                                            );
                                        let b0 = veorq_u64(be0, bp0);
                                        let b1 = veorq_u64(be1, bp1);
                                        vst1q_u64(a_out_ptr.add(d), a0);
                                        vst1q_u64(a_out_ptr.add(d + 2), a1);
                                        vst1q_u64(b_out_ptr.add(d), b0);
                                        vst1q_u64(b_out_ptr.add(d + 2), b1);
                                        let (g1, gi) = crate::gf2_kernels::k0_f128_mul_vec2_neon(
                                            a1,
                                            veorq_u64(a0, a1),
                                            b1,
                                            veorq_u64(b0, b1),
                                        );
                                        let eq = vld1q_u64(eq_lo_ptr.add(2 * x_lo));
                                        crate::gf2_kernels::k0_f128_mul_acc_pair_neon(
                                            eq,
                                            g1,
                                            gi,
                                            &mut p1_ll,
                                            &mut p1_mid,
                                            &mut p1_hh,
                                            &mut pi_ll,
                                            &mut pi_mid,
                                            &mut pi_hh,
                                        );
                                    }
                                }
                            } else {
                                let (ap0, ap1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                                    veorq_u64(ae0, ao0),
                                    veorq_u64(ae1, ao1),
                                    vr0,
                                    vr1,
                                    vr0,
                                    vr1,
                                );
                                let a0 = veorq_u64(ae0, ap0);
                                let a1 = veorq_u64(ae1, ap1);
                                let (bp0, bp1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                                    veorq_u64(be0, bo0),
                                    veorq_u64(be1, bo1),
                                    vr0,
                                    vr1,
                                    vr0,
                                    vr1,
                                );
                                let b0 = veorq_u64(be0, bp0);
                                let b1 = veorq_u64(be1, bp1);
                                vst1q_u64(a_out_ptr.add(d), a0);
                                vst1q_u64(a_out_ptr.add(d + 2), a1);
                                vst1q_u64(b_out_ptr.add(d), b0);
                                vst1q_u64(b_out_ptr.add(d + 2), b1);
                                let (g1, gi) = crate::gf2_kernels::k0_f128_mul_vec2_neon(
                                    a1,
                                    veorq_u64(a0, a1),
                                    b1,
                                    veorq_u64(b0, b1),
                                );
                                let eq = vld1q_u64(eq_lo_ptr.add(2 * x_lo));
                                crate::gf2_kernels::k0_f128_mul_acc_pair_neon(
                                    eq,
                                    g1,
                                    gi,
                                    &mut p1_ll,
                                    &mut p1_mid,
                                    &mut p1_hh,
                                    &mut pi_ll,
                                    &mut pi_mid,
                                    &mut pi_hh,
                                );
                            }
                        }};
                    }
                    if half >= (1 << 18) && (lo_size & 31) == 0 {
                        let mut blk = 0usize;
                        while blk < lo_size {
                            fold_step_neon!(blk, true);
                            fold_step_neon!(blk + 1, true);
                            for x_lo in (blk + 2)..(blk + 30) {
                                fold_step_neon!(x_lo, false);
                            }
                            fold_step_neon!(blk + 30, true);
                            fold_step_neon!(blk + 31, true);
                            blk += 32;
                        }
                    } else {
                        for x_lo in 0..lo_size {
                            fold_step_neon!(x_lo, false);
                        }
                    }
                    let p1 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(p1_ll, p1_mid, p1_hh);
                    let pi = crate::gf2_kernels::k0_f128_reduce_3lane_neon(pi_ll, pi_mid, pi_hh);
                    let eq_h = eq_hi[x_hi];
                    return (
                        eq_h.mul_unreduced(F128 {
                            lo: vgetq_lane_u64::<0>(p1),
                            hi: vgetq_lane_u64::<1>(p1),
                        }),
                        eq_h.mul_unreduced(F128 {
                            lo: vgetq_lane_u64::<0>(pi),
                            hi: vgetq_lane_u64::<1>(pi),
                        }),
                    );
                }
                #[allow(unreachable_code)]
                let (p1_lo, p1_hi, pinf_lo, pinf_hi) =
                    crate::gf2_kernels::k0_f128_fold_and_msg_round_slice(
                        a_in.as_ptr() as *const u64,
                        b_in.as_ptr() as *const u64,
                        a_out.as_mut_ptr() as *mut u64,
                        b_out.as_mut_ptr() as *mut u64,
                        eq_lo.as_ptr() as *const u64,
                        lo_size,
                        r_fold.lo,
                        r_fold.hi,
                    );
                #[allow(unreachable_code)]
                (
                    F128 {
                        lo: p1_lo,
                        hi: p1_hi,
                    },
                    F128 {
                        lo: pinf_lo,
                        hi: pinf_hi,
                    },
                )
            };

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let (p1, pinf) = {
                crate::field::f128_slice::fold_pairs(a_in, 0, a_out, r_fold);
                crate::field::f128_slice::fold_pairs(b_in, 0, b_out, r_fold);

                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                for x_lo in 0..lo_size {
                    let o = 2 * x_lo;
                    let a0 = a_out[o];
                    let a1 = a_out[o + 1];
                    let b0 = b_out[o];
                    let b1 = b_out[o + 1];
                    let eq_l = eq_lo[x_lo];
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                }
                (p1_acc.reduce(), pinf_acc.reduce())
            };
            let eq_h = eq_hi[x_hi];
            (eq_h.mul_unreduced(p1), eq_h.mul_unreduced(pinf))
        })
        .reduce(
            || (F256Unreduced::ZERO, F256Unreduced::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 ^ c1, sinf ^ cinf),
        );

    let s1 = sum1.reduce();
    let msg1 = if r_next[0] == F128::ONE {
        s1
    } else {
        r_next[0] * s1
    };
    (msg1, sum_inf.reduce())
}

#[cfg(test)]
fn uni_skip_fold_and_round_pair_optimized_packed_serial(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let mut a_folded = vec![F128::ZERO; n_out];
    let mut b_folded = vec![F128::ZERO; n_out];
    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let mut sum1 = F128::ZERO;
    let mut sum_inf = F128::ZERO;
    for x_hi in 0..hi_size {
        let mut p1_acc = F256Unreduced::ZERO;
        let mut pinf_acc = F256Unreduced::ZERO;
        let k_base = x_hi << eq.n_lo;
        for x_lo in 0..lo_size {
            let k = k_base | x_lo;
            let x0 = 2 * k;
            let x1 = x0 + 1;
            let a0 = table.fold_one_row(&a_packed[x0 * n_chunks..(x0 + 1) * n_chunks]);
            let b0 = table.fold_one_row(&b_packed[x0 * n_chunks..(x0 + 1) * n_chunks]);
            let a1 = table.fold_one_row(&a_packed[x1 * n_chunks..(x1 + 1) * n_chunks]);
            let b1 = table.fold_one_row(&b_packed[x1 * n_chunks..(x1 + 1) * n_chunks]);
            a_folded[x0] = a0;
            b_folded[x0] = b0;
            a_folded[x1] = a1;
            b_folded[x1] = b1;
            let eq_l = eq.lo[x_lo];
            let g1 = a1 * b1;
            p1_acc ^= eq_l.mul_unreduced(g1);
            let g_inf = (a0 + a1) * (b0 + b1);
            pinf_acc ^= eq_l.mul_unreduced(g_inf);
        }
        let p1 = p1_acc.reduce();
        let pinf = pinf_acc.reduce();
        sum1 += eq.hi[x_hi] * p1;
        sum_inf += eq.hi[x_hi] * pinf;
    }
    (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf)
}

pub fn uni_skip_fold_and_round_pair_optimized(
    a: &[bool],
    b: &[bool],
    m: usize,
    k_skip: usize,
    z: F128,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    let a_packed = pack_bits(a);
    let b_packed = pack_bits(b);
    let table = UniSkipFoldTable::new(k_skip, z);
    uni_skip_fold_and_round_pair_optimized_packed(
        &a_packed,
        &b_packed,
        m,
        k_skip,
        &table,
        mlv_challenges,
    )
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
        fn bit(&mut self) -> bool {
            (self.next_u64() & 1) != 0
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.bit()).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    #[test]
    fn lagrange_weights_sum_to_one() {
        let mut rng = Rng::new(1);
        for &k_skip in &[1usize, 2, 3, 4, 5, 6] {
            for _ in 0..4 {
                let z = rng.f128();
                let weights = lagrange_weights_naive(k_skip, z);
                let sum: F128 = weights.iter().copied().fold(F128::ZERO, |a, b| a + b);
                assert_eq!(sum, F128::ONE, "Σ L_i ≠ 1 at k_skip={k_skip}");
            }
        }
    }

    #[test]
    fn lagrange_at_node_is_indicator() {
        for k_skip in [2usize, 3, 4, 5] {
            let ell = 1usize << k_skip;
            for i in 0..ell {
                let z = PHI_8_TABLE[i];
                let weights = lagrange_weights_naive(k_skip, z);
                for j in 0..ell {
                    let expected = if j == i { F128::ONE } else { F128::ZERO };
                    assert_eq!(weights[j], expected, "k_skip={k_skip}, z=node{i}, j={j}");
                }
            }
        }
    }

    #[test]
    fn fold_at_node_recovers_witness_slice() {
        let m = 8;
        let k_skip = 3;
        let ell = 1usize << k_skip;
        let n_rest = 1usize << (m - k_skip);
        let mut rng = Rng::new(7);
        let a = rng.bits(1 << m);
        for i in 0..ell {
            let z = PHI_8_TABLE[i];
            let weights = lagrange_weights_naive(k_skip, z);
            let a_mlv = fold_at_z_naive(&a, m, k_skip, &weights);
            for x_rest in 0..n_rest {
                let expected = if a[x_rest * ell + i] {
                    F128::ONE
                } else {
                    F128::ZERO
                };
                assert_eq!(
                    a_mlv[x_rest], expected,
                    "fold at node {i} mismatch at x_rest={x_rest}"
                );
            }
        }
    }

    #[test]
    fn fold_is_xor_linear() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(11);
        let a = rng.bits(1 << m);
        let aprime = rng.bits(1 << m);
        let a_xor: Vec<bool> = a.iter().zip(&aprime).map(|(x, y)| x ^ y).collect();
        let z = rng.f128();
        let weights = lagrange_weights_naive(k_skip, z);

        let fa = fold_at_z_naive(&a, m, k_skip, &weights);
        let fap = fold_at_z_naive(&aprime, m, k_skip, &weights);
        let fxor = fold_at_z_naive(&a_xor, m, k_skip, &weights);
        for i in 0..fa.len() {
            assert_eq!(fa[i] + fap[i], fxor[i], "linearity broken at i={i}");
        }
    }

    #[test]
    fn zero_witness_gives_zero_round_message() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(20);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let zeros = vec![false; 1 << m];
        let (a_mlv, b_mlv, msg_1, msg_inf) =
            uni_skip_fold_and_round_pair_naive(&zeros, &zeros, m, k_skip, z, &mlv_challenges);
        assert!(a_mlv.iter().all(|v| v.is_zero()));
        assert!(b_mlv.iter().all(|v| v.is_zero()));
        assert_eq!(msg_1, F128::ZERO);
        assert_eq!(msg_inf, F128::ZERO);
    }

    #[test]
    fn deterministic() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(33);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let o1 = uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        let o2 = uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        assert_eq!(o1, o2);
    }

    #[test]
    fn round_pair_symmetric_in_ab() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(40);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let (_, _, m1_ab, minf_ab) =
            uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        let (_, _, m1_ba, minf_ba) =
            uni_skip_fold_and_round_pair_naive(&b, &a, m, k_skip, z, &mlv_challenges);
        assert_eq!(m1_ab, m1_ba);
        assert_eq!(minf_ab, minf_ba);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fold_one_row_neon_matches_scalar() {
        let k_skip = 6;
        let mut rng = Rng::new(70);
        let z = rng.f128();
        let table = UniSkipFoldTable::new(k_skip, z);

        for _ in 0..256 {
            let mut bytes = [0u8; 8];
            for byte in bytes.iter_mut() {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            let scalar = table.fold_one_row(&bytes);

            let neon = unsafe {
                fold_one_row_neon_unchecked_8(table.data.as_ptr() as *const u8, bytes.as_ptr())
            };
            assert_eq!(scalar, neon, "fold mismatch bytes={bytes:02x?}");
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold_round2_pair_x86_matches_scalar() {
        let mut rng = Rng::new(71);
        let table = UniSkipFoldTable::new(6, rng.f128());

        for _ in 0..256 {
            let mut rows = [[0u8; 8]; 4];
            for row in &mut rows {
                for byte in row {
                    *byte = (rng.next_u64() & 0xff) as u8;
                }
            }
            let expected = rows.map(|row| table.fold_one_row(&row));

            let actual = unsafe {
                fold_round2_pair_x86_unchecked_8(
                    table.data.as_ptr(),
                    rows[0].as_ptr(),
                    rows[1].as_ptr(),
                    rows[2].as_ptr(),
                    rows[3].as_ptr(),
                )
            };
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn fold_in_place_pair_matches_formula() {
        let mut rng = Rng::new(300);
        for &log_n in &[1usize, 2, 3, 4, 6] {
            let n = 1usize << log_n;
            let a_orig: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b_orig: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let challenge = rng.f128();

            let mut a = a_orig.clone();
            let mut b = b_orig.clone();
            fold_in_place_pair(&mut a, &mut b, challenge);

            assert_eq!(a.len(), n / 2);
            assert_eq!(b.len(), n / 2);
            for x in 0..(n / 2) {
                let a0 = a_orig[2 * x];
                let a1 = a_orig[2 * x + 1];
                let b0 = b_orig[2 * x];
                let b1 = b_orig[2 * x + 1];
                assert_eq!(a[x], a0 + challenge * (a1 + a0), "log_n={log_n}, x={x}");
                assert_eq!(b[x], b0 + challenge * (b1 + b0), "log_n={log_n}, x={x}");
            }
        }
    }

    #[test]
    fn c_eval_from_round1_c_matches_s0() {
        use crate::field::F8;
        use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
        use crate::zerocheck::univariate_skip_optimized::{
            c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed,
            small_challenges_ghash,
        };

        const K_SKIP: usize = 6;
        const N_INNER: usize = 7;

        for &m in &[14usize, 15, 16] {
            let mut rng = Rng::new(500 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);

            let mut r = vec![F128::ZERO; m];
            for slot in r[..K_SKIP].iter_mut() {
                *slot = rng.f128();
            }
            for (i, v) in small_challenges_ghash().iter().enumerate() {
                r[K_SKIP + i] = *v;
            }
            for (i, v) in medium_challenges_ghash().iter().enumerate() {
                r[K_SKIP + 3 + i] = *v;
            }
            for slot in r[K_SKIP + N_INNER..].iter_mut() {
                *slot = rng.f128();
            }
            let z = rng.f128();

            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);
            let c_packed = pack_bits(&c);

            let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
            let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            let (_round1_ab, round1_c) = round1_shift_reduce_extract_c_packed(
                &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &inv_table,
            );

            let c_eval_via_interpolation =
                c_s_f128() * interpolate_at_z_on_lambda(&round1_c, K_SKIP, z);

            let weights = lagrange_weights_naive(K_SKIP, z);
            let mut c_mlv = fold_at_z_naive(&c, m, K_SKIP, &weights);
            for &r_val in &r[K_SKIP..] {
                fold_in_place_single(&mut c_mlv, r_val);
            }
            assert_eq!(c_mlv.len(), 1);
            let c_eval_via_fold = c_mlv[0];

            assert_eq!(
                c_eval_via_interpolation, c_eval_via_fold,
                "c-claim identity broken at m={m}"
            );
        }
    }

    #[test]
    fn fused_round_matches_unfused() {
        let mut rng = Rng::new(310);

        for &log_n in &[10usize, 11, 12] {
            let n = 1usize << log_n;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let r_fold = rng.f128();
            let r_next = rng.f128_vec(log_n - 1);

            let (a_fused, b_fused, m1_fused, minf_fused) =
                fold_and_compute_round_pair_optimized(&a, &b, r_fold, &r_next);

            let mut a_unf = a.clone();
            let mut b_unf = b.clone();
            fold_in_place_pair(&mut a_unf, &mut b_unf, r_fold);
            let (m1_unf, minf_unf) = round_pair_naive(&a_unf, &b_unf, &r_next);

            assert_eq!(a_fused, a_unf, "a mismatch at log_n={log_n}");
            assert_eq!(b_fused, b_unf, "b mismatch at log_n={log_n}");
            assert_eq!(m1_fused, m1_unf, "msg_1 mismatch at log_n={log_n}");
            assert_eq!(minf_fused, minf_unf, "msg_inf mismatch at log_n={log_n}");
        }
    }

    #[test]
    fn parallel_matches_serial() {
        for &m in &[7usize, 8, 9, 10] {
            let k_skip = 6;
            if m <= k_skip {
                continue;
            }
            let mut rng = Rng::new(200 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - k_skip);
            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);
            let table = UniSkipFoldTable::new(k_skip, z);

            let par = uni_skip_fold_and_round_pair_optimized_packed(
                &a_packed,
                &b_packed,
                m,
                k_skip,
                &table,
                &mlv_challenges,
            );
            let ser = uni_skip_fold_and_round_pair_optimized_packed_serial(
                &a_packed,
                &b_packed,
                m,
                k_skip,
                &table,
                &mlv_challenges,
            );

            assert_eq!(par.0, ser.0, "a_mlv mismatch at m={m}");
            assert_eq!(par.1, ser.1, "b_mlv mismatch at m={m}");
            assert_eq!(par.2, ser.2, "msg_1 mismatch at m={m}");
            assert_eq!(par.3, ser.3, "msg_inf mismatch at m={m}");
        }
    }

    #[test]
    fn uni_skip_fold_round_pair_padded_matches_dense() {
        const K_SKIP: usize = 6;
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xFADE_F00D_u64.wrapping_add((k_log * 31 + m) as u64));
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;
            let n_blocks = 1usize << (m - k_log);

            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    a[blk * block_size + j] = false;
                    b[blk * block_size + j] = false;
                }
            }
            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);

            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - K_SKIP);
            let table = UniSkipFoldTable::new(K_SKIP, z);
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };

            let dense = uni_skip_fold_and_round_pair_optimized_packed(
                &a_packed,
                &b_packed,
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
            );
            let padded = uni_skip_fold_and_round_pair_optimized_packed_padded(
                &a_packed,
                &b_packed,
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
                &padding,
            );
            assert_eq!(
                dense.0, padded.0,
                "a_mlv: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.1, padded.1,
                "b_mlv: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.2, padded.2,
                "msg_1: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.3, padded.3,
                "msg_inf: m={m}, k_log={k_log}, useful={useful_bits}"
            );
        }
    }

    #[test]
    fn fold_table_one_row_matches_direct_lagrange() {
        let m = 8;
        let k_skip = 3;
        let mut rng = Rng::new(60);
        let z = rng.f128();
        let a = rng.bits(1 << m);
        let weights = lagrange_weights_naive(k_skip, z);
        let table = UniSkipFoldTable::new(k_skip, z);
        let a_packed = pack_bits(&a);

        let n_chunks = 1usize << (k_skip / 8);
        let _ = n_chunks;
        let n_chunks = table.n_chunks;

        for x_rest in 0..(1usize << (m - k_skip)) {
            let direct = {
                let mut acc = F128::ZERO;
                for s in 0..(1usize << k_skip) {
                    if a[x_rest * (1usize << k_skip) + s] {
                        acc += weights[s];
                    }
                }
                acc
            };
            let via_table =
                table.fold_one_row(&a_packed[x_rest * n_chunks..(x_rest + 1) * n_chunks]);
            assert_eq!(via_table, direct, "x_rest={x_rest}");
        }
    }

    #[test]
    fn optimized_matches_naive() {
        for &m in &[7usize, 8, 9, 10] {
            let k_skip = 6;
            if m <= k_skip {
                continue;
            }
            let mut rng = Rng::new(100 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - k_skip);

            let (a_n, b_n, m1_n, minf_n) =
                uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
            let (a_o, b_o, m1_o, minf_o) =
                uni_skip_fold_and_round_pair_optimized(&a, &b, m, k_skip, z, &mlv_challenges);

            assert_eq!(a_n, a_o, "a_mlv mismatch at m={m}");
            assert_eq!(b_n, b_o, "b_mlv mismatch at m={m}");
            assert_eq!(m1_n, m1_o, "msg_1 mismatch at m={m}");
            assert_eq!(minf_n, minf_o, "msg_inf mismatch at m={m}");
        }
    }

    #[test]
    fn round_pair_message_has_degree_two() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(55);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let r = rng.f128_vec(m - k_skip);

        let weights = lagrange_weights_naive(k_skip, z);
        let a_mlv = fold_at_z_naive(&a, m, k_skip, &weights);
        let b_mlv = fold_at_z_naive(&b, m, k_skip, &weights);

        let n = a_mlv.len();
        let half = n / 2;
        let eq_remaining = build_eq(&r[1..]);

        let mut g0 = F128::ZERO;
        let mut g1 = F128::ZERO;
        let mut g_inf = F128::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let eq_x = eq_remaining[x_prime];
            g0 += eq_x * a0 * b0;
            g1 += eq_x * a1 * b1;
            g_inf += eq_x * (a0 + a1) * (b0 + b1);
        }

        let (msg_1, msg_inf) = round_pair_naive(&a_mlv, &b_mlv, &r);
        assert_eq!(msg_1, r[0] * g1);
        assert_eq!(msg_inf, g_inf);

        let x = rng.f128();
        let g_via_poly = g0 + x * (g0 + g1) + x * (x + F128::ONE) * g_inf;
        let mut g_via_sum = F128::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let a_x = a0 + x * (a0 + a1);
            let b_x = b0 + x * (b0 + b1);
            g_via_sum += eq_remaining[x_prime] * a_x * b_x;
        }
        assert_eq!(g_via_poly, g_via_sum);
    }
}

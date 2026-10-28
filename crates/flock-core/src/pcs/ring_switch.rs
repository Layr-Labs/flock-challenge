use crate::bits::transpose_8x8_bits;
use crate::challenger::Challenger;
use crate::field::F128;
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::multilinear::lagrange_weights_naive;
use crate::zerocheck::univariate_skip::build_eq;
use serde::{Deserialize, Serialize};

use super::pack::LOG_PACKING;

#[derive(Clone, Copy, Debug)]
struct ChunkPadding {
    chunk_in_block_mask: usize,
    useful_chunks_per_block: usize,
}

impl ChunkPadding {
    fn new(padding: &PaddingSpec, chunk_width: usize) -> Self {
        if padding.k_log <= LOG_PACKING {
            return Self::no_skip();
        }
        let block_size_f128 = 1usize << (padding.k_log - LOG_PACKING);
        if block_size_f128 < chunk_width {
            return Self::no_skip();
        }
        let chunks_per_block = block_size_f128 / chunk_width;
        let useful_f128 = padding.useful_bits_per_block.div_ceil(1 << LOG_PACKING);
        let useful_chunks_per_block = useful_f128.div_ceil(chunk_width).min(chunks_per_block);
        if useful_chunks_per_block == chunks_per_block {
            return Self::no_skip();
        }
        debug_assert!(chunks_per_block.is_power_of_two());
        Self {
            chunk_in_block_mask: chunks_per_block - 1,
            useful_chunks_per_block,
        }
    }

    fn no_skip() -> Self {
        Self {
            chunk_in_block_mask: usize::MAX,
            useful_chunks_per_block: usize::MAX,
        }
    }

    #[inline(always)]
    fn skip(&self, chunk_idx: usize) -> bool {
        (chunk_idx & self.chunk_in_block_mask) >= self.useful_chunks_per_block
    }
}

pub fn build_claim_weights(z_skip: F128, x_outer_0: F128) -> Vec<F128> {
    const K_SKIP: usize = 6;
    let lambda = lagrange_weights_naive(K_SKIP, z_skip);
    debug_assert_eq!(lambda.len(), 1 << K_SKIP);

    let n = 1 << LOG_PACKING;
    let mut weights = vec![F128::ZERO; n];
    for i_lo in 0..64 {
        let v = lambda[i_lo];
        let vr = v * x_outer_0;
        weights[i_lo] = v + vr;
        weights[i_lo + 64] = vr;
    }
    weights
}

pub fn fold_1b_rows_multi(packed_witness: &[F128], suffix_tensors: &[&[F128]]) -> Vec<Vec<F128>> {
    let m = LOG_PACKING + (packed_witness.len().trailing_zeros() as usize);
    fold_1b_rows_multi_padded(packed_witness, suffix_tensors, &PaddingSpec::dense(m))
}

pub fn fold_1b_rows_multi_padded(
    packed_witness: &[F128],
    suffix_tensors: &[&[F128]],
    padding: &PaddingSpec,
) -> Vec<Vec<F128>> {
    use rayon::prelude::*;
    let k = suffix_tensors.len();
    let n = 1 << LOG_PACKING;
    assert!(
        suffix_tensors
            .iter()
            .all(|t| t.len() == packed_witness.len())
    );

    let zero_acc = || vec![vec![F128::ZERO; n]; k];

    if k == 2 {
        if packed_witness.len().is_multiple_of(16) {
            #[cfg(target_arch = "x86_64")]
            {
                let (a0, a1) = fold_1b_rows_2way_mfr_16wide_padded(
                    packed_witness,
                    suffix_tensors[0],
                    suffix_tensors[1],
                    padding,
                );
                return vec![a0, a1];
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let a0 =
                    fold_1b_rows_1way_mfr_16wide_padded(packed_witness, suffix_tensors[0], padding);
                let a1 =
                    fold_1b_rows_1way_mfr_16wide_padded(packed_witness, suffix_tensors[1], padding);
                return vec![a0, a1];
            }
        }
        if packed_witness.len().is_multiple_of(8) {
            let (a0, a1) = fold_1b_rows_2way_mfr_8wide_padded(
                packed_witness,
                suffix_tensors[0],
                suffix_tensors[1],
                padding,
            );
            return vec![a0, a1];
        }
        if packed_witness.len().is_multiple_of(4) {
            let (a0, a1) = fold_1b_rows_2way_mfr_padded(
                packed_witness,
                suffix_tensors[0],
                suffix_tensors[1],
                padding,
            );
            return vec![a0, a1];
        }
    }

    packed_witness
        .par_iter()
        .enumerate()
        .fold(zero_acc, |mut acc, (i_rest, elem)| {
            let mut lo = elem.lo;
            while lo != 0 {
                let r = lo.trailing_zeros() as usize;
                for (j, t) in suffix_tensors.iter().enumerate() {
                    acc[j][r] += t[i_rest];
                }
                lo &= lo - 1;
            }
            let mut hi = elem.hi;
            while hi != 0 {
                let r = hi.trailing_zeros() as usize;
                for (j, t) in suffix_tensors.iter().enumerate() {
                    acc[j][64 | r] += t[i_rest];
                }
                hi &= hi - 1;
            }
            acc
        })
        .reduce(zero_acc, |mut a, b| {
            for (av, bv) in a.iter_mut().zip(b.iter()) {
                for (avi, bvi) in av.iter_mut().zip(bv.iter()) {
                    *avi += *bvi;
                }
            }
            a
        })
}

pub(crate) fn build_eq_parallel(r: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    let n = r.len();
    let mut t = crate::alloc_uninit_f128_vec(1usize << n);
    t[0] = F128::ONE;
    if n == 0 {
        return t;
    }
    let r0 = r[0];
    t[1] = r0;
    t[0] = F128::ONE + r0;
    const PAR_THRESHOLD: usize = 1 << 12;
    const CHUNK: usize = 4096;
    for (i, &r_i) in r.iter().enumerate().skip(1) {
        let half = 1usize << i;
        let (lo, hi_rest) = t.split_at_mut(half);
        let hi = &mut hi_rest[..half];
        if half < PAR_THRESHOLD {
            unsafe {
                crate::gf2_kernels::k0_f128_eq_expand_slice(
                    lo.as_mut_ptr() as *mut u64,
                    hi.as_mut_ptr() as *mut u64,
                    half,
                    r_i.lo,
                    r_i.hi,
                );
            }
        } else {
            lo.par_chunks_mut(CHUNK)
                .zip(hi.par_chunks_mut(CHUNK))
                .for_each(|(lo_c, hi_c)| unsafe {
                    crate::gf2_kernels::k0_f128_eq_expand_slice(
                        lo_c.as_mut_ptr() as *mut u64,
                        hi_c.as_mut_ptr() as *mut u64,
                        lo_c.len(),
                        r_i.lo,
                        r_i.hi,
                    );
                });
        }
    }
    t
}

pub fn build_eq_split(r: &[F128], n_lo: usize) -> (Vec<F128>, Vec<F128>) {
    assert!(n_lo <= r.len());
    let eq_lo = build_eq_parallel(&r[..n_lo]);
    let eq_hi = build_eq_parallel(&r[n_lo..]);
    (eq_lo, eq_hi)
}

pub fn split_n_lo(n: usize) -> usize {
    (n / 2).clamp(4, n)
}

#[inline(always)]
fn subset_sums_4(elems: [F128; 4]) -> [F128; 16] {
    let mut sums = [F128::ZERO; 16];
    for (i, &e) in elems.iter().enumerate() {
        let span_log = i + 1;
        let half = 1 << i;
        for k in 0..half {
            sums[half + k] = sums[k] + e;
        }
        let _ = span_log;
    }
    sums
}

pub fn fold_1b_rows_2way_mfr(
    packed_witness: &[F128],
    t0: &[F128],
    t1: &[F128],
) -> (Vec<F128>, Vec<F128>) {
    let m = LOG_PACKING + (packed_witness.len().trailing_zeros() as usize);
    fold_1b_rows_2way_mfr_padded(packed_witness, t0, t1, &PaddingSpec::dense(m))
}

pub fn fold_1b_rows_2way_mfr_padded(
    packed_witness: &[F128],
    t0: &[F128],
    t1: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    assert_eq!(t0.len(), packed_witness.len());
    assert_eq!(t1.len(), packed_witness.len());
    assert!(
        packed_witness.len().is_multiple_of(4),
        "fold_1b_rows_2way_mfr requires len divisible by 4 (got {})",
        packed_witness.len()
    );
    let skip = ChunkPadding::new(padding, 4);

    let pair = packed_witness
        .par_chunks(4)
        .zip(t0.par_chunks(4))
        .zip(t1.par_chunks(4))
        .enumerate()
        .fold(
            || (vec![F128::ZERO; n], vec![F128::ZERO; n]),
            |(mut a0, mut a1), (chunk_idx, ((m_chunk, t0_chunk), t1_chunk))| {
                if skip.skip(chunk_idx) {
                    return (a0, a1);
                }
                let v0: [F128; 4] = [t0_chunk[0], t0_chunk[1], t0_chunk[2], t0_chunk[3]];
                let v1: [F128; 4] = [t1_chunk[0], t1_chunk[1], t1_chunk[2], t1_chunk[3]];

                let lookup0 = subset_sums_4(v0);
                let lookup1 = subset_sums_4(v1);

                let m_bytes: [[u8; 16]; 4] = [
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m_chunk[0].lo.to_le_bytes());
                        b[8..].copy_from_slice(&m_chunk[0].hi.to_le_bytes());
                        b
                    },
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m_chunk[1].lo.to_le_bytes());
                        b[8..].copy_from_slice(&m_chunk[1].hi.to_le_bytes());
                        b
                    },
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m_chunk[2].lo.to_le_bytes());
                        b[8..].copy_from_slice(&m_chunk[2].hi.to_le_bytes());
                        b
                    },
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m_chunk[3].lo.to_le_bytes());
                        b[8..].copy_from_slice(&m_chunk[3].hi.to_le_bytes());
                        b
                    },
                ];

                for r_byte in 0..16 {
                    let combined: u64 = (m_bytes[0][r_byte] as u64)
                        | ((m_bytes[1][r_byte] as u64) << 8)
                        | ((m_bytes[2][r_byte] as u64) << 16)
                        | ((m_bytes[3][r_byte] as u64) << 24);
                    let transposed = transpose_8x8_bits(combined);
                    let tb = transposed.to_le_bytes();
                    let base = r_byte * 8;
                    a0[base] += lookup0[(tb[0] & 0x0F) as usize];
                    a1[base] += lookup1[(tb[0] & 0x0F) as usize];
                    a0[base + 1] += lookup0[(tb[1] & 0x0F) as usize];
                    a1[base + 1] += lookup1[(tb[1] & 0x0F) as usize];
                    a0[base + 2] += lookup0[(tb[2] & 0x0F) as usize];
                    a1[base + 2] += lookup1[(tb[2] & 0x0F) as usize];
                    a0[base + 3] += lookup0[(tb[3] & 0x0F) as usize];
                    a1[base + 3] += lookup1[(tb[3] & 0x0F) as usize];
                    a0[base + 4] += lookup0[(tb[4] & 0x0F) as usize];
                    a1[base + 4] += lookup1[(tb[4] & 0x0F) as usize];
                    a0[base + 5] += lookup0[(tb[5] & 0x0F) as usize];
                    a1[base + 5] += lookup1[(tb[5] & 0x0F) as usize];
                    a0[base + 6] += lookup0[(tb[6] & 0x0F) as usize];
                    a1[base + 6] += lookup1[(tb[6] & 0x0F) as usize];
                    a0[base + 7] += lookup0[(tb[7] & 0x0F) as usize];
                    a1[base + 7] += lookup1[(tb[7] & 0x0F) as usize];
                }

                (a0, a1)
            },
        )
        .reduce(
            || (vec![F128::ZERO; n], vec![F128::ZERO; n]),
            |(mut a0, mut a1), (b0, b1)| {
                for r in 0..n {
                    a0[r] += b0[r];
                    a1[r] += b1[r];
                }
                (a0, a1)
            },
        );

    (pair.0, pair.1)
}

pub fn fold_1b_rows_2way_mfr_8wide(
    packed_witness: &[F128],
    t0: &[F128],
    t1: &[F128],
) -> (Vec<F128>, Vec<F128>) {
    let m = LOG_PACKING + (packed_witness.len().trailing_zeros() as usize);
    fold_1b_rows_2way_mfr_8wide_padded(packed_witness, t0, t1, &PaddingSpec::dense(m))
}

pub fn fold_1b_rows_2way_mfr_8wide_padded(
    packed_witness: &[F128],
    t0: &[F128],
    t1: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    assert_eq!(t0.len(), packed_witness.len());
    assert_eq!(t1.len(), packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(8));
    let skip = ChunkPadding::new(padding, 8);

    packed_witness
        .par_chunks(8)
        .zip(t0.par_chunks(8))
        .zip(t1.par_chunks(8))
        .enumerate()
        .fold(
            || (vec![F128::ZERO; n], vec![F128::ZERO; n]),
            |(mut a0, mut a1), (chunk_idx, ((m_chunk, t0_chunk), t1_chunk))| {
                if skip.skip(chunk_idx) {
                    return (a0, a1);
                }
                let t0_lo = subset_sums_4([t0_chunk[0], t0_chunk[1], t0_chunk[2], t0_chunk[3]]);
                let t0_hi = subset_sums_4([t0_chunk[4], t0_chunk[5], t0_chunk[6], t0_chunk[7]]);
                let t1_lo = subset_sums_4([t1_chunk[0], t1_chunk[1], t1_chunk[2], t1_chunk[3]]);
                let t1_hi = subset_sums_4([t1_chunk[4], t1_chunk[5], t1_chunk[6], t1_chunk[7]]);

                let mut m_bytes = [[0u8; 16]; 8];
                for (e, slot) in m_bytes.iter_mut().enumerate() {
                    slot[..8].copy_from_slice(&m_chunk[e].lo.to_le_bytes());
                    slot[8..].copy_from_slice(&m_chunk[e].hi.to_le_bytes());
                }

                for r_byte in 0..16 {
                    let combined: u64 = (m_bytes[0][r_byte] as u64)
                        | ((m_bytes[1][r_byte] as u64) << 8)
                        | ((m_bytes[2][r_byte] as u64) << 16)
                        | ((m_bytes[3][r_byte] as u64) << 24)
                        | ((m_bytes[4][r_byte] as u64) << 32)
                        | ((m_bytes[5][r_byte] as u64) << 40)
                        | ((m_bytes[6][r_byte] as u64) << 48)
                        | ((m_bytes[7][r_byte] as u64) << 56);
                    let tb = transpose_8x8_bits(combined).to_le_bytes();
                    let base = r_byte * 8;
                    for p in 0..8 {
                        let mask = tb[p];
                        let lo = (mask & 0x0F) as usize;
                        let hi = (mask >> 4) as usize;
                        a0[base + p] += t0_lo[lo] + t0_hi[hi];
                        a1[base + p] += t1_lo[lo] + t1_hi[hi];
                    }
                }
                (a0, a1)
            },
        )
        .reduce(
            || (vec![F128::ZERO; n], vec![F128::ZERO; n]),
            |(mut a0, mut a1), (b0, b1)| {
                for r in 0..n {
                    a0[r] += b0[r];
                    a1[r] += b1[r];
                }
                (a0, a1)
            },
        )
}

pub fn fold_1b_rows_1way_mfr(packed_witness: &[F128], t: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    assert_eq!(t.len(), packed_witness.len());
    assert!(
        packed_witness.len().is_multiple_of(4),
        "fold_1b_rows_1way_mfr requires len divisible by 4 (got {})",
        packed_witness.len()
    );

    packed_witness
        .par_chunks(4)
        .zip(t.par_chunks(4))
        .fold(
            || vec![F128::ZERO; n],
            |mut acc, (m_chunk, t_chunk)| {
                let v: [F128; 4] = [t_chunk[0], t_chunk[1], t_chunk[2], t_chunk[3]];
                let lookup = subset_sums_4(v);

                let m_bytes: [[u8; 16]; 4] = [
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m_chunk[0].lo.to_le_bytes());
                        b[8..].copy_from_slice(&m_chunk[0].hi.to_le_bytes());
                        b
                    },
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m_chunk[1].lo.to_le_bytes());
                        b[8..].copy_from_slice(&m_chunk[1].hi.to_le_bytes());
                        b
                    },
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m_chunk[2].lo.to_le_bytes());
                        b[8..].copy_from_slice(&m_chunk[2].hi.to_le_bytes());
                        b
                    },
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m_chunk[3].lo.to_le_bytes());
                        b[8..].copy_from_slice(&m_chunk[3].hi.to_le_bytes());
                        b
                    },
                ];

                for r_byte in 0..16 {
                    let combined: u64 = (m_bytes[0][r_byte] as u64)
                        | ((m_bytes[1][r_byte] as u64) << 8)
                        | ((m_bytes[2][r_byte] as u64) << 16)
                        | ((m_bytes[3][r_byte] as u64) << 24);
                    let transposed = transpose_8x8_bits(combined);
                    let tb = transposed.to_le_bytes();
                    let base = r_byte * 8;
                    acc[base] += lookup[(tb[0] & 0x0F) as usize];
                    acc[base + 1] += lookup[(tb[1] & 0x0F) as usize];
                    acc[base + 2] += lookup[(tb[2] & 0x0F) as usize];
                    acc[base + 3] += lookup[(tb[3] & 0x0F) as usize];
                    acc[base + 4] += lookup[(tb[4] & 0x0F) as usize];
                    acc[base + 5] += lookup[(tb[5] & 0x0F) as usize];
                    acc[base + 6] += lookup[(tb[6] & 0x0F) as usize];
                    acc[base + 7] += lookup[(tb[7] & 0x0F) as usize];
                }

                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; n],
            |mut a, b| {
                for r in 0..n {
                    a[r] += b[r];
                }
                a
            },
        )
}

pub fn fold_1b_rows_1way_mfr_8wide_k4(packed_witness: &[F128], t: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    assert_eq!(t.len(), packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(8));

    packed_witness
        .par_chunks(8)
        .zip(t.par_chunks(8))
        .fold(
            || vec![F128::ZERO; n],
            |mut acc, (m_chunk, t_chunk)| {
                let lo_tbl = subset_sums_4([t_chunk[0], t_chunk[1], t_chunk[2], t_chunk[3]]);
                let hi_tbl = subset_sums_4([t_chunk[4], t_chunk[5], t_chunk[6], t_chunk[7]]);

                let mut m_bytes = [[0u8; 16]; 8];
                for (e, slot) in m_bytes.iter_mut().enumerate() {
                    slot[..8].copy_from_slice(&m_chunk[e].lo.to_le_bytes());
                    slot[8..].copy_from_slice(&m_chunk[e].hi.to_le_bytes());
                }

                for r_byte in 0..16 {
                    let combined: u64 = (m_bytes[0][r_byte] as u64)
                        | ((m_bytes[1][r_byte] as u64) << 8)
                        | ((m_bytes[2][r_byte] as u64) << 16)
                        | ((m_bytes[3][r_byte] as u64) << 24)
                        | ((m_bytes[4][r_byte] as u64) << 32)
                        | ((m_bytes[5][r_byte] as u64) << 40)
                        | ((m_bytes[6][r_byte] as u64) << 48)
                        | ((m_bytes[7][r_byte] as u64) << 56);
                    let tb = transpose_8x8_bits(combined).to_le_bytes();
                    let base = r_byte * 8;
                    for p in 0..8 {
                        let mask = tb[p];
                        acc[base + p] +=
                            lo_tbl[(mask & 0x0F) as usize] + hi_tbl[(mask >> 4) as usize];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; n],
            |mut a, b| {
                for r in 0..n {
                    a[r] += b[r];
                }
                a
            },
        )
}

pub fn fold_1b_rows_1way_mfr_16wide_padded(
    packed_witness: &[F128],
    t: &[F128],
    padding: &PaddingSpec,
) -> Vec<F128> {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    assert_eq!(t.len(), packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(16));
    let skip = ChunkPadding::new(padding, 16);

    packed_witness
        .par_chunks(16)
        .zip(t.par_chunks(16))
        .enumerate()
        .fold(
            || vec![F128::ZERO; n],
            |mut acc, (chunk_idx, (m_chunk, t_chunk))| {
                if skip.skip(chunk_idx) {
                    return acc;
                }
                let tbl0 = subset_sums_4([t_chunk[0], t_chunk[1], t_chunk[2], t_chunk[3]]);
                let tbl1 = subset_sums_4([t_chunk[4], t_chunk[5], t_chunk[6], t_chunk[7]]);
                let tbl2 = subset_sums_4([t_chunk[8], t_chunk[9], t_chunk[10], t_chunk[11]]);
                let tbl3 = subset_sums_4([t_chunk[12], t_chunk[13], t_chunk[14], t_chunk[15]]);

                let mut tlo_all = [0u8; 128];
                let mut thi_all = [0u8; 128];
                let lo0_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[e].lo);
                let hi0_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[e].hi);
                let lo1_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[8 + e].lo);
                let hi1_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[8 + e].hi);
                crate::bits::transpose_8_u64s_to_64_bytes(&lo0_lanes, &mut tlo_all[..64]);
                crate::bits::transpose_8_u64s_to_64_bytes(&hi0_lanes, &mut tlo_all[64..128]);
                crate::bits::transpose_8_u64s_to_64_bytes(&lo1_lanes, &mut thi_all[..64]);
                crate::bits::transpose_8_u64s_to_64_bytes(&hi1_lanes, &mut thi_all[64..128]);

                for r in 0..128 {
                    let m_lo = tlo_all[r] as u64;
                    let m_hi = thi_all[r] as u64;
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    let (i_lo4, i_hi4, i_lo4h, i_hi4h) = unsafe {
                        (
                            crate::gf2_kernels::gf2_kern_nibble_lo(m_lo) as usize,
                            crate::gf2_kernels::gf2_kern_nibble_hi(m_lo) as usize,
                            crate::gf2_kernels::gf2_kern_nibble_lo(m_hi) as usize,
                            crate::gf2_kernels::gf2_kern_nibble_hi(m_hi) as usize,
                        )
                    };
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    let (i_lo4, i_hi4, i_lo4h, i_hi4h) = (
                        (m_lo & 0x0F) as usize,
                        (m_lo >> 4) as usize,
                        (m_hi & 0x0F) as usize,
                        (m_hi >> 4) as usize,
                    );
                    acc[r] += tbl0[i_lo4] + tbl1[i_hi4] + tbl2[i_lo4h] + tbl3[i_hi4h];
                }
                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; n],
            |mut a, b| {
                for r in 0..n {
                    a[r] += b[r];
                }
                a
            },
        )
}

pub fn fold_1b_rows_1way_mfr_16wide_k4(packed_witness: &[F128], t: &[F128]) -> Vec<F128> {
    let m = LOG_PACKING + (packed_witness.len().trailing_zeros() as usize);
    fold_1b_rows_1way_mfr_16wide_padded(packed_witness, t, &PaddingSpec::dense(m))
}

pub fn fold_1b_rows_2way_mfr_16wide_padded(
    packed_witness: &[F128],
    t0: &[F128],
    t1: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    assert_eq!(t0.len(), packed_witness.len());
    assert_eq!(t1.len(), packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(16));
    let skip = ChunkPadding::new(padding, 16);

    packed_witness
        .par_chunks(16)
        .zip(t0.par_chunks(16))
        .zip(t1.par_chunks(16))
        .enumerate()
        .fold(
            || (vec![F128::ZERO; n], vec![F128::ZERO; n]),
            |(mut a0, mut a1), (chunk_idx, ((m_chunk, t0_chunk), t1_chunk))| {
                if skip.skip(chunk_idx) {
                    return (a0, a1);
                }
                let t0_0 = subset_sums_4([t0_chunk[0], t0_chunk[1], t0_chunk[2], t0_chunk[3]]);
                let t0_1 = subset_sums_4([t0_chunk[4], t0_chunk[5], t0_chunk[6], t0_chunk[7]]);
                let t0_2 = subset_sums_4([t0_chunk[8], t0_chunk[9], t0_chunk[10], t0_chunk[11]]);
                let t0_3 = subset_sums_4([t0_chunk[12], t0_chunk[13], t0_chunk[14], t0_chunk[15]]);
                let t1_0 = subset_sums_4([t1_chunk[0], t1_chunk[1], t1_chunk[2], t1_chunk[3]]);
                let t1_1 = subset_sums_4([t1_chunk[4], t1_chunk[5], t1_chunk[6], t1_chunk[7]]);
                let t1_2 = subset_sums_4([t1_chunk[8], t1_chunk[9], t1_chunk[10], t1_chunk[11]]);
                let t1_3 = subset_sums_4([t1_chunk[12], t1_chunk[13], t1_chunk[14], t1_chunk[15]]);

                let mut tlo_all = [0u8; 128];
                let mut thi_all = [0u8; 128];
                let lo0_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[e].lo);
                let hi0_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[e].hi);
                let lo1_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[8 + e].lo);
                let hi1_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[8 + e].hi);
                crate::bits::transpose_8_u64s_to_64_bytes(&lo0_lanes, &mut tlo_all[..64]);
                crate::bits::transpose_8_u64s_to_64_bytes(&hi0_lanes, &mut tlo_all[64..128]);
                crate::bits::transpose_8_u64s_to_64_bytes(&lo1_lanes, &mut thi_all[..64]);
                crate::bits::transpose_8_u64s_to_64_bytes(&hi1_lanes, &mut thi_all[64..128]);

                for r in 0..128 {
                    let m_lo = tlo_all[r] as u64;
                    let m_hi = thi_all[r] as u64;
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    let (lo_lo, lo_hi, hi_lo, hi_hi) = unsafe {
                        (
                            crate::gf2_kernels::gf2_kern_nibble_lo(m_lo) as usize,
                            crate::gf2_kernels::gf2_kern_nibble_hi(m_lo) as usize,
                            crate::gf2_kernels::gf2_kern_nibble_lo(m_hi) as usize,
                            crate::gf2_kernels::gf2_kern_nibble_hi(m_hi) as usize,
                        )
                    };
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    let (lo_lo, lo_hi, hi_lo, hi_hi) = (
                        (m_lo & 0x0F) as usize,
                        (m_lo >> 4) as usize,
                        (m_hi & 0x0F) as usize,
                        (m_hi >> 4) as usize,
                    );
                    a0[r] += t0_0[lo_lo] + t0_1[lo_hi] + t0_2[hi_lo] + t0_3[hi_hi];
                    a1[r] += t1_0[lo_lo] + t1_1[lo_hi] + t1_2[hi_lo] + t1_3[hi_hi];
                }
                (a0, a1)
            },
        )
        .reduce(
            || (vec![F128::ZERO; n], vec![F128::ZERO; n]),
            |(mut a0, mut a1), (b0, b1)| {
                for r in 0..n {
                    a0[r] += b0[r];
                    a1[r] += b1[r];
                }
                (a0, a1)
            },
        )
}

pub fn fold_1b_rows_split(
    packed_witness: &[F128],
    eq_lo: &[F128],
    eq_hi: &[F128],
    padding: &PaddingSpec,
) -> Vec<F128> {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    let b = eq_lo.len();
    assert!(
        b.is_multiple_of(16),
        "fold_1b_rows_split: eq_lo block size must be a multiple of 16 (got {b})"
    );
    assert_eq!(packed_witness.len(), b * eq_hi.len());
    let chunks_per_block = b / 16;
    let skip = ChunkPadding::new(padding, 16);

    let tables: Vec<[[F128; 16]; 4]> = (0..chunks_per_block)
        .map(|c| {
            let o = c * 16;
            [
                subset_sums_4([eq_lo[o], eq_lo[o + 1], eq_lo[o + 2], eq_lo[o + 3]]),
                subset_sums_4([eq_lo[o + 4], eq_lo[o + 5], eq_lo[o + 6], eq_lo[o + 7]]),
                subset_sums_4([eq_lo[o + 8], eq_lo[o + 9], eq_lo[o + 10], eq_lo[o + 11]]),
                subset_sums_4([eq_lo[o + 12], eq_lo[o + 13], eq_lo[o + 14], eq_lo[o + 15]]),
            ]
        })
        .collect();

    packed_witness
        .par_chunks(b)
        .enumerate()
        .fold(
            || vec![F128::ZERO; n],
            |mut acc, (i_hi, w_block)| {
                let mut inner = [F128::ZERO; 128];
                let base_chunk = i_hi * chunks_per_block;
                for c in 0..chunks_per_block {
                    if skip.skip(base_chunk + c) {
                        continue;
                    }
                    let m_chunk = &w_block[c * 16..c * 16 + 16];
                    let [tbl0, tbl1, tbl2, tbl3] = &tables[c];

                    let mut tlo_all = [0u8; 128];
                    let mut thi_all = [0u8; 128];
                    let lo0_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[e].lo);
                    let hi0_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[e].hi);
                    let lo1_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[8 + e].lo);
                    let hi1_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[8 + e].hi);
                    crate::bits::transpose_8_u64s_to_64_bytes(&lo0_lanes, &mut tlo_all[..64]);
                    crate::bits::transpose_8_u64s_to_64_bytes(&hi0_lanes, &mut tlo_all[64..128]);
                    crate::bits::transpose_8_u64s_to_64_bytes(&lo1_lanes, &mut thi_all[..64]);
                    crate::bits::transpose_8_u64s_to_64_bytes(&hi1_lanes, &mut thi_all[64..128]);

                    for r in 0..128 {
                        let m_lo = tlo_all[r] as u64;
                        let m_hi = thi_all[r] as u64;
                        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                        let (i_lo4, i_hi4, i_lo4h, i_hi4h) = unsafe {
                            (
                                crate::gf2_kernels::gf2_kern_nibble_lo(m_lo) as usize,
                                crate::gf2_kernels::gf2_kern_nibble_hi(m_lo) as usize,
                                crate::gf2_kernels::gf2_kern_nibble_lo(m_hi) as usize,
                                crate::gf2_kernels::gf2_kern_nibble_hi(m_hi) as usize,
                            )
                        };
                        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                        let (i_lo4, i_hi4, i_lo4h, i_hi4h) = (
                            (m_lo & 0x0F) as usize,
                            (m_lo >> 4) as usize,
                            (m_hi & 0x0F) as usize,
                            (m_hi >> 4) as usize,
                        );
                        inner[r] += tbl0[i_lo4] + tbl1[i_hi4] + tbl2[i_lo4h] + tbl3[i_hi4h];
                    }
                }
                let e = eq_hi[i_hi];
                crate::field::f128_slice::scalar_mul_add_assign(&mut acc, &inner, e);
                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; n],
            |mut a, b| {
                crate::field::f128_slice::xor_assign(&mut a, &b);
                a
            },
        )
}

pub fn fold_1b_rows_split_2way(
    packed_witness: &[F128],
    eq_lo_0: &[F128],
    eq_hi_0: &[F128],
    eq_lo_1: &[F128],
    eq_hi_1: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    let b = eq_lo_0.len();
    assert_eq!(eq_lo_1.len(), b);
    let n_hi = eq_hi_0.len();
    assert_eq!(eq_hi_1.len(), n_hi);
    assert!(
        b.is_multiple_of(16),
        "fold_1b_rows_split_2way: eq_lo block size must be a multiple of 16 (got {b})"
    );
    assert_eq!(packed_witness.len(), b * n_hi);
    let chunks_per_block = b / 16;
    let skip = ChunkPadding::new(padding, 16);

    let tables_0: Vec<[[F128; 16]; 4]> = (0..chunks_per_block)
        .map(|c| {
            let o = c * 16;
            [
                subset_sums_4([eq_lo_0[o], eq_lo_0[o + 1], eq_lo_0[o + 2], eq_lo_0[o + 3]]),
                subset_sums_4([
                    eq_lo_0[o + 4],
                    eq_lo_0[o + 5],
                    eq_lo_0[o + 6],
                    eq_lo_0[o + 7],
                ]),
                subset_sums_4([
                    eq_lo_0[o + 8],
                    eq_lo_0[o + 9],
                    eq_lo_0[o + 10],
                    eq_lo_0[o + 11],
                ]),
                subset_sums_4([
                    eq_lo_0[o + 12],
                    eq_lo_0[o + 13],
                    eq_lo_0[o + 14],
                    eq_lo_0[o + 15],
                ]),
            ]
        })
        .collect();
    let tables_1: Vec<[[F128; 16]; 4]> = (0..chunks_per_block)
        .map(|c| {
            let o = c * 16;
            [
                subset_sums_4([eq_lo_1[o], eq_lo_1[o + 1], eq_lo_1[o + 2], eq_lo_1[o + 3]]),
                subset_sums_4([
                    eq_lo_1[o + 4],
                    eq_lo_1[o + 5],
                    eq_lo_1[o + 6],
                    eq_lo_1[o + 7],
                ]),
                subset_sums_4([
                    eq_lo_1[o + 8],
                    eq_lo_1[o + 9],
                    eq_lo_1[o + 10],
                    eq_lo_1[o + 11],
                ]),
                subset_sums_4([
                    eq_lo_1[o + 12],
                    eq_lo_1[o + 13],
                    eq_lo_1[o + 14],
                    eq_lo_1[o + 15],
                ]),
            ]
        })
        .collect();

    let zero_acc = || (vec![F128::ZERO; n], vec![F128::ZERO; n]);

    packed_witness
        .par_chunks(b)
        .enumerate()
        .fold(zero_acc, |(mut acc0, mut acc1), (i_hi, w_block)| {
            let mut inner0 = [F128::ZERO; 128];
            let mut inner1 = [F128::ZERO; 128];
            let base_chunk = i_hi * chunks_per_block;
            for c in 0..chunks_per_block {
                if skip.skip(base_chunk + c) {
                    continue;
                }
                let m_chunk = &w_block[c * 16..c * 16 + 16];
                let [t0a, t0b, t0c, t0d] = &tables_0[c];
                let [t1a, t1b, t1c, t1d] = &tables_1[c];

                let mut tlo_all = [0u8; 128];
                let mut thi_all = [0u8; 128];
                let lo0_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[e].lo);
                let hi0_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[e].hi);
                let lo1_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[8 + e].lo);
                let hi1_lanes: [u64; 8] = core::array::from_fn(|e| m_chunk[8 + e].hi);
                crate::bits::transpose_8_u64s_to_64_bytes(&lo0_lanes, &mut tlo_all[..64]);
                crate::bits::transpose_8_u64s_to_64_bytes(&hi0_lanes, &mut tlo_all[64..128]);
                crate::bits::transpose_8_u64s_to_64_bytes(&lo1_lanes, &mut thi_all[..64]);
                crate::bits::transpose_8_u64s_to_64_bytes(&hi1_lanes, &mut thi_all[64..128]);

                for r in 0..128 {
                    let m_lo = tlo_all[r] as u64;
                    let m_hi = thi_all[r] as u64;
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    let (i_lo4, i_hi4, i_lo4h, i_hi4h) = unsafe {
                        (
                            crate::gf2_kernels::gf2_kern_nibble_lo(m_lo) as usize,
                            crate::gf2_kernels::gf2_kern_nibble_hi(m_lo) as usize,
                            crate::gf2_kernels::gf2_kern_nibble_lo(m_hi) as usize,
                            crate::gf2_kernels::gf2_kern_nibble_hi(m_hi) as usize,
                        )
                    };
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    let (i_lo4, i_hi4, i_lo4h, i_hi4h) = (
                        (m_lo & 0x0F) as usize,
                        (m_lo >> 4) as usize,
                        (m_hi & 0x0F) as usize,
                        (m_hi >> 4) as usize,
                    );
                    inner0[r] += t0a[i_lo4] + t0b[i_hi4] + t0c[i_lo4h] + t0d[i_hi4h];
                    inner1[r] += t1a[i_lo4] + t1b[i_hi4] + t1c[i_lo4h] + t1d[i_hi4h];
                }
            }
            let e0 = eq_hi_0[i_hi];
            let e1 = eq_hi_1[i_hi];
            crate::field::f128_slice::scalar_mul_add_assign(&mut acc0, &inner0, e0);
            crate::field::f128_slice::scalar_mul_add_assign(&mut acc1, &inner1, e1);
            (acc0, acc1)
        })
        .reduce(zero_acc, |(mut a0, mut a1), (b0, b1)| {
            crate::field::f128_slice::xor_assign(&mut a0, &b0);
            crate::field::f128_slice::xor_assign(&mut a1, &b1);
            (a0, a1)
        })
}

pub fn s_hat_v_from_z_vec(z_vec: &[F128], x_inner_rest_tail: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    let n_packed = 1usize << LOG_PACKING;
    let n_tail = 1usize << x_inner_rest_tail.len();
    assert_eq!(
        z_vec.len(),
        n_packed * n_tail,
        "z_vec length {} mismatches 2^(LOG_PACKING + tail.len()) = {}",
        z_vec.len(),
        n_packed * n_tail,
    );

    if x_inner_rest_tail.is_empty() {
        return z_vec.to_vec();
    }

    let eq_tail = build_eq_parallel(x_inner_rest_tail);

    eq_tail
        .par_iter()
        .enumerate()
        .fold(
            || vec![F128::ZERO; n_packed],
            |mut acc, (k, &w)| {
                let block = &z_vec[k * n_packed..(k + 1) * n_packed];
                crate::field::f128_slice::scalar_mul_add_assign(&mut acc, block, w);
                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; n_packed],
            |mut a, b| {
                crate::field::f128_slice::xor_assign(&mut a, &b);
                a
            },
        )
}

pub fn s_hat_v_fold8_from_z_vec(z_vec: &[F128], x_inner_rest_tail: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    assert!(
        x_inner_rest_tail.len() >= 6,
        "s_hat_v_fold8_from_z_vec requires at least 6 suffix coordinates"
    );
    let n_packed = 1usize << LOG_PACKING;
    let stride = 64 * n_packed;
    let n_q = 1usize << (x_inner_rest_tail.len() - 6);
    assert_eq!(
        z_vec.len(),
        n_q * stride,
        "z_vec length mismatch: expected 2^{} * 8192 = {}, got {}",
        x_inner_rest_tail.len() - 6,
        n_q * stride,
        z_vec.len(),
    );

    let eq_tail = build_eq_parallel(&x_inner_rest_tail[6..]);
    let mut acc = vec![F128::ZERO; stride];
    if n_q <= 2 {
        for (q, &w) in eq_tail.iter().enumerate() {
            let block = &z_vec[q * stride..(q + 1) * stride];
            crate::field::f128_slice::scalar_mul_add_assign(&mut acc, block, w);
        }
        return acc;
    }

    const CHUNK_LEN: usize = 1024;
    acc.par_chunks_mut(CHUNK_LEN)
        .enumerate()
        .for_each(|(c, out_chunk)| {
            let offset = c * CHUNK_LEN;
            let len = out_chunk.len();
            for (q, &w) in eq_tail.iter().enumerate() {
                let base = q * stride + offset;
                let block = &z_vec[base..base + len];
                crate::field::f128_slice::scalar_mul_add_assign(out_chunk, block, w);
            }
        });
    acc
}

pub fn collapse_s_hat_v_fold8(s_hat_v_fold8: &[F128], r_low6: &[F128]) -> Vec<F128> {
    let n_packed = 1usize << LOG_PACKING;
    assert_eq!(s_hat_v_fold8.len(), 64 * n_packed);
    assert_eq!(r_low6.len(), 6);
    let eq_low6 = build_eq(r_low6);
    let mut out = vec![F128::ZERO; n_packed];
    for (bank, &w) in eq_low6.iter().enumerate() {
        let src = &s_hat_v_fold8[bank * n_packed..(bank + 1) * n_packed];
        crate::field::f128_slice::scalar_mul_add_assign(&mut out, src, w);
    }
    out
}

pub fn fold_1b_rows_naive(packed_witness: &[F128], suffix_tensor: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    assert_eq!(packed_witness.len(), suffix_tensor.len());
    let n = 1 << LOG_PACKING;

    if packed_witness.len().is_multiple_of(16) {
        return fold_1b_rows_1way_mfr_16wide_k4(packed_witness, suffix_tensor);
    }
    if packed_witness.len().is_multiple_of(8) {
        return fold_1b_rows_1way_mfr_8wide_k4(packed_witness, suffix_tensor);
    }
    if packed_witness.len() >= 4 && packed_witness.len().is_multiple_of(4) {
        return fold_1b_rows_1way_mfr(packed_witness, suffix_tensor);
    }

    let zero_acc = || vec![F128::ZERO; n];

    packed_witness
        .par_iter()
        .zip(suffix_tensor.par_iter())
        .fold(zero_acc, |mut acc, (elem, &w)| {
            let mut lo = elem.lo;
            while lo != 0 {
                let r = lo.trailing_zeros() as usize;
                acc[r] += w;
                lo &= lo - 1;
            }
            let mut hi = elem.hi;
            while hi != 0 {
                let r = hi.trailing_zeros() as usize;
                acc[64 | r] += w;
                hi &= hi - 1;
            }
            acc
        })
        .reduce(zero_acc, |mut a, b| {
            for (av, bv) in a.iter_mut().zip(b.iter()) {
                *av += *bv;
            }
            a
        })
}

pub fn claim_check(weights: &[F128], s_hat_v: &[F128]) -> F128 {
    inner_product(weights, s_hat_v)
}

pub fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    crate::field::f128_slice::inner_product(a, b)
}

pub fn tensor_algebra_transpose(s_hat_v: &[F128]) -> Vec<F128> {
    assert_eq!(s_hat_v.len(), 1 << LOG_PACKING);
    let mut in_bytes = [[0u8; 16]; 128];
    for (i, elem) in s_hat_v.iter().enumerate() {
        in_bytes[i][..8].copy_from_slice(&elem.lo.to_le_bytes());
        in_bytes[i][8..].copy_from_slice(&elem.hi.to_le_bytes());
    }
    let mut out_bytes = [[0u8; 16]; 128];
    for i_blk in 0..16 {
        let row_base = 8 * i_blk;
        for b_blk in 0..16 {
            let combined: u64 = (in_bytes[row_base][b_blk] as u64)
                | ((in_bytes[row_base + 1][b_blk] as u64) << 8)
                | ((in_bytes[row_base + 2][b_blk] as u64) << 16)
                | ((in_bytes[row_base + 3][b_blk] as u64) << 24)
                | ((in_bytes[row_base + 4][b_blk] as u64) << 32)
                | ((in_bytes[row_base + 5][b_blk] as u64) << 40)
                | ((in_bytes[row_base + 6][b_blk] as u64) << 48)
                | ((in_bytes[row_base + 7][b_blk] as u64) << 56);
            let tb = transpose_8x8_bits(combined).to_le_bytes();
            let col_base = 8 * b_blk;
            for c in 0..8 {
                out_bytes[col_base + c][i_blk] = tb[c];
            }
        }
    }
    let mut s_hat_u = Vec::with_capacity(128);
    for b in &out_bytes {
        s_hat_u.push(F128 {
            lo: u64::from_le_bytes(b[..8].try_into().unwrap()),
            hi: u64::from_le_bytes(b[8..].try_into().unwrap()),
        });
    }
    s_hat_u
}

pub fn fold_b128_elems_naive(suffix_tensor: &[F128], eq_r_dprime: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    suffix_tensor
        .par_iter()
        .map(|&elem| {
            let mut acc = F128::ZERO;
            let mut lo = elem.lo;
            while lo != 0 {
                let b = lo.trailing_zeros() as usize;
                acc += eq_r_dprime[b];
                lo &= lo - 1;
            }
            let mut hi = elem.hi;
            while hi != 0 {
                let b = hi.trailing_zeros() as usize;
                acc += eq_r_dprime[64 | b];
                hi &= hi - 1;
            }
            acc
        })
        .collect()
}

pub fn fold_b128_elems(suffix_tensor: &[F128], eq_r_dprime: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);

    let tables = build_fold_byte_table(eq_r_dprime);

    suffix_tensor
        .par_iter()
        .map(|&elem| fold_one_slot(elem, &tables))
        .collect()
}

const FOLD_N_BYTES: usize = 16;
const FOLD_TABLE_SIZE: usize = 256;
pub(crate) const FOLD_TABLE_TOTAL: usize = FOLD_N_BYTES * FOLD_TABLE_SIZE;

fn build_fold_byte_table(eq_r_dprime: &[F128]) -> Vec<F128> {
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    let mut tables = vec![F128::ZERO; FOLD_N_BYTES * FOLD_TABLE_SIZE];
    for byte_idx in 0..FOLD_N_BYTES {
        let bit_base = byte_idx * 8;
        let sub = &mut tables[byte_idx * FOLD_TABLE_SIZE..(byte_idx + 1) * FOLD_TABLE_SIZE];
        sub[0] = F128::ZERO;
        for bit_in_byte in 0..8 {
            let e = eq_r_dprime[bit_base + bit_in_byte];
            let half = 1usize << bit_in_byte;
            for j in 0..half {
                sub[half + j] = sub[j] + e;
            }
        }
    }
    tables
}

pub(crate) fn build_direct_fold8_table_from_generators(generators: &[F128; 128]) -> Vec<F128> {
    build_fold_byte_table(generators)
}

#[inline(always)]
pub(crate) fn fold_one_slot(elem: F128, tables: &[F128]) -> F128 {
    debug_assert_eq!(tables.len(), FOLD_N_BYTES * FOLD_TABLE_SIZE);
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        unsafe {
            return core::mem::transmute::<__m128i, F128>(fold_one_slot_m128i(
                core::mem::transmute::<F128, __m128i>(elem),
                tables.as_ptr(),
            ));
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        unsafe {
            return core::mem::transmute::<uint64x2_t, F128>(fold_one_slot_neon(
                core::mem::transmute::<F128, uint64x2_t>(elem),
                tables.as_ptr(),
            ));
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let lo_bytes = elem.lo.to_le_bytes();
        let hi_bytes = elem.hi.to_le_bytes();
        let tables_ptr = tables.as_ptr();
        let (l0, l1, l2, l3, l4, l5, l6, l7, h0, h1, h2, h3, h4, h5, h6, h7) = unsafe {
            (
                *tables_ptr.add(lo_bytes[0] as usize),
                *tables_ptr.add(FOLD_TABLE_SIZE + lo_bytes[1] as usize),
                *tables_ptr.add(2 * FOLD_TABLE_SIZE + lo_bytes[2] as usize),
                *tables_ptr.add(3 * FOLD_TABLE_SIZE + lo_bytes[3] as usize),
                *tables_ptr.add(4 * FOLD_TABLE_SIZE + lo_bytes[4] as usize),
                *tables_ptr.add(5 * FOLD_TABLE_SIZE + lo_bytes[5] as usize),
                *tables_ptr.add(6 * FOLD_TABLE_SIZE + lo_bytes[6] as usize),
                *tables_ptr.add(7 * FOLD_TABLE_SIZE + lo_bytes[7] as usize),
                *tables_ptr.add(8 * FOLD_TABLE_SIZE + hi_bytes[0] as usize),
                *tables_ptr.add(9 * FOLD_TABLE_SIZE + hi_bytes[1] as usize),
                *tables_ptr.add(10 * FOLD_TABLE_SIZE + hi_bytes[2] as usize),
                *tables_ptr.add(11 * FOLD_TABLE_SIZE + hi_bytes[3] as usize),
                *tables_ptr.add(12 * FOLD_TABLE_SIZE + hi_bytes[4] as usize),
                *tables_ptr.add(13 * FOLD_TABLE_SIZE + hi_bytes[5] as usize),
                *tables_ptr.add(14 * FOLD_TABLE_SIZE + hi_bytes[6] as usize),
                *tables_ptr.add(15 * FOLD_TABLE_SIZE + hi_bytes[7] as usize),
            )
        };
        let p0 = l0 + l1;
        let p1 = l2 + l3;
        let p2 = l4 + l5;
        let p3 = l6 + l7;
        let p4 = h0 + h1;
        let p5 = h2 + h3;
        let p6 = h4 + h5;
        let p7 = h6 + h7;
        let q0 = p0 + p1;
        let q1 = p2 + p3;
        let q2 = p4 + p5;
        let q3 = p6 + p7;
        let r0 = q0 + q1;
        let r1 = q2 + q3;
        r0 + r1
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fold_half_slot_u64_m128i(
    word: u64,
    half_tables_ptr: *const F128,
) -> core::arch::x86_64::__m128i {
    use core::arch::x86_64::*;
    unsafe {
        let bytes = word.to_le_bytes();
        let l0 = _mm_load_si128(half_tables_ptr.add(bytes[0] as usize) as *const __m128i);
        let l1 = _mm_load_si128(
            half_tables_ptr.add(FOLD_TABLE_SIZE + bytes[1] as usize) as *const __m128i
        );
        let l2 = _mm_load_si128(
            half_tables_ptr.add(2 * FOLD_TABLE_SIZE + bytes[2] as usize) as *const __m128i
        );
        let l3 = _mm_load_si128(
            half_tables_ptr.add(3 * FOLD_TABLE_SIZE + bytes[3] as usize) as *const __m128i
        );
        let l4 = _mm_load_si128(
            half_tables_ptr.add(4 * FOLD_TABLE_SIZE + bytes[4] as usize) as *const __m128i
        );
        let l5 = _mm_load_si128(
            half_tables_ptr.add(5 * FOLD_TABLE_SIZE + bytes[5] as usize) as *const __m128i
        );
        let l6 = _mm_load_si128(
            half_tables_ptr.add(6 * FOLD_TABLE_SIZE + bytes[6] as usize) as *const __m128i
        );
        let l7 = _mm_load_si128(
            half_tables_ptr.add(7 * FOLD_TABLE_SIZE + bytes[7] as usize) as *const __m128i
        );
        let p0 = _mm_xor_si128(l0, l1);
        let p1 = _mm_xor_si128(l2, l3);
        let p2 = _mm_xor_si128(l4, l5);
        let p3 = _mm_xor_si128(l6, l7);
        let q0 = _mm_xor_si128(p0, p1);
        let q1 = _mm_xor_si128(p2, p3);
        _mm_xor_si128(q0, q1)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fold_one_slot_u64_pair_m128i(
    lo: u64,
    hi: u64,
    tables_ptr: *const F128,
) -> core::arch::x86_64::__m128i {
    use core::arch::x86_64::*;
    unsafe {
        let r0 = fold_half_slot_u64_m128i(lo, tables_ptr);
        let r1 = fold_half_slot_u64_m128i(hi, tables_ptr.add(8 * FOLD_TABLE_SIZE));
        _mm_xor_si128(r0, r1)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fold_one_slot_m128i(
    elem: core::arch::x86_64::__m128i,
    tables_ptr: *const F128,
) -> core::arch::x86_64::__m128i {
    use core::arch::x86_64::*;
    unsafe {
        let lo = _mm_cvtsi128_si64(elem) as u64;
        let hi = _mm_extract_epi64::<1>(elem) as u64;
        fold_one_slot_u64_pair_m128i(lo, hi, tables_ptr)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn build_scaled_fold_byte_table_m128i(
    e_hi: F128,
    base_table_ptr: *const F128,
    out_table_ptr: *mut F128,
) {
    use core::arch::x86_64::*;
    unsafe {
        let zero = _mm_setzero_si128();
        let mut cur_lo = e_hi.lo;
        let mut cur_hi = e_hi.hi;
        for byte_idx in 0..FOLD_N_BYTES {
            let sub_ptr = out_table_ptr.add(byte_idx * FOLD_TABLE_SIZE) as *mut __m128i;
            let b0 = fold_one_slot_u64_pair_m128i(cur_lo, cur_hi, base_table_ptr);
            let next_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(cur_lo, cur_hi);
            let next_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(cur_lo, cur_hi);
            cur_lo = next_lo;
            cur_hi = next_hi;

            let b1 = fold_one_slot_u64_pair_m128i(cur_lo, cur_hi, base_table_ptr);
            let next_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(cur_lo, cur_hi);
            let next_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(cur_lo, cur_hi);
            cur_lo = next_lo;
            cur_hi = next_hi;

            let b2 = fold_one_slot_u64_pair_m128i(cur_lo, cur_hi, base_table_ptr);
            let next_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(cur_lo, cur_hi);
            let next_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(cur_lo, cur_hi);
            cur_lo = next_lo;
            cur_hi = next_hi;

            let b01 = _mm_xor_si128(b0, b1);
            let b02 = _mm_xor_si128(b0, b2);
            let b12 = _mm_xor_si128(b1, b2);
            let b012 = _mm_xor_si128(b01, b2);
            _mm_store_si128(sub_ptr, zero);
            _mm_store_si128(sub_ptr.add(1), b0);
            _mm_store_si128(sub_ptr.add(2), b1);
            _mm_store_si128(sub_ptr.add(3), b01);
            _mm_store_si128(sub_ptr.add(4), b2);
            _mm_store_si128(sub_ptr.add(5), b02);
            _mm_store_si128(sub_ptr.add(6), b12);
            _mm_store_si128(sub_ptr.add(7), b012);

            for bit_in_byte in 3..8 {
                let b_vec = fold_one_slot_u64_pair_m128i(cur_lo, cur_hi, base_table_ptr);
                let next_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(cur_lo, cur_hi);
                let next_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(cur_lo, cur_hi);
                cur_lo = next_lo;
                cur_hi = next_hi;

                let half = 1usize << bit_in_byte;
                let mut j = 0usize;
                while j < half {
                    let s0 = _mm_load_si128(sub_ptr.add(j));
                    let s1 = _mm_load_si128(sub_ptr.add(j + 1));
                    let s2 = _mm_load_si128(sub_ptr.add(j + 2));
                    let s3 = _mm_load_si128(sub_ptr.add(j + 3));
                    let s4 = _mm_load_si128(sub_ptr.add(j + 4));
                    let s5 = _mm_load_si128(sub_ptr.add(j + 5));
                    let s6 = _mm_load_si128(sub_ptr.add(j + 6));
                    let s7 = _mm_load_si128(sub_ptr.add(j + 7));
                    _mm_store_si128(sub_ptr.add(half + j), _mm_xor_si128(s0, b_vec));
                    _mm_store_si128(sub_ptr.add(half + j + 1), _mm_xor_si128(s1, b_vec));
                    _mm_store_si128(sub_ptr.add(half + j + 2), _mm_xor_si128(s2, b_vec));
                    _mm_store_si128(sub_ptr.add(half + j + 3), _mm_xor_si128(s3, b_vec));
                    _mm_store_si128(sub_ptr.add(half + j + 4), _mm_xor_si128(s4, b_vec));
                    _mm_store_si128(sub_ptr.add(half + j + 5), _mm_xor_si128(s5, b_vec));
                    _mm_store_si128(sub_ptr.add(half + j + 6), _mm_xor_si128(s6, b_vec));
                    _mm_store_si128(sub_ptr.add(half + j + 7), _mm_xor_si128(s7, b_vec));
                    j += 8;
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fold_half_slot_u64_neon(
    word: u64,
    half_tables_ptr: *const F128,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let bytes = word.to_le_bytes();
        let l0 = vld1q_u8(half_tables_ptr.add(bytes[0] as usize) as *const u8);
        let l1 = vld1q_u8(half_tables_ptr.add(FOLD_TABLE_SIZE + bytes[1] as usize) as *const u8);
        let l2 =
            vld1q_u8(half_tables_ptr.add(2 * FOLD_TABLE_SIZE + bytes[2] as usize) as *const u8);
        let l3 =
            vld1q_u8(half_tables_ptr.add(3 * FOLD_TABLE_SIZE + bytes[3] as usize) as *const u8);
        let l4 =
            vld1q_u8(half_tables_ptr.add(4 * FOLD_TABLE_SIZE + bytes[4] as usize) as *const u8);
        let l5 =
            vld1q_u8(half_tables_ptr.add(5 * FOLD_TABLE_SIZE + bytes[5] as usize) as *const u8);
        let l6 =
            vld1q_u8(half_tables_ptr.add(6 * FOLD_TABLE_SIZE + bytes[6] as usize) as *const u8);
        let l7 =
            vld1q_u8(half_tables_ptr.add(7 * FOLD_TABLE_SIZE + bytes[7] as usize) as *const u8);
        let p0 = veorq_u8(l0, l1);
        let p1 = veorq_u8(l2, l3);
        let p2 = veorq_u8(l4, l5);
        let p3 = veorq_u8(l6, l7);
        let q0 = veorq_u8(p0, p1);
        let q1 = veorq_u8(p2, p3);
        vreinterpretq_u64_u8(veorq_u8(q0, q1))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fold_one_slot_u64_pair_neon(
    lo: u64,
    hi: u64,
    tables_ptr: *const F128,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let r0 = fold_half_slot_u64_neon(lo, tables_ptr);
        let r1 = fold_half_slot_u64_neon(hi, tables_ptr.add(8 * FOLD_TABLE_SIZE));
        veorq_u64(r0, r1)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fold_one_slot_neon(
    elem: core::arch::aarch64::uint64x2_t,
    tables_ptr: *const F128,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let lo = vgetq_lane_u64::<0>(elem);
        let hi = vgetq_lane_u64::<1>(elem);
        fold_one_slot_u64_pair_neon(lo, hi, tables_ptr)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn build_scaled_fold_byte_table_neon(
    e_hi: F128,
    base_table_ptr: *const F128,
    out_table_ptr: *mut F128,
) {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let mut cur_lo = e_hi.lo;
        let mut cur_hi = e_hi.hi;
        for byte_idx in 0..FOLD_N_BYTES {
            let sub_ptr = out_table_ptr.add(byte_idx * FOLD_TABLE_SIZE) as *mut u64;
            let b0 = fold_one_slot_u64_pair_neon(cur_lo, cur_hi, base_table_ptr);
            let next_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(cur_lo, cur_hi);
            let next_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(cur_lo, cur_hi);
            cur_lo = next_lo;
            cur_hi = next_hi;

            let b1 = fold_one_slot_u64_pair_neon(cur_lo, cur_hi, base_table_ptr);
            let next_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(cur_lo, cur_hi);
            let next_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(cur_lo, cur_hi);
            cur_lo = next_lo;
            cur_hi = next_hi;

            let b2 = fold_one_slot_u64_pair_neon(cur_lo, cur_hi, base_table_ptr);
            let next_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(cur_lo, cur_hi);
            let next_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(cur_lo, cur_hi);
            cur_lo = next_lo;
            cur_hi = next_hi;

            let b01 = veorq_u64(b0, b1);
            let b02 = veorq_u64(b0, b2);
            let b12 = veorq_u64(b1, b2);
            let b012 = veorq_u64(b01, b2);
            vst1q_u64(sub_ptr, zero);
            vst1q_u64(sub_ptr.add(2), b0);
            vst1q_u64(sub_ptr.add(4), b1);
            vst1q_u64(sub_ptr.add(6), b01);
            vst1q_u64(sub_ptr.add(8), b2);
            vst1q_u64(sub_ptr.add(10), b02);
            vst1q_u64(sub_ptr.add(12), b12);
            vst1q_u64(sub_ptr.add(14), b012);

            for bit_in_byte in 3..8 {
                let b_vec = fold_one_slot_u64_pair_neon(cur_lo, cur_hi, base_table_ptr);
                let next_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(cur_lo, cur_hi);
                let next_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(cur_lo, cur_hi);
                cur_lo = next_lo;
                cur_hi = next_hi;

                let half = 1usize << bit_in_byte;
                let mut j = 0usize;
                while j < half {
                    let s0 = vld1q_u64(sub_ptr.add(2 * j));
                    let s1 = vld1q_u64(sub_ptr.add(2 * (j + 1)));
                    let s2 = vld1q_u64(sub_ptr.add(2 * (j + 2)));
                    let s3 = vld1q_u64(sub_ptr.add(2 * (j + 3)));
                    let s4 = vld1q_u64(sub_ptr.add(2 * (j + 4)));
                    let s5 = vld1q_u64(sub_ptr.add(2 * (j + 5)));
                    let s6 = vld1q_u64(sub_ptr.add(2 * (j + 6)));
                    let s7 = vld1q_u64(sub_ptr.add(2 * (j + 7)));
                    vst1q_u64(sub_ptr.add(2 * (half + j)), veorq_u64(s0, b_vec));
                    vst1q_u64(sub_ptr.add(2 * (half + j + 1)), veorq_u64(s1, b_vec));
                    vst1q_u64(sub_ptr.add(2 * (half + j + 2)), veorq_u64(s2, b_vec));
                    vst1q_u64(sub_ptr.add(2 * (half + j + 3)), veorq_u64(s3, b_vec));
                    vst1q_u64(sub_ptr.add(2 * (half + j + 4)), veorq_u64(s4, b_vec));
                    vst1q_u64(sub_ptr.add(2 * (half + j + 5)), veorq_u64(s5, b_vec));
                    vst1q_u64(sub_ptr.add(2 * (half + j + 6)), veorq_u64(s6, b_vec));
                    vst1q_u64(sub_ptr.add(2 * (half + j + 7)), veorq_u64(s7, b_vec));
                    j += 8;
                }
            }
        }
    }
}

#[inline]
pub(crate) fn prepare_pair_factored_eq_lo(eq_lo: &[F128]) -> Option<(Vec<F128>, F128)> {
    let b = eq_lo.len();
    if b < 4 || !b.is_multiple_of(4) {
        return None;
    }
    let half = b / 2;
    let mut u_half = Vec::with_capacity(half);
    let mut r0 = F128::ZERO;
    for t in 0..half {
        let even = eq_lo[2 * t];
        let odd = eq_lo[2 * t + 1];
        u_half.push(even + odd);
        r0 += odd;
    }
    let one_plus_r0 = F128::ONE + r0;
    if u_half[0] * one_plus_r0 != eq_lo[0] || u_half[1] * r0 != eq_lo[3] {
        return None;
    }
    Some((u_half, r0))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,pclmulqdq")]
#[inline]
unsafe fn build_scaled_fold_2byte_table_pair_m256i(
    even_lo: &mut u64,
    even_hi: &mut u64,
    odd_lo: &mut u64,
    odd_hi: &mut u64,
    base_table_ptr: *const F128,
    out_2byte_ptr: *mut core::arch::x86_64::__m256i,
) {
    use core::arch::x86_64::*;
    unsafe {
        let zero = _mm256_setzero_si256();
        let mut ev_lo = *even_lo;
        let mut ev_hi = *even_hi;
        let mut od_lo = *odd_lo;
        let mut od_hi = *odd_hi;
        for byte_idx in 0..2 {
            let sub_ptr = out_2byte_ptr.add(byte_idx * FOLD_TABLE_SIZE);
            let b0_ev = fold_one_slot_u64_pair_m128i(ev_lo, ev_hi, base_table_ptr);
            let b0_od = fold_one_slot_u64_pair_m128i(od_lo, od_hi, base_table_ptr);
            let b0 = _mm256_set_m128i(b0_od, b0_ev);
            let n_ev_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(ev_lo, ev_hi);
            let n_ev_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(ev_lo, ev_hi);
            let n_od_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(od_lo, od_hi);
            let n_od_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(od_lo, od_hi);
            ev_lo = n_ev_lo;
            ev_hi = n_ev_hi;
            od_lo = n_od_lo;
            od_hi = n_od_hi;

            let b1_ev = fold_one_slot_u64_pair_m128i(ev_lo, ev_hi, base_table_ptr);
            let b1_od = fold_one_slot_u64_pair_m128i(od_lo, od_hi, base_table_ptr);
            let b1 = _mm256_set_m128i(b1_od, b1_ev);
            let n_ev_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(ev_lo, ev_hi);
            let n_ev_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(ev_lo, ev_hi);
            let n_od_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(od_lo, od_hi);
            let n_od_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(od_lo, od_hi);
            ev_lo = n_ev_lo;
            ev_hi = n_ev_hi;
            od_lo = n_od_lo;
            od_hi = n_od_hi;

            let b2_ev = fold_one_slot_u64_pair_m128i(ev_lo, ev_hi, base_table_ptr);
            let b2_od = fold_one_slot_u64_pair_m128i(od_lo, od_hi, base_table_ptr);
            let b2 = _mm256_set_m128i(b2_od, b2_ev);
            let n_ev_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(ev_lo, ev_hi);
            let n_ev_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(ev_lo, ev_hi);
            let n_od_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(od_lo, od_hi);
            let n_od_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(od_lo, od_hi);
            ev_lo = n_ev_lo;
            ev_hi = n_ev_hi;
            od_lo = n_od_lo;
            od_hi = n_od_hi;

            let b01 = _mm256_xor_si256(b0, b1);
            let b02 = _mm256_xor_si256(b0, b2);
            let b12 = _mm256_xor_si256(b1, b2);
            let b012 = _mm256_xor_si256(b01, b2);
            _mm256_store_si256(sub_ptr, zero);
            _mm256_store_si256(sub_ptr.add(1), b0);
            _mm256_store_si256(sub_ptr.add(2), b1);
            _mm256_store_si256(sub_ptr.add(3), b01);
            _mm256_store_si256(sub_ptr.add(4), b2);
            _mm256_store_si256(sub_ptr.add(5), b02);
            _mm256_store_si256(sub_ptr.add(6), b12);
            _mm256_store_si256(sub_ptr.add(7), b012);

            for bit_in_byte in 3..8 {
                let b_ev = fold_one_slot_u64_pair_m128i(ev_lo, ev_hi, base_table_ptr);
                let b_od = fold_one_slot_u64_pair_m128i(od_lo, od_hi, base_table_ptr);
                let b_vec = _mm256_set_m128i(b_od, b_ev);
                let n_ev_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(ev_lo, ev_hi);
                let n_ev_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(ev_lo, ev_hi);
                let n_od_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(od_lo, od_hi);
                let n_od_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(od_lo, od_hi);
                ev_lo = n_ev_lo;
                ev_hi = n_ev_hi;
                od_lo = n_od_lo;
                od_hi = n_od_hi;

                let half = 1usize << bit_in_byte;
                let mut j = 0usize;
                while j < half {
                    let s0 = _mm256_load_si256(sub_ptr.add(j));
                    let s1 = _mm256_load_si256(sub_ptr.add(j + 1));
                    let s2 = _mm256_load_si256(sub_ptr.add(j + 2));
                    let s3 = _mm256_load_si256(sub_ptr.add(j + 3));
                    let s4 = _mm256_load_si256(sub_ptr.add(j + 4));
                    let s5 = _mm256_load_si256(sub_ptr.add(j + 5));
                    let s6 = _mm256_load_si256(sub_ptr.add(j + 6));
                    let s7 = _mm256_load_si256(sub_ptr.add(j + 7));
                    _mm256_store_si256(sub_ptr.add(half + j), _mm256_xor_si256(s0, b_vec));
                    _mm256_store_si256(sub_ptr.add(half + j + 1), _mm256_xor_si256(s1, b_vec));
                    _mm256_store_si256(sub_ptr.add(half + j + 2), _mm256_xor_si256(s2, b_vec));
                    _mm256_store_si256(sub_ptr.add(half + j + 3), _mm256_xor_si256(s3, b_vec));
                    _mm256_store_si256(sub_ptr.add(half + j + 4), _mm256_xor_si256(s4, b_vec));
                    _mm256_store_si256(sub_ptr.add(half + j + 5), _mm256_xor_si256(s5, b_vec));
                    _mm256_store_si256(sub_ptr.add(half + j + 6), _mm256_xor_si256(s6, b_vec));
                    _mm256_store_si256(sub_ptr.add(half + j + 7), _mm256_xor_si256(s7, b_vec));
                    j += 8;
                }
            }
        }
        *even_lo = ev_lo;
        *even_hi = ev_hi;
        *odd_lo = od_lo;
        *odd_hi = od_hi;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[target_feature(enable = "avx2,pclmulqdq")]
#[allow(clippy::too_many_arguments)]
unsafe fn combine_two_deferred_block_and_round0_pair_factored_avx2(
    out_block: &mut [F128],
    witness_block: &[F128],
    u0_half: &[F128],
    r0_0: F128,
    e_hi0: F128,
    table0: &[F128],
    u1_half: &[F128],
    r0_1: F128,
    e_hi1: F128,
    table1: &[F128],
) -> (F128, F128) {
    use core::arch::x86_64::*;
    unsafe {
        let b = out_block.len();
        let out_ptr = out_block.as_mut_ptr();
        let wit_ptr = witness_block.as_ptr();
        let u0_ptr = u0_half.as_ptr();
        let u1_ptr = u1_half.as_ptr();
        let t0_ptr = table0.as_ptr();
        let t1_ptr = table1.as_ptr();

        let e0_odd = e_hi0 * r0_0;
        let e0_even = e_hi0 + e0_odd;
        let e1_odd = e_hi1 * r0_1;
        let e1_even = e_hi1 + e1_odd;

        let mut u0_ll = _mm_setzero_si128();
        let mut u0_mid = _mm_setzero_si128();
        let mut u0_hh = _mm_setzero_si128();
        let mut u2_ll = _mm_setzero_si128();
        let mut u2_mid = _mm_setzero_si128();
        let mut u2_hh = _mm_setzero_si128();

        const TILE_Q: usize = 1024;
        #[repr(align(64))]
        struct AlignedPairScratch {
            sub_t: [core::arch::x86_64::__m256i; 2 * FOLD_TABLE_SIZE],
            tile: [core::arch::x86_64::__m256i; TILE_Q * 2],
        }
        std::thread_local! {
            static PAIR_SCRATCH: core::cell::UnsafeCell<AlignedPairScratch> = const {
                core::cell::UnsafeCell::new(unsafe { core::mem::zeroed() })
            };
        }
        let scratch_ptr = PAIR_SCRATCH.with(|c| c.get());
        let sub_ptr = (*scratch_ptr).sub_t.as_mut_ptr();
        let sub_cptr = sub_ptr as *const __m256i;
        let tile_ptr = (*scratch_ptr).tile.as_mut_ptr();

        macro_rules! fold_u16_m256i {
            ($w16:expr) => {{
                let b0 = ($w16 & 0xFF) as usize;
                let b1 = (($w16 >> 8) & 0xFF) as usize;
                _mm256_xor_si256(
                    _mm256_load_si256(sub_cptr.add(b0)),
                    _mm256_load_si256(sub_cptr.add(FOLD_TABLE_SIZE + b1)),
                )
            }};
        }

        let n_q = b / 4;
        let mut q0 = 0usize;
        while q0 < n_q {
            let q1 = (q0 + TILE_Q).min(n_q);
            let mut ev0_lo = e0_even.lo;
            let mut ev0_hi = e0_even.hi;
            let mut od0_lo = e0_odd.lo;
            let mut od0_hi = e0_odd.hi;

            build_scaled_fold_2byte_table_pair_m256i(
                &mut ev0_lo,
                &mut ev0_hi,
                &mut od0_lo,
                &mut od0_hi,
                t0_ptr,
                sub_ptr,
            );
            for q in q0..q1 {
                let p = 2 * q;
                let ti = 2 * (q - q0);
                let ua = (*u0_ptr.add(p)).lo;
                let ub = (*u0_ptr.add(p + 1)).lo;
                *tile_ptr.add(ti) = fold_u16_m256i!(ua);
                *tile_ptr.add(ti + 1) = fold_u16_m256i!(ub);
            }
            for shift in [16usize, 32, 48] {
                build_scaled_fold_2byte_table_pair_m256i(
                    &mut ev0_lo,
                    &mut ev0_hi,
                    &mut od0_lo,
                    &mut od0_hi,
                    t0_ptr,
                    sub_ptr,
                );
                for q in q0..q1 {
                    let p = 2 * q;
                    let ti = 2 * (q - q0);
                    let ua = (*u0_ptr.add(p)).lo >> shift;
                    let ub = (*u0_ptr.add(p + 1)).lo >> shift;
                    *tile_ptr.add(ti) = _mm256_xor_si256(*tile_ptr.add(ti), fold_u16_m256i!(ua));
                    *tile_ptr.add(ti + 1) =
                        _mm256_xor_si256(*tile_ptr.add(ti + 1), fold_u16_m256i!(ub));
                }
            }
            for shift in [0usize, 16, 32, 48] {
                build_scaled_fold_2byte_table_pair_m256i(
                    &mut ev0_lo,
                    &mut ev0_hi,
                    &mut od0_lo,
                    &mut od0_hi,
                    t0_ptr,
                    sub_ptr,
                );
                for q in q0..q1 {
                    let p = 2 * q;
                    let ti = 2 * (q - q0);
                    let ua = (*u0_ptr.add(p)).hi >> shift;
                    let ub = (*u0_ptr.add(p + 1)).hi >> shift;
                    *tile_ptr.add(ti) = _mm256_xor_si256(*tile_ptr.add(ti), fold_u16_m256i!(ua));
                    *tile_ptr.add(ti + 1) =
                        _mm256_xor_si256(*tile_ptr.add(ti + 1), fold_u16_m256i!(ub));
                }
            }

            let mut ev1_lo = e1_even.lo;
            let mut ev1_hi = e1_even.hi;
            let mut od1_lo = e1_odd.lo;
            let mut od1_hi = e1_odd.hi;
            for shift in [0usize, 16, 32, 48] {
                build_scaled_fold_2byte_table_pair_m256i(
                    &mut ev1_lo,
                    &mut ev1_hi,
                    &mut od1_lo,
                    &mut od1_hi,
                    t1_ptr,
                    sub_ptr,
                );
                for q in q0..q1 {
                    let p = 2 * q;
                    let ti = 2 * (q - q0);
                    let ua = (*u1_ptr.add(p)).lo >> shift;
                    let ub = (*u1_ptr.add(p + 1)).lo >> shift;
                    *tile_ptr.add(ti) = _mm256_xor_si256(*tile_ptr.add(ti), fold_u16_m256i!(ua));
                    *tile_ptr.add(ti + 1) =
                        _mm256_xor_si256(*tile_ptr.add(ti + 1), fold_u16_m256i!(ub));
                }
            }
            for shift in [0usize, 16, 32] {
                build_scaled_fold_2byte_table_pair_m256i(
                    &mut ev1_lo,
                    &mut ev1_hi,
                    &mut od1_lo,
                    &mut od1_hi,
                    t1_ptr,
                    sub_ptr,
                );
                for q in q0..q1 {
                    let p = 2 * q;
                    let ti = 2 * (q - q0);
                    let ua = (*u1_ptr.add(p)).hi >> shift;
                    let ub = (*u1_ptr.add(p + 1)).hi >> shift;
                    *tile_ptr.add(ti) = _mm256_xor_si256(*tile_ptr.add(ti), fold_u16_m256i!(ua));
                    *tile_ptr.add(ti + 1) =
                        _mm256_xor_si256(*tile_ptr.add(ti + 1), fold_u16_m256i!(ub));
                }
            }
            build_scaled_fold_2byte_table_pair_m256i(
                &mut ev1_lo,
                &mut ev1_hi,
                &mut od1_lo,
                &mut od1_hi,
                t1_ptr,
                sub_ptr,
            );
            for q in q0..q1 {
                let p = 2 * q;
                let i = 4 * q;
                let ti = 2 * (q - q0);
                let ua = (*u1_ptr.add(p)).hi >> 48;
                let ub = (*u1_ptr.add(p + 1)).hi >> 48;
                let s01 = _mm256_xor_si256(*tile_ptr.add(ti), fold_u16_m256i!(ua));
                let s23 = _mm256_xor_si256(*tile_ptr.add(ti + 1), fold_u16_m256i!(ub));

                let s0 = _mm256_castsi256_si128(s01);
                let s1 = _mm256_extracti128_si256::<1>(s01);
                let s2 = _mm256_castsi256_si128(s23);
                let s3 = _mm256_extracti128_si256::<1>(s23);
                _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, s01);
                _mm256_storeu_si256(out_ptr.add(i + 2) as *mut __m256i, s23);

                let a0 = _mm_loadu_si128(wit_ptr.add(i) as *const __m128i);
                let a1 = _mm_loadu_si128(wit_ptr.add(i + 1) as *const __m128i);
                let a2 = _mm_loadu_si128(wit_ptr.add(i + 2) as *const __m128i);
                let a3 = _mm_loadu_si128(wit_ptr.add(i + 3) as *const __m128i);
                crate::gf2_kernels::k0_f128_mul_acc2_m128i(
                    a0,
                    s0,
                    a2,
                    s2,
                    &mut u0_ll,
                    &mut u0_mid,
                    &mut u0_hh,
                );
                crate::gf2_kernels::k0_f128_mul_acc2_m128i(
                    _mm_xor_si128(a0, a1),
                    _mm_xor_si128(s0, s1),
                    _mm_xor_si128(a2, a3),
                    _mm_xor_si128(s2, s3),
                    &mut u2_ll,
                    &mut u2_mid,
                    &mut u2_hh,
                );
            }
            q0 = q1;
        }
        let r0 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u0_ll, u0_mid, u0_hh);
        let r2 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u2_ll, u2_mid, u2_hh);
        (
            core::mem::transmute::<__m128i, F128>(r0),
            core::mem::transmute::<__m128i, F128>(r2),
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn build_scaled_fold_byte_table_pair_neon(
    e_even: F128,
    e_odd: F128,
    base_table_ptr: *const F128,
    out_table_ptr: *mut F128,
) {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let mut even_lo = e_even.lo;
        let mut even_hi = e_even.hi;
        let mut odd_lo = e_odd.lo;
        let mut odd_hi = e_odd.hi;
        for byte_idx in 0..FOLD_N_BYTES {
            let sub_ptr = out_table_ptr.add(2 * byte_idx * FOLD_TABLE_SIZE) as *mut u64;
            let b0_ev = fold_one_slot_u64_pair_neon(even_lo, even_hi, base_table_ptr);
            let b0_od = fold_one_slot_u64_pair_neon(odd_lo, odd_hi, base_table_ptr);
            let n_ev_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(even_lo, even_hi);
            let n_ev_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(even_lo, even_hi);
            let n_od_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(odd_lo, odd_hi);
            let n_od_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(odd_lo, odd_hi);
            even_lo = n_ev_lo;
            even_hi = n_ev_hi;
            odd_lo = n_od_lo;
            odd_hi = n_od_hi;

            let b1_ev = fold_one_slot_u64_pair_neon(even_lo, even_hi, base_table_ptr);
            let b1_od = fold_one_slot_u64_pair_neon(odd_lo, odd_hi, base_table_ptr);
            let n_ev_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(even_lo, even_hi);
            let n_ev_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(even_lo, even_hi);
            let n_od_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(odd_lo, odd_hi);
            let n_od_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(odd_lo, odd_hi);
            even_lo = n_ev_lo;
            even_hi = n_ev_hi;
            odd_lo = n_od_lo;
            odd_hi = n_od_hi;

            let b2_ev = fold_one_slot_u64_pair_neon(even_lo, even_hi, base_table_ptr);
            let b2_od = fold_one_slot_u64_pair_neon(odd_lo, odd_hi, base_table_ptr);
            let n_ev_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(even_lo, even_hi);
            let n_ev_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(even_lo, even_hi);
            let n_od_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(odd_lo, odd_hi);
            let n_od_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(odd_lo, odd_hi);
            even_lo = n_ev_lo;
            even_hi = n_ev_hi;
            odd_lo = n_od_lo;
            odd_hi = n_od_hi;

            let b01_ev = veorq_u64(b0_ev, b1_ev);
            let b02_ev = veorq_u64(b0_ev, b2_ev);
            let b12_ev = veorq_u64(b1_ev, b2_ev);
            let b012_ev = veorq_u64(b01_ev, b2_ev);
            let b01_od = veorq_u64(b0_od, b1_od);
            let b02_od = veorq_u64(b0_od, b2_od);
            let b12_od = veorq_u64(b1_od, b2_od);
            let b012_od = veorq_u64(b01_od, b2_od);

            vst1q_u64(sub_ptr, zero);
            vst1q_u64(sub_ptr.add(2), zero);
            vst1q_u64(sub_ptr.add(4), b0_ev);
            vst1q_u64(sub_ptr.add(6), b0_od);
            vst1q_u64(sub_ptr.add(8), b1_ev);
            vst1q_u64(sub_ptr.add(10), b1_od);
            vst1q_u64(sub_ptr.add(12), b01_ev);
            vst1q_u64(sub_ptr.add(14), b01_od);
            vst1q_u64(sub_ptr.add(16), b2_ev);
            vst1q_u64(sub_ptr.add(18), b2_od);
            vst1q_u64(sub_ptr.add(20), b02_ev);
            vst1q_u64(sub_ptr.add(22), b02_od);
            vst1q_u64(sub_ptr.add(24), b12_ev);
            vst1q_u64(sub_ptr.add(26), b12_od);
            vst1q_u64(sub_ptr.add(28), b012_ev);
            vst1q_u64(sub_ptr.add(30), b012_od);

            for bit_in_byte in 3..8 {
                let b_ev = fold_one_slot_u64_pair_neon(even_lo, even_hi, base_table_ptr);
                let b_od = fold_one_slot_u64_pair_neon(odd_lo, odd_hi, base_table_ptr);
                let n_ev_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(even_lo, even_hi);
                let n_ev_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(even_lo, even_hi);
                let n_od_lo = crate::gf2_kernels::gf2_kern_mul_by_x_lo(odd_lo, odd_hi);
                let n_od_hi = crate::gf2_kernels::gf2_kern_mul_by_x_hi(odd_lo, odd_hi);
                even_lo = n_ev_lo;
                even_hi = n_ev_hi;
                odd_lo = n_od_lo;
                odd_hi = n_od_hi;

                let half = 1usize << bit_in_byte;
                let mut j = 0usize;
                while j < half {
                    let s0_ev = vld1q_u64(sub_ptr.add(4 * j));
                    let s0_od = vld1q_u64(sub_ptr.add(4 * j + 2));
                    let s1_ev = vld1q_u64(sub_ptr.add(4 * (j + 1)));
                    let s1_od = vld1q_u64(sub_ptr.add(4 * (j + 1) + 2));
                    let s2_ev = vld1q_u64(sub_ptr.add(4 * (j + 2)));
                    let s2_od = vld1q_u64(sub_ptr.add(4 * (j + 2) + 2));
                    let s3_ev = vld1q_u64(sub_ptr.add(4 * (j + 3)));
                    let s3_od = vld1q_u64(sub_ptr.add(4 * (j + 3) + 2));
                    vst1q_u64(sub_ptr.add(4 * (half + j)), veorq_u64(s0_ev, b_ev));
                    vst1q_u64(sub_ptr.add(4 * (half + j) + 2), veorq_u64(s0_od, b_od));
                    vst1q_u64(sub_ptr.add(4 * (half + j + 1)), veorq_u64(s1_ev, b_ev));
                    vst1q_u64(sub_ptr.add(4 * (half + j + 1) + 2), veorq_u64(s1_od, b_od));
                    vst1q_u64(sub_ptr.add(4 * (half + j + 2)), veorq_u64(s2_ev, b_ev));
                    vst1q_u64(sub_ptr.add(4 * (half + j + 2) + 2), veorq_u64(s2_od, b_od));
                    vst1q_u64(sub_ptr.add(4 * (half + j + 3)), veorq_u64(s3_ev, b_ev));
                    vst1q_u64(sub_ptr.add(4 * (half + j + 3) + 2), veorq_u64(s3_od, b_od));
                    j += 4;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn combine_two_deferred_block_and_round0_pair_factored(
    out_block: &mut [F128],
    witness_block: &[F128],
    u0_half: &[F128],
    r0_0: F128,
    e_hi0: F128,
    table0: &[F128],
    u1_half: &[F128],
    r0_1: F128,
    e_hi1: F128,
    table1: &[F128],
) -> (F128, F128) {
    let b = out_block.len();
    debug_assert_eq!(witness_block.len(), b);
    debug_assert_eq!(u0_half.len(), b / 2);
    debug_assert_eq!(u1_half.len(), b / 2);
    debug_assert_eq!(table0.len(), FOLD_N_BYTES * FOLD_TABLE_SIZE);
    debug_assert_eq!(table1.len(), FOLD_N_BYTES * FOLD_TABLE_SIZE);
    debug_assert_eq!(b % 4, 0);

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                combine_two_deferred_block_and_round0_pair_factored_avx2(
                    out_block,
                    witness_block,
                    u0_half,
                    r0_0,
                    e_hi0,
                    table0,
                    u1_half,
                    r0_1,
                    e_hi1,
                    table1,
                )
            };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let out_ptr = out_block.as_mut_ptr() as *mut u64;
        let wit_ptr = witness_block.as_ptr() as *const u64;
        let u0_ptr = u0_half.as_ptr();
        let u1_ptr = u1_half.as_ptr();
        let t0_ptr = table0.as_ptr();
        let t1_ptr = table1.as_ptr();

        let e0_odd = e_hi0 * r0_0;
        let e0_even = e_hi0 + e0_odd;
        let e1_odd = e_hi1 * r0_1;
        let e1_even = e_hi1 + e1_odd;

        let mut u0_ll = z;
        let mut u0_mid = z;
        let mut u0_hh = z;
        let mut u2_ll = z;
        let mut u2_mid = z;
        let mut u2_hh = z;

        const TILE_Q: usize = 1024;
        #[repr(align(64))]
        struct AlignedPairScratchNeon {
            t0: [F128; 2 * FOLD_N_BYTES * FOLD_TABLE_SIZE],
            t1: [F128; 2 * FOLD_N_BYTES * FOLD_TABLE_SIZE],
            tile: [core::arch::aarch64::uint64x2_t; TILE_Q * 4],
        }
        std::thread_local! {
            static PAIR_SCRATCH_NEON: core::cell::UnsafeCell<AlignedPairScratchNeon> = const {
                core::cell::UnsafeCell::new(unsafe { core::mem::zeroed() })
            };
        }
        let scratch_ptr = PAIR_SCRATCH_NEON.with(|c| c.get());
        let st0_ptr = (*scratch_ptr).t0.as_mut_ptr();
        let st1_ptr = (*scratch_ptr).t1.as_mut_ptr();
        let tile_buf = &mut (*scratch_ptr).tile;

        let st0_q0_ptr = st0_ptr as *const F128;
        let st0_q1_ptr = (st0_ptr as *const F128).add(8 * FOLD_TABLE_SIZE);
        let st0_q2_ptr = (st0_ptr as *const F128).add(16 * FOLD_TABLE_SIZE);
        let st0_q3_ptr = (st0_ptr as *const F128).add(24 * FOLD_TABLE_SIZE);

        let st1_q0_ptr = st1_ptr as *const F128;
        let st1_q1_ptr = (st1_ptr as *const F128).add(8 * FOLD_TABLE_SIZE);
        let st1_q2_ptr = (st1_ptr as *const F128).add(16 * FOLD_TABLE_SIZE);
        let st1_q3_ptr = (st1_ptr as *const F128).add(24 * FOLD_TABLE_SIZE);

        macro_rules! fold_q_neon_pair {
            ($w32:expr, $q_ptr:expr) => {{
                let base = $q_ptr as *const u8;
                let b0 = base.add(32 * (($w32 & 0xFF) as usize));
                let b1 = base.add(32 * (FOLD_TABLE_SIZE + (($w32 >> 8) & 0xFF) as usize));
                let b2 = base.add(32 * (2 * FOLD_TABLE_SIZE + (($w32 >> 16) & 0xFF) as usize));
                let b3 = base.add(32 * (3 * FOLD_TABLE_SIZE + (($w32 >> 24) & 0xFF) as usize));
                let l0_ev = vld1q_u8(b0);
                let l0_od = vld1q_u8(b0.add(16));
                let l1_ev = vld1q_u8(b1);
                let l1_od = vld1q_u8(b1.add(16));
                let l2_ev = vld1q_u8(b2);
                let l2_od = vld1q_u8(b2.add(16));
                let l3_ev = vld1q_u8(b3);
                let l3_od = vld1q_u8(b3.add(16));
                (
                    vreinterpretq_u64_u8(veorq_u8(veorq_u8(l0_ev, l1_ev), veorq_u8(l2_ev, l3_ev))),
                    vreinterpretq_u64_u8(veorq_u8(veorq_u8(l0_od, l1_od), veorq_u8(l2_od, l3_od))),
                )
            }};
        }

        build_scaled_fold_byte_table_pair_neon(e0_even, e0_odd, t0_ptr, st0_ptr);
        build_scaled_fold_byte_table_pair_neon(e1_even, e1_odd, t1_ptr, st1_ptr);
        let n_q = b / 4;
        let mut q0 = 0usize;
        while q0 < n_q {
            let q1 = (q0 + TILE_Q).min(n_q);
            for q in q0..q1 {
                let p = 2 * q;
                let ti = 4 * (q - q0);
                let ua = (*u0_ptr.add(p)).lo as u32;
                let ub = (*u0_ptr.add(p + 1)).lo as u32;
                let (s0, s1) = fold_q_neon_pair!(ua, st0_q0_ptr);
                let (s2, s3) = fold_q_neon_pair!(ub, st0_q0_ptr);
                *tile_buf.get_unchecked_mut(ti) = s0;
                *tile_buf.get_unchecked_mut(ti + 1) = s1;
                *tile_buf.get_unchecked_mut(ti + 2) = s2;
                *tile_buf.get_unchecked_mut(ti + 3) = s3;
            }
            for q in q0..q1 {
                let p = 2 * q;
                let ti = 4 * (q - q0);
                let ua = ((*u0_ptr.add(p)).lo >> 32) as u32;
                let ub = ((*u0_ptr.add(p + 1)).lo >> 32) as u32;
                let (s0, s1) = fold_q_neon_pair!(ua, st0_q1_ptr);
                let (s2, s3) = fold_q_neon_pair!(ub, st0_q1_ptr);
                *tile_buf.get_unchecked_mut(ti) = veorq_u64(*tile_buf.get_unchecked(ti), s0);
                *tile_buf.get_unchecked_mut(ti + 1) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 1), s1);
                *tile_buf.get_unchecked_mut(ti + 2) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 2), s2);
                *tile_buf.get_unchecked_mut(ti + 3) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 3), s3);
            }
            for q in q0..q1 {
                let p = 2 * q;
                let ti = 4 * (q - q0);
                let ua = (*u0_ptr.add(p)).hi as u32;
                let ub = (*u0_ptr.add(p + 1)).hi as u32;
                let (s0, s1) = fold_q_neon_pair!(ua, st0_q2_ptr);
                let (s2, s3) = fold_q_neon_pair!(ub, st0_q2_ptr);
                *tile_buf.get_unchecked_mut(ti) = veorq_u64(*tile_buf.get_unchecked(ti), s0);
                *tile_buf.get_unchecked_mut(ti + 1) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 1), s1);
                *tile_buf.get_unchecked_mut(ti + 2) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 2), s2);
                *tile_buf.get_unchecked_mut(ti + 3) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 3), s3);
            }
            for q in q0..q1 {
                let p = 2 * q;
                let ti = 4 * (q - q0);
                let ua = ((*u0_ptr.add(p)).hi >> 32) as u32;
                let ub = ((*u0_ptr.add(p + 1)).hi >> 32) as u32;
                let (h0, h1) = fold_q_neon_pair!(ua, st0_q3_ptr);
                let (h2, h3) = fold_q_neon_pair!(ub, st0_q3_ptr);
                *tile_buf.get_unchecked_mut(ti) = veorq_u64(*tile_buf.get_unchecked(ti), h0);
                *tile_buf.get_unchecked_mut(ti + 1) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 1), h1);
                *tile_buf.get_unchecked_mut(ti + 2) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 2), h2);
                *tile_buf.get_unchecked_mut(ti + 3) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 3), h3);
            }
            for q in q0..q1 {
                let p = 2 * q;
                let ti = 4 * (q - q0);
                let ua = (*u1_ptr.add(p)).lo as u32;
                let ub = (*u1_ptr.add(p + 1)).lo as u32;
                let (s0, s1) = fold_q_neon_pair!(ua, st1_q0_ptr);
                let (s2, s3) = fold_q_neon_pair!(ub, st1_q0_ptr);
                *tile_buf.get_unchecked_mut(ti) = veorq_u64(*tile_buf.get_unchecked(ti), s0);
                *tile_buf.get_unchecked_mut(ti + 1) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 1), s1);
                *tile_buf.get_unchecked_mut(ti + 2) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 2), s2);
                *tile_buf.get_unchecked_mut(ti + 3) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 3), s3);
            }
            for q in q0..q1 {
                let p = 2 * q;
                let ti = 4 * (q - q0);
                let ua = ((*u1_ptr.add(p)).lo >> 32) as u32;
                let ub = ((*u1_ptr.add(p + 1)).lo >> 32) as u32;
                let (s0, s1) = fold_q_neon_pair!(ua, st1_q1_ptr);
                let (s2, s3) = fold_q_neon_pair!(ub, st1_q1_ptr);
                *tile_buf.get_unchecked_mut(ti) = veorq_u64(*tile_buf.get_unchecked(ti), s0);
                *tile_buf.get_unchecked_mut(ti + 1) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 1), s1);
                *tile_buf.get_unchecked_mut(ti + 2) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 2), s2);
                *tile_buf.get_unchecked_mut(ti + 3) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 3), s3);
            }
            for q in q0..q1 {
                let p = 2 * q;
                let ti = 4 * (q - q0);
                let ua = (*u1_ptr.add(p)).hi as u32;
                let ub = (*u1_ptr.add(p + 1)).hi as u32;
                let (s0, s1) = fold_q_neon_pair!(ua, st1_q2_ptr);
                let (s2, s3) = fold_q_neon_pair!(ub, st1_q2_ptr);
                *tile_buf.get_unchecked_mut(ti) = veorq_u64(*tile_buf.get_unchecked(ti), s0);
                *tile_buf.get_unchecked_mut(ti + 1) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 1), s1);
                *tile_buf.get_unchecked_mut(ti + 2) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 2), s2);
                *tile_buf.get_unchecked_mut(ti + 3) =
                    veorq_u64(*tile_buf.get_unchecked(ti + 3), s3);
            }
            for q in q0..q1 {
                let p = 2 * q;
                let idx = 8 * q;
                let ti = 4 * (q - q0);
                let ua = ((*u1_ptr.add(p)).hi >> 32) as u32;
                let ub = ((*u1_ptr.add(p + 1)).hi >> 32) as u32;
                let (h0, h1) = fold_q_neon_pair!(ua, st1_q3_ptr);
                let (h2, h3) = fold_q_neon_pair!(ub, st1_q3_ptr);
                let s0 = veorq_u64(*tile_buf.get_unchecked(ti), h0);
                let s1 = veorq_u64(*tile_buf.get_unchecked(ti + 1), h1);
                let s2 = veorq_u64(*tile_buf.get_unchecked(ti + 2), h2);
                let s3 = veorq_u64(*tile_buf.get_unchecked(ti + 3), h3);
                vst1q_u64(out_ptr.add(idx), s0);
                vst1q_u64(out_ptr.add(idx + 2), s1);
                vst1q_u64(out_ptr.add(idx + 4), s2);
                vst1q_u64(out_ptr.add(idx + 6), s3);

                let a0 = vld1q_u64(wit_ptr.add(idx));
                let a1 = vld1q_u64(wit_ptr.add(idx + 2));
                let a2 = vld1q_u64(wit_ptr.add(idx + 4));
                let a3 = vld1q_u64(wit_ptr.add(idx + 6));
                crate::gf2_kernels::k0_f128_mul_acc_neon(
                    a0,
                    s0,
                    &mut u0_ll,
                    &mut u0_mid,
                    &mut u0_hh,
                );
                crate::gf2_kernels::k0_f128_mul_acc_neon(
                    a2,
                    s2,
                    &mut u0_ll,
                    &mut u0_mid,
                    &mut u0_hh,
                );
                crate::gf2_kernels::k0_f128_mul_acc_neon(
                    veorq_u64(a0, a1),
                    veorq_u64(s0, s1),
                    &mut u2_ll,
                    &mut u2_mid,
                    &mut u2_hh,
                );
                crate::gf2_kernels::k0_f128_mul_acc_neon(
                    veorq_u64(a2, a3),
                    veorq_u64(s2, s3),
                    &mut u2_ll,
                    &mut u2_mid,
                    &mut u2_hh,
                );
            }
            q0 = q1;
        }
        let r0 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u0_ll, u0_mid, u0_hh);
        let r2 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u2_ll, u2_mid, u2_hh);
        return (
            core::mem::transmute::<uint64x2_t, F128>(r0),
            core::mem::transmute::<uint64x2_t, F128>(r2),
        );
    }

    #[allow(unreachable_code)]
    {
        let e0_odd = e_hi0 * r0_0;
        let e0_even = e_hi0 + e0_odd;
        let e1_odd = e_hi1 * r0_1;
        let e1_even = e_hi1 + e1_odd;
        let mut u0_unred = crate::field::F256Unreduced::ZERO;
        let mut u2_unred = crate::field::F256Unreduced::ZERO;
        for q in 0..(b / 4) {
            let p = 2 * q;
            let i = 4 * q;
            let ua0 = u0_half[p];
            let ub0 = u0_half[p + 1];
            let ua1 = u1_half[p];
            let ub1 = u1_half[p + 1];
            let s0 = fold_one_slot(ua0 * e0_even, table0) + fold_one_slot(ua1 * e1_even, table1);
            let s1 = fold_one_slot(ua0 * e0_odd, table0) + fold_one_slot(ua1 * e1_odd, table1);
            let s2 = fold_one_slot(ub0 * e0_even, table0) + fold_one_slot(ub1 * e1_even, table1);
            let s3 = fold_one_slot(ub0 * e0_odd, table0) + fold_one_slot(ub1 * e1_odd, table1);
            out_block[i] = s0;
            out_block[i + 1] = s1;
            out_block[i + 2] = s2;
            out_block[i + 3] = s3;
            let a0 = witness_block[i];
            let a1 = witness_block[i + 1];
            let a2 = witness_block[i + 2];
            let a3 = witness_block[i + 3];
            u0_unred ^= a0.mul_unreduced(s0) ^ a2.mul_unreduced(s2);
            u2_unred ^= (a0 + a1).mul_unreduced(s0 + s1) ^ (a2 + a3).mul_unreduced(s2 + s3);
        }
        (u0_unred.reduce(), u2_unred.reduce())
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn combine_two_deferred_block_and_round0(
    out_block: &mut [F128],
    witness_block: &[F128],
    eq_lo0: &[F128],
    e_hi0: F128,
    table0: &[F128],
    eq_lo1: &[F128],
    e_hi1: F128,
    table1: &[F128],
) -> (F128, F128) {
    let b = out_block.len();
    debug_assert_eq!(witness_block.len(), b);
    debug_assert_eq!(eq_lo0.len(), b);
    debug_assert_eq!(eq_lo1.len(), b);
    debug_assert_eq!(table0.len(), FOLD_N_BYTES * FOLD_TABLE_SIZE);
    debug_assert_eq!(table1.len(), FOLD_N_BYTES * FOLD_TABLE_SIZE);
    debug_assert_eq!(b % 4, 0);

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let out_ptr = out_block.as_mut_ptr();
        let wit_ptr = witness_block.as_ptr();
        let lo0_ptr = eq_lo0.as_ptr();
        let lo1_ptr = eq_lo1.as_ptr();
        let t0_ptr = table0.as_ptr();
        let t1_ptr = table1.as_ptr();

        let mut u0_ll = _mm_setzero_si128();
        let mut u0_mid = _mm_setzero_si128();
        let mut u0_hh = _mm_setzero_si128();
        let mut u2_ll = _mm_setzero_si128();
        let mut u2_mid = _mm_setzero_si128();
        let mut u2_hh = _mm_setzero_si128();

        const TILE_Q: usize = 64;
        #[repr(align(64))]
        struct AlignedScratch {
            t0: [F128; FOLD_N_BYTES * FOLD_TABLE_SIZE],
            t1: [F128; FOLD_N_BYTES * FOLD_TABLE_SIZE],
            tile: [core::arch::x86_64::__m128i; TILE_Q * 4],
        }
        std::thread_local! {
            static SCRATCH: core::cell::UnsafeCell<AlignedScratch> = const {
                core::cell::UnsafeCell::new(unsafe { core::mem::zeroed() })
            };
        }
        let scratch_ptr = SCRATCH.with(|c| c.get());
        let tile_ptr = (*scratch_ptr).tile.as_mut_ptr();
        let n_q = b / 4;

        if b >= 256 {
            let st0_ptr = (*scratch_ptr).t0.as_mut_ptr();
            let st1_ptr = (*scratch_ptr).t1.as_mut_ptr();
            build_scaled_fold_byte_table_m128i(e_hi0, t0_ptr, st0_ptr);
            build_scaled_fold_byte_table_m128i(e_hi1, t1_ptr, st1_ptr);
            let st0_lo_ptr = st0_ptr as *const F128;
            let st0_hi_ptr = (st0_ptr as *const F128).add(8 * FOLD_TABLE_SIZE);
            let st1_lo_ptr = st1_ptr as *const F128;
            let st1_hi_ptr = (st1_ptr as *const F128).add(8 * FOLD_TABLE_SIZE);

            let mut q0 = 0usize;
            while q0 < n_q {
                let q1 = (q0 + TILE_Q).min(n_q);
                for q in q0..q1 {
                    let i = 4 * q;
                    let ti = 4 * (q - q0);
                    let e0_lo = (*lo0_ptr.add(i)).lo;
                    let e1_lo = (*lo0_ptr.add(i + 1)).lo;
                    let e2_lo = (*lo0_ptr.add(i + 2)).lo;
                    let e3_lo = (*lo0_ptr.add(i + 3)).lo;
                    *tile_ptr.add(ti) = fold_half_slot_u64_m128i(e0_lo, st0_lo_ptr);
                    *tile_ptr.add(ti + 1) = fold_half_slot_u64_m128i(e1_lo, st0_lo_ptr);
                    *tile_ptr.add(ti + 2) = fold_half_slot_u64_m128i(e2_lo, st0_lo_ptr);
                    *tile_ptr.add(ti + 3) = fold_half_slot_u64_m128i(e3_lo, st0_lo_ptr);
                }
                for q in q0..q1 {
                    let i = 4 * q;
                    let ti = 4 * (q - q0);
                    let e0_hi = (*lo0_ptr.add(i)).hi;
                    let e1_hi = (*lo0_ptr.add(i + 1)).hi;
                    let e2_hi = (*lo0_ptr.add(i + 2)).hi;
                    let e3_hi = (*lo0_ptr.add(i + 3)).hi;
                    *tile_ptr.add(ti) = _mm_xor_si128(
                        *tile_ptr.add(ti),
                        fold_half_slot_u64_m128i(e0_hi, st0_hi_ptr),
                    );
                    *tile_ptr.add(ti + 1) = _mm_xor_si128(
                        *tile_ptr.add(ti + 1),
                        fold_half_slot_u64_m128i(e1_hi, st0_hi_ptr),
                    );
                    *tile_ptr.add(ti + 2) = _mm_xor_si128(
                        *tile_ptr.add(ti + 2),
                        fold_half_slot_u64_m128i(e2_hi, st0_hi_ptr),
                    );
                    *tile_ptr.add(ti + 3) = _mm_xor_si128(
                        *tile_ptr.add(ti + 3),
                        fold_half_slot_u64_m128i(e3_hi, st0_hi_ptr),
                    );
                }
                for q in q0..q1 {
                    let i = 4 * q;
                    let ti = 4 * (q - q0);
                    let e0_lo = (*lo1_ptr.add(i)).lo;
                    let e1_lo = (*lo1_ptr.add(i + 1)).lo;
                    let e2_lo = (*lo1_ptr.add(i + 2)).lo;
                    let e3_lo = (*lo1_ptr.add(i + 3)).lo;
                    *tile_ptr.add(ti) = _mm_xor_si128(
                        *tile_ptr.add(ti),
                        fold_half_slot_u64_m128i(e0_lo, st1_lo_ptr),
                    );
                    *tile_ptr.add(ti + 1) = _mm_xor_si128(
                        *tile_ptr.add(ti + 1),
                        fold_half_slot_u64_m128i(e1_lo, st1_lo_ptr),
                    );
                    *tile_ptr.add(ti + 2) = _mm_xor_si128(
                        *tile_ptr.add(ti + 2),
                        fold_half_slot_u64_m128i(e2_lo, st1_lo_ptr),
                    );
                    *tile_ptr.add(ti + 3) = _mm_xor_si128(
                        *tile_ptr.add(ti + 3),
                        fold_half_slot_u64_m128i(e3_lo, st1_lo_ptr),
                    );
                }

                for q in q0..q1 {
                    let i = 4 * q;
                    let ti = 4 * (q - q0);
                    let e0_hi = (*lo1_ptr.add(i)).hi;
                    let e1_hi = (*lo1_ptr.add(i + 1)).hi;
                    let e2_hi = (*lo1_ptr.add(i + 2)).hi;
                    let e3_hi = (*lo1_ptr.add(i + 3)).hi;
                    let s0 = _mm_xor_si128(
                        *tile_ptr.add(ti),
                        fold_half_slot_u64_m128i(e0_hi, st1_hi_ptr),
                    );
                    let s1 = _mm_xor_si128(
                        *tile_ptr.add(ti + 1),
                        fold_half_slot_u64_m128i(e1_hi, st1_hi_ptr),
                    );
                    let s2 = _mm_xor_si128(
                        *tile_ptr.add(ti + 2),
                        fold_half_slot_u64_m128i(e2_hi, st1_hi_ptr),
                    );
                    let s3 = _mm_xor_si128(
                        *tile_ptr.add(ti + 3),
                        fold_half_slot_u64_m128i(e3_hi, st1_hi_ptr),
                    );
                    _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, s0);
                    _mm_storeu_si128(out_ptr.add(i + 1) as *mut __m128i, s1);
                    _mm_storeu_si128(out_ptr.add(i + 2) as *mut __m128i, s2);
                    _mm_storeu_si128(out_ptr.add(i + 3) as *mut __m128i, s3);

                    let a0 = _mm_loadu_si128(wit_ptr.add(i) as *const __m128i);
                    let a1 = _mm_loadu_si128(wit_ptr.add(i + 1) as *const __m128i);
                    let a2 = _mm_loadu_si128(wit_ptr.add(i + 2) as *const __m128i);
                    let a3 = _mm_loadu_si128(wit_ptr.add(i + 3) as *const __m128i);
                    crate::gf2_kernels::k0_f128_mul_acc2_m128i(
                        a0,
                        s0,
                        a2,
                        s2,
                        &mut u0_ll,
                        &mut u0_mid,
                        &mut u0_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc2_m128i(
                        _mm_xor_si128(a0, a1),
                        _mm_xor_si128(s0, s1),
                        _mm_xor_si128(a2, a3),
                        _mm_xor_si128(s2, s3),
                        &mut u2_ll,
                        &mut u2_mid,
                        &mut u2_hh,
                    );
                }
                q0 = q1;
            }
        } else {
            let ve0 = _mm_set_epi64x(e_hi0.hi as i64, e_hi0.lo as i64);
            let ve1 = _mm_set_epi64x(e_hi1.hi as i64, e_hi1.lo as i64);
            let (ve0_r0, ve0_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(ve0);
            let (ve1_r0, ve1_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(ve1);
            let mut q0 = 0usize;
            while q0 < n_q {
                let q1 = (q0 + TILE_Q).min(n_q);
                for q in q0..q1 {
                    let i = 4 * q;
                    let ti = 4 * (q - q0);
                    let l0 = _mm_loadu_si128(lo0_ptr.add(i) as *const __m128i);
                    let l1 = _mm_loadu_si128(lo0_ptr.add(i + 1) as *const __m128i);
                    let l2 = _mm_loadu_si128(lo0_ptr.add(i + 2) as *const __m128i);
                    let l3 = _mm_loadu_si128(lo0_ptr.add(i + 3) as *const __m128i);
                    let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        l0, l1, ve0_r0, ve0_r1, ve0_r0, ve0_r1,
                    );
                    let (p2, p3) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        l2, l3, ve0_r0, ve0_r1, ve0_r0, ve0_r1,
                    );
                    *tile_ptr.add(ti) = fold_one_slot_m128i(p0, t0_ptr);
                    *tile_ptr.add(ti + 1) = fold_one_slot_m128i(p1, t0_ptr);
                    *tile_ptr.add(ti + 2) = fold_one_slot_m128i(p2, t0_ptr);
                    *tile_ptr.add(ti + 3) = fold_one_slot_m128i(p3, t0_ptr);
                }

                for q in q0..q1 {
                    let i = 4 * q;
                    let ti = 4 * (q - q0);
                    let l0 = _mm_loadu_si128(lo1_ptr.add(i) as *const __m128i);
                    let l1 = _mm_loadu_si128(lo1_ptr.add(i + 1) as *const __m128i);
                    let l2 = _mm_loadu_si128(lo1_ptr.add(i + 2) as *const __m128i);
                    let l3 = _mm_loadu_si128(lo1_ptr.add(i + 3) as *const __m128i);
                    let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        l0, l1, ve1_r0, ve1_r1, ve1_r0, ve1_r1,
                    );
                    let (p2, p3) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        l2, l3, ve1_r0, ve1_r1, ve1_r0, ve1_r1,
                    );
                    let s0 = _mm_xor_si128(*tile_ptr.add(ti), fold_one_slot_m128i(p0, t1_ptr));
                    let s1 = _mm_xor_si128(*tile_ptr.add(ti + 1), fold_one_slot_m128i(p1, t1_ptr));
                    let s2 = _mm_xor_si128(*tile_ptr.add(ti + 2), fold_one_slot_m128i(p2, t1_ptr));
                    let s3 = _mm_xor_si128(*tile_ptr.add(ti + 3), fold_one_slot_m128i(p3, t1_ptr));
                    _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, s0);
                    _mm_storeu_si128(out_ptr.add(i + 1) as *mut __m128i, s1);
                    _mm_storeu_si128(out_ptr.add(i + 2) as *mut __m128i, s2);
                    _mm_storeu_si128(out_ptr.add(i + 3) as *mut __m128i, s3);

                    let a0 = _mm_loadu_si128(wit_ptr.add(i) as *const __m128i);
                    let a1 = _mm_loadu_si128(wit_ptr.add(i + 1) as *const __m128i);
                    let a2 = _mm_loadu_si128(wit_ptr.add(i + 2) as *const __m128i);
                    let a3 = _mm_loadu_si128(wit_ptr.add(i + 3) as *const __m128i);
                    crate::gf2_kernels::k0_f128_mul_acc2_m128i(
                        a0,
                        s0,
                        a2,
                        s2,
                        &mut u0_ll,
                        &mut u0_mid,
                        &mut u0_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc2_m128i(
                        _mm_xor_si128(a0, a1),
                        _mm_xor_si128(s0, s1),
                        _mm_xor_si128(a2, a3),
                        _mm_xor_si128(s2, s3),
                        &mut u2_ll,
                        &mut u2_mid,
                        &mut u2_hh,
                    );
                }
                q0 = q1;
            }
        }
        let r0 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u0_ll, u0_mid, u0_hh);
        let r2 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(u2_ll, u2_mid, u2_hh);
        return (
            core::mem::transmute::<__m128i, F128>(r0),
            core::mem::transmute::<__m128i, F128>(r2),
        );
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let out_ptr = out_block.as_mut_ptr() as *mut u64;
        let wit_ptr = witness_block.as_ptr() as *const u64;
        let lo0_files_ptr = eq_lo0.as_ptr();
        let lo1_files_ptr = eq_lo1.as_ptr();
        let lo0_ptr = eq_lo0.as_ptr() as *const u64;
        let lo1_ptr = eq_lo1.as_ptr() as *const u64;
        let t0_ptr = table0.as_ptr();
        let t1_ptr = table1.as_ptr();

        let mut u0_ll = z;
        let mut u0_mid = z;
        let mut u0_hh = z;
        let mut u2_ll = z;
        let mut u2_mid = z;
        let mut u2_hh = z;

        const TILE_Q: usize = 64;
        #[repr(align(64))]
        struct AlignedScratch {
            t0: [F128; FOLD_N_BYTES * FOLD_TABLE_SIZE],
            t1: [F128; FOLD_N_BYTES * FOLD_TABLE_SIZE],
            tile: [core::arch::aarch64::uint64x2_t; TILE_Q * 4],
        }
        std::thread_local! {
            static SCRATCH: core::cell::UnsafeCell<AlignedScratch> = const {
                core::cell::UnsafeCell::new(unsafe { core::mem::zeroed() })
            };
        }
        let scratch_ptr = SCRATCH.with(|c| c.get());
        let tile_buf = &mut (*scratch_ptr).tile;
        let n_q = b / 4;

        if b >= 256 {
            let st0_ptr = (*scratch_ptr).t0.as_mut_ptr();
            let st1_ptr = (*scratch_ptr).t1.as_mut_ptr();
            build_scaled_fold_byte_table_neon(e_hi0, t0_ptr, st0_ptr);
            build_scaled_fold_byte_table_neon(e_hi1, t1_ptr, st1_ptr);
            let st0_lo_ptr = st0_ptr as *const F128;
            let st0_hi_ptr = (st0_ptr as *const F128).add(8 * FOLD_TABLE_SIZE);
            let st1_lo_ptr = st1_ptr as *const F128;
            let st1_hi_ptr = (st1_ptr as *const F128).add(8 * FOLD_TABLE_SIZE);

            let mut q0 = 0usize;
            while q0 < n_q {
                let q1 = (q0 + TILE_Q).min(n_q);
                for q in q0..q1 {
                    let i = 4 * q;
                    let ti = 4 * (q - q0);
                    let e0_lo = (*lo0_files_ptr.add(i)).lo;
                    let e1_lo = (*lo0_files_ptr.add(i + 1)).lo;
                    let e2_lo = (*lo0_files_ptr.add(i + 2)).lo;
                    let e3_lo = (*lo0_files_ptr.add(i + 3)).lo;
                    *tile_buf.get_unchecked_mut(ti) = fold_half_slot_u64_neon(e0_lo, st0_lo_ptr);
                    *tile_buf.get_unchecked_mut(ti + 1) =
                        fold_half_slot_u64_neon(e1_lo, st0_lo_ptr);
                    *tile_buf.get_unchecked_mut(ti + 2) =
                        fold_half_slot_u64_neon(e2_lo, st0_lo_ptr);
                    *tile_buf.get_unchecked_mut(ti + 3) =
                        fold_half_slot_u64_neon(e3_lo, st0_lo_ptr);
                }
                for q in q0..q1 {
                    let i = 4 * q;
                    let ti = 4 * (q - q0);
                    let e0_hi = (*lo0_files_ptr.add(i)).hi;
                    let e1_hi = (*lo0_files_ptr.add(i + 1)).hi;
                    let e2_hi = (*lo0_files_ptr.add(i + 2)).hi;
                    let e3_hi = (*lo0_files_ptr.add(i + 3)).hi;
                    *tile_buf.get_unchecked_mut(ti) = veorq_u64(
                        *tile_buf.get_unchecked(ti),
                        fold_half_slot_u64_neon(e0_hi, st0_hi_ptr),
                    );
                    *tile_buf.get_unchecked_mut(ti + 1) = veorq_u64(
                        *tile_buf.get_unchecked(ti + 1),
                        fold_half_slot_u64_neon(e1_hi, st0_hi_ptr),
                    );
                    *tile_buf.get_unchecked_mut(ti + 2) = veorq_u64(
                        *tile_buf.get_unchecked(ti + 2),
                        fold_half_slot_u64_neon(e2_hi, st0_hi_ptr),
                    );
                    *tile_buf.get_unchecked_mut(ti + 3) = veorq_u64(
                        *tile_buf.get_unchecked(ti + 3),
                        fold_half_slot_u64_neon(e3_hi, st0_hi_ptr),
                    );
                }
                for q in q0..q1 {
                    let i = 4 * q;
                    let ti = 4 * (q - q0);
                    let e0_lo = (*lo1_files_ptr.add(i)).lo;
                    let e1_lo = (*lo1_files_ptr.add(i + 1)).lo;
                    let e2_lo = (*lo1_files_ptr.add(i + 2)).lo;
                    let e3_lo = (*lo1_files_ptr.add(i + 3)).lo;
                    *tile_buf.get_unchecked_mut(ti) = veorq_u64(
                        *tile_buf.get_unchecked(ti),
                        fold_half_slot_u64_neon(e0_lo, st1_lo_ptr),
                    );
                    *tile_buf.get_unchecked_mut(ti + 1) = veorq_u64(
                        *tile_buf.get_unchecked(ti + 1),
                        fold_half_slot_u64_neon(e1_lo, st1_lo_ptr),
                    );
                    *tile_buf.get_unchecked_mut(ti + 2) = veorq_u64(
                        *tile_buf.get_unchecked(ti + 2),
                        fold_half_slot_u64_neon(e2_lo, st1_lo_ptr),
                    );
                    *tile_buf.get_unchecked_mut(ti + 3) = veorq_u64(
                        *tile_buf.get_unchecked(ti + 3),
                        fold_half_slot_u64_neon(e3_lo, st1_lo_ptr),
                    );
                }

                for q in q0..q1 {
                    let i = 4 * q;
                    let idx = 8 * q;
                    let ti = 4 * (q - q0);
                    let e0_hi = (*lo1_files_ptr.add(i)).hi;
                    let e1_hi = (*lo1_files_ptr.add(i + 1)).hi;
                    let e2_hi = (*lo1_files_ptr.add(i + 2)).hi;
                    let e3_hi = (*lo1_files_ptr.add(i + 3)).hi;
                    let s0 = veorq_u64(
                        *tile_buf.get_unchecked(ti),
                        fold_half_slot_u64_neon(e0_hi, st1_hi_ptr),
                    );
                    let s1 = veorq_u64(
                        *tile_buf.get_unchecked(ti + 1),
                        fold_half_slot_u64_neon(e1_hi, st1_hi_ptr),
                    );
                    let s2 = veorq_u64(
                        *tile_buf.get_unchecked(ti + 2),
                        fold_half_slot_u64_neon(e2_hi, st1_hi_ptr),
                    );
                    let s3 = veorq_u64(
                        *tile_buf.get_unchecked(ti + 3),
                        fold_half_slot_u64_neon(e3_hi, st1_hi_ptr),
                    );
                    vst1q_u64(out_ptr.add(idx), s0);
                    vst1q_u64(out_ptr.add(idx + 2), s1);
                    vst1q_u64(out_ptr.add(idx + 4), s2);
                    vst1q_u64(out_ptr.add(idx + 6), s3);

                    let a0 = vld1q_u64(wit_ptr.add(idx));
                    let a1 = vld1q_u64(wit_ptr.add(idx + 2));
                    let a2 = vld1q_u64(wit_ptr.add(idx + 4));
                    let a3 = vld1q_u64(wit_ptr.add(idx + 6));
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        a0,
                        s0,
                        &mut u0_ll,
                        &mut u0_mid,
                        &mut u0_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        a2,
                        s2,
                        &mut u0_ll,
                        &mut u0_mid,
                        &mut u0_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        veorq_u64(a0, a1),
                        veorq_u64(s0, s1),
                        &mut u2_ll,
                        &mut u2_mid,
                        &mut u2_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        veorq_u64(a2, a3),
                        veorq_u64(s2, s3),
                        &mut u2_ll,
                        &mut u2_mid,
                        &mut u2_hh,
                    );
                }
                q0 = q1;
            }
        } else {
            let e0_arr = [e_hi0.lo, e_hi0.hi];
            let e1_arr = [e_hi1.lo, e_hi1.hi];
            let ve0 = vld1q_u64(e0_arr.as_ptr());
            let ve1 = vld1q_u64(e1_arr.as_ptr());
            let (ve0_r0, ve0_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(ve0);
            let (ve1_r0, ve1_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(ve1);
            let mut q0 = 0usize;
            while q0 < n_q {
                let q1 = (q0 + TILE_Q).min(n_q);
                for q in q0..q1 {
                    let idx = 8 * q;
                    let ti = 4 * (q - q0);
                    let l00 = vld1q_u64(lo0_ptr.add(idx));
                    let l01 = vld1q_u64(lo0_ptr.add(idx + 2));
                    let l02 = vld1q_u64(lo0_ptr.add(idx + 4));
                    let l03 = vld1q_u64(lo0_ptr.add(idx + 6));
                    let (p00, p01) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        l00, l01, ve0_r0, ve0_r1, ve0_r0, ve0_r1,
                    );
                    let (p02, p03) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        l02, l03, ve0_r0, ve0_r1, ve0_r0, ve0_r1,
                    );
                    *tile_buf.get_unchecked_mut(ti) = fold_one_slot_neon(p00, t0_ptr);
                    *tile_buf.get_unchecked_mut(ti + 1) = fold_one_slot_neon(p01, t0_ptr);
                    *tile_buf.get_unchecked_mut(ti + 2) = fold_one_slot_neon(p02, t0_ptr);
                    *tile_buf.get_unchecked_mut(ti + 3) = fold_one_slot_neon(p03, t0_ptr);
                }

                for q in q0..q1 {
                    let idx = 8 * q;
                    let ti = 4 * (q - q0);
                    let l10 = vld1q_u64(lo1_ptr.add(idx));
                    let l11 = vld1q_u64(lo1_ptr.add(idx + 2));
                    let l12 = vld1q_u64(lo1_ptr.add(idx + 4));
                    let l13 = vld1q_u64(lo1_ptr.add(idx + 6));
                    let (p10, p11) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        l10, l11, ve1_r0, ve1_r1, ve1_r0, ve1_r1,
                    );
                    let (p12, p13) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        l12, l13, ve1_r0, ve1_r1, ve1_r0, ve1_r1,
                    );
                    let s0 =
                        veorq_u64(*tile_buf.get_unchecked(ti), fold_one_slot_neon(p10, t1_ptr));
                    let s1 = veorq_u64(
                        *tile_buf.get_unchecked(ti + 1),
                        fold_one_slot_neon(p11, t1_ptr),
                    );
                    let s2 = veorq_u64(
                        *tile_buf.get_unchecked(ti + 2),
                        fold_one_slot_neon(p12, t1_ptr),
                    );
                    let s3 = veorq_u64(
                        *tile_buf.get_unchecked(ti + 3),
                        fold_one_slot_neon(p13, t1_ptr),
                    );
                    vst1q_u64(out_ptr.add(idx), s0);
                    vst1q_u64(out_ptr.add(idx + 2), s1);
                    vst1q_u64(out_ptr.add(idx + 4), s2);
                    vst1q_u64(out_ptr.add(idx + 6), s3);

                    let a0 = vld1q_u64(wit_ptr.add(idx));
                    let a1 = vld1q_u64(wit_ptr.add(idx + 2));
                    let a2 = vld1q_u64(wit_ptr.add(idx + 4));
                    let a3 = vld1q_u64(wit_ptr.add(idx + 6));
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        a0,
                        s0,
                        &mut u0_ll,
                        &mut u0_mid,
                        &mut u0_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        a2,
                        s2,
                        &mut u0_ll,
                        &mut u0_mid,
                        &mut u0_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        veorq_u64(a0, a1),
                        veorq_u64(s0, s1),
                        &mut u2_ll,
                        &mut u2_mid,
                        &mut u2_hh,
                    );
                    crate::gf2_kernels::k0_f128_mul_acc_neon(
                        veorq_u64(a2, a3),
                        veorq_u64(s2, s3),
                        &mut u2_ll,
                        &mut u2_mid,
                        &mut u2_hh,
                    );
                }
                q0 = q1;
            }
        }
        let r0 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u0_ll, u0_mid, u0_hh);
        let r2 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(u2_ll, u2_mid, u2_hh);
        return (
            core::mem::transmute::<uint64x2_t, F128>(r0),
            core::mem::transmute::<uint64x2_t, F128>(r2),
        );
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    {
        let e0_x4 = [e_hi0; 4];
        let e1_x4 = [e_hi1; 4];
        for (slots, lo0s) in out_block.chunks_exact_mut(4).zip(eq_lo0.chunks_exact(4)) {
            let p = crate::field::f128_slice::mul_4([lo0s[0], lo0s[1], lo0s[2], lo0s[3]], e0_x4);
            slots[0] = fold_one_slot(p[0], table0);
            slots[1] = fold_one_slot(p[1], table0);
            slots[2] = fold_one_slot(p[2], table0);
            slots[3] = fold_one_slot(p[3], table0);
        }
        let mut u0_unred = crate::field::F256Unreduced::ZERO;
        let mut u2_unred = crate::field::F256Unreduced::ZERO;
        for q in 0..(b / 4) {
            let i = 4 * q;
            let p = crate::field::f128_slice::mul_4(
                [eq_lo1[i], eq_lo1[i + 1], eq_lo1[i + 2], eq_lo1[i + 3]],
                e1_x4,
            );
            let s0 = out_block[i] + fold_one_slot(p[0], table1);
            let s1 = out_block[i + 1] + fold_one_slot(p[1], table1);
            let s2 = out_block[i + 2] + fold_one_slot(p[2], table1);
            let s3 = out_block[i + 3] + fold_one_slot(p[3], table1);
            out_block[i] = s0;
            out_block[i + 1] = s1;
            out_block[i + 2] = s2;
            out_block[i + 3] = s3;
            let a0 = witness_block[i];
            let a1 = witness_block[i + 1];
            let a2 = witness_block[i + 2];
            let a3 = witness_block[i + 3];
            u0_unred ^= a0.mul_unreduced(s0) ^ a2.mul_unreduced(s2);
            u2_unred ^= (a0 + a1).mul_unreduced(s0 + s1) ^ (a2 + a3).mul_unreduced(s2 + s3);
        }
        (u0_unred.reduce(), u2_unred.reduce())
    }
}

#[inline(always)]
pub(crate) fn deferred_dense_value(
    eq_lo: &[F128],
    eq_hi: &[F128],
    table: &[F128],
    log_b: usize,
    j: usize,
) -> F128 {
    let mask = (1usize << log_b) - 1;
    fold_one_slot(eq_lo[j & mask] * eq_hi[j >> log_b], table)
}

pub fn fold_b128_elems_split(eq_lo: &[F128], eq_hi: &[F128], eq_r_dprime: &[F128]) -> Vec<F128> {
    let tables = build_fold_byte_table(eq_r_dprime);
    fold_b128_from_table(eq_lo, eq_hi, &tables)
}

pub(crate) fn fold_b128_from_table(eq_lo: &[F128], eq_hi: &[F128], tables: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    let b = eq_lo.len();
    debug_assert_eq!(tables.len(), FOLD_N_BYTES * FOLD_TABLE_SIZE);
    let mut out = crate::scratch::take_f128(b * eq_hi.len());
    out.par_chunks_mut(b)
        .zip(eq_hi.par_iter())
        .for_each(|(out_block, &e_hi)| {
            #[cfg(target_arch = "x86_64")]
            if b >= 256 {
                unsafe {
                    use core::arch::x86_64::*;
                    let mut scaled_t =
                        core::mem::MaybeUninit::<[F128; FOLD_N_BYTES * FOLD_TABLE_SIZE]>::uninit();
                    let st_ptr = scaled_t.as_mut_ptr() as *mut F128;
                    build_scaled_fold_byte_table_m128i(e_hi, tables.as_ptr(), st_ptr);
                    let st_ptr = st_ptr as *const F128;
                    let out_ptr = out_block.as_mut_ptr() as *mut __m128i;
                    let lo_ptr = eq_lo.as_ptr();
                    for i_lo in 0..b {
                        let e = *lo_ptr.add(i_lo);
                        _mm_storeu_si128(
                            out_ptr.add(i_lo),
                            fold_one_slot_u64_pair_m128i(e.lo, e.hi, st_ptr),
                        );
                    }
                    return;
                }
            }
            #[cfg(target_arch = "aarch64")]
            if b >= 256 {
                unsafe {
                    use core::arch::aarch64::*;
                    let mut scaled_t =
                        core::mem::MaybeUninit::<[F128; FOLD_N_BYTES * FOLD_TABLE_SIZE]>::uninit();
                    let st_ptr = scaled_t.as_mut_ptr() as *mut F128;
                    build_scaled_fold_byte_table_neon(e_hi, tables.as_ptr(), st_ptr);
                    let st_ptr = st_ptr as *const F128;
                    let out_ptr = out_block.as_mut_ptr() as *mut u64;
                    let lo_ptr = eq_lo.as_ptr();
                    for i_lo in 0..b {
                        let e = *lo_ptr.add(i_lo);
                        vst1q_u64(
                            out_ptr.add(2 * i_lo),
                            fold_one_slot_u64_pair_neon(e.lo, e.hi, st_ptr),
                        );
                    }
                    return;
                }
            }
            for (i_lo, slot) in out_block.iter_mut().enumerate() {
                *slot = fold_one_slot(eq_lo[i_lo] * e_hi, tables);
            }
        });
    out
}

const SPARSE_ZERO_THRESHOLD: usize = 3;

#[derive(Clone, Debug)]
pub struct SparseEqTensor {
    pub live_tensor: Vec<F128>,
    pub live_positions: Vec<usize>,
    pub live_mask: u64,
}

impl SparseEqTensor {
    #[inline(always)]
    pub fn scatter_idx(&self, c: usize) -> usize {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        unsafe {
            return core::arch::x86_64::_pdep_u64(c as u64, self.live_mask) as usize;
        }
        #[cfg(all(
            any(target_arch = "x86_64", target_arch = "aarch64"),
            not(all(target_arch = "x86_64", target_feature = "bmi2"))
        ))]
        unsafe {
            let mut full = 0u64;
            let mut m_rem = self.live_mask;
            let mut c_rem = c as u64;
            while c_rem != 0 && m_rem != 0 {
                full |= crate::gf2_kernels::gf2_kern_bit_deposit_lowest_one(c_rem, m_rem);
                m_rem = crate::gf2_kernels::gf2_kern_clear_lowest_one(m_rem);
                c_rem >>= 1;
            }
            return full as usize;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let mut full = 0usize;
            for (j, &pos) in self.live_positions.iter().enumerate() {
                full |= ((c >> j) & 1) << pos;
            }
            full
        }
    }

    pub fn materialize(&self) -> Vec<(usize, F128)> {
        self.live_tensor
            .iter()
            .enumerate()
            .map(|(c, &v)| (self.scatter_idx(c), v))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.live_tensor.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live_tensor.is_empty()
    }
}

pub fn build_eq_sparse(coords: &[F128]) -> SparseEqTensor {
    let live_positions: Vec<usize> = coords
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if c == F128::ZERO { None } else { Some(i) })
        .collect();
    let live_coords: Vec<F128> = live_positions.iter().map(|&i| coords[i]).collect();
    let live_tensor = build_eq(&live_coords);
    let mut live_mask = 0u64;
    for &pos in &live_positions {
        debug_assert!(pos < 64);
        live_mask |= 1u64 << pos;
    }
    SparseEqTensor {
        live_tensor,
        live_positions,
        live_mask,
    }
}

pub fn fold_1b_rows_sparse(packed_witness: &[F128], eq: &SparseEqTensor) -> Vec<F128> {
    fold_1b_rows_sparse_scalar(packed_witness, eq)
}

fn fold_1b_rows_sparse_scalar(packed_witness: &[F128], eq: &SparseEqTensor) -> Vec<F128> {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    let zero_acc = || vec![F128::ZERO; n];

    eq.live_tensor
        .par_iter()
        .enumerate()
        .fold(zero_acc, |mut acc, (c, &val)| {
            let idx = eq.scatter_idx(c);
            let elem = packed_witness[idx];
            let mut lo = elem.lo;
            while lo != 0 {
                let r = lo.trailing_zeros() as usize;
                acc[r] += val;
                lo &= lo - 1;
            }
            let mut hi = elem.hi;
            while hi != 0 {
                let r = hi.trailing_zeros() as usize;
                acc[64 | r] += val;
                hi &= hi - 1;
            }
            acc
        })
        .reduce(zero_acc, |mut a, b| {
            for r in 0..n {
                a[r] += b[r];
            }
            a
        })
}

#[allow(dead_code)]
fn detect_block_stride(support: &[(usize, F128)]) -> Option<(usize, usize)> {
    if support.len() < 8 {
        return None;
    }
    if support[0].0 != 0 {
        return None;
    }
    let mut block_size = 1usize;
    while block_size < support.len() && support[block_size].0 == block_size {
        block_size += 1;
    }
    if block_size >= support.len() || !support.len().is_multiple_of(block_size) {
        return None;
    }
    let stride = support[block_size].0;
    if stride < block_size || !stride.is_power_of_two() {
        return None;
    }
    let num_groups = support.len() / block_size;
    for g in 0..num_groups {
        let base = g * stride;
        for k in 0..block_size {
            if support[g * block_size + k].0 != base + k {
                return None;
            }
        }
    }
    Some((block_size, stride))
}

#[allow(dead_code)]
fn fold_1b_rows_sparse_mfr_block4(
    packed_witness: &[F128],
    support: &[(usize, F128)],
    stride: usize,
) -> Vec<F128> {
    use rayon::prelude::*;
    let n = 1 << LOG_PACKING;
    debug_assert!(support.len().is_multiple_of(4));
    let num_groups = support.len() / 4;

    (0..num_groups)
        .into_par_iter()
        .fold(
            || vec![F128::ZERO; n],
            |mut acc, g| {
                let base = g * stride;
                let m0 = packed_witness[base];
                let m1 = packed_witness[base + 1];
                let m2 = packed_witness[base + 2];
                let m3 = packed_witness[base + 3];
                let v: [F128; 4] = [
                    support[g * 4].1,
                    support[g * 4 + 1].1,
                    support[g * 4 + 2].1,
                    support[g * 4 + 3].1,
                ];
                let lookup = subset_sums_4(v);

                let m_bytes: [[u8; 16]; 4] = [
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m0.lo.to_le_bytes());
                        b[8..].copy_from_slice(&m0.hi.to_le_bytes());
                        b
                    },
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m1.lo.to_le_bytes());
                        b[8..].copy_from_slice(&m1.hi.to_le_bytes());
                        b
                    },
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m2.lo.to_le_bytes());
                        b[8..].copy_from_slice(&m2.hi.to_le_bytes());
                        b
                    },
                    {
                        let mut b = [0u8; 16];
                        b[..8].copy_from_slice(&m3.lo.to_le_bytes());
                        b[8..].copy_from_slice(&m3.hi.to_le_bytes());
                        b
                    },
                ];

                for r_byte in 0..16 {
                    let combined: u64 = (m_bytes[0][r_byte] as u64)
                        | ((m_bytes[1][r_byte] as u64) << 8)
                        | ((m_bytes[2][r_byte] as u64) << 16)
                        | ((m_bytes[3][r_byte] as u64) << 24);
                    let transposed = transpose_8x8_bits(combined);
                    let tb = transposed.to_le_bytes();
                    let b = r_byte * 8;
                    acc[b] += lookup[(tb[0] & 0x0F) as usize];
                    acc[b + 1] += lookup[(tb[1] & 0x0F) as usize];
                    acc[b + 2] += lookup[(tb[2] & 0x0F) as usize];
                    acc[b + 3] += lookup[(tb[3] & 0x0F) as usize];
                    acc[b + 4] += lookup[(tb[4] & 0x0F) as usize];
                    acc[b + 5] += lookup[(tb[5] & 0x0F) as usize];
                    acc[b + 6] += lookup[(tb[6] & 0x0F) as usize];
                    acc[b + 7] += lookup[(tb[7] & 0x0F) as usize];
                }
                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; n],
            |mut a, b| {
                for r in 0..n {
                    a[r] += b[r];
                }
                a
            },
        )
}

pub fn fold_b128_elems_sparse_pairs(
    eq: &SparseEqTensor,
    eq_r_dprime: &[F128],
) -> Vec<(usize, F128)> {
    use rayon::prelude::*;
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    let tables = build_fold_byte_table(eq_r_dprime);
    eq.live_tensor
        .par_iter()
        .enumerate()
        .map(|(c, &tensor_val)| (eq.scatter_idx(c), fold_one_slot(tensor_val, &tables)))
        .collect()
}

pub fn fold_b128_elems_sparse(len: usize, eq: &SparseEqTensor, eq_r_dprime: &[F128]) -> Vec<F128> {
    let pairs = fold_b128_elems_sparse_pairs(eq, eq_r_dprime);
    let mut out = vec![F128::ZERO; len];
    for (idx, val) in pairs {
        out[idx] = val;
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingSwitchProof {
    pub s_hat_v: Vec<F128>,
}

#[derive(Clone, Debug)]
pub struct RingSwitchOutput {
    pub rs_eq_ind: Vec<F128>,
    pub sumcheck_claim: F128,
}

#[derive(Clone, Debug)]
pub struct RingSwitchBatchOutput {
    pub rs_eq_ind: RsEqInd,
    pub sumcheck_claim: F128,
    pub(crate) direct_fold8: Option<DirectFold8Factors>,
}

#[derive(Clone, Debug)]
pub(crate) struct DirectFold8Factors {
    pub eq_lo: Vec<F128>,
    pub eq_hi: Vec<F128>,
    pub a_state: Vec<F128>,
    pub w_state: Vec<F128>,
}

#[derive(Clone, Debug)]
pub enum RsEqInd {
    Dense(Vec<F128>),
    DeferredDense {
        logical_len: usize,
        eq_lo: Vec<F128>,
        eq_hi: Vec<F128>,
        table: Vec<F128>,
    },
    Sparse {
        len: usize,
        entries: Vec<(usize, F128)>,
    },
}

impl RsEqInd {
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::DeferredDense { logical_len, .. } => *logical_len,
            Self::Sparse { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add_scaled_into(&self, gamma: F128, out: &mut [F128]) {
        debug_assert_eq!(out.len(), self.len());
        match self {
            Self::Dense(v) => {
                for (o, &x) in out.iter_mut().zip(v.iter()) {
                    *o += gamma * x;
                }
            }
            Self::DeferredDense {
                logical_len,
                eq_lo,
                eq_hi,
                table,
            } => {
                assert_eq!(
                    eq_lo.len() * eq_hi.len(),
                    *logical_len,
                    "DeferredDense basis materialization was elided"
                );
                let log_b = eq_lo.len().trailing_zeros() as usize;
                for (j, o) in out.iter_mut().enumerate() {
                    *o += gamma * deferred_dense_value(eq_lo, eq_hi, table, log_b, j);
                }
            }
            Self::Sparse { entries, .. } => {
                for &(idx, val) in entries {
                    out[idx] += gamma * val;
                }
            }
        }
    }

    pub fn to_dense(&self) -> Vec<F128> {
        match self {
            Self::Dense(v) => v.clone(),
            Self::DeferredDense {
                logical_len,
                eq_lo,
                eq_hi,
                table,
            } => {
                assert_eq!(
                    eq_lo.len() * eq_hi.len(),
                    *logical_len,
                    "DeferredDense basis materialization was elided"
                );
                let log_b = eq_lo.len().trailing_zeros() as usize;
                let l = *logical_len;
                (0..l)
                    .map(|j| deferred_dense_value(eq_lo, eq_hi, table, log_b, j))
                    .collect()
            }
            Self::Sparse { len, entries } => {
                let mut out = vec![F128::ZERO; *len];
                for &(idx, val) in entries {
                    out[idx] = val;
                }
                out
            }
        }
    }

    pub fn into_dense(self) -> Vec<F128> {
        match self {
            Self::Dense(v) => v,
            Self::DeferredDense { .. } => self.to_dense(),
            Self::Sparse { len, entries } => {
                let mut out = vec![F128::ZERO; len];
                for (idx, val) in entries {
                    out[idx] = val;
                }
                out
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    ClaimMismatch,
}

pub fn prove<Ch: Challenger>(
    packed_witness: &[F128],
    x_outer: &[F128],
    challenger: &mut Ch,
) -> (RingSwitchProof, RingSwitchOutput) {
    assert!(
        !x_outer.is_empty(),
        "x_outer must contain at least 1 coord (the 7th-bit factor)"
    );
    let l = packed_witness.len();
    assert_eq!(l, 1 << (x_outer.len() - 1).saturating_add(0));
    assert_eq!(l, 1 << (x_outer.len() - 1));

    let trace = std::env::var("PCS_TRACE").is_ok();

    challenger.observe_label(b"flock-ring-switch-v0");

    let suffix = &x_outer[1..];
    let t = std::time::Instant::now();
    let suffix_tensor = build_eq_parallel(suffix);
    if trace {
        eprintln!(
            "    [rs::prove] build_eq(suffix L={}): {:6.2} ms",
            suffix.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    debug_assert_eq!(suffix_tensor.len(), l);

    let t = std::time::Instant::now();
    let s_hat_v = fold_1b_rows_naive(packed_witness, &suffix_tensor);
    if trace {
        eprintln!(
            "    [rs::prove] fold_1b_rows:          {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    challenger.observe_f128_slice(&s_hat_v);

    let r_dprime = challenger.sample_f128_vec(LOG_PACKING);
    let eq_r_dprime = build_eq(&r_dprime);

    let s_hat_u = tensor_algebra_transpose(&s_hat_v);
    let sumcheck_claim = inner_product(&s_hat_u, &eq_r_dprime);

    let t = std::time::Instant::now();
    let rs_eq_ind = fold_b128_elems(&suffix_tensor, &eq_r_dprime);
    if trace {
        eprintln!(
            "    [rs::prove] fold_b128_elems:       {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (
        RingSwitchProof { s_hat_v },
        RingSwitchOutput {
            rs_eq_ind,
            sumcheck_claim,
        },
    )
}

pub fn prove_batched<Ch: Challenger>(
    packed_witness: &[F128],
    x_outers: &[&[F128]],
    challenger: &mut Ch,
) -> (Vec<(RingSwitchProof, RingSwitchBatchOutput)>, Vec<F128>) {
    let m = LOG_PACKING + (packed_witness.len().trailing_zeros() as usize);
    prove_batched_padded(packed_witness, x_outers, &PaddingSpec::dense(m), challenger)
}

pub fn prove_batched_padded<Ch: Challenger>(
    packed_witness: &[F128],
    x_outers: &[&[F128]],
    padding: &PaddingSpec,
    challenger: &mut Ch,
) -> (Vec<(RingSwitchProof, RingSwitchBatchOutput)>, Vec<F128>) {
    prove_batched_padded_with_precomputed(packed_witness, x_outers, &[], padding, challenger)
}

pub fn prove_batched_padded_with_precomputed<Ch: Challenger>(
    packed_witness: &[F128],
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    padding: &PaddingSpec,
    challenger: &mut Ch,
) -> (Vec<(RingSwitchProof, RingSwitchBatchOutput)>, Vec<F128>) {
    prove_batched_padded_with_precomputed_elidable(
        packed_witness,
        x_outers,
        precomputed_s_hat_v,
        padding,
        false,
        challenger,
    )
}

pub(crate) fn prove_batched_padded_with_precomputed_elidable<Ch: Challenger>(
    packed_witness: &[F128],
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    padding: &PaddingSpec,
    elide_deferred_dense_basis: bool,
    challenger: &mut Ch,
) -> (Vec<(RingSwitchProof, RingSwitchBatchOutput)>, Vec<F128>) {
    assert!(!x_outers.is_empty());
    let trace = std::env::var("PCS_TRACE").is_ok();
    let n = x_outers.len();
    let l = packed_witness.len();
    for x in x_outers {
        assert!(!x.is_empty());
        assert_eq!(l, 1 << (x.len() - 1));
    }
    assert!(
        precomputed_s_hat_v.is_empty() || precomputed_s_hat_v.len() == n,
        "precomputed_s_hat_v: must be empty or length {n}, got {}",
        precomputed_s_hat_v.len(),
    );
    let n_packed = 1usize << LOG_PACKING;
    for p in precomputed_s_hat_v.iter().flatten() {
        assert!(
            p.len() == n_packed || p.len() == 64 * n_packed,
            "precomputed_s_hat_v entry must have length 128 or 64*128, got {}",
            p.len()
        );
    }

    let has_precomputed =
        |orig: usize| -> bool { precomputed_s_hat_v.get(orig).copied().flatten().is_some() };

    #[derive(Clone, Copy)]
    enum Kind {
        Dense(usize),
        Sparse(usize),
    }
    let mut kinds: Vec<Kind> = Vec::with_capacity(n);
    let mut dense_suffixes: Vec<&[F128]> = Vec::new();
    let mut sparse_suffixes: Vec<&[F128]> = Vec::new();
    let mut dense_to_orig: Vec<usize> = Vec::new();
    let mut sparse_to_orig: Vec<usize> = Vec::new();
    for (orig, x) in x_outers.iter().enumerate() {
        let suffix = &x[1..];
        let n_zeros = suffix.iter().filter(|&&c| c == F128::ZERO).count();
        if n_zeros >= SPARSE_ZERO_THRESHOLD {
            kinds.push(Kind::Sparse(sparse_suffixes.len()));
            sparse_to_orig.push(orig);
            sparse_suffixes.push(suffix);
        } else {
            kinds.push(Kind::Dense(dense_suffixes.len()));
            dense_to_orig.push(orig);
            dense_suffixes.push(suffix);
        }
    }

    let dense_needs_fold: Vec<usize> = (0..dense_suffixes.len())
        .filter(|&d| !has_precomputed(dense_to_orig[d]))
        .collect();
    let sparse_needs_fold: Vec<usize> = (0..sparse_suffixes.len())
        .filter(|&s| !has_precomputed(sparse_to_orig[s]))
        .collect();

    let use_split = l.is_multiple_of(16);
    let elide_dense_basis =
        elide_deferred_dense_basis && use_split && sparse_suffixes.is_empty() && l >= 64;
    let t = std::time::Instant::now();
    let dense_splits: Vec<(Vec<F128>, Vec<F128>)> = if use_split && !elide_dense_basis {
        dense_suffixes
            .iter()
            .map(|s| {
                let n_lo = split_n_lo(s.len());
                build_eq_split(s, n_lo)
            })
            .collect()
    } else {
        Vec::new()
    };
    let dense_tensors: Vec<Vec<F128>> = if use_split {
        Vec::new()
    } else {
        dense_suffixes
            .iter()
            .map(|s| build_eq_parallel(s))
            .collect()
    };
    let sparse_supports: Vec<SparseEqTensor> =
        sparse_suffixes.iter().map(|s| build_eq_sparse(s)).collect();

    let mut dense_direct_setup: Vec<Option<([F128; 6], Vec<F128>, Vec<F128>, Vec<F128>)>> =
        vec![None; dense_suffixes.len()];
    if use_split && l >= 64 && (elide_dense_basis || !dense_needs_fold.is_empty()) {
        for (d, &suffix) in dense_suffixes.iter().enumerate() {
            if !elide_dense_basis && !dense_needs_fold.contains(&d) {
                continue;
            }
            assert!(suffix.len() >= 6);
            let mut low_coords = [F128::ZERO; 6];
            low_coords.copy_from_slice(&suffix[..6]);
            let tail = &suffix[6..];
            let n_lo = split_n_lo(tail.len());
            let (eq_lo, eq_hi) = build_eq_split(tail, n_lo);
            let pre = precomputed_s_hat_v.get(dense_to_orig[d]).copied().flatten();
            let a_state = match pre {
                Some(p) if p.len() == 64 * n_packed => {
                    let mut a_state = vec![F128::ZERO; 64 * 128];
                    for bank in 0..64 {
                        let bank_v = &p[bank * 128..(bank + 1) * 128];
                        let bank_u = tensor_algebra_transpose(bank_v);
                        a_state[bank * 128..(bank + 1) * 128].copy_from_slice(&bank_u);
                    }
                    a_state
                }
                _ => direct_fold8_states_par(packed_witness, &eq_lo, &eq_hi, padding),
            };
            dense_direct_setup[d] = Some((low_coords, eq_lo, eq_hi, a_state));
        }
    }
    if trace {
        eprintln!(
            "    [rs::prove_batched] build_eq/direct_setup dense×{} ({}) + sparse×{}: {:6.2} ms",
            dense_suffixes.len(),
            if elide_dense_basis {
                "direct-fold8"
            } else if use_split {
                "split"
            } else {
                "full"
            },
            sparse_supports.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let t = std::time::Instant::now();
    let mut dense_s_hat_v: Vec<Vec<F128>> = vec![Vec::new(); dense_suffixes.len()];
    let mut sparse_s_hat_v: Vec<Vec<F128>> = vec![Vec::new(); sparse_suffixes.len()];
    for d in 0..dense_suffixes.len() {
        if let Some(p) = precomputed_s_hat_v.get(dense_to_orig[d]).copied().flatten() {
            if p.len() == n_packed {
                dense_s_hat_v[d] = p.to_vec();
            } else {
                dense_s_hat_v[d] = collapse_s_hat_v_fold8(p, &dense_suffixes[d][..6]);
            }
        }
    }
    for s in 0..sparse_suffixes.len() {
        if let Some(p) = precomputed_s_hat_v
            .get(sparse_to_orig[s])
            .copied()
            .flatten()
        {
            assert_eq!(p.len(), n_packed);
            sparse_s_hat_v[s] = p.to_vec();
        }
    }

    if use_split && l >= 64 {
        for &d in &dense_needs_fold {
            let (low_coords, _, _, a_state) = dense_direct_setup[d]
                .as_ref()
                .expect("dense_direct_setup built above when use_split && l >= 64");
            let eq_low = build_eq(low_coords);
            let mut s_hat_v = vec![F128::ZERO; 128];
            for (bank, &w_low) in eq_low.iter().enumerate() {
                let a_bank = &a_state[(bank << 7)..((bank + 1) << 7)];
                let a_t = tensor_algebra_transpose(a_bank);
                for r in 0..128 {
                    s_hat_v[r] += w_low * a_t[r];
                }
            }
            dense_s_hat_v[d] = s_hat_v;
        }
    } else if use_split {
        match dense_needs_fold.len() {
            0 => {}
            2 => {
                let d0 = dense_needs_fold[0];
                let d1 = dense_needs_fold[1];
                let (lo0, hi0) = (dense_splits[d0].0.as_slice(), dense_splits[d0].1.as_slice());
                let (lo1, hi1) = (dense_splits[d1].0.as_slice(), dense_splits[d1].1.as_slice());
                let (a, b) = fold_1b_rows_split_2way(packed_witness, lo0, hi0, lo1, hi1, padding);
                dense_s_hat_v[d0] = a;
                dense_s_hat_v[d1] = b;
            }
            _ => {
                for &d in &dense_needs_fold {
                    let (eq_lo, eq_hi) = (&dense_splits[d].0, &dense_splits[d].1);
                    dense_s_hat_v[d] = fold_1b_rows_split(packed_witness, eq_lo, eq_hi, padding);
                }
            }
        }
    } else if !dense_needs_fold.is_empty() {
        let dense_refs: Vec<&[F128]> = dense_needs_fold
            .iter()
            .map(|&d| dense_tensors[d].as_slice())
            .collect();
        let out = fold_1b_rows_multi_padded(packed_witness, &dense_refs, padding);
        for (i, &d) in dense_needs_fold.iter().enumerate() {
            dense_s_hat_v[d] = out[i].clone();
        }
    }
    for &s in &sparse_needs_fold {
        sparse_s_hat_v[s] = fold_1b_rows_sparse(packed_witness, &sparse_supports[s]);
    }
    if trace {
        eprintln!(
            "    [rs::prove_batched] fold_1b_rows dense(k={})+sparse(k={}): {:6.2} ms",
            dense_s_hat_v.len(),
            sparse_s_hat_v.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let t = std::time::Instant::now();

    struct ClaimWork {
        s_hat_v: Vec<F128>,
        sumcheck_claim: F128,
        eq_r_dprime: Vec<F128>,
    }
    let mut work: Vec<ClaimWork> = Vec::with_capacity(n);
    for i in 0..n {
        challenger.observe_label(b"flock-ring-switch-v0");
        let s_hat_v: Vec<F128> = match kinds[i] {
            Kind::Dense(d) => dense_s_hat_v[d].clone(),
            Kind::Sparse(s) => sparse_s_hat_v[s].clone(),
        };
        challenger.observe_f128_slice(&s_hat_v);
        let r_dprime = challenger.sample_f128_vec(LOG_PACKING);
        let eq_r_dprime = build_eq(&r_dprime);

        let s_hat_u = tensor_algebra_transpose(&s_hat_v);
        let sumcheck_claim = inner_product(&s_hat_u, &eq_r_dprime);

        work.push(ClaimWork {
            s_hat_v,
            sumcheck_claim,
            eq_r_dprime,
        });
    }

    let gammas_rs: Vec<F128> = (0..n).map(|_| challenger.sample_f128()).collect();

    let results: Vec<(RingSwitchProof, RingSwitchBatchOutput)> = work
        .into_iter()
        .zip(gammas_rs.iter())
        .enumerate()
        .map(|(i, (w, &g))| {
            let scaled_eq_r_dprime: Vec<F128> = w.eq_r_dprime.iter().map(|x| g * *x).collect();
            let (rs_eq_ind, direct_fold8) = match kinds[i] {
                Kind::Dense(d) => {
                    if use_split {
                        let rs_eq_ind = if elide_dense_basis {
                            RsEqInd::DeferredDense {
                                logical_len: l,
                                eq_lo: Vec::new(),
                                eq_hi: Vec::new(),
                                table: Vec::new(),
                            }
                        } else {
                            let (eq_lo, eq_hi) = &dense_splits[d];
                            RsEqInd::DeferredDense {
                                logical_len: l,
                                eq_lo: eq_lo.clone(),
                                eq_hi: eq_hi.clone(),
                                table: build_fold_byte_table(&scaled_eq_r_dprime),
                            }
                        };
                        let direct_fold8 = if elide_dense_basis {
                            dense_direct_setup[d].take().map(
                                |(low_coords, eq_lo, eq_hi, a_state)| {
                                    build_direct_fold8_factors_reusing_a_state(
                                        low_coords,
                                        eq_lo,
                                        eq_hi,
                                        a_state,
                                        scaled_eq_r_dprime,
                                    )
                                },
                            )
                        } else {
                            None
                        };
                        (rs_eq_ind, direct_fold8)
                    } else {
                        (
                            RsEqInd::Dense(fold_b128_elems(&dense_tensors[d], &scaled_eq_r_dprime)),
                            None,
                        )
                    }
                }
                Kind::Sparse(s) => (
                    RsEqInd::Sparse {
                        len: l,
                        entries: fold_b128_elems_sparse_pairs(
                            &sparse_supports[s],
                            &scaled_eq_r_dprime,
                        ),
                    },
                    None,
                ),
            };
            (
                RingSwitchProof { s_hat_v: w.s_hat_v },
                RingSwitchBatchOutput {
                    rs_eq_ind,
                    sumcheck_claim: w.sumcheck_claim,
                    direct_fold8,
                },
            )
        })
        .collect();

    if trace {
        eprintln!(
            "    [rs::prove_batched] per-opening tail ×{}: {:6.2} ms",
            n,
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    (results, gammas_rs)
}

fn build_direct_fold8_factors_reusing_a_state(
    low_coords: [F128; 6],
    eq_lo: Vec<F128>,
    eq_hi: Vec<F128>,
    a_state: Vec<F128>,
    scaled_eq_r_dprime: Vec<F128>,
) -> DirectFold8Factors {
    assert_eq!(a_state.len(), 64 * 128);
    let eq_low = build_eq(&low_coords);
    let base_table = build_fold_byte_table(&scaled_eq_r_dprime);
    let mut w_state = vec![F128::ZERO; 64 * 128];
    direct_fold8_w_state_into(&base_table, &eq_low, &mut w_state);
    DirectFold8Factors {
        eq_lo,
        eq_hi,
        a_state,
        w_state,
    }
}

fn direct_fold8_states_par(
    packed_witness: &[F128],
    eq_lo: &[F128],
    eq_hi: &[F128],
    padding: &PaddingSpec,
) -> Vec<F128> {
    use rayon::prelude::*;

    let b = eq_lo.len();
    let h = eq_hi.len();
    assert_eq!(packed_witness.len(), (b * h) << 6);

    const B_SUB: usize = 64;
    if b < B_SUB || !b.is_multiple_of(B_SUB) {
        let skip = ChunkPadding::new(padding, 64);
        let mut a_state = vec![F128::ZERO; 64 * 128];
        for (i_hi, &e_hi) in eq_hi.iter().enumerate() {
            if e_hi == F128::ZERO {
                continue;
            }
            for (i_lo, &e_lo) in eq_lo.iter().enumerate() {
                let q = i_hi * b + i_lo;
                if skip.skip(q) {
                    continue;
                }
                let base = q << 6;
                let e = e_lo * e_hi;
                if e == F128::ZERO {
                    continue;
                }
                let w_block = &packed_witness[base..base + 64];
                let mut lo = e.lo;
                while lo != 0 {
                    let t = lo.trailing_zeros() as usize;
                    for (bank, &w) in w_block.iter().enumerate() {
                        a_state[(bank << 7) | t] += w;
                    }
                    lo &= lo - 1;
                }
                let mut hi = e.hi;
                while hi != 0 {
                    let t = 64 + (hi.trailing_zeros() as usize);
                    for (bank, &w) in w_block.iter().enumerate() {
                        a_state[(bank << 7) | t] += w;
                    }
                    hi &= hi - 1;
                }
            }
        }
        return a_state;
    }

    let sub_per_hi = b / B_SUB;
    let total_subs = h * sub_per_hi;
    let skip = ChunkPadding::new(padding, B_SUB << 6);
    let zero_acc = || vec![F128::ZERO; 64 * 128];

    (0..total_subs)
        .into_par_iter()
        .fold(zero_acc, |mut acc, sub_idx| {
            let i_hi = sub_idx / sub_per_hi;
            let s = sub_idx % sub_per_hi;
            let e_hi = eq_hi[i_hi];
            if e_hi == F128::ZERO {
                return acc;
            }
            if skip.skip(sub_idx) {
                return acc;
            }
            let q_start = i_hi * b + s * B_SUB;
            let w_start = q_start << 6;
            let w_len = B_SUB << 6;
            let eq_lo_sub = &eq_lo[s * B_SUB..(s + 1) * B_SUB];
            let w_sub = &packed_witness[w_start..w_start + w_len];
            direct_fold8_a_state_into(w_sub, eq_lo_sub, e_hi, &mut acc);
            acc
        })
        .reduce(zero_acc, |mut a, b_acc| {
            crate::field::f128_slice::xor_assign(&mut a, &b_acc);
            a
        })
}

fn direct_fold8_a_state_into(w_sub: &[F128], eq_lo_sub: &[F128], e_hi: F128, a_state: &mut [F128]) {
    let n_q = eq_lo_sub.len();
    debug_assert_eq!(w_sub.len(), n_q << 6);
    debug_assert_eq!(a_state.len(), 64 * 128);
    debug_assert_eq!(n_q % 16, 0);

    let mut q_vals = [F128::ZERO; 16];
    let mut g_bytes = [0u8; 16 * 16];
    let mut col_masks = [0u16; 128];
    let mut table = [F128::ZERO; 16];

    for chunk in 0..(n_q / 16) {
        let q_base = chunk * 16;
        for m in 0..16 {
            q_vals[m] = eq_lo_sub[q_base + m] * e_hi;
            let b = m * 16;
            g_bytes[b..b + 8].copy_from_slice(&q_vals[m].lo.to_le_bytes());
            g_bytes[b + 8..b + 16].copy_from_slice(&q_vals[m].hi.to_le_bytes());
        }
        for byte_pos in 0..16 {
            let mut w_lo: u64 = 0;
            let mut w_hi: u64 = 0;
            for r in 0..8 {
                w_lo |= (g_bytes[r * 16 + byte_pos] as u64) << (r * 8);
                w_hi |= (g_bytes[(r + 8) * 16 + byte_pos] as u64) << (r * 8);
            }
            let t_lo = transpose_8x8_bits(w_lo).to_le_bytes();
            let t_hi = transpose_8x8_bits(w_hi).to_le_bytes();
            let base_t = byte_pos * 8;
            for b in 0..8 {
                col_masks[base_t + b] = (t_lo[b] as u16) | ((t_hi[b] as u16) << 8);
            }
        }

        let w_chunk = &w_sub[(q_base << 6)..((q_base + 16) << 6)];
        for bank in 0..64 {
            let a_bank = &mut a_state[(bank << 7)..((bank + 1) << 7)];
            for nib in 0..4 {
                let m_base = nib * 4;
                let e0 = w_chunk[((m_base) << 6) | bank];
                let e1 = w_chunk[((m_base + 1) << 6) | bank];
                let e2 = w_chunk[((m_base + 2) << 6) | bank];
                let e3 = w_chunk[((m_base + 3) << 6) | bank];
                let e01 = e0 + e1;
                let e23 = e2 + e3;
                table[0] = F128::ZERO;
                table[1] = e0;
                table[2] = e1;
                table[3] = e01;
                table[4] = e2;
                table[5] = e0 + e2;
                table[6] = e1 + e2;
                table[7] = e01 + e2;
                table[8] = e3;
                table[9] = e0 + e3;
                table[10] = e1 + e3;
                table[11] = e01 + e3;
                table[12] = e23;
                table[13] = e0 + e23;
                table[14] = e1 + e23;
                table[15] = e01 + e23;

                let shift = nib * 4;
                for t in 0..128 {
                    let idx = ((col_masks[t] >> shift) & 0xF) as usize;
                    a_bank[t] += table[idx];
                }
            }
        }
    }
}

fn direct_fold8_w_state_into(base_table: &[F128], eq_low: &[F128], w_state: &mut [F128]) {
    debug_assert_eq!(base_table.len(), FOLD_TABLE_TOTAL);
    debug_assert_eq!(eq_low.len(), 64);
    debug_assert_eq!(w_state.len(), 64 * 128);
    for (bank, &e_low) in eq_low.iter().enumerate() {
        let mut x_t_elow = e_low;
        let dst = &mut w_state[(bank << 7)..((bank + 1) << 7)];
        for slot in dst.iter_mut() {
            *slot = fold_one_slot(x_t_elow, base_table);
            x_t_elow = crate::field::mul_by_x(x_t_elow);
        }
    }
}

pub(crate) fn direct_fold8_round0(factors: &DirectFold8Factors) -> (F128, F128) {
    let mut u0 = crate::field::F256Unreduced::ZERO;
    let mut u2 = crate::field::F256Unreduced::ZERO;
    for pair in 0..32 {
        let a0 = &factors.a_state[(2 * pair) * 128..(2 * pair + 1) * 128];
        let a1 = &factors.a_state[(2 * pair + 1) * 128..(2 * pair + 2) * 128];
        let w0 = &factors.w_state[(2 * pair) * 128..(2 * pair + 1) * 128];
        let w1 = &factors.w_state[(2 * pair + 1) * 128..(2 * pair + 2) * 128];
        for t in 0..128 {
            u0 ^= a0[t].mul_unreduced(w0[t]);
            u2 ^= (a0[t] + a1[t]).mul_unreduced(w0[t] + w1[t]);
        }
    }
    (u0.reduce(), u2.reduce())
}

pub fn verify<Ch: Challenger>(
    claim: F128,
    z_skip: F128,
    x_outer: &[F128],
    proof: &RingSwitchProof,
    challenger: &mut Ch,
) -> Result<RingSwitchOutput, VerifyError> {
    assert!(!x_outer.is_empty());
    let l = 1usize << (x_outer.len() - 1);
    assert_eq!(proof.s_hat_v.len(), 1 << LOG_PACKING);

    challenger.observe_label(b"flock-ring-switch-v0");

    challenger.observe_f128_slice(&proof.s_hat_v);

    let weights = build_claim_weights(z_skip, x_outer[0]);
    if claim_check(&weights, &proof.s_hat_v) != claim {
        return Err(VerifyError::ClaimMismatch);
    }

    let r_dprime = challenger.sample_f128_vec(LOG_PACKING);
    let eq_r_dprime = build_eq(&r_dprime);

    let s_hat_u = tensor_algebra_transpose(&proof.s_hat_v);
    let sumcheck_claim = inner_product(&s_hat_u, &eq_r_dprime);

    let suffix = &x_outer[1..];
    let suffix_tensor = build_eq(suffix);
    debug_assert_eq!(suffix_tensor.len(), l);
    let rs_eq_ind = fold_b128_elems(&suffix_tensor, &eq_r_dprime);

    Ok(RingSwitchOutput {
        rs_eq_ind,
        sumcheck_claim,
    })
}

#[derive(Clone, Debug)]
pub struct RingSwitchVerifierOutput {
    pub sumcheck_claim: F128,
    pub eq_r_dprime: Vec<F128>,
}

pub fn verify_succinct<Ch: Challenger>(
    claim: F128,
    z_skip: F128,
    x_outer: &[F128],
    proof: &RingSwitchProof,
    challenger: &mut Ch,
) -> Result<RingSwitchVerifierOutput, VerifyError> {
    assert!(!x_outer.is_empty());
    assert_eq!(proof.s_hat_v.len(), 1 << LOG_PACKING);

    challenger.observe_label(b"flock-ring-switch-v0");
    challenger.observe_f128_slice(&proof.s_hat_v);

    let weights = build_claim_weights(z_skip, x_outer[0]);
    if claim_check(&weights, &proof.s_hat_v) != claim {
        return Err(VerifyError::ClaimMismatch);
    }

    let r_dprime = challenger.sample_f128_vec(LOG_PACKING);
    let eq_r_dprime = build_eq(&r_dprime);

    let s_hat_u = tensor_algebra_transpose(&proof.s_hat_v);
    let sumcheck_claim = inner_product(&s_hat_u, &eq_r_dprime);

    Ok(RingSwitchVerifierOutput {
        sumcheck_claim,
        eq_r_dprime,
    })
}

pub fn eval_rs_eq(z_vals: &[F128], query: &[F128], eq_r_dprime: &[F128]) -> F128 {
    use crate::pcs::tensor_algebra::TensorAlgebra;

    assert_eq!(
        z_vals.len(),
        query.len(),
        "eval_rs_eq: z_vals and query must have equal length"
    );
    assert_eq!(
        eq_r_dprime.len(),
        1 << LOG_PACKING,
        "eval_rs_eq: eq_r_dprime length must be 128"
    );

    let mut eval = TensorAlgebra::from_vertical(F128::ONE);
    for (&z_i, &q_i) in z_vals.iter().zip(query.iter()) {
        let vert_scaled = eval.clone().scale_vertical(z_i);
        let hztl_scaled = eval.clone().scale_horizontal(q_i);
        eval += &vert_scaled;
        eval += &hztl_scaled;
    }
    eval.fold_vertical(eq_r_dprime)
}

pub fn eval_rs_eq_prefix(
    z_vals: &[F128],
    query_prefix: &[F128],
) -> crate::pcs::tensor_algebra::TensorAlgebra {
    use crate::pcs::tensor_algebra::TensorAlgebra;
    assert!(query_prefix.len() <= z_vals.len());
    let mut eval = TensorAlgebra::from_vertical(F128::ONE);
    for (&z_i, &q_i) in z_vals.iter().zip(query_prefix.iter()) {
        let vert_scaled = eval.clone().scale_vertical(z_i);
        let hztl_scaled = eval.clone().scale_horizontal(q_i);
        eval += &vert_scaled;
        eval += &hztl_scaled;
    }
    eval
}

pub fn eval_rs_eq_finish_from_prefix(
    prefix: &crate::pcs::tensor_algebra::TensorAlgebra,
    z_vals_suffix: &[F128],
    query_suffix: &[F128],
    eq_r_dprime: &[F128],
) -> F128 {
    assert_eq!(z_vals_suffix.len(), query_suffix.len());
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    let mut eval = prefix.clone();
    for (&z_i, &q_i) in z_vals_suffix.iter().zip(query_suffix.iter()) {
        let vert_scaled = eval.clone().scale_vertical(z_i);
        let hztl_scaled = eval.clone().scale_horizontal(q_i);
        eval += &vert_scaled;
        eval += &hztl_scaled;
    }
    eval.fold_vertical(eq_r_dprime)
}

pub fn eval_rs_eq_finish_from_prefix_binary_q(
    prefix: &crate::pcs::tensor_algebra::TensorAlgebra,
    z_vals_suffix: &[F128],
    y_bits: u32,
    eq_r_dprime: &[F128],
) -> F128 {
    assert_eq!(eq_r_dprime.len(), 1 << LOG_PACKING);
    debug_assert!(
        z_vals_suffix.len() <= 32,
        "y_bits is u32; suffix > 32 not supported"
    );
    let mut eval = prefix.clone();
    for (j, &z_i) in z_vals_suffix.iter().enumerate() {
        let scalar = if (y_bits >> j) & 1 == 1 {
            z_i
        } else {
            F128::ONE + z_i
        };
        for e in eval.elems.iter_mut() {
            *e *= scalar;
        }
    }
    eval.fold_vertical(eq_r_dprime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::pack::pack_witness;
    use crate::zerocheck::univariate_skip::build_eq;

    #[test]
    fn eval_rs_eq_finish_binary_q_matches_general() {
        use crate::challenger::Challenger;
        let mut rng = crate::challenger::RandomChallenger::new(0x_B17_0BBE);
        let log_n = 20usize;
        let prefix_len = 15usize;
        let suffix_len = log_n - prefix_len;
        let z_vals: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let query_prefix: Vec<F128> = (0..prefix_len).map(|_| rng.sample_f128()).collect();
        let eq_r_dprime: Vec<F128> = (0..(1 << LOG_PACKING)).map(|_| rng.sample_f128()).collect();
        let prefix = eval_rs_eq_prefix(&z_vals[..prefix_len], &query_prefix);

        for y in 0..(1usize << suffix_len) {
            let query_suffix: Vec<F128> = (0..suffix_len)
                .map(|j| {
                    if (y >> j) & 1 == 1 {
                        F128::ONE
                    } else {
                        F128::ZERO
                    }
                })
                .collect();
            let general = eval_rs_eq_finish_from_prefix(
                &prefix,
                &z_vals[prefix_len..],
                &query_suffix,
                &eq_r_dprime,
            );
            let binary = eval_rs_eq_finish_from_prefix_binary_q(
                &prefix,
                &z_vals[prefix_len..],
                y as u32,
                &eq_r_dprime,
            );
            assert_eq!(general, binary, "y={y} mismatch");
        }
    }

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
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    fn zhat_skip_reference(z: &[bool], m: usize, z_skip: F128, x_outer: &[F128]) -> F128 {
        const K_SKIP: usize = 6;
        let ell = 1usize << K_SKIP;
        assert_eq!(z.len(), 1 << m);
        assert_eq!(x_outer.len(), m - K_SKIP);

        let lambda = lagrange_weights_naive(K_SKIP, z_skip);
        let eq_outer = build_eq(x_outer);

        let mut acc = F128::ZERO;
        for i_outer in 0..(1usize << (m - K_SKIP)) {
            let base = i_outer * ell;
            let mut inner = F128::ZERO;
            for i_skip in 0..ell {
                if z[base + i_skip] {
                    inner += lambda[i_skip];
                }
            }
            acc += eq_outer[i_outer] * inner;
        }
        acc
    }

    #[test]
    fn claim_check_recovers_zhat_skip() {
        let mut rng = Rng::new(0xAA7);
        for &m in &[8usize, 9, 10] {
            let z = rng.bits(1 << m);
            let z_skip = rng.f128();
            let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();

            let expected = zhat_skip_reference(&z, m, z_skip, &x_outer);

            let packed = pack_witness(&z, m);
            let suffix_tensor = build_eq(&x_outer[1..]);
            assert_eq!(packed.len(), suffix_tensor.len());
            let s_hat_v = fold_1b_rows_naive(&packed, &suffix_tensor);

            let weights = build_claim_weights(z_skip, x_outer[0]);
            let got = claim_check(&weights, &s_hat_v);

            assert_eq!(got, expected, "claim-check mismatch at m={m}");
        }
    }

    #[test]
    fn weights_have_correct_length() {
        let w = build_claim_weights(F128 { lo: 1, hi: 0 }, F128 { lo: 2, hi: 0 });
        assert_eq!(w.len(), 128);
    }

    #[test]
    fn prove_verify_roundtrip() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0xBEEF);
        for &m in &[8usize, 9, 10, 11] {
            let z = rng.bits(1 << m);
            let z_skip = rng.f128();
            let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();

            let claim = zhat_skip_reference(&z, m, z_skip, &x_outer);

            let packed = pack_witness(&z, m);

            let mut ch_p = FsChallenger::new(b"flock-test-v0");
            let (proof, out_p) = prove(&packed, &x_outer, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"flock-test-v0");
            let out_v = verify(claim, z_skip, &x_outer, &proof, &mut ch_v)
                .unwrap_or_else(|e| panic!("verify rejected honest at m={m}: {e:?}"));

            assert_eq!(
                out_p.sumcheck_claim, out_v.sumcheck_claim,
                "sumcheck_claim mismatch at m={m}"
            );
            assert_eq!(
                out_p.rs_eq_ind, out_v.rs_eq_ind,
                "rs_eq_ind mismatch at m={m}"
            );
        }
    }

    #[test]
    fn dp24_identity_holds() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0xABCD);
        for &m in &[8usize, 9, 10, 11] {
            let z = rng.bits(1 << m);
            let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();

            let packed = pack_witness(&z, m);
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let (_proof, out) = prove(&packed, &x_outer, &mut ch);

            let lhs = inner_product(&packed, &out.rs_eq_ind);
            assert_eq!(lhs, out.sumcheck_claim, "DP24 identity fails at m={m}");
        }
    }

    #[test]
    fn verify_rejects_mutated_proof() {
        use crate::challenger::FsChallenger;
        let m = 10usize;
        let mut rng = Rng::new(0x99);
        let z = rng.bits(1 << m);
        let z_skip = rng.f128();
        let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
        let claim = zhat_skip_reference(&z, m, z_skip, &x_outer);
        let packed = pack_witness(&z, m);

        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (mut proof, _) = prove(&packed, &x_outer, &mut ch_p);
        proof.s_hat_v[0].lo ^= 1;

        let mut ch_v = FsChallenger::new(b"flock-test-v0");
        let res = verify(claim, z_skip, &x_outer, &proof, &mut ch_v);
        assert!(matches!(res, Err(VerifyError::ClaimMismatch)));
    }

    #[test]
    fn transpose_is_involution() {
        let mut rng = Rng::new(0xDEAD);
        let s_hat_v: Vec<F128> = (0..128).map(|_| rng.f128()).collect();
        let twice = tensor_algebra_transpose(&tensor_algebra_transpose(&s_hat_v));
        assert_eq!(s_hat_v, twice);
    }

    #[test]
    fn prove_batched_matches_sequential() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0x1234_5678);
        for &m in &[8usize, 9, 10, 11] {
            let z = rng.bits(1 << m);
            let x_a: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
            let x_b: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
            let packed = pack_witness(&z, m);

            let mut ch_seq = FsChallenger::new(b"flock-test-v0");
            let (p_a, o_a) = prove(&packed, &x_a, &mut ch_seq);
            let (p_b, o_b) = prove(&packed, &x_b, &mut ch_seq);

            let mut ch_batch = FsChallenger::new(b"flock-test-v0");
            let (results, _gammas_rs) = prove_batched(&packed, &[&x_a, &x_b], &mut ch_batch);

            assert_eq!(results[0].0, p_a, "s_hat_v[0] mismatch at m={m}");
            assert_eq!(results[1].0, p_b, "s_hat_v[1] mismatch at m={m}");
            assert_eq!(results[0].1.sumcheck_claim, o_a.sumcheck_claim);
            assert_eq!(results[1].1.sumcheck_claim, o_b.sumcheck_claim);
            assert_eq!(results[0].1.rs_eq_ind.len(), o_a.rs_eq_ind.len());
            assert_eq!(results[1].1.rs_eq_ind.len(), o_b.rs_eq_ind.len());
        }
    }

    #[test]
    fn mfr_fold_matches_scalar_bit_scan() {
        let mut rng = Rng::new(0xBEEF_D00D);
        for &m in &[9usize, 11, 13, 14] {
            let l = m - 7;
            let pw_len = 1usize << l;
            let pw: Vec<F128> = (0..pw_len).map(|_| rng.f128()).collect();
            let suffix0: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let suffix1: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let tensor0 = build_eq(&suffix0);
            let tensor1 = build_eq(&suffix1);

            let s0_ref = fold_1b_rows_naive(&pw, &tensor0);
            let s1_ref = fold_1b_rows_naive(&pw, &tensor1);

            let (s0_mfr, s1_mfr) = fold_1b_rows_2way_mfr(&pw, &tensor0, &tensor1);

            assert_eq!(s0_mfr, s0_ref, "s_hat_v0 mismatch at m={m}");
            assert_eq!(s1_mfr, s1_ref, "s_hat_v1 mismatch at m={m}");
        }
    }

    #[test]
    fn mfr_fold_8wide_matches_scalar() {
        let mut rng = Rng::new(0x8888_1357);
        for &m in &[10usize, 12, 13, 16] {
            let l = m - 7;
            let pw_len = 1usize << l;
            let pw: Vec<F128> = (0..pw_len).map(|_| rng.f128()).collect();
            let suffix0: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let suffix1: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let t0 = build_eq(&suffix0);
            let t1 = build_eq(&suffix1);

            let s0_ref = fold_1b_rows_naive(&pw, &t0);
            let s1_ref = fold_1b_rows_naive(&pw, &t1);
            assert_eq!(
                fold_1b_rows_1way_mfr_8wide_k4(&pw, &t0),
                s0_ref,
                "1-way 8wide m={m}"
            );
            let (s0, s1) = fold_1b_rows_2way_mfr_8wide(&pw, &t0, &t1);
            assert_eq!(s0, s0_ref, "2-way 8wide s0 m={m}");
            assert_eq!(s1, s1_ref, "2-way 8wide s1 m={m}");
        }
    }

    #[test]
    fn fold_1b_padded_matches_dense() {
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xCAFE_FACE_u64.wrapping_add((k_log * 31 + m) as u64));
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;
            let n_blocks = 1usize << (m - k_log);

            let mut z = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    z[blk * block_size + j] = false;
                }
            }
            let packed = pack_witness(&z, m);

            let len = packed.len();
            let t0: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
            let t1: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };

            if packed.len().is_multiple_of(8) {
                let dense = fold_1b_rows_2way_mfr_8wide(&packed, &t0, &t1);
                let padded = fold_1b_rows_2way_mfr_8wide_padded(&packed, &t0, &t1, &padding);
                assert_eq!(
                    dense, padded,
                    "8-wide mismatch: m={m}, k_log={k_log}, useful={useful_bits}"
                );
            }
            if packed.len().is_multiple_of(4) {
                let dense = fold_1b_rows_2way_mfr(&packed, &t0, &t1);
                let padded = fold_1b_rows_2way_mfr_padded(&packed, &t0, &t1, &padding);
                assert_eq!(
                    dense, padded,
                    "4-wide mismatch: m={m}, k_log={k_log}, useful={useful_bits}"
                );
            }
        }
    }

    #[test]
    fn build_eq_split_reconstructs_full() {
        let mut rng = Rng::new(0x9911);
        for &l in &[4usize, 7, 10] {
            let r: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let full = build_eq(&r);
            for n_lo in 0..=l {
                let (eq_lo, eq_hi) = build_eq_split(&r, n_lo);
                assert_eq!(eq_lo.len(), 1 << n_lo);
                assert_eq!(eq_hi.len(), 1 << (l - n_lo));
                let mask = (1usize << n_lo) - 1;
                for (i, &f) in full.iter().enumerate() {
                    let recon = eq_lo[i & mask] * eq_hi[i >> n_lo];
                    assert_eq!(recon, f, "reconstruct mismatch l={l} n_lo={n_lo} i={i}");
                }
            }
        }
    }

    #[test]
    fn fold_1b_rows_split_matches_16wide() {
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let l = m - LOG_PACKING;
            let len = 1usize << l;
            let mut rng = Rng::new(0x5757_u64.wrapping_add((m * 131 + k_log) as u64));
            let w: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
            let r: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let full_eq = build_eq(&r);
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };

            let reference = fold_1b_rows_1way_mfr_16wide_padded(&w, &full_eq, &padding);
            for n_lo in 4..=l {
                let (eq_lo, eq_hi) = build_eq_split(&r, n_lo);
                let got = fold_1b_rows_split(&w, &eq_lo, &eq_hi, &padding);
                assert_eq!(
                    got, reference,
                    "fold_1b_rows_split mismatch: m={m}, k_log={k_log}, n_lo={n_lo}"
                );
            }
            let (eq_lo, eq_hi) = build_eq_split(&r, split_n_lo(l));
            assert_eq!(
                fold_1b_rows_split(&w, &eq_lo, &eq_hi, &padding),
                reference,
                "fold_1b_rows_split mismatch at split_n_lo: m={m}"
            );
        }
    }

    #[test]
    fn fold_1b_rows_split_2way_matches_per_claim() {
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let l = m - LOG_PACKING;
            let len = 1usize << l;
            let mut rng = Rng::new(0xBEEF_u64.wrapping_add((m * 131 + k_log) as u64));
            let w: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };
            let n_lo = split_n_lo(l);
            let r0: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let r1: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let (lo0, hi0) = build_eq_split(&r0, n_lo);
            let (lo1, hi1) = build_eq_split(&r1, n_lo);
            let (got0, got1) = fold_1b_rows_split_2way(&w, &lo0, &hi0, &lo1, &hi1, &padding);
            let want0 = fold_1b_rows_split(&w, &lo0, &hi0, &padding);
            let want1 = fold_1b_rows_split(&w, &lo1, &hi1, &padding);
            assert_eq!(
                got0, want0,
                "fold_1b_rows_split_2way mismatch (claim 0) m={m}"
            );
            assert_eq!(
                got1, want1,
                "fold_1b_rows_split_2way mismatch (claim 1) m={m}"
            );
        }
    }

    #[test]
    fn fold_b128_elems_split_matches_dense() {
        let mut rng = Rng::new(0xB0B0);
        for &l in &[4usize, 8, 10] {
            let r: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let full_eq = build_eq(&r);
            let eq_r: Vec<F128> = (0..128).map(|_| rng.f128()).collect();
            let reference = fold_b128_elems(&full_eq, &eq_r);
            for n_lo in 4..=l {
                let (eq_lo, eq_hi) = build_eq_split(&r, n_lo);
                let got = fold_b128_elems_split(&eq_lo, &eq_hi, &eq_r);
                assert_eq!(
                    got, reference,
                    "fold_b128_elems_split mismatch l={l} n_lo={n_lo}"
                );
            }
        }
    }

    #[test]
    fn s_hat_v_from_z_vec_matches_fold_1b_rows_ab() {
        use crate::lincheck::{pack_z_lincheck, partial_fold_packed_z};
        const K_SKIP: usize = 6;
        let cases: &[(usize, usize)] = &[(13, 10), (15, 11), (17, 13)];
        for &(m, k_log) in cases {
            assert!(k_log >= LOG_PACKING);
            assert!(k_log >= K_SKIP);
            let n_log = m - k_log;
            assert!(n_log >= 3);
            let mut rng = Rng::new(0xCAFE_u64.wrapping_add((m * 131 + k_log) as u64));

            let z = rng.bits(1 << m);
            let packed = pack_witness(&z, m);
            let z_packed_lincheck = pack_z_lincheck(&z, m, k_log);

            let x_inner_rest: Vec<F128> = (0..(k_log - K_SKIP)).map(|_| rng.f128()).collect();
            let x_outer: Vec<F128> = (0..n_log).map(|_| rng.f128()).collect();

            let mut x_outer_full = Vec::with_capacity(x_inner_rest.len() + x_outer.len());
            x_outer_full.extend_from_slice(&x_inner_rest);
            x_outer_full.extend_from_slice(&x_outer);
            let suffix = &x_outer_full[1..];
            let suffix_tensor = build_eq(suffix);
            let want = fold_1b_rows_naive(&packed, &suffix_tensor);

            let eq_x_outer = build_eq(&x_outer);
            let z_vec = partial_fold_packed_z(&z_packed_lincheck, m, k_log, &eq_x_outer);
            let got = s_hat_v_from_z_vec(&z_vec, &x_inner_rest[1..]);

            assert_eq!(got, want, "s_hat_v mismatch at m={m}, k_log={k_log}");
        }
    }

    #[test]
    fn prove_batched_with_precomputed_matches_unprecomputed() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0xF00D);
        for &m in &[8usize, 9, 10, 11] {
            let z = rng.bits(1 << m);
            let x_a: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
            let x_b: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
            let packed = pack_witness(&z, m);

            let mut ch_base = FsChallenger::new(b"flock-test-v0");
            let (base, _) = prove_batched(&packed, &[&x_a, &x_b], &mut ch_base);
            let s_hat_v_a = base[0].0.s_hat_v.clone();
            let s_hat_v_b = base[1].0.s_hat_v.clone();

            let padding = PaddingSpec::dense(m);

            for &(pre_a, pre_b) in &[(false, false), (true, false), (false, true), (true, true)] {
                let pa: Option<&[F128]> = if pre_a { Some(&s_hat_v_a) } else { None };
                let pb: Option<&[F128]> = if pre_b { Some(&s_hat_v_b) } else { None };
                let mut ch = FsChallenger::new(b"flock-test-v0");
                let (got, _) = prove_batched_padded_with_precomputed(
                    &packed,
                    &[&x_a, &x_b],
                    &[pa, pb],
                    &padding,
                    &mut ch,
                );
                assert_eq!(
                    got[0].0, base[0].0,
                    "proof[0] mismatch (pre_a={pre_a}, pre_b={pre_b}, m={m})"
                );
                assert_eq!(
                    got[1].0, base[1].0,
                    "proof[1] mismatch (pre_a={pre_a}, pre_b={pre_b}, m={m})"
                );
                assert_eq!(got[0].1.sumcheck_claim, base[0].1.sumcheck_claim);
                assert_eq!(got[1].1.sumcheck_claim, base[1].1.sumcheck_claim);
                assert_eq!(
                    got[0].1.rs_eq_ind.to_dense(),
                    base[0].1.rs_eq_ind.to_dense()
                );
                assert_eq!(
                    got[1].1.rs_eq_ind.to_dense(),
                    base[1].1.rs_eq_ind.to_dense()
                );
            }
        }
    }

    #[test]
    fn s_hat_v_from_z_vec_degenerate_tail() {
        let mut rng = Rng::new(0xDEAD);
        let z_vec: Vec<F128> = (0..(1 << LOG_PACKING)).map(|_| rng.f128()).collect();
        let got = s_hat_v_from_z_vec(&z_vec, &[]);
        assert_eq!(got, z_vec);
    }

    #[test]
    #[ignore]
    fn zzz_bench_fold_1b() {
        let l = 22;
        let pw_len = 1usize << l;
        let mut rng = Rng::new(0x1111);
        let pw: Vec<F128> = (0..pw_len).map(|_| rng.f128()).collect();
        let t0 = build_eq(&(0..l).map(|_| rng.f128()).collect::<Vec<_>>());
        let t1 = build_eq(&(0..l).map(|_| rng.f128()).collect::<Vec<_>>());

        let iters = 20;
        let bench = |f: &dyn Fn()| {
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        };
        let t_k4 = bench(&|| {
            std::hint::black_box(fold_1b_rows_1way_mfr(&pw, &t0));
        });
        let t_8 = bench(&|| {
            std::hint::black_box(fold_1b_rows_1way_mfr_8wide_k4(&pw, &t0));
        });
        let t_2k4 = bench(&|| {
            std::hint::black_box(fold_1b_rows_2way_mfr(&pw, &t0, &t1));
        });
        let t_28 = bench(&|| {
            std::hint::black_box(fold_1b_rows_2way_mfr_8wide(&pw, &t0, &t1));
        });
        eprintln!(
            "\n  [fold_1b @ m=29] 1-way: {t_k4:5.2}→{t_8:5.2} ms ({:.2}x) | 2-way: {t_2k4:5.2}→{t_28:5.2} ms ({:.2}x)\n",
            t_k4 / t_8,
            t_2k4 / t_28,
        );
    }

    #[test]
    fn mfr_fold_16wide_2way_matches_scalar() {
        let mut rng = Rng::new(0x161616);
        for &m in &[11usize, 12, 14, 16] {
            let l = m - 7;
            let pw_len = 1usize << l;
            let pw: Vec<F128> = (0..pw_len).map(|_| rng.f128()).collect();
            let suffix0: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let suffix1: Vec<F128> = (0..l).map(|_| rng.f128()).collect();
            let t0 = build_eq(&suffix0);
            let t1 = build_eq(&suffix1);

            let s0_ref = fold_1b_rows_naive(&pw, &t0);
            let s1_ref = fold_1b_rows_naive(&pw, &t1);
            let dense = PaddingSpec::dense(m);
            let (s0, s1) = fold_1b_rows_2way_mfr_16wide_padded(&pw, &t0, &t1, &dense);
            assert_eq!(s0, s0_ref, "2-way 16wide s0 m={m}");
            assert_eq!(s1, s1_ref, "2-way 16wide s1 m={m}");
        }
    }

    #[test]
    fn subset_sums_4_correctness() {
        let mut rng = Rng::new(0xABCD);
        let elems: [F128; 4] = [rng.f128(), rng.f128(), rng.f128(), rng.f128()];
        let sums = subset_sums_4(elems);
        assert_eq!(sums[0], F128::ZERO);
        for mask in 0..16 {
            let mut expected = F128::ZERO;
            for k in 0..4 {
                if (mask >> k) & 1 == 1 {
                    expected += elems[k];
                }
            }
            assert_eq!(sums[mask], expected, "mask={mask:04b}");
        }
    }

    #[test]
    fn fold_b128_elems_matches_naive() {
        let mut rng = Rng::new(0xF00D);
        for &l in &[1usize, 4, 8, 12] {
            let len = 1usize << l;
            let suffix: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
            let eq_r: Vec<F128> = (0..128).map(|_| rng.f128()).collect();
            let a = fold_b128_elems_naive(&suffix, &eq_r);
            let b = fold_b128_elems(&suffix, &eq_r);
            assert_eq!(a, b, "fold_b128_elems mismatch at L={l}");
        }
    }

    #[test]
    fn s_hat_v_is_linear_in_witness() {
        let mut rng = Rng::new(0x42);
        let m = 9;
        let z1 = rng.bits(1 << m);
        let z2 = rng.bits(1 << m);
        let z_xor: Vec<bool> = z1.iter().zip(&z2).map(|(a, b)| a ^ b).collect();
        let x_outer: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
        let suffix_tensor = build_eq(&x_outer[1..]);

        let s1 = fold_1b_rows_naive(&pack_witness(&z1, m), &suffix_tensor);
        let s2 = fold_1b_rows_naive(&pack_witness(&z2, m), &suffix_tensor);
        let sx = fold_1b_rows_naive(&pack_witness(&z_xor, m), &suffix_tensor);

        for (i, ((&a, &b), &c)) in s1.iter().zip(&s2).zip(&sx).enumerate() {
            assert_eq!(a + b, c, "linearity fails at i={i}");
        }
    }

    fn mk_coords(rng: &mut Rng, n: usize, zero_positions: &[usize]) -> Vec<F128> {
        (0..n)
            .map(|i| {
                if zero_positions.contains(&i) {
                    F128::ZERO
                } else {
                    rng.f128()
                }
            })
            .collect()
    }

    #[test]
    fn build_eq_sparse_matches_dense() {
        let mut rng = Rng::new(0xCAFE_F00D);
        let cases: &[(usize, &[usize])] = &[
            (1, &[0]),
            (4, &[1, 3]),
            (6, &[0, 1, 2, 3, 4]),
            (8, &[2, 3, 4, 5, 6]),
            (10, &[]),
            (10, &[0, 5, 9]),
        ];
        for &(n_coords, zero_pos) in cases {
            let coords = mk_coords(&mut rng, n_coords, zero_pos);
            let dense = build_eq(&coords);
            let sparse_eq = build_eq_sparse(&coords);
            let materialized = sparse_eq.materialize();

            let mut covered = vec![false; dense.len()];
            for &(idx, val) in &materialized {
                assert_eq!(
                    val, dense[idx],
                    "sparse value mismatch at idx={idx} (n={n_coords}, zeros={zero_pos:?})"
                );
                assert_ne!(
                    val,
                    F128::ZERO,
                    "sparse entry is zero — should have been skipped"
                );
                covered[idx] = true;
            }
            for (i, &c) in covered.iter().enumerate() {
                if !c {
                    assert_eq!(
                        dense[i],
                        F128::ZERO,
                        "dense[{i}] nonzero but absent from sparse (n={n_coords}, zeros={zero_pos:?})"
                    );
                }
            }
            for w in materialized.windows(2) {
                assert!(w[0].0 < w[1].0, "support not strictly ascending");
            }
            let live_count = n_coords - zero_pos.len();
            assert_eq!(sparse_eq.len(), 1usize << live_count);
        }
    }

    #[test]
    fn fold_1b_rows_sparse_matches_naive() {
        let mut rng = Rng::new(0x5EED_DEAD);
        for &m in &[9usize, 11, 13] {
            let l = m - 7;
            let pw_len = 1usize << l;
            let pw: Vec<F128> = (0..pw_len).map(|_| rng.f128()).collect();
            let zero_pos: Vec<usize> = (0..l.min(3)).collect();
            let suffix = mk_coords(&mut rng, l, &zero_pos);

            let dense_tensor = build_eq(&suffix);
            let sparse_eq = build_eq_sparse(&suffix);

            let dense_s = fold_1b_rows_naive(&pw, &dense_tensor);
            let sparse_s = fold_1b_rows_sparse(&pw, &sparse_eq);

            assert_eq!(dense_s, sparse_s, "s_hat_v mismatch at m={m}");
        }
    }

    #[test]
    fn fold_b128_elems_sparse_matches_dense() {
        let mut rng = Rng::new(0xC0DE_BABE);
        for &l in &[4usize, 8, 12] {
            let len = 1usize << l;
            let zero_pos: Vec<usize> = (0..l.min(3)).collect();
            let suffix = mk_coords(&mut rng, l, &zero_pos);
            let dense_tensor = build_eq(&suffix);
            let sparse_eq = build_eq_sparse(&suffix);
            let eq_r: Vec<F128> = (0..128).map(|_| rng.f128()).collect();

            let dense_out = fold_b128_elems(&dense_tensor, &eq_r);
            let sparse_out = fold_b128_elems_sparse(len, &sparse_eq, &eq_r);

            assert_eq!(dense_out, sparse_out, "rs_eq_ind mismatch at L={l}");
        }
    }

    #[test]
    fn prove_batched_with_sparse_claim_matches_sequential() {
        use crate::challenger::FsChallenger;
        let mut rng = Rng::new(0xBEEF_CAFE);
        for &m in &[10usize, 11, 12] {
            let z = rng.bits(1 << m);
            let packed = pack_witness(&z, m);
            let mut x_chain: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
            for j in 0..3 {
                x_chain[(m - 6) - 1 - j] = F128::ZERO;
            }
            let x_ab: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();
            let x_c: Vec<F128> = (0..(m - 6)).map(|_| rng.f128()).collect();

            let mut ch_seq = FsChallenger::new(b"flock-test-sparse");
            let (p_ab, o_ab) = prove(&packed, &x_ab, &mut ch_seq);
            let (p_c, o_c) = prove(&packed, &x_c, &mut ch_seq);
            let (p_chain, o_chain) = prove(&packed, &x_chain, &mut ch_seq);

            let mut ch_batch = FsChallenger::new(b"flock-test-sparse");
            let (results, _) = prove_batched(&packed, &[&x_ab, &x_c, &x_chain], &mut ch_batch);

            assert_eq!(results[0].0, p_ab, "s_hat_v[ab] mismatch at m={m}");
            assert_eq!(results[1].0, p_c, "s_hat_v[c]  mismatch at m={m}");
            assert_eq!(results[2].0, p_chain, "s_hat_v[chain] mismatch at m={m}");
            assert!(
                matches!(results[2].1.rs_eq_ind, RsEqInd::Sparse { .. }),
                "chain claim should be sparse"
            );
            assert!(
                matches!(
                    results[0].1.rs_eq_ind,
                    RsEqInd::Dense(_) | RsEqInd::DeferredDense { .. }
                ),
                "ab claim should be dense"
            );
            assert!(
                matches!(
                    results[1].1.rs_eq_ind,
                    RsEqInd::Dense(_) | RsEqInd::DeferredDense { .. }
                ),
                "c claim should be dense"
            );
            assert_eq!(results[0].1.sumcheck_claim, o_ab.sumcheck_claim);
            assert_eq!(results[1].1.sumcheck_claim, o_c.sumcheck_claim);
            assert_eq!(results[2].1.sumcheck_claim, o_chain.sumcheck_claim);
            let _ = (&o_ab.rs_eq_ind, &o_c.rs_eq_ind, &o_chain.rs_eq_ind);
        }
    }

    #[test]
    fn eval_rs_eq_matches_dense() {
        fn mle_eval_naive(values: &[F128], r: &[F128]) -> F128 {
            assert_eq!(values.len(), 1 << r.len());
            let mut buf = values.to_vec();
            for &r_i in r.iter().rev() {
                let half = buf.len() / 2;
                for i in 0..half {
                    let lo = buf[i];
                    let hi = buf[i + half];
                    buf[i] = lo + r_i * (lo + hi);
                }
                buf.truncate(half);
            }
            buf[0]
        }

        let mut rng = Rng::new(0xDEADBEEF);
        for &l_prime in &[3usize, 6, 10, 14] {
            for _trial in 0..3 {
                let z_vals: Vec<F128> = (0..l_prime).map(|_| rng.f128()).collect();
                let query: Vec<F128> = (0..l_prime).map(|_| rng.f128()).collect();
                let r_dprime: Vec<F128> = (0..LOG_PACKING).map(|_| rng.f128()).collect();
                let eq_r_dprime = build_eq(&r_dprime);

                let suffix_tensor = build_eq(&z_vals);
                let rs_eq_ind_dense = fold_b128_elems(&suffix_tensor, &eq_r_dprime);
                let dense_eval = mle_eval_naive(&rs_eq_ind_dense, &query);

                let succinct_eval = eval_rs_eq(&z_vals, &query, &eq_r_dprime);

                assert_eq!(
                    succinct_eval, dense_eval,
                    "eval_rs_eq mismatch at l_prime={l_prime}"
                );
            }
        }
    }
}

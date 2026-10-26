use super::super::{F128, build_sum_table};

const NEON_TILE_T: usize = 8;

#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_single(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let k = 1usize << k_log;
    partial_fold_packed_z_neon_single_padded(z_packed, m, k_log, k, eq_outer)
}

#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_single_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    const TILE_T: usize = NEON_TILE_T;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(
        n_log >= 3 + TILE_T.trailing_zeros() as usize,
        "need n_outer ≥ 8·TILE_T stripes"
    );
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_tiles = n_stripes / TILE_T;
    let n_blocks_full = k / BLOCK_K;

    let n_blocks = useful_bits.div_ceil(BLOCK_K).min(n_blocks_full);

    let tiles_per_chunk = (n_tiles / 256).max(1);
    let bytes_per_chunk = tiles_per_chunk * TILE_T * k;

    z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![F128::ZERO; k],
            |mut out, (chunk_idx, chunk_bytes)| {
                let tile_start = chunk_idx * tiles_per_chunk;

                let mut tables = vec![F128::ZERO; TILE_T * 256];

                let n_tiles_in_chunk = chunk_bytes.len() / (TILE_T * k);
                for tile_rel in 0..n_tiles_in_chunk {
                    let tile_idx = tile_start + tile_rel;
                    let stripe_base = tile_idx * TILE_T;
                    let tile_bytes_ptr = unsafe { chunk_bytes.as_ptr().add(tile_rel * TILE_T * k) };

                    for t in 0..TILE_T {
                        let byte_idx = stripe_base + t;
                        let eq_off = 8 * byte_idx;
                        build_sum_table(
                            &eq_outer[eq_off..eq_off + 8],
                            &mut tables[t * 256..(t + 1) * 256],
                        );
                    }

                    let tables_ptr = tables.as_ptr() as *const u8;

                    for block_idx in 0..n_blocks {
                        let bs = block_idx * BLOCK_K;
                        unsafe {
                            process_block_neon_single(
                                tile_bytes_ptr,
                                k,
                                bs,
                                tables_ptr,
                                out.as_mut_ptr().add(bs),
                            );
                        }
                    }
                }

                let _ = unsafe { vdupq_n_u8(0) };
                out
            },
        )
        .reduce(
            || vec![F128::ZERO; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn process_block_neon_single(
    tile_bytes_ptr: *const u8,
    k: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    use std::arch::aarch64::*;
    const TILE_T: usize = NEON_TILE_T;

    let o = out_ptr as *mut u8;

    let mut a0 = vld1q_u8(o);
    let mut a1 = vld1q_u8(o.add(16));
    let mut a2 = vld1q_u8(o.add(32));
    let mut a3 = vld1q_u8(o.add(48));
    let mut a4 = vld1q_u8(o.add(64));
    let mut a5 = vld1q_u8(o.add(80));
    let mut a6 = vld1q_u8(o.add(96));
    let mut a7 = vld1q_u8(o.add(112));

    let base_ptr = tile_bytes_ptr.add(bs);
    for t in 0..TILE_T {
        let w = (base_ptr.add(t * k) as *const u64).read_unaligned() as usize;
        let ta = tables_ptr.add(t * 256 * 16);

        a0 = veorq_u8(a0, vld1q_u8(ta.add((w & 0xFF) << 4)));
        a1 = veorq_u8(a1, vld1q_u8(ta.add(((w >> 8) & 0xFF) << 4)));
        a2 = veorq_u8(a2, vld1q_u8(ta.add(((w >> 16) & 0xFF) << 4)));
        a3 = veorq_u8(a3, vld1q_u8(ta.add(((w >> 24) & 0xFF) << 4)));
        a4 = veorq_u8(a4, vld1q_u8(ta.add(((w >> 32) & 0xFF) << 4)));
        a5 = veorq_u8(a5, vld1q_u8(ta.add(((w >> 40) & 0xFF) << 4)));
        a6 = veorq_u8(a6, vld1q_u8(ta.add(((w >> 48) & 0xFF) << 4)));
        a7 = veorq_u8(a7, vld1q_u8(ta.add((w >> 56) << 4)));
    }

    vst1q_u8(o, a0);
    vst1q_u8(o.add(16), a1);
    vst1q_u8(o.add(32), a2);
    vst1q_u8(o.add(48), a3);
    vst1q_u8(o.add(64), a4);
    vst1q_u8(o.add(80), a5);
    vst1q_u8(o.add(96), a6);
    vst1q_u8(o.add(112), a7);
}

#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_iblock_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = NEON_TILE_T;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(
        n_log >= 3 + TILE_T.trailing_zeros() as usize,
        "need n_outer ≥ 8·TILE_T stripes"
    );
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_tiles = n_stripes / TILE_T;

    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);

    let mut out = vec![F128::ZERO; k];
    if useful == 0 {
        return out;
    }

    let p = rayon::current_num_threads().max(1);
    let chunks_per_worker = if p <= crate::perf_core_count_cached() {
        1
    } else {
        3
    };
    let i_chunk = (useful / (p * chunks_per_worker))
        .max(BLOCK_K)
        .next_multiple_of(BLOCK_K);

    out[..useful]
        .par_chunks_mut(i_chunk)
        .enumerate()
        .for_each(|(ci, out_slice)| {
            let i_base = ci * i_chunk;
            let n_block = out_slice.len() / BLOCK_K;

            let mut tables = vec![F128::ZERO; TILE_T * 256];
            for tile in 0..n_tiles {
                let stripe_base = tile * TILE_T;
                for t in 0..TILE_T {
                    let eq_off = 8 * (stripe_base + t);
                    build_sum_table(
                        &eq_outer[eq_off..eq_off + 8],
                        &mut tables[t * 256..(t + 1) * 256],
                    );
                }
                let tables_ptr = tables.as_ptr() as *const u8;

                let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k + i_base) };
                for b in 0..n_block {
                    let i = b * BLOCK_K;
                    unsafe {
                        process_block_neon_single(
                            z_base,
                            k,
                            i,
                            tables_ptr,
                            out_slice.as_mut_ptr().add(i),
                        );
                    }
                }
            }
        });
    out
}

#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_oblock_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = NEON_TILE_T;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(
        n_log >= 3 + TILE_T.trailing_zeros() as usize,
        "need n_outer ≥ 8·TILE_T stripes"
    );
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_tiles = n_stripes / TILE_T;

    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);
    if useful == 0 {
        return vec![F128::ZERO; k];
    }

    let p = rayon::current_num_threads().max(1);
    let tiles_per_worker = n_tiles.div_ceil(p);
    let n_workers = n_tiles.div_ceil(tiles_per_worker);

    let mut partials = vec![F128::ZERO; n_workers * k];
    partials
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(w, partial)| {
            let tile_lo = w * tiles_per_worker;
            let tile_hi = ((w + 1) * tiles_per_worker).min(n_tiles);

            let mut tables = vec![F128::ZERO; TILE_T * 256];
            for tile in tile_lo..tile_hi {
                let stripe_base = tile * TILE_T;
                for t in 0..TILE_T {
                    let eq_off = 8 * (stripe_base + t);
                    build_sum_table(
                        &eq_outer[eq_off..eq_off + 8],
                        &mut tables[t * 256..(t + 1) * 256],
                    );
                }
                let tables_ptr = tables.as_ptr() as *const u8;
                let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k) };
                let mut bs = 0usize;
                while bs < useful {
                    unsafe {
                        process_block_neon_single(
                            z_base,
                            k,
                            bs,
                            tables_ptr,
                            partial.as_mut_ptr().add(bs),
                        );
                    }
                    bs += BLOCK_K;
                }
            }
        });

    let (first, rest) = partials.split_at_mut(k);
    first
        .par_chunks_mut(256)
        .enumerate()
        .for_each(|(ci, out_chunk)| {
            let col_off = ci * 256;
            let chunk_len = out_chunk.len();
            for w in 0..(n_workers - 1) {
                let src = &rest[w * k + col_off..w * k + col_off + chunk_len];
                crate::field::f128_slice::xor_assign(out_chunk, src);
            }
        });
    partials.truncate(k);
    partials
}

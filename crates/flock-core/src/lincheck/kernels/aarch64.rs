use super::super::{F128, NEON_TILE_T, build_sum_table};

/// Single-matrix partial fold with **tiled + NEON-register accumulators**.
/// Keeps `BLOCK_K = 8` accumulators in NEON registers across a `NEON_TILE_T`
/// stripe sweep — no per-byte accumulator LD/ST. Hand-rolled aarch64
/// intrinsics force the F128 XOR to a single `EOR.16B` and pin the 8 accs
/// in Q registers.
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

/// Padding-aware variant of [`partial_fold_packed_z_neon_single`]. Rounds
/// `useful_bits` up to a multiple of `BLOCK_K = 8` and processes only the
/// covered blocks; the trailing blocks (entirely padding) stay zero in the
/// accumulator. Any partially-useful boundary block is processed in full —
/// its padding bytes are zero, table[0] = 0, so they contribute nothing.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_single_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    single_padded_tiled::<NEON_TILE_T>(z_packed, m, k_log, useful_bits, eq_outer)
}

#[cfg(target_arch = "aarch64")]
fn single_padded_tiled<const TILE_T: usize>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

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
    // Cover only the blocks that touch useful bits. The boundary block
    // contains padding bytes which are 0 — table[0] = 0 → they contribute
    // nothing to the per-block XOR chain.
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
                // TILE_T × 256 F128 tables. L1 resident.
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
                            process_block_neon_single::<TILE_T>(
                                tile_bytes_ptr,
                                k,
                                bs,
                                tables_ptr,
                                out.as_mut_ptr().add(bs),
                            );
                        }
                    }
                }
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

/// Single-matrix NEON inner kernel — sweep `TILE_T` stripes of a stripe-tile
/// for one BLOCK_K=8 block of i_inner positions, keeping all 8 accumulators
/// in NEON Q-registers.
///
/// The 8 z index bytes for a stripe are consecutive, so they are fetched with
/// **one** 8-byte scalar load and shifted out of the register rather than with
/// eight `LDRB`s: the gather already issues one 128-bit table load per index,
/// and a second load per index would nearly double this kernel's load-port
/// pressure for data that is already in a register.
///
/// # Safety
/// - `row_ptr` must point to at least `(TILE_T - 1) * row_stride + bs + 8` bytes.
/// - `tables_ptr` must point to at least `TILE_T * 256 * 16` bytes.
/// - `out_ptr` must point to at least 8 F128 (128 bytes) of mutable storage.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn process_block_neon_single<const TILE_T: usize>(
    row_ptr: *const u8,
    row_stride: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    use std::arch::aarch64::*;

    let o = out_ptr as *mut u8;

    let mut a0 = vld1q_u8(o);
    let mut a1 = vld1q_u8(o.add(16));
    let mut a2 = vld1q_u8(o.add(32));
    let mut a3 = vld1q_u8(o.add(48));
    let mut a4 = vld1q_u8(o.add(64));
    let mut a5 = vld1q_u8(o.add(80));
    let mut a6 = vld1q_u8(o.add(96));
    let mut a7 = vld1q_u8(o.add(112));

    for t in 0..TILE_T {
        let stripe_ptr = row_ptr.add(t * row_stride + bs);
        let ta = tables_ptr.add(t * 256 * 16);

        // One unaligned 8-byte load replaces eight LDRBs.
        let w = (stripe_ptr as *const u64).read_unaligned();

        let i0 = (w & 0xff) as usize;
        let i1 = ((w >> 8) & 0xff) as usize;
        let i2 = ((w >> 16) & 0xff) as usize;
        let i3 = ((w >> 24) & 0xff) as usize;
        let i4 = ((w >> 32) & 0xff) as usize;
        let i5 = ((w >> 40) & 0xff) as usize;
        let i6 = ((w >> 48) & 0xff) as usize;
        let i7 = (w >> 56) as usize;

        a0 = veorq_u8(a0, vld1q_u8(ta.add(i0 * 16)));
        a1 = veorq_u8(a1, vld1q_u8(ta.add(i1 * 16)));
        a2 = veorq_u8(a2, vld1q_u8(ta.add(i2 * 16)));
        a3 = veorq_u8(a3, vld1q_u8(ta.add(i3 * 16)));
        a4 = veorq_u8(a4, vld1q_u8(ta.add(i4 * 16)));
        a5 = veorq_u8(a5, vld1q_u8(ta.add(i5 * 16)));
        a6 = veorq_u8(a6, vld1q_u8(ta.add(i6 * 16)));
        a7 = veorq_u8(a7, vld1q_u8(ta.add(i7 * 16)));
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

/// **i_inner-partitioned** NEON partial fold. Same result as
/// [`partial_fold_packed_z_neon_single_padded`] but parallelizes over the
/// **output** (`i_inner`) instead of over z stripes.
///
/// Why: the stripe-parallel kernel gives every worker its own full length-`k`
/// accumulator (2 MB at k = 2¹⁷). With P workers that's `P · 2 MB` of live
/// accumulators — past ~3 workers it exceeds L2, so each worker's accumulator
/// spills and gets re-streamed from **main memory** once per stripe-tile
/// (≈ `n_tiles · 2·k` F128 of memory traffic). Measured: scaling saturates at
/// ~5× on 10 cores (memory-bound), not ~10×.
///
/// Here the workers own **disjoint** slices of a single shared `out`, so the
/// total live accumulator is just `k` F128 = 2 MB — it stays L2-resident, never
/// re-streamed from memory, and there is **no final reduction**. Main-memory
/// traffic drops to one pass over z plus one write of `out`. Each worker still
/// uses the register-tiled inner kernel (8 accumulators across `TILE_T`
/// stripes); it just rebuilds the per-tile sum tables for its own slice (a few
/// % of redundant table-build XORs, far cheaper than the memory re-streaming).
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_iblock_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    iblock_padded_tiled::<NEON_TILE_T>(z_packed, m, k_log, useful_bits, eq_outer)
}

#[cfg(target_arch = "aarch64")]
fn iblock_padded_tiled<const TILE_T: usize>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

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

    // Only i_inner < useful_bits can be nonzero (padded rows fold to 0). Round
    // up to BLOCK_K; the boundary block's padding bytes are 0 ⇒ table[0] = 0 ⇒
    // contribute nothing. Rows [useful, k) stay zero from the vec init.
    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);

    let mut out = vec![F128::ZERO; k];
    if useful == 0 {
        return out;
    }

    // Partition the useful i_inner range across workers. Each chunk independently
    // rebuilds the per-tile sum tables, so chunk count drives redundant table
    // work — work that does NOT scale with cores and dominates the residual at
    // m=30 (≈3.3 ms/core at 3 chunks/worker). On the homogeneous pinned P-core
    // pool, 1 chunk/worker is perfectly balanced (par_chunks_mut → exactly `p`
    // equal chunks) and cuts that residual ~3×: partial-fold MT 6.2 → 4.5 ms,
    // no ST change. Oversubscribe (3/worker) only when the pool is larger than
    // the P-core count — i.e. likely includes slower E-cores — so rayon can
    // steal from a straggler. Each chunk is a BLOCK_K multiple.
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
            // TILE_T × 256 F128 tables, L1-resident, rebuilt per tile.
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
                // Base of this (tile, i_base): process_block reads
                // z_base[t·k + bs] = z[(stripe_base+t)·k + i_base + bs].
                let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k + i_base) };
                for b in 0..n_block {
                    let i = b * BLOCK_K;
                    unsafe {
                        process_block_neon_single::<TILE_T>(
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

/// Outer(tile)-partitioned sibling of [`partial_fold_packed_z_neon_iblock_padded`]
/// — same result, parallelized to remove the redundant per-worker sum-table
/// rebuilds that cap iblock's multicore scaling. **This is the default fold**
/// (`partial_fold_packed_z_best`); set [`FOLD_IBLOCK`] to fall back to iblock.
///
/// iblock partitions the length-k **output** across workers, so every worker
/// rebuilds **all** `n_stripes` tile tables — table work is done `p`× and does not
/// shrink with cores (≈44 % of the MT wall at m=32). Here we partition the **tiles**
/// (outer/stripe dim): each worker owns a contiguous tile band, builds each of its
/// tile tables exactly **once**, folds them into a private length-k partial, and the
/// `p` partials are XOR-reduced at the end. The partial is the full length-k
/// (256 KB at k_log=14 ⇒ spills L1 to L2), but the register-tiled inner kernel keeps
/// 8 F128 accumulators in NEON registers, so the L2 traffic is mild — measured ≈2 %
/// ST cost at m=32, none at m=30 — and far cheaper than iblock's redundant tables:
/// the fold scales ~8.5× vs iblock's ~6.5× on 10 P-cores at m=32, and the margin
/// grows with the outer dim (the redundant-table cost it removes is ∝ `n_stripes`).
///
/// # Safety / preconditions: identical to the iblock kernel.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_oblock_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    oblock_padded_tiled::<NEON_TILE_T>(z_packed, m, k_log, useful_bits, eq_outer)
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn oblock_padded_tiled<const TILE_T: usize>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

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

    // Only i_inner < useful_bits can be nonzero (padded rows fold to 0). Rounded
    // up to BLOCK_K; columns [useful, k) stay zero from the partial init.
    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);
    if useful == 0 {
        return vec![F128::ZERO; k];
    }

    // One private length-k partial per worker; workers own contiguous tile bands,
    // so each tile's sum-tables are built exactly once (not once per worker).
    let p = rayon::current_num_threads().max(1);
    let tiles_per_worker = n_tiles.div_ceil(p);
    let n_workers = n_tiles.div_ceil(tiles_per_worker); // ≤ p, every band non-empty

    let mut partials = vec![F128::ZERO; n_workers * k];
    partials
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(w, partial)| {
            let tile_lo = w * tiles_per_worker;
            let tile_hi = ((w + 1) * tiles_per_worker).min(n_tiles);
            // TILE_T × 256 F128 tables, L1-resident, built once per tile.
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
                        process_block_neon_single::<TILE_T>(
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

    // XOR-reduce the per-worker partials in ONE parallel pass over column bands:
    // each worker owns a band of the output and XORs it across all `n_workers`
    // partials, so the band lands in registers/L1 once and is written once.
    //
    // The obvious alternative — fold one partial at a time with a parallel
    // `zip` per partial — costs `n_workers - 1` separate parallel regions, each
    // a full read-modify-write of the 256 KB accumulator plus a rayon barrier.
    // That is `n_workers - 1` accumulator reads and writes (≈2.5× the traffic
    // here) and 9 barriers instead of 1 at `n_workers = 10`; measured ≈0.7 ms
    // of the fold at m=32, k_log=14.
    let band = k.div_ceil(rayon::current_num_threads().max(1)).max(1024);
    let mut out = vec![F128::ZERO; k];
    out.par_chunks_mut(band).enumerate().for_each(|(bi, dst)| {
        let lo = bi * band;
        for w in 0..n_workers {
            let src = &partials[w * k + lo..w * k + lo + dst.len()];
            for (o, s) in dst.iter_mut().zip(src.iter()) {
                *o += *s;
            }
        }
    });
    out
}

/// Row-major sibling of the outer-partitioned stripe fold. Each worker reads
/// 64 consecutive canonical witness blocks per tile, transposes one 64-bit
/// inner word into a 512-byte stack buffer, and immediately feeds the existing
/// eight-register NEON lookup kernel.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_row_major_neon_oblock_padded(
    z_row_major: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use crate::bits::transpose_8_u64s_to_64_bytes;
    use rayon::prelude::*;

    const TILE_T: usize = NEON_TILE_T;
    const BLOCKS_PER_STRIPE: usize = 8;
    const BLOCKS_PER_TILE: usize = TILE_T * BLOCKS_PER_STRIPE;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_row_major.len(), 1usize << (m - 7));
    assert_eq!(eq_outer.len(), n_outer);
    assert!(k_log >= 7);
    assert!(useful_bits <= k);
    let n_stripes = n_outer / BLOCKS_PER_STRIPE;
    assert_eq!(n_stripes % TILE_T, 0);
    let n_tiles = n_stripes / TILE_T;
    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);
    if useful == 0 {
        return vec![F128::ZERO; k];
    }

    let words_per_block = k / 64;
    let useful_words = useful_bits.div_ceil(64);
    // SAFETY: F128 is repr(C) with two consecutive u64 halves.
    let z_words = unsafe {
        std::slice::from_raw_parts(z_row_major.as_ptr() as *const u64, z_row_major.len() * 2)
    };

    let p = rayon::current_num_threads().max(1);
    let tiles_per_worker = n_tiles.div_ceil(p);
    let n_workers = n_tiles.div_ceil(tiles_per_worker);
    let mut partials = vec![F128::ZERO; n_workers * k];
    partials
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(worker, partial)| {
            let tile_lo = worker * tiles_per_worker;
            let tile_hi = ((worker + 1) * tiles_per_worker).min(n_tiles);
            let mut tables = vec![F128::ZERO; TILE_T * 256];
            let mut transposed = [[0u8; 64]; TILE_T];

            for tile in tile_lo..tile_hi {
                for t in 0..TILE_T {
                    let eq_off = tile * BLOCKS_PER_TILE + t * BLOCKS_PER_STRIPE;
                    build_sum_table(
                        &eq_outer[eq_off..eq_off + BLOCKS_PER_STRIPE],
                        &mut tables[t * 256..(t + 1) * 256],
                    );
                }
                let tables_ptr = tables.as_ptr() as *const u8;
                let tile_block = tile * BLOCKS_PER_TILE;

                for word_idx in 0..useful_words {
                    for t in 0..TILE_T {
                        let stripe_block = tile_block + t * BLOCKS_PER_STRIPE;
                        let lanes: [u64; BLOCKS_PER_STRIPE] = std::array::from_fn(|lane| {
                            z_words[(stripe_block + lane) * words_per_block + word_idx]
                        });
                        transpose_8_u64s_to_64_bytes(&lanes, &mut transposed[t]);
                    }

                    let inner_base = word_idx * 64;
                    let n_inner = (useful - inner_base).min(64);
                    let mut bs = 0usize;
                    while bs < n_inner {
                        unsafe {
                            process_block_neon_single::<TILE_T>(
                                transposed.as_ptr() as *const u8,
                                64,
                                bs,
                                tables_ptr,
                                partial.as_mut_ptr().add(inner_base + bs),
                            );
                        }
                        bs += BLOCK_K;
                    }
                }
            }
        });

    let band = k.div_ceil(rayon::current_num_threads().max(1)).max(1024);
    let mut out = vec![F128::ZERO; k];
    out.par_chunks_mut(band)
        .enumerate()
        .for_each(|(band_idx, dst)| {
            let lo = band_idx * band;
            for worker in 0..n_workers {
                let src = &partials[worker * k + lo..worker * k + lo + dst.len()];
                for (o, s) in dst.iter_mut().zip(src.iter()) {
                    *o += *s;
                }
            }
        });
    out
}

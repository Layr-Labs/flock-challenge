use super::super::{F128, build_sum_table};

const NEON_TILE_T: usize = 8;

/// Runtime gate for the all-NEON q-form lincheck kernels (packed-index
/// wavefront leaf + register-resident gather/transpose).
/// `FLOCK_NO_LINCHECK_QFORM=1` is the kill switch for local A/B diagnostics;
/// the ranked worker's cleared environment never sets it. Read once per
/// process.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn lincheck_qform_enabled() -> bool {
    static QFORM: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LINCHECK_QFORM").is_none());
    *QFORM
}

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
                // TILE_T × 256 F128 = 32 KB tables. L1 resident.
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
                // Suppress unused variable warning when not aarch64
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

/// Single-matrix NEON inner kernel — sweep TILE_T=8 stripes of a stripe-tile
/// for one BLOCK_K=8 block of i_inner positions, keeping all 8 accumulators
/// in NEON Q-registers.
///
/// # Safety
/// - `tile_bytes_ptr` must point to at least `TILE_T * k` bytes.
/// - `tables_ptr` must point to at least `TILE_T * 256 * 16` bytes.
/// - `out_ptr` must point to at least 8 F128 (128 bytes) of mutable storage.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
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

    for t in 0..TILE_T {
        let stripe_ptr = tile_bytes_ptr.add(t * k + bs);
        let ta = tables_ptr.add(t * 256 * 16);

        let i0 = *stripe_ptr as usize;
        let i1 = *stripe_ptr.add(1) as usize;
        let i2 = *stripe_ptr.add(2) as usize;
        let i3 = *stripe_ptr.add(3) as usize;
        let i4 = *stripe_ptr.add(4) as usize;
        let i5 = *stripe_ptr.add(5) as usize;
        let i6 = *stripe_ptr.add(6) as usize;
        let i7 = *stripe_ptr.add(7) as usize;

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
            // TILE_T × 256 F128 = 32 KB tables, L1-resident, rebuilt per tile.
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
            // TILE_T × 256 F128 = 32 KB tables, L1-resident, built once per tile.
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

    // XOR-reduce the per-worker partials: parallel over columns, sequential over
    // workers so each 256 KB partial is streamed once (cache-friendly).
    let (first, rest) = partials.split_at(k);
    let mut out = first.to_vec();
    for chunk in rest.chunks(k) {
        out.par_iter_mut()
            .zip(chunk.par_iter())
            .for_each(|(o, s)| *o += *s);
    }
    out
}

// ---------------------------------------------------------------------------
// Two-stream software wavefront for the **block-major** tile sweep.
//
// The block-major fold's leaf (`partial_fold_packed_z_block_major_padded_with_tables`,
// scalar form) is, per output column `b`, a serial depth-8 XOR chain where every
// link is gated on an L1 table load (`ldrb` index → `ldr q` table entry →
// `eor.16b`) — the same latency shape as the zerocheck round-1 kernel that the
// `shift_reduce_inner_ab_fused_neon_x2` wavefront fixed. LLVM compiles the
// scalar loop to ONE live accumulator chain per column plus a
// `cmp`/`b.eq` branch per link (runtime `tile_stripes`), so the core idles on
// load latency.
//
// Treatment: process TWO adjacent 8-column blocks of the 128-column output
// group per call — two fully independent streams with their own 8 NEON
// accumulators each. Both streams share the tile's 32 KiB of sum tables and
// the 1 KiB transposed index buffer (no extra L1 footprint) but write DISJOINT
// `partial` slots. Segments are one stream's stripe-row (8 index loads,
// 8 table loads, 8 EORs); `pin_accs` — an empty `asm!` fence taking the
// segment's accumulators as `inout(vreg)` — bounds each segment so LLVM
// cannot merge the 16 lookup streams into one region and spill.
//
// Register budget: 16 accumulators live across the body + one segment's 8
// short-lived table loads + pointers ≤ ~26 of 32 q-registers — no spills.
//
// Bit-identity: each accumulator XORs the exact same table entries in the
// exact same stripe order (t ascending) as the scalar loop; the two streams
// share no state. Output is bit-identical, not merely XOR-reordered.
// ---------------------------------------------------------------------------

/// One stripe-row segment of one stream: 8 index bytes → 8 table lookups →
/// 8 accumulator EORs.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fold_seg8(
    idx_ptr: *const u8,
    table_ptr: *const u8,
    a0: &mut core::arch::aarch64::uint8x16_t,
    a1: &mut core::arch::aarch64::uint8x16_t,
    a2: &mut core::arch::aarch64::uint8x16_t,
    a3: &mut core::arch::aarch64::uint8x16_t,
    a4: &mut core::arch::aarch64::uint8x16_t,
    a5: &mut core::arch::aarch64::uint8x16_t,
    a6: &mut core::arch::aarch64::uint8x16_t,
    a7: &mut core::arch::aarch64::uint8x16_t,
) {
    use std::arch::aarch64::*;
    unsafe {
        let i0 = *idx_ptr as usize;
        let i1 = *idx_ptr.add(1) as usize;
        let i2 = *idx_ptr.add(2) as usize;
        let i3 = *idx_ptr.add(3) as usize;
        let i4 = *idx_ptr.add(4) as usize;
        let i5 = *idx_ptr.add(5) as usize;
        let i6 = *idx_ptr.add(6) as usize;
        let i7 = *idx_ptr.add(7) as usize;
        *a0 = veorq_u8(*a0, vld1q_u8(table_ptr.add(i0 * 16)));
        *a1 = veorq_u8(*a1, vld1q_u8(table_ptr.add(i1 * 16)));
        *a2 = veorq_u8(*a2, vld1q_u8(table_ptr.add(i2 * 16)));
        *a3 = veorq_u8(*a3, vld1q_u8(table_ptr.add(i3 * 16)));
        *a4 = veorq_u8(*a4, vld1q_u8(table_ptr.add(i4 * 16)));
        *a5 = veorq_u8(*a5, vld1q_u8(table_ptr.add(i5 * 16)));
        *a6 = veorq_u8(*a6, vld1q_u8(table_ptr.add(i6 * 16)));
        *a7 = veorq_u8(*a7, vld1q_u8(table_ptr.add(i7 * 16)));
    }
}

/// Compile-time-only scheduling fence: zero instructions emitted, but the
/// segment's 8 accumulators are `inout(vreg)` operands and memory is
/// clobbered (default `asm!` behavior), so no table/index load may be hoisted
/// or sunk across a segment boundary and each segment schedules as a unit.
/// See the zerocheck round-1 wavefront for the calibration of this pattern.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn pin_accs(
    a0: &mut core::arch::aarch64::uint8x16_t,
    a1: &mut core::arch::aarch64::uint8x16_t,
    a2: &mut core::arch::aarch64::uint8x16_t,
    a3: &mut core::arch::aarch64::uint8x16_t,
    a4: &mut core::arch::aarch64::uint8x16_t,
    a5: &mut core::arch::aarch64::uint8x16_t,
    a6: &mut core::arch::aarch64::uint8x16_t,
    a7: &mut core::arch::aarch64::uint8x16_t,
) {
    unsafe {
        core::arch::asm!(
            "/* pin {0:v} {1:v} {2:v} {3:v} {4:v} {5:v} {6:v} {7:v} */",
            inout(vreg) * a0,
            inout(vreg) * a1,
            inout(vreg) * a2,
            inout(vreg) * a3,
            inout(vreg) * a4,
            inout(vreg) * a5,
            inout(vreg) * a6,
            inout(vreg) * a7,
            options(nostack, preserves_flags)
        );
    }
}

/// Two-stream leaf: fold all `NEON_TILE_T = 8` stripe tables into two
/// adjacent 8-column accumulator blocks.
///
/// # Safety
/// - `bytes0` must point at the stream-0 index bytes inside a transposed tile
///   buffer with row stride 128; rows `t ∈ 0..8` and columns `0..16`
///   (stream 0 at `+0..8`, stream 1 at `+8..16`) must be in bounds.
/// - `tables_ptr` must point to `8 × 256` F128 sum tables (32 KiB).
/// - `acc_ptr` must point to 16 consecutive mutable F128 slots
///   (stream 0 = `0..8`, stream 1 = `8..16`).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fold_block_major_bpair_x2_neon(
    bytes0: *const u8,
    tables_ptr: *const u8,
    acc_ptr: *mut F128,
) {
    use std::arch::aarch64::*;

    let o = acc_ptr as *mut u8;
    // Stream 0 accumulators.
    let mut a0 = vld1q_u8(o);
    let mut a1 = vld1q_u8(o.add(16));
    let mut a2 = vld1q_u8(o.add(32));
    let mut a3 = vld1q_u8(o.add(48));
    let mut a4 = vld1q_u8(o.add(64));
    let mut a5 = vld1q_u8(o.add(80));
    let mut a6 = vld1q_u8(o.add(96));
    let mut a7 = vld1q_u8(o.add(112));
    // Stream 1 accumulators.
    let mut c0 = vld1q_u8(o.add(128));
    let mut c1 = vld1q_u8(o.add(144));
    let mut c2 = vld1q_u8(o.add(160));
    let mut c3 = vld1q_u8(o.add(176));
    let mut c4 = vld1q_u8(o.add(192));
    let mut c5 = vld1q_u8(o.add(208));
    let mut c6 = vld1q_u8(o.add(224));
    let mut c7 = vld1q_u8(o.add(240));

    for t in 0..NEON_TILE_T {
        let idx = bytes0.add(t * 128);
        let ta = tables_ptr.add(t * 256 * 16);
        // Stream 0 stripe-row, then stream 1 stripe-row: two independent
        // instruction bundles back to back for the OoO engine to overlap.
        fold_seg8(
            idx, ta, &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
        );
        pin_accs(
            &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
        );
        fold_seg8(
            idx.add(8),
            ta,
            &mut c0,
            &mut c1,
            &mut c2,
            &mut c3,
            &mut c4,
            &mut c5,
            &mut c6,
            &mut c7,
        );
        pin_accs(
            &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut c7,
        );
    }

    vst1q_u8(o, a0);
    vst1q_u8(o.add(16), a1);
    vst1q_u8(o.add(32), a2);
    vst1q_u8(o.add(48), a3);
    vst1q_u8(o.add(64), a4);
    vst1q_u8(o.add(80), a5);
    vst1q_u8(o.add(96), a6);
    vst1q_u8(o.add(112), a7);
    vst1q_u8(o.add(128), c0);
    vst1q_u8(o.add(144), c1);
    vst1q_u8(o.add(160), c2);
    vst1q_u8(o.add(176), c3);
    vst1q_u8(o.add(192), c4);
    vst1q_u8(o.add(208), c5);
    vst1q_u8(o.add(224), c6);
    vst1q_u8(o.add(240), c7);
}

/// q-form sibling of [`fold_seg8`]: same 8 lookups + 8 EORs, but the 8 index
/// bytes arrive as ONE unaligned `u64` load + 8 bitfield extracts instead of
/// 8 `ldrb`s, and the table pointer is used as an opaque base so the lookup
/// compiles to a single `ldr q, [ta, idx, lsl #4]` (the legacy leaf pays two
/// dependent `add`s per lookup for LLVM's strength-reduced running offset).
/// Per segment: 25 instructions / 9 load-port slots vs the legacy 40 / 16.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fold_seg8_q(
    idx_ptr: *const u8,
    table_ptr: *const u8,
    a0: &mut core::arch::aarch64::uint8x16_t,
    a1: &mut core::arch::aarch64::uint8x16_t,
    a2: &mut core::arch::aarch64::uint8x16_t,
    a3: &mut core::arch::aarch64::uint8x16_t,
    a4: &mut core::arch::aarch64::uint8x16_t,
    a5: &mut core::arch::aarch64::uint8x16_t,
    a6: &mut core::arch::aarch64::uint8x16_t,
    a7: &mut core::arch::aarch64::uint8x16_t,
) {
    use std::arch::aarch64::*;
    unsafe {
        // Little-endian: byte i of the row is bits [8i, 8i+8) of the u64.
        let bits = (idx_ptr as *const u64).read_unaligned();
        let i0 = (bits & 0xff) as usize;
        let i1 = ((bits >> 8) & 0xff) as usize;
        let i2 = ((bits >> 16) & 0xff) as usize;
        let i3 = ((bits >> 24) & 0xff) as usize;
        let i4 = ((bits >> 32) & 0xff) as usize;
        let i5 = ((bits >> 40) & 0xff) as usize;
        let i6 = ((bits >> 48) & 0xff) as usize;
        let i7 = (bits >> 56) as usize;
        *a0 = veorq_u8(*a0, vld1q_u8(table_ptr.add(i0 << 4)));
        *a1 = veorq_u8(*a1, vld1q_u8(table_ptr.add(i1 << 4)));
        *a2 = veorq_u8(*a2, vld1q_u8(table_ptr.add(i2 << 4)));
        *a3 = veorq_u8(*a3, vld1q_u8(table_ptr.add(i3 << 4)));
        *a4 = veorq_u8(*a4, vld1q_u8(table_ptr.add(i4 << 4)));
        *a5 = veorq_u8(*a5, vld1q_u8(table_ptr.add(i5 << 4)));
        *a6 = veorq_u8(*a6, vld1q_u8(table_ptr.add(i6 << 4)));
        *a7 = veorq_u8(*a7, vld1q_u8(table_ptr.add(i7 << 4)));
    }
}

/// q-form sibling of [`fold_block_major_bpair_x2_neon`] — identical contract,
/// identical per-accumulator XOR order (bit-identical output), but each
/// stripe's table base is pinned opaque so lookups use one-instruction
/// `[base, idx, lsl #4]` addressing, and index bytes load 8-at-a-time.
/// Gated by [`lincheck_qform_enabled`].
///
/// # Safety
/// Same preconditions as [`fold_block_major_bpair_x2_neon`].
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fold_block_major_bpair_x2_neon_q(
    bytes0: *const u8,
    tables_ptr: *const u8,
    acc_ptr: *mut F128,
) {
    use std::arch::aarch64::*;

    let o = acc_ptr as *mut u8;
    // Stream 0 accumulators.
    let mut a0 = vld1q_u8(o);
    let mut a1 = vld1q_u8(o.add(16));
    let mut a2 = vld1q_u8(o.add(32));
    let mut a3 = vld1q_u8(o.add(48));
    let mut a4 = vld1q_u8(o.add(64));
    let mut a5 = vld1q_u8(o.add(80));
    let mut a6 = vld1q_u8(o.add(96));
    let mut a7 = vld1q_u8(o.add(112));
    // Stream 1 accumulators.
    let mut c0 = vld1q_u8(o.add(128));
    let mut c1 = vld1q_u8(o.add(144));
    let mut c2 = vld1q_u8(o.add(160));
    let mut c3 = vld1q_u8(o.add(176));
    let mut c4 = vld1q_u8(o.add(192));
    let mut c5 = vld1q_u8(o.add(208));
    let mut c6 = vld1q_u8(o.add(224));
    let mut c7 = vld1q_u8(o.add(240));

    for t in 0..NEON_TILE_T {
        let idx = bytes0.add(t * 128);
        let mut ta = tables_ptr.add(t * 256 * 16);
        // Opaque pointer pin: without it LLVM strength-reduces `t * 4096`
        // into a running offset and spends two dependent `add`s per lookup.
        core::arch::asm!(
            "/* {0} */",
            inout(reg) ta,
            options(nomem, nostack, preserves_flags)
        );
        fold_seg8_q(
            idx, ta, &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
        );
        pin_accs(
            &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut a6, &mut a7,
        );
        fold_seg8_q(
            idx.add(8),
            ta,
            &mut c0,
            &mut c1,
            &mut c2,
            &mut c3,
            &mut c4,
            &mut c5,
            &mut c6,
            &mut c7,
        );
        pin_accs(
            &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut c7,
        );
    }

    vst1q_u8(o, a0);
    vst1q_u8(o.add(16), a1);
    vst1q_u8(o.add(32), a2);
    vst1q_u8(o.add(48), a3);
    vst1q_u8(o.add(64), a4);
    vst1q_u8(o.add(80), a5);
    vst1q_u8(o.add(96), a6);
    vst1q_u8(o.add(112), a7);
    vst1q_u8(o.add(128), c0);
    vst1q_u8(o.add(144), c1);
    vst1q_u8(o.add(160), c2);
    vst1q_u8(o.add(176), c3);
    vst1q_u8(o.add(192), c4);
    vst1q_u8(o.add(208), c5);
    vst1q_u8(o.add(224), c6);
    vst1q_u8(o.add(240), c7);
}

/// `y ^ t ^ (t << S)` on a 2×u64 lane pair, forced to stay in NEON.
///
/// LLVM proves `t` and `t << S` bit-disjoint (masked bit-swap rounds) and
/// "optimizes" the vector XOR into scalar `orr reg, reg, reg, lsl #S` — at
/// the cost of `fmov`/`mov.d` round-trips per 64-bit half in both directions
/// (measured in the legacy transpose codegen). Three asm instructions keep
/// the whole round vector-resident.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_shl<const S: i32>(
    y: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let mut y = y;
    unsafe {
        core::arch::asm!(
            "eor {y:v}.16b, {y:v}.16b, {t:v}.16b",
            "shl {tmp:v}.2d, {t:v}.2d, #{s}",
            "eor {y:v}.16b, {y:v}.16b, {tmp:v}.16b",
            y = inout(vreg) y,
            t = in(vreg) t,
            tmp = out(vreg) _,
            s = const S,
            options(nomem, nostack, preserves_flags, pure)
        );
    }
    y
}

/// One Hacker's-Delight bit-swap round (distance `S`, mask `m`) over four
/// 128-bit lanes, all vector-resident.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_round_4<const S: i32>(
    y: &mut [core::arch::aarch64::uint64x2_t; 4],
    m: core::arch::aarch64::uint64x2_t,
) {
    use std::arch::aarch64::*;
    unsafe {
        for yi in y.iter_mut() {
            let t = vandq_u64(veorq_u64(*yi, vshrq_n_u64::<S>(*yi)), m);
            *yi = xor_shl::<S>(*yi, t);
        }
    }
}

/// All-NEON gather + bit-transpose for one full 8-stripe tile chunk.
///
/// Replaces, for each stripe row `t ∈ 0..8`, the scalar-formed
/// `lanes: [F128; 8]` gather + [`transpose_8_f128s_to_128_bytes`] pair.
/// The legacy codegen loads each witness lane through GPR `ldp`, assembles
/// vectors with a `fmov`+`mov.d` per half, re-scalarizes the bit-swap rounds
/// through GPR `orr ..., lsl` (two cross-domain moves each way per lane), and
/// pays eight slice bounds checks per row. Here the eight lanes load straight
/// into q registers, `uzp1`/`uzp2` split lo/hi halves, `tbl` + three masked
/// swap rounds transpose entirely in NEON, and rows store with `stp q`.
///
/// Output layout is byte-identical to the legacy path: row `t` at
/// `out + 128·t`, low u64 table in bytes `0..64`, high in `64..128`.
///
/// # Safety
/// - Lane `(t, r)` is read at `src + (8·t + r) · stride` (F128 elements);
///   all 64 such positions must be in bounds.
/// - `out` must point to `8 × 128` writable bytes.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn gather_transpose_tile_neon(src: *const F128, stride: usize, out: *mut u8) {
    use std::arch::aarch64::*;

    // vqtbl4q indexes bringing byte c of every lane into contiguous 8-byte
    // runs (byte-chunk c of the 8×8 byte matrix), two chunks per Q register.
    // Identical to the zerocheck `bit_transpose_64bytes_neon` stage-1 tables.
    const IDX0: [u8; 16] = [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57];
    const IDX1: [u8; 16] = [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59];
    const IDX2: [u8; 16] = [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61];
    const IDX3: [u8; 16] = [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63];

    let idx0 = vld1q_u8(IDX0.as_ptr());
    let idx1 = vld1q_u8(IDX1.as_ptr());
    let idx2 = vld1q_u8(IDX2.as_ptr());
    let idx3 = vld1q_u8(IDX3.as_ptr());
    let m1 = vdupq_n_u64(0x00AA00AA00AA00AA);
    let m2 = vdupq_n_u64(0x0000CCCC0000CCCC);
    let m3 = vdupq_n_u64(0x00000000F0F0F0F0);

    let stride_bytes = stride * 16;
    let mut p = src as *const u8;
    let mut row = out;
    for _t in 0..NEON_TILE_T {
        let w0 = vreinterpretq_u64_u8(vld1q_u8(p));
        let w1 = vreinterpretq_u64_u8(vld1q_u8(p.add(stride_bytes)));
        let w2 = vreinterpretq_u64_u8(vld1q_u8(p.add(2 * stride_bytes)));
        let w3 = vreinterpretq_u64_u8(vld1q_u8(p.add(3 * stride_bytes)));
        let w4 = vreinterpretq_u64_u8(vld1q_u8(p.add(4 * stride_bytes)));
        let w5 = vreinterpretq_u64_u8(vld1q_u8(p.add(5 * stride_bytes)));
        let w6 = vreinterpretq_u64_u8(vld1q_u8(p.add(6 * stride_bytes)));
        let w7 = vreinterpretq_u64_u8(vld1q_u8(p.add(7 * stride_bytes)));

        // Prefetch each lane two cache lines ahead (256 B = 16 chunk-steps).
        // The sweep advances every lane by 16 B per (tile, chunk) call across
        // 64 concurrent 2 KiB-strided streams — beyond what the hardware
        // prefetcher tracks, so without this every 8th touch per lane stalls
        // on DRAM (probe: ~0.3–0.5 ms of lincheck::prove at m=32; 256 B lead
        // measured best of {128, 256, 512}). Redundant issues are absorbed by
        // the LSU.
        for r in 0..8 {
            core::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) p.add(r * stride_bytes + 256),
                options(nomem, nostack, preserves_flags)
            );
        }

        // Split lo/hi u64 halves: lo table = lo0‖lo1 … lo6‖lo7 (the byte
        // image of the legacy `[u64; 8]` lo array), hi table likewise.
        let lo = uint8x16x4_t(
            vreinterpretq_u8_u64(vuzp1q_u64(w0, w1)),
            vreinterpretq_u8_u64(vuzp1q_u64(w2, w3)),
            vreinterpretq_u8_u64(vuzp1q_u64(w4, w5)),
            vreinterpretq_u8_u64(vuzp1q_u64(w6, w7)),
        );
        let hi = uint8x16x4_t(
            vreinterpretq_u8_u64(vuzp2q_u64(w0, w1)),
            vreinterpretq_u8_u64(vuzp2q_u64(w2, w3)),
            vreinterpretq_u8_u64(vuzp2q_u64(w4, w5)),
            vreinterpretq_u8_u64(vuzp2q_u64(w6, w7)),
        );

        let mut ylo = [
            vreinterpretq_u64_u8(vqtbl4q_u8(lo, idx0)),
            vreinterpretq_u64_u8(vqtbl4q_u8(lo, idx1)),
            vreinterpretq_u64_u8(vqtbl4q_u8(lo, idx2)),
            vreinterpretq_u64_u8(vqtbl4q_u8(lo, idx3)),
        ];
        let mut yhi = [
            vreinterpretq_u64_u8(vqtbl4q_u8(hi, idx0)),
            vreinterpretq_u64_u8(vqtbl4q_u8(hi, idx1)),
            vreinterpretq_u64_u8(vqtbl4q_u8(hi, idx2)),
            vreinterpretq_u64_u8(vqtbl4q_u8(hi, idx3)),
        ];
        transpose_round_4::<7>(&mut ylo, m1);
        transpose_round_4::<7>(&mut yhi, m1);
        transpose_round_4::<14>(&mut ylo, m2);
        transpose_round_4::<14>(&mut yhi, m2);
        transpose_round_4::<28>(&mut ylo, m3);
        transpose_round_4::<28>(&mut yhi, m3);

        vst1q_u8(row, vreinterpretq_u8_u64(ylo[0]));
        vst1q_u8(row.add(16), vreinterpretq_u8_u64(ylo[1]));
        vst1q_u8(row.add(32), vreinterpretq_u8_u64(ylo[2]));
        vst1q_u8(row.add(48), vreinterpretq_u8_u64(ylo[3]));
        vst1q_u8(row.add(64), vreinterpretq_u8_u64(yhi[0]));
        vst1q_u8(row.add(80), vreinterpretq_u8_u64(yhi[1]));
        vst1q_u8(row.add(96), vreinterpretq_u8_u64(yhi[2]));
        vst1q_u8(row.add(112), vreinterpretq_u8_u64(yhi[3]));

        p = p.add(8 * stride_bytes);
        row = row.add(128);
    }
}

/// NEON entry for one (tile, 128-column chunk) of the block-major sweep.
/// Requires a FULL tile (`tile_stripes == 8`); the caller falls back to the
/// scalar loop otherwise. Folds the leading `8·⌊chunk_bits/8⌋` columns —
/// paired 8-column blocks through the two-stream wavefront leaf, one odd
/// trailing full block through the single-stream register-tiled leaf — and
/// returns how many columns were folded. The caller finishes `chunk_bits % 8`
/// columns in scalar.
///
/// `transposed` is the 8×128 index buffer (row stride 128), `tables` the
/// 8×256 F128 sum tables, `partial` the output group at `inner_base`
/// (`partial[b]` accumulates column `b`).
#[cfg(target_arch = "aarch64")]
pub fn fold_block_major_chunk_neon_x2(
    transposed: &[u8],
    tables: &[F128],
    partial: &mut [F128],
    chunk_bits: usize,
) -> usize {
    assert_eq!(transposed.len(), NEON_TILE_T * 128);
    assert_eq!(tables.len(), NEON_TILE_T * 256);
    assert!(chunk_bits <= 128);
    assert!(partial.len() >= chunk_bits);

    let full_blocks = chunk_bits / 8;
    let pairs = full_blocks / 2;
    let tables_ptr = tables.as_ptr() as *const u8;
    let qform = lincheck_qform_enabled();
    for p in 0..pairs {
        let b0 = p * 16;
        unsafe {
            if qform {
                fold_block_major_bpair_x2_neon_q(
                    transposed.as_ptr().add(b0),
                    tables_ptr,
                    partial.as_mut_ptr().add(b0),
                );
            } else {
                fold_block_major_bpair_x2_neon(
                    transposed.as_ptr().add(b0),
                    tables_ptr,
                    partial.as_mut_ptr().add(b0),
                );
            }
        }
    }
    if full_blocks % 2 == 1 {
        let b0 = pairs * 16;
        unsafe {
            process_block_neon_single(
                transposed.as_ptr(),
                128,
                b0,
                tables_ptr,
                partial.as_mut_ptr().add(b0),
            );
        }
    }
    full_blocks * 8
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            // splitmix64
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

    /// The q-form wavefront leaf must be BIT-identical to the legacy leaf on
    /// random tables / indices / non-zero starting accumulators.
    #[test]
    fn bpair_x2_q_leaf_matches_legacy() {
        let mut rng = Rng(0x1EAF_0001);
        for round in 0..32 {
            let transposed: Vec<u8> = (0..NEON_TILE_T * 128)
                .map(|_| rng.next_u64() as u8)
                .collect();
            let tables: Vec<F128> = (0..NEON_TILE_T * 256).map(|_| rng.f128()).collect();
            let accs: Vec<F128> = (0..16).map(|_| rng.f128()).collect();
            let tables_ptr = tables.as_ptr() as *const u8;
            // Both leaves see the same pair offset within the row.
            let b0 = (round % 8) * 16;

            let mut legacy = accs.clone();
            let mut qform = accs.clone();
            unsafe {
                fold_block_major_bpair_x2_neon(
                    transposed.as_ptr().add(b0),
                    tables_ptr,
                    legacy.as_mut_ptr(),
                );
                fold_block_major_bpair_x2_neon_q(
                    transposed.as_ptr().add(b0),
                    tables_ptr,
                    qform.as_mut_ptr(),
                );
            }
            assert_eq!(legacy, qform, "round={round} b0={b0}");
        }
    }

    /// The all-NEON gather+transpose must reproduce the scalar definition:
    /// output row t, byte b, bit r = bit b of lane (t, r).
    #[test]
    fn gather_transpose_tile_matches_scalar() {
        let mut rng = Rng(0x72A5_0002);
        for &stride in &[1usize, 2, 7, 128] {
            // Lane (t, r) lives at src[(8t + r) * stride].
            let src: Vec<F128> = (0..64 * stride).map(|_| rng.f128()).collect();
            let mut got = [0u8; NEON_TILE_T * 128];
            unsafe {
                gather_transpose_tile_neon(src.as_ptr(), stride, got.as_mut_ptr());
            }
            for t in 0..NEON_TILE_T {
                for b in 0..128 {
                    let mut want = 0u8;
                    for r in 0..8 {
                        let lane = src[(8 * t + r) * stride];
                        let bit = if b < 64 {
                            (lane.lo >> b) & 1
                        } else {
                            (lane.hi >> (b - 64)) & 1
                        };
                        want |= (bit as u8) << r;
                    }
                    assert_eq!(got[t * 128 + b], want, "stride={stride} t={t} b={b}");
                }
            }
        }
    }
}

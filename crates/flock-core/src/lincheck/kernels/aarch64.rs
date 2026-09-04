use super::super::{F128, NEON_TILE_T, build_sum_table};

// The SHA3 extension includes EOR3; retain the two-EOR form for generic
// AArch64 builds that do not enable it.
#[cfg(target_feature = "sha3")]
#[inline(always)]
unsafe fn xor3_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::veor3q_u8(a, b, c) }
}

#[cfg(not(target_feature = "sha3"))]
#[inline(always)]
unsafe fn xor3_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::veorq_u8(a, core::arch::aarch64::veorq_u8(b, c)) }
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

    let mut t = 0;
    while t + 1 < TILE_T {
        let stripe0 = row_ptr.add(t * row_stride + bs);
        let stripe1 = row_ptr.add((t + 1) * row_stride + bs);
        let table0 = tables_ptr.add(t * 256 * 16);
        let table1 = tables_ptr.add((t + 1) * 256 * 16);

        // One unaligned 8-byte load per stripe replaces eight LDRBs.
        let w0 = (stripe0 as *const u64).read_unaligned();
        let w1 = (stripe1 as *const u64).read_unaligned();

        a0 = xor3_u8(
            a0,
            vld1q_u8(table0.add((w0 & 0xff) as usize * 16)),
            vld1q_u8(table1.add((w1 & 0xff) as usize * 16)),
        );
        a1 = xor3_u8(
            a1,
            vld1q_u8(table0.add(((w0 >> 8) & 0xff) as usize * 16)),
            vld1q_u8(table1.add(((w1 >> 8) & 0xff) as usize * 16)),
        );
        a2 = xor3_u8(
            a2,
            vld1q_u8(table0.add(((w0 >> 16) & 0xff) as usize * 16)),
            vld1q_u8(table1.add(((w1 >> 16) & 0xff) as usize * 16)),
        );
        a3 = xor3_u8(
            a3,
            vld1q_u8(table0.add(((w0 >> 24) & 0xff) as usize * 16)),
            vld1q_u8(table1.add(((w1 >> 24) & 0xff) as usize * 16)),
        );
        a4 = xor3_u8(
            a4,
            vld1q_u8(table0.add(((w0 >> 32) & 0xff) as usize * 16)),
            vld1q_u8(table1.add(((w1 >> 32) & 0xff) as usize * 16)),
        );
        a5 = xor3_u8(
            a5,
            vld1q_u8(table0.add(((w0 >> 40) & 0xff) as usize * 16)),
            vld1q_u8(table1.add(((w1 >> 40) & 0xff) as usize * 16)),
        );
        a6 = xor3_u8(
            a6,
            vld1q_u8(table0.add(((w0 >> 48) & 0xff) as usize * 16)),
            vld1q_u8(table1.add(((w1 >> 48) & 0xff) as usize * 16)),
        );
        a7 = xor3_u8(
            a7,
            vld1q_u8(table0.add((w0 >> 56) as usize * 16)),
            vld1q_u8(table1.add((w1 >> 56) as usize * 16)),
        );
        t += 2;
    }
    if t < TILE_T {
        let stripe = row_ptr.add(t * row_stride + bs);
        let table = tables_ptr.add(t * 256 * 16);
        let w = (stripe as *const u64).read_unaligned();
        a0 = veorq_u8(a0, vld1q_u8(table.add((w & 0xff) as usize * 16)));
        a1 = veorq_u8(a1, vld1q_u8(table.add(((w >> 8) & 0xff) as usize * 16)));
        a2 = veorq_u8(a2, vld1q_u8(table.add(((w >> 16) & 0xff) as usize * 16)));
        a3 = veorq_u8(a3, vld1q_u8(table.add(((w >> 24) & 0xff) as usize * 16)));
        a4 = veorq_u8(a4, vld1q_u8(table.add(((w >> 32) & 0xff) as usize * 16)));
        a5 = veorq_u8(a5, vld1q_u8(table.add(((w >> 40) & 0xff) as usize * 16)));
        a6 = veorq_u8(a6, vld1q_u8(table.add(((w >> 48) & 0xff) as usize * 16)));
        a7 = veorq_u8(a7, vld1q_u8(table.add((w >> 56) as usize * 16)));
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

/// SHA3/EOR3 specialization that fuses two adjacent BLOCK_K=8 drains into one
/// stripe/table traversal. The fixed register assignment avoids callee-saved
/// vector registers and uses scaled table loads fed directly by UBFX.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn process_block16_neon_single_sha3(
    row_ptr: *const u8,
    row_stride: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    use std::arch::asm;

    asm!(
        "ldp q0, q1, [{dst}]",
        "ldp q2, q3, [{dst}, #32]",
        "ldp q4, q5, [{dst}, #64]",
        "ldp q6, q7, [{dst}, #96]",
        "ldp q16, q17, [{dst}, #128]",
        "ldp q18, q19, [{dst}, #160]",
        "ldp q20, q21, [{dst}, #192]",
        "ldp q22, q23, [{dst}, #224]",
        "add x8, {row}, {block_start}",
        "lsl x9, {stride}, #1",
        "mov x10, #4",
        "2:",
        "ldp x11, x12, [x8]",
        "add x15, x8, {stride}",
        "ldp x13, x14, [x15]",
        "add x15, {table}, #4096",

        "and x16, x11, #0xff",
        "and x17, x13, #0xff",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v0, v0, v24, v25",

        "ubfx x16, x11, #8, #8",
        "ubfx x17, x13, #8, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v1, v1, v24, v25",

        "ubfx x16, x11, #16, #8",
        "ubfx x17, x13, #16, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v2, v2, v24, v25",

        "ubfx x16, x11, #24, #8",
        "ubfx x17, x13, #24, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v3, v3, v24, v25",

        "ubfx x16, x11, #32, #8",
        "ubfx x17, x13, #32, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v4, v4, v24, v25",

        "ubfx x16, x11, #40, #8",
        "ubfx x17, x13, #40, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v5, v5, v24, v25",

        "ubfx x16, x11, #48, #8",
        "ubfx x17, x13, #48, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v6, v6, v24, v25",

        "lsr x16, x11, #56",
        "lsr x17, x13, #56",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v7, v7, v24, v25",

        "and x16, x12, #0xff",
        "and x17, x14, #0xff",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v16, v16, v24, v25",

        "ubfx x16, x12, #8, #8",
        "ubfx x17, x14, #8, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v17, v17, v24, v25",

        "ubfx x16, x12, #16, #8",
        "ubfx x17, x14, #16, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v18, v18, v24, v25",

        "ubfx x16, x12, #24, #8",
        "ubfx x17, x14, #24, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v19, v19, v24, v25",

        "ubfx x16, x12, #32, #8",
        "ubfx x17, x14, #32, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v20, v20, v24, v25",

        "ubfx x16, x12, #40, #8",
        "ubfx x17, x14, #40, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v21, v21, v24, v25",

        "ubfx x16, x12, #48, #8",
        "ubfx x17, x14, #48, #8",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v22, v22, v24, v25",

        "lsr x16, x12, #56",
        "lsr x17, x14, #56",
        "ldr q24, [{table}, x16, lsl #4]",
        "ldr q25, [x15, x17, lsl #4]",
        "eor3.16b v23, v23, v24, v25",

        "add x8, x8, x9",
        "add {table}, {table}, #8192",
        "subs x10, x10, #1",
        "b.ne 2b",

        "stp q0, q1, [{dst}]",
        "stp q2, q3, [{dst}, #32]",
        "stp q4, q5, [{dst}, #64]",
        "stp q6, q7, [{dst}, #96]",
        "stp q16, q17, [{dst}, #128]",
        "stp q18, q19, [{dst}, #160]",
        "stp q20, q21, [{dst}, #192]",
        "stp q22, q23, [{dst}, #224]",
        row = in(reg) row_ptr,
        stride = in(reg) row_stride,
        block_start = in(reg) bs,
        table = inout(reg) tables_ptr => _,
        dst = in(reg) out_ptr,
        out("x8") _,
        out("x9") _,
        out("x10") _,
        out("x11") _,
        out("x12") _,
        out("x13") _,
        out("x14") _,
        out("x15") _,
        out("x16") _,
        out("x17") _,
        out("v0") _,
        out("v1") _,
        out("v2") _,
        out("v3") _,
        out("v4") _,
        out("v5") _,
        out("v6") _,
        out("v7") _,
        out("v16") _,
        out("v17") _,
        out("v18") _,
        out("v19") _,
        out("v20") _,
        out("v21") _,
        out("v22") _,
        out("v23") _,
        out("v24") _,
        out("v25") _,
        options(nostack),
    );
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
///
/// Ranked BLAKE3 additionally fuses adjacent Block8 drains into the
/// fixed-register SHA3 Block16 helper. `FLOCK_NO_LINCHECK_BLOCK16=1` restores
/// the incumbent two-call schedule for same-binary proof/performance A/B.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_oblock_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    oblock_padded_tiled::<NEON_TILE_T>(z_packed, m, k_log, useful_bits, eq_outer, 0, usize::MAX)
}

/// Correctness-oracle entry point that forces the Block16 schedule whenever
/// the binary supports SHA3/EOR3. Production callers use the ranked-shape gate
/// inside [`oblock_padded_tiled`] instead.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_oblock16_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    oblock_padded_tiled_impl::<NEON_TILE_T>(
        z_packed,
        m,
        k_log,
        useful_bits,
        eq_outer,
        0,
        usize::MAX,
        true,
    )
}

/// Tile claims per queue chunk in the oblock fold. Exposed so a co-processor
/// arm can express its stripe prefix in the same units the CPU claims use.
pub(crate) const OBLOCK_TILES_PER_CLAIM: usize = 64;

/// Number of oblock tile claims for `(m, k_log)` at the default tile factor.
pub(crate) fn oblock_claim_count(m: usize, k_log: usize) -> usize {
    let n_stripes = (1usize << (m - k_log)) / 8;
    (n_stripes / NEON_TILE_T).div_ceil(OBLOCK_TILES_PER_CLAIM)
}

/// First stripe owned by claim `claim`.
pub(crate) fn oblock_claim_stripe_base(claim: usize) -> usize {
    claim * OBLOCK_TILES_PER_CLAIM * NEON_TILE_T
}

/// Stripe-**suffix** oblock fold: claims `[claim_lo, n_claims)` only. The
/// skipped prefix is folded by the GPU arm of the round-one C fold and XORed
/// in by the caller. GF(2¹²⁸) add is XOR — associative and commutative — so
/// any partition of the stripe range reproduces the whole-range result bit
/// for bit.
#[cfg(target_arch = "aarch64")]
pub(crate) fn partial_fold_packed_z_neon_oblock_padded_suffix(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
    claim_lo: usize,
) -> Vec<F128> {
    oblock_padded_tiled::<NEON_TILE_T>(
        z_packed,
        m,
        k_log,
        useful_bits,
        eq_outer,
        claim_lo,
        usize::MAX,
    )
}

/// Half-open claim range `[claim_lo, claim_hi)` of the oblock fold. Used as
/// the exact CPU oracle for the GPU prefix arm, and as the CPU redo path when
/// a submitted GPU prefix fails.
#[cfg(target_arch = "aarch64")]
pub(crate) fn partial_fold_packed_z_neon_oblock_padded_range(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
    claim_lo: usize,
    claim_hi: usize,
) -> Vec<F128> {
    oblock_padded_tiled::<NEON_TILE_T>(
        z_packed,
        m,
        k_log,
        useful_bits,
        eq_outer,
        claim_lo,
        claim_hi,
    )
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn oblock_padded_tiled<const TILE_T: usize>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
    claim_lo: usize,
    claim_hi: usize,
) -> Vec<F128> {
    let block16 = TILE_T == 8
        && m == 32
        && k_log == 14
        && useful_bits == 15_409
        && std::env::var_os("FLOCK_NO_LINCHECK_BLOCK16").is_none();
    oblock_padded_tiled_impl::<TILE_T>(
        z_packed,
        m,
        k_log,
        useful_bits,
        eq_outer,
        claim_lo,
        claim_hi,
        block16,
    )
}

#[cfg(target_arch = "aarch64")]
fn oblock_padded_tiled_impl<const TILE_T: usize>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
    claim_lo: usize,
    claim_hi: usize,
    block16: bool,
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

    // Contiguous tile claims drained through the two-pool chunk queue (the
    // shape promoted across zerocheck rounds 1–2, the top-NTT passes, and
    // Merkle hashing). This fold is the right side of the queue's selection
    // rule: it is load-port/L1-gather bound (≈38 GB/s at the ranked shape),
    // not DRAM-bound, so efficiency-core claims add real throughput instead
    // of join-tail latency, and there is exactly one join per prove. Each
    // claim owns a contiguous tile band, builds each of its tile tables
    // exactly once (the property that makes oblock beat iblock), and
    // accumulates into its own private length-k partial. The partial backing
    // is allocated uninitialized: every claim zeroes exactly its own slot
    // before its first accumulate, so there is no up-front 16 MiB fault pass
    // and first-touch lands on whichever core does the work.
    const TILES_PER_CLAIM: usize = OBLOCK_TILES_PER_CLAIM;
    let n_claims_total = n_tiles.div_ceil(TILES_PER_CLAIM);
    let claim_hi = claim_hi.min(n_claims_total);
    let claim_lo = claim_lo.min(claim_hi);
    let n_claims = claim_hi - claim_lo;
    if n_claims == 0 {
        return vec![F128::ZERO; k];
    }
    let mut partials = crate::alloc_uninit_f128_vec(n_claims * k);
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(n_claims, |c| {
        let claim = c + claim_lo;
        let tile_lo = claim * TILES_PER_CLAIM;
        let tile_hi = ((claim + 1) * TILES_PER_CLAIM).min(n_tiles);
        // SAFETY: the queue hands out each claim index exactly once; claim
        // `c` exclusively owns `partials[c·k .. (c+1)·k]`, which it fully
        // zero-initializes below before any read. The queue join publishes
        // all writes before the reduction reads them.
        let partial = unsafe { std::slice::from_raw_parts_mut(partials_base.ptr().add(c * k), k) };
        // SAFETY: F128 is Copy and all-zero bytes are valid F128::ZERO.
        unsafe {
            std::ptr::write_bytes(partial.as_mut_ptr(), 0, k);
        }
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
            #[cfg(target_feature = "sha3")]
            if block16 {
                while bs + 16 <= useful {
                    unsafe {
                        process_block16_neon_single_sha3(
                            z_base,
                            k,
                            bs,
                            tables_ptr,
                            partial.as_mut_ptr().add(bs),
                        );
                    }
                    bs += 16;
                }
            }
            #[cfg(not(target_feature = "sha3"))]
            let _ = block16;
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
    let n_workers = n_claims;

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

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

/// Fuse two adjacent BLOCK_K=8 drains into one stripe/table traversal.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn process_block16_neon_single_sha3(
    row_ptr: *const u8,
    row_stride: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    static UNSCALED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_LC_UNSCALED").is_none_or(|v| v != *"1")
    });
    if *UNSCALED {
        process_block16_neon_single_sha3_unscaled(row_ptr, row_stride, bs, tables_ptr, out_ptr);
    } else {
        process_block16_neon_single_sha3_scaled(row_ptr, row_stride, bs, tables_ptr, out_ptr);
    }
}

// Keep the original asm as the rollback and equivalence-test reference.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn process_block16_neon_single_sha3_scaled(
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

// Byte offsets avoid the extra scaled-load uop on Apple cores.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn process_block16_neon_single_sha3_unscaled(
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

        "ubfiz x16, x11, #4, #8",
        "ubfiz x17, x13, #4, #8",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v0, v0, v24, v25",

        "and x16, {mask}, x11, lsr #4",
        "and x17, {mask}, x13, lsr #4",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v1, v1, v24, v25",

        "and x16, {mask}, x11, lsr #12",
        "and x17, {mask}, x13, lsr #12",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v2, v2, v24, v25",

        "and x16, {mask}, x11, lsr #20",
        "and x17, {mask}, x13, lsr #20",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v3, v3, v24, v25",

        "and x16, {mask}, x11, lsr #28",
        "and x17, {mask}, x13, lsr #28",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v4, v4, v24, v25",

        "and x16, {mask}, x11, lsr #36",
        "and x17, {mask}, x13, lsr #36",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v5, v5, v24, v25",

        "and x16, {mask}, x11, lsr #44",
        "and x17, {mask}, x13, lsr #44",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v6, v6, v24, v25",

        "and x16, {mask}, x11, lsr #52",
        "and x17, {mask}, x13, lsr #52",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v7, v7, v24, v25",

        "ubfiz x16, x12, #4, #8",
        "ubfiz x17, x14, #4, #8",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v16, v16, v24, v25",

        "and x16, {mask}, x12, lsr #4",
        "and x17, {mask}, x14, lsr #4",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v17, v17, v24, v25",

        "and x16, {mask}, x12, lsr #12",
        "and x17, {mask}, x14, lsr #12",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v18, v18, v24, v25",

        "and x16, {mask}, x12, lsr #20",
        "and x17, {mask}, x14, lsr #20",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v19, v19, v24, v25",

        "and x16, {mask}, x12, lsr #28",
        "and x17, {mask}, x14, lsr #28",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v20, v20, v24, v25",

        "and x16, {mask}, x12, lsr #36",
        "and x17, {mask}, x14, lsr #36",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v21, v21, v24, v25",

        "and x16, {mask}, x12, lsr #44",
        "and x17, {mask}, x14, lsr #44",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
        "eor3.16b v22, v22, v24, v25",

        "and x16, {mask}, x12, lsr #52",
        "and x17, {mask}, x14, lsr #52",
        "ldr q24, [{table}, x16]",
        "ldr q25, [x15, x17]",
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
        mask = in(reg) 0xff0u64,
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

/// Tile-rows form of the unscaled Block16 drain: every 16-column block of one
/// tile in a single call. Same per-block work as
/// [`process_block16_neon_single_sha3_unscaled`] (eight stripe rows, sixteen
/// accumulators, two table lookups and one EOR3 per accumulator per stripe
/// pair), with the tile's eight stripe-row pointers and eight table bases
/// held in registers for the whole tile: the rows advance by post-indexed
/// loads, the tables are addressed from their own bases, and the loop over
/// column blocks runs inside the asm. Columns `[0, 16 * n_blocks)` of
/// `out_ptr` are updated in place.
///
/// # Safety
/// `n_blocks >= 1`; stripe row `s` (`0..8`) is readable for `16 * n_blocks`
/// bytes at `row_ptr + s * row_stride`; `tables_ptr` points to eight 256-entry
/// `F128` tables (32 KiB); `out_ptr` is valid for `16 * n_blocks` reads and
/// writes.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn process_tile_block16_neon_sha3_unscaled(
    row_ptr: *const u8,
    row_stride: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
    n_blocks: usize,
) {
    use std::arch::asm;
    debug_assert!(n_blocks > 0);
    const TABLE_BYTES: usize = 256 * core::mem::size_of::<F128>();

    asm!(
        "2:",
        "ldp q0, q1, [{acc}]",
        "ldp q2, q3, [{acc}, #32]",
        "ldp q4, q5, [{acc}, #64]",
        "ldp q6, q7, [{acc}, #96]",
        "ldp q16, q17, [{acc}, #128]",
        "ldp q18, q19, [{acc}, #160]",
        "ldp q20, q21, [{acc}, #192]",
        "ldp q22, q23, [{acc}, #224]",
        "ldp x11, x12, [{r0}], #16",
        "ldp x13, x14, [{r1}], #16",
        "ubfiz x16, x11, #4, #8",
        "ubfiz x17, x13, #4, #8",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v0, v0, v24, v25",
        "and x16, {mask}, x11, lsr #4",
        "and x17, {mask}, x13, lsr #4",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v1, v1, v24, v25",
        "and x16, {mask}, x11, lsr #12",
        "and x17, {mask}, x13, lsr #12",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v2, v2, v24, v25",
        "and x16, {mask}, x11, lsr #20",
        "and x17, {mask}, x13, lsr #20",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v3, v3, v24, v25",
        "and x16, {mask}, x11, lsr #28",
        "and x17, {mask}, x13, lsr #28",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v4, v4, v24, v25",
        "and x16, {mask}, x11, lsr #36",
        "and x17, {mask}, x13, lsr #36",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v5, v5, v24, v25",
        "and x16, {mask}, x11, lsr #44",
        "and x17, {mask}, x13, lsr #44",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v6, v6, v24, v25",
        "and x16, {mask}, x11, lsr #52",
        "and x17, {mask}, x13, lsr #52",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v7, v7, v24, v25",
        "ubfiz x16, x12, #4, #8",
        "ubfiz x17, x14, #4, #8",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v16, v16, v24, v25",
        "and x16, {mask}, x12, lsr #4",
        "and x17, {mask}, x14, lsr #4",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v17, v17, v24, v25",
        "and x16, {mask}, x12, lsr #12",
        "and x17, {mask}, x14, lsr #12",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v18, v18, v24, v25",
        "and x16, {mask}, x12, lsr #20",
        "and x17, {mask}, x14, lsr #20",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v19, v19, v24, v25",
        "and x16, {mask}, x12, lsr #28",
        "and x17, {mask}, x14, lsr #28",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v20, v20, v24, v25",
        "and x16, {mask}, x12, lsr #36",
        "and x17, {mask}, x14, lsr #36",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v21, v21, v24, v25",
        "and x16, {mask}, x12, lsr #44",
        "and x17, {mask}, x14, lsr #44",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v22, v22, v24, v25",
        "and x16, {mask}, x12, lsr #52",
        "and x17, {mask}, x14, lsr #52",
        "ldr q24, [{t0}, x16]",
        "ldr q25, [{t1}, x17]",
        "eor3.16b v23, v23, v24, v25",
        "ldp x11, x12, [{r2}], #16",
        "ldp x13, x14, [{r3}], #16",
        "ubfiz x16, x11, #4, #8",
        "ubfiz x17, x13, #4, #8",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v0, v0, v24, v25",
        "and x16, {mask}, x11, lsr #4",
        "and x17, {mask}, x13, lsr #4",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v1, v1, v24, v25",
        "and x16, {mask}, x11, lsr #12",
        "and x17, {mask}, x13, lsr #12",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v2, v2, v24, v25",
        "and x16, {mask}, x11, lsr #20",
        "and x17, {mask}, x13, lsr #20",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v3, v3, v24, v25",
        "and x16, {mask}, x11, lsr #28",
        "and x17, {mask}, x13, lsr #28",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v4, v4, v24, v25",
        "and x16, {mask}, x11, lsr #36",
        "and x17, {mask}, x13, lsr #36",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v5, v5, v24, v25",
        "and x16, {mask}, x11, lsr #44",
        "and x17, {mask}, x13, lsr #44",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v6, v6, v24, v25",
        "and x16, {mask}, x11, lsr #52",
        "and x17, {mask}, x13, lsr #52",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v7, v7, v24, v25",
        "ubfiz x16, x12, #4, #8",
        "ubfiz x17, x14, #4, #8",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v16, v16, v24, v25",
        "and x16, {mask}, x12, lsr #4",
        "and x17, {mask}, x14, lsr #4",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v17, v17, v24, v25",
        "and x16, {mask}, x12, lsr #12",
        "and x17, {mask}, x14, lsr #12",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v18, v18, v24, v25",
        "and x16, {mask}, x12, lsr #20",
        "and x17, {mask}, x14, lsr #20",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v19, v19, v24, v25",
        "and x16, {mask}, x12, lsr #28",
        "and x17, {mask}, x14, lsr #28",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v20, v20, v24, v25",
        "and x16, {mask}, x12, lsr #36",
        "and x17, {mask}, x14, lsr #36",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v21, v21, v24, v25",
        "and x16, {mask}, x12, lsr #44",
        "and x17, {mask}, x14, lsr #44",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v22, v22, v24, v25",
        "and x16, {mask}, x12, lsr #52",
        "and x17, {mask}, x14, lsr #52",
        "ldr q24, [{t2}, x16]",
        "ldr q25, [{t3}, x17]",
        "eor3.16b v23, v23, v24, v25",
        "ldp x11, x12, [{r4}], #16",
        "ldp x13, x14, [{r5}], #16",
        "ubfiz x16, x11, #4, #8",
        "ubfiz x17, x13, #4, #8",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v0, v0, v24, v25",
        "and x16, {mask}, x11, lsr #4",
        "and x17, {mask}, x13, lsr #4",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v1, v1, v24, v25",
        "and x16, {mask}, x11, lsr #12",
        "and x17, {mask}, x13, lsr #12",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v2, v2, v24, v25",
        "and x16, {mask}, x11, lsr #20",
        "and x17, {mask}, x13, lsr #20",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v3, v3, v24, v25",
        "and x16, {mask}, x11, lsr #28",
        "and x17, {mask}, x13, lsr #28",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v4, v4, v24, v25",
        "and x16, {mask}, x11, lsr #36",
        "and x17, {mask}, x13, lsr #36",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v5, v5, v24, v25",
        "and x16, {mask}, x11, lsr #44",
        "and x17, {mask}, x13, lsr #44",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v6, v6, v24, v25",
        "and x16, {mask}, x11, lsr #52",
        "and x17, {mask}, x13, lsr #52",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v7, v7, v24, v25",
        "ubfiz x16, x12, #4, #8",
        "ubfiz x17, x14, #4, #8",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v16, v16, v24, v25",
        "and x16, {mask}, x12, lsr #4",
        "and x17, {mask}, x14, lsr #4",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v17, v17, v24, v25",
        "and x16, {mask}, x12, lsr #12",
        "and x17, {mask}, x14, lsr #12",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v18, v18, v24, v25",
        "and x16, {mask}, x12, lsr #20",
        "and x17, {mask}, x14, lsr #20",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v19, v19, v24, v25",
        "and x16, {mask}, x12, lsr #28",
        "and x17, {mask}, x14, lsr #28",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v20, v20, v24, v25",
        "and x16, {mask}, x12, lsr #36",
        "and x17, {mask}, x14, lsr #36",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v21, v21, v24, v25",
        "and x16, {mask}, x12, lsr #44",
        "and x17, {mask}, x14, lsr #44",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v22, v22, v24, v25",
        "and x16, {mask}, x12, lsr #52",
        "and x17, {mask}, x14, lsr #52",
        "ldr q24, [{t4}, x16]",
        "ldr q25, [{t5}, x17]",
        "eor3.16b v23, v23, v24, v25",
        "ldp x11, x12, [{r6}], #16",
        "ldp x13, x14, [{r7}], #16",
        "ubfiz x16, x11, #4, #8",
        "ubfiz x17, x13, #4, #8",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v0, v0, v24, v25",
        "and x16, {mask}, x11, lsr #4",
        "and x17, {mask}, x13, lsr #4",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v1, v1, v24, v25",
        "and x16, {mask}, x11, lsr #12",
        "and x17, {mask}, x13, lsr #12",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v2, v2, v24, v25",
        "and x16, {mask}, x11, lsr #20",
        "and x17, {mask}, x13, lsr #20",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v3, v3, v24, v25",
        "and x16, {mask}, x11, lsr #28",
        "and x17, {mask}, x13, lsr #28",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v4, v4, v24, v25",
        "and x16, {mask}, x11, lsr #36",
        "and x17, {mask}, x13, lsr #36",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v5, v5, v24, v25",
        "and x16, {mask}, x11, lsr #44",
        "and x17, {mask}, x13, lsr #44",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v6, v6, v24, v25",
        "and x16, {mask}, x11, lsr #52",
        "and x17, {mask}, x13, lsr #52",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v7, v7, v24, v25",
        "ubfiz x16, x12, #4, #8",
        "ubfiz x17, x14, #4, #8",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v16, v16, v24, v25",
        "and x16, {mask}, x12, lsr #4",
        "and x17, {mask}, x14, lsr #4",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v17, v17, v24, v25",
        "and x16, {mask}, x12, lsr #12",
        "and x17, {mask}, x14, lsr #12",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v18, v18, v24, v25",
        "and x16, {mask}, x12, lsr #20",
        "and x17, {mask}, x14, lsr #20",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v19, v19, v24, v25",
        "and x16, {mask}, x12, lsr #28",
        "and x17, {mask}, x14, lsr #28",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v20, v20, v24, v25",
        "and x16, {mask}, x12, lsr #36",
        "and x17, {mask}, x14, lsr #36",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v21, v21, v24, v25",
        "and x16, {mask}, x12, lsr #44",
        "and x17, {mask}, x14, lsr #44",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v22, v22, v24, v25",
        "and x16, {mask}, x12, lsr #52",
        "and x17, {mask}, x14, lsr #52",
        "ldr q24, [{t6}, x16]",
        "ldr q25, [{t7}, x17]",
        "eor3.16b v23, v23, v24, v25",
        "stp q2, q3, [{acc}, #32]",
        "stp q4, q5, [{acc}, #64]",
        "stp q6, q7, [{acc}, #96]",
        "stp q16, q17, [{acc}, #128]",
        "stp q18, q19, [{acc}, #160]",
        "stp q20, q21, [{acc}, #192]",
        "stp q22, q23, [{acc}, #224]",
        "stp q0, q1, [{acc}], #256",
        "subs {n}, {n}, #1",
        "b.ne 2b",

        acc = inout(reg) out_ptr => _,
        n = inout(reg) n_blocks => _,
        r0 = inout(reg) row_ptr => _,
        r1 = inout(reg) row_ptr.add(row_stride) => _,
        r2 = inout(reg) row_ptr.add(2 * row_stride) => _,
        r3 = inout(reg) row_ptr.add(3 * row_stride) => _,
        r4 = inout(reg) row_ptr.add(4 * row_stride) => _,
        r5 = inout(reg) row_ptr.add(5 * row_stride) => _,
        r6 = inout(reg) row_ptr.add(6 * row_stride) => _,
        r7 = inout(reg) row_ptr.add(7 * row_stride) => _,
        t0 = in(reg) tables_ptr,
        t1 = in(reg) tables_ptr.add(TABLE_BYTES),
        t2 = in(reg) tables_ptr.add(2 * TABLE_BYTES),
        t3 = in(reg) tables_ptr.add(3 * TABLE_BYTES),
        t4 = in(reg) tables_ptr.add(4 * TABLE_BYTES),
        t5 = in(reg) tables_ptr.add(5 * TABLE_BYTES),
        t6 = in(reg) tables_ptr.add(6 * TABLE_BYTES),
        t7 = in(reg) tables_ptr.add(7 * TABLE_BYTES),
        mask = in(reg) 0xff0u64,
        out("x11") _,
        out("x12") _,
        out("x13") _,
        out("x14") _,
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

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
macro_rules! define_tile_block16_neon_sha3_unscaled_alt {
    ($fn_name:ident, $rev:expr, $step_r:literal, $step_acc:literal) => {
        #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
        #[inline(never)]
        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn $fn_name(
            row_ptr: *const u8,
            row_stride: usize,
            tables_ptr: *const u8,
            out_ptr: *mut F128,
            n_blocks: usize,
        ) {
            use std::arch::asm;
            debug_assert!(n_blocks > 0);
            const TABLE_BYTES: usize = 256 * core::mem::size_of::<F128>();

            let (start_r, start_acc) = if $rev {
                (
                    row_ptr.add((n_blocks - 1) * 16),
                    out_ptr.add((n_blocks - 1) * 16),
                )
            } else {
                (row_ptr, out_ptr)
            };

            macro_rules! stripe_pair_alt {
                ($ra:literal, $rb:literal, $ta:literal, $tb:literal) => {
                    concat!(
                        "ldp x11, x12, [", $ra, "], ", $step_r, "\n",
                        "ldp x13, x14, [", $rb, "], ", $step_r, "\n",
                        "ubfiz x16, x11, #4, #8\n",
                        "ubfiz x17, x13, #4, #8\n",
                        "ldr q24, [", $ta, ", x16]\n",
                        "ldr q25, [", $tb, ", x17]\n",
                        "eor3.16b v0, v0, v24, v25\n",
                        "and x16, {mask}, x11, lsr #4\n",
                        "and x17, {mask}, x13, lsr #4\n",
                        "ldr q26, [", $ta, ", x16]\n",
                        "ldr q27, [", $tb, ", x17]\n",
                        "eor3.16b v1, v1, v26, v27\n",
                        "and x16, {mask}, x11, lsr #12\n",
                        "and x17, {mask}, x13, lsr #12\n",
                        "ldr q24, [", $ta, ", x16]\n",
                        "ldr q25, [", $tb, ", x17]\n",
                        "eor3.16b v2, v2, v24, v25\n",
                        "and x16, {mask}, x11, lsr #20\n",
                        "and x17, {mask}, x13, lsr #20\n",
                        "ldr q26, [", $ta, ", x16]\n",
                        "ldr q27, [", $tb, ", x17]\n",
                        "eor3.16b v3, v3, v26, v27\n",
                        "and x16, {mask}, x11, lsr #28\n",
                        "and x17, {mask}, x13, lsr #28\n",
                        "ldr q24, [", $ta, ", x16]\n",
                        "ldr q25, [", $tb, ", x17]\n",
                        "eor3.16b v4, v4, v24, v25\n",
                        "and x16, {mask}, x11, lsr #36\n",
                        "and x17, {mask}, x13, lsr #36\n",
                        "ldr q26, [", $ta, ", x16]\n",
                        "ldr q27, [", $tb, ", x17]\n",
                        "eor3.16b v5, v5, v26, v27\n",
                        "and x16, {mask}, x11, lsr #44\n",
                        "and x17, {mask}, x13, lsr #44\n",
                        "ldr q24, [", $ta, ", x16]\n",
                        "ldr q25, [", $tb, ", x17]\n",
                        "eor3.16b v6, v6, v24, v25\n",
                        "and x16, {mask}, x11, lsr #52\n",
                        "and x17, {mask}, x13, lsr #52\n",
                        "ldr q26, [", $ta, ", x16]\n",
                        "ldr q27, [", $tb, ", x17]\n",
                        "eor3.16b v7, v7, v26, v27\n",
                        "ubfiz x16, x12, #4, #8\n",
                        "ubfiz x17, x14, #4, #8\n",
                        "ldr q24, [", $ta, ", x16]\n",
                        "ldr q25, [", $tb, ", x17]\n",
                        "eor3.16b v16, v16, v24, v25\n",
                        "and x16, {mask}, x12, lsr #4\n",
                        "and x17, {mask}, x14, lsr #4\n",
                        "ldr q26, [", $ta, ", x16]\n",
                        "ldr q27, [", $tb, ", x17]\n",
                        "eor3.16b v17, v17, v26, v27\n",
                        "and x16, {mask}, x12, lsr #12\n",
                        "and x17, {mask}, x14, lsr #12\n",
                        "ldr q24, [", $ta, ", x16]\n",
                        "ldr q25, [", $tb, ", x17]\n",
                        "eor3.16b v18, v18, v24, v25\n",
                        "and x16, {mask}, x12, lsr #20\n",
                        "and x17, {mask}, x14, lsr #20\n",
                        "ldr q26, [", $ta, ", x16]\n",
                        "ldr q27, [", $tb, ", x17]\n",
                        "eor3.16b v19, v19, v26, v27\n",
                        "and x16, {mask}, x12, lsr #28\n",
                        "and x17, {mask}, x14, lsr #28\n",
                        "ldr q24, [", $ta, ", x16]\n",
                        "ldr q25, [", $tb, ", x17]\n",
                        "eor3.16b v20, v20, v24, v25\n",
                        "and x16, {mask}, x12, lsr #36\n",
                        "and x17, {mask}, x14, lsr #36\n",
                        "ldr q26, [", $ta, ", x16]\n",
                        "ldr q27, [", $tb, ", x17]\n",
                        "eor3.16b v21, v21, v26, v27\n",
                        "and x16, {mask}, x12, lsr #44\n",
                        "and x17, {mask}, x14, lsr #44\n",
                        "ldr q24, [", $ta, ", x16]\n",
                        "ldr q25, [", $tb, ", x17]\n",
                        "eor3.16b v22, v22, v24, v25\n",
                        "and x16, {mask}, x12, lsr #52\n",
                        "and x17, {mask}, x14, lsr #52\n",
                        "ldr q26, [", $ta, ", x16]\n",
                        "ldr q27, [", $tb, ", x17]\n",
                        "eor3.16b v23, v23, v26, v27\n",
                    )
                };
            }

            asm!(
                "2:",
                "ldp q0, q1, [{acc}]",
                "ldp q2, q3, [{acc}, #32]",
                "ldp q4, q5, [{acc}, #64]",
                "ldp q6, q7, [{acc}, #96]",
                "ldp q16, q17, [{acc}, #128]",
                "ldp q18, q19, [{acc}, #160]",
                "ldp q20, q21, [{acc}, #192]",
                "ldp q22, q23, [{acc}, #224]",
                stripe_pair_alt!("{r0}", "{r1}", "{t0}", "{t1}"),
                stripe_pair_alt!("{r2}", "{r3}", "{t2}", "{t3}"),
                stripe_pair_alt!("{r4}", "{r5}", "{t4}", "{t5}"),
                stripe_pair_alt!("{r6}", "{r7}", "{t6}", "{t7}"),
                "stp q2, q3, [{acc}, #32]",
                "stp q4, q5, [{acc}, #64]",
                "stp q6, q7, [{acc}, #96]",
                "stp q16, q17, [{acc}, #128]",
                "stp q18, q19, [{acc}, #160]",
                "stp q20, q21, [{acc}, #192]",
                "stp q22, q23, [{acc}, #224]",
                concat!("stp q0, q1, [{acc}], ", $step_acc),
                "subs {n}, {n}, #1",
                "b.ne 2b",
                acc = inout(reg) start_acc => _,
                n = inout(reg) n_blocks => _,
                r0 = inout(reg) start_r => _,
                r1 = inout(reg) start_r.add(row_stride) => _,
                r2 = inout(reg) start_r.add(2 * row_stride) => _,
                r3 = inout(reg) start_r.add(3 * row_stride) => _,
                r4 = inout(reg) start_r.add(4 * row_stride) => _,
                r5 = inout(reg) start_r.add(5 * row_stride) => _,
                r6 = inout(reg) start_r.add(6 * row_stride) => _,
                r7 = inout(reg) start_r.add(7 * row_stride) => _,
                t0 = in(reg) tables_ptr,
                t1 = in(reg) tables_ptr.add(TABLE_BYTES),
                t2 = in(reg) tables_ptr.add(2 * TABLE_BYTES),
                t3 = in(reg) tables_ptr.add(3 * TABLE_BYTES),
                t4 = in(reg) tables_ptr.add(4 * TABLE_BYTES),
                t5 = in(reg) tables_ptr.add(5 * TABLE_BYTES),
                t6 = in(reg) tables_ptr.add(6 * TABLE_BYTES),
                t7 = in(reg) tables_ptr.add(7 * TABLE_BYTES),
                mask = in(reg) 0xff0u64,
                out("x11") _,
                out("x12") _,
                out("x13") _,
                out("x14") _,
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
                out("v26") _,
                out("v27") _,
                options(nostack),
            );
        }
    };
}

// Forward whole-tile Block16 drain with disjoint scratch-register alternation
// (`(q24, q25)` on even columns, `(q26, q27)` on odd columns) to break
// back-to-back WAW hazards on the table-lookup load pair.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
define_tile_block16_neon_sha3_unscaled_alt!(
    process_tile_block16_neon_sha3_unscaled_alt,
    false,
    "#16",
    "#256"
);

// Reverse whole-tile Block16 drain paired with
// `process_tile_block16_neon_sha3_unscaled_alt` for alternating snake-order
// tile traversal: keeps the hot trailing ~96 KiB of the 180 KiB per-worker
// accumulator resident in Apple P-core L1D across adjacent tile boundaries
// without rebuilding any sum tables.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
define_tile_block16_neon_sha3_unscaled_alt!(
    process_tile_block16_neon_sha3_unscaled_alt_rev,
    true,
    "#-16",
    "#-256"
);

/// Whole-tile Block16 drain (default on). `FLOCK_NO_LC_TILE_ROWS=1` (exact
/// `"1"`) restores the per-block calls. Only the unscaled load form has a
/// tile-rows kernel, so `FLOCK_NO_LC_UNSCALED=1` also takes the per-block
/// calls. Read once per process.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
fn lc_tile_rows_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("FLOCK_NO_LC_TILE_ROWS").ok().as_deref() != Some("1")
            && std::env::var_os("FLOCK_NO_LC_UNSCALED").is_none_or(|v| v != *"1")
    });
    *ON
}

/// Kill switch: exactly `FLOCK_NO_LC_TABLE_REGS=1` builds the oblock drain's
/// per-stripe sum tables through [`build_sum_table`] (the doubling that reads
/// back every earlier entry); otherwise [`build_sum_table_regs`] writes the
/// same 256 entries from registers. Read once per process (a plain atomic,
/// no lazy-cell instance).
fn lc_table_regs_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    // 0: not read yet, 1: on, 2: off.
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let on = std::env::var_os("FLOCK_NO_LC_TABLE_REGS").as_deref() != Some(std::ffi::OsStr::new("1"));
            STATE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
    }
}

/// [`build_sum_table`] written from registers: `table[b]` is the XOR of
/// `eq8[r]` over the set bits `r` of `b`. The sixteen sums of `eq8[0..4]` stay
/// in registers; the sixteen rows of sixteen entries (high nibble `g`, row
/// base `16g`) are visited in Gray-code order of `g`, so each row's high sum
/// is the previous one plus one `eq8[4 + i]`, and each entry is one XOR and
/// one store. Nothing is read back from the table. F128 addition is XOR, so
/// every entry is the same field element the doubling builds.
///
/// # Safety
/// `eq8` must be valid for 8 reads and `table` for 256 writes.
#[inline(never)]
unsafe fn build_sum_table_regs(eq8: *const F128, table: *mut F128) {
    use core::arch::aarch64::*;
    unsafe {
        let ld = |i: usize| vld1q_u64(eq8.add(i).cast::<u64>());
        let x = |a: uint64x2_t, b: uint64x2_t| veorq_u64(a, b);
        let (e0, e1, e2, e3) = (ld(0), ld(1), ld(2), ld(3));
        let (e4, e5, e6, e7) = (ld(4), ld(5), ld(6), ld(7));
        let l1 = e0;
        let l2 = e1;
        let l3 = x(e0, e1);
        let l4 = e2;
        let l5 = x(e2, l1);
        let l6 = x(e2, l2);
        let l7 = x(e2, l3);
        let l8 = e3;
        let l9 = x(e3, l1);
        let l10 = x(e3, l2);
        let l11 = x(e3, l3);
        let l12 = x(e3, l4);
        let l13 = x(e3, l5);
        let l14 = x(e3, l6);
        let l15 = x(e3, l7);
        let st = |i: usize, v: uint64x2_t| vst1q_u64(table.add(i).cast::<u64>(), v);
        // Row 0: the low sums themselves.
        st(0, vdupq_n_u64(0));
        st(1, l1);
        st(2, l2);
        st(3, l3);
        st(4, l4);
        st(5, l5);
        st(6, l6);
        st(7, l7);
        st(8, l8);
        st(9, l9);
        st(10, l10);
        st(11, l11);
        st(12, l12);
        st(13, l13);
        st(14, l14);
        st(15, l15);
        macro_rules! row {
            ($g:expr, $h:expr) => {{
                let h = $h;
                let base = 16 * $g;
                st(base, h);
                st(base + 1, x(l1, h));
                st(base + 2, x(l2, h));
                st(base + 3, x(l3, h));
                st(base + 4, x(l4, h));
                st(base + 5, x(l5, h));
                st(base + 6, x(l6, h));
                st(base + 7, x(l7, h));
                st(base + 8, x(l8, h));
                st(base + 9, x(l9, h));
                st(base + 10, x(l10, h));
                st(base + 11, x(l11, h));
                st(base + 12, x(l12, h));
                st(base + 13, x(l13, h));
                st(base + 14, x(l14, h));
                st(base + 15, x(l15, h));
            }};
        }
        // Gray-code order of the high nibble: 1, 3, 2, 6, 7, 5, 4, 12, 13,
        // 15, 14, 10, 11, 9, 8; `h` is the sum of `eq8[4..8]` over `g`'s bits.
        let mut h = e4;
        row!(1, h);
        h = x(h, e5);
        row!(3, h);
        h = x(h, e4);
        row!(2, h);
        h = x(h, e6);
        row!(6, h);
        h = x(h, e4);
        row!(7, h);
        h = x(h, e5);
        row!(5, h);
        h = x(h, e4);
        row!(4, h);
        h = x(h, e7);
        row!(12, h);
        h = x(h, e4);
        row!(13, h);
        h = x(h, e5);
        row!(15, h);
        h = x(h, e4);
        row!(14, h);
        h = x(h, e6);
        row!(10, h);
        h = x(h, e4);
        row!(11, h);
        h = x(h, e5);
        row!(9, h);
        h = x(h, e4);
        row!(8, h);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn build_sum_table_regs_matches_doubling() {
    let mut s = 0x51ab_1e5e_ed00_c0deu64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for _ in 0..64 {
        let eq8: Vec<F128> = (0..8).map(|_| F128::new(next(), next())).collect();
        let mut want = vec![F128::ZERO; 256];
        build_sum_table(&eq8, &mut want);
        let mut got = vec![F128::new(next(), next()); 256];
        unsafe { build_sum_table_regs(eq8.as_ptr(), got.as_mut_ptr()) };
        assert_eq!(got, want);
    }
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "sha3"))]
#[test]
fn block16_tile_rows_match_per_block_kernel() {
    let mut state = 0x0bad_5eed_1234_9876u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let tables: Vec<F128> = (0..8 * 256)
        .map(|_| F128 {
            lo: next(),
            hi: next(),
        })
        .collect();
    for &n_blocks in &[1usize, 2, 3, 5, 17, 64] {
        let stride = 16 * n_blocks + 40;
        let mut rows = vec![0u8; 8 * stride];
        for byte in &mut rows {
            *byte = next() as u8;
        }
        // Every byte value at every column position of every stripe.
        for (i, byte) in rows.iter_mut().enumerate().take(8 * stride).skip(stride) {
            if i % 7 == 0 {
                *byte = (i / 7) as u8;
            }
        }
        let initial: Vec<F128> = (0..16 * n_blocks + 3)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let mut expected = initial.clone();
        for c in 0..16 * n_blocks {
            for stripe in 0..8 {
                expected[c] += tables[stripe * 256 + rows[stripe * stride + c] as usize];
            }
        }
        let mut per_block = initial.clone();
        let mut tile = initial.clone();
        unsafe {
            for b in 0..n_blocks {
                process_block16_neon_single_sha3_unscaled(
                    rows.as_ptr(),
                    stride,
                    16 * b,
                    tables.as_ptr().cast(),
                    per_block.as_mut_ptr().add(16 * b),
                );
            }
            process_tile_block16_neon_sha3_unscaled(
                rows.as_ptr(),
                stride,
                tables.as_ptr().cast(),
                tile.as_mut_ptr(),
                n_blocks,
            );
        }
        assert_eq!(per_block, expected, "per-block kernel, n_blocks {n_blocks}");
        assert_eq!(tile, expected, "tile-rows kernel, n_blocks {n_blocks}");
        assert_eq!(tile[16 * n_blocks..], initial[16 * n_blocks..], "past the tile");
    }
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "sha3"))]
#[test]
fn block16_unscaled_matches_scaled_all_byte_values() {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let tables: Vec<F128> = (0..8 * 256)
        .map(|_| F128 {
            lo: next(),
            hi: next(),
        })
        .collect();
    const STRIDE: usize = 48;
    const BS: usize = 5;
    let mut rows = [0u8; 8 * STRIDE];
    for byte in &mut rows {
        *byte = next() as u8;
    }
    for position in 0..8 {
        for value in 0..=255u8 {
            for stripe in 0..8 {
                rows[stripe * STRIDE + BS + position] = value;
                rows[stripe * STRIDE + BS + 8 + position] = value;
            }
            let initial: Vec<F128> = (0..16)
                .map(|_| F128 {
                    lo: next(),
                    hi: next(),
                })
                .collect();
            let mut scaled = initial.clone();
            let mut unscaled = initial.clone();
            let mut expected = initial;
            for stripe in 0..8 {
                for i in 0..16 {
                    expected[i] += tables[stripe * 256 + rows[stripe * STRIDE + BS + i] as usize];
                }
            }
            unsafe {
                process_block16_neon_single_sha3_scaled(
                    rows.as_ptr(),
                    STRIDE,
                    BS,
                    tables.as_ptr().cast(),
                    scaled.as_mut_ptr(),
                );
                process_block16_neon_single_sha3_unscaled(
                    rows.as_ptr(),
                    STRIDE,
                    BS,
                    tables.as_ptr().cast(),
                    unscaled.as_mut_ptr(),
                );
            }
            assert_eq!(scaled, expected, "scaled: byte {position}, value {value}");
            assert_eq!(
                unscaled, expected,
                "unscaled: byte {position}, value {value}"
            );
        }
    }
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "sha3"))]
#[test]
fn tile_block16_snake_alt_matches_incumbent_forward() {
    const STRIDE: usize = 512;
    const TABLES_LEN: usize = 8 * 256;

    let mut tables = vec![F128::ZERO; TABLES_LEN];
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next_u64 = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };
    for t in 0..8 {
        for e in 1..256 {
            tables[t * 256 + e] = F128 {
                lo: next_u64(),
                hi: next_u64(),
            };
        }
    }
    let mut rows_a = vec![0u8; 8 * STRIDE];
    let mut rows_b = vec![0u8; 8 * STRIDE];
    for b in rows_a.iter_mut() {
        *b = (next_u64() >> 56) as u8;
    }
    for b in rows_b.iter_mut() {
        *b = (next_u64() >> 56) as u8;
    }

    for n_blocks in [1usize, 2, 7, 19] {
        let cols = n_blocks * 16;
        let mut ref_out = vec![F128::ZERO; cols + 8];
        let mut fwd_out = vec![F128::ZERO; cols + 8];
        let mut rev_out = vec![F128::ZERO; cols + 8];
        let mut snake_out = vec![F128::ZERO; cols + 8];
        let mut ref_snake = vec![F128::ZERO; cols + 8];
        for i in 0..cols + 8 {
            let v = F128 {
                lo: next_u64(),
                hi: next_u64(),
            };
            ref_out[i] = v;
            fwd_out[i] = v;
            rev_out[i] = v;
            snake_out[i] = v;
            ref_snake[i] = v;
        }
        unsafe {
            process_tile_block16_neon_sha3_unscaled(
                rows_a.as_ptr(),
                STRIDE,
                tables.as_ptr().cast(),
                ref_out.as_mut_ptr(),
                n_blocks,
            );
            process_tile_block16_neon_sha3_unscaled_alt(
                rows_a.as_ptr(),
                STRIDE,
                tables.as_ptr().cast(),
                fwd_out.as_mut_ptr(),
                n_blocks,
            );
            process_tile_block16_neon_sha3_unscaled_alt_rev(
                rows_a.as_ptr(),
                STRIDE,
                tables.as_ptr().cast(),
                rev_out.as_mut_ptr(),
                n_blocks,
            );
            process_tile_block16_neon_sha3_unscaled(
                rows_a.as_ptr(),
                STRIDE,
                tables.as_ptr().cast(),
                ref_snake.as_mut_ptr(),
                n_blocks,
            );
            process_tile_block16_neon_sha3_unscaled(
                rows_b.as_ptr(),
                STRIDE,
                tables.as_ptr().cast(),
                ref_snake.as_mut_ptr(),
                n_blocks,
            );
            process_tile_block16_neon_sha3_unscaled_alt(
                rows_a.as_ptr(),
                STRIDE,
                tables.as_ptr().cast(),
                snake_out.as_mut_ptr(),
                n_blocks,
            );
            process_tile_block16_neon_sha3_unscaled_alt_rev(
                rows_b.as_ptr(),
                STRIDE,
                tables.as_ptr().cast(),
                snake_out.as_mut_ptr(),
                n_blocks,
            );
        }
        assert_eq!(fwd_out, ref_out, "alt forward mismatch for n_blocks={n_blocks}");
        assert_eq!(rev_out, ref_out, "alt reverse mismatch for n_blocks={n_blocks}");
        assert_eq!(
            snake_out, ref_snake,
            "two-tile snake mismatch for n_blocks={n_blocks}"
        );
    }
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
        // 11_313 / 11_249 = the ranked BLAKE3 root stripe widths
        // (`lincheck::RootStripe`; 11_249 also drops the fixed block_len/flags
        // input columns).
        && (useful_bits == 15_409 || useful_bits == 11_313 || useful_bits == 11_249)
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
    // Fold tiles `[tile_lo, tile_hi)` into the length-k accumulator at
    // `partial` (only `[0..useful)` is read or written; the caller zeroed it).
    // Shared by both claim schedules below, so they cannot drift apart.
    // Whole-tile Block16 drain (one call per tile instead of one per 16-column
    // block); decided once per fold.
    #[cfg(target_feature = "sha3")]
    let tile_rows = block16 && lc_tile_rows_enabled();
    #[cfg(target_feature = "sha3")]
    let tile_snake = tile_rows && lc_tile_snake_enabled();
    // The tables from registers (`FLOCK_NO_LC_TABLE_REGS=1` keeps the
    // doubling); decided once per fold.
    let table_regs = lc_table_regs_enabled();
    let fold_tiles = |partial: *mut F128, tile_lo: usize, tile_hi: usize| {
        // TILE_T × 256 F128 tables, L1-resident on stack, built once per tile.
        let mut tables_uninit = core::mem::MaybeUninit::<[[F128; 256]; TILE_T]>::uninit();
        let tables_ptr = tables_uninit.as_mut_ptr() as *const u8;
        for tile in tile_lo..tile_hi {
            let stripe_base = tile * TILE_T;
            for t in 0..TILE_T {
                let eq_off = 8 * (stripe_base + t);
                // SAFETY: `t < TILE_T` and `build_sum_table` writes all 256 entries.
                let table_t = unsafe {
                    std::slice::from_raw_parts_mut(
                        (tables_uninit.as_mut_ptr() as *mut F128).add(t * 256),
                        256,
                    )
                };
                let eq8 = &eq_outer[eq_off..eq_off + 8];
                if table_regs {
                    // SAFETY: `eq8` holds 8 entries and `table_t` 256.
                    unsafe { build_sum_table_regs(eq8.as_ptr(), table_t.as_mut_ptr()) };
                } else {
                    build_sum_table(eq8, table_t);
                }
            }
            let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k) };
            let mut bs = 0usize;
            #[cfg(target_feature = "sha3")]
            if block16 {
                let n16 = useful / 16;
                if tile_rows && n16 > 0 {
                    // SAFETY: TILE_T == 8 (the block16 gate), so `tables_ptr`
                    // holds eight complete tables; stripe rows `0..8` of this
                    // tile are `k`-byte rows at `z_base + s * k` and
                    // `16 * n16 <= useful <= k`; `partial` holds `useful`
                    // accumulators. Same blocks, rows, tables and
                    // accumulators as the per-block calls below; GF(2^128)
                    // XOR across disjoint 16-column blocks is order-invariant.
                    unsafe {
                        if tile_snake {
                            if tile & 1 == 0 {
                                process_tile_block16_neon_sha3_unscaled_alt(
                                    z_base, k, tables_ptr, partial, n16,
                                );
                            } else {
                                process_tile_block16_neon_sha3_unscaled_alt_rev(
                                    z_base, k, tables_ptr, partial, n16,
                                );
                            }
                        } else {
                            process_tile_block16_neon_sha3_unscaled(
                                z_base, k, tables_ptr, partial, n16,
                            );
                        }
                    }
                    bs = 16 * n16;
                }
                while bs + 16 <= useful {
                    unsafe {
                        process_block16_neon_single_sha3(z_base, k, bs, tables_ptr, partial.add(bs));
                    }
                    bs += 16;
                }
            }
            #[cfg(not(target_feature = "sha3"))]
            let _ = block16;
            while bs < useful {
                unsafe {
                    process_block_neon_single::<TILE_T>(z_base, k, bs, tables_ptr, partial.add(bs));
                }
                bs += BLOCK_K;
            }
        }
    };

    // XOR-reduce `parts` (each `[0..useful)` valid) in ONE parallel pass over
    // column bands: each worker owns a band of the output and reduces all
    // partials for that band so the band stays L1-resident without an up-front
    // zero-fill of `out`. When `FLOCK_NO_LC_REDUCE_EOR3` is unset, each band
    // uses 3-stream initial + 2-stream in-place `eor3.16b` (`ldp`/`stp`) passes
    // to halve memory passes over `dst` while preserving stream-prefetcher
    // locality.
    let use_eor3_reduce = lc_reduce_eor3_enabled();
    let reduce = |parts: &[&[F128]]| -> Vec<F128> {
        let band = useful.div_ceil(rayon::current_num_threads().max(1)).max(1024);
        let mut out = crate::scratch::take_lincheck_z_vec(k);
        out[useful..k].fill(F128::ZERO);
        out[..useful]
            .par_chunks_mut(band)
            .enumerate()
            .for_each(|(bi, dst)| {
                let lo = bi * band;
                if use_eor3_reduce {
                    // SAFETY: `parts` slices each cover `lo..lo + dst.len()` and
                    // `aarch64` NEON is always available (with `eor3` under `sha3`).
                    unsafe { xor_reduce_band_neon(parts, lo, dst) };
                } else {
                    dst.copy_from_slice(&parts[0][lo..lo + dst.len()]);
                    for part in &parts[1..] {
                        let src = &part[lo..lo + dst.len()];
                        for (o, s) in dst.iter_mut().zip(src.iter()) {
                            *o += *s;
                        }
                    }
                }
            });
        out
    };

    if lincheck_worker_partials_enabled() {
        // Worker-owned partials over fine sub-claims. The claim range (the
        // unit the GPU prefix split is expressed in) is unchanged, but the
        // CPU queue hands out `SUB_CLAIM_TILES`-tile sub-claims and each
        // worker accumulates every sub-claim it runs into ONE private
        // length-k partial, taken from a per-drain slot table on its first
        // claim. A 64-tile claim is several milliseconds on an efficiency
        // core, so the drain used to wait for whichever helper claimed one
        // near the end; a quarter-size claim cuts that join tail, and the
        // reduction reads one partial per participating worker instead of
        // one per claim.
        //
        // Exact: every tile's contribution is XORed into exactly one partial
        // and the partials are XOR-summed. GF(2^128) addition is XOR
        // (associative and commutative), so the result does not depend on
        // the tile -> worker assignment or the summation order; it equals
        // the per-claim schedule bit for bit. `FLOCK_NO_LINCHECK_WORKER_PARTIALS=1`
        // (exact '1') restores the per-claim schedule.
        const SUB_CLAIM_TILES: usize = 16;
        let tile_lo_all = claim_lo * TILES_PER_CLAIM;
        let tile_hi_all = (claim_hi * TILES_PER_CLAIM).min(n_tiles);
        let n_sub = (tile_hi_all - tile_lo_all).div_ceil(SUB_CLAIM_TILES);
        let max_slots = rayon::current_num_threads()
            + crate::epool::helper_pool().map_or(0, |pool| pool.current_num_threads());
        let mut partials = crate::scratch::take_f128(max_slots * k);
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        let next_slot = std::sync::atomic::AtomicUsize::new(0);
        // Defensive: a participant beyond the slot table (not expected: only
        // main-pool and helper-pool threads drain the queue) gets its own
        // allocation. Moving the Vec into the list never moves its buffer.
        let overflow: std::sync::Mutex<Vec<Vec<F128>>> = std::sync::Mutex::new(Vec::new());
        crate::epool::run_hetero_chunks_stateful(
            n_sub,
            || None::<crate::epool::SyncPtr<F128>>,
            |slot, sub| {
                let partial = slot
                    .get_or_insert_with(|| {
                        let i = next_slot.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let ptr = if i < max_slots {
                            // SAFETY: slot `i` is handed out exactly once;
                            // it owns `partials[i·k .. (i+1)·k]`.
                            unsafe { partials_base.ptr().add(i * k) }
                        } else {
                            let mut own: Vec<F128> = crate::alloc_uninit_vec(k);
                            let ptr = own.as_mut_ptr();
                            overflow.lock().unwrap_or_else(|e| e.into_inner()).push(own);
                            ptr
                        };
                        // SAFETY: all-zero bytes are F128::ZERO; only
                        // `[0..useful)` is ever read or written.
                        unsafe { std::ptr::write_bytes(ptr, 0, useful) };
                        crate::epool::SyncPtr(ptr)
                    })
                    .ptr();
                let tile_lo = tile_lo_all + sub * SUB_CLAIM_TILES;
                let tile_hi = (tile_lo + SUB_CLAIM_TILES).min(tile_hi_all);
                fold_tiles(partial, tile_lo, tile_hi);
            },
        );
        // The queue join published every slot write.
        let n_used = next_slot.load(std::sync::atomic::Ordering::Relaxed).min(max_slots);
        let overflow = overflow.into_inner().unwrap_or_else(|e| e.into_inner());
        let mut parts: Vec<&[F128]> = (0..n_used)
            .map(|i| &partials[i * k..i * k + useful])
            .collect();
        parts.extend(overflow.iter().map(|own| &own[..useful]));
        let out = reduce(&parts);
        drop(parts);
        crate::scratch::give_f128(partials);
        return out;
    }

    let mut partials = crate::scratch::take_f128(n_claims * k);
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(n_claims, |c| {
        let claim = c + claim_lo;
        let tile_lo = claim * TILES_PER_CLAIM;
        let tile_hi = ((claim + 1) * TILES_PER_CLAIM).min(n_tiles);
        // SAFETY: the queue hands out each claim index exactly once; claim
        // `c` exclusively owns `partials[c·k .. (c+1)·k]`, and zeroes the
        // `[0..useful)` prefix below before any read. The queue join publishes
        // all writes before the reduction reads them.
        let partial = unsafe { partials_base.ptr().add(c * k) };
        // SAFETY: F128 is Copy and all-zero bytes are valid F128::ZERO; only
        // `[0..useful)` is ever read or written by the tile loops.
        unsafe {
            std::ptr::write_bytes(partial, 0, useful);
        }
        fold_tiles(partial, tile_lo, tile_hi);
    });
    let parts: Vec<&[F128]> = (0..n_claims).map(|c| &partials[c * k..c * k + useful]).collect();
    let out = reduce(&parts);
    drop(parts);
    crate::scratch::give_f128(partials);
    out
}

/// Worker-owned partials over quarter-claims in the oblock fold (default on;
/// `FLOCK_NO_LINCHECK_WORKER_PARTIALS=1`, exact `"1"`, restores the per-claim
/// partial schedule). Read once per process.
fn lincheck_worker_partials_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FLOCK_NO_LINCHECK_WORKER_PARTIALS").ok().as_deref() != Some("1")
    })
}

/// Kill switch for the 3-way-init + 2-way `eor3.16b` (`ldp`/`stp`) oblock
/// partial-sum reducer (`FLOCK_NO_LC_REDUCE_EOR3`, any value restores the
/// incumbent `copy_from_slice` + per-partial XOR loop). Read once per process.
pub const ENV_NO_LC_REDUCE_EOR3: &str = "FLOCK_NO_LC_REDUCE_EOR3";

fn lc_reduce_eor3_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(ENV_NO_LC_REDUCE_EOR3).is_none())
}

/// Kill switch for alternating forward/reverse snake tile traversal with
/// disjoint `(q24, q25)/(q26, q27)` scratch-register alternation in the
/// whole-tile Block16 drain (`FLOCK_NO_LC_TILE_SNAKE`, any value restores
/// forward-only [`process_tile_block16_neon_sha3_unscaled`]). Read once per
/// process.
pub const ENV_NO_LC_TILE_SNAKE: &str = "FLOCK_NO_LC_TILE_SNAKE";

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
fn lc_tile_snake_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(ENV_NO_LC_TILE_SNAKE).is_none())
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn lc_load_pair_q(
    ptr: *const F128,
) -> (core::arch::aarch64::uint64x2_t, core::arch::aarch64::uint64x2_t) {
    let a: core::arch::aarch64::uint64x2_t;
    let b: core::arch::aarch64::uint64x2_t;
    unsafe {
        core::arch::asm!(
            "ldp {a:q}, {b:q}, [{ptr}]",
            ptr = in(reg) ptr,
            a = lateout(vreg) a,
            b = lateout(vreg) b,
            options(nostack, readonly, preserves_flags),
        );
    }
    (a, b)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn lc_store_quad_q(
    p: *mut F128,
    a0: core::arch::aarch64::uint64x2_t,
    a1: core::arch::aarch64::uint64x2_t,
    a2: core::arch::aarch64::uint64x2_t,
    a3: core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        core::arch::asm!(
            "stp {a0:q}, {a1:q}, [{p}]",
            "stp {a2:q}, {a3:q}, [{p}, #32]",
            p = in(reg) p,
            a0 = in(vreg) a0,
            a1 = in(vreg) a1,
            a2 = in(vreg) a2,
            a3 = in(vreg) a3,
            options(nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn lc_xor3_q(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    c: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    #[cfg(target_feature = "sha3")]
    {
        unsafe { veor3q_u64(a, b, c) }
    }
    #[cfg(not(target_feature = "sha3"))]
    {
        unsafe { veorq_u64(a, veorq_u64(b, c)) }
    }
}

/// Stream-friendly 3-way-init + 2-way `eor3.16b` (`ldp`/`stp`) band XOR reducer.
/// Bit-identical to copying `parts[0][lo..lo+len]` and XORing `parts[1..]`.
///
/// # Safety
/// Every slice in `parts` must have length at least `lo + dst.len()`.
#[cfg(target_arch = "aarch64")]
unsafe fn xor_reduce_band_neon(parts: &[&[F128]], lo: usize, dst: &mut [F128]) {
    use core::arch::aarch64::*;

    let len = dst.len();
    let n_parts = parts.len();
    if n_parts == 0 || len == 0 {
        return;
    }
    if n_parts == 1 {
        dst.copy_from_slice(&parts[0][lo..lo + len]);
        return;
    }
    let dst_ptr = dst.as_mut_ptr();
    let vec_len = len & !3usize;
    let mut p;
    unsafe {
        if n_parts >= 3 {
            let s0 = parts[0].as_ptr().add(lo);
            let s1 = parts[1].as_ptr().add(lo);
            let s2 = parts[2].as_ptr().add(lo);
            let mut i = 0usize;
            while i < vec_len {
                let (a0, a1) = lc_load_pair_q(s0.add(i));
                let (b0, b1) = lc_load_pair_q(s1.add(i));
                let (c0, c1) = lc_load_pair_q(s2.add(i));
                let r0 = lc_xor3_q(a0, b0, c0);
                let r1 = lc_xor3_q(a1, b1, c1);
                let (a2, a3) = lc_load_pair_q(s0.add(i + 2));
                let (b2, b3) = lc_load_pair_q(s1.add(i + 2));
                let (c2, c3) = lc_load_pair_q(s2.add(i + 2));
                let r2 = lc_xor3_q(a2, b2, c2);
                let r3 = lc_xor3_q(a3, b3, c3);
                lc_store_quad_q(dst_ptr.add(i), r0, r1, r2, r3);
                i += 4;
            }
            while i < len {
                *dst_ptr.add(i) = *s0.add(i) + *s1.add(i) + *s2.add(i);
                i += 1;
            }
            p = 3;
        } else {
            let s0 = parts[0].as_ptr().add(lo);
            let s1 = parts[1].as_ptr().add(lo);
            let mut i = 0usize;
            while i < vec_len {
                let (a0, a1) = lc_load_pair_q(s0.add(i));
                let (b0, b1) = lc_load_pair_q(s1.add(i));
                let (a2, a3) = lc_load_pair_q(s0.add(i + 2));
                let (b2, b3) = lc_load_pair_q(s1.add(i + 2));
                lc_store_quad_q(
                    dst_ptr.add(i),
                    veorq_u64(a0, b0),
                    veorq_u64(a1, b1),
                    veorq_u64(a2, b2),
                    veorq_u64(a3, b3),
                );
                i += 4;
            }
            while i < len {
                *dst_ptr.add(i) = *s0.add(i) + *s1.add(i);
                i += 1;
            }
            p = 2;
        }
        while p + 1 < n_parts {
            let s0 = parts[p].as_ptr().add(lo);
            let s1 = parts[p + 1].as_ptr().add(lo);
            let mut i = 0usize;
            while i < vec_len {
                let d = dst_ptr.add(i);
                let (d0, d1) = lc_load_pair_q(d);
                let (a0, a1) = lc_load_pair_q(s0.add(i));
                let (b0, b1) = lc_load_pair_q(s1.add(i));
                let r0 = lc_xor3_q(d0, a0, b0);
                let r1 = lc_xor3_q(d1, a1, b1);
                let (d2, d3) = lc_load_pair_q(d.add(2));
                let (a2, a3) = lc_load_pair_q(s0.add(i + 2));
                let (b2, b3) = lc_load_pair_q(s1.add(i + 2));
                let r2 = lc_xor3_q(d2, a2, b2);
                let r3 = lc_xor3_q(d3, a3, b3);
                lc_store_quad_q(d, r0, r1, r2, r3);
                i += 4;
            }
            while i < len {
                *dst_ptr.add(i) += *s0.add(i) + *s1.add(i);
                i += 1;
            }
            p += 2;
        }
        if p < n_parts {
            let s0 = parts[p].as_ptr().add(lo);
            let mut i = 0usize;
            while i < vec_len {
                let d = dst_ptr.add(i);
                let (d0, d1) = lc_load_pair_q(d);
                let (a0, a1) = lc_load_pair_q(s0.add(i));
                let (d2, d3) = lc_load_pair_q(d.add(2));
                let (a2, a3) = lc_load_pair_q(s0.add(i + 2));
                lc_store_quad_q(
                    d,
                    veorq_u64(d0, a0),
                    veorq_u64(d1, a1),
                    veorq_u64(d2, a2),
                    veorq_u64(d3, a3),
                );
                i += 4;
            }
            while i < len {
                *dst_ptr.add(i) += *s0.add(i);
                i += 1;
            }
        }
    }
}

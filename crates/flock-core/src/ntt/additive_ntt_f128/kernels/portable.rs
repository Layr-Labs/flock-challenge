use crate::field::F128;

#[inline]
pub(super) fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    for lane in 0..top.len() {
        let v = bot[lane];
        let new_u = top[lane] + v * twiddle;
        top[lane] = new_u;
        bot[lane] = v + new_u;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    for lane in 0..a.len() {
        let mut xa = a[lane];
        let mut xb = b[lane];
        let mut xc = c[lane];
        let mut xd = d[lane];
        let na = xa + xc * t_outer;
        xc += na;
        xa = na;
        let nb = xb + xd * t_outer;
        xd += nb;
        xb = nb;
        let na2 = xa + xb * t_inner_a;
        xb += na2;
        xa = na2;
        let nc2 = xc + xd * t_inner_b;
        xd += nc2;
        xc = nc2;
        a[lane] = xa;
        b[lane] = xb;
        c[lane] = xc;
        d[lane] = xd;
    }
}

/// # Safety
/// The caller guarantees that every selected row and lane is valid and that
/// concurrent calls use disjoint row groups.
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *ptr.add((i * eighth + r) * num_ntts + lane);
            }
            for i in 0..4 {
                butterfly(&mut values, i, i + 4, twiddles[0]);
            }
            for s in 0..2 {
                for i in 0..2 {
                    butterfly(&mut values, 4 * s + i, 4 * s + i + 2, twiddles[1 + s]);
                }
            }
            for s in 0..4 {
                butterfly(&mut values, 2 * s, 2 * s + 1, twiddles[3 + s]);
            }
            for (i, value) in values.iter().enumerate() {
                *ptr.add((i * eighth + r) * num_ntts + lane) = *value;
            }
        }
    }
}

/// # Safety
/// The caller guarantees that every selected source and destination row is
/// valid, source and destination do not overlap, and concurrent calls write
/// disjoint destination row groups.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(super) unsafe fn butterfly_fused_3layer_row_from(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *src.add((i * eighth + r) * num_ntts + lane);
            }
            for i in 0..4 {
                butterfly(&mut values, i, i + 4, twiddles[0]);
            }
            for s in 0..2 {
                for i in 0..2 {
                    butterfly(&mut values, 4 * s + i, 4 * s + i + 2, twiddles[1 + s]);
                }
            }
            for s in 0..4 {
                butterfly(&mut values, 2 * s, 2 * s + 1, twiddles[3 + s]);
            }
            for (i, value) in values.iter().enumerate() {
                *dst.add((i * eighth + r) * num_ntts + lane) = *value;
            }
        }
    }
}

/// # Safety
/// The caller guarantees that every selected source and destination row is
/// valid, source and destination do not overlap, and concurrent calls write
/// disjoint destination row groups.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(super) unsafe fn butterfly_fused_3layer_row_from_sparse(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 4],
) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    // Breadth-first non-zero twiddles for block zero:
    // [layer2-right, layer3 blocks 1, 2, 3]. The seven omitted butterflies
    // have t=0, so `(u, v)` becomes `(u, v+u)` without a field multiply.
    let [t_l2_right, t_l3_1, t_l3_2, t_l3_3] = *twiddles;
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *src.add((i * eighth + r) * num_ntts + lane);
            }

            // Layer 1: all four twiddles are zero.
            values[4] += values[0];
            values[5] += values[1];
            values[6] += values[2];
            values[7] += values[3];

            // Layer 2: the left two butterflies have zero twiddle.
            values[2] += values[0];
            values[3] += values[1];
            butterfly(&mut values, 4, 6, t_l2_right);
            butterfly(&mut values, 5, 7, t_l2_right);

            // Layer 3: only the first butterfly has zero twiddle.
            values[1] += values[0];
            butterfly(&mut values, 2, 3, t_l3_1);
            butterfly(&mut values, 4, 5, t_l3_2);
            butterfly(&mut values, 6, 7, t_l3_3);

            for (i, value) in values.iter().enumerate() {
                *dst.add((i * eighth + r) * num_ntts + lane) = *value;
            }
        }
    }
}

/// # Safety
/// The caller guarantees the source/destination geometry, non-aliasing, and
/// disjoint-write contract documented by the architecture-neutral wrapper.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
fn seed_layer6_butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
    let new_u = values[u] + values[v] * twiddle;
    values[v] += new_u;
    values[u] = new_u;
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
const SEED_LAYER6_LANES: usize = 32;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
fn seed_layer6_fused_3(values: &mut [F128; 8], twiddles: &[F128; 7]) {
    for i in 0..4 {
        seed_layer6_butterfly(values, i, i + 4, twiddles[0]);
    }
    for s in 0..2 {
        for i in 0..2 {
            seed_layer6_butterfly(values, 4 * s + i, 4 * s + i + 2, twiddles[1 + s]);
        }
    }
    for s in 0..4 {
        seed_layer6_butterfly(values, 2 * s, 2 * s + 1, twiddles[3 + s]);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
fn seed_layer6_fused_3_sparse(values: &mut [F128; 8], twiddles: &[F128; 4]) {
    let [t_l2_right, t_l3_1, t_l3_2, t_l3_3] = *twiddles;

    values[4] += values[0];
    values[5] += values[1];
    values[6] += values[2];
    values[7] += values[3];

    values[2] += values[0];
    values[3] += values[1];
    seed_layer6_butterfly(values, 4, 6, t_l2_right);
    seed_layer6_butterfly(values, 5, 7, t_l2_right);

    values[1] += values[0];
    seed_layer6_butterfly(values, 2, 3, t_l3_1);
    seed_layer6_butterfly(values, 4, 5, t_l3_2);
    seed_layer6_butterfly(values, 6, 7, t_l3_3);
}

/// Fill a 32-lane 16x8 transposed post-layer-3 tile. Keeping this phase out of
/// line forces the 64 KiB scratch to materialize instead of letting LLVM
/// turn a larger fused expression into register spills.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn seed_layer6_stage_1_through_3(
    src: *const F128,
    scratch: *mut F128,
    sixty_fourth: usize,
    num_ntts: usize,
    r: usize,
    lane_base: usize,
    lane_count: usize,
    first_twiddles: &[[F128; 7]; 2],
    first_sparse_twiddles: &[F128; 4],
) {
    for column in 0..8 {
        for lane_in_tile in 0..lane_count {
            let lane = lane_base + lane_in_tile;
            let mut values = [F128::ZERO; 8];
            for (row, value) in values.iter_mut().enumerate() {
                let pos = (row * 8 + column) * sixty_fourth + r;
                // SAFETY: guaranteed by the caller's tile geometry.
                *value = unsafe { *src.add(pos * num_ntts + lane) };
            }
            seed_layer6_fused_3_sparse(&mut values, first_sparse_twiddles);
            for (row, value) in values.into_iter().enumerate() {
                let slot = (row * 8 + column) * SEED_LAYER6_LANES + lane_in_tile;
                // SAFETY: all live-lane tile slots are distinct and valid.
                unsafe { scratch.add(slot).write(value) };
            }

            // The second half consumes the same eight source values
            // immediately. Reloading keeps only one 8-value tree live, and
            // the source cache lines are hot from the sparse half above.
            for (row, value) in values.iter_mut().enumerate() {
                let pos = (row * 8 + column) * sixty_fourth + r;
                // SAFETY: guaranteed by the caller's tile geometry.
                *value = unsafe { *src.add(pos * num_ntts + lane) };
            }
            seed_layer6_fused_3(&mut values, &first_twiddles[1]);
            for (row, value) in values.into_iter().enumerate() {
                let slot = ((8 + row) * 8 + column) * SEED_LAYER6_LANES + lane_in_tile;
                // SAFETY: all live-lane tile slots are distinct and valid.
                unsafe { scratch.add(slot).write(value) };
            }
        }
    }
}

/// Drain one 32-lane transposed tile through layers 4--6 and write every
/// final codeword slot once. This phase is kept out of line for the same
/// register lifetime boundary as [`seed_layer6_stage_1_through_3`].
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn seed_layer6_stage_4_through_6(
    scratch: *const F128,
    dst: *mut F128,
    sixty_fourth: usize,
    num_ntts: usize,
    r: usize,
    lane_base: usize,
    lane_count: usize,
    second_twiddles: &[[F128; 7]; 16],
    second_sparse_twiddles: &[F128; 4],
) {
    for global_block in 0..16 {
        for lane_in_tile in 0..lane_count {
            let lane = lane_base + lane_in_tile;
            let mut values = [F128::ZERO; 8];
            for (column, value) in values.iter_mut().enumerate() {
                let slot = (global_block * 8 + column) * SEED_LAYER6_LANES + lane_in_tile;
                // SAFETY: stage 1--3 initialized every live-lane scratch slot.
                *value = unsafe { *scratch.add(slot) };
            }
            if global_block == 0 {
                seed_layer6_fused_3_sparse(&mut values, second_sparse_twiddles);
            } else {
                seed_layer6_fused_3(&mut values, &second_twiddles[global_block]);
            }
            for (column, value) in values.into_iter().enumerate() {
                let pos = (global_block * 8 + column) * sixty_fourth + r;
                // SAFETY: guaranteed by the caller's destination geometry.
                unsafe { *dst.add(pos * num_ntts + lane) = value };
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
// The 64 KiB scratch frame must remain an out-of-line range boundary. If
// ThinLTO inlines it into Rayon's recursive bridge helper, each split level
// consumes another 64 KiB and can overflow the worker's 2 MiB stack.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_6layer_rows_from(
    src: *const F128,
    dst: *mut F128,
    sixty_fourth: usize,
    num_ntts: usize,
    r_start: usize,
    r_end: usize,
    first_twiddles: &[[F128; 7]; 2],
    second_twiddles: &[[F128; 7]; 16],
    first_sparse_twiddles: &[F128; 4],
    second_sparse_twiddles: &[F128; 4],
) {
    debug_assert!(r_start < r_end && r_end <= sixty_fourth);
    // This scratch is intentionally uninitialized. Stage 1--3 writes every
    // live-lane slot before stage 4--6 reads it, so no tile zero fill occurs.
    let mut scratch = [core::mem::MaybeUninit::<F128>::uninit(); 128 * SEED_LAYER6_LANES];
    let scratch_ptr = scratch.as_mut_ptr().cast::<F128>();
    for r in r_start..r_end {
        for lane_base in (0..num_ntts).step_by(SEED_LAYER6_LANES) {
            let lane_count = (num_ntts - lane_base).min(SEED_LAYER6_LANES);
            unsafe {
                seed_layer6_stage_1_through_3(
                    src,
                    scratch_ptr,
                    sixty_fourth,
                    num_ntts,
                    r,
                    lane_base,
                    lane_count,
                    first_twiddles,
                    first_sparse_twiddles,
                );
                seed_layer6_stage_4_through_6(
                    scratch_ptr,
                    dst,
                    sixty_fourth,
                    num_ntts,
                    r,
                    lane_base,
                    lane_count,
                    second_twiddles,
                    second_sparse_twiddles,
                );
            }
        }
    }
}

#[inline]
pub(super) fn butterfly_fused_4layer(values: &mut [F128; 16], twiddles: &[F128; 15]) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 16], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    for i in 0..8 {
        butterfly(values, i, i + 8, twiddles[0]);
    }
    for s in 0..2 {
        for i in 0..4 {
            butterfly(values, 8 * s + i, 8 * s + i + 4, twiddles[1 + s]);
        }
    }
    for s in 0..4 {
        for i in 0..2 {
            butterfly(values, 4 * s + i, 4 * s + i + 2, twiddles[3 + s]);
        }
    }
    for s in 0..8 {
        butterfly(values, 2 * s, 2 * s + 1, twiddles[7 + s]);
    }
}

/// # Safety
/// The caller guarantees that every selected row and lane is valid and that
/// concurrent calls use disjoint row groups.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
)))]
pub(super) unsafe fn butterfly_fused_4layer_row(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 16];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *ptr.add((i * sixteenth + r) * num_ntts + lane);
            }
            butterfly_fused_4layer(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *ptr.add((i * sixteenth + r) * num_ntts + lane) = *value;
            }
        }
    }
}

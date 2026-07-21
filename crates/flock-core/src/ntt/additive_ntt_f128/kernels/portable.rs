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
    active_lanes: usize,
) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    debug_assert!(active_lanes <= num_ntts);
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..active_lanes {
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
/// valid, source and destination are either identical or do not overlap, and
/// concurrent calls write disjoint destination row groups.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(super) unsafe fn butterfly_fused_3layer_row_from(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
    active_lanes: usize,
) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    debug_assert!(active_lanes <= num_ntts);
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..active_lanes {
            let mut values = [F128::ZERO; 8];
            // Complete all reads for this lane before any write, which also
            // permits the exact in-place use from the block-zero fast path.
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
/// valid, source and destination are either identical or do not overlap, and
/// concurrent calls write disjoint destination row groups.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(super) unsafe fn butterfly_fused_3layer_row_from_sparse(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 4],
    active_lanes: usize,
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
    debug_assert!(active_lanes <= num_ntts);
    unsafe {
        for lane in 0..active_lanes {
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

/// Finish a fused three-layer group whose third layer is the deepest NTT
/// layer. The caller has already transformed lanes `0..dense_lanes` normally.
/// In the remaining lanes every odd input row is zero, so the first two layers
/// operate only on rows 0, 2, 4, 6 and the final `(even, odd=0)` butterflies
/// are exact copies with no multiplication.
///
/// # Safety
/// The caller guarantees one complete eight-row group (`eighth == 1`) and
/// valid lanes `dense_lanes..num_ntts`.
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row_final_odd_zero_tail(
    ptr: *mut F128,
    num_ntts: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 4], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    debug_assert!(dense_lanes <= num_ntts);
    unsafe {
        for lane in dense_lanes..num_ntts {
            let mut even = [
                *ptr.add(lane),
                *ptr.add(2 * num_ntts + lane),
                *ptr.add(4 * num_ntts + lane),
                *ptr.add(6 * num_ntts + lane),
            ];

            // Layers L and L+1 preserve row parity. Their even-row trees are
            // (0,4), (2,6), then (0,2), (4,6), respectively.
            butterfly(&mut even, 0, 2, twiddles[0]);
            butterfly(&mut even, 1, 3, twiddles[0]);
            butterfly(&mut even, 0, 1, twiddles[1]);
            butterfly(&mut even, 2, 3, twiddles[2]);

            for (i, value) in even.iter().copied().enumerate() {
                let even_row = 2 * i;
                *ptr.add(even_row * num_ntts + lane) = value;
                *ptr.add((even_row + 1) * num_ntts + lane) = value;
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

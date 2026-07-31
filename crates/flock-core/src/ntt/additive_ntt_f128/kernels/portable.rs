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

/// Fused three-layer (radix-8) butterfly on one lane's 8 values.
///
/// `twiddles` holds 7 values in the same breadth-first order
/// [`butterfly_fused_4layer`] uses, truncated by one level: `[0]` is layer L
/// (shared by all 4 pairs), `[1..3]` layer L+1 (one per half), `[3..7]`
/// layer L+2 (one per quarter).
#[inline]
pub(super) fn butterfly_fused_3layer(values: &mut [F128; 8], twiddles: &[F128; 7]) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    for i in 0..4 {
        butterfly(values, i, i + 4, twiddles[0]);
    }
    for s in 0..2 {
        for i in 0..2 {
            butterfly(values, 4 * s + i, 4 * s + i + 2, twiddles[1 + s]);
        }
    }
    for s in 0..4 {
        butterfly(values, 2 * s, 2 * s + 1, twiddles[3 + s]);
    }
}

/// Fused three-layer butterfly for the root block, whose left-spine
/// twiddles are identically zero: `twiddles[0] == twiddles[1] ==
/// twiddles[3] == 0`.
///
/// A zero-twiddle forward butterfly maps `(u, v)` to `(u, v + u)`, so the
/// seven butterflies on that spine need only XORs. The remaining five use
/// the ordinary field multiply. In particular, `values[0]` is unchanged by
/// all three layers.
#[inline]
pub(super) fn butterfly_fused_3layer_zero_root(values: &mut [F128; 8], twiddles: &[F128; 7]) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    // Layer L, t[0] = 0: four XOR-only butterflies.
    for i in 0..4 {
        let u = values[i];
        values[i + 4] += u;
    }

    // Layer L+1. The top-half twiddle t[1] is zero; t[2] is general.
    for i in 0..2 {
        let u = values[i];
        values[i + 2] += u;
    }
    butterfly(values, 4, 6, twiddles[2]);
    butterfly(values, 5, 7, twiddles[2]);

    // Layer L+2. The first quarter's twiddle t[3] is zero.
    let u = values[0];
    values[1] += u;
    butterfly(values, 2, 3, twiddles[4]);
    butterfly(values, 4, 5, twiddles[5]);
    butterfly(values, 6, 7, twiddles[6]);
}

/// Process one fused-three-layer row group across every interleaved lane.
///
/// Eight row streams at stride `eighth * num_ntts` elements. Every such
/// stride is a multiple of the L1 set-repeat period at these shapes, so all
/// eight streams map onto the same set — exactly 8 ways on an 8-way L1D, the
/// widest fusion that still fits. Radix-16 would demand 16 ways and thrash.
///
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
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *ptr.add((i * eighth + r) * num_ntts + lane);
            }
            butterfly_fused_3layer(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *ptr.add((i * eighth + r) * num_ntts + lane) = *value;
            }
        }
    }
}

/// Root-block specialization of [`butterfly_fused_3layer_row`].
///
/// The caller additionally guarantees that `twiddles[0]`, `twiddles[1]`, and
/// `twiddles[3]` are zero. Row stream zero is not written back because the
/// specialized butterfly proves it is unchanged.
///
/// # Safety
/// The caller guarantees that every selected row and lane is valid and that
/// concurrent calls use disjoint row groups.
pub(super) unsafe fn butterfly_fused_3layer_zero_root_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *ptr.add((i * eighth + r) * num_ntts + lane);
            }
            butterfly_fused_3layer_zero_root(&mut values, twiddles);
            // values[0] is unchanged, so leave its cache line clean.
            for (i, value) in values.iter().enumerate().skip(1) {
                *ptr.add((i * eighth + r) * num_ntts + lane) = *value;
            }
        }
    }
}

/// Pass-1-from-message row: load the eight source rows once from `msg`, then
/// write the zero-root (block 0) outputs to `out0` and the generic (block 1)
/// outputs to `out1`. Used by the ranked L0 commit to run the first radix-8
/// pass directly from the packed message: after the trivial rate-1/2 layer,
/// both codeword halves hold the same replica of `msg`, so one source read
/// serves both blocks and the replica copies are never materialized.
///
/// Unlike the in-place zero-root kernel, row stream 0 of `out0` IS written
/// (the destination is uninitialized; the unchanged value still has to land).
///
/// # Safety
/// The caller guarantees every selected row and lane is valid in all three
/// buffers and that concurrent calls use disjoint row groups. `tw0[0]`,
/// `tw0[1]`, and `tw0[3]` must be zero.
pub(super) unsafe fn butterfly_fused_3layer_pass1_from_message_row(
    msg: *const F128,
    out0: *mut F128,
    out1: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    tw0: &[F128; 7],
    tw1: &[F128; 7],
) {
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *msg.add((i * eighth + r) * num_ntts + lane);
            }
            let mut zero_root = values;
            butterfly_fused_3layer_zero_root(&mut zero_root, tw0);
            for (i, value) in zero_root.iter().enumerate() {
                *out0.add((i * eighth + r) * num_ntts + lane) = *value;
            }
            butterfly_fused_3layer(&mut values, tw1);
            for (i, value) in values.iter().enumerate() {
                *out1.add((i * eighth + r) * num_ntts + lane) = *value;
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

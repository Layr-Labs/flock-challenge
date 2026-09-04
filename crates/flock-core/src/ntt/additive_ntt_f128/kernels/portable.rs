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

/// Rate-1/2 first-pass row kernel: load the radix-8 row group from `src`
/// once, evaluate both layer-1 blocks' fused-3 butterflies — the zero-root
/// set to `dst0`, the general set to `dst1`. Both destinations hold stale
/// bytes on entry; every selected slot is written.
///
/// # Safety
/// Row/lane geometry valid for all three pointers, disjoint row groups
/// across concurrent calls, and `t_zero[0] == t_zero[1] == t_zero[3] == 0`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_3layer_dual_from_src_row(
    src: *const F128,
    dst0: *mut F128,
    dst1: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    t_zero: &[F128; 7],
    t_gen: &[F128; 7],
) {
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..num_ntts {
            let mut loaded = [F128::ZERO; 8];
            for (i, value) in loaded.iter_mut().enumerate() {
                *value = *src.add((i * eighth + r) * num_ntts + lane);
            }
            let mut v0 = loaded;
            butterfly_fused_3layer_zero_root(&mut v0, t_zero);
            for (i, value) in v0.iter().enumerate() {
                *dst0.add((i * eighth + r) * num_ntts + lane) = *value;
            }
            let mut v1 = loaded;
            butterfly_fused_3layer(&mut v1, t_gen);
            for (i, value) in v1.iter().enumerate() {
                *dst1.add((i * eighth + r) * num_ntts + lane) = *value;
            }
        }
    }
}

/// Out-of-place fused-three-layer row used by the recursive commitment's
/// from-message first pass. `dst` may contain stale bytes; every selected
/// slot is initialized from the corresponding radix-8 row group in `src`.
///
/// # Safety
/// The caller supplies valid source/destination row geometry and ensures
/// concurrent calls own disjoint destination row groups.
pub(super) unsafe fn butterfly_fused_3layer_from_src_row(
    src: *const F128,
    dst: *mut F128,
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
                *value = *src.add((i * eighth + r) * num_ntts + lane);
            }
            butterfly_fused_3layer(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *dst.add((i * eighth + r) * num_ntts + lane) = *value;
            }
        }
    }
}

/// Zero-root specialization of [`butterfly_fused_3layer_from_src_row`].
///
/// # Safety
/// As the general form, plus `twiddles[0] == twiddles[1] == twiddles[3] == 0`.
pub(super) unsafe fn butterfly_fused_3layer_zero_root_from_src_row(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    // SAFETY: caller supplies the pointer geometry and zero-root contract.
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *src.add((i * eighth + r) * num_ntts + lane);
            }
            butterfly_fused_3layer_zero_root(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *dst.add((i * eighth + r) * num_ntts + lane) = *value;
            }
        }
    }
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
    lanes: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..lanes {
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
    lanes: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..lanes {
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

use crate::field::F128;

#[inline]
pub(super) fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    butterfly_row_pair_gen::<false>(top, bot, twiddle)
}

/// `v * t` where `LOW` asserts `t.hi == 0`. Monomorphized so the choice is
/// resolved at compile time and no branch enters the lane loop.
#[inline(always)]
fn mul_t<const LOW: bool>(v: F128, t: F128) -> F128 {
    if LOW {
        crate::field::gf2_128::mul_low_rhs(v, t)
    } else {
        v * t
    }
}

#[inline]
pub(super) fn butterfly_row_pair_gen<const LOW: bool>(
    top: &mut [F128],
    bot: &mut [F128],
    twiddle: F128,
) {
    for lane in 0..top.len() {
        let v = bot[lane];
        let new_u = top[lane] + mul_t::<LOW>(v, twiddle);
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
    butterfly_fused_2layer_gen::<false, false>(a, b, c, d, t_outer, t_inner_a, t_inner_b)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn butterfly_fused_2layer_gen<const OUTER_LOW: bool, const INNER_LOW: bool>(
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
        let na = xa + mul_t::<OUTER_LOW>(xc, t_outer);
        xc += na;
        xa = na;
        let nb = xb + mul_t::<OUTER_LOW>(xd, t_outer);
        xd += nb;
        xb = nb;
        let na2 = xa + mul_t::<INNER_LOW>(xb, t_inner_a);
        xb += na2;
        xa = na2;
        let nc2 = xc + mul_t::<INNER_LOW>(xd, t_inner_b);
        xd += nc2;
        xc = nc2;
        a[lane] = xa;
        b[lane] = xb;
        c[lane] = xc;
        d[lane] = xd;
    }
}

/// # Safety
/// The caller guarantees that every selected source and destination row is
/// valid, source and destination do not overlap, and concurrent calls write
/// disjoint destination row groups.
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 3],
) {
    // SAFETY: forwarded caller contract; identical geometry on both sides.
    unsafe {
        butterfly_fused_2layer_row_from_geo(src, quarter, r, dst, quarter, r, num_ntts, twiddles)
    }
}

/// [`butterfly_fused_2layer_row_from`] with independent source and
/// destination row geometry: source rows `(i·src_quarter + src_r)`,
/// destination rows `(i·dst_quarter + dst_r)`, `i ∈ 0..4`.
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_2layer_row_from_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    twiddles: &[F128; 3],
) {
    let [t_outer, t_inner_a, t_inner_b] = *twiddles;
    unsafe {
        for lane in 0..num_ntts {
            let mut a = *src.add(src_r * num_ntts + lane);
            let mut b = *src.add((src_quarter + src_r) * num_ntts + lane);
            let mut c = *src.add((2 * src_quarter + src_r) * num_ntts + lane);
            let mut d = *src.add((3 * src_quarter + src_r) * num_ntts + lane);

            let new_a = a + c * t_outer;
            c += new_a;
            a = new_a;
            let new_b = b + d * t_outer;
            d += new_b;
            b = new_b;

            let new_a = a + b * t_inner_a;
            b += new_a;
            a = new_a;
            let new_c = c + d * t_inner_b;
            d += new_c;
            c = new_c;

            *dst.add(dst_r * num_ntts + lane) = a;
            *dst.add((dst_quarter + dst_r) * num_ntts + lane) = b;
            *dst.add((2 * dst_quarter + dst_r) * num_ntts + lane) = c;
            *dst.add((3 * dst_quarter + dst_r) * num_ntts + lane) = d;
        }
    }
}

/// # Safety
/// The caller guarantees that every selected source and destination row is
/// valid, source and destination do not overlap, and concurrent calls write
/// disjoint destination row groups.
#[cfg(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
))]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    r: usize,
    right_twiddle: F128,
) {
    // SAFETY: forwarded caller contract; identical geometry on both sides.
    unsafe {
        butterfly_fused_2layer_row_from_sparse_geo(
            src,
            quarter,
            r,
            dst,
            quarter,
            r,
            num_ntts,
            right_twiddle,
        )
    }
}

/// [`butterfly_fused_2layer_row_from_sparse`] with independent source and
/// destination row geometry (see [`butterfly_fused_2layer_row_from_geo`]).
///
/// # Safety
/// Same contract as [`butterfly_fused_2layer_row_from`].
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_2layer_row_from_sparse_geo(
    src: *const F128,
    src_quarter: usize,
    src_r: usize,
    dst: *mut F128,
    dst_quarter: usize,
    dst_r: usize,
    num_ntts: usize,
    right_twiddle: F128,
) {
    unsafe {
        for lane in 0..num_ntts {
            let a = *src.add(src_r * num_ntts + lane);
            let mut b = *src.add((src_quarter + src_r) * num_ntts + lane);
            let mut c = *src.add((2 * src_quarter + src_r) * num_ntts + lane);
            let mut d = *src.add((3 * src_quarter + src_r) * num_ntts + lane);

            // Layer 1 and the left layer-2 butterfly have zero twiddle.
            c += a;
            d += b;
            b += a;
            let new_c = c + d * right_twiddle;
            d += new_c;
            c = new_c;

            *dst.add(dst_r * num_ntts + lane) = a;
            *dst.add((dst_quarter + dst_r) * num_ntts + lane) = b;
            *dst.add((2 * dst_quarter + dst_r) * num_ntts + lane) = c;
            *dst.add((3 * dst_quarter + dst_r) * num_ntts + lane) = d;
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
    active_lanes: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..active_lanes {
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

/// Fused three-layer butterfly network on EIGHT rows, in registers.
///
/// `values[i]` is row `i` of the group; layer `L` pairs `(i, i+4)` with
/// `twiddles[0]`, layer `L+1` pairs `(4s+i, 4s+i+2)` with `twiddles[1+s]`,
/// and layer `L+2` pairs `(2s, 2s+1)` with `twiddles[3+s]` — the same
/// (block, twiddle) convention [`butterfly_fused_4layer`] uses, one level
/// shallower.
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

/// [`butterfly_fused_3layer`] specialized for the published zero tail: every
/// ODD row of the group is zero in this lane, so both layer-`L` and
/// layer-`L+1` butterflies that pair two odd rows are 0⊕0 and vanish, and
/// every layer-`L+2` butterfly degenerates to `v[2s+1] ← v[2s]` (its
/// multiply reads a zero operand). Four multiplies instead of twelve, and
/// only the four even rows have to be read.
#[inline]
pub(super) fn butterfly_fused_3layer_zero_odd(values: &mut [F128; 8], twiddles: &[F128; 7]) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    butterfly(values, 0, 4, twiddles[0]);
    butterfly(values, 2, 6, twiddles[0]);
    butterfly(values, 0, 2, twiddles[1]);
    butterfly(values, 4, 6, twiddles[2]);
    for s in 0..4 {
        values[2 * s + 1] = values[2 * s];
    }
}

/// Process one fused-three-layer group of eight CONSECUTIVE rows.
///
/// `ptr` addresses row 0; row `i` starts at `i · num_ntts`. Lanes
/// `0..dense_lanes` run the full network; lanes `dense_lanes..num_ntts` run
/// [`butterfly_fused_3layer_zero_odd`].
///
/// # Safety
/// The caller must ensure the eight rows are valid and disjoint from any
/// group processed concurrently, `dense_lanes <= num_ntts`, and that rows
/// 1, 3, 5 and 7 are zero on lanes `dense_lanes..num_ntts`.
#[cfg(any(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )),
    test
))]
pub(super) unsafe fn butterfly_fused_3layer_rows(
    ptr: *mut F128,
    num_ntts: usize,
    dense_lanes: usize,
    twiddles: &[F128; 7],
) {
    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *ptr.add(i * num_ntts + lane);
            }
            if lane < dense_lanes {
                butterfly_fused_3layer(&mut values, twiddles);
            } else {
                butterfly_fused_3layer_zero_odd(&mut values, twiddles);
            }
            for (i, value) in values.iter().enumerate() {
                *ptr.add(i * num_ntts + lane) = *value;
            }
        }
    }
}

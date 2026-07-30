use crate::field::F128;

/// Process two butterflies at a time within a block sharing one twiddle.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert!(half >= 2);
    debug_assert_eq!(chunk.len(), 2 * half);
    let mut idx0 = 0;
    while idx0 < half {
        let idx1 = idx0 + half;
        let u_a = chunk[idx0];
        let v_a = chunk[idx1];
        let u_b = chunk[idx0 + 1];
        let v_b = chunk[idx1 + 1];

        // SAFETY: caller guarantees the aes target feature.
        let product = unsafe { ghash_mul_vec2_neon([twiddle, twiddle], [v_a, v_b]) };
        let new_u_a = F128 {
            lo: u_a.lo ^ product[0].lo,
            hi: u_a.hi ^ product[0].hi,
        };
        let new_u_b = F128 {
            lo: u_b.lo ^ product[1].lo,
            hi: u_b.hi ^ product[1].hi,
        };
        let new_v_a = F128 {
            lo: v_a.lo ^ new_u_a.lo,
            hi: v_a.hi ^ new_u_a.hi,
        };
        let new_v_b = F128 {
            lo: v_b.lo ^ new_u_b.lo,
            hi: v_b.hi ^ new_u_b.hi,
        };

        chunk[idx0] = new_u_a;
        chunk[idx1] = new_v_a;
        chunk[idx0 + 1] = new_u_b;
        chunk[idx1 + 1] = new_v_b;
        idx0 += 2;
    }
}

/// Process the single pair in each of two adjacent blocks with distinct
/// twiddles.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block_pair(chunk: &mut [F128], t_a: F128, t_b: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert_eq!(chunk.len(), 4);
    let u_a = chunk[0];
    let v_a = chunk[1];
    let u_b = chunk[2];
    let v_b = chunk[3];

    // SAFETY: caller guarantees the aes target feature.
    let product = unsafe { ghash_mul_vec2_neon([t_a, t_b], [v_a, v_b]) };
    let new_u_a = F128 {
        lo: u_a.lo ^ product[0].lo,
        hi: u_a.hi ^ product[0].hi,
    };
    let new_u_b = F128 {
        lo: u_b.lo ^ product[1].lo,
        hi: u_b.hi ^ product[1].hi,
    };
    let new_v_a = F128 {
        lo: v_a.lo ^ new_u_a.lo,
        hi: v_a.hi ^ new_u_a.hi,
    };
    let new_v_b = F128 {
        lo: v_b.lo ^ new_u_b.lo,
        hi: v_b.hi ^ new_u_b.hi,
    };

    chunk[0] = new_u_a;
    chunk[1] = new_v_a;
    chunk[2] = new_u_b;
    chunk[3] = new_v_b;
}

/// XOR two field elements. `F128` addition is bitwise XOR, so this is the
/// field `+` written in terms of the raw words the NEON path keeps in GPRs.
#[inline(always)]
fn xor(a: F128, b: F128) -> F128 {
    F128 {
        lo: a.lo ^ b.lo,
        hi: a.hi ^ b.hi,
    }
}

/// Fused three-layer (radix-8) butterfly on one lane's 8 values, batching the
/// independent twiddle products of each layer into `ghash_mul_vec2_neon`.
///
/// Mirrors `portable::butterfly_fused_3layer` exactly: layer L pairs `(i, i+4)`
/// with `twiddles[0]`, layer L+1 pairs `(4s+i, 4s+i+2)` with `twiddles[1 + s]`,
/// layer L+2 pairs `(2s, 2s+1)` with `twiddles[3 + s]`.
///
/// Within one layer the four sub-butterflies touch four disjoint index pairs,
/// so evaluating them in a different order — here, two at a time — computes
/// exactly the same products and the same XOR network. `ghash_mul_vec2_neon`
/// is bit-identical to the scalar `Mul` (both are the full GHASH product,
/// only the reduction is vectorised), so the result is bit-identical to the
/// portable kernel.
///
/// Pairing is *within* one lane rather than across two lanes: each of the
/// three layers has four mutually independent products, which is already
/// enough to feed two `ghash_mul_vec2_neon` calls per layer (four independent
/// PMULL chains, matching the two PMULL units). Carrying two lanes would need
/// 16 live `F128` = 32 general-purpose registers, one more than AArch64 has,
/// so it would spill on every layer for no extra batching.
///
/// Values stay in general-purpose registers as `F128` word pairs. PMULL takes
/// GPR operands, so parking them in `q` registers would only add `fmov`
/// crossings for every multiply operand while saving cheaper `eor`s.
///
/// # Safety
/// Requires the `aes` target feature.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn fused_3layer_lane(v: &mut [F128; 8], t: &[F128; 7]) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    // SAFETY: this function carries the `aes` target feature.
    unsafe {
        // Layer L: (0,4) (1,5) (2,6) (3,7), all with t[0].
        let p01 = ghash_mul_vec2_neon([t[0], t[0]], [v[4], v[5]]);
        let p23 = ghash_mul_vec2_neon([t[0], t[0]], [v[6], v[7]]);
        let n0 = xor(v[0], p01[0]);
        let n1 = xor(v[1], p01[1]);
        let n2 = xor(v[2], p23[0]);
        let n3 = xor(v[3], p23[1]);
        v[4] = xor(v[4], n0);
        v[5] = xor(v[5], n1);
        v[6] = xor(v[6], n2);
        v[7] = xor(v[7], n3);
        v[0] = n0;
        v[1] = n1;
        v[2] = n2;
        v[3] = n3;

        // Layer L+1: (0,2) (1,3) with t[1]; (4,6) (5,7) with t[2].
        let q_lo = ghash_mul_vec2_neon([t[1], t[1]], [v[2], v[3]]);
        let q_hi = ghash_mul_vec2_neon([t[2], t[2]], [v[6], v[7]]);
        let m0 = xor(v[0], q_lo[0]);
        let m1 = xor(v[1], q_lo[1]);
        let m4 = xor(v[4], q_hi[0]);
        let m5 = xor(v[5], q_hi[1]);
        v[2] = xor(v[2], m0);
        v[3] = xor(v[3], m1);
        v[6] = xor(v[6], m4);
        v[7] = xor(v[7], m5);
        v[0] = m0;
        v[1] = m1;
        v[4] = m4;
        v[5] = m5;

        // Layer L+2: (0,1) t[3], (2,3) t[4], (4,5) t[5], (6,7) t[6].
        let s_lo = ghash_mul_vec2_neon([t[3], t[4]], [v[1], v[3]]);
        let s_hi = ghash_mul_vec2_neon([t[5], t[6]], [v[5], v[7]]);
        let k0 = xor(v[0], s_lo[0]);
        let k2 = xor(v[2], s_lo[1]);
        let k4 = xor(v[4], s_hi[0]);
        let k6 = xor(v[6], s_hi[1]);
        v[1] = xor(v[1], k0);
        v[3] = xor(v[3], k2);
        v[5] = xor(v[5], k4);
        v[7] = xor(v[7], k6);
        v[0] = k0;
        v[2] = k2;
        v[4] = k4;
        v[6] = k6;
    }
}

/// Process one fused-three-layer row group across every interleaved NTT lane.
///
/// Touches exactly the 8 row streams `{i * eighth + r}` that the portable
/// kernel touches — the pairing that feeds `ghash_mul_vec2_neon` is *inside*
/// one lane's 8 values, so it adds no live stream. That keeps the group at 8
/// concurrent streams, which is the widest an 8-way L1D admits at these
/// strides.
///
/// # Safety
/// Requires the `aes` target feature. The caller guarantees that every
/// selected row and lane is in bounds for `ptr` and that concurrent calls use
/// disjoint row groups (distinct `r`), so the writes never alias.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    // SAFETY: caller supplies the pointer geometry and disjointness contract;
    // `base` is row `r` of the group and `stride` is the element distance
    // between consecutive rows of the group, so every access below is one of
    // the `(i * eighth + r) * num_ntts + lane` slots the portable kernel uses.
    unsafe {
        let stride = eighth * num_ntts;
        let base = ptr.add(r * num_ntts);
        for lane in 0..num_ntts {
            let cell = base.add(lane);
            let mut v = [F128::ZERO; 8];
            for (i, value) in v.iter_mut().enumerate() {
                *value = *cell.add(i * stride);
            }
            fused_3layer_lane(&mut v, twiddles);
            for (i, value) in v.iter().enumerate() {
                *cell.add(i * stride) = *value;
            }
        }
    }
}

/// Fused two-layer (radix-4) butterfly on one lane's 4 values.
///
/// Mirrors `portable::butterfly_fused_2layer`'s body for a single lane. Both
/// products of a layer are independent (`xc·t_outer` / `xd·t_outer`, then
/// `xb·t_inner_a` / `xd·t_inner_b`), so each layer becomes one
/// `ghash_mul_vec2_neon`.
///
/// # Safety
/// Requires the `aes` target feature.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn fused_2layer_lane(
    xa: F128,
    xb: F128,
    xc: F128,
    xd: F128,
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) -> [F128; 4] {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    // SAFETY: this function carries the `aes` target feature.
    unsafe {
        let outer = ghash_mul_vec2_neon([t_outer, t_outer], [xc, xd]);
        let na = xor(xa, outer[0]);
        let nb = xor(xb, outer[1]);
        let yc = xor(xc, na);
        let yd = xor(xd, nb);

        let inner = ghash_mul_vec2_neon([t_inner_a, t_inner_b], [nb, yd]);
        let na2 = xor(na, inner[0]);
        let nc2 = xor(yc, inner[1]);
        [na2, xor(nb, na2), nc2, xor(yd, nc2)]
    }
}

/// In-place fused two-layer butterfly over four interleaved rows.
///
/// Two lanes are unrolled per iteration so two independent
/// `ghash_mul_vec2_neon` chains are in flight; both lanes live in the same
/// four rows, so the live-stream count stays at 4.
///
/// # Safety
/// Requires the `aes` target feature.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    let lanes = a.len();
    let pairs = lanes & !1;
    let mut lane = 0;
    // SAFETY: this function carries the `aes` target feature.
    unsafe {
        while lane < pairs {
            let r0 = fused_2layer_lane(
                a[lane], b[lane], c[lane], d[lane], t_outer, t_inner_a, t_inner_b,
            );
            let r1 = fused_2layer_lane(
                a[lane + 1],
                b[lane + 1],
                c[lane + 1],
                d[lane + 1],
                t_outer,
                t_inner_a,
                t_inner_b,
            );
            a[lane] = r0[0];
            b[lane] = r0[1];
            c[lane] = r0[2];
            d[lane] = r0[3];
            a[lane + 1] = r1[0];
            b[lane + 1] = r1[1];
            c[lane + 1] = r1[2];
            d[lane + 1] = r1[3];
            lane += 2;
        }
        if lane < lanes {
            let r0 = fused_2layer_lane(
                a[lane], b[lane], c[lane], d[lane], t_outer, t_inner_a, t_inner_b,
            );
            a[lane] = r0[0];
            b[lane] = r0[1];
            c[lane] = r0[2];
            d[lane] = r0[3];
        }
    }
}

/// Out-of-place form of [`butterfly_fused_2layer`]; same arithmetic, four
/// disjoint destination rows.
///
/// # Safety
/// Requires the `aes` target feature.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_2layer_out_of_place(
    src_a: &[F128],
    src_b: &[F128],
    src_c: &[F128],
    src_d: &[F128],
    dst_a: &mut [F128],
    dst_b: &mut [F128],
    dst_c: &mut [F128],
    dst_d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    let lanes = src_a.len();
    let pairs = lanes & !1;
    let mut lane = 0;
    // SAFETY: this function carries the `aes` target feature.
    unsafe {
        while lane < pairs {
            let r0 = fused_2layer_lane(
                src_a[lane],
                src_b[lane],
                src_c[lane],
                src_d[lane],
                t_outer,
                t_inner_a,
                t_inner_b,
            );
            let r1 = fused_2layer_lane(
                src_a[lane + 1],
                src_b[lane + 1],
                src_c[lane + 1],
                src_d[lane + 1],
                t_outer,
                t_inner_a,
                t_inner_b,
            );
            dst_a[lane] = r0[0];
            dst_b[lane] = r0[1];
            dst_c[lane] = r0[2];
            dst_d[lane] = r0[3];
            dst_a[lane + 1] = r1[0];
            dst_b[lane + 1] = r1[1];
            dst_c[lane + 1] = r1[2];
            dst_d[lane + 1] = r1[3];
            lane += 2;
        }
        if lane < lanes {
            let r0 = fused_2layer_lane(
                src_a[lane],
                src_b[lane],
                src_c[lane],
                src_d[lane],
                t_outer,
                t_inner_a,
                t_inner_b,
            );
            dst_a[lane] = r0[0];
            dst_b[lane] = r0[1];
            dst_c[lane] = r0[2];
            dst_d[lane] = r0[3];
        }
    }
}

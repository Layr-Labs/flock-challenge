use crate::field::F128;
use core::arch::aarch64::*;
use core::mem::transmute;

#[cfg(target_feature = "sha3")]
#[inline(always)]
unsafe fn xor3_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    unsafe { veor3q_u64(a, b, c) }
}

#[cfg(not(target_feature = "sha3"))]
#[inline(always)]
unsafe fn xor3_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    unsafe { veorq_u64(a, veorq_u64(b, c)) }
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn pmull(a: u64, b: u64) -> uint64x2_t {
    unsafe { transmute::<u128, uint64x2_t>(vmull_p64(a, b)) }
}

/// Six-PMULL Binius multiply that never materializes an `F128` in a GPR.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn ghash_mul_binius_neon(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe {
        let zero = vdupq_n_u64(0);
        let t0 = pmull(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b));
        let t1a = pmull(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<1>(b));
        let t1b = pmull(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<0>(b));
        let t2 = pmull(vgetq_lane_u64::<1>(a), vgetq_lane_u64::<1>(b));
        let t1_cross = veorq_u64(t1a, t1b);

        let t2_shifted = vextq_u64::<1>(zero, t2);
        let t2_red = pmull(vgetq_lane_u64::<1>(t2), 0x87);
        let t1 = xor3_u64(t1_cross, t2_shifted, t2_red);

        let t1_shifted = vextq_u64::<1>(zero, t1);
        let t1_red = pmull(vgetq_lane_u64::<1>(t1), 0x87);
        xor3_u64(t0, t1_shifted, t1_red)
    }
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn butterfly(u: &mut uint64x2_t, v: &mut uint64x2_t, twiddle: uint64x2_t) {
    unsafe {
        let new_u = veorq_u64(*u, ghash_mul_binius_neon(*v, twiddle));
        *v = veorq_u64(*v, new_u);
        *u = new_u;
    }
}

/// Fused two-layer row butterfly, one interleaved lane at a time.
///
/// # Safety
/// Requires the `aes` target feature.
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
    unsafe {
        let outer = transmute::<F128, uint64x2_t>(t_outer);
        let inner_a = transmute::<F128, uint64x2_t>(t_inner_a);
        let inner_b = transmute::<F128, uint64x2_t>(t_inner_b);
        for lane in 0..a.len() {
            let mut xa = vld1q_u64(a.as_ptr().add(lane).cast::<u64>());
            let mut xb = vld1q_u64(b.as_ptr().add(lane).cast::<u64>());
            let mut xc = vld1q_u64(c.as_ptr().add(lane).cast::<u64>());
            let mut xd = vld1q_u64(d.as_ptr().add(lane).cast::<u64>());
            butterfly(&mut xa, &mut xc, outer);
            butterfly(&mut xb, &mut xd, outer);
            butterfly(&mut xa, &mut xb, inner_a);
            butterfly(&mut xc, &mut xd, inner_b);
            vst1q_u64(a.as_mut_ptr().add(lane).cast::<u64>(), xa);
            vst1q_u64(b.as_mut_ptr().add(lane).cast::<u64>(), xb);
            vst1q_u64(c.as_mut_ptr().add(lane).cast::<u64>(), xc);
            vst1q_u64(d.as_mut_ptr().add(lane).cast::<u64>(), xd);
        }
    }
}

/// Fused three-layer row butterfly, one interleaved lane at a time.
///
/// # Safety
/// Requires the `aes` target feature and the caller's row-geometry contract.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    unsafe {
        // Load twiddles via vld1q: `&[F128; 7]` is only 8-aligned, so a
        // reference transmute to the 16-aligned NEON type would be UB.
        let mut tw = [vdupq_n_u64(0); 7];
        for (dst, src) in tw.iter_mut().zip(twiddles.iter()) {
            *dst = vld1q_u64((src as *const F128).cast::<u64>());
        }
        let twiddles = &tw;
        for lane in 0..num_ntts {
            let mut values = [vdupq_n_u64(0); 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = vld1q_u64(ptr.add((i * eighth + r) * num_ntts + lane).cast::<u64>());
            }
            for i in 0..4 {
                let (top, bottom) = values.split_at_mut(4);
                butterfly(&mut top[i], &mut bottom[i], twiddles[0]);
            }
            for s in 0..2 {
                let base = 4 * s;
                for i in 0..2 {
                    let (left, right) = values[base..base + 4].split_at_mut(2);
                    butterfly(&mut left[i], &mut right[i], twiddles[1 + s]);
                }
            }
            for s in 0..4 {
                let (left, right) = values[2 * s..2 * s + 2].split_at_mut(1);
                butterfly(&mut left[0], &mut right[0], twiddles[3 + s]);
            }
            for (i, value) in values.iter().enumerate() {
                vst1q_u64(ptr.add((i * eighth + r) * num_ntts + lane).cast::<u64>(), *value);
            }
        }
    }
}

/// Root-block fused three-layer row butterfly, one interleaved lane at a time.
///
/// # Safety
/// Requires the `aes` target feature, the row-geometry contract, and zero
/// twiddles at indices 0, 1, and 3.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_zero_root_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    unsafe {
        // Load twiddles via vld1q: `&[F128; 7]` is only 8-aligned, so a
        // reference transmute to the 16-aligned NEON type would be UB.
        let mut tw = [vdupq_n_u64(0); 7];
        for (dst, src) in tw.iter_mut().zip(twiddles.iter()) {
            *dst = vld1q_u64((src as *const F128).cast::<u64>());
        }
        let twiddles = &tw;
        for lane in 0..num_ntts {
            let mut values = [vdupq_n_u64(0); 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = vld1q_u64(ptr.add((i * eighth + r) * num_ntts + lane).cast::<u64>());
            }
            for i in 0..4 {
                values[i + 4] = veorq_u64(values[i + 4], values[i]);
            }
            for i in 0..2 {
                values[i + 2] = veorq_u64(values[i + 2], values[i]);
            }
            {
                let (left, right) = values[4..8].split_at_mut(2);
                butterfly(&mut left[0], &mut right[0], twiddles[2]);
                butterfly(&mut left[1], &mut right[1], twiddles[2]);
            }
            values[1] = veorq_u64(values[1], values[0]);
            for s in 1..4 {
                let (left, right) = values[2 * s..2 * s + 2].split_at_mut(1);
                butterfly(&mut left[0], &mut right[0], twiddles[3 + s]);
            }
            for (i, value) in values.iter().enumerate().skip(1) {
                vst1q_u64(ptr.add((i * eighth + r) * num_ntts + lane).cast::<u64>(), *value);
            }
        }
    }
}

/// Fused two-layer butterfly specialized for three low-limb-only twiddles.
/// Two products are issued together at each stage, using four PMULLs instead
/// of twelve for the pair under the generic Binius field multiplier.
///
/// # Safety
/// Requires the `aes` target feature.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "aes")]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_low_twiddles(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    use crate::field::gf2_128::aarch64::ghash_mul_low_constants_vec2_neon;

    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), d.len());
    debug_assert_eq!(t_outer.hi, 0);
    debug_assert_eq!(t_inner_a.hi, 0);
    debug_assert_eq!(t_inner_b.hi, 0);

    for lane in 0..a.len() {
        let mut xa = a[lane];
        let mut xb = b[lane];
        let mut xc = c[lane];
        let mut xd = d[lane];

        // SAFETY: this function carries aes and all constants have zero high
        // limbs by the ranked gate and assertions above.
        let outer = unsafe {
            ghash_mul_low_constants_vec2_neon([t_outer, t_outer], [xc, xd])
        };
        xa += outer[0];
        xc += xa;
        xb += outer[1];
        xd += xb;

        let inner = unsafe {
            ghash_mul_low_constants_vec2_neon([t_inner_a, t_inner_b], [xb, xd])
        };
        xa += inner[0];
        xb += xa;
        xc += inner[1];
        xd += xc;

        a[lane] = xa;
        b[lane] = xb;
        c[lane] = xc;
        d[lane] = xd;
    }
}

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

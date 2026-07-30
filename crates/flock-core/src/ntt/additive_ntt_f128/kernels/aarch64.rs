use crate::field::F128;
use core::arch::aarch64::*;
use core::mem::transmute;

#[inline(always)]
unsafe fn pmull(a: u64, b: u64) -> uint64x2_t {
    // SAFETY: the function's target feature enables PMULL and both values are
    // 128-bit bit containers.
    unsafe { transmute::<u128, uint64x2_t>(vmull_p64(a, b)) }
}

/// Binius GHASH multiplication with both operands and the result kept in Q
/// registers. This is the same arithmetic as `F128::mul`, without crossing
/// through scalar structs between the twelve radix-8 butterflies.
#[inline(always)]
unsafe fn ghash_mul_q(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe {
        let zero = vdupq_n_u64(0);
        let a_lo = vgetq_lane_u64::<0>(a);
        let a_hi = vgetq_lane_u64::<1>(a);
        let b_lo = vgetq_lane_u64::<0>(b);
        let b_hi = vgetq_lane_u64::<1>(b);
        let mut t0 = pmull(a_lo, b_lo);
        let t1a = pmull(a_lo, b_hi);
        let t1b = pmull(a_hi, b_lo);
        let t2 = pmull(a_hi, b_hi);
        let mut t1 = veorq_u64(t1a, t1b);

        t1 = veorq_u64(t1, vextq_u64::<1>(zero, t2));
        t1 = veorq_u64(t1, pmull(vgetq_lane_u64::<1>(t2), 0x87));
        t0 = veorq_u64(t0, vextq_u64::<1>(zero, t1));
        veorq_u64(t0, pmull(vgetq_lane_u64::<1>(t1), 0x87))
    }
}

/// Q-register-native radix-8 row kernel for the fused top-layer NTT pass.
/// Seven twiddles and eight values stay live across all twelve butterflies.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    #[inline(always)]
    unsafe fn butterfly(values: &mut [uint64x2_t; 8], a: usize, b: usize, twiddle: uint64x2_t) {
        unsafe {
            let new_a = veorq_u64(values[a], ghash_mul_q(twiddle, values[b]));
            values[b] = veorq_u64(values[b], new_a);
            values[a] = new_a;
        }
    }

    unsafe {
        let tw: [uint64x2_t; 7] =
            std::array::from_fn(|i| vld1q_u64((&twiddles[i] as *const F128).cast::<u64>()));
        for lane in 0..num_ntts {
            let mut values: [uint64x2_t; 8] = std::array::from_fn(|i| {
                vld1q_u64(ptr.add((i * eighth + r) * num_ntts + lane).cast::<u64>())
            });
            for i in 0..4 {
                butterfly(&mut values, i, i + 4, tw[0]);
            }
            for half in 0..2 {
                for i in 0..2 {
                    butterfly(&mut values, 4 * half + i, 4 * half + i + 2, tw[1 + half]);
                }
            }
            for quarter in 0..4 {
                butterfly(&mut values, 2 * quarter, 2 * quarter + 1, tw[3 + quarter]);
            }
            for (i, value) in values.iter().enumerate() {
                vst1q_u64(
                    ptr.add((i * eighth + r) * num_ntts + lane).cast::<u64>(),
                    *value,
                );
            }
        }
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

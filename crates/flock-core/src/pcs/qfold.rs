//! Vector-resident fold-4 for the direct-AB materializer.
//!
//! Same mechanism class as the promoted NTT vector-resident kernels
//! (`de8d57e`, `85490cc2`): keep fold-4 operands and intermediates in
//! `uint64x2_t` across the three multiplies so values do not round-trip the
//! GPR↔NEON boundary on every `F128` multiply. Arithmetic is the same
//! four-PMULL / two-stage-reduction algorithm as `ghash_mul_binius`.
//!
//! Gate: `FLOCK_NO_NEON_FOLD4=1` restores the scalar `F128` fold chain in the
//! same binary. Non-AArch64 targets always take the scalar path.

use crate::field::F128;

/// Whether the vector-resident fold-4 path is enabled.
#[inline]
pub(crate) fn neon_fold4_enabled() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("FLOCK_NO_NEON_FOLD4").is_none())
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Scalar fold-4 reference.
#[inline(always)]
pub(crate) fn fold4_scalar(input: &[F128], slot: usize, r0: F128, r1: F128) -> F128 {
    let a0 = input[4 * slot];
    let a1 = input[4 * slot + 1];
    let a2 = input[4 * slot + 2];
    let a3 = input[4 * slot + 3];
    let low = a0 + r0 * (a0 + a1);
    let high = a2 + r0 * (a2 + a3);
    low + r1 * (low + high)
}

/// Fold one output slot; NEON on AArch64 when enabled, else scalar.
#[inline(always)]
pub(crate) fn fold4(input: &[F128], slot: usize, r0: F128, r1: F128) -> F128 {
    if neon_fold4_enabled() {
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { aarch64::fold4_neon(input, slot, r0, r1) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            unreachable!("vector-resident fold4 is AArch64-only");
        }
    }
    fold4_scalar(input, slot, r0, r1)
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn pmull_ll(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe {
            core::mem::transmute::<u128, uint64x2_t>(vmull_p64(
                vgetq_lane_u64::<0>(a),
                vgetq_lane_u64::<0>(b),
            ))
        }
    }

    #[inline(always)]
    unsafe fn pmull_87(x: u64) -> uint64x2_t {
        unsafe { core::mem::transmute::<u128, uint64x2_t>(vmull_p64(x, 0x87)) }
    }

    #[cfg(target_feature = "sha3")]
    #[inline(always)]
    unsafe fn xor3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
        unsafe { veor3q_u64(a, b, c) }
    }

    #[cfg(not(target_feature = "sha3"))]
    #[inline(always)]
    unsafe fn xor3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
        unsafe { veorq_u64(a, veorq_u64(b, c)) }
    }

    #[inline(always)]
    unsafe fn mul_q(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe {
            let zero = vdupq_n_u64(0);
            let t0 = pmull_ll(a, b);
            let t1a = core::mem::transmute::<u128, uint64x2_t>(vmull_p64(
                vgetq_lane_u64::<0>(a),
                vgetq_lane_u64::<1>(b),
            ));
            let t1b = core::mem::transmute::<u128, uint64x2_t>(vmull_p64(
                vgetq_lane_u64::<1>(a),
                vgetq_lane_u64::<0>(b),
            ));
            let t2 = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
                vreinterpretq_p64_u64(a),
                vreinterpretq_p64_u64(b),
            ));
            let t1_cross = veorq_u64(t1a, t1b);
            let t1 = xor3(
                t1_cross,
                vextq_u64::<1>(zero, t2),
                pmull_87(vgetq_lane_u64::<1>(t2)),
            );
            xor3(
                t0,
                vextq_u64::<1>(zero, t1),
                pmull_87(vgetq_lane_u64::<1>(t1)),
            )
        }
    }

    #[inline(always)]
    unsafe fn load_q(p: *const F128) -> uint64x2_t {
        unsafe { vld1q_u64(p as *const u64) }
    }

    #[inline(always)]
    unsafe fn store_q(p: *mut F128, v: uint64x2_t) {
        unsafe { vst1q_u64(p as *mut u64, v) }
    }

    #[inline(always)]
    pub(super) unsafe fn fold4_neon(input: &[F128], slot: usize, r0: F128, r1: F128) -> F128 {
        unsafe {
            let base = input.as_ptr().add(4 * slot);
            let a0 = load_q(base);
            let a1 = load_q(base.add(1));
            let a2 = load_q(base.add(2));
            let a3 = load_q(base.add(3));
            let rq0 = load_q(&r0 as *const F128);
            let rq1 = load_q(&r1 as *const F128);
            let s0 = veorq_u64(a0, a1);
            let low = veorq_u64(a0, mul_q(rq0, s0));
            let s1 = veorq_u64(a2, a3);
            let high = veorq_u64(a2, mul_q(rq0, s1));
            let s2 = veorq_u64(low, high);
            let out = veorq_u64(low, mul_q(rq1, s2));
            let mut result = F128::ZERO;
            store_q(&mut result as *mut F128, out);
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold4_scalar_matches_manual() {
        let mut state = 0xA11CE_u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            F128::new(state, state.rotate_left(17))
        };
        let input: Vec<F128> = (0..64).map(|_| rnd()).collect();
        let r0 = rnd();
        let r1 = rnd();
        for slot in 0..16 {
            let got = fold4_scalar(&input, slot, r0, r1);
            let a0 = input[4 * slot];
            let a1 = input[4 * slot + 1];
            let a2 = input[4 * slot + 2];
            let a3 = input[4 * slot + 3];
            let low = a0 + r0 * (a0 + a1);
            let high = a2 + r0 * (a2 + a3);
            assert_eq!(got, low + r1 * (low + high), "slot {slot}");
        }
    }

    #[test]
    fn fold4_dispatch_matches_scalar_on_this_host() {
        let mut state = 0xBEEF_u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(7);
            F128::new(state, !state)
        };
        let input: Vec<F128> = (0..128).map(|_| rnd()).collect();
        let r0 = rnd();
        let r1 = rnd();
        for slot in 0..32 {
            assert_eq!(
                fold4(&input, slot, r0, r1),
                fold4_scalar(&input, slot, r0, r1),
                "slot {slot}"
            );
        }
    }
}

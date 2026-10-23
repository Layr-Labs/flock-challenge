
#![allow(dead_code)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct DeferredAcc {
    pub lo: uint8x16_t,
    pub hi: uint8x16_t,
}

#[cfg(target_arch = "aarch64")]
impl DeferredAcc {
    #[inline]
    pub fn zero() -> Self {
        unsafe {
            DeferredAcc {
                lo: vdupq_n_u8(0),
                hi: vdupq_n_u8(0),
            }
        }
    }

    #[inline]
    pub fn mul_add(&mut self, a: uint8x16_t, b: uint8x16_t) {
        unsafe {
            let a_lo = vgetq_lane_u64(vreinterpretq_u64_u8(a), 0);
            let a_hi = vgetq_lane_u64(vreinterpretq_u64_u8(a), 1);
            let b_lo = vgetq_lane_u64(vreinterpretq_u64_u8(b), 0);
            let b_hi = vgetq_lane_u64(vreinterpretq_u64_u8(b), 1);
            let p00: uint8x16_t = vmull_p64(a_lo, b_lo);
            let p11: uint8x16_t = vmull_high_p64(vreinterpretq_u64_u8(a), vreinterpretq_u64_u8(b));
            let p01: uint8x16_t = vmull_p64(a_lo, b_hi);
            let p10: uint8x16_t = vmull_p64(a_hi, b_lo);
            let mid = veorq_u8(p01, p10);
            let mid_lo = vextq_u8(vdupq_n_u8(0), mid, 8);
            let mid_hi = vextq_u8(mid, vdupq_n_u8(0), 8);
            self.lo = veorq_u8(self.lo, veorq_u8(p00, mid_lo));
            self.hi = veorq_u8(self.hi, veorq_u8(p11, mid_hi));
        }
    }
}

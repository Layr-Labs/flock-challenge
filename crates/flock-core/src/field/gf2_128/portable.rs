#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use super::ghash_reduce;
use super::{F128, F256Unreduced};

#[inline]
pub fn clmul64(a: u64, b: u64) -> (u64, u64) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return (
            crate::gf2_kernels::gf2_kern_clmul64_lo(a, b),
            crate::gf2_kernels::gf2_kern_clmul64_hi(a, b),
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut lo: u64 = 0;
        let mut hi: u64 = 0;
        let mut i = 0;
        while i < 64 {
            if (a >> i) & 1 != 0 {
                lo ^= b << i;
                if i != 0 {
                    hi ^= b >> (64 - i);
                }
            }
            i += 1;
        }
        (lo, hi)
    }
}

pub fn ghash_mul_unreduced(a: F128, b: F128) -> F256Unreduced {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return F256Unreduced {
            r0: crate::gf2_kernels::gf2_kern_clmul64_lo(a.lo, b.lo),
            r1: crate::gf2_kernels::gf2_kern_ghash_mul_unred_r1(a.lo, a.hi, b.lo, b.hi),
            r2: crate::gf2_kernels::gf2_kern_ghash_mul_unred_r2(a.lo, a.hi, b.lo, b.hi),
            r3: crate::gf2_kernels::gf2_kern_clmul64_hi(a.hi, b.hi),
        };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let (ll_lo, ll_hi) = clmul64(a.lo, b.lo);
        let (lh_lo, lh_hi) = clmul64(a.lo, b.hi);
        let (hl_lo, hl_hi) = clmul64(a.hi, b.lo);
        let (hh_lo, hh_hi) = clmul64(a.hi, b.hi);
        let cr_lo = lh_lo ^ hl_lo;
        let cr_hi = lh_hi ^ hl_hi;
        F256Unreduced {
            r0: ll_lo,
            r1: ll_hi ^ cr_lo,
            r2: hh_lo ^ cr_hi,
            r3: hh_hi,
        }
    }
}

#[inline]
pub fn ghash_mul(a: F128, b: F128) -> F128 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return F128 {
            lo: crate::gf2_kernels::gf2_kern_ghash_mul_lo(a.lo, a.hi, b.lo, b.hi),
            hi: crate::gf2_kernels::gf2_kern_ghash_mul_hi(a.lo, a.hi, b.lo, b.hi),
        };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let u = ghash_mul_unreduced(a, b);
        ghash_reduce(u.r0, u.r1, u.r2, u.r3)
    }
}

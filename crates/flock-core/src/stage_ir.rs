#![allow(unused_variables, unused_mut, unused_parens, unused_unsafe, dead_code, non_snake_case, clippy::all)]
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_isolate_lowest_one(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 0u64;
        let v2: u64 = (0u64.wrapping_sub(v0));
        let v3: u64 = (v0 & v2);
        v3
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_clear_lowest_one(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 1u64;
        let v2: u64 = (v0.wrapping_sub(1u64));
        let v3: u64 = (v0 & v2);
        v3
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_monomial_basis128_lo(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 6u64;
        let v2: u64 = (v0.wrapping_shr((6u64 & 63) as u32));
        let v3: u64 = 1u64;
        let v4: u64 = (v2 & 1u64);
        let v5: u64 = (v4 ^ 1u64);
        let v6: u64 = 63u64;
        let v7: u64 = (v0 & 63u64);
        let v8: u64 = (v5.wrapping_shl((v7 & 63) as u32));
        v8
    }
}
#[cfg_attr(not(debug_assertions), inline(always))]
pub unsafe fn gf2_kern_monomial_basis128_hi(v0: u64) -> u64 {
    unsafe {
        let v1: u64 = 6u64;
        let v2: u64 = (v0.wrapping_shr((6u64 & 63) as u32));
        let v3: u64 = 1u64;
        let v4: u64 = (v2 & 1u64);
        let v5: u64 = 63u64;
        let v6: u64 = (v0 & 63u64);
        let v7: u64 = (v4.wrapping_shl((v6 & 63) as u32));
        v7
    }
}

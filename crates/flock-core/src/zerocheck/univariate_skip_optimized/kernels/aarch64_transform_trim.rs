// A constant polynomial stays constant under inverse interpolation and
// extension. Thus T(all-ones) = [1; 64], and GF(2)-linearity gives
// T(b) = [1; 64] XOR T(!b). In a guarded static-B word every fixed byte
// is 0xff, so !b has zero at exactly the omitted byte positions.

pub(crate) const ABV_TRANSFORM_TRIM: u32 = 1 << 17;

#[inline(always)]
fn transform_partial_word<const V: u32>(word: u64) -> u64 {
    if V & ABV_TRANSFORM_TRIM != 0 {
        !word
    } else {
        word
    }
}

#[inline(always)]
unsafe fn transform_partial_seed<const V: u32>(
    partial: *const u8,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        if V & ABV_TRANSFORM_TRIM != 0 {
            let one = vdupq_n_u8(1);
            (one, one, one, one)
        } else {
            (
                vld1q_u8(partial),
                vld1q_u8(partial.add(16)),
                vld1q_u8(partial.add(32)),
                vld1q_u8(partial.add(48)),
            )
        }
    }
}

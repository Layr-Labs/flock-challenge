//! Publication of the last two live witness words without a stage round trip.

use core::arch::aarch64::*;

/// Read once at the witness entry, before selecting a const-specialized driver.
/// There is no cached initialization or per-quad switch lookup.
pub(super) fn enabled() -> bool {
    std::env::var_os("FLOCK_NO_WITNESS_CONST").as_deref()
        != Some(std::ffi::OsStr::new("1"))
}

/// Publish four blocks' last two live words and two zero padding words.
/// The remaining padding must already be zero under the caller's complete
/// witness provenance receipt. Destination strides and offsets come from the
/// constraint layout, independently of the compression inputs.
#[inline(always)]
pub(super) unsafe fn store_tail_nt(
    first: uint32x4_t,
    last: uint32x4_t,
    dst: *mut u32,
) {
    const WORDS: usize = super::K / 32;
    const TAIL: usize = (super::USEFUL_BITS - 1) / 32 - 1;
    const {
        assert!(super::USEFUL_BITS % 32 != 0);
        assert!(TAIL + 4 <= WORDS);
    }
    unsafe {
        // ZIP joins the two words of each block; the high 64-bit halves
        // supply the adjacent block. Keep the non-temporal store flavor.
        core::arch::asm!(
            "zip1 {lo:v}.4s, {first:v}.4s, {last:v}.4s",
            "zip2 {hi:v}.4s, {first:v}.4s, {last:v}.4s",
            "movi {zero:v}.2d, #0",
            "stnp {lo:d}, {zero:d}, [{p0}]",
            "dup {tmp:v}.2d, {lo:v}.d[1]",
            "stnp {tmp:d}, {zero:d}, [{p1}]",
            "stnp {hi:d}, {zero:d}, [{p2}]",
            "dup {tmp:v}.2d, {hi:v}.d[1]",
            "stnp {tmp:d}, {zero:d}, [{p3}]",
            first = in(vreg) first,
            last = in(vreg) last,
            p0 = in(reg) dst.add(TAIL),
            p1 = in(reg) dst.add(WORDS + TAIL),
            p2 = in(reg) dst.add(2 * WORDS + TAIL),
            p3 = in(reg) dst.add(3 * WORDS + TAIL),
            lo = out(vreg) _,
            hi = out(vreg) _,
            zero = out(vreg) _,
            tmp = out(vreg) _,
            options(nostack, preserves_flags),
        );
    }
}

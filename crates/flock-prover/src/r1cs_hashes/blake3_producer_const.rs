//! Publish variable words while preserving token-owned boundary constants.

use super::{GS_BASE, K, USEFUL_BITS};

/// A publishes words 0..482; B publishes words 36..472. The omitted A
/// words 482..488 are padding zeros, and B words 32..36 are identity ones.
/// All other skipped words were already skipped by the chunk publisher.
///
/// Safety: four complete, disjoint source/destination blocks; destination
/// padding and identity constants must be certified by the layout token.
#[inline(always)]
pub(super) unsafe fn publish<const A: bool>(scratch: *const u32, dst: *mut u32) {
    const {
        assert!(K / 32 == 512);
        assert!(USEFUL_BITS.div_ceil(32) == 482);
        assert!(GS_BASE / 32 == 36);
    }
    for j in 0..4usize {
        unsafe {
            let start = if A { 0 } else { 32 };
            let src = scratch.add(j * (K / 32) + start);
            let out = dst.add(j * (K / 32) + start);
            if A {
                // The paired loop stops before the final two variable words.
                core::arch::asm!(
                    ".p2align 6",
                    "2:",
                    "ldp {a:q}, {b:q}, [{s}], #32",
                    "stnp {a:q}, {b:q}, [{d}]",
                    "add {d}, {d}, #32",
                    "subs {n}, {n}, #1",
                    "b.ne 2b",
                    "ldp {lo:w}, {hi:w}, [{s}]",
                    "stnp {lo:w}, {hi:w}, [{d}]",
                    s = inout(reg) src => _,
                    d = inout(reg) out => _,
                    n = inout(reg) 60usize => _,
                    a = out(vreg) _,
                    b = out(vreg) _,
                    lo = out(reg) _,
                    hi = out(reg) _,
                    options(nostack),
                );
            } else {
                // The first half-chunk is constant; copy its variable half.
                core::arch::asm!(
                    "ldp {a:d}, {b:d}, [{s}, #16]",
                    "stnp {a:d}, {b:d}, [{d}, #16]",
                    "add {s}, {s}, #32",
                    "add {d}, {d}, #32",
                    ".p2align 6",
                    "2:",
                    "ldp {a:q}, {b:q}, [{s}], #32",
                    "stnp {a:q}, {b:q}, [{d}]",
                    "add {d}, {d}, #32",
                    "subs {n}, {n}, #1",
                    "b.ne 2b",
                    s = inout(reg) src => _,
                    d = inout(reg) out => _,
                    n = inout(reg) 54usize => _,
                    a = out(vreg) _,
                    b = out(vreg) _,
                    options(nostack),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publication_preserves_constants() {
        let words = K / 32;
        let mut state = 0x2941_9ab7u32;
        for a in [false, true] {
            for _ in 0..8 {
                let mut scratch = vec![0u32; 4 * words];
                let mut dst = vec![0u32; 4 * words];
                for values in [&mut scratch, &mut dst] {
                    for value in values {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        *value = state;
                    }
                }
                // Model the complete layout constants certified by the token.
                for values in [&mut scratch, &mut dst] {
                    for block in values.chunks_exact_mut(words) {
                        block[482..].fill(0);
                        if !a {
                            block[..36].fill(u32::MAX);
                            block[472..481].fill(u32::MAX);
                            block[481] = (1 << (USEFUL_BITS % 32)) - 1;
                        }
                    }
                }
                let mut expected = dst.clone();
                for j in 0..4 {
                    let base = j * words;
                    let range = if a { base..base + 488 } else { base + 32..base + 472 };
                    expected[range.clone()].copy_from_slice(&scratch[range]);
                }
                unsafe {
                    if a {
                        publish::<true>(scratch.as_ptr(), dst.as_mut_ptr());
                    } else {
                        publish::<false>(scratch.as_ptr(), dst.as_mut_ptr());
                    }
                }
                assert_eq!(dst, expected);
            }
        }
    }
}

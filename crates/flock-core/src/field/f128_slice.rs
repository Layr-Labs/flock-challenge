//! Architecture-selected kernels over contiguous [`F128`] slices.

use super::F128;

#[cfg(any(
    test,
    not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ),
        all(target_arch = "aarch64", target_feature = "aes")
    ))
))]
mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod x86_64;

/// Non-temporal 32-byte store of two `F128` values to `dst[0..2]`.
///
/// One q-form `stnp` writes both values with a no-allocate hint, so a huge
/// write-only output stream reaches DRAM without read-for-ownership traffic
/// or LLC pollution. Only profitable when the destination is far larger than
/// the LLC and its next read happens after a Fiat–Shamir barrier — callers
/// gate on both (see the NT thresholds at the call sites).
///
/// # Safety
/// `dst` must be valid for writing 32 bytes. 32-byte alignment is not
/// architecturally required but callers keep it for full store throughput.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn nt_store_pair(dst: *mut F128, v0: F128, v1: F128) {
    unsafe {
        let q0: core::arch::aarch64::uint8x16_t = core::mem::transmute(v0);
        let q1: core::arch::aarch64::uint8x16_t = core::mem::transmute(v1);
        core::arch::asm!(
            "stnp {v0:q}, {v1:q}, [{d}]",
            v0 = in(vreg) q0,
            v1 = in(vreg) q1,
            d = in(reg) dst,
            options(nostack, preserves_flags)
        );
    }
}

/// Minimum per-buffer output length (in `F128`s) for the non-temporal store
/// paths: 2^22 × 16 B = 64 MB. Below this the stream can be LLC-resident and
/// `stnp` only costs (measured on M-series: NT stores on cache-reachable
/// data regress).
#[cfg(target_arch = "aarch64")]
pub(crate) const NT_STORE_MIN_F128: usize = 1 << 22;

/// Fused pair-fold of `(f, b)` with non-temporal output stores plus the next
/// sumcheck round's `(u_0, u_2)` message terms computed from the folded
/// registers. See the aarch64 kernel for the exact contract; value-identical
/// to `fold_pairs` on each array followed by the pairwise message loop.
///
/// # Safety
/// Output slices must satisfy the length contract (`f_out.len()` even,
/// inputs twice as long) and be valid for 32-byte writes at even indices.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(crate) unsafe fn fold_pairs_msg_nt(
    f_in: &[F128],
    b_in: &[F128],
    f_out: &mut [F128],
    b_out: &mut [F128],
    r: F128,
) -> (F128, F128) {
    // SAFETY: the cfg gate guarantees the aes feature; the kernel asserts the
    // slice geometry.
    unsafe { aarch64::fold_pairs_msg_nt(f_in, b_in, f_out, b_out, r) }
}

/// Fold adjacent pairs from `src` into `dst`, starting at pair `base`.
///
/// Computes `dst[t] = src[2j] * (1 + r) + src[2j + 1] * r`, where
/// `j = base + t`. Architecture selection is resolved at compile time.
#[inline]
pub(crate) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    assert!(
        base <= src.len() / 2 && dst.len() <= src.len() / 2 - base,
        "fold source must contain both elements for every destination pair"
    );

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the required target features and the
    // bounds check above guarantees both source elements for every output.
    unsafe {
        x86_64::fold_pairs(src, base, dst, r);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    // SAFETY: the cfg gate guarantees PMULL support through the aes feature;
    // the bounds check above guarantees both source elements for every output.
    unsafe {
        aarch64::fold_pairs(src, base, dst, r);
    }

    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    portable::fold_pairs(src, base, dst, r);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_fold_matches_portable_with_offset_and_tail() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let src: Vec<F128> = (0..30)
            .map(|_| F128 {
                lo: next(),
                hi: next(),
            })
            .collect();
        let r = F128 {
            lo: next(),
            hi: next(),
        };
        let mut expected = vec![F128::ZERO; 9];
        let mut actual = vec![F128::ZERO; 9];

        portable::fold_pairs(&src, 3, &mut expected, r);
        fold_pairs(&src, 3, &mut actual, r);

        assert_eq!(actual, expected);
    }
}

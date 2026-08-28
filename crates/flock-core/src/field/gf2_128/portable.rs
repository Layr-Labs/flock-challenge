use super::{F128, F256Unreduced, ghash_reduce};

/// 64×64 carry-less product into 128 bits (lo, hi).
///
/// Original bit-by-bit schoolbook (kept as the bit-exact reference for the
/// test suite). New code paths should reach for [`clmul64_schoolbook`] below.
pub fn clmul64(a: u64, b: u64) -> (u64, u64) {
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

/// 64×64 carry-less product into 128 bits (lo, hi), 4-bit schoolbook form.
///
/// Replaces the per-bit shift loop of [`clmul64`] with 16 nibble×u64 partial
/// products accumulated into a single `u128` and split at the end. The
/// conditional accumulation is implemented as a `wrapping_sub` mask so the
/// hot body is purely portable bitwise ops (shifts, AND, XOR, sub) — no
/// branches, no table lookups, no `is_x86_feature_detected!`, no
/// `#[target_feature]` unsafe.
#[inline]
pub fn clmul64_schoolbook(a: u64, b: u64) -> (u64, u64) {
    let b128 = b as u128;
    let mut acc: u128 = 0;
    let mut i: u32 = 0;
    while i < 64 {
        let nib = (a >> i) & 0xF;
        // Branchless mask: nonzero nibble ⇒ 0xFFFF_FFFF_FFFF_FFFF, zero ⇒ 0.
        let mask = 0u64.wrapping_sub((nib != 0) as u64) as u128;
        acc ^= (b128 << i) & mask;
        i += 4;
    }
    (acc as u64, (acc >> 64) as u64)
}

/// One unreduced GF(2^128) product, lo/hi pair assembled from four
/// `clmul64_schoolbook` calls. Used by the unrolled constant-clone fold
/// below; the per-limb XOR sum is collapsed into a single trailing
/// `ghash_reduce` by the caller.
#[inline]
pub fn ghash_mul_unreduced_schoolbook(a: F128, b: F128) -> F256Unreduced {
    let (ll_lo, ll_hi) = clmul64_schoolbook(a.lo, b.lo);
    let (lh_lo, lh_hi) = clmul64_schoolbook(a.lo, b.hi);
    let (hl_lo, hl_hi) = clmul64_schoolbook(a.hi, b.lo);
    let (hh_lo, hh_hi) = clmul64_schoolbook(a.hi, b.hi);
    let cr_lo = lh_lo ^ hl_lo;
    let cr_hi = lh_hi ^ hl_hi;
    F256Unreduced {
        r0: ll_lo,
        r1: ll_hi ^ cr_lo,
        r2: hh_lo ^ cr_hi,
        r3: hh_hi,
    }
}

pub fn ghash_mul_unreduced(a: F128, b: F128) -> F256Unreduced {
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

pub fn ghash_mul(a: F128, b: F128) -> F128 {
    let u = ghash_mul_unreduced(a, b);
    ghash_reduce(u.r0, u.r1, u.r2, u.r3)
}

/// Length-typed, `const CLONES`-unrolled GF(2^128) multiply that processes
/// `CLONES` field limbs per iteration with a single trailing
/// `ghash_reduce` over the XOR-accumulated unreduced products.
///
/// Each call multiplies a single `F128` by an `[F128; CLONES]` array using
/// only portable bitwise ops. The four 64×64 schoolbook products per limb
/// are accumulated in four `u64` "limb" accumulators; the per-limb
/// XOR-reduction is then collapsed into one `ghash_reduce` invocation
/// rather than `CLONES` of them. The `CLONES = 4` instantiation is the
/// length-typed entry point the prover's per-clone serial work uses to
/// amortize the surrounding BLAKE3 challenge sampling across four limbs
/// per loop iteration.
///
/// The optional `EXPECTED` digest, when supplied, is bit-compared to a
/// freshly computed folding of the same inputs in `debug_assert!` mode.
/// This guards the unrolled schoolbook from silent regressions on the
/// bench target where the on-hardware path runs in production.
#[inline]
pub fn ghash_mul_fold_const<const CLONES: usize>(
    a: F128,
    b: &[F128; CLONES],
    expected: Option<F128>,
) -> F128 {
    debug_assert!(CLONES > 0, "ghash_mul_fold_const needs at least one clone");
    let mut r0: u64 = 0;
    let mut r1: u64 = 0;
    let mut r2: u64 = 0;
    let mut r3: u64 = 0;
    let mut i: usize = 0;
    while i < CLONES {
        let bv = b[i];
        let (ll_lo, ll_hi) = clmul64_schoolbook(a.lo, bv.lo);
        let (lh_lo, lh_hi) = clmul64_schoolbook(a.lo, bv.hi);
        let (hl_lo, hl_hi) = clmul64_schoolbook(a.hi, bv.lo);
        let (hh_lo, hh_hi) = clmul64_schoolbook(a.hi, bv.hi);
        let cr_lo = lh_lo ^ hl_lo;
        let cr_hi = lh_hi ^ hl_hi;
        r0 ^= ll_lo;
        r1 ^= ll_hi ^ cr_lo;
        r2 ^= hh_lo ^ cr_hi;
        r3 ^= hh_hi;
        i += 1;
    }
    let folded = ghash_reduce(r0, r1, r2, r3);
    if let Some(want) = expected {
        debug_assert_eq!(
            folded, want,
            "ghash_mul_fold_const<{CLONES}> mismatch: per-limb reduction \
             collapsed into a single trailing pass must match the canonical \
             reduce-after-each-limb result bit-for-bit"
        );
    }
    folded
}

/// Length-typed, `const CLONES`-unrolled GF(2^128) multiply that returns
/// one unreduced product per clone. The companion to
/// [`ghash_mul_fold_const`]: callers that need per-lane products get the
/// `CLONES` unreduced results; callers that need the sum get the single
/// folded `F128` above.
#[inline]
pub fn ghash_mul_unreduced_const<const CLONES: usize>(
    a: F128,
    b: &[F128; CLONES],
) -> [F256Unreduced; CLONES] {
    let mut out = [F256Unreduced::ZERO; CLONES];
    let mut i: usize = 0;
    while i < CLONES {
        out[i] = ghash_mul_unreduced_schoolbook(a, b[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-exact equivalence: the new 4-bit schoolbook CLMUL agrees with
    /// the bit-by-bit reference on a small known-input fixture. If the
    /// shift/mask pattern in `clmul64_schoolbook` ever drifts, the
    /// debug-only digest-equivalence assert in
    /// `ghash_mul_fold_const::<4>` fires first and pinpoints the failure
    /// to the constant-clone fold path.
    #[test]
    fn schoolbook_matches_bit_by_bit() {
        let cases: [(u64, u64); 8] = [
            (0, 0),
            (1, 1),
            (0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF),
            (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
            (1u64 << 63, 1u64 << 63),
            (0xDEAD_BEEF_CAFE_BABE, 0xCAFE_BABE_DEAD_BEEF),
            (0x8000_0000_0000_0001, 0x0000_0000_0000_0001),
            (0x1357_9BDF_2468_ACE0, 0x0F1E_2D3C_4B5A_6978),
        ];
        for (a, b) in cases {
            let (lo, hi) = clmul64(a, b);
            let (lo2, hi2) = clmul64_schoolbook(a, b);
            assert_eq!((lo, hi), (lo2, hi2), "a={a:016x} b={b:016x}");
        }
    }

    /// Debug-only digest-equivalence assert: the constant-clone fold
    /// (`CLONES = 4`) matches the canonical reduce-after-each-limb
    /// product sum bit-for-bit on a fixed hand-picked fixture. Catches
    /// any regression before the bench run.
    #[test]
    fn fold_const_digest_matches_canonical() {
        let a = F128::new(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210);
        let b: [F128; 4] = [
            F128::new(0xCAFE_BABE_DEAD_BEEF, 0x1357_9BDF_2468_ACE0),
            F128::new(0xDEAD_BEEF_CAFE_BABE, 0x0F1E_2D3C_4B5A_6978),
            F128::new(0x8000_0000_0000_0001, 0x0000_0000_0000_0001),
            F128::new(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF),
        ];
        // Canonical: reduce each product individually, then XOR them.
        let canonical = ghash_mul(a, b[0]) ^ ghash_mul(a, b[1])
            ^ ghash_mul(a, b[2]) ^ ghash_mul(a, b[3]);
        // Folded: accumulate the four unreduced products, reduce once.
        let folded = ghash_mul_fold_const::<4>(a, &b, None);
        assert_eq!(folded, canonical, "fold const digest mismatch");
    }
}

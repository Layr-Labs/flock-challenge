use crate::field::F128;

#[inline]
pub(super) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    // Char-2: even*(1+r) + odd*r = even + r*(even+odd). One mul per pair.
    for (t, value) in dst.iter_mut().enumerate() {
        let s = 2 * (base + t);
        let even = src[s];
        *value = even + r * (even + src[s + 1]);
    }
}

#[inline]
pub(super) fn fold4_nested(src: &[F128], dst: &mut [F128], r0: F128, r1: F128) {
    // Nested pair-fold: r0 on (a0,a1) and (a2,a3), then r1 on (low, high).
    for (t, value) in dst.iter_mut().enumerate() {
        let a0 = src[4 * t];
        let a1 = src[4 * t + 1];
        let a2 = src[4 * t + 2];
        let a3 = src[4 * t + 3];
        let low = a0 + r0 * (a0 + a1);
        let high = a2 + r0 * (a2 + a3);
        *value = low + r1 * (low + high);
    }
}

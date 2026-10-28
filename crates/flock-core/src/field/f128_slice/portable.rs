use crate::field::{F128, F256Unreduced};

#[inline]
#[allow(dead_code)]
pub(super) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    for (t, value) in dst.iter_mut().enumerate() {
        let s = 2 * (base + t);
        let v0 = src[s];
        let v1 = src[s + 1];
        *value = v0 + (v0 + v1) * r;
    }
}

#[inline]
#[allow(dead_code)]
pub(super) fn fold_pairs_and_msg(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
) -> (F128, F128) {
    fold_pairs(f, base, fc, r);
    fold_pairs(b, base, bc, r);
    let mut u0 = F256Unreduced::ZERO;
    let mut u2 = F256Unreduced::ZERO;
    let mut k = 0;
    while k + 1 < fc.len() {
        let f0 = fc[k];
        let f1 = fc[k + 1];
        let b0 = bc[k];
        let b1 = bc[k + 1];
        u0 ^= f0.mul_unreduced(b0);
        u2 ^= (f0 + f1).mul_unreduced(b0 + b1);
        k += 2;
    }
    (u0.reduce(), u2.reduce())
}

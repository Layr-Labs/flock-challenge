use crate::field::F128;

#[inline]
pub(super) fn expand_eq_table_level(lo: &mut [F128], hi: &mut [F128], r: F128) {
    debug_assert_eq!(lo.len(), hi.len());
    for (lo_i, hi_i) in lo.iter_mut().zip(hi) {
        let value = *lo_i;
        let product = value * r;
        *hi_i = product;
        *lo_i = value + product;
    }
}

#[inline]
pub(super) fn round0_factorized_eq(f: &[F128], eq_tail: &[F128]) -> (F128, F128) {
    debug_assert_eq!(f.len(), 2 * eq_tail.len());
    let mut a = F128::ZERO;
    let mut s = F128::ZERO;
    for (pair, &weight) in f.chunks_exact(2).zip(eq_tail) {
        let f_0 = pair[0];
        a += f_0 * weight;
        s += (f_0 + pair[1]) * weight;
    }
    (a, s)
}

#[inline]
pub(super) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    let one_plus_r = F128::ONE + r;
    for (t, value) in dst.iter_mut().enumerate() {
        let s = 2 * (base + t);
        *value = src[s] * one_plus_r + src[s + 1] * r;
    }
}

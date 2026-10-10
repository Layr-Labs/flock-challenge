use crate::field::{F128, F256Unreduced};

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
    // Reduction is F2-linear: defer it until both product sums are complete.
    let mut a = F256Unreduced::ZERO;
    let mut s = F256Unreduced::ZERO;
    for (pair, &weight) in f.chunks_exact(2).zip(eq_tail) {
        let f_0 = pair[0];
        a ^= f_0.mul_unreduced(weight);
        s ^= (f_0 + pair[1]).mul_unreduced(weight);
    }
    (a.reduce(), s.reduce())
}

#[inline]
pub(super) fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    for (t, value) in dst.iter_mut().enumerate() {
        let s = 2 * (base + t);
        // Characteristic two: x0(1+r) + x1*r = x0 + (x0+x1)*r.
        // Keep only one field multiplication per destination.
        *value = src[s] + (src[s] + src[s + 1]) * r;
    }
}

/// Portable in-place counterpart of the fused AArch64 fold/message kernel.
/// Each four-element source group is copied before its two output slots are
/// overwritten, so later source groups remain intact.
#[inline]
#[allow(dead_code)]
pub(super) fn fold_two_and_msg_in_place(f: &mut [F128], b: &mut [F128], r: F128) -> (F128, F128) {
    debug_assert_eq!(f.len(), b.len());
    debug_assert!(f.len().is_multiple_of(4));
    let half = f.len() / 2;
    // XOR wide products; reduction commutes with the message sum.
    let mut u_0 = F256Unreduced::ZERO;
    let mut u_2 = F256Unreduced::ZERO;
    let mut t = 0;
    while t < half {
        let source = 2 * t;
        let f_even_0 = f[source];
        let f_odd_0 = f[source + 1];
        let f_even_1 = f[source + 2];
        let f_odd_1 = f[source + 3];
        let b_even_0 = b[source];
        let b_odd_0 = b[source + 1];
        let b_even_1 = b[source + 2];
        let b_odd_1 = b[source + 3];

        let f_0 = f_even_0 + (f_even_0 + f_odd_0) * r;
        let f_1 = f_even_1 + (f_even_1 + f_odd_1) * r;
        let b_0 = b_even_0 + (b_even_0 + b_odd_0) * r;
        let b_1 = b_even_1 + (b_even_1 + b_odd_1) * r;
        f[t] = f_0;
        f[t + 1] = f_1;
        b[t] = b_0;
        b[t + 1] = b_1;
        u_0 ^= f_0.mul_unreduced(b_0);
        u_2 ^= (f_0 + f_1).mul_unreduced(b_0 + b_1);
        t += 2;
    }
    (u_0.reduce(), u_2.reduce())
}

/// Portable reference for the fused deferred-basis/OOD-correction fold.
/// Shape and alignment checks live in the architecture-selecting wrapper.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn fold_two_and_msg_with_deferred_basis_and_scaled_local_addend(
    f: &[F128],
    b: &[F128],
    deferred_basis: &[F128],
    local_addend: &[F128],
    base: usize,
    nf: &mut [F128],
    nb: &mut [F128],
    r: F128,
    alpha: F128,
    alpha_r: F128,
    gamma: F128,
) -> (F128, F128) {
    // XOR wide products; reduction commutes with the message sum.
    let mut u_0 = F256Unreduced::ZERO;
    let mut u_2 = F256Unreduced::ZERO;
    let mut t = 0usize;
    while t < nf.len() {
        let source = 2 * (base + t);
        let f_0 = f[source] + r * (f[source] + f[source + 1]);
        let f_1 = f[source + 2] + r * (f[source + 2] + f[source + 3]);
        let b_0 = b[source]
            + r * (b[source] + b[source + 1])
            + alpha * deferred_basis[source]
            + alpha_r * (deferred_basis[source] + deferred_basis[source + 1])
            + gamma * local_addend[t];
        let b_1 = b[source + 2]
            + r * (b[source + 2] + b[source + 3])
            + alpha * deferred_basis[source + 2]
            + alpha_r * (deferred_basis[source + 2] + deferred_basis[source + 3])
            + gamma * local_addend[t + 1];
        nf[t] = f_0;
        nf[t + 1] = f_1;
        nb[t] = b_0;
        nb[t + 1] = b_1;
        u_0 ^= f_0.mul_unreduced(b_0);
        u_2 ^= (f_0 + f_1).mul_unreduced(b_0 + b_1);
        t += 2;
    }
    (u_0.reduce(), u_2.reduce())
}

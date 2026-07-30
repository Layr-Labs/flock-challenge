use crate::field::F128;

/// Two-lane pair fold using NEON and PMULL.
///
/// # Safety
/// Requires the `aes` target feature.
pub(super) unsafe fn fold_pairs(src: &[F128], base: usize, dst: &mut [F128], r: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    let lanes = dst.len() & !1;
    let mut t = 0;
    while t < lanes {
        let s = 2 * (base + t);
        let e0 = src[s];
        let o0 = src[s + 1];
        let e1 = src[s + 2];
        let o1 = src[s + 3];
        let x0 = F128 {
            lo: e0.lo ^ o0.lo,
            hi: e0.hi ^ o0.hi,
        };
        let x1 = F128 {
            lo: e1.lo ^ o1.lo,
            hi: e1.hi ^ o1.hi,
        };
        // SAFETY: caller guarantees the aes target feature.
        let prod = unsafe { ghash_mul_vec2_neon([r, r], [x0, x1]) };
        dst[t] = F128 {
            lo: e0.lo ^ prod[0].lo,
            hi: e0.hi ^ prod[0].hi,
        };
        dst[t + 1] = F128 {
            lo: e1.lo ^ prod[1].lo,
            hi: e1.hi ^ prod[1].hi,
        };
        t += 2;
    }

    let one_plus_r = F128::ONE + r;
    while t < dst.len() {
        let s = 2 * (base + t);
        dst[t] = src[s] * one_plus_r + src[s + 1] * r;
        t += 1;
    }
}

/// Fold two polynomials and build the next sumcheck message while each pair of
/// folded values is still in registers.
///
/// # Safety
/// Requires the `aes` target feature. The caller guarantees equal, even output
/// lengths and that both source elements exist for every destination element.
pub(super) unsafe fn fold_pairs_and_message(
    f: &[F128],
    b: &[F128],
    base: usize,
    folded_f: &mut [F128],
    folded_b: &mut [F128],
    r: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    let mut u_0 = F128::ZERO;
    let mut u_2 = F128::ZERO;
    let mut t = 0;
    while t < folded_f.len() {
        let s = 2 * (base + t);
        let fe0 = f[s];
        let fo0 = f[s + 1];
        let fe1 = f[s + 2];
        let fo1 = f[s + 3];
        let be0 = b[s];
        let bo0 = b[s + 1];
        let be1 = b[s + 2];
        let bo1 = b[s + 3];

        // SAFETY: this module is selected only when the aes target feature is
        // enabled.
        let fp = unsafe {
            ghash_mul_vec2_neon(
                [r, r],
                [
                    F128 {
                        lo: fe0.lo ^ fo0.lo,
                        hi: fe0.hi ^ fo0.hi,
                    },
                    F128 {
                        lo: fe1.lo ^ fo1.lo,
                        hi: fe1.hi ^ fo1.hi,
                    },
                ],
            )
        };
        // SAFETY: this module is selected only when the aes target feature is
        // enabled.
        let bp = unsafe {
            ghash_mul_vec2_neon(
                [r, r],
                [
                    F128 {
                        lo: be0.lo ^ bo0.lo,
                        hi: be0.hi ^ bo0.hi,
                    },
                    F128 {
                        lo: be1.lo ^ bo1.lo,
                        hi: be1.hi ^ bo1.hi,
                    },
                ],
            )
        };
        let f0 = F128 {
            lo: fe0.lo ^ fp[0].lo,
            hi: fe0.hi ^ fp[0].hi,
        };
        let f1 = F128 {
            lo: fe1.lo ^ fp[1].lo,
            hi: fe1.hi ^ fp[1].hi,
        };
        let b0 = F128 {
            lo: be0.lo ^ bp[0].lo,
            hi: be0.hi ^ bp[0].hi,
        };
        let b1 = F128 {
            lo: be1.lo ^ bp[1].lo,
            hi: be1.hi ^ bp[1].hi,
        };

        // SAFETY: this module is selected only when the aes target feature is
        // enabled. Batching the independent u_0 and u_2 products saves four
        // PMULL instructions compared with two scalar Binius multiplications.
        let message = unsafe {
            ghash_mul_vec2_neon(
                [
                    f0,
                    F128 {
                        lo: f0.lo ^ f1.lo,
                        hi: f0.hi ^ f1.hi,
                    },
                ],
                [
                    b0,
                    F128 {
                        lo: b0.lo ^ b1.lo,
                        hi: b0.hi ^ b1.hi,
                    },
                ],
            )
        };
        u_0 += message[0];
        u_2 += message[1];
        folded_f[t] = f0;
        folded_f[t + 1] = f1;
        folded_b[t] = b0;
        folded_b[t + 1] = b1;
        t += 2;
    }
    (u_0, u_2)
}

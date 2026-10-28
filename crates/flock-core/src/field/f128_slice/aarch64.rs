use crate::field::F128;

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

    while t < dst.len() {
        let s = 2 * (base + t);
        let v0 = src[s];
        dst[t] = v0 + (v0 + src[s + 1]) * r;
        t += 1;
    }
}

pub(super) unsafe fn fold_pairs_and_msg(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    let len = fc.len();
    let lanes = len & !1;
    let mut u0 = F128::ZERO;
    let mut u2 = F128::ZERO;
    let mut t = 0;
    while t < lanes {
        let s = 2 * (base + t);
        let fe0 = f[s];
        let fo0 = f[s + 1];
        let fe1 = f[s + 2];
        let fo1 = f[s + 3];
        let fx0 = F128 {
            lo: fe0.lo ^ fo0.lo,
            hi: fe0.hi ^ fo0.hi,
        };
        let fx1 = F128 {
            lo: fe1.lo ^ fo1.lo,
            hi: fe1.hi ^ fo1.hi,
        };
        let f_prod = unsafe { ghash_mul_vec2_neon([r, r], [fx0, fx1]) };
        let nf0 = F128 {
            lo: fe0.lo ^ f_prod[0].lo,
            hi: fe0.hi ^ f_prod[0].hi,
        };
        let nf1 = F128 {
            lo: fe1.lo ^ f_prod[1].lo,
            hi: fe1.hi ^ f_prod[1].hi,
        };
        fc[t] = nf0;
        fc[t + 1] = nf1;

        let be0 = b[s];
        let bo0 = b[s + 1];
        let be1 = b[s + 2];
        let bo1 = b[s + 3];
        let bx0 = F128 {
            lo: be0.lo ^ bo0.lo,
            hi: be0.hi ^ bo0.hi,
        };
        let bx1 = F128 {
            lo: be1.lo ^ bo1.lo,
            hi: be1.hi ^ bo1.hi,
        };
        let b_prod = unsafe { ghash_mul_vec2_neon([r, r], [bx0, bx1]) };
        let nb0 = F128 {
            lo: be0.lo ^ b_prod[0].lo,
            hi: be0.hi ^ b_prod[0].hi,
        };
        let nb1 = F128 {
            lo: be1.lo ^ b_prod[1].lo,
            hi: be1.hi ^ b_prod[1].hi,
        };
        bc[t] = nb0;
        bc[t + 1] = nb1;

        let msg_prod = unsafe { ghash_mul_vec2_neon([nf0, nf0 + nf1], [nb0, nb0 + nb1]) };
        u0 += msg_prod[0];
        u2 += msg_prod[1];
        t += 2;
    }
    while t < len {
        let s = 2 * (base + t);
        let fv0 = f[s];
        fc[t] = fv0 + (fv0 + f[s + 1]) * r;
        let bv0 = b[s];
        bc[t] = bv0 + (bv0 + b[s + 1]) * r;
        t += 1;
    }
    (u0, u2)
}

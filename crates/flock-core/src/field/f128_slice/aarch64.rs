use crate::field::F128;

/// Fused pair-fold of `(f, b)` with non-temporal output stores, plus the next
/// sumcheck round's `(u_0, u_2)` message terms accumulated from the folded
/// values while they are still in registers — the written lines are never
/// reloaded, which is what makes the 32 B `stnp` stores legal.
///
/// Folds `dst[k] = src[2k] + r · (src[2k] + src[2k+1])` for both arrays
/// (value-identical to [`fold_pairs`]) and accumulates
/// `u_0 += nf[2t]·nb[2t]`, `u_2 += (nf[2t]+nf[2t+1])·(nb[2t]+nb[2t+1])`
/// over the output pairs, exactly as the read-back message loop it replaces.
///
/// # Safety
/// Requires the `aes` target feature. Output slices must be valid for
/// 32-byte writes at every even index (guaranteed by the length contract:
/// `f_out.len() = b_out.len()` even, inputs twice as long).
pub(super) unsafe fn fold_pairs_msg_nt(
    f_in: &[F128],
    b_in: &[F128],
    f_out: &mut [F128],
    b_out: &mut [F128],
    r: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    let len = f_out.len();
    assert!(len >= 2 && len % 2 == 0);
    assert_eq!(b_out.len(), len);
    assert_eq!(f_in.len(), 2 * len);
    assert_eq!(b_in.len(), 2 * len);

    let f_ptr = f_out.as_mut_ptr();
    let b_ptr = b_out.as_mut_ptr();
    let mut u0 = F128::ZERO;
    let mut u2 = F128::ZERO;
    let mut k = 0;
    while k < len {
        let i = 2 * k;
        let fe0 = f_in[i];
        let fo0 = f_in[i + 1];
        let fe1 = f_in[i + 2];
        let fo1 = f_in[i + 3];
        let be0 = b_in[i];
        let bo0 = b_in[i + 1];
        let be1 = b_in[i + 2];
        let bo1 = b_in[i + 3];
        // SAFETY: caller guarantees the aes target feature.
        let pf = unsafe { ghash_mul_vec2_neon([r, r], [fe0 + fo0, fe1 + fo1]) };
        let pb = unsafe { ghash_mul_vec2_neon([r, r], [be0 + bo0, be1 + bo1]) };
        let nf0 = fe0 + pf[0];
        let nf1 = fe1 + pf[1];
        let nb0 = be0 + pb[0];
        let nb1 = be1 + pb[1];
        // SAFETY: k is even and k + 1 < len, so the pair is in range.
        unsafe {
            super::nt_store_pair(f_ptr.add(k), nf0, nf1);
            super::nt_store_pair(b_ptr.add(k), nb0, nb1);
        }
        // SAFETY: aes per the caller contract.
        let g = unsafe { ghash_mul_vec2_neon([nf0, nf0 + nf1], [nb0, nb0 + nb1]) };
        u0 += g[0];
        u2 += g[1];
        k += 2;
    }
    (u0, u2)
}

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

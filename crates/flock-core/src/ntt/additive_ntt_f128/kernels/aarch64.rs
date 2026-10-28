use crate::field::F128;

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert!(half >= 2);
    debug_assert_eq!(chunk.len(), 2 * half);
    let mut idx0 = 0;
    while idx0 < half {
        let idx1 = idx0 + half;
        let u_a = chunk[idx0];
        let v_a = chunk[idx1];
        let u_b = chunk[idx0 + 1];
        let v_b = chunk[idx1 + 1];

        let product = unsafe { ghash_mul_vec2_neon([twiddle, twiddle], [v_a, v_b]) };
        let new_u_a = unsafe {
            F128 {
                lo: crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u_a.lo, product[0].lo),
                hi: crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u_a.hi, product[0].hi),
            }
        };
        let new_u_b = unsafe {
            F128 {
                lo: crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u_b.lo, product[1].lo),
                hi: crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u_b.hi, product[1].hi),
            }
        };
        let new_v_a = unsafe {
            F128 {
                lo: crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u_a.lo, v_a.lo, product[0].lo),
                hi: crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u_a.hi, v_a.hi, product[0].hi),
            }
        };
        let new_v_b = unsafe {
            F128 {
                lo: crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u_b.lo, v_b.lo, product[1].lo),
                hi: crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u_b.hi, v_b.hi, product[1].hi),
            }
        };

        chunk[idx0] = new_u_a;
        chunk[idx1] = new_v_a;
        chunk[idx0 + 1] = new_u_b;
        chunk[idx1 + 1] = new_v_b;
        idx0 += 2;
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block_pair(chunk: &mut [F128], t_a: F128, t_b: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert_eq!(chunk.len(), 4);
    let u_a = chunk[0];
    let v_a = chunk[1];
    let u_b = chunk[2];
    let v_b = chunk[3];

    let product = unsafe { ghash_mul_vec2_neon([t_a, t_b], [v_a, v_b]) };
    let new_u_a = unsafe {
        F128 {
            lo: crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u_a.lo, product[0].lo),
            hi: crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u_a.hi, product[0].hi),
        }
    };
    let new_u_b = unsafe {
        F128 {
            lo: crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u_b.lo, product[1].lo),
            hi: crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u_b.hi, product[1].hi),
        }
    };
    let new_v_a = unsafe {
        F128 {
            lo: crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u_a.lo, v_a.lo, product[0].lo),
            hi: crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u_a.hi, v_a.hi, product[0].hi),
        }
    };
    let new_v_b = unsafe {
        F128 {
            lo: crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u_b.lo, v_b.lo, product[1].lo),
            hi: crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u_b.hi, v_b.hi, product[1].hi),
        }
    };

    chunk[0] = new_u_a;
    chunk[1] = new_v_a;
    chunk[2] = new_u_b;
    chunk[3] = new_v_b;
}

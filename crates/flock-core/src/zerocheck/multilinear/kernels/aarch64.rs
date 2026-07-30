use crate::field::F128;
use crate::field::gf2_128::F256Unreduced;

/// NEON one-row fold: 8 aligned 16-byte loads + 8 XORs, hand-unrolled for
/// `n_chunks = 8` (the k_skip=6 protocol size). Returns the folded F128.
///
/// The table is `Vec<F128>` with each entry 16-byte aligned (F128 is
/// `repr(C, align(16))`), so every `vld1q_u8` lands on an aligned address.
///
/// # Safety
/// Caller must guarantee `table_data` points to ≥ 8 × 256 × 16 valid bytes
/// (an `n_chunks ≥ 8` table) and `bytes_ptr` to ≥ 8 valid bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn fold_one_row_neon_unchecked_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> F128 {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        let mut acc = vld1q_u8(table_data.add((*bytes_ptr) as usize * 16));
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(1 * STRIDE + (*bytes_ptr.add(1)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(2 * STRIDE + (*bytes_ptr.add(2)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(3 * STRIDE + (*bytes_ptr.add(3)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(4 * STRIDE + (*bytes_ptr.add(4)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(5 * STRIDE + (*bytes_ptr.add(5)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(6 * STRIDE + (*bytes_ptr.add(6)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(7 * STRIDE + (*bytes_ptr.add(7)) as usize * 16)),
        );
        let acc_u64 = vreinterpretq_u64_u8(acc);
        F128 {
            lo: vgetq_lane_u64::<0>(acc_u64),
            hi: vgetq_lane_u64::<1>(acc_u64),
        }
    }
}

/// Fused tail-round worker kernel with non-temporal output stores: fold one
/// chunk of `(a_in, b_in)` at `r_fold` into `(a_out, b_out)` AND accumulate
/// the next round's message terms from the register copies of the folded
/// values, so the freshly-written output lines are never reloaded. That
/// no-readback property is what makes the 32 B `stnp` stores legal: the next
/// read of the folded buffers happens only after the round message is
/// absorbed and the next ρ is sampled (a Fiat–Shamir barrier).
///
/// Value-identical to `f128_slice::fold_pairs` on each buffer followed by the
/// eq-weighted message loop: the fold uses the same `e ^ r·(e ^ o)` form, and
/// the unreduced accumulation is an XOR (order-independent, exact).
///
/// Geometry: `a_in.len() = b_in.len() = 4·eq_lo.len()`,
/// `a_out.len() = b_out.len() = 2·eq_lo.len()`.
///
/// # Safety
/// Requires the `aes` target feature (PMULL). `a_out`/`b_out` must be valid
/// for 32-byte writes at every even index (guaranteed by the length contract).
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(crate) unsafe fn fold_and_message_neon_nt(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F256Unreduced, F256Unreduced) {
    use crate::field::f128_slice::nt_store_pair;
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    let lo_size = eq_lo.len();
    assert_eq!(a_in.len(), 4 * lo_size);
    assert_eq!(b_in.len(), 4 * lo_size);
    assert_eq!(a_out.len(), 2 * lo_size);
    assert_eq!(b_out.len(), 2 * lo_size);

    let a_out_ptr = a_out.as_mut_ptr();
    let b_out_ptr = b_out.as_mut_ptr();
    let mut p1_acc = F256Unreduced::ZERO;
    let mut pinf_acc = F256Unreduced::ZERO;

    for t in 0..lo_size {
        let i = 4 * t;
        let ae0 = a_in[i];
        let ao0 = a_in[i + 1];
        let ae1 = a_in[i + 2];
        let ao1 = a_in[i + 3];
        let be0 = b_in[i];
        let bo0 = b_in[i + 1];
        let be1 = b_in[i + 2];
        let bo1 = b_in[i + 3];

        let ax0 = F128 {
            lo: ae0.lo ^ ao0.lo,
            hi: ae0.hi ^ ao0.hi,
        };
        let ax1 = F128 {
            lo: ae1.lo ^ ao1.lo,
            hi: ae1.hi ^ ao1.hi,
        };
        let bx0 = F128 {
            lo: be0.lo ^ bo0.lo,
            hi: be0.hi ^ bo0.hi,
        };
        let bx1 = F128 {
            lo: be1.lo ^ bo1.lo,
            hi: be1.hi ^ bo1.hi,
        };
        // SAFETY: the cfg gate guarantees the aes feature.
        let pa = unsafe { ghash_mul_vec2_neon([r_fold, r_fold], [ax0, ax1]) };
        let pb = unsafe { ghash_mul_vec2_neon([r_fold, r_fold], [bx0, bx1]) };
        let sa0 = F128 {
            lo: ae0.lo ^ pa[0].lo,
            hi: ae0.hi ^ pa[0].hi,
        };
        let sa1 = F128 {
            lo: ae1.lo ^ pa[1].lo,
            hi: ae1.hi ^ pa[1].hi,
        };
        let sb0 = F128 {
            lo: be0.lo ^ pb[0].lo,
            hi: be0.hi ^ pb[0].hi,
        };
        let sb1 = F128 {
            lo: be1.lo ^ pb[1].lo,
            hi: be1.hi ^ pb[1].hi,
        };

        // SAFETY: t < lo_size, so 2t is an even index < a_out.len() with a
        // full pair in range.
        unsafe {
            nt_store_pair(a_out_ptr.add(2 * t), sa0, sa1);
            nt_store_pair(b_out_ptr.add(2 * t), sb0, sb1);
        }

        // Message terms from the register copies: g1 = a1·b1 and
        // g∞ = (a0+a1)·(b0+b1), eq-weighted with deferred reduction.
        // SAFETY: aes feature per the cfg gate.
        let g = unsafe {
            ghash_mul_vec2_neon(
                [sa1, sa0 + sa1],
                [sb1, sb0 + sb1],
            )
        };
        let eq_l = eq_lo[t];
        p1_acc ^= eq_l.mul_unreduced(g[0]);
        pinf_acc ^= eq_l.mul_unreduced(g[1]);
    }

    (p1_acc, pinf_acc)
}

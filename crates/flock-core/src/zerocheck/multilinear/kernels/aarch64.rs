use crate::field::F128;

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

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn fold_round3_pair_scaled_neon_unchecked_8(
    table0_data: *const u8,
    table1_data: *const u8,
    a_bytes: *const u8,
    b_bytes: *const u8,
) -> (F128, F128) {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        let mut acc_a = veorq_u8(
            vld1q_u8(table0_data.add((*a_bytes) as usize * 16)),
            vld1q_u8(table1_data.add((*a_bytes.add(8)) as usize * 16)),
        );
        let mut acc_b = veorq_u8(
            vld1q_u8(table0_data.add((*b_bytes) as usize * 16)),
            vld1q_u8(table1_data.add((*b_bytes.add(8)) as usize * 16)),
        );
        macro_rules! step {
            ($c:literal) => {{
                let off = $c * STRIDE;
                let a0 = vld1q_u8(table0_data.add(off + (*a_bytes.add($c)) as usize * 16));
                let a1 = vld1q_u8(table1_data.add(off + (*a_bytes.add(8 + $c)) as usize * 16));
                let b0 = vld1q_u8(table0_data.add(off + (*b_bytes.add($c)) as usize * 16));
                let b1 = vld1q_u8(table1_data.add(off + (*b_bytes.add(8 + $c)) as usize * 16));
                acc_a = veorq_u8(acc_a, veorq_u8(a0, a1));
                acc_b = veorq_u8(acc_b, veorq_u8(b0, b1));
            }};
        }
        step!(1);
        step!(2);
        step!(3);
        step!(4);
        step!(5);
        step!(6);
        step!(7);
        let a_u64 = vreinterpretq_u64_u8(acc_a);
        let b_u64 = vreinterpretq_u64_u8(acc_b);
        (
            F128 {
                lo: vgetq_lane_u64::<0>(a_u64),
                hi: vgetq_lane_u64::<1>(a_u64),
            },
            F128 {
                lo: vgetq_lane_u64::<0>(b_u64),
                hi: vgetq_lane_u64::<1>(b_u64),
            },
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn round2_msg_only_chunk_neon_unchecked_8(
    table_data: *const u8,
    a_pkt_ptr: *const u8,
    b_pkt_ptr: *const u8,
    eq_lo: &[F128],
    base: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    use core::arch::aarch64::*;

    unsafe {
        const STRIDE: usize = 256 * 16;
        let zero = vdupq_n_u64(0);
        let mut p1_ll = zero;
        let mut p1_mid = zero;
        let mut p1_hh = zero;
        let mut pinf_ll = zero;
        let mut pinf_mid = zero;
        let mut pinf_hh = zero;
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr() as *const u64;

        macro_rules! fold_pair {
            ($x_lo:expr) => {{
                let x_lo = $x_lo;
                let x0g = base + 2 * x_lo;
                let ap = a_pkt_ptr.add(x0g * 8);
                let bp = b_pkt_ptr.add(x0g * 8);
                let a0_u64 = (ap as *const u64).read_unaligned();
                let a1_u64 = (ap.add(8) as *const u64).read_unaligned();
                let b0_u64 = (bp as *const u64).read_unaligned();
                let b1_u64 = (bp.add(8) as *const u64).read_unaligned();
                let ad_u64 = a0_u64 ^ a1_u64;
                let bd_u64 = b0_u64 ^ b1_u64;

                let mut va1 = vld1q_u8(table_data.add(((a1_u64 & 0xFF) as usize) * 16));
                let mut vad = vld1q_u8(table_data.add(((ad_u64 & 0xFF) as usize) * 16));
                let mut vb1 = vld1q_u8(table_data.add(((b1_u64 & 0xFF) as usize) * 16));
                let mut vbd = vld1q_u8(table_data.add(((bd_u64 & 0xFF) as usize) * 16));

                macro_rules! fold_step {
                    ($c:literal) => {{
                        let off = $c * STRIDE;
                        let sh = $c * 8;
                        va1 = veorq_u8(
                            va1,
                            vld1q_u8(table_data.add(off + (((a1_u64 >> sh) & 0xFF) as usize) * 16)),
                        );
                        vad = veorq_u8(
                            vad,
                            vld1q_u8(table_data.add(off + (((ad_u64 >> sh) & 0xFF) as usize) * 16)),
                        );
                        vb1 = veorq_u8(
                            vb1,
                            vld1q_u8(table_data.add(off + (((b1_u64 >> sh) & 0xFF) as usize) * 16)),
                        );
                        vbd = veorq_u8(
                            vbd,
                            vld1q_u8(table_data.add(off + (((bd_u64 >> sh) & 0xFF) as usize) * 16)),
                        );
                    }};
                }
                fold_step!(1);
                fold_step!(2);
                fold_step!(3);
                fold_step!(4);
                fold_step!(5);
                fold_step!(6);
                fold_step!(7);

                let ua1 = vreinterpretq_u64_u8(va1);
                let uad = vreinterpretq_u64_u8(vad);
                let ub1 = vreinterpretq_u64_u8(vb1);
                let ubd = vreinterpretq_u64_u8(vbd);
                let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_neon(ua1, uad, ub1, ubd);

                let veq = vld1q_u64(eq_ptr.add(2 * x_lo));
                crate::gf2_kernels::k0_f128_mul_acc_pair_neon(
                    veq,
                    g1,
                    ginf,
                    &mut p1_ll,
                    &mut p1_mid,
                    &mut p1_hh,
                    &mut pinf_ll,
                    &mut pinf_mid,
                    &mut pinf_hh,
                );
            }};
        }

        macro_rules! fold_pair_unit_b {
            ($x_lo:expr) => {{
                let x_lo = $x_lo;
                let x0g = base + 2 * x_lo;
                let ap = a_pkt_ptr.add(x0g * 8);
                let a1_u64 = (ap.add(8) as *const u64).read_unaligned();

                let mut va1 = vld1q_u8(table_data.add(((a1_u64 & 0xFF) as usize) * 16));

                macro_rules! fold_step_a {
                    ($c:literal) => {{
                        let off = $c * STRIDE;
                        let sh = $c * 8;
                        va1 = veorq_u8(
                            va1,
                            vld1q_u8(table_data.add(off + (((a1_u64 >> sh) & 0xFF) as usize) * 16)),
                        );
                    }};
                }
                fold_step_a!(1);
                fold_step_a!(2);
                fold_step_a!(3);
                fold_step_a!(4);
                fold_step_a!(5);
                fold_step_a!(6);
                fold_step_a!(7);

                let g1 = vreinterpretq_u64_u8(va1);
                let veq = vld1q_u64(eq_ptr.add(2 * x_lo));
                let (t_ll, t_mid, t_hh) = crate::gf2_kernels::k0_f128_mul_unreduced_neon(veq, g1);
                p1_ll = veorq_u64(p1_ll, t_ll);
                p1_mid = veorq_u64(p1_mid, t_mid);
                p1_hh = veorq_u64(p1_hh, t_hh);
            }};
        }

        if pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
            && (pair_idx_base & 127) == 0
            && (lo_size & 127) == 0
        {
            let mut blk = 0usize;
            while blk < lo_size {
                let bp_blk = b_pkt_ptr.add((base + 2 * blk) * 8) as *const u64;
                let head_ones = (bp_blk.read_unaligned()
                    & bp_blk.add(1).read_unaligned()
                    & bp_blk.add(2).read_unaligned()
                    & bp_blk.add(3).read_unaligned()
                    & bp_blk.add(4).read_unaligned()
                    & bp_blk.add(5).read_unaligned()
                    & bp_blk.add(6).read_unaligned()
                    & bp_blk.add(7).read_unaligned()
                    & bp_blk.add(8).read_unaligned()
                    & bp_blk.add(9).read_unaligned()
                    & bp_blk.add(10).read_unaligned()
                    & bp_blk.add(11).read_unaligned()
                    & bp_blk.add(12).read_unaligned()
                    & bp_blk.add(13).read_unaligned()
                    & bp_blk.add(14).read_unaligned()
                    & bp_blk.add(15).read_unaligned()
                    & bp_blk.add(16).read_unaligned()
                    & bp_blk.add(17).read_unaligned())
                    == !0u64;
                let tail_ones = (bp_blk.add(236).read_unaligned()
                    & bp_blk.add(237).read_unaligned()
                    & bp_blk.add(238).read_unaligned()
                    & bp_blk.add(239).read_unaligned())
                    == !0u64;
                if head_ones {
                    for x_lo in blk..(blk + 9) {
                        fold_pair_unit_b!(x_lo);
                    }
                } else {
                    for x_lo in blk..(blk + 9) {
                        fold_pair!(x_lo);
                    }
                }
                for x_lo in (blk + 9)..(blk + 118) {
                    fold_pair!(x_lo);
                }
                if tail_ones {
                    for x_lo in (blk + 118)..(blk + 120) {
                        fold_pair_unit_b!(x_lo);
                    }
                } else {
                    for x_lo in (blk + 118)..(blk + 120) {
                        fold_pair!(x_lo);
                    }
                }
                {
                    let x_lo = blk + 120;
                    let x0g = base + 2 * x_lo;
                    let ap = a_pkt_ptr.add(x0g * 8);
                    let bp = b_pkt_ptr.add(x0g * 8);
                    let a1_u64 = (ap.add(8) as *const u64).read_unaligned();
                    let b1_u64 = (bp.add(8) as *const u64).read_unaligned();
                    if (a1_u64 | b1_u64) == 0 {
                        let a0_u64 = (ap as *const u64).read_unaligned();
                        let b0_u64 = (bp as *const u64).read_unaligned();
                        let mut va0 = vld1q_u8(table_data.add(((a0_u64 & 0xFF) as usize) * 16));
                        let mut vb0 = vld1q_u8(table_data.add(((b0_u64 & 0xFF) as usize) * 16));
                        macro_rules! fold_step_ab2 {
                            ($c:literal) => {{
                                let off = $c * STRIDE;
                                let sh = $c * 8;
                                va0 = veorq_u8(
                                    va0,
                                    vld1q_u8(
                                        table_data
                                            .add(off + (((a0_u64 >> sh) & 0xFF) as usize) * 16),
                                    ),
                                );
                                vb0 = veorq_u8(
                                    vb0,
                                    vld1q_u8(
                                        table_data
                                            .add(off + (((b0_u64 >> sh) & 0xFF) as usize) * 16),
                                    ),
                                );
                            }};
                        }
                        fold_step_ab2!(1);
                        fold_step_ab2!(2);
                        fold_step_ab2!(3);
                        fold_step_ab2!(4);
                        fold_step_ab2!(5);
                        fold_step_ab2!(6);
                        fold_step_ab2!(7);
                        let ginf = crate::gf2_kernels::k0_f128_mul_neon(
                            vreinterpretq_u64_u8(va0),
                            vreinterpretq_u64_u8(vb0),
                        );
                        let veq = vld1q_u64(eq_ptr.add(2 * x_lo));
                        let (t_ll, t_mid, t_hh) =
                            crate::gf2_kernels::k0_f128_mul_unreduced_neon(veq, ginf);
                        pinf_ll = veorq_u64(pinf_ll, t_ll);
                        pinf_mid = veorq_u64(pinf_mid, t_mid);
                        pinf_hh = veorq_u64(pinf_hh, t_hh);
                    } else {
                        fold_pair!(x_lo);
                    }
                }
                blk += 128;
            }
        } else {
            for x_lo in 0..lo_size {
                if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                    continue;
                }
                fold_pair!(x_lo);
            }
        }

        let res1 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(p1_ll, p1_mid, p1_hh);
        let resi = crate::gf2_kernels::k0_f128_reduce_3lane_neon(pinf_ll, pinf_mid, pinf_hh);
        (
            F128 {
                lo: vgetq_lane_u64::<0>(res1),
                hi: vgetq_lane_u64::<1>(res1),
            },
            F128 {
                lo: vgetq_lane_u64::<0>(resi),
                hi: vgetq_lane_u64::<1>(resi),
            },
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn round3_fold_in_place_chunk_neon_unchecked_8(
    table_data: *const u8,
    rho_0: F128,
    a_chunk: &mut [F128],
    b_chunk: &mut [F128],
    eq_lo: &[F128],
    base_j: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    use core::arch::aarch64::*;

    unsafe {
        const STRIDE: usize = 256 * 16;
        let zero = vdupq_n_u64(0);
        let mut p1_ll = zero;
        let mut p1_mid = zero;
        let mut p1_hh = zero;
        let mut pinf_ll = zero;
        let mut pinf_mid = zero;
        let mut pinf_hh = zero;
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr() as *const u64;
        let a_ptr = a_chunk.as_mut_ptr() as *mut u64;
        let b_ptr = b_chunk.as_mut_ptr() as *mut u64;
        let v_rho0 = vcombine_u64(vcreate_u64(rho_0.lo), vcreate_u64(rho_0.hi));
        let (rho0_r0, rho0_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(v_rho0);
        let one_u64 = vcombine_u64(vcreate_u64(1), vcreate_u64(0));
        let v_opr0 = veorq_u64(one_u64, v_rho0);
        let (opr0_r0, opr0_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(v_opr0);

        macro_rules! fold_single_table4 {
            ($tbl:expr, $a0_u64:expr, $b0_u64:expr, $a1_u64:expr, $b1_u64:expr) => {{
                let a0w = $a0_u64;
                let b0w = $b0_u64;
                let a1w = $a1_u64;
                let b1w = $b1_u64;
                let mut acc_a0 = vld1q_u8($tbl.add(((a0w & 0xFF) as usize) * 16));
                let mut acc_b0 = vld1q_u8($tbl.add(((b0w & 0xFF) as usize) * 16));
                let mut acc_a1 = vld1q_u8($tbl.add(((a1w & 0xFF) as usize) * 16));
                let mut acc_b1 = vld1q_u8($tbl.add(((b1w & 0xFF) as usize) * 16));
                macro_rules! step4 {
                    ($c:literal) => {{
                        let off = $c * STRIDE;
                        let sh = $c * 8;
                        acc_a0 = veorq_u8(
                            acc_a0,
                            vld1q_u8($tbl.add(off + (((a0w >> sh) & 0xFF) as usize) * 16)),
                        );
                        acc_b0 = veorq_u8(
                            acc_b0,
                            vld1q_u8($tbl.add(off + (((b0w >> sh) & 0xFF) as usize) * 16)),
                        );
                        acc_a1 = veorq_u8(
                            acc_a1,
                            vld1q_u8($tbl.add(off + (((a1w >> sh) & 0xFF) as usize) * 16)),
                        );
                        acc_b1 = veorq_u8(
                            acc_b1,
                            vld1q_u8($tbl.add(off + (((b1w >> sh) & 0xFF) as usize) * 16)),
                        );
                    }};
                }
                step4!(1);
                step4!(2);
                step4!(3);
                step4!(4);
                step4!(5);
                step4!(6);
                step4!(7);
                (
                    vreinterpretq_u64_u8(acc_a0),
                    vreinterpretq_u64_u8(acc_b0),
                    vreinterpretq_u64_u8(acc_a1),
                    vreinterpretq_u64_u8(acc_b1),
                )
            }};
        }

        macro_rules! fold_single_table2 {
            ($tbl:expr, $a0_u64:expr, $a1_u64:expr) => {{
                let a0w = $a0_u64;
                let a1w = $a1_u64;
                let mut acc_a0 = vld1q_u8($tbl.add(((a0w & 0xFF) as usize) * 16));
                let mut acc_a1 = vld1q_u8($tbl.add(((a1w & 0xFF) as usize) * 16));
                macro_rules! step2 {
                    ($c:literal) => {{
                        let off = $c * STRIDE;
                        let sh = $c * 8;
                        acc_a0 = veorq_u8(
                            acc_a0,
                            vld1q_u8($tbl.add(off + (((a0w >> sh) & 0xFF) as usize) * 16)),
                        );
                        acc_a1 = veorq_u8(
                            acc_a1,
                            vld1q_u8($tbl.add(off + (((a1w >> sh) & 0xFF) as usize) * 16)),
                        );
                    }};
                }
                step2!(1);
                step2!(2);
                step2!(3);
                step2!(4);
                step2!(5);
                step2!(6);
                step2!(7);
                (vreinterpretq_u64_u8(acc_a0), vreinterpretq_u64_u8(acc_a1))
            }};
        }

        macro_rules! round3_pair_unit_b {
            ($x_lo:expr) => {{
                let x_lo = $x_lo;
                let j0l = 2 * x_lo;
                let j1l = j0l + 1;
                let ap0 = a_ptr.add(2 * j0l);
                let ap1 = a_ptr.add(2 * j1l);
                let a0_lo = ap0.read_unaligned();
                let a0_hi = ap0.add(1).read_unaligned();
                let a1_lo = ap1.read_unaligned();
                let a1_hi = ap1.add(1).read_unaligned();
                let (ua0_lo, ua0_xor, ua1_lo, ua1_xor) =
                    fold_single_table4!(table_data, a0_lo, a0_lo ^ a0_hi, a1_lo, a1_lo ^ a1_hi);
                let (ua0_p, ua1_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    ua0_xor, ua1_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                );
                let ua_j0 = veorq_u64(ua0_lo, ua0_p);
                let ua_j1 = veorq_u64(ua1_lo, ua1_p);
                vst1q_u64(a_ptr.add(2 * j0l), ua_j0);
                vst1q_u64(a_ptr.add(2 * j1l), ua_j1);
                vst1q_u64(b_ptr.add(2 * j0l), one_u64);
                vst1q_u64(b_ptr.add(2 * j1l), one_u64);

                let veq = vld1q_u64(eq_ptr.add(2 * x_lo));
                let (t_ll, t_mid, t_hh) =
                    crate::gf2_kernels::k0_f128_mul_unreduced_neon(veq, ua_j1);
                p1_ll = veorq_u64(p1_ll, t_ll);
                p1_mid = veorq_u64(p1_mid, t_mid);
                p1_hh = veorq_u64(p1_hh, t_hh);
            }};
        }

        macro_rules! round3_pair_both {
            ($x_lo:expr) => {{
                let x_lo = $x_lo;
                let j0l = 2 * x_lo;
                let j1l = j0l + 1;
                let ap0 = a_ptr.add(2 * j0l);
                let bp0 = b_ptr.add(2 * j0l);
                let a0_lo = ap0.read_unaligned();
                let a0_hi = ap0.add(1).read_unaligned();
                let b0_lo = bp0.read_unaligned();
                let b0_hi = bp0.add(1).read_unaligned();
                let (ua0_lo, ub0_lo, ua0_xor, ub0_xor) =
                    fold_single_table4!(table_data, a0_lo, b0_lo, a0_lo ^ a0_hi, b0_lo ^ b0_hi);
                let (ua0_p, ub0_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    ua0_xor, ub0_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                );
                let ua_j0 = veorq_u64(ua0_lo, ua0_p);
                let ub_j0 = veorq_u64(ub0_lo, ub0_p);

                let ap1 = a_ptr.add(2 * j1l);
                let bp1 = b_ptr.add(2 * j1l);
                let a1_lo = ap1.read_unaligned();
                let a1_hi = ap1.add(1).read_unaligned();
                let b1_lo = bp1.read_unaligned();
                let b1_hi = bp1.add(1).read_unaligned();
                let (ua1_lo, ub1_lo, ua1_xor, ub1_xor) =
                    fold_single_table4!(table_data, a1_lo, b1_lo, a1_lo ^ a1_hi, b1_lo ^ b1_hi);
                let (ua1_p, ub1_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    ua1_xor, ub1_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                );
                let ua_j1 = veorq_u64(ua1_lo, ua1_p);
                let ub_j1 = veorq_u64(ub1_lo, ub1_p);

                vst1q_u64(a_ptr.add(2 * j0l), ua_j0);
                vst1q_u64(a_ptr.add(2 * j1l), ua_j1);
                vst1q_u64(b_ptr.add(2 * j0l), ub_j0);
                vst1q_u64(b_ptr.add(2 * j1l), ub_j1);

                let uad = veorq_u64(ua_j0, ua_j1);
                let ubd = veorq_u64(ub_j0, ub_j1);
                let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_neon(ua_j1, uad, ub_j1, ubd);

                let veq = vld1q_u64(eq_ptr.add(2 * x_lo));
                crate::gf2_kernels::k0_f128_mul_acc_pair_neon(
                    veq,
                    g1,
                    ginf,
                    &mut p1_ll,
                    &mut p1_mid,
                    &mut p1_hh,
                    &mut pinf_ll,
                    &mut pinf_mid,
                    &mut pinf_hh,
                );
            }};
        }

        if pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
            && (base_j & 127) == 0
            && (lo_size & 63) == 0
        {
            let mut blk = 0usize;
            while blk < lo_size {
                let bp_blk = b_ptr.add(4 * blk);
                let head_ones = (bp_blk.read_unaligned()
                    & bp_blk.add(1).read_unaligned()
                    & bp_blk.add(2).read_unaligned()
                    & bp_blk.add(3).read_unaligned()
                    & bp_blk.add(4).read_unaligned()
                    & bp_blk.add(5).read_unaligned()
                    & bp_blk.add(6).read_unaligned()
                    & bp_blk.add(7).read_unaligned()
                    & bp_blk.add(8).read_unaligned()
                    & bp_blk.add(9).read_unaligned()
                    & bp_blk.add(10).read_unaligned()
                    & bp_blk.add(11).read_unaligned()
                    & bp_blk.add(12).read_unaligned()
                    & bp_blk.add(13).read_unaligned()
                    & bp_blk.add(14).read_unaligned()
                    & bp_blk.add(15).read_unaligned())
                    == !0u64;
                let tail_ones = (bp_blk.add(236).read_unaligned()
                    & bp_blk.add(237).read_unaligned()
                    & bp_blk.add(238).read_unaligned()
                    & bp_blk.add(239).read_unaligned())
                    == !0u64;
                if head_ones {
                    for x_lo in blk..(blk + 4) {
                        round3_pair_unit_b!(x_lo);
                    }
                } else {
                    for x_lo in blk..(blk + 4) {
                        round3_pair_both!(x_lo);
                    }
                }
                for x_lo in (blk + 4)..(blk + 59) {
                    round3_pair_both!(x_lo);
                }
                if tail_ones {
                    round3_pair_unit_b!(blk + 59);
                } else {
                    round3_pair_both!(blk + 59);
                }

                {
                    let x_lo = blk + 60;
                    let j0l = 2 * x_lo;
                    let j1l = j0l + 1;
                    let ap0 = a_ptr.add(2 * j0l);
                    let bp0 = b_ptr.add(2 * j0l);
                    let a0_lo = ap0.read_unaligned();
                    let a0_hi = ap0.add(1).read_unaligned();
                    let b0_lo = bp0.read_unaligned();
                    let b0_hi = bp0.add(1).read_unaligned();
                    let (ua_j0, ub_j0) = if (a0_hi | b0_hi) == 0 {
                        let (ua0_lo, ub0_lo) = fold_single_table2!(table_data, a0_lo, b0_lo);
                        crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                            ua0_lo, ub0_lo, opr0_r0, opr0_r1, opr0_r0, opr0_r1,
                        )
                    } else {
                        let (ua0_lo, ub0_lo, ua0_xor, ub0_xor) = fold_single_table4!(
                            table_data,
                            a0_lo,
                            b0_lo,
                            a0_lo ^ a0_hi,
                            b0_lo ^ b0_hi
                        );
                        let (ua0_p, ub0_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                            ua0_xor, ub0_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                        );
                        (veorq_u64(ua0_lo, ua0_p), veorq_u64(ub0_lo, ub0_p))
                    };

                    vst1q_u64(a_ptr.add(2 * j0l), ua_j0);
                    vst1q_u64(a_ptr.add(2 * j1l), zero);
                    vst1q_u64(b_ptr.add(2 * j0l), ub_j0);
                    vst1q_u64(b_ptr.add(2 * j1l), zero);

                    let ginf = crate::gf2_kernels::k0_f128_mul_neon(ua_j0, ub_j0);
                    let veq = vld1q_u64(eq_ptr.add(2 * x_lo));
                    let (t_ll, t_mid, t_hh) =
                        crate::gf2_kernels::k0_f128_mul_unreduced_neon(veq, ginf);
                    pinf_ll = veorq_u64(pinf_ll, t_ll);
                    pinf_mid = veorq_u64(pinf_mid, t_mid);
                    pinf_hh = veorq_u64(pinf_hh, t_hh);
                }

                for x_lo in (blk + 61)..(blk + 64) {
                    let j0l = 2 * x_lo;
                    let j1l = j0l + 1;
                    vst1q_u64(a_ptr.add(2 * j0l), zero);
                    vst1q_u64(a_ptr.add(2 * j1l), zero);
                    vst1q_u64(b_ptr.add(2 * j0l), zero);
                    vst1q_u64(b_ptr.add(2 * j1l), zero);
                }
                blk += 64;
            }
        } else {
            for x_lo in 0..lo_size {
                let j0l = 2 * x_lo;
                let j1l = j0l + 1;
                let j0g = base_j + j0l;
                let j1g = j0g + 1;

                if (j0g & pair_in_block_mask) >= useful_pairs_inclusive {
                    vst1q_u64(a_ptr.add(2 * j0l), zero);
                    vst1q_u64(a_ptr.add(2 * j1l), zero);
                    vst1q_u64(b_ptr.add(2 * j0l), zero);
                    vst1q_u64(b_ptr.add(2 * j1l), zero);
                    continue;
                }

                let ap0 = a_ptr.add(2 * j0l);
                let bp0 = b_ptr.add(2 * j0l);
                let a0_lo = ap0.read_unaligned();
                let a0_hi = ap0.add(1).read_unaligned();
                let b0_lo = bp0.read_unaligned();
                let b0_hi = bp0.add(1).read_unaligned();
                let (ua0_lo, ub0_lo, ua0_xor, ub0_xor) =
                    fold_single_table4!(table_data, a0_lo, b0_lo, a0_lo ^ a0_hi, b0_lo ^ b0_hi);
                let (ua0_p, ub0_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    ua0_xor, ub0_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                );
                let ua_j0 = veorq_u64(ua0_lo, ua0_p);
                let ub_j0 = veorq_u64(ub0_lo, ub0_p);

                let (ua_j1, ub_j1) = if (j1g & pair_in_block_mask) < useful_pairs_inclusive {
                    let ap1 = a_ptr.add(2 * j1l);
                    let bp1 = b_ptr.add(2 * j1l);
                    let a1_lo = ap1.read_unaligned();
                    let a1_hi = ap1.add(1).read_unaligned();
                    let b1_lo = bp1.read_unaligned();
                    let b1_hi = bp1.add(1).read_unaligned();
                    let (ua1_lo, ub1_lo, ua1_xor, ub1_xor) =
                        fold_single_table4!(table_data, a1_lo, b1_lo, a1_lo ^ a1_hi, b1_lo ^ b1_hi);
                    let (ua1_p, ub1_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        ua1_xor, ub1_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                    );
                    (veorq_u64(ua1_lo, ua1_p), veorq_u64(ub1_lo, ub1_p))
                } else {
                    (zero, zero)
                };

                vst1q_u64(a_ptr.add(2 * j0l), ua_j0);
                vst1q_u64(a_ptr.add(2 * j1l), ua_j1);
                vst1q_u64(b_ptr.add(2 * j0l), ub_j0);
                vst1q_u64(b_ptr.add(2 * j1l), ub_j1);

                let uad = veorq_u64(ua_j0, ua_j1);
                let ubd = veorq_u64(ub_j0, ub_j1);

                let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_neon(ua_j1, uad, ub_j1, ubd);

                let veq = vld1q_u64(eq_ptr.add(2 * x_lo));
                crate::gf2_kernels::k0_f128_mul_acc_pair_neon(
                    veq,
                    g1,
                    ginf,
                    &mut p1_ll,
                    &mut p1_mid,
                    &mut p1_hh,
                    &mut pinf_ll,
                    &mut pinf_mid,
                    &mut pinf_hh,
                );
            }
        }

        let res1 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(p1_ll, p1_mid, p1_hh);
        let resi = crate::gf2_kernels::k0_f128_reduce_3lane_neon(pinf_ll, pinf_mid, pinf_hh);
        (
            F128 {
                lo: vgetq_lane_u64::<0>(res1),
                hi: vgetq_lane_u64::<1>(res1),
            },
            F128 {
                lo: vgetq_lane_u64::<0>(resi),
                hi: vgetq_lane_u64::<1>(resi),
            },
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn round23_lookahead_chunk_neon_unchecked_8(
    table_data: *const u8,
    a_pkt_ptr: *const u8,
    b_pkt_ptr: *const u8,
    eq_lo: &[F128],
    q_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [F128; 8] {
    use core::arch::aarch64::*;

    unsafe {
        const STRIDE: usize = 256 * 16;
        let zero = vdupq_n_u64(0);
        let mut acc = [[zero; 3]; 8];
        let mut p240_ll = zero;
        let mut p240_mid = zero;
        let mut p240_hh = zero;
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr() as *const u64;

        macro_rules! fold_single_table4 {
            ($w0:expr, $w1:expr, $w2:expr, $w3:expr) => {{
                let w0 = $w0;
                let w1 = $w1;
                let w2 = $w2;
                let w3 = $w3;
                let mut v0 = vld1q_u8(table_data.add(((w0 & 0xFF) as usize) * 16));
                let mut v1 = vld1q_u8(table_data.add(((w1 & 0xFF) as usize) * 16));
                let mut v2 = vld1q_u8(table_data.add(((w2 & 0xFF) as usize) * 16));
                let mut v3 = vld1q_u8(table_data.add(((w3 & 0xFF) as usize) * 16));
                macro_rules! step4 {
                    ($c:literal) => {{
                        let off = $c * STRIDE;
                        let sh = $c * 8;
                        v0 = veorq_u8(
                            v0,
                            vld1q_u8(table_data.add(off + (((w0 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v1 = veorq_u8(
                            v1,
                            vld1q_u8(table_data.add(off + (((w1 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v2 = veorq_u8(
                            v2,
                            vld1q_u8(table_data.add(off + (((w2 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v3 = veorq_u8(
                            v3,
                            vld1q_u8(table_data.add(off + (((w3 >> sh) & 0xFF) as usize) * 16)),
                        );
                    }};
                }
                step4!(1);
                step4!(2);
                step4!(3);
                step4!(4);
                step4!(5);
                step4!(6);
                step4!(7);
                (
                    vreinterpretq_u64_u8(v0),
                    vreinterpretq_u64_u8(v1),
                    vreinterpretq_u64_u8(v2),
                    vreinterpretq_u64_u8(v3),
                )
            }};
        }

        macro_rules! fold_single_table3 {
            ($w0:expr, $w1:expr, $w2:expr) => {{
                let w0 = $w0;
                let w1 = $w1;
                let w2 = $w2;
                let mut v0 = vld1q_u8(table_data.add(((w0 & 0xFF) as usize) * 16));
                let mut v1 = vld1q_u8(table_data.add(((w1 & 0xFF) as usize) * 16));
                let mut v2 = vld1q_u8(table_data.add(((w2 & 0xFF) as usize) * 16));
                macro_rules! step3 {
                    ($c:literal) => {{
                        let off = $c * STRIDE;
                        let sh = $c * 8;
                        v0 = veorq_u8(
                            v0,
                            vld1q_u8(table_data.add(off + (((w0 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v1 = veorq_u8(
                            v1,
                            vld1q_u8(table_data.add(off + (((w1 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v2 = veorq_u8(
                            v2,
                            vld1q_u8(table_data.add(off + (((w2 >> sh) & 0xFF) as usize) * 16)),
                        );
                    }};
                }
                step3!(1);
                step3!(2);
                step3!(3);
                step3!(4);
                step3!(5);
                step3!(6);
                step3!(7);
                (
                    vreinterpretq_u64_u8(v0),
                    vreinterpretq_u64_u8(v1),
                    vreinterpretq_u64_u8(v2),
                )
            }};
        }

        macro_rules! acc_unred_slot_with_sum {
            ($idx:literal, $u:expr, $v:expr, $u_sum:expr, $v_sum:expr) => {{
                let u = $u;
                let v = $v;
                let u_lo = vgetq_lane_u64::<0>(u);
                let v_lo = vgetq_lane_u64::<0>(v);
                let t_ll = vreinterpretq_u64_p128(vmull_p64(u_lo, v_lo));
                let t_mid = vreinterpretq_u64_p128(vmull_p64($u_sum, $v_sum));
                let t_hh = vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(u),
                    vreinterpretq_p64_u64(v),
                ));
                acc[$idx][0] = veorq_u64(acc[$idx][0], t_ll);
                acc[$idx][1] = veorq_u64(acc[$idx][1], t_mid);
                acc[$idx][2] = veorq_u64(acc[$idx][2], t_hh);
            }};
        }

        macro_rules! acc_unred_slot {
            ($idx:literal, $u:expr, $v:expr) => {{
                let (t_ll, t_mid, t_hh) = crate::gf2_kernels::k0_f128_mul_unreduced_neon($u, $v);
                acc[$idx][0] = veorq_u64(acc[$idx][0], t_ll);
                acc[$idx][1] = veorq_u64(acc[$idx][1], t_mid);
                acc[$idx][2] = veorq_u64(acc[$idx][2], t_hh);
            }};
        }

        macro_rules! round23_general_from_folded {
            ($q_lo:expr, $va00:expr, $vb00:expr, $va10:expr, $vb10:expr, $va01:expr, $vb01:expr, $va11:expr, $vb11:expr) => {{
                let veq = vld1q_u64(eq_ptr.add(2 * $q_lo));
                let (eq_r0, eq_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(veq);
                let (tb00, tb10) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    $vb00, $vb10, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                let (tb01, tb11) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    $vb01, $vb11, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                let da0 = veorq_u64($va00, $va10);
                let da1 = veorq_u64($va01, $va11);
                let da_0 = veorq_u64($va00, $va01);
                let da_1 = veorq_u64($va10, $va11);
                let da_i = veorq_u64(da0, da1);

                let dtb0 = veorq_u64(tb00, tb10);
                let dtb1 = veorq_u64(tb01, tb11);
                let dtb_0 = veorq_u64(tb00, tb01);
                let dtb_1 = veorq_u64(tb10, tb11);
                let dtb_i = veorq_u64(dtb0, dtb1);

                let sva10 = vgetq_lane_u64::<0>($va10) ^ vgetq_lane_u64::<1>($va10);
                let sva11 = vgetq_lane_u64::<0>($va11) ^ vgetq_lane_u64::<1>($va11);
                let sva01 = vgetq_lane_u64::<0>($va01) ^ vgetq_lane_u64::<1>($va01);
                let sda0 = vgetq_lane_u64::<0>(da0) ^ vgetq_lane_u64::<1>(da0);
                let sda1 = sva01 ^ sva11;
                let sda_1 = sva10 ^ sva11;
                let sda_0 = sda0 ^ (sva10 ^ sva01);
                let sda_i = sda0 ^ sda1;

                let stb10 = vgetq_lane_u64::<0>(tb10) ^ vgetq_lane_u64::<1>(tb10);
                let stb11 = vgetq_lane_u64::<0>(tb11) ^ vgetq_lane_u64::<1>(tb11);
                let stb01 = vgetq_lane_u64::<0>(tb01) ^ vgetq_lane_u64::<1>(tb01);
                let sdtb0 = vgetq_lane_u64::<0>(dtb0) ^ vgetq_lane_u64::<1>(dtb0);
                let sdtb1 = stb01 ^ stb11;
                let sdtb_1 = stb10 ^ stb11;
                let sdtb_0 = sdtb0 ^ (stb10 ^ stb01);
                let sdtb_i = sdtb0 ^ sdtb1;

                acc_unred_slot_with_sum!(0, $va10, tb10, sva10, stb10);
                acc_unred_slot_with_sum!(1, da0, dtb0, sda0, sdtb0);
                acc_unred_slot_with_sum!(2, $va11, tb11, sva11, stb11);
                acc_unred_slot_with_sum!(3, da1, dtb1, sda1, sdtb1);
                acc_unred_slot_with_sum!(4, $va01, tb01, sva01, stb01);
                acc_unred_slot_with_sum!(5, da_0, dtb_0, sda_0, sdtb_0);
                acc_unred_slot_with_sum!(6, da_1, dtb_1, sda_1, sdtb_1);
                acc_unred_slot_with_sum!(7, da_i, dtb_i, sda_i, sdtb_i);
            }};
        }

        macro_rules! round23_general_step {
            ($q_lo:expr) => {{
                let ql = $q_lo;
                let qg = q_base + ql;
                let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                let bp = (b_pkt_ptr as *const u64).add(4 * qg);
                let (va00, vb00, va10, vb10) = fold_single_table4!(
                    ap.read_unaligned(),
                    bp.read_unaligned(),
                    ap.add(1).read_unaligned(),
                    bp.add(1).read_unaligned()
                );
                let (va01, vb01, va11, vb11) = fold_single_table4!(
                    ap.add(2).read_unaligned(),
                    bp.add(2).read_unaligned(),
                    ap.add(3).read_unaligned(),
                    bp.add(3).read_unaligned()
                );
                round23_general_from_folded!(ql, va00, vb00, va10, vb10, va01, vb01, va11, vb11);
            }};
        }

        macro_rules! round23_unit_b_step {
            ($q_lo:expr) => {{
                let ql = $q_lo;
                let qg = q_base + ql;
                let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                let (va10, va01, va11) = fold_single_table3!(
                    ap.add(1).read_unaligned(),
                    ap.add(2).read_unaligned(),
                    ap.add(3).read_unaligned()
                );
                let veq = vld1q_u64(eq_ptr.add(2 * ql));
                acc_unred_slot!(0, va10, veq);
                acc_unred_slot!(2, va11, veq);
                acc_unred_slot!(4, va01, veq);
            }};
        }

        if pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
            && (q_base & 63) == 0
            && lo_size >= 64
            && (lo_size & 63) == 0
        {
            let n_blocks = lo_size >> 6;
            for blk in 0..n_blocks {
                let blk_q = blk << 6;
                let bp_blk = (b_pkt_ptr as *const u64).add(4 * (q_base + blk_q));
                let head_ones = (bp_blk.read_unaligned()
                    & bp_blk.add(1).read_unaligned()
                    & bp_blk.add(2).read_unaligned()
                    & bp_blk.add(3).read_unaligned()
                    & bp_blk.add(4).read_unaligned()
                    & bp_blk.add(5).read_unaligned()
                    & bp_blk.add(6).read_unaligned()
                    & bp_blk.add(7).read_unaligned()
                    & bp_blk.add(8).read_unaligned()
                    & bp_blk.add(9).read_unaligned()
                    & bp_blk.add(10).read_unaligned()
                    & bp_blk.add(11).read_unaligned()
                    & bp_blk.add(12).read_unaligned()
                    & bp_blk.add(13).read_unaligned()
                    & bp_blk.add(14).read_unaligned()
                    & bp_blk.add(15).read_unaligned())
                    == !0u64;
                let tail_ones = (bp_blk.add(236).read_unaligned()
                    & bp_blk.add(237).read_unaligned()
                    & bp_blk.add(238).read_unaligned()
                    & bp_blk.add(239).read_unaligned())
                    == !0u64;
                if head_ones {
                    for q_lo in blk_q..(blk_q + 4) {
                        round23_unit_b_step!(q_lo);
                    }
                } else {
                    for q_lo in blk_q..(blk_q + 4) {
                        round23_general_step!(q_lo);
                    }
                }
                for q_lo in (blk_q + 4)..(blk_q + 59) {
                    round23_general_step!(q_lo);
                }
                if tail_ones {
                    round23_unit_b_step!(blk_q + 59);
                } else {
                    round23_general_step!(blk_q + 59);
                }
                {
                    let ql = blk_q + 60;
                    let qg = q_base + ql;
                    let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                    let bp = (b_pkt_ptr as *const u64).add(4 * qg);
                    let a10 = ap.add(1).read_unaligned();
                    let b10 = bp.add(1).read_unaligned();
                    if (a10 | b10) == 0 {
                        let a00 = ap.read_unaligned();
                        let b00 = bp.read_unaligned();
                        let mut va00 = vld1q_u8(table_data.add(((a00 & 0xFF) as usize) * 16));
                        let mut vb00 = vld1q_u8(table_data.add(((b00 & 0xFF) as usize) * 16));
                        macro_rules! step2 {
                            ($c:literal) => {{
                                let off = $c * STRIDE;
                                let sh = $c * 8;
                                va00 = veorq_u8(
                                    va00,
                                    vld1q_u8(
                                        table_data.add(off + (((a00 >> sh) & 0xFF) as usize) * 16),
                                    ),
                                );
                                vb00 = veorq_u8(
                                    vb00,
                                    vld1q_u8(
                                        table_data.add(off + (((b00 >> sh) & 0xFF) as usize) * 16),
                                    ),
                                );
                            }};
                        }
                        step2!(1);
                        step2!(2);
                        step2!(3);
                        step2!(4);
                        step2!(5);
                        step2!(6);
                        step2!(7);
                        let g00 = crate::gf2_kernels::k0_f128_mul_neon(
                            vreinterpretq_u64_u8(va00),
                            vreinterpretq_u64_u8(vb00),
                        );
                        let veq = vld1q_u64(eq_ptr.add(2 * ql));
                        let (t_ll, t_mid, t_hh) =
                            crate::gf2_kernels::k0_f128_mul_unreduced_neon(veq, g00);
                        p240_ll = veorq_u64(p240_ll, t_ll);
                        p240_mid = veorq_u64(p240_mid, t_mid);
                        p240_hh = veorq_u64(p240_hh, t_hh);
                    } else {
                        let (va00, vb00, va10, vb10) =
                            fold_single_table4!(ap.read_unaligned(), bp.read_unaligned(), a10, b10);
                        round23_general_from_folded!(
                            ql, va00, vb00, va10, vb10, zero, zero, zero, zero
                        );
                    }
                }
            }
            acc[1][0] = veorq_u64(acc[1][0], p240_ll);
            acc[1][1] = veorq_u64(acc[1][1], p240_mid);
            acc[1][2] = veorq_u64(acc[1][2], p240_hh);
            acc[5][0] = veorq_u64(acc[5][0], p240_ll);
            acc[5][1] = veorq_u64(acc[5][1], p240_mid);
            acc[5][2] = veorq_u64(acc[5][2], p240_hh);
            acc[7][0] = veorq_u64(acc[7][0], p240_ll);
            acc[7][1] = veorq_u64(acc[7][1], p240_mid);
            acc[7][2] = veorq_u64(acc[7][2], p240_hh);
        } else {
            for q_lo in 0..lo_size {
                let qg = q_base + q_lo;
                let j0 = 2 * qg;
                let j1 = j0 + 1;
                if (j0 & pair_in_block_mask) >= useful_pairs_inclusive {
                    continue;
                }
                let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                let bp = (b_pkt_ptr as *const u64).add(4 * qg);
                let (va00, vb00, va10, vb10) = fold_single_table4!(
                    ap.read_unaligned(),
                    bp.read_unaligned(),
                    ap.add(1).read_unaligned(),
                    bp.add(1).read_unaligned()
                );
                let (va01, vb01, va11, vb11) = if (j1 & pair_in_block_mask) < useful_pairs_inclusive
                {
                    fold_single_table4!(
                        ap.add(2).read_unaligned(),
                        bp.add(2).read_unaligned(),
                        ap.add(3).read_unaligned(),
                        bp.add(3).read_unaligned()
                    )
                } else {
                    (zero, zero, zero, zero)
                };
                round23_general_from_folded!(q_lo, va00, vb00, va10, vb10, va01, vb01, va11, vb11);
            }
        }

        let mut out = [F128::ZERO; 8];
        for k in 0..8 {
            let r = crate::gf2_kernels::k0_f128_reduce_3lane_neon(acc[k][0], acc[k][1], acc[k][2]);
            out[k] = F128 {
                lo: vgetq_lane_u64::<0>(r),
                hi: vgetq_lane_u64::<1>(r),
            };
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn round4_fold_2vars_into_chunk_neon_unchecked_8(
    table_data: *const u8,
    rho_0: F128,
    rho_1: F128,
    a_pkt_ptr: *const u8,
    b_pkt_ptr: *const u8,
    a_out_chunk: &mut [F128],
    b_out_chunk: &mut [F128],
    eq_lo: &[F128],
    q_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    use core::arch::aarch64::*;

    unsafe {
        const STRIDE: usize = 256 * 16;
        let zero = vdupq_n_u64(0);
        let one_u64 = vcombine_u64(vcreate_u64(1), vcreate_u64(0));
        let mut p1_ll = zero;
        let mut p1_mid = zero;
        let mut p1_hh = zero;
        let mut pinf_ll = zero;
        let mut pinf_mid = zero;
        let mut pinf_hh = zero;
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr() as *const u64;
        let a_out_ptr = a_out_chunk.as_mut_ptr() as *mut u64;
        let b_out_ptr = b_out_chunk.as_mut_ptr() as *mut u64;

        let v_rho0 = vcombine_u64(vcreate_u64(rho_0.lo), vcreate_u64(rho_0.hi));
        let v_rho1 = vcombine_u64(vcreate_u64(rho_1.lo), vcreate_u64(rho_1.hi));
        let rho_01 = rho_0 * rho_1;
        let v_rho01 = vcombine_u64(vcreate_u64(rho_01.lo), vcreate_u64(rho_01.hi));
        let opr01 = (F128::ONE + rho_0) * (F128::ONE + rho_1);
        let v_opr01 = vcombine_u64(vcreate_u64(opr01.lo), vcreate_u64(opr01.hi));
        let (opr01_r0, opr01_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(v_opr01);

        macro_rules! fold_single_table4 {
            ($w0:expr, $w1:expr, $w2:expr, $w3:expr) => {{
                let w0 = $w0;
                let w1 = $w1;
                let w2 = $w2;
                let w3 = $w3;
                let mut v0 = vld1q_u8(table_data.add(((w0 & 0xFF) as usize) * 16));
                let mut v1 = vld1q_u8(table_data.add(((w1 & 0xFF) as usize) * 16));
                let mut v2 = vld1q_u8(table_data.add(((w2 & 0xFF) as usize) * 16));
                let mut v3 = vld1q_u8(table_data.add(((w3 & 0xFF) as usize) * 16));
                macro_rules! step4 {
                    ($c:literal) => {{
                        let off = $c * STRIDE;
                        let sh = $c * 8;
                        v0 = veorq_u8(
                            v0,
                            vld1q_u8(table_data.add(off + (((w0 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v1 = veorq_u8(
                            v1,
                            vld1q_u8(table_data.add(off + (((w1 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v2 = veorq_u8(
                            v2,
                            vld1q_u8(table_data.add(off + (((w2 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v3 = veorq_u8(
                            v3,
                            vld1q_u8(table_data.add(off + (((w3 >> sh) & 0xFF) as usize) * 16)),
                        );
                    }};
                }
                step4!(1);
                step4!(2);
                step4!(3);
                step4!(4);
                step4!(5);
                step4!(6);
                step4!(7);
                (
                    vreinterpretq_u64_u8(v0),
                    vreinterpretq_u64_u8(v1),
                    vreinterpretq_u64_u8(v2),
                    vreinterpretq_u64_u8(v3),
                )
            }};
        }

        macro_rules! combine_2vars_from_folded_neon {
            ($f00:expr, $f10:expr, $f01:expr, $f11:expr) => {{
                let (l10, m10, h10) = crate::gf2_kernels::k0_f128_mul_unreduced_neon($f10, v_rho0);
                let (l01, m01, h01) = crate::gf2_kernels::k0_f128_mul_unreduced_neon($f01, v_rho1);
                let (l11, m11, h11) = crate::gf2_kernels::k0_f128_mul_unreduced_neon($f11, v_rho01);
                let ll = veorq_u64(veorq_u64(l10, l01), l11);
                let mid = veorq_u64(veorq_u64(m10, m01), m11);
                let hh = veorq_u64(veorq_u64(h10, h01), h11);
                veorq_u64(
                    $f00,
                    crate::gf2_kernels::k0_f128_reduce_3lane_neon(ll, mid, hh),
                )
            }};
        }

        macro_rules! fold_2vars_one {
            ($wp:expr) => {{
                let wp = $wp;
                let w00 = wp.read_unaligned();
                let w10 = wp.add(1).read_unaligned();
                let w01 = wp.add(2).read_unaligned();
                let w11 = wp.add(3).read_unaligned();
                let u10 = w00 ^ w10;
                let u01 = w00 ^ w01;
                let u11 = u10 ^ (w01 ^ w11);
                let (f00, f10, f01, f11) = fold_single_table4!(w00, u10, u01, u11);
                combine_2vars_from_folded_neon!(f00, f10, f01, f11)
            }};
        }

        macro_rules! fold_2vars_ab_one {
            ($ap:expr, $bp:expr) => {{ (fold_2vars_one!($ap), fold_2vars_one!($bp)) }};
        }

        macro_rules! fold_2vars_a_pair_unit_b {
            ($x_lo:expr) => {{
                let xl = $x_lo;
                let q0l = 2 * xl;
                let q1l = q0l + 1;
                let q0g = q_base + q0l;
                let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                let va_q0 = fold_2vars_one!(ap0);
                let va_q1 = fold_2vars_one!(ap0.add(4));

                vst1q_u64(a_out_ptr.add(2 * q0l), va_q0);
                vst1q_u64(a_out_ptr.add(2 * q1l), va_q1);
                vst1q_u64(b_out_ptr.add(2 * q0l), one_u64);
                vst1q_u64(b_out_ptr.add(2 * q1l), one_u64);

                let veq = vld1q_u64(eq_ptr.add(2 * xl));
                let (t_ll, t_mid, t_hh) =
                    crate::gf2_kernels::k0_f128_mul_unreduced_neon(veq, va_q1);
                p1_ll = veorq_u64(p1_ll, t_ll);
                p1_mid = veorq_u64(p1_mid, t_mid);
                p1_hh = veorq_u64(p1_hh, t_hh);
            }};
        }

        macro_rules! round4_general_step {
            ($x_lo:expr) => {{
                let xl = $x_lo;
                let q0l = 2 * xl;
                let q1l = q0l + 1;
                let q0g = q_base + q0l;
                let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                let bp0 = (b_pkt_ptr as *const u64).add(4 * q0g);
                let (va_q0, vb_q0) = fold_2vars_ab_one!(ap0, bp0);
                let (va_q1, vb_q1) = fold_2vars_ab_one!(ap0.add(4), bp0.add(4));

                vst1q_u64(a_out_ptr.add(2 * q0l), va_q0);
                vst1q_u64(a_out_ptr.add(2 * q1l), va_q1);
                vst1q_u64(b_out_ptr.add(2 * q0l), vb_q0);
                vst1q_u64(b_out_ptr.add(2 * q1l), vb_q1);

                let vad = veorq_u64(va_q0, va_q1);
                let vbd = veorq_u64(vb_q0, vb_q1);
                let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_neon(va_q1, vad, vb_q1, vbd);
                let veq = vld1q_u64(eq_ptr.add(2 * xl));
                crate::gf2_kernels::k0_f128_mul_acc_pair_neon(
                    veq,
                    g1,
                    ginf,
                    &mut p1_ll,
                    &mut p1_mid,
                    &mut p1_hh,
                    &mut pinf_ll,
                    &mut pinf_mid,
                    &mut pinf_hh,
                );
            }};
        }

        if pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
            && (q_base & 63) == 0
            && lo_size >= 32
            && (lo_size & 31) == 0
        {
            let n_blocks = lo_size >> 5;
            for blk in 0..n_blocks {
                let blk_x = blk << 5;
                let bp_blk = (b_pkt_ptr as *const u64).add(4 * (q_base + 2 * blk_x));
                let head_ones = (bp_blk.read_unaligned()
                    & bp_blk.add(1).read_unaligned()
                    & bp_blk.add(2).read_unaligned()
                    & bp_blk.add(3).read_unaligned()
                    & bp_blk.add(4).read_unaligned()
                    & bp_blk.add(5).read_unaligned()
                    & bp_blk.add(6).read_unaligned()
                    & bp_blk.add(7).read_unaligned()
                    & bp_blk.add(8).read_unaligned()
                    & bp_blk.add(9).read_unaligned()
                    & bp_blk.add(10).read_unaligned()
                    & bp_blk.add(11).read_unaligned()
                    & bp_blk.add(12).read_unaligned()
                    & bp_blk.add(13).read_unaligned()
                    & bp_blk.add(14).read_unaligned()
                    & bp_blk.add(15).read_unaligned())
                    == !0u64;
                let tail_ones = (bp_blk.add(236).read_unaligned()
                    & bp_blk.add(237).read_unaligned()
                    & bp_blk.add(238).read_unaligned()
                    & bp_blk.add(239).read_unaligned())
                    == !0u64;
                if head_ones {
                    fold_2vars_a_pair_unit_b!(blk_x);
                    fold_2vars_a_pair_unit_b!(blk_x + 1);
                } else {
                    round4_general_step!(blk_x);
                    round4_general_step!(blk_x + 1);
                }
                for x_lo in (blk_x + 2)..(blk_x + 29) {
                    round4_general_step!(x_lo);
                }
                if tail_ones {
                    let xl = blk_x + 29;
                    let q0l = 2 * xl;
                    let q1l = q0l + 1;
                    let q0g = q_base + q0l;
                    let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                    let bp0 = (b_pkt_ptr as *const u64).add(4 * q0g);
                    let (va_q0, vb_q0) = fold_2vars_ab_one!(ap0, bp0);
                    let va_q1 = fold_2vars_one!(ap0.add(4));
                    let vb_q1 = one_u64;

                    vst1q_u64(a_out_ptr.add(2 * q0l), va_q0);
                    vst1q_u64(a_out_ptr.add(2 * q1l), va_q1);
                    vst1q_u64(b_out_ptr.add(2 * q0l), vb_q0);
                    vst1q_u64(b_out_ptr.add(2 * q1l), vb_q1);

                    let vad = veorq_u64(va_q0, va_q1);
                    let vbd = veorq_u64(vb_q0, vb_q1);
                    let ginf = crate::gf2_kernels::k0_f128_mul_neon(vad, vbd);
                    let veq = vld1q_u64(eq_ptr.add(2 * xl));
                    crate::gf2_kernels::k0_f128_mul_acc_pair_neon(
                        veq,
                        va_q1,
                        ginf,
                        &mut p1_ll,
                        &mut p1_mid,
                        &mut p1_hh,
                        &mut pinf_ll,
                        &mut pinf_mid,
                        &mut pinf_hh,
                    );
                } else {
                    round4_general_step!(blk_x + 29);
                }
                {
                    let xl = blk_x + 30;
                    let q0l = 2 * xl;
                    let q1l = q0l + 1;
                    let q0g = q_base + q0l;
                    let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                    let bp0 = (b_pkt_ptr as *const u64).add(4 * q0g);
                    let a10 = ap0.add(1).read_unaligned();
                    let b10 = bp0.add(1).read_unaligned();
                    let (va_q0, vb_q0) = if (a10 | b10) == 0 {
                        let a00 = ap0.read_unaligned();
                        let b00 = bp0.read_unaligned();
                        let mut va0 = vld1q_u8(table_data.add(((a00 & 0xFF) as usize) * 16));
                        let mut vb0 = vld1q_u8(table_data.add(((b00 & 0xFF) as usize) * 16));
                        macro_rules! step2 {
                            ($c:literal) => {{
                                let off = $c * STRIDE;
                                let sh = $c * 8;
                                va0 = veorq_u8(
                                    va0,
                                    vld1q_u8(
                                        table_data.add(off + (((a00 >> sh) & 0xFF) as usize) * 16),
                                    ),
                                );
                                vb0 = veorq_u8(
                                    vb0,
                                    vld1q_u8(
                                        table_data.add(off + (((b00 >> sh) & 0xFF) as usize) * 16),
                                    ),
                                );
                            }};
                        }
                        step2!(1);
                        step2!(2);
                        step2!(3);
                        step2!(4);
                        step2!(5);
                        step2!(6);
                        step2!(7);
                        crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                            vreinterpretq_u64_u8(va0),
                            vreinterpretq_u64_u8(vb0),
                            opr01_r0,
                            opr01_r1,
                            opr01_r0,
                            opr01_r1,
                        )
                    } else {
                        fold_2vars_ab_one!(ap0, bp0)
                    };
                    vst1q_u64(a_out_ptr.add(2 * q0l), va_q0);
                    vst1q_u64(a_out_ptr.add(2 * q1l), zero);
                    vst1q_u64(b_out_ptr.add(2 * q0l), vb_q0);
                    vst1q_u64(b_out_ptr.add(2 * q1l), zero);
                    let ginf = crate::gf2_kernels::k0_f128_mul_neon(va_q0, vb_q0);
                    let veq = vld1q_u64(eq_ptr.add(2 * xl));
                    let (t_ll, t_mid, t_hh) =
                        crate::gf2_kernels::k0_f128_mul_unreduced_neon(veq, ginf);
                    pinf_ll = veorq_u64(pinf_ll, t_ll);
                    pinf_mid = veorq_u64(pinf_mid, t_mid);
                    pinf_hh = veorq_u64(pinf_hh, t_hh);
                }
                {
                    let xl = blk_x + 31;
                    let q0l = 2 * xl;
                    let q1l = q0l + 1;
                    vst1q_u64(a_out_ptr.add(2 * q0l), zero);
                    vst1q_u64(a_out_ptr.add(2 * q1l), zero);
                    vst1q_u64(b_out_ptr.add(2 * q0l), zero);
                    vst1q_u64(b_out_ptr.add(2 * q1l), zero);
                }
            }
        } else {
            for x_lo in 0..lo_size {
                let q0l = 2 * x_lo;
                let q1l = q0l + 1;
                let q0g = q_base + q0l;
                let q1g = q0g + 1;

                let fold_one_padded = |qg: usize| -> (uint64x2_t, uint64x2_t) {
                    let j0 = 2 * qg;
                    let j1 = j0 + 1;
                    if (j0 & pair_in_block_mask) >= useful_pairs_inclusive {
                        return (zero, zero);
                    }
                    let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                    let bp = (b_pkt_ptr as *const u64).add(4 * qg);
                    let a00 = ap.read_unaligned();
                    let a10 = ap.add(1).read_unaligned();
                    let b00 = bp.read_unaligned();
                    let b10 = bp.add(1).read_unaligned();
                    let (a01, a11, b01, b11) = if (j1 & pair_in_block_mask) < useful_pairs_inclusive
                    {
                        (
                            ap.add(2).read_unaligned(),
                            ap.add(3).read_unaligned(),
                            bp.add(2).read_unaligned(),
                            bp.add(3).read_unaligned(),
                        )
                    } else {
                        (0, 0, 0, 0)
                    };
                    let ua10 = a00 ^ a10;
                    let ua01 = a00 ^ a01;
                    let ua11 = ua10 ^ (a01 ^ a11);
                    let ub10 = b00 ^ b10;
                    let ub01 = b00 ^ b01;
                    let ub11 = ub10 ^ (b01 ^ b11);
                    let (fa00, fa10, fa01, fa11) = fold_single_table4!(a00, ua10, ua01, ua11);
                    let (fb00, fb10, fb01, fb11) = fold_single_table4!(b00, ub10, ub01, ub11);
                    (
                        combine_2vars_from_folded_neon!(fa00, fa10, fa01, fa11),
                        combine_2vars_from_folded_neon!(fb00, fb10, fb01, fb11),
                    )
                };

                let (va_q0, vb_q0) = fold_one_padded(q0g);
                let (va_q1, vb_q1) = fold_one_padded(q1g);
                vst1q_u64(a_out_ptr.add(2 * q0l), va_q0);
                vst1q_u64(a_out_ptr.add(2 * q1l), va_q1);
                vst1q_u64(b_out_ptr.add(2 * q0l), vb_q0);
                vst1q_u64(b_out_ptr.add(2 * q1l), vb_q1);

                let vad = veorq_u64(va_q0, va_q1);
                let vbd = veorq_u64(vb_q0, vb_q1);
                let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_neon(va_q1, vad, vb_q1, vbd);
                let veq = vld1q_u64(eq_ptr.add(2 * x_lo));
                crate::gf2_kernels::k0_f128_mul_acc_pair_neon(
                    veq,
                    g1,
                    ginf,
                    &mut p1_ll,
                    &mut p1_mid,
                    &mut p1_hh,
                    &mut pinf_ll,
                    &mut pinf_mid,
                    &mut pinf_hh,
                );
            }
        }

        let res1 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(p1_ll, p1_mid, p1_hh);
        let resi = crate::gf2_kernels::k0_f128_reduce_3lane_neon(pinf_ll, pinf_mid, pinf_hh);
        (
            F128 {
                lo: vgetq_lane_u64::<0>(res1),
                hi: vgetq_lane_u64::<1>(res1),
            },
            F128 {
                lo: vgetq_lane_u64::<0>(resi),
                hi: vgetq_lane_u64::<1>(resi),
            },
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn round45_fold_2vars_into_and_lookahead_chunk_neon_unchecked_8(
    table_data: *const u8,
    rho_0: F128,
    rho_1: F128,
    a_pkt_ptr: *const u8,
    b_pkt_ptr: *const u8,
    a_out_chunk: &mut [F128],
    b_out_chunk: &mut [F128],
    eq_lo: &[F128],
    q_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [F128; 8] {
    use core::arch::aarch64::*;

    unsafe {
        const STRIDE: usize = 256 * 16;
        let zero = vdupq_n_u64(0);
        let one_u64 = vcombine_u64(vcreate_u64(1), vcreate_u64(0));
        let mut acc = [[zero; 3]; 8];
        let mut p60_ll = zero;
        let mut p60_mid = zero;
        let mut p60_hh = zero;
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr() as *const u64;
        let a_out_ptr = a_out_chunk.as_mut_ptr() as *mut u64;
        let b_out_ptr = b_out_chunk.as_mut_ptr() as *mut u64;

        let v_rho0 = vcombine_u64(vcreate_u64(rho_0.lo), vcreate_u64(rho_0.hi));
        let v_rho1 = vcombine_u64(vcreate_u64(rho_1.lo), vcreate_u64(rho_1.hi));
        let rho_01 = rho_0 * rho_1;
        let v_rho01 = vcombine_u64(vcreate_u64(rho_01.lo), vcreate_u64(rho_01.hi));
        let opr01 = (F128::ONE + rho_0) * (F128::ONE + rho_1);
        let v_opr01 = vcombine_u64(vcreate_u64(opr01.lo), vcreate_u64(opr01.hi));
        let (opr01_r0, opr01_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(v_opr01);

        let mut v_all_ones_u8 = vld1q_u8(table_data.add(255 * 16));
        for c in 1..8 {
            v_all_ones_u8 = veorq_u8(
                v_all_ones_u8,
                vld1q_u8(table_data.add(c * STRIDE + 255 * 16)),
            );
        }
        let v_all_ones = vreinterpretq_u64_u8(v_all_ones_u8);
        let is_unit_ones =
            (vgetq_lane_u64::<0>(v_all_ones) == 1) && (vgetq_lane_u64::<1>(v_all_ones) == 0);

        macro_rules! fold_single_table4 {
            ($w0:expr, $w1:expr, $w2:expr, $w3:expr) => {{
                let w0 = $w0;
                let w1 = $w1;
                let w2 = $w2;
                let w3 = $w3;
                let mut v0 = vld1q_u8(table_data.add(((w0 & 0xFF) as usize) * 16));
                let mut v1 = vld1q_u8(table_data.add(((w1 & 0xFF) as usize) * 16));
                let mut v2 = vld1q_u8(table_data.add(((w2 & 0xFF) as usize) * 16));
                let mut v3 = vld1q_u8(table_data.add(((w3 & 0xFF) as usize) * 16));
                macro_rules! step4 {
                    ($c:literal) => {{
                        let off = $c * STRIDE;
                        let sh = $c * 8;
                        v0 = veorq_u8(
                            v0,
                            vld1q_u8(table_data.add(off + (((w0 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v1 = veorq_u8(
                            v1,
                            vld1q_u8(table_data.add(off + (((w1 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v2 = veorq_u8(
                            v2,
                            vld1q_u8(table_data.add(off + (((w2 >> sh) & 0xFF) as usize) * 16)),
                        );
                        v3 = veorq_u8(
                            v3,
                            vld1q_u8(table_data.add(off + (((w3 >> sh) & 0xFF) as usize) * 16)),
                        );
                    }};
                }
                step4!(1);
                step4!(2);
                step4!(3);
                step4!(4);
                step4!(5);
                step4!(6);
                step4!(7);
                (
                    vreinterpretq_u64_u8(v0),
                    vreinterpretq_u64_u8(v1),
                    vreinterpretq_u64_u8(v2),
                    vreinterpretq_u64_u8(v3),
                )
            }};
        }

        macro_rules! combine_2vars_from_folded_neon {
            ($f00:expr, $f10:expr, $f01:expr, $f11:expr) => {{
                let (l10, m10, h10) = crate::gf2_kernels::k0_f128_mul_unreduced_neon($f10, v_rho0);
                let (l01, m01, h01) = crate::gf2_kernels::k0_f128_mul_unreduced_neon($f01, v_rho1);
                let (l11, m11, h11) = crate::gf2_kernels::k0_f128_mul_unreduced_neon($f11, v_rho01);
                let ll = veorq_u64(veorq_u64(l10, l01), l11);
                let mid = veorq_u64(veorq_u64(m10, m01), m11);
                let hh = veorq_u64(veorq_u64(h10, h01), h11);
                veorq_u64(
                    $f00,
                    crate::gf2_kernels::k0_f128_reduce_3lane_neon(ll, mid, hh),
                )
            }};
        }

        macro_rules! fold_2vars_one {
            ($wp:expr) => {{
                let wp = $wp;
                let w00 = wp.read_unaligned();
                let w10 = wp.add(1).read_unaligned();
                let w01 = wp.add(2).read_unaligned();
                let w11 = wp.add(3).read_unaligned();
                let u10 = w00 ^ w10;
                let u01 = w00 ^ w01;
                let u11 = u10 ^ (w01 ^ w11);
                let (f00, f10, f01, f11) = fold_single_table4!(w00, u10, u01, u11);
                combine_2vars_from_folded_neon!(f00, f10, f01, f11)
            }};
        }

        macro_rules! fold_2vars_ab_one {
            ($ap:expr, $bp:expr) => {{ (fold_2vars_one!($ap), fold_2vars_one!($bp)) }};
        }

        macro_rules! acc_unred_slot_with_sum {
            ($idx:literal, $u:expr, $v:expr, $u_sum:expr, $v_sum:expr) => {{
                let u = $u;
                let v = $v;
                let t_ll = vreinterpretq_u64_p128(vmull_p64(
                    vgetq_lane_u64::<0>(u),
                    vgetq_lane_u64::<0>(v),
                ));
                let t_mid = vreinterpretq_u64_p128(vmull_p64($u_sum, $v_sum));
                let t_hh = vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(u),
                    vreinterpretq_p64_u64(v),
                ));
                acc[$idx][0] = veorq_u64(acc[$idx][0], t_ll);
                acc[$idx][1] = veorq_u64(acc[$idx][1], t_mid);
                acc[$idx][2] = veorq_u64(acc[$idx][2], t_hh);
            }};
        }

        macro_rules! acc_unred_slot {
            ($idx:literal, $u:expr, $v:expr) => {{
                let (t_ll, t_mid, t_hh) = crate::gf2_kernels::k0_f128_mul_unreduced_neon($u, $v);
                acc[$idx][0] = veorq_u64(acc[$idx][0], t_ll);
                acc[$idx][1] = veorq_u64(acc[$idx][1], t_mid);
                acc[$idx][2] = veorq_u64(acc[$idx][2], t_hh);
            }};
        }

        macro_rules! lookahead_from_folded {
            ($p_lo:expr, $va00:expr, $vb00:expr, $va10:expr, $vb10:expr, $va01:expr, $vb01:expr, $va11:expr, $vb11:expr) => {{
                let veq = vld1q_u64(eq_ptr.add(2 * $p_lo));
                let (eq_r0, eq_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(veq);
                let (tb00, tb10) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    $vb00, $vb10, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                let (tb01, tb11) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    $vb01, $vb11, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                let da0 = veorq_u64($va00, $va10);
                let da1 = veorq_u64($va01, $va11);
                let da_0 = veorq_u64($va00, $va01);
                let da_1 = veorq_u64($va10, $va11);
                let da_i = veorq_u64(da0, da1);

                let dtb0 = veorq_u64(tb00, tb10);
                let dtb1 = veorq_u64(tb01, tb11);
                let dtb_0 = veorq_u64(tb00, tb01);
                let dtb_1 = veorq_u64(tb10, tb11);
                let dtb_i = veorq_u64(dtb0, dtb1);

                let sva10 = vgetq_lane_u64::<0>($va10) ^ vgetq_lane_u64::<1>($va10);
                let sva11 = vgetq_lane_u64::<0>($va11) ^ vgetq_lane_u64::<1>($va11);
                let sva01 = vgetq_lane_u64::<0>($va01) ^ vgetq_lane_u64::<1>($va01);
                let sda0 = vgetq_lane_u64::<0>(da0) ^ vgetq_lane_u64::<1>(da0);
                let sda1 = sva01 ^ sva11;
                let sda_1 = sva10 ^ sva11;
                let sda_0 = sda0 ^ (sva10 ^ sva01);
                let sda_i = sda0 ^ sda1;

                let stb10 = vgetq_lane_u64::<0>(tb10) ^ vgetq_lane_u64::<1>(tb10);
                let stb11 = vgetq_lane_u64::<0>(tb11) ^ vgetq_lane_u64::<1>(tb11);
                let stb01 = vgetq_lane_u64::<0>(tb01) ^ vgetq_lane_u64::<1>(tb01);
                let sdtb0 = vgetq_lane_u64::<0>(dtb0) ^ vgetq_lane_u64::<1>(dtb0);
                let sdtb1 = stb01 ^ stb11;
                let sdtb_1 = stb10 ^ stb11;
                let sdtb_0 = sdtb0 ^ (stb10 ^ stb01);
                let sdtb_i = sdtb0 ^ sdtb1;

                acc_unred_slot_with_sum!(0, $va10, tb10, sva10, stb10);
                acc_unred_slot_with_sum!(1, da0, dtb0, sda0, sdtb0);
                acc_unred_slot_with_sum!(2, $va11, tb11, sva11, stb11);
                acc_unred_slot_with_sum!(3, da1, dtb1, sda1, sdtb1);
                acc_unred_slot_with_sum!(4, $va01, tb01, sva01, stb01);
                acc_unred_slot_with_sum!(5, da_0, dtb_0, sda_0, sdtb_0);
                acc_unred_slot_with_sum!(6, da_1, dtb_1, sda_1, sdtb_1);
                acc_unred_slot_with_sum!(7, da_i, dtb_i, sda_i, sdtb_i);
            }};
        }

        macro_rules! round45_unit_b_step {
            ($p_lo:expr) => {{
                let pl = $p_lo;
                let q0l = 4 * pl;
                let q0g = q_base + q0l;
                let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                let va00 = fold_2vars_one!(ap0);
                let va10 = fold_2vars_one!(ap0.add(4));
                let va01 = fold_2vars_one!(ap0.add(8));
                let va11 = fold_2vars_one!(ap0.add(12));

                vst1q_u64(a_out_ptr.add(2 * q0l), va00);
                vst1q_u64(a_out_ptr.add(2 * (q0l + 1)), va10);
                vst1q_u64(a_out_ptr.add(2 * (q0l + 2)), va01);
                vst1q_u64(a_out_ptr.add(2 * (q0l + 3)), va11);
                vst1q_u64(b_out_ptr.add(2 * q0l), one_u64);
                vst1q_u64(b_out_ptr.add(2 * (q0l + 1)), one_u64);
                vst1q_u64(b_out_ptr.add(2 * (q0l + 2)), one_u64);
                vst1q_u64(b_out_ptr.add(2 * (q0l + 3)), one_u64);

                let veq = vld1q_u64(eq_ptr.add(2 * pl));
                acc_unred_slot!(0, va10, veq);
                acc_unred_slot!(2, va11, veq);
                acc_unred_slot!(4, va01, veq);
            }};
        }

        macro_rules! round45_general_step {
            ($p_lo:expr) => {{
                let pl = $p_lo;
                let q0l = 4 * pl;
                let q0g = q_base + q0l;
                let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                let bp0 = (b_pkt_ptr as *const u64).add(4 * q0g);
                let (va00, vb00) = fold_2vars_ab_one!(ap0, bp0);
                let (va10, vb10) = fold_2vars_ab_one!(ap0.add(4), bp0.add(4));
                let (va01, vb01) = fold_2vars_ab_one!(ap0.add(8), bp0.add(8));
                let (va11, vb11) = fold_2vars_ab_one!(ap0.add(12), bp0.add(12));

                vst1q_u64(a_out_ptr.add(2 * q0l), va00);
                vst1q_u64(a_out_ptr.add(2 * (q0l + 1)), va10);
                vst1q_u64(a_out_ptr.add(2 * (q0l + 2)), va01);
                vst1q_u64(a_out_ptr.add(2 * (q0l + 3)), va11);
                vst1q_u64(b_out_ptr.add(2 * q0l), vb00);
                vst1q_u64(b_out_ptr.add(2 * (q0l + 1)), vb10);
                vst1q_u64(b_out_ptr.add(2 * (q0l + 2)), vb01);
                vst1q_u64(b_out_ptr.add(2 * (q0l + 3)), vb11);

                lookahead_from_folded!(pl, va00, vb00, va10, vb10, va01, vb01, va11, vb11);
            }};
        }

        if is_unit_ones
            && pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
            && (q_base & 63) == 0
            && lo_size >= 16
            && (lo_size & 15) == 0
        {
            let n_blocks = lo_size >> 4;
            for blk in 0..n_blocks {
                let blk_p = blk << 4;
                let bp_blk = (b_pkt_ptr as *const u64).add(4 * (q_base + 4 * blk_p));
                let head_ones = (bp_blk.read_unaligned()
                    & bp_blk.add(1).read_unaligned()
                    & bp_blk.add(2).read_unaligned()
                    & bp_blk.add(3).read_unaligned()
                    & bp_blk.add(4).read_unaligned()
                    & bp_blk.add(5).read_unaligned()
                    & bp_blk.add(6).read_unaligned()
                    & bp_blk.add(7).read_unaligned()
                    & bp_blk.add(8).read_unaligned()
                    & bp_blk.add(9).read_unaligned()
                    & bp_blk.add(10).read_unaligned()
                    & bp_blk.add(11).read_unaligned()
                    & bp_blk.add(12).read_unaligned()
                    & bp_blk.add(13).read_unaligned()
                    & bp_blk.add(14).read_unaligned()
                    & bp_blk.add(15).read_unaligned())
                    == !0u64;
                if head_ones {
                    round45_unit_b_step!(blk_p);
                } else {
                    round45_general_step!(blk_p);
                }
                for p_lo in (blk_p + 1)..(blk_p + 15) {
                    round45_general_step!(p_lo);
                }
                {
                    let pl = blk_p + 15;
                    let q0l = 4 * pl;
                    let q0g = q_base + q0l;
                    let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                    let bp0 = (b_pkt_ptr as *const u64).add(4 * q0g);
                    let a10 = ap0.add(1).read_unaligned();
                    let b10 = bp0.add(1).read_unaligned();
                    let (va00, vb00) = if (a10 | b10) == 0 {
                        let a00 = ap0.read_unaligned();
                        let b00 = bp0.read_unaligned();
                        let mut va0 = vld1q_u8(table_data.add(((a00 & 0xFF) as usize) * 16));
                        let mut vb0 = vld1q_u8(table_data.add(((b00 & 0xFF) as usize) * 16));
                        macro_rules! step2 {
                            ($c:literal) => {{
                                let off = $c * STRIDE;
                                let sh = $c * 8;
                                va0 = veorq_u8(
                                    va0,
                                    vld1q_u8(
                                        table_data.add(off + (((a00 >> sh) & 0xFF) as usize) * 16),
                                    ),
                                );
                                vb0 = veorq_u8(
                                    vb0,
                                    vld1q_u8(
                                        table_data.add(off + (((b00 >> sh) & 0xFF) as usize) * 16),
                                    ),
                                );
                            }};
                        }
                        step2!(1);
                        step2!(2);
                        step2!(3);
                        step2!(4);
                        step2!(5);
                        step2!(6);
                        step2!(7);
                        crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                            vreinterpretq_u64_u8(va0),
                            vreinterpretq_u64_u8(vb0),
                            opr01_r0,
                            opr01_r1,
                            opr01_r0,
                            opr01_r1,
                        )
                    } else {
                        fold_2vars_ab_one!(ap0, bp0)
                    };
                    vst1q_u64(a_out_ptr.add(2 * q0l), va00);
                    vst1q_u64(a_out_ptr.add(2 * (q0l + 1)), zero);
                    vst1q_u64(a_out_ptr.add(2 * (q0l + 2)), zero);
                    vst1q_u64(a_out_ptr.add(2 * (q0l + 3)), zero);
                    vst1q_u64(b_out_ptr.add(2 * q0l), vb00);
                    vst1q_u64(b_out_ptr.add(2 * (q0l + 1)), zero);
                    vst1q_u64(b_out_ptr.add(2 * (q0l + 2)), zero);
                    vst1q_u64(b_out_ptr.add(2 * (q0l + 3)), zero);
                    let g00 = crate::gf2_kernels::k0_f128_mul_neon(va00, vb00);
                    let veq = vld1q_u64(eq_ptr.add(2 * pl));
                    let (t_ll, t_mid, t_hh) =
                        crate::gf2_kernels::k0_f128_mul_unreduced_neon(veq, g00);
                    p60_ll = veorq_u64(p60_ll, t_ll);
                    p60_mid = veorq_u64(p60_mid, t_mid);
                    p60_hh = veorq_u64(p60_hh, t_hh);
                }
            }
            acc[1][0] = veorq_u64(acc[1][0], p60_ll);
            acc[1][1] = veorq_u64(acc[1][1], p60_mid);
            acc[1][2] = veorq_u64(acc[1][2], p60_hh);
            acc[5][0] = veorq_u64(acc[5][0], p60_ll);
            acc[5][1] = veorq_u64(acc[5][1], p60_mid);
            acc[5][2] = veorq_u64(acc[5][2], p60_hh);
            acc[7][0] = veorq_u64(acc[7][0], p60_ll);
            acc[7][1] = veorq_u64(acc[7][1], p60_mid);
            acc[7][2] = veorq_u64(acc[7][2], p60_hh);
        } else {
            for p_lo in 0..lo_size {
                let q0l = 4 * p_lo;
                let q0g = q_base + q0l;
                let fold_one_padded = |qg: usize| -> (uint64x2_t, uint64x2_t) {
                    let j0 = 2 * qg;
                    let j1 = j0 + 1;
                    if (j0 & pair_in_block_mask) >= useful_pairs_inclusive {
                        return (zero, zero);
                    }
                    let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                    let bp = (b_pkt_ptr as *const u64).add(4 * qg);
                    let a00 = ap.read_unaligned();
                    let a10 = ap.add(1).read_unaligned();
                    let b00 = bp.read_unaligned();
                    let b10 = bp.add(1).read_unaligned();
                    let (a01, a11, b01, b11) = if (j1 & pair_in_block_mask) < useful_pairs_inclusive
                    {
                        (
                            ap.add(2).read_unaligned(),
                            ap.add(3).read_unaligned(),
                            bp.add(2).read_unaligned(),
                            bp.add(3).read_unaligned(),
                        )
                    } else {
                        (0, 0, 0, 0)
                    };
                    let ua10 = a00 ^ a10;
                    let ua01 = a00 ^ a01;
                    let ua11 = ua10 ^ (a01 ^ a11);
                    let ub10 = b00 ^ b10;
                    let ub01 = b00 ^ b01;
                    let ub11 = ub10 ^ (b01 ^ b11);
                    let (fa00, fa10, fa01, fa11) = fold_single_table4!(a00, ua10, ua01, ua11);
                    let (fb00, fb10, fb01, fb11) = fold_single_table4!(b00, ub10, ub01, ub11);
                    (
                        combine_2vars_from_folded_neon!(fa00, fa10, fa01, fa11),
                        combine_2vars_from_folded_neon!(fb00, fb10, fb01, fb11),
                    )
                };
                let (va00, vb00) = fold_one_padded(q0g);
                let (va10, vb10) = fold_one_padded(q0g + 1);
                let (va01, vb01) = fold_one_padded(q0g + 2);
                let (va11, vb11) = fold_one_padded(q0g + 3);
                vst1q_u64(a_out_ptr.add(2 * q0l), va00);
                vst1q_u64(a_out_ptr.add(2 * (q0l + 1)), va10);
                vst1q_u64(a_out_ptr.add(2 * (q0l + 2)), va01);
                vst1q_u64(a_out_ptr.add(2 * (q0l + 3)), va11);
                vst1q_u64(b_out_ptr.add(2 * q0l), vb00);
                vst1q_u64(b_out_ptr.add(2 * (q0l + 1)), vb10);
                vst1q_u64(b_out_ptr.add(2 * (q0l + 2)), vb01);
                vst1q_u64(b_out_ptr.add(2 * (q0l + 3)), vb11);
                lookahead_from_folded!(p_lo, va00, vb00, va10, vb10, va01, vb01, va11, vb11);
            }
        }

        let mut out = [F128::ZERO; 8];
        for k in 0..8 {
            let r = crate::gf2_kernels::k0_f128_reduce_3lane_neon(acc[k][0], acc[k][1], acc[k][2]);
            out[k] = F128 {
                lo: vgetq_lane_u64::<0>(r),
                hi: vgetq_lane_u64::<1>(r),
            };
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold_2vars_and_lookahead_chunk_neon_unchecked(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_0: F128,
    rho_1: F128,
    eq_lo: &[F128],
) -> [F128; 8] {
    use core::arch::aarch64::*;

    unsafe {
        let zero = vdupq_n_u64(0);
        let mut acc = [[zero; 3]; 8];
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr() as *const u64;
        let a_in_ptr = a_in.as_ptr() as *const u64;
        let b_in_ptr = b_in.as_ptr() as *const u64;
        let a_out_ptr = a_out.as_mut_ptr() as *mut u64;
        let b_out_ptr = b_out.as_mut_ptr() as *mut u64;

        let v_rho0 = vcombine_u64(vcreate_u64(rho_0.lo), vcreate_u64(rho_0.hi));
        let v_rho1 = vcombine_u64(vcreate_u64(rho_1.lo), vcreate_u64(rho_1.hi));
        let rho_01 = rho_0 * rho_1;
        let v_rho01 = vcombine_u64(vcreate_u64(rho_01.lo), vcreate_u64(rho_01.hi));

        macro_rules! fold_2vars_f128 {
            ($ptr:expr) => {{
                let p = $ptr;
                let w00 = vld1q_u64(p);
                let w10 = vld1q_u64(p.add(2));
                let w01 = vld1q_u64(p.add(4));
                let w11 = vld1q_u64(p.add(6));
                let f10 = veorq_u64(w00, w10);
                let f01 = veorq_u64(w00, w01);
                let f11 = veorq_u64(f10, veorq_u64(w01, w11));
                let (l10, m10, h10) = crate::gf2_kernels::k0_f128_mul_unreduced_neon(f10, v_rho0);
                let (l01, m01, h01) = crate::gf2_kernels::k0_f128_mul_unreduced_neon(f01, v_rho1);
                let (l11, m11, h11) = crate::gf2_kernels::k0_f128_mul_unreduced_neon(f11, v_rho01);
                let ll = veorq_u64(veorq_u64(l10, l01), l11);
                let mid = veorq_u64(veorq_u64(m10, m01), m11);
                let hh = veorq_u64(veorq_u64(h10, h01), h11);
                veorq_u64(
                    w00,
                    crate::gf2_kernels::k0_f128_reduce_3lane_neon(ll, mid, hh),
                )
            }};
        }

        macro_rules! acc_unred_slot_with_sum {
            ($idx:literal, $u:expr, $v:expr, $u_sum:expr, $v_sum:expr) => {{
                let u = $u;
                let v = $v;
                let t_ll = vreinterpretq_u64_p128(vmull_p64(
                    vgetq_lane_u64::<0>(u),
                    vgetq_lane_u64::<0>(v),
                ));
                let t_mid = vreinterpretq_u64_p128(vmull_p64($u_sum, $v_sum));
                let t_hh = vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(u),
                    vreinterpretq_p64_u64(v),
                ));
                acc[$idx][0] = veorq_u64(acc[$idx][0], t_ll);
                acc[$idx][1] = veorq_u64(acc[$idx][1], t_mid);
                acc[$idx][2] = veorq_u64(acc[$idx][2], t_hh);
            }};
        }

        for p_lo in 0..lo_size {
            let s = 32 * p_lo;
            let d = 8 * p_lo;
            let ap = a_in_ptr.add(s);
            let bp = b_in_ptr.add(s);
            let va00 = fold_2vars_f128!(ap);
            let vb00 = fold_2vars_f128!(bp);
            let va10 = fold_2vars_f128!(ap.add(8));
            let vb10 = fold_2vars_f128!(bp.add(8));
            let va01 = fold_2vars_f128!(ap.add(16));
            let vb01 = fold_2vars_f128!(bp.add(16));
            let va11 = fold_2vars_f128!(ap.add(24));
            let vb11 = fold_2vars_f128!(bp.add(24));

            vst1q_u64(a_out_ptr.add(d), va00);
            vst1q_u64(a_out_ptr.add(d + 2), va10);
            vst1q_u64(a_out_ptr.add(d + 4), va01);
            vst1q_u64(a_out_ptr.add(d + 6), va11);
            vst1q_u64(b_out_ptr.add(d), vb00);
            vst1q_u64(b_out_ptr.add(d + 2), vb10);
            vst1q_u64(b_out_ptr.add(d + 4), vb01);
            vst1q_u64(b_out_ptr.add(d + 6), vb11);

            let veq = vld1q_u64(eq_ptr.add(2 * p_lo));
            let (eq_r0, eq_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(veq);
            let (tb00, tb10) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                vb00, vb10, eq_r0, eq_r1, eq_r0, eq_r1,
            );
            let (tb01, tb11) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                vb01, vb11, eq_r0, eq_r1, eq_r0, eq_r1,
            );
            let da0 = veorq_u64(va00, va10);
            let da1 = veorq_u64(va01, va11);
            let da_0 = veorq_u64(va00, va01);
            let da_1 = veorq_u64(va10, va11);
            let da_i = veorq_u64(da0, da1);

            let dtb0 = veorq_u64(tb00, tb10);
            let dtb1 = veorq_u64(tb01, tb11);
            let dtb_0 = veorq_u64(tb00, tb01);
            let dtb_1 = veorq_u64(tb10, tb11);
            let dtb_i = veorq_u64(dtb0, dtb1);

            let sva10 = vgetq_lane_u64::<0>(va10) ^ vgetq_lane_u64::<1>(va10);
            let sva11 = vgetq_lane_u64::<0>(va11) ^ vgetq_lane_u64::<1>(va11);
            let sva01 = vgetq_lane_u64::<0>(va01) ^ vgetq_lane_u64::<1>(va01);
            let sda0 = vgetq_lane_u64::<0>(da0) ^ vgetq_lane_u64::<1>(da0);
            let sda1 = sva01 ^ sva11;
            let sda_1 = sva10 ^ sva11;
            let sda_0 = sda0 ^ (sva10 ^ sva01);
            let sda_i = sda0 ^ sda1;

            let stb10 = vgetq_lane_u64::<0>(tb10) ^ vgetq_lane_u64::<1>(tb10);
            let stb11 = vgetq_lane_u64::<0>(tb11) ^ vgetq_lane_u64::<1>(tb11);
            let stb01 = vgetq_lane_u64::<0>(tb01) ^ vgetq_lane_u64::<1>(tb01);
            let sdtb0 = vgetq_lane_u64::<0>(dtb0) ^ vgetq_lane_u64::<1>(dtb0);
            let sdtb1 = stb01 ^ stb11;
            let sdtb_1 = stb10 ^ stb11;
            let sdtb_0 = sdtb0 ^ (stb10 ^ stb01);
            let sdtb_i = sdtb0 ^ sdtb1;

            acc_unred_slot_with_sum!(0, va10, tb10, sva10, stb10);
            acc_unred_slot_with_sum!(1, da0, dtb0, sda0, sdtb0);
            acc_unred_slot_with_sum!(2, va11, tb11, sva11, stb11);
            acc_unred_slot_with_sum!(3, da1, dtb1, sda1, sdtb1);
            acc_unred_slot_with_sum!(4, va01, tb01, sva01, stb01);
            acc_unred_slot_with_sum!(5, da_0, dtb_0, sda_0, sdtb_0);
            acc_unred_slot_with_sum!(6, da_1, dtb_1, sda_1, sdtb_1);
            acc_unred_slot_with_sum!(7, da_i, dtb_i, sda_i, sdtb_i);
        }

        let mut out = [F128::ZERO; 8];
        for k in 0..8 {
            let r = crate::gf2_kernels::k0_f128_reduce_3lane_neon(acc[k][0], acc[k][1], acc[k][2]);
            out[k] = F128 {
                lo: vgetq_lane_u64::<0>(r),
                hi: vgetq_lane_u64::<1>(r),
            };
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold_2vars_and_round_pair_chunk_neon_unchecked(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_0: F128,
    rho_1: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use core::arch::aarch64::*;

    unsafe {
        let zero = vdupq_n_u64(0);
        let mut p1_ll = zero;
        let mut p1_mid = zero;
        let mut p1_hh = zero;
        let mut pinf_ll = zero;
        let mut pinf_mid = zero;
        let mut pinf_hh = zero;
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr() as *const u64;
        let a_in_ptr = a_in.as_ptr() as *const u64;
        let b_in_ptr = b_in.as_ptr() as *const u64;
        let a_out_ptr = a_out.as_mut_ptr() as *mut u64;
        let b_out_ptr = b_out.as_mut_ptr() as *mut u64;

        let v_rho0 = vcombine_u64(vcreate_u64(rho_0.lo), vcreate_u64(rho_0.hi));
        let v_rho1 = vcombine_u64(vcreate_u64(rho_1.lo), vcreate_u64(rho_1.hi));
        let rho_01 = rho_0 * rho_1;
        let v_rho01 = vcombine_u64(vcreate_u64(rho_01.lo), vcreate_u64(rho_01.hi));

        macro_rules! fold_2vars_f128 {
            ($ptr:expr) => {{
                let p = $ptr;
                let w00 = vld1q_u64(p);
                let w10 = vld1q_u64(p.add(2));
                let w01 = vld1q_u64(p.add(4));
                let w11 = vld1q_u64(p.add(6));
                let f10 = veorq_u64(w00, w10);
                let f01 = veorq_u64(w00, w01);
                let f11 = veorq_u64(f10, veorq_u64(w01, w11));
                let (l10, m10, h10) = crate::gf2_kernels::k0_f128_mul_unreduced_neon(f10, v_rho0);
                let (l01, m01, h01) = crate::gf2_kernels::k0_f128_mul_unreduced_neon(f01, v_rho1);
                let (l11, m11, h11) = crate::gf2_kernels::k0_f128_mul_unreduced_neon(f11, v_rho01);
                let ll = veorq_u64(veorq_u64(l10, l01), l11);
                let mid = veorq_u64(veorq_u64(m10, m01), m11);
                let hh = veorq_u64(veorq_u64(h10, h01), h11);
                veorq_u64(
                    w00,
                    crate::gf2_kernels::k0_f128_reduce_3lane_neon(ll, mid, hh),
                )
            }};
        }

        for x_lo in 0..lo_size {
            let s = 16 * x_lo;
            let d = 4 * x_lo;
            let ap = a_in_ptr.add(s);
            let bp = b_in_ptr.add(s);
            let va0 = fold_2vars_f128!(ap);
            let vb0 = fold_2vars_f128!(bp);
            let va1 = fold_2vars_f128!(ap.add(8));
            let vb1 = fold_2vars_f128!(bp.add(8));

            vst1q_u64(a_out_ptr.add(d), va0);
            vst1q_u64(a_out_ptr.add(d + 2), va1);
            vst1q_u64(b_out_ptr.add(d), vb0);
            vst1q_u64(b_out_ptr.add(d + 2), vb1);

            let vad = veorq_u64(va0, va1);
            let vbd = veorq_u64(vb0, vb1);
            let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_neon(va1, vad, vb1, vbd);
            let veq = vld1q_u64(eq_ptr.add(2 * x_lo));
            crate::gf2_kernels::k0_f128_mul_acc_pair_neon(
                veq,
                g1,
                ginf,
                &mut p1_ll,
                &mut p1_mid,
                &mut p1_hh,
                &mut pinf_ll,
                &mut pinf_mid,
                &mut pinf_hh,
            );
        }

        let res1 = crate::gf2_kernels::k0_f128_reduce_3lane_neon(p1_ll, p1_mid, p1_hh);
        let resi = crate::gf2_kernels::k0_f128_reduce_3lane_neon(pinf_ll, pinf_mid, pinf_hh);
        (
            F128 {
                lo: vgetq_lane_u64::<0>(res1),
                hi: vgetq_lane_u64::<1>(res1),
            },
            F128 {
                lo: vgetq_lane_u64::<0>(resi),
                hi: vgetq_lane_u64::<1>(resi),
            },
        )
    }
}

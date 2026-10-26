use super::super::{F8, F128, InvNttTableByteSingleGf8, N_CHUNKS};

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [super::super::KaratsubaAcc; 64],
    partial_c: &mut [super::super::KaratsubaAcc; 64],
) {
    use core::arch::aarch64::*;

    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let veq = vld1q_u64((&eq_lo_val as *const F128).cast::<u64>());
        let eq_lo = vgetq_lane_u64::<0>(veq);
        let eq_sum = eq_lo ^ vgetq_lane_u64::<1>(veq);
        macro_rules! mul_karatsuba_acc_neon {
            ($dst:expr, $ab:expr) => {{
                let v = vreinterpretq_u64_u8($ab);
                let v_lo = vgetq_lane_u64::<0>(v);
                let v_sum = v_lo ^ vgetq_lane_u64::<1>(v);
                let p_ll = vreinterpretq_u64_p128(vmull_p64(v_lo, eq_lo));
                let p_mid = vreinterpretq_u64_p128(vmull_p64(v_sum, eq_sum));
                let p_hh = vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(v),
                    vreinterpretq_p64_u64(veq),
                ));
                let ptr = ($dst as *mut super::super::KaratsubaAcc).cast::<u64>();
                vst1q_u64(ptr, veorq_u64(vld1q_u64(ptr), p_ll));
                vst1q_u64(ptr.add(2), veorq_u64(vld1q_u64(ptr.add(2)), p_mid));
                vst1q_u64(ptr.add(4), veorq_u64(vld1q_u64(ptr.add(4)), p_hh));
            }};
        }
        for tile in 0..16 {
            let l0 = tile * 4;
            let l1 = l0 + 1;
            let l2 = l0 + 2;
            let l3 = l0 + 3;
            let ab_row0 = &chunk_ab_bytes[0];
            let c_row0 = &chunk_c_bytes[0];
            let mut ab0 = vld1q_u8(convert_ptr.add(ab_row0[l0] as usize * 16));
            let mut ab1 = vld1q_u8(convert_ptr.add(ab_row0[l1] as usize * 16));
            let mut ab2 = vld1q_u8(convert_ptr.add(ab_row0[l2] as usize * 16));
            let mut ab3 = vld1q_u8(convert_ptr.add(ab_row0[l3] as usize * 16));
            let mut c0 = vld1q_u8(convert_ptr.add(c_row0[l0] as usize * 16));
            let mut c1 = vld1q_u8(convert_ptr.add(c_row0[l1] as usize * 16));
            let mut c2 = vld1q_u8(convert_ptr.add(c_row0[l2] as usize * 16));
            let mut c3 = vld1q_u8(convert_ptr.add(c_row0[l3] as usize * 16));
            for b_med in 1..n_b_med {
                let tab = convert_ptr.add(b_med * 256 * 16);
                let ab_row = &chunk_ab_bytes[b_med];
                let c_row = &chunk_c_bytes[b_med];
                ab0 = veorq_u8(ab0, vld1q_u8(tab.add(ab_row[l0] as usize * 16)));
                ab1 = veorq_u8(ab1, vld1q_u8(tab.add(ab_row[l1] as usize * 16)));
                ab2 = veorq_u8(ab2, vld1q_u8(tab.add(ab_row[l2] as usize * 16)));
                ab3 = veorq_u8(ab3, vld1q_u8(tab.add(ab_row[l3] as usize * 16)));
                c0 = veorq_u8(c0, vld1q_u8(tab.add(c_row[l0] as usize * 16)));
                c1 = veorq_u8(c1, vld1q_u8(tab.add(c_row[l1] as usize * 16)));
                c2 = veorq_u8(c2, vld1q_u8(tab.add(c_row[l2] as usize * 16)));
                c3 = veorq_u8(c3, vld1q_u8(tab.add(c_row[l3] as usize * 16)));
            }
            mul_karatsuba_acc_neon!(&mut partial_ab[l0], ab0);
            mul_karatsuba_acc_neon!(&mut partial_c[l0], c0);
            mul_karatsuba_acc_neon!(&mut partial_ab[l1], ab1);
            mul_karatsuba_acc_neon!(&mut partial_c[l1], c1);
            mul_karatsuba_acc_neon!(&mut partial_ab[l2], ab2);
            mul_karatsuba_acc_neon!(&mut partial_c[l2], c2);
            mul_karatsuba_acc_neon!(&mut partial_ab[l3], ab3);
            mul_karatsuba_acc_neon!(&mut partial_c[l3], c3);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [super::super::KaratsubaAcc; 64],
    partial_c_0: &mut [super::super::KaratsubaAcc; 64],
    partial_c_1: &mut [super::super::KaratsubaAcc; 64],
) {
    use core::arch::aarch64::*;

    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let veq = vld1q_u64((&eq_lo_val as *const F128).cast::<u64>());
        let eq_lo = vgetq_lane_u64::<0>(veq);
        let eq_sum = eq_lo ^ vgetq_lane_u64::<1>(veq);
        macro_rules! mul_karatsuba_acc_neon {
            ($dst:expr, $ab:expr) => {{
                let v = vreinterpretq_u64_u8($ab);
                let v_lo = vgetq_lane_u64::<0>(v);
                let v_sum = v_lo ^ vgetq_lane_u64::<1>(v);
                let p_ll = vreinterpretq_u64_p128(vmull_p64(v_lo, eq_lo));
                let p_mid = vreinterpretq_u64_p128(vmull_p64(v_sum, eq_sum));
                let p_hh = vreinterpretq_u64_p128(vmull_high_p64(
                    vreinterpretq_p64_u64(v),
                    vreinterpretq_p64_u64(veq),
                ));
                let ptr = ($dst as *mut super::super::KaratsubaAcc).cast::<u64>();
                vst1q_u64(ptr, veorq_u64(vld1q_u64(ptr), p_ll));
                vst1q_u64(ptr.add(2), veorq_u64(vld1q_u64(ptr.add(2)), p_mid));
                vst1q_u64(ptr.add(4), veorq_u64(vld1q_u64(ptr.add(4)), p_hh));
            }};
        }
        for tile in 0..16 {
            let l0 = tile * 4;
            let l1 = l0 + 1;
            let l2 = l0 + 2;
            let l3 = l0 + 3;
            let ab_row0 = &chunk_ab_bytes[0];
            let c_row0 = &chunk_c_bytes[0];
            let mut ab0 = vld1q_u8(convert_ptr.add(ab_row0[l0] as usize * 16));
            let mut ab1 = vld1q_u8(convert_ptr.add(ab_row0[l1] as usize * 16));
            let mut ab2 = vld1q_u8(convert_ptr.add(ab_row0[l2] as usize * 16));
            let mut ab3 = vld1q_u8(convert_ptr.add(ab_row0[l3] as usize * 16));
            let mut c0_0 = vld1q_u8(convert_ptr.add((c_row0[l0] & 0x55) as usize * 16));
            let mut c0_1 = vld1q_u8(convert_ptr.add((c_row0[l1] & 0x55) as usize * 16));
            let mut c0_2 = vld1q_u8(convert_ptr.add((c_row0[l2] & 0x55) as usize * 16));
            let mut c0_3 = vld1q_u8(convert_ptr.add((c_row0[l3] & 0x55) as usize * 16));
            let mut c1_0 = vld1q_u8(convert_ptr.add((c_row0[l0] & 0xaa) as usize * 16));
            let mut c1_1 = vld1q_u8(convert_ptr.add((c_row0[l1] & 0xaa) as usize * 16));
            let mut c1_2 = vld1q_u8(convert_ptr.add((c_row0[l2] & 0xaa) as usize * 16));
            let mut c1_3 = vld1q_u8(convert_ptr.add((c_row0[l3] & 0xaa) as usize * 16));
            for b_med in 1..n_b_med {
                let tab = convert_ptr.add(b_med * 256 * 16);
                let ab_row = &chunk_ab_bytes[b_med];
                let c_row = &chunk_c_bytes[b_med];
                ab0 = veorq_u8(ab0, vld1q_u8(tab.add(ab_row[l0] as usize * 16)));
                ab1 = veorq_u8(ab1, vld1q_u8(tab.add(ab_row[l1] as usize * 16)));
                ab2 = veorq_u8(ab2, vld1q_u8(tab.add(ab_row[l2] as usize * 16)));
                ab3 = veorq_u8(ab3, vld1q_u8(tab.add(ab_row[l3] as usize * 16)));
                c0_0 = veorq_u8(c0_0, vld1q_u8(tab.add((c_row[l0] & 0x55) as usize * 16)));
                c0_1 = veorq_u8(c0_1, vld1q_u8(tab.add((c_row[l1] & 0x55) as usize * 16)));
                c0_2 = veorq_u8(c0_2, vld1q_u8(tab.add((c_row[l2] & 0x55) as usize * 16)));
                c0_3 = veorq_u8(c0_3, vld1q_u8(tab.add((c_row[l3] & 0x55) as usize * 16)));
                c1_0 = veorq_u8(c1_0, vld1q_u8(tab.add((c_row[l0] & 0xaa) as usize * 16)));
                c1_1 = veorq_u8(c1_1, vld1q_u8(tab.add((c_row[l1] & 0xaa) as usize * 16)));
                c1_2 = veorq_u8(c1_2, vld1q_u8(tab.add((c_row[l2] & 0xaa) as usize * 16)));
                c1_3 = veorq_u8(c1_3, vld1q_u8(tab.add((c_row[l3] & 0xaa) as usize * 16)));
            }
            mul_karatsuba_acc_neon!(&mut partial_ab[l0], ab0);
            mul_karatsuba_acc_neon!(&mut partial_c_0[l0], c0_0);
            mul_karatsuba_acc_neon!(&mut partial_c_1[l0], c1_0);
            mul_karatsuba_acc_neon!(&mut partial_ab[l1], ab1);
            mul_karatsuba_acc_neon!(&mut partial_c_0[l1], c0_1);
            mul_karatsuba_acc_neon!(&mut partial_c_1[l1], c1_1);
            mul_karatsuba_acc_neon!(&mut partial_ab[l2], ab2);
            mul_karatsuba_acc_neon!(&mut partial_c_0[l2], c0_2);
            mul_karatsuba_acc_neon!(&mut partial_c_1[l2], c1_2);
            mul_karatsuba_acc_neon!(&mut partial_ab[l3], ab3);
            mul_karatsuba_acc_neon!(&mut partial_c_0[l3], c0_3);
            mul_karatsuba_acc_neon!(&mut partial_c_1[l3], c1_3);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_batch(
    batch_ab_bytes: &[[[u8; 64]; 16]; super::super::URM_BATCH],
    batch_c_bytes: &[[[u8; 64]; 16]; super::super::URM_BATCH],
    batch_n_b_med: &[u8; super::super::URM_BATCH],
    batch_eq_lo: &[F128; super::super::URM_BATCH],
    batch_len: usize,
    convert: &[F128],
    partial_ab: &mut [super::super::KaratsubaAcc; 64],
    partial_c: &mut [super::super::KaratsubaAcc; 64],
) {
    let _ = batch_n_b_med;
    use core::arch::aarch64::*;

    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let zero = vdupq_n_u8(0);
        let mut conv_ab = [[zero; 64]; super::super::URM_BATCH];
        let mut conv_c = [[zero; 64]; super::super::URM_BATCH];

        for q in 0..4 {
            let b_base = q * 4;
            let t0 = convert_ptr.add(b_base * 256 * 16);
            let t1 = convert_ptr.add((b_base + 1) * 256 * 16);
            let t2 = convert_ptr.add((b_base + 2) * 256 * 16);
            let t3 = convert_ptr.add((b_base + 3) * 256 * 16);
            for b_idx in 0..batch_len {
                let ab_rows = &batch_ab_bytes[b_idx];
                let c_rows = &batch_c_bytes[b_idx];
                let dst_ab = &mut conv_ab[b_idx];
                let dst_c = &mut conv_c[b_idx];
                for tile in 0..16 {
                    let l0 = tile * 4;
                    let l1 = l0 + 1;
                    let l2 = l0 + 2;
                    let l3 = l0 + 3;
                    let ab_r0 = &ab_rows[b_base];
                    let c_r0 = &c_rows[b_base];
                    let mut ab0 = vld1q_u8(t0.add(ab_r0[l0] as usize * 16));
                    let mut ab1 = vld1q_u8(t0.add(ab_r0[l1] as usize * 16));
                    let mut ab2 = vld1q_u8(t0.add(ab_r0[l2] as usize * 16));
                    let mut ab3 = vld1q_u8(t0.add(ab_r0[l3] as usize * 16));
                    let mut c0 = vld1q_u8(t0.add(c_r0[l0] as usize * 16));
                    let mut c1 = vld1q_u8(t0.add(c_r0[l1] as usize * 16));
                    let mut c2 = vld1q_u8(t0.add(c_r0[l2] as usize * 16));
                    let mut c3 = vld1q_u8(t0.add(c_r0[l3] as usize * 16));

                    macro_rules! xor_step {
                        ($tb:ident, $idx:expr) => {{
                            let ab_r = &ab_rows[$idx];
                            let c_r = &c_rows[$idx];
                            ab0 = veorq_u8(ab0, vld1q_u8($tb.add(ab_r[l0] as usize * 16)));
                            ab1 = veorq_u8(ab1, vld1q_u8($tb.add(ab_r[l1] as usize * 16)));
                            ab2 = veorq_u8(ab2, vld1q_u8($tb.add(ab_r[l2] as usize * 16)));
                            ab3 = veorq_u8(ab3, vld1q_u8($tb.add(ab_r[l3] as usize * 16)));
                            c0 = veorq_u8(c0, vld1q_u8($tb.add(c_r[l0] as usize * 16)));
                            c1 = veorq_u8(c1, vld1q_u8($tb.add(c_r[l1] as usize * 16)));
                            c2 = veorq_u8(c2, vld1q_u8($tb.add(c_r[l2] as usize * 16)));
                            c3 = veorq_u8(c3, vld1q_u8($tb.add(c_r[l3] as usize * 16)));
                        }};
                    }
                    xor_step!(t1, b_base + 1);
                    xor_step!(t2, b_base + 2);
                    xor_step!(t3, b_base + 3);

                    if q == 0 {
                        dst_ab[l0] = ab0;
                        dst_ab[l1] = ab1;
                        dst_ab[l2] = ab2;
                        dst_ab[l3] = ab3;
                        dst_c[l0] = c0;
                        dst_c[l1] = c1;
                        dst_c[l2] = c2;
                        dst_c[l3] = c3;
                    } else {
                        dst_ab[l0] = veorq_u8(dst_ab[l0], ab0);
                        dst_ab[l1] = veorq_u8(dst_ab[l1], ab1);
                        dst_ab[l2] = veorq_u8(dst_ab[l2], ab2);
                        dst_ab[l3] = veorq_u8(dst_ab[l3], ab3);
                        dst_c[l0] = veorq_u8(dst_c[l0], c0);
                        dst_c[l1] = veorq_u8(dst_c[l1], c1);
                        dst_c[l2] = veorq_u8(dst_c[l2], c2);
                        dst_c[l3] = veorq_u8(dst_c[l3], c3);
                    }
                }
            }
        }

        #[cfg(target_feature = "aes")]
        {
            let zero_u64 = vdupq_n_u64(0);
            let mut veq_r0 = [zero_u64; super::super::URM_BATCH];
            let mut veq_r1 = [zero_u64; super::super::URM_BATCH];
            for b_idx in 0..batch_len {
                let eq_lo_val = batch_eq_lo[b_idx];
                let veq = vld1q_u64((&eq_lo_val as *const F128).cast::<u64>());
                let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(veq);
                veq_r0[b_idx] = r0;
                veq_r1[b_idx] = r1;
            }
            for l in 0..64 {
                let ptr_ab = (&mut partial_ab[l].ll as *mut F128).cast::<u64>();
                let ptr_c = (&mut partial_c[l].ll as *mut F128).cast::<u64>();
                let mut acc_ab = vld1q_u64(ptr_ab);
                let mut acc_c = vld1q_u64(ptr_c);
                for b_idx in 0..batch_len {
                    let r0 = veq_r0[b_idx];
                    let r1 = veq_r1[b_idx];
                    let vab = vreinterpretq_u64_u8(conv_ab[b_idx][l]);
                    let vc = vreinterpretq_u64_u8(conv_c[b_idx][l]);
                    let (p_ab, p_c) =
                        crate::gf2_kernels::k0_f128_mul_const_vec2_neon(vab, vc, r0, r1, r0, r1);
                    acc_ab = veorq_u64(acc_ab, p_ab);
                    acc_c = veorq_u64(acc_c, p_c);
                }
                vst1q_u64(ptr_ab, acc_ab);
                vst1q_u64(ptr_c, acc_c);
            }
            return;
        }
        let zero_u64 = vdupq_n_u64(0);
        let mut veqs = [zero_u64; super::super::URM_BATCH];
        let mut eq_los = [0u64; super::super::URM_BATCH];
        let mut eq_sums = [0u64; super::super::URM_BATCH];
        for b_idx in 0..batch_len {
            let eq_lo_val = batch_eq_lo[b_idx];
            let veq = vld1q_u64((&eq_lo_val as *const F128).cast::<u64>());
            let eq_lo = vgetq_lane_u64::<0>(veq);
            veqs[b_idx] = veq;
            eq_los[b_idx] = eq_lo;
            eq_sums[b_idx] = eq_lo ^ vgetq_lane_u64::<1>(veq);
        }

        for l in 0..64 {
            let mut ab_ll = zero_u64;
            let mut ab_mid = zero_u64;
            let mut ab_hh = zero_u64;
            let mut c_ll = zero_u64;
            let mut c_mid = zero_u64;
            let mut c_hh = zero_u64;
            for b_idx in 0..batch_len {
                let veq = veqs[b_idx];
                let eq_lo = eq_los[b_idx];
                let eq_sum = eq_sums[b_idx];
                let vab = vreinterpretq_u64_u8(conv_ab[b_idx][l]);
                let vc = vreinterpretq_u64_u8(conv_c[b_idx][l]);
                let ab_lo = vgetq_lane_u64::<0>(vab);
                let ab_sum = ab_lo ^ vgetq_lane_u64::<1>(vab);
                let c_lo = vgetq_lane_u64::<0>(vc);
                let c_sum = c_lo ^ vgetq_lane_u64::<1>(vc);
                ab_ll = veorq_u64(ab_ll, vreinterpretq_u64_p128(vmull_p64(ab_lo, eq_lo)));
                ab_mid = veorq_u64(ab_mid, vreinterpretq_u64_p128(vmull_p64(ab_sum, eq_sum)));
                ab_hh = veorq_u64(
                    ab_hh,
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64(vab),
                        vreinterpretq_p64_u64(veq),
                    )),
                );
                c_ll = veorq_u64(c_ll, vreinterpretq_u64_p128(vmull_p64(c_lo, eq_lo)));
                c_mid = veorq_u64(c_mid, vreinterpretq_u64_p128(vmull_p64(c_sum, eq_sum)));
                c_hh = veorq_u64(
                    c_hh,
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64(vc),
                        vreinterpretq_p64_u64(veq),
                    )),
                );
            }
            let ptr_ab = (&mut partial_ab[l] as *mut super::super::KaratsubaAcc).cast::<u64>();
            vst1q_u64(ptr_ab, veorq_u64(vld1q_u64(ptr_ab), ab_ll));
            vst1q_u64(ptr_ab.add(2), veorq_u64(vld1q_u64(ptr_ab.add(2)), ab_mid));
            vst1q_u64(ptr_ab.add(4), veorq_u64(vld1q_u64(ptr_ab.add(4)), ab_hh));
            let ptr_c = (&mut partial_c[l] as *mut super::super::KaratsubaAcc).cast::<u64>();
            vst1q_u64(ptr_c, veorq_u64(vld1q_u64(ptr_c), c_ll));
            vst1q_u64(ptr_c.add(2), veorq_u64(vld1q_u64(ptr_c.add(2)), c_mid));
            vst1q_u64(ptr_c.add(4), veorq_u64(vld1q_u64(ptr_c.add(4)), c_hh));
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_batch_with_s_hat_v(
    batch_ab_bytes: &[[[u8; 64]; 16]; super::super::URM_BATCH],
    batch_c_bytes: &[[[u8; 64]; 16]; super::super::URM_BATCH],
    batch_n_b_med: &[u8; super::super::URM_BATCH],
    batch_eq_lo: &[F128; super::super::URM_BATCH],
    batch_len: usize,
    convert: &[F128],
    convert_c_pair: &[F128],
    partial_ab: &mut [super::super::KaratsubaAcc; 64],
    partial_c_0: &mut [super::super::KaratsubaAcc; 64],
    partial_c_1: &mut [super::super::KaratsubaAcc; 64],
) {
    use core::arch::aarch64::*;

    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let convert_c_ptr = convert_c_pair.as_ptr() as *const u8;
        let zero = vdupq_n_u8(0);
        let mut conv_ab = [[zero; 64]; super::super::URM_BATCH];
        let mut conv_c0 = [[zero; 64]; super::super::URM_BATCH];
        let mut conv_c1 = [[zero; 64]; super::super::URM_BATCH];

        for q in 0..4 {
            let b_base = q * 4;
            let t0 = convert_ptr.add(b_base * 256 * 16);
            let t1 = convert_ptr.add((b_base + 1) * 256 * 16);
            let t2 = convert_ptr.add((b_base + 2) * 256 * 16);
            let t3 = convert_ptr.add((b_base + 3) * 256 * 16);
            for b_idx in 0..batch_len {
                let ab_rows = &batch_ab_bytes[b_idx];
                let dst_ab = &mut conv_ab[b_idx];
                let ab_r0 = &ab_rows[b_base];
                let ab_r1 = &ab_rows[b_base + 1];
                let ab_r2 = &ab_rows[b_base + 2];
                let has_r3 = (b_base + 3) < (batch_n_b_med[b_idx] as usize);
                if has_r3 {
                    let ab_r3 = &ab_rows[b_base + 3];
                    for tile in 0..16 {
                        let l0 = tile * 4;
                        let l1 = l0 + 1;
                        let l2 = l0 + 2;
                        let l3 = l0 + 3;
                        let mut ab0 = vld1q_u8(t0.add(ab_r0[l0] as usize * 16));
                        let mut ab1 = vld1q_u8(t0.add(ab_r0[l1] as usize * 16));
                        let mut ab2 = vld1q_u8(t0.add(ab_r0[l2] as usize * 16));
                        let mut ab3 = vld1q_u8(t0.add(ab_r0[l3] as usize * 16));

                        ab0 = veorq_u8(ab0, vld1q_u8(t1.add(ab_r1[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t1.add(ab_r1[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t1.add(ab_r1[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t1.add(ab_r1[l3] as usize * 16)));

                        ab0 = veorq_u8(ab0, vld1q_u8(t2.add(ab_r2[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t2.add(ab_r2[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t2.add(ab_r2[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t2.add(ab_r2[l3] as usize * 16)));

                        ab0 = veorq_u8(ab0, vld1q_u8(t3.add(ab_r3[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t3.add(ab_r3[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t3.add(ab_r3[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t3.add(ab_r3[l3] as usize * 16)));

                        if q == 0 {
                            dst_ab[l0] = ab0;
                            dst_ab[l1] = ab1;
                            dst_ab[l2] = ab2;
                            dst_ab[l3] = ab3;
                        } else {
                            dst_ab[l0] = veorq_u8(dst_ab[l0], ab0);
                            dst_ab[l1] = veorq_u8(dst_ab[l1], ab1);
                            dst_ab[l2] = veorq_u8(dst_ab[l2], ab2);
                            dst_ab[l3] = veorq_u8(dst_ab[l3], ab3);
                        }
                    }
                } else {
                    for tile in 0..16 {
                        let l0 = tile * 4;
                        let l1 = l0 + 1;
                        let l2 = l0 + 2;
                        let l3 = l0 + 3;
                        let mut ab0 = vld1q_u8(t0.add(ab_r0[l0] as usize * 16));
                        let mut ab1 = vld1q_u8(t0.add(ab_r0[l1] as usize * 16));
                        let mut ab2 = vld1q_u8(t0.add(ab_r0[l2] as usize * 16));
                        let mut ab3 = vld1q_u8(t0.add(ab_r0[l3] as usize * 16));

                        ab0 = veorq_u8(ab0, vld1q_u8(t1.add(ab_r1[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t1.add(ab_r1[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t1.add(ab_r1[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t1.add(ab_r1[l3] as usize * 16)));

                        ab0 = veorq_u8(ab0, vld1q_u8(t2.add(ab_r2[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t2.add(ab_r2[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t2.add(ab_r2[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t2.add(ab_r2[l3] as usize * 16)));

                        dst_ab[l0] = veorq_u8(dst_ab[l0], ab0);
                        dst_ab[l1] = veorq_u8(dst_ab[l1], ab1);
                        dst_ab[l2] = veorq_u8(dst_ab[l2], ab2);
                        dst_ab[l3] = veorq_u8(dst_ab[l3], ab3);
                    }
                }
            }
        }

        for h in 0..2 {
            let m_base = h * 4;
            let r_base = h * 8;
            let tc0 = convert_c_ptr.add(m_base * 256 * 16);
            let tc1 = convert_c_ptr.add((m_base + 1) * 256 * 16);
            let tc2 = convert_c_ptr.add((m_base + 2) * 256 * 16);
            let tc3 = convert_c_ptr.add((m_base + 3) * 256 * 16);
            for b_idx in 0..batch_len {
                let c_rows = &batch_c_bytes[b_idx];
                let dst_c0 = &mut conv_c0[b_idx];
                let dst_c1 = &mut conv_c1[b_idx];
                let c0_r0 = &c_rows[r_base];
                let c1_r0 = &c_rows[r_base + 1];
                let c0_r1 = &c_rows[r_base + 2];
                let c1_r1 = &c_rows[r_base + 3];
                let c0_r2 = &c_rows[r_base + 4];
                let c1_r2 = &c_rows[r_base + 5];
                let c0_r3 = &c_rows[r_base + 6];
                let c1_r3 = &c_rows[r_base + 7];
                for tile in 0..16 {
                    let l0 = tile * 4;
                    let l1 = l0 + 1;
                    let l2 = l0 + 2;
                    let l3 = l0 + 3;
                    let mut c0_0 = vld1q_u8(tc0.add(c0_r0[l0] as usize * 16));
                    let mut c0_1 = vld1q_u8(tc0.add(c0_r0[l1] as usize * 16));
                    let mut c0_2 = vld1q_u8(tc0.add(c0_r0[l2] as usize * 16));
                    let mut c0_3 = vld1q_u8(tc0.add(c0_r0[l3] as usize * 16));

                    c0_0 = veorq_u8(c0_0, vld1q_u8(tc1.add(c0_r1[l0] as usize * 16)));
                    c0_1 = veorq_u8(c0_1, vld1q_u8(tc1.add(c0_r1[l1] as usize * 16)));
                    c0_2 = veorq_u8(c0_2, vld1q_u8(tc1.add(c0_r1[l2] as usize * 16)));
                    c0_3 = veorq_u8(c0_3, vld1q_u8(tc1.add(c0_r1[l3] as usize * 16)));

                    c0_0 = veorq_u8(c0_0, vld1q_u8(tc2.add(c0_r2[l0] as usize * 16)));
                    c0_1 = veorq_u8(c0_1, vld1q_u8(tc2.add(c0_r2[l1] as usize * 16)));
                    c0_2 = veorq_u8(c0_2, vld1q_u8(tc2.add(c0_r2[l2] as usize * 16)));
                    c0_3 = veorq_u8(c0_3, vld1q_u8(tc2.add(c0_r2[l3] as usize * 16)));

                    c0_0 = veorq_u8(c0_0, vld1q_u8(tc3.add(c0_r3[l0] as usize * 16)));
                    c0_1 = veorq_u8(c0_1, vld1q_u8(tc3.add(c0_r3[l1] as usize * 16)));
                    c0_2 = veorq_u8(c0_2, vld1q_u8(tc3.add(c0_r3[l2] as usize * 16)));
                    c0_3 = veorq_u8(c0_3, vld1q_u8(tc3.add(c0_r3[l3] as usize * 16)));

                    let mut c1_0 = vld1q_u8(tc0.add(c1_r0[l0] as usize * 16));
                    let mut c1_1 = vld1q_u8(tc0.add(c1_r0[l1] as usize * 16));
                    let mut c1_2 = vld1q_u8(tc0.add(c1_r0[l2] as usize * 16));
                    let mut c1_3 = vld1q_u8(tc0.add(c1_r0[l3] as usize * 16));

                    c1_0 = veorq_u8(c1_0, vld1q_u8(tc1.add(c1_r1[l0] as usize * 16)));
                    c1_1 = veorq_u8(c1_1, vld1q_u8(tc1.add(c1_r1[l1] as usize * 16)));
                    c1_2 = veorq_u8(c1_2, vld1q_u8(tc1.add(c1_r1[l2] as usize * 16)));
                    c1_3 = veorq_u8(c1_3, vld1q_u8(tc1.add(c1_r1[l3] as usize * 16)));

                    c1_0 = veorq_u8(c1_0, vld1q_u8(tc2.add(c1_r2[l0] as usize * 16)));
                    c1_1 = veorq_u8(c1_1, vld1q_u8(tc2.add(c1_r2[l1] as usize * 16)));
                    c1_2 = veorq_u8(c1_2, vld1q_u8(tc2.add(c1_r2[l2] as usize * 16)));
                    c1_3 = veorq_u8(c1_3, vld1q_u8(tc2.add(c1_r2[l3] as usize * 16)));

                    c1_0 = veorq_u8(c1_0, vld1q_u8(tc3.add(c1_r3[l0] as usize * 16)));
                    c1_1 = veorq_u8(c1_1, vld1q_u8(tc3.add(c1_r3[l1] as usize * 16)));
                    c1_2 = veorq_u8(c1_2, vld1q_u8(tc3.add(c1_r3[l2] as usize * 16)));
                    c1_3 = veorq_u8(c1_3, vld1q_u8(tc3.add(c1_r3[l3] as usize * 16)));

                    if h == 0 {
                        dst_c0[l0] = c0_0;
                        dst_c0[l1] = c0_1;
                        dst_c0[l2] = c0_2;
                        dst_c0[l3] = c0_3;
                        dst_c1[l0] = c1_0;
                        dst_c1[l1] = c1_1;
                        dst_c1[l2] = c1_2;
                        dst_c1[l3] = c1_3;
                    } else {
                        dst_c0[l0] = veorq_u8(dst_c0[l0], c0_0);
                        dst_c0[l1] = veorq_u8(dst_c0[l1], c0_1);
                        dst_c0[l2] = veorq_u8(dst_c0[l2], c0_2);
                        dst_c0[l3] = veorq_u8(dst_c0[l3], c0_3);
                        dst_c1[l0] = veorq_u8(dst_c1[l0], c1_0);
                        dst_c1[l1] = veorq_u8(dst_c1[l1], c1_1);
                        dst_c1[l2] = veorq_u8(dst_c1[l2], c1_2);
                        dst_c1[l3] = veorq_u8(dst_c1[l3], c1_3);
                    }
                }
            }
        }

        #[cfg(target_feature = "aes")]
        {
            for b_idx in 0..batch_len {
                let eq_lo_val = batch_eq_lo[b_idx];
                let veq = vld1q_u64((&eq_lo_val as *const F128).cast::<u64>());
                let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(veq);
                let src_ab = &conv_ab[b_idx];
                let src_c0 = &conv_c0[b_idx];
                let src_c1 = &conv_c1[b_idx];
                for pair in 0..32 {
                    let l0 = pair * 2;
                    let l1 = l0 + 1;
                    let (p_ab0, p_ab1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        vreinterpretq_u64_u8(src_ab[l0]),
                        vreinterpretq_u64_u8(src_ab[l1]),
                        r0,
                        r1,
                        r0,
                        r1,
                    );
                    let ptr_ab0 = (&mut partial_ab[l0].ll as *mut F128).cast::<u64>();
                    let ptr_ab1 = (&mut partial_ab[l1].ll as *mut F128).cast::<u64>();
                    vst1q_u64(ptr_ab0, veorq_u64(vld1q_u64(ptr_ab0), p_ab0));
                    vst1q_u64(ptr_ab1, veorq_u64(vld1q_u64(ptr_ab1), p_ab1));

                    let (p_c0_0, p_c1_0) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        vreinterpretq_u64_u8(src_c0[l0]),
                        vreinterpretq_u64_u8(src_c1[l0]),
                        r0,
                        r1,
                        r0,
                        r1,
                    );
                    let ptr_c0_0 = (&mut partial_c_0[l0].ll as *mut F128).cast::<u64>();
                    let ptr_c1_0 = (&mut partial_c_1[l0].ll as *mut F128).cast::<u64>();
                    vst1q_u64(ptr_c0_0, veorq_u64(vld1q_u64(ptr_c0_0), p_c0_0));
                    vst1q_u64(ptr_c1_0, veorq_u64(vld1q_u64(ptr_c1_0), p_c1_0));

                    let (p_c0_1, p_c1_1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        vreinterpretq_u64_u8(src_c0[l1]),
                        vreinterpretq_u64_u8(src_c1[l1]),
                        r0,
                        r1,
                        r0,
                        r1,
                    );
                    let ptr_c0_1 = (&mut partial_c_0[l1].ll as *mut F128).cast::<u64>();
                    let ptr_c1_1 = (&mut partial_c_1[l1].ll as *mut F128).cast::<u64>();
                    vst1q_u64(ptr_c0_1, veorq_u64(vld1q_u64(ptr_c0_1), p_c0_1));
                    vst1q_u64(ptr_c1_1, veorq_u64(vld1q_u64(ptr_c1_1), p_c1_1));
                }
            }
            return;
        }
        let zero_u64 = vdupq_n_u64(0);
        let mut veqs = [zero_u64; super::super::URM_BATCH];
        let mut eq_los = [0u64; super::super::URM_BATCH];
        let mut eq_sums = [0u64; super::super::URM_BATCH];
        for b_idx in 0..batch_len {
            let eq_lo_val = batch_eq_lo[b_idx];
            let veq = vld1q_u64((&eq_lo_val as *const F128).cast::<u64>());
            let eq_lo = vgetq_lane_u64::<0>(veq);
            veqs[b_idx] = veq;
            eq_los[b_idx] = eq_lo;
            eq_sums[b_idx] = eq_lo ^ vgetq_lane_u64::<1>(veq);
        }

        for l in 0..64 {
            let mut ab_ll = zero_u64;
            let mut ab_mid = zero_u64;
            let mut ab_hh = zero_u64;
            let mut c0_ll = zero_u64;
            let mut c0_mid = zero_u64;
            let mut c0_hh = zero_u64;
            let mut c1_ll = zero_u64;
            let mut c1_mid = zero_u64;
            let mut c1_hh = zero_u64;
            for b_idx in 0..batch_len {
                let veq = veqs[b_idx];
                let eq_lo = eq_los[b_idx];
                let eq_sum = eq_sums[b_idx];
                let vab = vreinterpretq_u64_u8(conv_ab[b_idx][l]);
                let vc0 = vreinterpretq_u64_u8(conv_c0[b_idx][l]);
                let vc1 = vreinterpretq_u64_u8(conv_c1[b_idx][l]);
                let ab_lo = vgetq_lane_u64::<0>(vab);
                let ab_sum = ab_lo ^ vgetq_lane_u64::<1>(vab);
                let c0_lo = vgetq_lane_u64::<0>(vc0);
                let c0_sum = c0_lo ^ vgetq_lane_u64::<1>(vc0);
                let c1_lo = vgetq_lane_u64::<0>(vc1);
                let c1_sum = c1_lo ^ vgetq_lane_u64::<1>(vc1);
                ab_ll = veorq_u64(ab_ll, vreinterpretq_u64_p128(vmull_p64(ab_lo, eq_lo)));
                ab_mid = veorq_u64(ab_mid, vreinterpretq_u64_p128(vmull_p64(ab_sum, eq_sum)));
                ab_hh = veorq_u64(
                    ab_hh,
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64(vab),
                        vreinterpretq_p64_u64(veq),
                    )),
                );
                c0_ll = veorq_u64(c0_ll, vreinterpretq_u64_p128(vmull_p64(c0_lo, eq_lo)));
                c0_mid = veorq_u64(c0_mid, vreinterpretq_u64_p128(vmull_p64(c0_sum, eq_sum)));
                c0_hh = veorq_u64(
                    c0_hh,
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64(vc0),
                        vreinterpretq_p64_u64(veq),
                    )),
                );
                c1_ll = veorq_u64(c1_ll, vreinterpretq_u64_p128(vmull_p64(c1_lo, eq_lo)));
                c1_mid = veorq_u64(c1_mid, vreinterpretq_u64_p128(vmull_p64(c1_sum, eq_sum)));
                c1_hh = veorq_u64(
                    c1_hh,
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64(vc1),
                        vreinterpretq_p64_u64(veq),
                    )),
                );
            }
            let ptr_ab = (&mut partial_ab[l] as *mut super::super::KaratsubaAcc).cast::<u64>();
            vst1q_u64(ptr_ab, veorq_u64(vld1q_u64(ptr_ab), ab_ll));
            vst1q_u64(ptr_ab.add(2), veorq_u64(vld1q_u64(ptr_ab.add(2)), ab_mid));
            vst1q_u64(ptr_ab.add(4), veorq_u64(vld1q_u64(ptr_ab.add(4)), ab_hh));
            let ptr_c0 = (&mut partial_c_0[l] as *mut super::super::KaratsubaAcc).cast::<u64>();
            vst1q_u64(ptr_c0, veorq_u64(vld1q_u64(ptr_c0), c0_ll));
            vst1q_u64(ptr_c0.add(2), veorq_u64(vld1q_u64(ptr_c0.add(2)), c0_mid));
            vst1q_u64(ptr_c0.add(4), veorq_u64(vld1q_u64(ptr_c0.add(4)), c0_hh));
            let ptr_c1 = (&mut partial_c_1[l] as *mut super::super::KaratsubaAcc).cast::<u64>();
            vst1q_u64(ptr_c1, veorq_u64(vld1q_u64(ptr_c1), c1_ll));
            vst1q_u64(ptr_c1.add(2), veorq_u64(vld1q_u64(ptr_c1.add(2)), c1_mid));
            vst1q_u64(ptr_c1.add(4), veorq_u64(vld1q_u64(ptr_c1.add(4)), c1_hh));
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_batch_ab_only(
    batch_ab_bytes: &[[[u8; 64]; 16]; super::super::URM_BATCH],
    batch_n_b_med: &[u8; super::super::URM_BATCH],
    batch_eq_lo: &[F128; super::super::URM_BATCH],
    batch_len: usize,
    convert: &[F128],
    partial_ab: &mut [super::super::KaratsubaAcc; 64],
) {
    use core::arch::aarch64::*;

    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let zero = vdupq_n_u8(0);
        let mut conv_ab = [[zero; 64]; super::super::URM_BATCH];

        for q in 0..4 {
            let b_base = q * 4;
            let t0 = convert_ptr.add(b_base * 256 * 16);
            let t1 = convert_ptr.add((b_base + 1) * 256 * 16);
            let t2 = convert_ptr.add((b_base + 2) * 256 * 16);
            let t3 = convert_ptr.add((b_base + 3) * 256 * 16);
            for b_idx in 0..batch_len {
                let n_b_med = batch_n_b_med[b_idx] as usize;
                let dst_ab = &mut conv_ab[b_idx];
                if b_base >= n_b_med {
                    if q == 0 {
                        dst_ab.fill(zero);
                    }
                    continue;
                }
                let ab_rows = &batch_ab_bytes[b_idx];
                let rem = n_b_med - b_base;
                if rem >= 4 {
                    let ab_r0 = &ab_rows[b_base];
                    let ab_r1 = &ab_rows[b_base + 1];
                    let ab_r2 = &ab_rows[b_base + 2];
                    let ab_r3 = &ab_rows[b_base + 3];
                    for tile in 0..16 {
                        let l0 = tile * 4;
                        let l1 = l0 + 1;
                        let l2 = l0 + 2;
                        let l3 = l0 + 3;
                        let mut ab0 = vld1q_u8(t0.add(ab_r0[l0] as usize * 16));
                        let mut ab1 = vld1q_u8(t0.add(ab_r0[l1] as usize * 16));
                        let mut ab2 = vld1q_u8(t0.add(ab_r0[l2] as usize * 16));
                        let mut ab3 = vld1q_u8(t0.add(ab_r0[l3] as usize * 16));

                        ab0 = veorq_u8(ab0, vld1q_u8(t1.add(ab_r1[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t1.add(ab_r1[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t1.add(ab_r1[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t1.add(ab_r1[l3] as usize * 16)));

                        ab0 = veorq_u8(ab0, vld1q_u8(t2.add(ab_r2[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t2.add(ab_r2[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t2.add(ab_r2[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t2.add(ab_r2[l3] as usize * 16)));

                        ab0 = veorq_u8(ab0, vld1q_u8(t3.add(ab_r3[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t3.add(ab_r3[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t3.add(ab_r3[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t3.add(ab_r3[l3] as usize * 16)));

                        if q == 0 {
                            dst_ab[l0] = ab0;
                            dst_ab[l1] = ab1;
                            dst_ab[l2] = ab2;
                            dst_ab[l3] = ab3;
                        } else {
                            dst_ab[l0] = veorq_u8(dst_ab[l0], ab0);
                            dst_ab[l1] = veorq_u8(dst_ab[l1], ab1);
                            dst_ab[l2] = veorq_u8(dst_ab[l2], ab2);
                            dst_ab[l3] = veorq_u8(dst_ab[l3], ab3);
                        }
                    }
                } else if rem == 3 {
                    let ab_r0 = &ab_rows[b_base];
                    let ab_r1 = &ab_rows[b_base + 1];
                    let ab_r2 = &ab_rows[b_base + 2];
                    for tile in 0..16 {
                        let l0 = tile * 4;
                        let l1 = l0 + 1;
                        let l2 = l0 + 2;
                        let l3 = l0 + 3;
                        let mut ab0 = vld1q_u8(t0.add(ab_r0[l0] as usize * 16));
                        let mut ab1 = vld1q_u8(t0.add(ab_r0[l1] as usize * 16));
                        let mut ab2 = vld1q_u8(t0.add(ab_r0[l2] as usize * 16));
                        let mut ab3 = vld1q_u8(t0.add(ab_r0[l3] as usize * 16));

                        ab0 = veorq_u8(ab0, vld1q_u8(t1.add(ab_r1[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t1.add(ab_r1[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t1.add(ab_r1[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t1.add(ab_r1[l3] as usize * 16)));

                        ab0 = veorq_u8(ab0, vld1q_u8(t2.add(ab_r2[l0] as usize * 16)));
                        ab1 = veorq_u8(ab1, vld1q_u8(t2.add(ab_r2[l1] as usize * 16)));
                        ab2 = veorq_u8(ab2, vld1q_u8(t2.add(ab_r2[l2] as usize * 16)));
                        ab3 = veorq_u8(ab3, vld1q_u8(t2.add(ab_r2[l3] as usize * 16)));

                        if q == 0 {
                            dst_ab[l0] = ab0;
                            dst_ab[l1] = ab1;
                            dst_ab[l2] = ab2;
                            dst_ab[l3] = ab3;
                        } else {
                            dst_ab[l0] = veorq_u8(dst_ab[l0], ab0);
                            dst_ab[l1] = veorq_u8(dst_ab[l1], ab1);
                            dst_ab[l2] = veorq_u8(dst_ab[l2], ab2);
                            dst_ab[l3] = veorq_u8(dst_ab[l3], ab3);
                        }
                    }
                } else {
                    for l in 0..64 {
                        let mut ab = vld1q_u8(t0.add(ab_rows[b_base][l] as usize * 16));
                        for r in 1..rem {
                            ab = veorq_u8(
                                ab,
                                vld1q_u8(convert_ptr.add(
                                    ((b_base + r) * 256 + ab_rows[b_base + r][l] as usize) * 16,
                                )),
                            );
                        }
                        if q == 0 {
                            dst_ab[l] = ab;
                        } else {
                            dst_ab[l] = veorq_u8(dst_ab[l], ab);
                        }
                    }
                }
            }
        }

        let zero_u64 = vdupq_n_u64(0);
        let mut veqs = [zero_u64; super::super::URM_BATCH];
        let mut eq_los = [0u64; super::super::URM_BATCH];
        let mut eq_sums = [0u64; super::super::URM_BATCH];
        for b_idx in 0..batch_len {
            let eq_lo_val = batch_eq_lo[b_idx];
            let veq = vld1q_u64((&eq_lo_val as *const F128).cast::<u64>());
            let eq_lo = vgetq_lane_u64::<0>(veq);
            veqs[b_idx] = veq;
            eq_los[b_idx] = eq_lo;
            eq_sums[b_idx] = eq_lo ^ vgetq_lane_u64::<1>(veq);
        }

        for l in 0..64 {
            let mut ab_ll = zero_u64;
            let mut ab_mid = zero_u64;
            let mut ab_hh = zero_u64;
            for b_idx in 0..batch_len {
                let veq = veqs[b_idx];
                let eq_lo = eq_los[b_idx];
                let eq_sum = eq_sums[b_idx];
                let vab = vreinterpretq_u64_u8(conv_ab[b_idx][l]);
                let ab_lo = vgetq_lane_u64::<0>(vab);
                let ab_sum = ab_lo ^ vgetq_lane_u64::<1>(vab);
                ab_ll = veorq_u64(ab_ll, vreinterpretq_u64_p128(vmull_p64(ab_lo, eq_lo)));
                ab_mid = veorq_u64(ab_mid, vreinterpretq_u64_p128(vmull_p64(ab_sum, eq_sum)));
                ab_hh = veorq_u64(
                    ab_hh,
                    vreinterpretq_u64_p128(vmull_high_p64(
                        vreinterpretq_p64_u64(vab),
                        vreinterpretq_p64_u64(veq),
                    )),
                );
            }
            let ptr_ab = (&mut partial_ab[l] as *mut super::super::KaratsubaAcc).cast::<u64>();
            vst1q_u64(ptr_ab, veorq_u64(vld1q_u64(ptr_ab), ab_ll));
            vst1q_u64(ptr_ab.add(2), veorq_u64(vld1q_u64(ptr_ab.add(2)), ab_mid));
            vst1q_u64(ptr_ab.add(4), veorq_u64(vld1q_u64(ptr_ab.add(4)), ab_hh));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn bit_transpose_64bytes_neon(input: &[u8; 64], output: &mut [u8; 64]) {
    unsafe {
        crate::gf2_kernels::k0_f128_bit_transpose_64bytes(input.as_ptr(), output.as_mut_ptr());
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub(crate) fn shift_reduce_inner_ab_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    use crate::field::gf2_8::neon::{gf8_mul_vec16, gf8_reduce_vec16};
    use core::arch::aarch64::*;

    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        macro_rules! step_k {
            ($k:literal) => {{
                let chunk_off = byte_base_b + $k * N_CHUNKS;
                inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
                inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
                let a_ptr = a_col.as_ptr() as *const u8;
                let b_ptr = b_col.as_ptr() as *const u8;
                let y0 = gf8_mul_vec16(vld1q_u8(a_ptr), vld1q_u8(b_ptr));
                let y1 = gf8_mul_vec16(vld1q_u8(a_ptr.add(16)), vld1q_u8(b_ptr.add(16)));
                let y2 = gf8_mul_vec16(vld1q_u8(a_ptr.add(32)), vld1q_u8(b_ptr.add(32)));
                let y3 = gf8_mul_vec16(vld1q_u8(a_ptr.add(48)), vld1q_u8(b_ptr.add(48)));
                acc0_lo = veorq_u16(acc0_lo, vshll_n_u8::<$k>(vget_low_u8(y0)));
                acc0_hi = veorq_u16(acc0_hi, vshll_n_u8::<$k>(vget_high_u8(y0)));
                acc1_lo = veorq_u16(acc1_lo, vshll_n_u8::<$k>(vget_low_u8(y1)));
                acc1_hi = veorq_u16(acc1_hi, vshll_n_u8::<$k>(vget_high_u8(y1)));
                acc2_lo = veorq_u16(acc2_lo, vshll_n_u8::<$k>(vget_low_u8(y2)));
                acc2_hi = veorq_u16(acc2_hi, vshll_n_u8::<$k>(vget_high_u8(y2)));
                acc3_lo = veorq_u16(acc3_lo, vshll_n_u8::<$k>(vget_low_u8(y3)));
                acc3_hi = veorq_u16(acc3_hi, vshll_n_u8::<$k>(vget_high_u8(y3)));
            }};
        }

        step_k!(0);
        step_k!(1);
        step_k!(2);
        step_k!(3);
        step_k!(4);
        step_k!(5);
        step_k!(6);
        step_k!(7);

        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));

        let out_ptr = out.as_mut_ptr();
        vst1q_u8(out_ptr, r0);
        vst1q_u8(out_ptr.add(16), r1);
        vst1q_u8(out_ptr.add(32), r2);
        vst1q_u8(out_ptr.add(48), r3);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_8_regs<const BH: usize>(
    table_base: *const u8,
    a_byte: u8,
    b_byte: u8,
    da0: &mut core::arch::aarch64::uint8x16_t,
    da1: &mut core::arch::aarch64::uint8x16_t,
    da2: &mut core::arch::aarch64::uint8x16_t,
    da3: &mut core::arch::aarch64::uint8x16_t,
    db0: &mut core::arch::aarch64::uint8x16_t,
    db1: &mut core::arch::aarch64::uint8x16_t,
    db2: &mut core::arch::aarch64::uint8x16_t,
    db3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let ra = table_base.add(a_byte as usize * 64);
        let rb = table_base.add(b_byte as usize * 64);
        *da0 = veorq_u8(*da0, vld1q_u8(ra.add((0 ^ BH) * 16)));
        *da1 = veorq_u8(*da1, vld1q_u8(ra.add((1 ^ BH) * 16)));
        *da2 = veorq_u8(*da2, vld1q_u8(ra.add((2 ^ BH) * 16)));
        *da3 = veorq_u8(*da3, vld1q_u8(ra.add((3 ^ BH) * 16)));
        *db0 = veorq_u8(*db0, vld1q_u8(rb.add((0 ^ BH) * 16)));
        *db1 = veorq_u8(*db1, vld1q_u8(rb.add((1 ^ BH) * 16)));
        *db2 = veorq_u8(*db2, vld1q_u8(rb.add((2 ^ BH) * 16)));
        *db3 = veorq_u8(*db3, vld1q_u8(rb.add((3 ^ BH) * 16)));
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fused_apply_one_k<const FIRST: bool>(
    table_base: *const u8,
    a_row: *const u8,
    b_row: *const u8,
    acc0_lo: &mut core::arch::aarch64::uint16x8_t,
    acc0_hi: &mut core::arch::aarch64::uint16x8_t,
    acc1_lo: &mut core::arch::aarch64::uint16x8_t,
    acc1_hi: &mut core::arch::aarch64::uint16x8_t,
    acc2_lo: &mut core::arch::aarch64::uint16x8_t,
    acc2_hi: &mut core::arch::aarch64::uint16x8_t,
    acc3_lo: &mut core::arch::aarch64::uint16x8_t,
    acc3_hi: &mut core::arch::aarch64::uint16x8_t,
) {
    use core::arch::aarch64::*;
    use core::mem::transmute;
    unsafe {
        let ra0 = table_base.add(*a_row as usize * 64);
        let rb0 = table_base.add(*b_row as usize * 64);
        let mut da0 = vld1q_u8(ra0);
        let mut da1 = vld1q_u8(ra0.add(16));
        let mut da2 = vld1q_u8(ra0.add(32));
        let mut da3 = vld1q_u8(ra0.add(48));
        let mut db0 = vld1q_u8(rb0);
        let mut db1 = vld1q_u8(rb0.add(16));
        let mut db2 = vld1q_u8(rb0.add(32));
        let mut db3 = vld1q_u8(rb0.add(48));

        xor_apply_byte_into_8_regs::<1>(
            table_base,
            *a_row.add(2),
            *b_row.add(2),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<2>(
            table_base,
            *a_row.add(4),
            *b_row.add(4),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<3>(
            table_base,
            *a_row.add(6),
            *b_row.add(6),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );

        let ra1 = table_base.add(*a_row.add(1) as usize * 64);
        let rb1 = table_base.add(*b_row.add(1) as usize * 64);
        let mut oa0 = vld1q_u8(ra1);
        let mut oa1 = vld1q_u8(ra1.add(16));
        let mut oa2 = vld1q_u8(ra1.add(32));
        let mut oa3 = vld1q_u8(ra1.add(48));
        let mut ob0 = vld1q_u8(rb1);
        let mut ob1 = vld1q_u8(rb1.add(16));
        let mut ob2 = vld1q_u8(rb1.add(32));
        let mut ob3 = vld1q_u8(rb1.add(48));

        xor_apply_byte_into_8_regs::<1>(
            table_base,
            *a_row.add(3),
            *b_row.add(3),
            &mut oa0,
            &mut oa1,
            &mut oa2,
            &mut oa3,
            &mut ob0,
            &mut ob1,
            &mut ob2,
            &mut ob3,
        );
        xor_apply_byte_into_8_regs::<2>(
            table_base,
            *a_row.add(5),
            *b_row.add(5),
            &mut oa0,
            &mut oa1,
            &mut oa2,
            &mut oa3,
            &mut ob0,
            &mut ob1,
            &mut ob2,
            &mut ob3,
        );
        xor_apply_byte_into_8_regs::<3>(
            table_base,
            *a_row.add(7),
            *b_row.add(7),
            &mut oa0,
            &mut oa1,
            &mut oa2,
            &mut oa3,
            &mut ob0,
            &mut ob1,
            &mut ob2,
            &mut ob3,
        );

        da0 = veorq_u8(da0, vextq_u8::<8>(oa0, oa0));
        da1 = veorq_u8(da1, vextq_u8::<8>(oa1, oa1));
        da2 = veorq_u8(da2, vextq_u8::<8>(oa2, oa2));
        da3 = veorq_u8(da3, vextq_u8::<8>(oa3, oa3));
        db0 = veorq_u8(db0, vextq_u8::<8>(ob0, ob0));
        db1 = veorq_u8(db1, vextq_u8::<8>(ob1, ob1));
        db2 = veorq_u8(db2, vextq_u8::<8>(ob2, ob2));
        db3 = veorq_u8(db3, vextq_u8::<8>(ob3, ob3));

        macro_rules! clmul_lo_hi {
            ($a:expr, $b:expr) => {{
                let lo = vreinterpretq_u16_p16(vmull_p8(
                    transmute::<uint8x8_t, poly8x8_t>(vget_low_u8($a)),
                    transmute::<uint8x8_t, poly8x8_t>(vget_low_u8($b)),
                ));
                let hi = vreinterpretq_u16_p16(vmull_p8(
                    transmute::<uint8x8_t, poly8x8_t>(vget_high_u8($a)),
                    transmute::<uint8x8_t, poly8x8_t>(vget_high_u8($b)),
                ));
                (lo, hi)
            }};
        }

        let (p0_lo, p0_hi) = clmul_lo_hi!(da0, db0);
        let (p1_lo, p1_hi) = clmul_lo_hi!(da1, db1);
        let (p2_lo, p2_hi) = clmul_lo_hi!(da2, db2);
        let (p3_lo, p3_hi) = clmul_lo_hi!(da3, db3);

        if FIRST {
            *acc0_lo = p0_lo;
            *acc0_hi = p0_hi;
            *acc1_lo = p1_lo;
            *acc1_hi = p1_hi;
            *acc2_lo = p2_lo;
            *acc2_hi = p2_hi;
            *acc3_lo = p3_lo;
            *acc3_hi = p3_hi;
        } else {
            let v8d80 = vdupq_n_u16(0x8d80);
            macro_rules! horner_step {
                ($acc:expr, $p:expr) => {{
                    let s = vshlq_n_u16::<1>(*$acc);
                    let mask = vreinterpretq_u16_s16(vshrq_n_s16::<15>(vreinterpretq_s16_u16(s)));
                    *$acc = veorq_u16(veorq_u16(s, vandq_u16(mask, v8d80)), $p);
                }};
            }
            horner_step!(acc0_lo, p0_lo);
            horner_step!(acc0_hi, p0_hi);
            horner_step!(acc1_lo, p1_lo);
            horner_step!(acc1_hi, p1_hi);
            horner_step!(acc2_lo, p2_lo);
            horner_step!(acc2_hi, p2_hi);
            horner_step!(acc3_lo, p3_lo);
            horner_step!(acc3_hi, p3_hi);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn shift_reduce_inner_ab_fused_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;

    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let table_base = inv_table.data_ptr();

    unsafe {
        let b_base_ptr = b_packed.as_ptr().add(byte_base_b) as *const u64;
        let b0 = b_base_ptr.read_unaligned();
        let b7 = b_base_ptr.add(7).read_unaligned();
        if b0 == !0u64
            && (b_base_ptr.add(1).read_unaligned()
                & b_base_ptr.add(2).read_unaligned()
                & b_base_ptr.add(3).read_unaligned()
                & b_base_ptr.add(4).read_unaligned()
                & b_base_ptr.add(5).read_unaligned())
                == !0u64
        {
            let v1b = vdupq_n_u8(0x1b);
            let mut acc0 = vdupq_n_u8(0);
            let mut acc1 = vdupq_n_u8(0);
            let mut acc2 = vdupq_n_u8(0);
            let mut acc3 = vdupq_n_u8(0);
            macro_rules! gf8_mul_x_16 {
                ($x:expr) => {{
                    let shifted = vshlq_n_u8::<1>($x);
                    let mask = vreinterpretq_u8_s8(vshrq_n_s8::<7>(vreinterpretq_s8_u8($x)));
                    veorq_u8(shifted, vandq_u8(mask, v1b))
                }};
            }
            macro_rules! apply_a_only_step {
                ($k:expr, $is_first:expr) => {{
                    let a_row = a_packed.as_ptr().add(byte_base_b + $k * N_CHUNKS);
                    let r0 = table_base.add(*a_row as usize * 64);
                    let r2 = table_base.add(*a_row.add(2) as usize * 64);
                    let r4 = table_base.add(*a_row.add(4) as usize * 64);
                    let r6 = table_base.add(*a_row.add(6) as usize * 64);
                    let mut d0 = veorq_u8(
                        veorq_u8(vld1q_u8(r0), vld1q_u8(r2.add(16))),
                        veorq_u8(vld1q_u8(r4.add(32)), vld1q_u8(r6.add(48))),
                    );
                    let mut d1 = veorq_u8(
                        veorq_u8(vld1q_u8(r0.add(16)), vld1q_u8(r2)),
                        veorq_u8(vld1q_u8(r4.add(48)), vld1q_u8(r6.add(32))),
                    );
                    let mut d2 = veorq_u8(
                        veorq_u8(vld1q_u8(r0.add(32)), vld1q_u8(r2.add(48))),
                        veorq_u8(vld1q_u8(r4), vld1q_u8(r6.add(16))),
                    );
                    let mut d3 = veorq_u8(
                        veorq_u8(vld1q_u8(r0.add(48)), vld1q_u8(r2.add(32))),
                        veorq_u8(vld1q_u8(r4.add(16)), vld1q_u8(r6)),
                    );

                    let r1 = table_base.add(*a_row.add(1) as usize * 64);
                    let r3 = table_base.add(*a_row.add(3) as usize * 64);
                    let r5 = table_base.add(*a_row.add(5) as usize * 64);
                    let r7 = table_base.add(*a_row.add(7) as usize * 64);
                    let o0 = veorq_u8(
                        veorq_u8(vld1q_u8(r1), vld1q_u8(r3.add(16))),
                        veorq_u8(vld1q_u8(r5.add(32)), vld1q_u8(r7.add(48))),
                    );
                    let o1 = veorq_u8(
                        veorq_u8(vld1q_u8(r1.add(16)), vld1q_u8(r3)),
                        veorq_u8(vld1q_u8(r5.add(48)), vld1q_u8(r7.add(32))),
                    );
                    let o2 = veorq_u8(
                        veorq_u8(vld1q_u8(r1.add(32)), vld1q_u8(r3.add(48))),
                        veorq_u8(vld1q_u8(r5), vld1q_u8(r7.add(16))),
                    );
                    let o3 = veorq_u8(
                        veorq_u8(vld1q_u8(r1.add(48)), vld1q_u8(r3.add(32))),
                        veorq_u8(vld1q_u8(r5.add(16)), vld1q_u8(r7)),
                    );

                    d0 = veorq_u8(d0, vextq_u8::<8>(o0, o0));
                    d1 = veorq_u8(d1, vextq_u8::<8>(o1, o1));
                    d2 = veorq_u8(d2, vextq_u8::<8>(o2, o2));
                    d3 = veorq_u8(d3, vextq_u8::<8>(o3, o3));

                    if $is_first {
                        acc0 = d0;
                        acc1 = d1;
                        acc2 = d2;
                        acc3 = d3;
                    } else {
                        acc0 = veorq_u8(gf8_mul_x_16!(acc0), d0);
                        acc1 = veorq_u8(gf8_mul_x_16!(acc1), d1);
                        acc2 = veorq_u8(gf8_mul_x_16!(acc2), d2);
                        acc3 = veorq_u8(gf8_mul_x_16!(acc3), d3);
                    }
                }};
            }
            if (b_base_ptr.add(6).read_unaligned() & b7) == !0u64 {
                for k in (0..8).rev() {
                    apply_a_only_step!(k, k == 7);
                }
            } else {
                let mut acc0_lo = vdupq_n_u16(0);
                let mut acc0_hi = vdupq_n_u16(0);
                let mut acc1_lo = vdupq_n_u16(0);
                let mut acc1_hi = vdupq_n_u16(0);
                let mut acc2_lo = vdupq_n_u16(0);
                let mut acc2_hi = vdupq_n_u16(0);
                let mut acc3_lo = vdupq_n_u16(0);
                let mut acc3_hi = vdupq_n_u16(0);
                let off7 = byte_base_b + 7 * N_CHUNKS;
                fused_apply_one_k::<true>(
                    table_base,
                    a_packed.as_ptr().add(off7),
                    b_packed.as_ptr().add(off7),
                    &mut acc0_lo,
                    &mut acc0_hi,
                    &mut acc1_lo,
                    &mut acc1_hi,
                    &mut acc2_lo,
                    &mut acc2_hi,
                    &mut acc3_lo,
                    &mut acc3_hi,
                );
                let off6 = byte_base_b + 6 * N_CHUNKS;
                fused_apply_one_k::<false>(
                    table_base,
                    a_packed.as_ptr().add(off6),
                    b_packed.as_ptr().add(off6),
                    &mut acc0_lo,
                    &mut acc0_hi,
                    &mut acc1_lo,
                    &mut acc1_hi,
                    &mut acc2_lo,
                    &mut acc2_hi,
                    &mut acc3_lo,
                    &mut acc3_hi,
                );
                acc0 =
                    gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
                acc1 =
                    gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
                acc2 =
                    gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
                acc3 =
                    gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));
                for k in (0..6).rev() {
                    apply_a_only_step!(k, false);
                }
            }
            let p = out.as_mut_ptr();
            vst1q_u8(p, acc0);
            vst1q_u8(p.add(16), acc1);
            vst1q_u8(p.add(32), acc2);
            vst1q_u8(p.add(48), acc3);
            return;
        }

        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        macro_rules! do_k {
            ($k:literal, $first:literal) => {{
                let off = byte_base_b + $k * N_CHUNKS;
                fused_apply_one_k::<$first>(
                    table_base,
                    a_packed.as_ptr().add(off),
                    b_packed.as_ptr().add(off),
                    &mut acc0_lo,
                    &mut acc0_hi,
                    &mut acc1_lo,
                    &mut acc1_hi,
                    &mut acc2_lo,
                    &mut acc2_hi,
                    &mut acc3_lo,
                    &mut acc3_hi,
                );
            }};
        }
        if b_base_ptr.add(7).read_unaligned() == 0
            && (b_base_ptr.add(1).read_unaligned()
                | b_base_ptr.add(2).read_unaligned()
                | b_base_ptr.add(3).read_unaligned()
                | b_base_ptr.add(4).read_unaligned()
                | b_base_ptr.add(5).read_unaligned()
                | b_base_ptr.add(6).read_unaligned())
                == 0
        {
            do_k!(0, true);
        } else {
            do_k!(7, true);
            do_k!(6, false);
            do_k!(5, false);
            do_k!(4, false);
            do_k!(3, false);
            do_k!(2, false);
            do_k!(1, false);
            do_k!(0, false);
        }

        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));

        let p = out.as_mut_ptr();
        vst1q_u8(p, r0);
        vst1q_u8(p.add(16), r1);
        vst1q_u8(p.add(32), r2);
        vst1q_u8(p.add(48), r3);
    }
}

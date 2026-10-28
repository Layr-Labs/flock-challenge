use super::super::{F8, F128, InvNttTableByteSingleGf8, N_CHUNKS};

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn xor3_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::veor3q_u8(a, b, c) }
}

#[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
#[inline(always)]
unsafe fn xor3_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::veorq_u8(a, core::arch::aarch64::veorq_u8(b, c)) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn xor3_u16(
    a: core::arch::aarch64::uint16x8_t,
    b: core::arch::aarch64::uint16x8_t,
    c: core::arch::aarch64::uint16x8_t,
) -> core::arch::aarch64::uint16x8_t {
    unsafe { core::arch::aarch64::veor3q_u16(a, b, c) }
}

#[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
#[inline(always)]
unsafe fn xor3_u16(
    a: core::arch::aarch64::uint16x8_t,
    b: core::arch::aarch64::uint16x8_t,
    c: core::arch::aarch64::uint16x8_t,
) -> core::arch::aarch64::uint16x8_t {
    unsafe { core::arch::aarch64::veorq_u16(a, core::arch::aarch64::veorq_u16(b, c)) }
}

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
                    let ab_r1 = &ab_rows[b_base + 1];
                    let ab_r2 = &ab_rows[b_base + 2];
                    let ab_r3 = &ab_rows[b_base + 3];
                    let c_r0 = &c_rows[b_base];
                    let c_r1 = &c_rows[b_base + 1];
                    let c_r2 = &c_rows[b_base + 2];
                    let c_r3 = &c_rows[b_base + 3];

                    let ab0 = xor3_u8(
                        vld1q_u8(t0.add(ab_r0[l0] as usize * 16)),
                        vld1q_u8(t1.add(ab_r1[l0] as usize * 16)),
                        vld1q_u8(t2.add(ab_r2[l0] as usize * 16)),
                    );
                    let ab1 = xor3_u8(
                        vld1q_u8(t0.add(ab_r0[l1] as usize * 16)),
                        vld1q_u8(t1.add(ab_r1[l1] as usize * 16)),
                        vld1q_u8(t2.add(ab_r2[l1] as usize * 16)),
                    );
                    let ab2 = xor3_u8(
                        vld1q_u8(t0.add(ab_r0[l2] as usize * 16)),
                        vld1q_u8(t1.add(ab_r1[l2] as usize * 16)),
                        vld1q_u8(t2.add(ab_r2[l2] as usize * 16)),
                    );
                    let ab3 = xor3_u8(
                        vld1q_u8(t0.add(ab_r0[l3] as usize * 16)),
                        vld1q_u8(t1.add(ab_r1[l3] as usize * 16)),
                        vld1q_u8(t2.add(ab_r2[l3] as usize * 16)),
                    );
                    let c0 = xor3_u8(
                        vld1q_u8(t0.add(c_r0[l0] as usize * 16)),
                        vld1q_u8(t1.add(c_r1[l0] as usize * 16)),
                        vld1q_u8(t2.add(c_r2[l0] as usize * 16)),
                    );
                    let c1 = xor3_u8(
                        vld1q_u8(t0.add(c_r0[l1] as usize * 16)),
                        vld1q_u8(t1.add(c_r1[l1] as usize * 16)),
                        vld1q_u8(t2.add(c_r2[l1] as usize * 16)),
                    );
                    let c2 = xor3_u8(
                        vld1q_u8(t0.add(c_r0[l2] as usize * 16)),
                        vld1q_u8(t1.add(c_r1[l2] as usize * 16)),
                        vld1q_u8(t2.add(c_r2[l2] as usize * 16)),
                    );
                    let c3 = xor3_u8(
                        vld1q_u8(t0.add(c_r0[l3] as usize * 16)),
                        vld1q_u8(t1.add(c_r1[l3] as usize * 16)),
                        vld1q_u8(t2.add(c_r2[l3] as usize * 16)),
                    );

                    if q == 0 {
                        dst_ab[l0] = veorq_u8(ab0, vld1q_u8(t3.add(ab_r3[l0] as usize * 16)));
                        dst_ab[l1] = veorq_u8(ab1, vld1q_u8(t3.add(ab_r3[l1] as usize * 16)));
                        dst_ab[l2] = veorq_u8(ab2, vld1q_u8(t3.add(ab_r3[l2] as usize * 16)));
                        dst_ab[l3] = veorq_u8(ab3, vld1q_u8(t3.add(ab_r3[l3] as usize * 16)));
                        dst_c[l0] = veorq_u8(c0, vld1q_u8(t3.add(c_r3[l0] as usize * 16)));
                        dst_c[l1] = veorq_u8(c1, vld1q_u8(t3.add(c_r3[l1] as usize * 16)));
                        dst_c[l2] = veorq_u8(c2, vld1q_u8(t3.add(c_r3[l2] as usize * 16)));
                        dst_c[l3] = veorq_u8(c3, vld1q_u8(t3.add(c_r3[l3] as usize * 16)));
                    } else {
                        dst_ab[l0] =
                            xor3_u8(dst_ab[l0], ab0, vld1q_u8(t3.add(ab_r3[l0] as usize * 16)));
                        dst_ab[l1] =
                            xor3_u8(dst_ab[l1], ab1, vld1q_u8(t3.add(ab_r3[l1] as usize * 16)));
                        dst_ab[l2] =
                            xor3_u8(dst_ab[l2], ab2, vld1q_u8(t3.add(ab_r3[l2] as usize * 16)));
                        dst_ab[l3] =
                            xor3_u8(dst_ab[l3], ab3, vld1q_u8(t3.add(ab_r3[l3] as usize * 16)));
                        dst_c[l0] =
                            xor3_u8(dst_c[l0], c0, vld1q_u8(t3.add(c_r3[l0] as usize * 16)));
                        dst_c[l1] =
                            xor3_u8(dst_c[l1], c1, vld1q_u8(t3.add(c_r3[l1] as usize * 16)));
                        dst_c[l2] =
                            xor3_u8(dst_c[l2], c2, vld1q_u8(t3.add(c_r3[l2] as usize * 16)));
                        dst_c[l3] =
                            xor3_u8(dst_c[l3], c3, vld1q_u8(t3.add(c_r3[l3] as usize * 16)));
                    }
                }
            }
        }

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
            let mut b_idx = 0usize;
            while b_idx + 2 <= batch_len {
                let (p_ab0, p_c0) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_ab[b_idx][l]),
                    vreinterpretq_u64_u8(conv_c[b_idx][l]),
                    veq_r0[b_idx],
                    veq_r1[b_idx],
                    veq_r0[b_idx],
                    veq_r1[b_idx],
                );
                let (p_ab1, p_c1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_ab[b_idx + 1][l]),
                    vreinterpretq_u64_u8(conv_c[b_idx + 1][l]),
                    veq_r0[b_idx + 1],
                    veq_r1[b_idx + 1],
                    veq_r0[b_idx + 1],
                    veq_r1[b_idx + 1],
                );
                acc_ab = crate::gf2_kernels::k0_xor3_neon(acc_ab, p_ab0, p_ab1);
                acc_c = crate::gf2_kernels::k0_xor3_neon(acc_c, p_c0, p_c1);
                b_idx += 2;
            }
            while b_idx < batch_len {
                let r0 = veq_r0[b_idx];
                let r1 = veq_r1[b_idx];
                let vab = vreinterpretq_u64_u8(conv_ab[b_idx][l]);
                let vc = vreinterpretq_u64_u8(conv_c[b_idx][l]);
                let (p_ab, p_c) =
                    crate::gf2_kernels::k0_f128_mul_const_vec2_neon(vab, vc, r0, r1, r0, r1);
                acc_ab = veorq_u64(acc_ab, p_ab);
                acc_c = veorq_u64(acc_c, p_c);
                b_idx += 1;
            }
            vst1q_u64(ptr_ab, acc_ab);
            vst1q_u64(ptr_c, acc_c);
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
                        let ab0 = xor3_u8(
                            vld1q_u8(t0.add(ab_r0[l0] as usize * 16)),
                            vld1q_u8(t1.add(ab_r1[l0] as usize * 16)),
                            vld1q_u8(t2.add(ab_r2[l0] as usize * 16)),
                        );
                        let ab1 = xor3_u8(
                            vld1q_u8(t0.add(ab_r0[l1] as usize * 16)),
                            vld1q_u8(t1.add(ab_r1[l1] as usize * 16)),
                            vld1q_u8(t2.add(ab_r2[l1] as usize * 16)),
                        );
                        let ab2 = xor3_u8(
                            vld1q_u8(t0.add(ab_r0[l2] as usize * 16)),
                            vld1q_u8(t1.add(ab_r1[l2] as usize * 16)),
                            vld1q_u8(t2.add(ab_r2[l2] as usize * 16)),
                        );
                        let ab3 = xor3_u8(
                            vld1q_u8(t0.add(ab_r0[l3] as usize * 16)),
                            vld1q_u8(t1.add(ab_r1[l3] as usize * 16)),
                            vld1q_u8(t2.add(ab_r2[l3] as usize * 16)),
                        );

                        if q == 0 {
                            dst_ab[l0] = veorq_u8(ab0, vld1q_u8(t3.add(ab_r3[l0] as usize * 16)));
                            dst_ab[l1] = veorq_u8(ab1, vld1q_u8(t3.add(ab_r3[l1] as usize * 16)));
                            dst_ab[l2] = veorq_u8(ab2, vld1q_u8(t3.add(ab_r3[l2] as usize * 16)));
                            dst_ab[l3] = veorq_u8(ab3, vld1q_u8(t3.add(ab_r3[l3] as usize * 16)));
                        } else {
                            dst_ab[l0] =
                                xor3_u8(dst_ab[l0], ab0, vld1q_u8(t3.add(ab_r3[l0] as usize * 16)));
                            dst_ab[l1] =
                                xor3_u8(dst_ab[l1], ab1, vld1q_u8(t3.add(ab_r3[l1] as usize * 16)));
                            dst_ab[l2] =
                                xor3_u8(dst_ab[l2], ab2, vld1q_u8(t3.add(ab_r3[l2] as usize * 16)));
                            dst_ab[l3] =
                                xor3_u8(dst_ab[l3], ab3, vld1q_u8(t3.add(ab_r3[l3] as usize * 16)));
                        }
                    }
                } else {
                    for tile in 0..16 {
                        let l0 = tile * 4;
                        let l1 = l0 + 1;
                        let l2 = l0 + 2;
                        let l3 = l0 + 3;
                        let ab0 = xor3_u8(
                            vld1q_u8(t0.add(ab_r0[l0] as usize * 16)),
                            vld1q_u8(t1.add(ab_r1[l0] as usize * 16)),
                            vld1q_u8(t2.add(ab_r2[l0] as usize * 16)),
                        );
                        let ab1 = xor3_u8(
                            vld1q_u8(t0.add(ab_r0[l1] as usize * 16)),
                            vld1q_u8(t1.add(ab_r1[l1] as usize * 16)),
                            vld1q_u8(t2.add(ab_r2[l1] as usize * 16)),
                        );
                        let ab2 = xor3_u8(
                            vld1q_u8(t0.add(ab_r0[l2] as usize * 16)),
                            vld1q_u8(t1.add(ab_r1[l2] as usize * 16)),
                            vld1q_u8(t2.add(ab_r2[l2] as usize * 16)),
                        );
                        let ab3 = xor3_u8(
                            vld1q_u8(t0.add(ab_r0[l3] as usize * 16)),
                            vld1q_u8(t1.add(ab_r1[l3] as usize * 16)),
                            vld1q_u8(t2.add(ab_r2[l3] as usize * 16)),
                        );

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
                    let c0_0 = xor3_u8(
                        vld1q_u8(tc0.add(c0_r0[l0] as usize * 16)),
                        vld1q_u8(tc1.add(c0_r1[l0] as usize * 16)),
                        vld1q_u8(tc2.add(c0_r2[l0] as usize * 16)),
                    );
                    let c0_1 = xor3_u8(
                        vld1q_u8(tc0.add(c0_r0[l1] as usize * 16)),
                        vld1q_u8(tc1.add(c0_r1[l1] as usize * 16)),
                        vld1q_u8(tc2.add(c0_r2[l1] as usize * 16)),
                    );
                    let c0_2 = xor3_u8(
                        vld1q_u8(tc0.add(c0_r0[l2] as usize * 16)),
                        vld1q_u8(tc1.add(c0_r1[l2] as usize * 16)),
                        vld1q_u8(tc2.add(c0_r2[l2] as usize * 16)),
                    );
                    let c0_3 = xor3_u8(
                        vld1q_u8(tc0.add(c0_r0[l3] as usize * 16)),
                        vld1q_u8(tc1.add(c0_r1[l3] as usize * 16)),
                        vld1q_u8(tc2.add(c0_r2[l3] as usize * 16)),
                    );

                    let c1_0 = xor3_u8(
                        vld1q_u8(tc0.add(c1_r0[l0] as usize * 16)),
                        vld1q_u8(tc1.add(c1_r1[l0] as usize * 16)),
                        vld1q_u8(tc2.add(c1_r2[l0] as usize * 16)),
                    );
                    let c1_1 = xor3_u8(
                        vld1q_u8(tc0.add(c1_r0[l1] as usize * 16)),
                        vld1q_u8(tc1.add(c1_r1[l1] as usize * 16)),
                        vld1q_u8(tc2.add(c1_r2[l1] as usize * 16)),
                    );
                    let c1_2 = xor3_u8(
                        vld1q_u8(tc0.add(c1_r0[l2] as usize * 16)),
                        vld1q_u8(tc1.add(c1_r1[l2] as usize * 16)),
                        vld1q_u8(tc2.add(c1_r2[l2] as usize * 16)),
                    );
                    let c1_3 = xor3_u8(
                        vld1q_u8(tc0.add(c1_r0[l3] as usize * 16)),
                        vld1q_u8(tc1.add(c1_r1[l3] as usize * 16)),
                        vld1q_u8(tc2.add(c1_r2[l3] as usize * 16)),
                    );

                    if h == 0 {
                        dst_c0[l0] = veorq_u8(c0_0, vld1q_u8(tc3.add(c0_r3[l0] as usize * 16)));
                        dst_c0[l1] = veorq_u8(c0_1, vld1q_u8(tc3.add(c0_r3[l1] as usize * 16)));
                        dst_c0[l2] = veorq_u8(c0_2, vld1q_u8(tc3.add(c0_r3[l2] as usize * 16)));
                        dst_c0[l3] = veorq_u8(c0_3, vld1q_u8(tc3.add(c0_r3[l3] as usize * 16)));
                        dst_c1[l0] = veorq_u8(c1_0, vld1q_u8(tc3.add(c1_r3[l0] as usize * 16)));
                        dst_c1[l1] = veorq_u8(c1_1, vld1q_u8(tc3.add(c1_r3[l1] as usize * 16)));
                        dst_c1[l2] = veorq_u8(c1_2, vld1q_u8(tc3.add(c1_r3[l2] as usize * 16)));
                        dst_c1[l3] = veorq_u8(c1_3, vld1q_u8(tc3.add(c1_r3[l3] as usize * 16)));
                    } else {
                        dst_c0[l0] =
                            xor3_u8(dst_c0[l0], c0_0, vld1q_u8(tc3.add(c0_r3[l0] as usize * 16)));
                        dst_c0[l1] =
                            xor3_u8(dst_c0[l1], c0_1, vld1q_u8(tc3.add(c0_r3[l1] as usize * 16)));
                        dst_c0[l2] =
                            xor3_u8(dst_c0[l2], c0_2, vld1q_u8(tc3.add(c0_r3[l2] as usize * 16)));
                        dst_c0[l3] =
                            xor3_u8(dst_c0[l3], c0_3, vld1q_u8(tc3.add(c0_r3[l3] as usize * 16)));
                        dst_c1[l0] =
                            xor3_u8(dst_c1[l0], c1_0, vld1q_u8(tc3.add(c1_r3[l0] as usize * 16)));
                        dst_c1[l1] =
                            xor3_u8(dst_c1[l1], c1_1, vld1q_u8(tc3.add(c1_r3[l1] as usize * 16)));
                        dst_c1[l2] =
                            xor3_u8(dst_c1[l2], c1_2, vld1q_u8(tc3.add(c1_r3[l2] as usize * 16)));
                        dst_c1[l3] =
                            xor3_u8(dst_c1[l3], c1_3, vld1q_u8(tc3.add(c1_r3[l3] as usize * 16)));
                    }
                }
            }
        }

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

        for l_pair in 0..32 {
            let l0 = 2 * l_pair;
            let l1 = l0 + 1;
            let ptr_ab0 = (&mut partial_ab[l0].ll as *mut F128).cast::<u64>();
            let ptr_ab1 = (&mut partial_ab[l1].ll as *mut F128).cast::<u64>();
            let ptr_c0_0 = (&mut partial_c_0[l0].ll as *mut F128).cast::<u64>();
            let ptr_c1_0 = (&mut partial_c_1[l0].ll as *mut F128).cast::<u64>();
            let ptr_c0_1 = (&mut partial_c_0[l1].ll as *mut F128).cast::<u64>();
            let ptr_c1_1 = (&mut partial_c_1[l1].ll as *mut F128).cast::<u64>();

            let mut acc_ab0 = vld1q_u64(ptr_ab0);
            let mut acc_ab1 = vld1q_u64(ptr_ab1);
            let mut acc_c0_0 = vld1q_u64(ptr_c0_0);
            let mut acc_c1_0 = vld1q_u64(ptr_c1_0);
            let mut acc_c0_1 = vld1q_u64(ptr_c0_1);
            let mut acc_c1_1 = vld1q_u64(ptr_c1_1);

            let mut b_idx = 0usize;
            while b_idx + 2 <= batch_len {
                let r0_a = veq_r0[b_idx];
                let r1_a = veq_r1[b_idx];
                let r0_b = veq_r0[b_idx + 1];
                let r1_b = veq_r1[b_idx + 1];

                let (p_ab0_a, p_ab1_a) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_ab[b_idx][l0]),
                    vreinterpretq_u64_u8(conv_ab[b_idx][l1]),
                    r0_a,
                    r1_a,
                    r0_a,
                    r1_a,
                );
                let (p_ab0_b, p_ab1_b) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_ab[b_idx + 1][l0]),
                    vreinterpretq_u64_u8(conv_ab[b_idx + 1][l1]),
                    r0_b,
                    r1_b,
                    r0_b,
                    r1_b,
                );
                acc_ab0 = crate::gf2_kernels::k0_xor3_neon(acc_ab0, p_ab0_a, p_ab0_b);
                acc_ab1 = crate::gf2_kernels::k0_xor3_neon(acc_ab1, p_ab1_a, p_ab1_b);

                let (p_c0_0_a, p_c1_0_a) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_c0[b_idx][l0]),
                    vreinterpretq_u64_u8(conv_c1[b_idx][l0]),
                    r0_a,
                    r1_a,
                    r0_a,
                    r1_a,
                );
                let (p_c0_0_b, p_c1_0_b) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_c0[b_idx + 1][l0]),
                    vreinterpretq_u64_u8(conv_c1[b_idx + 1][l0]),
                    r0_b,
                    r1_b,
                    r0_b,
                    r1_b,
                );
                acc_c0_0 = crate::gf2_kernels::k0_xor3_neon(acc_c0_0, p_c0_0_a, p_c0_0_b);
                acc_c1_0 = crate::gf2_kernels::k0_xor3_neon(acc_c1_0, p_c1_0_a, p_c1_0_b);

                let (p_c0_1_a, p_c1_1_a) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_c0[b_idx][l1]),
                    vreinterpretq_u64_u8(conv_c1[b_idx][l1]),
                    r0_a,
                    r1_a,
                    r0_a,
                    r1_a,
                );
                let (p_c0_1_b, p_c1_1_b) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_c0[b_idx + 1][l1]),
                    vreinterpretq_u64_u8(conv_c1[b_idx + 1][l1]),
                    r0_b,
                    r1_b,
                    r0_b,
                    r1_b,
                );
                acc_c0_1 = crate::gf2_kernels::k0_xor3_neon(acc_c0_1, p_c0_1_a, p_c0_1_b);
                acc_c1_1 = crate::gf2_kernels::k0_xor3_neon(acc_c1_1, p_c1_1_a, p_c1_1_b);

                b_idx += 2;
            }
            while b_idx < batch_len {
                let r0 = veq_r0[b_idx];
                let r1 = veq_r1[b_idx];
                let (p_ab0, p_ab1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_ab[b_idx][l0]),
                    vreinterpretq_u64_u8(conv_ab[b_idx][l1]),
                    r0,
                    r1,
                    r0,
                    r1,
                );
                acc_ab0 = veorq_u64(acc_ab0, p_ab0);
                acc_ab1 = veorq_u64(acc_ab1, p_ab1);

                let (p_c0_0, p_c1_0) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_c0[b_idx][l0]),
                    vreinterpretq_u64_u8(conv_c1[b_idx][l0]),
                    r0,
                    r1,
                    r0,
                    r1,
                );
                acc_c0_0 = veorq_u64(acc_c0_0, p_c0_0);
                acc_c1_0 = veorq_u64(acc_c1_0, p_c1_0);

                let (p_c0_1, p_c1_1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    vreinterpretq_u64_u8(conv_c0[b_idx][l1]),
                    vreinterpretq_u64_u8(conv_c1[b_idx][l1]),
                    r0,
                    r1,
                    r0,
                    r1,
                );
                acc_c0_1 = veorq_u64(acc_c0_1, p_c0_1);
                acc_c1_1 = veorq_u64(acc_c1_1, p_c1_1);

                b_idx += 1;
            }

            vst1q_u64(ptr_ab0, acc_ab0);
            vst1q_u64(ptr_ab1, acc_ab1);
            vst1q_u64(ptr_c0_0, acc_c0_0);
            vst1q_u64(ptr_c1_0, acc_c1_0);
            vst1q_u64(ptr_c0_1, acc_c0_1);
            vst1q_u64(ptr_c1_1, acc_c1_1);
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
    unsafe {
        crate::gf2_kernels::k0_f128_urm_accumulate_convert_batch_ab_slice(
            batch_ab_bytes.as_ptr() as *const u8,
            batch_n_b_med.as_ptr(),
            batch_eq_lo.as_ptr() as *const u64,
            batch_len,
            convert.as_ptr() as *const u64,
            partial_ab.as_mut_ptr() as *mut u64,
        );
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
pub(crate) const HORNER_CARRY_X16: u16 = 0x5e;

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn bcax_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::vbcaxq_u8(a, b, c) }
}

#[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
#[inline(always)]
unsafe fn bcax_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::veorq_u8(a, core::arch::aarch64::vbicq_u8(b, c)) }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
const fn byte_from_word<const J: u32>(word: u64) -> usize {
    ((word >> (8 * J)) & 0xff) as usize
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull_lo_u16(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint16x8_t {
    use core::arch::aarch64::*;
    unsafe {
        vreinterpretq_u16_p16(vmull_p8(
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(a)),
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(b)),
        ))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull_hi_u16(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint16x8_t {
    use core::arch::aarch64::*;
    unsafe {
        vreinterpretq_u16_p16(vmull_p8(
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(a)),
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(b)),
        ))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_4_regs<const BH: usize, const ODD: bool>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    a_byte: usize,
    da0: &mut core::arch::aarch64::uint8x16_t,
    da1: &mut core::arch::aarch64::uint8x16_t,
    da2: &mut core::arch::aarch64::uint8x16_t,
    da3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let t = if ODD {
            half_swapped_table_base
        } else {
            table_base
        };
        let ra = t.add(a_byte * 64);
        *da0 = veorq_u8(*da0, vld1q_u8(ra.add((0 ^ BH) * 16)));
        *da1 = veorq_u8(*da1, vld1q_u8(ra.add((1 ^ BH) * 16)));
        *da2 = veorq_u8(*da2, vld1q_u8(ra.add((2 ^ BH) * 16)));
        *da3 = veorq_u8(*da3, vld1q_u8(ra.add((3 ^ BH) * 16)));
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn xor_apply_byte_pair_into_4_regs<
    const BH0: usize,
    const ODD0: bool,
    const BH1: usize,
    const ODD1: bool,
>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    a_byte0: usize,
    a_byte1: usize,
    da0: &mut core::arch::aarch64::uint8x16_t,
    da1: &mut core::arch::aarch64::uint8x16_t,
    da2: &mut core::arch::aarch64::uint8x16_t,
    da3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let t0 = if ODD0 {
            half_swapped_table_base
        } else {
            table_base
        };
        let t1 = if ODD1 {
            half_swapped_table_base
        } else {
            table_base
        };
        let ra0 = t0.add(a_byte0 * 64);
        let ra1 = t1.add(a_byte1 * 64);
        *da0 = xor3_u8(
            *da0,
            vld1q_u8(ra0.add((0 ^ BH0) * 16)),
            vld1q_u8(ra1.add((0 ^ BH1) * 16)),
        );
        *da1 = xor3_u8(
            *da1,
            vld1q_u8(ra0.add((1 ^ BH0) * 16)),
            vld1q_u8(ra1.add((1 ^ BH1) * 16)),
        );
        *da2 = xor3_u8(
            *da2,
            vld1q_u8(ra0.add((2 ^ BH0) * 16)),
            vld1q_u8(ra1.add((2 ^ BH1) * 16)),
        );
        *da3 = xor3_u8(
            *da3,
            vld1q_u8(ra0.add((3 ^ BH0) * 16)),
            vld1q_u8(ra1.add((3 ^ BH1) * 16)),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_into_4_regs(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    word: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let r0 = table_base.add(byte_from_word::<0>(word) * 64);
        let mut d0 = vld1q_u8(r0);
        let mut d1 = vld1q_u8(r0.add(16));
        let mut d2 = vld1q_u8(r0.add(32));
        let mut d3 = vld1q_u8(r0.add(48));
        xor_apply_byte_pair_into_4_regs::<0, true, 1, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<1>(word),
            byte_from_word::<2>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_pair_into_4_regs::<1, true, 2, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<3>(word),
            byte_from_word::<4>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_pair_into_4_regs::<2, true, 3, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<5>(word),
            byte_from_word::<6>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs::<3, true>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<7>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        (d0, d1, d2, d3)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn horner_absorb_first(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &[core::arch::aarch64::uint16x8_t; 8],
    hi: &[core::arch::aarch64::uint16x8_t; 8],
) {
    use core::arch::aarch64::*;
    unsafe {
        for i in 0..8 {
            let shifted = vshlq_n_u16::<1>(acc[i]);
            acc[i] = xor3_u16(shifted, lo[i], hi[i]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn horner_absorb(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &[core::arch::aarch64::uint16x8_t; 8],
    hi: &[core::arch::aarch64::uint16x8_t; 8],
    carry_not: core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        for i in 0..8 {
            let cur = acc[i];
            let shifted = vreinterpretq_u8_u16(vshlq_n_u16::<1>(cur));
            let msb = vreinterpretq_u8_s16(vshrq_n_s16::<15>(vreinterpretq_s16_u16(cur)));
            let folded = bcax_u8(shifted, msb, carry_not);
            acc[i] = vreinterpretq_u16_u8(xor3_u8(
                folded,
                vreinterpretq_u8_u16(lo[i]),
                vreinterpretq_u8_u16(hi[i]),
            ));
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
    use crate::field::gf2_8::neon::{gf8_mul_vec16, gf8_reduce_vec16};
    use core::arch::aarch64::*;

    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let plain = inv_table.data_ptr();
    let plain_sw = inv_table.half_swapped_data_ptr();
    let scaled = inv_table.scaled_x4_data_ptr();
    let scaled_sw = inv_table.scaled_x4_half_swapped_data_ptr();

    unsafe {
        let a_base = a_packed.as_ptr().add(byte_base_b);
        let b_base = b_packed.as_ptr().add(byte_base_b);
        let b_base_ptr = b_base as *const u64;
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
            let not_1b = vdupq_n_u8(!0x1bu8);
            macro_rules! gf8_mul_x_16_xor {
                ($x:expr, $d:expr) => {{
                    let shifted = vshlq_n_u8::<1>($x);
                    let mask = vreinterpretq_u8_s8(vshrq_n_s8::<7>(vreinterpretq_s8_u8($x)));
                    let x_mul = bcax_u8(shifted, mask, not_1b);
                    veorq_u8(x_mul, $d)
                }};
            }
            macro_rules! a_pair_sum {
                ($r:literal) => {{
                    let aw_lo = u64::from_le(core::ptr::read_unaligned(
                        a_base.add($r * N_CHUNKS).cast::<u64>(),
                    ));
                    let aw_hi = u64::from_le(core::ptr::read_unaligned(
                        a_base.add(($r + 4) * N_CHUNKS).cast::<u64>(),
                    ));
                    let (lo0, lo1, lo2, lo3) = apply_word_into_4_regs(plain, plain_sw, aw_lo);
                    let (hi0, hi1, hi2, hi3) = apply_word_into_4_regs(scaled, scaled_sw, aw_hi);
                    (
                        veorq_u8(lo0, hi0),
                        veorq_u8(lo1, hi1),
                        veorq_u8(lo2, hi2),
                        veorq_u8(lo3, hi3),
                    )
                }};
            }
            if (b_base_ptr.add(6).read_unaligned() & b7) == !0u64 {
                let (mut acc0, mut acc1, mut acc2, mut acc3) = a_pair_sum!(3);
                let (d2_0, d2_1, d2_2, d2_3) = a_pair_sum!(2);
                acc0 = gf8_mul_x_16_xor!(acc0, d2_0);
                acc1 = gf8_mul_x_16_xor!(acc1, d2_1);
                acc2 = gf8_mul_x_16_xor!(acc2, d2_2);
                acc3 = gf8_mul_x_16_xor!(acc3, d2_3);
                let (d1_0, d1_1, d1_2, d1_3) = a_pair_sum!(1);
                acc0 = gf8_mul_x_16_xor!(acc0, d1_0);
                acc1 = gf8_mul_x_16_xor!(acc1, d1_1);
                acc2 = gf8_mul_x_16_xor!(acc2, d1_2);
                acc3 = gf8_mul_x_16_xor!(acc3, d1_3);
                let (d0_0, d0_1, d0_2, d0_3) = a_pair_sum!(0);
                acc0 = gf8_mul_x_16_xor!(acc0, d0_0);
                acc1 = gf8_mul_x_16_xor!(acc1, d0_1);
                acc2 = gf8_mul_x_16_xor!(acc2, d0_2);
                acc3 = gf8_mul_x_16_xor!(acc3, d0_3);

                let p = out.as_mut_ptr();
                vst1q_u8(p, acc0);
                vst1q_u8(p.add(16), acc1);
                vst1q_u8(p.add(32), acc2);
                vst1q_u8(p.add(48), acc3);
                return;
            }
        }

        if b7 == 0
            && (b_base_ptr.add(1).read_unaligned()
                | b_base_ptr.add(2).read_unaligned()
                | b_base_ptr.add(3).read_unaligned()
                | b_base_ptr.add(4).read_unaligned()
                | b_base_ptr.add(5).read_unaligned()
                | b_base_ptr.add(6).read_unaligned())
                == 0
        {
            let aw0 = u64::from_le(core::ptr::read_unaligned(a_base.cast::<u64>()));
            let bw0 = u64::from_le(b0);
            let (da0, da1, da2, da3) = apply_word_into_4_regs(plain, plain_sw, aw0);
            let (db0, db1, db2, db3) = apply_word_into_4_regs(plain, plain_sw, bw0);
            let r0 = gf8_mul_vec16(da0, db0);
            let r1 = gf8_mul_vec16(da1, db1);
            let r2 = gf8_mul_vec16(da2, db2);
            let r3 = gf8_mul_vec16(da3, db3);
            let p = out.as_mut_ptr();
            vst1q_u8(p, r0);
            vst1q_u8(p.add(16), r1);
            vst1q_u8(p.add(32), r2);
            vst1q_u8(p.add(48), r3);
            return;
        }

        macro_rules! products {
            ($at:expr, $at_sw:expr, $k:literal) => {{
                let aw = u64::from_le(core::ptr::read_unaligned(
                    a_base.add($k * N_CHUNKS).cast::<u64>(),
                ));
                let bw = u64::from_le(core::ptr::read_unaligned(
                    b_base.add($k * N_CHUNKS).cast::<u64>(),
                ));
                let (da0, da1, da2, da3) = apply_word_into_4_regs($at, $at_sw, aw);
                let (db0, db1, db2, db3) = apply_word_into_4_regs(plain, plain_sw, bw);
                [
                    pmull_lo_u16(da0, db0),
                    pmull_hi_u16(da0, db0),
                    pmull_lo_u16(da1, db1),
                    pmull_hi_u16(da1, db1),
                    pmull_lo_u16(da2, db2),
                    pmull_hi_u16(da2, db2),
                    pmull_lo_u16(da3, db3),
                    pmull_hi_u16(da3, db3),
                ]
            }};
        }

        let lo3 = products!(plain, plain_sw, 3);
        let hi3 = products!(scaled, scaled_sw, 7);
        let mut acc = [
            veorq_u16(lo3[0], hi3[0]),
            veorq_u16(lo3[1], hi3[1]),
            veorq_u16(lo3[2], hi3[2]),
            veorq_u16(lo3[3], hi3[3]),
            veorq_u16(lo3[4], hi3[4]),
            veorq_u16(lo3[5], hi3[5]),
            veorq_u16(lo3[6], hi3[6]),
            veorq_u16(lo3[7], hi3[7]),
        ];

        let lo2 = products!(plain, plain_sw, 2);
        let hi2 = products!(scaled, scaled_sw, 6);
        horner_absorb_first(&mut acc, &lo2, &hi2);

        let carry_not = vreinterpretq_u8_u16(vdupq_n_u16(!HORNER_CARRY_X16));

        let lo1 = products!(plain, plain_sw, 1);
        let hi1 = products!(scaled, scaled_sw, 5);
        horner_absorb(&mut acc, &lo1, &hi1, carry_not);

        let lo0 = products!(plain, plain_sw, 0);
        let hi0 = products!(scaled, scaled_sw, 4);
        horner_absorb(&mut acc, &lo0, &hi0, carry_not);

        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[0]), vreinterpretq_u8_u16(acc[1]));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[2]), vreinterpretq_u8_u16(acc[3]));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[4]), vreinterpretq_u8_u16(acc[5]));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[6]), vreinterpretq_u8_u16(acc[7]));

        let p = out.as_mut_ptr();
        vst1q_u8(p, r0);
        vst1q_u8(p.add(16), r1);
        vst1q_u8(p.add(32), r2);
        vst1q_u8(p.add(48), r3);
    }
}

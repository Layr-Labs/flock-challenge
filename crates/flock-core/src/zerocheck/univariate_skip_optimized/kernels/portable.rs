#[cfg(not(target_arch = "aarch64"))]
use super::super::F128;
use super::super::{F8, InvNttTableByteSingleGf8, N_CHUNKS};

#[allow(dead_code)]
pub(in super::super) fn bit_transpose_64bytes_scalar(input: &[u8; 64], output: &mut [u8; 64]) {
    for b_chunk in 0..8 {
        let y = u64::from_le_bytes([
            input[b_chunk],
            input[8 + b_chunk],
            input[16 + b_chunk],
            input[24 + b_chunk],
            input[32 + b_chunk],
            input[40 + b_chunk],
            input[48 + b_chunk],
            input[56 + b_chunk],
        ]);
        let transposed = unsafe { crate::gf2_kernels::gf2_kern_transpose_8x8_u64(y) };
        output[b_chunk * 8..b_chunk * 8 + 8].copy_from_slice(&transposed.to_le_bytes());
    }
}

#[inline(always)]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn shift_reduce_inner_ab_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let _ = (a_col, b_col);
        debug_assert_eq!(inv_table.ell, 64);
        debug_assert_eq!(inv_table.n_chunks, 8);
        unsafe {
            crate::gf2_kernels::k0_gf8_shift_reduce_inner_ab_slice(
                a_packed.as_ptr().add(byte_base_b),
                b_packed.as_ptr().add(byte_base_b),
                inv_table.data_ptr(),
                inv_table.composite_data_ptr(),
                out.as_mut_ptr(),
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if inv_table.ell == 64 && inv_table.n_chunks == 8 {
        let _ = (a_col, b_col);
        unsafe {
            let mut a_rows = [[0u64; 8]; 8];
            let mut b_rows = [[0u64; 8]; 8];
            for k in 0..8 {
                let off = byte_base_b + k * N_CHUNKS;
                a_rows[k] = inv_table.apply_x86_u64x8_unchecked(a_packed.as_ptr().add(off));
                b_rows[k] = inv_table.apply_x86_u64x8_unchecked(b_packed.as_ptr().add(off));
            }
            for w in 0..8 {
                let mut ev0 = 0u64;
                let mut od0 = 0u64;
                let mut ev1 = 0u64;
                let mut od1 = 0u64;
                for k in 0..8 {
                    let shift = k as u64;
                    let aw = a_rows[k][w];
                    let bw = b_rows[k][w];
                    ev0 ^= crate::gf2_kernels::gf2_kern_gf8_unreduced_even_x2(aw, bw) << shift;
                    od0 ^= crate::gf2_kernels::gf2_kern_gf8_unreduced_odd_x2(aw, bw) << shift;
                    let aw_hi = aw >> 32;
                    let bw_hi = bw >> 32;
                    ev1 ^=
                        crate::gf2_kernels::gf2_kern_gf8_unreduced_even_x2(aw_hi, bw_hi) << shift;
                    od1 ^= crate::gf2_kernels::gf2_kern_gf8_unreduced_odd_x2(aw_hi, bw_hi) << shift;
                }
                let r0 = crate::gf2_kernels::gf2_kern_gf8_reduce_deg21_pair_x4(ev0, od0);
                let r1 = crate::gf2_kernels::gf2_kern_gf8_reduce_deg21_pair_x4(ev1, od1);
                let packed = crate::gf2_kernels::gf2_kern_pack_even_bytes_x8(r0, r1);
                let base = w * 8;
                out[base..base + 8].copy_from_slice(&packed.to_le_bytes());
            }
        }
        return;
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut acc_even = [0u64; 16];
        let mut acc_odd = [0u64; 16];
        for k in 0..8 {
            let off = byte_base_b + k * N_CHUNKS;
            let shift = k as u64;
            inv_table.apply(&a_packed[off..off + N_CHUNKS], a_col);
            inv_table.apply(&b_packed[off..off + N_CHUNKS], b_col);
            let a_ptr = a_col.as_ptr() as *const u8;
            let b_ptr = b_col.as_ptr() as *const u8;
            for g in 0..16 {
                unsafe {
                    let a4 = core::ptr::read_unaligned(a_ptr.add(g * 4) as *const u32) as u64;
                    let b4 = core::ptr::read_unaligned(b_ptr.add(g * 4) as *const u32) as u64;
                    acc_even[g] ^=
                        crate::gf2_kernels::gf2_kern_gf8_unreduced_even_x2(a4, b4) << shift;
                    acc_odd[g] ^=
                        crate::gf2_kernels::gf2_kern_gf8_unreduced_odd_x2(a4, b4) << shift;
                }
            }
        }
        for g in 0..16 {
            let packed_u32 = unsafe {
                let r4 =
                    crate::gf2_kernels::gf2_kern_gf8_reduce_deg21_pair_x4(acc_even[g], acc_odd[g]);
                crate::gf2_kernels::gf2_kern_pack_even_bytes_u32(r4) as u32
            };
            let base = g * 4;
            out[base..base + 4].copy_from_slice(&packed_u32.to_le_bytes());
        }
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [super::super::KaratsubaAcc; 64],
    partial_c: &mut [super::super::KaratsubaAcc; 64],
) {
    assert!(convert.len() >= 16 * 256 && n_b_med <= 16);
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let conv_ptr = convert.as_ptr();
        let veq = _mm_set_epi64x(eq_lo_val.hi as i64, eq_lo_val.lo as i64);
        let veq_sum = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4e>(veq));
        macro_rules! mul_karatsuba_acc {
            ($dst:expr, $acc:expr) => {{
                let p_ll = _mm_clmulepi64_si128::<0x00>($acc, veq);
                let p_hh = _mm_clmulepi64_si128::<0x11>($acc, veq);
                let a_sum = _mm_xor_si128($acc, _mm_shuffle_epi32::<0x4e>($acc));
                let p_mid = _mm_clmulepi64_si128::<0x00>(a_sum, veq_sum);
                let ptr = ($dst as *mut super::super::KaratsubaAcc).cast::<__m128i>();
                _mm_store_si128(ptr, _mm_xor_si128(_mm_load_si128(ptr), p_ll));
                _mm_store_si128(ptr.add(1), _mm_xor_si128(_mm_load_si128(ptr.add(1)), p_mid));
                _mm_store_si128(ptr.add(2), _mm_xor_si128(_mm_load_si128(ptr.add(2)), p_hh));
            }};
        }
        for tile in 0..16 {
            let l0 = tile * 4;
            let l1 = l0 + 1;
            let l2 = l0 + 2;
            let l3 = l0 + 3;
            let ab_row0 = &chunk_ab_bytes[0];
            let c_row0 = &chunk_c_bytes[0];
            let mut ab0 = _mm_load_si128(conv_ptr.add(ab_row0[l0] as usize).cast::<__m128i>());
            let mut ab1 = _mm_load_si128(conv_ptr.add(ab_row0[l1] as usize).cast::<__m128i>());
            let mut ab2 = _mm_load_si128(conv_ptr.add(ab_row0[l2] as usize).cast::<__m128i>());
            let mut ab3 = _mm_load_si128(conv_ptr.add(ab_row0[l3] as usize).cast::<__m128i>());
            let mut c0 = _mm_load_si128(conv_ptr.add(c_row0[l0] as usize).cast::<__m128i>());
            let mut c1 = _mm_load_si128(conv_ptr.add(c_row0[l1] as usize).cast::<__m128i>());
            let mut c2 = _mm_load_si128(conv_ptr.add(c_row0[l2] as usize).cast::<__m128i>());
            let mut c3 = _mm_load_si128(conv_ptr.add(c_row0[l3] as usize).cast::<__m128i>());

            for b_med in 1..n_b_med {
                let table_base = conv_ptr.add(b_med * 256);
                let ab_row = &chunk_ab_bytes[b_med];
                let c_row = &chunk_c_bytes[b_med];
                ab0 = _mm_xor_si128(
                    ab0,
                    _mm_load_si128(table_base.add(ab_row[l0] as usize).cast::<__m128i>()),
                );
                ab1 = _mm_xor_si128(
                    ab1,
                    _mm_load_si128(table_base.add(ab_row[l1] as usize).cast::<__m128i>()),
                );
                ab2 = _mm_xor_si128(
                    ab2,
                    _mm_load_si128(table_base.add(ab_row[l2] as usize).cast::<__m128i>()),
                );
                ab3 = _mm_xor_si128(
                    ab3,
                    _mm_load_si128(table_base.add(ab_row[l3] as usize).cast::<__m128i>()),
                );
                c0 = _mm_xor_si128(
                    c0,
                    _mm_load_si128(table_base.add(c_row[l0] as usize).cast::<__m128i>()),
                );
                c1 = _mm_xor_si128(
                    c1,
                    _mm_load_si128(table_base.add(c_row[l1] as usize).cast::<__m128i>()),
                );
                c2 = _mm_xor_si128(
                    c2,
                    _mm_load_si128(table_base.add(c_row[l2] as usize).cast::<__m128i>()),
                );
                c3 = _mm_xor_si128(
                    c3,
                    _mm_load_si128(table_base.add(c_row[l3] as usize).cast::<__m128i>()),
                );
            }

            mul_karatsuba_acc!(&mut partial_ab[l0], ab0);
            mul_karatsuba_acc!(&mut partial_c[l0], c0);
            mul_karatsuba_acc!(&mut partial_ab[l1], ab1);
            mul_karatsuba_acc!(&mut partial_c[l1], c1);
            mul_karatsuba_acc!(&mut partial_ab[l2], ab2);
            mul_karatsuba_acc!(&mut partial_c[l2], c2);
            mul_karatsuba_acc!(&mut partial_ab[l3], ab3);
            mul_karatsuba_acc!(&mut partial_c[l3], c3);
        }
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    for lane in 0..64 {
        let mut converted_ab = F128::ZERO;
        let mut converted_c = F128::ZERO;
        for b_med in 0..n_b_med {
            let table_base = b_med * 256;
            unsafe {
                converted_ab +=
                    *convert.get_unchecked(table_base + chunk_ab_bytes[b_med][lane] as usize);
                converted_c +=
                    *convert.get_unchecked(table_base + chunk_c_bytes[b_med][lane] as usize);
            }
        }
        let ab_prod = converted_ab * eq_lo_val;
        let c_prod = converted_c * eq_lo_val;
        partial_ab[lane].ll += ab_prod;
        partial_c[lane].ll += c_prod;
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn accumulate_convert_batch(
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
    assert!(convert.len() >= 16 * 256 && batch_len <= super::super::URM_BATCH);
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let conv_ptr = convert.as_ptr();
        let mut conv_ab = [[_mm_setzero_si128(); 64]; super::super::URM_BATCH];
        let mut conv_c = [[_mm_setzero_si128(); 64]; super::super::URM_BATCH];

        for q in 0..4 {
            let b_base = q * 4;
            let t0 = conv_ptr.add(b_base * 256);
            let t1 = conv_ptr.add((b_base + 1) * 256);
            let t2 = conv_ptr.add((b_base + 2) * 256);
            let t3 = conv_ptr.add((b_base + 3) * 256);
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
                    let mut ab0 =
                        _mm_load_si128(t0.add(ab_rows[b_base][l0] as usize).cast::<__m128i>());
                    let mut ab1 =
                        _mm_load_si128(t0.add(ab_rows[b_base][l1] as usize).cast::<__m128i>());
                    let mut ab2 =
                        _mm_load_si128(t0.add(ab_rows[b_base][l2] as usize).cast::<__m128i>());
                    let mut ab3 =
                        _mm_load_si128(t0.add(ab_rows[b_base][l3] as usize).cast::<__m128i>());
                    let mut c0 =
                        _mm_load_si128(t0.add(c_rows[b_base][l0] as usize).cast::<__m128i>());
                    let mut c1 =
                        _mm_load_si128(t0.add(c_rows[b_base][l1] as usize).cast::<__m128i>());
                    let mut c2 =
                        _mm_load_si128(t0.add(c_rows[b_base][l2] as usize).cast::<__m128i>());
                    let mut c3 =
                        _mm_load_si128(t0.add(c_rows[b_base][l3] as usize).cast::<__m128i>());

                    ab0 = _mm_xor_si128(
                        ab0,
                        _mm_load_si128(t1.add(ab_rows[b_base + 1][l0] as usize).cast::<__m128i>()),
                    );
                    ab1 = _mm_xor_si128(
                        ab1,
                        _mm_load_si128(t1.add(ab_rows[b_base + 1][l1] as usize).cast::<__m128i>()),
                    );
                    ab2 = _mm_xor_si128(
                        ab2,
                        _mm_load_si128(t1.add(ab_rows[b_base + 1][l2] as usize).cast::<__m128i>()),
                    );
                    ab3 = _mm_xor_si128(
                        ab3,
                        _mm_load_si128(t1.add(ab_rows[b_base + 1][l3] as usize).cast::<__m128i>()),
                    );
                    c0 = _mm_xor_si128(
                        c0,
                        _mm_load_si128(t1.add(c_rows[b_base + 1][l0] as usize).cast::<__m128i>()),
                    );
                    c1 = _mm_xor_si128(
                        c1,
                        _mm_load_si128(t1.add(c_rows[b_base + 1][l1] as usize).cast::<__m128i>()),
                    );
                    c2 = _mm_xor_si128(
                        c2,
                        _mm_load_si128(t1.add(c_rows[b_base + 1][l2] as usize).cast::<__m128i>()),
                    );
                    c3 = _mm_xor_si128(
                        c3,
                        _mm_load_si128(t1.add(c_rows[b_base + 1][l3] as usize).cast::<__m128i>()),
                    );

                    ab0 = _mm_xor_si128(
                        ab0,
                        _mm_load_si128(t2.add(ab_rows[b_base + 2][l0] as usize).cast::<__m128i>()),
                    );
                    ab1 = _mm_xor_si128(
                        ab1,
                        _mm_load_si128(t2.add(ab_rows[b_base + 2][l1] as usize).cast::<__m128i>()),
                    );
                    ab2 = _mm_xor_si128(
                        ab2,
                        _mm_load_si128(t2.add(ab_rows[b_base + 2][l2] as usize).cast::<__m128i>()),
                    );
                    ab3 = _mm_xor_si128(
                        ab3,
                        _mm_load_si128(t2.add(ab_rows[b_base + 2][l3] as usize).cast::<__m128i>()),
                    );
                    c0 = _mm_xor_si128(
                        c0,
                        _mm_load_si128(t2.add(c_rows[b_base + 2][l0] as usize).cast::<__m128i>()),
                    );
                    c1 = _mm_xor_si128(
                        c1,
                        _mm_load_si128(t2.add(c_rows[b_base + 2][l1] as usize).cast::<__m128i>()),
                    );
                    c2 = _mm_xor_si128(
                        c2,
                        _mm_load_si128(t2.add(c_rows[b_base + 2][l2] as usize).cast::<__m128i>()),
                    );
                    c3 = _mm_xor_si128(
                        c3,
                        _mm_load_si128(t2.add(c_rows[b_base + 2][l3] as usize).cast::<__m128i>()),
                    );

                    ab0 = _mm_xor_si128(
                        ab0,
                        _mm_load_si128(t3.add(ab_rows[b_base + 3][l0] as usize).cast::<__m128i>()),
                    );
                    ab1 = _mm_xor_si128(
                        ab1,
                        _mm_load_si128(t3.add(ab_rows[b_base + 3][l1] as usize).cast::<__m128i>()),
                    );
                    ab2 = _mm_xor_si128(
                        ab2,
                        _mm_load_si128(t3.add(ab_rows[b_base + 3][l2] as usize).cast::<__m128i>()),
                    );
                    ab3 = _mm_xor_si128(
                        ab3,
                        _mm_load_si128(t3.add(ab_rows[b_base + 3][l3] as usize).cast::<__m128i>()),
                    );
                    c0 = _mm_xor_si128(
                        c0,
                        _mm_load_si128(t3.add(c_rows[b_base + 3][l0] as usize).cast::<__m128i>()),
                    );
                    c1 = _mm_xor_si128(
                        c1,
                        _mm_load_si128(t3.add(c_rows[b_base + 3][l1] as usize).cast::<__m128i>()),
                    );
                    c2 = _mm_xor_si128(
                        c2,
                        _mm_load_si128(t3.add(c_rows[b_base + 3][l2] as usize).cast::<__m128i>()),
                    );
                    c3 = _mm_xor_si128(
                        c3,
                        _mm_load_si128(t3.add(c_rows[b_base + 3][l3] as usize).cast::<__m128i>()),
                    );

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
                        dst_ab[l0] = _mm_xor_si128(dst_ab[l0], ab0);
                        dst_ab[l1] = _mm_xor_si128(dst_ab[l1], ab1);
                        dst_ab[l2] = _mm_xor_si128(dst_ab[l2], ab2);
                        dst_ab[l3] = _mm_xor_si128(dst_ab[l3], ab3);
                        dst_c[l0] = _mm_xor_si128(dst_c[l0], c0);
                        dst_c[l1] = _mm_xor_si128(dst_c[l1], c1);
                        dst_c[l2] = _mm_xor_si128(dst_c[l2], c2);
                        dst_c[l3] = _mm_xor_si128(dst_c[l3], c3);
                    }
                }
            }
        }

        let mut veq_r0 = [_mm_setzero_si128(); super::super::URM_BATCH];
        let mut veq_r1 = [_mm_setzero_si128(); super::super::URM_BATCH];
        for b_idx in 0..batch_len {
            let eq_lo_val = batch_eq_lo[b_idx];
            let veq = _mm_set_epi64x(eq_lo_val.hi as i64, eq_lo_val.lo as i64);
            let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(veq);
            veq_r0[b_idx] = r0;
            veq_r1[b_idx] = r1;
        }

        for l in 0..64 {
            let ptr_ab = (&mut partial_ab[l].ll as *mut F128).cast::<__m128i>();
            let ptr_c = (&mut partial_c[l].ll as *mut F128).cast::<__m128i>();
            let mut acc_ab = _mm_load_si128(ptr_ab);
            let mut acc_c = _mm_load_si128(ptr_c);
            for b_idx in 0..batch_len {
                let r0 = veq_r0[b_idx];
                let r1 = veq_r1[b_idx];
                let ab = conv_ab[b_idx][l];
                let c = conv_c[b_idx][l];
                let (p_ab, p_c) =
                    crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(ab, c, r0, r1, r0, r1);
                acc_ab = _mm_xor_si128(acc_ab, p_ab);
                acc_c = _mm_xor_si128(acc_c, p_c);
            }
            _mm_store_si128(ptr_ab, acc_ab);
            _mm_store_si128(ptr_c, acc_c);
        }
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    for b_idx in 0..batch_len {
        accumulate_convert(
            &batch_ab_bytes[b_idx],
            &batch_c_bytes[b_idx],
            16,
            convert,
            batch_eq_lo[b_idx],
            partial_ab,
            partial_c,
        );
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [super::super::KaratsubaAcc; 64],
    partial_c_0: &mut [super::super::KaratsubaAcc; 64],
    partial_c_1: &mut [super::super::KaratsubaAcc; 64],
) {
    assert!(convert.len() >= 16 * 256 && n_b_med <= 16);
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let conv_ptr = convert.as_ptr();
        let veq = _mm_set_epi64x(eq_lo_val.hi as i64, eq_lo_val.lo as i64);
        let veq_sum = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4e>(veq));
        macro_rules! mul_karatsuba_acc {
            ($dst:expr, $acc:expr) => {{
                let p_ll = _mm_clmulepi64_si128::<0x00>($acc, veq);
                let p_hh = _mm_clmulepi64_si128::<0x11>($acc, veq);
                let a_sum = _mm_xor_si128($acc, _mm_shuffle_epi32::<0x4e>($acc));
                let p_mid = _mm_clmulepi64_si128::<0x00>(a_sum, veq_sum);
                let ptr = ($dst as *mut super::super::KaratsubaAcc).cast::<__m128i>();
                _mm_store_si128(ptr, _mm_xor_si128(_mm_load_si128(ptr), p_ll));
                _mm_store_si128(ptr.add(1), _mm_xor_si128(_mm_load_si128(ptr.add(1)), p_mid));
                _mm_store_si128(ptr.add(2), _mm_xor_si128(_mm_load_si128(ptr.add(2)), p_hh));
            }};
        }
        for tile in 0..16 {
            let l0 = tile * 4;
            let l1 = l0 + 1;
            let l2 = l0 + 2;
            let l3 = l0 + 3;
            let ab_row0 = &chunk_ab_bytes[0];
            let c_row0 = &chunk_c_bytes[0];
            let mut ab0 = _mm_load_si128(conv_ptr.add(ab_row0[l0] as usize).cast::<__m128i>());
            let mut ab1 = _mm_load_si128(conv_ptr.add(ab_row0[l1] as usize).cast::<__m128i>());
            let mut ab2 = _mm_load_si128(conv_ptr.add(ab_row0[l2] as usize).cast::<__m128i>());
            let mut ab3 = _mm_load_si128(conv_ptr.add(ab_row0[l3] as usize).cast::<__m128i>());
            let mut c0_0 =
                _mm_load_si128(conv_ptr.add((c_row0[l0] & 0x55) as usize).cast::<__m128i>());
            let mut c0_1 =
                _mm_load_si128(conv_ptr.add((c_row0[l1] & 0x55) as usize).cast::<__m128i>());
            let mut c0_2 =
                _mm_load_si128(conv_ptr.add((c_row0[l2] & 0x55) as usize).cast::<__m128i>());
            let mut c0_3 =
                _mm_load_si128(conv_ptr.add((c_row0[l3] & 0x55) as usize).cast::<__m128i>());
            let mut c1_0 =
                _mm_load_si128(conv_ptr.add((c_row0[l0] & 0xaa) as usize).cast::<__m128i>());
            let mut c1_1 =
                _mm_load_si128(conv_ptr.add((c_row0[l1] & 0xaa) as usize).cast::<__m128i>());
            let mut c1_2 =
                _mm_load_si128(conv_ptr.add((c_row0[l2] & 0xaa) as usize).cast::<__m128i>());
            let mut c1_3 =
                _mm_load_si128(conv_ptr.add((c_row0[l3] & 0xaa) as usize).cast::<__m128i>());

            for b_med in 1..n_b_med {
                let table_base = conv_ptr.add(b_med * 256);
                let ab_row = &chunk_ab_bytes[b_med];
                let c_row = &chunk_c_bytes[b_med];
                ab0 = _mm_xor_si128(
                    ab0,
                    _mm_load_si128(table_base.add(ab_row[l0] as usize).cast::<__m128i>()),
                );
                ab1 = _mm_xor_si128(
                    ab1,
                    _mm_load_si128(table_base.add(ab_row[l1] as usize).cast::<__m128i>()),
                );
                ab2 = _mm_xor_si128(
                    ab2,
                    _mm_load_si128(table_base.add(ab_row[l2] as usize).cast::<__m128i>()),
                );
                ab3 = _mm_xor_si128(
                    ab3,
                    _mm_load_si128(table_base.add(ab_row[l3] as usize).cast::<__m128i>()),
                );
                c0_0 = _mm_xor_si128(
                    c0_0,
                    _mm_load_si128(
                        table_base
                            .add((c_row[l0] & 0x55) as usize)
                            .cast::<__m128i>(),
                    ),
                );
                c0_1 = _mm_xor_si128(
                    c0_1,
                    _mm_load_si128(
                        table_base
                            .add((c_row[l1] & 0x55) as usize)
                            .cast::<__m128i>(),
                    ),
                );
                c0_2 = _mm_xor_si128(
                    c0_2,
                    _mm_load_si128(
                        table_base
                            .add((c_row[l2] & 0x55) as usize)
                            .cast::<__m128i>(),
                    ),
                );
                c0_3 = _mm_xor_si128(
                    c0_3,
                    _mm_load_si128(
                        table_base
                            .add((c_row[l3] & 0x55) as usize)
                            .cast::<__m128i>(),
                    ),
                );
                c1_0 = _mm_xor_si128(
                    c1_0,
                    _mm_load_si128(
                        table_base
                            .add((c_row[l0] & 0xaa) as usize)
                            .cast::<__m128i>(),
                    ),
                );
                c1_1 = _mm_xor_si128(
                    c1_1,
                    _mm_load_si128(
                        table_base
                            .add((c_row[l1] & 0xaa) as usize)
                            .cast::<__m128i>(),
                    ),
                );
                c1_2 = _mm_xor_si128(
                    c1_2,
                    _mm_load_si128(
                        table_base
                            .add((c_row[l2] & 0xaa) as usize)
                            .cast::<__m128i>(),
                    ),
                );
                c1_3 = _mm_xor_si128(
                    c1_3,
                    _mm_load_si128(
                        table_base
                            .add((c_row[l3] & 0xaa) as usize)
                            .cast::<__m128i>(),
                    ),
                );
            }

            mul_karatsuba_acc!(&mut partial_ab[l0], ab0);
            mul_karatsuba_acc!(&mut partial_c_0[l0], c0_0);
            mul_karatsuba_acc!(&mut partial_c_1[l0], c1_0);
            mul_karatsuba_acc!(&mut partial_ab[l1], ab1);
            mul_karatsuba_acc!(&mut partial_c_0[l1], c0_1);
            mul_karatsuba_acc!(&mut partial_c_1[l1], c1_1);
            mul_karatsuba_acc!(&mut partial_ab[l2], ab2);
            mul_karatsuba_acc!(&mut partial_c_0[l2], c0_2);
            mul_karatsuba_acc!(&mut partial_c_1[l2], c1_2);
            mul_karatsuba_acc!(&mut partial_ab[l3], ab3);
            mul_karatsuba_acc!(&mut partial_c_0[l3], c0_3);
            mul_karatsuba_acc!(&mut partial_c_1[l3], c1_3);
        }
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    for lane in 0..64 {
        let mut converted_ab = F128::ZERO;
        let mut converted_c_0 = F128::ZERO;
        let mut converted_c_1 = F128::ZERO;
        for b_med in 0..n_b_med {
            let table_base = b_med * 256;
            let c = chunk_c_bytes[b_med][lane] as u64;
            unsafe {
                let split = crate::gf2_kernels::gf2_kern_split_even_odd_bits_u8(c);
                let c0 = (split as u32) as usize;
                let c1 = ((split >> 32) as u32) as usize;
                converted_ab +=
                    *convert.get_unchecked(table_base + chunk_ab_bytes[b_med][lane] as usize);
                converted_c_0 += *convert.get_unchecked(table_base + c0);
                converted_c_1 += *convert.get_unchecked(table_base + c1);
            }
        }
        partial_ab[lane].ll += converted_ab * eq_lo_val;
        partial_c_0[lane].ll += converted_c_0 * eq_lo_val;
        partial_c_1[lane].ll += converted_c_1 * eq_lo_val;
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn accumulate_convert_batch_with_s_hat_v(
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
    let _ = batch_n_b_med;
    assert!(
        convert.len() >= 16 * 256
            && convert_c_pair.len() >= 8 * 256
            && batch_len <= super::super::URM_BATCH
    );
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let conv_ptr = convert.as_ptr();
        let conv_c_ptr = convert_c_pair.as_ptr();
        let mut conv_buf0 = [[_mm_setzero_si128(); 64]; super::super::URM_BATCH];
        let mut conv_c1 = [[_mm_setzero_si128(); 64]; super::super::URM_BATCH];

        let mut veq_r0 = [_mm_setzero_si128(); super::super::URM_BATCH];
        let mut veq_r1 = [_mm_setzero_si128(); super::super::URM_BATCH];
        for b_idx in 0..batch_len {
            let eq_lo_val = batch_eq_lo[b_idx];
            let veq = _mm_set_epi64x(eq_lo_val.hi as i64, eq_lo_val.lo as i64);
            let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(veq);
            veq_r0[b_idx] = r0;
            veq_r1[b_idx] = r1;
        }

        let _ = conv_ptr;
        accumulate_convert_batch_ab_only(
            batch_ab_bytes,
            batch_n_b_med,
            batch_eq_lo,
            batch_len,
            convert,
            partial_ab,
        );

        for h in 0..2 {
            let m_base = h * 4;
            let r_base = h * 8;
            let tc0 = conv_c_ptr.add(m_base * 256);
            let tc1 = conv_c_ptr.add((m_base + 1) * 256);
            let tc2 = conv_c_ptr.add((m_base + 2) * 256);
            let tc3 = conv_c_ptr.add((m_base + 3) * 256);
            for b_idx in 0..batch_len {
                let c_rows = &batch_c_bytes[b_idx];
                let dst_c0 = &mut conv_buf0[b_idx];
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
                    let mut c0_0 = _mm_load_si128(tc0.add(c0_r0[l0] as usize).cast::<__m128i>());
                    let mut c0_1 = _mm_load_si128(tc0.add(c0_r0[l1] as usize).cast::<__m128i>());
                    let mut c0_2 = _mm_load_si128(tc0.add(c0_r0[l2] as usize).cast::<__m128i>());
                    let mut c0_3 = _mm_load_si128(tc0.add(c0_r0[l3] as usize).cast::<__m128i>());
                    let mut c1_0 = _mm_load_si128(tc0.add(c1_r0[l0] as usize).cast::<__m128i>());
                    let mut c1_1 = _mm_load_si128(tc0.add(c1_r0[l1] as usize).cast::<__m128i>());
                    let mut c1_2 = _mm_load_si128(tc0.add(c1_r0[l2] as usize).cast::<__m128i>());
                    let mut c1_3 = _mm_load_si128(tc0.add(c1_r0[l3] as usize).cast::<__m128i>());

                    c0_0 = _mm_xor_si128(
                        c0_0,
                        _mm_load_si128(tc1.add(c0_r1[l0] as usize).cast::<__m128i>()),
                    );
                    c0_1 = _mm_xor_si128(
                        c0_1,
                        _mm_load_si128(tc1.add(c0_r1[l1] as usize).cast::<__m128i>()),
                    );
                    c0_2 = _mm_xor_si128(
                        c0_2,
                        _mm_load_si128(tc1.add(c0_r1[l2] as usize).cast::<__m128i>()),
                    );
                    c0_3 = _mm_xor_si128(
                        c0_3,
                        _mm_load_si128(tc1.add(c0_r1[l3] as usize).cast::<__m128i>()),
                    );
                    c1_0 = _mm_xor_si128(
                        c1_0,
                        _mm_load_si128(tc1.add(c1_r1[l0] as usize).cast::<__m128i>()),
                    );
                    c1_1 = _mm_xor_si128(
                        c1_1,
                        _mm_load_si128(tc1.add(c1_r1[l1] as usize).cast::<__m128i>()),
                    );
                    c1_2 = _mm_xor_si128(
                        c1_2,
                        _mm_load_si128(tc1.add(c1_r1[l2] as usize).cast::<__m128i>()),
                    );
                    c1_3 = _mm_xor_si128(
                        c1_3,
                        _mm_load_si128(tc1.add(c1_r1[l3] as usize).cast::<__m128i>()),
                    );

                    c0_0 = _mm_xor_si128(
                        c0_0,
                        _mm_load_si128(tc2.add(c0_r2[l0] as usize).cast::<__m128i>()),
                    );
                    c0_1 = _mm_xor_si128(
                        c0_1,
                        _mm_load_si128(tc2.add(c0_r2[l1] as usize).cast::<__m128i>()),
                    );
                    c0_2 = _mm_xor_si128(
                        c0_2,
                        _mm_load_si128(tc2.add(c0_r2[l2] as usize).cast::<__m128i>()),
                    );
                    c0_3 = _mm_xor_si128(
                        c0_3,
                        _mm_load_si128(tc2.add(c0_r2[l3] as usize).cast::<__m128i>()),
                    );
                    c1_0 = _mm_xor_si128(
                        c1_0,
                        _mm_load_si128(tc2.add(c1_r2[l0] as usize).cast::<__m128i>()),
                    );
                    c1_1 = _mm_xor_si128(
                        c1_1,
                        _mm_load_si128(tc2.add(c1_r2[l1] as usize).cast::<__m128i>()),
                    );
                    c1_2 = _mm_xor_si128(
                        c1_2,
                        _mm_load_si128(tc2.add(c1_r2[l2] as usize).cast::<__m128i>()),
                    );
                    c1_3 = _mm_xor_si128(
                        c1_3,
                        _mm_load_si128(tc2.add(c1_r2[l3] as usize).cast::<__m128i>()),
                    );

                    c0_0 = _mm_xor_si128(
                        c0_0,
                        _mm_load_si128(tc3.add(c0_r3[l0] as usize).cast::<__m128i>()),
                    );
                    c0_1 = _mm_xor_si128(
                        c0_1,
                        _mm_load_si128(tc3.add(c0_r3[l1] as usize).cast::<__m128i>()),
                    );
                    c0_2 = _mm_xor_si128(
                        c0_2,
                        _mm_load_si128(tc3.add(c0_r3[l2] as usize).cast::<__m128i>()),
                    );
                    c0_3 = _mm_xor_si128(
                        c0_3,
                        _mm_load_si128(tc3.add(c0_r3[l3] as usize).cast::<__m128i>()),
                    );
                    c1_0 = _mm_xor_si128(
                        c1_0,
                        _mm_load_si128(tc3.add(c1_r3[l0] as usize).cast::<__m128i>()),
                    );
                    c1_1 = _mm_xor_si128(
                        c1_1,
                        _mm_load_si128(tc3.add(c1_r3[l1] as usize).cast::<__m128i>()),
                    );
                    c1_2 = _mm_xor_si128(
                        c1_2,
                        _mm_load_si128(tc3.add(c1_r3[l2] as usize).cast::<__m128i>()),
                    );
                    c1_3 = _mm_xor_si128(
                        c1_3,
                        _mm_load_si128(tc3.add(c1_r3[l3] as usize).cast::<__m128i>()),
                    );

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
                        dst_c0[l0] = _mm_xor_si128(dst_c0[l0], c0_0);
                        dst_c0[l1] = _mm_xor_si128(dst_c0[l1], c0_1);
                        dst_c0[l2] = _mm_xor_si128(dst_c0[l2], c0_2);
                        dst_c0[l3] = _mm_xor_si128(dst_c0[l3], c0_3);
                        dst_c1[l0] = _mm_xor_si128(dst_c1[l0], c1_0);
                        dst_c1[l1] = _mm_xor_si128(dst_c1[l1], c1_1);
                        dst_c1[l2] = _mm_xor_si128(dst_c1[l2], c1_2);
                        dst_c1[l3] = _mm_xor_si128(dst_c1[l3], c1_3);
                    }
                }
            }
        }

        for b_idx in 0..batch_len {
            let r0 = veq_r0[b_idx];
            let r1 = veq_r1[b_idx];
            let src_c0 = &conv_buf0[b_idx];
            let src_c1 = &conv_c1[b_idx];
            for l in 0..64 {
                let (p_c0, p_c1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    src_c0[l], src_c1[l], r0, r1, r0, r1,
                );
                let ptr_c0 = (&mut partial_c_0[l].ll as *mut F128).cast::<__m128i>();
                let ptr_c1 = (&mut partial_c_1[l].ll as *mut F128).cast::<__m128i>();
                _mm_store_si128(ptr_c0, _mm_xor_si128(_mm_load_si128(ptr_c0), p_c0));
                _mm_store_si128(ptr_c1, _mm_xor_si128(_mm_load_si128(ptr_c1), p_c1));
            }
        }
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    for b_idx in 0..batch_len {
        let eq_lo_val = batch_eq_lo[b_idx];
        let ab_rows = &batch_ab_bytes[b_idx];
        let c_rows = &batch_c_bytes[b_idx];
        for lane in 0..64 {
            let mut converted_ab = F128::ZERO;
            let mut converted_c_0 = F128::ZERO;
            let mut converted_c_1 = F128::ZERO;
            for b_med in 0..16 {
                converted_ab += convert[b_med * 256 + ab_rows[b_med][lane] as usize];
            }
            for m in 0..8 {
                converted_c_0 += convert_c_pair[m * 256 + c_rows[2 * m][lane] as usize];
                converted_c_1 += convert_c_pair[m * 256 + c_rows[2 * m + 1][lane] as usize];
            }
            partial_ab[lane].ll += converted_ab * eq_lo_val;
            partial_c_0[lane].ll += converted_c_0 * eq_lo_val;
            partial_c_1[lane].ll += converted_c_1 * eq_lo_val;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn accumulate_convert_batch_ab_only(
    batch_ab_bytes: &[[[u8; 64]; 16]; super::super::URM_BATCH],
    batch_n_b_med: &[u8; super::super::URM_BATCH],
    batch_eq_lo: &[F128; super::super::URM_BATCH],
    batch_len: usize,
    convert: &[F128],
    partial_ab: &mut [super::super::KaratsubaAcc; 64],
) {
    assert!(convert.len() >= 16 * 256 && batch_len <= super::super::URM_BATCH);
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

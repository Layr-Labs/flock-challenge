// GENERATED from the static-B plans of aarch64_bstatic_gen.rs (ag-ab2 gen_v3.py).
//
// The static-B plans of [`shift_reduce_inner_ab_bstatic`] (same masks, same
// partials, same varying bytes) evaluated in the Horner layout of
// [`shift_reduce_inner_ab_fused_neon_h4`]. Each row checks all of its static
// K-words once, up front; the body is then straight-line (no per-K branch),
// so the scheduler and register allocator see one basic block exactly like
// the generic h4 kernel. A row whose static bytes do not all match runs the
// generic h4 kernel instead (cold), so the output is bit-identical for any
// witness. Each live arm is its own monomorphized body; a window's static
// rows can run as one unrolled sequence of direct calls (no per-row jump).
//
// Included by kernels/aarch64.rs via include!.

#[cfg(target_arch = "aarch64")]
#[cold]
#[inline(never)]
fn bstatic_horner_fallback(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    shift_reduce_inner_ab_fused_neon_h4(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
}

/// One static-B arm `BLK = w * 16 + b_med` (3..=28) in the h4 Horner layout.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(clippy::too_many_lines)]
fn bstatic_horner_blk<const BLK: usize>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: &'static [[u8; 64]; 248],
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;
    let b_med = BLK % 16;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    assert!(byte_base_b + 8 * N_CHUNKS <= a_packed.len() && byte_base_b + 8 * N_CHUNKS <= b_packed.len());
    let plain = inv_table.data_ptr();
    let plain_sw = inv_table.half_swapped_data_ptr();
    let scaled = inv_table.scaled_x4_data_ptr();
    let scaled_sw = inv_table.scaled_x4_half_swapped_data_ptr();
    unsafe {
        let a_base = a_packed.as_ptr().add(byte_base_b);
        let b_base = b_packed.as_ptr().add(byte_base_b);
        let pbase = partials.as_ptr().cast::<u8>();
        macro_rules! bw {
            ($k:literal) => {
                u64::from_le(core::ptr::read_unaligned(b_base.add($k * N_CHUNKS).cast::<u64>()))
            };
        }
        macro_rules! bimg_g {
            ($k:literal) => {
                apply_word_into_4_regs(plain, plain_sw, bw!($k))
            };
        }
        macro_rules! bimg_s {
            ($blk:literal, $k:literal, [$(($j0:literal, $j1:literal)),*], [$($js:literal),*]) => {{
                let pp = pbase.add(($blk * 8 + $k) * 64);
                #[allow(unused_mut)]
                let mut db0 = vld1q_u8(pp);
                #[allow(unused_mut)]
                let mut db1 = vld1q_u8(pp.add(16));
                #[allow(unused_mut)]
                let mut db2 = vld1q_u8(pp.add(32));
                #[allow(unused_mut)]
                let mut db3 = vld1q_u8(pp.add(48));
                #[allow(unused_variables)]
                let b_word = bw!($k);
                $(xor_vary_pair::<$j0, $j1>(plain, plain_sw, b_word, &mut db0, &mut db1, &mut db2, &mut db3);)*
                $(xor_vary_single::<$js>(plain, plain_sw, b_word, &mut db0, &mut db1, &mut db2, &mut db3);)*
                (db0, db1, db2, db3)
            }};
        }
        // Raw 15-bit products of one K row, in the h4 order (a then b).
        macro_rules! prods {
            ($at:expr, $at_sw:expr, $k:literal, $bimg:expr) => {{
                let aw = u64::from_le(core::ptr::read_unaligned(a_base.add($k * N_CHUNKS).cast::<u64>()));
                let (da0, da1, da2, da3) = apply_word_into_4_regs($at, $at_sw, aw);
                let (db0, db1, db2, db3) = $bimg;
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
        macro_rules! seed {
            ($lo:ident, $hi:ident) => {
                [
                    veorq_u16($lo[0], $hi[0]),
                    veorq_u16($lo[1], $hi[1]),
                    veorq_u16($lo[2], $hi[2]),
                    veorq_u16($lo[3], $hi[3]),
                    veorq_u16($lo[4], $hi[4]),
                    veorq_u16($lo[5], $hi[5]),
                    veorq_u16($lo[6], $hi[6]),
                    veorq_u16($lo[7], $hi[7]),
                ]
            };
        }
        let carry_not = vreinterpretq_u8_u16(vdupq_n_u16(!HORNER_CARRY_X16));
        let acc = match BLK {
            3 => {
                if ((!bw!(1) & 0x0000ffffffffffff) | (!bw!(4) & 0xffff000000000000) | (!bw!(5) & 0x000000ffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(3, 1, [(6, 7)], []));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(3, 5, [(5, 6)], [7]));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(3, 4, [(0, 1), (2, 3), (4, 5)], []));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            4 => {
                if ((!bw!(0) & 0xffff000000000000) | (!bw!(1) & 0x000000ffffffffff) | (!bw!(4) & 0xffffff0000000000) | (!bw!(5) & 0x00000000ffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(4, 1, [(5, 6)], [7]));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(4, 5, [(4, 5), (6, 7)], []));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(4, 0, [(0, 1), (2, 3), (4, 5)], []));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(4, 4, [(0, 1), (2, 3)], [4]));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            5 => {
                if ((!bw!(0) & 0xffffffff00000000) | (!bw!(1) & 0x0000000000ffffff) | (!bw!(4) & 0xffffffffff000000) | (!bw!(5) & 0x000000000000ffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(5, 1, [(3, 4), (5, 6)], [7]));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(5, 5, [(2, 3), (4, 5), (6, 7)], []));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(5, 0, [(0, 1), (2, 3)], []));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(5, 4, [(0, 1)], [2]));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            6 => {
                if ((!bw!(0) & 0xffffffffff000000) | (!bw!(1) & 0x000000000000ffff) | (!bw!(4) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(6, 1, [(2, 3), (4, 5), (6, 7)], []));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(6, 0, [(0, 1)], [2]));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(6, 4, [(0, 1)], []));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            7 => {
                if ((!bw!(0) & 0xffffffffffffff00) | (!bw!(4) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(7, 0, [], [0]));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(7, 4, [], [7]));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            8 => {
                if ((!bw!(0) & 0x00ffffffffffffff) | (!bw!(4) & 0x0000ffffffffffff) | (!bw!(7) & 0xffff000000000000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_s!(8, 7, [(0, 1), (2, 3), (4, 5)], []));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(8, 0, [], [7]));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(8, 4, [(6, 7)], []));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            9 => {
                if ((!bw!(0) & 0x000000ffffffffff) | (!bw!(3) & 0xffffff0000000000) | (!bw!(4) & 0x00000000ffffffff) | (!bw!(7) & 0xffffff0000000000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_s!(9, 3, [(0, 1), (2, 3)], [4]));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_s!(9, 7, [(0, 1), (2, 3)], [4]));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(9, 0, [(5, 6)], [7]));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(9, 4, [(4, 5), (6, 7)], []));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            10 => {
                if ((!bw!(0) & 0x00000000ffffffff) | (!bw!(3) & 0xffffffff00000000) | (!bw!(4) & 0x0000000000ffffff) | (!bw!(7) & 0xffffffffff000000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_s!(10, 3, [(0, 1), (2, 3)], []));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_s!(10, 7, [(0, 1)], [2]));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(10, 0, [(4, 5), (6, 7)], []));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(10, 4, [(3, 4), (5, 6)], [7]));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            11 => {
                if ((!bw!(0) & 0x000000000000ffff) | (!bw!(3) & 0xffffffffffff0000) | (!bw!(7) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_s!(11, 3, [(0, 1)], []));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_s!(11, 7, [(0, 1)], []));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(11, 0, [(2, 3), (4, 5), (6, 7)], []));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            12 => {
                if ((!bw!(3) & 0xffffffffffffff00) | (!bw!(7) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_s!(12, 3, [], [0]));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_s!(12, 7, [], [7]));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            13 => {
                if ((!bw!(3) & 0x0000ffffffffffff) | (!bw!(7) & 0x0000ffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_s!(13, 3, [(6, 7)], []));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_s!(13, 7, [(6, 7)], []));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            14 => {
                if ((!bw!(2) & 0xffff000000000000) | (!bw!(3) & 0x000000ffffffffff) | (!bw!(6) & 0xffffff0000000000) | (!bw!(7) & 0x00000000ffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_s!(14, 3, [(5, 6)], [7]));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_s!(14, 7, [(4, 5), (6, 7)], []));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_s!(14, 2, [(0, 1), (2, 3), (4, 5)], []));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_s!(14, 6, [(0, 1), (2, 3)], [4]));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            15 => {
                if ((!bw!(2) & 0xffffffff00000000) | (!bw!(3) & 0x0000000000ffffff) | (!bw!(6) & 0xffffffff00000000) | (!bw!(7) & 0x0000000000ffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_s!(15, 3, [(3, 4), (5, 6)], [7]));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_s!(15, 7, [(3, 4), (5, 6)], [7]));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_s!(15, 2, [(0, 1), (2, 3)], []));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_s!(15, 6, [(0, 1), (2, 3)], []));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            16 => {
                if ((!bw!(2) & 0xffffffffff000000) | (!bw!(3) & 0x000000000000ffff) | (!bw!(6) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_s!(16, 3, [(2, 3), (4, 5), (6, 7)], []));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_s!(16, 2, [(0, 1)], [2]));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_s!(16, 6, [(0, 1)], []));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            17 => {
                if ((!bw!(2) & 0xffffffffffffff00) | (!bw!(6) & 0xffffffffffffff00)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_s!(17, 2, [], [0]));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_s!(17, 6, [], [0]));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            18 => {
                if ((!bw!(2) & 0x00ffffffffffffff) | (!bw!(6) & 0x0000ffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_s!(18, 2, [], [7]));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_s!(18, 6, [(6, 7)], []));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            19 => {
                if ((!bw!(1) & 0xffff000000000000) | (!bw!(2) & 0x000000ffffffffff) | (!bw!(5) & 0xffff000000000000) | (!bw!(6) & 0x000000ffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_s!(19, 2, [(5, 6)], [7]));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_s!(19, 6, [(5, 6)], [7]));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(19, 1, [(0, 1), (2, 3), (4, 5)], []));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(19, 5, [(0, 1), (2, 3), (4, 5)], []));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            20 => {
                if ((!bw!(1) & 0xffffff0000000000) | (!bw!(2) & 0x00000000ffffffff) | (!bw!(5) & 0xffffffff00000000) | (!bw!(6) & 0x0000000000ffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_s!(20, 2, [(4, 5), (6, 7)], []));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_s!(20, 6, [(3, 4), (5, 6)], [7]));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(20, 1, [(0, 1), (2, 3)], [4]));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(20, 5, [(0, 1), (2, 3)], []));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            21 => {
                if ((!bw!(1) & 0xffffffffff000000) | (!bw!(2) & 0x000000000000ffff) | (!bw!(5) & 0xffffffffff000000) | (!bw!(6) & 0x000000000000ffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_s!(21, 2, [(2, 3), (4, 5), (6, 7)], []));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_s!(21, 6, [(2, 3), (4, 5), (6, 7)], []));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(21, 1, [(0, 1)], [2]));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(21, 5, [(0, 1)], [2]));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            22 => {
                if ((!bw!(1) & 0xffffffffffff0000) | (!bw!(5) & 0xffffffffffffff00)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(22, 1, [(0, 1)], []));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(22, 5, [], [0]));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            23 => {
                if ((!bw!(1) & 0x00ffffffffffffff) | (!bw!(5) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(23, 1, [], [7]));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(23, 5, [], [7]));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_g!(4));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            24 => {
                if ((!bw!(1) & 0x0000ffffffffffff) | (!bw!(4) & 0xffff000000000000) | (!bw!(5) & 0x000000ffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(24, 1, [(6, 7)], []));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(24, 5, [(5, 6)], [7]));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_g!(0));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(24, 4, [(0, 1), (2, 3), (4, 5)], []));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            25 => {
                if ((!bw!(0) & 0xffffff0000000000) | (!bw!(1) & 0x00000000ffffffff) | (!bw!(4) & 0xffffff0000000000) | (!bw!(5) & 0x00000000ffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(25, 1, [(4, 5), (6, 7)], []));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(25, 5, [(4, 5), (6, 7)], []));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(25, 0, [(0, 1), (2, 3)], [4]));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(25, 4, [(0, 1), (2, 3)], [4]));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            26 => {
                if ((!bw!(0) & 0xffffffff00000000) | (!bw!(1) & 0x0000000000ffffff) | (!bw!(4) & 0xffffffffff000000) | (!bw!(5) & 0x000000000000ffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_s!(26, 1, [(3, 4), (5, 6)], [7]));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_s!(26, 5, [(2, 3), (4, 5), (6, 7)], []));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(26, 0, [(0, 1), (2, 3)], []));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(26, 4, [(0, 1)], [2]));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            27 => {
                if ((!bw!(0) & 0xffffffffffff0000) | (!bw!(4) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(27, 0, [(0, 1)], []));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(27, 4, [(0, 1)], []));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            28 => {
                if ((!bw!(0) & 0xffffffffffffff00) | (!bw!(4) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let lo3 = prods!(plain, plain_sw, 3, bimg_g!(3));
                let hi3 = prods!(scaled, scaled_sw, 7, bimg_g!(7));
                let mut acc = seed!(lo3, hi3);
                let lo2 = prods!(plain, plain_sw, 2, bimg_g!(2));
                let hi2 = prods!(scaled, scaled_sw, 6, bimg_g!(6));
                horner_absorb(&mut acc, &lo2, &hi2, carry_not);
                let lo1 = prods!(plain, plain_sw, 1, bimg_g!(1));
                let hi1 = prods!(scaled, scaled_sw, 5, bimg_g!(5));
                horner_absorb(&mut acc, &lo1, &hi1, carry_not);
                let lo0 = prods!(plain, plain_sw, 0, bimg_s!(28, 0, [], [0]));
                let hi0 = prods!(scaled, scaled_sw, 4, bimg_s!(28, 4, [], [7]));
                horner_absorb(&mut acc, &lo0, &hi0, carry_not);
                acc
            }
            _ => unreachable!("no static-B Horner plan for this arm"),
        };
        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[0]), vreinterpretq_u8_u16(acc[1]));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[2]), vreinterpretq_u8_u16(acc[3]));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[4]), vreinterpretq_u8_u16(acc[5]));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[6]), vreinterpretq_u8_u16(acc[7]));
        store_row_64(out.as_mut_ptr(), nt_store, r0, r1, r2, r3);
    }
}

/// Per-row entry: runs the Horner static-B body of arm `(w, b_med)` and
/// returns `true`, or returns `false` when the arm has no plan (the caller
/// then runs another kernel).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn shift_reduce_inner_ab_bstatic_horner(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    w: usize,
    partials: &'static [[u8; 64]; 248],
    out: &mut [u8; 64],
    nt_store: bool,
) -> bool {
    if w > 1 || b_med >= 16 {
        return false;
    }
    match w * 16 + b_med {
        3 => bstatic_horner_blk::<3>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        4 => bstatic_horner_blk::<4>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        5 => bstatic_horner_blk::<5>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        6 => bstatic_horner_blk::<6>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        7 => bstatic_horner_blk::<7>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        8 => bstatic_horner_blk::<8>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        9 => bstatic_horner_blk::<9>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        10 => bstatic_horner_blk::<10>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        11 => bstatic_horner_blk::<11>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        12 => bstatic_horner_blk::<12>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        13 => bstatic_horner_blk::<13>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        14 => bstatic_horner_blk::<14>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        15 => bstatic_horner_blk::<15>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        16 => bstatic_horner_blk::<16>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        17 => bstatic_horner_blk::<17>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        18 => bstatic_horner_blk::<18>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        19 => bstatic_horner_blk::<19>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        20 => bstatic_horner_blk::<20>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        21 => bstatic_horner_blk::<21>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        22 => bstatic_horner_blk::<22>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        23 => bstatic_horner_blk::<23>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        24 => bstatic_horner_blk::<24>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        25 => bstatic_horner_blk::<25>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        26 => bstatic_horner_blk::<26>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        27 => bstatic_horner_blk::<27>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        28 => bstatic_horner_blk::<28>(a_packed, b_packed, inv_table, chunk_byte_base, partials, out, nt_store),
        _ => return false,
    }
    true
}

/// First and one-past-last `b_med` of window `w`'s static-B Horner rows,
/// i.e. the rows [`bstatic_horner_window_rows`] writes.
#[cfg(target_arch = "aarch64")]
pub(crate) const BSTATIC_HORNER_WINDOW_ROWS: [(usize, usize); 2] = [(3, 16), (0, 13)];

/// All static-B Horner rows of window `w` (see [`BSTATIC_HORNER_WINDOW_ROWS`])
/// as one unrolled sequence of direct calls into the per-arm bodies, writing
/// row `b_med` to `out_outer[b_med * 64..][..64]`.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bstatic_horner_window_rows(
    w: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: &'static [[u8; 64]; 248],
    out_outer: &mut [u8],
    nt_store: bool,
) {
    assert!(w <= 1 && out_outer.len() >= 16 * 64);
    let out = out_outer.as_mut_ptr();
    macro_rules! rows {
        ($($blk:literal),*) => {{
            $(
                // SAFETY: `$blk % 16 < 16` and `out_outer` holds 16 rows of 64 bytes.
                let dst = unsafe { &mut *out.add(($blk % 16) * 64).cast::<[u8; 64]>() };
                bstatic_horner_blk::<$blk>(a_packed, b_packed, inv_table, chunk_byte_base, partials, dst, nt_store);
            )*
        }};
    }
    if w == 0 {
        rows!(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    } else {
        rows!(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28);
    }
}

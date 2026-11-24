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
// `V` selects the form of a body at compile time (`ABV_KARA`: the table
// apply in the Karatsuba form, 18 loads per word instead of 32;
// `ABV_INLINE_WINDOW`: a window's rows as one function body). `V = 0` is the
// original form. All forms write the same bytes.
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
    shift_reduce_inner_ab_fused_neon_h4::<0>(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
}

/// One static-B arm `BLK = w * 16 + b_med` (3..=28) in the h4 Horner layout.
#[cfg(all(test, target_arch = "aarch64"))]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn bstatic_horner_blk<const BLK: usize, const V: u32>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: &'static [[u8; 64]; 248],
    kara: *const u8,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    bstatic_horner_blk_body::<BLK, V, false>(
        a_packed, b_packed, inv_table, chunk_byte_base, partials, kara, out, nt_store,
    );
}

/// Body of [`bstatic_horner_blk`]. `kara` is [`ab_kara_tables`] when `V`
/// has `ABV_KARA` and is not dereferenced otherwise.
///
/// `HOIST = true` (window rows only, see [`bstatic_horner_window_rows`]):
/// the caller has already checked that the row's eight K words lie inside
/// both slices (one check for the window's largest `b_med`), so the row
/// does not repeat it. `HOIST = false`: the row checks its own bounds.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn bstatic_horner_blk_body<const BLK: usize, const V: u32, const HOIST: bool>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: &'static [[u8; 64]; 248],
    kara: *const u8,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;
    // The fused accumulation implements the `ABV_HORNER2` steps only.
    const { assert!(V & ABV_FUSE == 0 || V & ABV_HORNER2 != 0) };
    // The sparse scale reads the complemented static-B words (the trim) and
    // is written for the fused accumulation only.
    const { assert!(V & ABV_SPARSE_SCALE == 0 || (V & ABV_TRANSFORM_TRIM != 0 && V & ABV_FUSE != 0 && V & ABV_KARA != 0)) };
    let b_med = BLK % 16;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    if HOIST {
        debug_assert!(byte_base_b + 8 * N_CHUNKS <= a_packed.len() && byte_base_b + 8 * N_CHUNKS <= b_packed.len());
    } else {
        assert!(byte_base_b + 8 * N_CHUNKS <= a_packed.len() && byte_base_b + 8 * N_CHUNKS <= b_packed.len());
    }
    let plain = inv_table.data_ptr();
    let plain_sw = inv_table.half_swapped_data_ptr();
    let scaled = inv_table.scaled_x4_data_ptr();
    let scaled_sw = inv_table.scaled_x4_half_swapped_data_ptr();
    // Karatsuba images of the plain and the scaled table (dangling and
    // unused unless `V` applies the table in that form).
    let kplain = kara;
    let kscaled = kara.wrapping_add(AB_KARA_IMAGE_BYTES);
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
                apply_word_into_4_regs_v::<V, false>(plain, plain_sw, kplain, bw!($k))
            };
        }
        macro_rules! bimg_s {
            ($blk:literal, $k:literal, [$(($j0:literal, $j1:literal)),*], [$($js:literal),*]) => {{
                let pp = pbase.add(($blk * 8 + $k) * 64);
                #[allow(unused_mut)]
                let (mut db0, mut db1, mut db2, mut db3) = transform_partial_seed::<V>(pp);
                #[allow(unused_variables)]
                let b_word = transform_partial_word::<V>(bw!($k));
                if V & ABV_NOSW != 0 {
                    $(xor_vary_ext::<$j0, $j1>(plain, b_word, [&mut db0, &mut db1, &mut db2, &mut db3]);)*
                    $(xor_vary_ext::<$js, 8>(plain, b_word, [&mut db0, &mut db1, &mut db2, &mut db3]);)*
                } else {
                    $(xor_vary_pair::<$j0, $j1>(plain, plain_sw, b_word, &mut db0, &mut db1, &mut db2, &mut db3);)*
                    $(xor_vary_single::<$js>(plain, plain_sw, b_word, &mut db0, &mut db1, &mut db2, &mut db3);)*
                }
                (db0, db1, db2, db3)
            }};
        }
        // A static b word with four or more varying bytes: 4 partial loads
        // plus 4 per varying byte is at least 20, the whole-word Karatsuba
        // apply is 18. The row's guard established the static bytes, so the
        // whole-word image is the partial image plus the varying rows.
        macro_rules! bimg_sk {
            ($blk:literal, $k:literal, $pairs:tt, $singles:tt) => {{
                if V & ABV_KARA != 0 {
                    apply_word_into_4_regs_v::<V, false>(plain, plain_sw, kplain, bw!($k))
                } else {
                    bimg_s!($blk, $k, $pairs, $singles)
                }
            }};
        }
        // A static b word whose varying bytes are exactly bytes 0..NB: under
        // `ABV_BSTATIC_Z` its partial image plus the varying bytes' z-form
        // image (with `[1; 64]` absorbed directly under `ABV_TRANSFORM_TRIM`);
        // otherwise the incumbent form `$inc`.
        macro_rules! bimg_lz {
            ($blk:literal, $k:literal, $nb:literal, $inc:expr) => {{
                if V & ABV_BSTATIC_Z != 0 {
                    if V & ABV_TRANSFORM_TRIM != 0 {
                        apply_word_z_low::<$nb, true>(plain, kplain, !bw!($k))
                    } else {
                        let pp = pbase.add(($blk * 8 + $k) * 64);
                        let (z0, z1, z2, z3) = apply_word_z_low::<$nb, false>(
                            plain, kplain, bw!($k),
                        );
                        let (d0, d1, d2, d3) = transform_partial_seed::<V>(pp);
                        (
                            veorq_u8(d0, z0),
                            veorq_u8(d1, z1),
                            veorq_u8(d2, z2),
                            veorq_u8(d3, z3),
                        )
                    }
                } else {
                    $inc
                }
            }};
        }
        // A static b word whose varying bytes are exactly bytes 8-NB..8: under
        // `ABV_BSTATIC_Z | ABV_TRANSFORM_TRIM` its high-end z-form image with
        // `[1; 64]` absorbed; otherwise the incumbent form `$inc`.
        #[allow(unused_macros)]
        macro_rules! bimg_hz {
            ($blk:literal, $k:literal, $nb:literal, $inc:expr) => {{
                if V & ABV_BSTATIC_Z != 0 && V & ABV_TRANSFORM_TRIM != 0 {
                    apply_word_z_high::<$nb, true>(plain, kplain, !bw!($k))
                } else {
                    $inc
                }
            }};
        }
        // Raw 15-bit products of one K row, in the h4 order (a then b).
        macro_rules! prods {
            ($at:expr, $at_sw:expr, $kara:expr, $k:literal, $bimg:expr) => {{
                let aw = u64::from_le(core::ptr::read_unaligned(a_base.add($k * N_CHUNKS).cast::<u64>()));
                let (da0, da1, da2, da3) = apply_word_into_4_regs_v::<V, { $k >= 4 }>($at, $at_sw, $kara, aw);
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
        // The operands of one K row (the A image of `$k`, then the B image),
        // for the `ABV_FUSE` accumulation: the same applies as `prods!`.
        macro_rules! opnds {
            ($at:expr, $at_sw:expr, $kara:expr, $k:literal, $bimg:expr) => {{
                let aw = u64::from_le(core::ptr::read_unaligned(a_base.add($k * N_CHUNKS).cast::<u64>()));
                let (da0, da1, da2, da3) = apply_word_into_4_regs_v::<V, { $k >= 4 }>($at, $at_sw, $kara, aw);
                let (db0, db1, db2, db3) = $bimg;
                ([da0, da1, da2, da3], [db0, db1, db2, db3])
            }};
        }
        // [`ABV_SPARSE_SCALE`]: the operands of the row's one `x^4` word whose
        // B varies only in `$nb` bytes at one end (`$high`): the A word out
        // of the plain image, the B word out of the `x^4` image restricted to
        // its varying bytes. The products are congruent mod p to those of
        // `opnds!` with the scaled A image.
        #[allow(unused_macros)]
        macro_rules! opnds_ss {
            ($k:literal, $nb:literal, $high:literal) => {{
                let aw = u64::from_le(core::ptr::read_unaligned(a_base.add($k * N_CHUNKS).cast::<u64>()));
                let (da0, da1, da2, da3) = apply_word_into_4_regs_v::<V, false>(plain, plain_sw, kplain, aw);
                let (db0, db1, db2, db3) = apply_word_sparse_x4::<$nb, $high>(kscaled, !bw!($k));
                ([da0, da1, da2, da3], [db0, db1, db2, db3])
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
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_s!(3, 1, [(6, 7)], []));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_s!(3, 5, [(5, 6)], [7]));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_sk!(3, 4, [(0, 1), (2, 3), (4, 5)], []));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_hz!(3, 1, 2, bimg_s!(3, 1, [(6, 7)], [])));
                    let o5 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(5, 3, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 5, bimg_hz!(3, 5, 3, bimg_s!(3, 5, [(5, 6)], [7])))
                    };
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_sk!(3, 4, [(0, 1), (2, 3), (4, 5)], []));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            4 => {
                if ((!bw!(0) & 0xffff000000000000) | (!bw!(1) & 0x000000ffffffffff) | (!bw!(4) & 0xffffff0000000000) | (!bw!(5) & 0x00000000ffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_s!(4, 1, [(5, 6)], [7]));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_sk!(4, 5, [(4, 5), (6, 7)], []));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_sk!(4, 0, [(0, 1), (2, 3), (4, 5)], []));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_sk!(4, 4, [(0, 1), (2, 3)], [4]));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_hz!(4, 1, 3, bimg_s!(4, 1, [(5, 6)], [7])));
                    let o5 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(5, 4, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 5, bimg_hz!(4, 5, 4, bimg_sk!(4, 5, [(4, 5), (6, 7)], [])))
                    };
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_sk!(4, 0, [(0, 1), (2, 3), (4, 5)], []));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_sk!(4, 4, [(0, 1), (2, 3)], [4]));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            5 => {
                if ((!bw!(0) & 0xffffffff00000000) | (!bw!(1) & 0x0000000000ffffff) | (!bw!(4) & 0xffffffffff000000) | (!bw!(5) & 0x000000000000ffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_sk!(5, 1, [(3, 4), (5, 6)], [7]));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_sk!(5, 5, [(2, 3), (4, 5), (6, 7)], []));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_lz!(5, 0, 4, bimg_sk!(5, 0, [(0, 1), (2, 3)], [])));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_lz!(5, 4, 3, bimg_s!(5, 4, [(0, 1)], [2])));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_sk!(5, 1, [(3, 4), (5, 6)], [7]));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_sk!(5, 5, [(2, 3), (4, 5), (6, 7)], []));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_lz!(5, 0, 4, bimg_sk!(5, 0, [(0, 1), (2, 3)], [])));
                    let o4 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(4, 3, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 4, bimg_lz!(5, 4, 3, bimg_s!(5, 4, [(0, 1)], [2])))
                    };
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            6 => {
                if ((!bw!(0) & 0xffffffffff000000) | (!bw!(1) & 0x000000000000ffff) | (!bw!(4) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_sk!(6, 1, [(2, 3), (4, 5), (6, 7)], []));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_lz!(6, 0, 3, bimg_s!(6, 0, [(0, 1)], [2])));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_lz!(6, 4, 2, bimg_s!(6, 4, [(0, 1)], [])));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_sk!(6, 1, [(2, 3), (4, 5), (6, 7)], []));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_lz!(6, 0, 3, bimg_s!(6, 0, [(0, 1)], [2])));
                    let o4 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(4, 2, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 4, bimg_lz!(6, 4, 2, bimg_s!(6, 4, [(0, 1)], [])))
                    };
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            7 => {
                if ((!bw!(0) & 0xffffffffffffff00) | (!bw!(4) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_s!(7, 0, [], [0]));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_s!(7, 4, [], [7]));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_lz!(7, 0, 1, bimg_s!(7, 0, [], [0])));
                    let o4 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(4, 1, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 4, bimg_s!(7, 4, [], [7]))
                    };
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            8 => {
                if ((!bw!(0) & 0x00ffffffffffffff) | (!bw!(4) & 0x0000ffffffffffff) | (!bw!(7) & 0xffff000000000000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_sk!(8, 7, [(0, 1), (2, 3), (4, 5)], []));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_s!(8, 0, [], [7]));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_s!(8, 4, [(6, 7)], []));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_sk!(8, 7, [(0, 1), (2, 3), (4, 5)], []));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_s!(8, 0, [], [7]));
                    let o4 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(4, 2, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 4, bimg_hz!(8, 4, 2, bimg_s!(8, 4, [(6, 7)], [])))
                    };
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            9 => {
                if ((!bw!(0) & 0x000000ffffffffff) | (!bw!(3) & 0xffffff0000000000) | (!bw!(4) & 0x00000000ffffffff) | (!bw!(7) & 0xffffff0000000000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_sk!(9, 3, [(0, 1), (2, 3)], [4]));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_sk!(9, 7, [(0, 1), (2, 3)], [4]));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_s!(9, 0, [(5, 6)], [7]));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_sk!(9, 4, [(4, 5), (6, 7)], []));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_sk!(9, 3, [(0, 1), (2, 3)], [4]));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_sk!(9, 7, [(0, 1), (2, 3)], [4]));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_hz!(9, 0, 3, bimg_s!(9, 0, [(5, 6)], [7])));
                    let o4 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(4, 4, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 4, bimg_hz!(9, 4, 4, bimg_sk!(9, 4, [(4, 5), (6, 7)], [])))
                    };
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            10 => {
                if ((!bw!(0) & 0x00000000ffffffff) | (!bw!(3) & 0xffffffff00000000) | (!bw!(4) & 0x0000000000ffffff) | (!bw!(7) & 0xffffffffff000000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_lz!(10, 3, 4, bimg_sk!(10, 3, [(0, 1), (2, 3)], [])));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_lz!(10, 7, 3, bimg_s!(10, 7, [(0, 1)], [2])));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_sk!(10, 0, [(4, 5), (6, 7)], []));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_sk!(10, 4, [(3, 4), (5, 6)], [7]));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_lz!(10, 3, 4, bimg_sk!(10, 3, [(0, 1), (2, 3)], [])));
                    let o7 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(7, 3, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 7, bimg_lz!(10, 7, 3, bimg_s!(10, 7, [(0, 1)], [2])))
                    };
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_hz!(10, 0, 4, bimg_sk!(10, 0, [(4, 5), (6, 7)], [])));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_sk!(10, 4, [(3, 4), (5, 6)], [7]));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            11 => {
                if ((!bw!(0) & 0x000000000000ffff) | (!bw!(3) & 0xffffffffffff0000) | (!bw!(7) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_lz!(11, 3, 2, bimg_s!(11, 3, [(0, 1)], [])));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_lz!(11, 7, 2, bimg_s!(11, 7, [(0, 1)], [])));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_sk!(11, 0, [(2, 3), (4, 5), (6, 7)], []));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_lz!(11, 3, 2, bimg_s!(11, 3, [(0, 1)], [])));
                    let o7 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(7, 2, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 7, bimg_lz!(11, 7, 2, bimg_s!(11, 7, [(0, 1)], [])))
                    };
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_sk!(11, 0, [(2, 3), (4, 5), (6, 7)], []));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            12 => {
                if ((!bw!(3) & 0xffffffffffffff00) | (!bw!(7) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_s!(12, 3, [], [0]));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_s!(12, 7, [], [7]));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_lz!(12, 3, 1, bimg_s!(12, 3, [], [0])));
                    let o7 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(7, 1, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 7, bimg_s!(12, 7, [], [7]))
                    };
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            13 => {
                if ((!bw!(3) & 0x0000ffffffffffff) | (!bw!(7) & 0x0000ffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_s!(13, 3, [(6, 7)], []));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_s!(13, 7, [(6, 7)], []));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_hz!(13, 3, 2, bimg_s!(13, 3, [(6, 7)], [])));
                    let o7 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(7, 2, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 7, bimg_hz!(13, 7, 2, bimg_s!(13, 7, [(6, 7)], [])))
                    };
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            14 => {
                if ((!bw!(2) & 0xffff000000000000) | (!bw!(3) & 0x000000ffffffffff) | (!bw!(6) & 0xffffff0000000000) | (!bw!(7) & 0x00000000ffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_s!(14, 3, [(5, 6)], [7]));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_sk!(14, 7, [(4, 5), (6, 7)], []));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_sk!(14, 2, [(0, 1), (2, 3), (4, 5)], []));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_sk!(14, 6, [(0, 1), (2, 3)], [4]));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_hz!(14, 3, 3, bimg_s!(14, 3, [(5, 6)], [7])));
                    let o7 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(7, 4, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 7, bimg_hz!(14, 7, 4, bimg_sk!(14, 7, [(4, 5), (6, 7)], [])))
                    };
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_sk!(14, 2, [(0, 1), (2, 3), (4, 5)], []));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_sk!(14, 6, [(0, 1), (2, 3)], [4]));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            15 => {
                if ((!bw!(2) & 0xffffffff00000000) | (!bw!(3) & 0x0000000000ffffff) | (!bw!(6) & 0xffffffff00000000) | (!bw!(7) & 0x0000000000ffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_sk!(15, 3, [(3, 4), (5, 6)], [7]));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_sk!(15, 7, [(3, 4), (5, 6)], [7]));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_lz!(15, 2, 4, bimg_sk!(15, 2, [(0, 1), (2, 3)], [])));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_lz!(15, 6, 4, bimg_sk!(15, 6, [(0, 1), (2, 3)], [])));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_sk!(15, 3, [(3, 4), (5, 6)], [7]));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_sk!(15, 7, [(3, 4), (5, 6)], [7]));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_lz!(15, 2, 4, bimg_sk!(15, 2, [(0, 1), (2, 3)], [])));
                    let o6 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(6, 4, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 6, bimg_lz!(15, 6, 4, bimg_sk!(15, 6, [(0, 1), (2, 3)], [])))
                    };
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            16 => {
                if ((!bw!(2) & 0xffffffffff000000) | (!bw!(3) & 0x000000000000ffff) | (!bw!(6) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_sk!(16, 3, [(2, 3), (4, 5), (6, 7)], []));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_lz!(16, 2, 3, bimg_s!(16, 2, [(0, 1)], [2])));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_lz!(16, 6, 2, bimg_s!(16, 6, [(0, 1)], [])));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_sk!(16, 3, [(2, 3), (4, 5), (6, 7)], []));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_lz!(16, 2, 3, bimg_s!(16, 2, [(0, 1)], [2])));
                    let o6 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(6, 2, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 6, bimg_lz!(16, 6, 2, bimg_s!(16, 6, [(0, 1)], [])))
                    };
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            17 => {
                if ((!bw!(2) & 0xffffffffffffff00) | (!bw!(6) & 0xffffffffffffff00)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_s!(17, 2, [], [0]));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_s!(17, 6, [], [0]));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_lz!(17, 2, 1, bimg_s!(17, 2, [], [0])));
                    let o6 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(6, 1, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 6, bimg_lz!(17, 6, 1, bimg_s!(17, 6, [], [0])))
                    };
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            18 => {
                if ((!bw!(2) & 0x00ffffffffffffff) | (!bw!(6) & 0x0000ffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_s!(18, 2, [], [7]));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_s!(18, 6, [(6, 7)], []));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_s!(18, 2, [], [7]));
                    let o6 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(6, 2, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 6, bimg_hz!(18, 6, 2, bimg_s!(18, 6, [(6, 7)], [])))
                    };
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            19 => {
                if ((!bw!(1) & 0xffff000000000000) | (!bw!(2) & 0x000000ffffffffff) | (!bw!(5) & 0xffff000000000000) | (!bw!(6) & 0x000000ffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_s!(19, 2, [(5, 6)], [7]));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_s!(19, 6, [(5, 6)], [7]));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_sk!(19, 1, [(0, 1), (2, 3), (4, 5)], []));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_sk!(19, 5, [(0, 1), (2, 3), (4, 5)], []));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_hz!(19, 2, 3, bimg_s!(19, 2, [(5, 6)], [7])));
                    let o6 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(6, 3, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 6, bimg_hz!(19, 6, 3, bimg_s!(19, 6, [(5, 6)], [7])))
                    };
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_sk!(19, 1, [(0, 1), (2, 3), (4, 5)], []));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_sk!(19, 5, [(0, 1), (2, 3), (4, 5)], []));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            20 => {
                if ((!bw!(1) & 0xffffff0000000000) | (!bw!(2) & 0x00000000ffffffff) | (!bw!(5) & 0xffffffff00000000) | (!bw!(6) & 0x0000000000ffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_sk!(20, 2, [(4, 5), (6, 7)], []));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_sk!(20, 6, [(3, 4), (5, 6)], [7]));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_sk!(20, 1, [(0, 1), (2, 3)], [4]));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_lz!(20, 5, 4, bimg_sk!(20, 5, [(0, 1), (2, 3)], [])));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_hz!(20, 2, 4, bimg_sk!(20, 2, [(4, 5), (6, 7)], [])));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_sk!(20, 6, [(3, 4), (5, 6)], [7]));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_sk!(20, 1, [(0, 1), (2, 3)], [4]));
                    let o5 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(5, 4, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 5, bimg_lz!(20, 5, 4, bimg_sk!(20, 5, [(0, 1), (2, 3)], [])))
                    };
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            21 => {
                if ((!bw!(1) & 0xffffffffff000000) | (!bw!(2) & 0x000000000000ffff) | (!bw!(5) & 0xffffffffff000000) | (!bw!(6) & 0x000000000000ffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_sk!(21, 2, [(2, 3), (4, 5), (6, 7)], []));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_sk!(21, 6, [(2, 3), (4, 5), (6, 7)], []));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_lz!(21, 1, 3, bimg_s!(21, 1, [(0, 1)], [2])));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_lz!(21, 5, 3, bimg_s!(21, 5, [(0, 1)], [2])));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_sk!(21, 2, [(2, 3), (4, 5), (6, 7)], []));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_sk!(21, 6, [(2, 3), (4, 5), (6, 7)], []));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_lz!(21, 1, 3, bimg_s!(21, 1, [(0, 1)], [2])));
                    let o5 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(5, 3, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 5, bimg_lz!(21, 5, 3, bimg_s!(21, 5, [(0, 1)], [2])))
                    };
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            22 => {
                if ((!bw!(1) & 0xffffffffffff0000) | (!bw!(5) & 0xffffffffffffff00)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_lz!(22, 1, 2, bimg_s!(22, 1, [(0, 1)], [])));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_s!(22, 5, [], [0]));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_lz!(22, 1, 2, bimg_s!(22, 1, [(0, 1)], [])));
                    let o5 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(5, 1, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 5, bimg_lz!(22, 5, 1, bimg_s!(22, 5, [], [0])))
                    };
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            23 => {
                if ((!bw!(1) & 0x00ffffffffffffff) | (!bw!(5) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_s!(23, 1, [], [7]));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_s!(23, 5, [], [7]));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_s!(23, 1, [], [7]));
                    let o5 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(5, 1, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 5, bimg_s!(23, 5, [], [7]))
                    };
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_g!(4));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            24 => {
                if ((!bw!(1) & 0x0000ffffffffffff) | (!bw!(4) & 0xffff000000000000) | (!bw!(5) & 0x000000ffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_s!(24, 1, [(6, 7)], []));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_s!(24, 5, [(5, 6)], [7]));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_sk!(24, 4, [(0, 1), (2, 3), (4, 5)], []));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_hz!(24, 1, 2, bimg_s!(24, 1, [(6, 7)], [])));
                    let o5 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(5, 3, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 5, bimg_hz!(24, 5, 3, bimg_s!(24, 5, [(5, 6)], [7])))
                    };
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_g!(0));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_sk!(24, 4, [(0, 1), (2, 3), (4, 5)], []));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            25 => {
                if ((!bw!(0) & 0xffffff0000000000) | (!bw!(1) & 0x00000000ffffffff) | (!bw!(4) & 0xffffff0000000000) | (!bw!(5) & 0x00000000ffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_sk!(25, 1, [(4, 5), (6, 7)], []));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_sk!(25, 5, [(4, 5), (6, 7)], []));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_sk!(25, 0, [(0, 1), (2, 3)], [4]));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_sk!(25, 4, [(0, 1), (2, 3)], [4]));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_hz!(25, 1, 4, bimg_sk!(25, 1, [(4, 5), (6, 7)], [])));
                    let o5 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(5, 4, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 5, bimg_hz!(25, 5, 4, bimg_sk!(25, 5, [(4, 5), (6, 7)], [])))
                    };
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_sk!(25, 0, [(0, 1), (2, 3)], [4]));
                    let o4 = opnds!(scaled, scaled_sw, kscaled, 4, bimg_sk!(25, 4, [(0, 1), (2, 3)], [4]));
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            26 => {
                if ((!bw!(0) & 0xffffffff00000000) | (!bw!(1) & 0x0000000000ffffff) | (!bw!(4) & 0xffffffffff000000) | (!bw!(5) & 0x000000000000ffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_sk!(26, 1, [(3, 4), (5, 6)], [7]));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_sk!(26, 5, [(2, 3), (4, 5), (6, 7)], []));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_lz!(26, 0, 4, bimg_sk!(26, 0, [(0, 1), (2, 3)], [])));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_lz!(26, 4, 3, bimg_s!(26, 4, [(0, 1)], [2])));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_sk!(26, 1, [(3, 4), (5, 6)], [7]));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_sk!(26, 5, [(2, 3), (4, 5), (6, 7)], []));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_lz!(26, 0, 4, bimg_sk!(26, 0, [(0, 1), (2, 3)], [])));
                    let o4 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(4, 3, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 4, bimg_lz!(26, 4, 3, bimg_s!(26, 4, [(0, 1)], [2])))
                    };
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            27 => {
                if ((!bw!(0) & 0xffffffffffff0000) | (!bw!(4) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_lz!(27, 0, 2, bimg_s!(27, 0, [(0, 1)], [])));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_lz!(27, 4, 2, bimg_s!(27, 4, [(0, 1)], [])));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_lz!(27, 0, 2, bimg_s!(27, 0, [(0, 1)], [])));
                    let o4 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(4, 2, false)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 4, bimg_lz!(27, 4, 2, bimg_s!(27, 4, [(0, 1)], [])))
                    };
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
            }
            28 => {
                if ((!bw!(0) & 0xffffffffffffff00) | (!bw!(4) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                if V & ABV_FUSE == 0 {
                    let lo3 = prods!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let hi3 = prods!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = seed!(lo3, hi3);
                    let lo2 = prods!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let hi2 = prods!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    horner_absorb_first(&mut acc, &lo2, &hi2);
                    let lo1 = prods!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let hi1 = prods!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
                    let lo0 = prods!(plain, plain_sw, kplain, 0, bimg_s!(28, 0, [], [0]));
                    let hi0 = prods!(scaled, scaled_sw, kscaled, 4, bimg_s!(28, 4, [], [7]));
                    horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
                    acc
                } else {
                    let o3 = opnds!(plain, plain_sw, kplain, 3, bimg_g!(3));
                    let o7 = opnds!(scaled, scaled_sw, kscaled, 7, bimg_g!(7));
                    let mut acc = fuse_seed(&o3, &o7);
                    let o2 = opnds!(plain, plain_sw, kplain, 2, bimg_g!(2));
                    let o6 = opnds!(scaled, scaled_sw, kscaled, 6, bimg_g!(6));
                    fuse_absorb_first(&mut acc, &o2, &o6);
                    let o1 = opnds!(plain, plain_sw, kplain, 1, bimg_g!(1));
                    let o5 = opnds!(scaled, scaled_sw, kscaled, 5, bimg_g!(5));
                    fuse_absorb_k1(&mut acc, &o1, &o5);
                    let o0 = opnds!(plain, plain_sw, kplain, 0, bimg_lz!(28, 0, 1, bimg_s!(28, 0, [], [0])));
                    let o4 = if V & ABV_SPARSE_SCALE != 0 {
                        opnds_ss!(4, 1, true)
                    } else {
                        opnds!(scaled, scaled_sw, kscaled, 4, bimg_s!(28, 4, [], [7]))
                    };
                    fuse_absorb_k0(&mut acc, &o0, &o4);
                    acc
                }
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

/// Plain-only split-Horner4 body for [`ABV_SPLIT_H4`]. Never touches `scaled`,
/// `scaled_sw`, or `kscaled`: evaluates `k = 4..7` and `k = 0..3` as two
/// single-stream Horner4 accumulators out of the plain table images, reducing
/// the high half via `gf8_reduce_x4_vec16` and XORing with the `gf8_reduce_vec16`
/// low half. Separated from [`bstatic_horner_blk_body`] so non-split
/// monomorphizations do not pay MIR/compile cost for split arms and vice versa.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn bstatic_horner_blk_body_split<const BLK: usize, const V: u32, const HOIST: bool>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: &'static [[u8; 64]; 248],
    kara: *const u8,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::{gf8_reduce_vec16, gf8_reduce_x4_vec16};
    use core::arch::aarch64::*;
    const { assert!(V & ABV_SPLIT_H4 != 0 && V & ABV_FUSE != 0 && V & ABV_HORNER2 != 0) };
    let b_med = BLK % 16;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    if HOIST {
        debug_assert!(byte_base_b + 8 * N_CHUNKS <= a_packed.len() && byte_base_b + 8 * N_CHUNKS <= b_packed.len());
    } else {
        assert!(byte_base_b + 8 * N_CHUNKS <= a_packed.len() && byte_base_b + 8 * N_CHUNKS <= b_packed.len());
    }
    let plain = inv_table.data_ptr();
    let plain_sw = inv_table.half_swapped_data_ptr();
    let kplain = kara;
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
                apply_word_into_4_regs_v::<V, false>(plain, plain_sw, kplain, bw!($k))
            };
        }
        macro_rules! bimg_s {
            ($blk:literal, $k:literal, [$(($j0:literal, $j1:literal)),*], [$($js:literal),*]) => {{
                let pp = pbase.add(($blk * 8 + $k) * 64);
                #[allow(unused_mut)]
                let (mut db0, mut db1, mut db2, mut db3) = transform_partial_seed::<V>(pp);
                #[allow(unused_variables)]
                let b_word = transform_partial_word::<V>(bw!($k));
                if V & ABV_NOSW != 0 {
                    $(xor_vary_ext::<$j0, $j1>(plain, b_word, [&mut db0, &mut db1, &mut db2, &mut db3]);)*
                    $(xor_vary_ext::<$js, 8>(plain, b_word, [&mut db0, &mut db1, &mut db2, &mut db3]);)*
                } else {
                    $(xor_vary_pair::<$j0, $j1>(plain, plain_sw, b_word, &mut db0, &mut db1, &mut db2, &mut db3);)*
                    $(xor_vary_single::<$js>(plain, plain_sw, b_word, &mut db0, &mut db1, &mut db2, &mut db3);)*
                }
                (db0, db1, db2, db3)
            }};
        }
        macro_rules! bimg_sk {
            ($blk:literal, $k:literal, $pairs:tt, $singles:tt) => {{
                if V & ABV_KARA != 0 {
                    apply_word_into_4_regs_v::<V, false>(plain, plain_sw, kplain, bw!($k))
                } else {
                    bimg_s!($blk, $k, $pairs, $singles)
                }
            }};
        }
        macro_rules! bimg_lz {
            ($blk:literal, $k:literal, $nb:literal, $inc:expr) => {{
                if V & ABV_BSTATIC_Z != 0 {
                    if V & ABV_TRANSFORM_TRIM != 0 {
                        apply_word_z_low::<$nb, true>(plain, kplain, !bw!($k))
                    } else {
                        let pp = pbase.add(($blk * 8 + $k) * 64);
                        let (z0, z1, z2, z3) = apply_word_z_low::<$nb, false>(
                            plain, kplain, bw!($k),
                        );
                        let (d0, d1, d2, d3) = transform_partial_seed::<V>(pp);
                        (
                            veorq_u8(d0, z0),
                            veorq_u8(d1, z1),
                            veorq_u8(d2, z2),
                            veorq_u8(d3, z3),
                        )
                    }
                } else {
                    $inc
                }
            }};
        }
        macro_rules! bimg_hz {
            ($blk:literal, $k:literal, $nb:literal, $inc:expr) => {{
                if V & ABV_BSTATIC_Z != 0 && V & ABV_TRANSFORM_TRIM != 0 {
                    apply_word_z_high::<$nb, true>(plain, kplain, !bw!($k))
                } else {
                    $inc
                }
            }};
        }
        macro_rules! opnds_p {
            ($k:literal, $bimg:expr) => {{
                let aw = u64::from_le(core::ptr::read_unaligned(a_base.add($k * N_CHUNKS).cast::<u64>()));
                let (da0, da1, da2, da3) = apply_word_into_4_regs_v::<V, false>(plain, plain_sw, kplain, aw);
                let (db0, db1, db2, db3) = $bimg;
                ([da0, da1, da2, da3], [db0, db1, db2, db3])
            }};
        }
        macro_rules! reduce_x4_4 {
            ($acc:expr) => {{
                let a = $acc;
                split_fence4((
                    gf8_reduce_x4_vec16(vreinterpretq_u8_u16(a[0]), vreinterpretq_u8_u16(a[1])),
                    gf8_reduce_x4_vec16(vreinterpretq_u8_u16(a[2]), vreinterpretq_u8_u16(a[3])),
                    gf8_reduce_x4_vec16(vreinterpretq_u8_u16(a[4]), vreinterpretq_u8_u16(a[5])),
                    gf8_reduce_x4_vec16(vreinterpretq_u8_u16(a[6]), vreinterpretq_u8_u16(a[7])),
                ))
            }};
        }
        macro_rules! reduce_xor_store {
            ($acc:expr, $rhi:expr) => {{
                let a = $acc;
                let (h0, h1, h2, h3) = $rhi;
                let r0 = veorq_u8(gf8_reduce_vec16(vreinterpretq_u8_u16(a[0]), vreinterpretq_u8_u16(a[1])), h0);
                let r1 = veorq_u8(gf8_reduce_vec16(vreinterpretq_u8_u16(a[2]), vreinterpretq_u8_u16(a[3])), h1);
                let r2 = veorq_u8(gf8_reduce_vec16(vreinterpretq_u8_u16(a[4]), vreinterpretq_u8_u16(a[5])), h2);
                let r3 = veorq_u8(gf8_reduce_vec16(vreinterpretq_u8_u16(a[6]), vreinterpretq_u8_u16(a[7])), h3);
                store_row_64(out.as_mut_ptr(), nt_store, r0, r1, r2, r3);
            }};
        }
        match BLK {
            3 => {
                if ((!bw!(1) & 0x0000ffffffffffff) | (!bw!(4) & 0xffff000000000000) | (!bw!(5) & 0x000000ffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_hz!(3, 5, 3, bimg_s!(3, 5, [(5, 6)], [7])));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_sk!(3, 4, [(0, 1), (2, 3), (4, 5)], []));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_hz!(3, 1, 2, bimg_s!(3, 1, [(6, 7)], [])));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            4 => {
                if ((!bw!(0) & 0xffff000000000000) | (!bw!(1) & 0x000000ffffffffff) | (!bw!(4) & 0xffffff0000000000) | (!bw!(5) & 0x00000000ffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_hz!(4, 5, 4, bimg_sk!(4, 5, [(4, 5), (6, 7)], [])));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_sk!(4, 4, [(0, 1), (2, 3)], [4]));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_hz!(4, 1, 3, bimg_s!(4, 1, [(5, 6)], [7])));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_sk!(4, 0, [(0, 1), (2, 3), (4, 5)], []));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            5 => {
                if ((!bw!(0) & 0xffffffff00000000) | (!bw!(1) & 0x0000000000ffffff) | (!bw!(4) & 0xffffffffff000000) | (!bw!(5) & 0x000000000000ffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_sk!(5, 5, [(2, 3), (4, 5), (6, 7)], []));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_lz!(5, 4, 3, bimg_s!(5, 4, [(0, 1)], [2])));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_sk!(5, 1, [(3, 4), (5, 6)], [7]));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_lz!(5, 0, 4, bimg_sk!(5, 0, [(0, 1), (2, 3)], [])));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            6 => {
                if ((!bw!(0) & 0xffffffffff000000) | (!bw!(1) & 0x000000000000ffff) | (!bw!(4) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_lz!(6, 4, 2, bimg_s!(6, 4, [(0, 1)], [])));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_sk!(6, 1, [(2, 3), (4, 5), (6, 7)], []));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_lz!(6, 0, 3, bimg_s!(6, 0, [(0, 1)], [2])));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            7 => {
                if ((!bw!(0) & 0xffffffffffffff00) | (!bw!(4) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_s!(7, 4, [], [7]));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_lz!(7, 0, 1, bimg_s!(7, 0, [], [0])));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            8 => {
                if ((!bw!(0) & 0x00ffffffffffffff) | (!bw!(4) & 0x0000ffffffffffff) | (!bw!(7) & 0xffff000000000000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_sk!(8, 7, [(0, 1), (2, 3), (4, 5)], []));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_hz!(8, 4, 2, bimg_s!(8, 4, [(6, 7)], [])));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_s!(8, 0, [], [7]));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            9 => {
                if ((!bw!(0) & 0x000000ffffffffff) | (!bw!(3) & 0xffffff0000000000) | (!bw!(4) & 0x00000000ffffffff) | (!bw!(7) & 0xffffff0000000000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_sk!(9, 7, [(0, 1), (2, 3)], [4]));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_hz!(9, 4, 4, bimg_sk!(9, 4, [(4, 5), (6, 7)], [])));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_sk!(9, 3, [(0, 1), (2, 3)], [4]));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_hz!(9, 0, 3, bimg_s!(9, 0, [(5, 6)], [7])));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            10 => {
                if ((!bw!(0) & 0x00000000ffffffff) | (!bw!(3) & 0xffffffff00000000) | (!bw!(4) & 0x0000000000ffffff) | (!bw!(7) & 0xffffffffff000000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_lz!(10, 7, 3, bimg_s!(10, 7, [(0, 1)], [2])));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_sk!(10, 4, [(3, 4), (5, 6)], [7]));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_lz!(10, 3, 4, bimg_sk!(10, 3, [(0, 1), (2, 3)], [])));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_hz!(10, 0, 4, bimg_sk!(10, 0, [(4, 5), (6, 7)], [])));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            11 => {
                if ((!bw!(0) & 0x000000000000ffff) | (!bw!(3) & 0xffffffffffff0000) | (!bw!(7) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_lz!(11, 7, 2, bimg_s!(11, 7, [(0, 1)], [])));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_lz!(11, 3, 2, bimg_s!(11, 3, [(0, 1)], [])));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_sk!(11, 0, [(2, 3), (4, 5), (6, 7)], []));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            12 => {
                if ((!bw!(3) & 0xffffffffffffff00) | (!bw!(7) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_s!(12, 7, [], [7]));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_lz!(12, 3, 1, bimg_s!(12, 3, [], [0])));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            13 => {
                if ((!bw!(3) & 0x0000ffffffffffff) | (!bw!(7) & 0x0000ffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_hz!(13, 7, 2, bimg_s!(13, 7, [(6, 7)], [])));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_hz!(13, 3, 2, bimg_s!(13, 3, [(6, 7)], [])));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            14 => {
                if ((!bw!(2) & 0xffff000000000000) | (!bw!(3) & 0x000000ffffffffff) | (!bw!(6) & 0xffffff0000000000) | (!bw!(7) & 0x00000000ffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_hz!(14, 7, 4, bimg_sk!(14, 7, [(4, 5), (6, 7)], [])));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_sk!(14, 6, [(0, 1), (2, 3)], [4]));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_hz!(14, 3, 3, bimg_s!(14, 3, [(5, 6)], [7])));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_sk!(14, 2, [(0, 1), (2, 3), (4, 5)], []));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            15 => {
                if ((!bw!(2) & 0xffffffff00000000) | (!bw!(3) & 0x0000000000ffffff) | (!bw!(6) & 0xffffffff00000000) | (!bw!(7) & 0x0000000000ffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_sk!(15, 7, [(3, 4), (5, 6)], [7]));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_lz!(15, 6, 4, bimg_sk!(15, 6, [(0, 1), (2, 3)], [])));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_sk!(15, 3, [(3, 4), (5, 6)], [7]));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_lz!(15, 2, 4, bimg_sk!(15, 2, [(0, 1), (2, 3)], [])));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            16 => {
                if ((!bw!(2) & 0xffffffffff000000) | (!bw!(3) & 0x000000000000ffff) | (!bw!(6) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_lz!(16, 6, 2, bimg_s!(16, 6, [(0, 1)], [])));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_sk!(16, 3, [(2, 3), (4, 5), (6, 7)], []));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_lz!(16, 2, 3, bimg_s!(16, 2, [(0, 1)], [2])));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            17 => {
                if ((!bw!(2) & 0xffffffffffffff00) | (!bw!(6) & 0xffffffffffffff00)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_lz!(17, 6, 1, bimg_s!(17, 6, [], [0])));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_lz!(17, 2, 1, bimg_s!(17, 2, [], [0])));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            18 => {
                if ((!bw!(2) & 0x00ffffffffffffff) | (!bw!(6) & 0x0000ffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_hz!(18, 6, 2, bimg_s!(18, 6, [(6, 7)], [])));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_s!(18, 2, [], [7]));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            19 => {
                if ((!bw!(1) & 0xffff000000000000) | (!bw!(2) & 0x000000ffffffffff) | (!bw!(5) & 0xffff000000000000) | (!bw!(6) & 0x000000ffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_hz!(19, 6, 3, bimg_s!(19, 6, [(5, 6)], [7])));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_sk!(19, 5, [(0, 1), (2, 3), (4, 5)], []));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_hz!(19, 2, 3, bimg_s!(19, 2, [(5, 6)], [7])));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_sk!(19, 1, [(0, 1), (2, 3), (4, 5)], []));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            20 => {
                if ((!bw!(1) & 0xffffff0000000000) | (!bw!(2) & 0x00000000ffffffff) | (!bw!(5) & 0xffffffff00000000) | (!bw!(6) & 0x0000000000ffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_sk!(20, 6, [(3, 4), (5, 6)], [7]));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_lz!(20, 5, 4, bimg_sk!(20, 5, [(0, 1), (2, 3)], [])));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_hz!(20, 2, 4, bimg_sk!(20, 2, [(4, 5), (6, 7)], [])));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_sk!(20, 1, [(0, 1), (2, 3)], [4]));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            21 => {
                if ((!bw!(1) & 0xffffffffff000000) | (!bw!(2) & 0x000000000000ffff) | (!bw!(5) & 0xffffffffff000000) | (!bw!(6) & 0x000000000000ffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_sk!(21, 6, [(2, 3), (4, 5), (6, 7)], []));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_lz!(21, 5, 3, bimg_s!(21, 5, [(0, 1)], [2])));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_sk!(21, 2, [(2, 3), (4, 5), (6, 7)], []));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_lz!(21, 1, 3, bimg_s!(21, 1, [(0, 1)], [2])));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            22 => {
                if ((!bw!(1) & 0xffffffffffff0000) | (!bw!(5) & 0xffffffffffffff00)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_lz!(22, 5, 1, bimg_s!(22, 5, [], [0])));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_lz!(22, 1, 2, bimg_s!(22, 1, [(0, 1)], [])));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            23 => {
                if ((!bw!(1) & 0x00ffffffffffffff) | (!bw!(5) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_s!(23, 5, [], [7]));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_g!(4));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_s!(23, 1, [], [7]));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            24 => {
                if ((!bw!(1) & 0x0000ffffffffffff) | (!bw!(4) & 0xffff000000000000) | (!bw!(5) & 0x000000ffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_hz!(24, 5, 3, bimg_s!(24, 5, [(5, 6)], [7])));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_sk!(24, 4, [(0, 1), (2, 3), (4, 5)], []));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_hz!(24, 1, 2, bimg_s!(24, 1, [(6, 7)], [])));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_g!(0));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            25 => {
                if ((!bw!(0) & 0xffffff0000000000) | (!bw!(1) & 0x00000000ffffffff) | (!bw!(4) & 0xffffff0000000000) | (!bw!(5) & 0x00000000ffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_hz!(25, 5, 4, bimg_sk!(25, 5, [(4, 5), (6, 7)], [])));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_sk!(25, 4, [(0, 1), (2, 3)], [4]));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_hz!(25, 1, 4, bimg_sk!(25, 1, [(4, 5), (6, 7)], [])));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_sk!(25, 0, [(0, 1), (2, 3)], [4]));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            26 => {
                if ((!bw!(0) & 0xffffffff00000000) | (!bw!(1) & 0x0000000000ffffff) | (!bw!(4) & 0xffffffffff000000) | (!bw!(5) & 0x000000000000ffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_sk!(26, 5, [(2, 3), (4, 5), (6, 7)], []));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_lz!(26, 4, 3, bimg_s!(26, 4, [(0, 1)], [2])));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_sk!(26, 1, [(3, 4), (5, 6)], [7]));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_lz!(26, 0, 4, bimg_sk!(26, 0, [(0, 1), (2, 3)], [])));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            27 => {
                if ((!bw!(0) & 0xffffffffffff0000) | (!bw!(4) & 0xffffffffffff0000)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_lz!(27, 4, 2, bimg_s!(27, 4, [(0, 1)], [])));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_lz!(27, 0, 2, bimg_s!(27, 0, [(0, 1)], [])));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            28 => {
                if ((!bw!(0) & 0xffffffffffffff00) | (!bw!(4) & 0x00ffffffffffffff)) != 0 {
                    bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out, nt_store);
                    return;
                }
                let o7 = opnds_p!(7, bimg_g!(7));
                let mut acc_hi = fuse_seed_one(&o7);
                let o6 = opnds_p!(6, bimg_g!(6));
                fuse_absorb_first_one(&mut acc_hi, &o6);
                let o5 = opnds_p!(5, bimg_g!(5));
                fuse_absorb_k1_one(&mut acc_hi, &o5);
                let o4 = opnds_p!(4, bimg_s!(28, 4, [], [7]));
                fuse_absorb_k0_one(&mut acc_hi, &o4);
                let r_hi = reduce_x4_4!(acc_hi);
                let o3 = opnds_p!(3, bimg_g!(3));
                let mut acc_lo = fuse_seed_one(&o3);
                let o2 = opnds_p!(2, bimg_g!(2));
                fuse_absorb_first_one(&mut acc_lo, &o2);
                let o1 = opnds_p!(1, bimg_g!(1));
                fuse_absorb_k1_one(&mut acc_lo, &o1);
                let o0 = opnds_p!(0, bimg_lz!(28, 0, 1, bimg_s!(28, 0, [], [0])));
                fuse_absorb_k0_one(&mut acc_lo, &o0);
                reduce_xor_store!(acc_lo, r_hi);
            }
            _ => unreachable!("no static-B Horner plan for this arm"),
        }
    }
}


/// Per-row entry: runs the Horner static-B body of arm `(w, b_med)` and
/// returns `true`, or returns `false` when the arm has no plan (the caller
/// then runs another kernel). In non-test builds the ranked static-B rows run
/// through the window/unit kernels (`bstatic_horner_window_unit_impl`), so
/// returning `false` avoids monomorphizing 26 legacy `V = 0` `#[inline(never)]`
/// per-row bodies while falling back to the bit-identical generic h4 kernel.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(unused_variables))]
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
    #[cfg(not(test))]
    {
        false
    }
    #[cfg(test)]
    {
        if w > 1 || b_med >= 16 {
            return false;
        }
        match w * 16 + b_med {
            3 => bstatic_horner_blk::<3, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            4 => bstatic_horner_blk::<4, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            5 => bstatic_horner_blk::<5, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            6 => bstatic_horner_blk::<6, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            7 => bstatic_horner_blk::<7, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            8 => bstatic_horner_blk::<8, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            9 => bstatic_horner_blk::<9, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            10 => bstatic_horner_blk::<10, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            11 => bstatic_horner_blk::<11, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            12 => bstatic_horner_blk::<12, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            13 => bstatic_horner_blk::<13, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            14 => bstatic_horner_blk::<14, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            15 => bstatic_horner_blk::<15, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            16 => bstatic_horner_blk::<16, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            17 => bstatic_horner_blk::<17, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            18 => bstatic_horner_blk::<18, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            19 => bstatic_horner_blk::<19, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            20 => bstatic_horner_blk::<20, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            21 => bstatic_horner_blk::<21, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            22 => bstatic_horner_blk::<22, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            23 => bstatic_horner_blk::<23, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            24 => bstatic_horner_blk::<24, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            25 => bstatic_horner_blk::<25, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            26 => bstatic_horner_blk::<26, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            27 => bstatic_horner_blk::<27, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            28 => bstatic_horner_blk::<28, 0>(a_packed, b_packed, inv_table, chunk_byte_base, partials, core::ptr::null(), out, nt_store),
            _ => return false,
        }
        true
    }
}

/// First and one-past-last `b_med` of window `w`'s static-B Horner rows,
/// i.e. the rows [`bstatic_horner_window_rows`] writes.
#[cfg(target_arch = "aarch64")]
pub(crate) const BSTATIC_HORNER_WINDOW_ROWS: [(usize, usize); 2] = [(3, 16), (0, 13)];

/// Kill switch for the hoisted bounds check of the inlined static-B window
/// rows: any value of `FLOCK_NO_ZC_AB_ROW_SETUP_HOIST` restores one bounds
/// check per row. Read once per process. The rows written are
/// byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_row_setup_hoist_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_ROW_SETUP_HOIST").is_none())
}

/// Kill switch for the hoisted bounds check of the helper threads' static-B
/// window rows ([`AB_HZ_VARIANT`] with [`ABV_HORNER2`]): any value of
/// `FLOCK_NO_ZC_AB_HZ_HOIST` restores one bounds check per row there, as
/// `FLOCK_NO_ZC_AB_ROW_SETUP_HOIST` does. Read once per process. The rows
/// written are byte-identical: only the bounds checks move.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_hz_hoist_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_HZ_HOIST").is_none())
}

/// All static-B Horner rows of window `w` (see [`BSTATIC_HORNER_WINDOW_ROWS`])
/// as one unrolled sequence of direct calls into the per-arm bodies, writing
/// row `b_med` to `out_outer[b_med * 64..][..64]`. Each row checks its own
/// bounds.
#[cfg(all(test, target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn bstatic_horner_window_rows<const V: u32>(
    w: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: &'static [[u8; 64]; 248],
    out_outer: &mut [u8],
    nt_store: bool,
) {
    bstatic_horner_window_rows_impl::<V, false>(
        w, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_outer, nt_store,
    );
}

/// [`bstatic_horner_window_rows`] with one bounds check per window instead
/// of one per row (`V` must have [`ABV_INLINE_WINDOW`]).
#[cfg(all(test, target_arch = "aarch64"))]
#[allow(dead_code, clippy::too_many_arguments)]
#[inline]
pub(crate) fn bstatic_horner_window_rows_hoisted<const V: u32>(
    w: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: &'static [[u8; 64]; 248],
    out_outer: &mut [u8],
    nt_store: bool,
) {
    const { assert!(V & ABV_INLINE_WINDOW != 0) };
    bstatic_horner_window_rows_impl::<V, true>(
        w, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_outer, nt_store,
    );
}

/// [`bstatic_horner_window_rows`] with the per-row bounds check hoisted to
/// one check per window (`HOIST = true`, inlined bodies only) or not.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn bstatic_horner_window_rows_impl<const V: u32, const HOIST: bool>(
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
    // Every row of window `w` has `b_med < hi` and reads bytes up to
    // `chunk_byte_base + (b_med + 1) * 8 * N_CHUNKS`; this one check is the
    // last row's own check, so it covers every row's and is never stricter.
    if HOIST {
        let end = chunk_byte_base + BSTATIC_HORNER_WINDOW_ROWS[w].1 * N_CHUNKS * 8;
        assert!(end <= a_packed.len() && end <= b_packed.len());
    }
    let out = out_outer.as_mut_ptr();
    let kara = if V & ABV_KARA != 0 {
        ab_kara_tables(inv_table)
    } else if V & ABV_HZ != 0 {
        ab_hz_tables(inv_table)
    } else {
        core::ptr::null()
    };
    macro_rules! rows {
        ($($blk:literal),*) => {{
            $(
                // SAFETY: `$blk % 16 < 16` and `out_outer` holds 16 rows of 64 bytes.
                let dst = unsafe { &mut *out.add(($blk % 16) * 64).cast::<[u8; 64]>() };
                if const { V & ABV_INLINE_WINDOW != 0 } {
                    bstatic_horner_blk_body::<$blk, V, HOIST>(
                        a_packed, b_packed, inv_table, chunk_byte_base, partials, kara, dst, nt_store,
                    );
                } else {
                    #[cfg(test)]
                    bstatic_horner_blk::<$blk, V>(a_packed, b_packed, inv_table, chunk_byte_base, partials, kara, dst, nt_store);
                    #[cfg(not(test))]
                    {
                        let _ = (partials, kara);
                        bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, $blk % 16, dst, nt_store);
                    }
                }
            )*
        }};
    }
    if w == 0 {
        rows!(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    } else {
        rows!(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28);
    }
}

/// Kill switch for the fixed store flavour of the unit window rows: any
/// value of `FLOCK_NO_ZC_AB_ROWS_NT_CONST` makes every row of a
/// [`bstatic_horner_window_unit_impl`] call test `nt_store` again. Read
/// once per process. The rows written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_rows_nt_const_enabled() -> bool {
    *AB_ROWS_NT_CONST.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_ROWS_NT_CONST").is_none())
}

#[cfg(target_arch = "aarch64")]
static AB_ROWS_NT_CONST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// The window rows of consecutive chunks in one call: chunk `i` of
/// `out_unit` (1 KiB each) is chunk `x_outer + i` of window
/// `(x_outer + i) & window_mask`, reads its rows at
/// `chunk_byte_base + i * 1 KiB` with `partials[w]` and gets the rows, the
/// checks and the stores of one [`bstatic_horner_window_rows_impl`] call
/// with those arguments. The table pointer is fetched once per call.
/// `nt_store` is the same for every row of the call: with non-temporal
/// stores it picks the instance of [`bstatic_horner_window_unit_nt`] once;
/// cached stores (a kill switch only) take the per-chunk rows. With
/// `FLOCK_NO_ZC_AB_ROWS_NT_CONST` set every row tests it
/// ([`bstatic_horner_window_unit_flag`]).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn bstatic_horner_window_unit_impl<const V: u32, const HOIST: bool>(
    x_outer: usize,
    window_mask: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: [&'static [[u8; 64]; 248]; 2],
    out_unit: &mut [u8],
    nt_store: bool,
) {
    // The first call reads the switch (the cold arm); later calls only look
    // at its stored value.
    match AB_ROWS_NT_CONST.get() {
        Some(true) if nt_store => {
            if const { V & !ABV_TRANSFORM_TRIM == AB_UNIT_MAIN && HOIST } {
                // The main-pool instance with non-temporal stores runs the
                // rows with the z-basis applies in the [`ABV_Z_OUT3`] form;
                // `FLOCK_NO_ZC_AB_Z_EOR3` takes the per-chunk rows. On top,
                // the fused products ([`ABV_FUSE`], `FLOCK_NO_ZC_AB_FUSE`
                // kills).
                if ab_z_out3_enabled() {
                    const BASE: u32 = AB_UNIT_MAIN | ABV_Z_OUT3;
                    if ab_fuse_enabled() {
                        if const { V & ABV_TRANSFORM_TRIM != 0 } {
                            // The fused rows with the complemented static-B
                            // images take each row's short-B `x^4` word with
                            // the weight on B ([`ABV_SPARSE_SCALE`]) and the
                            // other `x^4` words in the lane z form
                            // ([`ABV_KARA_LANEZ`]); this is the only fused
                            // instance with them. `FLOCK_NO_AB_SPARSE_SCALE=1`
                            // or `FLOCK_NO_ZC_AB_LANEZ=1` takes the per-chunk
                            // rows, which write the same bytes (build budget:
                            // no separate unit instance).
                            // P-cores (128 KiB L1D) keep the combined Horner4 form
                            // (with `bimg_hz!` + `one`-absorption + sparse scale +
                            // lane-z): split Horner4 adds +20 `tbl`s/row which
                            // outweighs the 36 KiB L1D savings on 128 KiB P-cores.
                            // E-cores (64 KiB L1D) enable `ABV_SPLIT_H4` below,
                            // shrinking their working set from 80 KiB to 56 KiB.
                            if ab_sparse_scale_enabled() && ab_kara_lanez_enabled() {
                                bstatic_horner_window_unit_nt::<
                                    { BASE | AB_FUSE_FORM | ABV_TRANSFORM_TRIM | ABV_SPARSE_SCALE | ABV_KARA_LANEZ },
                                    true,
                                    true,
                                >(
                                    x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials,
                                    out_unit,
                                )
                            } else {
                                bstatic_horner_window_unit_per_chunk::<V, HOIST>(
                                    x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials,
                                    out_unit, true,
                                )
                            }
                        } else {
                            bstatic_horner_window_unit_nt::<{ BASE | AB_FUSE_FORM }, true, true>(
                                x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit,
                            )
                        }
                    } else {
                        // Kill switch only (build budget): the per-chunk rows
                        // write the same bytes; no separate unit instance.
                        let _ = BASE;
                        bstatic_horner_window_unit_per_chunk::<V, HOIST>(
                            x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit,
                            true,
                        )
                    }
                } else {
                    bstatic_horner_window_unit_per_chunk::<V, HOIST>(
                        x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit, true,
                    )
                }
            } else if const { V & !ABV_TRANSFORM_TRIM == AB_UNIT_HELPER } {
                // The helper threads' half-pair z rows with the fused
                // products ([`ABV_FUSE`], `FLOCK_NO_ZC_AB_FUSE_E` kills), with
                // the same bounds checks (`HOIST`: one per window).
                if ab_fuse_e_enabled() {
                    if const { V & ABV_TRANSFORM_TRIM != 0 } {
                        if ab_split_h4_enabled() {
                            bstatic_horner_window_unit_split_nt::<{ AB_UNIT_HELPER | AB_FUSE_FORM | ABV_TRANSFORM_TRIM | ABV_SPLIT_H4 }, HOIST, true>(
                                x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit,
                            )
                        } else {
                            bstatic_horner_window_unit_nt::<{ AB_UNIT_HELPER | AB_FUSE_FORM | ABV_TRANSFORM_TRIM }, HOIST, true>(
                                x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit,
                            )
                        }
                    } else {
                        bstatic_horner_window_unit_nt::<{ AB_UNIT_HELPER | AB_FUSE_FORM }, HOIST, true>(
                            x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit,
                        )
                    }
                } else {
                    // Kill switch only (build budget): the per-chunk rows
                    // write the same bytes; no separate unit instance.
                    bstatic_horner_window_unit_per_chunk::<V, HOIST>(
                        x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit,
                        true,
                    )
                }
            } else {
                bstatic_horner_window_unit_nt::<V, HOIST, true>(
                    x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit,
                )
            }
        }
        // Cached stores: kill switch only (build budget). Every AB precompute
        // entry stores non-temporally unless `FLOCK_NO_ZC_AB_PRE_NT` is set, so
        // the ranked prove never reaches this arm (counted at log2=18: zero
        // calls in the warm-up and the timed proof). The per-chunk rows write
        // the same bytes; no separate cached-store unit instance.
        Some(true) => bstatic_horner_window_unit_per_chunk::<V, HOIST>(
            x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit, false,
        ),
        Some(false) => bstatic_horner_window_unit_per_chunk::<V, HOIST>(
            x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit, nt_store,
        ),
        None => bstatic_horner_window_unit_first::<V, HOIST>(
            x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit, nt_store,
        ),
    }
}

/// [`bstatic_horner_window_unit_impl`] as one
/// [`bstatic_horner_window_rows_impl`] call per chunk (the same rows, checks
/// and stores), each row testing `nt_store`.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn bstatic_horner_window_unit_per_chunk<const V: u32, const HOIST: bool>(
    x_outer: usize,
    window_mask: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: [&'static [[u8; 64]; 248]; 2],
    out_unit: &mut [u8],
    nt_store: bool,
) {
    for (i, out_outer) in out_unit.chunks_exact_mut(16 * 64).enumerate() {
        let w = (x_outer + i) & window_mask;
        assert!(w <= 1);
        bstatic_horner_window_rows_impl::<V, HOIST>(
            w,
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base + i * 16 * 64,
            partials[w],
            out_outer,
            nt_store,
        );
    }
}

/// The main-pool workers' instance of [`bstatic_horner_window_unit_impl`]
/// (the form `ab_window_unit_plan` maps their window rows to). With
/// non-temporal stores it runs with [`ABV_Z_OUT3`] added unless
/// `FLOCK_NO_ZC_AB_Z_EOR3` is set.
#[cfg(target_arch = "aarch64")]
const AB_UNIT_MAIN: u32 =
    AB_Z_VARIANT | ABV_INLINE_WINDOW | ABV_KARA_XOR16 | ABV_BSTATIC_Z | ABV_KARA_SLOT | ABV_HORNER2;

/// The helper threads' instance of [`bstatic_horner_window_unit_impl`]
/// (`ab_window_unit_plan`'s helper form). With non-temporal stores it runs
/// with [`ABV_FUSE`] added unless `FLOCK_NO_ZC_AB_FUSE_E` is set, with
/// either bounds-check form (`HOIST`, see [`ab_hz_hoist_enabled`]).
#[cfg(target_arch = "aarch64")]
const AB_UNIT_HELPER: u32 = AB_HZ_VARIANT | ABV_HORNER2;

/// The fused-product form the unit rows take ([`ABV_FUSE`]).
#[cfg(target_arch = "aarch64")]
const AB_FUSE_FORM: u32 = ABV_FUSE;

/// [`bstatic_horner_window_unit_impl`] while the switch is unread: reads it,
/// then runs the form it picks.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn bstatic_horner_window_unit_first<const V: u32, const HOIST: bool>(
    x_outer: usize,
    window_mask: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: [&'static [[u8; 64]; 248]; 2],
    out_unit: &mut [u8],
    nt_store: bool,
) {
    ab_rows_nt_const_enabled();
    bstatic_horner_window_unit_impl::<V, HOIST>(
        x_outer, window_mask, a_packed, b_packed, inv_table, chunk_byte_base, partials, out_unit, nt_store,
    );
}

/// [`bstatic_horner_window_unit_impl`] with the store flavour tested by
/// every row.
#[cfg(all(test, target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn bstatic_horner_window_unit_flag<const V: u32, const HOIST: bool>(
    x_outer: usize,
    window_mask: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: [&'static [[u8; 64]; 248]; 2],
    out_unit: &mut [u8],
    nt_store: bool,
) {
    let kara = if V & ABV_KARA != 0 {
        ab_kara_tables(inv_table)
    } else if V & ABV_HZ != 0 {
        ab_hz_tables(inv_table)
    } else {
        core::ptr::null()
    };
    for (i, out_outer) in out_unit.chunks_exact_mut(16 * 64).enumerate() {
        let w = (x_outer + i) & window_mask;
        let chunk_byte_base = chunk_byte_base + i * 16 * 64;
        assert!(w <= 1);
        let partials = partials[w];
        if HOIST {
            let end = chunk_byte_base + BSTATIC_HORNER_WINDOW_ROWS[w].1 * N_CHUNKS * 8;
            assert!(end <= a_packed.len() && end <= b_packed.len());
        }
        let out = out_outer.as_mut_ptr();
        macro_rules! rows {
            ($($blk:literal),*) => {{
                $(
                    // SAFETY: `$blk % 16 < 16` and `out_outer` holds 16 rows of 64 bytes.
                    let dst = unsafe { &mut *out.add(($blk % 16) * 64).cast::<[u8; 64]>() };
                    if V & ABV_INLINE_WINDOW != 0 {
                        bstatic_horner_blk_body::<$blk, V, HOIST>(
                            a_packed, b_packed, inv_table, chunk_byte_base, partials, kara, dst, nt_store,
                        );
                    } else {
                        bstatic_horner_blk::<$blk, V>(a_packed, b_packed, inv_table, chunk_byte_base, partials, kara, dst, nt_store);
                    }
                )*
            }};
        }
        if w == 0 {
            rows!(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        } else {
            rows!(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28);
        }
    }
}

/// [`bstatic_horner_window_unit_flag`] with the store flavour fixed for the
/// whole call: every row stores with `NT` (the `nt_store` the flag form
/// passes to each row), so no row tests a flag. The rows, the checks and the
/// fallback calls are those of the flag form.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn bstatic_horner_window_unit_nt<const V: u32, const HOIST: bool, const NT: bool>(
    x_outer: usize,
    window_mask: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: [&'static [[u8; 64]; 248]; 2],
    out_unit: &mut [u8],
) {
    let kara = if V & ABV_KARA != 0 {
        ab_kara_tables(inv_table)
    } else if V & ABV_HZ != 0 {
        ab_hz_tables(inv_table)
    } else {
        core::ptr::null()
    };
    for (i, out_outer) in out_unit.chunks_exact_mut(16 * 64).enumerate() {
        let w = (x_outer + i) & window_mask;
        let chunk_byte_base = chunk_byte_base + i * 16 * 64;
        assert!(w <= 1);
        let partials = partials[w];
        if HOIST {
            let end = chunk_byte_base + BSTATIC_HORNER_WINDOW_ROWS[w].1 * N_CHUNKS * 8;
            assert!(end <= a_packed.len() && end <= b_packed.len());
        }
        let out = out_outer.as_mut_ptr();
        macro_rules! rows {
            ($($blk:literal),*) => {{
                $(
                    // SAFETY: `$blk % 16 < 16` and `out_outer` holds 16 rows of 64 bytes.
                    let dst = unsafe { &mut *out.add(($blk % 16) * 64).cast::<[u8; 64]>() };
                    if const { V & ABV_INLINE_WINDOW != 0 } {
                        bstatic_horner_blk_body::<$blk, V, HOIST>(
                            a_packed, b_packed, inv_table, chunk_byte_base, partials, kara, dst, NT,
                        );
                    } else {
                        #[cfg(test)]
                        bstatic_horner_blk::<$blk, V>(a_packed, b_packed, inv_table, chunk_byte_base, partials, kara, dst, NT);
                        #[cfg(not(test))]
                        {
                            let _ = (partials, kara);
                            bstatic_horner_fallback(a_packed, b_packed, inv_table, chunk_byte_base, $blk % 16, dst, NT);
                        }
                    }
                )*
            }};
        }
        if w == 0 {
            rows!(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        } else {
            rows!(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28);
        }
    }
}

/// Plain-only split-Horner4 counterpart of [`bstatic_horner_window_unit_nt`]:
/// calls [`bstatic_horner_blk_body_split`] directly so non-split code is never
/// monomorphized into split unit instances.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn bstatic_horner_window_unit_split_nt<const V: u32, const HOIST: bool, const NT: bool>(
    x_outer: usize,
    window_mask: usize,
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    partials: [&'static [[u8; 64]; 248]; 2],
    out_unit: &mut [u8],
) {
    let kara = if V & ABV_KARA != 0 {
        ab_kara_tables(inv_table)
    } else if V & ABV_HZ != 0 {
        ab_hz_tables(inv_table)
    } else {
        core::ptr::null()
    };
    for (i, out_outer) in out_unit.chunks_exact_mut(16 * 64).enumerate() {
        let w = (x_outer + i) & window_mask;
        let chunk_byte_base = chunk_byte_base + i * 16 * 64;
        assert!(w <= 1);
        let partials = partials[w];
        if HOIST {
            let end = chunk_byte_base + BSTATIC_HORNER_WINDOW_ROWS[w].1 * N_CHUNKS * 8;
            assert!(end <= a_packed.len() && end <= b_packed.len());
        }
        let out = out_outer.as_mut_ptr();
        macro_rules! rows {
            ($($blk:literal),*) => {{
                $(
                    // SAFETY: `$blk % 16 < 16` and `out_outer` holds 16 rows of 64 bytes.
                    let dst = unsafe { &mut *out.add(($blk % 16) * 64).cast::<[u8; 64]>() };
                    bstatic_horner_blk_body_split::<$blk, V, HOIST>(
                        a_packed, b_packed, inv_table, chunk_byte_base, partials, kara, dst, NT,
                    );
                )*
            }};
        }
        if w == 0 {
            rows!(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        } else {
            rows!(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28);
        }
    }
}

/// [`bstatic_horner_window_unit_nt`] writes the bytes of
/// [`bstatic_horner_window_unit_flag`] for both store flavours, in the
/// instances of the main-pool workers and of the helper threads, at the
/// ranked unit geometry: units of eight chunks from window 0 and of three
/// chunks from window 1, in a unit scratch and at a chunk offset in a whole
/// buffer, with B rows that satisfy the static masks (every row on its
/// plan), random B rows (every row through the fallback) and B rows where
/// every other row satisfies them, with random and zero A rows. Rows
/// outside the windows keep their fill in both forms.
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_window_unit_nt_matches_flag_form() {
    use crate::ntt::AdditiveNttGf8;
    const OUTER_BYTES: usize = 16 * 64;
    const MAIN: u32 =
        AB_Z_VARIANT | ABV_INLINE_WINDOW | ABV_KARA_XOR16 | ABV_BSTATIC_Z | ABV_KARA_SLOT | ABV_HORNER2;
    const HELPER: u32 = AB_HZ_VARIANT | ABV_HORNER2;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let partials = bstatic_partials(&table);
    let mut state = 0xAB0F_1A60_57C0_0001u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut checked = 0usize;
    // (first chunk, chunks)
    for (x0, n) in [(8usize, 8usize), (5, 3)] {
        let total = (x0 + n) * OUTER_BYTES;
        for case in 0..4usize {
            let mut a: Vec<u8> = (0..total).map(|_| next() as u8).collect();
            let mut b: Vec<u8> = (0..total).map(|_| (next() >> 9) as u8).collect();
            if case != 1 {
                for chunk in 0..x0 + n {
                    let w = chunk & 1;
                    for b_med in 0..16usize {
                        if w * 16 + b_med > 30 || (case == 2 && b_med % 2 == 1) {
                            continue;
                        }
                        for k in 0..N_CHUNKS {
                            let off = chunk * OUTER_BYTES + b_med * 64 + k * N_CHUNKS;
                            let (mask, expected) = BSTATIC_MASKS[w * 16 + b_med][k];
                            let word = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
                            b[off..off + 8].copy_from_slice(&((word & !mask) | expected).to_le_bytes());
                        }
                    }
                }
            }
            if case == 3 {
                a.fill(0);
            }
            for in_scratch in [true, false] {
                let (a_in, b_in, base) = if in_scratch {
                    (&a[x0 * OUTER_BYTES..], &b[x0 * OUTER_BYTES..], 0)
                } else {
                    (&a[..], &b[..], x0 * OUTER_BYTES)
                };
                for nt in [false, true] {
                    macro_rules! pair {
                        ($v:expr, $hoist:literal) => {{
                            let mut want = vec![0x3cu8; n * OUTER_BYTES];
                            let mut got = vec![0x3cu8; n * OUTER_BYTES];
                            bstatic_horner_window_unit_flag::<{ $v }, $hoist>(
                                x0, 1, a_in, b_in, &table, base, [partials, partials], &mut want, nt,
                            );
                            if nt {
                                bstatic_horner_window_unit_nt::<{ $v }, $hoist, true>(
                                    x0, 1, a_in, b_in, &table, base, [partials, partials], &mut got,
                                );
                            } else {
                                bstatic_horner_window_unit_nt::<{ $v }, $hoist, false>(
                                    x0, 1, a_in, b_in, &table, base, [partials, partials], &mut got,
                                );
                            }
                            assert!(want.iter().any(|&x| x != 0x3c));
                            assert_eq!(got, want, "x0={x0} n={n} case={case} scratch={in_scratch} nt={nt}");
                            checked += 1;
                        }};
                    }
                    pair!(MAIN, true);
                    pair!(HELPER, false);
                }
            }
        }
    }
    assert_eq!(checked, 2 * 4 * 2 * 2 * 2);
}

/// The main-pool unit rows with [`ABV_Z_OUT3`] write the parent instance's
/// bytes for both store flavours, and [`bstatic_horner_window_unit_impl`]
/// writes them too (main-pool instance) or the flag form's (helper
/// instances): units of eight chunks from window 0, three from window 1 and
/// one from either window, in a unit scratch and at a chunk offset in a
/// whole buffer, with B rows that satisfy the static masks (every row on its
/// plan), random B rows (every row through the fallback) and B rows where
/// every other row satisfies them, with random, zero and all-ones A rows.
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_window_unit_z_out3_matches_parent() {
    use crate::ntt::AdditiveNttGf8;
    const OUTER_BYTES: usize = 16 * 64;
    const NEW: u32 = AB_UNIT_MAIN | ABV_Z_OUT3;
    const HELPER: u32 = AB_HZ_VARIANT | ABV_HORNER2;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let partials = bstatic_partials(&table);
    let mut state = 0x0E03_A0B7_7AB1_E5C3u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut checked = 0usize;
    // (first chunk, chunks)
    for (x0, n) in [(8usize, 8usize), (5, 3), (2, 1), (7, 1)] {
        let total = (x0 + n) * OUTER_BYTES;
        for case in 0..5usize {
            let mut a: Vec<u8> = (0..total).map(|_| next() as u8).collect();
            let mut b: Vec<u8> = (0..total).map(|_| (next() >> 9) as u8).collect();
            if case != 1 {
                for chunk in 0..x0 + n {
                    let w = chunk & 1;
                    for b_med in 0..16usize {
                        if w * 16 + b_med > 30 || (case == 2 && b_med % 2 == 1) {
                            continue;
                        }
                        for k in 0..N_CHUNKS {
                            let off = chunk * OUTER_BYTES + b_med * 64 + k * N_CHUNKS;
                            let (mask, expected) = BSTATIC_MASKS[w * 16 + b_med][k];
                            let word = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
                            b[off..off + 8].copy_from_slice(&((word & !mask) | expected).to_le_bytes());
                        }
                    }
                }
            }
            if case == 3 {
                a.fill(0);
            }
            if case == 4 {
                a.fill(0xff);
            }
            for in_scratch in [true, false] {
                let (a_in, b_in, base) = if in_scratch {
                    (&a[x0 * OUTER_BYTES..], &b[x0 * OUTER_BYTES..], 0)
                } else {
                    (&a[..], &b[..], x0 * OUTER_BYTES)
                };
                for nt in [false, true] {
                    let run = |f: &dyn Fn(&mut [u8])| {
                        let mut out = vec![0x3cu8; n * OUTER_BYTES];
                        f(&mut out);
                        out
                    };
                    let p = [partials, partials];
                    let (want, got) = if nt {
                        (
                            run(&|o| bstatic_horner_window_unit_nt::<AB_UNIT_MAIN, true, true>(x0, 1, a_in, b_in, &table, base, p, o)),
                            run(&|o| bstatic_horner_window_unit_nt::<NEW, true, true>(x0, 1, a_in, b_in, &table, base, p, o)),
                        )
                    } else {
                        (
                            run(&|o| bstatic_horner_window_unit_nt::<AB_UNIT_MAIN, true, false>(x0, 1, a_in, b_in, &table, base, p, o)),
                            run(&|o| bstatic_horner_window_unit_nt::<NEW, true, false>(x0, 1, a_in, b_in, &table, base, p, o)),
                        )
                    };
                    let via = run(&|o| bstatic_horner_window_unit_impl::<AB_UNIT_MAIN, true>(x0, 1, a_in, b_in, &table, base, p, o, nt));
                    let helper_want = run(&|o| bstatic_horner_window_unit_flag::<HELPER, false>(x0, 1, a_in, b_in, &table, base, p, o, nt));
                    let helper_got = run(&|o| bstatic_horner_window_unit_impl::<HELPER, false>(x0, 1, a_in, b_in, &table, base, p, o, nt));
                    let helper_hoisted = run(&|o| bstatic_horner_window_unit_impl::<HELPER, true>(x0, 1, a_in, b_in, &table, base, p, o, nt));
                    let at = format!("x0={x0} n={n} case={case} scratch={in_scratch} nt={nt}");
                    assert!(want.iter().any(|&x| x != 0x3c), "{at}");
                    assert_eq!(got, want, "{at}");
                    assert_eq!(via, want, "{at}");
                    assert_eq!(helper_got, helper_want, "{at}");
                    assert_eq!(helper_hoisted, helper_want, "hoisted {at}");
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 4 * 5 * 2 * 2);
}

/// The unit rows with [`ABV_FUSE`] on the main-pool and on the helper
/// instance write their
/// parent instance's bytes for both store flavours: units of eight chunks
/// from window 0, three from window 1 and one from either window, in a unit
/// scratch and at a chunk offset in a whole buffer, with B rows that satisfy
/// the static masks, random B rows (every row through the fallback) and B
/// rows where every other row satisfies them, with random, zero and
/// all-ones A rows.
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_window_unit_fuse_matches_parent() {
    use crate::ntt::AdditiveNttGf8;
    const OUTER_BYTES: usize = 16 * 64;
    const BASE: u32 = AB_UNIT_MAIN | ABV_Z_OUT3;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let partials = bstatic_partials(&table);
    let mut state = 0x5EED_0F_F05E_D00Du64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut checked = 0usize;
    for (x0, n) in [(8usize, 8usize), (5, 3), (2, 1), (7, 1)] {
        let total = (x0 + n) * OUTER_BYTES;
        for case in 0..5usize {
            let mut a: Vec<u8> = (0..total).map(|_| next() as u8).collect();
            let mut b: Vec<u8> = (0..total).map(|_| (next() >> 9) as u8).collect();
            if case != 1 {
                for chunk in 0..x0 + n {
                    let w = chunk & 1;
                    for b_med in 0..16usize {
                        if w * 16 + b_med > 30 || (case == 2 && b_med % 2 == 1) {
                            continue;
                        }
                        for k in 0..N_CHUNKS {
                            let off = chunk * OUTER_BYTES + b_med * 64 + k * N_CHUNKS;
                            let (mask, expected) = BSTATIC_MASKS[w * 16 + b_med][k];
                            let word = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
                            b[off..off + 8].copy_from_slice(&((word & !mask) | expected).to_le_bytes());
                        }
                    }
                }
            }
            if case == 3 {
                a.fill(0);
            }
            if case == 4 {
                a.fill(0xff);
            }
            for in_scratch in [true, false] {
                let (a_in, b_in, base) = if in_scratch {
                    (&a[x0 * OUTER_BYTES..], &b[x0 * OUTER_BYTES..], 0)
                } else {
                    (&a[..], &b[..], x0 * OUTER_BYTES)
                };
                let run = |f: &dyn Fn(&mut [u8])| {
                    let mut out = vec![0x3cu8; n * OUTER_BYTES];
                    f(&mut out);
                    out
                };
                let p = [partials, partials];
                let at = format!("x0={x0} n={n} case={case} scratch={in_scratch}");
                macro_rules! both {
                    ($v:expr, $hoist:literal) => {
                        (
                            run(&|o| bstatic_horner_window_unit_nt::<{ $v }, $hoist, false>(x0, 1, a_in, b_in, &table, base, p, o)),
                            run(&|o| bstatic_horner_window_unit_nt::<{ $v }, $hoist, true>(x0, 1, a_in, b_in, &table, base, p, o)),
                        )
                    };
                }
                let want = both!(BASE, true);
                assert!(want.0.iter().any(|&x| x != 0x3c), "{at}");
                assert_eq!(want.0, want.1, "{at}");
                assert_eq!(both!(BASE | ABV_FUSE, true), want, "FUSE {at}");
                let helper_want = both!(AB_UNIT_HELPER, false);
                assert_eq!(helper_want.0, want.0, "helper {at}");
                assert_eq!(both!(AB_UNIT_HELPER | ABV_FUSE, false), helper_want, "FUSE_E {at}");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 4 * 5 * 2);
}

/// The default main-pool unit instance (with [`ABV_TRANSFORM_TRIM`] and
/// [`ABV_SPARSE_SCALE`]) writes the bytes of the fused form without them,
/// with the trim only, and of the parent unit form, for both store
/// flavours, at the geometry of [`ab_window_unit_fuse_matches_parent`]: B
/// rows on their static plans with random varying bytes, with all varying
/// bytes zero and all 0xff, random B rows (every row through the fallback),
/// every other row on its plan, and random, zero and all-ones A rows.
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_window_unit_sparse_scale_matches_parent() {
    use crate::ntt::AdditiveNttGf8;
    const OUTER_BYTES: usize = 16 * 64;
    const FUSED: u32 = AB_UNIT_MAIN | ABV_Z_OUT3 | ABV_FUSE;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let partials = bstatic_partials(&table);
    let mut state = 0x5BA5_E5CA_1E00_0B07u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut checked = 0usize;
    for (x0, n) in [(8usize, 8usize), (5, 3), (2, 1), (7, 1)] {
        let total = (x0 + n) * OUTER_BYTES;
        for case in 0..7usize {
            let mut a: Vec<u8> = (0..total).map(|_| next() as u8).collect();
            let mut b: Vec<u8> = (0..total).map(|_| (next() >> 9) as u8).collect();
            if case != 1 {
                for chunk in 0..x0 + n {
                    let w = chunk & 1;
                    for b_med in 0..16usize {
                        if w * 16 + b_med > 30 || (case == 2 && b_med % 2 == 1) {
                            continue;
                        }
                        for k in 0..N_CHUNKS {
                            let off = chunk * OUTER_BYTES + b_med * 64 + k * N_CHUNKS;
                            let (mask, expected) = BSTATIC_MASKS[w * 16 + b_med][k];
                            let word = match case {
                                5 => 0,
                                6 => u64::MAX,
                                _ => u64::from_le_bytes(b[off..off + 8].try_into().unwrap()),
                            };
                            b[off..off + 8].copy_from_slice(&((word & !mask) | expected).to_le_bytes());
                        }
                    }
                }
            }
            if case == 3 {
                a.fill(0);
            }
            if case == 4 {
                a.fill(0xff);
            }
            for in_scratch in [true, false] {
                let (a_in, b_in, base) = if in_scratch {
                    (&a[x0 * OUTER_BYTES..], &b[x0 * OUTER_BYTES..], 0)
                } else {
                    (&a[..], &b[..], x0 * OUTER_BYTES)
                };
                let run = |f: &dyn Fn(&mut [u8])| {
                    let mut out = vec![0x3cu8; n * OUTER_BYTES];
                    f(&mut out);
                    out
                };
                let p = [partials, partials];
                let at = format!("x0={x0} n={n} case={case} scratch={in_scratch}");
                macro_rules! both {
                    ($v:expr) => {
                        (
                            run(&|o| bstatic_horner_window_unit_nt::<{ $v }, true, false>(x0, 1, a_in, b_in, &table, base, p, o)),
                            run(&|o| bstatic_horner_window_unit_nt::<{ $v }, true, true>(x0, 1, a_in, b_in, &table, base, p, o)),
                        )
                    };
                }
                let want = both!(FUSED);
                assert!(want.0.iter().any(|&x| x != 0x3c), "{at}");
                assert_eq!(want.0, want.1, "{at}");
                assert_eq!(both!(AB_UNIT_MAIN | ABV_Z_OUT3).0, want.0, "parent {at}");
                assert_eq!(both!(FUSED | ABV_TRANSFORM_TRIM), want, "TRIM {at}");
                assert_eq!(both!(FUSED | ABV_TRANSFORM_TRIM | ABV_SPARSE_SCALE), want, "TRIM SPARSE {at}");
                assert_eq!(both!(FUSED | ABV_TRANSFORM_TRIM | ABV_SPARSE_SCALE | ABV_KARA_LANEZ), want, "TRIM SPARSE LANEZ {at}");
                macro_rules! both_split {
                    ($v:expr) => {
                        (
                            run(&|o| bstatic_horner_window_unit_split_nt::<{ $v }, true, false>(x0, 1, a_in, b_in, &table, base, p, o)),
                            run(&|o| bstatic_horner_window_unit_split_nt::<{ $v }, true, true>(x0, 1, a_in, b_in, &table, base, p, o)),
                        )
                    };
                }
                assert_eq!(both_split!(FUSED | ABV_TRANSFORM_TRIM | ABV_SPLIT_H4), want, "TRIM SPLIT_H4 {at}");
                assert_eq!(
                    both!(AB_UNIT_HELPER | ABV_FUSE | ABV_TRANSFORM_TRIM),
                    want,
                    "helper TRIM {at}"
                );
                assert_eq!(
                    both_split!(AB_UNIT_HELPER | ABV_FUSE | ABV_TRANSFORM_TRIM | ABV_SPLIT_H4),
                    want,
                    "helper TRIM SPLIT_H4 {at}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 4 * 7 * 2);
}

/// [`ABV_SPARSE_SCALE`] on adversarial words, one K word at a time in every
/// row of both windows (all other A words zero or random, all other B words
/// all ones, which every guard admits): for every K, (1) A all ones with B
/// all ones and with B one bit off at both ends of the word's unchecked
/// bytes (the review's counterexample is BLK 20, K6: A = all ones,
/// B = 0xfefffffffeffffff, which the guard admits); (2) B one bit off in
/// each of the 64 positions (inside the unchecked bytes the row stays on its
/// plan, inside a checked byte the guard sends the row to the fallback),
/// with A all ones and with random A; (3) A a single bit in each of the 64
/// positions with B random on its plan. The sparse-scale unit writes the
/// bytes of the incumbent fused unit, the trimmed fused unit, the
/// helper-thread unit and the parent form, for both store flavours.
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_window_unit_sparse_scale_adversarial() {
    use crate::ntt::AdditiveNttGf8;
    const OUTER_BYTES: usize = 16 * 64;
    const FUSED: u32 = AB_UNIT_MAIN | ABV_Z_OUT3 | ABV_FUSE;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let partials = bstatic_partials(&table);
    let p = [partials, partials];
    let mut state = 0xADF0_5A11_C0DE_0018u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    // Unchecked bytes of word `k` of row `blk` (all bytes if the row has no plan).
    let free = |blk: usize, k: usize| -> u64 {
        if blk <= 30 { !BSTATIC_MASKS[blk][k].0 } else { u64::MAX }
    };
    // Two chunks: window 0, then window 1. `a_of`/`b_of` give word `k` of row `b_med`.
    let run_case = |label: &str, ksel: usize, a_of: &mut dyn FnMut(usize, usize) -> u64, b_of: &mut dyn FnMut(usize, usize) -> u64| -> [u8; 2 * OUTER_BYTES] {
        let mut a = vec![0u8; 2 * OUTER_BYTES];
        let mut b = vec![0u8; 2 * OUTER_BYTES];
        for chunk in 0..2usize {
            for b_med in 0..16usize {
                let blk = chunk * 16 + b_med;
                for k in 0..N_CHUNKS {
                    let off = chunk * OUTER_BYTES + b_med * 64 + k * N_CHUNKS;
                    a[off..off + 8].copy_from_slice(&a_of(blk, k).to_le_bytes());
                    b[off..off + 8].copy_from_slice(&b_of(blk, k).to_le_bytes());
                }
            }
        }
        let run = |f: &dyn Fn(&mut [u8])| {
            let mut out = vec![0x3cu8; 2 * OUTER_BYTES];
            f(&mut out);
            out
        };
        macro_rules! both {
            ($v:expr) => {
                (
                    run(&|o| bstatic_horner_window_unit_nt::<{ $v }, true, false>(0, 1, &a, &b, &table, 0, p, o)),
                    run(&|o| bstatic_horner_window_unit_nt::<{ $v }, true, true>(0, 1, &a, &b, &table, 0, p, o)),
                )
            };
        }
        let at = format!("{label} k={ksel}");
        let want = both!(FUSED);
        assert_eq!(want.0, want.1, "{at}");
        assert_eq!(both!(AB_UNIT_MAIN | ABV_Z_OUT3).0, want.0, "parent {at}");
        assert_eq!(both!(FUSED | ABV_TRANSFORM_TRIM), want, "TRIM {at}");
        assert_eq!(both!(FUSED | ABV_TRANSFORM_TRIM | ABV_SPARSE_SCALE), want, "TRIM SPARSE {at}");
        assert_eq!(both!(FUSED | ABV_TRANSFORM_TRIM | ABV_SPARSE_SCALE | ABV_KARA_LANEZ), want, "TRIM SPARSE LANEZ {at}");
        macro_rules! both_split {
            ($v:expr) => {
                (
                    run(&|o| bstatic_horner_window_unit_split_nt::<{ $v }, true, false>(0, 1, &a, &b, &table, 0, p, o)),
                    run(&|o| bstatic_horner_window_unit_split_nt::<{ $v }, true, true>(0, 1, &a, &b, &table, 0, p, o)),
                )
            };
        }
        assert_eq!(both_split!(FUSED | ABV_TRANSFORM_TRIM | ABV_SPLIT_H4), want, "TRIM SPLIT_H4 {at}");
        assert_eq!(both!(AB_UNIT_HELPER | ABV_FUSE | ABV_TRANSFORM_TRIM), want, "helper TRIM {at}");
        assert_eq!(both_split!(AB_UNIT_HELPER | ABV_FUSE | ABV_TRANSFORM_TRIM | ABV_SPLIT_H4), want, "helper TRIM SPLIT_H4 {at}");
        let mut out = [0u8; 2 * OUTER_BYTES];
        out.copy_from_slice(&want.0);
        out
    };
    let mut cases = 0usize;
    for ksel in 0..N_CHUNKS {
        // (1) A all ones; B all ones, and B one bit off at both ends of the unchecked bytes.
        run_case("ones", ksel, &mut |_, k| if k == ksel { u64::MAX } else { 0 }, &mut |_, _| u64::MAX);
        let ends = |blk: usize, k: usize| -> u64 {
            let f = free(blk, k);
            if f == 0 {
                return u64::MAX;
            }
            let lo = f.trailing_zeros() / 8;
            let hi = 7 - f.leading_zeros() / 8;
            !((1u64 << (8 * lo)) | (1u64 << (8 * hi)))
        };
        let out = run_case(
            "ends",
            ksel,
            &mut |_, k| if k == ksel { u64::MAX } else { 0 },
            &mut |blk, k| if k == ksel { ends(blk, k) } else { u64::MAX },
        );
        if ksel == 6 {
            // The review's counterexample row: BLK 20 = window 1, b_med 4.
            assert_eq!(ends(20, 6), 0xfeff_ffff_feff_ffff);
            eprintln!("BLK 20 K6 counterexample: row bytes 0..4 = {:02x?}", &out[OUTER_BYTES + 4 * 64..OUTER_BYTES + 4 * 64 + 4]);
        }
        cases += 2;
        // (2) B one bit off in every position, A all ones and random.
        for bit in 0..64u32 {
            for rand_a in [false, true] {
                let mut ra = || next();
                let ra_words: Vec<u64> = (0..2 * 16 * N_CHUNKS).map(|_| ra()).collect();
                run_case(
                    "b-bit",
                    ksel,
                    &mut |blk, k| {
                        if rand_a { ra_words[blk * N_CHUNKS + k] } else if k == ksel { u64::MAX } else { 0 }
                    },
                    &mut |_, k| if k == ksel { !(1u64 << bit) } else { u64::MAX },
                );
                cases += 1;
            }
        }
        // (3) A a single bit in every position, B random on its plan.
        for bit in 0..64u32 {
            let rb: Vec<u64> = (0..2 * 16 * N_CHUNKS).map(|_| next()).collect();
            run_case(
                "a-bit",
                ksel,
                &mut |_, k| if k == ksel { 1u64 << bit } else { 0 },
                &mut |blk, k| {
                    let f = free(blk, k);
                    if k == ksel { (rb[blk * N_CHUNKS + k] & f) | !f } else { u64::MAX }
                },
            );
            cases += 1;
        }
    }
    assert_eq!(cases, N_CHUNKS * (2 + 128 + 64));
}

/// The default main-pool unit instance (with [`ABV_TRANSFORM_TRIM`] and
/// [`ABV_KARA_LANEZ`]) writes the bytes of the fused form without them, with
/// the trim only and with lane z only, and of the parent unit form, for both
/// store flavours, at the geometry and inputs of
/// [`ab_window_unit_fuse_matches_parent`].
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_window_unit_lanez_matches_parent() {
    use crate::ntt::AdditiveNttGf8;
    const OUTER_BYTES: usize = 16 * 64;
    const FUSED: u32 = AB_UNIT_MAIN | ABV_Z_OUT3 | ABV_FUSE;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let partials = bstatic_partials(&table);
    let mut state = 0x1A2E_5EED_0BAD_CAFEu64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut checked = 0usize;
    for (x0, n) in [(8usize, 8usize), (5, 3), (2, 1), (7, 1)] {
        let total = (x0 + n) * OUTER_BYTES;
        for case in 0..5usize {
            let mut a: Vec<u8> = (0..total).map(|_| next() as u8).collect();
            let mut b: Vec<u8> = (0..total).map(|_| (next() >> 9) as u8).collect();
            if case != 1 {
                for chunk in 0..x0 + n {
                    let w = chunk & 1;
                    for b_med in 0..16usize {
                        if w * 16 + b_med > 30 || (case == 2 && b_med % 2 == 1) {
                            continue;
                        }
                        for k in 0..N_CHUNKS {
                            let off = chunk * OUTER_BYTES + b_med * 64 + k * N_CHUNKS;
                            let (mask, expected) = BSTATIC_MASKS[w * 16 + b_med][k];
                            let word = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
                            b[off..off + 8].copy_from_slice(&((word & !mask) | expected).to_le_bytes());
                        }
                    }
                }
            }
            if case == 3 {
                a.fill(0);
            }
            if case == 4 {
                a.fill(0xff);
            }
            for in_scratch in [true, false] {
                let (a_in, b_in, base) = if in_scratch {
                    (&a[x0 * OUTER_BYTES..], &b[x0 * OUTER_BYTES..], 0)
                } else {
                    (&a[..], &b[..], x0 * OUTER_BYTES)
                };
                let run = |f: &dyn Fn(&mut [u8])| {
                    let mut out = vec![0x3cu8; n * OUTER_BYTES];
                    f(&mut out);
                    out
                };
                let p = [partials, partials];
                let at = format!("x0={x0} n={n} case={case} scratch={in_scratch}");
                macro_rules! both {
                    ($v:expr) => {
                        (
                            run(&|o| bstatic_horner_window_unit_nt::<{ $v }, true, false>(x0, 1, a_in, b_in, &table, base, p, o)),
                            run(&|o| bstatic_horner_window_unit_nt::<{ $v }, true, true>(x0, 1, a_in, b_in, &table, base, p, o)),
                        )
                    };
                }
                let want = both!(FUSED);
                assert!(want.0.iter().any(|&x| x != 0x3c), "{at}");
                assert_eq!(want.0, want.1, "{at}");
                assert_eq!(both!(AB_UNIT_MAIN | ABV_Z_OUT3).0, want.0, "parent {at}");
                assert_eq!(both!(FUSED | ABV_KARA_LANEZ), want, "LANEZ {at}");
                assert_eq!(both!(FUSED | ABV_TRANSFORM_TRIM), want, "TRIM {at}");
                assert_eq!(both!(FUSED | ABV_TRANSFORM_TRIM | ABV_KARA_LANEZ), want, "TRIM LANEZ {at}");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 4 * 5 * 2);
}

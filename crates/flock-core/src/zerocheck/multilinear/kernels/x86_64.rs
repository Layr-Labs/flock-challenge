use crate::field::F128;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use crate::field::F256Unreduced;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use crate::field::gf2_128::x86_64::{WideGhashX4, f128x4_loadu};

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn fold_round2_pair_x86_unchecked_8(
    table_data: *const F128,
    a0_bytes: *const u8,
    a1_bytes: *const u8,
    b0_bytes: *const u8,
    b1_bytes: *const u8,
) -> [F128; 4] {
    use core::arch::x86_64::*;

    unsafe {
        let rows = [a0_bytes, a1_bytes, b0_bytes, b1_bytes];
        let mut acc = [_mm_setzero_si128(); 4];
        for chunk in 0..8 {
            let table_chunk = table_data.add(chunk * 256);
            for lane in 0..4 {
                let entry = table_chunk.add(*rows[lane].add(chunk) as usize);
                acc[lane] = _mm_xor_si128(acc[lane], _mm_load_si128(entry.cast::<__m128i>()));
            }
        }
        acc.map(|value| core::mem::transmute::<__m128i, F128>(value))
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn fold_round3_pair_scaled_x86_unchecked_8(
    table0_data: *const F128,
    table1_data: *const F128,
    a_bytes: *const u8,
    b_bytes: *const u8,
) -> (F128, F128) {
    use core::arch::x86_64::*;

    unsafe {
        let mut acc_a = _mm_setzero_si128();
        let mut acc_b = _mm_setzero_si128();
        for chunk in 0..8 {
            let t0 = table0_data.add(chunk * 256);
            let t1 = table1_data.add(chunk * 256);
            let a0_v = _mm_load_si128(t0.add(*a_bytes.add(chunk) as usize).cast::<__m128i>());
            let a1_v = _mm_load_si128(t1.add(*a_bytes.add(8 + chunk) as usize).cast::<__m128i>());
            let b0_v = _mm_load_si128(t0.add(*b_bytes.add(chunk) as usize).cast::<__m128i>());
            let b1_v = _mm_load_si128(t1.add(*b_bytes.add(8 + chunk) as usize).cast::<__m128i>());
            acc_a = _mm_xor_si128(acc_a, _mm_xor_si128(a0_v, a1_v));
            acc_b = _mm_xor_si128(acc_b, _mm_xor_si128(b0_v, b1_v));
        }
        (
            core::mem::transmute::<__m128i, F128>(acc_a),
            core::mem::transmute::<__m128i, F128>(acc_b),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub(crate) unsafe fn round2_msg_only_chunk_x86_unchecked_8(
    table_data: *const F128,
    a_pkt_ptr: *const u8,
    b_pkt_ptr: *const u8,
    eq_lo: &[F128],
    base: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    use core::arch::x86_64::*;

    unsafe {
        let mut p1_ll = _mm_setzero_si128();
        let mut p1_mid = _mm_setzero_si128();
        let mut p1_hh = _mm_setzero_si128();
        let mut pinf_ll = _mm_setzero_si128();
        let mut pinf_mid = _mm_setzero_si128();
        let mut pinf_hh = _mm_setzero_si128();
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr();

        macro_rules! round2_general_step {
            ($x_lo:expr) => {{
                let xl = $x_lo;
                let x0g = base + 2 * xl;
                let ap = a_pkt_ptr.add(x0g * 8);
                let bp = b_pkt_ptr.add(x0g * 8);
                let a0_u64 = (ap as *const u64).read_unaligned();
                let a1_u64 = (ap.add(8) as *const u64).read_unaligned();
                let b0_u64 = (bp as *const u64).read_unaligned();
                let b1_u64 = (bp.add(8) as *const u64).read_unaligned();
                let ad_u64 = a0_u64 ^ a1_u64;
                let bd_u64 = b0_u64 ^ b1_u64;

                let mut va1 =
                    _mm_load_si128(table_data.add((a1_u64 & 0xFF) as usize).cast::<__m128i>());
                let mut vad =
                    _mm_load_si128(table_data.add((ad_u64 & 0xFF) as usize).cast::<__m128i>());
                let mut vb1 =
                    _mm_load_si128(table_data.add((b1_u64 & 0xFF) as usize).cast::<__m128i>());
                let mut vbd =
                    _mm_load_si128(table_data.add((bd_u64 & 0xFF) as usize).cast::<__m128i>());

                macro_rules! fold_step {
                    ($c:literal) => {{
                        let tc = table_data.add($c * 256);
                        let sh = $c * 8;
                        va1 = _mm_xor_si128(
                            va1,
                            _mm_load_si128(
                                tc.add(((a1_u64 >> sh) & 0xFF) as usize).cast::<__m128i>(),
                            ),
                        );
                        vad = _mm_xor_si128(
                            vad,
                            _mm_load_si128(
                                tc.add(((ad_u64 >> sh) & 0xFF) as usize).cast::<__m128i>(),
                            ),
                        );
                        vb1 = _mm_xor_si128(
                            vb1,
                            _mm_load_si128(
                                tc.add(((b1_u64 >> sh) & 0xFF) as usize).cast::<__m128i>(),
                            ),
                        );
                        vbd = _mm_xor_si128(
                            vbd,
                            _mm_load_si128(
                                tc.add(((bd_u64 >> sh) & 0xFF) as usize).cast::<__m128i>(),
                            ),
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

                let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_m128i(va1, vad, vb1, vbd);

                let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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

        let v_all_ones = _mm_xor_si128(
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(2 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(3 * 256 + 255).cast::<__m128i>()),
                ),
            ),
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(4 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(5 * 256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(6 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(7 * 256 + 255).cast::<__m128i>()),
                ),
            ),
        );
        let is_unit_ones =
            (_mm_extract_epi64::<0>(v_all_ones) == 1) && (_mm_extract_epi64::<1>(v_all_ones) == 0);

        if is_unit_ones
            && lo_size >= 128
            && (lo_size & 127) == 0
            && (pair_idx_base & 127) == 0
            && pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
        {
            macro_rules! round2_unit_b_step {
                ($x_lo:expr) => {{
                    let xl = $x_lo;
                    let x0g = base + 2 * xl;
                    let a1_u64 = (a_pkt_ptr.add(x0g * 8 + 8) as *const u64).read_unaligned();
                    let s0 = _mm_xor_si128(
                        _mm_load_si128(table_data.add((a1_u64 & 0xFF) as usize).cast::<__m128i>()),
                        _mm_load_si128(
                            table_data
                                .add(256 + (((a1_u64 >> 8) & 0xFF) as usize))
                                .cast::<__m128i>(),
                        ),
                    );
                    let s1 = _mm_xor_si128(
                        _mm_load_si128(
                            table_data
                                .add(512 + (((a1_u64 >> 16) & 0xFF) as usize))
                                .cast::<__m128i>(),
                        ),
                        _mm_load_si128(
                            table_data
                                .add(768 + (((a1_u64 >> 24) & 0xFF) as usize))
                                .cast::<__m128i>(),
                        ),
                    );
                    let s2 = _mm_xor_si128(
                        _mm_load_si128(
                            table_data
                                .add(1024 + (((a1_u64 >> 32) & 0xFF) as usize))
                                .cast::<__m128i>(),
                        ),
                        _mm_load_si128(
                            table_data
                                .add(1280 + (((a1_u64 >> 40) & 0xFF) as usize))
                                .cast::<__m128i>(),
                        ),
                    );
                    let s3 = _mm_xor_si128(
                        _mm_load_si128(
                            table_data
                                .add(1536 + (((a1_u64 >> 48) & 0xFF) as usize))
                                .cast::<__m128i>(),
                        ),
                        _mm_load_si128(
                            table_data
                                .add(1792 + (((a1_u64 >> 56) & 0xFF) as usize))
                                .cast::<__m128i>(),
                        ),
                    );
                    let va1 = _mm_xor_si128(_mm_xor_si128(s0, s1), _mm_xor_si128(s2, s3));
                    let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                    let veq_sum = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4E>(veq));
                    let va1_sum = _mm_xor_si128(va1, _mm_shuffle_epi32::<0x4E>(va1));
                    p1_ll = _mm_xor_si128(p1_ll, _mm_clmulepi64_si128::<0x00>(veq, va1));
                    p1_mid = _mm_xor_si128(p1_mid, _mm_clmulepi64_si128::<0x00>(veq_sum, va1_sum));
                    p1_hh = _mm_xor_si128(p1_hh, _mm_clmulepi64_si128::<0x11>(veq, va1));
                }};
            }

            let n_blocks = lo_size >> 7;
            for blk in 0..n_blocks {
                let blk_x = blk << 7;
                let bp_blk = b_pkt_ptr.add((base + 2 * blk_x) * 8) as *const u64;
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
                    for x_lo in blk_x..(blk_x + 9) {
                        round2_unit_b_step!(x_lo);
                    }
                } else {
                    for x_lo in blk_x..(blk_x + 9) {
                        round2_general_step!(x_lo);
                    }
                }
                for x_lo in (blk_x + 9)..(blk_x + 118) {
                    round2_general_step!(x_lo);
                }
                if tail_ones {
                    for x_lo in (blk_x + 118)..(blk_x + 120) {
                        round2_unit_b_step!(x_lo);
                    }
                } else {
                    for x_lo in (blk_x + 118)..(blk_x + 120) {
                        round2_general_step!(x_lo);
                    }
                }
                {
                    let xl = blk_x + 120;
                    let x0g = base + 2 * xl;
                    let ap = a_pkt_ptr.add(x0g * 8);
                    let bp = b_pkt_ptr.add(x0g * 8);
                    let a1_u64 = (ap.add(8) as *const u64).read_unaligned();
                    let b1_u64 = (bp.add(8) as *const u64).read_unaligned();
                    if (a1_u64 | b1_u64) == 0 {
                        let a0_u64 = (ap as *const u64).read_unaligned();
                        let b0_u64 = (bp as *const u64).read_unaligned();
                        let mut va0 = _mm_load_si128(
                            table_data.add((a0_u64 & 0xFF) as usize).cast::<__m128i>(),
                        );
                        let mut vb0 = _mm_load_si128(
                            table_data.add((b0_u64 & 0xFF) as usize).cast::<__m128i>(),
                        );
                        macro_rules! fold_step_ab2 {
                            ($c:literal) => {{
                                let tc = table_data.add($c * 256);
                                let sh = $c * 8;
                                va0 = _mm_xor_si128(
                                    va0,
                                    _mm_load_si128(
                                        tc.add(((a0_u64 >> sh) & 0xFF) as usize).cast::<__m128i>(),
                                    ),
                                );
                                vb0 = _mm_xor_si128(
                                    vb0,
                                    _mm_load_si128(
                                        tc.add(((b0_u64 >> sh) & 0xFF) as usize).cast::<__m128i>(),
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
                        let ginf = crate::gf2_kernels::k0_f128_mul_m128i(va0, vb0);
                        let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                        crate::gf2_kernels::k0_f128_mul_acc_m128i(
                            veq,
                            ginf,
                            &mut pinf_ll,
                            &mut pinf_mid,
                            &mut pinf_hh,
                        );
                    } else {
                        round2_general_step!(xl);
                    }
                }
            }
        } else {
            for x_lo in 0..lo_size {
                if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                    continue;
                }
                round2_general_step!(x_lo);
            }
        }

        let res1 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(p1_ll, p1_mid, p1_hh);
        let resi = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(pinf_ll, pinf_mid, pinf_hh);
        (
            core::mem::transmute::<__m128i, F128>(res1),
            core::mem::transmute::<__m128i, F128>(resi),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub(crate) unsafe fn pre_reduce_eq_lo_x86(eq_lo: &[F128]) -> Vec<[F128; 2]> {
    let n = eq_lo.len();
    let mut out = Vec::<[F128; 2]>::with_capacity(n);
    unsafe {
        crate::gf2_kernels::k0_f128_pre_reduce_eq_lo_slice(
            eq_lo.as_ptr() as *const u64,
            out.as_mut_ptr() as *mut u64,
            n,
        );
        out.set_len(n);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn round23_lookahead_chunk_x86_unchecked_8(
    table_data: *const F128,
    a_pkt_ptr: *const u8,
    b_pkt_ptr: *const u8,
    eq_lo: &[F128],
    eq_lo_prered: &[[F128; 2]],
    q_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [F128; 8] {
    use core::arch::x86_64::*;

    unsafe {
        let z = _mm_setzero_si128();
        let mut acc = [[z; 3]; 8];
        let mut p240_ll = z;
        let mut p240_mid = z;
        let mut p240_hh = z;
        let lo_size = eq_lo.len();
        let _eq_ptr = eq_lo.as_ptr();
        let eq_prered_ptr = eq_lo_prered.as_ptr();
        let v_all_ones = _mm_xor_si128(
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(2 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(3 * 256 + 255).cast::<__m128i>()),
                ),
            ),
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(4 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(5 * 256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(6 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(7 * 256 + 255).cast::<__m128i>()),
                ),
            ),
        );
        let is_unit_ones =
            (_mm_extract_epi64::<0>(v_all_ones) == 1) && (_mm_extract_epi64::<1>(v_all_ones) == 0);
        let use_gfni = crate::gf2_kernels::k0_has_avx512_gfni();
        let mut c1_mats = [0u64; 128];
        if use_gfni {
            for j in 0..8usize {
                let tbl = table_data.add(j * 256);
                let mut basis = [F128::ZERO; 8];
                for b in 0..8usize {
                    basis[b] = *tbl.add(1usize << b);
                }
                crate::gf2_kernels::k0_f128_transpose_8_to_16_gfni_mats(
                    basis.as_ptr() as *const u64,
                    c1_mats.as_mut_ptr().add(16 * j),
                );
            }
        }

        macro_rules! fold_table_4 {
            ($w0:expr, $w1:expr, $w2:expr, $w3:expr) => {{
                let w0 = $w0;
                let w1 = $w1;
                let w2 = $w2;
                let w3 = $w3;
                let mut v0 = _mm_load_si128(table_data.add((w0 & 0xFF) as usize).cast::<__m128i>());
                let mut v1 = _mm_load_si128(table_data.add((w1 & 0xFF) as usize).cast::<__m128i>());
                let mut v2 = _mm_load_si128(table_data.add((w2 & 0xFF) as usize).cast::<__m128i>());
                let mut v3 = _mm_load_si128(table_data.add((w3 & 0xFF) as usize).cast::<__m128i>());
                macro_rules! step4 {
                    ($c:literal) => {{
                        let tc = table_data.add($c * 256);
                        let sh = $c * 8;
                        v0 = _mm_xor_si128(
                            v0,
                            _mm_load_si128(tc.add(((w0 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v1 = _mm_xor_si128(
                            v1,
                            _mm_load_si128(tc.add(((w1 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v2 = _mm_xor_si128(
                            v2,
                            _mm_load_si128(tc.add(((w2 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v3 = _mm_xor_si128(
                            v3,
                            _mm_load_si128(tc.add(((w3 >> sh) & 0xFF) as usize).cast::<__m128i>()),
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
                (v0, v1, v2, v3)
            }};
        }

        macro_rules! fold_table_3 {
            ($w0:expr, $w1:expr, $w2:expr) => {{
                let w0 = $w0;
                let w1 = $w1;
                let w2 = $w2;
                let mut v0 = _mm_load_si128(table_data.add((w0 & 0xFF) as usize).cast::<__m128i>());
                let mut v1 = _mm_load_si128(table_data.add((w1 & 0xFF) as usize).cast::<__m128i>());
                let mut v2 = _mm_load_si128(table_data.add((w2 & 0xFF) as usize).cast::<__m128i>());
                macro_rules! step3 {
                    ($c:literal) => {{
                        let tc = table_data.add($c * 256);
                        let sh = $c * 8;
                        v0 = _mm_xor_si128(
                            v0,
                            _mm_load_si128(tc.add(((w0 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v1 = _mm_xor_si128(
                            v1,
                            _mm_load_si128(tc.add(((w1 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v2 = _mm_xor_si128(
                            v2,
                            _mm_load_si128(tc.add(((w2 >> sh) & 0xFF) as usize).cast::<__m128i>()),
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
                (v0, v1, v2)
            }};
        }

        macro_rules! acc_unred_slot_with_sum {
            ($idx:literal, $u:expr, $v:expr, $u_sum:expr, $v_sum:expr) => {{
                let u = $u;
                let v = $v;
                acc[$idx][0] = _mm_xor_si128(acc[$idx][0], _mm_clmulepi64_si128::<0x00>(u, v));
                acc[$idx][1] =
                    _mm_xor_si128(acc[$idx][1], _mm_clmulepi64_si128::<0x00>($u_sum, $v_sum));
                acc[$idx][2] = _mm_xor_si128(acc[$idx][2], _mm_clmulepi64_si128::<0x11>(u, v));
            }};
        }

        macro_rules! round23_from_tb {
            ($va00:expr, $va10:expr, $va01:expr, $va11:expr, $tb00:expr, $tb10:expr, $tb01:expr, $tb11:expr) => {{
                let sva10 = _mm_xor_si128($va10, _mm_shuffle_epi32::<0x4E>($va10));
                let stb10 = _mm_xor_si128($tb10, _mm_shuffle_epi32::<0x4E>($tb10));
                acc_unred_slot_with_sum!(0, $va10, $tb10, sva10, stb10);

                let da0 = _mm_xor_si128($va00, $va10);
                let dtb0 = _mm_xor_si128($tb00, $tb10);
                let sda0 = _mm_xor_si128(da0, _mm_shuffle_epi32::<0x4E>(da0));
                let sdtb0 = _mm_xor_si128(dtb0, _mm_shuffle_epi32::<0x4E>(dtb0));
                acc_unred_slot_with_sum!(1, da0, dtb0, sda0, sdtb0);

                let sva11 = _mm_xor_si128($va11, _mm_shuffle_epi32::<0x4E>($va11));
                let stb11 = _mm_xor_si128($tb11, _mm_shuffle_epi32::<0x4E>($tb11));
                acc_unred_slot_with_sum!(2, $va11, $tb11, sva11, stb11);

                let sva01 = _mm_xor_si128($va01, _mm_shuffle_epi32::<0x4E>($va01));
                let stb01 = _mm_xor_si128($tb01, _mm_shuffle_epi32::<0x4E>($tb01));
                acc_unred_slot_with_sum!(4, $va01, $tb01, sva01, stb01);

                let da_0 = _mm_xor_si128($va00, $va01);
                let dtb_0 = _mm_xor_si128($tb00, $tb01);
                let sda_0 = _mm_xor_si128(sda0, _mm_xor_si128(sva10, sva01));
                let sdtb_0 = _mm_xor_si128(sdtb0, _mm_xor_si128(stb10, stb01));
                acc_unred_slot_with_sum!(5, da_0, dtb_0, sda_0, sdtb_0);

                let da_1 = _mm_xor_si128($va10, $va11);
                let dtb_1 = _mm_xor_si128($tb10, $tb11);
                let sda_1 = _mm_xor_si128(sva10, sva11);
                let sdtb_1 = _mm_xor_si128(stb10, stb11);
                acc_unred_slot_with_sum!(6, da_1, dtb_1, sda_1, sdtb_1);

                let da1 = _mm_xor_si128($va01, $va11);
                let dtb1 = _mm_xor_si128($tb01, $tb11);
                let sda1 = _mm_xor_si128(sva01, sva11);
                let sdtb1 = _mm_xor_si128(stb01, stb11);
                acc_unred_slot_with_sum!(3, da1, dtb1, sda1, sdtb1);

                let da_i = _mm_xor_si128(da0, da1);
                let dtb_i = _mm_xor_si128(dtb0, dtb1);
                let sda_i = _mm_xor_si128(sda0, sda1);
                let sdtb_i = _mm_xor_si128(sdtb0, sdtb1);
                acc_unred_slot_with_sum!(7, da_i, dtb_i, sda_i, sdtb_i);
            }};
        }

        macro_rules! round23_general_from_folded {
            ($q_lo:expr, $va00:expr, $vb00:expr, $va10:expr, $vb10:expr, $va01:expr, $vb01:expr, $va11:expr, $vb11:expr) => {{
                let eq_pair = eq_prered_ptr.add($q_lo) as *const __m128i;
                let eq_r0 = _mm_loadu_si128(eq_pair);
                let eq_r1 = _mm_loadu_si128(eq_pair.add(1));
                let (tb00, tb10) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    $vb00, $vb10, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                let (tb01, tb11) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    $vb01, $vb11, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                round23_from_tb!($va00, $va10, $va01, $va11, tb00, tb10, tb01, tb11);
            }};
        }

        macro_rules! round23_general_step {
            ($q_lo:expr) => {{
                let ql = $q_lo;
                let qg = q_base + ql;
                let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                let bp = (b_pkt_ptr as *const u64).add(4 * qg);
                let (va00, vb00, va10, vb10) = fold_table_4!(
                    ap.read_unaligned(),
                    bp.read_unaligned(),
                    ap.add(1).read_unaligned(),
                    bp.add(1).read_unaligned()
                );
                let eq_pair = eq_prered_ptr.add(ql) as *const __m128i;
                let eq_r0 = _mm_loadu_si128(eq_pair);
                let eq_r1 = _mm_loadu_si128(eq_pair.add(1));
                let (tb00, tb10) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    vb00, vb10, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                let (va01, vb01, va11, vb11) = fold_table_4!(
                    ap.add(2).read_unaligned(),
                    bp.add(2).read_unaligned(),
                    ap.add(3).read_unaligned(),
                    bp.add(3).read_unaligned()
                );
                let (tb01, tb11) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    vb01, vb11, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                round23_from_tb!(va00, va10, va01, va11, tb00, tb10, tb01, tb11);
            }};
        }

        macro_rules! round23_unit_b_step {
            ($q_lo:expr) => {{
                let ql = $q_lo;
                let qg = q_base + ql;
                let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                let (va10, va01, va11) = fold_table_3!(
                    ap.add(1).read_unaligned(),
                    ap.add(2).read_unaligned(),
                    ap.add(3).read_unaligned()
                );
                let eq_pair = eq_prered_ptr.add(ql) as *const __m128i;
                let veq =
                    _mm_unpacklo_epi64(_mm_loadu_si128(eq_pair), _mm_loadu_si128(eq_pair.add(1)));
                let veq_sum = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4E>(veq));

                let s10_sum = _mm_xor_si128(va10, _mm_shuffle_epi32::<0x4E>(va10));
                acc[0][0] = _mm_xor_si128(acc[0][0], _mm_clmulepi64_si128::<0x00>(veq, va10));
                acc[0][1] =
                    _mm_xor_si128(acc[0][1], _mm_clmulepi64_si128::<0x00>(veq_sum, s10_sum));
                acc[0][2] = _mm_xor_si128(acc[0][2], _mm_clmulepi64_si128::<0x11>(veq, va10));

                let s11_sum = _mm_xor_si128(va11, _mm_shuffle_epi32::<0x4E>(va11));
                acc[2][0] = _mm_xor_si128(acc[2][0], _mm_clmulepi64_si128::<0x00>(veq, va11));
                acc[2][1] =
                    _mm_xor_si128(acc[2][1], _mm_clmulepi64_si128::<0x00>(veq_sum, s11_sum));
                acc[2][2] = _mm_xor_si128(acc[2][2], _mm_clmulepi64_si128::<0x11>(veq, va11));

                let s01_sum = _mm_xor_si128(va01, _mm_shuffle_epi32::<0x4E>(va01));
                acc[4][0] = _mm_xor_si128(acc[4][0], _mm_clmulepi64_si128::<0x00>(veq, va01));
                acc[4][1] =
                    _mm_xor_si128(acc[4][1], _mm_clmulepi64_si128::<0x00>(veq_sum, s01_sum));
                acc[4][2] = _mm_xor_si128(acc[4][2], _mm_clmulepi64_si128::<0x11>(veq, va01));
            }};
        }

        if is_unit_ones
            && lo_size >= 64
            && (lo_size & 63) == 0
            && (q_base & 63) == 0
            && pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
        {
            let n_blocks = lo_size >> 6;
            for blk in 0..n_blocks {
                let blk_q = blk << 6;
                let bp_blk = (b_pkt_ptr as *const u64).add(4 * (q_base + blk_q));
                let tail_ones = (bp_blk.add(236).read_unaligned()
                    & bp_blk.add(237).read_unaligned()
                    & bp_blk.add(238).read_unaligned()
                    & bp_blk.add(239).read_unaligned())
                    == !0u64;
                if use_gfni {
                    let mut a_buf = [z; 64];
                    let mut b_buf = [z; 64];
                    for batch in 0..3usize {
                        let q_batch_lo = blk_q + 16 * batch;
                        let q_batch_g = q_base + q_batch_lo;
                        crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_slice(
                            (a_pkt_ptr as *const u64).add(4 * q_batch_g) as *const u8,
                            c1_mats.as_ptr(),
                            a_buf.as_mut_ptr() as *mut u64,
                        );
                        crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_slice(
                            (b_pkt_ptr as *const u64).add(4 * q_batch_g) as *const u8,
                            c1_mats.as_ptr(),
                            b_buf.as_mut_ptr() as *mut u64,
                        );
                        for t in 0..16usize {
                            let ql = q_batch_lo + t;
                            round23_general_from_folded!(
                                ql,
                                a_buf[4 * t],
                                b_buf[4 * t],
                                a_buf[4 * t + 1],
                                b_buf[4 * t + 1],
                                a_buf[4 * t + 2],
                                b_buf[4 * t + 2],
                                a_buf[4 * t + 3],
                                b_buf[4 * t + 3]
                            );
                        }
                    }
                    for q_lo in (blk_q + 48)..(blk_q + 59) {
                        round23_general_step!(q_lo);
                    }
                } else {
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
                        let mut va00 =
                            _mm_load_si128(table_data.add((a00 & 0xFF) as usize).cast::<__m128i>());
                        let mut vb00 =
                            _mm_load_si128(table_data.add((b00 & 0xFF) as usize).cast::<__m128i>());
                        macro_rules! step2 {
                            ($c:literal) => {{
                                let tc = table_data.add($c * 256);
                                let sh = $c * 8;
                                va00 = _mm_xor_si128(
                                    va00,
                                    _mm_load_si128(
                                        tc.add(((a00 >> sh) & 0xFF) as usize).cast::<__m128i>(),
                                    ),
                                );
                                vb00 = _mm_xor_si128(
                                    vb00,
                                    _mm_load_si128(
                                        tc.add(((b00 >> sh) & 0xFF) as usize).cast::<__m128i>(),
                                    ),
                                );
                            }};
                        }
                        step2!(1);
                        step2!(2);
                        step2!(3);
                        if ((a00 | b00) >> 32) != 0 {
                            step2!(4);
                            step2!(5);
                            step2!(6);
                            step2!(7);
                        }
                        let g00 = crate::gf2_kernels::k0_f128_mul_m128i(va00, vb00);
                        let eq_pair = eq_prered_ptr.add(ql) as *const __m128i;
                        let veq = _mm_unpacklo_epi64(
                            _mm_loadu_si128(eq_pair),
                            _mm_loadu_si128(eq_pair.add(1)),
                        );
                        crate::gf2_kernels::k0_f128_mul_acc_m128i(
                            veq,
                            g00,
                            &mut p240_ll,
                            &mut p240_mid,
                            &mut p240_hh,
                        );
                    } else {
                        let (va00, vb00, va10, vb10) =
                            fold_table_4!(ap.read_unaligned(), bp.read_unaligned(), a10, b10);
                        round23_general_from_folded!(ql, va00, vb00, va10, vb10, z, z, z, z);
                    }
                }
            }
            acc[1][0] = _mm_xor_si128(acc[1][0], p240_ll);
            acc[1][1] = _mm_xor_si128(acc[1][1], p240_mid);
            acc[1][2] = _mm_xor_si128(acc[1][2], p240_hh);
            acc[5][0] = _mm_xor_si128(acc[5][0], p240_ll);
            acc[5][1] = _mm_xor_si128(acc[5][1], p240_mid);
            acc[5][2] = _mm_xor_si128(acc[5][2], p240_hh);
            acc[7][0] = _mm_xor_si128(acc[7][0], p240_ll);
            acc[7][1] = _mm_xor_si128(acc[7][1], p240_mid);
            acc[7][2] = _mm_xor_si128(acc[7][2], p240_hh);
        } else {
            let mut q_lo = 0usize;
            if use_gfni {
                let mut a_buf = [z; 64];
                let mut b_buf = [z; 64];
                while q_lo + 16 <= lo_size {
                    let q_batch_g = q_base + q_lo;
                    let all_active = (0..16usize).all(|t| {
                        let j1 = 2 * (q_batch_g + t) + 1;
                        (j1 & pair_in_block_mask) < useful_pairs_inclusive
                    });
                    if !all_active {
                        break;
                    }
                    crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_slice(
                        (a_pkt_ptr as *const u64).add(4 * q_batch_g) as *const u8,
                        c1_mats.as_ptr(),
                        a_buf.as_mut_ptr() as *mut u64,
                    );
                    crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_slice(
                        (b_pkt_ptr as *const u64).add(4 * q_batch_g) as *const u8,
                        c1_mats.as_ptr(),
                        b_buf.as_mut_ptr() as *mut u64,
                    );
                    for t in 0..16usize {
                        let ql = q_lo + t;
                        round23_general_from_folded!(
                            ql,
                            a_buf[4 * t],
                            b_buf[4 * t],
                            a_buf[4 * t + 1],
                            b_buf[4 * t + 1],
                            a_buf[4 * t + 2],
                            b_buf[4 * t + 2],
                            a_buf[4 * t + 3],
                            b_buf[4 * t + 3]
                        );
                    }
                    q_lo += 16;
                }
            }
            while q_lo < lo_size {
                let qg = q_base + q_lo;
                let j0 = 2 * qg;
                let j1 = j0 + 1;
                if (j0 & pair_in_block_mask) >= useful_pairs_inclusive {
                    q_lo += 1;
                    continue;
                }
                let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                let bp = (b_pkt_ptr as *const u64).add(4 * qg);
                let (va00, vb00, va10, vb10) = fold_table_4!(
                    ap.read_unaligned(),
                    bp.read_unaligned(),
                    ap.add(1).read_unaligned(),
                    bp.add(1).read_unaligned()
                );
                let (va01, vb01, va11, vb11) = if (j1 & pair_in_block_mask) < useful_pairs_inclusive
                {
                    fold_table_4!(
                        ap.add(2).read_unaligned(),
                        bp.add(2).read_unaligned(),
                        ap.add(3).read_unaligned(),
                        bp.add(3).read_unaligned()
                    )
                } else {
                    (z, z, z, z)
                };
                round23_general_from_folded!(q_lo, va00, vb00, va10, vb10, va01, vb01, va11, vb11);
                q_lo += 1;
            }
        }

        let mut out = [F128::ZERO; 8];
        for k in 0..8 {
            let r = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(acc[k][0], acc[k][1], acc[k][2]);
            out[k] = core::mem::transmute::<__m128i, F128>(r);
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn round4_fold_2vars_into_chunk_x86_unchecked_8(
    table_data: *const F128,
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
    use_stream: bool,
) -> (F128, F128) {
    use core::arch::x86_64::*;

    unsafe {
        let z = _mm_setzero_si128();
        let mut p1_ll = z;
        let mut p1_mid = z;
        let mut p1_hh = z;
        let mut pinf_ll = z;
        let mut pinf_mid = z;
        let mut pinf_hh = z;
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr();
        let a_out_ptr = a_out_chunk.as_mut_ptr();
        let b_out_ptr = b_out_chunk.as_mut_ptr();

        let v_rho0 = _mm_set_epi64x(rho_0.hi as i64, rho_0.lo as i64);
        let v_rho1 = _mm_set_epi64x(rho_1.hi as i64, rho_1.lo as i64);
        let v_rho01_sums =
            _mm_set_epi64x((rho_1.lo ^ rho_1.hi) as i64, (rho_0.lo ^ rho_0.hi) as i64);
        let rho_01 = rho_0 * rho_1;
        let v_rho01 = _mm_set_epi64x(rho_01.hi as i64, rho_01.lo as i64);
        let v_rho01_sum = _mm_xor_si128(v_rho01, _mm_shuffle_epi32::<0x4E>(v_rho01));
        let opr01 = (F128::ONE + rho_0) * (F128::ONE + rho_1);
        let v_opr01 = _mm_set_epi64x(opr01.hi as i64, opr01.lo as i64);
        let (opr01_r0, opr01_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(v_opr01);

        let v_all_ones = _mm_xor_si128(
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(2 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(3 * 256 + 255).cast::<__m128i>()),
                ),
            ),
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(4 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(5 * 256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(6 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(7 * 256 + 255).cast::<__m128i>()),
                ),
            ),
        );
        let is_unit_ones =
            (_mm_extract_epi64::<0>(v_all_ones) == 1) && (_mm_extract_epi64::<1>(v_all_ones) == 0);
        let use_gfni = crate::gf2_kernels::k0_has_avx512_gfni();
        let mut bcast_mats = [[0u64; 8]; 64];
        if use_gfni {
            let coeffs: [F128; 4] = [opr01, rho_0 + rho_01, rho_1 + rho_01, rho_01];
            crate::gf2_kernels::k0_f128_build_gf2_c4_fold_mats_gfni_slice(
                table_data as *const u64,
                coeffs.as_ptr() as *const u64,
                bcast_mats.as_mut_ptr() as *mut u64,
            );
        }

        macro_rules! fold_table_4 {
            ($w0:expr, $w1:expr, $w2:expr, $w3:expr) => {{
                let w0 = $w0;
                let w1 = $w1;
                let w2 = $w2;
                let w3 = $w3;
                let mut v0 = _mm_load_si128(table_data.add((w0 & 0xFF) as usize).cast::<__m128i>());
                let mut v1 = _mm_load_si128(table_data.add((w1 & 0xFF) as usize).cast::<__m128i>());
                let mut v2 = _mm_load_si128(table_data.add((w2 & 0xFF) as usize).cast::<__m128i>());
                let mut v3 = _mm_load_si128(table_data.add((w3 & 0xFF) as usize).cast::<__m128i>());
                macro_rules! step4 {
                    ($c:literal) => {{
                        let tc = table_data.add($c * 256);
                        let sh = $c * 8;
                        v0 = _mm_xor_si128(
                            v0,
                            _mm_load_si128(tc.add(((w0 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v1 = _mm_xor_si128(
                            v1,
                            _mm_load_si128(tc.add(((w1 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v2 = _mm_xor_si128(
                            v2,
                            _mm_load_si128(tc.add(((w2 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v3 = _mm_xor_si128(
                            v3,
                            _mm_load_si128(tc.add(((w3 >> sh) & 0xFF) as usize).cast::<__m128i>()),
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
                (v0, v1, v2, v3)
            }};
        }

        macro_rules! combine_2vars_from_folded {
            ($f00:expr, $f10:expr, $f01:expr, $f11:expr) => {{
                let f10 = $f10;
                let f01 = $f01;
                let f11 = $f11;
                let f10_01_sum =
                    _mm_xor_si128(_mm_unpacklo_epi64(f10, f01), _mm_unpackhi_epi64(f10, f01));
                let f11_sum = _mm_xor_si128(f11, _mm_shuffle_epi32::<0x4E>(f11));
                let ll = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>(f10, v_rho0),
                        _mm_clmulepi64_si128::<0x00>(f01, v_rho1),
                    ),
                    _mm_clmulepi64_si128::<0x00>(f11, v_rho01),
                );
                let mid = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>(f10_01_sum, v_rho01_sums),
                        _mm_clmulepi64_si128::<0x11>(f10_01_sum, v_rho01_sums),
                    ),
                    _mm_clmulepi64_si128::<0x00>(f11_sum, v_rho01_sum),
                );
                let hh = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x11>(f10, v_rho0),
                        _mm_clmulepi64_si128::<0x11>(f01, v_rho1),
                    ),
                    _mm_clmulepi64_si128::<0x11>(f11, v_rho01),
                );
                _mm_xor_si128(
                    $f00,
                    crate::gf2_kernels::k0_f128_reduce_3lane_m128i(ll, mid, hh),
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
                let (f00, f10, f01, f11) = fold_table_4!(w00, u10, u01, u11);
                combine_2vars_from_folded!(f00, f10, f01, f11)
            }};
        }

        macro_rules! fold_2vars_ab_one {
            ($ap:expr, $bp:expr) => {{ (fold_2vars_one!($ap), fold_2vars_one!($bp)) }};
        }

        macro_rules! fold_2vars_a_pair_unit_b {
            ($x_lo:expr, $store_fn:ident) => {{
                let xl = $x_lo;
                let q0l = 2 * xl;
                let q1l = q0l + 1;
                let q0g = q_base + q0l;
                let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                let va_q0 = fold_2vars_one!(ap0);
                let va_q1 = fold_2vars_one!(ap0.add(4));

                $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va_q0);
                $store_fn(a_out_ptr.add(q1l).cast::<__m128i>(), va_q1);
                $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), v_all_ones);
                $store_fn(b_out_ptr.add(q1l).cast::<__m128i>(), v_all_ones);

                let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                let veq_sum = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4E>(veq));
                let va1_sum = _mm_xor_si128(va_q1, _mm_shuffle_epi32::<0x4E>(va_q1));
                p1_ll = _mm_xor_si128(p1_ll, _mm_clmulepi64_si128::<0x00>(veq, va_q1));
                p1_mid = _mm_xor_si128(p1_mid, _mm_clmulepi64_si128::<0x00>(veq_sum, va1_sum));
                p1_hh = _mm_xor_si128(p1_hh, _mm_clmulepi64_si128::<0x11>(veq, va_q1));
            }};
        }

        macro_rules! round4_general_step {
            ($x_lo:expr, $store_fn:ident) => {{
                let xl = $x_lo;
                let q0l = 2 * xl;
                let q1l = q0l + 1;
                let q0g = q_base + q0l;
                let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                let bp0 = (b_pkt_ptr as *const u64).add(4 * q0g);
                let (va_q0, vb_q0) = fold_2vars_ab_one!(ap0, bp0);
                let (va_q1, vb_q1) = fold_2vars_ab_one!(ap0.add(4), bp0.add(4));

                $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va_q0);
                $store_fn(a_out_ptr.add(q1l).cast::<__m128i>(), va_q1);
                $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), vb_q0);
                $store_fn(b_out_ptr.add(q1l).cast::<__m128i>(), vb_q1);

                let vad = _mm_xor_si128(va_q0, va_q1);
                let vbd = _mm_xor_si128(vb_q0, vb_q1);
                let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_m128i(va_q1, vad, vb_q1, vbd);
                let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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

        macro_rules! run_round4_body {
            ($store_fn:ident) => {{
                if is_unit_ones
                    && lo_size >= 32
                    && (lo_size & 31) == 0
                    && (q_base & 63) == 0
                    && pair_in_block_mask == 127
                    && useful_pairs_inclusive == 121
                {
                    let n_blocks = lo_size >> 5;
                    if use_gfni {
                        let mats_ptr = bcast_mats.as_ptr() as *const u64;
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
                            for b_idx in 0..4usize {
                                let q_start_l = 2 * blk_x + 16 * b_idx;
                                let q_start_g = q_base + q_start_l;
                                let dead_lines: u8 = if b_idx == 3 { 0xC0 } else { 0 };
                                crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_c4_slice(
                                    a_pkt_ptr.add(32 * q_start_g),
                                    mats_ptr,
                                    a_out_ptr.add(q_start_l) as *mut u64,
                                    dead_lines,
                                );
                                crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_c4_slice(
                                    b_pkt_ptr.add(32 * q_start_g),
                                    mats_ptr,
                                    b_out_ptr.add(q_start_l) as *mut u64,
                                    dead_lines,
                                );
                            }
                            let x_start = if head_ones {
                                for xl in [blk_x, blk_x + 1] {
                                    let q1l = 2 * xl + 1;
                                    let va_q1 =
                                        _mm_loadu_si128(a_out_ptr.add(q1l).cast::<__m128i>());
                                    let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                                    crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                        veq,
                                        va_q1,
                                        &mut p1_ll,
                                        &mut p1_mid,
                                        &mut p1_hh,
                                    );
                                }
                                blk_x + 2
                            } else {
                                blk_x
                            };
                            let x_end = if tail_ones { blk_x + 29 } else { blk_x + 30 };
                            for xl in x_start..x_end {
                                let q0l = 2 * xl;
                                let q1l = q0l + 1;
                                let va_q0 = _mm_loadu_si128(a_out_ptr.add(q0l).cast::<__m128i>());
                                let va_q1 = _mm_loadu_si128(a_out_ptr.add(q1l).cast::<__m128i>());
                                let vb_q0 = _mm_loadu_si128(b_out_ptr.add(q0l).cast::<__m128i>());
                                let vb_q1 = _mm_loadu_si128(b_out_ptr.add(q1l).cast::<__m128i>());
                                let vad = _mm_xor_si128(va_q0, va_q1);
                                let vbd = _mm_xor_si128(vb_q0, vb_q1);
                                let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_m128i(
                                    va_q1, vad, vb_q1, vbd,
                                );
                                let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                                crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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
                            if tail_ones {
                                let xl = blk_x + 29;
                                let q0l = 2 * xl;
                                let q1l = q0l + 1;
                                let va_q0 = _mm_loadu_si128(a_out_ptr.add(q0l).cast::<__m128i>());
                                let va_q1 = _mm_loadu_si128(a_out_ptr.add(q1l).cast::<__m128i>());
                                let vb_q0 = _mm_loadu_si128(b_out_ptr.add(q0l).cast::<__m128i>());
                                let vb_q1 = v_all_ones;
                                let vad = _mm_xor_si128(va_q0, va_q1);
                                let vbd = _mm_xor_si128(vb_q0, vb_q1);
                                let ginf = crate::gf2_kernels::k0_f128_mul_m128i(vad, vbd);
                                let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                                crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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
                            }
                            {
                                let xl = blk_x + 30;
                                let q0l = 2 * xl;
                                let q0g = q_base + q0l;
                                let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                                let bp0 = (b_pkt_ptr as *const u64).add(4 * q0g);
                                let a10 = ap0.add(1).read_unaligned();
                                let b10 = bp0.add(1).read_unaligned();
                                let (va_q0, vb_q0) = if (a10 | b10) == 0 {
                                    let a00 = ap0.read_unaligned();
                                    let b00 = bp0.read_unaligned();
                                    let mut va0 = _mm_load_si128(
                                        table_data.add((a00 & 0xFF) as usize).cast::<__m128i>(),
                                    );
                                    let mut vb0 = _mm_load_si128(
                                        table_data.add((b00 & 0xFF) as usize).cast::<__m128i>(),
                                    );
                                    macro_rules! step2 {
                                        ($c:literal) => {{
                                            let tc = table_data.add($c * 256);
                                            let sh = $c * 8;
                                            va0 = _mm_xor_si128(
                                                va0,
                                                _mm_load_si128(
                                                    tc.add(((a00 >> sh) & 0xFF) as usize)
                                                        .cast::<__m128i>(),
                                                ),
                                            );
                                            vb0 = _mm_xor_si128(
                                                vb0,
                                                _mm_load_si128(
                                                    tc.add(((b00 >> sh) & 0xFF) as usize)
                                                        .cast::<__m128i>(),
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
                                    crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                        va0, vb0, opr01_r0, opr01_r1, opr01_r0, opr01_r1,
                                    )
                                } else {
                                    fold_2vars_ab_one!(ap0, bp0)
                                };
                                _mm_storeu_si128(a_out_ptr.add(q0l).cast::<__m128i>(), va_q0);
                                _mm_storeu_si128(b_out_ptr.add(q0l).cast::<__m128i>(), vb_q0);
                                let ginf = crate::gf2_kernels::k0_f128_mul_m128i(va_q0, vb_q0);
                                let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                                crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                    veq,
                                    ginf,
                                    &mut pinf_ll,
                                    &mut pinf_mid,
                                    &mut pinf_hh,
                                );
                            }
                        }
                    } else {
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
                                fold_2vars_a_pair_unit_b!(blk_x, $store_fn);
                                fold_2vars_a_pair_unit_b!(blk_x + 1, $store_fn);
                            } else {
                                round4_general_step!(blk_x, $store_fn);
                                round4_general_step!(blk_x + 1, $store_fn);
                            }
                            for x_lo in (blk_x + 2)..(blk_x + 29) {
                                round4_general_step!(x_lo, $store_fn);
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
                                let vb_q1 = v_all_ones;

                                $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va_q0);
                                $store_fn(a_out_ptr.add(q1l).cast::<__m128i>(), va_q1);
                                $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), vb_q0);
                                $store_fn(b_out_ptr.add(q1l).cast::<__m128i>(), vb_q1);

                                let vad = _mm_xor_si128(va_q0, va_q1);
                                let vbd = _mm_xor_si128(vb_q0, vb_q1);
                                let ginf = crate::gf2_kernels::k0_f128_mul_m128i(vad, vbd);
                                let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                                crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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
                                round4_general_step!(blk_x + 29, $store_fn);
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
                                    let mut va0 = _mm_load_si128(
                                        table_data.add((a00 & 0xFF) as usize).cast::<__m128i>(),
                                    );
                                    let mut vb0 = _mm_load_si128(
                                        table_data.add((b00 & 0xFF) as usize).cast::<__m128i>(),
                                    );
                                    macro_rules! step2 {
                                        ($c:literal) => {{
                                            let tc = table_data.add($c * 256);
                                            let sh = $c * 8;
                                            va0 = _mm_xor_si128(
                                                va0,
                                                _mm_load_si128(
                                                    tc.add(((a00 >> sh) & 0xFF) as usize)
                                                        .cast::<__m128i>(),
                                                ),
                                            );
                                            vb0 = _mm_xor_si128(
                                                vb0,
                                                _mm_load_si128(
                                                    tc.add(((b00 >> sh) & 0xFF) as usize)
                                                        .cast::<__m128i>(),
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
                                    crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                        va0, vb0, opr01_r0, opr01_r1, opr01_r0, opr01_r1,
                                    )
                                } else {
                                    fold_2vars_ab_one!(ap0, bp0)
                                };
                                $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va_q0);
                                $store_fn(a_out_ptr.add(q1l).cast::<__m128i>(), z);
                                $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), vb_q0);
                                $store_fn(b_out_ptr.add(q1l).cast::<__m128i>(), z);
                                let ginf = crate::gf2_kernels::k0_f128_mul_m128i(va_q0, vb_q0);
                                let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                                crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                    veq,
                                    ginf,
                                    &mut pinf_ll,
                                    &mut pinf_mid,
                                    &mut pinf_hh,
                                );
                            }
                            {
                                let xl = blk_x + 31;
                                let q0l = 2 * xl;
                                let q1l = q0l + 1;
                                $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), z);
                                $store_fn(a_out_ptr.add(q1l).cast::<__m128i>(), z);
                                $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), z);
                                $store_fn(b_out_ptr.add(q1l).cast::<__m128i>(), z);
                            }
                        }
                    }
                } else if use_gfni && pair_in_block_mask == 0 && (lo_size & 7) == 0 {
                    let mats_ptr = bcast_mats.as_ptr() as *const u64;
                    for blk8 in 0..(lo_size >> 3) {
                        let x_start = blk8 << 3;
                        let q_start_l = 2 * x_start;
                        let q_start_g = q_base + q_start_l;
                        crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_c4_slice(
                            a_pkt_ptr.add(32 * q_start_g),
                            mats_ptr,
                            a_out_ptr.add(q_start_l) as *mut u64,
                            0,
                        );
                        crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_c4_slice(
                            b_pkt_ptr.add(32 * q_start_g),
                            mats_ptr,
                            b_out_ptr.add(q_start_l) as *mut u64,
                            0,
                        );
                        for x_lo in x_start..(x_start + 8) {
                            let q0l = 2 * x_lo;
                            let q1l = q0l + 1;
                            let va_q0 = _mm_loadu_si128(a_out_ptr.add(q0l).cast::<__m128i>());
                            let va_q1 = _mm_loadu_si128(a_out_ptr.add(q1l).cast::<__m128i>());
                            let vb_q0 = _mm_loadu_si128(b_out_ptr.add(q0l).cast::<__m128i>());
                            let vb_q1 = _mm_loadu_si128(b_out_ptr.add(q1l).cast::<__m128i>());
                            let vad = _mm_xor_si128(va_q0, va_q1);
                            let vbd = _mm_xor_si128(vb_q0, vb_q1);
                            let (g1, ginf) =
                                crate::gf2_kernels::k0_f128_mul_vec2_m128i(va_q1, vad, vb_q1, vbd);
                            let veq = _mm_loadu_si128(eq_ptr.add(x_lo).cast::<__m128i>());
                            crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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
                } else {
                    for x_lo in 0..lo_size {
                        let q0l = 2 * x_lo;
                        let q1l = q0l + 1;
                        let q0g = q_base + q0l;
                        let q1g = q0g + 1;

                        let fold_one_padded = |qg: usize| -> (__m128i, __m128i) {
                            let j0 = 2 * qg;
                            let j1 = j0 + 1;
                            if (j0 & pair_in_block_mask) >= useful_pairs_inclusive {
                                return (z, z);
                            }
                            let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                            let bp = (b_pkt_ptr as *const u64).add(4 * qg);
                            let a00 = ap.read_unaligned();
                            let a10 = ap.add(1).read_unaligned();
                            let b00 = bp.read_unaligned();
                            let b10 = bp.add(1).read_unaligned();
                            let (a01, a11, b01, b11) =
                                if (j1 & pair_in_block_mask) < useful_pairs_inclusive {
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
                            let (fa00, fa10, fa01, fa11) = fold_table_4!(a00, ua10, ua01, ua11);
                            let (fb00, fb10, fb01, fb11) = fold_table_4!(b00, ub10, ub01, ub11);
                            (
                                combine_2vars_from_folded!(fa00, fa10, fa01, fa11),
                                combine_2vars_from_folded!(fb00, fb10, fb01, fb11),
                            )
                        };

                        let (va_q0, vb_q0) = fold_one_padded(q0g);
                        let (va_q1, vb_q1) = fold_one_padded(q1g);
                        $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va_q0);
                        $store_fn(a_out_ptr.add(q1l).cast::<__m128i>(), va_q1);
                        $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), vb_q0);
                        $store_fn(b_out_ptr.add(q1l).cast::<__m128i>(), vb_q1);

                        let vad = _mm_xor_si128(va_q0, va_q1);
                        let vbd = _mm_xor_si128(vb_q0, vb_q1);
                        let (g1, ginf) =
                            crate::gf2_kernels::k0_f128_mul_vec2_m128i(va_q1, vad, vb_q1, vbd);
                        let veq = _mm_loadu_si128(eq_ptr.add(x_lo).cast::<__m128i>());
                        crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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
            }};
        }

        if use_stream {
            run_round4_body!(_mm_stream_si128);
            _mm_sfence();
        } else {
            run_round4_body!(_mm_storeu_si128);
        }

        let res1 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(p1_ll, p1_mid, p1_hh);
        let resi = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(pinf_ll, pinf_mid, pinf_hh);
        (
            core::mem::transmute::<__m128i, F128>(res1),
            core::mem::transmute::<__m128i, F128>(resi),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn round45_fold_2vars_into_and_lookahead_chunk_x86_unchecked_8(
    table_data: *const F128,
    rho_0: F128,
    rho_1: F128,
    a_pkt_ptr: *const u8,
    b_pkt_ptr: *const u8,
    a_out_chunk: &mut [F128],
    b_out_chunk: &mut [F128],
    eq_lo: &[F128],
    eq_lo_prered: &[[F128; 2]],
    q_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    use_stream: bool,
) -> [F128; 8] {
    use core::arch::x86_64::*;

    unsafe {
        let z = _mm_setzero_si128();
        let mut acc = [[z; 3]; 8];
        let mut p60_ll = z;
        let mut p60_mid = z;
        let mut p60_hh = z;
        let lo_size = eq_lo.len();
        let _eq_ptr = eq_lo.as_ptr();
        let eq_prered_ptr = eq_lo_prered.as_ptr();
        let a_out_ptr = a_out_chunk.as_mut_ptr();
        let b_out_ptr = b_out_chunk.as_mut_ptr();

        let v_rho0 = _mm_set_epi64x(rho_0.hi as i64, rho_0.lo as i64);
        let v_rho1 = _mm_set_epi64x(rho_1.hi as i64, rho_1.lo as i64);
        let v_rho01_sums =
            _mm_set_epi64x((rho_1.lo ^ rho_1.hi) as i64, (rho_0.lo ^ rho_0.hi) as i64);
        let rho_01 = rho_0 * rho_1;
        let v_rho01 = _mm_set_epi64x(rho_01.hi as i64, rho_01.lo as i64);
        let v_rho01_sum = _mm_xor_si128(v_rho01, _mm_shuffle_epi32::<0x4E>(v_rho01));
        let opr01 = (F128::ONE + rho_0) * (F128::ONE + rho_1);
        let v_opr01 = _mm_set_epi64x(opr01.hi as i64, opr01.lo as i64);
        let (opr01_r0, opr01_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(v_opr01);

        let v_all_ones = _mm_xor_si128(
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(2 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(3 * 256 + 255).cast::<__m128i>()),
                ),
            ),
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(4 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(5 * 256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(6 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(7 * 256 + 255).cast::<__m128i>()),
                ),
            ),
        );
        let is_unit_ones =
            (_mm_extract_epi64::<0>(v_all_ones) == 1) && (_mm_extract_epi64::<1>(v_all_ones) == 0);
        let use_gfni = crate::gf2_kernels::k0_has_avx512_gfni();
        let mut bcast_mats = [[0u64; 8]; 64];
        if use_gfni {
            let coeffs: [F128; 4] = [opr01, rho_0 + rho_01, rho_1 + rho_01, rho_01];
            crate::gf2_kernels::k0_f128_build_gf2_c4_fold_mats_gfni_slice(
                table_data as *const u64,
                coeffs.as_ptr() as *const u64,
                bcast_mats.as_mut_ptr() as *mut u64,
            );
        }

        macro_rules! fold_table_4 {
            ($w0:expr, $w1:expr, $w2:expr, $w3:expr) => {{
                let w0 = $w0;
                let w1 = $w1;
                let w2 = $w2;
                let w3 = $w3;
                let mut v0 = _mm_load_si128(table_data.add((w0 & 0xFF) as usize).cast::<__m128i>());
                let mut v1 = _mm_load_si128(table_data.add((w1 & 0xFF) as usize).cast::<__m128i>());
                let mut v2 = _mm_load_si128(table_data.add((w2 & 0xFF) as usize).cast::<__m128i>());
                let mut v3 = _mm_load_si128(table_data.add((w3 & 0xFF) as usize).cast::<__m128i>());
                macro_rules! step4 {
                    ($c:literal) => {{
                        let tc = table_data.add($c * 256);
                        let sh = $c * 8;
                        v0 = _mm_xor_si128(
                            v0,
                            _mm_load_si128(tc.add(((w0 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v1 = _mm_xor_si128(
                            v1,
                            _mm_load_si128(tc.add(((w1 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v2 = _mm_xor_si128(
                            v2,
                            _mm_load_si128(tc.add(((w2 >> sh) & 0xFF) as usize).cast::<__m128i>()),
                        );
                        v3 = _mm_xor_si128(
                            v3,
                            _mm_load_si128(tc.add(((w3 >> sh) & 0xFF) as usize).cast::<__m128i>()),
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
                (v0, v1, v2, v3)
            }};
        }

        macro_rules! combine_2vars_from_folded {
            ($f00:expr, $f10:expr, $f01:expr, $f11:expr) => {{
                let f10 = $f10;
                let f01 = $f01;
                let f11 = $f11;
                let f10_01_sum =
                    _mm_xor_si128(_mm_unpacklo_epi64(f10, f01), _mm_unpackhi_epi64(f10, f01));
                let f11_sum = _mm_xor_si128(f11, _mm_shuffle_epi32::<0x4E>(f11));
                let ll = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>(f10, v_rho0),
                        _mm_clmulepi64_si128::<0x00>(f01, v_rho1),
                    ),
                    _mm_clmulepi64_si128::<0x00>(f11, v_rho01),
                );
                let mid = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>(f10_01_sum, v_rho01_sums),
                        _mm_clmulepi64_si128::<0x11>(f10_01_sum, v_rho01_sums),
                    ),
                    _mm_clmulepi64_si128::<0x00>(f11_sum, v_rho01_sum),
                );
                let hh = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x11>(f10, v_rho0),
                        _mm_clmulepi64_si128::<0x11>(f01, v_rho1),
                    ),
                    _mm_clmulepi64_si128::<0x11>(f11, v_rho01),
                );
                _mm_xor_si128(
                    $f00,
                    crate::gf2_kernels::k0_f128_reduce_3lane_m128i(ll, mid, hh),
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
                let (f00, f10, f01, f11) = fold_table_4!(w00, u10, u01, u11);
                combine_2vars_from_folded!(f00, f10, f01, f11)
            }};
        }

        macro_rules! fold_2vars_ab_one {
            ($ap:expr, $bp:expr) => {{ (fold_2vars_one!($ap), fold_2vars_one!($bp)) }};
        }

        macro_rules! acc_unred_slot_with_sum {
            ($idx:literal, $u:expr, $v:expr, $u_sum:expr, $v_sum:expr) => {{
                let u = $u;
                let v = $v;
                acc[$idx][0] = _mm_xor_si128(acc[$idx][0], _mm_clmulepi64_si128::<0x00>(u, v));
                acc[$idx][1] =
                    _mm_xor_si128(acc[$idx][1], _mm_clmulepi64_si128::<0x00>($u_sum, $v_sum));
                acc[$idx][2] = _mm_xor_si128(acc[$idx][2], _mm_clmulepi64_si128::<0x11>(u, v));
            }};
        }

        macro_rules! lookahead_from_tb {
            ($va00:expr, $va10:expr, $va01:expr, $va11:expr, $tb00:expr, $tb10:expr, $tb01:expr, $tb11:expr) => {{
                let sva10 = _mm_xor_si128($va10, _mm_shuffle_epi32::<0x4E>($va10));
                let stb10 = _mm_xor_si128($tb10, _mm_shuffle_epi32::<0x4E>($tb10));
                acc_unred_slot_with_sum!(0, $va10, $tb10, sva10, stb10);

                let da0 = _mm_xor_si128($va00, $va10);
                let dtb0 = _mm_xor_si128($tb00, $tb10);
                let sda0 = _mm_xor_si128(da0, _mm_shuffle_epi32::<0x4E>(da0));
                let sdtb0 = _mm_xor_si128(dtb0, _mm_shuffle_epi32::<0x4E>(dtb0));
                acc_unred_slot_with_sum!(1, da0, dtb0, sda0, sdtb0);

                let sva11 = _mm_xor_si128($va11, _mm_shuffle_epi32::<0x4E>($va11));
                let stb11 = _mm_xor_si128($tb11, _mm_shuffle_epi32::<0x4E>($tb11));
                acc_unred_slot_with_sum!(2, $va11, $tb11, sva11, stb11);

                let sva01 = _mm_xor_si128($va01, _mm_shuffle_epi32::<0x4E>($va01));
                let stb01 = _mm_xor_si128($tb01, _mm_shuffle_epi32::<0x4E>($tb01));
                acc_unred_slot_with_sum!(4, $va01, $tb01, sva01, stb01);

                let da_0 = _mm_xor_si128($va00, $va01);
                let dtb_0 = _mm_xor_si128($tb00, $tb01);
                let sda_0 = _mm_xor_si128(sda0, _mm_xor_si128(sva10, sva01));
                let sdtb_0 = _mm_xor_si128(sdtb0, _mm_xor_si128(stb10, stb01));
                acc_unred_slot_with_sum!(5, da_0, dtb_0, sda_0, sdtb_0);

                let da_1 = _mm_xor_si128($va10, $va11);
                let dtb_1 = _mm_xor_si128($tb10, $tb11);
                let sda_1 = _mm_xor_si128(sva10, sva11);
                let sdtb_1 = _mm_xor_si128(stb10, stb11);
                acc_unred_slot_with_sum!(6, da_1, dtb_1, sda_1, sdtb_1);

                let da1 = _mm_xor_si128($va01, $va11);
                let dtb1 = _mm_xor_si128($tb01, $tb11);
                let sda1 = _mm_xor_si128(sva01, sva11);
                let sdtb1 = _mm_xor_si128(stb01, stb11);
                acc_unred_slot_with_sum!(3, da1, dtb1, sda1, sdtb1);

                let da_i = _mm_xor_si128(da0, da1);
                let dtb_i = _mm_xor_si128(dtb0, dtb1);
                let sda_i = _mm_xor_si128(sda0, sda1);
                let sdtb_i = _mm_xor_si128(sdtb0, sdtb1);
                acc_unred_slot_with_sum!(7, da_i, dtb_i, sda_i, sdtb_i);
            }};
        }

        macro_rules! lookahead_from_folded {
            ($p_lo:expr, $va00:expr, $vb00:expr, $va10:expr, $vb10:expr, $va01:expr, $vb01:expr, $va11:expr, $vb11:expr) => {{
                let eq_pair = eq_prered_ptr.add($p_lo) as *const __m128i;
                let eq_r0 = _mm_loadu_si128(eq_pair);
                let eq_r1 = _mm_loadu_si128(eq_pair.add(1));
                let (tb00, tb10) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    $vb00, $vb10, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                let (tb01, tb11) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    $vb01, $vb11, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                lookahead_from_tb!($va00, $va10, $va01, $va11, tb00, tb10, tb01, tb11);
            }};
        }

        macro_rules! round45_unit_b_step {
            ($p_lo:expr, $store_fn:ident) => {{
                let pl = $p_lo;
                let q0l = 4 * pl;
                let q0g = q_base + q0l;
                let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                let va00 = fold_2vars_one!(ap0);
                let va10 = fold_2vars_one!(ap0.add(4));
                let va01 = fold_2vars_one!(ap0.add(8));
                let va11 = fold_2vars_one!(ap0.add(12));

                $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va00);
                $store_fn(a_out_ptr.add(q0l + 1).cast::<__m128i>(), va10);
                $store_fn(a_out_ptr.add(q0l + 2).cast::<__m128i>(), va01);
                $store_fn(a_out_ptr.add(q0l + 3).cast::<__m128i>(), va11);
                $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), v_all_ones);
                $store_fn(b_out_ptr.add(q0l + 1).cast::<__m128i>(), v_all_ones);
                $store_fn(b_out_ptr.add(q0l + 2).cast::<__m128i>(), v_all_ones);
                $store_fn(b_out_ptr.add(q0l + 3).cast::<__m128i>(), v_all_ones);

                let eq_pair = eq_prered_ptr.add(pl) as *const __m128i;
                let veq =
                    _mm_unpacklo_epi64(_mm_loadu_si128(eq_pair), _mm_loadu_si128(eq_pair.add(1)));
                let veq_sum = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4E>(veq));
                let s10_sum = _mm_xor_si128(va10, _mm_shuffle_epi32::<0x4E>(va10));
                acc[0][0] = _mm_xor_si128(acc[0][0], _mm_clmulepi64_si128::<0x00>(veq, va10));
                acc[0][1] =
                    _mm_xor_si128(acc[0][1], _mm_clmulepi64_si128::<0x00>(veq_sum, s10_sum));
                acc[0][2] = _mm_xor_si128(acc[0][2], _mm_clmulepi64_si128::<0x11>(veq, va10));

                let s11_sum = _mm_xor_si128(va11, _mm_shuffle_epi32::<0x4E>(va11));
                acc[2][0] = _mm_xor_si128(acc[2][0], _mm_clmulepi64_si128::<0x00>(veq, va11));
                acc[2][1] =
                    _mm_xor_si128(acc[2][1], _mm_clmulepi64_si128::<0x00>(veq_sum, s11_sum));
                acc[2][2] = _mm_xor_si128(acc[2][2], _mm_clmulepi64_si128::<0x11>(veq, va11));

                let s01_sum = _mm_xor_si128(va01, _mm_shuffle_epi32::<0x4E>(va01));
                acc[4][0] = _mm_xor_si128(acc[4][0], _mm_clmulepi64_si128::<0x00>(veq, va01));
                acc[4][1] =
                    _mm_xor_si128(acc[4][1], _mm_clmulepi64_si128::<0x00>(veq_sum, s01_sum));
                acc[4][2] = _mm_xor_si128(acc[4][2], _mm_clmulepi64_si128::<0x11>(veq, va01));
            }};
        }

        macro_rules! round45_general_step {
            ($p_lo:expr, $store_fn:ident) => {{
                let pl = $p_lo;
                let q0l = 4 * pl;
                let q0g = q_base + q0l;
                let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                let bp0 = (b_pkt_ptr as *const u64).add(4 * q0g);
                let (va00, vb00) = fold_2vars_ab_one!(ap0, bp0);
                $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va00);
                $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), vb00);
                let (va10, vb10) = fold_2vars_ab_one!(ap0.add(4), bp0.add(4));
                $store_fn(a_out_ptr.add(q0l + 1).cast::<__m128i>(), va10);
                $store_fn(b_out_ptr.add(q0l + 1).cast::<__m128i>(), vb10);
                let eq_pair = eq_prered_ptr.add(pl) as *const __m128i;
                let eq_r0 = _mm_loadu_si128(eq_pair);
                let eq_r1 = _mm_loadu_si128(eq_pair.add(1));
                let (tb00, tb10) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    vb00, vb10, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                let (va01, vb01) = fold_2vars_ab_one!(ap0.add(8), bp0.add(8));
                $store_fn(a_out_ptr.add(q0l + 2).cast::<__m128i>(), va01);
                $store_fn(b_out_ptr.add(q0l + 2).cast::<__m128i>(), vb01);
                let (va11, vb11) = fold_2vars_ab_one!(ap0.add(12), bp0.add(12));
                $store_fn(a_out_ptr.add(q0l + 3).cast::<__m128i>(), va11);
                $store_fn(b_out_ptr.add(q0l + 3).cast::<__m128i>(), vb11);
                let (tb01, tb11) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    vb01, vb11, eq_r0, eq_r1, eq_r0, eq_r1,
                );
                lookahead_from_tb!(va00, va10, va01, va11, tb00, tb10, tb01, tb11);
            }};
        }

        macro_rules! run_round45_body {
            ($store_fn:ident) => {{
                if is_unit_ones
                    && lo_size >= 16
                    && (lo_size & 15) == 0
                    && (q_base & 63) == 0
                    && pair_in_block_mask == 127
                    && useful_pairs_inclusive == 121
                {
                    let n_blocks = lo_size >> 4;
                    if use_gfni {
                        let mats_ptr = bcast_mats.as_ptr() as *const u64;
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
                            for b_idx in 0..4usize {
                                let q_start_l = 4 * blk_p + 16 * b_idx;
                                let q_start_g = q_base + q_start_l;
                                let dead_lines: u8 = if b_idx == 3 { 0xC0 } else { 0 };
                                crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_c4_slice(
                                    a_pkt_ptr.add(32 * q_start_g),
                                    mats_ptr,
                                    a_out_ptr.add(q_start_l) as *mut u64,
                                    dead_lines,
                                );
                                crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_c4_slice(
                                    b_pkt_ptr.add(32 * q_start_g),
                                    mats_ptr,
                                    b_out_ptr.add(q_start_l) as *mut u64,
                                    dead_lines,
                                );
                            }
                            let p_start = if head_ones {
                                let pl = blk_p;
                                let q0l = 4 * pl;
                                let va10 =
                                    _mm_loadu_si128(a_out_ptr.add(q0l + 1).cast::<__m128i>());
                                let va01 =
                                    _mm_loadu_si128(a_out_ptr.add(q0l + 2).cast::<__m128i>());
                                let va11 =
                                    _mm_loadu_si128(a_out_ptr.add(q0l + 3).cast::<__m128i>());
                                let eq_pair = eq_prered_ptr.add(pl) as *const __m128i;
                                let veq = _mm_unpacklo_epi64(
                                    _mm_loadu_si128(eq_pair),
                                    _mm_loadu_si128(eq_pair.add(1)),
                                );
                                let veq_sum = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4E>(veq));
                                let s10_sum = _mm_xor_si128(va10, _mm_shuffle_epi32::<0x4E>(va10));
                                acc[0][0] = _mm_xor_si128(
                                    acc[0][0],
                                    _mm_clmulepi64_si128::<0x00>(veq, va10),
                                );
                                acc[0][1] = _mm_xor_si128(
                                    acc[0][1],
                                    _mm_clmulepi64_si128::<0x00>(veq_sum, s10_sum),
                                );
                                acc[0][2] = _mm_xor_si128(
                                    acc[0][2],
                                    _mm_clmulepi64_si128::<0x11>(veq, va10),
                                );

                                let s11_sum = _mm_xor_si128(va11, _mm_shuffle_epi32::<0x4E>(va11));
                                acc[2][0] = _mm_xor_si128(
                                    acc[2][0],
                                    _mm_clmulepi64_si128::<0x00>(veq, va11),
                                );
                                acc[2][1] = _mm_xor_si128(
                                    acc[2][1],
                                    _mm_clmulepi64_si128::<0x00>(veq_sum, s11_sum),
                                );
                                acc[2][2] = _mm_xor_si128(
                                    acc[2][2],
                                    _mm_clmulepi64_si128::<0x11>(veq, va11),
                                );

                                let s01_sum = _mm_xor_si128(va01, _mm_shuffle_epi32::<0x4E>(va01));
                                acc[4][0] = _mm_xor_si128(
                                    acc[4][0],
                                    _mm_clmulepi64_si128::<0x00>(veq, va01),
                                );
                                acc[4][1] = _mm_xor_si128(
                                    acc[4][1],
                                    _mm_clmulepi64_si128::<0x00>(veq_sum, s01_sum),
                                );
                                acc[4][2] = _mm_xor_si128(
                                    acc[4][2],
                                    _mm_clmulepi64_si128::<0x11>(veq, va01),
                                );
                                blk_p + 1
                            } else {
                                blk_p
                            };
                            for pl in p_start..(blk_p + 15) {
                                let q0l = 4 * pl;
                                let va00 = _mm_loadu_si128(a_out_ptr.add(q0l).cast::<__m128i>());
                                let va10 =
                                    _mm_loadu_si128(a_out_ptr.add(q0l + 1).cast::<__m128i>());
                                let va01 =
                                    _mm_loadu_si128(a_out_ptr.add(q0l + 2).cast::<__m128i>());
                                let va11 =
                                    _mm_loadu_si128(a_out_ptr.add(q0l + 3).cast::<__m128i>());
                                let vb00 = _mm_loadu_si128(b_out_ptr.add(q0l).cast::<__m128i>());
                                let vb10 =
                                    _mm_loadu_si128(b_out_ptr.add(q0l + 1).cast::<__m128i>());
                                let vb01 =
                                    _mm_loadu_si128(b_out_ptr.add(q0l + 2).cast::<__m128i>());
                                let vb11 =
                                    _mm_loadu_si128(b_out_ptr.add(q0l + 3).cast::<__m128i>());
                                lookahead_from_folded!(
                                    pl, va00, vb00, va10, vb10, va01, vb01, va11, vb11
                                );
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
                                    let mut va0 = _mm_load_si128(
                                        table_data.add((a00 & 0xFF) as usize).cast::<__m128i>(),
                                    );
                                    let mut vb0 = _mm_load_si128(
                                        table_data.add((b00 & 0xFF) as usize).cast::<__m128i>(),
                                    );
                                    macro_rules! step2 {
                                        ($c:literal) => {{
                                            let tc = table_data.add($c * 256);
                                            let sh = $c * 8;
                                            va0 = _mm_xor_si128(
                                                va0,
                                                _mm_load_si128(
                                                    tc.add(((a00 >> sh) & 0xFF) as usize)
                                                        .cast::<__m128i>(),
                                                ),
                                            );
                                            vb0 = _mm_xor_si128(
                                                vb0,
                                                _mm_load_si128(
                                                    tc.add(((b00 >> sh) & 0xFF) as usize)
                                                        .cast::<__m128i>(),
                                                ),
                                            );
                                        }};
                                    }
                                    step2!(1);
                                    step2!(2);
                                    step2!(3);
                                    if ((a00 | b00) >> 32) != 0 {
                                        step2!(4);
                                        step2!(5);
                                        step2!(6);
                                        step2!(7);
                                    }
                                    crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                        va0, vb0, opr01_r0, opr01_r1, opr01_r0, opr01_r1,
                                    )
                                } else {
                                    fold_2vars_ab_one!(ap0, bp0)
                                };
                                _mm_storeu_si128(a_out_ptr.add(q0l).cast::<__m128i>(), va00);
                                _mm_storeu_si128(b_out_ptr.add(q0l).cast::<__m128i>(), vb00);
                                let g00 = crate::gf2_kernels::k0_f128_mul_m128i(va00, vb00);
                                let eq_pair = eq_prered_ptr.add(pl) as *const __m128i;
                                let veq = _mm_unpacklo_epi64(
                                    _mm_loadu_si128(eq_pair),
                                    _mm_loadu_si128(eq_pair.add(1)),
                                );
                                crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                    veq,
                                    g00,
                                    &mut p60_ll,
                                    &mut p60_mid,
                                    &mut p60_hh,
                                );
                            }
                        }
                    } else {
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
                                round45_unit_b_step!(blk_p, $store_fn);
                            } else {
                                round45_general_step!(blk_p, $store_fn);
                            }
                            for p_lo in (blk_p + 1)..(blk_p + 14) {
                                round45_general_step!(p_lo, $store_fn);
                            }
                            {
                                let pl = blk_p + 14;
                                let tail_ones = (bp_blk.add(236).read_unaligned()
                                    & bp_blk.add(237).read_unaligned()
                                    & bp_blk.add(238).read_unaligned()
                                    & bp_blk.add(239).read_unaligned())
                                    == !0u64;
                                if tail_ones {
                                    let q0l = 4 * pl;
                                    let q0g = q_base + q0l;
                                    let ap0 = (a_pkt_ptr as *const u64).add(4 * q0g);
                                    let bp0 = (b_pkt_ptr as *const u64).add(4 * q0g);
                                    let (va00, vb00) = fold_2vars_ab_one!(ap0, bp0);
                                    let (va10, vb10) = fold_2vars_ab_one!(ap0.add(4), bp0.add(4));
                                    let (va01, vb01) = fold_2vars_ab_one!(ap0.add(8), bp0.add(8));
                                    let va11 = fold_2vars_one!(ap0.add(12));
                                    let vb11 = v_all_ones;

                                    $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va00);
                                    $store_fn(a_out_ptr.add(q0l + 1).cast::<__m128i>(), va10);
                                    $store_fn(a_out_ptr.add(q0l + 2).cast::<__m128i>(), va01);
                                    $store_fn(a_out_ptr.add(q0l + 3).cast::<__m128i>(), va11);
                                    $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), vb00);
                                    $store_fn(b_out_ptr.add(q0l + 1).cast::<__m128i>(), vb10);
                                    $store_fn(b_out_ptr.add(q0l + 2).cast::<__m128i>(), vb01);
                                    $store_fn(b_out_ptr.add(q0l + 3).cast::<__m128i>(), vb11);

                                    lookahead_from_folded!(
                                        pl, va00, vb00, va10, vb10, va01, vb01, va11, vb11
                                    );
                                } else {
                                    round45_general_step!(pl, $store_fn);
                                }
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
                                    let mut va0 = _mm_load_si128(
                                        table_data.add((a00 & 0xFF) as usize).cast::<__m128i>(),
                                    );
                                    let mut vb0 = _mm_load_si128(
                                        table_data.add((b00 & 0xFF) as usize).cast::<__m128i>(),
                                    );
                                    macro_rules! step2 {
                                        ($c:literal) => {{
                                            let tc = table_data.add($c * 256);
                                            let sh = $c * 8;
                                            va0 = _mm_xor_si128(
                                                va0,
                                                _mm_load_si128(
                                                    tc.add(((a00 >> sh) & 0xFF) as usize)
                                                        .cast::<__m128i>(),
                                                ),
                                            );
                                            vb0 = _mm_xor_si128(
                                                vb0,
                                                _mm_load_si128(
                                                    tc.add(((b00 >> sh) & 0xFF) as usize)
                                                        .cast::<__m128i>(),
                                                ),
                                            );
                                        }};
                                    }
                                    step2!(1);
                                    step2!(2);
                                    step2!(3);
                                    if ((a00 | b00) >> 32) != 0 {
                                        step2!(4);
                                        step2!(5);
                                        step2!(6);
                                        step2!(7);
                                    }
                                    crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                        va0, vb0, opr01_r0, opr01_r1, opr01_r0, opr01_r1,
                                    )
                                } else {
                                    fold_2vars_ab_one!(ap0, bp0)
                                };
                                $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va00);
                                $store_fn(a_out_ptr.add(q0l + 1).cast::<__m128i>(), z);
                                $store_fn(a_out_ptr.add(q0l + 2).cast::<__m128i>(), z);
                                $store_fn(a_out_ptr.add(q0l + 3).cast::<__m128i>(), z);
                                $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), vb00);
                                $store_fn(b_out_ptr.add(q0l + 1).cast::<__m128i>(), z);
                                $store_fn(b_out_ptr.add(q0l + 2).cast::<__m128i>(), z);
                                $store_fn(b_out_ptr.add(q0l + 3).cast::<__m128i>(), z);
                                let g00 = crate::gf2_kernels::k0_f128_mul_m128i(va00, vb00);
                                let eq_pair = eq_prered_ptr.add(pl) as *const __m128i;
                                let veq = _mm_unpacklo_epi64(
                                    _mm_loadu_si128(eq_pair),
                                    _mm_loadu_si128(eq_pair.add(1)),
                                );
                                crate::gf2_kernels::k0_f128_mul_acc_m128i(
                                    veq,
                                    g00,
                                    &mut p60_ll,
                                    &mut p60_mid,
                                    &mut p60_hh,
                                );
                            }
                        }
                    }
                    acc[1][0] = _mm_xor_si128(acc[1][0], p60_ll);
                    acc[1][1] = _mm_xor_si128(acc[1][1], p60_mid);
                    acc[1][2] = _mm_xor_si128(acc[1][2], p60_hh);
                    acc[5][0] = _mm_xor_si128(acc[5][0], p60_ll);
                    acc[5][1] = _mm_xor_si128(acc[5][1], p60_mid);
                    acc[5][2] = _mm_xor_si128(acc[5][2], p60_hh);
                    acc[7][0] = _mm_xor_si128(acc[7][0], p60_ll);
                    acc[7][1] = _mm_xor_si128(acc[7][1], p60_mid);
                    acc[7][2] = _mm_xor_si128(acc[7][2], p60_hh);
                } else if use_gfni && pair_in_block_mask == 0 && (lo_size & 3) == 0 {
                    let mats_ptr = bcast_mats.as_ptr() as *const u64;
                    for blk4 in 0..(lo_size >> 2) {
                        let p_start = blk4 << 2;
                        let q_start_l = 4 * p_start;
                        let q_start_g = q_base + q_start_l;
                        crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_c4_slice(
                            a_pkt_ptr.add(32 * q_start_g),
                            mats_ptr,
                            a_out_ptr.add(q_start_l) as *mut u64,
                            0,
                        );
                        crate::gf2_kernels::k0_gf2_fold_64_rows_gfni_c4_slice(
                            b_pkt_ptr.add(32 * q_start_g),
                            mats_ptr,
                            b_out_ptr.add(q_start_l) as *mut u64,
                            0,
                        );
                        for p_lo in p_start..(p_start + 4) {
                            let q0l = 4 * p_lo;
                            let va00 = _mm_loadu_si128(a_out_ptr.add(q0l).cast::<__m128i>());
                            let va10 = _mm_loadu_si128(a_out_ptr.add(q0l + 1).cast::<__m128i>());
                            let va01 = _mm_loadu_si128(a_out_ptr.add(q0l + 2).cast::<__m128i>());
                            let va11 = _mm_loadu_si128(a_out_ptr.add(q0l + 3).cast::<__m128i>());
                            let vb00 = _mm_loadu_si128(b_out_ptr.add(q0l).cast::<__m128i>());
                            let vb10 = _mm_loadu_si128(b_out_ptr.add(q0l + 1).cast::<__m128i>());
                            let vb01 = _mm_loadu_si128(b_out_ptr.add(q0l + 2).cast::<__m128i>());
                            let vb11 = _mm_loadu_si128(b_out_ptr.add(q0l + 3).cast::<__m128i>());
                            lookahead_from_folded!(
                                p_lo, va00, vb00, va10, vb10, va01, vb01, va11, vb11
                            );
                        }
                    }
                } else {
                    for p_lo in 0..lo_size {
                        let q0l = 4 * p_lo;
                        let q0g = q_base + q0l;
                        let fold_one_padded = |qg: usize| -> (__m128i, __m128i) {
                            let j0 = 2 * qg;
                            let j1 = j0 + 1;
                            if (j0 & pair_in_block_mask) >= useful_pairs_inclusive {
                                return (z, z);
                            }
                            let ap = (a_pkt_ptr as *const u64).add(4 * qg);
                            let bp = (b_pkt_ptr as *const u64).add(4 * qg);
                            let a00 = ap.read_unaligned();
                            let a10 = ap.add(1).read_unaligned();
                            let b00 = bp.read_unaligned();
                            let b10 = bp.add(1).read_unaligned();
                            let (a01, a11, b01, b11) =
                                if (j1 & pair_in_block_mask) < useful_pairs_inclusive {
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
                            let (fa00, fa10, fa01, fa11) = fold_table_4!(a00, ua10, ua01, ua11);
                            let (fb00, fb10, fb01, fb11) = fold_table_4!(b00, ub10, ub01, ub11);
                            (
                                combine_2vars_from_folded!(fa00, fa10, fa01, fa11),
                                combine_2vars_from_folded!(fb00, fb10, fb01, fb11),
                            )
                        };
                        let (va00, vb00) = fold_one_padded(q0g);
                        let (va10, vb10) = fold_one_padded(q0g + 1);
                        let (va01, vb01) = fold_one_padded(q0g + 2);
                        let (va11, vb11) = fold_one_padded(q0g + 3);
                        $store_fn(a_out_ptr.add(q0l).cast::<__m128i>(), va00);
                        $store_fn(a_out_ptr.add(q0l + 1).cast::<__m128i>(), va10);
                        $store_fn(a_out_ptr.add(q0l + 2).cast::<__m128i>(), va01);
                        $store_fn(a_out_ptr.add(q0l + 3).cast::<__m128i>(), va11);
                        $store_fn(b_out_ptr.add(q0l).cast::<__m128i>(), vb00);
                        $store_fn(b_out_ptr.add(q0l + 1).cast::<__m128i>(), vb10);
                        $store_fn(b_out_ptr.add(q0l + 2).cast::<__m128i>(), vb01);
                        $store_fn(b_out_ptr.add(q0l + 3).cast::<__m128i>(), vb11);
                        lookahead_from_folded!(
                            p_lo, va00, vb00, va10, vb10, va01, vb01, va11, vb11
                        );
                    }
                }
            }};
        }

        if use_stream {
            run_round45_body!(_mm_stream_si128);
            _mm_sfence();
        } else {
            run_round45_body!(_mm_storeu_si128);
        }

        let mut out = [F128::ZERO; 8];
        for k in 0..8 {
            let r = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(acc[k][0], acc[k][1], acc[k][2]);
            out[k] = core::mem::transmute::<__m128i, F128>(r);
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold_2vars_and_lookahead_chunk_x86_unchecked(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_0: F128,
    rho_1: F128,
    eq_lo_prered: &[[F128; 2]],
    use_stream: bool,
) -> [F128; 8] {
    use core::arch::x86_64::*;

    unsafe {
        let z = _mm_setzero_si128();
        let mut acc = [[z; 3]; 8];
        let lo_size = eq_lo_prered.len();
        let eq_prered_ptr = eq_lo_prered.as_ptr();
        let a_in_ptr = a_in.as_ptr();
        let b_in_ptr = b_in.as_ptr();
        let a_out_ptr = a_out.as_mut_ptr();
        let b_out_ptr = b_out.as_mut_ptr();

        let v_rho0 = _mm_set_epi64x(rho_0.hi as i64, rho_0.lo as i64);
        let v_rho1 = _mm_set_epi64x(rho_1.hi as i64, rho_1.lo as i64);
        let v_rho01_sums =
            _mm_set_epi64x((rho_1.lo ^ rho_1.hi) as i64, (rho_0.lo ^ rho_0.hi) as i64);
        let rho_01 = rho_0 * rho_1;
        let v_rho01 = _mm_set_epi64x(rho_01.hi as i64, rho_01.lo as i64);
        let v_rho01_sum = _mm_xor_si128(v_rho01, _mm_shuffle_epi32::<0x4E>(v_rho01));

        macro_rules! fold_2vars_f128 {
            ($ptr:expr) => {{
                let p = $ptr;
                let w00 = _mm_loadu_si128(p.cast::<__m128i>());
                let w10 = _mm_loadu_si128(p.add(1).cast::<__m128i>());
                let w01 = _mm_loadu_si128(p.add(2).cast::<__m128i>());
                let w11 = _mm_loadu_si128(p.add(3).cast::<__m128i>());
                let f10 = _mm_xor_si128(w00, w10);
                let f01 = _mm_xor_si128(w00, w01);
                let f11 = _mm_xor_si128(f10, _mm_xor_si128(w01, w11));
                let f10_01_sum =
                    _mm_xor_si128(_mm_unpacklo_epi64(f10, f01), _mm_unpackhi_epi64(f10, f01));
                let f11_sum = _mm_xor_si128(f11, _mm_shuffle_epi32::<0x4E>(f11));
                let ll = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>(f10, v_rho0),
                        _mm_clmulepi64_si128::<0x00>(f01, v_rho1),
                    ),
                    _mm_clmulepi64_si128::<0x00>(f11, v_rho01),
                );
                let mid = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>(f10_01_sum, v_rho01_sums),
                        _mm_clmulepi64_si128::<0x11>(f10_01_sum, v_rho01_sums),
                    ),
                    _mm_clmulepi64_si128::<0x00>(f11_sum, v_rho01_sum),
                );
                let hh = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x11>(f10, v_rho0),
                        _mm_clmulepi64_si128::<0x11>(f01, v_rho1),
                    ),
                    _mm_clmulepi64_si128::<0x11>(f11, v_rho01),
                );
                _mm_xor_si128(
                    w00,
                    crate::gf2_kernels::k0_f128_reduce_3lane_m128i(ll, mid, hh),
                )
            }};
        }

        macro_rules! acc_unred_slot_with_sum {
            ($idx:literal, $u:expr, $v:expr, $u_sum:expr, $v_sum:expr) => {{
                let u = $u;
                let v = $v;
                acc[$idx][0] = _mm_xor_si128(acc[$idx][0], _mm_clmulepi64_si128::<0x00>(u, v));
                acc[$idx][1] =
                    _mm_xor_si128(acc[$idx][1], _mm_clmulepi64_si128::<0x00>($u_sum, $v_sum));
                acc[$idx][2] = _mm_xor_si128(acc[$idx][2], _mm_clmulepi64_si128::<0x11>(u, v));
            }};
        }

        macro_rules! step_body {
            ($store_fn:ident) => {{
                for p_lo in 0..lo_size {
                    let s = 16 * p_lo;
                    let d = 4 * p_lo;
                    let ap = a_in_ptr.add(s);
                    let bp = b_in_ptr.add(s);
                    let va00 = fold_2vars_f128!(ap);
                    $store_fn(a_out_ptr.add(d).cast::<__m128i>(), va00);
                    let vb00 = fold_2vars_f128!(bp);
                    $store_fn(b_out_ptr.add(d).cast::<__m128i>(), vb00);
                    let va10 = fold_2vars_f128!(ap.add(4));
                    $store_fn(a_out_ptr.add(d + 1).cast::<__m128i>(), va10);
                    let vb10 = fold_2vars_f128!(bp.add(4));
                    $store_fn(b_out_ptr.add(d + 1).cast::<__m128i>(), vb10);

                    let eq_pair = eq_prered_ptr.add(p_lo) as *const __m128i;
                    let eq_r0 = _mm_loadu_si128(eq_pair);
                    let eq_r1 = _mm_loadu_si128(eq_pair.add(1));
                    let (tb00, tb10) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        vb00, vb10, eq_r0, eq_r1, eq_r0, eq_r1,
                    );

                    let va01 = fold_2vars_f128!(ap.add(8));
                    $store_fn(a_out_ptr.add(d + 2).cast::<__m128i>(), va01);
                    let vb01 = fold_2vars_f128!(bp.add(8));
                    $store_fn(b_out_ptr.add(d + 2).cast::<__m128i>(), vb01);
                    let va11 = fold_2vars_f128!(ap.add(12));
                    $store_fn(a_out_ptr.add(d + 3).cast::<__m128i>(), va11);
                    let vb11 = fold_2vars_f128!(bp.add(12));
                    $store_fn(b_out_ptr.add(d + 3).cast::<__m128i>(), vb11);

                    let (tb01, tb11) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        vb01, vb11, eq_r0, eq_r1, eq_r0, eq_r1,
                    );

                    let sva10 = _mm_xor_si128(va10, _mm_shuffle_epi32::<0x4E>(va10));
                    let stb10 = _mm_xor_si128(tb10, _mm_shuffle_epi32::<0x4E>(tb10));
                    acc_unred_slot_with_sum!(0, va10, tb10, sva10, stb10);

                    let da0 = _mm_xor_si128(va00, va10);
                    let dtb0 = _mm_xor_si128(tb00, tb10);
                    let sda0 = _mm_xor_si128(da0, _mm_shuffle_epi32::<0x4E>(da0));
                    let sdtb0 = _mm_xor_si128(dtb0, _mm_shuffle_epi32::<0x4E>(dtb0));
                    acc_unred_slot_with_sum!(1, da0, dtb0, sda0, sdtb0);

                    let sva11 = _mm_xor_si128(va11, _mm_shuffle_epi32::<0x4E>(va11));
                    let stb11 = _mm_xor_si128(tb11, _mm_shuffle_epi32::<0x4E>(tb11));
                    acc_unred_slot_with_sum!(2, va11, tb11, sva11, stb11);

                    let sva01 = _mm_xor_si128(va01, _mm_shuffle_epi32::<0x4E>(va01));
                    let stb01 = _mm_xor_si128(tb01, _mm_shuffle_epi32::<0x4E>(tb01));
                    acc_unred_slot_with_sum!(4, va01, tb01, sva01, stb01);

                    let da_0 = _mm_xor_si128(va00, va01);
                    let dtb_0 = _mm_xor_si128(tb00, tb01);
                    let sda_0 = _mm_xor_si128(sda0, _mm_xor_si128(sva10, sva01));
                    let sdtb_0 = _mm_xor_si128(sdtb0, _mm_xor_si128(stb10, stb01));
                    acc_unred_slot_with_sum!(5, da_0, dtb_0, sda_0, sdtb_0);

                    let da_1 = _mm_xor_si128(va10, va11);
                    let dtb_1 = _mm_xor_si128(tb10, tb11);
                    let sda_1 = _mm_xor_si128(sva10, sva11);
                    let sdtb_1 = _mm_xor_si128(stb10, stb11);
                    acc_unred_slot_with_sum!(6, da_1, dtb_1, sda_1, sdtb_1);

                    let da1 = _mm_xor_si128(va01, va11);
                    let dtb1 = _mm_xor_si128(tb01, tb11);
                    let sda1 = _mm_xor_si128(sva01, sva11);
                    let sdtb1 = _mm_xor_si128(stb01, stb11);
                    acc_unred_slot_with_sum!(3, da1, dtb1, sda1, sdtb1);

                    let da_i = _mm_xor_si128(da0, da1);
                    let dtb_i = _mm_xor_si128(dtb0, dtb1);
                    let sda_i = _mm_xor_si128(sda0, sda1);
                    let sdtb_i = _mm_xor_si128(sdtb0, sdtb1);
                    acc_unred_slot_with_sum!(7, da_i, dtb_i, sda_i, sdtb_i);
                }
            }};
        }

        if use_stream {
            step_body!(_mm_stream_si128);
            _mm_sfence();
        } else {
            step_body!(_mm_storeu_si128);
        }

        let mut out = [F128::ZERO; 8];
        for k in 0..8 {
            let r = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(acc[k][0], acc[k][1], acc[k][2]);
            out[k] = core::mem::transmute::<__m128i, F128>(r);
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fold_2vars_and_round_pair_chunk_x86_unchecked(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho_0: F128,
    rho_1: F128,
    eq_lo: &[F128],
    use_stream: bool,
) -> (F128, F128) {
    use core::arch::x86_64::*;

    unsafe {
        let z = _mm_setzero_si128();
        let mut p1_ll = z;
        let mut p1_mid = z;
        let mut p1_hh = z;
        let mut pinf_ll = z;
        let mut pinf_mid = z;
        let mut pinf_hh = z;
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr();
        let a_in_ptr = a_in.as_ptr();
        let b_in_ptr = b_in.as_ptr();
        let a_out_ptr = a_out.as_mut_ptr();
        let b_out_ptr = b_out.as_mut_ptr();

        let v_rho0 = _mm_set_epi64x(rho_0.hi as i64, rho_0.lo as i64);
        let v_rho1 = _mm_set_epi64x(rho_1.hi as i64, rho_1.lo as i64);
        let v_rho01_sums =
            _mm_set_epi64x((rho_1.lo ^ rho_1.hi) as i64, (rho_0.lo ^ rho_0.hi) as i64);
        let rho_01 = rho_0 * rho_1;
        let v_rho01 = _mm_set_epi64x(rho_01.hi as i64, rho_01.lo as i64);
        let v_rho01_sum = _mm_xor_si128(v_rho01, _mm_shuffle_epi32::<0x4E>(v_rho01));

        macro_rules! fold_2vars_f128 {
            ($ptr:expr) => {{
                let p = $ptr;
                let w00 = _mm_loadu_si128(p.cast::<__m128i>());
                let w10 = _mm_loadu_si128(p.add(1).cast::<__m128i>());
                let w01 = _mm_loadu_si128(p.add(2).cast::<__m128i>());
                let w11 = _mm_loadu_si128(p.add(3).cast::<__m128i>());
                let f10 = _mm_xor_si128(w00, w10);
                let f01 = _mm_xor_si128(w00, w01);
                let f11 = _mm_xor_si128(f10, _mm_xor_si128(w01, w11));
                let f10_01_sum =
                    _mm_xor_si128(_mm_unpacklo_epi64(f10, f01), _mm_unpackhi_epi64(f10, f01));
                let f11_sum = _mm_xor_si128(f11, _mm_shuffle_epi32::<0x4E>(f11));
                let ll = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>(f10, v_rho0),
                        _mm_clmulepi64_si128::<0x00>(f01, v_rho1),
                    ),
                    _mm_clmulepi64_si128::<0x00>(f11, v_rho01),
                );
                let mid = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>(f10_01_sum, v_rho01_sums),
                        _mm_clmulepi64_si128::<0x11>(f10_01_sum, v_rho01_sums),
                    ),
                    _mm_clmulepi64_si128::<0x00>(f11_sum, v_rho01_sum),
                );
                let hh = _mm_xor_si128(
                    _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x11>(f10, v_rho0),
                        _mm_clmulepi64_si128::<0x11>(f01, v_rho1),
                    ),
                    _mm_clmulepi64_si128::<0x11>(f11, v_rho01),
                );
                _mm_xor_si128(
                    w00,
                    crate::gf2_kernels::k0_f128_reduce_3lane_m128i(ll, mid, hh),
                )
            }};
        }

        macro_rules! step_body {
            ($store_fn:ident) => {{
                for x_lo in 0..lo_size {
                    let s = 8 * x_lo;
                    let d = 2 * x_lo;
                    let ap = a_in_ptr.add(s);
                    let bp = b_in_ptr.add(s);
                    let va0 = fold_2vars_f128!(ap);
                    let vb0 = fold_2vars_f128!(bp);
                    let va1 = fold_2vars_f128!(ap.add(4));
                    let vb1 = fold_2vars_f128!(bp.add(4));

                    $store_fn(a_out_ptr.add(d).cast::<__m128i>(), va0);
                    $store_fn(a_out_ptr.add(d + 1).cast::<__m128i>(), va1);
                    $store_fn(b_out_ptr.add(d).cast::<__m128i>(), vb0);
                    $store_fn(b_out_ptr.add(d + 1).cast::<__m128i>(), vb1);

                    let vad = _mm_xor_si128(va0, va1);
                    let vbd = _mm_xor_si128(vb0, vb1);
                    let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_m128i(va1, vad, vb1, vbd);
                    let veq = _mm_loadu_si128(eq_ptr.add(x_lo).cast::<__m128i>());
                    crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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
            }};
        }

        if use_stream {
            step_body!(_mm_stream_si128);
            _mm_sfence();
        } else {
            step_body!(_mm_storeu_si128);
        }

        let res1 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(p1_ll, p1_mid, p1_hh);
        let resi = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(pinf_ll, pinf_mid, pinf_hh);
        (
            core::mem::transmute::<__m128i, F128>(res1),
            core::mem::transmute::<__m128i, F128>(resi),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse4.1")]
pub(crate) unsafe fn round3_fold_in_place_chunk_x86_unchecked_8(
    table_data: *const F128,
    rho_0: F128,
    a_chunk: &mut [F128],
    b_chunk: &mut [F128],
    eq_lo: &[F128],
    base_j: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    use core::arch::x86_64::*;

    unsafe {
        let mut p1_ll = _mm_setzero_si128();
        let mut p1_mid = _mm_setzero_si128();
        let mut p1_hh = _mm_setzero_si128();
        let mut pinf_ll = _mm_setzero_si128();
        let mut pinf_mid = _mm_setzero_si128();
        let mut pinf_hh = _mm_setzero_si128();
        let lo_size = eq_lo.len();
        let eq_ptr = eq_lo.as_ptr();
        let a_ptr = a_chunk.as_mut_ptr();
        let b_ptr = b_chunk.as_mut_ptr();
        let v_rho0 = _mm_set_epi64x(rho_0.hi as i64, rho_0.lo as i64);
        let (rho0_r0, rho0_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(v_rho0);
        let v_opr0 = _mm_xor_si128(_mm_set_epi64x(0, 1), v_rho0);
        let (opr0_r0, opr0_r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(v_opr0);
        let v_all_ones = _mm_xor_si128(
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(2 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(3 * 256 + 255).cast::<__m128i>()),
                ),
            ),
            _mm_xor_si128(
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(4 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(5 * 256 + 255).cast::<__m128i>()),
                ),
                _mm_xor_si128(
                    _mm_load_si128(table_data.add(6 * 256 + 255).cast::<__m128i>()),
                    _mm_load_si128(table_data.add(7 * 256 + 255).cast::<__m128i>()),
                ),
            ),
        );
        let is_unit_ones =
            (_mm_extract_epi64::<0>(v_all_ones) == 1) && (_mm_extract_epi64::<1>(v_all_ones) == 0);

        macro_rules! fold_table_ab4 {
            ($tbl:expr, $a0_u64:expr, $b0_u64:expr, $a1_u64:expr, $b1_u64:expr) => {{
                let a0_w = $a0_u64;
                let b0_w = $b0_u64;
                let a1_w = $a1_u64;
                let b1_w = $b1_u64;
                let mut va0 = _mm_load_si128($tbl.add((a0_w & 0xFF) as usize).cast::<__m128i>());
                let mut vb0 = _mm_load_si128($tbl.add((b0_w & 0xFF) as usize).cast::<__m128i>());
                let mut va1 = _mm_load_si128($tbl.add((a1_w & 0xFF) as usize).cast::<__m128i>());
                let mut vb1 = _mm_load_si128($tbl.add((b1_w & 0xFF) as usize).cast::<__m128i>());
                macro_rules! step_ab4 {
                    ($c:literal) => {{
                        let tc = $tbl.add($c * 256);
                        let sh = $c * 8;
                        va0 = _mm_xor_si128(
                            va0,
                            _mm_load_si128(
                                tc.add(((a0_w >> sh) & 0xFF) as usize).cast::<__m128i>(),
                            ),
                        );
                        vb0 = _mm_xor_si128(
                            vb0,
                            _mm_load_si128(
                                tc.add(((b0_w >> sh) & 0xFF) as usize).cast::<__m128i>(),
                            ),
                        );
                        va1 = _mm_xor_si128(
                            va1,
                            _mm_load_si128(
                                tc.add(((a1_w >> sh) & 0xFF) as usize).cast::<__m128i>(),
                            ),
                        );
                        vb1 = _mm_xor_si128(
                            vb1,
                            _mm_load_si128(
                                tc.add(((b1_w >> sh) & 0xFF) as usize).cast::<__m128i>(),
                            ),
                        );
                    }};
                }
                step_ab4!(1);
                step_ab4!(2);
                step_ab4!(3);
                step_ab4!(4);
                step_ab4!(5);
                step_ab4!(6);
                step_ab4!(7);
                (va0, vb0, va1, vb1)
            }};
        }

        if is_unit_ones
            && lo_size >= 64
            && (lo_size & 63) == 0
            && (base_j & 127) == 0
            && pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
        {
            macro_rules! round3_unit_b_step {
                ($x_lo:expr) => {{
                    let xl = $x_lo;
                    let j0l = 2 * xl;
                    let j1l = j0l + 1;
                    let ap0 = a_ptr.add(j0l).cast::<u64>();
                    let a0_lo = ap0.read_unaligned();
                    let a0_hi = ap0.add(1).read_unaligned();
                    let a1_lo = ap0.add(2).read_unaligned();
                    let a1_hi = ap0.add(3).read_unaligned();
                    let (va0_lo, va0_xor, va1_lo, va1_xor) =
                        fold_table_ab4!(table_data, a0_lo, a0_lo ^ a0_hi, a1_lo, a1_lo ^ a1_hi);
                    let (va0_p, va1_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        va0_xor, va1_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                    );
                    let va_j0 = _mm_xor_si128(va0_lo, va0_p);
                    let va_j1 = _mm_xor_si128(va1_lo, va1_p);
                    _mm_store_si128(a_ptr.add(j0l).cast::<__m128i>(), va_j0);
                    _mm_store_si128(a_ptr.add(j1l).cast::<__m128i>(), va_j1);
                    _mm_store_si128(b_ptr.add(j0l).cast::<__m128i>(), v_all_ones);
                    _mm_store_si128(b_ptr.add(j1l).cast::<__m128i>(), v_all_ones);

                    let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                    let veq_sum = _mm_xor_si128(veq, _mm_shuffle_epi32::<0x4E>(veq));
                    let va1_sum = _mm_xor_si128(va_j1, _mm_shuffle_epi32::<0x4E>(va_j1));
                    p1_ll = _mm_xor_si128(p1_ll, _mm_clmulepi64_si128::<0x00>(veq, va_j1));
                    p1_mid = _mm_xor_si128(p1_mid, _mm_clmulepi64_si128::<0x00>(veq_sum, va1_sum));
                    p1_hh = _mm_xor_si128(p1_hh, _mm_clmulepi64_si128::<0x11>(veq, va_j1));
                }};
            }

            macro_rules! round3_general_both_step {
                ($x_lo:expr) => {{
                    let xl = $x_lo;
                    let j0l = 2 * xl;
                    let j1l = j0l + 1;
                    let ap0 = a_ptr.add(j0l).cast::<u64>();
                    let bp0 = b_ptr.add(j0l).cast::<u64>();
                    let a0_lo = ap0.read_unaligned();
                    let a0_hi = ap0.add(1).read_unaligned();
                    let b0_lo = bp0.read_unaligned();
                    let b0_hi = bp0.add(1).read_unaligned();
                    let (va0_lo, vb0_lo, va0_xor, vb0_xor) =
                        fold_table_ab4!(table_data, a0_lo, b0_lo, a0_lo ^ a0_hi, b0_lo ^ b0_hi);
                    let (va0_p, vb0_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        va0_xor, vb0_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                    );
                    let va_j0 = _mm_xor_si128(va0_lo, va0_p);
                    let vb_j0 = _mm_xor_si128(vb0_lo, vb0_p);

                    let a1_lo = ap0.add(2).read_unaligned();
                    let a1_hi = ap0.add(3).read_unaligned();
                    let b1_lo = bp0.add(2).read_unaligned();
                    let b1_hi = bp0.add(3).read_unaligned();
                    let (va1_lo, vb1_lo, va1_xor, vb1_xor) =
                        fold_table_ab4!(table_data, a1_lo, b1_lo, a1_lo ^ a1_hi, b1_lo ^ b1_hi);
                    let (va1_p, vb1_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        va1_xor, vb1_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                    );
                    let va_j1 = _mm_xor_si128(va1_lo, va1_p);
                    let vb_j1 = _mm_xor_si128(vb1_lo, vb1_p);

                    _mm_store_si128(a_ptr.add(j0l).cast::<__m128i>(), va_j0);
                    _mm_store_si128(a_ptr.add(j1l).cast::<__m128i>(), va_j1);
                    _mm_store_si128(b_ptr.add(j0l).cast::<__m128i>(), vb_j0);
                    _mm_store_si128(b_ptr.add(j1l).cast::<__m128i>(), vb_j1);

                    let vad = _mm_xor_si128(va_j0, va_j1);
                    let vbd = _mm_xor_si128(vb_j0, vb_j1);
                    let (g1, ginf) =
                        crate::gf2_kernels::k0_f128_mul_vec2_m128i(va_j1, vad, vb_j1, vbd);
                    let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                    crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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

            let z = _mm_setzero_si128();
            let n_blocks = lo_size >> 6;
            for blk in 0..n_blocks {
                let blk_x = blk << 6;
                let bp_blk = b_ptr.add(2 * blk_x).cast::<u64>();
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
                    for x_lo in blk_x..(blk_x + 4) {
                        round3_unit_b_step!(x_lo);
                    }
                } else {
                    for x_lo in blk_x..(blk_x + 4) {
                        round3_general_both_step!(x_lo);
                    }
                }
                for x_lo in (blk_x + 4)..(blk_x + 59) {
                    round3_general_both_step!(x_lo);
                }
                if tail_ones {
                    round3_unit_b_step!(blk_x + 59);
                } else {
                    round3_general_both_step!(blk_x + 59);
                }
                {
                    let xl = blk_x + 60;
                    let j0l = 2 * xl;
                    let j1l = j0l + 1;
                    let ap0 = a_ptr.add(j0l).cast::<u64>();
                    let bp0 = b_ptr.add(j0l).cast::<u64>();
                    let a0_lo = ap0.read_unaligned();
                    let a0_hi = ap0.add(1).read_unaligned();
                    let b0_lo = bp0.read_unaligned();
                    let b0_hi = bp0.add(1).read_unaligned();
                    let (va_j0, vb_j0) = if (a0_hi | b0_hi) == 0 {
                        let mut va0 = _mm_load_si128(
                            table_data.add((a0_lo & 0xFF) as usize).cast::<__m128i>(),
                        );
                        let mut vb0 = _mm_load_si128(
                            table_data.add((b0_lo & 0xFF) as usize).cast::<__m128i>(),
                        );
                        macro_rules! step_ab2 {
                            ($c:literal) => {{
                                let tc = table_data.add($c * 256);
                                let sh = $c * 8;
                                va0 = _mm_xor_si128(
                                    va0,
                                    _mm_load_si128(
                                        tc.add(((a0_lo >> sh) & 0xFF) as usize).cast::<__m128i>(),
                                    ),
                                );
                                vb0 = _mm_xor_si128(
                                    vb0,
                                    _mm_load_si128(
                                        tc.add(((b0_lo >> sh) & 0xFF) as usize).cast::<__m128i>(),
                                    ),
                                );
                            }};
                        }
                        step_ab2!(1);
                        step_ab2!(2);
                        step_ab2!(3);
                        step_ab2!(4);
                        step_ab2!(5);
                        step_ab2!(6);
                        step_ab2!(7);
                        crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                            va0, vb0, opr0_r0, opr0_r1, opr0_r0, opr0_r1,
                        )
                    } else {
                        let (va0_lo, vb0_lo, va0_xor, vb0_xor) =
                            fold_table_ab4!(table_data, a0_lo, b0_lo, a0_lo ^ a0_hi, b0_lo ^ b0_hi);
                        let (va0_p, vb0_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                            va0_xor, vb0_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                        );
                        (_mm_xor_si128(va0_lo, va0_p), _mm_xor_si128(vb0_lo, vb0_p))
                    };
                    _mm_store_si128(a_ptr.add(j0l).cast::<__m128i>(), va_j0);
                    _mm_store_si128(a_ptr.add(j1l).cast::<__m128i>(), z);
                    _mm_store_si128(b_ptr.add(j0l).cast::<__m128i>(), vb_j0);
                    _mm_store_si128(b_ptr.add(j1l).cast::<__m128i>(), z);
                    let ginf = crate::gf2_kernels::k0_f128_mul_m128i(va_j0, vb_j0);
                    let veq = _mm_loadu_si128(eq_ptr.add(xl).cast::<__m128i>());
                    crate::gf2_kernels::k0_f128_mul_acc_m128i(
                        veq,
                        ginf,
                        &mut pinf_ll,
                        &mut pinf_mid,
                        &mut pinf_hh,
                    );
                }
                for x_lo in (blk_x + 61)..(blk_x + 64) {
                    let j0l = 2 * x_lo;
                    let j1l = j0l + 1;
                    _mm_store_si128(a_ptr.add(j0l).cast::<__m128i>(), z);
                    _mm_store_si128(a_ptr.add(j1l).cast::<__m128i>(), z);
                    _mm_store_si128(b_ptr.add(j0l).cast::<__m128i>(), z);
                    _mm_store_si128(b_ptr.add(j1l).cast::<__m128i>(), z);
                }
            }
        } else {
            for x_lo in 0..lo_size {
                let j0l = 2 * x_lo;
                let j1l = j0l + 1;
                let j0g = base_j + j0l;
                let j1g = j0g + 1;

                if (j0g & pair_in_block_mask) >= useful_pairs_inclusive {
                    let z = _mm_setzero_si128();
                    _mm_store_si128(a_ptr.add(j0l).cast::<__m128i>(), z);
                    _mm_store_si128(a_ptr.add(j1l).cast::<__m128i>(), z);
                    _mm_store_si128(b_ptr.add(j0l).cast::<__m128i>(), z);
                    _mm_store_si128(b_ptr.add(j1l).cast::<__m128i>(), z);
                    continue;
                }

                let ap0 = a_ptr.add(j0l).cast::<u64>();
                let bp0 = b_ptr.add(j0l).cast::<u64>();
                let a0_lo = ap0.read_unaligned();
                let a0_hi = ap0.add(1).read_unaligned();
                let b0_lo = bp0.read_unaligned();
                let b0_hi = bp0.add(1).read_unaligned();
                let (va0_lo, vb0_lo, va0_xor, vb0_xor) =
                    fold_table_ab4!(table_data, a0_lo, b0_lo, a0_lo ^ a0_hi, b0_lo ^ b0_hi);
                let (va0_p, vb0_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    va0_xor, vb0_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                );
                let va_j0 = _mm_xor_si128(va0_lo, va0_p);
                let vb_j0 = _mm_xor_si128(vb0_lo, vb0_p);

                let (va_j1, vb_j1) = if (j1g & pair_in_block_mask) < useful_pairs_inclusive {
                    let ap1 = a_ptr.add(j1l).cast::<u64>();
                    let bp1 = b_ptr.add(j1l).cast::<u64>();
                    let a1_lo = ap1.read_unaligned();
                    let a1_hi = ap1.add(1).read_unaligned();
                    let b1_lo = bp1.read_unaligned();
                    let b1_hi = bp1.add(1).read_unaligned();
                    let (va1_lo, vb1_lo, va1_xor, vb1_xor) =
                        fold_table_ab4!(table_data, a1_lo, b1_lo, a1_lo ^ a1_hi, b1_lo ^ b1_hi);
                    let (va1_p, vb1_p) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        va1_xor, vb1_xor, rho0_r0, rho0_r1, rho0_r0, rho0_r1,
                    );
                    (_mm_xor_si128(va1_lo, va1_p), _mm_xor_si128(vb1_lo, vb1_p))
                } else {
                    (_mm_setzero_si128(), _mm_setzero_si128())
                };

                _mm_store_si128(a_ptr.add(j0l).cast::<__m128i>(), va_j0);
                _mm_store_si128(a_ptr.add(j1l).cast::<__m128i>(), va_j1);
                _mm_store_si128(b_ptr.add(j0l).cast::<__m128i>(), vb_j0);
                _mm_store_si128(b_ptr.add(j1l).cast::<__m128i>(), vb_j1);

                let vad = _mm_xor_si128(va_j0, va_j1);
                let vbd = _mm_xor_si128(vb_j0, vb_j1);

                let (g1, ginf) = crate::gf2_kernels::k0_f128_mul_vec2_m128i(va_j1, vad, vb_j1, vbd);

                let veq = _mm_loadu_si128(eq_ptr.add(x_lo).cast::<__m128i>());
                crate::gf2_kernels::k0_f128_mul_acc_pair_m128i(
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

        let res1 = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(p1_ll, p1_mid, p1_hh);
        let resi = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(pinf_ll, pinf_mid, pinf_hh);
        (
            core::mem::transmute::<__m128i, F128>(res1),
            core::mem::transmute::<__m128i, F128>(resi),
        )
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(crate) unsafe fn fold_and_message_x86_avx512(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4;
    use core::arch::x86_64::*;

    debug_assert_eq!(a_in.len(), 2 * a_out.len());
    debug_assert_eq!(b_in.len(), 2 * b_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());

    #[inline(always)]
    unsafe fn fold_x4(
        src: *const F128,
        r: __m512i,
        even_idx: __m512i,
        odd_idx: __m512i,
    ) -> __m512i {
        use crate::field::gf2_128::x86_64::ghash_mul_x4;
        use core::arch::x86_64::*;

        unsafe {
            let lo = _mm512_loadu_si512(src.cast::<__m512i>());
            let hi = _mm512_loadu_si512(src.add(4).cast::<__m512i>());
            let even = _mm512_permutex2var_epi64(lo, even_idx, hi);
            let odd = _mm512_permutex2var_epi64(lo, odd_idx, hi);
            _mm512_xor_si512(even, ghash_mul_x4(r, _mm512_xor_si512(even, odd)))
        }
    }

    unsafe {
        let r = _mm512_broadcast_i32x4(_mm_set_epi64x(r_fold.hi as i64, r_fold.lo as i64));
        let even_idx = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let odd_idx = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut p1_wide = WideGhashX4::zero();
        let mut pinf_wide = WideGhashX4::zero();
        let mut p1_tail = F256Unreduced::ZERO;
        let mut pinf_tail = F256Unreduced::ZERO;
        let mut x_lo = 0;

        while x_lo + 4 <= eq_lo.len() {
            let output = 2 * x_lo;
            let a_lo = fold_x4(a_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            let a_hi = fold_x4(a_in.as_ptr().add(2 * (output + 4)), r, even_idx, odd_idx);
            let b_lo = fold_x4(b_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            let b_hi = fold_x4(b_in.as_ptr().add(2 * (output + 4)), r, even_idx, odd_idx);

            _mm512_storeu_si512(a_out.as_mut_ptr().add(output).cast::<__m512i>(), a_lo);
            _mm512_storeu_si512(a_out.as_mut_ptr().add(output + 4).cast::<__m512i>(), a_hi);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output).cast::<__m512i>(), b_lo);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output + 4).cast::<__m512i>(), b_hi);

            let a0 = _mm512_permutex2var_epi64(a_lo, even_idx, a_hi);
            let a1 = _mm512_permutex2var_epi64(a_lo, odd_idx, a_hi);
            let b0 = _mm512_permutex2var_epi64(b_lo, even_idx, b_hi);
            let b1 = _mm512_permutex2var_epi64(b_lo, odd_idx, b_hi);
            let g1 = ghash_mul_x4(a1, b1);
            let g_inf = ghash_mul_x4(_mm512_xor_si512(a0, a1), _mm512_xor_si512(b0, b1));
            let eq = f128x4_loadu(eq_lo.as_ptr().add(x_lo));
            p1_wide.mul_acc(eq, g1);
            pinf_wide.mul_acc(eq, g_inf);
            x_lo += 4;
        }

        if x_lo < eq_lo.len() {
            debug_assert_eq!(eq_lo.len() - x_lo, 2);
            let output = 2 * x_lo;
            let a_folded = fold_x4(a_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            let b_folded = fold_x4(b_in.as_ptr().add(2 * output), r, even_idx, odd_idx);
            _mm512_storeu_si512(a_out.as_mut_ptr().add(output).cast::<__m512i>(), a_folded);
            _mm512_storeu_si512(b_out.as_mut_ptr().add(output).cast::<__m512i>(), b_folded);

            for lane in 0..2 {
                let o = output + 2 * lane;
                let a0 = a_out[o];
                let a1 = a_out[o + 1];
                let b0 = b_out[o];
                let b1 = b_out[o + 1];
                let eq = eq_lo[x_lo + lane];
                p1_tail ^= eq.mul_unreduced(a1 * b1);
                pinf_tail ^= eq.mul_unreduced((a0 + a1) * (b0 + b1));
            }
        }

        p1_tail ^= p1_wide.fold();
        pinf_tail ^= pinf_wide.fold();
        (p1_tail.reduce(), pinf_tail.reduce())
    }
}

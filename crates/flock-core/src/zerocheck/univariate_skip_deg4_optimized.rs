use std::sync::OnceLock;

use crate::field::gf2_8::gf8_reduce;
use crate::field::{F8, F128, mul_by_x, phi8};
use crate::ntt::{AdditiveNttGf8, InvNttTableSToV8Gf8};

use super::univariate_skip::build_eq;
use super::univariate_skip_deg4::{K_SKIP, K_V8, LAMBDA4_SIZE, S_SIZE, V8_SIZE};
use super::univariate_skip_optimized::{
    bit_transpose_64bytes, medium_challenges_ghash, small_challenges_ghash,
};

const N_INNER: usize = 7;
const N_MEDIUM: usize = 4;
const N_CHUNKS: usize = S_SIZE / 8;

pub fn small_challenges_deg4() -> [F128; 3] {
    small_challenges_ghash()
}
pub fn medium_challenges_deg4() -> [F128; 4] {
    medium_challenges_ghash()
}

fn d_inv() -> F128 {
    use crate::field::F128 as F;
    let g1 = F {
        lo: 1u64 << 1,
        hi: 0,
    };
    let g2 = F {
        lo: 1u64 << 2,
        hi: 0,
    };
    let g4 = F {
        lo: 1u64 << 4,
        hi: 0,
    };
    let g8 = F {
        lo: 1u64 << 8,
        hi: 0,
    };
    ((F::ONE + g1) * (F::ONE + g2) * (F::ONE + g4) * (F::ONE + g8)).inv()
}

fn build_convert_table() -> Vec<F128> {
    use crate::field::PHI_8_TABLE;
    let mut gamma_pow = [F128::ZERO; 16];
    gamma_pow[0] = F128::ONE;
    for b in 1..16 {
        gamma_pow[b] = mul_by_x(gamma_pow[b - 1]);
    }
    let mut table = vec![F128::ZERO; 16 * 256];
    for b in 0..16 {
        let g_b = gamma_pow[b];
        for v in 0..256 {
            table[b * 256 + v] = g_b * PHI_8_TABLE[v];
        }
    }
    table
}

static CONVERT_TABLE_CACHE: OnceLock<Vec<F128>> = OnceLock::new();
fn convert_table() -> &'static [F128] {
    CONVERT_TABLE_CACHE.get_or_init(build_convert_table)
}

#[derive(Clone, Debug)]
pub struct NttPairDeg4 {
    pub ntt_s: AdditiveNttGf8,
    pub ntt_v8: AdditiveNttGf8,
}

impl Default for NttPairDeg4 {
    fn default() -> Self {
        Self::new()
    }
}

impl NttPairDeg4 {
    pub fn new() -> Self {
        Self {
            ntt_s: AdditiveNttGf8::new(K_SKIP, F8::ZERO),
            ntt_v8: AdditiveNttGf8::new(K_V8, F8::ZERO),
        }
    }

    pub fn extend_row(&self, bits: &[u8], out: &mut [F8]) {
        debug_assert_eq!(bits.len(), N_CHUNKS);
        debug_assert_eq!(out.len(), V8_SIZE);
        for s in 0..S_SIZE {
            let bit = (bits[s / 8] >> (s % 8)) & 1;
            out[s] = F8(bit);
        }
        for s in S_SIZE..V8_SIZE {
            out[s] = F8::ZERO;
        }
        self.ntt_s.inverse(&mut out[..S_SIZE]);
        self.ntt_v8.forward(out);
    }
}

pub fn ntt_extend_f128_vec_ghash_deg4(in_s: &[F128], table: &InvNttTableSToV8Gf8) -> Vec<F128> {
    assert_eq!(in_s.len(), S_SIZE);
    assert_eq!(table.k_in, K_SKIP);
    assert_eq!(table.k_out, K_V8);

    let mut out = vec![F128::ZERO; LAMBDA4_SIZE];

    let mut gamma_pow = [F128::ZERO; 128];
    gamma_pow[0] = F128::ONE;
    for b in 1..128 {
        gamma_pow[b] = mul_by_x(gamma_pow[b - 1]);
    }

    let mut input_bits = vec![0u8; N_CHUNKS];
    let mut out_bytes = vec![F8::ZERO; V8_SIZE];

    let mut transposed_lo = [[0u8; 64]; N_CHUNKS];
    let mut transposed_hi = [[0u8; 64]; N_CHUNKS];
    for c in 0..N_CHUNKS {
        let base = c * 8;
        let lo_lanes: [u64; 8] = core::array::from_fn(|j| in_s[base + j].lo);
        let hi_lanes: [u64; 8] = core::array::from_fn(|j| in_s[base + j].hi);
        crate::bits::transpose_8_u64s_to_64_bytes(&lo_lanes, &mut transposed_lo[c]);
        crate::bits::transpose_8_u64s_to_64_bytes(&hi_lanes, &mut transposed_hi[c]);
    }

    for b in 0..128 {
        if b < 64 {
            for c in 0..N_CHUNKS {
                input_bits[c] = transposed_lo[c][b];
            }
        } else {
            let b_hi = b - 64;
            for c in 0..N_CHUNKS {
                input_bits[c] = transposed_hi[c][b_hi];
            }
        }

        table.apply(&input_bits, &mut out_bytes);

        let g_b = gamma_pow[b];
        for i in 0..LAMBDA4_SIZE {
            out[i] += g_b * phi8(out_bytes[S_SIZE + i]);
        }
    }

    out
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_4_regs_deg4<
    const BH: usize,
    const ODD: bool,
    const TILE_BASE: usize,
>(
    table_base: *const u8,
    byte: u8,
    d0: &mut core::arch::aarch64::uint8x16_t,
    d1: &mut core::arch::aarch64::uint8x16_t,
    d2: &mut core::arch::aarch64::uint8x16_t,
    d3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let row = table_base.add(byte as usize * V8_SIZE);
        let v0 = vld1q_u8(row.add(((TILE_BASE + 0) ^ BH) * 16));
        let v1 = vld1q_u8(row.add(((TILE_BASE + 1) ^ BH) * 16));
        let v2 = vld1q_u8(row.add(((TILE_BASE + 2) ^ BH) * 16));
        let v3 = vld1q_u8(row.add(((TILE_BASE + 3) ^ BH) * 16));
        let (v0, v1, v2, v3) = if ODD {
            (
                vextq_u8::<8>(v0, v0),
                vextq_u8::<8>(v1, v1),
                vextq_u8::<8>(v2, v2),
                vextq_u8::<8>(v3, v3),
            )
        } else {
            (v0, v1, v2, v3)
        };
        *d0 = veorq_u8(*d0, v0);
        *d1 = veorq_u8(*d1, v1);
        *d2 = veorq_u8(*d2, v2);
        *d3 = veorq_u8(*d3, v3);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn build_factor_row_4chunks_deg4<const TILE_BASE: usize>(
    table_base: *const u8,
    factor_row: *const u8,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let row0 = table_base.add(*factor_row as usize * V8_SIZE);
        let mut d0 = vld1q_u8(row0.add((TILE_BASE + 0) * 16));
        let mut d1 = vld1q_u8(row0.add((TILE_BASE + 1) * 16));
        let mut d2 = vld1q_u8(row0.add((TILE_BASE + 2) * 16));
        let mut d3 = vld1q_u8(row0.add((TILE_BASE + 3) * 16));

        xor_apply_byte_into_4_regs_deg4::<0, true, TILE_BASE>(
            table_base,
            *factor_row.add(1),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<1, false, TILE_BASE>(
            table_base,
            *factor_row.add(2),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<1, true, TILE_BASE>(
            table_base,
            *factor_row.add(3),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<2, false, TILE_BASE>(
            table_base,
            *factor_row.add(4),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<2, true, TILE_BASE>(
            table_base,
            *factor_row.add(5),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<3, false, TILE_BASE>(
            table_base,
            *factor_row.add(6),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs_deg4::<3, true, TILE_BASE>(
            table_base,
            *factor_row.add(7),
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
unsafe fn process_tile_deg4<const TILE_BASE: usize>(
    a_packed: *const u8,
    b_packed: *const u8,
    c_packed: *const u8,
    d_packed: *const u8,
    table_base: *const u8,
    byte_base_b: usize,
    out_tile: *mut u8,
) {
    use crate::field::gf2_8::neon::{gf8_mul_vec16, gf8_reduce_vec16};
    use core::arch::aarch64::*;
    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        macro_rules! do_k {
            ($k:literal) => {{
                let off = byte_base_b + $k * N_CHUNKS;
                let (a0, a1, a2, a3) =
                    build_factor_row_4chunks_deg4::<TILE_BASE>(table_base, a_packed.add(off));
                let (b0, b1, b2, b3) =
                    build_factor_row_4chunks_deg4::<TILE_BASE>(table_base, b_packed.add(off));
                let (c0, c1, c2, c3) =
                    build_factor_row_4chunks_deg4::<TILE_BASE>(table_base, c_packed.add(off));
                let (d0, d1, d2, d3) =
                    build_factor_row_4chunks_deg4::<TILE_BASE>(table_base, d_packed.add(off));

                let ab0 = gf8_mul_vec16(a0, b0);
                let cd0 = gf8_mul_vec16(c0, d0);
                let ab1 = gf8_mul_vec16(a1, b1);
                let cd1 = gf8_mul_vec16(c1, d1);
                let ab2 = gf8_mul_vec16(a2, b2);
                let cd2 = gf8_mul_vec16(c2, d2);
                let ab3 = gf8_mul_vec16(a3, b3);
                let cd3 = gf8_mul_vec16(c3, d3);
                let y0 = gf8_mul_vec16(ab0, cd0);
                let y1 = gf8_mul_vec16(ab1, cd1);
                let y2 = gf8_mul_vec16(ab2, cd2);
                let y3 = gf8_mul_vec16(ab3, cd3);

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
        do_k!(0);
        do_k!(1);
        do_k!(2);
        do_k!(3);
        do_k!(4);
        do_k!(5);
        do_k!(6);
        do_k!(7);

        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));
        vst1q_u8(out_tile, r0);
        vst1q_u8(out_tile.add(16), r1);
        vst1q_u8(out_tile.add(32), r2);
        vst1q_u8(out_tile.add(48), r3);
    }
}

#[cfg(target_arch = "aarch64")]
fn shift_reduce_inner_abcd_fused_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    d_packed: &[u8],
    table: &InvNttTableSToV8Gf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; LAMBDA4_SIZE],
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let table_base = table.data_ptr();
    let a_p = a_packed.as_ptr();
    let b_p = b_packed.as_ptr();
    let c_p = c_packed.as_ptr();
    let d_p = d_packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    unsafe {
        process_tile_deg4::<4>(a_p, b_p, c_p, d_p, table_base, byte_base_b, out_ptr);
        process_tile_deg4::<8>(a_p, b_p, c_p, d_p, table_base, byte_base_b, out_ptr.add(64));
        process_tile_deg4::<12>(
            a_p,
            b_p,
            c_p,
            d_p,
            table_base,
            byte_base_b,
            out_ptr.add(128),
        );
    }
}

#[inline]
fn shift_reduce_inner_abcd(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    d_packed: &[u8],
    table: &InvNttTableSToV8Gf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; LAMBDA4_SIZE],
    a_col: &mut [F8; V8_SIZE],
    b_col: &mut [F8; V8_SIZE],
    c_col: &mut [F8; V8_SIZE],
    d_col: &mut [F8; V8_SIZE],
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (a_col, b_col, c_col, d_col);
        shift_reduce_inner_abcd_fused_neon(
            a_packed,
            b_packed,
            c_packed,
            d_packed,
            table,
            chunk_byte_base,
            b_med,
            out,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        shift_reduce_inner_abcd_scalar(
            a_packed,
            b_packed,
            c_packed,
            d_packed,
            table,
            chunk_byte_base,
            b_med,
            out,
            a_col,
            b_col,
            c_col,
            d_col,
        );
    }
}

#[allow(dead_code)]
fn shift_reduce_inner_abcd_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    d_packed: &[u8],
    table: &InvNttTableSToV8Gf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; LAMBDA4_SIZE],
    a_col: &mut [F8; V8_SIZE],
    b_col: &mut [F8; V8_SIZE],
    c_col: &mut [F8; V8_SIZE],
    d_col: &mut [F8; V8_SIZE],
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let _ = gf8_reduce;
        let mut acc_even = [0u64; LAMBDA4_SIZE / 4];
        let mut acc_odd = [0u64; LAMBDA4_SIZE / 4];
        for k in 0..8 {
            let chunk_off = byte_base_b + k * N_CHUNKS;
            let shift = k as u64;
            table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
            table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
            table.apply(&c_packed[chunk_off..chunk_off + N_CHUNKS], c_col);
            table.apply(&d_packed[chunk_off..chunk_off + N_CHUNKS], d_col);

            let a_ptr = unsafe { (a_col.as_ptr() as *const u8).add(S_SIZE) };
            let b_ptr = unsafe { (b_col.as_ptr() as *const u8).add(S_SIZE) };
            let c_ptr = unsafe { (c_col.as_ptr() as *const u8).add(S_SIZE) };
            let d_ptr = unsafe { (d_col.as_ptr() as *const u8).add(S_SIZE) };
            for g in 0..(LAMBDA4_SIZE / 4) {
                unsafe {
                    let a4 = core::ptr::read_unaligned(a_ptr.add(g * 4) as *const u32) as u64;
                    let b4 = core::ptr::read_unaligned(b_ptr.add(g * 4) as *const u32) as u64;
                    let c4 = core::ptr::read_unaligned(c_ptr.add(g * 4) as *const u32) as u64;
                    let d4 = core::ptr::read_unaligned(d_ptr.add(g * 4) as *const u32) as u64;
                    let ab_wide = crate::gf2_kernels::gf2_kern_gf8_mul_x4(a4, b4);
                    let cd_wide = crate::gf2_kernels::gf2_kern_gf8_mul_x4(c4, d4);
                    let ab4 = crate::gf2_kernels::gf2_kern_pack_even_bytes_u32(ab_wide);
                    let cd4 = crate::gf2_kernels::gf2_kern_pack_even_bytes_u32(cd_wide);
                    acc_even[g] ^=
                        crate::gf2_kernels::gf2_kern_gf8_unreduced_even_x2(ab4, cd4) << shift;
                    acc_odd[g] ^=
                        crate::gf2_kernels::gf2_kern_gf8_unreduced_odd_x2(ab4, cd4) << shift;
                }
            }
        }
        for w in 0..(LAMBDA4_SIZE / 8) {
            let g0 = 2 * w;
            let g1 = g0 + 1;
            let packed = unsafe {
                let r0 = crate::gf2_kernels::gf2_kern_gf8_reduce_deg21_pair_x4(
                    acc_even[g0],
                    acc_odd[g0],
                );
                let r1 = crate::gf2_kernels::gf2_kern_gf8_reduce_deg21_pair_x4(
                    acc_even[g1],
                    acc_odd[g1],
                );
                crate::gf2_kernels::gf2_kern_pack_even_bytes_x8(r0, r1)
            };
            let base = w * 8;
            out[base..base + 8].copy_from_slice(&packed.to_le_bytes());
        }
        return;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut acc = [0u16; LAMBDA4_SIZE];

        for k in 0..8 {
            let chunk_off = byte_base_b + k * N_CHUNKS;
            table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
            table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
            table.apply(&c_packed[chunk_off..chunk_off + N_CHUNKS], c_col);
            table.apply(&d_packed[chunk_off..chunk_off + N_CHUNKS], d_col);

            for i in 0..LAMBDA4_SIZE {
                let lane = S_SIZE + i;
                let y = (a_col[lane] * b_col[lane] * c_col[lane] * d_col[lane]).0 as u16;
                acc[i] ^= y << k;
            }
        }

        for i in 0..LAMBDA4_SIZE {
            out[i] = gf8_reduce(acc[i]);
        }
    }
}

pub fn round1_shift_reduce_extract_z_packed_deg4(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    d_packed: &[u8],
    z_packed: &[u8],
    m: usize,
    r: &[F128],
    ntts: &NttPairDeg4,
    table: &InvNttTableSToV8Gf8,
) -> (Vec<F128>, Vec<F128>) {
    assert!(m >= K_SKIP + N_INNER, "m must be ≥ K_SKIP + N_INNER (=13)");
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(d_packed.len(), total_bytes);
    assert_eq!(z_packed.len(), total_bytes);
    assert_eq!(r.len(), m);

    let convert = convert_table();
    let d_inv_val = d_inv();

    let n_outer = m - K_SKIP - N_INNER;
    let eq_outer = build_eq(&r[K_SKIP + N_INNER..]);
    let eq_outer_scaled: Vec<F128> = eq_outer.iter().map(|v| *v * d_inv_val).collect();
    let big_outer_size = 1usize << n_outer;

    let mut res_abcd = [F128::ZERO; LAMBDA4_SIZE];
    let mut res_z_on_s = [F128::ZERO; S_SIZE];

    let mut chunk_abcd_bytes: Vec<[u8; LAMBDA4_SIZE]> = vec![[0u8; LAMBDA4_SIZE]; 1 << N_MEDIUM];
    let mut chunk_z_bytes: Vec<[u8; S_SIZE]> = vec![[0u8; S_SIZE]; 1 << N_MEDIUM];
    let mut a_col = [F8::ZERO; V8_SIZE];
    let mut b_col = [F8::ZERO; V8_SIZE];
    let mut c_col = [F8::ZERO; V8_SIZE];
    let mut d_col = [F8::ZERO; V8_SIZE];
    let _ = ntts;

    for x_outer in 0..big_outer_size {
        let chunk_byte_base = (x_outer << N_INNER) * N_CHUNKS;
        let eq_outer_val = eq_outer_scaled[x_outer];

        for b_med in 0..(1usize << N_MEDIUM) {
            shift_reduce_inner_abcd(
                a_packed,
                b_packed,
                c_packed,
                d_packed,
                table,
                chunk_byte_base,
                b_med,
                &mut chunk_abcd_bytes[b_med],
                &mut a_col,
                &mut b_col,
                &mut c_col,
                &mut d_col,
            );
            let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
            let z_in: &[u8; 64] = (&z_packed[byte_base_b..byte_base_b + 64])
                .try_into()
                .expect("64 z-bytes per medium position");
            bit_transpose_64bytes(z_in, &mut chunk_z_bytes[b_med]);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let convert_ptr = convert.as_ptr() as *const u8;
            for lane in 0..LAMBDA4_SIZE {
                let mut cf = vdupq_n_u8(0);
                for b_med in 0..(1usize << N_MEDIUM) {
                    let v = chunk_abcd_bytes[b_med][lane] as usize;
                    cf = veorq_u8(cf, vld1q_u8(convert_ptr.add((b_med * 256 + v) * 16)));
                }
                let cf_u64 = vreinterpretq_u64_u8(cf);
                let cf_f = F128 {
                    lo: vgetq_lane_u64::<0>(cf_u64),
                    hi: vgetq_lane_u64::<1>(cf_u64),
                };
                res_abcd[lane] += cf_f * eq_outer_val;
            }
            for lane in 0..S_SIZE {
                let mut cf = vdupq_n_u8(0);
                for b_med in 0..(1usize << N_MEDIUM) {
                    let v = chunk_z_bytes[b_med][lane] as usize;
                    cf = veorq_u8(cf, vld1q_u8(convert_ptr.add((b_med * 256 + v) * 16)));
                }
                let cf_u64 = vreinterpretq_u64_u8(cf);
                let cf_f = F128 {
                    lo: vgetq_lane_u64::<0>(cf_u64),
                    hi: vgetq_lane_u64::<1>(cf_u64),
                };
                res_z_on_s[lane] += cf_f * eq_outer_val;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for lane in 0..LAMBDA4_SIZE {
                let mut cf = F128::ZERO;
                for b_med in 0..(1usize << N_MEDIUM) {
                    let v = chunk_abcd_bytes[b_med][lane] as usize;
                    cf += convert[b_med * 256 + v];
                }
                res_abcd[lane] += cf * eq_outer_val;
            }
            for lane in 0..S_SIZE {
                let mut cf = F128::ZERO;
                for b_med in 0..(1usize << N_MEDIUM) {
                    let v = chunk_z_bytes[b_med][lane] as usize;
                    cf += convert[b_med * 256 + v];
                }
                res_z_on_s[lane] += cf * eq_outer_val;
            }
        }
    }

    let res_z_lifted = ntt_extend_f128_vec_ghash_deg4(&res_z_on_s, table);
    (res_abcd.to_vec(), res_z_lifted)
}

#[cfg(test)]
mod tests {
    use super::super::univariate_skip::pack_bits;
    use super::super::univariate_skip_deg4::round1_deg4_naive;
    use super::super::univariate_skip_optimized::c_s_f128;
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    fn build_r(m: usize, rng: &mut Rng) -> Vec<F128> {
        let mut r = vec![F128::ZERO; m];
        for i in 0..K_SKIP {
            r[i] = rng.f128();
        }
        r[K_SKIP..K_SKIP + 3].copy_from_slice(&small_challenges_deg4());
        r[K_SKIP + 3..K_SKIP + 7].copy_from_slice(&medium_challenges_deg4());
        for i in (K_SKIP + N_INNER)..m {
            r[i] = rng.f128();
        }
        r
    }

    #[test]
    fn extend_row_recovers_input_on_s() {
        let ntts = NttPairDeg4::new();
        let mut rng = Rng::new(0xA17);
        let input_bytes: Vec<u8> = (0..N_CHUNKS).map(|_| rng.next_u64() as u8).collect();
        let mut out = vec![F8::ZERO; V8_SIZE];
        ntts.extend_row(&input_bytes, &mut out);
        for s in 0..S_SIZE {
            let expected_bit = (input_bytes[s / 8] >> (s % 8)) & 1;
            assert_eq!(
                out[s],
                F8(expected_bit),
                "extend_row mismatch at s={s} (expected bit {expected_bit}, got {:?})",
                out[s]
            );
        }
    }

    #[test]
    fn optimized_matches_naive_modulo_cs() {
        let m = K_SKIP + N_INNER;
        let mut rng = Rng::new(0xBEEF);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let d = rng.bits(1 << m);
        let z = rng.bits(1 << m);
        let r = build_r(m, &mut rng);

        let ntts = NttPairDeg4::new();
        let table = InvNttTableSToV8Gf8::new(&ntts.ntt_s, &ntts.ntt_v8);
        let (a_p, b_p, c_p, d_p, z_p) = (
            pack_bits(&a),
            pack_bits(&b),
            pack_bits(&c),
            pack_bits(&d),
            pack_bits(&z),
        );

        let (opt_abcd, opt_z) = round1_shift_reduce_extract_z_packed_deg4(
            &a_p, &b_p, &c_p, &d_p, &z_p, m, &r, &ntts, &table,
        );
        let (naive_abcd, naive_z) = round1_deg4_naive(&a, &b, &c, &d, &z, m, &r);

        let cs = c_s_f128();
        for i in 0..LAMBDA4_SIZE {
            let lhs = cs * (opt_abcd[i] + opt_z[i]);
            let rhs = naive_abcd[i] + naive_z[i];
            assert_eq!(
                lhs, rhs,
                "round1 mismatch at i={i}: C_s·(opt_ab+opt_z) = {lhs:?}, naive sum = {rhs:?}"
            );
        }
    }

    #[test]
    fn optimized_matches_naive_modulo_cs_m14() {
        let m = K_SKIP + N_INNER + 1;
        let mut rng = Rng::new(0xCAFE);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let d = rng.bits(1 << m);
        let z = rng.bits(1 << m);
        let r = build_r(m, &mut rng);

        let ntts = NttPairDeg4::new();
        let table = InvNttTableSToV8Gf8::new(&ntts.ntt_s, &ntts.ntt_v8);
        let (a_p, b_p, c_p, d_p, z_p) = (
            pack_bits(&a),
            pack_bits(&b),
            pack_bits(&c),
            pack_bits(&d),
            pack_bits(&z),
        );

        let (opt_abcd, opt_z) = round1_shift_reduce_extract_z_packed_deg4(
            &a_p, &b_p, &c_p, &d_p, &z_p, m, &r, &ntts, &table,
        );
        let (naive_abcd, naive_z) = round1_deg4_naive(&a, &b, &c, &d, &z, m, &r);

        let cs = c_s_f128();
        for i in 0..LAMBDA4_SIZE {
            let lhs = cs * (opt_abcd[i] + opt_z[i]);
            let rhs = naive_abcd[i] + naive_z[i];
            assert_eq!(lhs, rhs, "mismatch at i={i} (m=14)");
        }
    }
}

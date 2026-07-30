use super::super::{F8, F128, InvNttTableByteSingleGf8, N_CHUNKS};

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
    partial_c: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        for lane in 0..64 {
            let mut converted_ab = vdupq_n_u8(0);
            let mut converted_c = vdupq_n_u8(0);
            for b_med in 0..n_b_med {
                let ab = chunk_ab_bytes[b_med][lane] as usize;
                let c = chunk_c_bytes[b_med][lane] as usize;
                converted_ab = veorq_u8(
                    converted_ab,
                    vld1q_u8(convert_ptr.add((b_med * 256 + ab) * 16)),
                );
                converted_c = veorq_u8(
                    converted_c,
                    vld1q_u8(convert_ptr.add((b_med * 256 + c) * 16)),
                );
            }
            let ab = vreinterpretq_u64_u8(converted_ab);
            let c = vreinterpretq_u64_u8(converted_c);
            partial_ab[lane] += F128 {
                lo: vgetq_lane_u64::<0>(ab),
                hi: vgetq_lane_u64::<1>(ab),
            } * eq_lo_val;
            partial_c[lane] += F128 {
                lo: vgetq_lane_u64::<0>(c),
                hi: vgetq_lane_u64::<1>(c),
            } * eq_lo_val;
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
    partial_ab: &mut [F128; 64],
    partial_c_0: &mut [F128; 64],
    partial_c_1: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        for lane in (0..64).step_by(4) {
            // Four independent lanes expose enough lookup-level parallelism
            // to hide the convert table's load latency.
            let mut ab0 = vdupq_n_u8(0);
            let mut ab1 = vdupq_n_u8(0);
            let mut ab2 = vdupq_n_u8(0);
            let mut ab3 = vdupq_n_u8(0);
            let mut c00 = vdupq_n_u8(0);
            let mut c01 = vdupq_n_u8(0);
            let mut c02 = vdupq_n_u8(0);
            let mut c03 = vdupq_n_u8(0);
            let mut c10 = vdupq_n_u8(0);
            let mut c11 = vdupq_n_u8(0);
            let mut c12 = vdupq_n_u8(0);
            let mut c13 = vdupq_n_u8(0);
            for b_med in 0..n_b_med {
                let table = convert_ptr.add(b_med * 256 * 16);
                let a0 = chunk_ab_bytes[b_med][lane] as usize;
                let a1 = chunk_ab_bytes[b_med][lane + 1] as usize;
                let a2 = chunk_ab_bytes[b_med][lane + 2] as usize;
                let a3 = chunk_ab_bytes[b_med][lane + 3] as usize;
                let c0 = chunk_c_bytes[b_med][lane] as usize;
                let c1 = chunk_c_bytes[b_med][lane + 1] as usize;
                let c2 = chunk_c_bytes[b_med][lane + 2] as usize;
                let c3 = chunk_c_bytes[b_med][lane + 3] as usize;
                ab0 = veorq_u8(ab0, vld1q_u8(table.add(a0 * 16)));
                ab1 = veorq_u8(ab1, vld1q_u8(table.add(a1 * 16)));
                ab2 = veorq_u8(ab2, vld1q_u8(table.add(a2 * 16)));
                ab3 = veorq_u8(ab3, vld1q_u8(table.add(a3 * 16)));
                c00 = veorq_u8(c00, vld1q_u8(table.add((c0 & 0x55) * 16)));
                c01 = veorq_u8(c01, vld1q_u8(table.add((c1 & 0x55) * 16)));
                c02 = veorq_u8(c02, vld1q_u8(table.add((c2 & 0x55) * 16)));
                c03 = veorq_u8(c03, vld1q_u8(table.add((c3 & 0x55) * 16)));
                c10 = veorq_u8(c10, vld1q_u8(table.add((c0 & 0xaa) * 16)));
                c11 = veorq_u8(c11, vld1q_u8(table.add((c1 & 0xaa) * 16)));
                c12 = veorq_u8(c12, vld1q_u8(table.add((c2 & 0xaa) * 16)));
                c13 = veorq_u8(c13, vld1q_u8(table.add((c3 & 0xaa) * 16)));
            }

            macro_rules! drain_lane {
                ($offset:literal, $ab:ident, $c0:ident, $c1:ident) => {{
                    let ab = vreinterpretq_u64_u8($ab);
                    let c0 = vreinterpretq_u64_u8($c0);
                    let c1 = vreinterpretq_u64_u8($c1);
                    partial_ab[lane + $offset] += F128 {
                        lo: vgetq_lane_u64::<0>(ab),
                        hi: vgetq_lane_u64::<1>(ab),
                    } * eq_lo_val;
                    partial_c_0[lane + $offset] += F128 {
                        lo: vgetq_lane_u64::<0>(c0),
                        hi: vgetq_lane_u64::<1>(c0),
                    } * eq_lo_val;
                    partial_c_1[lane + $offset] += F128 {
                        lo: vgetq_lane_u64::<0>(c1),
                        hi: vgetq_lane_u64::<1>(c1),
                    } * eq_lo_val;
                }};
            }
            drain_lane!(0, ab0, c00, c10);
            drain_lane!(1, ab1, c01, c11);
            drain_lane!(2, ab2, c02, c12);
            drain_lane!(3, ab3, c03, c13);
        }
    }
}

/// NEON 64-byte bit-transpose. Two-stage:
///   1. `vqtbl4q_u8` reorders the 64 input bytes so each 8-byte group within
///      the output is one byte-chunk's worth of `x_small=0..8` bytes.
///   2. Three rounds of bit-swap at distances 7, 14, 28 across `uint64x2_t`
///      lanes do the actual 8×8 bit transpose.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn bit_transpose_64bytes_neon(input: &[u8; 64], output: &mut [u8; 64]) {
    use core::arch::aarch64::*;

    unsafe {
        let in_ptr = input.as_ptr();
        let v0 = vld1q_u8(in_ptr);
        let v1 = vld1q_u8(in_ptr.add(16));
        let v2 = vld1q_u8(in_ptr.add(32));
        let v3 = vld1q_u8(in_ptr.add(48));
        let table = uint8x16x4_t(v0, v1, v2, v3);

        // vqtbl4q indexes that bring bytes belonging to byte-chunk b ∈ 0..8
        // into contiguous 8-byte runs, packed two-chunks-per-Q-reg.
        const IDX0: [u8; 16] = [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57];
        const IDX1: [u8; 16] = [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59];
        const IDX2: [u8; 16] = [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61];
        const IDX3: [u8; 16] = [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63];

        let mut y0 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX0.as_ptr())));
        let mut y1 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX1.as_ptr())));
        let mut y2 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX2.as_ptr())));
        let mut y3 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX3.as_ptr())));

        let mask1 = vdupq_n_u64(0x00AA00AA00AA00AA);
        let mask2 = vdupq_n_u64(0x0000CCCC0000CCCC);
        let mask3 = vdupq_n_u64(0x00000000F0F0F0F0);

        // Round 1: distance 7.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<7>(y0)), mask1);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<7>(y1)), mask1);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<7>(y2)), mask1);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<7>(y3)), mask1);
        y0 = veorq_u64(y0, veorq_u64(t0, vshlq_n_u64::<7>(t0)));
        y1 = veorq_u64(y1, veorq_u64(t1, vshlq_n_u64::<7>(t1)));
        y2 = veorq_u64(y2, veorq_u64(t2, vshlq_n_u64::<7>(t2)));
        y3 = veorq_u64(y3, veorq_u64(t3, vshlq_n_u64::<7>(t3)));

        // Round 2: distance 14.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<14>(y0)), mask2);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<14>(y1)), mask2);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<14>(y2)), mask2);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<14>(y3)), mask2);
        y0 = veorq_u64(y0, veorq_u64(t0, vshlq_n_u64::<14>(t0)));
        y1 = veorq_u64(y1, veorq_u64(t1, vshlq_n_u64::<14>(t1)));
        y2 = veorq_u64(y2, veorq_u64(t2, vshlq_n_u64::<14>(t2)));
        y3 = veorq_u64(y3, veorq_u64(t3, vshlq_n_u64::<14>(t3)));

        // Round 3: distance 28.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<28>(y0)), mask3);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<28>(y1)), mask3);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<28>(y2)), mask3);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<28>(y3)), mask3);
        y0 = veorq_u64(y0, veorq_u64(t0, vshlq_n_u64::<28>(t0)));
        y1 = veorq_u64(y1, veorq_u64(t1, vshlq_n_u64::<28>(t1)));
        y2 = veorq_u64(y2, veorq_u64(t2, vshlq_n_u64::<28>(t2)));
        y3 = veorq_u64(y3, veorq_u64(t3, vshlq_n_u64::<28>(t3)));

        let out_ptr = output.as_mut_ptr();
        vst1q_u8(out_ptr, vreinterpretq_u8_u64(y0));
        vst1q_u8(out_ptr.add(16), vreinterpretq_u8_u64(y1));
        vst1q_u8(out_ptr.add(32), vreinterpretq_u8_u64(y2));
        vst1q_u8(out_ptr.add(48), vreinterpretq_u8_u64(y3));
    }
}

// Intermediate-stage NEON kernel: scalar `inv_table.apply` writing to
// `a_col`/`b_col` Vecs, then NEON `gf8_mul_vec16` from those Vecs. Superseded
// by `shift_reduce_inner_ab_fused_neon` which keeps everything register-
// resident; kept under `#[allow(dead_code)]` as a cross-check oracle.
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

    // Four (lo, hi) pairs of u16x8 accumulators = 64 u16 lanes total, matching
    // the 64 lanes of the inv-NTT output.
    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        // Per-K step: scalar inv-NTT apply into a_col/b_col, then NEON load +
        // 4× gf8_mul_vec16 + 8× vshll_n_u8::<K> + 8× veorq_u16 into the accs.
        // K is `const` so vshll_n_u8 specializes per call site.
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

        // Final F_8 reduction: each (acc_lo, acc_hi) pair → 16 reduced u8 values.
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

// ---------------------------------------------------------------------------
// Fused NEON inner kernel: inv_NTT apply + F_8 mul + shift_reduce, all in
// NEON registers (no Vec<F8> round-trip).
//
// `xor_apply_byte_into_8_regs::<BH, ODD>` handles one byte position (b ≥ 1).
// `BH` (= b >> 1) selects which chunk-index XOR to apply; `ODD` (= b & 1)
// switches on the within-chunk half-swap. Both const-generic so the compiler
// dead-code-eliminates the if-branch and folds the chunk-index XORs.
//
// `fused_apply_one_k::<K>` runs one full K-row: the initial b=0 plain load,
// 7 calls to the byte helper for b=1..7 (with the specific protocol BH/ODD
// pattern), one 16-lane F_8 mul per output chunk, and finally widen-shift-XOR
// into the per-(K, lane) 16-bit accumulators.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_8_regs<const BH: usize, const ODD: bool>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
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
        let selected_table = if ODD {
            half_swapped_table_base
        } else {
            table_base
        };
        let ra = selected_table.add(a_byte as usize * 64);
        let rb = selected_table.add(b_byte as usize * 64);
        let va0 = vld1q_u8(ra.add((0 ^ BH) * 16));
        let va1 = vld1q_u8(ra.add((1 ^ BH) * 16));
        let va2 = vld1q_u8(ra.add((2 ^ BH) * 16));
        let va3 = vld1q_u8(ra.add((3 ^ BH) * 16));
        let vb0 = vld1q_u8(rb.add((0 ^ BH) * 16));
        let vb1 = vld1q_u8(rb.add((1 ^ BH) * 16));
        let vb2 = vld1q_u8(rb.add((2 ^ BH) * 16));
        let vb3 = vld1q_u8(rb.add((3 ^ BH) * 16));
        *da0 = veorq_u8(*da0, va0);
        *da1 = veorq_u8(*da1, va1);
        *da2 = veorq_u8(*da2, va2);
        *da3 = veorq_u8(*da3, va3);
        *db0 = veorq_u8(*db0, vb0);
        *db1 = veorq_u8(*db1, vb1);
        *db2 = veorq_u8(*db2, vb2);
        *db3 = veorq_u8(*db3, vb3);
    }
}

/// Process one K-row: 8 byte positions of `a` and `b` via the inv_NTT table,
/// F_8 multiply, shift by K, XOR into the four `(acc_lo, acc_hi)` pairs.
///
/// All K-rows accumulate **unreduced** 16-bit polynomial products (reduction
/// mod q is a ring homomorphism, so one final reduce is exact). For K ≥ 2 the
/// shifted product overflows 16 bits; the overflowed high bits (≤ 6 per lane)
/// are packed byte-wise into the per-chunk `ov` bank and folded back by the
/// caller as `ov · (x^16 mod q)`. `KINV` must equal `16 - K` (passed
/// explicitly because generic const expressions are unavailable).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fused_apply_one_k<const K: i32, const KINV: i32>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
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
    ov0: &mut core::arch::aarch64::uint8x16_t,
    ov1: &mut core::arch::aarch64::uint8x16_t,
    ov2: &mut core::arch::aarch64::uint8x16_t,
    ov3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        // The 8 index bytes per operand are consecutive; one 8-byte load plus
        // shift extraction replaces eight scalar byte loads, freeing load-port
        // slots that otherwise compete with the table's q-register loads.
        // (`byte_base_b` is a multiple of 8, so these are aligned reads.)
        let a_bits = (a_row as *const u64).read_unaligned();
        let b_bits = (b_row as *const u64).read_unaligned();

        // b = 0: identity permutation — plain load of the 4 chunks.
        let ra0 = table_base.add((a_bits & 0xff) as usize * 64);
        let rb0 = table_base.add((b_bits & 0xff) as usize * 64);
        let mut da0 = vld1q_u8(ra0);
        let mut da1 = vld1q_u8(ra0.add(16));
        let mut da2 = vld1q_u8(ra0.add(32));
        let mut da3 = vld1q_u8(ra0.add(48));
        let mut db0 = vld1q_u8(rb0);
        let mut db1 = vld1q_u8(rb0.add(16));
        let mut db2 = vld1q_u8(rb0.add(32));
        let mut db3 = vld1q_u8(rb0.add(48));

        // b = 1..7: XOR with table row[bytes[b]], permuted per (BH, ODD).
        xor_apply_byte_into_8_regs::<0, true>(
            table_base,
            half_swapped_table_base,
            ((a_bits >> 8) & 0xff) as u8,
            ((b_bits >> 8) & 0xff) as u8,
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<1, false>(
            table_base,
            half_swapped_table_base,
            ((a_bits >> 16) & 0xff) as u8,
            ((b_bits >> 16) & 0xff) as u8,
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<1, true>(
            table_base,
            half_swapped_table_base,
            ((a_bits >> 24) & 0xff) as u8,
            ((b_bits >> 24) & 0xff) as u8,
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<2, false>(
            table_base,
            half_swapped_table_base,
            ((a_bits >> 32) & 0xff) as u8,
            ((b_bits >> 32) & 0xff) as u8,
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<2, true>(
            table_base,
            half_swapped_table_base,
            ((a_bits >> 40) & 0xff) as u8,
            ((b_bits >> 40) & 0xff) as u8,
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<3, false>(
            table_base,
            half_swapped_table_base,
            ((a_bits >> 48) & 0xff) as u8,
            ((b_bits >> 48) & 0xff) as u8,
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_into_8_regs::<3, true>(
            table_base,
            half_swapped_table_base,
            (a_bits >> 56) as u8,
            (b_bits >> 56) as u8,
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );

        use crate::field::gf2_8::neon::gf8_mul_vec16_unreduced;
        let (c0_lo, c0_hi) = gf8_mul_vec16_unreduced(da0, db0);
        let (c1_lo, c1_hi) = gf8_mul_vec16_unreduced(da1, db1);
        let (c2_lo, c2_hi) = gf8_mul_vec16_unreduced(da2, db2);
        let (c3_lo, c3_hi) = gf8_mul_vec16_unreduced(da3, db3);

        // Main bank: low 16 bits of each shifted product.
        *acc0_lo = veorq_u16(*acc0_lo, vshlq_n_u16::<K>(c0_lo));
        *acc0_hi = veorq_u16(*acc0_hi, vshlq_n_u16::<K>(c0_hi));
        *acc1_lo = veorq_u16(*acc1_lo, vshlq_n_u16::<K>(c1_lo));
        *acc1_hi = veorq_u16(*acc1_hi, vshlq_n_u16::<K>(c1_hi));
        *acc2_lo = veorq_u16(*acc2_lo, vshlq_n_u16::<K>(c2_lo));
        *acc2_hi = veorq_u16(*acc2_hi, vshlq_n_u16::<K>(c2_hi));
        *acc3_lo = veorq_u16(*acc3_lo, vshlq_n_u16::<K>(c3_lo));
        *acc3_hi = veorq_u16(*acc3_hi, vshlq_n_u16::<K>(c3_hi));

        if K >= 2 {
            // Overflow bank: bits ≥ 16 of P << K, i.e. P >> (16-K), at most
            // K-1 ≤ 6 bits per lane. uzp1 keeps the low byte of each u16
            // lane, packing both halves of a chunk into one u8x16 register.
            let o0 = vuzp1q_u8(
                vreinterpretq_u8_u16(vshrq_n_u16::<KINV>(c0_lo)),
                vreinterpretq_u8_u16(vshrq_n_u16::<KINV>(c0_hi)),
            );
            let o1 = vuzp1q_u8(
                vreinterpretq_u8_u16(vshrq_n_u16::<KINV>(c1_lo)),
                vreinterpretq_u8_u16(vshrq_n_u16::<KINV>(c1_hi)),
            );
            let o2 = vuzp1q_u8(
                vreinterpretq_u8_u16(vshrq_n_u16::<KINV>(c2_lo)),
                vreinterpretq_u8_u16(vshrq_n_u16::<KINV>(c2_hi)),
            );
            let o3 = vuzp1q_u8(
                vreinterpretq_u8_u16(vshrq_n_u16::<KINV>(c3_lo)),
                vreinterpretq_u8_u16(vshrq_n_u16::<KINV>(c3_hi)),
            );
            *ov0 = veorq_u8(*ov0, o0);
            *ov1 = veorq_u8(*ov1, o1);
            *ov2 = veorq_u8(*ov2, o2);
            *ov3 = veorq_u8(*ov3, o3);
        }
    }
}

/// Fold one packed overflow bank back into a reduced chunk: computes
/// `red(x^16 · ov)` and XORs it into `r`. In this field (q = 0x11B),
/// `x^16 ≡ 0x5E (mod q)`, so the fold is one carry-less multiply by the
/// constant 0x5E followed by the standard 16→8-bit reduction.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fold_overflow_x16(
    r: core::arch::aarch64::uint8x16_t,
    ov: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;
    use core::mem::transmute;
    unsafe {
        let x16: poly8x8_t = transmute::<u64, poly8x8_t>(0x5e5e5e5e5e5e5e5e_u64);
        let t0 = vreinterpretq_u8_p16(vmull_p8(
            transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(ov)),
            x16,
        ));
        let t1 = vreinterpretq_u8_p16(vmull_p8(
            transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(ov)),
            x16,
        ));
        veorq_u8(r, gf8_reduce_vec16(t0, t1))
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
    let half_swapped_table_base = inv_table.half_swapped_data_ptr();

    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);
        // Packed per-chunk overflow banks for the K >= 2 unreduced products.
        let mut ov0 = vdupq_n_u8(0);
        let mut ov1 = vdupq_n_u8(0);
        let mut ov2 = vdupq_n_u8(0);
        let mut ov3 = vdupq_n_u8(0);

        // 8 K-iterations — each consumes N_CHUNKS = 8 packed witness bytes
        // for `a` and `b`. K is a const generic so `vshll_n_u8::<K>` specializes.
        macro_rules! do_k {
            ($k:literal, $kinv:literal) => {{
                let off = byte_base_b + $k * N_CHUNKS;
                fused_apply_one_k::<$k, $kinv>(
                    table_base,
                    half_swapped_table_base,
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
                    &mut ov0,
                    &mut ov1,
                    &mut ov2,
                    &mut ov3,
                );
            }};
        }
        do_k!(0, 16);
        do_k!(1, 15);
        do_k!(2, 14);
        do_k!(3, 13);
        do_k!(4, 12);
        do_k!(5, 11);
        do_k!(6, 10);
        do_k!(7, 9);

        // Reduce 16-bit accs → 16-byte F_8 results (4 × 16 lanes).
        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));

        // Fold the packed x^16-overflow banks back in (red is linear).
        let r0 = fold_overflow_x16(r0, ov0);
        let r1 = fold_overflow_x16(r1, ov1);
        let r2 = fold_overflow_x16(r2, ov2);
        let r3 = fold_overflow_x16(r3, ov3);

        let p = out.as_mut_ptr();
        vst1q_u8(p, r0);
        vst1q_u8(p.add(16), r1);
        vst1q_u8(p.add(32), r2);
        vst1q_u8(p.add(48), r3);
    }
}

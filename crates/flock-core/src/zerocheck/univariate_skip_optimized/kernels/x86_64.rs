#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
use super::super::{ELL, F128, N_MEDIUM};
#[cfg(target_feature = "gfni")]
use super::super::{F8, InvNttTableByteSingleGf8, N_CHUNKS};

/// AVX-512 (VBMI) 64-byte bit-transpose — direct port of the NEON two-stage
/// algorithm. `_mm512_permutexvar_epi8` does the byte-gather (NEON `vqtbl4q`)
/// in one instruction; the three masked bit-swap rounds (distances 7/14/28)
/// are identical to the NEON version, applied to all eight 64-bit lanes at once.
///
/// Replaces `bit_transpose_64bytes_scalar` (512 branchy bit ops/call) — which
/// profiling showed was ~85% of round1's time on x86.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
#[target_feature(enable = "avx512vbmi,avx512bw,avx512f")]
pub(crate) unsafe fn bit_transpose_64bytes_avx512(input: &[u8; 64], output: &mut [u8; 64]) {
    use core::arch::x86_64::*;
    // Gather index = NEON IDX0 ++ IDX1 ++ IDX2 ++ IDX3 (the 8×8 byte transpose).
    #[rustfmt::skip]
    const IDX: [i8; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56,  1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,  3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,  5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,  7, 15, 23, 31, 39, 47, 55, 63,
    ];
    unsafe {
        let inp = _mm512_loadu_si512(input.as_ptr() as *const __m512i);
        let idx = _mm512_loadu_si512(IDX.as_ptr() as *const __m512i);
        let mut y = _mm512_permutexvar_epi8(idx, inp); // y[i] = input[IDX[i]]

        let mask1 = _mm512_set1_epi64(0x00AA00AA00AA00AAu64 as i64);
        let mask2 = _mm512_set1_epi64(0x0000CCCC0000CCCCu64 as i64);
        let mask3 = _mm512_set1_epi64(0x00000000F0F0F0F0u64 as i64);

        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<7>(y)), mask1);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<7>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<14>(y)), mask2);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<14>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<28>(y)), mask3);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<28>(t)));

        _mm512_storeu_si512(output.as_mut_ptr() as *mut __m512i, y);
    }
}

/// SSE/GFNI x86 kernel. The inverse-NTT apply uses its best available x86 path,
/// writes two 64-byte columns, and this kernel multiplies them four XMM chunks
/// at a time. Kept as the fallback for GFNI CPUs without AVX-512.
#[inline]
#[allow(dead_code)] // unused in native AVX-512 builds; exercised by its oracle test
#[cfg(all(target_arch = "x86_64", target_feature = "gfni"))]
#[target_feature(enable = "gfni,sse2")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_sse(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    use core::arch::x86_64::*;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // SAFETY: function carries gfni+sse2; raw loads/stores stay within the
    // validated `a_col`/`b_col` (len ELL) and `out` ([u8; 64]) buffers.
    unsafe {
        // 4 byte-accumulators × 16 lanes = ELL = 64 lanes, reduced F_8 values.
        let mut acc = [_mm_setzero_si128(); 4];
        for k in 0..8usize {
            let chunk_off = byte_base_b + k * N_CHUNKS;
            inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
            inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
            let a_ptr = a_col.as_ptr() as *const u8;
            let b_ptr = b_col.as_ptr() as *const u8;
            let xk = _mm_set1_epi8((1u8 << k) as i8); // x^k as an F_8 byte; k=0 ⇒ 1
            for c in 0..4usize {
                let av = _mm_loadu_si128(a_ptr.add(c * 16) as *const __m128i);
                let bv = _mm_loadu_si128(b_ptr.add(c * 16) as *const __m128i);
                // y = (a·b) · x^k in F_8. For k=0, xk=1 ⇒ second mul is identity.
                let y = _mm_gf2p8mul_epi8(_mm_gf2p8mul_epi8(av, bv), xk);
                acc[c] = _mm_xor_si128(acc[c], y);
            }
        }
        let out_ptr = out.as_mut_ptr();
        for c in 0..4usize {
            _mm_storeu_si128(out_ptr.add(c * 16) as *mut __m128i, acc[c]);
        }
    }
}

/// Fused AVX-512/GFNI x86 kernel. Each inverse-NTT apply returns all 64 F_8
/// evaluations in one ZMM register; the product and x^k scaling stay 64-wide
/// and register-resident through the final XOR accumulation.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    use core::arch::x86_64::*;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // SAFETY: the caller's packed-input bounds guarantee 8 readable bytes at
    // every K-row offset. The table has the protocol-fixed ell=64/chunks=8
    // shape, and `out` is exactly one writable ZMM register.
    unsafe {
        let mut acc = _mm512_setzero_si512();
        for k in 0..8usize {
            let off = byte_base_b + k * N_CHUNKS;
            let av = inv_table.apply_x86_avx512_register_unchecked(a_packed.as_ptr().add(off));
            let bv = inv_table.apply_x86_avx512_register_unchecked(b_packed.as_ptr().add(off));
            let product = _mm512_gf2p8mul_epi8(av, bv);
            // x^0 is the multiplicative identity, so avoid one GFNI operation
            // for the first row.
            let scaled = if k == 0 {
                product
            } else {
                _mm512_gf2p8mul_epi8(product, _mm512_set1_epi8((1u8 << k) as i8))
            };
            acc = _mm512_xor_si512(acc, scaled);
        }
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
    }
}
/// x86 AVX-512 convert-table fold, AB half of the eight-bank C variant. Table
/// lookups stay scalar because their byte-selected addresses are irregular,
/// while four lanes of the resulting F128 accumulator are multiplied by
/// `eq_lo_val` in one VPCLMULQDQ batch before being XORed into the worker
/// partials. The C side is table-free (see `kernels::accumulate_c_banks`).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_convert_ab_x86_avx512(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; ELL],
) {
    use crate::field::gf2_128::x86_64::{f128x4_set, ghash_mul_x4};
    use core::arch::x86_64::*;
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert_eq!(ELL % 4, 0);

    // SAFETY: the fixed-size input/partial arrays contain every four-lane load
    // and store below. Convert indices are `b_med * 256 + u8`, bounded by the
    // 16*256-entry table. The cfg gate supplies both required target features.
    unsafe {
        let eq = f128x4_set(eq_lo_val, eq_lo_val, eq_lo_val, eq_lo_val);
        for lane in (0..ELL).step_by(4) {
            let mut cf_ab = [F128::ZERO; 4];
            for b_med in 0..n_b_med {
                let table_base = b_med * 256;
                for j in 0..4 {
                    let v_ab = chunk_ab_bytes[b_med][lane + j] as usize;
                    cf_ab[j] += convert[table_base + v_ab];
                }
            }

            let scaled_ab = ghash_mul_x4(f128x4_set(cf_ab[0], cf_ab[1], cf_ab[2], cf_ab[3]), eq);

            let ab_ptr = partial_ab.as_mut_ptr().add(lane) as *mut __m512i;
            _mm512_storeu_si512(
                ab_ptr,
                _mm512_xor_si512(_mm512_loadu_si512(ab_ptr), scaled_ab),
            );
        }
    }
}

/// Multiply-free twin of [`accumulate_convert_ab_x86_avx512`] for the banked
/// `eq_lo` tensor fold.
///
/// Same scalar table gathers; the VPCLMULQDQ batch is gone. The `eq_lo` scale
/// now rides on the table the caller selected (`T_w[i] = convert[i] ·
/// eq_top[w]`) and on the once-per-band `eq_bot[u]` fold of the bank, so the
/// four-lane accumulator is XORed straight into the bank.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_convert_ab_nomul_x86_avx512(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    convert: &[F128],
    bank: &mut [F128; ELL],
) {
    use crate::field::gf2_128::x86_64::f128x4_set;
    use core::arch::x86_64::*;
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert_eq!(ELL % 4, 0);

    // SAFETY: the fixed-size input/bank arrays contain every four-lane load and
    // store below. Convert indices are `b_med * 256 + u8`, bounded by the
    // 16*256-entry table. The cfg gate supplies the required target features.
    unsafe {
        for lane in (0..ELL).step_by(4) {
            let mut cf_ab = [F128::ZERO; 4];
            for b_med in 0..n_b_med {
                let table_base = b_med * 256;
                for j in 0..4 {
                    let v_ab = chunk_ab_bytes[b_med][lane + j] as usize;
                    cf_ab[j] += convert[table_base + v_ab];
                }
            }

            let ab_ptr = bank.as_mut_ptr().add(lane) as *mut __m512i;
            _mm512_storeu_si512(
                ab_ptr,
                _mm512_xor_si512(
                    _mm512_loadu_si512(ab_ptr),
                    f128x4_set(cf_ab[0], cf_ab[1], cf_ab[2], cf_ab[3]),
                ),
            );
        }
    }
}

/// AVX2 64-byte bit-transpose — the missing x86 mid-tier kernel.
///
/// The AVX-512 (VBMI) kernel and the scalar fallback are the only x86 paths;
/// an AVX2-only CPU (no AVX-512) therefore runs the 512-branchy-op scalar
/// kernel on every 64-byte C-block of round 1. This kernel replicates the
/// same two-stage algorithm with AVX2 primitives:
///
/// 1. Stage 1 is the `IDX` byte-gather (the 8×8 byte transpose) done as two
///    dword permutes (`_mm256_permutevar8x32_epi32`, lanes duplicated so one
///    in-lane table serves both destination lanes) + per-lane byte shuffles
///    + per-lane blends, OR-combined across the lo/hi 32-byte contributions.
/// 2. Stage 2 is the identical three masked bit-swap rounds (distances
///    7/14/28) as the AVX-512 kernel, applied per 64-bit lane on each
///    256-bit half.
///
/// Output is byte-identical to [`super::super::kernels::portable::bit_transpose_64bytes_scalar`]
/// (oracle test `avx2_bit_transpose_matches_scalar`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bit_transpose_64bytes_avx2(input: &[u8; 64], output: &mut [u8; 64]) {
    use core::arch::x86_64::*;

    // Byte-shuffle tables (16 entries, applied to each 128-bit lane; 0x80
    // zeroes the byte). After the dword permute, each lane holds the four
    // gathered dwords; `A`/`C` pick byte offsets {0,1}/{2,3} of those dwords
    // into the low half of the destination lane, `B`/`D` the same offsets
    // into the high half (the lo/hi 32-byte source contributions).
    #[rustfmt::skip]
    const TA: [i8; 32] = [
        0, 4, 8, 12, -128, -128, -128, -128, 1, 5, 9, 13, -128, -128, -128, -128,
        0, 4, 8, 12, -128, -128, -128, -128, 1, 5, 9, 13, -128, -128, -128, -128,
    ];
    #[rustfmt::skip]
    const TC: [i8; 32] = [
        2, 6, 10, 14, -128, -128, -128, -128, 3, 7, 11, 15, -128, -128, -128, -128,
        2, 6, 10, 14, -128, -128, -128, -128, 3, 7, 11, 15, -128, -128, -128, -128,
    ];
    #[rustfmt::skip]
    const TB: [i8; 32] = [
        -128, -128, -128, -128, 0, 4, 8, 12, -128, -128, -128, -128, 1, 5, 9, 13,
        -128, -128, -128, -128, 0, 4, 8, 12, -128, -128, -128, -128, 1, 5, 9, 13,
    ];
    #[rustfmt::skip]
    const TD: [i8; 32] = [
        -128, -128, -128, -128, 2, 6, 10, 14, -128, -128, -128, -128, 3, 7, 11, 15,
        -128, -128, -128, -128, 2, 6, 10, 14, -128, -128, -128, -128, 3, 7, 11, 15,
    ];

    unsafe {
        let lo = _mm256_loadu_si256(input.as_ptr() as *const __m256i);
        let hi = _mm256_loadu_si256(input.as_ptr().add(32) as *const __m256i);

        // Stage 1: 8×8 byte transpose (the IDX gather). Dest lanes 0,1 read
        // source dwords {0,2,4,6} (+{8,10,12,14} from hi); dest lanes 2,3
        // read {1,3,5,7} (+{9,11,13,15}). Duplicating the permute index per
        // lane lets one in-lane shuffle table serve both destination lanes.
        let perm_a = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let perm_b = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
        let pa_lo = _mm256_permutevar8x32_epi32(lo, perm_a);
        let pa_hi = _mm256_permutevar8x32_epi32(hi, perm_a);
        let pb_lo = _mm256_permutevar8x32_epi32(lo, perm_b);
        let pb_hi = _mm256_permutevar8x32_epi32(hi, perm_b);

        let ta = _mm256_loadu_si256(TA.as_ptr() as *const __m256i);
        let tb = _mm256_loadu_si256(TB.as_ptr() as *const __m256i);
        let tc = _mm256_loadu_si256(TC.as_ptr() as *const __m256i);
        let td = _mm256_loadu_si256(TD.as_ptr() as *const __m256i);
        // lane0 → A/B pattern, lane1 → C/D pattern.
        let mask0 = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1,
        );

        let lo_part = _mm256_blendv_epi8(
            _mm256_shuffle_epi8(pa_lo, ta),
            _mm256_shuffle_epi8(pa_lo, tc),
            mask0,
        );
        let hi_part = _mm256_blendv_epi8(
            _mm256_shuffle_epi8(pa_hi, tb),
            _mm256_shuffle_epi8(pa_hi, td),
            mask0,
        );
        let mut y0 = _mm256_or_si256(lo_part, hi_part); // dest lanes 0,1

        let lo_part = _mm256_blendv_epi8(
            _mm256_shuffle_epi8(pb_lo, ta),
            _mm256_shuffle_epi8(pb_lo, tc),
            mask0,
        );
        let hi_part = _mm256_blendv_epi8(
            _mm256_shuffle_epi8(pb_hi, tb),
            _mm256_shuffle_epi8(pb_hi, td),
            mask0,
        );
        let mut y1 = _mm256_or_si256(lo_part, hi_part); // dest lanes 2,3

        // Stage 2: three masked bit-swap rounds on 64-bit lanes (identical to
        // the AVX-512 kernel, applied to each 256-bit half).
        let mask1 = _mm256_set1_epi64x(0x00AA00AA00AA00AAu64 as i64);
        let mask2 = _mm256_set1_epi64x(0x0000CCCC0000CCCCu64 as i64);
        let mask3 = _mm256_set1_epi64x(0x00000000F0F0F0F0u64 as i64);

        let t = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<7>(y0)), mask1);
        y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t, _mm256_slli_epi64::<7>(t)));
        let t = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<14>(y0)), mask2);
        y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t, _mm256_slli_epi64::<14>(t)));
        let t = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<28>(y0)), mask3);
        y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t, _mm256_slli_epi64::<28>(t)));

        let t = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<7>(y1)), mask1);
        y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t, _mm256_slli_epi64::<7>(t)));
        let t = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<14>(y1)), mask2);
        y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t, _mm256_slli_epi64::<14>(t)));
        let t = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<28>(y1)), mask3);
        y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t, _mm256_slli_epi64::<28>(t)));

        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, y0);
        _mm256_storeu_si256(output.as_mut_ptr().add(32) as *mut __m256i, y1);
    }
}

/// x86 SSE2 acceleration of the paired Fold4 C drain
/// (`accumulate_c_fold4_q_pair_banks` scalar fallback).
///
/// The scalar fallback bit-transposes each live c row separately (phase 1),
/// then re-extracts bit `k` of every transposed row per (lane, bank) in the
/// drain — ~24 scalar ops per (lane, bank). This kernel keeps phase 1 exactly
/// (the `bit_transpose_64bytes` arch dispatch, which is the AVX2 kernel on
/// modern x86) and replaces the per-lane scalar extraction with a 16-lane
/// vectorized nibble assembly: for each bank `k`, the four transposed rows
/// are loaded as 128-bit vectors, bit `k` of every byte is isolated with a
/// masked 16-bit shift, shifted into the `h` nibble position and OR-accumulated
/// into `masks[side][k][lane]`. The drain (64-bit mask loads, `even | odd << 4`
/// index assembly, 16-byte pair-table gathers) is unchanged. SSE2 is a
/// baseline x86_64 feature, so the 128-bit ops need no runtime gate. Output
/// is byte-identical to `accumulate_c_fold4_q_pair_banks_scalar` (differential
/// oracle in the bt_cdrain standalone bench).
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn accumulate_c_fold4_q_pair_banks_x86(
    c_block_even: &[u8; 16 * 64],
    n_b_med_even: usize,
    c_block_odd: &[u8; 16 * 64],
    n_b_med_odd: usize,
    q: usize,
    pair_mask_table: &[super::super::F128; 256],
    partial_c: &mut [[super::super::F128; 64]; 8],
) {
    use core::arch::x86_64::*;

    debug_assert!(n_b_med_even <= 16);
    debug_assert!(n_b_med_odd <= 16);
    debug_assert!(q < 4);
    debug_assert_eq!(pair_mask_table.len(), 256);

    // SAFETY: every live row is bounds-checked against its block's independent
    // padding count; rows at or beyond `n_b_med` read as zero through the
    // zero-initialized `transposed`/`masks` buffers. Each mask byte is a
    // four-bit h-nibble, so the concatenated pair is a valid u8 table index.
    unsafe {
        // Phase 1: per-row 64-byte bit-transposes through the arch dispatch
        // (AVX2 on this class of CPU) — byte-identical to the scalar fallback.
        let mut transposed = [[[0u8; 64]; 4]; 2];
        for (side, (c_block, n_b_med)) in [(c_block_even, n_b_med_even), (c_block_odd, n_b_med_odd)]
            .into_iter()
            .enumerate()
        {
            for (h, row_out) in transposed[side].iter_mut().enumerate() {
                let b_med = q + 4 * h;
                if b_med < n_b_med {
                    let row: &[u8; 64] = c_block[b_med * 64..(b_med + 1) * 64]
                        .try_into()
                        .expect("64 c-bytes per medium position");
                    super::super::bit_transpose_64bytes(row, row_out);
                }
            }
        }

        // Phase 2: vectorized nibble assembly. `masks[side][k][lane]` = the
        // h-mask for bank k at lane, built 16 lanes at a time. The per-byte
        // `(v >> k) & 1` uses a masked 16-bit shift: `_mm_srli_epi16` carries
        // byte j+1's low bits into byte j's high bits, but the `& 0x0101`
        // keeps only bit 0, which the carry can never reach. The `<< h`
        // carries bit 0 into the next byte's bit 0, masked by `0xFF << h`.
        let mut masks = [[[0u8; 64]; 8]; 2];
        let one = _mm_set1_epi16(0x0101);
        for side in 0..2 {
            for chunk in 0..4 {
                let col = chunk * 16;
                let rows = [
                    _mm_loadu_si128(transposed[side][0].as_ptr().add(col) as *const __m128i),
                    _mm_loadu_si128(transposed[side][1].as_ptr().add(col) as *const __m128i),
                    _mm_loadu_si128(transposed[side][2].as_ptr().add(col) as *const __m128i),
                    _mm_loadu_si128(transposed[side][3].as_ptr().add(col) as *const __m128i),
                ];
                for k in 0..8 {
                    let mut nib = _mm_setzero_si128();
                    for h in 0..4 {
                        let bit = match k {
                            0 => _mm_and_si128(rows[h], one),
                            1 => _mm_and_si128(_mm_srli_epi16::<1>(rows[h]), one),
                            2 => _mm_and_si128(_mm_srli_epi16::<2>(rows[h]), one),
                            3 => _mm_and_si128(_mm_srli_epi16::<3>(rows[h]), one),
                            4 => _mm_and_si128(_mm_srli_epi16::<4>(rows[h]), one),
                            5 => _mm_and_si128(_mm_srli_epi16::<5>(rows[h]), one),
                            6 => _mm_and_si128(_mm_srli_epi16::<6>(rows[h]), one),
                            _ => _mm_and_si128(_mm_srli_epi16::<7>(rows[h]), one),
                        };
                        let shifted = match h {
                            0 => bit,
                            1 => _mm_and_si128(
                                _mm_slli_epi16::<1>(bit),
                                _mm_set1_epi16(0xFEFEu16 as i16),
                            ),
                            2 => _mm_and_si128(
                                _mm_slli_epi16::<2>(bit),
                                _mm_set1_epi16(0xFCFCu16 as i16),
                            ),
                            _ => _mm_and_si128(
                                _mm_slli_epi16::<3>(bit),
                                _mm_set1_epi16(0xF8F8u16 as i16),
                            ),
                        };
                        nib = _mm_or_si128(nib, shifted);
                    }
                    _mm_storeu_si128(masks[side][k].as_mut_ptr().add(col) as *mut __m128i, nib);
                }
            }
        }

        // Phase 3: drain — identical to the scalar semantics: 64-bit mask
        // loads, per-byte index assembly (`even | odd << 4` cannot carry
        // across bytes because both operands hold only low nibbles), then
        // 16-byte pair-table gathers XORed into the eight bank accumulators.
        let table = pair_mask_table.as_ptr() as *const u8;
        for k in 0..8 {
            let bank = partial_c[k].as_mut_ptr() as *mut u8;
            for lane in (0..64).step_by(8) {
                let even = (masks[0][k].as_ptr().add(lane) as *const u64).read_unaligned();
                let odd = (masks[1][k].as_ptr().add(lane) as *const u64).read_unaligned();
                let indices = even | (odd << 4);
                let i0 = usize::from((indices & 0xff) as u8);
                let i1 = usize::from(((indices >> 8) & 0xff) as u8);
                let i2 = usize::from(((indices >> 16) & 0xff) as u8);
                let i3 = usize::from(((indices >> 24) & 0xff) as u8);
                let i4 = usize::from(((indices >> 32) & 0xff) as u8);
                let i5 = usize::from(((indices >> 40) & 0xff) as u8);
                let i6 = usize::from(((indices >> 48) & 0xff) as u8);
                let i7 = usize::from((indices >> 56) as u8);

                let t0 = _mm_loadu_si128(table.add(i0 * 16) as *const __m128i);
                let t1 = _mm_loadu_si128(table.add(i1 * 16) as *const __m128i);
                let t2 = _mm_loadu_si128(table.add(i2 * 16) as *const __m128i);
                let t3 = _mm_loadu_si128(table.add(i3 * 16) as *const __m128i);
                let t4 = _mm_loadu_si128(table.add(i4 * 16) as *const __m128i);
                let t5 = _mm_loadu_si128(table.add(i5 * 16) as *const __m128i);
                let t6 = _mm_loadu_si128(table.add(i6 * 16) as *const __m128i);
                let t7 = _mm_loadu_si128(table.add(i7 * 16) as *const __m128i);

                let p0 = bank.add(lane * 16);
                let p1 = p0.add(16);
                let p2 = p1.add(16);
                let p3 = p2.add(16);
                let p4 = p3.add(16);
                let p5 = p4.add(16);
                let p6 = p5.add(16);
                let p7 = p6.add(16);
                _mm_storeu_si128(
                    p0 as *mut __m128i,
                    _mm_xor_si128(_mm_loadu_si128(p0 as *const __m128i), t0),
                );
                _mm_storeu_si128(
                    p1 as *mut __m128i,
                    _mm_xor_si128(_mm_loadu_si128(p1 as *const __m128i), t1),
                );
                _mm_storeu_si128(
                    p2 as *mut __m128i,
                    _mm_xor_si128(_mm_loadu_si128(p2 as *const __m128i), t2),
                );
                _mm_storeu_si128(
                    p3 as *mut __m128i,
                    _mm_xor_si128(_mm_loadu_si128(p3 as *const __m128i), t3),
                );
                _mm_storeu_si128(
                    p4 as *mut __m128i,
                    _mm_xor_si128(_mm_loadu_si128(p4 as *const __m128i), t4),
                );
                _mm_storeu_si128(
                    p5 as *mut __m128i,
                    _mm_xor_si128(_mm_loadu_si128(p5 as *const __m128i), t5),
                );
                _mm_storeu_si128(
                    p6 as *mut __m128i,
                    _mm_xor_si128(_mm_loadu_si128(p6 as *const __m128i), t6),
                );
                _mm_storeu_si128(
                    p7 as *mut __m128i,
                    _mm_xor_si128(_mm_loadu_si128(p7 as *const __m128i), t7),
                );
            }
        }
    }
}

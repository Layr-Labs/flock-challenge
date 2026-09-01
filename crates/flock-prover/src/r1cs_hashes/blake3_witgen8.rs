//! 8-wide AVX2 lockstep BLAKE3 witness builder (`__m256i`, 8×u32).
//!
//! Same G-function / carry-bit / packed-row stream as the 4-wide SSE kernel,
//! widened to one rayon group (8 compressions) per call. The z drain
//! optionally publishes through streaming stores (`z_nt`): z's next reader is
//! the commit encode, a later phase, so its write-allocate RFO is pure waste
//! (see the caller's gate).
//!
//! The a/b drains have two shapes, selected by `win_ab`:
//!  * `None` — the incumbent: temporal (`storeu`), because the caller re-reads
//!    a/b L1-hot for the round-1 window precompute in the same task.
//!  * `Some(..)` — FUSED: the very same `tr8` registers feed the main a/b
//!    buffers non-temporally AND a compact per-octa window buffer that the
//!    caller projects from instead. The main buffers are then never re-read,
//!    so their write-allocate RFO (1 GiB at the ranked shape) is deleted too.
//!
//! Ranked live path: `generate_witness_with_ab_packed_and_round1_inner_impl`
//! (`FLOCK_NO_WITGEN_LIVE_SIMD=1` restores the scalar 1-block loop).

use super::{
    ADDS_PER_G, BLAKE3_IV, CARRY_BITS_PER_ADD, Compression, G_STRIDE, GS_BASE, K, OUT_HI_BASE,
    USEFUL_BITS, WORD_BITS,
};
use core::arch::x86_64::*;
use flock_core::ntt::InvNttTableByteSingleGf8;
use flock_core::zerocheck::univariate_skip_optimized::{
    ROUND1_AB_OFF_WORDS, Round1AbTableImages, Round1AbWindowPlan,
    round1_ab_inner_window_from_offsets, round1_ab_inner_window_from_offsets_nt2,
    round1_ab_inner_window_from_offsets_nt2_bcomplement_static,
    round1_ab_inner_window_from_offsets_nt2_bcomplement_static_const,
    round1_ab_inner_window_from_offsets_nt2_residual, round1_ab_inner_window30_k0,
    round1_ab_inner_window_with_images, round1_ab_table_images,
};

const REC_C0: usize = 0;
const REC_C1: usize = CARRY_BITS_PER_ADD;
const REC_C2: usize = 2 * CARRY_BITS_PER_ADD;
const REC_C3: usize = 3 * CARRY_BITS_PER_ADD;
const REC_C4: usize = 4 * CARRY_BITS_PER_ADD;
const REC_C5: usize = 5 * CARRY_BITS_PER_ADD;
const REC_LIN0: usize = ADDS_PER_G * CARRY_BITS_PER_ADD;
const REC_LIN1: usize = REC_LIN0 + WORD_BITS;
const U32_PER_BLOCK: usize = K / 32;
const BYTES_PER_BLOCK: usize = K / 8;
const DUMP_CHUNKS: usize = U32_PER_BLOCK / 8;
/// Words a drain step publishes at once — sixteen, which is exactly one
/// 64-byte round-1 medium window per block.
const STEP_WORDS: usize = 16;
/// u32s of the streaming projection's staging pair: one step's sixteen words
/// for eight blocks, a side.
pub(crate) const STREAM_STAGE_WORDS: usize = 2 * 8 * STEP_WORDS;
// 62, not 61: an odd boundary leaves the z drain's paired NT loop with a
// lone 32-byte tail chunk — one masked, partially-written NT line per block
// (an ECC read-modify-write at the memory controller, 2^18 times per
// proof). Chunk 61 is entirely inside the zero tail (LAST_WORD = 481), so
// storing it is redundant-but-correct and the ragged tail branch vanishes.
const ELIDE_ZERO_CHUNK: usize = 62;
// 60, not 59, for the FUSED (`dump_range_nt_win`) drain, for exactly the reason
// `ELIDE_ZERO_CHUNK` is 62: an odd first-elided chunk leaves chunk 58 as a lone
// 32-byte NT store inside the 64-byte line (58, 59) — a partially-filled
// write-combining buffer, i.e. a read-modify-write at the memory controller,
// once per block per prove. Chunks 60..64 are entirely inside b's fixed
// lin-id/out_hi ones + zero padding run (which starts at bit 15,089 < 256*60),
// so storing chunk 59 is redundant-but-correct and the ragged line vanishes.
// The temporal (`dump_range`) arm has no write-combining buffer to leave open,
// so it keeps the tighter 59.
const ELIDE_B_TAIL_CHUNK_WIN: usize = 60;
const ELIDE_B_PREFIX_CHUNKS: usize = 4;
const LAST_WORD: usize = (USEFUL_BITS - 1) / 32;
const _ELIDE_GEOMETRY: () = {
    assert!(8 * ELIDE_ZERO_CHUNK >= USEFUL_BITS.div_ceil(32));
    assert!(8 * ELIDE_ZERO_CHUNK < U32_PER_BLOCK);
    assert!(8 * ELIDE_B_PREFIX_CHUNKS <= 36);
    assert!(LAST_WORD == 481);
    // The fused drain walks chunk PAIRS from 0 while `dump_range_nt` walks
    // pairs from `g0`; the two agree on which chunks share a 64-byte line
    // only while every elided PREFIX is pair-aligned. (Elided tails need no
    // such property: a half-covered trailing pair degrades to the same
    // single-chunk stream in both.)
    assert!(DUMP_CHUNKS.is_multiple_of(2));
    assert!(ELIDE_B_PREFIX_CHUNKS.is_multiple_of(2));
    // The fused b tail is a pair-aligned SUBSET of the temporal one, so it
    // stays strictly inside the same content-independent constant run.
    assert!(ELIDE_B_TAIL_CHUNK_WIN.is_multiple_of(2));
    assert!(ELIDE_B_TAIL_CHUNK_WIN < DUMP_CHUNKS);
};

type V8 = __m256i;

/// Input form for one eight-compression witness kernel invocation.
///
/// The ranked speculative path carries the generator's closed form all the
/// way here. Producing the 25 draws directly as eight SIMD lanes avoids
/// materializing and immediately transposing an 896-byte `[Compression; 8]`.
/// Slice callers retain the original borrowed-input route verbatim.
pub(crate) enum OctaInputs<'a> {
    Blocks([&'a Compression; 8]),
    Closed { init: u64, base: usize },
}

struct PreparedInputs {
    cv: [V8; 8],
    message: [V8; 16],
    counter_lo: V8,
    counter_hi: V8,
    block_len: V8,
    flags: V8,
}

/// `FLOCK_NO_EY_DEAD_W31=1` restores the incumbent projection of ranked
/// window 31. Default ON; the ranked worker's cleared environment never
/// disables it.
///
/// Window 31 is `w = 1, b_med = 15`. Round one reads window `w * 16 + b_med`
/// only for `b_med < n_b_med[w]`, and `build_b_med_counts` gives
/// `n_b_med[1] = ceil((USEFUL_BITS - 8192) / 512) = 15` at the pinned BLAKE3
/// shape (`K_LOG = 14`, `USEFUL_BITS = 15409`), so `b_med = 15` is past the
/// last live window. Nothing else reads those bytes: every `ab_inner`
/// consumer is one of the three `b_med < n_b_med` copy loops, and the cold
/// `restore_full_if_ranked_one_rows_elided` fallback recomputes the whole
/// buffer from raw a/b rather than reading it.
///
/// Same precedent as the elided one-rows: under `one_rows_elided` windows 0
/// and 1 are already left unwritten because round one starts at
/// `first_b_med = 2`. The elision is gated on that same brand, so the exact
/// shape assumptions above are the ones already asserted for it.
#[inline(always)]
pub(crate) fn ey_dead_w31_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_EY_DEAD_W31").is_none());
    *ON
}

/// AVX2 fallback for low-half 64-bit multiplication. `_mm256_mul_epu32`
/// handles the low limbs; the two cross-products supply bits 32..63. The
/// high×high term is above bit 63 and is discarded, exactly like
/// `u64::wrapping_mul`.
#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn mullo_u64x4(a: __m256i, b: u64) -> __m256i {
    unsafe {
        let b_lo = _mm256_set1_epi64x(u64::from(b as u32) as i64);
        let b_hi = _mm256_set1_epi64x((b >> 32) as i64);
        let lo = _mm256_mul_epu32(a, b_lo);
        let a_hi = _mm256_srli_epi64::<32>(a);
        let cross_lo = _mm256_mul_epu32(a, b_hi);
        let cross_hi = _mm256_mul_epu32(a_hi, b_lo);
        let cross = _mm256_add_epi64(cross_lo, cross_hi);
        _mm256_add_epi64(lo, _mm256_slli_epi64::<32>(cross))
    }
}

#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn mix_u64x4(mut z: __m256i) -> __m256i {
    unsafe {
        z = _mm256_xor_si256(z, _mm256_srli_epi64::<30>(z));
        z = mullo_u64x4(z, 0xBF58_476D_1CE4_E5B9);
        z = _mm256_xor_si256(z, _mm256_srli_epi64::<27>(z));
        z = mullo_u64x4(z, 0x94D0_49BB_1331_11EB);
        _mm256_xor_si256(z, _mm256_srli_epi64::<31>(z))
    }
}

/// Pack the low u32 of eight u64 lanes into one `V8`, preserving block order.
#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn pack_low_u32x8(lo: __m256i, hi: __m256i) -> V8 {
    unsafe {
        let indices = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let lo = _mm256_permutevar8x32_epi32(lo, indices);
        let hi = _mm256_permutevar8x32_epi32(hi, indices);
        _mm256_permute2x128_si256::<0x20>(lo, hi)
    }
}

#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn next_generator_draw(states: &mut [__m256i; 2]) -> V8 {
    unsafe {
        let golden = _mm256_set1_epi64x(crate::seed_pipe::GOLDEN as i64);
        states[0] = _mm256_add_epi64(states[0], golden);
        states[1] = _mm256_add_epi64(states[1], golden);
        pack_low_u32x8(mix_u64x4(states[0]), mix_u64x4(states[1]))
    }
}

/// Generate the exact protected-wrapper inputs for blocks `base..base+8`
/// directly in the word-major form consumed by the witness kernel. AVX2
/// targets use two four-lane SplitMix states and the low-limb multiplication
/// above; Sapphire Rapids uses the eight-lane AVX-512DQ implementation below.
#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn prepare_closed_inputs(init: u64, base: usize) -> PreparedInputs {
    unsafe {
        let stride =
            crate::seed_pipe::GOLDEN.wrapping_mul(crate::seed_pipe::DRAWS_PER_BLOCK as u64);
        let first = init.wrapping_add((base as u64).wrapping_mul(stride));
        let mut states = [
            _mm256_setr_epi64x(
                first as i64,
                first.wrapping_add(stride) as i64,
                first.wrapping_add(stride.wrapping_mul(2)) as i64,
                first.wrapping_add(stride.wrapping_mul(3)) as i64,
            ),
            _mm256_setr_epi64x(
                first.wrapping_add(stride.wrapping_mul(4)) as i64,
                first.wrapping_add(stride.wrapping_mul(5)) as i64,
                first.wrapping_add(stride.wrapping_mul(6)) as i64,
                first.wrapping_add(stride.wrapping_mul(7)) as i64,
            ),
        ];
        let cv = std::array::from_fn(|_| next_generator_draw(&mut states));
        let message = std::array::from_fn(|_| next_generator_draw(&mut states));
        let counter_lo = next_generator_draw(&mut states);
        PreparedInputs {
            cv,
            message,
            counter_lo,
            counter_hi: _mm256_setzero_si256(),
            block_len: _mm256_set1_epi32(64),
            flags: _mm256_set1_epi32(11),
        }
    }
}

#[cfg(target_feature = "avx512dq")]
#[inline(always)]
unsafe fn mix_u64x8(mut z: __m512i) -> __m512i {
    unsafe {
        z = _mm512_xor_si512(z, _mm512_srli_epi64::<30>(z));
        z = _mm512_mullo_epi64(z, _mm512_set1_epi64(0xBF58_476D_1CE4_E5B9u64 as i64));
        z = _mm512_xor_si512(z, _mm512_srli_epi64::<27>(z));
        z = _mm512_mullo_epi64(z, _mm512_set1_epi64(0x94D0_49BB_1331_11EBu64 as i64));
        _mm512_xor_si512(z, _mm512_srli_epi64::<31>(z))
    }
}

#[cfg(target_feature = "avx512dq")]
#[inline(always)]
unsafe fn next_generator_draw(state: &mut __m512i) -> V8 {
    unsafe {
        *state = _mm512_add_epi64(*state, _mm512_set1_epi64(crate::seed_pipe::GOLDEN as i64));
        _mm512_cvtepi64_epi32(mix_u64x8(*state))
    }
}

#[cfg(target_feature = "avx512dq")]
#[inline(always)]
unsafe fn prepare_closed_inputs(init: u64, base: usize) -> PreparedInputs {
    unsafe {
        let stride =
            crate::seed_pipe::GOLDEN.wrapping_mul(crate::seed_pipe::DRAWS_PER_BLOCK as u64);
        let first = init.wrapping_add((base as u64).wrapping_mul(stride));
        let mut state = _mm512_setr_epi64(
            first as i64,
            first.wrapping_add(stride) as i64,
            first.wrapping_add(stride.wrapping_mul(2)) as i64,
            first.wrapping_add(stride.wrapping_mul(3)) as i64,
            first.wrapping_add(stride.wrapping_mul(4)) as i64,
            first.wrapping_add(stride.wrapping_mul(5)) as i64,
            first.wrapping_add(stride.wrapping_mul(6)) as i64,
            first.wrapping_add(stride.wrapping_mul(7)) as i64,
        );
        let cv = std::array::from_fn(|_| next_generator_draw(&mut state));
        let message = std::array::from_fn(|_| next_generator_draw(&mut state));
        let counter_lo = next_generator_draw(&mut state);
        PreparedInputs {
            cv,
            message,
            counter_lo,
            counter_hi: _mm256_setzero_si256(),
            block_len: _mm256_set1_epi32(64),
            flags: _mm256_set1_epi32(11),
        }
    }
}

#[inline(always)]
unsafe fn load_v8(p: *const u32) -> V8 {
    unsafe { _mm256_loadu_si256(p.cast::<__m256i>()) }
}

#[inline(always)]
unsafe fn store_v8(p: *mut u32, v: V8) {
    unsafe { _mm256_storeu_si256(p.cast::<__m256i>(), v) }
}

/// Store one complete ranked projection row. `StreamProj::stage` is backed by
/// `AbWinLine`, and both its side stride and row stride are whole cache lines.
#[cfg(target_feature = "avx512f")]
#[inline(always)]
unsafe fn store_ranked_stage_line(p: *mut u32, v: __m512i) {
    unsafe { _mm512_store_si512(p.cast::<__m512i>(), v) }
}

/// Widen one staged 64-byte window side (its two transposed V8 rows, still in
/// registers) to the pre-scaled `u16` offset half the split shift-reduce
/// kernel consumes: `byte * 64` per lane, two ZMM stores. Identical values to
/// the kernel's own prologue — the source is the register the staging store
/// was made from instead of a reload of that store.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn widen_off_half(lo: V8, hi: V8, op: *mut u16) {
    unsafe {
        let w = |v: V8| _mm512_slli_epi16::<6>(_mm512_cvtepu8_epi16(v));
        _mm512_store_si512(op.cast::<__m512i>(), w(lo));
        _mm512_store_si512(op.add(32).cast::<__m512i>(), w(hi));
    }
}

/// ZMM-row twin of [`widen_off_half`]. The low cast is free; the one high-half
/// extract lets the fused transpose keep its row whole for the aligned stage
/// store while still feeding the two byte-to-word widening instructions.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn widen_off_line(v: __m512i, op: *mut u16) {
    unsafe {
        widen_off_half(
            _mm512_castsi512_si256(v),
            _mm512_extracti64x4_epi64::<1>(v),
            op,
        )
    }
}

#[inline(always)]
fn dup_u32(x: u32) -> V8 {
    unsafe { _mm256_set1_epi32(x as i32) }
}

#[inline(always)]
fn xor_v8(a: V8, b: V8) -> V8 {
    unsafe { _mm256_xor_si256(a, b) }
}

/// Three-way XOR `a ^ b ^ c`. On AVX-512VL this folds the two `vpxord`s of the
/// carry-in chain into one `vpternlogd` with immediate `0x96` (the truth table
/// of `a ^ b ^ c`, order-independent, bit-identical to the paired XORs).
#[allow(dead_code)]
#[inline(always)]
fn xor3_v8(a: V8, b: V8, c: V8) -> V8 {
    #[cfg(target_feature = "avx512vl")]
    unsafe {
        _mm256_ternarylogic_epi32::<0x96>(a, b, c)
    }
    // AVX2-only hosts (no AVX-512VL): the paired XORs this folds. Bit-identical.
    #[cfg(not(target_feature = "avx512vl"))]
    unsafe {
        _mm256_xor_si256(_mm256_xor_si256(a, b), c)
    }
}

#[inline(always)]
fn or_v8(a: V8, b: V8) -> V8 {
    unsafe { _mm256_or_si256(a, b) }
}

#[inline(always)]
fn and_v8(a: V8, b: V8) -> V8 {
    unsafe { _mm256_and_si256(a, b) }
}

#[inline(always)]
fn add_v8(a: V8, b: V8) -> V8 {
    unsafe { _mm256_add_epi32(a, b) }
}

#[inline(always)]
fn shr_v8<const N: i32>(v: V8) -> V8 {
    unsafe { _mm256_srli_epi32::<N>(v) }
}

#[inline(always)]
fn shl_v8<const N: i32>(v: V8) -> V8 {
    unsafe { _mm256_slli_epi32::<N>(v) }
}

/// NEON `vsli` #N, 8 lanes: bits `N..32` from `b << N`, bits `0..N` keep `a`.
#[inline(always)]
fn vsli_v8<const N: i32>(a: V8, b: V8) -> V8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // One VPTERNLOGD (AVX512F, EVEX.256) folds the AND+OR of the AVX2
        // emulation: out = (b << N) | (a & mask) == ternlog<0xF8>(b<<N, a, mask).
        // Bit-identical; the carry-packing is the hot half of the witness
        // G-functions, and on the ranked runner the ternlog saves one op per
        // push (~16 per G, ~900 per 8-block call). The mask is a compile-time
        // constant here (N is literal), so it folds into the ternlog's memory
        // operand or a constant broadcast; no register pressure change.
        unsafe {
            let mask = _mm256_set1_epi32(((1u64 << N) - 1) as u32 as i32);
            _mm256_ternarylogic_epi32::<0xF8>(_mm256_slli_epi32::<N>(b), a, mask)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512vl")))]
    unsafe {
        let mask = _mm256_set1_epi32(((1u64 << N) - 1) as u32 as i32);
        _mm256_or_si256(_mm256_slli_epi32::<N>(b), _mm256_and_si256(a, mask))
    }
}

/// 8×8 u32 transpose. `r[i]` lane `j` becomes `out[j]` lane `i`.
#[inline(always)]
fn tr8(v0: V8, v1: V8, v2: V8, v3: V8, v4: V8, v5: V8, v6: V8, v7: V8) -> [V8; 8] {
    unsafe {
        let t0 = _mm256_unpacklo_epi32(v0, v1);
        let t1 = _mm256_unpackhi_epi32(v0, v1);
        let t2 = _mm256_unpacklo_epi32(v2, v3);
        let t3 = _mm256_unpackhi_epi32(v2, v3);
        let t4 = _mm256_unpacklo_epi32(v4, v5);
        let t5 = _mm256_unpackhi_epi32(v4, v5);
        let t6 = _mm256_unpacklo_epi32(v6, v7);
        let t7 = _mm256_unpackhi_epi32(v6, v7);

        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);
        let u4 = _mm256_unpacklo_epi64(t4, t6);
        let u5 = _mm256_unpackhi_epi64(t4, t6);
        let u6 = _mm256_unpacklo_epi64(t5, t7);
        let u7 = _mm256_unpackhi_epi64(t5, t7);

        [
            _mm256_permute2x128_si256::<0x20>(u0, u4),
            _mm256_permute2x128_si256::<0x20>(u1, u5),
            _mm256_permute2x128_si256::<0x20>(u2, u6),
            _mm256_permute2x128_si256::<0x20>(u3, u7),
            _mm256_permute2x128_si256::<0x31>(u0, u4),
            _mm256_permute2x128_si256::<0x31>(u1, u5),
            _mm256_permute2x128_si256::<0x31>(u2, u6),
            _mm256_permute2x128_si256::<0x31>(u3, u7),
        ]
    }
}

/// 16×8 u32 transpose specialized for one complete ranked projection side.
/// `stage[w + i]` is word-major (lane `j` belongs to block `j`); the result
/// is block-major: `out[j]` lane `i == stage[w + i]` lane `j`.
///
/// The eight ZMM loads each join two adjacent input words. Three binary
/// `permutex2var` layers then grow each contiguous output group:
///
/// 1. four input rows × four columns,
/// 2. eight input rows × two columns,
/// 3. sixteen input rows × one column.
///
/// Formally, let `M[i][j]` denote lane `j` of input word `i`. After layer 1,
/// `p[2g+h][4c+r] = M[4g+r][4h+c]`; after layer 2,
/// `q[4g+h][8c+r] = M[8g+r][2h+c]`; the final two selectors therefore give
/// `out[2h+c][r] = M[r][2h+c]`. The index vectors below are those equations
/// written in `permutex2var_epi32`'s concatenated `[a|b]` lane numbering.
///
/// # Safety
/// `stage[w..w+16]` contains sixteen initialized `V8`s. AVX-512F is enabled.
#[cfg(target_feature = "avx512f")]
#[inline(always)]
unsafe fn tr8x16_zmm(stage: *const V8, w: usize) -> [__m512i; 8] {
    unsafe {
        debug_assert!(w + STEP_WORDS <= RING_WORDS);
        debug_assert!(w.is_multiple_of(STEP_WORDS));

        // Each unaligned ZMM is [word 2k lanes 0..7 | word 2k+1 lanes 0..7].
        let x0 = _mm512_loadu_si512(stage.add(w).cast::<__m512i>());
        let x1 = _mm512_loadu_si512(stage.add(w + 2).cast::<__m512i>());
        let x2 = _mm512_loadu_si512(stage.add(w + 4).cast::<__m512i>());
        let x3 = _mm512_loadu_si512(stage.add(w + 6).cast::<__m512i>());
        let x4 = _mm512_loadu_si512(stage.add(w + 8).cast::<__m512i>());
        let x5 = _mm512_loadu_si512(stage.add(w + 10).cast::<__m512i>());
        let x6 = _mm512_loadu_si512(stage.add(w + 12).cast::<__m512i>());
        let x7 = _mm512_loadu_si512(stage.add(w + 14).cast::<__m512i>());

        let i10 = _mm512_setr_epi32(0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27);
        let i11 = _mm512_setr_epi32(4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31);
        let p00 = _mm512_permutex2var_epi32(x0, i10, x1);
        let p01 = _mm512_permutex2var_epi32(x0, i11, x1);
        let p10 = _mm512_permutex2var_epi32(x2, i10, x3);
        let p11 = _mm512_permutex2var_epi32(x2, i11, x3);
        let p20 = _mm512_permutex2var_epi32(x4, i10, x5);
        let p21 = _mm512_permutex2var_epi32(x4, i11, x5);
        let p30 = _mm512_permutex2var_epi32(x6, i10, x7);
        let p31 = _mm512_permutex2var_epi32(x6, i11, x7);

        let i20 = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23);
        let i21 = _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31);
        let q00 = _mm512_permutex2var_epi32(p00, i20, p10);
        let q01 = _mm512_permutex2var_epi32(p00, i21, p10);
        let q02 = _mm512_permutex2var_epi32(p01, i20, p11);
        let q03 = _mm512_permutex2var_epi32(p01, i21, p11);
        let q10 = _mm512_permutex2var_epi32(p20, i20, p30);
        let q11 = _mm512_permutex2var_epi32(p20, i21, p30);
        let q12 = _mm512_permutex2var_epi32(p21, i20, p31);
        let q13 = _mm512_permutex2var_epi32(p21, i21, p31);

        let i30 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
        let i31 = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
        [
            _mm512_permutex2var_epi32(q00, i30, q10),
            _mm512_permutex2var_epi32(q00, i31, q10),
            _mm512_permutex2var_epi32(q01, i30, q11),
            _mm512_permutex2var_epi32(q01, i31, q11),
            _mm512_permutex2var_epi32(q02, i30, q12),
            _mm512_permutex2var_epi32(q02, i31, q12),
            _mm512_permutex2var_epi32(q03, i30, q13),
            _mm512_permutex2var_epi32(q03, i31, q13),
        ]
    }
}

/// Transpose one fully-live ranked side and widen its offsets while the
/// complete block-major rows are still live.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn ranked_dense_rows_and_offsets_rollback(
    ring: *const V8,
    w: usize,
    op: *mut u16,
) -> [__m512i; 8] {
    unsafe {
        let rows = tr8x16_zmm(ring, w);
        widen_ranked_dense_rows(&rows, op);
        rows
    }
}

/// `FLOCK_NO_WITGEN_RANKED_DIRECT_PUBLISH=1` restores the ranked hot-window
/// stage stores plus stage reload publication. Default ON; the ranked worker's
/// cleared environment never disables it.
#[inline(always)]
fn ranked_direct_dense_publish_enabled() -> bool {
    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("FLOCK_NO_WITGEN_RANKED_DIRECT_PUBLISH").is_none()
        });
        *ON
    }
    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
    {
        false
    }
}

/// `FLOCK_NO_WITGEN_RANKED_DIRECT_INLINE=1` restores the original
/// by-value-row + outlined-consumer ABI inside the direct-publish path while
/// preserving its bytes and publication schedule.
#[inline(always)]
fn ranked_direct_dense_inline_enabled() -> bool {
    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("FLOCK_NO_WITGEN_RANKED_DIRECT_INLINE").is_none()
        });
        *ON
    }
    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
    {
        false
    }
}

/// `FLOCK_NO_R1_B_CONSTWIN=1` restores the runtime-block B-complement leaf.
/// Cache the rollback once and share it between the outlined and direct-inline
/// publishers so the default ranked path cannot bypass the const-window leaf.
#[inline(always)]
fn ranked_b_constwin_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_R1_B_CONSTWIN").as_deref()
            != Some(std::ffi::OsStr::new("1"))
    });
    *ON
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn widen_ranked_dense_rows(rows: &[__m512i; 8], op: *mut u16) {
    unsafe {
        let mut r = 0usize;
        while r != 8 {
            widen_off_line(rows[r], op.add(r * ROUND1_AB_OFF_WORDS));
            r += 1;
        }
    }
}

/// Transpose and stage one fully-live ranked side, preserving each complete
/// ZMM row long enough to build its two offset halves without a stage reload.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn stage_ranked_dense_side(ring: *const V8, w: usize, stage: *mut u32, op: *mut u16) {
    unsafe {
        let rows = tr8x16_zmm(ring, w);
        widen_ranked_dense_rows(&rows, op);
        let mut r = 0usize;
        while r != 8 {
            store_ranked_stage_line(stage.add(r * STEP_WORDS), rows[r]);
            r += 1;
        }
    }
}

/// Caller-owned direct-publish path: keep the ranked rows in the caller's
/// frame, widen their offsets there, and consume them without an outlined ABI
/// boundary or a by-value `[__m512i; 8]` return.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn project_blocks_ranked_hot_offsets_direct_inline(
    proj: &StreamProj<'_>,
    blk: usize,
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
    rows: RankedRows,
    a_ring: *const V8,
    b_ring: *const V8,
    rw: usize,
    off: *mut u16,
) {
    unsafe {
        debug_assert!(blk > 1 && blk < 30);
        let a_rows = tr8x16_zmm(a_ring, rw);
        widen_ranked_dense_rows(&a_rows, off);
        let b_rows = tr8x16_zmm(b_ring, rw);
        widen_ranked_dense_rows(&b_rows, off.add(64));

        if proj.one_rows_elided && blk == 2 {
            let mut j = 0usize;
            while j != 8 {
                rows.publish_dense_values(j, a_rows[j], b_rows[j]);
                let out = &mut *proj
                    .out
                    .add(j * proj.out_stride + 2 * 64 - proj.out_bias)
                    .cast::<[u8; 64]>();
                round1_ab_inner_window_from_offsets_nt2_residual(
                    &*off
                        .add(j * ROUND1_AB_OFF_WORDS)
                        .cast::<[u16; ROUND1_AB_OFF_WORDS]>(),
                    out,
                    plan,
                    imgs,
                    0xfc,
                );
                j += 1;
            }
            return;
        }
        if proj.one_rows_elided && blk == 29 {
            let mut j = 0usize;
            while j != 8 {
                rows.publish_dense_values(j, a_rows[j], b_rows[j]);
                let out = &mut *proj
                    .out
                    .add(j * proj.out_stride + 29 * 64 - proj.out_bias)
                    .cast::<[u8; 64]>();
                round1_ab_inner_window_from_offsets_nt2_residual(
                    &*off
                        .add(j * ROUND1_AB_OFF_WORDS)
                        .cast::<[u16; ROUND1_AB_OFF_WORDS]>(),
                    out,
                    plan,
                    imgs,
                    0x0f,
                );
                j += 1;
            }
            return;
        }

        if plan.bcomplement_static_eligible() && ranked_b_constwin_enabled() {
            match blk {
                3 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<3>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                4 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<4>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                5 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<5>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                6 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<6>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                7 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<7>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                8 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<8>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                9 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<9>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                10 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<10>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                11 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<11>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                12 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<12>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                13 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<13>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                14 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<14>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                15 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<15>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                16 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<16>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                17 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<17>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                18 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<18>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                19 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<19>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                20 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<20>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                21 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<21>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                22 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<22>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                23 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<23>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                24 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<24>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                25 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<25>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                26 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<26>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                27 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<27>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                28 => return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<28>(proj,plan,imgs,rows,&a_rows,&b_rows,off),
                _ => {}
            }
        }

        let mut j = 0usize;
        while j != 8 {
            rows.publish_dense_values(j, a_rows[j], b_rows[j]);
            let out = &mut *proj
                .out
                .add(j * proj.out_stride + blk * 64 - proj.out_bias)
                .cast::<[u8; 64]>();
            if plan.bcomplement_static_eligible() {
                round1_ab_inner_window_from_offsets_nt2_bcomplement_static(
                    &*off
                        .add(j * ROUND1_AB_OFF_WORDS)
                        .cast::<[u16; ROUND1_AB_OFF_WORDS]>(),
                    out,
                    plan,
                    imgs,
                    blk,
                );
            } else {
                round1_ab_inner_window_from_offsets_nt2(
                    &*off
                        .add(j * ROUND1_AB_OFF_WORDS)
                        .cast::<[u16; ROUND1_AB_OFF_WORDS]>(),
                    out,
                    plan,
                    imgs,
                );
            }
            j += 1;
        }
    }
}

/// Direct-inline const-window leaf. The producer-owned transposed rows remain
/// live across publication, while the block index is fixed for the corrected
/// B-complement core wrapper.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const<const BLK: usize>(
    proj: &StreamProj<'_>,
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
    rows: RankedRows,
    a_rows: &[__m512i; 8],
    b_rows: &[__m512i; 8],
    off: *const u16,
) {
    unsafe {
        debug_assert!((3..=28).contains(&BLK));
        debug_assert!(plan.bcomplement_static_eligible());
        let mut j = 0usize;
        while j != 8 {
            rows.publish_dense_values(j, a_rows[j], b_rows[j]);
            let out = &mut *proj
                .out
                .add(j * proj.out_stride + BLK * 64 - proj.out_bias)
                .cast::<[u8; 64]>();
            round1_ab_inner_window_from_offsets_nt2_bcomplement_static_const::<BLK>(
                &*off
                    .add(j * ROUND1_AB_OFF_WORDS)
                    .cast::<[u16; ROUND1_AB_OFF_WORDS]>(),
                out,
                plan,
                imgs,
            );
            j += 1;
        }
    }
}

const RING_WORDS: usize = 32;
/// Words the pre-round prologue fills, starting at word 16.
const PROLOGUE_WORDS: usize = 20;
/// Ranked static windows whose B side is supplied without reading the ring.
const RANKED_STATIC_B_PREFIX_END: usize = 2 * STEP_WORDS;
const RANKED_STATIC_B_TAIL_START: usize = 30 * STEP_WORDS;
const _RING_GEOMETRY: () = {
    assert!(RING_WORDS >= 32);
    assert!(RING_WORDS <= U32_PER_BLOCK);
    assert!(RING_WORDS & (RING_WORDS - 1) == 0);
    // Every epoch boundary is a whole number of drain steps.
    assert!(RING_WORDS.is_multiple_of(STEP_WORDS));
    assert!(STEP_WORDS == 16);
    // One step row is exactly one cache line, and every witness-block stride
    // preserves that alignment. The ranked publisher relies on both facts to
    // issue aligned ZMM non-temporal stores without a per-line alignment test.
    assert!(STEP_WORDS * core::mem::size_of::<u32>() == 64);
    assert!((U32_PER_BLOCK * core::mem::size_of::<u32>()).is_multiple_of(64));
    // blk30's packed qwords plus the sliding-reference guard occupy exactly
    // the first 128 bytes of the A staging side.
    assert!(8 * STEP_WORDS >= 32);
    assert!(RANKED_STATIC_B_PREFIX_END == 32);
    assert!(RANKED_STATIC_B_TAIL_START == 480);
    assert!(LAST_WORD == RANKED_STATIC_B_TAIL_START + 1);
};

/// Streaming round-1 projection wired into the a/b drain: every 16-word drain
/// step is one 64-byte round-1 medium window per block, so the transform runs
/// off a `STREAM_STAGE_WORDS` staging pair as the words are produced instead
/// of off two full-block window buffers.
///
/// `stage` owns `STREAM_STAGE_WORDS` u32s (a side then b side, eight 16-word
/// block rows each) and is 64-byte aligned. `out` owns this octa's eight
/// `BYTES_PER_BLOCK` ab_inner blocks.
pub(crate) struct StreamProj<'t> {
    pub(crate) stage: *mut u32,
    /// Physical start of this octa's output region. `out_bias` maps logical
    /// ranked row 2 to physical row 0 without ever forming a pointer before
    /// the allocation.
    pub(crate) out: *mut u8,
    pub(crate) out_stride: usize,
    pub(crate) out_bias: usize,
    pub(crate) inv_table: &'t InvNttTableByteSingleGf8,
    pub(crate) plan: Round1AbWindowPlan,
    /// Ranked residual representation; dense producers leave this off.
    pub(crate) one_rows_elided: bool,
}

#[repr(C, align(64))]
struct RankedStaticWindow([u8; 64]);

const fn ranked_b30() -> [u8; 64] {
    let mut x = [0u8; 64];
    let w = 0x0001_ffff_ffff_ffffu64.to_le_bytes();
    let mut i = 0;
    while i != 8 {
        x[i] = w[i];
        i += 1;
    }
    x
}

static RANKED_B_MAX: RankedStaticWindow = RankedStaticWindow([u8::MAX; 64]);
static RANKED_B30: RankedStaticWindow = RankedStaticWindow(ranked_b30());
static RANKED_ZERO: RankedStaticWindow = RankedStaticWindow([0u8; 64]);

impl StreamProj<'_> {
    #[inline(always)]
    fn sides(&self) -> (*mut u32, *mut u32) {
        // SAFETY: the staging owns `STREAM_STAGE_WORDS` u32s.
        (self.stage, unsafe { self.stage.add(8 * STEP_WORDS) })
    }

    /// The window's per-block invariants: its static-B eligibility and the
    /// table images the kernel addresses. Both are the same for all eight
    /// blocks, so a producer that walks the eight blocks in several pieces
    /// resolves them ONCE and hands them to every piece.
    #[inline(always)]
    fn window_prep(&self, blk: usize) -> (Round1AbWindowPlan, Round1AbTableImages) {
        let p = self.plan.for_window(blk);
        (p, round1_ab_table_images(self.inv_table, p))
    }

    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked(&self, blk: usize, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: StepRows, off: *const u16, use_off: bool) {
        unsafe {
            let (sa,sb)=self.sides();
            if use_off {
                let mut j=0usize;
                while j!=8 {
                    rows.publish(j,sa,sb);
                    let out=&mut *self.out.add(j*self.out_stride+blk*64-self.out_bias).cast::<[u8;64]>();
                    round1_ab_inner_window_from_offsets(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs);
                    j+=1;
                }
            } else {
                let mut j=0usize;
                while j!=8 {
                    rows.publish(j,sa,sb);
                    let out=&mut *self.out.add(j*self.out_stride+blk*64-self.out_bias).cast::<[u8;64]>();
                    round1_ab_inner_window_with_images(&*sa.add(j*STEP_WORDS).cast::<[u8;64]>(),&*sb.add(j*STEP_WORDS).cast::<[u8;64]>(),out,blk,self.inv_table,plan,imgs);
                    j+=1;
                }
            }
        }
    }

    /// Ranked all-elide windows 2..29 always consume producer-built offsets,
    /// and every half-row is live with wide non-temporal publication. Keeping
    /// that fixed policy out of [`Self::project_blocks_ranked`] removes its
    /// policy branch and the generic row publisher from the measured path.
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets(&self, blk: usize, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, off: *const u16) {
        unsafe {
            debug_assert!(blk > 1 && blk < 30);
            if self.one_rows_elided && blk == 2 {
                self.project_blocks_ranked_hot_offsets_residual::<2,0xfc>(plan,imgs,rows,off);
                return;
            }
            if self.one_rows_elided && blk == 29 {
                self.project_blocks_ranked_hot_offsets_residual::<29,0x0f>(plan,imgs,rows,off);
                return;
            }
            // E=true is selected only for the ranked all-elide witness
            // geometry. Its structural B=1 bytes are proved by the
            // b-complement mode-plan test; the core plan separately verifies
            // the exact inverse-table identity. Keep the cold/dense fallback
            // in this same outlined boundary behind its cached kill switch.
            if plan.bcomplement_static_eligible() {
                self.project_blocks_ranked_hot_offsets_bcomplement(blk,plan,imgs,rows,off);
                return;
            }
            let (sa,sb)=self.sides();
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense(j,sa,sb);
                let out=&mut *self.out.add(j*self.out_stride+blk*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs);
                j+=1;
            }
        }
    }

    /// Ranked static-B complement projection for full windows. The producer's
    /// publish-then-project pacing is intentionally identical to the generic
    /// hot leaf; only B's table application is shortened.
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets_bcomplement(&self, blk: usize, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, off: *const u16) {
        unsafe {
            debug_assert!(blk > 1 && blk < 30);
            debug_assert!(plan.bcomplement_static_eligible());
            if ranked_b_constwin_enabled() {
                match blk {
                    3 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<3>(plan,imgs,rows,off),
                    4 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<4>(plan,imgs,rows,off),
                    5 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<5>(plan,imgs,rows,off),
                    6 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<6>(plan,imgs,rows,off),
                    7 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<7>(plan,imgs,rows,off),
                    8 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<8>(plan,imgs,rows,off),
                    9 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<9>(plan,imgs,rows,off),
                    10 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<10>(plan,imgs,rows,off),
                    11 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<11>(plan,imgs,rows,off),
                    12 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<12>(plan,imgs,rows,off),
                    13 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<13>(plan,imgs,rows,off),
                    14 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<14>(plan,imgs,rows,off),
                    15 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<15>(plan,imgs,rows,off),
                    16 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<16>(plan,imgs,rows,off),
                    17 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<17>(plan,imgs,rows,off),
                    18 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<18>(plan,imgs,rows,off),
                    19 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<19>(plan,imgs,rows,off),
                    20 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<20>(plan,imgs,rows,off),
                    21 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<21>(plan,imgs,rows,off),
                    22 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<22>(plan,imgs,rows,off),
                    23 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<23>(plan,imgs,rows,off),
                    24 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<24>(plan,imgs,rows,off),
                    25 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<25>(plan,imgs,rows,off),
                    26 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<26>(plan,imgs,rows,off),
                    27 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<27>(plan,imgs,rows,off),
                    28 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<28>(plan,imgs,rows,off),
                    _ => {}
                }
            }
            let (sa,sb)=self.sides();
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense(j,sa,sb);
                let out=&mut *self.out.add(j*self.out_stride+blk*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2_bcomplement_static(
                    &*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),
                    out,
                    plan,
                    imgs,
                    blk,
                );
                j+=1;
            }
        }
    }

    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets_bcomplement_const<const BLK: usize>(&self, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, off: *const u16) {
        unsafe {
            debug_assert!((3..=28).contains(&BLK));
            debug_assert!(plan.bcomplement_static_eligible());
            let (sa,sb)=self.sides();
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense(j,sa,sb);
                let out=&mut *self.out.add(j*self.out_stride+BLK*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2_bcomplement_static_const::<BLK>(
                    &*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),
                    out,
                    plan,
                    imgs,
                );
                j+=1;
            }
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets_direct_rollback(&self, blk: usize, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, a_rows: &[__m512i; 8], b_rows: &[__m512i; 8], off: *const u16) {
        unsafe {
            debug_assert!(blk > 1 && blk < 30);
            if self.one_rows_elided && blk == 2 {
                self.project_blocks_ranked_hot_offsets_residual_direct_rollback::<2,0xfc>(plan,imgs,rows,a_rows,b_rows,off);
                return;
            }
            if self.one_rows_elided && blk == 29 {
                self.project_blocks_ranked_hot_offsets_residual_direct_rollback::<29,0x0f>(plan,imgs,rows,a_rows,b_rows,off);
                return;
            }
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense_values(j,a_rows[j],b_rows[j]);
                let out=&mut *self.out.add(j*self.out_stride+blk*64-self.out_bias).cast::<[u8;64]>();
                if plan.bcomplement_static_eligible() {
                    round1_ab_inner_window_from_offsets_nt2_bcomplement_static(
                        &*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),
                        out,
                        plan,
                        imgs,
                        blk,
                    );
                } else {
                    round1_ab_inner_window_from_offsets_nt2(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs);
                }
                j+=1;
            }
        }
    }

    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets_residual<const BLK:usize,const KEEP:u8>(&self, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, off: *const u16) {
        unsafe {
            const { assert!((BLK==2 && KEEP==0xfc)||(BLK==29 && KEEP==0x0f)); }
            let (sa,sb)=self.sides();
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense(j,sa,sb);
                let out=&mut *self.out.add(j*self.out_stride+BLK*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2_residual(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs,KEEP);
                j+=1;
            }
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets_residual_direct_rollback<const BLK:usize,const KEEP:u8>(&self, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, a_rows: &[__m512i; 8], b_rows: &[__m512i; 8], off: *const u16) {
        unsafe {
            const { assert!((BLK==2 && KEEP==0xfc)||(BLK==29 && KEEP==0x0f)); }
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense_values(j,a_rows[j],b_rows[j]);
                let out=&mut *self.out.add(j*self.out_stride+BLK*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2_residual(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs,KEEP);
                j+=1;
            }
        }
    }

    #[inline(never)]
    unsafe fn project_blocks_ranked_static<const BLK: usize>(
        &self,
        plan: Round1AbWindowPlan,
        imgs: Round1AbTableImages,
        rows: RankedRows,
    ) {
        unsafe {
            const { assert!(BLK <= 1) };
            let mut j = 0usize;
            while j != 8 {
                rows.publish_static::<BLK>(j, self.stage);
                let a = &*self.stage.add(j * STEP_WORDS).cast::<[u8; 64]>();
                let out = &mut *self
                    .out
                    .add(j * self.out_stride + BLK * 64 - self.out_bias)
                    .cast::<[u8; 64]>();
                round1_ab_inner_window_with_images(
                    a,
                    &RANKED_B_MAX.0,
                    out,
                    BLK,
                    self.inv_table,
                    plan,
                    imgs,
                );
                j += 1;
            }
        }
    }

    #[inline(never)]
    unsafe fn project_blocks_ranked_zero(
        &self,
        plan: Round1AbWindowPlan,
        imgs: Round1AbTableImages,
    ) {
        unsafe {
            let mut j = 0usize;
            while j != 8 {
                let out = &mut *self
                    .out
                    .add(j * self.out_stride + 31 * 64 - self.out_bias)
                    .cast::<[u8; 64]>();
                round1_ab_inner_window_with_images(
                    &RANKED_ZERO.0,
                    &RANKED_ZERO.0,
                    out,
                    31,
                    self.inv_table,
                    plan,
                    imgs,
                );
                j += 1;
            }
        }
    }

    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_30(&self, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows) {
        unsafe {
            // The first eight qwords are the eight blocks' real row-0 A
            // values. The following eight are initialized guard bytes. A
            // sliding 64-byte reference starting at qword j is therefore
            // valid even for j=7. Its later rows may belong to another block,
            // but B30 rows 1..7 are exactly zero, so both the trusted kernel
            // and its generic kill-switch fallback produce the same product.
            let q=self.stage.cast::<u64>();
            let mut j=0usize;
            while j!=8 {
                rows.publish_sparse_30(j,q.add(j));
                let a=&*q.add(j).cast::<[u8;64]>();
                let out=&mut *self.out.add(j*self.out_stride+30*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window30_k0(a,&RANKED_B30.0,out,self.inv_table,plan,imgs,30);
                j+=1;
            }
        }
    }
}

/// Rolling drain state shared by the A/B packed writers. The witness uses two
/// reusable `RING_WORDS`-word epochs instead of full 512-word stages; B's
/// writer flushes after A has written the epoch, and Z is derived from the
/// transposed A/B rows during that flush.
struct Drain8<'t> {
    ast: *mut V8,
    bs: *mut V8,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    proj: StreamProj<'t>,
    elide: [bool; 3],
    ranked_static: bool,
}

/// Convert one low-aligned prior bit to the representation used by [`W8`].
/// Sapphire Rapids keeps pending bits at the high end of each lane so VBMI2
/// can join them to the next field with one `vpshldd`; the portable AVX2
/// fallback retains the incumbent low-aligned representation.
#[inline(always)]
fn packer_initial_bit(bit: V8) -> V8 {
    #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
    {
        shl_v8::<31>(bit)
    }
    #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
    {
        bit
    }
}

/// Recover low-aligned pending bits from the VBMI2 representation. `BACK` is
/// `32 - used`; the ranked stream finishes with `used = 17`, hence `BACK=15`.
#[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
#[inline(always)]
fn packer_high_to_low<const BACK: i32>(pending: V8) -> V8 {
    const {
        assert!(BACK > 0 && BACK < 32);
    }
    shr_v8::<BACK>(pending)
}

/// Lane-wise packed-word writer: 8 independent `PackedWordWriter`s.
///
/// On AVX-512VBMI2+VL targets `pending` is high-aligned: if `USED = u > 0`,
/// its pending low bits occupy bits `32-u..31`.  This makes
/// `vpshldd(v, pending, u)` equal `(v << u) | pending_low` in one instruction.
/// Bits below that high-aligned range are don't-care. Other targets keep the
/// incumbent low-aligned `pending` representation.
struct W8<'t, const FLUSH: bool> {
    pending: V8,
    stage: *mut V8,
    drain: *mut Drain8<'t>,
    ranked_static: bool,
}

impl<'t, const FLUSH: bool> W8<'t, FLUSH> {
    #[inline(always)]
    fn at(stage: *mut V8, pending: V8, drain: *mut Drain8<'t>, ranked_static: bool) -> Self {
        Self {
            pending,
            stage,
            drain,
            ranked_static,
        }
    }

    #[inline(always)]
    unsafe fn write_word<const WORD: usize>(&mut self, v: V8) {
        unsafe {
            // In the ranked all-elide path, static windows provide B words
            // 0..31 and 480..511 without touching this ring. Keep the flush
            // check independent of the store: a skipped epoch-ending B word
            // must still publish the completed A/B epoch.
            let static_b = FLUSH
                && (WORD < RANKED_STATIC_B_PREFIX_END || WORD >= RANKED_STATIC_B_TAIL_START)
                && self.ranked_static;
            if !static_b {
                store_v8(self.stage.add(WORD & (RING_WORDS - 1)) as *mut u32, v);
            }
            if FLUSH && WORD % RING_WORDS == RING_WORDS - 1 {
                // Words 0..15 cannot be published until the final chaining
                // value is known.  The first rolling epoch therefore starts
                // at word 16; later epochs cover their complete 128 words.
                if WORD + 1 == RING_WORDS {
                    (*self.drain).drain_range(16, 16, RING_WORDS - 16);
                } else {
                    (*self.drain).drain_range(WORD + 1 - RING_WORDS, 0, RING_WORDS);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn push<const USED: i32, const WIDTH: i32, const BACK: i32, const WORD: usize>(
        &mut self,
        v: V8,
    ) {
        const {
            assert!(USED >= 0 && USED < 32);
            assert!(WIDTH == 31 || WIDTH == 32);
            assert!(BACK >= 1 && BACK < 32);
            assert!(WORD < U32_PER_BLOCK);
        }
        debug_assert!(USED + WIDTH <= 32 || BACK == 32 - USED);
        unsafe {
            #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
            {
                // Every u>0 push crosses a word boundary because WIDTH is 31
                // or 32.  VPSHLD supplies the old high-aligned pending bits
                // from its second source while shifting the new field left.
                if USED == 0 {
                    if WIDTH == 32 {
                        self.write_word::<WORD>(v);
                        self.pending = dup_u32(0);
                    } else {
                        // 31 pending low bits, moved to bits 1..31.
                        self.pending = shl_v8::<1>(v);
                    }
                } else {
                    let out = _mm256_shldi_epi32::<USED>(v, self.pending);
                    self.write_word::<WORD>(out);
                    // A crossed 31-bit field leaves u-1 low bits; high-aligning
                    // them is exactly v<<1.  A 32-bit field leaves u bits, so
                    // v itself already is the high-aligned representation.
                    self.pending = if WIDTH == 31 { shl_v8::<1>(v) } else { v };
                }
            }
            #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
            {
                if USED == 0 {
                    if WIDTH == 32 {
                        self.write_word::<WORD>(v);
                        self.pending = dup_u32(0);
                    } else {
                        self.pending = v;
                    }
                } else if USED + WIDTH < 32 {
                    self.pending = vsli_v8::<USED>(self.pending, v);
                } else {
                    let out = vsli_v8::<USED>(self.pending, v);
                    self.write_word::<WORD>(out);
                    if USED + WIDTH == 32 {
                        self.pending = dup_u32(0);
                    } else {
                        self.pending = shr_v8::<BACK>(v);
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn finish(&mut self) {
        const {
            assert!(USEFUL_BITS % 32 == 17);
        }
        unsafe {
            // The ranked stream finishes with 17 pending bits.  The VBMI2
            // representation holds those bits in 15..31; the incumbent
            // fallback is already low-aligned.
            #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
            let pending = packer_high_to_low::<15>(self.pending);
            #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
            let pending = self.pending;
            self.write_word::<LAST_WORD>(pending);
        }
    }
}

macro_rules! pushf8 {
    ($w:ident, $pos:expr, $width:literal, $v:expr) => {{
        $w.push::<{ ($pos % 32) as i32 }, $width, {
            let u = ($pos % 32) as i32;
            if u == 0 { 1 } else { 32 - u }
        }, { $pos / 32 }>($v);
    }};
}

#[inline(always)]
fn add_carry_parts_v8(x: V8, y: V8) -> (V8, V8, V8) {
    // `cin = sum ^ x ^ y` is never consumed directly: the pushed parts are
    // `left = x ^ cin` and `right = y ^ cin`, and both collapse algebraically
    // (`left = sum ^ y`, `right = sum ^ x`). Computing them off `sum` removes
    // the carry-in XOR chain entirely — bit-identical outputs, one op less.
    let sum = add_v8(x, y);
    let left = xor_v8(sum, y);
    let right = xor_v8(sum, x);
    (sum, left, right)
}

#[inline(always)]
fn xor_rotr8<const N: i32, const M: i32>(x: V8, y: V8) -> V8 {
    debug_assert_eq!(N + M, 32);
    let v = xor_v8(x, y);
    ror_v8::<N>(v)
}

/// Rotate-right by N on AVX-512 (single `vprold`, EVEX.256) -- three ops on
/// AVX2. Bit-identical either way; the witness phase is ALU-bound so the
/// uop cut matters on the ranked runner. The EVEX form needs `avx512f`.
#[inline(always)]
fn ror_v8<const N: i32>(v: V8) -> V8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // SAFETY: target feature checked at compile time.
        unsafe { _mm256_ror_epi32::<N>(v) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        // N is a monomorphized literal (16/12/8/7 in the G-functions), so the
        // match folds to the single shift+or pair at compile time.
        match N {
            16 => or_v8(shr_v8::<16>(v), shl_v8::<16>(v)),
            12 => or_v8(shr_v8::<12>(v), shl_v8::<20>(v)),
            8 => or_v8(shr_v8::<8>(v), shl_v8::<24>(v)),
            7 => or_v8(shr_v8::<7>(v), shl_v8::<25>(v)),
            _ => panic!("unexpected rotate count {N}"),
        }
    }
}

/// Drain 8 consecutive stage words (`dump` chunk `g`) to eight row-major
/// 32-byte block runs. Temporal stores only.
#[allow(dead_code)]
#[inline(always)]
unsafe fn dump_range(stage: *const V8, dst: *mut u32, g0: usize, g1: usize) {
    unsafe {
        for g in g0..g1 {
            let w = 8 * g;
            let r0 = load_v8(stage.add(w) as *const u32);
            let r1 = load_v8(stage.add(w + 1) as *const u32);
            let r2 = load_v8(stage.add(w + 2) as *const u32);
            let r3 = load_v8(stage.add(w + 3) as *const u32);
            let r4 = load_v8(stage.add(w + 4) as *const u32);
            let r5 = load_v8(stage.add(w + 5) as *const u32);
            let r6 = load_v8(stage.add(w + 6) as *const u32);
            let r7 = load_v8(stage.add(w + 7) as *const u32);
            let t = tr8(r0, r1, r2, r3, r4, r5, r6, r7);
            store_v8(dst.add(w), t[0]);
            store_v8(dst.add(U32_PER_BLOCK + w), t[1]);
            store_v8(dst.add(2 * U32_PER_BLOCK + w), t[2]);
            store_v8(dst.add(3 * U32_PER_BLOCK + w), t[3]);
            store_v8(dst.add(4 * U32_PER_BLOCK + w), t[4]);
            store_v8(dst.add(5 * U32_PER_BLOCK + w), t[5]);
            store_v8(dst.add(6 * U32_PER_BLOCK + w), t[6]);
            store_v8(dst.add(7 * U32_PER_BLOCK + w), t[7]);
        }
    }
}

/// Non-temporal twin of [`dump_range`]: identical bytes. Recyclable-class
/// destinations are 64-aligned on this lineage, and `U32_PER_BLOCK = 512`
/// keeps every row start 64-aligned too, so a pair of 32-byte V8s is one
/// cache line. Publish that line with a single ZMM stream when `avx512f` is
/// compiled in, otherwise one YMM stream per V8. `FLOCK_NO_WIDE_NT=1` keeps
/// the historical two-XMM form. Chunks still drain in PAIRS so each line's
/// write-combining buffer closes as soon as it fills.
///
/// Caller contract: destinations are not read again until after an
/// `_mm_sfence()` on this thread (the witness task issues one per rayon
/// task; same-thread reads are self-consistent regardless).
#[allow(dead_code)]
#[inline(always)]
unsafe fn dump_range_nt(stage: *const V8, dst: *mut u32, g0: usize, g1: usize, wide_nt: bool) {
    unsafe {
        debug_assert_eq!(dst as usize % 16, 0);
        let mut g = g0;
        while g + 2 <= g1 {
            let w = 8 * g;
            let ta = tr8_chunk(stage, w);
            let tb = tr8_chunk(stage, w + 8);
            for r in 0..8 {
                stream_pair_v8(dst.add(r * U32_PER_BLOCK + w), ta[r], tb[r], wide_nt);
            }
            g += 2;
        }
        if g < g1 {
            let w = 8 * g;
            let t = tr8_chunk(stage, w);
            for r in 0..8 {
                stream_v8(dst.add(r * U32_PER_BLOCK + w), t[r], wide_nt);
            }
        }
    }
}

/// Transpose the eight stage words at `w` (one `dump` chunk) into eight
/// row-major 32-byte runs.
#[inline(always)]
unsafe fn tr8_chunk(stage: *const V8, w: usize) -> [V8; 8] {
    unsafe {
        tr8(
            load_v8(stage.add(w) as *const u32),
            load_v8(stage.add(w + 1) as *const u32),
            load_v8(stage.add(w + 2) as *const u32),
            load_v8(stage.add(w + 3) as *const u32),
            load_v8(stage.add(w + 4) as *const u32),
            load_v8(stage.add(w + 5) as *const u32),
            load_v8(stage.add(w + 6) as *const u32),
            load_v8(stage.add(w + 7) as *const u32),
        )
    }
}

/// Publish one 32-byte transposed run non-temporally.
///
/// # Safety
/// Caller guarantees 16-byte alignment of `p`; the YMM arm additionally
/// requires 32-byte alignment (true for every in-loop pointer when `dst` is
/// 32-aligned and `w` is a multiple of 8).
#[inline(always)]
unsafe fn stream_v8(p: *mut u32, v: V8, wide_nt: bool) {
    unsafe {
        if wide_nt && (p as usize).is_multiple_of(32) {
            _mm256_stream_si256(p.cast::<__m256i>(), v);
            return;
        }
        _mm_stream_si128(p.cast::<__m128i>(), _mm256_castsi256_si128(v));
        _mm_stream_si128(p.add(4).cast::<__m128i>(), _mm256_extracti128_si256::<1>(v));
    }
}

/// Publish a chunk PAIR — two consecutive 32-byte runs, i.e. one 64-byte
/// cache line when `p` is line-aligned — non-temporally, closing the line's
/// write-combining buffer in one shot where the ISA allows it.
///
/// # Safety
/// Same alignment contract as [`stream_v8`], for both `p` and `p.add(8)`.
#[inline(always)]
unsafe fn stream_pair_v8(p: *mut u32, va: V8, vb: V8, wide_nt: bool) {
    unsafe {
        #[cfg(target_feature = "avx512f")]
        if wide_nt && p as usize % 64 == 0 {
            let z = _mm512_castsi256_si512(va);
            let z = _mm512_inserti64x4::<1>(z, vb);
            _mm512_stream_si512(p.cast::<__m512i>(), z);
            return;
        }
        stream_v8(p, va, wide_nt);
        stream_v8(p.add(8), vb, wide_nt);
    }
}

/// Fixed ranked publisher for one complete cache line.
///
/// Unlike [`stream_pair_v8`], this has no width policy, alignment branch, or
/// YMM-to-ZMM assembly. The ranked caller supplies an already-complete ZMM and
/// a line-aligned destination, so publication is exactly one `vmovntdq`.
///
/// # Safety
/// `p` is 64-byte aligned and owns 64 writable bytes that are not observed
/// before the task-level `_mm_sfence()`.
#[cfg(target_feature = "avx512f")]
#[inline(always)]
unsafe fn stream_ranked_line(p: *mut u32, v: __m512i) {
    unsafe { _mm512_stream_si512(p.cast::<__m512i>(), v) }
}

/// Publish one row of a drain step to `p` (its low 32-byte run) and `p+8`
/// (its high run), under the same `nt`/liveness policy — and therefore with
/// the same bytes at the same addresses as the ranked drain publisher.
///
/// # Safety
/// As for [`stream_pair_v8`] and [`store_v8`].
#[inline(always)]
unsafe fn emit_pair(
    p: *mut u32,
    lo: V8,
    hi: V8,
    lo_live: bool,
    hi_live: bool,
    nt: bool,
    wide_nt: bool,
) {
    unsafe {
        match (nt, lo_live, hi_live) {
            (true, true, true) => stream_pair_v8(p, lo, hi, wide_nt),
            (true, true, false) => stream_v8(p, lo, wide_nt),
            (true, false, true) => stream_v8(p.add(8), hi, wide_nt),
            (false, true, true) => {
                store_v8(p, lo);
                store_v8(p.add(8), hi);
            }
            (false, true, false) => store_v8(p, lo),
            (false, false, true) => store_v8(p.add(8), hi),
            (_, false, false) => {}
        }
    }
}

/// One drain step's three destination rows, staged for publication BETWEEN
/// block transforms by [`StreamProj::project_blocks_ranked`].
///
/// Nothing here is kept in registers across the transform: a's and b's bytes
/// are already in the projection's own staging (they are its input). The
/// publisher loads each staged row once and derives z immediately, before the
/// transform, so no separate z-row scratch has to survive the outlined call.
#[derive(Clone, Copy)]
struct StepRows {
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    /// bit 0 z-lo, 1 z-hi, 2 a-lo, 3 a-hi, 4 b-lo, 5 b-hi live;
    /// bit 6 z non-temporal, bit 7 wide streaming stores.
    flags: u8,
}

impl StepRows {
    /// Publish block `j`'s three rows.
    ///
    /// # Safety
    /// `self` describes one live drain step and `j < 8`.
    #[inline(always)]
    unsafe fn publish(&self, j: usize, sa: *const u32, sb: *const u32) {
        unsafe {
            let f = self.flags;
            let wide = f & 0x80 != 0;
            let o = j * U32_PER_BLOCK;
            let ap = sa.add(j * STEP_WORDS);
            let a_lo = load_v8(ap);
            let a_hi = load_v8(ap.add(8));
            let bp = sb.add(j * STEP_WORDS);
            let b_lo = load_v8(bp);
            let b_hi = load_v8(bp.add(8));
            emit_pair(
                self.z.add(o),
                and_v8(a_lo, b_lo),
                and_v8(a_hi, b_hi),
                f & 1 != 0,
                f & 2 != 0,
                f & 0x40 != 0,
                wide,
            );
            emit_pair(
                self.a.add(o),
                a_lo,
                a_hi,
                f & 4 != 0,
                f & 8 != 0,
                true,
                wide,
            );
            emit_pair(
                self.b.add(o),
                b_lo,
                b_hi,
                f & 0x10 != 0,
                f & 0x20 != 0,
                true,
                wide,
            );
        }
    }
}

/// Fixed-policy row publisher for the measured all-elide path.
///
/// The ranked AVX-512 binary installs `RecycleAlloc` globally. Its cleared
/// environment selects the allocator's 64-byte-aligned recyclable class for
/// the large z/a/b vectors. Rayon group starts (32 KiB), octa-half starts
/// (16 KiB), block rows (2 KiB), and step offsets (64 B) all preserve that
/// base alignment. `StreamProj::stage` is backed by `AbWinLine`, also aligned
/// to 64 bytes, and its side/row strides preserve it. Consequently no ranked
/// store needs to carry the generic publisher's flags, width policy, or
/// runtime alignment recovery.
#[derive(Clone, Copy)]
struct RankedRows {
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
}

impl RankedRows {
    #[inline(always)]
    fn new(z: *mut u32, a: *mut u32, b: *mut u32) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            debug_assert!((z as usize).is_multiple_of(64));
            debug_assert!((a as usize).is_multiple_of(64));
            debug_assert!((b as usize).is_multiple_of(64));
        }
        Self { z, a, b }
    }

    /// Dense ranked windows 2..29: z, a, and b are all fully live.
    #[inline(always)]
    unsafe fn publish_dense(&self, j: usize, sa: *const u32, sb: *const u32) {
        unsafe {
            let o = j * U32_PER_BLOCK;
            let ap = sa.add(j * STEP_WORDS);
            let bp = sb.add(j * STEP_WORDS);
            #[cfg(target_feature = "avx512f")]
            {
                let av = _mm512_load_si512(ap.cast::<__m512i>());
                let bv = _mm512_load_si512(bp.cast::<__m512i>());
                stream_ranked_line(self.z.add(o), _mm512_and_si512(av, bv));
                stream_ranked_line(self.a.add(o), av);
                stream_ranked_line(self.b.add(o), bv);
            }
            #[cfg(not(target_feature = "avx512f"))]
            {
                let a_lo = load_v8(ap);
                let a_hi = load_v8(ap.add(8));
                let b_lo = load_v8(bp);
                let b_hi = load_v8(bp.add(8));
                stream_pair_v8(self.z.add(o), and_v8(a_lo, b_lo), and_v8(a_hi, b_hi), true);
                stream_pair_v8(self.a.add(o), a_lo, a_hi, true);
                stream_pair_v8(self.b.add(o), b_lo, b_hi, true);
            }
        }
    }

    /// Dense ranked windows 2..29, sourced directly from the live transposed
    /// rows instead of staging and immediately reloading them.
    #[cfg(target_feature = "avx512f")]
    #[inline(always)]
    unsafe fn publish_dense_values(&self, j: usize, av: __m512i, bv: __m512i) {
        unsafe {
            let o = j * U32_PER_BLOCK;
            stream_ranked_line(self.z.add(o), _mm512_and_si512(av, bv));
            stream_ranked_line(self.a.add(o), av);
            stream_ranked_line(self.b.add(o), bv);
        }
    }

    /// Static-B ranked windows 0/1: B=1, so the complete z and a lines match.
    #[inline(always)]
    unsafe fn publish_static<const BLK: usize>(&self, j: usize, sa: *const u32) {
        unsafe {
            const { assert!(BLK <= 1) };
            let o = j * U32_PER_BLOCK;
            let ap = sa.add(j * STEP_WORDS);
            #[cfg(target_feature = "avx512f")]
            {
                let av = _mm512_load_si512(ap.cast::<__m512i>());
                stream_ranked_line(self.z.add(o), av);
                stream_ranked_line(self.a.add(o), av);
            }
            #[cfg(not(target_feature = "avx512f"))]
            {
                let a_lo = load_v8(ap);
                let a_hi = load_v8(ap.add(8));
                stream_pair_v8(self.z.add(o), a_lo, a_hi, true);
                stream_pair_v8(self.a.add(o), a_lo, a_hi, true);
            }
        }
    }

    /// Sparse ranked window 30: only its first qword is live; publish the
    /// complete line with the other seven qwords zeroed in the load itself.
    #[inline(always)]
    unsafe fn publish_sparse_30(&self, j: usize, p: *const u64) {
        unsafe {
            let o = j * U32_PER_BLOCK;
            #[cfg(target_feature = "avx512f")]
            {
                let av = _mm512_maskz_loadu_epi64(1, p.cast::<i64>());
                stream_ranked_line(self.z.add(o), av);
                stream_ranked_line(self.a.add(o), av);
            }
            #[cfg(not(target_feature = "avx512f"))]
            {
                let lo = _mm256_set_epi64x(0, 0, 0, core::ptr::read_unaligned(p) as i64);
                let hi = _mm256_setzero_si256();
                stream_pair_v8(self.z.add(o), lo, hi, true);
                stream_pair_v8(self.a.add(o), lo, hi, true);
            }
        }
    }
}

impl Drain8<'_> {
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn drain_range(&mut self, base_word: usize, ring_word: usize, words: usize) {
        unsafe {
            if self.ranked_static{self.drain_range_spread::<true>(&self.proj,base_word,ring_word,words)}else{self.drain_range_spread::<false>(&self.proj,base_word,ring_word,words)};
        }
    }

    /// Same bytes as the bunched arm of [`Self::drain_range`], published on a
    /// different SCHEDULE.
    ///
    /// The bunched form emits all twenty-four of a step's 64-byte streaming
    /// stores back to back — eight rows 2048 bytes apart, three destinations —
    /// and then runs ~1000 core-cycles of projection with none. Twenty-four
    /// lines in ~70 core-cycles is ~8x this core's sustained non-temporal
    /// write rate, so the write-combining buffers saturate, the streaming
    /// store at the head of the store buffer cannot retire, and allocation
    /// stalls behind it (`resource_stalls.sb`).
    ///
    /// Here nothing is published until the staging is complete, and then the
    /// window's eight block transforms publish their OWN three rows as they
    /// go. Identical addresses, identical bytes, three lines per block instead
    /// of twenty-four per step — and still exactly one call per step, so the
    /// spread costs no extra call or spill traffic.
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn drain_range_spread<const E:bool>(
        &self,
        proj: &StreamProj<'_>,
        base_word: usize,
        ring_word: usize,
        words: usize,
    ) {
        unsafe {
            let z_g1=if E||self.elide[0]{ELIDE_ZERO_CHUNK}else{DUMP_CHUNKS};
            let a_g1=if E||self.elide[1]{ELIDE_ZERO_CHUNK}else{DUMP_CHUNKS};
            let b_g0=if E||self.elide[2]{ELIDE_B_PREFIX_CHUNKS}else{0};
            let b_g1=if E||self.elide[2]{ELIDE_B_TAIL_CHUNK_WIN}else{DUMP_CHUNKS};
            let (sa, sb) = proj.sides();
            // Window 31 is `w = 1, b_med = 15`, one past round one's last live
            // b_med at the pinned BLAKE3 shape — its 64 output bytes are never
            // read, exactly as windows 0 and 1 are never read under the same
            // `one_rows_elided` brand. Resolved once per call, not per window.
            let dead_w31 = E && proj.one_rows_elided && ey_dead_w31_enabled();
            for off in (0..words).step_by(STEP_WORDS) {
                let abs_word = base_word + off;
                let rw = ring_word + off;
                let blk = abs_word / STEP_WORDS;
                if dead_w31 && blk == 31 {
                    continue;
                }
                let (plan, imgs) = proj.window_prep(blk);
                if E && (blk <= 1 || blk == 30 || blk == 31) {
                    let rows=RankedRows::new(self.z.add(abs_word),self.a.add(abs_word),self.b.add(abs_word));
                    if blk == 30 {
                        // Only A words 480 and 481 are live. Pair those two
                        // word-major ring vectors into eight block-major
                        // qwords, followed by a fully initialized 64-byte
                        // guard for the sliding synthetic A windows.
                        let x0=load_v8(self.ast.add(rw).cast::<u32>());
                        let x1=load_v8(self.ast.add(rw+1).cast::<u32>());
                        let l=_mm256_unpacklo_epi32(x0,x1);
                        let h=_mm256_unpackhi_epi32(x0,x1);
                        store_v8(sa,_mm256_permute2x128_si256::<0x20>(l,h));
                        store_v8(sa.add(8),_mm256_permute2x128_si256::<0x31>(l,h));
                        let z=_mm256_setzero_si256();
                        store_v8(sa.add(16),z);
                        store_v8(sa.add(24),z);
                        proj.project_blocks_ranked_30(plan,imgs,rows);
                        continue;
                    }
                    if blk != 31 {
                        let a_lo = tr8_chunk(self.ast, rw);
                        let a_hi = tr8_chunk(self.ast, rw + 8);
                        for r in 0..8 {
                            let p = sa.add(r * STEP_WORDS);
                            store_v8(p, a_lo[r]);
                            store_v8(p.add(8), a_hi[r]);
                        }
                    }
                    if blk == 0 {
                        if proj.one_rows_elided {
                            let mut j=0usize;
                            while j!=8 { rows.publish_static::<0>(j,proj.stage); j+=1; }
                        } else {
                            proj.project_blocks_ranked_static::<0>(plan, imgs, rows);
                        }
                    } else if blk == 1 {
                        if proj.one_rows_elided {
                            let mut j=0usize;
                            while j!=8 { rows.publish_static::<1>(j,proj.stage); j+=1; }
                        } else {
                            proj.project_blocks_ranked_static::<1>(plan, imgs, rows);
                        }
                    } else {
                        proj.project_blocks_ranked_zero(plan, imgs);
                    }
                    continue;
                }
                // Fused offset arena: while the transposed rows are still in
                // registers, also widen them to the split kernel's pre-scaled
                // `u16` offsets. The kernel then consumes the arena and never
                // reloads the staging, and no consuming load executes in the
                // shadow of its own offset stores.
                // The E=true static cases continued above, so its remaining
                // windows are exactly 2..29 and offsets are unconditional.
                // E=false retains the original per-window eligibility test.
                #[repr(align(64))]
                struct OffArena([u16; 8 * ROUND1_AB_OFF_WORDS]);
                let mut arena = core::mem::MaybeUninit::<OffArena>::uninit();
                let op = core::ptr::addr_of_mut!((*arena.as_mut_ptr()).0) as *mut u16;

                if E {
                    // Ranked dense path: each side's two AVX2 half-transposes
                    // become one 16-word ZMM transpose, and the same live ZMM
                    // rows feed offset widening. The incumbent fallback stages
                    // those rows once; the direct arm streams them straight to
                    // z/a/b.
                    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
                    {
                        let rows=RankedRows::new(self.z.add(abs_word),self.a.add(abs_word),self.b.add(abs_word));
                        if ranked_direct_dense_publish_enabled() {
                            if ranked_direct_dense_inline_enabled() {
                                project_blocks_ranked_hot_offsets_direct_inline(
                                    proj,
                                    blk,
                                    plan,
                                    imgs,
                                    rows,
                                    self.ast,
                                    self.bs,
                                    rw,
                                    op,
                                );
                            } else {
                                let a_rows=ranked_dense_rows_and_offsets_rollback(self.ast,rw,op);
                                let b_rows=ranked_dense_rows_and_offsets_rollback(self.bs,rw,op.add(64));
                                proj.project_blocks_ranked_hot_offsets_direct_rollback(blk,plan,imgs,rows,&a_rows,&b_rows,op as *const u16);
                            }
                        } else {
                            stage_ranked_dense_side(self.ast,rw,sa,op);
                            stage_ranked_dense_side(self.bs,rw,sb,op.add(64));
                            proj.project_blocks_ranked_hot_offsets(blk,plan,imgs,rows,op as *const u16);
                        }
                    }
                    // Portable builds cannot select E=true: the ranked gate
                    // requires the AVX-512 offset plan. Retain the old staging
                    // shape so this monomorphization remains well-formed.
                    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
                    {
                        let a_lo=tr8_chunk(self.ast,rw);
                        let a_hi=tr8_chunk(self.ast,rw+8);
                        for r in 0..8 {
                            let p=sa.add(r*STEP_WORDS);
                            store_v8(p,a_lo[r]);
                            store_v8(p.add(8),a_hi[r]);
                        }
                        let b_lo=tr8_chunk(self.bs,rw);
                        let b_hi=tr8_chunk(self.bs,rw+8);
                        for r in 0..8 {
                            let p=sb.add(r*STEP_WORDS);
                            store_v8(p,b_lo[r]);
                            store_v8(p.add(8),b_hi[r]);
                        }
                        let rows=RankedRows::new(self.z.add(abs_word),self.a.add(abs_word),self.b.add(abs_word));
                        proj.project_blocks_ranked_hot_offsets(blk,plan,imgs,rows,op as *const u16);
                    }
                } else {
                    // Cold/generic path stays on the two incumbent AVX2
                    // transposes and its per-window offset eligibility gate.
                    let use_off=plan.offsets_eligible(blk);
                    let a_lo=tr8_chunk(self.ast,rw);
                    let a_hi=tr8_chunk(self.ast,rw+8);
                    for r in 0..8 {
                        let p=sa.add(r*STEP_WORDS);
                        store_v8(p,a_lo[r]);
                        store_v8(p.add(8),a_hi[r]);
                        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
                        if use_off {
                            widen_off_half(a_lo[r],a_hi[r],op.add(r*ROUND1_AB_OFF_WORDS));
                        }
                    }
                    let b_lo=tr8_chunk(self.bs,rw);
                    let b_hi=tr8_chunk(self.bs,rw+8);
                    for r in 0..8 {
                        let p=sb.add(r*STEP_WORDS);
                        store_v8(p,b_lo[r]);
                        store_v8(p.add(8),b_hi[r]);
                        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
                        if use_off {
                            widen_off_half(b_lo[r],b_hi[r],op.add(r*ROUND1_AB_OFF_WORDS+64));
                        }
                    }
                    let base=0xC0u8;
                    let g = abs_word / 8;
                    let mut flags = base;
                    if g < z_g1 {
                        flags |= 1;
                    }
                    if g + 1 < z_g1 {
                        flags |= 2;
                    }
                    if g < a_g1 {
                        flags |= 4;
                    }
                    if g + 1 < a_g1 {
                        flags |= 8;
                    }
                    if g >= b_g0 && g < b_g1 {
                        flags |= 0x10;
                    }
                    if g + 1 >= b_g0 && g + 1 < b_g1 {
                        flags |= 0x20;
                    }
                    let rows=StepRows { z:self.z.add(abs_word), a:self.a.add(abs_word), b:self.b.add(abs_word), flags };
                    proj.project_blocks_ranked(blk,plan,imgs,rows,op as *const u16,use_off);
                }
            }
        }
    }
}

/// Dual-destination twin of [`dump_range_nt`] for the a/b sides: one
/// transpose feeds BOTH
///  * `dst` — the main witness buffer, published NON-TEMPORALLY over the
///    un-elided chunk range `[g0, g1)`, in the same paired-chunk order (and
///    with the same bytes) `dump_range_nt(stage, dst, g0, g1)` would use; and
///  * `win` — a compact 8-block window buffer with the same row-major
///    geometry (`U32_PER_BLOCK` row stride), written TEMPORALLY over ALL of
///    `0..DUMP_CHUNKS`.
///
/// `win` deliberately ignores the elide range, so it always carries the FULL
/// `U32_PER_BLOCK` words per block. The round-1 window projection reads a
/// whole block; eliding a `dst` chunk is only legal because `dst` already
/// holds those exact constant bytes from a previous witgen (pool provenance
/// token), and those constants are *not* uniformly zero — b's elided prefix
/// is all-ones. Rebuilding every chunk into `win` rather than zero-filling
/// (or reading `dst` back) keeps the projection's input byte-identical to the
/// incumbent's for every elide setting, by construction.
///
/// # Safety
/// AVX2 required. `dst` owns 8 contiguous `U32_PER_BLOCK`-word blocks and is
/// 16-byte aligned; `win` owns 8 contiguous `U32_PER_BLOCK`-word blocks and
/// is disjoint from `dst` and from `stage`.
#[allow(dead_code)]
#[inline(always)]
unsafe fn dump_range_nt_win(
    stage: *const V8,
    dst: *mut u32,
    win: *mut u32,
    g0: usize,
    g1: usize,
    wide_nt: bool,
) {
    unsafe {
        debug_assert_eq!(dst as usize % 16, 0);
        let mut g = 0usize;
        while g < DUMP_CHUNKS {
            let w = 8 * g;
            let ta = tr8_chunk(stage, w);
            let tb = tr8_chunk(stage, w + 8);
            // Window first: plain stores to a 16 KiB L1-resident buffer, and
            // grouping them ahead of the streams keeps each row's write-
            // combining buffer open across consecutive NT stores.
            for r in 0..8 {
                let p = win.add(r * U32_PER_BLOCK + w);
                store_v8(p, ta[r]);
                store_v8(p.add(8), tb[r]);
            }
            let lo = g >= g0 && g < g1;
            let hi = g + 1 >= g0 && g + 1 < g1;
            if lo && hi {
                for r in 0..8 {
                    stream_pair_v8(dst.add(r * U32_PER_BLOCK + w), ta[r], tb[r], wide_nt);
                }
            } else if lo {
                for r in 0..8 {
                    stream_v8(dst.add(r * U32_PER_BLOCK + w), ta[r], wide_nt);
                }
            } else if hi {
                for r in 0..8 {
                    stream_v8(dst.add(r * U32_PER_BLOCK + w + 8), tb[r], wide_nt);
                }
            }
            g += 2;
        }
    }
}

#[allow(dead_code)]
#[inline(always)]
unsafe fn dump_elide(
    stage: *const V8,
    dst: *mut u32,
    elide_tail: bool,
    elide_prefix: bool,
    tail_chunk: usize,
    nt: bool,
    wide_nt: bool,
) {
    let g0 = if elide_prefix {
        ELIDE_B_PREFIX_CHUNKS
    } else {
        0
    };
    let g1 = if elide_tail { tail_chunk } else { DUMP_CHUNKS };
    unsafe {
        if nt {
            dump_range_nt(stage, dst, g0, g1, wide_nt)
        } else {
            dump_range(stage, dst, g0, g1)
        }
    }
}

/// [`dump_elide`]'s dual-destination form: same elide range selection, always
/// non-temporal into `dst`, always FULL into `win`. See [`dump_range_nt_win`].
#[allow(dead_code)]
#[inline(always)]
unsafe fn dump_elide_win(
    stage: *const V8,
    dst: *mut u32,
    win: *mut u32,
    elide_tail: bool,
    elide_prefix: bool,
    tail_chunk: usize,
    wide_nt: bool,
) {
    let g0 = if elide_prefix {
        ELIDE_B_PREFIX_CHUNKS
    } else {
        0
    };
    let g1 = if elide_tail { tail_chunk } else { DUMP_CHUNKS };
    unsafe { dump_range_nt_win(stage, dst, win, g0, g1, wide_nt) }
}

/// Build `(z, a, b)` for EIGHT compressions in u32-lane lockstep.
/// Bit-exact with two 4-wide quads and with the scalar driver ×8.
///
/// `proj` selects the STREAMING form of the a/b fusion: a/b are
/// published non-temporally, and each drain step's
/// eight 64-byte round-1 medium windows are transformed straight into the
/// caller's ab_inner blocks out of a small staging pair, so no full-block
/// window buffer exists.
///
/// # Safety
/// Caller must have AVX2. `z`/`a`/`b` each own 8 contiguous 512-word blocks.
/// `proj`'s staging and `out` satisfy [`StreamProj`]'s contract. In every
/// non-temporal arm the caller must `_mm_sfence()` on this thread after its
/// last octa, before releasing a/b to another thread (same-thread reads are
/// self-consistent regardless).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn build_octa_witness_ab_stream_elide(
    inputs: OctaInputs<'_>,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    proj: StreamProj<'_>,
    elide: [bool; 3],
) {
    unsafe {
        // Only the all-elide provenance state selects the ranked static
        // windows. Partial/cold states still read both rings in every window.
        let ranked_static = elide == [true; 3] && proj.plan.offsets_eligible(2);
        let prepared = match inputs {
            OctaInputs::Blocks(inputs) => {
                let ptrs = [
                    inputs[0].0.as_ptr(),
                    inputs[1].0.as_ptr(),
                    inputs[2].0.as_ptr(),
                    inputs[3].0.as_ptr(),
                    inputs[4].0.as_ptr(),
                    inputs[5].0.as_ptr(),
                    inputs[6].0.as_ptr(),
                    inputs[7].0.as_ptr(),
                ];
                let cv_rows = [
                    load_v8(ptrs[0]),
                    load_v8(ptrs[1]),
                    load_v8(ptrs[2]),
                    load_v8(ptrs[3]),
                    load_v8(ptrs[4]),
                    load_v8(ptrs[5]),
                    load_v8(ptrs[6]),
                    load_v8(ptrs[7]),
                ];
                let cv = tr8(
                    cv_rows[0], cv_rows[1], cv_rows[2], cv_rows[3], cv_rows[4], cv_rows[5],
                    cv_rows[6], cv_rows[7],
                );

                let mptrs = [
                    inputs[0].1.as_ptr(),
                    inputs[1].1.as_ptr(),
                    inputs[2].1.as_ptr(),
                    inputs[3].1.as_ptr(),
                    inputs[4].1.as_ptr(),
                    inputs[5].1.as_ptr(),
                    inputs[6].1.as_ptr(),
                    inputs[7].1.as_ptr(),
                ];
                let m_lo = tr8(
                    load_v8(mptrs[0]),
                    load_v8(mptrs[1]),
                    load_v8(mptrs[2]),
                    load_v8(mptrs[3]),
                    load_v8(mptrs[4]),
                    load_v8(mptrs[5]),
                    load_v8(mptrs[6]),
                    load_v8(mptrs[7]),
                );
                let m_hi = tr8(
                    load_v8(mptrs[0].add(8)),
                    load_v8(mptrs[1].add(8)),
                    load_v8(mptrs[2].add(8)),
                    load_v8(mptrs[3].add(8)),
                    load_v8(mptrs[4].add(8)),
                    load_v8(mptrs[5].add(8)),
                    load_v8(mptrs[6].add(8)),
                    load_v8(mptrs[7].add(8)),
                );
                let mut message = [dup_u32(0); 16];
                message[..8].copy_from_slice(&m_lo);
                message[8..].copy_from_slice(&m_hi);

                let mut tlo_a = [0u32; 8];
                let mut thi_a = [0u32; 8];
                let mut bl_a = [0u32; 8];
                let mut fl_a = [0u32; 8];
                for j in 0..8 {
                    tlo_a[j] = inputs[j].2 as u32;
                    thi_a[j] = (inputs[j].2 >> 32) as u32;
                    bl_a[j] = inputs[j].3;
                    fl_a[j] = inputs[j].4;
                }
                PreparedInputs {
                    cv,
                    message,
                    counter_lo: load_v8(tlo_a.as_ptr()),
                    counter_hi: load_v8(thi_a.as_ptr()),
                    block_len: load_v8(bl_a.as_ptr()),
                    flags: load_v8(fl_a.as_ptr()),
                }
            }
            OctaInputs::Closed { init, base } => prepare_closed_inputs(init, base),
        };
        let cv_v = prepared.cv;
        let m = prepared.message;
        let tlo = prepared.counter_lo;
        let thi = prepared.counter_hi;
        let blen = prepared.block_len;
        let flags = prepared.flags;

        let mut state: [V8; 16] = [
            cv_v[0],
            cv_v[1],
            cv_v[2],
            cv_v[3],
            cv_v[4],
            cv_v[5],
            cv_v[6],
            cv_v[7],
            dup_u32(BLAKE3_IV[0]),
            dup_u32(BLAKE3_IV[1]),
            dup_u32(BLAKE3_IV[2]),
            dup_u32(BLAKE3_IV[3]),
            tlo,
            thi,
            blen,
            flags,
        ];

        let zero = dup_u32(0);
        let mut ast = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let mut bs = core::mem::MaybeUninit::<[V8; RING_WORDS]>::uninit();
        let ast = ast.as_mut_ptr().cast::<V8>();
        let bs = bs.as_mut_ptr().cast::<V8>();

        let mut drain = Drain8 {
            ast,
            bs,
            z,
            a,
            b,
            proj,
            elide,
            ranked_static,
        };
        let maxv = dup_u32(u32::MAX);
        let one = dup_u32(1);
        let chain: [V8; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15], tlo, thi, blen, flags,
        ];
        // Words 16..35 are available before the rounds. Retain them in the
        // rolling epochs; the writer publishes each epoch when it completes
        // that epoch's last word.
        for k in 0..20usize {
            let v = if k == 0 {
                or_v8(one, shl_v8::<1>(chain[0]))
            } else {
                or_v8(shr_v8::<31>(chain[k - 1]), shl_v8::<1>(chain[k]))
            };
            let w = 16 + k;
            store_v8(ast.add(w & (RING_WORDS - 1)) as *mut u32, v);
        }
        // Unswitch the static-prefix decision once per octa. Ranked blocks
        // 0/1 source B from RANKED_B_MAX; block 2 still needs words 32..35.
        if !ranked_static {
            for w in 16..RANKED_STATIC_B_PREFIX_END {
                store_v8(bs.add(w & (RING_WORDS - 1)) as *mut u32, maxv);
            }
        }
        for w in RANKED_STATIC_B_PREFIX_END..16 + PROLOGUE_WORDS {
            store_v8(bs.add(w & (RING_WORDS - 1)) as *mut u32, maxv);
        }

        // The round stream starts at word `PROLOGUE_WORDS + 16`, so a first
        // epoch that ends inside the prologue never reaches the writers'
        // epoch boundary — publish it here instead. Constant-folded away for
        // every ring long enough to reach the round stream.
        if RING_WORDS <= PROLOGUE_WORDS + 16 {
            drain.drain_range(16, 16, RING_WORDS - 16);
        }

        let pending_bit = packer_initial_bit(shr_v8::<31>(flags));
        let drain_ptr = &mut drain as *mut Drain8;
        let mut wa = W8::<false>::at(ast, pending_bit, drain_ptr, false);
        // B is pushed after A at every site; it alone triggers a band drain
        // once both rings contain the completed word. Z is derived there.
        let mut wb = W8::<true>::at(bs, packer_initial_bit(one), drain_ptr, ranked_static);

        macro_rules! g {
            ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
             $mx:literal, $my:literal) => {{
                let (t0, l0, r0) = add_carry_parts_v8(state[$la], state[$lb]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, l0);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C0, 31, r0);
                let (a1, l1, r1) = add_carry_parts_v8(t0, m[$mx]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, l1);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C1, 31, r1);
                let d1 = xor_rotr8::<16, 16>(state[$ld], a1);
                let (c1s, l2, r2) = add_carry_parts_v8(state[$lc], d1);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, l2);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C2, 31, r2);
                let b1 = xor_rotr8::<12, 20>(state[$lb], c1s);
                let (t1, l3, r3) = add_carry_parts_v8(a1, b1);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, l3);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C3, 31, r3);
                let (a2, l4, r4) = add_carry_parts_v8(t1, m[$my]);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, l4);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C4, 31, r4);
                let d2 = xor_rotr8::<8, 24>(d1, a2);
                let (c2s, l5, r5) = add_carry_parts_v8(c1s, d2);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, l5);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C5, 31, r5);
                let bn = xor_rotr8::<7, 25>(b1, c2s);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, maxv);
                pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
                pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, maxv);
                state[$la] = a2;
                state[$lb] = bn;
                state[$lc] = c2s;
                state[$ld] = d2;
            }};
        }
        macro_rules! round {
            ($gb:literal, $m0:literal, $m1:literal, $m2:literal, $m3:literal,
             $m4:literal, $m5:literal, $m6:literal, $m7:literal,
             $m8:literal, $m9:literal, $m10:literal, $m11:literal,
             $m12:literal, $m13:literal, $m14:literal, $m15:literal) => {{
                g!($gb, 0, 4, 8, 12, $m0, $m1);
                g!($gb + 1, 1, 5, 9, 13, $m2, $m3);
                g!($gb + 2, 2, 6, 10, 14, $m4, $m5);
                g!($gb + 3, 3, 7, 11, 15, $m6, $m7);
                g!($gb + 4, 0, 5, 10, 15, $m8, $m9);
                g!($gb + 5, 1, 6, 11, 12, $m10, $m11);
                g!($gb + 6, 2, 7, 8, 13, $m12, $m13);
                g!($gb + 7, 3, 4, 9, 14, $m14, $m15);
            }};
        }
        round!(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        round!(8, 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
        round!(16, 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
        round!(24, 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
        round!(32, 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
        round!(40, 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
        round!(48, 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

        const {
            assert!(OUT_HI_BASE % 32 == 17);
        }
        macro_rules! oh {
            ($w:literal) => {{
                let hv = xor_v8(state[$w + 8], cv_v[$w]);
                pushf8!(wa, OUT_HI_BASE + 32 * $w, 32, hv);
                pushf8!(wb, OUT_HI_BASE + 32 * $w, 32, maxv);
            }};
        }
        oh!(0);
        oh!(1);
        oh!(2);
        oh!(3);
        oh!(4);
        oh!(5);
        oh!(6);
        oh!(7);
        wa.finish();
        wb.finish();

        const ZF: usize = USEFUL_BITS.div_ceil(32);
        const {
            assert!(U32_PER_BLOCK - ZF == 30);
        }
        // finish() completed word 481. The ranked blk30 path reads only ring
        // words 480/481 and constructs its zero tail directly; blk31 is a
        // direct zero output. Cold/rollback paths retain the complete tail.
        if !ranked_static {
            for w in ZF..U32_PER_BLOCK {
                let i = w & (RING_WORDS - 1);
                store_v8(ast.add(i) as *mut u32, zero);
                store_v8(bs.add(i) as *mut u32, zero);
            }
        }
        drain.drain_range(U32_PER_BLOCK - RING_WORDS, 0, RING_WORDS);

        // Band 0 is the one intentional deferral: words 0..7 are the input
        // CV, while words 8..15 depend on the final compression state. Build
        // the complete cache line now, then publish it through the exact same
        // elide/NT/window policy as every rolling band.
        for w in 0..8usize {
            let lo = xor_v8(state[w], state[w + 8]);
            store_v8(ast.add(w) as *mut u32, cv_v[w]);
            store_v8(ast.add(8 + w) as *mut u32, lo);
        }
        if !ranked_static {
            for w in 0..8usize {
                store_v8(bs.add(w) as *mut u32, maxv);
                store_v8(bs.add(8 + w) as *mut u32, maxv);
            }
        }
        drain.drain_range(0, 0, 16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn lanes(v: V8) -> [u32; 8] {
        let mut out = [0u32; 8];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), v) };
        out
    }

    /// Compile every W8 offset/width specialization used by the witness and
    /// compare its high-aligned VBMI2 state transition with an independent
    /// scalar bitstream append.  Bit-basis inputs include the deliberately
    /// dirty bit 31 of a width-31 field, so the oracle also proves that the
    /// next field, rather than an eager mask, discards that irrelevant bit.
    #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn check_vbmi2_push<const USED: i32, const WIDTH: i32, const BACK: i32>() {
        const {
            assert!(USED >= 0 && USED < 32);
            assert!(WIDTH == 31 || WIDTH == 32);
            assert!(BACK > 0 && BACK < 32);
        }
        let pending_mask = if USED == 0 {
            0
        } else {
            ((1u64 << USED) - 1) as u32
        };
        let field_mask = if WIDTH == 32 {
            u32::MAX
        } else {
            (1u32 << WIDTH) - 1
        };
        let mut pending_cases = vec![0, pending_mask];
        for bit in 0..USED {
            pending_cases.push(1u32 << bit);
        }
        pending_cases.sort_unstable();
        pending_cases.dedup();
        let mut field_cases = vec![0, field_mask, u32::MAX];
        for bit in 0..32 {
            field_cases.push(1u32 << bit);
        }
        field_cases.sort_unstable();
        field_cases.dedup();

        for &pending_low in &pending_cases {
            for &raw_v in &field_cases {
                let v = raw_v & field_mask;
                let acc = u64::from(pending_low) | (u64::from(v) << USED);
                let total = USED + WIDTH;
                let expected_out = (total >= 32).then_some(acc as u32);
                let next_used = total % 32;
                let expected_pending = if total >= 32 {
                    (acc >> 32) as u32
                } else {
                    acc as u32
                };
                let pending_hi = if USED == 0 {
                    0
                } else {
                    pending_low << (32 - USED)
                };

                let sentinel = 0xA5A5_5A5A;
                let mut stage = [dup_u32(sentinel); 1];
                let mut writer = W8::<false>::at(
                    stage.as_mut_ptr(),
                    dup_u32(pending_hi),
                    core::ptr::null_mut::<Drain8<'static>>(),
                    false,
                );
                unsafe {
                    writer.push::<USED, WIDTH, BACK, 0>(dup_u32(raw_v));
                }

                let got_out = lanes(stage[0])[0];
                assert_eq!(
                    got_out,
                    expected_out.unwrap_or(sentinel),
                    "out: used={USED} width={WIDTH} pending={pending_low:#x} v={raw_v:#x}"
                );
                let got_hi = lanes(writer.pending)[0];
                if next_used != 0 {
                    assert_eq!(
                        got_hi >> (32 - next_used),
                        expected_pending,
                        "pending: used={USED} width={WIDTH} pending={pending_low:#x} v={raw_v:#x}"
                    );
                }
            }
        }
    }

    #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn check_vbmi2_finish<const USED: i32, const BACK: i32>() {
        const {
            assert!(USED > 0 && USED < 32);
            assert!(BACK == 32 - USED);
        }
        let mask = ((1u64 << USED) - 1) as u32;
        let mut cases = vec![0, mask];
        for bit in 0..USED {
            cases.push(1u32 << bit);
        }
        for pending_low in cases {
            let pending_hi = dup_u32(pending_low << BACK);
            let got = lanes(packer_high_to_low::<BACK>(pending_hi));
            assert_eq!(got, [pending_low; 8], "finish: used={USED}");
        }
    }

    #[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
    #[test]
    fn vbmi2_high_aligned_w8_matches_scalar_all_offsets() {
        unsafe {
            macro_rules! check_offset {
                ($used:literal, $back:literal) => {{
                    check_vbmi2_push::<$used, 31, $back>();
                    check_vbmi2_push::<$used, 32, $back>();
                    check_vbmi2_finish::<$used, $back>();
                }};
            }
            check_vbmi2_push::<0, 31, 1>();
            check_vbmi2_push::<0, 32, 1>();
            check_offset!(1, 31);
            check_offset!(2, 30);
            check_offset!(3, 29);
            check_offset!(4, 28);
            check_offset!(5, 27);
            check_offset!(6, 26);
            check_offset!(7, 25);
            check_offset!(8, 24);
            check_offset!(9, 23);
            check_offset!(10, 22);
            check_offset!(11, 21);
            check_offset!(12, 20);
            check_offset!(13, 19);
            check_offset!(14, 18);
            check_offset!(15, 17);
            check_offset!(16, 16);
            check_offset!(17, 15);
            check_offset!(18, 14);
            check_offset!(19, 13);
            check_offset!(20, 12);
            check_offset!(21, 11);
            check_offset!(22, 10);
            check_offset!(23, 9);
            check_offset!(24, 8);
            check_offset!(25, 7);
            check_offset!(26, 6);
            check_offset!(27, 5);
            check_offset!(28, 4);
            check_offset!(29, 3);
            check_offset!(30, 2);
            check_offset!(31, 1);

            let flags = [0, 1, 0x7FFF_FFFF, 0x8000_0000, u32::MAX, 11, 64, 9];
            let flags_v = load_v8(flags.as_ptr());
            assert_eq!(
                lanes(packer_initial_bit(shr_v8::<31>(flags_v))),
                flags.map(|v| v & 0x8000_0000),
                "A prior flag must occupy bit 31"
            );
            assert_eq!(
                lanes(packer_initial_bit(dup_u32(1))),
                [0x8000_0000; 8],
                "B prior flag must occupy bit 31"
            );

            // Exercise the production finish itself at the ranked used=17.
            let pending_low = 0x1_5A5A;
            let mut stage = [dup_u32(0); RING_WORDS];
            let mut writer = W8::<false>::at(
                stage.as_mut_ptr(),
                dup_u32(pending_low << 15),
                core::ptr::null_mut::<Drain8<'static>>(),
                false,
            );
            writer.finish();
            assert_eq!(lanes(stage[LAST_WORD & (RING_WORDS - 1)]), [pending_low; 8]);
        }
    }

    #[test]
    fn closed_octa_inputs_match_scalar_generator() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe {
            for (init, base) in [
                (0u64, 0usize),
                (0x0123_4567_89AB_CDEF, 17),
                (u64::MAX - 31, (1 << 18) - 8),
            ] {
                let got = prepare_closed_inputs(init, base);
                let cv: [[u32; 8]; 8] = got.cv.map(|v| lanes(v));
                let message: [[u32; 8]; 16] = got.message.map(|v| lanes(v));
                let counter_lo = lanes(got.counter_lo);
                let counter_hi = lanes(got.counter_hi);
                let block_len = lanes(got.block_len);
                let flags = lanes(got.flags);
                for lane in 0..8 {
                    let expected = crate::seed_pipe::gen_block(init, base + lane);
                    for word in 0..8 {
                        assert_eq!(cv[word][lane], expected.0[word]);
                    }
                    for word in 0..16 {
                        assert_eq!(message[word][lane], expected.1[word]);
                    }
                    assert_eq!(counter_lo[lane], expected.2 as u32);
                    assert_eq!(counter_hi[lane], (expected.2 >> 32) as u32);
                    assert_eq!(block_len[lane], expected.3);
                    assert_eq!(flags[lane], expected.4);
                }
            }
        }
    }

    /// Independent algebra/codegen oracle for the production Z derivation.
    /// The direct path packs `a & b` and then transposes; the new path
    /// transposes A/B separately and ANDs corresponding row vectors.
    #[test]
    fn derived_z_after_tr8_matches_direct_z() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe {
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            for case in 0..66usize {
                let mut a = [[0u32; 8]; 8];
                let mut b = [[0u32; 8]; 8];
                for i in 0..8 {
                    for j in 0..8 {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        a[i][j] = if case == 0 { u32::MAX } else { state as u32 };
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        b[i][j] = if case == 1 { u32::MAX } else { state as u32 };
                    }
                }
                let av: [V8; 8] = core::array::from_fn(|i| load_v8(a[i].as_ptr()));
                let bv: [V8; 8] = core::array::from_fn(|i| load_v8(b[i].as_ptr()));
                let zv: [V8; 8] = core::array::from_fn(|i| and_v8(av[i], bv[i]));
                let direct = tr8(zv[0], zv[1], zv[2], zv[3], zv[4], zv[5], zv[6], zv[7]);
                let at = tr8(av[0], av[1], av[2], av[3], av[4], av[5], av[6], av[7]);
                let bt = tr8(bv[0], bv[1], bv[2], bv[3], bv[4], bv[5], bv[6], bv[7]);
                for row in 0..8 {
                    let derived = and_v8(at[row], bt[row]);
                    let mut direct_words = [0u32; 8];
                    let mut derived_words = [0u32; 8];
                    store_v8(direct_words.as_mut_ptr(), direct[row]);
                    store_v8(derived_words.as_mut_ptr(), derived);
                    assert_eq!(derived_words, direct_words, "case={case} row={row}");
                }
            }
        }
    }

    #[test]
    fn witgen8_tr8_is_8x8_u32_transpose() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        unsafe { tr8_check() }
    }

    /// Independent scalar-layout oracle for the ranked 16×8 ZMM network.
    #[cfg(target_feature = "avx512f")]
    #[test]
    fn witgen8_tr8x16_zmm_is_word_to_block_transpose() {
        #[repr(C, align(64))]
        struct Input([[u32; 8]; STEP_WORDS]);

        let input = Input(core::array::from_fn(|word| {
            core::array::from_fn(|block| (1000 * word + block) as u32)
        }));
        unsafe {
            let out = tr8x16_zmm(input.0.as_ptr().cast::<V8>(), 0);
            for (block, row) in out.into_iter().enumerate() {
                let mut got = [0u32; STEP_WORDS];
                _mm512_storeu_si512(got.as_mut_ptr().cast::<__m512i>(), row);
                let expected = core::array::from_fn(|word| (1000 * word + block) as u32);
                assert_eq!(got, expected, "block {block}");
            }
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[test]
    fn ranked_direct_dense_rows_and_publish_match_staged_path() {
        #[repr(C, align(64))]
        struct Input([[u32; 8]; STEP_WORDS]);
        #[repr(C, align(64))]
        struct Stage {
            a: [u32; 8 * STEP_WORDS],
            b: [u32; 8 * STEP_WORDS],
        }
        #[repr(C, align(64))]
        struct Offsets([u16; 8 * ROUND1_AB_OFF_WORDS]);
        #[repr(C, align(64))]
        struct RankedBuf([u32; 8 * U32_PER_BLOCK]);

        unsafe {
            for case in 0..6usize {
                let a_input = Input(core::array::from_fn(|word| {
                    core::array::from_fn(|block| match case {
                        0 => u32::MAX,
                        1 => 0,
                        _ => {
                            0xA5A5_0000u32
                                ^ ((case as u32) << 20)
                                ^ ((word as u32) << 9)
                                ^ ((block as u32).wrapping_mul(0x0101_0101))
                        }
                    })
                }));
                let b_input = Input(core::array::from_fn(|word| {
                    core::array::from_fn(|block| match case {
                        0 => ((word as u32) << 16) ^ ((block as u32) * 0x1111_1111),
                        1 => u32::MAX,
                        _ => {
                            0x5A5A_0000u32
                                ^ ((case as u32) << 17)
                                ^ ((word as u32).wrapping_mul(0x1021))
                                ^ ((block as u32).wrapping_mul(0x0102_0408))
                        }
                    })
                }));

                let a_rows = tr8x16_zmm(a_input.0.as_ptr().cast::<V8>(), 0);
                let b_rows = tr8x16_zmm(b_input.0.as_ptr().cast::<V8>(), 0);
                let mut inline_off = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                widen_ranked_dense_rows(&a_rows, inline_off.0.as_mut_ptr());
                widen_ranked_dense_rows(&b_rows, inline_off.0.as_mut_ptr().add(64));

                let mut rollback_off = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                let rollback_a_rows = ranked_dense_rows_and_offsets_rollback(
                    a_input.0.as_ptr().cast::<V8>(),
                    0,
                    rollback_off.0.as_mut_ptr(),
                );
                let rollback_b_rows = ranked_dense_rows_and_offsets_rollback(
                    b_input.0.as_ptr().cast::<V8>(),
                    0,
                    rollback_off.0.as_mut_ptr().add(64),
                );

                let mut staged = Stage {
                    a: [0u32; 8 * STEP_WORDS],
                    b: [0u32; 8 * STEP_WORDS],
                };
                let mut staged_off = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                stage_ranked_dense_side(
                    a_input.0.as_ptr().cast::<V8>(),
                    0,
                    staged.a.as_mut_ptr(),
                    staged_off.0.as_mut_ptr(),
                );
                stage_ranked_dense_side(
                    b_input.0.as_ptr().cast::<V8>(),
                    0,
                    staged.b.as_mut_ptr(),
                    staged_off.0.as_mut_ptr().add(64),
                );

                for row in 0..8 {
                    let mut got_a = [0u32; STEP_WORDS];
                    let mut got_b = [0u32; STEP_WORDS];
                    _mm512_storeu_si512(got_a.as_mut_ptr().cast::<__m512i>(), a_rows[row]);
                    _mm512_storeu_si512(got_b.as_mut_ptr().cast::<__m512i>(), b_rows[row]);
                    let mut rollback_a = [0u32; STEP_WORDS];
                    let mut rollback_b = [0u32; STEP_WORDS];
                    _mm512_storeu_si512(rollback_a.as_mut_ptr().cast::<__m512i>(), rollback_a_rows[row]);
                    _mm512_storeu_si512(rollback_b.as_mut_ptr().cast::<__m512i>(), rollback_b_rows[row]);
                    assert_eq!(
                        &staged.a[row * STEP_WORDS..(row + 1) * STEP_WORDS],
                        &got_a,
                        "staged a row mismatch, case={case} row={row}"
                    );
                    assert_eq!(
                        &staged.b[row * STEP_WORDS..(row + 1) * STEP_WORDS],
                        &got_b,
                        "staged b row mismatch, case={case} row={row}"
                    );
                    assert_eq!(rollback_a, got_a, "rollback a row mismatch, case={case} row={row}");
                    assert_eq!(rollback_b, got_b, "rollback b row mismatch, case={case} row={row}");
                }
                assert_eq!(inline_off.0, staged_off.0, "inline offset mismatch, case={case}");
                assert_eq!(rollback_off.0, staged_off.0, "rollback offset mismatch, case={case}");

                let mut staged_z = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut staged_a = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut staged_b = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut direct_z = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut direct_a = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut direct_b = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let staged_rows =
                    RankedRows::new(staged_z.0.as_mut_ptr(), staged_a.0.as_mut_ptr(), staged_b.0.as_mut_ptr());
                let direct_rows =
                    RankedRows::new(direct_z.0.as_mut_ptr(), direct_a.0.as_mut_ptr(), direct_b.0.as_mut_ptr());
                for j in 0..8 {
                    staged_rows.publish_dense(j, staged.a.as_ptr(), staged.b.as_ptr());
                    direct_rows.publish_dense_values(j, a_rows[j], b_rows[j]);
                }
                assert_eq!(staged_z.0, direct_z.0, "z mismatch, case={case}");
                assert_eq!(staged_a.0, direct_a.0, "a mismatch, case={case}");
                assert_eq!(staged_b.0, direct_b.0, "b mismatch, case={case}");
            }
        }
    }

    unsafe fn tr8_check() {
        unsafe {
            let rows: [V8; 8] = core::array::from_fn(|i| {
                _mm256_setr_epi32(
                    (1000 * i) as i32,
                    (1000 * i + 1) as i32,
                    (1000 * i + 2) as i32,
                    (1000 * i + 3) as i32,
                    (1000 * i + 4) as i32,
                    (1000 * i + 5) as i32,
                    (1000 * i + 6) as i32,
                    (1000 * i + 7) as i32,
                )
            });
            let t = tr8(
                rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7],
            );
            for j in 0..8 {
                let mut buf = [0i32; 8];
                _mm256_storeu_si256(buf.as_mut_ptr().cast(), t[j]);
                for i in 0..8 {
                    assert_eq!(buf[i], (1000 * i + j) as i32, "t[{j}] lane {i}");
                }
            }
            let back = tr8(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
            for i in 0..8 {
                let mut a = [0i32; 8];
                let mut b = [0i32; 8];
                _mm256_storeu_si256(a.as_mut_ptr().cast(), rows[i]);
                _mm256_storeu_si256(b.as_mut_ptr().cast(), back[i]);
                assert_eq!(a, b, "tr8² row {i}");
            }
        }
    }
}
// witfire-7 draw marker 10514
// witfire-16 draw marker 2194
// fire51: fresh draw on new bar 3852475 (rival NTT promotion)

// vbmi2draw-1: independent official timing sample of promoted fbc3001 VBMI2 W8 packer; no executable change.
// rivaldraw-28 marker 76708 on 2758bc1
// rivaldraw-37 marker 33507 on ca47b5d

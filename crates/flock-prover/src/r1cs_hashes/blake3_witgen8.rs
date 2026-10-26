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
    round1_ab_inner_window_from_offsets_nt2_residual, round1_ab_inner_window_with_images,
    round1_ab_inner_window30_k0, round1_ab_table_images,
};

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
#[path = "blake3_witgen8_root.rs"]
pub(crate) mod root_emit;

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

const STEP_WORDS: usize = 16;

pub(crate) const STREAM_STAGE_WORDS: usize = 2 * 8 * STEP_WORDS;

const ELIDE_ZERO_CHUNK: usize = 62;

const ELIDE_B_TAIL_CHUNK_WIN: usize = 60;
const ELIDE_B_PREFIX_CHUNKS: usize = 4;
const LAST_WORD: usize = (USEFUL_BITS - 1) / 32;
const _ELIDE_GEOMETRY: () = {
    assert!(8 * ELIDE_ZERO_CHUNK >= USEFUL_BITS.div_ceil(32));
    assert!(8 * ELIDE_ZERO_CHUNK < U32_PER_BLOCK);
    assert!(8 * ELIDE_B_PREFIX_CHUNKS <= 36);
    assert!(LAST_WORD == 481);

    assert!(DUMP_CHUNKS.is_multiple_of(2));
    assert!(ELIDE_B_PREFIX_CHUNKS.is_multiple_of(2));

    assert!(ELIDE_B_TAIL_CHUNK_WIN.is_multiple_of(2));
    assert!(ELIDE_B_TAIL_CHUNK_WIN < DUMP_CHUNKS);
};

type V8 = __m256i;

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

#[inline(always)]
pub(crate) fn ey_dead_w31_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_EY_DEAD_W31").is_none());
    *ON
}

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
        let golden = _mm256_set1_epi64x(crate::s1::GOLDEN as i64);
        states[0] = _mm256_add_epi64(states[0], golden);
        states[1] = _mm256_add_epi64(states[1], golden);
        pack_low_u32x8(mix_u64x4(states[0]), mix_u64x4(states[1]))
    }
}

#[cfg(not(target_feature = "avx512dq"))]
#[inline(always)]
unsafe fn prepare_closed_inputs(init: u64, base: usize) -> PreparedInputs {
    unsafe {
        let stride = crate::s1::GOLDEN.wrapping_mul(crate::s1::DRAWS_PER_BLOCK as u64);
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
        *state = _mm512_add_epi64(*state, _mm512_set1_epi64(crate::s1::GOLDEN as i64));
        _mm512_cvtepi64_epi32(mix_u64x8(*state))
    }
}

#[cfg(target_feature = "avx512dq")]
#[inline(always)]
unsafe fn prepare_closed_inputs(init: u64, base: usize) -> PreparedInputs {
    unsafe {
        let stride = crate::s1::GOLDEN.wrapping_mul(crate::s1::DRAWS_PER_BLOCK as u64);
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

#[cfg(target_feature = "avx512f")]
#[inline(always)]
unsafe fn store_ranked_stage_line(p: *mut u32, v: __m512i) {
    unsafe { _mm512_store_si512(p.cast::<__m512i>(), v) }
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn widen_off_half(lo: V8, hi: V8, op: *mut u16) {
    unsafe {
        let w = |v: V8| _mm512_slli_epi16::<6>(_mm512_cvtepu8_epi16(v));
        _mm512_store_si512(op.cast::<__m512i>(), w(lo));
        _mm512_store_si512(op.add(32).cast::<__m512i>(), w(hi));
    }
}

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
pub(crate) fn widen_maddubs_enabled() -> bool {
    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    {
        static ON: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_WIDEN_MADDUBS").is_none());
        *ON
    }
    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
    {
        false
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[derive(Clone, Copy)]
struct WidenConsts {
    even: __m512i,
    odd: __m512i,
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
impl WidenConsts {
    #[inline(always)]
    fn new() -> Self {
        unsafe {
            Self {
                even: _mm512_set1_epi16(0x0040),
                odd: _mm512_set1_epi16(0x4000),
            }
        }
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn widen_off_line_parity(v: __m512i, op: *mut u16, c: WidenConsts) {
    unsafe {
        _mm512_store_si512(op.cast::<__m512i>(), _mm512_maddubs_epi16(v, c.even));
        _mm512_store_si512(op.add(32).cast::<__m512i>(), _mm512_maddubs_epi16(v, c.odd));
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

#[allow(dead_code)]
#[inline(always)]
fn xor3_v8(a: V8, b: V8, c: V8) -> V8 {
    #[cfg(target_feature = "avx512vl")]
    unsafe {
        _mm256_ternarylogic_epi32::<0x96>(a, b, c)
    }

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

#[inline(always)]
fn vsli_v8<const N: i32>(a: V8, b: V8) -> V8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
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

#[cfg(target_feature = "avx512f")]
#[inline(always)]
unsafe fn tr8x16_zmm(stage: *const V8, w: usize) -> [__m512i; 8] {
    unsafe {
        debug_assert!(w + STEP_WORDS <= RING_WORDS);
        debug_assert!(w.is_multiple_of(STEP_WORDS));

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

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn ranked_dense_rows_and_offsets_rollback<const P: bool>(
    ring: *const V8,
    w: usize,
    op: *mut u16,
    c: WidenConsts,
) -> [__m512i; 8] {
    unsafe {
        let rows = tr8x16_zmm(ring, w);
        widen_ranked_dense_rows::<P>(&rows, op, c);
        rows
    }
}

#[inline(always)]
pub(crate) fn ranked_direct_dense_publish_enabled() -> bool {
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

#[inline(always)]
pub(crate) fn ranked_direct_dense_inline_enabled() -> bool {
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

#[inline(always)]
fn ranked_b_constwin_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_R1_B_CONSTWIN").as_deref() != Some(std::ffi::OsStr::new("1"))
    });
    *ON
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn widen_ranked_dense_rows<const P: bool>(
    rows: &[__m512i; 8],
    op: *mut u16,
    c: WidenConsts,
) {
    unsafe {
        let mut r = 0usize;
        while r != 8 {
            if P {
                widen_off_line_parity(rows[r], op.add(r * ROUND1_AB_OFF_WORDS), c);
            } else {
                widen_off_line(rows[r], op.add(r * ROUND1_AB_OFF_WORDS));
            }
            r += 1;
        }
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn stage_ranked_dense_side<const P: bool>(
    ring: *const V8,
    w: usize,
    stage: *mut u32,
    op: *mut u16,
    c: WidenConsts,
) {
    unsafe {
        let rows = tr8x16_zmm(ring, w);
        widen_ranked_dense_rows::<P>(&rows, op, c);
        let mut r = 0usize;
        while r != 8 {
            store_ranked_stage_line(stage.add(r * STEP_WORDS), rows[r]);
            r += 1;
        }
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn project_blocks_ranked_hot_offsets_direct_inline_fast<const P: bool>(
    proj: &StreamProj<'_>,
    blk: usize,
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
    rows: RankedRows,
    a_ring: *const V8,
    b_ring: *const V8,
    rw: usize,
    off: *mut u16,
    c: WidenConsts,
) {
    unsafe {
        debug_assert!(blk > 1 && blk < 30);
        debug_assert!(proj.one_rows_elided);
        let a_rows = tr8x16_zmm(a_ring, rw);
        widen_ranked_dense_rows::<P>(&a_rows, off, c);
        let b_rows = tr8x16_zmm(b_ring, rw);
        widen_ranked_dense_rows::<P>(&b_rows, off.add(64), c);

        if blk == 2 {
            let mut j = 0usize;
            while j != 8 {
                rows.publish_dense_values(j, a_rows[j], b_rows[j]);
                let out = &mut *proj
                    .out
                    .add(j * proj.out_stride + 2 * 64 - proj.out_bias)
                    .cast::<[u8; 64]>();
                round1_ab_inner_window_from_offsets_nt2_residual::<P>(
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
        if blk == 29 {
            let mut j = 0usize;
            while j != 8 {
                rows.publish_dense_values(j, a_rows[j], b_rows[j]);
                let out = &mut *proj
                    .out
                    .add(j * proj.out_stride + 29 * 64 - proj.out_bias)
                    .cast::<[u8; 64]>();
                round1_ab_inner_window_from_offsets_nt2_residual::<P>(
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

        if plan.bcomplement_static_eligible() {
            match blk {
                3 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<3, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                4 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<4, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                5 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<5, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                6 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<6, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                7 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<7, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                8 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<8, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                9 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<9, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                10 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        10,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                11 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        11,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                12 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        12,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                13 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        13,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                14 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        14,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                15 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        15,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                16 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        16,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                17 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        17,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                18 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        18,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                19 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        19,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                20 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        20,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                21 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        21,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                22 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        22,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                23 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        23,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                24 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        24,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                25 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        25,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                26 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        26,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                27 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        27,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                28 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        28,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
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
            round1_ab_inner_window_from_offsets_nt2::<P>(
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

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn project_blocks_ranked_hot_offsets_direct_inline<const P: bool>(
    proj: &StreamProj<'_>,
    blk: usize,
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
    rows: RankedRows,
    a_ring: *const V8,
    b_ring: *const V8,
    rw: usize,
    off: *mut u16,
    c: WidenConsts,
) {
    unsafe {
        debug_assert!(blk > 1 && blk < 30);
        let a_rows = tr8x16_zmm(a_ring, rw);
        widen_ranked_dense_rows::<P>(&a_rows, off, c);
        let b_rows = tr8x16_zmm(b_ring, rw);
        widen_ranked_dense_rows::<P>(&b_rows, off.add(64), c);

        if proj.one_rows_elided && blk == 2 {
            let mut j = 0usize;
            while j != 8 {
                rows.publish_dense_values(j, a_rows[j], b_rows[j]);
                let out = &mut *proj
                    .out
                    .add(j * proj.out_stride + 2 * 64 - proj.out_bias)
                    .cast::<[u8; 64]>();
                round1_ab_inner_window_from_offsets_nt2_residual::<P>(
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
                round1_ab_inner_window_from_offsets_nt2_residual::<P>(
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
                3 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<3, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                4 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<4, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                5 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<5, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                6 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<6, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                7 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<7, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                8 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<8, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                9 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<9, P>(
                        proj, plan, imgs, rows, &a_rows, &b_rows, off,
                    );
                }
                10 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        10,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                11 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        11,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                12 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        12,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                13 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        13,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                14 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        14,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                15 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        15,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                16 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        16,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                17 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        17,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                18 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        18,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                19 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        19,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                20 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        20,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                21 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        21,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                22 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        22,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                23 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        23,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                24 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        24,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                25 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        25,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                26 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        26,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                27 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        27,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
                28 => {
                    return project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const::<
                        28,
                        P,
                    >(proj, plan, imgs, rows, &a_rows, &b_rows, off);
                }
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
                round1_ab_inner_window_from_offsets_nt2_bcomplement_static::<P>(
                    &*off
                        .add(j * ROUND1_AB_OFF_WORDS)
                        .cast::<[u16; ROUND1_AB_OFF_WORDS]>(),
                    out,
                    plan,
                    imgs,
                    blk,
                );
            } else {
                round1_ab_inner_window_from_offsets_nt2::<P>(
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

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn project_blocks_ranked_hot_offsets_direct_inline_bcomplement_const<
    const BLK: usize,
    const P: bool,
>(
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
            round1_ab_inner_window_from_offsets_nt2_bcomplement_static_const::<BLK, P>(
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

const PROLOGUE_WORDS: usize = 20;

const RANKED_STATIC_B_PREFIX_END: usize = 2 * STEP_WORDS;
const RANKED_STATIC_B_TAIL_START: usize = 30 * STEP_WORDS;
const _RING_GEOMETRY: () = {
    assert!(RING_WORDS >= 32);
    assert!(RING_WORDS <= U32_PER_BLOCK);
    assert!(RING_WORDS & (RING_WORDS - 1) == 0);

    assert!(RING_WORDS.is_multiple_of(STEP_WORDS));
    assert!(STEP_WORDS == 16);

    assert!(STEP_WORDS * core::mem::size_of::<u32>() == 64);
    assert!((U32_PER_BLOCK * core::mem::size_of::<u32>()).is_multiple_of(64));

    assert!(8 * STEP_WORDS >= 32);
    assert!(RANKED_STATIC_B_PREFIX_END == 32);
    assert!(RANKED_STATIC_B_TAIL_START == 480);
    assert!(LAST_WORD == RANKED_STATIC_B_TAIL_START + 1);
};

pub(crate) struct StreamProj<'t> {
    pub(crate) stage: *mut u32,

    pub(crate) out: *mut u8,
    pub(crate) out_stride: usize,
    pub(crate) out_bias: usize,
    pub(crate) inv_table: &'t InvNttTableByteSingleGf8,
    pub(crate) plan: Round1AbWindowPlan,

    pub(crate) one_rows_elided: bool,

    pub(crate) w30_elided: bool,
}

#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[path = "blake3_witgen8_xor.rs"]
mod xor_projection;

pub(crate) struct RankedClosedWindowPreps {
    windows: [(Round1AbWindowPlan, Round1AbTableImages); 31],
    #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    tables: xor_projection::Tables,
}
impl core::ops::Deref for RankedClosedWindowPreps {
    type Target = [(Round1AbWindowPlan, Round1AbTableImages); 31];
    fn deref(&self) -> &Self::Target {
        &self.windows
    }
}

pub(crate) fn prepare_ranked_closed_window_preps(
    inv_table: &InvNttTableByteSingleGf8,
    plan: Round1AbWindowPlan,
) -> RankedClosedWindowPreps {
    RankedClosedWindowPreps {
        windows: std::array::from_fn(|blk| {
            let plan = plan.for_window(blk);
            (plan, round1_ab_table_images(inv_table, plan))
        }),
        #[cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi",
            target_feature = "gfni"
        ))]
        tables: xor_projection::Tables::new(inv_table),
    }
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
        (self.stage, unsafe { self.stage.add(8 * STEP_WORDS) })
    }

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
                    round1_ab_inner_window_from_offsets::<false>(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs);
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





    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets<const P: bool>(&self, blk: usize, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, off: *const u16) {
        unsafe {
            debug_assert!(blk > 1 && blk < 30);
            if self.one_rows_elided && blk == 2 {
                self.project_blocks_ranked_hot_offsets_residual::<2,0xfc,P>(plan,imgs,rows,off);
                return;
            }
            if self.one_rows_elided && blk == 29 {
                self.project_blocks_ranked_hot_offsets_residual::<29,0x0f,P>(plan,imgs,rows,off);
                return;
            }





            if plan.bcomplement_static_eligible() {
                self.project_blocks_ranked_hot_offsets_bcomplement::<P>(blk,plan,imgs,rows,off);
                return;
            }
            let (sa,sb)=self.sides();
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense(j,sa,sb);
                let out=&mut *self.out.add(j*self.out_stride+blk*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2::<P>(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs);
                j+=1;
            }
        }
    }




    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets_bcomplement<const P: bool>(&self, blk: usize, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, off: *const u16) {
        unsafe {
            debug_assert!(blk > 1 && blk < 30);
            debug_assert!(plan.bcomplement_static_eligible());
            if ranked_b_constwin_enabled() {
                match blk {
                    3 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<3,P>(plan,imgs,rows,off),
                    4 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<4,P>(plan,imgs,rows,off),
                    5 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<5,P>(plan,imgs,rows,off),
                    6 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<6,P>(plan,imgs,rows,off),
                    7 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<7,P>(plan,imgs,rows,off),
                    8 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<8,P>(plan,imgs,rows,off),
                    9 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<9,P>(plan,imgs,rows,off),
                    10 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<10,P>(plan,imgs,rows,off),
                    11 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<11,P>(plan,imgs,rows,off),
                    12 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<12,P>(plan,imgs,rows,off),
                    13 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<13,P>(plan,imgs,rows,off),
                    14 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<14,P>(plan,imgs,rows,off),
                    15 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<15,P>(plan,imgs,rows,off),
                    16 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<16,P>(plan,imgs,rows,off),
                    17 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<17,P>(plan,imgs,rows,off),
                    18 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<18,P>(plan,imgs,rows,off),
                    19 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<19,P>(plan,imgs,rows,off),
                    20 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<20,P>(plan,imgs,rows,off),
                    21 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<21,P>(plan,imgs,rows,off),
                    22 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<22,P>(plan,imgs,rows,off),
                    23 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<23,P>(plan,imgs,rows,off),
                    24 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<24,P>(plan,imgs,rows,off),
                    25 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<25,P>(plan,imgs,rows,off),
                    26 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<26,P>(plan,imgs,rows,off),
                    27 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<27,P>(plan,imgs,rows,off),
                    28 => return self.project_blocks_ranked_hot_offsets_bcomplement_const::<28,P>(plan,imgs,rows,off),
                    _ => {}
                }
            }
            let (sa,sb)=self.sides();
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense(j,sa,sb);
                let out=&mut *self.out.add(j*self.out_stride+blk*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2_bcomplement_static::<P>(
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
    unsafe fn project_blocks_ranked_hot_offsets_bcomplement_const<const BLK: usize, const P: bool>(&self, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, off: *const u16) {
        unsafe {
            debug_assert!((3..=28).contains(&BLK));
            debug_assert!(plan.bcomplement_static_eligible());
            let (sa,sb)=self.sides();
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense(j,sa,sb);
                let out=&mut *self.out.add(j*self.out_stride+BLK*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2_bcomplement_static_const::<BLK, P>(
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
    unsafe fn project_blocks_ranked_hot_offsets_direct_rollback<const P: bool>(&self, blk: usize, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, a_rows: &[__m512i; 8], b_rows: &[__m512i; 8], off: *const u16) {
        unsafe {
            debug_assert!(blk > 1 && blk < 30);
            if self.one_rows_elided && blk == 2 {
                self.project_blocks_ranked_hot_offsets_residual_direct_rollback::<2,0xfc,P>(plan,imgs,rows,a_rows,b_rows,off);
                return;
            }
            if self.one_rows_elided && blk == 29 {
                self.project_blocks_ranked_hot_offsets_residual_direct_rollback::<29,0x0f,P>(plan,imgs,rows,a_rows,b_rows,off);
                return;
            }
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense_values(j,a_rows[j],b_rows[j]);
                let out=&mut *self.out.add(j*self.out_stride+blk*64-self.out_bias).cast::<[u8;64]>();
                if plan.bcomplement_static_eligible() {
                    round1_ab_inner_window_from_offsets_nt2_bcomplement_static::<P>(
                        &*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),
                        out,
                        plan,
                        imgs,
                        blk,
                    );
                } else {
                    round1_ab_inner_window_from_offsets_nt2::<P>(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs);
                }
                j+=1;
            }
        }
    }

    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets_residual<const BLK:usize,const KEEP:u8,const P:bool>(&self, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, off: *const u16) {
        unsafe {
            const { assert!((BLK==2 && KEEP==0xfc)||(BLK==29 && KEEP==0x0f)); }
            let (sa,sb)=self.sides();
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense(j,sa,sb);
                let out=&mut *self.out.add(j*self.out_stride+BLK*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2_residual::<P>(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs,KEEP);
                j+=1;
            }
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn project_blocks_ranked_hot_offsets_residual_direct_rollback<const BLK:usize,const KEEP:u8,const P:bool>(&self, plan: Round1AbWindowPlan, imgs: Round1AbTableImages, rows: RankedRows, a_rows: &[__m512i; 8], b_rows: &[__m512i; 8], off: *const u16) {
        unsafe {
            const { assert!((BLK==2 && KEEP==0xfc)||(BLK==29 && KEEP==0x0f)); }
            let mut j=0usize;
            while j!=8 {
                rows.publish_dense_values(j,a_rows[j],b_rows[j]);
                let out=&mut *self.out.add(j*self.out_stride+BLK*64-self.out_bias).cast::<[u8;64]>();
                round1_ab_inner_window_from_offsets_nt2_residual::<P>(&*off.add(j*ROUND1_AB_OFF_WORDS).cast::<[u16;ROUND1_AB_OFF_WORDS]>(),out,plan,imgs,KEEP);
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






            let q=self.stage.cast::<u64>();
            let mut j=0usize;
            while j!=8 {
                rows.publish_sparse_30(j,q.add(j));
                if !self.w30_elided {
                    let a=&*q.add(j).cast::<[u8;64]>();
                    let out=&mut *self.out.add(j*self.out_stride+30*64-self.out_bias).cast::<[u8;64]>();
                    round1_ab_inner_window30_k0(a,&RANKED_B30.0,out,self.inv_table,plan,imgs,30);
                }
                j+=1;
            }
        }
    }
}

struct Drain8<'t> {
    ast: *mut V8,
    bs: *mut V8,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    proj: StreamProj<'t>,
    elide: [bool; 3],
    ranked_static: bool,
    #[allow(dead_code)]
    ranked_fast: bool,
    #[allow(dead_code)]
    ranked_preps: Option<&'t RankedClosedWindowPreps>,
    #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    roots: Option<root_emit::RootOctaWriter<'t>>,
}

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

#[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
#[inline(always)]
fn packer_high_to_low<const BACK: i32>(pending: V8) -> V8 {
    const {
        assert!(BACK > 0 && BACK < 32);
    }
    shr_v8::<BACK>(pending)
}

struct W8<'t, const FLUSH: bool, const RANKED_CLOSED: bool = false> {
    pending: V8,
    stage: *mut V8,
    drain: *mut Drain8<'t>,
    ranked_static: bool,
}

impl<'t, const FLUSH: bool, const RANKED_CLOSED: bool> W8<'t, FLUSH, RANKED_CLOSED> {
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
            #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
            let is_ranked_static = RANKED_CLOSED || self.ranked_static;
            #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
            let is_ranked_static = self.ranked_static;
            let static_b = FLUSH
                && (WORD < RANKED_STATIC_B_PREFIX_END || WORD >= RANKED_STATIC_B_TAIL_START)
                && is_ranked_static;
            if !static_b {
                store_v8(self.stage.add(WORD & (RING_WORDS - 1)) as *mut u32, v);
            }
            if FLUSH && WORD % RING_WORDS == RING_WORDS - 1 {
                if WORD + 1 == RING_WORDS {
                    (*self.drain).drain_range_gen::<RANKED_CLOSED>(16, 16, RING_WORDS - 16);
                } else {
                    (*self.drain).drain_range_gen::<RANKED_CLOSED>(
                        WORD + 1 - RING_WORDS,
                        0,
                        RING_WORDS,
                    );
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
                if USED == 0 {
                    if WIDTH == 32 {
                        self.write_word::<WORD>(v);
                        self.pending = dup_u32(0);
                    } else {
                        self.pending = shl_v8::<1>(v);
                    }
                } else {
                    let out = _mm256_shldi_epi32::<USED>(v, self.pending);
                    self.write_word::<WORD>(out);

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

#[inline(always)]
fn ror_v8<const N: i32>(v: V8) -> V8 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        unsafe { _mm256_ror_epi32::<N>(v) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        match N {
            16 => or_v8(shr_v8::<16>(v), shl_v8::<16>(v)),
            12 => or_v8(shr_v8::<12>(v), shl_v8::<20>(v)),
            8 => or_v8(shr_v8::<8>(v), shl_v8::<24>(v)),
            7 => or_v8(shr_v8::<7>(v), shl_v8::<25>(v)),
            _ => panic!("unexpected rotate count {N}"),
        }
    }
}

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

#[cfg(target_feature = "avx512f")]
#[inline(always)]
unsafe fn stream_ranked_line(p: *mut u32, v: __m512i) {
    unsafe { _mm512_stream_si512(p.cast::<__m512i>(), v) }
}

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

#[derive(Clone, Copy)]
struct StepRows {
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,

    flags: u8,
}

impl StepRows {
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

    #[cfg(target_feature = "avx512f")]
    #[inline(always)]
    unsafe fn publish_static_values(&self, j: usize, av: __m512i) {
        unsafe {
            let o = j * U32_PER_BLOCK;
            stream_ranked_line(self.z.add(o), av);
            stream_ranked_line(self.a.add(o), av);
        }
    }

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

#[inline(always)]
#[allow(dead_code)]
pub(crate) fn ranked_static01_direct_enabled() -> bool {
    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("FLOCK_NO_WITGEN_STATIC01_DIRECT").is_none()
        });
        *ON
    }
    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
    {
        false
    }
}

impl Drain8<'_> {
    #[inline(always)]
    unsafe fn drain_range_gen<const RANKED_CLOSED: bool>(
        &mut self,
        base_word: usize,
        ring_word: usize,
        words: usize,
    ) {
        unsafe {
            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni",
                target_feature = "vpclmulqdq"
            ))]
            if RANKED_CLOSED && let Some(roots) = self.roots.as_mut() {
                roots.emit_range(base_word, ring_word, words, self.ast, self.bs);
            }
            #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
            if RANKED_CLOSED {
                if base_word >= 32 && base_word <= 448 {
                    debug_assert_eq!(ring_word, 0);
                    debug_assert_eq!(words, 32);
                    self.drain_pair_ranked_closed_exact_fast(base_word);
                } else {
                    self.drain_static_or_sparse_ranked_closed_exact_fast(
                        base_word, ring_word, words,
                    );
                }
                return;
            }
            self.drain_range(base_word, ring_word, words);
        }
    }

    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn drain_range(&mut self, base_word: usize, ring_word: usize, words: usize) {
        unsafe {
            if self.ranked_static{
                #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
                if let Some(preps)=self.ranked_preps{
                    self.drain_range_spread_ranked_closed_exact(&self.proj,preps,base_word,ring_word,words);
                    return;
                }
                self.drain_range_spread::<true>(&self.proj,base_word,ring_word,words)
            }else{
                self.drain_range_spread::<false>(&self.proj,base_word,ring_word,words)
            };
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn drain_range_spread_ranked_closed_exact(
        &self,
        proj: &StreamProj<'_>,
        preps: &RankedClosedWindowPreps,
        base_word: usize,
        ring_word: usize,
        words: usize,
    ) {
        unsafe {
            debug_assert!(self.ranked_static);
            debug_assert!(proj.one_rows_elided);
            if base_word >= 32 && base_word <= 448 && ring_word == 0 && words == 32 {
                return self.drain_pair_ranked_closed_exact_fast(base_word);
            }
            if (base_word == 16 && ring_word == 16 && words == 16)
                || (base_word == 480 && ring_word == 0 && words == 32)
                || (base_word == 0 && ring_word == 0 && words == 16)
            {
                return self.drain_static_or_sparse_ranked_closed_exact_fast(
                    base_word,
                    ring_word,
                    words,
                );
            }
            self.drain_range_spread_ranked_closed_exact_fallback(
                proj,
                preps,
                base_word,
                ring_word,
                words,
            );
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[cfg_attr(
        all(
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512dq",
            target_feature = "gfni"
        ),
        target_feature(
            enable = "avx2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512dq,gfni"
        )
    )]
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn drain_static_or_sparse_ranked_closed_exact_fast(
        &self,
        base_word: usize,
        ring_word: usize,
        words: usize,
    ) {
        unsafe {
            if !self.ranked_fast {
                let preps = self.ranked_preps.unwrap_unchecked();
                return self.drain_range_spread_ranked_closed_exact_fallback(
                    &self.proj,
                    preps,
                    base_word,
                    ring_word,
                    words,
                );
            }
            if base_word == 480 {
                let x0 = load_v8(self.ast.cast::<u32>());
                let x1 = load_v8(self.ast.add(1).cast::<u32>());
                let l = _mm256_unpacklo_epi32(x0, x1);
                let h = _mm256_unpackhi_epi32(x0, x1);
                let mut q = [0u64; 8];
                store_v8(q.as_mut_ptr().cast::<u32>(), _mm256_permute2x128_si256::<0x20>(l, h));
                store_v8(q.as_mut_ptr().cast::<u32>().add(8), _mm256_permute2x128_si256::<0x31>(l, h));
                let rows = RankedRows::new(self.z.add(480), self.a.add(480), self.b.add(480));
                let qp = q.as_ptr();
                let mut j = 0usize;
                while j != 8 {
                    rows.publish_sparse_30(j, qp.add(j));
                    j += 1;
                }
            } else {
                debug_assert!(base_word == 0 || base_word == 16);
                let a_rows = tr8x16_zmm(self.ast, base_word);
                let rows = RankedRows::new(
                    self.z.add(base_word),
                    self.a.add(base_word),
                    self.b.add(base_word),
                );
                let mut j = 0usize;
                while j != 8 {
                    rows.publish_static_values(j, a_rows[j]);
                    j += 1;
                }
            }
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[cfg_attr(
        all(
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512dq",
            target_feature = "gfni"
        ),
        target_feature(
            enable = "avx2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512dq,gfni"
        )
    )]
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn drain_pair_ranked_closed_exact_fast(&self, base_word: usize) {
        unsafe {
            debug_assert!(base_word >= 32 && base_word <= 448);
            debug_assert!(base_word.is_multiple_of(32));
            let proj = &self.proj;
            let preps = self.ranked_preps.unwrap_unchecked();
            if !self.ranked_fast {
                return self.drain_range_spread_ranked_closed_exact_fallback(
                    proj,
                    preps,
                    base_word,
                    0,
                    32,
                );
            }
            for step in 0..2usize {
                let rw = step * STEP_WORDS;
                let abs_word = base_word + rw;
                let blk = abs_word / STEP_WORDS;
                let rows = RankedRows::new(
                    self.z.add(abs_word),
                    self.a.add(abs_word),
                    self.b.add(abs_word),
                );
                let plan = preps.get_unchecked(blk).0;
#[cfg(all(target_feature="avx512f",target_feature="avx512bw",target_feature="avx512vbmi",target_feature="gfni"))]
                if (2..=29).contains(&blk) && plan.bcomplement_static_eligible(){
                    xor_projection::project(proj,blk,rows,self.ast,self.bs,rw,&preps.tables);
                    continue;
                }
                self.drain_ranked_offset_window(abs_word, rw);
            }
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[cfg_attr(
        all(
            target_feature = "avx512vl",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512dq",
            target_feature = "gfni"
        ),
        target_feature(
            enable = "avx2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512dq,gfni"
        )
    )]
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn drain_ranked_offset_window(&self, abs_word: usize, rw: usize) {



        unsafe {
            let proj = &self.proj;
            let preps = self.ranked_preps.unwrap_unchecked();
            let blk = abs_word / STEP_WORDS;
            let rows = RankedRows::new(
                self.z.add(abs_word),
                self.a.add(abs_word),
                self.b.add(abs_word),
            );
            let (plan, imgs) = *preps.get_unchecked(blk);
            let wc = WidenConsts::new();
            #[repr(align(64))]
            struct OffArena([u16; 8 * ROUND1_AB_OFF_WORDS]);
            let mut arena = core::mem::MaybeUninit::<OffArena>::uninit();
            let op = core::ptr::addr_of_mut!((*arena.as_mut_ptr()).0) as *mut u16;
            project_blocks_ranked_hot_offsets_direct_inline_fast::<true>(
                proj,
                blk,
                plan,
                imgs,
                rows,
                self.ast,
                self.bs,
                rw,
                op,
                wc,
            );
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[rustfmt::skip]
    #[inline(never)]
    unsafe fn drain_range_spread_ranked_closed_exact_fallback(
        &self,
        proj: &StreamProj<'_>,
        preps: &RankedClosedWindowPreps,
        base_word: usize,
        ring_word: usize,
        words: usize,
    ) {
        unsafe {
            let wc = WidenConsts::new();
            let static01_direct = ranked_static01_direct_enabled();
            for off in (0..words).step_by(STEP_WORDS) {
                let abs_word = base_word + off;
                let rw = ring_word + off;
                let blk = abs_word / STEP_WORDS;
                match blk {
                    0 | 1 if static01_direct => {
                        let a_rows = tr8x16_zmm(self.ast, rw);
                        let rows=RankedRows::new(self.z.add(abs_word),self.a.add(abs_word),self.b.add(abs_word));
                        let mut j = 0usize;
                        while j != 8 {
                            rows.publish_static_values(j, a_rows[j]);
                            j += 1;
                        }
                    }
                    0 => {
                        let (sa, _) = proj.sides();
                        let a_lo = tr8_chunk(self.ast, rw);
                        let a_hi = tr8_chunk(self.ast, rw + 8);
                        for r in 0..8 {
                            let p = sa.add(r * STEP_WORDS);
                            store_v8(p, a_lo[r]);
                            store_v8(p.add(8), a_hi[r]);
                        }
                        let rows=RankedRows::new(self.z.add(abs_word),self.a.add(abs_word),self.b.add(abs_word));
                        let mut j = 0usize;
                        while j != 8 {
                            rows.publish_static::<0>(j, sa);
                            j += 1;
                        }
                    }
                    1 => {
                        let (sa, _) = proj.sides();
                        let a_lo = tr8_chunk(self.ast, rw);
                        let a_hi = tr8_chunk(self.ast, rw + 8);
                        for r in 0..8 {
                            let p = sa.add(r * STEP_WORDS);
                            store_v8(p, a_lo[r]);
                            store_v8(p.add(8), a_hi[r]);
                        }
                        let rows=RankedRows::new(self.z.add(abs_word),self.a.add(abs_word),self.b.add(abs_word));
                        let mut j = 0usize;
                        while j != 8 {
                            rows.publish_static::<1>(j, sa);
                            j += 1;
                        }
                    }
                    30 => {
                        let (sa, _) = proj.sides();
                        let x0 = load_v8(self.ast.add(rw).cast::<u32>());
                        let x1 = load_v8(self.ast.add(rw + 1).cast::<u32>());
                        let l = _mm256_unpacklo_epi32(x0, x1);
                        let h = _mm256_unpackhi_epi32(x0, x1);
                        store_v8(sa, _mm256_permute2x128_si256::<0x20>(l, h));
                        store_v8(sa.add(8), _mm256_permute2x128_si256::<0x31>(l, h));
                        let z = _mm256_setzero_si256();
                        store_v8(sa.add(16), z);
                        store_v8(sa.add(24), z);
                        let rows=RankedRows::new(self.z.add(abs_word),self.a.add(abs_word),self.b.add(abs_word));
                        let (plan, imgs) = preps[30];
                        proj.project_blocks_ranked_30(plan, imgs, rows);
                    }
                    31 => continue,
                    2..=29 => {
                        #[repr(align(64))]
                        struct OffArena([u16; 8 * ROUND1_AB_OFF_WORDS]);
                        let mut arena = core::mem::MaybeUninit::<OffArena>::uninit();
                        let op = core::ptr::addr_of_mut!((*arena.as_mut_ptr()).0) as *mut u16;
                        let rows=RankedRows::new(self.z.add(abs_word),self.a.add(abs_word),self.b.add(abs_word));
                        let (plan, imgs) = preps[blk];
                        project_blocks_ranked_hot_offsets_direct_inline::<true>(
                            proj,
                            blk,
                            plan,
                            imgs,
                            rows,
                            self.ast,
                            self.bs,
                            rw,
                            op,
                            wc,
                        );
                    }
                    _ => unreachable!("unexpected ranked closed window {blk}"),
                }
            }
        }
    }

















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




            let dead_w31 = E && proj.one_rows_elided && ey_dead_w31_enabled();




            #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
            let maddubs = E && widen_maddubs_enabled();
            #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
            let wc = WidenConsts::new();
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








                #[repr(align(64))]
                struct OffArena([u16; 8 * ROUND1_AB_OFF_WORDS]);
                let mut arena = core::mem::MaybeUninit::<OffArena>::uninit();
                let op = core::ptr::addr_of_mut!((*arena.as_mut_ptr()).0) as *mut u16;

                if E {





                    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
                    {
                        let rows=RankedRows::new(self.z.add(abs_word),self.a.add(abs_word),self.b.add(abs_word));
                        if ranked_direct_dense_publish_enabled() {
                            if ranked_direct_dense_inline_enabled() {
                                if maddubs {
                                    project_blocks_ranked_hot_offsets_direct_inline::<true>(proj,blk,plan,imgs,rows,self.ast,self.bs,rw,op,wc);
                                } else {
                                    project_blocks_ranked_hot_offsets_direct_inline::<false>(proj,blk,plan,imgs,rows,self.ast,self.bs,rw,op,wc);
                                }
                            } else if maddubs {
                                let a_rows=ranked_dense_rows_and_offsets_rollback::<true>(self.ast,rw,op,wc);
                                let b_rows=ranked_dense_rows_and_offsets_rollback::<true>(self.bs,rw,op.add(64),wc);
                                proj.project_blocks_ranked_hot_offsets_direct_rollback::<true>(blk,plan,imgs,rows,&a_rows,&b_rows,op as *const u16);
                            } else {
                                let a_rows=ranked_dense_rows_and_offsets_rollback::<false>(self.ast,rw,op,wc);
                                let b_rows=ranked_dense_rows_and_offsets_rollback::<false>(self.bs,rw,op.add(64),wc);
                                proj.project_blocks_ranked_hot_offsets_direct_rollback::<false>(blk,plan,imgs,rows,&a_rows,&b_rows,op as *const u16);
                            }
                        } else if maddubs {
                            stage_ranked_dense_side::<true>(self.ast,rw,sa,op,wc);
                            stage_ranked_dense_side::<true>(self.bs,rw,sb,op.add(64),wc);
                            proj.project_blocks_ranked_hot_offsets::<true>(blk,plan,imgs,rows,op as *const u16);
                        } else {
                            stage_ranked_dense_side::<false>(self.ast,rw,sa,op,wc);
                            stage_ranked_dense_side::<false>(self.bs,rw,sb,op.add(64),wc);
                            proj.project_blocks_ranked_hot_offsets::<false>(blk,plan,imgs,rows,op as *const u16);
                        }
                    }



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
                        proj.project_blocks_ranked_hot_offsets::<false>(blk,plan,imgs,rows,op as *const u16);
                    }
                } else {


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

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn build_octa_witness_ab_stream_prepared_elide<const RANKED_CLOSED: bool>(
    prepared: &PreparedInputs,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    proj: StreamProj<'_>,
    elide: [bool; 3],
    ranked_static: bool,
    ranked_preps: Option<&RankedClosedWindowPreps>,
    #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    roots: Option<root_emit::RootOctaWriter<'_>>,
) {
    unsafe {
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
        let ranked_static = RANKED_CLOSED || ranked_static;
        let ranked_fast = RANKED_CLOSED
            && ranked_static
            && proj.w30_elided
            && ranked_static01_direct_enabled()
            && ranked_b_constwin_enabled();
        let cv_v = &prepared.cv;
        let m = &prepared.message;
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
            ranked_fast,
            ranked_preps,
            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni",
                target_feature = "vpclmulqdq"
            ))]
            roots,
        };
        let maxv = dup_u32(u32::MAX);
        let one = dup_u32(1);
        let chain: [V8; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15], tlo, thi, blen, flags,
        ];

        for k in 0..20usize {
            let v = if k == 0 {
                or_v8(one, shl_v8::<1>(chain[0]))
            } else {
                or_v8(shr_v8::<31>(chain[k - 1]), shl_v8::<1>(chain[k]))
            };
            let w = 16 + k;
            store_v8(ast.add(w & (RING_WORDS - 1)) as *mut u32, v);
        }

        if !ranked_static {
            for w in 16..RANKED_STATIC_B_PREFIX_END {
                store_v8(bs.add(w & (RING_WORDS - 1)) as *mut u32, maxv);
            }
        }
        for w in RANKED_STATIC_B_PREFIX_END..16 + PROLOGUE_WORDS {
            store_v8(bs.add(w & (RING_WORDS - 1)) as *mut u32, maxv);
        }

        if RING_WORDS <= PROLOGUE_WORDS + 16 {
            drain.drain_range_gen::<RANKED_CLOSED>(16, 16, RING_WORDS - 16);
        }

        let pending_bit = packer_initial_bit(shr_v8::<31>(flags));
        let drain_ptr = &mut drain as *mut Drain8;
        let mut wa = W8::<false, RANKED_CLOSED>::at(ast, pending_bit, drain_ptr, false);

        let mut wb =
            W8::<true, RANKED_CLOSED>::at(bs, packer_initial_bit(one), drain_ptr, ranked_static);

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

        if !ranked_static {
            for w in ZF..U32_PER_BLOCK {
                let i = w & (RING_WORDS - 1);
                store_v8(ast.add(i) as *mut u32, zero);
                store_v8(bs.add(i) as *mut u32, zero);
            }
        }
        drain.drain_range_gen::<RANKED_CLOSED>(U32_PER_BLOCK - RING_WORDS, 0, RING_WORDS);

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
        drain.drain_range_gen::<RANKED_CLOSED>(0, 0, 16);
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi",
        target_feature = "avx512vbmi2",
        target_feature = "avx512dq",
        target_feature = "gfni"
    ),
    target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512dq,gfni")
)]
pub(crate) unsafe fn build_octa_witness_ab_stream_elide(
    inputs: OctaInputs<'_>,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    proj: StreamProj<'_>,
    elide: [bool; 3],
) {
    unsafe {
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
        build_octa_witness_ab_stream_prepared_elide::<false>(
            &prepared,
            z,
            a,
            b,
            proj,
            elide,
            ranked_static,
            None,
            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni",
                target_feature = "vpclmulqdq"
            ))]
            None,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vl",
        target_feature = "avx512vbmi",
        target_feature = "avx512vbmi2",
        target_feature = "avx512dq",
        target_feature = "gfni"
    ),
    target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512dq,gfni")
)]
pub(crate) unsafe fn build_octa_witness_ab_stream_ranked_closed(
    init: u64,
    base: usize,
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
    proj: StreamProj<'_>,
    preps: &RankedClosedWindowPreps,
    elide: [bool; 3],
    #[cfg(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    roots: Option<root_emit::RootOctaWriter<'_>>,
) {
    unsafe {
        debug_assert_eq!(elide, [true; 3]);
        debug_assert_eq!(proj.one_rows_elided, proj.plan.offsets_eligible(2));
        debug_assert_eq!(proj.out_bias, if proj.one_rows_elided { 2 * 64 } else { 0 });
        debug_assert!(ranked_direct_dense_publish_enabled());
        debug_assert!(ranked_direct_dense_inline_enabled());
        debug_assert!(widen_maddubs_enabled());
        let ranked_static = proj.plan.offsets_eligible(2);
        let prepared = prepare_closed_inputs(init, base);
        build_octa_witness_ab_stream_prepared_elide::<true>(
            &prepared,
            z,
            a,
            b,
            proj,
            elide,
            ranked_static,
            Some(preps),
            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni",
                target_feature = "vpclmulqdq"
            ))]
            roots,
        );
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
                    let expected = crate::s1::gen_block(init, base + lane);
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
    fn ranked_static01_direct_publish_matches_staged() {
        #[repr(C, align(64))]
        struct Input([[u32; 8]; STEP_WORDS]);
        #[repr(C, align(64))]
        struct Stage([u32; 8 * STEP_WORDS]);
        #[repr(C, align(64))]
        struct RankedBuf([u32; 8 * U32_PER_BLOCK]);

        unsafe {
            for case in 0..4u32 {
                let input = Input(core::array::from_fn(|word| {
                    core::array::from_fn(|block| match case {
                        0 => 0,
                        1 => u32::MAX,
                        _ => {
                            0x9E37_79B9u32.wrapping_mul(case)
                                ^ ((word as u32) << 11)
                                ^ (block as u32).wrapping_mul(0x0101_0101)
                        }
                    })
                }));
                let ring = input.0.as_ptr().cast::<V8>();
                let mut stage = Stage([0u32; 8 * STEP_WORDS]);
                let a_lo = tr8_chunk(ring, 0);
                let a_hi = tr8_chunk(ring, 8);
                for r in 0..8 {
                    let p = stage.0.as_mut_ptr().add(r * STEP_WORDS);
                    store_v8(p, a_lo[r]);
                    store_v8(p.add(8), a_hi[r]);
                }
                let a_rows = tr8x16_zmm(ring, 0);
                for blk in 0..2usize {
                    let mut staged_z = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                    let mut staged_a = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                    let mut staged_b = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                    let mut direct_z = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                    let mut direct_a = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                    let mut direct_b = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                    let staged = RankedRows::new(
                        staged_z.0.as_mut_ptr(),
                        staged_a.0.as_mut_ptr(),
                        staged_b.0.as_mut_ptr(),
                    );
                    let direct = RankedRows::new(
                        direct_z.0.as_mut_ptr(),
                        direct_a.0.as_mut_ptr(),
                        direct_b.0.as_mut_ptr(),
                    );
                    for j in 0..8 {
                        if blk == 0 {
                            staged.publish_static::<0>(j, stage.0.as_ptr());
                        } else {
                            staged.publish_static::<1>(j, stage.0.as_ptr());
                        }
                        direct.publish_static_values(j, a_rows[j]);
                    }
                    _mm_sfence();
                    assert_eq!(staged_z.0, direct_z.0, "z mismatch, case={case} blk={blk}");
                    assert_eq!(staged_a.0, direct_a.0, "a mismatch, case={case} blk={blk}");
                    assert_eq!(staged_b.0, direct_b.0, "b mismatch, case={case} blk={blk}");
                }
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
                let wc = WidenConsts::new();
                let mut inline_off = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                widen_ranked_dense_rows::<false>(&a_rows, inline_off.0.as_mut_ptr(), wc);
                widen_ranked_dense_rows::<false>(&b_rows, inline_off.0.as_mut_ptr().add(64), wc);

                let mut rollback_off = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                let rollback_a_rows = ranked_dense_rows_and_offsets_rollback::<false>(
                    a_input.0.as_ptr().cast::<V8>(),
                    0,
                    rollback_off.0.as_mut_ptr(),
                    wc,
                );
                let rollback_b_rows = ranked_dense_rows_and_offsets_rollback::<false>(
                    b_input.0.as_ptr().cast::<V8>(),
                    0,
                    rollback_off.0.as_mut_ptr().add(64),
                    wc,
                );

                let mut staged = Stage {
                    a: [0u32; 8 * STEP_WORDS],
                    b: [0u32; 8 * STEP_WORDS],
                };
                let mut staged_off = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                stage_ranked_dense_side::<false>(
                    a_input.0.as_ptr().cast::<V8>(),
                    0,
                    staged.a.as_mut_ptr(),
                    staged_off.0.as_mut_ptr(),
                    wc,
                );
                stage_ranked_dense_side::<false>(
                    b_input.0.as_ptr().cast::<V8>(),
                    0,
                    staged.b.as_mut_ptr(),
                    staged_off.0.as_mut_ptr().add(64),
                    wc,
                );

                let mut parity_off = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                widen_ranked_dense_rows::<true>(&a_rows, parity_off.0.as_mut_ptr(), wc);
                widen_ranked_dense_rows::<true>(&b_rows, parity_off.0.as_mut_ptr().add(64), wc);
                let mut parity_rollback_off = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                let _ = ranked_dense_rows_and_offsets_rollback::<true>(
                    a_input.0.as_ptr().cast::<V8>(),
                    0,
                    parity_rollback_off.0.as_mut_ptr(),
                    wc,
                );
                let _ = ranked_dense_rows_and_offsets_rollback::<true>(
                    b_input.0.as_ptr().cast::<V8>(),
                    0,
                    parity_rollback_off.0.as_mut_ptr().add(64),
                    wc,
                );
                let mut parity_staged = Stage {
                    a: [0u32; 8 * STEP_WORDS],
                    b: [0u32; 8 * STEP_WORDS],
                };
                let mut parity_staged_off = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                stage_ranked_dense_side::<true>(
                    a_input.0.as_ptr().cast::<V8>(),
                    0,
                    parity_staged.a.as_mut_ptr(),
                    parity_staged_off.0.as_mut_ptr(),
                    wc,
                );
                stage_ranked_dense_side::<true>(
                    b_input.0.as_ptr().cast::<V8>(),
                    0,
                    parity_staged.b.as_mut_ptr(),
                    parity_staged_off.0.as_mut_ptr().add(64),
                    wc,
                );
                assert_eq!(parity_staged.a, staged.a, "parity staged a, case={case}");
                assert_eq!(parity_staged.b, staged.b, "parity staged b, case={case}");
                for j in 0..8 {
                    for side in 0..2 {
                        for i in 0..64 {
                            let w = j * ROUND1_AB_OFF_WORDS + side * 64 + i;
                            let p = j * ROUND1_AB_OFF_WORDS + side * 64 + (i & 1) * 32 + (i >> 1);
                            assert_eq!(
                                parity_off.0[p], staged_off.0[w],
                                "parity slot mismatch, case={case} j={j} side={side} i={i}"
                            );
                        }
                    }
                }
                assert_eq!(
                    parity_rollback_off.0, parity_off.0,
                    "parity rollback, case={case}"
                );
                assert_eq!(
                    parity_staged_off.0, parity_off.0,
                    "parity staged, case={case}"
                );

                for row in 0..8 {
                    let mut got_a = [0u32; STEP_WORDS];
                    let mut got_b = [0u32; STEP_WORDS];
                    _mm512_storeu_si512(got_a.as_mut_ptr().cast::<__m512i>(), a_rows[row]);
                    _mm512_storeu_si512(got_b.as_mut_ptr().cast::<__m512i>(), b_rows[row]);
                    let mut rollback_a = [0u32; STEP_WORDS];
                    let mut rollback_b = [0u32; STEP_WORDS];
                    _mm512_storeu_si512(
                        rollback_a.as_mut_ptr().cast::<__m512i>(),
                        rollback_a_rows[row],
                    );
                    _mm512_storeu_si512(
                        rollback_b.as_mut_ptr().cast::<__m512i>(),
                        rollback_b_rows[row],
                    );
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
                    assert_eq!(
                        rollback_a, got_a,
                        "rollback a row mismatch, case={case} row={row}"
                    );
                    assert_eq!(
                        rollback_b, got_b,
                        "rollback b row mismatch, case={case} row={row}"
                    );
                }
                assert_eq!(
                    inline_off.0, staged_off.0,
                    "inline offset mismatch, case={case}"
                );
                assert_eq!(
                    rollback_off.0, staged_off.0,
                    "rollback offset mismatch, case={case}"
                );

                let mut staged_z = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut staged_a = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut staged_b = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut direct_z = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut direct_a = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let mut direct_b = RankedBuf([0u32; 8 * U32_PER_BLOCK]);
                let staged_rows = RankedRows::new(
                    staged_z.0.as_mut_ptr(),
                    staged_a.0.as_mut_ptr(),
                    staged_b.0.as_mut_ptr(),
                );
                let direct_rows = RankedRows::new(
                    direct_z.0.as_mut_ptr(),
                    direct_a.0.as_mut_ptr(),
                    direct_b.0.as_mut_ptr(),
                );
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

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
    #[test]
    fn parity_widen_and_consumers_match_byte_order_arena() {
        use super::round1_ab_inner_window_from_offsets_nt2 as nt2;
        use super::round1_ab_inner_window_from_offsets_nt2_bcomplement_static as bcomp;
        use super::round1_ab_inner_window_from_offsets_nt2_bcomplement_static_const as bcomp_const;
        use super::round1_ab_inner_window_from_offsets_nt2_residual as residual;
        use flock_core::zerocheck::univariate_skip_optimized::{
            abinner_publish_fence, prepare_round1_ab_window_plan,
        };
        type Block = [u16; ROUND1_AB_OFF_WORDS];
        #[repr(C, align(64))]
        struct Input([[u32; 8]; STEP_WORDS]);
        #[repr(C, align(64))]
        struct Offsets([u16; 8 * ROUND1_AB_OFF_WORDS]);
        #[repr(C, align(64))]
        struct Out([u8; 64]);

        let ntt_s = flock_core::ntt::AdditiveNttGf8::new(6, flock_core::field::F8::ZERO);
        let ntt_l = flock_core::ntt::AdditiveNttGf8::new(6, flock_core::field::F8(64));
        let inv_table = flock_core::ntt::InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        let mut out_w = Out([0u8; 64]);
        let mut out_p = Out([0u8; 64]);
        let plan = prepare_round1_ab_window_plan(&inv_table, &out_w.0, true);
        let imgs = round1_ab_table_images(&inv_table, plan);
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };
        macro_rules! check {
            ($name:literal, $case:expr, $j:expr, $blk:expr, $w:expr, $p:expr) => {{
                out_w.0 = [0xA5; 64];
                out_p.0 = [0x5A; 64];
                $w;
                $p;
                abinner_publish_fence();
                assert_eq!(
                    out_w.0, out_p.0,
                    "{} case={} j={} blk={}",
                    $name, $case, $j, $blk
                );
            }};
        }
        unsafe {
            let wc = WidenConsts::new();
            for case in 0..8usize {
                let a_input = Input(core::array::from_fn(|_| core::array::from_fn(|_| next())));
                let b_input = Input(core::array::from_fn(|_| {
                    core::array::from_fn(|_| if case == 7 { u32::MAX } else { next() })
                }));
                let a_rows = tr8x16_zmm(a_input.0.as_ptr().cast::<V8>(), 0);
                let b_rows = tr8x16_zmm(b_input.0.as_ptr().cast::<V8>(), 0);
                let mut off_w = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                widen_ranked_dense_rows::<false>(&a_rows, off_w.0.as_mut_ptr(), wc);
                widen_ranked_dense_rows::<false>(&b_rows, off_w.0.as_mut_ptr().add(64), wc);
                let mut off_p = Offsets([0u16; 8 * ROUND1_AB_OFF_WORDS]);
                widen_ranked_dense_rows::<true>(&a_rows, off_p.0.as_mut_ptr(), wc);
                widen_ranked_dense_rows::<true>(&b_rows, off_p.0.as_mut_ptr().add(64), wc);
                for j in 0..8usize {
                    let w = &*off_w
                        .0
                        .as_ptr()
                        .add(j * ROUND1_AB_OFF_WORDS)
                        .cast::<Block>();
                    let p = &*off_p
                        .0
                        .as_ptr()
                        .add(j * ROUND1_AB_OFF_WORDS)
                        .cast::<Block>();
                    check!(
                        "nt2",
                        case,
                        j,
                        0,
                        nt2::<false>(w, &mut out_w.0, plan, imgs),
                        nt2::<true>(p, &mut out_p.0, plan, imgs)
                    );
                    for keep in [0xfcu8, 0x0f] {
                        check!(
                            "residual",
                            case,
                            j,
                            keep,
                            residual::<false>(w, &mut out_w.0, plan, imgs, keep),
                            residual::<true>(p, &mut out_p.0, plan, imgs, keep)
                        );
                    }
                    if plan.bcomplement_static_eligible() {
                        for blk in 2..=29usize {
                            check!(
                                "bcomplement",
                                case,
                                j,
                                blk,
                                bcomp::<false>(w, &mut out_w.0, plan, imgs, blk),
                                bcomp::<true>(p, &mut out_p.0, plan, imgs, blk)
                            );
                        }
                        macro_rules! const_blk {
                            ($($n:literal),*) => {$(
                                check!(
                                    "bcomplement const",
                                    case,
                                    j,
                                    $n,
                                    bcomp_const::<$n, false>(w, &mut out_w.0, plan, imgs),
                                    bcomp_const::<$n, true>(p, &mut out_p.0, plan, imgs)
                                );
                            )*};
                        }
                        const_blk!(
                            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                            22, 23, 24, 25, 26, 27, 28
                        );
                    }
                }
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

    #[test]
    fn ranked_closed_matches_generic_octa() {
        #[repr(C, align(64))]
        #[derive(Clone, PartialEq, Eq, Debug)]
        struct AlignedWords([u32; 8 * U32_PER_BLOCK]);
        #[repr(C, align(64))]
        #[derive(Clone)]
        struct AlignedAbCompact([u8; 8 * 29 * 64]);
        #[repr(C, align(64))]
        #[derive(Clone)]
        struct AlignedAbFull([u8; 8 * 32 * 64]);
        #[repr(C, align(64))]
        struct AlignedStage([u32; STREAM_STAGE_WORDS]);

        let ntt_s =
            flock_core::ntt::AdditiveNttGf8::new(super::super::K_SKIP, flock_core::field::F8::ZERO);
        let ntt_l = flock_core::ntt::AdditiveNttGf8::new(
            super::super::K_SKIP,
            flock_core::field::F8(1u8 << super::super::K_SKIP),
        );
        let inv_table = flock_core::ntt::InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        let dummy_ab = Box::new(AlignedAbCompact([0u8; 8 * 29 * 64]));
        let win_plan =
            flock_core::zerocheck::univariate_skip_optimized::prepare_round1_ab_window_plan(
                &inv_table,
                &dummy_ab.0,
                true,
            );
        let preps = prepare_ranked_closed_window_preps(&inv_table, win_plan);
        let crate::s1::BlockSource::Closed { init, .. } =
            crate::s1::BlockSource::closed(18, 0xD1CE_C105_EDA5_1E5Du64)
        else {
            unreachable!()
        };
        let compact = win_plan.offsets_eligible(2);
        let out_stride = if compact { 29 * 64 } else { 32 * 64 };
        let out_bias = if compact { 2 * 64 } else { 0 };
        let mut stage_ref = Box::new(AlignedStage([0u32; STREAM_STAGE_WORDS]));
        let mut stage_new = Box::new(AlignedStage([0u32; STREAM_STAGE_WORDS]));

        let mut z_seed = Box::new(AlignedWords([0u32; 8 * U32_PER_BLOCK]));
        let mut a_seed = Box::new(AlignedWords([0u32; 8 * U32_PER_BLOCK]));
        let mut b_seed = Box::new(AlignedWords([0u32; 8 * U32_PER_BLOCK]));
        let mut ab_seed = Box::new(AlignedAbFull([0u8; 8 * 32 * 64]));
        unsafe {
            build_octa_witness_ab_stream_elide(
                OctaInputs::Closed { init, base: 0 },
                z_seed.0.as_mut_ptr(),
                a_seed.0.as_mut_ptr(),
                b_seed.0.as_mut_ptr(),
                StreamProj {
                    stage: stage_ref.0.as_mut_ptr(),
                    out: ab_seed.0.as_mut_ptr(),
                    out_stride: 32 * 64,
                    out_bias: 0,
                    inv_table: &inv_table,
                    plan: win_plan,
                    one_rows_elided: false,
                    w30_elided: false,
                },
                [false; 3],
            );
        }

        for base in [0usize, 8, 240, 131_064] {
            let mut z_ref = z_seed.clone();
            let mut a_ref = a_seed.clone();
            let mut b_ref = b_seed.clone();
            let mut ab_ref = Box::new(AlignedAbFull([0u8; 8 * 32 * 64]));

            let mut z_new = z_seed.clone();
            let mut a_new = a_seed.clone();
            let mut b_new = b_seed.clone();
            let mut ab_new = Box::new(AlignedAbFull([0u8; 8 * 32 * 64]));
            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni",
                target_feature = "vpclmulqdq"
            ))]
            let fold = super::super::Blake3AdjointPlan::build().linear_fold_plan();
            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni",
                target_feature = "vpclmulqdq"
            ))]
            let roots = root_emit::RootEmitPlan::new(&fold.roots).unwrap();
            #[repr(align(64))]
            struct RootTile([u8; 176 * 512]);
            let mut root_tile = Box::new(RootTile([0xa5; 176 * 512]));

            unsafe {
                build_octa_witness_ab_stream_elide(
                    OctaInputs::Closed { init, base },
                    z_ref.0.as_mut_ptr(),
                    a_ref.0.as_mut_ptr(),
                    b_ref.0.as_mut_ptr(),
                    StreamProj {
                        stage: stage_ref.0.as_mut_ptr(),
                        out: ab_ref.0.as_mut_ptr(),
                        out_stride,
                        out_bias,
                        inv_table: &inv_table,
                        plan: win_plan,
                        one_rows_elided: compact,
                        w30_elided: compact,
                    },
                    [true; 3],
                );
                build_octa_witness_ab_stream_ranked_closed(
                    init,
                    base,
                    z_new.0.as_mut_ptr(),
                    a_new.0.as_mut_ptr(),
                    b_new.0.as_mut_ptr(),
                    StreamProj {
                        stage: stage_new.0.as_mut_ptr(),
                        out: ab_new.0.as_mut_ptr(),
                        out_stride,
                        out_bias,
                        inv_table: &inv_table,
                        plan: win_plan,
                        one_rows_elided: compact,
                        w30_elided: compact,
                    },
                    &preps,
                    [true; 3],
                    #[cfg(all(
                        target_feature = "avx512f",
                        target_feature = "avx512bw",
                        target_feature = "avx512vbmi",
                        target_feature = "gfni",
                        target_feature = "vpclmulqdq"
                    ))]
                    Some(root_emit::RootOctaWriter::new(
                        &roots,
                        root_tile.0.as_mut_ptr(),
                    )),
                );
                _mm_sfence();
            }
            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni",
                target_feature = "vpclmulqdq"
            ))]
            for group in 0..176 {
                for column in 0..64 {
                    let lane = column % 32 / 4;
                    let expected = (0..8).fold(0u8, |bits, bit| {
                        let value = fold
                            .roots
                            .get({
                                let index = group * 64 + column / 32 * 32 + column % 4 * 8 + bit;
                                index + if index >= 833 { 64 } else { 0 }
                            })
                            .map_or(0, |&wire| {
                                (z_new.0[lane * U32_PER_BLOCK + wire / 32] >> (wire % 32)) & 1
                            });
                        bits | (value as u8) << bit
                    });
                    assert_eq!(
                        root_tile.0[group * 512 + column],
                        expected,
                        "live drain base={base}, group={group}, column={column}"
                    );
                }
                assert!(
                    root_tile.0[group * 512 + 64..(group + 1) * 512]
                        .iter()
                        .all(|&v| v == 0xa5)
                );
            }
            assert_eq!(z_ref, z_new, "z mismatch at base={base}");
            assert_eq!(a_ref, a_new, "a mismatch at base={base}");
            assert_eq!(b_ref, b_new, "b mismatch at base={base}");
            let check_bytes = if compact { 28 * 64 } else { 31 * 64 };
            for j in 0..8 {
                let s = j * out_stride;
                assert_eq!(
                    &ab_ref.0[s..s + check_bytes],
                    &ab_new.0[s..s + check_bytes],
                    "ab mismatch at base={base} lane={j}"
                );
            }
        }
    }
}

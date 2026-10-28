use flock_core::field::F128;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::{ADDS_PER_G, BLAKE3_IV, CARRY_BITS_PER_ADD, G_STRIDE, GS_BASE, OUT_HI_BASE, WORD_BITS};
use super::{Compression, K, USEFUL_BITS};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REC_C0: usize = 0;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REC_C1: usize = CARRY_BITS_PER_ADD;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REC_C2: usize = 2 * CARRY_BITS_PER_ADD;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REC_C3: usize = 3 * CARRY_BITS_PER_ADD;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REC_C4: usize = 4 * CARRY_BITS_PER_ADD;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REC_C5: usize = 5 * CARRY_BITS_PER_ADD;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REC_LIN0: usize = ADDS_PER_G * CARRY_BITS_PER_ADD;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REC_LIN1: usize = REC_LIN0 + WORD_BITS;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const U32_PER_BLOCK: usize = K / 32;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LAST_WORD: usize = (USEFUL_BITS - 1) / 32;

const F128_PER_BLOCK: usize = K / 128;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) enum OctaInputs<'a> {
    Blocks([&'a Compression; 8]),
    Closed { init: u64, base: usize },
}

#[cfg(target_arch = "x86_64")]
mod x86_fast {
    use super::*;
    use core::arch::x86_64::*;

    type V8 = __m256i;

    struct PreparedInputs {
        cv: [V8; 8],
        message: [V8; 16],
        counter_lo: V8,
        counter_hi: V8,
        block_len: V8,
        flags: V8,
    }

    #[inline(always)]
    unsafe fn load_v8(p: *const u32) -> V8 {
        unsafe { _mm256_loadu_si256(p.cast::<__m256i>()) }
    }

    #[inline(always)]
    unsafe fn store_v8(p: *mut u32, v: V8) {
        unsafe { _mm256_storeu_si256(p.cast::<__m256i>(), v) }
    }

    #[inline(always)]
    fn dup_u32(x: u32) -> V8 {
        unsafe { _mm256_set1_epi32(x as i32) }
    }

    #[inline(always)]
    fn xor_v8(a: V8, b: V8) -> V8 {
        unsafe { _mm256_xor_si256(a, b) }
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

    #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
    #[inline(always)]
    fn vsli_v8<const N: i32>(a: V8, b: V8) -> V8 {
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
        unsafe {
            let mask = _mm256_set1_epi32(((1u64 << N) - 1) as u32 as i32);
            _mm256_ternarylogic_epi32::<0xF8>(_mm256_slli_epi32::<N>(b), a, mask)
        }
        #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512vl")))]
        unsafe {
            let mask = _mm256_set1_epi32(((1u64 << N) - 1) as u32 as i32);
            _mm256_or_si256(_mm256_slli_epi32::<N>(b), _mm256_and_si256(a, mask))
        }
    }

    #[inline(always)]
    fn ror_v8<const N: i32>(v: V8) -> V8 {
        #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
        unsafe {
            _mm256_ror_epi32::<N>(v)
        }
        #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512vl")))]
        {
            match N {
                16 => or_v8(shr_v8::<16>(v), shl_v8::<16>(v)),
                12 => or_v8(shr_v8::<12>(v), shl_v8::<20>(v)),
                8 => or_v8(shr_v8::<8>(v), shl_v8::<24>(v)),
                7 => or_v8(shr_v8::<7>(v), shl_v8::<25>(v)),
                _ => unreachable!(),
            }
        }
    }

    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512dq")))]
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

    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512dq")))]
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

    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512dq")))]
    #[inline(always)]
    unsafe fn pack_low_u32x8(lo: __m256i, hi: __m256i) -> V8 {
        unsafe {
            let indices = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
            let lo = _mm256_permutevar8x32_epi32(lo, indices);
            let hi = _mm256_permutevar8x32_epi32(hi, indices);
            _mm256_permute2x128_si256::<0x20>(lo, hi)
        }
    }

    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512dq")))]
    #[inline(always)]
    unsafe fn next_generator_draw(states: &mut [__m256i; 2]) -> V8 {
        unsafe {
            let golden = _mm256_set1_epi64x(0x9E37_79B9_7F4A_7C15u64 as i64);
            states[0] = _mm256_add_epi64(states[0], golden);
            states[1] = _mm256_add_epi64(states[1], golden);
            pack_low_u32x8(mix_u64x4(states[0]), mix_u64x4(states[1]))
        }
    }

    #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512dq")))]
    #[inline(always)]
    unsafe fn prepare_closed_inputs(init: u64, base: usize) -> PreparedInputs {
        unsafe {
            const DRAWS_PER_BLOCK: u64 = 25;
            const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
            let stride = GOLDEN.wrapping_mul(DRAWS_PER_BLOCK);
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

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512dq"))]
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

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512dq"))]
    #[inline(always)]
    unsafe fn next_generator_draw(state: &mut __m512i) -> V8 {
        unsafe {
            *state = _mm512_add_epi64(*state, _mm512_set1_epi64(0x9E37_79B9_7F4A_7C15u64 as i64));
            _mm512_cvtepi64_epi32(mix_u64x8(*state))
        }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512dq"))]
    #[inline(always)]
    unsafe fn prepare_closed_inputs(init: u64, base: usize) -> PreparedInputs {
        unsafe {
            const DRAWS_PER_BLOCK: u64 = 25;
            const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
            let stride = GOLDEN.wrapping_mul(DRAWS_PER_BLOCK);
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

    struct W32 {
        pending: V8,
        stage: *mut V8,
    }

    impl W32 {
        #[inline(always)]
        fn at(stage: *mut V8, pending: V8) -> Self {
            Self { pending, stage }
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
                            store_v8(self.stage.add(WORD) as *mut u32, v);
                            self.pending = dup_u32(0);
                        } else {
                            self.pending = shl_v8::<1>(v);
                        }
                    } else {
                        let out = _mm256_shldi_epi32::<USED>(v, self.pending);
                        store_v8(self.stage.add(WORD) as *mut u32, out);
                        self.pending = if WIDTH == 31 { shl_v8::<1>(v) } else { v };
                    }
                }
                #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
                {
                    if USED == 0 {
                        if WIDTH == 32 {
                            store_v8(self.stage.add(WORD) as *mut u32, v);
                            self.pending = dup_u32(0);
                        } else {
                            self.pending = v;
                        }
                    } else if USED + WIDTH < 32 {
                        self.pending = vsli_v8::<USED>(self.pending, v);
                    } else {
                        let out = vsli_v8::<USED>(self.pending, v);
                        store_v8(self.stage.add(WORD) as *mut u32, out);
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
                let pending = shr_v8::<15>(self.pending);
                #[cfg(not(all(target_feature = "avx512vbmi2", target_feature = "avx512vl")))]
                let pending = self.pending;
                store_v8(self.stage.add(LAST_WORD) as *mut u32, pending);
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
    fn xor_rotr8<const N: i32>(x: V8, y: V8) -> V8 {
        ror_v8::<N>(xor_v8(x, y))
    }

    unsafe fn run_octa_into_stages(
        cv_v: &[V8; 8],
        m: &[V8; 16],
        tlo: V8,
        thi: V8,
        blen: V8,
        flags: V8,
        ast: *mut V8,
        bs: *mut V8,
    ) {
        unsafe {
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

            for w in 0..8usize {
                store_v8(ast.add(w) as *mut u32, cv_v[w]);
            }
            let maxv = dup_u32(u32::MAX);
            for w in 32..36usize {
                store_v8(bs.add(w) as *mut u32, maxv);
            }
            let one = dup_u32(1);
            let chain: [V8; 20] = [
                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12],
                m[13], m[14], m[15], tlo, thi, blen, flags,
            ];
            store_v8(ast.add(16) as *mut u32, or_v8(one, shl_v8::<1>(chain[0])));
            for k in 1..20usize {
                let w = or_v8(shr_v8::<31>(chain[k - 1]), shl_v8::<1>(chain[k]));
                store_v8(ast.add(16 + k) as *mut u32, w);
            }

            let pending_bit = packer_initial_bit(shr_v8::<31>(flags));
            let mut wa = W32::at(ast, pending_bit);
            let mut wb = W32::at(bs, packer_initial_bit(one));

            macro_rules! g {
                ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
                 $mx:literal, $my:literal) => {{
                    let (t0, l0, r0) = add_carry_parts_v8(state[$la], state[$lb]);
                    pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, l0);
                    pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C0, 31, r0);
                    let (a1, l1, r1) = add_carry_parts_v8(t0, m[$mx]);
                    pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, l1);
                    pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C1, 31, r1);
                    let d1 = xor_rotr8::<16>(state[$ld], a1);
                    let (c1s, l2, r2) = add_carry_parts_v8(state[$lc], d1);
                    pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, l2);
                    pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C2, 31, r2);
                    let b1 = xor_rotr8::<12>(state[$lb], c1s);
                    let (t1, l3, r3) = add_carry_parts_v8(a1, b1);
                    pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, l3);
                    pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C3, 31, r3);
                    let (a2, l4, r4) = add_carry_parts_v8(t1, m[$my]);
                    pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, l4);
                    pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C4, 31, r4);
                    let d2 = xor_rotr8::<8>(d1, a2);
                    let (c2s, l5, r5) = add_carry_parts_v8(c1s, d2);
                    pushf8!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, l5);
                    pushf8!(wb, GS_BASE + G_STRIDE * $g + REC_C5, 31, r5);
                    let bn = xor_rotr8::<7>(b1, c2s);
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

            for w in 0..8usize {
                let lo = xor_v8(state[w], state[w + 8]);
                store_v8(ast.add(8 + w) as *mut u32, lo);
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

            let zero = dup_u32(0);
            for w in (LAST_WORD + 1)..496 {
                store_v8(ast.add(w) as *mut u32, zero);
                store_v8(bs.add(w) as *mut u32, zero);
            }
        }
    }

    #[inline(always)]
    unsafe fn tr8_at(stage: *const V8, w: usize) -> [V8; 8] {
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
    unsafe fn stream_pair(p: *mut u32, lo: V8, hi: V8) {
        unsafe {
            #[cfg(target_feature = "avx512f")]
            if (p as usize) & 63 == 0 {
                let z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
                _mm512_stream_si512(p.cast::<__m512i>(), z);
                return;
            }
            if (p as usize) & 31 == 0 {
                _mm256_stream_si256(p.cast::<__m256i>(), lo);
                _mm256_stream_si256(p.add(8).cast::<__m256i>(), hi);
            } else {
                _mm_stream_si128(p.cast::<__m128i>(), _mm256_castsi256_si128(lo));
                _mm_stream_si128(
                    p.add(4).cast::<__m128i>(),
                    _mm256_extracti128_si256::<1>(lo),
                );
                _mm_stream_si128(p.add(8).cast::<__m128i>(), _mm256_castsi256_si128(hi));
                _mm_stream_si128(
                    p.add(12).cast::<__m128i>(),
                    _mm256_extracti128_si256::<1>(hi),
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn drain_octa_nt(ast: *const V8, bs: *const V8, z: *mut u32, a: *mut u32, b: *mut u32) {
        unsafe {
            let maxv = dup_u32(u32::MAX);
            let zero = dup_u32(0);
            for g in (0..4usize).step_by(2) {
                let w = 8 * g;
                let ax = tr8_at(ast, w);
                let ay = tr8_at(ast, w + 8);
                for r in 0..8usize {
                    let off = r * U32_PER_BLOCK + w;
                    stream_pair(z.add(off), ax[r], ay[r]);
                    stream_pair(a.add(off), ax[r], ay[r]);
                    stream_pair(b.add(off), maxv, maxv);
                }
            }
            for g in (4..62usize).step_by(2) {
                let w = 8 * g;
                let ax = tr8_at(ast, w);
                let ay = tr8_at(ast, w + 8);
                let bx = tr8_at(bs, w);
                let by = tr8_at(bs, w + 8);
                for r in 0..8usize {
                    let off = r * U32_PER_BLOCK + w;
                    stream_pair(z.add(off), and_v8(ax[r], bx[r]), and_v8(ay[r], by[r]));
                    stream_pair(a.add(off), ax[r], ay[r]);
                    stream_pair(b.add(off), bx[r], by[r]);
                }
            }
            for g in (62..64usize).step_by(2) {
                let w = 8 * g;
                for r in 0..8usize {
                    let off = r * U32_PER_BLOCK + w;
                    stream_pair(z.add(off), zero, zero);
                    stream_pair(a.add(off), zero, zero);
                    stream_pair(b.add(off), zero, zero);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn emit_stripe_from_ab(ast: *const V8, bs: *const V8, stripe: *mut u8) {
        unsafe {
            let gather_bytes = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2,
                6, 10, 14, 3, 7, 11, 15,
            );
            let gather_dwords = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            let m1 = _mm256_set1_epi64x(0x00AA00AA00AA00AAu64 as i64);
            let m2 = _mm256_set1_epi64x(0x0000CCCC0000CCCCu64 as i64);
            let m3 = _mm256_set1_epi64x(0x00000000F0F0F0F0u64 as i64);
            let stripe_aligned16 = (stripe as usize) & 15 == 0;

            for i in 0..241usize {
                let w_lo = 2 * i;
                let w_hi = 2 * i + 1;
                let (zv_lo, zv_hi) = if w_lo < 36 {
                    (
                        load_v8(ast.add(w_lo) as *const u32),
                        load_v8(ast.add(w_hi) as *const u32),
                    )
                } else {
                    (
                        and_v8(
                            load_v8(ast.add(w_lo) as *const u32),
                            load_v8(bs.add(w_lo) as *const u32),
                        ),
                        and_v8(
                            load_v8(ast.add(w_hi) as *const u32),
                            load_v8(bs.add(w_hi) as *const u32),
                        ),
                    )
                };

                let mut y0 = _mm256_permutevar8x32_epi32(
                    _mm256_shuffle_epi8(zv_lo, gather_bytes),
                    gather_dwords,
                );
                let mut y1 = _mm256_permutevar8x32_epi32(
                    _mm256_shuffle_epi8(zv_hi, gather_bytes),
                    gather_dwords,
                );

                let t0 = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<7>(y0)), m1);
                let t1 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<7>(y1)), m1);
                #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
                {
                    y0 = _mm256_ternarylogic_epi64::<0x96>(y0, t0, _mm256_slli_epi64::<7>(t0));
                    y1 = _mm256_ternarylogic_epi64::<0x96>(y1, t1, _mm256_slli_epi64::<7>(t1));
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512vl")))]
                {
                    y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t0, _mm256_slli_epi64::<7>(t0)));
                    y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t1, _mm256_slli_epi64::<7>(t1)));
                }

                let t0 = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<14>(y0)), m2);
                let t1 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<14>(y1)), m2);
                #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
                {
                    y0 = _mm256_ternarylogic_epi64::<0x96>(y0, t0, _mm256_slli_epi64::<14>(t0));
                    y1 = _mm256_ternarylogic_epi64::<0x96>(y1, t1, _mm256_slli_epi64::<14>(t1));
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512vl")))]
                {
                    y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t0, _mm256_slli_epi64::<14>(t0)));
                    y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t1, _mm256_slli_epi64::<14>(t1)));
                }

                let t0 = _mm256_and_si256(_mm256_xor_si256(y0, _mm256_srli_epi64::<28>(y0)), m3);
                let t1 = _mm256_and_si256(_mm256_xor_si256(y1, _mm256_srli_epi64::<28>(y1)), m3);
                #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
                {
                    y0 = _mm256_ternarylogic_epi64::<0x96>(y0, t0, _mm256_slli_epi64::<28>(t0));
                    y1 = _mm256_ternarylogic_epi64::<0x96>(y1, t1, _mm256_slli_epi64::<28>(t1));
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "avx512vl")))]
                {
                    y0 = _mm256_xor_si256(y0, _mm256_xor_si256(t0, _mm256_slli_epi64::<28>(t0)));
                    y1 = _mm256_xor_si256(y1, _mm256_xor_si256(t1, _mm256_slli_epi64::<28>(t1)));
                }

                let dst = stripe.add(i * 64) as *mut u32;
                if stripe_aligned16 {
                    stream_pair(dst, y0, y1);
                } else {
                    store_v8(dst, y0);
                    store_v8(dst.add(8), y1);
                }
            }

            let zero = dup_u32(0);
            for i in 241..256usize {
                let dst = stripe.add(i * 64) as *mut u32;
                if stripe_aligned16 {
                    stream_pair(dst, zero, zero);
                } else {
                    store_v8(dst, zero);
                    store_v8(dst.add(8), zero);
                }
            }
            _mm_sfence();
        }
    }

    pub(crate) unsafe fn build_octa_witness_ab_and_lincheck_x86(
        inputs: OctaInputs<'_>,
        z: *mut u32,
        a: *mut u32,
        b: *mut u32,
        stripe: *mut u8,
    ) {
        unsafe {
            let mut ast = core::mem::MaybeUninit::<[V8; U32_PER_BLOCK]>::uninit();
            let mut bs = core::mem::MaybeUninit::<[V8; U32_PER_BLOCK]>::uninit();
            let ast = ast.as_mut_ptr().cast::<V8>();
            let bs = bs.as_mut_ptr().cast::<V8>();

            let prepared = match inputs {
                OctaInputs::Closed { init, base } => prepare_closed_inputs(init, base),
                OctaInputs::Blocks(blks) => {
                    let cv = tr8(
                        load_v8(blks[0].0.as_ptr()),
                        load_v8(blks[1].0.as_ptr()),
                        load_v8(blks[2].0.as_ptr()),
                        load_v8(blks[3].0.as_ptr()),
                        load_v8(blks[4].0.as_ptr()),
                        load_v8(blks[5].0.as_ptr()),
                        load_v8(blks[6].0.as_ptr()),
                        load_v8(blks[7].0.as_ptr()),
                    );
                    let m_lo = tr8(
                        load_v8(blks[0].1.as_ptr()),
                        load_v8(blks[1].1.as_ptr()),
                        load_v8(blks[2].1.as_ptr()),
                        load_v8(blks[3].1.as_ptr()),
                        load_v8(blks[4].1.as_ptr()),
                        load_v8(blks[5].1.as_ptr()),
                        load_v8(blks[6].1.as_ptr()),
                        load_v8(blks[7].1.as_ptr()),
                    );
                    let m_hi = tr8(
                        load_v8(blks[0].1.as_ptr().add(8)),
                        load_v8(blks[1].1.as_ptr().add(8)),
                        load_v8(blks[2].1.as_ptr().add(8)),
                        load_v8(blks[3].1.as_ptr().add(8)),
                        load_v8(blks[4].1.as_ptr().add(8)),
                        load_v8(blks[5].1.as_ptr().add(8)),
                        load_v8(blks[6].1.as_ptr().add(8)),
                        load_v8(blks[7].1.as_ptr().add(8)),
                    );
                    let message = [
                        m_lo[0], m_lo[1], m_lo[2], m_lo[3], m_lo[4], m_lo[5], m_lo[6], m_lo[7],
                        m_hi[0], m_hi[1], m_hi[2], m_hi[3], m_hi[4], m_hi[5], m_hi[6], m_hi[7],
                    ];
                    let mut tlo_a = [0u32; 8];
                    let mut thi_a = [0u32; 8];
                    let mut bl_a = [0u32; 8];
                    let mut fl_a = [0u32; 8];
                    for j in 0..8 {
                        tlo_a[j] = blks[j].2 as u32;
                        thi_a[j] = (blks[j].2 >> 32) as u32;
                        bl_a[j] = blks[j].3;
                        fl_a[j] = blks[j].4;
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
            };

            run_octa_into_stages(
                &prepared.cv,
                &prepared.message,
                prepared.counter_lo,
                prepared.counter_hi,
                prepared.block_len,
                prepared.flags,
                ast,
                bs,
            );
            drain_octa_nt(ast, bs, z, a, b);
            emit_stripe_from_ab(ast, bs, stripe);
        }
    }

    pub(crate) fn drive_from_init(
        init: u64,
        n_blocks: usize,
        n_blocks_log: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        use rayon::prelude::*;

        let n_total = 1usize << n_blocks_log;
        let total_f128 = n_total * F128_PER_BLOCK;
        let mut z = flock_core::scratch::take_f128(total_f128);
        let mut a = flock_core::scratch::take_f128(total_f128);
        let mut b = flock_core::scratch::take_f128(total_f128);
        let mut z_lincheck = flock_core::scratch::take_u8((n_total / 8) * K);

        let ilp = !crate::seed_pipe::gen_block_ilp_killed();
        let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

        z.par_chunks_mut(8 * F128_PER_BLOCK)
            .zip(a.par_chunks_mut(8 * F128_PER_BLOCK))
            .zip(b.par_chunks_mut(8 * F128_PER_BLOCK))
            .zip(z_lincheck.par_chunks_mut(K))
            .enumerate()
            .for_each(|(g, (((z_grp, a_grp), b_grp), stripe))| {
                let base = 8 * g;
                let zp = z_grp.as_mut_ptr() as *mut u32;
                let ap = a_grp.as_mut_ptr() as *mut u32;
                let bp = b_grp.as_mut_ptr() as *mut u32;
                let sp = stripe.as_mut_ptr();

                if base + 8 <= n_blocks {
                    unsafe {
                        build_octa_witness_ab_and_lincheck_x86(
                            OctaInputs::Closed { init, base },
                            zp,
                            ap,
                            bp,
                            sp,
                        );
                    }
                } else {
                    let owned: [Compression; 8] = std::array::from_fn(|k_in| {
                        let idx = base + k_in;
                        if idx < n_blocks {
                            crate::seed_pipe::gen_block_with(init, idx, ilp)
                        } else {
                            padding
                        }
                    });
                    let refs: [&Compression; 8] = std::array::from_fn(|k_in| &owned[k_in]);
                    unsafe {
                        build_octa_witness_ab_and_lincheck_x86(
                            OctaInputs::Blocks(refs),
                            zp,
                            ap,
                            bp,
                            sp,
                        );
                    }
                }
            });

        (z, a, b, z_lincheck)
    }

    pub(crate) fn drive_from_blocks(
        blocks: &[Compression],
        n_blocks_log: usize,
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        use rayon::prelude::*;

        let n_blocks = blocks.len();
        let n_total = 1usize << n_blocks_log;
        let total_f128 = n_total * F128_PER_BLOCK;
        let mut z = flock_core::scratch::take_f128(total_f128);
        let mut a = flock_core::scratch::take_f128(total_f128);
        let mut b = flock_core::scratch::take_f128(total_f128);
        let mut z_lincheck = flock_core::scratch::take_u8((n_total / 8) * K);
        let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

        z.par_chunks_mut(8 * F128_PER_BLOCK)
            .zip(a.par_chunks_mut(8 * F128_PER_BLOCK))
            .zip(b.par_chunks_mut(8 * F128_PER_BLOCK))
            .zip(z_lincheck.par_chunks_mut(K))
            .enumerate()
            .for_each(|(g, (((z_grp, a_grp), b_grp), stripe))| {
                let base = 8 * g;
                let refs: [&Compression; 8] = std::array::from_fn(|k_in| {
                    let idx = base + k_in;
                    if idx < n_blocks {
                        &blocks[idx]
                    } else {
                        &padding
                    }
                });
                let zp = z_grp.as_mut_ptr() as *mut u32;
                let ap = a_grp.as_mut_ptr() as *mut u32;
                let bp = b_grp.as_mut_ptr() as *mut u32;
                let sp = stripe.as_mut_ptr();

                unsafe {
                    build_octa_witness_ab_and_lincheck_x86(
                        OctaInputs::Blocks(refs),
                        zp,
                        ap,
                        bp,
                        sp,
                    );
                }
            });

        (z, a, b, z_lincheck)
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_neon {
    use super::*;
    use core::arch::aarch64::*;

    type V4 = uint32x4_t;

    #[inline(always)]
    fn tr4(w0: V4, w1: V4, w2: V4, w3: V4) -> (V4, V4, V4, V4) {
        unsafe {
            let t0 = vtrn1q_u32(w0, w1);
            let t1 = vtrn2q_u32(w0, w1);
            let t2 = vtrn1q_u32(w2, w3);
            let t3 = vtrn2q_u32(w2, w3);
            (
                vreinterpretq_u32_u64(vtrn1q_u64(
                    vreinterpretq_u64_u32(t0),
                    vreinterpretq_u64_u32(t2),
                )),
                vreinterpretq_u32_u64(vtrn1q_u64(
                    vreinterpretq_u64_u32(t1),
                    vreinterpretq_u64_u32(t3),
                )),
                vreinterpretq_u32_u64(vtrn2q_u64(
                    vreinterpretq_u64_u32(t0),
                    vreinterpretq_u64_u32(t2),
                )),
                vreinterpretq_u32_u64(vtrn2q_u64(
                    vreinterpretq_u64_u32(t1),
                    vreinterpretq_u64_u32(t3),
                )),
            )
        }
    }

    #[inline(always)]
    unsafe fn store_nt_pair(x: V4, y: V4, p: *mut u32) {
        unsafe {
            core::arch::asm!(
                "stnp {0:q}, {1:q}, [{2}]",
                in(vreg) x,
                in(vreg) y,
                in(reg) p,
                options(nostack)
            );
        }
    }

    struct W32 {
        pending: V4,
        stage: *mut V4,
    }

    impl W32 {
        #[inline(always)]
        fn at(stage: *mut V4, pending: V4) -> Self {
            Self { pending, stage }
        }

        #[inline(always)]
        unsafe fn push<const USED: i32, const WIDTH: i32, const BACK: i32, const WORD: usize>(
            &mut self,
            v: V4,
        ) {
            const {
                assert!(USED >= 0 && USED < 32);
                assert!(WIDTH == 31 || WIDTH == 32);
                assert!(BACK >= 1 && BACK < 32);
                assert!(WORD < U32_PER_BLOCK);
            }
            debug_assert!(USED + WIDTH <= 32 || BACK == 32 - USED);
            unsafe {
                if USED == 0 {
                    if WIDTH == 32 {
                        vst1q_u32(self.stage.add(WORD) as *mut u32, v);
                        self.pending = vdupq_n_u32(0);
                    } else {
                        self.pending = v;
                    }
                } else if USED + WIDTH < 32 {
                    self.pending = vsliq_n_u32::<USED>(self.pending, v);
                } else {
                    let out = vsliq_n_u32::<USED>(self.pending, v);
                    vst1q_u32(self.stage.add(WORD) as *mut u32, out);
                    if USED + WIDTH == 32 {
                        self.pending = vdupq_n_u32(0);
                    } else {
                        self.pending = vshrq_n_u32::<BACK>(v);
                    }
                }
            }
        }

        #[inline(always)]
        unsafe fn finish(&mut self) {
            unsafe {
                vst1q_u32(self.stage.add(LAST_WORD) as *mut u32, self.pending);
            }
        }
    }

    macro_rules! pushf4 {
        ($w:ident, $pos:expr, $width:literal, $v:expr) => {{
            $w.push::<{ ($pos % 32) as i32 }, $width, {
                let u = ($pos % 32) as i32;
                if u == 0 { 1 } else { 32 - u }
            }, { $pos / 32 }>($v);
        }};
    }

    #[inline(always)]
    fn add_carry_parts_v4(x: V4, y: V4) -> (V4, V4, V4) {
        unsafe {
            let sum = vaddq_u32(x, y);
            let left = veorq_u32(sum, y);
            let right = veorq_u32(sum, x);
            (sum, left, right)
        }
    }

    #[inline(always)]
    fn xor_rotr4<const N: i32, const M: i32>(x: V4, y: V4) -> V4 {
        debug_assert_eq!(N + M, 32);
        unsafe {
            let v = veorq_u32(x, y);
            vorrq_u32(vshrq_n_u32::<N>(v), vshlq_n_u32::<M>(v))
        }
    }

    unsafe fn run_quad_into_stages(
        cv_v: &[V4; 8],
        m: &[V4; 16],
        tlo: V4,
        thi: V4,
        blen: V4,
        flags: V4,
        ast: *mut V4,
        bs: *mut V4,
    ) {
        unsafe {
            let mut state: [V4; 16] = [
                cv_v[0],
                cv_v[1],
                cv_v[2],
                cv_v[3],
                cv_v[4],
                cv_v[5],
                cv_v[6],
                cv_v[7],
                vdupq_n_u32(BLAKE3_IV[0]),
                vdupq_n_u32(BLAKE3_IV[1]),
                vdupq_n_u32(BLAKE3_IV[2]),
                vdupq_n_u32(BLAKE3_IV[3]),
                tlo,
                thi,
                blen,
                flags,
            ];

            for w in 0..8usize {
                vst1q_u32(ast.add(w) as *mut u32, cv_v[w]);
            }
            let maxv = vdupq_n_u32(u32::MAX);
            for w in 32..36usize {
                vst1q_u32(bs.add(w) as *mut u32, maxv);
            }
            let one = vdupq_n_u32(1);
            let chain: [V4; 20] = [
                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12],
                m[13], m[14], m[15], tlo, thi, blen, flags,
            ];
            vst1q_u32(
                ast.add(16) as *mut u32,
                vorrq_u32(one, vshlq_n_u32::<1>(chain[0])),
            );
            for k in 1..20usize {
                let w = vorrq_u32(vshrq_n_u32::<31>(chain[k - 1]), vshlq_n_u32::<1>(chain[k]));
                vst1q_u32(ast.add(16 + k) as *mut u32, w);
            }

            let pending_bit = vshrq_n_u32::<31>(flags);
            let mut wa = W32::at(ast, pending_bit);
            let mut wb = W32::at(bs, one);

            macro_rules! g {
                ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
                 $mx:literal, $my:literal) => {{
                    let (t0, l0, r0) = add_carry_parts_v4(state[$la], state[$lb]);
                    pushf4!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, l0);
                    pushf4!(wb, GS_BASE + G_STRIDE * $g + REC_C0, 31, r0);
                    let (a1, l1, r1) = add_carry_parts_v4(t0, m[$mx]);
                    pushf4!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, l1);
                    pushf4!(wb, GS_BASE + G_STRIDE * $g + REC_C1, 31, r1);
                    let d1 = xor_rotr4::<16, 16>(state[$ld], a1);
                    let (c1s, l2, r2) = add_carry_parts_v4(state[$lc], d1);
                    pushf4!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, l2);
                    pushf4!(wb, GS_BASE + G_STRIDE * $g + REC_C2, 31, r2);
                    let b1 = xor_rotr4::<12, 20>(state[$lb], c1s);
                    let (t1, l3, r3) = add_carry_parts_v4(a1, b1);
                    pushf4!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, l3);
                    pushf4!(wb, GS_BASE + G_STRIDE * $g + REC_C3, 31, r3);
                    let (a2, l4, r4) = add_carry_parts_v4(t1, m[$my]);
                    pushf4!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, l4);
                    pushf4!(wb, GS_BASE + G_STRIDE * $g + REC_C4, 31, r4);
                    let d2 = xor_rotr4::<8, 24>(d1, a2);
                    let (c2s, l5, r5) = add_carry_parts_v4(c1s, d2);
                    pushf4!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, l5);
                    pushf4!(wb, GS_BASE + G_STRIDE * $g + REC_C5, 31, r5);
                    let bn = xor_rotr4::<7, 25>(b1, c2s);
                    pushf4!(wa, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                    pushf4!(wb, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, maxv);
                    pushf4!(wa, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
                    pushf4!(wb, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, maxv);
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

            for w in 0..8usize {
                let lo = veorq_u32(state[w], state[w + 8]);
                vst1q_u32(ast.add(8 + w) as *mut u32, lo);
            }
            macro_rules! oh {
                ($w:literal) => {{
                    let hv = veorq_u32(state[$w + 8], cv_v[$w]);
                    pushf4!(wa, OUT_HI_BASE + 32 * $w, 32, hv);
                    pushf4!(wb, OUT_HI_BASE + 32 * $w, 32, maxv);
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

            let zero = vdupq_n_u32(0);
            for w in (LAST_WORD + 1)..488 {
                vst1q_u32(ast.add(w) as *mut u32, zero);
                vst1q_u32(bs.add(w) as *mut u32, zero);
            }
        }
    }

    #[inline(always)]
    unsafe fn drain_quad_nt(ast: *const V4, bs: *const V4, z: *mut u32, a: *mut u32, b: *mut u32) {
        unsafe {
            let maxv = vdupq_n_u32(u32::MAX);
            let zero = vdupq_n_u32(0);
            for g in 0..4usize {
                let w = 8 * g;
                let ax = vld4q_u32(ast.add(w) as *const u32);
                let ay = vld4q_u32(ast.add(w + 4) as *const u32);
                for r in 0..4usize {
                    let off = r * U32_PER_BLOCK + w;
                    let (axr, ayr) = match r {
                        0 => (ax.0, ay.0),
                        1 => (ax.1, ay.1),
                        2 => (ax.2, ay.2),
                        _ => (ax.3, ay.3),
                    };
                    store_nt_pair(axr, ayr, z.add(off));
                    store_nt_pair(axr, ayr, a.add(off));
                    store_nt_pair(maxv, maxv, b.add(off));
                }
            }
            for g in 4..61usize {
                let w = 8 * g;
                let ax = vld4q_u32(ast.add(w) as *const u32);
                let ay = vld4q_u32(ast.add(w + 4) as *const u32);
                let bx = vld4q_u32(bs.add(w) as *const u32);
                let by = vld4q_u32(bs.add(w + 4) as *const u32);
                for r in 0..4usize {
                    let off = r * U32_PER_BLOCK + w;
                    let (axr, ayr, bxr, byr) = match r {
                        0 => (ax.0, ay.0, bx.0, by.0),
                        1 => (ax.1, ay.1, bx.1, by.1),
                        2 => (ax.2, ay.2, bx.2, by.2),
                        _ => (ax.3, ay.3, bx.3, by.3),
                    };
                    store_nt_pair(vandq_u32(axr, bxr), vandq_u32(ayr, byr), z.add(off));
                    store_nt_pair(axr, ayr, a.add(off));
                    store_nt_pair(bxr, byr, b.add(off));
                }
            }
            for g in 61..64usize {
                let w = 8 * g;
                for r in 0..4usize {
                    let off = r * U32_PER_BLOCK + w;
                    store_nt_pair(zero, zero, z.add(off));
                    store_nt_pair(zero, zero, a.add(off));
                    store_nt_pair(zero, zero, b.add(off));
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn bit_transpose_2x64_neon(
        mut y_lo: uint64x2_t,
        mut y_hi: uint64x2_t,
        m1: uint64x2_t,
        m2: uint64x2_t,
        m3: uint64x2_t,
    ) -> (uint64x2_t, uint64x2_t) {
        unsafe {
            let t0 = vandq_u64(veorq_u64(y_lo, vshrq_n_u64::<7>(y_lo)), m1);
            let t1 = vandq_u64(veorq_u64(y_hi, vshrq_n_u64::<7>(y_hi)), m1);
            y_lo = veorq_u64(y_lo, veorq_u64(t0, vshlq_n_u64::<7>(t0)));
            y_hi = veorq_u64(y_hi, veorq_u64(t1, vshlq_n_u64::<7>(t1)));

            let t0 = vandq_u64(veorq_u64(y_lo, vshrq_n_u64::<14>(y_lo)), m2);
            let t1 = vandq_u64(veorq_u64(y_hi, vshrq_n_u64::<14>(y_hi)), m2);
            y_lo = veorq_u64(y_lo, veorq_u64(t0, vshlq_n_u64::<14>(t0)));
            y_hi = veorq_u64(y_hi, veorq_u64(t1, vshlq_n_u64::<14>(t1)));

            let t0 = vandq_u64(veorq_u64(y_lo, vshrq_n_u64::<28>(y_lo)), m3);
            let t1 = vandq_u64(veorq_u64(y_hi, vshrq_n_u64::<28>(y_hi)), m3);
            y_lo = veorq_u64(y_lo, veorq_u64(t0, vshlq_n_u64::<28>(t0)));
            y_hi = veorq_u64(y_hi, veorq_u64(t1, vshlq_n_u64::<28>(t1)));

            (y_lo, y_hi)
        }
    }

    pub(crate) unsafe fn build_octa_witness_ab_and_lincheck_neon(
        inputs: OctaInputs<'_>,
        z: *mut u32,
        a: *mut u32,
        b: *mut u32,
        stripe: *mut u8,
    ) {
        unsafe {
            let mut ast0 = core::mem::MaybeUninit::<[V4; U32_PER_BLOCK]>::uninit();
            let mut bs0 = core::mem::MaybeUninit::<[V4; U32_PER_BLOCK]>::uninit();
            let mut ast1 = core::mem::MaybeUninit::<[V4; U32_PER_BLOCK]>::uninit();
            let mut bs1 = core::mem::MaybeUninit::<[V4; U32_PER_BLOCK]>::uninit();
            let ast0 = ast0.as_mut_ptr().cast::<V4>();
            let bs0 = bs0.as_mut_ptr().cast::<V4>();
            let ast1 = ast1.as_mut_ptr().cast::<V4>();
            let bs1 = bs1.as_mut_ptr().cast::<V4>();

            match inputs {
                OctaInputs::Closed { init, base } => {
                    for (q, (ast_q, bs_q)) in [(ast0, bs0), (ast1, bs1)].into_iter().enumerate() {
                        let soa = crate::seed_pipe::gen_quad_soa(init, base + 4 * q);
                        let cv_v: [V4; 8] = std::array::from_fn(|w| vld1q_u32(soa.cv[w].as_ptr()));
                        let m: [V4; 16] =
                            std::array::from_fn(|w| vld1q_u32(soa.message[w].as_ptr()));
                        run_quad_into_stages(
                            &cv_v,
                            &m,
                            vld1q_u32(soa.counter_lo.as_ptr()),
                            vld1q_u32(soa.counter_hi.as_ptr()),
                            vld1q_u32(soa.block_len.as_ptr()),
                            vld1q_u32(soa.flags.as_ptr()),
                            ast_q,
                            bs_q,
                        );
                    }
                }
                OctaInputs::Blocks(blks) => {
                    for (q, (ast_q, bs_q)) in [(ast0, bs0), (ast1, bs1)].into_iter().enumerate() {
                        let sub = &blks[4 * q..4 * q + 4];
                        let ptrs = [
                            sub[0].0.as_ptr(),
                            sub[1].0.as_ptr(),
                            sub[2].0.as_ptr(),
                            sub[3].0.as_ptr(),
                        ];
                        let (cv0, cv1, cv2, cv3) = tr4(
                            vld1q_u32(ptrs[0]),
                            vld1q_u32(ptrs[1]),
                            vld1q_u32(ptrs[2]),
                            vld1q_u32(ptrs[3]),
                        );
                        let (cv4, cv5, cv6, cv7) = tr4(
                            vld1q_u32(ptrs[0].add(4)),
                            vld1q_u32(ptrs[1].add(4)),
                            vld1q_u32(ptrs[2].add(4)),
                            vld1q_u32(ptrs[3].add(4)),
                        );
                        let cv_v = [cv0, cv1, cv2, cv3, cv4, cv5, cv6, cv7];
                        let mptrs = [
                            sub[0].1.as_ptr(),
                            sub[1].1.as_ptr(),
                            sub[2].1.as_ptr(),
                            sub[3].1.as_ptr(),
                        ];
                        let mut m: [V4; 16] = [cv0; 16];
                        for wgrp in 0..4 {
                            let (m0, m1, m2, m3) = tr4(
                                vld1q_u32(mptrs[0].add(4 * wgrp)),
                                vld1q_u32(mptrs[1].add(4 * wgrp)),
                                vld1q_u32(mptrs[2].add(4 * wgrp)),
                                vld1q_u32(mptrs[3].add(4 * wgrp)),
                            );
                            m[4 * wgrp] = m0;
                            m[4 * wgrp + 1] = m1;
                            m[4 * wgrp + 2] = m2;
                            m[4 * wgrp + 3] = m3;
                        }
                        let mut tlo_a = [0u32; 4];
                        let mut thi_a = [0u32; 4];
                        let mut bl_a = [0u32; 4];
                        let mut fl_a = [0u32; 4];
                        for j in 0..4 {
                            tlo_a[j] = sub[j].2 as u32;
                            thi_a[j] = (sub[j].2 >> 32) as u32;
                            bl_a[j] = sub[j].3;
                            fl_a[j] = sub[j].4;
                        }
                        run_quad_into_stages(
                            &cv_v,
                            &m,
                            vld1q_u32(tlo_a.as_ptr()),
                            vld1q_u32(thi_a.as_ptr()),
                            vld1q_u32(bl_a.as_ptr()),
                            vld1q_u32(fl_a.as_ptr()),
                            ast_q,
                            bs_q,
                        );
                    }
                }
            }

            drain_quad_nt(ast0, bs0, z, a, b);
            drain_quad_nt(
                ast1,
                bs1,
                z.add(4 * U32_PER_BLOCK),
                a.add(4 * U32_PER_BLOCK),
                b.add(4 * U32_PER_BLOCK),
            );

            let m1 = vdupq_n_u64(0x00AA00AA00AA00AAu64);
            let m2 = vdupq_n_u64(0x0000CCCC0000CCCCu64);
            let m3 = vdupq_n_u64(0x00000000F0F0F0F0u64);
            let stripe_aligned = (stripe as usize) & 15 == 0;

            for i in 0..241usize {
                let w_lo = 2 * i;
                let w_hi = 2 * i + 1;
                let (zv0_lo, zv0_hi, zv1_lo, zv1_hi) = if w_lo < 32 {
                    (
                        vld1q_u32(ast0.add(w_lo) as *const u32),
                        vld1q_u32(ast0.add(w_hi) as *const u32),
                        vld1q_u32(ast1.add(w_lo) as *const u32),
                        vld1q_u32(ast1.add(w_hi) as *const u32),
                    )
                } else {
                    (
                        vandq_u32(
                            vld1q_u32(ast0.add(w_lo) as *const u32),
                            vld1q_u32(bs0.add(w_lo) as *const u32),
                        ),
                        vandq_u32(
                            vld1q_u32(ast0.add(w_hi) as *const u32),
                            vld1q_u32(bs0.add(w_hi) as *const u32),
                        ),
                        vandq_u32(
                            vld1q_u32(ast1.add(w_lo) as *const u32),
                            vld1q_u32(bs1.add(w_lo) as *const u32),
                        ),
                        vandq_u32(
                            vld1q_u32(ast1.add(w_hi) as *const u32),
                            vld1q_u32(bs1.add(w_hi) as *const u32),
                        ),
                    )
                };

                let a_lo = vreinterpretq_u8_u32(zv0_lo);
                let b_lo = vreinterpretq_u8_u32(zv1_lo);
                let u0_lo = vuzp1q_u8(a_lo, b_lo);
                let u1_lo = vuzp2q_u8(a_lo, b_lo);
                let y0_lo = vreinterpretq_u64_u8(vuzp1q_u8(u0_lo, u1_lo));
                let y0_hi = vreinterpretq_u64_u8(vuzp2q_u8(u0_lo, u1_lo));
                let (y0_lo, y0_hi) = bit_transpose_2x64_neon(y0_lo, y0_hi, m1, m2, m3);

                let a_hi = vreinterpretq_u8_u32(zv0_hi);
                let b_hi = vreinterpretq_u8_u32(zv1_hi);
                let u0_hi = vuzp1q_u8(a_hi, b_hi);
                let u1_hi = vuzp2q_u8(a_hi, b_hi);
                let y1_lo = vreinterpretq_u64_u8(vuzp1q_u8(u0_hi, u1_hi));
                let y1_hi = vreinterpretq_u64_u8(vuzp2q_u8(u0_hi, u1_hi));
                let (y1_lo, y1_hi) = bit_transpose_2x64_neon(y1_lo, y1_hi, m1, m2, m3);

                let dst = stripe.add(i * 64) as *mut u32;
                if stripe_aligned {
                    store_nt_pair(
                        vreinterpretq_u32_u64(y0_lo),
                        vreinterpretq_u32_u64(y0_hi),
                        dst,
                    );
                    store_nt_pair(
                        vreinterpretq_u32_u64(y1_lo),
                        vreinterpretq_u32_u64(y1_hi),
                        dst.add(8),
                    );
                } else {
                    vst1q_u64(dst as *mut u64, y0_lo);
                    vst1q_u64(dst.add(4) as *mut u64, y0_hi);
                    vst1q_u64(dst.add(8) as *mut u64, y1_lo);
                    vst1q_u64(dst.add(12) as *mut u64, y1_hi);
                }
            }

            let zero = vdupq_n_u32(0);
            for i in 241..256usize {
                let dst = stripe.add(i * 64) as *mut u32;
                if stripe_aligned {
                    store_nt_pair(zero, zero, dst);
                    store_nt_pair(zero, zero, dst.add(8));
                } else {
                    vst1q_u32(dst, zero);
                    vst1q_u32(dst.add(4), zero);
                    vst1q_u32(dst.add(8), zero);
                    vst1q_u32(dst.add(12), zero);
                }
            }
        }
    }
}

pub(crate) fn drive_blake3_witness_from_seed(
    log2_size: u32,
    seed: u64,
    n_blocks: usize,
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    drive_blake3_witness_from_init(
        crate::seed_pipe::generator_init(log2_size, seed),
        n_blocks,
        n_blocks_log,
    )
}

pub(crate) fn drive_blake3_witness_from_init(
    init: u64,
    n_blocks: usize,
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        return x86_fast::drive_from_init(init, n_blocks, n_blocks_log);
    }

    #[cfg(target_arch = "aarch64")]
    {
        use rayon::prelude::*;

        let n_total = 1usize << n_blocks_log;
        let total_f128 = n_total * F128_PER_BLOCK;
        let mut z = flock_core::scratch::take_f128(total_f128);
        let mut a = flock_core::scratch::take_f128(total_f128);
        let mut b = flock_core::scratch::take_f128(total_f128);
        let mut z_lincheck = flock_core::scratch::take_u8((n_total / 8) * K);

        let ilp = !crate::seed_pipe::gen_block_ilp_killed();
        let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

        z.par_chunks_mut(8 * F128_PER_BLOCK)
            .zip(a.par_chunks_mut(8 * F128_PER_BLOCK))
            .zip(b.par_chunks_mut(8 * F128_PER_BLOCK))
            .zip(z_lincheck.par_chunks_mut(K))
            .enumerate()
            .for_each(|(g, (((z_grp, a_grp), b_grp), stripe))| {
                let base = 8 * g;
                let zp = z_grp.as_mut_ptr() as *mut u32;
                let ap = a_grp.as_mut_ptr() as *mut u32;
                let bp = b_grp.as_mut_ptr() as *mut u32;
                let sp = stripe.as_mut_ptr();

                if base + 8 <= n_blocks {
                    unsafe {
                        aarch64_neon::build_octa_witness_ab_and_lincheck_neon(
                            OctaInputs::Closed { init, base },
                            zp,
                            ap,
                            bp,
                            sp,
                        );
                    }
                } else {
                    let owned: [Compression; 8] = std::array::from_fn(|k_in| {
                        let idx = base + k_in;
                        if idx < n_blocks {
                            crate::seed_pipe::gen_block_with(init, idx, ilp)
                        } else {
                            padding
                        }
                    });
                    let refs: [&Compression; 8] = std::array::from_fn(|k_in| &owned[k_in]);
                    unsafe {
                        aarch64_neon::build_octa_witness_ab_and_lincheck_neon(
                            OctaInputs::Blocks(refs),
                            zp,
                            ap,
                            bp,
                            sp,
                        );
                    }
                }
            });

        return (z, a, b, z_lincheck);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let ilp = !crate::seed_pipe::gen_block_ilp_killed();
        let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);
        super::super::common::drive_witness_packed_and_lincheck_overwrite_by_index(
            n_blocks,
            n_blocks_log,
            super::K_LOG,
            USEFUL_BITS,
            |idx, z_u64, a_u64, b_u64| {
                let (cv, m, t, bl, fl) = crate::seed_pipe::gen_block_with(init, idx, ilp);
                super::build_block_witness_ab_packed_into(&cv, &m, t, bl, fl, z_u64, a_u64, b_u64);
            },
            Some(|z_u64: &mut [u64], a_u64: &mut [u64], b_u64: &mut [u64]| {
                let (cv, m, t, bl, fl) = &padding;
                super::build_block_witness_ab_packed_into(cv, m, *t, *bl, *fl, z_u64, a_u64, b_u64);
            }),
        )
    }
}

pub(crate) fn drive_blake3_witness_from_blocks(
    blocks: &[Compression],
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        return x86_fast::drive_from_blocks(blocks, n_blocks_log);
    }

    #[cfg(target_arch = "aarch64")]
    {
        use rayon::prelude::*;

        let n_blocks = blocks.len();
        let n_total = 1usize << n_blocks_log;
        let total_f128 = n_total * F128_PER_BLOCK;
        let mut z = flock_core::scratch::take_f128(total_f128);
        let mut a = flock_core::scratch::take_f128(total_f128);
        let mut b = flock_core::scratch::take_f128(total_f128);
        let mut z_lincheck = flock_core::scratch::take_u8((n_total / 8) * K);

        let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

        z.par_chunks_mut(8 * F128_PER_BLOCK)
            .zip(a.par_chunks_mut(8 * F128_PER_BLOCK))
            .zip(b.par_chunks_mut(8 * F128_PER_BLOCK))
            .zip(z_lincheck.par_chunks_mut(K))
            .enumerate()
            .for_each(|(g, (((z_grp, a_grp), b_grp), stripe))| {
                let base = 8 * g;
                let refs: [&Compression; 8] = std::array::from_fn(|k_in| {
                    let idx = base + k_in;
                    if idx < n_blocks {
                        &blocks[idx]
                    } else {
                        &padding
                    }
                });
                let zp = z_grp.as_mut_ptr() as *mut u32;
                let ap = a_grp.as_mut_ptr() as *mut u32;
                let bp = b_grp.as_mut_ptr() as *mut u32;
                let sp = stripe.as_mut_ptr();

                unsafe {
                    aarch64_neon::build_octa_witness_ab_and_lincheck_neon(
                        OctaInputs::Blocks(refs),
                        zp,
                        ap,
                        bp,
                        sp,
                    );
                }
            });

        return (z, a, b, z_lincheck);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);
        super::super::common::drive_witness_packed_and_lincheck_overwrite(
            blocks,
            Some(&padding),
            n_blocks_log,
            super::K_LOG,
            USEFUL_BITS,
            |block: &Compression, z_u64, a_u64, b_u64| {
                let (cv, m, t, bl, fl) = block;
                super::build_block_witness_ab_packed_into(cv, m, *t, *bl, *fl, z_u64, a_u64, b_u64);
            },
        )
    }
}
